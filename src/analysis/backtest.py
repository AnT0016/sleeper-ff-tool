"""Full-season "what if I'd used this tool" backtest over a *completed* Sleeper season.

Because the season is over, we have ground truth: ``GET /league/<id>/matchups/<week>`` returns each
roster's real ``players_points`` (already in our exact league scoring) and the lineup that was
actually ``starters``. So we never trust past-week *projections* for scoring — we score every lineup
by the points players really put up. Four retrospective views (all read-only):

* **Weekly** — your actual lineup & score vs. the **hindsight-optimal** lineup from your roster that
  week (best legal lineup by actual points → *points left on the bench*), plus the
  **projection-recommended** lineup (built from that week's Sleeper projections) *scored by actual
  points* — i.e. what blindly following the tool would really have netted. Each week carries the real
  result vs. your opponent and whether optimal / the tool's lineup would have flipped it.
* **Draft** — replay the real snake draft: at each of your actual pick numbers the tool takes the
  best-available by **VOR** (Phase 2 board) from the genuinely-remaining pool, with positional caps.
  Pick-by-pick you-vs-tool, each player graded by full-season actual points.
* **Season summary** — actual vs. optimal vs. tool totals & records, total bench points lost, and a
  recomputed "always-optimal" standings (flips your head-to-heads the optimal lineup would have won).
* **League-wide rank** — your actual (and optimal) weekly score ranked among all 12 teams.

The pure helpers (:func:`simulate_draft`, :func:`lineup_from_points`, :func:`optimal_standings`) are
unit-tested offline; :func:`build_backtest` is the networked orchestrator the CLI runs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pandas as pd

from analysis.snapshot import team_names_by_roster, write_sqlite
from draft import roster as draft_roster
from draft.vor import add_vor, replacement_levels
from optimizer.inputs import STARTABLE_POSITIONS, bye_teams, find_my_roster, score_projections
from optimizer.lineup import LineupPlayer, lineup_slots, optimize
from projections.board import build_board
from scoring.engine import points
from sleeper import client
from waivers.league import standings

_LOG = logging.getLogger(__name__)

#: Positional caps for the tool's draft policy (RB/WR effectively uncapped at `rounds`). K/DEF capped
#: at 1 each; the "don't draft K/DEF early" rule already falls out of VOR, this just prevents hoarding.
DRAFT_CAPS: dict[str, int] = {"QB": 2, "TE": 2, "K": 1, "DEF": 1}

#: Default fantasy-playoff start if the league object doesn't say (CLAUDE.md: Weeks 15-17).
DEFAULT_PLAYOFF_START: int = 15


# --------------------------------------------------------------------------- pure helpers
def _pos_of(pid: str, meta: Mapping) -> str | None:
    """Resolve a player's position; team-abbreviation ids (DEF) carry no metadata."""
    pos = meta.get("position")
    if pos:
        return pos
    return "DEF" if pid.isalpha() else None


def lineup_from_points(
    roster_ids: Sequence[str],
    points_map: Mapping[str, float],
    players_map: Mapping[str, Mapping],
    slots: Mapping[str, int],
):
    """Best legal lineup from ``roster_ids`` using ``points_map`` as each player's value.

    Used for the hindsight-optimal lineup (value = actual points scored). All players are treated as
    startable — a player who didn't play simply scored ~0 and won't be chosen. Returns the
    :class:`~optimizer.lineup.LineupSolution`.
    """
    pool: list[LineupPlayer] = []
    for pid in roster_ids:
        pid = str(pid)
        meta = players_map.get(pid) or {}
        pos = _pos_of(pid, meta)
        if pos not in STARTABLE_POSITIONS:
            continue
        team = pid if pos == "DEF" else meta.get("team")
        pool.append(
            LineupPlayer(
                player_id=pid,
                name=meta.get("full_name") or pid,
                pos=pos,
                team=team,
                proj_pts=round(float(points_map.get(pid, 0.0)), 2),
            )
        )
    return optimize(pool, slots)


def simulate_draft(
    picks: Sequence[Mapping],
    my_user_id: str,
    vor_by_id: Mapping[str, float],
    pos_by_id: Mapping[str, str],
    *,
    caps: Mapping[str, int] = DRAFT_CAPS,
    default_cap: int = 999,
) -> list[dict]:
    """Replay the snake draft, substituting a VOR-greedy policy for *my* picks only.

    Other teams' picks are held fixed (they remove players from the pool as they really did); at each
    of my pick numbers the tool takes the highest-VOR available player whose position is below its cap.
    Returns ``[{pick_no, round, my_pid, tool_pid}, ...]`` for my picks, in order.
    """
    taken: set[str] = set()
    pos_count: dict[str, int] = defaultdict(int)
    board_ids = list(vor_by_id)
    rows: list[dict] = []
    for pk in sorted(picks, key=lambda p: int(p["pick_no"])):
        pid = str(pk.get("player_id"))
        if str(pk.get("picked_by")) != str(my_user_id):
            taken.add(pid)  # another team drafts the player they really drafted
            continue
        best_id, best_vor = None, float("-inf")
        for cand in board_ids:
            if cand in taken:
                continue
            cpos = pos_by_id.get(cand)
            if pos_count[cpos] >= caps.get(cpos, default_cap):
                continue
            v = vor_by_id[cand]
            if v > best_vor:
                best_id, best_vor = cand, v
        if best_id is not None:
            taken.add(best_id)
            pos_count[pos_by_id.get(best_id)] += 1
        rows.append(
            {"pick_no": int(pk["pick_no"]), "round": int(pk["round"]), "my_pid": pid, "tool_pid": best_id}
        )
    return rows


def optimal_standings(
    weekly_matchups: Mapping[int, Sequence[Mapping]],
    my_roster_id: int,
    my_optimal_by_week: Mapping[int, float],
) -> list[int]:
    """Recompute the standings (roster_ids, best first) if *I* had started optimally every week.

    Other teams' weekly scores are held at their actuals; only my side of each head-to-head is swapped
    to my optimal total, which can flip the winner. Ranked by wins then total points.
    """
    wins: dict[int, float] = defaultdict(float)
    pts: dict[int, float] = defaultdict(float)
    for week, rows in weekly_matchups.items():
        by_mid: dict[object, list[Mapping]] = defaultdict(list)
        for r in rows:
            by_mid[r.get("matchup_id")].append(r)
        for pair in by_mid.values():
            if len(pair) != 2:
                continue
            a, b = pair
            ra, rb = int(a["roster_id"]), int(b["roster_id"])
            pa, pb = float(a.get("points") or 0.0), float(b.get("points") or 0.0)
            if ra == my_roster_id:
                pa = my_optimal_by_week.get(week, pa)
            if rb == my_roster_id:
                pb = my_optimal_by_week.get(week, pb)
            pts[ra] += pa
            pts[rb] += pb
            if pa > pb:
                wins[ra] += 1
            elif pb > pa:
                wins[rb] += 1
            else:
                wins[ra] += 0.5
                wins[rb] += 0.5
    rids = set(wins) | set(pts)
    return sorted(rids, key=lambda rid: (wins[rid], pts[rid]), reverse=True)


def _result(a: float, b: float | None) -> str:
    if b is None:
        return "—"
    return "W" if a > b else ("T" if a == b else "L")


def _record(rows: Sequence[Mapping], key: str) -> str:
    w = sum(1 for r in rows if r[key] == "W")
    loss = sum(1 for r in rows if r[key] == "L")
    t = sum(1 for r in rows if r[key] == "T")
    return f"{w}-{loss}" + (f"-{t}" if t else "")


def _top_miss(actual_ids, optimal_starters, pts_map, players_map) -> str:
    """The single most costly bench decision: best optimal starter you sat, vs the starter it'd swap."""
    actual = set(actual_ids)
    optimal_ids = {sp.player.player_id for sp in optimal_starters}
    gained = [(sp.player.player_id, pts_map.get(sp.player.player_id, 0.0))
              for sp in optimal_starters if sp.player.player_id not in actual]
    if not gained:
        return ""
    bid, bpts = max(gained, key=lambda x: x[1])

    def nm(pid):
        return (players_map.get(pid) or {}).get("full_name") or pid

    benched_out = [(pid, pts_map.get(pid, 0.0)) for pid in actual_ids if pid not in optimal_ids]
    if benched_out:
        wid, wpts = min(benched_out, key=lambda x: x[1])
        return f"{nm(bid)} {bpts:.1f} over {nm(wid)} {wpts:.1f}"
    return f"{nm(bid)} {bpts:.1f}"


# --------------------------------------------------------------------------- networked orchestrator
def build_backtest(
    league_id: str,
    user_id: str,
    season: int,
    *,
    sleeper=client,
    max_week: int = 17,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Compute the full-season backtest. Returns ``(tables, meta)`` for :func:`write_sqlite`."""
    league = sleeper.get_league(league_id)
    scoring = league["scoring_settings"]
    slots = lineup_slots(league.get("roster_positions") or [])
    playoff_start = int((league.get("settings") or {}).get("playoff_week_start") or DEFAULT_PLAYOFF_START)

    rosters = sleeper.get_rosters(league_id)
    users = sleeper.get_users(league_id)
    names = team_names_by_roster(rosters, users)
    my_rid = int(find_my_roster(rosters, user_id)["roster_id"])
    players_map = sleeper.get_players_nfl()

    # --- collect the weeks I actually played -----------------------------------------------------
    weeks: list[tuple[int, list, dict]] = []
    for w in range(1, max_week + 1):
        m = sleeper.get_matchups(league_id, w)
        myrow = next((r for r in m if int(r["roster_id"]) == my_rid), None)
        if myrow and (myrow.get("starters") or []) and (myrow.get("points") or 0):
            weeks.append((w, m, myrow))
    weekly_matchups = {w: m for w, m, _ in weeks}

    # --- season actual totals (every player, in our scoring) for the draft grade -----------------
    season_pts: dict[str, float] = defaultdict(float)
    for w, _, _ in weeks:
        for r in sleeper.get_stats(season, w):
            season_pts[str(r["player_id"])] += points(r.get("stats") or {}, scoring)

    # --- weekly backtest -------------------------------------------------------------------------
    weekly_rows: list[dict] = []
    optimal_by_week: dict[int, float] = {}
    for w, m, myrow in weeks:
        pp = {str(k): float(v) for k, v in (myrow.get("players_points") or {}).items()}
        roster_ids = [str(x) for x in (myrow.get("players") or [])]
        actual_starters = [str(x) for x in (myrow.get("starters") or [])]
        actual_pts = round(float(myrow.get("points") or 0.0), 2)

        sol = lineup_from_points(roster_ids, pp, players_map, slots)
        optimal_pts = sol.total
        optimal_by_week[w] = optimal_pts

        proj = score_projections(sleeper.get_projections(season, w), scoring)
        byes = bye_teams(season, w)
        tool_pool: list[LineupPlayer] = []
        for pid in roster_ids:
            row = proj.get(pid) or {}
            meta = players_map.get(pid) or {}
            pos = row.get("pos") or _pos_of(pid, meta)
            if pos not in STARTABLE_POSITIONS:
                continue
            team = pid if pos == "DEF" else (meta.get("team") or row.get("team"))
            tool_pool.append(
                LineupPlayer(
                    player_id=pid, name=row.get("name") or meta.get("full_name") or pid, pos=pos,
                    team=team, proj_pts=round(float(row.get("proj") or 0.0), 2),
                    on_bye=bool(team) and team in byes,
                )
            )
        tool_starter_ids = [sp.player.player_id for sp in optimize(tool_pool, slots).starters]
        tool_pts = round(sum(pp.get(pid, 0.0) for pid in tool_starter_ids), 2)

        mid = myrow.get("matchup_id")
        opp = next((r for r in m if r.get("matchup_id") == mid and int(r["roster_id"]) != my_rid), None)
        opp_pts = round(float(opp.get("points") or 0.0), 2) if opp else None
        opp_name = names.get(int(opp["roster_id"]), "—") if opp else "—"

        team_pts = {int(r["roster_id"]): float(r.get("points") or 0.0) for r in m}
        others = [p for rid, p in team_pts.items() if rid != my_rid]

        weekly_rows.append(
            {
                "week": w,
                "playoff": w >= playoff_start,
                "actual": actual_pts,
                "optimal": optimal_pts,
                "tool": tool_pts,
                "bench_lost": round(optimal_pts - actual_pts, 2),
                "opponent": opp_name,
                "opp_pts": opp_pts if opp_pts is not None else 0.0,
                "result": _result(actual_pts, opp_pts),
                "optimal_result": _result(optimal_pts, opp_pts),
                "tool_result": _result(tool_pts, opp_pts),
                "actual_rank": 1 + sum(1 for p in team_pts.values() if p > actual_pts),
                "optimal_rank": 1 + sum(1 for p in others if p > optimal_pts),
                "top_miss": _top_miss(actual_starters, sol.starters, pp, players_map),
            }
        )

    # --- draft backtest --------------------------------------------------------------------------
    draft_id = (sleeper.get_league_drafts(league_id) or [{}])[0].get("draft_id")
    cfg = draft_roster.roster_config((sleeper.get_draft(draft_id).get("settings") or {})) if draft_id else None
    picks = sleeper.get_draft_picks(draft_id) if draft_id else []
    board = build_board(season, scoring)
    if board and cfg:
        repl = replacement_levels(board, draft_roster.base_starters(cfg),
                                  flex_slots=draft_roster.flex_slots_total(cfg))
        add_vor(board, repl)
    vor_by_id = {p.player_id: p.vor for p in board}
    name_by_id = {p.player_id: p.name for p in board}
    pos_by_id = {p.player_id: p.pos for p in board}
    pick_meta = {str(pk["player_id"]): (pk.get("metadata") or {}) for pk in picks}

    def _pname(pid):
        if pid is None:
            return "—"
        if pid in name_by_id:
            return name_by_id[pid]
        m = pick_meta.get(pid) or {}
        return f"{m.get('first_name', '')} {m.get('last_name', '')}".strip() or pid

    def _ppos(pid):
        if pid is None:
            return ""
        return pos_by_id.get(pid) or (pick_meta.get(pid) or {}).get("position") or ""

    sim = simulate_draft(picks, user_id, vor_by_id, pos_by_id, default_cap=(cfg.rounds if cfg else 999))
    draft_rows = []
    for r in sim:
        my_pid, tool_pid = r["my_pid"], r["tool_pid"]
        my_p = round(season_pts.get(my_pid, 0.0), 1)
        tool_p = round(season_pts.get(tool_pid, 0.0), 1)
        draft_rows.append(
            {
                "pick": r["pick_no"], "round": r["round"],
                "my_pick": _pname(my_pid), "my_pos": _ppos(my_pid), "my_pts": my_p,
                "tool_pick": _pname(tool_pid), "tool_pos": _ppos(tool_pid), "tool_pts": tool_p,
                "diff": round(tool_p - my_p, 1),
            }
        )

    # --- season summary --------------------------------------------------------------------------
    reg = [r for r in weekly_rows if not r["playoff"]]
    opt_ranking = optimal_standings(weekly_matchups, my_rid, optimal_by_week)
    actual_rank = next((t.rank for t in standings(rosters) if t.roster_id == my_rid), 0)
    optimal_rank_overall = (opt_ranking.index(my_rid) + 1) if my_rid in opt_ranking else 0

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "league_id": str(league_id),
        "season": int(season),
        "my_team_name": names.get(my_rid, "me"),
        "my_roster_id": my_rid,
        "n_weeks": len(weekly_rows),
        "n_regular": len(reg),
        "n_teams": len(rosters),
        "actual_total": round(sum(r["actual"] for r in weekly_rows), 2),
        "optimal_total": round(sum(r["optimal"] for r in weekly_rows), 2),
        "tool_total": round(sum(r["tool"] for r in weekly_rows), 2),
        "bench_lost_total": round(sum(r["bench_lost"] for r in weekly_rows), 2),
        "actual_record": _record(reg, "result"),
        "optimal_record": _record(reg, "optimal_result"),
        "tool_record": _record(reg, "tool_result"),
        "actual_rank": actual_rank,
        "optimal_rank": optimal_rank_overall,
        "draft_my_total": round(sum(r["my_pts"] for r in draft_rows), 1),
        "draft_tool_total": round(sum(r["tool_pts"] for r in draft_rows), 1),
        "draft_id": str(draft_id),
    }
    tables = {"weekly": pd.DataFrame(weekly_rows), "draft": pd.DataFrame(draft_rows)}
    return tables, meta


def build_and_write(
    league_id: str,
    user_id: str,
    season: int,
    *,
    db_path,
    sleeper=client,
):
    """Build the backtest for ``season`` and write it to ``db_path``. Returns the path written."""
    tables, meta = build_backtest(league_id, user_id, season, sleeper=sleeper)
    out = write_sqlite(db_path, tables, meta)
    _LOG.info("wrote %s backtest -> %s", season, out)
    return out

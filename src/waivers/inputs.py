"""Assemble the waiver workflow's inputs from live data (the only networked module in Phase 4).

Pulls everything the pure waiver views need and joins it once, returning a :class:`WaiverInputs`
the runnable script renders. Read-only against Sleeper; nflverse via the 24h cache.

Pieces (all reused from earlier phases where possible):

* my roster + this week's projections re-scored in our live ``scoring_settings``
  (``optimizer.inputs.score_projections`` / ``assemble_players``) and my optimal lineup
  (``optimizer.lineup.optimize``) -> my starters + the per-week baseline for free agents;
* the free-agent pool (everyone not on a roster) and my standings + waiver priority
  (``waivers.league``);
* handcuff alerts (``waivers.handcuffs``) -- which also tell us who *just inherited a starting role*
  (a starter OUT this week) for the spend advice;
* a self-computed playoff SOS (``waivers.sos``) from season-to-date nflverse actuals re-scored in our
  settings, plus each team's Weeks 15-17 opponents and bye week;
* usage signals (``waivers.usage``) for the candidate set.

Every candidate that fails an ID join is logged per the CLAUDE.md rule.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

from data import nflverse
from data.ids import build_id_to_sleeper
from optimizer.inputs import (
    OUT_STATUSES,
    assemble_players,
    bye_teams,
    find_my_roster,
    score_projections,
)
from optimizer.lineup import LineupPlayer, lineup_slots, optimize
from scoring.engine import points
from sleeper import client
from waivers.handcuffs import find_handcuffs
from waivers.league import (
    PriorityScarcity,
    free_agents,
    my_standing,
    my_waiver_position,
    priority_scarcity,
)
from waivers.priority import SpendCandidate
from waivers.sos import (
    def_sos_multipliers,
    merge_sos,
    multiplier,
    normalize_team,
    opponents_by_week,
    points_allowed_by_position,
    points_allowed_to_def,
    sos_multipliers,
)
from waivers.stash import PLAYOFF_WEEKS
from waivers.streaming import STREAM_POSITIONS
from waivers.usage import usage_signals

_LOG = logging.getLogger(__name__)

#: Skill positions whose top free agents we evaluate as claim/upgrade candidates.
_SKILL = ("QB", "RB", "WR", "TE")
#: How many top free agents per position to feed the (LP-backed) spend advice.
_SPEND_POOL_PER_POS = {"QB": 6, "RB": 8, "WR": 8, "TE": 6, "K": 3, "DEF": 3}
#: How many top stash candidates to enrich with usage signals.
_USAGE_TOP = 40
#: Games in a fantasy season, for the streaming rest-of-season per-game level (season proj ÷ games).
_SEASON_GAMES = 17


def _stream_candidates(
    scored: Mapping[str, Mapping],
    next_scored: Mapping[str, Mapping],
    season_scored: Mapping[str, Mapping],
    fa_ids: set,
    players_map: Mapping[str, Mapping],
    sos: Mapping[str, Mapping[str, float]],
    opp_by_week: Mapping[str, Mapping[int, str]],
) -> list[dict]:
    """Free-agent K/DEF with their streaming horizon values (this week / next / ROS / playoffs)."""
    out: list[dict] = []
    for pid, row in scored.items():
        pos = row.get("pos")
        if pid not in fa_ids or pos not in STREAM_POSITIONS:
            continue
        team = pid if pos == "DEF" else (players_map.get(pid, {}).get("team") or row.get("team"))
        this_week = float(row.get("proj") or 0.0)
        next_week = float((next_scored.get(pid) or {}).get("proj") or 0.0)
        season_proj = float((season_scored.get(pid) or {}).get("proj") or 0.0)
        ros_pg = season_proj / _SEASON_GAMES if season_proj else this_week
        sched = opp_by_week.get(team or "", {})
        playoff = 0.0
        for w in PLAYOFF_WEEKS:
            opp = sched.get(int(w))
            if not opp:  # bye or unknown in a playoff week -> no game to tilt
                continue
            playoff += ros_pg * (multiplier(sos, opp, "DEF") if pos == "DEF" else 1.0)
        out.append(
            {
                "player_id": pid,
                "name": row.get("name") or pid,
                "pos": pos,
                "team": team,
                "this_week": round(this_week, 2),
                "next_week": round(next_week, 2),
                "ros_pg": round(ros_pg, 2),
                "playoff": round(playoff, 2),
            }
        )
    return out


@dataclass
class WaiverInputs:
    season: int
    week: int
    scoring: Mapping[str, float]
    slots: object
    players_map: Mapping[str, Mapping]
    my_players: list[LineupPlayer]
    my_starters: list[LineupPlayer]  # THIS WEEK's optimal-lineup starters (bye/Out excluded)
    depth_starters: list[LineupPlayer]  # season-basis starters ignoring this week's designations
    free_agent_ids: set
    scored: Mapping[str, Mapping]
    trending: Mapping[str, int]
    scarcity: PriorityScarcity
    my_rank: int
    n_teams: int
    waiver_position: int | None
    handcuffs: list
    spend_candidates: list[SpendCandidate]
    stash_candidates: list[dict]
    sos: Mapping[str, Mapping[str, float]]
    opponents_by_week: Mapping[str, Mapping[int, str]]
    bye_week_of_team: Mapping[str, int]
    usage: Mapping[str, object]
    unjoined: list = field(default_factory=list)
    stream_candidates: list[dict] = field(default_factory=list)  # FA K/DEF with horizon values
    stream_current: dict = field(default_factory=dict)  # pos -> my current starter {name, this_week}


def _name(meta: Mapping, pid: str) -> str:
    full = meta.get("full_name")
    if full:
        return full
    nm = f"{(meta.get('first_name') or '').strip()} {(meta.get('last_name') or '').strip()}".strip()
    return nm or pid


def _fa_lineup_player(
    pid: str, scored: Mapping[str, Mapping], players_map: Mapping[str, Mapping], byes: set[str]
) -> LineupPlayer:
    """Build a startable :class:`LineupPlayer` for a free agent (this week's proj + eligibility)."""
    row = scored.get(pid) or {}
    meta = players_map.get(pid) or {}
    pos = row.get("pos") or meta.get("position")
    is_def = pos == "DEF"
    team = pid if is_def else (meta.get("team") or row.get("team"))
    status = meta.get("injury_status") or None
    return LineupPlayer(
        player_id=pid,
        name=row.get("name") or _name(meta, pid),
        pos=pos,
        team=team,
        proj_pts=round(float(row.get("proj") or 0.0), 2),
        status=status,
        on_bye=bool(team) and team in byes,
        out=status in OUT_STATUSES,
        on_ir=False,
    )


def _depth_lineup_player(
    pid: str, season_scored: Mapping[str, Mapping], players_map: Mapping[str, Mapping]
) -> LineupPlayer | None:
    """A rostered player on a SEASON-projection basis, fully startable (no bye/injury exclusion).

    Used to solve my depth-chart lineup — who my real starters are regardless of this week's
    designations — so the handcuff detector still monitors a starter who is Out right now.
    ``None`` for players who can never occupy a lineup slot (IDP/unknown).
    """
    row = season_scored.get(pid) or {}
    meta = players_map.get(pid) or {}
    pos = row.get("pos") or meta.get("position")
    if pos not in {"QB", "RB", "WR", "TE", "K", "DEF"}:
        return None
    is_def = pos == "DEF"
    team = pid if is_def else (meta.get("team") or row.get("team"))
    return LineupPlayer(
        player_id=pid,
        name=row.get("name") or _name(meta, pid),
        pos=pos,
        team=team,
        proj_pts=round(float(row.get("proj") or 0.0), 2),
        status=meta.get("injury_status") or None,
    )


def _bye_week_by_team(schedule_rows) -> dict[str, int]:
    """Each team's bye week: the REG week in the season it does not play (normalized to Sleeper)."""
    played: dict[str, set] = defaultdict(set)
    weeks: set[int] = set()
    for g in schedule_rows:
        if g.get("game_type") != "REG":
            continue
        w = int(g.get("week"))
        weeks.add(w)
        for t in (normalize_team(g.get("home_team")), normalize_team(g.get("away_team"))):
            if t:
                played[t].add(w)
    max_w = max(weeks) if weeks else 0
    out: dict[str, int] = {}
    for t, pw in played.items():
        missing = [w for w in range(1, max_w + 1) if w not in pw]
        if missing:
            out[t] = missing[0]
    return out


def load_waiver_inputs(
    league_id: str,
    user_id: str,
    season: int,
    week: int,
    *,
    sleeper=client,
    enrich_usage: bool = True,
) -> WaiverInputs:
    """Fetch + join everything the waiver views need for one week (live, cached). Read-only."""
    league = sleeper.get_league(league_id)
    scoring = league["scoring_settings"]
    slots = lineup_slots(league.get("roster_positions") or [])

    rosters = sleeper.get_rosters(league_id)
    roster = find_my_roster(rosters, user_id)
    players_map = sleeper.get_players_nfl()
    scored = score_projections(sleeper.get_projections(season, week), scoring)
    byes = bye_teams(season, week)

    # My roster -> my optimal lineup -> my starters.
    my_players, unjoined = assemble_players(roster, players_map, scored, byes)
    base_sol = optimize(my_players, slots)
    my_starters = [sp.player for sp in base_sol.starters]

    # Season-long projections (per-game baselines for stashes; season basis for the depth lineup).
    season_scored = score_projections(sleeper.get_season_projections(season), scoring)

    # My DEPTH-CHART starters: the season-long optimal lineup with this week's bye/injury eligibility
    # ignored. The weekly optimal lineup hard-excludes Out/IR/bye players, so scanning IT for
    # handcuffs can never see the "my starter is Out" case (CLAUDE.md: flag Q/D/O) — the injured
    # starter is exactly who the detector must keep monitoring.
    depth_pool = [
        _depth_lineup_player(pid, season_scored, players_map)
        for pid in (str(x) for x in (roster.get("players") or []))
        if pid not in {str(x) for x in (roster.get("taxi") or [])}
    ]
    depth_pool = [p for p in depth_pool if p is not None]
    depth_starters = [sp.player for sp in optimize(depth_pool, slots).starters]

    # Free agents + standings/priority (posture weighs BOTH my standings rank and my actual slot in
    # the waiver order — the ordered resource being spent).
    fa_ids = free_agents(players_map.keys(), rosters)
    standing = my_standing(rosters, user_id)
    waiver_position = my_waiver_position(roster)
    scarcity = priority_scarcity(standing.rank, len(rosters), waiver_position=waiver_position)

    # Trending adds -> contention (velocity).
    trending = {str(t["player_id"]): int(t.get("count") or 0) for t in sleeper.get_trending("add", limit=200)}

    # Handcuffs (and which backups just inherited a starting role: their starter is OUT this week).
    handcuffs = find_handcuffs(depth_starters, players_map, fa_ids)
    new_starter_ids = {a.backup_id for a in handcuffs if a.starter_status == "Out"}

    # Spend candidates: top FAs per position (by this week's proj) + handcuff backups.
    by_pos: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for pid, row in scored.items():
        if pid in fa_ids and row.get("pos") in _SPEND_POOL_PER_POS:
            by_pos[row["pos"]].append((pid, float(row.get("proj") or 0.0)))
    cand_ids: set[str] = set(new_starter_ids)
    for pos, lst in by_pos.items():
        lst.sort(key=lambda pp: pp[1], reverse=True)
        cand_ids.update(pid for pid, _ in lst[: _SPEND_POOL_PER_POS[pos]])

    spend_candidates = [
        SpendCandidate(
            player=_fa_lineup_player(pid, scored, players_map, byes),
            is_new_starter=pid in new_starter_ids,
            contention=trending.get(pid, 0),
        )
        for pid in cand_ids
    ]

    # Stash candidates: every skill FA with a positive SEASON projection, at his per-game rate.
    # This week's projection would be 0 for exactly the archetypal stash targets — a player on bye
    # this week, or injured now but back by the playoff weeks. K/DEF are excluded: they stream on
    # their own logic (``stream_candidates`` below), never stash.
    stash_candidates = [
        {
            "player_id": pid,
            "name": row.get("name") or pid,
            "pos": row.get("pos"),
            "team": (players_map.get(pid, {}).get("team") or row.get("team")),
            "baseline": round(float(row.get("proj") or 0.0) / _SEASON_GAMES, 2),
        }
        for pid, row in season_scored.items()
        if pid in fa_ids and row.get("pos") in _SKILL and float(row.get("proj") or 0.0) > 0.0
    ]

    # Self-computed SOS (season-to-date actuals re-scored in our settings) + schedule lookups.
    # Early in a season (or before it) nflverse has no weekly actuals yet — degrade to a neutral
    # SOS (multiplier() defaults to 1.0) instead of crashing the whole waiver report.
    try:
        actuals = nflverse.load_weekly_actuals(season)
        reg = actuals.filter((actuals["season_type"] == "REG") & (actuals["week"] < week))
        pa = points_allowed_by_position(reg.iter_rows(named=True), scoring)
        sos = sos_multipliers(pa)
    except Exception as exc:
        _LOG.warning("weekly actuals for %s unavailable (%s) — SOS defaults to neutral", season, exc)
        sos = {}

    schedule_rows = list(nflverse.load_schedules(season).iter_rows(named=True))
    opp_by_week = opponents_by_week(schedule_rows, PLAYOFF_WEEKS)
    all_opp = opponents_by_week(schedule_rows, range(1, max(PLAYOFF_WEEKS) + 1))
    bye_week_of_team = _bye_week_by_team(schedule_rows)

    # DEF strength-of-schedule: re-score each prior week's DST lines and attribute to the offense
    # faced -> how generous each offense is to defenses (the matchup to stream a D into). Merged into
    # `sos` under a "DEF" key alongside the skill-position multipliers.
    dst_rows: list[dict] = []
    for w in range(1, week):
        for r in sleeper.get_stats(season, w, positions=("DEF",)):
            dst_rows.append(
                {"team": str(r.get("player_id")), "week": w, "points": points(r.get("stats") or {}, scoring)}
            )
    sos = merge_sos(sos, def_sos_multipliers(points_allowed_to_def(dst_rows, all_opp)))

    # K/DEF streaming horizons: this week + next week (real weekly projections), a rest-of-season
    # per-game level (season projection ÷ games), and a Weeks 15-17 outlook (DEF SOS-tilted, K flat).
    next_scored = score_projections(sleeper.get_projections(season, week + 1), scoring)
    stream_candidates = _stream_candidates(
        scored, next_scored, season_scored, fa_ids, players_map, sos, opp_by_week
    )
    stream_current = {
        sp.pos: {"name": sp.name, "this_week": sp.proj_pts}
        for sp in my_starters
        if sp.pos in STREAM_POSITIONS
    }

    # Usage enrichment for the candidate set (best effort).
    usage: dict[str, object] = {}
    if enrich_usage:
        top_stash = sorted(stash_candidates, key=lambda c: c["baseline"], reverse=True)[:_USAGE_TOP]
        usage_ids = (
            cand_ids
            | {c["player_id"] for c in top_stash}
            | {a.backup_id for a in handcuffs}
        )
        try:
            cw = nflverse.load_id_crosswalk()
            usage = usage_signals(
                usage_ids,
                season=season,
                week=week,
                snaps=nflverse.load_snap_counts(season),
                opportunity=nflverse.load_ff_opportunity(season),
                pfr_to_sleeper=build_id_to_sleeper(cw, "pfr_id"),
                gsis_to_sleeper=build_id_to_sleeper(cw, "gsis_id"),
            )
        except Exception as exc:  # nflverse is a best-effort enrichment, never fatal
            _LOG.warning("usage enrichment unavailable: %s", exc)

    if unjoined:
        _LOG.info("%d rostered players had no weekly projection (scored 0.0)", len(unjoined))

    return WaiverInputs(
        season=season,
        week=week,
        scoring=scoring,
        slots=slots,
        players_map=players_map,
        my_players=my_players,
        my_starters=my_starters,
        depth_starters=depth_starters,
        free_agent_ids=fa_ids,
        scored=scored,
        trending=trending,
        scarcity=scarcity,
        my_rank=standing.rank,
        n_teams=len(rosters),
        waiver_position=waiver_position,
        handcuffs=handcuffs,
        spend_candidates=spend_candidates,
        stash_candidates=stash_candidates,
        sos=sos,
        opponents_by_week=opp_by_week,
        bye_week_of_team=bye_week_of_team,
        usage=usage,
        stream_candidates=stream_candidates,
        stream_current=stream_current,
    )

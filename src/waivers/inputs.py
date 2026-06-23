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
from sleeper import client
from waivers.handcuffs import find_handcuffs
from waivers.league import (
    PriorityScarcity,
    free_agents,
    my_standing,
    my_waiver_position,
    priority_scarcity,
    rostered_player_ids,
)
from waivers.priority import SpendCandidate
from waivers.sos import SOS_POSITIONS, normalize_team, opponents_by_week, points_allowed_by_position, sos_multipliers
from waivers.stash import PLAYOFF_WEEKS
from waivers.usage import usage_signals

_LOG = logging.getLogger(__name__)

#: Skill positions whose top free agents we evaluate as claim/upgrade candidates.
_SKILL = ("QB", "RB", "WR", "TE")
#: How many top free agents per position to feed the (LP-backed) spend advice.
_SPEND_POOL_PER_POS = {"QB": 6, "RB": 8, "WR": 8, "TE": 6, "K": 3, "DEF": 3}
#: How many top stash candidates to enrich with usage signals.
_USAGE_TOP = 40


@dataclass
class WaiverInputs:
    season: int
    week: int
    scoring: Mapping[str, float]
    slots: object
    players_map: Mapping[str, Mapping]
    my_players: list[LineupPlayer]
    my_starters: list[LineupPlayer]  # optimal-lineup starters (all slots)
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

    # Free agents + standings/priority.
    fa_ids = free_agents(players_map.keys(), rosters)
    standing = my_standing(rosters, user_id)
    scarcity = priority_scarcity(standing.rank, len(rosters))

    # Trending adds -> contention (velocity).
    trending = {str(t["player_id"]): int(t.get("count") or 0) for t in sleeper.get_trending("add", limit=200)}

    # Handcuffs (and which backups just inherited a starting role: their starter is OUT this week).
    handcuffs = find_handcuffs(my_starters, players_map, fa_ids)
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

    # Stash candidates: every FA with a positive baseline projection this week.
    stash_candidates = [
        {
            "player_id": pid,
            "name": row.get("name") or pid,
            "pos": row.get("pos"),
            "team": (pid if row.get("pos") == "DEF" else (players_map.get(pid, {}).get("team") or row.get("team"))),
            "baseline": float(row.get("proj") or 0.0),
        }
        for pid, row in scored.items()
        if pid in fa_ids and float(row.get("proj") or 0.0) > 0.0
    ]

    # Self-computed playoff SOS (season-to-date actuals re-scored in our settings) + schedule lookups.
    actuals = nflverse.load_weekly_actuals(season)
    reg = actuals.filter((actuals["season_type"] == "REG") & (actuals["week"] < week))
    pa = points_allowed_by_position(reg.iter_rows(named=True), scoring)
    sos = sos_multipliers(pa)

    schedule_rows = list(nflverse.load_schedules(season).iter_rows(named=True))
    opp_by_week = opponents_by_week(schedule_rows, PLAYOFF_WEEKS)
    bye_week_of_team = _bye_week_by_team(schedule_rows)

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
        free_agent_ids=fa_ids,
        scored=scored,
        trending=trending,
        scarcity=scarcity,
        my_rank=standing.rank,
        n_teams=len(rosters),
        waiver_position=my_waiver_position(roster),
        handcuffs=handcuffs,
        spend_candidates=spend_candidates,
        stash_candidates=stash_candidates,
        sos=sos,
        opponents_by_week=opp_by_week,
        bye_week_of_team=bye_week_of_team,
        usage=usage,
    )

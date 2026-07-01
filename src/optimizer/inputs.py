"""Assemble the weekly lineup optimizer's inputs from live data (the only networked module here).

Pulls the pieces the pure LP needs and joins them into a list of :class:`~optimizer.lineup.LineupPlayer`:

* **my roster** -- ``/league/<id>/rosters`` matched on ``owner_id`` (``players`` = full roster,
  ``reserve`` = the IR slot);
* **this week's projections** -- the Sleeper projections endpoint, each row re-scored in *our* live
  ``scoring_settings`` via the Phase 1 engine (never hand-coded);
* **byes** -- derived from the nflverse schedule (a team with no REG game that week);
* **injuries** -- the Sleeper player ``injury_status`` (authoritative per CLAUDE.md); nflverse
  ``load_injuries`` is available as a secondary cross-check.

Eligibility policy: OUT / IR / PUP / suspended / NA / DNR and anyone on the IR slot are hard-excluded
from starting, as are players whose team is on bye. Questionable / Doubtful stay startable but carry
their status through for the risky-start flags. Every rostered player that fails to join a projection
row is logged (and returned) per the CLAUDE.md ID-join rule.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from data import nflverse
from optimizer.lineup import LineupPlayer, lineup_slots
from scoring.engine import points
from sleeper import client

_LOG = logging.getLogger(__name__)

#: Positions that occupy a startable lineup slot in our league.
STARTABLE_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE", "K", "DEF"})

#: Sleeper ``injury_status`` values that rule a player out for the week (vs. Questionable/Doubtful,
#: which stay startable-but-flagged). The IR slot is handled separately via the roster's ``reserve``.
OUT_STATUSES: frozenset[str] = frozenset({"Out", "IR", "PUP", "Sus", "NA", "DNR"})

#: nflverse uses a few abbreviations that differ from Sleeper's; normalize to Sleeper's.
_NFLVERSE_TO_SLEEPER: dict[str, str] = {"LA": "LAR"}


@dataclass
class LineupInputs:
    players: list[LineupPlayer]
    slots: "OrderedDict[str, int]"
    season: int
    week: int
    byes: set[str]
    unjoined: list[tuple[str, str]]  # (player_id, name) rostered players with no weekly projection


def _normalize_team(team: str) -> str:
    return _NFLVERSE_TO_SLEEPER.get(team, team)


def _player_name(meta: Mapping) -> str | None:
    full = meta.get("full_name")
    if full:
        return full
    nm = f"{(meta.get('first_name') or '').strip()} {(meta.get('last_name') or '').strip()}".strip()
    return nm or None


def find_my_roster(rosters: Sequence[Mapping], user_id: str) -> dict:
    """The roster owned by ``user_id`` (``owner_id`` match)."""
    for r in rosters:
        if str(r.get("owner_id")) == str(user_id):
            return dict(r)
    raise ValueError(f"no roster with owner_id={user_id} in this league")


def opponent_roster(
    matchups: Sequence[Mapping], rosters: Sequence[Mapping], my_roster_id: int
) -> dict | None:
    """This week's head-to-head opponent roster (the roster sharing my ``matchup_id``), or ``None``.

    ``None`` when the week's matchups aren't set yet (pre-schedule) or I have no opponent — the caller
    then skips win-probability rather than erroring.
    """
    my_mid = next(
        (m.get("matchup_id") for m in matchups if int(m.get("roster_id")) == my_roster_id), None
    )
    if my_mid is None:
        return None
    opp_rid = next(
        (
            int(m["roster_id"])
            for m in matchups
            if m.get("matchup_id") == my_mid and int(m.get("roster_id")) != my_roster_id
        ),
        None,
    )
    if opp_rid is None:
        return None
    return next((dict(r) for r in rosters if int(r.get("roster_id")) == opp_rid), None)


def score_projections(
    projection_rows: Iterable[Mapping], scoring: Mapping[str, float]
) -> dict[str, dict]:
    """``player_id -> {proj, pos, team, name}`` re-scored in our league settings (best row per id)."""
    out: dict[str, dict] = {}
    for r in projection_rows:
        player = r.get("player") or {}
        pid = str(r.get("player_id"))
        row = {
            "proj": round(points(r.get("stats") or {}, scoring), 2),
            "pos": player.get("position"),
            "team": player.get("team"),
            "name": _player_name(player) or pid,
        }
        prev = out.get(pid)
        if prev is None or row["proj"] > prev["proj"]:
            out[pid] = row
    return out


def bye_teams(
    season: int,
    week: int,
    *,
    fetch_schedules: Callable[[int], pl.DataFrame] = nflverse.load_schedules,
) -> set[str]:
    """Teams on bye in ``week``: in the season's REG schedule but with no game that week."""
    sched = fetch_schedules(season)
    reg = sched.filter((pl.col("game_type") == "REG") & (pl.col("season") == season))
    all_teams = set(reg["home_team"].to_list()) | set(reg["away_team"].to_list())
    wk = reg.filter(pl.col("week") == week)
    playing = set(wk["home_team"].to_list()) | set(wk["away_team"].to_list())
    return {_normalize_team(t) for t in (all_teams - playing)}


def assemble_players(
    roster: Mapping,
    players_map: Mapping[str, Mapping],
    scored: Mapping[str, Mapping],
    byes: set[str],
) -> tuple[list[LineupPlayer], list[tuple[str, str]]]:
    """Join one roster against scored projections + injury + bye into ``LineupPlayer`` rows.

    Returns ``(players, unjoined)`` where ``unjoined`` lists rostered players with no projection row
    (kept in the lineup at 0.0 so they still show, and logged per the ID-join rule).
    """
    reserve = {str(x) for x in (roster.get("reserve") or [])}
    taxi = {str(x) for x in (roster.get("taxi") or [])}

    players: list[LineupPlayer] = []
    unjoined: list[tuple[str, str]] = []
    for pid in (str(x) for x in (roster.get("players") or [])):
        if pid in taxi:
            continue  # practice squad -- not on the active roster
        meta = players_map.get(pid) or {}
        proj_row = scored.get(pid)
        pos = (proj_row or {}).get("pos") or meta.get("position")
        if pos not in STARTABLE_POSITIONS:
            continue  # IDP / FB / unknown -- never occupies one of our lineup slots
        is_def = pos == "DEF"
        team = pid if is_def else (meta.get("team") or (proj_row or {}).get("team"))
        name = (proj_row or {}).get("name") or _player_name(meta) or pid

        if proj_row is None:
            unjoined.append((pid, name))
            proj = 0.0
        else:
            proj = float(proj_row["proj"])

        status = meta.get("injury_status") or None
        on_ir = pid in reserve
        out = status in OUT_STATUSES
        on_bye = bool(team) and team in byes
        players.append(
            LineupPlayer(
                player_id=pid,
                name=name,
                pos=pos,
                team=team,
                proj_pts=round(proj, 2),
                status=status,
                on_bye=on_bye,
                out=out,
                on_ir=on_ir,
            )
        )

    for pid, name in unjoined:
        _LOG.warning("no weekly projection for rostered player %s (%s); scoring as 0.0", name, pid)
    return players, unjoined


def load_lineup_inputs(
    league_id: str,
    user_id: str,
    season: int,
    week: int,
    *,
    sleeper=client,
    fetch_schedules: Callable[[int], pl.DataFrame] = nflverse.load_schedules,
) -> LineupInputs:
    """Fetch + join everything the optimizer needs for one week (live, cached). Read-only."""
    league = sleeper.get_league(league_id)
    scoring = league["scoring_settings"]
    slots = lineup_slots(league.get("roster_positions") or [])

    roster = find_my_roster(sleeper.get_rosters(league_id), user_id)
    players_map = sleeper.get_players_nfl()
    scored = score_projections(sleeper.get_projections(season, week), scoring)
    byes = bye_teams(season, week, fetch_schedules=fetch_schedules)

    players, unjoined = assemble_players(roster, players_map, scored, byes)
    return LineupInputs(
        players=players, slots=slots, season=season, week=week, byes=byes, unjoined=unjoined
    )

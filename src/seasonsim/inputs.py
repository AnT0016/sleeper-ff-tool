"""Live glue for the season simulator (the only networked module here).

Pulls this league's exact ``scoring_settings`` / ``roster_positions`` / playoff settings, the twelve
rosters, and (for a completed or in-progress season) the real head-to-head schedule and the actual
champion. Each rostered player is given its custom-scored season projection from the Phase-1/2 board;
players without a projection (deep waiver adds) carry a zero mean but keep their position so the lineup
optimiser still slots them. Read-only.

For a **completed** season this doubles as the calibration harness: the returned ``actual_*`` fields
let the report put the sim's championship odds next to what really happened, so the tool can be judged
rather than trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from draftsim.inputs import roster_config_from_league
from projections.board import build_board
from sleeper import client
from sleeper.config import MY_USER_ID

from .engine import SeasonPool


@dataclass
class SeasonInputs:
    pool: SeasonPool
    my_team: int
    season: int
    regular_weeks: list[int]
    playoff_weeks: list[int]
    n_playoff_teams: int
    schedule: dict[int, list[tuple[int, int]]]
    schedule_source: str  # "real (matchups)" | "round-robin (generated)"
    completed: bool
    actual_champion: int | None  # team index, if the playoffs have finished
    actual_rank: list[int] | None  # actual REGULAR-SEASON finish per team index (1 = 1st)
    actual_scores: dict[int, list[float]]  # week -> per-team real scores (in-season conditioning)


def _team_name(user: dict) -> str:
    meta = user.get("metadata") or {}
    return meta.get("team_name") or user.get("display_name") or str(user.get("user_id"))


def _position(pid: str, board: dict, master: dict) -> str | None:
    if pid in board:
        return board[pid].pos
    p = master.get(pid)
    if p and p.get("position"):
        return p["position"]
    if pid.isalpha() and pid.isupper() and len(pid) <= 3:  # team defense keyed by abbreviation
        return "DEF"
    return None


def _display(pid: str, board: dict, master: dict) -> str:
    if pid in board:
        return board[pid].name
    p = master.get(pid) or {}
    return p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or pid


def _actual_standings(rosters: list[dict]) -> list[int]:
    """Actual REGULAR-SEASON finish per team index (wins then points-for).

    Note this is the seeding order, NOT the final placement — Sleeper decides final placements by
    the playoff brackets (the champion is surfaced separately via the winners bracket).
    """
    def key(t: int):
        s = rosters[t].get("settings") or {}
        fpts = float(s.get("fpts", 0)) + float(s.get("fpts_decimal", 0)) / 100.0
        return (int(s.get("wins", 0)), fpts)

    order = sorted(range(len(rosters)), key=key, reverse=True)
    rank = [0] * len(rosters)
    for placing, t in enumerate(order, start=1):
        rank[t] = placing
    return rank


def _actual_champion(sleeper, league_id: str, roster_index: dict[int, int]) -> int | None:
    try:
        bracket = sleeper.get_winners_bracket(league_id)
    except Exception:
        return None
    for m in bracket or []:
        if m.get("p") == 1 and m.get("w") is not None:  # p==1 is the championship game
            return roster_index.get(int(m["w"]))
    return None


def _actual_week_scores(
    sleeper, league_id: str, weeks: list[int], roster_ids: list[int]
) -> dict[int, list[float]]:
    """Real per-team scores for already-played weeks (a week counts as played when anyone scored).

    Used to condition a mid-season run on the actual results so the sim answers "odds from here",
    not the preseason question. Weeks with no points yet (unplayed/future) are omitted.
    """
    idx = {int(r): i for i, r in enumerate(roster_ids)}
    out: dict[int, list[float]] = {}
    for w in weeks:
        try:
            rows = sleeper.get_matchups(league_id, w)
        except Exception:
            continue
        vals = [0.0] * len(roster_ids)
        any_points = False
        for row in rows or []:
            rid = row.get("roster_id")
            if rid is None or int(rid) not in idx:
                continue
            pts = float(row.get("points") or 0.0)
            vals[idx[int(rid)]] = pts
            any_points = any_points or pts > 0.0
        if any_points:
            out[int(w)] = vals
    return out


def load_season_inputs(
    league_id: str,
    season: int | None = None,
    *,
    user_id: str = MY_USER_ID,
    sleeper=client,
) -> SeasonInputs:
    """Fetch + assemble everything :func:`engine.simulate_season` needs for one league/season.

    ``season`` defaults to the league's own season (the right projection year to score its roster).
    """
    from .schedule import round_robin, schedule_from_matchups

    league = sleeper.get_league(league_id)
    # Fail fast on a pre-draft league — empty rosters would otherwise "simulate" to garbage (all
    # ties, team index 0 crowned in every sim) without an error.
    status = (league.get("status") or "").lower()
    if status in ("pre_draft", "drafting"):
        raise ValueError(
            f"league {league_id} is {status!r} — rosters aren't drafted yet, nothing to simulate"
        )
    season = int(season or league.get("season") or 0)
    scoring = league["scoring_settings"]
    settings = league.get("settings") or {}
    cfg = roster_config_from_league(league)
    n_teams = cfg.teams

    n_playoff_teams = int(settings.get("playoff_teams") or 6)
    playoff_start = int(settings.get("playoff_week_start") or 15)
    n_rounds = max(1, math.ceil(math.log2(max(n_playoff_teams, 2))))
    playoff_weeks = [playoff_start + i for i in range(n_rounds)]
    regular_weeks = list(range(1, playoff_start))

    board_rows = build_board(season, scoring)
    board = {r.player_id: r for r in board_rows}
    master = sleeper.get_players_nfl()

    rosters = sleeper.get_rosters(league_id)
    users = {u["user_id"]: u for u in sleeper.get_users(league_id)}
    roster_ids = [int(r["roster_id"]) for r in rosters]
    roster_index = {rid: i for i, rid in enumerate(roster_ids)}

    # Flatten every rostered player into disjoint per-player arrays (a player is on one team only).
    team_rosters: list[list[int]] = []
    pos: list[str] = []
    mean: list[float] = []
    names: list[str] = []
    team_names: list[str] = []
    for r in rosters:
        owner = r.get("owner_id")
        team_names.append(_team_name(users.get(owner, {"user_id": owner})))
        cols: list[int] = []
        for pid in r.get("players") or []:
            pid = str(pid)
            p = _position(pid, board, master)
            if p is None:
                continue
            cols.append(len(pos))
            pos.append(p)
            mean.append(board[pid].proj_pts if pid in board else 0.0)
            names.append(_display(pid, board, master))
        team_rosters.append(cols)

    pool = SeasonPool(
        rosters=team_rosters,
        pos=pos,
        mean=np.array(mean, dtype=float),
        cv=np.zeros(len(pos)),  # filled per-position in the engine (pool_arrays)
        p_setback=np.zeros(len(pos)),
        severity=np.zeros(len(pos)),
        names=names,
        team_names=team_names,
        slots=cfg.slots,
        n_teams=n_teams,
    )

    # Fail loudly on the remaining degenerate states the sim would otherwise "answer" with garbage.
    if any(not cols for cols in team_rosters):
        raise ValueError(
            f"league {league_id} has at least one empty roster — a season sim over it is meaningless"
        )
    if not np.any(pool.mean > 0):
        raise ValueError(
            f"no rostered player carries a positive {season} projection — Sleeper hasn't published "
            f"{season} season projections yet (or the board failed to build)"
        )

    schedule = schedule_from_matchups(sleeper, league_id, regular_weeks, roster_ids)
    if schedule:
        schedule_source = "real (matchups)"
    else:
        schedule = round_robin(n_teams, regular_weeks)
        schedule_source = "round-robin (generated)"

    completed = status == "complete"
    actual_champion = _actual_champion(sleeper, league_id, roster_index) if completed else None
    actual_rank = _actual_standings(rosters) if completed else None
    # In-season: pin already-played weeks to the real scores (a completed season stays fully
    # simulated — that's the preseason calibration question; conditioning it would be circular).
    actual_scores = (
        _actual_week_scores(sleeper, league_id, regular_weeks, roster_ids)
        if status == "in_season"
        else {}
    )

    my_team = next(
        (i for i, r in enumerate(rosters) if str(r.get("owner_id")) == str(user_id)), 0
    )
    return SeasonInputs(
        pool=pool,
        my_team=my_team,
        season=season,
        regular_weeks=regular_weeks,
        playoff_weeks=playoff_weeks,
        n_playoff_teams=n_playoff_teams,
        schedule=schedule,
        schedule_source=schedule_source,
        completed=completed,
        actual_champion=actual_champion,
        actual_rank=actual_rank,
        actual_scores=actual_scores,
    )

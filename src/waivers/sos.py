"""Playoff strength-of-schedule, computed ourselves in OUR scoring (pure, no network).

Public fantasy SOS tools score in generic presets; this league's scoring is non-standard (4-pt
passing TDs, half-PPR, distance-based K, rich DST), so we compute matchup difficulty from scratch:

1. Re-score every offensive player-week of nflverse actuals in our live ``scoring_settings`` (the
   Phase 1 :func:`scoring.engine.points` + :func:`data.ids.nflverse_to_sleeper_stats` path).
2. Attribute those points to the **opponent defense** and the player's **position**, and average
   over the games each defense has played -> *points allowed per position per game* for each defense.
3. A defense's :func:`sos_multipliers` value for a position = its PA-to-position ÷ the league average
   for that position. ``> 1`` is a soft matchup (allows more than average), ``< 1`` is tough.

The stash ranker (``waivers.stash``) multiplies a player's per-game baseline by the multiplier of
each Week 15/16/17 opponent. The core here takes plain row dicts + a scoring map (so it is fully
unit-testable offline); the networked glue feeds it nflverse rows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping

from data.ids import nflverse_to_sleeper_stats
from scoring.engine import points

#: Offensive positions we compute matchup difficulty for. K/DEF stream on other logic -> no SOS tilt.
SOS_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE"})

#: nflverse abbreviations that differ from Sleeper's (DEF player_ids use Sleeper's). Mirrors the map
#: in ``optimizer.inputs`` -- keep them in sync.
_NFLVERSE_TO_SLEEPER: dict[str, str] = {"LA": "LAR"}


def normalize_team(team: str | None) -> str | None:
    return _NFLVERSE_TO_SLEEPER.get(team, team) if team else team


def _rescore(row: Mapping, scoring: Mapping[str, float]) -> float:
    return points(nflverse_to_sleeper_stats(row), scoring)


def points_allowed_by_position(
    weekly_rows: Iterable[Mapping],
    scoring: Mapping[str, float],
    *,
    positions: frozenset[str] = SOS_POSITIONS,
    score: Callable[[Mapping, Mapping[str, float]], float] = _rescore,
) -> dict[str, dict[str, float]]:
    """``defense -> {position -> points allowed per game}`` in our scoring.

    ``weekly_rows`` are per-player-week dicts carrying ``position``, ``opponent_team``, ``week`` and
    the nflverse stat columns. Rows should be the regular season only (filter upstream). A defense's
    games-played is the number of distinct weeks it appears as an opponent.
    """
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    games: dict[str, set] = defaultdict(set)
    for row in weekly_rows:
        opp = normalize_team(row.get("opponent_team"))
        if not opp:
            continue
        games[opp].add(row.get("week"))
        pos = row.get("position")
        if pos not in positions:
            continue
        totals[opp][pos] += score(row, scoring)

    per_game: dict[str, dict[str, float]] = {}
    for opp, n in games.items():
        n_games = len(n) or 1
        per_game[opp] = {
            pos: round(totals[opp].get(pos, 0.0) / n_games, 2) for pos in positions
        }
    return per_game


def sos_multipliers(
    pa_by_pos: Mapping[str, Mapping[str, float]],
    *,
    positions: frozenset[str] = SOS_POSITIONS,
) -> dict[str, dict[str, float]]:
    """``defense -> {position -> multiplier}``, where 1.0 = league-average matchup for that position.

    ``multiplier = defense PA-to-position ÷ league-average PA-to-position``. ``> 1`` is a soft
    (favorable) matchup. Positions with no league-average signal default to 1.0.
    """
    league_avg: dict[str, float] = {}
    for pos in positions:
        vals = [d.get(pos, 0.0) for d in pa_by_pos.values() if d.get(pos)]
        league_avg[pos] = sum(vals) / len(vals) if vals else 0.0

    out: dict[str, dict[str, float]] = {}
    for defense, by_pos in pa_by_pos.items():
        out[defense] = {
            pos: round(by_pos.get(pos, 0.0) / league_avg[pos], 3) if league_avg.get(pos) else 1.0
            for pos in positions
        }
    return out


def multiplier(
    sos: Mapping[str, Mapping[str, float]], defense: str | None, pos: str
) -> float:
    """Look up one (defense, position) multiplier, defaulting to 1.0 (no tilt) when unknown."""
    if not defense:
        return 1.0
    return float((sos.get(normalize_team(defense)) or {}).get(pos, 1.0))


def opponents_by_week(
    schedule_rows: Iterable[Mapping], weeks: Iterable[int]
) -> dict[str, dict[int, str]]:
    """``team -> {week -> opponent}`` for the given REG ``weeks``, teams normalized to Sleeper abbrevs.

    ``schedule_rows`` are nflverse schedule dicts (``game_type``, ``week``, ``home_team``,
    ``away_team``).
    """
    weeks = set(int(w) for w in weeks)
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for g in schedule_rows:
        if g.get("game_type") != "REG" or int(g.get("week")) not in weeks:
            continue
        home, away = normalize_team(g.get("home_team")), normalize_team(g.get("away_team"))
        wk = int(g.get("week"))
        if home and away:
            out[home][wk] = away
            out[away][wk] = home
    return dict(out)

"""The single Sleeper API client -- every outbound Sleeper call lives here.

Sleeper exposes two hosts:
  - ``api.sleeper.app/v1`` -- core league / roster / player / state data
  - ``api.sleeper.com``    -- projections and stats (different host, repeated ``position[]`` params)

Endpoints are unofficial and undocumented; keeping them all behind this module means a breaking
change touches exactly one file. All calls go through the shared cached session (see
``http.get_session``) to stay well under Sleeper's ~1000 req/min ceiling. Read-only: GET only.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .http import get_session

V1 = "https://api.sleeper.app/v1"
DATA = "https://api.sleeper.com"

#: Positions covered by our league (skill + K + DEF).
DEFAULT_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")

_session = None


def _sess():
    global _session
    if _session is None:
        _session = get_session()
    return _session


def _get(url: str, params: Any = None) -> Any:
    resp = _sess().get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _position_params(positions: Iterable[str], season_type: str) -> list[tuple[str, str]]:
    return [("season_type", season_type), *[("position[]", p) for p in positions]]


# --------------------------------------------------------------------------- app host (v1)
def get_state(sport: str = "nfl") -> dict:
    """Current season/week and ``previous_season`` pointer."""
    return _get(f"{V1}/state/{sport}")


def get_league(league_id: str) -> dict:
    """Full league object including ``scoring_settings`` and ``roster_positions`` (source of truth)."""
    return _get(f"{V1}/league/{league_id}")


def get_rosters(league_id: str) -> list[dict]:
    return _get(f"{V1}/league/{league_id}/rosters")


def get_users(league_id: str) -> list[dict]:
    return _get(f"{V1}/league/{league_id}/users")


def get_matchups(league_id: str, week: int) -> list[dict]:
    """Per-roster matchup rows; each carries ``players_points`` (player_id -> league-scored points)."""
    return _get(f"{V1}/league/{league_id}/matchups/{week}")


def get_transactions(league_id: str, week: int) -> list[dict]:
    """Waiver/free-agent/trade transactions for a scoring period (``week`` == round)."""
    return _get(f"{V1}/league/{league_id}/transactions/{week}")


def get_players_nfl() -> dict:
    """~5MB master player map keyed by ``player_id``. Cached 24h by the HTTP layer."""
    return _get(f"{V1}/players/nfl")


def get_trending(kind: str, lookback_hours: int = 24, limit: int = 25) -> list[dict]:
    if kind not in ("add", "drop"):
        raise ValueError("kind must be 'add' or 'drop'")
    return _get(
        f"{V1}/players/nfl/trending/{kind}",
        params={"lookback_hours": lookback_hours, "limit": limit},
    )


# ----------------------------------------------------------------- data host (projections/stats)
def get_projections(
    year: int,
    week: int,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    season_type: str = "regular",
) -> list[dict]:
    """Weekly projections. Each row's ``stats`` dict carries raw stats + precomputed pts_* fields."""
    return _get(f"{DATA}/projections/nfl/{year}/{week}", _position_params(positions, season_type))


def get_season_projections(
    year: int,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    season_type: str = "regular",
) -> list[dict]:
    """Full-season projections (no week segment)."""
    return _get(f"{DATA}/projections/nfl/{year}", _position_params(positions, season_type))


def get_stats(
    year: int,
    week: int,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    season_type: str = "regular",
) -> list[dict]:
    """Weekly ACTUAL stats (Sleeper-keyed; includes FG-distance and pts-allowed buckets)."""
    return _get(f"{DATA}/stats/nfl/{year}/{week}", _position_params(positions, season_type))

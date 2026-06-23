"""nflverse ingest via nflreadpy (NOT the deprecated ``nfl_data_py``).

nflreadpy returns polars DataFrames and manages its own download cache. We point that cache at
``data_cache/nflverse_cache`` with a 24h TTL (its default duration) so repeated runs don't re-hit
the nflverse GitHub releases.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import polars as pl

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "nflverse_cache"

_configured = False


def _configure_cache() -> None:
    global _configured
    if _configured:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nfl.config.update_config(
        cache_mode="filesystem",
        cache_dir=_CACHE_DIR,
        cache_duration=24 * 3600,
    )
    _configured = True


def load_weekly_actuals(seasons: int | list[int]) -> pl.DataFrame:
    """Weekly player actual stats (offense skill + kicking distance buckets + individual defense).

    Note: this is per-*player* data. Team DST fantasy aggregates (team points-allowed buckets) are
    NOT here -- validate Defenses via the Sleeper actual-stats endpoint instead.
    """
    _configure_cache()
    return nfl.load_player_stats(seasons=seasons, summary_level="week")


def load_id_crosswalk() -> pl.DataFrame:
    """Player ID crosswalk; carries ``gsis_id`` <-> ``sleeper_id`` (+ name/position/team)."""
    _configure_cache()
    return nfl.load_ff_playerids()

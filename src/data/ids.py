"""ID + stat-key mapping between nflverse and Sleeper.

Two distinct mappings live here:

1. **Identity** -- which nflverse row is which Sleeper player. Skill players join on ``gsis_id``
   (the crosswalk's ``gsis_id`` -> ``sleeper_id``); DST joins on team abbreviation (a Sleeper DEF
   ``player_id`` *is* the team abbreviation, e.g. ``"PHI"``).

2. **Stat keys** -- nflverse weekly columns use verbose names; our ``scoring_settings`` is keyed by
   Sleeper's short codes. ``nflverse_to_sleeper_stats`` translates one nflverse row into a
   Sleeper-keyed stat dict so the generic scoring engine can re-score it in our league settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

# nflverse column -> Sleeper scoring key (1:1).
STAT_MAP: dict[str, str] = {
    "passing_yards": "pass_yd",
    "passing_tds": "pass_td",
    "passing_interceptions": "pass_int",
    "passing_2pt_conversions": "pass_2pt",
    "rushing_yards": "rush_yd",
    "rushing_tds": "rush_td",
    "rushing_2pt_conversions": "rush_2pt",
    "receptions": "rec",
    "receiving_yards": "rec_yd",
    "receiving_tds": "rec_td",
    "receiving_2pt_conversions": "rec_2pt",
    "special_teams_tds": "st_td",
    "fumble_recovery_tds": "fum_rec_td",
    # kicking (distance buckets line up directly)
    "fg_made_0_19": "fgm_0_19",
    "fg_made_20_29": "fgm_20_29",
    "fg_made_30_39": "fgm_30_39",
    "fg_made_40_49": "fgm_40_49",
    "pat_made": "xpm",
}

# Sleeper's fum_lost is one number; nflverse splits lost fumbles across play types.
_FUMBLE_LOST_COLS = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
# Sleeper's 50+ FG bucket combines nflverse's 50-59 and 60+.
_FG_50P_COLS = ("fg_made_50_59", "fg_made_60_")
# Sleeper scores a BLOCKED kick as a miss; nflverse files blocked kicks separately from missed.
_FGMISS_COLS = ("fg_missed", "fg_blocked")
_XPMISS_COLS = ("pat_missed", "pat_blocked")


def _num(row: Mapping[str, Any], col: str) -> float:
    val = row.get(col)
    return float(val) if val is not None else 0.0


def nflverse_to_sleeper_stats(row: Mapping[str, Any]) -> dict[str, float]:
    """Translate one nflverse weekly-stats row into a Sleeper-keyed stat dict.

    Only keys our scoring cares about are emitted; everything else is dropped (and would score
    zero anyway). Composite keys (fumbles lost, the 50+ FG bucket) are summed here.
    """
    stats: dict[str, float] = {}
    for nfl_col, sleeper_key in STAT_MAP.items():
        v = _num(row, nfl_col)
        if v:
            stats[sleeper_key] = v
    fum_lost = sum(_num(row, c) for c in _FUMBLE_LOST_COLS)
    if fum_lost:
        stats["fum_lost"] = fum_lost
    fgm_50p = sum(_num(row, c) for c in _FG_50P_COLS)
    if fgm_50p:
        stats["fgm_50p"] = fgm_50p
    fgmiss = sum(_num(row, c) for c in _FGMISS_COLS)
    if fgmiss:
        stats["fgmiss"] = fgmiss
    xpmiss = sum(_num(row, c) for c in _XPMISS_COLS)
    if xpmiss:
        stats["xpmiss"] = xpmiss
    return stats


def build_id_to_sleeper(crosswalk: pl.DataFrame, source_col: str) -> dict[str, str]:
    """Map an arbitrary crosswalk id column -> Sleeper ``sleeper_id`` from the ff_playerids crosswalk.

    ``source_col`` is any id column the crosswalk carries (e.g. ``"gsis_id"`` for nflverse weekly
    stats / opportunity, ``"pfr_id"`` for snap counts). Rows missing either side are dropped.
    """
    sub = crosswalk.select(source_col, "sleeper_id").drop_nulls()
    return {row[source_col]: str(row["sleeper_id"]) for row in sub.iter_rows(named=True)}


def build_gsis_to_sleeper(crosswalk: pl.DataFrame) -> dict[str, str]:
    """Map nflverse ``gsis_id`` -> Sleeper ``sleeper_id`` from the ff_playerids crosswalk."""
    return build_id_to_sleeper(crosswalk, "gsis_id")

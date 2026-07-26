"""The authoritative registry of collectable sources.

One table, consulted by three different things, which is why it exists at all: the collectors
(``collect.sleeper`` / ``nflverse`` / ``market`` / ``weather``) read their ``key_cols`` from here,
the runners (``scripts/collect.py``, ``scripts/backfill_lake.py``) decide *what to run* from
``cadence`` / ``backfillable``, and the assembler reads ``grain`` to know what a row means. Adding a
source is a single entry here plus a collector — never a change to the storage layer.

Two distinctions worth keeping straight:

* ``grain`` is the grain of a **row** (``week`` = one player-week, ``season`` = one player-season,
  ``game`` = one game). It is *not* the partition layout: nflverse loaders return a whole season at
  once, so ``nflverse_player_week`` has week-grain rows living in a season-partitioned file (that's
  ``write_snapshot(..., week=None)``). Sleeper's weekly endpoints are fetched a week at a time and
  land in week partitions. Either way, ``key_cols`` identifies a row *within its partition*.
* ``cadence`` is *when a capture runs*: ``prelock`` (Thu/Sun, before kickoff — the point-in-time
  snapshots whose value is that they were taken before the outcome existed), ``postgame`` (Tue,
  finalized actuals/usage) and ``backfill`` (the one-time historical pull).

``backfillable`` is the honest answer to "can we recover this for past seasons?". Sleeper's
projection endpoints only ever serve the *latest* numbers, so ``sleeper_proj_*`` are **forward-only**
— they start accumulating at 2026 Week 1 and no amount of work recovers 2016-2025. Everything else
is recoverable from nflverse releases today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from store.lake import RESERVED

#: Row grains a source may declare.
GRAINS: tuple[str, ...] = ("week", "season", "game")

#: Capture cadences a source may participate in.
CADENCES: tuple[str, ...] = ("prelock", "postgame", "backfill")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Source:
    """One collectable dataset. Validated on construction so a bad entry fails at import."""

    name: str
    grain: Literal["week", "season", "game"]
    key_cols: tuple[str, ...]
    cadence: frozenset[str]
    backfillable: bool

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name or ""):
            raise ValueError(f"source name {self.name!r} must be lower_snake_case")
        if self.grain not in GRAINS:
            raise ValueError(f"{self.name}: grain {self.grain!r} not in {GRAINS}")
        if not self.key_cols:
            raise ValueError(f"{self.name}: key_cols must be non-empty")
        if len(set(self.key_cols)) != len(self.key_cols):
            raise ValueError(f"{self.name}: key_cols has duplicates: {self.key_cols}")
        reserved = sorted(set(self.key_cols) & set(RESERVED))
        if reserved:
            # Provenance is the store's, not the row's: a key on _captured_at would defeat the
            # whole point-in-time dedup (every capture would look like a brand-new row).
            raise ValueError(f"{self.name}: key_cols must exclude reserved columns {reserved}")
        if not self.cadence:
            raise ValueError(f"{self.name}: cadence must be non-empty")
        unknown = sorted(set(self.cadence) - set(CADENCES))
        if unknown:
            raise ValueError(f"{self.name}: unknown cadence {unknown}; known: {CADENCES}")
        if ("backfill" in self.cadence) != bool(self.backfillable):
            raise ValueError(
                f"{self.name}: backfillable={self.backfillable} contradicts cadence "
                f"{sorted(self.cadence)} — a backfillable source runs in the backfill, and a "
                "forward-only one cannot"
            )


def _source(
    name: str,
    grain: str,
    key_cols: tuple[str, ...],
    cadence: tuple[str, ...],
    *,
    backfillable: bool,
) -> Source:
    return Source(
        name=name,
        grain=grain,  # type: ignore[arg-type]
        key_cols=key_cols,
        cadence=frozenset(cadence),
        backfillable=backfillable,
    )


_REGISTRY: tuple[Source, ...] = (
    # --- Sleeper (forward-only: the endpoints serve only the latest values) --------------------
    # Raw per-stat projection rows, captured before lock. Week-partitioned, so player_id alone is
    # the key. THE reason this phase exists — unrecoverable once the week is played.
    _source("sleeper_proj_week", "week", ("player_id",), ("prelock",), backfillable=False),
    # Season-long projections; drift over the season is itself a signal (role changes, injuries).
    _source("sleeper_proj_season", "season", ("player_id",), ("prelock",), backfillable=False),
    # Sleeper's own weekly actuals — the cross-check against nflverse for DST/K, where nflverse's
    # player-level feed has no team-defense aggregate.
    _source(
        "sleeper_stats_week", "week", ("player_id",), ("postgame", "backfill"), backfillable=True
    ),
    # --- nflverse (backfillable from public releases) ------------------------------------------
    # load_player_stats(summary_level="week"). season_type is in the key because postseason weeks
    # share the numbering space with the regular season.
    _source(
        "nflverse_player_week",
        "week",
        ("player_id", "season_type", "week"),
        ("postgame", "backfill"),
        backfillable=True,
    ),
    # load_snap_counts: PFR-keyed (no gsis_id). game_id pins the week *and* the opponent.
    _source(
        "nflverse_snaps",
        "week",
        ("pfr_player_id", "game_id"),
        ("postgame", "backfill"),
        backfillable=True,
    ),
    # load_ff_opportunity: expected points + volume shares. posteam is in the key so a team-level
    # row (null player_id) can't collide with another.
    _source(
        "nflverse_ff_opp",
        "week",
        ("game_id", "posteam", "player_id"),
        ("postgame", "backfill"),
        backfillable=True,
    ),
    # Weekly injury report. Point-in-time: the same player-week is re-captured as the report firms
    # up through the week, and each day's capture is kept.
    # date_modified is the report *revision* and belongs in the key: a player is commonly listed
    # twice in one week (e.g. Questionable early, Out after the final practice), and without it the
    # two revisions collapse into whichever the provider happened to list last — persisting a stale
    # status. Verified against real 2024 data: 0 duplicates with it, silent row loss without it.
    _source(
        "nflverse_injuries",
        "week",
        ("gsis_id", "game_type", "week", "date_modified"),
        ("prelock", "backfill"),
        backfillable=True,
    ),
    # Schedules carry the closing Vegas lines and the observed temp/wind, so they're captured on
    # both cadences: pre-lock for the lines, post-game for the final result.
    _source(
        "nflverse_schedules",
        "game",
        ("game_id",),
        ("prelock", "postgame", "backfill"),
        backfillable=True,
    ),
    # Depth charts are time-stamped snapshots; dt is part of the key because several snapshots can
    # land in one week and a player can appear at more than one position.
    _source(
        "nflverse_depth",
        "week",
        ("dt", "team", "gsis_id", "pos_abb"),
        ("prelock", "backfill"),
        backfillable=True,
    ),
    # load_ff_playerids. mfl_id is ffverse's primary key (sleeper_id/gsis_id are nullable).
    _source("id_crosswalk", "season", ("mfl_id",), ("postgame", "backfill"), backfillable=True),
    # --- derived (market + weather) ------------------------------------------------------------
    # Implied team totals derived from schedules' total_line/spread_line. Kept as its own source so
    # the derivation is inspectable rather than buried in the assembler.
    _source("vegas_odds", "game", ("game_id",), ("prelock", "backfill"), backfillable=True),
    # open-meteo forecast pre-lock; historical temp/wind from schedules on backfill. Domes are
    # flagged rather than given fabricated numbers.
    _source("weather", "game", ("game_id",), ("prelock", "backfill"), backfillable=True),
)

#: The registry, keyed by source name.
SOURCES: dict[str, Source] = {s.name: s for s in _REGISTRY}

if len(SOURCES) != len(_REGISTRY):  # pragma: no cover - guards a copy/paste duplicate at import
    raise ValueError("duplicate source name in the registry")


def sources_for_cadence(cadence: str) -> tuple[Source, ...]:
    """Every source a ``--mode <cadence>`` run should collect, in registry order."""
    if cadence not in CADENCES:
        raise ValueError(f"unknown cadence {cadence!r}; known: {CADENCES}")
    return tuple(s for s in _REGISTRY if cadence in s.cadence)


def backfillable_sources() -> tuple[Source, ...]:
    """Every source recoverable for past seasons (what ``scripts/backfill_lake.py`` pulls)."""
    return tuple(s for s in _REGISTRY if s.backfillable)

"""The authoritative registry of collectable sources.

One table, consulted by three different things, which is why it exists at all: the collectors
(``collect.sleeper`` / ``nflverse`` / ``market`` / ``weather``) read their ``key_cols`` from here,
the runners (``scripts/collect.py``, ``scripts/backfill_lake.py``) decide *what to run* from
``cadence`` / ``backfillable``, and the assembler reads ``grain`` to know what a row means. Adding a
source is a single entry here plus a collector — never a change to the storage layer.

Three distinctions worth keeping straight:

* ``grain`` is the grain of a **row** (``week`` = one player-week, ``season`` = one player-season,
  ``game`` = one game). It is *not* the partition layout: nflverse loaders return a whole season at
  once, so ``nflverse_player_week`` has week-grain rows living in a season-partitioned file (that's
  ``write_snapshot(..., week=None)``). Sleeper's weekly endpoints are fetched a week at a time and
  land in week partitions. Either way, ``key_cols`` identifies a row *within its partition*.
* ``cadence`` is *when a capture runs*: ``prelock`` (Thu/Sun, before kickoff — the point-in-time
  snapshots whose value is that they were taken before the outcome existed), ``postgame`` (Tue,
  finalized actuals/usage) and ``backfill`` (the one-time historical pull).
* ``content_known`` is *when the row's week-N content came into existence*, which is a different
  question and the one the assembler needs (see :class:`Source`). Do not read one off the other:
  ``nflverse_schedules`` runs on the ``prelock`` cadence and carries ``result``/``home_score``/
  ``away_score``/``total``, so "captured pre-lock" says nothing about whether the payload is
  pre-kickoff.

``backfillable`` is the honest answer to "can we recover this for past seasons?". Sleeper's
projection endpoints only ever serve the *latest* numbers, so ``sleeper_proj_*`` are **forward-only**
— they start accumulating at 2026 Week 1 and no amount of work recovers 2016-2025. Everything else
is recoverable from nflverse releases today — some of it only back to a point, which is what
``backfillable_from`` records (see :meth:`Source.backfills_season`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from store.lake import DEDUP_POLICIES, DEFAULT_DEDUP, RESERVED

#: Row grains a source may declare.
GRAINS: tuple[str, ...] = ("week", "season", "game")

#: Capture cadences a source may participate in.
CADENCES: tuple[str, ...] = ("prelock", "postgame", "backfill")

#: When a row's content became knowable, independent of when it was captured (see
#: :attr:`Source.content_known`).
CONTENT_KNOWN: tuple[str, ...] = ("pre_kickoff", "post_game", "row_timestamp")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Source:
    """One collectable dataset. Validated on construction so a bad entry fails at import."""

    name: str
    grain: Literal["week", "season", "game"]
    key_cols: tuple[str, ...]
    cadence: frozenset[str]
    backfillable: bool
    #: When the content of a week-N row came into existence — **not** when it was captured. The
    #: assembler needs this because a backfilled row's ``_captured_at`` is the backfill run, which
    #: says nothing about knowability: every 2016-2025 row in the lake is stamped with one 2026
    #: instant, so a literal "captured before week N's lock" rule admits *nothing* same-week and
    #: leaves the training frame with no market, injury or role features at all.
    #:
    #: * ``pre_kickoff`` — week-N content exists before week N's first kickoff (a projection, a
    #:   practice report, a betting line), so a backfilled row is legal as a week-N feature.
    #: * ``post_game`` — week-N content exists only once week N has been played (actuals, usage,
    #:   final scores), so it is legal only for weeks **after** N.
    #: * ``row_timestamp`` — the row carries its own event stamp, and admissibility is resolved
    #:   from that column rather than from the week (``nflverse_depth.dt``).
    #:
    #: A **mixed** source takes the label of its most dangerous content unless the safe family is
    #: separately named *and* separately guarded. ``weather`` qualifies (``forecast_*`` vs.
    #: ``observed_*``, the latter gated on ``_captured_at >= kickoff_utc``); ``nflverse_schedules``
    #: does not — its post-game columns are the outcome itself, and ``vegas_odds`` already exposes
    #: the sanctioned pre-game view of the same feed.
    content_known: Literal["pre_kickoff", "post_game", "row_timestamp"]
    #: How the store collapses repeat captures of one key (implemented by ``store.lake._dedup``):
    #:
    #: * ``per_capture_date`` (default) — keep the latest row per key **per UTC capture date**, so a
    #:   later-day capture is a new point-in-time snapshot. Right for any source the provider can
    #:   revise in place (a corrected stat line, an injury report that firms up through the week).
    #: * ``first_capture`` — keep the earliest capture of each key and ignore the capture date. Right
    #:   only for a source whose natural key already carries its own observation timestamp, so a row
    #:   is immutable once seen and re-capturing the cumulative feed records nothing new. It is what
    #:   stops ``nflverse_depth`` multiplying by capture count once the pre-lock cron runs (#15).
    #:
    #: **Declared, not derived from "does the key hold a timestamp".** ``nflverse_injuries`` carried
    #: ``date_modified`` in its key until #17 and would have been mis-classified immutable by such a
    #: rule, yet it is genuinely revisable (Thursday's *Questionable*, Sunday's *Out*) and must keep
    #: the default. The property is semantic — is a row immutable once observed? — not syntactic.
    #:
    #: ``first_capture`` **assumes** immutability: if a provider ever revised an already-published
    #: keyed observation, the correction would be dropped with no drift signal at all, because
    #: retaining a per-date copy is exactly what the policy removes. For ``nflverse_depth`` that trade
    #: is accepted; weigh it before declaring ``first_capture`` on any future source.
    dedup: Literal["per_capture_date", "first_capture"] = DEFAULT_DEDUP
    #: Earliest season the backfill can actually recover, when that is later than the lake's span.
    #: ``None`` means "as far back as anyone asks" — the normal case.
    backfillable_from: int | None = None

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
        if self.content_known not in CONTENT_KNOWN:
            raise ValueError(
                f"{self.name}: content_known {self.content_known!r} not in {CONTENT_KNOWN}"
            )
        if self.dedup not in DEDUP_POLICIES:
            raise ValueError(f"{self.name}: dedup {self.dedup!r} not in {DEDUP_POLICIES}")
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
        if self.backfillable_from is not None and not self.backfillable:
            raise ValueError(
                f"{self.name}: backfillable_from={self.backfillable_from} on a source that is not "
                "backfillable at all — drop it rather than implying history starts somewhere"
            )

    def backfills_season(self, season: int) -> bool:
        """Is ``season`` actually recoverable for this source?

        The two questions ``backfillable`` alone cannot separate. ``nflverse_depth`` *is*
        recoverable — nflverse publishes it, and the collector reads it fine — but only from 2025,
        because the pre-2025 feed is a different shape with no clean key (see ``collect.nflverse``).
        Expressing that by dropping ``"backfill"`` from its cadence is impossible: the invariant
        above ties cadence to ``backfillable``, so it would have to claim the feed is unrecoverable,
        which is false for 2025+. A start year states it exactly, and keeps the backfill from walking
        2016-2024 to write nine empty partitions.
        """
        if not self.backfillable:
            return False
        return self.backfillable_from is None or int(season) >= int(self.backfillable_from)


def _source(
    name: str,
    grain: str,
    key_cols: tuple[str, ...],
    cadence: tuple[str, ...],
    *,
    backfillable: bool,
    content_known: str,
    dedup: str = DEFAULT_DEDUP,
    backfillable_from: int | None = None,
) -> Source:
    return Source(
        name=name,
        grain=grain,  # type: ignore[arg-type]
        key_cols=key_cols,
        cadence=frozenset(cadence),
        backfillable=backfillable,
        content_known=content_known,  # type: ignore[arg-type]
        dedup=dedup,  # type: ignore[arg-type]
        backfillable_from=backfillable_from,
    )


_REGISTRY: tuple[Source, ...] = (
    # --- Sleeper (forward-only: the endpoints serve only the latest values) --------------------
    # Raw per-stat projection rows, captured before lock. Week-partitioned, so player_id alone is
    # the key. THE reason this phase exists — unrecoverable once the week is played.
    _source(
        "sleeper_proj_week",
        "week",
        ("player_id",),
        ("prelock",),
        backfillable=False,
        content_known="pre_kickoff",
    ),
    # Season-long projections; drift over the season is itself a signal (role changes, injuries).
    _source(
        "sleeper_proj_season",
        "season",
        ("player_id",),
        ("prelock",),
        backfillable=False,
        content_known="pre_kickoff",
    ),
    # Sleeper's own weekly actuals — the cross-check against nflverse for DST/K, where nflverse's
    # player-level feed has no team-defense aggregate.
    _source(
        "sleeper_stats_week",
        "week",
        ("player_id",),
        ("postgame", "backfill"),
        backfillable=True,
        content_known="post_game",
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
        content_known="post_game",
    ),
    # load_snap_counts: PFR-keyed (no gsis_id). game_id pins the week *and* the opponent.
    _source(
        "nflverse_snaps",
        "week",
        ("pfr_player_id", "game_id"),
        ("postgame", "backfill"),
        backfillable=True,
        content_known="post_game",
    ),
    # load_ff_opportunity: expected points + volume shares.
    # This key is exact but wider than it needs to be, and the original reason for that was wrong.
    # It said posteam was here so a team-level row (null player_id) could not collide with another;
    # that guarantee is unreachable, because a null key value is filtered before the store ever sees
    # it (collect.nflverse._identified). posteam is in fact redundant: (game_id, player_id) is
    # already unique across the player rows -- verified 5441/5441 in 2020 and 5586/5586 in 2024.
    # Kept anyway (harmless, and narrowing a shipped key would rewrite partitions), but do not
    # mistake it for load-bearing.
    _source(
        "nflverse_ff_opp",
        "week",
        ("game_id", "posteam", "player_id"),
        ("postgame", "backfill"),
        backfillable=True,
        content_known="post_game",
    ),
    # Weekly injury report. Point-in-time: the same player-week is re-captured as the report firms
    # up through the week, and each day's capture is kept -- Thursday's Questionable and Sunday's Out
    # are two rows because their capture dates differ, which is the revision stream that matters.
    # This key used to carry date_modified, on the stated grounds that "a player is commonly listed
    # twice in one week". Measured on the real feed, that is 2 player-weeks out of 6,213 in 2024
    # (0.03%) -- the row loss #12 found was real but tiny, not systemic. nflverse then dropped
    # date_modified entirely in the 2025 release, making the source uncapturable from 2025 on. So the
    # key is the player-week itself: unique in both eras (6,213/6,213 and 6,068/6,068), zero
    # null/blank gsis_id in 2016/2020/2024/2025. date_modified survives as a payload column where the
    # feed has it, and collect_injuries ranks on it so a legacy collapse keeps the *final* pre-game
    # report rather than whichever row the provider happened to list last.
    _source(
        "nflverse_injuries",
        "week",
        ("gsis_id", "game_type", "week"),
        ("prelock", "backfill"),
        backfillable=True,
        content_known="pre_kickoff",
    ),
    # Schedules carry the closing Vegas lines and the observed temp/wind, so they're captured on
    # both cadences: pre-lock for the lines, post-game for the final result.
    _source(
        "nflverse_schedules",
        "game",
        ("game_id",),
        ("prelock", "postgame", "backfill"),
        backfillable=True,
        content_known="post_game",
    ),
    # Depth charts are time-stamped snapshots; dt is part of the key because several snapshots can
    # land in one week and a player can appear at more than one position.
    # 2025-forward: nflverse rewrote the feed that season and the key above is the modern one. The
    # legacy shape carries none of it and has no clean key of its own, so collect_depth_charts
    # returns an empty capture for an older season -- backfillable_from keeps the backfill from
    # walking 2016-2024 to produce nine of those.
    # dedup=first_capture (the ONLY source that departs from the default): the feed is a cumulative
    # append-only log and dt is in the key, so a row is immutable and twice-weekly pre-lock captures
    # would otherwise re-write the whole season-to-date as new rows -- ~10M rows for ~548k keys (#15).
    _source(
        "nflverse_depth",
        "week",
        ("dt", "team", "gsis_id", "pos_abb"),
        ("prelock", "backfill"),
        backfillable=True,
        content_known="row_timestamp",
        dedup="first_capture",
        backfillable_from=2025,
    ),
    # load_ff_playerids. mfl_id is ffverse's primary key (sleeper_id/gsis_id are nullable).
    _source(
        "id_crosswalk",
        "season",
        ("mfl_id",),
        ("postgame", "backfill"),
        backfillable=True,
        content_known="post_game",
    ),
    # --- derived (market + weather) ------------------------------------------------------------
    # Implied team totals derived from schedules' total_line/spread_line. Kept as its own source so
    # the derivation is inspectable rather than buried in the assembler.
    _source(
        "vegas_odds",
        "game",
        ("game_id",),
        ("prelock", "backfill"),
        backfillable=True,
        content_known="pre_kickoff",
    ),
    # open-meteo forecast pre-lock; historical temp/wind from schedules on backfill. Domes are
    # flagged rather than given fabricated numbers.
    _source(
        "weather",
        "game",
        ("game_id",),
        ("prelock", "backfill"),
        backfillable=True,
        content_known="pre_kickoff",
    ),
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


def backfillable_sources(season: int | None = None) -> tuple[Source, ...]:
    """Every source recoverable for past seasons (what ``scripts/backfill_lake.py`` pulls).

    With a ``season``, narrowed to the sources whose history actually reaches it — see
    :meth:`Source.backfills_season`.
    """
    if season is None:
        return tuple(s for s in _REGISTRY if s.backfillable)
    return tuple(s for s in _REGISTRY if s.backfills_season(season))

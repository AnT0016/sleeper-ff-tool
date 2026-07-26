"""Point-in-time capture of Sleeper's own projections and actuals.

These are the captures this whole phase exists for: Sleeper's projection endpoints serve **only the
latest** numbers, so a week's pre-lock projection is unrecoverable the moment the week is played.
``sleeper_proj_week`` / ``sleeper_proj_season`` are therefore forward-only (capture starts 2026 Week
1); ``sleeper_stats_week`` is the finalized-actuals cross-check, and is backfillable.

**Nothing is scored here.** Rows carry the provider's raw stat keys and values verbatim — including
Sleeper's own preset ``pts_half_ppr`` / ``pts_ppr`` / ``pts_std`` — so the lake stays
scoring-agnostic and the assembler (ticket 7) can re-score any snapshot in whatever
``scoring_settings`` the league has at the time. Nothing is filtered either: rows with no game and
an ADP-only stat line are stored as the feed returned them, because "who wasn't projected this week"
is itself data.

Row shape is **flat** — a handful of identity/meta columns, then the raw ``stats`` dict merged in
one key per column. Parquet is columnar and the store's readers project columns
(``read_parquet(key, columns=...)``); a nested ``stats`` struct would defeat that and force every
consumer to unpack. The cost is that a stat a player has no value for reads back as ``NaN`` rather
than an absent key, so re-scoring must drop nulls before summing.

Only *point-in-time* meta is copied. ``team``/``opponent``/``game_id``/``date`` come from the
projection row itself, never from its embedded ``player`` object: that object is the current state
of Sleeper's mutable player master (for 2025 Week 1 rows pulled today, 78 of them disagree — those
players have since changed team), and copying it in would inject tomorrow's truth into a snapshot
whose entire value is that it only knows today's. ``position``/``fantasy_positions`` exist nowhere
else in the payload, so they are taken from ``player`` and are as-of capture time — correct for the
live weekly captures these sources are built for.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from sleeper import client
from sleeper.client import DEFAULT_POSITIONS

from .base import Collected, dedupe_rows
from .registry import SOURCES

_LOG = logging.getLogger(__name__)

#: Top-level row fields kept verbatim next to the stats. ``season``/``week`` are deliberately
#: absent: the store stamps ``_season``/``_week`` on every row, and ``sport``/``category``/
#: ``week_shard`` are constants or Sleeper-internal. ``last_modified`` is the provider's revision —
#: paired with ``_captured_at`` it tells you how stale a projection was when we captured it.
_META: tuple[str, ...] = (
    "team",
    "opponent",
    "game_id",
    "date",
    "season_type",
    "company",
    "last_modified",
)


def _player_id(raw: Mapping[str, Any]) -> str | None:
    """The row's key. Coerced to ``str`` (DEF rows key on a team abbreviation, e.g. ``"PHI"``)."""
    pid = raw.get("player_id")
    if pid is None:
        return None
    return str(pid).strip() or None


def _fantasy_positions(player: Mapping[str, Any]) -> str | None:
    """``fantasy_positions`` as a comma-joined string.

    A list column would be the faithful shape, but it makes the partition a nested parquet type for
    one rarely-read field; the joined string keeps every lake column flat and greppable. Kept at all
    because it explains rows the position filter otherwise seems to contradict — a FB with
    ``fantasy_positions=["RB"]`` is returned by a ``position[]=RB`` request.
    """
    values = player.get("fantasy_positions")
    if not isinstance(values, (list, tuple)) or not values:
        return None
    return ",".join(str(v) for v in values)


def _flatten(raw: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """One raw Sleeper row -> one flat lake row (meta + verbatim stats)."""
    player = raw.get("player") or {}
    stats = raw.get("stats") or {}
    row: dict[str, Any] = {
        "player_id": _player_id(raw),
        "position": player.get("position"),
        "fantasy_positions": _fantasy_positions(player),
    }
    row.update({name: raw.get(name) for name in _META})
    clashing = sorted(set(row) & set(stats))
    if clashing:
        # Never observed (Sleeper's stat keys are stat-shaped), but if the feed ever grows a stat
        # named like our meta, the raw stat wins — the promise is that stats pass through unchanged.
        _LOG.warning(
            "%s: stat key(s) %s shadow meta columns; keeping the raw stat", source, clashing
        )
    row.update(stats)
    return row


def _freshness(row: Mapping[str, Any]) -> int:
    """Provider revision stamp (epoch ms), used to pick a winner if a player is listed twice."""
    value = row.get("last_modified")
    return int(value) if isinstance(value, (int, float)) else 0


def _collect(
    source: str,
    season: int,
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    week: int | None = None,
) -> Collected:
    rows = dedupe_rows(
        (_flatten(r, source=source) for r in raw_rows),
        SOURCES[source].key_cols,
        source=source,
        freshness=_freshness,
    )
    return Collected.for_source(source, season, rows, week=week)


def collect_proj_week(
    season: int,
    week: int,
    *,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    fetch: Callable[..., list[dict]] | None = None,
) -> Collected:
    """``sleeper_proj_week``: the pre-lock weekly projection snapshot (forward-only).

    ``fetch`` defaults to ``sleeper.client.get_projections`` and is injectable for offline tests.
    ``season_type`` is fixed to the client default (``regular``): a postseason capture would land in
    the same week partition and collide with the regular-season week of the same number.
    """
    get = fetch or client.get_projections
    return _collect(
        "sleeper_proj_week", season, get(season, week, positions=positions), week=int(week)
    )


def collect_proj_season(
    season: int,
    *,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    fetch: Callable[..., list[dict]] | None = None,
) -> Collected:
    """``sleeper_proj_season``: full-season projections (forward-only).

    Captured on the pre-lock cadence like the weekly one: the *drift* of a season projection across
    captures is the signal (role changes, injuries), which is why a later-day capture is retained
    beside the earlier one rather than overwriting it.
    """
    get = fetch or client.get_season_projections
    return _collect("sleeper_proj_season", season, get(season, positions=positions), week=None)


def collect_stats_week(
    season: int,
    week: int,
    *,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    fetch: Callable[..., list[dict]] | None = None,
) -> Collected:
    """``sleeper_stats_week``: finalized weekly actuals (post-game cadence, backfillable).

    Sleeper-keyed, so it is the cross-check for DST and K — where nflverse's player-level feed has
    no team-defense aggregate and no FG-distance buckets.
    """
    get = fetch or client.get_stats
    return _collect(
        "sleeper_stats_week", season, get(season, week, positions=positions), week=int(week)
    )

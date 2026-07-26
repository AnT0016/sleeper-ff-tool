"""Shared collector plumbing: the :class:`Collected` envelope and key hygiene.

Collectors are deliberately storage-free — they return rows plus provenance and the *runner*
persists them (``store.write_snapshot``), which is what makes every collector unit-testable offline
against a fixture. :class:`Collected` is that hand-off, and it is defined once here so tickets 3-4
(nflverse, market/weather) reuse the same envelope rather than three near-identical dataclasses.

Two invariants live in this module because getting either wrong corrupts the lake *silently*:

* **The declared key is the registry's key.** ``Collected`` refuses a source it doesn't know and
  refuses key columns that disagree with ``collect.registry.SOURCES``. The store deduplicates on
  whatever key it is handed, so a collector that invented its own key would quietly delete real rows
  on merge. Pinning it here means an integration drift is an import-time/constructor error rather
  than data loss discovered a season later.
* **A key must actually identify a row.** :func:`dedupe_rows` drops null-keyed rows and collapses
  repeats *before* the store sees them. ``store.lake`` warns about both (nulls compare equal when
  deduping; a repeated key looks like a superseded row), and those warnings are a defect signal, not
  routine noise — a collector's output should produce none.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .registry import SOURCES

_LOG = logging.getLogger(__name__)

#: How many offending keys a hygiene warning names before truncating (mirrors ``store.lake``).
_SAMPLE_KEYS = 3


@dataclass(frozen=True)
class Collected:
    """One capture of one source: the rows, plus everything the store needs to file them.

    ``week`` is the *partition* week, not the row grain: nflverse loaders return a whole season at
    once (``week=None``, week-grain rows keyed by a ``week`` column), while Sleeper's weekly
    endpoints are fetched a week at a time and land in week partitions. ``key_cols`` identifies a
    row within its partition and always mirrors the registry entry.
    """

    source: str
    season: int
    week: int | None
    rows: list[dict] = field(repr=False)
    key_cols: tuple[str, ...]

    def __post_init__(self) -> None:
        entry = SOURCES.get(self.source)
        if entry is None:
            raise ValueError(
                f"unknown source {self.source!r}; add it to collect.registry.SOURCES "
                f"(known: {sorted(SOURCES)})"
            )
        key_cols = tuple(self.key_cols)
        if key_cols != entry.key_cols:
            raise ValueError(
                f"{self.source}: key_cols {key_cols} disagree with the registry {entry.key_cols}; "
                "the store dedups on the key it is given, so a private key deletes real rows"
            )
        object.__setattr__(self, "key_cols", key_cols)
        object.__setattr__(self, "season", int(self.season))
        object.__setattr__(self, "week", None if self.week is None else int(self.week))
        object.__setattr__(self, "rows", list(self.rows))

    @classmethod
    def for_source(
        cls,
        source: str,
        season: int,
        rows: Iterable[Mapping[str, Any]],
        *,
        week: int | None = None,
    ) -> Collected:
        """Build a capture whose ``key_cols`` come from the registry (never hand-written)."""
        entry = SOURCES.get(source)
        return cls(
            source=source,
            season=season,
            week=week,
            rows=[dict(r) for r in rows],
            key_cols=entry.key_cols if entry is not None else (),
        )

    def __len__(self) -> int:
        return len(self.rows)


def _is_missing(value: Any) -> bool:
    """Null-ish for key purposes: ``None``, NaN, or a blank/whitespace-only string."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return isinstance(value, str) and not value.strip()


def _key_of(row: Mapping[str, Any], key_cols: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(c) for c in key_cols)


def dedupe_rows(
    rows: Iterable[Mapping[str, Any]],
    key_cols: Sequence[str],
    *,
    source: str,
    freshness: Callable[[Mapping[str, Any]], Any] | None = None,
) -> list[dict]:
    """Rows with a usable, unique key — in first-seen key order.

    Drops rows whose key is null-ish (they cannot be identified, and the store would fold them all
    into one), then collapses repeats of the same key keeping the row with the highest
    ``freshness(row)`` (default: the last one the provider listed). Both losses are logged with a
    sample: real feeds do occasionally repeat a player, and silently picking one is how a stale row
    ends up looking like a point-in-time fact.
    """
    keys = list(key_cols)
    if not keys:
        raise ValueError(f"{source}: key_cols must be non-empty")

    rank = freshness or (lambda _row: 0)
    kept: dict[tuple[Any, ...], dict] = {}
    kept_rank: dict[tuple[Any, ...], Any] = {}
    dropped_null: list[dict] = []
    repeated: list[tuple[Any, ...]] = []

    for raw in rows:
        row = dict(raw)
        key = _key_of(row, keys)
        if any(_is_missing(v) for v in key):
            dropped_null.append(row)
            continue
        if key in kept:
            repeated.append(key)
            # >= keeps the provider's later listing on a tie, matching the store's "a later row
            # supersedes an equally-stamped earlier one".
            if rank(row) >= kept_rank[key]:
                kept[key], kept_rank[key] = row, rank(row)
            continue
        kept[key], kept_rank[key] = row, rank(row)

    if dropped_null:
        _LOG.warning(
            "%s: dropped %d row(s) with a null/blank value in key column(s) %s. Sample: %s",
            source,
            len(dropped_null),
            keys,
            [_key_of(r, keys) for r in dropped_null[:_SAMPLE_KEYS]],
        )
    if repeated:
        distinct = list(dict.fromkeys(repeated))
        _LOG.warning(
            "%s: %d duplicate row(s) across %d key(s) %s in one capture — kept the freshest per "
            "key. Sample: %s",
            source,
            len(repeated),
            len(distinct),
            keys,
            distinct[:_SAMPLE_KEYS],
        )
    return list(kept.values())

"""The point-in-time data lake: append-only, lookahead-free partitioned parquet.

Sleeper's projection endpoints only ever serve the *latest* value, so every week we don't capture a
pre-lock snapshot is a training label lost forever. This module is the storage half of that capture:
collectors hand it raw provider-native rows, it stamps provenance and persists them so that "what was
known at time T" stays recoverable later.

Layout (identical on every backend, so the *where* is a config flip and never a rewrite)::

    <source>/season=<YYYY>/<source>_<YYYY>_wk<WW>.parquet    # a week-partitioned capture
    <source>/season=<YYYY>/<source>_<YYYY>_season.parquet    # a season-partitioned capture

Every row carries the four :data:`RESERVED` provenance columns — ``_source``, ``_season``, ``_week``
(null for season partitions), ``_captured_at`` (ISO-8601 **UTC**). Collectors must not supply them;
the store owns provenance so a collector can't accidentally forge it.

**Dedup / point-in-time rule.** Within a partition, a row is identified by its natural ``key_cols``
*plus its capture date* (``_captured_at[:10]``, UTC). Re-running a capture the same day overwrites
that day's row (idempotent re-runs), while a capture on a **later** day is retained alongside the
earlier one — that is what makes projection/injury *drift* observable rather than silently flattened.
Finalized sources (completed-week actuals) converge to one row per key naturally, because they stop
changing.

``captured_at`` is always **passed in**, never read from the clock in here: a test must be able to
write "Tuesday's" and "Sunday's" captures deterministically.

Writes go through a temp file in the destination directory plus :func:`os.replace`, so a crashed or
failed write can never leave a half-written partition behind. Temp files do not end in ``.parquet``
and are therefore invisible to every reader/lister in this module.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

_LOG = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Local materialization of the lake. Gitignored — the production store is the cloud backend, and a
#: local run is either a dev backfill or a test.
LAKE_ROOT: Path = _REPO_ROOT / "data_cache" / "lake"

#: Provenance columns the store attaches to every row. Collectors must not emit these.
RESERVED: tuple[str, ...] = ("_source", "_season", "_week", "_captured_at")

#: Backend selected by env (``local`` | ``r2``). Default keeps every run working with no credentials.
LAKE_BACKEND: str = (os.environ.get("LAKE_BACKEND") or "local").strip().lower() or "local"

#: Suffix of in-flight writes. Deliberately not ``.parquet`` so listers skip it.
TMP_SUFFIX: str = ".tmp"

_KEY_RE = re.compile(r"^(?P<source>[^/]+)/season=(?P<season>\d{4})/(?P=source)_(?P=season)_"
                     r"(?:wk(?P<week>\d{2})|season)\.parquet$")


# --------------------------------------------------------------------------- backend interface
@runtime_checkable
class StorageBackend(Protocol):
    """Where lake partitions live.

    Keys are POSIX-style paths relative to the lake root (see :func:`partition_key`) — the same key
    is a file path locally and an object key in cloud storage, which is why the layout is identical
    on both. Implementations are responsible for making :meth:`write_parquet` atomic.
    """

    def path_for(self, key: str) -> Path:
        """A displayable locator for ``key``. A real filesystem path only for local backends."""

    def exists(self, key: str) -> bool:
        ...

    def read_parquet(self, key: str) -> pd.DataFrame:
        ...

    def write_parquet(self, key: str, df: pd.DataFrame) -> Path:
        """Persist ``df`` at ``key`` atomically; returns the locator written."""

    def list_keys(self, prefix: str = "") -> list[str]:
        """Every partition key under ``prefix`` (a *directory* prefix, e.g. a source name)."""


class LocalParquetBackend:
    """Parquet files under ``root`` (default :data:`LAKE_ROOT`). The dev/default backend.

    No credentials, no account — this is what tests, local backfills and anyone without cloud creds
    use. The cloud backend (ticket 9) is a drop-in sibling behind :class:`StorageBackend`.
    """

    def __init__(self, root: Path | str = LAKE_ROOT) -> None:
        self.root = Path(root)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LocalParquetBackend(root={str(self.root)!r})"

    def path_for(self, key: str) -> Path:
        return self.root.joinpath(*key.split("/"))

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def read_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.path_for(key))

    def write_parquet(self, key: str, df: pd.DataFrame) -> Path:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp in the destination directory: unique (safe against a concurrent writer) and on the
        # same filesystem, which is what makes os.replace atomic.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=TMP_SUFFIX)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)  # a failed write leaves the committed partition untouched
            raise
        return path

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self.root.joinpath(*prefix.split("/")) if prefix else self.root
        if not base.is_dir():
            return []
        return sorted(
            p.relative_to(self.root).as_posix() for p in base.rglob("*.parquet") if p.is_file()
        )


_BACKENDS: dict[str, Callable[[], StorageBackend]] = {"local": LocalParquetBackend}
_ACTIVE: StorageBackend | None = None


def register_backend(name: str, factory: Callable[[], StorageBackend]) -> None:
    """Register a backend factory under ``name`` (how ``store.r2`` will plug itself in)."""
    _BACKENDS[name.strip().lower()] = factory


def get_backend(name: str | None = None) -> StorageBackend:
    """The backend for ``name``, or the process-wide one selected by :data:`LAKE_BACKEND`."""
    global _ACTIVE
    if name is None:
        if _ACTIVE is None:
            _ACTIVE = get_backend(LAKE_BACKEND)
        return _ACTIVE
    key = name.strip().lower()
    if key not in _BACKENDS:
        # Backends live in sibling modules that self-register on import (store.r2 -> "r2"), so a
        # name we haven't seen gets one import attempt before it's called unknown.
        try:
            importlib.import_module(f"{__package__}.{key}")
        except ImportError:
            pass
    if key not in _BACKENDS:
        raise ValueError(
            f"unknown LAKE_BACKEND {name!r}; available: {sorted(_BACKENDS)}. "
            "Set LAKE_BACKEND=local to use the committed-parquet dev backend."
        )
    return _BACKENDS[key]()


def set_backend(backend: StorageBackend | None) -> None:
    """Override the process-wide backend (``None`` re-resolves from :data:`LAKE_BACKEND`)."""
    global _ACTIVE
    _ACTIVE = backend


def _resolve(backend: StorageBackend | None) -> StorageBackend:
    return backend if backend is not None else get_backend()


# --------------------------------------------------------------------------- keys & paths
def partition_key(source: str, season: int, week: int | None = None) -> str:
    """The backend-agnostic key of one partition (see the module layout diagram)."""
    season = int(season)
    stem = f"{source}_{season}_season" if week is None else f"{source}_{season}_wk{int(week):02d}"
    return f"{source}/season={season}/{stem}.parquet"


def snapshot_path(
    source: str, season: int, week: int | None = None, *, backend: StorageBackend | None = None
) -> Path:
    """Where a partition lives on the active backend."""
    return _resolve(backend).path_for(partition_key(source, season, week))


def _parse_key(key: str) -> tuple[str, int | None, int | None]:
    """``(source, season, week)`` from a partition key; unparseable keys degrade to ``(dir, …)``."""
    m = _KEY_RE.match(key)
    if not m:
        head = key.split("/", 1)[0]
        return head, None, None
    week = m.group("week")
    return m.group("source"), int(m.group("season")), (int(week) if week is not None else None)


# --------------------------------------------------------------------------- provenance
def normalize_captured_at(value: str) -> str:
    """Validate an ISO-8601 capture stamp and normalize it to UTC (``…+00:00``).

    Normalizing here means ``_captured_at[:10]`` is always the *UTC* capture date (so a European
    evening run and a US afternoon run land on the dates their instants really are), and that two
    spellings of the same instant dedup against each other. A naive stamp is read as UTC.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("captured_at must be a non-empty ISO-8601 UTC string")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"captured_at {value!r} is not ISO-8601") from exc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat()


def _attach_reserved(
    df: pd.DataFrame, *, source: str, season: int, week: int | None, captured_at: str
) -> pd.DataFrame:
    forged = [c for c in RESERVED if c in df.columns]
    if forged:
        raise ValueError(
            f"{source}: collected rows carry reserved column(s) {forged}; the store owns provenance"
        )
    n = len(df)
    out = df.copy()
    out["_source"] = source
    out["_season"] = pd.array([int(season)] * n, dtype="Int64")
    out["_week"] = pd.array([None if week is None else int(week)] * n, dtype="Int64")
    out["_captured_at"] = captured_at
    return out


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Payload columns in their original order, provenance last."""
    payload = [c for c in df.columns if c not in RESERVED]
    return df[[*payload, *RESERVED]]


def _dedup(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    """Keep the latest ``_captured_at`` per ``key_cols`` **per UTC capture date** (see module doc)."""
    stamps = pd.to_datetime(df["_captured_at"], utc=True, format="ISO8601")
    work = df.assign(_ts=stamps, _capture_date=stamps.dt.strftime("%Y-%m-%d"))
    work = work.sort_values("_ts", kind="stable")  # stable: a same-instant re-capture wins over the
    work = work.drop_duplicates(  # already-stored copy, which is what makes a re-run idempotent
        subset=[*key_cols, "_capture_date"], keep="last"
    )
    return work.drop(columns=["_ts", "_capture_date"]).reset_index(drop=True)


def _empty_frame() -> pd.DataFrame:
    """An absent partition, shaped so callers can concat/filter without special-casing."""
    return pd.DataFrame(
        {
            "_source": pd.Series(dtype="object"),
            "_season": pd.Series(dtype="Int64"),
            "_week": pd.Series(dtype="Int64"),
            "_captured_at": pd.Series(dtype="object"),
        }
    )


# --------------------------------------------------------------------------- read / write
def write_snapshot(
    source: str,
    season: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    captured_at: str,
    week: int | None = None,
    key_cols: Sequence[str],
    backend: StorageBackend | None = None,
) -> Path:
    """Merge ``rows`` into one partition, point-in-time deduped, written atomically.

    ``key_cols`` is the row's natural key *within this source* (see ``collect.registry.SOURCES``);
    it must not include the reserved columns, and every name must exist in ``rows`` — a missing key
    column raises rather than silently deduping on the wrong thing (which would delete real rows).

    Empty ``rows`` is a no-op: the path is returned but nothing is created or overwritten, so an
    off-season or failed collector never blanks a good partition.
    """
    store = _resolve(backend)
    key = partition_key(source, season, week)
    path = store.path_for(key)

    materialized = list(rows)
    if not materialized:
        _LOG.info("%s season=%s week=%s: no rows — nothing written", source, season, week)
        return path

    key_cols = tuple(key_cols)
    if not key_cols:
        raise ValueError(f"{source}: key_cols must be non-empty (the natural key of a row)")
    overlap = [c for c in key_cols if c in RESERVED]
    if overlap:
        raise ValueError(f"{source}: key_cols must exclude reserved columns {overlap}")

    stamp = normalize_captured_at(captured_at)
    fresh = pd.DataFrame(materialized)
    missing = [c for c in key_cols if c not in fresh.columns]
    if missing:
        raise ValueError(
            f"{source}: key_cols {missing} absent from the collected rows "
            f"(columns: {sorted(fresh.columns)[:20]})"
        )
    fresh = _attach_reserved(fresh, source=source, season=season, week=week, captured_at=stamp)

    if store.exists(key):
        prior = store.read_parquet(key)
        if not prior.empty:
            fresh = pd.concat([prior, fresh], ignore_index=True)

    merged = _ordered(_dedup(fresh, key_cols))
    written = store.write_parquet(key, merged)
    _LOG.info("%s season=%s week=%s: %d rows -> %s", source, season, week, len(merged), written)
    return written


def read_snapshot(
    source: str, season: int, week: int | None = None, *, backend: StorageBackend | None = None
) -> pd.DataFrame:
    """One partition, or an empty (reserved-column-only) frame when it was never captured."""
    store = _resolve(backend)
    key = partition_key(source, season, week)
    if not store.exists(key):
        return _empty_frame()
    return store.read_parquet(key)


def read_source(
    source: str,
    seasons: Iterable[int] | None = None,
    *,
    backend: StorageBackend | None = None,
) -> pd.DataFrame:
    """Every captured partition of ``source`` (optionally limited to ``seasons``), concatenated."""
    store = _resolve(backend)
    wanted = None if seasons is None else {int(s) for s in seasons}
    frames: list[pd.DataFrame] = []
    for key in store.list_keys(source):
        _, key_season, _ = _parse_key(key)
        if wanted is not None and key_season not in wanted:
            continue
        part = store.read_parquet(key)
        if not part.empty:
            frames.append(part)
    if not frames:
        return _empty_frame()
    return pd.concat(frames, ignore_index=True)


def lake_inventory(*, backend: StorageBackend | None = None) -> pd.DataFrame:
    """One row per partition: ``source, season, week, n_rows, path, latest_captured_at``."""
    store = _resolve(backend)
    rows: list[dict[str, Any]] = []
    for key in store.list_keys():
        source, season, week = _parse_key(key)
        part = store.read_parquet(key)
        stamps = part["_captured_at"].dropna() if "_captured_at" in part.columns else pd.Series([])
        rows.append(
            {
                "source": source,
                "season": season,
                "week": week,
                "n_rows": len(part),
                "path": str(store.path_for(key)),
                "latest_captured_at": max(stamps) if len(stamps) else "",
            }
        )
    inventory = pd.DataFrame(
        rows, columns=["source", "season", "week", "n_rows", "path", "latest_captured_at"]
    )
    inventory["season"] = inventory["season"].astype("Int64")
    inventory["week"] = inventory["week"].astype("Int64")
    return inventory.sort_values(["source", "season", "week"], na_position="first").reset_index(
        drop=True
    )

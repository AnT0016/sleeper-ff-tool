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

That capture-date retention is the **default** policy (``per_capture_date``). A source whose natural
key already carries its own observation timestamp — ``nflverse_depth.dt`` — declares ``first_capture``
instead (:data:`DEDUP_POLICIES`, resolved from the registry): its rows are immutable, so a later
capture of the cumulative feed records nothing new and only the earliest per key is kept. Without that
policy, twice-weekly pre-lock captures multiply the partition by the capture count (#15). The policy
changes *which capture survives*, never the set of distinct keys — so a downstream read that keys on
the timestamp (the assembler's as-of position join on ``dt``) sees exactly the same rows either way.

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
import pyarrow.parquet as pq

_LOG = logging.getLogger(__name__)

#: How many offending keys a capture-integrity warning names before truncating.
_SAMPLE_KEYS = 3

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Local materialization of the lake. Gitignored — the production store is the cloud backend, and a
#: local run is either a dev backfill or a test.
LAKE_ROOT: Path = _REPO_ROOT / "data_cache" / "lake"

#: Provenance columns the store attaches to every row. Collectors must not emit these.
RESERVED: tuple[str, ...] = ("_source", "_season", "_week", "_captured_at")

#: The dedup policies :func:`_dedup` implements, declared per source as ``Source.dedup``. Defined
#: here because the store owns the merge; ``collect.registry`` imports these to validate its
#: declarations, exactly as it imports :data:`RESERVED`.
DEDUP_POLICIES: tuple[str, ...] = ("per_capture_date", "first_capture")

#: The policy a source keeps unless it declares otherwise — today's behaviour for every source but
#: ``nflverse_depth`` (#15). Also the fallback for a ``source`` the registry does not know.
DEFAULT_DEDUP: str = "per_capture_date"

#: Backend selected by env (``local`` | ``s3``). Default keeps every run working with no credentials.
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

    def path_for(self, key: str) -> Path | str:
        """A displayable locator for ``key``.

        A real filesystem path only for local backends — an object store returns its URI
        (``s3://bucket/key``), which is not expressible as a :class:`~pathlib.Path` (``//``
        collapses). Callers may only ``str()`` it; anything path-shaped is local-backend territory.
        """

    def exists(self, key: str) -> bool:
        ...

    def open_partition(self, key: str) -> pq.ParquetFile:
        """A **footer-first** handle on ``key``: schema and row count with no payload fetched.

        Parquet metadata lives in a footer, so a backend that can serve byte ranges answers
        ``schema_arrow.names`` / ``metadata.num_rows`` from a tail read alone, and
        :meth:`~pyarrow.parquet.ParquetFile.read` then pulls only the requested column chunks.
        Handing back the *handle* rather than the answers is what lets :func:`_read_projected`
        resolve which columns exist and read them over **one** open — see its docstring for why the
        alternative costs a round-trip per partition.
        """

    def read_parquet(self, key: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """``key`` as a frame. ``columns`` projects — a backend must not fetch the rest.

        Every name in ``columns`` must exist in the partition; tolerating drift is
        :func:`_read_projected`'s job, and doing it here would hide a genuine typo.
        """

    def write_parquet(self, key: str, df: pd.DataFrame) -> Path | str:
        """Persist ``df`` at ``key`` atomically; returns the locator written."""

    def list_keys(self, prefix: str = "") -> list[str]:
        """Every partition key under ``prefix`` (a *directory* prefix, e.g. a source name)."""

    def partition_summary(self, key: str) -> tuple[int, str]:
        """``(n_rows, latest _captured_at)`` for ``key`` **without** reading payload columns.

        This is on the protocol because only the backend knows the cheap way: locally it is the
        parquet footer (row count is metadata) plus a single column chunk. A backend that answered
        by materializing the object would silently turn :func:`lake_inventory` — a call whose name
        promises it is cheap — into a download of the entire lake.
        """


def partition_summary_from(handle: pq.ParquetFile) -> tuple[int, str]:
    """``(n_rows, latest _captured_at)`` from an open footer handle — the backend-agnostic half.

    The row count is pure footer metadata (no row group is decoded) and the stamp costs exactly one
    column chunk. Both backends delegate here so "cheap" cannot quietly drift apart between them.
    """
    n_rows = int(handle.metadata.num_rows)
    if "_captured_at" not in handle.schema_arrow.names:
        return n_rows, ""
    stamps = [
        s for s in handle.read(columns=["_captured_at"]).column("_captured_at").to_pylist() if s
    ]
    # Stamps are normalized to '...+00:00' on write, so max() over the strings is chronological.
    return n_rows, (max(stamps) if stamps else "")


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

    def read_parquet(self, key: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(
            self.path_for(key), columns=list(columns) if columns is not None else None
        )

    def open_partition(self, key: str) -> pq.ParquetFile:
        return pq.ParquetFile(self.path_for(key))

    def partition_summary(self, key: str) -> tuple[int, str]:
        with self.open_partition(key) as handle:
            return partition_summary_from(handle)

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
    """Register a backend factory under ``name`` (how ``store.s3`` will plug itself in)."""
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
        # Backends live in sibling modules that self-register on import (store.s3 -> "s3"), so a
        # name we haven't seen gets one import attempt before it's called unknown.
        try:
            importlib.import_module(f"{__package__}.{key}")
        except ModuleNotFoundError as exc:
            # Only "there is no such backend module" is a miss. A backend module that exists but
            # can't import its *client* library (store.s3 without boto3) must surface that, or the
            # operator is told "unknown backend s3" and goes looking in the wrong place entirely.
            if exc.name != f"{__package__}.{key}":
                raise
    if key not in _BACKENDS:
        raise ValueError(
            f"unknown LAKE_BACKEND {name!r}; available: {sorted(_BACKENDS)}. "
            "Set LAKE_BACKEND=local to use the local-parquet dev backend."
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
) -> Path | str:
    """Where a partition lives on the active backend.

    A :class:`~pathlib.Path` on the local backend, an ``s3://…`` URI string on a cloud one — see
    :meth:`StorageBackend.path_for`. Treat it as displayable, not as something to ``open()``.
    """
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


def _check_capture_integrity(fresh: pd.DataFrame, key_cols: Sequence[str], source: str) -> None:
    """Warn when a capture's declared key does not actually identify its rows.

    :func:`_dedup` collapses on ``key_cols`` + capture date, and pandas treats ``NaN == NaN`` when
    deduping. Two shapes therefore lose real rows *silently*:

    * a **null** in a key column — every null-keyed row in the capture folds into one;
    * a **repeated key within one capture** — ``key_cols`` is not a key for this source. Seen in real
      data: ``nflverse_injuries`` keyed without ``date_modified`` folded one player's *Questionable*
      and *Out* reports for the same week into a single row, keeping whichever the provider happened
      to list last — i.e. persisting a stale status and saying nothing.

    Superseding an *earlier capture* of the same key is the intended point-in-time rule and is never
    warned about; only loss *within* one capture is a defect. Warn rather than raise: real provider
    feeds carry nulls, and a cron that hard-fails collects nothing at all. The reconciliation counts
    in :func:`write_snapshot`'s log line make the loss auditable either way.
    """
    keys = list(key_cols)

    nulls = fresh[keys].isna().any(axis=1)
    if bool(nulls.any()):
        _LOG.warning(
            "%s: %d/%d rows carry a null in key column(s) %s — nulls compare equal when deduping, "
            "so they collapse into one row. Widen key_cols or fix the collector.",
            source, int(nulls.sum()), len(fresh), [c for c in keys if fresh[c].isna().any()],
        )

    dups = fresh.duplicated(subset=keys, keep=False)
    if bool(dups.any()):
        clashing = fresh.loc[dups, keys].drop_duplicates()
        _LOG.warning(
            "%s: key_cols %s do not identify rows within one capture — %d rows share %d key(s), so "
            "%d row(s) are dropped as if superseded. Sample: %s. That is a collector/registry bug, "
            "not point-in-time dedup.",
            source, keys, int(dups.sum()), len(clashing), int(dups.sum()) - len(clashing),
            clashing.head(_SAMPLE_KEYS).to_dict("records"),
        )


def _dedup(df: pd.DataFrame, key_cols: Sequence[str], policy: str) -> pd.DataFrame:
    """Collapse a merged partition to one row per key, per the source's dedup ``policy``.

    ``per_capture_date`` (the default) keeps the latest ``_captured_at`` per ``key_cols`` **per UTC
    capture date**, so a later-day capture is a new point-in-time snapshot and a same-day re-run is
    idempotent (see the module docstring). Right for anything the provider can revise in place.

    ``first_capture`` keeps the **earliest** ``_captured_at`` per ``key_cols`` and ignores the capture
    date. Right only for a source whose key already carries its own observation timestamp
    (``nflverse_depth.dt``): such a row is immutable, so re-capturing the cumulative feed on a later
    date records nothing new, and retaining a copy per capture date would multiply the partition by
    the capture count (#15). The surviving stamp is the first time we observed the key — the more
    honest point-in-time answer than an arbitrary later re-observation.
    """
    stamps = pd.to_datetime(df["_captured_at"], utc=True, format="ISO8601")
    if policy == "first_capture":
        # Ascending sort + keep="first": the earliest capture of each key wins, and a later capture
        # of an already-stored key is dropped rather than appended. Stable, so a same-instant re-run
        # keeps the already-stored copy (idempotent), matching the per_capture_date branch below.
        work = df.assign(_ts=stamps).sort_values("_ts", kind="stable")
        work = work.drop_duplicates(subset=list(key_cols), keep="first")
        return work.drop(columns=["_ts"]).reset_index(drop=True)
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
    dedup: str | None = None,
    backend: StorageBackend | None = None,
) -> Path | str:
    """Merge ``rows`` into one partition, point-in-time deduped, written atomically.

    ``key_cols`` is the row's natural key *within this source* (see ``collect.registry.SOURCES``);
    it must not include the reserved columns, and every name must exist in ``rows`` — a missing key
    column raises rather than silently deduping on the wrong thing (which would delete real rows).

    ``dedup`` selects how repeat captures of a key collapse (see :func:`_dedup`). ``None`` — the
    normal case — resolves the policy the registry **declares** for ``source``, so the policy sitting
    beside ``key_cols`` is honoured by every caller rather than only the runner. An explicit value
    (one of :data:`DEDUP_POLICIES`) overrides that, for tests and one-offs.

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

    if dedup is None:
        # Resolve from the registry, imported here rather than at module scope: registry imports
        # RESERVED from this module, so a top-level import would be circular (the same reason
        # runner.plan_run imports analysis.snapshot inside the function). A source the registry does
        # not know keeps the default, so write_snapshot stays usable for an ad-hoc/unregistered key.
        from collect.registry import SOURCES

        entry = SOURCES.get(source)
        dedup = entry.dedup if entry is not None else DEFAULT_DEDUP
    elif dedup not in DEDUP_POLICIES:
        raise ValueError(f"{source}: unknown dedup policy {dedup!r}; known: {DEDUP_POLICIES}")

    stamp = normalize_captured_at(captured_at)
    fresh = pd.DataFrame(materialized)
    missing = [c for c in key_cols if c not in fresh.columns]
    if missing:
        raise ValueError(
            f"{source}: key_cols {missing} absent from the collected rows "
            f"(columns: {sorted(fresh.columns)[:20]})"
        )
    fresh = _attach_reserved(fresh, source=source, season=season, week=week, captured_at=stamp)
    _check_capture_integrity(fresh, key_cols, source)

    n_new, n_prior = len(fresh), 0
    if store.exists(key):
        prior = store.read_parquet(key)
        if not prior.empty:
            n_prior = len(prior)
            fresh = pd.concat([prior, fresh], ignore_index=True)

    merged = _ordered(_dedup(fresh, key_cols, dedup))
    written = store.write_parquet(key, merged)
    # Reconcile rows in vs. rows out: a bare post-dedup count would hide every dropped row. Name the
    # side that was dropped, too, because it flips with the policy: per_capture_date keeps the fresh
    # row and drops the stored one ("superseded"), while first_capture keeps the stored row and drops
    # the fresh re-observation. "N superseded" on a first_capture cron would read as "the fresh
    # capture replaced the stored data" -- the exact opposite of what happened.
    dropped = n_new + n_prior - len(merged)
    collapse = "re-observations dropped" if dedup == "first_capture" else "superseded"
    _LOG.info(
        "%s season=%s week=%s: %d new + %d existing -> %d rows (%d %s) -> %s",
        source, season, week, n_new, n_prior, len(merged), dropped, collapse, written,
    )
    return written


def _read_projected(
    store: StorageBackend, key: str, columns: Sequence[str] | None
) -> pd.DataFrame:
    """``key`` projected to ``columns``, tolerating a partition that lacks some of them.

    The projection is the point of having ``columns=`` on the protocol at all: the assembler wants
    a dozen columns out of ``nflverse_player_week``'s 150, and materializing the rest is wasted
    memory locally and wasted transfer on a cloud backend.

    Tolerating a miss is not laziness — a raw layer's schema legitimately drifts across seasons
    (nflverse dropped ``injuries.date_modified`` in 2025, and rewrote the depth feed entirely), so a
    hard failure would make a ten-season read impossible for any column that has not always
    existed. A missing column comes back as all-NA, which is what a reader would have had to
    reindex to anyway.

    **Why the intersection is resolved from the footer rather than by catching a failed read.** The
    obvious shape — try the projection, fall back to a full read on error — quietly becomes a full
    read for any *routinely* sparse source. Measured on the real lake against the exact column set
    the assembler asks ``sleeper_stats_week`` for, **175 of 175 partitions** took the fallback
    (median 64 columns missing, max 79), because Sleeper writes only the stat keys a week actually
    produced: a week in which no defence pitched a shutout has no ``pts_allow_0`` column at all.
    That is the widest-fanned source in the lake, and on an object store the fallback is a full
    object download *plus* a wasted round-trip on the attempt that failed — precisely the transfer
    the projection exists to avoid. Reading the footer first costs one tail read (which opening the
    partition pays anyway) and makes the projection real on every backend.

    The handle is opened **once**: schema resolution and the read share it, so a cloud backend pays
    one round-trip per partition, not one to ask and another to fetch.
    """
    if columns is None:
        return store.read_parquet(key)
    wanted = list(dict.fromkeys(columns))
    with store.open_partition(key) as handle:
        present = set(handle.schema_arrow.names)
        hit = [c for c in wanted if c in present]
        # An empty `hit` still yields the partition's row count (an Arrow table keeps num_rows with
        # no columns selected), so reindexing below gives N rows of NA rather than an empty frame.
        part = handle.read(columns=hit).to_pandas()
    if len(hit) == len(wanted):
        return part
    # DEBUG, not INFO: for a sparse stat feed this is routine rather than notable. Logging it per
    # partition would be 180 lines on a ten-season read and would teach the reader to skim.
    _LOG.debug(
        "%s: column(s) %s absent from this partition — returning them as NA (a raw layer's "
        "schema drifts across seasons)", key, [c for c in wanted if c not in present],
    )
    return part.reindex(columns=wanted)


def read_snapshot(
    source: str,
    season: int,
    week: int | None = None,
    *,
    columns: Sequence[str] | None = None,
    backend: StorageBackend | None = None,
) -> pd.DataFrame:
    """One partition, or an empty (reserved-column-only) frame when it was never captured."""
    store = _resolve(backend)
    key = partition_key(source, season, week)
    if not store.exists(key):
        return _empty_frame()
    return _read_projected(store, key, columns)


def read_source(
    source: str,
    seasons: Iterable[int] | None = None,
    *,
    columns: Sequence[str] | None = None,
    backend: StorageBackend | None = None,
) -> pd.DataFrame:
    """Every captured partition of ``source`` (optionally limited to ``seasons``), concatenated.

    ``columns`` projects each partition (see :func:`_read_projected`); a column absent from a given
    partition reads back as NA rather than raising.
    """
    store = _resolve(backend)
    wanted = None if seasons is None else {int(s) for s in seasons}
    frames: list[pd.DataFrame] = []
    for key in store.list_keys(source):
        _, key_season, _ = _parse_key(key)
        if wanted is not None and key_season not in wanted:
            continue
        part = _read_projected(store, key, columns)
        if not part.empty:
            frames.append(part)
    if not frames:
        return _empty_frame()
    return pd.concat(frames, ignore_index=True)


def lake_inventory(*, backend: StorageBackend | None = None) -> pd.DataFrame:
    """One row per partition: ``source, season, week, n_rows, path, latest_captured_at``.

    Goes through :meth:`StorageBackend.partition_summary`, so **no payload column is ever fetched** —
    on a cloud backend this is a listing plus one small metadata read per partition, not a download
    of the lake.
    """
    store = _resolve(backend)
    rows: list[dict[str, Any]] = []
    for key in store.list_keys():
        source, season, week = _parse_key(key)
        n_rows, latest = store.partition_summary(key)
        rows.append(
            {
                "source": source,
                "season": season,
                "week": week,
                "n_rows": n_rows,
                "path": str(store.path_for(key)),
                "latest_captured_at": latest,
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

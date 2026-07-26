"""The production lake backend: any S3-compatible object store (Backblaze B2 today).

This is :class:`store.lake.LocalParquetBackend`'s cloud sibling — same key layout, same
point-in-time merge semantics, different *where*. The cloud crons write here so captured
pre-lock data outlives the runner that collected it; local runs keep using parquet on disk.

Nothing in here is vendor-specific. B2 is simply the endpoint we point at, and moving to
R2/AWS/Tigris later is one ``LAKE_S3_ENDPOINT`` string, not a rewrite.

Configuration is **env only** (the four values live in GitHub Actions secrets and the owner's user
env, never in the repo)::

    LAKE_S3_ENDPOINT          https://s3.eu-central-003.backblazeb2.com
    LAKE_S3_ACCESS_KEY_ID     the bucket-scoped application key id
    LAKE_S3_SECRET_ACCESS_KEY the application key
    LAKE_S3_BUCKET            the private bucket name
    LAKE_S3_REGION            optional; derived from the endpoint host when unset

A missing value raises :class:`S3ConfigError` at construction. It must never degrade to the local
backend: a cron that silently wrote to a container-local ``data_cache/lake/`` would report success
every week and accumulate nothing, which is the one failure this phase cannot detect after the fact.

**Atomicity without rename.** Object stores have no rename, so the local backend's
temp-file-plus-``os.replace`` trick has no analogue. Here "atomic" means the fully merged frame goes
up in a **single** ``PutObject``: readers see either the old object or the new one, and an
interrupted run leaves nothing behind to clean up. (This is why the backend is built on ``boto3``
rather than ``pyarrow.fs.S3FileSystem``, whose output stream uses a three-request multipart upload —
see the PR for the full comparison.)

**Cheap metadata.** Parquet is a footer-first format, and this backend is careful to exploit that:
:meth:`S3Backend.partition_summary` and every projected read go through :class:`_S3RangeReader`, so
they fetch a tail block plus the requested column chunks instead of the object. That is what keeps
:func:`store.lake.lake_inventory` — 412 partitions today — a listing plus small range reads rather
than a download of the entire lake.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Sequence
from contextlib import closing
from urllib.parse import urlparse

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import ClientError

from store.lake import partition_summary_from, register_backend

_LOG = logging.getLogger(__name__)

#: Env var per setting. The first four are required; the fifth is an escape hatch (see
#: :func:`_region_from_endpoint`).
ENDPOINT_ENV = "LAKE_S3_ENDPOINT"
ACCESS_KEY_ENV = "LAKE_S3_ACCESS_KEY_ID"
SECRET_KEY_ENV = "LAKE_S3_SECRET_ACCESS_KEY"
BUCKET_ENV = "LAKE_S3_BUCKET"
REGION_ENV = "LAKE_S3_REGION"

#: Used when the endpoint host doesn't encode a region. Most S3-compatible vendors ignore it; it
#: only has to be *consistent*, because it is part of the SigV4 credential scope.
DEFAULT_REGION = "us-east-1"

#: How much of the object's tail to pull in one go. A parquet footer is a few KB for these
#: partitions (the widest is ~250 columns), and reading it takes three seeks — magic, footer length,
#: footer body — so caching one block turns three round-trips into one.
TAIL_BYTES = 64 * 1024

#: Error codes S3 implementations use for "no such object". Vendors differ, hence the sets.
_MISSING_OBJECT_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
#: ...and for "no such *bucket*", which arrives as a 404 too but must never be read as "absent".
_MISSING_BUCKET_CODES = frozenset({"NoSuchBucket"})


class S3ConfigError(RuntimeError):
    """The S3 backend was selected but not (fully) configured."""


def _region_from_endpoint(endpoint: str) -> str:
    """Best-effort region from the endpoint host.

    Vendors that pin a region into the hostname spell it ``s3.<region>.<vendor>.<tld>`` — Backblaze
    (``s3.eu-central-003.backblazeb2.com``) and AWS (``s3.eu-west-1.amazonaws.com``) both do, so the
    common cases need no extra configuration. The dash test rejects the legacy global
    ``s3.amazonaws.com``, whose second label is a vendor name rather than a region. Anything this
    can't read falls back to :data:`DEFAULT_REGION`; set :data:`REGION_ENV` if a vendor is fussy.
    """
    labels = (urlparse(endpoint).hostname or "").split(".")
    if len(labels) >= 3 and labels[0] == "s3" and "-" in labels[1]:
        return labels[1]
    return DEFAULT_REGION


def _required(value: str | None, env: str, missing: list[str]) -> str:
    resolved = (value if value is not None else os.environ.get(env) or "").strip()
    if not resolved:
        missing.append(env)
    return resolved


def _is_missing_object(exc: ClientError) -> bool:
    """Is this "that object was never captured" — as opposed to a real failure?

    The bucket check comes first and deliberately: S3 answers a ``HEAD`` against a **missing
    bucket** with a 404 as well, so a status-only test would turn a typo in ``LAKE_S3_BUCKET`` into
    "every partition is absent" — an empty lake, no error, and a cron that reports success forever.
    A mistyped bucket must raise.
    """
    response = getattr(exc, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    if code in _MISSING_BUCKET_CODES:
        return False
    if code in _MISSING_OBJECT_CODES:
        return True
    return response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404


class _S3RangeReader(io.RawIOBase):
    """A seekable, byte-range view of one object, for pyarrow to read a parquet footer through.

    Handing pyarrow a downloaded buffer would defeat the entire point of the footer: it seeks to the
    end for the metadata, then fetches only the column chunks it was asked for. Serving those seeks
    as ``Range`` requests is what makes ``partition_summary`` and projected reads cheap.

    The tail block is fetched at most once and cached, because footer parsing hits it repeatedly. An
    object smaller than :data:`TAIL_BYTES` is therefore read in exactly one request, whole.
    """

    def __init__(self, client, bucket: str, key: str) -> None:
        super().__init__()
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
        self._tail_start = max(0, self._size - TAIL_BYTES)
        self._tail: bytes | None = None
        self._pos = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_S3RangeReader(s3://{self._bucket}/{self._key}, size={self._size})"

    # -- io plumbing ---------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def size(self) -> int:
        """Object length. pyarrow's ``PythonFile`` asks for this rather than seeking to the end."""
        return self._size

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self._pos + offset
        elif whence == os.SEEK_END:
            target = self._size + offset
        else:
            raise ValueError(f"invalid whence {whence!r}")
        if target < 0:
            raise OSError(f"negative seek position {target}")
        self._pos = target
        return self._pos

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0:
            size = max(0, self._size - self._pos)
        start = min(self._pos, self._size)
        end = min(start + size, self._size)
        if end <= start:
            self._pos = start
            return b""
        if start >= self._tail_start:
            block = self._tail_block()
            chunk = block[start - self._tail_start : end - self._tail_start]
        else:
            # A read spanning the tail boundary is served whole rather than stitched: parquet never
            # issues one, and splitting it would cost an extra request to save nothing.
            chunk = self._range(start, end)
        self._pos = end
        return chunk

    def readall(self) -> bytes:
        return self.read(-1)

    def readinto(self, buffer) -> int:  # noqa: ANN001 - any writable buffer
        chunk = self.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)

    # -- transport -----------------------------------------------------------
    def _tail_block(self) -> bytes:
        if self._tail is None:
            self._tail = self._range(self._tail_start, self._size)
        return self._tail

    def _range(self, start: int, end: int) -> bytes:
        """``[start, end)`` as bytes. ``Range`` is inclusive on both ends, hence ``end - 1``."""
        response = self._client.get_object(
            Bucket=self._bucket, Key=self._key, Range=f"bytes={start}-{end - 1}"
        )
        with closing(response["Body"]) as body:
            return body.read()


class S3Backend:
    """Lake partitions as objects in an S3-compatible bucket. See the module docstring.

    Keys are used verbatim as object keys, so the layout is byte-identical to the local backend and
    ``read_source`` / the assembler never learn which one they are talking to.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        bucket: str | None = None,
        region: str | None = None,
        client=None,  # noqa: ANN001 - a botocore S3 client; injected by tests to observe requests
    ) -> None:
        missing: list[str] = []
        self.endpoint = _required(endpoint, ENDPOINT_ENV, missing)
        self.bucket = _required(bucket, BUCKET_ENV, missing)
        access_key_id = _required(access_key_id, ACCESS_KEY_ENV, missing)
        secret_access_key = _required(secret_access_key, SECRET_KEY_ENV, missing)
        if missing:
            raise S3ConfigError(
                f"LAKE_BACKEND=s3 needs {', '.join(missing)} in the environment. "
                "Set the four LAKE_S3_* values (GitHub Actions secrets in CI, user env locally; "
                "see docs/b2-setup.md), or set LAKE_BACKEND=local to use the local-parquet "
                "dev backend."
            )
        # Validated even when `client` is injected: a half-configured backend is exactly the state
        # this error exists to catch, and a test that skipped the check would not be testing it.
        self.region = (region or os.environ.get(REGION_ENV) or "").strip() or _region_from_endpoint(
            self.endpoint
        )
        self._client = client if client is not None else self._build_client(
            access_key_id, secret_access_key
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"S3Backend(bucket={self.bucket!r}, endpoint={self.endpoint!r})"

    def _build_client(self, access_key_id: str, secret_access_key: str):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=self.region,
            # `standard` retry mode, not botocore's `legacy` default: this runs unattended in a
            # cron, where a transient 5xx should cost a retry rather than a week of lost capture.
            config=Config(retries={"max_attempts": 5, "mode": "standard"}),
        )

    # -- StorageBackend ------------------------------------------------------
    def path_for(self, key: str) -> str:
        """``s3://bucket/key``. A string, not a ``Path`` — ``Path`` collapses the ``//``."""
        return f"s3://{self.bucket}/{key}"

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if _is_missing_object(exc):
                return False
            raise
        return True

    def open_partition(self, key: str) -> pq.ParquetFile:
        return pq.ParquetFile(self._reader(key))

    def read_parquet(self, key: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(
            self._reader(key), columns=list(columns) if columns is not None else None
        )

    def partition_summary(self, key: str) -> tuple[int, str]:
        with self.open_partition(key) as handle:
            return partition_summary_from(handle)

    def write_parquet(self, key: str, df: pd.DataFrame) -> str:
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)  # same writer as local, so the bytes are comparable
        # One PutObject of the whole merged frame: the object flips from old to new in a single
        # step, which is this store's substitute for the local backend's atomic os.replace.
        self._client.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())
        return self.path_for(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        """Every ``.parquet`` key under ``prefix``, treated as a **directory** prefix.

        Two things this must not get wrong, both of which fail silently:

        * ``list_objects_v2`` returns at most 1000 keys per response, so it is paginated. A bare
          call would truncate at 1000 and ``read_source`` would quietly return fewer partitions
          than the lake holds — the lake is 412 partitions today and grows every week.
        * the prefix is a directory, not a string prefix. Without the trailing slash,
          ``list_keys("sleeper_proj")`` would sweep in ``sleeper_proj_week/`` too, which the local
          backend (where it is a real directory lookup) never does.
        """
        trimmed = prefix.strip("/")
        marker = f"{trimmed}/" if trimmed else ""
        keys: list[str] = []
        for page in self._client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=marker
        ):
            keys.extend(
                obj["Key"] for obj in page.get("Contents", ()) if obj["Key"].endswith(".parquet")
            )
        return sorted(keys)

    # -- internals -----------------------------------------------------------
    def _reader(self, key: str) -> _S3RangeReader:
        return _S3RangeReader(self._client, self.bucket, key)


register_backend("s3", S3Backend)

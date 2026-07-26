"""Offline tests for the cloud lake backend (``store.s3``).

**No live S3, ever.** Everything runs against ``moto``'s in-process mock, which intercepts botocore
below the client API and above the socket — so these tests are as offline as the local-parquet ones,
and :func:`test_a_full_round_trip_touches_no_socket` proves it rather than asserting it in a comment.
The endpoint string is a real Backblaze-shaped URL precisely because nothing ever dials it.

Two properties get more attention than the rest, because both fail *silently* in production and
neither is visible locally:

* **cheap metadata** — ``partition_summary`` and projected reads must fetch byte ranges, not
  objects. A backend that quietly materialized each partition would turn ``lake_inventory()`` (412
  partitions today) into a download of the whole lake, and the only symptom would be a slow cron.
* **complete listings** — ``list_objects_v2`` truncates at 1000 keys per response, so an
  unpaginated ``list_keys`` would make ``read_source`` return fewer partitions than exist, with no
  error anywhere.
"""

from __future__ import annotations

import io
import socket
import sys

import boto3
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from store import lake as lake_module
from store.lake import (
    LocalParquetBackend,
    StorageBackend,
    get_backend,
    lake_inventory,
    partition_key,
    read_snapshot,
    read_source,
    set_backend,
    write_snapshot,
)
from store.s3 import (
    DEFAULT_REGION,
    TAIL_BYTES,
    S3Backend,
    S3ConfigError,
    _region_from_endpoint,
)

BUCKET = "ff-lake-test"
#: moto's in-process stubber dispatches on the *hostname*, and only recognizes AWS-shaped ones — a
#: `…backblazeb2.com` endpoint falls straight through to the real network. So the round-trip tests
#: run against an AWS-shaped endpoint (the backend is generic S3 either way; that genericity is the
#: point of the design), and the Backblaze-specific behaviour that *does* depend on the hostname —
#: region derivation — is covered separately by the pure `_region_from_endpoint` tests below.
ENDPOINT = "https://s3.eu-central-1.amazonaws.com"
REGION = "eu-central-1"

SOURCE = "sleeper_proj_week"
DRIFTED = "sleeper_stats_week"
KEY = ("player_id",)
MON = "2026-09-07T12:00:00+00:00"
MON_LATER = "2026-09-07T18:30:00+00:00"
TUE = "2026-09-08T12:00:00+00:00"


# --------------------------------------------------------------------------- harness
class _RecordingClient:
    """Delegates to a botocore S3 client, remembering every call so a test can price it.

    The interesting assertions here are about *transfer*, not about return values: which operations
    ran, and how many bytes each ``Range`` actually asked for.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name.startswith("_") or not callable(attr):
            return attr

        def recorded(*args, **kwargs):
            self.calls.append((name, kwargs))
            return attr(*args, **kwargs)

        return recorded

    @property
    def operations(self) -> list[str]:
        return [name for name, _ in self.calls]

    @property
    def unranged_gets(self) -> list[dict]:
        """Whole-object downloads — the thing every cheap-read test here is guarding against."""
        return [kw for name, kw in self.calls if name == "get_object" and "Range" not in kw]

    @property
    def bytes_requested(self) -> int:
        total = 0
        for name, kwargs in self.calls:
            if name != "get_object":
                continue
            span = kwargs.get("Range")
            if span is None:  # an unranged GET is the entire object
                return 1 << 62
            first, _, last = span.removeprefix("bytes=").partition("-")
            total += int(last) - int(first) + 1
        return total


@pytest.fixture()
def s3_env(monkeypatch):
    """The four production env vars plus a live (mocked) bucket."""
    monkeypatch.setenv("LAKE_S3_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("LAKE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("LAKE_S3_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("LAKE_S3_BUCKET", BUCKET)
    monkeypatch.delenv("LAKE_S3_REGION", raising=False)
    with mock_aws():
        boto3.client("s3", region_name=REGION).create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        yield


@pytest.fixture()
def s3(s3_env):
    return S3Backend()


@pytest.fixture()
def spy(s3_env):
    """An :class:`S3Backend` whose every request is recorded."""
    inner = boto3.client(
        "s3", endpoint_url=ENDPOINT, region_name=REGION,
        aws_access_key_id="test-key-id", aws_secret_access_key="test-secret",
    )
    return S3Backend(client=_RecordingClient(inner))


@pytest.fixture(params=["local", "s3"])
def either(request, tmp_path, s3_env):
    """The same test body against both backends — parity is an acceptance criterion."""
    if request.param == "local":
        return LocalParquetBackend(root=tmp_path)
    return S3Backend()


def _rows(*pairs):
    return [{"player_id": pid, "proj": proj} for pid, proj in pairs]


def _write(backend, rows, *, captured_at=MON, season=2026, week=1, source=SOURCE, key_cols=KEY):
    return write_snapshot(
        source, season, rows, captured_at=captured_at, week=week, key_cols=key_cols,
        backend=backend,
    )


def _big_partition(backend, *, n_rows=40_000):
    """A partition comfortably larger than one tail block, so range reads are measurable.

    Under :data:`TAIL_BYTES` the whole object arrives in a single request, which is *optimal* but
    makes "did it avoid downloading the object" vacuously true. Distinct floats keep parquet from
    compressing the payload away.
    """
    rows = [
        {"player_id": str(i), "proj": i * 0.37, "snaps": i * 1.11, "tgt": i * 2.5}
        for i in range(n_rows)
    ]
    _write(backend, rows, source=DRIFTED, key_cols=("player_id",))
    return partition_key(DRIFTED, 2026, 1)


# --------------------------------------------------------------------------- configuration
def test_backend_conforms_to_the_storage_backend_protocol(s3):
    assert isinstance(s3, StorageBackend)
    # `runtime_checkable` only checks for the *presence* of members, so pin the surface explicitly
    # against the reference implementation too.
    surface = {
        "path_for", "exists", "open_partition", "read_parquet",
        "write_parquet", "list_keys", "partition_summary",
    }
    assert surface <= set(dir(s3))
    assert surface <= set(dir(LocalParquetBackend))


def test_importing_the_module_registers_it_under_s3(s3_env):
    assert lake_module._BACKENDS["s3"] is S3Backend
    assert isinstance(get_backend("s3"), S3Backend)
    set_backend(None)


def test_lake_backend_s3_resolves_from_a_cold_registry(s3_env, monkeypatch):
    """`LAKE_BACKEND=s3` must work with no edit to store/lake.py -- via #1's lazy-import hook.

    Both the module cache and the registry are cleared, so this exercises the real cold path a
    fresh cron process takes rather than a registry some earlier import happened to warm.
    """
    monkeypatch.delitem(sys.modules, "store.s3", raising=False)
    lake_module._BACKENDS.pop("s3", None)
    try:
        resolved = get_backend("s3")
        # A fresh import means a *new* class object, so compare by name rather than identity.
        assert type(resolved).__name__ == "S3Backend"
        assert type(resolved).__module__ == "store.s3"
    finally:
        lake_module._BACKENDS["s3"] = S3Backend
        set_backend(None)


def test_an_unimportable_client_library_is_not_reported_as_an_unknown_backend(monkeypatch):
    """Missing boto3 must say "no module named boto3", not "unknown LAKE_BACKEND 's3'"."""
    monkeypatch.delitem(sys.modules, "store.s3", raising=False)
    monkeypatch.delitem(sys.modules, "boto3", raising=False)
    monkeypatch.setitem(sys.modules, "boto3", None)  # import boto3 -> ImportError
    lake_module._BACKENDS.pop("s3", None)
    try:
        with pytest.raises(ImportError, match="boto3"):
            get_backend("s3")
    finally:
        lake_module._BACKENDS["s3"] = S3Backend
        set_backend(None)


@pytest.mark.parametrize(
    "missing",
    ["LAKE_S3_ENDPOINT", "LAKE_S3_ACCESS_KEY_ID", "LAKE_S3_SECRET_ACCESS_KEY", "LAKE_S3_BUCKET"],
)
def test_missing_credentials_fail_loudly_and_never_fall_back_to_local(s3_env, monkeypatch, missing):
    """A cron that silently wrote to a container-local lake would look healthy and collect nothing."""
    monkeypatch.delenv(missing)
    with pytest.raises(S3ConfigError) as excinfo:
        S3Backend()
    assert missing in str(excinfo.value)
    assert "LAKE_BACKEND=local" in str(excinfo.value)  # says how to opt out deliberately


def test_blank_credentials_are_treated_as_missing(s3_env, monkeypatch):
    monkeypatch.setenv("LAKE_S3_SECRET_ACCESS_KEY", "   ")
    with pytest.raises(S3ConfigError, match="LAKE_S3_SECRET_ACCESS_KEY"):
        S3Backend()


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://s3.eu-central-003.backblazeb2.com", "eu-central-003"),  # Backblaze B2
        ("https://s3.eu-west-1.amazonaws.com", "eu-west-1"),              # AWS regional
        ("https://s3.amazonaws.com", DEFAULT_REGION),                     # AWS legacy global
        ("https://abc123.r2.cloudflarestorage.com", DEFAULT_REGION),      # Cloudflare R2
        ("", DEFAULT_REGION),
    ],
)
def test_region_is_read_off_the_endpoint_when_the_vendor_encodes_it(endpoint, expected):
    assert _region_from_endpoint(endpoint) == expected


def test_an_explicit_region_env_wins(s3_env, monkeypatch):
    monkeypatch.setenv("LAKE_S3_REGION", "us-west-004")
    assert S3Backend().region == "us-west-004"


def test_path_for_is_an_s3_uri_not_a_mangled_path(s3):
    key = partition_key(SOURCE, 2026, 1)
    assert s3.path_for(key) == f"s3://{BUCKET}/{key}"
    # The reason `path_for` is typed `Path | str`: pathlib eats the double slash.
    assert "//" in str(s3.path_for(key))


# --------------------------------------------------------------------------- parity
def test_round_trip_matches_the_local_backend_exactly(either):
    _write(either, _rows(("4046", 21.5), ("6794", 14.0)))
    df = read_snapshot(SOURCE, 2026, 1, backend=either)

    assert df["player_id"].tolist() == ["4046", "6794"]
    assert df["_source"].tolist() == [SOURCE, SOURCE]
    assert df["_season"].tolist() == [2026, 2026]
    assert df["_week"].tolist() == [1, 1]
    assert str(df["_season"].dtype) == "Int64"  # nullable dtypes survive the object store
    assert set(df["_captured_at"]) == {MON}


def test_same_day_recapture_supersedes_on_both_backends(either):
    _write(either, _rows(("4046", 21.5)), captured_at=MON)
    _write(either, _rows(("4046", 23.9)), captured_at=MON_LATER)

    df = read_snapshot(SOURCE, 2026, 1, backend=either)
    assert len(df) == 1
    assert df.loc[0, "proj"] == 23.9
    assert df.loc[0, "_captured_at"] == MON_LATER


def test_later_day_capture_keeps_the_drift_on_both_backends(either):
    _write(either, _rows(("4046", 21.5)), captured_at=MON)
    _write(either, _rows(("4046", 23.9)), captured_at=TUE)

    df = read_snapshot(SOURCE, 2026, 1, backend=either).sort_values("_captured_at")
    assert df["proj"].tolist() == [21.5, 23.9]


def test_a_same_capture_rewrite_is_idempotent_on_both_backends(either):
    rows = _rows(("4046", 21.5), ("6794", 14.0))
    _write(either, rows)
    _write(either, rows)
    assert len(read_snapshot(SOURCE, 2026, 1, backend=either)) == 2


def test_empty_rows_write_nothing_on_both_backends(either):
    _write(either, [])
    assert not either.exists(partition_key(SOURCE, 2026, 1))
    assert read_snapshot(SOURCE, 2026, 1, backend=either).empty


def test_empty_rows_never_blank_an_existing_partition_on_both_backends(either):
    _write(either, _rows(("4046", 21.5)))
    _write(either, [], captured_at=TUE)
    assert len(read_snapshot(SOURCE, 2026, 1, backend=either)) == 1


def test_absent_partitions_read_back_empty_on_both_backends(either):
    assert read_snapshot(SOURCE, 1999, 1, backend=either).empty
    assert read_source("never_collected", backend=either).empty
    assert not either.exists(partition_key(SOURCE, 1999, 1))


def test_read_source_and_inventory_match_the_local_backend(either):
    _write(either, _rows(("4046", 21.5)), season=2026, week=1)
    _write(either, _rows(("4046", 18.0)), season=2026, week=2)
    _write(either, _rows(("4046", 12.0)), season=2025, week=1)

    assert len(read_source(SOURCE, backend=either)) == 3
    assert len(read_source(SOURCE, [2026], backend=either)) == 2

    inv = lake_inventory(backend=either)
    assert inv["source"].tolist() == [SOURCE] * 3
    assert inv["season"].tolist() == [2025, 2026, 2026]
    assert inv["week"].tolist() == [1, 1, 2]
    assert inv["n_rows"].tolist() == [1, 1, 1]
    assert inv["latest_captured_at"].tolist() == [MON] * 3


def test_a_drifted_projection_reads_the_same_on_both_backends(either):
    _write(either, [{"player_id": "4046", "pass_yd": 300}], source=DRIFTED, week=1)
    _write(either, [{"player_id": "9999", "pts_allow_0": 10}], source=DRIFTED, week=2)

    df = read_source(DRIFTED, columns=["player_id", "pass_yd", "pts_allow_0"], backend=either)
    assert df.columns.tolist() == ["player_id", "pass_yd", "pts_allow_0"]
    by_player = df.set_index("player_id")
    assert by_player.loc["4046", "pass_yd"] == 300
    assert pd.isna(by_player.loc["4046", "pts_allow_0"])


# --------------------------------------------------------------------------- listing
def test_list_keys_skips_non_parquet_objects(s3):
    _write(s3, _rows(("4046", 21.5)))
    # Parity with the local backend's temp-file guard: junk in the bucket is not a partition.
    s3._client.put_object(Bucket=BUCKET, Key=f"{SOURCE}/season=2026/notes.txt", Body=b"hi")
    s3._client.put_object(Bucket=BUCKET, Key=f"{SOURCE}/season=2026/x.parquet.tmp", Body=b"junk")

    assert s3.list_keys() == [partition_key(SOURCE, 2026, 1)]
    assert lake_inventory(backend=s3)["source"].tolist() == [SOURCE]


def test_list_keys_treats_the_prefix_as_a_directory(s3):
    """`sleeper_proj` must not sweep in `sleeper_proj_week/` -- the local backend never would."""
    _write(s3, _rows(("4046", 21.5)), source="sleeper_proj_week")
    _write(s3, _rows(("4046", 260.0)), source="sleeper_proj_season", week=None)

    assert s3.list_keys("sleeper_proj") == []
    assert s3.list_keys("sleeper_proj_week") == [partition_key("sleeper_proj_week", 2026, 1)]
    assert s3.list_keys("sleeper_proj_week/") == [partition_key("sleeper_proj_week", 2026, 1)]
    assert len(s3.list_keys()) == 2


def test_list_keys_pages_past_the_thousand_key_response_cap(s3):
    """`list_objects_v2` returns at most 1000 keys; truncating here loses partitions in silence."""
    total = 1005
    for i in range(total):
        s3._client.put_object(Bucket=BUCKET, Key=f"{DRIFTED}/season=2026/p{i:05d}.parquet", Body=b"")

    keys = s3.list_keys(DRIFTED)
    assert len(keys) == total
    assert keys == sorted(keys)


# --------------------------------------------------------------------------- cheap metadata
def test_partition_summary_never_downloads_the_object(spy):
    key = _big_partition(spy)
    size = spy._client.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    assert size > TAIL_BYTES, "fixture must exceed one tail block or the assertion is vacuous"
    spy._client.calls.clear()

    n_rows, latest = spy.partition_summary(key)

    assert (n_rows, latest) == (40_000, MON)
    assert spy._client.unranged_gets == []
    # Footer tail plus one (highly compressible, constant-valued) `_captured_at` chunk.
    assert spy._client.bytes_requested < size // 2


def test_lake_inventory_never_downloads_the_lake(spy):
    key = _big_partition(spy)
    size = spy._client.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    spy._client.calls.clear()

    inv = lake_inventory(backend=spy)

    assert inv["n_rows"].tolist() == [40_000]
    assert spy._client.unranged_gets == []
    assert spy._client.bytes_requested < size // 2


def test_a_projected_read_transfers_less_than_the_object(spy):
    key = _big_partition(spy)
    size = spy._client.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    spy._client.calls.clear()

    df = read_snapshot(DRIFTED, 2026, 1, columns=["player_id", "proj"], backend=spy)

    assert df.columns.tolist() == ["player_id", "proj"]
    assert len(df) == 40_000
    assert spy._client.unranged_gets == []
    assert spy._client.bytes_requested < size


def test_a_drifted_projection_costs_one_open_not_a_download(spy):
    """The #19 finding, priced on the backend it actually mattered for.

    All 175 real ``sleeper_stats_week`` partitions lack some column the assembler asks for. Under
    the old try/except projection that was 175 whole-object downloads plus 175 wasted attempts.
    """
    key = _big_partition(spy)
    size = spy._client.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    spy._client.calls.clear()

    df = read_snapshot(DRIFTED, 2026, 1, columns=["player_id", "pts_allow_0"], backend=spy)

    assert df.columns.tolist() == ["player_id", "pts_allow_0"]
    assert df["pts_allow_0"].isna().all()
    assert spy._client.unranged_gets == []
    assert spy._client.bytes_requested < size
    # Exactly two HEADs: `read_snapshot`'s existence check, and the one reader that serves both the
    # schema lookup and the payload read. A third would mean the projection re-opened the object to
    # ask which columns it has -- a round-trip per partition, 412 of them today.
    assert spy._client.operations.count("head_object") == 2


# --------------------------------------------------------------------------- writes
def test_a_write_is_a_single_put_object(spy):
    """Object stores have no rename, so "atomic" is one PutObject of the fully merged frame."""
    spy._client.calls.clear()
    _write(spy, _rows(("4046", 21.5)))

    puts = [name for name in spy._client.operations if name.startswith(("put_", "create_", "upload"))]
    assert puts == ["put_object"]
    assert "complete_multipart_upload" not in spy._client.operations


def test_a_merging_write_still_ends_in_one_put_object(spy):
    _write(spy, _rows(("4046", 21.5)), captured_at=MON)
    spy._client.calls.clear()
    _write(spy, _rows(("6794", 14.0)), captured_at=TUE)

    assert [n for n in spy._client.operations if n == "put_object"] == ["put_object"]
    assert len(read_snapshot(SOURCE, 2026, 1, backend=spy)) == 2


def test_write_snapshot_returns_the_s3_uri(s3):
    written = _write(s3, _rows(("4046", 21.5)))
    assert written == f"s3://{BUCKET}/{partition_key(SOURCE, 2026, 1)}"


# --------------------------------------------------------------------------- reader internals
def test_the_range_reader_serves_seeks_and_reads_like_a_file(s3):
    key = partition_key(SOURCE, 2026, 1)
    _write(s3, _rows(("4046", 21.5)))
    raw = s3._client.get_object(Bucket=BUCKET, Key=key)["Body"].read()

    reader = s3._reader(key)
    assert reader.size() == len(raw)
    assert reader.readable() and reader.seekable() and not reader.writable()

    assert reader.read(4) == raw[:4] == b"PAR1"
    assert reader.tell() == 4
    assert reader.seek(-4, io.SEEK_END) == len(raw) - 4
    assert reader.read() == raw[-4:] == b"PAR1"
    assert reader.read() == b""  # at EOF

    reader.seek(0)
    assert reader.readall() == raw
    with pytest.raises(OSError):
        reader.seek(-1)


def test_an_object_smaller_than_a_tail_block_is_fetched_once(spy):
    key = partition_key(SOURCE, 2026, 1)
    _write(spy, _rows(("4046", 21.5)))
    spy._client.calls.clear()

    with spy.open_partition(key) as handle:
        assert handle.metadata.num_rows == 1

    # Small partitions are the common case; splitting them into footer + column reads would cost
    # more requests than it saves, so the cached tail block covers the whole object.
    assert spy._client.operations.count("get_object") == 1
    assert spy._client.unranged_gets == []


def test_exists_reports_an_uncaptured_partition_as_absent(s3):
    key = partition_key(SOURCE, 2026, 1)
    assert s3.exists(key) is False
    _write(s3, _rows(("4046", 21.5)))
    assert s3.exists(key) is True


def test_a_mistyped_bucket_raises_instead_of_looking_like_an_empty_lake(s3_env):
    """S3 answers HEAD-on-a-missing-bucket with a 404 too, and conflating the two is disastrous:
    every partition would read as "not captured yet", the crons would report success every week,
    and the lake would silently stay empty."""
    typo = S3Backend(bucket="ff-lake-tset")
    with pytest.raises(ClientError, match="NoSuchBucket|does not exist"):
        typo.exists(partition_key(SOURCE, 2026, 1))
    with pytest.raises(ClientError, match="NoSuchBucket|does not exist"):
        write_snapshot(SOURCE, 2026, _rows(("4046", 21.5)), captured_at=MON, week=1,
                       key_cols=KEY, backend=typo)


# --------------------------------------------------------------------------- offline guarantee
def test_a_full_round_trip_touches_no_socket(s3, monkeypatch):
    """CI must never reach a real bucket; moto intercepts below botocore, above the network."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the S3 backend opened a socket — these tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)

    _write(s3, _rows(("4046", 21.5)))
    assert len(read_snapshot(SOURCE, 2026, 1, backend=s3)) == 1
    assert s3.list_keys() == [partition_key(SOURCE, 2026, 1)]
    assert lake_inventory(backend=s3)["n_rows"].tolist() == [1]

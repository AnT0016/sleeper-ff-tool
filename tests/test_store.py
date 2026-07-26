"""Offline unit tests for the Phase 8 lake storage layer (``store.lake``).

Everything here runs against a ``LocalParquetBackend`` rooted in ``tmp_path`` — no network, and the
repo's real ``data_cache/lake/`` is never touched. The point-in-time behaviour (same-day capture is
idempotent, later-day capture is retained) is the contract the whole phase depends on, so it is
tested from both directions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from store.lake import (
    LAKE_ROOT,
    RESERVED,
    TMP_SUFFIX,
    LocalParquetBackend,
    get_backend,
    lake_inventory,
    normalize_captured_at,
    partition_key,
    read_snapshot,
    read_source,
    set_backend,
    snapshot_path,
    write_snapshot,
)

SOURCE = "sleeper_proj_week"
KEY = ("player_id",)
MON = "2026-09-07T12:00:00+00:00"
MON_LATER = "2026-09-07T18:30:00+00:00"
TUE = "2026-09-08T12:00:00+00:00"


@pytest.fixture()
def backend(tmp_path):
    return LocalParquetBackend(root=tmp_path)


def _rows(*pairs):
    return [{"player_id": pid, "proj": proj} for pid, proj in pairs]


def _write(backend, rows, *, captured_at=MON, season=2026, week=1, source=SOURCE, key_cols=KEY):
    return write_snapshot(
        source, season, rows, captured_at=captured_at, week=week, key_cols=key_cols,
        backend=backend,
    )


# --------------------------------------------------------------------------- paths
def test_partition_key_week_and_season_layout():
    assert partition_key(SOURCE, 2026, 3) == f"{SOURCE}/season=2026/{SOURCE}_2026_wk03.parquet"
    assert partition_key(SOURCE, 2026) == f"{SOURCE}/season=2026/{SOURCE}_2026_season.parquet"


def test_snapshot_path_defaults_to_the_local_lake_root():
    assert snapshot_path(SOURCE, 2026, 3) == LAKE_ROOT / SOURCE / "season=2026" / (
        f"{SOURCE}_2026_wk03.parquet"
    )


# --------------------------------------------------------------------------- round trip
def test_round_trip_populates_every_reserved_column(backend):
    _write(backend, _rows(("4046", 21.5), ("6794", 14.0)))
    df = read_snapshot(SOURCE, 2026, 1, backend=backend)

    assert list(df["player_id"]) == ["4046", "6794"]
    assert list(df.columns[-4:]) == list(RESERVED)  # payload first, provenance last
    assert set(df["_source"]) == {SOURCE}
    assert set(df["_season"]) == {2026}
    assert set(df["_week"]) == {1}

    stamps = pd.to_datetime(df["_captured_at"], utc=True, format="ISO8601")
    assert str(stamps.dt.tz) == "UTC"
    assert set(stamps.dt.strftime("%Y-%m-%dT%H:%M:%S%z")) == {"2026-09-07T12:00:00+0000"}


def test_season_partition_leaves_week_null(backend):
    write_snapshot(
        "sleeper_proj_season", 2026, _rows(("4046", 260.0)),
        captured_at=MON, key_cols=KEY, backend=backend,
    )
    df = read_snapshot("sleeper_proj_season", 2026, backend=backend)
    assert df["_week"].isna().all()
    assert df["_season"].tolist() == [2026]


def test_read_snapshot_of_an_absent_partition_is_empty_but_shaped(backend):
    df = read_snapshot(SOURCE, 1999, 1, backend=backend)
    assert df.empty
    assert list(df.columns) == list(RESERVED)


# --------------------------------------------------------------------------- point-in-time dedup
def test_same_capture_rewrite_is_idempotent(backend):
    rows = _rows(("4046", 21.5), ("6794", 14.0))
    path = _write(backend, rows)
    first_bytes = path.read_bytes()

    _write(backend, rows)
    df = read_snapshot(SOURCE, 2026, 1, backend=backend)

    assert len(df) == 2  # no duplicate rows
    assert path.read_bytes() == first_bytes


def test_same_day_recapture_keeps_only_the_latest(backend):
    _write(backend, _rows(("4046", 21.5)), captured_at=MON)
    _write(backend, _rows(("4046", 23.9)), captured_at=MON_LATER)

    df = read_snapshot(SOURCE, 2026, 1, backend=backend)
    assert len(df) == 1
    assert df.loc[0, "proj"] == 23.9
    assert df.loc[0, "_captured_at"] == MON_LATER


def test_later_date_capture_keeps_both_snapshots(backend):
    _write(backend, _rows(("4046", 21.5)), captured_at=MON)
    _write(backend, _rows(("4046", 23.9)), captured_at=TUE)

    df = read_snapshot(SOURCE, 2026, 1, backend=backend).sort_values("_captured_at")
    assert len(df) == 2  # drift across days is preserved -- that is the whole point
    assert df["proj"].tolist() == [21.5, 23.9]
    assert df["_captured_at"].tolist() == [MON, TUE]


def test_dedup_is_per_key_not_across_keys(backend):
    _write(backend, _rows(("4046", 21.5), ("6794", 14.0)), captured_at=MON)
    _write(backend, _rows(("4046", 23.9), ("6794", 15.5)), captured_at=MON_LATER)

    df = read_snapshot(SOURCE, 2026, 1, backend=backend)
    assert len(df) == 2
    assert dict(zip(df["player_id"], df["proj"])) == {"4046": 23.9, "6794": 15.5}


def test_captured_at_is_normalized_to_utc(backend):
    # 01:30 in Paris is the previous UTC day -- the capture date must follow the instant, not the
    # wall clock of whoever ran it.
    _write(backend, _rows(("4046", 21.5)), captured_at="2026-09-08T01:30:00+02:00")
    df = read_snapshot(SOURCE, 2026, 1, backend=backend)
    assert df.loc[0, "_captured_at"] == "2026-09-07T23:30:00+00:00"


def test_normalize_captured_at_accepts_z_and_naive_and_rejects_junk():
    assert normalize_captured_at("2026-09-07T12:00:00Z") == MON
    assert normalize_captured_at("2026-09-07T12:00:00") == MON  # naive is read as UTC
    for bad in ("", "   ", "not-a-date", None):
        with pytest.raises(ValueError):
            normalize_captured_at(bad)


# --------------------------------------------------------------------------- guard rails
def test_empty_rows_writes_nothing(backend):
    path = _write(backend, [])
    assert not path.exists()
    assert read_snapshot(SOURCE, 2026, 1, backend=backend).empty


def test_empty_rows_never_blanks_an_existing_partition(backend):
    _write(backend, _rows(("4046", 21.5)))
    _write(backend, [], captured_at=TUE)
    assert len(read_snapshot(SOURCE, 2026, 1, backend=backend)) == 1


def test_missing_key_column_raises(backend):
    with pytest.raises(ValueError, match="absent from the collected rows"):
        _write(backend, [{"gsis_id": "00-0031234", "proj": 9.0}])


def test_rows_may_not_forge_provenance(backend):
    with pytest.raises(ValueError, match="reserved column"):
        _write(backend, [{"player_id": "4046", "_captured_at": "1999-01-01T00:00:00+00:00"}])


def test_key_cols_must_be_non_empty_and_unreserved(backend):
    with pytest.raises(ValueError, match="non-empty"):
        _write(backend, _rows(("4046", 1.0)), key_cols=())
    with pytest.raises(ValueError, match="exclude reserved"):
        _write(backend, _rows(("4046", 1.0)), key_cols=("player_id", "_season"))


# --------------------------------------------------------------------------- atomicity
def test_a_stray_temp_file_never_corrupts_the_partition(backend):
    path = _write(backend, _rows(("4046", 21.5)))
    # Simulate a process killed mid-write: a temp file left next to the committed partition.
    (path.parent / f"{path.name}.abc123{TMP_SUFFIX}").write_bytes(b"garbage-not-parquet")

    assert len(read_snapshot(SOURCE, 2026, 1, backend=backend)) == 1
    assert backend.list_keys() == [partition_key(SOURCE, 2026, 1)]  # listers skip temp files
    assert len(lake_inventory(backend=backend)) == 1

    _write(backend, _rows(("6794", 14.0)), captured_at=TUE)  # and a later write still succeeds
    assert len(read_snapshot(SOURCE, 2026, 1, backend=backend)) == 2


def test_a_failed_write_leaves_the_committed_partition_intact(backend, monkeypatch):
    path = _write(backend, _rows(("4046", 21.5)))
    good = path.read_bytes()

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        _write(backend, _rows(("6794", 14.0)), captured_at=TUE)
    monkeypatch.undo()

    assert path.read_bytes() == good
    leftovers = [p.name for p in path.parent.iterdir() if p.name.endswith(TMP_SUFFIX)]
    assert leftovers == []
    assert len(read_snapshot(SOURCE, 2026, 1, backend=backend)) == 1


# --------------------------------------------------------------------------- multi-partition reads
def test_read_source_concats_partitions_and_filters_seasons(backend):
    _write(backend, _rows(("4046", 21.5)), season=2026, week=1)
    _write(backend, _rows(("4046", 18.0)), season=2026, week=2)
    _write(backend, _rows(("4046", 12.0)), season=2025, week=1)

    everything = read_source(SOURCE, backend=backend)
    assert len(everything) == 3
    assert sorted(everything["_season"].unique()) == [2025, 2026]

    only_2026 = read_source(SOURCE, [2026], backend=backend)
    assert len(only_2026) == 2
    assert set(only_2026["_week"]) == {1, 2}

    assert read_source("never_collected", backend=backend).empty


def test_lake_inventory_reports_one_row_per_partition(backend):
    _write(backend, _rows(("4046", 21.5), ("6794", 14.0)), season=2026, week=1)
    _write(backend, _rows(("4046", 22.0)), season=2026, week=1, captured_at=TUE)
    write_snapshot(
        "sleeper_proj_season", 2026, _rows(("4046", 260.0)),
        captured_at=MON, key_cols=KEY, backend=backend,
    )

    inv = lake_inventory(backend=backend)
    assert inv["source"].tolist() == ["sleeper_proj_season", "sleeper_proj_week"]

    season_row = inv[inv["source"] == "sleeper_proj_season"].iloc[0]
    assert pd.isna(season_row["week"]) and season_row["n_rows"] == 1

    week_row = inv[inv["source"] == "sleeper_proj_week"].iloc[0]
    assert week_row["week"] == 1 and week_row["season"] == 2026
    assert week_row["n_rows"] == 3  # 2 from Monday + 1 Tuesday re-capture (drift kept)
    assert week_row["latest_captured_at"] == TUE
    assert week_row["path"].endswith("sleeper_proj_week_2026_wk01.parquet")


def test_lake_inventory_of_an_empty_lake_has_the_right_columns(backend):
    inv = lake_inventory(backend=backend)
    assert inv.empty
    assert inv.columns.tolist() == [
        "source", "season", "week", "n_rows", "path", "latest_captured_at",
    ]


# --------------------------------------------------------------------------- backend selection
def test_default_backend_is_local_parquet():
    try:
        assert isinstance(get_backend(), LocalParquetBackend)
        assert get_backend().root == LAKE_ROOT
    finally:
        set_backend(None)


def test_unknown_backend_names_fail_loudly():
    with pytest.raises(ValueError, match="unknown LAKE_BACKEND"):
        get_backend("dropbox")

"""Offline unit tests for the Phase 8 lake storage layer (``store.lake``).

Everything here runs against a ``LocalParquetBackend`` rooted in ``tmp_path`` — no network, and the
repo's real ``data_cache/lake/`` is never touched. The point-in-time behaviour (same-day capture is
idempotent, later-day capture is retained) is the contract the whole phase depends on, so it is
tested from both directions.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from collect.registry import SOURCES
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


# --------------------------------------------------------------------------- capture integrity
# _dedup collapses on key_cols + capture date, and pandas treats NaN == NaN, so a key that doesn't
# actually identify a capture's rows destroys data silently. Superseding an *earlier capture* is the
# intended rule; loss *within* one capture is a bug and must be audible.
INJURY_REVISIONS = [
    {"gsis_id": "00-0039359", "game_type": "REG", "week": 15,
     "date_modified": "2024-12-15 03:34", "report_status": "Questionable"},
    {"gsis_id": "00-0039359", "game_type": "REG", "week": 15,
     "date_modified": "2024-12-15 14:17", "report_status": "Out"},
]


def test_rows_one_capture_cannot_separate_are_dropped_but_never_silently(backend, caplog):
    """Pins the *class* of bug on the real 2024 W15 shape: two same-week revisions of one player.

    The registry key is the player-week (#17), so this pair is not separable within one capture and
    the store keeps one of them. That is correct behaviour *given these rows* — and the reason
    ``collect.nflverse._latest_revision`` resolves the collision to the newest revision before the
    store ever sees it. What must never happen is the drop being quiet.
    """
    with caplog.at_level(logging.WARNING, logger="store.lake"):
        write_snapshot(
            "nflverse_injuries", 2024, INJURY_REVISIONS, captured_at=MON,
            key_cols=SOURCES["nflverse_injuries"].key_cols, backend=backend,
        )

    assert len(read_snapshot("nflverse_injuries", 2024, backend=backend)) == 1  # the loss
    assert "do not identify rows within one capture" in caplog.text            # now audible
    assert "00-0039359" in caplog.text                                         # names the offender


def test_a_player_weeks_status_change_survives_as_two_captures_on_different_days(backend):
    """What replaced ``date_modified``: the revision stream is the *capture* stream.

    Pre-lock runs Thursday and Sunday, and the store keeps a row per key per UTC capture date. So
    Thursday's Questionable and Sunday's Out are two rows — stamped with when *we* observed them
    rather than when the provider edited the report, which is the more useful point-in-time fact and
    the one #7 can actually gate on.
    """
    key_cols = SOURCES["nflverse_injuries"].key_cols
    thursday, sunday = INJURY_REVISIONS
    write_snapshot("nflverse_injuries", 2024, [thursday], captured_at="2024-12-12T22:00:00+00:00",
                   key_cols=key_cols, backend=backend)
    write_snapshot("nflverse_injuries", 2024, [sunday], captured_at="2024-12-15T15:00:00+00:00",
                   key_cols=key_cols, backend=backend)

    df = read_snapshot("nflverse_injuries", 2024, backend=backend)
    assert len(df) == 2
    assert set(df["report_status"]) == {"Questionable", "Out"}
    # And they are told apart by capture date, not by anything in the payload.
    assert dict(zip(df["_captured_at"].str[:10], df["report_status"], strict=True)) == {
        "2024-12-12": "Questionable",
        "2024-12-15": "Out",
    }


def test_null_key_values_warn(backend, caplog):
    with caplog.at_level(logging.WARNING, logger="store.lake"):
        _write(backend, [{"player_id": None, "proj": 1.0}, {"player_id": None, "proj": 2.0}])
    assert "null in key column" in caplog.text


def test_intended_point_in_time_dedup_never_warns(backend, caplog):
    """Guards against a false positive: cross-day supersede is the design, not a defect."""
    _write(backend, _rows(("4046", 21.5)), captured_at=MON)
    with caplog.at_level(logging.WARNING, logger="store.lake"):
        _write(backend, _rows(("4046", 23.9)), captured_at=TUE)   # later day: both kept
        _write(backend, _rows(("4046", 24.1)), captured_at=MON_LATER)  # same day: supersedes
    assert caplog.text == ""


# --------------------------------------------------------------------------- first_capture policy (#15)
# nflverse_depth is keyed on its own observation timestamp (dt), so a row is immutable and
# re-capturing the cumulative feed on a later date records nothing new. Its declared "first_capture"
# policy keeps one row per key however many times it is captured, where the per-capture-date default
# would multiply the partition by the number of pre-lock runs (36/season) -- the #15 defect.
DEPTH = "nflverse_depth"
DEPTH_KEYS = SOURCES[DEPTH].key_cols  # ("dt", "team", "gsis_id", "pos_abb")
WED = "2025-09-10T12:00:00+00:00"
FRI = "2025-09-12T12:00:00+00:00"
SUN = "2025-09-14T12:00:00+00:00"


def _depth(dt, gsis, pos, rank):
    return {"dt": dt, "team": "KC", "gsis_id": gsis, "pos_abb": pos, "pos_rank": rank}


def _write_depth(backend, rows, *, captured_at, **kwargs):
    return write_snapshot(
        DEPTH, 2025, rows, captured_at=captured_at, key_cols=DEPTH_KEYS, backend=backend, **kwargs
    )


def test_first_capture_is_resolved_from_the_registry_with_no_explicit_argument(backend):
    """The whole point of resolving inside write_snapshot: a caller that passes no ``dedup`` (the
    runner, and every assembler-test fixture) still gets the source's declared policy."""
    row = _depth("2025-09-08T07:00:00Z", "00-0001", "RB", 1)
    _write_depth(backend, [row], captured_at=WED)
    _write_depth(backend, [row], captured_at=SUN)  # later day, immutable key -> nothing new

    df = read_snapshot(DEPTH, 2025, backend=backend)
    assert len(df) == 1                          # NOT two, as the per-capture-date default would give
    assert df.loc[0, "_captured_at"] == WED      # the earliest observation survives


def test_first_capture_holds_the_distinct_key_count_across_many_captures(backend):
    """#15's headline acceptance criterion: N captures of a cumulative feed on N dates must leave the
    partition holding the distinct-key count, not N x it."""
    feed = [
        _depth("2025-09-07T07:00:00Z", "00-0001", "RB", 1),
        _depth("2025-09-07T07:00:00Z", "00-0002", "WR", 1),
        _depth("2025-09-08T07:00:00Z", "00-0001", "RB", 1),  # a later snapshot of the same player
    ]
    for captured_at in (WED, FRI, SUN):
        _write_depth(backend, feed, captured_at=captured_at)

    df = read_snapshot(DEPTH, 2025, backend=backend)
    assert len(df) == 3                                  # 3 distinct keys, not 3 captures x 3 rows
    assert set(df["_captured_at"]) == {WED}              # every survivor is the first observation


def test_first_capture_keeps_the_earliest_row_even_when_a_later_capture_differs(backend):
    """Teeth the row-count test lacks: earliest wins *by capture*, not by coincidence. A later
    capture carrying revised payload for the same immutable key must not overwrite the first."""
    key_row = _depth("2025-09-08T07:00:00Z", "00-0001", "RB", 1)
    _write_depth(backend, [key_row], captured_at=WED)
    _write_depth(backend, [{**key_row, "pos_rank": 9}], captured_at=SUN)  # same key, revised rank

    df = read_snapshot(DEPTH, 2025, backend=backend)
    assert len(df) == 1
    assert df.loc[0, "pos_rank"] == 1            # the first capture, not the later revision
    assert df.loc[0, "_captured_at"] == WED


def test_first_capture_same_day_rerun_is_byte_idempotent(backend):
    row = _depth("2025-09-08T07:00:00Z", "00-0001", "RB", 1)
    path = _write_depth(backend, [row], captured_at=WED)
    first_bytes = path.read_bytes()
    _write_depth(backend, [row], captured_at=WED)
    assert path.read_bytes() == first_bytes


def test_the_per_capture_date_default_is_untouched_by_the_new_policy(backend):
    """The default path must be behaviourally identical to before #15: a later-day capture of a
    revisable source is still retained as a new point-in-time snapshot."""
    _write(backend, _rows(("4046", 21.5)), captured_at=MON)
    _write(backend, _rows(("4046", 23.9)), captured_at=TUE)
    df = read_snapshot(SOURCE, 2026, 1, backend=backend)
    assert len(df) == 2  # drift preserved -- same contract as test_later_date_capture_keeps_both


def test_an_explicit_dedup_override_wins_over_the_registry(backend):
    """An explicit policy is the documented escape hatch for tests/one-offs, and it overrides the
    source's declaration -- here forcing depth back onto the multiplying per-capture-date path."""
    row = _depth("2025-09-08T07:00:00Z", "00-0001", "RB", 1)
    _write_depth(backend, [row], captured_at=WED, dedup="per_capture_date")
    _write_depth(backend, [row], captured_at=SUN, dedup="per_capture_date")
    assert len(read_snapshot(DEPTH, 2025, backend=backend)) == 2  # both kept, unlike first_capture


def test_an_unknown_explicit_dedup_policy_is_rejected(backend):
    with pytest.raises(ValueError, match="unknown dedup policy"):
        write_snapshot(
            SOURCE, 2026, _rows(("4046", 1.0)), captured_at=MON, week=1, key_cols=KEY,
            dedup="keep_everything", backend=backend,
        )


def test_the_reconciliation_log_names_the_side_dropped_per_policy(backend, caplog):
    """The reconciliation line flips meaning with the policy and must say which side was dropped.
    Under ``first_capture`` the *fresh re-observation* is dropped and the stored row kept, so the
    per-capture-date word "superseded" would read as the exact opposite of what happened -- and this
    line exists precisely so a cron log does not hide which row was lost."""
    depth_row = _depth("2025-09-08T07:00:00Z", "00-0001", "RB", 1)
    _write_depth(backend, [depth_row], captured_at=WED)
    with caplog.at_level(logging.INFO, logger="store.lake"):
        _write_depth(backend, [depth_row], captured_at=SUN)  # later day, immutable key
    assert "1 re-observations dropped" in caplog.text
    assert "superseded" not in caplog.text

    caplog.clear()
    _write(backend, _rows(("4046", 21.5)), captured_at=MON)
    with caplog.at_level(logging.INFO, logger="store.lake"):
        _write(backend, _rows(("4046", 23.9)), captured_at=MON_LATER)  # same day supersede
    assert "1 superseded" in caplog.text
    assert "re-observations dropped" not in caplog.text


# --------------------------------------------------------------------------- cheap inventory
class _SpyBackend(LocalParquetBackend):
    """Records every materializing read so a test can prove what was actually fetched.

    ``reads`` is the payload path (:meth:`read_parquet`); ``opens`` is the footer path
    (:meth:`open_partition`). The distinction is the whole point of both cheap-read tests below.
    """

    def __init__(self, root):
        super().__init__(root)
        self.reads: list[tuple[str, tuple[str, ...] | None]] = []
        self.opens: list[str] = []

    def read_parquet(self, key, columns=None):
        self.reads.append((key, tuple(columns) if columns is not None else None))
        return super().read_parquet(key, columns)

    def open_partition(self, key):
        self.opens.append(key)
        return super().open_partition(key)


def test_partition_summary_reports_rows_and_latest_stamp(backend):
    _write(backend, _rows(("4046", 21.5)), captured_at=MON)
    _write(backend, _rows(("6794", 14.0)), captured_at=TUE)
    assert backend.partition_summary(partition_key(SOURCE, 2026, 1)) == (2, TUE)


def test_read_parquet_projects_columns(backend):
    _write(backend, _rows(("4046", 21.5)))
    df = backend.read_parquet(partition_key(SOURCE, 2026, 1), columns=["player_id"])
    assert df.columns.tolist() == ["player_id"]


def test_lake_inventory_never_fetches_payload_columns(tmp_path):
    """On a cloud backend, answering this by materializing partitions downloads the whole lake."""
    spy = _SpyBackend(tmp_path)
    _write(spy, _rows(("4046", 21.5), ("6794", 14.0)))
    spy.reads.clear()

    inv = lake_inventory(backend=spy)

    assert inv.loc[0, "n_rows"] == 2
    assert inv.loc[0, "latest_captured_at"] == MON
    assert spy.reads == []  # footer metadata + one column chunk, never a materialized frame


# --------------------------------------------------------------------------- projected reads
# A raw layer's schema drifts: ``sleeper_stats_week`` writes only the stat keys a week actually
# produced, so a week in which no defence pitched a shutout has no ``pts_allow_0`` column at all.
# The projection has to survive that *without* degrading into a full read -- measured on the real
# lake, all 175 sleeper_stats_week partitions lack at least one column the assembler asks for
# (median 64, max 79), so a try/except-on-failure projection was 175 full reads plus 175 wasted
# attempts. Locally that is noise; on S3 it is the entire transfer the projection exists to avoid.
DRIFTED = "sleeper_stats_week"


def test_a_projected_read_of_a_drifted_partition_never_falls_back_to_a_full_read(tmp_path):
    spy = _SpyBackend(tmp_path)
    _write(spy, [{"player_id": "4046", "pass_yd": 300}], source=DRIFTED)
    spy.reads.clear()
    spy.opens.clear()

    df = read_snapshot(
        DRIFTED, 2026, 1, columns=["player_id", "pts_allow_0"], backend=spy,
    )

    # The requested shape, with the absent column reindexed to NA rather than raising.
    assert df.columns.tolist() == ["player_id", "pts_allow_0"]
    assert df["player_id"].tolist() == ["4046"]
    assert df["pts_allow_0"].isna().all()

    # ...and it cost no unprojected read. This is the assertion that fails on a try/except design.
    assert [key for key, columns in spy.reads if columns is None] == []
    # One footer resolution, not one per attempt: `column_names` and the projected read must share
    # a handle, or an S3 backend pays two round-trips per partition to save one.
    assert spy.opens == [partition_key(DRIFTED, 2026, 1)]


def test_a_projection_that_matches_the_partition_exactly_still_projects(tmp_path):
    """Guards the happy path against a fix that "solves" drift by always reading everything."""
    spy = _SpyBackend(tmp_path)
    _write(spy, [{"player_id": "4046", "pass_yd": 300, "rec": 4.0}], source=DRIFTED)
    spy.reads.clear()

    df = read_snapshot(DRIFTED, 2026, 1, columns=["player_id", "rec"], backend=spy)

    assert df.columns.tolist() == ["player_id", "rec"]
    assert [key for key, columns in spy.reads if columns is None] == []


def test_read_source_projects_across_partitions_of_differing_schemas(backend):
    """Ten seasons of a sparse feed: the union of columns is never present in any one partition."""
    _write(backend, [{"player_id": "4046", "pass_yd": 300}], source=DRIFTED, week=1)
    _write(backend, [{"player_id": "9999", "pts_allow_0": 10}], source=DRIFTED, week=2)

    df = read_source(DRIFTED, columns=["player_id", "pass_yd", "pts_allow_0"], backend=backend)

    assert df.columns.tolist() == ["player_id", "pass_yd", "pts_allow_0"]
    assert len(df) == 2
    by_player = df.set_index("player_id")
    assert by_player.loc["4046", "pass_yd"] == 300 and pd.isna(by_player.loc["9999", "pass_yd"])
    assert by_player.loc["9999", "pts_allow_0"] == 10 and pd.isna(
        by_player.loc["4046", "pts_allow_0"]
    )


def test_a_projection_of_columns_none_of_which_exist_keeps_the_row_count(backend):
    """Degenerate but reachable: every requested column absent must still yield N rows of NA."""
    _write(backend, [{"player_id": "4046"}, {"player_id": "6794"}], source=DRIFTED)

    df = read_snapshot(DRIFTED, 2026, 1, columns=["fgm_50p", "xpm"], backend=backend)

    assert df.columns.tolist() == ["fgm_50p", "xpm"]
    assert len(df) == 2 and df.isna().all().all()


def test_open_partition_exposes_the_footer_without_a_payload_read(backend):
    _write(backend, _rows(("4046", 21.5), ("6794", 14.0)))
    with backend.open_partition(partition_key(SOURCE, 2026, 1)) as handle:
        assert handle.metadata.num_rows == 2
        assert handle.schema_arrow.names == ["player_id", "proj", *RESERVED]


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

"""Offline unit tests for the nflverse collectors (``collect.nflverse``).

Every test runs against the parquet samples in ``tests/fixtures/nflverse/`` — real rows cut from the
nflverse releases by ``scripts/make_nflverse_fixture.py``, kept as parquet so the provider's *dtypes*
survive into the test. That matters more here than for the Sleeper fixtures: ``date_modified`` is a
tz-aware ``Datetime`` **inside the key**, ``ff_opportunity`` types ``season`` as a string and
``week`` as a float, and a JSON round-trip would hand these tests a tidied frame nflverse never
produces.

The samples were chosen for the shapes that break things, and the tests below are mostly about those:
the residual rows nflverse files per team-game with a null ``player_id``, an injury player-week
carrying two report revisions (the entire reason ``date_modified`` is in the key), a null-``gsis_id``
depth-chart collision, and the pre-2025 depth-chart schema that the registry key cannot address.

The property that gets the most attention is the one the acceptance criteria turn on: a capture must
reach ``store.write_snapshot`` with **zero** capture-integrity warnings, because both of the shapes
that provoke them (a null key value, a key repeated within one capture) delete real rows silently.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from collect.nflverse import (
    _iso_utc,
    collect_depth_charts,
    collect_ff_opportunity,
    collect_id_crosswalk,
    collect_injuries,
    collect_player_week,
    collect_schedules,
    collect_snaps,
)
from collect.registry import SOURCES
from store.lake import RESERVED, LocalParquetBackend, read_snapshot, write_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "nflverse"

SEASON = 2024
#: The rewritten depth-chart feed starts here; ``id_crosswalk`` files under its own ``db_season``.
DEPTH_SEASON = 2025
CROSSWALK_SEASON = 2026
CAPTURED_AT = "2026-07-26T12:00:00+00:00"

_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: ``collector, fixture file, expected source, expected partition season``.
CASES: tuple[tuple[object, str, str, int], ...] = (
    (collect_player_week, "player_week_2024", "nflverse_player_week", SEASON),
    (collect_snaps, "snaps_2024", "nflverse_snaps", SEASON),
    (collect_ff_opportunity, "ff_opp_2024", "nflverse_ff_opp", SEASON),
    (collect_injuries, "injuries_2024", "nflverse_injuries", SEASON),
    (collect_schedules, "schedules_2024", "nflverse_schedules", SEASON),
    (collect_depth_charts, "depth_2025", "nflverse_depth", DEPTH_SEASON),
    (collect_id_crosswalk, "id_crosswalk", "id_crosswalk", CROSSWALK_SEASON),
)

#: Sources whose rows are one player-week / game-week and must therefore say which week.
WEEKLY_SOURCES = (
    "nflverse_player_week",
    "nflverse_snaps",
    "nflverse_ff_opp",
    "nflverse_injuries",
    "nflverse_schedules",
)


def frame(name: str) -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / f"{name}.parquet")


def _loader(name: str):
    """Stand in for the ``data.nflverse`` loader: records its arguments, returns the fixture."""

    def load(*args, **kwargs):
        load.calls.append((args, kwargs))
        return frame(name)

    load.calls = []
    return load


def _capture(collector, fixture: str, season: int):
    """Run ``collector`` against a fixture. ``collect_id_crosswalk`` derives its own season."""
    if collector is collect_id_crosswalk:
        return collector(load=_loader(fixture))
    return collector(season, load=_loader(fixture))


@pytest.fixture(scope="module")
def captures() -> dict[str, object]:
    return {
        source: _capture(collector, fixture, season)
        for collector, fixture, source, season in CASES
    }


def _keyed(capture) -> dict[tuple, dict]:
    return {tuple(row[c] for c in capture.key_cols): row for row in capture.rows}


# --------------------------------------------------------------------------- registry agreement
def test_sources_are_registered_with_matching_keys(captures):
    """The envelope's key is the registry's key — the store dedups on whatever it is handed."""
    for _collector, _fixture, source, season in CASES:
        capture = captures[source]
        assert capture.source == source
        assert capture.key_cols == SOURCES[source].key_cols
        assert capture.season == season
        assert capture.rows


def test_every_source_is_season_partitioned(captures):
    """nflverse loaders return a whole season per call, so the week lives in the row, not the path."""
    for capture in captures.values():
        assert capture.week is None


def test_collectors_forward_the_season_to_the_loader():
    load = _loader("player_week_2024")
    collect_player_week(2019, load=load)
    assert load.calls == [((2019,), {})]

    crosswalk = _loader("id_crosswalk")
    collect_id_crosswalk(load=crosswalk)
    assert crosswalk.calls == [((), {})]  # the crosswalk is a live master, not a season's archive


# --------------------------------------------------------------------------- shape contract
def test_weekly_sources_carry_a_week_column(captures):
    for source in WEEKLY_SOURCES:
        for row in captures[source].rows:
            assert "week" in row, f"{source}: no week column"
            assert row["week"] is not None


def test_depth_charts_carry_dt_instead_of_a_week(captures):
    """The documented exception, pinned: the 2025 feed replaced ``week`` with a finer ``dt``.

    Deriving a week means an as-of join against ``nflverse_schedules`` — an assembler decision, not
    something to bake into the raw layer.
    """
    row = captures["nflverse_depth"].rows[0]
    assert "week" not in row
    assert _ISO_UTC_RE.match(row["dt"])


def test_id_crosswalk_is_season_grain_and_takes_its_season_from_db_season(captures):
    capture = captures["id_crosswalk"]
    assert capture.week is None
    assert SOURCES["id_crosswalk"].grain == "season"
    assert capture.season == CROSSWALK_SEASON == frame("id_crosswalk")["db_season"][0]
    # ...and an explicit season still wins, so a backfill can file the master where it wants.
    assert collect_id_crosswalk(2019, load=_loader("id_crosswalk")).season == 2019


def test_mfl_id_is_the_key_because_the_other_id_columns_are_nullable(captures):
    """The crosswalk exists *because* sleeper_id/gsis_id are incomplete — so neither can key it."""
    rows = captures["id_crosswalk"].rows
    assert any(row["sleeper_id"] is None for row in rows)
    assert all(row["mfl_id"] is not None for row in rows)


def test_no_column_is_renamed_added_or_dropped(captures):
    """Provider-native, verbatim: the row is the frame's schema, nothing more and nothing less."""
    for _collector, fixture, source, _season in CASES:
        expected = set(frame(fixture).columns)
        for row in captures[source].rows:
            assert set(row) == expected, f"{source}: row schema drifted from the provider's"


def test_collectors_never_emit_reserved_columns(captures):
    """Provenance is the store's; a collector that forged it would make write_snapshot raise."""
    for capture in captures.values():
        for row in capture.rows:
            assert not set(row) & set(RESERVED)


def test_rows_are_json_safe(captures):
    for source, capture in captures.items():
        for row in capture.rows:
            for name, value in row.items():
                assert value is None or isinstance(value, (str, int, float, bool)), (
                    f"{source}.{name} is {type(value).__name__}, not a json-safe scalar"
                )


# --------------------------------------------------------------------------- provider-native stats
@pytest.mark.parametrize(
    ("source", "fixture", "columns"),
    [
        (
            "nflverse_player_week",
            "player_week_2024",
            ("passing_yards", "passing_tds", "receiving_yards", "receptions", "fg_made_40_49"),
        ),
        ("nflverse_snaps", "snaps_2024", ("offense_snaps", "offense_pct")),
        ("nflverse_ff_opp", "ff_opp_2024", ("rec_attempt", "rec_attempt_team",
                                            "total_fantasy_points_exp")),
        ("nflverse_schedules", "schedules_2024", ("spread_line", "total_line", "roof", "temp")),
    ],
)
def test_stat_columns_keep_their_provider_names_and_values(captures, source, fixture, columns):
    """``ids.nflverse_to_sleeper_stats`` re-scores these later — it needs the nflverse spelling."""
    capture = captures[source]
    original = {
        tuple(row[c] for c in capture.key_cols): row for row in frame(fixture).to_dicts()
    }
    collected = _keyed(capture)
    assert collected

    for key, row in collected.items():
        for name in columns:
            assert name in row, f"{source}: {name} was renamed or dropped"
            assert row[name] == original[key][name], f"{source}: {name} was altered"


def test_no_scoring_is_applied_at_collect_time(captures):
    """The lake is scoring-agnostic — nflverse's own generic points stay, ours never appear."""
    row = captures["nflverse_player_week"].rows[0]
    assert "fantasy_points_ppr" in row  # the provider's, kept as data
    assert not [c for c in row if c.startswith("custom") or c in ("y_custom_points", "proj_pts")]


def test_provider_dtype_quirks_are_left_alone(captures):
    """ff_opportunity types season as a string and week as a float. Tidying a raw layer hides it."""
    row = captures["nflverse_ff_opp"].rows[0]
    assert isinstance(row["season"], str)
    assert isinstance(row["week"], float)


# --------------------------------------------------------------------------- key hygiene
def test_key_cols_identify_rows_on_the_fixture(captures):
    """No duplicate keys within a capture and no null key — the store's two silent loss modes."""
    for source, capture in captures.items():
        keys = [tuple(row[c] for c in capture.key_cols) for row in capture.rows]
        assert len(keys) == len(set(keys)), f"{source}: duplicate key in one capture"
        for key in keys:
            for value in key:
                assert value is not None and str(value).strip(), f"{source}: null key value"


@pytest.mark.parametrize(
    ("source", "fixture", "key_col", "dropped"),
    [
        ("nflverse_player_week", "player_week_2024", "player_id", 2),
        ("nflverse_ff_opp", "ff_opp_2024", "player_id", 2),
        ("nflverse_depth", "depth_2025", "gsis_id", 3),
    ],
)
def test_rows_the_key_cannot_address_are_filtered_to_the_grain(
    captures, caplog, source, fixture, key_col, dropped
):
    """nflverse files a residual line per team-game for production it cannot attribute.

    They are real output at a *team* grain, not broken player rows, so they are filtered here rather
    than left to trip ``dedupe_rows``' defect warning on every run — with the count logged, at INFO,
    so the loss stays auditable. The depth-chart case also carries a *collision*: the unnamed rows
    share one (dt, team, null gsis, pos_abb) key, so filtering removes the duplicate too.
    """
    raw = frame(fixture)
    assert int(raw[key_col].null_count()) == dropped, "fixture no longer covers this shape"

    with caplog.at_level(logging.DEBUG, logger="collect.nflverse"):
        capture = _capture(dict((c[2], c[0]) for c in CASES)[source], fixture, SEASON)

    assert len(capture.rows) == raw.height - dropped
    assert "filtered" in caplog.text and key_col in caplog.text
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_injury_revisions_are_kept_because_date_modified_is_in_the_key(captures):
    """A player listed twice in one week is two point-in-time facts, not a duplicate.

    Without ``date_modified`` the pair collapses to whichever the provider listed last, persisting a
    stale status silently. The fixture carries exactly one such re-reported player-week.
    """
    raw = frame("injuries_2024")
    without = raw.select("gsis_id", "game_type", "week").unique().height
    assert without < raw.height, "fixture no longer carries a re-reported player-week"

    capture = captures["nflverse_injuries"]
    assert len(capture.rows) == raw.height  # both revisions survive

    revisions = {}
    for row in capture.rows:
        revisions.setdefault((row["gsis_id"], row["game_type"], row["week"]), []).append(
            row["date_modified"]
        )
    assert any(len(stamps) > 1 for stamps in revisions.values())


def test_temporal_columns_become_iso_utc_strings(captures):
    """``date_modified`` is a key value: as a string it round-trips parquet and sorts chronologically."""
    expected = sorted(
        row["date_modified"].astimezone(timezone.utc) for row in frame("injuries_2024").to_dicts()
    )
    parsed = []
    for row in captures["nflverse_injuries"].rows:
        stamp = row["date_modified"]
        assert isinstance(stamp, str) and _ISO_UTC_RE.match(stamp), stamp
        parsed.append(datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
    # Equal instant-for-instant, so seconds precision is provably lossless on this feed.
    assert sorted(parsed) == expected


def test_a_date_column_renders_as_a_plain_iso_date():
    rendered = _iso_utc(pl.DataFrame({"gameday": [date(2024, 9, 8)]}))
    assert rendered.to_dicts() == [{"gameday": "2024-09-08"}]


def test_an_unrenderable_temporal_dtype_raises_rather_than_reaching_the_lake():
    """``Collected.rows`` promises json-safe scalars; a timedelta in the lake surfaces far too late."""
    with pytest.raises(ValueError, match="no ISO rendering"):
        _iso_utc(pl.DataFrame({"elapsed": [timedelta(seconds=90)]}))


# --------------------------------------------------------------------------- schema breaks
def test_the_pre_2025_depth_schema_captures_nothing_and_says_so(caplog):
    """nflverse rewrote the feed for 2025; the legacy shape has no key the store can dedup on.

    An empty capture is a documented no-op in ``write_snapshot``, so a multi-season backfill neither
    aborts nor blanks a good partition — it just records that depth features start at 2025.
    """
    legacy = frame("depth_legacy_2024")
    assert "dt" not in legacy.columns and "club_code" in legacy.columns

    with caplog.at_level(logging.WARNING):
        capture = collect_depth_charts(SEASON, load=_loader("depth_legacy_2024"))

    assert capture.rows == []
    assert capture.source == "nflverse_depth"
    assert "pre-2025 depth-chart schema" in caplog.text


def test_an_empty_capture_never_blanks_an_existing_partition(tmp_path):
    backend = LocalParquetBackend(root=tmp_path)
    good = collect_depth_charts(DEPTH_SEASON, load=_loader("depth_2025"))
    write_snapshot(good.source, DEPTH_SEASON, good.rows, captured_at=CAPTURED_AT,
                   week=None, key_cols=good.key_cols, backend=backend)

    empty = collect_depth_charts(DEPTH_SEASON, load=_loader("depth_legacy_2024"))
    write_snapshot(empty.source, DEPTH_SEASON, empty.rows, captured_at="2026-07-27T12:00:00+00:00",
                   week=None, key_cols=empty.key_cols, backend=backend)

    assert len(read_snapshot(good.source, DEPTH_SEASON, backend=backend)) == len(good.rows)


def test_a_missing_key_column_raises_rather_than_keying_on_what_is_left():
    """A provider dropping a key column must not yield a capture the store merges on a partial key."""
    truncated = frame("injuries_2024").drop("date_modified")

    def load(*_args, **_kwargs):
        return truncated

    with pytest.raises(ValueError, match="date_modified"):
        collect_injuries(SEASON, load=load)


# --------------------------------------------------------------------------- store hand-off
@pytest.mark.parametrize(("collector", "fixture", "source", "season"), CASES)
def test_capture_round_trips_through_the_store_without_warnings(
    tmp_path, caplog, collector, fixture, source, season
):
    """The acceptance criterion end-to-end: the store logs nothing about this capture's keys."""
    capture = _capture(collector, fixture, season)
    backend = LocalParquetBackend(root=tmp_path)

    with caplog.at_level(logging.WARNING):
        write_snapshot(
            capture.source,
            capture.season,
            capture.rows,
            captured_at=CAPTURED_AT,
            week=capture.week,
            key_cols=capture.key_cols,
            backend=backend,
        )
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []

    stored = read_snapshot(capture.source, capture.season, capture.week, backend=backend)
    assert len(stored) == len(capture.rows)
    assert not stored[list(capture.key_cols)].duplicated().any()
    assert stored["_source"].eq(source).all()
    assert stored["_season"].eq(season).all()
    assert stored["_week"].isna().all()
    assert stored["_captured_at"].eq(CAPTURED_AT).all()


def test_stats_survive_the_parquet_round_trip(tmp_path):
    capture = _capture(collect_player_week, "player_week_2024", SEASON)
    backend = LocalParquetBackend(root=tmp_path)
    write_snapshot(capture.source, capture.season, capture.rows, captured_at=CAPTURED_AT,
                   week=None, key_cols=capture.key_cols, backend=backend)

    stored = read_snapshot(capture.source, capture.season, backend=backend)
    stored = stored.set_index(list(capture.key_cols))
    for key, row in _keyed(capture).items():
        for name in ("passing_yards", "receiving_yards", "fg_made_40_49", "fantasy_points"):
            expected = row[name]
            if expected is None:
                continue
            assert stored.loc[key, name] == expected, f"{name} did not survive parquet"

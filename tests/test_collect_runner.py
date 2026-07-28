"""Offline unit tests for the collection runners (``collect.runner`` + the two ``scripts/`` CLIs).

Nothing here is a mock of the thing under test. The **real** collectors, the real dispatch table and
the real store all run; what is replaced is the outside world — ``sleeper.client``'s three fetches,
``data.nflverse``'s loaders and open-meteo's fetcher — so a run exercises the same code path the Tue
cron does, against the committed fixtures, with no network. Everything is captured under ``SEASON``
(2025) because the depth-chart fixture is the modern feed; the other fixtures are 2024 rows, which is
fine — the partition season is what the run asks for, and no test here reads a game date.

Four properties carry this ticket, and each is a review finding from an earlier PR rather than a
guess about what might break:

* **the ``_backfill`` marker lands on every row**, both values, so a partition holding a live capture
  *and* a backfilled one can be told apart row by row (ticket 7 resolves an anachronistic ``position``
  through it);
* **one capture is written and released before the next is collected** — the real depth capture peaks
  at ~858 MB, and a runner that collected all seven before writing would need seven times that;
* **the season schedule is loaded exactly once**, then handed to ``nflverse_schedules``, ``vegas_odds``
  and every week's ``weather`` — the naive version re-parses it ~180 times over a 10-season backfill;
* **``nflverse_depth`` is not walked before 2025**, via the registry's ``backfillable_from``.

And the one that guards all of them: a full run must reach the store with **zero warnings**. Both
shapes that provoke one (a null key, a key repeated within a capture) delete real rows silently, so a
warning here is a defect signal and never routine noise.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from collect import runner
from collect import weather as collect_weather
from collect.registry import SOURCES, backfillable_sources, sources_for_cadence
from data import nflverse as nflverse_data
from sleeper import client
from store.lake import LocalParquetBackend, read_snapshot, set_backend

FIXTURES = Path(__file__).parent / "fixtures"
NFLVERSE = FIXTURES / "nflverse"
SLEEPER_RAW = FIXTURES / "sleeper_raw_2025_w1.json"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

SEASON = 2025
WEEK = 1
#: The id crosswalk is a live master: the collector files it under the *feed's* own ``db_season``,
#: not under the season the run asked about. The committed fixture says 2026.
CROSSWALK_SEASON = 2026
#: The season before nflverse rewrote the depth-chart feed — the backfill must not walk it.
LEGACY_DEPTH_SEASON = 2024

CAPTURED_AT = "2026-09-13T16:00:00+00:00"
LATER_CAPTURE = "2026-09-14T16:00:00+00:00"

#: Fixture file -> the loader it stands in for, and the counter key the tests assert on.
_LOADERS: tuple[tuple[str, str, str], ...] = (
    ("load_weekly_actuals", "player_week_2024", "player_week"),
    ("load_snap_counts", "snaps_2024", "snaps"),
    ("load_ff_opportunity", "ff_opp_2024", "ff_opp"),
    ("load_injuries", "injuries_2024", "injuries"),
    ("load_depth_charts", "depth_2025", "depth"),
    ("load_id_crosswalk", "id_crosswalk", "crosswalk"),
)


@pytest.fixture
def lake(tmp_path):
    """A process-wide lake rooted in ``tmp_path`` — what the CLIs write through with no arguments."""
    set_backend(LocalParquetBackend(root=tmp_path))
    yield tmp_path
    set_backend(None)


@pytest.fixture
def offline(monkeypatch) -> Counter:
    """Replace every outside-world call with a fixture; return a per-source call counter."""
    raw = json.loads(SLEEPER_RAW.read_text(encoding="utf-8"))
    calls: Counter = Counter()

    def sleeper_fetch(name):
        def fetch(*args, **kwargs):
            calls[name] += 1
            return raw[name]["rows"]

        return fetch

    monkeypatch.setattr(client, "get_projections", sleeper_fetch("proj_week"))
    monkeypatch.setattr(client, "get_season_projections", sleeper_fetch("proj_season"))
    monkeypatch.setattr(client, "get_stats", sleeper_fetch("stats_week"))

    def loader(fixture: str, key: str):
        def load(*args, **kwargs):
            calls[key] += 1
            return pl.read_parquet(NFLVERSE / f"{fixture}.parquet")

        return load

    for name, fixture, key in _LOADERS:
        monkeypatch.setattr(nflverse_data, name, loader(fixture, key))

    def load_schedules(season, *args, **kwargs):
        """The week-1 2024 sample, restamped to whichever season is being captured."""
        calls["schedules"] += 1
        frame = pl.read_parquet(NFLVERSE / "schedules_2024.parquet")
        return frame.with_columns(
            pl.lit(int(season)).cast(frame.schema["season"]).alias("season")
        )

    monkeypatch.setattr(nflverse_data, "load_schedules", load_schedules)
    # No HTTP: outside its window the real fetcher returns None too, and this pins it.
    monkeypatch.setattr(collect_weather, "open_meteo_hourly", lambda *a, **k: None)
    return calls


def _load_cli(name: str):
    """Import ``scripts/<name>.py`` as a module (the CLIs are not on the import path)."""
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _partitions(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.parquet"))


# --------------------------------------------------------------------------- dispatch table
def test_every_registered_source_has_a_collector():
    """A source in the registry with no collector is silently uncaptured — catch it here."""
    assert set(runner.COLLECTORS) == set(SOURCES)


def test_a_source_without_a_collector_is_skipped_rather_than_fatal(monkeypatch, caplog):
    """A half-landed future ticket must not cost a pre-lock run the projections it exists for."""
    monkeypatch.delitem(runner.COLLECTORS, "nflverse_snaps")
    with caplog.at_level(logging.WARNING, logger="collect.runner"):
        tasks = runner.plan_tasks(
            [SOURCES["nflverse_snaps"]], SEASON, [WEEK], runner.RunContext()
        )
    assert tasks == []
    assert "has no collector" in caplog.text


# --------------------------------------------------------------------------- cadence runs
def test_postgame_writes_every_registered_postgame_partition(offline, lake, caplog):
    """The headline acceptance criterion: every post-game source captured, counted and filed."""
    with caplog.at_level(logging.WARNING):
        results = runner.run_cadence("postgame", SEASON, WEEK, captured_at=CAPTURED_AT)

    assert {r.source for r in results} == {s.name for s in sources_for_cadence("postgame")}
    assert all(r.ok for r in results), [r.error for r in results if not r.ok]
    assert all(r.rows > 0 and r.path is not None for r in results)
    # `lake` is a LocalParquetBackend, so the locator really is a filesystem path. On the S3
    # backend it is an `s3://bucket/key` string and only `str()` of it means anything -- hence the
    # local-only assertion rather than one on `CaptureResult.path` in general.
    assert all(Path(r.path).is_file() for r in results)
    assert runner.exit_code(results) == 0
    # A capture with no partition on disk would still "succeed" — check the store really has them.
    assert len(_partitions(lake)) == len(results)
    assert not caplog.records, [r.getMessage() for r in caplog.records]

    summary = runner.format_summary(results)
    for result in results:
        assert result.source in summary
    assert "6/6 source(s) captured" in summary


def test_prelock_writes_every_registered_prelock_partition(offline, lake, caplog):
    with caplog.at_level(logging.WARNING):
        results = runner.run_cadence("prelock", SEASON, WEEK, captured_at=CAPTURED_AT)

    assert {r.source for r in results} == {s.name for s in sources_for_cadence("prelock")}
    assert all(r.ok and r.rows > 0 for r in results), [r.error for r in results if not r.ok]
    assert not caplog.records, [r.getMessage() for r in caplog.records]

    # The forward-only sources are captured first, so a run that dies halfway has still saved the
    # rows nothing can ever recover.
    assert [r.source for r in results][:2] == ["sleeper_proj_week", "sleeper_proj_season"]


def test_week_partitioned_and_season_partitioned_sources_land_where_the_registry_says(
    offline, lake
):
    runner.run_cadence("postgame", SEASON, WEEK, captured_at=CAPTURED_AT)
    written = _partitions(lake)
    # Sleeper is fetched a week at a time; the nflverse loaders return a whole season.
    assert f"sleeper_stats_week/season={SEASON}/sleeper_stats_week_{SEASON}_wk01.parquet" in written
    assert (
        f"nflverse_player_week/season={SEASON}/nflverse_player_week_{SEASON}_season.parquet"
    ) in written


def test_the_crosswalk_is_filed_under_the_feeds_own_season(offline, lake):
    """Season and week come from the capture envelope, not from the run — the crosswalk proves it."""
    results = runner.run_cadence("postgame", SEASON, WEEK, captured_at=CAPTURED_AT)
    crosswalk = next(r for r in results if r.source == "id_crosswalk")
    assert crosswalk.season == CROSSWALK_SEASON != SEASON
    assert not read_snapshot("id_crosswalk", CROSSWALK_SEASON).empty


# --------------------------------------------------------------------------- best effort
def test_one_failing_collector_does_not_stop_the_others(offline, lake, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("nflverse release is unreachable")

    monkeypatch.setattr(nflverse_data, "load_snap_counts", boom)
    results = runner.run_cadence("postgame", SEASON, WEEK, captured_at=CAPTURED_AT)

    failed = [r for r in results if not r.ok]
    assert [r.source for r in failed] == ["nflverse_snaps"]
    assert "RuntimeError: nflverse release is unreachable" in failed[0].error
    assert all(r.rows > 0 for r in results if r.ok)
    # Best-effort: the other five sources are on disk and the run is still a success.
    assert len(_partitions(lake)) == len(results) - 1
    assert runner.exit_code(results) == 0
    assert "FAILED 1/1" in runner.format_summary(results)


def test_a_run_is_a_failure_only_when_every_capture_failed(offline, lake, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("nflverse release is unreachable")

    monkeypatch.setattr(nflverse_data, "load_snap_counts", boom)
    results = runner.run_cadence(
        "postgame", SEASON, WEEK, captured_at=CAPTURED_AT, sources=["nflverse_snaps"]
    )
    assert [r.ok for r in results] == [False]
    assert runner.exit_code(results) == 1
    assert _partitions(lake) == []


@pytest.mark.parametrize(
    ("errors", "expected"),
    [((), 0), ((False,), 0), ((True,), 1), ((True, False), 0), ((True, True), 1)],
)
def test_exit_code_rule(errors, expected):
    results = [
        runner.CaptureResult("nflverse_snaps", SEASON, None, error="boom" if failed else None)
        for failed in errors
    ]
    assert runner.exit_code(results) == expected


def test_a_lost_forward_only_capture_fails_the_run_on_its_own(offline, lake, monkeypatch, caplog):
    """The one failure no backfill undoes must be able to turn the cron red by itself.

    Sleeper serves only the *latest* projections, so a pre-lock capture that does not happen is gone
    for good. Under a pure all-or-nothing rule that loss is invisible: the other six sources succeed
    and the run is green.
    """
    def boom(*args, **kwargs):
        raise ConnectionError("sleeper projections timed out")

    monkeypatch.setattr(client, "get_projections", boom)
    with caplog.at_level(logging.ERROR, logger="collect.runner"):
        results = runner.run_cadence("prelock", SEASON, WEEK, captured_at=CAPTURED_AT)

    failed = [r.source for r in results if not r.ok]
    assert failed == ["sleeper_proj_week"]
    assert runner.exit_code(results) == 1
    assert "unrecoverable" in caplog.text
    # Still best-effort: everything recoverable was captured anyway, which is the whole point of
    # reporting the loss rather than aborting on it.
    assert all(r.ok and r.rows > 0 for r in results if r.source != "sleeper_proj_week")
    assert len(_partitions(lake)) == len(results) - 1


def test_a_lost_backfillable_capture_leaves_the_run_green(offline, lake, monkeypatch):
    """The counterweight: nflverse is recoverable, so a release hiccup is not a failed run."""
    def boom(*args, **kwargs):
        raise RuntimeError("nflverse release is unreachable")

    monkeypatch.setattr(nflverse_data, "load_injuries", boom)
    results = runner.run_cadence("prelock", SEASON, WEEK, captured_at=CAPTURED_AT)
    assert [r.source for r in results if not r.ok] == ["nflverse_injuries"]
    assert runner.exit_code(results) == 0


def test_only_the_sleeper_projections_are_forward_only():
    """Pins what the exit rule is actually about, so adding a source re-decides it deliberately."""
    forward_only = {name for name in SOURCES if runner._is_forward_only(name)}
    assert forward_only == {"sleeper_proj_week", "sleeper_proj_season"}
    # A backfill run therefore cannot trip the rule: none of its sources qualify.
    assert not forward_only & {s.name for s in backfillable_sources()}
    # An unregistered name is not treated as unrecoverable (format_summary groups on the raw string).
    assert not runner._is_forward_only("not_a_source")


def test_an_empty_capture_is_a_success_that_wrote_nothing(offline, lake, monkeypatch):
    """``write_snapshot`` treats no rows as a no-op, so the result must not point at a partition."""
    monkeypatch.setattr(
        nflverse_data, "load_depth_charts", lambda *a, **k: pl.read_parquet(
            NFLVERSE / "depth_legacy_2024.parquet"
        )
    )
    results = runner.run_cadence(
        "prelock", SEASON, WEEK, captured_at=CAPTURED_AT, sources=["nflverse_depth"]
    )
    assert [(r.ok, r.rows, r.path) for r in results] == [(True, 0, None)]
    assert _partitions(lake) == []
    assert "nothing written" in runner.format_summary(results)


# --------------------------------------------------------------------------- memory discipline
def test_each_capture_is_written_before_the_next_is_collected(offline, lake, monkeypatch):
    """Peak memory is one capture, not the sum of them (the depth capture alone peaks at ~858 MB).

    Asserted as a strict ``collect, write, collect, write`` alternation: a runner that built its
    captures up front and wrote them at the end would show every collect before the first write.
    """
    events: list[tuple[str, str]] = []

    def recorded(name, per_week, collect):
        def wrapper(season, week, ctx):
            events.append(("collect", name))
            return collect(season, week, ctx)

        return name, (per_week, wrapper)

    monkeypatch.setattr(
        runner,
        "COLLECTORS",
        dict(recorded(name, per_week, fn) for name, (per_week, fn) in runner.COLLECTORS.items()),
    )
    real_write = runner.write_snapshot

    def spy(source, season, rows, **kwargs):
        events.append(("write", source))
        return real_write(source, season, rows, **kwargs)

    monkeypatch.setattr(runner, "write_snapshot", spy)

    results = runner.run_cadence("postgame", SEASON, WEEK, captured_at=CAPTURED_AT)

    assert len(events) == 2 * len(results)
    assert [kind for kind, _ in events] == ["collect", "write"] * len(results)
    assert [name for _, name in events[0::2]] == [name for _, name in events[1::2]]


def test_a_capture_result_never_carries_the_rows_it_wrote(offline, lake):
    """What survives a task is counts and a path — holding the frame would defeat the whole point."""
    results = runner.run_cadence("postgame", SEASON, WEEK, captured_at=CAPTURED_AT)
    fields = set(vars(results[0]))
    assert fields == {"source", "season", "week", "rows", "path", "error"}


# --------------------------------------------------------------------------- schedule reuse
def test_the_season_schedule_is_loaded_once_per_run(offline, lake):
    """Three pre-lock sources need it; left to their defaults each would load its own copy."""
    results = runner.run_cadence("prelock", SEASON, WEEK, captured_at=CAPTURED_AT)
    consumers = {"nflverse_schedules", "vegas_odds", "weather"}
    assert consumers <= {r.source for r in results}
    assert offline["schedules"] == 1


def test_the_backfill_loads_each_seasons_schedule_once(offline, lake):
    """~180 re-parses over a 10-season backfill is the failure this pins (one per week, per source)."""
    seasons = (2023, 2024, 2025)
    runner.run_backfill(seasons, captured_at=CAPTURED_AT)
    assert offline["schedules"] == len(seasons)


def test_the_run_context_repr_does_not_dump_the_schedule(offline):
    """A context that renders its payload turns any log line or traceback into a data dump."""
    ctx = runner.RunContext()
    ctx.schedules(SEASON)
    assert "_frame" not in repr(ctx)
    assert len(repr(ctx)) < 200


def test_the_schedule_frame_is_shared_not_copied_per_consumer(offline):
    ctx = runner.RunContext()
    assert ctx.schedules(SEASON) is ctx.schedules(SEASON)
    assert offline["schedules"] == 1
    ctx.schedules(SEASON + 1)  # a new season replaces the cache rather than accumulating
    assert offline["schedules"] == 2
    ctx.release()
    assert ctx.schedules(SEASON + 1) is not None
    assert offline["schedules"] == 3


def test_a_missing_schedule_does_not_skip_the_season(offline, lake, monkeypatch, caplog):
    """The weeks fall back to a static range; the two schedule-derived sources report their own loss."""
    monkeypatch.setattr(
        nflverse_data, "load_schedules", lambda *a, **k: (_ for _ in ()).throw(OSError("no release"))
    )
    with caplog.at_level(logging.WARNING, logger="collect.runner"):
        results = runner.run_backfill([SEASON], captured_at=CAPTURED_AT)

    by_source = {r.source: r for r in results}
    assert not by_source["vegas_odds"].ok and not by_source["weather"].ok
    assert by_source["nflverse_player_week"].ok
    assert "falling back to a static week range" in caplog.text
    # 18 weekly captures attempted rather than none: weather/stats_week still get their whole season.
    assert sum(1 for r in results if r.source == "sleeper_stats_week") == 18


@pytest.mark.parametrize(
    ("season", "expected"), [(2016, tuple(range(1, 18))), (2021, tuple(range(1, 19)))]
)
def test_the_static_week_fallback_matches_the_seasons_length(season, expected):
    assert runner.regular_season_weeks(season, None) == expected


def test_weeks_come_from_the_schedule_when_it_is_there(offline):
    ctx = runner.RunContext()
    assert runner.regular_season_weeks(SEASON, ctx.schedules(SEASON)) == (1,)


# --------------------------------------------------------------------------- the backfill marker
def test_backfilled_rows_and_live_rows_are_distinguishable_row_by_row(offline, lake):
    """The marker's real job: one partition holding both kinds, with no null in between.

    A backfilled ``sleeper_stats_week`` row carries the player's *today* position where a live one
    carries that week's, so ticket 7 has to be able to tell them apart per row — a run-level log line
    could not.
    """
    runner.run_cadence("postgame", SEASON, WEEK, captured_at=CAPTURED_AT)
    runner.run_backfill([SEASON], captured_at=LATER_CAPTURE, sources=["sleeper_stats_week"])

    frame = read_snapshot("sleeper_stats_week", SEASON, WEEK)
    marker = frame[runner.BACKFILL_COL]
    assert not marker.isna().any()
    assert marker.dtype == bool
    assert set(marker) == {True, False}
    # Both captures survive in full: a later capture date is a new point-in-time snapshot, not an
    # overwrite, so the live rows are still there to compare against.
    assert marker.sum() == (~marker).sum() == len(frame) / 2


@pytest.mark.parametrize("source", [s.name for s in backfillable_sources()])
def test_every_backfilled_row_carries_the_marker(offline, lake, source):
    results = runner.run_backfill([SEASON], captured_at=CAPTURED_AT, sources=[source])
    assert results and all(r.ok for r in results)
    for result in results:
        frame = read_snapshot(result.source, result.season, result.week)
        assert not frame.empty
        assert frame[runner.BACKFILL_COL].all()


def test_a_scheduled_capture_marks_its_rows_as_not_backfilled(offline, lake):
    results = runner.run_cadence("prelock", SEASON, WEEK, captured_at=CAPTURED_AT)
    for result in results:
        frame = read_snapshot(result.source, result.season, result.week)
        assert not frame[runner.BACKFILL_COL].any()


# --------------------------------------------------------------------------- backfill scope
def test_the_backfill_skips_depth_charts_before_the_feed_existed(offline, lake, caplog):
    """``backfillable_from`` — the alternative was nine empty partitions and nine warnings."""
    with caplog.at_level(logging.WARNING):
        legacy = runner.run_backfill([LEGACY_DEPTH_SEASON], captured_at=CAPTURED_AT)
    assert "nflverse_depth" not in {r.source for r in legacy}
    assert not caplog.records, [r.getMessage() for r in caplog.records]

    modern = runner.run_backfill([SEASON], captured_at=CAPTURED_AT)
    depth = next(r for r in modern if r.source == "nflverse_depth")
    assert depth.ok and depth.rows > 0


def test_the_backfill_never_touches_a_forward_only_source(offline, lake):
    results = runner.run_backfill([SEASON], captured_at=CAPTURED_AT)
    assert not {"sleeper_proj_week", "sleeper_proj_season"} & {r.source for r in results}


def test_the_live_crosswalk_master_is_pulled_once_for_the_whole_run(offline, lake):
    """It is not a per-season archive: N seasons would re-download it N times into one partition."""
    runner.run_backfill([2023, 2024, SEASON], captured_at=CAPTURED_AT, sources=["id_crosswalk"])
    assert offline["crosswalk"] == 1


def test_a_failed_season_invariant_capture_is_retried_on_the_next_season(offline, lake, monkeypatch):
    """Marked done from the *results*, not the plan.

    Marked at plan time, one transient error on the first season of a ten-season run costs the whole
    run its crosswalk — and says nothing, because the other sources succeed and the exit code is 0.
    """
    attempts: list[int] = []
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        attempts.append(calls["n"])
        if calls["n"] == 1:
            raise RuntimeError("ffverse release is unreachable")
        return pl.read_parquet(NFLVERSE / "id_crosswalk.parquet")

    monkeypatch.setattr(nflverse_data, "load_id_crosswalk", flaky)
    results = runner.run_backfill(
        [2023, 2024, SEASON], captured_at=CAPTURED_AT, sources=["id_crosswalk"]
    )

    # Retried on 2024, succeeded, and then *not* attempted again for 2025.
    assert len(attempts) == 2
    assert [r.ok for r in results] == [False, True]
    assert not read_snapshot("id_crosswalk", CROSSWALK_SEASON).empty


def test_a_subset_that_reaches_none_of_the_seasons_raises(offline, lake):
    """``parse_sources`` validates names, not reach — so this used to be a silent green no-op.

    ``--sources nflverse_depth --seasons 2016-2024`` passed every check, wrote nothing, and printed
    a success banner. A one-time backfill that reports success is expensive to disbelieve.
    """
    with pytest.raises(ValueError, match="backfills from 2025"):
        runner.run_backfill(
            [2016, LEGACY_DEPTH_SEASON], captured_at=CAPTURED_AT, sources=["nflverse_depth"]
        )
    assert _partitions(lake) == []


def test_partial_reach_runs_the_seasons_it_can_and_skips_the_rest(offline, lake):
    """A subset reaching *some* seasons runs those and skips the others before spending anything."""
    results = runner.run_backfill(
        [LEGACY_DEPTH_SEASON, SEASON], captured_at=CAPTURED_AT, sources=["nflverse_depth"]
    )
    assert [(r.season, r.ok) for r in results] == [(SEASON, True)]


def test_dropping_the_eager_load_does_not_cost_the_sharing(offline, lake):
    """``nflverse_schedules`` + ``vegas_odds`` are both season-level, so nothing loads at plan time.

    They then reach ``ctx.schedules`` from inside their own thunks — and must still land on the same
    cached frame. This is the review constraint the plan-time skip is most likely to quietly undo.
    """
    results = runner.run_backfill(
        [SEASON], captured_at=CAPTURED_AT, sources=["nflverse_schedules", "vegas_odds"]
    )
    assert [r.ok for r in results] == [True, True]
    assert offline["schedules"] == 1


def test_a_backfill_with_no_week_partitioned_source_downloads_no_schedule(offline, lake):
    """The schedule is planning input for the week fan-out, nothing else.

    ``nflverse_player_week`` returns a whole season at once, so a run of it alone needs no weeks —
    and the schedule-consuming collectors load lazily inside their own thunk. Ten seasons of this
    used to cost ten schedule downloads that nothing read.
    """
    runner.run_backfill(
        [2023, 2024, SEASON], captured_at=CAPTURED_AT, sources=["nflverse_player_week"]
    )
    assert offline["schedules"] == 0
    assert offline["player_week"] == 3


def test_the_backfill_walks_every_season_it_is_given(offline, lake):
    results = runner.run_backfill(
        [2023, 2024], captured_at=CAPTURED_AT, sources=["nflverse_player_week"]
    )
    assert {r.season for r in results} == {2023, 2024}
    for season in (2023, 2024):
        assert not read_snapshot("nflverse_player_week", season).empty


# --------------------------------------------------------------------------- argument parsing
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2016-2025", tuple(range(2016, 2026))),
        ("2019", (2019,)),
        ("2016,2018-2020", (2016, 2018, 2019, 2020)),
        ("2020, 2020 , 2019", (2019, 2020)),
    ],
)
def test_parse_seasons(text, expected):
    assert runner.parse_seasons(text) == expected


@pytest.mark.parametrize("text", ["", "not-a-year", "20x6", "2025-2016", "2016-"])
def test_parse_seasons_rejects_nonsense(text):
    with pytest.raises(ValueError):
        runner.parse_seasons(text)


def test_parse_sources_validates_against_the_registry():
    assert runner.parse_sources(None) is None
    assert runner.parse_sources("nflverse_snaps, weather") == ("nflverse_snaps", "weather")
    with pytest.raises(ValueError, match="unknown source"):
        runner.parse_sources("nflverse_snapz")
    with pytest.raises(ValueError, match="forward-only"):
        runner.parse_sources("sleeper_proj_week", backfill=True)


# --------------------------------------------------------------------------- run planning
def test_plan_run_reads_the_season_and_week_from_sleeper():
    plan = runner.plan_run({"season": "2026", "season_type": "regular", "week": 4})
    assert (plan.season, plan.week, plan.skip) == (2026, 4, None)


def test_plan_run_skips_outside_the_regular_season():
    plan = runner.plan_run({"season": "2026", "season_type": "pre", "week": 0})
    assert plan.skip and "off-season" in plan.skip


def test_an_explicit_season_and_week_always_run():
    """That is how a missed week is re-captured — including in the off-season."""
    plan = runner.plan_run(
        {"season": "2026", "season_type": "post", "week": 1}, season=2025, week=12
    )
    assert (plan.season, plan.week, plan.skip) == (2025, 12, None)


@pytest.mark.parametrize("state", [{}, None])
def test_plan_run_refuses_to_guess_when_sleeper_is_unreachable(state):
    with pytest.raises(ValueError, match="could not determine"):
        runner.plan_run(state)


def test_plan_run_refuses_a_state_with_no_season():
    with pytest.raises(ValueError, match="no usable season"):
        runner.plan_run({"season_type": "regular", "week": 3})


# --------------------------------------------------------------------------- postgame week (#16)
#: A two-week REG schedule. Week 1 ends Monday 2026-09-14 20:15 ET (= Tue 00:15 UTC); week 2 opens
#: Thursday 2026-09-17 20:15 ET (= Fri 00:15 UTC). Nothing else changes between the two states below.
_FLIP_SCHEDULE = (
    (1, "2026-09-10", "20:15"),  # Thu
    (1, "2026-09-13", "13:00"),  # Sun
    (1, "2026-09-14", "20:15"),  # Mon — week 1's last kickoff
    (2, "2026-09-17", "20:15"),  # Thu — week 2's first kickoff
    (2, "2026-09-20", "13:00"),  # Sun
)
#: Tuesday 12:00 UTC, the planned postgame cron: after week 1 settled (Mon kickoff + 6h = 06:15 UTC),
#: before week 2 even kicks off. This is exactly when Sleeper's ``state.week`` flips to the upcoming
#: week, so it is the moment the naive resolver races.
_TUE_NOON_UTC = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _schedule(rows):
    """A minimal REG schedule frame from ``(week, gameday, gametime)`` triples."""
    return pl.DataFrame(
        {
            "game_type": ["REG"] * len(rows),
            "week": [int(w) for w, _, _ in rows],
            "gameday": [d for _, d, _ in rows],
            "gametime": [t for _, _, t in rows],
        },
        schema={"game_type": pl.Utf8, "week": pl.Int64, "gameday": pl.Utf8, "gametime": pl.Utf8},
    )


def _schedule_ctx(frame):
    """A ``RunContext`` whose schedule loader returns ``frame`` and counts how often it is called."""
    calls = Counter()

    def load(season, *args, **kwargs):
        calls["load"] += 1
        return frame

    return runner.RunContext(load_schedules=load), calls


@pytest.mark.parametrize("state_week", [1, 2])
def test_postgame_resolves_the_completed_week_across_the_tuesday_flip(state_week):
    """The headline fix: same schedule, ``state.week`` = N and N+1, postgame resolves N both times.

    ``state_week=1`` is the run landing before Sleeper's Tuesday rollover; ``2`` is after it. The
    naive resolver would file week 2 (a zeroed, not-yet-played snapshot) in the second case.
    """
    ctx, _ = _schedule_ctx(_schedule(_FLIP_SCHEDULE))
    state = {"season": "2026", "season_type": "regular", "week": state_week}
    plan = runner.plan_run(state, mode="postgame", now=_TUE_NOON_UTC, ctx=ctx)
    assert (plan.season, plan.week, plan.skip) == (2026, 1, None)


def test_prelock_still_captures_the_upcoming_week():
    """``prelock`` is unchanged: it keeps ``state.week`` (the upcoming week) and never reads the schedule."""
    ctx, calls = _schedule_ctx(_schedule(_FLIP_SCHEDULE))
    state = {"season": "2026", "season_type": "regular", "week": 2}
    plan = runner.plan_run(state, mode="prelock", now=_TUE_NOON_UTC, ctx=ctx)
    assert (plan.season, plan.week, plan.skip) == (2026, 2, None)
    assert calls["load"] == 0


def test_an_explicit_week_overrides_postgame_without_reading_the_schedule():
    """``--season/--week`` still wins over everything — the schedule is not even consulted."""
    ctx, calls = _schedule_ctx(_schedule(_FLIP_SCHEDULE))
    state = {"season": "2026", "season_type": "regular", "week": 2}
    plan = runner.plan_run(state, mode="postgame", now=_TUE_NOON_UTC, season=2025, week=9, ctx=ctx)
    assert (plan.season, plan.week, plan.skip) == (2025, 9, None)
    assert calls["load"] == 0


def test_the_resolved_postgame_week_and_the_next_rejection_are_logged(caplog):
    """A wrong answer must be visible in the cron log, not only in the data."""
    ctx, _ = _schedule_ctx(_schedule(_FLIP_SCHEDULE))
    state = {"season": "2026", "season_type": "regular", "week": 2}
    with caplog.at_level(logging.INFO, logger="collect.runner"):
        runner.plan_run(state, mode="postgame", now=_TUE_NOON_UTC, ctx=ctx)
    assert "completed week 1" in caplog.text
    assert "week 2 rejected" in caplog.text


def test_an_off_season_postgame_returns_before_touching_the_schedule():
    """Off-season is decided first: next season's schedule is often unpublished and must not load."""
    ctx, calls = _schedule_ctx(_schedule(_FLIP_SCHEDULE))
    state = {"season": "2026", "season_type": "pre", "week": 0}
    plan = runner.plan_run(state, mode="postgame", now=_TUE_NOON_UTC, ctx=ctx)
    assert plan.skip and "off-season" in plan.skip
    assert calls["load"] == 0


def test_a_postgame_schedule_load_failure_skips_rather_than_captures(caplog):
    """An unresolvable week is a green skip, never a fall back to the upcoming ``state.week``."""
    def boom(season, *args, **kwargs):
        raise OSError("nflverse release is unreachable")

    ctx = runner.RunContext(load_schedules=boom)
    state = {"season": "2026", "season_type": "regular", "week": 2}
    with caplog.at_level(logging.WARNING, logger="collect.runner"):
        plan = runner.plan_run(state, mode="postgame", now=_TUE_NOON_UTC, ctx=ctx)
    assert plan.season == 2026
    assert plan.skip and "schedule is unavailable" in plan.skip


def test_a_postgame_run_before_any_week_has_finished_skips():
    """Regular season declared but no REG week settled yet -> nothing to capture, so skip."""
    ctx, _ = _schedule_ctx(_schedule(_FLIP_SCHEDULE))
    state = {"season": "2026", "season_type": "regular", "week": 1}
    before_kickoff = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)  # before week 1's Thursday game
    plan = runner.plan_run(state, mode="postgame", now=before_kickoff, ctx=ctx)
    assert plan.skip and "has finished" in plan.skip


def test_latest_completed_week_requires_games_to_have_finished_not_just_started():
    """``kicked off`` is not ``finished``: a game counts only after the settle margin, not at kickoff."""
    schedule = _schedule([
        (1, "2026-09-13", "13:00"),  # Sun 13:00 ET = 17:00 UTC, long finished
        (2, "2026-09-20", "13:00"),  # Sun 13:00 ET = 17:00 UTC, +6h settle -> 23:00 UTC
    ])
    just_after_kickoff = datetime(2026, 9, 20, 18, 0, tzinfo=timezone.utc)  # 1h in, < 6h settle
    assert runner.latest_completed_week(schedule, just_after_kickoff) == 1
    past_the_settle = datetime(2026, 9, 21, 0, 0, tzinfo=timezone.utc)
    assert runner.latest_completed_week(schedule, past_the_settle) == 2


def test_latest_completed_week_treats_an_unreadable_kickoff_as_not_finished():
    """A week with any blank ``gametime`` (nflverse's unflexed late games) is not complete.

    Dropping the ``None`` kickoffs before ``max()`` would mark an unstarted week finished — the bug
    the resolver must avoid. Here week 2 has a readable game months in the past *and* a blank one, so
    the week resolves to 1 however late ``now`` is.
    """
    schedule = _schedule([
        (1, "2026-09-13", "13:00"),
        (2, "2026-09-20", "13:00"),
        (2, "2026-09-21", ""),  # unflexed: no kickoff time published yet
    ])
    months_later = datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)
    assert runner.latest_completed_week(schedule, months_later) == 1


def test_latest_completed_week_is_none_when_no_week_has_finished():
    schedule = _schedule(_FLIP_SCHEDULE)
    before_the_season = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    assert runner.latest_completed_week(schedule, before_the_season) is None
    assert runner.latest_completed_week(None, _TUE_NOON_UTC) is None


def test_format_summary_of_nothing():
    assert "no sources" in runner.format_summary([])


# --------------------------------------------------------------------------- the CLIs
def test_collect_cli_captures_and_reports_per_source(offline, lake, monkeypatch, capsys):
    cli = _load_cli("collect")
    monkeypatch.setattr(
        cli.client, "get_state", lambda *a, **k: {"season": str(SEASON), "season_type": "regular",
                                                  "week": WEEK}
    )
    assert cli.main(["--mode", "postgame"]) == 0

    out = capsys.readouterr().out
    for source in sources_for_cadence("postgame"):
        assert source.name in out
    assert "rows" in out and _partitions(lake)


def test_collect_cli_no_ops_in_the_off_season(offline, lake, monkeypatch, capsys):
    """Exit 0 and write nothing, so the cron stays green until the season starts."""
    cli = _load_cli("collect")
    monkeypatch.setattr(
        cli.client, "get_state", lambda *a, **k: {"season": "2026", "season_type": "pre", "week": 0}
    )
    assert cli.main(["--mode", "prelock"]) == 0
    assert _partitions(lake) == []
    assert "Skipping" in capsys.readouterr().out


def test_collect_cli_aborts_rather_than_guess_the_week(offline, lake, monkeypatch):
    """A wrong partition for a forward-only source is a week lost for good — fail red instead."""
    def unreachable(*args, **kwargs):
        raise ConnectionError("sleeper is down")

    cli = _load_cli("collect")
    monkeypatch.setattr(cli.client, "get_state", unreachable)
    assert cli.main(["--mode", "prelock"]) == 1
    assert _partitions(lake) == []


def test_collect_cli_reports_a_broken_install_rather_than_a_traceback(offline, lake, monkeypatch):
    """``plan_run`` imports ``analysis.snapshot`` lazily — the whole optimizer/pulp stack.

    A cron log wants the actionable one-liner, not a stack trace from an import three layers down.
    """
    cli = _load_cli("collect")
    monkeypatch.setattr(cli.client, "get_state", lambda *a, **k: {})
    monkeypatch.setattr(
        cli.runner, "plan_run",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("No module named 'pulp'")),
    )
    assert cli.main(["--mode", "prelock"]) == 1
    assert _partitions(lake) == []


def test_collect_cli_honours_an_explicit_season_and_week(offline, lake, monkeypatch):
    cli = _load_cli("collect")
    monkeypatch.setattr(cli.client, "get_state", lambda *a, **k: {})
    assert cli.main(["--mode", "postgame", "--season", str(SEASON), "--week", "3"]) == 0
    assert not read_snapshot("sleeper_stats_week", SEASON, 3).empty


def test_backfill_cli_marks_what_it_writes(offline, lake, capsys):
    cli = _load_cli("backfill_lake")
    assert cli.main(["--seasons", str(SEASON), "--sources", "nflverse_snaps"]) == 0
    frame = read_snapshot("nflverse_snaps", SEASON)
    assert not frame.empty and frame[runner.BACKFILL_COL].all()
    assert "nflverse_snaps" in capsys.readouterr().out


def test_backfill_cli_rejects_a_forward_only_source(offline, lake, capsys):
    cli = _load_cli("backfill_lake")
    assert cli.main(["--seasons", "2020", "--sources", "sleeper_proj_week"]) == 2
    assert _partitions(lake) == []


def test_backfill_cli_reports_an_unreachable_subset_rather_than_success(offline, lake, capsys):
    cli = _load_cli("backfill_lake")
    assert cli.main(["--seasons", "2016-2024", "--sources", "nflverse_depth"]) == 2
    assert _partitions(lake) == []
    assert "backfills from 2025" in capsys.readouterr().err


def test_backfill_cli_defaults_to_the_spec_span():
    cli = _load_cli("backfill_lake")
    assert runner.parse_seasons(cli.DEFAULT_SEASONS) == tuple(range(2016, 2026))


@pytest.mark.parametrize("script", ["collect", "backfill_lake"])
def test_the_scripts_are_runnable_the_way_the_cron_runs_them(script):
    """``python scripts/<x>.py`` — a real subprocess, because the import path differs from pytest's.

    Running a script directly puts ``scripts/`` at the front of ``sys.path``, where ``collect.py``
    shadows the ``collect`` package both scripts import. Every test above loads them by file path
    with only ``src`` on the path and so cannot see it; this one runs them exactly as ticket 6's
    workflow will. ``--help`` exits before any network call.
    """
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / f"{script}.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert "--seasons" in done.stdout or "--mode" in done.stdout

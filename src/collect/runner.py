"""What a scheduled capture and the one-time backfill actually *do* — the logic both runners share.

``scripts/collect.py`` and ``scripts/backfill_lake.py`` are thin argument-parsing shells over this
module. The logic lives here for the same reason ``analysis.snapshot`` exists behind
``scripts/refresh_data.py``: ``src/`` is what the test suite imports, so a runner written inside a
script is a runner that can only be tested by running it against the network.

Three properties are the whole job, and each is a rule the naive version gets wrong:

* **Best-effort, per source.** A collector that raises costs its own capture and nothing else. The
  pre-lock run captures Sleeper's weekly projections, which are *unrecoverable* once the week is
  played — letting an open-meteo timeout or an nflverse release hiccup abort the run would throw
  those away to report a failure nobody is watching for at 22:00 UTC. The run exits non-zero only
  when **every** source failed, which is the signal that something systemic (network, credentials,
  a bad season/week) is wrong rather than one provider being flaky.
* **One capture in memory at a time.** Every task is collected, written and *released* before the
  next is collected — never a list of captures written at the end. Measured on the real 2025 depth
  chart, by far the largest source: 554,215 rows, ~858 MB of peak heap for that one collector. A
  GitHub Actions runner has ~7 GB, so one at a time is comfortable and all seven at once is not.
* **The season schedule is loaded once.** ``vegas_odds`` derives from it, ``weather`` needs it for
  every week of the season, and ``nflverse_schedules`` *is* it. Left to their defaults each of those
  calls ``load_schedules(season)`` itself — 18 re-parses per season, ~180 across a 2016-2025
  backfill. :class:`RunContext` holds one frame per season and hands the same object to all three.

**Provenance: every row gets a ``_backfill`` marker**, ``True`` from :func:`run_backfill` and
``False`` from :func:`run_cadence`. It is not bookkeeping. Some sources are captured *both* live and
by backfill, and the two are not interchangeable: a backfilled ``sleeper_stats_week`` row carries the
player's *today* position and a live one carries the position as of that week (see
``collect.sleeper``'s ``.. warning::``), so ticket 7 needs a per-row answer to "was this observed at
the time, or reconstructed later?". Stamping ``False`` as well as ``True`` is what makes the column
*total* — with the marker only on backfilled rows, a null would mean both "captured live" and
"written before this column existed", and only one of those is safe to train on.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import polars as pl

from data import nflverse as nflverse_data
from store.lake import StorageBackend, write_snapshot

from . import market as collect_market
from . import nflverse as collect_nflverse
from . import sleeper as collect_sleeper
from . import weather as collect_weather
from .base import Collected
from .registry import SOURCES, Source, backfillable_sources, sources_for_cadence

_LOG = logging.getLogger(__name__)

#: Marker column stamped on every row this module writes (see the module docstring).
BACKFILL_COL = "_backfill"

#: The NFL went to an 18-week regular season in 2021. Only used when the schedule is unreachable —
#: normally the weeks come from the season's own schedule, which is already in hand.
_EIGHTEEN_WEEK_FROM = 2021
_WEEKS_BEFORE, _WEEKS_AFTER = 17, 18

#: Sources whose capture does not depend on the season asked for, so a multi-season backfill runs
#: them **once**. Only ``id_crosswalk``: it is ffverse's live player master rather than a season's
#: archive, and the collector files it under the feed's own ``db_season`` — so running it per season
#: would re-download the same master N times and write it to the same partition N times.
_SEASON_INVARIANT: frozenset[str] = frozenset({"id_crosswalk"})

#: ``--seasons`` accepts ``2016-2025``, ``2019``, or any comma-separated mix of the two.
_SEASON_RE = re.compile(r"^(?P<start>\d{4})(?:\s*-\s*(?P<end>\d{4}))?$")

Loader = Callable[..., pl.DataFrame]


# --------------------------------------------------------------------------- run context
@dataclass
class RunContext:
    """Everything a collector needs that is worth *sharing* between collectors.

    Today that is exactly one thing: the season's schedule frame. It is small (272 rows) and three
    sources need it, so it is cached per season and replaced — not accumulated — when the season
    changes. ``load_schedules`` is the single injection point that keeps this module offline-testable.
    """

    load_schedules: Loader | None = None
    _season: int | None = None
    _frame: pl.DataFrame | None = None

    def schedules(self, season: int) -> pl.DataFrame:
        """The season's schedule, loaded at most once per season."""
        season = int(season)
        if self._frame is None or self._season != season:
            self._frame = (self.load_schedules or nflverse_data.load_schedules)(season)
            self._season = season
            _LOG.info("schedules %s: loaded once for vegas_odds + weather + nflverse_schedules",
                      season)
        return self._frame

    def release(self) -> None:
        """Drop the cached frame (between seasons of a backfill)."""
        self._season, self._frame = None, None


def regular_season_weeks(season: int, schedules: pl.DataFrame | None = None) -> tuple[int, ...]:
    """The weeks a capture should walk for ``season``.

    Read off the schedule when it is in hand — free, and exactly right for the season (17 weeks
    through 2020, 18 from 2021, and only the weeks that exist on a partial forward schedule). The
    static fallback is for the case where the schedule could not be loaded at all: a week too many
    costs one empty capture, which ``write_snapshot`` treats as a documented no-op.
    """
    if schedules is not None and schedules.height and "week" in schedules.columns:
        games = schedules
        if "game_type" in games.columns:
            games = games.filter(pl.col("game_type") == "REG")
        weeks = sorted({int(w) for w in games["week"].drop_nulls().to_list()})
        if weeks:
            return tuple(weeks)
    span = _WEEKS_AFTER if int(season) >= _EIGHTEEN_WEEK_FROM else _WEEKS_BEFORE
    return tuple(range(1, span + 1))


# --------------------------------------------------------------------------- collector dispatch
#: ``(season, week, ctx) -> Collected``. ``week`` is the partition week and is ``None`` for the
#: season-partitioned sources.
Collector = Callable[[int, int | None, RunContext], Collected]


def _proj_week(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_sleeper.collect_proj_week(season, int(week or 0))


def _proj_season(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_sleeper.collect_proj_season(season)


def _stats_week(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_sleeper.collect_stats_week(season, int(week or 0))


def _player_week(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_nflverse.collect_player_week(season)


def _snaps(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_nflverse.collect_snaps(season)


def _ff_opportunity(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_nflverse.collect_ff_opportunity(season)


def _injuries(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_nflverse.collect_injuries(season)


def _schedules(season: int, week: int | None, ctx: RunContext) -> Collected:
    # Hand the collector the frame the context already holds, rather than let it load its own.
    return collect_nflverse.collect_schedules(season, load=lambda _season: ctx.schedules(season))


def _depth_charts(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_nflverse.collect_depth_charts(season)


def _id_crosswalk(season: int, week: int | None, ctx: RunContext) -> Collected:
    # Deliberately no season: the crosswalk is a live master, and the collector files it under the
    # feed's own db_season. Passing the run's season would file today's master under 2016.
    return collect_nflverse.collect_id_crosswalk()


def _vegas(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_market.collect_vegas_from_schedules(ctx.schedules(season), season=season)


def _weather(season: int, week: int | None, ctx: RunContext) -> Collected:
    return collect_weather.collect_weather_forecast(season, int(week or 0), ctx.schedules(season))


#: Which function collects each registered source, and whether it is called per week. Week-partitioned
#: sources fan out over the run's weeks; the rest are one capture per season.
#: ``tests/test_collect_runner.py`` pins this table against the registry, so a source added there
#: without a collector fails a test rather than going quietly uncaptured.
COLLECTORS: dict[str, tuple[bool, Collector]] = {
    # --- Sleeper: fetched a week at a time, so these land in week partitions -------------------
    "sleeper_proj_week": (True, _proj_week),
    "sleeper_proj_season": (False, _proj_season),
    "sleeper_stats_week": (True, _stats_week),
    # --- nflverse: the loaders return a whole season at once -----------------------------------
    "nflverse_player_week": (False, _player_week),
    "nflverse_snaps": (False, _snaps),
    "nflverse_ff_opp": (False, _ff_opportunity),
    "nflverse_injuries": (False, _injuries),
    "nflverse_schedules": (False, _schedules),
    "nflverse_depth": (False, _depth_charts),
    "id_crosswalk": (False, _id_crosswalk),
    # --- derived --------------------------------------------------------------------------------
    "vegas_odds": (False, _vegas),
    "weather": (True, _weather),
}


# --------------------------------------------------------------------------- tasks & results
@dataclass(frozen=True)
class Task:
    """One capture waiting to happen. ``collect`` is a thunk so nothing is fetched at plan time."""

    source: str
    season: int
    week: int | None
    collect: Callable[[], Collected]


@dataclass(frozen=True)
class CaptureResult:
    """What one task did. Holds counts and a path — never the rows, which are released on write."""

    source: str
    season: int
    week: int | None
    rows: int = 0
    path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def plan_tasks(
    sources: Sequence[Source], season: int, weeks: Sequence[int], ctx: RunContext
) -> list[Task]:
    """One task per capture, in registry order — week-partitioned sources fanned out over ``weeks``.

    A registered source with no entry in :data:`COLLECTORS` is skipped with a warning rather than
    raising: a half-landed future ticket must not be able to cost a pre-lock run the Sleeper
    projections it was really there for. The pinning test makes sure that stays hypothetical.
    """
    tasks: list[Task] = []
    for source in sources:
        entry = COLLECTORS.get(source.name)
        if entry is None:
            _LOG.warning(
                "%s is registered but has no collector — skipping it. Add one to "
                "collect.runner.COLLECTORS.", source.name,
            )
            continue
        per_week, collect = entry
        if per_week:
            tasks.extend(
                Task(source.name, season, int(week), partial(collect, season, int(week), ctx))
                for week in weeks
            )
        else:
            tasks.append(Task(source.name, season, None, partial(collect, season, None, ctx)))
    return tasks


def _persist(
    capture: Collected, *, captured_at: str, backfill: bool, backend: StorageBackend | None
) -> Path:
    """Stamp the provenance marker and hand the capture to the store.

    Season and week come from the :class:`~collect.base.Collected` envelope rather than from the
    task, because they are not always the same thing: ``collect_id_crosswalk`` files the master
    under the feed's own ``db_season``, whatever season the run asked about.

    The marker is written into the row dicts **in place**. The envelope owns those dicts
    (``Collected.for_source`` copied them on the way in) and is discarded immediately after this
    call, so mutating is safe — and it avoids rebuilding a 554k-row list to add one column.
    """
    for row in capture.rows:
        row[BACKFILL_COL] = backfill
    return write_snapshot(
        capture.source,
        capture.season,
        capture.rows,
        captured_at=captured_at,
        week=capture.week,
        key_cols=capture.key_cols,
        backend=backend,
    )


def run_tasks(
    tasks: Iterable[Task],
    *,
    captured_at: str,
    backfill: bool = False,
    backend: StorageBackend | None = None,
) -> list[CaptureResult]:
    """Collect → write → release, one task at a time; a failure costs only its own task.

    The release is the point: ``capture`` goes out of scope before the next task is collected, so
    peak memory is one capture rather than the sum of them (see the module docstring).
    """
    results: list[CaptureResult] = []
    for task in tasks:
        capture: Collected | None = None
        try:
            capture = task.collect()
            path = _persist(capture, captured_at=captured_at, backfill=backfill, backend=backend)
            # An empty capture is a documented no-op in the store, so it reports no path: there is
            # no partition to point at, and saying otherwise would read as "wrote 0 rows over it".
            results.append(
                CaptureResult(
                    task.source,
                    capture.season,
                    capture.week,
                    len(capture.rows),
                    path if capture.rows else None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - best-effort by design; the run reports it
            _LOG.exception(
                "%s season=%s week=%s: capture failed (%s) — continuing with the other sources",
                task.source, task.season, task.week, type(exc).__name__,
            )
            results.append(
                CaptureResult(
                    task.source, task.season, task.week, error=f"{type(exc).__name__}: {exc}"
                )
            )
        finally:
            # Released *here*, in every path, so the next task is collected with this one already
            # gone — the 858 MB depth capture must never overlap the one after it.
            capture = None
    return results


# --------------------------------------------------------------------------- the two runs
def _selected(sources: Sequence[Source], only: Iterable[str] | None) -> list[Source]:
    if only is None:
        return list(sources)
    wanted = {name.strip() for name in only if name and name.strip()}
    return [s for s in sources if s.name in wanted]


def run_cadence(
    mode: str,
    season: int,
    week: int,
    *,
    captured_at: str,
    sources: Iterable[str] | None = None,
    backend: StorageBackend | None = None,
    ctx: RunContext | None = None,
) -> list[CaptureResult]:
    """Run every source whose cadence includes ``mode`` (``prelock`` / ``postgame``).

    Registry order is capture order, and it is not arbitrary: the forward-only Sleeper projections
    come first, so a run that dies halfway has still saved the rows nothing can recover.
    """
    ctx = ctx or RunContext()
    selected = _selected(sources_for_cadence(mode), sources)
    tasks = plan_tasks(selected, int(season), [int(week)], ctx)
    _LOG.info(
        "%s capture: season %s week %s — %d source(s), %d task(s)",
        mode, season, week, len(selected), len(tasks),
    )
    return run_tasks(tasks, captured_at=captured_at, backfill=False, backend=backend)


def run_backfill(
    seasons: Sequence[int],
    *,
    captured_at: str,
    sources: Iterable[str] | None = None,
    backend: StorageBackend | None = None,
    ctx: RunContext | None = None,
) -> list[CaptureResult]:
    """Pull every backfillable source for ``seasons`` once, marked ``_backfill=True``.

    Per season: load the schedule (once), take the weeks from it, then walk the sources whose
    history actually reaches that season — ``Source.backfills_season``, which is what keeps
    ``nflverse_depth`` out of 2016-2024 instead of writing nine empty partitions there.
    """
    ctx = ctx or RunContext()
    results: list[CaptureResult] = []
    done_invariant: set[str] = set()

    for season in (int(s) for s in seasons):
        try:
            schedules = ctx.schedules(season)
        except Exception as exc:  # noqa: BLE001 - a missing schedule must not skip the season
            _LOG.warning(
                "schedules %s could not be loaded (%s: %s) — falling back to a static week range; "
                "vegas_odds and weather will report their own failure",
                season, type(exc).__name__, exc,
            )
            schedules = None
        weeks = regular_season_weeks(season, schedules)

        selected = [
            s for s in _selected(backfillable_sources(season), sources)
            if s.name not in done_invariant
        ]
        done_invariant |= {s.name for s in selected if s.name in _SEASON_INVARIANT}
        tasks = plan_tasks(selected, season, weeks, ctx)
        _LOG.info(
            "backfill %s: %d source(s), weeks %s-%s, %d task(s)",
            season, len(selected), weeks[0] if weeks else "-", weeks[-1] if weeks else "-",
            len(tasks),
        )
        results.extend(
            run_tasks(tasks, captured_at=captured_at, backfill=True, backend=backend)
        )
        ctx.release()
    return results


# --------------------------------------------------------------------------- CLI helpers
@dataclass(frozen=True)
class RunPlan:
    """What a scheduled run resolved to: the season/week to capture, or why it should no-op."""

    season: int
    week: int
    skip: str | None = None


def plan_run(
    state: Mapping | None, *, season: int | None = None, week: int | None = None
) -> RunPlan:
    """Resolve season/week from Sleeper's state, or say why the run should skip.

    Off-season handling is ``analysis.snapshot.offseason_skip_reason`` — the same check the weekly
    ``refresh.yml`` no-ops on, so the two crons agree about when the season is over. It is passed no
    league: a capture is league-agnostic (nothing here is scored, and the lake serves every season),
    so the league-rollover fail-safe that matters for the dashboard would only stop a perfectly good
    capture here.

    An explicit ``--season``/``--week`` always runs (that is how a missed week is re-captured).
    Without one, an unreachable state raises: guessing the week would file a capture under the wrong
    partition, and for the forward-only sources a wrong partition is a lost week.
    """
    # Imported here, not at module scope: analysis.snapshot pulls in the whole Phase 3-5 stack
    # (optimizer, waivers, pulp), and a collection run has no use for any of it.
    from analysis.snapshot import offseason_skip_reason

    state = state or {}
    if not state and season is None and week is None:
        raise ValueError(
            "could not determine the current season/week (Sleeper state unreachable) — aborting "
            "the scheduled capture; pass --season/--week to run anyway"
        )
    reason = offseason_skip_reason(state, week, season)
    resolved_season = int(season if season is not None else state.get("season") or 0)
    resolved_week = int(week if week is not None else max(1, int(state.get("week") or 0)))
    if reason is None and resolved_season <= 0:
        raise ValueError(
            f"Sleeper state carries no usable season ({state.get('season')!r}) — pass --season "
            "rather than capture into a season=0 partition"
        )
    return RunPlan(resolved_season, resolved_week, reason)


def parse_seasons(text: str) -> tuple[int, ...]:
    """``"2016-2025"`` / ``"2019"`` / ``"2016,2018-2020"`` -> a sorted, de-duplicated tuple."""
    seasons: set[int] = set()
    for part in str(text).split(","):
        chunk = part.strip()
        if not chunk:
            continue
        match = _SEASON_RE.match(chunk)
        if not match:
            raise ValueError(f"cannot read {chunk!r} as a season or a YYYY-YYYY range")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start:
            raise ValueError(f"season range {chunk!r} ends before it starts")
        seasons.update(range(start, end + 1))
    if not seasons:
        raise ValueError("no seasons given")
    return tuple(sorted(seasons))


def parse_sources(text: str | None, *, backfill: bool = False) -> tuple[str, ...] | None:
    """``"a,b"`` -> validated source names, or ``None`` for "all of them".

    An unknown or non-backfillable name is rejected up front rather than silently collecting
    nothing — a typo in a one-time backfill that reports success is expensive to notice.
    """
    if text is None:
        return None
    names = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not names:
        return None
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; known: {sorted(SOURCES)}")
    if backfill:
        forward_only = [n for n in names if not SOURCES[n].backfillable]
        if forward_only:
            raise ValueError(
                f"source(s) {forward_only} are forward-only — Sleeper serves only the latest "
                "values, so there is no history to backfill"
            )
    return names


def exit_code(results: Sequence[CaptureResult]) -> int:
    """``1`` only when every capture failed — one flaky provider is not a failed run."""
    if not results:
        return 0
    return 1 if all(not r.ok for r in results) else 0


def format_summary(results: Sequence[CaptureResult]) -> str:
    """A per-source summary: partitions written, rows, and the path (or the first failure)."""
    if not results:
        return "  (no sources to collect)"

    order: list[str] = []
    grouped: dict[str, list[CaptureResult]] = {}
    for result in results:
        if result.source not in grouped:
            order.append(result.source)
            grouped[result.source] = []
        grouped[result.source].append(result)

    width = max(len(name) for name in order)
    lines: list[str] = []
    total_rows, failed = 0, 0
    for name in order:
        captures = grouped[name]
        good = [c for c in captures if c.ok]
        bad = [c for c in captures if not c.ok]
        rows = sum(c.rows for c in good)
        total_rows += rows
        failed += len(bad)
        paths = [c.path for c in good if c.path is not None]
        if not paths:
            where = "nothing written"
        else:
            where = str(paths[0]) if len(paths) == 1 else f"{len(paths)} partitions"
        detail = f"{rows:>9,} rows  {where}" if good else "nothing written"
        if bad:
            detail += f"  [FAILED {len(bad)}/{len(captures)}: {bad[0].error}]"
        lines.append(f"  {name:<{width}}  {detail}")

    ok_sources = sum(1 for name in order if any(c.ok for c in grouped[name]))
    lines.append(
        f"  -> {ok_sources}/{len(order)} source(s) captured, {total_rows:,} rows"
        + (f", {failed} capture(s) failed" if failed else "")
    )
    return "\n".join(lines)

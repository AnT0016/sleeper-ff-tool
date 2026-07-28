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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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

#: A game counts as *finished*, not merely kicked off, once this many hours have passed since its
#: kickoff. An NFL game runs ~3.5h; the margin keeps the completed-week rule (see
#: :func:`latest_completed_week`) correct if the postgame cron time moves or a game is flexed later
#: in the slate.
_POSTGAME_SETTLE_HOURS = 6

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
    #: ``repr=False`` for the same reason ``Collected.rows`` has it: this is a 272-row frame, and a
    #: context that renders its payload turns any log line or traceback into a data dump.
    _frame: pl.DataFrame | None = field(default=None, repr=False)

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
    # ``ctx.schedules`` *is* a loader (``season -> frame``), so it goes straight in: a lambda that
    # closed over ``season`` and ignored its argument would silently win any disagreement with the
    # collector about which season it is loading.
    return collect_nflverse.collect_schedules(season, load=ctx.schedules)


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
    """What one task did. Holds counts and a locator — never the rows, released on write.

    ``path`` is whatever the active backend calls the partition it wrote: a filesystem
    :class:`~pathlib.Path` locally, an ``s3://bucket/key`` string on the cloud backend (an S3 URI
    is not expressible as a ``Path`` — the ``//`` collapses). Only ``str()`` it; anything
    path-shaped is local-backend territory.
    """

    source: str
    season: int
    week: int | None
    rows: int = 0
    path: Path | str | None = None
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
) -> Path | str:
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

    Per season: pick the sources whose history actually reaches it (``Source.backfills_season``,
    which is what keeps ``nflverse_depth`` out of 2016-2024 instead of writing nine empty partitions
    there), then load the schedule *once* and take the weeks from it.

    The selection happens **before** the schedule load so a season with nothing to collect costs no
    download at all, and a ``sources`` subset that reaches none of ``seasons`` raises instead of
    reporting a successful run that wrote nothing.
    """
    ctx = ctx or RunContext()
    results: list[CaptureResult] = []
    done_invariant: set[str] = set()
    planned_any = False

    for season in (int(s) for s in seasons):
        selected = [
            s for s in _selected(backfillable_sources(season), sources)
            if s.name not in done_invariant
        ]
        if not selected:
            _LOG.info("backfill %s: nothing left to collect for this season - skipping it", season)
            continue
        planned_any = True

        # The eager load exists *only* to fan the week-partitioned sources out over real weeks:
        # ``_schedules`` / ``_vegas`` / ``_weather`` all call ``ctx.schedules`` inside their own
        # thunk, so a season with no per-week source in the selection needs no schedule at plan
        # time — and downloading one anyway is ten wasted requests on a targeted backfill.
        weeks: tuple[int, ...] = ()
        if any(COLLECTORS[s.name][0] for s in selected if s.name in COLLECTORS):
            try:
                schedules = ctx.schedules(season)
            except Exception as exc:  # noqa: BLE001 - a missing schedule must not skip the season
                _LOG.warning(
                    "schedules %s could not be loaded (%s: %s) - falling back to a static week "
                    "range; vegas_odds and weather will report their own failure",
                    season, type(exc).__name__, exc,
                )
                schedules = None
            weeks = regular_season_weeks(season, schedules)

        tasks = plan_tasks(selected, season, weeks, ctx)
        _LOG.info(
            "backfill %s: %d source(s), weeks %s-%s, %d task(s)",
            season, len(selected), weeks[0] if weeks else "-", weeks[-1] if weeks else "-",
            len(tasks),
        )
        season_results = run_tasks(tasks, captured_at=captured_at, backfill=True, backend=backend)
        results.extend(season_results)
        # Marked done from the *results*, not from the plan: a season-invariant source that failed
        # has not been collected, and skipping it for every later season would mean one transient
        # error on 2016 costs a ten-season run the crosswalk entirely — silently, since the other
        # sources succeed and the run exits 0.
        done_invariant |= {
            r.source for r in season_results if r.ok and r.source in _SEASON_INVARIANT
        }
        ctx.release()

    if not planned_any and sources is not None:
        raise ValueError(_unreachable_message(sources, seasons))
    return results


def _unreachable_message(sources: Iterable[str], seasons: Sequence[int]) -> str:
    """Why a ``--sources``/``--seasons`` pair collects nothing — the silent-success trap.

    ``parse_sources`` validates the *names* and rejects forward-only ones, but it knows nothing about
    seasons, so ``--sources nflverse_depth --seasons 2016-2024`` passed every check and then wrote
    nothing while reporting success. A one-time backfill that says it worked is expensive to
    disbelieve, so say exactly which source starts when.
    """
    wanted = [n for n in sources]
    spans = ", ".join(
        f"{n} backfills from {SOURCES[n].backfillable_from}"
        for n in wanted
        if n in SOURCES and SOURCES[n].backfillable_from is not None
    )
    span = f"{min(seasons)}-{max(seasons)}" if seasons else "(no seasons)"
    return (
        f"none of the requested source(s) {wanted} can be backfilled for season(s) {span}"
        + (f" - {spans}" if spans else "")
        + "; refusing to report a successful run that collected nothing"
    )


# --------------------------------------------------------------------------- completed-week resolver
def _ensure_utc(now: datetime | None) -> datetime:
    """A tz-aware UTC ``now``; ``None`` reads the wall clock.

    A **naive** datetime raises rather than being read as UTC. Coercing looks harmless and is not:
    ``datetime.now()`` in this project's own timezone (CEST) is two hours ahead of UTC, and two hours
    is enough to settle a week whose last game is still being played. The one caller that matters
    passes an aware UTC ``now`` down from ``scripts/collect.py``, so a naive one is a programming
    error and should read as one.
    """
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError(
            f"naive datetime {now!r}: the completed-week resolver needs a tz-aware moment, and "
            "reading a local clock as UTC can settle a week that is still being played"
        )
    return now.astimezone(timezone.utc)


def _week_statuses(schedules: pl.DataFrame | None, now: datetime) -> list[tuple[int, bool, str]]:
    """Per REG week, ascending: ``(week, finished, why_not)``.

    ``finished`` is ``True`` only when **every** game of the week has a readable kickoff and the
    latest of them settled (kickoff + :data:`_POSTGAME_SETTLE_HOURS`) at or before ``now``. A week
    with any unreadable kickoff is reported unfinished (``why_not`` says how many) rather than
    dropped — see :func:`latest_completed_week` for why that matters. An empty/columnless schedule
    yields ``[]``.
    """
    now = _ensure_utc(now)
    if schedules is None or not schedules.height:
        return []
    if not {"week", "gameday", "gametime"} <= set(schedules.columns):
        return []
    games = schedules
    if "game_type" in games.columns:
        games = games.filter(pl.col("game_type") == "REG")
    if not games.height:
        return []

    settle = timedelta(hours=_POSTGAME_SETTLE_HOURS)
    statuses: list[tuple[int, bool, str]] = []
    for week in sorted({int(w) for w in games["week"].drop_nulls().to_list()}):
        kicks = [
            collect_weather.kickoff_utc(gameday, gametime)
            for gameday, gametime in games.filter(pl.col("week") == week)
            .select(["gameday", "gametime"])
            .iter_rows()
        ]
        unreadable = sum(1 for k in kicks if k is None)
        if unreadable or not kicks:
            statuses.append(
                (week, False, f"{unreadable}/{len(kicks)} game(s) have no readable kickoff time")
            )
            continue
        settled_at = max(kicks) + settle
        if settled_at <= now:
            statuses.append((week, True, ""))
        else:
            statuses.append((
                week,
                False,
                f"latest game settles ~{settled_at.strftime('%Y-%m-%dT%H:%MZ')}, "
                f"after now {now.strftime('%Y-%m-%dT%H:%MZ')}",
            ))
    return statuses


def _finished_week(statuses: Sequence[tuple[int, bool, str]]) -> int | None:
    """The highest week reported finished, or ``None``. Shared so the resolver the runner uses and
    the one :func:`latest_completed_week` documents cannot drift apart."""
    return max((week for week, ok, _ in statuses if ok), default=None)


def latest_completed_week(schedules: pl.DataFrame | None, now: datetime) -> int | None:
    """Highest REG week whose games have all *finished* by ``now`` — the postgame capture week.

    Why a kickoff-time rule and not a result-based one ("the week is done when every game has a
    score"): a result-based rule asks *has nflverse published the scores yet*, which makes the
    capture week a function of a third party's publishing latency — the same kind of data-arrival
    race this ticket exists to remove, just moved from Sleeper's clock to nflverse's. Kickoff times
    are fixed when the schedule is published, so they answer the question without asking anyone
    whether they have finished writing. The caching layer sharpens the point: ``data.nflverse`` runs
    ``nflreadpy`` with a 24h filesystem cache, so a local Tuesday run can be handed a frame fetched
    before Sunday's games — correct kickoff times, stale null results. (A cold GitHub Actions runner
    starts with an empty cache today, but that stops being true the moment ticket 6 caches
    ``data_cache/nflverse_cache`` to save bandwidth.)

    *Finished*, not merely *kicked off*: a game counts only once ``kickoff + _POSTGAME_SETTLE_HOURS
    <= now``, because an NFL game runs ~3.5h and the rule has to stay correct if the cron time moves
    or a game is flexed later in the day. A week with **any** game whose kickoff cannot be read
    (nflverse leaves ``gametime`` blank on unflexed late-season games) is treated as *not* finished
    — dropping those unknowns before taking the max would silently mark an unstarted week complete.

    Resolving the *highest* finished week can skip a week when a single game is postponed past the
    cron (a Monday game moved to Tuesday). That is acceptable: ``sleeper_stats_week`` is
    backfillable, and every other postgame source is season-grain — it carries its own ``week``
    column and is re-captured whole — so the postponed week's rows land on a later run regardless.

    ``None`` when no REG week has finished yet (start of season) or the schedule cannot be read.
    """
    return _finished_week(_week_statuses(schedules, now))


# --------------------------------------------------------------------------- CLI helpers
@dataclass(frozen=True)
class RunPlan:
    """What a scheduled run resolved to: the season/week to capture, or why it should no-op."""

    season: int
    week: int
    skip: str | None = None


def plan_run(
    state: Mapping | None,
    *,
    mode: str = "prelock",
    now: datetime | None = None,
    season: int | None = None,
    week: int | None = None,
    ctx: RunContext | None = None,
) -> RunPlan:
    """Resolve season/week for a scheduled capture, or say why the run should skip.

    **Mode-aware week resolution.** ``prelock`` keeps Sleeper's ``state.week`` — the *upcoming*
    week, which is exactly what a pre-lock snapshot wants. ``postgame`` wants the opposite (the week
    that just finished) and must not use ``state.week`` at all: Sleeper advances it to the upcoming
    week early on Tuesday, racing the Tue postgame cron, so a run landing after the flip would file a
    zeroed not-yet-played snapshot into ``week=N+1`` and never capture week ``N``. Instead it derives
    the completed week from the season schedule via :func:`latest_completed_week` — a fact about the
    NFL, not about Sleeper's internal clock. ``ctx`` supplies the schedule (loaded at most once per
    run and shared with :func:`run_cadence`'s ``nflverse_schedules`` collector).

    Off-season handling (``analysis.snapshot.offseason_skip_reason``, the same check ``refresh.yml``
    no-ops on) is evaluated **first and returns without touching the schedule** — an off-season run
    must not load next season's schedule, which is often unpublished and would warn. It is passed no
    league: a capture is league-agnostic, so the dashboard's league-rollover fail-safe would only
    stop a perfectly good capture here.

    An explicit ``--season``/``--week`` always runs and overrides everything, schedule included (that
    is how a missed week is re-captured). Without one, an unreachable state raises. A postgame run
    whose completed week cannot be resolved (schedule unavailable, or no week finished yet) **skips**
    (exit 0, green cron, logged) rather than falling back to ``state.week``: that fallback is the
    upcoming week — the very defect this resolves — and every postgame source is backfillable, so a
    missed run is recovered with ``--week``, whereas a wrong ``week=N+1`` write is silent contamination.
    """
    # Imported here, not at module scope: analysis.snapshot pulls in the whole Phase 3-5 stack
    # (optimizer, waivers, pulp), and a collection run has no use for any of it.
    from analysis.snapshot import offseason_skip_reason

    state = state or {}
    if not state and season is None and week is None:
        raise ValueError(
            "could not determine the current season/week (Sleeper state unreachable) - aborting "
            "the scheduled capture; pass --season/--week to run anyway"
        )
    resolved_season = int(season if season is not None else state.get("season") or 0)
    fallback_week = int(week) if week is not None else max(1, int(state.get("week") or 0))

    reason = offseason_skip_reason(state, week, season)
    if reason is not None:
        # Off-season: return before any schedule work — next season's schedule may not exist yet.
        return RunPlan(resolved_season, fallback_week, reason)
    if resolved_season <= 0:
        raise ValueError(
            f"Sleeper state carries no usable season ({state.get('season')!r}) - pass --season "
            "rather than capture into a season=0 partition"
        )

    if week is not None:
        _LOG.info("%s plan: season %s week %s (explicit override)", mode, resolved_season, week)
        return RunPlan(resolved_season, int(week), None)
    if mode == "postgame":
        return _postgame_plan(resolved_season, now, ctx, fallback_week)
    _LOG.info(
        "prelock plan: season %s week %s (upcoming week, from Sleeper state)",
        resolved_season, fallback_week,
    )
    return RunPlan(resolved_season, fallback_week, None)


def _postgame_plan(
    season: int, now: datetime | None, ctx: RunContext | None, fallback_week: int
) -> RunPlan:
    """Resolve the *completed* week for a postgame capture, or a skip reason (never ``state.week``).

    The schedule comes from ``ctx`` so the load is shared with the run's ``nflverse_schedules``
    collector. A load failure or a season with no finished week is a **skip**, not a guess: see
    :func:`plan_run` for why falling back to the upcoming week would be the defect this exists to fix.

    Two things are logged, and the second is the one that is easy to leave out. Resolving the
    *highest* finished week means an earlier week can be stepped over — a game postponed past the
    cron, or an ``gametime`` nflverse had not filled in on the Tuesday that week was current. Once
    the resolver has moved past it that week is finished too, so it never appears in the
    "next week rejected" line and the run reads as clean while its live ``sleeper_stats_week``
    capture was never taken. A week below the resolved one is therefore reported on its own, as a
    WARNING, because it is actionable (``--week N`` recovers it) — and it fires only when there is
    actually a gap, so a normal run stays silent.
    """
    now = _ensure_utc(now)
    try:
        schedules = (ctx or RunContext()).schedules(season)
    except Exception as exc:  # noqa: BLE001 - a missing schedule must skip, never guess the week
        skip = (
            f"the {season} schedule is unavailable ({type(exc).__name__}: {exc}), so the completed "
            "week cannot be resolved; every postgame source is backfillable - re-run with --week "
            "once the schedule is published"
        )
        _LOG.warning("postgame plan skipped: %s", skip)
        return RunPlan(season, fallback_week, skip)

    statuses = _week_statuses(schedules, now)
    resolved = _finished_week(statuses)
    if resolved is None:
        why = statuses[0][2] if statuses else "the schedule carries no REG games"
        skip = (
            f"no {season} REG week has finished as of {now.strftime('%Y-%m-%dT%H:%MZ')} ({why}); "
            "nothing to capture postgame yet"
        )
        _LOG.info("postgame plan skipped: %s", skip)
        return RunPlan(season, fallback_week, skip)

    rejected = next(
        (f"week {week} rejected - {why}" for week, ok, why in statuses if not ok and week > resolved),
        "no later week is scheduled",
    )
    _LOG.info(
        "postgame plan: season %s -> completed week %s (from the schedule, not Sleeper state); "
        "next %s", season, resolved, rejected,
    )
    # Weeks the resolver stepped over. Silent in the normal case; a WARNING when there is a gap,
    # because nothing else in the run would ever mention it (see the docstring).
    stepped_over = [week for week, ok, _ in statuses if not ok and week < resolved]
    if stepped_over:
        _LOG.warning(
            "postgame plan: season %s REG week(s) %s are below the resolved week %s, so they were "
            "never captured live - re-run with --week to recover each of them (%s)",
            season, stepped_over, resolved,
            "; ".join(f"week {w}: {why}" for w, ok, why in statuses if not ok and w < resolved),
        )
    return RunPlan(season, resolved, None)


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


def _is_forward_only(source: str) -> bool:
    """Is a lost capture of ``source`` lost *for good*? (An unregistered name is not.)"""
    entry = SOURCES.get(source)
    return entry is not None and not entry.backfillable


def exit_code(results: Sequence[CaptureResult]) -> int:
    """``1`` when every capture failed, **or** when a forward-only one did.

    One flaky provider is not a failed run — that is what makes the best-effort loop worth having.
    But a forward-only source is different in kind: Sleeper's projection endpoints serve only the
    latest numbers, so a pre-lock capture that does not happen is gone and no backfill recovers it.
    Under an all-or-nothing rule that is exactly the failure the exit status cannot express: six
    sources succeed, ``sleeper_proj_week`` raises, the cron is green, and the week nobody can rebuild
    is the week nobody was told about. GitHub Actions surfaces red runs and nothing else, so the one
    permanent loss has to be able to turn the run red on its own.

    Recoverable failures stay green *and stay visible* — :func:`format_summary` reports every one of
    them per source, and :func:`run_tasks` logs each with a traceback.
    """
    if not results:
        return 0
    if all(not r.ok for r in results):
        return 1
    lost = sorted({r.source for r in results if not r.ok and _is_forward_only(r.source)})
    if lost:
        _LOG.error(
            "forward-only source(s) %s failed to capture - those rows are unrecoverable (the "
            "endpoints serve only the latest values), so this run is a failure even though the "
            "other sources succeeded",
            lost,
        )
        return 1
    return 0


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

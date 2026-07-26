"""Point-in-time capture of the nflverse feeds: the label, the usage features, the market fields.

Where ``collect.sleeper`` captures numbers that are *unrecoverable* once a week is played, these are
the **backfillable** half of the lake — nflverse publishes them as versioned releases, so 2016-2025
can be pulled in one go. That makes them the training set: ``nflverse_player_week`` is the source of
the label (re-scored week-N actuals), and snaps / opportunity / injuries / depth are the usage and
role features. ``nflverse_schedules`` carries the closing Vegas lines and the observed temp/wind, so
tickets 4 and 7 read the market and weather columns straight off it.

Every collector here is a thin wrapper over the corresponding ``data.nflverse`` loader (which owns
the download cache) plus three pieces of hygiene, and nothing else:

* **Nothing is renamed and nothing is re-scored.** Rows go into the lake with the provider's own
  column names and values, so ``ids.nflverse_to_sleeper_stats`` can re-score them downstream in
  whatever ``scoring_settings`` the league has at the time. That includes the provider's quirks:
  ``ff_opportunity`` types ``season`` as a *string* and ``week`` as a *float*, and most feeds carry a
  ``season``/``week`` column that duplicates the store's ``_season``/``_week``. Left alone on
  purpose — a raw layer that has been tidied is a raw layer you can no longer audit against the
  source. (Contrast ``collect.sleeper``, which *builds* a row out of a nested payload and therefore
  has to choose what goes in it.)
* **Temporal columns become ISO-8601 UTC strings.** ``Collected.rows`` is ``list[dict]`` of
  json-safe values, and ``injuries.date_modified`` — a ``Datetime(time_zone="UTC")`` that is *part of
  the key* — would otherwise be a ``datetime`` object whose equality depends on the tzinfo it came
  back with. Rendered as ``2024-09-06T19:05:30Z``, the same spelling the depth feed already uses for
  ``dt``, it is stable across a parquet round-trip and sorts chronologically as a plain string.
  Verified lossless on 2024: no ``date_modified`` carries a sub-second component.
* **Rows the registry key cannot identify are filtered to the source's grain**, before
  :func:`collect.base.dedupe_rows` ever sees them (see :func:`_identified` for why that is a grain
  filter and not a swept-under defect) — quietly up to a per-source rate calibrated on real
  releases, and as a **WARNING** above it, so a provider change cannot hide behind the routine
  residual.

**Keys come from** :data:`collect.registry.SOURCES` **and are never written here** — the store dedups
on whatever key it is handed, so a collector inventing its own would delete real rows on merge.
:class:`collect.base.Collected` enforces the agreement at construction.

Partitioning: nflverse loaders return a **whole season** in one call, so every source here is
season-partitioned (``week=None``) with week-grain rows carrying their own ``week`` column. Only
Sleeper's endpoints, fetched a week at a time, land in week partitions.

.. warning::

   **``nflverse_depth`` is effectively 2025-forward.** nflverse replaced the depth-chart feed for
   2025: it is now ``dt``/``team``/``gsis_id``/``pos_abb``/``pos_rank`` (an ESPN-style timestamped
   snapshot, no ``season`` or ``week`` column), where 2001-2024 is
   ``season``/``club_code``/``week``/…/``depth_position``. The registry key is the modern one, and it
   is exact there — unique on all 548,638 identifiable 2025 rows. The legacy shape cannot be stored
   under it (none of ``dt``/``team``/``pos_abb`` exist), and it has **no clean natural key of its
   own** either: the best seven-column candidate still leaves 207 duplicate rows in 2024, so storing
   it would mean logging a capture-integrity warning on every backfill run and silently dropping
   rows. :func:`collect_depth_charts` therefore returns an **empty** capture for a legacy-schema
   season and says so — an empty capture is a documented no-op in ``store.write_snapshot``, so a
   multi-season backfill neither aborts nor blanks a good partition.

   Consequence for ticket 7: role/depth features exist from 2025 on. As-of resolution should join on
   ``dt`` (strictly finer than a week) rather than a week label — which is also why this collector
   does not synthesise a ``week`` column for the modern feed. Deriving one means an as-of join
   against ``nflverse_schedules``, and that belongs in the assembler, not baked into the raw layer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import polars as pl

from data import nflverse

from .base import Collected, dedupe_rows
from .registry import SOURCES

_LOG = logging.getLogger(__name__)

#: How many offending keys a filter log names before truncating (mirrors ``collect.base``).
_SAMPLE_KEYS = 3

#: Per-source ceiling on the share of rows the registry key cannot address. At or below it the
#: filter is routine provider grain (INFO); above it something has changed and it escalates to
#: WARNING. Calibrated on 2016-2025 releases with headroom for normal variation:
#:
#: ===========================  ========  =======
#: source                       measured  ceiling
#: ===========================  ========  =======
#: ``nflverse_player_week``     0.12%     1%
#: ``nflverse_ff_opp``          3.7-7.0%  12%
#: ``nflverse_depth``           1.0%      3%
#: everything else              0%        0%
#: ===========================  ========  =======
#:
#: A source with no known residual gets no allowance on purpose: the first unidentifiable row it
#: ever produces is already an anomaly worth a warning (this is what would catch, say, a release
#: that broke ``gsis_id`` on the injury report).
_GRAIN_FILTER_CEILING: dict[str, float] = {
    "nflverse_player_week": 0.01,
    "nflverse_ff_opp": 0.12,
    "nflverse_depth": 0.03,
}
_DEFAULT_CEILING = 0.0

#: ISO-8601 UTC, seconds precision — the spelling the modern depth feed already uses for ``dt``.
_ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

#: Columns unique to the pre-2025 depth-chart schema. Presence of either means the provider handed
#: us the legacy feed, which the registry key cannot address (see the module ``.. warning::``).
_LEGACY_DEPTH_COLS: tuple[str, ...] = ("club_code", "depth_position")

Loader = Callable[..., pl.DataFrame]


def _iso_utc(frame: pl.DataFrame) -> pl.DataFrame:
    """Temporal columns as ISO-8601 strings — json-safe, and stable as a key value.

    A tz-aware column is converted to UTC first, so two spellings of one instant render identically
    (the same normalization ``store.lake.normalize_captured_at`` applies to provenance). A naive
    column is read as UTC, matching that function's rule.

    A temporal dtype with no rendering here raises rather than passing through: ``Collected.rows``
    promises json-safe scalars, and the alternative is a ``timedelta`` sitting in the lake that only
    surfaces when something downstream tries to serialize it. Every source here is backfillable, so
    a re-run after teaching this function the new dtype loses nothing.
    """
    casts = []
    for name, dtype in frame.schema.items():
        if dtype == pl.Date:
            casts.append(pl.col(name).dt.strftime("%Y-%m-%d"))
        elif isinstance(dtype, pl.Datetime):
            column = pl.col(name)
            if dtype.time_zone is not None:
                column = column.dt.convert_time_zone("UTC")
            casts.append(column.dt.strftime(_ISO_UTC))
        elif dtype.is_temporal():
            raise ValueError(
                f"column {name!r} has temporal dtype {dtype} with no ISO rendering; "
                "Collected.rows must hold json-safe scalars"
            )
    return frame.with_columns(casts) if casts else frame


def _usable_key(frame: pl.DataFrame, key_cols: Sequence[str]) -> pl.Series:
    """Mask of rows whose every key column holds a value that can identify them.

    Mirrors ``collect.base._is_missing`` in polars: null, NaN and blank/whitespace strings are all
    unusable, because the store folds them together when it dedups.
    """
    checks = []
    for name in key_cols:
        dtype = frame.schema[name]
        check = pl.col(name).is_not_null()
        if dtype == pl.String:
            check = check & (pl.col(name).str.strip_chars() != "")
        elif dtype.is_float():
            check = check & pl.col(name).is_not_nan()
        checks.append(check)
    return frame.select(pl.all_horizontal(checks).alias("ok"))["ok"].fill_null(False)


def _rate_is_judgeable(height: int, ceiling: float) -> bool:
    """Whether a drop *rate* means anything on a frame this size.

    A ceiling only says something once the frame is large enough for it to permit at least one row.
    On a 23-row test sample a 1% ceiling is 0.23 rows, so a single unattributable line reads as 8.7%
    and the comparison is noise rather than signal — which is exactly what the enriched fixtures are
    (they over-represent the awkward shapes on purpose). Real captures are nowhere near this bound:
    the smallest non-zero-ceiling source is ``ff_opp`` at ~5.6k rows a season.

    Zero-ceiling sources never take this path: with no residual expected they are judged on count,
    so one bad row warns however small the frame.
    """
    return height * ceiling >= 1


def _identified(frame: pl.DataFrame, key_cols: Sequence[str], source: str) -> pl.DataFrame:
    """``frame`` minus the rows its registry key cannot address, logged with a count.

    This is a **grain filter, not a defect swept under the rug**, and the distinction is the reason
    it lives here rather than being left to ``collect.base.dedupe_rows``. nflverse's player-grain
    feeds carry a residual line per team-game for production it could not attribute to a player:
    21-22 rows a season in ``player_week`` (null name and position, zero fantasy points, only
    penalty and safety yardage) and 202-419 in ``ff_opportunity`` (unattributed expected points),
    plus 5,577 unnamed 2025 depth-chart entries with no ``gsis_id``. They are real provider output
    at a *team* grain, not broken player rows — and no key on a player-grain source can address them.

    Letting ``dedupe_rows``/``write_snapshot`` warn about them instead would fire a defect warning on
    every single run of every backfill, which is precisely how an operator learns to ignore the
    warning that matters. Filtering them here keeps those warnings meaning "something is wrong",
    while the count below keeps the loss auditable.

    Nothing recoverable is lost for ``ff_opportunity``: the team aggregate lives on every player row
    of that team-game in the ``*_team`` columns, so the unattributed residual is
    ``total_fantasy_points_exp_team`` minus the sum over the player rows.

    **Quiet only up to a known rate.** Demoting this to INFO is the right trade for a residual whose
    size is known, but "202 rows nflverse could not attribute" and "the ``gsis_id`` column broke in
    this release" would otherwise produce the same-shaped, equally quiet line — and at 50% that trade
    is plainly wrong. Above :data:`_GRAIN_FILTER_CEILING` for the source it escalates to WARNING and
    names the rate, so a provider change reports itself instead of surfacing seasons later as a join
    failure in the assembler.
    """
    keys = list(key_cols)
    usable = _usable_key(frame, keys)
    n_dropped = int((~usable).sum())
    if not n_dropped:
        return frame

    share = n_dropped / frame.height
    ceiling = _GRAIN_FILTER_CEILING.get(source, _DEFAULT_CEILING)
    sample = frame.filter(~usable).select(keys).head(_SAMPLE_KEYS).rows()
    if ceiling <= 0:
        over = True  # no residual expected here, and n_dropped > 0 by the guard above
    else:
        over = share > ceiling and _rate_is_judgeable(frame.height, ceiling)
    if over:
        _LOG.warning(
            "%s: filtered %d/%d row(s) (%.1f%%) with no usable value in key column(s) %s - above "
            "this source's expected residual of %.1f%%, so this is a provider change rather than "
            "routine unattributed production. Sample: %s",
            source, n_dropped, frame.height, share * 100, keys, ceiling * 100, sample,
        )
    else:
        _LOG.info(
            "%s: filtered %d/%d row(s) (%.2f%%) with no usable value in key column(s) %s - they "
            "are not rows at this source's grain and cannot be identified. Sample: %s",
            source, n_dropped, frame.height, share * 100, keys, sample,
        )
    return frame.filter(usable)


def _collect(source: str, season: int, frame: pl.DataFrame) -> Collected:
    """One loader frame -> one season-partitioned capture, keyed by the registry."""
    key_cols = SOURCES[source].key_cols
    missing = [c for c in key_cols if c not in frame.columns]
    if missing:
        # Raise rather than degrade: a provider schema change that removes a key column must not
        # quietly produce a capture keyed on whatever is left, which the store would then merge
        # against the existing partition and dedup wrongly.
        raise ValueError(
            f"{source}: key column(s) {missing} absent from the nflverse frame "
            f"(columns: {sorted(frame.columns)[:20]}); the provider schema has changed"
        )
    rows = dedupe_rows(_identified(_iso_utc(frame), key_cols, source).to_dicts(), key_cols,
                       source=source)
    return Collected.for_source(source, season, rows, week=None)


def collect_player_week(season: int, *, load: Loader | None = None) -> Collected:
    """``nflverse_player_week``: weekly player actuals — **the label source**.

    ``season_type`` is in the key because postseason weeks reuse the regular season's numbering.
    Per-*player* only: team DST aggregates are not in this feed, which is what
    ``sleeper_stats_week`` is the cross-check for.
    """
    return _collect(
        "nflverse_player_week", season, (load or nflverse.load_weekly_actuals)(season)
    )


def collect_snaps(season: int, *, load: Loader | None = None) -> Collected:
    """``nflverse_snaps``: weekly snap counts and shares (Pro Football Reference).

    PFR-keyed — there is no ``gsis_id`` in this feed, so ``ids.build_id_to_sleeper(cw, "pfr_id")``
    is the join downstream. ``game_id`` pins the week *and* the opponent, so it completes the key.
    """
    return _collect("nflverse_snaps", season, (load or nflverse.load_snap_counts)(season))


def collect_ff_opportunity(season: int, *, load: Loader | None = None) -> Collected:
    """``nflverse_ff_opp``: expected points and volume shares (ffopportunity).

    The spec's one caveat applies downstream, not here: expected points are computed from a week's
    *actual* usage, so they are a same-week quantity that leaks if used as a pre-game feature —
    legal only lagged. Captured raw; the assembler enforces the lag.

    Per-team rows for unattributed production carry a null ``player_id`` and are filtered by
    :func:`_identified`; their content remains derivable from the ``*_team`` columns.
    """
    return _collect("nflverse_ff_opp", season, (load or nflverse.load_ff_opportunity)(season))


def collect_injuries(season: int, *, load: Loader | None = None) -> Collected:
    """``nflverse_injuries``: the weekly practice/game-status report.

    ``date_modified`` is in the key because the report is a *revision stream*, not a weekly fact: a
    player is routinely listed twice in one week (Questionable after Wednesday's practice, Out after
    Friday's), and a key without it collapses the two into whichever the provider happened to list
    last — persisting a stale status with no warning. Verified on real 2024 data: 6,215/6,215 rows
    unique with it, 2 collisions without.

    Secondary to Sleeper's ``injury_status`` for start/sit (CLAUDE.md), but this one is genuinely
    point-in-time and backfillable, which the Sleeper master is not.
    """
    return _collect("nflverse_injuries", season, (load or nflverse.load_injuries)(season))


def collect_schedules(season: int, *, load: Loader | None = None) -> Collected:
    """``nflverse_schedules``: one row per game — and the free Vegas + weather feed.

    Game-grain, so ``game_id`` alone is the key. Carries ``spread_line`` / ``total_line`` /
    moneylines (ticket 4 derives implied team totals from these) and ``roof`` / ``temp`` / ``wind``
    (the historical weather side, where the open-meteo forecast is the pre-lock side).
    """
    return _collect("nflverse_schedules", season, (load or nflverse.load_schedules)(season))


def collect_depth_charts(season: int, *, load: Loader | None = None) -> Collected:
    """``nflverse_depth``: timestamped depth-chart snapshots — **2025 forward only**.

    A legacy-schema season (2001-2024) yields an empty capture and a warning rather than a wrong
    one; see the module ``.. warning::`` for why the legacy feed cannot be keyed at all. Unnamed
    entries with no ``gsis_id`` (5,577 of 554,215 in 2025) are filtered; the remainder is unique on
    the registry key.
    """
    frame = (load or nflverse.load_depth_charts)(season)
    legacy = [c for c in _LEGACY_DEPTH_COLS if c in frame.columns]
    if legacy:
        _LOG.warning(
            "nflverse_depth season=%s: the provider returned the pre-2025 depth-chart schema "
            "(saw %s), which carries none of the registry key %s and has no clean key of its own. "
            "Captured nothing - depth/role features start at 2025.",
            season, legacy, list(SOURCES["nflverse_depth"].key_cols),
        )
        return Collected.for_source("nflverse_depth", season, [], week=None)
    return _collect("nflverse_depth", season, frame)


def collect_id_crosswalk(
    season: int | None = None, *, load: Loader | None = None
) -> Collected:
    """``id_crosswalk``: ffverse's player-id master (season-grain, ``week=None``).

    Keyed on ``mfl_id`` — ffverse's own primary key, and the only id column that is complete
    (``sleeper_id`` and ``gsis_id`` are both nullable, which is exactly why the crosswalk exists).

    The feed is a live master rather than a season's archive, so the partition season comes from its
    own ``db_season`` column when not given: that is ffverse's statement of which season the file
    *is*, and reading it beats reading the clock (a backfill re-run in a later season would
    otherwise file the identical master under a different season).
    """
    frame = (load or nflverse.load_id_crosswalk)()
    if season is None:
        season = _db_season(frame)
    return _collect("id_crosswalk", season, frame)


def _db_season(frame: pl.DataFrame) -> int:
    """The crosswalk's own season stamp; the newest one if the release ever mixes them."""
    if "db_season" not in frame.columns:
        raise ValueError(
            "id_crosswalk: no db_season column to take the partition season from — "
            "pass season= explicitly"
        )
    stamps = frame["db_season"].drop_nulls().unique().sort().to_list()
    if not stamps:
        raise ValueError("id_crosswalk: db_season is empty — pass season= explicitly")
    if len(stamps) > 1:
        _LOG.info("id_crosswalk: db_season spans %s — filing the whole master under %s",
                  stamps, stamps[-1])
    return int(stamps[-1])

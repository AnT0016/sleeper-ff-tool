"""The player x week training frame, assembled from the lake under a strict lookahead guard.

This is the hand-off point between Phase 8 (collect and store) and the modelling phase: one row per
``(sleeper player_id, season, week)``, carrying a **real re-scored label**, the baseline to beat, and
features that were all knowable before that week locked. The lake never stores a join precisely so
that this module can be the single auditable place where "what was known when" is decided.

Reading order, if you only read one thing: :func:`lookahead_ok` is the gate, and
:func:`_resolved_known_at` is why the gate can be applied to a lake whose historical rows all carry
one 2026 capture stamp.

The gate
--------
A feature row is legal for target week *N* when **either** holds:

* **content rule** — the row is about a strictly earlier week (``feature_week < N``). A backfilled
  week-3 actual is legal as a week-5 feature however late it was captured.
* **capture rule** — the row was known strictly before week *N*'s lock (``known_at < lock_utc``).
  This is what admits a genuine pre-lock snapshot of week *N* itself.

``lock_utc`` is the **first kickoff of the target week**, one lock per ``(season, week)``, exactly as
the spec words it ("week N's first game lock"). A week-level lock is deliberately stricter than a
per-player one: it means no same-week capture can carry a post-kickoff fact about *any* game that
week, which collapses a whole family of leaks into one rule.

Resolved availability, and why ``_captured_at`` alone is not enough
------------------------------------------------------------------
Every 2016-2025 row in the lake was written by one backfill run, so its ``_captured_at`` is a 2026
instant. Applied literally, the capture rule then admits **nothing** same-week for the entire
training span — no Vegas implied total, no injury report, no depth chart — and the frame collapses
to lagged usage. That is not the guard being strict; it is the guard being asked a question
``_captured_at`` cannot answer.

So a row's *known-at* instant is resolved per source from
:attr:`collect.registry.Source.content_known`, and only for rows carrying ``_backfill=True``:

===================  ==========================================================================
``content_known``    resolved ``known_at`` for a backfilled row
===================  ==========================================================================
``pre_kickoff``      its own week's lock, minus :data:`_RESOLVED_LEAD` — the content existed
                     before that lock by construction (a practice report, a closing line)
``post_game``        never (``NaT``) — the content did not exist until the week was played, so
                     only the content rule can admit it
``row_timestamp``    the row's own event stamp (``nflverse_depth.dt``), a real as-of time
===================  ==========================================================================

Live rows (``_backfill=False``) always use ``_captured_at``; a missing ``_backfill`` marker is read
as live, which is the restrictive direction.

``content_known`` is **not** ``cadence``. ``nflverse_schedules`` runs on the pre-lock cadence *and*
carries ``result``/``home_score``/``away_score``/``total``, so deriving knowability from the capture
schedule would hand this module the label. Schedules is therefore registered ``post_game`` and is
used here as a **calendar only** — kickoff times, which team plays whom, and the resulting locks.
Its sanctioned pre-game view is ``vegas_odds``, which carries the lines and deliberately not the
outcome (see ``collect.market``).

Weather: what a backfilled row actually contributes
--------------------------------------------------
Nothing numeric. Measured over the whole populated lake, ``forecast_*`` is **0% populated**
historically (2,639 rows, zero forecasts) — open-meteo's forecast endpoint reaches back about 92
days, so a 2016-2025 backfill was always going to be null there. And ``observed_*`` is nflverse's
**at-kickoff** measurement, which is post-kickoff data for the target week and is correctly rejected
by :func:`observed_weather_ok`. What survives for the training seasons is the venue: the three-state
``is_indoor`` and the roof/stadium resolution behind it, both fixed well before kickoff.

``observed_*`` is still emitted (structurally all-null under ``asof="prelock"``) rather than dropped,
so the rule is visible and tested instead of implicit-by-omission. The guard is per row and needs no
join: ``observed_* is post-kickoff data <=> _captured_at >= kickoff_utc``. A ``_backfill`` flag
cannot make this call — the pre-lock cron runs Thursday **and** Sunday into the same week partition,
so the Sunday row is a legitimate pre-lock capture that nonetheless carries the already-played
Thursday game's real temperature. Where ``kickoff_utc`` is null (``weather_status == 'no_kickoff'``)
the guard falls back to the coarse rule: legal only for a week strictly later than the row's own.

``is_indoor`` is three-state throughout: ``True`` / ``False`` / ``<NA>``, where ``<NA>`` is a
retractable venue whose roof state is unrecorded. Those rows do carry a forecast (outdoor conditions
are still the informative prior) but must not be read as confirmed open-air, so the null is
preserved rather than filled.

Position, and the two structural discontinuities
------------------------------------------------
``position`` is resolved **as-of** from ``nflverse_depth`` (``dt`` is a real timestamp, ``pos_abb``
the value) via a backward as-of join on ``dt < lock``. Sleeper's ``position`` and
``fantasy_positions`` both come from a mutable player master and return *today's* value for every
season asked about — Taysom Hill reads TE for 2018, Cordarrelle Patterson reads RB — and nflverse's
weekly ``position`` is a static per-player attribute too, so neither is a fallback. Where depth has
no coverage (it starts at 2025) the static label is used and the row is flagged
``position_is_static=True`` so a consumer can exclude it. That matters most for a breakout
classifier, where a WR->RB conversion *is* the role step-up being predicted, so a current position
leaks the label rather than blurring a feature. DST rows are exempt: their position follows from the
row's own team-abbreviation key, not from a mutable master, so they are flagged ``False``.

Two discontinuities are structural and are left visible rather than papered over:

* **Baseline.** ``baseline_sleeper_points`` exists only from 2026 W1, because Sleeper's projection
  endpoints serve only the latest values and no ex-ante historical projection is freely recoverable.
  Training on 2016-2025 needs no projection; the market-beating grade is a forward, out-of-sample one.
* **Injury.** ``inj_report_*`` (nflverse, ``gsis_id``-keyed, official practice report) covers
  2016-2025; ``inj_sleeper_*`` exists 2026-forward only. They are **not** interchangeable — a live
  Sleeper capture returns PUP / NA / IR / DNR states the NFL report has no equivalent for — so they
  are kept as separate columns and never coalesced.

The label is a union, because ``nflverse_player_week`` has **zero** DEF rows and DEF is a starting
slot: skill players and kickers score from nflverse actuals through ``ids.nflverse_to_sleeper_stats``,
team defences score from ``sleeper_stats_week``, which is already Sleeper-keyed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import pandas as pd
import polars as pl

from collect.registry import SOURCES
from collect.weather import kickoff_utc as _schedule_kickoff
from data.ids import NFLVERSE_STAT_COLS, build_id_to_sleeper, nflverse_to_sleeper_stats
from scoring.engine import points
from store.lake import StorageBackend, read_source

_LOG = logging.getLogger(__name__)

#: The frame's natural key.
KEY_COLS: tuple[str, ...] = ("player_id", "season", "week")

#: Provenance every read needs. ``_backfill`` is ticket 5's marker; a partition written without it
#: reads back as NA and is treated as a live capture (the restrictive branch).
_PROVENANCE: tuple[str, ...] = ("_season", "_week", "_captured_at", "_backfill")

#: How far before its own week's lock a backfilled ``pre_kickoff`` row is stamped. Only the ordering
#: matters — the margin exists so the single strict-``<`` comparison in :func:`lookahead_ok` can
#: express "known before that lock" without a second code path.
_RESOLVED_LEAD = pd.Timedelta(seconds=1)

#: Span of the exponentially-weighted usage averages, in prior appearances.
_EWMA_SPAN = 4

#: Per-source ceiling on the share of rows the crosswalk cannot map to a Sleeper id (see
#: :func:`_map_ids`). Calibrated on the populated 2016-2025 lake, with headroom:
#:
#: ========================  ===========  =======
#: source                    measured     ceiling
#: ========================  ===========  =======
#: ``nflverse_player_week``  4.5-6.4%     10%
#: ``nflverse_snaps``        17.2-20.2%   25%
#: ``nflverse_ff_opp``       0.07-0.48%   2%
#: ========================  ===========  =======
#:
#: Snaps is the outlier by design rather than by defect: it lists every player who took a snap, and
#: Sleeper has no id for an offensive lineman. A source with no entry gets no allowance, so the
#: first unmappable row it ever produces warns.
_UNJOINED_CEILING: dict[str, float] = {
    "nflverse_player_week": 0.10,
    "nflverse_snaps": 0.25,
    "nflverse_ff_opp": 0.02,
}
_DEFAULT_UNJOINED_CEILING = 0.0

#: Ceiling on the share of label rows with no scheduled game (see :func:`_attach_calendar`).
#: Measured 4/169,689 = 0.002% on 2016-2025 — two abandoned games' worth — so 0.1% is generous and
#: still tight enough that a broken calendar join reports itself instead of looking routine.
_ORPHAN_LABEL_CEILING = 0.001

#: Franchise codes normalized to one vocabulary before anything joins on a team.
#:
#: Two feeds that both call themselves nflverse disagree: ``nflverse_schedules`` spells a team the
#: way it was spelled **that season** (``SD`` in 2016, ``OAK`` through 2019) while
#: ``nflverse_player_week`` applies today's codes retroactively (``LAC``, ``LV``). Joining them
#: as-is silently loses every Chargers and Raiders player-week of those seasons — 796 rows over
#: 2016-2025, each keeping its label and quietly losing its market and weather features. Sleeper
#: adds a third spelling (``LAR`` for the Rams, where nflverse says ``LA``).
#:
#: Modern nflverse codes win, because that is what the label source already uses.
_TEAM_ALIASES: dict[str, str] = {
    "LAR": "LA",   # Sleeper's Rams
    "STL": "LA",   # pre-2016 Rams; before the lake's span, kept so a wider backfill stays correct
    "SD": "LAC",   # San Diego Chargers, as nflverse_schedules spells 2016
    "OAK": "LV",   # Oakland Raiders, as nflverse_schedules spells 2016-2019
}

#: Depth-chart position codes worth resolving a fantasy position from, and the one rename needed
#: (the feed spells kicker ``PK``).
_DEPTH_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "FB", "WR", "TE", "PK"})
_DEPTH_POSITION_RENAME: dict[str, str] = {"PK": "K"}

#: Per-week quantities the lagged usage features are built from: ``column -> output prefix``.
_USAGE_COLS: dict[str, str] = {
    "pts": "points",
    "snap_pct": "snap_pct",
    "target_share": "target_share",
    "rush_share": "rush_share",
    "exp_points": "exp_points",
}
#: Usage features that also get a trend (the last lag minus the one before it).
_TRENDED: frozenset[str] = frozenset({"points", "snap_pct"})

#: Identity and label first, features after, so the frame reads top-down.
_LEADING_COLS: tuple[str, ...] = (
    *KEY_COLS,
    "position",
    "position_is_static",
    "team",
    "opponent",
    "is_home",
    "game_id",
    "gsis_id",
    "is_dst",
    "y_custom_points",
    "baseline_sleeper_points",
)


# --------------------------------------------------------------------------- time helpers
def _as_utc(value: Any) -> pd.Timestamp | None:
    """One timestamp as tz-aware UTC, or ``None`` when it cannot be read as one."""
    if value is None or value is pd.NaT:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


#: One resolution for every timestamp in this module. pandas infers a unit per column (a stamp with
#: microseconds parses as ``us``, one without as ``s``), and ``merge_asof`` refuses to join two
#: tz-aware keys whose units differ — so the as-of position join fails on nothing more than how the
#: provider happened to spell a time. Normalizing on read makes that unrepresentable.
_UTC = "datetime64[ns, UTC]"


def _utc_series(values: pd.Series) -> pd.Series:
    """A column as tz-aware UTC datetimes at one fixed resolution; unreadable entries become ``NaT``."""
    return pd.to_datetime(values, utc=True, errors="coerce", format="mixed").astype(_UTC)


def _int_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64")


# --------------------------------------------------------------------------- the gate
def lookahead_ok(
    feature_week: int | None,
    feature_captured_at: str | None,
    target_week: int,
    lock_utc: str,
) -> bool:
    """May a feature row be used for ``target_week``? The one gate, applied to every source.

    ``True`` when the row is about a strictly earlier week, **or** when it was known strictly before
    the target week's lock. Anything else — including an unreadable or missing stamp on a row that
    is not about an earlier week — is refused: the gate fails closed, because a leak is invisible in
    the output and a dropped feature is not.

    ``feature_captured_at`` is the row's **resolved** known-at instant, which is not always its raw
    ``_captured_at`` (see the module docstring). ``feature_week`` is ``None`` for season-grain and
    timestamped rows, which the content rule then cannot admit — only the capture rule can.
    """
    if feature_week is not None and not pd.isna(feature_week):
        if int(feature_week) < int(target_week):
            return True
    known = _as_utc(feature_captured_at)
    lock = _as_utc(lock_utc)
    if known is None or lock is None:
        return False
    return known < lock


def _admissible(
    feature_week: pd.Series, known_at: pd.Series, target_week: pd.Series, lock: pd.Series
) -> pd.Series:
    """Vectorised :func:`lookahead_ok`. Pinned against the scalar version by the test suite."""
    weeks = pd.to_numeric(pd.Series(feature_week).reset_index(drop=True), errors="coerce")
    targets = pd.to_numeric(pd.Series(target_week).reset_index(drop=True), errors="coerce")
    known = pd.Series(known_at).reset_index(drop=True)
    locks = pd.Series(lock).reset_index(drop=True)
    content = weeks.notna() & targets.notna() & (weeks.fillna(0) < targets.fillna(0))
    capture = known.notna() & locks.notna() & (known < locks)
    out = (content.fillna(False) | capture.fillna(False)).astype(bool)
    out.index = pd.Series(feature_week).index
    return out


def observed_weather_ok(
    captured_at: Any, kickoff: Any, feature_week: int | None, target_week: int
) -> bool:
    """May a row's ``observed_*`` weather be used for ``target_week``?

    ``observed_*`` is nflverse's at-kickoff measurement, so its event time is the **game's kickoff**
    rather than the row's capture. It is usable only when it is not post-kickoff data relative to
    the target: either the row is about a strictly earlier week, or it was captured before that game
    kicked off (in which case it holds no measurement anyway).

    Where ``kickoff`` is unreadable (``weather_status == 'no_kickoff'``) the fine rule cannot be
    evaluated, so it falls back to the coarse one: legal only for a week strictly later than the
    row's own.
    """
    if feature_week is not None and not pd.isna(feature_week):
        if int(feature_week) < int(target_week):
            return True
    stamp, kick = _as_utc(captured_at), _as_utc(kickoff)
    if stamp is None or kick is None:
        return False
    return stamp < kick


# --------------------------------------------------------------------------- lake reads
def _read(
    source: str,
    seasons: Sequence[int] | None,
    columns: Sequence[str],
    *,
    backend: StorageBackend | None,
) -> pd.DataFrame:
    """One source, projected to ``columns`` plus provenance. Empty frame when never captured."""
    frame = read_source(source, seasons, columns=[*columns, *_PROVENANCE], backend=backend)
    if frame.empty:
        return pd.DataFrame(columns=[*columns, *_PROVENANCE])
    if "_backfill" not in frame.columns:
        frame = frame.assign(_backfill=pd.NA)
    return frame


def _lock_for(
    frame: pd.DataFrame, locks: pd.Series, *, season_col: str, week_col: str | None
) -> pd.Series:
    """The first-kickoff lock for each row's ``(season, week)``, aligned to ``frame``'s index."""
    if week_col is None or week_col not in frame.columns or frame.empty:
        return pd.Series(pd.NaT, index=frame.index, dtype=_UTC)
    left = pd.DataFrame(
        {"season": _int_series(frame[season_col]), "week": _int_series(frame[week_col])}
    )
    merged = left.merge(locks.rename("lock").reset_index(), on=["season", "week"], how="left")
    return pd.Series(merged["lock"].to_numpy(), index=frame.index).astype(_UTC)


def _resolved_known_at(
    frame: pd.DataFrame, source: str, locks: pd.Series, *, week_col: str | None
) -> pd.Series:
    """When each row's content became knowable — see the table in the module docstring."""
    known = SOURCES[source].content_known
    if known == "row_timestamp":
        # The row states its own event time, true whether it was captured live or backfilled.
        return _utc_series(frame["dt"])

    captured = _utc_series(frame["_captured_at"])
    backfilled = frame["_backfill"].fillna(False).astype(bool)
    if known == "post_game":
        return captured.where(~backfilled)  # backfilled -> NaT: only the content rule can admit it
    own_lock = _lock_for(frame, locks, season_col="_season", week_col=week_col)
    return captured.where(~backfilled, own_lock - _RESOLVED_LEAD)


def _latest(frame: pd.DataFrame, keys: Sequence[str], order: Sequence[str]) -> pd.DataFrame:
    """One row per ``keys`` — the last after sorting by ``order`` (stable)."""
    if frame.empty:
        return frame
    return (
        frame.sort_values([c for c in order if c in frame.columns], kind="stable")
        .drop_duplicates(subset=list(keys), keep="last")
        .reset_index(drop=True)
    )


def _latest_capture(frame: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    """Collapse a multi-partition read to its newest capture per row — **partition-scoped**.

    A registry ``key_cols`` identifies a row *within its partition*, which is all the store ever
    needs. Here it is not enough, and the failure is silent: ``read_source`` concatenates ten
    seasons, and ``nflverse_player_week``'s key ``(player_id, season_type, week)`` then makes one
    player's 2016 week 1 and 2017 week 1 look like the same row — 35k rows collapsing to 25.6k on
    two seasons alone, with no warning anywhere, because superseding an older capture is exactly
    what the store is supposed to do. Prepending the partition's own ``_season``/``_week`` restores
    the grain the key was written against.
    """
    scope = [c for c in ("_season", "_week") if c in frame.columns]
    return _latest(frame, [*scope, *key_cols], ["_captured_at"])


def _blank(columns: Sequence[str], index: pd.Index) -> pd.DataFrame:
    """An all-null feature block, so an absent source costs columns their values, not their existence."""
    return pd.DataFrame({c: pd.Series(pd.NA, index=index, dtype="object") for c in columns})


# --------------------------------------------------------------------------- calendar
def _calendar(
    seasons: Sequence[int], *, backend: StorageBackend | None
) -> tuple[pd.DataFrame, pd.Series]:
    """``(team_games, locks)`` from ``nflverse_schedules`` — **calendar only**, never features.

    ``team_games`` is one row per ``(season, week, team)``: which game, against whom, home or away.
    ``locks`` is the first kickoff of each ``(season, week)``. Every column read here is fixed when
    the schedule is published, so none of it can leak; the outcome columns sitting beside them in the
    same feed are exactly why this source is registered ``post_game`` and is read for nothing else.
    """
    frame = _read(
        "nflverse_schedules",
        seasons,
        ["game_id", "season", "game_type", "week", "gameday", "gametime", "away_team", "home_team"],
        backend=backend,
    )
    if frame.empty:
        raise ValueError(
            "nflverse_schedules is empty for the requested seasons — the assembler needs it for the "
            "week locks and the team/game calendar. Run scripts/backfill_lake.py first."
        )

    games = _latest_capture(frame, ["game_id"])
    games = games.loc[games["game_type"].astype("string") == "REG"].copy()
    games["season"] = _int_series(games["season"])
    games["week"] = _int_series(games["week"])
    games["kickoff_utc"] = _utc_series(
        pd.Series(
            [
                _as_utc(_schedule_kickoff(day, clock))
                for day, clock in zip(games["gameday"], games["gametime"], strict=True)
            ],
            index=games.index,
            dtype="object",
        )
    )

    unreadable = int(games["kickoff_utc"].isna().sum())
    if unreadable:
        _LOG.warning(
            "calendar: %d/%d REG game(s) have no readable kickoff (gameday/gametime) — they set no "
            "lock and their players get no game context", unreadable, len(games),
        )

    locks = games.groupby(["season", "week"], dropna=True)["kickoff_utc"].min()
    _LOG.info(
        "calendar: %d game(s) over %d season-week(s), locks %s..%s",
        len(games), len(locks), locks.min(), locks.max(),
    )

    sides = []
    for own, other, home in (("home_team", "away_team", True), ("away_team", "home_team", False)):
        side = games[["season", "week", "game_id", own, other]].rename(
            columns={own: "team", other: "opponent"}
        )
        side["team"] = _canonical_team(side["team"])
        side["opponent"] = _canonical_team(side["opponent"])
        side["is_home"] = home
        sides.append(side)
    team_games = pd.concat(sides, ignore_index=True)
    team_games["team"] = team_games["team"].astype("string")
    team_games["opponent"] = team_games["opponent"].astype("string")
    return team_games, locks


# --------------------------------------------------------------------------- identity
def _crosswalk(*, backend: StorageBackend | None) -> pd.DataFrame:
    """The ffverse id master, latest capture — the only bridge from nflverse ids to Sleeper's.

    Read across every season rather than the requested ones: it is a live master filed under the
    feed's own ``db_season``, so limiting it to the seasons being assembled would find nothing.
    """
    frame = read_source(
        "id_crosswalk",
        None,
        columns=["mfl_id", "gsis_id", "sleeper_id", "pfr_id", *_PROVENANCE],
        backend=backend,
    )
    if frame.empty:
        raise ValueError(
            "id_crosswalk is empty — every nflverse row reaches Sleeper's id space through it, so "
            "the frame cannot be assembled. Run scripts/backfill_lake.py first."
        )
    return _latest_capture(frame, ["mfl_id"])


def _canonical_id(values: pd.Series) -> pd.Series:
    """Ids as the strings Sleeper actually uses.

    ffverse stores ``sleeper_id`` as a **float** (``13269.0``), so a naive cast produces
    ``"13269.0"`` — which joins against nothing Sleeper ever emits. Silent, too: the crosswalk maps
    fine, the frame builds, and only the baseline column comes back empty in 2026. The trailing
    ``.0`` is stripped here, at the one place ids cross from ffverse's space into Sleeper's.
    """
    text = values.astype("string").str.strip()
    return text.str.replace(r"\.0$", "", regex=True).replace("", pd.NA)


def _id_map(crosswalk: pd.DataFrame, column: str) -> dict[str, str]:
    """``<column> -> sleeper_id``, via the Phase 1 helper (which owns the null handling)."""
    pairs = pd.DataFrame(
        {
            column: _canonical_id(crosswalk[column]),
            "sleeper_id": _canonical_id(crosswalk["sleeper_id"]),
        }
    )
    return build_id_to_sleeper(pl.from_pandas(pairs), column)


def _map_ids(
    frame: pd.DataFrame, source_col: str, mapping: Mapping[str, str], what: str
) -> pd.DataFrame:
    """Attach ``player_id`` from ``source_col``; rows that do not join are logged and dropped.

    Quiet only up to a known rate, the same trade ``collect.nflverse._identified`` makes. These
    feeds legitimately cover players Sleeper has no id for — ``nflverse_snaps`` lists every player
    who took a snap, offensive line included — so a residual is grain, not a defect, and warning
    about it on every run is how an operator learns to skim the warning that matters. Above the
    per-source ceiling it escalates, because "the crosswalk lags a rookie class" and "``gsis_id``
    broke in this release" must not read the same.
    """
    out = frame.copy()
    out["player_id"] = out[source_col].astype("string").map(mapping).astype("string")
    unjoined = out["player_id"].isna()
    n_dropped = int(unjoined.sum())
    if n_dropped:
        share = n_dropped / len(out)
        sample = sorted(out.loc[unjoined, source_col].dropna().astype(str).unique())[:5]
        ceiling = _UNJOINED_CEILING.get(what, _DEFAULT_UNJOINED_CEILING)
        args = (what, n_dropped, len(out), share * 100, source_col, sample)
        if share > ceiling:
            _LOG.warning(
                "%s: %d/%d row(s) (%.1f%%) have no sleeper_id for %s and are dropped — above this "
                "source's expected residual, so the crosswalk or the feed has changed. Sample: %s",
                *args,
            )
        else:
            _LOG.info(
                "%s: %d/%d row(s) (%.1f%%) have no sleeper_id for %s and are dropped — the "
                "crosswalk does not cover them. Sample: %s",
                *args,
            )
    return out.loc[~unjoined].reset_index(drop=True)


# --------------------------------------------------------------------------- label
def _score_rows(records: Sequence[Mapping[str, Any]], scoring: Mapping[str, float]) -> list[float]:
    """Re-score Sleeper-keyed stat rows with the Phase 1 engine, dropping nulls first.

    The lake stores a flat row per player, so a stat the player has no value for reads back as
    ``NaN`` rather than an absent key — and ``NaN * coefficient`` would poison the whole sum.
    """
    keys = list(scoring)
    return [
        points({k: row[k] for k in keys if k in row and pd.notna(row[k])}, scoring)
        for row in records
    ]


_LABEL_COLS: tuple[str, ...] = (
    *KEY_COLS, "gsis_id", "source_team", "static_position", "is_dst", "y_custom_points",
    "target_share",
)


def _player_labels(
    seasons: Sequence[int],
    scoring: Mapping[str, float],
    crosswalk: pd.DataFrame,
    *,
    backend: StorageBackend | None,
) -> pd.DataFrame:
    """Skill players and kickers: week-N nflverse actuals re-scored in ``scoring``."""
    frame = _read(
        "nflverse_player_week",
        seasons,
        ["player_id", "season", "week", "season_type", "position", "team", "target_share",
         *NFLVERSE_STAT_COLS],
        backend=backend,
    )
    if frame.empty:
        return pd.DataFrame(columns=list(_LABEL_COLS))

    frame = frame.loc[frame["season_type"].astype("string") == "REG"].copy()
    frame = _latest_capture(frame, ["player_id", "season_type", "week"])
    frame["gsis_id"] = frame["player_id"].astype("string")
    frame = _map_ids(frame, "gsis_id", _id_map(crosswalk, "gsis_id"), "nflverse_player_week")
    if frame.empty:
        return pd.DataFrame(columns=list(_LABEL_COLS))

    # Nulls become zeros *before* translating. ``ids.nflverse_to_sleeper_stats`` is written against
    # a polars row where an unrecorded stat is absent; a lake row is flat, so the same stat reads
    # back as NaN, and NaN is truthy — it would be copied into the Sleeper-keyed dict and poison the
    # whole sum, turning one missing column into a NaN label for every player. Same rule
    # ``_score_rows`` applies on the Sleeper side, and the one ``collect.sleeper`` states in its
    # docstring ("re-scoring must drop nulls before summing").
    stat_cols = [c for c in NFLVERSE_STAT_COLS if c in frame.columns]
    stats = frame[stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame["y_custom_points"] = [
        points(nflverse_to_sleeper_stats(row), scoring) for row in stats.to_dict("records")
    ]
    frame["season"] = _int_series(frame["season"])
    frame["week"] = _int_series(frame["week"])
    frame["source_team"] = _canonical_team(frame["team"])
    frame["static_position"] = frame["position"].astype("string")
    frame["is_dst"] = False
    frame["target_share"] = pd.to_numeric(frame["target_share"], errors="coerce")
    return frame[list(_LABEL_COLS)]


def _dst_labels(
    seasons: Sequence[int], scoring: Mapping[str, float], *, backend: StorageBackend | None
) -> pd.DataFrame:
    """Team defences — ``nflverse_player_week`` has **zero** DEF rows and DEF is a starting slot.

    ``sleeper_stats_week`` is the only registered source with a team-defence aggregate, and it is
    already Sleeper-keyed, so the row scores directly with no stat translation. Its ``player_id``
    *is* the team abbreviation, in Sleeper's era-correct vocabulary.
    """
    frame = _read(
        "sleeper_stats_week",
        seasons,
        ["player_id", "position", "season_type", *sorted(scoring)],
        backend=backend,
    )
    if frame.empty:
        return pd.DataFrame(columns=list(_LABEL_COLS))

    frame = frame.loc[
        (frame["position"].astype("string") == "DEF")
        & (frame["season_type"].astype("string") == "regular")
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=list(_LABEL_COLS))

    frame = _latest_capture(frame, ["player_id"])
    frame["y_custom_points"] = _score_rows(frame.to_dict("records"), scoring)
    frame["player_id"] = frame["player_id"].astype("string")
    frame["season"] = _int_series(frame["_season"])
    frame["week"] = _int_series(frame["_week"])
    frame["gsis_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["source_team"] = _canonical_team(frame["player_id"])
    frame["static_position"] = pd.Series("DEF", index=frame.index, dtype="string")
    frame["is_dst"] = True
    frame["target_share"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return frame[list(_LABEL_COLS)]


def _canonical_team(values: pd.Series) -> pd.Series:
    """Franchise codes in one vocabulary — see :data:`_TEAM_ALIASES`."""
    return values.astype("string").replace(_TEAM_ALIASES)


# --------------------------------------------------------------------------- features
def _positions(
    targets: pd.DataFrame, locks: pd.Series, *, backend: StorageBackend | None
) -> pd.DataFrame:
    """As-of position and depth rank from ``nflverse_depth``: the latest snapshot with ``dt < lock``.

    A backward as-of join on the real timestamp, not on a week label — ``dt`` is strictly finer than
    a week, and the feed carries no week column precisely because deriving one belongs here.
    """
    columns = ["position", "position_is_static", "depth_pos_rank", "depth_dt"]
    out = pd.DataFrame(index=targets.index)
    out["position"] = pd.Series(pd.NA, index=targets.index, dtype="string")
    out["position_is_static"] = True
    out["depth_pos_rank"] = pd.Series(pd.NA, index=targets.index, dtype="Float64")
    out["depth_dt"] = pd.Series(pd.NaT, index=targets.index, dtype=_UTC)

    seasons = sorted({int(s) for s in targets["season"].dropna().unique()})
    depth = _read(
        "nflverse_depth", seasons, ["dt", "team", "gsis_id", "pos_abb", "pos_rank"], backend=backend
    )
    matched_n = 0
    if depth.empty:
        _LOG.info(
            "nflverse_depth: no coverage for season(s) %s — every position falls back to the static "
            "label (the feed starts at 2025)", seasons,
        )
    else:
        depth = depth.loc[depth["pos_abb"].astype("string").isin(_DEPTH_POSITIONS)].copy()
        depth["known_at"] = _resolved_known_at(depth, "nflverse_depth", locks, week_col=None)
        depth["gsis_id"] = depth["gsis_id"].astype("string")
        depth["position"] = depth["pos_abb"].astype("string").replace(_DEPTH_POSITION_RENAME)
        depth["pos_rank"] = pd.to_numeric(depth["pos_rank"], errors="coerce")
        # A player can hold more than one listed position in a single snapshot (156 of 150,569
        # skill player-snapshots in 2025). Highest listing wins, then alphabetical, so it is stable.
        depth = _latest(
            depth.dropna(subset=["known_at", "gsis_id"]),
            ["gsis_id", "known_at"],
            ["pos_rank", "position"],
        ).sort_values("known_at", kind="stable")

        left = pd.DataFrame(
            {
                "gsis_id": targets["gsis_id"].astype("string"),
                "lock": targets["lock"],
                "_row": targets.index,
            }
        ).dropna(subset=["gsis_id", "lock"]).sort_values("lock", kind="stable")

        if not left.empty and not depth.empty:
            asof = pd.merge_asof(
                left,
                depth[["gsis_id", "known_at", "position", "pos_rank"]],
                left_on="lock",
                right_on="known_at",
                by="gsis_id",
                direction="backward",
                allow_exact_matches=False,  # strictly before the lock, like the gate itself
            )
            hit = asof.loc[asof["position"].notna()]
            matched_n = len(hit)
            rows = hit["_row"].to_numpy()
            out.loc[rows, "position"] = hit["position"].to_numpy()
            out.loc[rows, "depth_pos_rank"] = hit["pos_rank"].to_numpy()
            out.loc[rows, "depth_dt"] = hit["known_at"].to_numpy()
            out.loc[rows, "position_is_static"] = False

    # DST is exempt: its position follows from the row's own team-abbreviation key, so it is not a
    # fallback to a mutable player master and must not be flagged as one.
    out.loc[targets["is_dst"].fillna(False).to_numpy(), "position_is_static"] = False

    static = out["position"].isna()
    out.loc[static, "position"] = targets.loc[static, "static_position"].astype("string")
    _LOG.info(
        "position: %d/%d row(s) resolved as-of from nflverse_depth, %d fell back to the static "
        "label (position_is_static=True)", matched_n, len(out), int(out["position_is_static"].sum()),
    )
    return out[columns]


def _market(
    targets: pd.DataFrame, locks: pd.Series, *, backend: StorageBackend | None
) -> pd.DataFrame:
    """Implied team total, spread and total from ``vegas_odds`` — the sanctioned pre-game view."""
    columns = ["implied_team_total", "opp_implied_total", "team_spread_line", "total_line",
               "is_div_game"]
    seasons = sorted({int(s) for s in targets["season"].dropna().unique()})
    frame = _read(
        "vegas_odds",
        seasons,
        ["game_id", "week", "home_implied_total", "away_implied_total", "spread_line",
         "total_line", "div_game"],
        backend=backend,
    )
    if frame.empty:
        return _blank(columns, targets.index)

    frame["known_at"] = _resolved_known_at(frame, "vegas_odds", locks, week_col="week")
    frame["lock"] = _lock_for(frame, locks, season_col="_season", week_col="week")
    frame = frame.loc[_admissible(frame["week"], frame["known_at"], frame["week"], frame["lock"])]
    frame = _latest(frame, ["game_id"], ["known_at"])
    if frame.empty:
        return _blank(columns, targets.index)

    merged = targets[["game_id", "is_home"]].reset_index(drop=True).merge(
        frame[["game_id", "home_implied_total", "away_implied_total", "spread_line", "total_line",
               "div_game"]],
        on="game_id", how="left",
    )
    merged.index = targets.index

    home = merged["is_home"].fillna(False).astype(bool)
    out = pd.DataFrame(index=targets.index)
    out["implied_team_total"] = merged["home_implied_total"].where(home, merged["away_implied_total"])
    out["opp_implied_total"] = merged["away_implied_total"].where(home, merged["home_implied_total"])
    # nflverse's spread is the *home* team's expected margin (collect.market pins the sign), so the
    # away side takes it negated: every row then reads "my team's expected margin".
    spread = pd.to_numeric(merged["spread_line"], errors="coerce")
    out["team_spread_line"] = spread.where(home, -spread)
    out["total_line"] = pd.to_numeric(merged["total_line"], errors="coerce")
    out["is_div_game"] = _int_series(merged["div_game"])
    return out[columns]


def _weather(
    targets: pd.DataFrame, locks: pd.Series, *, backend: StorageBackend | None
) -> pd.DataFrame:
    """Venue and forecast for the target week's game; ``observed_*`` under its own kickoff guard."""
    columns = ["is_indoor", "wx_forecast_temp_f", "wx_forecast_wind_mph",
               "wx_forecast_precip_prob_pct", "wx_forecast_lead_hours", "wx_observed_temp_f",
               "wx_observed_wind_mph"]
    seasons = sorted({int(s) for s in targets["season"].dropna().unique()})
    frame = _read(
        "weather",
        seasons,
        ["game_id", "week", "kickoff_utc", "is_indoor", "forecast_time_utc", "forecast_temp_f",
         "forecast_wind_mph", "forecast_precip_prob_pct", "observed_temp_f", "observed_wind_mph"],
        backend=backend,
    )
    if frame.empty:
        return _blank(columns, targets.index)

    frame["known_at"] = _resolved_known_at(frame, "weather", locks, week_col="week")
    frame["lock"] = _lock_for(frame, locks, season_col="_season", week_col="week")
    frame = frame.loc[
        _admissible(frame["week"], frame["known_at"], frame["week"], frame["lock"])
    ].copy()
    if frame.empty:
        return _blank(columns, targets.index)

    # The observed guard, per row and with no join: at-kickoff data is post-kickoff data for its own
    # week, so it survives only where the row is about an earlier week (never, joining same-week) or
    # was captured before that game kicked off (where it holds nothing). A _backfill flag cannot make
    # this call -- the Sunday pre-lock capture carries the played Thursday game's real weather.
    observed_ok = pd.Series(
        [
            observed_weather_ok(captured, kickoff, week, week)
            for captured, kickoff, week in zip(
                frame["_captured_at"], frame["kickoff_utc"], frame["week"], strict=True
            )
        ],
        index=frame.index,
        dtype=bool,
    )
    withheld = int((~observed_ok & frame["observed_temp_f"].notna()).sum())
    if withheld:
        _LOG.info(
            "weather: %d row(s) carry an at-kickoff observed_* measurement that is post-kickoff data "
            "for their own week — those values are withheld; the venue columns are not", withheld,
        )
    for column in ("observed_temp_f", "observed_wind_mph"):
        frame[column] = frame[column].where(observed_ok)

    # Lead time varies inside one capture (a Thursday run forecasts TNF hours out and MNF four days
    # out), so it is a usable confidence weight rather than a constant.
    frame["forecast_lead_hours"] = (
        _utc_series(frame["forecast_time_utc"]) - _utc_series(frame["_captured_at"])
    ).dt.total_seconds() / 3600.0
    frame = _latest(frame, ["game_id"], ["known_at"])

    merged = targets[["game_id"]].reset_index(drop=True).merge(
        frame[["game_id", "is_indoor", "forecast_temp_f", "forecast_wind_mph",
               "forecast_precip_prob_pct", "forecast_lead_hours", "observed_temp_f",
               "observed_wind_mph"]],
        on="game_id", how="left",
    )
    merged.index = targets.index

    out = pd.DataFrame(index=targets.index)
    # Three-state on purpose: <NA> is a retractable venue whose roof state is unrecorded, which is
    # not the same statement as "open air". Never filled.
    out["is_indoor"] = merged["is_indoor"].astype("boolean")
    for name in ("temp_f", "wind_mph", "precip_prob_pct"):
        out[f"wx_forecast_{name}"] = pd.to_numeric(merged[f"forecast_{name}"], errors="coerce")
    out["wx_forecast_lead_hours"] = merged["forecast_lead_hours"]
    out["wx_observed_temp_f"] = pd.to_numeric(merged["observed_temp_f"], errors="coerce")
    out["wx_observed_wind_mph"] = pd.to_numeric(merged["observed_wind_mph"], errors="coerce")
    return out[columns]


def _nflverse_injuries(
    targets: pd.DataFrame, locks: pd.Series, *, backend: StorageBackend | None
) -> pd.DataFrame:
    """The official practice/game-status report — the only injury signal the backfill seasons have."""
    columns = ["inj_report_status", "inj_practice_status", "inj_report_primary"]
    seasons = sorted({int(s) for s in targets["season"].dropna().unique()})
    frame = _read(
        "nflverse_injuries",
        seasons,
        ["gsis_id", "game_type", "week", "report_status", "practice_status",
         "report_primary_injury", "date_modified"],
        backend=backend,
    )
    if frame.empty:
        return _blank(columns, targets.index)

    frame = frame.loc[frame["game_type"].astype("string") == "REG"].copy()
    frame["known_at"] = _resolved_known_at(frame, "nflverse_injuries", locks, week_col="week")
    frame["lock"] = _lock_for(frame, locks, season_col="_season", week_col="week")
    frame = frame.loc[
        _admissible(frame["week"], frame["known_at"], frame["week"], frame["lock"])
    ].copy()
    if frame.empty:
        return _blank(columns, targets.index)

    frame["season"] = _int_series(frame["_season"])
    frame["week"] = _int_series(frame["week"])
    frame["gsis_id"] = frame["gsis_id"].astype("string")
    # date_modified is a payload column the 2025+ feed no longer carries; where it exists it orders
    # the revision stream so the surviving row is the final pre-game report.
    frame = _latest(frame, ["gsis_id", "season", "week"], ["known_at", "date_modified"])

    merged = targets[["gsis_id", "season", "week"]].reset_index(drop=True).merge(
        frame[["gsis_id", "season", "week", "report_status", "practice_status",
               "report_primary_injury"]],
        on=["gsis_id", "season", "week"], how="left",
    )
    merged.index = targets.index
    return pd.DataFrame(
        {
            "inj_report_status": merged["report_status"].astype("string"),
            "inj_practice_status": merged["practice_status"].astype("string"),
            "inj_report_primary": merged["report_primary_injury"].astype("string"),
        },
        index=targets.index,
    )


def _sleeper_prelock(
    targets: pd.DataFrame,
    locks: pd.Series,
    scoring: Mapping[str, float],
    *,
    backend: StorageBackend | None,
) -> pd.DataFrame:
    """The forward-only pre-lock snapshot: the baseline to beat, plus Sleeper's own injury state.

    Both are 2026-forward — Sleeper's projection endpoints serve only the latest values. The injury
    columns are kept apart from ``inj_report_*`` and never coalesced: a live Sleeper capture returns
    PUP / NA / IR / DNR states the NFL practice report has no equivalent for, so one column holding
    either would silently change meaning at the 2026 boundary.
    """
    columns = ["baseline_sleeper_points", "inj_sleeper_status", "inj_sleeper_body_part"]
    seasons = sorted({int(s) for s in targets["season"].dropna().unique()})
    frame = _read(
        "sleeper_proj_week",
        seasons,
        ["player_id", "injury_status", "injury_body_part", *sorted(scoring)],
        backend=backend,
    )
    if frame.empty:
        _LOG.info(
            "sleeper_proj_week: nothing captured for season(s) %s — baseline_sleeper_points and "
            "inj_sleeper_* are null. Expected for 2016-2025: the endpoint serves only the latest "
            "values, so the market-beating baseline exists forward from 2026 W1 only.", seasons,
        )
        return _blank(columns, targets.index)

    frame["known_at"] = _resolved_known_at(frame, "sleeper_proj_week", locks, week_col="_week")
    frame["lock"] = _lock_for(frame, locks, season_col="_season", week_col="_week")
    admitted = _admissible(frame["_week"], frame["known_at"], frame["_week"], frame["lock"])
    excluded = int((~admitted).sum())
    frame = frame.loc[admitted].copy()
    if excluded:
        _LOG.info(
            "sleeper_proj_week: %d captured row(s) were not known before their week's lock and are "
            "excluded from the baseline", excluded,
        )
    if frame.empty:
        _LOG.warning(
            "sleeper_proj_week: every captured row for season(s) %s is post-lock, so the baseline "
            "column is null. Check that the pre-lock cron runs before kickoff.", seasons,
        )
        return _blank(columns, targets.index)

    frame["player_id"] = frame["player_id"].astype("string")
    frame["season"] = _int_series(frame["_season"])
    frame["week"] = _int_series(frame["_week"])
    frame = _latest(frame, list(KEY_COLS), ["known_at"])
    frame["baseline_sleeper_points"] = _score_rows(frame.to_dict("records"), scoring)

    merged = targets[list(KEY_COLS)].reset_index(drop=True).merge(
        frame[[*KEY_COLS, "baseline_sleeper_points", "injury_status", "injury_body_part"]],
        on=list(KEY_COLS), how="left",
    )
    merged.index = targets.index
    return pd.DataFrame(
        {
            "baseline_sleeper_points": merged["baseline_sleeper_points"],
            "inj_sleeper_status": merged["injury_status"].astype("string"),
            "inj_sleeper_body_part": merged["injury_body_part"].astype("string"),
        },
        index=targets.index,
    )


# --------------------------------------------------------------------------- lagged usage
def _snap_pct(
    seasons: Sequence[int], crosswalk: pd.DataFrame, *, backend: StorageBackend | None
) -> pd.DataFrame:
    frame = _read(
        "nflverse_snaps",
        seasons,
        ["pfr_player_id", "season", "week", "game_type", "offense_pct"],
        backend=backend,
    )
    if frame.empty:
        return pd.DataFrame(columns=[*KEY_COLS, "snap_pct"])
    frame = frame.loc[frame["game_type"].astype("string") == "REG"].copy()
    frame = _map_ids(frame, "pfr_player_id", _id_map(crosswalk, "pfr_id"), "nflverse_snaps")
    if frame.empty:
        return pd.DataFrame(columns=[*KEY_COLS, "snap_pct"])
    frame["season"] = _int_series(frame["season"])
    frame["week"] = _int_series(frame["week"])
    frame["snap_pct"] = pd.to_numeric(frame["offense_pct"], errors="coerce")
    return _latest(frame, list(KEY_COLS), ["_captured_at"])[[*KEY_COLS, "snap_pct"]]


def _opportunity(
    seasons: Sequence[int], crosswalk: pd.DataFrame, *, backend: StorageBackend | None
) -> pd.DataFrame:
    """Expected points and rush share. Same-week by construction, so only ever used lagged."""
    frame = _read(
        "nflverse_ff_opp",
        seasons,
        ["player_id", "season", "week", "total_fantasy_points_exp", "rush_attempt",
         "rush_attempt_team"],
        backend=backend,
    )
    if frame.empty:
        return pd.DataFrame(columns=[*KEY_COLS, "exp_points", "rush_share"])
    frame["gsis_id"] = frame["player_id"].astype("string")
    frame = _map_ids(frame, "gsis_id", _id_map(crosswalk, "gsis_id"), "nflverse_ff_opp")
    if frame.empty:
        return pd.DataFrame(columns=[*KEY_COLS, "exp_points", "rush_share"])
    # The provider types season as a string and week as a float; left raw in the lake on purpose.
    frame["season"] = _int_series(frame["season"])
    frame["week"] = _int_series(frame["week"])
    frame["exp_points"] = pd.to_numeric(frame["total_fantasy_points_exp"], errors="coerce")
    attempts = pd.to_numeric(frame["rush_attempt"], errors="coerce")
    team_attempts = pd.to_numeric(frame["rush_attempt_team"], errors="coerce")
    frame["rush_share"] = attempts / team_attempts.where(team_attempts > 0)
    return _latest(frame, list(KEY_COLS), ["_captured_at"])[[*KEY_COLS, "exp_points", "rush_share"]]


def _lagged_usage(
    targets: pd.DataFrame,
    crosswalk: pd.DataFrame,
    seasons: Sequence[int],
    *,
    backend: StorageBackend | None,
) -> pd.DataFrame:
    """Usage history over weeks **strictly before** the target, per player-season.

    Every input is ``post_game`` content, so the content rule is the only thing that can admit it and
    the lag is what makes it legal. Built with a within-``(player_id, season)`` shift rather than an
    N-by-N join: sorting by week and shifting once is the same predicate at a fraction of the cost,
    and "the previous row" is the previous week the player actually appeared in — a bye or an
    inactive week is skipped rather than imputed.
    """
    facts = (
        targets[[*KEY_COLS, "target_share", "y_custom_points"]]
        .rename(columns={"y_custom_points": "pts"})
        .reset_index(drop=True)
    )
    for extra in (
        _snap_pct(seasons, crosswalk, backend=backend),
        _opportunity(seasons, crosswalk, backend=backend),
    ):
        facts = facts.merge(extra, on=list(KEY_COLS), how="left") if not extra.empty else facts
    for column in _USAGE_COLS:
        facts[column] = (
            pd.to_numeric(facts[column], errors="coerce")
            if column in facts.columns
            else pd.Series(pd.NA, index=facts.index, dtype="Float64")
        )

    facts = facts.sort_values(["player_id", "season", "week"], kind="stable")
    keys = ["player_id", "season"]
    group = facts.groupby(keys, sort=False, observed=True)

    out = pd.DataFrame(index=facts.index)
    out["games_played_prior"] = group.cumcount().astype("Int64")
    for column, prefix in _USAGE_COLS.items():
        lagged = group[column].shift(1)
        out[f"{prefix}_last"] = lagged
        ewma = (
            facts.assign(_lagged=lagged)
            .groupby(keys, sort=False, observed=True)["_lagged"]
            .ewm(span=_EWMA_SPAN, min_periods=1)
            .mean()
        )
        out[f"{prefix}_ewma"] = ewma.droplevel(list(range(len(keys)))).reindex(facts.index)
        if prefix in _TRENDED:
            out[f"{prefix}_trend"] = lagged - group[column].shift(2)

    return out.reindex(range(len(targets))).set_axis(targets.index)


# --------------------------------------------------------------------------- entry point
def build_training_frame(
    seasons: Iterable[int],
    scoring: Mapping[str, float],
    *,
    asof: Literal["prelock"] = "prelock",
    backend: StorageBackend | None = None,
) -> pd.DataFrame:
    """One row per ``(sleeper player_id, season, week)``, label included, lookahead-free.

    ``scoring`` is the league's live ``scoring_settings`` dict — the label and the baseline are both
    re-scored with the Phase 1 engine, so this function never hard-codes a coefficient.

    Rows that fail to join to a Sleeper id, and rows whose ``(season, week, team)`` finds no
    scheduled game, are counted and logged. Nothing is dropped silently.
    """
    if asof != "prelock":
        raise ValueError(
            f"asof={asof!r} is not supported — 'prelock' is the only point-in-time view the lake "
            "can answer for, because that is the cadence its snapshots are captured on"
        )
    if not scoring:
        raise ValueError("scoring is empty — pass the league's live scoring_settings dict")
    wanted = sorted({int(s) for s in seasons})
    if not wanted:
        raise ValueError("no seasons given")

    team_games, locks = _calendar(wanted, backend=backend)
    crosswalk = _crosswalk(backend=backend)

    parts = [
        part
        for part in (
            _player_labels(wanted, scoring, crosswalk, backend=backend),
            _dst_labels(wanted, scoring, backend=backend),
        )
        if not part.empty
    ]
    if not parts:
        raise ValueError(
            f"no week actuals in the lake for season(s) {wanted} — there is no label to train on. "
            "Run scripts/backfill_lake.py first."
        )
    labels = pd.concat(parts, ignore_index=True)
    labels["player_id"] = labels["player_id"].astype("string")
    labels = _one_row_per_key(_attach_calendar(labels, team_games))

    targets = labels.assign(lock=_lock_for(labels, locks, season_col="season", week_col="week"))
    no_lock = int(targets["lock"].isna().sum())
    if no_lock:
        _LOG.warning(
            "%d/%d row(s) sit in a season-week with no readable kickoff, so no lock could be "
            "computed — their same-week features are withheld (the gate fails closed)",
            no_lock, len(targets),
        )

    frame = pd.concat(
        [
            targets[[*KEY_COLS, "gsis_id", "team", "opponent", "is_home", "game_id", "is_dst",
                     "y_custom_points"]],
            _positions(targets, locks, backend=backend),
            _sleeper_prelock(targets, locks, scoring, backend=backend),
            _lagged_usage(targets, crosswalk, wanted, backend=backend),
            _nflverse_injuries(targets, locks, backend=backend),
            _market(targets, locks, backend=backend),
            _weather(targets, locks, backend=backend),
        ],
        axis=1,
    )
    ordered = [*_LEADING_COLS, *[c for c in frame.columns if c not in set(_LEADING_COLS)]]
    frame = frame[ordered]

    _LOG.info(
        "training frame: %d row(s), %d player(s), season(s) %s — %d with a baseline, %d DST, "
        "%d with an as-of position",
        len(frame), frame["player_id"].nunique(), wanted,
        int(frame["baseline_sleeper_points"].notna().sum()), int(frame["is_dst"].sum()),
        int((~frame["position_is_static"].astype(bool)).sum()),
    )
    return frame.sort_values(list(KEY_COLS), kind="stable").reset_index(drop=True)


def _attach_calendar(labels: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """Give every label row its game: opponent, home/away, ``game_id``.

    The team comes from the row's own source — nflverse's vocabulary for players, Sleeper's for DST,
    both normalized by :data:`_TEAM_ALIASES` — so the two sides land on one calendar.

    A row that finds no game is **dropped**, not kept feature-less, because on the real lake it is
    never a player-week that merely lacks context — it is a **phantom label**. All four such rows in
    2016-2025 are Sleeper DEF lines for games that were never played: MIA and TB in 2017 week 1
    (postponed to week 11 for Hurricane Irma) and BUF and CIN in 2022 week 17 (abandoned after Damar
    Hamlin's cardiac arrest). Each scores a tidy 10.0 or better, because "0 points allowed" is what a
    game that did not happen looks like through a points-allowed bucket. Training on those would be
    teaching the model that a cancelled game is a shutout. ``nflverse_player_week`` produces no such
    rows at all, so no genuine skill-player week is at risk.
    """
    out = labels.rename(columns={"source_team": "team"})
    merged = out.merge(team_games, on=["season", "week", "team"], how="left")
    orphaned = merged["game_id"].isna()
    n_orphaned = int(orphaned.sum())
    if n_orphaned:
        sample = merged.loc[orphaned, ["player_id", "season", "week", "team"]].head(3).to_dict(
            "records"
        )
        args = (n_orphaned, len(merged), n_orphaned / len(merged) * 100, sample)
        if n_orphaned / len(merged) > _ORPHAN_LABEL_CEILING:
            _LOG.warning(
                "%d/%d label row(s) (%.3f%%) have no scheduled game for their (season, week, team) "
                "and are dropped — far above the handful of postponed/cancelled games this is "
                "expected for, so the calendar or a team vocabulary has changed. Sample: %s", *args,
            )
        else:
            _LOG.info(
                "%d/%d label row(s) (%.3f%%) have no scheduled game for their (season, week, team) "
                "and are dropped as phantom labels (a postponed or cancelled game). Sample: %s",
                *args,
            )
    return merged.loc[~orphaned].reset_index(drop=True)


def _one_row_per_key(labels: pd.DataFrame) -> pd.DataFrame:
    """The frame's stated grain, enforced rather than assumed."""
    duplicated = labels.duplicated(subset=list(KEY_COLS), keep=False)
    if bool(duplicated.any()):
        sample = labels.loc[duplicated, list(KEY_COLS)].drop_duplicates().head(3).to_dict("records")
        _LOG.error(
            "%d row(s) share a (player_id, season, week) key, which the frame's grain forbids — "
            "keeping the highest-scoring row per key. Sample: %s", int(duplicated.sum()), sample,
        )
        labels = _latest(labels, list(KEY_COLS), ["y_custom_points"])
    return labels.reset_index(drop=True)

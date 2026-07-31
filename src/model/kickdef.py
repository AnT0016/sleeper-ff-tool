"""The weekly K + DST model — component-wise through the scoring engine (Phase 9, ticket #30).

The league's two most mis-valued positions, and the clearest structural edge in the project: kicker
scoring is **distance-based** (``fgm_40_49`` 4, ``fgm_50p`` 5) and DST scoring is **bucketed** (``pts_allow_0``
10, ``pts_allow_1_6`` 7). A model that regresses *points* directly learns those buckets implicitly and
badly, because points are a **step function** of the underlying stats: ``E[f(X)] != f(E[X])`` and only a
component model is unbiased (Decision #2, and Decision #7 states explicitly why #29's points head is
*not* licence for one here). So this model predicts the **components** and lets
:func:`scoring.engine.points` price them — the "never hand-code scoring" rule held through the model
layer, and a mid-season scoring change re-prices predictions with no retraining.

The one asymmetry a future reader will most likely get wrong
------------------------------------------------------------
K and DST are both non-linear, but in **different ways**, and the treatment differs:

* **Kicker — additive counts across bands.** A kicker's makes decompose into independent per-band
  counts (``fgm_0_19 … fgm_50p``, ``xpm``, misses). Predict the *expected makes per band*; because the
  coefficient is **constant within a band**, ``E[coef * makes] = coef * E[makes]`` and the engine's
  linear sum over bands is exact — **no distribution is needed at all**. The distance non-linearity is
  handled purely by predicting at the band grain; collapsing to a mean distance and bucketing *that*
  would be the bias (pinned by ``test_fg_per_band_makes_differ_from_a_mean_distance_collapse``).
* **DST — one value, bucketed.** A defense allows *one* points total, landing in exactly one
  ``pts_allow_*`` bucket — a step function of a **single** random variable, not a decomposable count. So
  it genuinely needs a **distribution** over points-allowed, then ``E[bucket points] = sum_k P(k)
  coef(bucket(k))``. Elegantly, because the engine is linear, that expectation *is*
  ``engine.points({bucket_key: P(bucket)}, scoring)`` — the predicted "stat line" carries **probability
  mass** per bucket and the engine turns it into the correct expectation. Distribution-through-the-engine
  with no special case in the scorer.

The points-allowed distribution, and where it quietly fails
-----------------------------------------------------------
The distribution is an **empirical residual model**: a ridge predicts the mean points allowed ``μ`` from
the Vegas market (``opp_implied_total`` is close to the market's direct opinion), and the spread around
it is the empirical distribution of training residuals, shifted by ``μ`` and clamped at 0.

The clamp is where a naive single grid fails, and it fails *most* on the highest-value cell. Points
allowed is **heteroskedastic** — variance grows with the mean, and the left tail is compressed for an
elite (low-``μ``) defense — so one shared residual grid shifts too much low-``μ`` mass below zero, and
the clamp piles it onto ``pts_allow_0`` (10 points, the highest cell in the table), *inventing* shutout
probability exactly where it is most expensive to get wrong. So the grid is **conditioned on ``μ``**
(:data:`DEFAULT_PA_BINS` bins); within a narrow ``μ`` band the residuals shifted by a same-band ``μ``
reproduce realistic non-negative points and the clamp barely bites. ``scripts/eval_kickdef.py`` reports
predicted-vs-realized rate for ``pts_allow_0`` **and** ``pts_allow ≤ 6`` by predicted-``μ`` decile — a
direct calibration check on the left tail, the ``≤ 6`` cell added because shutouts are ~0.3% of rows and
too thin to read alone.

Bar, gate and safety (Decisions #8, #9)
---------------------------------------
The bar is the three naive baselines in ``docs/model-baselines.md`` — beat all three on **both** MAE and
within-slate Spearman ρ, and it is the **lowest bar in the phase**: both K and DEF MAE winners *are* the
learned position mean, and nothing in the project orders a kicker or a defence (baseline ρ 0.067 / 0.095
against 0.49-0.66 at skill after #29). So "beat the mean" on accuracy, and give the boards an ordering at
all. Measured, the component model is graded **per (position × cold/warm) cell** — four cells, because
"beats the baselines for K and DEF" is a claim with four cells behind it (Decision #9 item 6): a warm
gain must not be thrown away by a cold loss. A cell where the component model does not clear the bar on
both metrics — or that rests on too few held-out rows (:data:`_MIN_CELL_N`) — **defers** to the
prior-season baseline; ties defer. The shipped :class:`KickDefModel` reads that measured gate from the
committed artifact, so a bare constructor is the *shipped* configuration and ``KickDefModel(defer=())``
is the pure-component diagnostic (Decision #9 item 3). The artifact is read back by
:meth:`KickDefModel.load_fitted` (item 4), and refits reproducibly from the lake (closed-form ridge,
deterministic residual grids) — no hand-edited file.

No new dependency: #29 logged the escalation path (gradient boosting) and left it unreached; ridge over
numpy is the floor here too.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.ids import NFLVERSE_STAT_COLS, nflverse_to_sleeper_stats
from model.baselines import LaggedExpectedPoints, PriorSeasonRank, TrailingMean
from model.evaluate import (
    DEFAULT_TEST_SEASONS,
    LABEL_COL,
    PositionMetrics,
    evaluate,
    per_position_metrics,
    walk_forward_splits,
)
from model.season import _fit_ridge, _PositionFit  # deliberate reuse of #31's closed-form solver
from model.weekly import is_cold_start  # reuse the cold-start cohort marker (games_played_prior == 0)
from scoring.engine import points as engine_points
from store.lake import StorageBackend, read_source

_LOG = logging.getLogger(__name__)

#: The positions this model owns. K and DST are #30's precisely because their scoring is bucketed;
#: QB/RB/WR/TE are #29's (a linear points head, Decision #7).
KICKDEF_POSITIONS: tuple[str, ...] = ("K", "DEF")

#: The model's features, all provably pre-lock and 100%-present for **both** K and DEF (profile #27
#: §5b): the Vegas market (the approach's whole input — close to the market's direct opinion on both a
#: kicker's scoring environment and a defence's opponent), venue, and the position's own points lags.
#: The list is **shared** across K and DEF; DEF carries no snap/target/rush/exp usage column at all
#: (§5b, 100% null), so usage cannot enter a shared list — and for K, where those columns *are* present
#: (~7.9% null), excluding them is **parsimony** at ~4.2k rows, not a claim they carry no signal. A
#: per-position ridge zeroes any feature it cannot use, so one list costs nothing.
KICKDEF_FEATURES: tuple[str, ...] = (
    # market (the pre-game view, 0% null every season for K and DEF)
    "implied_team_total",
    "opp_implied_total",
    "team_spread_line",
    "total_line",
    # context (fixed well before kickoff)
    "is_indoor",
    "is_div_game",
    # own points lags (null on a first appearance — the cold-start cohort)
    "points_last",
    "points_ewma",
    "points_trend",
)

#: Ridge penalty on the standardised features — the same modest value #29/#31 use, and **not** tuned on
#: any test season. Stabilises a near-collinear market block without shrinking the fit to the mean.
_RIDGE_ALPHA = 1.0

#: The DST counting-stat scoring keys — everything a defence scores that is **not** a ``pts_allow``
#: bucket. Curated from the league's 42-key table (CLAUDE.md); the model predicts an expected count for
#: each present in the live scoring, and the engine prices them linearly. ``st_*`` (player special
#: teams) are near-always zero on a team-defence row, so their ridge is a harmless ~0 — kept so the
#: correctness anchor stays complete against any scoring the league returns.
KNOWN_DST_COUNTING: frozenset[str] = frozenset(
    {
        "def_td", "sack", "int", "fum_rec", "safe", "ff", "blk_kick",
        "def_st_td", "def_st_ff", "def_st_fum_rec", "st_td", "st_ff", "st_fum_rec",
        "def_2pt",
    }
)

#: The number of ``μ`` bins the points-allowed residual grid is conditioned on (heteroskedasticity — see
#: the module docstring). 1 recovers the single shared grid (the diagnostic). Declared, not tuned on the
#: bar: the choice between 1 and 3 is driven by the left-tail calibration ``scripts/eval_kickdef.py``
#: reports, never by MAE/ρ.
DEFAULT_PA_BINS = 3

#: Quantiles of the residual distribution stored per ``μ`` bin — a compact, deterministic stand-in for
#: the full residual sample (101 points captures the shape without freezing thousands of numbers).
_RESID_QUANTILES = 101

#: Minimum held-out rows for a (position × cohort) cell's gate to be *decided* on evidence. Below it the
#: cell defers — a 400-row cold-start decision must not read like a 4,000-row warm one (Decision #9 item
#: 6 / the review). K cold-start is ~250-400 held-out rows, so this is the line it sits just above.
_MIN_CELL_N = 200

#: Two metrics within this are the same number (a deferred cell predicts exactly its baseline).
_TIE_EPS = 1e-9


# =============================================================== points-allowed buckets (from scoring)
#: A ``pts_allow_*`` key, e.g. ``pts_allow_0`` / ``pts_allow_1_6`` / ``pts_allow_35p``.
_PA_PREFIX = "pts_allow_"
_PA_RANGE_RE = re.compile(r"^(\d+)_(\d+)$")


def pts_allow_keys(scoring: Mapping[str, float]) -> tuple[str, ...]:
    """The league's points-allowed bucket keys, ordered by their lower bound (never hard-coded)."""
    keys = [k for k in scoring if k.startswith(_PA_PREFIX)]
    return tuple(sorted(keys, key=lambda k: _parse_pts_allow_bounds(k)[0]))


def _parse_pts_allow_bounds(key: str) -> tuple[int, float]:
    """``(lo, hi)`` for a bucket key, hi ``inf`` for the open top bucket — parsed, not hand-coded.

    ``pts_allow_0`` -> ``(0, 0)``; ``pts_allow_1_6`` -> ``(1, 6)``; ``pts_allow_35p`` -> ``(35, inf)``.
    Reading the bounds off the key name keeps the immutable "the API is the source of truth for scoring"
    rule true — the buckets come from the league's own key vocabulary, the coefficients from its scoring.
    """
    body = key[len(_PA_PREFIX):]
    if body.endswith("p"):
        return int(body[:-1]), float("inf")
    m = _PA_RANGE_RE.match(body)
    if m:
        return int(m.group(1)), int(m.group(2))
    return int(body), int(body)  # a single value, e.g. "0"


def bucket_of_points_allowed(points_allowed: float, keys: Sequence[str]) -> str | None:
    """The one bucket key a points-allowed value falls in (buckets are mutually exclusive)."""
    if points_allowed is None or pd.isna(points_allowed):
        return None
    value = float(points_allowed)
    for key in keys:
        lo, hi = _parse_pts_allow_bounds(key)
        if lo <= value <= hi:
            return key
    return None


def pts_allow_points(points_allowed: float, scoring: Mapping[str, float]) -> float:
    """Fantasy points from a **single** points-allowed value — its bucket's coefficient."""
    key = bucket_of_points_allowed(points_allowed, pts_allow_keys(scoring))
    return 0.0 if key is None else float(scoring.get(key, 0.0))


def expected_pts_allow_points(
    distribution: Mapping[float, float], scoring: Mapping[str, float]
) -> float:
    """``E[points]`` over a **distribution** of points-allowed values: ``sum_k P(k) coef(bucket(k))``.

    This is the quantity the DST model must produce, and it is **not** ``pts_allow_points(mean)``: the
    buckets are steps, so the expectation over a distribution straddling a boundary lands between two
    bucket values while the mean lands in one. Swapping this for ``pts_allow_points(sum_k k P(k))`` is the
    exact bias Decision #2 exists to prevent — pinned red-first by ``test_model_kickdef``.
    """
    return float(sum(p * pts_allow_points(k, scoring) for k, p in distribution.items()))


def pts_allow_bucket_probs(
    points_allowed_samples: np.ndarray, keys: Sequence[str]
) -> dict[str, float]:
    """Probability mass per bucket from a sample of points-allowed values (a discrete distribution)."""
    bounds = [(_parse_pts_allow_bounds(k), k) for k in keys]
    n = len(points_allowed_samples)
    out = {k: 0.0 for k in keys}
    if n == 0:
        return out
    for value in points_allowed_samples:
        for (lo, hi), key in bounds:
            if lo <= value <= hi:
                out[key] += 1.0 / n
                break
    return out


# =============================================================== the component frame (labels attached)
#: The prefix that marks a **label** column: a week-N post-game component outcome, extracted so the
#: model has a per-component target. A ``comp_*`` column is never a feature — the leak test pins that
#: ``KICKDEF_FEATURES`` contains no ``comp_*`` name, exactly as the weekly model excludes raw same-week
#: usage. These columns are legal because they are only ever the regression **target** for their own
#: week; nothing here is used to predict a *later* week.
COMP_PREFIX = "comp_"


def _latest_capture(frame: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    """One row per ``key_cols`` per partition — the newest capture (mirrors ``dataset.assemble``)."""
    if frame.empty:
        return frame
    scope = [c for c in ("_season", "_week") if c in frame.columns]
    order = [c for c in ("_captured_at",) if c in frame.columns]
    return (
        frame.sort_values(order, kind="stable")
        .drop_duplicates(subset=[*scope, *key_cols], keep="last")
        .reset_index(drop=True)
    )


def _kicker_component_labels(
    seasons: Sequence[int], keys: pd.DataFrame, *, backend: StorageBackend | None
) -> pd.DataFrame:
    """The kicker component line per ``(gsis_id, season, week)``, Sleeper-keyed via the Phase 1 helper.

    Re-reads ``nflverse_player_week`` — the same source ``dataset.assemble._player_labels`` scored the
    K label from — and translates each row with :func:`data.ids.nflverse_to_sleeper_stats`, so
    ``engine(components)`` reproduces ``y_custom_points`` by construction (the correctness anchor). The
    join is on ``gsis_id`` (which the frame already carries), so no crosswalk is repeated here; a
    semi-join to the frame's kicker keys keeps the Python translation to the ~5k rows that matter.
    """
    comp_cols = [f"{COMP_PREFIX}{v}" for v in _kicker_sleeper_keys()]
    empty = pd.DataFrame(columns=["gsis_id", "season", "week", *comp_cols])
    if keys.empty:
        return empty
    frame = read_source(
        "nflverse_player_week",
        seasons,
        columns=["player_id", "season", "week", "season_type", "_season", "_week", "_captured_at",
                 *NFLVERSE_STAT_COLS],
        backend=backend,
    )
    if frame.empty:
        return empty
    frame = frame.loc[frame["season_type"].astype("string") == "REG"].copy()
    frame = _latest_capture(frame, ["player_id", "season_type", "week"])
    frame["gsis_id"] = frame["player_id"].astype("string")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")

    want = keys[["gsis_id", "season", "week"]].drop_duplicates()
    want["gsis_id"] = want["gsis_id"].astype("string")
    want["season"] = pd.to_numeric(want["season"], errors="coerce").astype("Int64")
    want["week"] = pd.to_numeric(want["week"], errors="coerce").astype("Int64")
    frame = frame.merge(want, on=["gsis_id", "season", "week"], how="inner")
    if frame.empty:
        return empty

    stat_cols = [c for c in NFLVERSE_STAT_COLS if c in frame.columns]
    stats = frame[stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    translated = [nflverse_to_sleeper_stats(row) for row in stats.to_dict("records")]
    for key in _kicker_sleeper_keys():
        frame[f"{COMP_PREFIX}{key}"] = [float(t.get(key, 0.0)) for t in translated]
    return frame[["gsis_id", "season", "week", *comp_cols]].reset_index(drop=True)


def _defense_component_labels(
    seasons: Sequence[int], scoring: Mapping[str, float], *, backend: StorageBackend | None
) -> pd.DataFrame:
    """The DST component line per ``(player_id, season, week)`` from ``sleeper_stats_week``.

    The same source ``dataset.assemble._dst_labels`` scored the DEF label from, already Sleeper-keyed
    (``player_id`` *is* the team abbreviation). Carries every non-``pts_allow`` DST counting key plus the
    **raw** ``pts_allow`` number (``comp_pts_allow``) — the raw value, not the bucket flag, so the
    distribution can be modelled from it. Counting stats are sparse (null where the play never happened),
    so they read back as 0.
    """
    counting = _dst_counting_keys(scoring)
    pa_flags = pts_allow_keys(scoring)
    comp_cols = [f"{COMP_PREFIX}{k}" for k in counting] + [f"{COMP_PREFIX}pts_allow"]
    empty = pd.DataFrame(columns=["player_id", "season", "week", *comp_cols])
    frame = read_source(
        "sleeper_stats_week",
        seasons,
        columns=["player_id", "position", "season_type", "pts_allow", "_season", "_week",
                 "_captured_at", *counting, *pa_flags],
        backend=backend,
    )
    if frame.empty:
        return empty
    frame = frame.loc[
        (frame["position"].astype("string") == "DEF")
        & (frame["season_type"].astype("string") == "regular")
    ].copy()
    if frame.empty:
        return empty
    frame = _latest_capture(frame, ["player_id"])
    frame["player_id"] = frame["player_id"].astype("string")
    frame["season"] = pd.to_numeric(frame["_season"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["_week"], errors="coerce").astype("Int64")
    for key in counting:
        frame[f"{COMP_PREFIX}{key}"] = pd.to_numeric(frame[key], errors="coerce").fillna(0.0)
    frame[f"{COMP_PREFIX}pts_allow"] = _reconstruct_pts_allow(frame, pa_flags)
    return frame[["player_id", "season", "week", *comp_cols]].reset_index(drop=True)


def _reconstruct_pts_allow(frame: pd.DataFrame, pa_flags: Sequence[str]) -> pd.Series:
    """The raw points-allowed number, filling Sleeper's **null-on-shutout** quirk from the bucket flags.

    Sleeper leaves the raw ``pts_allow`` field empty when a defence pitches a shutout (0 points allowed),
    setting only the ``pts_allow_0`` flag — so a naive read loses every shutout, which is both a 10-point
    label error (the correctness anchor's original miss) **and** a distribution disaster: the model would
    train on zero shutouts and predict ``P(pts_allow_0) = 0`` forever, the highest-value cell in the
    table. Where the raw value is null, this fills it from whichever bucket flag is set — its lower bound,
    which is exactly 0 for a shutout and a defensible representative for any other (non-observed) case.
    """
    raw = pd.to_numeric(frame.get("pts_allow"), errors="coerce")
    null = raw.isna()
    if not bool(null.any()):
        return raw
    filled = raw.copy()
    for key in pa_flags:
        flag_col = frame.get(key)
        if flag_col is None:
            continue
        lo, _hi = _parse_pts_allow_bounds(key)
        flag_set = pd.to_numeric(flag_col, errors="coerce").fillna(0) == 1
        filled = filled.mask(null & flag_set, float(lo))
    return filled


@functools.cache
def _kicker_sleeper_keys() -> tuple[str, ...]:
    """Every Sleeper key :func:`nflverse_to_sleeper_stats` can emit — the complete kicker anchor line.

    Complete on purpose (fake-field-goal passing/rushing and a kicker's fumble are covered), so
    ``engine(components) == y_custom_points`` holds to the tolerance and the anchor's floor is met by
    construction rather than by hoping kickers never do anything but kick.
    """
    from data.ids import STAT_MAP  # local import: only the frame builder needs the raw map

    extras = ("fum_lost", "fgm_50p", "fgmiss", "xpmiss")
    return tuple(dict.fromkeys((*STAT_MAP.values(), *extras)))


def _dst_counting_keys(scoring: Mapping[str, float]) -> tuple[str, ...]:
    """The live scoring's DST counting keys, ordered — non-zero DST keys that are not ``pts_allow``."""
    return tuple(k for k in sorted(KNOWN_DST_COUNTING) if scoring.get(k))


def _kicker_model_keys(scoring: Mapping[str, float]) -> tuple[str, ...]:
    """The kicker scoring keys the model **fits a ridge for** — the distance bands, XP and misses.

    A strict subset of the anchor line: the rest (a fake-FG pass, a kicker fumble) is unpredictable noise
    the model rightly predicts at ~0, so it is not modelled, only reproduced by the anchor.
    """
    band = ("fgm_0_19", "fgm_20_29", "fgm_30_39", "fgm_40_49", "fgm_50p", "xpm", "fgmiss", "xpmiss")
    return tuple(k for k in band if scoring.get(k) is not None)


def build_kickdef_frame(
    seasons: Iterable[int],
    scoring: Mapping[str, float],
    *,
    backend: StorageBackend | None = None,
) -> pd.DataFrame:
    """The weekly K + DST frame: the training frame's features + universe, with component **labels**.

    ``scoring`` is the league's live ``scoring_settings`` — passed straight to
    :func:`dataset.assemble.build_training_frame` (which re-scores every label through the Phase 1
    engine), then used to decide which component keys to extract. This function adds the per-component
    outcome columns (``comp_*``) that the frame builder deliberately discards; it never touches the
    assembler (additive rule) and never scores anything by hand.

    Component columns are **labels**, valid only as the target for their own week — the feature list
    (:data:`KICKDEF_FEATURES`) excludes them and the leak test pins it. Rows keep their
    ``y_custom_points`` from the assembler, so the correctness anchor (:func:`anchor_mismatch`) can check
    ``engine(components) == y_custom_points`` per row.
    """
    if not scoring:
        raise ValueError("scoring is empty — pass the league's live scoring_settings dict")
    from dataset.assemble import build_training_frame  # local: avoids a heavy import at module load

    wanted = sorted({int(s) for s in seasons})
    weekly = build_training_frame(wanted, scoring, backend=backend)
    frame = weekly[weekly["position"].isin(KICKDEF_POSITIONS)].copy()
    if frame.empty:
        _LOG.warning("build_kickdef_frame: no K/DEF rows for season(s) %s", wanted)
        return frame

    k_rows = frame[frame["position"] == "K"]
    kicker = _kicker_component_labels(wanted, k_rows, backend=backend)
    defense = _defense_component_labels(wanted, scoring, backend=backend)

    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    frame["gsis_id"] = frame["gsis_id"].astype("string")
    if not kicker.empty:
        frame = frame.merge(kicker, on=["gsis_id", "season", "week"], how="left")
    if not defense.empty:
        frame = frame.merge(defense, on=["player_id", "season", "week"], how="left")

    n_k = int((frame["position"] == "K").sum())
    n_def = int((frame["position"] == "DEF").sum())
    miss = anchor_mismatch(frame, scoring)
    _LOG.info(
        "kickdef frame: %d row(s) (%d K, %d DEF); component anchor engine(components)==label: "
        "K %.2f%% match, DEF %.2f%% match",
        len(frame), n_k, n_def, miss["K"]["match_pct"], miss["DEF"]["match_pct"],
    )
    return frame.reset_index(drop=True)


# =============================================================== correctness anchor (declared floor)
#: ``engine(observed components)`` must reproduce ``y_custom_points`` within this many points.
ANCHOR_TOL = 0.01

#: Declared **before** measuring (the anti-pattern Decision #3 exists to prevent, applied to the anchor):
#: at least this share of each position's rows must reproduce the label from their components. Below it,
#: a scoring key is going unextracted and the decomposition is incomplete — the eval **raises**.
ANCHOR_FLOOR_PCT = 99.5


def _row_component_points(row: Mapping[str, object], scoring: Mapping[str, float]) -> float:
    """``engine.points`` over a row's observed component line — the anchor's left-hand side.

    For K the ``comp_*`` columns are the full translated stat line. For DST they are the counting stats
    plus the raw ``comp_pts_allow``, whose bucket is resolved here — so this also proves the parsed bucket
    bounds agree with Sleeper's own flag-based scoring of the label.
    """
    stats: dict[str, float] = {}
    for name, value in row.items():
        if not isinstance(name, str) or not name.startswith(COMP_PREFIX):
            continue
        if value is None or pd.isna(value):
            continue
        key = name[len(COMP_PREFIX):]
        if key == "pts_allow":
            bucket = bucket_of_points_allowed(value, pts_allow_keys(scoring))
            if bucket is not None:
                stats[bucket] = stats.get(bucket, 0.0) + 1.0
        else:
            stats[key] = float(value)
    return engine_points(stats, scoring)


def anchor_mismatch(frame: pd.DataFrame, scoring: Mapping[str, float]) -> dict[str, dict]:
    """Per-position match rate of ``engine(components)`` against ``y_custom_points`` (the anchor).

    Returns, per position, ``{n, matched, match_pct, mismatch_keys}`` where ``mismatch_keys`` counts the
    scoring keys implicated on the rows that miss — the breakdown the eval prints when the floor is
    breached, so "which key went unextracted" is answered rather than guessed.
    """
    out: dict[str, dict] = {}
    for pos in KICKDEF_POSITIONS:
        sub = frame[frame["position"] == pos]
        label = pd.to_numeric(sub.get(LABEL_COL), errors="coerce") if LABEL_COL in sub else None
        if sub.empty or label is None:
            out[pos] = {"n": 0, "matched": 0, "match_pct": 100.0, "mismatch_keys": {}}
            continue
        recomputed = np.array(
            [_row_component_points(r, scoring) for r in sub.to_dict("records")], dtype="float64"
        )
        diff = np.abs(recomputed - label.to_numpy(dtype="float64"))
        # diff is NaN only where the label was unscored (never in the real frame); such rows compare
        # False against the tolerance, so count them as excused rather than as misses.
        matched = int(np.sum(diff <= ANCHOR_TOL) + np.sum(np.isnan(diff)))
        n = int(len(sub))
        mism_keys: dict[str, int] = {}
        for row, d in zip(sub.to_dict("records"), diff, strict=True):
            if not np.isnan(d) and d > ANCHOR_TOL:
                for name in row:
                    if isinstance(name, str) and name.startswith(COMP_PREFIX):
                        val = row[name]
                        if val is not None and not pd.isna(val) and float(val) != 0.0:
                            mism_keys[name] = mism_keys.get(name, 0) + 1
        out[pos] = {
            "n": n,
            "matched": matched,
            "match_pct": 100.0 * matched / n if n else 100.0,
            "mismatch_keys": dict(sorted(mism_keys.items(), key=lambda kv: -kv[1])),
        }
    return out


# =============================================================== feature matrix
def _to_float(col: pd.Series) -> np.ndarray:
    """A column as float64, nulls preserved as NaN (booleans → 1.0/0.0, ``<NA>`` → ``NaN``).

    ``is_indoor`` is a nullable boolean and ``is_div_game`` a nullable int, so a plain ``astype(float)``
    would raise on ``<NA>``; mapping the booleans and coercing the rest keeps a missing value a genuine
    ``NaN`` for imputation (the same handling ``model.weekly`` uses).
    """
    if col.dtype == bool or str(col.dtype) == "boolean":
        col = col.astype("object").map({True: 1.0, False: 0.0})
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """:data:`KICKDEF_FEATURES` as a float ``(n, k)`` array; an absent column is all-NaN for imputation."""
    cols = [
        _to_float(frame[name]) if name in frame.columns else np.full(len(frame), np.nan)
        for name in KICKDEF_FEATURES
    ]
    return np.column_stack(cols) if cols else np.empty((len(frame), 0))


# =============================================================== component heads
class _CountingHead:
    """A per-key closed-form ridge predicting expected component **counts**, one ridge per key.

    Predictions are clamped at 0 (a count cannot be negative; a ridge can dip below it), then the engine
    prices them. Because counting scoring is linear, the clamped expectation is exactly what the engine
    should see — ``E[coef * count] = coef * E[count]``.
    """

    def __init__(self, keys: Sequence[str], alpha: float) -> None:
        self.keys = tuple(keys)
        self.alpha = float(alpha)
        self.fits: dict[str, _PositionFit] = {}

    def fit(self, x: np.ndarray, comp: pd.DataFrame) -> _CountingHead:
        self.fits = {}
        if len(x) == 0:
            return self
        for key in self.keys:
            col = comp.get(f"{COMP_PREFIX}{key}")
            y = pd.to_numeric(col, errors="coerce").fillna(0.0).to_numpy(dtype="float64") if col is not None else None
            if y is None or len(y) == 0:
                continue
            self.fits[key] = _fit_ridge(x, y, self.alpha)
        return self

    def predict_stats(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return {key: np.clip(fit.predict(x), 0.0, None) for key, fit in self.fits.items()}

    def to_dict(self) -> dict:
        return {"keys": list(self.keys), "alpha": self.alpha,
                "fits": {k: _fit_to_dict(f) for k, f in self.fits.items()}}

    @classmethod
    def from_dict(cls, payload: Mapping) -> _CountingHead:
        head = cls(payload["keys"], float(payload.get("alpha", _RIDGE_ALPHA)))
        head.fits = {k: _fit_from_dict(v) for k, v in payload.get("fits", {}).items()}
        return head


@dataclass
class _PtsAllowDist:
    """The points-allowed distribution: a ridge for the mean, μ-binned empirical residual grids.

    ``predict_bucket_probs`` returns the probability mass per ``pts_allow_*`` bucket per row — the
    predicted stat line's points-allowed component, which the engine then prices linearly into
    ``E[pts_allow points]``. The grid is conditioned on ``μ`` because points allowed is heteroskedastic
    and the clamp-at-0 otherwise invents shutout mass for elite defences (module docstring).
    """

    mu_fit: _PositionFit
    bin_edges: list[float]  # interior μ quantile edges; len == n_bins - 1
    resid_grids: list[list[float]]  # one residual quantile grid per μ bin

    def _bin_of(self, mu: np.ndarray) -> np.ndarray:
        if len(self.bin_edges) == 0:
            return np.zeros(len(mu), dtype=int)
        return np.digitize(mu, self.bin_edges, right=False)

    def predict_bucket_probs(self, x: np.ndarray, keys: Sequence[str]) -> dict[str, np.ndarray]:
        mu = self.mu_fit.predict(x)
        bins = self._bin_of(mu)
        out = {k: np.zeros(len(mu), dtype="float64") for k in keys}
        for i in range(len(mu)):
            grid = np.asarray(self.resid_grids[int(bins[i])], dtype="float64")
            samples = np.clip(mu[i] + grid, 0.0, None)
            probs = pts_allow_bucket_probs(np.round(samples), keys)
            for k, p in probs.items():
                out[k][i] = p
        return out

    def predict_prob_leq(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """``P(points_allowed <= threshold)`` per row — the left-tail calibration probe."""
        mu = self.mu_fit.predict(x)
        bins = self._bin_of(mu)
        out = np.zeros(len(mu), dtype="float64")
        for i in range(len(mu)):
            grid = np.asarray(self.resid_grids[int(bins[i])], dtype="float64")
            samples = np.clip(mu[i] + grid, 0.0, None)
            out[i] = float(np.mean(np.round(samples) <= threshold))
        return out

    def predict_mu(self, x: np.ndarray) -> np.ndarray:
        return self.mu_fit.predict(x)

    def to_dict(self) -> dict:
        return {"mu_fit": _fit_to_dict(self.mu_fit), "bin_edges": list(self.bin_edges),
                "resid_grids": [list(g) for g in self.resid_grids]}

    @classmethod
    def from_dict(cls, payload: Mapping) -> _PtsAllowDist:
        return cls(_fit_from_dict(payload["mu_fit"]), list(payload["bin_edges"]),
                   [list(g) for g in payload["resid_grids"]])


def _fit_pts_allow_dist(
    x: np.ndarray, pts_allow: np.ndarray, *, n_bins: int, alpha: float
) -> _PtsAllowDist:
    """Fit the mean ridge, then the residual quantile grid per ``μ`` bin (deterministic)."""
    keep = ~np.isnan(pts_allow)
    x, pts_allow = x[keep], pts_allow[keep]
    mu_fit = _fit_ridge(x, pts_allow, alpha)
    mu_hat = mu_fit.predict(x)
    resid = pts_allow - mu_hat

    n_bins = max(1, int(n_bins))
    qs = np.linspace(0.0, 1.0, _RESID_QUANTILES)
    if n_bins == 1 or len(resid) < n_bins * 2:
        return _PtsAllowDist(mu_fit, [], [list(np.quantile(resid, qs))])
    edges = list(np.quantile(mu_hat, np.linspace(0, 1, n_bins + 1)[1:-1]))
    assign = np.digitize(mu_hat, edges, right=False)
    grids: list[list[float]] = []
    for b in range(n_bins):
        r = resid[assign == b]
        r = r if len(r) else resid  # a degenerate empty bin borrows the pooled residuals
        grids.append(list(np.quantile(r, qs)))
    return _PtsAllowDist(mu_fit, edges, grids)


class _KickerHead:
    """K: per-band make/miss ridges, priced through the engine (no distribution needed — see docstring)."""

    def __init__(self, scoring: Mapping[str, float], alpha: float = _RIDGE_ALPHA) -> None:
        self.scoring = dict(scoring)
        self.counting = _CountingHead(_kicker_model_keys(scoring), alpha)

    def fit(self, frame: pd.DataFrame) -> _KickerHead:
        self.counting.fit(_feature_matrix(frame), frame)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        stats = self.counting.predict_stats(_feature_matrix(frame))
        return _score_stat_lines(stats, {}, self.scoring, len(frame))

    def to_dict(self) -> dict:
        return {"counting": self.counting.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping, scoring: Mapping[str, float]) -> _KickerHead:
        head = cls(scoring)
        head.counting = _CountingHead.from_dict(payload["counting"])
        return head


class _DefenseHead:
    """DST: counting ridges + the points-allowed distribution, priced together through the engine."""

    def __init__(
        self, scoring: Mapping[str, float], *, alpha: float = _RIDGE_ALPHA, n_pa_bins: int = DEFAULT_PA_BINS
    ) -> None:
        self.scoring = dict(scoring)
        self.n_pa_bins = int(n_pa_bins)
        self.counting = _CountingHead(_dst_counting_keys(scoring), alpha)
        self.pa: _PtsAllowDist | None = None

    def fit(self, frame: pd.DataFrame) -> _DefenseHead:
        x = _feature_matrix(frame)
        self.counting.fit(x, frame)
        pa_col = frame.get(f"{COMP_PREFIX}pts_allow")
        self.pa = None
        if pa_col is not None and len(x):
            pa = pd.to_numeric(pa_col, errors="coerce").to_numpy(dtype="float64")
            if bool(np.isfinite(pa).any()):
                self.pa = _fit_pts_allow_dist(x, pa, n_bins=self.n_pa_bins, alpha=self.counting.alpha)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = _feature_matrix(frame)
        stats = self.counting.predict_stats(x)
        bucket_probs = self.pa.predict_bucket_probs(x, pts_allow_keys(self.scoring)) if self.pa else {}
        return _score_stat_lines(stats, bucket_probs, self.scoring, len(frame))

    def to_dict(self) -> dict:
        return {"counting": self.counting.to_dict(), "n_pa_bins": self.n_pa_bins,
                "pts_allow": self.pa.to_dict() if self.pa else None}

    @classmethod
    def from_dict(cls, payload: Mapping, scoring: Mapping[str, float]) -> _DefenseHead:
        head = cls(scoring, n_pa_bins=int(payload.get("n_pa_bins", DEFAULT_PA_BINS)))
        head.counting = _CountingHead.from_dict(payload["counting"])
        head.pa = _PtsAllowDist.from_dict(payload["pts_allow"]) if payload.get("pts_allow") else None
        return head


def _score_stat_lines(
    counting: Mapping[str, np.ndarray],
    bucket_probs: Mapping[str, np.ndarray],
    scoring: Mapping[str, float],
    n: int,
) -> np.ndarray:
    """Price predicted stat lines through ``scoring.engine.points`` — row by row, never a points head.

    Each row's line is its expected counts plus (for DST) the **probability mass per points-allowed
    bucket**; because the engine is linear, ``engine({bucket: P(bucket)}, scoring)`` is exactly
    ``E[pts_allow points]``. This is the concrete form of Decision #2: the model emits a stat line (a
    distribution, for the bucketed part) and the engine scores it.
    """
    out = np.zeros(n, dtype="float64")
    for i in range(n):
        line = {k: float(v[i]) for k, v in counting.items()}
        for k, v in bucket_probs.items():
            line[k] = line.get(k, 0.0) + float(v[i])
        out[i] = engine_points(line, scoring)
    return out


def _fit_to_dict(fit: _PositionFit) -> dict:
    return {"mean": [float(x) for x in fit.mean], "std_safe": [float(x) for x in fit.std_safe],
            "weights": [float(x) for x in fit.weights], "intercept": float(fit.intercept),
            "y_mean": float(fit.y_mean)}


def _fit_from_dict(payload: Mapping) -> _PositionFit:
    return _PositionFit(
        mean=np.asarray(payload["mean"], dtype="float64"),
        std_safe=np.asarray(payload["std_safe"], dtype="float64"),
        weights=np.asarray(payload["weights"], dtype="float64"),
        intercept=float(payload["intercept"]),
        y_mean=float(payload["y_mean"]),
    )


# =============================================================== the per-cell measured gate
_BASELINE_FACTORIES: dict[str, type] = {
    "TrailingMean": TrailingMean,
    "PriorSeasonRank": PriorSeasonRank,
    "LaggedExpectedPoints": LaggedExpectedPoints,
}

#: The baseline a deferred cell predicts with — last season's per-week level, present at week 1 and the
#: toughest baseline on the cold cells where deferral matters most (the #29 choice).
_DEFER_TO = PriorSeasonRank


def cell_key(position: str, cold: bool) -> str:
    """The (position × cohort) cell identifier, e.g. ``"K:cold"`` / ``"DEF:warm"``."""
    return f"{position}:{'cold' if cold else 'warm'}"


def cell_metrics(
    predictions: pd.DataFrame, frame: pd.DataFrame, *, positions: Sequence[str] = KICKDEF_POSITIONS
) -> dict[str, PositionMetrics]:
    """Per-(position × cohort) metrics over a walk-forward result — four cells for K + DEF.

    The out-of-sample ``predictions`` carry no ``games_played_prior``, so the cold cohort is recovered by
    a 1:1 join back to ``frame`` on the natural key (every frame row is one player-week).
    """
    cold = frame.loc[is_cold_start(frame), ["player_id", "season", "week"]].copy()
    cold["_cold"] = True
    merged = predictions.merge(cold, on=["player_id", "season", "week"], how="left")
    is_cold = merged["_cold"].fillna(False).to_numpy()
    out: dict[str, PositionMetrics] = {}
    for cohort_cold in (False, True):
        rows = merged[is_cold == cohort_cold]
        for pos, pm in per_position_metrics(rows, positions=positions).items():
            out[cell_key(pos, cohort_cold)] = pm
    return out


def deferred_cells(
    component: Mapping[str, PositionMetrics],
    baselines: Mapping[str, Mapping[str, PositionMetrics]],
    *,
    positions: Sequence[str] = KICKDEF_POSITIONS,
    min_n: int = _MIN_CELL_N,
) -> tuple[str, ...]:
    """The cells where the component model does **not** earn its place — the measured deferral set.

    A cell is fielded only when the component model beats **every** baseline on **both** MAE (lower) and
    within-slate ρ (higher) **and** rests on at least ``min_n`` held-out rows. Everything else defers:
    a loss on either metric, a tie (ties defer), or too little evidence to decide (a thin cell must not
    read like a thick one). Pure — the same function the eval and the fit call.
    """
    out: list[str] = []
    for pos in positions:
        for cold in (False, True):
            cell = cell_key(pos, cold)
            c = component.get(cell)
            base = {name: b[cell] for name, b in baselines.items() if cell in b}
            maes = [m.mae for m in base.values()]
            rhos = [m.spearman for m in base.values() if m.spearman is not None]
            best_mae = min(maes) if maes else None
            best_rho = max(rhos) if rhos else None
            wins = (
                c is not None
                and c.n >= min_n
                and c.spearman is not None
                and best_mae is not None
                and best_rho is not None
                and c.mae < best_mae - _TIE_EPS
                and c.spearman > best_rho + _TIE_EPS
            )
            if not wins:
                out.append(cell)
    return tuple(out)


# =============================================================== the shipped model
DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("fit") / "kickdef.json"


@functools.cache
def recorded_gate(path: str | Path = DEFAULT_ARTIFACT_PATH) -> tuple[str, ...]:
    """The measured per-cell deferral gate from the committed artifact — the **safe default**.

    A default-constructed :class:`KickDefModel` reads its gate here, so it ships the deferring
    configuration the artifact recorded, never the ungated component model. If the artifact is missing
    (a fresh checkout before ``eval_kickdef`` has run), it falls back to **every** cell — defer
    everything to the baseline, the safe side, never ``()`` (which would field an unproven component
    model silently). Cached, so the read happens once per process.
    """
    try:
        gate = json.loads(Path(path).read_text(encoding="utf-8"))["deferral"]
        return tuple(str(c) for c in gate)
    except (OSError, ValueError, KeyError, TypeError):
        return tuple(cell_key(p, cold) for p in KICKDEF_POSITIONS for cold in (False, True))


class KickDefModel:
    """The **shipped** K + DST model: component heads through the engine, per-cell measured deferral.

    Fields the component head (kicker or defence) for a (position × cohort) cell where it was **measured**
    to beat the baselines on both metrics; **defers** the cell to :class:`model.baselines.PriorSeasonRank`
    where it was not — a per-cell gate, so a warm win is not thrown away by a cold loss (Decision #9 item
    6). ``defer`` defaults to :func:`recorded_gate` (the committed measured gate), so a bare
    ``KickDefModel()`` is the shipped, safe-by-default configuration; ``KickDefModel(defer=())`` is the
    pure-component diagnostic opt-out (the analogue of ``WeeklyModel(defer_cold_start=())``).

    ``predict`` on a deferred cell returns **exactly** the baseline's number (pinned by a test); a model
    whose baseline is unfit (straight from :meth:`load_fitted`) **raises** on a deferred row rather than
    silently falling back to the ungated component model.
    """

    def __init__(
        self,
        scoring: Mapping[str, float] | None = None,
        *,
        alpha: float = _RIDGE_ALPHA,
        n_pa_bins: int = DEFAULT_PA_BINS,
        defer: Iterable[str] | None = None,
        positions: Sequence[str] = KICKDEF_POSITIONS,
    ) -> None:
        self.scoring: dict[str, float] = dict(scoring) if scoring else {}
        self.alpha = float(alpha)
        self.n_pa_bins = int(n_pa_bins)
        self.positions = tuple(positions)
        self.defer = tuple(recorded_gate() if defer is None else defer)
        self._kicker: _KickerHead | None = None
        self._defense: _DefenseHead | None = None
        self._prior = _DEFER_TO()
        self._prior_fitted = False

    def fit(self, frame: pd.DataFrame) -> KickDefModel:
        if not self.scoring:
            raise ValueError(
                "KickDefModel needs the league's scoring_settings to score components through the "
                "engine — construct it as KickDefModel(scoring) (eval_kickdef passes the live scoring)."
            )
        if "K" in self.positions:
            self._kicker = _KickerHead(self.scoring, self.alpha).fit(frame[frame["position"] == "K"])
        if "DEF" in self.positions:
            self._defense = _DefenseHead(self.scoring, alpha=self.alpha, n_pa_bins=self.n_pa_bins).fit(
                frame[frame["position"] == "DEF"]
            )
        self._prior.fit(frame)
        self._prior_fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=frame.index, dtype="float64")
        pos = frame["position"].astype("string")
        cold = is_cold_start(frame)
        prior: pd.Series | None = None
        for position, head in (("K", self._kicker), ("DEF", self._defense)):
            if position not in self.positions:
                continue
            for cold_val in (False, True):
                mask = ((pos == position) & (cold == cold_val)).to_numpy()
                if not mask.any():
                    continue
                idx = np.where(mask)[0]
                if cell_key(position, cold_val) in self.defer or head is None:
                    if not self._prior_fitted:
                        raise RuntimeError(
                            f"KickDefModel defers cell {cell_key(position, cold_val)} to PriorSeasonRank, "
                            "but its baseline is unfit — call .fit(frame) before predicting deferred rows "
                            "(load_fitted returns an unfit baseline on purpose; fit it on recent data)."
                        )
                    if prior is None:
                        prior = self._prior.predict(frame)
                    out.iloc[idx] = prior.iloc[idx].to_numpy()
                else:
                    out.iloc[idx] = head.predict(frame.iloc[idx])
        return out

    # ----------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        return {
            "model": "KickDefModel",
            "features": list(KICKDEF_FEATURES),
            "alpha": self.alpha,
            "n_pa_bins": self.n_pa_bins,
            "scoring": dict(self.scoring),
            "defer": list(self.defer),
            "kicker": self._kicker.to_dict() if self._kicker else None,
            "defense": self._defense.to_dict() if self._defense else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> KickDefModel:
        if list(payload.get("features", [])) != list(KICKDEF_FEATURES):
            raise ValueError(
                "fitted artifact's feature list does not match model.kickdef.KICKDEF_FEATURES — the "
                "frame's columns drifted since it was fit; regenerate with scripts/eval_kickdef.py"
            )
        scoring = dict(payload.get("scoring", {}))
        model = cls(
            scoring,
            alpha=float(payload.get("alpha", _RIDGE_ALPHA)),
            n_pa_bins=int(payload.get("n_pa_bins", DEFAULT_PA_BINS)),
            defer=tuple(str(c) for c in payload.get("defer", ())),
        )
        if payload.get("kicker"):
            model._kicker = _KickerHead.from_dict(payload["kicker"], scoring)
        if payload.get("defense"):
            model._defense = _DefenseHead.from_dict(payload["defense"], scoring)
        return model

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> KickDefModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_fitted(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> KickDefModel:
        """Load the committed artifact into a correctly-configured model: heads **and** the measured gate.

        The runtime path from generated artifact to working model (what #34 loads). The component heads
        are reconstructed (warm rows predict immediately) and ``defer`` is the gate the artifact records.
        The deferral baseline is a cheap group mean re-fit from live data, so it is **not** stored; call
        ``.fit(frame)`` with recent data before predicting deferred rows (``predict`` raises otherwise).
        """
        return cls.load(path)


# =============================================================== evaluation convenience
def component_model(scoring: Mapping[str, float], *, n_pa_bins: int = DEFAULT_PA_BINS) -> KickDefModel:
    """The pure-component model (no deferral) — the diagnostic the gate is measured against."""
    return KickDefModel(scoring, n_pa_bins=n_pa_bins, defer=())


#: The left-tail cells the calibration probes — the two highest-value points-allowed buckets, where a
#: single homoskedastic residual grid quietly fails (module docstring). ``0`` is the shutout; ``6`` is
#: the top of ``pts_allow_1_6``, so ``≤ 6`` is the combined top-two cells — enough held-out rows to read
#: where the ~1%-of-rows shutout alone is too thin.
_LEFT_TAIL_THRESHOLDS: tuple[int, ...] = (0, 6)


def pts_allow_calibration(
    frame: pd.DataFrame,
    scoring: Mapping[str, float],
    *,
    n_pa_bins: int = DEFAULT_PA_BINS,
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
    deciles: int = 10,
) -> pd.DataFrame:
    """Predicted-vs-realized ``P(pts_allow ≤ t)`` by predicted-``μ`` decile — the left-tail check.

    Walk-forward and held-out: for each test season the defence distribution is fit on strictly-earlier
    seasons and predicts the shutout / ``≤ 6`` probability on the season's DEF rows, which are then binned
    by predicted ``μ``. A well-calibrated left tail has predicted rates tracking realized ones across the
    deciles — the direct check on the ``pts_allow_0`` (10-point) cell that a homoskedastic grid gets
    wrong. Columns: ``decile``, ``n``, ``mu_mean``, and ``pred_le{t}`` / ``real_le{t}`` per threshold.
    """
    dst = frame[frame["position"] == "DEF"]
    chunks: list[pd.DataFrame] = []
    for split in walk_forward_splits(dst, test_seasons=test_seasons):
        head = _DefenseHead(scoring, n_pa_bins=n_pa_bins).fit(split.train)
        if head.pa is None:
            continue
        x = _feature_matrix(split.test)
        actual = pd.to_numeric(split.test.get(f"{COMP_PREFIX}pts_allow"), errors="coerce")
        piece = pd.DataFrame({"mu": head.pa.predict_mu(x), "actual": actual.to_numpy()})
        for t in _LEFT_TAIL_THRESHOLDS:
            piece[f"pred_le{t}"] = head.pa.predict_prob_leq(x, t)
        chunks.append(piece)
    if not chunks:
        return pd.DataFrame()
    allrows = pd.concat(chunks, ignore_index=True).dropna(subset=["mu", "actual"])
    if allrows.empty:
        return pd.DataFrame()
    codes = pd.qcut(allrows["mu"].rank(method="first"), min(deciles, len(allrows)), labels=False)
    allrows["decile"] = codes.astype(int) + 1
    out_rows: list[dict] = []
    for decile, grp in allrows.groupby("decile", sort=True):
        row = {"decile": int(decile), "n": int(len(grp)), "mu_mean": float(grp["mu"].mean())}
        for t in _LEFT_TAIL_THRESHOLDS:
            row[f"pred_le{t}"] = float(grp[f"pred_le{t}"].mean())
            row[f"real_le{t}"] = float((grp["actual"] <= t).mean())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def measure_gate(
    frame: pd.DataFrame,
    scoring: Mapping[str, float],
    *,
    n_pa_bins: int = DEFAULT_PA_BINS,
    positions: Sequence[str] = KICKDEF_POSITIONS,
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
    min_n: int = _MIN_CELL_N,
) -> tuple[str, ...]:
    """Measure the per-cell gate by walk-forward: which cells the component model does not win.

    Runs the pure-component model and the three baselines through the harness, slices each into the four
    cells, and returns :func:`deferred_cells`. The same measurement ``scripts/eval_kickdef.py`` reports;
    a fit uses it so the committed gate is measured, not hand-set.
    """
    comp = evaluate(
        component_model(scoring, n_pa_bins=n_pa_bins), frame, positions=positions, test_seasons=test_seasons
    )
    comp_cells = cell_metrics(comp.predictions, frame, positions=positions)
    base_cells = {
        name: cell_metrics(
            evaluate(factory(), frame, positions=positions, test_seasons=test_seasons).predictions,
            frame,
            positions=positions,
        )
        for name, factory in _BASELINE_FACTORIES.items()
    }
    return deferred_cells(comp_cells, base_cells, positions=positions, min_n=min_n)

"""The weekly point model for the skill positions QB/RB/WR/TE (Phase 9, ticket #29).

The core weekly regression: predict ``y_custom_points`` at player x week, pre-lock. It is graded
against the three naive baselines in :mod:`model.baselines` and must beat **all three** on **both**
MAE and within-slate Spearman rho, **per position** — the recorded bar in ``docs/model-baselines.md``
(spec ticket #29). The measured grade lives in ``docs/model-weekly.md`` (produced by
``scripts/eval_weekly.py``); this module is only the model.

The learner: a per-position ridge, reusing #31's solver
--------------------------------------------------------
The spec caps this phase at gradient boosting and regularised linear models, and names the regularised
linear model as the sanity floor: *"if boosting cannot beat ridge, the features are the problem, not
the learner."* This module is that floor, and it deliberately carries **no new dependency** — the same
choice #28 made (``np.corrcoef`` over scipy) and #31 made (a closed-form ridge over sklearn). The
solver is #31's: :func:`model.season._fit_ridge` / :class:`model.season._PositionFit` are imported and
reused verbatim rather than re-derived, so the standardise-impute-solve discipline (unpenalised
intercept, ``_STD_EPS`` guard on constant columns, deterministic ``solve`` with an ``lstsq`` fallback)
is shared with the draft model. That is a deliberate cross-module reuse within ``src/model/``;
consolidating it into a shared module is deferred to #34, the ticket sanctioned to touch existing
modules. Whether ridge is enough is a question for the real lake, not this docstring — if it misses at
a position the eval script reports the gap in numbers and the learner decision is escalated, never
reached for silently.

Why a **points head**, and the condition it rests on (Decision #7)
------------------------------------------------------------------
The model regresses the already-engine-scored label ``y_custom_points`` directly, rather than
predicting stat components and scoring them (the Decision #2 discipline #30 uses for K/DST). This is
legitimate **only because this league's skill scoring is linear in the stats** — every skill key is a
per-unit coefficient (``rec`` 0.5, ``rush_yd`` 0.1, ``pass_td`` 4, ...), with no bucket or threshold.
For a linear scoring function ``E[points] = sum(coef * E[stat])``, so ``score(E[X]) = E[score(X))``:
a points head is unbiased and identical in expectation to a component head, at no accuracy cost. K and
DST differ **in kind** — their points are step functions of stat buckets (``fgm_40_49`` 4 vs
``fgm_50p`` 5; ``pts_allow_0`` 10 vs ``pts_allow_1_6`` 7), where ``E[f(X)] != f(E[X])`` and only a
component model is unbiased. That is why the component rule is #30's and must **not** be read here as
license for a points head there.

The equivalence holds only while scoring stays linear, and ``sleeper.config.LEAGUE_ID`` still points at
the 2026 **test** sandbox — the real 2026 league does not exist yet and CLAUDE.md says to re-confirm its
scoring against the API. :func:`assert_linear_skill_scoring` is the fail-closed guard: it reads a live
``scoring_settings`` and **raises** if any skill-relevant key is bonus/threshold-shaped, so a real
league that comes back with a yardage bonus breaks the fit loudly instead of biasing the model
silently. ``scripts/eval_weekly.py`` and ``scripts/fit_weekly.py`` call it against the live scoring they
load; ``tests/test_model_weekly.py`` pins its logic offline.

Cold start (spec acceptance #2)
-------------------------------
Week 1 of every season has no current-season lags (``points_last`` and the rest are null by
construction — profile #27 §3). There is **no separate code path**: nulls are imputed to the training
column mean and standardised, so a week-1 row lands at the standardised origin on its lag features and
is predicted at the position level shifted by the features that *are* known week 1 — the Vegas market
(``implied_team_total``, ``team_spread_line``, ``total_line``) and ``is_indoor``. That is exactly where
the lag-blind baselines fall back to a flat position mean, so week 1 is scored **separately** in the
report (#28 measured every week-1 board a pure fallback for them) rather than hidden in a season mean.

Immunity to the #31 warm-up trap
--------------------------------
Every feature here is a within-``(player, season)`` lag or a same-week market/venue value; none is
derived from a player's *first appearance in the collection window*. So unlike the season frame's
``is_rookie`` (100% true by arithmetic in the earliest built season), this model has nothing that is
systematically wrong in 2016-2017 — those seasons train on valid within-season lags and are simply
never scored (``DEFAULT_TEST_SEASONS`` starts 2018).
"""

from __future__ import annotations

import functools
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from model.baselines import LaggedExpectedPoints, PriorSeasonRank, TrailingMean
from model.evaluate import (
    DEFAULT_TEST_SEASONS,
    LABEL_COL,
    PositionMetrics,
    evaluate,
    per_position_metrics,
)
from model.season import _PositionFit, _fit_ridge  # deliberate reuse of #31's solver (see docstring)

_LOG = logging.getLogger(__name__)

#: The positions this model owns. K and DST are #30's (component-wise through the engine), so they are
#: deliberately absent here — a points head is invalid for their bucketed scoring (Decision #7).
SKILL_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

#: The model's features, all provably pre-lock and populated over the full 2016+ training span
#: (profile #27 §5b). Three families, in this order:
#:
#: * **usage lags (13)** — nflverse actuals/snaps/opportunity, lagged within (player, season), so the
#:   content rule keeps them legal (``dataset.assemble._lagged_usage``); null on a player's first
#:   appearance of a season (the cold-start cohort).
#: * **market (4)** — the Vegas implied totals and lines, the sanctioned pre-game view, 0% null every
#:   season and **present at week 1** — the model's cold-start edge over the lag-based baselines.
#: * **context (2)** — divisional game and indoor venue, both fixed well before kickoff.
#:
#: Everything else the frame carries is excluded *because #27 measured it unusable over this span*, and
#: that exclusion is a reported finding (spec acceptance #3), not an oversight: ``depth_*`` is 2025+
#: (~89-100% null), ``wx_forecast_*`` / ``wx_observed_*`` are 100% null (endpoint reach / withheld
#: pre-lock), ``inj_sleeper_*`` and ``baseline_sleeper_points`` are forward-only (2026+), and the
#: ``inj_report_*`` columns are ~95% null *and* their strongest cases — inactive "Out" players — have no
#: stat-line row, so never enter the frame at all.
WEEKLY_FEATURES: tuple[str, ...] = (
    # usage lags
    "games_played_prior",
    "points_last",
    "points_ewma",
    "points_trend",
    "snap_pct_last",
    "snap_pct_ewma",
    "snap_pct_trend",
    "target_share_last",
    "target_share_ewma",
    "rush_share_last",
    "rush_share_ewma",
    "exp_points_last",
    "exp_points_ewma",
    # market
    "implied_team_total",
    "opp_implied_total",
    "team_spread_line",
    "total_line",
    # context
    "is_div_game",
    "is_indoor",
)

#: Ridge penalty on the standardised features. Modest by design and **not** tuned on any test season —
#: the row counts (5k-19k per position) and feature count (19) are close to the draft model's, where the
#: same value stabilises a near-collinear block without shrinking the fit to the mean.
_RIDGE_ALPHA = 1.0

# --------------------------------------------------------------------------- the linearity guard
#: Skill scoring keys known to be **linear** in the underlying stat count (a per-unit coefficient).
#: The points-head equivalence ``E[points] = sum(coef * E[stat])`` holds iff every non-zero
#: skill-relevant key in the live scoring is one of these. Curated from the league's 42-key table
#: (CLAUDE.md); the common per-unit variants a future league might add (completions, attempts, targets,
#: first downs) are included so the guard fires on *bonuses*, not on benign linear additions.
LINEAR_SKILL_KEYS: frozenset[str] = frozenset(
    {
        "pass_yd", "pass_td", "pass_2pt", "pass_int", "pass_cmp", "pass_att", "pass_inc",
        "pass_fd", "pass_sack",
        "rush_yd", "rush_td", "rush_2pt", "rush_att", "rush_fd",
        "rec", "rec_yd", "rec_td", "rec_2pt", "rec_tgt", "rec_fd",
        "fum", "fum_lost", "fum_rec", "fum_rec_td",
    }
)

#: A key ending in a number, a numeric range, or a distance band (``_100``, ``_1_6``, ``_40p``,
#: ``_35p``) is a bucket/threshold — a step in the scoring function, not a per-unit rate.
_THRESHOLD_RE = re.compile(r"_\d+(?:_\d+|p)?$")

#: Tokens that mark a key as K or DST rather than skill — so a ``bonus_def_*`` (#30's concern) is not
#: mistaken for a skill bonus. Kept as substrings because Sleeper composes them (``def_st_ff``).
_NON_SKILL_TOKENS: tuple[str, ...] = (
    "def", "_st_", "allow", "fgm", "fgmiss", "xp", "sack", "tkl", "safe", "blk", "idp",
)


def _is_skill_relevant(key: str) -> bool:
    """Does this scoring key pertain to QB/RB/WR/TE production?

    Passing, rushing, receiving and fumbles by prefix, plus any ``bonus`` key that is **not** clearly K
    or DST — Sleeper's skill milestones (``bonus_rush_yd_100``, ``bonus_rec_yd_100``, a generic
    ``bonus_fd``) all land here, while ``bonus_def_*`` and the K/DST buckets (``fgm_*``,
    ``pts_allow_*``, ``sack``, ...) are excluded because their non-linearity is exactly what #30 owns.
    """
    if key in LINEAR_SKILL_KEYS:
        return True
    if key.startswith(("pass_", "rush_", "rec_")) or key == "rec" or key.startswith("fum"):
        return True
    if "bonus" in key and not any(tok in key for tok in _NON_SKILL_TOKENS):
        return True
    return False


def assert_linear_skill_scoring(scoring: Mapping[str, float]) -> None:
    """Raise ``ValueError`` unless every non-zero skill-relevant scoring key is linear (Decision #7).

    Fail-closed in two tiers, both of which invalidate the points head:

    * a skill-relevant key that is **bonus/threshold-shaped** (``bonus_rush_yd_100``, any distance/range
      suffix) — a definite non-linearity, and
    * a skill-relevant key with a non-zero weight that is **not recognised** as linear — one we cannot
      *prove* linear, so it must be confirmed and added to :data:`LINEAR_SKILL_KEYS` by hand rather than
      trusted.

    Called against the **live** ``scoring_settings`` by the fit and eval scripts, so a real 2026 league
    that returns with a yardage bonus breaks the run loudly. A zero-weight key (e.g. ``fum`` 0.0) does
    not affect points and is ignored.
    """
    if not scoring:
        raise ValueError("scoring is empty — pass the league's live scoring_settings dict")
    nonlinear: list[str] = []
    unknown: list[str] = []
    for key, coef in scoring.items():
        if not coef or not _is_skill_relevant(key):
            continue
        if "bonus" in key.lower() or _THRESHOLD_RE.search(key):
            nonlinear.append(f"{key}={coef}")
        elif key not in LINEAR_SKILL_KEYS:
            unknown.append(f"{key}={coef}")
    if nonlinear:
        raise ValueError(
            "non-linear skill scoring — the weekly points head is invalid for this league. "
            f"Bucket/threshold/bonus key(s): {sorted(nonlinear)}. A points head assumes "
            "E[points] = sum(coef * E[stat]); a bonus breaks that (E[f(X)] != f(E[X])). Model the "
            "components through scoring.engine instead (the #30 discipline), or remove the key."
        )
    if unknown:
        raise ValueError(
            "unrecognised skill scoring key(s) — cannot prove the points head is unbiased. "
            f"Key(s): {sorted(unknown)}. Confirm each is a per-unit (linear) coefficient and add it to "
            "model.weekly.LINEAR_SKILL_KEYS, or model it component-wise if it is a bucket."
        )


# --------------------------------------------------------------------------- feature matrix
def _to_float(col: pd.Series) -> np.ndarray:
    """A column as ``float64`` with nulls preserved as ``NaN`` (booleans → 1.0/0.0, ``<NA>`` → ``NaN``).

    ``is_indoor`` is a nullable boolean and ``is_div_game`` a nullable int, so a plain ``astype(float)``
    would raise on the ``<NA>`` cells; mapping the booleans and coercing the rest keeps a missing venue
    a genuine ``NaN`` for imputation rather than a spurious 0.
    """
    if col.dtype == bool or str(col.dtype) == "boolean":
        col = col.astype("object").map({True: 1.0, False: 0.0})
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The model features as a float ``(n, k)`` array; a column absent from the frame is all-``NaN``.

    Null-preserving on purpose: :class:`model.season._PositionFit` imputes each ``NaN`` to the fitted
    training mean and standardises, which is the cold-start path (week-1 lags are ``NaN``).
    """
    cols = [
        _to_float(frame[name]) if name in frame.columns else np.full(len(frame), np.nan)
        for name in WEEKLY_FEATURES
    ]
    return np.column_stack(cols) if cols else np.empty((len(frame), 0))


# --------------------------------------------------------------------------- the model
class WeeklyRidge:
    """Per-position ridge on the weekly features — a :class:`model.evaluate.Predictor`.

    ``fit`` learns one standardised ridge per skill position present in the training frame (reusing
    :func:`model.season._fit_ridge`); ``predict`` returns a Series index-aligned to its input, ``NaN``
    for any position it did not fit (K/DST, which are #30's). It fits **every** skill position it sees —
    there is no internal deferral: which positions actually cleared the recorded bar is a question the
    eval report answers on the real lake, and wiring the per-position fall back to Sleeper is #34's job.
    """

    def __init__(
        self, alpha: float = _RIDGE_ALPHA, *, positions: Sequence[str] = SKILL_POSITIONS
    ) -> None:
        self.alpha = float(alpha)
        self.positions = tuple(positions)
        self._fits: dict[str, _PositionFit] = {}
        self.fit_positions: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame) -> WeeklyRidge:
        self._fits = {}
        pos = frame["position"].astype("string")
        modeled: list[str] = []
        for position in self.positions:
            sub = frame[pos == position]
            if sub.empty:
                continue
            y = pd.to_numeric(sub[LABEL_COL], errors="coerce").to_numpy(dtype="float64")
            keep = ~np.isnan(y)
            if int(keep.sum()) == 0:
                continue
            self._fits[position] = _fit_ridge(_feature_matrix(sub)[keep], y[keep], self.alpha)
            modeled.append(position)
        self.fit_positions = tuple(modeled)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=frame.index, dtype="float64")
        pos = frame["position"].astype("string")
        for position, fit in self._fits.items():
            mask = (pos == position).to_numpy()
            if not mask.any():
                continue
            out.iloc[np.where(mask)[0]] = fit.predict(_feature_matrix(frame[mask]))
        return out

    # ----------------------------------------------------------------- feature importances
    def feature_importances(self) -> dict[str, dict[str, float]]:
        """``{position: {feature: standardised coefficient}}`` for each fitted position.

        The features are standardised before the fit, so the coefficient magnitude *is* the importance
        (a one-standard-deviation move in the feature shifts the prediction by ``|coef|`` points),
        directly comparable across features. A constant/all-null column was zeroed by the ``_STD_EPS``
        guard in the solve, so it reads as exactly 0 here — which is how the report shows the model does
        not lean on a feature #27 found mostly null (spec acceptance #3).
        """
        return {
            position: {feat: float(w) for feat, w in zip(WEEKLY_FEATURES, fit.weights, strict=True)}
            for position, fit in self._fits.items()
        }

    # ----------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        """A JSON-serialisable snapshot of the fitted model — the committed-artifact payload.

        Deterministic: ridge is a closed-form solve, so re-fitting the same frame reproduces these
        numbers exactly, which is what makes the artifact regenerable rather than hand-edited (spec
        acceptance #4). ``features`` is stored so a load can refuse a frame whose columns drifted.
        """
        return {
            "model": "WeeklyRidge",
            "alpha": self.alpha,
            "features": list(WEEKLY_FEATURES),
            "positions": {
                position: {
                    "mean": [float(x) for x in fit.mean],
                    "std_safe": [float(x) for x in fit.std_safe],
                    "weights": [float(x) for x in fit.weights],
                    "intercept": float(fit.intercept),
                    "y_mean": float(fit.y_mean),
                }
                for position, fit in self._fits.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> WeeklyRidge:
        if list(payload.get("features", [])) != list(WEEKLY_FEATURES):
            raise ValueError(
                "fitted artifact's feature list does not match model.weekly.WEEKLY_FEATURES — the "
                "frame's columns drifted since it was fit; regenerate with scripts/fit_weekly.py"
            )
        model = cls(alpha=float(payload.get("alpha", _RIDGE_ALPHA)))
        model._fits = {
            position: _PositionFit(
                mean=np.asarray(spec["mean"], dtype="float64"),
                std_safe=np.asarray(spec["std_safe"], dtype="float64"),
                weights=np.asarray(spec["weights"], dtype="float64"),
                intercept=float(spec["intercept"]),
                y_mean=float(spec["y_mean"]),
            )
            for position, spec in payload.get("positions", {}).items()
        }
        model.fit_positions = tuple(model._fits)
        return model

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WeeklyRidge:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- cold-start deferral
#: The within-season game count (``dataset.assemble`` cumcount), 0 on a player's **first appearance of
#: a season** — the exact marker of the cold-start cohort (week 1 for most, a mid-season debut for the
#: rest). Never null in the real frame, so it is a cleaner gate than ``week == 1``, which would miss
#: the debuts the deferral is also meant to cover.
COLD_START_MARKER = "games_played_prior"

#: The baseline the cold start defers to: last season's per-week average, the strongest pre-lock signal
#: on a week with no within-season history. K/DST are #30's, so it is fielded here only for the skills.
_DEFER_TO = PriorSeasonRank

_BASELINE_FACTORIES: dict[str, type] = {
    "TrailingMean": TrailingMean,
    "PriorSeasonRank": PriorSeasonRank,
    "LaggedExpectedPoints": LaggedExpectedPoints,
}


def is_cold_start(frame: pd.DataFrame) -> pd.Series:
    """A boolean Series: rows with **no within-season lag** — the cold-start cohort.

    ``games_played_prior == 0`` is a player's first appearance of the season. If the marker is absent
    (a narrowed frame), every row is treated as cold — the safe side, since deferring to the
    prior-season baseline never does worse than a lag-less ridge on a lag-less row.
    """
    if COLD_START_MARKER not in frame.columns:
        return pd.Series(True, index=frame.index)
    g = pd.to_numeric(frame[COLD_START_MARKER], errors="coerce")
    return g.fillna(0) == 0


#: The committed fitted artifact — the deployment ridge, plus the measured cold-start gate and its
#: margins. Written by ``scripts/eval_weekly.py``; read here so the artifact is a *source*, not a
#: write-only record.
DEFAULT_ARTIFACT_PATH = Path(__file__).with_name("fit") / "weekly_ridge.json"


@functools.cache
def recorded_cold_start_gate(path: str | Path = DEFAULT_ARTIFACT_PATH) -> tuple[str, ...]:
    """The measured cold-start deferral gate recorded in the committed artifact — the **safe default**.

    This is the path from "generated fitted artifact" to "correctly configured model": a
    default-constructed :class:`WeeklyModel` reads the gate from here rather than shipping the pure-ridge
    variant the artifact itself records as losing the cold start. If the artifact is unavailable (a fresh
    checkout before ``eval_weekly`` has run), it falls back to **every** skill position — never to ``()``,
    because a silent pure-ridge default is exactly the fallback-discovered-in-production the spec warns
    against (finding #4). Cached, so the read happens once per process.
    """
    try:
        gate = json.loads(Path(path).read_text(encoding="utf-8"))["cold_start_deferral"]
        return tuple(str(p) for p in gate)
    except (OSError, ValueError, KeyError, TypeError):
        return SKILL_POSITIONS


class WeeklyModel:
    """The **shipped** weekly model: ridge, deferring cold-start rows to the prior-season baseline.

    The deferral is the exact shape of :class:`model.season.SeasonModel`'s — field the stronger predictor
    where it is stronger, defer where it is not — but the gate here is **measured per position**, not a
    data-availability rule. ``defer_cold_start`` is the set of positions at which pure ridge was measured
    to *lose* the cold start (``docs/model-weekly.md`` §B / :func:`cold_start_deferral`); at those
    positions a cold-start row (no within-season lag) is handed to :class:`model.baselines.PriorSeasonRank`
    and every other row to the ridge. Where ridge wins the cold start, the set omits the position and the
    ridge is fielded there too — the deferral is a measured gate, never a blanket rule (spec ticket #29 /
    the review of #31's blanket DEF deferral).

    **Safe by default.** ``defer_cold_start`` defaults to :func:`recorded_cold_start_gate` — the measured
    gate read from the committed artifact — so a bare ``WeeklyModel()`` defers where the evidence says to,
    exactly as :class:`model.season.SeasonModel` is safe (``require_usage=True``) by default. The
    diagnostic **pure-ridge** variant is the explicit opt-out ``WeeklyModel(defer_cold_start=())`` — the
    analogue of ``SeasonModel(require_usage=False)`` — kept reachable because it is a legitimate
    comparison, but never what you get by accident.

    ``predict`` on a deferred cold-start row returns **exactly** the contained baseline's number, not an
    approximation — pinned by a test, the way #31 pins its DEF deferral. A model whose prior is unfit
    (e.g. straight from :meth:`load_fitted`) **raises** on a deferred cold-start row rather than silently
    falling back to pure ridge.
    """

    def __init__(
        self,
        *,
        alpha: float = _RIDGE_ALPHA,
        defer_cold_start: Iterable[str] | None = None,
        positions: Sequence[str] = SKILL_POSITIONS,
    ) -> None:
        # None → the measured gate from the artifact (safe default); an explicit () is pure ridge.
        gate = recorded_cold_start_gate() if defer_cold_start is None else defer_cold_start
        self.defer_cold_start = tuple(gate)
        self.positions = tuple(positions)
        self._ridge = WeeklyRidge(alpha=alpha, positions=positions)
        self._prior = _DEFER_TO()
        self._prior_fitted = False

    def fit(self, frame: pd.DataFrame) -> WeeklyModel:
        self._ridge.fit(frame)
        self._prior.fit(frame)
        self._prior_fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        out = self._ridge.predict(frame).astype("float64")
        if not self.defer_cold_start:
            return out
        defer = (
            is_cold_start(frame) & frame["position"].astype("string").isin(self.defer_cold_start)
        ).to_numpy()
        if defer.any():
            if not self._prior_fitted:
                raise RuntimeError(
                    "WeeklyModel defers cold-start rows to PriorSeasonRank, but its prior is unfit — "
                    "call .fit(frame) before predicting cold-start rows so the deferral is never a "
                    "silent pure-ridge fallback (WeeklyModel.load_fitted returns an unfit prior on "
                    "purpose; fit it on recent data before serving cold-start weeks)."
                )
            prior = self._prior.predict(frame)
            idx = np.where(defer)[0]
            out.iloc[idx] = prior.iloc[idx].to_numpy()
        return out

    def feature_importances(self) -> dict[str, dict[str, float]]:
        return self._ridge.feature_importances()

    @classmethod
    def load_fitted(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> WeeklyModel:
        """Load the committed artifact into a correctly-configured model: recorded ridge **and** gate.

        The runtime path from the generated artifact to a working model (what #34 loads). The ridge
        weights are reconstructed (warm rows predict immediately) and ``defer_cold_start`` is set to the
        gate the artifact records — so the loaded model is the deferring variant, never pure ridge. The
        prior-season deferral target is a cheap group mean re-fit from live data, so it is **not** stored;
        call ``.fit(frame)`` with recent data before predicting cold-start rows (``predict`` raises
        otherwise). Fitting also refreshes the ridge on that data, which is the intended serving flow.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            alpha=float(payload.get("alpha", _RIDGE_ALPHA)),
            defer_cold_start=tuple(str(p) for p in payload.get("cold_start_deferral", ())),
        )
        model._ridge = WeeklyRidge.from_dict(payload)
        return model


# --------------------------------------------------------------------------- the measured gate
def cold_start_metrics(
    predictions: pd.DataFrame, frame: pd.DataFrame, *, positions: Sequence[str] = SKILL_POSITIONS
) -> dict[str, PositionMetrics]:
    """Per-position metrics over the **cold-start rows** of a walk-forward result's predictions.

    The out-of-sample ``predictions`` (``player_id, season, week, position, actual, pred``) carry no
    ``games_played_prior``, so the cold-start cohort is recovered by a 1:1 join back to ``frame`` on the
    natural key — every row of the frame is one player-week, so the join cannot fan out.
    """
    cold = frame.loc[is_cold_start(frame), ["player_id", "season", "week"]].copy()
    cold["_cold"] = True
    merged = predictions.merge(cold, on=["player_id", "season", "week"], how="left")
    flag = merged["_cold"].fillna(False).to_numpy()
    return per_position_metrics(predictions[flag], positions=positions)


def deferred_positions(
    ridge_cold: Mapping[str, PositionMetrics],
    baselines_cold: Mapping[str, Mapping[str, PositionMetrics]],
    *,
    positions: Sequence[str] = SKILL_POSITIONS,
) -> tuple[str, ...]:
    """Positions where pure ridge does **not** clear the cold-start bar — the measured deferral set.

    Pure. A position is deferred unless ridge beats **every** baseline on **both** MAE (lower) and
    within-slate ρ (higher) at the cold start — the same all-three-both-metrics bar the all-weeks grade
    uses. So ridge is fielded at a cold start only where it was measured to win it (spec ticket #29).
    """
    out: list[str] = []
    for pos in positions:
        r = ridge_cold.get(pos)
        maes = [b[pos].mae for b in baselines_cold.values() if pos in b]
        rhos = [b[pos].spearman for b in baselines_cold.values() if pos in b and b[pos].spearman is not None]
        best_mae = min(maes) if maes else None
        best_rho = max(rhos) if rhos else None
        wins = (
            r is not None
            and r.spearman is not None
            and best_mae is not None
            and best_rho is not None
            and r.mae < best_mae
            and r.spearman > best_rho
        )
        if not wins:
            out.append(pos)
    return tuple(out)


def cold_start_deferral(
    frame: pd.DataFrame,
    *,
    positions: Sequence[str] = SKILL_POSITIONS,
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
) -> tuple[str, ...]:
    """Measure the per-position cold-start gate by walk-forward: which positions ridge loses the cold start.

    Runs pure ridge and the three baselines through the harness, slices each to the cold-start cohort,
    and returns :func:`deferred_positions`. This is the same measurement ``scripts/eval_weekly.py``
    reports; a fit uses it so the committed artifact's gate is the measured one, not a hand-set constant.
    """
    ridge = evaluate(WeeklyRidge(positions=positions), frame, positions=positions, test_seasons=test_seasons)
    ridge_cold = cold_start_metrics(ridge.predictions, frame, positions=positions)
    base_cold = {
        name: cold_start_metrics(
            evaluate(factory(), frame, positions=positions, test_seasons=test_seasons).predictions,
            frame,
            positions=positions,
        )
        for name, factory in _BASELINE_FACTORIES.items()
    }
    return deferred_positions(ridge_cold, base_cold, positions=positions)

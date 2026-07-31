"""The draft-value model: prior-season aggregates → the next season's custom total (Phase 9, #31).

The draft path's deliverable, and a genuinely separate model from the weekly one (spec finding #2 /
Decision #4). It regresses :data:`model.frame.SEASON_FEATURES` — all provably ≤ S-1 — onto
:data:`model.frame.LABEL_COL`, the real re-scored season total for S.

What "beats the baseline" means here
------------------------------------
Draft value is a **ranking** problem: the only thing a draft board consumes is the order of players
within a position (a VOR/tier surface is a within-position ordering, then a cross-position replacement
comparison). MAE on a season total is nearly irrelevant to whether the fourth-round pick was right. So
the model is graded, and must beat :class:`PriorSeasonTotal`, on **within-``(season, position)``
Spearman ρ** — the season analogue of the weekly harness's within-slate ρ (:mod:`model.evaluate`).

The model, and where it defers
------------------------------
:class:`SeasonModel` is a per-position **ridge** regression on standardised features. The spec caps this
phase at gradient boosting and regularised linear models (no deep learning), and season grain is the
data-thinnest surface in the project. A regularised linear model is the right tool at that size, and it
carries no new dependency: the fit is a closed-form ridge solve in numpy, fully deterministic, so it
refits reproducibly from the frame with no artifact to hand-edit.

The model's entire edge is **usage** — snap, target and rush share, expected points — the signal a
generic points projection throws away. Of the six draftable positions only **DEF** lacks it entirely:
measured on the real lake, DEF carries no usage column at all (profile #27 §5b), so a ridge there is a
shrunk "last year's total" and, measured, orders *worse* than the baseline it shrinks (an ungated ridge
scores DEF ρ 0.265 vs the bar's 0.296 over 2018-2025). K is *not* in that boat — kickers are on the
field, so their snap and target share are populated — so the model fields a ridge for it too. The rule
is therefore mechanical: **field a ridge wherever a usage signal exists (QB/RB/WR/TE/K), and defer to an
embedded :class:`PriorSeasonTotal` where none does (DEF)** — the emitted board never ranks a position
with the weaker orderer. Measured, the fielded model beats the bar on ρ at all five modelled positions
and ties DEF; DEF is #30's job (a component model through the scoring engine), and the committed bar in
``docs/model-draft-baseline.md`` records the whole table.

Other decisions
---------------
* **Per position**, never pooled — a QB season total is several times a kicker's, and a pooled fit would
  be dominated by that scale (Decision #5), exactly as the metrics are.
* **The warm-up season does not train the model.** The earliest built season (``is_warmup``) has no
  in-window prior, so every player there reads ``is_rookie`` by arithmetic rather than observation — a
  cohort that is ~80% veterans mislabelled. Training on it would learn the rookie level from the wrong
  population, so :meth:`SeasonModel.fit` drops those rows (:attr:`SeasonModel.fit_seasons` records what
  actually trained). Genuine rookies in later seasons — correctly flagged — still teach it.
* **Rookies fall out for free.** Standardising centres every prior-season feature on its training mean,
  so a real rookie — all prior features null, imputed to that mean — lands at the standardised origin and
  is predicted at the position's learned level, shifted by the ``is_rookie`` indicator. No separate code
  path, the same shape as :class:`model.baselines.PriorSeasonRank`'s fallback (spec acceptance #4).
* **Points, not stats.** The season label is *already* engine-scored (it is a sum of the weekly
  ``y_custom_points``), so regressing it directly hand-codes no scoring — the immutable rule holds. The
  "predict components, let the engine score" rule (Decision #2) is about K and DST, whose points are a
  discontinuous function of stat buckets; it is #30's concern, not the season draft value model's.

Feeding the existing draft machinery
------------------------------------
:func:`season_value_board` emits the model's ranking as :class:`projections.board.PlayerRow` objects —
the exact shape ``projections.board.build_board`` already produces — so ``draft.vor``'s replacement
levels, VOR and tiers are reused unchanged (spec acceptance #3) rather than reimplemented. No consumer
is wired to it here; making the model a *selectable* projection source is ticket #34.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.evaluate import (
    DEFAULT_TEST_SEASONS,
    FANTASY_POSITIONS,
    spearman,
    walk_forward_splits,
)
from model.frame import LABEL_COL, SEASON_FEATURES
from projections.board import DRAFTABLE, PlayerRow

_LOG = logging.getLogger(__name__)

#: The columns the model reads: the ≤ S-1 features plus two cohort indicators. ``changed_team_prior``
#: is already inside :data:`SEASON_FEATURES`; the indicators let the fit learn a rookie / returning
#: level explicitly rather than leaning on imputation alone.
MODEL_FEATURES: tuple[str, ...] = (*SEASON_FEATURES, "is_rookie", "has_prior_season")

#: The usage features — the model's whole edge over "last year's total". A position with a value in
#: none of these (DEF alone, on the real lake) has no edge to field and is deferred to the baseline.
_USAGE_FEATURES: tuple[str, ...] = (
    "prior_snap_share",
    "prior_target_share",
    "prior_rush_share",
    "prior_exp_points",
)

#: Ridge penalty on the standardised features. Modest by design — the features are few and the row
#: counts are small, so the penalty is there to stabilise a near-collinear prior-usage block, not to
#: shrink the fit to the mean.
_RIDGE_ALPHA = 1.0

#: A column whose training standard deviation is below this is treated as constant and contributes
#: nothing (K/DEF carry no snap/target/rush/exp usage, so those columns are all-null there).
_STD_EPS = 1e-9


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The model features as a float ``(n, k)`` array, nulls preserved as ``NaN`` for imputation."""
    cols = []
    for name in MODEL_FEATURES:
        if name in frame.columns:
            col = frame[name]
            if col.dtype == bool or col.dtype == "boolean":
                col = col.astype("float64")
            cols.append(pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64"))
        else:
            cols.append(np.full(len(frame), np.nan))
    return np.column_stack(cols) if cols else np.empty((len(frame), 0))


def _has_usage_signal(frame: pd.DataFrame) -> bool:
    """Does this (single-position) frame carry any usage feature at all? On the real lake DEF never does."""
    return any(
        col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().any()
        for col in _USAGE_FEATURES
    )


def _safe_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and standard deviations over the non-null entries; all-null columns → (0, 0).

    ``nanmean`` / ``nanstd`` over an all-null column raise a *warnings-module* ``RuntimeWarning``
    ("Mean of empty slice", "Degrees of freedom <= 0"), which ``np.errstate`` does **not** govern — it
    covers floating-point errors, a different mechanism. ``catch_warnings`` is the one that silences
    them, so a modelled position with an incidentally empty column does not breach the zero-warnings bar.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 0.0)
    return mean, std


@dataclass
class _PositionFit:
    """A fitted per-position ridge: standardisation, weights and the label mean as ultimate fallback."""

    mean: np.ndarray
    std_safe: np.ndarray
    weights: np.ndarray
    intercept: float
    y_mean: float

    def _standardise(self, matrix: np.ndarray) -> np.ndarray:
        imputed = np.where(np.isnan(matrix), self.mean, matrix)
        return (imputed - self.mean) / self.std_safe

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] == 0:
            return np.empty(0)
        return self.intercept + self._standardise(matrix) @ self.weights


def _fit_ridge(matrix: np.ndarray, y: np.ndarray, alpha: float) -> _PositionFit:
    """Closed-form ridge on standardised features, intercept unpenalised. Deterministic."""
    mean, std = _safe_stats(matrix)
    std_safe = np.where(std < _STD_EPS, 1.0, std)
    imputed = np.where(np.isnan(matrix), mean, matrix)
    z = (imputed - mean) / std_safe
    z[:, std < _STD_EPS] = 0.0  # a constant feature contributes nothing, regardless of its scale

    design = np.column_stack([np.ones(len(z)), z])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0  # never penalise the intercept
    gram = design.T @ design + penalty
    target = design.T @ y
    try:
        coef = np.linalg.solve(gram, target)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(gram, target, rcond=None)
    return _PositionFit(
        mean=mean,
        std_safe=std_safe,
        weights=coef[1:],
        intercept=float(coef[0]),
        y_mean=float(np.mean(y)) if len(y) else 0.0,
    )


class PriorSeasonTotal:
    """The bar: predict season S's total as season S-1's total, position-mean for a rookie.

    The season analogue of :class:`model.baselines.PriorSeasonRank` and the honest thing to beat — "next
    year looks like last year" is the whole of most manual draft reasoning. A player with no prior
    season (a rookie, ``prior_points_total`` null) falls back to the learned per-position mean, so it
    covers every row and is graded on the same universe as the model. It is also the fallback
    :class:`SeasonModel` defers to for positions it cannot beat.
    """

    def __init__(self) -> None:
        self._position_mean: dict[str, float] = {}
        self._global_mean: float = 0.0

    def fit(self, frame: pd.DataFrame) -> PriorSeasonTotal:
        y = pd.to_numeric(frame[LABEL_COL], errors="coerce")
        self._global_mean = float(y.mean()) if bool(y.notna().any()) else 0.0
        means = (
            pd.DataFrame({"position": frame["position"].astype("string"), "y": y})
            .dropna(subset=["y"])
            .groupby("position", observed=True)["y"]
            .mean()
        )
        self._position_mean = {str(p): float(v) for p, v in means.items()}
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        prior = pd.to_numeric(frame["prior_points_total"], errors="coerce")
        fallback = (
            frame["position"].astype("string").map(self._position_mean).astype("float64")
        )
        fallback = fallback.where(fallback.notna(), self._global_mean)
        return prior.where(prior.notna(), fallback).reindex(frame.index).astype("float64")


class SeasonModel:
    """Per-position ridge on prior-season aggregates, deferring to the baseline where it has no edge.

    ``fit`` learns one standardised ridge per position **that carries a usage signal**, on the training
    frame with the warm-up season excluded; a position with none (DEF, on the real lake) is served by an
    embedded :class:`PriorSeasonTotal`. ``predict`` returns a Series index-aligned to its input, so a
    misaligned prediction surfaces as nulls rather than a silent scramble (the same contract the weekly
    harness enforces).

    ``require_usage=False`` fields a ridge for every position regardless — the *ungated* variant, kept
    so the committed report can show, with its own numbers, that ridge orders K/DEF worse than the
    baseline and thereby justify the default fallback rather than asserting it.
    """

    def __init__(self, alpha: float = _RIDGE_ALPHA, *, require_usage: bool = True) -> None:
        self.alpha = float(alpha)
        self.require_usage = bool(require_usage)
        self._baseline = PriorSeasonTotal()
        self._fits: dict[str, _PositionFit] = {}
        self.modeled_positions: tuple[str, ...] = ()
        self.fit_seasons: tuple[int, ...] = ()

    def fit(self, frame: pd.DataFrame) -> SeasonModel:
        train = frame
        if "is_warmup" in train.columns:
            train = train[~train["is_warmup"].fillna(False).astype(bool)]
        self.fit_seasons = tuple(
            sorted({int(s) for s in pd.to_numeric(train["season"], errors="coerce").dropna().unique()})
        )
        self._baseline.fit(train)  # covers every position, and every position the ridge declines

        self._fits = {}
        modeled: list[str] = []
        pos = train["position"].astype("string")
        for position in sorted(p for p in pos.dropna().unique()):
            sub = train[pos == position]
            if self.require_usage and not _has_usage_signal(sub):
                continue  # no usage edge → defer to last-year's-total (DEF has no usage column at all)
            y = pd.to_numeric(sub[LABEL_COL], errors="coerce").to_numpy(dtype="float64")
            keep = ~np.isnan(y)
            if int(keep.sum()) == 0:
                continue
            self._fits[str(position)] = _fit_ridge(_feature_matrix(sub)[keep], y[keep], self.alpha)
            modeled.append(str(position))
        self.modeled_positions = tuple(modeled)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        out = self._baseline.predict(frame).astype("float64")  # baseline everywhere...
        pos = frame["position"].astype("string")
        for position, fit in self._fits.items():  # ...overridden by ridge where the model is fielded
            mask = (pos == position).to_numpy()
            if not mask.any():
                continue
            out.iloc[np.where(mask)[0]] = fit.predict(_feature_matrix(frame[mask]))
        return out


# --------------------------------------------------------------------------- evaluation
@dataclass(frozen=True)
class SeasonPositionMetrics:
    """Held-out accuracy and within-position ordering for one position."""

    position: str
    n: int
    mae: float
    rmse: float
    spearman: float | None  # mean within-(season, position) rho over the test seasons that admit one
    slates: int  # test seasons that contributed a rho


@dataclass(frozen=True)
class SeasonEvalResult:
    """A predictor's walk-forward season grade. Per position by construction — no pooled number."""

    predictor: str
    test_seasons: tuple[int, ...]
    per_position: dict[str, SeasonPositionMetrics]
    predictions: pd.DataFrame


def _season_metrics(
    predictions: pd.DataFrame, *, positions: Sequence[str]
) -> dict[str, SeasonPositionMetrics]:
    """MAE and within-``(season, position)`` Spearman, per position — the VOR-relevant ordering."""
    out: dict[str, SeasonPositionMetrics] = {}
    for pos in positions:
        valid = predictions[
            (predictions["position"] == pos)
            & predictions["pred"].notna()
            & predictions["actual"].notna()
        ]
        if valid.empty:
            continue
        err = (valid["pred"] - valid["actual"]).to_numpy(dtype=float)
        rhos = [
            rho
            for _season, grp in valid.groupby("season", sort=True)
            if (rho := spearman(grp["pred"], grp["actual"])) is not None
        ]
        out[pos] = SeasonPositionMetrics(
            position=pos,
            n=int(len(valid)),
            mae=float(np.mean(np.abs(err))),
            rmse=float(np.sqrt(np.mean(err**2))),
            spearman=float(np.mean(rhos)) if rhos else None,
            slates=len(rhos),
        )
    return out


def evaluate_season(
    predictor: SeasonModel | PriorSeasonTotal,
    frame: pd.DataFrame,
    *,
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
    positions: Sequence[str] = FANTASY_POSITIONS,
    name: str | None = None,
) -> SeasonEvalResult:
    """Walk-forward by season, scored per position on MAE and within-position ρ.

    Reuses :func:`model.evaluate.walk_forward_splits`, so the leak gate applies unchanged: a split whose
    training rows reach into the test season refuses to exist. The frame is scoped to the fantasy
    ``positions`` first, both sides of every split, so a fit learns its fallback from the cohort it is
    graded on.
    """
    cohort = frame[frame["position"].isin(positions)]
    seasons_used: list[int] = []
    chunks: list[pd.DataFrame] = []
    for split in walk_forward_splits(cohort, test_seasons=test_seasons):
        model = predictor.fit(split.train)
        preds = pd.to_numeric(model.predict(split.test).reindex(split.test.index), errors="coerce")
        chunks.append(
            pd.DataFrame(
                {
                    "player_id": split.test["player_id"].to_numpy(),
                    "season": split.test["season"].to_numpy(),
                    "position": split.test["position"].to_numpy(),
                    "actual": pd.to_numeric(split.test[LABEL_COL], errors="coerce").to_numpy(),
                    "pred": preds.to_numpy(),
                }
            )
        )
        seasons_used.append(split.test_season)
    predictions = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=["player_id", "season", "position", "actual", "pred"])
    )
    return SeasonEvalResult(
        predictor=name or type(predictor).__name__,
        test_seasons=tuple(seasons_used),
        per_position=_season_metrics(predictions, positions=positions),
        predictions=predictions,
    )


# --------------------------------------------------------------------------- draft board adapter
def _team_of(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def to_player_rows(
    frame: pd.DataFrame,
    preds: pd.Series,
    *,
    names: dict[str, str] | None = None,
    positions: Sequence[str] = DRAFTABLE,
) -> list[PlayerRow]:
    """Turn model predictions into :class:`projections.board.PlayerRow` objects for ``draft.vor``.

    ``proj_pts`` is the predicted season total; ``vor`` / ``tier`` are left at their defaults for
    ``draft.vor`` to fill, exactly as ``build_board`` leaves them. ``adp`` defaults to ``+inf``
    (undrafted) because the season frame carries no market ADP — VOR and tiers never read it. Rows with
    a null prediction or a non-draftable position are dropped.
    """
    keep = set(positions)
    names = names or {}
    aligned = pd.to_numeric(preds.reindex(frame.index), errors="coerce")
    rows: list[PlayerRow] = []
    for idx, row in frame.iterrows():
        pos = row.get("position")
        pred = aligned.get(idx)
        if pos not in keep or pred is None or pd.isna(pred):
            continue
        pid = str(row["player_id"])
        rows.append(
            PlayerRow(
                player_id=pid,
                name=names.get(pid, pid),
                pos=str(pos),
                team=_team_of(row.get("team")),
                proj_pts=round(float(pred), 2),
                adp=float("inf"),
            )
        )
    return rows


def season_value_board(
    model: SeasonModel,
    frame: pd.DataFrame,
    season: int,
    *,
    names: dict[str, str] | None = None,
    positions: Sequence[str] = DRAFTABLE,
) -> list[PlayerRow]:
    """The model's draft board for one season: predict, adapt to ``PlayerRow``, rank best-first.

    ``model`` must already be fit on strictly-earlier seasons. The board is the same object
    ``build_board`` returns, so ``draft.vor.replacement_levels`` / ``add_vor`` / ``tierize`` consume it
    with no change. K and DEF appear ranked by the deferred baseline, never by the weaker ridge.
    """
    target = frame[pd.to_numeric(frame["season"], errors="coerce") == int(season)]
    rows = to_player_rows(target, model.predict(target), names=names, positions=positions)
    rows.sort(key=lambda p: p.proj_pts, reverse=True)
    return rows

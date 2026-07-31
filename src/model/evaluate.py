"""Walk-forward evaluation for the modelling phase — the bar every model is graded against.

This is Phase 9's analogue of Phase 1's "validate the scoring engine before building on it" and
Phase 8's lookahead gate: it defines *good* before anything is fit. With ``baseline_sleeper_points``
null for every 2016-2025 row (the market baseline is forward-only from 2026 W1), there is no
historical projection to beat, so the definition of "better" has to be one we set ourselves and hold
fixed. Everything here exists to make a model's claim of improvement checkable against a number that
was written down first (``docs/plans/modeling.md``, Decision #1).

The three rules this module enforces, each a place a leak or an illusion hides
-----------------------------------------------------------------------------
* **Walk-forward by season, and nothing else.** Train on every season strictly before ``S``, test on
  ``S`` (:func:`walk_forward_splits`). A random k-fold over player-weeks would put a player's week 3
  in train and his week 4 in test and score a model on its own memory; the split axis is the season
  and there is deliberately no other entry point, so that fold cannot be built by accident. A split
  that reaches into its own test season is rejected by :class:`Split` on construction — the same
  fail-closed shape as ``dataset.assemble``'s gate (Decision #6).
* **Per position, never pooled.** Metrics are returned as a ``dict`` keyed by position
  (:class:`EvalResult.per_position`); there is no pooled attribute, so a QB-only gain cannot
  masquerade as a general one. A pooled MAE is dominated by the fact that QBs score several times
  what kickers do (Decision #5).
* **Spearman within (position, week).** The *ordering* a start/sit or waiver decision consumes is the
  ordering among the players available in one position in one week — not across weeks, where the
  week-to-week scoring level would inflate the correlation. :func:`per_position_metrics` computes one
  rho per slate and averages them. Because the walk-forward pool spans many test seasons, a *slate* is
  one ``(season, week)`` — a bare week number would fold 2019 week 5 into 2022 week 5, two decisions
  that were never on the same board, and let a per-season fallback level read as a fake ordering.
  **A predictor that emits one constant across a slate scores rho = 0 on it, not "undefined".** It
  offered no ordering, and that is a result. Excusing those slates instead would average a predictor
  over only the boards it happened to say something on, so two predictors' rho would be means over
  different universes and the larger number could be the one that answered less often
  (:attr:`PositionMetrics.spearman_ordered_slates` reports how often it actually did answer).

Scoring note: the baselines predict points directly, which is legitimate *for baselines* — they are
the naive bar, not a model. The Decision #2 rule ("models predict stats, the engine scores them")
governs the real models (#29/#30). This harness only reads the already-scored label
``y_custom_points`` the assembler produced from the league's live ``scoring_settings``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

#: The positions any model in this phase predicts. ``nflverse_player_week`` carries a stat line for
#: every defender and special-teamer who took the field, so the raw frame is majority non-fantasy
#: rows scoring ~0 under this scoring; the harness scopes to this cohort (profile ticket #27, §2).
FANTASY_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")

#: Walk-forward test seasons. 2016-2017 are consumed as lag/EWMA warm-up (Decision #6) and are never
#: scored on; 2018 is the earliest test season because its train set (2016+2017) is the first that
#: carries a real prior season for the ``PriorSeasonRank`` baseline to stand on.
DEFAULT_TEST_SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)

#: The re-scored label column the assembler produces (``dataset.assemble``).
LABEL_COL = "y_custom_points"

#: Number of predicted-value buckets in the calibration table.
CALIBRATION_BINS = 10


@runtime_checkable
class Predictor(Protocol):
    """Fit on a training frame, predict a points value per row of a frame.

    A baseline and a gradient-booster implement the identical interface, so the harness scores both
    through one path (``docs/plans/modeling.md``). ``predict`` must return a Series **index-aligned to
    its input frame** — the harness reindexes to the frame's index and a misaligned Series would come
    back as all-null.
    """

    def fit(self, frame: pd.DataFrame) -> Predictor:
        ...

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        ...


class LeakError(AssertionError):
    """A walk-forward split whose training rows reach into (or past) the test season.

    Subclasses :class:`AssertionError` because it is the modelling phase's fail-closed gate — the
    analogue of Phase 8's lookahead guard. A leak is invisible in the output and a rejected split is
    not, so the split refuses to exist rather than quietly scoring a model on its own future.
    """


def assert_walk_forward(train: pd.DataFrame, test_season: int) -> None:
    """Raise :class:`LeakError` unless every training row is strictly earlier than ``test_season``.

    Fails closed on a season it cannot read, too. ``pd.to_numeric(..., errors="coerce")`` turns an
    unparseable season into ``NaN``, and a ``NaN`` compares false against every threshold — so a row
    the gate cannot parse is a row the gate cannot clear, and silently passing it would be the one
    outcome a fail-closed guard must not have. ``walk_forward_splits`` never produces such a row (it
    filters on the same coercion), so this can only fire on a hand-built :class:`Split`.
    """
    if "season" not in train.columns:
        raise LeakError("train frame has no 'season' column, so a leak cannot even be checked")
    if train.empty:
        return
    seasons = pd.to_numeric(train["season"], errors="coerce")
    if bool(seasons.isna().any()):
        unreadable = sorted({str(v) for v in train.loc[seasons.isna(), "season"].unique()})[:5]
        raise LeakError(
            f"train frame has {int(seasons.isna().sum())} row(s) whose 'season' does not parse as a "
            f"number (e.g. {unreadable}); a season the gate cannot read is a leak it cannot rule out"
        )
    leaked = sorted({int(s) for s in seasons.dropna().unique() if int(s) >= test_season})
    if leaked:
        raise LeakError(
            f"walk-forward split for test season {test_season} has training rows from season(s) "
            f"{leaked} — training must be strictly earlier than the test season. Scoring a model on "
            "the season it trained on measures memory, not skill."
        )


@dataclass(frozen=True)
class Split:
    """One strictly-forward train/test partition. Validates itself on construction (fail-closed)."""

    test_season: int
    train: pd.DataFrame
    test: pd.DataFrame

    def __post_init__(self) -> None:
        assert_walk_forward(self.train, self.test_season)
        if not self.test.empty:
            stray = sorted(
                {
                    int(s)
                    for s in pd.to_numeric(self.test["season"], errors="coerce").dropna().unique()
                    if int(s) != self.test_season
                }
            )
            if stray:
                raise LeakError(
                    f"test partition for season {self.test_season} also carries season(s) {stray}; "
                    "a split's test rows must all belong to its one test season"
                )


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
) -> Iterator[Split]:
    """Yield one validated :class:`Split` per test season, train = every strictly-earlier season.

    The split axis is the season and there is no other. A random or player-stratified fold is not an
    option this module offers, precisely because it would mix a player's own adjacent weeks across the
    train/test line — the leak Decision #6 exists to forbid.
    """
    for season in sorted({int(s) for s in test_seasons}):
        train = frame[pd.to_numeric(frame["season"], errors="coerce") < season]
        test = frame[pd.to_numeric(frame["season"], errors="coerce") == season]
        if test.empty:
            _LOG.warning("no rows for test season %d — skipping it", season)
            continue
        if train.empty:
            _LOG.warning(
                "no training rows before test season %d — skipping (need >= 1 prior season)", season
            )
            continue
        yield Split(test_season=season, train=train.copy(), test=test.copy())


# --------------------------------------------------------------------------- metrics
def spearman(pred: pd.Series, actual: pd.Series) -> float | None:
    """Spearman rank correlation; ``0.0`` for a no-ordering prediction, ``None`` where undefined.

    Computed as the Pearson correlation of average ranks (``np.corrcoef``) rather than via
    ``Series.corr(method="spearman")``, which pulls in scipy — a dependency this project does not
    carry. Average-rank ties are the textbook Spearman convention.

    The two degenerate cases are deliberately *not* symmetric:

    * A **constant prediction** scores ``0.0``. The predictor was asked to order the board and
      declined; a no-information ordering has expected rank correlation zero, and scoring it as one
      keeps every predictor's rho a mean over the same slates. Returning ``None`` here would let a
      predictor that answers on 7 boards out of 141 post a rho next to one that answered on all 141.
    * A **constant actual** returns ``None`` — every player really did score the same, so no
      prediction could have ordered them. That is the board's fault, not the predictor's, and
      charging a zero for it would penalise whichever model happened to draw that week.
    """
    mask = pred.notna() & actual.notna()
    if int(mask.sum()) < 2:
        return None
    p = pred[mask]
    a = actual[mask]
    if a.nunique() < 2:
        return None
    if p.nunique() < 2:
        return 0.0
    rho = float(np.corrcoef(p.rank().to_numpy(), a.rank().to_numpy())[0, 1])
    return None if np.isnan(rho) else rho


def calibration_by_decile(
    pred: pd.Series, actual: pd.Series, *, bins: int = CALIBRATION_BINS
) -> pd.DataFrame:
    """Predicted vs. realized mean, per predicted-value decile.

    Rows are split into equal-count buckets by predicted value (ranking first so the pervasive ties
    of a position-mean fallback do not collapse the buckets). Columns: ``decile`` (1-based), ``n``,
    ``pred_mean``, ``realized_mean``. A model with good MAE but a realized mean that does not climb
    with the predicted decile is mis-ranking the boom/bust players the FLEX decision turns on.
    """
    mask = pred.notna() & actual.notna()
    p = pred[mask].reset_index(drop=True)
    a = actual[mask].reset_index(drop=True)
    empty = pd.DataFrame(columns=["decile", "n", "pred_mean", "realized_mean"])
    if p.empty:
        return empty
    codes = (
        pd.qcut(p.rank(method="first"), bins, labels=False)
        if len(p) >= bins
        else pd.Series(np.zeros(len(p), dtype=int), index=p.index)
    )
    out = (
        pd.DataFrame({"decile": codes.astype(int) + 1, "pred": p, "actual": a})
        .groupby("decile", sort=True)
        .agg(n=("pred", "size"), pred_mean=("pred", "mean"), realized_mean=("actual", "mean"))
        .reset_index()
    )
    return out


@dataclass(frozen=True)
class PositionMetrics:
    """Held-out accuracy and ordering for one position, plus its calibration table."""

    position: str
    n: int
    mae: float
    rmse: float
    spearman: float | None  # mean within-(season, week) rho, over the slates that admit one
    spearman_slates: int  # (season, week) slates that contributed a rho (a flat one contributes 0)
    spearman_ordered_slates: int  # of those, how many the predictor gave a non-constant ordering on
    calibration: pd.DataFrame


@dataclass(frozen=True)
class EvalResult:
    """A predictor's walk-forward grade. Per position by construction — there is no pooled number."""

    predictor: str
    test_seasons: tuple[int, ...]
    per_position: dict[str, PositionMetrics]
    predictions: pd.DataFrame  # out-of-sample: player_id, season, week, position, actual, pred


def per_position_metrics(
    predictions: pd.DataFrame,
    *,
    positions: Sequence[str] = FANTASY_POSITIONS,
) -> dict[str, PositionMetrics]:
    """MAE, RMSE, within-week Spearman and calibration, computed per position.

    ``predictions`` carries at least ``position``, ``season``, ``week``, ``actual`` and ``pred``. The
    return is a ``dict`` keyed by position on purpose: a pooled summary is not something a caller can
    ask this function for, because a pooled metric is dominated by cross-position scoring scale
    (Decision #5). Spearman is computed per ``(season, week)`` slate and averaged — never across
    slates (see the module docstring for why a bare week number is the wrong grain here). A slate the
    predictor answered with one constant counts as a scored rho of 0 and is *not* counted in
    ``spearman_ordered_slates``, so the mean and the number of boards it was actually earned on are
    both visible.
    """
    out: dict[str, PositionMetrics] = {}
    for pos in positions:
        sub = predictions[predictions["position"] == pos]
        valid = sub[sub["pred"].notna() & sub["actual"].notna()]
        if valid.empty:
            continue
        err = (valid["pred"] - valid["actual"]).to_numpy(dtype=float)
        rhos: list[float] = []
        ordered = 0
        for _slate, grp in valid.groupby(["season", "week"], sort=True):
            rho = spearman(grp["pred"], grp["actual"])
            if rho is None:  # the board itself admits no ordering — see spearman()
                continue
            rhos.append(rho)
            ordered += int(grp["pred"].nunique() > 1)
        out[pos] = PositionMetrics(
            position=pos,
            n=int(len(valid)),
            mae=float(np.mean(np.abs(err))),
            rmse=float(np.sqrt(np.mean(err**2))),
            spearman=float(np.mean(rhos)) if rhos else None,
            spearman_slates=len(rhos),
            spearman_ordered_slates=ordered,
            calibration=calibration_by_decile(valid["pred"], valid["actual"]),
        )
    return out


def evaluate(
    predictor: Predictor,
    frame: pd.DataFrame,
    *,
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
    positions: Sequence[str] = FANTASY_POSITIONS,
    name: str | None = None,
) -> EvalResult:
    """Walk-forward fit→predict over ``test_seasons``, scored per position on pooled out-of-sample rows.

    The frame is scoped to the fantasy ``positions`` first (both sides of every split), so a baseline
    learns its fallback from the cohort it is graded on and the IDP tail never enters the metric.
    Each split fits on strictly-earlier seasons and predicts the one test season; predictions are
    reindexed to the test frame, so a predictor that returns a misaligned Series is caught as nulls
    rather than silently scrambled.
    """
    cohort = frame[frame["position"].isin(positions)]
    seasons_used: list[int] = []
    chunks: list[pd.DataFrame] = []
    for split in walk_forward_splits(cohort, test_seasons=test_seasons):
        model = predictor.fit(split.train)
        preds = model.predict(split.test)
        if not isinstance(preds, pd.Series):
            raise TypeError(
                f"{type(predictor).__name__}.predict must return a pandas Series, got "
                f"{type(preds).__name__}"
            )
        preds = pd.to_numeric(preds.reindex(split.test.index), errors="coerce")
        chunks.append(
            pd.DataFrame(
                {
                    "player_id": split.test["player_id"].to_numpy(),
                    "season": split.test["season"].to_numpy(),
                    "week": split.test["week"].to_numpy(),
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
        else pd.DataFrame(columns=["player_id", "season", "week", "position", "actual", "pred"])
    )
    return EvalResult(
        predictor=name or type(predictor).__name__,
        test_seasons=tuple(seasons_used),
        per_position=per_position_metrics(predictions, positions=positions),
        predictions=predictions,
    )

"""The breakout / waiver classifier: which player is *about to matter* (Phase 9, ticket #33).

The one model in this phase whose **label is deliberately in the future**. Every other model here
predicts week *N* from data known before week *N* locked; this one asks, of a player who is currently
a marginal contributor, whether his role and production **step up over the next N weeks** — the waiver
question. That inversion is the whole point, and it is exactly where a leak hides that no reviewer can
see by inspection: the *features* must still be strictly pre-lock while the *label* reaches forward. So
the label is computed by :func:`add_forward_label` alone — never routed through
``dataset.assemble.lookahead_ok`` (that gate governs *features*, and it is left untouched) — and the
asymmetry is pinned by a two-halved test (``tests/test_model_breakout.py``): spiking the forward
*outcomes* must move the label, and spiking the forward *features* must not move the prediction.

Scope: RB / WR / TE
-------------------
The role proxies that exist in the lake — snap share, **target** share (WR/TE), **rush** share (RB) —
are the skill-usage columns, and a breakout *is* a role-and-production step-up in those. QB is out (no
target/rush role trajectory, and in a 1-QB league a "breakout QB" is not the waiver decision this
serves); K/DEF are out (streamed weekly on matchup, no role trajectory — #30 owns them). Metrics are
reported **per position, never pooled**: the base rate differs 3.5x across positions (below), so a
pooled precision@k would be dominated by whichever position has the most candidates (WR), and three
positions get three verdicts.

The cohort — declared from waiver relevance, before any measurement
------------------------------------------------------------------
A model that ranks *all* player-weeks puts the already-rostered stars at the top: they "break out" by
any absolute bar, precision@k reads beautifully, and the ranking is worthless because you cannot claim
a player who is already producing. So the cohort is fixed first, structurally: this league rosters 168
of ~600 active players (14x12), and an entrenched starter plays a **majority of his team's offensive
snaps**. The waiver-relevant cohort is therefore RB/WR/TE player-weeks that are **not already a
starter entering the decision week** — trailing snap share ``snap_pct_ewma <= 0.5``, *or* no snap
history yet (:func:`snap_cohort_mask`). The null arm is ~10% of rows (RB 10.5 / WR 9.9 / TE 10.9%) —
an emerging player who has not established a role — and is reported, not swept in.

A **second, production-axis cohort** (:func:`production_cohort_mask` — trailing ``points_ewma`` below
the position's startable line) is measured as a robustness check, exactly as #32 cross-checks its
cohort. If the two disagree on the verdict, the report says so rather than letting one number stand.

There is **no ownership / rostered signal in the lake** (no roster source in ``collect.registry``), so
"rostered-or-free" cannot be a training filter — the model ranks player-weeks in the declared cohort,
and the free-agent filter applies at **serve** time (#34), never here. No proxy is invented for it.

The label — forward production crossing the startable line
---------------------------------------------------------
* **N = 3** (:data:`FORWARD_WINDOW`). Waivers clear Wednesday 09:00 CEST and a claim is held for weeks
  while spending a **single ordered reverse-standings claim** (not FAAB) — so a 1-week window is noise
  (one spike) and a 6-week one is not actionable and collides with byes and the Week 15-17 playoffs.
  N=3 is long enough that a single game cannot mint a breakout, short enough to mean "about to matter
  *now*". If N shrinks it becomes a one-week bet; if it grows, season-end truncation bites harder and
  the target dilutes toward "matters eventually".
* **Production, not forward role.** "Stepped up" is a threshold on forward *points*, not forward snap
  share, because forward snap share reintroduces the ``nflverse_snaps`` join gaps on the **label** side
  — the one place they cannot be absorbed — while engine-scored points are present for every played
  game. Role is the *input* (the trajectories); production is the *outcome* a claim is for.
* **Per played game, so a bye is not a zero.** Forward PPG is the mean custom points over the games the
  player actually **played** in weeks ``w+1..w+N`` (:func:`add_forward_label`). A bye or DNP contributes
  no game to the mean rather than a zero — the silent label-corruption path, closed.
* **A full N-week window, or no label (season-end truncation).** A row is a *decision row* only where a
  full ``N``-week window exists ahead of it in the season (``week <= W_last - N``, ``W_last`` the last
  played week that season). So the label is always over the same N-week horizon — a ragged 2-week window
  at the season edge would quietly change what "stepped up" means. The Week 15-17 playoff weeks are thus
  never decision weeks (you do not run this to pick up someone for after the season) but are legitimate
  *forward-window* weeks for an earlier decision.
* **Evaluable requires >= 2 played games** (:data:`MIN_FORWARD_GAMES`) *within* that full window: a
  single cameo can neither mint nor hide a breakout. This rule drops ~19% of decision rows
  and it drops **genuine negatives too** — a player who played 1 of 3 because he was *benched* is the
  clearest negative and is removed alongside the injured. That thinning is characterised in the report
  (dropped count split by whether the decision row carried an injury-report status), not left invisible.
* **``y_breakout = 1`` iff forward PPG >= ``T_pos``**, the position's weekly-startable line
  (:func:`startable_thresholds`): the custom-points level of the ``S_pos``-th weekly scorer (RB 30 / WR
  30 / TE 14 = 12 teams x 2 starters + a FLEX split), fit from the **2016-2017 warm-up seasons only** so
  the label is defined identically across every scored season. Measured: RB 8.19 / WR 10.44 / TE 8.09.

The base rate is a **consequence** of that threshold, not a fact, and it is not comparable across
positions: in-cohort it is **RB 0.283 / WR 0.080 / TE 0.098** (all-evaluable RB 0.432 / WR 0.234 / TE
0.218). A sub-50%-snap committee back clears 8.19 half-PPR points routinely (12 carries + 3 catches);
a sub-50%-snap WR3 rarely clears 10.44 — so a breakout is a materially rarer event at WR by
construction. **Raw precision@k is therefore reported next to lift over the base rate** (:func:`lift`),
because lift is comparable across positions and raw precision is not — without it RB reads as the
model's strongest position when it is only the *easiest*.

What this can and cannot claim (the selection effect)
-----------------------------------------------------
The training frame carries a row only where the player **played** (the label source is a recorded stat
line — ``dataset.assemble``). So the breakouts this ranks are among players already getting *some*
snaps and about to get more (the committee back taking over, the WR3 about to get targets) — **not** an
inactive player being activated, whose pre-breakout weeks are not in the frame at all. That is the real
waiver population, and stating its bound is the honest version of the claim.

The bar, and the gate
---------------------
Same discipline as #28-#32: three naive baselines the model must beat, ranked within each slate — last
week's points, snap-share trend, and last week's target/rush share (:data:`BASELINE_RANKERS`). The
verdict is **per position at k = 1** (reverse-priority means the real decision is a single claim; k=3/5
are reported for stability). Where the logistic beats every baseline at k=1 it is **fielded**; where it
does not it **defers to the winning baseline** (the gate), exactly as #31 defers DEF and #30 defers a
thin cell. Safe by default: a bare :class:`BreakoutModel` reads the recorded gate from the committed
artifact, and a **missing artifact defers every position** — an unproven logistic is never fielded by
accident. The diagnostic pure-logistic variant is the explicit opt-out ``BreakoutModel(defer={})``.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from model.evaluate import DEFAULT_TEST_SEASONS, walk_forward_splits

_LOG = logging.getLogger(__name__)

#: The positions the breakout classifier ranks — the skill positions with a role trajectory. QB, K and
#: DEF are excluded by design (see the module docstring).
BREAKOUT_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")

#: The forward label window, in weeks after the decision week. Defended in the module docstring.
FORWARD_WINDOW = 3

#: Minimum played games inside the forward window for a row to be *evaluable*. Below this the row has no
#: trustworthy forward signal (a single cameo) and is dropped, which also truncates the season end.
MIN_FORWARD_GAMES = 2

#: Trailing snap share above which a player is treated as an established starter and excluded from the
#: waiver cohort. Structural: an every-down starter plays a majority of his team's offensive snaps.
STARTER_SNAP_EWMA = 0.5

#: League-wide weekly *startable* count per position — 12 teams x (2 RB, 2 WR, 1 TE) + a 12-slot FLEX
#: split (roughly 6 RB / 6 WR / 2 TE). ``T_pos`` is the custom-points level of the ``S_pos``-th weekly
#: scorer; a player who reaches it is producing like someone you would start.
STARTABLE_RANK: dict[str, int] = {"RB": 30, "WR": 30, "TE": 14}

#: The seasons ``T_pos`` is fit from — the lag warm-up, never a scored test season, so the label is
#: defined identically across all of 2018-2025 and cannot read a season it is evaluated on.
WARMUP_SEASONS: tuple[int, ...] = (2016, 2017)

LABEL_COL = "y_breakout"
_POINTS_COL = "y_custom_points"

#: Ridge/L2 penalty on the standardised logistic features. Modest — enough to keep a near-collinear
#: share block conditioned and to forbid separation blow-up, not enough to shrink the fit to the mean.
_LOGIT_ALPHA = 1.0

#: A feature whose training standard deviation is below this is treated as constant and zeroed (RB has
#: no target share, WR/TE no rush share — the irrelevant column contributes nothing rather than noise).
_STD_EPS = 1e-9

#: Existing pre-lock frame columns the classifier reads directly — usage trajectories + light context.
_RAW_FEATURES: tuple[str, ...] = (
    "snap_pct_last",
    "snap_pct_ewma",
    "snap_pct_trend",
    "points_last",
    "points_ewma",
    "points_trend",
    "target_share_last",
    "target_share_ewma",
    "rush_share_last",
    "rush_share_ewma",
    "exp_points_last",
    "exp_points_ewma",
    "games_played_prior",
    "implied_team_total",
    "team_spread_line",
)

#: Trajectory features derived in-module from the ``_last``/``_ewma`` pair, because the frame carries no
#: ``*_trend`` column for the shares (only ``points`` and ``snap_pct`` are trended in
#: ``dataset.assemble._TRENDED``). ``last - ewma`` is a rising-role signal: this week's share above the
#: smoothed recent average. Reading a non-existent ``target_share_trend`` column would silently be null.
_DERIVED_TREND_BASES: tuple[str, ...] = ("target_share", "rush_share")
_DERIVED_FEATURES: tuple[str, ...] = tuple(f"{b}_trend" for b in _DERIVED_TREND_BASES)

#: The full feature vector, in a fixed order (the artifact records it, so a reordering cannot silently
#: reassign a weight to a different feature). ``depth_pos_rank`` is deliberately **absent**: it is 0%
#: populated 2016-2024 and ~37% in 2025 (the only test season it exists in), so it can never be a
#: required feature — a 2025+ refinement at most, and the model is proven without it.
BREAKOUT_FEATURES: tuple[str, ...] = (*_RAW_FEATURES, *_DERIVED_FEATURES)

DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent / "fit" / "breakout.json"


def _num_col(frame: pd.DataFrame, name: str) -> pd.Series:
    """A frame column as a numeric Series, or an all-null Series when the column is absent."""
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


# --------------------------------------------------------------------------- thresholds
def startable_thresholds(
    frame: pd.DataFrame,
    *,
    warmup_seasons: Sequence[int] = WARMUP_SEASONS,
    startable_rank: Mapping[str, int] = STARTABLE_RANK,
    positions: Sequence[str] = BREAKOUT_POSITIONS,
) -> dict[str, float]:
    """``T_pos`` per position: the mean weekly ``S_pos``-th-highest custom score over the warm-up seasons.

    Computed over **all** player-weeks at the position (not the cohort) — the startable line is a
    league-wide fact — and only over ``warmup_seasons`` so it is fixed independently of any scored
    season. A ``(season, week)`` slate with fewer than ``S_pos`` scorers contributes nothing (never an
    issue for a full week at these ranks). Returns ``nan`` for a position with no warm-up data.
    """
    season = pd.to_numeric(frame["season"], errors="coerce")
    warm = frame[season.isin(list(warmup_seasons))]
    pos = warm["position"].astype("string")
    out: dict[str, float] = {}
    for position in positions:
        rank = int(startable_rank[position])
        sub = warm[pos == position]
        levels: list[float] = []
        for _slate, grp in sub.groupby(["season", "week"], observed=True):
            scores = pd.to_numeric(grp[_POINTS_COL], errors="coerce").dropna()
            if len(scores) >= rank:
                levels.append(float(scores.sort_values(ascending=False).iloc[rank - 1]))
        out[position] = float(np.mean(levels)) if levels else float("nan")
    return out


# --------------------------------------------------------------------------- the forward label
def add_forward_label(
    frame: pd.DataFrame,
    thresholds: Mapping[str, float],
    *,
    n: int = FORWARD_WINDOW,
    min_games: int = MIN_FORWARD_GAMES,
) -> pd.DataFrame:
    """Attach the **forward** breakout label — the only place in this phase that looks ahead.

    For each row (a played player-week, the decision week ``w``), gathers the player's games actually
    **played** in weeks ``w+1..w+n`` of the **same season** and computes ``forward_ppg`` over them. A row
    is a *decision row* (``has_forward_window``) only where a full ``n``-week window exists that season
    (``week <= W_last - n``), and ``is_evaluable`` when in addition it has ``>= min_games`` forward played
    games. ``y_breakout`` is 1 when an evaluable row's ``forward_ppg`` reaches ``thresholds[position]``, 0
    when it does not, and ``NA`` otherwise (dropped downstream). The window never crosses a season
    boundary, so no walk-forward split can leak through the label.

    Adds ``forward_games``, ``forward_ppg``, ``has_forward_window``, ``is_evaluable``, ``y_breakout`` and
    leaves ``frame`` otherwise untouched (a copy is returned). This is *not* routed through
    ``lookahead_ok``: that gate is for features, and reaching forward is precisely this function's job.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    work = frame.copy()
    pid = work["player_id"].astype("string")
    season = pd.to_numeric(work["season"], errors="coerce").astype("Int64")
    week = pd.to_numeric(work["week"], errors="coerce").astype("Int64")
    pts = pd.to_numeric(work[_POINTS_COL], errors="coerce")

    plays = pd.DataFrame({"player_id": pid, "season": season, "week": week, "pts": pts}).dropna(
        subset=["player_id", "season", "week"]
    )
    # A game played at week ``wk`` is a forward game for decision weeks ``wk-1 .. wk-n``. Emitting one
    # (decision_week, pts) contribution per offset and grouping is the same predicate as an N-by-N
    # forward join at a fraction of the cost; contributions landing on weeks < 1 simply match no row.
    contribs = []
    for d in range(1, n + 1):
        c = plays[["player_id", "season", "pts"]].copy()
        c["week"] = plays["week"] - d
        contribs.append(c)
    forward = (
        pd.concat(contribs, ignore_index=True)
        .groupby(["player_id", "season", "week"], observed=True)
        .agg(forward_points_sum=("pts", "sum"), forward_games=("pts", "size"))
        .reset_index()
    )

    key = pd.DataFrame(
        {"player_id": pid, "season": season, "week": week, "_row": np.arange(len(work))}
    )
    merged = key.merge(forward, on=["player_id", "season", "week"], how="left").sort_values("_row")
    games = merged["forward_games"].fillna(0).to_numpy(dtype="int64")
    sums = merged["forward_points_sum"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        ppg = np.where(games > 0, sums / np.where(games > 0, games, 1), np.nan)

    # A full n-week window exists only where the season runs at least n weeks past this one. W_last is
    # the last played week of the season in the frame (17 pre-2021, 18 after) — so the label is always a
    # full-n-week horizon and the ragged season edge is dropped rather than silently shortened.
    last_week = week.groupby(season).transform("max")
    has_window = (week <= (last_week - n)).fillna(False).to_numpy()
    evaluable = has_window & (games >= int(min_games))

    tvec = work["position"].astype("string").map(dict(thresholds)).astype("float64").to_numpy()
    label = np.where(evaluable & np.isfinite(tvec), (ppg >= tvec).astype("float64"), np.nan)

    work["forward_games"] = games
    work["forward_ppg"] = ppg
    work["has_forward_window"] = has_window
    work["is_evaluable"] = evaluable
    work[LABEL_COL] = label
    return work


# --------------------------------------------------------------------------- cohorts
def snap_cohort_mask(
    frame: pd.DataFrame,
    *,
    snap_ewma_max: float = STARTER_SNAP_EWMA,
    positions: Sequence[str] = BREAKOUT_POSITIONS,
) -> pd.Series:
    """The waiver cohort: RB/WR/TE below the starter snap-share line, **or** with no snap history yet.

    ``snap_pct_ewma`` null is kept in (the "no role established yet" arm, ~10% of rows) rather than
    dropped — an emerging player is exactly a waiver candidate. Returns a boolean Series on ``frame``'s
    index.
    """
    pos = frame["position"].astype("string")
    snap = _num_col(frame, "snap_pct_ewma")
    return pos.isin(list(positions)) & (snap.isna() | (snap <= float(snap_ewma_max)))


def production_cohort_mask(
    frame: pd.DataFrame,
    thresholds: Mapping[str, float],
    *,
    positions: Sequence[str] = BREAKOUT_POSITIONS,
) -> pd.Series:
    """The robustness cohort on the **production** axis: trailing ``points_ewma`` below the startable line.

    A different structural cut from :func:`snap_cohort_mask` (production, not role), so agreement between
    the two precision@k reports is evidence the ranking is not an artifact of one cohort definition.
    """
    pos = frame["position"].astype("string")
    ewma = _num_col(frame, "points_ewma")
    tvec = pos.map(dict(thresholds)).astype("float64")
    return pos.isin(list(positions)) & (ewma.isna() | (ewma <= tvec))


# --------------------------------------------------------------------------- features
def _pair_trend(frame: pd.DataFrame, base: str) -> pd.Series:
    """A share trajectory derived from the ``_last``/``_ewma`` pair (no ``*_trend`` column exists)."""
    return _num_col(frame, f"{base}_last") - _num_col(frame, f"{base}_ewma")


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The features as a float ``(n, k)`` array in :data:`BREAKOUT_FEATURES` order; nulls kept as NaN."""
    cols: list[np.ndarray] = []
    for name in BREAKOUT_FEATURES:
        if name in _DERIVED_FEATURES:
            base = name[: -len("_trend")]
            series = _pair_trend(frame, base)
        elif name in frame.columns:
            series = pd.to_numeric(frame[name], errors="coerce")
        else:
            series = pd.Series(np.nan, index=frame.index)
        cols.append(series.to_numpy(dtype="float64"))
    return np.column_stack(cols) if cols else np.empty((len(frame), 0))


# --------------------------------------------------------------------------- logistic model
def _safe_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means / std over non-null entries; all-null columns -> (0, 0). Silences numpy's empty-slice
    ``RuntimeWarning`` (not an ``np.errstate`` category) so an incidentally empty column keeps the run
    at zero warnings — the same handling as ``model.season._safe_stats``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 0.0)
    return mean, std


@dataclass
class _LogisticFit:
    """A fitted per-position L2 logistic: standardisation, weights, intercept and the training base rate."""

    mean: np.ndarray
    std_safe: np.ndarray
    const_mask: np.ndarray
    weights: np.ndarray
    intercept: float
    base_rate: float

    def _standardise(self, matrix: np.ndarray) -> np.ndarray:
        imputed = np.where(np.isnan(matrix), self.mean, matrix)
        z = (imputed - self.mean) / self.std_safe
        z[:, self.const_mask] = 0.0
        return z

    def score(self, matrix: np.ndarray) -> np.ndarray:
        """P(breakout) per row — a monotone ranking score; exact probability is not relied on."""
        if matrix.shape[0] == 0:
            return np.empty(0)
        eta = self.intercept + self._standardise(matrix) @ self.weights
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))

    def to_dict(self) -> dict:
        return {
            "mean": [float(v) for v in self.mean],
            "std_safe": [float(v) for v in self.std_safe],
            "const_mask": [bool(v) for v in self.const_mask],
            "weights": [float(v) for v in self.weights],
            "intercept": float(self.intercept),
            "base_rate": float(self.base_rate),
            "features": list(BREAKOUT_FEATURES),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> _LogisticFit:
        features = list(payload.get("features", BREAKOUT_FEATURES))
        if tuple(features) != BREAKOUT_FEATURES:
            raise ValueError(
                f"artifact feature order {features} != current BREAKOUT_FEATURES — the weights would "
                "bind to the wrong features; refit with scripts/eval_breakout.py"
            )
        return cls(
            mean=np.asarray(payload["mean"], dtype="float64"),
            std_safe=np.asarray(payload["std_safe"], dtype="float64"),
            const_mask=np.asarray(payload["const_mask"], dtype=bool),
            weights=np.asarray(payload["weights"], dtype="float64"),
            intercept=float(payload["intercept"]),
            base_rate=float(payload["base_rate"]),
        )


def _fit_logistic(
    matrix: np.ndarray, y: np.ndarray, alpha: float, *, max_iter: int = 50, tol: float = 1e-8
) -> _LogisticFit:
    """L2-regularised logistic regression by Newton-Raphson on standardised features. Deterministic.

    The intercept is unpenalised and initialised to the base-rate logit; ``alpha`` keeps the penalised
    Hessian positive-definite, so a linearly separable position (rare, but possible at TE with few
    positives) converges instead of diverging. A constant feature is zeroed, contributing nothing.
    """
    mean, std = _safe_stats(matrix)
    std_safe = np.where(std < _STD_EPS, 1.0, std)
    const_mask = std < _STD_EPS
    imputed = np.where(np.isnan(matrix), mean, matrix)
    z = (imputed - mean) / std_safe
    z[:, const_mask] = 0.0

    design = np.column_stack([np.ones(len(z)), z])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    base = float(np.mean(y)) if len(y) else 0.0
    w = np.zeros(design.shape[1])
    if 0.0 < base < 1.0:
        w[0] = float(np.log(base / (1.0 - base)))

    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(design @ w, -30.0, 30.0)))
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        grad = design.T @ (p - y) + penalty @ w
        hess = design.T @ (design * (p * (1.0 - p))[:, None]) + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(hess, grad, rcond=None)
        w = w - step
        if float(np.max(np.abs(step))) < tol:
            break

    return _LogisticFit(
        mean=mean,
        std_safe=std_safe,
        const_mask=const_mask,
        weights=w[1:],
        intercept=float(w[0]),
        base_rate=base,
    )


# --------------------------------------------------------------------------- baseline rankers
def _rank_last_week_points(frame: pd.DataFrame) -> pd.Series:
    return _num_col(frame, "points_last")


def _rank_snap_share_trend(frame: pd.DataFrame) -> pd.Series:
    return _num_col(frame, "snap_pct_trend")


def _rank_role_share_last(frame: pd.DataFrame) -> pd.Series:
    """Rush share for RB, target share for WR/TE — the position's own role signal, last week."""
    pos = frame["position"].astype("string")
    return _num_col(frame, "rush_share_last").where(pos == "RB", _num_col(frame, "target_share_last"))


#: The three naive bars, ranked within each slate. Beating **all three** at k=1 is the shipping test.
BASELINE_RANKERS = {
    "last_week_points": _rank_last_week_points,
    "snap_share_trend": _rank_snap_share_trend,
    "role_share_last": _rank_role_share_last,
}

#: The fallback a position defers to when unproven/unfielded — the strongest, most obvious naive rule.
_SAFE_BASELINE = "last_week_points"


@runtime_checkable
class Ranker(Protocol):
    """Fit on a labelled cohort frame, emit a ranking score per row (higher = more likely to break out).

    :class:`BreakoutModel` and :class:`ColumnRanker` both implement it, so the harness scores the model
    and a naive baseline through one identical path (the #28 ``Predictor`` shape, ranking-valued)."""

    def fit(self, frame: pd.DataFrame) -> Ranker: ...
    def predict(self, frame: pd.DataFrame) -> pd.Series: ...


@dataclass
class ColumnRanker:
    """A naive baseline as a :class:`Ranker`: no fit, ranks by one of :data:`BASELINE_RANKERS`."""

    key: str

    def fit(self, frame: pd.DataFrame) -> ColumnRanker:
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return BASELINE_RANKERS[self.key](frame).astype("float64")


# --------------------------------------------------------------------------- the gate / recorded config
def recorded_gate(path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, str]:
    """The per-position deferral gate read from the committed artifact — the **safe default**.

    Maps a deferred position to the baseline it ranks by; a fielded position is absent. A missing or
    unreadable artifact defers **every** position to :data:`_SAFE_BASELINE`, so a bare
    :class:`BreakoutModel` never fields an unproven logistic — the same fail-safe as
    ``model.weekly.recorded_cold_start_gate`` and ``model.kickdef``'s gate.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {p: _SAFE_BASELINE for p in BREAKOUT_POSITIONS}
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        return {p: _SAFE_BASELINE for p in BREAKOUT_POSITIONS}
    return {str(k): str(v) for k, v in gate.items()}


class BreakoutModel:
    """Per-position L2 logistic on usage trajectories, deferring to the winning baseline where it lost.

    ``fit`` learns one standardised logistic per fielded position on the labelled cohort; ``predict``
    returns a ranking score index-aligned to its input — the logistic P(breakout) at a **fielded**
    position, and the recorded baseline's signal at a **deferred** one. A position is fielded only on a
    **strict win at every k in ``K_VALUES``** (``scripts/eval_breakout.py`` → :func:`breakout_gate` → the
    artifact's ``gate``); ties and losses defer, because a k=1 win on ~117 slates sits inside its own
    standard error and only the deeper k=3/5 confirm the ranking is genuinely better. It is the same
    field-where-it-wins/defer-where-it-loses shape as ``model.season``/``model.weekly``/``model.kickdef``,
    with the every-k requirement its noise floor demands.

    **Safe by default.** ``defer=None`` reads :func:`recorded_gate` (missing artifact -> defer every
    position), so a bare ``BreakoutModel()`` never fields an unproven logistic. The diagnostic
    pure-logistic variant is the explicit opt-out ``BreakoutModel(defer={})``. A position that is
    neither fielded nor fit (e.g. straight from :meth:`load_fitted` before ``fit``, or an unfielded
    position with no stored weights) ranks by :data:`_SAFE_BASELINE` rather than emitting nulls.
    """

    def __init__(
        self,
        *,
        alpha: float = _LOGIT_ALPHA,
        defer: Mapping[str, str] | None = None,
        positions: Sequence[str] = BREAKOUT_POSITIONS,
    ) -> None:
        self.alpha = float(alpha)
        self.positions = tuple(positions)
        self.gate: dict[str, str] = dict(recorded_gate()) if defer is None else dict(defer)
        self._fits: dict[str, _LogisticFit] = {}

    @property
    def fielded_positions(self) -> tuple[str, ...]:
        return tuple(p for p in self.positions if p not in self.gate)

    def fit(self, frame: pd.DataFrame) -> BreakoutModel:
        y_all = pd.to_numeric(frame[LABEL_COL], errors="coerce")
        pos = frame["position"].astype("string")
        self._fits = {}
        for position in self.positions:
            if position in self.gate:
                continue  # deferred positions rank by a baseline — nothing to fit
            sub = frame[(pos == position) & y_all.notna()]
            if sub.empty:
                continue
            y = pd.to_numeric(sub[LABEL_COL], errors="coerce").to_numpy(dtype="float64")
            self._fits[position] = _fit_logistic(_feature_matrix(sub), y, self.alpha)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=frame.index, dtype="float64")
        pos = frame["position"].astype("string")
        for position in self.positions:
            mask = (pos == position).to_numpy()
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            sub = frame[mask]
            if position in self.gate:
                values = BASELINE_RANKERS[self.gate[position]](sub).to_numpy(dtype="float64")
            elif position in self._fits:
                values = self._fits[position].score(_feature_matrix(sub))
            else:
                values = BASELINE_RANKERS[_SAFE_BASELINE](sub).to_numpy(dtype="float64")
            out.iloc[idx] = values
        return out

    def feature_importances(self) -> dict[str, dict[str, float]]:
        """Per fielded position, the standardised logistic coefficient per feature (comparable in scale)."""
        return {
            pos: {feat: float(c) for feat, c in zip(BREAKOUT_FEATURES, fit.weights, strict=True)}
            for pos, fit in self._fits.items()
        }

    def to_dict(self) -> dict:
        return {
            "model": "BreakoutModel",
            "alpha": self.alpha,
            "positions": list(self.positions),
            "gate": dict(self.gate),
            "fits": {pos: fit.to_dict() for pos, fit in self._fits.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> BreakoutModel:
        model = cls(
            alpha=float(payload.get("alpha", _LOGIT_ALPHA)),
            defer={str(k): str(v) for k, v in payload.get("gate", {}).items()},
            positions=tuple(payload.get("positions", BREAKOUT_POSITIONS)),
        )
        model._fits = {
            str(pos): _LogisticFit.from_dict(fit) for pos, fit in payload.get("fits", {}).items()
        }
        return model

    @classmethod
    def load_fitted(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> BreakoutModel:
        """Load the committed artifact into a correctly-configured model: recorded logistic **and** gate.

        The runtime path #34 loads. Fielded positions predict immediately from the stored weights;
        deferred positions rank by their recorded baseline (which needs no fit). Refit from the lake with
        ``scripts/eval_breakout.py`` — the artifact is never hand-edited.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


# --------------------------------------------------------------------------- precision@k harness
#: The top-k depths reported. k=1 is the real decision (a single reverse-priority claim); 3 and 5 are
#: reported for stability. There is deliberately no precision@50 — the league does not act 50 deep.
K_VALUES: tuple[int, ...] = (1, 3, 5)


def precision_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float | None:
    """Fraction of the top-``k`` by ``scores`` that are true breakouts, within one slate.

    ``None`` when the slate has fewer than ``k`` candidates carrying **both** a finite score and a
    label — a board too shallow to act ``k`` deep is not a zero, it is a slate that does not admit the
    question (the same not-scored-vs-scored-zero care as ``model.evaluate.spearman``). Ties are broken
    by the stable input order.
    """
    s = pd.to_numeric(scores, errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(labels, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(s) & np.isfinite(y)
    s, y = s[valid], y[valid]
    if len(s) < k:
        return None
    top = np.argsort(-s, kind="stable")[:k]
    return float(np.mean(y[top]))


def lift(precision: float | None, base_rate: float) -> float | None:
    """Precision@k over the base rate — 1.0 is chance, and it is comparable across positions.

    Raw precision is **not** comparable across positions here (RB's base rate is 3.5x WR's), so every
    reported precision carries its lift; without it the easiest position reads as the strongest.
    """
    if precision is None or not np.isfinite(base_rate) or base_rate <= 0.0:
        return None
    return float(precision / base_rate)


@dataclass(frozen=True)
class BreakoutPositionMetrics:
    """Ranking quality for one position: precision@k and its lift, over the ``(season, week)`` slates."""

    position: str
    n: int  # cohort rows scored (evaluable, labelled)
    base_rate: float  # pooled positive fraction in the cohort at this position
    n_slates: int
    precision: dict[int, float | None]  # k -> mean precision over slates admitting k candidates
    slates_at_k: dict[int, int]  # k -> number of slates that contributed
    lift: dict[int, float | None]  # k -> precision[k] / base_rate


@dataclass(frozen=True)
class BreakoutEvalResult:
    """A ranker's walk-forward grade. Per position by construction — no pooled precision (Decision #5)."""

    predictor: str
    test_seasons: tuple[int, ...]
    per_position: dict[str, BreakoutPositionMetrics]
    predictions: pd.DataFrame  # player_id, season, week, position, score, label


def _position_metrics(
    predictions: pd.DataFrame, *, positions: Sequence[str], ks: Sequence[int]
) -> dict[str, BreakoutPositionMetrics]:
    """precision@k (per ``(season, week)`` slate, averaged) + base rate + lift, per position."""
    out: dict[str, BreakoutPositionMetrics] = {}
    for pos in positions:
        sub = predictions[predictions["position"] == pos]
        labelled = sub[pd.to_numeric(sub["label"], errors="coerce").notna()]
        if labelled.empty:
            continue
        base = float(pd.to_numeric(labelled["label"], errors="coerce").mean())
        prec: dict[int, float | None] = {}
        slates: dict[int, int] = {}
        for k in ks:
            vals = [
                p
                for _slate, grp in labelled.groupby(["season", "week"], sort=True)
                if (p := precision_at_k(grp["score"], grp["label"], k)) is not None
            ]
            prec[k] = float(np.mean(vals)) if vals else None
            slates[k] = len(vals)
        out[pos] = BreakoutPositionMetrics(
            position=pos,
            n=int(len(labelled)),
            base_rate=base,
            n_slates=int(labelled.groupby(["season", "week"], sort=False).ngroups),
            precision=prec,
            slates_at_k=slates,
            lift={k: lift(prec[k], base) for k in ks},
        )
    return out


def evaluate_breakout(
    ranker: Ranker,
    frame: pd.DataFrame,
    *,
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
    positions: Sequence[str] = BREAKOUT_POSITIONS,
    ks: Sequence[int] = K_VALUES,
    name: str | None = None,
) -> BreakoutEvalResult:
    """Walk-forward fit->rank over ``test_seasons``, scored per position on precision@k.

    ``frame`` is the **labelled cohort** (evaluable rows only, one position group per row). Reuses
    ``model.evaluate.walk_forward_splits``, so the leak gate applies unchanged — a split whose training
    rows reach into the test season refuses to exist. Each split fits on strictly-earlier seasons and
    ranks the test season; ranks are reindexed to the test frame, so a misaligned Series surfaces as
    nulls rather than a silent scramble.
    """
    cohort = frame[frame["position"].isin(list(positions))]
    seasons_used: list[int] = []
    chunks: list[pd.DataFrame] = []
    for split in walk_forward_splits(cohort, test_seasons=test_seasons):
        model = ranker.fit(split.train)
        scores = model.predict(split.test)
        if not isinstance(scores, pd.Series):
            raise TypeError(
                f"{type(ranker).__name__}.predict must return a pandas Series, got "
                f"{type(scores).__name__}"
            )
        scores = pd.to_numeric(scores.reindex(split.test.index), errors="coerce")
        chunks.append(
            pd.DataFrame(
                {
                    "player_id": split.test["player_id"].to_numpy(),
                    "season": split.test["season"].to_numpy(),
                    "week": split.test["week"].to_numpy(),
                    "position": split.test["position"].to_numpy(),
                    "score": scores.to_numpy(),
                    "label": pd.to_numeric(split.test[LABEL_COL], errors="coerce").to_numpy(),
                }
            )
        )
        seasons_used.append(split.test_season)
    predictions = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=["player_id", "season", "week", "position", "score", "label"])
    )
    return BreakoutEvalResult(
        predictor=name or type(ranker).__name__,
        test_seasons=tuple(seasons_used),
        per_position=_position_metrics(predictions, positions=positions, ks=ks),
        predictions=predictions,
    )


def _strongest_baseline(
    baseline_metrics: Mapping[str, Mapping[str, BreakoutPositionMetrics]], pos: str, ks: Sequence[int]
) -> str:
    """The baseline a deferred position ranks by: the strongest at the decision k (the smallest k)."""
    k0 = min(ks)
    best_key, best_p = _SAFE_BASELINE, None
    for key, per_pos in baseline_metrics.items():
        bp = per_pos.get(pos)
        p = None if bp is None else bp.precision.get(k0)
        if p is not None and (best_p is None or p > best_p):
            best_key, best_p = key, p
    return best_key


def breakout_gate(
    model_metrics: Mapping[str, BreakoutPositionMetrics],
    baseline_metrics: Mapping[str, Mapping[str, BreakoutPositionMetrics]],
    *,
    positions: Sequence[str] = BREAKOUT_POSITIONS,
    ks: Sequence[int] = K_VALUES,
) -> dict[str, str]:
    """The measured deferral gate: field the logistic only where it beats every baseline at **every k**.

    Pure. precision@1 on ~117 binary slates has a standard error near 0.04, so a k=1 win inside that band
    is indistinguishable from noise unless the *ranking* is genuinely better — and the deeper k=3/5, which
    use more of each slate, are the corroboration. So a position is **fielded** only when the model
    strictly beats every baseline at **every** k in ``ks``; any loss or tie at any k **defers** it to the
    strongest baseline (the ties-defer rule shared with ``model.season``/``model.weekly``/
    ``model.kickdef``). k=1 stays the decision you act on — requiring the deeper ks is what keeps that call
    from resting on one noisy proportion. A win present only at the noisiest k, and absent at every other,
    is noise; the paired :func:`mcnemar_k1` test is the evidence.
    """
    gate: dict[str, str] = {}
    for pos in positions:
        m = model_metrics.get(pos)
        beats_all = m is not None
        for per_pos in baseline_metrics.values():
            bp = per_pos.get(pos)
            if bp is None:
                continue
            for k in ks:
                mp = m.precision.get(k) if m is not None else None
                bpk = bp.precision.get(k)
                if mp is None or bpk is None or mp <= bpk + 1e-12:
                    beats_all = False
        if not beats_all:
            gate[pos] = _strongest_baseline(baseline_metrics, pos, ks)
    return gate


def _top1_by_slate(predictions: pd.DataFrame, position: str) -> dict[tuple, int]:
    """Per ``(season, week)`` slate at ``position``, the label of the top-1 candidate by score.

    Ties broken by the stable input order — the same top-1 pick :func:`precision_at_k` scores.
    """
    sub = predictions[predictions["position"] == position]
    valid = sub[
        pd.to_numeric(sub["score"], errors="coerce").notna()
        & pd.to_numeric(sub["label"], errors="coerce").notna()
    ]
    out: dict[tuple, int] = {}
    for slate, grp in valid.groupby(["season", "week"], sort=True):
        s = pd.to_numeric(grp["score"], errors="coerce").to_numpy(dtype="float64")
        y = pd.to_numeric(grp["label"], errors="coerce").to_numpy(dtype="float64")
        out[slate] = int(y[int(np.argsort(-s, kind="stable")[0])])
    return out


def mcnemar_k1(
    model_predictions: pd.DataFrame, baseline_predictions: pd.DataFrame, position: str
) -> dict:
    """Paired McNemar on the k=1 pick, model vs a baseline over the shared ``(season, week)`` slates.

    Both rankers score the same slates, so the correct comparison is **paired**: over each slate's single
    top-1 pick, count the slates where only the model's pick broke out (``model_plus`` = b) and where only
    the baseline's did (``base_plus`` = c). ``z = (b - c) / sqrt(b + c)``; ``|z| > 1.96`` is significant at
    0.05. Concordant slates (both right, or both wrong) carry no information about which ranker is better
    and are excluded — which is exactly why a raw precision@1 gap over 117 slates can look real while
    resting on a handful of discordant picks.
    """
    mh = _top1_by_slate(model_predictions, position)
    bh = _top1_by_slate(baseline_predictions, position)
    slates = set(mh) & set(bh)
    b = sum(1 for s in slates if mh[s] == 1 and bh[s] == 0)
    c = sum(1 for s in slates if mh[s] == 0 and bh[s] == 1)
    disc = b + c
    return {
        "discordant": disc,
        "model_plus": b,
        "base_plus": c,
        "z": float((b - c) / np.sqrt(disc)) if disc > 0 else 0.0,
        "significant": bool(disc > 0 and abs(b - c) / np.sqrt(disc) > 1.96),
    }

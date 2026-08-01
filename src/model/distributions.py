"""Fit the two simulators' outcome/durability knobs from the lake (Phase 9, ticket #32).

``draftsim`` and ``seasonsim`` carry three per-position knobs — season CV (:data:`POSITION_CV`),
single-game CV (:data:`GAME_CV`) and injury ``(P(setback), mean games missed)`` (:data:`INJURY_RISK`) —
that shipped as hand-picked heuristics. This module **earns them from the lake**. It is the fitting
logic only; ``scripts/eval_distributions.py`` orchestrates it, writes the artifact
(``src/model/fit/distributions.json``) and the measured report (``docs/model-distributions.md``), and
runs the before/after championship-odds comparison.

The one statistical trap, and how each knob is fit
--------------------------------------------------
The sims want the CV of ``actual | projection`` — the spread around *a drafted player's projection* —
**not** the CV of actuals (which is cross-player dispersion, far wider and a different quantity). We
cannot condition on a historical projection (``baseline_sleeper_points`` is null for all of 2016-2025),
so the honest route is the models this phase already shipped, fit from their **walk-forward
out-of-sample** residuals ``r = actual / pred``:

* :data:`POSITION_CV` ← :class:`model.season.SeasonModel` season-grain residuals on the **drafted
  cohort** (:data:`SEASON_COHORT_BY_POSITION` — the per-season top-N by projection this league rosters,
  from the locked 12 x 14 roster), **setback-free (healthy)**. Both cuts are load-bearing. The *drafted*
  cut, because a knob fit over the wider fringe measures volatile backups the sim never drafts: on the
  ~367/season pool the residual CVs run 0.53-1.07 at the five skill positions (recorded in the artifact
  as ``upper_bound_position_cv``) and fail the coherence gate at 5 of 6; on the 168/season drafted pool
  they run 0.24-0.61 and
  pass at 5 of 6 — the verdict inverts on the cohort alone. The *healthy* cut, because the injury
  knob owns games-missed variance and the two must not double-count. The wide-cohort CV is reported as
  an upper bound and the full (setback-inclusive) CV alongside it, so both removed tails stay visible.
* :data:`GAME_CV` (QB/RB/WR/TE) ← :class:`model.weekly.WeeklyModel` residuals; (K/DEF) ←
  :class:`model.kickdef.KickDefModel` residuals. A week a player did not play has **no stat-line row**
  (the #29 finding), so weekly residuals are already conditioned on availability — no healthy filter is
  needed or applied at game grain.
* :data:`INJURY_RISK` ← a contiguous **injury absence** of ≥ :data:`SETBACK_MIN_WEEKS` weeks — a gap in
  a player's *played* weeks, **tenure-bounded** (a real opportunity to play, so pre-arrival absence is
  not injury) and **injury-corroborated** (the player carries an injury-report status in the run or the
  week it began, which excludes byes and clean benchings). Fit on the same **drafted cohort** as the
  season CV, for the same reason. Emphatically **not** contiguous ``report_status == "Out"`` runs, which
  was the first attempt and is wrong: a player placed on IR drops off the weekly report entirely, so a
  season-ending injury produces *zero* ``Out`` weeks. Measured, only ~35% of season-enders keep any
  ``Out`` row, and the Out-only rate undercounts by 2.5-4x — precisely on the severe injuries the knob
  exists to model. The Out-only rate is kept in the report as the IR-truncated lower bound. DST is never
  on the injury report, so it falls back to the heuristic; any position too thin to fit does too.

Estimator. ``CV = std(r) / mean(r)`` — matches the sim's mean-preserving lognormal exactly, is
scale-free across heterogeneous projections, and tolerates the real weekly zeros a log-space estimator
cannot. Fit **per season, then averaged** across seasons (:func:`_season_averaged_cv`): a walk-forward
model trained only on 16-game seasons predicts the first 17-game season (2021) with a systematically low
mean, and averaging per-season CVs — each divided by *its own* mean — keeps that per-season *level* drift
out of the pooled *dispersion*. ``mean(r)`` and ``CV(r)`` are reported by season (to show the drift) and
by prediction tercile (to record whether CV slides with the projection — the sim takes one constant
either way).

Thresholds, honestly. The cohorts and the coherence *principle* were declared **before** measuring (the
anti-pattern spec Decision #3 exists to prevent); two floors were not, and the report says so rather
than burying it. Re-cutting to the drafted roster makes the per-position cohorts small **by
construction** (12-60 players/season), so floors sized for the old wide pools were the wrong instrument
and were re-derived on the cohort structure: :data:`MIN_CV_N` 100→40 (flipped **K and DEF** season CV
from fallback to fitted) and :data:`MIN_INJURY_SEASONS` 200→50 (keeps **QB, TE, K** injury fitted, whose
drafted-cohort n's are 178/140/94). Both are **verdict-affecting**, for a legitimate reason, and neither
is tuned to a verdict — the coherence gate and injury coverage still decide. Likewise
:data:`MAX_SEASON_FACTOR_CV`: the principle (a >50% whole-season baseline swing is projection error, not
outcome uncertainty) is pre-data, but the exact 0.5 was chosen after seeing the factors; the split is
robust anywhere in ~[0.44, 0.55]. All of it is in the report so the pattern is judgeable, not
discoverable.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset.assemble import build_training_frame
from draftsim.distributions import (
    HEURISTIC_GAME_CV,
    HEURISTIC_INJURY_RISK,
    HEURISTIC_POSITION_CV,
    SEASON_GAMES,
)
from model.evaluate import DEFAULT_TEST_SEASONS, FANTASY_POSITIONS, evaluate
from model.frame import season_frame_from_weekly
from model.kickdef import KICKDEF_POSITIONS, KickDefModel, build_kickdef_frame
from model.season import SeasonModel, evaluate_season
from model.weekly import SKILL_POSITIONS, WeeklyModel
from seasonsim.distributions import season_factor_cv
from store.lake import StorageBackend, read_source

_LOG = logging.getLogger(__name__)

POSITIONS: tuple[str, ...] = FANTASY_POSITIONS

# --------------------------------------------------------------------------- thresholds (declared first)
#: Rows a single season needs to contribute its within-season CV to the per-season average. A CV from
#: fewer than five residuals is meaningless; the season is dropped from the average (and reported as such).
MIN_CV_SEASON_N = 5

#: Kept residual rows a CV cell needs to be **fitted**; below it the CV is too thin and the cell defers.
#: Re-derived for the **drafted cohort** (Amendment A): the earlier 100 was calibrated for the old wide
#: ~367/season pool, and on the drafted cohort — structurally :data:`SEASON_COHORT_BY_POSITION` × the 8
#: test seasons, i.e. 96 (K/DEF) … 480 (WR) — it excluded the small-by-construction K/DEF (96) and thin
#: TE by a hair, a threshold for a cohort that no longer exists. The per-season average already requires
#: ≥ :data:`MIN_CV_SEASON_N` rows *per season*, so the total-n floor need only ensure every test season
#: can contribute: ``MIN_CV_SEASON_N × |test seasons| = 5 × 8 = 40``. The coherence gate, not this floor,
#: is the decider on the drafted cohort — tied to the roster structure, not to any target verdict.
MIN_CV_N = 40

#: Drafted-cohort player-seasons an injury cell needs to be **fitted**. Like the injury *cohort* itself
#: (Amendment C), this is re-derived for the drafted denominator: the injury knob is applied to the same
#: rostered players season CV is, so it is fit on the same per-season top-N by projection — which is small
#: by construction (12-60 players/season → 96-480 over the 8 OOS seasons). The wide-pool 200 excludes the
#: structurally-small K/TE/QB drafted cohorts (94/140/178) purely for being small, the wrong instrument. A
#: Bernoulli setback rate is adequately estimated from ~50 player-seasons (SE ≈ 0.07 at p ≈ 0.4), safely
#: below the smallest drafted cohort (K/DEF, ~96), so **50** admits every injury-coverable position and a
#: position below it defers on evidence. Declared on the cohort structure, not a target verdict.
MIN_INJURY_SEASONS = 50

#: The per-game projection floor for :data:`GAME_CV` — the ratio ``actual/pred`` is noise-dominated for a
#: player the sim never starts, so the floor keeps the fit to a flex-worthy weekly projection. Declared on
#: role-relevance grounds; the by-tercile table reports whether it does real work.
GAME_PRED_FLOOR = 4.0

#: **The season-CV cohort — this league's drafted players per position, from roster math** (not a
#: projection-value floor). A per-position knob is applied to the players of that position a 12-team
#: league rosters; fitting it over a wider pool measures the volatile fringe (backup QBs, RB3s) the sim
#: never drafts, which is what inflates the pooled CV (the tercile slide, §B). Derived from the locked
#: roster (CLAUDE.md: 12 teams; 9 starters + 5 bench = 14 active): ~2 QB, ~4 RB, ~5 WR, ~1.5 TE, 1 K,
#: 1 DEF rostered per team → the per-position counts below (sum 174 ≈ the 168 active spots + streaming
#: churn). Declared from roster structure **before** re-measuring — the same class of pre-data declaration
#: as #30's ``ANCHOR_FLOOR_PCT`` — and the per-season top-N by projection is taken as the cohort. The
#: alternative structural cut (top-:data:`SEASON_COHORT_TOTAL` overall by projection) is reported as a
#: robustness check; the two disagree only at the margin about which single position (if any) fails.
SEASON_COHORT_BY_POSITION: dict[str, int] = {"QB": 24, "RB": 48, "WR": 60, "TE": 18, "K": 12, "DEF": 12}

#: Total active roster spots (12 teams × 14) — the cross-position robustness cohort (top-N overall by
#: projection). It over-samples the high-scoring positions (a value-ranked cut is mostly QB/RB), which is
#: why the per-position cut above is the primary denominator for a per-position knob.
SEASON_COHORT_TOTAL = 168

#: The wide projection floor kept only to report the **upper-bound** season CV (the cohort of ~367/season
#: the old ``SEASON_PRED_FLOOR`` admitted). Recorded, never shipped: a residual CV over the fringe bounds
#: the drafted-cohort CV from above, and SeasonModel's residual itself bounds the spread around *its own*
#: projection rather than around Sleeper's — a caveat that is right but historically unmeasurable
#: (``baseline_sleeper_points`` is null for 2016-2025).
WIDE_SEASON_PRED_FLOOR = 50.0

#: A genuine roster member is around by week 4 and plays at least three weeks — the anti-censoring
#: denominator (a mid-season call-up or a one-week cameo is not "an opportunity to play a full season").
QUALIFY_FIRST_SEEN_WEEK = 4
QUALIFY_MIN_PLAYED = 3

#: A "significant" setback is a contiguous **injury absence** of at least this many weeks. An absence is
#: injury-linked when the player carries an injury-report status (Out/Doubtful/Questionable) in the run or
#: the week it began — which catches the case a pure ``Out``-run misses: a season-ending injury moves the
#: player to IR, dropping him off the weekly report entirely (measured: only ~35% of season-enders have
#: any ``Out`` row in their gap), so the pre-IR designation is the only corroboration left. A bye or a
#: clean benching/cut carries no injury report and is excluded. A one-week absence is week-to-week noise
#: already inside :data:`GAME_CV`, not a durability event a bench is drafted for.
SETBACK_MIN_WEEKS = 2

#: The **season-factor coherence** band a fitted season CV must land in — the sim's own
#: :func:`seasonsim.distributions.season_factor_cv` identity turned into a gate (trap 2, both ends). A
#: season total is a ~17-game sum, so ``CV_season`` implies a season-level factor
#: ``CV_factor = season_factor_cv(CV_season, CV_game, 17)`` that must be **> 0** (else single-game noise
#: alone exceeds the season CV and the factor floors — the season loses its correlation) **and not
#: implausibly large** (above the cap, a player's whole-season baseline swings more than the projection
#: could plausibly be uncertain by). **Provenance, stated plainly:** the *principle* (a season-baseline
#: swing beyond ~50% is not outcome uncertainty) is pre-data, but the exact **0.5 was chosen after seeing
#: the fitted factors** — it sits above the coherent positions (≤ ~0.44 on the drafted cohort) and below
#: the incoherent one (~0.56); it was not declared blind. The fitted/fallback split is robust to any
#: value in ~[0.44, 0.55].
MAX_SEASON_FACTOR_CV = 0.5

#: Terciles the by-pred CV slide is reported over (Amendment 3 — the constant-CV assumption, recorded).
_N_TERCILES = 3


# =========================================================================== residual CV
@dataclass(frozen=True)
class CvCell:
    """A fitted CV for one position at one grain, plus the diagnostics that make it judgeable."""

    position: str
    grain: str  # "season" | "game"
    cv: float | None  # the shipped estimate: healthy (season) / played (game), per-season-averaged
    mean_r: float | None  # overall mean of actual/pred — a bias check, ~1 for a calibrated model
    n: int  # kept residual rows behind ``cv``
    n_seasons: int  # seasons that contributed a within-season CV to the average
    full_cohort_cv: float | None  # season grain: CV including setback seasons; None at game grain
    upper_bound_cv: float | None  # season grain: CV over the wide (~367/season) cohort; None at game grain
    by_season: pd.DataFrame  # season, n, mean_r, cv_r — the era-robustness table (Amendment 2)
    by_tercile: pd.DataFrame  # tercile, n, pred_lo, pred_hi, mean_r, cv_r — the slide table (Amendment 3)


def _cv(values: np.ndarray) -> float | None:
    """Coefficient of variation ``std/mean`` (sample std), or ``None`` if undefined (< 2 rows / mean≈0)."""
    v = values[np.isfinite(values)]
    if v.size < 2:
        return None
    mean = float(v.mean())
    if abs(mean) < 1e-9:
        return None
    return float(v.std(ddof=1) / mean)


def _cv_by(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """``group_col, n, mean_r, cv_r`` for each group of a frame carrying an ``r`` column."""
    rows: list[dict] = []
    for key, grp in frame.groupby(group_col, sort=True):
        r = grp["r"].to_numpy(dtype="float64")
        rows.append(
            {
                group_col: key,
                "n": int(np.isfinite(r).sum()),
                "mean_r": float(np.nanmean(r)) if np.isfinite(r).any() else None,
                "cv_r": _cv(r),
            }
        )
    return pd.DataFrame(rows, columns=[group_col, "n", "mean_r", "cv_r"])


def _season_averaged_cv(by_season: pd.DataFrame) -> tuple[float | None, int]:
    """Mean of the per-season CVs over seasons with enough rows — the era-robust estimate.

    Averaging per-season CVs (each divided by that season's own mean) keeps a per-season *level* drift —
    e.g. 2021's low season-grain predictions from a 16-game-only train set — out of the *dispersion*
    (Amendment 2). Returns ``(cv, n_seasons_used)``.
    """
    usable = by_season[(by_season["n"] >= MIN_CV_SEASON_N) & by_season["cv_r"].notna()]
    if usable.empty:
        return None, 0
    return float(usable["cv_r"].mean()), int(len(usable))


def _tercile_cv(frame: pd.DataFrame) -> pd.DataFrame:
    """``CV(r)`` within each prediction tercile — records whether CV slides with the projection.

    The sim takes one constant per position regardless; this converts a known-unknown into a recorded
    limitation and shows whether :data:`GAME_PRED_FLOOR` / :data:`SEASON_PRED_FLOOR` is doing real work.
    """
    if len(frame) < _N_TERCILES:
        return pd.DataFrame(columns=["tercile", "n", "pred_lo", "pred_hi", "mean_r", "cv_r"])
    codes = pd.qcut(frame["pred"].rank(method="first"), _N_TERCILES, labels=False)
    rows: list[dict] = []
    for t in range(_N_TERCILES):
        grp = frame[codes == t]
        r = grp["r"].to_numpy(dtype="float64")
        rows.append(
            {
                "tercile": t + 1,
                "n": int(len(grp)),
                "pred_lo": float(grp["pred"].min()) if len(grp) else None,
                "pred_hi": float(grp["pred"].max()) if len(grp) else None,
                "mean_r": float(np.nanmean(r)) if np.isfinite(r).any() else None,
                "cv_r": _cv(r),
            }
        )
    return pd.DataFrame(rows)


def _residual_frame(pred: pd.DataFrame, position: str) -> pd.DataFrame:
    """One position's rows with finite ``actual``/``pred`` (``pred > 0``) and the ratio ``r`` — no cohort
    cut yet (the caller selects the cohort: a floor for game grain, per-season top-N for season grain)."""
    sub = pred[pred["position"].astype("string") == position].copy()
    sub["actual"] = pd.to_numeric(sub["actual"], errors="coerce")
    sub["pred"] = pd.to_numeric(sub["pred"], errors="coerce")
    sub = sub[sub["actual"].notna() & sub["pred"].notna() & (sub["pred"] > 0)]
    sub["r"] = sub["actual"] / sub["pred"]
    return sub[np.isfinite(sub["r"])]


def _select_cohort(frame: pd.DataFrame, *, floor: float | None, top_n: int | None) -> pd.DataFrame:
    """The cohort the CV is fit on: per-season **top-N by projection** (season) or ``pred >= floor`` (game)."""
    if top_n is not None:
        if frame.empty:
            return frame
        return pd.concat(
            [g.nlargest(top_n, "pred") for _, g in frame.groupby("season", sort=True)],
            ignore_index=True,
        )
    if floor is not None:
        return frame[frame["pred"] >= floor]
    return frame


def _healthy_mask(cohort: pd.DataFrame, setback_keys: set[tuple]) -> np.ndarray:
    keys = [
        (str(pid), int(s)) if pd.notna(s) else None
        for pid, s in zip(cohort["player_id"], cohort["season"], strict=False)
    ]
    return np.array([k is not None and k not in setback_keys for k in keys], dtype=bool)


def _cohort_cv(cohort: pd.DataFrame, setback_keys: set[tuple] | None) -> dict:
    """Per-season-averaged CV of the healthy cohort, plus the full-cohort CV and the diagnostics."""
    healthy = cohort[_healthy_mask(cohort, setback_keys)] if setback_keys is not None else cohort
    by_season = _cv_by(healthy, "season")
    cv, n_seasons = _season_averaged_cv(by_season)
    full_cv = _season_averaged_cv(_cv_by(cohort, "season"))[0] if setback_keys is not None else None
    r = healthy["r"].to_numpy(dtype="float64")
    return {
        "cv": cv,
        "full_cv": full_cv,
        "n": int(len(healthy)),
        "n_seasons": n_seasons,
        "mean_r": float(np.nanmean(r)) if r.size and np.isfinite(r).any() else None,
        "by_season": by_season,
        "by_tercile": _tercile_cv(healthy),
    }


def fit_cv_cells(
    pred: pd.DataFrame,
    *,
    grain: str,
    floor: float | None = None,
    top_n_by_position: Mapping[str, int] | None = None,
    setback_keys: set[tuple] | None = None,
    wide_floor: float | None = None,
    positions: Sequence[str] = POSITIONS,
) -> dict[str, CvCell]:
    """Per-position :class:`CvCell` from a walk-forward prediction frame.

    ``pred`` carries ``player_id, season, position, actual, pred`` (weekly frames also carry ``week``).
    The **season** grain fits over the **drafted cohort** (``top_n_by_position`` — the per-season top-N by
    projection, the players this league actually rosters at that position) and removes ``setback_keys`` for
    the healthy CV, with the full-cohort and a wide-cohort (``wide_floor``) **upper-bound** CV recorded
    alongside. The **game** grain fits over ``pred >= floor`` with no healthy filter (an out week has no
    row).
    """
    out: dict[str, CvCell] = {}
    for pos in positions:
        rf = _residual_frame(pred, pos)
        top_n = top_n_by_position.get(pos) if top_n_by_position else None
        m = _cohort_cv(_select_cohort(rf, floor=floor, top_n=top_n), setback_keys)
        upper = None
        if wide_floor is not None:
            upper = _cohort_cv(_select_cohort(rf, floor=wide_floor, top_n=None), setback_keys)["cv"]
        out[pos] = CvCell(
            position=pos,
            grain=grain,
            cv=m["cv"],
            mean_r=m["mean_r"],
            n=m["n"],
            n_seasons=m["n_seasons"],
            full_cohort_cv=m["full_cv"],
            upper_bound_cv=upper,
            by_season=m["by_season"],
            by_tercile=m["by_tercile"],
        )
    return out


def robustness_overall_cohort(
    pred: pd.DataFrame,
    setback_keys: set[tuple] | None,
    game_cv: Mapping[str, float],
    *,
    n_overall: int = SEASON_COHORT_TOTAL,
    positions: Sequence[str] = POSITIONS,
) -> pd.DataFrame:
    """Season-CV verdicts under the **top-N-overall** cohort — the robustness cross-check for Amendment A.

    A cross-position cut (rank every position together per season, keep the top ``n_overall`` by
    projection), then the same per-position CV and coherence gate. Reported next to the primary
    per-position cut because the two disagree at the margin about which single position fails.
    """
    work = pred.copy()
    work["actual"] = pd.to_numeric(work["actual"], errors="coerce")
    work["pred"] = pd.to_numeric(work["pred"], errors="coerce")
    work = work[work["actual"].notna() & work["pred"].notna() & (work["pred"] > 0)]
    top = (
        pd.concat(
            [g.nlargest(n_overall, "pred") for _, g in work.groupby("season", sort=True)],
            ignore_index=True,
        )
        if len(work)
        else work
    )
    top["r"] = top["actual"] / top["pred"]
    top = top[np.isfinite(top["r"])]
    rows: list[dict] = []
    for pos in positions:
        m = _cohort_cv(top[top["position"].astype("string") == pos], setback_keys)
        cv = m["cv"]
        factor = collapse_row(pos, cv, game_cv[pos])["factor_cv"] if cv is not None else None
        fitted = (
            cv is not None and m["n"] >= MIN_CV_N and factor is not None and 0.0 < factor <= MAX_SEASON_FACTOR_CV
        )
        rows.append(
            {"position": pos, "cv": cv, "n": m["n"], "factor_cv": factor,
             "verdict": "fitted" if fitted else "heuristic-fallback"}
        )
    return pd.DataFrame(rows)


# =========================================================================== injuries
@dataclass(frozen=True)
class InjuryCell:
    """A fitted ``(P(setback), mean games missed)`` for one position, plus its denominator and coverage."""

    position: str
    p_setback: float | None  # shipped rate — the DRAFTED cohort (Amendment C)
    mean_missed: float | None
    n_qualified: int  # drafted-cohort player-seasons (the denominator the sim applies the knob to)
    n_setback: int
    coverage: int  # drafted-cohort player-seasons that appeared on the injury report at all
    reason: str | None  # fallback reason, or None if the cell is fittable
    p_out_only: float | None = None  # the Out-run-only rate — the IR-truncated lower bound, for contrast
    p_wide: float | None = None  # the wide (all-qualified) cohort rate — the pre-Amendment-C number
    n_wide: int = 0  # wide-cohort player-seasons (reported next to the drafted n)


def _contiguous_runs(weeks: Iterable[int]) -> list[int]:
    """Lengths of the maximal runs of consecutive week numbers in ``weeks`` (order/dupes irrelevant)."""
    ws = sorted({int(w) for w in weeks})
    if not ws:
        return []
    runs: list[int] = []
    start = prev = ws[0]
    for w in ws[1:]:
        if w == prev + 1:
            prev = w
        else:
            runs.append(prev - start + 1)
            start = prev = w
    runs.append(prev - start + 1)
    return runs


def _corroborated_setback(
    played: frozenset[int], report: frozenset[int], first_played: int, season_end: int
) -> int:
    """Longest **injury-corroborated absence** run (≥ ``SETBACK_MIN_WEEKS``) within tenure; else 0.

    An absence run is the maximal stretch of REG weeks in ``[first_played, season_end]`` the player did not
    play. It counts as a setback only if the player carries an injury-report week (any status) in
    ``[run_start - 1, run_end]`` — the corroboration that turns a bye or a clean benching (no report) away
    while catching a season-ender that dropped to IR (the pre-IR designation sits at the run's start).
    """
    absent = sorted(wk for wk in range(first_played, season_end + 1) if wk not in played)
    best, cur = 0, []

    def _flush(run: list[int]) -> None:
        nonlocal best
        if len(run) >= SETBACK_MIN_WEEKS and any(w in report for w in range(run[0] - 1, run[-1] + 1)):
            best = max(best, len(run))

    for wk in absent:
        if cur and wk == cur[-1] + 1:
            cur.append(wk)
        else:
            _flush(cur)
            cur = [wk]
    _flush(cur)
    return best


def _player_season_injury_table(played: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(gsis_id, season)`` fantasy player-season: tenure, played weeks, and its setback.

    ``played`` is the training frame's played weeks (one row per played player-week, fantasy positions,
    ``gsis_id`` non-null); ``injuries`` is the REG ``nflverse_injuries`` rows (``gsis_id, season, week,
    report_status``). A key present only on the injury report — no fantasy-position played week — is
    dropped, so the denominator is fantasy contributors, not every listed body. ``longest_setback`` is the
    corroborated-absence run (:func:`_corroborated_setback`); ``longest_out_run`` (the ``Out``-only run) is
    kept only to report the IR-truncated lower bound alongside it.
    """
    pl = played.dropna(subset=["gsis_id"]).copy()
    pl["gsis_id"] = pl["gsis_id"].astype("string")
    pl["season"] = pd.to_numeric(pl["season"], errors="coerce").astype("Int64")
    pl["week"] = pd.to_numeric(pl["week"], errors="coerce").astype("Int64")
    pl = pl.dropna(subset=["gsis_id", "season", "week"])
    season_end = {int(s): int(w) for s, w in pl.groupby("season")["week"].max().items()}

    def _modal(series: pd.Series):
        m = series.dropna().mode()
        return m.iat[0] if not m.empty else (series.iloc[0] if len(series) else None)

    def _weeks(series: pd.Series) -> frozenset:
        return frozenset(int(x) for x in series.dropna())

    base = (
        pl.groupby(["gsis_id", "season"], observed=True)
        .agg(
            player_id=("player_id", _modal),
            position=("position", _modal),
            n_played=("week", "nunique"),
            first_played=("week", "min"),
            played_set=("week", _weeks),
        )
        .reset_index()
    )

    inj = injuries.copy()
    inj["gsis_id"] = inj["gsis_id"].astype("string")
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce").astype("Int64")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce").astype("Int64")
    inj = inj.dropna(subset=["gsis_id", "season", "week"])
    is_out = inj["report_status"].astype("string") == "Out"
    rep = inj.groupby(["gsis_id", "season"], observed=True)["week"].agg(_weeks).reset_index(name="report_set")
    outs = inj[is_out].groupby(["gsis_id", "season"], observed=True)["week"].agg(_weeks).reset_index(name="out_set")

    tab = base.merge(rep, on=["gsis_id", "season"], how="left").merge(outs, on=["gsis_id", "season"], how="left")
    empty = frozenset()
    tab["report_set"] = tab["report_set"].apply(lambda x: x if isinstance(x, frozenset) else empty)
    tab["out_set"] = tab["out_set"].apply(lambda x: x if isinstance(x, frozenset) else empty)
    tab["on_report"] = tab["report_set"].apply(len) > 0

    first_seen = tab.apply(
        lambda r: min([int(r.first_played), *( [min(r.report_set)] if r.report_set else [])]), axis=1
    )
    tab["qualified"] = (first_seen <= QUALIFY_FIRST_SEEN_WEEK) & (tab["n_played"] >= QUALIFY_MIN_PLAYED)
    tab["longest_out_run"] = tab["out_set"].apply(lambda s: max(_contiguous_runs(s), default=0))
    tab["longest_setback"] = tab.apply(
        lambda r: _corroborated_setback(
            r.played_set, r.report_set, int(r.first_played), season_end.get(int(r.season), int(r.first_played))
        ),
        axis=1,
    )
    # _corroborated_setback already applies SETBACK_MIN_WEEKS (it returns 0 or a run >= the minimum), so a
    # non-zero corroborated run is a setback — the single source of the threshold is that function.
    tab["setback"] = tab["longest_setback"] > 0
    return tab


def drafted_cohort_keys(
    season_pred: pd.DataFrame, *, top_n_by_position: Mapping[str, int] = SEASON_COHORT_BY_POSITION
) -> set[tuple]:
    """The ``(player_id, season)`` set the sim drafts — per-season top-N by projection per position.

    The same cut :func:`fit_cv_cells` applies to the season CV, exposed as a set so the injury fit uses the
    **identical** denominator (Amendment C: the injury knob is applied to exactly these rostered players,
    not to the loose qualified pool that inverted TE above RB).
    """
    work = season_pred.copy()
    work["pred"] = pd.to_numeric(work["pred"], errors="coerce")
    work = work[work["pred"].notna()]
    keys: set[tuple] = set()
    for pos, n in top_n_by_position.items():
        sub = work[work["position"].astype("string") == pos]
        for _, g in sub.groupby("season", sort=True):
            top = g.nlargest(n, "pred")
            keys.update((str(pid), int(s)) for pid, s in zip(top["player_id"], top["season"], strict=False))
    return keys


def fit_injuries(
    played: pd.DataFrame,
    injuries: pd.DataFrame,
    *,
    drafted_keys: set[tuple] | None = None,
    positions: Sequence[str] = POSITIONS,
) -> tuple[dict[str, InjuryCell], set[tuple], pd.DataFrame]:
    """Per-position injury knobs (on the **drafted cohort**), the setback keys, and the per-season table.

    Returns ``(cells, setback_keys, table)``. The shipped rate is fit on the drafted cohort
    (``drafted_keys`` — the same per-season top-N by projection the season CV uses, Amendment C); the wide
    (all-qualified) rate is kept for the side-by-side. ``setback_keys`` are all qualified ``(player_id,
    season)`` setbacks — removed from the season-CV healthy cohort so the two knobs do not double-count
    games-missed. A position with no injury-report coverage (DST) or too few drafted-cohort seasons defers.
    """
    table = _player_season_injury_table(played, injuries)
    cells: dict[str, InjuryCell] = {}
    for pos in positions:
        q_wide = table[(table["position"].astype("string") == pos) & table["qualified"]]
        if drafted_keys is not None and len(q_wide):
            in_drafted = np.array(
                [(str(pid), int(s)) in drafted_keys for pid, s in zip(q_wide["player_id"], q_wide["season"])],
                dtype=bool,
            )
            q = q_wide[in_drafted]
        else:
            q = q_wide
        n = int(len(q))
        coverage = int(q["on_report"].sum())
        n_setback = int(q["setback"].sum())
        p_out_only = float((q["longest_out_run"] >= SETBACK_MIN_WEEKS).mean()) if n else None
        p_wide = float(q_wide["setback"].mean()) if len(q_wide) else None
        n_wide = int(len(q_wide))
        if coverage == 0:
            reason = "never on the injury report (team defense)" if pos == "DEF" else "no injury-report coverage"
            cells[pos] = InjuryCell(pos, None, None, n, n_setback, coverage, reason, p_out_only, p_wide, n_wide)
            continue
        if n < MIN_INJURY_SEASONS:
            cells[pos] = InjuryCell(
                pos, None, None, n, n_setback, coverage,
                f"only {n} drafted-cohort player-seasons (< {MIN_INJURY_SEASONS})", p_out_only, p_wide, n_wide,
            )
            continue
        p = n_setback / n
        missed = q.loc[q["setback"], "longest_setback"]
        mean_missed = float(missed.mean()) if not missed.empty else None
        cells[pos] = InjuryCell(pos, float(p), mean_missed, n, n_setback, coverage, None, p_out_only, p_wide, n_wide)

    setback_keys = {
        (str(row.player_id), int(row.season))
        for row in table[table["qualified"] & table["setback"]].itertuples()
        if pd.notna(row.player_id) and pd.notna(row.season)
    }
    return cells, setback_keys, table


# =========================================================================== availability & collapse
def expected_availability(
    injury_risk: Mapping[str, tuple[float, float]], season_games: int = SEASON_GAMES
) -> dict[str, float]:
    """``E[availability multiplier]`` per position, closed form — the mean the haircut shifts by.

    ``E[avail] = 1 - p * E[clip(Poisson(sev), 1, G)] / G``, matching
    :func:`draftsim.distributions.sample_availability`. Reported under heuristic and fitted knobs so the
    mean-shift the injury knob induces (Amendment 1 — the double-count the healthy-CV read exposes but
    this ticket does not fix) is legible next to the before/after.

    The Poisson pmf is built by the recurrence ``pmf(k) = pmf(k-1)·sev/k`` (stable, no factorial
    overflow); the tail beyond ``k_max`` all clips to ``G`` and is added as ``G·P(X > k_max)``.
    """
    k_max = 200
    ks = np.arange(0, k_max + 1)
    out: dict[str, float] = {}
    for pos, (p, sev) in injury_risk.items():
        pmf = np.empty(k_max + 1)
        pmf[0] = np.exp(-float(sev))
        for k in range(1, k_max + 1):
            pmf[k] = pmf[k - 1] * float(sev) / k
        e_clip = float(np.sum(pmf * np.clip(ks, 1, season_games)))
        e_clip += season_games * max(0.0, 1.0 - float(np.sum(pmf)))  # tail mass clips to G
        out[pos] = 1.0 - float(p) * e_clip / float(season_games)
    return out


def collapse_row(position: str, season_cv: float, game_cv: float, season_games: int = SEASON_GAMES) -> dict:
    """The ``season_factor_cv`` identity for one position: does single-game noise collapse the factor?

    ``1 + CV_total² = (1 + CV_factor²)(1 + CV_week²/W)``. If ``CV_week²/W >= CV_total²`` the factor floors
    to 0 and the season loses its season-level correlation (:mod:`seasonsim.distributions`). Returns the
    factor CV, the CV the factor+week pair reconstructs, and a ``collapse`` flag.
    """
    factor = float(season_factor_cv(np.array([season_cv]), np.array([game_cv]), season_games)[0])
    reconstructed = float(np.sqrt((1.0 + factor**2) * (1.0 + game_cv**2 / season_games) - 1.0))
    return {
        "position": position,
        "season_cv": season_cv,
        "game_cv": game_cv,
        "factor_cv": factor,
        "reconstructed_total_cv": reconstructed,
        "collapse": factor <= 0.0,
    }


# =========================================================================== assembling the verdicts
def _game_cv_verdict(cell: CvCell | None) -> tuple[float | None, str, int, str | None]:
    """``(value, verdict, n, reason)`` for a **game** CV cell — the primitive spread, gated on ``n`` only."""
    if cell is None or cell.cv is None:
        return None, "heuristic-fallback", 0, "no residuals"
    if cell.n < MIN_CV_N:
        return cell.cv, "heuristic-fallback", cell.n, f"only {cell.n} rows (< {MIN_CV_N})"
    return cell.cv, "fitted", cell.n, None


def _season_cv_verdict(
    cell: CvCell | None, game_cv: float, season_games: int
) -> tuple[float | None, str, int, str | None, float | None]:
    """``(value, verdict, n, reason, factor_cv)`` for a **season** CV cell via the coherence gate.

    Fitted only when the implied season-factor CV lands in ``(0, MAX_SEASON_FACTOR_CV]`` against the
    shipped game CV: at 0 the factor floors (collapse — trap 2's low end); above the cap the "season CV"
    is season-ahead projection error, not outcome spread (the high end the SeasonModel residual hits).
    """
    if cell is None or cell.cv is None:
        return None, "heuristic-fallback", 0, "no residuals", None
    if cell.n < MIN_CV_N:
        return cell.cv, "heuristic-fallback", cell.n, f"only {cell.n} rows (< {MIN_CV_N})", None
    factor = collapse_row(cell.position, cell.cv, game_cv, season_games)["factor_cv"]
    if factor <= 0.0:
        return cell.cv, "heuristic-fallback", cell.n, "season-factor collapse (game noise ≥ season CV)", factor
    if factor > MAX_SEASON_FACTOR_CV:
        return (
            cell.cv, "heuristic-fallback", cell.n,
            f"implied season-factor CV {factor:.2f} > {MAX_SEASON_FACTOR_CV} on the drafted cohort — not "
            "coherent with the game CV (a season-baseline swing this large exceeds what a season factor "
            "carries); wide-cohort CV recorded as an upper bound",
            factor,
        )
    return cell.cv, "fitted", cell.n, None, factor


def build_verdicts(
    position_cv: Mapping[str, CvCell],
    game_cv: Mapping[str, CvCell],
    injury: Mapping[str, InjuryCell],
    *,
    season_games: int = SEASON_GAMES,
    positions: Sequence[str] = POSITIONS,
) -> dict:
    """Finalise every cell to ``value/verdict/n/reason`` and resolve season-factor coherence loudly.

    Game CV is the primitive single-game spread (gated on ``n``). Season CV must be **coherent** with the
    shipped game CV under the sim's own ``season_factor_cv`` identity — both a collapse (factor floors)
    and an over-large factor (projection error) fail it back to the heuristic, never silently
    (ticket #32 trap 2). Returns the artifact-shaped dicts, the shipped knob dicts (fitted where fitted,
    heuristic elsewhere), and the coherence table.
    """
    game_out: dict[str, dict] = {}
    game_ship: dict[str, float] = {}
    for pos in positions:
        value, verdict, n, reason = _game_cv_verdict(game_cv.get(pos))
        game_ship[pos] = value if verdict == "fitted" else HEURISTIC_GAME_CV[pos]
        game_out[pos] = {"value": value, "verdict": verdict, "n": n, "reason": reason}

    coherence_rows: list[dict] = []
    pos_out: dict[str, dict] = {}
    pos_ship: dict[str, float] = {}
    for pos in positions:
        value, verdict, n, reason, factor = _season_cv_verdict(
            position_cv.get(pos), game_ship[pos], season_games
        )
        pos_ship[pos] = value if verdict == "fitted" else HEURISTIC_POSITION_CV[pos]
        pos_out[pos] = {"value": value, "verdict": verdict, "n": n, "reason": reason}
        shipped_factor = collapse_row(pos, pos_ship[pos], game_ship[pos], season_games)["factor_cv"]
        coherence_rows.append(
            {
                "position": pos,
                "fitted_season_cv": position_cv[pos].cv if pos in position_cv else None,
                "game_cv": game_ship[pos],
                "fitted_factor_cv": factor,
                "verdict": verdict,
                "shipped_season_cv": pos_ship[pos],
                "shipped_factor_cv": shipped_factor,
            }
        )

    injury_out: dict[str, dict] = {}
    injury_ship: dict[str, tuple[float, float]] = {}
    for pos in positions:
        cell = injury.get(pos)
        if cell is not None and cell.reason is None and cell.p_setback is not None:
            injury_out[pos] = {
                "p": cell.p_setback, "games": cell.mean_missed,
                "verdict": "fitted", "n": cell.n_qualified, "reason": None,
            }
            injury_ship[pos] = (cell.p_setback, cell.mean_missed or HEURISTIC_INJURY_RISK[pos][1])
        else:
            injury_ship[pos] = HEURISTIC_INJURY_RISK[pos]
            injury_out[pos] = {
                "p": None, "games": None, "verdict": "heuristic-fallback",
                "n": cell.n_qualified if cell else 0, "reason": cell.reason if cell else "no data",
            }

    return {
        "position_cv": pos_out,
        "game_cv": game_out,
        "injury_risk": injury_out,
        "coherence": pd.DataFrame(coherence_rows),
        "shipped_position_cv": pos_ship,
        "shipped_game_cv": game_ship,
        "shipped_injury": injury_ship,
    }


# =========================================================================== the orchestrator
@dataclass(frozen=True)
class FitResult:
    """Everything the report and the artifact need — the whole measured fit in one object."""

    seasons: tuple[int, ...]
    test_seasons: tuple[int, ...]
    position_cv: dict[str, CvCell]
    game_cv: dict[str, CvCell]
    injury: dict[str, InjuryCell]
    position_cv_out: dict[str, dict]
    game_cv_out: dict[str, dict]
    injury_out: dict[str, dict]
    coherence: pd.DataFrame
    robustness: pd.DataFrame  # season-CV verdicts under the top-N-overall cohort (Amendment A cross-check)
    shipped_position_cv: dict[str, float]
    shipped_game_cv: dict[str, float]
    shipped_injury: dict[str, tuple[float, float]]
    avail_heuristic: dict[str, float]
    avail_fitted: dict[str, float]
    injury_table: pd.DataFrame
    n_frame_rows: int
    n_players: int


def _game_predictions(
    weekly: pd.DataFrame, kickdef: pd.DataFrame, scoring: Mapping[str, float], test_seasons: Iterable[int]
) -> pd.DataFrame:
    """Walk-forward out-of-sample weekly predictions for all six positions (skill + K/DEF, one frame)."""
    skill = evaluate(
        WeeklyModel(), weekly, positions=SKILL_POSITIONS, test_seasons=test_seasons
    ).predictions
    kd = evaluate(
        KickDefModel(scoring), kickdef, positions=KICKDEF_POSITIONS, test_seasons=test_seasons
    ).predictions
    return pd.concat([skill, kd], ignore_index=True)


def fit_distributions(
    seasons: Iterable[int],
    scoring: Mapping[str, float],
    *,
    backend: StorageBackend | None = None,
    test_seasons: Iterable[int] = DEFAULT_TEST_SEASONS,
) -> FitResult:
    """Fit all three knobs on the real lake and finalise the verdicts. The single path the eval calls."""
    if not scoring:
        raise ValueError("scoring is empty — pass the league's live scoring_settings dict")
    wanted = sorted({int(s) for s in seasons})
    test = tuple(sorted({int(s) for s in test_seasons}))

    weekly = build_training_frame(wanted, scoring, backend=backend)
    season_frame = season_frame_from_weekly(weekly)
    kickdef = build_kickdef_frame(wanted, scoring, backend=backend)

    season_pred = evaluate_season(
        SeasonModel(), season_frame, positions=POSITIONS, test_seasons=test
    ).predictions
    game_pred = _game_predictions(weekly, kickdef, scoring, test)

    injuries = read_source(
        "nflverse_injuries",
        wanted,
        columns=["gsis_id", "season", "week", "game_type", "report_status", "_season"],
        backend=backend,
    )
    if not injuries.empty:
        injuries = injuries[injuries["game_type"].astype("string") == "REG"].copy()
        injuries["season"] = pd.to_numeric(injuries["_season"], errors="coerce")
    played = weekly.loc[
        weekly["position"].isin(POSITIONS), ["player_id", "gsis_id", "position", "season", "week"]
    ]

    drafted_keys = drafted_cohort_keys(season_pred)
    injury_cells, setback_keys, injury_table = fit_injuries(played, injuries, drafted_keys=drafted_keys)
    position_cv = fit_cv_cells(
        season_pred, grain="season", top_n_by_position=SEASON_COHORT_BY_POSITION,
        setback_keys=setback_keys, wide_floor=WIDE_SEASON_PRED_FLOOR,
    )
    game_cv = fit_cv_cells(game_pred, grain="game", floor=GAME_PRED_FLOOR)

    verdicts = build_verdicts(position_cv, game_cv, injury_cells, season_games=SEASON_GAMES)
    robustness = robustness_overall_cohort(season_pred, setback_keys, verdicts["shipped_game_cv"])

    return FitResult(
        seasons=tuple(wanted),
        test_seasons=test,
        position_cv=position_cv,
        game_cv=game_cv,
        injury=injury_cells,
        position_cv_out=verdicts["position_cv"],
        game_cv_out=verdicts["game_cv"],
        injury_out=verdicts["injury_risk"],
        coherence=verdicts["coherence"],
        robustness=robustness,
        shipped_position_cv=verdicts["shipped_position_cv"],
        shipped_game_cv=verdicts["shipped_game_cv"],
        shipped_injury=verdicts["shipped_injury"],
        avail_heuristic=expected_availability(HEURISTIC_INJURY_RISK),
        avail_fitted=expected_availability(verdicts["shipped_injury"]),
        injury_table=injury_table,
        n_frame_rows=int(len(weekly)),
        n_players=int(weekly["player_id"].nunique()),
    )


def to_artifact(result: FitResult, *, generated: str) -> dict:
    """The committed JSON payload — knobs, verdicts and the diagnostics that justify each fallback."""
    return {
        "model": "distributions",
        "generated": generated,
        "season_games": SEASON_GAMES,
        "test_seasons": list(result.test_seasons),
        "thresholds": {
            "min_cv_n": MIN_CV_N,
            "min_injury_seasons": MIN_INJURY_SEASONS,
            "game_pred_floor": GAME_PRED_FLOOR,
            "season_cohort_by_position": dict(SEASON_COHORT_BY_POSITION),
            "season_cohort_total": SEASON_COHORT_TOTAL,
            "wide_season_pred_floor": WIDE_SEASON_PRED_FLOOR,
            "max_season_factor_cv": MAX_SEASON_FACTOR_CV,
            "setback_min_weeks": SETBACK_MIN_WEEKS,
        },
        "position_cv": {p: _round_cell(c) for p, c in result.position_cv_out.items()},
        "game_cv": {p: _round_cell(c) for p, c in result.game_cv_out.items()},
        "injury_risk": {p: _round_injury(c) for p, c in result.injury_out.items()},
        "diagnostics": {
            "expected_availability_heuristic": {p: round(v, 4) for p, v in result.avail_heuristic.items()},
            "expected_availability_fitted": {p: round(v, 4) for p, v in result.avail_fitted.items()},
            "full_cohort_position_cv": {
                p: (round(c.full_cohort_cv, 4) if c.full_cohort_cv is not None else None)
                for p, c in result.position_cv.items()
            },
            "upper_bound_position_cv": {
                p: (round(c.upper_bound_cv, 4) if c.upper_bound_cv is not None else None)
                for p, c in result.position_cv.items()
            },
            "injury_out_only_rate": {
                p: (round(c.p_out_only, 4) if c.p_out_only is not None else None)
                for p, c in result.injury.items()
            },
            "injury_wide_cohort_rate": {
                p: (round(c.p_wide, 4) if c.p_wide is not None else None)
                for p, c in result.injury.items()
            },
            "season_factor_coherence": [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                for row in result.coherence.to_dict("records")
            ],
            "season_cv_robustness_overall": [
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                for row in result.robustness.to_dict("records")
            ],
        },
    }


def _round_cell(cell: Mapping) -> dict:
    value = cell.get("value")
    return {
        "value": round(float(value), 4) if isinstance(value, (int, float)) and value is not None else None,
        "verdict": cell["verdict"],
        "n": cell["n"],
        "reason": cell["reason"],
    }


def _round_injury(cell: Mapping) -> dict:
    p, g = cell.get("p"), cell.get("games")
    return {
        "p": round(float(p), 4) if p is not None else None,
        "games": round(float(g), 4) if g is not None else None,
        "verdict": cell["verdict"],
        "n": cell["n"],
        "reason": cell["reason"],
    }


# The season-CV cohort is now roster-math (top-N by position), not a value floor; ``model.distributions``
# exports ``SEASON_COHORT_BY_POSITION`` for anyone who needs the denominator.

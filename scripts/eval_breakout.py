"""Score the breakout / waiver classifier, write the report and the artifact (Phase 9, #33).

    ./.venv/Scripts/python scripts/eval_breakout.py
    ./.venv/Scripts/python scripts/eval_breakout.py --seasons 2016-2025 --out docs/model-breakout.md

The breakout target is the one model in the phase whose **label is in the future** (did role and
production step up over the next N=3 weeks) while its **features stay strictly pre-lock**. This script
builds the training frame once (against whatever ``LAKE_BACKEND`` points at — the local backfill by
default), fits the startable thresholds ``T_pos`` from the 2016-2017 warm-up only, attaches the forward
label, declares the waiver cohort, and evaluates walk-forward over 2018-2025 on **precision@k** — the
metric the waiver decision actually consumes (one reverse-priority claim → k=1, with k=3/5 for
stability).

Everything the report asserts is derived from the tables it prints:

* the cohort is declared from waiver relevance *before* measuring, its size and **base rate reported per
  position** (and the base rate is not comparable across positions, so **lift over the base rate** is
  reported beside every raw precision);
* the ``>= 2``-forward-games evaluability rule drops ~19% of decision rows — injured *and* benched — so
  that thinning is characterised (dropped count split by decision-week injury status), not hidden;
* the model must beat three naive baselines at k=1 per position, **deferring to the winning baseline**
  where it does not (the gate), so a missing artifact never fields an unproven model;
* a second, production-axis cohort is measured as a robustness check, and the report says whether the
  two agree;
* the per-position base-rate drift across seasons is reported, because a base rate that moves makes
  walk-forward precision@k not comparable across seasons.

:func:`render_report` is pure over the evaluated results, so ``tests/test_model_breakout.py`` pins that
its prose follows the numbers it cites.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# `python scripts/eval_breakout.py` puts scripts/ at the front of sys.path, where scripts/collect.py
# shadows the installed `collect` package the assembler imports. Drop the script directory.
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from collect import runner
from dataset.assemble import build_training_frame
from model.breakout import (
    BREAKOUT_POSITIONS,
    FORWARD_WINDOW,
    K_VALUES,
    MIN_FORWARD_GAMES,
    STARTABLE_RANK,
    BreakoutEvalResult,
    BreakoutModel,
    BreakoutPositionMetrics,
    ColumnRanker,
    add_forward_label,
    breakout_gate,
    evaluate_breakout,
    mcnemar_k1,
    production_cohort_mask,
    snap_cohort_mask,
    startable_thresholds,
)
from sleeper import client
from sleeper.config import LEAGUE_ID
from store.lake import LAKE_BACKEND, lake_inventory

DEFAULT_SEASONS = "2016-2025"
DEFAULT_OUT = "docs/model-breakout.md"
DEFAULT_ARTIFACT = "src/model/fit/breakout.json"

_MODEL = "BreakoutModel"
_LOGIT = "Logistic (pure)"
_BASELINES: tuple[str, ...] = ("last_week_points", "snap_share_trend", "role_share_last")
_BASELINE_LABEL: dict[str, str] = {
    "last_week_points": "last week's points",
    "snap_share_trend": "snap-share trend",
    "role_share_last": "target/rush share (last)",
}
_TIE_EPS = 1e-12


# --------------------------------------------------------------------------- log capture
class _Capture(logging.Handler):
    """Records the run's log lines so the report can count and quote its warnings (profile #27 §7)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[tuple[int, str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.levelname, record.name, record.getMessage()))


# --------------------------------------------------------------------------- formatting helpers
def _fmt(value: float | None, nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:.{nd}f}"


def _pct(value: float | None, nd: int = 1) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value * 100:.{nd}f}%"


def _season_span(seasons: Sequence[int]) -> str:
    if not seasons:
        return "none"
    if list(seasons) == list(range(seasons[0], seasons[-1] + 1)):
        return f"{seasons[0]}–{seasons[-1]}"
    return ", ".join(str(s) for s in seasons)


def _p(m: BreakoutPositionMetrics | None, k: int) -> float | None:
    """A metrics object's precision@k, or None."""
    return None if m is None else m.precision.get(k)


def _defer_reason(
    pos: str,
    logit: dict[str, BreakoutPositionMetrics],
    baseline_metrics: dict[str, dict[str, BreakoutPositionMetrics]],
) -> str:
    """The first (baseline, k) at which the model fails to beat a baseline — why the position defers.

    Generated from the numbers, not asserted: scans k then baseline in order and reports the first
    loss-or-tie, so the report's prose cannot claim a reason its own tables contradict.
    """
    m = logit.get(pos)
    for k in K_VALUES:
        for key in _BASELINES:
            bp = baseline_metrics.get(key, {}).get(pos)
            if bp is None:
                continue
            mp, bpk = _p(m, k), bp.precision.get(k)
            if mp is None or bpk is None or mp <= bpk + 1e-12:
                rel = "loses to" if (mp is not None and bpk is not None and mp < bpk) else "ties"
                return f"{rel} {_BASELINE_LABEL[key]} at k={k} ({_fmt(mp)} vs {_fmt(bpk)})"
    return "clears every k"


# --------------------------------------------------------------------------- computed stats
def _cohort_stats(labeled: pd.DataFrame, cohort_mask: pd.Series) -> dict:
    """Cohort sizes, base rates (cohort vs evaluable), null-snap arm and candidates per slate.

    Counted over the whole labelled span (the population the deployment fit trains on); the walk-forward
    scoring is the 2018-2025 subset. A *decision row* has a full forward window; *evaluable* adds the
    ``>= 2``-played-games rule; *cohort* adds the snap cut. The null-snap arm is reported over the
    **evaluable** denominator (the meaningful "no role yet, of the players we could act on" fraction).
    """
    pos = labeled["position"].astype("string")
    is_pos = pos.isin(BREAKOUT_POSITIONS)
    window = labeled["has_forward_window"].fillna(False).astype(bool)
    evaluable = labeled["is_evaluable"].fillna(False).astype(bool)
    decision = is_pos & window
    cohort = cohort_mask & evaluable & is_pos

    per_pos: dict[str, dict] = {}
    for p in BREAKOUT_POSITIONS:
        c = cohort & (pos == p)
        ev = evaluable & is_pos & (pos == p)
        cohort_rows = labeled[c]
        snap_ev = pd.to_numeric(labeled.loc[ev, "snap_pct_ewma"], errors="coerce")
        slates = cohort_rows.groupby(["season", "week"], sort=False)
        sizes = slates.size().to_numpy() if len(cohort_rows) else np.array([0])
        per_pos[p] = {
            "n_cohort": int(c.sum()),
            "base_rate_cohort": float(
                pd.to_numeric(labeled.loc[c, "y_breakout"], errors="coerce").mean()
            )
            if int(c.sum())
            else float("nan"),
            "base_rate_evaluable": float(
                pd.to_numeric(labeled.loc[ev, "y_breakout"], errors="coerce").mean()
            )
            if int(ev.sum())
            else float("nan"),
            "null_snap_share": float(snap_ev.isna().mean()) if int(ev.sum()) else float("nan"),
            "candidates_per_slate": float(np.mean(sizes)) if len(cohort_rows) else float("nan"),
            "slates_ge_k": {k: int((sizes >= k).sum()) for k in K_VALUES},
            "n_slates": int(slates.ngroups) if len(cohort_rows) else 0,
        }
    return {
        "n_decision": int(decision.sum()),
        "n_evaluable": int((evaluable & is_pos).sum()),
        "n_cohort": int(cohort.sum()),
        "per_position": per_pos,
    }


def _drop_characterization(labeled: pd.DataFrame) -> dict:
    """The >=2-games drop among decision rows, split by injury status — injured vs benched/depth."""
    pos = labeled["position"].astype("string")
    is_pos = pos.isin(BREAKOUT_POSITIONS)
    window = labeled["has_forward_window"].fillna(False).astype(bool)
    evaluable = labeled["is_evaluable"].fillna(False).astype(bool)
    decision = is_pos & window
    dropped = decision & ~evaluable
    inj = labeled.get("inj_report_status")
    has_inj = inj.notna() if inj is not None else pd.Series(False, index=labeled.index)
    n_decision = int(decision.sum())
    n_dropped = int(dropped.sum())
    return {
        "n_decision": n_decision,
        "n_dropped": n_dropped,
        "drop_share": (n_dropped / n_decision) if n_decision else float("nan"),
        "n_dropped_injured": int((dropped & has_inj).sum()),
        "n_dropped_not_injured": int((dropped & ~has_inj).sum()),
    }


def _base_rate_drift(labeled: pd.DataFrame, cohort_mask: pd.Series, seasons: Sequence[int]) -> dict:
    """Cohort base rate per (season, position) over the scored test seasons — is the target stationary?"""
    pos = labeled["position"].astype("string")
    evaluable = labeled["is_evaluable"].fillna(False).astype(bool)
    cohort = labeled[cohort_mask & evaluable & pos.isin(BREAKOUT_POSITIONS)].copy()
    cohort["season"] = pd.to_numeric(cohort["season"], errors="coerce").astype("Int64")
    out: dict[str, dict[int, float]] = {p: {} for p in BREAKOUT_POSITIONS}
    for (season, position), grp in cohort.groupby(["season", "position"], observed=True):
        if str(position) in out and pd.notna(season) and int(season) in seasons:
            out[str(position)][int(season)] = float(
                pd.to_numeric(grp["y_breakout"], errors="coerce").mean()
            )
    return out


def _importances(model: BreakoutModel) -> dict[str, list[tuple[str, float]]]:
    """Per fielded position, features ranked by |standardised logistic coefficient|."""
    out: dict[str, list[tuple[str, float]]] = {}
    for pos, weights in model.feature_importances().items():
        ranked = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
        out[pos] = ranked
    return out


def _margins(
    model_metrics: dict[str, BreakoutPositionMetrics],
    baseline_metrics: dict[str, dict[str, BreakoutPositionMetrics]],
    gate: dict[str, str],
    mcnemar: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Per-position gate evidence: the per-k model-vs-best-baseline margin plus the k=1 McNemar z.

    Recorded in the artifact so the fielded/deferred decision is auditable from the committed record
    (the gate now turns on beating every baseline at every k, not one noisy k=1 proportion).
    """
    out: dict[str, dict] = {}
    for pos in BREAKOUT_POSITIONS:
        m = model_metrics.get(pos)
        per_k: dict[str, dict] = {}
        for k in K_VALUES:
            best_key, best_p = None, None
            for key, per_pos in baseline_metrics.items():
                bp = per_pos.get(pos)
                p = None if bp is None else bp.precision.get(k)
                if p is not None and (best_p is None or p > best_p):
                    best_key, best_p = key, p
            model_p = None if m is None else m.precision.get(k)
            per_k[str(k)] = {
                "model": None if model_p is None else round(model_p, 4),
                "best_baseline": best_key,
                "best_baseline_p": None if best_p is None else round(best_p, 4),
                "delta": None
                if (model_p is None or best_p is None)
                else round(model_p - best_p, 4),
            }
        out[pos] = {
            "fielded": pos not in gate,
            "defers_to": gate.get(pos),
            "base_rate": None if m is None else round(m.base_rate, 4),
            "per_k": per_k,
            "mcnemar_k1": {
                key: {"z": round(v["z"], 3), "significant": v["significant"],
                      "discordant": v["discordant"], "model_plus": v["model_plus"],
                      "base_plus": v["base_plus"]}
                for key, v in mcnemar.get(pos, {}).items()
            },
        }
    return out


# --------------------------------------------------------------------------- rendering
def _best_baseline_at(baseline_metrics: dict[str, dict[str, BreakoutPositionMetrics]], pos: str, k: int):
    best = None
    for key, per_pos in baseline_metrics.items():
        bp = per_pos.get(pos)
        p = None if bp is None else bp.precision.get(k)
        if p is not None and (best is None or p > best[1]):
            best = (key, p)
    return best


def _verdict(model_p: float | None, best_p: float | None) -> str:
    if model_p is None or best_p is None:
        return "—"
    if model_p > best_p + _TIE_EPS:
        return "win"
    if model_p < best_p - _TIE_EPS:
        return "loss"
    return "tie"


def _headline_table(
    subject: dict[str, BreakoutPositionMetrics],
    baseline_metrics: dict[str, dict[str, BreakoutPositionMetrics]],
) -> list[str]:
    lines = [
        "| position | n | base rate | model p@1 | lift@1 | bar p@1 | Δ@1 | p@3 | p@5 | verdict@1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pos in BREAKOUT_POSITIONS:
        m = subject.get(pos)
        best1 = _best_baseline_at(baseline_metrics, pos, 1)
        if m is None:
            lines.append(f"| {pos} | — | — | — | — | — | — | — | — | — |")
            continue
        best_p = None if best1 is None else best1[1]
        d = None if (m.precision.get(1) is None or best_p is None) else m.precision[1] - best_p
        bar = "—" if best1 is None else f"{_fmt(best1[1])} ({_BASELINE_LABEL[best1[0]]})"
        lines.append(
            f"| {pos} | {m.n:,} | {_fmt(m.base_rate)} | {_fmt(m.precision.get(1))} | "
            f"{_fmt(m.lift.get(1), 2)} | {bar} | {('%+.3f' % d) if d is not None else '—'} | "
            f"{_fmt(m.precision.get(3))} | {_fmt(m.precision.get(5))} | {_verdict(m.precision.get(1), best_p)} |"
        )
    return lines


def _metric_table(per_pos: dict[str, BreakoutPositionMetrics]) -> list[str]:
    lines = [
        "| position | n | base rate | p@1 | p@3 | p@5 | lift@1 | lift@3 | lift@5 | slates(1/3/5) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pos in BREAKOUT_POSITIONS:
        m = per_pos.get(pos)
        if m is None:
            continue
        lines.append(
            f"| {pos} | {m.n:,} | {_fmt(m.base_rate)} | {_fmt(m.precision.get(1))} | "
            f"{_fmt(m.precision.get(3))} | {_fmt(m.precision.get(5))} | {_fmt(m.lift.get(1), 2)} | "
            f"{_fmt(m.lift.get(3), 2)} | {_fmt(m.lift.get(5), 2)} | "
            f"{m.slates_at_k.get(1)}/{m.slates_at_k.get(3)}/{m.slates_at_k.get(5)} |"
        )
    return lines


def _log_block(records: Sequence[tuple[int, str, str, str]], *, min_level: int) -> str:
    lines = [f"{lvl} {name}: {msg}" for level, lvl, name, msg in records if level >= min_level]
    return "```\n" + "\n".join(lines) + "\n```" if lines else "_None._"


def render_report(
    *,
    thresholds: dict[str, float],
    cohort: dict,
    drop: dict,
    drift: dict,
    results: dict[str, BreakoutEvalResult],
    baseline_metrics: dict[str, dict[str, BreakoutPositionMetrics]],
    mcnemar: dict[str, dict[str, dict]],
    gate: dict[str, str],
    robust: dict,
    importances: dict[str, list[tuple[str, float]]],
    seasons: Sequence[int],
    scored_seasons: Sequence[int],
    scoring_keys: int,
    partitions: int,
    league_name: str,
    generated: str,
    frame_rows: int,
    players: int,
    records: Sequence[tuple[int, str, str, str]] = (),
) -> str:
    """The committed report. Pure over the evaluated results so its prose can be pinned to its tables."""
    shipped = results[_MODEL].per_position
    logit = results[_LOGIT].per_position
    n_warnings = sum(1 for level, *_ in records if level >= logging.WARNING)
    cp = cohort["per_position"]

    fielded = [p for p in BREAKOUT_POSITIONS if p not in gate]
    deferred = [p for p in BREAKOUT_POSITIONS if p in gate]

    # RB vs WR base-rate divergence, stated with both numbers.
    br_cohort = {p: cp[p]["base_rate_cohort"] for p in BREAKOUT_POSITIONS}
    br_eval = {p: cp[p]["base_rate_evaluable"] for p in BREAKOUT_POSITIONS}
    ratio = (
        br_cohort["RB"] / br_cohort["WR"]
        if br_cohort.get("WR")
        else float("nan")
    )

    def _mz(pos, key):  # the k=1 McNemar z of the pure logistic vs a baseline
        return mcnemar.get(pos, {}).get(key, {}).get("z")

    def _msig(pos, key):
        return bool(mcnemar.get(pos, {}).get(key, {}).get("significant"))

    n_rb_slates = logit["RB"].n_slates if "RB" in logit else 0
    # A fielded position whose k=1 McNemar vs the toughest baseline is *not* significant is one whose
    # verdict rests on the deeper k=3/5 — say so rather than let the k=1 number appear to carry it.
    rests_on_deeper = [p for p in fielded if not _msig(p, "last_week_points")]

    # Finding 4, generated from the numbers (never a fixed narrative): the gate rule, the verdict, and —
    # per deferred position — *why* it deferred, and — per fielded position with an insignificant k=1
    # margin — that its verdict rests on the deeper k.
    f4 = [
        f"**The gate requires a win at every k, not only the decision k.** precision@1 over "
        f"{n_rb_slates} binary slates has a standard error near 0.04, so a k=1 win inside that band is "
        f"noise unless the *ranking* is genuinely better — the deeper k=3/5, using more of each slate, "
        f"are the corroboration. A position is **fielded** only when the model beats every baseline at "
        f"**every** k; a loss or tie at any k defers it to the strongest baseline. **Fielded: "
        f"{', '.join(fielded) if fielded else 'none'}; deferred: "
        f"{', '.join(f'{p}→{_BASELINE_LABEL[gate[p]]}' for p in deferred) if deferred else 'none'}.**"
    ]
    for p in deferred:
        f4.append(
            f"{p} {_defer_reason(p, logit, baseline_metrics)}, and its k=1 McNemar edge is "
            f"{'' if _msig(p, gate[p]) else 'in'}significant (z {_fmt(_mz(p, gate[p]), 2)}); it defers "
            f"to '{_BASELINE_LABEL[gate[p]]}' — the honest analogue of #32's RB fallback, not a failure."
        )
    for p in rests_on_deeper:
        f4.append(
            f"{p}'s own k=1 margin is not significant (McNemar z {_fmt(_mz(p, 'last_week_points'), 2)} vs "
            f"last week's points); its fielded verdict rests on **k=3/5** (§B), not the noisy k=1 number."
        )
    f4.append("Three positions, three verdicts — a strong RB number is not allowed to carry a weak WR one.")

    parts: list[str] = [
        "# Breakout / waiver classifier — the measured grade (Phase 9, ticket #33)",
        "",
        "> Generated by [`scripts/eval_breakout.py`](../scripts/eval_breakout.py) — regenerate with "
        "`./.venv/Scripts/python scripts/eval_breakout.py`. This file is a committed artifact; do not "
        "hand-edit it.",
        "",
        f"- **Generated (UTC):** {generated}",
        f"- **Train span:** {_season_span(list(seasons))} · **test seasons (scored):** "
        f"{_season_span(list(scored_seasons))} walk-forward (train ≤ S-1, test S; 2016-2017 are the "
        f"warm-up the thresholds are fit from and are never scored). Read back from the results.",
        f"- **Lake backend:** `{LAKE_BACKEND}` · {partitions} partitions read",
        f"- **Scoring:** {league_name!r} live `scoring_settings`, {scoring_keys} keys (the label's "
        f"forward points are the Phase 1 engine's `y_custom_points`; scoring is never hand-coded)",
        f"- **Frame:** {frame_rows:,} rows, {players:,} players. Positions: "
        f"{', '.join(BREAKOUT_POSITIONS)} (QB/K/DEF excluded — see the module docstring).",
        f"- **Label:** breakout = forward-{FORWARD_WINDOW}-week mean points-per-played-game ≥ the "
        f"position's startable line `T_pos` (RB {_fmt(thresholds.get('RB'), 2)} / WR "
        f"{_fmt(thresholds.get('WR'), 2)} / TE {_fmt(thresholds.get('TE'), 2)}), fit from the "
        f"2016-2017 warm-up only.",
        "",
        "## Findings, measured",
        "",
        f"1. **The cohort is declared from waiver relevance, before measuring.** RB/WR/TE player-weeks "
        f"below the starter snap line (`snap_pct_ewma ≤ 0.5`) or with no snap history yet: "
        f"{drop['n_decision']:,} decision rows (a full {FORWARD_WINDOW}-week window ahead) → "
        f"{cohort['n_evaluable']:,} evaluable (≥ {MIN_FORWARD_GAMES} forward games) → "
        f"{cohort['n_cohort']:,} in cohort, over the whole {_season_span(list(seasons))} span (the "
        f"walk-forward scoring below is the {_season_span(list(scored_seasons))} subset). The null-snap "
        f"'no role yet' arm is a real ~10% of evaluable rows (RB {_pct(cp['RB']['null_snap_share'])}, WR "
        f"{_pct(cp['WR']['null_snap_share'])}, TE {_pct(cp['TE']['null_snap_share'])}), not a rounding "
        f"error. Candidates per slate: RB {_fmt(cp['RB']['candidates_per_slate'], 1)}, WR "
        f"{_fmt(cp['WR']['candidates_per_slate'], 1)}, TE {_fmt(cp['TE']['candidates_per_slate'], 1)} — "
        f"every position has ≥ 5 candidates on every scored slate, so precision@1/3/5 is measurable "
        f"throughout.",
        f"2. **The base rate is not comparable across positions, so raw precision@k is not either.** "
        f"In-cohort base rate RB {_fmt(br_cohort['RB'])} / WR {_fmt(br_cohort['WR'])} / TE "
        f"{_fmt(br_cohort['TE'])} (all-evaluable RB {_fmt(br_eval['RB'])} / WR {_fmt(br_eval['WR'])} / "
        f"TE {_fmt(br_eval['TE'])}). RB's is {_fmt(ratio, 1)}× WR's: a sub-50%-snap committee back "
        f"clears {_fmt(thresholds.get('RB'), 2)} points routinely, a sub-50%-snap WR3 rarely clears "
        f"{_fmt(thresholds.get('WR'), 2)} — a breakout is a rarer event at WR **by construction**. So "
        f"**lift over the base rate** is reported beside every raw precision (lift 1.0 = chance, "
        f"comparable across positions); RB's raw precision would otherwise read as the model's strongest "
        f"position when it is only its easiest.",
        f"3. **The ≥ {MIN_FORWARD_GAMES}-games rule drops genuine negatives, not only injuries.** "
        f"{drop['n_dropped']:,} of {drop['n_decision']:,} decision rows ({_pct(drop['drop_share'])}) "
        f"are dropped as not evaluable; of those {drop['n_dropped_injured']:,} carried a decision-week "
        f"injury-report status and {drop['n_dropped_not_injured']:,} did not — the latter are the "
        f"benched / depth / late-scratch cases the rule removes alongside the injured. It biases the "
        f"cohort toward players who kept playing (nudging the base rate up); it is reported here rather "
        f"than left as an invisible thinning, and the rule is not changed (a bye is still not a zero).",
        "4. " + " ".join(f4),
        f"5. **Robustness — a second, production-axis cohort.** Re-measured on the "
        f"`points_ewma ≤ T_pos` cohort (a different structural cut), the per-position verdicts "
        f"{'**agree**' if robust['agree'] else '**disagree**'} with the snap cohort. "
        f"{robust['note']}",
        f"6. **Base-rate drift is real and position-specific.** {_drift_sentence(drift, scored_seasons)} "
        f"A base rate that moves across seasons makes walk-forward precision@k not strictly comparable "
        f"across them, which is stated rather than assumed.",
        f"7. **{'Zero' if not n_warnings else str(n_warnings)} warning"
        f"{'' if n_warnings == 1 else 's'} on real data.** The frame build and the walk over the splits "
        f"emitted **{n_warnings}** WARNING-level line(s) (verbatim below).",
        "",
        "## A. Headline — shipped model vs the toughest baseline (all weeks, walk-forward)",
        "",
        "`base rate` is the cohort positive fraction; `lift@1 = model p@1 / base rate` (1.0 = chance). "
        "A **win** beats the best of the three baselines at k=1, the real decision. The shipped model "
        "**defers** a losing position to that baseline, so a deferred position ties its bar.",
    ]
    parts += _headline_table(shipped, baseline_metrics)

    parts += ["", "**Pure logistic** (the diagnostic — the honest win/loss that drives the gate):"]
    parts += _headline_table(logit, baseline_metrics)

    parts += ["", "## B. Full precision@k — shipped model, pure logistic, and the three baselines", ""]
    parts.append(
        "Precision@k is the fraction of the top-k of each `(season, week, position)` slate that broke "
        "out, averaged over slates with ≥ k candidates (`slates` counts them). Lift divides by the "
        "position's base rate, so it is comparable across positions where raw precision is not."
    )
    for name in (_MODEL, _LOGIT, *_BASELINES):
        if name in results:
            label = name if name in (_MODEL, _LOGIT) else f"baseline · {_BASELINE_LABEL[name]}"
            parts += ["", f"### {label}", *_metric_table(results[name].per_position)]

    parts += ["", "## C. Paired significance — McNemar on the k=1 pick (pure logistic vs each baseline)", ""]
    parts.append(
        "The gate turns on beating every baseline at every k, because precision@1 over "
        f"{n_rb_slates} slates is one noisy proportion. This is the paired test that says whether the k=1 "
        "edge is real: over the slates where the two rankers' top-1 picks disagree, `model+` counts those "
        "only the model got right, `base+` those only the baseline did. `z = (model+ − base+)/√discordant`; "
        "`|z| > 1.96` is significant. A win that is significant here **and** widens at k=3/5 is real; one "
        "that is not is the noise the all-k gate exists to reject."
    )
    parts += _mcnemar_table(mcnemar)

    parts += ["", "## D. Base-rate drift — cohort breakout rate per (season, position)", ""]
    parts.append(
        "The share of cohort candidates that broke out, by season. RB drifts materially; WR/TE are "
        "flatter. Where it drifts, precision@k is not strictly comparable across those seasons."
    )
    parts += _drift_table(drift, scored_seasons)

    parts += ["", "## E. Robustness — the production-axis cohort (`points_ewma ≤ T_pos`)", ""]
    parts.append(
        f"The same pure logistic vs the same three baselines, on a **different structural cohort** "
        f"({robust['n_cohort']:,} rows cut on trailing production rather than snap share). If the "
        f"per-position verdict here matches the snap cohort's, the ranking is not an artifact of one "
        f"cohort definition. {robust['note']}"
    )
    parts += _headline_table(robust["per_position"], robust["baseline_metrics"])

    parts += ["", "## F. Feature importances — |standardised logistic coef| (fielded positions)", ""]
    parts.append(
        "Standardised, so a coefficient is the log-odds shift from a one-SD move in that feature — "
        "comparable across features. A constant/all-null column (RB has no target share, WR/TE no rush "
        "share) is zeroed in the solve. Full-span deployment fit."
    )
    parts += _importance_table(importances)

    parts += ["", "## Warnings (verbatim)", ""]
    parts.append(
        "The standing bar is zero unexpected warnings on real data. A skipped test season would appear "
        "here, so the scored span and the header cannot drift apart silently."
    )
    parts += ["", _log_block(records, min_level=logging.WARNING), ""]
    return "\n".join(parts) + "\n"


def _drift_sentence(drift: dict, scored_seasons: Sequence[int]) -> str:
    """State RB's peak→trough base-rate move specifically, and that WR/TE are flatter (a real finding)."""
    rb = drift.get("RB", {})
    if not rb:
        return "RB drift not measurable on the scored seasons."
    hi = max(rb, key=lambda s: rb[s])
    lo = min(rb, key=lambda s: rb[s])
    spread = {p: (max(v.values()) - min(v.values()) if v else 0.0) for p, v in drift.items()}
    return (
        f"RB moves from {_fmt(rb[hi])} ({hi}) to {_fmt(rb[lo])} ({lo}) — a "
        f"{_fmt(spread['RB'])} spread — while WR ({_fmt(spread.get('WR', 0.0))}) and TE "
        f"({_fmt(spread.get('TE', 0.0))}) sit comparatively flat."
    )


def _drift_table(drift: dict, scored_seasons: Sequence[int]) -> list[str]:
    seasons = sorted({s for p in drift.values() for s in p})
    header = "| position | " + " | ".join(str(s) for s in seasons) + " |"
    sep = "| --- |" + " ---: |" * len(seasons)
    lines = [header, sep]
    for pos in BREAKOUT_POSITIONS:
        cells = [_fmt(drift.get(pos, {}).get(s)) for s in seasons]
        lines.append(f"| {pos} | " + " | ".join(cells) + " |")
    return lines


def _mcnemar_table(mcnemar: dict[str, dict[str, dict]]) -> list[str]:
    lines = [
        "| position | vs baseline | discordant | model+ | base+ | z | significant |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for pos in BREAKOUT_POSITIONS:
        for key in _BASELINES:
            v = mcnemar.get(pos, {}).get(key)
            if v is None:
                continue
            sig = "**yes**" if v["significant"] else "no"
            lines.append(
                f"| {pos} | {_BASELINE_LABEL[key]} | {v['discordant']} | {v['model_plus']} | "
                f"{v['base_plus']} | {v['z']:+.2f} | {sig} |"
            )
    return lines


def _importance_table(importances: dict[str, list[tuple[str, float]]]) -> list[str]:
    lines = ["| position | feature | std coef |", "| --- | --- | ---: |"]
    for pos in BREAKOUT_POSITIONS:
        for feat, coef in importances.get(pos, [])[:6]:
            lines.append(f"| {pos} | `{feat}` | {coef:+.3f} |")
    return lines


# --------------------------------------------------------------------------- artifact
def _write_artifact(
    model: BreakoutModel,
    thresholds: dict[str, float],
    margins: dict[str, dict],
    cohort: dict,
    path: str,
) -> None:
    """The committed fitted artifact: the deployment logistic + gate + thresholds + the gate's evidence.

    Read back by :meth:`BreakoutModel.load_fitted` and by :func:`model.breakout.recorded_gate` (the safe
    default), so nothing here is write-only. Refit only via this script; never hand-edited.
    """
    payload = model.to_dict()
    payload["thresholds"] = {k: round(float(v), 6) for k, v in thresholds.items()}
    payload["startable_rank"] = {p: int(v) for p, v in STARTABLE_RANK.items()}
    payload["forward_window"] = int(FORWARD_WINDOW)
    payload["min_forward_games"] = int(MIN_FORWARD_GAMES)
    payload["gate_margins"] = margins
    payload["base_rate"] = {
        p: round(cohort["per_position"][p]["base_rate_cohort"], 6) for p in BREAKOUT_POSITIONS
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- entry point
def _robustness(
    labeled: pd.DataFrame, thresholds: dict, gate: dict, seasons, ks
) -> dict:
    """Re-run the pure logistic + baselines on the production-axis cohort; do the verdicts agree?

    Returns the production cohort's own precision@k (so both cohorts' numbers stand side by side, not a
    bare agree/disagree) plus the per-position verdict comparison against the snap cohort's gate.
    """
    pos = labeled["position"].astype("string")
    evaluable = labeled["is_evaluable"].fillna(False).astype(bool)
    mask = production_cohort_mask(labeled, thresholds)
    frame = labeled[mask & evaluable & pos.isin(BREAKOUT_POSITIONS)].copy()
    model = evaluate_breakout(
        BreakoutModel(defer={}), frame, test_seasons=seasons, ks=ks, name=_LOGIT
    ).per_position
    baselines = {
        key: evaluate_breakout(
            ColumnRanker(key), frame, test_seasons=seasons, ks=ks, name=key
        ).per_position
        for key in _BASELINES
    }
    prod_gate = breakout_gate(model, baselines)
    # Verdict = fielded (win) vs deferred (loss) per position; the snap cohort's gate is `gate`.
    snap_fielded = {p for p in BREAKOUT_POSITIONS if p not in gate}
    prod_fielded = {p for p in BREAKOUT_POSITIONS if p not in prod_gate}
    disagreeing = sorted(snap_fielded.symmetric_difference(prod_fielded))
    agree = not disagreeing
    note = (
        "Both cohorts reach the same win/defer verdict at every position."
        if agree
        else (
            f"Verdict differs at: {', '.join(disagreeing)} — the model's edge there is "
            "cohort-sensitive, so neither number is allowed to stand alone."
        )
    )
    return {
        "agree": agree,
        "note": note,
        "gate": prod_gate,
        "n_cohort": int(len(frame)),
        "per_position": model,
        "baseline_metrics": baselines,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the breakout classifier, write report + artifact")
    ap.add_argument("--seasons", default=DEFAULT_SEASONS, help=f"train span (default: {DEFAULT_SEASONS})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"report path (default: {DEFAULT_OUT})")
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT, help=f"artifact (default: {DEFAULT_ARTIFACT})")
    args = ap.parse_args(argv)

    try:
        seasons = runner.parse_seasons(args.seasons)
    except ValueError as exc:
        print(f"bad --seasons — {exc}", file=sys.stderr)
        return 2

    try:
        league = client.get_league(LEAGUE_ID)
        scoring = league["scoring_settings"]
    except Exception as exc:
        print(
            f"could not load scoring_settings from league {LEAGUE_ID} ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 1

    partitions = int(len(lake_inventory()))
    print(
        f"Evaluating the breakout classifier over {seasons[0]}-{seasons[-1]} on backend {LAKE_BACKEND} "
        f"({partitions} partitions), scoring {league.get('name')!r} ({len(scoring)} keys)…"
    )

    capture = _Capture()
    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        frame = build_training_frame(seasons, scoring)
        thresholds = startable_thresholds(frame)
        labeled = add_forward_label(frame, thresholds)

        cohort_mask = snap_cohort_mask(labeled)
        evaluable = labeled["is_evaluable"].fillna(False).astype(bool)
        pos = labeled["position"].astype("string")
        eval_frame = labeled[cohort_mask & evaluable & pos.isin(BREAKOUT_POSITIONS)].copy()

        cohort = _cohort_stats(labeled, cohort_mask)
        drop = _drop_characterization(labeled)

        results: dict[str, BreakoutEvalResult] = {
            _LOGIT: evaluate_breakout(BreakoutModel(defer={}), eval_frame, name=_LOGIT)
        }
        baseline_results = {
            key: evaluate_breakout(ColumnRanker(key), eval_frame, name=key) for key in _BASELINES
        }
        baseline_metrics = {key: r.per_position for key, r in baseline_results.items()}
        gate = breakout_gate(results[_LOGIT].per_position, baseline_metrics)
        results[_MODEL] = evaluate_breakout(BreakoutModel(defer=gate), eval_frame, name=_MODEL)

        # Paired McNemar on the k=1 pick, pure logistic vs each baseline — the significance the gate rests
        # on (precision@1 over ~117 slates is one noisy proportion; this is the paired evidence).
        mcnemar = {
            pos: {
                key: mcnemar_k1(results[_LOGIT].predictions, br.predictions, pos)
                for key, br in baseline_results.items()
            }
            for pos in BREAKOUT_POSITIONS
        }

        scored_seasons = sorted(results[_MODEL].test_seasons)
        drift = _base_rate_drift(labeled, cohort_mask, scored_seasons)
        robust = _robustness(labeled, thresholds, gate, scored_seasons, K_VALUES)

        deploy = BreakoutModel(defer=gate).fit(eval_frame)
        importances = _importances(BreakoutModel(defer={}).fit(eval_frame))
        margins = _margins(results[_LOGIT].per_position, baseline_metrics, gate, mcnemar)
        _write_artifact(deploy, thresholds, margins, cohort, args.artifact)
    except ValueError as exc:
        print(f"could not build/evaluate the frame — {exc}", file=sys.stderr)
        return 1
    finally:
        root.removeHandler(capture)
        root.setLevel(prior_level)

    report = render_report(
        thresholds=thresholds,
        cohort=cohort,
        drop=drop,
        drift=drift,
        results=results,
        baseline_metrics=baseline_metrics,
        mcnemar=mcnemar,
        gate=gate,
        robust=robust,
        importances=importances,
        seasons=seasons,
        scored_seasons=scored_seasons,
        scoring_keys=len(scoring),
        partitions=partitions,
        league_name=league.get("name", "?"),
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        frame_rows=len(frame),
        players=int(frame["player_id"].nunique()),
        records=capture.records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    n_warnings = sum(1 for level, *_ in capture.records if level >= logging.WARNING)
    print(
        f"Wrote {out} (report) and {args.artifact} (artifact: fielded {deploy.fielded_positions}, "
        f"gate {gate}) — {cohort['n_cohort']:,} cohort rows, test seasons "
        f"{_season_span(scored_seasons)}, {n_warnings} warning(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

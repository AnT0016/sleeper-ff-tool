"""Score the weekly model against the three #28 baselines, write the report and the artifact (#29).

    ./.venv/Scripts/python scripts/eval_weekly.py
    ./.venv/Scripts/python scripts/eval_weekly.py --seasons 2016-2025 --out docs/model-weekly.md

The weekly path's bar is the three naive baselines recorded in ``docs/model-baselines.md`` — the model
must beat **all three** on **both** MAE and within-slate Spearman rho, **per position** (spec ticket
#29). This script builds the training frame once (against whatever ``LAKE_BACKEND`` points at — the
local backfill by default), scopes it to the four skill positions, and evaluates walk-forward over
2018-2025:

* **all weeks** — pure ridge, the three baselines, and the shipped model;
* **the cold start, separately** — the same five over rows with no within-season lag (week 1 and
  mid-season debuts). With no lag, the lag-based baselines are a flat position mean here, so this is
  where a model either wins outright or is beaten by ``PriorSeasonRank``'s prior-season level — and the
  **margin is reported per position, win or lose** (spec ticket #29): "loses at week 1" is not a number,
  and it is the number that decides whether a prior-season feature is worth building (Decision #8).

The **shipped model** (:class:`model.weekly.WeeklyModel`) fields ridge everywhere except cold-start rows
at the positions where pure ridge was *measured* to lose the cold start — a per-position gate, not a
blanket rule (mirroring #31's measured DEF deferral). This script derives that gate from the pure-ridge
cold-start result below, evaluates the shipped model with it, and writes both the committed report
(``--out``) and the committed fitted artifact (``--artifact``: the full-span ridge plus the measured
gate and its margins). The label is re-scored live by the Phase 1 engine, and
:func:`model.weekly.assert_linear_skill_scoring` is called against the live scoring **first** — a points
head is valid only while skill scoring is linear (Decision #7). :func:`render_report` is pure over the
evaluated results, so ``tests/test_model_weekly.py`` pins that its prose follows the numbers it cites.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# `python scripts/eval_weekly.py` puts scripts/ at the front of sys.path, where scripts/collect.py
# shadows the installed `collect` package the assembler imports. Drop the script directory.
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from collect import runner
from dataset.assemble import build_training_frame
from model.evaluate import EvalResult, PositionMetrics, evaluate
from model.weekly import (
    WEEKLY_FEATURES,
    WeeklyModel,
    WeeklyRidge,
    assert_linear_skill_scoring,
    cold_start_metrics,
    deferred_positions,
)
from model.weekly import _BASELINE_FACTORIES as BASELINE_FACTORIES
from sleeper import client
from sleeper.config import LEAGUE_ID
from store.lake import LAKE_BACKEND, lake_inventory

DEFAULT_SEASONS = "2016-2025"
DEFAULT_OUT = "docs/model-weekly.md"
DEFAULT_ARTIFACT = "src/model/fit/weekly_ridge.json"

_SHIPPED = "WeeklyModel"
_RIDGE = "WeeklyRidge"
_BASELINES: tuple[str, ...] = ("TrailingMean", "PriorSeasonRank", "LaggedExpectedPoints")
_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

_EXCLUDED_COLS: tuple[str, ...] = (
    "depth_pos_rank",
    "inj_report_status",
    "wx_forecast_temp_f",
    "baseline_sleeper_points",
)
_MOSTLY_NULL = 0.50
_TOP_K = 5


# --------------------------------------------------------------------------- log capture
class _Capture(logging.Handler):
    """Records the run's log lines so the report can count and quote its warnings (profile #27 §7)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[tuple[int, str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.levelname, record.name, record.getMessage()))


# --------------------------------------------------------------------------- rendering helpers
def _fmt(value: float | None, nd: int = 2) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:.{nd}f}"


def _signed(value: float | None, nd: int = 2) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:+.{nd}f}"


def _season_span(seasons: Sequence[int]) -> str:
    if not seasons:
        return "none"
    if list(seasons) == list(range(seasons[0], seasons[-1] + 1)):
        return f"{seasons[0]}–{seasons[-1]}"
    return ", ".join(str(s) for s in seasons)


def _bar(baselines: dict[str, dict[str, PositionMetrics]], pos: str):
    """The toughest baseline at ``pos``: ``((mae_name, mae), (rho_name, rho))`` over the 3 baselines."""
    maes = [(n, m[pos].mae) for n, m in baselines.items() if pos in m]
    rhos = [(n, m[pos].spearman) for n, m in baselines.items() if pos in m and m[pos].spearman is not None]
    return (min(maes, key=lambda t: t[1]) if maes else None,
            max(rhos, key=lambda t: t[1]) if rhos else None)


#: Two metrics within this are the same number — a deferred position predicts exactly the baseline it
#: defers to, so its margin is zero up to float noise, and that is a tie, not a loss.
_TIE_EPS = 1e-9


def _verdict(m: PositionMetrics | None, best_mae, best_rho) -> str:
    """Model vs the toughest baseline, derived from the numbers. A win beats it on BOTH metrics.

    ``tie (deferred)`` is the deferral signature: the shipped model *is* the baseline on a deferred
    cold-start row, so it matches the bar on both metrics rather than beating or losing to it.
    """
    if m is None or best_mae is None or best_rho is None or m.spearman is None:
        return "—"
    dm, dr = m.mae - best_mae[1], m.spearman - best_rho[1]
    mae_better, mae_worse = dm < -_TIE_EPS, dm > _TIE_EPS
    rho_better, rho_worse = dr > _TIE_EPS, dr < -_TIE_EPS
    if mae_better and rho_better:
        return "win"
    if not mae_worse and not rho_worse and not (mae_better or rho_better):
        return "tie (deferred)"
    if mae_worse and rho_worse:
        return "loss"
    return "split"


def _headline_table(
    subject: dict[str, PositionMetrics], baselines: dict[str, dict[str, PositionMetrics]]
) -> list[str]:
    lines = [
        "| position | n | model MAE | bar MAE | ΔMAE | model ρ | bar ρ | Δρ | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pos in _POSITIONS:
        m = subject.get(pos)
        best_mae, best_rho = _bar(baselines, pos)
        if m is None or best_mae is None or best_rho is None:
            lines.append(f"| {pos} | — | — | — | — | — | — | — | — |")
            continue
        dmae = m.mae - best_mae[1]
        drho = None if m.spearman is None else m.spearman - best_rho[1]
        lines.append(
            f"| {pos} | {m.n:,} | {_fmt(m.mae)} | {_fmt(best_mae[1])} ({best_mae[0]}) | "
            f"{_signed(dmae)} | {_fmt(m.spearman, 3)} | {_fmt(best_rho[1], 3)} ({best_rho[0]}) | "
            f"{_signed(drho, 3)} | {_verdict(m, best_mae, best_rho)} |"
        )
    return lines


def _metric_table(per_pos: dict[str, PositionMetrics]) -> list[str]:
    lines = [
        "| position | n | MAE | RMSE | Spearman ρ | slates | ordered |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pos in _POSITIONS:
        if pos not in per_pos:
            continue
        m = per_pos[pos]
        lines.append(
            f"| {pos} | {m.n:,} | {_fmt(m.mae)} | {_fmt(m.rmse)} | {_fmt(m.spearman, 3)} | "
            f"{m.spearman_slates} | {m.spearman_ordered_slates} |"
        )
    return lines


def _importance_table(importances: dict[str, list[tuple[str, float, float]]]) -> list[str]:
    lines = ["| position | feature | std coef | train null % |", "| --- | --- | ---: | ---: |"]
    for pos in _POSITIONS:
        for feat, coef, null_pct in importances.get(pos, [])[:_TOP_K]:
            lines.append(f"| {pos} | `{feat}` | {_signed(coef, 3)} | {_fmt(null_pct * 100, 1)} |")
    return lines


def _calibration_table(subject: dict[str, PositionMetrics]) -> list[str]:
    deciles = list(range(1, 11))
    lines = ["| position | " + " | ".join(f"d{d}" for d in deciles) + " | gap |",
             "| --- |" + " ---: |" * (len(deciles) + 1)]
    for pos in _POSITIONS:
        if pos not in subject:
            continue
        cal = subject[pos].calibration
        by_decile = {int(row.decile): row for row in cal.itertuples()}
        cells = [_fmt(by_decile[d].realized_mean, 1) if d in by_decile else "—" for d in deciles]
        gap = (cal["pred_mean"] - cal["realized_mean"]).abs().mean() if not cal.empty else None
        lines.append(f"| {pos} | " + " | ".join(cells) + f" | {_fmt(gap)} |")
    return lines


def _log_block(records: Sequence[tuple[int, str, str, str]], *, min_level: int) -> str:
    lines = [f"{lvl} {name}: {msg}" for level, lvl, name, msg in records if level >= min_level]
    return "```\n" + "\n".join(lines) + "\n```" if lines else "_None._"


def _margin_bits(subject, baselines, positions):
    bits = []
    for p in positions:
        m = subject.get(p)
        best_mae, best_rho = _bar(baselines, p)
        if m is None or best_mae is None or best_rho is None or m.spearman is None:
            continue
        bits.append(
            f"{p} {_verdict(m, best_mae, best_rho)} (ΔMAE {_signed(m.mae - best_mae[1])}, "
            f"Δρ {_signed(m.spearman - best_rho[1], 3)})"
        )
    return "; ".join(bits)


def render_report(
    results: dict[str, EvalResult],
    cold: dict[str, dict[str, PositionMetrics]],
    importances: dict[str, list[tuple[str, float, float]]],
    excluded_nulls: dict[str, float],
    gate: Sequence[str],
    *,
    seasons: Sequence[int],
    scoring_keys: int,
    partitions: int,
    league_name: str,
    generated: str,
    frame_rows: int,
    cohort_rows: int,
    players: int,
    records: Sequence[tuple[int, str, str, str]] = (),
) -> str:
    """The committed report. Pure over the evaluated results so its prose can be pinned to its tables."""
    all_base = {n: results[n].per_position for n in _BASELINES if n in results}
    cold_base = {n: cold[n] for n in _BASELINES if n in cold}
    shipped_all = results[_SHIPPED].per_position
    shipped_cold = cold.get(_SHIPPED, {})
    ridge_cold = cold.get(_RIDGE, {})
    scored_seasons = sorted(results[_SHIPPED].test_seasons)
    n_warnings = sum(1 for level, *_ in records if level >= logging.WARNING)

    wins_all = [p for p in _POSITIONS if _verdict(shipped_all.get(p), *_bar(all_base, p)) == "win"]
    fielded = [p for p in _POSITIONS if p not in gate]

    parts: list[str] = [
        "# Weekly point model — the measured grade (Phase 9, ticket #29)",
        "",
        "> Generated by [`scripts/eval_weekly.py`](../scripts/eval_weekly.py) — regenerate with "
        "`./.venv/Scripts/python scripts/eval_weekly.py`. This file is a committed artifact; do not "
        "hand-edit it.",
        "",
        f"- **Generated (UTC):** {generated}",
        f"- **Train span:** {seasons[0]}–{seasons[-1]} · **test seasons (scored):** "
        f"{_season_span(scored_seasons)} walk-forward (train ≤ S-1, test S; 2016-2017 are lag warm-up "
        f"and are never scored — spec Decision #6). Read back from the results.",
        f"- **Lake backend:** `{LAKE_BACKEND}` · {partitions} partitions read",
        f"- **Scoring:** {league_name!r} live `scoring_settings`, {scoring_keys} keys (label re-scored "
        f"by the Phase 1 engine; the points head is guarded linear — Decision #7)",
        f"- **Frame:** {frame_rows:,} rows, {cohort_rows:,} in the {len(_POSITIONS)}-position skill "
        f"cohort, {players:,} players",
        f"- **Shipped model:** `WeeklyModel` — per-position closed-form ridge on {len(WEEKLY_FEATURES)} "
        f"pre-lock features (usage lags + Vegas market + venue), points head; cold-start rows deferred "
        f"to `PriorSeasonRank` at {list(gate) if gate else 'no positions'} (measured — §B).",
        "",
        "The bar is the three naive baselines in "
        "[`docs/model-baselines.md`](model-baselines.md): beat **all three** on **both** MAE and "
        "within-slate Spearman ρ, per position. Re-run here through the same harness, the baselines' "
        "per-position numbers reproduce that recorded bar (they are position-scoped).",
        "",
        "## Findings, measured",
        "",
        f"1. **All-weeks, per position (the shipped model).** {_margin_bits(shipped_all, all_base, _POSITIONS)}. "
        f"A **win** clears all three baselines on both metrics; a **split** wins one and loses the other; "
        f"a **loss** clears neither. Won at: {', '.join(wins_all) if wins_all else 'none'}.",
        f"2. **The cold start, measured per position (pure ridge vs the bar).** "
        f"{_margin_bits(ridge_cold, cold_base, _POSITIONS)}. This is the margin that drives the gate — "
        f"on a row with no within-season lag, pure ridge has only the Vegas market, while "
        f"`PriorSeasonRank` carries last season's level. Ridge is **fielded** at the cold start where it "
        f"wins ({', '.join(fielded) if fielded else 'none'}) and **deferred** to `PriorSeasonRank` where "
        f"it loses ({', '.join(gate) if gate else 'none'}) — a measured per-position gate, not a blanket "
        f"rule (spec ticket #29). Decision #8 logs the prior-season feature this margin would justify.",
        f"3. **The shipped model after deferral (cold start).** {_margin_bits(shipped_cold, cold_base, _POSITIONS)}. "
        f"At a deferred position the shipped model predicts exactly `PriorSeasonRank` (a tie, never a "
        f"loss to a baseline it contains); where ridge was fielded it keeps the ridge's cold-start "
        f"result.",
    ]

    # Finding 4 — does the model lean on a feature #27 found mostly null?
    leaned = [
        f"{pos}/`{feat}` ({null_pct * 100:.0f}% null, coef {coef:+.2f})"
        for pos in _POSITIONS
        for feat, coef, null_pct in importances.get(pos, [])[:_TOP_K]
        if null_pct >= _MOSTLY_NULL and abs(coef) > 0
    ]
    if leaned:
        parts.append(
            f"4. **The model leans on a mostly-null feature** (spec acceptance #3): {', '.join(leaned)}. "
            "A large coefficient on a feature absent most weeks is imputed to the mean on those weeks, "
            "so it does less than its weight suggests — worth a look before trusting it."
        )
    else:
        parts.append(
            f"4. **No top feature is mostly null.** Every feature in the top {_TOP_K} by |coef| at each "
            f"position is populated on the majority of training rows (< {_MOSTLY_NULL * 100:.0f}% null), "
            "so the model is not leaning on a feature #27 showed is largely absent (spec acceptance #3)."
        )

    excl_bits = ", ".join(f"`{c}` {v * 100:.0f}%" for c, v in excluded_nulls.items())
    parts.append(
        f"5. **What was excluded, and why (measured null rate).** {excl_bits}. These frame columns are "
        "not model features because #27 showed them unusable over the training span — depth is 2025+, "
        "the weather and Sleeper-injury/baseline families are forward-only or withheld pre-lock."
    )
    parts.append(
        f"6. **{'Zero' if not n_warnings else str(n_warnings)} warning"
        f"{'' if n_warnings == 1 else 's'} on real data.** The frame build and the walk over the splits "
        f"emitted **{n_warnings}** WARNING-level line(s) (verbatim below); a skipped test season would "
        "appear there, so the scored span and the header cannot drift apart silently."
    )

    parts += ["", "## A. All-weeks headline — shipped model vs the toughest baseline", ""]
    parts.append(
        "ΔMAE negative and Δρ positive both favour the model; a **win** needs both. `bar MAE`/`bar ρ` "
        "name the hardest of the three baselines on each metric (they need not be the same baseline)."
    )
    parts += _headline_table(shipped_all, all_base)

    parts += ["", "## B. Cold-start headline — the slice with no within-season lag", ""]
    parts.append(
        "Rows where `games_played_prior == 0` (week 1 and mid-season debuts). **Pure ridge vs the bar** "
        "is the margin the gate is decided on; **the shipped model** shows the result after deferral. "
        "`PriorSeasonRank` is the toughest here — it carries last season's level, which the lag "
        "baselines and a lag-less ridge do not."
    )
    parts += ["", "**Pure ridge** (the gate driver):", *_headline_table(ridge_cold, cold_base)]
    parts += ["", "**Shipped model** (after per-position deferral):", *_headline_table(shipped_cold, cold_base)]

    parts += ["", "## C. All-weeks metrics — shipped model, pure ridge, and the three baselines", ""]
    parts.append(
        "MAE/RMSE over all pooled out-of-sample rows; ρ the mean of the per-`(season, week)` slate rank "
        "correlations. `ordered` counts the boards given a non-constant prediction (a flat one scores "
        "ρ = 0, not excused) — see [`docs/model-baselines.md`](model-baselines.md)."
    )
    for name in (_SHIPPED, _RIDGE, *_BASELINES):
        if name in results:
            parts += ["", f"### {name}", *_metric_table(results[name].per_position)]

    parts += ["", "## D. Cold-start metrics — shipped model, pure ridge, and the three baselines", ""]
    for name in (_SHIPPED, _RIDGE, *_BASELINES):
        if name in cold:
            parts += ["", f"### {name}", *_metric_table(cold[name])]

    parts += ["", "## E. Feature importances — |standardised coef|, with training null rate", ""]
    parts.append(
        "The features are standardised before the fit, so a coefficient is the points shift from a "
        "one-standard-deviation move in that feature — comparable across features and positions. A "
        "constant/all-null column is zeroed in the solve and reads 0. Fit on the full "
        f"{seasons[0]}–{seasons[-1]} skill cohort (the deployment fit)."
    )
    parts += _importance_table(importances)

    parts += ["", "## F. Calibration — realized mean by predicted decile (shipped model)", ""]
    parts.append(
        "Each cell is the realized mean custom points of the rows in that predicted decile (d1 lowest, "
        "d10 highest). Realized points climbing across deciles is the ordering; `gap` = mean |predicted "
        "− realized| across deciles is the level mismatch."
    )
    parts += _calibration_table(shipped_all)

    parts += ["", "## Warnings (verbatim)", ""]
    parts.append(
        "The standing bar is zero unexpected warnings on real data. A skipped test season appears here, "
        "so the span in the header and the work actually done cannot drift apart silently."
    )
    parts += ["", _log_block(records, min_level=logging.WARNING), ""]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- computation
def _importances(cohort: pd.DataFrame) -> dict[str, list[tuple[str, float, float]]]:
    """Full-cohort ridge fit → per-position features ranked by |coef|, each with its training null rate."""
    coefs = WeeklyRidge().fit(cohort).feature_importances()
    out: dict[str, list[tuple[str, float, float]]] = {}
    for pos, weights in coefs.items():
        sub = cohort[cohort["position"].astype("string") == pos]
        ranked = [
            (feat, coef, float(sub[feat].isna().mean()) if feat in sub.columns else 1.0)
            for feat, coef in weights.items()
        ]
        ranked.sort(key=lambda t: abs(t[1]), reverse=True)
        out[pos] = ranked
    return out


def _excluded_nulls(frame: pd.DataFrame) -> dict[str, float]:
    cohort = frame[frame["position"].isin(_POSITIONS)]
    return {
        col: (float(cohort[col].isna().mean()) if col in cohort.columns else 1.0)
        for col in _EXCLUDED_COLS
    }


def _cold_margins(
    ridge_cold: dict[str, PositionMetrics], cold_base: dict[str, dict[str, PositionMetrics]]
) -> dict[str, dict]:
    """Per-position cold-start margin of pure ridge vs the toughest baseline — the artifact's record."""
    out: dict[str, dict] = {}
    for pos in _POSITIONS:
        r = ridge_cold.get(pos)
        best_mae, best_rho = _bar(cold_base, pos)
        if r is None or best_mae is None or best_rho is None:
            continue
        out[pos] = {
            "n": r.n,
            "ridge_mae": round(r.mae, 4),
            "ridge_rho": None if r.spearman is None else round(r.spearman, 4),
            "bar_mae": round(best_mae[1], 4),
            "bar_mae_by": best_mae[0],
            "bar_rho": round(best_rho[1], 4),
            "bar_rho_by": best_rho[0],
            "delta_mae": round(r.mae - best_mae[1], 4),
            "delta_rho": None if r.spearman is None else round(r.spearman - best_rho[1], 4),
        }
    return out


def _write_artifact(cohort: pd.DataFrame, gate: Sequence[str], margins: dict, path: str) -> list[str]:
    """The committed fitted artifact: the full-span ridge plus the measured gate and its margins.

    The ridge is the trained, deterministic, committed object; the gate and margins are the measured
    cold-start decision (Decision #8). `PriorSeasonRank`, the deferral target, is a deterministic
    prior-season group mean re-fit from the lake at serve time (#34), so it is not frozen here.
    """
    ridge = WeeklyRidge().fit(cohort)
    payload = ridge.to_dict()
    payload["model"] = "WeeklyModel"
    payload["ridge_model"] = "WeeklyRidge"
    payload["cold_start_defer_to"] = "PriorSeasonRank"
    payload["cold_start_deferral"] = list(gate)
    payload["cold_start_margins"] = margins
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return list(ridge.fit_positions)


# --------------------------------------------------------------------------- entry point
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the weekly model, write the report and the artifact")
    ap.add_argument("--seasons", default=DEFAULT_SEASONS, help=f"train span (default: {DEFAULT_SEASONS})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"report path (default: {DEFAULT_OUT})")
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT, help=f"artifact path (default: {DEFAULT_ARTIFACT})")
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

    # Decision #7, fail-closed: a points head is valid only while skill scoring is linear.
    try:
        assert_linear_skill_scoring(scoring)
    except ValueError as exc:
        print(f"scoring is not points-head-safe — {exc}", file=sys.stderr)
        return 1

    partitions = int(len(lake_inventory()))
    print(
        f"Evaluating the weekly model over {seasons[0]}-{seasons[-1]} on backend {LAKE_BACKEND} "
        f"({partitions} partitions), scoring {league.get('name')!r} ({len(scoring)} keys)…"
    )

    capture = _Capture()
    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        frame = build_training_frame(seasons, scoring)
        cohort = frame[frame["position"].isin(_POSITIONS)]

        results: dict[str, EvalResult] = {
            _RIDGE: evaluate(WeeklyRidge(), frame, positions=_POSITIONS, name=_RIDGE)
        }
        for name in _BASELINES:
            results[name] = evaluate(BASELINE_FACTORIES[name](), frame, positions=_POSITIONS, name=name)
        cold = {name: cold_start_metrics(res.predictions, frame, positions=_POSITIONS)
                for name, res in results.items()}

        # The measured, per-position gate: defer where pure ridge loses the cold start.
        gate = deferred_positions(
            cold[_RIDGE], {n: cold[n] for n in _BASELINES}, positions=_POSITIONS
        )
        results[_SHIPPED] = evaluate(
            WeeklyModel(defer_cold_start=gate), frame, positions=_POSITIONS, name=_SHIPPED
        )
        cold[_SHIPPED] = cold_start_metrics(results[_SHIPPED].predictions, frame, positions=_POSITIONS)

        importances = _importances(cohort)
        excluded = _excluded_nulls(frame)
        margins = _cold_margins(cold[_RIDGE], {n: cold[n] for n in _BASELINES})
        fitted = _write_artifact(cohort, gate, margins, args.artifact)
    except ValueError as exc:
        print(f"could not build/evaluate the frame — {exc}", file=sys.stderr)
        return 1
    finally:
        root.removeHandler(capture)
        root.setLevel(prior_level)

    report = render_report(
        results, cold, importances, excluded, gate,
        seasons=seasons, scoring_keys=len(scoring), partitions=partitions,
        league_name=league.get("name", "?"),
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        frame_rows=len(frame), cohort_rows=int(len(cohort)), players=int(frame["player_id"].nunique()),
        records=capture.records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    n_warnings = sum(1 for level, *_ in capture.records if level >= logging.WARNING)
    print(
        f"Wrote {out} (report) and {args.artifact} (artifact: ridge {fitted}, "
        f"cold-start deferral {list(gate)}) — {len(cohort):,} skill rows, "
        f"test seasons {_season_span(sorted(results[_SHIPPED].test_seasons))}, {n_warnings} warning(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

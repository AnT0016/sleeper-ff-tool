"""Score the K + DST component model against the #28 baselines, write the report + artifact (#30).

    ./.venv/Scripts/python scripts/eval_kickdef.py
    ./.venv/Scripts/python scripts/eval_kickdef.py --seasons 2016-2025 --out docs/model-kickdef.md

K and DST are the phase's **lowest bar** and its clearest structural edge: both MAE baseline winners *are*
the learned position mean, and nothing in the project orders a kicker or a defence (baseline ρ 0.067 /
0.095). The model predicts stat **components** and lets ``scoring.engine.points`` price them — never a
points-valued head, because their scoring is a step function of stat buckets (Decision #2 / #7). This
script builds the component frame once, checks the correctness anchor (``engine(components) == label``,
which must clear a declared floor or the run fails), evaluates the pure-component model and the three
baselines **all-weeks and per (position × cold/warm) cell**, derives the measured per-cell gate, and
writes the committed report (``--out``) and fitted artifact (``--artifact``).

The distribution over the points-allowed bucket is the interesting quantity (``E[f(X)] != f(E[X])``); its
left tail — the 10-point shutout cell — is where a homoskedastic grid quietly fails, so the report
carries a **predicted-vs-realized rate by predicted-μ decile** calibration (``pts_allow_0`` and the
combined ``≤ 6`` cells). :func:`render_report` is pure over the evaluated results, so
``tests/test_model_kickdef.py`` pins that its prose follows the numbers it cites.
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

# `python scripts/eval_kickdef.py` puts scripts/ at the front of sys.path, where scripts/collect.py
# shadows the installed `collect` package the assembler imports. Drop the script directory.
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from collect import runner
from model.evaluate import EvalResult, PositionMetrics, evaluate
from model.kickdef import (
    ANCHOR_FLOOR_PCT,
    DEFAULT_PA_BINS,
    KickDefModel,
    anchor_mismatch,
    build_kickdef_frame,
    cell_metrics,
    component_model,
    deferred_cells,
    pts_allow_calibration,
)
from model.kickdef import _BASELINE_FACTORIES as BASELINE_FACTORIES
from model.kickdef import _LEFT_TAIL_THRESHOLDS as LEFT_TAIL_THRESHOLDS
from sleeper import client
from sleeper.config import LEAGUE_ID
from store.lake import LAKE_BACKEND, lake_inventory

DEFAULT_SEASONS = "2016-2025"
DEFAULT_OUT = "docs/model-kickdef.md"
DEFAULT_ARTIFACT = "src/model/fit/kickdef.json"

_SHIPPED = "KickDefModel"
_COMPONENT = "Component"
_BASELINES: tuple[str, ...] = ("TrailingMean", "PriorSeasonRank", "LaggedExpectedPoints")
_POSITIONS: tuple[str, ...] = ("K", "DEF")
_CELLS: tuple[str, ...] = ("K:warm", "K:cold", "DEF:warm", "DEF:cold")

_TIE_EPS = 1e-9


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


def _bar(baselines: dict[str, dict[str, PositionMetrics]], key: str):
    """The toughest baseline at ``key`` (a position or a cell): ((mae_name, mae), (rho_name, rho))."""
    maes = [(n, m[key].mae) for n, m in baselines.items() if key in m]
    rhos = [(n, m[key].spearman) for n, m in baselines.items()
            if key in m and m[key].spearman is not None]
    return (min(maes, key=lambda t: t[1]) if maes else None,
            max(rhos, key=lambda t: t[1]) if rhos else None)


def _verdict(m: PositionMetrics | None, best_mae, best_rho) -> str:
    """Model vs the toughest baseline, derived from the numbers. A win beats it on BOTH metrics."""
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


def _headline_table(subject: dict[str, PositionMetrics], baselines, keys: Sequence[str],
                    *, label: str = "position") -> list[str]:
    lines = [
        f"| {label} | n | model MAE | bar MAE | ΔMAE | model ρ | bar ρ | Δρ | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in keys:
        m = subject.get(key)
        best_mae, best_rho = _bar(baselines, key)
        if m is None or best_mae is None or best_rho is None:
            lines.append(f"| {key} | — | — | — | — | — | — | — | — |")
            continue
        dmae = m.mae - best_mae[1]
        drho = None if m.spearman is None else m.spearman - best_rho[1]
        lines.append(
            f"| {key} | {m.n:,} | {_fmt(m.mae)} | {_fmt(best_mae[1])} ({best_mae[0]}) | "
            f"{_signed(dmae)} | {_fmt(m.spearman, 3)} | {_fmt(best_rho[1], 3)} ({best_rho[0]}) | "
            f"{_signed(drho, 3)} | {_verdict(m, best_mae, best_rho)} |"
        )
    return lines


def _metric_table(per: dict[str, PositionMetrics], keys: Sequence[str], *, label: str = "position"):
    lines = [
        f"| {label} | n | MAE | RMSE | Spearman ρ | slates | ordered |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in keys:
        if key not in per:
            continue
        m = per[key]
        lines.append(
            f"| {key} | {m.n:,} | {_fmt(m.mae)} | {_fmt(m.rmse)} | {_fmt(m.spearman, 3)} | "
            f"{m.spearman_slates} | {m.spearman_ordered_slates} |"
        )
    return lines


def _calibration_table(cal: pd.DataFrame) -> list[str]:
    if cal.empty:
        return ["_No DEF distribution fitted — calibration unavailable._"]
    heads = ["decile", "n", "μ mean"]
    for t in LEFT_TAIL_THRESHOLDS:
        heads += [f"pred ≤{t}", f"real ≤{t}"]
    lines = ["| " + " | ".join(heads) + " |", "| --- |" + " ---: |" * (len(heads) - 1)]
    for row in cal.itertuples():
        cells = [str(int(row.decile)), f"{int(row.n):,}", _fmt(row.mu_mean, 1)]
        for t in LEFT_TAIL_THRESHOLDS:
            cells += [_fmt(getattr(row, f"pred_le{t}"), 4), _fmt(getattr(row, f"real_le{t}"), 4)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _anchor_table(anchor: dict[str, dict]) -> list[str]:
    lines = ["| position | rows | matched | match % | mismatch keys |", "| --- | ---: | ---: | ---: | --- |"]
    for pos in _POSITIONS:
        a = anchor.get(pos, {})
        keys = ", ".join(f"`{k}`×{v}" for k, v in list(a.get("mismatch_keys", {}).items())[:4]) or "—"
        lines.append(
            f"| {pos} | {a.get('n', 0):,} | {a.get('matched', 0):,} | "
            f"{_fmt(a.get('match_pct'), 2)} | {keys} |"
        )
    return lines


def _log_block(records: Sequence[tuple[int, str, str, str]], *, min_level: int) -> str:
    lines = [f"{lvl} {name}: {msg}" for level, lvl, name, msg in records if level >= min_level]
    return "```\n" + "\n".join(lines) + "\n```" if lines else "_None._"


def _cell_bits(component_cells, baseline_cells, cells) -> str:
    bits = []
    for cell in cells:
        m = component_cells.get(cell)
        best_mae, best_rho = _bar(baseline_cells, cell)
        if m is None or best_mae is None or best_rho is None:
            continue
        bits.append(f"{cell} n={m.n:,} {_verdict(m, best_mae, best_rho)}")
    return "; ".join(bits)


def _pos_bits(subject, baselines, positions) -> str:
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


def _tail_note(cal1: pd.DataFrame, cal3: pd.DataFrame) -> str:
    """A generated one-liner justifying μ-conditioning from the two calibrations' lowest-μ decile."""
    if cal1.empty or cal3.empty:
        return "left-tail calibration unavailable"
    d1_single = cal1.iloc[0]
    d1_binned = cal3.iloc[0]
    return (
        f"in the lowest-μ decile (elite defences) the single shared grid predicts "
        f"P(shutout) {d1_single['pred_le0']:.4f} against a realized {d1_single['real_le0']:.4f} — the "
        f"clamp inventing shutout mass — while the μ-conditioned grid closes it to "
        f"{d1_binned['pred_le0']:.4f}"
    )


# --------------------------------------------------------------------------- render
def render_report(
    results: dict[str, EvalResult],
    component_cells: dict[str, PositionMetrics],
    baseline_cells: dict[str, dict[str, PositionMetrics]],
    gate: Sequence[str],
    anchor: dict[str, dict],
    cal_binned: pd.DataFrame,
    cal_single: pd.DataFrame,
    *,
    seasons: Sequence[int],
    n_pa_bins: int,
    scoring_keys: int,
    partitions: int,
    league_name: str,
    generated: str,
    frame_rows: int,
    k_rows: int,
    def_rows: int,
    players: int,
    records: Sequence[tuple[int, str, str, str]] = (),
) -> str:
    """The committed report. Pure over the evaluated results so its prose can be pinned to its tables."""
    all_base = {n: results[n].per_position for n in _BASELINES if n in results}
    shipped_all = results[_SHIPPED].per_position
    scored_seasons = sorted(results[_SHIPPED].test_seasons)
    n_warnings = sum(1 for level, *_ in records if level >= logging.WARNING)

    wins = [p for p in _POSITIONS if _verdict(shipped_all.get(p), *_bar(all_base, p)) == "win"]
    fielded_cells = [c for c in _CELLS if c not in gate]
    deferred_cells_list = [c for c in _CELLS if c in gate]

    parts: list[str] = [
        "# Weekly K + DST model — the measured grade (Phase 9, ticket #30)",
        "",
        "> Generated by [`scripts/eval_kickdef.py`](../scripts/eval_kickdef.py) — regenerate with "
        "`./.venv/Scripts/python scripts/eval_kickdef.py`. This file is a committed artifact; do not "
        "hand-edit it.",
        "",
        f"- **Generated (UTC):** {generated}",
        f"- **Train span:** {seasons[0]}–{seasons[-1]} · **test seasons (scored):** "
        f"{_season_span(scored_seasons)} walk-forward (train ≤ S-1, test S; 2016-2017 are lag warm-up "
        f"and are never scored — spec Decision #6). Read back from the results.",
        f"- **Lake backend:** `{LAKE_BACKEND}` · {partitions} partitions read",
        f"- **Scoring:** {league_name!r} live `scoring_settings`, {scoring_keys} keys (label re-scored "
        "by the Phase 1 engine; predictions **priced by the same engine over predicted stat lines** — "
        "never a points head, Decision #2).",
        f"- **Frame:** {frame_rows:,} K+DST rows ({k_rows:,} K, {def_rows:,} DEF), {players:,} players",
        f"- **Shipped model:** `KickDefModel` — per-component closed-form ridge through the scoring "
        f"engine (K: per-band makes; DST: counting stats + a **μ-conditioned ({n_pa_bins}-bin) "
        f"points-allowed distribution**); component fielded at "
        f"{fielded_cells if fielded_cells else 'no'} cell(s), deferred to `PriorSeasonRank` at "
        f"{deferred_cells_list if deferred_cells_list else 'none'} (measured — §B).",
        "",
        "The bar is the three naive baselines in "
        "[`docs/model-baselines.md`](model-baselines.md): beat **all three** on **both** MAE and "
        "within-slate Spearman ρ, per position — the **lowest bar in the phase** (both MAE winners are "
        "the position mean; nothing else orders a kicker or a defence, baseline ρ 0.067 / 0.095).",
        "",
        "## Findings, measured",
        "",
        f"1. **All-weeks, per position.** {_pos_bits(shipped_all, all_base, _POSITIONS)}. A **win** "
        f"clears all three baselines on both metrics. Won at: {', '.join(wins) if wins else 'none'} — "
        "the structural edge (a distribution over the bucketed scoring, not a points head) is what "
        "buys the ordering the baselines cannot give (ρ 0.067 / 0.095).",
        f"2. **The gate, per (position × cold/warm) cell.** {_cell_bits(component_cells, baseline_cells, _CELLS)}. "
        f"The component model is fielded where it beats the baselines on both metrics and rests on "
        f"enough held-out rows, and defers where it does not — measured per **cell**, not per position, "
        f"so a warm win is not thrown away by a cold loss (Decision #9 item 6). Fielded: "
        f"{', '.join(fielded_cells) if fielded_cells else 'none'}; deferred: "
        f"{', '.join(deferred_cells_list) if deferred_cells_list else 'none'}.",
        f"3. **The points-allowed distribution's left tail is calibrated (§E).** The 10-point shutout "
        f"cell is the highest in the table and the most sensitive to left-tail shape, so the residual "
        f"grid is conditioned on μ: {_tail_note(cal_single, cal_binned)}.",
        f"4. **The decomposition is complete (anchor, §F).** `engine(observed components)` reproduces "
        f"`y_custom_points` for K {_fmt(anchor['K']['match_pct'], 2)}% and DEF "
        f"{_fmt(anchor['DEF']['match_pct'], 2)}% of rows (floor {ANCHOR_FLOOR_PCT}%) — the proof that "
        "the model prices the *whole* stat line through the engine and misses no scoring key.",
        f"5. **{'Zero' if not n_warnings else str(n_warnings)} warning"
        f"{'' if n_warnings == 1 else 's'} on real data.** The frame build and the walk over the splits "
        f"emitted **{n_warnings}** WARNING-level line(s) (verbatim below); a skipped test season would "
        "appear there, so the scored span and the header cannot drift apart silently.",
        "",
        "## A. All-weeks headline — shipped model vs the toughest baseline",
        "",
        "ΔMAE negative and Δρ positive both favour the model; a **win** needs both. `bar MAE`/`bar ρ` "
        "name the hardest of the three baselines on each metric (they need not be the same baseline).",
    ]
    parts += _headline_table(shipped_all, all_base, _POSITIONS)

    parts += ["", "## B. Per-cell headline — the measured gate (position × cold/warm)", ""]
    parts.append(
        "Four cells, because \"beats the baselines for K and DEF\" is a claim with four cells behind it. "
        "A cell is **fielded** only when the component model beats every baseline on both metrics *and* "
        "rests on enough held-out rows (a thin cold cell must not read like a thick warm one); "
        "everything else **defers** to `PriorSeasonRank`. `n` is the held-out row count behind each "
        "cell's decision."
    )
    parts += _headline_table(component_cells, baseline_cells, _CELLS, label="cell")

    parts += ["", "## C. All-weeks metrics — shipped, pure component, and the three baselines", ""]
    parts.append(
        "MAE/RMSE over all pooled out-of-sample rows; ρ the mean of the per-`(season, week)` slate rank "
        "correlations. `ordered` counts the boards given a non-constant prediction (a flat one scores "
        "ρ = 0, not excused) — the honest headline for K/DST, where the baselines default that column "
        "(see [`docs/model-baselines.md`](model-baselines.md))."
    )
    for name in (_SHIPPED, _COMPONENT, *_BASELINES):
        if name in results:
            parts += ["", f"### {name}", *_metric_table(results[name].per_position, _POSITIONS)]

    parts += ["", "## D. Per-cell metrics — component vs the baselines (position × cold/warm)", ""]
    parts.append(
        "The four cells the gate is decided on. The cold cells (a kicker's or defence's first appearance "
        "of a season, no within-season lag) are where the market-based components either win outright or "
        "are beaten by a prior-season level — reported separately per Decision #8, win or lose."
    )
    parts += ["", "### Component", *_metric_table(component_cells, _CELLS, label="cell")]
    for name in _BASELINES:
        if name in baseline_cells:
            parts += ["", f"### {name}", *_metric_table(baseline_cells[name], _CELLS, label="cell")]

    parts += ["", "## E. Points-allowed distribution — left-tail calibration by predicted-μ decile", ""]
    parts.append(
        "Predicted vs realized `P(pts_allow ≤ t)` per predicted-μ decile (d1 = lowest μ, the elite "
        "defences). The buckets are discontinuous and `pts_allow_0` is 10 points — the highest cell — so "
        "the shutout probability for elite defences is the error that would look fine on MAE and be wrong "
        "on the boards. A homoskedastic grid over-predicts it (the clamp piles sub-zero mass onto 0); the "
        f"μ-conditioned ({n_pa_bins}-bin) grid tracks realized. The `≤ 6` cell is the combined top-two "
        "buckets — enough rows to read where the ~1%-of-rows shutout alone is too thin."
    )
    parts += ["", f"**Shipped ({n_pa_bins}-bin μ-conditioned grid):**", *_calibration_table(cal_binned)]
    parts += ["", "**Single shared grid (the diagnostic, 1 bin) — why μ-conditioning:**",
              *_calibration_table(cal_single)]

    parts += ["", "## F. Correctness anchor — `engine(observed components)` == `y_custom_points`", ""]
    parts.append(
        f"The proof that the component decomposition is complete: every K/DST row's observed component "
        f"line, priced by the engine, must reproduce the re-scored label to within 0.01 points. The "
        f"floor is **{ANCHOR_FLOOR_PCT}%** per position, declared before measuring; below it a scoring "
        "key is going unextracted and the run fails. `mismatch keys` names the scoring keys implicated "
        "on any rows that miss."
    )
    parts += _anchor_table(anchor)

    parts += ["", "## Warnings (verbatim)", ""]
    parts.append(
        "The standing bar is zero unexpected warnings on real data. A skipped test season appears here, "
        "so the span in the header and the work actually done cannot drift apart silently."
    )
    parts += ["", _log_block(records, min_level=logging.WARNING), ""]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- computation
def _cell_margins(component_cells, baseline_cells) -> dict[str, dict]:
    """Per-cell margin of the component model vs the toughest baseline — the artifact's record."""
    out: dict[str, dict] = {}
    for cell in _CELLS:
        m = component_cells.get(cell)
        best_mae, best_rho = _bar(baseline_cells, cell)
        if m is None or best_mae is None or best_rho is None:
            continue
        out[cell] = {
            "n": m.n,
            "component_mae": round(m.mae, 4),
            "component_rho": None if m.spearman is None else round(m.spearman, 4),
            "bar_mae": round(best_mae[1], 4),
            "bar_mae_by": best_mae[0],
            "bar_rho": round(best_rho[1], 4),
            "bar_rho_by": best_rho[0],
            "verdict": _verdict(m, best_mae, best_rho),
        }
    return out


def _write_artifact(
    frame, scoring, gate, margins, anchor, cal_binned, n_pa_bins, path: str
) -> None:
    """The committed fitted artifact: the full-span component model, the measured gate, and its evidence.

    The component heads are the trained, deterministic, committed object; the gate, per-cell margins,
    anchor and left-tail calibration are the measured evidence behind the deferral decision (Decision #9
    item 1/4). `PriorSeasonRank`, the deferral target, is a group mean re-fit from the lake at serve
    time, so it is not frozen here.
    """
    model = KickDefModel(scoring, n_pa_bins=n_pa_bins, defer=gate).fit(frame)
    payload = model.to_dict()
    payload["deferral"] = list(gate)
    payload["cell_margins"] = margins
    payload["anchor"] = {
        p: {k: anchor[p][k] for k in ("n", "matched", "match_pct")} for p in _POSITIONS
    }
    payload["pts_allow_calibration"] = cal_binned.round(6).to_dict("records") if not cal_binned.empty else []
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the K + DST model, write the report and artifact")
    ap.add_argument("--seasons", default=DEFAULT_SEASONS, help=f"train span (default: {DEFAULT_SEASONS})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"report path (default: {DEFAULT_OUT})")
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT, help=f"artifact path (default: {DEFAULT_ARTIFACT})")
    ap.add_argument("--n-pa-bins", type=int, default=DEFAULT_PA_BINS,
                    help=f"points-allowed μ bins (default: {DEFAULT_PA_BINS}; 1 is the diagnostic)")
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
        print(f"could not load scoring_settings from league {LEAGUE_ID} ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return 1

    partitions = int(len(lake_inventory()))
    print(
        f"Evaluating the K+DST model over {seasons[0]}-{seasons[-1]} on backend {LAKE_BACKEND} "
        f"({partitions} partitions), scoring {league.get('name')!r} ({len(scoring)} keys)…"
    )

    capture = _Capture()
    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        frame = build_kickdef_frame(seasons, scoring)

        # The correctness anchor gates the run: a decomposition that cannot reproduce the label is wrong.
        anchor = anchor_mismatch(frame, scoring)
        below = {p: a for p, a in anchor.items() if a["match_pct"] < ANCHOR_FLOOR_PCT}
        if below:
            for pos, a in below.items():
                print(
                    f"ANCHOR FAILURE {pos}: engine(components) reproduced only {a['match_pct']:.2f}% of "
                    f"the label (floor {ANCHOR_FLOOR_PCT}%). Mismatch keys: {a['mismatch_keys']}. A "
                    "scoring key is going unextracted — the component decomposition is incomplete.",
                    file=sys.stderr,
                )
            return 1

        results: dict[str, EvalResult] = {
            _COMPONENT: evaluate(component_model(scoring, n_pa_bins=args.n_pa_bins), frame,
                                 positions=_POSITIONS, name=_COMPONENT)
        }
        for name in _BASELINES:
            results[name] = evaluate(BASELINE_FACTORIES[name](), frame, positions=_POSITIONS, name=name)

        component_cells = cell_metrics(results[_COMPONENT].predictions, frame)
        baseline_cells = {n: cell_metrics(results[n].predictions, frame) for n in _BASELINES}

        gate = deferred_cells(component_cells, baseline_cells)
        results[_SHIPPED] = evaluate(
            KickDefModel(scoring, n_pa_bins=args.n_pa_bins, defer=gate), frame,
            positions=_POSITIONS, name=_SHIPPED,
        )

        cal_binned = pts_allow_calibration(frame, scoring, n_pa_bins=args.n_pa_bins)
        cal_single = pts_allow_calibration(frame, scoring, n_pa_bins=1)
        margins = _cell_margins(component_cells, baseline_cells)
        _write_artifact(frame, scoring, gate, margins, anchor, cal_binned, args.n_pa_bins, args.artifact)
    except ValueError as exc:
        print(f"could not build/evaluate the frame — {exc}", file=sys.stderr)
        return 1
    finally:
        root.removeHandler(capture)
        root.setLevel(prior_level)

    report = render_report(
        results, component_cells, baseline_cells, gate, anchor, cal_binned, cal_single,
        seasons=seasons, n_pa_bins=args.n_pa_bins, scoring_keys=len(scoring), partitions=partitions,
        league_name=league.get("name", "?"), generated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        frame_rows=len(frame), k_rows=int((frame["position"] == "K").sum()),
        def_rows=int((frame["position"] == "DEF").sum()), players=int(frame["player_id"].nunique()),
        records=capture.records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    n_warnings = sum(1 for level, *_ in capture.records if level >= logging.WARNING)
    print(
        f"Wrote {out} (report) and {args.artifact} (artifact) — {len(frame):,} K+DST rows, "
        f"test seasons {_season_span(sorted(results[_SHIPPED].test_seasons))}, gate {list(gate)}, "
        f"{n_warnings} warning(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

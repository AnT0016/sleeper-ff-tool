"""The live model-vs-Sleeper swap gate — the only honest "beat the market" grade (Phase 9, #34).

    ./.venv/Scripts/python scripts/eval_swap_gate.py
    ./.venv/Scripts/python scripts/eval_swap_gate.py --season 2026 --out docs/model-swap-gate.md

Decision #3, made a check rather than a note: the model replaces Sleeper **as the default** for a
position only after it beats ``baseline_sleeper_points`` on **both** MAE and within-week Spearman ρ over
**≥ 4 live 2026 weeks**. That comparison is **forward-only** — ``baseline_sleeper_points`` is null for
every row of 2016-2025 (Sleeper's pre-lock capture produces its first rows in 2026 W1, spec finding #1)
— so today this ships a scoreboard with **zero rows**, and its whole correctness is behaviour on empty
and partial input: with < 4 live weeks a position is **NOT MET**, never a vacuous pass (a vacuous pass
would swap the default projection source for every surface in the tool).

The recorded state (``src/model/fit/swap_gate.json``) is read by :func:`projections.source.default_source`,
so a position flips to the model only through a deliberate regeneration + commit of that artifact — no
cron regenerates it (pinned by ``tests/test_workflows.py``). :func:`render_report` and
:func:`swap_gate_state` are pure over the scoreboard, so ``tests/test_projection_source.py`` pins that
the report's verdicts follow the numbers and that the gate fails closed at 0 and < 4 weeks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# `python scripts/eval_swap_gate.py` puts scripts/ at the front of sys.path, where scripts/collect.py
# shadows the installed `collect` package the assembler imports. Drop the script directory.
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from model.evaluate import spearman
from sleeper import client
from sleeper.config import LEAGUE_ID
from store.lake import LAKE_BACKEND, lake_inventory

DEFAULT_OUT = "docs/model-swap-gate.md"
DEFAULT_ARTIFACT = "src/model/fit/swap_gate.json"

#: The first live season — ``baseline_sleeper_points`` starts here (spec finding #1).
DEFAULT_SEASON = 2026

#: The six positions the gate reports (Decision #3 is per position).
POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")

#: Decision #3's week floor: a position needs at least this many live weeks before its evidence can
#: decide the swap. Below it the position is NOT MET regardless of the metrics — the fail-closed rule.
MIN_LIVE_WEEKS = 4

#: The re-scored actual and the forward-only market projection, both from ``build_training_frame``.
_ACTUAL_COL = "y_custom_points"
_MARKET_COL = "baseline_sleeper_points"

_TIE_EPS = 1e-9


# --------------------------------------------------------------------------- log capture
class _Capture(logging.Handler):
    """Records the run's log lines so the report can count and quote its warnings."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[tuple[int, str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.levelname, record.name, record.getMessage()))


# --------------------------------------------------------------------------- the scoreboard
def build_scoreboard(
    frame: pd.DataFrame,
    *,
    predict_fn: Callable[[pd.DataFrame], pd.Series],
    positions: Sequence[str] = POSITIONS,
) -> pd.DataFrame:
    """One comparable row per player-week: ``position, season, week, actual, market, model``.

    A row is comparable only where the actual, the market projection **and** the model prediction are all
    present — a completed week with a captured pre-lock market number. ``predict_fn`` maps the frame to a
    model prediction Series aligned to its index (injectable so the gate logic is testable offline).
    """
    cols = ["position", "season", "week", "actual", "market", "model"]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    work = frame[frame["position"].isin(tuple(positions))].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)
    model = pd.to_numeric(predict_fn(work).reindex(work.index), errors="coerce")
    board = pd.DataFrame(
        {
            "position": work["position"].astype("string").to_numpy(),
            "season": pd.to_numeric(work["season"], errors="coerce").to_numpy(),
            "week": pd.to_numeric(work["week"], errors="coerce").to_numpy(),
            "actual": pd.to_numeric(work.get(_ACTUAL_COL), errors="coerce").to_numpy(),
            "market": pd.to_numeric(work.get(_MARKET_COL), errors="coerce").to_numpy(),
            "model": model.to_numpy(),
        }
    )
    return board.dropna(subset=["actual", "market", "model", "week"]).reset_index(drop=True)


def _position_cell(sub: pd.DataFrame) -> dict:
    """The metric cell for one position: pooled MAE and mean within-week ρ, for model and market.

    A **live week** is a ``(season, week)`` slate with ≥ 2 comparable rows (so a within-week ρ exists);
    MAE is pooled over those weeks' rows, ρ is the mean of the per-week slate correlations. ``weeks`` is
    the count that both metrics and the gate read — the fail-closed quantity.
    """
    model_rhos: list[float] = []
    market_rhos: list[float] = []
    live_rows: list[pd.DataFrame] = []
    for _slate, grp in sub.groupby(["season", "week"], sort=True):
        if len(grp) < 2:
            continue
        mr = spearman(grp["model"], grp["actual"])
        kr = spearman(grp["market"], grp["actual"])
        if mr is None or kr is None:  # a slate every player tied on — neither source could order it
            continue
        model_rhos.append(mr)
        market_rhos.append(kr)
        live_rows.append(grp)
    weeks = len(live_rows)
    if weeks == 0:
        return {"weeks": 0, "n": 0, "model_mae": None, "market_mae": None,
                "model_rho": None, "market_rho": None}
    pooled = pd.concat(live_rows, ignore_index=True)
    return {
        "weeks": weeks,
        "n": int(len(pooled)),
        "model_mae": float(np.mean(np.abs(pooled["model"] - pooled["actual"]))),
        "market_mae": float(np.mean(np.abs(pooled["market"] - pooled["actual"]))),
        "model_rho": float(np.mean(model_rhos)),
        "market_rho": float(np.mean(market_rhos)),
    }


def _met(cell: Mapping, *, min_weeks: int) -> bool:
    """Decision #3, derived from the cell: ≥ min_weeks live weeks **and** model wins BOTH metrics.

    Fails closed: any missing metric, too few weeks, a tie, or a loss on either metric → not met.
    """
    if cell["weeks"] < min_weeks:
        return False
    if cell["model_mae"] is None or cell["market_mae"] is None:
        return False
    if cell["model_rho"] is None or cell["market_rho"] is None:
        return False
    return (
        cell["model_mae"] < cell["market_mae"] - _TIE_EPS
        and cell["model_rho"] > cell["market_rho"] + _TIE_EPS
    )


def _reason(cell: Mapping, *, min_weeks: int, met: bool) -> str:
    """A one-line, table-derived justification for the verdict — never written alongside the number."""
    if cell["weeks"] < min_weeks:
        return f"{cell['weeks']} live week(s) < {min_weeks} required — fails closed"
    if met:
        return f"beats Sleeper on both metrics over {cell['weeks']} live week(s)"
    bits = []
    if cell["model_mae"] is None or cell["model_mae"] >= cell["market_mae"] - _TIE_EPS:
        bits.append("MAE not better")
    if cell["model_rho"] is None or cell["model_rho"] <= cell["market_rho"] + _TIE_EPS:
        bits.append("ρ not better")
    return f"{cell['weeks']} live week(s) but {', '.join(bits) or 'no improvement'}"


def swap_gate_state(
    scoreboard: pd.DataFrame,
    *,
    positions: Sequence[str] = POSITIONS,
    min_weeks: int = MIN_LIVE_WEEKS,
) -> dict[str, dict]:
    """Per-position gate state — the recorded artifact's ``positions`` block. Pure; fails closed at < 4.

    Every position appears, so a position with no live rows is an explicit ``met=false`` NOT MET, not a
    silent absence. This is the same shape :func:`projections.source.recorded_swap_gate` reads back.
    """
    out: dict[str, dict] = {}
    for pos in positions:
        sub = scoreboard[scoreboard["position"] == pos] if not scoreboard.empty else scoreboard
        cell = _position_cell(sub if not scoreboard.empty else pd.DataFrame(columns=scoreboard.columns))
        met = _met(cell, min_weeks=min_weeks)
        out[pos] = {**cell, "met": met, "reason": _reason(cell, min_weeks=min_weeks, met=met)}
    return out


# --------------------------------------------------------------------------- rendering
def _fmt(value: float | None, nd: int = 2) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:.{nd}f}"


def _signed(value: float | None, nd: int = 2) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:+.{nd}f}"


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _verdict(cell: Mapping) -> str:
    return "✅ MET" if cell.get("met") else "NOT MET"


def _gate_table(state: Mapping[str, Mapping]) -> list[str]:
    lines = [
        "| position | live weeks | model MAE | Sleeper MAE | ΔMAE | model ρ | Sleeper ρ | Δρ | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pos, cell in state.items():
        dmae = _delta(cell["model_mae"], cell["market_mae"])
        drho = _delta(cell["model_rho"], cell["market_rho"])
        lines.append(
            f"| {pos} | {cell['weeks']} | {_fmt(cell['model_mae'])} | {_fmt(cell['market_mae'])} | "
            f"{_signed(dmae)} | {_fmt(cell['model_rho'], 3)} | {_fmt(cell['market_rho'], 3)} | "
            f"{_signed(drho, 3)} | {_verdict(cell)} |"
        )
    return lines


def _log_block(records: Sequence[tuple[int, str, str, str]], *, min_level: int) -> str:
    lines = [f"{lvl} {name}: {msg}" for level, lvl, name, msg in records if level >= min_level]
    return "```\n" + "\n".join(lines) + "\n```" if lines else "_None._"


def render_report(
    state: Mapping[str, Mapping],
    *,
    season: int,
    min_weeks: int,
    scoring_keys: int,
    partitions: int,
    league_name: str,
    generated: str,
    scoreboard_rows: int,
    records: Sequence[tuple[int, str, str, str]] = (),
) -> str:
    """The committed report. Pure over the gate state, so every verdict sentence is **derived** from the
    table that could contradict it (Decision #9 item 1, this ticket's headline addition)."""
    met = [p for p, c in state.items() if c.get("met")]
    not_met = [p for p, c in state.items() if not c.get("met")]
    total_live = sum(int(c["weeks"]) for c in state.values())
    n_warnings = sum(1 for level, *_ in records if level >= logging.WARNING)

    parts: list[str] = [
        "# Live model-vs-Sleeper swap gate (Phase 9, ticket #34)",
        "",
        "> Generated by [`scripts/eval_swap_gate.py`](../scripts/eval_swap_gate.py) — regenerate with "
        "`./.venv/Scripts/python scripts/eval_swap_gate.py`. This file is a committed artifact; do not "
        "hand-edit it. Its verdicts are computed from the scoreboard, not written beside it.",
        "",
        f"- **Generated (UTC):** {generated}",
        f"- **Live season:** {season} · **week floor:** {min_weeks} (Decision #3). The grade is "
        "**forward-only** — `baseline_sleeper_points` is null for all of 2016–2025, so the model-vs-market "
        "comparison exists only from 2026 W1, accumulating one week at a time.",
        f"- **Lake backend:** `{LAKE_BACKEND}` · {partitions} partitions read",
        f"- **Scoring:** {league_name!r} live `scoring_settings`, {scoring_keys} keys.",
        f"- **Scoreboard:** {scoreboard_rows:,} comparable player-week(s) (actual + market + model all "
        f"present), {total_live} position-live-week(s) in total.",
        "",
        f"**Verdict: {len(met)} of {len(state)} position(s) have met the swap bar** — "
        f"met: {', '.join(met) if met else 'none'}; not met: {', '.join(not_met) if not_met else 'none'}. "
        f"The default projection source stays **Sleeper** for every NOT-MET position "
        f"([`projections.source.default_source`](../src/projections/source.py) reads the recorded state "
        f"below).",
        "",
        "## The gate — model vs Sleeper, per position",
        "",
        "A position is **MET** only when the model beats `baseline_sleeper_points` on **both** MAE "
        "(lower) and within-week Spearman ρ (higher) over at least the week floor of **live** weeks. "
        "`ΔMAE` negative and `Δρ` positive both favour the model; a verdict needs both **and** the week "
        "count. Fewer than the floor → NOT MET regardless of the metrics (fails closed).",
    ]
    parts += _gate_table(state)

    parts += ["", "## Why each verdict (derived from the row above)", ""]
    for pos, cell in state.items():
        parts.append(f"- **{pos}** — {_verdict(cell)}: {cell['reason']}.")

    parts += [
        "",
        "## Reading it today",
        "",
        f"With **{scoreboard_rows:,}** comparable rows the scoreboard is "
        f"{'empty' if scoreboard_rows == 0 else 'sparse'} — expected before 2026 W1, and exactly the "
        "case the gate must get right: **every position is NOT MET on too few weeks, never a vacuous "
        "pass**. As live weeks accumulate the table fills in, and a position flips only when its own "
        "four-plus weeks clear both metrics — RB clearing the bar never drags K with it (Decision #9 "
        "item 6). Re-run this script each week to refresh the recorded state; committing the flip is a "
        "deliberate human act (no cron regenerates it — `tests/test_workflows.py`).",
        "",
        "## Warnings (verbatim)",
        "",
        f"The standing bar is zero unexpected warnings on real data ({n_warnings} emitted).",
        "",
        _log_block(records, min_level=logging.WARNING),
        "",
    ]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- artifact
def build_artifact(state: Mapping[str, Mapping], *, season: int, min_weeks: int, generated: str) -> dict:
    """The recorded gate state ``projections.source.recorded_swap_gate`` reads back."""
    return {
        "model": "swap_gate",
        "season": season,
        "min_live_weeks": min_weeks,
        "generated": generated,
        "positions": {pos: dict(cell) for pos, cell in state.items()},
    }


def _model_predict(frame: pd.DataFrame, scoring: Mapping[str, float], season: int) -> pd.Series:
    """The shipped weekly models over the frame — skill ridge + K/DST components, each own gate.

    Trains on seasons strictly before ``season`` (the honest forward setup) via the shared seam helper,
    so the gate and the serving path predict identically.
    """
    from projections.source import _fit_predict_weekly

    preds = _fit_predict_weekly(list(range(2016, season)), frame, scoring, POSITIONS)
    return pd.Series(preds, dtype="float64").reindex(frame.index)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the live model-vs-Sleeper swap gate, write report + artifact")
    ap.add_argument("--season", type=int, default=DEFAULT_SEASON, help=f"live season (default: {DEFAULT_SEASON})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"report path (default: {DEFAULT_OUT})")
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT, help=f"artifact path (default: {DEFAULT_ARTIFACT})")
    ap.add_argument("--min-weeks", type=int, default=MIN_LIVE_WEEKS,
                    help=f"live-week floor (default: {MIN_LIVE_WEEKS}; Decision #3)")
    args = ap.parse_args(argv)

    try:
        league = client.get_league(LEAGUE_ID)
        scoring = league["scoring_settings"]
    except Exception as exc:
        print(f"could not load scoring_settings from league {LEAGUE_ID} ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return 1

    partitions = int(len(lake_inventory()))
    print(
        f"Scoring the swap gate for {args.season} on backend {LAKE_BACKEND} ({partitions} partitions), "
        f"scoring {league.get('name')!r} ({len(scoring)} keys)…"
    )

    capture = _Capture()
    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    log = logging.getLogger(__name__)
    try:
        from dataset.assemble import build_training_frame

        try:
            frame = build_training_frame([args.season], scoring)
        except ValueError as exc:
            # The assembler fails closed when the live season isn't in the lake yet (no schedule → no
            # week locks). That IS the pre-2026-W1 state, and the correct gate output for it is an empty
            # scoreboard → every position NOT MET on zero weeks — never a hard error, never a vacuous
            # pass. Logged at INFO (expected, not a warning), so the zero-unexpected-warnings bar holds.
            log.info(
                "no assemblable lake data for %d yet (%s) — scoreboard empty, every position NOT MET "
                "on zero live weeks (fails closed).", args.season, exc,
            )
            frame = pd.DataFrame()
        scoreboard = build_scoreboard(
            frame, predict_fn=lambda f: _model_predict(f, scoring, args.season)
        )
        state = swap_gate_state(scoreboard, min_weeks=args.min_weeks)
    finally:
        root.removeHandler(capture)
        root.setLevel(prior_level)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    artifact = build_artifact(state, season=args.season, min_weeks=args.min_weeks, generated=generated)
    out_artifact = Path(args.artifact)
    out_artifact.parent.mkdir(parents=True, exist_ok=True)
    out_artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = render_report(
        state, season=args.season, min_weeks=args.min_weeks, scoring_keys=len(scoring),
        partitions=partitions, league_name=league.get("name", "?"), generated=generated,
        scoreboard_rows=int(len(scoreboard)), records=capture.records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    met = [p for p, c in state.items() if c.get("met")]
    print(
        f"Wrote {out} (report) and {args.artifact} (artifact) — {len(scoreboard):,} comparable rows, "
        f"{len(met)} of {len(state)} position(s) MET: {met or 'none'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

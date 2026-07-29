"""Score the three naive baselines walk-forward and commit the recorded bar (Phase 9, ticket #28).

    ./.venv/Scripts/python scripts/eval_baselines.py
    ./.venv/Scripts/python scripts/eval_baselines.py --seasons 2016-2025 --out docs/model-baselines.md

With no historical Sleeper projection to beat (``baseline_sleeper_points`` is null for all of
2016-2025), "better" has no meaning until the bar is written down. This script builds the training
frame once (against whatever ``LAKE_BACKEND`` points at — the local backfill by default), runs
:class:`~model.baselines.TrailingMean`, :class:`~model.baselines.PriorSeasonRank` and
:class:`~model.baselines.LaggedExpectedPoints` through the walk-forward harness over 2018-2025, and
writes :data:`DEFAULT_OUT` as a **committed artifact**. Every later ticket's claim of improvement is
then checkable against a fixed number rather than a remembered impression.

The label is re-scored live from the active league's ``scoring_settings`` (the Phase 1 engine), never
hard-coded — the same rule the assembler and the profiler follow. :func:`render_report` is a pure
function of the evaluated results, so ``tests/test_model_evaluate.py`` can pin that its prose follows
the numbers it cites (the failure mode a committed report has: a headline the table beneath it
contradicts).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

# Render UTF-8 regardless of the console code page (Windows defaults to cp1252) — the tables and the
# em-dashes below are UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# `python scripts/eval_baselines.py` puts scripts/ at the FRONT of sys.path, where scripts/collect.py
# shadows the installed `collect` package the assembler imports. Drop the script directory; nothing in
# scripts/ imports a sibling script. (The truthiness check keeps the cwd entry from matching.)
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from collect import runner
from dataset.assemble import build_training_frame
from model.baselines import LaggedExpectedPoints, PriorSeasonRank, TrailingMean
from model.evaluate import (
    DEFAULT_TEST_SEASONS,
    FANTASY_POSITIONS,
    EvalResult,
    evaluate,
)
from sleeper import client
from sleeper.config import LEAGUE_ID
from store.lake import LAKE_BACKEND, lake_inventory

#: Ten seasons of training data, the span the backfill pulls (backfill_lake.DEFAULT_SEASONS). Test
#: seasons are always 2018-2025 (DEFAULT_TEST_SEASONS); 2016-2017 are lag warm-up.
DEFAULT_SEASONS = "2016-2025"
DEFAULT_OUT = "docs/model-baselines.md"

#: The baselines, in the order the report lists them: recency, level, opportunity.
BASELINES = (
    ("TrailingMean", lambda: TrailingMean()),
    ("PriorSeasonRank", lambda: PriorSeasonRank()),
    ("LaggedExpectedPoints", lambda: LaggedExpectedPoints()),
)


# --------------------------------------------------------------------------- rendering helpers
def _fmt(value: float | None, nd: int = 2) -> str:
    """A metric, or an em-dash where it is undefined (a Spearman with no admissible slate)."""
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "—"
    return f"{value:.{nd}f}"


def _best_by_mae(results: Sequence[EvalResult], pos: str) -> tuple[str, float] | None:
    scored = [(r.predictor, r.per_position[pos].mae) for r in results if pos in r.per_position]
    return min(scored, key=lambda t: t[1]) if scored else None


def _best_by_rho(results: Sequence[EvalResult], pos: str) -> tuple[str, float] | None:
    scored = [
        (r.predictor, r.per_position[pos].spearman)
        for r in results
        if pos in r.per_position and r.per_position[pos].spearman is not None
    ]
    return max(scored, key=lambda t: t[1]) if scored else None


def _positions_without_ordering(result: EvalResult) -> list[str]:
    """Positions where this baseline admitted no Spearman slate — a pure position-mean fallback."""
    return [
        pos
        for pos in FANTASY_POSITIONS
        if pos in result.per_position and result.per_position[pos].spearman is None
    ]


def _metric_table(result: EvalResult) -> list[str]:
    lines = [
        "| position | n | MAE | RMSE | Spearman ρ | slates |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pos in FANTASY_POSITIONS:
        if pos not in result.per_position:
            continue
        m = result.per_position[pos]
        lines.append(
            f"| {pos} | {m.n:,} | {_fmt(m.mae)} | {_fmt(m.rmse)} | {_fmt(m.spearman, 3)} | "
            f"{m.spearman_slates} |"
        )
    return lines


def _calibration_table(result: EvalResult) -> list[str]:
    """Realized mean by predicted decile, one row per position, plus the mean predicted-vs-realized gap.

    The gap column is ``mean |pred_mean − realized_mean|`` over the deciles — a single number for
    "does the predicted level match the realized level", while the decile cells show whether realized
    points climb monotonically with the predicted decile (the ordering the FLEX call consumes).
    """
    deciles = list(range(1, 11))
    header = "| position | " + " | ".join(f"d{d}" for d in deciles) + " | gap |"
    lines = [header, "| --- |" + " ---: |" * (len(deciles) + 1)]
    for pos in FANTASY_POSITIONS:
        if pos not in result.per_position:
            continue
        cal = result.per_position[pos].calibration
        by_decile = {int(row.decile): row for row in cal.itertuples()}
        cells = []
        for d in deciles:
            cells.append(_fmt(by_decile[d].realized_mean, 1) if d in by_decile else "—")
        gap = (cal["pred_mean"] - cal["realized_mean"]).abs().mean() if not cal.empty else None
        lines.append(f"| {pos} | " + " | ".join(cells) + f" | {_fmt(gap)} |")
    return lines


def render_report(
    results: Sequence[EvalResult],
    *,
    seasons: Sequence[int],
    test_seasons: Sequence[int],
    scoring_keys: int,
    partitions: int,
    league_name: str,
    generated: str,
    frame_rows: int,
    cohort_rows: int,
    players: int,
) -> str:
    """The committed report. Pure over ``results`` so its prose can be pinned to its own tables."""
    by_name = {r.predictor: r for r in results}
    mae_bar = {pos: _best_by_mae(results, pos) for pos in FANTASY_POSITIONS}
    rho_bar = {pos: _best_by_rho(results, pos) for pos in FANTASY_POSITIONS}

    # Finding 2: where do accuracy (MAE) and ordering (ρ) name *different* winners?
    disagree = [
        pos
        for pos in FANTASY_POSITIONS
        if mae_bar[pos] and rho_bar[pos] and mae_bar[pos][0] != rho_bar[pos][0]
    ]

    parts: list[str] = [
        "# Model baselines — the recorded bar (Phase 9, ticket #28)",
        "",
        "> Generated by [`scripts/eval_baselines.py`](../scripts/eval_baselines.py) — regenerate "
        "with `./.venv/Scripts/python scripts/eval_baselines.py`. This file is a committed artifact; "
        "do not hand-edit it.",
        "",
        f"- **Generated (UTC):** {generated}",
        f"- **Train span:** {seasons[0]}–{seasons[-1]} · **test seasons (scored):** "
        f"{test_seasons[0]}–{test_seasons[-1]} walk-forward (train ≤ S-1, test S; 2016–2017 are lag "
        f"warm-up and never scored — spec Decision #6)",
        f"- **Lake backend:** `{LAKE_BACKEND}` · {partitions} partitions read",
        f"- **Scoring:** {league_name!r} live `scoring_settings`, {scoring_keys} keys (label "
        f"re-scored by the Phase 1 engine, never hard-coded)",
        f"- **Frame:** {frame_rows:,} rows, {cohort_rows:,} in the {len(FANTASY_POSITIONS)}-position "
        f"fantasy cohort, {players:,} players",
        "",
        "These are the numbers every later ticket (#29–#33) must beat, **per position, on both MAE "
        "and within-slate Spearman ρ** — a model that wins one metric while losing the other is a "
        "result to discuss, not a pass (spec, ticket #29).",
        "",
        "## Findings, measured",
        "",
    ]

    # Finding 1 — the bar, named per position by MAE.
    bar_bits = ", ".join(
        f"{pos} → {mae_bar[pos][0]} ({_fmt(mae_bar[pos][1])})" for pos in FANTASY_POSITIONS if mae_bar[pos]
    )
    parts.append(
        f"1. **The bar, per position (lowest held-out MAE).** {bar_bits}. No single baseline wins "
        "everywhere — which is the whole argument for per-position metrics (Decision #5): a pooled "
        "MAE would just report whichever baseline is best at QBs."
    )

    # Finding 2 — MAE and ρ can disagree.
    if disagree:
        bits = "; ".join(
            f"{pos}: MAE→{mae_bar[pos][0]} but ρ→{rho_bar[pos][0]}" for pos in disagree
        )
        parts.append(
            f"2. **Accuracy and ordering do not always name the same winner.** {bits}. Start/sit and "
            "waiver decisions consume the *ordering*, so ρ is not a tiebreaker for MAE — both are "
            "reported and both must be cleared."
        )
    else:
        parts.append(
            "2. **Accuracy and ordering agree here.** For every position the MAE-best baseline is also "
            "the ρ-best one — but both are still reported, because a model can win one and lose the "
            "other and that asymmetry is a result, not a pass."
        )

    # Finding 3 — LaggedExpectedPoints has no ordering signal for K/DEF (measured, not asserted).
    lep = by_name.get("LaggedExpectedPoints")
    if lep is not None:
        no_order = _positions_without_ordering(lep)
        if no_order:
            parts.append(
                f"3. **Expected points is a skill-position signal only.** `LaggedExpectedPoints` "
                f"admits no Spearman slate for {', '.join(no_order)} (ρ = —): `exp_points` is not "
                "collected for kickers or defenses, so there the baseline is a flat position-mean "
                "fallback with no ordering. #30 owns exactly those two positions, and this is why the "
                "bar is kept per position rather than pooled."
            )
        else:
            parts.append(
                "3. **Expected points carried an ordering signal for every position** in this run — "
                "including K and DEF, which was not expected; worth a look before #30 leans on it."
            )

    # Finding 4 — identical coverage, so the three are compared on one row universe.
    coverage = "; ".join(
        f"{pos} {by_name[list(by_name)[0]].per_position[pos].n:,}"
        for pos in FANTASY_POSITIONS
        if pos in by_name[list(by_name)[0]].per_position
    )
    parts.append(
        f"4. **All three baselines cover the same rows** (their shared position-mean fallback fills "
        f"every cold-start row), so the head-to-head is on one universe, not three. Scored rows per "
        f"position: {coverage}."
    )

    parts.append("")
    parts.append("## The bar — per-position MAE, RMSE and within-slate Spearman ρ")
    parts.append("")
    parts.append(
        "MAE/RMSE are over all pooled out-of-sample rows; ρ is the mean of the per-`(season, week)` "
        "slate rank correlations (one real weekly board per slate), and `slates` counts the boards "
        "that admitted one — a slate where a baseline predicts a single constant (a pure fallback) "
        "has no ordering and is skipped."
    )
    for result in results:
        parts.append("")
        parts.append(f"### {result.predictor}")
        parts.extend(_metric_table(result))

    parts.append("")
    parts.append("## Best baseline per position")
    parts.append("")
    parts.append("| position | lowest MAE | highest ρ |")
    parts.append("| --- | --- | --- |")
    for pos in FANTASY_POSITIONS:
        mae = f"{mae_bar[pos][0]} ({_fmt(mae_bar[pos][1])})" if mae_bar[pos] else "—"
        rho = f"{rho_bar[pos][0]} ({_fmt(rho_bar[pos][1], 3)})" if rho_bar[pos] else "—"
        parts.append(f"| {pos} | {mae} | {rho} |")

    parts.append("")
    parts.append("## Calibration — realized mean by predicted decile")
    parts.append("")
    parts.append(
        "Each cell is the realized mean custom points of the rows in that predicted decile (d1 = "
        "lowest predicted, d10 = highest). Realized points climbing monotonically across the deciles "
        "is the ordering property; `gap` = mean |predicted − realized| across deciles is the level "
        "mismatch."
    )
    for result in results:
        parts.append("")
        parts.append(f"### {result.predictor}")
        parts.extend(_calibration_table(result))

    parts.append("")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- entry point
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the naive baselines walk-forward, write the bar")
    ap.add_argument("--seasons", default=DEFAULT_SEASONS, help=f"train span (default: {DEFAULT_SEASONS})")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"report path (default: {DEFAULT_OUT})")
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
            f"could not load scoring_settings from league {LEAGUE_ID} "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 1

    partitions = int(len(lake_inventory()))
    print(
        f"Evaluating baselines over {seasons[0]}-{seasons[-1]} (test {DEFAULT_TEST_SEASONS[0]}-"
        f"{DEFAULT_TEST_SEASONS[-1]}) on backend {LAKE_BACKEND} ({partitions} partitions), "
        f"scoring {league.get('name')!r} ({len(scoring)} keys)…"
    )

    try:
        frame = build_training_frame(seasons, scoring)
    except ValueError as exc:
        print(f"could not build the frame — {exc}", file=sys.stderr)
        return 1

    cohort_rows = int(frame["position"].isin(FANTASY_POSITIONS).sum())
    results = [evaluate(factory(), frame, name=name) for name, factory in BASELINES]

    report = render_report(
        results,
        seasons=seasons,
        test_seasons=DEFAULT_TEST_SEASONS,
        scoring_keys=len(scoring),
        partitions=partitions,
        league_name=league.get("name", "?"),
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        frame_rows=len(frame),
        cohort_rows=cohort_rows,
        players=int(frame["player_id"].nunique()),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(f"Wrote {out} — {len(results)} baseline(s), {cohort_rows:,} fantasy-cohort rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

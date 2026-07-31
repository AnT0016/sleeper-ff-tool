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
contradicts). Every claim in the report is therefore *derived*: the scored season span comes from the
results rather than from the default constant, and "the baselines cover the same rows" is a comparison
across all of them rather than one baseline's count printed three times.
"""

from __future__ import annotations

import argparse
import logging
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

#: A position is called out in finding #3 when the baseline gave a non-constant ordering on fewer
#: than this fraction of the boards it was scored on. The cut is a judgement call, but the data is
#: not close to it: on the real lake ``LaggedExpectedPoints`` orders 94% of skill-position boards and
#: 5%/0% of K/DEF ones, so any threshold in 0.1-0.9 names the same two positions.
_THIN_ORDERING = 0.5


# --------------------------------------------------------------------------- log capture
class _Capture(logging.Handler):
    """Records the run's log lines, so the report can count and quote its warnings.

    The standing bar is zero unexpected warnings on real data (profile #27 §7). This run builds the
    frame *and* walks the splits, and ``walk_forward_splits`` warns when it skips a season — the one
    event that would make a committed bar quietly narrower than its header claims.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[tuple[int, str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.levelname, record.name, record.getMessage()))


# --------------------------------------------------------------------------- rendering helpers
def _fmt(value: float | None, nd: int = 2) -> str:
    """A metric, or an em-dash where it is undefined (a Spearman with no admissible slate)."""
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "—"
    return f"{value:.{nd}f}"


def _season_span(seasons: Sequence[int]) -> str:
    """The seasons actually scored, contiguous ones as a range — never a default constant."""
    if not seasons:
        return "none"
    if list(seasons) == list(range(seasons[0], seasons[-1] + 1)):
        return f"{seasons[0]}–{seasons[-1]}"
    return ", ".join(str(s) for s in seasons)


def _coverage_by_position(results: Sequence[EvalResult]) -> dict[str, dict[str, int]]:
    """``{position: {predictor: scored rows}}`` — what finding #4 claims, measured across all of them.

    An earlier draft printed the *first* result's counts under the sentence "all three cover the same
    rows", so the claim survived its own counterexample. Comparing the counts is the only version of
    that sentence worth committing.
    """
    out: dict[str, dict[str, int]] = {}
    for pos in FANTASY_POSITIONS:
        ns = {r.predictor: r.per_position[pos].n for r in results if pos in r.per_position}
        if ns:
            out[pos] = ns
    return out


def _ordering_coverage(result: EvalResult) -> dict[str, tuple[int, int]]:
    """``{position: (boards actually ordered, boards scored)}`` for one baseline.

    A flat prediction scores rho = 0 rather than being excused (``model.evaluate.spearman``), so every
    baseline's rho is a mean over the same boards. This is the number that says how many of them it
    earned rather than defaulted.
    """
    return {
        pos: (m.spearman_ordered_slates, m.spearman_slates) for pos, m in result.per_position.items()
    }


def _thinly_ordered(result: EvalResult) -> list[tuple[str, int, int]]:
    """Positions this baseline ordered on fewer than :data:`_THIN_ORDERING` of its scored boards."""
    return [
        (pos, ordered, total)
        for pos, (ordered, total) in _ordering_coverage(result).items()
        if total and ordered / total < _THIN_ORDERING
    ]


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


def _metric_table(result: EvalResult) -> list[str]:
    lines = [
        "| position | n | MAE | RMSE | Spearman ρ | slates | ordered |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pos in FANTASY_POSITIONS:
        if pos not in result.per_position:
            continue
        m = result.per_position[pos]
        lines.append(
            f"| {pos} | {m.n:,} | {_fmt(m.mae)} | {_fmt(m.rmse)} | {_fmt(m.spearman, 3)} | "
            f"{m.spearman_slates} | {m.spearman_ordered_slates} |"
        )
    return lines


def _log_block(records: Sequence[tuple[int, str, str, str]], *, min_level: int) -> str:
    """The captured log at or above ``min_level`` as a fenced block, or a stated absence."""
    lines = [f"{lvl} {name}: {msg}" for level, lvl, name, msg in records if level >= min_level]
    return "```\n" + "\n".join(lines) + "\n```" if lines else "_None._"


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
    scoring_keys: int,
    partitions: int,
    league_name: str,
    generated: str,
    frame_rows: int,
    cohort_rows: int,
    players: int,
    records: Sequence[tuple[int, str, str, str]] = (),
) -> str:
    """The committed report. Pure over ``results`` so its prose can be pinned to its own tables.

    There is deliberately no ``test_seasons`` parameter: the scored span is read back out of the
    results. Passing it in is how the header came to print ``DEFAULT_TEST_SEASONS`` while a narrowed
    ``--seasons`` had actually scored two of them.
    """
    by_name = {r.predictor: r for r in results}
    scored_seasons = sorted({s for r in results for s in r.test_seasons})
    mae_bar = {pos: _best_by_mae(results, pos) for pos in FANTASY_POSITIONS}
    rho_bar = {pos: _best_by_rho(results, pos) for pos in FANTASY_POSITIONS}
    n_warnings = sum(1 for level, *_ in records if level >= logging.WARNING)

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
        f"{_season_span(scored_seasons)} walk-forward (train ≤ S-1, test S; the earliest seasons are "
        f"lag warm-up and are never scored — spec Decision #6). Read back from the results, so a "
        f"narrowed run cannot advertise a span it did not score.",
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

    # Finding 3 — how often expected points actually ordered a board, per position (measured).
    lep = by_name.get("LaggedExpectedPoints")
    if lep is not None:
        thin = _thinly_ordered(lep)
        cov = _ordering_coverage(lep)
        if thin:
            thin_bits = ", ".join(f"**{pos}** {o}/{t}" for pos, o, t in thin)
            thin_names = {pos for pos, _o, _t in thin}
            rest = ", ".join(
                f"{pos} {o}/{t}" for pos, (o, t) in cov.items() if pos not in thin_names
            )
            parts.append(
                f"3. **Expected points is a skill-position signal.** Boards `LaggedExpectedPoints` "
                f"gave a real (non-constant) ordering on: {thin_bits} — against {rest} for the rest. "
                "`nflverse_ff_opp` covers offensive skill players, so on those positions the baseline "
                "is the flat position-mean fallback on all but a handful of boards; the few "
                "exceptions are enough to make a ρ printable while carrying no ordering anyone could "
                "use, which is what the `ordered` column exists to show. Its ρ there is a zero it "
                "earned by declining to order, not an ordering that failed. #30 owns those positions."
            )
        else:
            parts.append(
                "3. **Expected points ordered a board for every position** in this run — including K "
                "and DEF, which was not expected; worth a look before #30 leans on it."
            )

    # Finding 4 — coverage compared across every baseline, never one baseline's count printed N times.
    cov_by_pos = _coverage_by_position(results)
    mismatched = {pos: ns for pos, ns in cov_by_pos.items() if len(set(ns.values())) > 1}
    if mismatched:
        bits = "; ".join(
            f"{pos}: " + ", ".join(f"{name} {n:,}" for name, n in ns.items())
            for pos, ns in mismatched.items()
        )
        parts.append(
            f"4. **The baselines do NOT all cover the same rows.** {bits}. The head-to-head is "
            "therefore not on one universe and the MAE column is not directly comparable at those "
            "positions — a baseline scored on fewer rows may simply have skipped the hard ones."
        )
    else:
        coverage = "; ".join(f"{pos} {next(iter(ns.values())):,}" for pos, ns in cov_by_pos.items())
        parts.append(
            f"4. **All {len(results)} baselines cover the same rows** — compared per position, not "
            f"assumed (their shared position-mean fallback fills every cold-start row), so the "
            f"head-to-head is on one universe. Scored rows per position: {coverage}."
        )

    # Finding 5 — the standing zero-unexpected-warnings bar, counted from the captured log.
    parts.append(
        f"5. **{'Zero' if not n_warnings else str(n_warnings)} warning"
        f"{'' if n_warnings == 1 else 's'} on real data.** The frame build and the walk over the "
        f"splits emitted **{n_warnings}** WARNING-level line(s) (verbatim below). This matters here "
        "beyond the standing bar: `walk_forward_splits` warns when it *skips* a test season, so a "
        "silent count is the difference between a bar over the advertised span and a bar over less."
    )

    parts.append("")
    parts.append("## The bar — per-position MAE, RMSE and within-slate Spearman ρ")
    parts.append("")
    parts.append(
        "MAE/RMSE are over all pooled out-of-sample rows; ρ is the mean of the per-`(season, week)` "
        "slate rank correlations (one real weekly board per slate). `slates` counts the boards scored "
        "and `ordered` how many of those the baseline gave a **non-constant** prediction on: a board "
        "it answered with one flat value scores ρ = 0 rather than being excused, so every baseline's "
        "ρ is a mean over the same boards and a low `ordered` says the ρ was defaulted, not earned. "
        "(A board where every player scored the same admits no ordering for anyone and is the one "
        "case still skipped.)"
    )
    for result in results:
        parts.append("")
        parts.append(f"### {result.predictor}")
        parts.extend(_metric_table(result))

    parts.append("")
    parts.append("## Best baseline per position")
    parts.append("")
    slate_counts = {
        pos: {r.per_position[pos].spearman_slates for r in results if pos in r.per_position}
        for pos in FANTASY_POSITIONS
    }
    uneven = sorted(pos for pos, counts in slate_counts.items() if len(counts) > 1)
    parts.append(
        "Both columns compare across baselines, which is only legitimate while they are scored on the "
        + (
            f"same boards — and at {', '.join(uneven)} they are **not** (`slates` differs), so read "
            "the ρ column there as indicative only."
            if uneven
            else "same boards: `slates` is identical across baselines at every position here, so the "
            "ρ column is a like-for-like comparison."
        )
    )
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
    flat = sorted(
        {
            f"{r.predictor}/{pos}"
            for r in results
            for pos, (ordered, total) in _ordering_coverage(r).items()
            if total and ordered / total < _THIN_ORDERING
        }
    )
    if flat:
        parts.append("")
        parts.append(
            f"**Read `gap` next to `ordered`.** A baseline that predicts the position mean scores a "
            f"near-zero gap *by construction* — it is the mean — while its deciles are noise. That is "
            f"the case for {', '.join(flat)} below, whose small gaps are the arithmetic of a flat "
            f"prediction rather than evidence of calibration."
        )
    for result in results:
        parts.append("")
        parts.append(f"### {result.predictor}")
        parts.extend(_calibration_table(result))

    parts.append("")
    parts.append("## Warnings (verbatim)")
    parts.append("")
    parts.append(
        "The standing bar is zero unexpected warnings on real data. A skipped test season appears "
        "here, so the span in the header above and the work actually done cannot drift apart silently."
    )
    parts.append("")
    parts.append(_log_block(records, min_level=logging.WARNING))

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

    # Capture the build's *and* the walk's log (INFO+) so the report can count and quote warnings.
    # basicConfig is NOT called: ours is the only handler, so the report gets the sole copy. The walk
    # is inside the capture on purpose — walk_forward_splits warns when it skips a test season, and
    # that warning is the only signal that the committed bar covers less than its header says.
    capture = _Capture()
    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        frame = build_training_frame(seasons, scoring)
        cohort_rows = int(frame["position"].isin(FANTASY_POSITIONS).sum())
        results = [evaluate(factory(), frame, name=name) for name, factory in BASELINES]
    except ValueError as exc:
        print(f"could not build the frame — {exc}", file=sys.stderr)
        return 1
    finally:
        root.removeHandler(capture)
        root.setLevel(prior_level)

    report = render_report(
        results,
        seasons=seasons,
        scoring_keys=len(scoring),
        partitions=partitions,
        league_name=league.get("name", "?"),
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        frame_rows=len(frame),
        cohort_rows=cohort_rows,
        players=int(frame["player_id"].nunique()),
        records=capture.records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    n_warnings = sum(1 for level, *_ in capture.records if level >= logging.WARNING)
    scored = sorted({s for r in results for s in r.test_seasons})
    print(
        f"Wrote {out} — {len(results)} baseline(s), {cohort_rows:,} fantasy-cohort rows, "
        f"test seasons scored {_season_span(scored)}, {n_warnings} warning(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

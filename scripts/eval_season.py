"""Score the draft-value model against the prior-season-total bar and commit the result (#31).

    ./.venv/Scripts/python scripts/eval_season.py
    ./.venv/Scripts/python scripts/eval_season.py --seasons 2016-2025 --out docs/model-draft-baseline.md

The draft path has no historical Sleeper baseline (``sleeper_proj_season`` is forward-only from 2026),
so — exactly as #28 did for the weekly path — "better" has to be measured against a bar we define and
write down. Here the bar is :class:`~model.season.PriorSeasonTotal` ("next year looks like last year"),
and the graded metric is **within-``(season, position)`` Spearman ρ**, because draft value is a ranking
problem (spec acceptance #2). This script builds the season frame once (against whatever ``LAKE_BACKEND``
points at — the local backfill by default), evaluates the bar, the fielded model, and an *ungated* ridge
walk-forward over 2018-2025, and writes :data:`DEFAULT_OUT` as a **committed artifact**.

The ungated ridge is evaluated only so the report can *show*, from its own table, that a ridge orders K
and DEF worse than last-year's-total — the measured justification for the fielded model deferring those
positions to the baseline, rather than an assertion. :func:`render_report` is pure over the evaluated
results, so ``tests/test_model_season.py`` pins that its prose follows the numbers it cites; the label is
re-scored live from the active league's ``scoring_settings`` (the Phase 1 engine), never hard-coded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# `python scripts/eval_season.py` puts scripts/ at the front of sys.path, where scripts/collect.py
# shadows the installed `collect` package the assembler imports. Drop the script directory.
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from collect import runner
from model.evaluate import FANTASY_POSITIONS
from model.frame import build_season_frame
from model.season import (
    PriorSeasonTotal,
    SeasonEvalResult,
    SeasonModel,
    evaluate_season,
)
from sleeper import client
from sleeper.config import LEAGUE_ID
from store.lake import LAKE_BACKEND, lake_inventory

DEFAULT_SEASONS = "2016-2025"
DEFAULT_OUT = "docs/model-draft-baseline.md"

#: The predictors, in report order: the bar, the fielded model, the ungated diagnostic.
_BAR = "PriorSeasonTotal"
_MODEL = "SeasonModel"
_UNGATED = "SeasonModel (ungated ridge)"

#: Two ρ values within this are treated as the same ordering — a deferred position predicts exactly the
#: baseline, so its ρ is identical up to float noise.
_TIE_EPS = 1e-6


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


def _season_span(seasons: Sequence[int]) -> str:
    if not seasons:
        return "none"
    if list(seasons) == list(range(seasons[0], seasons[-1] + 1)):
        return f"{seasons[0]}–{seasons[-1]}"
    return ", ".join(str(s) for s in seasons)


def _rho(result: SeasonEvalResult, pos: str) -> float | None:
    m = result.per_position.get(pos)
    return m.spearman if m else None


def _mae(result: SeasonEvalResult, pos: str) -> float | None:
    m = result.per_position.get(pos)
    return m.mae if m else None


def _verdict(model_rho: float | None, base_rho: float | None) -> str:
    """Model vs bar on ordering, derived from the two numbers — never asserted."""
    if model_rho is None or base_rho is None:
        return "—"
    if model_rho > base_rho + _TIE_EPS:
        return "win"
    if model_rho < base_rho - _TIE_EPS:
        return "loss"
    return "tie (deferred)"


def _metric_table(result: SeasonEvalResult) -> list[str]:
    lines = [
        "| position | n | MAE | RMSE | Spearman ρ | slates |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pos in FANTASY_POSITIONS:
        m = result.per_position.get(pos)
        if m is None:
            continue
        lines.append(
            f"| {pos} | {m.n:,} | {_fmt(m.mae)} | {_fmt(m.rmse)} | {_fmt(m.spearman, 3)} | "
            f"{m.slates} |"
        )
    return lines


def _log_block(records: Sequence[tuple[int, str, str, str]], *, min_level: int) -> str:
    lines = [f"{lvl} {name}: {msg}" for level, lvl, name, msg in records if level >= min_level]
    return "```\n" + "\n".join(lines) + "\n```" if lines else "_None._"


def render_report(
    results: dict[str, SeasonEvalResult],
    *,
    seasons: Sequence[int],
    scoring_keys: int,
    partitions: int,
    league_name: str,
    generated: str,
    frame_rows: int,
    players: int,
    rookie_share: dict[int, float],
    records: Sequence[tuple[int, str, str, str]] = (),
) -> str:
    """The committed report. Pure over ``results`` so its prose can be pinned to its own tables."""
    bar, model, ungated = results[_BAR], results[_MODEL], results.get(_UNGATED)
    scored_seasons = sorted(model.test_seasons)
    n_warnings = sum(1 for level, *_ in records if level >= logging.WARNING)

    wins = [p for p in FANTASY_POSITIONS if _verdict(_rho(model, p), _rho(bar, p)) == "win"]
    deferred = [
        p for p in FANTASY_POSITIONS if _verdict(_rho(model, p), _rho(bar, p)) == "tie (deferred)"
    ]
    ungated_losses = (
        [
            p
            for p in FANTASY_POSITIONS
            if ungated and _verdict(_rho(ungated, p), _rho(bar, p)) == "loss"
        ]
        if ungated
        else []
    )

    parts: list[str] = [
        "# Draft-value model — the recorded bar (Phase 9, ticket #31)",
        "",
        "> Generated by [`scripts/eval_season.py`](../scripts/eval_season.py) — regenerate with "
        "`./.venv/Scripts/python scripts/eval_season.py`. This file is a committed artifact; do not "
        "hand-edit it.",
        "",
        f"- **Generated (UTC):** {generated}",
        f"- **Train span:** {seasons[0]}–{seasons[-1]} · **test seasons (scored):** "
        f"{_season_span(scored_seasons)} walk-forward (train ≤ S-1, test S; the earliest season is a "
        f"lag warm-up and is never scored). Read back from the results.",
        f"- **Lake backend:** `{LAKE_BACKEND}` · {partitions} partitions read",
        f"- **Scoring:** {league_name!r} live `scoring_settings`, {scoring_keys} keys (season label = "
        f"the sum of the engine-scored weekly totals, never hard-coded)",
        f"- **Season frame:** {frame_rows:,} rows, {players:,} players",
        "",
        "Draft value is a **ranking** problem — a VOR/tier board consumes the order of players within a "
        "position, not the level — so the graded metric is **within-`(season, position)` Spearman ρ**. "
        "MAE is reported too, but a model that wins MAE while losing ρ has not earned the board (spec "
        "acceptance #2).",
        "",
        "## Findings, measured",
        "",
    ]

    # Finding 1 — where the model beats the bar, derived from the tables.
    win_bits = ", ".join(
        f"{p} ρ {_fmt(_rho(model, p), 3)} vs {_fmt(_rho(bar, p), 3)}" for p in wins
    )
    defer_note = (
        f" It **defers to the bar** at {', '.join(deferred)} (a tie by construction), carrying no "
        "usage signal — see finding 2."
        if deferred
        else ""
    )
    parts.append(
        f"1. **The model beats last-year's-total on ordering where it has a usage edge.** Positions "
        f"won on ρ: {win_bits or 'none'}.{defer_note} Usage — snap, target and rush share, expected "
        "points — is the model's whole edge over a points-only projection, so it wins exactly where "
        "usage exists."
    )

    # Finding 2 — the ungated ridge is worse at the deferred positions: the measured reason to defer.
    if ungated_losses:
        loss_bits = ", ".join(
            f"{p} ρ {_fmt(_rho(ungated, p), 3)} vs {_fmt(_rho(bar, p), 3)}" for p in ungated_losses
        )
        defer_names = ", ".join(ungated_losses)
        parts.append(
            f"2. **Why {defer_names} defers, measured.** An *ungated* ridge — one fielded at every "
            f"position — orders {loss_bits} **worse** than last-year's-total: with no usage signal to "
            "add, a ridge only shrinks the one feature that matters (the prior total) and forfeits the "
            f"ordering. So the fielded model defers {defer_names} to the bar rather than ranking it with "
            "a known-worse orderer; a proper component model for it is #30's job."
        )
    else:
        parts.append(
            "2. **The ungated ridge did not lose at any position in this run** — so the fielded model "
            "models every position; worth a look before trusting the deferral logic on this data."
        )

    # Finding 3 — rookies are a large, every-year cohort (measured share).
    share_bits = " · ".join(
        f"{season} {share * 100:.0f}%" for season, share in sorted(rookie_share.items())
    )
    parts.append(
        f"3. **Rookies are a large cohort, every year, not an edge case.** Share of the fantasy season "
        f"frame flagged `is_rookie` (no prior season in the window): {share_bits}. They carry null "
        "prior features and are predicted at the position's learned level; the earliest season (100% "
        "by arithmetic) is the warm-up and never trains the model."
    )

    # Finding 4 — the standing zero-warnings bar.
    parts.append(
        f"4. **{'Zero' if not n_warnings else str(n_warnings)} warning"
        f"{'' if n_warnings == 1 else 's'} on real data.** The frame build and the walk over the "
        f"splits emitted **{n_warnings}** WARNING-level line(s) (verbatim below)."
    )

    parts.append("")
    parts.append("## The bar vs the model — per-position MAE and within-slate Spearman ρ")
    parts.append("")
    parts.append(
        "One `(season, position)` board per test season; ρ is the mean of their rank correlations. The "
        "fielded `SeasonModel` fields a ridge only where usage exists and predicts the bar elsewhere; "
        "the ungated ridge is the diagnostic behind finding 2."
    )
    for name in (_BAR, _MODEL, _UNGATED):
        if name not in results:
            continue
        parts.append("")
        parts.append(f"### {name}")
        parts.extend(_metric_table(results[name]))

    parts.append("")
    parts.append("## Model vs bar — the headline, per position")
    parts.append("")
    parts.append("| position | bar ρ | model ρ | verdict | bar MAE | model MAE |")
    parts.append("| --- | ---: | ---: | --- | ---: | ---: |")
    for pos in FANTASY_POSITIONS:
        parts.append(
            f"| {pos} | {_fmt(_rho(bar, pos), 3)} | {_fmt(_rho(model, pos), 3)} | "
            f"{_verdict(_rho(model, pos), _rho(bar, pos))} | {_fmt(_mae(bar, pos))} | "
            f"{_fmt(_mae(model, pos))} |"
        )

    parts.append("")
    parts.append("## Warnings (verbatim)")
    parts.append("")
    parts.append(
        "The standing bar is zero unexpected warnings on real data. A skipped test season appears here, "
        "so the span in the header and the work actually done cannot drift apart silently."
    )
    parts.append("")
    parts.append(_log_block(records, min_level=logging.WARNING))
    parts.append("")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- entry point
def _rookie_share(frame) -> dict[int, float]:
    """Share of the fantasy-cohort season rows flagged ``is_rookie``, per season."""
    cohort = frame[frame["position"].isin(FANTASY_POSITIONS)]
    grouped = cohort.groupby("season")["is_rookie"]
    return {int(s): float(v) for s, v in grouped.mean().items()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the draft-value model against the bar, write it")
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
            f"could not load scoring_settings from league {LEAGUE_ID} ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 1

    partitions = int(len(lake_inventory()))
    print(
        f"Evaluating the draft model over {seasons[0]}-{seasons[-1]} on backend {LAKE_BACKEND} "
        f"({partitions} partitions), scoring {league.get('name')!r} ({len(scoring)} keys)…"
    )

    capture = _Capture()
    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        frame = build_season_frame(seasons, scoring)
        results = {
            _BAR: evaluate_season(PriorSeasonTotal(), frame, name=_BAR),
            _MODEL: evaluate_season(SeasonModel(), frame, name=_MODEL),
            _UNGATED: evaluate_season(SeasonModel(require_usage=False), frame, name=_UNGATED),
        }
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
        players=int(frame["player_id"].nunique()),
        rookie_share=_rookie_share(frame),
        records=capture.records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    n_warnings = sum(1 for level, *_ in capture.records if level >= logging.WARNING)
    scored = sorted(results[_MODEL].test_seasons)
    print(
        f"Wrote {out} — bar + model + ungated, {len(frame):,} season rows, "
        f"test seasons scored {_season_span(scored)}, {n_warnings} warning(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

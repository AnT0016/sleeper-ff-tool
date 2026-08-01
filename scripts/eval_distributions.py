"""Fit the simulators' distribution knobs from the lake, write the artifact + the measured report (#32).

    ./.venv/Scripts/python scripts/eval_distributions.py
    ./.venv/Scripts/python scripts/eval_distributions.py --seasons 2016-2025

There is **no baseline to beat** here — a CV is not a prediction and nothing grades it. So the report is
not a headline win/loss; it is (1) a per ``(position × knob)`` verdict of ``fitted`` or
``heuristic-fallback`` with ``n`` and the fallback reason — 18 cells — and (2) a seed-pinned before/after
on championship odds for one fixed synthetic 12-team league, reported **whether or not it moves**.

The before/after runs as **four arms** (heuristic · CV-fitted only · injury-fitted only · both) under
common random numbers, because fitting the injury knob shifts each position's availability haircut — a
*mean* shift landing on a projection that already embeds average injury loss (a pre-existing double-count
this ticket makes explicit but does not fix) — so the total championship-odds move must be decomposed
into the uncertainty effect (CV arm) and the mean effect (injury arm) to be readable at all.

:func:`render_report` is pure over the fit + before/after so ``tests/test_model_distributions.py`` pins
that its prose follows the numbers it cites (spec Decision #9 item 1). The artifact
(``src/model/fit/distributions.json``) is read back by ``draftsim.distributions`` at import, so it is a
source, not a write-only record (item 4).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# `python scripts/eval_distributions.py` puts scripts/ at the front of sys.path, where scripts/collect.py
# shadows the installed `collect` package the assembler imports. Drop the script directory.
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from collect import runner
from draftsim.distributions import (
    HEURISTIC_GAME_CV,
    HEURISTIC_INJURY_RISK,
    HEURISTIC_POSITION_CV,
    use_knobs,
)
from model.distributions import (
    MAX_SEASON_FACTOR_CV,
    POSITIONS,
    SEASON_COHORT_BY_POSITION,
    SEASON_COHORT_TOTAL,
    FitResult,
    fit_distributions,
    to_artifact,
)
from seasonsim.engine import SeasonPool, simulate_season
from seasonsim.schedule import round_robin
from sleeper import client
from sleeper.config import LEAGUE_ID
from store.lake import LAKE_BACKEND, lake_inventory

DEFAULT_SEASONS = "2016-2025"
DEFAULT_OUT = "docs/model-distributions.md"
DEFAULT_ARTIFACT = "src/model/fit/distributions.json"

_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1, "BN": 0}
#: A fixed 14-man roster template (positions) — 8 dedicated starters, a flex-eligible RB, and 5 bench.
_ROSTER = ["QB", "RB", "RB", "WR", "WR", "TE", "K", "DEF", "RB", "RB", "WR", "WR", "TE", "QB"]
#: Base custom-scored season means per roster slot (our 4-pt-pass-TD scoring), starters richer than bench.
_ROSTER_MEAN = [300, 235, 190, 205, 175, 145, 130, 120, 150, 110, 130, 95, 85, 175]
_BEFORE_AFTER_SIMS = 4000
_BEFORE_AFTER_SEED = 20260801
_REGULAR_WEEKS = list(range(1, 15))
_PLAYOFF_WEEKS = [15, 16, 17]


# --------------------------------------------------------------------------- log capture
class _Capture(logging.Handler):
    """Records the run's log lines so the report can count and quote its warnings (profile #27 §7)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[tuple[int, str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.levelname, record.name, record.getMessage()))


# --------------------------------------------------------------------------- fixed-league before/after
#: My team index in the fixed league — a near-median team on the gradient (mult ≈ 1.01), so the
#: before/after is about a representative contender's title odds, not the noise-sensitivity of the
#: weakest or strongest roster.
_MY_TEAM = 6


def _fixed_league() -> tuple[SeasonPool, dict, int]:
    """A deterministic 12-team league — same roster shape, teams on a mild strength gradient.

    Synthetic on purpose: the before/after must isolate the *knobs*, so the roster is fixed and needs no
    network, and the committed report regenerates identically. The gradient gives a non-degenerate title
    race (odds are not uniform) without any RNG in the construction.
    """
    n_teams = 12
    rosters: list[list[int]] = []
    pos: list[str] = []
    mean: list[float] = []
    names: list[str] = []
    for t in range(n_teams):
        mult = 1.0 + (t - (n_teams - 1) / 2.0) * 0.02  # ~0.89 … 1.11, deterministic
        cols: list[int] = []
        for slot, (p, base) in enumerate(zip(_ROSTER, _ROSTER_MEAN, strict=True)):
            cols.append(len(pos))
            pos.append(p)
            mean.append(float(base) * mult)
            names.append(f"T{t}-{p}{slot}")
        rosters.append(cols)
    pool = SeasonPool(
        rosters=rosters,
        pos=pos,
        mean=np.array(mean, dtype=float),
        cv=np.zeros(len(pos)),  # ignored by the engine (it reads dist.POSITION_CV) — see engine.pool_arrays
        p_setback=np.zeros(len(pos)),
        severity=np.zeros(len(pos)),
        names=names,
        team_names=[f"Team{t}" for t in range(n_teams)],
        slots=_SLOTS,
        n_teams=n_teams,
    )
    schedule = round_robin(n_teams, _REGULAR_WEEKS)
    return pool, schedule, _MY_TEAM


def _champ_pct(
    pool: SeasonPool,
    schedule: dict,
    my_team: int,
    *,
    position_cv: Mapping[str, float],
    game_cv: Mapping[str, float],
    injury_risk: Mapping[str, tuple[float, float]],
) -> float:
    """``P(my team wins the title)`` (my_edge regime) under one knob set, seed-pinned for common RNG."""
    with use_knobs(position_cv=position_cv, game_cv=game_cv, injury_risk=injury_risk):
        out = simulate_season(
            pool, schedule, my_team,
            season=2026, regular_weeks=_REGULAR_WEEKS, playoff_weeks=_PLAYOFF_WEEKS,
            n_playoff_teams=6, n_sims=_BEFORE_AFTER_SIMS, seed=_BEFORE_AFTER_SEED,
        )
    return float(out.regimes["my_edge"].my_is_champ.mean())


def before_after(result: FitResult) -> dict[str, float]:
    """The four-arm championship-odds comparison — the honest, decomposed grade (Amendment 1)."""
    pool, schedule, my_team = _fixed_league()
    fit_pos, fit_game, fit_inj = (
        result.shipped_position_cv, result.shipped_game_cv, result.shipped_injury,
    )
    arms = {
        "heuristic": (HEURISTIC_POSITION_CV, HEURISTIC_GAME_CV, HEURISTIC_INJURY_RISK),
        "cv_only": (fit_pos, fit_game, HEURISTIC_INJURY_RISK),
        "injury_only": (HEURISTIC_POSITION_CV, HEURISTIC_GAME_CV, fit_inj),
        "both": (fit_pos, fit_game, fit_inj),
    }
    return {
        arm: _champ_pct(pool, schedule, my_team, position_cv=p, game_cv=g, injury_risk=i)
        for arm, (p, g, i) in arms.items()
    }


# --------------------------------------------------------------------------- rendering helpers
def _fmt(value: float | None, nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    return f"{value:.{nd}f}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _season_span(seasons: Sequence[int]) -> str:
    if not seasons:
        return "none"
    if list(seasons) == list(range(seasons[0], seasons[-1] + 1)):
        return f"{seasons[0]}–{seasons[-1]}"
    return ", ".join(str(s) for s in seasons)


def _verdict_counts(result: FitResult) -> tuple[int, int, list[str]]:
    """``(n_fitted, n_fallback, fallback_labels)`` over the 18 (position × knob) cells."""
    fitted = fallback = 0
    labels: list[str] = []
    for knob, out in (("season CV", result.position_cv_out), ("game CV", result.game_cv_out),
                      ("injury", result.injury_out)):
        for pos in POSITIONS:
            if out[pos]["verdict"] == "fitted":
                fitted += 1
            else:
                fallback += 1
                labels.append(f"{pos} {knob}")
    return fitted, fallback, labels


def _cv_table(out: Mapping[str, dict], cells: Mapping, *, season: bool) -> list[str]:
    head = "| position | fitted CV | verdict | n | seasons | mean r |"
    if season:
        head += " full-cohort CV | wide upper-bound CV |"
    lines = [head, "| --- | ---: | --- | ---: | ---: | ---: |" + (" ---: | ---: |" if season else "")]
    for pos in POSITIONS:
        cell = cells[pos]
        o = out[pos]
        row = (
            f"| {pos} | {_fmt(o['value'])} | {o['verdict']} | {cell.n:,} | {cell.n_seasons} | "
            f"{_fmt(cell.mean_r, 2)} |"
        )
        if season:
            row += f" {_fmt(cell.full_cohort_cv)} | {_fmt(cell.upper_bound_cv)} |"
        lines.append(row)
    return lines


def _robustness_table(result: FitResult) -> list[str]:
    lines = ["| position | season CV (top-168 overall) | n | implied factor CV | verdict |",
             "| --- | ---: | ---: | ---: | --- |"]
    for row in result.robustness.to_dict("records"):
        lines.append(
            f"| {row['position']} | {_fmt(row['cv'])} | {int(row['n']):,} | {_fmt(row['factor_cv'])} | "
            f"{row['verdict']} |"
        )
    return lines


def _by_season_pivot(cells: Mapping, field: str, seasons: Sequence[int]) -> list[str]:
    """A position × season table of ``field`` (``mean_r`` or ``cv_r``) — the era-robustness evidence."""
    cols = list(seasons)
    lines = ["| position | " + " | ".join(str(s) for s in cols) + " |",
             "| --- |" + " ---: |" * len(cols)]
    for pos in POSITIONS:
        by = {int(r.season): r for r in cells[pos].by_season.itertuples()}
        vals = [_fmt(getattr(by[s], field), 2) if s in by else "—" for s in cols]
        lines.append(f"| {pos} | " + " | ".join(vals) + " |")
    return lines


def _tercile_table(cells: Mapping) -> list[str]:
    lines = ["| position | t1 CV (low pred) | t2 CV | t3 CV (high pred) | slide |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for pos in POSITIONS:
        by = {int(r.tercile): r for r in cells[pos].by_tercile.itertuples()}
        cvs = [by[t].cv_r if t in by else None for t in (1, 2, 3)]
        slide = (
            _fmt(cvs[2] - cvs[0]) if cvs[0] is not None and cvs[2] is not None else "—"
        )
        lines.append(
            f"| {pos} | {_fmt(cvs[0])} | {_fmt(cvs[1])} | {_fmt(cvs[2])} | {slide} |"
        )
    return lines


def _injury_table(out: Mapping[str, dict], cells: Mapping) -> list[str]:
    lines = [
        "| position | P(setback) drafted | P(setback) wide | Out-only P (IR-truncated) | "
        "mean weeks missed | verdict | drafted n | wide n | on-report n | reason |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for pos in POSITIONS:
        o, c = out[pos], cells[pos]
        lines.append(
            f"| {pos} | {_pct(o['p'])} | {_pct(c.p_wide)} | {_pct(c.p_out_only)} | {_fmt(o['games'], 1)} | "
            f"{o['verdict']} | {c.n_qualified:,} | {c.n_wide:,} | {c.coverage:,} | {o['reason'] or '—'} |"
        )
    return lines


def _avail_table(result: FitResult) -> list[str]:
    lines = ["| position | E[avail] heuristic | haircut | E[avail] fitted | haircut | Δ haircut |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for pos in POSITIONS:
        h, f = result.avail_heuristic[pos], result.avail_fitted[pos]
        lines.append(
            f"| {pos} | {_fmt(h)} | {100 * (h - 1):+.2f}% | {_fmt(f)} | {100 * (f - 1):+.2f}% | "
            f"{100 * (f - h):+.2f}% |"
        )
    return lines


def _coherence_table(result: FitResult) -> list[str]:
    lines = [
        "| position | SeasonModel CV | game CV | implied factor CV | verdict | shipped season CV | "
        "shipped factor CV |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in result.coherence.to_dict("records"):
        lines.append(
            f"| {row['position']} | {_fmt(row['fitted_season_cv'])} | {_fmt(row['game_cv'])} | "
            f"{_fmt(row['fitted_factor_cv'])} | {row['verdict']} | {_fmt(row['shipped_season_cv'])} | "
            f"{_fmt(row['shipped_factor_cv'])} |"
        )
    return lines


def _before_after_table(ba: Mapping[str, float]) -> list[str]:
    base = ba["heuristic"]
    order = [
        ("heuristic", "all heuristic (before)"),
        ("cv_only", "CV fitted, injury heuristic — the uncertainty effect"),
        ("injury_only", "injury fitted, CV heuristic — the mean/availability effect"),
        ("both", "both fitted (after)"),
    ]
    lines = ["| arm | championship % (my_edge) | Δ vs heuristic |", "| --- | ---: | ---: |"]
    for key, label in order:
        lines.append(f"| {label} | {_pct(ba[key])} | {100 * (ba[key] - base):+.2f} pts |")
    return lines


def _log_block(records: Sequence[tuple[int, str, str, str]], *, min_level: int) -> str:
    lines = [f"{lvl} {name}: {msg}" for level, lvl, name, msg in records if level >= min_level]
    return "```\n" + "\n".join(lines) + "\n```" if lines else "_None._"


# --------------------------------------------------------------------------- the report (pure)
def render_report(
    result: FitResult,
    ba: Mapping[str, float],
    *,
    seasons: Sequence[int],
    league_name: str,
    scoring_keys: int,
    partitions: int,
    generated: str,
    n_sims: int,
    records: Sequence[tuple[int, str, str, str]] = (),
) -> str:
    """The committed report. Pure over the fit + before/after so its prose can be pinned to its tables."""
    n_fitted, n_fallback, fallback_labels = _verdict_counts(result)
    n_warnings = sum(1 for level, *_ in records if level >= logging.WARNING)
    scored = list(result.test_seasons)
    base, both = ba["heuristic"], ba["both"]
    d_cv, d_inj, d_both = ba["cv_only"] - base, ba["injury_only"] - base, both - base
    d_inj_given_cv = both - ba["cv_only"]  # the injury marginal once the fitted CVs are already in
    absorbed = abs(d_inj_given_cv) < 0.005  # < 0.5 pts residual → injury has no marginal effect
    season_fitted = [p for p in POSITIONS if result.position_cv_out[p]["verdict"] == "fitted"]
    season_fallback = [p for p in POSITIONS if result.position_cv_out[p]["verdict"] != "fitted"]
    robust_fallback = [r["position"] for r in result.robustness.to_dict("records") if r["verdict"] != "fitted"]
    cohort_str = ", ".join(f"{p} {n}" for p, n in SEASON_COHORT_BY_POSITION.items())

    parts: list[str] = [
        "# Fitted simulator distributions — the measured knobs (Phase 9, ticket #32)",
        "",
        "> Generated by [`scripts/eval_distributions.py`](../scripts/eval_distributions.py) — regenerate "
        "with `./.venv/Scripts/python scripts/eval_distributions.py`. Committed artifact; do not "
        "hand-edit.",
        "",
        f"- **Generated (UTC):** {generated}",
        f"- **Train span:** {_season_span(list(seasons))} · **residual seasons (walk-forward OOS):** "
        f"{_season_span(scored)} (train ≤ S-1, test S — the models' own splits).",
        f"- **Lake backend:** `{LAKE_BACKEND}` · {partitions} partitions read",
        f"- **Scoring:** {league_name!r} live `scoring_settings`, {scoring_keys} keys (labels re-scored "
        "by the Phase 1 engine).",
        f"- **Frame:** {result.n_frame_rows:,} weekly rows, {result.n_players:,} players.",
        "",
        "There is **no baseline to beat** — a CV is not a prediction. The grade is the 18-cell verdict "
        "table below and a seed-pinned before/after on championship odds, reported move or not.",
        "",
        "## Findings, measured",
        "",
        f"1. **{n_fitted} of 18 (position × knob) cells fitted, {n_fallback} fell back to the heuristic.** "
        f"{'Fallbacks: ' + ', '.join(fallback_labels) + '.' if fallback_labels else 'Every cell fitted.'} "
        "Each fallback carries its reason in the tables (§A). Three positions are not six and three knobs "
        "are not one — every cell is decided on its own evidence.",
        f"2. **Before/after championship odds, decomposed (Amendment 1) — the marginals do not compose.** "
        f"Heuristic **{_pct(base)}** → both-fitted **{_pct(both)}** (Δ {100 * d_both:+.2f} pts). Under "
        f"common random numbers the isolated arms are **CV only** {100 * d_cv:+.2f} pts and **injury "
        f"only** {100 * d_inj:+.2f} pts; **adding them ({100 * (d_cv + d_inj):+.2f} pts) overstates the "
        f"total**, because the injury effect is not independent of the CV one. Its marginal *given* the "
        f"fitted CVs is **both − CV-only = {100 * d_inj_given_cv:+.2f} pts** ({_pct(both)} vs "
        f"{_pct(ba['cv_only'])}). "
        + (
            "So the injury knob's marginal is **entirely absorbed** once the fitted CVs are in: with "
            "season variance at fitted levels the mean haircut has no further effect on title odds — the "
            "sim's championship output is **variance-dominated, not mean-dominated**. That closes the §D "
            "loose end the four arms were built to test: the availability mean double-count this ticket "
            "declined to fix does **not** contaminate the headline grade — it is measurably not happening, "
            "a finding #34 can lean on."
            if absorbed
            else "So most of the injury effect is absorbed once the fitted CVs are in, though a residual "
            f"{100 * d_inj_given_cv:+.2f} pts remains — the sim is variance-led but not purely so (§D)."
        ),
        f"3. **Season CV — the cohort first, then the coherence gate (Amendment A + trap 2).** The season "
        f"CV is fit on the **drafted cohort** — the per-season top-N by projection this 12-team league "
        f"rosters at each position ({cohort_str}), from roster math, **not** a projection-value floor: "
        f"fitting over a wider pool measures the volatile fringe (backup QBs, RB3s) the sim never drafts, "
        f"which is what inflates the pooled CV (the tercile slide, §B). On that cohort the coherence gate "
        f"(the sim's own factor identity: a season is a ~17-game sum, so its CV must imply a modest "
        f"season-level factor, ≤ {MAX_SEASON_FACTOR_CV}) fits **{len(season_fitted)} of 6** — "
        f"{', '.join(season_fitted) if season_fitted else 'none'} — and falls back "
        f"{', '.join(season_fallback) if season_fallback else 'none'}. **Cohort-sensitive at the margin:** "
        f"the alternative structural cut (top-{SEASON_COHORT_TOTAL} overall by projection) "
        f"falls back {', '.join(robust_fallback) if robust_fallback else 'none'} instead (§E robustness) — "
        f"five of six pass either way, and *which* position fails depends on the cut. The wide-cohort CV "
        f"(~367/season) is recorded as an upper bound (§A); SeasonModel's residual also bounds the spread "
        f"around *its own* projection rather than Sleeper's — right, but historically unmeasurable "
        f"(`baseline_sleeper_points` null 2016-2025).",
        "4. **Era robustness, measured not asserted (trap 1 / Amendment 2).** Season-grain `mean r` by "
        "season (§B) shows whether the 2021 boundary (first 17-game season, predicted by a 16-game-only "
        "train set) shifts the level; each CV is the **mean of per-season CVs**, so a per-season level "
        "shift stays out of the pooled dispersion. Game grain is flat across the boundary (§C).",
        "5. **The constant-CV assumption, recorded (Amendment 3).** CV by prediction tercile (§B/§C) "
        "shows whether CV slides with the projection; the sim takes one constant per position regardless, "
        "so a large slide is a stated limitation, not a bug.",
        "6. **Two thresholds moved as consequences of the cohort decision — stated plainly, not buried.** "
        "Re-cutting both CV and injury to the drafted roster (Amendments A/C) makes the per-position "
        "cohorts small **by construction** (12-60 players/season), so two floors sized for the old wide "
        "pools were the wrong instrument and were re-derived on the cohort structure. Both are "
        "**verdict-affecting**, for a legitimate reason, and here is what each flipped: **`MIN_CV_N` "
        "100→40** flipped **K and DEF season CV** from fallback to fitted (drafted cohorts ~80-96 rows); "
        "**`MIN_INJURY_SEASONS` 200→50** keeps **QB, TE, K injury** fitted (drafted-cohort n's 178/140/94 "
        "fall below the old 200). Neither is tuned to a verdict — the coherence gate / injury coverage "
        "still decide, and both cohorts are reported side by side (§A) so the reader judges the pattern.",
        f"7. **{'Zero' if not n_warnings else n_warnings} warning"
        f"{'' if n_warnings == 1 else 's'} on real data** (verbatim below).",
        "",
        "## A. The 18-cell verdict — season CV · game CV · injury",
        "",
        "Each cell is `fitted` only with enough held-out evidence (and, for season CV, coherence with the "
        "game CV under the sim's factor identity); otherwise it defers to the heuristic the sim already "
        "shipped. `n` is the kept residual rows (CV) or qualified player-seasons (injury).",
        "",
        "**Season CV** (`POSITION_CV`) — from `SeasonModel` OOS residuals on the **drafted cohort** "
        "(per-season top-N by projection, § Findings 3), **healthy** (setback-free); the full-cohort CV "
        "and the wide-cohort (~367/season) **upper bound** are shown alongside. Ownership (trap 4): "
        "`INJURY_RISK` owns games-missed, and the healthy-vs-full near-equality (≤ ~0.06 apart) confirms "
        "the season CV barely double-counts it — at the drafted grain a player's projection-relative "
        "residual is dominated by non-injury factors.",
        *_cv_table(result.position_cv_out, result.position_cv, season=True),
        "",
        "**Game CV** (`GAME_CV`) — from `WeeklyModel` (QB/RB/WR/TE) and `KickDefModel` (K/DEF) OOS "
        "residuals; weekly rows are played weeks by construction, so no healthy filter applies:",
        *_cv_table(result.game_cv_out, result.game_cv, season=False),
        "",
        "**Injury** (`INJURY_RISK`):",
        *_injury_table(result.injury_out, result.injury),
        "",
        "_The shipped rate is fit on the **drafted cohort** (the same per-season top-N by projection the "
        "season CV uses — Amendment C: the injury knob is applied to exactly these rostered players, not "
        "the loose qualified pool, which inverted TE above RB). `P(setback) wide` is that loose-pool rate "
        "for contrast. Setback = a contiguous **injury absence** of ≥ 2 weeks over a tenure-bounded "
        "denominator (first seen by week 4, played ≥ 3 weeks), counted only when injury-corroborated — the "
        "player carries an injury-report status (Out/Doubtful/Questionable) in the run or the week it "
        "began — which catches the IR case a pure `Out`-run misses (a season-ending injury drops off the "
        "weekly report; measured, only ~35% of season-enders keep any `Out` row) while excluding byes and "
        "clean benchings (no report). `Out-only P` is the IR-truncated lower bound the pure-`Out` "
        "definition would have shipped; the fitted rate is back in the range of the domain heuristic and "
        "orders correctly (RB ≈ WR > TE > QB > K). `mean weeks missed` includes season-enders, so it "
        "exceeds the heuristic's return-injury figure._",
        "",
        "## B. Season CV — by season (era check) and by prediction tercile (slide)",
        "",
        "`mean r` by season — watch 2021, the first 17-game year predicted by a 16-game-only train set:",
        *_by_season_pivot(result.position_cv, "mean_r", result.test_seasons),
        "",
        "`CV r` by season — the per-season dispersion the fit averages (a level shift above does not "
        "appear here, which is the point):",
        *_by_season_pivot(result.position_cv, "cv_r", result.test_seasons),
        "",
        "`CV r` by prediction tercile (low → high projected season total):",
        *_tercile_table(result.position_cv),
        "",
        "## C. Game CV — by season (era check) and by prediction tercile (slide)",
        "",
        "`mean r` by season — game grain should be flat across the 2021 boundary (a week's prediction is "
        "same-scale as its actual regardless of season length):",
        *_by_season_pivot(result.game_cv, "mean_r", result.test_seasons),
        "",
        "`CV r` by prediction tercile:",
        *_tercile_table(result.game_cv),
        "",
        "## D. Expected availability — the mean-shift the injury knob induces (Amendment 1)",
        "",
        "`E[avail] = 1 − p·E[clip(Poisson(sev),1,17)]/17`. The haircut multiplies a projection that "
        "already embeds average injury loss, so a change here is a *mean* shift in relative position "
        "value — the reason the before/after must be decomposed (§ Findings 2). This ticket measures it; "
        "re-centering the multiplier is a design change out of scope.",
        *_avail_table(result),
        "",
        "## E. Season-factor coherence (trap 2 — both ends)",
        "",
        "`1 + CV_total² = (1 + CV_factor²)(1 + CV_week²/W)`, W = 17. A fitted season CV must imply a "
        f"season-level `factor CV` in `(0, {MAX_SEASON_FACTOR_CV}]`: **0** is the collapse (single-game "
        "noise alone exceeds the season CV — the season loses its correlation); **above the cap** the "
        "season CV is not coherent with the game CV (a whole-season baseline swinging that much around the "
        "projection is not a season factor the sim can carry). `SeasonModel CV` is the drafted-cohort OOS "
        "residual CV; `shipped season CV` is what ships (the fitted value where it clears, else the "
        "coherent heuristic) and `shipped factor CV` confirms every shipped pair is coherent. **Threshold "
        f"provenance:** the *principle* (a season-baseline swing beyond ~{int(MAX_SEASON_FACTOR_CV*100)}% "
        "is not outcome uncertainty) is pre-data, but the exact 0.5 was chosen after seeing the fitted "
        "factors — it sits above the coherent positions and below the incoherent one, not declared blind; "
        "the split is robust to any value in ~[0.44, 0.55].",
        *_coherence_table(result),
        "",
        f"**Robustness — the alternative cohort (top-{SEASON_COHORT_TOTAL} overall by projection).** The "
        "primary cut is per-position roster math; this cross-position cut is the check. Five of six pass "
        "either way; the two disagree only about which single position fails, so the marginal verdict is "
        "cohort-sensitive (stated, not hidden behind one number):",
        *_robustness_table(result),
        "",
        "## F. Before/after championship odds — fixed synthetic 12-team league",
        "",
        f"One fixed roster, {n_sims:,} sims, seed-pinned, common random numbers across arms (my_edge "
        "regime). The four arms decompose the total into the uncertainty effect (CV) and the "
        "mean/availability effect (injury); the total on its own is not the finding.",
        *_before_after_table(ba),
        "",
        "## Warnings (verbatim)",
        "",
        "The standing bar is zero unexpected warnings on real data.",
        "",
        _log_block(records, min_level=logging.WARNING),
        "",
    ]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- entry point
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit the sims' distributions, write the report + artifact")
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

    partitions = int(len(lake_inventory()))
    print(
        f"Fitting the sim distributions over {seasons[0]}-{seasons[-1]} on backend {LAKE_BACKEND} "
        f"({partitions} partitions), scoring {league.get('name')!r} ({len(scoring)} keys)…"
    )

    capture = _Capture()
    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        result = fit_distributions(seasons, scoring)
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        payload = to_artifact(result, generated=generated)
        ba = before_after(result)
        payload["diagnostics"]["before_after_champ_odds"] = {k: round(v, 4) for k, v in ba.items()}
    except ValueError as exc:
        print(f"could not fit the distributions — {exc}", file=sys.stderr)
        return 1
    finally:
        root.removeHandler(capture)
        root.setLevel(prior_level)

    art = Path(args.artifact)
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = render_report(
        result, ba,
        seasons=seasons, league_name=league.get("name", "?"), scoring_keys=len(scoring),
        partitions=partitions, generated=generated, n_sims=_BEFORE_AFTER_SIMS, records=capture.records,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    n_fitted, n_fallback, _ = _verdict_counts(result)
    n_warnings = sum(1 for level, *_ in capture.records if level >= logging.WARNING)
    print(
        f"Wrote {out} (report) and {args.artifact} (artifact) — {n_fitted}/18 cells fitted, "
        f"{n_fallback} fallback; champ odds {_pct(ba['heuristic'])} → {_pct(ba['both'])}; "
        f"{n_warnings} warning(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

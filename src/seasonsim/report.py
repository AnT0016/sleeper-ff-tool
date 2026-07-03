"""Format the season Monte Carlo into a plain-text report.

Sections, each after the explicit assumptions block (so the numbers can be judged, not trusted):

1. **My season** — championship / playoff odds, expected wins, and the seed & points distribution.
2. **Start/sit edge** — how much championship probability the weekly optimizer adds (the gap between
   the *equal_skill* and *my_edge* regimes) — i.e. what tightening the in-season levers is worth.
3. **League board** — every team's title / playoff odds and expected wins; for a completed season,
   shown next to the *actual* final standing and champion so the sim can be calibrated by eye.
"""

from __future__ import annotations

import numpy as np

from .distributions import GAME_CV, INJURY_RISK, POSITION_CV, SEASON_GAMES
from .engine import SeasonSimOutput
from .inputs import SeasonInputs

_POS = ("QB", "RB", "WR", "TE", "K", "DEF")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def assumptions_block(out: SeasonSimOutput, inp: SeasonInputs) -> str:
    cv = " ".join(f"{p} {POSITION_CV[p]:.2f}" for p in _POS)
    gcv = " ".join(f"{p} {GAME_CV[p]:.2f}" for p in _POS)
    inj = " ".join(f"{p} {INJURY_RISK[p][0]:.0%}/{INJURY_RISK[p][1]:.0f}g" for p in _POS)
    lines = [
        "ASSUMPTIONS (directional & heuristic — NOT fitted to data; tune in src/seasonsim/):",
        "  • Outcomes: each week = single-game lognormal (realistic one-week noise, shared with the",
        "      win-prob model) × a per-season factor sized so season totals keep the season CV.",
        f"      Game CV by pos: {gcv}.  Season CV by pos: {cv}",
        f"  • Injuries: one multi-week setback/season, P/mean-games (of {SEASON_GAMES}): {inj}",
        "      → the player's lineup slot is empty for those weeks (bench gets tested).",
        "  • Byes NOT modeled (season spread evenly over weeks) — uniform across teams; a v1 limit.",
        f"  • Start/sit: managers set lineups from projected means ± N(0, {out.opp_noise:.2f}×mean) "
        "noise;",
        "      equal_skill = everyone equally sloppy; my_edge = I play clean (noise 0), rivals don't.",
        f"  • Schedule: {inp.schedule_source}. Playoffs: top {out.n_playoff_teams}, "
        f"weeks {out.playoff_weeks[0]}-{out.playoff_weeks[-1]} (locked bracket, top-2 bye).",
        f"  • {out.n_sims} simulated seasons · seed {out.seed} · common random numbers across regimes.",
    ]
    if out.conditioned_weeks:
        w0, w1 = min(out.conditioned_weeks), max(out.conditioned_weeks)
        lines.append(
            f"  • Conditioned on the REAL results of {len(out.conditioned_weeks)} played week(s) "
            f"(W{w0}–W{w1}) — odds are from here, not preseason."
        )
    return "\n".join(lines)


def my_season(out: SeasonSimOutput) -> str:
    me = out.my_team
    edge = out.regimes["my_edge"]
    name = out.pool.team_names[me]
    champ = float(edge.my_is_champ.mean())
    playoffs = float((edge.my_rank <= out.n_playoff_teams).mean())
    p_bye = float((edge.my_rank <= 2).mean())
    wins = float(edge.my_wins.mean())
    n_reg = len(out.regular_weeks)
    p10, p50, p90 = (float(x) for x in np.percentile(edge.my_points, [10, 50, 90]))
    med_seed = int(np.median(edge.my_rank))
    return "\n".join(
        [
            f"--- My season — {name} (my_edge regime) ---",
            f"  Championship: {_pct(champ)}      Make playoffs: {_pct(playoffs)}      "
            f"Top-2 bye: {_pct(p_bye)}",
            f"  Expected record: {wins:.1f}-{n_reg - wins:.1f}   median finish: "
            f"seed {med_seed} of {out.pool.n_teams}",
            f"  Season points-for: p10 {p10:.0f} · median {p50:.0f} · p90 {p90:.0f}",
        ]
    )


def startsit_edge(out: SeasonSimOutput) -> str:
    base = out.regimes["equal_skill"]
    edge = out.regimes["my_edge"]
    c0, c1 = float(base.my_is_champ.mean()), float(edge.my_is_champ.mean())
    p0 = float((base.my_rank <= out.n_playoff_teams).mean())
    p1 = float((edge.my_rank <= out.n_playoff_teams).mean())
    return "\n".join(
        [
            "--- Start/sit edge — what the weekly optimizer is worth ---",
            f"  Championship: {_pct(c0)} (equal skill) → {_pct(c1)} (my edge)   "
            f"Δ {100 * (c1 - c0):+.1f} pts",
            f"  Make playoffs: {_pct(p0)} → {_pct(p1)}   Δ {100 * (p1 - p0):+.1f} pts",
            "  (Δ = playing clean lineups while opponents keep their start/sit noise. "
            "This is lever #2's value.)",
        ]
    )


def league_board(out: SeasonSimOutput, inp: SeasonInputs) -> str:
    edge = out.regimes["my_edge"]
    me = out.my_team
    order = sorted(range(out.pool.n_teams), key=lambda t: edge.champ_prob[t], reverse=True)
    completed = inp.completed and inp.actual_rank is not None

    head = f"  {'team':<22} {'title':>7} {'playoff':>8} {'exp W':>6}"
    if completed:
        head += f"  {'actual':>7}"
    lines = ["--- League board (my_edge regime) — title & playoff odds per team ---", head]
    for t in order:
        nm = out.pool.team_names[t][:22]
        mark = " ←" if t == me else ""
        row = (
            f"  {nm:<22} {_pct(edge.champ_prob[t]):>7} {_pct(edge.playoff_prob[t]):>8} "
            f"{edge.exp_wins[t]:6.1f}"
        )
        if completed:
            star = "★" if t == inp.actual_champion else ""
            row += f"  {'#' + str(inp.actual_rank[t]) + star:>7}"
        lines.append(row + mark)
    if completed:
        lines.append(
            "  (actual = real regular-season finish (seeding order), NOT final placement; "
            "★ = the team that actually won the title.)"
        )
    return "\n".join(lines)


def format_report(out: SeasonSimOutput, inp: SeasonInputs) -> str:
    parts = [
        f"=== Full-season championship Monte Carlo — {out.season} ===",
        "",
        assumptions_block(out, inp),
        "",
        my_season(out),
        "",
        startsit_edge(out),
        "",
        league_board(out, inp),
        "",
        "Directional only — odds are model-based, not a guarantee; a single real season is one draw "
        "from this distribution.",
    ]
    return "\n".join(parts)

"""Weekly win-probability start/sit — reframes the lineup call as *P(win this week)*, not just points.

The projection-optimal lineup maximizes *expected* points, but a fantasy week is won head-to-head, so
the right objective is the probability your lineup outscores *this week's opponent*. Those differ at
the tails: as a heavy favorite you want a high **floor** (don't bust), as a heavy underdog you want a
high **ceiling** (a safe lineup still loses). This module makes that explicit.

Each player's week is a lognormal draw whose mean is this week's custom-scored projection and whose CV
is a per-position **single-game** CV (below). We deliberately do *not* reuse Plan A's season CV × √games
weekly CV here: that (independence-derived) value is right for a *season sum* but absurdly skewed for one
game — it implies a 12-point WR most likely scores ~7 — so it makes variance pure downside. The moderate
single-game CVs below keep the one-week distribution realistic (median near the mean) and, as a result,
reproduce the real leverage: extra variance **helps an underdog** (fatter ceiling to reach a higher
opponent) and **hurts a favorite** (only adds downside). Simulating both lineups gives P(win); swapping
one bench player for one starter under **common random numbers** gives each start/sit call's honest
effect on win probability — occasionally *positive despite a lower projection*. Pure/offline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from draftsim.distributions import sample_season_points
from optimizer.lineup import LineupPlayer, optimize
from optimizer.startsit import start_sit_table

#: Per-position **single-game** coefficient of variation (std ÷ mean of one week's fantasy points).
#: Heuristic, informed by typical weekly fantasy variance — NOT the season CV × √17 used by the
#: season sim (that over-skews a single game). QB/K steadiest; DEF/TE/WR the boom-bust positions.
WEEKLY_CV: dict[str, float] = {
    "QB": 0.45, "RB": 0.60, "WR": 0.70, "TE": 0.75, "K": 0.50, "DEF": 0.85,
}
DEFAULT_WEEKLY_CV = 0.65

#: P(win) thresholds for the strategic posture (favor floor vs ceiling).
FAVORITE = 0.65
UNDERDOG = 0.35


@dataclass(frozen=True)
class WinProb:
    p_win: float
    my_mean: float
    opp_mean: float
    my_p10: float
    my_p50: float
    my_p90: float
    margin_mean: float  # my_mean - opp_mean


@dataclass(frozen=True)
class SwapLeverage:
    """Starting ``bench`` in place of ``starter`` this week: its projection delta vs its win-prob delta."""

    bench: LineupPlayer
    starter: LineupPlayer
    slot: str | None
    delta_proj: float  # bench.proj - starter.proj (<= 0 in a projection-optimal lineup)
    delta_winprob: float  # change in P(win) from making the swap (can be > 0 even when delta_proj < 0)
    swap_winprob: float


def _weekly_cv_vec(positions: Sequence[str]) -> np.ndarray:
    return np.array([WEEKLY_CV.get(p, DEFAULT_WEEKLY_CV) for p in positions], dtype=float)


def _lineup_samples(rng: np.random.Generator, means, positions, n_sims: int) -> np.ndarray:
    """``(n_sims,)`` sampled lineup totals: each player a weekly lognormal draw, summed."""
    means = np.asarray(means, dtype=float)
    if means.size == 0:
        return np.zeros(n_sims)
    draws = sample_season_points(rng, means, _weekly_cv_vec(positions), n_sims)
    return draws.sum(axis=1)


def _pwin(mine: np.ndarray, opp: np.ndarray) -> float:
    return float((mine > opp).mean() + 0.5 * (mine == opp).mean())


def win_probability(
    my_means, my_pos, opp_means, opp_pos, *, n_sims: int = 20000, seed: int = 0
) -> WinProb:
    """Probability my starting lineup outscores the opponent's this week, + my score distribution."""
    rng = np.random.default_rng(seed)
    mine = _lineup_samples(rng, my_means, my_pos, n_sims)
    opp = _lineup_samples(rng, opp_means, opp_pos, n_sims)
    p10, p50, p90 = (float(x) for x in np.percentile(mine, [10, 50, 90]))
    return WinProb(
        p_win=round(_pwin(mine, opp), 4),
        my_mean=round(float(mine.mean()), 2),
        opp_mean=round(float(opp.mean()), 2),
        my_p10=round(p10, 1),
        my_p50=round(p50, 1),
        my_p90=round(p90, 1),
        margin_mean=round(float(mine.mean() - opp.mean()), 2),
    )


def leverage_note(p_win: float) -> tuple[str, str]:
    """``(label, guidance)`` for the week's posture — favor floor when favored, ceiling when not."""
    if p_win >= FAVORITE:
        return ("favorite", "Clear favorite — favor FLOOR: start safe, high-floor players; don't chase "
                "ceiling and risk a bust that hands away a winnable week.")
    if p_win <= UNDERDOG:
        return ("underdog", "Underdog — favor CEILING: start boom/bust upside; a safe lineup likely "
                "still loses, so play for the high-variance outcome that can win.")
    return ("toss-up", "Coin-flip — start your highest-projected lineup; this is where small start/sit "
            "edges swing the most games.")


def startsit_leverage(
    my_players: Sequence[LineupPlayer],
    slots: Mapping[str, int],
    opp_means,
    opp_pos,
    *,
    n_sims: int = 20000,
    seed: int = 0,
) -> tuple[float, list[SwapLeverage]]:
    """Each bench→starter swap's effect on P(win), under common random numbers.

    Returns ``(baseline_win_prob, swaps)`` where ``swaps`` (sorted by win-prob gain) are the
    projection-optimal lineup's bench players and the starter each would displace. A positive
    ``delta_winprob`` on a negative ``delta_proj`` is the leverage play the points-only optimizer
    misses (start upside when you need it, floor when you're ahead).
    """
    base = optimize(my_players, slots)
    starters = [sp.player for sp in base.starters]
    deltas = [d for d in start_sit_table(base) if d.would_replace is not None]

    # Sample every player once (starters + candidate bench) so swaps share the same draws (CRN).
    pool = list({id(p): p for p in starters + [d.player for d in deltas]}.values())
    idx = {id(p): i for i, p in enumerate(pool)}
    rng = np.random.default_rng(seed)
    samples = sample_season_points(
        rng, np.array([p.proj_pts for p in pool], dtype=float),
        _weekly_cv_vec([p.pos for p in pool]), n_sims,
    )
    opp = _lineup_samples(rng, opp_means, opp_pos, n_sims)

    base_total = samples[:, [idx[id(p)] for p in starters]].sum(axis=1)
    base_p = _pwin(base_total, opp)

    swaps: list[SwapLeverage] = []
    for d in deltas:
        total = base_total - samples[:, idx[id(d.would_replace)]] + samples[:, idx[id(d.player)]]
        p = _pwin(total, opp)
        swaps.append(
            SwapLeverage(
                bench=d.player,
                starter=d.would_replace,
                slot=d.slot,
                delta_proj=d.delta,
                delta_winprob=round(p - base_p, 4),
                swap_winprob=round(p, 4),
            )
        )
    swaps.sort(key=lambda s: s.delta_winprob, reverse=True)
    return round(base_p, 4), swaps

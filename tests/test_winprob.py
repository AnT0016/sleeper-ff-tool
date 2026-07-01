"""Offline tests for the weekly win-probability start/sit (no network).

Covers the win-probability core (identical lineups ~ 50/50; a dominant lineup wins most), the leverage
insight that higher variance *helps* an underdog and *hurts* a favorite, the posture thresholds, and
the start/sit swap mechanics (baseline win prob, projection-optimal deltas ≤ 0, swap wiring). The live
opponent-resolution path is exercised via ``scripts/win_prob.py`` / the snapshot.
"""

from __future__ import annotations

from optimizer.lineup import LineupPlayer
from optimizer.winprob import (
    FAVORITE,
    UNDERDOG,
    leverage_note,
    startsit_leverage,
    win_probability,
)


def _p(pid, pos, proj):
    return LineupPlayer(player_id=pid, name=pid, pos=pos, team="X", proj_pts=proj)


# --------------------------------------------------------------------------- win probability core
def test_identical_lineups_are_a_coinflip():
    wp = win_probability([20.0, 12.0], ["QB", "RB"], [20.0, 12.0], ["QB", "RB"], seed=1)
    assert 0.45 <= wp.p_win <= 0.55


def test_dominant_lineup_wins_most():
    wp = win_probability([30, 25, 20], ["QB", "RB", "WR"], [8, 7, 6], ["QB", "RB", "WR"], seed=1)
    assert wp.p_win > 0.9
    assert wp.margin_mean > 0


# --------------------------------------------------------------------------- the leverage insight
#: Two nine-player lineups with the *same* mean (108) but different variance — all-TE (boom/bust) vs
#: all-QB (steady). The sum is symmetric enough for the variance effect to show cleanly.
_MEANS = [12.0] * 9
_HIGH_VAR = ["TE"] * 9
_LOW_VAR = ["QB"] * 9
_STEADY_OPP = ["QB"] * 9


def test_higher_variance_helps_the_underdog():
    # Trailing a steady ~120 opponent, the higher-ceiling lineup reaches over the top more often.
    opp = [120.0 / 9] * 9
    low = win_probability(_MEANS, _LOW_VAR, opp, _STEADY_OPP, n_sims=60000, seed=3)
    high = win_probability(_MEANS, _HIGH_VAR, opp, _STEADY_OPP, n_sims=60000, seed=3)
    assert high.p_win > low.p_win


def test_higher_variance_hurts_the_favorite():
    # Ahead of a steady ~96 opponent, extra variance is only downside — the safe lineup wins more.
    opp = [96.0 / 9] * 9
    low = win_probability(_MEANS, _LOW_VAR, opp, _STEADY_OPP, n_sims=60000, seed=3)
    high = win_probability(_MEANS, _HIGH_VAR, opp, _STEADY_OPP, n_sims=60000, seed=3)
    assert high.p_win < low.p_win


def test_leverage_note_posture_thresholds():
    assert leverage_note(FAVORITE + 0.05)[0] == "favorite"
    assert leverage_note(UNDERDOG - 0.05)[0] == "underdog"
    assert leverage_note(0.5)[0] == "toss-up"


# --------------------------------------------------------------------------- start/sit swap mechanics
def test_startsit_leverage_wires_swaps_and_optimal_deltas():
    my_players = [_p("qb", "QB", 20.0), _p("rb", "RB", 12.0), _p("wr", "WR", 11.0)]
    slots = {"QB": 1, "FLEX": 1}
    base_p, swaps = startsit_leverage(my_players, slots, [25.0], ["QB"], n_sims=20000, seed=0)

    assert 0.0 <= base_p <= 1.0
    assert len(swaps) == 1  # only the WR can swap into the RB's FLEX spot
    s = swaps[0]
    assert s.bench.name == "wr" and s.starter.name == "rb"
    assert s.delta_proj == -1.0  # projection-optimal baseline → the swap costs points
    assert abs(s.swap_winprob - (base_p + s.delta_winprob)) < 1e-9


def test_startsit_leverage_no_swaps_when_no_bench():
    # A roster that exactly fills its slots has no bench players to consider.
    my_players = [_p("qb", "QB", 20.0), _p("k", "K", 9.0)]
    base_p, swaps = startsit_leverage(my_players, {"QB": 1, "K": 1}, [18.0], ["QB"], seed=0)
    assert swaps == []
    assert 0.0 <= base_p <= 1.0

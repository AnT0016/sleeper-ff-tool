"""Offline unit tests for the Phase 6 Monte Carlo draft simulator (no network).

Covers the pure pieces: the lognormal/injury sampling (mean-preserving, bounded availability),
the fast lineup valuation (FLEX takes the best leftover, never a QB), the strategy guardrails
(mandatory end-of-draft fill, K/DEF never early, zero-RB avoidance), the ADP-bot caps, and a small
end-to-end Monte Carlo on a synthetic board (legal rosters, survival decreasing with later picks,
common-random-numbers reproducibility). The live glue + full-board run are validated manually
against the league (see docs/PROGRESS.md).
"""

from __future__ import annotations

import numpy as np

from draft.roster import RosterConfig
from draftsim import distributions as dist
from draftsim.bots import bot_allows
from draftsim.engine import build_pool, simulate
from draftsim.lineup import best_lineup_points, select_starters
from draftsim.strategy import STRATEGIES, forced_positions
from projections.board import PlayerRow

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1, "BN": 5}
CFG = RosterConfig(teams=12, rounds=14, slots=SLOTS)


# --------------------------------------------------------------------------- distributions
def test_lognormal_is_mean_preserving():
    rng = np.random.default_rng(1)
    mean = np.array([200.0, 100.0, 50.0])
    cv = np.array([0.3, 0.3, 0.3])
    pts = dist.sample_season_points(rng, mean, cv, 20000)
    # sample means should track the projection closely; lognormal is non-negative.
    assert np.allclose(pts.mean(axis=0), mean, rtol=0.03)
    assert (pts >= 0).all()


def test_zero_projection_stays_zero():
    rng = np.random.default_rng(0)
    pts = dist.sample_season_points(rng, np.array([0.0, 10.0]), np.array([0.3, 0.3]), 100)
    assert (pts[:, 0] == 0.0).all()


def test_availability_bounded_and_tracks_risk():
    rng = np.random.default_rng(2)
    p = np.array([0.0, 1.0])  # never hurt vs. always hurt
    sev = np.array([3.0, 3.0])
    mult, setback = dist.sample_availability(rng, p, sev, 5000)
    assert (mult > 0).all() and (mult <= 1.0).all()
    assert (mult[:, 0] == 1.0).all()  # p=0 -> never misses a game
    assert setback[:, 1].all() and not setback[:, 0].any()
    assert mult[:, 1].mean() < 1.0  # p=1 -> always loses some availability


# --------------------------------------------------------------------------- lineup valuation
def test_best_lineup_flex_takes_best_leftover_not_qb():
    # 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX. A huge spare QB must NOT take the flex; best leftover RB does.
    pos = ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "TE"]
    pts = [40, 39, 20, 18, 17, 15, 14, 10]  # spare QB=39, spare RB=17, are the flex candidates
    total = best_lineup_points(pos, pts, SLOTS)
    # starters: QB40 + RB20+RB18 + WR15+WR14 + TE10 + FLEX(best leftover = RB17) = 134
    assert total == 40 + 20 + 18 + 15 + 14 + 10 + 17


def test_select_starters_matches_total():
    pos = ["QB", "RB", "RB", "RB", "WR", "WR", "TE", "K", "DEF"]
    pts = [25, 20, 18, 16, 15, 14, 10, 8, 7]
    starters = select_starters(pos, pts, SLOTS)
    assert sum(pts[i] for i in starters) == best_lineup_points(pos, pts, SLOTS)
    assert len(starters) == 1 + 2 + 2 + 1 + 1 + 1 + 1  # QB,2RB,2WR,TE,FLEX,K,DEF = 9


def test_missing_position_is_a_hole_not_an_error():
    total = best_lineup_points(["QB", "RB"], [20, 10], SLOTS)  # no WR/TE/K/DEF
    assert total == 30  # only the players present contribute


# --------------------------------------------------------------------------- strategy guards
def test_forced_fill_when_picks_run_out():
    # Drafted nothing but skill players; 3 picks left but still owe QB, K, DEF -> all forced.
    counts = {"RB": 3, "WR": 3, "TE": 1}
    forced = forced_positions(counts, CFG, picks_left=3)
    assert forced == {"QB", "K", "DEF"}


def test_no_force_with_plenty_of_picks_left():
    assert forced_positions({"RB": 1}, CFG, picks_left=13) is None


def test_strategies_never_offer_k_or_def_early():
    for name, fn in STRATEGIES.items():
        for rnd in range(1, 11):
            pref = fn(rnd, {}, CFG)
            assert "K" not in pref and "DEF" not in pref, name


def test_zero_rb_avoids_rb_early_then_allows():
    fn = STRATEGIES["zero_rb"]
    assert "RB" not in fn(1, {}, CFG)
    assert "RB" not in fn(4, {}, CFG)
    assert "RB" in fn(5, {}, CFG)


# --------------------------------------------------------------------------- bots
def test_bot_caps_and_late_only():
    assert bot_allows("RB", {"RB": 0}, rnd=1, rounds=14)
    assert not bot_allows("QB", {"QB": 2}, rnd=5, rounds=14)  # QB cap = 2
    assert not bot_allows("DEF", {"DEF": 0}, rnd=2, rounds=14)  # too early for DEF
    assert bot_allows("DEF", {"DEF": 0}, rnd=12, rounds=14)  # late enough


# --------------------------------------------------------------------------- end-to-end (synthetic)
def _synthetic_board(n_per_pos=40):
    """A descending-value board per position with ADP roughly following overall value rank."""
    rows: list[PlayerRow] = []
    counter = 0
    # interleave positions by value so ADP ~ overall strength
    grid = []
    for rank in range(n_per_pos):
        for pos, base in (("RB", 300), ("WR", 295), ("QB", 290), ("TE", 250)):
            grid.append((pos, base - rank * 6))
    for pos, pts in grid:
        counter += 1
        rows.append(
            PlayerRow(
                player_id=f"{pos}{counter}",
                name=f"{pos}-{counter}",
                pos=pos,
                team=None,
                proj_pts=float(pts),
                adp=float(counter),  # ADP = overall order
                vor=float(pts),  # use proj as a stand-in VOR for the test
            )
        )
    # a handful of cheap K/DEF (undrafted ADP) so the mandatory guard can complete a lineup
    for pos in ("K", "DEF"):
        for j in range(16):
            counter += 1
            rows.append(
                PlayerRow(player_id=f"{pos}{j}", name=f"{pos}-{j}", pos=pos, team=None,
                          proj_pts=120.0 - j, adp=float("inf"), vor=-50.0 - j)
            )
    rows.sort(key=lambda r: r.vor, reverse=True)
    return rows


def test_build_pool_keeps_kdef_and_caps_vor():
    board = _synthetic_board()
    pool = build_pool(board, pool_size=50)
    assert "K" in pool.pos and "DEF" in pool.pos  # K/DEF always kept despite low VOR
    assert pool.n >= 50


def test_end_to_end_legal_rosters_and_survival_decreases():
    board = _synthetic_board()
    out = simulate(board, CFG, my_slot=7, n_sims=120, strategies=["best_vor", "zero_rb"], seed=3)
    res = out.results["best_vor"]

    # every sim produced a finish rank in [1, teams] and a positive season total
    assert res.my_rank.min() >= 1 and res.my_rank.max() <= CFG.teams
    assert (res.my_points > 0).all()

    # my modal roster fills exactly `rounds` picks
    assert int(res.draft_counts.sum()) == CFG.rounds * out.n_sims

    # a top-ADP player is (weakly) less likely to survive to my 2nd pick than my 1st
    pool = out.pool
    early = pool.vor_order[0]  # best player overall
    assert res.survival[1][early] <= res.survival[0][early] + 1e-9


def test_common_random_numbers_make_runs_reproducible():
    board = _synthetic_board()
    a = simulate(board, CFG, my_slot=3, n_sims=100, strategies=["best_vor"], seed=42)
    b = simulate(board, CFG, my_slot=3, n_sims=100, strategies=["best_vor"], seed=42)
    assert np.array_equal(a.results["best_vor"].my_points, b.results["best_vor"].my_points)

"""Offline unit tests for the Phase 7 full-season championship simulator (no network).

Covers the pure pieces: the weekly/injury sampling (mean-preserving weeks, bounded within-season
injury stretches), the vectorised lineup valuation (matches the scalar draft-sim optimizer; honours
select-vs-value), the round-robin + 6-team bracket, record tallying, and a small end-to-end season
(champion probabilities sum to 1, reproducible under a seed, a stacked roster wins most, and the
start/sit edge never hurts the sharp manager). The live glue + full-league calibration are validated
manually against real seasons (see docs/PROGRESS.md).
"""

from __future__ import annotations

import numpy as np

from draftsim.lineup import best_lineup_points
from seasonsim import distributions as dist
from seasonsim.engine import SeasonPool, simulate_season
from seasonsim.lineup import lineup_values
from seasonsim.schedule import playoff_champion, resolve_records, round_robin, seed_order

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1, "BN": 0}
STARTER_POS = ["QB", "RB", "RB", "RB", "WR", "WR", "TE", "K", "DEF"]  # 9-slot roster, extra RB → FLEX


# --------------------------------------------------------------------------- distributions
def test_weekly_sum_reproduces_season_mean():
    rng = np.random.default_rng(0)
    mean = np.array([170.0, 90.0])
    cv = np.array([0.3, 0.35])
    wk = dist.sample_weekly_points(rng, mean, cv, 20000, 17)
    season = wk.sum(axis=2)  # (n_sims, n_players)
    assert np.allclose(season.mean(axis=0), mean, rtol=0.05)
    assert (wk >= 0).all()


def test_zero_projection_stays_zero_weekly():
    rng = np.random.default_rng(0)
    wk = dist.sample_weekly_points(rng, np.array([0.0, 50.0]), np.array([0.3, 0.3]), 50, 17)
    assert (wk[:, 0, :] == 0).all()
    assert wk[:, 1, :].sum() > 0


def test_injury_stretch_is_within_season_and_contiguous():
    rng = np.random.default_rng(1)
    out = dist.sample_injury_out(rng, np.array([1.0]), np.array([3.0]), 5000, 17)
    per_sim = out[:, 0, :].sum(axis=1)
    assert per_sim.min() >= 1 and per_sim.max() <= 17  # a setback costs 1..season-length weeks
    # never-injured position stays fully available
    healthy = dist.sample_injury_out(rng, np.array([0.0]), np.array([3.0]), 100, 17)
    assert not healthy.any()


# --------------------------------------------------------------------------- lineup
def test_lineup_optimal_matches_scalar_optimizer():
    rng = np.random.default_rng(2)
    pts = rng.gamma(2.0, 5.0, size=(200, len(STARTER_POS)))
    vec = lineup_values(STARTER_POS, pts, pts, SLOTS)  # select == value → hindsight-optimal
    for s in range(pts.shape[0]):
        assert abs(vec[s] - best_lineup_points(STARTER_POS, pts[s], SLOTS)) < 1e-9


def test_lineup_selects_by_select_scores_by_value():
    # one RB slot, no flex: the higher-select RB starts even though it scored fewer points.
    pos = ["RB", "RB"]
    slots = {"RB": 1, "FLEX": 0}
    select = np.array([[1.0, 2.0]])  # player 1 preferred
    value = np.array([[10.0, 3.0]])  # but player 0 actually scored more
    assert lineup_values(pos, select, value, slots)[0] == 3.0


def test_flex_takes_best_leftover():
    # extra RB beyond the 2 RB slots should fill FLEX; total = sum of the 5 best startable here.
    pos = ["RB", "RB", "RB", "WR", "WR", "TE", "QB", "K", "DEF"]
    slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1}
    pts = np.array([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 10.0, 1.0, 2.0]])
    # starters: QB10, RB9, RB8, WR6, WR5, TE4, K1, DEF2, FLEX=best leftover(RB7)=7
    assert lineup_values(pos, pts, pts, slots)[0] == 9 + 8 + 7 + 6 + 5 + 4 + 10 + 1 + 2


# --------------------------------------------------------------------------- schedule / bracket
def test_round_robin_is_complete():
    sched = round_robin(4, [1, 2, 3])
    pairs = {frozenset(p) for wk in sched.values() for p in wk}
    assert len(pairs) == 6  # every one of C(4,2) pairings appears exactly once
    for wk in sched.values():
        assert sorted(t for p in wk for t in p) == [0, 1, 2, 3]  # everyone plays each week


def test_bracket_top_seed_and_upset():
    seeds = [0, 1, 2, 3, 4, 5]
    dom = np.zeros((6, 3))
    dom[0, :] = 100  # the 1-seed dominates every round it plays
    assert playoff_champion(seeds, dom) == 0
    ups = np.zeros((6, 3))
    ups[5, :] = 100  # the 6-seed runs the table (3v6, then 2-seed, then final)
    assert playoff_champion(seeds, ups) == 5


def test_bracket_tie_goes_to_higher_seed():
    seeds = [0, 1, 2, 3, 4, 5]
    scores = np.zeros((6, 3))  # every game tied → higher seed always advances → 1-seed champ
    assert playoff_champion(seeds, scores) == 0


def test_seed_order_breaks_ties_on_points():
    wins = np.array([10, 10, 5])
    points = np.array([900.0, 1000.0, 1200.0])
    assert seed_order(wins, points, 2) == [1, 0]  # equal wins → more points seeds higher


def test_resolve_records():
    scores = np.zeros((1, 2, 2))
    scores[0, 0, 0], scores[0, 1, 0] = 5, 3  # week1: team0 wins
    scores[0, 0, 1], scores[0, 1, 1] = 2, 4  # week2: team1 wins
    wins, points = resolve_records({1: [(0, 1)], 2: [(0, 1)]}, scores)
    assert list(wins[0]) == [1, 1]
    assert list(points[0]) == [7, 7]


# --------------------------------------------------------------------------- end-to-end
def _pool(n_teams=4, strong=0):
    rosters, pos, mean, names = [], [], [], []
    for t in range(n_teams):
        cols = []
        for p in STARTER_POS:
            cols.append(len(pos))
            pos.append(p)
            base = {"QB": 300, "RB": 200, "WR": 190, "TE": 130, "K": 130, "DEF": 120}[p]
            mean.append(base * (1.35 if t == strong else 1.0))
            names.append(f"T{t}-{p}")
        rosters.append(cols)
    return SeasonPool(
        rosters=rosters,
        pos=pos,
        mean=np.array(mean, dtype=float),
        cv=np.zeros(len(pos)),
        p_setback=np.zeros(len(pos)),
        severity=np.zeros(len(pos)),
        names=names,
        team_names=[f"Team{t}" for t in range(n_teams)],
        slots=SLOTS,
        n_teams=n_teams,
    )


def _run(pool, my_team=0, seed=0, n_sims=400):
    return simulate_season(
        pool,
        round_robin(pool.n_teams, [1, 2, 3]),
        my_team,
        season=2025,
        regular_weeks=[1, 2, 3],
        playoff_weeks=[4],  # 2-team, single final for the tiny league
        n_playoff_teams=2,
        n_sims=n_sims,
        seed=seed,
    )


def test_champ_probs_sum_to_one_and_reproducible():
    pool = _pool()
    a = _run(pool)
    b = _run(pool)
    for reg in ("equal_skill", "my_edge"):
        cp = a.regimes[reg].champ_prob
        assert abs(cp.sum() - 1.0) < 1e-9
        assert np.allclose(cp, b.regimes[reg].champ_prob)  # common seed → identical
        assert (a.regimes[reg].playoff_prob >= 0).all()
        assert (a.regimes[reg].playoff_prob <= 1).all()


def test_stacked_roster_wins_most_titles():
    out = _run(_pool(strong=0), my_team=0)
    cp = out.regimes["my_edge"].champ_prob
    assert cp.argmax() == 0  # the loaded roster takes the most titles
    assert cp[0] > 0.5


def test_startsit_edge_never_hurts_the_sharp_manager():
    # A roughly even league so the edge has room to show; my clean lineups shouldn't lower my title%.
    out = _run(_pool(strong=-1), my_team=0, n_sims=1500)  # strong=-1 → all teams equal
    base = out.regimes["equal_skill"].my_is_champ.mean()
    edge = out.regimes["my_edge"].my_is_champ.mean()
    assert edge >= base - 0.02

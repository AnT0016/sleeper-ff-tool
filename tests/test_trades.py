"""Offline tests for the player-for-player trade evaluator (no network).

Covers lineup valuation and the win-win finder: a complementary pair of rosters (I'm RB-deep / WR-poor,
the partner the reverse) yields a swap that upgrades **both** starting lineups; identical rosters yield
nothing; K/DEF are never offered. The live roster-fetch path is exercised via ``scripts/trade_finder.py``.
"""

from __future__ import annotations

from analysis.trades import TRADE_POSITIONS, find_trades, lineup_value
from optimizer.lineup import LineupPlayer

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1}


def _p(name, pos, proj):
    return LineupPlayer(player_id=name, name=name, pos=pos, team="X", proj_pts=proj)


def _fillers():
    """Constant QB/TE/K/DEF so only the RB/WR swap moves lineup value."""
    return [_p("qb", "QB", 200), _p("te", "TE", 150), _p("k", "K", 130), _p("def", "DEF", 120)]


def _rb_deep_wr_poor():
    # 3 startable RBs + a 4th on the bench (surplus); two weak WRs.
    return _fillers() + [
        _p("RB1", "RB", 250), _p("RB2", "RB", 240), _p("RB3", "RB", 235), _p("RB4", "RB", 200),
        _p("WR1", "WR", 100), _p("WR2", "WR", 90),
    ]


def _wr_deep_rb_poor():
    return _fillers() + [
        _p("pWR1", "WR", 230), _p("pWR2", "WR", 220), _p("pWR3", "WR", 210), _p("pWR4", "WR", 190),
        _p("pRB1", "RB", 120), _p("pRB2", "RB", 110),
    ]


def test_lineup_value_matches_greedy_optimal():
    # QB200 + RB250 + RB240 + WR100 + WR90 + TE150 + FLEX(RB235) + K130 + DEF120.
    assert lineup_value(_rb_deep_wr_poor(), SLOTS) == 200 + 250 + 240 + 100 + 90 + 150 + 235 + 130 + 120


def test_finds_win_win_complementary_swap():
    offers = find_trades(_rb_deep_wr_poor(), {7: _wr_deep_rb_poor()}, SLOTS, {7: "Partner"})
    assert offers, "a complementary RB-for-WR swap should exist"
    for o in offers:  # the win-win invariant holds for every returned offer
        assert o.my_gain > 0 and o.their_gain > 0
        assert o.give_pos in TRADE_POSITIONS and o.get_pos in TRADE_POSITIONS
    best = offers[0]
    assert best.give_pos == "RB" and best.get_pos == "WR"  # I give from RB surplus, get a WR upgrade
    assert best.partner == "Partner"


def test_identical_rosters_yield_no_trade():
    roster = _rb_deep_wr_poor()
    # A partner identical to me: no swap can improve either optimal lineup.
    assert find_trades(roster, {7: [_p(f"o_{p.name}", p.pos, p.proj_pts) for p in roster]}, SLOTS, {7: "Clone"}) == []


def test_kdef_never_offered():
    me = _fillers() + [_p("RB1", "RB", 250), _p("RB2", "RB", 240), _p("RB3", "RB", 235),
                       _p("WR1", "WR", 100), _p("WR2", "WR", 90), _p("K2", "K", 200)]
    them = _fillers() + [_p("pWR1", "WR", 230), _p("pWR2", "WR", 220), _p("pWR3", "WR", 210),
                         _p("pRB1", "RB", 120), _p("pRB2", "RB", 110), _p("dk", "DEF", 250)]
    for o in find_trades(me, {7: them}, SLOTS, {7: "P"}):
        assert "K" not in (o.give_pos, o.get_pos) and "DEF" not in (o.give_pos, o.get_pos)

"""Offline unit tests for Phase 5 team/league analysis (no network).

Covers the pure ``analysis.team`` views the hosted dashboard renders: per-team slot points, my
positional strength ranking + verdicts, bye-week gap detection (hole vs. thin), the positional-needs
fold, mutual-fit trade targets, and the Weeks 15-17 playoff outlook (which reuses the Phase 4 stash
ranker). The networked snapshot path (``analysis.snapshot``) is exercised manually via
``scripts/refresh_data.py``.
"""

from __future__ import annotations

from analysis.team import (
    PositionStrength,
    bye_week_gaps,
    playoff_outlook,
    position_strengths,
    positional_needs,
    slot_points_by_team,
    trade_targets,
)
from optimizer.lineup import LineupPlayer, StarterSpot

SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}


def _p(pid, pos, pts, *, team=None):
    return LineupPlayer(player_id=pid, name=pid, pos=pos, team=team or pos, proj_pts=pts)


def _spot(slot, pid, pos, pts, team=None):
    return StarterSpot(slot=slot, player=_p(pid, pos, pts, team=team))


def _strength(slot, verdict, *, rank=2, n=3):
    return PositionStrength(
        slot=slot, my_points=10.0, my_rank=rank, n_teams=n,
        league_avg=10.0, best=20.0, worst=5.0, verdict=verdict,
    )


# --------------------------------------------------------------------------- slot points
def test_slot_points_by_team_sums_by_slot():
    starters = {
        1: [_spot("QB", "q", "QB", 20.0), _spot("RB", "r1", "RB", 15.0),
            _spot("RB", "r2", "RB", 10.0), _spot("FLEX", "f", "WR", 8.0)],
    }
    sp = slot_points_by_team(starters)
    assert sp[1]["QB"] == 20.0
    assert sp[1]["RB"] == 25.0  # both RB starters summed
    assert sp[1]["FLEX"] == 8.0


# --------------------------------------------------------------------------- positional strength
def test_position_strengths_ranks_and_verdicts():
    sp = {1: {"QB": 20.0}, 2: {"QB": 15.0}, 3: {"QB": 10.0}}
    # middle team -> rank 2 of 3 -> average; carries league avg/best/worst.
    s = position_strengths(sp, my_roster_id=2, slot_order=["QB"])[0]
    assert (s.my_points, s.my_rank, s.n_teams) == (15.0, 2, 3)
    assert (s.league_avg, s.best, s.worst) == (15.0, 20.0, 10.0)
    assert s.verdict == "average"
    # best team -> strength; worst -> weakness.
    assert position_strengths(sp, 1, ["QB"])[0].verdict == "strength"
    assert position_strengths(sp, 3, ["QB"])[0].verdict == "weakness"


# --------------------------------------------------------------------------- bye-week gaps
def test_bye_week_gaps_flags_hole_and_thin():
    players = [
        _p("r1", "RB", 15.0, team="NE"),  # NE bye wk7
        _p("r2", "RB", 5.0, team="NE"),   # NE bye wk7
        _p("r3", "RB", 12.0, team="KC"),  # KC bye wk9 (a top-2 starter)
    ]
    bye = {"NE": 7, "KC": 9}
    gaps = {(g.week, g.pos): g for g in bye_week_gaps(players, bye, SLOTS, from_week=1)}
    # wk7: both NE RBs out -> only 1 available for 2 slots -> hole.
    assert gaps[(7, "RB")].severity == "hole" and gaps[(7, "RB")].available == 1
    assert set(gaps[(7, "RB")].idle) == {"r1", "r2"}
    # wk9: r3 (a top-2 RB) out, still 2 available for 2 slots -> forced backup -> thin.
    assert gaps[(9, "RB")].severity == "thin" and gaps[(9, "RB")].available == 2


def test_bye_week_gaps_respects_from_week():
    players = [_p("r1", "RB", 15.0, team="NE"), _p("r2", "RB", 5.0, team="NE")]
    bye = {"NE": 7}
    assert bye_week_gaps(players, bye, SLOTS, from_week=8) == []  # bye already passed


# --------------------------------------------------------------------------- positional needs
def test_positional_needs_folds_weakness_depth_and_holes():
    strengths = [
        _strength("QB", "strength"),
        _strength("RB", "weakness"),
        _strength("WR", "average"),
        _strength("TE", "average"),
    ]
    players = [
        _p("r1", "RB", 15.0), _p("r2", "RB", 5.0),                 # 2 RB, no bench depth
        _p("w1", "WR", 14.0), _p("w2", "WR", 11.0), _p("w3", "WR", 8.0),  # 3 WR, depth
        _p("t1", "TE", 9.0),                                       # 1 TE, no bench depth
        _p("q1", "QB", 20.0), _p("q2", "QB", 8.0),                 # 2 QB, depth
    ]
    bye_gaps = [type("G", (), {"pos": "RB", "severity": "hole"})()]
    needs = {n.pos: n for n in positional_needs(strengths, players, SLOTS, bye_gaps)}
    assert "QB" not in needs and "WR" not in needs            # strong / deep -> not needs
    assert needs["RB"].severity == "high" and len(needs["RB"].reasons) == 3
    assert needs["TE"].severity == "medium"                   # thin depth only


# --------------------------------------------------------------------------- trade targets
def test_trade_targets_finds_mutual_fit():
    # 4 teams; I'm roster 1, weak at RB, strong at WR.
    slot_points = {
        1: {"RB": 5.0, "WR": 20.0},   # me
        2: {"RB": 20.0, "WR": 6.0},   # strong RB (my need), weak WR (my surplus) -> fit
        3: {"RB": 18.0, "WR": 15.0},  # strong RB but not weak WR -> no give match
        4: {"RB": 8.0, "WR": 14.0},   # weak RB -> no get match
    }
    strengths = [_strength("RB", "weakness", rank=4, n=4), _strength("WR", "strength", rank=1, n=4)]
    names = {1: "Me", 2: "T2", 3: "T3", 4: "T4"}
    ideas = trade_targets(slot_points, my_roster_id=1, team_names=names, strengths=strengths)
    assert len(ideas) == 1
    idea = ideas[0]
    assert idea.team_name == "T2"
    assert idea.get_positions == ("RB",) and idea.give_positions == ("WR",)


def test_trade_targets_empty_without_strength_and_weakness():
    slot_points = {1: {"RB": 5.0, "WR": 20.0}, 2: {"RB": 20.0, "WR": 6.0}}
    only_weak = [_strength("RB", "weakness"), _strength("WR", "average")]
    assert trade_targets(slot_points, 1, {1: "Me", 2: "T2"}, only_weak) == []


# --------------------------------------------------------------------------- playoff outlook
def test_playoff_outlook_sos_tilts_my_starters():
    starters = [_p("a", "RB", 10.0, team="NE")]
    sos = {"DAL": {"RB": 1.5}}
    opp = {"NE": {15: "DAL", 16: "DAL", 17: "DAL"}}
    out = playoff_outlook(starters, sos, opp)
    assert out[0].raw_value == 30.0 and out[0].adj_value == 45.0 and out[0].sos_swing == 15.0
    assert len(out[0].weeks) == 3

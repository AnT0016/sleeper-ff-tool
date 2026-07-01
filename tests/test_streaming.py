"""Offline tests for the weekly K/DEF streaming guide + DEF strength-of-schedule (no network).

Covers the DEF SOS (re-scored DST points attributed to the offense faced → soft/tough multipliers,
merged into the shared sos map) and the streaming ranker (orders by this-week projection, computes the
edge over my current starter, stream/hold verdict, and handles empty positions). The live fetch path
is exercised via ``scripts/waiver_report.py`` / the snapshot.
"""

from __future__ import annotations

from waivers.sos import def_sos_multipliers, merge_sos, multiplier, points_allowed_to_def
from waivers.streaming import rank_streamers


# --------------------------------------------------------------------------- DEF strength-of-schedule
def test_points_allowed_to_def_attributes_to_offense():
    opp = {"SF": {1: "NYG", 2: "SEA"}, "DAL": {1: "SEA", 2: "NYG"}}
    dst = [
        {"team": "SF", "week": 1, "points": 12.0},  # NYG's offense let SF's D score 12
        {"team": "DAL", "week": 2, "points": 8.0},  # NYG let DAL score 8
        {"team": "DAL", "week": 1, "points": 2.0},  # SEA let DAL score 2
        {"team": "SF", "week": 2, "points": 4.0},  # SEA let SF score 4
    ]
    pa = points_allowed_to_def(dst, opp)
    assert pa["NYG"] == 10.0  # (12 + 8) / 2
    assert pa["SEA"] == 3.0  # (2 + 4) / 2


def test_def_sos_multipliers_flags_soft_and_tough():
    sos = def_sos_multipliers({"NYG": 10.0, "SEA": 3.0})  # league avg 6.5
    assert sos["NYG"]["DEF"] > 1.0  # generous offense → soft matchup to stream a D into
    assert sos["SEA"]["DEF"] < 1.0  # stingy offense → tough
    assert multiplier(sos, "NYG", "DEF") == sos["NYG"]["DEF"]


def test_def_sos_unknown_offense_defaults_flat():
    sos = def_sos_multipliers({"NYG": 10.0})
    assert multiplier(sos, "KC", "DEF") == 1.0  # unseen offense → no tilt


def test_merge_sos_combines_positions_per_team():
    merged = merge_sos({"NYG": {"WR": 1.2}}, {"NYG": {"DEF": 1.5}, "KC": {"DEF": 0.8}})
    assert merged["NYG"] == {"WR": 1.2, "DEF": 1.5}
    assert merged["KC"]["DEF"] == 0.8


# --------------------------------------------------------------------------- streaming ranker
def _cand(pid, pos, this_week, *, next_week=0.0, ros=0.0, playoff=0.0, team="X"):
    return {
        "player_id": pid, "name": pid, "pos": pos, "team": team,
        "this_week": this_week, "next_week": next_week, "ros_pg": ros, "playoff": playoff,
    }


def test_rank_streamers_orders_by_this_week_and_computes_gain():
    cands = [_cand("d1", "DEF", 10.0), _cand("d2", "DEF", 14.0), _cand("k1", "K", 8.0)]
    current = {"DEF": {"name": "myD", "this_week": 9.0}, "K": {"name": "myK", "this_week": 8.5}}
    advice = {a.pos: a for a in rank_streamers(cands, current, min_gain=1.5)}

    d = advice["DEF"]
    assert [o.name for o in d.options] == ["d2", "d1"]  # best this-week first
    assert d.options[0].gain == 5.0  # 14 − my 9
    assert d.verdict == "stream"  # +5 ≥ 1.5

    k = advice["K"]
    assert k.options[0].gain == -0.5  # 8 − my 8.5
    assert k.verdict == "hold"  # best edge below the threshold


def test_rank_streamers_no_candidates_but_current_starter():
    advice = rank_streamers([], {"DEF": {"name": "myD", "this_week": 9.0}})
    d = next(a for a in advice if a.pos == "DEF")
    assert d.options == ()
    assert d.verdict == "hold"
    assert d.current_name == "myD"


def test_rank_streamers_skips_positions_with_neither_candidate_nor_starter():
    advice = rank_streamers([_cand("d1", "DEF", 10.0)], {})
    assert {a.pos for a in advice} == {"DEF"}  # no K candidate and no K starter → no K row

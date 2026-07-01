"""Offline unit tests for Phase 5 team/league analysis (no network).

Covers the pure ``analysis.team`` views the hosted dashboard renders: per-team slot points, my
positional strength ranking + verdicts, bye-week gap detection (hole vs. thin), the positional-needs
fold, mutual-fit trade targets, and the Weeks 15-17 playoff outlook (which reuses the Phase 4 stash
ranker). The networked snapshot path (``analysis.snapshot``) is exercised manually via
``scripts/refresh_data.py``.
"""

from __future__ import annotations

import polars as pl

from analysis import backtest
from analysis.backtest import lineup_from_points, optimal_standings, simulate_draft
from analysis.snapshot import kickoff_by_team, offseason_skip_reason
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


# --------------------------------------------------------------------------- backtest helpers
def test_simulate_draft_takes_best_vor_from_real_remaining_pool():
    picks = [
        {"pick_no": 1, "round": 1, "picked_by": "X", "player_id": "A"},   # other team takes A
        {"pick_no": 2, "round": 1, "picked_by": "ME", "player_id": "Z1"},  # my pick -> tool acts
        {"pick_no": 3, "round": 1, "picked_by": "X", "player_id": "B"},
        {"pick_no": 4, "round": 1, "picked_by": "ME", "player_id": "Z2"},
    ]
    vor_order = {"A": -10.0, "B": -9.0, "C": -8.0, "D": -7.0}  # -vor: smallest key = best VOR
    pos = {"A": "RB", "B": "RB", "C": "RB", "D": "RB"}
    rows = simulate_draft(picks, "ME", vor_order, pos, default_cap=99)
    # A already gone -> tool takes B (9), then C (8); my actual picks recorded as my_pid.
    assert [r["tool_pid"] for r in rows] == ["B", "C"]
    assert [r["my_pid"] for r in rows] == ["Z1", "Z2"]


def test_simulate_draft_respects_positional_caps():
    picks = [
        {"pick_no": 1, "round": 1, "picked_by": "ME", "player_id": "Z1"},
        {"pick_no": 2, "round": 1, "picked_by": "ME", "player_id": "Z2"},
    ]
    vor_order = {"Q1": -10.0, "Q2": -9.0, "R1": -5.0}
    pos = {"Q1": "QB", "Q2": "QB", "R1": "RB"}
    rows = simulate_draft(picks, "ME", vor_order, pos, caps={"QB": 1}, default_cap=99)
    # QB cap of 1: tool takes the top QB once, then must take the RB (not a 2nd QB).
    assert [r["tool_pid"] for r in rows] == ["Q1", "R1"]


def test_simulate_draft_adp_baseline_takes_lowest_adp():
    picks = [
        {"pick_no": 1, "round": 1, "picked_by": "ME", "player_id": "Z1"},
        {"pick_no": 2, "round": 1, "picked_by": "ME", "player_id": "Z2"},
    ]
    # ADP order: lower drafts first (order_key == ADP). B (adp 3) then A (adp 5).
    adp_order = {"A": 5.0, "B": 3.0, "C": 8.0}
    pos = {"A": "RB", "B": "WR", "C": "RB"}
    rows = simulate_draft(picks, "ME", adp_order, pos, default_cap=99)
    assert [r["tool_pid"] for r in rows] == ["B", "A"]


def test_lineup_from_points_picks_best_legal_lineup():
    players_map = {
        "q1": {"position": "QB", "team": "KC", "full_name": "QB1"},
        "r1": {"position": "RB", "team": "SF", "full_name": "RB1"},
        "r2": {"position": "RB", "team": "DET", "full_name": "RB2"},
        "w1": {"position": "WR", "team": "MIA", "full_name": "WR1"},
    }
    pts = {"q1": 20.0, "r1": 15.0, "r2": 10.0, "w1": 12.0}
    sol = lineup_from_points(["q1", "r1", "r2", "w1"], pts, players_map, {"QB": 1, "RB": 1, "FLEX": 1})
    # QB q1 + RB r1 + FLEX best leftover (w1 12 > r2 10) = 47.
    assert sol.total == 47.0
    assert {sp.player.player_id for sp in sol.starters} == {"q1", "r1", "w1"}


def test_optimal_standings_subs_my_optimal_into_head_to_head():
    # 1 week, I actually lost 100-110; my optimal (115) flips it -> I should rank above the opponent.
    matchups = {
        1: [
            {"roster_id": 1, "matchup_id": 9, "points": 100.0},
            {"roster_id": 2, "matchup_id": 9, "points": 110.0},
        ]
    }
    assert optimal_standings(matchups, my_roster_id=1, my_optimal_by_week={1: 115.0}) == [1, 2]


# --------------------------------------------------------------------------- full-view helpers
def test_starting_slot_labels_excludes_bench_and_ir():
    rp = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", "IR"]
    assert backtest.starting_slot_labels(rp) == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]


def test_draftboard_rows_grade_flag_and_name_fallback():
    picks = [
        {"pick_no": 1, "round": 1, "draft_slot": 1, "roster_id": 5, "player_id": "100",
         "metadata": {"first_name": "Alpha", "last_name": "Back", "position": "RB", "team": "PHI"}},
        {"pick_no": 2, "round": 1, "draft_slot": 2, "roster_id": 9, "player_id": "PHI",
         "metadata": {"position": "DEF"}},
    ]
    rows = backtest.draftboard_rows(
        picks, {5: "Me", 9: "Them"}, {"100": "Alpha Back"}, {"100": "RB"}, {"100": 120.0}, my_rid=5
    )
    assert rows[0]["player"] == "Alpha Back" and rows[0]["is_mine"] and rows[0]["season_pts"] == 120.0
    assert rows[1]["team"] == "Them" and rows[1]["is_mine"] is False
    assert rows[1]["player"] == "PHI"  # not on the board, no name -> the id itself


def test_matchup_detail_aligns_slots_and_flags_empty():
    pm = {"1": {"full_name": "My QB"}, "2": {"full_name": "My RB"}, "3": {"full_name": "Opp QB"}}
    rows = backtest.matchup_detail_rows(
        3, ["QB", "RB"], ["1", "2"], ["3", "0"], {"1": 20.0, "2": 10.0}, {"3": 15.0}, {}, pm
    )
    assert rows[0]["slot"] == "QB" and rows[0]["my_player"] == "My QB" and rows[0]["my_pts"] == 20.0
    assert rows[0]["opp_player"] == "Opp QB" and rows[0]["opp_pts"] == 15.0
    assert rows[1]["opp_player"] == "(empty)"  # a "0" starter slot


def test_transaction_rows_split_per_roster_and_drop_incomplete():
    txns = [
        {"type": "waiver", "status": "complete", "adds": {"50": 4}, "drops": {"60": 4},
         "roster_ids": [4]},
        {"type": "trade", "status": "complete", "adds": {"70": 4, "80": 7},
         "drops": {"70": 7, "80": 4}, "roster_ids": [4, 7]},
        {"type": "free_agent", "status": "failed", "adds": {"99": 4}, "roster_ids": [4]},
    ]
    pm = {"50": {"full_name": "Add1"}, "60": {"full_name": "Drop1"},
          "70": {"full_name": "P70"}, "80": {"full_name": "P80"}}
    rows = backtest.transaction_rows(2, txns, {4: "Me", 7: "You"}, {}, pm, my_rid=4)
    waiver = [r for r in rows if r["type"] == "waiver"]
    assert len(waiver) == 1 and waiver[0]["added"] == "Add1" and waiver[0]["dropped"] == "Drop1"
    assert waiver[0]["is_mine"]
    trade = [r for r in rows if r["type"] == "trade"]
    assert len(trade) == 2  # one row per roster involved
    me = next(r for r in trade if r["team"] == "Me")
    assert me["added"] == "P70" and me["dropped"] == "P80"
    assert not any(r["type"] == "free_agent" for r in rows)  # failed transaction filtered out


def test_kickoff_by_team_labels_and_normalizes():
    sched = pl.DataFrame(
        {
            "season": [2025, 2025],
            "game_type": ["REG", "PRE"],
            "week": [5, 5],
            "weekday": ["Sunday", "Sunday"],
            "gametime": ["13:00", "13:00"],
            "home_team": ["LA", "ZZZ"],  # LA -> LAR; the PRE game must be ignored
            "away_team": ["BUF", "YYY"],
        }
    )
    k = kickoff_by_team(2025, 5, fetch_schedules=lambda s: sched)
    assert k["LAR"] == "Sun 13:00 vs BUF"
    assert k["BUF"] == "Sun 13:00 @ LAR"
    assert "ZZZ" not in k and "YYY" not in k  # preseason filtered out


def test_frozen_projections_round_trip(tmp_path):
    from data.frozen import frozen_fetch, load_frozen_rows, save_frozen

    rows = [{"player_id": "1", "player": {"position": "RB"}, "stats": {"pts_half_ppr": 210.0, "adp_half_ppr": 3.0}}]
    save_frozen(2026, rows, frozen_at="2026-07-01", directory=tmp_path)
    assert load_frozen_rows(2026, directory=tmp_path) == rows
    # a build_board-compatible fetch returns the frozen rows regardless of args
    fetch = frozen_fetch(2026, directory=tmp_path)
    assert fetch(2026, positions=("RB",)) == rows


def test_offseason_skip_reason():
    # Auto scheduled run: skip in the off/pre-season, run in the regular season.
    assert offseason_skip_reason({"season_type": "off", "week": 0}, None, None)
    assert offseason_skip_reason({"season_type": "pre", "week": 0}, None, None)
    assert offseason_skip_reason({"season_type": "regular", "week": 3}, None, None) is None
    # An explicit week/season override always runs (manual backfill), even off-season.
    assert offseason_skip_reason({"season_type": "off"}, 15, 2025) is None
    # Unknown/empty state falls through to a normal build (offline dev).
    assert offseason_skip_reason({}, None, None) is None

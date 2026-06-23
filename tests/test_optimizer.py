"""Offline unit tests for the Phase 3 weekly lineup optimizer (no network).

Covers the ILP slot constraints (with FLEX the headline case), bye/OUT/IR exclusion and hole
reporting, the start/sit delta table, the risky-start flags, and the pure input-join helpers
(``assemble_players`` / ``bye_teams``). The live Sleeper/nflverse fetch path is exercised manually
via ``scripts/optimize_lineup.py``.
"""

from __future__ import annotations

import polars as pl

from optimizer.inputs import assemble_players, bye_teams, score_projections
from optimizer.lineup import LineupPlayer, lineup_slots, optimize
from optimizer.startsit import idle_players, risky_starts, start_sit_table

# Our league's startable slots.
SLOTS = lineup_slots(
    ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", "BN", "BN", "BN", "IR"]
)


def _p(pid, pos, pts, *, status=None, on_bye=False, out=False, on_ir=False):
    return LineupPlayer(
        player_id=pid, name=pid, pos=pos, team=pos, proj_pts=pts,
        status=status, on_bye=on_bye, out=out, on_ir=on_ir,
    )


def _full_roster():
    """A healthy 14-player roster with distinct projections (FLEX winner = RB r3 @ 16)."""
    return [
        _p("q1", "QB", 25.0), _p("q2", "QB", 18.0),
        _p("r1", "RB", 20.0), _p("r2", "RB", 18.0), _p("r3", "RB", 16.0), _p("r4", "RB", 8.0),
        _p("w1", "WR", 19.0), _p("w2", "WR", 15.0), _p("w3", "WR", 14.0), _p("w4", "WR", 9.0),
        _p("t1", "TE", 12.0), _p("t2", "TE", 6.0),
        _p("k1", "K", 9.0),
        _p("d1", "DEF", 10.0),
    ]


# --------------------------------------------------------------------------- slot model
def test_lineup_slots_counts_startable_only():
    assert dict(SLOTS) == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1}
    assert list(SLOTS) == ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"]  # display order


# --------------------------------------------------------------------------- core LP / FLEX
def test_optimize_fills_exact_slots():
    sol = optimize(_full_roster(), SLOTS)
    filled = {}
    for sp in sol.starters:
        filled[sp.slot] = filled.get(sp.slot, 0) + 1
    assert filled == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1}
    assert not sol.holes
    assert len(sol.bench) == 5  # 14 rostered - 9 starters


def test_flex_takes_best_leftover_skill_player():
    sol = optimize(_full_roster(), SLOTS)
    flex = next(sp.player for sp in sol.starters if sp.slot == "FLEX")
    # Leftover skill players are r3(16), w3(14), r4(8), w4(9), t2(6); best is r3.
    assert flex.player_id == "r3"
    assert flex.pos in {"RB", "WR", "TE"}


def test_flex_never_takes_qb_even_if_higher_projected():
    # q2 (QB, 18) out-projects the best leftover skill player r3 (16) but is FLEX-ineligible.
    sol = optimize(_full_roster(), SLOTS)
    flex = next(sp.player for sp in sol.starters if sp.slot == "FLEX")
    assert flex.player_id == "r3"
    assert "q2" in {b.player_id for b in sol.bench}


def test_optimize_maximizes_total():
    sol = optimize(_full_roster(), SLOTS)
    # q1+r1+r2+w1+w2+t1+r3(flex)+k1+d1
    assert sol.total == 25 + 20 + 18 + 19 + 15 + 12 + 16 + 9 + 10


# --------------------------------------------------------------------------- exclusions / holes
def test_bye_out_ir_players_are_never_started():
    roster = _full_roster()
    # Knock out the would-be starters; their slots should fall to the next-best eligible player.
    by_id = {p.player_id: p for p in roster}
    by_id["q1"].out = True
    by_id["r1"].on_bye = True
    by_id["w1"].on_ir = True
    sol = optimize(roster, SLOTS)
    started = {sp.player.player_id for sp in sol.starters}
    assert {"q1", "r1", "w1"}.isdisjoint(started)
    assert "q2" in started  # the only other QB now starts


def test_unfillable_slot_reported_as_hole():
    # One QB only, and he is OUT -> the QB slot cannot be filled.
    roster = [
        _p("q1", "QB", 20.0, out=True),
        _p("r1", "RB", 15.0), _p("r2", "RB", 14.0),
        _p("w1", "WR", 13.0), _p("w2", "WR", 12.0),
        _p("t1", "TE", 10.0), _p("k1", "K", 8.0), _p("d1", "DEF", 9.0),
        _p("r3", "RB", 7.0),  # fills FLEX
    ]
    sol = optimize(roster, SLOTS)
    assert sol.holes == {"QB": 1}
    assert "Optimal" in sol.status  # still solves, just partially filled
    assert all(sp.slot != "QB" for sp in sol.starters)


def test_zero_projection_player_still_fills_an_empty_slot():
    # Only DEF available is 0.0-projected; the fill nudge should still start it (no phantom hole).
    roster = [
        _p("q1", "QB", 20.0),
        _p("r1", "RB", 15.0), _p("r2", "RB", 14.0), _p("r3", "RB", 7.0),
        _p("w1", "WR", 13.0), _p("w2", "WR", 12.0),
        _p("t1", "TE", 10.0), _p("k1", "K", 8.0),
        _p("d1", "DEF", 0.0),
    ]
    sol = optimize(roster, SLOTS)
    assert "DEF" not in sol.holes
    assert any(sp.slot == "DEF" and sp.player.player_id == "d1" for sp in sol.starters)


# --------------------------------------------------------------------------- start/sit table
def test_start_sit_deltas_are_nonpositive_and_target_the_right_starter():
    sol = optimize(_full_roster(), SLOTS)
    table = start_sit_table(sol)
    assert {d.player.player_id for d in table} == {"q2", "r4", "w3", "w4", "t2"}
    assert all(d.delta <= 0 for d in table)  # optimal lineup -> no bench upgrade
    # w3 (WR 14) would replace the weakest WR/FLEX starter: w2 (15) -> delta -1, the closest call.
    w3 = next(d for d in table if d.player.player_id == "w3")
    assert w3.would_replace.player_id == "w2" and w3.delta == -1.0
    assert table[0].player.player_id == "w3"  # sorted closest-first
    # q2 (QB) can only replace the QB starter q1.
    q2 = next(d for d in table if d.player.player_id == "q2")
    assert q2.would_replace.player_id == "q1" and q2.delta == -7.0


# --------------------------------------------------------------------------- risky starts
def test_risky_flag_for_questionable_starter():
    roster = _full_roster()
    next(p for p in roster if p.player_id == "q1").status = "Questionable"
    flags = risky_starts(optimize(roster, SLOTS), roster)
    q1 = next(f for f in flags if f.player.player_id == "q1")
    assert "Questionable" in q1.reasons


def test_forced_downgrade_flag_when_better_player_unavailable():
    roster = _full_roster()
    roster.append(_p("r0", "RB", 30.0, on_bye=True))  # a stud RB on bye
    sol = optimize(roster, SLOTS)
    flags = risky_starts(sol, roster)
    # The weakest eligible starter r0 could have replaced is the FLEX (r3 @ 16); flag it.
    forced = next(f for f in flags if f.player.player_id == "r3")
    assert any("forced" in r and "r0" in r for r in forced.reasons)


def test_idle_players_lists_bye_out_ir_best_first():
    roster = _full_roster()
    by_id = {p.player_id: p for p in roster}
    by_id["r1"].on_bye = True
    by_id["w1"].out = True
    idle = idle_players(roster)
    assert [p.player_id for p, _ in idle] == ["r1", "w1"]  # 20 then 19
    assert dict((p.player_id, reason) for p, reason in idle) == {"r1": "BYE", "w1": "OUT"}


# --------------------------------------------------------------------------- input-join helpers
def test_assemble_players_join_and_eligibility():
    roster = {
        "owner_id": "ME",
        "players": ["1", "2", "3", "PHI", "9", "8", "99"],
        "reserve": ["9"],
        "taxi": ["8"],
    }
    players_map = {
        "1": {"position": "QB", "full_name": "Q One", "team": "BUF"},
        "2": {"position": "RB", "full_name": "R Two", "team": "DAL", "injury_status": "Questionable"},
        "3": {"position": "WR", "full_name": "W Three", "team": "MIA", "injury_status": "Out"},
        "PHI": {"position": "DEF"},
        "9": {"position": "RB", "full_name": "IR Guy", "team": "GB"},
        "8": {"position": "RB", "full_name": "Taxi Guy", "team": "NYJ"},
        "99": {"position": "RB", "full_name": "Deep Back", "team": "SF"},
    }
    scored = {
        "1": {"proj": 20.0, "pos": "QB", "team": "BUF", "name": "Q One"},
        "2": {"proj": 15.0, "pos": "RB", "team": "DAL", "name": "R Two"},
        "3": {"proj": 10.0, "pos": "WR", "team": "MIA", "name": "W Three"},
        "PHI": {"proj": 8.0, "pos": "DEF", "team": "PHI", "name": "Eagles"},
        "9": {"proj": 5.0, "pos": "RB", "team": "GB", "name": "IR Guy"},
        # "99" has no projection row -> unjoined; "8" is taxi -> skipped entirely.
    }
    players, unjoined = assemble_players(roster, players_map, scored, byes={"PHI"})
    by_id = {p.player_id: p for p in players}

    assert "8" not in by_id  # taxi excluded
    assert unjoined == [("99", "Deep Back")] and by_id["99"].proj_pts == 0.0
    assert by_id["2"].status == "Questionable" and by_id["2"].eligible  # Q is startable
    assert by_id["3"].out and not by_id["3"].eligible  # OUT excluded
    assert by_id["PHI"].on_bye and by_id["PHI"].team == "PHI"  # DEF id == team, on bye
    assert by_id["9"].on_ir and not by_id["9"].eligible  # IR slot excluded


def test_score_projections_keeps_best_row_per_id():
    scoring = {"rec": 0.5, "rec_yd": 0.1}
    rows = [
        {"player_id": "x", "player": {"position": "WR", "team": "KC", "full_name": "X"},
         "stats": {"rec": 5, "rec_yd": 60}},  # 2.5 + 6 = 8.5
        {"player_id": "x", "player": {"position": "WR", "team": "KC", "full_name": "X"},
         "stats": {"rec": 4, "rec_yd": 40}},  # 2 + 4 = 6.0 (lower; ignored)
    ]
    scored = score_projections(rows, scoring)
    assert scored["x"]["proj"] == 8.5


def test_bye_teams_from_schedule_with_abbrev_normalization():
    sched = pl.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "game_type": ["REG", "REG", "REG"],
            "week": [1, 2, 2],
            "home_team": ["AAA", "CCC", "LA"],   # nflverse "LA" -> Sleeper "LAR"
            "away_team": ["BBB", "DDD", "EEE"],
        }
    )
    byes = bye_teams(2025, 1, fetch_schedules=lambda s: sched)
    assert byes == {"CCC", "DDD", "LAR", "EEE"}  # everyone not playing in week 1

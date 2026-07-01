"""Offline unit tests for the Phase 2 draft logic (no network).

Covers the pure helpers the live tracker relies on: snake pick numbers, ADP survival, VOR with
data-driven FLEX allocation, per-position tiers, roster-need tracking, positional runs, and the
live-pick helpers. The Streamlit UI and live Sleeper calls are validated manually against the 2025
completed draft (see docs/PROGRESS.md).
"""

from __future__ import annotations

from draft import roster, snake
from draft.grade import grade_draft, positional_ranks, team_picks
from draft.vor import add_vor, replacement_levels, tierize
from projections.board import PlayerRow


# --------------------------------------------------------------------------- snake math
def test_my_pick_numbers_slot7_12team():
    # CLAUDE.md spec: R odd -> (r-1)*12+S ; R even -> r*12-S+1. Slot 7.
    picks = snake.my_pick_numbers(slot=7, teams=12, rounds=5)
    assert picks == [7, 18, 31, 42, 55]


def test_my_pick_numbers_endpoints():
    assert snake.my_pick_numbers(1, 12, 3) == [1, 24, 25]   # turn (slot 1): back-to-back at 24/25
    assert snake.my_pick_numbers(12, 12, 3) == [12, 13, 36]  # turn (slot 12)


def test_upcoming_picks():
    mine = [7, 18, 31, 42]
    assert snake.upcoming_picks(mine, picks_made=18) == [31, 42]
    assert snake.upcoming_picks(mine, picks_made=0) == mine


def test_survival_labels():
    assert snake.survival(40.0, pick_no=31, cushion=6) == snake.AVAILABLE  # ADP well after our pick
    assert snake.survival(31.0, pick_no=31, cushion=6) == snake.TOSSUP     # right at our pick
    assert snake.survival(10.0, pick_no=31, cushion=6) == snake.GONE       # long gone
    assert snake.survival(float("inf"), pick_no=1) == snake.AVAILABLE      # effectively undrafted


# --------------------------------------------------------------------------- VOR + tiers
def _row(pid, pos, pts, adp=float("inf")):
    return PlayerRow(player_id=pid, name=pid, pos=pos, team=None, proj_pts=pts, adp=adp)


def test_replacement_uses_first_non_starter():
    # 3 QBs, 2 starters league-wide -> replacement = the 3rd QB (first non-starter) = 100.
    board = [_row("q1", "QB", 300), _row("q2", "QB", 200), _row("q3", "QB", 100)]
    repl = replacement_levels(board, {"QB": 2})
    assert repl["QB"] == 100.0


def test_flex_allocation_deepens_replacement():
    # 2 base RB starters, 2 base WR starters, plus 1 FLEX. The best leftover (RB3=120 > WR3=80)
    # takes the flex, so RB replacement drops to the 4th RB (90), WR stays at the 3rd WR (80).
    board = [
        _row("r1", "RB", 200), _row("r2", "RB", 150), _row("r3", "RB", 120), _row("r4", "RB", 90),
        _row("w1", "WR", 190), _row("w2", "WR", 140), _row("w3", "WR", 80), _row("w4", "WR", 70),
    ]
    repl = replacement_levels(board, {"RB": 2, "WR": 2}, flex_slots=1, flex_positions=("RB", "WR"))
    assert repl["RB"] == 90.0   # flex consumed RB3 -> first non-starter is RB4
    assert repl["WR"] == 80.0   # WR3 untouched


def test_add_vor_subtracts_replacement():
    board = [_row("a", "RB", 200), _row("b", "RB", 100)]
    add_vor(board, {"RB": 100.0})
    assert board[0].vor == 100.0 and board[1].vor == 0.0


def test_tierize_breaks_on_gap():
    # Tight cluster (100/98/96) then a cliff to 60: the cliff should start a new tier.
    board = [_row("a", "WR", 100), _row("b", "WR", 98), _row("c", "WR", 96), _row("d", "WR", 60)]
    add_vor(board, {"WR": 0.0})
    tierize(board, by="vor", tier_depth=10)
    tiers = {p.player_id: p.tier for p in board}
    assert tiers["a"] == tiers["b"] == tiers["c"] == 1
    assert tiers["d"] == 2


# --------------------------------------------------------------------------- roster config + needs
SETTINGS = {
    "teams": 12, "rounds": 14, "slots_qb": 1, "slots_rb": 2, "slots_wr": 2,
    "slots_te": 1, "slots_flex": 1, "slots_k": 1, "slots_def": 1, "slots_bn": 5,
}


def test_roster_config_and_baselines():
    cfg = roster.roster_config(SETTINGS)
    assert cfg.teams == 12 and cfg.rounds == 14
    base = roster.base_starters(cfg)
    assert base == {"QB": 12, "RB": 24, "WR": 24, "TE": 12, "K": 12, "DEF": 12}
    assert roster.flex_slots_total(cfg) == 12


def test_roster_status_fills_dedicated_then_flex():
    cfg = roster.roster_config(SETTINGS)
    # 1 QB, 3 RB, 1 WR drafted: 2 RB fill RB slots, the 3rd RB fills FLEX; WR still needs 1 more.
    status = roster.roster_status(["QB", "RB", "RB", "RB", "WR"], cfg)
    assert status["RB"]["filled"] == 2 and status["RB"]["need"] == 0
    assert status["FLEX"]["filled"] == 1 and status["FLEX"]["need"] == 0
    assert status["WR"]["filled"] == 1 and status["WR"]["need"] == 1
    assert status["TE"]["need"] == 1 and status["K"]["need"] == 1


def test_needed_positions_includes_flex_eligible_when_flex_open():
    cfg = roster.roster_config(SETTINGS)
    status = roster.roster_status(["QB"], cfg)  # nothing but QB -> flex open
    needs = roster.needed_positions(status, cfg)
    assert "QB" not in needs  # QB slot filled
    assert {"RB", "WR", "TE"} <= needs  # flex-eligible all flagged while FLEX open


# --------------------------------------------------------------------------- live-pick helpers
def _pick(no, pid, pos, by, slot=1, rnd=1):
    return {"pick_no": no, "round": rnd, "draft_slot": slot, "picked_by": by,
            "player_id": pid, "metadata": {"position": pos}}


def test_drafted_ids_and_my_drafted_and_diff():
    picks = [_pick(1, "7564", "WR", "U1"), _pick(2, "PHI", "DEF", "ME"), _pick(3, "9509", "RB", "ME")]
    assert roster.drafted_ids(picks) == {"7564", "PHI", "9509"}
    mine = roster.my_drafted(picks, "ME")
    assert [p["pick_no"] for p in mine] == [2, 3]
    assert [p["pick_no"] for p in roster.new_since(picks, last_pick_no=1)] == [2, 3]


def test_positional_runs_window():
    positions = ["RB", "RB", "WR", "RB", "WR", "WR"]
    runs = roster.positional_runs(positions, window=3)  # last 3: RB, WR, WR
    assert runs["WR"] == 2 and runs["RB"] == 1


# --------------------------------------------------------------------------- draft grades
def test_team_picks_groups_by_slot_in_pick_order():
    picks = [
        {"pick_no": 1, "draft_slot": 1, "player_id": "a"},
        {"pick_no": 2, "draft_slot": 2, "player_id": "b"},
        {"pick_no": 3, "draft_slot": 2, "player_id": "c"},
    ]
    tp = team_picks(picks)
    assert tp[1] == ["a"] and tp[2] == ["b", "c"]


def test_grade_draft_ranks_grades_and_positional():
    # 2-team league, 1 QB + 1 RB starter each.
    slots = {"QB": 1, "RB": 1, "WR": 0, "TE": 0, "FLEX": 0, "K": 0, "DEF": 0}
    board = [_row("q1", "QB", 300), _row("q2", "QB", 100), _row("r1", "RB", 200), _row("r2", "RB", 50)]
    picks = [
        {"pick_no": 1, "draft_slot": 1, "player_id": "q1", "metadata": {"position": "QB"}},
        {"pick_no": 2, "draft_slot": 2, "player_id": "q2", "metadata": {"position": "QB"}},
        {"pick_no": 3, "draft_slot": 2, "player_id": "r1", "metadata": {"position": "RB"}},
        {"pick_no": 4, "draft_slot": 1, "player_id": "r2", "metadata": {"position": "RB"}},
    ]
    grades = grade_draft(picks, board, slots, teams=2, my_slot=1)
    g1 = next(g for g in grades if g.slot == 1)
    g2 = next(g for g in grades if g.slot == 2)
    assert g1.starters_pts == 350.0 and g2.starters_pts == 300.0  # 300+50 vs 100+200
    assert g1.rank == 1 and g1.is_me and g1.grade == "A"
    # positional: my QB is best (1/2), my RB is worst (2/2)
    pr = positional_ranks(grades, my_slot=1)
    assert pr["QB"] == (1, 2) and pr["RB"] == (2, 2)

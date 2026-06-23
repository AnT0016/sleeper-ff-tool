"""Offline unit tests for Phase 4 waiver / stash / handcuff intelligence (no network).

Covers the league/roster helpers (free-agent split, standings, reverse-priority scarcity), the
handcuff next-man-up detector (+ URGENT escalation + gap walk), the reverse-priority spend advice
(integrated against the real Phase 3 optimizer), the playoff SOS math, the stash ranker + bye-week
suggestions, and the usage-signal joins. The live Sleeper/nflverse fetch path is exercised manually
via ``scripts/waiver_report.py``.
"""

from __future__ import annotations

import polars as pl

from optimizer.lineup import LineupPlayer, lineup_slots
from waivers.handcuffs import find_handcuffs
from waivers.league import (
    free_agents,
    my_standing,
    priority_scarcity,
    rostered_player_ids,
    standings,
)
from waivers.priority import SpendCandidate, lineup_gain, spend_advice
from waivers.sos import (
    multiplier,
    normalize_team,
    opponents_by_week,
    points_allowed_by_position,
    sos_multipliers,
)
from waivers.stash import bye_stash_suggestions, rank_playoff_stashes
from waivers.usage import usage_signals

SLOTS = lineup_slots(
    ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", "BN", "BN", "BN", "IR"]
)


def _p(pid, pos, pts, *, team=None, status=None, on_bye=False, out=False):
    return LineupPlayer(
        player_id=pid, name=pid, pos=pos, team=team or pos, proj_pts=pts,
        status=status, on_bye=on_bye, out=out,
    )


# --------------------------------------------------------------------------- league / rosters
def test_free_agents_excludes_players_reserve_and_taxi():
    rosters = [
        {"players": ["1", "2"], "reserve": ["3"], "taxi": ["4"]},
        {"players": ["5"]},
    ]
    assert rostered_player_ids(rosters) == {"1", "2", "3", "4", "5"}
    assert free_agents(["1", "5", "6", "7"], rosters) == {"6", "7"}


def _roster(rid, owner, wins, fpts, fpts_dec=0):
    return {
        "roster_id": rid, "owner_id": owner,
        "settings": {"wins": wins, "losses": 14 - wins, "ties": 0,
                     "fpts": fpts, "fpts_decimal": fpts_dec, "waiver_position": rid},
    }


def test_standings_order_by_wins_then_points():
    rosters = [
        _roster(1, "A", 8, 1500),
        _roster(2, "B", 10, 1400),
        _roster(3, "C", 8, 1520),  # ties A on wins, beats on points
    ]
    table = standings(rosters)
    assert [t.owner_id for t in table] == ["B", "C", "A"]
    assert [t.rank for t in table] == [1, 2, 3]
    assert my_standing(rosters, "A").rank == 3


def test_priority_scarcity_posture_by_standings_position():
    assert priority_scarcity(1, 12).posture == "selective"   # top of standings -> scarce claim
    assert priority_scarcity(6, 12).posture == "balanced"    # mid-table
    assert priority_scarcity(12, 12).posture == "aggressive"  # bottom -> durable high priority


# --------------------------------------------------------------------------- handcuffs
def _map(*rows):
    """rows: (pid, team, pos, dc_order)."""
    return {
        pid: {"team": team, "position": pos, "depth_chart_position": pos,
              "depth_chart_order": order, "full_name": pid}
        for pid, team, pos, order in rows
    }


def test_handcuff_flags_unrostered_next_man_up():
    pmap = _map(("star", "DET", "RB", 1), ("back", "DET", "RB", 2))
    alerts = find_handcuffs([_p("star", "RB", 18.0, team="DET")], pmap, free_agent_ids={"back"})
    assert len(alerts) == 1
    a = alerts[0]
    assert a.backup_id == "back" and a.gap == 0 and a.priority == "HIGH"


def test_handcuff_escalates_to_urgent_when_starter_out():
    pmap = _map(("star", "DET", "RB", 1), ("back", "DET", "RB", 2))
    alerts = find_handcuffs(
        [_p("star", "RB", 18.0, team="DET", status="Out")], pmap, free_agent_ids={"back"}
    )
    assert alerts[0].priority == "URGENT"


def test_handcuff_walks_past_rostered_backup_to_first_available():
    pmap = _map(("star", "DET", "RB", 1), ("mid", "DET", "RB", 2), ("deep", "DET", "RB", 3))
    # "mid" is rostered; only "deep" is a free agent.
    alerts = find_handcuffs([_p("star", "RB", 18.0, team="DET")], pmap, free_agent_ids={"deep"})
    assert alerts[0].backup_id == "deep" and alerts[0].gap == 1


def test_handcuff_none_when_backup_rostered():
    pmap = _map(("star", "DET", "RB", 1), ("back", "DET", "RB", 2))
    assert find_handcuffs([_p("star", "RB", 18.0, team="DET")], pmap, free_agent_ids=set()) == []


def test_handcuff_skips_kicker_and_defense():
    pmap = _map(("k1", "DET", "K", 1), ("k2", "DET", "K", 2))
    assert find_handcuffs([_p("k1", "K", 9.0, team="DET")], pmap, free_agent_ids={"k2"}) == []


# --------------------------------------------------------------------------- spend advice
def _my_full_roster():
    # exactly fills 9 starters, no bench: QB RB RB WR WR TE FLEX(=r3) K DEF -> total 104
    return [
        _p("q1", "QB", 20.0), _p("r1", "RB", 15.0), _p("r2", "RB", 12.0),
        _p("w1", "WR", 14.0), _p("w2", "WR", 11.0), _p("t1", "TE", 9.0),
        _p("r3", "RB", 8.0), _p("k1", "K", 8.0), _p("d1", "DEF", 7.0),
    ]


def test_lineup_gain_measures_optimal_improvement():
    base = _my_full_roster()
    # A 14.5 WR clearly cracks the lineup (bumps w2 11 -> flex, r3 8 -> bench).
    gain = lineup_gain(base, SLOTS, _p("cand", "WR", 14.5))
    assert gain > 2.0
    # A 5.0 WR cannot beat any starter -> no improvement.
    assert lineup_gain(base, SLOTS, _p("weak", "WR", 5.0)) == 0.0


def test_spend_advice_verdicts():
    base = _my_full_roster()
    selective = priority_scarcity(1, 12)
    aggressive = priority_scarcity(12, 12)

    clear = SpendCandidate(_p("clear", "WR", 14.5))
    marginal = SpendCandidate(_p("marg", "RB", 8.5))  # beats flex r3(8) by 0.5
    noupg = SpendCandidate(_p("none", "WR", 5.0))
    inherit = SpendCandidate(_p("heir", "RB", 0.0), is_new_starter=True)

    by_id = {a.player_id: a for a in spend_advice([clear, noupg, inherit], base, SLOTS, selective)}
    assert by_id["clear"].verdict == "spend"      # clear upgrade
    assert by_id["none"].verdict == "hold"        # no startable upgrade
    assert by_id["heir"].verdict == "spend"       # inherited a starting role

    # A marginal upgrade flips on posture: hold-the-claim when selective, use-it when aggressive.
    sel = spend_advice([marginal], base, SLOTS, selective)[0]
    agg = spend_advice([marginal], base, SLOTS, aggressive)[0]
    assert 0.0 < sel.lineup_gain < 2.0
    assert sel.verdict == "stream-later" and agg.verdict == "spend"

    # High contention forces a claim even under a selective posture.
    contested = SpendCandidate(_p("marg", "RB", 8.5), contention=1000)
    assert spend_advice([contested], base, SLOTS, selective)[0].verdict == "spend"


# --------------------------------------------------------------------------- playoff SOS
def test_points_allowed_and_multipliers_in_our_scoring():
    rows = [
        {"opponent_team": "DAL", "position": "RB", "week": 1, "pts": 20.0},
        {"opponent_team": "DAL", "position": "RB", "week": 2, "pts": 10.0},  # DAL: 15/gm to RB
        {"opponent_team": "PHI", "position": "RB", "week": 1, "pts": 5.0},
        {"opponent_team": "PHI", "position": "RB", "week": 2, "pts": 5.0},   # PHI: 5/gm to RB
    ]
    pa = points_allowed_by_position(rows, scoring={}, score=lambda r, s: r["pts"])
    assert pa["DAL"]["RB"] == 15.0 and pa["PHI"]["RB"] == 5.0
    sos = sos_multipliers(pa)  # league avg RB = 10 -> DAL 1.5 (soft), PHI 0.5 (tough)
    assert sos["DAL"]["RB"] == 1.5 and sos["PHI"]["RB"] == 0.5
    assert multiplier(sos, "UNKNOWN", "RB") == 1.0  # default no-tilt


def test_normalize_team_and_opponents_by_week():
    assert normalize_team("LA") == "LAR"
    sched = [
        {"game_type": "REG", "week": 15, "home_team": "LA", "away_team": "SF"},
        {"game_type": "REG", "week": 16, "home_team": "DAL", "away_team": "PHI"},
        {"game_type": "POST", "week": 15, "home_team": "KC", "away_team": "BUF"},  # ignored
    ]
    obw = opponents_by_week(sched, [15, 16])
    assert obw["LAR"][15] == "SF" and obw["SF"][15] == "LAR"
    assert obw["DAL"][16] == "PHI"
    assert "KC" not in obw  # postseason row excluded


# --------------------------------------------------------------------------- stash ranker
def test_rank_playoff_stashes_applies_sos_and_sorts():
    cands = [
        {"player_id": "a", "name": "A", "pos": "RB", "team": "NE", "baseline": 10.0},
        {"player_id": "b", "name": "B", "pos": "RB", "team": "NYJ", "baseline": 10.0},
    ]
    sos = {"DAL": {"RB": 1.5}, "PHI": {"RB": 0.5}}
    opp = {"NE": {15: "DAL", 16: "DAL", 17: "DAL"}, "NYJ": {15: "PHI", 16: "PHI", 17: "PHI"}}
    ranked = rank_playoff_stashes(cands, sos, opp)
    assert [s.player_id for s in ranked] == ["a", "b"]  # soft schedule first
    a = ranked[0]
    assert a.raw_value == 30.0 and a.adj_value == 45.0 and a.sos_swing == 15.0
    assert len(a.weeks) == 3 and a.weeks[0].opponent == "DAL"


def test_rank_playoff_stashes_skips_bye_week_and_filters_low_baseline():
    cands = [
        {"player_id": "a", "name": "A", "pos": "WR", "team": "NE", "baseline": 8.0},
        {"player_id": "z", "name": "Z", "pos": "WR", "team": "NE", "baseline": 0.0},  # dropped
    ]
    opp = {"NE": {15: "DAL", 17: "DAL"}}  # no week 16 -> bye/unknown
    ranked = rank_playoff_stashes(cands, sos={}, opponents_by_week=opp)
    assert [s.player_id for s in ranked] == ["a"]
    assert len(ranked[0].weeks) == 2  # only the two weeks with a game


def test_bye_stash_suggestions_groups_and_filters_past_weeks():
    starters = [_p("w1", "WR", 14.0, team="NE"), _p("r1", "RB", 15.0, team="DAL")]
    bye_week = {"NE": 14, "DAL": 9}  # NE bye in wk14 (upcoming from wk10), DAL bye already passed
    fas = [
        {"player_id": "x", "name": "FA WR1", "pos": "WR", "baseline": 9.0},
        {"player_id": "y", "name": "FA WR2", "pos": "WR", "baseline": 6.0},
    ]
    out = bye_stash_suggestions(starters, bye_week, fas, from_week=11, top=2)
    assert len(out) == 1
    b = out[0]
    assert b.week == 14 and b.pos == "WR" and b.idle_starters == ("w1",)
    assert [n for n, _ in b.suggestions] == ["FA WR1", "FA WR2"]


# --------------------------------------------------------------------------- usage signals
def test_usage_signals_join_and_shares():
    snaps = pl.DataFrame(
        {"season": [2025, 2025], "game_type": ["REG", "REG"], "week": [8, 9],
         "pfr_player_id": ["PFR1", "PFR1"], "offense_pct": [0.5, 0.7]}
    )
    opp = pl.DataFrame(
        {"season": [2025, 2025], "week": [8, 9], "player_id": ["G1", "G1"],
         "rec_attempt": [5.0, 7.0], "rec_attempt_team": [20.0, 30.0],
         "rush_attempt": [0.0, 0.0], "rush_attempt_team": [25.0, 25.0],
         "total_fantasy_points_exp": [10.0, 14.0]}
    )
    sig = usage_signals(
        ["100"], season=2025, week=10, lookback=4,
        snaps=snaps, opportunity=opp,
        pfr_to_sleeper={"PFR1": "100"}, gsis_to_sleeper={"G1": "100"},
    )["100"]
    assert sig.snap_pct == 0.6 and sig.snap_trend == 0.2
    assert sig.target_share == 0.24 and sig.exp_points == 12.0 and sig.weeks == 2

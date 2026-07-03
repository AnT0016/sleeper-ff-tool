"""Regression tests for the 2026-07 project audit fixes (all offline).

Covers the rollover fail-safes in the refresh pipeline, the HTTP cache-TTL contracts, the
waiver-order-aware spend posture, the bye-cover filter, the autopick-proof my-pick joins, the season
sim's degenerate-input guards, and the decomposed weekly-variance model.
"""

from __future__ import annotations

import numpy as np
import pytest
from requests_cache.policy.expiration import get_url_expiration

from analysis.backtest import simulate_draft
from analysis.snapshot import final_fantasy_week, offseason_skip_reason
from draft.roster import my_drafted
from seasonsim.distributions import sample_weekly_points, season_factor_cv
from sleeper.http import DO_NOT_CACHE, URL_TTLS
from waivers.league import priority_scarcity
from waivers.stash import bye_stash_suggestions


# --------------------------------------------------------------------------- refresh fail-safes
def test_final_fantasy_week_from_league_settings():
    assert final_fantasy_week({"settings": {"playoff_week_start": 15, "playoff_teams": 6}}) == 17
    assert final_fantasy_week({"settings": {"playoff_week_start": 14, "playoff_teams": 4}}) == 15
    assert final_fantasy_week(None) == 17  # CLAUDE.md defaults: 15 + 3 rounds - 1


def test_offseason_skip_explicit_args_always_run():
    state = {"season_type": "off", "season": "2026"}
    assert offseason_skip_reason(state, 15, 2025, {"season": "2025"}) is None


def test_offseason_skip_out_of_season():
    reason = offseason_skip_reason({"season_type": "off", "week": 0}, None, None)
    assert reason and "off-season" in reason


def test_rollover_failsafe_league_season_mismatch():
    state = {"season_type": "regular", "season": "2026", "week": 1}
    league = {"league_id": "L25", "season": "2025", "status": "complete"}
    reason = offseason_skip_reason(state, None, None, league)
    assert reason and "2025" in reason and "2026" in reason and "LEAGUE_ID" in reason


def test_rollover_failsafe_completed_league():
    state = {"season_type": "regular", "season": "2026", "week": 3}
    league = {"league_id": "L", "season": "2026", "status": "complete"}
    reason = offseason_skip_reason(state, None, None, league)
    assert reason and "complete" in reason


def test_week_clamp_after_championship():
    state = {"season_type": "regular", "season": "2026", "week": 18}
    league = {
        "league_id": "L", "season": "2026", "status": "in_season",
        "settings": {"playoff_week_start": 15, "playoff_teams": 6},
    }
    reason = offseason_skip_reason(state, None, None, league)
    assert reason and "championship" in reason


def test_normal_in_season_run_proceeds():
    state = {"season_type": "regular", "season": "2026", "week": 5}
    league = {
        "league_id": "L", "season": "2026", "status": "in_season",
        "settings": {"playoff_week_start": 15, "playoff_teams": 6},
    }
    assert offseason_skip_reason(state, None, None, league) is None


# --------------------------------------------------------------------------- HTTP cache contracts
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.sleeper.app/v1/draft/123/picks", DO_NOT_CACHE),  # live draft polling
        ("https://api.sleeper.app/v1/draft/123", DO_NOT_CACHE),
        ("https://api.sleeper.app/v1/league/99/drafts", DO_NOT_CACHE),  # draft-day discovery
        ("https://api.sleeper.app/v1/players/nfl/trending/add", 3600),  # hourly waiver signal
        ("https://api.sleeper.app/v1/players/nfl", 86400),  # ~5MB dump, once/day
        ("https://api.sleeper.app/v1/league/99", 3600),
        ("https://api.sleeper.com/projections/nfl/2026/1", 21600),
    ],
)
def test_url_ttl_contracts(url, expected):
    assert get_url_expiration(url, URL_TTLS) == expected


# --------------------------------------------------------------------------- waiver posture
def test_priority_scarcity_back_of_order_is_aggressive_even_for_contender():
    s = priority_scarcity(3, 12, waiver_position=12)  # rank 3 but already at the back
    assert s.posture == "aggressive" and "#12" in s.note


def test_priority_scarcity_front_of_order_contender_is_selective():
    s = priority_scarcity(2, 12, waiver_position=1)
    assert s.posture == "selective" and "#1" in s.note


def test_priority_scarcity_falls_back_to_standings_without_position():
    assert priority_scarcity(1, 12).posture == "selective"
    assert priority_scarcity(12, 12).posture == "aggressive"
    assert priority_scarcity(6, 12).posture == "balanced"


# --------------------------------------------------------------------------- bye-cover filter
class _P:
    def __init__(self, name, pos, team):
        self.name, self.pos, self.team = name, pos, team


def test_bye_stash_skips_covers_sharing_the_bye():
    starters = [_P("My WR1", "WR", "NE")]
    byes = {"NE": 14, "KC": 14, "DAL": 9}
    fas = [
        {"pos": "WR", "name": "SameBye", "baseline": 9.0, "team": "KC"},  # useless: also bye W14
        {"pos": "WR", "name": "RealCover", "baseline": 8.0, "team": "DAL"},
    ]
    out = bye_stash_suggestions(starters, byes, fas, from_week=10)
    assert len(out) == 1 and out[0].week == 14
    names = [n for n, _ in out[0].suggestions]
    assert "RealCover" in names and "SameBye" not in names


# --------------------------------------------------------------------------- autopick-proof joins
def test_my_drafted_catches_autopicks_via_slot():
    picks = [
        {"pick_no": 1, "picked_by": "me", "draft_slot": 4},
        {"pick_no": 21, "picked_by": "", "draft_slot": 4},  # CPU autopick: empty picked_by
        {"pick_no": 2, "picked_by": "rival", "draft_slot": 5},
    ]
    assert [p["pick_no"] for p in my_drafted(picks, "me")] == [1]
    assert [p["pick_no"] for p in my_drafted(picks, "me", my_slot=4)] == [1, 21]


def test_simulate_draft_substitutes_autopicked_slot_picks():
    picks = [
        {"pick_no": 1, "round": 1, "player_id": "a", "picked_by": "rival", "draft_slot": 1},
        {"pick_no": 2, "round": 1, "player_id": "b", "picked_by": "", "draft_slot": 2},  # my autopick
    ]
    order = {"a": 1.0, "b": 2.0, "c": 3.0}
    pos = {"a": "RB", "b": "RB", "c": "RB"}
    rows = simulate_draft(picks, "me", order, pos, my_slot=2)
    assert len(rows) == 1 and rows[0]["my_pid"] == "b" and rows[0]["tool_pid"] == "b"
    # without the slot, the autopick is misread as a rival's pick and my replay is empty
    assert simulate_draft(picks, "me", order, pos) == []


# --------------------------------------------------------------------------- season sim variance
def test_weekly_draws_keep_single_game_noise_but_season_cv():
    rng = np.random.default_rng(7)
    season_mean = np.array([170.0])
    season_cv = np.array([0.32])  # RB-like
    game_cv = np.array([0.60])
    pts = sample_weekly_points(rng, season_mean, season_cv, n_sims=6000, n_weeks=17, game_cv=game_cv)

    totals = pts.sum(axis=2)[:, 0]
    assert abs(totals.mean() - 170.0) / 170.0 < 0.05  # mean-preserving
    total_cv = totals.std() / totals.mean()
    assert 0.25 < total_cv < 0.40  # season spread kept

    one_week = pts[:, 0, 0]
    week_cv = one_week.std() / one_week.mean()
    # single-game noise stays realistic: ~sqrt((1+f²)(1+g²)-1), FAR below the old 0.32×√17 ≈ 1.32
    assert week_cv < 0.9


def test_season_factor_cv_floors_at_zero():
    out = season_factor_cv(np.array([0.05]), np.array([0.85]), 17)
    assert out[0] >= 0.0


# --------------------------------------------------------------------------- season sim guards
class _PreDraftSleeper:
    def get_league(self, league_id):
        return {"league_id": league_id, "status": "pre_draft", "season": "2026"}


def test_season_sim_rejects_pre_draft_league():
    from seasonsim.inputs import load_season_inputs

    with pytest.raises(ValueError, match="pre_draft"):
        load_season_inputs("TEST", sleeper=_PreDraftSleeper())


class _NoProjectionSleeper:
    """A drafted 2-team league whose season projections don't exist yet."""

    def get_league(self, league_id):
        return {
            "league_id": league_id, "status": "in_season", "season": "2026",
            "scoring_settings": {}, "total_rosters": 2,
            "roster_positions": ["QB", "RB", "BN"],
            "settings": {"playoff_teams": 2, "playoff_week_start": 15},
        }

    def get_players_nfl(self):
        return {"1": {"position": "QB"}, "2": {"position": "QB"}}

    def get_rosters(self, league_id):
        return [
            {"roster_id": 1, "owner_id": "u1", "players": ["1"]},
            {"roster_id": 2, "owner_id": "u2", "players": ["2"]},
        ]

    def get_users(self, league_id):
        return [{"user_id": "u1"}, {"user_id": "u2"}]

    def get_matchups(self, league_id, week):
        raise ConnectionError("no matchups")


def test_season_sim_rejects_projectionless_league(monkeypatch):
    import seasonsim.inputs as si

    monkeypatch.setattr(si, "build_board", lambda season, scoring: [])
    with pytest.raises(ValueError, match="projection"):
        si.load_season_inputs("TEST", sleeper=_NoProjectionSleeper())

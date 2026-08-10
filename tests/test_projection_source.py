"""Offline tests for the projection-source seam and the live swap gate (Phase 9, ticket #34).

The evidence discipline (Decision #9): **no beats-the-market number is asserted here** — that grade is
forward-only and lives in ``docs/model-swap-gate.md`` once 2026 weeks accumulate. These tests pin
**mechanics**, each guard written so reverting the guarded code turns the test red:

* the seam defaults to Sleeper per the recorded gate, and the **default path never touches the lake**;
* a model position **overlays** onto the Sleeper base, and a degrade / missing model row **falls back**
  to Sleeper — the whole next-week call when the market is unavailable;
* the ensemble composes **through** the seam without re-gating — a deferred value passes unchanged;
* the swap gate **fails closed** at 0 and < 4 weeks and needs **both** metrics (the 12-cell rule);
* the report's verdicts are **computed** from the gate state, never written beside it;
* ``KickDefModel.load_fitted(scoring=...)`` **re-prices** with the live scoring (the #30 carry-over).

No lake and no network: every frame/board is synthetic and every Sleeper is a fake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import projections.source as ps
import scripts.eval_swap_gate as eg
from projections.board import PlayerRow, build_board
from projections.source import (
    MODEL,
    SLEEPER,
    compose_season_board,
    default_source,
    recorded_swap_gate,
    resolve_positions_source,
    season_board,
    weekly_projections,
)


# =============================================================== fakes
class _FakeSleeper:
    """A Sleeper stand-in returning fixed weekly/season rows and a tiny players map."""

    def __init__(self, weekly_rows=None, season_rows=None, players=None) -> None:
        self._weekly = weekly_rows or []
        self._season = season_rows or []
        self._players = players or {}

    def get_projections(self, season, week):
        return self._weekly

    def get_season_projections(self, season, *, positions=None):
        return self._season

    def get_players_nfl(self):
        return self._players


def _wk_row(pid, pos, stats, *, team="BUF", name=None):
    return {"player_id": pid, "player": {"position": pos, "team": team, "full_name": name or pid},
            "stats": stats}


_SCORING = {"rec": 0.5, "rec_yd": 0.1, "rush_yd": 0.1, "pass_yd": 0.04}


@pytest.fixture(autouse=True)
def _clear_caches():
    """Every test starts from a clean gate cache and model cache."""
    recorded_swap_gate.cache_clear()
    ps.clear_model_cache()
    yield
    recorded_swap_gate.cache_clear()
    ps.clear_model_cache()


def _set_gate(monkeypatch, met: set[str]) -> None:
    """Force the recorded gate to a chosen met-set, deterministically (no committed artifact read)."""
    monkeypatch.setattr(ps, "recorded_swap_gate", lambda path=ps.DEFAULT_GATE_PATH: frozenset(met))


# =============================================================== gate → default source
def test_missing_gate_artifact_defaults_every_position_to_sleeper(tmp_path):
    """A gate artifact that isn't there → no position swapped to the model (safe fallback)."""
    assert recorded_swap_gate(tmp_path / "nope.json") == frozenset()
    assert default_source("QB", gate_path=tmp_path / "nope.json") == SLEEPER


def test_recorded_gate_reads_only_met_positions(tmp_path):
    import json

    path = tmp_path / "swap_gate.json"
    path.write_text(json.dumps({"positions": {
        "WR": {"met": True}, "QB": {"met": False}, "RB": {"met": True},
    }}), encoding="utf-8")
    assert recorded_swap_gate(path) == frozenset({"WR", "RB"})
    assert default_source("WR", gate_path=path) == MODEL
    assert default_source("QB", gate_path=path) == SLEEPER


def test_resolve_positions_source_explicit_forces_all_and_none_consults_gate(monkeypatch):
    _set_gate(monkeypatch, {"WR"})
    # None → per-position default from the gate.
    assert resolve_positions_source(("QB", "WR"), None) == {"QB": SLEEPER, "WR": MODEL}
    # explicit forces every position.
    assert resolve_positions_source(("QB", "WR"), MODEL) == {"QB": MODEL, "WR": MODEL}
    assert resolve_positions_source(("QB", "WR"), SLEEPER) == {"QB": SLEEPER, "WR": SLEEPER}


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown projection source"):
        resolve_positions_source(("QB",), "made_up")


# =============================================================== the weekly seam
def test_weekly_default_is_exactly_the_sleeper_scored_dict(monkeypatch):
    from optimizer.inputs import score_projections

    _set_gate(monkeypatch, set())  # nothing met → all Sleeper
    rows = [_wk_row("p1", "WR", {"rec": 4, "rec_yd": 50}, name="A")]
    sleeper = _FakeSleeper(weekly_rows=rows)
    out = weekly_projections(2026, 1, _SCORING, source=None, sleeper=sleeper)
    assert out == score_projections(rows, _SCORING)
    assert out == {"p1": {"proj": 7.0, "pos": "WR", "team": "BUF", "name": "A"}}


def test_weekly_default_never_touches_the_lake(monkeypatch):
    """The Sleeper default must not import/run the lake — the trap that makes a seam unusable."""
    import dataset.assemble as assemble

    def _boom(*a, **k):
        raise AssertionError("build_training_frame must not be reached on the Sleeper default path")

    monkeypatch.setattr(assemble, "build_training_frame", _boom)
    _set_gate(monkeypatch, set())
    rows = [_wk_row("p1", "WR", {"rec": 4, "rec_yd": 50})]
    out = weekly_projections(2026, 1, _SCORING, source=None, sleeper=_FakeSleeper(weekly_rows=rows))
    assert "p1" in out  # returned without ever building a frame


def test_weekly_model_overlays_and_falls_back_per_player(monkeypatch):
    """Model rows win for the positions/players they cover; everyone else keeps the Sleeper base."""
    rows = [_wk_row("p1", "WR", {"rec": 4, "rec_yd": 50}, name="A"),
            _wk_row("p2", "WR", {"rec": 2, "rec_yd": 20}, name="B")]
    sleeper = _FakeSleeper(weekly_rows=rows)

    def provider(season, week, scoring, positions, *, sleeper):
        return {"p1": {"proj": 99.0, "pos": "WR", "team": "BUF", "name": "A(model)"}}

    out = weekly_projections(2026, 1, _SCORING, source=MODEL, sleeper=sleeper, model_provider=provider)
    assert out["p1"]["proj"] == 99.0 and out["p1"]["name"] == "A(model)"  # model overlaid
    assert out["p2"]["proj"] == 3.0  # not in model rows → Sleeper base (a deferred/uncovered row)


def test_weekly_whole_call_degrades_to_sleeper(monkeypatch):
    """A provider that degrades (returns {}) leaves the entire call on the Sleeper base."""
    rows = [_wk_row("p1", "WR", {"rec": 4, "rec_yd": 50})]
    sleeper = _FakeSleeper(weekly_rows=rows)
    out = weekly_projections(
        2026, 2, _SCORING, source=MODEL, sleeper=sleeper,
        model_provider=lambda *a, **k: {},
    )
    assert out == weekly_projections(2026, 2, _SCORING, source=SLEEPER, sleeper=sleeper)


def test_market_unavailable_detects_all_null_market_columns():
    present = pd.DataFrame({"implied_team_total": [24.0, 22.0], "total_line": [45.0, 44.0]})
    absent = pd.DataFrame({"implied_team_total": [np.nan, np.nan], "total_line": [np.nan, np.nan]})
    assert ps._market_available(present) is True
    assert ps._market_available(absent) is False
    assert ps._market_available(pd.DataFrame({"x": [1]})) is False  # no market columns at all


# =============================================================== the season seam
def _season_fetch(rows):
    def fetch(season, *, positions):
        return [r for r in rows if r["player"]["position"] in positions]
    return fetch


def _srow(pid, pos, pts):
    # a Sleeper season row whose custom score is `pts` under _SCORING (rec_yd 0.1)
    return {"player_id": pid, "player": {"position": pos, "team": "BUF", "full_name": pid},
            "stats": {"rec_yd": pts * 10}}


def test_build_board_default_is_untouched_sleeper(monkeypatch):
    _set_gate(monkeypatch, set())
    rows = [_srow("w1", "WR", 30), _srow("r1", "RB", 20)]
    board = build_board(2026, _SCORING, positions=("WR", "RB"), fetch=_season_fetch(rows))
    assert [r.player_id for r in board] == ["w1", "r1"]  # ranked best-first, Sleeper-scored
    assert board[0].proj_pts == 30.0


def test_compose_season_board_merges_sleeper_and_model_positions_ranked():
    def sleeper_builder(season, scoring, *, positions, adp_key, fetch):
        return [PlayerRow("r1", "RB1", "RB", "BUF", 20.0, float("inf"))]

    def model_provider(season, scoring, positions):
        return [PlayerRow("w1", "WR1", "WR", "BUF", 35.0, float("inf"))]

    board = compose_season_board(
        2026, _SCORING, positions=("WR", "RB"),
        srcmap={"WR": MODEL, "RB": SLEEPER}, sleeper_builder=sleeper_builder, model_provider=model_provider,
    )
    assert [r.player_id for r in board] == ["w1", "r1"]  # 35 (model WR) ranked above 20 (Sleeper RB)


def test_season_board_front_door_dispatches_by_source(monkeypatch):
    """``season_board`` is the direct season entry: Sleeper builds via ``build_board``, model via provider."""
    _set_gate(monkeypatch, set())
    rows = [_srow("s1", "RB", 12)]
    sleeper_only = season_board(2026, _SCORING, positions=("RB",), source=SLEEPER, fetch=_season_fetch(rows))
    assert [r.player_id for r in sleeper_only] == ["s1"]

    model_only = season_board(
        2026, _SCORING, positions=("WR",), source=MODEL,
        model_provider=lambda season, scoring, positions: [PlayerRow("mw", "MW", "WR", "BUF", 40.0, float("inf"))],
    )
    assert [r.player_id for r in model_only] == ["mw"]


def test_build_board_model_source_uses_the_model_provider(monkeypatch):
    """``build_board(source='model')`` composes via the seam's model provider (unflattened)."""
    monkeypatch.setattr(
        ps, "_season_model_board",
        lambda season, scoring, positions: [PlayerRow("m1", "M", "WR", "BUF", 50.0, float("inf"))],
    )
    rows = [_srow("s1", "RB", 10)]
    board = build_board(2026, _SCORING, positions=("WR", "RB"), fetch=_season_fetch(rows), source=MODEL)
    ids = [r.player_id for r in board]
    assert "m1" in ids  # the model WR row is present; RB path unaffected


# =============================================================== the swap gate (pure)
def _cell(weeks, mmae, kmae, mrho, krho):
    return {"weeks": weeks, "n": weeks * 5, "model_mae": mmae, "market_mae": kmae,
            "model_rho": mrho, "market_rho": krho}


def test_gate_needs_four_weeks_and_both_metrics():
    win = _cell(4, 3.0, 4.0, 0.60, 0.50)
    assert eg._met(win, min_weeks=4)                                   # both metrics, 4 weeks → MET
    assert not eg._met({**win, "weeks": 3}, min_weeks=4)               # < 4 weeks → fails closed
    assert not eg._met({**win, "model_mae": 4.0}, min_weeks=4)         # MAE tie → not met
    assert not eg._met({**win, "model_mae": 5.0}, min_weeks=4)         # MAE loss → not met
    assert not eg._met({**win, "model_rho": 0.50}, min_weeks=4)        # rho tie → not met
    assert not eg._met({**win, "model_rho": 0.40}, min_weeks=4)        # rho loss → not met


def test_gate_fails_closed_on_zero_weeks():
    empty = _cell(0, None, None, None, None)
    assert not eg._met(empty, min_weeks=4)
    state = eg.swap_gate_state(pd.DataFrame(columns=["position", "season", "week", "actual", "market", "model"]))
    assert set(state) == set(eg.POSITIONS)
    assert all(not c["met"] and c["weeks"] == 0 for c in state.values())  # every position NOT MET, explicit


def _board_rows(pos, weeks, model, market, actual, season=2026):
    rows = []
    for w in range(1, weeks + 1):
        for i in range(len(actual)):
            rows.append({"position": pos, "season": season, "week": w,
                         "actual": actual[i], "model": model[i], "market": market[i]})
    return rows


def test_swap_gate_state_marks_a_winning_position_met():
    actual = [10.0, 20.0, 30.0, 40.0, 50.0]
    model = list(actual)                         # MAE 0, perfect order (rho 1.0)
    market = [19.0, 12.0, 31.0, 42.0, 48.0]      # MAE 4.4, first two swapped (rho 0.9)
    board = pd.DataFrame(_board_rows("WR", 4, model, market, actual))
    state = eg.swap_gate_state(board)
    assert state["WR"]["met"] is True and state["WR"]["weeks"] == 4
    assert state["WR"]["model_mae"] < state["WR"]["market_mae"]
    assert state["WR"]["model_rho"] > state["WR"]["market_rho"]
    assert state["QB"]["met"] is False  # a position with no rows stays NOT MET


def test_build_scoreboard_drops_rows_missing_any_of_actual_market_model():
    frame = pd.DataFrame({
        "player_id": ["a", "b", "c"], "position": ["WR", "WR", "WR"],
        "season": [2026, 2026, 2026], "week": [1, 1, 1],
        "y_custom_points": [10.0, 20.0, np.nan],          # c has no actual → dropped
        "baseline_sleeper_points": [9.0, np.nan, 15.0],   # b has no market → dropped
    })
    preds = pd.Series([11.0, 19.0, 14.0], index=frame.index)
    board = eg.build_scoreboard(frame, predict_fn=lambda f: preds.reindex(f.index))
    assert list(board["position"]) == ["WR"]  # only row a survives (all three present)


def test_report_verdicts_are_derived_from_the_table():
    actual = [10.0, 20.0, 30.0, 40.0, 50.0]
    board = pd.DataFrame(
        _board_rows("WR", 4, actual, [19.0, 12.0, 31.0, 42.0, 48.0], actual)
    )
    state = eg.swap_gate_state(board)
    report = eg.render_report(
        state, season=2026, min_weeks=4, scoring_keys=42, partitions=0,
        league_name="Test", generated="2026-09-01", scoreboard_rows=int(len(board)),
    )
    assert "1 of 6 position(s) have met" in report      # count computed, not written
    assert "| WR | 4 |" in report and "✅ MET" in report  # the met row and its verdict
    assert "**QB** — NOT MET" in report                 # a not-met verdict, table-derived


# =============================================================== the #30 carry-over: load_fitted(scoring=)
def _kicker_frame(n=40):
    """A tiny K-only component frame: constant makes so a fitted head predicts ~1 fgm_50p per game."""
    from model.kickdef import KICKDEF_FEATURES

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {f: float(rng.uniform(-1, 30)) for f in KICKDEF_FEATURES}
        row.update({"player_id": f"K{i%5}", "season": 2020 + i % 4, "week": 1 + i % 5,
                    "position": "K", "games_played_prior": 3.0, "is_indoor": bool(i % 2),
                    "comp_fgm_0_19": 0.0, "comp_fgm_20_29": 0.0, "comp_fgm_30_39": 1.0,
                    "comp_fgm_40_49": 0.0, "comp_fgm_50p": 1.0, "comp_xpm": 2.0,
                    "comp_fgmiss": 0.0, "comp_xpmiss": 0.0})
        # y is the engine over this line; not asserted, just present for the prior's fit
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["is_indoor"] = frame["is_indoor"].astype("boolean")
    frame["y_custom_points"] = 3.0 * 1 + 5.0 * 1 + 1.0 * 2  # fgm_30_39*3 + fgm_50p*5 + xpm*1 (base scoring)
    return frame


_K_SCORING = {"fgm_0_19": 3, "fgm_20_29": 3, "fgm_30_39": 3, "fgm_40_49": 4, "fgm_50p": 5,
              "xpm": 1, "fgmiss": -1, "xpmiss": -1}


def test_kickdef_load_fitted_reprices_with_live_scoring(tmp_path, caplog):
    """A live scoring override re-prices K/DST through the engine — the seam #34 makes reachable."""
    from model.kickdef import KickDefModel

    frame = _kicker_frame()
    k_rows = frame[frame["position"] == "K"]
    path = tmp_path / "kd.json"
    KickDefModel(_K_SCORING, defer=()).fit(frame).save(path)  # pure-component: predict needs no prior

    base = KickDefModel.load_fitted(path)                     # recorded scoring
    base_mean = float(base.predict(k_rows).mean())

    bumped = {**_K_SCORING, "fgm_50p": 99.0}                  # a 50+ FG now worth 99, not 5
    with caplog.at_level("WARNING"):
        reprice = KickDefModel.load_fitted(path, scoring=bumped)
    reprice_mean = float(reprice.predict(k_rows).mean())

    assert reprice_mean > base_mean + 50.0                    # ~+94 per predicted 50+ make
    assert any("re-priced" in r.getMessage() for r in caplog.records)  # the diff is logged, not silent


def test_kickdef_load_fitted_collapses_scoring_to_one_shared_dict(tmp_path):
    """The model and both heads share one scoring dict after a load — no stale third copy."""
    from model.kickdef import KickDefModel

    frame = _kicker_frame()
    path = tmp_path / "kd.json"
    KickDefModel(_K_SCORING, defer=()).fit(frame).save(path)
    model = KickDefModel.load_fitted(path, scoring={**_K_SCORING, "fgm_50p": 42.0})
    assert model.scoring is model._kicker.scoring  # one object, not three copies
    assert model.scoring["fgm_50p"] == 42.0

"""Offline tests for the breakout / waiver classifier (Phase 9, ticket #33).

The property that matters most, and is the heart of this ticket, is the **lookahead asymmetry**, pinned
by a two-halved test because a reviewer cannot see it by inspection:

* **The label really is forward** (:func:`test_label_is_forward_not_contemporaneous`): spiking a
  player's *forward-window* outcomes flips his label, while spiking his *own decision-week* points does
  **not**. The mutant this catches is a label that accidentally reads week ``w`` — it would flip on the
  own-week spike and this test would fail.
* **The features never reach forward** (:func:`test_prediction_ignores_future_features`): mutating the
  feature values of the forward-week rows leaves the decision-week prediction bit-identical. The mutant
  this catches is a feature built from week ``w+1`` — the decision-week score would move and this test
  would fail. A test with only this half passes trivially on a model that ignores the future entirely,
  so both halves are here.

Everything else pins a mechanic: the startable thresholds, the per-played-game forward mean (a bye is
not a zero), the ``>= 2``-games evaluability drop, the two cohorts, the precision@k / lift math, the
deferral gate (a deferred position ranks by its baseline **exactly**), the artifact round-trip, safe by
default, and that the report's prose follows its own tables. No lake and no network: every frame is
built to make the property under test decidable.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.breakout import (
    BREAKOUT_FEATURES,
    BREAKOUT_LABEL_COL,
    BREAKOUT_POSITIONS,
    K_VALUES,
    _fit_logistic,
    BreakoutEvalResult,
    BreakoutModel,
    BreakoutPositionMetrics,
    ColumnRanker,
    add_forward_label,
    breakout_gate,
    evaluate_breakout,
    lift,
    precision_at_k,
    production_cohort_mask,
    recorded_gate,
    snap_cohort_mask,
    startable_thresholds,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli(name: str):
    """Import ``scripts/<name>.py`` as a module (the CLIs are not on the import path)."""
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


eb = _load_cli("eval_breakout")


# --------------------------------------------------------------------------- synthetic frame
def _frame(rows: list[dict]) -> pd.DataFrame:
    """A build_training_frame-shaped frame from per-week specs; every feature defaults to NaN."""
    out = pd.DataFrame(rows)
    defaults = {
        "position": "RB",
        "team": "AAA",
        "is_dst": False,
        "inj_report_status": pd.NA,
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val
    for col in (*BREAKOUT_FEATURES, "snap_pct_ewma", "points_ewma"):
        if col not in out.columns:
            out[col] = np.nan
    return out


def _player_weeks(pid, season, position, weeks_points, **features):
    """One row per (week, points) for a player; feature kwargs are broadcast to every week."""
    rows = []
    for wk, pts in weeks_points:
        row = {"player_id": pid, "season": season, "week": wk, "position": position,
               "y_custom_points": pts}
        row.update(features)
        rows.append(row)
    return rows


# =========================================================================== thresholds
def test_startable_thresholds_is_the_spos_th_weekly_rank_over_warmup():
    # One warm-up slate (2016 wk1), 5 RBs scoring 20,16,12,8,4. S_pos=3 -> the 3rd-highest = 12.
    rows = _player_weeks("p", 2016, "RB", [(1, 0)])  # placeholder, replaced below
    rows = []
    for i, pts in enumerate([20.0, 16.0, 12.0, 8.0, 4.0]):
        rows += _player_weeks(f"rb{i}", 2016, "RB", [(1, pts)])
    frame = _frame(rows)
    t = startable_thresholds(frame, warmup_seasons=(2016,), startable_rank={"RB": 3, "WR": 3, "TE": 3})
    assert t["RB"] == pytest.approx(12.0)
    # A season outside the warm-up must not move it.
    frame2 = _frame(rows + _player_weeks("late", 2019, "RB", [(1, 999.0)]))
    t2 = startable_thresholds(frame2, warmup_seasons=(2016,), startable_rank={"RB": 3, "WR": 3, "TE": 3})
    assert t2["RB"] == pytest.approx(12.0)


# =========================================================================== the asymmetry (half 1)
def test_label_is_forward_not_contemporaneous():
    """Spiking the forward window flips the label; spiking the decision week's own points does not."""
    thresholds = {"RB": 10.0, "WR": 10.0, "TE": 10.0}
    base = _player_weeks("rb", 2020, "RB", [(1, 2.0), (2, 3.0), (3, 3.0), (4, 3.0)])
    frame = _frame(base)
    labelled = add_forward_label(frame, thresholds)
    row1 = labelled[labelled["week"] == 1].iloc[0]
    assert row1["y_breakout"] == 0.0  # forward weeks 2-4 average 3 < 10

    # (a) Spike the DECISION week's own points -> label must NOT change (a contemporaneous label would).
    own = frame.copy()
    own.loc[own["week"] == 1, "y_custom_points"] = 99.0
    lab_own = add_forward_label(own, thresholds)
    assert lab_own[lab_own["week"] == 1].iloc[0]["y_breakout"] == 0.0

    # (b) Spike the FORWARD window -> label must flip to 1 (proves the label reaches forward).
    fut = frame.copy()
    fut.loc[fut["week"].isin([2, 3, 4]), "y_custom_points"] = 30.0
    lab_fut = add_forward_label(fut, thresholds)
    assert lab_fut[lab_fut["week"] == 1].iloc[0]["y_breakout"] == 1.0


def test_forward_window_never_crosses_a_season_boundary():
    thresholds = {"RB": 10.0, "WR": 10.0, "TE": 10.0}
    rows = _player_weeks("rb", 2020, "RB", [(16, 2.0), (17, 2.0)])
    rows += _player_weeks("rb", 2021, "RB", [(1, 40.0), (2, 40.0), (3, 40.0)])
    labelled = add_forward_label(_frame(rows), thresholds)
    # 2020 wk16's forward window is 17,18,19 in 2020 — it must not pull 2021's huge weeks.
    r = labelled[(labelled["season"] == 2020) & (labelled["week"] == 16)].iloc[0]
    assert r["forward_games"] == 1  # only 2020 wk17 played -> not evaluable
    assert pd.isna(r["y_breakout"])


# =========================================================================== the asymmetry (half 2)
def _labelled_frame() -> pd.DataFrame:
    """A labelled frame (weeks 1-8 per player) with a feature carrying signal, both classes present.

    W_last = 8, so weeks 1-5 carry a full 3-week forward window and are decision rows; weeks 6-8 are
    forward-window-only. The forward outcomes at weeks 2-8 set each decision row's label.
    """
    rows = []
    for i in range(24):
        season = 2018 + (i % 2)
        snap = 0.1 + 0.012 * i  # stays under the 0.5 cohort line
        fwd = 20.0 if i % 2 == 0 else 2.0
        rows += _player_weeks(
            f"rb{i}", season, "RB",
            [(w, (5.0 if w == 1 else fwd)) for w in range(1, 9)],
            snap_pct_ewma=snap, snap_pct_last=snap, points_ewma=5.0, points_last=5.0,
            rush_share_last=0.1 + 0.01 * i,
        )
    return add_forward_label(_frame(rows), {"RB": 10.0, "WR": 10.0, "TE": 10.0})


def _training_cohort() -> pd.DataFrame:
    """The evaluable cohort of :func:`_labelled_frame` — what the model is fit and scored on."""
    labelled = _labelled_frame()
    return labelled[snap_cohort_mask(labelled) & labelled["is_evaluable"].astype(bool)].copy()


def test_prediction_ignores_future_features():
    """A model fit and asked to score week w is invariant to the feature values of weeks w+1..w+N."""
    labelled = _labelled_frame()
    model = BreakoutModel(defer={}).fit(_training_cohort())

    decision = labelled[labelled["week"] == 1]  # decision rows, with forward rows present in `labelled`
    before = model.predict(decision)

    # Mutate the FORWARD rows' features to extremes, then re-predict the SAME decision rows. A feature
    # that reached into w+1..w+N (the leak this catches) would move `after`; a strictly per-row one cannot.
    poisoned = labelled.copy()
    fwd = poisoned["week"] > 1
    for col in ("snap_pct_ewma", "snap_pct_last", "points_ewma", "points_last", "rush_share_last"):
        poisoned.loc[fwd, col] = 999.0
    after = model.predict(poisoned[poisoned["week"] == 1])

    assert np.allclose(before.to_numpy(), after.to_numpy(), equal_nan=True)


# =========================================================================== per-played-game / bye
def test_bye_in_window_is_not_a_zero():
    thresholds = {"RB": 10.0, "WR": 10.0, "TE": 10.0}
    # Week 3 is a bye (no row). Forward window of wk1 = 2,3,4; played = wk2, wk4.
    rows = _player_weeks("rb", 2020, "RB", [(1, 1.0), (2, 12.0), (4, 14.0), (5, 1.0)])
    labelled = add_forward_label(_frame(rows), thresholds)
    r = labelled[labelled["week"] == 1].iloc[0]
    assert r["forward_games"] == 2
    assert r["forward_ppg"] == pytest.approx(13.0)  # mean(12, 14), NOT mean(12, 0, 14)
    assert r["y_breakout"] == 1.0


def test_fewer_than_two_forward_games_is_not_evaluable():
    thresholds = {"RB": 10.0, "WR": 10.0, "TE": 10.0}
    rows = _player_weeks("rb", 2020, "RB", [(1, 1.0), (2, 40.0)])  # only one forward game
    labelled = add_forward_label(_frame(rows), thresholds)
    r = labelled[labelled["week"] == 1].iloc[0]
    assert r["forward_games"] == 1
    assert not bool(r["is_evaluable"])
    assert pd.isna(r["y_breakout"])


def test_full_window_cap_drops_the_ragged_season_edge():
    """A near-end week with >= 2 forward games is still not a decision row without a full N-week window."""
    thresholds = {"RB": 10.0, "WR": 10.0, "TE": 10.0}
    # W_last = 4, so week 2's window would be weeks 3,4,5 — week 5 does not exist. Even with 2 forward
    # games played (weeks 3,4), the row lacks a full 3-week window and must not be labelled. Removing the
    # cap would make it evaluable (2 games), so this pins the cap.
    rows = _player_weeks("rb", 2020, "RB", [(1, 1.0), (2, 1.0), (3, 20.0), (4, 20.0)])
    labelled = add_forward_label(_frame(rows), thresholds)
    r2 = labelled[labelled["week"] == 2].iloc[0]
    assert r2["forward_games"] == 2  # it *has* two forward games …
    assert not bool(r2["has_forward_window"])  # … but no full window, so it is not a decision row
    assert not bool(r2["is_evaluable"])
    assert pd.isna(r2["y_breakout"])
    # Week 1 *does* have a full window (2,3,4) and is labelled.
    assert bool(labelled[labelled["week"] == 1].iloc[0]["has_forward_window"])


# =========================================================================== cohorts
def test_snap_cohort_keeps_sub_starter_and_null_arm_only():
    frame = _frame(
        _player_weeks("a", 2020, "RB", [(1, 5.0)], snap_pct_ewma=0.30)
        + _player_weeks("b", 2020, "RB", [(1, 5.0)], snap_pct_ewma=0.70)  # established starter -> out
        + _player_weeks("c", 2020, "RB", [(1, 5.0)])  # snap_pct_ewma null -> the "no role yet" arm, in
        + _player_weeks("q", 2020, "QB", [(1, 5.0)], snap_pct_ewma=0.10)  # wrong position -> out
    )
    mask = snap_cohort_mask(frame)
    assert list(mask) == [True, False, True, False]


def test_production_cohort_is_a_different_axis():
    thresholds = {"RB": 10.0, "WR": 10.0, "TE": 10.0}
    frame = _frame(
        _player_weeks("a", 2020, "RB", [(1, 5.0)], points_ewma=6.0)  # below startable line -> in
        + _player_weeks("b", 2020, "RB", [(1, 5.0)], points_ewma=15.0)  # producing like a starter -> out
    )
    mask = production_cohort_mask(frame, thresholds)
    assert list(mask) == [True, False]


# =========================================================================== precision@k / lift
def test_precision_at_k_and_lift_math():
    scores = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = pd.Series([1, 0, 1, 0, 0])
    assert precision_at_k(scores, labels, 1) == pytest.approx(1.0)
    assert precision_at_k(scores, labels, 3) == pytest.approx(2 / 3)
    assert precision_at_k(scores, labels, 5) == pytest.approx(0.4)
    # A slate shallower than k does not admit the question.
    assert precision_at_k(scores.head(2), labels.head(2), 3) is None
    # Lift is precision over the base rate; undefined at base rate 0.
    assert lift(0.5, 0.25) == pytest.approx(2.0)
    assert lift(0.5, 0.0) is None


def test_precision_ignores_nan_scores_and_still_needs_k_valid():
    scores = pd.Series([0.9, np.nan, np.nan])
    labels = pd.Series([1, 0, 1])
    assert precision_at_k(scores, labels, 1) == pytest.approx(1.0)
    assert precision_at_k(scores, labels, 2) is None  # only one finite score


# =========================================================================== deferral gate
def test_deferred_position_ranks_by_its_baseline_exactly():
    labelled = _training_cohort()
    # Force WR-style deferral on RB: a gate mapping RB -> the last-week-points baseline.
    model = BreakoutModel(defer={"RB": "last_week_points"}).fit(labelled)
    scored = model.predict(labelled)
    expected = pd.to_numeric(labelled["points_last"], errors="coerce")
    assert np.allclose(scored.to_numpy(), expected.to_numpy(), equal_nan=True)
    # And a deferred position is never fit.
    assert "RB" not in model._fits


def test_breakout_gate_fields_a_winner_defers_a_loser():
    def _pos(p1):
        return BreakoutPositionMetrics(
            position="RB", n=10, base_rate=0.3, n_slates=5,
            precision={1: p1, 3: p1, 5: p1}, slates_at_k={1: 5, 3: 5, 5: 5},
            lift={1: None, 3: None, 5: None},
        )

    model = {"RB": _pos(0.5), "WR": _pos(0.2)}
    baselines = {"last_week_points": {"RB": _pos(0.4), "WR": _pos(0.3)}}
    gate = breakout_gate(model, baselines, positions=("RB", "WR"))
    assert "RB" not in gate  # 0.5 > 0.4 at every k -> fielded
    assert gate["WR"] == "last_week_points"  # 0.2 < 0.3 at every k -> deferred


def test_gate_requires_a_win_at_every_k_not_only_k1():
    """A k=1 win that reverses at k=3 must defer — precision@1 on ~117 slates is one noisy proportion."""
    def _mk(pos, p1, p3, p5):
        return BreakoutPositionMetrics(
            position=pos, n=10, base_rate=0.08, n_slates=117,
            precision={1: p1, 3: p3, 5: p5}, slates_at_k={1: 117, 3: 117, 5: 117},
            lift={1: None, 3: None, 5: None},
        )

    # WR: wins at k=1 (0.16 > 0.13) but LOSES at k=3 (0.10 < 0.11) — the real WR shape.
    model = {"WR": _mk("WR", 0.16, 0.10, 0.09), "RB": _mk("RB", 0.70, 0.64, 0.60)}
    base = {"last_week_points": {"WR": _mk("WR", 0.13, 0.11, 0.09), "RB": _mk("RB", 0.65, 0.55, 0.49)}}

    gate = breakout_gate(model, base, positions=("RB", "WR"))
    assert "RB" not in gate  # wins at every k -> fielded
    assert gate["WR"] == "last_week_points"  # loses at k=3 -> deferred, despite the k=1 win

    # Revert-check the rule: the mutant (a k=1-only gate) would FIELD WR — the opposite verdict, which
    # this test's assertion above catches. Make the mutant's disagreement explicit.
    k1_only_would_field_wr = model["WR"].precision[1] > base["last_week_points"]["WR"].precision[1]
    assert k1_only_would_field_wr  # so the all-k gate deferring WR is a strictly different decision to the winner


# =========================================================================== artifact round-trip
def test_load_fitted_round_trip_is_bit_identical(tmp_path):
    labelled = _training_cohort()
    gate = {"WR": "last_week_points", "TE": "role_share_last"}
    model = BreakoutModel(defer=gate).fit(labelled)
    path = tmp_path / "breakout.json"
    path.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    loaded = BreakoutModel.load_fitted(path)
    assert loaded.gate == gate
    before = model.predict(labelled).to_numpy()
    after = loaded.predict(labelled).to_numpy()
    assert np.allclose(before, after, equal_nan=True)


# =========================================================================== safe by default
def test_recorded_gate_missing_artifact_defers_every_position(tmp_path):
    gate = recorded_gate(tmp_path / "does-not-exist.json")
    assert set(gate) == set(BREAKOUT_POSITIONS)
    assert all(v == "last_week_points" for v in gate.values())


def test_missing_artifact_model_fields_nothing_and_ranks_by_baseline(tmp_path):
    labelled = _training_cohort()
    gate = recorded_gate(tmp_path / "nope.json")  # all deferred
    model = BreakoutModel(defer=gate).fit(labelled)
    assert model._fits == {}  # nothing fielded -> nothing fit (never an unproven logistic)
    scored = model.predict(labelled)
    expected = pd.to_numeric(labelled["points_last"], errors="coerce")
    assert np.allclose(scored.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_diagnostic_pure_logistic_fields_every_position():
    labelled = _training_cohort()
    model = BreakoutModel(defer={}).fit(labelled)
    assert model.fielded_positions == BREAKOUT_POSITIONS
    assert "RB" in model._fits


# =========================================================================== walk-forward leak gate
def test_evaluate_breakout_is_walk_forward():
    labelled = _training_cohort()  # seasons 2018 and 2019
    res = evaluate_breakout(ColumnRanker("last_week_points"), labelled, test_seasons=(2019,))
    # Only the requested test season is scored; nothing from 2018 appears as a scored row.
    assert set(res.predictions["season"].unique()) <= {2019}
    assert res.test_seasons == (2019,)


# =========================================================================== report follows tables
def _metrics(pos, base_rate, p1):
    return _mixed(pos, base_rate, p1, p1, p1)


def _mixed(pos, base_rate, p1, p3, p5):
    prec = {1: p1, 3: p3, 5: p5}
    return BreakoutPositionMetrics(
        position=pos, n=100, base_rate=base_rate, n_slates=117,
        precision=prec, slates_at_k={1: 117, 3: 117, 5: 117},
        lift={k: (prec[k] / base_rate if base_rate else None) for k in K_VALUES},
    )


def _mc(discordant, model_plus, base_plus):
    z = (model_plus - base_plus) / (discordant ** 0.5) if discordant else 0.0
    return {"discordant": discordant, "model_plus": model_plus, "base_plus": base_plus,
            "z": z, "significant": abs(z) > 1.96}


def test_render_report_prose_follows_its_tables():
    # RB wins at every k (fielded); WR wins k=1 (0.16) but loses k=3 (0.10<0.11) -> deferred; TE fielded.
    model_pp = {"RB": _metrics("RB", 0.3, 0.5), "WR": _mixed("WR", 0.08, 0.16, 0.10, 0.09),
                "TE": _metrics("TE", 0.1, 0.15)}
    base_pp = {"last_week_points": {"RB": _metrics("RB", 0.3, 0.4),
                                    "WR": _mixed("WR", 0.08, 0.13, 0.11, 0.09),
                                    "TE": _metrics("TE", 0.1, 0.12)}}
    empty = pd.DataFrame(columns=["player_id", "season", "week", "position", "score", "label"])
    results = {
        "BreakoutModel": BreakoutEvalResult("BreakoutModel", (2019,), model_pp, empty),
        "Logistic (pure)": BreakoutEvalResult("Logistic (pure)", (2019,), model_pp, empty),
    }
    gate = breakout_gate(model_pp, base_pp)
    assert "WR" in gate and "RB" not in gate and "TE" not in gate
    mcnemar = {  # RB k=1 insignificant (rests on k=3/5); WR k=1 insignificant (defers); TE significant
        "RB": {"last_week_points": _mc(47, 28, 19)},
        "WR": {"last_week_points": _mc(12, 7, 5)},
        "TE": {"last_week_points": _mc(33, 23, 10)},
    }

    report = eb.render_report(
        thresholds={"RB": 8.19, "WR": 10.44, "TE": 8.09},
        cohort={
            "n_decision": 41106, "n_evaluable": 33234, "n_cohort": 17060,
            "per_position": {
                p: {"n_cohort": 100, "base_rate_cohort": model_pp[p].base_rate,
                    "base_rate_evaluable": 0.4, "null_snap_share": 0.10,
                    "candidates_per_slate": 45.0, "slates_ge_k": {1: 10, 3: 10, 5: 10}, "n_slates": 10}
                for p in BREAKOUT_POSITIONS
            },
        },
        drop={"n_decision": 41106, "n_dropped": 7872, "drop_share": 0.192,
              "n_dropped_injured": 3000, "n_dropped_not_injured": 4872},
        drift={"RB": {2021: 0.34, 2024: 0.21}, "WR": {2021: 0.08, 2024: 0.08},
               "TE": {2021: 0.10, 2024: 0.09}},
        results=results,
        baseline_metrics=base_pp,
        mcnemar=mcnemar,
        gate=gate,
        robust={
            "agree": True,
            "note": "Both cohorts reach the same win/defer verdict at every position.",
            "n_cohort": 15000,
            "per_position": model_pp,
            "baseline_metrics": base_pp,
        },
        importances={"RB": [("points_ewma", 0.5)]},
        seasons=[2016, 2025], scored_seasons=[2019], scoring_keys=42, partitions=10,
        league_name="Test", generated="2026-08-10", frame_rows=169685, players=4603,
    )
    # The base-rate divergence sentence carries both numbers.
    assert "0.300" in report and "0.080" in report
    # The deferred WR must be named as deferring to its baseline (prose follows the gate).
    assert "WR→last week's points" in report
    # The deferral reason is generated from the table: WR loses to last week's points at k=3.
    assert "loses to last week's points at k=3" in report
    # RB's insignificant k=1 margin -> verdict stated to rest on k=3/5.
    assert "rests on **k=3/5**" in report
    # The RB drift peak->trough is stated with its seasons.
    assert "0.340" in report and "0.210" in report


# =========================================================================== solver convergence
def test_logistic_warns_only_when_it_fails_to_converge(caplog):
    """A non-converged fit ships weights indistinguishable from a converged one — make it audible.

    Reverting the ``converged`` flag (so ``max_iter`` exhaustion passes silently) drops the warning and
    turns this red. The normal path must stay silent, because the report counts WARNING records and a
    guard that cries wolf would make "zero warnings on real data" meaningless.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, len(BREAKOUT_FEATURES)))
    y = (x[:, 0] + rng.normal(0, 0.5, size=400) > 0).astype("float64")

    with caplog.at_level(logging.WARNING, logger="model.breakout"):
        _fit_logistic(x, y, 1.0)
    assert caplog.records == []  # converges well inside max_iter -> silent

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="model.breakout"):
        _fit_logistic(x, y, 1.0, max_iter=1)  # one Newton step cannot reach tol
    assert len(caplog.records) == 1
    assert "did not converge" in caplog.records[0].getMessage()


def test_breakout_label_column_is_not_the_regression_label():
    """This model's label is its own column; ``y_custom_points`` is its *input*, never its target."""
    from model.evaluate import LABEL_COL as REGRESSION_LABEL_COL

    assert BREAKOUT_LABEL_COL == "y_breakout"
    assert BREAKOUT_LABEL_COL != REGRESSION_LABEL_COL

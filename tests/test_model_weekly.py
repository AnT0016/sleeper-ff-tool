"""Offline tests for the weekly point model (Phase 9, ticket #29).

The evidence discipline this ticket is held to: **the beats-the-baseline claim is not made here.** A
synthetic frame can be built to make any learner win, and #31 was caught doing exactly that — its
fixture guaranteed a win it then lost at two positions on the real lake. So the only bar-clearing
number lives in ``docs/model-weekly.md``, measured on the real lake by ``scripts/eval_weekly.py``.
These tests pin **mechanics** instead:

* the points-head **linearity guard** is fail-closed (Decision #7): it raises on a bonus/threshold
  skill key and on an unrecognised one, and stays quiet on a linear league and on K/DST buckets;
* the model reads **only pre-lock features** — spiking a row's own-week label does not move its
  prediction, and ``WEEKLY_FEATURES`` carries no same-week outcome;
* **cold start** (week 1, all lags null) produces a finite prediction that moves with the Vegas market;
* the fitted artifact is **reproducible** (fit twice → identical) and round-trips through JSON;
* and ``scripts/eval_weekly.py``'s report has its prose *generated from* the numbers it cites.

The one "beats a baseline" test here is explicitly a **mechanism** check: on a frame where the signal
lives in the market and not in recency, the model — which sees the market — must out-rank
``TrailingMean``, which cannot. That proves the model can exploit a signal the baseline is blind to; it
says nothing about the real lake, which is the point.

No lake and no network: every frame is synthetic, built to make the property under test decidable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.baselines import PriorSeasonRank, TrailingMean
from model.evaluate import EvalResult, PositionMetrics, Predictor, evaluate
from model.weekly import (
    DEFAULT_ARTIFACT_PATH,
    LINEAR_SKILL_KEYS,
    SKILL_POSITIONS,
    WEEKLY_FEATURES,
    WeeklyModel,
    WeeklyRidge,
    assert_linear_skill_scoring,
    cold_start_metrics,
    deferred_positions,
    is_cold_start,
    recorded_cold_start_gate,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli(name: str):
    """Import ``scripts/<name>.py`` as a module (the CLIs are not on the import path)."""
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


ew = _load_cli("eval_weekly")

#: A minimal linear skill scoring, plus K/DST buckets — the shape of the real 42-key league. The guard
#: must pass on this and would fail if a bonus were added (see the bonus tests).
_LINEAR_LEAGUE = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_int": -1.0, "pass_2pt": 2.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
    "fum": 0.0, "fum_lost": -2.0, "fum_rec": 2.0, "fum_rec_td": 6.0,
    "fgm_40_49": 4.0, "fgm_50p": 5.0, "xpm": 1.0,           # K buckets — not skill
    "pts_allow_0": 10.0, "pts_allow_1_6": 7.0, "sack": 1.0, "def_st_td": 6.0,  # DST — not skill
}


# =============================================================== the linearity guard (Decision #7)
def test_guard_passes_on_a_linear_league_including_kicker_and_dst_buckets():
    """The guard is about SKILL keys: K's fgm_* and DST's pts_allow_* buckets must not trip it.

    Those buckets are real non-linearities, but they are #30's — the skill points head is unaffected by
    them, so flagging them here would be a false positive that blocks a valid league.
    """
    assert_linear_skill_scoring(_LINEAR_LEAGUE)  # does not raise


@pytest.mark.parametrize(
    "bad_key",
    ["bonus_rush_yd_100", "bonus_rec_yd_100", "bonus_pass_yd_300", "rush_rec_yd_200"],
)
def test_guard_raises_on_a_bucket_or_threshold_skill_key(bad_key):
    """Fail-closed on a yardage bonus/milestone — the exact case that biases a points head.

    Reverting the ``nonlinear`` branch in ``assert_linear_skill_scoring`` (letting these through) is what
    would let the real 2026 league silently bias the model, which is why this is written red-first.
    """
    with pytest.raises(ValueError, match="non-linear skill scoring"):
        assert_linear_skill_scoring({**_LINEAR_LEAGUE, bad_key: 1.0})


def test_guard_raises_on_an_unrecognised_skill_key_it_cannot_prove_linear():
    """A skill-prefixed key not in the allowlist and not obviously a bucket is still refused.

    ``rec_air_yd`` is plausibly linear, but the guard cannot *prove* it, so fail-closed means a human
    confirms and adds it to LINEAR_SKILL_KEYS rather than the model trusting an unknown key.
    """
    with pytest.raises(ValueError, match="unrecognised skill scoring key"):
        assert_linear_skill_scoring({**_LINEAR_LEAGUE, "rec_air_yd": 0.02})


def test_guard_spares_a_defensive_bonus_because_it_is_not_a_skill_key():
    """A ``bonus_def_*`` is #30's non-linearity, not the skill head's — it must not trip this guard."""
    assert_linear_skill_scoring({**_LINEAR_LEAGUE, "bonus_def_td_50": 2.0})  # does not raise


def test_guard_ignores_a_zero_weight_skill_key_and_rejects_empty_scoring():
    assert_linear_skill_scoring({**_LINEAR_LEAGUE, "bonus_rush_yd_100": 0.0})  # zero weight → no effect
    with pytest.raises(ValueError, match="scoring is empty"):
        assert_linear_skill_scoring({})


# =============================================================== features are all pre-lock
def test_weekly_features_carry_no_same_week_outcome_or_label():
    """Lookahead by construction: the feature list must contain no target-week outcome.

    The raw (un-lagged) usage columns and the label are the same-week quantities a leak would smuggle
    in; the model uses only their lagged forms. Adding ``y_custom_points`` or ``target_share`` (raw) to
    WEEKLY_FEATURES would break this immediately.
    """
    forbidden = {
        "y_custom_points", "target_share", "snap_pct", "exp_points", "rush_share",
        "wx_observed_temp_f", "wx_observed_wind_mph", "baseline_sleeper_points",
    }
    assert forbidden.isdisjoint(WEEKLY_FEATURES)
    # every feature the model names is a lag, a market line, or a fixed-pre-kickoff context flag
    assert all(
        f.endswith(("_last", "_ewma", "_trend"))
        or f in {"games_played_prior", "implied_team_total", "opp_implied_total", "team_spread_line",
                 "total_line", "is_div_game", "is_indoor"}
        for f in WEEKLY_FEATURES
    )


# =============================================================== synthetic weekly frame
def _frame(
    *, seasons=range(2016, 2021), positions=("QB", "RB", "WR", "TE"), n_players=8, weeks=range(1, 7),
    seed=0, market_driven=False,
) -> pd.DataFrame:
    """A synthetic weekly frame.

    Default: points track a stable per-player level with a lag the recency baseline can read. With
    ``market_driven=True`` the week's outcome tracks that week's ``implied_team_total`` **shock** around
    a nearly-flat player level — a signal recency averages away but the market carries, so the model can
    out-rank TrailingMean (the mechanism test).
    """
    rng = np.random.default_rng(seed)
    base = {"QB": 18.0, "RB": 11.0, "WR": 10.0, "TE": 7.0}
    rows: list[dict] = []
    for season in seasons:
        for pos in positions:
            for p in range(n_players):
                level = base[pos] + (0.3 * p if market_driven else float(p))
                prev = np.nan
                for w in weeks:
                    shock = rng.normal(0, 6) if market_driven else 0.0
                    implied = level + shock
                    if market_driven:
                        y = implied + rng.normal(0, 0.5)
                    else:
                        y = level + 0.1 * w + rng.normal(0, 0.3)
                    rows.append(
                        {
                            "player_id": f"{pos}{p}", "season": season, "week": w, "position": pos,
                            "y_custom_points": float(y),
                            "games_played_prior": float(w - 1),
                            "points_last": prev, "points_ewma": prev, "points_trend": np.nan,
                            "snap_pct_last": np.nan, "snap_pct_ewma": np.nan, "snap_pct_trend": np.nan,
                            "target_share_last": np.nan, "target_share_ewma": np.nan,
                            "rush_share_last": np.nan, "rush_share_ewma": np.nan,
                            "exp_points_last": np.nan, "exp_points_ewma": np.nan,
                            "implied_team_total": float(implied), "opp_implied_total": 21.0,
                            "team_spread_line": -1.0, "total_line": 44.0,
                            "is_div_game": False, "is_indoor": (p % 2 == 0),
                        }
                    )
                    prev = float(y)
    out = pd.DataFrame(rows)
    out["is_indoor"] = out["is_indoor"].astype("boolean")
    return out


# =============================================================== predictor protocol & shape
def test_weekly_ridge_is_a_predictor():
    assert isinstance(WeeklyRidge(), Predictor)


def test_fit_covers_the_four_skill_positions_and_predicts_finite_index_aligned():
    frame = _frame()
    model = WeeklyRidge().fit(frame)
    assert set(model.fit_positions) == {"QB", "RB", "WR", "TE"}
    preds = model.predict(frame)
    assert preds.index.equals(frame.index)
    assert preds.notna().all() and np.isfinite(preds.to_numpy()).all()


def test_predict_returns_null_for_a_position_the_model_did_not_fit():
    """K/DST are #30's — the weekly model must not silently invent a number for them."""
    frame = _frame()
    model = WeeklyRidge().fit(frame)
    kdef = pd.DataFrame(
        [{**{c: 0.0 for c in WEEKLY_FEATURES}, "player_id": "K1", "season": 2020, "week": 3,
          "position": "K", "y_custom_points": 8.0}]
    )
    assert model.predict(kdef).isna().all()


def test_predict_preserves_a_shuffled_frame_index():
    frame = _frame(seasons=[2016, 2017]).sample(frac=1.0, random_state=3)
    train = frame[frame["season"] == 2016]
    test = frame[frame["season"] == 2017]
    pred = WeeklyRidge().fit(train).predict(test)
    assert pred.index.equals(test.index)


# =============================================================== lookahead safety
def test_prediction_does_not_read_the_targets_own_week_label():
    """The model reads features, never the target-week outcome: spiking the label leaves predict fixed.

    This is the weekly analogue of the frame's lookahead gate. If ``_feature_matrix`` ever included
    ``y_custom_points`` (or a raw same-week stat), spiking it here would move the prediction and this
    fails.
    """
    frame = _frame(seasons=[2016, 2017])
    train = frame[frame["season"] == 2016]
    test = frame[frame["season"] == 2017]
    model = WeeklyRidge().fit(train)
    normal = model.predict(test)
    spiked = test.copy()
    spiked["y_custom_points"] = 999.0
    assert np.allclose(normal.to_numpy(), model.predict(spiked).to_numpy(), equal_nan=True)


# =============================================================== cold start (acceptance #2)
def test_week1_all_lags_null_predicts_finite_and_moves_with_the_market():
    """Acceptance #2: a week-1 row (no current-season lags) is predicted, and on the Vegas market.

    Reverting the mean-imputation in ``model.season._PositionFit`` (the reused solver) would leave the
    null lags as NaN through standardisation and make this prediction NaN — so a finite, market-varying
    week-1 prediction is exactly what proves the cold-start path.
    """
    model = WeeklyRidge().fit(_frame(market_driven=True))
    lo = {**{c: np.nan for c in WEEKLY_FEATURES}, "player_id": "WR9", "season": 2020, "week": 1,
          "position": "WR", "y_custom_points": np.nan, "games_played_prior": 0.0,
          "implied_team_total": 8.0, "opp_implied_total": 21.0, "team_spread_line": 7.0,
          "total_line": 38.0, "is_div_game": False, "is_indoor": False}
    hi = {**lo, "implied_team_total": 34.0, "team_spread_line": -10.0, "total_line": 54.0}
    cold = pd.DataFrame([lo, hi])
    cold["is_indoor"] = cold["is_indoor"].astype("boolean")
    preds = model.predict(cold)
    assert preds.notna().all() and np.isfinite(preds.to_numpy()).all()
    assert preds.iloc[1] > preds.iloc[0]  # a higher implied team total → a higher week-1 projection


# =============================================================== the mechanism (NOT the real bar)
def test_model_out_ranks_trailing_mean_when_the_signal_is_in_the_market_not_recency():
    """MECHANISM only: the model can exploit a market signal recency is blind to.

    This is deliberately NOT a claim about the real lake — that lives in docs/model-weekly.md. Here the
    week's outcome is the market shock around a flat player level, so TrailingMean (a per-player mean)
    cannot order within a week while the model (which sees implied_team_total) can.
    """
    frame = _frame(market_driven=True, seasons=range(2016, 2021))
    model = evaluate(WeeklyRidge(), frame, positions=("QB", "RB", "WR", "TE"), test_seasons=[2019, 2020])
    base = evaluate(TrailingMean(), frame, positions=("QB", "RB", "WR", "TE"), test_seasons=[2019, 2020])
    for pos in ("QB", "RB", "WR", "TE"):
        m, b = model.per_position[pos], base.per_position[pos]
        assert m.mae < b.mae, f"{pos}: model MAE {m.mae:.2f} not below TrailingMean {b.mae:.2f}"
        assert m.spearman > b.spearman, f"{pos}: model ρ {m.spearman:.3f} not above {b.spearman:.3f}"


# =============================================================== cold-start deferral (measured gate)
def test_is_cold_start_flags_the_first_appearance_of_a_season():
    """The cold-start cohort is games_played_prior == 0 — week 1 and mid-season debuts alike."""
    frame = pd.DataFrame(
        {"games_played_prior": [0.0, 1.0, 0.0, 3.0], "player_id": list("abcd")}
    )
    assert list(is_cold_start(frame)) == [True, False, True, False]


def test_is_cold_start_treats_a_frame_without_the_marker_as_all_cold():
    """No history column → the safe side: defer everything (a lag-less ridge never beats the baseline)."""
    assert is_cold_start(pd.DataFrame({"player_id": ["a", "b"]})).all()


def _cs_pm(mae: float, rho: float) -> PositionMetrics:
    return PositionMetrics(
        position="X", n=50, mae=mae, rmse=mae * 1.2, spearman=rho,
        spearman_slates=8, spearman_ordered_slates=8, calibration=pd.DataFrame(),
    )


def test_deferred_positions_defers_only_where_ridge_loses_the_cold_start():
    """The gate is measured and per-position: field ridge where it wins the cold start, defer where not.

    Reverting the ``if not wins`` gate to ``defer everything`` would put QB (a measured cold-start win)
    into the deferral set, and this fails — the blanket-deferral bug #31's DEF review warned against.
    """
    ridge_cold = {"QB": _cs_pm(6.5, 0.40), "RB": _cs_pm(5.0, 0.50)}
    baselines_cold = {
        "PriorSeasonRank": {"QB": _cs_pm(7.0, 0.30), "RB": _cs_pm(4.5, 0.60)},
        "TrailingMean": {"QB": _cs_pm(8.0, 0.0), "RB": _cs_pm(6.0, 0.0)},
    }
    # QB: ridge beats the toughest on both → fielded. RB: ridge worse MAE (5.0 > 4.5) → deferred.
    assert deferred_positions(ridge_cold, baselines_cold, positions=("QB", "RB")) == ("RB",)


def test_weekly_model_deferred_cold_start_rows_are_identical_to_the_contained_baseline():
    """Requirement #3 / acceptance analogue of #31's DEF pin: deferred rows == PriorSeasonRank exactly.

    Not merely close — a near-tie that drifts is not a guarantee. Reverting the deferral in
    ``WeeklyModel.predict`` (returning the ridge everywhere) makes the deferred WR cold-start rows
    differ from PriorSeasonRank, and this fails.
    """
    frame = _frame(seasons=range(2016, 2020))
    train, test = frame[frame["season"] < 2019], frame[frame["season"] == 2019]
    model = WeeklyModel(defer_cold_start=("WR",)).fit(train)
    prior = PriorSeasonRank().fit(train)
    ridge = WeeklyRidge().fit(train)

    preds = model.predict(test)
    cold = is_cold_start(test)
    wr = test["position"] == "WR"

    wr_cold = (wr & cold).to_numpy()
    assert wr_cold.any()
    # deferred rows (WR, cold) are the baseline, to the bit
    assert np.array_equal(preds.to_numpy()[wr_cold], prior.predict(test).to_numpy()[wr_cold])
    # WR non-cold rows, and cold rows at non-deferred positions, are the ridge
    wr_warm = (wr & ~cold).to_numpy()
    qb_cold = ((test["position"] == "QB") & cold).to_numpy()
    ridge_pred = ridge.predict(test).to_numpy()
    assert np.array_equal(preds.to_numpy()[wr_warm], ridge_pred[wr_warm])
    assert np.array_equal(preds.to_numpy()[qb_cold], ridge_pred[qb_cold])


def test_weekly_model_with_no_deferral_is_exactly_the_ridge():
    """The pure-ridge diagnostic — the explicit opt-out, analogous to SeasonModel(require_usage=False)."""
    frame = _frame()
    model = WeeklyModel(defer_cold_start=()).fit(frame)
    ridge = WeeklyRidge().fit(frame)
    assert np.array_equal(model.predict(frame).to_numpy(), ridge.predict(frame).to_numpy())


def test_default_weekly_model_is_safe_and_defers_at_the_recorded_gate():
    """The blocking review fix: a bare WeeklyModel() must NOT be the pure-ridge variant.

    Reverting __init__'s default to `()` (instead of the recorded gate) makes a default-constructed model
    pure ridge — the configuration the artifact itself records as losing the cold start at all four
    positions — and this fails. That is the polarity #31 got right (`require_usage=True` safe by default)
    and this had backwards.
    """
    gate = recorded_cold_start_gate()
    assert gate  # the committed artifact records a real, non-empty gate
    assert WeeklyModel().defer_cold_start == gate  # the bare default reads it — never ()


def test_recorded_gate_matches_the_committed_artifact_not_a_duplicated_constant():
    recorded = tuple(json.loads(DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))["cold_start_deferral"])
    assert recorded_cold_start_gate() == recorded  # the default IS the artifact's gate, cannot drift


def test_recorded_gate_falls_back_to_all_skills_when_the_artifact_is_absent():
    """A missing artifact must never yield () (silent pure ridge) — it defers every skill instead."""
    assert recorded_cold_start_gate(Path("does-not-exist-weekly-artifact.json")) == SKILL_POSITIONS


def test_default_constructed_model_returns_the_baseline_on_cold_rows():
    """Exercised via the DEFAULT path (no defer_cold_start argument), unlike the explicit-gate tests."""
    frame = _frame(seasons=range(2016, 2020))
    train, test = frame[frame["season"] < 2019], frame[frame["season"] == 2019]
    model = WeeklyModel().fit(train)
    prior = PriorSeasonRank().fit(train)
    defer = (is_cold_start(test) & test["position"].astype("string").isin(model.defer_cold_start)).to_numpy()
    assert defer.any()
    assert np.array_equal(model.predict(test).to_numpy()[defer], prior.predict(test).to_numpy()[defer])


def test_predict_raises_on_cold_rows_when_the_prior_is_unfit():
    """A constructed-but-unfit deferring model fails loudly on cold rows — never a silent pure ridge."""
    frame = _frame(seasons=[2016, 2017])
    model = WeeklyModel(defer_cold_start=("WR",))  # never fit → prior unfit
    with pytest.raises(RuntimeError, match="prior is unfit"):
        model.predict(frame[frame["season"] == 2017])  # contains WR cold rows


def test_load_fitted_reads_the_gate_and_weights_from_the_committed_artifact():
    """load_fitted is the artifact→model path: recorded gate + reconstructed ridge, prior left to fit."""
    model = WeeklyModel.load_fitted()
    recorded = tuple(json.loads(DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))["cold_start_deferral"])
    assert model.defer_cold_start == recorded
    assert set(model._ridge.fit_positions)  # ridge weights reconstructed from the artifact
    frame = _frame(seasons=[2016, 2017])
    test = frame[frame["season"] == 2017]
    with pytest.raises(RuntimeError, match="prior is unfit"):  # cold rows before the prior is fit
        model.predict(test)
    model.fit(frame[frame["season"] == 2016])  # fits the prior (and refreshes the ridge)
    assert model.predict(test).notna().all()


def test_cold_start_metrics_scores_only_the_first_appearance_rows():
    """cold_start_metrics recovers the cold cohort by joining predictions back to the frame's marker."""
    frame = _frame(seasons=range(2016, 2020))
    res = evaluate(WeeklyRidge(), frame, positions=("QB", "RB", "WR", "TE"), test_seasons=[2018, 2019])
    cold = cold_start_metrics(res.predictions, frame, positions=("QB", "RB", "WR", "TE"))
    # every scored cold row must be a games_played_prior==0 row; count them independently to confirm
    preds = res.predictions.merge(
        frame.loc[is_cold_start(frame), ["player_id", "season", "week"]].assign(_c=True),
        on=["player_id", "season", "week"], how="left",
    )
    expected = preds[preds["_c"].fillna(False) & (preds["position"] == "QB")].shape[0]
    assert cold["QB"].n == expected and expected > 0


# =============================================================== feature importances (acceptance #3)
def test_feature_importances_are_per_position_and_aligned_to_the_feature_list():
    model = WeeklyRidge().fit(_frame(market_driven=True))
    imp = model.feature_importances()
    assert set(imp) == {"QB", "RB", "WR", "TE"}
    for weights in imp.values():
        assert list(weights) == list(WEEKLY_FEATURES)  # keyed by feature, in order
    # in the market-driven frame the market feature must carry real weight
    assert abs(imp["WR"]["implied_team_total"]) > 0.0


# =============================================================== reproducible artifact (acceptance #4)
def test_fit_is_deterministic_and_round_trips_through_json():
    frame = _frame(market_driven=True)
    a, b = WeeklyRidge().fit(frame), WeeklyRidge().fit(frame)
    assert a.to_dict() == b.to_dict()  # closed-form ridge → identical refit, not a hand-edited file
    restored = WeeklyRidge.from_dict(a.to_dict())
    assert np.allclose(a.predict(frame).to_numpy(), restored.predict(frame).to_numpy(), equal_nan=True)


def test_save_load_round_trips_on_disk(tmp_path):
    frame = _frame(market_driven=True)
    model = WeeklyRidge().fit(frame)
    path = tmp_path / "weekly.json"
    model.save(path)
    reloaded = WeeklyRidge.load(path)
    assert np.allclose(model.predict(frame).to_numpy(), reloaded.predict(frame).to_numpy(), equal_nan=True)


def test_from_dict_rejects_a_drifted_feature_list():
    """A committed artifact whose feature list no longer matches the code must refuse to load."""
    payload = WeeklyRidge().fit(_frame()).to_dict()
    payload["features"] = payload["features"][:-1]  # a column was dropped since the fit
    with pytest.raises(ValueError, match="feature list does not match"):
        WeeklyRidge.from_dict(payload)


# =============================================================== the committed report follows its data
def _pm(pos: str, mae: float, rho: float | None, *, n: int = 100, ordered: int | None = None):
    cal = pd.DataFrame(
        {"decile": [1, 2], "n": [50, 50], "pred_mean": [1.0, 9.0], "realized_mean": [1.0, 9.0]}
    )
    slates = 0 if rho is None else 8
    return PositionMetrics(
        position=pos, n=n, mae=mae, rmse=mae * 1.3, spearman=rho, spearman_slates=slates,
        spearman_ordered_slates=slates if ordered is None else ordered, calibration=cal,
    )


def _permap(per):
    return {pos: _pm(pos, mae, rho) for pos, (mae, rho) in per.items()}


def _result(name, per, *, test_seasons=(2018, 2019)):
    return EvalResult(
        predictor=name, test_seasons=test_seasons, per_position=_permap(per), predictions=pd.DataFrame()
    )


#: All-weeks: baselines are the real recorded bar. The shipped model wins RB & WR on both, splits QB
#: (MAE better, ρ worse) and TE (MAE worse, ρ better).
_BAR = {"QB": (6.81, 0.409), "RB": (4.72, 0.577), "WR": (4.42, 0.527), "TE": (3.42, 0.432)}
_SHIPPED_ALL = {"QB": (6.0, 0.40), "RB": (4.0, 0.60), "WR": (4.2, 0.55), "TE": (3.5, 0.45)}
#: Cold start: PriorSeasonRank carries last-season level (the toughest); the other two are flat (ρ 0).
#: Pure ridge loses all four; the shipped model (all deferred) equals PriorSeasonRank → ties.
_COLD_PRIOR = {"QB": (6.90, 0.319), "RB": (4.69, 0.597), "WR": (4.67, 0.541), "TE": (3.28, 0.430)}
_COLD_FLAT = {"QB": (7.38, 0.0), "RB": (5.82, 0.0), "WR": (5.43, 0.0), "TE": (3.63, 0.0)}
_RIDGE_COLD = {"QB": (7.31, 0.212), "RB": (5.65, 0.084), "WR": (5.35, 0.054), "TE": (3.52, 0.046)}
_GATE = ("QB", "RB", "WR", "TE")


def _results(*, test_seasons=(2018, 2019)):
    out = {
        ew._SHIPPED: _result(ew._SHIPPED, _SHIPPED_ALL, test_seasons=test_seasons),
        ew._RIDGE: _result(ew._RIDGE, _SHIPPED_ALL, test_seasons=test_seasons),
    }
    for name in ew._BASELINES:
        out[name] = _result(name, _BAR, test_seasons=test_seasons)
    return out


def _cold():
    return {
        ew._SHIPPED: _permap(_COLD_PRIOR),  # all deferred → equals PriorSeasonRank
        ew._RIDGE: _permap(_RIDGE_COLD),
        "TrailingMean": _permap(_COLD_FLAT),
        "PriorSeasonRank": _permap(_COLD_PRIOR),
        "LaggedExpectedPoints": _permap(_COLD_FLAT),
    }


def _importances(leaned=False):
    base = [("implied_team_total", 2.5, 0.0), ("points_ewma", 1.8, 0.10), ("total_line", 0.4, 0.0)]
    if leaned:
        base = [("exp_points_last", 3.0, 0.62), *base]  # a big coef on a mostly-null feature
    return {pos: base for pos in ("QB", "RB", "WR", "TE")}


def _render(*, importances=None, records=(), gate=_GATE):
    return ew.render_report(
        _results(), _cold(), importances or _importances(),
        {"depth_pos_rank": 0.89, "baseline_sleeper_points": 1.0}, gate,
        seasons=[2016, 2025], scoring_keys=42, partitions=412, league_name="Test league",
        generated="2026-07-31", frame_rows=169685, cohort_rows=56429, players=4603, records=records,
    )


def _finding(report: str, n: int) -> str:
    return next(line for line in report.splitlines() if line.startswith(f"{n}. **"))


def test_report_all_weeks_headline_verdicts_are_derived_win_and_split():
    section = _render().split("## A. All-weeks headline")[1].split("## B.")[0]
    rows = {p: next(ln for ln in section.splitlines() if ln.strip().startswith(f"| {p} |"))
            for p in ("QB", "RB", "WR", "TE")}
    assert "win" in rows["RB"] and "win" in rows["WR"]  # shipped model beats the bar on both metrics
    assert "split" in rows["QB"] and "split" in rows["TE"]


def test_report_cold_start_shows_ridge_losing_and_shipped_tying_after_deferral():
    section = _render().split("## B. Cold-start headline")[1].split("## C.")[0]
    ridge_block, shipped_block = section.split("**Shipped model**")
    ridge_qb = next(ln for ln in ridge_block.splitlines() if ln.strip().startswith("| QB |"))
    shipped_qb = next(ln for ln in shipped_block.splitlines() if ln.strip().startswith("| QB |"))
    assert "loss" in ridge_qb  # pure ridge loses the cold start
    assert "tie (deferred)" in shipped_qb  # the shipped model defers → equals PriorSeasonRank


def test_report_finding_1_names_only_the_positions_won_on_both_metrics():
    finding = _finding(_render(), 1)
    assert "Won at: RB, WR" in finding  # derived from the shipped-model verdicts, not asserted
    assert "QB split" in finding and "TE split" in finding


def test_report_finding_2_reports_the_cold_start_margin_and_the_measured_gate():
    """Requirement #2/#1: the cold-start margin is a number, and the deferral is measured per position."""
    finding = _finding(_render(), 2)
    assert "QB loss (ΔMAE +0.41, Δρ -0.107)" in finding  # the margin, derived from the tables
    assert "where it loses (QB, RB, WR, TE)" in finding
    assert "where it wins (none)" in finding


def test_report_finding_2_fields_ridge_where_it_wins_the_cold_start():
    """The gate is not blanket: a position absent from the gate is reported as fielded, not deferred."""
    finding = _finding(_render(gate=("RB", "WR", "TE")), 2)  # QB fielded (ridge won its cold start)
    assert "where it wins (QB)" in finding
    assert "where it loses (RB, WR, TE)" in finding


def test_report_finding_3_shows_the_shipped_model_ties_the_baseline_it_defers_to():
    finding = _finding(_render(), 3)
    assert "tie (deferred)" in finding  # deferred cold-start rows equal the baseline → a tie, not a loss


def test_report_finding_4_flags_a_mostly_null_feature_the_model_leans_on():
    """Acceptance #3: a big coefficient on a feature #27 showed is mostly null is a stated finding."""
    finding = _finding(_render(importances=_importances(leaned=True)), 4)
    assert "leans on a mostly-null feature" in finding
    assert "exp_points_last" in finding and "62% null" in finding


def test_report_finding_4_is_clean_when_no_top_feature_is_mostly_null():
    assert "No top feature is mostly null" in _finding(_render(), 4)


def test_report_finding_5_reports_the_excluded_columns_with_measured_null_rates():
    finding = _finding(_render(), 5)
    assert "depth_pos_rank" in finding and "89%" in finding
    assert "baseline_sleeper_points" in finding and "100%" in finding


def test_report_counts_warnings_from_the_captured_log_and_quotes_them():
    records = [(30, "WARNING", "model.evaluate", "no rows for test season 2018 — skipping it")]
    report = _render(records=records)
    assert "emitted **1** WARNING-level" in _finding(report, 6)
    assert "no rows for test season 2018" in report.split("## Warnings (verbatim)")[1]


def test_report_says_zero_warnings_when_the_log_carried_none():
    report = _render()
    assert "Zero warning" in _finding(report, 6)
    assert "_None._" in report.split("## Warnings (verbatim)")[1]


def test_report_header_states_the_seasons_actually_scored_not_a_default():
    report = ew.render_report(
        _results(test_seasons=(2021, 2022)), _cold(), _importances(), {"depth_pos_rank": 0.89}, _GATE,
        seasons=[2016, 2025], scoring_keys=42, partitions=412, league_name="Test league",
        generated="2026-07-31", frame_rows=1, cohort_rows=1, players=1,
    )
    header = next(ln for ln in report.splitlines() if ln.startswith("- **Train span"))
    assert "2021–2022" in header


def test_linear_skill_keys_cover_the_leagues_real_keys():
    """Every non-zero skill key of the real 42-key league is recognised as linear (no false alarm)."""
    for key in ("pass_yd", "rush_td", "rec", "rec_yd", "fum_lost", "fum_rec"):
        assert key in LINEAR_SKILL_KEYS

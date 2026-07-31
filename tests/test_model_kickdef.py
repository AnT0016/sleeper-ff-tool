"""Offline tests for the weekly K + DST component model (Phase 9, ticket #30).

The evidence discipline (Decision #9 item 1): **no beats-the-baseline number is asserted here.** A
synthetic frame can be built to make any model win, and #31 was caught doing exactly that. The only
bar-clearing numbers live in ``docs/model-kickdef.md``, measured on the real lake by
``scripts/eval_kickdef.py``. These tests pin **mechanics**, each guard written so that reverting the
guarded code turns the test red:

* the two structural claims — a **distribution** over the points-allowed bucket beats bucketing the mean,
  and per-**band** FG makes beat a mean-distance collapse (``E[f(X)] != f(E[X])``, Decision #2);
* predictions are **priced by ``scoring.engine.points``** over predicted stat lines, so a scoring change
  re-prices with no retraining — never a points head (Decision #7's carve-out);
* the correctness **anchor** (``engine(components) == label``) is complete, and the shutout reconstruction
  that makes it so;
* the model reads **only pre-lock features** (component columns are labels, never features);
* **cold start** produces a finite prediction that moves with the Vegas market;
* the **per-cell gate** defers where the component model loses or the cell is thin, ships safe by default
  (a bare constructor is the recorded gate), and the fitted artifact round-trips and is read back.

No lake and no network: every frame is synthetic, built to make the property under test decidable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.evaluate import PositionMetrics, Predictor, evaluate
from model.kickdef import (
    ANCHOR_FLOOR_PCT,
    COMP_PREFIX,
    DEFAULT_ARTIFACT_PATH,
    KICKDEF_FEATURES,
    KickDefModel,
    _CountingHead,
    _DefenseHead,
    _feature_matrix,
    _fit_pts_allow_dist,
    _KickerHead,
    _parse_pts_allow_bounds,
    _PtsAllowDist,
    _reconstruct_pts_allow,
    anchor_mismatch,
    bucket_of_points_allowed,
    cell_key,
    cell_metrics,
    component_model,
    deferred_cells,
    expected_pts_allow_points,
    pts_allow_bucket_probs,
    pts_allow_keys,
    pts_allow_points,
    recorded_gate,
)
from model.season import _PositionFit
from scoring.engine import points as engine_points

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli(name: str):
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


ek = _load_cli("eval_kickdef")

#: A realistic scoring shape: kicker distance bands, DST counting keys and the seven points-allowed
#: buckets — the step functions #30 exists for.
_SCORING = {
    "fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0, "fgm_40_49": 4.0, "fgm_50p": 5.0,
    "xpm": 1.0, "fgmiss": -1.0, "xpmiss": -1.0,
    "sack": 1.0, "int": 2.0, "fum_rec": 2.0, "def_td": 6.0, "safe": 2.0, "ff": 1.0, "blk_kick": 2.0,
    "def_st_td": 6.0, "def_st_ff": 1.0, "def_st_fum_rec": 1.0,
    "pts_allow_0": 10.0, "pts_allow_1_6": 7.0, "pts_allow_7_13": 4.0, "pts_allow_14_20": 1.0,
    "pts_allow_21_27": 0.0, "pts_allow_28_34": -1.0, "pts_allow_35p": -4.0,
}

_PA_KEYS = pts_allow_keys(_SCORING)


# =============================================================== points-allowed buckets (from scoring)
def test_bucket_bounds_are_parsed_from_key_names_not_hardcoded():
    assert _parse_pts_allow_bounds("pts_allow_0") == (0, 0)
    assert _parse_pts_allow_bounds("pts_allow_1_6") == (1, 6)
    assert _parse_pts_allow_bounds("pts_allow_7_13") == (7, 13)
    assert _parse_pts_allow_bounds("pts_allow_35p") == (35, float("inf"))


@pytest.mark.parametrize(
    "value,expected",
    [(0, "pts_allow_0"), (6, "pts_allow_1_6"), (7, "pts_allow_7_13"), (20, "pts_allow_14_20"),
     (34, "pts_allow_28_34"), (35, "pts_allow_35p"), (70, "pts_allow_35p")],
)
def test_bucket_of_points_allowed(value, expected):
    assert bucket_of_points_allowed(value, _PA_KEYS) == expected


# =============================================================== THE distribution claim (Decision #2)
def test_distribution_over_the_bucket_beats_bucketing_the_mean():
    """The centrepiece: E over a distribution straddling a boundary != bucketing the mean.

    Points allowed of 6 or 8 with equal probability has mean 7 (a `pts_allow_7_13` = 4-point game), but
    the expectation over the distribution is 0.5·7 + 0.5·4 = 5.5. Swapping the distribution for the mean
    — the bias Decision #2 forbids — gives 4.0, and this fails. Written red-first.
    """
    dist = {6.0: 0.5, 8.0: 0.5}
    e_dist = expected_pts_allow_points(dist, _SCORING)
    e_mean = pts_allow_points(sum(k * p for k, p in dist.items()), _SCORING)
    assert e_dist == pytest.approx(5.5)
    assert e_mean == pytest.approx(4.0)
    assert e_dist != pytest.approx(e_mean)  # the point estimate is provably wrong


def test_expected_points_equals_engine_over_bucket_probabilities():
    """The engine prices a probability-mass line into the exact expectation (why it needs no special case).

    ``expected_pts_allow_points({k: P(k)})`` must equal ``engine.points({bucket_key: P(bucket)})`` — the
    identity that lets the model emit bucket *probabilities* as a stat line and the linear engine turn
    them into ``E[pts_allow points]``.
    """
    samples = np.array([0, 3, 8, 8, 25])  # a small empirical distribution over points allowed
    probs = pts_allow_bucket_probs(samples, _PA_KEYS)
    engine_scored = engine_points(probs, _SCORING)
    # RHS computed directly over the samples (a dict would collapse the duplicate 8s and undercount).
    by_value = float(np.mean([pts_allow_points(v, _SCORING) for v in samples]))
    assert engine_scored == pytest.approx(by_value)


def test_pts_allow_dist_predicts_a_spread_not_a_single_bucket():
    """A hand-built distribution straddling the 6/7 boundary yields mass in two buckets, not one."""
    mu_fit = _PositionFit(mean=np.array([0.0]), std_safe=np.array([1.0]), weights=np.array([0.0]),
                          intercept=7.0, y_mean=7.0)  # predicts μ = 7 for any row
    dist = _PtsAllowDist(mu_fit, bin_edges=[], resid_grids=[[-1.0, 1.0]])  # samples land at 6 and 8
    probs = dist.predict_bucket_probs(np.array([[0.0]]), _PA_KEYS)
    assert probs["pts_allow_1_6"][0] == pytest.approx(0.5)
    assert probs["pts_allow_7_13"][0] == pytest.approx(0.5)


def test_defense_head_uses_the_distribution_not_the_mean():
    """The DST head prices the *distribution*: reverting it to bucket-the-mean changes the number.

    A defence head with zero counting stats and the straddling distribution above must score 5.5 (the
    distribution's expectation), not 4.0 (the bucket of μ = 7).
    """
    head = _DefenseHead(_SCORING)
    head.counting = _CountingHead((), 1.0)  # no counting contribution
    mu_fit = _PositionFit(mean=np.array([0.0]), std_safe=np.array([1.0]), weights=np.zeros(len(KICKDEF_FEATURES)),
                          intercept=7.0, y_mean=7.0)
    head.pa = _PtsAllowDist(mu_fit, bin_edges=[], resid_grids=[[-1.0, 1.0]])
    one_row = pd.DataFrame([{f: 0.0 for f in KICKDEF_FEATURES}])
    assert head.predict(one_row)[0] == pytest.approx(5.5)
    assert head.predict(one_row)[0] != pytest.approx(pts_allow_points(7, _SCORING))  # != 4.0


# =============================================================== THE FG-band claim (Decision #2)
def test_fg_per_band_makes_differ_from_a_mean_distance_collapse():
    """Per-band expected makes, priced by the engine, != collapsing to one make at the mean distance.

    A kicker with 0.5 expected makes in the 40-49 band and 0.5 in the 50+ band scores 0.5·4 + 0.5·5 = 4.5.
    Collapsing to "one make" at either band gives 4.0 or 5.0 — the distance non-linearity a band-blind
    model gets wrong. The band grain needs no distribution precisely because the coefficient is constant
    within a band, so this is exact.
    """
    per_band = engine_points({"fgm_40_49": 0.5, "fgm_50p": 0.5}, _SCORING)
    collapse_low = engine_points({"fgm_40_49": 1.0}, _SCORING)
    collapse_high = engine_points({"fgm_50p": 1.0}, _SCORING)
    assert per_band == pytest.approx(4.5)
    assert collapse_low == pytest.approx(4.0) and collapse_high == pytest.approx(5.0)
    assert per_band != pytest.approx(collapse_low) and per_band != pytest.approx(collapse_high)


# =============================================================== synthetic frame
def _bucket(pa: float) -> str:
    return bucket_of_points_allowed(pa, _PA_KEYS)


def _def_line(pa: float, sacks: float, ints: float) -> dict:
    return {f"{COMP_PREFIX}pts_allow": pa, f"{COMP_PREFIX}sack": sacks, f"{COMP_PREFIX}int": ints,
            f"{COMP_PREFIX}fum_rec": 1.0, f"{COMP_PREFIX}def_td": 0.0, f"{COMP_PREFIX}safe": 0.0,
            f"{COMP_PREFIX}ff": 1.0, f"{COMP_PREFIX}blk_kick": 0.0, f"{COMP_PREFIX}def_st_td": 0.0,
            f"{COMP_PREFIX}def_st_ff": 0.0, f"{COMP_PREFIX}def_st_fum_rec": 0.0}


def _k_line(fg30: float, fg50: float, xp: float) -> dict:
    return {f"{COMP_PREFIX}fgm_0_19": 0.0, f"{COMP_PREFIX}fgm_20_29": 0.0, f"{COMP_PREFIX}fgm_30_39": fg30,
            f"{COMP_PREFIX}fgm_40_49": 0.0, f"{COMP_PREFIX}fgm_50p": fg50, f"{COMP_PREFIX}xpm": xp,
            f"{COMP_PREFIX}fgmiss": 0.0, f"{COMP_PREFIX}xpmiss": 0.0}


def _label_from_components(row: dict) -> float:
    """``engine(components)`` for a synthetic row — the label the anchor must reproduce (independently)."""
    stats: dict[str, float] = {}
    for name, value in row.items():
        if not name.startswith(COMP_PREFIX):
            continue
        key = name[len(COMP_PREFIX):]
        if key == "pts_allow":
            stats[_bucket(value)] = stats.get(_bucket(value), 0.0) + 1.0
        elif value:
            stats[key] = float(value)
    return engine_points(stats, _SCORING)


def _frame(*, seasons=range(2016, 2021), n=6, weeks=range(1, 6), seed=0) -> pd.DataFrame:
    """A synthetic K+DST frame: components track the market so a fitted head learns, y = engine(line)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for season in seasons:
        for pos in ("K", "DEF"):
            for p in range(n):
                prev = np.nan
                for w in weeks:
                    implied = float(rng.uniform(16, 30))
                    opp = float(rng.uniform(16, 30))
                    row = {
                        "player_id": f"{pos}{p}", "season": season, "week": w, "position": pos,
                        "implied_team_total": implied, "opp_implied_total": opp,
                        "team_spread_line": float(rng.uniform(-7, 7)), "total_line": implied + opp,
                        "is_indoor": bool(p % 2), "is_div_game": bool(w % 2),
                        "games_played_prior": float(w - 1),
                        "points_last": prev, "points_ewma": prev, "points_trend": np.nan,
                    }
                    if pos == "K":
                        row.update(_k_line(fg30=round(implied / 12), fg50=round(implied / 20),
                                           xp=round(implied / 8)))
                    else:
                        pa = float(np.clip(round(opp + rng.normal(0, 3)), 0, 60))
                        row.update(_def_line(pa, sacks=round(35 - opp) / 8.0, ints=round((30 - opp) / 12)))
                    row["y_custom_points"] = _label_from_components(row)
                    rows.append(row)
                    prev = row["y_custom_points"]
    out = pd.DataFrame(rows)
    out["is_indoor"] = out["is_indoor"].astype("boolean")
    return out


# =============================================================== predictor protocol & pricing
def test_kickdef_model_is_a_predictor():
    assert isinstance(KickDefModel(_SCORING), Predictor)


def test_fit_requires_scoring():
    with pytest.raises(ValueError, match="scoring_settings"):
        KickDefModel().fit(_frame())  # no scoring → cannot price components through the engine


def test_predict_is_finite_index_aligned_and_null_for_unowned_positions():
    frame = _frame()
    model = component_model(_SCORING).fit(frame)
    preds = model.predict(frame)
    assert preds.index.equals(frame.index)
    assert preds.notna().all() and np.isfinite(preds.to_numpy()).all()
    qb = pd.DataFrame([{**{f: 0.0 for f in KICKDEF_FEATURES}, "player_id": "Q1", "season": 2020,
                        "week": 3, "position": "QB", "games_played_prior": 2.0}])
    assert model.predict(qb).isna().all()  # K/DEF only


def test_kicker_prediction_is_the_engine_over_predicted_bands_not_a_points_head():
    """The kicker prediction is ``engine.points`` over per-band expected makes, not a regressed number."""
    frame = _frame()
    kf = frame[frame["position"] == "K"]
    head = _KickerHead(_SCORING).fit(kf)
    stats = head.counting.predict_stats(_feature_matrix(kf))
    manual = np.array([
        engine_points({k: max(0.0, stats[k][i]) for k in stats}, _SCORING) for i in range(len(kf))
    ])
    assert np.allclose(head.predict(kf), manual)


def test_rescoring_reprices_predictions_with_no_retraining():
    """Decision #2's payoff: change the scoring, the engine re-prices the *same* components, no refit.

    Doubling `fgm_50p` from 5 to 10 must shift a kicker's prediction by exactly (predicted 50+ makes)·5,
    computed from the identical fitted weights. A points head could not do this — it regressed points
    under the old scoring and would be stale.
    """
    frame = _frame()
    kf = frame[frame["position"] == "K"]
    head = _KickerHead(_SCORING).fit(kf)
    richer = _KickerHead.from_dict(head.to_dict(), {**_SCORING, "fgm_50p": 10.0})  # same weights
    stats = head.counting.predict_stats(_feature_matrix(kf))
    delta = richer.predict(kf) - head.predict(kf)
    assert np.allclose(delta, np.clip(stats["fgm_50p"], 0.0, None) * (10.0 - 5.0))


# =============================================================== the correctness anchor
def test_anchor_reproduces_the_label_from_the_component_line():
    """``engine(observed components)`` reproduces ``y_custom_points`` for a synthetic frame (100%)."""
    frame = _frame()
    anchor = anchor_mismatch(frame, _SCORING)
    assert anchor["K"]["match_pct"] == pytest.approx(100.0)
    assert anchor["DEF"]["match_pct"] == pytest.approx(100.0)


def test_anchor_flags_an_incomplete_decomposition():
    """Reverting to an incomplete component set (a scoring key unextracted) drops the match rate.

    Spiking a DEF label so its components no longer reproduce it must show as a mismatch, with the
    implicated keys named — the guard that makes the declared floor meaningful.
    """
    frame = _frame()
    frame.loc[frame["position"] == "DEF", "y_custom_points"] += 6.0  # a phantom def_td the components lack
    anchor = anchor_mismatch(frame, _SCORING)
    assert anchor["DEF"]["match_pct"] < ANCHOR_FLOOR_PCT
    assert anchor["DEF"]["mismatch_keys"]  # names the scoring keys on the missed rows


def test_reconstruct_pts_allow_fills_shutouts_from_the_bucket_flag():
    """Sleeper leaves raw `pts_allow` null on a shutout; the flag says 0 — reconstruct it or lose 10 pts.

    Reverting the reconstruction (reading raw `pts_allow` alone) leaves the shutout row null, which the
    anchor then misses by 10 points *and* which would train the distribution on zero shutouts.
    """
    raw = pd.DataFrame({
        "pts_allow": [pd.NA, 24.0, pd.NA],
        "pts_allow_0": [1.0, pd.NA, pd.NA],       # row 0 is a shutout
        "pts_allow_21_27": [pd.NA, 1.0, pd.NA],
        "pts_allow_14_20": [pd.NA, pd.NA, 1.0],   # row 2: null raw, mid-bucket flag → lower bound 14
    })
    filled = _reconstruct_pts_allow(raw, _PA_KEYS)
    assert filled.iloc[0] == 0.0   # shutout filled to exactly 0
    assert filled.iloc[1] == 24.0  # present raw untouched
    assert filled.iloc[2] == 14.0  # inferred from the set bucket's lower bound


# =============================================================== lookahead safety
def test_features_carry_no_component_label_or_same_week_outcome():
    """Component columns are labels; the feature list must contain none of them, nor the label."""
    assert not any(f.startswith(COMP_PREFIX) for f in KICKDEF_FEATURES)
    assert "y_custom_points" not in KICKDEF_FEATURES
    assert all(
        f.endswith(("_last", "_ewma", "_trend"))
        or f in {"games_played_prior", "implied_team_total", "opp_implied_total", "team_spread_line",
                 "total_line", "is_div_game", "is_indoor"}
        for f in KICKDEF_FEATURES
    )


def test_prediction_does_not_read_the_targets_own_label_or_components():
    """Spiking a row's own-week label and components leaves the prediction fixed (reads only features)."""
    frame = _frame(seasons=[2016, 2017])
    train, test = frame[frame["season"] == 2016], frame[frame["season"] == 2017]
    model = component_model(_SCORING).fit(train)
    normal = model.predict(test)
    spiked = test.copy()
    spiked["y_custom_points"] = 999.0
    for col in [c for c in spiked.columns if c.startswith(COMP_PREFIX)]:
        spiked[col] = 999.0
    assert np.allclose(normal.to_numpy(), model.predict(spiked).to_numpy(), equal_nan=True)


# =============================================================== cold start
def test_cold_start_predicts_finite_and_moves_with_the_market():
    """A week-1 row (all lags null) is predicted, and on the Vegas market that is present at week 1."""
    model = component_model(_SCORING).fit(_frame())
    base = {**{f: np.nan for f in KICKDEF_FEATURES}, "games_played_prior": 0.0, "is_indoor": False,
            "is_div_game": False, "team_spread_line": 0.0}
    lo_def = {**base, "player_id": "DEFx", "season": 2020, "week": 1, "position": "DEF",
              "opp_implied_total": 17.0, "implied_team_total": 24.0, "total_line": 41.0}
    hi_def = {**lo_def, "opp_implied_total": 31.0, "total_line": 55.0}  # tougher opponent
    lo_k = {**base, "player_id": "Kx", "season": 2020, "week": 1, "position": "K",
            "implied_team_total": 16.0, "opp_implied_total": 21.0, "total_line": 37.0}
    hi_k = {**lo_k, "implied_team_total": 31.0, "total_line": 52.0}  # richer scoring environment
    cold = pd.DataFrame([lo_def, hi_def, lo_k, hi_k])
    cold["is_indoor"] = cold["is_indoor"].astype("boolean")
    preds = model.predict(cold)
    assert preds.notna().all() and np.isfinite(preds.to_numpy()).all()
    assert preds.iloc[1] < preds.iloc[0]   # a tougher opponent → fewer DST points
    assert preds.iloc[3] > preds.iloc[2]   # a richer environment → more kicker points


# =============================================================== the per-cell gate
def _pm(n: int, mae: float, rho: float) -> PositionMetrics:
    return PositionMetrics(position="X", n=n, mae=mae, rmse=mae * 1.2, spearman=rho,
                           spearman_slates=8, spearman_ordered_slates=8, calibration=pd.DataFrame())


def test_deferred_cells_defers_on_a_loss_a_tie_and_a_thin_cell():
    """The gate is measured, per cell, and fails safe: a loss, a tie, or too little evidence all defer.

    Reverting the ``c.n >= min_n`` guard would field the thin winning cell, and reverting ``wins`` to
    ``True`` would field the losing one — both fail here.
    """
    component = {
        "K:warm": _pm(4000, 3.5, 0.20),   # clear win, thick → fielded
        "K:cold": _pm(120, 3.0, 0.30),    # would win, but n < 200 → deferred (thin)
        "DEF:warm": _pm(4000, 5.2, 0.10),  # worse MAE than the bar → deferred (loss)
        "DEF:cold": _pm(4000, 4.5, 0.09),  # ties ρ → deferred (not a strict win on both)
    }
    baselines = {
        "TrailingMean": {"K:warm": _pm(4000, 4.0, 0.07), "K:cold": _pm(120, 3.8, 0.05),
                         "DEF:warm": _pm(4000, 5.0, 0.09), "DEF:cold": _pm(4000, 4.9, 0.09)},
    }
    gate = deferred_cells(component, baselines)
    assert set(gate) == {"K:cold", "DEF:warm", "DEF:cold"}
    assert "K:warm" not in gate  # the one clear, thick win is fielded


def test_deferred_cell_prediction_is_exactly_the_baseline():
    """A deferred cell predicts the contained baseline to the bit; fielded cells predict the component."""
    frame = _frame(seasons=range(2016, 2020))
    train, test = frame[frame["season"] < 2019], frame[frame["season"] == 2019]
    from model.baselines import PriorSeasonRank
    model = KickDefModel(_SCORING, defer=("K:cold",)).fit(train)
    prior = PriorSeasonRank().fit(train)
    preds = model.predict(test).to_numpy()

    is_k = (test["position"] == "K").to_numpy()
    is_cold = (test["games_played_prior"] == 0).to_numpy()
    k_cold = is_k & is_cold
    assert k_cold.any()
    assert np.array_equal(preds[k_cold], prior.predict(test).to_numpy()[k_cold])  # deferred == baseline
    # a fielded cell (K warm) is NOT the baseline (the component model differs from a group mean)
    k_warm = is_k & ~is_cold
    assert not np.allclose(preds[k_warm], prior.predict(test).to_numpy()[k_warm])


def test_no_deferral_is_exactly_the_pure_component_model():
    frame = _frame()
    a = KickDefModel(_SCORING, defer=()).fit(frame).predict(frame)
    b = component_model(_SCORING).fit(frame).predict(frame)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_predict_raises_on_a_deferred_cell_when_the_baseline_is_unfit():
    """A constructed-but-unfit deferring model fails loudly on a deferred row — never a silent component."""
    frame = _frame(seasons=[2016, 2017])
    model = KickDefModel(_SCORING, defer=("K:cold",))  # never fit → baseline unfit
    with pytest.raises(RuntimeError, match="unfit"):
        model.predict(frame[frame["season"] == 2017])  # contains K cold rows


# =============================================================== safe by default (Decision #9 item 3/5)
def test_bare_constructor_reads_the_recorded_gate():
    """A bare ``KickDefModel()`` is the shipped configuration — its gate is the artifact's, not a guess."""
    assert KickDefModel().defer == recorded_gate()  # constructed with no args (item 5)


def test_recorded_gate_matches_the_committed_artifact():
    recorded = tuple(json.loads(DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))["deferral"])
    assert recorded_gate() == recorded


def test_recorded_gate_falls_back_to_all_cells_when_the_artifact_is_absent():
    """A missing artifact must never yield () (silent field-everything) — it defers every cell."""
    gate = recorded_gate(Path("does-not-exist-kickdef-artifact.json"))
    assert set(gate) == {cell_key(p, cold) for p in ("K", "DEF") for cold in (False, True)}


# =============================================================== reproducible artifact (Decision #9 #4)
def test_fit_is_deterministic_and_round_trips_through_json():
    frame = _frame()
    a = KickDefModel(_SCORING, defer=()).fit(frame)
    b = KickDefModel(_SCORING, defer=()).fit(frame)
    assert a.to_dict() == b.to_dict()  # closed-form → identical refit, not a hand-edited file
    restored = KickDefModel.from_dict(a.to_dict())
    assert np.allclose(a.predict(frame).to_numpy(), restored.predict(frame).to_numpy(), equal_nan=True)


def test_save_load_round_trips_on_disk(tmp_path):
    frame = _frame()
    model = KickDefModel(_SCORING, defer=()).fit(frame)
    path = tmp_path / "kickdef.json"
    model.save(path)
    reloaded = KickDefModel.load(path)
    assert np.allclose(model.predict(frame).to_numpy(), reloaded.predict(frame).to_numpy(), equal_nan=True)


def test_from_dict_rejects_a_drifted_feature_list():
    payload = KickDefModel(_SCORING, defer=()).fit(_frame()).to_dict()
    payload["features"] = payload["features"][:-1]  # a column was dropped since the fit
    with pytest.raises(ValueError, match="feature list does not match"):
        KickDefModel.from_dict(payload)


def test_load_fitted_reads_the_gate_and_heads_but_leaves_the_baseline_to_fit():
    """load_fitted is the artifact→model path: recorded gate + reconstructed heads, baseline left to fit."""
    model = KickDefModel.load_fitted()
    recorded = tuple(json.loads(DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))["deferral"])
    assert model.defer == recorded
    assert model._kicker is not None and model._defense is not None  # heads reconstructed
    frame = _frame(seasons=[2016, 2017])
    model.fit(frame)  # fits the baseline (and refreshes the heads)
    assert model.predict(frame).notna().all()


# =============================================================== distribution fit shape
def test_pts_allow_dist_conditions_on_mu_and_is_deterministic():
    """Three μ bins → three residual grids and two interior edges; refit is bit-identical (closed form)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, len(KICKDEF_FEATURES)))
    pa = np.clip(20 + 5 * x[:, 1] + rng.normal(0, 4, size=600), 0, None)
    a = _fit_pts_allow_dist(x, pa, n_bins=3, alpha=1.0)
    b = _fit_pts_allow_dist(x, pa, n_bins=3, alpha=1.0)
    assert len(a.resid_grids) == 3 and len(a.bin_edges) == 2
    assert a.to_dict() == b.to_dict()


# =============================================================== per-cell metrics
def test_cell_metrics_splits_into_four_position_by_cohort_cells():
    frame = _frame(seasons=range(2016, 2020))
    res = evaluate(component_model(_SCORING), frame, positions=("K", "DEF"), test_seasons=[2018, 2019])
    cells = cell_metrics(res.predictions, frame)
    assert set(cells) == {"K:warm", "K:cold", "DEF:warm", "DEF:cold"}
    assert all(m.n > 0 for m in cells.values())


# =============================================================== the committed report follows its data
def _cell_pm(n, mae, rho):
    cal = pd.DataFrame({"decile": [1], "n": [n], "pred_mean": [mae], "realized_mean": [mae]})
    return PositionMetrics(position="X", n=n, mae=mae, rmse=mae * 1.3, spearman=rho,
                           spearman_slates=8, spearman_ordered_slates=8, calibration=cal)


def _result(name, per):
    from model.evaluate import EvalResult
    return EvalResult(predictor=name, test_seasons=(2018, 2019),
                      per_position={p: _cell_pm(100, *v) for p, v in per.items()}, predictions=pd.DataFrame())


#: A component model that beats the bar on both metrics at both positions (the real-lake shape).
_COMPONENT_ALL = {"K": (3.6, 0.18), "DEF": (4.7, 0.28)}
_BAR = {"K": (3.68, 0.067), "DEF": (4.95, 0.095)}
_COMPONENT_CELLS = {"K:warm": _cell_pm(3898, 3.6, 0.18), "K:cold": _cell_pm(340, 3.75, 0.14),
                    "DEF:warm": _cell_pm(3998, 4.74, 0.29), "DEF:cold": _cell_pm(256, 4.84, 0.20)}
_BASE_CELLS = {
    "TrailingMean": {"K:warm": _cell_pm(3898, 4.1, 0.07), "K:cold": _cell_pm(340, 3.8, 0.0),
                     "DEF:warm": _cell_pm(3998, 5.4, 0.09), "DEF:cold": _cell_pm(256, 4.9, 0.0)},
    "PriorSeasonRank": {"K:warm": _cell_pm(3898, 3.8, 0.03), "K:cold": _cell_pm(340, 3.88, 0.01),
                        "DEF:warm": _cell_pm(3998, 5.05, 0.09), "DEF:cold": _cell_pm(256, 4.91, 0.18)},
    "LaggedExpectedPoints": {"K:warm": _cell_pm(3898, 3.67, 0.0), "K:cold": _cell_pm(340, 3.78, 0.0),
                             "DEF:warm": _cell_pm(3998, 4.95, 0.0), "DEF:cold": _cell_pm(256, 4.94, 0.0)},
}


def _results():
    out = {ek._SHIPPED: _result(ek._SHIPPED, _COMPONENT_ALL), ek._COMPONENT: _result(ek._COMPONENT, _COMPONENT_ALL)}
    for name in ek._BASELINES:
        out[name] = _result(name, _BAR)
    return out


def _cal(pred0_d1):
    return pd.DataFrame([{"decile": 1, "n": 426, "mu_mean": 15.5, "pred_le0": pred0_d1, "real_le0": 0.0423,
                         "pred_le6": 0.15, "real_le6": 0.14}])


def _anchor(k=100.0, d=100.0):
    return {"K": {"n": 5253, "matched": 5253, "match_pct": k, "mismatch_keys": {}},
            "DEF": {"n": 5246, "matched": 5246, "match_pct": d, "mismatch_keys": {}}}


def _render(*, gate=(), records=()):
    return ek.render_report(
        _results(), _COMPONENT_CELLS, _BASE_CELLS, gate, _anchor(), _cal(0.0472), _cal(0.0555),
        seasons=[2016, 2025], n_pa_bins=3, scoring_keys=42, partitions=412, league_name="Test league",
        generated="2026-07-31", frame_rows=10499, k_rows=5253, def_rows=5246, players=134, records=records,
    )


def _finding(report: str, n: int) -> str:
    return next(line for line in report.splitlines() if line.startswith(f"{n}. **"))


def test_report_finding_1_names_the_won_positions_from_the_data():
    finding = _finding(_render(), 1)
    assert "K win" in finding and "DEF win" in finding
    assert "Won at: K, DEF" in finding


def test_report_finding_2_lists_the_fielded_and_deferred_cells():
    finding = _finding(_render(gate=("K:cold",)), 2)
    assert "K:warm n=3,898 win" in finding
    assert "deferred: K:cold" in finding and "Fielded: K:warm, DEF:warm, DEF:cold" in finding


def test_report_headline_verdict_is_derived_win():
    section = _render().split("## A. All-weeks headline")[1].split("## B.")[0]
    for pos in ("K", "DEF"):
        row = next(ln for ln in section.splitlines() if ln.strip().startswith(f"| {pos} |"))
        assert "win" in row


def test_report_calibration_note_compares_single_grid_to_binned():
    finding = _finding(_render(), 3)
    assert "0.0555" in finding and "0.0472" in finding  # single-grid over-prediction, closed by binning


def test_report_anchor_section_reports_the_match_rate():
    report = _render()
    section = report.split("## F. Correctness anchor")[1]
    assert "100.00" in section


def test_report_counts_warnings_and_quotes_them():
    records = [(30, "WARNING", "model.evaluate", "no rows for test season 2018 — skipping it")]
    report = _render(records=records)
    assert "emitted **1** WARNING-level" in _finding(report, 5)
    assert "no rows for test season 2018" in report.split("## Warnings (verbatim)")[1]


def test_report_says_zero_warnings_when_the_log_carried_none():
    report = _render()
    assert "Zero warning" in _finding(report, 5)
    assert "_None._" in report.split("## Warnings (verbatim)")[1]


def test_report_header_states_the_seasons_actually_scored():
    report = _render()
    header = next(ln for ln in report.splitlines() if ln.startswith("- **Train span"))
    assert "2018–2019" in header  # read back from the results' test_seasons

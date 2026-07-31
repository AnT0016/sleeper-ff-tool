"""Offline tests for the walk-forward harness and the three naive baselines (Phase 9, ticket #28).

The test that matters most is the **leak gate**, written red-first: a split whose training rows reach
into the test season must fail to exist. It is this ticket's analogue of Phase 8's lookahead gate, and
it fails closed the same way — because a model scored on a contaminated split is invisibly optimistic,
while a rejected split is loud.

Everything else pins a property the spec's decisions turn on:

* metrics are **per position** (a pooled number would let a QB-only gain masquerade as general);
* Spearman is computed **within a real ``(season, week)`` slate**, never pooled across boards;
* the baselines are lookahead-safe (an extreme current-week label does not move that week's
  prediction) and index-aligned;
* and ``scripts/eval_baselines.py``'s committed report has its prose *generated from* the numbers it
  cites, so a headline can never contradict the table beneath it.

No lake and no network: every frame here is synthetic, built to make the property under test decidable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.baselines import LaggedExpectedPoints, PriorSeasonRank, TrailingMean
from model.evaluate import (
    DEFAULT_TEST_SEASONS,
    FANTASY_POSITIONS,
    EvalResult,
    LeakError,
    PositionMetrics,
    Predictor,
    Split,
    assert_walk_forward,
    calibration_by_decile,
    evaluate,
    per_position_metrics,
    spearman,
    walk_forward_splits,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli(name: str):
    """Import ``scripts/<name>.py`` as a module (the CLIs are not on the import path)."""
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


eb = _load_cli("eval_baselines")

_POS_BASE = {"QB": 18.0, "RB": 11.0, "WR": 10.0, "TE": 7.0, "K": 8.0, "DEF": 7.0}


def _frame(
    *,
    seasons=range(2016, 2021),
    positions=FANTASY_POSITIONS,
    weeks=range(1, 6),
    n_players=4,
) -> pd.DataFrame:
    """A synthetic cohort frame with distinct, per-player scoring levels and a within-season lag.

    Values are deterministic and distinct within a slate (so a Spearman is always defined), with a
    modest per-week wobble. ``exp_points_last`` is present for skill positions from week 2 (null at
    week 1, mirroring the real cold-start shape) and absent for K/DEF.
    """
    rng = np.random.default_rng(0)
    rows = []
    for season in seasons:
        for pos in positions:
            for p in range(n_players):
                level = _POS_BASE[pos] + p
                for week in weeks:
                    y = level + 0.1 * week + rng.normal(0, 0.3)
                    exp = (
                        level + 0.1 * week
                        if pos in ("QB", "RB", "WR", "TE") and week > min(weeks)
                        else np.nan
                    )
                    rows.append(
                        {
                            "player_id": f"{pos}{p}",
                            "season": season,
                            "week": week,
                            "position": pos,
                            "y_custom_points": float(y),
                            "exp_points_last": exp,
                        }
                    )
    return pd.DataFrame(rows)


# =============================================================== the leak gate (red-first)
def test_a_split_that_trains_on_its_test_season_is_rejected():
    """The centrepiece: training rows from the test season make the split refuse to exist."""
    frame = _frame(seasons=[2016, 2017])
    leaky_train = frame  # includes 2017, the test season
    with pytest.raises(LeakError):
        Split(test_season=2017, train=leaky_train, test=frame[frame["season"] == 2017])


def test_assert_walk_forward_rejects_a_training_row_at_or_after_the_test_season():
    frame = _frame(seasons=[2016, 2017, 2018])
    with pytest.raises(LeakError):
        assert_walk_forward(frame[frame["season"] >= 2017], test_season=2017)


def test_a_clean_backward_split_is_accepted_and_carries_one_test_season():
    frame = _frame(seasons=[2016, 2017])
    split = Split(
        test_season=2017,
        train=frame[frame["season"] < 2017],
        test=frame[frame["season"] == 2017],
    )
    assert split.test_season == 2017
    assert set(split.test["season"].unique()) == {2017}


def test_a_test_partition_that_mixes_seasons_is_rejected():
    frame = _frame(seasons=[2016, 2017])
    with pytest.raises(LeakError):
        Split(test_season=2017, train=frame[frame["season"] < 2016], test=frame)


def test_a_season_the_gate_cannot_parse_is_rejected_rather_than_waved_through():
    """Fail-closed means closed on unreadable input too, not just on a readable violation.

    ``pd.to_numeric(errors="coerce")`` turns a junk season into ``NaN``, and ``NaN >= 2018`` is
    ``False`` — so the row slid past the leak check while a *missing* column raised. A gate that
    passes what it cannot read is the one failure mode that leaves no trace in the output.
    """
    junk = _frame(seasons=[2016]).astype({"season": object})
    junk.loc[junk.index[0], "season"] = "not-a-season"
    with pytest.raises(LeakError, match="does not parse"):
        assert_walk_forward(junk, test_season=2018)


# =============================================================== splits are by season only
def test_walk_forward_train_is_strictly_before_the_test_season():
    frame = _frame(seasons=range(2016, 2021))
    seen = []
    for split in walk_forward_splits(frame, test_seasons=[2018, 2019, 2020]):
        assert split.train["season"].max() < split.test_season
        assert set(split.test["season"].unique()) == {split.test_season}
        seen.append(split.test_season)
    assert seen == [2018, 2019, 2020]


def test_default_test_seasons_never_score_the_warmup_seasons():
    """2016-2017 are lag/EWMA warm-up (Decision #6): never a test season."""
    assert 2016 not in DEFAULT_TEST_SEASONS
    assert 2017 not in DEFAULT_TEST_SEASONS
    assert DEFAULT_TEST_SEASONS[0] == 2018
    assert DEFAULT_TEST_SEASONS[-1] == 2025


def test_a_test_season_with_no_earlier_training_is_skipped_not_leaked():
    frame = _frame(seasons=[2018, 2019])
    seasons = [s.test_season for s in walk_forward_splits(frame, test_seasons=[2018, 2019])]
    assert seasons == [2019]  # 2018 has no prior season in the frame → skipped, never self-trained


def test_no_player_week_from_the_test_season_ever_lands_in_train():
    """The property a random / player-stratified k-fold would violate — pinned directly."""
    frame = _frame(seasons=range(2016, 2020))  # every player recurs every season
    for split in walk_forward_splits(frame, test_seasons=[2018, 2019]):
        train_keys = set(
            zip(split.train["player_id"], split.train["season"], split.train["week"], strict=True)
        )
        test_keys = set(
            zip(split.test["player_id"], split.test["season"], split.test["week"], strict=True)
        )
        assert train_keys.isdisjoint(test_keys)


def test_the_module_exposes_no_random_or_kfold_splitter():
    """'Impossible to select' made enforceable: the only splitter is the season one."""
    import model.evaluate as ev

    banned = ("kfold", "k_fold", "random", "shuffle", "stratified")
    offenders = [n for n in dir(ev) if any(b in n.lower() for b in banned)]
    assert offenders == []


# =============================================================== per position, never pooled
def test_evaluate_reports_per_position_and_carries_no_pooled_metric():
    res = evaluate(TrailingMean(), _frame(), test_seasons=[2018, 2019])
    assert set(res.per_position) <= set(FANTASY_POSITIONS)
    assert len(res.per_position) >= 1
    for attr in ("mae", "rmse", "spearman", "pooled", "overall"):
        assert not hasattr(res, attr)  # no top-level number to pool a QB gain into


def test_a_qb_only_label_change_moves_qb_but_not_rb():
    frame = _frame(seasons=range(2016, 2020))
    base = evaluate(TrailingMean(), frame, test_seasons=[2018, 2019])
    bumped = frame.copy()
    is_qb = bumped["position"] == "QB"
    # Collapse QB scoring to a constant — a constant *shift* would move the trailing-mean prediction
    # in lockstep with the label and leave the error (and MAE) untouched, which is the wrong probe.
    bumped.loc[is_qb, "y_custom_points"] = 3.0
    after = evaluate(TrailingMean(), bumped, test_seasons=[2018, 2019])
    assert after.per_position["QB"].mae != pytest.approx(base.per_position["QB"].mae)
    assert after.per_position["RB"].mae == pytest.approx(base.per_position["RB"].mae)


# =============================================================== Spearman within (season, week)
def test_spearman_is_averaged_within_season_week_not_pooled_across_slates():
    """Two slates with exactly opposite orderings (+1 and -1) must average to 0.

    Pooling the six rows ignoring the slate would *not* give 0 — so a result of 0 is proof the grain
    is ``(season, week)``.
    """
    preds = pd.DataFrame(
        {
            "position": ["WR"] * 6,
            "season": [2018, 2018, 2018, 2019, 2019, 2019],
            "week": [1, 1, 1, 1, 1, 1],
            "pred": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "actual": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        }
    )
    m = per_position_metrics(preds)["WR"]
    assert m.spearman == pytest.approx(0.0)
    assert m.spearman_slates == 2


def test_within_slate_ordering_is_not_diluted_by_the_cross_week_level():
    preds = pd.DataFrame(
        {
            "position": ["RB"] * 6,
            "season": [2018] * 6,
            "week": [1, 1, 1, 2, 2, 2],
            "pred": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "actual": [7.0, 8.0, 9.0, 1.0, 2.0, 3.0],  # perfect within each week, crossed across them
        }
    )
    m = per_position_metrics(preds)["RB"]
    assert m.spearman == pytest.approx(1.0)
    assert m.spearman_slates == 2


def test_a_constant_prediction_slate_is_scored_zero_not_excused():
    """A flat prediction offered no ordering — that is a score of 0, not an absence of one.

    Excusing it was the original behaviour and it made the ρ column incomparable: on the real lake
    ``LaggedExpectedPoints`` was flat on 134 of 141 kicker boards, so its printed ρ was the mean over
    the 7 it happened to speak on, sitting in the same column as a baseline's mean over all 141.
    """
    preds = pd.DataFrame(
        {
            "position": ["K"] * 4,
            "season": [2018, 2018, 2019, 2019],
            "week": [1, 1, 1, 1],
            "pred": [5.0, 5.0, 5.0, 5.0],  # a pure position-mean fallback: no ordering
            "actual": [1.0, 9.0, 2.0, 8.0],
        }
    )
    m = per_position_metrics(preds)["K"]
    assert m.spearman == pytest.approx(0.0)
    assert m.spearman_slates == 2  # both boards scored...
    assert m.spearman_ordered_slates == 0  # ...and neither was actually ordered


def test_a_slate_where_everyone_scored_the_same_is_skipped_not_charged_as_zero():
    """No prediction could have ordered that board, so it is not the predictor's zero to carry."""
    preds = pd.DataFrame(
        {
            "position": ["DEF"] * 2,
            "season": [2018, 2018],
            "week": [1, 1],
            "pred": [3.0, 9.0],  # a real ordering, offered
            "actual": [7.0, 7.0],  # every defense scored the same
        }
    )
    m = per_position_metrics(preds)["DEF"]
    assert m.spearman is None
    assert m.spearman_slates == 0


def test_a_mostly_flat_slate_is_diluted_by_the_boards_it_stayed_silent_on():
    """The real ``LaggedExpectedPoints``/K shape, in miniature: one real value against a block of ties.

    Under the old "skip the flat boards" rule the ρ here would be the perfect +1 of the single board
    that spoke. Scoring the silent boards as 0 puts that 1 in its true context.
    """
    rows = []
    for season in (2018, 2019, 2020, 2021):
        # Only 2018 gets a real ordering, and it is a perfect one (ρ = +1); the other three boards
        # are pure fallback, so the baseline said nothing about them at all.
        for p, a in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]:
            rows.append(
                {
                    "position": "K",
                    "season": season,
                    "week": 1,
                    "pred": p if season == 2018 else 5.0,
                    "actual": a,
                }
            )
    m = per_position_metrics(pd.DataFrame(rows))["K"]
    assert m.spearman_slates == 4
    assert m.spearman_ordered_slates == 1
    assert m.spearman == pytest.approx(1.0 / 4)  # not 1.0 — three boards it declined to order


@pytest.mark.parametrize(
    "pred,actual,expected",
    [
        ([1, 2, 3], [10, 20, 30], 1.0),
        ([1, 2, 3], [30, 20, 10], -1.0),
        ([1, 1, 1], [1, 2, 3], 0.0),  # constant *prediction* → no ordering offered → scored 0
        ([1, 2, 3], [5, 5, 5], None),  # constant *actual* → undefined for anyone
        ([1], [1], None),  # n < 2 → undefined
    ],
)
def test_spearman_edge_cases(pred, actual, expected):
    rho = spearman(pd.Series(pred, dtype=float), pd.Series(actual, dtype=float))
    if expected is None:
        assert rho is None
    else:
        assert rho == pytest.approx(expected)


# =============================================================== calibration by decile
def test_calibration_is_by_decile_and_realized_climbs_with_the_predicted_decile():
    pred = pd.Series(np.arange(100, dtype=float))
    actual = pred.copy()  # perfectly calibrated and perfectly ordered
    cal = calibration_by_decile(pred, actual)
    assert len(cal) == 10
    assert list(cal["decile"]) == list(range(1, 11))
    assert cal["realized_mean"].is_monotonic_increasing
    assert (cal["pred_mean"] - cal["realized_mean"]).abs().max() == pytest.approx(0.0)


# =============================================================== the baselines
@pytest.mark.parametrize("factory", [TrailingMean, PriorSeasonRank, LaggedExpectedPoints])
def test_each_baseline_satisfies_the_predictor_protocol(factory):
    assert isinstance(factory(), Predictor)


def test_the_predictor_protocol_rejects_an_object_missing_predict():
    """``runtime_checkable`` checks method *presence* only, so pin that it at least does that.

    It cannot check signatures or return types — which is why ``evaluate`` re-checks at runtime that
    what came back is a Series (``test_evaluate_rejects_a_predictor_that_does_not_return_a_series``)
    rather than trusting this ``isinstance``.
    """

    class FitOnly:
        def fit(self, frame):
            return self

    assert not isinstance(FitOnly(), Predictor)


def test_trailing_mean_ignores_the_current_weeks_own_label():
    """The lookahead test: spiking week 5's own label must not move week 5's prediction."""
    frame = pd.DataFrame(
        [
            {
                "player_id": "p",
                "season": 2020,
                "week": w,
                "position": "RB",
                "y_custom_points": float(w),
                "exp_points_last": np.nan,
            }
            for w in range(1, 6)
        ]
    )
    normal = TrailingMean(n=4).fit(frame).predict(frame)
    spiked = frame.copy()
    spiked.loc[spiked["week"] == 5, "y_custom_points"] = 999.0
    spiked_pred = TrailingMean(n=4).fit(spiked).predict(spiked)

    w5 = frame.index[frame["week"] == 5][0]
    assert normal[w5] == pytest.approx(spiked_pred[w5])  # unchanged by its own label
    assert normal[w5] == pytest.approx(np.mean([1, 2, 3, 4]))  # the mean of the four prior weeks
    w3 = frame.index[frame["week"] == 3][0]
    assert normal[w3] == pytest.approx(np.mean([1, 2]))  # never reaches forward to weeks 4-5


def test_all_baselines_cover_every_cohort_row_and_stay_index_aligned():
    frame = _frame(seasons=range(2016, 2020))
    train = frame[frame["season"] < 2019]
    test = frame[frame["season"] == 2019]
    for factory in (TrailingMean, PriorSeasonRank, LaggedExpectedPoints):
        pred = factory().fit(train).predict(test)
        assert isinstance(pred, pd.Series)
        assert pred.notna().all()  # the shared fallback fills every cold-start row
        assert pred.index.equals(test.index)


def test_predict_preserves_a_shuffled_frame_index():
    frame = _frame(seasons=[2016, 2017]).sample(frac=1.0, random_state=3)  # non-monotone index
    train = frame[frame["season"] == 2016]
    test = frame[frame["season"] == 2017]
    pred = TrailingMean().fit(train).predict(test)
    assert pred.index.equals(test.index)


def test_lagged_expected_points_reads_the_lag_for_skill_and_falls_back_for_k():
    frame = pd.DataFrame(
        [
            {"player_id": "wr", "season": 2018, "week": 2, "position": "WR",
             "y_custom_points": 12.0, "exp_points_last": 9.0},
            {"player_id": "k", "season": 2018, "week": 2, "position": "K",
             "y_custom_points": 8.0, "exp_points_last": np.nan},
        ]
    )
    pred = LaggedExpectedPoints().fit(frame).predict(frame)
    assert pred.iloc[0] == pytest.approx(9.0)  # WR: the lagged expected-points value
    assert pred.iloc[1] == pytest.approx(8.0)  # K: no exp_points column → learned position mean


def test_prior_season_rank_uses_the_prior_season_not_the_current_one():
    train = pd.DataFrame(
        [
            {"player_id": "a", "season": 2017, "week": w, "position": "RB",
             "y_custom_points": 10.0, "exp_points_last": np.nan}
            for w in range(1, 5)
        ]
    )
    test = pd.DataFrame(
        [
            {"player_id": "a", "season": 2018, "week": w, "position": "RB",
             "y_custom_points": 99.0, "exp_points_last": np.nan}  # wildly different this year
            for w in range(1, 5)
        ]
    )
    pred = PriorSeasonRank().fit(train).predict(test)
    assert np.allclose(pred.to_numpy(), 10.0)  # last season's average, not this season's 99


def test_prior_season_rank_falls_back_for_a_rookie_with_no_prior_season():
    train = pd.DataFrame(
        [
            {"player_id": "vet", "season": 2017, "week": w, "position": "WR",
             "y_custom_points": 8.0, "exp_points_last": np.nan}
            for w in range(1, 5)
        ]
    )
    test = pd.DataFrame(
        [
            {"player_id": "rook", "season": 2018, "week": 1, "position": "WR",
             "y_custom_points": 20.0, "exp_points_last": np.nan}
        ]
    )
    pred = PriorSeasonRank().fit(train).predict(test)
    assert pred.iloc[0] == pytest.approx(8.0)  # no 2017 row → the learned WR position mean


def test_trailing_mean_rejects_a_nonpositive_window():
    with pytest.raises(ValueError):
        TrailingMean(n=0)


# =============================================================== evaluate() integration
def test_evaluate_scores_a_perfect_predictor_at_zero_error_and_unit_rho():
    class Oracle:
        def fit(self, frame):
            return self

        def predict(self, frame):
            return pd.to_numeric(frame["y_custom_points"], errors="coerce")

    res = evaluate(Oracle(), _frame(seasons=range(2016, 2020)), test_seasons=[2018, 2019], name="o")
    assert res.predictor == "o"
    for m in res.per_position.values():
        assert m.mae == pytest.approx(0.0)
        assert m.spearman == pytest.approx(1.0)


def test_evaluate_rejects_a_predictor_that_does_not_return_a_series():
    class Bad:
        def fit(self, frame):
            return self

        def predict(self, frame):
            return [0.0] * len(frame)

    with pytest.raises(TypeError):
        evaluate(Bad(), _frame(), test_seasons=[2018])


def test_evaluate_on_the_real_baselines_is_per_position_and_covers_the_cohort():
    frame = _frame(seasons=range(2016, 2021))
    res = evaluate(PriorSeasonRank(), frame)
    assert set(res.per_position) == set(FANTASY_POSITIONS)
    # one held-out row per (player, position, week) across the 2018-2020 test seasons in the frame
    assert all(m.n > 0 for m in res.per_position.values())


# =============================================================== the committed report follows its data
def _pm(
    pos: str, mae: float, rho: float | None, *, n: int = 100, ordered: int | None = None
) -> PositionMetrics:
    cal = pd.DataFrame(
        {"decile": [1, 2], "n": [50, 50], "pred_mean": [1.0, 9.0], "realized_mean": [1.0, 9.0]}
    )
    slates = 0 if rho is None else 5
    return PositionMetrics(
        position=pos, n=n, mae=mae, rmse=mae * 1.2, spearman=rho,
        spearman_slates=slates,
        spearman_ordered_slates=slates if ordered is None else ordered,
        calibration=cal,
    )


def _result(
    name: str,
    per: dict[str, tuple[float, float | None]],
    *,
    test_seasons: tuple[int, ...] = (2018, 2019),
    n: int = 100,
    ordered: dict[str, int] | None = None,
) -> EvalResult:
    return EvalResult(
        predictor=name,
        test_seasons=test_seasons,
        per_position={
            pos: _pm(pos, mae, rho, n=n, ordered=(ordered or {}).get(pos))
            for pos, (mae, rho) in per.items()
        },
        predictions=pd.DataFrame(),
    )


def _render(results, *, records=()) -> str:
    return eb.render_report(
        results,
        seasons=[2016, 2025],
        scoring_keys=42,
        partitions=1,
        league_name="Test league",
        generated="2026-07-29",
        frame_rows=1000,
        cohort_rows=600,
        players=50,
        records=records,
    )


def _finding(report: str, n: int) -> str:
    return next(line for line in report.splitlines() if line.startswith(f"{n}. **"))


def test_report_finding_1_names_the_mae_best_baseline_per_position_from_the_data():
    a = _result("A", {"QB": (2.0, 0.5), "RB": (5.0, 0.3)})
    b = _result("B", {"QB": (3.0, 0.6), "RB": (4.0, 0.4)})
    finding = _finding(_render([a, b]), 1)
    assert "QB → A (2.00)" in finding  # A wins QB on MAE
    assert "RB → B (4.00)" in finding  # B wins RB on MAE


def test_report_finding_2_names_positions_where_mae_and_rho_disagree():
    a = _result("A", {"QB": (2.0, 0.3)})  # best MAE
    b = _result("B", {"QB": (3.0, 0.9)})  # best ρ
    finding = _finding(_render([a, b]), 2)
    assert "MAE→A" in finding
    assert "ρ→B" in finding


def test_report_finding_2_uses_the_agreement_branch_when_one_baseline_wins_both():
    a = _result("A", {"QB": (2.0, 0.9)})  # best on both
    b = _result("B", {"QB": (3.0, 0.3)})
    assert "agree" in _finding(_render([a, b]), 2).lower()


def test_report_finding_3_counts_the_boards_the_baseline_actually_ordered():
    """The real defect this replaced: the prose said "kickers ... no ordering" while the table three
    lines below showed K with a ρ over 7 slates. The claim is now the count itself.
    """
    lep = _result(
        "LaggedExpectedPoints",
        {"QB": (3.0, 0.4), "K": (4.0, -0.06), "DEF": (4.0, 0.0)},
        ordered={"QB": 5, "K": 1, "DEF": 0},
    )
    finding = _finding(_render([lep]), 3)
    assert "**K** 1/5" in finding
    assert "**DEF** 0/5" in finding
    assert "QB 5/5" in finding


def test_report_finding_3_does_not_call_a_position_flat_while_its_table_shows_an_ordering():
    """A position the baseline orders on every board must not be named as one it does not order.

    This is the assertion the old hardcoded clause could not make: it named K and DEF from memory,
    so it stayed true-looking no matter what the table said.
    """
    lep = _result(
        "LaggedExpectedPoints",
        {"K": (4.0, 0.30), "DEF": (4.0, 0.0)},
        ordered={"K": 5, "DEF": 0},
    )
    finding = _finding(_render([lep]), 3)
    assert "**DEF** 0/5" in finding
    assert "**K**" not in finding  # K ordered every board — it is not one of the flat ones


def test_report_finding_3_flips_when_expected_points_orders_every_position():
    lep = _result("LaggedExpectedPoints", {"K": (4.0, 0.2), "DEF": (4.0, 0.2)})
    assert "not expected" in _finding(_render([lep]), 3)


def test_report_finding_4_compares_coverage_across_baselines_instead_of_printing_the_first():
    """The claim "all N cover the same rows" has to survive a baseline that does not.

    The original read ``results[0]``'s counts and printed them under the sentence regardless, so a
    half-covering predictor left the claim standing next to its own counterexample.
    """
    full = _result("Full", {"K": (4.0, 0.2)}, n=4238)
    holey = _result("Holey", {"K": (4.0, 0.2)}, n=2118)
    finding = _finding(_render([full, holey]), 4)

    assert "do NOT all cover the same rows" in finding
    assert "Full 4,238" in finding
    assert "Holey 2,118" in finding


def test_report_finding_4_affirms_equal_coverage_only_when_it_is_equal():
    a = _result("A", {"K": (4.0, 0.2)}, n=4238)
    b = _result("B", {"K": (4.5, 0.1)}, n=4238)
    finding = _finding(_render([a, b]), 4)
    assert "All 2 baselines cover the same rows" in finding  # the count is derived too
    assert "K 4,238" in finding


def test_report_header_states_the_seasons_actually_scored_not_the_default_span():
    """``--seasons 2020-2022`` scored 2021-2022 while the header advertised 2018-2025."""
    r = _result("A", {"QB": (3.0, 0.4)}, test_seasons=(2021, 2022))
    header = next(ln for ln in _render([r]).splitlines() if ln.startswith("- **Train span"))
    assert "2021–2022" in header
    assert "2018–2025" not in header


def test_report_counts_warnings_from_the_captured_log_and_quotes_them():
    """A skipped test season is a WARNING; a silent bar would be narrower than its header."""
    records = [(30, "WARNING", "model.evaluate", "no rows for test season 2018 — skipping it")]
    report = _render([_result("A", {"QB": (3.0, 0.4)})], records=records)

    assert "emitted **1** WARNING-level" in _finding(report, 5)
    assert "no rows for test season 2018" in report.split("## Warnings (verbatim)")[1]


def test_report_says_zero_warnings_only_when_the_log_carried_none():
    report = _render([_result("A", {"QB": (3.0, 0.4)})])
    assert "**Zero warning" in _finding(report, 5)
    assert "_None._" in report.split("## Warnings (verbatim)")[1]


def test_report_metric_table_prints_an_em_dash_for_an_undefined_spearman():
    report = _render([_result("LaggedExpectedPoints", {"K": (4.0, None)})])
    k_row = next(line for line in report.splitlines() if line.strip().startswith("| K |"))
    assert "—" in k_row  # the ρ cell of the first (metric) table


def test_report_best_baseline_table_reflects_the_measured_winners():
    a = _result("A", {"WR": (2.0, 0.2)})  # best MAE, worse ρ
    b = _result("B", {"WR": (3.0, 0.8)})  # worse MAE, best ρ
    report = _render([a, b])
    section = report.split("## Best baseline per position")[1]
    wr_row = next(line for line in section.splitlines() if line.strip().startswith("| WR |"))
    assert "A (2.00)" in wr_row  # lowest MAE column
    assert "B (0.800)" in wr_row  # highest ρ column


def test_report_flags_a_rho_comparison_made_over_different_slate_universes():
    """The ρ column is only a like-for-like comparison while the baselines were scored on one set of
    boards. When they were not, the report has to say so rather than crown a winner silently."""
    a = _result("A", {"WR": (2.0, 0.2)})
    b = _result("B", {"WR": (3.0, 0.8)})
    object.__setattr__(b.per_position["WR"], "spearman_slates", 7)  # frozen dataclass
    section = _render([a, b]).split("## Best baseline per position")[1]
    assert "at WR they are **not**" in section
    assert "indicative only" in section


def test_report_warns_that_a_flat_baselines_calibration_gap_is_arithmetic_not_evidence():
    """LaggedExpectedPoints posts the best `gap` in the table for K/DEF *because* it predicts the
    mean. Left unflagged, the most degenerate predictor reads as the best calibrated one."""
    lep = _result("LaggedExpectedPoints", {"K": (4.0, 0.0)}, ordered={"K": 0})
    section = _render([lep]).split("## Calibration")[1]
    assert "LaggedExpectedPoints/K" in section
    assert "by construction" in section

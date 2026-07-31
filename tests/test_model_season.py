"""Offline tests for the draft frame and the draft-value model (Phase 9, ticket #31).

The two properties that matter most, both written to be decidable on synthetic data:

* **Fail-closed at season grain.** A feature for season S may use only seasons ≤ S-1. The pin is the
  season analogue of the weekly harness's "a spike to week N's own label does not move week N's
  prediction": spiking a player's season-S points moves only the label, and ``changed_team_prior``
  ignores a move made *into* S — proving the season-S team is never read (spec acceptance #1).
* **Beats the bar on ordering.** Draft value is a ranking problem, so :class:`SeasonModel` must beat
  :class:`PriorSeasonTotal` on within-``(season, position)`` Spearman ρ, not on MAE (spec acceptance
  #2). The synthetic frame gives the cleaner prior-usage features real ranking signal the noisy
  prior-points baseline cannot see, so the multivariate model wins.

Everything else pins a spec acceptance point: rookies are a handled cohort (#4), the label is the summed
engine-scored weekly total (#5), and the model's board drives ``draft.vor`` unchanged (#3). No lake and
no network: every frame here is built to make the property under test decidable.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from draft.vor import add_vor, replacement_levels, tierize
from model.frame import (
    LABEL_COL,
    SEASON_FEATURES,
    build_season_frame,
    season_frame_from_weekly,
)
from model.season import (
    PriorSeasonTotal,
    SeasonEvalResult,
    SeasonModel,
    SeasonPositionMetrics,
    evaluate_season,
    season_value_board,
    to_player_rows,
)
from projections.board import PlayerRow

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli(name: str):
    """Import ``scripts/<name>.py`` as a module (the CLIs are not on the import path)."""
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


es = _load_cli("eval_season")

_USAGE_LAGS = ("snap_pct_last", "target_share_last", "rush_share_last", "exp_points_last")


# --------------------------------------------------------------------------- synthetic weekly frame
def _weekly(rows: list[dict]) -> pd.DataFrame:
    """A minimal build_training_frame-shaped weekly frame from per-week specs.

    Each spec needs at least player_id, season, week, position, team and y_custom_points; the four
    lagged-usage columns default to NaN so a caller can omit them.
    """
    out = pd.DataFrame(rows)
    for col in _USAGE_LAGS:
        if col not in out.columns:
            out[col] = np.nan
    if "is_dst" not in out.columns:
        out["is_dst"] = False
    return out


def _player_season(
    player_id: str, season: int, position: str, team: str, points: list[float], *, snap: float = 0.5
) -> list[dict]:
    """One player's weeks in one season, each week scoring the given points."""
    return [
        {
            "player_id": player_id,
            "season": season,
            "week": w,
            "position": position,
            "team": team,
            "is_dst": position == "DEF",
            "y_custom_points": pts,
            "snap_pct_last": snap if w > 1 else np.nan,
            "target_share_last": snap * 0.4 if w > 1 else np.nan,
            "rush_share_last": np.nan,
            "exp_points_last": pts if w > 1 else np.nan,
        }
        for w, pts in enumerate(points, start=1)
    ]


# =============================================================== the label is the summed weekly total
def test_season_label_is_the_summed_engine_scored_weekly_total():
    """Acceptance #5: the season label is a sum of the weekly ``y_custom_points``, no coefficient here."""
    weekly = _weekly(
        _player_season("a", 2016, "RB", "NYG", [1.0, 2.0, 3.0, 4.0])  # sum 10
        + _player_season("a", 2017, "RB", "NYG", [5.0, 5.0, 5.0, 5.0])  # sum 20
    )
    frame = season_frame_from_weekly(weekly)
    row17 = frame[(frame["player_id"] == "a") & (frame["season"] == 2017)].iloc[0]
    assert row17[LABEL_COL] == pytest.approx(20.0)
    assert row17["prior_points_total"] == pytest.approx(10.0)  # 2016 total, the prior season
    assert row17["prior_games"] == pytest.approx(4.0)
    assert row17["prior_points_per_game"] == pytest.approx(2.5)


# =============================================================== fail-closed: only ≤ S-1 feeds a feature
def test_spiking_season_s_points_moves_the_label_only_never_a_feature():
    """The centrepiece: a wild season-S outcome must not reach any feature of the season-S row."""
    base = _player_season("a", 2016, "RB", "NYG", [3.0, 3.0, 3.0, 3.0]) + _player_season(
        "a", 2017, "RB", "NYG", [4.0, 4.0, 4.0, 4.0]
    )
    calm = season_frame_from_weekly(_weekly(base))
    spiked_rows = _weekly(
        _player_season("a", 2016, "RB", "NYG", [3.0, 3.0, 3.0, 3.0])
        + _player_season("a", 2017, "RB", "NYG", [999.0, 999.0, 999.0, 999.0])
    )
    spiked = season_frame_from_weekly(spiked_rows)

    calm17 = calm[calm["season"] == 2017].iloc[0]
    spiked17 = spiked[spiked["season"] == 2017].iloc[0]
    assert spiked17[LABEL_COL] != pytest.approx(calm17[LABEL_COL])  # the label moved
    for feature in SEASON_FEATURES:
        spiked_val, calm_val = spiked17[feature], calm17[feature]
        if pd.isna(spiked_val) or pd.isna(calm_val):
            assert pd.isna(spiked_val) and pd.isna(calm_val), feature  # both stayed absent
        else:
            assert float(spiked_val) == pytest.approx(float(calm_val)), feature  # no feature moved


def test_changed_team_prior_reads_s1_vs_s2_and_ignores_a_move_into_s():
    """The season-S team is withheld (acceptance #1): a move *into* S produces no team-change signal.

    ``b`` stays on NYG through 2016-2017 and moves to PHI *in* 2018; his 2018 change flag must be 0,
    proving the 2018 (season-S) team is never read. ``c`` moved NYG→PHI between 2016 and 2017, so his
    2018 flag is 1 — a genuine ≤ S-1 change.
    """
    weekly = _weekly(
        _player_season("b", 2016, "WR", "NYG", [5.0])
        + _player_season("b", 2017, "WR", "NYG", [5.0])
        + _player_season("b", 2018, "WR", "PHI", [5.0])  # moved INTO 2018
        + _player_season("c", 2016, "WR", "NYG", [5.0])
        + _player_season("c", 2017, "WR", "PHI", [5.0])  # moved between 2016 and 2017
        + _player_season("c", 2018, "WR", "PHI", [5.0])
    )
    frame = season_frame_from_weekly(weekly)
    b18 = frame[(frame["player_id"] == "b") & (frame["season"] == 2018)].iloc[0]
    c18 = frame[(frame["player_id"] == "c") & (frame["season"] == 2018)].iloc[0]
    assert b18["changed_team_prior"] == pytest.approx(0.0)  # 2017 NYG == 2016 NYG, PHI-in-2018 unseen
    assert c18["changed_team_prior"] == pytest.approx(1.0)  # 2017 PHI != 2016 NYG

    b17 = frame[(frame["player_id"] == "b") & (frame["season"] == 2017)].iloc[0]
    assert pd.isna(b17["changed_team_prior"])  # no 2015 season → not enough prior to compare


# =============================================================== rookies are a handled cohort
def test_a_rookie_has_null_prior_features_and_is_flagged():
    """Acceptance #4: no prior season → every prior feature null, ``is_rookie`` set, no crash."""
    weekly = _weekly(
        _player_season("vet", 2016, "RB", "NYG", [8.0, 8.0])
        + _player_season("vet", 2017, "RB", "NYG", [9.0, 9.0])
        + _player_season("rook", 2017, "RB", "PHI", [20.0, 20.0])  # first appearance
    )
    frame = season_frame_from_weekly(weekly)
    rook = frame[frame["player_id"] == "rook"].iloc[0]
    vet17 = frame[(frame["player_id"] == "vet") & (frame["season"] == 2017)].iloc[0]
    assert bool(rook["is_rookie"]) and not bool(rook["has_prior_season"])
    assert pd.isna(rook["prior_points_total"])
    assert rook["career_seasons"] == pytest.approx(0.0)
    assert not bool(vet17["is_rookie"]) and bool(vet17["has_prior_season"])


def test_missing_usage_columns_do_not_break_the_aggregation():
    """A narrowed weekly build with no lagged-usage columns still assembles (features go null)."""
    weekly = pd.DataFrame(
        [
            {"player_id": "a", "season": s, "week": w, "position": "TE", "team": "NYG",
             "is_dst": False, "y_custom_points": 5.0}
            for s in (2016, 2017)
            for w in (1, 2)
        ]
    )
    frame = season_frame_from_weekly(weekly)
    row = frame[frame["season"] == 2017].iloc[0]
    assert row["prior_points_total"] == pytest.approx(10.0)
    assert pd.isna(row["prior_snap_share"])


# =============================================================== build_season_frame wiring
def test_build_season_frame_forwards_scoring_and_aggregates(monkeypatch):
    """``build_season_frame`` re-scores via the assembler (scoring passed through) then aggregates."""
    captured = {}

    def fake_build_training_frame(seasons, scoring, *, backend=None):
        captured["scoring"] = scoring
        captured["seasons"] = list(seasons)
        return _weekly(
            _player_season("a", 2016, "QB", "NYG", [10.0, 10.0])
            + _player_season("a", 2017, "QB", "NYG", [12.0, 12.0])
        )

    monkeypatch.setattr("model.frame.build_training_frame", fake_build_training_frame)
    frame = build_season_frame([2016, 2017], {"pass_td": 4.0}, backend=None)
    assert captured["scoring"] == {"pass_td": 4.0}
    assert frame[frame["season"] == 2017].iloc[0][LABEL_COL] == pytest.approx(24.0)


def test_build_season_frame_rejects_empty_scoring():
    with pytest.raises(ValueError, match="scoring is empty"):
        build_season_frame([2017], {})


# =============================================================== synthetic season frame for the model
_HIGH_SCALE = ("QB", "RB", "WR", "TE")  # positions whose season totals are large
#: Positions that carry a usage signal on the real lake — the four skill positions AND **K** (kickers
#: are on the field, so snap/target share are populated; profile #27 §5b shows K `snap_pct_last` only
#: 7.9% null). Only DEF carries no usage column at all, so DEF alone is deferred to the baseline.
_WITH_USAGE = ("QB", "RB", "WR", "TE", "K")
_ALL_POS = ("QB", "RB", "WR", "TE", "K", "DEF")


def _season_frame(
    *, seasons=range(2016, 2023), positions=_ALL_POS, n=60, seed=0
) -> pd.DataFrame:
    """A season frame with real within-position ranking signal, every position modelled honestly.

    Each player has a stable latent ``talent``. The season outcome tracks talent cleanly; the
    prior-points feature is a *noisy* view of it (so ranking by it alone is weak). Positions **with a
    usage signal** (:data:`_WITH_USAGE` — the skill positions and K) also carry *clean* prior snap and
    target share, views a model can exploit to out-rank the points-only baseline (acceptance #2). **DEF
    carries no usage column at all** (every usage feature null), mirroring the real lake (profile #27
    §5b): the model has no edge there and must defer to the baseline. The earliest season is the warm-up
    (no prior season in the window).
    """
    rng = np.random.default_rng(seed)
    seasons = list(seasons)
    first = min(seasons)
    talent = {(pos, p): float(rng.uniform(0.2, 1.0)) for pos in positions for p in range(n)}
    rows: list[dict] = []
    for pos in positions:
        big = pos in _HIGH_SCALE
        usage = pos in _WITH_USAGE
        for p in range(n):
            t = talent[(pos, p)]
            for s in seasons:
                y = (t * 230.0 if big else t * 125.0) + rng.normal(0, 12)
                row = {
                    "player_id": f"{pos}{p}", "season": s, "position": pos, "team": "AAA",
                    "is_dst": pos == "DEF", "is_warmup": s == first, LABEL_COL: y,
                    "prior_rush_share": np.nan, "changed_team_prior": np.nan,
                }
                if s == first:  # no prior season in the window → warm-up / rookie cohort
                    row.update(
                        is_rookie=True, has_prior_season=False, prior_points_total=np.nan,
                        prior_points_per_game=np.nan, prior_games=np.nan, prior_snap_share=np.nan,
                        prior_target_share=np.nan, prior_exp_points=np.nan,
                        career_games=0.0, career_seasons=0.0,
                    )
                    rows.append(row)
                    continue
                pts = (t * 220.0 if big else t * 120.0) + rng.normal(0, 60 if big else 22)
                row.update(
                    is_rookie=False, has_prior_season=True, prior_points_total=pts,
                    prior_points_per_game=pts / 16.0, prior_games=16.0,
                    prior_snap_share=float(np.clip(t + rng.normal(0, 0.05), 0, 1)) if usage else np.nan,
                    prior_target_share=(
                        float(np.clip(0.35 * t + rng.normal(0, 0.03), 0, 1)) if usage else np.nan
                    ),
                    prior_exp_points=(pts + rng.normal(0, 10)) if usage else np.nan,
                    career_games=16.0 * (s - first), career_seasons=float(s - first),
                    changed_team_prior=float(rng.integers(0, 2)),
                )
                rows.append(row)
    return pd.DataFrame(rows)


# =============================================================== beats the baseline on ordering
def test_season_model_beats_prior_season_total_on_within_position_ordering():
    """Acceptance #2, all six positions: strict ρ win where the model has usage, tie where it defers.

    The model orders every position **at least as well** as the prior-season-total baseline, and
    *strictly* better at the five positions that carry a usage signal (skill + K). DEF has none (profile
    #27 §5b), so the model defers to the baseline there — a tie, never a loss (the failure the review
    found: an ungated ridge orders DEF *worse* than last year's total).
    """
    frame = _season_frame()
    model = evaluate_season(SeasonModel(), frame, name="SeasonModel")
    base = evaluate_season(PriorSeasonTotal(), frame, name="PriorSeasonTotal")
    for pos in _WITH_USAGE:  # has usage → a real ordering edge
        m, b = model.per_position[pos].spearman, base.per_position[pos].spearman
        assert m is not None and b is not None
        assert m > b, f"{pos}: model ρ {m:.3f} did not beat baseline ρ {b:.3f}"
    for pos in ("DEF",):  # no usage → deferred to the baseline, so it ties (never worse)
        m, b = model.per_position[pos].spearman, base.per_position[pos].spearman
        assert m == pytest.approx(b), f"{pos}: model ρ {m} should equal the deferred baseline ρ {b}"


def test_model_defers_the_usageless_position_to_last_years_total():
    """Fix for acceptance #2 at DEF: it is ordered by the baseline, not a weaker ridge.

    K is *not* deferred — kickers carry snap/target share, so it fields a ridge like the skill positions
    (and, measured on the real lake, beats the bar). DEF alone has no usage column and defers.
    """
    frame = _season_frame()
    model = SeasonModel().fit(frame[frame["season"] < 2022])
    assert set(model.modeled_positions) == set(_WITH_USAGE)  # usage-carrying positions field a ridge
    assert "DEF" not in model.modeled_positions
    dst = frame[(frame["season"] == 2022) & (frame["position"] == "DEF")]
    # Every 2022 DEF has a prior season here, so deferring means predicting exactly last year's total.
    assert np.allclose(
        model.predict(dst).to_numpy(),
        pd.to_numeric(dst["prior_points_total"], errors="coerce").to_numpy(),
    )


def test_evaluation_is_per_position_and_carries_no_pooled_number():
    frame = _season_frame()
    res = evaluate_season(SeasonModel(), frame)
    assert set(res.per_position) <= set(_ALL_POS)
    for attr in ("mae", "spearman", "pooled", "overall"):
        assert not hasattr(res, attr)


def test_evaluate_season_trains_only_on_strictly_earlier_seasons():
    """The walk-forward leak gate is reused: no training row is from the test season or later."""
    seen: list[tuple[int, int]] = []  # (max train season, test season)

    class _Spy(SeasonModel):
        def fit(self, frame):
            seen.append((int(frame["season"].max()), -1))
            return super().fit(frame)

    frame = _season_frame()
    res = evaluate_season(_Spy(), frame, test_seasons=[2020, 2021, 2022])
    # walk_forward pairs each fit with the next test season, in order.
    for (max_train, _), test_season in zip(seen, res.test_seasons, strict=True):
        assert max_train < test_season


def test_fit_excludes_the_warmup_season_from_training():
    """Fix for the 100%-arithmetic `is_rookie`: the earliest built season must not reach the fit.

    Reverting the `is_warmup` exclusion in `SeasonModel.fit` puts 2016 — where every player reads rookie
    by arithmetic, not observation — back into the learned rookie level, and this fails.
    """
    frame = _season_frame(seasons=range(2016, 2019))  # 2016 warm-up, then 2017, 2018
    model = SeasonModel().fit(frame)
    assert 2016 not in model.fit_seasons  # the warm-up season did not train the model
    assert set(model.fit_seasons) == {2017, 2018}


def test_fit_emits_no_runtimewarning_on_all_null_usage():
    """Fix #3: `_safe_stats` must silence nanmean/nanstd's RuntimeWarning on an all-null usage column.

    `require_usage=False` forces a ridge onto K/DEF, whose usage columns are entirely null — the exact
    input that made numpy warn fourteen times on the real lake. Reverting the `warnings.catch_warnings`
    guard back to `np.errstate` (which does not govern these) makes them reappear, and this fails.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SeasonModel(require_usage=False).fit(_season_frame())
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime == [], [str(w.message) for w in runtime]


# =============================================================== rookie prediction is finite, not a leak
def test_model_predicts_a_finite_position_level_for_a_rookie():
    """Acceptance #4 at predict time: a rookie row scores near the position's learned level."""
    frame = _season_frame()
    train = frame[frame["season"] < 2022]
    model = SeasonModel().fit(train)
    rookies = frame[(frame["season"] == 2016) & (frame["position"] == "RB")]  # all-null features
    preds = model.predict(rookies)
    assert preds.notna().all()
    assert np.isfinite(preds.to_numpy()).all()
    # The rookie level should sit inside the range of realised RB season totals, not at an extreme.
    rb_totals = train[train["position"] == "RB"][LABEL_COL]
    assert rb_totals.min() <= preds.mean() <= rb_totals.max()


def test_prior_season_total_falls_back_to_position_mean_for_a_rookie():
    frame = _season_frame()
    base = PriorSeasonTotal().fit(frame[frame["season"] < 2020])
    rookie = frame[(frame["season"] == 2016) & (frame["position"] == "WR")].head(1)
    pred = base.predict(rookie).iloc[0]
    wr_mean = frame[(frame["season"] < 2020) & (frame["position"] == "WR")][LABEL_COL].mean()
    assert pred == pytest.approx(wr_mean, rel=0.05)


# =============================================================== feeds draft.vor unchanged
def test_season_board_drives_the_existing_vor_and_tier_machinery():
    """Acceptance #3: the model's board is the ``build_board`` shape, so ``draft.vor`` consumes it.

    All six draftable positions are present — K and DEF ranked by the deferred baseline — so the board
    exercises the full replacement/VOR/tier machinery, not just the modelled positions.
    """
    frame = _season_frame()
    model = SeasonModel().fit(frame[frame["season"] < 2022])
    board = season_value_board(model, frame, 2022)

    assert board and all(isinstance(row, PlayerRow) for row in board)
    assert {row.pos for row in board} == set(_ALL_POS)  # K/DEF made it onto the board via fallback
    assert board == sorted(board, key=lambda p: p.proj_pts, reverse=True)  # ranked best-first

    base = {"QB": 12, "RB": 24, "WR": 24, "TE": 12, "K": 12, "DEF": 12}
    replacement = replacement_levels(board, base, flex_slots=12)
    add_vor(board, replacement)
    tierize(board)
    assert any(row.vor != 0.0 for row in board)  # VOR was computed onto the rows
    assert all(row.tier >= 1 for row in board)  # every player landed in a tier


def test_to_player_rows_drops_null_predictions_and_undraftable_positions():
    frame = pd.DataFrame(
        [
            {"player_id": "a", "season": 2022, "position": "RB", "team": "NYG"},
            {"player_id": "b", "season": 2022, "position": "RB", "team": "PHI"},
            {"player_id": "lb", "season": 2022, "position": "LB", "team": "DAL"},  # not draftable
        ]
    )
    preds = pd.Series([10.0, np.nan, 5.0], index=frame.index)  # b has no prediction
    rows = to_player_rows(frame, preds)
    assert {r.player_id for r in rows} == {"a"}  # b dropped (null), lb dropped (position)
    assert rows[0].team == "NYG" and rows[0].adp == float("inf")


# =============================================================== the committed report follows its data
def _spm(pos: str, mae: float, rho: float | None, *, n: int = 100, slates: int = 8) -> SeasonPositionMetrics:
    return SeasonPositionMetrics(
        position=pos, n=n, mae=mae, rmse=mae * 1.3, spearman=rho, slates=(0 if rho is None else slates)
    )


def _sres(name: str, per: dict[str, tuple[float, float | None]], *, test_seasons=(2018, 2019)):
    return SeasonEvalResult(
        predictor=name,
        test_seasons=test_seasons,
        per_position={pos: _spm(pos, mae, rho) for pos, (mae, rho) in per.items()},
        predictions=pd.DataFrame(),
    )


def _render(results, *, rookie_share=None, records=()) -> str:
    return es.render_report(
        results,
        seasons=[2016, 2025],
        scoring_keys=42,
        partitions=1,
        league_name="Test league",
        generated="2026-07-31",
        frame_rows=16949,
        players=4603,
        rookie_share=rookie_share or {2018: 0.23, 2019: 0.22},
        records=records,
    )


def _finding(report: str, n: int) -> str:
    return next(line for line in report.splitlines() if line.startswith(f"{n}. **"))


def _standard_results(*, model_test_seasons=(2018, 2019)):
    """A bar / model / ungated triple where the model wins QB, ties (defers) DEF, ungated loses DEF."""
    return {
        es._BAR: _sres(es._BAR, {"QB": (75.6, 0.587), "DEF": (32.2, 0.296)}),
        es._MODEL: _sres(
            es._MODEL, {"QB": (69.5, 0.615), "DEF": (32.2, 0.296)}, test_seasons=model_test_seasons
        ),
        es._UNGATED: _sres(es._UNGATED, {"QB": (69.5, 0.615), "DEF": (27.0, 0.265)}),
    }


def test_report_header_states_the_seasons_actually_scored_not_a_default():
    report = _render(_standard_results(model_test_seasons=(2021, 2022)))
    header = next(ln for ln in report.splitlines() if ln.startswith("- **Train span"))
    assert "2021–2022" in header  # read back from the model's own test_seasons


def test_report_finding_1_names_the_won_positions_and_the_deferred_one_from_the_data():
    finding = _finding(_render(_standard_results()), 1)
    assert "QB ρ 0.615 vs 0.587" in finding  # a measured win, derived from the two tables
    assert "defers to the bar" in finding and "at DEF" in finding  # the tie, derived — not asserted


def test_report_finding_2_justifies_the_deferral_from_the_ungated_table():
    finding = _finding(_render(_standard_results()), 2)
    assert "Why DEF defers" in finding
    assert "DEF ρ 0.265 vs 0.296" in finding  # the ungated loss, the measured reason to defer


def test_report_headline_verdict_is_derived_win_and_tie():
    report = _render(_standard_results())
    section = report.split("## Model vs bar — the headline, per position")[1]
    qb_row = next(ln for ln in section.splitlines() if ln.strip().startswith("| QB |"))
    def_row = next(ln for ln in section.splitlines() if ln.strip().startswith("| DEF |"))
    assert "win" in qb_row
    assert "tie (deferred)" in def_row


def test_report_counts_warnings_from_the_captured_log_and_quotes_them():
    records = [(30, "WARNING", "model.evaluate", "no rows for test season 2018 — skipping it")]
    report = _render(_standard_results(), records=records)
    assert "emitted **1** WARNING-level" in _finding(report, 4)
    assert "no rows for test season 2018" in report.split("## Warnings (verbatim)")[1]


def test_report_says_zero_warnings_when_the_log_carried_none():
    report = _render(_standard_results())
    assert "Zero warning" in _finding(report, 4)
    assert "_None._" in report.split("## Warnings (verbatim)")[1]

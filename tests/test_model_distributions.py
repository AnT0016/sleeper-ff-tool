"""Offline unit tests for the fitted simulator distributions (Phase 9, ticket #32).

Mechanics only — the numbers live in ``docs/model-distributions.md`` and the committed
``src/model/fit/distributions.json``, never asserted here on an engineered fixture (spec Decision #9
item 1). These pin: the fitted-over-heuristic read path (the sims actually consume the artifact, and a
fallback never leaks or empties), the in-place ``use_knobs`` swap that all three surfaces must see, the
residual-CV and injury estimators, the season-factor coherence gate (trap 2, both ends) and the identity
holding on the shipped knobs, and that the report's prose follows its own tables.

Revert-checks (mutate the guarded line, the named test reddens):
* ``verdict == "fitted"`` guard in ``_merge_scalar`` → ``test_merge_scalar_keeps_fallback_value_out``
* in-place mutation in ``use_knobs`` (vs a rebind) → ``test_use_knobs_is_seen_by_winprob``
* ``SETBACK_MIN_WEEKS`` in ``_corroborated_setback`` → ``test_injury_table_qualification_and_setbacks``
* the coherence gate in ``_season_cv_verdict`` / ``build_verdicts`` → ``test_identity_holds_on_shipped_knobs``
  (an incoherent season CV would ship and the identity would no longer reconstruct it) and
  ``test_season_cv_verdict_coherence``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

import draftsim.distributions as dd
import optimizer.winprob as winprob
import seasonsim.distributions as ssd
from draftsim.distributions import (
    GAME_CV,
    HEURISTIC_GAME_CV,
    HEURISTIC_INJURY_RISK,
    HEURISTIC_POSITION_CV,
    INJURY_RISK,
    POSITION_CV,
    SEASON_GAMES,
    FITTED_ARTIFACT_PATH,
    is_fitted,
    sample_availability,
    use_knobs,
)
from draftsim.distributions import _merge_injury, _merge_scalar, _read_artifact
from model.distributions import (
    POSITIONS,
    SEASON_COHORT_BY_POSITION,
    CvCell,
    FitResult,
    InjuryCell,
    build_verdicts,
    drafted_cohort_keys,
    expected_availability,
    fit_cv_cells,
    fit_injuries,
    robustness_overall_cohort,
    _contiguous_runs,
    _corroborated_setback,
    _player_season_injury_table,
    _season_cv_verdict,
    _select_cohort,
)
from seasonsim.distributions import season_factor_cv

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli(name: str):
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


ed = _load_cli("eval_distributions")


# =============================================================== the fitted-over-heuristic read path
def test_merge_scalar_uses_fitted_over_heuristic():
    art = {"position_cv": {"QB": {"value": 0.5, "verdict": "fitted"}}}
    out = _merge_scalar(art, "position_cv", HEURISTIC_POSITION_CV)
    assert out["QB"] == 0.5  # fitted overrides
    assert set(out) == set(HEURISTIC_POSITION_CV)  # all six positions, never {}-empty
    assert out["RB"] == HEURISTIC_POSITION_CV["RB"]  # a position the artifact omits stays heuristic


def test_merge_scalar_keeps_fallback_value_out():
    # A cell can carry a value while its verdict is heuristic-fallback (the report records the residual
    # as evidence). That value must NOT ship. REVERT-CHECK for the `verdict == "fitted"` guard.
    art = {"position_cv": {"QB": {"value": 0.9, "verdict": "heuristic-fallback"}}}
    assert _merge_scalar(art, "position_cv", HEURISTIC_POSITION_CV)["QB"] == HEURISTIC_POSITION_CV["QB"]


def test_merge_scalar_missing_or_malformed_is_all_heuristic():
    assert _merge_scalar({}, "position_cv", HEURISTIC_POSITION_CV) == HEURISTIC_POSITION_CV
    assert _merge_scalar({"position_cv": "x"}, "game_cv", HEURISTIC_GAME_CV) == HEURISTIC_GAME_CV
    bad = {"position_cv": {"QB": {"value": -1.0, "verdict": "fitted"}}}  # non-positive is rejected
    assert _merge_scalar(bad, "position_cv", HEURISTIC_POSITION_CV)["QB"] == HEURISTIC_POSITION_CV["QB"]


def test_read_artifact_missing_or_bad_file_is_empty(tmp_path):
    assert _read_artifact(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _read_artifact(bad) == {}


def test_merge_injury_fitted_and_fallback():
    art = {
        "injury_risk": {
            "QB": {"p": 0.1, "games": 3.0, "verdict": "fitted"},
            "DEF": {"p": None, "games": None, "verdict": "heuristic-fallback"},
        }
    }
    out = _merge_injury(art, HEURISTIC_INJURY_RISK)
    assert out["QB"] == (0.1, 3.0)
    assert out["DEF"] == HEURISTIC_INJURY_RISK["DEF"]
    assert set(out) == set(HEURISTIC_INJURY_RISK)  # never partial


def test_season_cohort_is_roster_math():
    # The drafted-cohort denominator is roster structure, not data: ~2 QB / 4 RB / 5 WR / 1.5 TE / 1 K /
    # 1 DEF per team over 12 teams → sums near the 168 active spots (12 × 14).
    assert set(SEASON_COHORT_BY_POSITION) == set(POSITIONS)
    assert sum(SEASON_COHORT_BY_POSITION.values()) == 174


def test_module_knobs_are_read_from_the_committed_artifact():
    # The sims consume the artifact: the module dicts ARE the fitted-over-heuristic merge of it. No
    # write-only artifact (spec Decision #9 item 4).
    art = _read_artifact(FITTED_ARTIFACT_PATH)
    assert POSITION_CV == _merge_scalar(art, "position_cv", HEURISTIC_POSITION_CV)
    assert GAME_CV == _merge_scalar(art, "game_cv", HEURISTIC_GAME_CV)
    assert INJURY_RISK == _merge_injury(art, HEURISTIC_INJURY_RISK)
    assert set(POSITION_CV) == set(GAME_CV) == set(INJURY_RISK) == set(HEURISTIC_POSITION_CV)


# =============================================================== use_knobs — in place, all surfaces
def test_use_knobs_restores_and_stays_in_place():
    obj, before = dd.POSITION_CV, dict(dd.POSITION_CV)
    with use_knobs(position_cv={p: 0.11 for p in HEURISTIC_POSITION_CV}):
        assert dd.POSITION_CV["QB"] == 0.11
        assert dd.POSITION_CV is obj  # mutated in place, not rebound
    assert dd.POSITION_CV is obj and dd.POSITION_CV == before


def test_use_knobs_is_seen_by_winprob():
    # optimizer.winprob binds the GAME_CV *object* (WEEKLY_CV = GAME_CV); seasonsim re-exports it. A
    # runtime swap must mutate in place or these diverge silently. REVERT-CHECK for the in-place mutation.
    assert winprob.WEEKLY_CV is dd.GAME_CV and ssd.GAME_CV is dd.GAME_CV
    restored = dd.GAME_CV["WR"]
    with use_knobs(game_cv={p: 0.99 for p in HEURISTIC_GAME_CV}):
        assert winprob.WEEKLY_CV["WR"] == 0.99  # winprob sees the swap
        assert ssd.GAME_CV["WR"] == 0.99  # so does seasonsim
    assert winprob.WEEKLY_CV["WR"] == restored


def test_use_knobs_leaves_unpassed_knobs_untouched():
    before_game = dict(dd.GAME_CV)
    with use_knobs(position_cv={p: 0.1 for p in HEURISTIC_POSITION_CV}):
        assert dd.GAME_CV == before_game  # game_cv=None → untouched


# =============================================================== residual CV estimator
def test_fit_cv_recovers_a_known_lognormal_cv():
    rng = np.random.default_rng(0)
    n, cv_true = 6000, 0.30
    pred = np.full(n, 100.0)
    sigma = np.sqrt(np.log1p(cv_true**2))
    actual = pred * np.exp(-0.5 * sigma**2 + sigma * rng.standard_normal(n))
    df = pd.DataFrame(
        {"player_id": [f"p{i}" for i in range(n)], "season": 2020, "position": "WR",
         "actual": actual, "pred": pred}
    )
    cell = fit_cv_cells(df, grain="game", floor=1.0)["WR"]
    assert abs(cell.cv - cv_true) < 0.03  # mechanics: recovers the injected CV


def test_healthy_cohort_drops_setback_seasons():
    # Nine calm WR-seasons + one wild "setback" season; healthy (excluding the setback) is tighter.
    ids = [f"p{i}" for i in range(9)] + ["wild"]
    actual = [100.0] * 9 + [300.0]
    df = pd.DataFrame(
        {"player_id": ids, "season": 2020, "position": "WR", "actual": actual, "pred": 100.0}
    )
    cell = fit_cv_cells(
        df, grain="season", top_n_by_position={p: 100 for p in POSITIONS}, setback_keys={("wild", 2020)}
    )["WR"]
    assert cell.cv == 0.0  # the nine calm seasons alone → zero dispersion
    assert cell.full_cohort_cv is not None and cell.full_cohort_cv > cell.cv  # the tail is in `full`


def test_select_cohort_takes_top_n_per_season():
    df = pd.DataFrame(
        {"season": [2020] * 4 + [2021] * 4, "pred": [1.0, 9.0, 5.0, 7.0, 2.0, 8.0, 6.0, 4.0]}
    )
    top2 = _select_cohort(df, floor=None, top_n=2)
    assert sorted(top2[top2.season == 2020]["pred"]) == [7.0, 9.0]  # per-season top-2 by pred
    assert sorted(top2[top2.season == 2021]["pred"]) == [6.0, 8.0]
    assert set(_select_cohort(df, floor=5.0, top_n=None)["pred"]) == {9.0, 5.0, 7.0, 8.0, 6.0}  # floor


# =============================================================== injuries
def test_contiguous_runs():
    assert _contiguous_runs([1, 2, 3, 7, 8, 10]) == [3, 2, 1]
    assert _contiguous_runs([]) == []
    assert _contiguous_runs([5, 5, 6]) == [2]  # dupes collapse


def test_corroborated_setback_excludes_byes_catches_season_enders():
    # a 1-week gap (a bye), no report → not a setback.
    assert _corroborated_setback(frozenset({1, 2, 3, 5, 6, 7}), frozenset(), 1, 7) == 0
    # a season-ender: played 1-4, IR from week 5 (no Out rows after), but Out at week 5 (pre-IR) → caught.
    assert _corroborated_setback(frozenset({1, 2, 3, 4}), frozenset({5}), 1, 12) == 8
    # a multi-week absence with NO injury report → clean benching/cut, not a setback.
    assert _corroborated_setback(frozenset({1, 2, 3, 8, 9}), frozenset(), 1, 9) == 0


def test_injury_table_qualification_and_setbacks():
    played = pd.DataFrame(
        {
            "player_id": ["A"] * 8 + ["B"] * 10 + ["C"] * 4,
            "gsis_id": ["ga"] * 8 + ["gb"] * 10 + ["gc"] * 4,
            "position": ["RB"] * 22,
            "season": [2020] * 22,
            "week": [1, 2, 3, 4, 8, 9, 10, 11]  # A: injury-absent 5-7 (played before & after)
            + [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # B: absent only week 9
            + [8, 9, 10, 11],  # C: first seen week 8 → unqualified
        }
    )
    injuries = pd.DataFrame(
        {
            "gsis_id": ["ga", "ga", "ga", "gb"],
            "season": [2020] * 4,
            "week": [5, 6, 7, 9],
            "report_status": ["Out", "Out", "Out", "Out"],
        }
    )
    tab = _player_season_injury_table(played, injuries)

    def row(g):
        return tab[tab["gsis_id"] == g].iloc[0]

    # A: a 3-week injury-corroborated absence → setback.
    assert bool(row("ga").qualified) and bool(row("ga").setback) and int(row("ga").longest_setback) == 3
    # B: a single corroborated absence week → NOT a setback. REVERT-CHECK for `SETBACK_MIN_WEEKS`.
    assert bool(row("gb").qualified) and not bool(row("gb").setback)
    assert not bool(row("gc").qualified)  # first seen week 8 (> 4) → not a real opportunity

    cells, setback_keys, _ = fit_injuries(played, injuries)
    assert ("A", 2020) in setback_keys and ("B", 2020) not in setback_keys
    assert cells["DEF"].reason is not None and "defense" in cells["DEF"].reason  # DST → heuristic


def test_drafted_cohort_keys_takes_top_n_by_projection():
    df = pd.DataFrame(
        {
            "player_id": [f"q{i}" for i in range(5)] + [f"w{i}" for i in range(5)],
            "season": [2020] * 10,
            "position": ["QB"] * 5 + ["WR"] * 5,
            "pred": [10, 9, 8, 7, 6, 100, 90, 80, 70, 60],
        }
    )
    keys = drafted_cohort_keys(df, top_n_by_position={"QB": 2, "WR": 3, "RB": 0, "TE": 0, "K": 0, "DEF": 0})
    assert {("q0", 2020), ("q1", 2020)} <= keys and ("q2", 2020) not in keys  # top-2 QB by pred
    assert {("w0", 2020), ("w2", 2020)} <= keys and ("w3", 2020) not in keys  # top-3 WR by pred


def test_fit_injuries_uses_drafted_cohort_and_reports_wide():
    # Two qualified RBs; A has a corroborated setback, B does not. Draft only A (Amendment C).
    played = pd.DataFrame(
        {
            "player_id": ["A"] * 8 + ["B"] * 8,
            "gsis_id": ["ga"] * 8 + ["gb"] * 8,
            "position": ["RB"] * 16,
            "season": [2020] * 16,
            "week": [1, 2, 3, 4, 8, 9, 10, 11] + [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    injuries = pd.DataFrame(
        {"gsis_id": ["ga"] * 3, "season": [2020] * 3, "week": [5, 6, 7], "report_status": ["Out"] * 3}
    )
    cells, _, _ = fit_injuries(played, injuries, drafted_keys={("A", 2020)})
    rb = cells["RB"]
    assert rb.n_qualified == 1 and rb.n_wide == 2  # drafted = {A}; wide = {A, B}
    assert rb.p_wide == 0.5  # wide cohort: one of two is a setback
    assert rb.reason is not None and "drafted-cohort" in rb.reason  # n=1 < MIN_INJURY_SEASONS → defers


def test_robustness_overall_cohort_produces_a_verdict_per_position():
    rng = np.random.default_rng(0)
    rows = [
        {"player_id": f"{pos}{i}", "season": s, "position": pos, "pred": base - i,
         "actual": (base - i) * (1.0 + 0.2 * rng.standard_normal())}
        for s in range(2018, 2026)
        for pos, base in (("QB", 300), ("WR", 180))
        for i in range(30)
    ]
    out = robustness_overall_cohort(pd.DataFrame(rows), None, {"QB": 0.6, "WR": 0.84}, n_overall=20)
    assert set(out["position"]) == set(POSITIONS)  # a row for every position
    assert set(out["verdict"]) <= {"fitted", "heuristic-fallback"}
    assert out.loc[out.position == "QB", "n"].iat[0] > 0  # QB (highest pred) dominates the top-20 cut


# =============================================================== availability & coherence
def test_expected_availability_reproduces_the_sampler():
    rng = np.random.default_rng(0)
    mult, _ = sample_availability(rng, np.array([0.45]), np.array([4.0]), 200_000)
    assert abs(expected_availability({"RB": (0.45, 4.0)})["RB"] - float(mult.mean())) < 0.005


def _cv_cell(pos, cv, *, grain="season", n=500):
    empty = pd.DataFrame()
    extra = cv if grain == "season" else None  # full_cohort_cv, upper_bound_cv
    return CvCell(pos, grain, cv, 1.0, n, 8, extra, extra, empty, empty)


def test_season_cv_verdict_coherence():
    # coherent: a season CV well below the game CV → fitted.
    assert _season_cv_verdict(_cv_cell("WR", 0.30), 0.80, 17)[1] == "fitted"
    # incoherent: season CV >= game CV → implied factor > cap → fallback.
    v_hi = _season_cv_verdict(_cv_cell("WR", 1.0), 0.60, 17)
    assert v_hi[1] == "heuristic-fallback" and "not coherent" in v_hi[3]
    # collapse: a tiny season CV → single-game noise alone exceeds it → factor floors → fallback.
    v_lo = _season_cv_verdict(_cv_cell("WR", 0.05), 0.80, 17)
    assert v_lo[1] == "heuristic-fallback" and "collapse" in v_lo[3]


def test_build_verdicts_falls_back_incoherent_season_cv():
    pos = {p: _cv_cell(p, 1.0) for p in POSITIONS}  # every season CV incoherent with...
    game = {p: _cv_cell(p, 0.6, grain="game", n=5000) for p in POSITIONS}  # ...a 0.6 game CV
    inj = {p: InjuryCell(p, 0.1, 3.0, 500, 60, 400, None) for p in POSITIONS}
    v = build_verdicts(pos, game, inj)
    assert all(v["position_cv"][p]["verdict"] == "heuristic-fallback" for p in POSITIONS)
    assert all(v["game_cv"][p]["verdict"] == "fitted" for p in POSITIONS)
    assert v["shipped_position_cv"]["QB"] == HEURISTIC_POSITION_CV["QB"]  # ships the coherent heuristic


def test_identity_holds_on_shipped_knobs():
    # Acceptance criterion: `1 + CV_total² = (1 + CV_factor²)(1 + CV_week²/W)` holds against the SHIPPED
    # knobs, and no shipped pair collapses. Also the coherence-gate revert-check: were an incoherent
    # season CV shipped, the reconstruction below would not return it.
    for p in HEURISTIC_POSITION_CV:
        factor = float(season_factor_cv(np.array([POSITION_CV[p]]), np.array([GAME_CV[p]]), SEASON_GAMES)[0])
        assert factor > 0.0
        recon = float(np.sqrt((1.0 + factor**2) * (1.0 + GAME_CV[p] ** 2 / SEASON_GAMES) - 1.0))
        assert abs(recon - POSITION_CV[p]) < 1e-9


def test_committed_artifact_ships_a_coherent_configuration():
    # The shipped configuration is safe: every position has all three knobs, DEF injury is a fallback
    # (never on the injury report), and no position ships a collapsing season CV.
    assert not is_fitted("injury_risk", "DEF")
    for p in HEURISTIC_POSITION_CV:
        assert 0.0 < POSITION_CV[p] and 0.0 < GAME_CV[p]
        assert 0.0 <= INJURY_RISK[p][0] <= 1.0 and INJURY_RISK[p][1] > 0.0


# =============================================================== report prose follows its tables
def _fake_result(*, before_after):
    empty = pd.DataFrame()
    pos_cells = {p: _cv_cell(p, 0.3) for p in POSITIONS}
    game_cells = {p: _cv_cell(p, 0.7, grain="game", n=5000) for p in POSITIONS}
    inj_cells = {p: InjuryCell(p, 0.1, 2.5, 500, 60, 400, None) for p in POSITIONS}
    # verdicts: all game fitted; all season fallback; injury fitted except DEF → 6 + 0 + 5 = 11 fitted.
    pos_out = {p: {"value": 0.3, "verdict": "heuristic-fallback", "n": 500, "reason": "x"} for p in POSITIONS}
    game_out = {p: {"value": 0.7, "verdict": "fitted", "n": 5000, "reason": None} for p in POSITIONS}
    inj_out = {p: {"p": 0.1, "games": 2.5, "verdict": "fitted", "n": 500, "reason": None} for p in POSITIONS}
    inj_out["DEF"] = {"p": None, "games": None, "verdict": "heuristic-fallback", "n": 0, "reason": "team defense"}
    coherence = pd.DataFrame(
        [{"position": p, "fitted_season_cv": 0.3, "game_cv": 0.7, "fitted_factor_cv": 0.2,
          "verdict": "heuristic-fallback", "shipped_season_cv": HEURISTIC_POSITION_CV[p],
          "shipped_factor_cv": 0.2} for p in POSITIONS]
    )
    robustness = pd.DataFrame(
        [{"position": p, "cv": 0.3, "n": 200, "factor_cv": 0.2, "verdict": "fitted"} for p in POSITIONS]
    )
    return FitResult(
        seasons=(2016, 2025), test_seasons=(2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
        position_cv=pos_cells, game_cv=game_cells, injury=inj_cells,
        position_cv_out=pos_out, game_cv_out=game_out, injury_out=inj_out,
        coherence=coherence, robustness=robustness,
        shipped_position_cv=dict(HEURISTIC_POSITION_CV), shipped_game_cv={p: 0.7 for p in POSITIONS},
        shipped_injury={p: (0.1, 2.5) for p in POSITIONS},
        avail_heuristic={p: 0.95 for p in POSITIONS}, avail_fitted={p: 0.98 for p in POSITIONS},
        injury_table=empty, n_frame_rows=1000, n_players=100,
    )


def test_render_report_headcount_follows_the_verdicts():
    res = _fake_result(before_after={})
    ba = {"heuristic": 0.10, "cv_only": 0.09, "injury_only": 0.11, "both": 0.11}
    report = ed.render_report(
        res, ba, seasons=[2016, 2025], league_name="L", scoring_keys=42, partitions=1,
        generated="2026-08-01", n_sims=4000, records=(),
    )
    assert "11 of 18 (position × knob) cells fitted, 7 fell back" in report  # counted from the verdicts
    assert "+1.00 pts" in report and "-1.00 pts" in report  # injury_only + / cv_only − deltas
    assert "Zero warnings on real data" in report


def test_render_report_counts_warnings():
    res = _fake_result(before_after={})
    ba = {"heuristic": 0.1, "cv_only": 0.1, "injury_only": 0.1, "both": 0.1}
    recs = [(40, "WARNING", "x", "a skipped season")]
    report = ed.render_report(
        res, ba, seasons=[2016, 2025], league_name="L", scoring_keys=42, partitions=1,
        generated="2026-08-01", n_sims=4000, records=recs,
    )
    assert "1 warning on real data" in report and "a skipped season" in report

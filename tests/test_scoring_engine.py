"""Synthetic unit tests for the pure scoring engine (no network, no real data)."""

from __future__ import annotations

import pytest

from scoring.engine import points


def test_empty_inputs():
    assert points({}, {}) == 0.0
    assert points({"rush_yd": 100}, {}) == 0.0
    assert points({}, {"rush_yd": 0.1}) == 0.0


def test_missing_stat_keys_score_zero():
    # Scoring keys absent from the stat line contribute nothing.
    assert points({"rush_yd": 50.0}, {"rush_yd": 0.1, "rec_td": 6.0}) == pytest.approx(5.0)


def test_unscored_stat_keys_ignored():
    # Stat keys absent from scoring_settings contribute nothing (e.g. raw counting stats).
    stats = {"rush_yd": 50.0, "rush_att": 12, "target_share": 0.3}
    assert points(stats, {"rush_yd": 0.1}) == pytest.approx(5.0)


def test_basic_half_ppr_skill_line():
    stats = {"rec": 6, "rec_yd": 80, "rec_td": 1, "rush_yd": 10}
    scoring = {"rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "rush_yd": 0.1}
    # 3 + 8 + 6 + 1
    assert points(stats, scoring) == pytest.approx(18.0)


def test_negative_components():
    stats = {"pass_int": 2, "fum_lost": 1, "pass_td": 3}
    scoring = {"pass_int": -1.0, "fum_lost": -2.0, "pass_td": 4.0}
    # -2 - 2 + 12
    assert points(stats, scoring) == pytest.approx(8.0)


def test_fg_distance_buckets_are_summed_not_double_counted():
    # A kicker who made one 45-yarder: exactly one bucket flagged. The redundant total `fgm`
    # Sleeper also reports must NOT score, because our settings don't include `fgm`.
    stats = {"fgm": 1, "fgm_40_49": 1, "xpm": 3}
    scoring = {"fgm_0_19": 3.0, "fgm_40_49": 4.0, "fgm_50p": 5.0, "xpm": 1.0}
    assert points(stats, scoring) == pytest.approx(4.0 + 3.0)


def test_points_allowed_single_bucket_flag():
    # Defense allowing 17 points: only pts_allow_14_20 is flagged (==1); others absent.
    stats = {"pts_allow": 17, "pts_allow_14_20": 1, "sack": 3, "int": 1}
    scoring = {
        "pts_allow_0": 10.0,
        "pts_allow_14_20": 1.0,
        "pts_allow_35p": -4.0,
        "sack": 1.0,
        "int": 2.0,
    }
    # 1 (bucket) + 3 (sacks) + 2 (int);  raw `pts_allow` total is not a scoring key -> ignored
    assert points(stats, scoring) == pytest.approx(1.0 + 3.0 + 2.0)


def test_zero_weight_keys_have_no_effect():
    # `fum` carries a 0.0 coefficient in our league -> contributes nothing.
    assert points({"fum": 5}, {"fum": 0.0, "fum_lost": -2.0}) == 0.0

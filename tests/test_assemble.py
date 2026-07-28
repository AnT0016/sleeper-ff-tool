"""Offline unit tests for the dataset assembler (``dataset.assemble``).

Everything runs against a synthetic lake written into ``tmp_path`` with the real
``store.write_snapshot`` — so the partitions the assembler reads have genuine provenance columns,
genuine point-in-time dedup and the real registry keys. No network, and the repo's populated
``data_cache/lake/`` is never touched. The label test additionally reads the committed
``tests/fixtures/nflverse/player_week_2024.parquet``, so the re-scoring is pinned against real
provider rows rather than a hand-written idealisation of them.

The test that matters most is the **leak gate**, and it is written from both directions: the same
projection row, captured once before the week's lock and once after, must produce a baseline in the
first case and nothing in the second. Everything else in this phase is recoverable; a contaminated
training frame is not, because the contamination is invisible in the output.

Several tests here are regressions for defects that only appeared on the real 412-partition lake and
were silent in every sense that matters — a frame was still produced, with plausible numbers:

* a registry key is unique *within a partition*, so deduping a ten-season read on it alone collapsed
  one player's 2016 and 2017 week 1 into a single row (35k rows -> 25.6k on two seasons);
* ``nflverse_schedules`` spells 2016 teams ``SD``/``OAK`` while ``nflverse_player_week`` spells them
  ``LAC``/``LV``, so those players silently lost their market and weather features;
* ffverse stores ``sleeper_id`` as a float, so the crosswalk produced ``"13269.0"`` and the baseline
  column would have joined to nothing in 2026;
* Sleeper emits DEF stat lines for games that were never played, worth a tidy 10.0 points.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from dataset.assemble import (
    _admissible,
    build_training_frame,
    lookahead_ok,
    observed_weather_ok,
)
from store.lake import LocalParquetBackend, read_snapshot, write_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "nflverse"

SEASON = 2026
WEEK = 2

#: Week 2's first game is a Thursday-night kickoff: 2026-09-16 20:15 ET == 2026-09-17 00:15 UTC.
LOCK = pd.Timestamp("2026-09-17T00:15:00Z")
BEFORE_LOCK = "2026-09-16T12:00:00+00:00"
AFTER_LOCK = "2026-09-17T12:00:00+00:00"
#: What the one-time historical pull stamps on every 2016-2025 row: one instant, long after the fact.
BACKFILL_RUN = "2026-07-26T17:09:44+00:00"

GSIS = "00-0000001"
SLEEPER = "9001"

#: Half-PPR-shaped scoring, enough to make both the label and the baseline non-trivial.
SCORING = {"rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "rush_yd": 0.1, "rush_td": 6.0}

#: The league's reference scoring from CLAUDE.md — the non-standard bits (4-point passing TD,
#: half-PPR, distance-bucketed kicking) are exactly what the label test needs to discriminate.
LEAGUE_SCORING = {
    "pass_yd": 0.04, "pass_td": 4.0, "pass_2pt": 2.0, "pass_int": -1.0,
    "rush_yd": 0.1, "rush_td": 6.0, "rush_2pt": 2.0,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6.0, "rec_2pt": 2.0,
    "fum_lost": -2.0, "fum_rec_td": 6.0,
    "fgm_0_19": 3.0, "fgm_20_29": 3.0, "fgm_30_39": 3.0, "fgm_40_49": 4.0, "fgm_50p": 5.0,
    "xpm": 1.0, "fgmiss": -1.0, "xpmiss": -1.0, "st_td": 6.0,
}


@pytest.fixture()
def backend(tmp_path):
    return LocalParquetBackend(root=tmp_path)


# --------------------------------------------------------------------------- lake builders
def _write(backend, source, season, rows, *, captured_at, week=None, key_cols, backfill=True):
    return write_snapshot(
        source,
        season,
        [{**row, "_backfill": backfill} for row in rows],
        captured_at=captured_at,
        week=week,
        key_cols=key_cols,
        backend=backend,
    )


def _game(season, week, away, home, gameday, gametime="13:00"):
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season,
        "game_type": "REG",
        "week": week,
        "gameday": gameday,
        "gametime": gametime,
        "away_team": away,
        "home_team": home,
    }


def _calendar(backend, games=None, *, season=SEASON, captured_at=BEFORE_LOCK):
    """Week 2 by default: a Thursday nighter (which sets the lock) and a Sunday afternoon game."""
    if games is None:
        games = [
            _game(SEASON, WEEK, "KC", "LV", "2026-09-16", "20:15"),
            _game(SEASON, WEEK, "SF", "SEA", "2026-09-20"),
        ]
    _write(backend, "nflverse_schedules", season, games, captured_at=captured_at,
           key_cols=("game_id",))


def _crosswalk(backend, entries=None, *, season=SEASON):
    """ffverse's master. ``sleeper_id`` is a **float** here because that is how ffverse stores it."""
    entries = entries or [(GSIS, float(SLEEPER), "SmitJo00")]
    rows = [
        {"mfl_id": str(i), "gsis_id": gsis, "sleeper_id": sleeper, "pfr_id": pfr}
        for i, (gsis, sleeper, pfr) in enumerate(entries, start=1)
    ]
    _write(backend, "id_crosswalk", season, rows, captured_at=BEFORE_LOCK, key_cols=("mfl_id",))


def _actual(gsis=GSIS, *, season=SEASON, week=WEEK, team="KC", position="WR", **stats):
    return {
        "player_id": gsis,
        "season": season,
        "season_type": "REG",
        "week": week,
        "team": team,
        "position": position,
        **stats,
    }


def _actuals(backend, rows=None, *, season=SEASON, captured_at=AFTER_LOCK):
    """The label source. Captured after the fact, as post-game actuals always are."""
    if rows is None:
        # 5 receptions, 60 yards, 1 TD -> 2.5 + 6.0 + 6.0 == 14.5 under SCORING.
        rows = [_actual(receptions=5.0, receiving_yards=60.0, receiving_tds=1.0)]
    _write(backend, "nflverse_player_week", season, rows, captured_at=captured_at,
           key_cols=("player_id", "season_type", "week"))


def _projection(backend, *, captured_at, rec_yd, season=SEASON, week=WEEK, player_id=SLEEPER):
    _write(
        backend, "sleeper_proj_week", season,
        [{"player_id": player_id, "position": "WR", "team": "KC", "rec": 4.0, "rec_yd": rec_yd}],
        captured_at=captured_at, week=week, key_cols=("player_id",), backfill=False,
    )


def _base_lake(backend):
    _calendar(backend)
    _crosswalk(backend)
    _actuals(backend)


def _only(frame, player_id=SLEEPER, week=WEEK):
    rows = frame.loc[(frame["player_id"] == player_id) & (frame["week"] == week)]
    assert len(rows) == 1, f"expected exactly one row for {player_id} week {week}, got {len(rows)}"
    return rows.iloc[0]


# --------------------------------------------------------------------------- the scalar gate
def test_lookahead_ok_admits_a_capture_strictly_before_lock():
    assert lookahead_ok(WEEK, BEFORE_LOCK, WEEK, LOCK.isoformat())


def test_lookahead_ok_rejects_a_capture_after_lock():
    assert not lookahead_ok(WEEK, AFTER_LOCK, WEEK, LOCK.isoformat())


def test_lookahead_ok_rejects_a_capture_exactly_at_lock():
    # "strictly before" is the whole rule -- a row stamped at the lock instant is not pre-lock.
    assert not lookahead_ok(WEEK, LOCK.isoformat(), WEEK, LOCK.isoformat())


def test_lookahead_ok_admits_an_earlier_week_captured_after_lock():
    # A backfilled week-1 actual is legal as a week-2 feature however late it was captured.
    assert lookahead_ok(1, AFTER_LOCK, WEEK, LOCK.isoformat())


def test_lookahead_ok_rejects_a_later_week():
    assert not lookahead_ok(WEEK + 1, AFTER_LOCK, WEEK, LOCK.isoformat())


def test_lookahead_ok_fails_closed_on_an_unusable_stamp():
    assert not lookahead_ok(WEEK, None, WEEK, LOCK.isoformat())
    assert not lookahead_ok(None, "not-a-timestamp", WEEK, LOCK.isoformat())
    assert not lookahead_ok(None, BEFORE_LOCK, WEEK, None)


@pytest.mark.parametrize(
    ("feature_week", "known_at"),
    [
        (1, BEFORE_LOCK), (1, AFTER_LOCK), (2, BEFORE_LOCK), (2, AFTER_LOCK),
        (3, BEFORE_LOCK), (3, AFTER_LOCK), (None, BEFORE_LOCK), (None, AFTER_LOCK),
        (2, None), (None, None),
    ],
)
def test_the_vectorised_gate_agrees_with_the_scalar_one(feature_week, known_at):
    """The frame path and the documented rule must not be able to drift apart."""
    scalar = lookahead_ok(feature_week, known_at, WEEK, LOCK.isoformat())
    vector = _admissible(
        pd.Series([feature_week], dtype="Float64"),
        pd.to_datetime(pd.Series([known_at]), utc=True),
        pd.Series([WEEK], dtype="Float64"),
        pd.Series([LOCK]),
    )
    assert bool(vector.iloc[0]) is scalar


# --------------------------------------------------------------------------- the leak gate
def test_projection_captured_before_lock_is_included(backend):
    _base_lake(backend)
    _projection(backend, captured_at=BEFORE_LOCK, rec_yd=50.0)

    frame = build_training_frame([SEASON], SCORING, backend=backend)

    # 4 * 0.5 + 50 * 0.1 == 7.0
    assert _only(frame)["baseline_sleeper_points"] == pytest.approx(7.0)


def test_projection_captured_after_lock_is_excluded(backend):
    _base_lake(backend)
    _projection(backend, captured_at=AFTER_LOCK, rec_yd=50.0)

    frame = build_training_frame([SEASON], SCORING, backend=backend)

    assert pd.isna(_only(frame)["baseline_sleeper_points"])


def test_the_pre_lock_capture_wins_over_a_later_one_in_the_same_partition(backend):
    """The case the store is built for: both captures are kept, and only one is legal."""
    _base_lake(backend)
    _projection(backend, captured_at=BEFORE_LOCK, rec_yd=50.0)
    _projection(backend, captured_at=AFTER_LOCK, rec_yd=999.0)

    frame = build_training_frame([SEASON], SCORING, backend=backend)

    assert _only(frame)["baseline_sleeper_points"] == pytest.approx(7.0)


def test_lagged_usage_never_sees_the_target_week(backend):
    """``post_game`` content: a week-2 spike may inform week 3, never week 2 itself."""
    _calendar(
        backend,
        [
            _game(SEASON, 1, "KC", "LV", "2026-09-09"),
            _game(SEASON, 2, "KC", "LV", "2026-09-16"),
            _game(SEASON, 3, "KC", "LV", "2026-09-23"),
        ],
    )
    _crosswalk(backend)
    _actuals(
        backend,
        [
            _actual(week=1, receptions=2.0),
            _actual(week=2, receptions=100.0),  # the spike
            _actual(week=3, receptions=2.0),
        ],
    )

    frame = build_training_frame([SEASON], SCORING, backend=backend)

    assert _only(frame, week=1)["points_last"] != pytest.approx(50.0)
    assert pd.isna(_only(frame, week=1)["points_last"])
    assert _only(frame, week=2)["points_last"] == pytest.approx(1.0)  # week 1's 2 receptions
    assert _only(frame, week=3)["points_last"] == pytest.approx(50.0)  # the spike, one week later


def test_a_backfilled_pre_kickoff_row_is_admitted_for_its_own_week(backend):
    """The whole reason ``content_known`` exists.

    Every backfilled row is stamped with the 2026 backfill run, so a literal ``_captured_at < lock``
    rule would drop the week-2 betting line — the single most valuable feature — from all ten
    training seasons. ``vegas_odds`` is ``pre_kickoff``, so its content is resolved to its own week's
    lock instead.
    """
    _base_lake(backend)
    _write(
        backend, "vegas_odds", SEASON,
        [{"game_id": f"{SEASON}_{WEEK:02d}_KC_LV", "week": WEEK, "home_implied_total": 21.5,
          "away_implied_total": 24.5, "spread_line": -3.0, "total_line": 46.0, "div_game": 0}],
        captured_at=BACKFILL_RUN, key_cols=("game_id",), backfill=True,
    )

    row = _only(build_training_frame([SEASON], SCORING, backend=backend))

    assert row["implied_team_total"] == pytest.approx(24.5)  # KC is the away side
    assert row["opp_implied_total"] == pytest.approx(21.5)
    assert row["team_spread_line"] == pytest.approx(3.0)  # negated: nflverse quotes the home margin


def test_a_backfilled_post_game_row_is_not_admitted_for_its_own_week(backend):
    """The same resolution must not let actuals in. ``nflverse_snaps`` is ``post_game``."""
    _calendar(
        backend,
        [_game(SEASON, 1, "KC", "LV", "2026-09-09"), _game(SEASON, 2, "KC", "LV", "2026-09-16")],
    )
    _crosswalk(backend)
    _actuals(backend, [_actual(week=1, receptions=2.0), _actual(week=2, receptions=3.0)])
    _write(
        backend, "nflverse_snaps", SEASON,
        [
            {"pfr_player_id": "SmitJo00", "game_id": f"{SEASON}_01_KC_LV", "season": SEASON,
             "game_type": "REG", "week": 1, "offense_pct": 0.55},
            {"pfr_player_id": "SmitJo00", "game_id": f"{SEASON}_02_KC_LV", "season": SEASON,
             "game_type": "REG", "week": 2, "offense_pct": 0.95},
        ],
        captured_at=BACKFILL_RUN, key_cols=("pfr_player_id", "game_id"), backfill=True,
    )

    frame = build_training_frame([SEASON], SCORING, backend=backend)

    assert pd.isna(_only(frame, week=1)["snap_pct_last"])
    assert _only(frame, week=2)["snap_pct_last"] == pytest.approx(0.55)  # week 1's, not week 2's


# --------------------------------------------------------------------------- the label
def _fixture_actuals(backend):
    """The committed nflverse sample, minus the residual rows nflverse files with a null key."""
    frame = pl.read_parquet(FIXTURES / "player_week_2024.parquet")
    rows = [r for r in frame.to_dicts() if r.get("player_id") and r.get("season_type") == "REG"]
    _write(backend, "nflverse_player_week", 2024, rows, captured_at="2025-01-10T00:00:00+00:00",
           key_cols=("player_id", "season_type", "week"))
    return rows


def _fixture_lake(backend):
    rows = _fixture_actuals(backend)
    teams = sorted({r["team"] for r in rows})
    if len(teams) % 2:
        teams.append("ZZZ")  # an odd fixture would leave one team gameless, and its rows dropped
    pairs = [teams[i:i + 2] for i in range(0, len(teams), 2)]
    games = [
        _game(2024, week, away, home, f"2024-09-{6 + 7 * (week - 1):02d}")
        for week in (1, 2, 3)
        for away, home in pairs
    ]
    _calendar(backend, games, season=2024, captured_at="2024-08-01T00:00:00+00:00")
    _crosswalk(
        backend,
        [(r["player_id"], float(9000 + i), f"P{i:05d}") for i, r in enumerate(rows)],
        season=2024,
    )
    return {r["player_id"]: str(9000 + i) for i, r in enumerate(rows)}


@pytest.mark.parametrize(
    ("name", "gsis", "week", "expected"),
    [
        # A kicker: distance buckets are where public sheets misvalue K, and where the engine's
        # bucket handling has to be exact (Prater's week 2 is a 40-49, a 50+, and five XPs).
        ("Matt Prater", "00-0023853", 2, 14.0),
        # A QB: 4-point passing TDs, rushing, and a lost fumble in one line.
        ("Josh Allen", "00-0034857", 1, 31.18),
        # Half-PPR receiving plus rushing TDs.
        ("Saquon Barkley", "00-0034844", 1, 32.2),
        ("Ja'Marr Chase", "00-0036900", 3, 26.8),
    ],
)
def test_label_equals_the_engine_rescore_of_real_actuals(backend, name, gsis, week, expected):
    ids = _fixture_lake(backend)

    frame = build_training_frame([2024], LEAGUE_SCORING, backend=backend)
    row = _only(frame, ids[gsis], week=week)

    assert row["y_custom_points"] == pytest.approx(expected), name


def test_dst_labels_come_from_sleeper_because_nflverse_has_no_def_rows(backend):
    """``nflverse_player_week`` carries **zero** DEF rows, and DEF is a starting slot."""
    _base_lake(backend)
    _write(
        backend, "sleeper_stats_week", SEASON,
        [{"player_id": "LV", "position": "DEF", "season_type": "regular", "team": "LV",
          "sack": 4.0, "int": 2.0, "pts_allow_7_13": 1.0}],
        captured_at=AFTER_LOCK, week=WEEK, key_cols=("player_id",),
    )
    scoring = {**SCORING, "sack": 1.0, "int": 2.0, "pts_allow_7_13": 4.0}

    frame = build_training_frame([SEASON], scoring, backend=backend)
    row = _only(frame, "LV")

    assert row["is_dst"]
    assert row["position"] == "DEF"
    assert row["y_custom_points"] == pytest.approx(4.0 + 4.0 + 4.0)
    # DST position is not a fallback to a mutable master -- it follows from the row's own key.
    assert not row["position_is_static"]


def test_a_dst_row_for_a_game_that_was_never_played_is_dropped(backend):
    """Sleeper emits DEF lines for postponed/cancelled games, worth a tidy 10.0 by default."""
    _base_lake(backend)
    _write(
        backend, "sleeper_stats_week", SEASON,
        [{"player_id": "MIA", "position": "DEF", "season_type": "regular", "team": "MIA",
          "pts_allow_0": 1.0}],
        captured_at=AFTER_LOCK, week=WEEK, key_cols=("player_id",),
    )

    frame = build_training_frame([SEASON], {**SCORING, "pts_allow_0": 10.0}, backend=backend)

    assert "MIA" not in set(frame["player_id"])  # MIA has no game that week
    assert frame["game_id"].notna().all()


# --------------------------------------------------------------------------- grain & identity
def test_one_row_per_player_season_week_across_partitions(backend):
    """A registry key identifies a row *within a partition*; a multi-season read needs more.

    ``nflverse_player_week`` is keyed ``(player_id, season_type, week)``, so deduping a two-season
    read on that alone folded 2026 week 1 and 2027 week 1 into one row.
    """
    games = [_game(s, 1, "KC", "LV", f"{s}-09-10") for s in (SEASON, SEASON + 1)]
    for season in (SEASON, SEASON + 1):
        _calendar(backend, [g for g in games if g["season"] == season], season=season,
                  captured_at=BEFORE_LOCK)
        _actuals(backend, [_actual(season=season, week=1, receptions=float(season - 2000))],
                 season=season)
    _crosswalk(backend)

    frame = build_training_frame([SEASON, SEASON + 1], SCORING, backend=backend)

    assert not frame.duplicated(["player_id", "season", "week"]).any()
    assert len(frame) == 2
    assert sorted(frame["season"]) == [SEASON, SEASON + 1]


def test_sleeper_ids_are_canonical_strings_not_floats(backend):
    """ffverse stores ``sleeper_id`` as a float; ``"9001.0"`` joins to nothing Sleeper emits."""
    _base_lake(backend)
    _projection(backend, captured_at=BEFORE_LOCK, rec_yd=50.0)

    frame = build_training_frame([SEASON], SCORING, backend=backend)

    assert set(frame["player_id"]) == {SLEEPER}
    assert frame["baseline_sleeper_points"].notna().all()  # would be empty on a "9001.0" key


def test_legacy_and_modern_team_codes_land_on_one_calendar(backend):
    """``nflverse_schedules`` says ``SD``/``OAK`` for 2016; ``nflverse_player_week`` says ``LAC``/``LV``."""
    _calendar(backend, [_game(2016, 1, "SD", "OAK", "2016-09-11")], season=2016,
              captured_at="2016-08-01T00:00:00+00:00")
    _crosswalk(backend, season=2016)
    _actuals(backend, [_actual(season=2016, week=1, team="LAC", receptions=4.0)], season=2016)

    row = _only(build_training_frame([2016], SCORING, backend=backend), week=1)

    assert row["game_id"] == "2016_01_SD_OAK"
    assert row["team"] == "LAC"
    assert row["opponent"] == "LV"


def test_unjoined_rows_are_logged_and_dropped(backend, caplog):
    """The project rule: never drop a row that fails to join without saying so."""
    _calendar(backend)
    _crosswalk(backend)
    _actuals(
        backend,
        [_actual(receptions=5.0), _actual(gsis="00-0009999", team="SF", receptions=9.0)],
    )

    with caplog.at_level(logging.INFO, logger="dataset.assemble"):
        frame = build_training_frame([SEASON], SCORING, backend=backend)

    assert len(frame) == 1
    logged = [r.getMessage() for r in caplog.records]
    assert any("no sleeper_id" in m and "00-0009999" in m for m in logged), logged


# --------------------------------------------------------------------------- position
def test_position_resolves_as_of_the_depth_chart_before_lock(backend):
    """``nflverse_depth`` is the one registered source that is genuinely as-of a date."""
    _base_lake(backend)
    _write(
        backend, "nflverse_depth", SEASON,
        [
            {"dt": "2026-09-10T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": "QB",
             "pos_rank": 2},
            {"dt": "2026-09-15T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": "RB",
             "pos_rank": 1},
            # After the lock: must not be the one that wins.
            {"dt": "2026-09-18T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": "TE",
             "pos_rank": 1},
        ],
        captured_at=BACKFILL_RUN, key_cols=("dt", "team", "gsis_id", "pos_abb"),
    )

    row = _only(build_training_frame([SEASON], SCORING, backend=backend))

    assert row["position"] == "RB"  # the newest snapshot strictly before the lock
    assert not row["position_is_static"]
    assert row["depth_pos_rank"] == pytest.approx(1.0)


def test_the_dedup_policy_does_not_change_what_the_as_of_position_join_sees(tmp_path):
    """#15 decision 2, made executable. ``nflverse_depth`` resolves its as-of position from ``dt``
    (``content_known='row_timestamp'``) and dedups on ``(gsis_id, dt)`` — it never reads
    ``_captured_at``. So the pre-fix shape (the cumulative feed re-captured every pre-lock run, N
    rows per key) and the ``first_capture`` shape (one row per key) must produce the identical
    position, because an immutable ``dt``-keyed row carries the same payload in every capture.
    """
    depth_feed = [
        {"dt": "2026-09-10T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": "QB", "pos_rank": 2},
        {"dt": "2026-09-15T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": "RB", "pos_rank": 1},
    ]
    depth_keys = ("dt", "team", "gsis_id", "pos_abb")

    # Lake A: the registry default for depth (first_capture) -> one row per key.
    fixed = LocalParquetBackend(root=tmp_path / "first_capture")
    _base_lake(fixed)
    _write(fixed, "nflverse_depth", SEASON, depth_feed, captured_at=BACKFILL_RUN,
           key_cols=depth_keys)

    # Lake B: the #15 defect, reconstructed on purpose. The same feed captured on three dates under
    # the old per_capture_date policy (forced via the explicit override), so every key is stored 3x.
    multiplied = LocalParquetBackend(root=tmp_path / "per_capture_date")
    _base_lake(multiplied)
    for day in ("2026-09-11", "2026-09-14", "2026-09-18"):
        write_snapshot(
            "nflverse_depth", SEASON, [{**row, "_backfill": True} for row in depth_feed],
            captured_at=f"{day}T12:00:00+00:00", key_cols=depth_keys,
            dedup="per_capture_date", backend=multiplied,
        )

    # The two partitions really are different shapes — otherwise the test proves nothing.
    assert len(read_snapshot("nflverse_depth", SEASON, backend=fixed)) == 2
    assert len(read_snapshot("nflverse_depth", SEASON, backend=multiplied)) == 6

    fixed_row = _only(build_training_frame([SEASON], SCORING, backend=fixed))
    multiplied_row = _only(build_training_frame([SEASON], SCORING, backend=multiplied))

    for column in ("position", "position_is_static", "depth_pos_rank", "depth_dt"):
        assert fixed_row[column] == multiplied_row[column], column
    assert fixed_row["position"] == "RB"  # the newest snapshot strictly before the lock, either way


@pytest.mark.parametrize(
    ("starter", "backup"),
    [(("FB", 1), ("TE", 4)), (("TE", 2), ("QB", 3)), (("QB", 1), ("TE", 3))],
)
def test_a_player_listed_at_two_positions_resolves_to_the_one_he_starts_at(
    backend, starter, backup
):
    """``pos_rank`` 1 is the starter and higher numbers are deeper, so the *smallest* rank wins.

    Real shapes from the 2025 feed: 332 of 154,086 skill player-snapshots list a player twice, and
    they are two people — Taysom Hill (QB/TE) and Connor Heyward (FB/TE). Picking the deeper listing
    made Heyward's position oscillate TE(4) / FB(1) from week to week on a tie-break artifact rather
    than on his role, which is a categorical feature that lies about a role change.
    """
    _base_lake(backend)
    (start_pos, start_rank), (deep_pos, deep_rank) = starter, backup
    _write(
        backend, "nflverse_depth", SEASON,
        [
            # Same snapshot, same player, two listings — order reversed from rank on purpose, so a
            # tie-break that quietly follows provider row order fails this too.
            {"dt": "2026-09-15T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": deep_pos,
             "pos_rank": deep_rank},
            {"dt": "2026-09-15T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": start_pos,
             "pos_rank": start_rank},
        ],
        captured_at=BACKFILL_RUN, key_cols=("dt", "team", "gsis_id", "pos_abb"),
    )

    row = _only(build_training_frame([SEASON], SCORING, backend=backend))

    assert row["position"] == start_pos
    assert row["depth_pos_rank"] == pytest.approx(float(start_rank))
    assert not row["position_is_static"]


def test_an_unranked_listing_loses_to_a_ranked_one(backend):
    """A null ``pos_rank`` must not win by sorting after every real rank."""
    _base_lake(backend)
    _write(
        backend, "nflverse_depth", SEASON,
        [
            {"dt": "2026-09-15T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": "TE",
             "pos_rank": None},
            {"dt": "2026-09-15T07:00:00Z", "team": "KC", "gsis_id": GSIS, "pos_abb": "RB",
             "pos_rank": 2},
        ],
        captured_at=BACKFILL_RUN, key_cols=("dt", "team", "gsis_id", "pos_abb"),
    )

    assert _only(build_training_frame([SEASON], SCORING, backend=backend))["position"] == "RB"


def test_position_falls_back_to_the_static_label_under_a_flag(backend):
    """Depth starts at 2025, and the static label is anachronistic — so it is flagged, not hidden."""
    _base_lake(backend)

    row = _only(build_training_frame([SEASON], SCORING, backend=backend))

    assert row["position"] == "WR"  # the static nflverse label
    assert row["position_is_static"]


# --------------------------------------------------------------------------- weather
def test_observed_weather_is_rejected_for_its_own_week():
    kickoff = "2026-09-17T00:15:00Z"
    # The Sunday pre-lock capture, carrying the already-played Thursday game's real temperature.
    assert not observed_weather_ok("2026-09-20T15:00:00Z", kickoff, WEEK, WEEK)
    # A backfill row of a *past* week: legal, and the training case.
    assert observed_weather_ok(BACKFILL_RUN, kickoff, WEEK, WEEK + 1)
    # Captured before kickoff: not post-kickoff data (and holds no measurement anyway).
    assert observed_weather_ok("2026-09-16T12:00:00Z", kickoff, WEEK, WEEK)


def test_observed_weather_without_a_kickoff_falls_back_to_the_coarse_rule():
    assert not observed_weather_ok(BACKFILL_RUN, None, WEEK, WEEK)
    assert observed_weather_ok(BACKFILL_RUN, None, WEEK, WEEK + 1)


def test_the_sunday_capture_does_not_smuggle_thursdays_observed_weather_in(backend):
    """The exact partition shape from the review: two pre-lock captures, one carrying real weather."""
    _base_lake(backend)
    kickoff = "2026-09-17T00:15:00Z"
    game_id = f"{SEASON}_{WEEK:02d}_KC_LV"
    for captured, temp, wind in (
        (BEFORE_LOCK, None, None),          # Thursday pre-lock: nothing observed yet
        ("2026-09-20T15:00:00+00:00", 65.0, 7.0),  # Sunday pre-lock: the TNF game has been played
    ):
        _write(
            backend, "weather", SEASON,
            [{"game_id": game_id, "week": WEEK, "kickoff_utc": kickoff, "is_indoor": False,
              "weather_status": "unavailable", "forecast_time_utc": None, "forecast_temp_f": None,
              "forecast_wind_mph": None, "forecast_precip_prob_pct": None,
              "observed_temp_f": temp, "observed_wind_mph": wind}],
            captured_at=captured, week=WEEK, key_cols=("game_id",), backfill=False,
        )

    row = _only(build_training_frame([SEASON], SCORING, backend=backend))

    assert pd.isna(row["wx_observed_temp_f"])
    assert pd.isna(row["wx_observed_wind_mph"])
    # The venue still lands: it is the measurement that is withheld, not the whole row.
    assert not pd.isna(row["is_indoor"]) and not row["is_indoor"]


def test_is_indoor_stays_three_state(backend):
    """``<NA>`` is a retractable roof whose state is unrecorded — not the same claim as open air."""
    _base_lake(backend)
    _write(
        backend, "weather", SEASON,
        [{"game_id": f"{SEASON}_{WEEK:02d}_KC_LV", "week": WEEK,
          "kickoff_utc": "2026-09-17T00:15:00Z", "is_indoor": None, "weather_status": "forecast",
          "forecast_time_utc": "2026-09-17T00:00:00Z", "forecast_temp_f": 70.0,
          "forecast_wind_mph": 5.0, "forecast_precip_prob_pct": 10.0,
          "observed_temp_f": None, "observed_wind_mph": None}],
        captured_at=BEFORE_LOCK, week=WEEK, key_cols=("game_id",), backfill=False,
    )

    row = _only(build_training_frame([SEASON], SCORING, backend=backend))

    assert pd.isna(row["is_indoor"])
    assert row["is_indoor"] is not False
    assert row["wx_forecast_temp_f"] == pytest.approx(70.0)


# --------------------------------------------------------------------------- arguments
def test_a_usage_source_returning_duplicate_keys_aborts_rather_than_misaligning(
    backend, monkeypatch
):
    """The lag is built positionally, so a fanned-out merge shifts *other players'* history in.

    Both usage sources are deduped on the key, so this cannot happen today — which is exactly why
    the guard is worth having: the failure it prevents is silent, produces a full-looking frame, and
    would surface as a model that mysteriously underperforms.
    """
    _base_lake(backend)
    from dataset import assemble

    def duplicated(seasons, crosswalk, *, backend=None):
        return pd.DataFrame(
            {
                "player_id": [SLEEPER, SLEEPER],
                "season": [SEASON, SEASON],
                "week": [WEEK, WEEK],
                "exp_points": [1.0, 2.0],
                "rush_share": [0.1, 0.2],
            }
        )

    monkeypatch.setattr(assemble, "_opportunity", duplicated)
    with pytest.raises(AssertionError, match="duplicate"):
        build_training_frame([SEASON], SCORING, backend=backend)


def test_an_unsupported_asof_is_refused(backend):
    _base_lake(backend)
    with pytest.raises(ValueError, match="asof"):
        build_training_frame([SEASON], SCORING, asof="postgame", backend=backend)  # type: ignore[arg-type]


def test_empty_scoring_is_refused(backend):
    with pytest.raises(ValueError, match="scoring is empty"):
        build_training_frame([SEASON], {}, backend=backend)


def test_no_seasons_is_refused(backend):
    with pytest.raises(ValueError, match="no seasons"):
        build_training_frame([], SCORING, backend=backend)


def test_an_empty_lake_says_what_is_missing(backend):
    with pytest.raises(ValueError, match="nflverse_schedules is empty"):
        build_training_frame([SEASON], SCORING, backend=backend)

"""Offline unit tests for the Vegas collector (``collect.market``).

Everything runs against the parquet schedule samples in ``tests/fixtures/nflverse/`` — real nflverse
rows, kept as parquet so the provider's dtypes (``spread_line``/``total_line`` as ``Float64``,
moneylines as ``Int32``, both nullable) survive into the test rather than being tidied by a JSON
round-trip.

The tests that matter are the sign tests. ``home_implied + away_implied == total_line`` holds for
*either* sign convention, so on its own it proves nothing; what pins the convention is that the
side the market favours must end up with the **higher** implied total. Two independent anchors are
used for that, so a silent provider flip cannot pass:

* the **moneyline**, per game — whichever of ``home_moneyline``/``away_moneyline`` is more negative
  is the favourite, and that team must hold the larger implied total (all 16 fixture games);
* the **outcome**, in aggregate — nflverse's ``result`` is ``home_score - away_score``, so a
  positive ``spread_line`` must lean toward a positive ``result``.

(Both were also checked at full scale before the convention was written down: over the 2,761 played
games of 2016-2025, ``corr(spread_line, result)`` = +0.44, home wins 67.1% of ``spread_line > 0``
games, and the moneylines agree on the favourite in 99.2% of non-pick'em games.)
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from collect.market import collect_vegas_from_schedules, favored_team, implied_totals
from collect.registry import SOURCES
from store.lake import RESERVED, LocalParquetBackend, read_snapshot, write_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "nflverse"

SEASON = 2024
FORWARD_SEASON = 2026
CAPTURED_AT = "2026-07-26T12:00:00+00:00"

#: Post-game columns this source must never carry — it is what the market believed *before* kickoff.
OUTCOME_COLS = ("result", "total", "home_score", "away_score", "overtime")


def frame(name: str) -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / f"{name}.parquet")


@pytest.fixture(scope="module")
def schedules() -> pl.DataFrame:
    return frame(f"schedules_{SEASON}")


@pytest.fixture(scope="module")
def capture(schedules):
    return collect_vegas_from_schedules(schedules)


def _by_game(capture) -> dict[str, dict]:
    return {row["game_id"]: row for row in capture.rows}


def _moneyline_favorite(row) -> str | None:
    """The side the moneyline prices as the favourite (more negative = shorter odds)."""
    home, away = row["home_moneyline"], row["away_moneyline"]
    if home is None or away is None or home == away:
        return None
    return row["home_team"] if home < away else row["away_team"]


# --------------------------------------------------------------------------- registry agreement
def test_capture_is_registered_and_season_partitioned(capture, schedules):
    """Game-grain, season-partitioned like ``nflverse_schedules``: game_id is unique across a year."""
    assert capture.source == "vegas_odds"
    assert capture.key_cols == SOURCES["vegas_odds"].key_cols == ("game_id",)
    assert SOURCES["vegas_odds"].grain == "game"
    assert capture.week is None
    assert capture.season == SEASON
    assert len(capture.rows) == schedules.height


def test_the_season_comes_from_the_frame_or_the_loader(schedules):
    calls = []

    def load(season):
        calls.append(season)
        return schedules

    assert collect_vegas_from_schedules(season=2019, load=load).season == 2019
    assert calls == [2019]
    # ...and an explicit season still wins over the frame's own stamp.
    assert collect_vegas_from_schedules(schedules, season=2019).season == 2019


def test_a_multi_season_frame_is_refused_rather_than_filed_under_one_of_them(schedules):
    spanning = pl.concat(
        [schedules, schedules.with_columns(pl.lit(2025, dtype=pl.Int32).alias("season"))]
    )
    with pytest.raises(ValueError, match="spans seasons"):
        collect_vegas_from_schedules(spanning)


def test_a_missing_key_column_raises_rather_than_keying_on_what_is_left(schedules):
    with pytest.raises(ValueError, match="game_id"):
        collect_vegas_from_schedules(schedules.drop("game_id"))


@pytest.mark.parametrize("column", ["total_line", "spread_line", "home_team", "away_team"])
def test_a_missing_line_raises_rather_than_capturing_a_partition_of_nulls(schedules, column):
    """Without the two lines every implied total is null — and looks exactly like an unpriced week."""
    with pytest.raises(ValueError, match=column):
        collect_vegas_from_schedules(schedules.drop(column))


# --------------------------------------------------------------------------- the implied-total math
def test_implied_totals_sum_to_the_total_line(capture):
    """The acceptance criterion. True under either sign convention — see the sign tests below."""
    priced = 0
    for row in capture.rows:
        if row["total_line"] is None:
            continue
        priced += 1
        assert row["home_implied_total"] + row["away_implied_total"] == pytest.approx(
            row["total_line"]
        )
    assert priced == len(capture.rows), "fixture no longer covers a fully priced week"


def test_the_implied_difference_is_the_spread(capture):
    for row in capture.rows:
        assert row["home_implied_total"] - row["away_implied_total"] == pytest.approx(
            row["spread_line"]
        )


def test_a_known_game_anchors_the_sign_to_the_home_favourite(capture):
    """``2024_01_BAL_KC``: total 46.0, spread +3.0, KC (home) at -148 -> KC 24.5, BAL 21.5."""
    row = _by_game(capture)["2024_01_BAL_KC"]
    assert (row["home_team"], row["away_team"]) == ("KC", "BAL")
    assert (row["total_line"], row["spread_line"]) == (46.0, 3.0)
    assert row["home_moneyline"] == -148 < row["away_moneyline"]  # KC is the favourite
    assert row["home_implied_total"] == 24.5
    assert row["away_implied_total"] == 21.5
    assert row["favored_team"] == "KC"


def test_a_known_game_anchors_the_sign_to_the_away_favourite(capture):
    """``2024_01_HOU_IND``: spread -3.0 with HOU (away) at -155 -> HOU 26.0, IND 23.0.

    The companion to the test above, and the one that would catch a negated convention: a formula
    with the sign flipped still sums to ``total_line`` here, but hands the underdog the ceiling.
    """
    row = _by_game(capture)["2024_01_HOU_IND"]
    assert (row["home_team"], row["away_team"]) == ("IND", "HOU")
    assert (row["total_line"], row["spread_line"]) == (49.0, -3.0)
    assert row["away_moneyline"] == -155 < row["home_moneyline"]  # HOU is the favourite
    assert row["away_implied_total"] == 26.0
    assert row["home_implied_total"] == 23.0
    assert row["favored_team"] == "HOU"


def test_the_moneyline_favourite_always_holds_the_higher_implied_total(capture):
    """Per-game, across the whole fixture: two independent markets must name the same favourite."""
    checked = 0
    for row in capture.rows:
        favorite = _moneyline_favorite(row)
        if favorite is None or row["spread_line"] == 0:
            continue
        checked += 1
        assert row["favored_team"] == favorite, row["game_id"]
        higher = (
            row["home_team"]
            if row["home_implied_total"] > row["away_implied_total"]
            else row["away_team"]
        )
        assert higher == favorite, row["game_id"]
    assert checked == len(capture.rows), "fixture no longer covers a fully priced week"


def test_the_spread_sign_leans_the_way_the_games_actually_went(capture, schedules):
    """Anchored to the outcome, not just to another line: ``result`` is ``home - away``."""
    results = dict(zip(schedules["game_id"].to_list(), schedules["result"].to_list()))
    margins = [
        (row["spread_line"], results[row["game_id"]])
        for row in capture.rows
        if results.get(row["game_id"]) is not None
    ]
    assert len(margins) >= 16
    agreed = sum(1 for spread, result in margins if spread * result > 0)
    # A negated convention would put this strictly below half. (Full-scale: 67.1% over 2016-2025.)
    assert agreed / len(margins) > 0.5


@pytest.mark.parametrize(
    ("total", "spread", "expected"),
    [
        (46.0, 3.0, (24.5, 21.5)),
        (49.0, -3.0, (23.0, 26.0)),
        (44.0, 0.0, (22.0, 22.0)),  # pick'em splits the total evenly
        (None, 3.0, (None, None)),
        (46.0, None, (None, None)),
        (float("nan"), 3.0, (None, None)),
    ],
)
def test_implied_totals_handles_the_edges(total, spread, expected):
    assert implied_totals(total, spread) == expected


def test_a_pick_em_has_no_favourite():
    assert favored_team({"spread_line": 0.0, "home_team": "KC", "away_team": "BAL"}) is None
    assert favored_team({"spread_line": None, "home_team": "KC", "away_team": "BAL"}) is None


# --------------------------------------------------------------------------- unpriced games
def test_an_unpriced_game_is_kept_with_null_implied_totals():
    """"Not priced as of this capture" is point-in-time data; dropping it fakes a complete partition."""
    forward = frame(f"schedules_intl_{FORWARD_SEASON}")
    unpriced = forward.filter(pl.col("total_line").is_null())
    assert unpriced.height, "fixture no longer carries an unpriced game"

    capture = collect_vegas_from_schedules(forward)
    assert len(capture.rows) == forward.height  # nothing dropped

    rows = _by_game(capture)
    for game_id in unpriced["game_id"].to_list():
        row = rows[game_id]
        assert row["home_implied_total"] is None
        assert row["away_implied_total"] is None
        assert row["favored_team"] is None


# --------------------------------------------------------------------------- shape contract
def test_no_post_game_column_reaches_the_market_source(capture):
    """This source is the pre-kickoff belief. The outcome is one join away in nflverse_schedules."""
    for row in capture.rows:
        assert not set(row) & set(OUTCOME_COLS)


def test_the_raw_market_columns_are_carried_verbatim(capture, schedules):
    """The derivation is only auditable if its inputs sit next to it, unchanged."""
    original = {row["game_id"]: row for row in schedules.to_dicts()}
    for row in capture.rows:
        for name in ("spread_line", "total_line", "home_moneyline", "away_moneyline",
                     "over_odds", "under_odds", "home_spread_odds", "away_spread_odds"):
            assert row[name] == original[row["game_id"]][name], name


def test_the_row_says_which_week_it_is_but_not_which_season(capture):
    """The partition names the season; it does not name the week, so the row has to."""
    for row in capture.rows:
        assert row["week"] is not None
        assert "season" not in row


def test_collectors_never_emit_reserved_columns(capture):
    for row in capture.rows:
        assert not set(row) & set(RESERVED)


def test_rows_are_json_safe(capture):
    for row in capture.rows:
        for name, value in row.items():
            assert value is None or isinstance(value, (str, int, float, bool)), (
                f"{name} is {type(value).__name__}, not a json-safe scalar"
            )


# --------------------------------------------------------------------------- store hand-off
def test_key_cols_identify_rows_on_the_fixture(capture):
    keys = [row["game_id"] for row in capture.rows]
    assert len(keys) == len(set(keys))
    assert all(key and str(key).strip() for key in keys)


@pytest.mark.parametrize("fixture", [f"schedules_{SEASON}", f"schedules_intl_{FORWARD_SEASON}"])
def test_capture_round_trips_through_the_store_without_warnings(tmp_path, caplog, fixture):
    """The acceptance criterion end-to-end: the store logs nothing about this capture's keys."""
    capture = collect_vegas_from_schedules(frame(fixture))
    backend = LocalParquetBackend(root=tmp_path)

    with caplog.at_level(logging.WARNING):
        write_snapshot(
            capture.source,
            capture.season,
            capture.rows,
            captured_at=CAPTURED_AT,
            week=capture.week,
            key_cols=capture.key_cols,
            backend=backend,
        )
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []

    stored = read_snapshot(capture.source, capture.season, capture.week, backend=backend)
    assert len(stored) == len(capture.rows)
    assert not stored[list(capture.key_cols)].duplicated().any()
    assert stored["_source"].eq("vegas_odds").all()
    assert stored["_week"].isna().all()


def test_implied_totals_survive_the_parquet_round_trip(tmp_path, capture):
    backend = LocalParquetBackend(root=tmp_path)
    write_snapshot(capture.source, capture.season, capture.rows, captured_at=CAPTURED_AT,
                   week=None, key_cols=capture.key_cols, backend=backend)

    stored = read_snapshot(capture.source, capture.season, backend=backend).set_index("game_id")
    for row in capture.rows:
        assert stored.loc[row["game_id"], "home_implied_total"] == row["home_implied_total"]
        assert stored.loc[row["game_id"], "away_implied_total"] == row["away_implied_total"]

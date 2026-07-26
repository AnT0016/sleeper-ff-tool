"""Offline unit tests for the weather collector (``collect.weather``) and the venue table.

No test here touches the network: every forecast comes from an injected ``fetch``. Two committed
parquet schedule samples supply the real shapes —

* ``schedules_2024`` — a played week: fixed domes, a closed retractable roof, open-air games with
  nflverse's observed ``temp``/``wind``, and four distinct kickoff slots so the Eastern-to-UTC
  conversion is exercised across a date boundary;
* ``schedules_intl_2026`` — the *forward* schedule, which is where the awkward shapes live:
  international games filed under the **home team's** ``stadium_id`` and ``roof`` (Stade de France
  tagged ``roof='dome'`` from the Superdome), retractable venues whose ``roof`` is still null, and
  unpriced games.

Three properties carry the acceptance criteria: an enclosed game gets **null** forecast columns
rather than a fabricated calm day; nothing the fetcher can do makes a capture raise; and the whole
capture reaches ``store.write_snapshot`` with zero warnings.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from collect.registry import SOURCES
from collect.weather import (
    _FORECAST_COLS,
    collect_weather_forecast,
    kickoff_utc,
    resolve_indoor,
)
from data.stadiums import VENUES, Venue, venue_for_game, venue_for_team
from store.lake import RESERVED, LocalParquetBackend, read_snapshot, write_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "nflverse"

SEASON = 2024
WEEK = 1
FORWARD_SEASON = 2026
CAPTURED_AT = "2026-07-26T12:00:00+00:00"

FORECAST_COLS = tuple(column for column, _ in _FORECAST_COLS)


def frame(name: str) -> pl.DataFrame:
    return pl.read_parquet(FIXTURES / f"{name}.parquet")


def hourly_for(day: date, *, temp_at_noon: float = 60.0) -> dict[str, list]:
    """A whole UTC day of hourly values, distinct per hour so "nearest hour" is checkable."""
    return {
        "time": [f"{day.isoformat()}T{hour:02d}:00" for hour in range(24)],
        "temperature_2m": [temp_at_noon - 12 + hour for hour in range(24)],
        "relative_humidity_2m": [40 + hour for hour in range(24)],
        "precipitation": [round(hour / 100, 2) for hour in range(24)],
        "precipitation_probability": [hour * 4 for hour in range(24)],
        "wind_speed_10m": [float(hour) for hour in range(24)],
        "wind_gusts_10m": [float(hour) * 2 for hour in range(24)],
        "snowfall": [0.0 for _ in range(24)],
        "weather_code": [hour % 5 for hour in range(24)],
    }


def recording_fetch(payload=hourly_for):
    """A stand-in :data:`collect.weather.Fetcher` that records every call it receives."""

    def fetch(latitude, longitude, day):
        fetch.calls.append((latitude, longitude, day))
        return payload(day) if callable(payload) else payload

    fetch.calls = []
    return fetch


@pytest.fixture(scope="module")
def schedules() -> pl.DataFrame:
    return frame(f"schedules_{SEASON}")


@pytest.fixture(scope="module")
def forward() -> pl.DataFrame:
    return frame(f"schedules_intl_{FORWARD_SEASON}")


@pytest.fixture(scope="module")
def capture(schedules):
    return collect_weather_forecast(SEASON, WEEK, schedules, fetch=recording_fetch())


def _by_game(capture) -> dict[str, dict]:
    return {row["game_id"]: row for row in capture.rows}


# --------------------------------------------------------------------------- registry agreement
def test_capture_is_registered_and_week_partitioned(capture, schedules):
    """Fetched a week at a time (a forecast is only meaningful near the game), so: week partition."""
    assert capture.source == "weather"
    assert capture.key_cols == SOURCES["weather"].key_cols == ("game_id",)
    assert SOURCES["weather"].grain == "game"
    assert capture.season == SEASON
    assert capture.week == WEEK
    assert len(capture.rows) == schedules.height


def test_only_the_asked_for_week_is_captured(forward):
    """The forward fixture spans nine weeks; a capture must take exactly one of them."""
    weeks = set(forward["week"].to_list())
    assert len(weeks) > 1, "fixture no longer spans several weeks"

    capture = collect_weather_forecast(FORWARD_SEASON, 7, forward, fetch=recording_fetch())
    assert {row["week"] for row in capture.rows} == {7}
    assert [row["game_id"] for row in capture.rows] == ["2026_07_PIT_NO"]


def test_the_season_is_loaded_when_no_frame_is_given(schedules):
    calls = []

    def load(season):
        calls.append(season)
        return schedules

    capture = collect_weather_forecast(SEASON, WEEK, load=load, fetch=recording_fetch())
    assert calls == [SEASON]
    assert len(capture.rows) == schedules.height


def test_a_missing_key_column_raises_rather_than_keying_on_what_is_left(schedules):
    with pytest.raises(ValueError, match="game_id"):
        collect_weather_forecast(SEASON, WEEK, schedules.drop("game_id"), fetch=recording_fetch())


# --------------------------------------------------------------------------- domes
def test_enclosed_games_are_flagged_with_null_weather_and_never_fetched(capture, schedules):
    """The acceptance criterion: a flag, not a fabricated calm day.

    Covers both ways a game is enclosed — a fixed dome (Ford Field, the Superdome, SoFi) and a
    retractable roof the schedule says was **closed** for this game (Mercedes-Benz, Lucas Oil).
    """
    indoor = [row for row in capture.rows if row["is_indoor"] is True]
    assert {row["stadium"] for row in indoor} == {
        "Mercedes-Benz Superdome", "SoFi Stadium", "Ford Field",  # fixed domes
        "Mercedes-Benz Stadium", "Lucas Oil Stadium",             # retractable, closed that day
    }, "fixture no longer covers both kinds of enclosed game"

    for row in indoor:
        assert row["weather_status"] == "indoor"
        assert row["forecast_time_utc"] is None
        for column in FORECAST_COLS:
            assert row[column] is None, f"{row['game_id']}: fabricated {column}"

    fetch = recording_fetch()
    collect_weather_forecast(SEASON, WEEK, schedules, fetch=fetch)
    indoor_venues = {(row["latitude"], row["longitude"]) for row in indoor}
    assert not indoor_venues & {(lat, lon) for lat, lon, _day in fetch.calls}


def test_a_dome_venue_is_flagged_even_when_the_schedule_forgot_to_say_so(schedules):
    """The venue is authoritative for a fixed roof; a null ``roof`` must not un-dome the Superdome."""
    blanked = schedules.with_columns(
        pl.when(pl.col("stadium") == "Mercedes-Benz Superdome")
        .then(None)
        .otherwise(pl.col("roof"))
        .alias("roof")
    )
    capture = collect_weather_forecast(SEASON, WEEK, blanked, fetch=recording_fetch())
    row = _by_game(capture)["2024_01_CAR_NO"]
    assert row["roof"] is None
    assert row["is_indoor"] is True and row["weather_status"] == "indoor"


def test_an_open_retractable_roof_gets_a_forecast(schedules):
    opened = schedules.with_columns(
        pl.when(pl.col("stadium") == "Lucas Oil Stadium")
        .then(pl.lit("open"))
        .otherwise(pl.col("roof"))
        .alias("roof")
    )
    capture = collect_weather_forecast(SEASON, WEEK, opened, fetch=recording_fetch())
    row = _by_game(capture)["2024_01_HOU_IND"]
    assert row["roof_type"] == "retractable"
    assert row["is_indoor"] is False
    assert row["weather_status"] == "forecast"
    assert row["forecast_wind_mph"] is not None


def test_a_retractable_roof_with_no_state_recorded_is_unresolved_not_assumed(forward):
    """Common on the forward schedule. ``None`` says "unknown"; the outdoor forecast is still useful."""
    capture = collect_weather_forecast(FORWARD_SEASON, 1, forward, fetch=recording_fetch())
    row = _by_game(capture)["2026_01_BAL_IND"]
    assert row["roof"] is None and row["roof_type"] == "retractable"
    assert row["is_indoor"] is None
    assert row["weather_status"] == "forecast"


@pytest.mark.parametrize(
    ("roof_type", "roof", "expected"),
    [
        ("dome", "dome", True),
        ("dome", "outdoors", True),        # the venue wins where its roof is fixed...
        ("outdoor", "dome", False),        # ...in both directions
        ("outdoor", "outdoors", False),
        ("outdoor", None, False),
        ("retractable", "closed", True),   # ...and the game wins where only it can know
        ("retractable", "dome", True),
        ("retractable", "open", False),
        ("retractable", "outdoors", False),
        ("retractable", None, None),
    ],
)
def test_resolve_indoor_matrix(roof_type, roof, expected):
    venue = Venue("v", "V", roof_type, 0.0, 0.0)
    assert resolve_indoor(venue, roof) is expected


def test_resolve_indoor_falls_back_to_the_roof_when_the_venue_is_unknown():
    assert resolve_indoor(None, "closed") is True
    assert resolve_indoor(None, "outdoors") is False
    assert resolve_indoor(None, None) is None


# --------------------------------------------------------------------------- venue resolution
def test_an_international_game_resolves_by_stadium_name_not_the_home_teams_id(forward, caplog):
    """The reason ``data.stadiums`` looks up names first.

    nflverse files the 2026 international games under the *home* team's ``stadium_id`` **and**
    ``roof``: Stade de France arrives as ``NOR00``/``dome`` from the Superdome. Keying on the id
    would fetch New Orleans weather for a game in Paris; trusting ``roof`` would discard it.
    """
    row = forward.filter(pl.col("game_id") == "2026_07_PIT_NO").to_dicts()[0]
    assert (row["stadium_id"], row["roof"]) == ("NOR00", "dome"), "fixture no longer covers this"

    with caplog.at_level(logging.INFO, logger="collect.weather"):
        capture = collect_weather_forecast(FORWARD_SEASON, 7, forward, fetch=recording_fetch())

    collected = _by_game(capture)["2026_07_PIT_NO"]
    assert collected["venue_id"] == "par_stade_de_france"
    assert (round(collected["latitude"], 2), round(collected["longitude"], 2)) == (48.92, 2.36)
    assert collected["roof"] == "dome"          # the provider's value is still carried verbatim...
    assert collected["roof_type"] == "outdoor"  # ...but ours is what decides
    assert collected["is_indoor"] is False
    assert collected["weather_status"] == "forecast"
    assert "using the venue" in caplog.text
    # A known provider quirk we deliberately override is not a defect: it must not warn.
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_every_venue_in_both_fixtures_resolves_without_a_warning(caplog):
    """A missing venue is a real gap — so it warns, and on real rows it must never fire."""
    with caplog.at_level(logging.WARNING, logger="collect.weather"):
        for name in (f"schedules_{SEASON}", f"schedules_intl_{FORWARD_SEASON}"):
            for row in frame(name).to_dicts():
                venue = venue_for_game(
                    stadium=row["stadium"],
                    stadium_id=row["stadium_id"],
                    home_team=row["home_team"],
                    location=row["location"],
                )
                assert venue is not None, (row["game_id"], row["stadium"])
    assert caplog.records == []


def test_an_unknown_venue_yields_null_coordinates_and_a_warning(schedules, caplog):
    """A brand-new *neutral* site — the one case with no safe fallback left to try."""
    unknown = "2024_01_GB_PHI"  # Arena Corinthians, location='Neutral'
    renamed = schedules.with_columns(
        pl.when(pl.col("game_id") == unknown)
        .then(pl.lit("Some New Stadium"))
        .otherwise(pl.col("stadium"))
        .alias("stadium"),
        pl.when(pl.col("game_id") == unknown)
        .then(pl.lit("XXX99"))
        .otherwise(pl.col("stadium_id"))
        .alias("stadium_id"),
    )
    assert renamed.filter(pl.col("game_id") == unknown)["location"].item() == "Neutral"

    fetch = recording_fetch()
    with caplog.at_level(logging.WARNING, logger="collect.weather"):
        capture = collect_weather_forecast(SEASON, WEEK, renamed, fetch=fetch)

    row = _by_game(capture)[unknown]
    assert row["venue_id"] is None and row["latitude"] is None and row["longitude"] is None
    assert row["weather_status"] == "no_venue"
    assert all(row[column] is None for column in FORECAST_COLS)
    assert f"no venue for game {unknown}" in caplog.text


def test_a_neutral_site_never_falls_back_to_the_home_teams_stadium():
    """A wrong coordinate is worse than a missing one: it looks like a measurement."""
    assert venue_for_game(stadium="Some New Stadium", home_team="KC", location="Neutral") is None
    # ...while a home game with an unrecognised name is safely the home team's venue.
    home = venue_for_game(stadium="Some New Stadium", home_team="KC", location="Home")
    assert home is not None and home.venue_id == "kc_arrowhead"


def test_every_franchise_maps_to_a_registered_venue():
    for team in ("ARI", "KC", "LA", "LAC", "LV", "NYG", "NYJ", "OAK", "SD", "WAS"):
        assert venue_for_team(team) in VENUES.values()
    assert venue_for_team("XXX") is None


# --------------------------------------------------------------------------- kickoff times
@pytest.mark.parametrize(
    ("gameday", "gametime", "expected"),
    [
        ("2024-09-05", "20:20", "2024-09-06T00:20:00Z"),  # EDT (UTC-4), rolls past midnight
        ("2024-09-08", "13:00", "2024-09-08T17:00:00Z"),
        ("2024-12-08", "13:00", "2024-12-08T18:00:00Z"),  # EST (UTC-5), after the DST change
        ("2026-10-25", "09:30", "2026-10-25T13:30:00Z"),  # a London kickoff, still quoted in ET
    ],
)
def test_kickoff_converts_from_eastern_to_utc(gameday, gametime, expected):
    assert kickoff_utc(gameday, gametime).strftime("%Y-%m-%dT%H:%M:%SZ") == expected


@pytest.mark.parametrize(
    ("gameday", "gametime"), [(None, "13:00"), ("2024-09-08", None), ("", ""), ("nonsense", "13:00")]
)
def test_an_unreadable_kickoff_is_none_rather_than_an_exception(gameday, gametime):
    assert kickoff_utc(gameday, gametime) is None


def test_a_game_with_no_kickoff_keeps_null_weather_and_says_why(schedules):
    blanked = schedules.with_columns(
        pl.when(pl.col("game_id") == "2024_01_BAL_KC")
        .then(None)
        .otherwise(pl.col("gametime"))
        .alias("gametime")
    )
    capture = collect_weather_forecast(SEASON, WEEK, blanked, fetch=recording_fetch())
    row = _by_game(capture)["2024_01_BAL_KC"]
    assert row["kickoff_utc"] is None
    assert row["weather_status"] == "no_kickoff"
    assert all(row[column] is None for column in FORECAST_COLS)


# --------------------------------------------------------------------------- the forecast itself
def test_the_forecast_hour_nearest_kickoff_is_the_one_taken(capture):
    """1pm ET is 17:00 UTC, so the 17:00 row — not the day's first or its mean."""
    row = _by_game(capture)["2024_01_TEN_CHI"]
    assert row["kickoff_utc"] == "2024-09-08T17:00:00Z"
    assert row["forecast_time_utc"] == "2024-09-08T17:00:00Z"
    assert row["forecast_wind_mph"] == 17.0          # hour index 17 of the synthetic day
    assert row["forecast_temp_f"] == 60.0 - 12 + 17
    assert row["forecast_precip_prob_pct"] == 68
    assert row["weather_status"] == "forecast"


def test_a_kickoff_between_hours_rounds_to_the_nearer_one(capture):
    """4:25pm ET = 20:25 UTC. Nearest hour is 20:00, not 21:00."""
    row = _by_game(capture)["2024_01_DAL_CLE"]
    assert row["kickoff_utc"] == "2024-09-08T20:25:00Z"
    assert row["forecast_time_utc"] == "2024-09-08T20:00:00Z"


def test_the_forecast_is_requested_once_per_venue_day_not_once_per_game(schedules):
    fetch = recording_fetch()
    capture = collect_weather_forecast(SEASON, WEEK, schedules, fetch=fetch)
    outdoors = [row for row in capture.rows if row["weather_status"] == "forecast"]
    assert len(fetch.calls) == len(set(fetch.calls)) == len(outdoors)
    assert len(fetch.calls) < len(capture.rows)  # the enclosed games cost nothing


def test_the_request_uses_the_utc_date_of_kickoff_not_the_schedules_local_date(schedules):
    """Sunday-night football kicks off on Monday in UTC; asking for Sunday would miss the hour."""
    fetch = recording_fetch()
    capture = collect_weather_forecast(SEASON, WEEK, schedules, fetch=fetch)
    row = _by_game(capture)["2024_01_NYJ_SF"]
    assert row["gameday"] == "2024-09-09" and row["kickoff_utc"].startswith("2024-09-10")
    assert (row["latitude"], row["longitude"], date(2024, 9, 10)) in fetch.calls


# --------------------------------------------------------------------------- best effort
@pytest.mark.parametrize(
    "payload",
    [
        None,                                     # nothing came back
        {},                                       # an empty object
        {"time": []},                             # no hours
        {"temperature_2m": [50.0]},               # no time axis
        {"time": ["not-a-timestamp"]},            # unparseable stamps
        {"time": ["2024-09-08T17:00"]},           # a time axis with no variables
    ],
)
def test_an_empty_or_malformed_forecast_never_raises(schedules, payload):
    """Best effort is the contract: a weather outage must not cost the run its other captures."""
    capture = collect_weather_forecast(
        SEASON, WEEK, schedules, fetch=lambda _lat, _lon, _day: payload
    )
    assert len(capture.rows) == schedules.height
    unavailable = [row for row in capture.rows if row["weather_status"] == "unavailable"]
    assert unavailable
    for row in unavailable:
        assert all(row[column] is None for column in FORECAST_COLS)
        assert row["forecast_time_utc"] is None


def test_a_fetcher_that_raises_never_raises_out_of_the_collector(schedules, caplog):
    def exploding(_lat, _lon, _day):
        raise RuntimeError("open-meteo is down")

    with caplog.at_level(logging.WARNING, logger="collect.weather"):
        capture = collect_weather_forecast(SEASON, WEEK, schedules, fetch=exploding)

    assert len(capture.rows) == schedules.height
    assert all(
        row["weather_status"] in ("indoor", "unavailable") for row in capture.rows
    )
    assert "open-meteo is down" in caplog.text


def test_one_venue_days_outage_does_not_cost_the_other_games(schedules):
    """A per-venue-day failure is contained: the rest of the slate still gets its forecast."""

    def flaky(latitude, longitude, day):
        if round(latitude, 2) == 39.05:  # Arrowhead
            raise TimeoutError("timed out")
        return hourly_for(day)

    capture = collect_weather_forecast(SEASON, WEEK, schedules, fetch=flaky)
    rows = _by_game(capture)
    assert rows["2024_01_BAL_KC"]["weather_status"] == "unavailable"
    assert rows["2024_01_TEN_CHI"]["weather_status"] == "forecast"


# --------------------------------------------------------------------------- shape contract
def test_the_observed_columns_carry_nflverses_own_measurement(capture, schedules):
    """The only weather the 2016-2025 backfill has — named apart from the forecast on purpose."""
    original = {row["game_id"]: row for row in schedules.to_dicts()}
    measured = 0
    for row in capture.rows:
        assert row["observed_temp_f"] == original[row["game_id"]]["temp"]
        assert row["observed_wind_mph"] == original[row["game_id"]]["wind"]
        measured += row["observed_temp_f"] is not None
    assert measured, "fixture no longer carries a played game's observed weather"


def test_collectors_never_emit_reserved_columns(capture):
    for row in capture.rows:
        assert not set(row) & set(RESERVED)


def test_rows_are_json_safe(capture):
    for row in capture.rows:
        for name, value in row.items():
            assert value is None or isinstance(value, (str, int, float, bool)), (
                f"{name} is {type(value).__name__}, not a json-safe scalar"
            )


def test_every_row_declares_a_known_status(capture):
    from collect.weather import WEATHER_STATUSES

    for row in capture.rows:
        assert row["weather_status"] in WEATHER_STATUSES


def test_key_cols_identify_rows_on_the_fixture(capture):
    keys = [row["game_id"] for row in capture.rows]
    assert len(keys) == len(set(keys))
    assert all(key and str(key).strip() for key in keys)


# --------------------------------------------------------------------------- store hand-off
@pytest.mark.parametrize(
    ("fixture", "season", "week"),
    [(f"schedules_{SEASON}", SEASON, WEEK), (f"schedules_intl_{FORWARD_SEASON}", FORWARD_SEASON, 1)],
)
def test_capture_round_trips_through_the_store_without_warnings(
    tmp_path, caplog, fixture, season, week
):
    """The acceptance criterion end-to-end: the store logs nothing about this capture's keys."""
    capture = collect_weather_forecast(season, week, frame(fixture), fetch=recording_fetch())
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

    stored = read_snapshot(capture.source, season, week, backend=backend)
    assert len(stored) == len(capture.rows)
    assert not stored[list(capture.key_cols)].duplicated().any()
    assert stored["_source"].eq("weather").all()
    assert stored["_week"].eq(week).all()


def test_the_indoor_flag_and_the_forecast_survive_the_parquet_round_trip(tmp_path, forward):
    """``is_indoor`` is three-state (True / False / unknown) and a null must not read back False.

    The forward fixture is the one that carries all three: a fixed dome, an open-air venue, and a
    retractable whose roof state nobody has recorded yet.
    """
    capture = collect_weather_forecast(FORWARD_SEASON, 1, forward, fetch=recording_fetch())
    assert {row["is_indoor"] for row in capture.rows} == {False, None}, "fixture drifted"

    backend = LocalParquetBackend(root=tmp_path)
    write_snapshot(capture.source, FORWARD_SEASON, capture.rows, captured_at=CAPTURED_AT,
                   week=1, key_cols=capture.key_cols, backend=backend)

    stored = read_snapshot(capture.source, FORWARD_SEASON, 1, backend=backend).set_index("game_id")
    for row in capture.rows:
        value = stored.loc[row["game_id"], "is_indoor"]
        if row["is_indoor"] is None:
            assert pd.isna(value), f"{row['game_id']}: unknown roof state read back as {value!r}"
        else:
            assert value == row["is_indoor"]
        for column in ("forecast_temp_f", "forecast_wind_mph", "forecast_time_utc"):
            expected = row[column]
            stored_value = stored.loc[row["game_id"], column]
            assert pd.isna(stored_value) if expected is None else stored_value == expected

"""Point-in-time capture of game-day weather: open-meteo forecast pre-lock, nflverse observed after.

Wind and cold are the two things that reliably move a fantasy week — a 20 mph crosswind is worth
more to a kicker's projection than most usage features — and neither is knowable from any source
already in the lake. This collector fills that in from **open-meteo**, which needs no key, no
account and no attribution beyond its CC-BY licence (spec: "every v1 source is keyless").

Two weather columns, and the difference between them matters:

* ``forecast_*`` — what open-meteo predicted for the kickoff hour **at capture time**. Genuinely
  point-in-time and genuinely pre-lock, which is what a model gets to use at inference. Only
  available while the game is inside the forecast API's window (roughly 92 days back to 15 days
  ahead), so on a 2016-2025 backfill these are null.
* ``observed_*`` — nflverse's own ``temp``/``wind`` for the game, carried verbatim. This is an
  **at-kickoff** measurement, so it is null on a pre-lock capture and populated on a backfill. It
  exists because it is the *only* weather available for the training seasons; without it the
  2016-2025 half of the lake would have no weather column at all.

  Ticket 7 must decide deliberately how to use it. Training on ``observed_*`` and serving on
  ``forecast_*`` is a train/serve mismatch, not a label leak (kickoff weather is forecastable days
  out, and the label is the player's points), but it is a real modelling choice and should be made
  in the open rather than by accident. The columns are named apart so it cannot be made by accident.

**Domes are flagged, never fabricated.** A game with no weather gets ``is_indoor=True`` and null
``forecast_*``; it does not get a zero wind speed, which a model would read as a real measurement of
a calm day. Which games those are is decided by :func:`resolve_indoor`, from two independent facts —
``data.stadiums``' per-*venue* ``roof_type`` and the schedule's per-*game* ``roof`` — because neither
alone is sufficient:

* a **retractable** roof is genuinely open or closed on the day, and only the schedule knows;
* the schedule's ``roof`` is copied from the *home* stadium for the 2026 international games, so it
  calls Stade de France and the Melbourne Cricket Ground domes. Trusting it would silently drop the
  weather for eight open-air games, which is precisely the sort of quiet wrongness the lake is meant
  not to accumulate.

So the venue wins where it is certain (fixed dome, fixed open air) and the game wins where only it
can know (retractable). See ``data.stadiums`` for the venue-resolution half of the same problem.

**Best-effort, always.** A weather fetch is the one part of a capture that depends on a third party
being up, and a cron that dies on it collects nothing at all — including the Sleeper projections
that are unrecoverable afterwards. So every failure path here (unknown venue, unparseable kickoff,
HTTP error, timeout, malformed payload, a date outside the forecast window) produces a row with null
weather and a ``weather_status`` saying which, and **nothing in this module raises** once a schedule
frame is in hand.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

import polars as pl

from data import nflverse
from data.stadiums import Venue, venue_for_game

from .base import Collected, dedupe_rows
from .registry import SOURCES

_LOG = logging.getLogger(__name__)

#: Keyless forecast endpoint. No auth, no account, no key — see the module docstring.
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

#: Hourly variables requested, in the units nflverse already uses for ``temp``/``wind`` (Fahrenheit,
#: mph), so ``forecast_*`` and ``observed_*`` are directly comparable without a conversion step.
_HOURLY_VARS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_gusts_10m",
    "snowfall",
    "weather_code",
)

#: ``forecast_<name>`` <- ``hourly[<variable>]``.
_FORECAST_COLS: tuple[tuple[str, str], ...] = (
    ("forecast_temp_f", "temperature_2m"),
    ("forecast_humidity_pct", "relative_humidity_2m"),
    ("forecast_precip_in", "precipitation"),
    ("forecast_precip_prob_pct", "precipitation_probability"),
    ("forecast_wind_mph", "wind_speed_10m"),
    ("forecast_wind_gust_mph", "wind_gusts_10m"),
    ("forecast_snowfall_in", "snowfall"),
    ("forecast_weather_code", "weather_code"),
)

#: The endpoint's own accepted range is [today-93d, today+15d]; a day of margin each side absorbs
#: clock skew and the UTC/endpoint-local boundary. Outside it the default fetcher declines without
#: making a request, so a 2016-2025 backfill costs zero HTTP calls instead of 2,700 rejected ones.
_PAST_DAYS = 92
_FORECAST_DAYS = 14

#: Kickoff times on the schedule are Eastern (nflverse's documented convention), including for the
#: international games — ``09:30`` ET is a London afternoon.
_SCHEDULE_TZ = "America/New_York"

#: ISO-8601 UTC, seconds precision — the spelling ``collect.nflverse`` already writes.
_ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

#: ``roof`` values that mean the field is enclosed for this game.
_INDOOR_ROOFS: frozenset[str] = frozenset({"dome", "closed"})
#: ...and the ones that mean it is open to the sky.
_OUTDOOR_ROOFS: frozenset[str] = frozenset({"outdoors", "open"})

#: Why a row has (or has not) forecast weather. One of:
#:
#: ``forecast``    open-meteo answered for the kickoff hour.
#: ``indoor``      enclosed for this game; no forecast was requested and none is invented.
#: ``no_venue``    the stadium could not be resolved to coordinates (logged as a WARNING).
#: ``no_kickoff``  ``gameday``/``gametime`` could not be read as a timestamp.
#: ``unavailable`` a fetch was attempted or declined and produced nothing — the backfill case
#:                 (outside the forecast window) and the outage case both land here.
WEATHER_STATUSES: tuple[str, ...] = (
    "forecast", "indoor", "no_venue", "no_kickoff", "unavailable",
)

Loader = Callable[..., pl.DataFrame]
#: ``fetch(latitude, longitude, day) -> hourly mapping | None``. The mapping is open-meteo's
#: ``hourly`` object: parallel lists keyed by variable, plus ``time`` as naive UTC ISO strings.
Fetcher = Callable[[float, float, date], Mapping[str, Sequence[Any]] | None]


# --------------------------------------------------------------------------- roof / venue
def resolve_indoor(venue: Venue | None, roof: str | None) -> bool | None:
    """Is the field enclosed for this game? ``None`` when nothing available settles it.

    The venue is authoritative where it is certain and the game's ``roof`` is authoritative where
    only it can be (a retractable roof). With no venue we fall back to ``roof`` alone, which is
    right far more often than not — it is only the neutral-site rows where it is copied from the
    wrong stadium.
    """
    if venue is not None and venue.roof_type != "retractable":
        return venue.roof_type == "dome"
    text = (roof or "").strip().lower()
    if text in _INDOOR_ROOFS:
        return True
    if text in _OUTDOOR_ROOFS:
        return False
    return None  # retractable with no roof state recorded yet — common on the forward schedule


# --------------------------------------------------------------------------- time
def kickoff_utc(gameday: Any, gametime: Any) -> datetime | None:
    """``gameday`` + ``gametime`` (Eastern) as a UTC datetime, or ``None`` if unreadable.

    Both come off the schedule as strings (``'2024-09-08'`` / ``'13:00'``). ``None`` rather than an
    exception: a missing kickoff costs one row its forecast, not the capture.
    """
    day, clock = str(gameday or "").strip(), str(gametime or "").strip()
    if not day or not clock:
        return None
    try:
        naive = datetime.strptime(f"{day} {clock[:5]}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    try:
        from zoneinfo import ZoneInfo

        eastern = naive.replace(tzinfo=ZoneInfo(_SCHEDULE_TZ))
    except Exception:  # pragma: no cover - only without a tz database (pandas pins one on Windows)
        _LOG.warning(
            "weather: no %s time zone available; kickoff times cannot be converted to UTC",
            _SCHEDULE_TZ,
        )
        return None
    return eastern.astimezone(timezone.utc)


def _iso(stamp: datetime | None) -> str | None:
    return None if stamp is None else stamp.strftime(_ISO_UTC)


# --------------------------------------------------------------------------- forecast
def open_meteo_hourly(
    latitude: float, longitude: float, day: date
) -> Mapping[str, Sequence[Any]] | None:
    """One UTC day of hourly forecast at a point, or ``None`` — the default :data:`Fetcher`.

    Deliberately **not** routed through ``sleeper.http``'s cached session: that cache exists to keep
    us under Sleeper's rate ceiling, and here a cached response would be the opposite of what a
    point-in-time capture wants (the value of a pre-lock snapshot is that it is the forecast as of
    *this* run). Requests are cheap — about 13 distinct venue-days for a full NFL Sunday.

    Returns ``None``, never raises, for every failure including a ``day`` the endpoint will not
    serve; that date check happens *here* rather than in the collector so the collector needs no
    clock and stays deterministic under test.
    """
    today = datetime.now(timezone.utc).date()
    offset = (day - today).days
    if not -_PAST_DAYS <= offset <= _FORECAST_DAYS:
        _LOG.info(
            "weather: %s is %+d day(s) from today, outside open-meteo's forecast window "
            "(-%d..+%d) — no request made",
            day, offset, _PAST_DAYS, _FORECAST_DAYS,
        )
        return None

    try:
        import requests

        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": ",".join(_HOURLY_VARS),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "UTC",
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
            },
            timeout=20,
        )
        response.raise_for_status()
        hourly = response.json().get("hourly")
    except Exception as exc:
        _LOG.warning(
            "weather: open-meteo forecast for (%.4f, %.4f) on %s failed: %s — the row keeps null "
            "weather", latitude, longitude, day, exc,
        )
        return None
    return hourly if isinstance(hourly, Mapping) else None


def _nearest_hour(
    hourly: Mapping[str, Sequence[Any]] | None, kickoff: datetime
) -> dict[str, Any] | None:
    """The forecast hour closest to ``kickoff``, or ``None`` if the payload cannot be read.

    Defensive by design: a shape change at the provider must degrade to null weather, not to a
    ``KeyError`` that aborts a capture whose other sources are irreplaceable.
    """
    if not isinstance(hourly, Mapping):
        return None
    stamps = hourly.get("time")
    if not isinstance(stamps, Sequence) or isinstance(stamps, (str, bytes)) or not stamps:
        return None

    target = kickoff.replace(tzinfo=None)
    best_index, best_gap, best_stamp = None, None, None
    for index, raw in enumerate(stamps):
        try:
            moment = datetime.fromisoformat(str(raw))
        except (TypeError, ValueError):
            continue
        moment = moment.replace(tzinfo=None)
        gap = abs(moment - target)
        if best_gap is None or gap < best_gap:
            best_index, best_gap, best_stamp = index, gap, moment
    if best_index is None:
        return None
    if best_gap is not None and best_gap > timedelta(hours=1):
        # A whole-day request that lands >1h from kickoff means the wrong day was fetched.
        _LOG.info(
            "weather: nearest forecast hour %s is %s from kickoff %s — using it anyway",
            best_stamp, best_gap, target,
        )

    picked: dict[str, Any] = {"forecast_time_utc": _iso(best_stamp)}
    for column, variable in _FORECAST_COLS:
        series = hourly.get(variable)
        value = None
        if isinstance(series, Sequence) and not isinstance(series, (str, bytes)):
            if best_index < len(series):
                value = series[best_index]
        picked[column] = value
    if all(picked[column] is None for column, _ in _FORECAST_COLS):
        # A time axis carrying no variable is not a forecast. Reporting it as one would put a row
        # of nulls behind a ``weather_status`` of "forecast" — indistinguishable, downstream, from
        # a genuinely calm hour.
        return None
    return picked


# --------------------------------------------------------------------------- rows
def _blank_forecast() -> dict[str, Any]:
    return {"forecast_time_utc": None, **{column: None for column, _ in _FORECAST_COLS}}


def _roof_disagrees(venue: Venue, roof: str | None) -> bool:
    """Does the schedule's ``roof`` contradict a venue whose roof is *fixed*?

    True only for the neutral-site rows where nflverse copied the home stadium's roof onto a game
    played somewhere else. A retractable venue can never disagree — there ``roof`` is the answer.
    """
    if venue.roof_type == "retractable":
        return False
    from_schedule = resolve_indoor(None, roof)
    return from_schedule is not None and from_schedule != venue.dome


def _row(raw: Mapping[str, Any], venue: Venue | None, kickoff: datetime | None) -> dict[str, Any]:
    """Everything about a game that does not depend on the network."""
    return {
        "game_id": raw.get("game_id"),
        "game_type": raw.get("game_type"),
        "week": raw.get("week"),
        "gameday": raw.get("gameday"),
        "gametime": raw.get("gametime"),
        "kickoff_utc": _iso(kickoff),
        "away_team": raw.get("away_team"),
        "home_team": raw.get("home_team"),
        "location": raw.get("location"),
        "stadium_id": raw.get("stadium_id"),
        "stadium": raw.get("stadium"),
        "venue_id": None if venue is None else venue.venue_id,
        "latitude": None if venue is None else venue.latitude,
        "longitude": None if venue is None else venue.longitude,
        "roof": raw.get("roof"),
        "roof_type": None if venue is None else venue.roof_type,
        "is_indoor": resolve_indoor(venue, raw.get("roof")),
        "weather_status": None,  # filled in below; kept here to pin the column order
        **_blank_forecast(),
        # nflverse's own at-kickoff measurement. Null before the game, populated after — the only
        # weather the 2016-2025 backfill has. See the module docstring on how it differs.
        "observed_temp_f": raw.get("temp"),
        "observed_wind_mph": raw.get("wind"),
    }


def _log_venue_gap(raw: Mapping[str, Any]) -> None:
    _LOG.warning(
        "weather: no venue for game %s (stadium=%r, stadium_id=%r, location=%r) — null "
        "coordinates and no forecast. Add it to data.stadiums.",
        raw.get("game_id"), raw.get("stadium"), raw.get("stadium_id"), raw.get("location"),
    )


def _log_roof_override(raw: Mapping[str, Any], venue: Venue) -> None:
    _LOG.info(
        "weather: game %s at %s is tagged roof=%r, but %s is fixed %s — using the venue. "
        "(nflverse copies the home stadium's roof onto neutral-site games.)",
        raw.get("game_id"), raw.get("stadium"), raw.get("roof"), venue.name, venue.roof_type,
    )


def collect_weather_forecast(
    season: int,
    week: int,
    schedules_df: pl.DataFrame | None = None,
    *,
    fetch: Fetcher | None = None,
    load: Loader | None = None,
) -> Collected:
    """``weather``: one row per game of ``season``/``week``, forecast where there is weather to have.

    Game-grain and **week-partitioned** — unlike the nflverse collectors, this one is called a week
    at a time (a forecast is only meaningful near the game), and ``game_id`` is unique within a
    week. Pass ``schedules_df`` (ticket 5's runner downloads the schedule once and feeds both this
    and ``collect.market``) or leave it out to load the season.

    ``fetch`` defaults to :func:`open_meteo_hourly` and is the single injection point for offline
    tests. It is called once per distinct (latitude, longitude, UTC date), not once per game, so a
    16-game Sunday costs roughly 13 requests.

    Never raises: see the module docstring on why a weather outage must not cost a capture.
    """
    season, week = int(season), int(week)
    if schedules_df is None:
        schedules_df = (load or nflverse.load_schedules)(season)

    key_cols = SOURCES["weather"].key_cols
    missing = [c for c in key_cols if c not in schedules_df.columns]
    if missing:
        raise ValueError(
            f"weather: key column(s) {missing} absent from the schedules frame "
            f"(columns: {sorted(schedules_df.columns)[:20]}); the provider schema has changed"
        )

    games = schedules_df
    if "season" in games.columns:
        games = games.filter(pl.col("season") == season)
    if "week" in games.columns:
        games = games.filter(pl.col("week") == week)

    get = fetch or open_meteo_hourly
    cache: dict[tuple[float, float, date], Mapping[str, Sequence[Any]] | None] = {}
    rows: list[dict[str, Any]] = []
    tally = dict.fromkeys(WEATHER_STATUSES, 0)

    for raw in games.to_dicts():
        venue = venue_for_game(
            stadium=raw.get("stadium"),
            stadium_id=raw.get("stadium_id"),
            home_team=raw.get("home_team"),
            location=raw.get("location"),
        )
        if venue is None:
            _log_venue_gap(raw)
        elif _roof_disagrees(venue, raw.get("roof")):
            _log_roof_override(raw, venue)

        kickoff = kickoff_utc(raw.get("gameday"), raw.get("gametime"))
        row = _row(raw, venue, kickoff)
        if row["is_indoor"] is True:
            row["weather_status"] = "indoor"
        elif venue is None:
            row["weather_status"] = "no_venue"
        elif kickoff is None:
            row["weather_status"] = "no_kickoff"
        else:
            spot = (venue.latitude, venue.longitude, kickoff.date())
            if spot not in cache:
                try:
                    cache[spot] = get(*spot)
                except Exception as exc:
                    # An injected fetcher (or a future default) that raises must not abort the run:
                    # the point-in-time sources captured alongside this one are unrecoverable.
                    _LOG.warning(
                        "weather: forecast fetch for %s raised %s: %s — null weather for the "
                        "game(s) at that venue-day", spot, type(exc).__name__, exc,
                    )
                    cache[spot] = None
            picked = _nearest_hour(cache[spot], kickoff)
            if picked is None:
                row["weather_status"] = "unavailable"
            else:
                row.update(picked)
                row["weather_status"] = "forecast"
        tally[row["weather_status"]] += 1
        rows.append(row)

    rows = dedupe_rows(rows, key_cols, source="weather")
    _LOG.info(
        "weather season=%s week=%s: %d game(s), %d forecast request(s) — %s",
        season, week, len(rows), len(cache),
        ", ".join(f"{status}={count}" for status, count in tally.items() if count),
    )
    return Collected.for_source("weather", season, rows, week=week)

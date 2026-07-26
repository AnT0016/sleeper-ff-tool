"""Offline unit tests for the Sleeper collectors (``collect.sleeper``).

Every test runs against ``tests/fixtures/sleeper_raw_2025_w1.json`` — verbatim API rows captured by
``scripts/make_collect_fixture.py`` — so the collectors are exercised on the provider's real shapes
(a team-abbreviation-keyed DEF row, a kicker's distance buckets, a teamless free agent whose
``team``/``opponent``/``game_id``/``date`` are all null) with no network.

Two properties get the most attention because breaking either corrupts the lake *silently*:
the declared key must actually identify a row (the store dedups on it, so a duplicate or null key
deletes real rows on merge), and the raw stats must survive untouched (the lake is scoring-agnostic;
re-scoring happens downstream, in whatever scoring the league has at the time).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from collect.base import Collected
from collect.registry import SOURCES
from collect.sleeper import (
    _INJURY_FIELDS,
    collect_proj_season,
    collect_proj_week,
    collect_stats_week,
)
from store.lake import RESERVED, LocalParquetBackend, read_snapshot, write_snapshot

FIXTURE = Path(__file__).parent / "fixtures" / "sleeper_raw_2025_w1.json"

SEASON = 2025
WEEK = 1
CAPTURED_AT = "2026-09-13T16:00:00+00:00"

#: Known fixture rows: a QB (dense offensive stat line) and a DEF row keyed on a team abbreviation.
QB_ID = "4881"
DEF_ID = "DEN"


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fetch(rows):
    """Stand in for the ``sleeper.client`` call: records its arguments, returns fixture rows."""

    def fetch(*args, **kwargs):
        fetch.calls.append((args, kwargs))
        return rows

    fetch.calls = []
    return fetch


def _captures(raw: dict) -> dict[str, Collected]:
    return {
        "sleeper_proj_week": collect_proj_week(
            SEASON, WEEK, fetch=_fetch(raw["proj_week"]["rows"])
        ),
        "sleeper_proj_season": collect_proj_season(
            SEASON, fetch=_fetch(raw["proj_season"]["rows"])
        ),
        "sleeper_stats_week": collect_stats_week(
            SEASON, WEEK, fetch=_fetch(raw["stats_week"]["rows"])
        ),
    }


@pytest.fixture(scope="module")
def captures(raw) -> dict[str, Collected]:
    return _captures(raw)


def _by_id(capture: Collected) -> dict[str, dict]:
    return {row["player_id"]: row for row in capture.rows}


def _raw_by_id(rows: list[dict]) -> dict[str, dict]:
    return {str(r["player_id"]): r for r in rows}


# --------------------------------------------------------------------------- registry agreement
def test_sources_are_registered_with_matching_keys(captures):
    """The envelope's key is the registry's key — ticket 3/4 collectors inherit the same pin."""
    for name, capture in captures.items():
        assert capture.source == name
        assert capture.key_cols == SOURCES[name].key_cols == ("player_id",)
        assert capture.season == SEASON
        assert capture.rows


def test_week_partitioning_matches_the_endpoint_grain(captures):
    assert captures["sleeper_proj_week"].week == WEEK
    assert captures["sleeper_stats_week"].week == WEEK
    # Season projections are one partition per season, so there is no week to file them under.
    assert captures["sleeper_proj_season"].week is None


def test_collector_forwards_season_week_and_positions(raw):
    fetch = _fetch(raw["proj_week"]["rows"])
    collect_proj_week(2026, 5, positions=("QB", "TE"), fetch=fetch)
    assert fetch.calls == [((2026, 5), {"positions": ("QB", "TE")})]

    season_fetch = _fetch(raw["proj_season"]["rows"])
    collect_proj_season(2026, positions=("QB",), fetch=season_fetch)
    assert season_fetch.calls == [((2026,), {"positions": ("QB",)})]

    stats_fetch = _fetch(raw["stats_week"]["rows"])
    collect_stats_week(2026, 5, positions=("DEF",), fetch=stats_fetch)
    assert stats_fetch.calls == [((2026, 5), {"positions": ("DEF",)})]


def test_collected_rejects_an_unregistered_source_or_foreign_key():
    with pytest.raises(ValueError, match="unknown source"):
        Collected.for_source("sleeper_proj_daily", SEASON, [], week=WEEK)
    with pytest.raises(ValueError, match="disagree with the registry"):
        Collected(
            source="sleeper_proj_week",
            season=SEASON,
            week=WEEK,
            rows=[],
            key_cols=("player_id", "team"),
        )


# --------------------------------------------------------------------------- key hygiene
def test_key_cols_identify_rows_on_the_fixture(captures):
    """No duplicate keys within one capture, and no null key — the store's two loss modes."""
    for name, capture in captures.items():
        ids = [row["player_id"] for row in capture.rows]
        assert len(ids) == len(set(ids)), f"{name}: duplicate player_id in one capture"
        assert all(isinstance(pid, str) and pid.strip() for pid in ids), f"{name}: null player_id"


def test_duplicate_player_ids_collapse_to_the_freshest_row(caplog):
    """A player listed twice keeps the provider's newest revision, and says so."""
    rows = [
        {"player_id": "99", "last_modified": 100, "stats": {"pass_yd": 10.0}},
        {"player_id": "99", "last_modified": 300, "stats": {"pass_yd": 33.0}},
        {"player_id": "98", "last_modified": 200, "stats": {"pass_yd": 5.0}},
    ]
    with caplog.at_level(logging.WARNING):
        capture = collect_proj_week(SEASON, WEEK, fetch=_fetch(rows))

    assert [r["player_id"] for r in capture.rows] == ["99", "98"]  # first-seen key order
    assert _by_id(capture)["99"]["pass_yd"] == 33.0
    assert "duplicate row" in caplog.text


def test_rows_without_a_usable_player_id_are_dropped(caplog):
    rows = [
        {"player_id": "7", "stats": {"rec": 1.0}},
        {"player_id": None, "stats": {"rec": 2.0}},
        {"player_id": "  ", "stats": {"rec": 3.0}},
        {"stats": {"rec": 4.0}},
    ]
    with caplog.at_level(logging.WARNING):
        capture = collect_stats_week(SEASON, WEEK, fetch=_fetch(rows))

    assert [r["player_id"] for r in capture.rows] == ["7"]
    assert "null/blank" in caplog.text


def test_a_stat_shadowing_the_key_never_replaces_the_key(caplog):
    """Identity is inviolable: a stat named ``player_id`` would file the row as someone else.

    The store dedups on whatever key it is handed, so letting the stat win here corrupts the lake
    silently -- and it would also undo ``_player_id``'s str coercion (the key would become a float).
    """
    rows = [
        {"player_id": "4881", "team": "BAL", "stats": {"player_id": 999.0, "pass_yd": 300.0}},
        {"player_id": "4882", "team": "BUF", "stats": {"pass_yd": 250.0}},
    ]
    with caplog.at_level(logging.WARNING):
        capture = collect_proj_week(SEASON, WEEK, fetch=_fetch(rows))

    stored = _by_id(capture)
    assert sorted(stored) == ["4881", "4882"]
    assert stored["4881"]["player_id"] == "4881"      # the key, not 999.0
    assert stored["4881"]["pass_yd"] == 300.0         # the rest of the stat line is untouched
    assert "collide with the row key" in caplog.text
    assert any(r.levelno >= logging.ERROR for r in caplog.records)  # loud: a schema change


def test_a_stat_shadowing_non_key_meta_still_wins(caplog):
    """The documented trade-off, pinned: for *meta* the raw stat wins (stats pass through)."""
    rows = [{"player_id": "1", "team": "BAL", "stats": {"team": 42.0}}]
    with caplog.at_level(logging.WARNING):
        capture = collect_proj_week(SEASON, WEEK, fetch=_fetch(rows))

    assert capture.rows[0]["team"] == 42.0
    assert "shadow meta columns" in caplog.text


def test_collectors_never_emit_reserved_columns(captures):
    """Provenance is the store's; a collector that forged it would make write_snapshot raise."""
    for capture in captures.values():
        for row in capture.rows:
            assert not set(row) & set(RESERVED)


# --------------------------------------------------------------------------- raw pass-through
@pytest.mark.parametrize(
    ("source", "block", "player_id"),
    [
        ("sleeper_proj_week", "proj_week", QB_ID),
        ("sleeper_proj_season", "proj_season", QB_ID),
        ("sleeper_stats_week", "stats_week", DEF_ID),
    ],
)
def test_raw_stats_pass_through_unchanged(raw, captures, source, block, player_id):
    """Every stat key and value survives verbatim — no re-scoring, no renaming, no rounding."""
    original = _raw_by_id(raw[block]["rows"])[player_id]["stats"]
    collected = _by_id(captures[source])[player_id]

    assert original, "fixture row must carry stats for this assertion to mean anything"
    for key, value in original.items():
        assert collected[key] == value, f"{source}/{player_id}: stat {key} was altered"


def test_sleepers_preset_points_are_kept_and_no_custom_score_is_added(raw, captures):
    """Sleeper's own pts_* presets are data, not our scoring — and we add no scored column here."""
    collected = _by_id(captures["sleeper_proj_week"])[QB_ID]
    original = _raw_by_id(raw["proj_week"]["rows"])[QB_ID]["stats"]

    assert collected["pts_half_ppr"] == original["pts_half_ppr"]
    assert not [c for c in collected if c.startswith("custom") or c in ("points", "proj_pts")]


def test_defense_rows_key_on_the_team_abbreviation(captures):
    """DST has no player id — Sleeper keys it by team, and the DST stat line must survive."""
    row = _by_id(captures["sleeper_stats_week"])[DEF_ID]
    assert row["position"] == "DEF"
    assert row["team"] == DEF_ID
    assert any(key.startswith("def_") for key in row)


def test_meta_comes_from_the_point_in_time_row_not_the_player_master():
    """``player`` is Sleeper's *current* master data; trusting its team would inject lookahead."""
    rows = [
        {
            "player_id": "42",
            "team": "MIN",  # the team as of the projected week
            "opponent": "CHI",
            "game_id": "202510106",
            "date": "2025-09-08",
            "season_type": "regular",
            "company": "rotowire",
            "last_modified": 1,
            "player": {"position": "WR", "fantasy_positions": ["WR"], "team": "NYJ"},
            "stats": {"rec_yd": 55.0},
        }
    ]
    row = collect_proj_week(SEASON, WEEK, fetch=_fetch(rows)).rows[0]

    assert row["team"] == "MIN"
    assert (row["position"], row["fantasy_positions"]) == ("WR", "WR")
    assert row["opponent"] == "CHI"
    assert "season" not in row and "week" not in row  # the store stamps _season/_week


def test_injury_fields_are_captured_only_on_the_forward_only_sources(captures):
    """Sleeper's injury_status is the project's *authoritative* injury signal, and Sleeper-keyed.

    Captured on the live pre-lock sources, where the mutable ``player`` master is genuinely as-of
    capture. Deliberately NOT on ``sleeper_stats_week``: it is backfillable, so a 2018 row would
    carry today's status — sharper than the stale-position problem, because injury status is wildly
    time-varying (a 2018 row reading Questionable because he is questionable *now* is just false).
    """
    for name in ("sleeper_proj_week", "sleeper_proj_season"):
        for row in captures[name].rows:
            assert set(_INJURY_FIELDS) <= set(row), f"{name}: injury fields missing"

    for row in captures["sleeper_stats_week"].rows:
        assert not set(_INJURY_FIELDS) & set(row), "backfillable source must not carry injury fields"


def test_injury_fields_come_from_the_player_master():
    """They exist nowhere else in the payload — same source as position, same as-of-capture caveat."""
    rows = [
        {
            "player_id": "42",
            "team": "MIN",
            "player": {
                "position": "WR",
                "fantasy_positions": ["WR"],
                "injury_status": "Questionable",
                "injury_body_part": "Hamstring",
                "injury_start_date": "2026-09-10",
            },
            "stats": {"rec_yd": 55.0},
        }
    ]
    proj = collect_proj_week(SEASON, WEEK, fetch=_fetch(rows)).rows[0]
    assert proj["injury_status"] == "Questionable"
    assert proj["injury_body_part"] == "Hamstring"
    assert proj["injury_start_date"] == "2026-09-10"

    # Same input through the backfillable source: the fields are dropped, not merely nulled.
    stats = collect_stats_week(SEASON, WEEK, fetch=_fetch(rows)).rows[0]
    assert not set(_INJURY_FIELDS) & set(stats)


def test_teamless_rows_are_kept_with_null_game_meta(raw, captures):
    """A free agent has no game: keep the row (absence of a projection is data), nulls and all."""
    teamless = [r for r in captures["sleeper_proj_week"].rows if r["team"] is None]
    assert teamless, "fixture must contain a teamless row"
    row = teamless[0]
    assert (row["opponent"], row["game_id"], row["date"]) == (None, None, None)
    assert row["player_id"] and row["position"]


# --------------------------------------------------------------------------- store hand-off
@pytest.mark.parametrize(
    "source", ["sleeper_proj_week", "sleeper_proj_season", "sleeper_stats_week"]
)
def test_capture_round_trips_through_the_store_without_warnings(raw, tmp_path, caplog, source):
    """The acceptance criterion end-to-end: the store logs nothing about this capture's keys."""
    capture = _captures(raw)[source]
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
    assert not stored["player_id"].duplicated().any()
    assert stored["_source"].eq(source).all()
    assert stored["_season"].eq(SEASON).all()
    assert stored["_week"].isna().all() if capture.week is None else stored["_week"].eq(WEEK).all()
    assert stored["_captured_at"].eq(CAPTURED_AT).all()


def test_stats_survive_the_parquet_round_trip(raw, tmp_path):
    capture = _captures(raw)["sleeper_stats_week"]
    backend = LocalParquetBackend(root=tmp_path)
    write_snapshot(
        capture.source,
        capture.season,
        capture.rows,
        captured_at=CAPTURED_AT,
        week=capture.week,
        key_cols=capture.key_cols,
        backend=backend,
    )
    stored = read_snapshot(capture.source, capture.season, capture.week, backend=backend)
    row = stored.set_index("player_id").loc[DEF_ID]
    original = _raw_by_id(raw["stats_week"]["rows"])[DEF_ID]["stats"]

    for key, value in original.items():
        assert row[key] == value, f"stat {key} did not survive parquet"

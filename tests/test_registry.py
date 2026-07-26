"""Offline unit tests for the Phase 8 source registry (``collect.registry``).

This registry is the interface tickets 2-4 build against, so the tests pin it hard: the exact source
set from the spec, and the invariants (validated keys, forward-only vs. backfillable, cadence
membership) that keep the runners and the assembler honest.
"""

from __future__ import annotations

import pytest

from collect.registry import (
    CADENCES,
    GRAINS,
    SOURCES,
    Source,
    backfillable_sources,
    sources_for_cadence,
)
from store.lake import RESERVED

#: The source list from docs/plans/data-collection.md -- one entry per collector in the spec.
EXPECTED = {
    "sleeper_proj_week",
    "sleeper_proj_season",
    "sleeper_stats_week",
    "nflverse_player_week",
    "nflverse_snaps",
    "nflverse_ff_opp",
    "nflverse_injuries",
    "nflverse_schedules",
    "nflverse_depth",
    "id_crosswalk",
    "vegas_odds",
    "weather",
}

#: Sleeper's projection endpoints serve only the latest values -- history is unrecoverable.
FORWARD_ONLY = {"sleeper_proj_week", "sleeper_proj_season"}


def test_registry_covers_exactly_the_spec_sources():
    assert set(SOURCES) == EXPECTED


def test_every_entry_is_keyed_by_its_own_name():
    assert all(name == source.name for name, source in SOURCES.items())


@pytest.mark.parametrize("source", SOURCES.values(), ids=list(SOURCES))
def test_every_source_is_well_formed(source):
    assert source.grain in GRAINS
    assert source.key_cols and len(set(source.key_cols)) == len(source.key_cols)
    assert not set(source.key_cols) & set(RESERVED)
    assert source.cadence and set(source.cadence) <= set(CADENCES)
    assert isinstance(source.cadence, frozenset)  # hashable + immutable: safe to share


def test_forward_only_sources_are_not_backfillable():
    for name in FORWARD_ONLY:
        source = SOURCES[name]
        assert source.backfillable is False
        assert "backfill" not in source.cadence


def test_backfillable_flag_and_cadence_agree_everywhere():
    for source in SOURCES.values():
        assert source.backfillable == ("backfill" in source.cadence)


def test_grains_match_the_shape_of_each_source():
    assert SOURCES["sleeper_proj_season"].grain == "season"
    assert SOURCES["id_crosswalk"].grain == "season"
    # Game-grain sources are keyed by game_id alone.
    for name in ("nflverse_schedules", "vegas_odds", "weather"):
        assert SOURCES[name].grain == "game"
        assert SOURCES[name].key_cols == ("game_id",)
    # Weekly nflverse loaders return a whole season, so week is part of the row key.
    assert "week" in SOURCES["nflverse_player_week"].key_cols


def test_injury_key_includes_the_report_revision():
    """A player is listed twice in a week as the report firms up (Questionable -> Out).

    Without ``date_modified`` those revisions share a key and one is silently dropped -- see
    ``tests/test_store.py::test_injury_report_revisions_survive_the_registry_key``.
    """
    assert "date_modified" in SOURCES["nflverse_injuries"].key_cols


def test_prelock_cadence_is_the_point_in_time_capture():
    prelock = {s.name for s in sources_for_cadence("prelock")}
    # The unrecoverable ones must be captured before lock, alongside the state-of-the-world context.
    assert FORWARD_ONLY <= prelock
    assert {"nflverse_injuries", "nflverse_schedules", "vegas_odds", "weather"} <= prelock
    # Finalized actuals are not a pre-lock quantity -- capturing them there would leak.
    assert "nflverse_player_week" not in prelock
    assert "sleeper_stats_week" not in prelock


def test_postgame_cadence_collects_the_labels_and_usage():
    postgame = {s.name for s in sources_for_cadence("postgame")}
    assert {"sleeper_stats_week", "nflverse_player_week", "nflverse_snaps", "nflverse_ff_opp"} <= (
        postgame
    )
    assert not FORWARD_ONLY & postgame


def test_backfillable_sources_matches_the_flag():
    assert {s.name for s in backfillable_sources()} == EXPECTED - FORWARD_ONLY
    assert {s.name for s in sources_for_cadence("backfill")} == EXPECTED - FORWARD_ONLY


# --------------------------------------------------------------------------- backfillable_from
def test_depth_charts_are_backfillable_only_from_the_rewritten_feed():
    """nflverse replaced the depth-chart feed in 2025; the legacy shape has no usable key.

    Expressing that by dropping ``"backfill"`` from the cadence is impossible — the
    cadence/backfillable invariant would force ``backfillable=False``, which claims the feed is
    unrecoverable, and it isn't from 2025 on.
    """
    depth = SOURCES["nflverse_depth"]
    assert depth.backfillable is True
    assert depth.backfillable_from == 2025
    assert not depth.backfills_season(2024)
    assert depth.backfills_season(2025) and depth.backfills_season(2026)


def test_every_other_source_backfills_as_far_back_as_asked():
    for source in SOURCES.values():
        if source.name == "nflverse_depth":
            continue
        assert source.backfillable_from is None
        assert source.backfills_season(2016) == source.backfillable


def test_backfillable_sources_narrows_to_what_a_season_can_actually_recover():
    assert "nflverse_depth" not in {s.name for s in backfillable_sources(2024)}
    assert "nflverse_depth" in {s.name for s in backfillable_sources(2025)}
    # Everything else is unaffected by the season.
    assert {s.name for s in backfillable_sources(2024)} | {"nflverse_depth"} == (
        {s.name for s in backfillable_sources()}
    )


def test_a_start_year_on_a_forward_only_source_is_rejected():
    with pytest.raises(ValueError, match="not backfillable"):
        _make(backfillable_from=2020)


def test_sources_for_cadence_rejects_an_unknown_cadence():
    with pytest.raises(ValueError, match="unknown cadence"):
        sources_for_cadence("midweek")


# --------------------------------------------------------------------------- Source validation
def _make(**overrides):
    kwargs = {
        "name": "demo_source",
        "grain": "week",
        "key_cols": ("player_id",),
        "cadence": frozenset({"prelock"}),
        "backfillable": False,
    }
    kwargs.update(overrides)
    return Source(**kwargs)


def test_source_accepts_a_valid_definition():
    assert _make().name == "demo_source"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"name": "Demo Source"}, "lower_snake_case"),
        ({"grain": "drive"}, "grain"),
        ({"key_cols": ()}, "non-empty"),
        ({"key_cols": ("player_id", "player_id")}, "duplicates"),
        ({"key_cols": ("player_id", "_captured_at")}, "exclude reserved"),
        ({"cadence": frozenset()}, "cadence must be non-empty"),
        ({"cadence": frozenset({"whenever"})}, "unknown cadence"),
        ({"cadence": frozenset({"backfill"})}, "contradicts cadence"),
        ({"backfillable": True}, "contradicts cadence"),
    ],
)
def test_source_rejects_malformed_definitions(overrides, match):
    with pytest.raises(ValueError, match=match):
        _make(**overrides)


def test_source_is_frozen_and_hashable():
    source = _make()
    assert hash(source)
    with pytest.raises(Exception):
        source.name = "renamed"  # type: ignore[misc]

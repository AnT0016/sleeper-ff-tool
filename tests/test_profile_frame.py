"""Offline tests for ``scripts/profile_frame.py`` (Phase 9, ticket #27).

``render_report`` is a pure function over a DataFrame, and its output is a **committed artifact that
later tickets make decisions from** — #28 sizes its baselines against these numbers and #30 reads
them to learn what K and DST actually carry. That makes one failure mode worse than any other: a
headline sentence that states a fact the report's own table contradicts. A reader has no way to tell
which half is right, and the numbers that are most likely to drift are exactly the ones the phase
depends on (``baseline_sleeper_points`` starts filling at 2026 W1; ``nflverse_depth`` backfills only
from 2025).

Every test below builds a frame that **contradicts** the report's expected story, and asserts the
prose follows the data rather than the expectation. A literal in the f-string passes on the real
lake today and fails silently the first time the world changes.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli(name: str):
    """Import ``scripts/<name>.py`` as a module (the CLIs are not on the import path)."""
    spec = importlib.util.spec_from_file_location(f"_cli_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


pf = _load_cli("profile_frame")

SEASONS = [2016, 2017]
WEEKS = (1, 2, 3)


def _frame(**overrides) -> pd.DataFrame:
    """A minimal cohort frame; ``overrides`` set a column to a constant or a callable of the row."""
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            for pos in pf.FANTASY_POSITIONS:
                rows.append(
                    {
                        "player_id": f"{pos}-{week}",
                        "season": season,
                        "week": week,
                        "position": pos,
                        "y_custom_points": 8.0,
                        # Null in week 1 only, mirroring the real cold-start shape.
                        "points_last": None if week == 1 else 5.0,
                    }
                )
    frame = pd.DataFrame(rows)
    for cols in pf.FEATURE_GROUPS.values():
        for column in cols:
            if column not in frame.columns:
                frame[column] = 1.0
    for column, value in overrides.items():
        frame[column] = frame.apply(value, axis=1) if callable(value) else value
    return frame


def _report(frame: pd.DataFrame, records=()) -> str:
    return pf.render_report(
        frame, SEASONS, records, scoring_keys=42, partitions=1, league_name="Test league"
    )


def _finding(report: str, n: int) -> str:
    """The nth numbered line of the "Findings, measured" list."""
    return next(line for line in report.splitlines() if line.startswith(f"{n}. **"))


def _section(report: str, heading: str) -> str:
    return report.split(f"## {heading}")[1].split("\n## ")[0]


def _table_total(section: str) -> int:
    """The bold grand total from the last cell of a ``_count_table``."""
    total_row = [line for line in section.splitlines() if line.startswith("| **")][-1]
    return int(total_row.rstrip("| ").rsplit("|", 1)[-1].strip().strip("*").replace(",", ""))


# --------------------------------------------------------------- the headline follows the data
def test_the_baseline_headline_reports_the_measured_count_not_a_literal_zero():
    """Finding #1's whole point is that the report *proves* the null baseline rather than asserting it.

    The column exists so 2026-forward rows can fill it. On the first regeneration after 2026 W1 a
    hardcoded "0 /" would have the headline denying section 6 of the same document, about the one
    fact Phase 9's evaluation strategy rests on.
    """
    frame = _frame(baseline_sleeper_points=12.5)
    report = _report(frame)

    assert f"**{len(frame):,} / {len(frame):,}**" in _finding(report, 1)
    assert "**0 /" not in report
    # ... and it agrees with the table it cites.
    assert f"**{len(frame):,}**" in _section(report, "6.")


def test_the_baseline_headline_still_reads_zero_when_the_column_really_is_empty():
    report = _report(_frame(baseline_sleeper_points=None))
    assert f"**0 / {len(_frame()):,}**" in _finding(report, 1)


def test_the_depth_headline_does_not_assert_zero_coverage_for_earlier_seasons():
    """``depth_pos_rank`` is computed for every season but only the last one was read.

    ``nflverse_depth`` is ``backfillable_from=2025`` *today*; a future backfill or a feed change
    would populate earlier seasons, and the report must notice rather than repeat the registry
    comment back at the reader.
    """
    report = _report(_frame(depth_pos_rank=1.0))  # every season fully covered
    finding = _finding(report, 2)
    assert "100.0%" in finding
    assert "**0.0%** of every earlier season" not in finding


def test_the_week_one_headline_counts_the_same_rows_the_section_it_cites_does():
    """The headline said "counted apart in section 3" while quoting a different, larger number.

    Sections 2-5 are cohort-scoped; measuring week 1 over the full frame made the two disagree by
    every IDP row (10,127 vs 4,011 on the real lake).
    """
    frame = pd.concat([_frame(), _frame().assign(position="LB")], ignore_index=True)
    report = _report(frame)

    headline = int(re.search(r"(\d[\d,]*) week-1 cohort rows", _finding(report, 3)).group(1)
                   .replace(",", ""))
    assert headline == _table_total(_section(report, "3."))
    assert headline == len(pf.FANTASY_POSITIONS) * len(SEASONS)


def test_the_non_fantasy_claim_is_measured_rather_than_asserted():
    """"Score ~0 under this scoring" is a claim about this league's keys — ``fum_rec_td`` is 6."""
    frame = pd.concat(
        [_frame(), _frame().assign(position="LB", y_custom_points=6.0)], ignore_index=True
    )
    assert "average **6.00** custom points" in _finding(frame_report := _report(frame), 4)
    assert "50.0% of rows" in _finding(frame_report, 4)


# --------------------------------------------------------------- per-position holes are named
def test_a_family_absent_for_one_position_is_named_rather_than_pooled_away():
    """Section 5's rates are pooled over positions, so a 100%-null family reads as merely sparse.

    On the real lake DST carries no snap/target/rush/expected-points column and K no
    rushing/opportunity column (``nflverse_snaps`` and ``nflverse_ff_opp`` cover offensive skill
    players), yet the pooled rates read ~17% and ~32%. #30 owns exactly those two positions.
    """
    frame = _frame()
    frame.loc[frame["position"] == "DEF", list(pf.FEATURE_GROUPS["usage_lag"])] = None
    report = _report(frame)

    assert "**DEF** is missing 13/13 usage_lag" in _finding(report, 5)
    # Section 5b shows it; section 5 alone would have averaged it down to ~17%.
    row = next(r for r in _section(report, "5b.").splitlines() if r.startswith("| snap_pct_last "))
    assert "100.0" in row
    assert float(_section(report, "5.").split("| snap_pct_last |")[1].split("|")[1]) < 50.0


def test_a_partly_absent_family_is_counted_per_column_not_all_or_nothing():
    """The real shape of the DST hole, and the one an ``all(...)`` group test cannot see.

    DST keeps ``games_played_prior`` and its own points lags while losing every snap/target/rush/
    expected-points column. A group-level "is every column absent?" gate answers *no* and reports
    nothing at all — which is how the most consequential finding in this report went missing on the
    first pass.
    """
    gone = ["snap_pct_last", "snap_pct_ewma", "target_share_last", "exp_points_last"]
    kept = ["games_played_prior", "points_last"]
    frame = _frame()
    frame.loc[frame["position"] == "DEF", gone] = None
    report = _report(frame)

    assert f"**DEF** is missing {len(gone)}/{len(pf.FEATURE_GROUPS['usage_lag'])} usage_lag" in (
        _finding(report, 5)
    )
    section = _section(report, "5b.")
    for column in gone:
        assert f"| {column} |" in section
    # The kept columns must not be swept into the count — the point is which ones survive.
    assert all(
        float(section.split(f"| {column} |")[1].split("|")[-2]) < 99.0 for column in kept
    )


def test_section_5b_prose_states_the_measured_absences_not_a_remembered_shape():
    """Section 5b's own prose is a claim about the data and has to move with it.

    Its first draft asserted "DST has no usage feature at all" — which the table beneath it
    contradicted, because DST keeps ``games_played_prior`` and its points lags. Prose and table
    disagreeing is the exact defect this whole module exists to prevent, so the sentence is
    generated from the same measurement the finding uses.
    """
    frame = _frame()
    frame.loc[frame["position"] == "K", ["rush_share_last", "exp_points_last"]] = None
    section = _section(_report(frame), "5b.")

    assert "**K** is missing 2/13 usage_lag" in section
    # Only the position that actually has a hole is named as missing anything. ("DEF" still appears
    # in the prose as #30's scope — the claim under test is the generated absence list.)
    assert "**DEF** is missing" not in section


def test_a_family_absent_for_every_position_is_left_to_the_availability_table():
    """The forward-only sources (depth, the Sleeper families, the weather forecast) are section 4's
    story. Repeating them per position would bury the finding that exists for the *asymmetric* case."""
    frame = _frame()
    for column in pf.FEATURE_GROUPS["weather_forecast"]:
        frame[column] = None
    assert "weather_forecast" not in _finding(_report(frame), 5)


def test_no_asymmetric_hole_is_reported_honestly_as_none():
    assert "No feature group is absent for any one position" in _finding(_report(_frame()), 5)


# --------------------------------------------------------------- warnings are counted, not claimed
@pytest.mark.parametrize("n_warnings", [0, 3])
def test_the_warning_count_comes_from_the_captured_log(n_warnings):
    records = [(30, "WARNING", "dataset.assemble", f"w{i}") for i in range(n_warnings)]
    report = _report(_frame(), records)
    assert f"emitted **{n_warnings}**" in _finding(report, 6)
    assert ("_None._" in _section(report, "7.")) is (n_warnings == 0)

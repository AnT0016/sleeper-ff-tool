"""Generate (and refresh) the committed offline fixture for the Sleeper collectors.

Writes ``tests/fixtures/sleeper_raw_2025_w1.json``: a small sample of **verbatim** rows from the
three Sleeper endpoints ``collect.sleeper`` wraps, stored exactly as the API returned them (nested
``player``/``stats`` objects included). The collector tests then run fully offline against real
provider shapes rather than a hand-written idealisation of them — the shapes that matter here are
the awkward ones: a DEF row keyed on a team abbreviation, a K with FG-distance buckets, and a
teamless free-agent row whose ``team``/``opponent``/``game_id``/``date`` are all null.

Run manually when refreshing the fixture (mirrors ``scripts/make_fixture.py``); the tests never call
the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from sleeper.client import get_projections, get_season_projections, get_stats

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
OUT = FIXTURES / "sleeper_raw_2025_w1.json"
SEASON = 2025
WEEK = 1
#: One row per roster position, so K's distance buckets and DEF's team-keyed rows are both covered.
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

#: The 2016 duplicate-game fixture (#21). Sleeper emits the JAX@SD game of week 2 twice — every
#: player of both teams (incl. the JAX DST) under two game_ids, byte-identical stats — so the
#: collector's identical-stat collapse can be tested offline on the real provider shape.
DUPE_OUT = FIXTURES / "sleeper_stats_2016_w2_dupe.json"
DUPE_SEASON = 2016
DUPE_WEEK = 2
DUPE_GAME_IDS = ("201610200", "201610229")


def _points(row: Mapping[str, Any]) -> float:
    return float((row.get("stats") or {}).get("pts_half_ppr") or 0.0)


def _sample(rows: Iterable[Mapping[str, Any]], *, with_teamless: bool) -> list[dict]:
    """The top row per position (ties broken by player_id), plus one teamless row if asked."""
    rows = list(rows)  # walked twice below, so a one-shot iterator would silently lose the 2nd pass
    best: dict[str, dict] = {}
    for row in rows:
        pos = (row.get("player") or {}).get("position")
        if pos not in POSITIONS:
            continue
        current = best.get(pos)
        candidate = (_points(row), str(row.get("player_id")))
        if current is None or candidate > (_points(current), str(current.get("player_id"))):
            best[pos] = dict(row)

    picked = [best[p] for p in POSITIONS if p in best]
    if with_teamless:
        # Skip any teamless row already picked as its position's best, or the fixture would carry a
        # duplicate player_id -- which the collector tests (rightly) assert against.
        chosen = {str(r.get("player_id")) for r in picked}
        teamless = sorted(
            (r for r in rows if not r.get("team")), key=lambda r: str(r.get("player_id"))
        )
        extra = next((r for r in teamless if str(r.get("player_id")) not in chosen), None)
        if extra is not None:
            picked.append(dict(extra))
    return picked


def write_2025_fixture() -> None:
    """The main collector fixture: real 2025 W1 shapes across the roster positions."""
    proj_week = _sample(get_projections(SEASON, WEEK), with_teamless=True)
    proj_season = _sample(get_season_projections(SEASON), with_teamless=True)
    stats_week = _sample(get_stats(SEASON, WEEK), with_teamless=False)

    payload = {
        "_generated_by": "scripts/make_collect_fixture.py",
        "_note": "verbatim Sleeper API rows; regenerate with the script, never hand-edit",
        "proj_week": {"season": SEASON, "week": WEEK, "rows": proj_week},
        "proj_season": {"season": SEASON, "rows": proj_season},
        "stats_week": {"season": SEASON, "week": WEEK, "rows": stats_week},
    }
    for label in ("proj_week", "proj_season", "stats_week"):
        rows = payload[label]["rows"]
        print(f"{label:12} {len(rows)} rows: " + ", ".join(str(r.get("player_id")) for r in rows))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


def write_stats_dupe_fixture() -> None:
    """The 2016 duplicate-game fixture (#21): the whole JAX@SD game the feed emits twice.

    Verbatim rows whose ``game_id`` is either half of the artifact pair — 34 players × 2, byte-
    identical stats — so ``collect_stats_week``'s identical-stat collapse runs on the real shape.
    """
    rows = [r for r in get_stats(DUPE_SEASON, DUPE_WEEK) if str(r.get("game_id")) in DUPE_GAME_IDS]
    payload = {
        "_generated_by": "scripts/make_collect_fixture.py (write_stats_dupe_fixture)",
        "_note": "verbatim Sleeper API rows for the #21 duplicate-game artifact; never hand-edit",
        "season": DUPE_SEASON,
        "week": DUPE_WEEK,
        "game_id_pair": list(DUPE_GAME_IDS),
        "rows": rows,
    }
    DUPE_OUT.parent.mkdir(parents=True, exist_ok=True)
    # LF explicitly: .gitattributes stores *.json as eol=lf, so emit it that way on Windows too.
    DUPE_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {DUPE_OUT} ({len(rows)} rows, {DUPE_OUT.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    write_2025_fixture()
    write_stats_dupe_fixture()


if __name__ == "__main__":
    main()

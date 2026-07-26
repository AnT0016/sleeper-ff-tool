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

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sleeper_raw_2025_w1.json"
SEASON = 2025
WEEK = 1
#: One row per roster position, so K's distance buckets and DEF's team-keyed rows are both covered.
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


def _points(row: Mapping[str, Any]) -> float:
    return float((row.get("stats") or {}).get("pts_half_ppr") or 0.0)


def _sample(rows: Iterable[Mapping[str, Any]], *, with_teamless: bool) -> list[dict]:
    """The top row per position (ties broken by player_id), plus one teamless row if asked."""
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
        teamless = sorted(
            (r for r in rows if not r.get("team")), key=lambda r: str(r.get("player_id"))
        )
        if teamless:
            picked.append(dict(teamless[0]))
    return picked


def main() -> None:
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


if __name__ == "__main__":
    main()

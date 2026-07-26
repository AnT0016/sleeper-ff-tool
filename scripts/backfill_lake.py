"""One-time historical pull: every backfillable source, 2016-2025, into the lake (Phase 8).

    ./.venv/Scripts/python scripts/backfill_lake.py
    ./.venv/Scripts/python scripts/backfill_lake.py --seasons 2024-2025
    ./.venv/Scripts/python scripts/backfill_lake.py --seasons 2020 --sources nflverse_snaps,weather

This is the training half of the lake. It pulls what nflverse publishes as versioned releases
(actuals, snaps, opportunity, injuries, schedules, depth, the id crosswalk) plus the sources derived
from them (Vegas, weather) and Sleeper's finalized weekly stats. It deliberately cannot pull
``sleeper_proj_*``: those endpoints serve only the latest numbers, so pre-lock projections exist
**forward** from the first ``collect.py --mode prelock`` run and nowhere else.

Every row it writes carries ``_backfill=True``. That marker is load-bearing rather than cosmetic —
a backfilled row was reconstructed *after* the fact, so anything on it that comes from a mutable
provider master (a player's position, most sharply) is today's value rather than that week's, and
the assembler has to be able to tell the two apart. See ``collect.runner`` and ``collect.sleeper``.

Long-running by design (a full 2016-2025 run downloads every nflverse release), and safe to re-run:
the store merges point-in-time rather than appending, so a repeated same-day run is idempotent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

# Render UTF-8 regardless of the console code page (Windows defaults to cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# scripts/ goes to the front of sys.path when this file is run directly, and scripts/collect.py
# shadows the `collect` package the runner lives in. Drop the script directory. (The truthiness
# check keeps an empty entry — which means the cwd — from matching when the cwd is scripts/.)
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from collect import runner
from collect.registry import backfillable_sources
from store.lake import LAKE_BACKEND

#: The span resolved in the spec (decision log, Q-B): ten seasons of training data.
DEFAULT_SEASONS = "2016-2025"


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Backfill the lake's historical, recoverable sources")
    ap.add_argument(
        "--seasons",
        default=DEFAULT_SEASONS,
        help=f"season or range, e.g. 2016-2025 / 2019 / 2016,2018-2020 (default: {DEFAULT_SEASONS})",
    )
    ap.add_argument(
        "--sources",
        default=None,
        help="comma-separated subset; default is every backfillable source "
        f"({','.join(s.name for s in backfillable_sources())})",
    )
    args = ap.parse_args(argv)

    try:
        seasons = runner.parse_seasons(args.seasons)
        sources = runner.parse_sources(args.sources, backfill=True)
    except ValueError as exc:
        print(f"bad arguments — {exc}", file=sys.stderr)
        return 2

    captured_at = datetime.now(timezone.utc).isoformat()
    print(
        f"Backfilling {seasons[0]}-{seasons[-1]} ({len(seasons)} season(s)) — "
        f"{'all backfillable sources' if sources is None else ','.join(sources)} "
        f"(captured_at {captured_at}, backend {LAKE_BACKEND})"
    )
    try:
        results = runner.run_backfill(seasons, captured_at=captured_at, sources=sources)
    except ValueError as exc:
        # Raised only by the plan (a --sources subset that reaches none of --seasons); a collector
        # that raises is caught per task and never reaches here.
        print(f"nothing to collect - {exc}", file=sys.stderr)
        return 2

    print(runner.format_summary(results))
    return runner.exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())

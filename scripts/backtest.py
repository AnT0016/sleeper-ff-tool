"""Build the full-season "what if I'd used this tool" backtest for a (completed or in-progress) season.

    ./.venv/Scripts/python scripts/backtest.py                       # current season by default
    ./.venv/Scripts/python scripts/backtest.py --season 2025         # a specific past season
    ./.venv/Scripts/python scripts/backtest.py --season 2026 --league <2026_id>

Writes ``data_cache/backtest.db`` — a separate artifact from the live ``season.db`` (so the weekly
refresh never clobbers it). The dashboard's "Backtest" tab reads it. Read-only; scores every lineup by
real results (only weeks that already finished are included, so it's safe to run mid-season). See
``src/analysis/backtest.py``. For a *new* season, point ``--league`` at that season's league id (or
update ``LEAGUE_ID`` in src/sleeper/config.py once the new league exists).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from analysis.backtest import build_and_write
from sleeper import client
from sleeper.config import LEAGUE_ID, MY_USER_ID

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data_cache" / "backtest.db"


def _default_season() -> int:
    """The current Sleeper season (the one to backtest by default); falls back to 2025 offline."""
    try:
        return int(client.get_state().get("season") or 2025)
    except Exception:
        return 2025


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build a season backtest artifact")
    ap.add_argument("--season", type=int, default=None, help="season to backtest (default: current)")
    ap.add_argument("--league", default=LEAGUE_ID)
    ap.add_argument("--user", default=MY_USER_ID)
    ap.add_argument("--out", default=str(_DEFAULT_DB), help="output SQLite path")
    args = ap.parse_args()
    season = args.season or _default_season()

    print(f"Building {season} backtest (league {args.league}) → {args.out}")
    print("  (first run makes ~50 cached Sleeper calls: matchups/projections/stats/transactions × weeks)")
    out = build_and_write(args.league, args.user, season, db_path=args.out)
    print(f"✓ wrote {out}")


if __name__ == "__main__":
    main()

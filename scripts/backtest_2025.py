"""Build the full-season "what if I'd used this tool" backtest for a completed season.

    ./.venv/Scripts/python scripts/backtest_2025.py                 # 2025 by default
    ./.venv/Scripts/python scripts/backtest_2025.py --season 2024

Writes ``data_cache/backtest.db`` — a separate artifact from the live ``season.db`` (so the weekly
refresh never clobbers it). The dashboard's "2025 Backtest" tab reads it. Read-only; uses only
completed-season data (real matchup points, the real draft). See ``src/analysis/backtest.py``.
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
from sleeper.config import LEAGUE_ID, MY_USER_ID

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data_cache" / "backtest.db"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build a completed-season backtest artifact")
    ap.add_argument("--season", type=int, default=2025, help="completed season to backtest")
    ap.add_argument("--league", default=LEAGUE_ID)
    ap.add_argument("--user", default=MY_USER_ID)
    ap.add_argument("--out", default=str(_DEFAULT_DB), help="output SQLite path")
    args = ap.parse_args()

    print(f"Building {args.season} backtest (league {args.league}) → {args.out}")
    print("  (first run makes ~50 cached Sleeper calls: matchups/projections/stats × weeks)")
    out = build_and_write(args.league, args.user, args.season, db_path=args.out)
    print(f"✓ wrote {out}")


if __name__ == "__main__":
    main()

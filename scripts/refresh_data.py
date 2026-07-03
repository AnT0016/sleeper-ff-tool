"""Refresh the hosted season dashboard's data artifact (the weekly GitHub Actions job).

    ./.venv/Scripts/python scripts/refresh_data.py            # current season + week, auto-detected
    ./.venv/Scripts/python scripts/refresh_data.py --week 5
    ./.venv/Scripts/python scripts/refresh_data.py --week 15 --season 2025

Runs the ingest + recompute pipeline (``analysis.snapshot``) and writes ``data_cache/season.db`` --
the single SQLite artifact the offline Streamlit dashboard reads. GitHub Actions commits the updated
file back to the repo, and Streamlit Community Cloud auto-redeploys on that commit. Read-only against
Sleeper; it never writes to the league.
"""

from __future__ import annotations

import argparse
import logging
import sys

# Render UTF-8 regardless of the console code page (Windows defaults to cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from analysis.snapshot import DEFAULT_DB, build_and_write, offseason_skip_reason
from sleeper import client
from sleeper.config import LEAGUE_ID, MY_USER_ID


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Refresh the season dashboard data artifact")
    ap.add_argument("--week", type=int, default=None, help="NFL week (1-18); defaults to current")
    ap.add_argument("--season", type=int, default=None, help="defaults to the current season")
    ap.add_argument("--league", default=LEAGUE_ID)
    ap.add_argument("--user", default=MY_USER_ID)
    ap.add_argument("--out", default=str(DEFAULT_DB), help="output SQLite path")
    args = ap.parse_args()

    try:
        state = client.get_state()
    except Exception:
        state = {}
    try:
        league = client.get_league(args.league)
    except Exception:
        league = {}

    # A fully-auto (scheduled) run must never guess: with no reachable state there is no way to know
    # the season/week, and building a wrong-week snapshot would be silently committed. Fail red so
    # the cron surfaces the problem; a manual backfill can always pass --week/--season explicitly.
    if not state and args.week is None and args.season is None:
        sys.exit("could not determine the current season/week (Sleeper state unreachable) — "
                 "aborting the scheduled refresh; pass --week/--season to run anyway")

    # Off-season / rollover / post-championship no-op: exit 0 (not a failure) so the weekly cron
    # stays green; it resumes on its own once a *current-season* league is configured and playing.
    reason = offseason_skip_reason(state, args.week, args.season, league)
    if reason:
        print(f"Skipping refresh — {reason}.")
        return

    season = args.season or int(state["season"])
    week = args.week or max(1, int(state.get("week") or 0))

    print(f"Refreshing season snapshot — {season} Week {week} (league {args.league}) → {args.out}")
    out = build_and_write(args.league, args.user, season, week, db_path=args.out)
    print(f"✓ wrote {out}")


if __name__ == "__main__":
    main()

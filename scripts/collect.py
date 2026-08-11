"""Capture one point-in-time snapshot of the lake — the scheduled cron entry point (Phase 8).

    ./.venv/Scripts/python scripts/collect.py --mode prelock     # Thu/Sun, before kickoff
    ./.venv/Scripts/python scripts/collect.py --mode postgame    # Tue, once the week is final
    ./.venv/Scripts/python scripts/collect.py --mode postgame --season 2026 --week 3

``prelock`` collects the sources whose value is that they were captured *before* the outcome existed
(Sleeper's weekly/season projections, the injury report, the market, the weather forecast);
``postgame`` collects the finalized actuals and usage. Which sources those are is read from
``collect.registry`` — this script never names one.

Season and week come from Sleeper's own state unless overridden, and the run no-ops (exit 0) outside
the regular season, matching ``refresh.yml``'s behaviour so the crons agree about when the season is
over. One failing collector is reported and the rest still run; the exit status is non-zero only if
every source failed. Read-only against Sleeper, as always.

Where the rows land is ``LAKE_BACKEND`` (``local`` by default, ``s3`` in the cloud cron) — this
script neither knows nor cares.
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

# `python scripts/collect.py` puts scripts/ at the FRONT of sys.path, where this file shadows the
# installed `collect` package — the one holding the runner below, and the one backfill_lake.py
# imports. Drop the script directory: nothing in scripts/ imports a sibling script.
# The `sys.path[0]` truthiness check matters: an empty entry means "the cwd", and resolving it would
# match this test whenever the cwd happens to be scripts/ — dropping the cwd rather than the script
# directory.
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

from collect import runner
from sleeper import client
from store.lake import LAKE_BACKEND


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Capture a point-in-time snapshot into the lake")
    ap.add_argument(
        "--mode",
        required=True,
        choices=("prelock", "postgame"),
        help="which cadence to collect (the registry decides which sources that is)",
    )
    ap.add_argument("--season", type=int, default=None, help="defaults to the current season")
    ap.add_argument("--week", type=int, default=None, help="NFL week (1-18); defaults to current")
    args = ap.parse_args(argv)

    try:
        state = client.get_state()
    except Exception as exc:
        print(f"warning: could not read Sleeper state ({type(exc).__name__}: {exc})")
        state = {}

    # One clock reading for the whole run: it dates the capture *and* decides, for a postgame run,
    # which NFL week has finished (plan_run derives the completed week from the schedule as of now).
    now = datetime.now(timezone.utc)
    # One RunContext, shared by plan_run and run_cadence, so the season schedule a postgame plan
    # reads is the same object the nflverse_schedules collector writes — loaded at most once.
    ctx = runner.RunContext()

    try:
        # ImportError too: plan_run imports analysis.snapshot lazily (it drags in the whole
        # optimizer/pulp stack), so a broken install there would otherwise escape as a traceback
        # rather than the actionable one-liner a cron log wants.
        plan = runner.plan_run(
            state, mode=args.mode, now=now, season=args.season, week=args.week, ctx=ctx
        )
    except (ValueError, ImportError) as exc:
        # A scheduled run must never guess the week: the forward-only sources would be filed under
        # the wrong partition, and that week is then gone for good. Fail red so the cron surfaces it.
        print(f"aborting the {args.mode} capture — {exc}", file=sys.stderr)
        return 1

    if plan.skip:
        # Exit 0, not a failure: the cron stays green through the off-season and resumes on its own.
        print(f"Skipping the {args.mode} capture — {plan.skip}.")
        return 0

    captured_at = now.isoformat()
    scope = "" if plan.sources is None else f", forward-only sources {list(plan.sources)}"
    print(
        f"{args.mode} capture — {plan.season} Week {plan.week} "
        f"(captured_at {captured_at}, backend {LAKE_BACKEND}{scope})"
    )
    results = runner.run_cadence(
        args.mode, plan.season, plan.week, captured_at=captured_at, sources=plan.sources, ctx=ctx
    )
    print(runner.format_summary(results))
    return runner.exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())

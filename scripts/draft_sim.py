"""Forward Monte Carlo draft simulator (Phase 6, optional) — directional draft-prep aid.

    ./.venv/Scripts/python scripts/draft_sim.py --slot 7
    ./.venv/Scripts/python scripts/draft_sim.py --slot 4 --sims 4000 --season 2025
    ./.venv/Scripts/python scripts/draft_sim.py --strategies best_vor,zero_rb,hero_rb

Simulates thousands of 12-team snake drafts: 11 bots draft by market ADP (+ noise), I draft by our
custom VOR under each roster-build strategy, and every sim draws each player's season outcome (with
injuries) to score my roster's best lineup. Prints, per build, the *distribution* of season outcomes
and finish, the probability each target survives to my picks, and where I need a real backup.

Read-only. Output is directional — every assumption is printed alongside the numbers. If your slot
isn't revealed yet, pass --slot to test a specific one (CLAUDE.md keeps prep slot-agnostic until then).
"""

from __future__ import annotations

import argparse
import sys

from draftsim.engine import simulate
from draftsim.inputs import load_sim_inputs
from draftsim.report import format_report
from draftsim.strategy import STRATEGIES
from sleeper import client
from sleeper.config import LEAGUE_ID, MY_USER_ID


def _default_season() -> int:
    try:
        return int(client.get_state().get("season") or 2025)
    except Exception:
        return 2025


def main() -> None:
    ap = argparse.ArgumentParser(description="Monte Carlo draft simulator (directional)")
    ap.add_argument("--slot", type=int, default=None, help="my draft slot (else revealed/fallback)")
    ap.add_argument("--sims", type=int, default=2000, help="number of simulated drafts")
    ap.add_argument("--season", type=int, default=None, help="projection season (default: current)")
    ap.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help=f"comma-separated subset of: {', '.join(STRATEGIES)}",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=300, help="top-N by VOR kept draftable")
    ap.add_argument("--league", default=LEAGUE_ID)
    ap.add_argument("--user", default=MY_USER_ID)
    args = ap.parse_args()

    season = args.season or _default_season()
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in STRATEGIES]
    if bad:
        ap.error(f"unknown strateg(ies) {bad}; choose from {list(STRATEGIES)}")

    inp = load_sim_inputs(args.league, args.user, season, slot=args.slot)
    out = simulate(
        inp.board,
        inp.cfg,
        inp.my_slot,
        n_sims=args.sims,
        strategies=strategies,
        seed=args.seed,
        pool_size=args.pool_size,
    )

    # Windows consoles default to cp1252; the report uses ·/→/⚠/← etc.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print()
    print(format_report(out, inp.slot_source, season))
    print()


if __name__ == "__main__":
    main()

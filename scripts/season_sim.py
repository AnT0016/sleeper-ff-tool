"""Full-season championship Monte Carlo (Phase 7, optional) — turns draft-capital into title odds.

    ./.venv/Scripts/python scripts/season_sim.py                         # my current league
    ./.venv/Scripts/python scripts/season_sim.py --league <ID> --season 2024   # validate a past season
    ./.venv/Scripts/python scripts/season_sim.py --sims 5000 --opp-noise 0.6

Simulates thousands of full seasons for the twelve rosters as they stand: draws every player's weekly
points (with multi-week injuries), sets each team's lineup, resolves the real head-to-head schedule and
the Weeks 15-17 bracket, and reports my championship / playoff odds, the value of the weekly optimizer
(the start/sit edge), and every team's title odds. For a **completed** season the league board shows
the sim's odds next to the real final standings + champion, so the model can be calibrated by eye.

Read-only. Output is directional — every variance / injury / skill assumption is printed alongside it.
"""

from __future__ import annotations

import argparse
import sys

from seasonsim.engine import DEFAULT_MY_NOISE, DEFAULT_OPP_NOISE, simulate_season
from seasonsim.inputs import load_season_inputs
from seasonsim.report import format_report
from sleeper.config import LEAGUE_ID, MY_USER_ID


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-season championship Monte Carlo (directional)")
    ap.add_argument("--league", default=LEAGUE_ID, help="Sleeper league id")
    ap.add_argument("--season", type=int, default=None, help="projection season (default: league's)")
    ap.add_argument("--user", default=MY_USER_ID, help="my Sleeper user id (whose team is 'mine')")
    ap.add_argument("--sims", type=int, default=2000, help="number of simulated seasons")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opp-noise", type=float, default=DEFAULT_OPP_NOISE, help="rivals' start/sit noise")
    ap.add_argument("--my-noise", type=float, default=DEFAULT_MY_NOISE, help="my start/sit noise")
    args = ap.parse_args()

    inp = load_season_inputs(args.league, args.season, user_id=args.user)
    out = simulate_season(
        inp.pool,
        inp.schedule,
        inp.my_team,
        season=inp.season,
        regular_weeks=inp.regular_weeks,
        playoff_weeks=inp.playoff_weeks,
        n_playoff_teams=inp.n_playoff_teams,
        n_sims=args.sims,
        seed=args.seed,
        opp_noise=args.opp_noise,
        my_noise=args.my_noise,
    )

    # Windows consoles default to cp1252; the report uses ·/→/★/← etc.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print()
    print(format_report(out, inp))
    print()


if __name__ == "__main__":
    main()

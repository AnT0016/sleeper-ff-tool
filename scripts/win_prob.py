"""Weekly win-probability start/sit — how likely you are to win *this* week, and the leverage plays.

    ./.venv/Scripts/python scripts/win_prob.py --week 10
    ./.venv/Scripts/python scripts/win_prob.py --week 10 --season 2025 --sims 40000

Simulates your projection-optimal lineup against this week's real opponent (each player a weekly
lognormal draw around its custom-scored projection) to report P(win), your score distribution, and the
strategic posture (favor floor when favored, ceiling when not). Then it re-grades each start/sit call
by its effect on **win probability** — occasionally a lower-projected bench player *raises* your odds
(upside when you're an underdog), the leverage a points-only optimizer can't see. Read-only.
"""

from __future__ import annotations

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from optimizer.inputs import (
    assemble_players,
    bye_teams,
    find_my_roster,
    opponent_roster,
    score_projections,
)
from optimizer.lineup import lineup_slots, optimize
from optimizer.winprob import leverage_note, startsit_leverage, win_probability
from sleeper import client
from sleeper.config import LEAGUE_ID, MY_USER_ID


def _default_season() -> int:
    try:
        return int(client.get_state().get("season") or 2025)
    except Exception:
        return 2025


def _team_name(users: dict, owner_id) -> str:
    u = users.get(owner_id, {})
    return (u.get("metadata") or {}).get("team_name") or u.get("display_name") or "opponent"


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly win-probability start/sit")
    ap.add_argument("--week", type=int, required=True, help="upcoming NFL week (1-18)")
    ap.add_argument("--season", type=int, default=None, help="defaults to the current season")
    ap.add_argument("--league", default=LEAGUE_ID)
    ap.add_argument("--user", default=MY_USER_ID)
    ap.add_argument("--sims", type=int, default=20000, help="Monte Carlo iterations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    season = args.season or _default_season()

    league = client.get_league(args.league)
    scoring = league["scoring_settings"]
    slots = lineup_slots(league.get("roster_positions") or [])
    rosters = client.get_rosters(args.league)
    users = {u["user_id"]: u for u in client.get_users(args.league)}
    players_map = client.get_players_nfl()
    scored = score_projections(client.get_projections(season, args.week), scoring)
    byes = bye_teams(season, args.week)

    my_roster = find_my_roster(rosters, args.user)
    my_players, _ = assemble_players(my_roster, players_map, scored, byes)
    my_sol = optimize(my_players, slots)

    opp = opponent_roster(client.get_matchups(args.league, args.week), rosters, int(my_roster["roster_id"]))
    if not opp:
        print(f"\nNo head-to-head opponent set for {season} Week {args.week} — can't compute win prob.")
        return
    opp_players, _ = assemble_players(opp, players_map, scored, byes)
    opp_sol = optimize(opp_players, slots)
    opp_name = _team_name(users, opp.get("owner_id"))

    my_means = [sp.player.proj_pts for sp in my_sol.starters]
    my_pos = [sp.player.pos for sp in my_sol.starters]
    opp_means = [sp.player.proj_pts for sp in opp_sol.starters]
    opp_pos = [sp.player.pos for sp in opp_sol.starters]

    wp = win_probability(my_means, my_pos, opp_means, opp_pos, n_sims=args.sims, seed=args.seed)
    label, note = leverage_note(wp.p_win)
    _, swaps = startsit_leverage(my_players, slots, opp_means, opp_pos, n_sims=args.sims, seed=args.seed)

    print(f"\n=== Win probability — {season} Week {args.week} vs {opp_name} ===")
    print(f"  P(win): {wp.p_win:.0%}   posture: {label.upper()}")
    print(f"  You: proj {my_sol.total:.1f} (sim {wp.my_mean:.1f}; floor p10 {wp.my_p10:.0f} / "
          f"ceiling p90 {wp.my_p90:.0f})   {opp_name}: proj {opp_sol.total:.1f}")
    print(f"  {note}")

    movers = [s for s in swaps if abs(s.delta_winprob) >= 0.003]
    print("\n--- Start/sit calls re-graded by win probability ---")
    if not movers:
        print("  Your projection-optimal lineup is also win-prob optimal — no swap moves the needle.")
    for s in movers:
        arrow = "↑ START" if s.delta_winprob > 0 else "keep"
        print(
            f"  [{arrow:<7}] {s.bench.name} ({s.bench.pos}) over {s.starter.name} ({s.slot}): "
            f"P(win) {s.delta_winprob:+.1%}  (projection {s.delta_proj:+.1f})"
        )
    print("  (↑ = starting the bench player RAISES your win odds — upside when you're an underdog, "
          "even at a lower projection. Δproj is the points cost.)")
    print()


if __name__ == "__main__":
    main()

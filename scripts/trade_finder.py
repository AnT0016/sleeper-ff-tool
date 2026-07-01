"""Player-for-player trade finder — win-win 1-for-1 skill swaps across the league.

    ./.venv/Scripts/python scripts/trade_finder.py
    ./.venv/Scripts/python scripts/trade_finder.py --season 2025 --min-gain 5 --top 20

Scans every 1-for-1 skill swap with every other team and keeps the ones where **both** teams' best
starting lineup improves (season-long, in our scoring) — the deals a partner has a real reason to
accept. Value is each side's optimal-lineup change; K/DEF are excluded (streamed, not traded).
Read-only; it never proposes or executes anything on Sleeper.
"""

from __future__ import annotations

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from analysis.snapshot import _season_lineup_players, team_names_by_roster
from analysis.trades import find_trades
from optimizer.inputs import find_my_roster
from optimizer.lineup import lineup_slots
from projections.board import build_board
from sleeper import client
from sleeper.config import LEAGUE_ID, MY_USER_ID


def _default_season() -> int:
    try:
        return int(client.get_state().get("season") or 2025)
    except Exception:
        return 2025


def main() -> None:
    ap = argparse.ArgumentParser(description="Player-for-player trade finder (win-win 1-for-1)")
    ap.add_argument("--season", type=int, default=None, help="projection season (default: current)")
    ap.add_argument("--league", default=LEAGUE_ID)
    ap.add_argument("--user", default=MY_USER_ID)
    ap.add_argument("--min-gain", type=float, default=1.0, help="min season-point gain for BOTH sides")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    season = args.season or _default_season()

    league = client.get_league(args.league)
    scoring = league["scoring_settings"]
    slots = lineup_slots(league.get("roster_positions") or [])
    rosters = client.get_rosters(args.league)
    names = team_names_by_roster(rosters, client.get_users(args.league))
    players_map = client.get_players_nfl()
    season_scored = {
        r.player_id: {"proj": r.proj_pts, "pos": r.pos, "team": r.team, "name": r.name}
        for r in build_board(season, scoring)
    }

    my_rid = int(find_my_roster(rosters, args.user)["roster_id"])
    by_team = {
        int(r["roster_id"]): _season_lineup_players(r, players_map, season_scored) for r in rosters
    }
    offers = find_trades(
        by_team[my_rid],
        {rid: pls for rid, pls in by_team.items() if rid != my_rid},
        slots, names, min_gain=args.min_gain, top=args.top,
    )

    print(f"\n=== Trade finder — {season} (win-win 1-for-1, ≥ {args.min_gain:.0f} season pts each side) ===")
    print(f"  {names.get(my_rid, 'me')} · scanned {len(by_team) - 1} teams\n")
    if not offers:
        print("  No win-win 1-for-1 fits found — try a lower --min-gain, or the deals are all one-sided.")
    for o in offers:
        print(
            f"  {o.partner:<22}  GIVE {o.give_name} ({o.give_pos})  →  GET {o.get_name} ({o.get_pos})"
        )
        print(f"          your lineup +{o.my_gain:.1f}   ·   their lineup +{o.their_gain:.1f}  (season pts)")
    print()


if __name__ == "__main__":
    main()

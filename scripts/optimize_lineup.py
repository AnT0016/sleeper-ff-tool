"""Weekly lineup optimizer -- run it for a given week against live (cached) data.

    ./.venv/Scripts/python scripts/optimize_lineup.py --week 5
    ./.venv/Scripts/python scripts/optimize_lineup.py --week 15 --season 2025

Prints the optimal legal starting lineup + projected total, a start/sit table (each bench player's
delta vs. the starter they'd replace), risky-start flags, and the players idle this week (bye/OUT/IR).
Read-only; it never writes to the league.
"""

from __future__ import annotations

import argparse

from optimizer.inputs import load_lineup_inputs
from optimizer.lineup import optimize
from optimizer.startsit import idle_players, risky_starts, start_sit_table
from sleeper import client
from sleeper.config import LEAGUE_ID, MY_USER_ID


def _default_season() -> int:
    try:
        return int(client.get_state().get("season") or 2025)
    except Exception:
        return 2025


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly lineup optimizer")
    ap.add_argument("--week", type=int, required=True, help="NFL week (1-18)")
    ap.add_argument("--season", type=int, default=None, help="defaults to the current season")
    ap.add_argument("--league", default=LEAGUE_ID)
    ap.add_argument("--user", default=MY_USER_ID)
    args = ap.parse_args()
    season = args.season or _default_season()

    inp = load_lineup_inputs(args.league, args.user, season, args.week)
    sol = optimize(inp.players, inp.slots)

    slot_label = " · ".join(f"{n}x{s}" for s, n in inp.slots.items())
    print(f"\n=== Optimal lineup — {season} Week {args.week} ===")
    print(f"slots: {slot_label}\n")
    for sp in sol.starters:
        p = sp.player
        tag = f"  [{p.status}]" if p.status else ""
        print(f"  {sp.slot:<5} {p.name:<26} {p.pos:<3} {p.team or '':<4} {p.proj_pts:6.2f}{tag}")
    print(f"\n  PROJECTED TOTAL: {sol.total:.2f}  (status: {sol.status})")
    if sol.holes:
        holes = ", ".join(f"{n}x {s}" for s, n in sol.holes.items())
        print(f"  ⚠ UNFILLED SLOTS (no eligible player): {holes}")

    flags = risky_starts(sol, inp.players)
    if flags:
        print("\n--- Risky starts ---")
        for f in flags:
            print(f"  {f.slot:<5} {f.player.name:<26} {' ; '.join(f.reasons)}")

    print("\n--- Start/sit (bench delta vs. the starter they'd replace) ---")
    for d in start_sit_table(sol):
        if d.would_replace is None:
            print(f"  {d.player.name:<26} {d.player.pos:<3} {d.player.proj_pts:6.2f}   (no eligible slot)")
        else:
            print(
                f"  {d.player.name:<26} {d.player.pos:<3} {d.player.proj_pts:6.2f}   "
                f"Δ{d.delta:+6.2f} vs {d.slot} {d.would_replace.name}"
            )

    idle = idle_players(inp.players)
    if idle:
        print("\n--- Idle this week (bye / OUT / IR) ---")
        for p, reason in idle:
            print(f"  {p.name:<26} {p.pos:<3} {p.team or '':<4} {p.proj_pts:6.2f}   {reason}")

    if inp.unjoined:
        print("\n--- No projection (scored 0.0; check ID join) ---")
        for pid, name in inp.unjoined:
            print(f"  {name} ({pid})")
    print()


if __name__ == "__main__":
    main()

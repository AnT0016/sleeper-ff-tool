"""Weekly waiver / stash / handcuff report -- run it against live (cached) data.

    ./.venv/Scripts/python scripts/waiver_report.py --week 10
    ./.venv/Scripts/python scripts/waiver_report.py --week 15 --season 2025

Built for **Tuesday evening CEST**, before waivers clear Wednesday 09:00 CEST. This league uses
reverse-standings waiver **priority** (a single ordered claim), NOT FAAB -- so the spend advice
reasons about whether a target is worth burning my claim, never about dollars. Prints, in order:
handcuff / injury-replacement alerts, reverse-priority spend advice, the playoff (Weeks 15-17) stash
ranker, and bye-week stash suggestions. Read-only; it never writes to the league.
"""

from __future__ import annotations

import argparse
import sys

# Render UTF-8 regardless of the console code page (Windows defaults to cp1252, which chokes on the
# arrows / boxes used below).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sleeper import client
from sleeper.config import LEAGUE_ID, MY_USER_ID
from waivers.inputs import load_waiver_inputs
from waivers.priority import spend_advice
from waivers.stash import bye_stash_suggestions, rank_playoff_stashes
from waivers.streaming import rank_streamers


def _default_season() -> int:
    try:
        return int(client.get_state().get("season") or 2025)
    except Exception:
        return 2025


def _usage(inp, pid: str) -> str:
    sig = inp.usage.get(pid)
    s = sig.summary() if sig else ""
    return f"   [{s}]" if s else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly waiver / stash / handcuff report")
    ap.add_argument("--week", type=int, required=True, help="upcoming NFL week (1-18)")
    ap.add_argument("--season", type=int, default=None, help="defaults to the current season")
    ap.add_argument("--league", default=LEAGUE_ID)
    ap.add_argument("--user", default=MY_USER_ID)
    ap.add_argument("--stash-top", type=int, default=15, help="how many playoff stashes to list")
    args = ap.parse_args()
    season = args.season or _default_season()

    inp = load_waiver_inputs(args.league, args.user, season, args.week)

    print(f"\n=== Waiver report — {season} Week {args.week} ===")
    pos = f"#{inp.waiver_position}" if inp.waiver_position is not None else "n/a"
    print(
        f"standings: #{inp.my_rank} of {inp.n_teams}   waiver priority: {pos}   "
        f"posture: {inp.scarcity.posture.upper()}"
    )
    print(f"  ({inp.scarcity.note})")

    # 1) Handcuff / injury-replacement alerts ---------------------------------------------------
    print("\n--- Handcuff / injury-replacement alerts ---")
    if not inp.handcuffs:
        print("  (none — every starter's next man up is already rostered)")
    for a in inp.handcuffs:
        flag = "** URGENT" if a.priority == "URGENT" else " - HIGH  "
        print(f"  {flag}  add {a.backup_name:<22} ({a.pos} {a.team}){_usage(inp, a.backup_id)}")
        print(f"            {a.reason}")

    # 2) Reverse-priority spend advice ----------------------------------------------------------
    advice = spend_advice(inp.spend_candidates, inp.my_players, inp.slots, inp.scarcity)
    print("\n--- Reverse-priority spend advice (spend a top claim only on real upgrades) ---")
    shown = [a for a in advice if a.verdict != "hold"] or advice[:5]
    for a in shown:
        verdict = {"spend": "SPEND", "stream-later": "stream", "hold": "hold"}[a.verdict]
        cont = f"{a.contention_level}" + (f"/{a.contention}" if a.contention else "")
        print(
            f"  [{verdict:<6}] {a.name:<22} {a.pos:<3} {a.team or '':<4} "
            f"Δlineup {a.lineup_gain:+5.2f}  contention {cont}{_usage(inp, a.player_id)}"
        )
        print(f"            {a.reason}")

    # 2b) Weekly K/DEF streaming guide ----------------------------------------------------------
    streamers = rank_streamers(inp.stream_candidates, inp.stream_current)
    print("\n--- Weekly K/DEF streaming guide (best available this week, in our scoring) ---")
    for adv in streamers:
        cur = f"{adv.current_name} {adv.current_this_week:.1f}" if adv.current_name else "(none rostered)"
        tag = "STREAM" if adv.verdict == "stream" else " hold "
        print(f"  [{tag}] {adv.pos}  — current starter: {cur}")
        for o in adv.options:
            print(
                f"          {o.name:<22} {o.team or '':<4} this {o.this_week:5.1f} (Δ{o.gain:+.1f})   "
                f"next {o.next_week:5.1f}   ROS/g {o.ros_pg:4.1f}   playoff {o.playoff:5.1f}"
            )
    print("  (Δ = this-week edge over your current starter. K carries no SOS — kicker output is "
          "driven by its own offense, not the opponent.)")

    # 3) Playoff (Weeks 15-17) stash ranker -----------------------------------------------------
    stashes = rank_playoff_stashes(inp.stash_candidates, inp.sos, inp.opponents_by_week)
    print(f"\n--- Playoff stash ranker (Weeks 15-17, SOS-adjusted in our scoring) — top {args.stash_top} ---")
    for s in stashes[: args.stash_top]:
        wk = "  ".join(f"W{w.week} {w.opponent}×{w.multiplier:.2f}" for w in s.weeks)
        print(
            f"  {s.name:<22} {s.pos:<3} {s.team or '':<4} "
            f"value {s.adj_value:6.2f} (raw {s.raw_value:.2f}, SOS {s.sos_swing:+.2f}){_usage(inp, s.player_id)}"
        )
        if wk:
            print(f"            {wk}")

    # 4) Bye-week stash suggestions -------------------------------------------------------------
    byes = bye_stash_suggestions(
        inp.my_starters, inp.bye_week_of_team, inp.stash_candidates, from_week=args.week + 1
    )
    if byes:
        print("\n--- Upcoming starter byes (stash a fill-in now) ---")
        for b in byes:
            sugg = ", ".join(f"{n} ({v:.1f})" for n, v in b.suggestions) or "no clear FA"
            print(f"  {b.reason}  →  {sugg}")

    print()


if __name__ == "__main__":
    main()

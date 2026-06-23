"""Custom-scoring season validation -- the Phase 1 gate.

Re-scores last season in OUR league's live ``scoring_settings`` and compares, per player, against:
  1. Sleeper's reported matchup points (engine + settings correctness, ALL positions incl K/DEF), and
  2. an independent re-score of nflverse weekly actuals (skill + K; DEF n/a -- team aggregates
     aren't in nflverse player stats).

Both should track Sleeper's reported points within rounding. Prints a table for a sample of ~20
players spread across positions. Network-heavy but fully cached; safe to re-run.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import polars as pl

from data.ids import build_gsis_to_sleeper, nflverse_to_sleeper_stats
from data.nflverse import load_id_crosswalk, load_weekly_actuals
from scoring.engine import points
from sleeper import client
from sleeper.config import LEAGUE_ID, VALIDATION_SEASON

WEEKS = range(1, 18)
POS_SAMPLE = {"QB": 4, "RB": 4, "WR": 4, "TE": 3, "K": 3, "DEF": 3}


def _player_meta(players: dict):
    def meta(pid: str) -> tuple[str, str]:
        p = players.get(pid)
        if p:
            name = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            return (name or pid, p.get("position") or "?")
        # DST entries are keyed by team abbreviation.
        if pid.isalpha() and len(pid) <= 3:
            return (f"{pid} DEF", "DEF")
        return (pid, "?")

    return meta


def main() -> None:
    league = client.get_league(LEAGUE_ID)
    scoring = league["scoring_settings"]
    season = VALIDATION_SEASON
    print(f"League: {league['name']!r}  season={league['season']}  scoring_keys={len(scoring)}")
    print(f"Validating {season} | weeks {WEEKS.start}-{WEEKS.stop - 1}\n")

    players = client.get_players_nfl()
    meta = _player_meta(players)

    # (1) Sleeper-reported points per (player_id, week)
    reported: dict[tuple[str, int], float] = {}
    for wk in WEEKS:
        for row in client.get_matchups(LEAGUE_ID, wk) or []:
            for pid, pts in (row.get("players_points") or {}).items():
                if pts is not None:
                    reported[(pid, wk)] = float(pts)

    # (2) our scoring applied to Sleeper actual stats per (player_id, week)
    stat_pts: dict[tuple[str, int], float] = {}
    for wk in WEEKS:
        for row in client.get_stats(season, wk):
            stat_pts[(str(row.get("player_id")), wk)] = points(row.get("stats") or {}, scoring)

    # (3) our scoring applied to nflverse actuals per (sleeper_id, week) -- independent source
    g2s = build_gsis_to_sleeper(load_id_crosswalk())
    actuals = load_weekly_actuals(season).filter(pl.col("season_type") == "REG")
    nfl_pts: dict[tuple[str, int], float] = defaultdict(float)
    for row in actuals.iter_rows(named=True):
        sid = g2s.get(row.get("player_id"))
        if sid is not None:
            nfl_pts[(sid, row.get("week"))] += points(nflverse_to_sleeper_stats(row), scoring)

    # aggregate per player over the weeks the player was actually rostered (has a reported value)
    weeks_by_pid: dict[str, list[int]] = defaultdict(list)
    for pid, wk in reported:
        weeks_by_pid[pid].append(wk)

    records = []
    for pid, wks in weeks_by_pid.items():
        name, pos = meta(pid)
        records.append(
            {
                "name": name,
                "pos": pos,
                "wks": len(wks),
                "reported": sum(reported[(pid, w)] for w in wks),
                "sleeper_stat": sum(stat_pts.get((pid, w), 0.0) for w in wks),
                "nflverse": sum(nfl_pts.get((pid, w), 0.0) for w in wks),
            }
        )
    df = pd.DataFrame(records)

    sample = pd.concat(
        df[df.pos == pos].sort_values("reported", ascending=False).head(n)
        for pos, n in POS_SAMPLE.items()
    ).reset_index(drop=True)

    sample["d_stat"] = sample.sleeper_stat - sample.reported
    sample["nflverse"] = sample.apply(  # DEF has no nflverse team aggregate
        lambda r: float("nan") if r.pos == "DEF" else r.nflverse, axis=1
    )
    sample["d_nfl"] = sample.nflverse - sample.reported

    pd.options.display.float_format = lambda v: f"{v:8.2f}"
    show = sample[["name", "pos", "wks", "reported", "sleeper_stat", "d_stat", "nflverse", "d_nfl"]]
    print(show.to_string(index=False))

    eng = sample["d_stat"].abs().max()
    ind = sample.loc[sample.pos != "DEF", "d_nfl"].abs().max()
    print(f"\nEngine vs Sleeper-reported (all positions): max |diff| = {eng:.3f}")
    print(f"nflverse independent re-score (skill+K):     max |diff| = {ind:.3f}")


if __name__ == "__main__":
    main()

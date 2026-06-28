"""Precompute the season-dashboard snapshot and write it to a single committed SQLite artifact.

This is the **only networked module in Phase 5** -- the "ingest + recompute" pipeline the GitHub
Actions weekly cron runs (``scripts/refresh_data.py``). It reuses every earlier phase wholesale:

* the weekly **lineup** + start/sit (Phase 3 ``optimizer.inputs.load_lineup_inputs`` + ``startsit``);
* **waiver / stash / handcuff** intelligence (Phase 4 ``waivers.inputs.load_waiver_inputs``);
* **team analysis** (Phase 5 ``analysis.team``) over each roster's optimal lineup -- computed two
  ways: season-long projections (stable roster quality) and this week's projections (bye/injury-aware).

Everything is serialized to ``data_cache/season.db`` (tables + a single-row ``meta`` table) via pandas
``to_sql``. The hosted Streamlit app reads that file only -- it never hits an API on page load. Writing
goes through a temp file + atomic replace so a committed ``season.db`` is never half-written. Read-only
against Sleeper.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

from analysis import team
from data import nflverse
from optimizer.inputs import (
    STARTABLE_POSITIONS,
    assemble_players,
    find_my_roster,
    load_lineup_inputs,
)
from optimizer.lineup import LineupPlayer, optimize
from optimizer.startsit import idle_players, risky_starts, start_sit_table
from projections.board import build_board
from sleeper import client
from waivers.inputs import load_waiver_inputs
from waivers.priority import spend_advice
from waivers.stash import bye_stash_suggestions, rank_playoff_stashes

_LOG = logging.getLogger(__name__)

#: Week-11 trade deadline (CLAUDE.md). The dashboard surfaces trade ideas before it.
TRADE_DEADLINE_WEEK: int = 11

# data_cache/ is committed; season.db is the dashboard's only data source.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB: Path = _REPO_ROOT / "data_cache" / "season.db"


# --------------------------------------------------------------------------- small helpers
def _flatten_weeks(weeks: Sequence) -> str:
    """A StashCandidate's per-week breakdown as one display string."""
    return "  ".join(f"W{w.week} {w.opponent}×{w.multiplier:.2f}" for w in weeks)


def _n_tough(weeks: Sequence) -> int:
    return sum(1 for w in weeks if w.multiplier < team.TOUGH_MATCHUP)


def _usage_str(usage: Mapping, pid: str) -> str:
    sig = usage.get(pid)
    return sig.summary() if sig else ""


#: nflverse uses a couple of abbreviations that differ from Sleeper's; normalize to Sleeper's.
_NFLVERSE_TO_SLEEPER: dict[str, str] = {"LA": "LAR"}


def kickoff_by_team(
    season: int,
    week: int,
    *,
    fetch_schedules=nflverse.load_schedules,
) -> dict[str, str]:
    """``team -> "Sun 13:00 vs DAL"`` kickoff labels for the week (best-effort; ``{}`` on failure).

    Times are the schedule's ET kickoff. Lets the dashboard surface game day/time per starter so a
    FLEX call can be hedged manually — the optimizer itself maximizes projected points, not timing.
    """
    try:
        sched = fetch_schedules(season)
        reg = sched.filter(
            (pl.col("game_type") == "REG") & (pl.col("season") == season) & (pl.col("week") == week)
        )
        out: dict[str, str] = {}
        for r in reg.iter_rows(named=True):
            label = f"{(r.get('weekday') or '')[:3]} {r.get('gametime') or ''}".strip()
            home = _NFLVERSE_TO_SLEEPER.get(r.get("home_team"), r.get("home_team"))
            away = _NFLVERSE_TO_SLEEPER.get(r.get("away_team"), r.get("away_team"))
            if home:
                out[home] = f"{label} vs {away}"
            if away:
                out[away] = f"{label} @ {home}"
        return out
    except Exception:  # schedule fetch is non-critical -- never sink a snapshot over it
        _LOG.warning("could not load kickoff times for %s week %s", season, week)
        return {}


def team_names_by_roster(rosters: Sequence[Mapping], users: Sequence[Mapping]) -> dict[int, str]:
    """``roster_id -> team label`` (team name if set, else display name, else ``Team <id>``)."""
    name_by_uid: dict[str, str] = {}
    for u in users or []:
        meta = u.get("metadata") or {}
        name_by_uid[str(u.get("user_id"))] = (
            meta.get("team_name") or u.get("display_name") or str(u.get("user_id"))
        )
    out: dict[int, str] = {}
    for r in rosters:
        rid = int(r.get("roster_id"))
        out[rid] = name_by_uid.get(str(r.get("owner_id")), f"Team {rid}")
    return out


def _season_lineup_players(
    roster: Mapping,
    players_map: Mapping[str, Mapping],
    season_scored: Mapping[str, Mapping],
) -> list[LineupPlayer]:
    """A roster as season-projection ``LineupPlayer`` rows, all startable.

    For the season-long *roster-quality* view we ignore this-week injury status (a Questionable or
    one-week-Out player still counts season-long), but we still drop IR (``reserve``) and taxi -- a
    season-IR player is not usable roster strength.
    """
    skip = {str(x) for x in (roster.get("reserve") or [])} | {str(x) for x in (roster.get("taxi") or [])}
    players: list[LineupPlayer] = []
    for pid in (str(x) for x in (roster.get("players") or [])):
        if pid in skip:
            continue
        row = season_scored.get(pid) or {}
        meta = players_map.get(pid) or {}
        pos = row.get("pos") or meta.get("position")
        if pos not in STARTABLE_POSITIONS:
            continue
        is_def = pos == "DEF"
        team_abbr = pid if is_def else (meta.get("team") or row.get("team"))
        name = row.get("name") or meta.get("full_name") or pid
        players.append(
            LineupPlayer(
                player_id=pid,
                name=name,
                pos=pos,
                team=team_abbr,
                proj_pts=round(float(row.get("proj") or 0.0), 2),
            )
        )
    return players


def _strength_long(slot_points: Mapping[int, Mapping[str, float]], names: Mapping[int, str], my_rid: int):
    """Long-format rows (roster_id, team_name, is_me, slot, points) for a strength matrix."""
    rows = []
    for rid, by_slot in slot_points.items():
        for slot, pts in by_slot.items():
            rows.append(
                {
                    "roster_id": rid,
                    "team_name": names.get(rid, f"Team {rid}"),
                    "is_me": rid == my_rid,
                    "slot": slot,
                    "points": pts,
                }
            )
    return rows


# --------------------------------------------------------------------------- snapshot builder
def build_snapshot(
    league_id: str,
    user_id: str,
    season: int,
    week: int,
    *,
    sleeper=client,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Compute every dashboard table for one week. Returns ``(tables, meta)``; pure compute below."""
    # --- Phase 3: my weekly lineup + start/sit -------------------------------------------------
    lineup_inp = load_lineup_inputs(league_id, user_id, season, week, sleeper=sleeper)
    sol = optimize(lineup_inp.players, lineup_inp.slots)
    risky = {id(f.player): "; ".join(f.reasons) for f in risky_starts(sol, lineup_inp.players)}

    kickoffs = kickoff_by_team(season, week)
    lineup_rows = [
        {
            "slot": sp.slot,
            "player_id": sp.player.player_id,
            "name": sp.player.name,
            "pos": sp.player.pos,
            "team": sp.player.team or "",
            "proj": sp.player.proj_pts,
            "kickoff": kickoffs.get(sp.player.team or "", ""),
            "status": sp.player.status or "",
            "flags": risky.get(id(sp.player), ""),
        }
        for sp in sol.starters
    ]
    startsit_rows = [
        {
            "name": d.player.name,
            "pos": d.player.pos,
            "team": d.player.team or "",
            "proj": d.player.proj_pts,
            "delta": d.delta,
            "vs_slot": d.slot or "",
            "vs_name": d.would_replace.name if d.would_replace else "",
        }
        for d in start_sit_table(sol)
    ]
    idle_rows = [
        {"name": p.name, "pos": p.pos, "team": p.team or "", "proj": p.proj_pts, "reason": reason}
        for p, reason in idle_players(lineup_inp.players)
    ]

    # --- Phase 4: waivers / stash / handcuffs --------------------------------------------------
    w = load_waiver_inputs(league_id, user_id, season, week, sleeper=sleeper)
    handcuff_rows = [
        {
            "priority": a.priority,
            "backup_name": a.backup_name,
            "pos": a.pos,
            "team": a.team or "",
            "starter_name": a.starter_name,
            "starter_status": a.starter_status or "",
            "gap": a.gap,
            "reason": a.reason,
            "usage": _usage_str(w.usage, a.backup_id),
        }
        for a in w.handcuffs
    ]
    spend_rows = [
        {
            "verdict": a.verdict,
            "name": a.name,
            "pos": a.pos,
            "team": a.team or "",
            "lineup_gain": a.lineup_gain,
            "contention_level": a.contention_level,
            "contention": a.contention,
            "is_new_starter": a.is_new_starter,
            "reason": a.reason,
            "usage": _usage_str(w.usage, a.player_id),
        }
        for a in spend_advice(w.spend_candidates, w.my_players, w.slots, w.scarcity)
    ]
    stashes = rank_playoff_stashes(w.stash_candidates, w.sos, w.opponents_by_week)
    stash_rows = [
        {
            "name": s.name,
            "pos": s.pos,
            "team": s.team or "",
            "adj_value": s.adj_value,
            "raw_value": s.raw_value,
            "sos_swing": s.sos_swing,
            "weeks": _flatten_weeks(s.weeks),
            "usage": _usage_str(w.usage, s.player_id),
        }
        for s in stashes
    ]
    bye_stash_rows = [
        {
            "week": b.week,
            "pos": b.pos,
            "idle_starters": ", ".join(b.idle_starters),
            "suggestions": ", ".join(f"{n} ({v:.1f})" for n, v in b.suggestions) or "no clear FA",
        }
        for b in bye_stash_suggestions(
            w.my_starters, w.bye_week_of_team, w.stash_candidates, from_week=week + 1
        )
    ]

    # --- Phase 5: team analysis (season-long + this-week) --------------------------------------
    rosters = sleeper.get_rosters(league_id)
    users = sleeper.get_users(league_id)
    names = team_names_by_roster(rosters, users)
    my_roster = find_my_roster(rosters, user_id)
    my_rid = int(my_roster.get("roster_id"))
    slot_order = list(w.slots.keys())

    season_scored = {
        r.player_id: {"proj": r.proj_pts, "pos": r.pos, "team": r.team, "name": r.name}
        for r in build_board(season, w.scoring)
    }
    byes_this_week = {t for t, b in w.bye_week_of_team.items() if b == week}

    starters_season: dict[int, list] = {}
    starters_week: dict[int, list] = {}
    for r in rosters:
        rid = int(r.get("roster_id"))
        sea_players = _season_lineup_players(r, w.players_map, season_scored)
        starters_season[rid] = optimize(sea_players, w.slots).starters
        wk_players, _ = assemble_players(r, w.players_map, w.scored, byes_this_week)
        starters_week[rid] = optimize(wk_players, w.slots).starters

    season_pts = team.slot_points_by_team(starters_season)
    week_pts = team.slot_points_by_team(starters_week)
    season_strengths = team.position_strengths(season_pts, my_rid, slot_order)
    week_strengths = team.position_strengths(week_pts, my_rid, slot_order)

    my_season_players = _season_lineup_players(my_roster, w.players_map, season_scored)
    bye_gaps = team.bye_week_gaps(my_season_players, w.bye_week_of_team, w.slots, from_week=week)
    needs = team.positional_needs(season_strengths, my_season_players, w.slots, bye_gaps)
    trades = team.trade_targets(season_pts, my_rid, names, season_strengths)
    outlook = team.playoff_outlook(w.my_starters, w.sos, w.opponents_by_week)

    def _strength_rows(strengths: Sequence[team.PositionStrength]):
        return [
            {
                "slot": s.slot,
                "my_points": s.my_points,
                "my_rank": s.my_rank,
                "n_teams": s.n_teams,
                "league_avg": s.league_avg,
                "best": s.best,
                "worst": s.worst,
                "verdict": s.verdict,
            }
            for s in strengths
        ]

    playoff_rows = [
        {
            "name": s.name,
            "pos": s.pos,
            "team": s.team or "",
            "baseline": s.baseline,
            "raw_value": s.raw_value,
            "adj_value": s.adj_value,
            "sos_swing": s.sos_swing,
            "n_tough": _n_tough(s.weeks),
            "weeks": _flatten_weeks(s.weeks),
        }
        for s in outlook
    ]

    tables: dict[str, pd.DataFrame] = {
        "lineup": pd.DataFrame(lineup_rows),
        "startsit": pd.DataFrame(startsit_rows),
        "idle": pd.DataFrame(idle_rows),
        "handcuffs": pd.DataFrame(handcuff_rows),
        "spend": pd.DataFrame(spend_rows),
        "stashes": pd.DataFrame(stash_rows),
        "bye_stash": pd.DataFrame(bye_stash_rows),
        "team_strength_season": pd.DataFrame(_strength_long(season_pts, names, my_rid)),
        "team_strength_week": pd.DataFrame(_strength_long(week_pts, names, my_rid)),
        "position_strength_season": pd.DataFrame(_strength_rows(season_strengths)),
        "position_strength_week": pd.DataFrame(_strength_rows(week_strengths)),
        "bye_gaps": pd.DataFrame(
            [
                {
                    "week": g.week,
                    "pos": g.pos,
                    "needed": g.needed,
                    "available": g.available,
                    "idle": ", ".join(g.idle),
                    "severity": g.severity,
                }
                for g in bye_gaps
            ]
        ),
        "needs": pd.DataFrame(
            [{"pos": n.pos, "severity": n.severity, "reasons": "; ".join(n.reasons)} for n in needs]
        ),
        "trades": pd.DataFrame(
            [
                {
                    "team_name": t.team_name,
                    "give": "/".join(t.give_positions),
                    "get": "/".join(t.get_positions),
                    "rationale": t.rationale,
                }
                for t in trades
            ]
        ),
        "playoff_outlook": pd.DataFrame(playoff_rows),
        "unjoined": pd.DataFrame(
            [{"player_id": pid, "name": name} for pid, name in lineup_inp.unjoined]
        ),
    }

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "league_id": str(league_id),
        "season": int(season),
        "week": int(week),
        "my_roster_id": my_rid,
        "my_team_name": names.get(my_rid, "me"),
        "my_rank": w.my_rank,
        "n_teams": w.n_teams,
        "waiver_position": w.waiver_position if w.waiver_position is not None else -1,
        "posture": w.scarcity.posture,
        "posture_note": w.scarcity.note,
        "lineup_total": sol.total,
        "lineup_status": sol.status,
        "holes": json.dumps(sol.holes),
        "playoff_total": round(sum(s.adj_value for s in outlook), 2),
        "trade_deadline_week": TRADE_DEADLINE_WEEK,
    }
    return tables, meta


# --------------------------------------------------------------------------- serialize
def write_sqlite(db_path: Path | str, tables: Mapping[str, pd.DataFrame], meta: Mapping) -> Path:
    """Write all tables + a single-row ``meta`` table to ``db_path`` atomically (temp + replace)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp.db")
    if tmp.exists():
        tmp.unlink()

    con = sqlite3.connect(tmp)
    try:
        meta_row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in meta.items()}
        pd.DataFrame([meta_row]).to_sql("meta", con, index=False, if_exists="replace")
        for name, df in tables.items():
            # A column-less (empty) frame makes to_sql emit invalid `CREATE TABLE x ()`; give it a
            # placeholder column so the table exists and the app's `.empty` guard still fires.
            if df.shape[1] == 0:
                df = pd.DataFrame({"_empty": pd.Series(dtype="object")})
            df.to_sql(name, con, index=False, if_exists="replace")
        con.commit()
    finally:
        con.close()

    db_path.unlink(missing_ok=True)
    tmp.replace(db_path)
    return db_path


def build_and_write(
    league_id: str,
    user_id: str,
    season: int,
    week: int,
    *,
    db_path: Path | str = DEFAULT_DB,
    sleeper=client,
) -> Path:
    """Build the snapshot for one week and write it to ``db_path``. Returns the path written."""
    tables, meta = build_snapshot(league_id, user_id, season, week, sleeper=sleeper)
    out = write_sqlite(db_path, tables, meta)
    _LOG.info("wrote season snapshot for %s week %s -> %s", season, week, out)
    return out

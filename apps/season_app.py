"""Hosted season dashboard (Phase 5) -- read-only, offline, for Streamlit Community Cloud.

    streamlit run apps/season_app.py        # locally
    # deployed: share.streamlit.io -> this repo -> apps/season_app.py (see apps/README.md)

This app reads **only** the precomputed ``data_cache/season.db`` artifact -- it never calls an API on
page load. That snapshot is rebuilt weekly by ``scripts/refresh_data.py`` (the GitHub Actions cron),
committed back to the repo, and Streamlit auto-redeploys on the commit. All decision logic lives in
``src/`` (Phases 1-5); this file is just the read-and-render UI.

Tabs: This Week (optimal lineup + start/sit) · Waivers & Stash (handcuff/spend/stash/bye alerts) ·
Team Analysis (positional strength vs the league, bye gaps, needs, trade ideas, Weeks 15-17 outlook).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the src/ packages importable (kept for parity with the other apps; this app needs no src
# imports at runtime since it only reads the SQLite snapshot).
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import sqlite3  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

DB_PATH = _ROOT / "data_cache" / "season.db"

st.set_page_config(page_title="Fantasy season dashboard", page_icon="🏈", layout="wide")

_VERDICT_BADGE = {"strength": "🟢 strength", "average": "⚪ average", "weakness": "🔴 weakness"}
_SEVERITY_BADGE = {"hole": "🔴 hole", "thin": "🟡 thin", "high": "🔴 high", "medium": "🟡 medium"}
_SPEND_BADGE = {"spend": "✅ SPEND", "stream-later": "🟡 stream", "hold": "⚪ hold"}


# --------------------------------------------------------------------------- cached readers
def _mtime() -> float:
    try:
        return DB_PATH.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_table(name: str, _mtime: float) -> pd.DataFrame:
    """Read one table from the snapshot (empty DataFrame if it doesn't exist). ``_mtime`` busts the
    cache when the committed artifact changes."""
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
            return pd.read_sql_query(f"SELECT * FROM {name}", con)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_meta(_mtime: float) -> dict:
    df = load_table("meta", _mtime)
    if df.empty:
        return {}
    meta = df.iloc[0].to_dict()
    try:
        meta["holes"] = json.loads(meta.get("holes") or "{}")
    except (TypeError, json.JSONDecodeError):
        meta["holes"] = {}
    return meta


def show(df: pd.DataFrame, *, empty: str = "—", **kwargs) -> None:
    if df is None or df.empty:
        st.caption(empty)
    else:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)


# --------------------------------------------------------------------------- Sleeper-style lineup cards
_POS_COLOR = {
    "QB": "#c0392b", "RB": "#27ae60", "WR": "#2980b9", "TE": "#e67e22",
    "K": "#8e44ad", "DEF": "#7f8c8d", "FLEX": "#16a085",
}


def _slot_badge(slot: str) -> str:
    color = _POS_COLOR.get(str(slot), "#555")
    return (
        f"<span style='background:{color};color:#fff;padding:3px 9px;border-radius:6px;"
        f"font-size:0.8em;font-weight:700'>{slot}</span>"
    )


def _headshot(pid: str, pos: str) -> str:
    """Sleeper CDN image: a team logo for DST (id == team abbr), else the player's thumbnail."""
    if pos == "DEF":
        return f"https://sleepercdn.com/images/team_logos/nfl/{pid.lower()}.png"
    return f"https://sleepercdn.com/content/nfl/players/thumb/{pid}.jpg"


def render_lineup_cards(df: pd.DataFrame) -> None:
    """Render the optimal lineup as Sleeper-style cards: slot badge · headshot · player · proj."""
    for _, r in df.iterrows():
        pid, pos = str(r.get("player_id") or ""), str(r.get("pos") or "")
        with st.container(border=True):
            c_badge, c_img, c_main, c_proj = st.columns([0.7, 0.7, 5, 1.1])
            c_badge.markdown(_slot_badge(str(r["slot"])), unsafe_allow_html=True)
            if pid:
                c_img.image(_headshot(pid, pos), width=44)
            meta = f"{pos} · {r['team']}"
            if str(r.get("kickoff") or ""):
                meta += f" · 🕐 {r['kickoff']}"
            c_main.markdown(f"**{r['name']}**  \n{meta}")
            flag, status = str(r.get("flags") or ""), str(r.get("status") or "")
            if flag:
                c_main.markdown(f":orange[⚠ {flag}]")
            elif status:
                c_main.markdown(f":orange[{status}]")
            c_proj.markdown(f"### {float(r['proj']):.1f}")


# --------------------------------------------------------------------------- backtest readers
BACKTEST_DB = _ROOT / "data_cache" / "backtest.db"


def _bt_mtime() -> float:
    try:
        return BACKTEST_DB.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_bt(name: str, _mtime: float) -> pd.DataFrame:
    try:
        with sqlite3.connect(f"file:{BACKTEST_DB}?mode=ro", uri=True) as con:
            return pd.read_sql_query(f"SELECT * FROM {name}", con)
    except Exception:
        return pd.DataFrame()


def load_bt_meta(_mtime: float) -> dict:
    df = load_bt("meta", _mtime)
    return {} if df.empty else df.iloc[0].to_dict()


_RESULT_BADGE = {"W": "✅ W", "L": "❌ L", "T": "➖ T", "—": "—"}


# --------------------------------------------------------------------------- guard: no snapshot yet
if not DB_PATH.exists():
    st.title("🏈 Fantasy season dashboard")
    st.error(
        "No data snapshot found at `data_cache/season.db`.\n\n"
        "Generate it locally with:\n\n"
        "```\n./.venv/Scripts/python scripts/refresh_data.py --week <N>\n```\n\n"
        "On the hosted app this file is produced by the weekly GitHub Actions refresh."
    )
    st.stop()

mt = _mtime()
meta = load_meta(mt)
week = int(meta.get("week", 0))
season = int(meta.get("season", 0))
deadline = int(meta.get("trade_deadline_week", 11))

# --------------------------------------------------------------------------- header
st.title("🏈 Fantasy season dashboard")
wp = int(meta.get("waiver_position", -1))
wp_txt = f"#{wp}" if wp and wp > 0 else "n/a"
st.caption(
    f"**{meta.get('my_team_name', 'My team')}** · {season} Week {week} · "
    f"standings #{meta.get('my_rank', '?')}/{meta.get('n_teams', '?')} · "
    f"waiver priority {wp_txt} · posture **{str(meta.get('posture', '')).upper()}** · "
    f"data generated {meta.get('generated_at', 'unknown')}"
)
if st.button("🔄 Reload snapshot"):
    st.cache_data.clear()
    st.rerun()


# Staleness guard: the refresh cron deliberately no-ops in the off-season, leaving the last snapshot
# in place — say so instead of presenting months-old advice (win prob, deadline countdown) as live.
def _staleness_days(meta_row: dict) -> float | None:
    try:
        gen = datetime.strptime(
            str(meta_row.get("generated_at", "")), "%Y-%m-%d %H:%M:%S UTC"
        ).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - gen).total_seconds() / 86400.0
    except Exception:
        return None


stale_days = _staleness_days(meta)
snapshot_stale = stale_days is not None and stale_days > 8
if snapshot_stale:
    st.warning(
        f"🕰️ **This snapshot is {stale_days:.0f} days old** ({season} Week {week}). The weekly "
        "refresh pauses in the off-season, so everything below is the season's last state — "
        "historical context, not live advice."
    )

tab_week, tab_waiver, tab_team, tab_bt = st.tabs(
    ["📅 This Week", "🔁 Waivers & Stash", "📊 Team Analysis", "📈 Backtest"]
)

# =========================================================================== THIS WEEK
with tab_week:
    c1, c2, c3 = st.columns(3)
    c1.metric("Projected lineup total", f"{meta.get('lineup_total', 0):.2f}")
    c2.metric("Solver status", str(meta.get("lineup_status", "—")))
    holes = meta.get("holes") or {}
    c3.metric("Unfilled slots", sum(holes.values()) if holes else 0)
    if holes:
        st.warning("⚠ No eligible player for: " + ", ".join(f"{n}× {s}" for s, n in holes.items()))

    wprob = meta.get("win_prob", -1)
    if wprob is not None and float(wprob) >= 0:
        st.subheader("🎲 Win probability this week")
        lv = str(meta.get("leverage", ""))
        lv_badge = {
            "favorite": "🟢 favorite", "underdog": "🔴 underdog", "toss-up": "🟡 toss-up",
        }.get(lv, lv)
        w1, w2, w3 = st.columns(3)
        w1.metric(f"P(win) vs {meta.get('opp_name', 'opponent')}", f"{float(wprob):.0%}")
        w2.metric("Your projection", f"{meta.get('lineup_total', 0):.1f}")
        w3.metric("Opponent projection", f"{meta.get('opp_proj', 0):.1f}")
        st.caption(f"**{lv_badge}** — {meta.get('leverage_note', '')}")

        wpt = load_table("winprob", mt)
        if not wpt.empty:
            st.markdown("**Start/sit leverage — swaps that move your win odds**")
            v = wpt.copy()
            v["Δ win%"] = (v["delta_winprob"] * 100).map(lambda x: f"{x:+.1f}%")
            v["Δ proj"] = v["delta_proj"].map(lambda x: f"{x:+.1f}")
            v = v.rename(columns={
                "bench": "start (bench)", "bench_pos": "pos", "starter": "over (starter)",
            })
            show(v[["start (bench)", "pos", "over (starter)", "slot", "Δ win%", "Δ proj"]])
            st.caption(
                "A **+Δ win%** with a **−Δ proj** = start the upside play: it lifts your win odds "
                "despite fewer projected points (underdog leverage). When you're a favorite the "
                "reverse holds — protect the floor."
            )
        st.divider()

    st.subheader("Optimal starting lineup")
    lineup = load_table("lineup", mt)
    if lineup.empty:
        st.caption("No lineup in the snapshot.")
    else:
        render_lineup_cards(lineup)
        with st.expander("Detailed table"):
            cols = [c for c in ["slot", "name", "pos", "team", "proj", "kickoff", "status", "flags"]
                    if c in lineup.columns]
            show(lineup[cols])

    risky = lineup[lineup["flags"].astype(str).str.len() > 0] if not lineup.empty else pd.DataFrame()
    if not risky.empty:
        st.subheader("⚠ Risky starts")
        show(risky[["slot", "name", "pos", "flags"]])

    st.subheader("Start / sit — bench delta vs. the starter they'd replace")
    st.caption("Δ ≤ 0 in an optimal lineup; a near-zero Δ is a genuine coin-flip.")
    show(load_table("startsit", mt))

    st.subheader("Idle this week (bye / OUT / IR)")
    show(load_table("idle", mt), empty="Nobody idle — full strength.")

    unjoined = load_table("unjoined", mt)
    if not unjoined.empty:
        with st.expander(f"⚠ {len(unjoined)} rostered player(s) had no projection (scored 0.0)"):
            show(unjoined)

# =========================================================================== WAIVERS & STASH
with tab_waiver:
    st.caption(meta.get("posture_note", ""))

    st.subheader("🎯 Weekly K/DEF streaming — best available this week")
    st.caption(
        "Ranked by THIS week's projection in our scoring. Δ = edge over your current starter. "
        "The next / ROS·g / playoff columns show the run ahead — the DEF *playoff* column is tilted by "
        "our DEF strength-of-schedule; K carries no SOS (a kicker's output rides its own offense)."
    )
    streamers = load_table("streamers", mt)
    if streamers.empty:
        st.caption("No K/DEF free agents projected this week.")
    else:
        for pos in ("DEF", "K"):
            grp = streamers[streamers["pos"] == pos]
            if grp.empty:
                continue
            verdict = str(grp["verdict"].iloc[0])
            cur = str(grp["current_name"].iloc[0]) or "none rostered"
            cur_tw = float(grp["current_this_week"].iloc[0])
            badge = "✅ STREAM" if verdict == "stream" else "⚪ hold"
            st.markdown(f"**{pos}** — {badge} · current starter: {cur} ({cur_tw:.1f} proj)")
            view = grp.rename(columns={
                "name": "player", "this_week": "this wk", "next_week": "next wk",
                "ros_pg": "ROS·g", "playoff": "playoff",
            })
            show(view[["player", "team", "this wk", "gain", "next wk", "ROS·g", "playoff"]])

    st.divider()

    st.subheader("Handcuff / injury-replacement alerts")
    hc = load_table("handcuffs", mt)
    if hc.empty:
        st.caption("None — every starter's next man up is already rostered.")
    else:
        hc = hc.copy()
        hc["priority"] = hc["priority"].map(lambda p: "🚨 URGENT" if p == "URGENT" else "🔶 HIGH")
        show(hc[["priority", "backup_name", "pos", "team", "reason", "usage"]])

    st.subheader("Reverse-priority spend advice")
    st.caption("No FAAB — spend a single ordered claim only on real upgrades or new starters.")
    spend = load_table("spend", mt)
    if not spend.empty:
        spend = spend.copy()
        spend["verdict"] = spend["verdict"].map(lambda v: _SPEND_BADGE.get(v, v))
        show(
            spend[["verdict", "name", "pos", "team", "lineup_gain", "contention_level", "reason", "usage"]]
        )
    else:
        show(spend)

    st.subheader("Playoff stash ranker — Weeks 15-17, SOS-adjusted (top 15)")
    st.caption("Free agents ranked by Weeks 15-17 value in our scoring, tilted by playoff opponent SOS.")
    show(load_table("stashes", mt).head(15))

    st.subheader("Upcoming starter byes (stash a fill-in now)")
    show(load_table("bye_stash", mt), empty="No starter byes ahead in range.")

# =========================================================================== TEAM ANALYSIS
with tab_team:
    # ----- trade ideas (prominent before the deadline; silent when the snapshot is stale) ----
    weeks_left = deadline - week
    if snapshot_stale:
        pass  # a months-old countdown ("1 week left to deal!") is actively misleading
    elif week and week < deadline:
        st.warning(f"⏰ **Trade deadline is Week {deadline}** — {weeks_left} week(s) left to deal.")
    elif week and week >= deadline:
        st.info(f"Trade deadline (Week {deadline}) has passed — trades are closed.")

    st.subheader("🤝 Trade offers — win-win 1-for-1 swaps")
    st.caption(
        "Concrete player-for-player deals where **both** teams' best starting lineup improves, "
        "valued in **rest-of-season** points (this week through the championship) in our scoring — "
        "a deal the partner has a real reason to accept. A status flag (Out/Questionable/…) means "
        "the projection may not survive to the playoff weeks: check the injury timeline first."
    )
    offers = load_table("trade_offers", mt)
    if offers.empty:
        st.caption("No win-win 1-for-1 fits right now (need a swap that upgrades both lineups).")
    else:
        v = offers.copy()

        def _tag(name, pos, status):
            flag = f" ⚠{status}" if status else ""
            return f"{name} ({pos}){flag}"

        has_status = "give_status" in v.columns  # older committed snapshots predate the column
        v["you give"] = v.apply(
            lambda r: _tag(r["give"], r["give_pos"], r.get("give_status", "") if has_status else ""),
            axis=1,
        )
        v["you get"] = v.apply(
            lambda r: _tag(r["get"], r["get_pos"], r.get("get_status", "") if has_status else ""),
            axis=1,
        )
        v = v.rename(columns={"my_gain": "your gain", "their_gain": "their gain"})
        show(v[["partner", "you give", "you get", "your gain", "their gain"]])

    st.divider()

    st.subheader("💡 Trade-target ideas (positional fit)")
    st.caption("Teams strong where you're weak **and** weak where you're strong (mutual fit).")
    trades = load_table("trades", mt)
    if trades.empty:
        st.caption("No clear mutual-fit partners right now (need a clear strength *and* weakness).")
    else:
        trades = trades.rename(columns={"give": "offer (your surplus)", "get": "target (their surplus)"})
        show(trades)

    st.divider()

    # ----- positional strength vs the league ------------------------------------------------
    st.subheader("Starter strength vs. the league")
    basis = st.radio(
        "Basis", ["Season-long", "This week"], horizontal=True,
        help="Season-long = stable roster quality (preseason-style projections). This week = "
        "current projections with byes/injuries applied.",
    )
    suffix = "season" if basis == "Season-long" else "week"

    pos_str = load_table(f"position_strength_{suffix}", mt)
    if not pos_str.empty:
        view = pos_str.copy()
        view["rank"] = view.apply(lambda r: f"{int(r['my_rank'])}/{int(r['n_teams'])}", axis=1)
        view["verdict"] = view["verdict"].map(lambda v: _VERDICT_BADGE.get(v, v))
        view = view.rename(columns={"my_points": "my pts", "league_avg": "league avg"})
        show(view[["slot", "my pts", "rank", "league avg", "best", "worst", "verdict"]])
    else:
        show(pos_str)

    with st.expander("Full league strength matrix (points per slot, by team)"):
        long = load_table(f"team_strength_{suffix}", mt)
        if long.empty:
            st.caption("—")
        else:
            mat = long.pivot_table(index="team_name", columns="slot", values="points", aggfunc="sum")
            slot_cols = [c for c in ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"] if c in mat.columns]
            mat = mat[slot_cols + [c for c in mat.columns if c not in slot_cols]]
            mat["TOTAL"] = mat.sum(axis=1).round(2)
            mat = mat.sort_values("TOTAL", ascending=False)
            me = meta.get("my_team_name")
            mat.index = [f"⭐ {t}" if t == me else t for t in mat.index]
            st.dataframe(mat, width="stretch")

    st.divider()

    # ----- needs + bye gaps -----------------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Positional needs")
        needs = load_table("needs", mt)
        if needs.empty:
            st.caption("No glaring needs — roster is balanced.")
        else:
            needs = needs.copy()
            needs["severity"] = needs["severity"].map(lambda s: _SEVERITY_BADGE.get(s, s))
            show(needs[["pos", "severity", "reasons"]])
    with right:
        st.subheader("Bye-week gaps")
        st.caption("Weeks a bye forces a backup (🟡 thin) or leaves a hole (🔴).")
        gaps = load_table("bye_gaps", mt)
        if gaps.empty:
            st.caption("No upcoming bye-week gaps.")
        else:
            gaps = gaps.copy()
            gaps["severity"] = gaps["severity"].map(lambda s: _SEVERITY_BADGE.get(s, s))
            show(gaps[["week", "pos", "needed", "available", "idle", "severity"]])

    st.divider()

    # ----- playoff outlook ------------------------------------------------------------------
    st.subheader("Playoff outlook — Weeks 15-17 (my likely starters)")
    pt = meta.get("playoff_total", 0)
    st.metric("Total SOS-adjusted playoff projection", f"{pt:.2f}")
    st.caption(
        "Each starter's per-game baseline summed over Weeks 15-17, tilted by playoff opponent SOS in "
        "our scoring. `n_tough` = playoff weeks with a tough matchup (SOS multiplier < 0.90)."
    )
    outlook = load_table("playoff_outlook", mt)
    if not outlook.empty:
        outlook = outlook.rename(columns={"adj_value": "playoff value", "raw_value": "raw", "sos_swing": "SOS Δ"})
        show(outlook[["name", "pos", "team", "playoff value", "raw", "SOS Δ", "n_tough", "weeks"]])
    else:
        show(outlook)

# =========================================================================== 2025 BACKTEST
with tab_bt:
    if not BACKTEST_DB.exists():
        st.info(
            "No backtest artifact yet. Build it with:\n\n"
            "```\n./.venv/Scripts/python scripts/backtest.py --season 2025\n```\n\n"
            "It replays a *completed* season: best-possible lineup vs what you started, and the "
            "tool's VOR draft vs your actual draft — all scored by real results."
        )
    else:
        bmt = _bt_mtime()
        bm = load_bt_meta(bmt)
        st.caption(
            f"**{bm.get('my_team_name', '')}** · {bm.get('season', '')} season backtest · "
            f"every lineup scored by *actual* results · generated {bm.get('generated_at', '')}"
        )

        # ----- season summary --------------------------------------------------------------
        st.subheader("Season summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Your actual", f"{bm.get('actual_total', 0):.0f} pts",
            f"{bm.get('actual_record', '')} · #{bm.get('actual_rank', '?')}", delta_color="off",
        )
        c2.metric(
            "Hindsight-optimal", f"{bm.get('optimal_total', 0):.0f} pts",
            f"{bm.get('optimal_record', '')} · #{bm.get('optimal_rank', '?')}",
        )
        c3.metric(
            "Left on the bench", f"{bm.get('bench_lost_total', 0):.0f} pts",
            "best lineup every week", delta_color="off",
        )
        dmy, dtool = bm.get("draft_my_total", 0), bm.get("draft_tool_total", 0)
        c4.metric(
            "Tool's draft", f"{dtool:.0f} pts", f"{dtool - dmy:+.0f} vs your {dmy:.0f}",
        )

        weekly = load_bt("weekly", bmt)
        if not weekly.empty:
            # Regular-season flips only — the sentence pairs the count with the regular-season record.
            reg = weekly[~weekly["playoff"].astype(bool)]
            flips = int(((reg["result"] == "L") & (reg["optimal_result"] == "W")).sum())
            st.markdown(
                f"Starting your **best legal lineup every week** would have turned "
                f"**{flips}** regular-season loss(es) into wins — a "
                f"**{bm.get('actual_record','')} → {bm.get('optimal_record','')}** regular season — "
                f"by capturing the **{bm.get('bench_lost_total', 0):.0f}** points left on the bench. "
                f"Following the tool's *projection* lineup (scored by real results) went "
                f"**{bm.get('tool_record','')}** ({bm.get('tool_total',0):.0f} pts) — proof that the "
                f"edge is the optimizer's ceiling, not raw projections."
            )
            if bm.get("draft_note"):
                st.caption(f"⚠️ {bm['draft_note']}")

            # ----- actual vs optimal vs tool, by week --------------------------------------
            chart = weekly.set_index("week")[["actual", "optimal", "tool"]]
            st.line_chart(chart, height=240)

        sub_week, sub_h2h, sub_draft, sub_board, sub_tx = st.tabs(
            ["📅 Weekly detail", "⚔️ Matchups", "🏈 Draft replay", "🗂️ Draft board", "🔁 Transactions"]
        )

        # ----- weekly detail ---------------------------------------------------------------
        with sub_week:
            st.caption(
                "Each week: what you scored vs your real opponent, the best you *could* have scored, "
                "points left on the bench, the single most costly bench call, and your weekly rank "
                "(actual → optimal) among all 12 teams."
            )
            if weekly.empty:
                st.caption("—")
            else:
                v = weekly.copy()
                v["wk"] = v.apply(lambda r: f"{r['week']}{'*' if r['playoff'] else ''}", axis=1)
                for col in ("result", "optimal_result", "tool_result"):
                    v[col] = v[col].map(lambda x: _RESULT_BADGE.get(x, x))
                v["rank a→o"] = v.apply(lambda r: f"{int(r['actual_rank'])}→{int(r['optimal_rank'])}", axis=1)
                v = v.rename(columns={
                    "opponent": "opp", "opp_pts": "opp pts", "result": "res",
                    "optimal_result": "opt res", "bench_lost": "bench lost",
                    "tool_result": "tool res", "top_miss": "biggest miss",
                })
                show(v[["wk", "opp", "actual", "opp pts", "res", "optimal", "opt res",
                        "bench lost", "tool", "tool res", "rank a→o", "biggest miss"]])
                st.caption("`*` = fantasy playoff week (not counted in the record).")

        # ----- draft replay ----------------------------------------------------------------
        with sub_draft:
            st.caption(
                "At each of your snake picks: our **VOR** pick vs the naive **market-ADP** pick vs who "
                "you **actually** took — all graded by full-season actual points. The ADP column is the "
                "honest baseline: does drafting in *our* scoring beat just following ADP?"
            )
            draft = load_bt("draft", bmt)
            if draft.empty:
                st.caption("—")
            else:
                vor_t = bm.get("draft_tool_total", 0)
                adp_t = bm.get("draft_adp_total", 0)
                my_t = bm.get("draft_my_total", 0)
                m1, m2, m3 = st.columns(3)
                m1.metric("VOR draft", f"{vor_t:.0f}", f"{vor_t - my_t:+.0f} vs you")
                m2.metric("ADP draft (baseline)", f"{adp_t:.0f}",
                          f"{adp_t - my_t:+.0f} vs you", delta_color="off")
                m3.metric("VOR edge over ADP", f"{vor_t - adp_t:+.0f}", "the tool's real value-add")
                d = draft.rename(columns={
                    "my_pick": "your pick", "my_pos": "pos", "my_pts": "your pts",
                    "tool_pick": "VOR pick", "tool_pos": "pos.", "tool_pts": "VOR pts",
                    "adp_pick": "ADP pick", "adp_pos": "pos..", "adp_pts": "ADP pts",
                })
                cols = ["pick", "round", "your pick", "pos", "your pts", "VOR pick", "pos.",
                        "VOR pts", "ADP pick", "pos..", "ADP pts", "diff_vs_adp"]
                show(d[[c for c in cols if c in d.columns]])

        # ----- weekly head-to-head ---------------------------------------------------------
        with sub_h2h:
            st.caption("Your whole starting lineup vs your real opponent's, slot-by-slot, scored by actual points.")
            md = load_bt("matchup_detail", bmt)
            if md.empty:
                st.caption("—")
            else:
                wk_opts = sorted(int(w) for w in md["week"].unique())
                wsel = st.selectbox("Week", wk_opts, key="bt_h2h_week")
                wdf = md[md["week"] == wsel].copy()
                my_tot, opp_tot = round(wdf["my_pts"].sum(), 2), round(wdf["opp_pts"].sum(), 2)
                wrow = weekly[weekly["week"] == wsel]
                opp_name = wrow["opponent"].iloc[0] if not wrow.empty else "opponent"
                res = "✅ WIN" if my_tot > opp_tot else ("❌ LOSS" if my_tot < opp_tot else "➖ TIE")
                h1, h2, h3 = st.columns(3)
                h1.metric(bm.get("my_team_name", "Me"), f"{my_tot:.2f}")
                h2.metric(str(opp_name), f"{opp_tot:.2f}")
                h3.metric("Result", res, f"{my_tot - opp_tot:+.2f}", delta_color="off")
                view = wdf.rename(columns={
                    "my_player": bm.get("my_team_name", "me"), "my_pts": "pts",
                    "opp_pts": "pts ", "opp_player": str(opp_name),
                })
                show(view[["slot", bm.get("my_team_name", "me"), "pts", "pts ", str(opp_name)]])

        # ----- full draftboard (all teams) -------------------------------------------------
        with sub_board:
            st.caption(
                "The real snake draft — all 12 teams, round × slot. Your column is starred; each cell "
                "is the player and full-season fantasy points in our scoring."
            )
            db = load_bt("draftboard", bmt)
            if db.empty:
                st.caption("—")
            else:
                my_slot = int(bm.get("my_draft_slot", 0))
                db = db.copy()
                db["cell"] = db.apply(
                    lambda r: f"{r['player']} ({r['pos']}) · {r['season_pts']:.0f}", axis=1
                )
                slot_team = db.drop_duplicates("slot").set_index("slot")["team"].to_dict()
                grid = db.pivot(index="round", columns="slot", values="cell")
                grid = grid.reindex(sorted(grid.columns), axis=1)
                grid.columns = [
                    ("⭐ " if s == my_slot else "") + str(slot_team.get(s, f"S{s}")) for s in grid.columns
                ]
                grid.index = [f"R{r}" for r in grid.index]
                st.dataframe(grid, width="stretch", height=560)

        # ----- transactions timeline -------------------------------------------------------
        with sub_tx:
            st.caption(
                "Completed adds / drops / trades over the season. The weekly views above already score "
                "each week's *actual* roster (waivers + trades included) — this just surfaces the moves."
            )
            tx = load_bt("transactions", bmt)
            if tx.empty:
                st.caption("No transactions recorded.")
            else:
                only_mine = st.checkbox("Only my moves", value=False, key="bt_tx_mine")
                t = tx.copy()
                if only_mine:
                    t = t[t["is_mine"].astype(bool)]
                _TX_LABEL = {"waiver": "waiver", "free_agent": "free agent", "trade": "trade"}
                t["type"] = t["type"].map(lambda x: _TX_LABEL.get(x, x))
                t["team"] = t.apply(
                    lambda r: ("⭐ " if r["is_mine"] else "") + str(r["team"]), axis=1
                )
                show(t[["week", "type", "team", "added", "dropped"]])

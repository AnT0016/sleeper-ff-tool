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

tab_week, tab_waiver, tab_team, tab_bt = st.tabs(
    ["📅 This Week", "🔁 Waivers & Stash", "📊 Team Analysis", "📈 2025 Backtest"]
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

    st.subheader("Optimal starting lineup")
    show(load_table("lineup", mt), empty="No lineup in the snapshot.")

    lineup = load_table("lineup", mt)
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
    # ----- trade ideas (prominent before the deadline) -------------------------------------
    weeks_left = deadline - week
    if week and week < deadline:
        st.warning(f"⏰ **Trade deadline is Week {deadline}** — {weeks_left} week(s) left to deal.")
    elif week and week >= deadline:
        st.info(f"Trade deadline (Week {deadline}) has passed — trades are closed.")

    st.subheader("💡 Trade-target ideas")
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
            "```\n./.venv/Scripts/python scripts/backtest_2025.py --season 2025\n```\n\n"
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
            flips = int(((weekly["result"] == "L") & (weekly["optimal_result"] == "W")).sum())
            reg = weekly[~weekly["playoff"].astype(bool)]
            st.markdown(
                f"Starting your **best legal lineup every week** would have turned "
                f"**{flips}** loss(es) into wins — a **{bm.get('actual_record','')} → "
                f"**{bm.get('optimal_record','')}** regular season — and left **0** of those "
                f"{bm.get('bench_lost_total', 0):.0f} bench points behind. "
                f"Following the tool's *projection* lineup (scored by real results) went "
                f"**{bm.get('tool_record','')}** ({bm.get('tool_total',0):.0f} pts) — proof that the "
                f"edge is the optimizer's ceiling, not raw projections."
            )

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
                "At each of your snake picks, the tool's best-available by VOR vs who you actually "
                "took — each graded by full-season actual points. `diff` = tool − you."
            )
            draft = load_bt("draft", bmt)
            if draft.empty:
                st.caption("—")
            else:
                d = draft.rename(columns={
                    "my_pick": "your pick", "my_pos": "pos", "my_pts": "your pts",
                    "tool_pick": "tool pick", "tool_pos": "pos.", "tool_pts": "tool pts",
                })
                show(d[["pick", "round", "your pick", "pos", "your pts",
                        "tool pick", "pos.", "tool pts", "diff"]])
                st.caption(
                    f"Tool draft total **{bm.get('draft_tool_total', 0):.0f}** vs your "
                    f"**{bm.get('draft_my_total', 0):.0f}** season points "
                    f"(**{bm.get('draft_tool_total', 0) - bm.get('draft_my_total', 0):+.0f}**)."
                )

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

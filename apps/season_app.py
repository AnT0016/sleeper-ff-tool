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

tab_week, tab_waiver, tab_team = st.tabs(["📅 This Week", "🔁 Waivers & Stash", "📊 Team Analysis"])

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

"""Local live snake-draft tracker (Phase 2).

Run on draft night:  ``streamlit run apps/draft_app.py``

Polls the Sleeper draft every ~3s, re-scores every available player in *our* league's exact scoring
(reusing the Phase 1 client + engine), and shows a VOR-ranked, tiered best-available board with
roster-need highlighting, positional-run detection, and -- once our slot is revealed -- our snake
pick numbers and which targets should survive to each. Read-only; it never writes to the league.

All decision logic lives in ``src/`` (``projections.board``, ``draft.vor/snake/roster``); this file is
just the Streamlit UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ packages importable even when not pip-installed (defensive; editable install also works).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from draft import roster, snake  # noqa: E402
from draft.vor import add_vor, replacement_levels, tierize  # noqa: E402
from projections.board import build_board  # noqa: E402
from sleeper import client  # noqa: E402
from sleeper.config import LEAGUE_ID, MY_USER_ID  # noqa: E402

st.set_page_config(page_title="Draft tracker", page_icon="🏈", layout="wide")

_SURVIVAL_BADGE = {snake.AVAILABLE: "🟢", snake.TOSSUP: "🟡", snake.GONE: "🔴"}


# --------------------------------------------------------------------------- cached data loaders
@st.cache_data(ttl=60, show_spinner=False)
def load_draft(draft_id: str) -> dict:
    return client.get_draft(draft_id)


@st.cache_data(ttl=600, show_spinner=False)
def load_users(league_id: str) -> dict[str, str]:
    """user_id -> display label (team name if set, else username)."""
    out: dict[str, str] = {}
    for u in client.get_users(league_id) or []:
        meta = u.get("metadata") or {}
        out[str(u.get("user_id"))] = meta.get("team_name") or u.get("display_name") or u.get("user_id")
    return out


@st.cache_data(ttl=900, show_spinner="Re-scoring season projections in our league settings…")
def load_vor_board(season: int, league_id: str, base_starters: tuple, flex_total: int) -> list:
    """Fully-prepared board (custom-scored + VOR + tiers, sorted by VOR), cached so it isn't rebuilt
    on every rerun. Scoring is pulled from the draft's own league. ``base_starters`` is passed as a
    sorted tuple of ``(pos, count)`` so the args stay hashable for the cache."""
    scoring = client.get_league(league_id)["scoring_settings"]
    board = build_board(season, scoring)
    replacement = replacement_levels(board, dict(base_starters), flex_slots=flex_total)
    add_vor(board, replacement)
    tierize(board, by="vor")
    board.sort(key=lambda p: p.vor, reverse=True)
    return board


@st.cache_data(ttl=600, show_spinner=False)
def discover_drafts(league_id: str) -> list[dict]:
    try:
        return client.get_league_drafts(league_id) or []
    except Exception:
        return []


# --------------------------------------------------------------------------- helpers (UI-only)
def slot_of_pick(pick_no: int, teams: int) -> tuple[int, int]:
    """(round, draft_slot) for an overall pick number in a snake draft."""
    rnd = (pick_no - 1) // teams + 1
    pos_in_round = (pick_no - 1) % teams + 1
    slot = pos_in_round if rnd % 2 == 1 else teams - pos_in_round + 1
    return rnd, slot


def slot_names(draft_order: dict, users: dict[str, str]) -> dict[int, str]:
    return {int(slot): users.get(str(uid), str(uid)) for uid, slot in (draft_order or {}).items()}


def fmt_player(pick: dict) -> str:
    meta = pick.get("metadata") or {}
    name = f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip() or pick.get("player_id")
    return f"{name} ({meta.get('position', '?')}-{meta.get('team', '?')})"


# --------------------------------------------------------------------------- sidebar / inputs
st.sidebar.header("Draft")
league_id = st.sidebar.text_input("League ID", value=LEAGUE_ID, help="Used to discover the draft.").strip()
draft_id = st.sidebar.text_input("Draft ID", value="", help="Paste the draft_id, or pick one below.").strip()

if not draft_id:
    found = discover_drafts(league_id)
    if found:
        labels = {f"{d.get('season')} · {d.get('status')} · {d['draft_id']}": d["draft_id"] for d in found}
        choice = st.sidebar.selectbox("…or pick a draft for this league", list(labels))
        draft_id = labels[choice]
    else:
        st.sidebar.warning("No draft found for this league yet. Paste a Draft ID.")
        st.stop()

my_user_id = st.sidebar.text_input("My user ID", value=MY_USER_ID).strip()

try:
    default_season = int(client.get_state().get("season") or 2025)
except Exception:
    default_season = 2025
season = st.sidebar.number_input("Projection season", min_value=2020, max_value=2100, value=default_season)

st.sidebar.header("Live")
auto = st.sidebar.toggle("Auto-refresh", value=True)
interval = st.sidebar.number_input("Poll interval (s)", min_value=1, max_value=30, value=2)
cushion = st.sidebar.slider("Survival cushion (picks)", 2, 18, 6, help="ADP margin for 🟢/🟡/🔴.")
run_window = st.sidebar.slider("Run window (picks)", 4, 24, 12, help="Window for positional-run counts.")
top_n = st.sidebar.slider("Board rows", 15, 100, 40)
if st.sidebar.button("🔄 Reload projections / settings"):
    st.cache_data.clear()
    st.rerun()

# --------------------------------------------------------------------------- load + prep (once per full run)
draft = load_draft(draft_id)
settings = draft.get("settings") or {}
cfg = roster.roster_config(settings)
scoring_league_id = draft.get("league_id") or league_id
users = load_users(scoring_league_id)

board = load_vor_board(
    int(season),
    scoring_league_id,
    tuple(sorted(roster.base_starters(cfg).items())),
    roster.flex_slots_total(cfg),
)

draft_order = draft.get("draft_order") or {}
my_slot = int(draft_order[my_user_id]) if my_user_id in draft_order else None
my_picks_all = snake.my_pick_numbers(my_slot, cfg.teams, cfg.rounds) if my_slot else []
names_by_slot = slot_names(draft_order, users)

st.title("🏈 Live draft tracker")
cap = f"{cfg.teams}-team · {cfg.rounds} rounds · snake · scoring from league {scoring_league_id}"
st.caption(cap + (f" · **your slot: {my_slot}**" if my_slot else " · **slot not revealed — tiers + VOR only**"))


# --------------------------------------------------------------------------- live fragment (polls picks)
@st.fragment(run_every=(interval if auto else None))
def live_panel() -> None:
    try:
        picks = client.get_draft_picks(draft_id) or []
    except Exception as e:  # keep the UI alive on a flaky connection
        st.error(f"Could not fetch picks: {e}")
        return

    picks.sort(key=lambda p: p.get("pick_no") or 0)
    picks_made = len(picks)
    drafted = roster.drafted_ids(picks)
    last_seen = st.session_state.get("last_pick_no", 0)
    fresh = roster.new_since(picks, last_seen)
    st.session_state["last_pick_no"] = picks_made

    # --- our roster so far + needs
    mine = roster.my_drafted(picks, my_user_id)
    my_positions = [roster.pick_position(p) for p in mine if roster.pick_position(p)]
    status = roster.roster_status(my_positions, cfg)
    needs = roster.needed_positions(status, cfg)

    # --- where are we / our next picks
    upcoming = snake.upcoming_picks(my_picks_all, picks_made) if my_slot else []
    next_pick = upcoming[0] if upcoming else None

    # --- header metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Picks made", picks_made)
    if picks_made < cfg.teams * cfg.rounds:
        rnd, on_slot = slot_of_pick(picks_made + 1, cfg.teams)
        who = names_by_slot.get(on_slot, f"slot {on_slot}")
        m2.metric("On the clock", who, f"R{rnd} · pick {picks_made + 1}")
    else:
        m2.metric("On the clock", "— draft complete —")
    m3.metric("Your next pick", f"#{next_pick}" if next_pick else ("—" if my_slot else "slot TBD"))
    m4.metric("Picks until then", (next_pick - picks_made - 1) if next_pick else "—")

    if fresh and last_seen:
        st.success("🆕 " + " · ".join(f"#{p['pick_no']} {fmt_player(p)}" for p in fresh[-5:]))

    left, right = st.columns([5, 2], gap="large")

    # ----- best available board
    with left:
        st.subheader("Best available — by VOR")
        all_pos = ["QB", "RB", "WR", "TE", "K", "DEF"]
        sel = st.multiselect("Positions", all_pos, default=all_pos, key="posfilter")
        avail = [p for p in board if p.player_id not in drafted and p.pos in sel]
        surv_col = f"@#{next_pick}" if next_pick else None
        rows = []
        for p in avail[:top_n]:
            row = {
                "Need": "🎯" if p.pos in needs else "",
                "Tier": f"{p.pos}{p.tier}",  # already encodes position, so we drop a separate Pos column
                "Player": p.name,
                "Tm": p.team or "",
                "Proj": round(p.proj_pts, 1),
                "VOR": round(p.vor, 1),
                # keep the column homogeneously string -- mixing float + "—" breaks Arrow serialization
                "ADP": "—" if p.adp == float("inf") else f"{p.adp:.1f}",
            }
            if surv_col:
                row[surv_col] = _SURVIVAL_BADGE[snake.survival(p.adp, next_pick, cushion=cushion)]
            rows.append(row)

        # Explicit compact widths so ADP + the survival dots always fit (no horizontal scroll).
        col_cfg = {
            "Need": st.column_config.TextColumn("🎯", width="small"),
            "Tier": st.column_config.TextColumn("Tier", width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Tm": st.column_config.TextColumn("Tm", width="small"),
            "Proj": st.column_config.NumberColumn("Proj", width="small", format="%.1f"),
            "VOR": st.column_config.NumberColumn("VOR", width="small", format="%.1f"),
            "ADP": st.column_config.TextColumn("ADP", width="small", help="Market half-PPR ADP."),
        }
        if surv_col:
            col_cfg[surv_col] = st.column_config.TextColumn(
                surv_col, width="small",
                help="Survives to your next pick — 🟢 likely · 🟡 toss-up · 🔴 likely gone.",
            )
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch", height=620, column_config=col_cfg
        )
        st.caption("🎯 = open roster need · 🟢/🟡/🔴 = survives to your next pick · "
                   "K/DEF: stream weekly, don't reach early.")

    with right:
        # ----- roster needs
        st.subheader("Your roster")
        order = ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF"]
        need_rows = [
            {"Slot": s, "Filled": f"{status[s]['filled']}/{status[s]['slots']}",
             "Open": "🔴" * status[s]["need"]}
            for s in order if status[s]["slots"]
        ]
        st.dataframe(pd.DataFrame(need_rows), hide_index=True, width="stretch")
        st.caption(f"Bench: {cfg.slots.get('BN', 0)} · drafted {len(mine)}/{cfg.rounds}")

        # ----- positional runs
        st.subheader(f"Positional run (last {run_window})")
        runs = roster.positional_runs([roster.pick_position(p) for p in picks if roster.pick_position(p)], run_window)
        if runs:
            st.bar_chart(pd.Series({k: runs.get(k, 0) for k in all_pos}), horizontal=True, height=200)
        else:
            st.caption("No picks yet.")

        # ----- my upcoming picks + survivors
        if my_slot and upcoming:
            st.subheader("Targets likely to survive")
            for pk in upcoming[:4]:
                survivors = [
                    p for p in board
                    if p.player_id not in drafted
                    and snake.survival(p.adp, pk, cushion=cushion) == snake.AVAILABLE
                    and p.pos in (needs or {"QB", "RB", "WR", "TE", "K", "DEF"})
                ][:5]
                names = ", ".join(f"{p.name} ({p.pos})" for p in survivors) or "—"
                st.markdown(f"**#{pk}** (R{slot_of_pick(pk, cfg.teams)[0]}): {names}")
        elif not my_slot:
            st.info("Slot not revealed yet — snake picks & survival flags unlock once `draft_order` populates.")

    # ----- recent picks log
    st.subheader("Recent picks")
    log = [
        {"#": p["pick_no"], "Rd": p["round"],
         "Team": names_by_slot.get(p.get("draft_slot"), users.get(str(p.get("picked_by")), "?")),
         "Player": fmt_player(p)}
        for p in reversed(picks[-15:])
    ]
    st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch", height=300)


live_panel()

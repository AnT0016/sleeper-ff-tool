"""Live glue for the Monte Carlo draft simulator (the only networked module here).

Pulls this league's exact ``scoring_settings`` + ``roster_positions`` from the API, builds the
custom-scored draft board (re-scored in our settings via the Phase 1 engine), layers on VOR with the
data-driven FLEX allocation (Phase 2), and resolves my draft slot — revealed from ``draft_order`` if
the slot is out yet, else taken from ``--slot``, else a middle-slot fallback (CLAUDE.md keeps draft
prep slot-agnostic until the slot is revealed). Read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from draft.roster import RosterConfig
from draft.vor import add_vor, replacement_levels
from projections.board import PlayerRow, build_board
from sleeper import client

# Sleeper flex variants that draw from RB/WR/TE; our league only uses plain FLEX, but normalize any.
_FLEX_KEYS = frozenset({"FLEX", "WRRB_FLEX", "REC_FLEX", "W_R_T", "WRT_FLEX"})
_DEDICATED = ("QB", "RB", "WR", "TE", "K", "DEF")


@dataclass
class SimInputs:
    board: list[PlayerRow]  # custom-scored, with VOR set
    cfg: RosterConfig
    my_slot: int
    slot_source: str  # "revealed (draft_order)" | "supplied" | "fallback (slot unknown)"
    season: int
    scoring: dict


def roster_config_from_league(league: dict) -> RosterConfig:
    """Build a :class:`RosterConfig` from a league's live ``roster_positions`` (not draft settings).

    ``rounds`` = draftable spots = all roster slots except IR (IR is never drafted).
    """
    rp = league.get("roster_positions") or []
    slots = {k: 0 for k in (*_DEDICATED, "FLEX", "BN")}
    for p in rp:
        if p in slots:
            slots[p] += 1
        elif p in _FLEX_KEYS:
            slots["FLEX"] += 1
        # IR / TAXI / unknown -> not a draftable slot
    settings = league.get("settings") or {}
    teams = int(league.get("total_rosters") or settings.get("num_teams") or 12)
    rounds = sum(slots.values())
    return RosterConfig(teams=teams, rounds=rounds, slots=slots)


def resolve_slot(
    league_id: str, user_id: str, requested: int | None, teams: int, *, sleeper=client
) -> tuple[int, str]:
    """Resolve my draft slot: explicit ``requested`` wins, else a revealed ``draft_order``, else
    a middle-of-the-board fallback (with a label saying so)."""
    if requested:
        return int(requested), "supplied"
    try:
        drafts = sleeper.get_league_drafts(league_id) or []
        if drafts:
            draft = sleeper.get_draft(drafts[0]["draft_id"])
            order = draft.get("draft_order") or {}
            if str(user_id) in order:
                return int(order[str(user_id)]), "revealed (draft_order)"
    except Exception:
        pass
    return (teams + 1) // 2, "fallback (slot unknown — pass --slot)"


def load_sim_inputs(
    league_id: str,
    user_id: str,
    season: int,
    *,
    slot: int | None = None,
    sleeper=client,
) -> SimInputs:
    """Fetch + assemble everything the engine needs for one league/season (live, cached)."""
    league = sleeper.get_league(league_id)
    scoring = league["scoring_settings"]
    cfg = roster_config_from_league(league)

    board = build_board(season, scoring)
    base = {pos: cfg.slots.get(pos, 0) * cfg.teams for pos in _DEDICATED}
    replacement = replacement_levels(board, base, flex_slots=cfg.slots.get("FLEX", 0) * cfg.teams)
    add_vor(board, replacement)
    board.sort(key=lambda p: p.vor, reverse=True)

    my_slot, slot_source = resolve_slot(league_id, user_id, slot, cfg.teams, sleeper=sleeper)
    if not 1 <= my_slot <= cfg.teams:
        raise ValueError(f"slot {my_slot} out of range for a {cfg.teams}-team draft (use 1..{cfg.teams})")
    return SimInputs(
        board=board,
        cfg=cfg,
        my_slot=my_slot,
        slot_source=slot_source,
        season=season,
        scoring=scoring,
    )

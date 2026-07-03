"""Roster-need tracking, positional-run detection, and live-pick helpers.

Reads the draft's ``settings`` (slots_qb/rb/wr/te/flex/k/def/bn, teams, rounds) into a
``RosterConfig``, tracks which of *our* starting slots are still open as we draft, and surfaces
positional runs (a burst of one position going off the board) from the live pick feed. All pure --
the Streamlit app feeds it raw Sleeper pick dicts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Dedicated (non-FLEX, non-bench) starting slots, in display order.
_DEDICATED = ("QB", "RB", "WR", "TE", "K", "DEF")
FLEX_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")


@dataclass
class RosterConfig:
    teams: int
    rounds: int
    slots: dict[str, int]  # keys: QB, RB, WR, TE, FLEX, K, DEF, BN
    flex_positions: tuple[str, ...] = FLEX_POSITIONS


def roster_config(settings: Mapping) -> RosterConfig:
    """Build a ``RosterConfig`` from a Sleeper draft ``settings`` dict."""
    slots = {
        "QB": int(settings.get("slots_qb", 0)),
        "RB": int(settings.get("slots_rb", 0)),
        "WR": int(settings.get("slots_wr", 0)),
        "TE": int(settings.get("slots_te", 0)),
        "FLEX": int(settings.get("slots_flex", 0)),
        "K": int(settings.get("slots_k", 0)),
        "DEF": int(settings.get("slots_def", 0)),
        "BN": int(settings.get("slots_bn", 0)),
    }
    teams = int(settings.get("teams", 12))
    rounds = int(settings.get("rounds", sum(slots.values())))
    return RosterConfig(teams=teams, rounds=rounds, slots=slots)


def base_starters(cfg: RosterConfig) -> dict[str, int]:
    """League-wide guaranteed starters per position (``teams x dedicated slot``) for VOR baselines.
    Excludes FLEX (allocated separately) and bench."""
    return {pos: cfg.slots.get(pos, 0) * cfg.teams for pos in _DEDICATED}


def flex_slots_total(cfg: RosterConfig) -> int:
    """League-wide FLEX slots (``teams x slots_flex``)."""
    return cfg.slots.get("FLEX", 0) * cfg.teams


def roster_status(my_positions: Sequence[str], cfg: RosterConfig) -> dict[str, dict[str, int]]:
    """Per-slot fill status for our roster given the positions we've drafted.

    Dedicated slots fill first; flex-eligible players beyond their dedicated slots fill FLEX. Each
    entry is ``{"slots": n, "filled": k, "need": n-k}``.
    """
    counts = Counter(my_positions)
    status: dict[str, dict[str, int]] = {}
    flex_pool = 0
    for pos in _DEDICATED:
        slot = cfg.slots.get(pos, 0)
        filled = min(counts.get(pos, 0), slot)
        status[pos] = {"slots": slot, "filled": filled, "need": slot - filled}
        if pos in cfg.flex_positions:
            flex_pool += max(counts.get(pos, 0) - slot, 0)
    flex_slot = cfg.slots.get("FLEX", 0)
    flex_filled = min(flex_pool, flex_slot)
    status["FLEX"] = {"slots": flex_slot, "filled": flex_filled, "need": flex_slot - flex_filled}
    return status


def needed_positions(status: Mapping[str, Mapping[str, int]], cfg: RosterConfig) -> set[str]:
    """Positions worth highlighting on the board: any dedicated slot still open, plus all
    flex-eligible positions if the FLEX is still open."""
    need = {pos for pos in _DEDICATED if status[pos]["need"] > 0}
    if status["FLEX"]["need"] > 0:
        need |= set(cfg.flex_positions)
    return need


# --------------------------------------------------------------------------- live-pick helpers
def pick_position(pick: Mapping) -> str | None:
    return (pick.get("metadata") or {}).get("position")


def drafted_ids(picks: Sequence[Mapping]) -> set[str]:
    """Set of player_ids already drafted (team abbreviation for DEF)."""
    return {str(p.get("player_id")) for p in picks if p.get("player_id") is not None}


def my_drafted(picks: Sequence[Mapping], user_id: str, *, my_slot: int | None = None) -> list[dict]:
    """Our picks so far, ordered by overall pick number.

    Matches on ``picked_by`` OR (when known) on ``draft_slot`` — a CPU autopick after a missed
    clock, or a commissioner-made pick, may not carry our user id in ``picked_by``, but the slot
    always owns its picks in a snake draft.
    """
    mine = [
        p
        for p in picks
        if str(p.get("picked_by")) == str(user_id)
        or (my_slot is not None and int(p.get("draft_slot") or 0) == int(my_slot))
    ]
    return sorted(mine, key=lambda p: p.get("pick_no") or 0)


def new_since(picks: Sequence[Mapping], last_pick_no: int) -> list[dict]:
    """Picks with ``pick_no`` greater than the last one we'd already seen (for the live log)."""
    return sorted(
        (p for p in picks if (p.get("pick_no") or 0) > last_pick_no),
        key=lambda p: p.get("pick_no") or 0,
    )


def positional_runs(pick_positions: Sequence[str], window: int = 12) -> Counter:
    """Count positions taken in the last ``window`` picks -- a high count signals a run."""
    return Counter(pick_positions[-window:])

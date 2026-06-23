"""Snake-draft pick numbers and ADP-based survival likelihood.

Slot-agnostic until ``draft_order`` reveals our slot (~1 day to 1 week before the draft, per
CLAUDE.md). Once the slot ``S`` is known, our overall pick number in round ``r`` of a ``teams``-team
snake is ``(r-1)*teams + S`` on forward (odd) rounds and ``r*teams - S + 1`` on reverse (even)
rounds. Survival compares a target's market ADP to a pick number to flag whether it should still be
on the board when we're up.
"""

from __future__ import annotations

from collections.abc import Sequence

# Survival labels (relative to a given pick number).
AVAILABLE = "available"
TOSSUP = "tossup"
GONE = "gone"


def my_pick_numbers(slot: int, teams: int, rounds: int) -> list[int]:
    """Overall pick numbers for draft ``slot`` (1-indexed) across a ``teams``-team snake draft."""
    picks: list[int] = []
    for r in range(1, rounds + 1):
        if r % 2 == 1:  # forward round
            picks.append((r - 1) * teams + slot)
        else:  # reverse round
            picks.append(r * teams - slot + 1)
    return picks


def upcoming_picks(my_picks: Sequence[int], picks_made: int) -> list[int]:
    """Our pick numbers still ahead of us, given ``picks_made`` picks already off the board."""
    return [p for p in my_picks if p > picks_made]


def survival(adp: float, pick_no: int, *, cushion: int = 6) -> str:
    """Will a player with market ``adp`` survive to overall ``pick_no``?

    ``AVAILABLE`` if their ADP is at least ``cushion`` picks later than ``pick_no`` (or they're
    effectively undrafted), ``GONE`` if at least ``cushion`` picks earlier, else ``TOSSUP``.
    """
    if adp == float("inf"):
        return AVAILABLE
    margin = adp - pick_no
    if margin >= cushion:
        return AVAILABLE
    if margin <= -cushion:
        return GONE
    return TOSSUP

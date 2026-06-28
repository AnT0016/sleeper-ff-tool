"""Fast best-lineup valuation for the simulator's single-FLEX roster (no PuLP in the hot loop).

For our slot structure — dedicated slots (1 QB, 2 RB, 2 WR, 1 TE, 1 K, 1 DEF) plus *one* FLEX from
{RB, WR, TE} — the greedy fill is provably optimal: take the top scorers for each dedicated slot,
then the single best leftover flex-eligible player for FLEX. (With one flex slot there is no
swap that improves the total, so we skip the LP and score thousands of rosters per second on
*sampled* season points.) The Phase 3 PuLP optimizer remains the source of truth for live weekly
lineups; this is its cheap, sim-only cousin.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

# Dedicated (non-FLEX) lineup slots.
DEDICATED: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")
FLEX_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")


def best_lineup_points(
    positions: Sequence[str],
    points: Sequence[float],
    slots: Mapping[str, int],
    *,
    flex_positions: Sequence[str] = FLEX_POSITIONS,
) -> float:
    """Best legal starting-lineup total for a roster scored on ``points``.

    ``positions`` and ``points`` are parallel (one entry per rostered player). ``slots`` is the
    league's slot dict (``QB/RB/WR/TE/FLEX/K/DEF`` counts). A position with too few players simply
    fills what it can — missing starters contribute 0 (the roster left a hole), never an error.
    """
    by_pos: dict[str, list[float]] = defaultdict(list)
    for pos, pts in zip(positions, points):
        by_pos[pos].append(float(pts))
    for vals in by_pos.values():
        vals.sort(reverse=True)

    total = 0.0
    used: dict[str, int] = {}
    for pos in DEDICATED:
        n = int(slots.get(pos, 0))
        vals = by_pos.get(pos, ())
        total += sum(vals[:n])
        used[pos] = min(n, len(vals))

    flex_n = int(slots.get("FLEX", 0))
    if flex_n:
        leftovers: list[float] = []
        for pos in flex_positions:
            leftovers.extend(by_pos.get(pos, ())[used.get(pos, 0):])
        leftovers.sort(reverse=True)
        total += sum(leftovers[:flex_n])
    return total


def select_starters(
    positions: Sequence[str],
    points: Sequence[float],
    slots: Mapping[str, int],
    *,
    flex_positions: Sequence[str] = FLEX_POSITIONS,
) -> list[int]:
    """Indices (into ``positions``/``points``) of the players who'd start in the best lineup.

    Mirrors :func:`best_lineup_points` but returns the chosen roster slots rather than the total —
    used to tell apart starters from bench depth (e.g. for the injury / backup report).
    """
    by_pos: dict[str, list[int]] = defaultdict(list)
    for i, pos in enumerate(positions):
        by_pos[pos].append(i)
    for vals in by_pos.values():
        vals.sort(key=lambda i: points[i], reverse=True)

    starters: list[int] = []
    used: dict[str, int] = {}
    for pos in DEDICATED:
        n = int(slots.get(pos, 0))
        idxs = by_pos.get(pos, [])
        starters.extend(idxs[:n])
        used[pos] = min(n, len(idxs))

    flex_n = int(slots.get("FLEX", 0))
    if flex_n:
        leftovers = [i for pos in flex_positions for i in by_pos.get(pos, [])[used.get(pos, 0):]]
        leftovers.sort(key=lambda i: points[i], reverse=True)
        starters.extend(leftovers[:flex_n])
    return starters

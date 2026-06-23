"""Value over replacement (VOR) and tiers for the custom-scored board.

VOR lets us compare players *across positions* by how much they beat a freely-available
replacement at their own position -- the right lens for a snake draft (when to take a scarce TE
over a deep-pool WR, when K/DEF stop mattering). Replacement level is the projection of the first
player at a position who is *not* expected to start league-wide; FLEX slots are allocated to
positions data-drivenly (the best leftover RB/WR/TE fill them).

Tiers are per-position clusters: a new tier begins where the value drop to the next player exceeds
a gap threshold scaled to that position's spread -- so "last player in the tier" marks a cliff
worth reaching for.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from projections.board import PlayerRow

FLEX_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")


def replacement_levels(
    board: Sequence[PlayerRow],
    base_starters: Mapping[str, int],
    *,
    flex_slots: int = 0,
    flex_positions: Sequence[str] = FLEX_POSITIONS,
) -> dict[str, float]:
    """Replacement-level projection per position.

    ``base_starters`` is the league-wide count of guaranteed starters at each position
    (e.g. 12-team, 2 RB starters -> ``RB: 24``). ``flex_slots`` (league-wide) are then handed to the
    best leftover ``flex_positions`` players, deepening those positions' replacement level. The
    replacement is the projection of the first non-starter at each position.
    """
    by_pos: dict[str, list[PlayerRow]] = defaultdict(list)
    for p in board:
        by_pos[p.pos].append(p)
    for players in by_pos.values():
        players.sort(key=lambda p: p.proj_pts, reverse=True)

    depth = {pos: int(n) for pos, n in base_starters.items()}
    if flex_slots:
        leftover: list[PlayerRow] = []
        for pos in flex_positions:
            leftover.extend(by_pos.get(pos, [])[base_starters.get(pos, 0):])
        leftover.sort(key=lambda p: p.proj_pts, reverse=True)
        for p in leftover[:flex_slots]:
            depth[p.pos] = depth.get(p.pos, 0) + 1

    repl: dict[str, float] = {}
    for pos, players in by_pos.items():
        if not players:
            continue
        d = depth.get(pos, 0)
        repl[pos] = players[d].proj_pts if d < len(players) else players[-1].proj_pts
    return repl


def add_vor(board: Sequence[PlayerRow], replacement: Mapping[str, float]) -> Sequence[PlayerRow]:
    """Set ``row.vor = proj_pts - replacement[pos]`` in place; returns the same board."""
    for p in board:
        p.vor = round(p.proj_pts - replacement.get(p.pos, 0.0), 2)
    return board


def tierize(
    board: Sequence[PlayerRow],
    *,
    by: str = "vor",
    gap_mult: float = 1.6,
    tier_depth: int = 24,
    min_gap: float = 0.5,
) -> Sequence[PlayerRow]:
    """Assign per-position tiers (1 = best) in place by clustering on ``by`` (``vor`` or
    ``proj_pts``).

    A new tier starts where the drop to the next player exceeds ``gap_mult x`` the median gap over
    the position's top ``tier_depth`` players (floored at ``min_gap``). Tiers restart per position.
    """
    by_pos: dict[str, list[PlayerRow]] = defaultdict(list)
    for p in board:
        by_pos[p.pos].append(p)

    for players in by_pos.values():
        players.sort(key=lambda p: getattr(p, by), reverse=True)
        vals = [getattr(p, by) for p in players]
        gaps = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
        segment = [g for g in gaps[:tier_depth] if g > 0] or [min_gap]
        threshold = max(statistics.median(segment) * gap_mult, min_gap)
        tier = 1
        for i, p in enumerate(players):
            p.tier = tier
            if i < len(gaps) and gaps[i] > threshold:
                tier += 1
    return board

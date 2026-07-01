"""Vectorised weekly lineup valuation across all sims at once.

The season sim scores 12 teams × 17 weeks × thousands of sims, so the per-roster greedy fill from
:mod:`draftsim.lineup` (a Python loop) is too slow. Here the roster is fixed for a team, so a week's
points form an ``(n_sims, roster_size)`` matrix and the whole slate of sims is filled with a handful
of ``numpy`` sorts.

The one generalisation over the draft sim: starters are chosen by a **selection** score but scored by
a (possibly different) **value** matrix. With ``select == value`` this is the hindsight-optimal lineup
(the ceiling); with ``select`` = projected means it's the lineup a manager would actually set, scored
by what really happened — which is how the sim separates roster quality from start/sit skill. For our
single-FLEX slot structure the greedy fill is provably optimal (see :mod:`draftsim.lineup`).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

DEDICATED: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")
FLEX_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")


def lineup_values(
    positions: Sequence[str],
    select: np.ndarray,
    value: np.ndarray,
    slots: Mapping[str, int],
    *,
    flex_positions: Sequence[str] = FLEX_POSITIONS,
) -> np.ndarray:
    """``(n_sims,)`` best-lineup value per sim: pick starters by ``select``, sum their ``value``.

    ``positions`` labels the roster columns (length = roster size). ``select`` and ``value`` are both
    ``(n_sims, roster_size)``. A position with fewer players than slots simply fills what it can (a
    roster hole scores 0, never an error).
    """
    select = np.asarray(select, dtype=float)
    value = np.asarray(value, dtype=float)
    n_sims = select.shape[0]

    cols_by_pos: dict[str, list[int]] = defaultdict(list)
    for i, pos in enumerate(positions):
        cols_by_pos[pos].append(i)

    total = np.zeros(n_sims, dtype=float)
    leftover_val: list[np.ndarray] = []
    leftover_sel: list[np.ndarray] = []
    flex_set = set(flex_positions)

    for pos in DEDICATED:
        cols = cols_by_pos.get(pos)
        if not cols:
            continue
        n = int(slots.get(pos, 0))
        sel = select[:, cols]
        val = value[:, cols]
        order = np.argsort(-sel, axis=1, kind="stable")
        sorted_val = np.take_along_axis(val, order, axis=1)
        take = min(n, len(cols))
        if take:
            total += sorted_val[:, :take].sum(axis=1)
        if pos in flex_set and take < len(cols):
            sorted_sel = np.take_along_axis(sel, order, axis=1)
            leftover_val.append(sorted_val[:, take:])
            leftover_sel.append(sorted_sel[:, take:])

    flex_n = int(slots.get("FLEX", 0))
    if flex_n and leftover_val:
        lval = np.concatenate(leftover_val, axis=1)
        lsel = np.concatenate(leftover_sel, axis=1)
        order = np.argsort(-lsel, axis=1, kind="stable")
        sorted_lval = np.take_along_axis(lval, order, axis=1)
        take = min(flex_n, sorted_lval.shape[1])
        if take:
            total += sorted_lval[:, :take].sum(axis=1)
    return total

"""Weekly lineup optimizer -- the integer LP core (no network).

Given a set of rostered players with custom-scored weekly projections and per-player eligibility
(bye / OUT / IR exclusions resolved upstream in ``optimizer.inputs``), pick the single
highest-projected legal starting lineup for this league's exact slot rules.

The slot rules themselves are read live from the league's ``roster_positions`` (the locked source of
truth -- never hand-coded): for our league that is ``1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX{RB/WR/TE},
1 K, 1 DEF``. Bench/IR/taxi slots are not startable and are ignored here.

Formulation: one binary variable per *(eligible player x slot the player's position may fill)*.
Position and FLEX eligibility are enforced structurally -- a variable simply does not exist for an
illegal (player, slot) pair. Constraints: each player fills at most one slot; each slot is filled up
to its capacity. The objective maximizes total projected points (with a negligible per-assignment
nudge so the solver always *fills* a slot when any eligible player remains, rather than leaving a
0-projection K/DEF on the bench). When the eligible pool cannot fill a slot, the lineup is returned
partially filled with that slot reported in ``holes`` -- never an exception.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pulp

#: Startable lineup slot -> the set of positions eligible to fill it. Covers the common Sleeper
#: offensive slot codes; our league uses QB/RB/WR/TE/FLEX/K/DEF. Bench (``BN``), ``IR`` and
#: ``TAXI`` are intentionally absent -- they are not startable.
FLEX_ELIGIBILITY: dict[str, frozenset[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "DEF": frozenset({"DEF"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "WRRB_FLEX": frozenset({"RB", "WR"}),
    "REC_FLEX": frozenset({"WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
}

#: Canonical display order for lineup slots.
_SLOT_ORDER: tuple[str, ...] = (
    "QB", "RB", "WR", "TE", "FLEX", "WRRB_FLEX", "REC_FLEX", "SUPER_FLEX", "K", "DEF",
)

#: Negligible per-assignment reward: << any real projection difference, so it never changes *which*
#: player starts -- it only breaks ties toward filling an otherwise-empty slot with a 0-proj player.
_FILL_NUDGE = 1e-4


@dataclass
class LineupPlayer:
    """One rostered player as seen by the optimizer. Eligibility is resolved upstream:
    ``on_bye`` / ``out`` / ``on_ir`` players are hard-excluded from starting; ``status`` (the Sleeper
    ``injury_status``, e.g. "Questionable") is carried through for the risky-start flags."""

    player_id: str
    name: str
    pos: str
    team: str | None
    proj_pts: float
    status: str | None = None  # Sleeper injury_status: Questionable / Doubtful / Out / IR / ...
    on_bye: bool = False
    out: bool = False
    on_ir: bool = False

    @property
    def eligible(self) -> bool:
        """Startable this week: not on bye, not OUT, not on IR."""
        return not (self.on_bye or self.out or self.on_ir)

    @property
    def block_reason(self) -> str | None:
        if self.on_ir:
            return "IR"
        if self.out:
            return "OUT"
        if self.on_bye:
            return "BYE"
        return None


@dataclass
class StarterSpot:
    slot: str
    player: LineupPlayer


@dataclass
class LineupSolution:
    starters: list[StarterSpot]
    total: float
    bench: list[LineupPlayer]  # eligible players not chosen to start
    holes: dict[str, int] = field(default_factory=dict)  # slot -> unfilled count
    status: str = "Optimal"


def lineup_slots(roster_positions: Sequence[str]) -> "OrderedDict[str, int]":
    """Count startable slots from a league's ``roster_positions`` list.

    ``roster_positions`` (pulled live from the league object) looks like
    ``["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", ...]``. Bench / IR / taxi
    and any unrecognized (e.g. IDP) codes are dropped; the result is ordered for display.
    """
    counts: dict[str, int] = {}
    for slot in roster_positions:
        if slot in FLEX_ELIGIBILITY:
            counts[slot] = counts.get(slot, 0) + 1
    return OrderedDict(
        (slot, counts[slot]) for slot in _SLOT_ORDER if slot in counts
    )


def optimize(
    players: Sequence[LineupPlayer],
    slots: Mapping[str, int],
    *,
    flex_eligibility: Mapping[str, frozenset[str]] = FLEX_ELIGIBILITY,
) -> LineupSolution:
    """Solve for the highest-projected legal starting lineup.

    ``players`` is our roster; only ``eligible`` players (not bye/OUT/IR) are considered. ``slots`` is
    the ``{slot: count}`` map from :func:`lineup_slots`. Returns the chosen starters (in slot order),
    the projected total, the eligible bench, and any slots that could not be filled (``holes``).
    """
    # Deterministic order (proj desc, then id) so equal-value ties resolve identically every run.
    pool = sorted(
        (p for p in players if p.eligible),
        key=lambda p: (-p.proj_pts, p.player_id),
    )

    prob = pulp.LpProblem("weekly_lineup", pulp.LpMaximize)

    # x[(i, slot)] = 1 iff player i starts in `slot`. Created only for legal (position-eligible) pairs.
    x: dict[tuple[int, str], pulp.LpVariable] = {}
    for i, p in enumerate(pool):
        for slot in slots:
            if p.pos in flex_eligibility.get(slot, frozenset()):
                x[(i, slot)] = prob.add_variable(f"x_{i}_{slot}", 0, 1, cat="Binary")

    # Objective: maximize projected points, with a negligible nudge to fill empty slots.
    prob += pulp.lpSum(
        (pool[i].proj_pts + _FILL_NUDGE) * var for (i, _slot), var in x.items()
    )

    # Each player starts in at most one slot.
    for i in range(len(pool)):
        vars_i = [x[(i, slot)] for slot in slots if (i, slot) in x]
        if vars_i:
            prob += pulp.lpSum(vars_i) <= 1, f"one_slot_p{i}"

    # Each slot is filled up to (and, when the pool allows, exactly to) its capacity.
    for slot, count in slots.items():
        vars_s = [x[(i, slot)] for i in range(len(pool)) if (i, slot) in x]
        prob += pulp.lpSum(vars_s) <= count, f"cap_{slot}"

    # PULP_CBC_CMD ships its own solver binary (the non-deprecated COIN_CMD needs an external `cbc`
    # on PATH, which we don't bundle); silence only its v4 deprecation notice.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    chosen: dict[int, str] = {}
    filled: dict[str, int] = {slot: 0 for slot in slots}
    for (i, slot), var in x.items():
        if var.value() and var.value() > 0.5:
            chosen[i] = slot
            filled[slot] += 1

    starters = [
        StarterSpot(slot=slot, player=pool[i])
        for slot in _ordered_slots(slots)
        for i in sorted(
            (j for j, s in chosen.items() if s == slot),
            key=lambda j: (-pool[j].proj_pts, pool[j].player_id),
        )
    ]
    total = round(sum(s.player.proj_pts for s in starters), 2)
    bench = [p for i, p in enumerate(pool) if i not in chosen]
    holes = {slot: count - filled[slot] for slot, count in slots.items() if filled[slot] < count}
    return LineupSolution(starters=starters, total=total, bench=bench, holes=holes, status=status)


def _ordered_slots(slots: Mapping[str, int]) -> list[str]:
    known = [s for s in _SLOT_ORDER if s in slots]
    extra = [s for s in slots if s not in _SLOT_ORDER]
    return known + extra

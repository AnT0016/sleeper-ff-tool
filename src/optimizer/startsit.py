"""Start/sit deltas and risky-start flags layered on a solved lineup (no network).

Three read-only views over a :class:`~optimizer.lineup.LineupSolution`:

* :func:`start_sit_table` -- for every benched (eligible) player, the starter they would displace and
  by how much. In an optimal lineup every delta is <= 0; the value is *how close* each bench player
  is to cracking the lineup (a near-zero delta is a genuine coin-flip).
* :func:`risky_starts` -- starters worth a second look: ones tagged Questionable/Doubtful, and
  "forced downgrades" where a better rostered player at an eligible position is on bye / OUT / IR, so
  the optimizer had to reach for a weaker option.
* :func:`idle_players` -- rostered players sitting out this week (bye / OUT / IR), for context.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from optimizer.lineup import FLEX_ELIGIBILITY, LineupPlayer, LineupSolution, StarterSpot

#: Injury statuses that make a *chosen* starter risky (still startable, but flag it).
RISKY_STATUSES: frozenset[str] = frozenset({"Questionable", "Doubtful"})


@dataclass
class BenchDelta:
    """A bench player and the starter they would displace (lowest-projected starter in a slot the
    bench player is eligible for). ``delta = player.proj - would_replace.proj`` -- <= 0 when the
    lineup is optimal."""

    player: LineupPlayer
    would_replace: LineupPlayer | None
    slot: str | None
    delta: float


@dataclass
class RiskFlag:
    slot: str
    player: LineupPlayer
    reasons: list[str] = field(default_factory=list)


def _slot_starters(solution: LineupSolution) -> dict[str, list[StarterSpot]]:
    by_slot: dict[str, list[StarterSpot]] = defaultdict(list)
    for sp in solution.starters:
        by_slot[sp.slot].append(sp)
    return by_slot


def start_sit_table(
    solution: LineupSolution,
    *,
    flex_eligibility: Mapping[str, frozenset[str]] = FLEX_ELIGIBILITY,
) -> list[BenchDelta]:
    """One row per eligible bench player: the starter they'd replace and the projection delta.

    Sorted best-first (largest delta), so the bench players closest to starting come first.
    """
    starters = solution.starters
    rows: list[BenchDelta] = []
    for b in solution.bench:
        candidates = [
            sp for sp in starters if b.pos in flex_eligibility.get(sp.slot, frozenset())
        ]
        if candidates:
            repl = min(candidates, key=lambda sp: sp.player.proj_pts)
            rows.append(
                BenchDelta(
                    player=b,
                    would_replace=repl.player,
                    slot=repl.slot,
                    delta=round(b.proj_pts - repl.player.proj_pts, 2),
                )
            )
        else:
            rows.append(BenchDelta(player=b, would_replace=None, slot=None, delta=0.0))
    rows.sort(key=lambda r: r.delta, reverse=True)
    return rows


def risky_starts(
    solution: LineupSolution,
    players: Sequence[LineupPlayer],
    *,
    flex_eligibility: Mapping[str, frozenset[str]] = FLEX_ELIGIBILITY,
) -> list[RiskFlag]:
    """Flag chosen starters that warrant attention.

    Two triggers:

    * **Injury** -- the starter is Questionable or Doubtful. If, on top of that, no eligible bench
      player can cover their slot, we add a "no backup" note (your "no clean replacement" case).
    * **Forced downgrade** -- a higher-projected rostered player at an eligible position is on
      bye / OUT / IR, so the lineup had to use a weaker starter. Each unavailable stud flags the
      single weakest eligible starter it would have displaced.
    """
    flags: dict[int, RiskFlag] = {}

    def flag_for(sp: StarterSpot) -> RiskFlag:
        key = id(sp.player)
        if key not in flags:
            flags[key] = RiskFlag(slot=sp.slot, player=sp.player)
        return flags[key]

    bench = solution.bench

    # Injury flags (+ no-backup note).
    for sp in solution.starters:
        if sp.player.status in RISKY_STATUSES:
            f = flag_for(sp)
            f.reasons.append(sp.player.status)
            elig = flex_eligibility.get(sp.slot, frozenset())
            if not any(b.pos in elig for b in bench):
                f.reasons.append("no eligible backup on the bench")

    # Forced-downgrade flags: an unavailable better player points at the weakest starter it could
    # have replaced.
    for ex in (p for p in players if not p.eligible):
        candidates = [
            sp for sp in solution.starters if ex.pos in flex_eligibility.get(sp.slot, frozenset())
        ]
        if not candidates:
            continue
        repl = min(candidates, key=lambda sp: sp.player.proj_pts)
        if ex.proj_pts > repl.player.proj_pts:
            drop = round(ex.proj_pts - repl.player.proj_pts, 2)
            flag_for(repl).reasons.append(
                f"forced: {ex.name} ({ex.block_reason}) projects +{drop} but is unavailable"
            )

    # Preserve lineup (slot) order.
    order = {id(sp.player): i for i, sp in enumerate(solution.starters)}
    return sorted(flags.values(), key=lambda f: order.get(id(f.player), 0))


def idle_players(players: Sequence[LineupPlayer]) -> list[tuple[LineupPlayer, str]]:
    """Rostered players sitting out this week with the reason (BYE / OUT / IR), best-projected first."""
    idle = [(p, p.block_reason) for p in players if not p.eligible]
    idle.sort(key=lambda pr: pr[0].proj_pts, reverse=True)
    return [(p, r) for p, r in idle if r is not None]

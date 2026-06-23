"""Reverse-standings waiver **priority** spend advice (pure, no network).

This league has no FAAB: waiver priority is a single ordered resource and spending my top claim drops
me to the back. So the only question worth answering is *is this target worth my claim right now?*
Per CLAUDE.md the answer is yes only for players who are **startable upgrades** (positive value over
my current worst starter) or who **just inherited a starting role** -- never for speculative depth.

"Startable upgrade" is measured by reusing the Phase 3 optimizer (:func:`optimizer.lineup.optimize`):
solve my current optimal lineup, then re-solve with the candidate added. The increase in the optimal
total is exactly the candidate's value over the weakest starter it would displace -- if it is <= 0 the
player does not crack my lineup and is not worth a claim.

The recommendation is then tempered by two things:

* my **priority scarcity** (``waivers.league.priority_scarcity``) -- near the top of the standings a
  top claim is scarce and slow to recover (be selective); near the bottom it is durable (be
  aggressive);
* **contention** -- the Sleeper trending-add ``count`` (how fast the player is being added across
  Sleeper) approximates whether I must claim now or can wait for him to clear to free agency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from optimizer.lineup import LineupPlayer, LineupSolution, optimize
from waivers.league import PriorityScarcity

#: A lineup gain at/above this (custom-scored points added to my optimal lineup) is a clear upgrade.
CLEAR_UPGRADE: float = 2.0
#: Any positive gain is a (marginal) upgrade; below this it does not crack the lineup.
UPGRADE_EPS: float = 0.05

#: Trending-add ``count`` thresholds for contention (heuristic, tunable).
CONTENTION_HIGH: int = 400
CONTENTION_MED: int = 100


@dataclass
class SpendCandidate:
    """A free agent worth evaluating for a claim. ``player`` carries this week's re-scored projection
    and eligibility (a :class:`~optimizer.lineup.LineupPlayer`)."""

    player: LineupPlayer
    is_new_starter: bool = False  # just inherited a starting role (e.g. handcuff whose starter is OUT)
    contention: int = 0  # Sleeper trending-add count (velocity / how contested)


@dataclass(frozen=True)
class SpendAdvice:
    player_id: str
    name: str
    pos: str
    team: str | None
    lineup_gain: float
    is_new_starter: bool
    contention: int
    contention_level: str  # "high" | "medium" | "low"
    verdict: str  # "spend" | "stream-later" | "hold"
    reason: str


def _contention_level(count: int) -> str:
    if count >= CONTENTION_HIGH:
        return "high"
    if count >= CONTENTION_MED:
        return "medium"
    return "low"


def lineup_gain(
    my_players: Sequence[LineupPlayer],
    slots,
    candidate: LineupPlayer,
    *,
    optimize_fn: Callable[..., LineupSolution] = optimize,
    base_total: float | None = None,
) -> float:
    """Custom-scored points the candidate adds to my optimal lineup this week (>= 0 in practice).

    ``base_total`` (my current optimal total) can be passed to avoid re-solving it per candidate.
    """
    if base_total is None:
        base_total = optimize_fn(my_players, slots).total
    new_total = optimize_fn([*my_players, candidate], slots).total
    return round(new_total - base_total, 2)


def spend_advice(
    candidates: Sequence[SpendCandidate],
    my_players: Sequence[LineupPlayer],
    slots,
    scarcity: PriorityScarcity,
    *,
    optimize_fn: Callable[..., LineupSolution] = optimize,
) -> list[SpendAdvice]:
    """Rank claim targets with a ``spend | stream-later | hold`` verdict.

    Solves my base optimal lineup once, then measures each candidate's :func:`lineup_gain`. Verdict:

    * **spend** -- a clear upgrade, a new starter, or a marginal upgrade I shouldn't risk losing
      (aggressive priority posture, or high contention);
    * **stream-later** -- a marginal upgrade I can likely add off waivers without burning a top claim;
    * **hold** -- no startable upgrade; not worth the ordered priority resource.
    """
    base_total = optimize_fn(my_players, slots).total
    rows: list[SpendAdvice] = []
    for c in candidates:
        p = c.player
        gain = lineup_gain(my_players, slots, p, optimize_fn=optimize_fn, base_total=base_total)
        level = _contention_level(c.contention)
        verdict, reason = _decide(gain, c.is_new_starter, level, scarcity)
        rows.append(
            SpendAdvice(
                player_id=str(p.player_id),
                name=p.name,
                pos=p.pos,
                team=p.team,
                lineup_gain=gain,
                is_new_starter=c.is_new_starter,
                contention=c.contention,
                contention_level=level,
                verdict=verdict,
                reason=reason,
            )
        )
    order = {"spend": 0, "stream-later": 1, "hold": 2}
    rows.sort(key=lambda r: (order[r.verdict], -r.lineup_gain))
    return rows


def _decide(
    gain: float, is_new_starter: bool, contention_level: str, scarcity: PriorityScarcity
) -> tuple[str, str]:
    if is_new_starter:
        return "spend", "just inherited a starting role -- claim before the field does"
    if gain >= CLEAR_UPGRADE:
        return "spend", f"clear upgrade: +{gain:.2f} to your optimal lineup this week"
    if gain > UPGRADE_EPS:
        if scarcity.posture == "aggressive":
            return "spend", f"upgrade (+{gain:.2f}); you hold durable high priority -- use it"
        if contention_level == "high":
            return "spend", f"upgrade (+{gain:.2f}) and being added fast -- claim now or lose him"
        return (
            "stream-later",
            f"only +{gain:.2f}; with {scarcity.posture} priority and {contention_level} contention, "
            f"add off waivers if he clears",
        )
    return "hold", "no startable upgrade -- not worth spending a claim"

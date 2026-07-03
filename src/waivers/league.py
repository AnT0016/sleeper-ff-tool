"""Shared league/roster helpers for the waiver workflow (pure, no network).

Three things every waiver view needs and that the rest of Phase 4 builds on:

* **who is available** -- the free-agent pool is everyone *not* on a roster (a player on any roster's
  ``players``, ``reserve`` (IR) or ``taxi`` is owned);
* **the standings** -- ranked by wins, then total points (``fpts`` + ``fpts_decimal``), so we know my
  position;
* **my waiver priority** -- this league uses **reverse-standings priority, NOT FAAB**. Priority is a
  single ordered resource (Sleeper exposes my slot as ``settings.waiver_position``); spending my #1
  claim drops me to the back. Near the *top* of the standings my priority is numerically worst and
  slow to recover (scarce -> be selective); near the *bottom* I hold durable high priority (be
  aggressive). :func:`priority_scarcity` encodes that read.

Reuses :func:`optimizer.inputs.find_my_roster` for the owner-id match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from optimizer.inputs import find_my_roster

__all__ = [
    "rostered_player_ids",
    "free_agents",
    "TeamStanding",
    "standings",
    "my_standing",
    "my_waiver_position",
    "priority_scarcity",
]


def _owned(roster: Mapping) -> set[str]:
    ids: set[str] = set()
    for key in ("players", "reserve", "taxi"):
        ids.update(str(x) for x in (roster.get(key) or []))
    return ids


def rostered_player_ids(rosters: Sequence[Mapping]) -> set[str]:
    """Every owned ``player_id`` across the league (active roster + IR ``reserve`` + ``taxi``)."""
    owned: set[str] = set()
    for r in rosters:
        owned |= _owned(r)
    return owned


def free_agents(candidate_ids: object, rosters: Sequence[Mapping]) -> set[str]:
    """Of ``candidate_ids``, those not owned by any roster -- the addable free-agent pool."""
    owned = rostered_player_ids(rosters)
    return {str(pid) for pid in candidate_ids if str(pid) not in owned}


@dataclass(frozen=True)
class TeamStanding:
    rank: int  # 1 = best
    roster_id: int
    owner_id: str | None
    wins: int
    losses: int
    ties: int
    points: float  # fpts + fpts_decimal/100


def _points(settings: Mapping) -> float:
    return float(settings.get("fpts") or 0) + float(settings.get("fpts_decimal") or 0) / 100.0


def standings(rosters: Sequence[Mapping]) -> list[TeamStanding]:
    """League standings, best-first: ordered by wins, then total points (the usual tiebreak)."""
    rows = []
    for r in rosters:
        s = r.get("settings") or {}
        rows.append(
            TeamStanding(
                rank=0,
                roster_id=int(r.get("roster_id")),
                owner_id=(str(r["owner_id"]) if r.get("owner_id") is not None else None),
                wins=int(s.get("wins") or 0),
                losses=int(s.get("losses") or 0),
                ties=int(s.get("ties") or 0),
                points=_points(s),
            )
        )
    rows.sort(key=lambda t: (t.wins, t.points), reverse=True)
    return [
        TeamStanding(rank=i + 1, roster_id=t.roster_id, owner_id=t.owner_id, wins=t.wins,
                     losses=t.losses, ties=t.ties, points=t.points)
        for i, t in enumerate(rows)
    ]


def my_standing(rosters: Sequence[Mapping], user_id: str) -> TeamStanding:
    """My :class:`TeamStanding` (rank within the league)."""
    mine = find_my_roster(rosters, user_id)
    rid = int(mine.get("roster_id"))
    for t in standings(rosters):
        if t.roster_id == rid:
            return t
    raise ValueError(f"roster_id={rid} not found in standings")


def my_waiver_position(roster: Mapping) -> int | None:
    """My current ordered waiver priority (1 = next claim wins), from ``settings.waiver_position``.

    This is the actual ordered resource the league runs on -- not a budget. ``None`` if the league
    object doesn't expose it.
    """
    pos = (roster.get("settings") or {}).get("waiver_position")
    return int(pos) if pos is not None else None


@dataclass(frozen=True)
class PriorityScarcity:
    """How precious my single waiver claim is right now, given reverse-standings priority."""

    rank: int
    n_teams: int
    posture: str  # "aggressive" | "balanced" | "selective"
    note: str


def priority_scarcity(
    rank: int, n_teams: int, *, waiver_position: int | None = None
) -> PriorityScarcity:
    """Read my claim's value from my standings ``rank`` (1 = best) of ``n_teams``, refined by my
    actual slot in the waiver order (``waiver_position``, 1 = next claim wins) when known.

    Reverse-standings priority means the *worst* teams hold the highest (best) waiver priority and
    recover it quickly, while contenders sit at the back and regain a top claim slowly. So:

    * top third of the standings  -> priority is scarce and slow to recover -> **selective**;
    * bottom third                -> durable high priority -> **aggressive**;
    * middle                      -> **balanced**.

    The waiver order refines this: what a claim *costs* is the slot I currently hold. If I already
    sit at the back of the order, "spending" a claim costs me almost nothing regardless of my
    standings — be aggressive. Only a claim near the front is a scarce resource worth protecting.
    """
    n = max(int(n_teams), 1)
    frac = (rank - 1) / max(n - 1, 1)  # 0.0 = best team, 1.0 = worst team
    if waiver_position is not None and n > 1:
        wfrac = (int(waiver_position) - 1) / max(n - 1, 1)  # 0.0 = front of the order, 1.0 = back
        if wfrac >= 2 / 3:
            return PriorityScarcity(
                rank=rank,
                n_teams=n,
                posture="aggressive",
                note=(
                    f"you already sit #{int(waiver_position)} of {n} in the waiver order -- a claim "
                    "costs you almost nothing right now; spend on any startable upgrade"
                ),
            )
        if wfrac <= 1 / 3 and frac <= 1 / 3:
            return PriorityScarcity(
                rank=rank,
                n_teams=n,
                posture="selective",
                note=(
                    f"you hold a front-of-order claim (#{int(waiver_position)} of {n}) as a contender "
                    "-- it is scarce and slow to recover; spend it only on a clear startable upgrade"
                ),
            )
    if frac <= 1 / 3:
        posture, note = "selective", "near the top of the standings: a top claim is scarce and slow to recover -- spend it only on a clear startable upgrade"
    elif frac >= 2 / 3:
        posture, note = "aggressive", "near the bottom: you hold durable high priority -- spend freely on any startable upgrade or new starter"
    else:
        posture, note = "balanced", "mid-table: spend a top claim on a genuine upgrade, otherwise hold"
    return PriorityScarcity(rank=rank, n_teams=n, posture=posture, note=note)

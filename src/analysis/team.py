"""Team & league analysis (pure, no network) for the hosted season dashboard (Phase 5).

Five forward-looking views over the league, all built by reusing earlier phases:

* :func:`slot_points_by_team` / :func:`position_strengths` -- rank my starters' projected points by
  lineup slot against the other 11 rosters (strength / average / weakness). Each team's per-slot
  points come from the Phase 3 optimizer (:func:`optimizer.lineup.optimize`) run on that team's
  roster, so the FLEX is allocated exactly as the lineup rules dictate. The networked glue
  (``analysis.snapshot``) feeds this both ways: season-long projections (stable roster quality) and
  this-week's projections (bye/injury-aware).
* :func:`bye_week_gaps` -- upcoming weeks where a bye forces me to start a backup ("thin") or leaves
  a slot I can't fill ("hole").
* :func:`positional_needs` -- weaknesses + thin depth + bye holes folded into a needs list.
* :func:`trade_targets` -- mutual-fit ideas (a team strong where I'm weak *and* weak where I'm
  strong), surfaced before the Week-11 deadline.
* :func:`playoff_outlook` -- my likely starters' Weeks 15-17 value, each week's baseline tilted by
  the Phase-4 strength-of-schedule (reuses :func:`waivers.stash.rank_playoff_stashes`).

Everything here is pure -- per-team starters, schedule/SOS lookups and rosters are passed in, so the
whole module is unit-testable offline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from waivers.stash import PLAYOFF_WEEKS, StashCandidate, rank_playoff_stashes

#: Base positions worth proposing a trade around (K/DEF are streamed, not traded).
TRADE_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE")

#: SOS multiplier thresholds for labelling a playoff matchup (>1 = soft, the defense allows more).
TOUGH_MATCHUP: float = 0.90
SOFT_MATCHUP: float = 1.10


# --------------------------------------------------------------------------- roster strength
@dataclass(frozen=True)
class PositionStrength:
    """My projected starter points at one lineup ``slot`` vs. the rest of the league."""

    slot: str
    my_points: float
    my_rank: int  # 1 = best in the league
    n_teams: int
    league_avg: float
    best: float
    worst: float
    verdict: str  # "strength" | "average" | "weakness"


def slot_points_by_team(
    starters_by_team: Mapping[int, Sequence],
) -> dict[int, dict[str, float]]:
    """``roster_id -> {slot: total projected points}`` from each team's optimal starters.

    ``starters_by_team`` maps a roster id to that team's optimal-lineup
    :class:`~optimizer.lineup.StarterSpot` list (one per filled slot).
    """
    out: dict[int, dict[str, float]] = {}
    for rid, starters in starters_by_team.items():
        by_slot: dict[str, float] = defaultdict(float)
        for sp in starters:
            by_slot[sp.slot] += sp.player.proj_pts
        out[rid] = {slot: round(pts, 2) for slot, pts in by_slot.items()}
    return out


def _verdict(rank: int, n_teams: int) -> str:
    if n_teams <= 1:
        return "average"
    frac = (rank - 1) / (n_teams - 1)  # 0.0 = best, 1.0 = worst
    if frac <= 1 / 3:
        return "strength"
    if frac >= 2 / 3:
        return "weakness"
    return "average"


def position_strengths(
    slot_points: Mapping[int, Mapping[str, float]],
    my_roster_id: int,
    slot_order: Sequence[str],
) -> list[PositionStrength]:
    """Rank my per-slot points against every team, one :class:`PositionStrength` per slot.

    ``slot_points`` is the output of :func:`slot_points_by_team`. ``slot_order`` is the league's
    lineup slots in display order (from :func:`optimizer.lineup.lineup_slots`).
    """
    n = len(slot_points)
    rows: list[PositionStrength] = []
    for slot in slot_order:
        vals = {rid: float(sp.get(slot, 0.0)) for rid, sp in slot_points.items()}
        mine = vals.get(my_roster_id, 0.0)
        # Standard competition rank: 1 + number of teams strictly ahead of me.
        rank = 1 + sum(1 for v in vals.values() if v > mine)
        avg = sum(vals.values()) / n if n else 0.0
        rows.append(
            PositionStrength(
                slot=slot,
                my_points=round(mine, 2),
                my_rank=rank,
                n_teams=n,
                league_avg=round(avg, 2),
                best=round(max(vals.values()), 2) if vals else 0.0,
                worst=round(min(vals.values()), 2) if vals else 0.0,
                verdict=_verdict(rank, n),
            )
        )
    return rows


# --------------------------------------------------------------------------- bye-week gaps
@dataclass(frozen=True)
class ByeGap:
    """An upcoming week where a bye leaves me short at a position."""

    week: int
    pos: str
    needed: int  # dedicated starting slots for this position
    available: int  # rostered players at this position whose team is NOT on bye
    idle: tuple[str, ...]  # my players at this position on bye that week
    severity: str  # "hole" (can't field required starters) | "thin" (forced to start a backup)


def bye_week_gaps(
    my_players: Sequence,
    bye_week_of_team: Mapping[str, int],
    slots: Mapping[str, int],
    *,
    from_week: int,
    positions: Sequence[str] = ("QB", "RB", "WR", "TE", "K", "DEF"),
) -> list[ByeGap]:
    """Flag upcoming weeks where a bye forces a backup or leaves a hole.

    ``my_players`` are my rostered players (objects with ``player_id``, ``name``, ``pos``, ``team``,
    ``proj_pts`` -- e.g. :class:`~optimizer.lineup.LineupPlayer`). ``bye_week_of_team`` maps a team to
    its bye week; ``slots`` is the league's ``{slot: count}`` map. A "hole" is when fewer players than
    dedicated slots remain; "thin" is when one of my top-``needed`` starters is on bye but I can still
    field the slot from depth. FLEX cross-cover is intentionally ignored (conservative, clearer).
    """
    bye_weeks = sorted({w for w in bye_week_of_team.values() if w >= from_week})
    gaps: list[ByeGap] = []
    for week in bye_weeks:
        for pos in positions:
            needed = int(slots.get(pos, 0))
            if needed == 0:
                continue
            at_pos = [p for p in my_players if p.pos == pos]
            on_bye = [p for p in at_pos if p.team and bye_week_of_team.get(p.team) == week]
            available = len(at_pos) - len(on_bye)
            on_bye_ids = {p.player_id for p in on_bye}
            top = sorted(at_pos, key=lambda p: p.proj_pts, reverse=True)[:needed]
            top_hit = any(p.player_id in on_bye_ids for p in top)
            if available < needed:
                severity = "hole"
            elif top_hit:
                severity = "thin"
            else:
                continue
            gaps.append(
                ByeGap(
                    week=week,
                    pos=pos,
                    needed=needed,
                    available=available,
                    idle=tuple(p.name for p in on_bye),
                    severity=severity,
                )
            )
    return gaps


# --------------------------------------------------------------------------- positional needs
@dataclass(frozen=True)
class PositionalNeed:
    pos: str
    severity: str  # "high" | "medium"
    reasons: tuple[str, ...]


def positional_needs(
    strengths: Sequence[PositionStrength],
    my_players: Sequence,
    slots: Mapping[str, int],
    bye_gaps: Sequence[ByeGap],
    *,
    positions: Sequence[str] = TRADE_POSITIONS,
) -> list[PositionalNeed]:
    """Fold weaknesses, thin depth and bye holes into a per-position needs list."""
    count_by_pos = Counter(p.pos for p in my_players)
    hole_positions = {g.pos for g in bye_gaps if g.severity == "hole"}
    verdict_by_pos = {s.slot: s.verdict for s in strengths}

    needs: list[PositionalNeed] = []
    for pos in positions:
        needed = int(slots.get(pos, 0))
        if needed == 0:
            continue
        reasons: list[str] = []
        verdict = verdict_by_pos.get(pos)
        strength = next((s for s in strengths if s.slot == pos), None)
        if verdict == "weakness" and strength is not None:
            reasons.append(f"bottom-third starter strength (rank {strength.my_rank}/{strength.n_teams})")
        if count_by_pos.get(pos, 0) <= needed:
            reasons.append(f"no bench depth ({count_by_pos.get(pos, 0)} rostered for {needed} slot(s))")
        if pos in hole_positions:
            reasons.append("bye-week hole upcoming")
        if reasons:
            severity = "high" if (verdict == "weakness" or pos in hole_positions) else "medium"
            needs.append(PositionalNeed(pos=pos, severity=severity, reasons=tuple(reasons)))
    return needs


# --------------------------------------------------------------------------- trade targets
@dataclass(frozen=True)
class TradeIdea:
    roster_id: int
    team_name: str
    give_positions: tuple[str, ...]  # my surplus to offer
    get_positions: tuple[str, ...]  # their surplus I'd want
    rationale: str


def _rank_in_slot(slot_points: Mapping[int, Mapping[str, float]], slot: str) -> dict[int, int]:
    vals = {rid: float(sp.get(slot, 0.0)) for rid, sp in slot_points.items()}
    return {rid: 1 + sum(1 for v2 in vals.values() if v2 > v) for rid, v in vals.items()}


def trade_targets(
    slot_points: Mapping[int, Mapping[str, float]],
    my_roster_id: int,
    team_names: Mapping[int, str],
    strengths: Sequence[PositionStrength],
    *,
    positions: Sequence[str] = TRADE_POSITIONS,
) -> list[TradeIdea]:
    """Mutual-fit trade ideas: a team strong where I'm weak *and* weak where I'm strong.

    Reuses the per-slot point matrix; a team is "strong"/"weak" at a position by the same top-/
    bottom-third rule used for my own verdicts. Returns the best fits first (most positions matched).
    """
    n = len(slot_points)
    verdict_by_pos = {s.slot: s.verdict for s in strengths}
    my_weak = [p for p in positions if verdict_by_pos.get(p) == "weakness"]
    my_strong = [p for p in positions if verdict_by_pos.get(p) == "strength"]
    if not my_weak or not my_strong:
        return []

    ranks = {pos: _rank_in_slot(slot_points, pos) for pos in positions}
    ideas: list[TradeIdea] = []
    for rid in slot_points:
        if rid == my_roster_id:
            continue
        get = [p for p in my_weak if _verdict(ranks[p].get(rid, n), n) == "strength"]
        give = [p for p in my_strong if _verdict(ranks[p].get(rid, n), n) == "weakness"]
        if get and give:
            rationale = (
                f"strong at {'/'.join(get)} (your need), thin at {'/'.join(give)} (your surplus)"
            )
            ideas.append(
                TradeIdea(
                    roster_id=rid,
                    team_name=team_names.get(rid, f"Team {rid}"),
                    give_positions=tuple(give),
                    get_positions=tuple(get),
                    rationale=rationale,
                )
            )
    ideas.sort(key=lambda t: len(t.give_positions) + len(t.get_positions), reverse=True)
    return ideas


# --------------------------------------------------------------------------- playoff outlook
def playoff_outlook(
    my_starters: Sequence,
    sos: Mapping[str, Mapping[str, float]],
    opponents_by_week: Mapping[str, Mapping[int, str]],
    *,
    playoff_weeks: Sequence[int] = PLAYOFF_WEEKS,
) -> list[StashCandidate]:
    """My likely starters' Weeks 15-17 value, SOS-tilted in our scoring (best first).

    Reuses :func:`waivers.stash.rank_playoff_stashes` -- each starter's per-game baseline (this week's
    re-scored projection, ``proj_pts``) is multiplied by the matchup multiplier of each playoff
    opponent for the starter's position. ``min_baseline=0.0`` so K/DEF starters are kept.
    """
    candidates = [
        {
            "player_id": s.player_id,
            "name": s.name,
            "pos": s.pos,
            "team": s.team,
            "baseline": float(getattr(s, "proj_pts", 0.0) or 0.0),
        }
        for s in my_starters
    ]
    return rank_playoff_stashes(
        candidates, sos, opponents_by_week, playoff_weeks=playoff_weeks, min_baseline=0.0
    )

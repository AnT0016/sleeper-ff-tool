"""Post-draft report card: rank every team's roster in OUR scoring.

The other managers draft by market ADP; we draft by custom VOR — so scoring each team's *best legal
starting lineup* on our custom-scored projections quantifies that edge and turns it into a grade.
Pure/offline: the live draft app feeds it the raw pick feed + the VOR-scored board.

A team's grade is a percentile letter over the 12 projected starting-lineup totals; the positional
breakdown sums each team's dedicated starters per position so you can see where you won (or came up
thin) relative to the league.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from draftsim.lineup import best_lineup_points
from projections.board import PlayerRow

#: Dedicated starting positions used for the per-position strength breakdown (FLEX folds into the
#: overall lineup total, not the per-position split).
STARTER_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")


@dataclass
class TeamGrade:
    slot: int
    team: str
    starters_pts: float  # best legal starting lineup, our scoring
    rank: int  # 1 = best of the league
    grade: str  # A..F (percentile)
    is_me: bool
    by_pos: dict[str, float]  # dedicated-starter points per position


def team_picks(picks: Sequence[Mapping]) -> dict[int, list[str]]:
    """``draft_slot -> [player_id, ...]`` in pick order (a slot keeps its team across snake rounds)."""
    out: dict[int, list[str]] = defaultdict(list)
    for p in sorted(picks, key=lambda p: int(p.get("pick_no") or 0)):
        slot = int(p.get("draft_slot") or 0)
        pid = p.get("player_id")
        if slot and pid is not None:
            out[slot].append(str(pid))
    return out


def _positional(positions: Sequence[str], points: Sequence[float], slots: Mapping[str, int]):
    by_pos: dict[str, list[float]] = defaultdict(list)
    for pos, pt in zip(positions, points):
        by_pos[pos].append(pt)
    out: dict[str, float] = {}
    for pos in STARTER_POSITIONS:
        vals = sorted(by_pos.get(pos, []), reverse=True)
        out[pos] = round(sum(vals[: int(slots.get(pos, 0))]), 1)
    return out


def _letter(rank: int, n: int) -> str:
    pct = 1.0 - (rank - 1) / max(1, n - 1)  # rank 1 -> 1.0, rank n -> 0.0
    if pct >= 0.85:
        return "A"
    if pct >= 0.65:
        return "B"
    if pct >= 0.40:
        return "C"
    if pct >= 0.20:
        return "D"
    return "F"


def grade_draft(
    picks: Sequence[Mapping],
    board: Sequence[PlayerRow],
    slots: Mapping[str, int],
    *,
    teams: int,
    my_slot: int | None = None,
    slot_names: Mapping[int, str] | None = None,
) -> list[TeamGrade]:
    """Grade every team's draft by best-lineup projection in our scoring (best rank first).

    Player projections/positions come from ``board`` (our custom-scored VOR board); a drafted player
    absent from the board falls back to the pick's ``metadata`` position at 0 points (bench depth,
    won't start). Returns one :class:`TeamGrade` per team, ranked.
    """
    proj = {p.player_id: p.proj_pts for p in board}
    pos_by = {p.player_id: p.pos for p in board}
    meta_pos = {
        str(p.get("player_id")): (p.get("metadata") or {}).get("position")
        for p in picks
        if (p.get("metadata") or {}).get("position")
    }
    names = slot_names or {}

    rosters = team_picks(picks)
    computed = []
    for slot in range(1, teams + 1):
        pids = rosters.get(slot, [])
        positions = [pos_by.get(pid) or meta_pos.get(pid) or "" for pid in pids]
        points = [float(proj.get(pid, 0.0)) for pid in pids]
        computed.append(
            {
                "slot": slot,
                "starters_pts": round(best_lineup_points(positions, points, slots), 1),
                "by_pos": _positional(positions, points, slots),
            }
        )

    computed.sort(key=lambda g: g["starters_pts"], reverse=True)
    grades: list[TeamGrade] = []
    for rank, g in enumerate(computed, start=1):
        grades.append(
            TeamGrade(
                slot=g["slot"],
                team=names.get(g["slot"], f"Slot {g['slot']}"),
                starters_pts=g["starters_pts"],
                rank=rank,
                grade=_letter(rank, teams),
                is_me=(my_slot is not None and g["slot"] == my_slot),
                by_pos=g["by_pos"],
            )
        )
    return grades


def positional_ranks(grades: Sequence[TeamGrade], my_slot: int) -> dict[str, tuple[int, int]]:
    """My league rank (1 = best) per position -> ``{pos: (rank, n_teams)}``."""
    ranks: dict[str, tuple[int, int]] = {}
    n = len(grades)
    for pos in STARTER_POSITIONS:
        ordered = sorted(grades, key=lambda g: g.by_pos.get(pos, 0.0), reverse=True)
        for i, g in enumerate(ordered, start=1):
            if g.slot == my_slot:
                ranks[pos] = (i, n)
                break
    return ranks

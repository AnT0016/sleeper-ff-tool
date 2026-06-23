"""Playoff stash ranker + bye-week stash suggestions (pure, no network).

Two forward-looking views over the free-agent pool:

* :func:`rank_playoff_stashes` -- rank available players by their **Weeks 15-17** value. Each
  player's per-game baseline (this week's re-scored projection -- their talent/role level) is tilted
  by the self-computed strength-of-schedule (``waivers.sos``): for each playoff week we multiply the
  baseline by the matchup multiplier of that week's opponent for the player's position. We report
  both the flat ``raw_value`` and the SOS-adjusted ``adj_value`` plus the per-week breakdown, so the
  schedule's effect is transparent (per the confirmed design: SOS is the differentiator).
* :func:`bye_stash_suggestions` -- detect upcoming weeks where one of my starters is on bye and
  suggest the best same-position free agent to stash to cover the hole.

Everything here is pure: candidates and schedule/SOS lookups are passed in (the networked glue in
``waivers.inputs`` builds them), so it is fully unit-testable offline.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from waivers.sos import SOS_POSITIONS, multiplier

#: Our league's playoff weeks (CLAUDE.md: 6 teams, Weeks 15-16-17, one week per round).
PLAYOFF_WEEKS: tuple[int, ...] = (15, 16, 17)


@dataclass(frozen=True)
class WeekMatchup:
    week: int
    opponent: str | None
    multiplier: float
    value: float


@dataclass(frozen=True)
class StashCandidate:
    player_id: str
    name: str
    pos: str
    team: str | None
    baseline: float  # per-game baseline (this week's re-scored projection)
    raw_value: float  # baseline summed over playoff weeks the team plays (flat, no SOS)
    adj_value: float  # SOS-adjusted playoff value
    weeks: tuple[WeekMatchup, ...]

    @property
    def sos_swing(self) -> float:
        return round(self.adj_value - self.raw_value, 2)


def rank_playoff_stashes(
    candidates: Iterable[Mapping],
    sos: Mapping[str, Mapping[str, float]],
    opponents_by_week: Mapping[str, Mapping[int, str]],
    *,
    playoff_weeks: Sequence[int] = PLAYOFF_WEEKS,
    positions: frozenset[str] = SOS_POSITIONS,
    min_baseline: float = 0.1,
) -> list[StashCandidate]:
    """Rank free-agent ``candidates`` by SOS-adjusted Weeks 15-17 value, best first.

    Each candidate is a mapping with ``player_id``, ``name``, ``pos``, ``team`` and ``baseline``
    (per-game projection). ``positions`` get the SOS tilt; any other position keeps a flat 1.0
    multiplier (K/DEF stream on other logic). Candidates below ``min_baseline`` are dropped.
    """
    out: list[StashCandidate] = []
    for c in candidates:
        baseline = float(c.get("baseline") or 0.0)
        if baseline < min_baseline:
            continue
        pos, team = c.get("pos"), c.get("team")
        sched = opponents_by_week.get(team or "", {})
        weeks: list[WeekMatchup] = []
        raw = adj = 0.0
        for w in playoff_weeks:
            opp = sched.get(int(w))
            if opp is None:  # bye or unknown -> no game that week
                continue
            mult = multiplier(sos, opp, pos) if pos in positions else 1.0
            val = round(baseline * mult, 2)
            weeks.append(WeekMatchup(week=int(w), opponent=opp, multiplier=mult, value=val))
            raw += baseline
            adj += val
        out.append(
            StashCandidate(
                player_id=str(c.get("player_id")),
                name=c.get("name") or str(c.get("player_id")),
                pos=pos,
                team=team,
                baseline=round(baseline, 2),
                raw_value=round(raw, 2),
                adj_value=round(adj, 2),
                weeks=tuple(weeks),
            )
        )
    out.sort(key=lambda s: s.adj_value, reverse=True)
    return out


@dataclass(frozen=True)
class ByeStash:
    week: int
    pos: str
    idle_starters: tuple[str, ...]
    suggestions: tuple[tuple[str, float], ...]  # (free-agent name, baseline), best first

    @property
    def reason(self) -> str:
        who = ", ".join(self.idle_starters)
        return f"Week {self.week}: {who} ({self.pos}) on bye"


def bye_stash_suggestions(
    my_starters: Sequence,
    bye_week_of_team: Mapping[str, int],
    fa_candidates: Iterable[Mapping],
    *,
    from_week: int,
    top: int = 3,
) -> list[ByeStash]:
    """Flag upcoming starter byes and suggest the best same-position free agents to stash.

    ``my_starters`` are objects with ``name``, ``pos``, ``team`` (e.g. ``LineupPlayer``).
    ``bye_week_of_team`` maps a team to its bye week. Only byes in week ``from_week`` or later are
    reported. ``fa_candidates`` are mappings with ``pos``, ``name``, ``baseline``.
    """
    fa_by_pos: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for c in fa_candidates:
        fa_by_pos[c.get("pos")].append((c.get("name") or "?", float(c.get("baseline") or 0.0)))
    for lst in fa_by_pos.values():
        lst.sort(key=lambda nb: nb[1], reverse=True)

    grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
    for s in my_starters:
        bye = bye_week_of_team.get(s.team or "")
        if bye is None or bye < from_week:
            continue
        grouped[(int(bye), s.pos)].append(s.name)

    out: list[ByeStash] = []
    for (week, pos), idle in sorted(grouped.items()):
        out.append(
            ByeStash(
                week=week,
                pos=pos,
                idle_starters=tuple(idle),
                suggestions=tuple(fa_by_pos.get(pos, [])[:top]),
            )
        )
    return out

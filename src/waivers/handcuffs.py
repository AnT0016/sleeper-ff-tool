"""Handcuff / injury-replacement detector (pure, no network).

For each of my current optimal-lineup **skill** starters (QB/RB/WR/TE -- K/DEF have no meaningful
handcuff), find the next man up on that player's NFL team and flag it if it is unrostered in our
league. The depth chart comes from the Sleeper player map itself (each player carries
``depth_chart_position`` + ``depth_chart_order``, already Sleeper-keyed -- no ID join), so the backup
we name *is* an addable ``player_id``.

Priority:

* **URGENT** -- my starter currently carries a Questionable / Doubtful / Out designation, so the
  backup could be in the lineup this week (this is the "my starter is Q/D/O and the backup is
  unrostered" case from CLAUDE.md).
* **HIGH** -- my starter is healthy; the handcuff is a speculative, league-winning insurance add.

The "immediate backup" is the highest-ranked teammate at the same position *below* my starter on the
depth chart. If a rostered player sits between them we keep walking down to the best *available*
backup and record the ``gap`` (0 = literal next man up), so the alert always names a player you can
actually add.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Positions worth handcuffing (RB most of all -- the clear next-man-up). K/DEF excluded.
HANDCUFF_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE"})

#: Starter injury designations that escalate a handcuff to URGENT (per CLAUDE.md: Q/D/O).
URGENT_STATUSES: frozenset[str] = frozenset({"Questionable", "Doubtful", "Out"})


@dataclass(frozen=True)
class HandcuffAlert:
    starter_id: str
    starter_name: str
    pos: str
    team: str | None
    starter_status: str | None
    backup_id: str
    backup_name: str
    backup_dc_order: int | None
    gap: int  # rostered players between my starter and this available backup (0 = immediate)
    priority: str  # "URGENT" | "HIGH"
    starter_proj: float = 0.0

    @property
    def reason(self) -> str:
        who = "next man up" if self.gap == 0 else f"next available back ({self.gap} ahead rostered)"
        if self.priority == "URGENT":
            return (
                f"{self.starter_name} is {self.starter_status}; {self.backup_name} is the {who} "
                f"and is a free agent -- claim now"
            )
        return f"{self.backup_name} is {self.starter_name}'s {who} and is unrostered -- insurance add"


def _order(meta: Mapping) -> int | None:
    v = meta.get("depth_chart_order")
    return int(v) if v is not None else None


def _name(meta: Mapping, pid: str) -> str:
    full = meta.get("full_name")
    if full:
        return full
    nm = f"{(meta.get('first_name') or '').strip()} {(meta.get('last_name') or '').strip()}".strip()
    return nm or pid


def _depth_chart(team: str, pos: str, players_map: Mapping[str, Mapping]) -> list[tuple[str, Mapping]]:
    """Same-team, same-(depth-chart-)position players, ordered by depth_chart_order (None last)."""
    rows = [
        (pid, meta)
        for pid, meta in players_map.items()
        if meta.get("team") == team
        and (meta.get("depth_chart_position") or meta.get("position")) == pos
        and meta.get("active") is not False
    ]
    rows.sort(key=lambda pm: (_order(pm[1]) is None, _order(pm[1]) or 0))
    return rows


def find_handcuffs(
    starters: Sequence,
    players_map: Mapping[str, Mapping],
    free_agent_ids: set[str],
) -> list[HandcuffAlert]:
    """Flag unrostered next-men-up for my skill starters.

    ``starters`` are the optimal-lineup starters (objects with ``player_id``, ``name``, ``pos``,
    ``team``, ``status`` and optional ``proj_pts`` -- e.g. ``optimizer.lineup.LineupPlayer``).
    ``free_agent_ids`` is the addable pool (see :func:`waivers.league.free_agents`).
    """
    alerts: list[HandcuffAlert] = []
    for s in starters:
        if s.pos not in HANDCUFF_POSITIONS or not s.team:
            continue
        meta = players_map.get(str(s.player_id)) or {}
        dc_pos = meta.get("depth_chart_position") or s.pos
        chart = _depth_chart(s.team, dc_pos, players_map)

        s_order = _order(meta)
        # Players strictly below my starter on the chart (fall back to "after me in the list").
        below: list[tuple[str, Mapping]]
        if s_order is not None:
            below = [(pid, m) for pid, m in chart if (_order(m) or 10**6) > s_order]
        else:
            seen_me = False
            below = []
            for pid, m in chart:
                if pid == str(s.player_id):
                    seen_me = True
                    continue
                if seen_me:
                    below.append((pid, m))

        gap = 0
        backup: tuple[str, Mapping] | None = None
        for pid, m in below:
            if pid in free_agent_ids:
                backup = (pid, m)
                break
            gap += 1
        if backup is None:
            continue

        bid, bmeta = backup
        priority = "URGENT" if s.status in URGENT_STATUSES else "HIGH"
        alerts.append(
            HandcuffAlert(
                starter_id=str(s.player_id),
                starter_name=s.name,
                pos=s.pos,
                team=s.team,
                starter_status=s.status,
                backup_id=bid,
                backup_name=_name(bmeta, bid),
                backup_dc_order=_order(bmeta),
                gap=gap,
                priority=priority,
                starter_proj=float(getattr(s, "proj_pts", 0.0) or 0.0),
            )
        )

    alerts.sort(key=lambda a: (a.priority != "URGENT", -a.starter_proj))
    return alerts

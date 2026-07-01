"""Weekly K/DEF streaming guide (pure, no network).

K and DEF are *streamed*, not stashed: you play the best available one each week and churn. So the
primary decision is a one-week question — "who's the best free-agent kicker / defense for THIS week,
in our scoring, and is it worth burning a waiver claim over my current one?" — and that's what the
ranking is by.

Alongside each option we carry a **horizon**: next week, a rest-of-season per-game level, and a
Weeks 15-17 **playoff** outlook, so you can tell a one-week plug from a defense worth grabbing early
for the playoff run (claims are scarce here — reverse priority, not FAAB). The playoff column is
matchup-real for defenses via the DEF strength-of-schedule (``waivers.sos``); kickers carry no SOS on
purpose (a kicker's scoring is driven by its *own* offense, not the opposing defense), so K uses
projections only. The glue in ``waivers.inputs`` computes each horizon value; this module just ranks
and frames them, so it stays fully unit-testable offline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: Positions this guide streams. Skill positions are handled by the spend / stash views instead.
STREAM_POSITIONS: tuple[str, ...] = ("K", "DEF")

#: A streamer must beat my current starter by at least this (this-week) margin to earn a "stream"
#: verdict — a fractional edge at K/DEF isn't worth spending a scarce waiver claim on.
DEFAULT_MIN_GAIN = 1.5


@dataclass(frozen=True)
class StreamOption:
    player_id: str
    name: str
    pos: str
    team: str | None
    this_week: float  # this week's re-scored projection (the decision variable)
    next_week: float  # next week's re-scored projection (hold-vs-churn signal)
    ros_pg: float  # rest-of-season per-game level (season projection ÷ games)
    playoff: float  # Weeks 15-17 outlook (DEF: SOS-tilted; K: flat per-game)
    gain: float  # this_week minus my current starter's this_week


@dataclass(frozen=True)
class StreamAdvice:
    pos: str
    current_name: str | None
    current_this_week: float
    options: tuple[StreamOption, ...]  # best first, by this-week projection
    verdict: str  # "stream" | "hold"

    @property
    def best_gain(self) -> float:
        return self.options[0].gain if self.options else 0.0


def rank_streamers(
    candidates: Iterable[Mapping],
    current_by_pos: Mapping[str, Mapping],
    *,
    positions: Sequence[str] = STREAM_POSITIONS,
    top: int = 5,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> list[StreamAdvice]:
    """Rank free-agent streamers per position by this week's value, best first.

    ``candidates`` are free-agent mappings with ``player_id``/``name``/``pos``/``team`` and the
    precomputed horizon values ``this_week``/``next_week``/``ros_pg``/``playoff``. ``current_by_pos``
    maps a position to my currently-rostered starter there (``{"name", "this_week"}``) — the player a
    pickup would replace. Returns one :class:`StreamAdvice` per position (in ``positions`` order) that
    has any candidate or a current starter.
    """
    by_pos: dict[str, list[Mapping]] = {p: [] for p in positions}
    for c in candidates:
        if c.get("pos") in by_pos:
            by_pos[c["pos"]].append(c)

    out: list[StreamAdvice] = []
    for pos in positions:
        cur = current_by_pos.get(pos) or {}
        cur_tw = round(float(cur.get("this_week") or 0.0), 2)
        cands = sorted(by_pos[pos], key=lambda c: float(c.get("this_week") or 0.0), reverse=True)
        if not cands and not cur:
            continue
        options = tuple(
            StreamOption(
                player_id=str(c.get("player_id")),
                name=c.get("name") or str(c.get("player_id")),
                pos=pos,
                team=c.get("team"),
                this_week=round(float(c.get("this_week") or 0.0), 2),
                next_week=round(float(c.get("next_week") or 0.0), 2),
                ros_pg=round(float(c.get("ros_pg") or 0.0), 2),
                playoff=round(float(c.get("playoff") or 0.0), 2),
                gain=round(float(c.get("this_week") or 0.0) - cur_tw, 2),
            )
            for c in cands[:top]
        )
        best_gain = options[0].gain if options else 0.0
        out.append(
            StreamAdvice(
                pos=pos,
                current_name=cur.get("name"),
                current_this_week=cur_tw,
                options=options,
                verdict="stream" if best_gain >= min_gain else "hold",
            )
        )
    return out

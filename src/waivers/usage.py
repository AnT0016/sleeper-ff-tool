"""Usage / opportunity enrichment for add candidates (pure, no network).

Supporting evidence attached to waiver targets -- never a ranking of its own. For a set of Sleeper
``player_id``s over a trailing window of recent weeks we surface:

* **snap share** -- mean ``offense_pct`` (and a simple first->last trend) from nflverse snap counts
  (joined to Sleeper via the ``pfr_id`` crosswalk);
* **target / rush share** -- ``rec_attempt`` / ``rec_attempt_team`` and the rush equivalent from
  nflverse ``ff_opportunity`` (joined via ``gsis_id``);
* **expected fantasy points** -- ``total_fantasy_points_exp`` from the same source. NOTE: this is the
  ffopportunity model's *generic* scoring, not our league settings -- carried as a directional usage
  signal only.

The functions take the nflverse frames + the id maps so they are fully unit-testable offline; the
networked glue (``waivers.inputs``) builds those. Red-zone usage is intentionally deferred (no cheap
field in ff_opportunity; pulling full play-by-play is out of scope for v1).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class UsageSignal:
    player_id: str
    snap_pct: float | None  # mean offense snap share over the window (0-1)
    snap_trend: float | None  # last-week minus first-week snap share (rising role > 0)
    target_share: float | None
    rush_share: float | None
    exp_points: float | None  # generic-model expected fantasy points/game (NOT our scoring)
    weeks: int

    def summary(self) -> str:
        bits: list[str] = []
        if self.snap_pct is not None:
            arrow = ""
            if self.snap_trend is not None and abs(self.snap_trend) >= 0.1:
                arrow = " ↑" if self.snap_trend > 0 else " ↓"
            bits.append(f"snaps {self.snap_pct * 100:.0f}%{arrow}")
        if self.target_share is not None:
            bits.append(f"tgt {self.target_share * 100:.0f}%")
        if self.rush_share is not None:
            bits.append(f"rush {self.rush_share * 100:.0f}%")
        if self.exp_points is not None:
            bits.append(f"xFP {self.exp_points:.1f}/gm")
        return ", ".join(bits)


def _num(row: Mapping, col: str) -> float:
    v = row.get(col)
    return float(v) if v is not None else 0.0


def usage_signals(
    player_ids: Iterable[str],
    *,
    season: int,
    week: int,
    lookback: int = 4,
    snaps: pl.DataFrame | None = None,
    opportunity: pl.DataFrame | None = None,
    pfr_to_sleeper: Mapping[str, str] | None = None,
    gsis_to_sleeper: Mapping[str, str] | None = None,
) -> dict[str, UsageSignal]:
    """``player_id -> UsageSignal`` over the weeks in ``[week - lookback, week)`` (recent form).

    Any missing frame / id map is simply skipped, so partial data still yields partial signals.
    """
    want = {str(p) for p in player_ids}
    window = set(range(max(week - lookback, 1), week))  # weeks strictly before the target week

    snap_acc: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if snaps is not None and pfr_to_sleeper:
        for r in snaps.iter_rows(named=True):
            if r.get("season") != season or r.get("game_type") != "REG":
                continue
            w = r.get("week")
            if w is None or int(w) not in window:
                continue
            sid = pfr_to_sleeper.get(r.get("pfr_player_id"))
            if sid not in want:
                continue
            pct = r.get("offense_pct")
            if pct is not None:
                snap_acc[sid].append((int(w), float(pct)))

    opp_acc: dict[str, dict] = defaultdict(
        lambda: {"rec": 0.0, "rec_t": 0.0, "rush": 0.0, "rush_t": 0.0, "exp": []}
    )
    if opportunity is not None and gsis_to_sleeper:
        for r in opportunity.iter_rows(named=True):
            if r.get("season") != season:
                continue
            w = r.get("week")
            if w is None or int(w) not in window:
                continue
            sid = gsis_to_sleeper.get(r.get("player_id"))
            if sid not in want:
                continue
            a = opp_acc[sid]
            a["rec"] += _num(r, "rec_attempt")
            a["rec_t"] += _num(r, "rec_attempt_team")
            a["rush"] += _num(r, "rush_attempt")
            a["rush_t"] += _num(r, "rush_attempt_team")
            e = r.get("total_fantasy_points_exp")
            if e is not None:
                a["exp"].append(float(e))

    out: dict[str, UsageSignal] = {}
    for sid in set(snap_acc) | set(opp_acc):
        sn = sorted(snap_acc.get(sid, []))
        snap_pct = round(sum(p for _, p in sn) / len(sn), 3) if sn else None
        snap_trend = round(sn[-1][1] - sn[0][1], 3) if len(sn) >= 2 else None
        a = opp_acc.get(sid)
        tshare = round(a["rec"] / a["rec_t"], 3) if a and a["rec_t"] else None
        rshare = round(a["rush"] / a["rush_t"], 3) if a and a["rush_t"] else None
        exp = round(sum(a["exp"]) / len(a["exp"]), 2) if a and a["exp"] else None
        weeks = max(len(sn), len(a["exp"]) if a else 0)
        out[sid] = UsageSignal(
            player_id=sid,
            snap_pct=snap_pct,
            snap_trend=snap_trend,
            target_share=tshare,
            rush_share=rshare,
            exp_points=exp,
            weeks=weeks,
        )
    return out

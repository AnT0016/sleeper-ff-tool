"""Custom-scored draft board built from Sleeper's season projections.

Every projection row is re-scored through ``scoring.engine.points`` with this league's exact
``scoring_settings`` (pulled live from the API -- never hand-coded), so the board ranks players in
*our* scoring, not Sleeper's generic presets. The ADP carried alongside is a market signal only
(used by the draft tracker to judge whether a target survives to a given pick); it never influences
our value ranking.

Value-over-replacement and tiers are layered on later by ``draft.vor`` -- this module just produces
the ranked, custom-scored projections.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from scoring.engine import points
from sleeper import client

#: Positions our league drafts (skill + K + DEF). FB/P/CB/etc. projection rows are dropped.
DRAFTABLE: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")

#: Sleeper uses 999 (and occasionally 0/None) to mean "effectively undrafted".
_UNDRAFTED_ADP = 999.0


@dataclass
class PlayerRow:
    """One draftable player with our custom-scored projection. ``vor``/``tier`` are filled by
    ``draft.vor`` once the replacement baselines are known."""

    player_id: str  # Sleeper player_id; the team abbreviation (e.g. "PHI") for DEF.
    name: str
    pos: str
    team: str | None
    proj_pts: float
    adp: float  # market ADP (adp_half_ppr); +inf when effectively undrafted.
    vor: float = 0.0
    tier: int = 0


def _name(player: Mapping, player_id: str) -> str:
    full = player.get("full_name")
    if full:
        return full
    nm = f"{(player.get('first_name') or '').strip()} {(player.get('last_name') or '').strip()}"
    return nm.strip() or str(player_id)


def _adp(stats: Mapping, adp_key: str) -> float:
    val = stats.get(adp_key)
    if val in (None, 0) or float(val) >= _UNDRAFTED_ADP:
        return float("inf")
    return float(val)


def build_board(
    season: int,
    scoring: Mapping[str, float],
    *,
    positions: Iterable[str] = DRAFTABLE,
    adp_key: str = "adp_half_ppr",
    fetch: Callable[..., list[dict]] | None = None,
    source: str | None = None,
) -> list[PlayerRow]:
    """Build the custom-scored projection board, ranked by our league scoring (best first).

    ``scoring`` is this league's live ``scoring_settings`` dict. ``fetch`` defaults to
    ``client.get_season_projections`` and is injectable for offline tests. Duplicate player rows
    (rare) are collapsed to the highest-scoring one.

    ``source`` selects the projection source **per position** (Phase 9, #34): ``None`` (the default)
    consults the recorded swap gate — Sleeper everywhere until a model clears the live bar — while an
    explicit ``"sleeper"`` / ``"model"`` forces it. The simulators call ``build_board(season, scoring)``
    unchanged and stay on Sleeper today; when a position's gate flips, the seam composes the model board
    for that position. The Sleeper path below is untouched, and no lake module is imported unless the
    model source is actually selected (:mod:`projections.source`).
    """
    positions = tuple(positions)
    # Lazy: projections.source imports this module, so importing it at load would cycle. It is lake-free,
    # and its resolution reads only the (cached) gate file — the Sleeper default never touches the lake.
    from projections.source import MODEL, compose_season_board, resolve_positions_source

    srcmap = resolve_positions_source(positions, source)
    if any(s == MODEL for s in srcmap.values()):
        return compose_season_board(
            season, scoring, positions=positions, srcmap=srcmap, adp_key=adp_key, fetch=fetch,
            sleeper_builder=_sleeper_board,
        )
    return _sleeper_board(season, scoring, positions=positions, adp_key=adp_key, fetch=fetch)


def _sleeper_board(
    season: int,
    scoring: Mapping[str, float],
    *,
    positions: tuple[str, ...],
    adp_key: str = "adp_half_ppr",
    fetch: Callable[..., list[dict]] | None = None,
) -> list[PlayerRow]:
    """The Sleeper season board: fetch, re-score in our settings, collapse duplicates, rank best-first.

    Extracted verbatim from the old ``build_board`` body so :mod:`projections.source` composes a mixed
    board by calling it for the Sleeper positions — one scoring loop, never re-implemented.
    """
    get = fetch or client.get_season_projections
    by_id: dict[str, PlayerRow] = {}
    for r in get(season, positions=positions):
        player = r.get("player") or {}
        pos = player.get("position")
        if pos not in positions:
            continue
        stats = r.get("stats") or {}
        pid = str(r.get("player_id"))
        row = PlayerRow(
            player_id=pid,
            name=_name(player, pid),
            pos=pos,
            team=player.get("team"),
            proj_pts=round(points(stats, scoring), 2),
            adp=_adp(stats, adp_key),
        )
        prev = by_id.get(pid)
        if prev is None or row.proj_pts > prev.proj_pts:
            by_id[pid] = row
    board = sorted(by_id.values(), key=lambda p: p.proj_pts, reverse=True)
    return board

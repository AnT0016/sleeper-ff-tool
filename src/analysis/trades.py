"""Player-for-player trade evaluator (pure, no network).

The Phase-5 :func:`analysis.team.trade_targets` only says *which positions* fit a partner (strong where
I'm weak, weak where I'm strong). This turns that into concrete, gradeable offers: for every 1-for-1
skill-player swap with every other team, does it improve **both** teams' best starting lineup? A player
is valued by his *marginal* effect on the optimal lineup — give up a blocked bench body, receive a
starter — so the win-win swaps (each side's optimal lineup gains) fall straight out.

Value uses the fast greedy single-FLEX lineup total (``draftsim.lineup.best_lineup_points``, provably
optimal for one FLEX), so the full pairwise scan across the league is milliseconds — no LP in the loop.
Projections are the season-long custom-scored ones (roster quality), and K/DEF are excluded (streamed,
not traded). Rosters are passed in, so the module is fully unit-testable offline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from draftsim.lineup import best_lineup_points

#: Skill positions worth trading around (K/DEF are streamed weekly, never traded).
TRADE_POSITIONS: frozenset[str] = frozenset({"QB", "RB", "WR", "TE"})


@dataclass(frozen=True)
class TradeOffer:
    partner_id: int
    partner: str
    give_id: str
    give_name: str
    give_pos: str
    get_id: str
    get_name: str
    get_pos: str
    my_gain: float  # change in MY optimal starting-lineup value (in the caller's projection basis
    #                 — the snapshot passes rest-of-season points, not full-season totals)
    their_gain: float  # change in the PARTNER's optimal lineup value

    @property
    def combined(self) -> float:
        return round(self.my_gain + self.their_gain, 2)


def lineup_value(players: Sequence, slots: Mapping[str, int]) -> float:
    """Best legal starting-lineup total (season points) for a roster of ``LineupPlayer``-likes."""
    return best_lineup_points([p.pos for p in players], [p.proj_pts for p in players], slots)


def find_trades(
    my_players: Sequence,
    teams: Mapping[int, Sequence],
    slots: Mapping[str, int],
    team_names: Mapping[int, str],
    *,
    positions: frozenset[str] = TRADE_POSITIONS,
    min_gain: float = 1.0,
    top: int = 12,
) -> list[TradeOffer]:
    """Rank mutually-beneficial 1-for-1 skill swaps with every other team, best (for me) first.

    ``my_players`` / each ``teams[rid]`` are ``LineupPlayer``-likes (``player_id``/``name``/``pos``/
    ``proj_pts``) carrying season-long custom-scored projections. A swap qualifies only if **both**
    sides' optimal lineup gains at least ``min_gain`` season points — so the partner isn't made worse
    (their incentive is shown). Sorted by *my* gain first, so the biggest upgrades I can chase lead;
    the partner's (often smaller) gain is reported alongside so I can judge how sellable the deal is.
    """
    base_my = lineup_value(my_players, slots)
    offers: list[TradeOffer] = []
    for rid, their in teams.items():
        base_their = lineup_value(their, slots)
        for a in my_players:
            if a.pos not in positions:
                continue
            my_without = [p for p in my_players if p.player_id != a.player_id]
            for b in their:
                if b.pos not in positions:
                    continue
                my_gain = round(lineup_value(my_without + [b], slots) - base_my, 2)
                if my_gain < min_gain:
                    continue
                their_without = [p for p in their if p.player_id != b.player_id]
                their_gain = round(lineup_value(their_without + [a], slots) - base_their, 2)
                if their_gain < min_gain:
                    continue
                offers.append(
                    TradeOffer(
                        partner_id=rid,
                        partner=team_names.get(rid, f"Team {rid}"),
                        give_id=a.player_id, give_name=a.name, give_pos=a.pos,
                        get_id=b.player_id, get_name=b.name, get_pos=b.pos,
                        my_gain=my_gain, their_gain=their_gain,
                    )
                )
    offers.sort(key=lambda o: (o.my_gain, o.their_gain), reverse=True)
    return offers[:top]

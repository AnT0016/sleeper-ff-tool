"""ADP-following opponent bots — the 11 other managers.

Each simulation, every player's market ADP is perturbed by Gaussian noise and the bot on the clock
takes the lowest-ADP available player that passes a few legality caps (so a noisy draw can't make a
bot roster three kickers or a kicker in round 1). Two deliberate modelling choices:

* Bots use **market half-PPR ADP** (Sleeper's ``adp_half_ppr``), *not* our league's custom
  4-pt-passing-TD scoring. That mismatch — the field valuing players by generic rankings while we
  re-score in our settings — is precisely the inefficiency this whole project exploits.
* It is a *consensus* model: all bots share one noisy ADP ordering per sim (the market's read of the
  player), differing only in which positions their roster still has room for. This is what lets the
  simulator estimate the probability a given target survives to my next pick.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Std-dev (in overall picks) of the Gaussian noise added to each player's ADP per simulation.
#: ~8 picks ≈ how far real drafts wander from consensus ADP for a given player.
ADP_NOISE = 8.0

#: Most a bot will roster at each position (caps runaway noise; bench depth still allowed at RB/WR).
BOT_MAX_PER_POS: dict[str, int] = {"QB": 2, "RB": 6, "WR": 7, "TE": 2, "K": 1, "DEF": 1}

#: Positions a bot won't take until late, regardless of a freak-low ADP draw.
LATE_ONLY: tuple[str, ...] = ("K", "DEF")

#: K/DEF only become legal once the draft is this fraction of the way through its rounds.
LATE_ROUND_FRACTION = 0.78


def bot_allows(pos: str, counts: Mapping[str, int], rnd: int, rounds: int) -> bool:
    """May a bot whose current roster is ``counts`` draft ``pos`` in round ``rnd`` (of ``rounds``)?"""
    if counts.get(pos, 0) >= BOT_MAX_PER_POS.get(pos, 99):
        return False
    if pos in LATE_ONLY and rnd < int(rounds * LATE_ROUND_FRACTION):
        return False
    return True

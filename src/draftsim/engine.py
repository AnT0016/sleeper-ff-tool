"""The Monte Carlo draft engine: build the player pool, run many full snake drafts, aggregate.

One simulation:

1. (done once, shared across strategies for a fair *common-random-numbers* comparison) draw every
   player's season outcome and a per-sim noisy market ADP ordering;
2. run the snake draft — bots take the lowest available noisy-ADP player they have room for; I take
   the best available *VOR* player my strategy wants (with the end-of-draft mandatory guard);
3. score every team's best legal lineup on the **sampled** outcomes, recording my season points, my
   finish rank among the league, and which targets were still on the board at each of my picks.

Aggregated over ``n_sims`` drafts this yields, per strategy, a *distribution* of my season outcomes
(not a point estimate) plus target-survival probabilities. Pure/offline given a board with VOR set —
no network here (see :mod:`draftsim.inputs` for the live glue).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from draft.roster import RosterConfig
from draft.snake import my_pick_numbers
from projections.board import PlayerRow

from . import distributions as dist
from .bots import ADP_NOISE, bot_allows
from .lineup import best_lineup_points
from .strategy import CORE, STRATEGIES, forced_positions, position_has_room

_REPORT_POS = ("QB", "RB", "WR", "TE", "K", "DEF")


@dataclass
class SimPool:
    """Immutable, array-shaped view of the draftable players for fast simulation."""

    ids: list[str]
    names: list[str]
    pos: list[str]
    team: list[str | None]
    mean: np.ndarray  # custom-scored season projection (lognormal mean)
    adp: np.ndarray  # market ADP; effectively-undrafted -> large sentinel
    vor: np.ndarray
    cv: np.ndarray
    p_setback: np.ndarray
    severity: np.ndarray
    vor_order: np.ndarray  # player indices, best VOR first (fixed across sims)

    @property
    def n(self) -> int:
        return len(self.ids)


@dataclass
class StrategyResult:
    """Aggregated outcome of running one strategy over all sims."""

    name: str
    my_points: np.ndarray  # (n_sims,) my best-lineup season points
    my_rank: np.ndarray  # (n_sims,) my finish among `teams` (1 = best)
    survival: np.ndarray  # (n_my_picks, n_players) P(player available at my k-th pick)
    draft_counts: np.ndarray  # (n_players,) times I drafted each player
    builds: Counter  # composition signature -> count
    my_picks: list[int] = field(default_factory=list)


_UNDRAFTED_ADP = 1.0e9


def build_pool(board: Sequence[PlayerRow], *, pool_size: int = 300) -> SimPool:
    """Build a :class:`SimPool` from a VOR-scored board.

    Keeps every player with a real market ADP, every K/DEF, and the top ``pool_size`` by VOR — the
    only players a 12-team draft realistically reaches. (Anyone outside is effectively undraftable.)
    """
    order_by_vor = sorted(range(len(board)), key=lambda i: board[i].vor, reverse=True)
    top_vor = set(order_by_vor[:pool_size])
    sel = [
        i
        for i, r in enumerate(board)
        if r.adp != float("inf") or r.pos in ("K", "DEF") or i in top_vor
    ]
    rows = [board[i] for i in sel]

    pos = [r.pos for r in rows]
    mean = np.array([r.proj_pts for r in rows], dtype=float)
    adp = np.array([_UNDRAFTED_ADP if r.adp == float("inf") else r.adp for r in rows], dtype=float)
    vor = np.array([r.vor for r in rows], dtype=float)
    cv = np.array([dist.POSITION_CV.get(p, dist.DEFAULT_CV) for p in pos], dtype=float)
    p_set = np.array([dist.INJURY_RISK.get(p, dist.DEFAULT_RISK)[0] for p in pos], dtype=float)
    sev = np.array([dist.INJURY_RISK.get(p, dist.DEFAULT_RISK)[1] for p in pos], dtype=float)
    return SimPool(
        ids=[r.player_id for r in rows],
        names=[r.name for r in rows],
        pos=pos,
        team=[r.team for r in rows],
        mean=mean,
        adp=adp,
        vor=vor,
        cv=cv,
        p_setback=p_set,
        severity=sev,
        vor_order=np.argsort(-vor, kind="stable"),
    )


def sample_outcomes(rng: np.random.Generator, pool: SimPool, n_sims: int) -> np.ndarray:
    """``(n_sims, n_players)`` sampled season points = lognormal outcome × injury availability."""
    pts = dist.sample_season_points(rng, pool.mean, pool.cv, n_sims)
    avail, _ = dist.sample_availability(rng, pool.p_setback, pool.severity, n_sims)
    return pts * avail


def _composition(roster: list[int], pool: SimPool) -> str:
    c = Counter(pool.pos[i] for i in roster)
    return "/".join(f"{p}{c[p]}" for p in _REPORT_POS if c[p])


def _draft_once(
    pool: SimPool,
    adp_order: np.ndarray,
    my_slot: int,
    cfg: RosterConfig,
    strategy_fn,
    survival: np.ndarray,
    my_pick_index: dict[int, int],
) -> list[list[int]]:
    """Run one full snake draft; return each team's roster (list of pool indices).

    ``adp_order`` is this sim's player indices sorted by noisy market ADP (shared by all bots).
    ``survival[k]`` is incremented with the availability mask at my k-th pick (recorded *before* I
    pick). ``my_pick_index`` maps an overall pick number to my 0-based pick index.
    """
    n = pool.n
    teams, rounds = cfg.teams, cfg.rounds
    my_team = my_slot - 1
    taken = np.zeros(n, dtype=bool)
    rosters: list[list[int]] = [[] for _ in range(teams)]
    counts: list[defaultdict[str, int]] = [defaultdict(int) for _ in range(teams)]
    n_my_picks = len(my_pick_index)

    bot_ptr = 0  # front of the (taken-consumed) noisy-ADP order, shared by all bots
    my_ptr = 0  # front of the (taken-consumed) VOR order, mine

    for pick_no in range(1, teams * rounds + 1):
        rnd = (pick_no - 1) // teams + 1
        pos_in_round = (pick_no - 1) % teams
        slot = pos_in_round + 1 if rnd % 2 == 1 else teams - pos_in_round
        team = slot - 1

        if team == my_team:
            k = my_pick_index[pick_no]
            survival[k] += ~taken
            chosen, my_ptr = _my_pick(pool, taken, counts[team], cfg, strategy_fn, rnd,
                                      picks_left=n_my_picks - k, my_ptr=my_ptr)
        else:
            chosen, bot_ptr = _bot_pick(pool, adp_order, taken, counts[team], rnd, rounds,
                                        bot_ptr=bot_ptr)

        taken[chosen] = True
        rosters[team].append(chosen)
        counts[team][pool.pos[chosen]] += 1

    return rosters


def _bot_pick(pool, adp_order, taken, counts, rnd, rounds, *, bot_ptr):
    n = pool.n
    while bot_ptr < n and taken[adp_order[bot_ptr]]:
        bot_ptr += 1
    j = bot_ptr
    while j < n:
        idx = adp_order[j]
        if not taken[idx] and bot_allows(pool.pos[idx], counts, rnd, rounds):
            return int(idx), bot_ptr
        j += 1
    # nothing legal (caps exhausted the legal pool) -> take the best available regardless
    j = bot_ptr
    while j < n:
        idx = adp_order[j]
        if not taken[idx]:
            return int(idx), bot_ptr
        j += 1
    return int(adp_order[bot_ptr]), bot_ptr  # unreachable while pool > total picks


def _my_pick(pool, taken, counts, cfg, strategy_fn, rnd, *, picks_left, my_ptr):
    n = pool.n
    order = pool.vor_order
    while my_ptr < n and taken[order[my_ptr]]:
        my_ptr += 1

    forced = forced_positions(counts, cfg, picks_left)
    if forced is not None:
        allowed = forced
    else:
        allowed = {p for p in strategy_fn(rnd, counts, cfg) if position_has_room(p, counts, cfg)}

    chosen = _best_in(order, taken, pool.pos, my_ptr, allowed)
    if chosen < 0:  # nothing in the preferred set -> any core position with room
        relaxed = {p for p in CORE if position_has_room(p, counts, cfg)}
        chosen = _best_in(order, taken, pool.pos, my_ptr, relaxed)
    if chosen < 0:  # absolute fallback -> best available player, full stop
        chosen = _best_in(order, taken, pool.pos, my_ptr, None)
    return chosen, my_ptr


def _best_in(order, taken, pos, start, allowed) -> int:
    """First (best-VOR) untaken player index from ``start`` whose pos is in ``allowed`` (or any)."""
    j = start
    n = len(order)
    while j < n:
        idx = order[j]
        if not taken[idx] and (allowed is None or pos[idx] in allowed):
            return int(idx)
        j += 1
    return -1


def run_strategy(
    pool: SimPool,
    sampled: np.ndarray,
    adp_order: np.ndarray,
    my_slot: int,
    cfg: RosterConfig,
    name: str,
) -> StrategyResult:
    """Run ``name`` over every sim against the shared ``sampled`` outcomes and ``adp_order``."""
    n_sims = sampled.shape[0]
    strategy_fn = STRATEGIES[name]
    my_picks = my_pick_numbers(my_slot, cfg.teams, cfg.rounds)
    my_pick_index = {pno: k for k, pno in enumerate(my_picks)}
    my_team = my_slot - 1

    survival = np.zeros((len(my_picks), pool.n), dtype=float)
    my_points = np.empty(n_sims, dtype=float)
    my_rank = np.empty(n_sims, dtype=int)
    draft_counts = np.zeros(pool.n, dtype=int)
    builds: Counter = Counter()

    for s in range(n_sims):
        rosters = _draft_once(
            pool, adp_order[s], my_slot, cfg, strategy_fn, survival, my_pick_index
        )
        pts_row = sampled[s]
        totals = [
            best_lineup_points([pool.pos[i] for i in r], pts_row[r], cfg.slots) for r in rosters
        ]
        mine = totals[my_team]
        my_points[s] = mine
        my_rank[s] = 1 + sum(1 for t in (totals[:my_team] + totals[my_team + 1:]) if t > mine)
        for i in rosters[my_team]:
            draft_counts[i] += 1
        builds[_composition(rosters[my_team], pool)] += 1

    survival /= n_sims
    return StrategyResult(
        name=name,
        my_points=my_points,
        my_rank=my_rank,
        survival=survival,
        draft_counts=draft_counts,
        builds=builds,
        my_picks=my_picks,
    )


@dataclass
class SimOutput:
    pool: SimPool
    cfg: RosterConfig
    my_slot: int
    n_sims: int
    seed: int
    results: dict[str, StrategyResult]


def simulate(
    board: Sequence[PlayerRow],
    cfg: RosterConfig,
    my_slot: int,
    *,
    n_sims: int = 2000,
    strategies: Sequence[str] | None = None,
    seed: int = 0,
    pool_size: int = 300,
) -> SimOutput:
    """Run the full Monte Carlo across ``strategies`` (default: all) and return aggregated results.

    All strategies share the same sampled outcomes and noisy-ADP draws (common random numbers), so
    differences between builds reflect the strategy, not sampling noise.
    """
    strategies = list(strategies or STRATEGIES.keys())
    pool = build_pool(board, pool_size=pool_size)
    rng = np.random.default_rng(seed)
    sampled = sample_outcomes(rng, pool, n_sims)
    noisy = pool.adp[None, :] + rng.normal(0.0, ADP_NOISE, size=(n_sims, pool.n))
    adp_order = np.argsort(noisy, axis=1, kind="stable")

    results = {
        name: run_strategy(pool, sampled, adp_order, my_slot, cfg, name) for name in strategies
    }
    return SimOutput(
        pool=pool, cfg=cfg, my_slot=my_slot, n_sims=n_sims, seed=seed, results=results
    )

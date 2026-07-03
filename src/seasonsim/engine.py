"""The season Monte Carlo: draw a full season of weekly outcomes, run the schedule + playoffs, tally.

One ``simulate_season`` call, per sim:

1. (drawn once, shared across skill regimes for a fair common-random-numbers comparison) every
   rostered player's weekly points, injury out-weeks, and a raw start/sit noise field;
2. each team's weekly lineup score — starters chosen from projected means (± that team's start/sit
   noise, injured players benched), scored on the sampled outcomes;
3. regular-season records from the real (or round-robin) schedule → seed the top-6 → play the locked
   Weeks 15-17 bracket to a champion.

Aggregated over ``n_sims`` seasons this gives, per team, P(championship) / P(playoffs) / expected wins,
and for my team the full seed / wins / points distribution. Two regimes are returned: *equal_skill*
(all managers equally sloppy) and *my_edge* (I set clean lineups while opponents don't) — their gap is
the value of the weekly optimizer. Pure/offline given a :class:`SeasonPool` and a schedule.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import distributions as dist
from .lineup import lineup_values
from .schedule import playoff_champion, resolve_records, seed_order

#: Default start/sit noise: a manager's selection score carries N(0, NOISE × player's weekly mean),
#: so sloppier managers sometimes bench the wrong guy. 0 = always plays the projection-optimal lineup.
DEFAULT_OPP_NOISE = 0.5
DEFAULT_MY_NOISE = 0.0
_OUT_SENTINEL = -1.0e9  # a known-injured player is never chosen to start (benched for a healthy body)


@dataclass
class SeasonPool:
    """Array-shaped view of the twelve rosters for fast simulation."""

    rosters: list[list[int]]  # team index -> player-array indices (disjoint partition)
    pos: list[str]
    mean: np.ndarray  # custom-scored season projection per player
    cv: np.ndarray
    p_setback: np.ndarray
    severity: np.ndarray
    names: list[str]
    team_names: list[str]  # display name per team index
    slots: dict[str, int]
    n_teams: int


@dataclass
class RegimeResult:
    name: str
    champ_prob: np.ndarray  # (n_teams,)
    playoff_prob: np.ndarray  # (n_teams,)
    exp_wins: np.ndarray  # (n_teams,)
    exp_points: np.ndarray  # (n_teams,)
    my_wins: np.ndarray  # (n_sims,)
    my_rank: np.ndarray  # (n_sims,) 1 = best
    my_points: np.ndarray  # (n_sims,) season points-for
    my_is_champ: np.ndarray  # (n_sims,) bool


@dataclass
class SeasonSimOutput:
    pool: SeasonPool
    my_team: int
    season: int
    n_sims: int
    seed: int
    total_weeks: int
    regular_weeks: list[int]
    playoff_weeks: list[int]
    n_playoff_teams: int
    regimes: dict[str, RegimeResult]
    opp_noise: float
    conditioned_weeks: list[int] = field(default_factory=list)  # weeks pinned to real scores


def pool_arrays(pool: SeasonPool):
    """Per-player season CV / game CV / setback / severity from the shared per-position knobs."""
    cv = np.array([dist.POSITION_CV.get(p, dist.DEFAULT_CV) for p in pool.pos], dtype=float)
    gcv = np.array([dist.GAME_CV.get(p, dist.DEFAULT_GAME_CV) for p in pool.pos], dtype=float)
    p_set = np.array([dist.INJURY_RISK.get(p, dist.DEFAULT_RISK)[0] for p in pool.pos], dtype=float)
    sev = np.array([dist.INJURY_RISK.get(p, dist.DEFAULT_RISK)[1] for p in pool.pos], dtype=float)
    return cv, gcv, p_set, sev


def team_week_scores(
    pool: SeasonPool,
    realized: np.ndarray,  # (n_sims, P, W) sampled points, already zeroed on out-weeks
    out: np.ndarray,  # (n_sims, P, W) bool
    mean_week: np.ndarray,  # (P,) per-player weekly mean
    raw_noise: np.ndarray,  # (n_sims, P, W) standard-normal start/sit noise
    team_noise: Sequence[float],  # per-team noise multiplier
) -> np.ndarray:
    """``(n_sims, n_teams, W)`` weekly lineup scores under the given per-team start/sit noise."""
    n_sims, _, weeks = realized.shape
    scores = np.zeros((n_sims, pool.n_teams, weeks), dtype=float)
    noise_scale = raw_noise * mean_week[None, :, None]  # N(0, mean_week) field, scaled per team below
    for t, cols in enumerate(pool.rosters):
        if not cols:
            continue
        positions = [pool.pos[i] for i in cols]
        val = realized[:, cols, :]
        base = mean_week[None, cols, None] + team_noise[t] * noise_scale[:, cols, :]
        select = np.where(out[:, cols, :], _OUT_SENTINEL, base)
        for w in range(weeks):
            scores[:, t, w] = lineup_values(positions, select[:, :, w], val[:, :, w], pool.slots)
    return scores


def _rank_and_champ(
    scores: np.ndarray,
    schedule: Mapping[int, Sequence[tuple[int, int]]],
    playoff_cols: Sequence[int],
    n_playoff_teams: int,
    my_team: int,
) -> RegimeResult:
    wins, points = resolve_records(schedule, scores)  # (n_sims, n_teams)
    n_sims, n_teams = wins.shape

    # Full standings rank per team: 1 + number of strictly-better teams (wins, then points-for).
    rank = np.ones((n_sims, n_teams), dtype=int)
    for t in range(n_teams):
        better = np.zeros(n_sims, dtype=int)
        for u in range(n_teams):
            if u == t:
                continue
            better += (wins[:, u] > wins[:, t]) | (
                (wins[:, u] == wins[:, t]) & (points[:, u] > points[:, t])
            )
        rank[:, t] = 1 + better

    pw = np.asarray(playoff_cols, dtype=int)
    champions = np.empty(n_sims, dtype=int)
    for s in range(n_sims):
        seeds = seed_order(wins[s], points[s], n_playoff_teams)
        champions[s] = playoff_champion(seeds, scores[s][:, pw])

    champ_counts = np.bincount(champions, minlength=n_teams).astype(float)
    playoff_prob = (rank <= n_playoff_teams).mean(axis=0)
    return RegimeResult(
        name="",
        champ_prob=champ_counts / n_sims,
        playoff_prob=playoff_prob,
        exp_wins=wins.mean(axis=0),
        exp_points=points.mean(axis=0),
        my_wins=wins[:, my_team],
        my_rank=rank[:, my_team],
        my_points=points[:, my_team],
        my_is_champ=champions == my_team,
    )


def simulate_season(
    pool: SeasonPool,
    schedule: Mapping[int, Sequence[tuple[int, int]]],
    my_team: int,
    *,
    season: int,
    regular_weeks: Sequence[int],
    playoff_weeks: Sequence[int],
    n_playoff_teams: int = 6,
    n_sims: int = 2000,
    seed: int = 0,
    opp_noise: float = DEFAULT_OPP_NOISE,
    my_noise: float = DEFAULT_MY_NOISE,
    actual_scores: Mapping[int, Sequence[float]] | None = None,
) -> SeasonSimOutput:
    """Run the full-season Monte Carlo and return per-team + my-team aggregates for both regimes.

    ``actual_scores`` (week -> per-team-index real scores) pins already-played weeks to what actually
    happened in every sim — a mid-season run then answers "title odds from HERE (my real record)",
    not the preseason question. Weeks absent from the mapping are simulated as usual.
    """
    total_weeks = int(max(playoff_weeks))
    cv, gcv, p_set, sev = pool_arrays(pool)
    mean_week = pool.mean / float(total_weeks)

    rng = np.random.default_rng(seed)
    weekly = dist.sample_weekly_points(rng, pool.mean, cv, n_sims, total_weeks, game_cv=gcv)
    out = dist.sample_injury_out(rng, p_set, sev, n_sims, total_weeks)
    raw_noise = rng.standard_normal(weekly.shape)
    realized = weekly * (~out)

    playoff_cols = [w - 1 for w in playoff_weeks]
    regimes: dict[str, RegimeResult] = {}
    #: equal_skill — every manager equally sloppy (a realistic league, me included);
    #: my_edge — I set clean lineups while opponents keep their start/sit noise.
    noise_by_regime = {
        "equal_skill": [opp_noise] * pool.n_teams,
        "my_edge": [my_noise if t == my_team else opp_noise for t in range(pool.n_teams)],
    }
    for name, team_noise in noise_by_regime.items():
        scores = team_week_scores(pool, realized, out, mean_week, raw_noise, team_noise)
        for w, vals in (actual_scores or {}).items():
            if 1 <= int(w) <= total_weeks:
                scores[:, :, int(w) - 1] = np.asarray(vals, dtype=float)[None, :]
        res = _rank_and_champ(scores, schedule, playoff_cols, n_playoff_teams, my_team)
        res.name = name
        regimes[name] = res

    return SeasonSimOutput(
        pool=pool,
        my_team=my_team,
        season=season,
        n_sims=n_sims,
        seed=seed,
        total_weeks=total_weeks,
        regular_weeks=list(regular_weeks),
        playoff_weeks=list(playoff_weeks),
        n_playoff_teams=n_playoff_teams,
        regimes=regimes,
        opp_noise=opp_noise,
        conditioned_weeks=sorted(int(w) for w in (actual_scores or {})),
    )


def build_signature(pool: SeasonPool) -> Counter:
    """Roster-composition signature per team (debug aid): positional counts."""
    sig: Counter = Counter()
    for cols in pool.rosters:
        sig["/".join(sorted(pool.pos[i] for i in cols))] += 1
    return sig

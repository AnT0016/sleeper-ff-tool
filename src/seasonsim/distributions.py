"""Weekly outcome + injury draws for the season simulator.

The draft simulator draws a single *season* total per player. A championship sim needs **weekly**
granularity — head-to-head weeks are won and lost on single-game variance, and a multi-week injury has
to knock a player out of specific weeks so the bench actually gets tested. This module provides both
draws, reusing :mod:`draftsim.distributions`' per-position knobs (CV and injury risk) so the two
simulators stay consistent.

ASSUMPTIONS (heuristic, *not* fitted — printed in the report so they can be judged):

* **Weeks are independent** lognormal draws whose *sum* reproduces the player's season projection and
  season CV. If a week has mean ``m/W`` and the season (sum of ``W`` weeks) should have CV ``c``, then
  each week needs ``CV_week = c * sqrt(W)`` — single-game fantasy is much noisier than a full season
  (a WR who averages a WR2 line still has 2-point and 30-point weeks). Independence means we model no
  within-season hot/cold streak (a real effect that would *widen* season swings) — a stated v1 limit.
* **Injuries** are one *significant* multi-week setback per season (Bernoulli per position); if it
  fires, a contiguous ``Poisson(severity)``-week stretch starting at a uniformly-random week is zeroed
  out. That empties the player's lineup slot for those weeks — the durability risk a real bench covers.
* **Byes are not modeled** in v1: the season projection is spread evenly across all weeks, so there is
  no forced one-week hole for a team's bye. This is uniform across teams, so it barely moves *relative*
  championship odds; explicit byes are a documented future refinement.
"""

from __future__ import annotations

import numpy as np

# Reuse the draft simulator's per-position variance / durability knobs so the two stay in lockstep.
from draftsim.distributions import (  # noqa: F401  (re-exported for the report)
    DEFAULT_CV,
    DEFAULT_RISK,
    INJURY_RISK,
    POSITION_CV,
    SEASON_GAMES,
    lognormal_params,
)


def weekly_cv(season_cv: np.ndarray, n_weeks: int) -> np.ndarray:
    """Per-week CV whose ``n_weeks`` independent draws reproduce ``season_cv`` on the season total."""
    return np.asarray(season_cv, dtype=float) * np.sqrt(float(n_weeks))


def sample_weekly_points(
    rng: np.random.Generator,
    season_mean: np.ndarray,
    season_cv: np.ndarray,
    n_sims: int,
    n_weeks: int,
) -> np.ndarray:
    """``(n_sims, n_players, n_weeks)`` weekly points, independent lognormal, mean-preserving.

    Each week's mean is ``season_mean / n_weeks`` and each week's CV is ``season_cv * sqrt(n_weeks)``,
    so summing the weeks recovers a season total with the intended mean and CV. A non-positive (or
    missing) projection stays exactly zero every week rather than becoming lognormal noise.
    """
    season_mean = np.asarray(season_mean, dtype=float)
    season_cv = np.asarray(season_cv, dtype=float)
    wk_mean = season_mean / float(n_weeks)
    wk_cv = weekly_cv(season_cv, n_weeks)
    mu, sigma = lognormal_params(wk_mean, wk_cv)  # per-player (n_players,)
    z = rng.standard_normal((n_sims, season_mean.size, n_weeks))
    pts = np.exp(mu[None, :, None] + sigma[None, :, None] * z)
    pts[:, season_mean <= 0.0, :] = 0.0
    return pts


def sample_injury_out(
    rng: np.random.Generator,
    p_setback: np.ndarray,
    severity: np.ndarray,
    n_sims: int,
    n_weeks: int,
) -> np.ndarray:
    """``(n_sims, n_players, n_weeks)`` boolean "out this week" mask from one multi-week setback/season.

    Per (sim, player): a ``Bernoulli(p_setback)`` setback; if it fires, a contiguous stretch of
    ``clip(Poisson(severity), 1, n_weeks)`` weeks starting at a uniformly-random week is marked out
    (truncated at the season's end). Positions/players with ``p_setback == 0`` never miss time.
    """
    p = np.asarray(p_setback, dtype=float)
    sev = np.asarray(severity, dtype=float)
    n_players = p.size
    shape = (n_sims, n_players)
    setback = rng.random(shape) < p[None, :]
    dur = np.clip(rng.poisson(np.broadcast_to(sev[None, :], shape)), 1, n_weeks)
    start = rng.integers(0, n_weeks, size=shape)
    end = start + dur  # exclusive; weeks past n_weeks simply don't exist

    weeks = np.arange(n_weeks)[None, None, :]
    within = (weeks >= start[:, :, None]) & (weeks < end[:, :, None])
    return setback[:, :, None] & within

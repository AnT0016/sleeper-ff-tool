"""Weekly outcome + injury draws for the season simulator.

The draft simulator draws a single *season* total per player. A championship sim needs **weekly**
granularity — head-to-head weeks are won and lost on single-game variance, and a multi-week injury has
to knock a player out of specific weeks so the bench actually gets tested. This module provides both
draws, reusing :mod:`draftsim.distributions`' per-position knobs (CV and injury risk) so the two
simulators stay consistent.

ASSUMPTIONS (heuristic, *not* fitted — printed in the report so they can be judged):

* **A player's week = per-season factor × single-game draw.** The single-game draw is lognormal at
  the shared per-position ``GAME_CV`` (the same realistic one-week noise the win-probability model
  uses — head-to-head weeks are decided by THIS number, so it must not be inflated). The per-season
  factor (drawn once per sim × player, mean 1) carries the rest of the season-level uncertainty —
  role changes, breakouts, busts — sized so the season TOTAL still reproduces the position's season
  CV. The old independence-derived ``season CV × √W`` weekly noise reproduced the season total too,
  but made every single week ~2× too noisy, compressing records and title odds toward a coin flip.
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
    DEFAULT_GAME_CV,
    DEFAULT_RISK,
    GAME_CV,
    INJURY_RISK,
    POSITION_CV,
    SEASON_GAMES,
    lognormal_params,
)


def season_factor_cv(season_cv: np.ndarray, game_cv: np.ndarray, n_weeks: int) -> np.ndarray:
    """CV of the per-season factor so that ``factor × Σ weekly`` has the season CV.

    For independent lognormals, ``1 + CV_total² = (1 + CV_factor²) × (1 + CV_week² / W)`` — solve for
    ``CV_factor`` and floor at 0 (if single-game noise alone already exceeds the season CV).
    """
    c = np.asarray(season_cv, dtype=float)
    g = np.asarray(game_cv, dtype=float)
    ratio = (1.0 + c * c) / (1.0 + g * g / float(n_weeks))
    return np.sqrt(np.maximum(ratio - 1.0, 0.0))


def sample_weekly_points(
    rng: np.random.Generator,
    season_mean: np.ndarray,
    season_cv: np.ndarray,
    n_sims: int,
    n_weeks: int,
    *,
    game_cv: np.ndarray | None = None,
) -> np.ndarray:
    """``(n_sims, n_players, n_weeks)`` weekly points, mean-preserving.

    Each week is an independent lognormal at the realistic single-game ``game_cv`` around
    ``season_mean / n_weeks``, multiplied by a once-per-(sim, player) lognormal season factor
    (mean 1) sized by :func:`season_factor_cv` — so single weeks stay realistically noisy while the
    season total keeps the position's full season CV. A non-positive (or missing) projection stays
    exactly zero every week rather than becoming lognormal noise.
    """
    season_mean = np.asarray(season_mean, dtype=float)
    season_cv = np.asarray(season_cv, dtype=float)
    g = np.full(season_mean.shape, DEFAULT_GAME_CV) if game_cv is None else np.asarray(game_cv, dtype=float)

    wk_mean = season_mean / float(n_weeks)
    mu_w, sigma_w = lognormal_params(wk_mean, g)  # per-player (n_players,)
    z = rng.standard_normal((n_sims, season_mean.size, n_weeks))
    pts = np.exp(mu_w[None, :, None] + sigma_w[None, :, None] * z)

    f_cv = season_factor_cv(season_cv, g, n_weeks)
    mu_f, sigma_f = lognormal_params(np.ones_like(f_cv), f_cv)  # mean-1 factor
    zf = rng.standard_normal((n_sims, season_mean.size))
    factor = np.exp(mu_f[None, :] + sigma_f[None, :] * zf)
    pts *= factor[:, :, None]

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

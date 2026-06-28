"""Outcome distributions and the injury (durability) model for the Monte Carlo draft simulator.

Each player carries a *single* custom-scored season projection (its mean). To compare roster builds
by the **distribution** of season outcomes — not just expected value — every simulation redraws a
season total per player and applies a durability haircut. Drafters never see these draws; only the
post-draft evaluation does (see :mod:`draftsim.engine`).

ASSUMPTIONS (heuristic, *not* fitted to data — this is a directional v1 tool, and these knobs are
printed in the report so they can be judged):

* **Season points ~ lognormal**, parameterised so its *mean* equals the projection and its
  *coefficient of variation* (CV = std / mean) is a per-position constant. Lognormal keeps season
  totals non-negative and right-skewed — real fantasy seasons have a fat upside tail (a league
  winner) and a floor at zero.
* **Injuries** are modelled as one *significant* multi-week "setback" per season: a Bernoulli per
  position; if it fires, games missed ~ Poisson(severity) clipped to ``[1, SEASON_GAMES]``. The
  resulting availability multiplier ``(games_played / SEASON_GAMES)`` scales the sampled season
  total. Week-to-week noise (a quiet game, a dinged ankle) is already inside the CV above — the
  setback is the *durability* risk you'd want a real rostered backup for.
"""

from __future__ import annotations

import numpy as np

SEASON_GAMES = 17

#: Coefficient of variation (std / mean) of season fantasy points, by position. QB/K are the most
#: predictable; TE/RB the most volatile (boom/bust + injury-driven). Heuristic.
POSITION_CV: dict[str, float] = {
    "QB": 0.18,
    "RB": 0.32,
    "WR": 0.30,
    "TE": 0.35,
    "K": 0.20,
    "DEF": 0.28,
}
DEFAULT_CV = 0.30

#: (P(a significant multi-week setback in a season), mean games missed when it occurs), by position.
#: RBs get hurt most and miss the most time; K/DEF are nearly never the reason you lose a week.
INJURY_RISK: dict[str, tuple[float, float]] = {
    "QB": (0.25, 3.0),
    "RB": (0.45, 4.0),
    "WR": (0.35, 3.0),
    "TE": (0.35, 3.0),
    "K": (0.05, 2.0),
    "DEF": (0.02, 1.0),
}
DEFAULT_RISK = (0.30, 3.0)


def lognormal_params(mean: np.ndarray, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, sigma)`` of the underlying normal so the lognormal has this ``mean`` and ``cv``.

    For a lognormal, ``CV = sqrt(exp(sigma^2) - 1)`` and ``mean = exp(mu + sigma^2 / 2)``.
    """
    sigma = np.sqrt(np.log1p(cv * cv))
    mu = np.log(np.maximum(mean, 1e-9)) - 0.5 * sigma * sigma
    return mu, sigma


def sample_season_points(rng: np.random.Generator, mean, cv, n_sims: int) -> np.ndarray:
    """Sample an ``(n_sims, n_players)`` matrix of season points (lognormal, mean-preserving)."""
    mean = np.asarray(mean, dtype=float)
    cv = np.asarray(cv, dtype=float)
    mu, sigma = lognormal_params(mean, cv)
    z = rng.standard_normal((n_sims, mean.size))
    pts = np.exp(mu + sigma * z)
    pts[:, mean <= 0.0] = 0.0  # a zero (or missing) projection stays zero, not exp(-inf)-ish noise
    return pts


def sample_availability(rng: np.random.Generator, p_setback, severity, n_sims: int):
    """Sample the season availability multiplier and the setback flag.

    Returns ``(multiplier, setback)`` each shaped ``(n_sims, n_players)``: ``multiplier`` in
    ``(0, 1]`` is ``games_played / SEASON_GAMES``; ``setback`` is the boolean "had a significant
    injury this season" used for the injury-insight report.
    """
    p = np.asarray(p_setback, dtype=float)
    sev = np.asarray(severity, dtype=float)
    n = p.size
    setback = rng.random((n_sims, n)) < p
    games = rng.poisson(np.broadcast_to(sev, (n_sims, n)))
    games = np.clip(games, 1, SEASON_GAMES)  # a setback costs at least one game
    missed = np.where(setback, games, 0)
    multiplier = (SEASON_GAMES - missed) / SEASON_GAMES
    return multiplier, setback

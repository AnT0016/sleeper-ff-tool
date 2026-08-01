"""Outcome distributions and the injury (durability) model for the Monte Carlo draft simulator.

Each player carries a *single* custom-scored season projection (its mean). To compare roster builds
by the **distribution** of season outcomes — not just expected value — every simulation redraws a
season total per player and applies a durability haircut. Drafters never see these draws; only the
post-draft evaluation does (see :mod:`draftsim.engine`).

The knobs are FITTED, with a heuristic fallback (Phase 9, ticket #32)
--------------------------------------------------------------------
``POSITION_CV``, ``GAME_CV`` and ``INJURY_RISK`` are earned from the lake, not hand-picked. The fit
lives in a committed data artifact (``src/model/fit/distributions.json``, produced by
``scripts/eval_distributions.py`` → the measured report ``docs/model-distributions.md``); this module
**reads** it at import and merges the fitted value **over** the heuristic constant **per position**,
so a position the fit could not decide keeps its heuristic. Where each number came from:

* **Season points ~ lognormal**, parameterised so its *mean* equals the projection and its
  *coefficient of variation* (CV = std / mean) is a per-position constant :data:`POSITION_CV`. Lognormal
  keeps season totals non-negative and right-skewed. The CV is fitted from :class:`model.season.SeasonModel`'s
  **walk-forward out-of-sample** residuals ``actual/pred`` on the **drafted cohort** (the per-season top-N
  by projection this league rosters — fitting a per-position knob over the wider fringe measures the
  volatile backups the sim never drafts), **setback-free** — because week-to-week noise is already inside
  this CV while the *durability* risk below owns games-missed, the two must not double-count. A position
  whose fitted CV is not **coherent** with its game CV under the sim's season-factor identity keeps the
  heuristic (its residual is recorded as an upper bound in the report). The fit is a per-position
  upper-ish bound anyway: SeasonModel's residual bounds the spread around *its own* projection, wider than
  around a market projection the sim will eventually use.
* **Single-game CV** :data:`GAME_CV` — fitted from the weekly models' out-of-sample residuals
  (:class:`model.weekly.WeeklyModel` for QB/RB/WR/TE, :class:`model.kickdef.KickDefModel` for K/DEF). A
  week with no stat line does not exist in the frame (an inactive player is simply absent — the #29
  finding), so these residuals are already conditioned on availability; no healthy filter is needed at
  game grain. Shared with the win-probability model and the season sim so all three stay consistent.
* **Injuries** are one *significant* multi-week "setback" per season: a Bernoulli per position; if it
  fires, games missed ~ Poisson(severity) clipped to ``[1, SEASON_GAMES]``. The availability multiplier
  ``(games_played / SEASON_GAMES)`` scales the sampled season total. :data:`INJURY_RISK` is fitted on the
  same **drafted cohort** as the season CV, from a contiguous **injury absence** of ≥ 2 weeks — a gap in
  a player's *played* weeks, tenure-bounded and injury-corroborated by the weekly report (which excludes
  byes and clean benchings). Not from ``report_status == "Out"`` runs: a player on IR drops off the
  weekly report, so a season-ending injury leaves no ``Out`` weeks at all and an Out-only read undercounts
  the rate several-fold, exactly on the severe injuries this knob exists to model.
  **DST is never on the injury report, so it keeps the heuristic** (and any position too thin to fit does
  too — that is what the fallback is for).

Safe by default (spec Decision #9). A missing or unreadable artifact → **every** position falls back to
its heuristic; a position marked ``heuristic-fallback`` in the artifact → that position falls back. The
merge always yields all six positions, never a ``{}``-shaped emptiness that would read as "fitted with
zero spread". :func:`use_knobs` is the runtime swap used by the before/after harness and the tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SEASON_GAMES = 17

#: The fitted artifact this module reads at import. Written by ``scripts/eval_distributions.py``; the
#: sims read it here so it is a *source*, not a write-only record (spec Decision #9 item 4).
FITTED_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "model" / "fit" / "distributions.json"

#: Heuristic **fallbacks**, retained verbatim from the v1 hand-picked knobs. These are what a position
#: falls back to when the fit is missing or undecided — never edited to a fitted value, so the fallback
#: path stays honest and the artifact remains the single source of the fitted numbers.
HEURISTIC_POSITION_CV: dict[str, float] = {
    "QB": 0.18,
    "RB": 0.32,
    "WR": 0.30,
    "TE": 0.35,
    "K": 0.20,
    "DEF": 0.28,
}
DEFAULT_CV = 0.30

HEURISTIC_GAME_CV: dict[str, float] = {
    "QB": 0.45,
    "RB": 0.60,
    "WR": 0.70,
    "TE": 0.75,
    "K": 0.50,
    "DEF": 0.85,
}
DEFAULT_GAME_CV = 0.65

HEURISTIC_INJURY_RISK: dict[str, tuple[float, float]] = {
    "QB": (0.25, 3.0),
    "RB": (0.45, 4.0),
    "WR": (0.35, 3.0),
    "TE": (0.35, 3.0),
    "K": (0.05, 2.0),
    "DEF": (0.02, 1.0),
}
DEFAULT_RISK = (0.30, 3.0)


# --------------------------------------------------------------------------- the fitted-over-heuristic merge
def _read_artifact(path: str | Path) -> dict:
    """The fitted artifact as a dict, or ``{}`` if it is missing or unreadable (fail-safe, never raise).

    Import must not fall over on a fresh checkout that has not run ``eval_distributions`` yet, so any
    read/parse failure degrades to "no fitted values" — every position then takes its heuristic.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_scalar(artifact: Mapping, knob_key: str, heuristic: Mapping[str, float]) -> dict[str, float]:
    """Fitted-over-heuristic for a scalar CV knob: use the fit only where its verdict is ``fitted``.

    Starts from a copy of ``heuristic`` (so **all** positions are present) and overrides a position only
    when the artifact records a ``fitted`` verdict with a finite positive value. A ``heuristic-fallback``
    cell, an absent cell, or a malformed value all leave that position on its heuristic.
    """
    out = dict(heuristic)
    cells = artifact.get(knob_key)
    if not isinstance(cells, Mapping):
        return out
    for pos in heuristic:
        cell = cells.get(pos)
        if isinstance(cell, Mapping) and cell.get("verdict") == "fitted":
            value = cell.get("value")
            if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
                out[pos] = float(value)
    return out


def _merge_injury(
    artifact: Mapping, heuristic: Mapping[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """Fitted-over-heuristic for ``INJURY_RISK``: a fitted cell supplies ``(p_setback, mean_games)``."""
    out = {pos: (float(p), float(g)) for pos, (p, g) in heuristic.items()}
    cells = artifact.get("injury_risk")
    if not isinstance(cells, Mapping):
        return out
    for pos in heuristic:
        cell = cells.get(pos)
        if isinstance(cell, Mapping) and cell.get("verdict") == "fitted":
            p, g = cell.get("p"), cell.get("games")
            if (
                isinstance(p, (int, float))
                and isinstance(g, (int, float))
                and np.isfinite(p)
                and np.isfinite(g)
                and 0.0 <= p <= 1.0
                and g > 0.0
            ):
                out[pos] = (float(p), float(g))
    return out


_ARTIFACT = _read_artifact(FITTED_ARTIFACT_PATH)

#: Coefficient of variation (std / mean) of season fantasy points, by position — fitted where the
#: artifact decided it, heuristic elsewhere (see the module docstring). QB/K are the most predictable;
#: TE/RB the most volatile.
POSITION_CV: dict[str, float] = _merge_scalar(_ARTIFACT, "position_cv", HEURISTIC_POSITION_CV)

#: Coefficient of variation of a SINGLE GAME's fantasy points, by position — fitted from the weekly
#: models' out-of-sample residuals, heuristic elsewhere. Far larger than the season CV per week-slice
#: would suggest, but far smaller than the independence-derived ``season CV × √17`` (which over-skews
#: one game). Shared by the win-probability model and the season sim so the three stay consistent.
GAME_CV: dict[str, float] = _merge_scalar(_ARTIFACT, "game_cv", HEURISTIC_GAME_CV)

#: (P(a significant multi-week setback in a season), mean games missed when it occurs), by position —
#: fitted from injury-corroborated multi-week absences on the drafted cohort, heuristic where a position
#: cannot be separated cleanly (DST is never on the injury report). RB and WR carry the most risk, then
#: TE, then QB; K is an order of magnitude safer. ``mean games missed`` includes season-enders, so it
#: runs above the heuristic's return-from-injury figure.
INJURY_RISK: dict[str, tuple[float, float]] = _merge_injury(_ARTIFACT, HEURISTIC_INJURY_RISK)


def _knob_sources(artifact: Mapping) -> dict[str, dict[str, str]]:
    """Per-knob, per-position ``"fitted"`` / ``"heuristic"`` — what the sims' reports mark as judgeable."""
    out: dict[str, dict[str, str]] = {}
    for knob in ("position_cv", "game_cv", "injury_risk"):
        cells = artifact.get(knob)
        out[knob] = {}
        for pos in HEURISTIC_POSITION_CV:
            cell = cells.get(pos) if isinstance(cells, Mapping) else None
            out[knob][pos] = "fitted" if isinstance(cell, Mapping) and cell.get("verdict") == "fitted" else "heuristic"
    return out


#: ``{knob: {position: "fitted"|"heuristic"}}`` — read by both sims' reports so a fallback value is
#: shown as a fallback (a trailing ``*``), keeping every knob judgeable (spec Decision #9 / ticket #32).
KNOB_SOURCES: dict[str, dict[str, str]] = _knob_sources(_ARTIFACT)


def is_fitted(knob: str, position: str) -> bool:
    """Did ``position``'s ``knob`` (``"position_cv"`` / ``"game_cv"`` / ``"injury_risk"``) ship fitted?"""
    return KNOB_SOURCES.get(knob, {}).get(position) == "fitted"


# --------------------------------------------------------------------------- runtime knob swap
def _apply_inplace(target: dict, new: Mapping | None) -> None:
    """Replace ``target``'s contents with ``new`` **in place** (``clear``/``update``), or leave it be.

    In place, never a rebind, and that is load-bearing — see :func:`use_knobs`.
    """
    if new is None:
        return
    target.clear()
    target.update(new)


@contextmanager
def use_knobs(
    *,
    position_cv: Mapping[str, float] | None = None,
    game_cv: Mapping[str, float] | None = None,
    injury_risk: Mapping[str, tuple[float, float]] | None = None,
) -> Iterator[None]:
    """Temporarily replace the module knob dicts **in place**, restoring them on exit.

    Used by the before/after harness (``scripts/eval_distributions.py``) to run the sim under heuristic
    vs fitted knobs, and by the tests. A knob left ``None`` is untouched, so the four before/after arms
    are composed by passing exactly the dicts that arm should use.

    **The mutation must be in place, and this is why it is not merely a style choice.** Two other
    surfaces bind to these dict *objects* at import:

    * ``optimizer.winprob.WEEKLY_CV = GAME_CV`` — a name bound to the object, and
    * ``seasonsim.engine`` reads ``distributions.POSITION_CV.get(...)`` — an attribute lookup at call time,
      on the object ``seasonsim.distributions`` re-exported (the same object, by import binding).

    Rebinding ``draftsim.distributions.GAME_CV = {...}`` would leave ``winprob.WEEKLY_CV`` pointing at
    the *old* dict while ``seasonsim`` follows the *new* one — the two surfaces would silently disagree
    and nothing would error. Mutating the one shared object in place keeps all three consistent, which a
    test (``test_use_knobs_is_seen_by_winprob``) pins.
    """
    saved = (dict(POSITION_CV), dict(GAME_CV), dict(INJURY_RISK))
    _apply_inplace(POSITION_CV, position_cv)
    _apply_inplace(GAME_CV, game_cv)
    _apply_inplace(INJURY_RISK, injury_risk)
    try:
        yield
    finally:
        _apply_inplace(POSITION_CV, saved[0])
        _apply_inplace(GAME_CV, saved[1])
        _apply_inplace(INJURY_RISK, saved[2])


# --------------------------------------------------------------------------- samplers (unchanged)
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

"""The three naive baselines every real model must beat (Phase 9, ticket #28).

Each is a full :class:`~model.evaluate.Predictor`, so a five-line heuristic and a gradient-booster are
scored through the identical harness. A model that cannot clear these under the same walk-forward
evaluation is one we learn about in an afternoon rather than in October.

They are deliberately three *different* signals, so beating all three means beating recency, level and
opportunity at once:

* :class:`TrailingMean` — the player's recent within-season form (the heuristic every human uses).
* :class:`PriorSeasonRank` — last season's established scoring level, carried into this one.
* :class:`LaggedExpectedPoints` — last week's *opportunity* (nflverse expected points), independent of
  whether it converted to actual points.

Cold-start fallback. Every baseline shares one backstop: a per-position mean of the label learned on
the training seasons. A player's first week of a season has no within-season history; a kicker or a
defense has no expected-points column. Rather than emit a null there, the baseline emits the learned
position mean, so all three cover 100% of cohort rows and are compared on an identical row universe.
The fallback is learned strictly from the training frame — seasons ``< S`` under walk-forward — so it
is as lookahead-free as the signals it backstops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model.evaluate import LABEL_COL

_POSITION_COL = "position"


class _FallbackBaseline:
    """Shared machinery: a learned per-position label mean, used wherever the own signal is missing."""

    def __init__(self) -> None:
        self._position_mean: dict[str, float] = {}
        self._global_mean: float = 0.0

    def fit(self, frame: pd.DataFrame) -> _FallbackBaseline:
        y = pd.to_numeric(frame[LABEL_COL], errors="coerce")
        self._global_mean = float(y.mean()) if bool(y.notna().any()) else 0.0
        means = (
            pd.DataFrame({"position": frame[_POSITION_COL].astype("string"), "y": y})
            .dropna(subset=["y"])
            .groupby("position", observed=True)["y"]
            .mean()
        )
        self._position_mean = {str(pos): float(v) for pos, v in means.items()}
        return self

    def _fallback(self, positions: pd.Series) -> pd.Series:
        mapped = positions.astype("string").map(self._position_mean)
        return pd.to_numeric(mapped, errors="coerce").fillna(self._global_mean)

    def _signal(self, frame: pd.DataFrame) -> pd.Series:
        """The baseline's own prediction, frame-index-aligned, null where it does not apply."""
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        signal = pd.to_numeric(self._signal(frame), errors="coerce").reindex(frame.index)
        fallback = self._fallback(frame[_POSITION_COL]).reindex(frame.index)
        return signal.where(signal.notna(), fallback).astype("float64")


class TrailingMean(_FallbackBaseline):
    """A player's mean custom points over his last ``n`` completed weeks — the human heuristic.

    Computed within ``(player_id, season)`` and shifted one week, so week ``W``'s prediction never
    sees week ``W``'s own label. That is the same content-rule lag the training frame's usage features
    are built on (``dataset.assemble._lagged_usage``): the earlier weeks' outcomes were all realized
    before week ``W`` kicked off, so using them is legal, and the shift is what keeps it so. A player's
    first appearance of a season has no prior week and falls back to the learned position mean —
    matching the frame's own cold-start shape (profile #27, §3).
    """

    def __init__(self, n: int = 4) -> None:
        super().__init__()
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n

    def _signal(self, frame: pd.DataFrame) -> pd.Series:
        work = pd.DataFrame(
            {
                "y": pd.to_numeric(frame[LABEL_COL], errors="coerce"),
                "player_id": frame["player_id"].astype("string"),
                "season": pd.to_numeric(frame["season"], errors="coerce"),
                "week": pd.to_numeric(frame["week"], errors="coerce"),
            },
            index=frame.index,
        )
        ordered = work.sort_values(["player_id", "season", "week"], kind="stable")
        trailing = ordered.groupby(["player_id", "season"], sort=False, observed=True)["y"].transform(
            lambda s: s.shift(1).rolling(self.n, min_periods=1).mean()
        )
        return trailing.reindex(frame.index)


class PriorSeasonRank(_FallbackBaseline):
    """The player's prior-season (S-1) points-per-week average, held constant across season S.

    Learned in :meth:`fit` from the training frame and joined on ``(player_id, season - 1)`` at
    predict time. For test season S that prior season is S-1, always inside the walk-forward train
    set, so nothing from the future enters. The name is literal about the signal: giving every player
    of a position his own prior-season rate *orders* players within position by last season's scoring
    rank — the persistence a manager leans on for an early-season start/sit before this year's sample
    exists. A player with no prior season in the frame — a rookie, or anyone who recorded no stat line
    last year — has no rate and falls back to the learned position mean. Rookies are a large,
    predictable, every-year cohort, not an edge case (Decision, ticket #31 makes the same point).
    """

    def __init__(self) -> None:
        super().__init__()
        self._per_week_avg: dict[tuple[str, int], float] = {}

    def fit(self, frame: pd.DataFrame) -> PriorSeasonRank:
        super().fit(frame)
        avg = (
            pd.DataFrame(
                {
                    "player_id": frame["player_id"].astype("string"),
                    "season": pd.to_numeric(frame["season"], errors="coerce").astype("Int64"),
                    "y": pd.to_numeric(frame[LABEL_COL], errors="coerce"),
                }
            )
            .dropna(subset=["y", "season"])
            .groupby(["player_id", "season"], observed=True)["y"]
            .mean()
        )
        self._per_week_avg = {(str(pid), int(season)): float(v) for (pid, season), v in avg.items()}
        return self

    def _signal(self, frame: pd.DataFrame) -> pd.Series:
        pid = frame["player_id"].astype("string")
        prior = pd.to_numeric(frame["season"], errors="coerce") - 1
        values = [
            self._per_week_avg.get((str(p), int(s))) if pd.notna(s) else None
            for p, s in zip(pid, prior, strict=True)
        ]
        return pd.Series(values, index=frame.index, dtype="float64")


class LaggedExpectedPoints(_FallbackBaseline):
    """nflverse expected fantasy points, lagged one week — last week's earned opportunity.

    The frame already carries it lagged as ``exp_points_last`` (``dataset.assemble._lagged_usage``).
    Expected points is post-game content, so it is legal only lagged (Phase 8 Decision #6), which this
    column already is. It captures the volume a player earned last week regardless of whether it paid
    off in points — a distinct signal from either points baseline. It is an offensive-skill signal:
    ``nflverse_ff_opp`` covers ``QB/RB/WR/TE``, so ``DEF`` has the column empty outright and ``K`` is
    **99.8% empty** — not 100% (profile #27 §5b; 10 of 5,253 kicker rows carry a value, which is an
    ID-crosswalk residue, not a kicker signal). For both positions this baseline is therefore the
    learned position mean on essentially every row, and the handful of K exceptions are why
    ``spearman_ordered_slates`` exists: they are enough to make a rho *printable* while carrying no
    ordering anyone could use. Exactly why the recorded bar is kept per position rather than pooled.
    """

    _COLUMN = "exp_points_last"

    def _signal(self, frame: pd.DataFrame) -> pd.Series:
        if self._COLUMN not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[self._COLUMN], errors="coerce")

"""Point-in-time capture of the betting market, derived from nflverse's schedule.

The market is the cheapest good prior a fantasy projection can have: an implied team total is the
book's own forecast of how many points a team will score, which is most of what a player's ceiling
depends on. nflverse publishes ``spread_line`` / ``total_line`` / moneylines on every schedule row,
so v1 needs **no odds provider and no key** (spec Q-C: closing lines from schedules, revisit only if
too coarse). Reading a published line as a model feature is not betting — this project is fantasy
only, places no wagers, and never writes anywhere.

``vegas_odds`` is registered as its own source rather than left inside ``nflverse_schedules``
because the derivation is the point: two computed columns that a reader can check against the two
raw ones sitting beside them. The raw market columns are carried **verbatim** for exactly that
reason, and the post-game columns (``result``/``total``/scores) are deliberately *not* — this source
is what the market believed before kickoff, and mixing an outcome into it invites the leak this
whole phase exists to prevent. The outcome is one join away in ``nflverse_schedules``.

.. important::

   **Sign convention: a positive ``spread_line`` means the HOME team is favoured**, so

   .. code-block:: text

      home_implied = total_line / 2 + spread_line / 2
      away_implied = total_line / 2 - spread_line / 2

   This is the opposite of the sportsbook spelling most people carry around (where the favourite is
   quoted at ``-3``), and the ticket's suggested formula assumed that spelling. Verified on the real
   feed rather than assumed, over all 2,761 games with a result in 2016-2025:

   * ``corr(spread_line, result)`` = **+0.44**, where ``result`` is nflverse's ``home_score -
     away_score``. A negated convention would give -0.44.
   * of 1,711 games with ``spread_line > 0``, the home team won **67.1%**.
   * the moneylines agree on which side is favoured in **99.2%** of 2,756 non-pick'em games
     (the residual is pick'em-adjacent lines where the two markets disagree by a hair).

   Worked example from the committed fixture — ``2024_01_BAL_KC``: ``total_line`` 46.0,
   ``spread_line`` +3.0, ``home_moneyline`` -148 (KC favoured) -> KC 24.5, BAL 21.5, summing to
   46.0. And ``2024_01_HOU_IND``: ``spread_line`` -3.0 with ``away_moneyline`` -155 -> HOU (away)
   24.5, IND 21.5. ``tests/test_collect_market.py`` pins both directions.

Games with no line yet are kept with null implied totals rather than dropped: on the forward
schedule most of the season is unpriced in July (221 of 272 rows on the 2026 file today), and
"unpriced as of this capture" is itself point-in-time data. Dropping them would make a partition
look complete when it is not.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import polars as pl

from data import nflverse

from .base import Collected, dedupe_rows
from .registry import SOURCES

_LOG = logging.getLogger(__name__)

Loader = Callable[..., pl.DataFrame]

#: Schedule columns carried through unchanged. Identity and timing first, then the raw market.
#: ``season`` is absent because the store stamps ``_season`` on every row and this source is
#: season-partitioned; ``week`` is present because the partition does *not* say it.
_CARRIED: tuple[str, ...] = (
    "game_id",
    "game_type",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "location",
    "div_game",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
)

#: Columns without which this source has nothing to say. The two lines *are* the derivation, and the
#: team names are what tells you which side each implied total belongs to; if any of them vanish
#: from the feed the capture would be a full partition of nulls that looks like a real one.
_REQUIRED: tuple[str, ...] = ("total_line", "spread_line", "home_team", "away_team")


def _number(value: Any) -> float | None:
    """``value`` as a float, or ``None`` for null/NaN — the shape a missing line has."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN


def implied_totals(
    total_line: Any, spread_line: Any
) -> tuple[float | None, float | None]:
    """``(home_implied, away_implied)`` for a game, or ``(None, None)`` if either line is missing.

    The book prices a game as a total and a margin; splitting them back into two team totals is
    just solving ``home + away == total`` and ``home - away == spread`` (nflverse's spread is the
    home team's expected margin — see the module ``.. important::``). By construction the pair sums
    to ``total_line`` exactly, which is what the acceptance criterion checks.
    """
    total = _number(total_line)
    spread = _number(spread_line)
    if total is None or spread is None:
        return None, None
    return total / 2 + spread / 2, total / 2 - spread / 2


def favored_team(row: Mapping[str, Any]) -> str | None:
    """The side the spread favours, or ``None`` for a pick'em or an unpriced game."""
    spread = _number(row.get("spread_line"))
    if not spread:  # None or exactly 0.0 -> nobody is favoured
        return None
    return row.get("home_team") if spread > 0 else row.get("away_team")


def _season_of(frame: pl.DataFrame) -> int:
    """The frame's own season stamp; refuses a multi-season frame rather than guessing."""
    if "season" not in frame.columns:
        raise ValueError("vegas_odds: no season column to partition on — pass season= explicitly")
    seasons = frame["season"].drop_nulls().unique().sort().to_list()
    if not seasons:
        raise ValueError("vegas_odds: season column is empty — pass season= explicitly")
    if len(seasons) > 1:
        raise ValueError(
            f"vegas_odds: the frame spans seasons {seasons}; one capture is one partition, so pass "
            "one season's schedule (or season= and a pre-filtered frame)"
        )
    return int(seasons[0])


def _row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """One schedule row -> one market row: carried columns, then the derivation."""
    row: dict[str, Any] = {name: raw.get(name) for name in _CARRIED}
    home_implied, away_implied = implied_totals(raw.get("total_line"), raw.get("spread_line"))
    row["home_implied_total"] = home_implied
    row["away_implied_total"] = away_implied
    row["favored_team"] = favored_team(raw)
    return row


def collect_vegas_from_schedules(
    schedules_df: pl.DataFrame | None = None,
    *,
    season: int | None = None,
    load: Loader | None = None,
) -> Collected:
    """``vegas_odds``: closing lines plus implied team totals, one row per game.

    Game-grain and **season-partitioned** (``week=None``), mirroring ``nflverse_schedules`` — the
    loader hands back a whole season at once, and ``game_id`` is unique across it. Pass
    ``schedules_df`` (what ticket 5's runner does, so one download feeds both collectors) or leave
    it out and a ``season`` is loaded via ``data.nflverse.load_schedules``.

    The season comes from the frame's own ``season`` column when not given. A frame spanning several
    seasons is refused rather than silently filed under one of them.
    """
    if schedules_df is None:
        if season is None:
            raise ValueError("collect_vegas_from_schedules: pass a schedules frame or a season")
        schedules_df = (load or nflverse.load_schedules)(season)

    key_cols = SOURCES["vegas_odds"].key_cols
    missing = [c for c in (*key_cols, *_REQUIRED) if c not in schedules_df.columns]
    if missing:
        # Same rule as collect.nflverse: a schema change that removes the key must not yield a
        # capture the store then merges on whatever is left — and one that removes a line must not
        # yield a partition of nulls indistinguishable from an unpriced week.
        raise ValueError(
            f"vegas_odds: required column(s) {missing} absent from the schedules frame "
            f"(columns: {sorted(schedules_df.columns)[:20]}); the provider schema has changed"
        )

    if season is None:
        season = _season_of(schedules_df)
    rows = dedupe_rows(
        (_row(raw) for raw in schedules_df.to_dicts()), key_cols, source="vegas_odds"
    )
    priced = sum(1 for row in rows if row["home_implied_total"] is not None)
    _LOG.info(
        "vegas_odds season=%s: %d game(s), %d priced (%d with no line yet)",
        season, len(rows), priced, len(rows) - priced,
    )
    return Collected.for_source("vegas_odds", season, rows, week=None)

"""Generate (and refresh) the committed offline fixtures for the nflverse collectors.

Writes one small **parquet** file per source under ``tests/fixtures/nflverse/``, sampled from the
real nflverse releases. Parquet rather than JSON (which is what ``scripts/make_collect_fixture.py``
uses for Sleeper) because for these feeds the *dtypes* are half of what the collector has to get
right: ``injuries.date_modified`` is a tz-aware ``Datetime`` that is part of the key,
``ff_opportunity`` types ``season`` as a string and ``week`` as a float, and a JSON round-trip would
quietly hand the tests a tidied frame the provider never produces.

Every sample is chosen for a shape the collector must survive, not for volume:

* ``player_week`` — a top scorer per skill position (so a K's FG-distance buckets and a QB's passing
  line are both present), their postseason rows (``season_type`` shares the week numbering, which is
  why it is in the key), and the null-``player_id`` residual rows nflverse files per team-game.
* ``ff_opp`` — one whole game, which brings its per-team unattributed rows (null ``player_id``).
* ``injuries`` — a player-week carrying **two** report revisions with different ``date_modified``.
  That pair is the entire justification for the key, so the fixture must contain one.
* ``depth`` — the modern (2025+) schema over two ``dt`` snapshots, plus a null-``gsis_id`` pair that
  collides on the registry key; and ``depth_legacy``, a pre-2025 frame for the fallback path.
* ``schedules`` — a full week, so the Vegas (``spread_line``/``total_line``) and weather
  (``roof``/``temp``/``wind``) columns ticket 4 reads are covered, domes included. A played week, so
  the market columns can be checked against the outcome (``result`` is ``home_score - away_score``,
  which is what pins the ``spread_line`` sign).
* ``schedules_intl`` — the **forward** schedule's awkward shapes, which a played season has none of:
  international games carrying the *home team's* ``stadium_id`` and ``roof`` while only ``stadium``
  names the real venue, retractable-roof games with ``roof`` still null, and unpriced games with a
  null ``spread_line``/``total_line``.

Run manually when refreshing the fixtures (mirrors ``scripts/make_fixture.py``); the tests never
call the network.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from data.nflverse import (
    load_depth_charts,
    load_ff_opportunity,
    load_id_crosswalk,
    load_injuries,
    load_schedules,
    load_snap_counts,
    load_weekly_actuals,
)

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "nflverse"

#: The last complete season on the *legacy* depth-chart schema, and the season every other fixture
#: is sampled from (finalized, so the sample is stable across refreshes).
SEASON = 2024
#: First season of the rewritten depth-chart feed — the only shape the registry key addresses.
DEPTH_SEASON = 2025
#: An *unplayed* season, for the shapes only a forward schedule has (see ``schedules_intl``).
FORWARD_SEASON = 2026

POSITIONS = ("QB", "RB", "WR", "TE", "K")


def _player_week() -> pl.DataFrame:
    frame = load_weekly_actuals(SEASON)
    best = (
        frame.filter(
            (pl.col("season_type") == "REG") & pl.col("position").is_in(POSITIONS)
        )
        .sort("fantasy_points", descending=True, nulls_last=True)
        .group_by("position", maintain_order=True)
        .head(1)
    )
    picked = best["player_id"].to_list()
    kept = frame.filter(
        pl.col("player_id").is_in(picked)
        & ((pl.col("week") <= 3) | (pl.col("season_type") == "POST"))
    )
    residual = frame.filter(pl.col("player_id").is_null()).head(2)
    return pl.concat([kept, residual], how="vertical")


def _ff_opp() -> pl.DataFrame:
    frame = load_ff_opportunity(SEASON)
    # A game that actually has the null-player_id team rows, so the grain filter is exercised.
    game = frame.filter(pl.col("player_id").is_null())["game_id"][0]
    return frame.filter(pl.col("game_id") == game)


def _injuries() -> pl.DataFrame:
    frame = load_injuries(SEASON)
    revised = (
        frame.group_by("gsis_id", "game_type", "week")
        .agg(pl.col("date_modified").n_unique().alias("revisions"))
        .filter(pl.col("revisions") > 1)
    )
    if revised.is_empty():  # pragma: no cover - never seen; the report is a revision stream
        raise RuntimeError(f"{SEASON} injuries carry no re-reported player-week to sample")
    first = revised.head(1).to_dicts()[0]
    pair = frame.filter(
        (pl.col("gsis_id") == first["gsis_id"])
        & (pl.col("game_type") == first["game_type"])
        & (pl.col("week") == first["week"])
    )
    others = frame.filter(pl.col("week") == 1).head(20)
    return pl.concat([pair, others], how="vertical").unique(maintain_order=True)


def _depth() -> pl.DataFrame:
    frame = load_depth_charts(DEPTH_SEASON)
    snapshots = frame["dt"].unique().sort().head(2).to_list()
    team = frame.filter(pl.col("dt").is_in(snapshots) & pl.col("gsis_id").is_not_null())
    team = team.filter(pl.col("team") == team["team"][0])
    # A null-gsis pair that collides on the registry key: proves the grain filter removes the
    # duplicate as well as the null, so dedupe_rows has nothing left to warn about.
    unnamed = (
        frame.filter(pl.col("gsis_id").is_null())
        .group_by("dt", "team", "pos_abb")
        .len()
        .filter(pl.col("len") > 1)
        .head(1)
    )
    if unnamed.is_empty():  # pragma: no cover - 502 such pairs in 2025
        raise RuntimeError(f"{DEPTH_SEASON} depth charts carry no colliding unnamed entry")
    hit = unnamed.to_dicts()[0]
    collision = frame.filter(
        (pl.col("dt") == hit["dt"])
        & (pl.col("team") == hit["team"])
        & (pl.col("pos_abb") == hit["pos_abb"])
        & pl.col("gsis_id").is_null()
    )
    return pl.concat([team, collision], how="vertical")


def _schedules_intl() -> pl.DataFrame:
    """Forward-schedule rows the played seasons cannot supply.

    Three shapes, all of which ``collect.weather`` has to survive and none of which exist in a
    finished season: an international game filed under the home team's ``stadium_id`` **and**
    ``roof`` (so venue resolution must go by ``stadium`` name), a retractable-roof game whose
    ``roof`` is still null because nobody knows yet, and a game with no line priced.
    """
    frame = load_schedules(FORWARD_SEASON)
    neutral = frame.filter(pl.col("location") == "Neutral")
    no_roof = frame.filter((pl.col("location") == "Home") & pl.col("roof").is_null()).head(3)
    unpriced = frame.filter(pl.col("total_line").is_null()).head(3)
    priced = frame.filter(pl.col("total_line").is_not_null()).head(3)
    return pl.concat([neutral, no_roof, unpriced, priced], how="vertical").unique(
        subset=["game_id"], maintain_order=True
    )


def _crosswalk() -> pl.DataFrame:
    frame = load_id_crosswalk()
    joinable = frame.filter(
        pl.col("sleeper_id").is_not_null() & pl.col("gsis_id").is_not_null()
    ).head(20)
    # mfl_id is the key precisely because the other id columns are nullable — keep proof of that.
    orphan = frame.filter(pl.col("sleeper_id").is_null()).head(2)
    return pl.concat([joinable, orphan], how="vertical")


def main() -> None:
    samples: dict[str, pl.DataFrame] = {
        f"player_week_{SEASON}": _player_week(),
        f"snaps_{SEASON}": load_snap_counts(SEASON).filter(pl.col("week") == 1).head(40),
        f"ff_opp_{SEASON}": _ff_opp(),
        f"injuries_{SEASON}": _injuries(),
        f"schedules_{SEASON}": load_schedules(SEASON).filter(pl.col("week") == 1),
        f"schedules_intl_{FORWARD_SEASON}": _schedules_intl(),
        f"depth_{DEPTH_SEASON}": _depth(),
        f"depth_legacy_{SEASON}": load_depth_charts(SEASON).head(10),
        "id_crosswalk": _crosswalk(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in samples.items():
        path = OUT / f"{name}.parquet"
        frame.write_parquet(path)
        print(f"{name:24} {frame.height:4d} rows x {frame.width:3d} cols "
              f"-> {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()

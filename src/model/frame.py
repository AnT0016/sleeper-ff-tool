"""The player x season **draft** frame, aggregated from the weekly training frame (Phase 9, #31).

The draft model is a *separate* model, not an aggregation of the weekly one (spec finding #2): no
historical season-grain projection exists to train on, and the weekly model's features — current-season
lagged usage, that game's Vegas line, that game's weather — simply do not exist in August. So the draft
path needs its own construction: **prior-season aggregates → the next season's real custom-scored
total**. This module builds that frame; :mod:`model.season` fits the model on it.

One row per ``(sleeper player_id, season S)``:

* **label** ``y_season_points`` — the player's real, re-scored custom **season total** for S, i.e. the
  sum of the engine-scored weekly ``y_custom_points`` the assembler already produced. Scoring is never
  hand-coded here: the label flows out of :func:`dataset.assemble.build_training_frame`, which scores
  every week through the Phase 1 engine from the league's live ``scoring_settings``.
* **features** — aggregates drawn from seasons **≤ S-1 only**: last season's points, games, snap /
  target / rush share and expected points; a career-to-date games/season count; and whether the player
  changed teams between his two most recent seasons.

Why aggregating the weekly frame is the right construction
----------------------------------------------------------
The weekly frame is the sanctioned Phase 8 → 9 hand-off, and it is where "what was known when" was
already decided, audited and tested. Building the season frame as a **pure transform of that frame**
means the two models consume the identical lookahead-safe data, and any future correction to the
assembler's gate propagates to both for free. :func:`season_frame_from_weekly` is that pure transform
and is testable with no lake and no network.

The lookahead rule, at season grain
------------------------------------
A feature for target season S may use only seasons **strictly before** S. That is exactly the weekly
gate's *content rule* (``feature_week < N``) lifted to seasons: a whole strictly-earlier season is
knowable before S's first kickoff however late it was scored, so no capture-time reasoning is needed —
the season boundary does the gating. Everything in :data:`SEASON_FEATURES` is therefore computed from a
player's ``S-1`` (and, for the team-change flag, ``S-2``) aggregate row and never from S itself.

Two attributes are deliberately taken from S and are **not** features, only the grouping/label context
a draft board needs to exist at all:

* ``position`` — a near-static attribute, the grain the ranking is scored within (a rookie cannot be
  ranked without knowing he is an RB). The weekly frame carries the target week's ``position`` for the
  same reason; using it is not reading the season's outcome.
* ``team`` — the display team on the emitted board. It is never read as a predictive feature (the
  feature set is the explicit :data:`SEASON_FEATURES` list), and the fail-closed test pins that a
  player who moved *into* S gets no signal from that move.

Why "team change" is an S-1-vs-S-2 flag, not the offseason move into S
---------------------------------------------------------------------
The obviously useful draft signal is the offseason team switch *into* S (the WR on a new team). But the
team a player is on in season S is only *provable* pre-season from a roster source the historical lake
does not have — ``nflverse_depth`` starts 2025 and Sleeper's roster state is forward-only. Deriving it
instead from S's own game logs would be reading the season we are predicting. So, fail-closed (spec
acceptance #1), the season-S team is withheld and ``changed_team_prior`` is defined strictly within the
prior window: did the player's S-1 team differ from his S-2 team. It is a real "just moved" signal and
it cannot leak.

Rookies are a cohort, not an edge case
--------------------------------------
A player whose S is his first appearance in the built window has no S-1 row: every prior-season feature
is null and ``is_rookie`` is set. That is a large, predictable, every-year population (spec acceptance
#4), so it is represented explicitly rather than dropped — :mod:`model.season` learns a per-position
level for it exactly as :class:`model.baselines.PriorSeasonRank` falls back for a rookie.

The **earliest built season is a warm-up**, flagged ``is_warmup``: it has no in-window prior at all, so
every one of its players reads ``is_rookie`` by arithmetic rather than by observation — a cohort that is
overwhelmingly veterans whose history simply predates the collection. It is never a scored test season
(``model.evaluate.DEFAULT_TEST_SEASONS`` starts two seasons in), and :meth:`model.season.SeasonModel.fit`
also drops it from training so the learned rookie level is not taken from that mislabelled population.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import pandas as pd

from dataset.assemble import build_training_frame
from store.lake import StorageBackend

_LOG = logging.getLogger(__name__)

#: The season frame's natural key.
SEASON_KEY: tuple[str, ...] = ("player_id", "season")

#: The re-scored season-total label.
LABEL_COL = "y_season_points"

#: Weekly lagged-usage columns aggregated to a prior-season mean, and the season-feature they become.
#: The weekly frame carries usage only in lagged form (``dataset.assemble._lagged_usage``); a whole
#: season's mean of the lag is the prior season's mean weekly usage, and it is legal as an S feature
#: purely because the season is strictly earlier (the content rule) — the intra-season lag is immaterial
#: once the entire prior season is used.
_USAGE_MEAN: dict[str, str] = {
    "snap_pct_last": "prior_snap_share",
    "target_share_last": "prior_target_share",
    "rush_share_last": "prior_rush_share",
    "exp_points_last": "prior_exp_points",
}

#: The predictive features, all provably ≤ S-1. This tuple is the contract: :mod:`model.season` reads
#: exactly these (plus the ``is_rookie`` / ``has_prior_season`` indicators), so a column taken from S
#: cannot become a feature by accident.
SEASON_FEATURES: tuple[str, ...] = (
    "prior_points_total",
    "prior_points_per_game",
    "prior_games",
    "prior_snap_share",
    "prior_target_share",
    "prior_rush_share",
    "prior_exp_points",
    "career_games",
    "career_seasons",
    "changed_team_prior",
)

#: Identity / label / grouping context first, features after — the frame reads top-down.
_LEADING_COLS: tuple[str, ...] = (
    *SEASON_KEY,
    "position",
    "team",
    "is_dst",
    "is_rookie",
    "has_prior_season",
    "is_warmup",
    LABEL_COL,
)

#: Weekly columns the aggregation reads. A usage column absent from the frame is treated as all-null
#: rather than raising, so a narrowed weekly build still assembles.
_REQUIRED_WEEKLY: tuple[str, ...] = ("player_id", "season", "y_custom_points")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _modal_by_group(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """The most frequent non-null ``value_col`` per ``(player_id, season)``; ties broken alphabetically.

    Vectorised (a size-count then an idxmax-by-sort) rather than a per-group Python mode, so it stays
    cheap on the full ~170k-row frame. The alphabetical tie-break is arbitrary but stable across runs,
    which matters for a categorical the model groups on.
    """
    sub = frame[["player_id", "season", value_col]].dropna(subset=[value_col]).copy()
    sub[value_col] = sub[value_col].astype("string")
    if sub.empty:
        return pd.DataFrame({"player_id": [], "season": [], value_col: []})
    counts = (
        sub.groupby(["player_id", "season", value_col], observed=True)
        .size()
        .reset_index(name="_n")
        .sort_values(
            ["player_id", "season", "_n", value_col], ascending=[True, True, False, True]
        )
    )
    top = counts.drop_duplicates(["player_id", "season"], keep="first")
    return top[["player_id", "season", value_col]].reset_index(drop=True)


def _per_season(weekly: pd.DataFrame) -> pd.DataFrame:
    """Collapse the weekly frame to one aggregate row per ``(player_id, season)``.

    Numeric aggregates: the exact season total and games from the raw label, and the prior-season mean
    of each lagged-usage column. Categorical: the modal team and position for the season.
    """
    work = pd.DataFrame(
        {
            "player_id": weekly["player_id"].astype("string"),
            "season": pd.to_numeric(weekly["season"], errors="coerce").astype("Int64"),
            "y": _num(weekly["y_custom_points"]),
        }
    )
    for col in _USAGE_MEAN:
        work[col] = _num(weekly[col]) if col in weekly.columns else pd.Series(pd.NA, dtype="Float64")
    work["is_dst"] = (
        weekly["is_dst"].fillna(False).astype(bool)
        if "is_dst" in weekly.columns
        else pd.Series(False, index=weekly.index)
    )
    work = work.dropna(subset=["player_id", "season"])

    agg = {"points_total": ("y", "sum"), "games": ("y", "size"), "is_dst": ("is_dst", "max")}
    agg.update({src: (src, "mean") for src in _USAGE_MEAN})
    per = work.groupby(["player_id", "season"], observed=True).agg(**agg).reset_index()
    per = per.rename(columns=_USAGE_MEAN)
    per["points_per_game"] = per["points_total"] / per["games"].where(per["games"] > 0)

    for value_col, out_col in (("team", "team"), ("position", "position")):
        modal = _modal_by_group(weekly, value_col) if value_col in weekly.columns else None
        if modal is None or modal.empty:
            per[out_col] = pd.Series(pd.NA, index=per.index, dtype="string")
        else:
            per = per.merge(modal, on=["player_id", "season"], how="left")
            per[out_col] = per[value_col].astype("string")
            if value_col != out_col:
                per = per.drop(columns=[value_col])
    return per


def _career_to_date(per: pd.DataFrame) -> pd.DataFrame:
    """Games and seasons a player accrued **strictly before** each of his seasons (≤ S-1)."""
    ordered = per.sort_values(["player_id", "season"], kind="stable")
    group = ordered.groupby("player_id", observed=True)
    career = pd.DataFrame(
        {
            "player_id": ordered["player_id"].to_numpy(),
            "season": ordered["season"].to_numpy(),
            "career_seasons": group.cumcount().to_numpy(),
            "career_games": (group["games"].cumsum() - ordered["games"]).to_numpy(),
        }
    )
    return career


def season_frame_from_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a weekly training frame to the per-``(player, season)`` draft frame (pure, offline).

    ``weekly`` is the output of :func:`dataset.assemble.build_training_frame`. Every feature is derived
    from strictly-earlier seasons; ``position`` and ``team`` are the season's own static context, never
    predictive features (see the module docstring). Rows are emitted for every ``(player, season)`` that
    has a label, rookies included, sorted by the natural key.
    """
    missing = [c for c in _REQUIRED_WEEKLY if c not in weekly.columns]
    if missing:
        raise ValueError(
            f"weekly frame is missing column(s) {missing} — pass the output of build_training_frame"
        )
    if weekly.empty:
        return pd.DataFrame(columns=[*_LEADING_COLS, *SEASON_FEATURES])

    per = _per_season(weekly)

    # Prior season (S-1): join each aggregate row onto the row one season later.
    prior_cols = {
        "points_total": "prior_points_total",
        "points_per_game": "prior_points_per_game",
        "games": "prior_games",
        "prior_snap_share": "prior_snap_share",
        "prior_target_share": "prior_target_share",
        "prior_rush_share": "prior_rush_share",
        "prior_exp_points": "prior_exp_points",
        "team": "prior_team",
    }
    prior = per[["player_id", "season", *prior_cols]].copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns=prior_cols)

    # Two seasons back (S-2): only the team, for the change flag.
    prior2 = per[["player_id", "season", "team"]].copy()
    prior2["season"] = prior2["season"] + 2
    prior2 = prior2.rename(columns={"team": "prior2_team"})

    frame = per[["player_id", "season", "position", "team", "is_dst", "points_total"]].rename(
        columns={"points_total": LABEL_COL}
    )
    frame = frame.merge(prior, on=["player_id", "season"], how="left")
    frame = frame.merge(prior2, on=["player_id", "season"], how="left")
    frame = frame.merge(_career_to_date(per), on=["player_id", "season"], how="left")

    both_teams = frame["prior_team"].notna() & frame["prior2_team"].notna()
    frame["changed_team_prior"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    frame.loc[both_teams, "changed_team_prior"] = (
        (frame.loc[both_teams, "prior_team"] != frame.loc[both_teams, "prior2_team"])
        .astype("float64")
    )

    frame["has_prior_season"] = frame["prior_points_total"].notna()
    frame["is_rookie"] = frame["career_seasons"].fillna(0).astype(int) == 0
    frame["is_dst"] = frame["is_dst"].fillna(False).astype(bool)
    # The earliest built season has no in-window prior, so its is_rookie is arithmetic (100% true), not
    # observation — the model excludes this warm-up from training (see the module docstring).
    earliest = int(pd.to_numeric(frame["season"], errors="coerce").min())
    frame["is_warmup"] = pd.to_numeric(frame["season"], errors="coerce") == earliest

    for col in ("prior_games", "career_games", "career_seasons"):
        frame[col] = _num(frame[col])

    ordered = [*_LEADING_COLS, *SEASON_FEATURES]
    frame = frame[[c for c in ordered if c in frame.columns]]
    _LOG.info(
        "season frame: %d row(s), %d player(s), season(s) %s — %d rookie, %d with a prior season",
        len(frame), frame["player_id"].nunique(),
        sorted({int(s) for s in frame["season"].dropna().unique()}),
        int(frame["is_rookie"].sum()), int(frame["has_prior_season"].sum()),
    )
    return frame.sort_values(list(SEASON_KEY), kind="stable").reset_index(drop=True)


def build_season_frame(
    seasons: Iterable[int],
    scoring: Mapping[str, float],
    *,
    backend: StorageBackend | None = None,
) -> pd.DataFrame:
    """One row per ``(player, season)``: prior-season aggregates → that season's real custom total.

    ``scoring`` is the league's live ``scoring_settings`` dict — it is passed straight through to
    :func:`dataset.assemble.build_training_frame`, which re-scores every week through the Phase 1
    engine. This function itself never touches a scoring coefficient; the season label is a sum of the
    weekly label. ``seasons`` is the full span to build (e.g. ``2016-2025``): rows are emitted for every
    season in it, and the earliest are lag warm-up (their features are null for want of a prior season).
    """
    if not scoring:
        raise ValueError("scoring is empty — pass the league's live scoring_settings dict")
    weekly = build_training_frame(seasons, scoring, backend=backend)
    return season_frame_from_weekly(weekly)

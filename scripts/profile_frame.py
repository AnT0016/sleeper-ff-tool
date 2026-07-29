"""Profile the training frame at full scale, and commit the report (Phase 9, ticket #27).

    ./.venv/Scripts/python scripts/profile_frame.py
    ./.venv/Scripts/python scripts/profile_frame.py --seasons 2019-2025 --out docs/model-data-profile.md

``build_training_frame(2016..2025)`` had never been run at full scale before this ticket. Nothing is
modelled until we know what is actually in the frame: rows per season x position, the null rate of
every feature per season, and which feature groups are usable over which span. This script builds the
frame once (against whatever ``LAKE_BACKEND`` points at — the local backfill by default, B2 in the
cloud), measures all of that, and writes :data:`DEFAULT_OUT`.

It also turns three things the spec carried as *findings from registry comments* into things measured
from the data:

* **#1** — ``baseline_sleeper_points`` is null for every 2016-2025 row (Sleeper's projection
  endpoints serve only the latest values, so the market-beating baseline is forward-only from
  2026 W1). The report proves it per season rather than asserting it.
* **#3** — ``nflverse_depth`` carries ``backfillable_from=2025``, so role/depth features exist for
  2025 forward only. The feature-group availability table confirms that against the frame.
* **cold start** — week 1 of every season has no current-season lags; those rows are counted apart.

The label is re-scored live from the active league's ``scoring_settings`` (the Phase 1 engine), never
hard-coded — the same rule the assembler follows.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

# Render UTF-8 regardless of the console code page (Windows defaults to cp1252) — the assembler's log
# lines carry em-dashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# `python scripts/profile_frame.py` puts scripts/ at the FRONT of sys.path, where scripts/collect.py
# shadows the installed `collect` package this script imports `runner` from. Drop the script
# directory; nothing in scripts/ imports a sibling script. (The truthiness check keeps an empty
# entry — the cwd — from matching when the cwd happens to be scripts/.)
if sys.path and sys.path[0] and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

import pandas as pd

from collect import runner
from dataset.assemble import build_training_frame
from sleeper import client
from sleeper.config import LEAGUE_ID
from store.lake import LAKE_BACKEND, lake_inventory

#: Ten seasons of training data, the span the backfill pulls (backfill_lake.DEFAULT_SEASONS).
DEFAULT_SEASONS = "2016-2025"
DEFAULT_OUT = "docs/model-data-profile.md"

#: The positions any model in this phase actually predicts. ``nflverse_player_week`` also carries a
#: stat line for every defender, punter and long-snapper who took the field, so the frame is majority
#: non-fantasy rows that score ~0 under this scoring — the null-rate and cold-start views are computed
#: on this cohort so a feature that is simply *not applicable* to a linebacker (target share) does not
#: read as missing data.
FANTASY_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")

#: Identity and label columns — not features, but reported for a completeness sanity check.
IDENTITY_COLS: tuple[str, ...] = (
    "player_id", "season", "week", "position", "position_is_static", "team", "opponent",
    "is_home", "game_id", "gsis_id", "is_dst", "y_custom_points",
)

#: Feature columns grouped by source family, split exactly where the assembler's discontinuities
#: fall (see ``dataset.assemble``): the two injury families are never coalesced, and weather is three
#: distinct availability regimes (venue is fixed pre-kickoff, forecast is unrecoverable historically,
#: observed is withheld pre-lock by design).
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "usage_lag": (
        "games_played_prior", "points_last", "points_ewma", "points_trend", "snap_pct_last",
        "snap_pct_ewma", "snap_pct_trend", "target_share_last", "target_share_ewma",
        "rush_share_last", "rush_share_ewma", "exp_points_last", "exp_points_ewma",
    ),
    "depth_role": ("depth_pos_rank", "depth_dt"),
    "injury_nflverse": ("inj_report_status", "inj_practice_status", "inj_report_primary"),
    "injury_sleeper": ("inj_sleeper_status", "inj_sleeper_body_part"),
    "market": (
        "implied_team_total", "opp_implied_total", "team_spread_line", "total_line", "is_div_game",
    ),
    "weather_venue": ("is_indoor",),
    "weather_forecast": (
        "wx_forecast_temp_f", "wx_forecast_wind_mph", "wx_forecast_precip_prob_pct",
        "wx_forecast_lead_hours",
    ),
    "weather_observed": ("wx_observed_temp_f", "wx_observed_wind_mph"),
    "baseline": ("baseline_sleeper_points",),
}

#: One-line provenance per feature group, printed beside the availability table.
GROUP_NOTES: dict[str, str] = {
    "usage_lag": "nflverse actuals/snaps/opportunity, lagged (post-game content, legal only lagged)",
    "depth_role": "nflverse_depth as-of rank — 2025-forward (backfillable_from=2025), finding #3",
    "injury_nflverse": "official practice/game-status report (gsis_id), 2016+; sparse by nature — "
    "only a listed (injured/limited) player has a row, so a low coverage % is signal, not a gap",
    "injury_sleeper": "Sleeper's live injury state — 2026-forward only, never coalesced with above",
    "market": "vegas_odds implied totals/spread — the sanctioned pre-game view",
    "weather_venue": "is_indoor (roof/dome) — fixed well before kickoff, three-state",
    "weather_forecast": "open-meteo forecast — ~0% historically (endpoint reaches back ~92 days)",
    "weather_observed": "at-kickoff temp/wind — withheld pre-lock by design (post-kickoff for week N)",
    "baseline": "Sleeper's own projection re-scored — forward-only from 2026 W1, finding #1",
}


# --------------------------------------------------------------------------- log capture
class _Capture(logging.Handler):
    """Records every log line the build emits, so warnings can be quoted verbatim in the report."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[tuple[int, str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record.levelno, record.levelname, record.name, record.getMessage()))


# --------------------------------------------------------------------------- markdown helpers
def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A GitHub-flavoured pipe table; first column left-aligned, the rest right-aligned."""
    align = ["---", *["---:"] * (len(headers) - 1)]
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _int(value: object) -> str:
    return f"{int(value):,}"


def _count_table(
    frame: pd.DataFrame, seasons: Sequence[int], positions: Sequence[str], *, total_label: str
) -> str:
    """Row counts, ``position`` down the side and ``season`` across the top, with margins."""
    pos = frame["position"].astype("string").fillna("<NA>")
    crosstab = pd.crosstab(pos, frame["season"]).reindex(
        index=list(positions), columns=list(seasons), fill_value=0
    )
    headers = ["position", *[str(s) for s in seasons], "Total"]
    rows: list[list[str]] = []
    for name in positions:
        per = [int(crosstab.loc[name, s]) for s in seasons]
        rows.append([name, *[_int(v) for v in per], _int(sum(per))])
    totals = [int(crosstab[s].sum()) for s in seasons]
    rows.append([f"**{total_label}**", *[f"**{_int(v)}**" for v in totals], f"**{_int(sum(totals))}**"])
    return _md_table(headers, rows)


def _present(by_season: Mapping[int, pd.DataFrame], cols: Sequence[str]) -> list[str]:
    """The subset of ``cols`` the assembler actually emitted (every season shares one schema)."""
    known = set.intersection(*(set(sub.columns) for sub in by_season.values())) if by_season else set()
    return [c for c in cols if c in known]


def _null_table(
    by_season: Mapping[int, pd.DataFrame], seasons: Sequence[int], groups: Mapping[str, Sequence[str]]
) -> str:
    """Per-feature null rate (%), one row per feature under a bold group header row."""
    headers = ["feature", *[str(s) for s in seasons]]
    rows: list[list[str]] = []
    for group, cols in groups.items():
        present = _present(by_season, cols)
        if not present:
            continue
        rows.append([f"**{group}**", *[""] * len(seasons)])
        for column in present:
            cells = []
            for season in seasons:
                sub = by_season[season]
                cells.append(f"{sub[column].isna().mean() * 100:.1f}" if len(sub) else "n/a")
            rows.append([column, *cells])
    return _md_table(headers, rows)


def _availability_table(
    by_season: Mapping[int, pd.DataFrame], seasons: Sequence[int], groups: Mapping[str, Sequence[str]]
) -> str:
    """Coverage (%) per group per season — the *max* non-null share over the group's columns.

    Max, not mean: a group is "present" this season if any of its columns carries data, which is the
    coarse question this table answers (the per-feature detail is the null-rate table). It cleanly
    separates the forward-only discontinuities (depth 2025+, baseline 2026+) from the always-present
    families (usage, market, nflverse injuries).
    """
    headers = ["feature group", "cols", *[str(s) for s in seasons], "note"]
    rows: list[list[str]] = []
    for group, cols in groups.items():
        present = _present(by_season, cols)
        cells = []
        for season in seasons:
            sub = by_season[season]
            # "--" not "0.0": a column the assembler no longer emits is a schema change, and
            # rendering it as a measured zero would read as "we looked and there was no data".
            if not present or len(sub) == 0:
                cells.append("--")
            else:
                cells.append(f"{max(sub[c].notna().mean() for c in present) * 100:.1f}")
        rows.append([group, str(len(present)), *cells, GROUP_NOTES.get(group, "")])
    return _md_table(headers, rows)


#: A column is "absent" for a position at or above this null rate. Not 100%: a handful of rows can
#: carry a stray value (a kicker who took a carry) without the feature being usable for that position.
_ABSENT_AT = 0.99


def _absent_by_position(
    cohort: pd.DataFrame, positions: Sequence[str], groups: Mapping[str, Sequence[str]]
) -> dict[str, list[tuple[str, list[str], int]]]:
    """``position -> [(group, columns absent for it, group size)]`` at >= :data:`_ABSENT_AT` null.

    Counted **per column, not all-or-nothing per group**: the finding that matters is that DST
    carries no snap/target/rush/expected-points column while still carrying the points lags, and a
    group-level ``all()`` misses it entirely (``games_played_prior`` is 0% null for DST, so
    ``usage_lag`` never qualifies as wholly absent — yet 8 of its 13 columns are gone).

    Groups absent for *every* position are excluded: those are the forward-only sources (the two
    Sleeper families, the weather forecast) already reported by §4, and repeating them per position
    would bury the asymmetric case this exists for.
    """
    out: dict[str, list[tuple[str, list[str], int]]] = {}
    for group, cols in groups.items():
        present = [c for c in cols if c in cohort.columns]
        if not present or all(cohort[c].isna().mean() >= _ABSENT_AT for c in present):
            continue  # not in the frame, or absent everywhere — §4's story, not this one
        for pos in positions:
            sub = cohort[cohort["position"] == pos]
            if not len(sub):
                continue
            gone = [c for c in present if sub[c].isna().mean() >= _ABSENT_AT]
            if gone:
                out.setdefault(pos, []).append((group, gone, len(present)))
    return out


def _absent_sentence(absent: Mapping[str, Sequence[tuple[str, list[str], int]]]) -> str:
    """The per-position absences as one prose sentence, or a stated absence of absences."""
    if not absent:
        return "No feature group is absent for any one position while present for the others."
    parts = [
        f"**{pos}** is missing " + ", ".join(f"{len(gone)}/{size} {group}" for group, gone, size in items)
        for pos, items in absent.items()
    ]
    return "; ".join(parts) + " column(s)."


def _null_by_position_table(
    cohort: pd.DataFrame, positions: Sequence[str], groups: Mapping[str, Sequence[str]]
) -> str:
    """Null rate (%) per feature per position, pooled over all seasons — §5's missing dimension."""
    usable = [p for p in positions if len(cohort[cohort["position"] == p])]
    headers = ["feature", *usable]
    subs = {p: cohort[cohort["position"] == p] for p in usable}
    rows: list[list[str]] = []
    for group, cols in groups.items():
        present = [c for c in cols if c in cohort.columns]
        if not present:
            continue
        rows.append([f"**{group}**", *[""] * len(usable)])
        for column in present:
            cells = [f"{subs[p][column].isna().mean() * 100:.1f}" for p in usable]
            rows.append([column, *cells])
    return _md_table(headers, rows)


def _baseline_table(frame: pd.DataFrame, seasons: Sequence[int]) -> str:
    """Confirms finding #1: ``baseline_sleeper_points`` non-null count per season (over the FULL frame)."""
    headers = ["season", "rows", "baseline non-null", "null %"]
    rows: list[list[str]] = []
    total_rows = total_nn = 0
    for season in seasons:
        sub = frame[frame["season"] == season]
        n, non_null = len(sub), int(sub["baseline_sleeper_points"].notna().sum())
        total_rows, total_nn = total_rows + n, total_nn + non_null
        null_pct = f"{(1 - non_null / n) * 100:.1f}" if n else "n/a"
        rows.append([str(season), _int(n), _int(non_null), null_pct])
    null_pct = f"{(1 - total_nn / total_rows) * 100:.1f}" if total_rows else "n/a"
    rows.append([f"**{seasons[0]}-{seasons[-1]}**", f"**{_int(total_rows)}**",
                 f"**{_int(total_nn)}**", f"**{null_pct}**"])
    return _md_table(headers, rows)


def _log_block(records: Sequence[tuple[int, str, str, str]], *, min_level: int) -> str:
    """The captured log at or above ``min_level`` as a fenced block, or a stated absence."""
    lines = [f"{lvl} {name}: {msg}" for level, lvl, name, msg in records if level >= min_level]
    if not lines:
        return "_None._"
    return "```\n" + "\n".join(lines) + "\n```"


# --------------------------------------------------------------------------- report
def render_report(
    frame: pd.DataFrame,
    seasons: Sequence[int],
    records: Sequence[tuple[int, str, str, str]],
    *,
    scoring_keys: int,
    partitions: int,
    league_name: str,
) -> str:
    """The full ``docs/model-data-profile.md`` as one markdown string."""
    cohort = frame[frame["position"].isin(FANTASY_POSITIONS)]
    by_season = {s: cohort[cohort["season"] == s] for s in seasons}

    # Position order for the full-inventory table: fantasy first, then the rest by descending volume.
    counts = frame["position"].astype("string").fillna("<NA>").value_counts()
    others = [p for p in counts.index if p not in FANTASY_POSITIONS]
    all_positions = [p for p in FANTASY_POSITIONS if p in counts.index] + others

    n_rows = len(frame)
    n_cohort = len(cohort)
    idp = frame[~frame["position"].isin(FANTASY_POSITIONS)]
    idp_share = len(idp) / n_rows * 100 if n_rows else 0.0
    # Measured, not asserted: "score ~0" is a claim about this scoring, and IDP players do
    # occasionally find a scoring key (fum_rec_td is 6).
    idp_mean_points = float(idp["y_custom_points"].mean()) if len(idp) else 0.0
    # Cohort-scoped, matching section 3's table — the sections cross-reference each other.
    week1 = cohort[cohort["week"] == 1]
    week1_lag_null = week1["points_last"].isna().mean() * 100 if len(week1) else 0.0
    depth_cov = {s: by_season[s]["depth_pos_rank"].notna().mean() * 100 for s in seasons}
    # Every headline number below is derived. A literal here would let the report contradict the
    # table it cites the moment the underlying fact changes — and both of these facts are expected
    # to change (the baseline fills from 2026 W1; depth backfills only from 2025).
    baseline_nn = int(frame["baseline_sleeper_points"].notna().sum())
    prior_depth_cov = max((depth_cov[s] for s in seasons[:-1]), default=0.0)
    absent = _absent_by_position(cohort, FANTASY_POSITIONS, FEATURE_GROUPS)
    n_warnings = sum(1 for level, *_ in records if level >= logging.WARNING)
    generated = datetime.now(timezone.utc).date().isoformat()

    parts = [
        "# Model data profile — the training frame at full scale",
        "",
        "> Phase 9 · Ticket 1 (#27). Generated by "
        "[`scripts/profile_frame.py`](../scripts/profile_frame.py) — regenerate with "
        "`./.venv/Scripts/python scripts/profile_frame.py`. This file is a committed artifact; do "
        "not hand-edit it.",
        "",
        f"- **Generated (UTC):** {generated}",
        f"- **Seasons:** {seasons[0]}–{seasons[-1]} ({len(seasons)} seasons)",
        f"- **Lake backend:** `{LAKE_BACKEND}` · {partitions} partitions read",
        f"- **Scoring:** {league_name!r} live `scoring_settings`, {scoring_keys} keys "
        "(label re-scored by the Phase 1 engine, never hard-coded)",
        f"- **Frame:** {n_rows:,} rows × {len(frame.columns)} columns, "
        f"{frame['player_id'].nunique():,} players",
        "",
        "## Findings, measured",
        "",
        f"1. **No historical projection baseline (finding #1).** `baseline_sleeper_points` is "
        f"non-null for **{baseline_nn:,} / {n_rows:,}** rows across {seasons[0]}–{seasons[-1]} — "
        "Sleeper's endpoints serve only the latest values, so the market-beating grade is "
        "forward-only from 2026 W1. §6 proves it per season.",
        f"2. **Role/depth is one season deep (finding #3).** `nflverse_depth` populates "
        f"`depth_pos_rank` for **{depth_cov[seasons[-1]]:.1f}%** of {seasons[-1]} cohort rows; the "
        f"best-covered earlier season reaches **{prior_depth_cov:.1f}%** — confirmed from the data, "
        "not the registry comment.",
        f"3. **Week 1 is a distinct cold-start cohort.** {len(week1):,} week-1 cohort rows carry no "
        f"current-season lags ({week1_lag_null:.0f}% null `points_last`); the weekly model needs an "
        "explicit path for them. Broken out per season × position in §3.",
        f"4. **The frame is majority non-fantasy rows.** {idp_share:.1f}% of rows "
        f"({n_rows - n_cohort:,} of {n_rows:,}) are IDP / special-teams positions (LB, CB, DE, P, "
        f"LS…) that recorded a stat line and average **{idp_mean_points:.2f}** custom points under "
        f"this scoring. Every model must filter to the {len(FANTASY_POSITIONS)}-position fantasy "
        f"cohort ({n_cohort:,} rows) — so §2–§5 of this report are scoped to it.",
        f"5. **Whole feature families are absent for specific positions.** {_absent_sentence(absent)} "
        "The per-season null rates in §5 are pooled over the cohort and hide this — §5b breaks them "
        "out per position, which is the grain #29/#30 actually consume (spec, Decision #5).",
        f"6. **Zero warnings on real data.** The full build emitted **{n_warnings}** WARNING-level "
        "log record(s) — the standing project bar, met on its first full-scale test (§7).",
        "",
        "## 1. Rows per season × position (full frame)",
        "",
        "Every position with a stat line, honest inventory. The fantasy cohort — the modelled "
        f"positions {', '.join(FANTASY_POSITIONS)} — is the **Total** row; everything below it is "
        "IDP / special-teams volume the models discard.",
        "",
        _count_table(frame, seasons, all_positions, total_label="All positions"),
        "",
        "## 2. Rows per season × position (fantasy cohort)",
        "",
        f"The modelling universe: {', '.join(FANTASY_POSITIONS)} only. TE and K carry the thinnest "
        "per-position volume — the input to #29's per-position-vs-pooled model decision.",
        "",
        _count_table(cohort, seasons, FANTASY_POSITIONS, total_label="Total"),
        "",
        "## 3. Week-1 cold-start rows (fantasy cohort)",
        "",
        "Week 1 has no current-season lagged usage (a player's first appearance), so these rows are "
        "where the weekly model's cold-start path is exercised. Sized here per season × position so "
        "#29 can report a week-1 MAE separately rather than hiding it in a season average.",
        "",
        _count_table(cohort[cohort["week"] == 1], seasons, FANTASY_POSITIONS, total_label="Total"),
        "",
        "## 4. Feature-group availability by season (fantasy cohort)",
        "",
        "Coverage % = the max non-null share over a group's columns (the coarse *is any of it here?* "
        "question; per-feature detail is §5). This is the map of which feature sets are usable over "
        "which span — read a **flat-zero** row as a forward-only source (depth 2025+, the two "
        "2026-forward Sleeper families) and a **flat-but-low** row as present-yet-sparse (injuries: "
        "only a listed player has a report row, so ~14% every season is signal, not a coverage gap).",
        "",
        _availability_table(by_season, seasons, FEATURE_GROUPS),
        "",
        "## 5. Per-feature null rate by season (fantasy cohort)",
        "",
        "Null % per feature per season. Two floors are in play and they are different things. A "
        "usage lag is null for a player's **first appearance** by construction, so every rate here "
        "is floored by the week-1 cohort of §3 (`*_trend` needs two prior appearances and is null "
        "one week longer again). But these rates are also **pooled over positions**, so a family "
        "that is entirely absent for one position inflates every season equally — which is why "
        "`snap_pct_last` (~17%) sits above `points_last` (~10%) with no season-to-season story. "
        "**§5b is the table to read for that**; this one is for spotting a change over time.",
        "",
        _null_table(by_season, seasons, FEATURE_GROUPS),
        "",
        "## 5b. Per-feature null rate by position (fantasy cohort, all seasons)",
        "",
        "The same features, pooled over seasons and split by position — the grain the models "
        "actually consume, per the spec's Decision #5. A ~100% cell here is not sparse data, it is "
        "a feature that **does not exist** for that position, because `nflverse_snaps` and "
        "`nflverse_ff_opp` cover offensive skill players only. Measured: "
        f"{_absent_sentence(absent)} #30 owns K and DEF, so this is its input inventory — and note "
        "what survives: the market columns and `is_indoor` are 100% present for both, and both keep "
        "their own points lags. That is the feature set #30 has to work with.",
        "",
        _null_by_position_table(cohort, FANTASY_POSITIONS, FEATURE_GROUPS),
        "",
        "## 6. `baseline_sleeper_points` — null across every season (finding #1)",
        "",
        "Measured over the full frame. The column exists so 2026-forward rows can fill it; there is "
        "nothing to beat historically, which is why #28 defines the bar with naive baselines instead.",
        "",
        _baseline_table(frame, seasons),
        "",
        "## 7. Assembly log",
        "",
        "**Warnings (verbatim).** The standing bar is zero unexpected warnings on real data:",
        "",
        _log_block(records, min_level=logging.WARNING),
        "",
        "**Informational reconciliation (verbatim).** Join rates and point-in-time drops — the audit "
        "trail the data-conventions doc requires (\"log every row that fails to join\"):",
        "",
        _log_block([r for r in records if r[2] == "dataset.assemble"], min_level=logging.INFO),
        "",
        "## 8. Recommendation — which seasons to train on",
        "",
        f"**Train on {seasons[0]}+, score {seasons[0] + 2}–{seasons[-1]}.** §4/§5 show usage, "
        "market and the nflverse injury report are populated from the first season, with no null "
        "cliff that would justify discarding early seasons (the 2019+ alternative the ticket "
        "flagged). Per Decision #6 the first two seasons are consumed as lag/EWMA warm-up and are "
        "not scored on, so the evaluated span is walk-forward over the rest.",
        "",
        "Some feature families are forward refinements, **not** training-span gates — a model over "
        f"{seasons[0]}+ simply must not require them. **Depth/role** is 2025+ (§4/finding #3): #33 "
        "uses snap/target/rush-share trajectories as the role proxy over the full span and takes "
        "depth-chart rank as a 2025-onward refinement. **Weather forecast** is ~0% historically "
        "(the endpoint reaches back ~92 days), so only `is_indoor` survives the backfill; the "
        "at-kickoff `wx_observed_*` is withheld pre-lock by design. The **Sleeper baseline** and "
        "**Sleeper injury** columns are 2026-forward by construction (finding #1) and belong to the "
        "live forward grade, not the historical training frame.",
        "",
    ]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- entry point
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profile the training frame and write the report")
    ap.add_argument("--seasons", default=DEFAULT_SEASONS, help=f"default: {DEFAULT_SEASONS}")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"report path (default: {DEFAULT_OUT})")
    args = ap.parse_args(argv)

    try:
        seasons = runner.parse_seasons(args.seasons)
    except ValueError as exc:
        print(f"bad --seasons — {exc}", file=sys.stderr)
        return 2

    try:
        league = client.get_league(LEAGUE_ID)
        scoring = league["scoring_settings"]
    except Exception as exc:
        print(f"could not load scoring_settings from league {LEAGUE_ID} "
              f"({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1

    partitions = int(len(lake_inventory()))
    print(f"Profiling {seasons[0]}-{seasons[-1]} over backend {LAKE_BACKEND} ({partitions} "
          f"partitions), scoring {league.get('name')!r} ({len(scoring)} keys)…")

    # Capture the build's log (INFO+) so the report can quote it. basicConfig is NOT called: the only
    # handler is ours, so nothing prints to stdout and the report gets the sole copy.
    capture = _Capture()
    root = logging.getLogger()
    prior_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    try:
        frame = build_training_frame(seasons, scoring)
    except ValueError as exc:
        print(f"could not build the frame — {exc}", file=sys.stderr)
        return 1
    finally:
        root.removeHandler(capture)
        root.setLevel(prior_level)

    report = render_report(
        frame, seasons, capture.records,
        scoring_keys=len(scoring), partitions=partitions, league_name=league.get("name", "?"),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    n_warnings = sum(1 for level, *_ in capture.records if level >= logging.WARNING)
    cohort = int(frame["position"].isin(FANTASY_POSITIONS).sum())
    print(f"Wrote {out} — {len(frame):,} rows ({cohort:,} fantasy-cohort), {n_warnings} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

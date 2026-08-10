"""The **one** projection-source seam: Sleeper by default, our own models when they earn it (#34).

Every downstream surface — the lineup optimizer, the waiver ranker, both simulators, the draft board —
consumed Sleeper's re-scored projections directly until this ticket. Phase 9 built in-house models
(#29 weekly skill, #30 K/DST, #31 draft-value) and graded them; this module is where a person actually
*uses* them, as a **selectable** projection source that defaults to Sleeper and only swaps once a model
clears the live bar in Decision #3.

Two grains, because the call sites have two, and each returns the **exact shape** its Sleeper path
already produced so nothing downstream changes:

* **weekly** — ``weekly_projections`` returns ``dict[player_id -> {proj, pos, team, name}]``, the shape
  :func:`optimizer.inputs.score_projections` produces. Used by the optimizer and the waiver ranker.
* **season** — the season board is composed here and handed back as ``list[PlayerRow]``, the shape
  :func:`projections.board.build_board` produces. Reached through ``build_board``'s own ``source``
  parameter (the two simulators keep calling ``build_board(season, scoring)`` unchanged).

Selectable, not default (Decision #3)
-------------------------------------
The default is **Sleeper**, per position, and it stays Sleeper until a model beats
``baseline_sleeper_points`` on **both** MAE and within-week Spearman ρ over **≥ 4 live 2026 weeks** —
the forward-only grade in :mod:`scripts.eval_swap_gate`, recorded in ``src/model/fit/swap_gate.json``.
:func:`default_source` reads that recorded gate and returns ``MODEL`` only for a position it marks MET,
so today (0 live weeks → every position NOT MET → the artifact absent or all-``met=false``) every
surface stays on Sleeper and **every existing test passes untouched**. Meeting the bar is *supposed* to
change the default (the gate is a check, not a note), and it can only change through a deliberate
regeneration + commit of the artifact — pinned by ``tests/test_workflows.py`` (no cron regenerates it).

An explicit ``source=MODEL`` / ``source=SLEEPER`` forces all positions, bypassing the gate — the opt-in
trial and the Sleeper-forcing diagnostic.

Why the Sleeper default must never touch the lake (the trap that makes a seam unusable)
---------------------------------------------------------------------------------------
A Sleeper projection is one cached HTTP call; the model path needs ``build_training_frame`` over the
lake, which takes minutes. The optimizer is interactive and the draft tool polls every ~3s. So this
module imports **nothing** lake-touching at load: :mod:`model`, :mod:`dataset` and :mod:`store` are
imported **lazily inside the model branch only**, and the Sleeper path does exactly what the call sites
did before. The model path memoises its frame+predictions per ``(grain, season, week)`` so a poll or a
re-run pays the build once. A test (`test_default_path_never_touches_the_lake`) monkeypatches
``build_training_frame`` to raise and asserts the default path still returns.

Mixing sources: forbidden within a week, permitted across positions on the board
--------------------------------------------------------------------------------
The weekly path refuses a **per-row** hybrid — if the market is short, the whole call reverts to Sleeper
rather than scoring some rows on the model and some on imputed means, because a ranking built from two
differently-biased scores orders players by which source covered them as much as by merit. The season
board does exactly what that argument seems to forbid: it merges Sleeper rows and model rows and sorts
them together, and the sort is cross-position (VOR, draft order), which is the surface where comparing
across positions *is* the product. The asymmetry is deliberate, and here is why it is not the same thing:

* A **per-row** split inside one position is arbitrary — two RBs ranked against each other by different
  scorers, with nothing but "was the line posted for his game" deciding which. There is no defensible
  reading of the resulting order.
* A **per-position** split is the thing Decision #3 and Decision #9 item 6 mandate: a position swaps only
  on its own ≥ 4-live-week evidence, so a partially-swapped board is the intended steady state on the way
  to a fully-swapped one. Refusing to compose it would mean refusing to ship the gate's own output.

The cost is real and worth naming: while the board is mixed, cross-position ordering carries whatever
level difference remains between the two sources, so a swapped position can look better or worse than it
is *relative to* an unswapped one. That is bounded by the gate — a position only swaps after beating the
market on both MAE and ρ, so the two are close by construction where it matters — but it is a reason to
prefer swapping positions in a batch once several qualify, rather than one at a time mid-draft-prep.
Within-position ordering, which is what start/sit and waivers actually turn on, is unaffected.

The ensemble composes without flattening its gates
---------------------------------------------------
The model source is three models with their own deferrals: :class:`model.weekly.WeeklyModel` defers
cold-start rows per position, :class:`model.kickdef.KickDefModel` defers per (position × cohort) cell,
:class:`model.season.SeasonModel` defers DEF. The seam calls their ``predict`` and composes the
results, so a deferred cell returns its baseline **through the seam** — no gate is re-implemented or
flattened here.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from projections.board import DRAFTABLE, PlayerRow, build_board
from sleeper import client

_LOG = logging.getLogger(__name__)

#: The two projection sources. Strings (not an enum) so a caller can pass the literal and a recorded
#: artifact can name it without an import.
SLEEPER = "sleeper"
MODEL = "model"

#: The positions the weekly seam reasons about (the model owns exactly these; a Sleeper row of any other
#: position is passed through untouched by the composition).
_WEEKLY_POSITIONS: tuple[str, ...] = ("QB", "RB", "WR", "TE", "K", "DEF")

#: The market features the weekly model's edge rests on. When they are genuinely unavailable for a week
#: (a next-week call on Tuesday, before the line is posted) the model degrades to Sleeper for that call
#: rather than emitting a projection built from silently-null inputs (trap 2 / Decision #9 item 1).
_MARKET_FEATURES: tuple[str, ...] = (
    "implied_team_total",
    "opp_implied_total",
    "team_spread_line",
    "total_line",
)

#: Share of the rows being scored that must carry the **complete** market block before the model path
#: runs. Declared, and reported when it trips. Near-total on purpose — see :func:`_market_available`.
_MARKET_COVERAGE_FLOOR = 0.95

#: The committed forward-swap gate — per-position ``met`` recorded by ``scripts/eval_swap_gate.py``.
DEFAULT_GATE_PATH = Path(__file__).resolve().parents[1] / "model" / "fit" / "swap_gate.json"


# =============================================================== the recorded gate → default source
@functools.cache
def recorded_swap_gate(path: str | Path = DEFAULT_GATE_PATH) -> frozenset[str]:
    """The set of positions the recorded swap gate marks **MET** — the safe, gated default.

    Missing or unreadable artifact → the empty set → every position defaults to Sleeper, never a
    position swapped to the model on an artifact that isn't there. Cached, so the read happens once per
    process (the same pattern as :func:`model.kickdef.recorded_gate`).
    """
    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        positions = payload.get("positions") or {}
        return frozenset(str(p) for p, cell in positions.items() if bool(cell.get("met")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return frozenset()


def default_source(position: str, *, gate_path: str | Path = DEFAULT_GATE_PATH) -> str:
    """``MODEL`` iff the recorded gate marks ``position`` MET (Decision #3), else ``SLEEPER``.

    This is where "selectable, not default" is enforced per position: a position swaps to the model only
    on its own ≥ 4-live-week evidence, so a model win at RB never drags K along (Decision #9 item 6).
    """
    return MODEL if position in recorded_swap_gate(gate_path) else SLEEPER


def resolve_positions_source(
    positions: Iterable[str],
    source: str | None,
    *,
    gate_path: str | Path = DEFAULT_GATE_PATH,
) -> dict[str, str]:
    """Per-position source: an explicit ``source`` forces all; ``None`` consults the recorded gate.

    ``source=MODEL`` / ``source=SLEEPER`` is the manual override (a trial, or forcing Sleeper); ``None``
    is the safe default that reads :func:`default_source` per position.
    """
    positions = tuple(positions)
    if source is not None:
        if source not in (SLEEPER, MODEL):
            raise ValueError(f"unknown projection source {source!r} — use {SLEEPER!r} or {MODEL!r}")
        return {p: source for p in positions}
    return {p: default_source(p, gate_path=gate_path) for p in positions}


# =============================================================== the Sleeper path (lake-free)
def _sleeper_weekly(
    season: int, week: int, scoring: Mapping[str, float], sleeper
) -> dict[str, dict]:
    """This week's Sleeper projections re-scored in our settings — exactly the call sites' old line.

    ``score_projections`` is imported lazily to avoid an import cycle (``optimizer.inputs`` imports this
    module), and nothing lake-touching is imported on this path at all.
    """
    from optimizer.inputs import score_projections

    return score_projections(sleeper.get_projections(season, week), scoring)


# =============================================================== the weekly seam
def weekly_projections(
    season: int,
    week: int,
    scoring: Mapping[str, float],
    *,
    source: str | None = None,
    sleeper=client,
    model_provider: Callable[..., dict[str, dict]] | None = None,
) -> dict[str, dict]:
    """Weekly projections as ``dict[player_id -> {proj, pos, team, name}]`` — Sleeper by default.

    ``source=None`` resolves per position via the recorded gate (Sleeper everywhere today). An explicit
    ``source`` forces it. The model path overlays model-scored rows onto the Sleeper base **only for the
    positions selected for the model**, so Sleeper positions are untouched and a model position that
    degrades (next-week market features missing) falls back to its Sleeper row. ``model_provider`` is
    injectable for offline tests; the default is the real lake-backed one.

    The Sleeper fetch happens on **every** call, including ``source=MODEL``: it is the base the degrade
    falls back to, and without it a short market would leave those positions with no projection at all.
    So there is no pure-model path here by design — the cost is one cached HTTP call, and the alternative
    is a call that can return nothing.
    """
    srcmap = resolve_positions_source(_WEEKLY_POSITIONS, source)
    model_positions = tuple(p for p, s in srcmap.items() if s == MODEL)

    base = _sleeper_weekly(season, week, scoring, sleeper)
    if not model_positions:
        return base  # the pure-Sleeper default — identical to the pre-#34 call sites, no lake touched

    provider = model_provider or _weekly_model_scored
    model_rows = provider(season, week, scoring, model_positions, sleeper=sleeper)
    out = dict(base)
    for pid, row in model_rows.items():
        out[pid] = row  # model row wins for the positions it covers; degrade leaves the Sleeper base
    return out


# =============================================================== the season seam
def compose_season_board(
    season: int,
    scoring: Mapping[str, float],
    *,
    positions: Iterable[str] = DRAFTABLE,
    srcmap: Mapping[str, str],
    adp_key: str = "adp_half_ppr",
    fetch: Callable[..., list[dict]] | None = None,
    sleeper_builder: Callable[..., list[PlayerRow]],
    model_provider: Callable[..., list[PlayerRow]] | None = None,
) -> list[PlayerRow]:
    """Compose one board from Sleeper rows (Sleeper positions) + model rows (model positions), ranked.

    ``sleeper_builder`` is :func:`projections.board.build_board`'s own scoring loop, passed in so this
    module never re-implements it (and no import cycle forms). ``model_provider`` yields the model board
    for the model positions and is injectable for tests. The merged board is re-sorted best-first, the
    same order ``build_board`` returns, so ``draft.vor`` consumes it unchanged.
    """
    positions = tuple(positions)
    sleeper_positions = tuple(p for p in positions if srcmap.get(p, SLEEPER) == SLEEPER)
    model_positions = tuple(p for p in positions if srcmap.get(p) == MODEL)

    rows: list[PlayerRow] = []
    if sleeper_positions:
        rows += sleeper_builder(season, scoring, positions=sleeper_positions, adp_key=adp_key, fetch=fetch)
    if model_positions:
        provider = model_provider or _season_model_board
        rows += provider(season, scoring, model_positions)
    rows.sort(key=lambda p: p.proj_pts, reverse=True)
    return rows


def season_board(
    season: int,
    scoring: Mapping[str, float],
    *,
    positions: Iterable[str] = DRAFTABLE,
    source: str | None = None,
    adp_key: str = "adp_half_ppr",
    fetch: Callable[..., list[dict]] | None = None,
    model_provider: Callable[..., list[PlayerRow]] | None = None,
) -> list[PlayerRow]:
    """The season board as ``list[PlayerRow]`` — Sleeper by default, composed per-position otherwise.

    A thin front door over :func:`compose_season_board` for callers that want the season grain directly;
    ``build_board`` reaches the same composition through its own ``source`` parameter.
    """
    srcmap = resolve_positions_source(positions, source)
    return compose_season_board(
        season, scoring, positions=positions, srcmap=srcmap, adp_key=adp_key, fetch=fetch,
        sleeper_builder=build_board, model_provider=model_provider,
    )


# =============================================================== the model path (lazy, lake-backed)
#: Per-(grain, season, week, scoring) memoisation, so a poll or a re-run pays the frame build once.
_WEEKLY_CACHE: dict[tuple, dict[str, dict]] = {}
_SEASON_CACHE: dict[tuple, list[PlayerRow]] = {}


def _scoring_key(scoring: Mapping[str, float]) -> tuple:
    return tuple(sorted((str(k), float(v)) for k, v in scoring.items()))


def clear_model_cache() -> None:
    """Drop the memoised model frames/predictions (tests, or a forced refresh)."""
    _WEEKLY_CACHE.clear()
    _SEASON_CACHE.clear()


def market_coverage(frame) -> float:
    """The share of ``frame``'s rows carrying the **complete** market block (0.0 when the column is gone).

    Per row, not per column: a row missing any of the four is a row the model would score on imputed
    means, which is the thing the degrade exists to prevent. An "any column has any value" test would
    read a slate with one posted line as fully available and model the other ~399 rows blind.
    """
    import pandas as pd

    present = [c for c in _MARKET_FEATURES if c in frame.columns]
    if len(present) < len(_MARKET_FEATURES) or not len(frame):
        return 0.0
    complete = None
    for col in present:
        ok = pd.to_numeric(frame[col], errors="coerce").notna()
        complete = ok if complete is None else (complete & ok)
    return float(complete.mean())


def _market_available(frame) -> tuple[bool, float]:
    """``(usable, coverage)`` — is the week's market posted for the rows we are about to score?

    The next-week (``week + 1``) call on a Tuesday can land before the lines are posted; the model's whole
    weekly edge is the market block, and a per-row hybrid is not a coherent ranking, so the call degrades
    **whole** to Sleeper and this is logged (trap 2). Any week whose market is short degrades the same way
    — not special-cased to ``+1``.

    The bar is :data:`_MARKET_COVERAGE_FLOOR`, and it is deliberately near-total rather than "some rows
    have it". Lines post per game across a slate, so a genuinely posted week is ~100% covered and a
    part-posted one sits far below; a floor short of that would let the *normal* Tuesday case — a handful
    of games lined, the rest not — through, and score the unlined rows on imputed means. It is not 1.0
    only so that a stray null cannot torpedo a fully-posted week.
    """
    coverage = market_coverage(frame)
    return coverage >= _MARKET_COVERAGE_FLOOR, coverage


def _weekly_model_scored(
    season: int,
    week: int,
    scoring: Mapping[str, float],
    positions: Sequence[str],
    *,
    sleeper=client,
) -> dict[str, dict]:
    """Model-scored weekly rows for ``positions`` — skill via ``WeeklyModel``, K/DST via ``KickDefModel``.

    Trains the shipped models on seasons **strictly before** ``season`` (so a 2026 prediction never
    trains on 2026 — the honest forward setup, and the frame that reproduces the committed artifacts) and
    predicts the requested week. Degrades whole to Sleeper (returns ``{}``, logged at INFO) when the
    week's market features are unavailable. Memoised per ``(season, week, positions, scoring)``.
    """
    import numpy as np
    import pandas as pd

    key = (season, week, tuple(sorted(positions)), _scoring_key(scoring))
    if key in _WEEKLY_CACHE:
        return _WEEKLY_CACHE[key]

    from dataset.assemble import build_training_frame

    # ONE build over the whole span, sliced into train (< season) and predict (this week). The frame is
    # the expensive part — minutes over the lake — and building it per-purpose would pay that two or
    # three times per cache miss, which is exactly the cost asymmetry this module's docstring warns is
    # what makes a seam unusable in practice.
    span = build_training_frame(list(range(2016, season + 1)), scoring)
    seasons_col = pd.to_numeric(span["season"], errors="coerce") if not span.empty else None
    predict = span.iloc[0:0]
    if seasons_col is not None:
        predict = span[
            (seasons_col == season)
            & (pd.to_numeric(span["week"], errors="coerce") == week)
            & span["position"].isin(tuple(positions))
        ].copy()
    if predict.empty:
        _LOG.info("weekly model: no lake rows for %d W%d %s — nothing to score", season, week, tuple(positions))
        _WEEKLY_CACHE[key] = {}
        return {}
    usable, coverage = _market_available(predict)
    if not usable:
        _LOG.info(
            "weekly model: (%d W%d) only %.1f%% of the %d row(s) carry the complete market block %s "
            "(floor %.0f%%) — degrading this call to Sleeper for %s. A next-week call before the lines "
            "are posted; scoring the uncovered rows on imputed means, or mixing them with covered ones, "
            "is not a coherent ranking, so the whole call reverts.",
            season, week, 100 * coverage, len(predict), _MARKET_FEATURES,
            100 * _MARKET_COVERAGE_FLOOR, tuple(positions),
        )
        _WEEKLY_CACHE[key] = {}
        return {}

    skill_train = span[seasons_col < season] if seasons_col is not None else span.iloc[0:0]
    preds = _fit_predict_weekly(
        list(range(2016, season)), predict, scoring, tuple(positions), skill_train=skill_train
    )
    players_map = sleeper.get_players_nfl()
    out: dict[str, dict] = {}
    for idx, pid in zip(predict.index, predict["player_id"].astype("string"), strict=True):
        value = preds.get(idx)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        pos = str(predict.at[idx, "position"])
        out[str(pid)] = {
            "proj": round(float(value), 2),
            "pos": pos,
            "team": str(pid) if pos == "DEF" else _team_of(players_map.get(str(pid))),
            "name": _name_of(str(pid), pos, players_map),
        }
    _WEEKLY_CACHE[key] = out
    return out


def _fit_predict_weekly(train_seasons, predict_frame, scoring, positions, *, skill_train=None):
    """Fit the shipped weekly models on ``train_seasons`` and predict ``predict_frame`` (each own gate).

    Skill (QB/RB/WR/TE) via :class:`model.weekly.WeeklyModel`, K/DST via
    :class:`model.kickdef.KickDefModel` priced with the **live** scoring. Each model predicts only its
    own positions (NaN elsewhere); ``combine_first`` merges them, so their internal deferrals survive
    into the composed prediction unflattened.

    ``skill_train`` is the caller's already-built training frame for ``train_seasons``; it is reused for
    the skill fit **and** handed to ``build_kickdef_frame`` so the K/DST frame does not rebuild it either
    (the component labels still come from the lake). Omitted, both are built here.
    """
    import pandas as pd

    from dataset.assemble import build_training_frame
    from model.kickdef import KICKDEF_POSITIONS, KickDefModel, build_kickdef_frame
    from model.weekly import SKILL_POSITIONS, WeeklyModel

    out = pd.Series(pd.NA, index=predict_frame.index, dtype="Float64")
    want = set(positions)
    if skill_train is None:
        skill_train = build_training_frame(train_seasons, scoring)

    if want & set(SKILL_POSITIONS):
        weekly = WeeklyModel.load_fitted().fit(skill_train)
        out = out.combine_first(pd.to_numeric(weekly.predict(predict_frame), errors="coerce"))
    if want & set(KICKDEF_POSITIONS):
        kd_train = build_kickdef_frame(train_seasons, scoring, weekly=skill_train)
        kd = KickDefModel.load_fitted(scoring=scoring).fit(kd_train)
        out = out.combine_first(pd.to_numeric(kd.predict(predict_frame), errors="coerce"))
    return {idx: (None if pd.isna(v) else float(v)) for idx, v in out.items()}


def _season_model_board(
    season: int, scoring: Mapping[str, float], positions: Sequence[str]
) -> list[PlayerRow]:
    """The draft-value model's board for ``positions`` — :class:`model.season.SeasonModel` fit ≤ S-1.

    SeasonModel carries no committed artifact (it is a cheap closed-form ridge), so it is fit here on the
    strictly-earlier season frame and used to rank ``season``. DEF defers to the baseline inside the
    model (unflattened). Memoised per ``(season, positions, scoring)``.
    """
    import pandas as pd

    key = (season, tuple(sorted(positions)), _scoring_key(scoring))
    if key in _SEASON_CACHE:
        return _SEASON_CACHE[key]

    from model.frame import build_season_frame
    from model.season import SeasonModel, season_value_board
    from sleeper import client as _client

    frame = build_season_frame(list(range(2016, season + 1)), scoring)
    if frame.empty:
        _SEASON_CACHE[key] = []
        return []
    train = frame[pd.to_numeric(frame["season"], errors="coerce") < season]
    model = SeasonModel().fit(train)
    players_map = _client.get_players_nfl()
    names = {
        str(pid): _name_of(str(pid), str(pos), players_map)
        for pid, pos in zip(frame["player_id"].astype("string"), frame["position"].astype("string"), strict=True)
    }
    board = season_value_board(model, frame, season, names=names, positions=tuple(positions))
    _SEASON_CACHE[key] = board
    return board


# =============================================================== small display helpers
def _team_of(meta: Mapping | None) -> str | None:
    return (meta or {}).get("team")


def _name_of(pid: str, pos: str, players_map: Mapping[str, Mapping]) -> str:
    if pos == "DEF":
        return pid
    meta = players_map.get(pid) or {}
    full = meta.get("full_name")
    if full:
        return full
    nm = f"{(meta.get('first_name') or '').strip()} {(meta.get('last_name') or '').strip()}".strip()
    return nm or pid

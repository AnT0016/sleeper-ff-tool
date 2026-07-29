"""Point-in-time capture of Sleeper's own projections and actuals.

These are the captures this whole phase exists for: Sleeper's projection endpoints serve **only the
latest** numbers, so a week's pre-lock projection is unrecoverable the moment the week is played.
``sleeper_proj_week`` / ``sleeper_proj_season`` are therefore forward-only (capture starts 2026 Week
1); ``sleeper_stats_week`` is the finalized-actuals cross-check, and is backfillable.

**Nothing is scored here.** Rows carry the provider's raw stat keys and values verbatim — including
Sleeper's own preset ``pts_half_ppr`` / ``pts_ppr`` / ``pts_std`` — so the lake stays
scoring-agnostic and the assembler (ticket 7) can re-score any snapshot in whatever
``scoring_settings`` the league has at the time. Nothing is filtered on *content*: rows with no game
and an ADP-only stat line are stored as the feed returned them, because "who wasn't projected this
week" is itself data. The one exception is rows without a usable key — they cannot be identified or
deduplicated, so :func:`collect.base.dedupe_rows` drops them and logs it.

Row shape is **flat** — a handful of identity/meta columns, then the raw ``stats`` dict merged in
one key per column. Parquet is columnar and the store's readers project columns
(``read_parquet(key, columns=...)``); a nested ``stats`` struct would defeat that and force every
consumer to unpack. The cost is that a stat a player has no value for reads back as ``NaN`` rather
than an absent key, so re-scoring must drop nulls before summing.

Only *point-in-time* meta is copied. ``team``/``opponent``/``game_id``/``date`` come from the
projection row itself, never from its embedded ``player`` object: that object is the current state
of Sleeper's mutable player master (for 2025 Week 1 rows pulled today, 78 of them disagree — those
players have since changed team), and copying it in would inject tomorrow's truth into a snapshot
whose entire value is that it only knows today's. ``position``/``fantasy_positions`` exist nowhere
else in the payload, so they are taken from ``player`` and are as-of capture time — correct for the
live weekly captures these sources are built for.

.. warning::

   **``position`` is a static label, and on a backfill it is anachronistic.** It comes from the same
   mutable ``player`` master that ``team`` is refused from; the difference is only that no
   point-in-time alternative exists in the payload (the row-level ``team`` is non-null on all 758
   stats rows checked, so refusing the master cost nothing there). ``sleeper_proj_*`` are
   forward-only and captured live, so their ``position`` is genuinely as-of capture. Only a
   **backfilled ``sleeper_stats_week``** is exposed: a 2018 row gets the player's *today* position.

   Measured on 2018 (W1/W8/W15, 689 players): 617 join to nflverse via the gsis crosswalk and 15
   disagree (2.4%), mostly FB/RB taxonomy noise between providers. Note that **nflverse is not the
   fix** — its weekly ``position`` is a static per-player attribute too (of 563 players appearing in
   2018, 2019, 2021 and 2023, *zero* have a position that varies by season), so sourcing it there
   just relabels the same contamination. Real conversions are invisible in both: Taysom Hill (QB in
   2018) and Cordarrelle Patterson (WR in 2018) read back as TE and RB respectively.

   **``fantasy_positions`` is not a mitigation.** It comes from the same mutable master and collapses
   the same way. Sleeper carried Hill as QB/TE-eligible while he was starting at QB, but the master
   returns ``['TE']`` today — and so does the embedded ``player`` object on a **2018** stats row.
   Multi-eligibility that existed at the time is therefore unrecoverable from this field: it records
   what Sleeper believes *now*, for every season you ask about.

   Impact by consumer: the label ``y_custom_points`` is **unaffected** (the scoring engine sums
   stat × weight and never reads position); positional replacement level and the sims' per-position
   CVs take a small mislabel rate; DST/K are immune (DEF rows key on the team abbreviation). It bites
   the **breakout classifier**, where a WR→RB conversion *is* the role step-up being predicted — a
   current position leaks the label rather than blurring a feature.

   The fix belongs downstream, not here: ticket 5's ``_backfill=True`` marker makes these rows
   identifiable (necessary, not sufficient), and ticket 7 should resolve position from
   ``nflverse_depth`` — the one registered source that is genuinely as-of a date (``dt`` is in its
   key, ``pos_abb`` is the value) — falling back to this static label under an explicit
   ``position_is_static`` flag.

Sleeper's **injury fields** (``injury_status`` / ``injury_body_part`` / ``injury_start_date``) come
from that same ``player`` master, but are lifted onto the **forward-only sources only** — and that
restriction is the whole point. They are captured because Sleeper's ``injury_status`` is this
project's *authoritative* injury signal (``optimizer.inputs``: "authoritative per CLAUDE.md"; the
nflverse report is the secondary cross-check), because it needs no crosswalk (Sleeper-keyed, where
``nflverse_injuries`` is ``gsis_id``-keyed), and because it carries roster states the official
practice report has no equivalent for — a live master shows ``PUP``/``NA``/``Sus``/``IR`` alongside
``Questionable``. On a live pre-lock capture it is exactly "what we believed before kickoff", and it
is unrecoverable afterwards.

``sleeper_stats_week`` deliberately does **not** get them. It is backfillable, and a backfilled 2018
row would carry *today's* status — a sharper error than the stale-``position`` one above: position is
stable, so a stale label is merely imprecise, whereas injury status is wildly time-varying, so a 2018
row reading ``Questionable`` because the player is questionable *now* is simply false.

(``player.metadata`` carries occasional ``injury_override_<type>_<season>_<week>`` keys that look like
a per-week injury history. They are not one — across the entire master there are 279 such keys on 247
players, 170 of them from 2020 alone. A sparse legacy artifact, not a backfill source; checked so it
need not be re-investigated.)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from sleeper import client
from sleeper.client import DEFAULT_POSITIONS

from .base import Collected, dedupe_rows
from .registry import SOURCES

_LOG = logging.getLogger(__name__)

#: Top-level row fields kept verbatim next to the stats. ``season``/``week`` are deliberately
#: absent: the store stamps ``_season``/``_week`` on every row, and ``sport``/``category``/
#: ``week_shard`` are constants or Sleeper-internal. ``last_modified`` is the provider's revision —
#: paired with ``_captured_at`` it tells you how stale a projection was when we captured it.
_META: tuple[str, ...] = (
    "team",
    "opponent",
    "game_id",
    "date",
    "season_type",
    "company",
    "last_modified",
)

#: Injury fields lifted from the embedded ``player`` master. Sleeper's ``injury_status`` is the
#: project's authoritative injury signal and needs no crosswalk; it also covers roster states the
#: official practice report lacks (PUP/NA/Sus/IR).
_INJURY_FIELDS: tuple[str, ...] = ("injury_status", "injury_body_part", "injury_start_date")

#: Sources whose captures are live and pre-lock, so ``player``'s *mutable* injury fields are
#: genuinely as-of capture time. ``sleeper_stats_week`` is excluded on purpose: it is backfillable,
#: and a backfilled row would carry today's status rather than that week's (see the module docstring).
_INJURY_SOURCES: frozenset[str] = frozenset({"sleeper_proj_week", "sleeper_proj_season"})


def _player_id(raw: Mapping[str, Any]) -> str | None:
    """The row's key. Coerced to ``str`` (DEF rows key on a team abbreviation, e.g. ``"PHI"``)."""
    pid = raw.get("player_id")
    if pid is None:
        return None
    return str(pid).strip() or None


def _fantasy_positions(player: Mapping[str, Any]) -> str | None:
    """``fantasy_positions`` as a comma-joined string.

    A list column would be the faithful shape, but it makes the partition a nested parquet type for
    one rarely-read field; the joined string keeps every lake column flat and greppable. Kept at all
    because it explains rows the position filter otherwise seems to contradict — a FB with
    ``fantasy_positions=["RB"]`` is returned by a ``position[]=RB`` request.

    Same mutable-master caveat as ``position`` (see the module ``.. warning::``): this is today's
    eligibility for every season you ask about, and it has already collapsed for the players where
    multi-eligibility mattered most. Do not read it as "what he was eligible at that week".
    """
    values = player.get("fantasy_positions")
    if not isinstance(values, (list, tuple)) or not values:
        return None
    return ",".join(str(v) for v in values)


def _flatten(raw: Mapping[str, Any], *, source: str, key_cols: Sequence[str]) -> dict[str, Any]:
    """One raw Sleeper row -> one flat lake row (meta + verbatim stats).

    Merging stats over meta means a stat named like a meta column wins — the promise is that stats
    pass through unchanged. The **key** is the one exception: identity is inviolable.
    """
    player = raw.get("player") or {}
    stats = dict(raw.get("stats") or {})
    row: dict[str, Any] = {
        "player_id": _player_id(raw),
        "position": player.get("position"),
        "fantasy_positions": _fantasy_positions(player),
    }
    if source in _INJURY_SOURCES:
        row.update({name: player.get(name) for name in _INJURY_FIELDS})
    row.update({name: raw.get(name) for name in _META})

    # A stat shadowing a KEY column would replace the row's identity (and undo _player_id's str
    # coercion), after which the store dedups on a garbage key and files the row as someone else --
    # the silent-corruption family this phase exists to avoid. So the key wins and the stat is
    # dropped, loudly. Deliberately not an exception: raising would abandon the whole capture, and
    # for the forward-only sources that loss is permanent, which is strictly worse than one lost
    # column. Never observed; Sleeper's stat keys are stat-shaped.
    stolen = sorted(set(key_cols) & set(stats))
    if stolen:
        for name in stolen:
            stats.pop(name)
        _LOG.error(
            "%s: stat key(s) %s collide with the row key %s - dropped the stat, kept the key. "
            "The provider's schema has changed; this needs a look.",
            source, stolen, list(key_cols),
        )

    clashing = sorted(set(row) & set(stats))
    if clashing:
        _LOG.warning(
            "%s: stat key(s) %s shadow meta columns; keeping the raw stat", source, clashing
        )
    row.update(stats)
    return row


def _freshness(row: Mapping[str, Any]) -> int:
    """Provider revision stamp (epoch ms), used to pick a winner if a player is listed twice."""
    value = row.get("last_modified")
    return int(value) if isinstance(value, (int, float)) else 0


def _stat_signature(raw: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """A hashable, order-independent view of a row's raw ``stats`` payload.

    The signature is the **stats alone** — never the meta. Two rows sharing a key *and* this
    signature are the same observation the provider listed twice, not a stat correction, which is the
    exact split :func:`_collapse_repeated_games` turns on.
    """
    stats = raw.get("stats") or {}
    return tuple(sorted((str(k), stats[k]) for k in stats))


def _collapse_repeated_games(
    raw_rows: Iterable[Mapping[str, Any]],
    key_cols: Sequence[str],
    *,
    source: str,
    freshness: Callable[[Mapping[str, Any]], Any],
) -> list[dict]:
    """Collapse rows the provider listed more than once with a **byte-identical stat payload**.

    In 2016 ``sleeper_stats_week`` emits one whole game twice — every player of both teams (incl. the
    DST) under two game_ids ``2016<1><WW>00`` and ``...29`` (8 weeks; zero occurrences 2017+). Across
    a pair, ``game_id`` differs on **every** row, ``week_shard`` on most, and the **stats on none**;
    ``last_modified`` is null throughout, so this is an id-assignment artifact, not a stat correction
    (see #21). Letting ``dedupe_rows`` collapse them with its WARNING would fire a defect warning on
    every backfill — precisely how an operator learns to skim the one that matters.

    Only repeats whose **stats agree** collapse, at INFO; a genuine stat conflict is left in place for
    :func:`collect.base.dedupe_rows` to warn about (that is the loud case — "the provider disagrees
    with itself about a player's stat line"). This mirrors ``collect.nflverse._identified`` /
    ``_latest_revision``: a grain reduction with an INFO count, done before the store's dedup can
    mistake it for a defect. And like ``_latest_revision``, it is invoked from **one** collector
    (``collect_stats_week``), not the shared ``_collect``: the "benign" evidence is specific to this
    backfillable source, and a first-ever duplicate on the forward-only projection feeds — whose
    captures are unrecoverable — must stay a WARNING.

    The winner per ``(key, stats)`` is the **last** listing (keep-last on a freshness tie), matching
    ``dedupe_rows`` (``base.py`` ``rank(row) >= kept_rank``). The pair's order is mixed within a week,
    so no tie-break yields a consistent index; matching ``dedupe_rows`` instead makes this change
    provably output-identical to the plain dedup — the whole safety claim of #21 (game_id *is*
    retained in the collected row, so an arbitrary flip would land in the lake). Rows whose key is not
    usable pass through untouched, so ``dedupe_rows`` still drops and counts them.
    """
    kept: dict[tuple[Any, ...], dict] = {}       # (key, stat-signature) -> winning raw row
    kept_rank: dict[tuple[Any, ...], Any] = {}
    first_game_id: dict[tuple[Any, ...], Any] = {}
    unusable: list[dict] = []                     # null/blank key -> dedupe_rows drops + warns it
    collapsed = 0
    keys_hit: set[Any] = set()
    pairs: set[tuple[str, str]] = set()

    for raw in raw_rows:
        row = dict(raw)
        pid = _player_id(row)
        if pid is None:
            unusable.append(row)
            continue
        token = (pid, _stat_signature(row))
        game_id = row.get("game_id")
        if token in kept:
            collapsed += 1
            keys_hit.add(pid)
            if first_game_id[token] != game_id:
                pairs.add(tuple(sorted((str(first_game_id[token]), str(game_id)))))
            # keep-last on a freshness tie, exactly as dedupe_rows does (base.py).
            if freshness(row) >= kept_rank[token]:
                kept[token], kept_rank[token] = row, freshness(row)
            continue
        kept[token], kept_rank[token], first_game_id[token] = row, freshness(row), game_id

    if collapsed:
        total = len(kept) + collapsed + len(unusable)
        _LOG.info(
            "%s: collapsed %d/%d row(s) (%.2f%%) the provider listed twice with identical stats "
            "under game_id pair(s) %s - an id-assignment artifact (see #21), not a stat correction; "
            "kept the last listing per %s",
            source, collapsed, total, collapsed / total * 100, sorted(pairs), list(key_cols),
        )
    return list(kept.values()) + unusable


def _collect(
    source: str,
    season: int,
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    week: int | None = None,
) -> Collected:
    key_cols = SOURCES[source].key_cols
    rows = dedupe_rows(
        (_flatten(r, source=source, key_cols=key_cols) for r in raw_rows),
        key_cols,
        source=source,
        freshness=_freshness,
    )
    return Collected.for_source(source, season, rows, week=week)


def collect_proj_week(
    season: int,
    week: int,
    *,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    fetch: Callable[..., list[dict]] | None = None,
) -> Collected:
    """``sleeper_proj_week``: the pre-lock weekly projection snapshot (forward-only).

    ``fetch`` defaults to ``sleeper.client.get_projections`` and is injectable for offline tests.
    ``season_type`` is fixed to the client default (``regular``): a postseason capture would land in
    the same week partition and collide with the regular-season week of the same number.
    """
    get = fetch or client.get_projections
    return _collect(
        "sleeper_proj_week", season, get(season, week, positions=positions), week=int(week)
    )


def collect_proj_season(
    season: int,
    *,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    fetch: Callable[..., list[dict]] | None = None,
) -> Collected:
    """``sleeper_proj_season``: full-season projections (forward-only).

    Captured on the pre-lock cadence like the weekly one: the *drift* of a season projection across
    captures is the signal (role changes, injuries), which is why a later-day capture is retained
    beside the earlier one rather than overwriting it.
    """
    get = fetch or client.get_season_projections
    return _collect("sleeper_proj_season", season, get(season, positions=positions), week=None)


def collect_stats_week(
    season: int,
    week: int,
    *,
    positions: Iterable[str] = DEFAULT_POSITIONS,
    fetch: Callable[..., list[dict]] | None = None,
) -> Collected:
    """``sleeper_stats_week``: finalized weekly actuals (post-game cadence, backfillable).

    Sleeper-keyed, so it is the cross-check for DST and K — where nflverse's player-level feed has
    no team-defense aggregate and no FG-distance buckets.

    The one collector that pre-collapses identical-stat repeats (:func:`_collapse_repeated_games`):
    the 2016 feed lists one whole game twice, and without this a backfill warns every run (#21).
    """
    get = fetch or client.get_stats
    source = "sleeper_stats_week"
    raw_rows = _collapse_repeated_games(
        get(season, week, positions=positions),
        SOURCES[source].key_cols,
        source=source,
        freshness=_freshness,
    )
    return _collect(source, season, raw_rows, week=int(week))

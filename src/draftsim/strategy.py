"""My roster-construction strategies — the variable the simulator compares.

A strategy is a *position-preference policy* over draft rounds. On each of my picks the engine asks
the strategy which positions it wants this round, then takes the best **custom-VOR** available player
among them (VOR, never market ADP — I draft in our scoring). Two shared guardrails keep every
strategy honest:

* a **mandatory guard** (:func:`forced_positions`) that, when my remaining picks run down to exactly
  what a legal starting lineup still requires, forces the shortfall positions — so no build ever ends
  without a QB / K / DEF or enough RB/WR/TE/FLEX bodies; and
* **K/DEF are never offered early** — no strategy lists them, so they're only ever taken by the
  end-of-draft mandatory guard, matching CLAUDE.md's "stream K/DEF, don't reach" rule.

The builds: ``best_vor`` (pure best-available by VOR), ``rb_early`` (anchor the backfield),
``hero_rb`` (one stud RB then receivers), ``zero_rb`` (no RB until the position falls to value),
``balanced`` (fill empty starting slots first).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from draft.roster import RosterConfig

#: Skill positions a strategy may ever request (K/DEF are handled only by the mandatory guard).
CORE: tuple[str, ...] = ("RB", "WR", "TE", "QB")

#: Most I'll roster at each position outside the mandatory guard.
MY_MAX_PER_POS: dict[str, int] = {"QB": 2, "RB": 7, "WR": 8, "TE": 2, "K": 1, "DEF": 1}

#: Positions where a *backup* adds no real value in this league — you stream the position instead of
#: rostering a second body. So I only ever draft these up to their starting slot count, and spend the
#: pick on RB/WR depth instead (which is what bye-weeks and injuries actually need).
STREAM_NOT_STASH: frozenset[str] = frozenset({"QB", "TE"})


def position_has_room(pos: str, counts: Mapping[str, int], cfg: RosterConfig) -> bool:
    """Should I still draft another ``pos`` (outside the mandatory end-of-draft guard)?

    Capped at the starting slot count for stream-not-stash positions (QB/TE), else at the generous
    depth limit in :data:`MY_MAX_PER_POS`.
    """
    have = counts.get(pos, 0)
    if pos in STREAM_NOT_STASH:
        return have < cfg.slots.get(pos, 0)
    return have < MY_MAX_PER_POS.get(pos, 99)


def _best_vor(rnd: int, counts: Mapping[str, int], cfg: RosterConfig) -> list[str]:
    return ["RB", "WR", "TE", "QB"]


def _rb_early(rnd: int, counts: Mapping[str, int], cfg: RosterConfig) -> list[str]:
    if rnd <= 2:
        return ["RB"]
    if rnd == 3:
        return ["RB", "WR", "TE"]
    return ["RB", "WR", "TE", "QB"]


def _hero_rb(rnd: int, counts: Mapping[str, int], cfg: RosterConfig) -> list[str]:
    if rnd == 1:
        return ["RB"]
    if rnd <= 5:
        return ["WR", "TE", "QB"]  # one RB anchor, then load receivers
    return ["RB", "WR", "TE", "QB"]


def _zero_rb(rnd: int, counts: Mapping[str, int], cfg: RosterConfig) -> list[str]:
    if rnd <= 4:
        return ["WR", "TE", "QB"]  # no RB until they've slid to value
    return ["RB", "WR", "TE", "QB"]


def _balanced(rnd: int, counts: Mapping[str, int], cfg: RosterConfig) -> list[str]:
    """Fill any still-empty dedicated starting slot first, else best-available core."""
    s = cfg.slots
    need = [p for p in ("RB", "WR", "QB", "TE") if counts.get(p, 0) < s.get(p, 0)]
    return need or ["RB", "WR", "TE", "QB"]


#: name -> preference policy. Order here is the report's display order.
STRATEGIES: dict[str, Callable[[int, Mapping[str, int], RosterConfig], list[str]]] = {
    "best_vor": _best_vor,
    "balanced": _balanced,
    "rb_early": _rb_early,
    "hero_rb": _hero_rb,
    "zero_rb": _zero_rb,
}


def _mandatory(counts: Mapping[str, int], cfg: RosterConfig) -> tuple[set[str], int]:
    """Positions still short of a legal starting lineup, and the *minimum* picks to complete it.

    Covers the singleton slots (QB/K/DEF), the RB/WR/TE starting minimums, and the one extra
    flex-eligible body the FLEX slot needs beyond those minimums.
    """
    s = cfg.slots
    short: set[str] = set()
    total = 0
    for pos in ("QB", "K", "DEF"):
        need = max(0, s.get(pos, 0) - counts.get(pos, 0))
        if need:
            short.add(pos)
            total += need

    rb_need = max(0, s.get("RB", 0) - counts.get("RB", 0))
    wr_need = max(0, s.get("WR", 0) - counts.get("WR", 0))
    te_need = max(0, s.get("TE", 0) - counts.get("TE", 0))
    for pos, need in (("RB", rb_need), ("WR", wr_need), ("TE", te_need)):
        if need:
            short.add(pos)
    total += rb_need + wr_need + te_need

    flex_total = s.get("RB", 0) + s.get("WR", 0) + s.get("TE", 0) + s.get("FLEX", 0)
    have_flex = counts.get("RB", 0) + counts.get("WR", 0) + counts.get("TE", 0)
    extra = max(0, flex_total - have_flex)  # total flex-eligible starters still owed
    extra_beyond = max(0, extra - (rb_need + wr_need + te_need))  # the FLEX body past the minimums
    if extra_beyond:
        short |= {"RB", "WR", "TE"}
        total += extra_beyond
    return short, total


def forced_positions(
    counts: Mapping[str, int], cfg: RosterConfig, picks_left: int
) -> set[str] | None:
    """Positions I'm *forced* to draft from now, or ``None`` if there's still slack.

    ``picks_left`` is my remaining picks *including* the current one. When it drops to exactly the
    minimum required to finish a legal lineup, the picker must take from the returned set; each such
    pick reduces the requirement by one, so the lineup is always completable.
    """
    short, total = _mandatory(counts, cfg)
    if picks_left > total:
        return None
    return short

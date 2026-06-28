"""Format the Monte Carlo results into a plain-text report.

Three sections, each preceded by the explicit assumptions block (so the numbers can be judged):

1. **Roster-build comparison** — per strategy, the *distribution* of my starting-lineup season points
   (p10 / median / p90) and my finish among the league (mean rank, P(top-3), P(win)). The
   recommendation is by best finish distribution, not by best single expected value.
2. **Target survival** — for each of my pick numbers, the top players by VOR and the probability each
   is still on the board when I'm up (the joewlos "will my target survive?" question).
3. **Injury insight** — for the recommended build, which of my likely starters carry real durability
   risk *and* lack a rostered backup — i.e. where you need a true handcuff vs. a weekly streamer.
"""

from __future__ import annotations

import numpy as np

from .bots import ADP_NOISE, BOT_MAX_PER_POS, LATE_ROUND_FRACTION
from .distributions import INJURY_RISK, POSITION_CV, SEASON_GAMES
from .engine import SimOutput, StrategyResult, representative_draft
from .lineup import select_starters

# Positions that are realistically streamable week-to-week, so a missing rostered backup hurts less.
_STREAMABLE = frozenset({"QB", "K", "DEF", "TE"})
_HIGH_RISK = 0.30


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def assumptions_block(out: SimOutput, slot_source: str) -> str:
    cv = " ".join(f"{p} {POSITION_CV[p]:.2f}" for p in ("QB", "RB", "WR", "TE", "K", "DEF"))
    inj = " ".join(
        f"{p} {INJURY_RISK[p][0]:.0%}/{INJURY_RISK[p][1]:.0f}g"
        for p in ("QB", "RB", "WR", "TE", "K", "DEF")
    )
    caps = " ".join(f"{p}≤{n}" for p, n in BOT_MAX_PER_POS.items())
    late = int(out.cfg.rounds * LATE_ROUND_FRACTION)
    lines = [
        "ASSUMPTIONS (directional & heuristic — NOT fitted to data; tune in src/draftsim/):",
        "  • Outcomes: season points ~ lognormal, mean = our custom-scored projection.",
        f"      CV (std/mean) by pos: {cv}",
        f"  • Injuries: P(significant setback)/season & mean games missed (of {SEASON_GAMES}):",
        f"      {inj}  → availability haircut on the season total.",
        f"  • Opponents: 11 bots draft by MARKET half-PPR ADP + N(0,{ADP_NOISE:.0f}) picks; "
        "I draft by our CUSTOM VOR.",
        f"      bot roster caps: {caps}; K/DEF only from round {late}+.",
        "  • Decisions use projections/ADP (ex-ante); evaluation uses sampled outcomes (ex-post).",
        f"  • Pool: {out.pool.n} draftable players (top by ADP + top by VOR + all K/DEF).",
        f"  • Slot {out.my_slot} ({slot_source}) · {out.cfg.teams} teams · {out.cfg.rounds} rounds "
        f"· {out.n_sims} sims · seed {out.seed}.",
    ]
    return "\n".join(lines)


def _recommend(results: dict[str, StrategyResult]) -> str:
    """Best build by mean finish rank (lower = better), tie-broken by higher median points."""
    def key(name: str):
        r = results[name]
        return (float(np.mean(r.my_rank)), -float(np.median(r.my_points)))

    return min(results, key=key)


def recommended_strategy(out: SimOutput) -> str:
    """Public accessor for the recommended build (best finish distribution)."""
    return _recommend(out.results)


def _short(name: str) -> str:
    """Compact label for the draftboard grid: last name (or DEF team abbr), trimmed."""
    return name.split()[-1][:9] if name else "—"


def board_grid(out: SimOutput, name: str | None = None) -> str:
    """A representative simulated draft as a round×slot text grid (Sleeper-style).

    Columns are draft slots (your slot marked ``*`` and your picks in ``[brackets]``); rows are
    rounds. The draft shown is the median-outcome sim for the recommended (or given) build.
    """
    name = name or recommended_strategy(out)
    rosters = representative_draft(out, name)
    pool = out.pool
    teams, rounds = out.cfg.teams, out.cfg.rounds
    my_team = out.my_slot - 1
    w = 16

    head = "     " + "".join(
        f"{'S' + str(t + 1) + ('*' if t == my_team else ''):<{w}}" for t in range(teams)
    )
    lines = [
        f"--- Representative simulated draft (build: {name}, median outcome) ---",
        f"  round × slot; your column (S{out.my_slot}) marked * and your picks in [brackets]",
        head,
    ]
    for rnd in range(1, rounds + 1):
        row = f"R{rnd:>2}  "
        for t in range(teams):
            idx = rosters[t][rnd - 1]
            cell = f"{_short(pool.names[idx])} {pool.pos[idx]}"
            if t == my_team:
                cell = f"[{cell}]"
            row += f"{cell:<{w}}"
        lines.append(row)
    return "\n".join(lines)


def build_comparison(out: SimOutput, recommended: str) -> str:
    rows = []
    header = (
        f"  {'strategy':<10} {'p10':>7} {'median':>7} {'p90':>7} "
        f"{'mean-rk':>7} {'P(top3)':>7} {'P(win)':>6}  typical build"
    )
    for name, r in out.results.items():
        p10, p50, p90 = np.percentile(r.my_points, [10, 50, 90])
        mean_rank = float(np.mean(r.my_rank))
        p_top3 = float(np.mean(r.my_rank <= 3))
        p_win = float(np.mean(r.my_rank == 1))
        build = r.builds.most_common(1)[0][0] if r.builds else "—"
        mark = " ←" if name == recommended else ""
        rows.append(
            f"  {name:<10} {p10:7.0f} {p50:7.0f} {p90:7.0f} "
            f"{mean_rank:7.2f} {_pct(p_top3):>7} {_pct(p_win):>6}  {build}{mark}"
        )
    note = (
        f"→ Recommended: {recommended} — best finish distribution in OUR scoring "
        f"(mean rank among {out.cfg.teams}). 'typical build' = most common composition."
    )
    return "\n".join(
        ["--- Roster-build comparison (my starting-lineup season points & finish) ---", header, *rows,
         "", note]
    )


def survival_table(out: SimOutput, *, reference: str | None = None, top: int = 8) -> str:
    """Top targets by VOR likely available at each of my picks, from a reference strategy's draft."""
    ref = reference if reference in out.results else next(iter(out.results))
    r = out.results[ref]
    pool = out.pool
    lines = [
        f"--- Targets likely available at my picks (reference build: {ref}) ---",
        "  (P = chance the player is still on the board when I'm on the clock)",
    ]
    vor_order = pool.vor_order
    for k, pick_no in enumerate(r.my_picks):
        probs = r.survival[k]
        # walk the board best-VOR first, list the first `top` with a meaningful chance left
        shown = []
        for idx in vor_order:
            if pool.pos[idx] in ("K", "DEF"):
                continue
            p = probs[idx]
            if p <= 0.01:
                continue
            shown.append(f"{pool.names[idx]} ({pool.pos[idx]}) {_pct(p)}")
            if len(shown) >= top:
                break
        lines.append(f"  pick #{pick_no:>3}:  " + " · ".join(shown))
    return "\n".join(lines)


def _likely_roster(result: StrategyResult, pool, rounds: int) -> list[int]:
    """My modal roster for a strategy: the ``rounds`` players I draft most often."""
    order = np.argsort(-result.draft_counts, kind="stable")
    return [int(i) for i in order[:rounds] if result.draft_counts[i] > 0]


def injury_insight(out: SimOutput, recommended: str) -> str:
    r = out.results[recommended]
    pool = out.pool
    roster = _likely_roster(r, pool, out.cfg.rounds)
    positions = [pool.pos[i] for i in roster]
    means = [pool.mean[i] for i in roster]
    starter_local = set(select_starters(positions, means, out.cfg.slots))
    bench_pos = [positions[i] for i in range(len(roster)) if i not in starter_local]

    lines = [
        f"--- Injury insight (build: {recommended}) — where you need a real backup vs a streamer ---",
        f"  {'starter':<24} {'pos':<4} {'P(setback)':>10}  backup?  note",
    ]
    for i in sorted(starter_local, key=lambda i: -means[i]):
        idx = roster[i]
        pos = positions[i]
        if pos in ("K", "DEF"):
            continue
        risk = INJURY_RISK.get(pos, (0.3, 3.0))[0]
        has_backup = pos in bench_pos
        if has_backup:
            bench_pos.remove(pos)  # one backup covers one starter
            note = "covered (rostered backup)"
            flag = "yes"
        elif risk >= _HIGH_RISK and pos not in _STREAMABLE:
            note = f"⚠ DRAFT/HOLD A BACKUP — {pos} is hard to stream"
            flag = "NO"
        elif risk >= _HIGH_RISK:
            note = f"thin, but {pos} is stream-able weekly"
            flag = "no"
        else:
            note = "low durability risk"
            flag = "no"
        lines.append(f"  {pool.names[idx]:<24} {pos:<4} {_pct(risk):>10}  {flag:<7}  {note}")
    return "\n".join(lines)


def format_report(out: SimOutput, slot_source: str, season: int) -> str:
    recommended = _recommend(out.results)
    parts = [
        f"=== Monte Carlo draft simulator — {season} ===",
        "",
        assumptions_block(out, slot_source),
        "",
        build_comparison(out, recommended),
        "",
        survival_table(out, reference="best_vor"),
        "",
        injury_insight(out, recommended),
        "",
        "Directional only — re-run with --slot once your draft slot is revealed, and treat builds as "
        "a starting point, not a script.",
    ]
    return "\n".join(parts)

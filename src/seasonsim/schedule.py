"""Head-to-head schedule (real or generated) and the playoff bracket resolver.

Regular-season pairings come from the league's real matchups when they exist (a completed season, or a
2026 league once its schedule is posted); otherwise a round-robin stands in. The playoff bracket is
the league's locked format — **6 teams, Weeks 15-17, default seeding, teams stay on their bracket
side** (top two seeds get a first-round bye). Everything here works in *team indices* (0-based, one per
roster) so the engine never has to carry roster-id bookkeeping into its hot loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def schedule_from_matchups(
    sleeper, league_id: str, weeks: Sequence[int], roster_ids: Sequence[int]
) -> dict[int, list[tuple[int, int]]]:
    """Real per-week ``(team_i, team_j)`` pairings from Sleeper matchups, keyed by team index.

    ``roster_ids`` fixes the team-index order. Rows sharing a ``matchup_id`` in a week are opponents;
    weeks with no complete pairings (unplayed / not yet scheduled) are omitted.
    """
    idx = {int(r): i for i, r in enumerate(roster_ids)}
    schedule: dict[int, list[tuple[int, int]]] = {}
    for w in weeks:
        try:
            rows = sleeper.get_matchups(league_id, w)
        except Exception:
            continue
        by_mid: dict[int, list[int]] = {}
        for row in rows or []:
            mid = row.get("matchup_id")
            rid = row.get("roster_id")
            if mid is None or rid is None or int(rid) not in idx:
                continue
            by_mid.setdefault(int(mid), []).append(idx[int(rid)])
        pairs = [(a, b) for a, b in (tuple(v) for v in by_mid.values() if len(v) == 2)]
        if pairs:
            schedule[int(w)] = pairs
    return schedule


def round_robin(n_teams: int, weeks: Sequence[int]) -> dict[int, list[tuple[int, int]]]:
    """A circle-method round-robin over ``n_teams`` (even), cycling to cover every week in ``weeks``."""
    if n_teams % 2:
        raise ValueError("round_robin needs an even team count")
    rounds: list[list[tuple[int, int]]] = []
    order = list(range(n_teams))
    for _ in range(n_teams - 1):
        pairs = [(order[i], order[n_teams - 1 - i]) for i in range(n_teams // 2)]
        rounds.append(pairs)
        order = [order[0]] + [order[-1]] + order[1:-1]  # rotate all but the first
    return {int(w): rounds[k % len(rounds)] for k, w in enumerate(weeks)}


def seed_order(wins: np.ndarray, points: np.ndarray, n_playoff_teams: int) -> list[int]:
    """Playoff seeds (team indices, best first): by wins, then total points-for as the tiebreak."""
    ranked = sorted(range(len(wins)), key=lambda t: (wins[t], points[t]), reverse=True)
    return ranked[:n_playoff_teams]


def playoff_champion(seeds: Sequence[int], pw_scores: np.ndarray) -> int:
    """Champion (team index) for one sim's playoffs.

    ``seeds`` are team indices best-first; ``pw_scores[t, r]`` is team ``t``'s score in playoff round
    ``r`` (column 0 = first playoff week). Ties within a game go to the better seed. Implements the
    locked 6-team bracket (top two byes; 3v6 & 4v5 in round 1; 1 stays on the 4/5 side, 2 on the 3/6
    side); other sizes fall back to a generic reseeding single-elimination.
    """
    seed_rank = {t: i for i, t in enumerate(seeds)}

    def winner(a: int, b: int, rnd: int) -> int:
        sa, sb = pw_scores[a, rnd], pw_scores[b, rnd]
        if sa > sb:
            return a
        if sb > sa:
            return b
        return a if seed_rank[a] < seed_rank[b] else b  # tie → higher seed

    if len(seeds) == 6:
        s = seeds
        w_a = winner(s[2], s[5], 0)  # 3 vs 6
        w_b = winner(s[3], s[4], 0)  # 4 vs 5
        semi1 = winner(s[0], w_b, 1)  # 1 vs (4/5) — 1's bracket side
        semi2 = winner(s[1], w_a, 1)  # 2 vs (3/6) — 2's bracket side
        return winner(semi1, semi2, 2)

    # Generic fallback: reseed each round, best vs worst, higher seed byes an odd team out.
    alive = list(seeds)
    rnd = 0
    while len(alive) > 1:
        alive.sort(key=lambda t: seed_rank[t])
        nxt: list[int] = []
        i, j = 0, len(alive) - 1
        while i < j:
            nxt.append(winner(alive[i], alive[j], rnd))
            i += 1
            j -= 1
        if i == j:  # odd one out gets a bye
            nxt.append(alive[i])
        alive = nxt
        rnd += 1
    return alive[0]


def resolve_records(
    schedule: Mapping[int, Sequence[tuple[int, int]]], week_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Wins and points-for per team over the scheduled weeks, for every sim at once.

    ``week_scores[s, t, w]`` is team ``t``'s score in week ``w`` (1-based key into ``schedule``) for
    sim ``s``. A tied game counts as a win for neither side. Returns ``(wins, points)`` each
    ``(n_sims, n_teams)``.
    """
    n_sims, n_teams, _ = week_scores.shape
    wins = np.zeros((n_sims, n_teams), dtype=float)
    points = np.zeros((n_sims, n_teams), dtype=float)
    for w, pairs in schedule.items():
        col = int(w) - 1
        for a, b in pairs:
            sa = week_scores[:, a, col]
            sb = week_scores[:, b, col]
            wins[:, a] += sa > sb
            wins[:, b] += sb > sa
            points[:, a] += sa
            points[:, b] += sb
    return wins, points

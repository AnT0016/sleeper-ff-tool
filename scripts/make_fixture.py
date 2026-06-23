"""Generate (and verify) the committed mechanics-test fixture.

Pulls 2025 Week 1 projections for skill positions, confirms the standard half-PPR dict reproduces
each row's precomputed ``pts_half_ppr``, and writes a small sample to tests/fixtures/. Run manually
when refreshing the fixture; the test itself runs offline against the committed JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from scoring.engine import points
from scoring.standard import SLEEPER_STANDARD_HALF_PPR
from sleeper.client import get_projections

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "projections_2025_w1_skill.json"
PER_POSITION = 8
# RB/WR/TE only: the engine-mechanics check needs positions whose pts_half_ppr is exactly
# reproducible from the standard dict. QBs are excluded (see scoring.standard docstring).
POSITIONS = ("RB", "WR", "TE")


def main() -> None:
    rows = get_projections(2025, 1, positions=POSITIONS)
    by_pos: dict[str, list[dict]] = {p: [] for p in POSITIONS}
    for r in rows:
        stats = r.get("stats") or {}
        pos = (r.get("player") or {}).get("position")
        if pos in by_pos and stats.get("pts_half_ppr") is not None:
            by_pos[pos].append(r)

    sample: list[dict] = []
    worst = 0.0
    for pos in POSITIONS:
        top = sorted(by_pos[pos], key=lambda r: r["stats"]["pts_half_ppr"], reverse=True)[:PER_POSITION]
        for r in top:
            stats = r["stats"]
            expected = float(stats["pts_half_ppr"])
            got = points(stats, SLEEPER_STANDARD_HALF_PPR)
            worst = max(worst, abs(got - expected))
            p = r.get("player") or {}
            sample.append(
                {
                    "name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
                    "position": pos,
                    "player_id": r.get("player_id"),
                    "pts_half_ppr": expected,
                    "stats": stats,
                }
            )
            print(f"{pos:3} {sample[-1]['name']:24} expected={expected:7.2f} got={got:7.2f} d={got-expected:+.4f}")

    print(f"\nworst abs residual: {worst:.4f}  ({len(sample)} samples)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

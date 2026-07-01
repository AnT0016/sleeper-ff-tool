"""Freeze a season's Sleeper projections to a committed snapshot — an immutable, ex-ante baseline.

    ./.venv/Scripts/python scripts/freeze_projections.py               # current season
    ./.venv/Scripts/python scripts/freeze_projections.py --season 2026

Projection endpoints only serve the *latest* values, so capture 2026 **before Week 1** — then the tool
can be graded out-of-sample later (re-score the frozen rows in the real league's scoring) with no
hindsight contamination. Writes ``data_cache/frozen/projections_season_<year>.json`` (committed).
Refuses to overwrite an existing snapshot unless ``--force`` (freezing is meant to be immutable).
"""

from __future__ import annotations

import argparse
import datetime
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.frozen import frozen_path, save_frozen
from sleeper import client


def _default_season() -> int:
    try:
        return int(client.get_state().get("season") or 2026)
    except Exception:
        return 2026


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a season's projections (immutable ex-ante baseline)")
    ap.add_argument("--season", type=int, default=None, help="season to freeze (default: current)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing snapshot")
    args = ap.parse_args()
    season = args.season or _default_season()

    path = frozen_path(season)
    if path.exists() and not args.force:
        print(f"Already frozen: {path}\n  (freezing is immutable by design — pass --force to overwrite.)")
        return

    rows = client.get_season_projections(season)
    nonzero = sum(1 for r in rows if (r.get("stats") or {}).get("pts_half_ppr"))
    today = datetime.date.today().isoformat()
    out = save_frozen(season, rows, frozen_at=today)
    print(f"✓ Froze {len(rows)} {season} projection rows ({nonzero} with points) → {out}")
    print(f"  frozen_at {today}. Re-score later with build_board({season}, scoring, fetch=frozen_fetch({season})).")


if __name__ == "__main__":
    main()

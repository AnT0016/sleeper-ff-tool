"""Read/write **frozen** (immutable, ex-ante) projection snapshots.

Sleeper's projection endpoints only ever serve the *latest* values, so once a season starts you can no
longer recover what was projected preseason. Freezing a season's raw projection rows *before* Week 1
gives a clean out-of-sample baseline: re-score the frozen rows in the real league's scoring later and
grade the tool's decisions against reality with **zero hindsight contamination** — the thing a
retrospective backtest can't guarantee.

Raw rows are stored as-is (scoring-agnostic), so they can be re-scored in whatever the eventual league
scoring turns out to be. :func:`frozen_fetch` returns a ``build_board(..., fetch=)``-compatible loader.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

FROZEN_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "frozen"


def frozen_path(season: int, kind: str = "projections_season", *, directory: Path = FROZEN_DIR) -> Path:
    return Path(directory) / f"{kind}_{season}.json"


def save_frozen(
    season: int,
    rows: Sequence[dict],
    *,
    frozen_at: str,
    kind: str = "projections_season",
    source: str = "sleeper get_season_projections",
    directory: Path = FROZEN_DIR,
) -> Path:
    """Write ``rows`` (raw projection dicts) to an immutable snapshot; returns the path written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "season": int(season),
        "kind": kind,
        "frozen_at": frozen_at,
        "source": source,
        "n": len(rows),
        "rows": list(rows),
    }
    path = frozen_path(season, kind, directory=directory)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_frozen(season: int, *, kind: str = "projections_season", directory: Path = FROZEN_DIR) -> dict:
    """The full frozen payload (``season``/``frozen_at``/``rows``/…). Raises if not frozen."""
    return json.loads(frozen_path(season, kind, directory=directory).read_text(encoding="utf-8"))


def load_frozen_rows(season: int, *, kind: str = "projections_season", directory: Path = FROZEN_DIR):
    """Just the raw projection rows from the frozen snapshot."""
    return load_frozen(season, kind=kind, directory=directory)["rows"]


def frozen_fetch(
    season: int, *, kind: str = "projections_season", directory: Path = FROZEN_DIR
) -> Callable[..., list[dict]]:
    """A ``projections.board.build_board(..., fetch=)``-compatible callable over the frozen rows.

    Lets you rebuild the exact preseason board later: ``build_board(2026, real_scoring,
    fetch=frozen_fetch(2026))`` re-scores the frozen 2026 projections in the real league's scoring.
    """
    rows = load_frozen_rows(season, kind=kind, directory=directory)

    def _fetch(_season, positions=None, season_type="regular"):
        return rows

    return _fetch

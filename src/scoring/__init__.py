"""Custom league scoring engine.

Re-scores raw per-stat lines with this league's exact ``scoring_settings`` pulled live from the
Sleeper API. Engine = ``sum(stat_value * scoring_settings.get(key, 0))`` over all stat keys.
**Never hand-code scoring** — the API is the source of truth (see CLAUDE.md).
"""

from .engine import points

__all__ = ["points"]

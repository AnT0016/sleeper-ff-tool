"""Custom league scoring engine.

One pure function re-scores a raw stat line with an arbitrary Sleeper-keyed
``scoring_settings`` dict. The same function serves both our league's settings (pulled
live from the API -- never hand-coded) and Sleeper's "standard" half-PPR dict (used only
to validate that the engine math is correct, independent of our league).
"""

from __future__ import annotations

from collections.abc import Mapping


def points(stats: Mapping[str, float], scoring_settings: Mapping[str, float]) -> float:
    """Fantasy points = ``sum(stat_value * coefficient)`` over the scoring keys.

    ``scoring_settings`` is the source of truth -- pull it live from the Sleeper league
    endpoint. A stat key absent from ``scoring_settings`` scores zero; a scoring key absent
    from ``stats`` contributes zero.

    This is why FG-distance buckets (``fgm_0_19``..``fgm_50p``) and points-allowed buckets
    (``pts_allow_*``) need no special handling: the raw stat line flags exactly one bucket per
    event, and the redundant totals Sleeper also reports (e.g. a flat ``fgm``, or ``fgm_50_59``
    overlapping ``fgm_50p``) are simply not in our settings, so they never double-count.
    """
    return sum(
        float(stats.get(key, 0.0)) * float(coef)
        for key, coef in scoring_settings.items()
    )

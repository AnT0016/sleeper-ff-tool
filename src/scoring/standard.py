"""Sleeper's PUBLIC default half-PPR scoring -- for validating engine MECHANICS only.

This is **not** our league's scoring (which is always pulled live from the API, never hard-coded).
It exists solely so a test can confirm that ``points(projection["stats"], SLEEPER_STANDARD_HALF_PPR)``
reproduces the ``pts_half_ppr`` value Sleeper precomputes for each projection row -- proving the
engine arithmetic is correct, independent of our league.

The mechanics test is scoped to **RB/WR/TE**, where this dict reproduces ``pts_half_ppr`` to
rounding (~0.02). QBs are deliberately excluded: the projection provider's ``pts_half_ppr`` bakes
an extra, volume-correlated *passing* component into each QB row (beyond pass_yd/pass_td/pass_int)
that is provider-specific noise, not league scoring. QB scoring is instead validated end-to-end in
the custom-season check, which re-scores real game stats with our league's actual API settings and
compares against real Sleeper matchup points. The passing keys below are Sleeper's nominal defaults
for completeness and are not exercised by the mechanics test.
"""

from __future__ import annotations

SLEEPER_STANDARD_HALF_PPR: dict[str, float] = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -1.0,
    "pass_2pt": 2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rush_2pt": 2.0,
    "rec": 0.5,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "rec_2pt": 2.0,
    "fum_lost": -2.0,
    "fum_rec_td": 6.0,
}

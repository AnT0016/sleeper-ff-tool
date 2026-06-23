"""Engine-mechanics check against real Sleeper data (offline, via committed fixture).

Applying Sleeper's standard half-PPR dict to a projection's raw ``stats`` must reproduce the
``pts_half_ppr`` Sleeper precomputed for that row. This proves the engine arithmetic independent of
our league. Scoped to RB/WR/TE (see ``scoring.standard`` for why QBs are excluded). Refresh the
fixture with ``python scripts/make_fixture.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scoring.engine import points
from scoring.standard import SLEEPER_STANDARD_HALF_PPR

FIXTURE = Path(__file__).parent / "fixtures" / "projections_2025_w1_skill.json"


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_present_and_covers_positions(samples):
    assert samples, "fixture is empty"
    assert {"RB", "WR", "TE"} <= {s["position"] for s in samples}


def test_standard_dict_reproduces_pts_half_ppr(samples):
    bad = []
    for s in samples:
        got = points(s["stats"], SLEEPER_STANDARD_HALF_PPR)
        if abs(got - s["pts_half_ppr"]) > 0.1:
            bad.append((s["name"], s["pts_half_ppr"], round(got, 3)))
    assert not bad, f"engine did not reproduce pts_half_ppr for: {bad}"

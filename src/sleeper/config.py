"""Non-secret league configuration.

The Sleeper ``league_id`` is part of the public league URL, so it is committed here rather
than treated as a secret. Override any value via the matching ``SLEEPER_*`` env var.
"""

from __future__ import annotations

import os

# "Fantasy Campechano" -- 2025 season league (status=complete). This is our scoring source of
# truth and the validation data source (it has full reported matchup points).
#
# NOTE: the 2026 redraft league, once created, will likely be a NEW id (and CLAUDE.md describes
# a 6th bench + IR slot that this 2025 league does not have). Update LEAGUE_ID when it exists.
LEAGUE_ID: str = os.environ.get("SLEEPER_LEAGUE_ID", "1257071615817043968")
PREVIOUS_LEAGUE_ID: str = os.environ.get("SLEEPER_PREVIOUS_LEAGUE_ID", "1124851086289559552")

MY_USER_ID: str = os.environ.get("SLEEPER_USER_ID", "866260653093036032")
MY_USERNAME: str = "ant0016"

# Season whose data backs validation / the 2025 league above.
VALIDATION_SEASON: int = int(os.environ.get("SLEEPER_VALIDATION_SEASON", "2025"))

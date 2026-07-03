"""Non-secret league configuration.

Sleeper ``league_id``s are part of public league URLs, so they are committed here rather than
treated as secrets. Override any value via the matching ``SLEEPER_*`` env var (one-off runs), or
pass ``--league`` to the CLI scripts.

League registry (newest first). ``LEAGUE_ID`` is the ACTIVE league every tool operates on; at each
season rollover, point it at the new season's league (one line). The refresh pipeline fail-safes on
a season mismatch (see ``analysis.snapshot.offseason_skip_reason``), so a stale ``LEAGUE_ID`` skips
instead of publishing a wrong snapshot. Discover a new season's leagues with
``sleeper.client.get_user_leagues(MY_USER_ID, <season>)``.
"""

from __future__ import annotations

import os

#: 2026 "Test league" — a sandbox copy of the 2025 league (same scoring/roster/waiver settings),
#: created while waiting for the real 2026 redraft league. Replace with the real league id once
#: the league is recreated for 2026.
LEAGUE_ID_2026_TEST: str = "1378062197778833408"
#: "Fantasy Campechano" 2025 — completed; the scoring source of truth and validation data source
#: (it has full reported matchup points).
LEAGUE_ID_2025: str = "1257071615817043968"
#: 2024 season league — completed.
LEAGUE_ID_2024: str = "1124851086289559552"

#: The ACTIVE league (drives every tool). Currently the 2026 test league.
LEAGUE_ID: str = os.environ.get("SLEEPER_LEAGUE_ID", LEAGUE_ID_2026_TEST)
#: The most recent COMPLETED league (backtests / validation).
PREVIOUS_LEAGUE_ID: str = os.environ.get("SLEEPER_PREVIOUS_LEAGUE_ID", LEAGUE_ID_2025)

MY_USER_ID: str = os.environ.get("SLEEPER_USER_ID", "866260653093036032")
MY_USERNAME: str = "ant0016"

#: Season whose completed data backs scoring validation (the 2025 league above).
VALIDATION_SEASON: int = int(os.environ.get("SLEEPER_VALIDATION_SEASON", "2025"))

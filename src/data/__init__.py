"""nflverse ingest and player ID mapping.

Pulls weekly stats, snap counts, depth charts, injuries, schedules, and expected fantasy points
via ``nflreadpy`` (NOT the deprecated ``nfl_data_py``), plus the player ID crosswalk
(``load_ff_playerids``). Join skill players on ``gsis_id`` and DST by team abbreviation; log every
projection row that fails to join.
"""

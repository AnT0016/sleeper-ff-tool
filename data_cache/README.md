# data_cache/

Committed data artifacts — custom-scored projections and derived tables as **parquet** (and/or
SQLite). These are checked into git on purpose so the hosted dashboard redeploys from the latest
data committed by the weekly GitHub Actions refresh (Phase 5).

- `season.db` — the **single SQLite snapshot** the hosted dashboard (`apps/season_app.py`) reads.
  One table per dashboard section (lineup, start/sit, handcuffs, spend, stashes, team-strength
  matrices, bye gaps, needs, trades, playoff outlook) plus a single-row `meta` table (season, week,
  standings, posture, totals, `generated_at`). Rebuilt by `scripts/refresh_data.py` (the weekly
  GitHub Actions cron) and committed back, which triggers a Streamlit redeploy. Written atomically
  (temp file + replace) so a committed copy is never half-written.

Not committed (gitignored, regenerated on demand): `http_cache.sqlite` (the `requests-cache` HTTP
store, written by `src/sleeper/http.py`) and `nflverse_cache/` (nflreadpy's raw download cache).

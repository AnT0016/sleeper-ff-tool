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

- `backtest.db` — a **separate, one-off** artifact for the dashboard's "Backtest" tab: a
  season "what if I'd used this tool" review (`weekly` + `draft` + `draftboard` + `matchup_detail` +
  `transactions` tables + a `meta` row). Built by `scripts/backtest.py` from real matchup results and
  the real draft. Kept apart from `season.db` on purpose so the weekly refresh never overwrites it.
  Rebuild whenever you want to re-run it (safe mid-season — only finished weeks are included).

Not committed (gitignored, regenerated on demand): `http_cache.sqlite` (the `requests-cache` HTTP
store, written by `src/sleeper/http.py`) and `nflverse_cache/` (nflreadpy's raw download cache).

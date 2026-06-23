# data_cache/

Committed data artifacts — custom-scored projections and derived tables as **parquet** (and/or
SQLite). These are checked into git on purpose so the hosted dashboard redeploys from the latest
data committed by the weekly GitHub Actions refresh (Phase 5).

Not committed (gitignored, regenerated on demand): `http_cache.sqlite` (the `requests-cache` HTTP
store, written by `src/sleeper/http.py`) and `nflverse_cache/` (nflreadpy's raw download cache).

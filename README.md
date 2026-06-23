# Sleeper Fantasy Football Tool

Personal, free, **read-only** tooling for one specific Sleeper redraft league. It re-scores
projections in this league's *exact* scoring and automates weekly start/sit, waiver, and stash
decisions. Fantasy only — no betting/DFS.

- **Behavioral contract & league facts:** [CLAUDE.md](CLAUDE.md) — read this first. It is the
  source of truth for scoring rules, roster slots, and the immutable "never write to Sleeper" rule.
- **Phased plan:** [docs/PLAN.md](docs/PLAN.md)
- **Live status / checklist:** [docs/PROGRESS.md](docs/PROGRESS.md)

## Quick start

```sh
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/sleeper/` | Single Sleeper API client + the cached HTTP layer all calls go through |
| `src/scoring/` | Custom league scoring engine (driven by live `scoring_settings`) |
| `src/data/` | nflverse ingest (`nflreadpy`) + player ID crosswalk |
| `src/projections/` | Projection ingest + custom re-scoring |
| `src/optimizer/` | Weekly lineup optimizer (PuLP) |
| `src/waivers/` | Waiver / stash / handcuff intelligence |
| `src/analysis/` | Team & league analysis views |
| `apps/` | Streamlit apps (local draft tool, hosted season dashboard) |
| `data_cache/` | Committed parquet/SQLite data artifacts |
| `.github/workflows/` | Weekly data-refresh cron (Phase 5) |

# Plan

The authoritative *what* and *why* live in [CLAUDE.md](../CLAUDE.md) (see **Build order** and
**Immutable rules**). This file is the phase map; [PROGRESS.md](PROGRESS.md) tracks live status.

Build strictly in order — do not skip ahead.

| Phase | Deliverable | Key constraint |
| --- | --- | --- |
| 1 | Sleeper client + scoring engine | **Validate first:** re-score last season's nflverse actuals against Sleeper-reported points for ~20 players across all positions (incl. K & DEF) before building anything else |
| 2 | Local live draft tracker | Slot-agnostic (tiers + VOR) until `draft_order` is revealed, then simulate snake picks; local Streamlit polling `/draft/<id>/picks` ~3s |
| 3 | Weekly lineup optimizer | PuLP; exact slot constraints + FLEX from {RB,WR,TE}; exclude bye/OUT/IR |
| 4 | Waiver / stash / handcuff intelligence | Reverse-standings **priority** logic (not FAAB); handcuff free-agent detector; playoff-SOS stash ranker |
| 5 | Hosted season dashboard + weekly refresh | Streamlit Community Cloud + GitHub Actions cron (Tue evening CEST, UTC best-effort) that commits updated data |
| 6 | (Optional) Monte Carlo draft simulator | Forward simulation; build last |

## Foundations already decided (CLAUDE.md → Architecture)
- Python data layer → custom-scored projections → parquet/SQLite committed to the repo.
- All Sleeper calls isolated behind one client module; endpoints are undocumented and may change.
- nflverse via `nflreadpy` (not the deprecated `nfl_data_py`).
- No FastAPI / JS frontend unless the project outgrows Streamlit.

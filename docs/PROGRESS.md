# Progress Log

Living status for the Sleeper FF tool. The authoritative *what/why* lives in
[CLAUDE.md](../CLAUDE.md) (Build order + Immutable rules); this file tracks *where we are*.
**Update at the end of every phase:** check off what shipped and note what's next.

Legend: `[ ]` TODO · `[~]` in progress · `[x]` done

## Phase 0 — Scaffolding `[x]`
- [x] Repo layout (`src/`, `apps/`, `data_cache/`, `tests/`, `docs/`, `.github/workflows/`)
- [x] Dependency pinning (`pyproject.toml`, Python 3.11+)
- [x] Cached HTTP layer (`requests-cache` + SQLite, per-URL TTLs) in `src/sleeper/http.py`
- [x] `.gitignore` (commit `data_cache/`, ignore volatile `http_cache.sqlite`)
- [x] README + docs (PLAN.md, PROGRESS.md)
- [ ] Initial commit (repo not yet `git init`-ed — pending go-ahead)

## Phase 1 — Sleeper client + scoring engine `[x]` DONE
- [x] One Sleeper client module ([src/sleeper/client.py](../src/sleeper/client.py)) wrapping league /
      rosters / users / matchups / transactions / players / trending / state / projections / stats
- [x] nflverse ingest via `nflreadpy` ([src/data/nflverse.py](../src/data/nflverse.py)), 24h filesystem cache
- [x] ID + stat mapping ([src/data/ids.py](../src/data/ids.py)): gsis_id↔sleeper_id, DST by team,
      nflverse→Sleeper `STAT_MAP` (incl. fum-lost sum, 50+ FG bucket, blocked-kick→miss)
- [x] Generic scoring engine ([src/scoring/engine.py](../src/scoring/engine.py)): `sum(stat*coef)`,
      driven by live API `scoring_settings`
- [x] **Validation gate PASSED (2025 season, 21-player sample across all positions):**
      - Engine vs Sleeper-reported matchup points (incl. K & DEF): **max |diff| = 0.000**
      - Independent nflverse re-score (skill + K): **max |diff| = 0.000**
- [x] Tests: `tests/test_scoring_engine.py` (synthetic), `tests/test_scoring_mechanics.py`
      (offline fixture; standard half-PPR reproduces `pts_half_ppr` for RB/WR/TE)
- **Findings:** league's `pass_td=4` is non-standard vs Sleeper's `pts_half_ppr` (which uses 6);
  blocked FGs count as `fgmiss` in Sleeper but are separate in nflverse; roster reconciled via the
  API — **5 bench + 1 IR** (IR lives in `settings.reserve_slots`), CLAUDE.md corrected from 6 to 5 bench.
- **Next:** Phase 2 (local live draft tracker).

## Phase 2 — Local live draft tracker `[ ]` TODO
- [ ] Best-available by custom VOR, tiers, positional-run detection, roster needs
- [ ] Slot-agnostic until `draft_order` populates, then simulate snake pick numbers
- [ ] Local Streamlit app polling `/draft/<id>/picks` ~3s
- **Next:** Phase 3.

## Phase 3 — Weekly lineup optimizer (PuLP) `[ ]` TODO
- [ ] Slot constraints: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX{RB/WR/TE}, 1 K, 1 DEF
- [ ] Exclude bye-week and OUT/IR players; optimize over custom-scored weekly projections
- **Next:** Phase 4.

## Phase 4 — Waiver / stash / handcuff intelligence `[ ]` TODO
- [ ] Handcuff free-agent detector (flag high-priority when a starter is Q/D/O and backup unrostered)
- [ ] Reverse-standings **priority** spend advice (not FAAB)
- [ ] Playoff-SOS stash ranker (Weeks 15–17 value)
- **Next:** Phase 5.

## Phase 5 — Hosted season dashboard + weekly refresh `[ ]` TODO
- [ ] Streamlit Community Cloud dashboard + team-analysis views
- [ ] GitHub Actions weekly cron (Tue evening CEST, UTC best-effort) + `workflow_dispatch`,
      commits updated data so Streamlit auto-redeploys
- **Next:** Phase 6 (optional).

## Phase 6 — (Optional) Monte Carlo draft simulator `[ ]` TODO
- [ ] Forward simulation of draft outcomes — build last, only if useful

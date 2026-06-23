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

## Phase 2 — Local live draft tracker `[x]` DONE
- [x] Local Streamlit app ([apps/draft_app.py](../apps/draft_app.py)), run with
      `streamlit run apps/draft_app.py`. Polls `/draft/<id>/picks` every ~3s via
      `st.fragment(run_every=...)`; diffs against the last poll for a live "new picks" feed.
- [x] 3 read-only draft endpoints added behind the one client
      ([src/sleeper/client.py](../src/sleeper/client.py)): `get_league_drafts`, `get_draft`,
      `get_draft_picks` (HTTP layer already sets `…/draft/*` to DO_NOT_CACHE).
- [x] Custom-scored board ([src/projections/board.py](../src/projections/board.py)): every season
      projection re-scored in OUR live `scoring_settings` (reuses the Phase 1 engine); ADP
      (`adp_half_ppr`) carried as a market signal only, never for ranking.
- [x] VOR + tiers ([src/draft/vor.py](../src/draft/vor.py)): replacement = first non-starter at
      each position, with **data-driven FLEX allocation** (best leftover RB/WR/TE fill the flex);
      per-position gap-based tiers. Board ranks **by VOR** (cross-position).
- [x] Snake math + survival ([src/draft/snake.py](../src/draft/snake.py)): slot-agnostic (tiers+VOR)
      until `draft_order` populates, then our pick numbers `(r-1)*12+S` / `r*12-S+1` and per-pick
      🟢/🟡/🔴 "likely to survive" flags vs market ADP.
- [x] Roster needs + runs ([src/draft/roster.py](../src/draft/roster.py)): fills dedicated slots
      then FLEX from our picks (via `picked_by`); 🎯 highlights open needs; rolling positional-run
      counts.
- [x] Tests: `tests/test_draft.py` (23 offline unit tests — snake, VOR/flex, tiers, roster, runs).
- [x] **Validated end-to-end against the 2025 completed draft** (`draft_id …069`): reconstructed
      snake picks `[7,18,…,162]` match my actual pick numbers exactly; all 168 drafted players
      (incl. DEF-by-team) join the board; K first surfaces at VOR rank #53 and DEF at #69 (the
      "don't draft K/DEF early" rule falls out of VOR). Streamlit script verified headless via
      `AppTest` for both complete- and mid-draft states.
- **Next:** Phase 3 (weekly lineup optimizer, PuLP).

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

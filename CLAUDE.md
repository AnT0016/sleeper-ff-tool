# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Project: Sleeper Fantasy Football Tool.** Persistent project context. Read at the start of every session. Keep edits high-signal; this is a behavioral contract, not documentation.

## Repo status & commands
Scaffolded; Python 3.11+, `src/` layout. **Phase 1 (data + scoring foundation) is DONE and validated** — see [docs/PROGRESS.md](docs/PROGRESS.md). One-time setup: `python -m venv .venv` then `./.venv/Scripts/python -m pip install -e ".[dev]"`.

- Run tests (offline): `./.venv/Scripts/python -m pytest -q`
- Custom-scoring season validation (network, cached): `./.venv/Scripts/python scripts/validate_custom.py`
- Refresh the mechanics-test fixture: `./.venv/Scripts/python scripts/make_fixture.py`

All Sleeper calls live behind [src/sleeper/client.py](src/sleeper/client.py) over a cached session ([src/sleeper/http.py](src/sleeper/http.py)). The scoring engine ([src/scoring/engine.py](src/scoring/engine.py)) is generic — it re-scores any stat line with a live `scoring_settings` dict; validated to reproduce Sleeper's reported 2025 points to 0.00 for all positions incl. K & DEF.

## What this project is
A personal, free, DIY tool for **one specific Sleeper redraft league** (fantasy only — no betting/DFS). It replaces a paid subscription by (1) re-scoring projections in *this league's exact scoring* and (2) automating weekly start/sit, waiver, and stash decisions. **Read-only** against Sleeper; it never writes to the league.

## Source of truth for scoring
Pull `scoring_settings` and `roster_positions` live from `GET https://api.sleeper.app/v1/league/<LEAGUE_ID>`. **Never hand-code scoring.** The table below is a reference / validation target only — if it ever disagrees with the API, the API wins and this file should be updated.

## League facts (locked)
- Platform: Sleeper. **12 teams, redraft, snake** draft.
- Draft: 30s/pick, CPU autopick on, all players available. **Draft slot is randomized and revealed only ~1 day to 1 week before the draft** (usually the last week before the NFL season). → All draft-prep logic must be **slot-agnostic** (tiers + VOR) until `draft_order` is populated, then simulate snake pick numbers for the revealed slot.
- Waivers: **reverse-standings priority, NOT FAAB/FAB.** Waivers clear **Wednesday 09:00 CEST**; dropped players sit on waivers **2 days**. → Waiver advice reasons about *spending a single ordered priority*, not dollars. Refresh waiver advice **Tuesday evening CEST**.
- Trades: 2-day review, vetoes on (6 votes to veto), **trade deadline Week 11**.
- IR: **1 slot.** OUT / suspended / NA / DNR / doubtful are NOT IR-eligible (COVID is). → A player must hold an IR-eligible status to occupy the slot.
- Playoffs: **6 teams, Weeks 15-16-17**, one week per round, default seeding (teams stay on their bracket side), lower bracket = toilet bowl. → "Playoff value" = combined custom-scored projection over Weeks 15-17.
- Every NFL team has a bye — plan for roster holes on starters' bye weeks.

## Roster / lineup slots (locked)
Starters (9): `QB`, `RB`, `RB`, `WR`, `WR`, `TE`, `FLEX (RB/WR/TE)`, `K`, `DEF`
Bench: 5 (any position). IR: 1 (Sleeper stores this in `settings.reserve_slots`, not `roster_positions`).
Active roster = 14 players (+1 IR) = 15 roster spots.
→ Lineup-optimizer constraints: exactly 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX from {RB,WR,TE}, 1 K, 1 DEF; exclude bye-week and OUT/IR players. FLEX takes the best leftover RB/WR/TE.

## Scoring reference (half-PPR; non-standard bits flagged ⚠)
Source of truth is the API — keys below are for understanding & validation.

Offense
- `pass_yd` 0.04 · `pass_td` 4 ⚠ (4-pt passing TD, not 6) · `pass_2pt` 2 · `pass_int` -1
- `rush_yd` 0.1 · `rush_td` 6 · `rush_2pt` 2
- `rec` 0.5 ⚠ (HALF-PPR) · `rec_yd` 0.1 · `rec_td` 6 · `rec_2pt` 2
- ⚠ No TE premium. · `fum_lost` -2 · `fum_rec_td` 6

Kicker ⚠ — distance-based; most public sheets misvalue K
- `fgm_0_19` 3 · `fgm_20_29` 3 · `fgm_30_39` 3 · `fgm_40_49` 4 · `fgm_50p` 5
- `xpm` 1 · `fgmiss` -1 · `xpmiss` -1

Defense / Special Teams ⚠ — rich DST scoring; also commonly misvalued
- `def_td` 6 · `sack` 1 · `int` 2 · `fum_rec` 2 · `safe` 2 · `ff` 1 · `blk_kick` 2
- Points allowed (independent buckets, one applies): `pts_allow_0` 10 · `1_6` 7 · `7_13` 4 · `14_20` 1 · `21_27` 0 · `28_34` -1 · `35p` -4
- Special-teams defense: `def_st_td` 6 · `def_st_ff` 1 · `def_st_fum_rec` 1
- Special-teams player: `st_td` 6 · `st_ff` 1 · `st_fum_rec` 1

Engine = `sum(stat_value * scoring_settings.get(key, 0))` over all stat keys. FG-distance buckets and points-allowed buckets are **mutually exclusive** — a play scores in exactly one bucket.

## Architecture (decisions made)
- **Python data layer.** Ingest Sleeper + nflverse + projections → compute custom-scored projections → store as **parquet** (and/or SQLite) committed to the repo.
- **Live draft tool: LOCAL Streamlit app.** Polls `GET /draft/<DRAFT_ID>/picks` every ~3s. No idle/sleep risk during the draft hour.
- **Season tracker: HOSTED on Streamlit Community Cloud**, refreshed by a **GitHub Actions** weekly cron that commits updated data (Streamlit auto-redeploys on the commit). Schedule for Tuesday evening CEST + a `workflow_dispatch` manual trigger (cron is UTC, best-effort).
- Do NOT add FastAPI / a JS frontend unless the project outgrows Streamlit.

## Data sources & hard rules
- **Sleeper API** (unofficial, read-only, no auth). Stay **under ~1000 calls/min**. Cache `GET /players/nfl` (~5MB) **once per day**. Weekly projections via the undocumented `GET https://api.sleeper.com/projections/nfl/<year>/<week>?season_type=regular&position[]=QB&...` (returns raw per-stat fields to re-score). **Isolate ALL Sleeper calls behind one client module** — endpoints are undocumented and can change.
- **nflverse via `nflreadpy`** (NOT the deprecated `nfl_data_py`): weekly stats, snap counts, depth charts, injuries, schedules, expected fantasy points, and the player ID crosswalk (`load_ff_playerids`). Filesystem cache, 24h.
- **Projections v1:** use Sleeper's own projections endpoint (zero ID-mapping friction). Add `ffanalytics` (R) / FantasyPros later. FantasyPros & ffanalytics data are personal-use only — code can be public, but don't redistribute their data.
- **ID join:** skill players on `gsis_id`; DST by team abbreviation. **Log every projection row that fails to join.**

## Build order (do NOT skip ahead)
1. Sleeper client + scoring engine. **Validate first:** re-score last season's nflverse actuals and confirm totals match Sleeper's reported points for ~20 players across all positions (incl. K & DEF) before building anything else.
2. Local live draft tracker — best-available by custom VOR, tiers, positional runs, roster needs; snake-pick simulation once the slot is revealed.
3. Weekly lineup optimizer — PuLP, with FLEX + bye/injury constraints, over custom-scored weekly projections.
4. Waiver / stash / handcuff intelligence — handcuff free-agent detector (flag high-priority if my starter is Q/D/O and the backup is unrostered); reverse-priority spend advice; playoff-SOS stash ranker.
5. Hosted season dashboard + GitHub Actions weekly refresh + team-analysis views.
6. (Optional, last) forward Monte Carlo draft simulator.

## Immutable rules
- Read-only against Sleeper. Never automate roster / waiver / trade actions.
- The API is the source of truth for scoring & roster settings; this file is a reference.
- Don't draft K or DEF early — the edge is weekly streaming via custom scoring, not draft position.
- No secrets in the repo (Sleeper needs none; if a FantasyPros key is added, use Streamlit/Actions secrets).

## League ids (registry: [src/sleeper/config.py](src/sleeper/config.py))
- `LEAGUE_ID` (ACTIVE): `1378062197778833408` — 2026 **"Test league"** sandbox, settings copied from the 2025 league (scoring/roster/waivers/playoffs identical, verified against the API). **Swap to the real 2026 league id once the league is recreated** — one line in config.py, or per-run via `SLEEPER_LEAGUE_ID`; discover it with `client.get_user_leagues(MY_USER_ID, 2026)`. The weekly refresh fail-safes on a league-vs-NFL season mismatch, so a stale id skips instead of publishing a wrong snapshot.
- `LEAGUE_ID_2025` (= `PREVIOUS_LEAGUE_ID`): `1257071615817043968` — "Fantasy Campechano", complete; scoring source of truth + validation data.
- `LEAGUE_ID_2024`: `1124851086289559552` — complete.
- `DRAFT_ID`: test league's draft is `1378062202891685888`; the real one is known on draft day via `GET /league/<LEAGUE_ID>/drafts` (never HTTP-cached — a just-created draft is visible immediately).
- `MY_USER_ID` / team: `866260653093036032` (username `ant0016`)

> Reconciled against the API: roster is **5 bench + 1 IR** (IR lives in `settings.reserve_slots`; `reserve_allow_cov=1`, all other `reserve_allow_*` are 0 — matching the IR rules above). The earlier "Bench 6" was corrected to 5. Scoring (42 keys) matches this file exactly. The 2026 redraft league will be a new id — re-confirm roster/scoring against the API once it exists.

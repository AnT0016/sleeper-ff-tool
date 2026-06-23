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

## Phase 3 — Weekly lineup optimizer (PuLP) `[x]` DONE
- [x] Integer LP core ([src/optimizer/lineup.py](../src/optimizer/lineup.py)): one binary var per
      *(eligible player × slot the position may fill)*; each player ≤ 1 slot, each slot ≤ capacity,
      maximize custom-scored points (with a negligible fill-nudge so 0-proj K/DEF still get slotted).
      Slots are read live from the league's `roster_positions` (`lineup_slots`) — **1 QB, 2 RB, 2 WR,
      1 TE, 1 FLEX{RB/WR/TE}, 1 K, 1 DEF**. Position/FLEX eligibility is structural (illegal pairs get
      no variable). Unfillable slots are returned as `holes`, never an exception.
- [x] Start/sit + risk ([src/optimizer/startsit.py](../src/optimizer/startsit.py)): per-bench delta
      vs. the starter they'd replace (≤ 0 in an optimal lineup); risky-start flags for
      Questionable/Doubtful starters and **forced downgrades** (a higher-projected rostered player
      stuck on bye/OUT/IR), plus an idle (bye/OUT/IR) list.
- [x] Live-data glue ([src/optimizer/inputs.py](../src/optimizer/inputs.py)): joins my roster
      (`/rosters` by `owner_id`, `reserve` = IR) → weekly projections re-scored in our live
      `scoring_settings` (Phase 1 engine) → Sleeper `injury_status` (authoritative) → byes. Logs every
      rostered player that fails to join a projection. OUT/IR/PUP/Sus/NA/DNR + IR-slot excluded;
      Q/D stay startable-but-flagged.
- [x] nflverse loaders ([src/data/nflverse.py](../src/data/nflverse.py)): `load_schedules` (byes =
      teams with no REG game that week; nflverse `LA`→Sleeper `LAR` normalized) and `load_injuries`
      (secondary cross-check).
- [x] Runnable ([scripts/optimize_lineup.py](../scripts/optimize_lineup.py)):
      `python scripts/optimize_lineup.py --week N [--season Y]` — prints the optimal lineup + total,
      start/sit table, risky flags, and the idle list.
- [x] Tests: `tests/test_optimizer.py` (15 offline unit tests — exact slot fill, **FLEX takes the
      best leftover RB/WR/TE and never a higher-projected QB**, bye/OUT/IR exclusion, hole reporting,
      0-proj fill, start/sit targeting + sign, Q/D + forced-downgrade flags, `assemble_players`/
      `bye_teams` joins). Full suite **38 passed**.
- [x] **Validated end-to-end against the 2025 league** (Week 5): all 14 roster players join their
      projection rows; custom scoring matches (Jameson Williams 10.55 ≈ Sleeper `pts_half_ppr` 10.54);
      Garrett Wilson on the IR slot is excluded and surfaces as a forced-downgrade flag; Q/D starters
      flagged. PuLP migrated to `prob.add_variable` (no deprecation warnings); kept bundled
      `PULP_CBC_CMD` (the non-deprecated `COIN_CMD` needs an external `cbc` binary).
- **Note:** querying a *completed* season's weekly projections returns ADP-only (zero-stat) rows for
  many non-stars (e.g. wk5 DJ Moore = `{adp_dd_ppr: 1000}` → 0.0). That is a historical-data quirk,
  not a join bug; live in-season runs for an upcoming week return full stat lines.
- **Next:** Phase 4 (waiver / stash / handcuff intelligence). The handcuff/waiver logic will reuse
  `optimize(...)` to score "does adding free-agent X improve my optimal lineup?"; the lineup Streamlit
  view lands in Phase 5's hosted dashboard.

## Phase 4 — Waiver / stash / handcuff intelligence `[x]` DONE
- [x] Handcuff / injury-replacement detector ([src/waivers/handcuffs.py](../src/waivers/handcuffs.py)):
      for each of my **optimal-lineup skill starters** (reuses Phase 3 `optimize()`), finds the
      next-man-up from the Sleeper player map's own `depth_chart_position`/`depth_chart_order` (already
      Sleeper-keyed → zero ID friction; the named backup *is* an addable `player_id`). Walks past
      rostered backups to the first **free agent** below my starter (records the `gap`); **URGENT**
      when my starter is Q/D/O, else HIGH. K/DEF skipped.
- [x] Reverse-standings **priority** spend advice ([src/waivers/priority.py](../src/waivers/priority.py)):
      "startable upgrade" measured by re-solving `optimize()` with the candidate added
      (`lineup_gain = new_total − base_total`, as PROGRESS anticipated). Verdict `spend | stream-later
      | hold`, tempered by **priority scarcity** (selective near the top of the standings where a top
      claim is scarce/slow to recover, aggressive near the bottom — [src/waivers/league.py](../src/waivers/league.py),
      reads `settings.waiver_position`) and by **contention** from Sleeper trending-add velocity.
- [x] Playoff stash ranker ([src/waivers/stash.py](../src/waivers/stash.py)) over a **self-computed
      SOS in OUR scoring** ([src/waivers/sos.py](../src/waivers/sos.py)): season-to-date nflverse
      actuals re-scored via the Phase 1 engine, aggregated to **points-allowed-per-position per
      defense**, normalized to a per-position multiplier (league avg = 1.0). Playoff value = per-game
      baseline × each W15/16/17 opponent's multiplier, summed (reports raw + SOS-adjusted + per-week
      breakdown). Plus bye-week stash suggestions for my starters.
- [x] Usage enrichment ([src/waivers/usage.py](../src/waivers/usage.py)): recent snap share (+trend),
      target/rush share, and expected fantasy points (generic-model, labelled) from
      `load_snap_counts` / `load_ff_opportunity`, joined to Sleeper via the `pfr_id` / `gsis_id`
      crosswalk (new generic `ids.build_id_to_sleeper`). Best-effort — never fatal. (Red-zone usage
      deferred: no cheap field in ff_opportunity, pbp out of scope for v1.)
- [x] New nflverse loaders ([src/data/nflverse.py](../src/data/nflverse.py)): `load_snap_counts`,
      `load_ff_opportunity`, `load_depth_charts` (depth charts kept as a *secondary* cross-check —
      Sleeper's per-player depth fields are primary).
- [x] Live glue ([src/waivers/inputs.py](../src/waivers/inputs.py)) + runnable
      ([scripts/waiver_report.py](../scripts/waiver_report.py)):
      `python scripts/waiver_report.py --week N [--season Y]` — handcuff alerts, spend advice (with my
      standings rank + waiver position + posture), playoff stash ranker, bye-week stashes. Read-only.
      Built to run **Tuesday evening CEST** before waivers clear Wed 09:00 CEST.
- [x] Tests: `tests/test_waivers.py` (16 offline unit tests — FA split incl. reserve/taxi, standings +
      scarcity posture, handcuff next-man-up/URGENT/gap-walk/K-DEF-skip, spend verdicts via the **real**
      optimizer, SOS math + team normalization, stash SOS-tilt/bye-skip/sort, bye suggestions, usage
      joins). Full suite **54 passed**.
- [x] **Validated end-to-end against the 2025 league** (Weeks 10 & 15): FA split clean (15 rostered,
      ~12k FAs, no overlap; all handcuff backups genuinely unrostered); handcuffs resolve real backups
      (Achane→Wright URGENT on Q, CMC→Jordan James insurance); SOS league-avg = 1.000/pos with sane
      spread (≈0.6–1.9) and W15–17 opponents correct; PA-by-position re-scored in our settings (DAL
      soft to WR 34.3/gm, PHI tough on QB 16.3 with our 4-pt pass-TD).
- **Note:** as in Phase 3, a *completed* season's weekly projections return ADP-only (zero-stat) rows
  for many rostered non-stars, so historical FA `Δlineup` gains look inflated (QB/K/DEF especially).
  That's the known historical-data quirk, not a logic bug — live in-season runs return full stat lines.
  The waiver report prints UTF-8 explicitly (Windows console is cp1252).
- **Next:** Phase 5 (hosted Streamlit dashboard + GitHub Actions weekly refresh + team-analysis views).

## Phase 5 — Hosted season dashboard + weekly refresh `[ ]` TODO
- [ ] Streamlit Community Cloud dashboard + team-analysis views
- [ ] GitHub Actions weekly cron (Tue evening CEST, UTC best-effort) + `workflow_dispatch`,
      commits updated data so Streamlit auto-redeploys
- **Next:** Phase 6 (optional).

## Phase 6 — (Optional) Monte Carlo draft simulator `[ ]` TODO
- [ ] Forward simulation of draft outcomes — build last, only if useful

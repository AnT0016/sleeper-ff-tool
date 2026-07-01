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

## Phase 5 — Hosted season dashboard + weekly refresh `[x]` DONE
- [x] **Decoupled compute from display:** the dashboard reads a single precomputed SQLite artifact
      ([data_cache/season.db](../data_cache/season.db)) and **never** hits an API on page load. The
      networked "ingest + recompute" pipeline ([src/analysis/snapshot.py](../src/analysis/snapshot.py))
      reuses every earlier phase — `load_lineup_inputs` + `startsit` (Phase 3), `load_waiver_inputs` +
      `spend_advice`/`rank_playoff_stashes`/`bye_stash_suggestions` (Phase 4) — and writes one table per
      dashboard section + a single-row `meta` table via pandas `to_sql` (atomic temp-file + replace;
      empty frames get a placeholder column so `to_sql` never emits invalid SQL).
- [x] **Team-analysis views** ([src/analysis/team.py](../src/analysis/team.py), pure/offline): per-team
      optimal-starter points per slot (reuses the Phase 3 `optimize()` on each of the 12 rosters), my
      **positional strength ranking vs the league** with strength/average/weakness verdicts — computed
      **both** season-long (stable roster quality, all eligible) **and** this-week (bye/injury-aware);
      **bye-week gaps** (🔴 hole vs 🟡 forced-backup); **positional needs** (weakness + thin depth + bye
      holes); **mutual-fit trade targets** (a team strong where I'm weak *and* weak where I'm strong);
      and a **Weeks 15-17 playoff outlook** that reuses the Phase-4 SOS (`playoff_outlook` wraps
      `rank_playoff_stashes` over my likely starters).
- [x] **Hosted dashboard** ([apps/season_app.py](../apps/season_app.py)): offline, reads `season.db`
      only. Tabs — **This Week** (optimal lineup, start/sit, risky flags, holes), **Waivers & Stash**
      (handcuff/spend/stash/bye alerts), **Team Analysis** (strength matrix w/ Season vs This-week
      toggle, needs, bye gaps, trade ideas surfaced before the Week-11 deadline, playoff outlook).
      Caches keyed on the artifact mtime; guards cleanly when no snapshot exists.
- [x] **Runnable refresh** ([scripts/refresh_data.py](../scripts/refresh_data.py)):
      `python scripts/refresh_data.py [--week N --season Y]` — auto-detects the current season/week
      from Sleeper state (off-season-safe) and writes `data_cache/season.db`.
- [x] **GitHub Actions weekly cron** ([.github/workflows/refresh.yml](../.github/workflows/refresh.yml)):
      `cron: 0 18 * * 2` (Tue 18:00 UTC ≈ Tue evening CEST/CET, before Wed 09:00 CEST waivers; UTC
      best-effort) + `workflow_dispatch` with optional `week`/`season` inputs for off-season/ad-hoc
      runs. `permissions: contents: write`, a `concurrency` guard, and a **no-op-safe** commit step
      (`git diff --staged --quiet ||` commit+push) so empty diffs don't fail. Streamlit auto-redeploys
      on the commit.
- [x] Tests: [tests/test_analysis.py](../tests/test_analysis.py) (8 offline unit tests — slot points,
      strength ranking + verdicts, bye hole/thin + `from_week` filter, needs fold, mutual-fit trades,
      playoff SOS tilt). Full suite **62 passed**.
- [x] **Validated end-to-end** (2025 Week 10): `refresh_data.py` builds `season.db` from cached data;
      dashboard verified headless via Streamlit `AppTest` (3 tabs, 14 tables, metrics correct, Season↔
      This-week toggle re-runs clean). Sanity: my team ranks RB #2/TE #2 (strengths), QB #10/WR #11
      (weaknesses); QB flagged a Week-10 bye hole (Dak on bye); 3 mutual-fit trade partners surfaced;
      playoff total 260.83 with per-week SOS multipliers. `ruff` clean.
- **Connect to Streamlit Community Cloud:** see [apps/README.md](../apps/README.md) — public repo, no
      secrets, free tier; point a new app at `apps/season_app.py` on `main`. Deployed at
      `sleeper-ff-tool` (public). Hosting note: Streamlit Cloud installs from `requirements.txt`
      (hosted-app deps only — `streamlit`/`pandas`); `pyproject.toml` carries `[tool.poetry]
      package-mode=false` / `[tool.uv] package=false` so a host never tries to build the src/ repo.
- **Bonus — 2025 season backtest** ([src/analysis/backtest.py](../src/analysis/backtest.py),
      [scripts/backtest.py](../scripts/backtest.py) → `data_cache/backtest.db`, "📈 2025
      Backtest" tab): a completed-season "what if I'd used this tool" review. Scores every lineup by
      **real** results (`matchups.players_points`, never trusting past-week projections): **weekly**
      actual vs hindsight-optimal (points left on bench) vs the projection-lineup-scored-by-actuals;
      **draft** replay (VOR-greedy at my real pick slots vs my actual picks, graded by full-season
      points); **season summary** (actual vs optimal vs tool totals/records, recomputed "always-
      optimal" standings); **league-wide** weekly ranks. Pure helpers (`simulate_draft`,
      `lineup_from_points`, `optimal_standings`) unit-tested; full suite **66 passed**. Validated on
      2025: actual 1996 / optimal 2307 / tool 1977 pts, 11-3 → optimal 12-2, 311 bench pts lost; the
      VOR draft would have added ~436 season points.
- **Backtest "full-draft" views (follow-up):** three more retrospective views on real data, all from
      pieces `build_backtest` already fetched (pure helpers `draftboard_rows` / `matchup_detail_rows`
      / `transaction_rows` / `starting_slot_labels`, unit-tested in `tests/test_analysis.py`):
      **🗂️ Draft board** — the real snake draft as a 12-team round×slot grid, each pick graded by
      full-season points, my column starred; **⚔️ Matchups** — a week selector showing my whole
      starting lineup vs my opponent's, slot-by-slot, scored by real points (W2 head-to-head sums
      reconcile exactly to the weekly table: 116.60 vs 127.96); **🔁 Transactions** — the season's
      completed adds/drops/trades (one row per roster per move; note these are *already* reflected in
      the weekly lineups — the view just surfaces them). New tables in `backtest.db` (`draftboard`,
      `matchup_detail`, `transactions`) + `my_draft_slot`/`n_transactions` meta. Validated on 2025
      (168 board rows, 153 matchup rows, 267 transactions) and headless `AppTest` (19 tables, week
      selector + transaction filter exercised).
- **Next:** Phase 6 (optional Monte Carlo draft simulator).

## Phase 6 — (Optional) Monte Carlo draft simulator `[x]` DONE
New package [src/draftsim/](../src/draftsim/) — a forward Monte Carlo sim of full 12-team snake drafts.
**Directional draft-prep aid, not a core feature**: output is explicitly labelled heuristic, and every
variance / ADP / injury assumption is printed alongside the numbers so they can be judged. Read-only;
reuses every earlier phase.
- [x] **Decisions ex-ante, evaluation ex-post** ([engine.py](../src/draftsim/engine.py)): each sim the
      11 bots draft by *market* half-PPR ADP + noise and *I* draft by our *custom* VOR under a chosen
      build; then each player's **season outcome is sampled** (not known to any drafter) and my
      resulting roster's best legal lineup is scored on those samples. Aggregated over thousands of
      drafts → a **distribution** of my season outcomes per build, not a point estimate. All builds
      share the same sampled outcomes + noisy-ADP draws (**common random numbers**) so differences
      reflect the strategy, not sampling noise.
- [x] **Outcome + injury model** ([distributions.py](../src/draftsim/distributions.py)): season points
      ~ **lognormal** (mean = our custom-scored projection; non-negative, right-skewed) with a
      per-position **CV** (QB .18 / RB .32 / WR .30 / TE .35 / K .20 / DEF .28). Injuries = one
      **significant setback** per season (Bernoulli per position; RB .45/4g … DEF .02/1g) → an
      availability haircut on the season total. All knobs are heuristic, **not fitted** (stated in the
      report).
- [x] **ADP bots** ([bots.py](../src/draftsim/bots.py)): one shared noisy-ADP order per sim
      (`adp + N(0,8)` picks), each bot taking the lowest available it has room for (caps: QB≤2, TE≤2,
      K/DEF≤1, …; K/DEF only in the last ~22% of rounds). Bots using market half-PPR ADP while I use our
      4-pt-passing-TD VOR is the **edge being quantified**, borrowing the joewlos "will my target
      survive?" idea.
- [x] **My strategies** ([strategy.py](../src/draftsim/strategy.py)): `best_vor`, `balanced`,
      `rb_early`, `hero_rb`, `zero_rb` — each a per-round position-preference resolved by best **VOR**.
      Shared guardrails: an end-of-draft **mandatory guard** that force-fills QB/K/DEF + the
      RB/WR/TE/FLEX minimums so no build ever ends unable to field a legal lineup, **K/DEF never offered
      early** (only taken by the guard — matches the "stream K/DEF" rule), and QB/TE capped at their
      starting slot (you stream those, not stash a backup) so picks flow to RB/WR depth.
- [x] **Fast lineup valuation** ([lineup.py](../src/draftsim/lineup.py)): greedy fill (optimal for a
      single FLEX) scores rosters on sampled points without PuLP in the hot loop; `select_starters`
      separates starters from bench depth for the injury report.
- [x] **Live glue** ([inputs.py](../src/draftsim/inputs.py)): pulls live `scoring_settings` +
      `roster_positions`, builds the custom-scored board (Phase 1 engine) + VOR with data-driven FLEX
      (Phase 2), and resolves my slot — **revealed `draft_order`** if out, else `--slot`, else a
      middle-slot fallback (CLAUDE.md keeps prep slot-agnostic until the slot is revealed).
- [x] **Runnable** ([scripts/draft_sim.py](../scripts/draft_sim.py)):
      `python scripts/draft_sim.py [--slot N --sims 2000 --season Y --strategies a,b]`. Prints (1) the
      **build comparison** (p10/median/p90 of my season points + mean finish rank, P(top-3), P(win),
      typical composition; recommendation by best *finish distribution*, not best EV), (2) **target
      survival** — P(each top-VOR target is still on the board) at each of my picks, and (3) **injury
      insight** — which likely starters carry real durability risk *and* lack a rostered backup (where
      you need a true handcuff vs. a streamer). `--board` also prints a **representative simulated
      draftboard** (all 12 teams, Sleeper-style round×slot grid) — the *median-outcome* sim for the
      recommended build, reconstructed deterministically via `engine.representative_draft`. Numpy added
      as an explicit dependency.
- [x] **Pool tidy:** `build_pool` keeps the union of top-N by ADP + top-N by VOR + all K/DEF (~470 for
      2025, vs an unbounded pool before), so `--pool-size` is meaningful and the run stays light.
- [x] Tests: [tests/test_draftsim.py](../tests/test_draftsim.py) (16 offline unit tests — lognormal
      mean-preservation, bounded availability + risk tracking, FLEX-takes-best-leftover, mandatory
      fill, K/DEF-never-early, zero-RB avoidance, bot caps, end-to-end synthetic run (legal rosters,
      survival decreasing with later picks, common-random-numbers reproducibility), plus the
      representative-draft reconstruction + board rendering). Full suite **86 passed**; `ruff` clean.
- [x] **Validated end-to-end** against the 2025 league (slot 7 read from `draft_order`, ~4s for 2000
      sims × 5 builds): survival flags match intuition (elite RB/WR ~30–45% to survive to pick #7, gone
      by the #18 turn; QBs slide to the late rounds under our 4-pt-pass-TD scoring); builds rank
      `rb_early`≈`balanced`≈`best_vor` (mean rank ~3.1) clearly ahead of `zero_rb` (~4.2) — i.e. punting
      RB is costly when the field drafts by standard ADP — and the injury insight flags RB/WR starters
      lacking rostered depth.
- **Limitation (stated):** VOR-greedy builds skew WR-heavy and don't value handcuff *correlation* or
  lottery upside; treat builds as a starting point, and re-run with `--slot` once the real slot is out.

## 2026-readiness (follow-up) `[x]`
Prep so the new 2026 league is a config change, not a code change. See
[docs/2026_SETUP.md](2026_SETUP.md) for the full run checklist.
- [x] **Season/league-agnostic tooling:** `scripts/backtest_2025.py` → [scripts/backtest.py](../scripts/backtest.py)
      (git-mv, history preserved), now **defaults to the current Sleeper season** (like `draft_sim.py` /
      `optimize_lineup.py`); safe to run mid-season (only finished weeks included). References updated
      (app message, apps/README, data_cache/README, this log); the dashboard tab is now season-agnostic
      ("📈 Backtest").
- [x] **Sleeper-style lineup cards** ([apps/season_app.py](../apps/season_app.py)): the *This Week*
      optimal lineup renders as cards (slot badge · Sleeper-CDN headshot/▣ DST logo · player · proj),
      with the detailed table kept in an expander. Needed `player_id` added to the `lineup` snapshot rows.
- [x] **Kickoff day/time per starter** ([src/analysis/snapshot.py](../src/analysis/snapshot.py)
      `kickoff_by_team`): each starter card/row shows its game (e.g. `Sun 16:25 vs LAR`) from the
      nflverse schedule (ET; LA→LAR normalized; best-effort). The optimizer still maximizes projected
      points — kickoff time is **display-only**, for manual FLEX hedging at lock time.
- [x] **Setup checklist:** [docs/2026_SETUP.md](2026_SETUP.md) — config (new `LEAGUE_ID`), mock-draft
      rehearsal, prep→draft→in-season→retrospective run order, refresh cron, quick-reference table.
- [x] **Off-season-safe weekly cron** (`snapshot.offseason_skip_reason` + `scripts/refresh_data.py`):
      a *scheduled* refresh now **no-ops with exit 0** (green, no commit) when Sleeper isn't in a regular
      season — previously it auto-detected the rolled-over 2026 season and crashed on the nflverse
      `stats_player_week_2026.parquet` **404** (no data until games are played). An explicit
      `--week`/`--season` still forces a run (backfill); it resumes on its own once `season_type` flips
      to `regular`. Unit-tested.
- [x] Validated: `season.db` rebuilt (2025 W10) with `player_id`/`kickoff` populated (kickoffs internally
      consistent — Achane MIA `vs BUF` ↔ DJ Moore BUF `@ MIA`); dashboard verified headless via `AppTest`
      (card view renders, no exceptions). Full suite **86 passed**; `ruff` clean on changed files.

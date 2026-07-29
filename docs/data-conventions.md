# Data conventions (Phase 8 — the lake + modeling frame)

The point-in-time discipline for the historical dataset ("the lake") and the training frame assembled
from it. **Read this before writing or modifying any collection, feature-engineering, or model
code** — every convention here exists to keep hindsight out of a dataset whose entire value is that it
records *what was known before the outcome existed*. The same rules are packaged as an auto-loading
skill at [`.claude/skills/data-conventions/SKILL.md`](../.claude/skills/data-conventions/SKILL.md).

**Authority chain.** The live Sleeper API is the source of truth for scoring and roster settings
(see [CLAUDE.md](../CLAUDE.md)); the **code** is the source of truth for how a row is collected,
stored, deduped and assembled; **this doc explains and points at that code, it does not restate it.**
Where a section names a module, that module (and its tests) is authoritative and this doc is the map.
The overall design and every decision behind it live in
[docs/plans/data-collection.md](plans/data-collection.md).

---

## 1. Timezone — everything is UTC

Every `_captured_at` and every derived instant in the lake is **ISO-8601 UTC**. Game-time and lock
reasoning converts to ET only at the edges (schedule parsing, the human-readable kickoff strings on
the dashboard). A naive datetime is a bug, not a UTC value: the completed-week resolver
(`collect.runner._ensure_utc`) *raises* on one rather than reading a local clock as UTC, because this
project's own timezone (CEST) is two hours ahead and two hours is enough to settle a week that is
still being played.

## 2. The lookahead rule (the whole value)

A feature used for week *N* may only use data that was knowable **before week *N*'s first game
kicks off**. The gate is one function — [`dataset.assemble.lookahead_ok`](../src/dataset/assemble.py)
— applied to every source, and it admits a row on **either** of two rules:

- **content rule** — the row is about a strictly earlier week (`feature_week < N`). A backfilled
  week-3 actual is legal as a week-5 feature however late it was captured. This is what keeps the
  lagged usage features (`_lagged_usage`) legal.
- **capture rule** — the row was known strictly before week *N*'s lock (`known_at < lock_utc`). This
  is what admits a genuine pre-lock snapshot of week *N* itself (a projection, an injury report, a
  betting line).

Anything else — including an unreadable or missing timestamp on a row that is not about an earlier
week — is **refused. The gate fails closed**, because a leak is invisible in the output and a dropped
feature is not.

**One lock per `(season, week)`**, the first kickoff of the target week — deliberately stricter than a
per-player lock, so no same-week capture can carry a post-kickoff fact about *any* game that week.

**Why `known_at` is not always `_captured_at`.** Every 2016–2025 row was written by one backfill run,
so its `_captured_at` is a 2026 instant. Applied literally, the capture rule would then admit *nothing*
same-week for the whole training span. So a row's known-at instant is **resolved per source** from
`registry.Source.content_known` (`assemble._resolved_known_at`), and only for backfilled rows:

| `content_known` | resolved `known_at` for a backfilled row |
| --- | --- |
| `pre_kickoff` | its own week's lock minus a lead — the content existed before that lock by construction (a practice report, a closing line) |
| `post_game` | never (`NaT`) — the content did not exist until the week was played, so only the *content* rule can admit it |
| `row_timestamp` | the row's own event stamp (`nflverse_depth.dt`), a real as-of time |

`content_known` is **not** `cadence`. `nflverse_schedules` runs on the pre-lock cadence *and* carries
the final `result`/scores, so deriving knowability from the capture schedule would hand the assembler
the label — it is registered `post_game` and read as a **calendar only**; its sanctioned pre-game view
is `vegas_odds`. See the `Source.content_known` docstring in
[`src/collect/registry.py`](../src/collect/registry.py) for the full rationale, and the module
docstring of [`src/dataset/assemble.py`](../src/dataset/assemble.py) for how the gate is applied
column by column (including the separate `observed_weather_ok` guard for at-kickoff weather).

## 3. The label

`y_custom_points` is week-*N* **actuals re-scored in the league's live `scoring_settings`** with the
Phase 1 engine (`scoring.engine.points`) — never a hard-coded coefficient. It is a **union of two
sources**, because `nflverse_player_week` has zero DEF rows and DEF is a starting slot:

- **skill players + kickers** — `assemble._player_labels`: `nflverse_player_week` actuals translated
  through `data.ids.nflverse_to_sleeper_stats`, then scored.
- **team defenses** — `assemble._dst_labels`: `sleeper_stats_week`, which is already Sleeper-keyed and
  scores directly with no stat translation.

The **baseline to beat** (`baseline_sleeper_points`) is Sleeper's own `sleeper_proj_week` re-scored in
the same scoring — it exists **forward from 2026 W1 only** (the endpoints serve only the latest
values; no ex-ante historical projection is recoverable — Decision #6).

**Row universe is a selection effect, not a leak.** A row exists only where the player recorded a stat
line that week, so the frame is a sample of *player-weeks in which the player played*. A model fitted
on it estimates points **conditional on appearing** — start/sit consumers who need an unconditional
expectation must model availability separately (the injury columns and `games_played_prior` are kept
partly so that stays possible). Documented in full in the `assemble.py` module docstring.

## 4. Join keys

| Player kind | Join on | Notes |
| --- | --- | --- |
| Skill / K | `gsis_id` | nflverse's key; mapped to Sleeper via the crosswalk |
| Team defense (DST) | **team abbreviation** | in Sleeper's era-correct vocabulary (`sleeper_stats_week.player_id` *is* the team code) |
| Everything, in Sleeper space | `sleeper_id` | the frame's `player_id` is a Sleeper id |

The bridge from nflverse ids to Sleeper's is the `id_crosswalk` source (ffverse's `load_ff_playerids`,
keyed on `mfl_id`). Two hazards it hides, both handled in `assemble._canonical_id` / `_map_ids`:
ffverse stores `sleeper_id` as a **float** (`13269.0`), which joins against nothing Sleeper emits (the
trailing `.0` is stripped); and franchise codes disagree across feeds (`SD`/`OAK`/`LA`/`LAR`),
normalized by `_TEAM_ALIASES`. **Every row that fails to join is counted and logged, never silently
dropped** — the project-wide "log every projection row that fails to join" rule — up to a per-source
ceiling above which the warning escalates (`_UNJOINED_CEILING`).

## 5. Sources: forward-only vs backfillable

**`backfillable` is the honest answer to "can we recover this for past seasons?"** Sleeper's projection
endpoints serve only the *latest* numbers, so the `sleeper_proj_*` sources are **forward-only** — they
start accumulating at 2026 W1 and no work recovers 2016–2025. Everything else is recoverable from
nflverse releases today; some only from a start year, recorded in `Source.backfillable_from`
(`nflverse_depth` is 2025+ — the pre-2025 feed has no clean key).

The **authoritative source table is [`src/collect/registry.py`](../src/collect/registry.py)**, pinned
by [`tests/test_registry.py`](../tests/test_registry.py). It is the single source of truth for every
column — `grain`, `key_cols`, `cadence`, `dedup`, `content_known`, `backfillable`,
`backfillable_from` — consulted by the collectors, the runners and the assembler alike. **Do not
restate that table anywhere**; adding a source is one entry there plus a collector. The list below is
a reading aid only (name + purpose); the registry is authoritative for everything else:

- `sleeper_proj_week` — Sleeper weekly projections, captured pre-lock. **Forward-only. THE reason this
  phase exists** — unrecoverable once the week is played.
- `sleeper_proj_season` — Sleeper season-long projections (forward-only); in-season drift is itself a
  signal.
- `sleeper_stats_week` — Sleeper's own weekly actuals; the DST/K cross-check where nflverse has no
  team-defense aggregate. Also the DST half of the label.
- `nflverse_player_week` — weekly player actuals; the skill/K half of the label.
- `nflverse_snaps` — snap counts (PFR-keyed; no `gsis_id`).
- `nflverse_ff_opp` — expected fantasy points + volume shares (same-week by construction, so only ever
  used lagged).
- `nflverse_injuries` — the weekly practice/game-status report; point-in-time (Thursday *Questionable*
  → Sunday *Out* is a revision stream, kept).
- `nflverse_schedules` — the schedule, plus closing Vegas lines and observed weather. Read by the
  assembler as a **calendar only** (kickoffs, matchups, locks).
- `nflverse_depth` — time-stamped depth-chart snapshots (2025+). `dedup=first_capture` — the only
  source that departs from the default (see §6).
- `id_crosswalk` — ffverse's player-id master; the nflverse→Sleeper bridge (§4).
- `vegas_odds` — implied team totals derived from schedules; the **sanctioned pre-game market view**
  (carries the lines, deliberately not the outcome).
- `weather` — open-meteo forecast pre-lock + venue dome flag; historical temp/wind from schedules on
  backfill.

## 6. Per-source dedup

How the store collapses repeat captures of one key is declared per source
(`Source.dedup`) and resolved by `store.lake.write_snapshot`. The two policies, and the full rationale
(including the immutability cost of `first_capture` and why the policy is **declared, not derived**
from "does the key hold a timestamp"), are documented once in the **per-source dedup section of
[docs/plans/data-collection.md](plans/data-collection.md)** (Decision #15) and the `Source.dedup`
docstring in [`src/collect/registry.py`](../src/collect/registry.py). In one line each:

- **`per_capture_date`** (default) — keep the latest row per key **per UTC capture date**, so a
  later-day capture is a new point-in-time snapshot. Right for anything the provider can revise in
  place (a corrected stat line, an injury report firming up through the week).
- **`first_capture`** — keep the **earliest** capture of each key, ignoring capture date. Right only
  for a source whose natural key already carries its own observation timestamp, so a row is immutable
  once seen. **`nflverse_depth` alone** uses it (see the gotcha in §9, fact 2).

## 7. Cadence and the crons

`cadence` is *when a capture runs*: **`prelock`** (Thu ~22:00 UTC before the TNF lock + Sun ~15:00 UTC
before the 1pm ET main slate — the point-in-time snapshots), **`postgame`** (Tue ~12:00 UTC, finalized
actuals/usage, before the 18:00 UTC `season.db` refresh), and **`backfill`** (the one-time historical
pull). The workflows are
[`.github/workflows/collect-prelock.yml`](../.github/workflows/collect-prelock.yml) and
[`collect-postgame.yml`](../.github/workflows/collect-postgame.yml) (Ticket #6); each file's header
comment carries its own design notes and is not repeated here.

**The postgame week is resolved from the schedule, never from Sleeper's `state.week`.** Sleeper
advances `state.week` to the *upcoming* week early Tuesday, racing the cron; a run landing after the
flip would file a zeroed not-yet-played snapshot into `week=N+1` and never capture week *N*. Instead
`collect.runner.latest_completed_week` / `_postgame_plan` take the highest REG week whose games have
all *finished* (`kickoff + 6h <= now`) — a fact about the NFL, not about Sleeper's clock (Ticket #16).
Read those two functions for the exact rule.

**These crons are configuration, not code, and their contract is held by
[`tests/test_workflows.py`](../tests/test_workflows.py).** Two of its assertions are **cross-file** and
would silently break under a plausible single-file edit — if you touch a workflow, that test is what
tells you why:

- **One shared `concurrency` group across both files** (`group: lake-capture`). `write_snapshot` is
  read-modify-write on a partition, so mutual exclusion is the only thing stopping a manual dispatch
  from losing an update to a scheduled run. It holds *only* because the two files spell the same
  literal string — renaming one file's group passes every single-file review and breaks the invariant.
- **Nothing pinned on the scheduled path** — no literal `--season`/`--week`; both may reach
  `collect.py` only through the empty-on-schedule `workflow_dispatch` inputs. A hardcoded `--week`
  (the change someone debugging a flaky cron reaches for) would fix every Tuesday to one number
  forever and undo the schedule-based resolution of §7.

## 8. Storage backends & Backblaze B2

All lake I/O goes through a `StorageBackend` protocol so *where* is a config flip, never a rewrite:
`LocalParquetBackend` (`data_cache/lake/`, gitignored — the dev/default) and `S3Backend` (Backblaze B2
today — the production store the crons write to), selected by `LAKE_BACKEND={local|s3}`. Credentials
(`LAKE_S3_*`) live in GitHub Actions secrets, **never the repo**. The complete backend behaviour, the
one-time owner account/bucket checklist, and the credential table are in
**[docs/b2-setup.md](b2-setup.md)** — not duplicated here.

## 9. Facts that lived only in PR bodies and issue comments

Written down because nothing else in the tree records them, and each is a trap a future session would
otherwise re-derive:

1. **GitHub disables a scheduled workflow after 60 days of no repository *activity*** — pushes and
   commits, *not* workflow runs. This repo goes quiet roughly February–August, so the capture crons
   would auto-disable **exactly when the season starts**. Prevention and recovery are two different
   actions and one does not substitute for the other:

   - **Prevent.** Any push to `main` inside the 60-day window resets the clock. GitHub emails a
     warning before it disables anything.
   - **Recover — required once a workflow is already disabled; a push will *not* re-enable it.**

     ```bash
     gh workflow enable "Collect — pre-lock snapshot"
     gh workflow enable "Collect — post-game snapshot"
     gh workflow list --all     # confirm both read "active"
     ```

     `--all` matters: a disabled workflow drops out of the default listing, which is exactly how you
     would fail to notice.

   Do this as part of the annual league swap, before Week 1, so the first pre-lock capture — the one
   nothing can recover — is not silently skipped.
2. **Under `dedup=first_capture`, `nflverse_depth`'s `lake_inventory().latest_captured_at` means "the
   first observation of the newest key", not "when the partition was last written".** `first_capture`
   keeps the *earliest* `_captured_at` per key, so the max over the partition is the first-seen time of
   whichever key appeared most recently — not the last write. Do not read it as a freshness/"last
   updated" timestamp for depth charts (Ticket #15).
3. **A postgame run that cannot resolve a completed week is a GREEN skip — `collect.py` prints
   "Skipping …" and exits 0, by design (Ticket #16).** Schedule unavailable, or no REG week finished
   yet ⇒ skip, not a guess and not a failure. Every postgame source is backfillable, so a missed run
   is recovered with `--week`, whereas a wrong `week=N+1` write is silent contamination. `test_workflows.py`
   deliberately adds no grep-for-"Skipping" to the crons, precisely so this correct no-op is never
   misread as red. **Do not alert on it.**
4. **`sleeper_stats_week`'s 2016 duplicate-game artifact is collapsed at collect time** (Ticket #21):
   the feed emits **one whole game twice** — every player of both teams including the DST, under two
   `game_id`s (`2016<1><WW>00` and `...29`) with byte-identical stats — in **8 weeks of 2016 only**
   (2, 4, 6, 9, 10, 13, 15, 17), with **zero occurrences 2017+**. It is an id-assignment artifact, not
   a stat correction: `last_modified` is null on every row and the stats never disagree. The collector
   merges such duplicates **all-or-nothing per key** *only when the listings agree*. **A key whose
   listings disagree is left untouched on purpose**, so `dedupe_rows` (`collect.base`) still emits its
   warning rather than silently picking a winner. Don't "finish the job" by force-collapsing
   disagreeing keys — the surviving warning is the signal. And a lone duplicated player-week is **not**
   this artifact; treat it as new.
5. **`_backfill=False` means "written by a cadence run", NOT "observed contemporaneously".**
   `run_cadence` (both prelock and postgame via `scripts/collect.py`) always stamps `False`; only
   `run_backfill` stamps `True`. So a **postgame recovery re-run with an explicit past `--week`** — the
   documented fix named in #16's stepped-over-week WARNING — stamps `False` even while reconstructing a
   decade-old season. This is **inert for correctness**: those sources are `content_known="post_game"`,
   for which `assemble._resolved_known_at` returns `_captured_at` (the honest re-run time) and the
   assembler admits them as features only via the *content* rule (lagged) anyway. But the marker's
   *name* invites the wrong reading — do not treat `_backfill=False` as proof a row was captured live
   at the time. Verified 2026-07-29 on Actions run 30428096751; see the comments on Ticket #6.

## 10. Cron verification status (as of 2026-07-29)

Recorded so next July nobody re-derives it:

- **Both workflows dispatch green from `main`.** The four `LAKE_S3_*` secrets resolve and the
  `LAKE_BACKEND=s3` guard passes.
- **Postgame's write path to B2 is proven end to end** — a real dispatch wrote **6/6 sources,
  59,998 rows to `s3://`**.
- **Prelock's write path is NOT yet proven.** It has only ever taken the off-season skip, and it
  **cannot be tested out of season**: a prelock run with an explicit past week would file *today's*
  projections into that week's partition (the forward-only hazard — the projection endpoints serve
  only the latest values). **First real proof is the Thursday before Week 1** — check that run
  explicitly rather than assuming symmetry with postgame.

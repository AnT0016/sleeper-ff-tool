# Spec: Cloud data collection & feature store (Phase 8)

Status: **draft** — awaiting your sanity-check before issues are created
Issues: _TBD (one per ticket below, created after this spec is approved)_
Owner: you (EM/PM) · Planner: Claude (this session)

## Goal
Stand up a **point-in-time, lookahead-free historical dataset** ("the lake"), collected automatically by
a cloud cron, so we can start building our **own** models. Every downstream tool today consumes
Sleeper's projections re-scored into our scoring; there is no in-house model and no training data.
The single most valuable label — *what was projected before lock vs. what actually happened* — is
being discarded every week we don't capture it (Sleeper's projection endpoints only ever return the
**latest** value; historical weekly projections are unrecoverable). This phase captures and accumulates
that data with zero hindsight contamination, and assembles it into a modeling table on demand.

**This phase ships the data foundation only. The models themselves are the next phase** — but the
feature-store schema is shaped now to serve all three intended targets (see Non-goals).

## Non-goals
- **No model training in this phase.** No regression/classifier/distribution fitting yet. We only
  collect, store, and assemble; a `build_training_frame(...)` is the hand-off point.
- **No new prediction surface** in the dashboard/CLIs. Existing tools keep consuming Sleeper
  projections unchanged until a model is built and validated.
- **Cloud object storage (Cloudflare R2) is the production store**, behind a `StorageBackend` protocol;
  local git-committed parquet remains the dev/default backend (flip via one env var). No hosted DB.
  (Decision log #1.)
- **No writes to Sleeper**, ever (immutable rule). Collection is read-only GET only.
- **No secrets in the repo** (immutable rule). All v1 data sources are keyless: Sleeper (no auth),
  nflverse (public releases), open-meteo (no key). A paid odds provider is out of scope for v1. The
  **only** credentials introduced are the R2 S3 keys, and they live in **GitHub Actions secrets**, never
  the repo (the playbook explicitly permits Actions/Streamlit secrets).
- Not replacing `season.db` / `backtest.db` or the existing `refresh.yml`; the lake is additive.

## Intended model targets (why the schema looks the way it does)
You chose "all three." The schema must therefore support, without reshaping later:
1. **Weekly custom-scored point projections** — regression at player×week grain; baseline to beat =
   Sleeper's own weekly projection re-scored in our scoring.
2. **Fitting the sims' distributions** — the per-position CVs and injury knobs in `draftsim`/`seasonsim`
   are currently heuristic ("not fitted"); needs multi-season weekly actuals + availability history.
3. **Breakout / waiver classifier** — needs usage trajectories (snap %, target/rush share, expected
   points) and role/depth-chart movement, with a forward label (did role/production step up).

All three are served by one **player × week** table with point-in-time features and a real re-scored
label, plus availability, role, market and weather columns.

## Design

### Layered lake (raw → assembled)
```
lake/                                    (append-only, point-in-time; same key layout on both backends)
  <source>/season=<YYYY>/<source>_<YYYY>_wk<WW>.parquet   # weekly-grain sources
  <source>/season=<YYYY>/<source>_<YYYY>_season.parquet   # season-grain sources
```
Raw layers are stored **as-is** (scoring-agnostic, provider-native columns) so nothing is baked in
prematurely. Every row carries reserved columns: `_source`, `_season`, `_week` (nullable for
season-grain), `_captured_at` (ISO-8601 UTC). The assembler joins raw layers into a modeling frame on
demand — the lake never stores a "leaked" join.

### Two storage backends, one interface (Decision #1)
All lake I/O goes through a `StorageBackend` protocol so the *where* is a config flip, never a rewrite:
- **`LocalParquetBackend`** — writes under `data_cache/lake/` (gitignored). The **dev/default** backend;
  used by local runs, tests, and anyone without R2 creds. No secrets, no account.
- **`R2Backend`** — Cloudflare R2 (S3-compatible), the **production/cloud** store the cron workflows
  write to. Free tier (10 GB storage, 1M Class A + 10M Class B ops/mo, zero egress) is far above this
  project's needs (~<1 GB for a decade). Creds come from env (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`) set as **GitHub Actions secrets** — never committed.

Backend chosen by `LAKE_BACKEND={local|r2}` (default `local`). R2 is the source of truth for captured
point-in-time data; a local run can read from R2 (with read creds in the local env) or work off a local
backfill. **Manual, one-time, done by the owner (cannot be automated — account/payment):** create a
Cloudflare account → enable R2 (a card is required for verification even on the free tier; no charge
within limits) → create a bucket → generate an S3 API token → paste the 4 values into repo Actions
secrets. Until then, `LAKE_BACKEND=local` runs everything, so no ticket is blocked on R2 existing.

### Data flow
```
                         ┌─ collect_sleeper_*   ┐
GitHub Actions cron ──▶  ├─ collect_nflverse_*  ├──▶ store.write_snapshot(...)  ──▶  data_cache/lake/**   ──▶ git commit
 (pre-lock / post-game)  ├─ collect_market_*    │        (append + dedup,                                    (Streamlit-
                         └─ collect_weather_*   ┘         atomic temp+replace)                                irrelevant)
                                                                                          │
scripts/backfill_lake.py (one-time, workflow_dispatch) ───────────────────────────────────┘
                                                                                          │
dataset.build_training_frame(seasons, asof=...) ◀── reads lake ── enforces lookahead guard ┘  ──▶ (next phase: models)
```

### Point-in-time discipline (the whole value)
- **Timezone:** all `_captured_at` in **UTC**. Game/lock reasoning converts to ET at the edges only.
- **The rule:** a feature used for week *N* may only use data with `_captured_at` **before week *N*'s
  first game lock**. Backfilled nflverse actuals for week *K* are labelled `_week=K` and are legal as a
  feature only for weeks `> K`. The assembler enforces this; a unit test feeds a row that *would* leak
  and asserts it is excluded.
- **Snapshots vs. finals:** point-in-time sources (Sleeper weekly projections, injuries, odds, weather
  forecast) may be captured multiple times; each distinct capture-date is kept (so projection drift is
  observable). Finalized sources (completed-week actuals) converge to one row per key.
- **Forward-only vs. backfillable:** Sleeper weekly/season projections are **forward-only** (capture
  starts 2026 Week 1). Actuals, snaps, opportunity, injuries, schedules, Vegas lines and historical
  weather **are backfillable now**.

### Projection baselines: none are backfillable — and that's fine (Decision #6)
No **ex-ante** weekly projection is freely/cleanly recoverable for past seasons (Yahoo is OAuth-gated
with no historical projection feed; ESPN isn't archived; FantasyPros/ffanalytics are personal-use and
serve only *current* numbers; nflverse `ff_opportunity` "expected points" is computed from a week's
*actual* usage, so it is a same-week quantity that would **leak** as a pre-game projection — legal only
lagged). Consequence for the schema:
- **Training (2016–2025)** uses only backfillable, lookahead-safe features (lagged actuals, snap/target/
  rush share, expected-points *lagged*, Vegas, weather, depth/role); the label is the real re-scored
  week-N points. No historical projection is required to train.
- **The "beat the market" baseline** (model vs. Sleeper's own projection) exists **forward, from 2026
  Week 1**, once `sleeper_proj_week` capture begins — the honest out-of-sample grade, matching the
  frozen-2026 approach.
- **Future forward sources** (ESPN's undocumented API, an ffanalytics ensemble) are cheap to add later
  via the registry, for a multi-source projection ensemble — explicitly deferred, not designed out.

## Interface contracts

### Storage — `src/store/lake.py`
```python
LAKE_ROOT: Path  # data_cache/lake

RESERVED = ("_source", "_season", "_week", "_captured_at")

def snapshot_path(source: str, season: int, week: int | None = None) -> Path
    # week is not None -> .../<source>_<season>_wk{week:02d}.parquet
    # week is None     -> .../<source>_<season>_season.parquet

def write_snapshot(
    source: str,
    season: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    captured_at: str,             # ISO-8601 UTC, passed in (never Date.now() inside — testability)
    week: int | None = None,
    key_cols: Sequence[str],      # natural key of a row within this source (excl. reserved)
) -> Path
    # 1. build DataFrame from rows; attach reserved cols.
    # 2. merge with any existing partition file.
    # 3. dedup: keep the latest _captured_at per (key_cols + capture_date),
    #    where capture_date = _captured_at[:10]. (Same-day re-run is idempotent;
    #    a later-day capture of the same key is retained as a new point-in-time snapshot.)
    # 4. write atomically (temp file + os.replace).
    # Returns the path written. Empty rows -> no-op, returns the path (no file created).

def read_snapshot(source: str, season: int, week: int | None = None) -> pd.DataFrame  # empty if absent
def read_source(source: str, seasons: Iterable[int] | None = None) -> pd.DataFrame     # concat partitions
def lake_inventory() -> pd.DataFrame  # one row per partition: source, season, week, n_rows, path, latest _captured_at
```
Storage backend is behind a thin `StorageBackend` protocol (`write_parquet`/`read_parquet`/`exists`/
`list`); `LocalParquetBackend` is the only impl in v1. Swapping to R2/S3 later means one new impl + a
config toggle, with zero changes to collectors or the assembler.

### Source registry — `src/collect/registry.py`
```python
@dataclass(frozen=True)
class Source:
    name: str                    # partition dir, e.g. "sleeper_proj_week"
    grain: Literal["week", "season", "game"]
    key_cols: tuple[str, ...]    # natural key within the source
    cadence: frozenset[str]      # subset of {"prelock", "postgame", "backfill"}
    backfillable: bool

SOURCES: dict[str, Source]       # the authoritative registry (single source of truth for collectors + crons)
```

### Collectors — `src/collect/<provider>.py`
Each collector is storage-free (returns data + provenance; the runner persists it), so it is unit-testable
offline against a fixture.
```python
@dataclass(frozen=True)
class Collected:
    source: str
    season: int
    week: int | None
    rows: list[dict]
    key_cols: tuple[str, ...]

# sleeper.py
def collect_proj_week(season: int, week: int) -> Collected          # sleeper_proj_week   (forward-only)
def collect_proj_season(season: int) -> Collected                   # sleeper_proj_season (forward-only)
def collect_stats_week(season: int, week: int) -> Collected         # sleeper_stats_week

# nflverse.py  (wrap existing src/data/nflverse.py loaders; add a `week` column, keep provider-native names)
def collect_player_week(season: int) -> Collected                   # nflverse_player_week
def collect_snaps(season: int) -> Collected                         # nflverse_snaps
def collect_ff_opportunity(season: int) -> Collected                # nflverse_ff_opp
def collect_injuries(season: int) -> Collected                      # nflverse_injuries
def collect_schedules(season: int) -> Collected                     # nflverse_schedules (carries vegas+weather fields)
def collect_depth_charts(season: int) -> Collected                  # nflverse_depth
def collect_id_crosswalk() -> Collected                             # id_crosswalk (season-grain, latest)

# market.py  (Vegas: primary source is nflverse_schedules spread_line/total_line/moneylines)
def collect_vegas_from_schedules(schedules_df) -> Collected         # vegas_odds (derived; implied team totals computed)

# weather.py  (open-meteo forecast, keyless; stadium table with dome/indoor flag)
def collect_weather_forecast(season: int, week: int, schedules_df) -> Collected  # weather (prelock; dome -> null/flagged)
```

### Collection runner — `scripts/collect.py`
```
python scripts/collect.py --mode {prelock|postgame} [--season Y --week N]
```
- Auto-detects season/week from Sleeper state when omitted; **off-season-safe**: no-op exit 0 when
  `season_type != "regular"` (reuse `analysis.snapshot.offseason_skip_reason` logic).
- `prelock` runs the `cadence ∋ "prelock"` sources; `postgame` runs the `cadence ∋ "postgame"` sources.
- Writes each `Collected` via `store.write_snapshot`, `captured_at = <run start, UTC>`.
- Prints a per-source summary (rows written, partition path). Logs any collector that fails **without
  aborting the others** (best-effort per source; exit non-zero only if *all* sources fail).

### Backfill runner — `scripts/backfill_lake.py`
```
python scripts/backfill_lake.py --seasons 2016-2025 [--sources a,b,c]
```
Pulls every `backfillable` source for the given seasons into the lake once. `captured_at` stamped as the
backfill run time, with a `_backfill=True` marker column so backfilled rows are distinguishable from
genuine point-in-time captures.

### Dataset assembler — `src/dataset/assemble.py`
```python
def build_training_frame(
    seasons: Iterable[int],
    scoring: Mapping[str, float],     # league scoring_settings (label re-scoring uses the Phase 1 engine)
    *,
    asof: Literal["prelock"] = "prelock",
) -> pd.DataFrame
    # One row per (sleeper player_id, season, week). Columns:
    #   - keys: player_id, season, week, position, team
    #   - label: y_custom_points  (week-N nflverse actuals -> nflverse_to_sleeper_stats -> scoring.engine.score)
    #   - baseline: sleeper_proj_week re-scored in `scoring` (the number to beat)
    #   - features (all as-of < week N lock): recent usage (snap%/target-share/rush-share, ewma & trend
    #     over weeks < N), expected points, injury status as-of, implied team total + spread, weather
    #     forecast, depth/role rank.
    # Enforces the lookahead guard; drops (and logs) rows whose label has no matching actuals.

def lookahead_ok(feature_week: int, feature_captured_at: str, target_week: int, lock_utc: str) -> bool
```

## Acceptance criteria (written before implementation; tests derive from these)

**Storage**
- [ ] `write_snapshot` then `read_snapshot` round-trips rows with all 4 reserved columns populated and
      correctly typed (`_captured_at` parseable as UTC).
- [ ] Re-running `write_snapshot` with the **same** `captured_at` and rows is idempotent (no dup rows,
      same file bytes-stable modulo parquet metadata).
- [ ] A **later-date** capture of the same key retains **both** snapshots (drift preserved); a
      **same-date** re-capture keeps only the latest `_captured_at`.
- [ ] Writes are atomic: a simulated crash mid-write (temp file present) never corrupts the committed
      partition.
- [ ] Empty `rows` is a no-op (no empty/invalid parquet emitted).

**Collectors**
- [ ] Each collector returns a `Collected` whose `key_cols` uniquely identify rows in its output on an
      offline fixture (no duplicate keys within one capture).
- [ ] Sleeper collectors return raw stat dicts unchanged (scoring-agnostic; no re-scoring at collect time).
- [ ] nflverse collectors carry a `week` column and provider-native stat names (re-scorable later via
      `ids.nflverse_to_sleeper_stats`).
- [ ] `collect_weather_forecast` marks dome/indoor stadiums with a null/flag (no bogus wind/temp), and
      never raises when the forecast source returns nothing (best-effort).
- [ ] `collect_vegas_from_schedules` computes implied team totals from `total_line` + `spread_line`
      correctly for a known game (home/away totals sum to `total_line`).

**Runner & crons**
- [ ] `scripts/collect.py --mode postgame` on cached fixtures writes every post-game source partition
      and prints a per-source row count.
- [ ] Off-season (`season_type != regular`) → `collect.py` exits 0 without writing (matches
      `refresh.yml`'s off-season behavior).
- [ ] One collector failing does not prevent the others from writing; the run reports the failure.
- [ ] Both workflows: `permissions: contents: write`, a `concurrency` guard, `timeout-minutes`, and a
      **no-op-safe** commit (`git diff --staged --quiet || commit+push`). Validated with `act` or a dry
      `workflow_dispatch`.

**Dataset assembler (the lookahead gate — most important)**
- [ ] `build_training_frame` produces exactly one row per (player_id, season, week) present in both a
      projection snapshot and week-N actuals.
- [ ] **Leak test:** a synthetic feature row whose `_captured_at`/`_week` is not strictly before the
      target week's lock is **excluded**; the same row dated before lock is **included**. (Red test first.)
- [ ] `y_custom_points` for a known player/week equals the Phase 1 engine's re-score of that week's
      actuals in the league scoring (reuse an existing validated fixture, e.g. the 10.55 Jameson Williams
      check).
- [ ] The baseline column equals Sleeper's `sleeper_proj_week` re-scored in the same scoring.
- [ ] Every dropped/unjoined row is logged (never silently dropped) — matches the project's
      "log every projection row that fails to join" rule.

**Project-wide**
- [ ] Full test suite stays green (currently 124 passed); `ruff` clean on changed files.
- [ ] No secret added anywhere; no new **required** paid dependency. Any new dep justified in its PR.
- [ ] `.gitignore` keeps volatile caches ignored; the lake **is** committed. `data_cache/README.md`,
      `docs/PROGRESS.md`, `docs/PLAN.md`, and `CLAUDE.md` updated to describe Phase 8.

## Tickets (atomic — one PR each)
Dependency order: **1 → {2,3,4} → 5 → 9 → 6**, with **7** after {2,3,4} and **8** last. Tickets 2/3/4
all register into the ticket-1 registry, so ticket 1 pins that interface to avoid integration drift.
Ticket 9 (R2 backend) can land any time after 1; ticket 6 (crons) needs 5 **and** 9.

1. **`store` + source registry** — `src/store/lake.py` (`StorageBackend` protocol +
   `LocalParquetBackend` + `LAKE_BACKEND` selection) and `src/collect/registry.py` with the full
   `SOURCES` table. Tests: storage round-trip, dedup/point-in-time, atomicity, empty-rows.
2. **Sleeper collectors** — `src/collect/sleeper.py` (`proj_week`, `proj_season`, `stats_week`). Tests
   against an offline projections/stats fixture.
3. **nflverse collectors** — `src/collect/nflverse.py` (`player_week`, `snaps`, `ff_opportunity`,
   `injuries`, `schedules`, `depth_charts`, `id_crosswalk`), wrapping the existing loaders. Tests on a
   small cached frame.
4. **Market + weather collectors** — `src/collect/market.py` (Vegas from schedules; implied team totals)
   and `src/collect/weather.py` (open-meteo forecast + `data/stadiums.py` dome table + historical fields
   from schedules). Tests: implied-total math, dome flagging, best-effort no-raise.
5. **Runners** — `scripts/collect.py` (prelock/postgame, off-season-safe, best-effort per source) and
   `scripts/backfill_lake.py` (one-time 2016–2025 pull with `_backfill` marker). Tests on fixtures.
6. **Cloud crons** — `.github/workflows/collect-prelock.yml` (**two schedules**: Thu ~22:00 UTC before the
   TNF lock + Sun ~15:00 UTC before the 1pm ET main slate) and `collect-postgame.yml` (Tue ~12:00 UTC,
   before the 18:00 season.db refresh). Both `LAKE_BACKEND=r2` with the 4 R2 secrets; off-season skip,
   concurrency, timeouts. (No git-commit step — data goes to R2.)
7. **Dataset assembler** — `src/dataset/assemble.py` (`build_training_frame` + `lookahead_ok`), reusing
   the Phase 1 scoring engine for the label. Tests: the leak gate (red-first), label correctness,
   one-row-per-key, unjoined logging.
8. **Docs + conventions** — `docs/data-conventions.md`, `.claude/skills/data-conventions/SKILL.md`, and
   updates to `CLAUDE.md` (Phase 8 + data-conventions block), `docs/PLAN.md`, `docs/PROGRESS.md`,
   `data_cache/README.md`, `.gitignore` (ignore `data_cache/lake/`).
9. **R2 backend** — `src/store/r2.py` (`R2Backend`, S3-compatible via a to-be-pinned client, e.g.
   `s3fs`/`boto3`), env-driven creds, and an R2 setup section in `docs/data-conventions.md` + the owner
   checklist. Tests: backend conforms to the `StorageBackend` protocol (mocked S3; no live calls in CI).

## Open questions
- **Q-A (TNF pre-lock):** ✅ RESOLVED — **add a Thursday capture** (ticket 6 pre-lock workflow gets a
  second cron: Thu ~22:00 UTC before the TNF lock, plus the Sunday one).
- **Q-B (backfill span):** ✅ RESOLVED — **2016–2025** (10 seasons).
- **Q-C (live odds provider):** OPEN (deferred) — v1 Vegas = nflverse schedule closing lines (free,
  keyless). A live pre-lock odds snapshot would need a provider. _Default: schedules-only in v1; revisit
  if the closing-line approximation proves too coarse for the projection model._ Note: reading published
  lines as features is **not betting** and is unaffected by jurisdiction (owner is in France, places no
  wagers; this tool is fantasy-only per CLAUDE.md). The "edge" produced is a **fantasy-decision** edge
  over leaguemates (points / win-probability), not a sportsbook +EV edge.

## Decision log
- 2026-07-16 — **Storage = Cloudflare R2 (production) + local parquet (dev), behind a `StorageBackend`
  protocol.** R2 free tier (10 GB / 1M Class A / 10M Class B ops-mo / zero egress) far exceeds this
  project's needs; R2 creds live in GitHub Actions secrets (no repo secrets). Local git/parquet stays the
  default so nothing is blocked before the owner's one-time R2 setup (account + card + bucket + token).
  Supersedes the initial "git-committed parquet only" lean. (You: "Is R2 free? if it is lets set it up".)
- 2026-07-16 — **#6 Projection baselines are not backfillable** (no free ex-ante historical projections
  from Yahoo/ESPN/FantasyPros/nflverse); train on lagged-actuals features, grade the market-beating
  baseline forward from 2026 W1. Additional forward projection sources deferred but registry-ready.
  (You: "can we use any other weekly preds for the backfill?".)
- 2026-07-16 — **Pre-lock capture runs Thursday + Sunday** (Q-A resolved) so TNF players get a fresh
  pre-lock snapshot. **Backfill span = 2016–2025** (Q-B resolved).
- 2026-07-16 — **Schema serves all three model targets** (weekly projections, distribution fitting,
  breakout classifier), so one player×week table with usage/role/market/weather + a real re-scored label.
  (You: "all?".)
- 2026-07-16 — **Scope includes Vegas + weather**, sourced free/keyless (Vegas from nflverse schedules;
  weather from open-meteo), preserving the no-secrets rule. (You: "Also add Vegas + weather".)
- 2026-07-16 — **Two capture cadences**: pre-lock (Sun AM) for point-in-time projections/injuries/odds/
  weather, post-game (Tue) for actuals/usage. (You: "Add a pre-lock capture".)
- 2026-07-16 — **Read-only + no-secrets preserved**; collection is GET-only against Sleeper/nflverse, and
  every v1 source is keyless (upholds the two immutable rules in CLAUDE.md).

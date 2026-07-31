# Spec: In-house models over the lake (Phase 9)

Status: **approved** — tickets filed
Issues: #27 · #28 · #29 · #30 · #31 · #32 · #33 · #34 (one per ticket below)
Owner: you (EM/PM) · Planner: Claude
Predecessor: [Phase 8 — cloud data collection & feature store](data-collection.md) (shipped `da702af`)

## Goal
Replace Sleeper's projections as this tool's source of player value with **our own models**, trained on
the Phase 8 lake and graded honestly. Phase 8 deliberately shipped the data foundation and stopped:
`build_training_frame` is the hand-off, and today every downstream surface — the draft board, the
lineup optimizer, the waiver ranker, both simulators — still consumes Sleeper's numbers re-scored into
our scoring.

The edge this phase is chasing is not "a better generic projection". It is the three places where a
generic projection is *structurally* wrong for this league: **half-PPR with 4-point passing TDs**,
**distance-based kicker scoring**, and **rich DST scoring** — all of which public projections misvalue
because they are built for standard settings.

## Non-goals
- **No swap by default.** Model output becomes a *selectable* projection source; Sleeper stays the
  default until a model clears the pre-declared bar in Decision #3. A rushed model that is worse than
  Sleeper's would actively cost us the draft.
- **No new data sources.** This phase reads the lake as it stands. If a model needs a feature that is
  not there, that is a Phase 8 registry ticket (one entry + one collector), not a change here.
- **No writes to Sleeper**, ever (immutable rule). Models are read-only consumers of the lake.
- **No deep learning.** Gradient boosting and regularized linear models over ~500k player-weeks.
  Anything heavier is unjustifiable at this data size and unmaintainable for one person.
- **Not replacing the scoring engine.** Every point total — label, baseline, prediction — is scored by
  the Phase 1 engine from the league's live `scoring_settings`. No model ever emits "points" directly
  where it could emit stats and let the engine score them (see Decision #2).

## Three findings that shape this plan
These came out of profiling the lake and registry before writing the ticket list. Each one invalidates
an approach that would otherwise have looked obvious.

### 1. There is no historical projection baseline. At all.
`baseline_sleeper_points` is **null for every row of 2016–2025**. Sleeper's projection endpoints serve
only the latest value, so the pre-lock capture that fills that column started with Phase 8 and produces
its first rows in **2026 Week 1**. Consequences:

- "Beat the market" **cannot be measured on a single historical row**. It is a forward-only,
  accumulating grade, one week at a time, from 2026 W1.
- Therefore the historical bar has to be something we define ourselves. Ticket **#28** exists to define
  it *before* any model is fit, because otherwise "is this good?" has no answer and every subsequent
  ticket is unfalsifiable.

### 2. The draft model cannot be an aggregation of the weekly model — and needs its own frame.
Two separate reasons, either of which is sufficient:

- **Season-grain projections are forward-only too.** `sleeper_proj_season` is `backfillable=False` and
  starts 2026. There is not one historical season of season-projection data to train or validate on.
- **The weekly model's features do not exist in August.** Its inputs are current-season lagged usage,
  the game's Vegas line and the game's weather forecast. At draft time none of those exist for any week.
  Running the weekly model with all of them null is out-of-distribution inference, not a projection.

A season model *is* trainable — but from a **different feature construction**: prior-season aggregates
(points, games played, snap share, target/rush share, expected points) → the next season's actual
custom-scored total, over the nine train/test season pairs in 2016–2025. That is a new frame builder,
`build_season_frame`, and it is ticket **#31**, not a footnote on #29.

### 3. Role/depth history is one season deep.
`nflverse_depth` carries `backfillable_from=2025`: nflverse rewrote the feed that season, and the legacy
shape has no clean key, so `collect_depth_charts` returns an empty capture for anything older. Depth and
role features therefore exist for **2025 forward only** — which is exactly the feature the breakout
classifier was specified around. Ticket **#33** must lean on **snap-share and target/rush-share
trajectories** as its role proxy, and treat literal depth-chart rank as a 2025+ refinement.

A fourth, smaller one worth stating: **week 1 of every season has no current-season lags.** It is a real
week that needs a real lineup, so the weekly model needs an explicit cold-start path — not a fallback
discovered in production.

## The calendar
It is **2026-07-29**. Week 1 kicks off in roughly six weeks; the draft lands in the week before it, with
our slot revealed somewhere between a day and a week ahead of that. That splits the phase into two paths
with very different urgency:

| Path | Tickets | Deadline | If it slips |
| --- | --- | --- | --- |
| **Draft** | #27 → #28 → #31 | the draft, ~5 weeks | Fall back to the existing Sleeper-projection board. It works; it shipped in Phase 2. |
| **Weekly** | #29, #30, #34 | Week 1, ~6 weeks — but graceful | Sleeper projections keep feeding the optimizer, exactly as they do now. Swap mid-season when the bar is cleared. |
| **After** | #32, #33 | none | These improve the sims and waivers; neither blocks a decision we make this season. |

Nothing in this phase is on a critical path to a *working* season, because every surface it touches
already works on Sleeper's numbers. That is the safety margin, and it is why Decision #3 can afford to
be strict.

## Design

### Where the code lives
```
src/model/
  frame.py       # build_season_frame (#31); the weekly frame is dataset.assemble's already
  evaluate.py    # walk-forward splits, per-position metrics, the Predictor protocol (#28)
  baselines.py   # trailing mean, prior-season rank, lagged expected points (#28)
  weekly.py      # skill-position weekly model (#29)
  kickdef.py     # K + DST, component-wise through the scoring engine (#30)
  season.py      # draft-value model (#31)
  breakout.py    # waiver classifier (#33)
  fit/           # fitted artifacts (parquet/json), committed — small, and they must be reproducible
```
`src/model/` is new and additive. No existing module changes until ticket **#34**, which is the only one
that touches a consumer.

### One protocol, so a baseline and a model are interchangeable
```python
class Predictor(Protocol):
    def fit(self, frame: pd.DataFrame) -> Predictor: ...
    def predict(self, frame: pd.DataFrame) -> pd.Series: ...   # index-aligned to frame
```
The three naive baselines implement it, so the harness scores a baseline and a gradient-booster through
the identical path. A model that cannot beat a five-line trailing mean under the same evaluation is a
model we learn about in an afternoon rather than in October.

### Evaluation: walk-forward by season, scored per position
Random k-fold over player-weeks would leak catastrophically — the same player's week 4 predicting his
week 3. Splits are **by season**: train on ≤ S-1, test on S, for S in 2018…2025 (2016–2017 reserved as
the lag warm-up).

Metrics are computed **per position**, never pooled, because a pooled MAE is dominated by the fact that
QBs score three times what kickers do and would call a model "better" for improving on QBs alone:

- **MAE / RMSE** — absolute accuracy, what a projection number claims to be.
- **Spearman ρ within (position, week)** — the *ordering*, which is what start/sit and waiver decisions
  actually consume. A model can lose on MAE and still be the one you want.
- **Calibration** — predicted vs. realized mean by decile. A model with good MAE and bad calibration
  will systematically mis-rank the boom/bust players the FLEX decision hinges on.

## Tickets

| # | Issue | Ticket | Path | Depends on |
| --- | --- | --- | --- | --- |
| 1 | #27 | Profile the training frame on the real lake | draft | — |
| 2 | #28 | Evaluation harness + three naive baselines | draft | #27 |
| 3 | #29 | Weekly point model — QB/RB/WR/TE | weekly | #28 |
| 4 | #30 | Weekly K + DST model, component-wise | weekly | #28 |
| 5 | #31 | `build_season_frame` + draft-value model | draft | #28 |
| 6 | #32 | Fit the simulators' distributions | after | #29 |
| 7 | #33 | Breakout / waiver classifier | after | #28 |
| 8 | #34 | Wire-up behind a flag + the live swap gate | weekly | #29, #31 |

---

### Ticket 1 (issue #27) — Profile the training frame on the real lake
**Goal.** `build_training_frame(2016..2025)` has never been run at full scale. Before anything is
modeled, know what is actually in it: rows per season × position, null rate per feature per season, and
which feature sets are usable over which span. This is the ticket that decides whether we train on
2016+ or 2019+, and it is cheap.

**Files.** `scripts/profile_frame.py`; the report committed to `docs/model-data-profile.md`.

**Acceptance.**
- [ ] Per-season × per-position row counts and per-feature null rates, committed as a readable table.
- [ ] Names explicitly which seasons support which feature groups (usage / injury / market / weather /
      depth), confirming finding #3 against the data rather than the registry comment.
- [ ] Records the assembly's warning output verbatim — **zero unexpected warnings on real data** is the
      standing bar, and a full-scale run is the first honest test of it.
- [ ] Confirms `baseline_sleeper_points` is null across all of 2016–2025 (finding #1), so #2 is
      building against a measured fact and not a remembered one.

---

### Ticket 2 (issue #28) — Evaluation harness + three naive baselines
**Goal.** Define "good" before fitting anything. Walk-forward season splits, per-position metrics, and
three baselines that any real model must beat.

**Files.** `src/model/evaluate.py`, `src/model/baselines.py`, `tests/test_model_evaluate.py`.

**Baselines.** `TrailingMean` (the player's mean over the last N completed weeks — the heuristic every
human uses), `PriorSeasonRank` (prior-season per-week average by position rank), `LaggedExpectedPoints`
(nflverse `exp_points`, lagged one week — legal only lagged, per Phase 8 Decision #6).

**Acceptance.**
- [ ] **Leak test, written red first:** a split whose training rows include any row from the test season
      fails. This is the #28 analogue of Phase 8's lookahead gate and it fails closed the same way.
- [ ] Metrics are per position; a pooled-only report is a test failure.
- [ ] Spearman ρ is computed **within (position, week)**, not across weeks.
- [ ] The three baselines' scores are committed as the recorded bar, so every later ticket's claim of
      improvement is checkable against a fixed number.
- [ ] `ruff` clean; suite green.

---

### Ticket 3 (issue #29) — Weekly point model — QB/RB/WR/TE
**Goal.** The core regression: predict `y_custom_points` at player × week, pre-lock.

**Files.** `src/model/weekly.py`, `tests/test_model_weekly.py`, fitted artifact under `src/model/fit/`.

**Approach.** Gradient boosting per position (or one model with position interactions -- #27's row counts
decide; TE and K have thin per-position data). Start with a regularized linear model as the sanity floor:
if boosting cannot beat ridge, the features are the problem, not the learner.

**Acceptance.**
- [ ] Beats **all three** #28 baselines on held-out seasons, on **both** MAE and within-week Spearman ρ.
      Losing one metric to win the other is a result to discuss, not a pass.
- [ ] **Cold start is tested explicitly:** week 1 predictions, with no current-season lags, are produced
      and scored separately. A week-1 MAE hidden inside a season average is not an answer.
- [ ] Feature importances recorded — a model leaning hard on a feature that #27 showed is 60% null is a
      finding, not a detail.
- [ ] Refits reproducibly from the lake; the fitted artifact is not a hand-edited file.

---

### Ticket 4 (issue #30) — Weekly K + DST model, component-wise
**Goal.** The league's two most mis-valued positions, and the clearest structural edge in the project.
CLAUDE.md flags both: kicker scoring is distance-based (`fgm_40_49` 4, `fgm_50p` 5) and DST scoring is
rich, with independent points-allowed buckets. A model that predicts *points* directly learns those
buckets implicitly and badly.

**Approach.** Predict the **components** and score them through the Phase 1 engine: for K, attempts and
make rate by distance band; for DST, sacks / takeaways / TDs and the opponent's points-allowed
distribution. The Vegas implied team total and spread already in the frame are the natural inputs — they
are close to the market's direct opinion on both.

**Acceptance.**
- [ ] Predictions are produced by `scoring.engine.points` over predicted stat lines, never by a
      points-valued regression head. This is Decision #2 made concrete.
- [ ] Beats the #28 baselines for K and DEF specifically.
- [ ] A distribution over the points-allowed bucket, not just a point estimate — the buckets are
      discontinuous (0 pts allowed = 10, 1-6 = 7), so the expectation over the bucket distribution is
      the correct quantity and a mean-then-bucket estimate is provably biased.

---

### Ticket 5 (issue #31) — `build_season_frame` + draft-value model
**Goal.** The draft-path deliverable, and per finding #2 a genuinely separate model. Prior-season
aggregates → the next season's actual custom-scored total.

**Files.** `src/model/frame.py` (`build_season_frame`), `src/model/season.py`,
`tests/test_model_season.py`.

**Acceptance.**
- [ ] `build_season_frame(seasons, scoring)` — one row per (player, season), features from seasons ≤ S-1
      only, label = the real re-scored season total for S. Same fail-closed discipline as
      `dataset.assemble`: a feature that cannot be proven prior-season is withheld.
- [ ] Beats a prior-season-total baseline on held-out seasons, evaluated on **VOR-relevant ordering
      within position** — draft value is a ranking problem, and MAE on a season total is nearly
      irrelevant to whether the fourth-round pick was right.
- [ ] Feeds `draft.vor` in the same `PlayerRow` shape the board already uses, so the existing tier and
      VOR machinery is reused rather than reimplemented.
- [ ] Rookies are handled explicitly — they have no prior season, and "no prior season" is a large,
      predictable, every-year cohort, not an edge case.

---

### Ticket 6 (issue #32) — Fit the simulators' distributions
**Goal.** `draftsim/distributions.py` carries `POSITION_CV`, `GAME_CV` and `INJURY_RISK` as hand-picked
constants, honestly labelled *"heuristic, not fitted"* and printed in the report so they can be judged.
This is the ticket that earns them. `seasonsim` re-exports the same knobs, so both simulators improve at
once.

**Acceptance.**
- [ ] Per-position season CV, single-game CV, and injury (probability, mean games missed) fitted from
      2016–2025 actuals and availability history.
- [ ] Fitted values live as a **data artifact**, not edited constants, with the heuristics retained as
      the fallback when a position has too little data.
- [ ] The docstrings that currently say "heuristic, *not* fitted" are updated to say what they now are,
      and the report keeps printing them.
- [ ] A before/after on championship odds for one fixed roster — if fitting the knobs does not move the
      sim's output, that is worth knowing and recording too.

---

### Ticket 7 (issue #33) — Breakout / waiver classifier
**Goal.** The waiver-facing target: which rostered-or-free player is about to matter. Forward label —
did role and production step up over the next N weeks — with usage trajectories as the input.

**Acceptance.**
- [ ] The label's forward window is defined and defended, and its lookahead direction is the *opposite*
      of every other model here (the label is deliberately in the future; the **features** must still be
      strictly pre-lock). A test pins that asymmetry, because it is exactly the place a leak hides.
- [ ] Uses snap-share and target/rush-share trajectories as the role proxy per finding #3; depth-chart
      rank enters as a 2025+ refinement, not a required feature.
- [ ] Precision at the top of the ranking is the reported metric — we act on the top handful each week,
      so overall AUC is close to meaningless here.

---

### Ticket 8 (issue #34) — Wire-up behind a flag + the live swap gate
**Goal.** Make model output a *selectable* projection source, and stand up the forward comparison that
is the only honest "beat the market" grade (finding #1).

**Files.** `src/optimizer/inputs.py`, `src/waivers/inputs.py`, `src/draftsim/inputs.py`,
`src/seasonsim/inputs.py` — each currently calls `sleeper.get_projections` directly.

**Acceptance.**
- [ ] One projection-source seam, defaulting to Sleeper. Every existing test passes untouched with the
      default — if flipping the default is required to keep the suite green, the seam is wrong.
- [ ] A live scoreboard accumulating model vs. `baseline_sleeper_points` from 2026 W1, per position,
      week by week.
- [ ] The gate in Decision #3 is implemented as a check, not a note.

## Decision log
- **Decision #1 — Evaluation before models.** The harness and its baselines ship first (#28), because with no
  historical Sleeper baseline (finding #1) there is otherwise no definition of "better" and every model
  ticket becomes unfalsifiable. This mirrors Phase 1's "validate the scoring engine before building
  anything on it" and Phase 8's "the leak gate is the most important test in the ticket".
- **Decision #2 — Models predict stats; the engine scores them.** Wherever a target decomposes into stat
  components — K and DST unambiguously (#30) -- the model predicts components and `scoring.engine.points`
  converts them. Keeps the immutable "never hand-code scoring" rule true through the model layer, and
  means a mid-season scoring change re-prices predictions with no retraining.
- **Decision #3 — The swap bar, declared before the model exists.** Model output replaces Sleeper as a default
  only after it beats `baseline_sleeper_points` on **both** MAE and within-week Spearman ρ, per position,
  over **at least 4 live 2026 weeks**. Declared now, while there is no model to be attached to, because a
  bar written after seeing the results is not a bar. Until then: selectable, not default (#34).
- **Decision #4 — The draft model is a separate model, not an aggregation.** Per finding #2 — no season-grain
  training data exists, and the weekly model's features are all absent in August.
- **Decision #5 — Per-position metrics only.** A pooled metric is dominated by cross-position scoring scale and
  would let a QB-only improvement masquerade as a general one.
- **Decision #6 — Walk-forward by season; 2016–2017 are lag warm-up.** Random splits leak a player's own adjacent
  weeks. The first two seasons are consumed by the lag/EWMA windows and are not scored on.
- **Decision #7 — The weekly skill model uses a points head, valid only while skill scoring is linear.** #29's
  QB/RB/WR/TE model regresses the engine-scored `y_custom_points` directly rather than predicting stat
  components (a reading of Decision #2). The equivalence it rests on is stated explicitly: this league's
  skill scoring is **linear** in the stats (every skill key is a per-unit coefficient — `rec` 0.5,
  `rush_yd` 0.1, `pass_td` 4 — with no bucket or threshold), so `E[points] = sum(coef · E[stat])` and
  `score(E[X]) = E[score(X)]`: a points head is unbiased and identical in expectation to a component
  head. This does **not** license a points head for #30 — K and DST are the opposite case, step functions
  of stat buckets (`fgm_40_49` 4 vs `fgm_50p` 5; `pts_allow_0` 10 vs `1_6` 7) where `E[f(X)] ≠ f(E[X])`
  and only a component model is unbiased. The equivalence also holds only while scoring stays linear, and
  `LEAGUE_ID` still points at the 2026 test sandbox — so `model.weekly.assert_linear_skill_scoring` is a
  fail-closed guard, run against the live scoring at every fit/eval, that **raises** if any skill-relevant
  key is bonus/threshold-shaped. If the real 2026 league returns with a yardage bonus, the fit breaks
  loudly rather than biasing the model silently.
- **Decision #8 — Weekly cold start defers to the prior-season baseline, gated per position by measurement.**
  A week-1 (or mid-season-debut) row carries no within-season lag, so the weekly ridge has only the Vegas
  market there and — measured on the real lake (`docs/model-weekly.md` §B) — loses the cold start at all
  four positions to `PriorSeasonRank`, which carries last season's level. So `model.weekly.WeeklyModel`
  defers cold-start rows to `PriorSeasonRank`, but **per position and only where ridge was measured to
  lose** (the same measured-gate shape as #31's DEF deferral, not a blanket rule): a position where ridge
  wins its cold start is fielded. **Follow-up (not built here):** a prior-season points-per-game feature
  would give the ridge real cold-start signal and likely close this gap; its trigger is the recorded
  cold-start margin (§B). It does **not** need a Phase-8 assembler change — it is derivable in-model from
  the training frame the way `TrailingMean` derives its within-season lag — so its real cost is the
  warm-up trap #31 hit (a feature derived from "first appearance in the window" is 100% wrong in the
  earliest built season and must not train that season), not new collection scope.
- **Decision #9 — The evidence-and-shipping contract every model ticket is held to.** Surfaced across the
  #27–#29 reviews; #30–#34 inherit it rather than restate it.
  1. **Measured or not met** — no acceptance number that is not derived from a committed, regenerable
     artifact (`scripts/eval_*.py` → `docs/*.md`, pure `render_report`, warnings captured and counted).
     Synthetic tests pin **mechanics only**, never a bar-clearing number on a fixture engineered to
     satisfy it (#31's defect).
  2. **Revert-check every guard** — a test whose guarded code can be reverted with the test still green
     pins nothing.
  3. **Safe by default; the diagnostic is the explicit opt-out** — the shipped configuration is what the
     bare constructor yields; the weaker/diagnostic variant takes an explicit argument
     (`SeasonModel(require_usage=True)`; `WeeklyModel()` defers by default, vs the pure-ridge
     `WeeklyModel(defer_cold_start=())`). The #29 review caught the default being the variant the artifact
     records as *losing*.
  4. **No write-only artifact** — whatever a fitted artifact records (a gate, weights), something must
     read it back into a correctly-configured model (`load_fitted`), failing loud rather than degrading
     silently when a part is unfit.
  5. **Tests construct the object the way production will** — a test that always passes config explicitly
     cannot catch a bad default; #34 constructs the weekly model for real.
  6. **State where a claim holds** — a criterion that spans positions is met **per position**, or met
     partially and the summary says which. #31 reported "beats the baseline on within-position ordering"
     from a QB/RB/WR fixture against a six-position criterion; on the real lake it lost at K and DEF.
     Item 1 catches the synthetic half of that; nothing else catches reporting three positions as if they
     were six. Decision #5 governs the metric *grain*, not the *scope* of the verdict built on it.

---
name: data-conventions
description: >-
  Point-in-time, lookahead-free rules for the Phase 8 data lake and the training frame built from it.
  Use when writing or modifying ANY data-collection, feature-engineering, or model-training code — a
  collector, runner, or registry entry under src/collect/; the store under src/store/; the assembler
  under src/dataset/ (build_training_frame, lookahead_ok, a new feature or label column); the backfill;
  or any code that turns lake data into model inputs. Enforces: all timestamps UTC; a week-N feature
  may use only data knowable before week-N's first kickoff, and the gate fails closed; labels and the
  baseline are re-scored from the live scoring_settings, never hard-coded; join on gsis_id (skill) /
  team abbreviation (DST) / sleeper_id, logging every unjoined row; sleeper_proj_* are forward-only
  (2026 W1+); and src/collect/registry.py is the authoritative source table. Load it before touching
  the lake so a leak is designed out rather than reviewed in.
---

# Data conventions (Phase 8 lake)

Full reference: [docs/data-conventions.md](../../../docs/data-conventions.md). Design + decisions:
[docs/plans/data-collection.md](../../../docs/plans/data-collection.md). This skill is the working
checklist; when a rule and the code disagree, the code wins and the doc should be corrected.

## The rules, in one screen

1. **UTC everywhere.** Every `_captured_at` and derived instant is ISO-8601 UTC. A naive datetime is a
   bug — the runner *raises* on one rather than reading a local clock as UTC.
2. **The lookahead gate.** A week-*N* feature may use only data knowable **before week *N*'s first
   kickoff**. It fails **closed**: an unreadable/missing timestamp on a non-earlier-week row is
   refused, because a leak is invisible in the output and a dropped feature is not.
3. **Labels are re-scored live.** `y_custom_points` and `baseline_sleeper_points` come from the
   engine (`scoring.engine.points`) over the league's live `scoring_settings` — never a hard-coded
   coefficient. Drop nulls before summing (`NaN * coef` poisons the whole row).
4. **Join keys:** `gsis_id` (skill/K), **team abbreviation** (DST), `sleeper_id` (Sleeper space), via
   the `id_crosswalk`. **Log every row that fails to join** — never drop silently.
5. **Forward-only vs backfillable.** `sleeper_proj_*` are forward-only (2026 W1+; the endpoints serve
   only the latest values). Everything else is backfillable from nflverse. `content_known` ≠ `cadence`
   — do not derive one from the other.
6. **`src/collect/registry.py` is authoritative** for every source's grain / key / cadence / dedup /
   content_known / backfillable. Adding a source is one entry there plus a collector — never a change
   to the store. Don't restate the registry table in docs.

## The gate, in code

The gate is [`dataset.assemble.lookahead_ok`](../../../src/dataset/assemble.py):
`lookahead_ok(feature_week, feature_captured_at, target_week, lock_utc)`. It returns `True` on
**either** rule — content (`feature_week < target_week`) **or** capture (`known_at < lock_utc`) — and
`False` otherwise. `feature_captured_at` is the row's **resolved** known-at (`_resolved_known_at`),
which for a backfilled row is *not* its raw `_captured_at`.

### GOOD — a same-week snapshot known before the lock (capture rule)

```python
# A week-5 Vegas line / projection / injury report captured Sunday 15:00 UTC,
# before week 5's first kickoff at 17:00 UTC.
lookahead_ok(feature_week=5, feature_captured_at="2026-10-04T15:00:00Z",
             target_week=5, lock_utc="2026-10-04T17:00:00Z")
# -> True   content rule: 5 < 5 is False; capture rule: 15:00 < 17:00 is True → admitted
```

(The content rule is the other legal path: a week-3 actual as a week-5 feature — `feature_week 3 <
target_week 5` — which is exactly how the lagged usage features stay legal, whatever their capture
time.)

### BAD — the same week-5 fact known only at/after kickoff (a leak)

```python
# The SAME week-5 quantity, but known only at or after kickoff: a same-week actual,
# an at-kickoff weather reading, a projection captured late.
lookahead_ok(feature_week=5, feature_captured_at="2026-10-04T17:30:00Z",
             target_week=5, lock_utc="2026-10-04T17:00:00Z")
# -> False  content rule: 5 < 5 is False; capture rule: 17:30 < 17:00 is False → refused
```

The worst case is quieter: a `post_game` row (week-*N* actuals) used as a week-*N* feature resolves its
known-at to `NaT`, so `feature_captured_at` is `None`, the capture rule short-circuits, and the gate
returns `False`. That is why `post_game` sources are legal **only lagged** (content rule), and why
`nflverse_schedules` is read as a calendar only — deriving knowability from its pre-lock *cadence*
would hand the assembler the label.

## Before you commit collection/feature code

- New source? Add it to `registry.py` with the right `content_known`, and let `tests/test_registry.py`
  and the collector pinning tests catch a missing collector.
- New same-week feature? It must pass through `lookahead_ok` / `_admissible`; add a red-first leak test
  (a row that *would* leak is excluded, the same row dated before lock is included).
- Touching a workflow? [`tests/test_workflows.py`](../../../tests/test_workflows.py) holds the crons to
  their contract, including two **cross-file** invariants (the shared `lake-capture` concurrency group;
  no pinned `--week` on the scheduled path). A red test there is the design talking.
- Don't read `nflverse_depth`'s `latest_captured_at` as freshness, don't alert on a green postgame
  skip, and don't treat `_backfill=False` as "observed live" — see the gotchas in §9 of the full doc.

# 2026 season — setup & run checklist

Everything the tool needs when the **2026 redraft league** is created. The 2026 league is a *new*
Sleeper `league_id` (see CLAUDE.md), so the one real config change is pointing the tool at it; after
that the same scripts/apps work unchanged. All of it is read-only against Sleeper.

---

## 0. One-time config (do this first, once the 2026 league exists)

Edit [src/sleeper/config.py](../src/sleeper/config.py) (or set the matching env vars):

- `LEAGUE_ID` → the **2026** league id (find it in the league URL, or
  `GET /v1/user/<user_id>/leagues/nfl/2026`).
- `PREVIOUS_LEAGUE_ID` → the 2025 id (`1257071615817043968`), so prior-season lookups still work.
- `MY_USER_ID` stays the same (`866260653093036032`) unless you switch accounts.

Then **re-confirm scoring & roster against the API** — the 2026 league may differ (CLAUDE.md notes a
possible 6th bench + IR). The scoring engine reads `scoring_settings` live, so no code changes; just
sanity-check `scripts/validate_custom.py` once.

> For the **hosted** dashboard, also update `SLEEPER_LEAGUE_ID` in the GitHub Actions workflow env (or
> repo secrets) if you override it there — otherwise the committed default in `config.py` is used.

---

## 1. Pre-draft prep (projections out, ~weeks before the draft)

Monte Carlo simulator — compare roster builds and see which targets survive to your picks:

```bash
./.venv/Scripts/python scripts/draft_sim.py --season 2026 --slot <your-slot> --board
```

- `--slot` once it's revealed (~1 day–1 week before the draft); omit it earlier and it falls back to a
  middle slot with a warning (prep stays slot-agnostic until revealed).
- `--board` prints a representative simulated draftboard. Output is **directional** — read the printed
  assumptions block.

## 2. Rehearse the live tracker (any time) — use a Sleeper **mock draft**

```bash
streamlit run apps/draft_app.py
```

- In the sidebar, paste the **mock draft's `draft_id`** but keep your **real 2026 League ID** in the
  League ID box — the board then re-scores in *your* custom scoring even though the picks come from the
  bot mock. This is the full end-to-end rehearsal (polling, board, survival flags).
- A real "test league with bots" won't auto-draft an empty league; the mock-draft route is the way.

## 3. Draft night

```bash
streamlit run apps/draft_app.py
```

- Pick the real 2026 draft from the sidebar dropdown (auto-discovered from the league id) or paste its
  `draft_id`. Polls every ~3s. Slot, snake pick numbers, and survival flags unlock once `draft_order`
  populates.

## 4. In-season (weekly)

- The **hosted dashboard** (`apps/season_app.py`, reads `data_cache/season.db`) auto-refreshes every
  **Tuesday 18:00 UTC** via [.github/workflows/refresh.yml](../.github/workflows/refresh.yml) (before
  Wed 09:00 CEST waivers) and redeploys on the commit. With `LEAGUE_ID` pointed at 2026 it just works.
- Manual refresh / off-cron:
  ```bash
  ./.venv/Scripts/python scripts/refresh_data.py            # auto-detects season+week
  ./.venv/Scripts/python scripts/refresh_data.py --week N --season 2026
  ```
- Tabs: **This Week** (optimal lineup as Sleeper-style cards w/ kickoff times + start/sit), **Waivers &
  Stash**, **Team Analysis**.
- **FLEX & kickoff times:** the optimizer maximizes projected points (it does *not* use kickoff time).
  Each starter card shows its game day/time so you can hedge the FLEX manually (e.g. start the later
  game to keep optionality) — that's a manual lock-time call.

## 5. Retrospective (mid- or post-season)

```bash
./.venv/Scripts/python scripts/backtest.py --season 2026     # defaults to current season
```

- Writes the **separate** `data_cache/backtest.db` (the weekly cron never touches it). Safe mid-season:
  only finished weeks are included. Powers the dashboard's **Backtest** tab — season summary, weekly
  detail, draft replay, **full draftboard**, **weekly head-to-head matchups**, and **transactions**.
- To backtest 2025 after switching config to 2026: `--season 2025 --league 1257071615817043968`.

---

## Quick reference

| When | Command |
|---|---|
| Config | edit `LEAGUE_ID` / `PREVIOUS_LEAGUE_ID` in `src/sleeper/config.py` |
| Validate scoring | `scripts/validate_custom.py` |
| Draft prep | `scripts/draft_sim.py --season 2026 --slot N --board` |
| Live draft / rehearsal | `streamlit run apps/draft_app.py` |
| Weekly refresh | `scripts/refresh_data.py` (or the Tue cron) |
| Backtest | `scripts/backtest.py --season 2026` |

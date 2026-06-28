# Streamlit apps

Streamlit entrypoints (run with `streamlit run apps/<app>.py`). These import from the `src/`
packages; no business logic lives here.

- `draft_app.py` — **local** live draft tool (Phase 2, **done**). `streamlit run apps/draft_app.py`.
  Polls `/draft/<id>/picks` ~3s; VOR-ranked + tiered best-available board in our custom scoring,
  roster-need highlighting, positional runs, and snake-pick/survival flags once the slot is
  revealed. Sidebar discovers the draft from the League ID, or paste a Draft ID. Read-only.

- `season_app.py` — **hosted** season dashboard (Phase 5, **done**). `streamlit run apps/season_app.py`.
  Reads **only** the precomputed `data_cache/season.db` snapshot — it never calls an API on page load.
  Four tabs: **This Week** (optimal lineup + start/sit + risky flags), **Waivers & Stash**
  (handcuff/spend/stash/bye alerts), **Team Analysis** (positional strength vs the league —
  season-long *and* this-week — bye-week gaps, positional needs, trade-target ideas before the
  Week-11 deadline, and the Weeks 15–17 playoff outlook), and **📈 2025 Backtest** (a
  completed-season "what if I'd used this tool" review — best-possible lineup vs what you started,
  VOR draft vs your actual draft, season summary + weekly/league ranks — read from a separate
  `data_cache/backtest.db` built by `scripts/backtest_2025.py`). The live snapshot is rebuilt by
  `scripts/refresh_data.py` (run by the weekly GitHub Actions cron, which commits it back).

## Deploy `season_app.py` on Streamlit Community Cloud (free)

The repo is **public and needs no secrets** (Sleeper requires no auth), so hosting is free:

1. Push this repo to GitHub (must be public for the free tier; `data_cache/season.db` is committed).
2. Go to **https://share.streamlit.io** and sign in with GitHub (authorize Streamlit if prompted).
3. Click **Create app → Deploy a public app from GitHub**.
4. Set **Repository** = `<you>/FantasyFootball`, **Branch** = `main`,
   **Main file path** = `apps/season_app.py`. (Python deps are read from `pyproject.toml`.)
5. Click **Deploy**. First build takes a couple of minutes.
6. Done — no secrets to configure. The app **auto-redeploys** every time the GitHub Actions weekly
   refresh commits a new `data_cache/season.db`. Use the in-app **🔄 Reload snapshot** button to clear
   the read cache if you need it sooner. (Free apps sleep after ~7 days idle and wake on the next visit.)

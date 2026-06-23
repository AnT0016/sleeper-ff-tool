# Streamlit apps

Streamlit entrypoints (run with `streamlit run apps/<app>.py`). These import from the `src/`
packages; no business logic lives here.

- `draft_app.py` — **local** live draft tool (Phase 2, **done**). `streamlit run apps/draft_app.py`.
  Polls `/draft/<id>/picks` ~3s; VOR-ranked + tiered best-available board in our custom scoring,
  roster-need highlighting, positional runs, and snake-pick/survival flags once the slot is
  revealed. Sidebar discovers the draft from the League ID, or paste a Draft ID. Read-only.

Planned (not built yet):
- `season.py` — **hosted** season dashboard on Streamlit Community Cloud (Phase 5).

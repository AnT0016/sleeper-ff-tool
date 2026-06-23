# Streamlit apps

Streamlit entrypoints (run with `streamlit run apps/<app>.py`). These import from the `src/`
packages; no business logic lives here.

Planned (not built yet):
- `draft.py` — **local** live draft tool, polls `/draft/<id>/picks` ~3s (Phase 2).
- `season.py` — **hosted** season dashboard on Streamlit Community Cloud (Phase 5).

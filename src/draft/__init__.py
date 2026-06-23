"""Live-draft decision logic for the local draft tracker (Phase 2).

Pure, offline-testable helpers layered on the custom-scored board (``projections.board``):

- ``vor``    -- replacement-level baselines, value-over-replacement, and tiers.
- ``snake``  -- snake-draft pick numbers for a revealed slot, and survival likelihood vs ADP.
- ``roster`` -- roster-need tracking, positional-run detection, and pick diffing from live picks.

None of this touches the network; the Streamlit app (``apps/draft_app.py``) wires these to the
Sleeper client and renders them.
"""

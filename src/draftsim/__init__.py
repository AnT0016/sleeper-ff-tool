"""Forward Monte Carlo draft simulator (Phase 6 — optional, directional draft-prep aid).

Simulates thousands of full snake drafts. The 11 other managers draft by *market* ADP (+ noise);
*I* draft by our custom VOR under a chosen roster-build strategy. Each sim independently draws every
player's season outcome (lognormal around our custom-scored projection, with an injury haircut), then
scores my resulting roster's best legal lineup on those sampled outcomes — giving a *distribution* of
season results per build, not a single expected value.

Decisions are ex-ante (projections / ADP); evaluation is ex-post (sampled outcomes). Output is
directional, not gospel — every variance / ADP / injury assumption is printed alongside the results.

Reuses the rest of the project: ``projections.board`` (custom-scored board + ADP), ``draft.vor``
(VOR), ``draft.snake`` (pick numbers), ``draft.roster`` (slot config). Read-only.
"""

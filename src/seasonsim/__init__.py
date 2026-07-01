"""Full-season championship Monte Carlo (Phase 7 — optional, directional).

Where :mod:`draftsim` asks *"which roster build drafts the best distribution of season points?"*,
this package asks the next question: *"given the twelve rosters as they stand, how often does MY team
actually win the title?"* — turning draft-capital into a championship probability.

One simulated season: draw every rostered player's **weekly** points (with multi-week injury
stretches), set each of the 12 teams' lineups each week, resolve the real head-to-head schedule to a
win/loss record, seed the top-6 playoff bracket, and play Weeks 15-17 to a champion. Aggregated over
thousands of seasons this yields P(championship), P(playoffs), and the full seed/wins distribution for
my team — plus a head-to-head of roster builds under common random numbers.

Two skill regimes are reported so the value of good in-season management is visible, not assumed:
*equal-skill* (everyone sets lineups from the same projections) isolates roster quality + schedule
luck; *my-edge* (I set lineups cleanly while opponents make realistic start/sit errors) adds the
weekly-optimizer edge. The gap between them is what tightening the in-season levers is worth.

Decisions are ex-ante (projected weekly means); evaluation is ex-post (sampled outcomes). Directional,
not gospel — every variance / injury / skill assumption is printed alongside the numbers. Read-only.
"""

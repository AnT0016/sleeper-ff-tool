"""Weekly lineup optimizer (PuLP).

Optimizes the starting lineup over custom-scored weekly projections under the locked slot
constraints: exactly 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX from {RB,WR,TE}, 1 K, 1 DEF. Excludes
bye-week and OUT/IR players; FLEX takes the best leftover RB/WR/TE.
"""

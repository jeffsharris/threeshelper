# K1 Charter Amendment A3: Fused Exact Post-Spawn Rows

Date frozen: 2026-07-26

The original K1 charter and A1/A2 remain immutable. A spent-state profile,
performed before any fresh stream or gate, showed that the remaining K1 cost
was dominated by thousands of tiny Python/`ctypes` crossings:

- 15,272 native base-move calls;
- 3,782 native leaf calls;
- 6,174 `_post_spawn_state_value` invocations;
- zero value mismatch.

A3 permits one profiling-led call-granularity optimization and no other
variant. The native library adds exactly one export,
`k1_post_spawn_rows`. For one post-spawn board it:

1. evaluates all four exact base moves through the already frozen native
   primitive;
2. emits legal flags, shifted boards, eligible slots, before/after scores, and
   score deltas;
3. evaluates the already frozen exact composite leaf on each legal shifted
   board.

The Python wrapper uses that result to populate the same legal, base-move,
score, and afterstate caches before returning the same
`max(score_delta + leaf)` value. Cache keys, limits, pruning, player/chance
search, preview/deck semantics, chance limit, node budget, fallback, actions,
ties, compiler, flags, corpus, streams, timing schedule, and gates remain
unchanged.

Focused tests must compare every returned row and populated cache value against
the retained C1 implementation. A3 is the final development optimization; no
further kernel/call-layout variant is permitted before the fresh gate.

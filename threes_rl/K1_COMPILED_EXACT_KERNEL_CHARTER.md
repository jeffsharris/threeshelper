# K1 Compiled Exact Leaf and Transition Kernel Charter

Date frozen: 2026-07-26

## Decision Scope

K1 asks whether a separately implemented native kernel can make the retained
exact C1 depth-3 calculation satisfy the unchanged runtime gate without
selective admission.

C1 remains permanently `FAIL_STOP_C1` and is byte-locked as the exact
equivalence oracle. C2 remains permanently `KILL_C2_COST_ADMISSION`; its
untouched 48-state runtime gate may not be opened or reused. K1 is engineering
only and cannot change the incumbent, dashboard, or any gameplay claim.

An engineering pass may seal only `READY_K1_FULL_POLICY_PREFLIGHT`.

## Offline Toolchain Lock

The one selected toolchain is:

- compiler:
  `/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang`
- compiler identity: Apple Clang `21.0.0`, arm64 Apple target
- compiler file SHA-256:
  `7def90dd8829726686213a747fc5bff1583df933dae5edc55d755479e0bfe00a`
- source language: C11
- Python binding: standard-library `ctypes`
- flags, in this exact order:
  `-O3 -std=c11 -fPIC -dynamiclib -fno-fast-math -ffp-contract=off
  -fno-associative-math -Wall -Wextra -Werror -Wl,-no_uuid`

No dependency install, network access, alternate compiler, alternate flag set,
JIT, architecture-specific flag, kernel variant, or sweep is permitted.

## Native Surface

The native library exports exactly:

1. `k1_eval_composite`: exact frozen-incumbent n-tuple composite leaf values
   for a contiguous batch of 4x4 `int32` boards.
2. `k1_base_move`: exact Threes base swipe and eligible insertion slots for one
   board/action.
3. `k1_score_board`: exact board score.
4. `k1_kernel_abi_version`: fixed ABI identity `1`.

Everything else remains in the retained C1 implementation:

- player/chance recursion;
- exact chance and preview/deck enumeration;
- `chance_limit=8`;
- deterministic `value_node_budget=2048` and fallback;
- transposition and cache semantics;
- terminal handling;
- action selection and ties;
- incumbent component weights and phase gates.

There is no approximation, truncation, pruning, changed chance layer, changed
node budget, partial search, or alternate fallback.

## Exact Leaf Contract

The current incumbent has 21 identical patterns across four value components:
base MC1000, Student1, phase4 replay calibration, and phase4 endgame sidecar.
The native evaluator receives immutable NumPy table pointers from those exact
loaded checkpoints.

For every board:

1. Convert each supported tile value through the exact frozen rank lookup.
   Unsupported negative or out-of-range values fail closed.
2. Determine phase after removing exactly one `1536` starter, preferring
   position `(0,0)` and otherwise the first row-major `1536`.
3. Use phase boundaries `<384`, `<1536`, `<3072`, and `>=3072`.
4. For each model, accumulate `float32` table entries in a `double` scalar in
   exact symmetry-major, pattern-minor order, matching
   `VectorizedCompositeLeaf` flatten/cumulative-sum order.
5. Combine model values in exact incumbent order:
   - phase 0: `0.75*base + 0.25*student`;
   - phases 1-2: `0.70*base + 0.25*student + 0.05*replaycal`;
   - phase 3: `0.60*base + 0.25*student + 0.05*replaycal +
     0.10*endgame`.

Pattern cells, symmetry permutations, rank lookup, component identities,
table lengths, dtypes, contiguity, stages, and phase coefficients are bound in
the final build manifest. Any mismatch fails before loading.

## Exact Transition Contract

Actions use incumbent order `up=0`, `down=1`, `left=2`, `right=3`.

`k1_base_move` reproduces `advance_line_toward_start` literally:

- each cell advances at most one position per swipe;
- `1+2` and `2+1` merge to `3`;
- equal values `>=3` merge unless the value is the terminal tile;
- a moved-into or merged-into cell cannot merge again that swipe;
- eligible spawn positions are emitted in lane-index order and only for
  changed lanes whose insertion-edge cell is zero.

`k1_score_board` reproduces the exact Threes score mapping and rejects invalid
tile values. Native transition calls may not consume or mutate deck, slot, or
policy RNG state.

## Development Evidence

Development may use only spent C1 engineering states and C2 fit/validation
states. The C2 untouched runtime states are forbidden. Development may test:

- every action on crafted and spent boards;
- leaf values on spent boards;
- exact C1 depth-2/depth-3 values and selected actions on spent states;
- deterministic build and reload hashes.

No development score analysis or policy outcome is permitted.

Equivalence tolerances are frozen:

- base-move boards and eligible slots: exact equality;
- score: exact integer equality;
- leaf values: absolute difference `<=1e-9`;
- depth-2/depth-3 legal action sets: exact equality;
- action values: absolute difference `<=1e-8`;
- selected actions under the same frozen tie seed: exact equality.

## Fresh Corpus

Fresh stream bases are:

- logical reset: `69_000_000_000`;
- deck: `70_000_000_000`;
- slot: `71_000_000_000`;
- policy: `72_000_000_000`.

For family index `f` and game index `g`, each stream is
`base + f*1_000_000 + g`.

Exactly three previously admitted genuine behavior families are frozen:

1. `k1_corner2`;
2. `k1_parent_mc1000`;
3. `k1_replaycal`.

Their immutable policy specs and accepted 64-state action signatures are bound
in the execution preflight. Exactly 36 complete normal-start games are run per
family, 108 total, with starter `1536`, maximum 5,000 moves, one worker, and no
early family stopping.

A root is eligible when its completed replay contains at least four natural
states where:

- the state is live and has legal actions;
- built max excluding the fixed starter is in `[768,3072)`;
- unchanged R2a trigger is true: empty count `<=3` or normalized incumbent
  top-two margin `<=0.02`.

For high-empty states, at most eight frames per root are screened by lowest
SHA-256 of
`"K1-high-empty-v1|root|frame|state_sha256"`. All low-empty frames are
eligible without value screening. Four decorrelated states are selected by
temporal quartile, then lowest SHA-256 of
`"K1-state-v1|root|frame|state_sha256"` within each quartile.

The first 12 eligible roots per family by immutable stream order are retained.
The root partition is frozen by eligible-root order before timing:

- indices 0-3: `fresh_equivalence`;
- indices 4-7: `engineering_validation`;
- indices 8-11: `untouched_runtime_gate`.

Each partition therefore has exactly 12 roots and 48 states, with four
roots/16 states from each family. Family share is exactly one third. The
entire corpus is whole-ancestry disjoint from every prior C1/R2a/G1/G2/G3/G4/
S3/C2 root and stream. The C2 untouched roots are explicitly excluded.

Game score, chosen actions, or downstream outcomes may not be used to select,
filter, summarize, or interpret this corpus.

## Frozen Build and Timing Schedule

The final source, wrapper, tests, compiler binary, flags, policy payloads,
corpus manifest, stream manifest, hardware, process settings, and dynamic
library hashes are sealed before the fresh gate.

Execution uses one process, one worker, `nice=10`, no competing heavy process,
and no thermal/power mode change. Each state receives:

1. one untimed depth-2 warmup and one untimed K1 warmup;
2. one exact C1 depth-3 oracle calculation for equivalence only;
3. five interleaved timed passes of unchanged depth 2 and full K1 combined
   depth2+depth3, alternating order by pass and record parity;
4. decision caches cleared before every timed arm.

Timing uses `time.perf_counter_ns`. For each state, arm latency is the median
of its five passes. Relative latency is K1 combined median divided by depth-2
median. No interim timing, action, value, admission, family, or partition
result may be opened before terminal sealing.

## Untouched Runtime Gate

All checks are conjunctive on the 48-state untouched runtime partition:

- relative median `<=3x`;
- relative p90 `<=5x`;
- relative p99 `<=8x`;
- relative max `<=12x`;
- absolute K1 p99 `<2.5s`;
- zero base-move, score, leaf, depth-2 value/action, or depth-3 value/action
  mismatch;
- compiled depth-3 activity exactly `100%`;
- activity present in all three families;
- no operational, provenance, collision, ancestry, family-cap, service,
  storage, or sealing failure.

Denominator and absolute latency distributions are both reported. Validation
and untouched partitions are descriptive separately; no threshold may be
relaxed after results.

Any scientific gate failure seals `KILL_K1_COMPILED_KERNEL`. A genuine
operational or integrity fault seals `HOLD_K1_ENGINEERING_FAULT`. A complete
pass seals `READY_K1_FULL_POLICY_PREFLIGHT`, which authorizes only a future
separately frozen full-policy proposal.

## Orchestration and Storage

- output: `threes_rl/runs/forensics/k1_compiled_kernel_v1`;
- open-only marker before compilation or fresh work;
- exact marker-bound, resumption-safe command;
- one heavy process;
- `nice=10`;
- active runtime limit: 12 hours;
- incremental output limit: 4 GiB;
- pause/fail below 100 GiB free; target at least 120 GiB;
- ports 8765 and 8770, advisor, dashboard record `263670`, and protected top
  three `263670/261369/258561` must remain healthy.

The marker, preflight, build, source, compiler, corpus, timings, terminal
decision, and compact retained evidence are immutable. No dashboard point,
incumbent change, policy outcome, h10/h20/h40 outcome, label, training fit, or
human-action use is permitted.

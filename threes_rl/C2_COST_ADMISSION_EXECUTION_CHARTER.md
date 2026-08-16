# C2 Deterministic Cost-Admission Execution Charter

Status: frozen before any C2 game, timing, exact-depth result, fit, admission,
or runtime-gate result is generated.

Date: 2026-07-26

## Scientific And Historical Boundary

C2 asks one engineering question: can one deterministic, monotone cost model
admit the retained exact C1 depth-3 calculation on a useful subset of eligible
states while satisfying the frozen latency and equivalence contract?

C1 remains permanently `FAIL_STOP_C1`. C2 may import the byte-locked
`c1_search_optimization.py` operator at SHA-256
`c12852cc7dcc8211d8ecc47ccf8c5598d6055a5f12a9bcec497dc47715e0e789`
as an immutable oracle. C2 does not modify, tune, rerun, or reinterpret C1.
The R2a trigger, chance limit `8`, value-node budget `2048`, incumbent leaf,
chance semantics, tie semantics, and fallback calculation are unchanged.

G2 acquisition is spent and may not be extended. G3 marginal hazard and G4
conditional pairwise ranking are permanently killed. G3 transfer artifacts
remain unopened. Human actions are not labels. No game score, future
milestone, recorded action, or policy outcome enters C2.

## Fresh Corpus Contract

### Collector families

The exact family order and policy specs are:

1. `c2_corner2`: `corner2`
2. `c2_parent_mc1000`:
   `ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest`
3. `c2_replaycal`:
   `ntuple_expectimax2:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest`

Their immutable G1-R action signatures are, in the same order:

1. `4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043`
2. `e43dc11f3220557d7f9aef228db96dc6f06f49b26300d5a4128ea00bf8ba2064`
3. `e07c566b55d86a889ab7ca54d01c00c9b6cdf808fdb1627f70596bd829fdeab3`

The accepted panel audit reports pairwise disagreement `53.125%`, `51.5625%`,
and `15.625%` overall, with nonzero disagreement in both pre1536 and pre3072
strata. These are three genuine behavioral families, not checkpoint aliases.

### Streams and games

Generate exactly `72` complete normal-start games per family, `216` total,
with starter tile `1536` and maximum `5000` moves. Do not stop a family early.
Use one process and this exact split-stream mapping:

```text
offset = family_index * 1_000_000 + game_index
logical_seed     = 65_000_000_000 + offset
deck_stream_id   = 66_000_000_000 + offset
slot_stream_id   = 67_000_000_000 + offset
policy_stream_id = 68_000_000_000 + offset
```

All `864` IDs must be internally unique and collision-free against the full
historical stream union before opening and again before the first game.
Each game is one fresh whole ancestry with direct-root origin and exact reset
invariants. Partial, human, restart, continuation, synthetic, or copied replay
sources are forbidden.

Generation order is family round-robin in chunks of at most `6`, but output
identity is the immutable `(family_index, game_index)` row. A compact
completion row is retained for every attempt. Full replay/state data is
retained only for roots selected below. Final score may exist inside the
simulator replay format as unavoidable provenance, but C2 code may not read,
filter, summarize, report, or select on it.

### Root qualification and state selection

A complete root qualifies when its frozen candidate screen contains at least
four frames satisfying all of:

- non-starter built maximum is in `[768, 3072)`;
- the state restores exactly, including board, preview, tile-cycle/deck state,
  pending state, move count, score field, and starter;
- unchanged depth-2 action values are finite;
- the unchanged R2a trigger is true:
  `empty_count <= 3 OR normalized_top_two_margin <= 0.02`;
- the state is nonterminal and has at least one legal action.

Qualification uses no wall time, depth-3 result, final score, future frame,
future milestone, or recorded action.

To bound outcome-free extraction cost, the candidate screen includes every
frame in the built-max range with `empty_count <= 3`, plus at most eight
higher-empty frames per root. The latter are the SHA-256 argmin frames under

```text
SHA256("C2-high-empty-screen-v1"|root_ancestry|frame_index|canonical_state_hash)
```

and qualify only when the computed depth-2 normalized margin is at most
`0.02`. Depth-2 values and margin are then computed for every selected state,
including low-empty states. Unscreened higher-empty frames are not corpus
candidates. This screen is fixed before games and does not inspect timing or
future outcomes.

For each family, retain the first `12` qualifying roots in ascending immutable
game index after all `72` games finish. There is no favorable-root choice.
Assign roots by that order:

- roots `0..5`: `cost_fit`;
- roots `6..7`: `engineering_validation`;
- roots `8..11`: `untouched_runtime_gate`.

This yields exactly `6/2/4` roots per family and family shares of `1/3` in
every partition. No root or game appears in more than one partition.

Within each retained root, sort eligible frames by frame index and assign each
frame at position `i` among `n` frames to bucket
`min(3, floor(4*i/n))`. In each of the four buckets select the unique argmin of

```text
SHA256("C2-state-v1"|root_ancestry|frame_index|canonical_state_hash)
```

with frame index as the final tie break. This selects exactly four temporally
spread states per root. The frozen corpus therefore contains:

- `72` fit states from `18` roots;
- `24` validation states from `6` roots;
- `48` untouched-gate states from `12` roots;
- `144` states from `36` independent ancestries overall.

If any family has fewer than `12` qualifying roots, or any retained root lacks
four selected states, the exact C2 design seals `KILL_C2_COST_ADMISSION` at the
corpus-yield stage. Quotas are never reallocated.

### Exclusions and provenance

Before generation, hash an explicit exclusion manifest containing roots and
sources from every prior C1, R2a, G1, G2, G3, G4, and S3 corpus or gate.
Fresh requested roots and streams must have zero overlap. After generation,
every retained replay hash, direct-root field, stream tuple, frame, and exact
state restoration is revalidated before timing. A new external overlap,
source mutation, or provenance failure is an integrity fault.

## Frozen Cost Representation

C2 runs an instrumented but value-exact depth-2 calculation. Counters are reset
per decision and are computed before admission. Wall time is never an input.
The ordered feature vector has exactly 20 finite nonnegative columns:

1. `legal_actions = root_legal_count / 4`
2. `empty_fraction = empty_count / 16`
3. `preview_red = 1[preview.kind == "red"]`
4. `preview_blue = 1[preview.kind == "blue"]`
5. `preview_gray = 1[preview.kind == "gray"]`
6. `preview_bonus = 1[preview.kind == "bonus"]`
7. `preview_candidate_fraction = len(preview.candidates) / 3`
8. `low_margin_pressure = clip((0.02 - margin) / 0.02, 0, 1)`
9. `low_empty_pressure = clip((3 - empty_count) / 3, 0, 1)`
10. `action_calls = clip(log1p(action_calls)/log1p(64), 0, 1)`
11. `value_lookups = clip(log1p(value_lookups)/log1p(4096), 0, 1)`
12. `unique_value_states = clip(log1p(unique_value_states)/log1p(2048), 0, 1)`
13. `chance_calls = clip(log1p(chance_calls)/log1p(2048), 0, 1)`
14. `chance_outcomes = clip(log1p(chance_outcomes)/log1p(8192), 0, 1)`
15. `afterstate_lookups = clip(log1p(afterstate_lookups)/log1p(16384), 0, 1)`
16. `unique_afterstates = clip(log1p(unique_afterstates)/log1p(8192), 0, 1)`
17. `base_move_calls = clip(log1p(base_move_calls)/log1p(32768), 0, 1)`
18. `unique_base_moves = clip(log1p(unique_base_moves)/log1p(16384), 0, 1)`
19. `legal_lookup_calls = clip(log1p(legal_lookup_calls)/log1p(4096), 0, 1)`
20. `cheap_depth2_pressure = 1 / (1 + depth2_work_units / 256)`

where

```text
depth2_work_units =
    unique_value_states
  + unique_afterstates
  + chance_outcomes / 4
  + unique_base_moves / 4
```

Unknown preview kinds, negative counters, nonfinite values, feature-width
changes, or feature-order/hash changes fail closed. Instrumentation may not
change legal actions, depth-2 values, or selected actions.

## Frozen Cost Target And Model

For each fit or validation state, after one untimed warm-up, run three
interleaved timing repetitions. Alternate plain depth 2 first and exact C1
combined first by deterministic `(state_index + repeat) % 2`. Use per-state
medians.

The single scalar training target is:

```text
absolute_load = exact_C1_combined_seconds / 2.0
relative_load = (exact_C1_combined_seconds / plain_depth2_seconds) / 6.0
safety_load = max(absolute_load, relative_load)
```

The `2.0 s` and `6.0x` scales are frozen safety margins below the untouched
gate's `2.5 s` p99 and `8.0x` p99 limits. Historical C1 timings justify them
without using C2 outcomes.

Fit exactly one nonnegative linear ridge model with a nonnegative intercept:

```text
prediction = intercept + sum(coef_i * feature_i)
coef_i >= 0
intercept >= 0
L2 lambda = 0.001
```

Use deterministic SciPy `lsq_linear`, tolerance `1e-12`, maximum `10,000`
iterations, and the root-equal/family-equal fit weights implied by the exact
balanced corpus. There is no standardization, calibration model, alternate
target, feature selection, restart, seed selection, or sweep. A repeated fit
must reproduce coefficients and predictions within `1e-12`.

The fixed conservative upper estimate and admission rule are:

```text
upper_load = 1.25 * max(0, prediction) + 0.10
admit_exact_depth3 = eligible AND upper_load <= 1.0
```

No validation result changes `1.25`, `0.10`, `2.0`, `6.0`, or the threshold.

## Validation Gate

The fixed model proceeds to the untouched runtime gate only if validation has:

- solver success, finite nonnegative coefficients, and deterministic refit;
- exact depth-2 value/action equality on every state;
- Spearman correlation between predicted and observed safety load at least
  `0.25`;
- root-equal mean absolute safety-load error at most `0.35`;
- root-equal p90 absolute error at most `0.75`;
- at least `90%` one-sided coverage:
  `observed_safety_load <= upper_load`;
- predicted admission activity at least `15%` and at least one admitted state
  in each of the three families;
- every admitted validation state has measured exact C1 combined latency at
  most `2.5 s` and measured ratio at most `8.0x`;
- exact C1 equivalence for every validation oracle calculation.

Any failure seals `KILL_C2_COST_ADMISSION`. Thresholds are not relaxed.

## Exact Operator And Equivalence

C2 never truncates an admitted depth-3 search.

- Trigger-false or cost-rejected states return the instrumented depth-2
  values. These must match the plain incumbent depth-2 legal actions and values
  within `1e-9 * max(1, abs(reference))`, and deterministic selected actions
  must match exactly.
- Cost-admitted states complete the retained C1 depth-3 calculation with
  chance limit `8`, node budget `2048`, identical cache/fallback semantics,
  and no partial return. Values and selected actions must match a fresh
  byte-locked C1 oracle under the same tolerance and tie seed.
- Repeated calls on the same state must make the same admission and return the
  same values/action.

Any value, action, legal-set, trigger, budget, or determinism mismatch kills
C2.

## Untouched Runtime Gate

The gate is opened once after fit and validation pass. For every untouched
state, perform one untimed warm-up and three interleaved repetitions of plain
depth 2 and the complete C2 operator. Alternate order by deterministic state
index and repeat. Evaluate the distribution of per-state median C2/depth-2
ratios and absolute C2 latency.

All must pass:

- median ratio `<= 3.0x`;
- p90 ratio `<= 5.0x`;
- p99 ratio `<= 8.0x`;
- maximum ratio `<= 12.0x`;
- absolute C2 p99 `< 2.5 s`;
- zero value/action/legal/trigger mismatch;
- depth-3 admission activity `>=15%` of gate states;
- admitted states occur in all three families;
- each family is at most `40%` of gate states;
- admitted-state upper-load coverage `>=90%`;
- admitted-state root-equal mean absolute load error `<=0.35`;
- no integrity, provenance, service, storage, or determinism failure.

Report absolute depth-2 and C2 latencies together with ratios so denominator
effects remain visible. A failed activity, calibration, equivalence, absolute,
or relative gate seals `KILL_C2_COST_ADMISSION`. There is no threshold
adaptation.

## One-Shot Orchestration

Output namespace:
`threes_rl/runs/forensics/c2_cost_admission_v1`.

The zero-timing preflight must bind this charter, runner, tests/evidence,
policy payloads, incumbent, C1/R2a operator and artifacts, exclusion manifest,
corpus plan, exact `216`-row stream manifest, source hashes, feature schema,
commands, resources, and service/dashboard truth.

The open command creates one immutable `C2_EXECUTION_OPENED.json` and exits
before any game, stream, state selection, timing, exact-depth result, model, or
admission. The marker binds the exact execute command. Execute rejects a
missing or mismatched marker and is resumption-safe under the same marker.
There is one execution only.

Resources are frozen:

- jobs `1`;
- nice priority at least `10`;
- one heavy process;
- active wall limit `6 hours`;
- incremental output limit `4 GiB`;
- disk hard floor `100 GiB`, target `120 GiB`;
- deterministic chunks of at most `6` games;
- healthy dashboard `8765`, advisor `8770`, record `263670`, and protected top
  three `263670/261369/258561`.

No interim timing, coefficient, value, action, admission, family result, or
score is reported. Only phase/completeness/integrity/service/storage may be
monitored.

## Terminal Decisions

Seal exactly one immutable terminal:

- `READY_C2_FULL_POLICY_PREFLIGHT` only if corpus, validation, exactness,
  activity, calibration, runtime, provenance, and operations all pass;
- `KILL_C2_COST_ADMISSION` for any scientific, cost, activity, calibration,
  equivalence, corpus-yield, or tail failure;
- `HOLD_C2_ENGINEERING_FAULT` only for a genuine operational/integrity fault
  that prevents the frozen assay from completing.

All terminals are non-promotable. `READY_C2_FULL_POLICY_PREFLIGHT` authorizes
only a separately frozen, fresh-root full-policy causal proposal. That future
proposal must expose C2 on every eligible move in its arm, use shared fresh
streams and normal starts or complete 40-move continuations, preregister
base-rate-appropriate effect size and power, and use whole-ancestry inference.
It is not authorized here.

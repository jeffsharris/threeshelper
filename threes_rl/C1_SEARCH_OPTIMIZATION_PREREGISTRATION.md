# C1 Exact Adaptive-Search Optimization Preregistration

Status: frozen on 2026-07-11 before C1 profiling or optimization measurements.
C1 is engineering-only and makes no policy or capability claim.

## Immutable Reference

The reference calculation is the killed R2a implementation and configuration:

- exact incumbent composite leaf and legal actions;
- depth 2 first, then depth 3 only for non-starter built max in `[768,3072)`,
  empty count at most 3 or normalized top-two margin at most 0.02;
- depth-3 chance limit 8 and deterministic 2,048 expanded-value-node budget;
- identical preview, tile-cycle, slot, next-value, score-delta, tie, and budget
  fallback semantics;
- same action order and deterministic selection seed.

`R2A_ROOT_MANIFEST.json` and all A2/R2a KILL records remain immutable. C1 may
not alter the trigger, chance limit, node budget, leaf, or search objective.

## Frozen Engineering Corpus

Use naturally reachable A1 `train` roots only. They are disjoint by canonical
root from the ancestry/corner2 holdouts used by R2a. Select at most one
built-768/1536 triggered state per root, deterministic and source-diverse.

Freeze three disjoint partitions before profiling:

- `profile`: 12 roots, at least three behavioral families;
- `equivalence`: 24 roots, at least three families;
- `runtime_gate`: 48 roots, at least three families.

No final score, future trajectory, human choice, or rollout outcome enters
selection. The runtime gate is not inspected while implementing optimizations.

## Exact Equivalence Contract

For every equivalence and runtime-gate state:

- legal action set and trigger decision match exactly;
- every depth-2 and depth-3 root action value differs by at most
  `1e-9 * max(1, abs(reference))`;
- deterministic chosen actions match exactly;
- node-budget cutoffs and fallback semantics match;
- preview/chance outcome probabilities and resulting simulator states match;
- repeated runs and save/reload-equivalent processes are deterministic.

Any coherent optimization that fails this contract is rejected, not tuned.
When reference search reaches its node cutoff, optimized code must fall back to
the unoptimized depth-dependent search path unless it can prove identical
budget accounting.

## Profiling Contract

Run three timed repetitions per profile root after one untimed warm-up. Report:

- standalone depth-2, standalone reference depth-3, and combined adaptive time;
- cumulative player/action, chance expansion, leaf, and transposition lookup
  time and calls;
- unique value states, repeated/cache-hit states, action-cache and afterstate
  cache hit rates, expanded nodes, budget cutoffs, and chance outcomes;
- peak Python allocations when measurement overhead is acceptable;
- median/p90/p99/max depth3/depth2 and combined/depth2 ratios.

No score outcomes are generated or inspected.

## Bounded Optimization Sequence

Implement and benchmark coherent steps in order, retaining each only if exact:

1. Iterative deepening in one policy instance, reusing depth-independent
   base-move, score, legal, afterstate, and post-spawn caches plus safe
   depth-keyed work. Cutoff cases fall back to the reference path.
2. Bounded cross-decision exact transposition/chance reuse keyed by full
   board+preview+cycle state and all search parameters.
3. Memoized chance preview/slot expansions with immutable copied outcomes.
4. Depth-2 action ordering and fair deterministic per-action accounting only if
   it reproduces the reference calculation; otherwise reject without tuning.

Measure after each coherent retained step. This is not a parameter sweep.

## Runtime Gate

Open the frozen runtime-gate split only after implementation and equivalence
pass. The unchanged combined adaptive calculation must achieve:

- median combined/depth2 ratio at most 3.0;
- p90 ratio at most 5.0;
- p99 ratio at most 8.0 and maximum at most 12.0;
- no equivalence, determinism, or long-tail integrity failure.

Failure stops C1 and reports the irreducible hotspot. Do not tune the node
budget or run R2 outcomes. A pass permits a new R2b causal preregistration on
fresh roots only.

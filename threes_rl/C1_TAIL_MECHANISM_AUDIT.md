# C1 Tail-Root Existing-Output Audit

Date: 2026-07-12

## Scope And Locks

This is a read-only mechanism audit of the two p99 runtime-gate roots. It does
not reopen C1, rerun a benchmark, inspect gameplay outcomes, change the C1
implementation, or alter `FAIL_STOP_C1`.

Inputs are only the frozen `C1_CORPUS.json`, `C1_RUNTIME_GATE.json`,
`C1_REFERENCE_PROFILE.json`, and `C1_BYTEKEY_LEAF_BENCHMARK.json` artifacts.
The runtime gate did not persist per-root leaf calls, unique afterstate boards,
chance outcomes, or cache counters. Those quantities are therefore unavailable
for the two tail roots. Profile-split analogs below are explicitly mechanism
context, not reconstructed tail measurements.

## Matched Gate Comparison

For each tail root, three ordinary gate controls were selected by deterministic
nearest-state distance over stage, empties, legal actions, preview kind,
pending status, log support mass, and incumbent margin. No exact context-cell
control existed. The selected controls all happened to be below the gate p90;
none was removed after selection.

| Metric | Tail `450f...` | Control median | Tail `e31c...` | Control median |
|---|---:|---:|---:|---:|
| Standalone depth-2 seconds | 0.115 | 0.422 | 0.012 | 0.085 |
| Reference depth-3 seconds | 4.725 | 5.420 | 0.298 | 0.443 |
| Optimized combined seconds | 1.330 | 1.523 | 0.118 | 0.181 |
| Optimized/depth-2 ratio | 11.58x | 3.12x | 9.55x | 1.87x |
| Reference depth-3/depth-2 | 41.16x | 12.69x | 24.17x | 5.03x |
| Reference-combined speedup | 3.64x | 3.82x | 2.64x | 2.91x |

The first tail's optimized absolute latency is at the 77th percentile of the
48-root gate and the second is at the 29th percentile. Neither is an absolute
latency outlier: gate p99 optimized latency is 2.364 seconds and the maximum is
2.503 seconds. Both roots instead have unusually cheap standalone depth-2
search. Relative to matched controls, their denominators are 73% and 85%
smaller while optimized absolute time is also 13% and 35% smaller.

## Context And Search Shape

| Field | Tail `450f...` | Tail `e31c...` |
|---|---|---|
| Stage / built max | late-1536 / 1536 | late-1536 / 1536 |
| Empties / legal actions | 3 / 4 | 0 / 2 |
| Preview / P(plus) / pending | blue / 0 / false | red / 0 / false |
| Incumbent margin | 0.0136 | 0.2071 |
| Trigger | low-empty and low-margin | low-empty only |
| Top edge descending | false | true |
| Support score mass | 144 | 894 |

The two tails do not share one narrow preview, congestion, action-count, margin,
or geometry signature. What they share is a large depth-3/depth-2 work ratio.
That makes a generic cache defect less likely and a structural denominator/
branching interaction more likely.

Existing reference-profile analogs support, but do not prove, that search shape
drives this interaction:

| Existing profile analog | `450f...`-like | `e31c...`-like |
|---|---:|---:|
| Context match | same stage/3 empties/4 legal/blue | same stage/0 empties/2 legal; blue not red |
| Afterstate lookups / hits | 10,856 / 2,970 | 3,160 / 556 |
| Implied unique afterstate boards | 7,886 | 2,604 |
| Chance calls / outcomes | 1,696 / 3,913 | 791 / 1,385 |
| Unique value states | 451 | 244 |
| Transposition hits / lookups | 36 / 487 | 18 / 262 |
| Leaf time / total depth-3 time | 12.435 / 13.662 s | 4.266 / 4.687 s |

Across the full profile trace, leaf evaluation consumed about 413 of 456
depth-3 seconds. Afterstate and transposition hit rates were only 17.0% and
9.3%; the optimized chance cache recorded 120 hits versus 61,641 misses. The
remaining exact work is therefore dominated by many distinct board-only leaf
evaluations inside state-dependent depth-3 trees, not preview enumeration,
cross-decision reuse, or the node budget.

## Attribution

1. **Primary: denominator effect.** Both p99 failures are ordinary-to-low
   absolute latency states divided by exceptionally cheap depth-2 baselines.
2. **Secondary: structural search expansion.** Depth 3 performs 24-41 times the
   standalone depth-2 work on the tails, versus 5-13 times for matched controls.
   Existing analog traces associate these shapes with thousands of mostly
   unique afterstate boards and low cache reuse.
3. **Not supported:** a stage-specific, plus-preview, pending-tile, node-cutoff,
   or chance-expansion defect. The tails differ on those contexts, cutoffs are
   zero, and chance work was a small part of the reference profile.

The frozen C1 failure remains correct: its contract used a relative p99 limit,
and that limit failed. The audit only explains why.

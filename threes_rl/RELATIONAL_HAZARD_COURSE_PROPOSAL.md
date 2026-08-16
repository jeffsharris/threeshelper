# Relational Hazard Research Course Proposal

Date: 2026-07-25

Status: revised. The S3 full-policy scientific utility assay is authorized
under `S3_FULL_POLICY_UTILITY_CHARTER.md`; all later phases remain conditional.

## Research-Lead Decision

The external feedback contains a strong scientific correction but overstates
parts of the historical diagnosis.

Adopt:

- Measure the gameplay utility of frozen exact depth-3 decisions before
  spending more work on depth-3 admission or latency.
- Use a preregistered h40 milestone/progression hazard as the primary local
  endpoint, with score, survival, anchor preservation, and geometry as
  safeguards and mechanism readouts.
- Test an explicit relational, action-conditioned hazard representation.
- Elicit the human player's multi-move strategy as feature and constraint
  hypotheses, never as action labels.
- Separate exploratory scientific failures from immutable data and promotion
  locks.

Do not adopt:

- The claim that geometry has never entered a policy-facing component. The
  program already tested a hand-built geometry bonus, geometry-aware action
  features, and n-tuple reachability sidecars. Those variants failed
  same-start continuation gates.
- The claim that fixed n-tuples make geometry mathematically unrepresentable.
  They can encode some positional geometry, especially when relevant cells
  co-occur in a tuple, but the current local additive tables are
  sample-inefficient and aliased for relational, long-range properties.
- The claim that the `4.17%` and `5.15%` transition rates describe the whole
  program. They came from one fresh ancestry, root `1074`, dominated by
  Chebyshev-distance-3 geometry. Other root-diverse, mostly distance-2
  duplicate-1536 pools produced materially higher h40 rates. The event is
  useful, but its base rate is strongly state-distribution dependent.
- The claim that adjacent duplicate 1536 is already proven nearly sufficient
  for final score under normal-start play. Downstream conversion is easy in
  selected continuation pockets; sufficiency and transport to natural play
  remain unproven.

## Governance Amendment Required

The following historical evidence remains immutable:

- R1b confirmation remains spent and failed.
- A2 remains killed.
- R2a failed its frozen engineering prescreen.
- C1 remains `FAIL_STOP_C1` under its frozen latency contract.
- C1's p99 miss remains correctly attributed primarily to cheap depth-2
  denominators, with structural distinct-leaf expansion secondary.

This proposal does not reinterpret those outcomes.

It does require explicit authorization for a new branch, S3, that reuses the
frozen exact depth-3 calculation as an experimental treatment. S3 asks a
different question from R2a/C1:

> On fresh, ancestry-disjoint states, are the actions selected by frozen exact
> depth 3 causally better than frozen depth 2?

No search trigger, depth, leaf, floating-point order, tie rule, or cache
implementation may be changed. Runtime is recorded as engineering metadata but
is not an eligibility gate for this scientific assay.

## Program Order

1. Run S3, the frozen full-policy depth-3 scientific utility assay.
2. Decide C2 from S3:
   - S3 pass: C2 may be considered as an engineering bridge.
   - S3 fail: kill C2 because admitting a causally neutral search is useless.
3. Run G1, an action-conditioned relational hazard representation test, using
   a separate whole-ancestry corpus.
4. Consider a goal-conditioned planner only if G1 generalizes and improves
   first-action causal outcomes.
5. Open the human board-scenario training ground only if these self-learning
   branches are genuinely exhausted.

Only one heavy job may run at a time.

## Deferred Human Strategy Elicitation

This phase is no longer on the immediate critical path. The user has requested
that the solver first exhaust its self-learning options. If opened later, its
purpose is to convert the expert human's long-horizon intent into candidate
features and constraints, not labels.

Ask the player:

1. When the first non-starter 768 appears, what board shape are you trying to
   create over the next 20-40 moves?
2. Where should the built 1536 land relative to the fixed starter, and which
   alternate landing configurations are still recoverable?
3. What distinguishes a recoverable duplicate-1536 board from a board that is
   already strategically lost?
4. Which moves or structural changes are treated as forbidden even when they
   give immediate score or empty cells?
5. What support tiles must remain available, and where?
6. How do top-row and left-column monotonicity affect the plan?
7. How do preview color, bonus probability, and deck-cycle position alter the
   plan?
8. What conditions cause the player to abandon one plan and switch to another?

If this deferred phase is authorized, translate the answers into:

- a short feature dictionary;
- candidate hard constraints;
- an enumerable set of target configurations;
- explicit counterexamples where a seemingly good geometry is misleading.

The human's chosen action is never a training label.

## Phase II: S3 Frozen Depth-3 Scientific Utility Assay

The first-action protocol below is superseded by
`S3_FULL_POLICY_UTILITY_CHARTER.md`. S3 now applies the frozen treatment policy
throughout each h40 continuation, requires a pre-outcome MDE/power audit, uses
a strata-standardized common odds ratio, and has PASS/FAIL/INCONCLUSIVE
decisions. The text below is retained as historical proposal context only.

### Corpus

- Use only fresh, naturally reachable, whole-ancestry-disjoint roots.
- Exclude every R2a root, every C1 split, R1/R1b gates, selector gates, human
  roots, and prior continuation-outcome roots.
- Select at most one state per ancestry.
- Require at least 96 roots across at least three behavior families.
- Cap the largest family at 40%.
- Balance pre-1536 and pre-3072 states, success-window and failure-control
  roles, congestion, incumbent margin, and trigger reason.
- Freeze the root manifest and all hashes before any continuation outcome.

### Actions

- Compute the frozen depth-2 and frozen exact depth-3 actions once per root.
- Record same-action roots as structural zeroes.
- Require at least 20 changed-action roots across at least three families for a
  powered decision. Otherwise S3 stops as `FAIL_INACTIVE`.
- Force each selected first action, then use the frozen depth-2 incumbent for
  every continuation move. This isolates first-action quality.

### Rollouts

- Use 16 common-random-number replicates per root/action.
- Split replicates into two independent eight-stream blocks.
- One h40 rollout supplies the h10, h20, and h40 readouts.
- Pair deck, slot, preview, and policy streams across the compared actions.
- No partial outcome inspection and no sequential threshold adjustment.

### Endpoints

Primary:

- ancestry-balanced paired h40 probability of reaching the next natural
  milestone: first non-starter 1536 for pre-1536 roots and first non-starter
  3072 for pre-3072 roots.

Secondary:

- h10 and h20 milestone probability;
- h40 score delta;
- survival;
- starter-anchor preservation;
- terminal maximum-tile displacement;
- empty count and legal-action count;
- stage-appropriate geometry progression:
  duplicate 1536, near-adjacent 1536, adjacent 1536, and second 3072.

The geometry metrics diagnose mechanism. They do not replace the primary
endpoint after results are seen.

### Inference

- Weight roots equally, not rollout rows.
- Cluster bootstrap by whole ancestry.
- Report changed roots, same-action roots, behavior families, trigger reasons,
  and both independent stream blocks.
- Report absolute percentage-point effects and confidence intervals.
- Do not pool h10/h20/h40 into a favorable post hoc endpoint.

### S3 Gate

`PASS_SCIENTIFIC_UTILITY` requires all:

- at least 20 changed roots across at least three families;
- h40 next-milestone lift at least `+3.0 pp`;
- ancestry-cluster 95% lower confidence bound at least `0`;
- both independent stream blocks positive on the primary endpoint;
- h40 score effect nonnegative in both blocks, or aggregate score lower bound
  no worse than a preregistered `-5%` non-inferiority margin;
- survival no worse than `-2 pp`;
- anchor preservation no worse than `-3 pp`;
- no concentrated catastrophic family or ancestry failure.

Failing this gate kills S3 and C2. Passing it promotes only the scientific
hypothesis that exact depth-3 decisions help. It does not promote a policy.

## Phase III: C2 Conditional Engineering

C2 remains **HOLD** until S3 passes.

If S3 passes, the existing C2 proposal may be authorized unchanged:

- one outcome-free monotone cost model;
- fresh whole-ancestry fit/validation/untouched partitions;
- exact depth 3 for admitted states and unchanged depth 2 otherwise;
- no wall-clock policy input or post-admission truncation;
- median `<=3x`, p90 `<=5x`, p99 `<=8x`, max `<=12x`;
- absolute p99 `<=2.5s`;
- exact values/actions;
- depth-3 activity `>=15%` across at least three families.

If S3 fails, C2 is **KILL**. There is no value in optimizing admission for a
search treatment without demonstrated causal utility.

## Phase IV: G1 Relational Action-Hazard Representation

G1 is distinct from:

- the failed hand-written scalar geometry bonus;
- the failed geometry-risk action-prior labels;
- the failed n-tuple reachability value bonuses;
- A2's preview/deck-context residual.

G1 predicts an action-conditioned h40 milestone hazard from explicit
relational afterstate features.

### Fresh Data

- Use a separate corpus from S3.
- Partition whole ancestries into train, validation, and untouched test before
  rollout labels.
- Require at least five behavior families with no family above 40%.
- Evaluate every legal first action with shared h40 streams and frozen
  depth-2 continuation.
- Use empirical binomial success probability per root/action as the target.
- Keep success and failure controls at every milestone stratum.

### Frozen Relational Feature Family

The final feature list is frozen after the human interview and before labels.
It may include:

- current support-ladder stage;
- positions, Manhattan distance, and Chebyshev distance of target duplicates;
- pair orientation and whether the pair shares a row or column;
- blocker count and clear-line merge status;
- distance and orientation relative to the fixed starter;
- support-tile count, pair distance, components, and adjacency to the target;
- empty cells, legal actions, and local congestion around the target cells;
- top-row and left-column monotonic violations;
- starter-anchor integrity;
- current preview, large-pending state, and a compact deck-cycle summary.

Do not include final score, future outcomes, source identity, frame identity, or
wall-clock time.

### Models

Train exactly once:

1. A low-capacity regularized relational hazard model.
2. A capacity-controlled positional/cell-feature model as the representation
   control.

No architecture, feature, regularization, or threshold sweep is allowed after
validation is opened. The relational model should remain small enough to audit
directly; a preregistered logistic model with fixed interactions is preferred
over a generic large network.

### G1 Offline Gate

On untouched ancestries, require:

- relational Brier score and log loss better than the positional control;
- calibration slope and intercept inside preregistered tolerances;
- positive rank correlation between predicted and empirical action hazard;
- at least 20 model/incumbent action disagreements across at least three
  families;
- relational-selected action h40 milestone lift at least `+3 pp`, with
  ancestry-cluster lower bound at least `0`;
- both stream blocks nonnegative;
- survival, anchor, and score within the S3 no-harm margins.

Failure kills this relational feature/model formulation. It does not justify a
feature or regularization sweep on the same test set.

### Policy Integration

Only after the G1 offline gate passes may one frozen policy integration be
defined. The integration must not use an arbitrary uncalibrated scalar bonus.
Preferred options, in order:

1. a calibrated low-margin action reranker;
2. a multi-head leaf whose hazard-to-score scale is fixed from training data;
3. distillation of the verified action preference into a compact policy.

The integration receives a fresh first-action causal gate before any
normal-start evaluation.

## Phase V: Goal-Conditioned Planning

This is the most exciting longer-range hypothesis, but it is not the first
experiment.

The fixed starter reduces the goal space to a small set of target
configurations, such as placing the built 1536 adjacent to the starter along
the top row or left column while preserving support and anchor constraints.

Proceed only if G1 shows that relational progress is predictable and
actionable on untouched ancestries. Then compare one frozen goal-conditioned
leaf or macro-action planner against the verified G1 policy. Do not open a
planner sweep.

## Human H3 Integration

H3 remains held until a third substantial assisted ancestry completes.
Active or partial sessions remain excluded.

When unlocked:

- keep the existing compact 3-5 root, all-legal-action h10/h20/h40 audit;
- use the human action only as a query;
- use the resulting all-action outcomes as an independent transport check for
  the frozen relational feature dictionary;
- do not train on the human choice or claim normal-start capability.

## Immediate Decision

- **CONTINUE** S3 under `S3_FULL_POLICY_UTILITY_CHARTER.md`.
- **HOLD C2** until S3 demonstrates scientific utility.
- **HOLD** human strategy elicitation and the board-scenario training ground as
  last-resort paths.
- **AUTHORIZE NEXT:** S3 only.
- **KILL/PRESERVE:** all historical failed variants and sealed evidence remain
  unchanged.
- **PROMOTE:** nothing yet.

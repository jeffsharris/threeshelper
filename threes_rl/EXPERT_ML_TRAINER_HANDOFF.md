# Threes Solver ML Research Handoff

> Historical snapshot (2026-07-25). Use `RL_PROGRAM_HANDOFF.md` for the current
> program state and next-agent instructions.

Last updated: 2026-07-25

## Executive State

**CONTINUE:** S3, the full-policy exact-depth-3 scientific utility assay, is
authorized under `S3_FULL_POLICY_UTILITY_CHARTER.md`. It is currently restricted
to operational verification, fresh-root inventory, and the sealed outcome-blind
MDE/power preflight. Policy outcomes are permitted only if that preflight
selects a coherent design with at least 80% power for a common odds ratio of
1.50.

**PROMOTE:** Nothing new. The frozen dashboard-eligible record remains
`263,670`.

**KILL:** R1b, A2, R2a's failed engineering formulation, C1 under its frozen
runtime contract, direct human-action supervision, and several broad
fitting/search directions remain closed under their existing formulations.
S3 does not reinterpret those results; it uses the retained exact calculation
as a frozen treatment to answer the previously unmeasured gameplay question.

**HOLD:** C2 remains held pending S3 scientific utility. Human strategy
elicitation and a board-scenario training ground are deferred until the
self-learning branches are genuinely exhausted.

## Current Incumbent

The actor is `ntuple_phaseblend_expectimax2`:

- MC1000 parent value table.
- Student n-step/TC table blended at `0.25` in all phases.
- Replay-calibration table blended at `0.05` from midgame.
- Action-label/endgame sidecar blended at `0.10` during endgame.
- Depth-2 expectimax.

The exact policy specification is in `threes_rl/current_incumbent_policy.txt`.

Best eligible evaluation:

- Raw high score: `263,670`
- Score excluding seeded starter: `204,621`
- 20-game mean score-minus-starter: `34,576`
- Median score-minus-starter: `21,635`
- P(nonstarter 3072): `10%`
- P(nonstarter 6144): `0%`
- Interpretation: sparse tail improvement and a record game, but no broad
  median or 3072-rate breakthrough.

The global eligible top three remain `263,670 / 261,369 / 258,561`.

Corrected-start evaluations use score-minus-starter and maximum tile excluding
the seeded starter. Raw high score alone must not be used to promote a policy.

## Research Completed

### Value Learning

Work included Monte Carlo n-tuple fitting, n-step learning, temporal coherence,
larger pattern sets, student-as-actor training, replay calibration, and sparse
phase-specific sidecars. These produced the incumbent, but subsequent broad
fitting showed weak or negative transfer.

### R1b Policy Evaluation

A zero-initialized residual was trained on top of the exact incumbent.
Development D2 looked convincing:

- Paired mean score-minus-starter lift: `+4,506.57`
- 95% CI: `[+1,176.73, +7,894.16]`
- P3072: candidate `32/512`, incumbent `18/512`

The sealed confirmation did not replicate:

- Paired mean difference: `+788.18`
- 95% CI: `[-2,412.96, +4,021.11]`
- P3072: tied at `21/512`

Decision: **CONFIRMATION FAILED / NO PROMOTION**. The confirmation split is
spent. R1b must never be rerun, promoted, or reinterpreted from the positive D2
result.

### Context Modeling

H2 demonstrated that exact deck/context information can change good decisions.
The A2 compact board/context residual nevertheless failed source-disjoint error
and synthetic ranking-direction tests.

Decision: **KILL A2**. H2 establishes context sensitivity, not evidence for
direct human supervision or for fitting more variants of the same residual.

### Adaptive Exact Depth 3: R2a

R2a used depth 2 normally and exact incumbent-leaf depth 3 on triggered states.
It changed `23/64` held-out actions across three behavior families, proving that
the extra depth was behaviorally active.

Runtime relative to frozen depth 2:

- Median: `10.03x`
- P90: `16.42x`
- Max: `38.91x`
- Frozen limits: median `3x`, P90 `5x`

The `2048`-node cap never bound, so node-budget tuning was not the missing
ingredient. No continuation or policy outcomes were run.

Decision: **KILL R2a**.

### Exact Search Optimization: C1

C1 optimized the unchanged, exact R2a calculation. Its retained implementation
used batched grouped n-tuple evaluation, exact board-only leaf memoization
across chance contexts, and exact byte board keys.

All equivalence and gate roots matched exact values and actions.

Frozen untouched runtime gate:

- Median: `2.51x` against a `3x` limit
- P90: `3.89x` against a `5x` limit
- P99: `10.63x` against an `8x` limit
- Max: `11.58x` against a `12x` limit

Decision: **FAIL_STOP_C1** because P99 failed. C1 is permanently closed.

A read-only tail audit found that the P99 failure was driven primarily by
unusually cheap depth-2 denominators, not absolute optimized-latency outliers.
Distinct-leaf expansion was the likely secondary mechanism. Do not reopen C1,
relax its P99 gate, add cache variants, or reinterpret the result.

## Observed Strategic Bottleneck

Diagnostic rollouts suggest that once two 1536 tiles become adjacent, final
conversion is easy:

- Adjacent 1536s to second 3072: approximately `99.7%`
- Second 3072 to 6144: approximately `95.0%`

The scarce transitions occur earlier:

- Distance-3 duplicate geometry to near-adjacent diagonal-touch 1536s:
  approximately `4.2%`
- Near-adjacent 1536s to adjacent 1536s: approximately `5.1%`

These are diagnostic rates, not promotion evidence. They suggest that the
remaining problem is long-horizon board geometry and hazard management rather
than final tile conversion.

## Available Next Courses

### 1. S3 Full-Policy Scientific Utility Assay

This is the active authorized branch.

Control uses frozen depth 2 throughout each h40 continuation. Treatment uses
the retained exact C1-optimized R2a adaptive-depth policy throughout each h40
continuation, falling back to depth 2 only where the unchanged trigger is false.
Runtime is engineering metadata, not the scientific gate.

Before outcomes, S3 must freeze a fresh whole-ancestry-disjoint corpus and pass
an outcome-blind power preflight. The primary endpoint is the
strata-standardized h40 next-milestone odds ratio, with whole-ancestry
inference. Score, survival, anchor preservation, geometry, activity, and runtime
are safeguards and mechanism readouts.

The three-way decision is:

- `PASS_SCIENTIFIC_UTILITY`: C2 becomes eligible.
- `FAIL_NO_MEANINGFUL_UTILITY`: kill C2 and open G1.
- `HOLD_INCONCLUSIVE`: do not rerun blindly; prefer G1 unless a read-only audit
  identifies a clearly sufficient independent-root expansion.

### 2. C2 Deterministic Cost Admission

This is conditional on an S3 scientific pass and remains unauthorized.

Fit one deterministic, monotone, outcome-free model of exact depth-3
computational cost. Use exact C1 depth 3 only on admitted states and leave
rejected states exactly at depth 2. Wall-clock time is never a policy input,
and admitted searches cannot be truncated.

Required engineering protocol:

- New source-diverse corpus disjoint from R2a, every C1 split, and prior policy
  gates.
- Whole-ancestry partition into fit, validation, and untouched gate sets before
  timing.
- Exactly one preregistered nonnegative linear count model.
- No feature, regularization, threshold, or activity sweep.
- Rejected states exactly match depth 2; admitted states exactly match C1 depth
  3.

Frozen untouched gate:

- Median `<=3x`
- P90 `<=5x`
- P99 `<=8x`
- Max `<=12x`
- Absolute P99 `<=2.5s`
- No value or action mismatches
- Depth-3 activity `>=15%` across at least three families

An engineering pass permits only a fresh root-disjoint h10/h20/h40 causal
prescreen. It does not permit immediate training or promotion.

### 3. G1 Relational Action-Hazard Representation

This is the preferred new representation branch if S3 fails or remains
inconclusive. G1 predicts the h40 next-milestone hazard for every legal
afterstate from explicit relational geometry, support-ladder, congestion,
anchor, and compact context features. It uses fresh whole-ancestry
train/validation/test partitions and shared-stream h40 action labels.

G1 is materially different from the failed scalar geometry bonuses, broad
n-tuple sidecars, and A2 board/context residual. It targets the rare transition
directly and is action-conditioned rather than fitting terminal score.

### 4. Goal-Conditioned Planning

Open only if G1 generalizes and improves fresh first-action causal outcomes.
The fixed starter makes the useful target configurations enumerable, allowing a
small goal-conditioned value head or macro-action planner to maintain intent
across the 20-40 move bottleneck.

### 5. Compiled Exact Leaf Kernel

Fuse n-tuple leaf evaluation while preserving exact floating-point order. This
has lower expected value because C1's P99 failure was substantially
denominator-driven.

### 6. Abandon Exact Depth 3

Stop spending evidence on exact-depth variants until there is a materially new
representation or search family.

### 7. H3 Assisted-Play Audit

The frozen completed-session inventory contains two substantial assisted
ancestries. One more substantial completed game, preferably reaching a built
768 or larger, unlocks a compact decorrelated disagreement inventory.

The audit would select only 3-5 high-information roots and test every legal
first action with paired, shared-deck h10/h20/h40 continuations. Human choices
are queries, never labels. Active or partial games are excluded from analysis.

## Permanent Locks

Do not reopen, retune, or reinterpret:

- R1b or its spent confirmation split
- H0 direct human-action supervision
- A2 compact context residual
- R2a adaptive exact depth 3
- C1 exact search optimization
- UCT/MCTS under the previously tested framing
- Search node-budget sweeps
- Broad incumbent self-play fitting
- Broad action-prior or sidecar sweeps

Human-assisted sessions remain development and diagnostic evidence only.

## Questions for an Expert ML Trainer

1. Does S3's full-policy h40 design and strata-standardized milestone odds-ratio
   estimand answer the scientific utility question with adequate power?
2. What materially different representation could learn the rare long-horizon
   geometry transitions without repeating the failed board/context residual or
   broad self-play fitting approaches?
3. Would distributional value learning, successor features,
   option/goal-conditioned learning, hazard prediction, search distillation, or
   another objective be identifiable with the evidence already available?
4. How would you design one preregistered experiment that distinguishes
   representation failure, credit-assignment failure, and search-depth failure?
5. What should be the smallest causal prescreen, with ancestry-disjoint splits
   and explicit no-harm gates, before spending another large training block?

Any recommendation should respect the permanent locks above and the rule that
only one heavy job may run at a time.

## Operational State

As of 2026-07-25:

- The original training task has resumed S3 and is performing the outcome-blind
  root-availability and MDE/power preflight.
- Dashboard and assisted recorder respond successfully.
- Dashboard-eligible record remains `263,670`.
- The assisted recorder may contain an active partial game; partial sessions
  must not be inspected or counted.
- `152 GiB` disk space is free.
- `threes_rl/runs` occupies approximately `33 GiB`.
- Cleanup is allowed only through a reviewed manifest that preserves all
  incumbent, failed-branch, human-source, top-three, configuration, metric, and
  decision evidence.

## Primary References

- `threes_rl/S3_FULL_POLICY_UTILITY_CHARTER.md`
- `threes_rl/RELATIONAL_HAZARD_COURSE_PROPOSAL.md`
- `threes_rl/CURRENT_DECISION_LEDGER.md`
- `threes_rl/EXPERIMENT_LOG.md`
- `threes_rl/RESEARCH_COURSE_CHANGE_20260711.md`
- `threes_rl/C2_COST_ADMISSION_PROPOSAL.md`
- `threes_rl/R1B_PRE_C_DECISION.md`
- `threes_rl/C1_TAIL_MECHANISM_AUDIT.md`
- `threes_rl/current_incumbent_policy.txt`

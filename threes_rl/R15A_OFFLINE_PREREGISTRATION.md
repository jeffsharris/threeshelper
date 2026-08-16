# R1.5a Offline Context-Residual Preflight

Status: frozen before the 2026-07-11 source-diversity inventory. Label
generation and fitting are not authorized.

## Scope And Locks

R1.5a asks whether explicit simulator context improves source-disjoint
prediction over an equal-capacity board/stage-only residual. It is not a policy
experiment.

- R1b failed sealed C and remains permanently unpromoted. C is unavailable.
- H0 permanently killed direct human-action supervision.
- H2 established causal context sensitivity only. It did not establish that a
  context bonus or learned residual improves play.
- No rollout labels, fitting, normal-start evaluation, incumbent mutation, or
  dashboard eligibility may occur during this preflight.
- Synthetic same-board H2 swaps remain a separate diagnostic-only partition.
  They may never enter fitting or ordinary held-out metrics.

## Natural-State Source Catalog

Scan retained `replay.json` artifacts from evaluation replays, top-game
replays, stable replay copies, and exact human sessions. Accept a replay only
when:

- replay and root origin are genuine `fresh` or `human`;
- reset invariants pass;
- all retained states round-trip exactly and have a legal action;
- duplicate replay copies are removed by behavioral-family plus action-path
  signature;
- duplicate states are removed within root ancestry.

Behavioral families are coalesced as follows:

- current phaseblend lineage;
- cheap phaseblend approximation;
- corner2 lineage;
- non-learned expectimax baseline;
- TD/student/legacy learned lineage;
- legacy n-tuple lineage;
- random baseline;
- human observed.

Checkpoint names within one lineage do not create new family diversity.
Synthetic/frontier/replay-start/continuation artifacts are excluded from the
natural-state catalog even when they retain genuine root ancestry.

## Context Inventory

For every valid natural state, derive context after consuming the visible
preview using exact simulator mechanics:

- phase4 stage;
- visible preview kind and exact bonus candidates;
- next-preview `P(plus)`, full next-value probability vector, and
  `P(value | plus)`;
- post-visible small-bag probabilities and bag position;
- total-small count, span position, pending status, and distance to forced plus;
- empties, legal count, anchor state, top-edge ranks, support mass, and built
  maximum.

Frozen reporting bins:

- `P(plus next)`: `zero`, `(0,.10)`, `[.10,.25)`, `[.25,1]`;
- visible preview: `small` or `bonus`;
- empties: `0-1`, `2-3`, `4+`;
- each of four phase4 stages;
- pending yes/no and bag thirds `0-3`, `4-7`, `8-11`.

Report states and unique root ancestries for every marginal and joint bin.

## Root-Level Partitions

Partition whole canonical root ancestries before any labels:

1. `family_holdout`: every `corner2_lineage` ancestry.
2. `human_diagnostic`: every human ancestry; never fit or ordinary test.
3. Remaining nonhuman/non-synthetic ancestries: SHA-256 ancestry hash modulo
   five equal to zero becomes `ancestry_holdout`; the rest becomes `train`.
4. Within train and ancestry holdout, deterministically cap overrepresented
   families until no family exceeds `40%` of selected roots.

State selection is outcome-blind and deterministic:

- at most eight states per root ancestry;
- at most one state per ancestry/context cell
  `(stage, plus-bin, preview-bin, pending, empties-bin)`;
- round-robin over family, ancestry, and context cell;
- partition caps: train `1024`, ancestry holdout `256`, whole-family holdout
  `256`, human diagnostic `64`.

Trajectory success/failure may be reported after selection, but it may not
change partition membership, state weights, or sampling.

## Readiness Rules

The natural manifest is `READY` only when all hold:

- at least 100 selected train ancestries from at least three behavioral
  families;
- largest selected train family share at most `40%`;
- at least 25 ancestry-holdout roots;
- at least 20 whole-family-holdout roots;
- train and combined holdouts each contain all four phase4 stages;
- every plus-probability bin has at least 20 unique roots in train and at least
  five in combined nonhuman holdouts;
- train contains at least 20 unique roots in each pending stratum and each
  empties bin;
- exact state and provenance checks have zero failures.

If not ready, stop. Do not generate labels by relaxing these rules.

## Equal-Capacity Models

Freeze two models with identical parameter shapes and output heads:

- `board_stage_only`: encoded board/stage features occupy the first half of a
  fixed-width input; the context half is always zero.
- `board_plus_context`: identical board/stage features plus explicit mechanics
  context in the second half.

Both use input width 64, one tanh hidden layer of width 32, and identical
stage-aware output heads. The hidden layer uses the same deterministic initial
weights. Every output weight and bias starts at exact zero, so both residuals
are exact incumbent identity before fitting.

Frozen primary-h40 target heads:

- expected score-return residual;
- categorical return bins with edges
  `0,1k,2k,4k,8k,16k,32k,64k,+inf`;
- survival probability;
- first non-starter 1536 probability;
- first non-starter 3072 probability;
- anchor-preservation probability.

Auxiliary probabilities are calibration targets, not hand-weighted search
bonuses. Parameter count must be exactly equal between baselines.

## Frozen Multi-Step Target

One h40 frozen-incumbent continuation per replicate supplies h10/h20/h40
checkpoints. The primary horizon is exactly `H=40`; h10/h20 are diagnostics
and auxiliary calibration slices, not checkpoint-selection alternatives.

For every live checkpoint:

`y_H = score_accumulated_0_to_H + V_incumbent(s_H) - V_incumbent(s_0)`

For a terminal endpoint, the bootstrap term `V_incumbent(s_H)` is exactly
zero. `V_incumbent` is the frozen incumbent composite leaf evaluated on the
simulator-valid endpoint/root state; no learned context residual appears in
label generation. This is a simulator-consistent multi-step residual
correction, not raw short-horizon score.

The primary regression target is `y_40`. The categorical return head bins the
realized h40 multi-step target using the frozen edges above. Survival,
first-1536, first-3072, and anchor-preservation heads use h40 binary outcomes;
h10/h20 versions are retained in the label corpus for diagnostics.

## Frozen Label Contract

Labels remain unauthorized until readiness review passes. If authorized, the
frozen contract is:

- namespace `threes-r15a-labels-v1-20260711`;
- 16 replicates per ordinary natural state, blocks A/B of eight;
- independent logical/deck/slot/policy IDs, with identical model-comparison
  rows sharing the same already-generated sufficient statistics;
- frozen incumbent actor for every continuation action;
- one h40 path per replicate, compact checkpoint statistics at h10/h20/h40;
- one fixed replay audit per partition/context stratum, capped at 24 paths;
- append-only resumable task checkpoint, no full trajectory corpus;
- cluster unit is canonical root ancestry, never frames or replicates.

Before fitting, require zero stream collisions against all prior evaluation,
H0/H2, and human streams; exact task count; no missing partition/context cell;
deterministic replay of the fixed audit; source hashes unchanged; and terminal
bootstrap exactly zero.

## Frozen Optimization And Offline Metrics

If labels are later authorized, train exactly two final models from seed
`20260711`:

- Adam, learning rate `0.001`, weight decay `0.0001`;
- batch size `256`, exactly `200` epochs, no early stopping;
- train-partition standardization frozen once from train labels;
- identical batch order and updates for both modes;
- loss = standardized h40 Huber expected-return loss (`1.0`) + h40 return-bin
  cross-entropy (`1.0`) + each of four h40 binary cross-entropies (`0.25`);
- final epoch is the only candidate checkpoint; no checkpoint selection.

Primary offline metric is per-ancestry h40 expected-target MAE improvement:
`MAE_board_only - MAE_context`. Use 10,000 ancestry-cluster bootstrap draws.
The ancestry-holdout 95% interval must exclude zero positively. Whole-family
holdout improvement must be positive.

Additional frozen gates:

- context model survival/1536/3072/anchor Brier score may not regress more than
  `0.01` on either ordinary holdout;
- corresponding expected calibration error may not regress more than `0.02`;
- on the separate source-disjoint synthetic-context diagnostic, expected-target
  contrast Spearman correlation at least `0.25` and sign accuracy at least
  `65%`; opportunity (score/milestone) and risk (survival/anchor) directions
  must both pass independently;
- leave-one-fit-family-out primary improvement must stay nonnegative for every
  family and positive for a majority;
- no single ancestry, family, stage, or context bin may contribute more than
  `40%` of aggregate improvement;
- identity, save/load, schema, source hash, and incumbent nonmutation audits
  must still pass.

Any failed or ambiguous requirement kills/holds R1.5a before policy use. No
metric may be replaced after outcomes.

Required engineering tests:

- zero residual preserves arbitrary incumbent values exactly;
- equal parameter count and deterministic hidden initialization;
- board-only masking makes predictions invariant to context;
- context model distinguishes context only after nonzero context weights;
- save/load round trip preserves mode, metadata, parameters, and predictions;
- incompatible feature/target schemas fail clearly;
- frozen incumbent inputs are never mutated.

## Future Label Budget Estimate

Inventory only during this gate. Estimate, but do not launch, a frozen-incumbent
label plan with 16 split-stream replicates (A/B eight) and one h20 trajectory
supplying h10/h20 metrics. Report expected trajectories, actor decisions,
wall-clock time using measured H2 throughput, and compact storage.

## Stop/Go Rule

After inventory and tests, write an explicit `R15A_PREFLIGHT_STOP_GO.json`.

- `GO_FOR_REVIEW`: all readiness and engineering rules pass. This means the
  manifest is fit to review; labels and fitting remain held pending explicit
  authorization.
- `HOLD_DATA`: diversity/context coverage is insufficient.
- `HOLD_ENGINEERING`: identity, parity, schema, or save/load checks fail.

No other result is permitted. No policy-facing work follows automatically.

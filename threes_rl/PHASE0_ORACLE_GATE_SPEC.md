# Phase-0 Oracle Gate Specification

Status date: 2026-07-09

This spec freezes the next self-play-only research branch before any rollout
labels, search jobs, value fitting, or policy fitting are allowed. It is a
design and dry-run corpus-audit artifact only.

## Purpose

Test whether a new empirical oracle can select a better first action than the
current incumbent on source-diverse pre-first-`1536` states.

This is not a retry of the killed MCTS/UCT family. The previously tested
incumbent-leaf and target-directed UCT configurations remain killed.

## Sentinel Endpoint

Primary endpoint:

- `P(reach first non-starter 1536 within 40 moves)`.

Secondary metrics, reported on the same ancestry-clustered roots:

- `P(reach first non-starter 1536 within 10 moves)`;
- `P(reach first non-starter 1536 within 20 moves)`;
- h40 score distribution;
- h40 survival.

Do not require separate 20-root corpora for every milestone/horizon slice.
Phase 0 uses one sentinel corpus and reports h10/h20 only as secondary outcome
cuts.

## Corpus Requirements

Candidate roots must be:

- normal-start ancestry roots with reset/provenance invariants;
- `root_origin in {fresh, human}`;
- one state per ancestry after root-capping;
- within 40 moves of the first non-starter `1536` promotion or a matched
  failure/death window;
- sourced from both success and failure strata.

Minimum corpus before rollout execution:

- at least 20 independent ancestry roots;
- at least two behavior policy families;
- no single behavior policy family above 50% of selected roots;
- at least four success roots and four matched-failure roots.

These are launch requirements, not promotion evidence.

## Behavior Policy Family Identity

Policy family is behavioral, not a checkpoint-name count.

Multiple checkpoints, seed slices, replay folders, or aliases from the same
actor lineage count as one behavior family. For example, current
`ntuple_phaseblend*` runs and `current_incumbent` aliases are one
`phaseblend_incumbent_lineage` family. Cheap approximations, human imports,
corner2, expectimax baselines, TD students, and synthetic/frontier starts are
separate only when their behavior actually differs.

The dry-run audit records the exact family coalescing rules used for a given
report.

## Pilot And Evaluation Seed Separation

For each root:

1. Enumerate all legal first actions.
2. Use pilot common-random-number rollout seeds only to choose the oracle
   action.
3. Freeze that selected action before evaluation.
4. Discard pilot outcomes from all gate scoring.
5. Evaluate incumbent action versus frozen oracle action with independent,
   preregistered seed blocks.

Pilot seeds must never appear in evaluation blocks. Evaluation block IDs and
random stream IDs must be written before execution.

## Exogenous Randomness Coupling

Use common random numbers within `(root, eval_block, repeat)` when comparing
the incumbent and oracle first actions.

If action-dependent spawn locations make trajectories diverge, the stream
identity still stays paired by root/block/repeat, but the report must state
that the coupling is stream-level rather than state-identical after divergence.

## Phase-0 Model Boundary

Phase 0 uses empirical continuation outcomes only:

- milestone hit by h10/h20/h40;
- score at horizon or death;
- survival at h40;
- optional score quantiles.

Do not fit a learned distributional value model in phase 0. Distributional
value fitting is phase 1 work and is allowed only after this empirical oracle
proves action advantage.

## Numeric Gate Margins

Primary milestone lift:

- h40 first-`1536` hit-rate lift must be positive in every independent
  evaluation seed block;
- root-cluster bootstrap confidence interval lower bound must be greater than
  zero;
- pooled point estimate must be at least `+5 pp`.

Score non-inferiority:

- mean h40 score-delta point estimate must be at least `-500`;
- lower confidence bound must be at least `-1500`.

Survival non-inferiority:

- h40 survival-delta point estimate must be at least `-1.0 pp`;
- lower confidence bound must be at least `-2.5 pp`.

Margin rationale:

- The previous direct gates showed effects around `-1.04 pp` and `+0.78 pp`,
  with intervals touching zero. A new operator must clear a larger practical
  bar before it earns fitting compute.
- A +5 pp milestone lift is the minimum useful signal for reopening a costly
  search/value loop from a sparse frontier corpus.
- Score and survival margins prevent a teacher from buying milestone hits by
  causing materially earlier deaths or large score regressions.

## Concentration And Robustness

The old "30% of changed roots nonnegative" guard is retired.

A passing oracle must also satisfy:

- all leave-one-root-out milestone-lift point estimates remain positive;
- all leave-one-behavior-family-out point estimates remain positive when at
  least two behavior families are present;
- no behavior family contributes more than half of the positive lift;
- at least two behavior families contribute positive aggregate lift.

If lift is carried by one ancestry or one behavior family, kill the operator
before any fitting.

## Kill Rules

Kill the operator before fitting if any of these occur:

- pilot/evaluation seed leakage is found;
- either evaluation seed block is non-positive on h40 milestone lift;
- the root-cluster interval touches zero;
- pooled h40 milestone lift is below `+5 pp`;
- score or survival violates non-inferiority;
- gains concentrate in one ancestry or behavior family;
- action changes are too rare to matter.

## Allowed Next Step

Only the read-only dry-run corpus/power audit may run before this gate is
unfrozen:

```bash
.venv/bin/python -m threes_rl.phase0_oracle_corpus_audit \
  --out-dir threes_rl/runs/forensics/phase0_oracle_corpus_audit/phase0_prefirst1536_dryrun_YYYYMMDD
```

The retained-replay coverage inventory is also allowed as read-only spec work:

```bash
.venv/bin/python -m threes_rl.phase0_replay_coverage_inventory \
  --out-dir threes_rl/runs/forensics/phase0_replay_coverage_inventory/phase0_prefirst1536_retained_replay_inventory_YYYYMMDD
```

These audits may count existing records/replays and write JSON/HTML reports.
They must not run rollouts, labels, search, training, or normal-start
evaluation.

# O3 Option Training Execution Charter

Date: 2026-07-27

Status: frozen implementation of the training phase authorized by
`READY_O3_OPTION_TRAINING_INTEGRITY_RESEALED_V3`. This charter changes no
scientific choice in `O3_EVENT_CONDITIONED_DESIGNATED_PAIR_CHARTER.md`.

## 1. Immutable Inputs

The phase binds:

- the O3 course charter and 102,557-parameter schema;
- V3 integrity envelope file/payload SHA-256
  `5bb80bc02597ea934c02f8ebd07eaf0158623232f88ea0408532cdc0039e6696` /
  `622ebf6361527be7283fd51c7a7acff99aa8125b06c76dbc4ee8a801faf3904d`;
- selected-root file SHA-256
  `9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049`;
- P0 stream-manifest file/payload SHA-256
  `94e7b0dfe83e568b4e9686dd3ee44cc70739c0312349fe36a05bb6df80c77225` /
  `27e3200e88d31d4f38a921965b631f264aa43f0ef02cb380f41b0c04d8455d1b`;
- the exact simulator, replay restoration, lineage/feature/model, provenance,
  and service-audit source hashes recorded in the preflight lock.

The 96 train roots are ordered exactly as their rows appear in the sealed
selected manifest after filtering `role=train`. The 32 development and 192
untouched-mechanism rows are byte-hash-bound but their replay JSON content is
not opened during this phase.

## 2. Zero-Outcome Preflight

`prepare` runs before any learning stream is consumed. It:

- verifies all immutable identities and V3 READY;
- restores and validates exactly the selected frame for every train root,
  including source hash, state hash, canonical pair, legal actions,
  hard-start eligibility, and finite action features;
- hashes but does not parse development or untouched replay files;
- extracts exactly the 1,152 P0 learning rows and proves zero collision
  outside the immutable P0 reservation;
- verifies one process, nice at least 10, no competing heavy Threes process,
  more than 100 GiB free with 120 GiB target, healthy ports/advisor/dashboard,
  and protected top three;
- freezes config, root, source, task, stream, collision, and preflight
  artifacts.

The preflight decision is `READY_O3_OPTION_TRAINING_EXECUTION`,
`HOLD_O3_TRAINING_PREFLIGHT`, or `KILL_O3_TRAINING_INTEGRITY`. READY still
contains zero labels, rollouts, fits, or outcomes.

`open` revalidates the lock and writes one immutable zero-label
`O3_OPTION_TRAINING_OPENED.json`. `execute` rejects a missing or mismatched
marker.

## 3. Frozen Rollout Semantics

There are four rounds, three trajectories per train root per round, exactly
1,152 episodes. Task identity and stream derivation are the P0 rows:
`root_index*12 + round_index*3 + replicate`, with 109B/110B/111B/112B
logical/deck/slot/policy bases.

- Round 1 chooses uniformly among legal actions.
- Rounds 2–4 use epsilon `0.15,0.10,0.05`; exploration chooses uniformly,
  exploitation uses the exact frozen action ordering.
- Each episode starts from the exact selected root and canonical pair.
- The learned policy controls every option move until pair-specific success,
  frozen failure, or h40 censor.
- Only the chosen action receives a row. Relative event and h10/h20/h40
  geometry labels use the course-charter masking rules.
- Successor geometry is normalized exactly as Manhattan/6, Chebyshev/3,
  blockers/6, same-row, same-column, empties/16, legal-count/4, and exact
  two-descendant-lineage integrity.
- Final score, future game milestone, behavior/human action, and
  counterfactual action outcomes are absent.

Each episode is committed by an immutable compact metadata artifact that binds
its compressed array SHA. The array is atomically renamed first and metadata is
atomically written last. An array without metadata is therefore an uncommitted
orphan: same-marker resume deterministically regenerates that exact task from
the same immutable streams, compares every array name, dtype, shape, and value,
never overwrites the orphan, and writes metadata only on an exact match.
Metadata without its array is an integrity failure. A complete array/metadata
pair written before an interrupted attempt close is verified and closed without
regeneration.

Attempts are append-only. Every task has exactly one open and at most one close;
same-stream resume records may occur only between them. Unknown tasks,
duplicate opens/closes, post-close resumes, identity or stream drift, and a
round checkpoint preceding complete committed inputs are integrity failures.

## 4. Frozen Fit

PyTorch CPU, seed `2026072703`, AdamW learning rate `3e-4`, weight decay
`1e-4`, batch 128, gradient cap 1.0, and five cumulative-buffer epochs after
each round are fixed. Model and optimizer state continue across rounds.

Rows are stored in deterministic task/decision order. Each epoch uses one
deterministic permutation from seed
`2026072703 + 100*round_number + epoch_index`. Event loss is weighted cross
entropy. Each checkpoint geometry loss uses sigmoid outputs, SmoothL1 for
Manhattan/Chebyshev/blockers/empties/legal-count, BCE for same-row/
same-column/lineage-integrity, and coefficient `0.10/3`. Batch weights are
normalized within each valid head exactly once. No alternate ordering,
optimizer, checkpoint, seed, feature, loss, or calibration exists.

The inter-round fits are required collection-policy updates, not optional
checkpoint selection. The round-4 checkpoint is mandatory. The label-support
gate is evaluated after all 1,152 frozen episodes and the mandatory checkpoint
seal, before any development or untouched replay content opens.

## 5. Label-Support Gate

The frozen gate from the course charter is unchanged:

- at least 40 pair-specific successes overall;
- at least six successes at each target 48, 96, and 192;
- at least four families with at least three successes each;
- at least 40 failures and 40 true h40 censors;
- all finite arrays and at least two nonempty success-time bins.

A miss seals `HOLD_O3_LABEL_SUPPORT`. Immutable identity, task/stream
semantics, deterministic convergence, nonfinite model/label, checkpoint, or
serialization failure seals `KILL_O3_TRAINING_INTEGRITY`. A transient process,
service, disk, output-size, or active-runtime guard seals
`HOLD_O3_TRAINING_OPERATIONAL`, preserves partial work, and is not scientific
evidence. A pass seals `READY_O3_OPTION_DEVELOPMENT`; it authorizes only the
separately sealed 32-root paired option-development phase.

No partial support count, coefficient, action, label rate, or checkpoint
comparison is reported during execution.

## 6. Resources And Governance

One worker and one heavy process run at nice at least 10. Active runtime is
limited to 18 hours and incremental training output to 4 GiB. Work pauses
below 100 GiB free; 120 GiB remains the target. Services, dashboard record
263670, and protected top three 263670/261369/258561 must remain healthy.

The output namespace is
`threes_rl/runs/forensics/o3_option_training_v1`. Development, untouched
mechanism, normal-start capability, confirmation, incumbent, and dashboard
changes remain held regardless of this phase until their own gates pass.

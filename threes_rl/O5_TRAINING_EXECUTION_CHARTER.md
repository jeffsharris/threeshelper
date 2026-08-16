# O5 Domain-Safe Option Training Execution Charter

Status: frozen before any O5 learning stream, episode label, optimizer step, or
checkpoint. This charter governs only the O5 label-and-fit execution after the
sealed `READY_O5_FOUR_FAMILY_DOMAIN_SAFE_PREFLIGHT` boundary.

## 1. Immutable Scientific Inputs

- The authoritative source is the sealed O5 P0 namespace
  `threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1`.
- Its selected-root scientific manifest SHA is
  `05850e87eaa03010e06c27b548d04d22bf22768dfa94d62d0a2a1cba96d20612`.
- The allocation is fixed at 448 whole ancestries: 192 train, 64 development,
  and 192 untouched mechanism. The four families contribute 112 roots each.
- Only the 192 train replay bodies and their selected current support frames
  may be restored. Development and untouched source files are hash-verified
  only. Their replay bodies, geometry, labels, predictions, actions, and
  outcomes remain sealed.
- O3 selected roots, O3 option-training bodies, and all O4/O3 labels,
  checkpoints, actions, and outcomes remain forbidden. O4 is used only through
  the byte-bound domain-safe operator.
- The runner binds the exact P0 marker, result, selected roots, streams,
  collision audit, domain proof, policy audit, power table, source pool, source
  replay manifest, test evidence, and their canonical payload hashes. It also
  binds the simulator, provenance/service scanner, O4 operator, O4 power
  contract, and the current frozen policy dependency identities.

## 2. State And Feature Access

- Train states are restored with the O5 local whitelist only:
  `board`, `preview.kind`, `preview.candidates`, `tile_cycle.small_counts`,
  `small_pos`, `small_seen_total`, `span_small_pos`, `large_pending`,
  `move_count`, and `game_over`.
- `max_tile` is derived from the current board. Final score, frame score,
  payload max-tile fields, recorded move/action, future milestone, terminal
  outcome, and replay-level policy outcomes are never accessed.
- The selected frame, source replay SHA, whitelisted state SHA, designated
  pair, target, legal count, family, and ancestry must reproduce exactly.
- The model is exactly `O4DesignatedPairNet`: 102,557 parameters and schema SHA
  `60a83881d8e8275a4aa2d03df06815d65e5b247b16f36118009f42f2ce3098ba`.
- Every model input and every stored target is finite and in `[0,1]`. Event
  targets are five-way one-hot rows with an explicit mask. Successor targets
  are the O4 density-safe eight-value vectors at h10/h20/h40 with explicit
  masks. Only the chosen action receives a row; no behavior or human action is
  a target.

## 3. Frozen Learning Tasks

- Exactly 1,152 P0 rows with purpose `learning` are used, six per train root.
  No option-development, untouched-mechanism, normal-development, or
  confirmation row may be consumed.
- Round schedule per root:
  - round 1: trajectories 0 and 1, uniform legal action exploration;
  - round 2: trajectories 2 and 3, epsilon 0.15;
  - round 3: trajectory 4, epsilon 0.10;
  - round 4: trajectory 5, epsilon 0.05.
- Seed is `2026072804`. Logical/deck/slot/policy IDs come verbatim from the
  sealed P0 stream rows.
- All 1,152 trajectories are generated before any optimizer construction or
  optimizer step. For rounds 2-4, exploitation uses the deterministic,
  seed-initialized O5 model before training. This removes ambiguity in the
  optimizer barrier and makes label collection independent of support outcomes.
- Each trajectory is closed-loop for at most 40 decisions. The queried policy
  controls every move. Designated lineage, pair-specific merge success,
  third-party merge failure, anchor safety, air safety, legal transition,
  h10/h20/h40 censoring, and density-safe successor geometry are exactly the O4
  semantics.

## 4. Episode Durability And Resume

- One task is one atomic compressed episode artifact containing arrays and
  canonical metadata. A temporary file is fsynced and renamed once.
- The append-only attempt ledger records `opened`, optional
  `resumed_same_stream`, and exactly one `completed` row per task. A completed
  task can never be regenerated or double-closed.
- An opened task without an artifact may be deterministically regenerated from
  its exact streams. An artifact without a close is validated and only closed;
  it is not regenerated.
- Measured generation runtime is charged to the atomic runtime state before the
  episode artifact is committed.
- A terminal result forbids every resume or retry.

## 5. Pre-Fit Support Gate

The gate opens only after all 1,152 tasks and artifacts close. Before that
boundary, only phase/completeness/integrity/resource information may be
reported. The aggregate gate requires all:

- at least 40 pair-specific successes overall;
- at least 6 successes at each target 48, 96, and 192;
- at least 3 successes in each of the four families;
- at least 40 failures;
- at least 40 true h40 censors;
- finite, domain-valid arrays;
- at least two nonempty success-time bins among 1-10, 11-20, and 21-40.

A clean miss seals `HOLD_O5_TRAINING_DATA_SUPPORT`; it creates no optimizer or
checkpoint. A transient resource/service stop is also a non-scientific HOLD.
Identity, schema, domain, label, serialization, or numerical corruption seals
`KILL_O5_TRAINING_INTEGRITY`.

## 6. Frozen Fit

Only after the support PASS:

- initialize exactly one CPU model with seed `2026072804`;
- use AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size 128,
  gradient clip 1.0, and PyTorch 2.12.1 with one thread and deterministic
  algorithms;
- perform five epochs on cumulative round-1 data, then five on cumulative
  rounds 1-2, five on cumulative rounds 1-3, and five on all four rounds;
- use family/root/trajectory/valid-row balancing so each family has equal total
  weight and each root contributes equally within family;
- use event cross-entropy plus the frozen O4 successor loss
  (`0.10/3` per checkpoint);
- save exactly one mandatory round-4 checkpoint after all 20 epochs;
- verify finite parameters, exact schema/config identity, deterministic
  save/load equality, and checkpoint hash.

There is no checkpoint selection, sweep, restart selection, calibration,
alternate objective, sign flip, O3 reuse, or holdout access.

## 7. Orchestration And Gates

The immutable namespace is
`threes_rl/runs/forensics/o5_domain_safe_training_v1`.

1. `prepare` writes only frozen manifests, audits, a zero-label preflight lock,
   and preflight result.
2. `open` revalidates every binding, source hash, stream collision, process,
   service, storage, and zero-work condition; it writes only
   `O5_TRAINING_OPENED.json`.
3. `execute` requires that exact marker and is same-marker resumable until a
   terminal result exists.

One nice-at-least-10 CPU process is allowed. Active runtime is at most 18 hours,
incremental output is below 4 GiB, free disk stays above 100 GiB with a 120 GiB
target, and dashboard/recorder/protected-top-three services must remain healthy.

Terminal decisions are exactly:

- `READY_O5_TRAINED_CHECKPOINT`;
- `HOLD_O5_TRAINING_DATA_SUPPORT`;
- `KILL_O5_TRAINING_INTEGRITY`.

Every terminal remains non-promotable. Development, untouched mechanism,
normal-start evaluation, incumbent changes, dashboard changes, and promotion
remain held.

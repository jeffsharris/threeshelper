# O5 Adaptive Domain-Safe Training Execution Charter V2

Status: preparation-only draft. It becomes authoritative only after its source
and tests are frozen into separate immutable V2 test evidence and a zero-label
preflight lock. No O5 stream, label, optimizer step, or checkpoint existed when
this V2 charter was written.

The non-authoritative V1 draft files are immutable evidence at:

- `threes_rl/O5_TRAINING_EXECUTION_CHARTER.md`,
  SHA `9348f4b24930df6ec2e463bb6f74272c0bd18cadc8b7a7d9a544ad2b04f4952f`;
- `threes_rl/o5_training.py`,
  SHA `e9baa328f2091e8f51ed287f774b862314af08e828eb2fe032dda4409e06504d`;
- `tests/test_rl_o5_training.py`,
  SHA `cfe8a2b4e090d2b3e73b550adeeedb2a0058eab4f40d8e7eb4af28c99d815a9e`.

Those paths may never be edited, bound as authoritative, or executed.

## 1. Immutable O5 Inputs

The sole scientific source is the sealed
`READY_O5_FOUR_FAMILY_DOMAIN_SAFE_PREFLIGHT` namespace:
`threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1`.

- Selected scientific manifest:
  `05850e87eaa03010e06c27b548d04d22bf22768dfa94d62d0a2a1cba96d20612`.
- Allocation: 448 unique ancestries, roles 192 train / 64 development / 192
  untouched mechanism, four families exactly 112 each, targets 150/149/149.
- Learning reservation: exactly 1,152 still-unconsumed rows, six per train
  root, from the fresh 181B/182B/183B/184B namespace.
- Model schema:
  `60a83881d8e8275a4aa2d03df06815d65e5b247b16f36118009f42f2ce3098ba`;
  parameter count 102,557.

The V2 lock binds the exact P0 marker, result, selection, stream, collision,
domain, policy, power, source-pool, source-replay-manifest, and test-evidence
file and payload hashes. It also binds the O4 domain-safe operator and power
contract, simulator, service/provenance scanner, policy/checkpoint dependencies,
this charter, the V2 runner/tests/evidence, and the three V1 draft identities.

Only the 192 train replay bodies and their selected current support frames may
be restored. Development and untouched files are hash-only: their replay
bodies, geometry, labels, predictions, actions, and outcomes remain sealed.
O3 option-training episode/metadata/label/checkpoint bodies are forbidden.

## 2. Whitelisted State And Domain

Train state restoration accesses only:

- board;
- preview kind and candidates;
- tile-cycle small counts, small position, small seen total, span position, and
  large-pending;
- move count and game-over.

`max_tile` is derived from the current board. Final/frame score, payload
max-tile, recorded move/action, legal-action annotation, future milestone, and
terminal/policy outcome fields are never accessed.

The model is exactly `O4DesignatedPairNet`. Inputs use the O4 designated pair,
lineage, safety, and blocker-density representation. Every input and successor
target is finite and in `[0,1]`. Event targets are masked five-way one-hot
vectors in `[0,1]`. Labels describe only the policy's queried chosen action;
human and behavior actions are never labels.

## 3. Adaptive Closed-Loop Sequence

One CPU model and one AdamW optimizer continue across all rounds:

1. R1: collect two uniform-legal trajectories per root, then train five epochs
   on cumulative R1 rows.
2. R2: collect two trajectories per root with epsilon 0.15, exploiting the
   exact R1 model, then continue the same model and optimizer for five epochs
   on cumulative R1-R2 rows.
3. R3: collect one trajectory per root with epsilon 0.10, exploiting the exact
   R2 model, then continue for five epochs on cumulative R1-R3 rows.
4. R4: collect one trajectory per root with epsilon 0.05, exploiting the exact
   R3 model, then continue for five epochs on all R1-R4 rows.

The schedule is 192 roots x `(2,2,1,1)` = 1,152 episodes. Seed is
`2026072804`; AdamW learning rate is `3e-4`, weight decay `1e-4`, batch size
128, gradient clip 1.0, one CPU thread, deterministic PyTorch 2.12.1. Epoch
permutation seed is `2026072804 + 100*round_number + epoch_index`.

Each trajectory is a sustained closed-loop option for at most h40. The queried
policy controls every move. Pair-specific merge success, third-party merge or
lineage loss, terminal state, anchor/air violation, censoring, and density-safe
h10/h20/h40 successor geometry use the exact O4 semantics.

R1-R3 checkpoints are provisional collection-policy state only. R4 is also
provisional until the aggregate support gate passes. Each checkpoint stores
the continued model and optimizer states, schema/config identities, round
number, and `authoritative=false`.

## 4. Atomic Work And Resume

- Every task is one atomic compressed artifact containing arrays plus canonical
  metadata. A temporary file is fsynced and renamed once.
- The append-only ledger records one `opened`, optional explicit
  `resumed_same_stream`, and exactly one `completed` event per task.
- Runtime is charged before the task artifact is committed.
- A completed task is validated, never regenerated. An opened task with no
  artifact may be deterministically regenerated from the same checkpoint and
  streams. An artifact without its close is validated and closed without
  regeneration.
- A round checkpoint can exist only after every task in that round closes and
  exactly five cumulative epochs finish. Its predecessor hash and optimizer
  continuity are recorded. An existing valid checkpoint is loaded, never fit
  again.
- Any terminal result forbids resume or retry.

## 5. Aggregate Support And Authority

After all 1,152 labels and all four five-epoch fits complete, but before any
checkpoint may be opened downstream, aggregate support must satisfy all:

- at least 40 successes overall;
- at least 6 successes at each target 48, 96, and 192;
- at least 3 successes in each of the four families;
- at least 40 failures;
- at least 40 true h40 censors;
- finite/domain-valid arrays;
- at least two nonempty success-time bins among 1-10, 11-20, and 21-40.

Only aggregate counts are exposed and only after all tasks close. A clean miss
seals `HOLD_O5_TRAINING_DATA_SUPPORT` plus an immutable quarantine manifest
listing every round checkpoint as non-authoritative and unusable. A transient
resource/service stop also HOLDs without scientific interpretation. A PASS
creates a separate authority envelope naming only the exact R4 checkpoint as
the candidate and seals `READY_O5_TRAINED_CHECKPOINT`. R1-R3 never become
candidates.

Immutable identity, schema, domain, label, serialization, model, optimizer, or
nonfinite corruption seals `KILL_O5_TRAINING_INTEGRITY`.

## 6. Orchestration

V2 paths:

- charter: `threes_rl/O5_TRAINING_EXECUTION_CHARTER_V2.md`;
- runner: `threes_rl/o5_training_v2.py`;
- focused tests: `tests/test_rl_o5_training_v2.py`;
- future test evidence:
  `threes_rl/runs/forensics/o5_domain_safe_training_v2_test_evidence.json`;
- future execution namespace:
  `threes_rl/runs/forensics/o5_domain_safe_training_v2`.

Commands are distinct: `write-test-evidence`, `prepare`, `open`, and `execute`.
`prepare` is zero-label and writes only immutable manifests/audits/lock/result.
`open` writes only `O5_TRAINING_V2_OPENED.json`. `execute` requires that exact
marker and command.

One nice-at-least-10 process is allowed. Active runtime is at most 18 hours,
incremental output is below 4 GiB, free disk remains above 100 GiB with a
120 GiB target, and dashboard/recorder/protected-top-three services stay
healthy.

This execution can never open development or untouched content, run mechanism
or normal-start evaluation, change the incumbent/dashboard, or promote.

# J1c Training Execution Orchestration Repair Charter

Status: frozen before J1c implementation, tests, manifests, or readiness
evidence

## 1. Scope And Parent Locks

J1c is a separately named orchestration-only successor to the spent J1b open
attempt. J1b may never be retried, resumed, overwritten, or reinterpreted.
J1c preserves the complete scientific contract of J1, J1a, and J1b:

- a from-scratch 411,656-parameter joint policy/value model;
- the same initialization seed and initial model identity;
- true normal starts with `starter_tile=None`;
- 16,384 independent training roots in 64 rounds of 256;
- 16 synchronous environments in one process;
- dense score-delta reward with fixed, nonadaptive auxiliaries;
- transition-t GAE, root-equal clipped PPO, four epochs, minibatch 4096,
  continued AdamW state, and the exact learning-rate schedule;
- the same bounded training engine, 32-tick durability cadence, runtime,
  storage, ownership, reclaim, retirement, checkpoint, sanity, and terminal
  gates;
- one round-64 candidate only, with no sweep, restart, checkpoint selection,
  development, confirmation, or promotion in this course.

No scientific source, model, objective, reward, schedule, gate, workload, or
runtime/resource threshold may change. Any such drift is an integrity HOLD.

## 2. Spent Evidence

The complete J1/J1a/J1b source and readiness identities remain immutable.
J1c additionally binds:

- the spent J1b three-file execution namespace;
- external decision `HOLD_J1B_OPEN_SERIALIZATION_INTEGRITY`;
- external terminal file/payload SHA-256
  `2f9cdfacb04a064b67785ab9bb00cac7d3d46bd057912b40ac4c06db0a0ed122` /
  `1cf98c5676b23c6168be4feef4d3e3a4ffeb98fb90f028af515f1646eb5e2369`;
- external retention file/payload SHA-256
  `28738328f724a544ee92fc7992ef8f256f0886c2e138a234a863ec0fe55c5f67` /
  `f88941a61da8909bd852d180892ce6c22d84c8ef4749b2110f9f2d58db8dd37a`.

All J1b-declared training IDs are spent for future allocation despite zero
reservation or consumption.

## 3. Sole Repair

The only semantic repair is immutable JSON orchestration:

1. normalize prospective payloads recursively to JSON-native values before
   canonical payload hashing and before in-memory equality checks; and
2. compare the exact serialized bytes after reload.

Tuple-to-list normalization must not relax payload hashes, create-once
behavior, fsync, collision rejection, tamper rejection, or any later identity
check. A clean fresh process must use the real scientific-mode
`operational_audit` shape and prove that its tuple-valued dashboard top three
round-trips to a stable immutable marker.

## 4. Fresh Training Manifest

The compact authority audit fixes the next contiguous 16,384 IDs after the
entire J1 and J1b declared prefixes:

| role | start | end inclusive |
| --- | ---: | ---: |
| logical | 213000032768 | 213000049151 |
| deck | 214000032768 | 214000049151 |
| slot | 215000032768 | 215000049151 |
| candidate policy | 216000032768 | 216000049151 |

The manifest has exactly one arm and one unique whole ancestry per row.
Root IDs use the accepted marker-root formula and a new immutable J1c
root-commitment payload. Rows, root IDs, ancestries, and all stream roles must
be unique and disjoint from the complete compact J1/J1b interval and manifest
authority. No broad heterogeneous payload scan is permitted.

The outcome-free readiness namespace is:

`threes_rl/runs/forensics/j1c_training_execution_surface_readiness_v1`

The future execution root is:

`threes_rl/runs/forensics/j1c_execution_v1`

It must remain absent throughout this readiness course.

## 5. Production Dispatcher

The J1c public dispatcher exposes exactly:

- `seal-phase-lock`
- `open`
- `materialize`
- `execute`

It is training-only. Runtime configuration and the unchanged first parent
operational guard must pass before owner acquisition, reservation, consumption,
commit genesis, game generation, or optimizer work. Scientific execution may
reach only `execute_training_engine_bounded`; legacy/fixture engines and all
development, confirmation, outcome-inspection, and promotion commands are
unreachable.

Materialization must copy the exact readiness-sealed J1c rows without
regeneration, substitution, filtering, or count change.

## 6. Readiness Gates

Before a READY decision, J1c must seal:

- source and parent identity audit;
- exact spent J1 and J1b identities;
- compact stream authority and collision audit;
- the exact fresh 16,384-row manifest and root commitment;
- model/schema/parameter and scientific-contract parity;
- clean-process Torch inter-op/intra-op `1/1`, deterministic algorithms, and
  unchanged first-guard ordering;
- real operational-audit JSON-native marker roundtrip;
- miniature create-once phase-lock/open/materialize/bounded-execute chain and
  exact crash resume;
- source/readiness/manifest/tamper and collision rejection;
- central and 5,000-move runtime/storage projections unchanged except for the
  negligible J1c wrapper/readiness bytes;
- healthy services, protected top three, opaque human sessions, one-heavy-job
  status, nice at least 10, free disk above 100 GiB with 120 GiB target;
- exact zero phase locks, markers, materialized rows, owners, reservations,
  consumptions, commits, games, transitions, optimizer steps, checkpoints,
  outcomes, holdout reads, incumbent/dashboard changes, and promotion actions
  in the future J1c execution root.

Readiness files are immutable, create-once, self-hashed, and JSON reload stable.
The terminal decision is exactly one of:

- `READY_J1C_TRAINING_EXECUTION_SURFACE`
- `HOLD_J1C_TRAINING_EXECUTION_SURFACE`
- `KILL_J1C_TRAINING_EXECUTION_SURFACE_INTEGRITY`

READY authorizes only later research-lead review. It does not authorize a phase
lock, marker, materialization, stream use, or science.

## 7. Current Boundary

- `CONTINUE`: J1c orchestration-only implementation, tests, and readiness;
- `HOLD`: all J1b retry and all J1/J1c science;
- `KILL`: false for the J1 scientific hypothesis; historical kills unchanged;
- `PROMOTE`: false.

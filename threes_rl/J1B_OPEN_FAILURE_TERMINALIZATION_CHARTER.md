# J1b Open-Failure External Terminalization Charter

Status: frozen before external failure evidence

## 1. Scope

This charter authorizes only an external, zero-science terminalization of the
spent J1b training open attempt. It does not authorize retry, repair, open,
materialization, ownership, stream reservation or consumption, commit genesis,
game generation, optimizer work, checkpoint creation, development,
confirmation, promotion, or mutation of the incumbent or dashboard.

The original namespace is immutable:

`threes_rl/runs/forensics/j1b_execution_v1`

It must contain exactly these three files:

- `training/phase_lock.json`, SHA-256
  `ac12b9f21977a3adcd61ef5f0d8ba60b058306dcc05fdfed423d2ca77c17a0ce`
- `training/phase_lock_result.json`, SHA-256
  `6a2f63dc8875db394333ac901a919466a6a432083e29feba32ba8917f3ee9bcf`
- `training/execution_opened.json`, SHA-256
  `e99099b87aa6417b4200ee236ef2b770d1524d11b26a878e9f3bf0d749a54cff`

The marker canonical payload SHA-256 is
`c9e48e972a59f699627bfaa949854930672a8c45a6c671be591e175522a107e4`.
No byte in the original namespace may be changed.

## 2. Defect And Decision

The exact defect is orchestration-only. The live operational audit emits
`services.dashboard.top_three` as a Python tuple. The J1b immutable writer
serializes it to a JSON array, reloads it as a Python list, and then rejects the
valid, self-hashed file through raw object equality (`observed != body`).

The authoritative external decision is:

`HOLD_J1B_OPEN_SERIALIZATION_INTEGRITY`

This is not scientific evidence and does not kill the J1 hypothesis. The J1b
execution namespace is spent and may never be retried, resumed, overwritten,
or used for future stream allocation.

## 3. Zero-Work Contract

The terminal must attest exact zero for:

- materialized rows and manifests;
- owners and recovery records;
- stream reservations and stream consumptions;
- commit genesis and later commits;
- completed or active roots, games, and transitions;
- optimizer steps, round aggregates, and checkpoints;
- policy or score outcomes;
- development, confirmation, promotion, incumbent, dashboard, or human-session
  reads or changes.

The only scientific-surface artifact is the failed zero-work marker itself.

## 4. External Evidence

New evidence lives only in:

`threes_rl/runs/forensics/j1b_open_failure_terminal_v1`

The terminalizer must:

1. bind this charter, its implementation, focused tests, and immutable test
   evidence by exact file and canonical payload SHA-256;
2. verify the exact three-file original inventory and every expected identity;
3. independently verify the marker payload hash;
4. reproduce the tuple-to-list raw-equality defect without changing source;
5. prove every forbidden-work path is absent;
6. create immutable JSON with create-once semantics, canonical self hashes,
   JSON reload stability, file fsync, and parent-directory fsync;
7. fail closed on source drift, extra or missing original files, marker
   tampering, nonzero work, output collision, or changed evidence.

The new namespace contains exactly:

- `J1B_OPEN_FAILURE_TEST_EVIDENCE.json`
- `J1B_OPEN_FAILURE_TERMINAL.json`
- `J1B_OPEN_FAILURE_RETENTION.json`

The retention record protects the original three files and all new external
evidence. Governance may be appended only after terminal and retention both
seal and independently reverify.

## 5. Terminal Boundary

After sealing, status is:

- `CONTINUE`: J1c orchestration-only readiness construction;
- `HOLD`: all J1b retry and all J1/J1c science;
- `KILL`: false for J1; historical kills unchanged;
- `PROMOTE`: false.

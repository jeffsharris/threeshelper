# J2A1 V3 Distillation Recovery Readiness

Status: outcome-free recovery readiness only. No collector, label, optimizer,
fidelity, or policy-outcome work is authorized.

## 1. Immutable boundary

The authoritative V2 execution at
`threes_rl/runs/forensics/j2a1_distillation_fidelity_execution_v2`
is permanently spent and remains
`HOLD_J2A1_V2_DISTILLATION_OPERATIONAL`. V3 must not edit, overwrite,
resume, reinterpret, or add a file to that namespace.

The operational HOLD is not scientific evidence. It occurred after 3,048
complete teacher roots and before family support, optimization, mechanism
validation, student fidelity arms, checkpoint authority, PPO, development, or
confirmation. The V2 scientific contents remain sealed.

V3 binds the V2 source identities:

- charter
  `d9c5382d803c606c29415fc020fa7d63762dfcb053232d1ac904f21827d74dd4`;
- runner
  `044a67bf9b34b311787e3e7de246c4ce62a33f4f8ae47d211f6a76dd231a22f3`;
- tests
  `b211bfac0bb2e18c87dddcd72a0c8e7f1a0c3cbd76fee92572133aefa7abd95d`.

It binds all eight V2 readiness artifacts, the V2 authorization, phase lock,
marker, active manifest, reservation, consumption, genesis, ownership ledger,
attempt ledger, completion ledger, terminal evidence, terminal, retention,
and every retained teacher-root file by exact path, byte count, and SHA-256.
In particular:

- terminal file/payload
  `13dd5c3a8eeb79d03149da0fa99a19aee3e6a657109e7fe4104a149d5d02ca6b` /
  `c3ad1135034b33a6118d3239f88e94a760eaa9afe98a9ca589bd70e351ce91a3`;
- terminal evidence file/payload
  `6a855bb18ca73cfef3dc465a3885e88901a317bbb08ce7624f2fb726438fdc7c` /
  `304da0e20042485e5e913d65e99cce81c93b613ad98b68f491085c772f5eeb5d`;
- retention file/payload
  `93f3a5ac0e155b16af84fc06165cc4e23cbd4184b10b96cc77dc9870b1c315ac` /
  `d0f637b76345694ded4679afb1fe6740a55065a135263398db59a9c0df3cad74`;
- attempt ledger file
  `06ec8ff51b35f722175f698edb3fadd3ae0d98c4b4e0bf77b8c4787ea24f5ca8`;
- completion ledger file
  `7ca78f7090d6c3df1cbe3bb522a0cd12f7ba85e8c1559f138b6e840a31011acd`.

## 2. Read boundary

The readiness runner may read only:

1. immutable JSON governance, authority, marker, manifest, terminal, and
   retention artifacts;
2. the outcome-free attempt, ownership, and completion metadata ledgers;
3. teacher-root files through a streaming byte hash and byte-count operation
   only.

It must never deserialize or otherwise inspect a teacher-root body. It must
not access board state, feature family, action, label, score, progression,
teacher output, student output, metric, trajectory, checkpoint, optimizer, or
policy outcome. Completion and attempt metadata are rejected if they contain
any scientific-content key.

The readiness runner must not import a teacher policy or expose any command,
callable route, or import-time side effect that can launch a game, collector,
teacher query, label build, optimizer, fidelity arm, or outcome analysis.

## 3. Frozen recovery authority

The active V2 manifest is authoritative and contains exactly 14,336 unique
whole-root rows and 63,488 unique streams. The V2 completion ledger and
retention inventory must prove exactly:

- 3,048 unique completed roots and ancestries;
- 3,048 retained root blobs with unique paths and file hashes;
- 6,096 attempt records: 3,048 starts followed by 3,048 exact finishes;
- zero open, abandoned, duplicate, or hidden retry attempts;
- every completion binds the exact authoritative row, ancestry, stage, path,
  file hash, and content hash;
- every finished attempt binds the corresponding completion output identity.

V3 derives the recovery authority once as the canonical manifest-order set
difference. It must contain exactly 11,288 unfinished rows. Completed and
unfinished rows must be disjoint and concatenate, after canonical index
ordering, to the exact original V2 authority. No root, ancestry, row, stream,
role, or stage may be replaced, filtered, regenerated, rebalanced, or
reassigned.

The V2 reservation and consumption remain the sole stream-use authority. V3
must not reserve or consume a stream in readiness. A future recovery may reuse
only the exact unfinished rows under the existing consumption identity and
must never write a second reservation or consumption for those streams.

Stage B and every family/mechanism/fidelity read remain sealed until all
14,336 roots have exact completion metadata and retained byte identities.

## 4. Wall-clock cap repair

The sole operational repair is cap accounting. The 72-hour Stage A cap is
strictly top-level elapsed wall time, not the sum of eight collector worker
durations.

V2 attempt metadata must reproduce:

- aggregate worker seconds `259763.24813699722`, descriptive only;
- earliest start `1785246546.567381`;
- latest finish `1785279154.845043`;
- observed top-level wall span `32608.277662038803` seconds;
- observed completed roots `3048`.

The outcome-free readiness projection is fixed:

1. observed rate = completed roots / observed wall hours;
2. projected remaining wall = 11,288 / observed rate;
3. projected Stage A wall = observed wall + projected remaining wall;
4. conservative projected Stage A wall =
   observed wall + 1.25 * projected remaining wall.

Both projected totals must be below 72 hours. The future recovery runtime
ledger must charge authenticated live top-level owner segments and bounded
abandoned units only. It must not charge adjudication, reboot, or dead-process
downtime. Aggregate worker seconds, CPU seconds, and per-worker durations
remain descriptive resource accounting and can never trigger the 72-hour
wall-clock gate.

## 5. Future recovery ownership

The zero-work readiness namespace is
`threes_rl/runs/forensics/`
`j2a1_distillation_fidelity_recovery_readiness_v3`.
The separately authorized future execution namespace is
`threes_rl/runs/forensics/`
`j2a1_distillation_fidelity_recovery_v3` and must remain absent in this turn.

A future execution must use one top-level heavy job at nice 10 with exactly
eight fixed single-thread collectors and original shard ownership. Ownership
is create-once and append-only. A live, unrelated, wrong-marker,
wrong-authority, wrong-command, or malformed owner fails closed. A verifiably
dead same-marker owner may be reclaimed only through an immutable record that
binds the old and new owners, process-death evidence, recovery authority,
commit head, and zero concurrent writer.

Recovery resumes from the last authenticated boundary. Completed roots are
never rerun. An uncommitted root may be deterministically replayed in its
original shard and stream, with abandoned work charged once. No duplicate
completion, attempt closure, stream use, or scientific record is permitted.

## 6. Operational and retention gates

Readiness requires:

- one-heavy-job audit passes and launches no heavy job;
- nice capability is at least 10;
- free disk is greater than 100 GiB, with greater than 120 GiB the target;
- ports 8765 and 8770, advisor, dashboard, and protected top three are healthy;
- human sessions remain opaque and unread;
- V2 remains byte-for-byte exact;
- the future V3 execution namespace is absent;
- all collector, query, label, game, optimizer, checkpoint, family,
  mechanism, fidelity, PPO, development, confirmation, human-read,
  incumbent-change, dashboard-change, and promotion counters are zero.

No cleanup is authorized. V3 readiness artifacts are create-once, self-hashed,
JSON-stable, and covered by an immutable retention inventory.

## 7. Readiness surface and evidence

The public CLI exposes exactly:

1. `audit-zero-work`
2. `write-test-evidence`
3. `prepare`

The immutable readiness namespace contains:

1. `J2A1_V3_RECOVERY_TEST_EVIDENCE.json`;
2. `J2A1_V3_RECOVERY_INPUT_BINDINGS.json`;
3. `J2A1_V3_RECOVERY_V2_INTEGRITY_AUDIT.json`;
4. `J2A1_V3_RECOVERY_AUTHORITY.json`;
5. `J2A1_V3_RECOVERY_WALL_PROJECTION.json`;
6. `J2A1_V3_RECOVERY_SCHEMA.json`;
7. `J2A1_V3_RECOVERY_READINESS_LOCK.json`;
8. `J2A1_V3_RECOVERY_READINESS_RESULT.json`;
9. `J2A1_V3_RECOVERY_RETENTION.json`.

Focused tests must cover eight-collector wall versus aggregate time,
projection arithmetic, hash-only body handling, forbidden-key rejection,
attempt/completion chain tampering, exact set difference, duplicate/retry
rejection, create-once ownership, live/wrong-owner rejection, dead-owner
reclaim identity, crash/restart determinism, no scientific peeking, CLI
confinement, source/artifact tampering, zero work, and namespace absence.

## 8. Decisions

The only readiness decisions are:

- `READY_J2A1_V3_DISTILLATION_RECOVERY_PREFLIGHT`;
- `HOLD_J2A1_V3_DISTILLATION_RECOVERY_PREFLIGHT`;
- `KILL_J2A1_V3_DISTILLATION_RECOVERY_PREFLIGHT_INTEGRITY`.

READY authorizes only research-lead review of a separately frozen execution
surface and marker. It does not authorize collectors or any scientific work.

`CONTINUE` means V3 recovery-readiness review only. `HOLD` covers V3
execution, PPO, optimizer, family/mechanism/fidelity reads, alternate
branches, confirmation, promotion, and human training. Historical KILLs and
V1 execution reuse remain killed; V2 remains an immutable operational HOLD.
`PROMOTE=false`.

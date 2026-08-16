# J2A1 Distillation and Fidelity Execution Surface Charter

Status: frozen implementation-readiness contract only. This charter does not
authorize a phase lock, stream reservation or consumption, teacher query,
game, label, optimizer step, checkpoint, fidelity outcome, PPO work,
development, confirmation, incumbent change, dashboard change, or promotion.

## 1. Authority and scope

This surface is additive. It must not modify or reinterpret the accepted J2
readiness package, J2 A1 amendment, either exact-teacher pilot, or any J1/J2
historical terminal. The authoritative zero-work parent decision is
`READY_J2_A1_INCUMBENT_DISTILLATION_PREFLIGHT`.

The surface binds:

- J2 A1 charter
  `371e16088a4cbe3a7a3c5e6668fdd13424cad439f1a8f7e85b4dc7c120573e6a`.
- J2 A1 runner
  `80b4e0d88bbfa25b494d3c0f5783c1996ee057405e2277152107821265eb5a7d`.
- J2 A1 tests
  `bbfb98088431c0421fd82d5faa067736783ed262c5e841772eb9af09d1616db6`.
- J2 A1 result file/payload
  `0cfccc9fdc0cd7310200b31d1ec65890153541799a30a95808f92028b114e804`
  /
  `d37783ba9fed257f24c0d1888fd749c2c102e2ee399e5511d2839d201c0c0b52`.
- J2 A1 authority file/payload
  `0b421aa2da3e1f88d9b47b4d0dc26f7e696ca8e5ec6e8e43abd4052c6f7b2b94`
  /
  `4c120dbc5830385b59ac10cc0fff3505f3a39da197cd312d5f3bf2a5126f6c5f`.
- J2 A1 retention file/payload
  `a9932178a5c62758e35c668a5f44e8415f7c66bb1c8abcda10a303a65ff5adb6`
  /
  `d08b968c0f5205ebbb5c0ff6828798ab3f9ada69a64b68ef43ab67eada29da54`.
- Exact-teacher V2 pilot terminal file/payload
  `3ee2b204307bb96489ffd0fc3ff5c6c0cef488d6b5cfe986c4940f808354fcd9`
  /
  `8b98a0ec9892b615dd5072849b9fc655f7d043c7a257d90619ddbf35ad925089`.
- Exact-teacher V2 retention file/payload
  `6fe6563d6d676bf93455f0f3060ae3d851bf4b87b0c440216c52c703d0ff53a0`
  /
  `e8b9e6365a449689f0b485790ea9f3e4a27d1351562b0d387cd113c6db4702d1`.

The future execution root is exactly
`threes_rl/runs/forensics/j2a1_distillation_fidelity_execution_v1`.
It must remain absent during this readiness turn. A later, separate
research-lead authorization is required before any phase command.

## 2. Activated authority

Only two A1 stages may be activated by this future surface:

| Stage | Whole-root rows | Arms | Streams per row | Unique streams |
| --- | ---: | ---: | ---: | ---: |
| teacher_behavior_cloning | 8,192 | 8,192 teacher | 4 | 32,768 |
| distillation_validation | 6,144 | 6,144 teacher + 6,144 student | 5 | 30,720 |
| **Total** | **14,336** | **20,480** | | **63,488** |

The rows are copied exactly from the canonical A1 authority. BC uses prefixes
227B through 230B. Validation uses 231B through 235B. Root IDs, ancestry IDs,
row indices, stages, and streams may not be regenerated, substituted,
filtered, reordered, or resized. PPO, development, and confirmation rows are
hash-only protected and cannot be materialized or opened here.

Each validation row is one CRN pair authority. Its logical, deck, and slot
streams are shared by the teacher and student arms; its policy streams are
distinct. The row is reserved and consumed once as a pair authority. Arm-use
records bind the teacher and student uses without creating a second
reservation or a false stream collision.

## 3. Phase state machine

The only future public production commands are:

1. `seal-phase-lock`
2. `open`
3. `materialize`
4. `execute`

The first three create immutable, create-once artifacts. `execute` alone may
reach the bounded engines. No direct legacy or fixture engine is reachable in
scientific mode. A future authorization artifact, exact readiness result, and
all predecessor identities are reloaded and hash-verified at every command.

Execution has five irreversible internal stages:

- Stage A collects complete exact-teacher trajectories for every 8,192 BC
  root and every 6,144 validation teacher root.
- Stage B constructs the complete validation teacher-state feature inventory
  before any optimizer step and applies the support gate.
- Stage C runs exactly eight cumulative BC epochs if and only if Stage B
  passes.
- Stage D applies the frozen BC mechanism gate to untouched validation
  teacher states if and only if Stage C completes.
- Stage E runs all 6,144 sustained student full-policy arms and applies the
  closed-loop fidelity gate if and only if Stage D passes.

A clean Stage B, D, or E miss is a HOLD and quarantines every checkpoint.
Immutable identity, chronology, label, numerical, checkpoint, or retention
corruption is a KILL. An operational stop before a scientific disposition is
a HOLD. PASS authorizes only a separately reviewed PPO execution surface.

## 4. Teacher collection and labels

The teacher is the exact protected composite incumbent associated with the
263670 record. Its configuration, checkpoint, policy lock, and every source
on its load/act path are byte-bound and reverified before worker creation.
Human actions and human-session content are forbidden.

One top-level nice-10 job owns exactly eight single-thread collector
processes. Canonical row index `i` belongs to shard `i mod 8`; there is no work
stealing. Merge order is canonical row index. Each worker loads the same
teacher identity and proves legal deterministic actions. A complete root blob
contains the authoritative immutable row, normal-start simulator provenance,
all teacher-visited observations and legal masks, teacher actions, current
scores, dense score deltas, terminal score, feature-only board family, and
hash chain needed to authenticate the root. Every root is retained
unconditionally.

For decision `t`, the value target is exactly
`1e-5 * (final_score - current_score_t)`. The remaining dense score deltas
must sum exactly to `final_score - current_score_t`. Any illegal action,
nonfinite field, failed telescope, partial root, wrong stream, cross-shard
row, duplicate, omission, or chronology drift fails closed.

Teacher labels remain only in this authorized future scientific namespace.
No development or confirmation authority is accessible.

## 5. Validation inventory and mechanism gate

Every state from all 6,144 complete validation teacher roots contributes to
the natural inventory. No root or state is dropped because of family,
trajectory length, score, progression, or future outcome. The feature-only
families are exactly `low_air`, `low_constrained`, `mid_progression`, and
`upper_progression`.

Before the first optimizer step, require:

- at least 1,024 natural states in every family;
- at least 256 distinct roots in every family;
- natural maximum family share strictly below 0.70;
- a deterministic capped inventory with every family represented and maximum
  share strictly below 0.40.

The capped inventory is a deterministic metric reference only; it cannot
change complete-root retention or training data. Natural and capped family
counts/frequencies and trajectory-position quartile counts are mandatory.
Any shortfall seals `HOLD_J2A1_FAMILY_DATA_SUPPORT`.

After training, untouched validation teacher states must satisfy:

- root-equal legal-action accuracy at least 0.97 overall;
- root-equal legal-action accuracy at least 0.94 in each capped family;
- finite policy loss, value MSE, zero-predictor MSE, and family metrics;
- value MSE strictly below the zero-predictor MSE;
- exactly zero illegal labels or selected actions.

Any miss seals `HOLD_J2A1_BC_MECHANISM` and quarantines the checkpoint.

## 6. Model and optimizer

The model is the frozen no-auxiliary J2 actor-critic: 282 inputs, two
512-wide ReLU layers, a legal-masked four-action policy head, and one scalar
value head, exactly 410,117 parameters. Initialization seed is 2026072806.
There are no auxiliary heads or auxiliary losses.

Training uses root-equal cross entropy plus `0.5 * value_MSE`, Adam with
learning rate `3e-4` and epsilon `1e-5`, gradient clip `0.5`, CPU,
deterministic Torch intra/inter-op 1/1, exactly eight cumulative epochs, and
minibatch 4,096 with every final short minibatch retained. The deterministic
epoch/minibatch plan is frozen by the parent implementation. One final
epoch-8 checkpoint exists; no checkpoint selection, sweep, alternate seed,
restart after a clean miss, or warm start is permitted.

Model, optimizer, RNG, plan, cursor, immutable batch identity, and closed
step IDs are durably resumable. An uncommitted optimizer step restarts from
its authenticated pre-state; a committed step resumes after it. The
uninterrupted and resumed checkpoint bytes must be identical.

## 7. Closed-loop fidelity

Only after Stage D PASS may the student checkpoint control every move of all
6,144 validation student arms. The exact teacher controlled every move of the
paired control arms in Stage A. Both arms use the precommitted shared
logical/deck/slot streams and distinct policy streams. No one-action
continuation or first-action proxy is a gate.

No student/teacher score or progression result may be opened, summarized, or
passed to analysis until an immutable completeness seal proves all 6,144
pairs, exact row identities, both arm hashes, and zero duplicate/missing
roots.

The score estimand is the root-equal paired mean of:

`log1p(max(student_final - student_start, 0))`
minus
`log1p(max(teacher_final - teacher_start, 0))`.

Its confidence interval uses 4,096 global paired-root bootstrap replicates,
seed 2026072831, NumPy linear quantiles at 0.025 and 0.975. Require point
greater than `log(0.97)` and lower 95% greater than `log(0.90)`.

P1536 uses eight precommitted equal strata of 768 pairs. The common OR uses
the accepted Mantel-Haenszel implementation and applies 0.5
Haldane-Anscombe correction whenever its numerator or denominator is zero.
Its confidence interval uses 4,096 whole-root bootstrap replicates sampled
independently within each stratum with fixed stratum totals, seed
2026072832, and NumPy linear quantiles at 0.025 and 0.975. Require point OR
at least 0.90 and lower 95% greater than 0.50.

Family/stratum signs are descriptive. Maximum, P95, P99, latency, and
survival summaries are mandatory safeguards but are not adaptive or
conjunctive utility gates.

## 8. Durable execution and retention

All immutable JSON uses canonical JSON-native bytes, create-once creation,
file and parent-directory fsync, and exact post-write byte verification.
Ownership is append-only. A live or mismatched owner fails closed. A
verifiably dead same-marker, same-lock, same-manifest, same-command owner may
be reclaimed only through an immutable recovery record after authenticating
the latest boundary and proving zero concurrent writer.

Attempts, runtime charges, reservations, pair uses, and consumptions are
append-only and hash chained. Every attempt is charged before its output
write. Resume may deterministically replay only the uncommitted root or
optimizer unit in its original shard; committed work is never repeated.

Complete teacher roots and final pair results are immutable write-once blobs.
An in-progress teacher root remains process-local and is never published as a
partial label artifact. A crash may deterministically replay only that
uncommitted root in its original shard; the abandoned attempt receives its
frozen conservative runtime charge. The full BC tensor batch is recovery
material:

- the BC batch may be retired only after the epoch-8 checkpoint and metric
  seal authenticate the root blobs, batch identity, plan, and checkpoint;
- every retirement has a write-once intent/manifest with exact paths, hashes,
  counts, bytes, and predecessor seal;
- interrupted retirement is idempotently completed before later work;
- complete-root blobs, pair blobs, terminal summaries, checkpoint,
  attempt/runtime/ownership ledgers, and retirement manifests are retained.

Two rolling resume slots plus one bounded orphan are allowed. Append-only
journals and commit stores are indexed in memory after one full resume scan;
hot-path appends are O(1). Full audits occur at stage and terminal seals.

## 9. Resource contract

Scientific execution requires one top-level heavy job, `nice >= 10`, exactly
eight single-thread teacher workers, Torch intra/inter-op 1/1, deterministic
algorithms, healthy ports 8765/8770, healthy advisor/dashboard, unchanged top
three, opaque human sessions, free disk above 100 GiB with 120 GiB target,
active runtime below 72 hours, and retained/peak output below 24 GiB.

The preflight maximum-shape projection uses 512 moves/root and measured
production-shape bytes:

- 14,336 teacher roots x 512 x 1,519 bytes/root-transition;
- one ephemeral 8,192-root BC batch x 512 x 1,261 bytes/row;
- 6,144 final pair blobs x 24,576 bytes;
- a conservative allowance equal to eight maximum active root serialization
  envelopes x 512 x 1,526 bytes, although no partial root file is retained;
- 896 MiB for rolling slots/orphan, ledgers, manifests, analysis, model,
  optimizer, checkpoint, and terminal evidence.

This is 17,535,295,488 bytes before margin and 21,919,119,360 bytes after the
frozen 25% margin, leaving 3,850,684,416 bytes below 24 GiB. Actual namespace
bytes and file counts are incrementally accounted and fully reconciled at
every stage seal.

Teacher runtime admission uses the V2 observed steady-state p99
`0.1316514358320273` seconds, 14,336 x 512 calls, eight fixed workers, the
sealed optimizer fixture, and 25% margin: `41.98247721555496` hours. Student
fidelity advances at most 16 live roots in one deterministic network batch,
but its admission projection conservatively charges the measured batch-16
p99 once per root transition rather than assuming full batch occupancy.
Simulator time and the inherited administrative allowance are added
separately; total central projection remains below 72 hours. The 5,000-move
sensitivity is mandatory and descriptive, not an alternate workload or
conjunctive gate.

Resource/service/process guards run before ownership, before reservation,
before consumption, after every fixed collector block of at most eight roots,
after every optimizer minibatch, after every fixed fidelity block of at most
16 pairs, at every stage seal, and before terminal sealing. Operational
failure seals a HOLD without corrupting the marker.

## 10. Readiness package

This implementation turn may run only:

- source/hash/schema/dependency validation;
- synthetic and miniature production-path tests;
- py_compile and non-scientific regressions;
- maximum-shape fixture projections;
- read-only process, disk, service, advisor, dashboard, top-three, and
  human-opacity audits;
- `audit-zero-work`, `write-test-evidence`, and `prepare-readiness`.

The readiness terminal is exactly one of:

- `READY_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE`
- `HOLD_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE`
- `KILL_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_INTEGRITY`

READY authorizes only research-lead review of a later
`seal-phase-lock/open/materialize/execute` sequence. Every real-work counter
must remain zero and the future execution root must remain absent.

Explicit boundary:

- CONTINUE: source, tests, immutable zero-work readiness only.
- HOLD: all teacher queries, labels, games, reservations, consumption,
  training, fidelity outcomes, PPO, development, and confirmation.
- KILL: historical kills remain unchanged; no pilot or A1 rerun.
- PROMOTE: false.

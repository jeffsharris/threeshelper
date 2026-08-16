# J1d Metric Authentication Repair Charter

Status: frozen before J1d implementation, tests, manifests, or readiness
evidence

## 1. Scope And Permanent Locks

J1d is a separately named, outcome-free metric-authentication repair successor
to the spent J1c training execution. J1c remains permanently
`KILL_J1C_TRAINING_INTEGRITY`; its execution, checkpoint, metrics, roots,
transitions, streams, optimizer state, and episode bodies may never be retried,
resumed, reused, rehabilitated, or reinterpreted. Development and confirmation
remain unopened.

J1d preserves the complete scientific contract of J1, J1a, J1b, and J1c:

- the from-scratch 411,656-parameter joint policy/value model and identical
  initialization seed;
- true normal starts with `starter_tile=None`;
- 16,384 independent roots in 64 rounds of 256 with 16 synchronous
  environments in one process;
- dense score-delta reward, fixed nonadaptive auxiliaries, transition-t GAE,
  root-equal clipped PPO, four epochs, minibatch 4096, continued AdamW state,
  and the exact learning-rate schedule;
- the bounded training engine, 32-tick durability cadence, ownership/reclaim,
  runtime charging, retirement, storage, checkpoint, sanity, and terminal
  contracts;
- one round-64 candidate only, with no sweep, restart, checkpoint selection,
  development, confirmation, or promotion in this course.

No model, seed, optimizer, objective, reward, auxiliary, mask, schedule,
threshold, tolerance, workload, resource limit, or scientific decision rule
may change.

## 2. Authoritative Failure Attribution

The readiness package binds the immutable J1c terminal and retention
identities and records only the independently audited structural attribution:

- every one of the 64 rounds failed only `aggregates_recomputed_exact`;
- the failing subchecks were exactly `legal_entropy`, `auxiliary_brier`, and
  `auxiliary_prevalence_brier`;
- all root, per-root metric, transition, round metric, record, buffer, value-MSE,
  and recursive commit integrity checks passed;
- maximum absolute discrepancies were
  `2.62641797199592e-10`, `4.422892815880708e-10`, and
  `4.555544760864727e-10` respectively, while value-MSE discrepancies were at
  most `1.71e-13`;
- the writer reduced all transitions directly with root-equal weights, whereas
  the verifier averaged authenticated per-root reductions. Floating reduction
  order exceeded the unchanged absolute tolerance `1e-12`.

These facts are an orchestration diagnosis, not a scientific result. J1d may
not open J1c root records, transitions, episode bodies, checkpoint, optimizer,
metrics, score outcomes, or policy outputs.

## 3. Sole Repair

J1d defines one pure canonical derivation from the ordered authenticated
per-root metric rows. It produces exactly:

- ordered root log scores;
- legal entropy;
- value MSE and zero-value MSE;
- three auxiliary Brier values; and
- three auxiliary prevalence-Brier values.

Each scalar mean is computed from the per-root row values in manifest order
with one frozen implementation. Auxiliary prevalence Brier is computed only
from the corresponding canonical mean prevalence. The training writer must
publish these fields from that function after constructing the per-root rows;
the verifier must invoke the same function on the authenticated rows. The
per-root rows and their hash remain in the recursively authenticated round
checkpoint state.

J1d must not relax the `1e-12` check, suppress an integrity gate, discard
per-root evidence, or post-process a spent J1c artifact. A legacy direct global
weighted reduction that differs from the canonical per-root derivation is
rejected.

## 4. Fresh Training Manifest

All full declared J1, J1b, and J1c stream ranges are spent. J1d uses exactly
the next contiguous 16,384 rows:

| role | start | end inclusive |
| --- | ---: | ---: |
| logical | 213000049152 | 213000065535 |
| deck | 214000049152 | 214000065535 |
| slot | 215000049152 | 215000065535 |
| candidate policy | 216000049152 | 216000065535 |

The manifest has one arm and one unique whole ancestry per row. Root IDs use a
new immutable J1d marker-root commitment. Rows, root IDs, ancestries, and all
stream roles must be unique, exact, and disjoint from compact immutable
authorities for every spent prefix. Any collision or ambiguity is a fail-closed
HOLD. No heterogeneous historical payload scan is permitted.

The readiness namespace is:

`threes_rl/runs/forensics/j1d_metric_authentication_readiness_v1`

The future execution root is:

`threes_rl/runs/forensics/j1d_execution_v1`

The future execution root must remain absent throughout this readiness course.

## 5. Production Surface

The public dispatcher exposes exactly:

- `seal-phase-lock`
- `open`
- `materialize`
- `execute`

It is training-only, and none of these commands may be invoked in this course.
Future scientific execution may reach only the J1d bounded training path whose
round writer and sanity verifier share the canonical aggregate function.
Development, confirmation, promotion, legacy fixture engines, alternate seeds,
and outcome-inspection commands are unreachable.

## 6. Tests And Readiness Gates

Before READY, J1d must prove:

- immutable parent source/readiness identities and exact spent J1/J1b/J1c
  terminal and retention identities;
- the canonical function is shared by writer and verifier;
- unequal root lengths and adversarial floating values reproduce the old
  greater-than-`1e-12` reduction-order discrepancy, while J1d publishes and
  verifies exactly canonical aggregates;
- single-bit or field tampering, row/hash substitution, duplicate roots,
  changed order, legacy aggregates, collision, and identity drift fail closed;
- a clean miniature production path reaches a READY training-sanity terminal,
  and create-once, crash-resume, and immutable terminal behavior hold;
- an independent real-shape synthetic 64-round aggregate-authentication fixture
  passes without scientific games or outcomes;
- exact fresh manifest/range authority and zero collision;
- unchanged runtime/storage projection, deterministic Torch inter-op/intra-op
  `1/1`, deterministic algorithms, one-heavy-job status, nice at least 10,
  free disk above 100 GiB with 120 GiB target, healthy services/advisor/
  dashboard/top-three, and opaque human sessions;
- exact zero J1d phase locks, markers, materialized rows, owners, reservations,
  consumptions, commits, games, transitions, optimizer steps, checkpoints,
  outcomes, holdout reads, incumbent/dashboard changes, and promotion actions.

Readiness artifacts are immutable, create-once, self-hashed, and JSON-reload
stable. The terminal readiness decision is exactly one of:

- `READY_J1D_METRIC_AUTHENTICATION_PREFLIGHT`
- `HOLD_J1D_METRIC_AUTHENTICATION_PREFLIGHT`
- `KILL_J1D_METRIC_AUTHENTICATION_PREFLIGHT_INTEGRITY`

READY authorizes only research-lead review. It does not authorize any execution
phase or scientific work.

## 7. Current Boundary

- `CONTINUE`: J1d metric-authentication implementation and readiness only;
- `HOLD`: all J1d science, development, confirmation, and promotion;
- `KILL`: exact J1c execution plus historical kills; the J1 hypothesis remains
  live;
- `PROMOTE`: false.

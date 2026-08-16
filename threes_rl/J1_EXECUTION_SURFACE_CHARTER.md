# J1 Execution Surface Charter

Status: implementation and zero-work readiness only. No phase marker, stream
reservation or consumption, normal-start game, scientific label, optimizer
step, checkpoint, holdout read, policy outcome, human-session read, incumbent
change, dashboard change, or promotion is authorized by this charter.

## 1. Immutable Scientific Parent

The scientific contract is the conjunction of:

- `J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md`, SHA-256
  `26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2`;
- `J1_IMPLEMENTATION_READINESS_AUDIT.json`, file/payload SHA-256
  `f3e4e8029e159a1db7767164e1623d2e166b139be319d6077d61d7d107a44042` /
  `5b6b9a2383296f82b6547bbd46ddc892b486e4b89f4c325aa88f9c8b15944f99`;
- `J1_IMPLEMENTATION_PREFLIGHT_CHARTER.md`, SHA-256
  `7f87bc29c5764ccb290b25558f1cfe999083e9fddb089ea652cac9d0b92ab137`;
- `j1_joint_policy_value.py`, SHA-256
  `55d9e3206c2905509466c4962006e6cf3426f76647af6d2e60afe674b80c9bfe`;
- `tests/test_rl_j1_joint_policy_value.py`, SHA-256
  `e6b169f2d629021f96315380a3cf0ff6eece94a30e5027b1ace4d741499fbfa4`;
- the accepted J1 implementation test evidence, denylist, cost projection,
  preflight lock, and terminal `HOLD_J1_IMPLEMENTATION_PREFLIGHT`; and
- the accepted J1a amendment, runner, tests, test evidence, arithmetic, lock,
  and `READY_J1A_COST_POWER_AMENDMENT` result at the exact hashes frozen by
  the research lead.

No J1, J1a, O2, or `train_ppo.py` file is modified. The only scientific
change relative to J1 is the accepted J1a evaluation count amendment:
development is 896 pairs and confirmation is 4,480 pairs.

## 2. Frozen Learning Contract

Training uses the reviewed J1 primitives without semantic wrappers:

- one from-scratch 411,656-parameter, 282-512-512 policy/value/three-auxiliary
  network initialized with seed `2026072806`;
- CPU, one process, one PyTorch thread, deterministic algorithms;
- `starter_tile=None` at every reset;
- 16,384 independent complete natural-terminal roots;
- 64 rounds of 256 new roots, with 16 synchronous environments;
- policy sampling from the legal masked categorical distribution using each
  root's dedicated policy stream;
- dense reward `score_delta * 1e-5`, with exact episodic telescoping to final
  score minus start score and no auxiliary reward;
- transition-t GAE with gamma `1.0` and lambda `0.95`;
- root-equal row weights, total weight one per root;
- PPO clip `0.20`, value coefficient `0.50`, entropy coefficient `0.01`,
  fixed auxiliary coefficient `0.05`, Adam `3e-4`/epsilon `1e-5`, gradient
  clip `0.50`;
- four deterministic epochs per round, minibatch 4,096, final short minibatch
  retained, and the frozen linear round learning-rate schedule; and
- exactly one potentially authoritative round-64 checkpoint, with no sweep,
  restart, early stop, alternate seed, or checkpoint selection.

Every root is retained irrespective of score, progression, duration, maximum
tile, or policy behavior. A live root reaching 5,000 moves is an integrity
failure, never a natural completion.

The three chosen-action auxiliary labels are final maximum tile at least 1536,
final maximum tile at least 3072, and survival for at least 64 moves after the
queried decision. No behavior, incumbent, recorded, or human action is a
label.

## 3. Prospective Manifest Contract

The compact readiness manifest freezes every row through deterministic ranges:

| phase | logical | deck | slot | candidate policy | control policy | rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 213B | 214B | 215B | 216B | n/a | 16,384 |
| development | 217B | 218B | 219B | 220B | 221B | 896 |
| confirmation | 222B | 223B | 224B | 225B | 226B | 4,480 |

For base `B` and zero-based row `i`, the stream ID is `B+i`. Candidate and
control arms of an evaluation pair share logical/deck/slot exactly. Their
policy identities differ. Training has one arm. Block is `i mod 8`.

The compact manifest stores the canonical hash of every deterministically
expanded row, every phase row set, and every arm set. It proves each amended
range is an exact prefix of the immutable J1 denylist range, all ranges are
above the historical ceiling `212999999999`, and all phase/arm/stream sets are
disjoint except the intentional within-pair exogenous equality. This is
prospective arithmetic only: no stream is reserved or consumed.

The readiness manifest freezes a distinct immutable 256-bit phase nonce for
training, development, and confirmation. For each phase it preregisters an
immutable marker root-commitment payload/core containing the parent partition
name (`train`, `development`, or `confirmation`), nonce, compact row-manifest
hash, accepted readiness identities, and root-identity version. Let `M` be
that core's canonical `marker_payload_sha256`. For manifest row `r`, the root
and ancestry identity exactly matches the accepted parent derivation:

`SHA256(canonical_json({"marker_payload_sha256": M, "partition": partition, "row": row, "logical_stream_id": logical, "deck_stream_id": deck, "slot_stream_id": slot}))`.

Each later operational activation marker must embed the exact root-commitment
payload and hash. The activation marker has a separate hash. Activation time,
host, command, service/process/storage evidence, or other non-root operational
metadata cannot change any row, root, or ancestry identity. Changing the
preregistered marker root commitment necessarily changes those identities.

The readiness package commits all three phase root sets by exact canonical
hash without reserving a stream. Before the development marker can exist, the
development phase-lock operation must materialize and seal both the full
development and full confirmation root manifests, prove their mutual and
training disjointness, and bind their file/payload hashes. Confirmation
content and streams remain inactive and unread. A later confirmation marker
may only activate the already sealed confirmation manifest; it cannot
regenerate or replace it. Every claim that confirmation content is unread and
its streams unreserved/unconsumed must be backed by an explicit self-hashed
access-counter audit; it is never inferred or hardcoded by the runner.

## 4. Phase State Machine

The execution root is separate from this readiness namespace and has three
subdirectories: `training`, `development`, and `confirmation`.

Each phase has an immutable phase lock, immutable open marker, append-only
work journal or paired result rows, charged runtime clock, immutable terminal
result, and retention manifest. Each phase is one-shot. A crash may resume
only under the same marker, lock, command, manifest, writer identity contract,
and last valid atomic boundary.

The only production dispatcher verbs are `seal-phase-lock`, `open`,
`materialize`, and `execute`. Scientific training can reach only
`execute_training_engine_bounded`; scientific development and confirmation can
reach only `execute_paired_evaluation_engine_bounded`. The older verbose
engines require `execution_mode=miniature_fixture` and fail closed in
scientific mode. There is no promotion verb.

Every immutable phase artifact is created exactly once. Creation uses an
exclusive temporary file, file `fsync`, atomic hard-link creation at the final
path, and parent-directory `fsync`; it never replaces an existing destination.
A same-byte concurrent creation is verified and reported as already present.
A different-byte collision fails closed without changing the winning bytes.

Allowed future order:

1. training phase lock;
2. training open marker;
3. training execute/resume;
4. immutable training-sanity result;
5. development phase lock only after
   `READY_J1_TRAINING_SANITY`;
6. content-blind development and confirmation manifests sealed together;
7. development open and execute;
8. immutable development result;
9. confirmation phase lock only after
   `READY_J1_DEVELOPMENT_FULL_POLICY`;
10. confirmation marker activates the precommitted confirmation manifest;
11. confirmation execute;
12. immutable confirmation result.

No command can skip a predecessor, use the wrong phase marker, reopen a
terminal phase, or use a marker/lock from another directory.

Each scientific `execute` acquires or verifiably reclaims the append-only
writer ownership ledger before work. It then creates exact immutable stream
reservation and consumption records bound to the phase marker, phase lock,
manifest, root set, command, and ownership lineage. A recovery reuses an
existing consumption opener only when that opener is an ancestor in the
ownership ledger and the current recovery record links the old owner to the
new owner. It never rewrites stream evidence. Confirmation reservation and
consumption are impossible before the separately authorized confirmation
marker.

## 5. Atomic Resume and Ownership

A single-writer hash-chained ownership ledger is created atomically. Its
append-only owner records bind phase, marker, phase lock, host, PID, verified
process-start identity, runner hash, exact command, and predecessor commit
head.

An unrelated or live owner, wrong marker/phase/lock/command/runner, malformed
ledger, unverifiable process identity, or concurrent heavy writer fails
closed. A verifiably dead owner may be reclaimed only under the same
marker/phase/lock/command/runner and an intact last committed boundary. The
atomic ledger update appends, without deleting or overwriting history:

- a new owner record;
- an immutable recovery record binding old and new owner identities;
- process-death evidence;
- the predecessor commit-head, state, and latest journal hashes; and
- a zero-concurrent-writer process audit.

There is no silent owner deletion or theft.

Production initializes the immutable commit, rolling-resume, runtime-charge,
and output-accounting indexes with one complete validation scan per
process/resume. Hot-path append operations validate only the authenticated
head and predecessor in constant time. Full ordered chain materialization and
directory reconciliation occur at sparse round/block and terminal seals, not
at every vector tick, arm, or minibatch. The current contract hash on every
rolling record binds phase, marker file/payload, phase lock, manifest
file/payload, exact command, runner, and execution mode.

Collection and update use one transactional commit protocol. For each
deterministic unit:

1. load the immutable predecessor commit head and state;
2. compute the post-state in memory;
3. atomically write the hash-bound immutable post-state;
4. atomically write an immutable commit record binding unit identity,
   predecessor head/state, post-state, journal payload, and runner/marker;
5. atomically advance the commit head last.

A crash before head advancement resumes from the predecessor and
deterministically reproduces or byte-verifies the uncommitted post-state and
record. A crash after head advancement resumes after the unit. Unit identities
are present exactly once in the state and hash chain, so no collection batch
or optimizer update can be duplicated or skipped.

Training collection commits at a fixed cadence of at most 32 synchronous
vector ticks, stopping earlier only at round completion or a frozen resource
boundary. At most 32 vector ticks may be deterministically replayed after a
crash. Transition rows are appended once in immutable chunks of at most 1,024
rows. Rolling slots retain only active simulator/RNG state, a bounded unflushed
buffer, cursors, counts, and authenticated chunk/root references. Finalized
root blobs are immutable and written once; completion order may differ from
manifest order, and final PPO assembly sorts by authoritative manifest index.

Collection evidence contains actions, observations, legal masks, old log
probabilities, values, rewards, transition-t done flags, post-step simulator
states, root/task cursors, per-root policy RNGs, global Python/NumPy/Torch RNGs,
transition commitments, aggregate metrics, charged runtime, and resource
checks. It never serializes the full growing root prefix on every tick.

Before update, the complete 256-root buffer is atomically materialized into one
current-round immutable PPO batch. It is loaded once per uninterrupted process
segment. Rolling optimizer states carry only that batch identity, model, Adam
state, round, epoch, minibatch cursor, deterministic permutation identity,
closed-step identities, metrics, RNGs, and charged runtime. Every
round/epoch/minibatch identity is committed exactly once.

Transition chunks are ephemeral current-round recovery material. They may be
retired only after all 256 root blobs and the authenticated
collection/pre-update seal bind their exact paths, hashes, counts, and bytes.
The current-round PPO batch may be retired only after the authenticated round
checkpoint. Each retirement has a write-once manifest and is idempotently
recovered after crashes before the manifest, after the manifest, or during
listed-file deletion. Finalized root blobs remain authoritative evidence.

Paired evaluation writes each arm/pair result blob once. Its rolling state
contains only a pending arm, compact current-block references/counters, and the
authenticated prior block-seal head. It never serializes or reloads the full
completed-pair prefix per arm. Compact block and terminal seals authenticate
the complete ordered result set.

Round checkpoints are resume-only except the exact round-64 candidate.
Corrupt, missing, nonfinite, mismatched, or partially written state is
`KILL_J1_INTEGRITY`.

Training has a 72-active-hour and 24-GiB cap. Development has 24 hours and
8 GiB. Confirmation has 120 hours and 16 GiB. Each process is nice at least
10, one process/thread, and the sole heavy job. Work pauses below 100 GiB free
disk and targets above 120 GiB. Service or resource interruption is
`HOLD_J1_OPERATIONAL`; immutable identity, semantic, collision, duplicate,
partial-root, resume, or checkpoint corruption is `KILL_J1_INTEGRITY`.

Every attempt is charged before its scientific artifact is committed.
Recovery of an open attempt charges the frozen conservative unit ceiling,
independent of outage duration: 600 seconds per 32-tick collection block,
300 seconds per minibatch, and 900 seconds per paired arm unit. The output
accountant starts with one namespace scan, updates authenticated byte/file
counters at each atomic write/retirement, and performs full reconciliation at
round/block and terminal seals. It does not walk the directory at every unit.

The outcome-free central projection uses 512 moves/root or arm, the accepted
J1a compute evidence, measured bounded fixture bytes, fixed I/O costs, and a
1.25 safety multiplier:

- training: 8,388,608 transitions, 16,384 collection checkpoints, 8,192 PPO
  steps, 35,689 created files, 158,633 `fsync` operations, 3.3093 active hours,
  and 17.4822 GiB peak retained-plus-current-round storage after margin;
- development: 21.6779 active hours and 0.0614 GiB after margin;
- confirmation: 108.3894 active hours and 0.1992 GiB after margin.

Development and confirmation consume 90.3245% of their runtime caps, below the
J1a 91% admission ceiling. The mandatory 5,000-move sensitivity is diagnostic:
training projects 30.8247 active hours but 152.010 GiB, 179,305 files, and
1,380,009 `fsync` operations and therefore fails the storage/file/I/O caps.
Reaching 5,000 moves live remains an integrity failure; the sensitivity is not
used to relax the central readiness gate.

## 6. Train-Only Sanity

After all 16,384 roots and optimizer rows close, and before any development
content can open:

- every manifest root and every frozen optimizer-step identity occurs once;
- the root-equal mean `log1p(max(final_score-start_score,0))` over rounds
  61-64 exceeds the corresponding mean over rounds 1-4;
- final-round root-equal legal entropy is finite and at least 0.15 nats;
- final-round root-equal value MSE is below the zero-value MSE;
- at least two of three final-round auxiliary Brier scores beat their
  train-prevalence constant baselines; and
- the round-64 checkpoint, model, optimizer, schema, and parameter count
  reproduce exactly after save/load.

All pass gives `READY_J1_TRAINING_SANITY`. A clean learning miss gives
`HOLD_J1_LEARNING_SANITY`; there is no alternate seed or restart.

## 7. Full-Policy Development

Development is structurally inaccessible without a hash-bound
`READY_J1_TRAINING_SANITY` and the jointly sealed content-blind development
and confirmation manifests. The deterministic masked-argmax candidate
controls every move of its arm. The frozen depth-2 incumbent controls every
move of the control arm. Both use `starter_tile=None` and shared paired
exogenous streams.

The whole root is the cluster unit. The score estimand is paired
`log1p(max(final_score-start_score,0))`. P1536 uses the eight-block
Mantel-Haenszel common OR and deterministic whole-root bootstrap. P3072 risk
difference, raw score summaries, lower decile, survival/moves, illegal
actions, crashes, and latency are safeguards or descriptive.

Actual evaluation uses exactly 4,096 bootstrap replicates. The development
seed is `2026072817`; the confirmation seed is `2026072818`. The paired score
bootstrap resamples whole paired roots globally. The progression bootstrap
independently resamples whole paired roots within each of the eight fixed
`row_index mod 8` strata, preserving every stratum total, then calls the
accepted O2/J1a Mantel-Haenszel log-OR implementation. If its aggregate
numerator or denominator is nonpositive, that implementation adds 0.5 to
`a,b,c,d` in every stratum before recomputing. Confidence limits are the
0.025 and 0.975 `numpy.quantile` values with `method="linear"`. The 199
bootstraps in the parent proposal apply only to prospective power simulation,
not actual evaluation.

Development PASS requires:

- score point above 0, lower 95% bound above `log(0.95)`, and upper bound at
  least `log(1.07)`;
- P1536 common OR at least 1.0 and upper bound at least 1.50;
- P3072 risk difference at least -0.02;
- candidate lower-decile score and mean survival/moves each at least 95% of
  control;
- zero illegal actions and crashes; and
- candidate latency p95 at most 1.5 times incumbent p95 and candidate
  absolute p99 below 0.100 seconds.

A score upper bound below 0, or P1536 upper bound below 1.0 together with a
material safeguard harm, gives `KILL_J1_FULL_POLICY_UTILITY`. Any other miss is
`HOLD_J1_DEVELOPMENT_INCONCLUSIVE`. PASS alone gives
`READY_J1_DEVELOPMENT_FULL_POLICY` and permits only a separately authorized
confirmation lock.

Family and block signs and score maximum/P95/P99 are mandatory descriptive
reports, never conjunction gates.

## 8. Sealed Confirmation

Confirmation is structurally inaccessible without a hash-bound
`READY_J1_DEVELOPMENT_FULL_POLICY`. It uses 4,480 fresh paired roots and the
same full-policy, cluster, CRN, estimand, and safeguard semantics.

If the sealed control P1536 rate is below 0.02, the result is
`HOLD_J1_PROGRESSION_UNDERPOWERED`, never KILL.

PASS requires:

- score point at least `log(1.07)` and lower 95% bound above 0;
- P1536 common OR at least 1.50 and lower bound above 1.0;
- positive raw 10%-trimmed mean score direction;
- candidate median score, lower-decile score, and mean survival/moves each at
  least 95% of control;
- zero illegal actions and crashes; and
- the development latency bounds.

If a co-primary misses but its upper 95% bound includes its minimum meaningful
target, the result is `HOLD_J1_CONFIRMATION_INCONCLUSIVE`. If both targets are
excluded or a material safeguard harm is confirmed, the result is
`KILL_J1_FULL_POLICY_CAPABILITY`. PASS gives
`READY_J1_PROMOTION_REVIEW`; promotion remains a separate explicit action.
Maximum/P95/P99 remain descriptive and cannot veto PASS.

## 9. Readiness Package

The fresh readiness namespace is
`threes_rl/runs/forensics/j1_execution_surface_readiness_v1/` and may contain
only:

1. `J1_EXECUTION_TEST_EVIDENCE.json`;
2. `J1_EXECUTION_SCHEMA.json`;
3. `J1_PROSPECTIVE_MANIFEST.json`;
4. `J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json`;
5. `J1_EXECUTION_READINESS_LOCK.json`; and
6. `J1_EXECUTION_READINESS_RESULT.json`.

The readiness result may be `READY_J1_EXECUTION_SURFACE`,
`HOLD_J1_EXECUTION_SURFACE`, or `KILL_J1_EXECUTION_SURFACE_INTEGRITY`.
READY permits research-lead review and a separately authorized training phase
lock/open only. It does not create an execution marker.

Current state after an expected READY:

- `CONTINUE`: research-lead review, then separately authorized train lock/open;
- `HOLD`: all J1 scientific execution;
- `KILL`: historical kills only; J1 is not killed;
- `PROMOTE`: false.

# J1 Joint Policy/Value Implementation Preflight Charter

Frozen: 2026-07-27 before J1 runner test evidence, denylist, cost
projection, or readiness preflight.

Parent proposal SHA-256:
`26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2`.
Parent readiness-audit file/payload SHA-256:
`f3e4e8029e159a1db7767164e1623d2e166b139be319d6077d61d7d107a44042` /
`5b6b9a2383296f82b6547bbd46ddc892b486e4b89f4c325aa88f9c8b15944f99`.

## 1. Scope

This charter authorizes implementation and outcome-free readiness work only.
It does not authorize an execution marker, stream reservation or consumption,
normal-start game generation, score/policy outcome, scientific label,
optimizer step, checkpoint, development/confirmation access, human-session
read, incumbent change, dashboard change, or promotion.

`train_ppo.py` is a reviewed source pattern and remains unchanged. J1 uses a
new runner and tests. The runner CLI exposes only:

- `audit-zero-work`;
- `write-test-evidence`; and
- `prepare`.

There is no `open`, `execute`, `train`, `evaluate`, `reserve`, or `marker`
command.

## 2. Frozen Model and Schedule

The model is exactly:

```
282 -> Linear(512) -> ReLU -> Linear(512) -> ReLU
    -> policy logits(4)
    -> scalar value(1)
    -> auxiliary logits(3)
```

It has exactly 411,656 trainable parameters under PyTorch 2.12.1. CPU,
one thread, and deterministic algorithms are mandatory. Initialization seed is
2026072806. Every train, development, and confirmation reset uses
`starter_tile=None`.

The future schedule remains 64 rounds, 256 complete roots per round, four
root-balanced PPO epochs, minibatch 4,096, gamma 1.0, GAE lambda 0.95, PPO
clip 0.20, value coefficient 0.50, entropy coefficient 0.01, fixed auxiliary
coefficient 0.05, Adam learning rate 3e-4 and eps 1e-5, gradient clip 0.50,
and linear learning-rate decay. Auxiliaries are never rewards or adaptive
weights.

No J1 implementation may emit a scientific checkpoint until a later execution
charter and marker exist.

The canonical update is clipped PPO, not a proxy loss. Old log probabilities
are captured under the collecting policy with the exact legal mask. Advantages
are normalized once over the complete round buffer with the root-equal row
weights:

```
mean_w = sum(w*x)/sum(w)
adv_norm = (adv - mean_w) /
           sqrt(sum(w*(adv-mean_w)^2)/sum(w) + 1e-8)
```

Policy, half-squared value, entropy, and mean-three-label BCE terms all use
that same row-weight vector. A minibatch loss is its globally normalized
weighted sum multiplied by the number of minibatches in that epoch, so the
mean of frozen-model minibatch losses is the full root-equal objective. The
optimizer steps after every retained minibatch, including the final short
minibatch.

For round `r` in `1..64`, the update learning rate is
`3e-4 * (65-r)/64`; after round `r` it is `3e-4 * (64-r)/64`. Thus round 1
starts at 3e-4, round 64 still performs its frozen update at `3e-4/64`, and
the schedule is exactly zero after round 64.

For epoch `e` in `0..3`, the row permutation uses NumPy PCG64 seeded by the
first unsigned 64 bits of
`SHA256("J1-minibatch-v1|2026072806|r|e")`. This ordering is independent of
ambient RNG state and is saved by identity with the complete transition
buffer.

## 3. Correctness Contracts

### GAE

For transition `t`, `done_after_transition[t]` masks the value after that
transition. The recursion is:

```
delta[t] = reward[t] + gamma * V_next[t] *
           (1 - done_after_transition[t]) - value[t]
adv[t] = delta[t] + gamma * lambda *
         (1 - done_after_transition[t]) * adv[t+1]
```

where `V_next[t]` is `value[t+1]` inside a trajectory and the supplied
bootstrap value only for the final collected transition. A hand-computed
multi-step fixture must fail if `done[t+1]` is substituted.

### Complete roots and weights

A root enters a training buffer only after natural terminal completion.
Operational truncation is not completion. No transition may cross root or
partition identity. A root with `L` transitions receives total weight one and
each row receives `1/L`.

### Objective

For every complete simulator trajectory:

```
sum(score_delta) == score_board(final) - score_board(start)
```

The dense reward is this score delta times 1e-5. Auxiliary labels never enter
return. Crafted and fixed-random complete fixtures must prove the identity.

### Deterministic resume

Atomic resume state must include model, optimizer, complete vector simulator
states, root/task cursors, in-progress complete-root buffers, Python RNG,
NumPy RNG, Torch global RNG, policy Torch generator, and every simulator deck
and slot RNG. Tests interrupt at pre-action, post-step, mid-vector-game,
pre-update, and post-checkpoint boundaries. Resumed and uninterrupted fixture
runs must have identical actions, observations, simulator states, transition
hashes, model/optimizer tensors, RNG states, cursors, and final checkpoint
identity.

The pre-update and post-checkpoint fixtures execute the canonical clipped PPO
path on complete synthetic roots with transition-t GAE, unequal root lengths,
root-equal weighting, legal masks, fixed auxiliaries, deterministic four-epoch
minibatches, gradient clipping, continued Adam state, and the frozen round
learning rate. A simpler serialization fixture cannot substitute for this
proof.

Corrupt, partial, schema-mismatched, or nonfinite checkpoints fail closed.

## 4. Compact Protected Denylist

The denylist is byte-oriented and does not parse heterogeneous historical
payload schemas.

It binds a reviewed explicit set of authoritative protected root/corpus
manifests by path and exact file SHA-256. It also records a byte-hash inventory
of every current file below `threes_rl/runs` whose basename contains both
`stream` and `manifest`. The inventory reads bytes for hashing only. A changed,
missing, symlinked, duplicate, or newly appearing path fails closed.

All historical stream IDs through `212999999999` are denied by one explicit
closed interval. Prospective 213B-226B ranges are arithmetic only in this
turn. Their exact counts are 16,384 train rows, 1,024 development pairs, and
5,120 confirmation pairs. A future root ID is derived from marker payload,
partition, row, logical stream, deck stream, and slot stream. No root ID or
stream is created or reserved now.

The denylist must bind at least the frozen R2a, G2, human H0, O1, O3, O4, O5,
G3, S3, C2, K1, and R1.5a corpus/root identities listed in the runner. It must
never discover payload fields recursively or infer identities from arbitrary
JSON keys.

## 5. Outcome-Free Cost Projection

The exact course workload is 16,384 training roots plus 6,144 paired
evaluation roots, or 28,672 complete game arms. Projection uses:

- exact maximum-shape NumPy transition-buffer schemas without allocating the
  full maximum;
- fixed synthetic tensors for J1 forward/backward timing, with no optimizer
  step;
- fixed crafted simulator states for transition timing;
- the exact incumbent loaded only for fixed-state action-latency timing, with
  action identities discarded; and
- a fixed planning length of 512 decisions per complete arm plus a separately
  reported 5,000-move contract-maximum sensitivity.

Warmup calls are excluded. Report median, p90, p99, and maximum. Use p90 for
the central projection and multiply runtime and storage by 1.25. The
contract-maximum sensitivity must also include the 1.25 multiplier and report
whether each phase cap would pass. It is descriptive, not a conjunctive
readiness gate; execution still fails closed at the phase runtime cap.

The update timing fixture executes the canonical J1 clipped-PPO forward,
weighted policy/value/entropy/auxiliary reductions, backward pass, and
gradient clipping at batch 4,096. It performs no optimizer step. Projection
accounts for exactly four epochs over each projected complete-root buffer.

Frozen caps:

| Phase | Active hours | Incremental storage |
| --- | ---: | ---: |
| training | 72 | 24 GiB |
| development | 24 | 8 GiB |
| confirmation | 120 | 16 GiB |

Free disk must exceed 100 GiB and targets more than 120 GiB. One heavy job is
forbidden in this preflight; the fixtures are single-process, bounded, and
nice at least 10 when run for the immutable projection.

## 6. Preflight and Decision

Test evidence binds the charter, runner, tests, exact commands, pass counts,
and documented deselections. `prepare` requires that test evidence be the sole
existing output artifact. It then seals:

1. `J1_PROTECTED_ID_DENYLIST.json`;
2. `J1_RUNTIME_STORAGE_PROJECTION.json`;
3. `J1_IMPLEMENTATION_PREFLIGHT_LOCK.json`; and
4. `J1_IMPLEMENTATION_PREFLIGHT_RESULT.json`.

The frozen output namespace is
`threes_rl/runs/forensics/j1_implementation_preflight_v1`.
No marker is defined.

`READY_J1_IMPLEMENTATION_PREFLIGHT` requires all source identities, tests,
model/schema checks, GAE/objective/resume contracts, denylist and arithmetic
collision checks, cost caps, disk/process/service/advisor/dashboard/top-three
checks, and zero-work counters to pass.

A mutable operational fault is `HOLD_J1_IMPLEMENTATION_PREFLIGHT`. An
immutable source, schema, denylist, checkpoint, or semantic mismatch is
`KILL_J1_IMPLEMENTATION_INTEGRITY`. READY permits research-lead review only,
not execution.

Current state:

- `CONTINUE`: J1 implementation readiness work under this charter.
- `HOLD`: all J1 execution and science.
- `KILL`: historical kills, including further O6 continuation, remain.
- `PROMOTE`: false.

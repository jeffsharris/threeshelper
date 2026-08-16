# J1 Fresh Normal-Start Joint Policy/Value Self-Play Proposal

Frozen: 2026-07-27, before J1 implementation, stream reservation, game
generation, training, or policy outcomes.

This proposal is the authoritative course design for J1. Its file identity is
bound by `J1_IMPLEMENTATION_READINESS_AUDIT.json`.

## 1. Course Boundary

O6 remains authoritatively `HOLD_O6_DATA_PREFLIGHT`. Its one-shot is spent,
and further O6 continuation or retry is permanently killed as a research
course. The HOLD is an operational data-preflight stop, not evidence about
competing-risks learning or policy utility. O5 checkpoints remain quarantined
and unusable. O3, O4, C1, C2, K1, G3, G4, S3, the incumbent, and all historical
locks retain their recorded status and bytes.

J1 asks one new question:

> Can a machine-only learner trained from scratch on fresh complete
> normal-start games improve score and progression when its policy controls
> every move?

J1 is not a first-action intervention, fixed-continuation labeler, designated-
pair option, exact depth-3 search, human imitation system, or reuse of any
quarantined checkpoint or episode body.

## 2. Exactly Two Implementation Choices

The choice is made from code and installed-runtime facts before J1 outcomes.

### Choice A: selected joint policy/value PPO

Use the existing `train_ppo.py` architecture pattern: a legal-action-masked
categorical policy and scalar value head over the existing 282-value full
board, preview, and deck-cycle observation. The existing two-layer 512-unit
`ActorCritic` has 410,117 parameters under PyTorch 2.12.1.

The current training implementation is explicitly **not implementation-ready**
for J1:

1. its nonfinal GAE recursion uses `done_buf[t+1]` where the terminal mask for
   transition `t` appears required;
2. checkpoint resume restores model and optimizer only, not vector
   environments or Python/NumPy/Torch/simulator RNG state; and
3. its default start uses a 1536 starter rather than a true normal start.

J1 may reuse reviewed architecture, masking, and PPO primitives only after a
new J1 runner passes the focused correctness and resume gates in Section 8.

J1 adds one three-logit machine-derived auxiliary head. The frozen model is:

```
282 -> Linear(512) -> ReLU -> Linear(512) -> ReLU
    -> policy logits(4)
    -> scalar value(1)
    -> auxiliary logits(3)
```

It has exactly 411,656 trainable parameters. This is a single model, not an
ensemble.

### Choice B: rejected n-tuple TD value actor

The existing `train_td.py` / `ntuple.py` path can learn a board afterstate
value with n-step targets, temporal coherence, deterministic save/load, and
an induced search policy. It is operationally mature and CPU-native, but it
has no direct policy head and its principal value basis omits the explicit
preview/deck context present in the 282-value observation. Acting also remains
coupled to search.

### Frozen choice

Choice A is selected. It directly optimizes a policy under sustained
self-play, represents the full observable context, and jointly learns action
and return. It is materially stronger than spent R1b because R1b's residual
never controlled trajectory generation: every R1b move came from the frozen
depth-2 incumbent. J1's current policy controls every training and treatment
move. Choice B is not a fallback after outcomes; changing to it is a new
course decision.

## 3. Model, Objective, and Labels

- Initialize every weight from scratch using PyTorch 2.12.1 default
  `nn.Linear` initialization after `torch.manual_seed(2026072806)`.
- Device is CPU, one process, one PyTorch thread, deterministic algorithms.
- Action order and equal-logit tie order are simulator order
  `up, down, left, right`.
- Training actions are sampled from the masked categorical policy with a
  dedicated policy RNG. Evaluation actions are deterministic masked argmax.
- Every game uses `starter_tile=None`.
- The per-move reward is `score_delta * 1e-5`. For a complete trajectory
  `s_0,...,s_L`, the simulator definition gives
  `sum_t score_delta_t = score_board(s_L) - score_board(s_0)` exactly.
  The root score is exogenous and action-independent, so maximizing this
  dense return has exactly the same policy ordering as maximizing final game
  score. No shaped reward enters return.
- `gamma=1.0`, `GAE lambda=0.95`, PPO clip `0.20`, value coefficient `0.50`,
  entropy coefficient `0.01`, auxiliary coefficient `0.05`, maximum gradient
  norm `0.50`, and numerical epsilon `1e-8`.
- The three auxiliary labels for a decision at move `t` are machine-derived:
  final maximum tile at least 1536, final maximum tile at least 3072, and
  natural survival for at least 64 additional moves. Binary cross-entropy is
  averaged over the three logits. Their coefficient is fixed at 0.05 and
  cannot be adapted from support, loss, or outcome values.
- No recorded behavior action, incumbent action, human choice, human session,
  source policy, final action sequence, or protected episode is an input or
  label.
- A root of length `L` contributes total loss weight one; each of its
  transitions receives weight `1/L`. Root weights are equal.

All observations, returns, advantages, probabilities, losses, gradients, and
weights must be finite. Legal masking must make an illegal sampled or
evaluation action impossible.

## 4. Frozen Training Schedule

Training uses exactly 16,384 independent complete normal-start ancestries:

- 64 rounds in fixed order;
- 256 new roots per round;
- 16 synchronous environments inside one process;
- all 256 games in a round finish naturally before its update;
- four PPO epochs over that round's complete root-balanced transition buffer;
- minibatch size 4,096, with the final short minibatch retained;
- Adam, initial learning rate `3e-4`, `eps=1e-5`;
- learning rate decreases linearly by round to zero at the end of round 64;
- `starter_tile=None` and operational move cap 5,000.

Hitting 5,000 moves without natural termination is an integrity failure, not
a completed root. Every completed root is retained regardless of score,
milestone, maximum tile, duration, or policy behavior. No root is repeated.

Round checkpoints are resume state only and are explicitly non-candidates.
Model, optimizer, Torch RNG, NumPy RNG, stream cursor, completed-root
identities, and transition-buffer hash must resume exactly. Only the
mandatory round-64 checkpoint can become the J1 candidate. There is no
checkpoint selection, restart, sweep, early stopping, or alternate seed.

The training phase has a 72-active-hour and 24-GiB hard cap. It pauses below
100 GiB free disk and targets more than 120 GiB. Compact summaries and
resume state are retained; full per-transition buffers are deleted only after
their immutable aggregate/hash and next resume checkpoint are sealed under a
reviewed retention manifest.

Development has a 24-active-hour and 8-GiB hard cap. Confirmation has a
120-active-hour and 16-GiB hard cap. Across the course there are exactly
16,384 training roots and 6,144 paired evaluation roots, or 28,672 complete
game arms including both evaluation policies.

## 5. Prospective Identity and Stream Contract

No J1 stream is consumed in this proposal turn. The exact prospective bases
are:

| Partition | Logical | Deck | Slot | Candidate policy | Control policy | Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 213B | 214B | 215B | 216B | n/a | 16,384 |
| development | 217B | 218B | 219B | 220B | 221B | 1,024 pairs |
| confirmation | 222B | 223B | 224B | 225B | 226B | 5,120 pairs |

For a base `B` and zero-based row `i`, the stream ID is `B+i`. Candidate and
control arms share logical, deck, and slot IDs exactly within a pair. Their
policy identities differ; the deterministic incumbent consumes zero policy
RNG draws. Training has one arm. The four partitions are arithmetically
disjoint.

Before generation, a J1 implementation must seal:

1. a compact immutable denylist binding all protected root manifests and
   historical stream intervals recorded by governance;
2. the spent/reserved historical ceiling through 212B, including O6;
3. exact J1 row manifests and the set
   `SHA256(marker_payload | partition | row | logical | deck | slot)` of
   prospective root IDs; and
4. zero intersection between J1 IDs and the denylist.

The denylist is assembled from authoritative manifest identities and explicit
namespace intervals. It must not crawl or parse a schema-heterogeneous global
payload collection. Any unclassified historical stream at or above 213B,
unknown protected root identity, collision, or changed bound manifest fails
closed before a marker.

Train, development, and confirmation ancestries never cross partitions.
Development and confirmation manifests are sealed before any development
outcome is opened.

## 6. Full-Policy Evaluation

The candidate controls every legal move in its arm using deterministic masked
argmax. The frozen depth-2 incumbent controls every move in the control arm.
Both start normally and use paired shared exogenous logical/deck/slot streams.
There is no one-action treatment or incumbent continuation in the candidate
arm.

Development uses 1,024 fresh paired roots. Confirmation uses 5,120 separately
sealed fresh paired roots. The whole root is the cluster unit. Eight stream
blocks, assigned by row index modulo eight, and any training-round provenance
are descriptive only, never conjunction gates.

The primary score estimand is the root-paired difference in
`log1p(max(final_score - start_score,0))`. Because paired arms have the exact
same normal-start state, the start score cancels from the paired contrast.
Raw paired final-score-minus-start mean, trimmed mean, median, lower decile,
P90/P95/P99, and maximum are mandatory reports. The meaningful score target
is a 7% geometric lift
(`log(1.07)=0.06765864847381481`); the development noninferiority floor is
-5% (`log(0.95)=-0.05129329438755058`).

The progression co-primary is P1536, analyzed by an eight-stratum
Mantel-Haenszel common odds ratio with whole-root bootstrap. P3072, P6144,
survival/moves, illegal actions, crashes, and per-decision latency are
safeguards or descriptive endpoints. The historical starter-1536 P3072
calibration is not reused as if it described this starter-free endpoint.

## 7. Power and MDE

No starter-free policy outcome is opened for this design. Score power therefore
uses the prospective conservative paired log-score SD `1.25`, deliberately
above the spent starter-1536 sensitivity `1.1804313028078002`.

Progression power uses eight equal stream strata, 768 simulated datasets per
cell, 199 whole-root bootstraps per dataset, control base-rate sensitivity
`{0.02,0.04,0.08,0.15}`, paired coupling `{0.00,0.05,0.10}`, OR grid
`{1.25,1.50,1.75,2.00,2.50,3.00}`, point gate OR at least 1.25, and 95%
root-bootstrap lower bound above 1.0. The reported power is the minimum over
base rate and coupling. This is a prospective sensitivity envelope, not an
observed starter-free baseline claim.

At development N=1,024:

- 80%-power score MDE is 11.564970%;
- power for the 7% score target is 0.409861;
- worst-case P1536 power at OR 1.50 is 0.300781;
- the first 80%-power P1536 grid point is OR 2.50, with power 0.945312.

At confirmation N=5,120:

- 80%-power score MDE is 5.015910%;
- power for the 7% score target is 0.972129;
- worst-case P1536 power at OR 1.50 is 0.885417;
- the first 80%-power P1536 grid point is OR 1.50.

Thus development is a permissive utility screen and cannot kill a target-sized
null merely for nonsignificance. Confirmation is powered for the declared
score and progression effects provided the control P1536 rate is at least 2%.
If the sealed control rate is below 2%, P1536 evidence is
`HOLD_J1_PROGRESSION_UNDERPOWERED`, never a utility KILL.

## 8. Frozen Gates

### Engineering integrity

Before scientific work, require exact source/dependency hashes, 411,656
parameters, deterministic split-stream reset, observation and legal-mask
round trips, exact save/load/resume, root/stream disjointness, zero protected
artifact access, one process at nice at least 10, no competing heavy job,
healthy ports 8765/8770 and advisor, protected top three
`263670/261369/258561`, and disk/resource caps. An immutable marker precedes
each phase.

The new J1 runner must pass:

- a hand-computed trajectory fixture covering terminal and nonterminal steps
  and proving GAE uses the terminal status produced by transition `t`, not
  `done[t+1]`;
- exact telescoping tests proving dense score deltas equal
  `final_score-start_score` for crafted and random complete games;
- `starter_tile is None` assertions at every train/development/confirmation
  reset;
- interrupted-run tests at pre-action, post-step, mid-vector-game, pre-update,
  and post-checkpoint boundaries; and
- bit-identical resumed actions, observations, simulator states, completed
  roots, model/optimizer tensors, Python/NumPy/Torch RNG states, and final
  checkpoint versus uninterrupted execution; and
- an outcome-free runtime/storage projection using synthetic maximum-shape
  transition buffers and fixed simulator/network fixtures. It must report
  bytes per transition/root/checkpoint, peak and retained bytes, fixture
  throughput distributions, a 25% safety margin, and projected train,
  development, and confirmation costs. Every phase must fit its frozen cap,
  disk must remain above 100 GiB, and no game score or policy outcome may be
  generated to make the estimate.

Until those tests and a separate zero-work preflight pass, `train_ppo.py` is
an implementation reference only and J1 execution is not READY.

Immutable identity, illegal-action, nonfinite, partition, collision, resume,
or checkpoint corruption is `KILL_J1_INTEGRITY`. A transient service, disk,
or process-ownership fault is `HOLD_J1_OPERATIONAL`.

### Train-only learning sanity

After all 16,384 roots and before development opens:

- every root and optimizer step is present exactly once;
- the final four-round root-equal mean log score exceeds the first four-round
  mean;
- final legal-action entropy is finite and at least 0.15 nats;
- final-round value mean squared error is below the zero-value baseline;
- at least two of three auxiliary Brier scores beat their train-only constant
  prevalence baselines; and
- the round-64 checkpoint reproduces exactly after save/load.

A clean miss is `HOLD_J1_LEARNING_SANITY`; it does not authorize a restart or
alternate schedule. Integrity failures follow the engineering decision above.

### Development full-policy utility

PASS requires all of:

- score point estimate above zero, lower 95% bound above `log(0.95)`, and upper
  bound reaching `log(1.07)`;
- P1536 common-OR point at least 1.0 and upper bound reaching 1.50;
- P3072 risk difference at least -2 percentage points;
- lower-decile score and survival/moves each no worse than -5%;
- zero illegal actions/crashes; and
- candidate decision-latency p95 no more than 1.5 times incumbent p95 and
  absolute p99 below 100 ms.

A score upper bound below zero, or P1536 upper bound below 1.0 accompanied by
material safeguard harm, is `KILL_J1_FULL_POLICY_UTILITY`. Any other miss is
`HOLD_J1_DEVELOPMENT_INCONCLUSIVE`. Only PASS opens confirmation.

### Sealed confirmation

Confirmation PASS requires:

- score point at least `log(1.07)` and 95% lower bound above zero;
- P1536 common-OR point at least 1.50 and lower bound above 1.0;
- positive raw trimmed-mean score direction, no median/lower-decile/survival
  material harm, zero illegal actions/crashes, and the development runtime
  bounds.

If a co-primary misses but its 95% upper bound still includes its minimum
meaningful target, seal `HOLD_J1_CONFIRMATION_INCONCLUSIVE`. If both
minimum meaningful targets are excluded or a material safeguard harm is
confirmed, seal `KILL_J1_FULL_POLICY_CAPABILITY`. A complete PASS seals
`READY_J1_PROMOTION_REVIEW`; dashboard/incumbent mutation still requires a
separate explicit promotion action. The existing 263670 dashboard record is
from a different frozen start regime and remains protected; it is not silently
used as an apples-to-oranges starter-free statistical threshold.

Maximum, P95, and P99 score are mandatory descriptive reports, never
conjunctive PASS gates. Any maximum-score or record claim is adjudicated only
in the separate promotion review after the powered score and progression
confirmation has passed.

## 9. Governance and Current Decision

Each phase is one-shot, marker-bound, deterministic, and separately sealed.
No partial outcome may alter the architecture, schedule, stream rows, gates,
or sample sizes. Human sessions remain opaque and human actions are never
labels. One heavy job runs at a time.

Current state:

- `CONTINUE`: J1 implementation, tests, compact denylist, and zero-work
  execution preflight may be proposed after review.
- `HOLD`: all J1 streams, games, training, checkpoints, development,
  confirmation, incumbent, and dashboard work.
- `KILL`: further O6 continuation/retry; all prior permanent kills remain.
- `PROMOTE`: false.

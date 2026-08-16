# J2 Incumbent-Distilled Joint Policy/Value Readiness Charter

Status: outcome-free design and readiness only. All J2 teacher trajectories,
teacher action labels, games, optimizer steps, checkpoints, development,
confirmation, promotion, and incumbent or dashboard changes remain held.

## 1. Course boundary

J1d V2 is permanently `HOLD_J1_LEARNING_SANITY`. Its round-64 checkpoint,
optimizer state, roots, transitions, streams, metrics, and episode bodies are
spent, quarantined, and forbidden for J2. J1c remains
`KILL_J1C_TRAINING_INTEGRITY`. Every earlier kill, hold, transfer barrier,
human-source barrier, incumbent lock, top-three lock, and retention rule remains
in force.

J2 asks one new question: did J1 fail because a from-scratch actor/critic could
not bootstrap, rather than because sustained full-policy learning has no useful
actor signal? J2 uses ordinary behavior cloning on complete teacher-distribution
trajectories, requires a separate closed-loop full-policy teacher-fidelity gate,
and only then permits one sustained full-policy PPO fine-tune. The behavior
cloning phase is not DAgger and does not claim protection from student-induced
covariate shift. J2 does not use a J1/O3/O5 checkpoint, protected episode body,
recorded behavior action, human action, or human session as an input, target,
initialization, or selection signal.

This charter authorizes only a new readiness namespace:
`threes_rl/runs/forensics/j2_incumbent_distillation_readiness_v1`.
The future scientific namespaces must remain absent:

- `threes_rl/runs/forensics/j2_distillation_execution_v1`;
- `threes_rl/runs/forensics/j2_on_policy_training_v1`;
- `threes_rl/runs/forensics/j2_development_v1`; and
- `threes_rl/runs/forensics/j2_confirmation_v1`.

The readiness runner may expose only `audit-zero-work`,
`write-test-evidence`, and `prepare`. It must not contain or dispatch any
marker, reservation, consumption, game collection, teacher query, scientific
loss update, checkpoint, development, confirmation, or promotion command.

## 2. Teacher authority

The sole teacher is the exact protected composite software incumbent named by
`threes_rl/current_incumbent_policy.txt`: expectimax search plus its parent,
student, replay-calibration, and action-label sidecars. Readiness binds,
byte-for-byte:

- the incumbent policy file and resolved specification;
- all four checkpoint directory manifests;
- `eval.py`, `expectimax.py`, `ntuple.py`, `action_prior.py`, `sim.py`,
  `train_td.py`, `obs.py`, and `env.py`; and
- the composite incumbent binding produced by the reviewed J1 execution
  surface;
- the exact governance excerpt naming this composite frozen actor and the
  retained `ntuple_phaseblend_labelcorr_w010_endgame_1000_1020` replays;
- that run's immutable summary and top-game manifest; and
- the exact rank-1 replay byte hash for seed 1011 and score 263670.

The live dashboard must point eligible record 263670 to that same retained run
and replay. The governance excerpt is matched as exact text inside the
append-only log; the generated dashboard is checked by semantic fields rather
than bound by whole-file hash. Any source, configuration, checkpoint,
provenance, or composite binding drift is an integrity stop. The dashboard must
still report top three `[263670, 261369, 258561]`.

Future distillation must query a legal deterministic incumbent action at every
visited teacher state in each complete normal-start teacher trajectory.
Teacher actions are outputs of that exact composite policy, never human or
recorded behavior labels. An illegal teacher action is an integrity failure.

## 3. Prospective whole-root authority

Every J2 scientific row is prospectively fixed before outcomes. Root and
ancestry identifiers are SHA-256 commitments over the readiness-frozen stage,
row index, and exact logical/deck/slot/policy stream identities. A later
execution marker may activate only those committed rows and may not change a
root, ancestry, stage, stream, count, or order.

All J1/J1a prospective namespaces 213B through 226B are denied, including
declared but unopened evaluation ranges. All actually used J1b, J1c, and J1d
subranges are also spent. J2 uses these fresh ranges:

| Stage | Roots or pairs | Logical | Deck | Slot | Candidate/teacher | Control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher behavior cloning | 8,192 roots | 227B | 228B | 229B | 230B | none |
| BC validation and closed-loop fidelity | 2,048 pairs | 231B | 232B | 233B | 234B | 235B |
| on-policy training | 16,384 roots | 236B | 237B | 238B | 239B | none |
| development | 896 pairs | 240B | 241B | 242B | 243B | 244B |
| confirmation | 4,480 pairs | 245B | 246B | 247B | 248B | 249B |

Within each block the row offset starts at zero and is added to every listed
base. Validation/fidelity, development, and confirmation candidate/control
arms share logical, deck, and slot streams within a pair and have distinct
policy streams. The validation control arm is the complete teacher trajectory;
the same pair's student arm supplies sustained closed-loop fidelity. Every stage
has disjoint roots, ancestries, and streams. The readiness manifest has 32,000
ordered whole-root rows or pairs. The unique-stream count is derived, never
copied: `4*8192 + 5*2048 + 4*16384 + 5*896 + 5*4480 = 135424`.
It is a content-blind commitment only: no stream is reserved or consumed at
readiness.

Any collision with a compact immutable historical authority, duplicate row,
duplicate root or ancestry, cross-stage overlap, pair mismatch, or unknown
authority is an integrity stop. No schema-heterogeneous global payload scan is
allowed.

## 4. Model and initialization

J2 uses exactly one model:

- input width 282 and the frozen J1 observation encoding;
- fully connected 282 -> 512 ReLU -> 512 ReLU body;
- legal-masked four-action policy head;
- one scalar value head; and
- no auxiliary head, auxiliary parameter, or auxiliary loss.

The exact parameter count is 410,117. Initialization is from scratch with seed
`2026072806`; no J1, J1c, J1d, O3, or O5 tensor may be loaded. CPU PyTorch
2.12.1, intra-op threads 1, inter-op threads 1, deterministic algorithms, and
one heavy job are binding. Torch optimization is one process. Teacher rollout
may use only the fixed child-process collector in Section 8.

All observations, logits, probabilities, values, losses, gradients, model
tensors, and optimizer tensors must be finite. Every live row must contain at
least one legal action. Masked actions can never be selected or labeled.

## 5. Distillation contract

The proposed counts are frozen without a sweep:

- 8,192 complete teacher roots for fitting;
- 2,048 whole-root-disjoint paired teacher/student roots for validation;
- eight deterministic cumulative epochs;
- Adam learning rate `3e-4`, epsilon `1e-5`, no weight decay;
- root-equal minibatches of 4,096 rows, retaining the final short minibatch;
- gradient norm clip 0.5; and
- loss = root-equal legal-masked teacher cross-entropy plus
  `0.5 * root-equal value MSE`.

The value target at a visited state is
`1e-5 * (final_score - current_score)`. It must equal the sum of all remaining
dense score-delta rewards in that complete trajectory. No auxiliary, terminal
milestone, human, or adaptive reward enters the return.

The deterministic epoch/minibatch plan and optimizer state must resume exactly.
Every root has total effective weight one in both policy and value reductions,
independent of trajectory length.

Every complete validation pair is retained unconditionally. Trajectory-position
quartiles remain descriptive only. Capability-relevant state families are
frozen from current board features, never future outcomes:

- `low_air`: current maximum tile below 192 and at least four empty cells;
- `low_constrained`: current maximum tile below 192 and at most three empty
  cells;
- `mid_progression`: current maximum tile from 192 through 767; and
- `upper_progression`: current maximum tile at least 768.

The natural inventory contains every teacher-arm state and reports both
feature-family and trajectory-quartile frequencies. A second feature-only
capped inventory uses a deterministic support rule: let `K` be the smaller of
8,192 and the smallest natural feature-family state count, then select the
first `K` rows in canonical root/transition order from each family. This capped
inventory is used only for mechanism metrics. It never removes, replaces,
filters, or outcome-selects a complete validation pair.

Each feature family must contain at least 1,024 natural states from at least
256 distinct validation roots; no natural family may exceed 0.70 of validation
states; and the capped inventory must contain all four families with maximum
share at most 0.40. A shortfall is `HOLD_J2_DISTILLATION_DATA_SUPPORT`, not
representation evidence.

The immutable BC mechanism gate is:

- root-equal legal-action accuracy at least 0.97 overall on the complete
  natural inventory;
- legal-action accuracy at least 0.94 in each of the four capped feature-family
  inventories;
- no capped state family exceeds 0.40 of capped validation states;
- finite policy and value losses;
- value MSE strictly below the zero-prediction MSE; and
- every teacher action legal.

A clean gate miss holds J2. Counts, epochs, architecture, optimizer, or gates
may not be changed after evidence. Readiness must report any prospective
cost/power infeasibility and stop rather than silently changing the proposal.

Behavior cloning alone may not open PPO. The same separately sealed,
root-disjoint 2,048-pair validation authority must compare the deterministic
masked-argmax distilled student against the exact teacher from normal starts.
Both policies control their entire arm to natural termination, share
logical/deck/slot streams within each pair, and retain every pair. Four
content-blind root families are assigned by `row_index mod 4`; crossing each
with `floor(row_index/4) mod 2` yields eight equal prospective strata. Family
and stratum signs are descriptive, not conjunctive.

The fidelity score estimand is the paired root-equal difference in
`log1p(max(final_score - start_score, 0))`. The gate requires its point
estimate above `log(0.97)` and its global paired-root bootstrap lower 95% bound
above `log(0.90)`. Under the frozen J1a paired standard deviation 1.25,
N=2,048 has 80%-power score MDE 8.045645% and 96.816575% power for the 10%
CI-only noninferiority condition when the true policies are equal. The stricter
point floor makes the exact combined score-gate power 86.493019%. The 5% margin
has only 45.900198% CI-only power at this N and is explicitly not claimed.

The fidelity progression estimand is the accepted eight-stratum P1536
common odds ratio with 0.5 edge correction and fixed-total within-stratum
whole-root bootstrap. The gate requires point OR at least 0.90 and lower 95%
bound above 0.50. Readiness must simulate the exact gate over the frozen
2%-15% teacher-rate and 0.0/0.05/0.10 pairing-coupling envelope with 768
datasets and 199 bootstraps per cell, report an OR noninferiority-margin MDE
grid, and require worst-case power at least 0.80 under true OR 1.0. A teacher
rate below 0.02 is `HOLD_INCONCLUSIVE`. Illegal student actions must be zero;
latency, survival, family, and stratum summaries are mandatory safeguards.

Only a sealed PASS on both the BC mechanism gate and this sustained
full-policy fidelity gate can authorize PPO opening. No first-action fidelity
gate may substitute.

## 6. On-policy fine-tuning

Only a separately authorized, authoritative distilled checkpoint may initialize
one 16,384-root normal-start PPO run:

- 64 rounds of 256 complete roots;
- `starter_tile=None` for every reset;
- 16 synchronous environments in one process;
- dense score-delta reward with exact episodic telescoping;
- transition-t GAE, gamma 1.0, lambda 0.95;
- root-equal clipped PPO, clip 0.2;
- value coefficient 0.5, entropy coefficient 0.01;
- four epochs per round, minibatch 4,096 with final short batch retained;
- Adam learning rate `3e-4`, epsilon `1e-5`;
- gradient clip 0.5; and
- exactly one round-64 candidate, with no sweep or checkpoint selection.

The only new PPO term is a deterministic-teacher KL anchor. Exact-teacher
queries on student-visited states in rounds 1 through 16 are the prespecified
DAgger-like covariate-shift mitigation; the preceding teacher-distribution
pretraining remains ordinary behavior cloning. Because the teacher is
deterministic, the per-row anchor is the legal-masked negative log probability
of its queried action. Its coefficient is:

`0.05 * (17 - round) / 16` for rounds 1 through 16, so round 1 is exactly
0.05 and round 16 is exactly 0.003125. It decays across rounds 1 through 16
and reaches exactly zero before round 17; it is exactly zero for rounds 17
through 64.

The teacher is queried on every current on-policy state in all of rounds 1
through 16 and is neither queried nor carried as a label in rounds 17 through
64. No alternate schedule, imitation mixture, replay source, warm start, seed,
or contingency variant is permitted.

The training-sanity gate requires:

- all 16,384 roots and every frozen optimizer step exactly once;
- exact save/load and authenticated round-64 checkpoint identity;
- final-four-round root-equal mean log score above the first four rounds;
- finite final legal entropy at least 0.15 nats;
- final-round value MSE below the zero-value baseline;
- retained deterministic-teacher action fidelity at least 0.90 overall and
  at least 0.85 in each frozen feature family on the untouched
  2,048-root distillation-validation inventory; and
- no auxiliary head, target, loss, or gate.

A clean miss is HOLD, never a seed retry. A pass permits only a separately
authorized development screen.

## 7. Full-policy evaluation

Development and confirmation compare sustained deterministic masked-argmax J2
control against sustained unchanged-incumbent control from normal starts.
Candidate and control share exogenous logical/deck/slot streams within each
pair. No first-action continuation gate is permitted.

The accepted J1a counts and arithmetic remain exact:

- development: 896 fresh pairs, permissive screen;
- confirmation: 4,480 separately sealed fresh pairs;
- score estimand:
  `log1p(max(final_score - start_score, 0))`, paired by root;
- progression estimand: eight-stratum common odds ratio for P1536, with the
  accepted 0.5 edge correction and within-stratum whole-root bootstrap;
- 4,096 evaluation bootstrap replicates, development seeds
  `2026072817`/`2026072818`, confirmation seeds
  `2026072819`/`2026072820`, and linear 0.025/0.975 quantiles;
- family/block signs descriptive, never conjunctive; and
- maximum, P95, and P99 mandatory descriptive statistics only.

The accepted J1a arithmetic reports:

- development score 80%-power MDE 12.411155%;
- development worst-case OR1.50 power 0.272135 and OR MDE grid 2.5;
- confirmation score 80%-power MDE 5.371378%, with 95.183401% power for a
  7% score lift;
- confirmation worst-case OR1.50 power 0.84375 and OR MDE grid 1.5.

Development requires score point direction above zero, lower 95% bound above
`log(0.95)`, upper 95% bound reaching `log(1.07)`, progression point at least
1.0, progression upper bound at least 1.5, and frozen legality/survival/runtime
safeguards. It is a permissive screen, not a powered KILL gate.

Confirmation requires score point at least `log(1.07)` with lower bound above
zero, progression point at least 1.5 with lower bound above 1.0, and frozen
safeguards. Control P1536 below 0.02 is `HOLD_INCONCLUSIVE`. A null can KILL
only when the frozen design excludes the minimum meaningful effect.
Confirmation must be precommitted before development opens and separately
authorized before any confirmation content is read.

J2 is a competence bootstrap, not proof of improved maximum-score capability.
Development and confirmation must also report P3072, P6144, P95, P99, and the
single maximum. P3072 receives a preregistered common-OR non-regression
analysis only if its control rate is at least 0.02 and the N=4,480 outcome-free
power simulation supports the frozen margin; otherwise it is
`HOLD_INCONCLUSIVE`. P6144, P95, P99, and maximum remain mandatory descriptive
statistics because no powered tail gate is asserted without adequate base-rate
support. If J2 confirms only central log-score or P1536 improvement, it may
authorize only a separately frozen tail-capability stage. It cannot promote or
support a record claim.

No mechanism, development, confirmation, promotion, or dashboard command is
present in the readiness runner.

## 8. Resource and retention contract

One heavy job at a time, nice at least 10, more than 100 GiB free with a
120 GiB target, healthy recorder/dashboard/advisor services, and protected top
three are binding. Human sessions stay opaque and uninterrupted.

Prospective caps after a fixed 25% safety margin are:

- behavior cloning, mechanism validation, and closed-loop fidelity:
  72 active wall-clock hours and 24 GiB;
- on-policy training: 72 active hours and 24 GiB;
- development: 24 active hours and 8 GiB; and
- confirmation: 120 active hours and 16 GiB.

Readiness uses only synthetic maximum-shape buffers and previously sealed
engineering timing identities. It must report central 512-move projections and
a descriptive 5,000-move sensitivity. It may not query the teacher, run a game,
retime a scientific source, or treat sensitivity as a favorable adaptation.

The pre-PPO teacher workload is exactly 10,240 teacher-controlled roots: 8,192
BC roots and 2,048 validation/fidelity control arms. Readiness must project the
sealed incumbent fixed-state median and p99 action costs both serially and
under the following prospective sharding contract:

- one top-level owner and bounded job;
- exactly eight child worker processes, each single-threaded;
- row `i` is assigned to shard `i mod 8`;
- each worker emits immutable complete-root blobs in increasing row order;
- the parent merges only in global row order after exact per-root identity and
  hash checks;
- no work stealing, adaptive shard count, early stopping, or content-based
  allocation; and
- resume restarts only an uncommitted root in its original shard and charges
  abandoned work conservatively.

The machine exposes 12 logical CPUs, but core count alone is not throughput
evidence. The projection must report central 512-move wall time, total active
CPU time, and 5,000-move sensitivity. A READY decision additionally requires
pre-existing, outcome-free real-incumbent multi-process throughput and memory
evidence proving the eight-shard projection fits 72 hours and 24 GiB.
Synthetic sharding tests prove deterministic root ownership, exact equivalence,
and merge identity, but cannot supply this real-incumbent cost evidence. If that
evidence is absent, readiness must seal
`HOLD_J2_INCUMBENT_DISTILLATION_PREFLIGHT` for feasibility; it may not bless
ideal scaling or change the 8,192/2,048 counts.

The PPO projection must separately include all exact-teacher KL queries in
rounds 1 through 16: 4,096 on-policy roots and their central 512-move state
sequences. Those queries use the same fixed eight-shard ownership within each
round, while Torch optimization remains single-process and single-threaded.
Serial and sharded teacher CPU/wall projections are added to the inherited
bounded J1 training cost. If the online teacher anchor cannot fit the frozen
72-hour and 24-GiB caps under validated real-incumbent sharding, readiness is
HOLD; it may not replace exact teacher actions with a cheaper proxy.

The overall exact-teacher workload is therefore 14,336 root equivalents:
10,240 pre-PPO teacher arms plus 4,096 on-policy roots. Student fidelity arms
are excluded from teacher-query accounting. Pretraining collector throughput
and on-policy synchronous querying are distinct readiness gates. The round-r
PPO update may begin only after all 256 round-r roots and their exact-teacher
labels have been merged in canonical root/state order. If fixed eight-way
collection cannot preserve that synchronous round boundary, deterministic
teacher identity, or exact resume semantics, readiness must HOLD specifically
for on-policy teacher-query orchestration even if pretraining sharding is
otherwise feasible.

Future execution requires per-phase create-once markers, manifests, ownership,
reservations, consumption, runtime clocks, deterministic resume, terminal
results, and retention. No cleanup is allowed except a preregistered,
hash-bound retirement manifest.

## 9. Readiness evidence and terminal

Before sealing readiness:

- reproduce the J1d V2 readiness and terminal identities, J1a arithmetic,
  protected incumbent binding, historical authority, and prospective J2
  manifest;
- prove model parameter count, no-aux schema, legal masking, root-equal
  weighting, target telescoping, deterministic distillation resume, fixed
  eight-shard assignment/merge, natural and capped inventory accounting,
  full-policy fidelity power, KL endpoints, fixed development/confirmation
  arithmetic, create-once behavior, tamper rejection, and all
  stage/stream/ancestry disjointness in synthetic fixtures;
- run `py_compile`, focused J2 tests, immutable parent suites, and applicable
  non-scientific regressions;
- audit operational state and all future J2 namespaces; and
- attest zero markers, reservations, consumption, games, teacher queries,
  scientific labels, scientific optimizer steps, scientific checkpoints,
  holdout reads, policy/score outcomes, human-session reads, incumbent or
  dashboard changes, and promotion actions.

Seal exactly one of:

- `READY_J2_INCUMBENT_DISTILLATION_PREFLIGHT`;
- `HOLD_J2_INCUMBENT_DISTILLATION_PREFLIGHT`; or
- `KILL_J2_READINESS_INTEGRITY`.

Even READY authorizes no scientific work. It permits only research-lead review
and a separately frozen execution surface.

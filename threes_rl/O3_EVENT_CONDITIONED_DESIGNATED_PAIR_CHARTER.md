# O3 Event-Conditioned Designated-Pair Option Charter

Date: 2026-07-26

Status: outcome-free P0 feasibility and power preflight. No fresh game,
rollout, label, fit, policy outcome, score access, incumbent change, or
dashboard change is permitted before P0 seals READY.

## 1. Historical Locks

O2 remains authoritatively `HOLD_O2_DATA_SUPPORT`. Its 128-root pilot,
recovery marker/result, A4 cells, source replays, and gates are immutable and
may not be enlarged, rerun, retuned, or reinterpreted. O2 is used here only
through these aggregate facts already sealed in
`CURRENT_DECISION_LEDGER.md` and `EXPERIMENT_LOG.md`:

- 128 unconditional roots, 32 from each of four collector families;
- 7,192 frames scanned and 267 aggregate candidates;
- credited stage counts, ordered stage 0/1/2/3:
  `T768=0/0/0/0`, `T384=4/4/0/3`, `T192=5/5/0/9`,
  `T96=9/9/0/9`, and `T48=9/9/2/9`;
- 7 of 20 availability cells passed.

The O2 support JSON is byte-hash-bound only. O3 code must never parse its
candidate rows, state hashes, replay paths/content, pair coordinates, or
per-root availability. No O2 root may enter an O3 partition, rollout, label,
fit, or outcome.

G3/G4 hazard/ranking models, exact-depth-3 search, MCTS/UCT, O1/O2
stage-balanced classification, broad score/value fitting, behavior imitation,
and human-action supervision remain killed or held as previously recorded.
Human training remains held.

## 2. Scientific Question

Can one scale-relative relational policy sustain closed-loop control of a
designated equal-valued tile pair until those exact descendants merge within
40 moves, while preserving anchor and air, when trained only on simulator
event/time and successor-geometry signals at ordinary
`T={48,96,192}`?

O3's absorbing event is pair-specific merge, not occupancy of an intermediate
geometry stage. The old stage 0/1/2/3 name may be reported descriptively, but
stage is never an acquisition quota, partition cell, training label,
checkpoint-selection input, pass gate, or failure gate.

## 3. Exact State, Pair, And Lineage Contract

### Safety

- Full-state anchor safety is true when `starter_tile` is null or the exact
  starter remains at board coordinate `(0,0)`.
- Full-state air safety is exactly `empty_count >= 2`.
- A pre-spawn afterstate is unconditionally air-safe only at
  `empty_count >= 3`, because the mandatory insertion consumes one empty.
- An option root must be live, anchor-safe, air-safe, have at least two legal
  actions, contain an eligible designated pair, and have zero pair-specific
  safe-merge actions for the canonical designated pair. A root with an
  immediate safe merge is merge-ready descriptive/calibration support only
  and cannot enter train, development, or powered mechanism counts.

### Target and pair

The starter is replaced by zero only in the nonmutating target-selection view.
Training, development, and mechanism-test targets are exactly
`T={48,96,192}`. Integrated normal-start use may additionally invoke
`T={384,768}`; those scales never enter fitting or checkpoint choice. O3
never invokes an option at `T=1536`: the starter/anchor contract makes that
merge incompatible with O3 success, so the frozen incumbent handles every
`T>=1536` decision.

For a requested `T`, enumerate every lexicographically ordered coordinate pair
whose two live values equal T. For each pair define:

- Manhattan and Chebyshev distance;
- aligned-row and aligned-column bits;
- strict-between blockers when aligned;
- otherwise occupied cells in the pair's closed axis-aligned rectangle,
  excluding the pair coordinates;
- exact pair-specific safe-merge actions under the tagged simulator move.

Choose the pair by minimum tuple:

`(no_safe_merge_action, manhattan + blocker_count, blocker_count,
manhattan, chebyshev, pair_coordinates)`.

When no target is requested, choose the largest eligible T first, then apply
that pair rule. Pair choice is deterministic and does not use future events,
score, behavior action, policy value, source identity, or wall time.

### Designated lineage

At option start, the two selected coordinates receive immutable lineage bits
`A=1` and `B=2`; every other tile has bit zero. Exact simulator line motion
moves each bit with its tile and bitwise-unions lineages on a merge, using the
same moved/merged guards as the simulator.

- `success`: A and B merge each other into one `2*T` tile and the resulting
  full post-spawn state is anchor-safe and air-safe.
- `third_party_merge_failure`: a tile carrying A or B merges with any
  zero-lineage tile before A and B merge each other.
- `lineage_failure`: either bit disappears, duplicates, or leaves exactly one
  live descendant before success.
- Other failure: terminal state, no legal action, anchor violation, or
  `empty_count < 2`.
- `censor`: still live without success after exactly 40 option moves.

Pair merge in an unsafe post-spawn state is failure, never success. Every
tagged board and insertion-slot result must equal the exact simulator result.
Spawned tiles always have lineage zero. No count increase of `2*T` can
substitute for the designated A+B merge.

## 4. Representation And Model

O3 is scale-relative, not claimed exactly scale-equivariant. It contains no
absolute T scalar, but exact small tiles and deck mechanics break literal
scale symmetry.

Each legal queried action receives one forward pass. The 16 cell tokens have
37 ordered finite inputs:

1. current rank category, 11-way one-hot:
   `empty,small1,small2,small3,<=T/16,T/8,T/4,T/2,T,2T,>=4T`;
2. queried pre-spawn afterstate rank category, the same 11-way order;
3. current A and B lineage masks;
4. afterstate A, B, and merged-A+B lineage masks;
5. starter mask;
6. queried action's legal insertion-slot mask;
7. row one-hot and column one-hot, four each.

The 35 ordered global inputs are:

- preview kind one-hot `blue,red,gray,bonus`;
- normalized remaining small-bag counts `red,blue,gray`;
- normalized small position, total-small seen, span position, and
  large-pending;
- minimum, mean, and maximum visible bonus-candidate rank relative to T,
  with all-zero missing convention;
- current and queried-afterstate empty counts;
- current legal-action count;
- distance to forced plus;
- maximum board rank relative to T;
- pair Manhattan, Chebyshev, blocker count, same-row, same-column, clear-line,
  adjacent, and diagonal-touch indicators;
- queried action one-hot in simulator order;
- queried afterstate pair-merge bit, third-party-merge bit, anchor-safe bit,
  and three-pre-spawn-empty bit.

Continuous distances/counts use the fixed board maxima in the implementation
schema and are clipped to `[0,1]`. Source, family, root, frame, seed, score,
future milestone, behavior/human action, prior model value, and wall time are
excluded.

The sole model is:

- `Linear(37,64)` plus GELU for each of 16 tokens;
- two `TransformerEncoderLayer` blocks with width 64, four heads,
  feed-forward width 128, GELU, dropout zero, batch-first;
- concatenate mean token pool, max token pool, designated-lineage token pool,
  and the 35 globals;
- `Linear(227,128)`, GELU, `LayerNorm(128)`;
- `Linear(128,29)`.

The first five logits are mutually exclusive event classes:
safe pair merge in moves 1-10, 11-20, 21-40, failure, censor. The remaining
24 values are eight bounded successor-geometry quantities at h10, h20, h40:
Manhattan, Chebyshev, blockers, same-row, same-column, empty count, legal
count, and exact two-descendant-lineage integrity. These are continuous or
binary relations, not stage labels.

Only the action actually chosen at a decision receives a row; O3 generates no
all-action counterfactual label rollouts. The acting head is therefore trained
on the same queried-action semantics it ranks, with exploration supplying
action coverage. For a chosen action at option decision `d`, offsets are
measured from that decision, not from option activation:

- safe merge at relative move `1..10`, `11..20`, or `21..40` supplies the
  corresponding event class;
- any frozen failure before safe merge supplies the failure class;
- no event through 40 relative moves supplies the censor class;
- when the option-start h40 boundary leaves fewer than 40 observed future
  moves, a success/failure observed before that boundary remains valid, but a
  still-live row is right-censored early and its event loss is masked rather
  than called h40 censor;
- successor geometry at offsets 10, 20, and 40 is measured from `d` and is
  present only when the tagged option remains live through that exact offset;
  unavailable or post-termination checkpoints are masked.

No later checkpoint is fabricated as success or merge. The cumulative replay
buffer retains every chosen decision row. In each epoch, each represented
family has total weight `1/F`, each root within that family has
`1/(F*n_family)`, each of its 12 trajectories has one twelfth of that root
weight, and valid rows within a trajectory split that trajectory's weight
equally and separately for the event loss and each checkpoint head. Thus long
or prolific trajectories cannot dominate.

Loss is the weighted valid-row event cross entropy plus `0.10/3` times each
observed checkpoint geometry loss. Continuous quantities use SmoothL1;
binary quantities use BCE. Terminal score is absent. There is no dense reward
or shaping term.

PyTorch `2.12.1` CPU is frozen. AdamW uses learning rate `3e-4`, weight decay
`1e-4`, batch 128, gradient-norm cap 1.0, deterministic seed `2026072703`.
There is no architecture, reward, optimizer, regularization, seed, or
checkpoint sweep.

## 5. Closed-Loop Learning And Acting

Exactly 96 selected train roots generate four rounds and three trajectories
per root per round, for 1,152 option episodes. Round 1 uses uniform legal
actions. Rounds 2-4 use the current model with epsilon
`0.15,0.10,0.05`. Five cumulative-buffer epochs follow each round.
The round-4 checkpoint is mandatory. Development cannot choose a checkpoint,
threshold, sign, feature, or hyperparameter.

Every learning episode starts from the frozen hard-start root and uses one
h40 option clock. At each live decision, uniform/epsilon exploration chooses
one legal action; otherwise the frozen ordering below chooses it. The complete
subsequent tagged trajectory supplies labels for every chosen decision under
the relative-offset and masking rules in section 4. There is no behavior-action
label and no counterfactual branch.

For remaining horizon r, action ordering is lexicographic:

1. predicted safe-merge probability in event bins whose lower endpoint is at
   most r: bin 1 for `r>=1`, bin 2 for `r>=11`, bin 3 for `r>=21`;
2. predicted nonfailure probability;
3. predicted h10 successor potential
   `-0.45*manhattan -0.20*chebyshev -0.20*blockers
   +0.05*same_line +0.05*empties +0.05*legal_mobility`;
4. lowest simulator action enum.

If `p=softmax(logits[0:5])`, item 1 is the sum of the included `p[0:3]`
entries and item 2 is exactly `1-p[3]`; censor probability is therefore
nonfailure. The h10 potential always uses output offsets `5:13`, with
same-line equal to `max(same_row,same_column)` and every geometry output
mapped through sigmoid before applying the fixed coefficients.

Comparisons use float64 values and a tie tolerance of `1e-12`. Illegal actions
are never forwarded.

The normal-start lifecycle is identical in development and confirmation:

1. Outside an option and outside cooldown, inspect targets in descending order
   `768,384,192,96,48`. At the highest target with at least two non-starter
   tiles, select its canonical pair. Activate only if the full hard-start
   predicate holds. Do not fall through to a lower target.
2. If the selected outside pair already has a safe-merge action, do not
   activate O3; the incumbent controls that move.
3. Once active, preserve the exact target and A/B lineage. If one or more
   pair-specific safe-merge actions become available, execute the lowest
   simulator action enum among them. Otherwise query every legal action and
   use the frozen ordering above.
4. Success, failure, or exactly 40 option moves terminates the option. The
   incumbent controls exactly the next complete game move as a one-move
   cooldown; activation is reconsidered on the following move.
5. When no hard-start option exists, during cooldown, or whenever the highest
   relevant target is `>=1536`, the frozen incumbent controls the move.

The model therefore controls every non-cooldown move of an active option;
incumbent and human actions are never targets.

## 6. Prospective Acquisition And Partitions

Exactly five genuine collectors are frozen in this order:

1. `o3_corner2`;
2. `o3_expectimax2`;
3. `o3_parent_mc1000`;
4. `o3_replaycal`;
5. `o3_qd_v2`.

They inherit the five pairwise-distinct immutable 64-state signatures from the
sealed G1-R QD5 preflight. All policy/checkpoint payloads are re-hashed and
loaded before execution.

The acquisition is nonadaptive and unconditional:

- exactly 20,500 complete fresh normal-start roots, 4,100 per family;
- one worker, nice at least 10, one heavy process, deterministic one-game
  family round robin, chunks of five;
- every completion row and replay retained regardless of support, score,
  milestone, maximum tile, geometry, or action;
- no partial support peeking or family early stop;
- hard limits 144 active hours, 28 GiB incremental output, 100 GiB free;
  target 120 GiB free.

Before replay content, each family's immutable stream order assigns roots:

- indices `0..1003`: train pool, 5,020 roots total;
- `1004..1338`: development pool, 1,675 total;
- `1339..4099`: untouched mechanism pool, 13,805 total.

Sizing uses only the target-specific conservative O2 aggregates: non-merge-ready
stage-0/1 support was `9/128` for T48 and T96, and `5/128` for T192. The
weakest `5/128` 90-percent Wilson lower endpoint is exactly
`0.01914143013104029`. Here "90-percent Wilson lower" means the lower endpoint
of the two-sided 90-percent interval, with
`z=1.6448536269514722`; it is equivalently a one-sided 95-percent lower bound.
At that bound, the role pools project respectively
`96.089979`, `32.061895`, and `264.247443` eligible hard-start roots, just
above their frozen requirements 96, 32, and at most 264. The full acquisition
projects `392.399318`. Stage-3 support is not used. This is conservative
prospective sizing, not a claim that O3 support will materialize.

The historical outcome-free throughput bound is
`9.195378486979166` seconds per completed game. Thus 20,500 games project
`52.362572` worker-hours nominal and `130.906430` hours at the frozen 2.5x
factor, below the 144-hour hard limit. With the frozen maximum replay size
`1,000,401` bytes, `65,536` compact-overhead bytes per game, a 1.25 filesystem
factor, and a 512 MiB fixed reserve, the conservative projection is
`27,851,506,537` bytes (`25.938737 GiB`), below 28 GiB. P0 must reproduce
these arithmetic values from byte-hash-bound aggregate constants.

After all games complete, support-only allocation selects at most one state
per whole ancestry. Within each role it visits targets rare-first
`192,96,48`, families in deterministic rotated round robin, and candidates by
`SHA256("O3-event-root-v1"|role|target|family|root|frame|state_hash)`.
No root can serve two targets or roles.

Required selected counts are:

- train 96: `T48=48,T96=29,T192=19`;
- development 32: `T48=16,T96=10,T192=6`;
- untouched test: the selected power N, allocated 50/30/20 by deterministic
  largest remainder.

Every role requires at least four genuine families, maximum family share
40 percent, and minimum per represented family `4/2/8` for train/dev/test.
Geometry-stage occupancy is reported only. Failure to allocate is
`HOLD_O3_DATA_OR_POWER`, never a representation kill.

## 7. Streams

All stream rows are frozen in P0 and must have zero historical collision.

| Purpose | logical | deck | slot | policy |
|---|---:|---:|---:|---:|
| acquisition | 105B | 106B | 107B | 108B |
| learning | 109B | 110B | 111B | 112B |
| option dev/test | 113B | 114B | 115B | 116B |
| normal-start development | 117B | 118B | 119B | 120B |
| sealed confirmation | 121B | 122B | 123B | 124B |

Acquisition code is `family_index*4100 + game_index`. Learning code is
`root_index*12 + round_index*3 + replicate`. Paired option evaluation code is
`1_000_000 + partition_offset + root_index*8 + replicate`. Paired arms share
logical/deck/slot streams and use separate policy streams
`base + 2*code + arm`; slot uniforms map independently over each arm's legal
insertion slots. Policy IDs are globally unique.

Normal-start development has 512 paired roots. Confirmation has 2,560 paired
roots. Those complete stream manifests are frozen before acquisition and are
root/stream-disjoint from every prior branch.

## 8. P0 Power And Decision

The mechanism endpoint is safe designated A+B merge by h40 under sustained O3
control versus sustained frozen-incumbent control. Eight paired CRN
replicates are read from one h40 path per root. The primary estimator is a
Mantel-Haenszel common odds ratio over target
`T={48,96,192}` crossed with starting alignment
`{aligned,unaligned}`. Empty observed strata are omitted and disclosed;
alignment and old stage signs are descriptive, not gates.

For each nonempty stratum, let `(a,b,c,d)` be treatment success/failure and
control success/failure counts across its root-replicates. Add 0.5 to all four
cells in that stratum only when any of its four cells is zero. With
`n=a+b+c+d`, the frozen point estimate is
`OR_MH=sum(a*d/n)/sum(b*c/n)`. A bootstrap draw resamples whole roots with
replacement independently inside each stratum, preserving all eight paired
replicates and both arms, then recomputes the same estimate. The interval is
the percentile `[0.025,0.975]` quantile over 399 draws. At N192 the target
counts are `96/58/38`; at N264 they are `132/79/53`. Within each target,
aligned receives the ceiling half and unaligned the floor half.

Prospective power simulates the exact estimator and final pass event:

- target OR 1.50;
- root base hazard `Beta(1.5,28.5)`, mean 0.05;
- target factors `1.20,1.00,0.80` and alignment factors `1.15,0.85`,
  clipped to `[0.005,0.50]`;
- independently for every root-replicate, a Bernoulli(0.50) coupling bit:
  when one, both arms use the same uniform; when zero, the arms use
  independent uniforms;
- eight repeats/root;
- target proportions 50/30/20 and deterministic balanced alignment;
- point gate OR at least 1.25 and 95-percent whole-root bootstrap lower bound
  above 1.00;
- 1,024 simulated datasets and 399 bootstrap replicates;
- candidate N `192,264`; OR grid
  `1.25,1.50,1.75,2.00,2.50,3.00,4.00`;
- PCG64 seed family rooted at `2026072704`.

The smallest N with at least 80-percent full-gate power at OR 1.50 is selected.
MDE is the smallest grid OR with at least 80-percent power at that N. If
neither N passes, P0 is HOLD before acquisition.

The frozen implementation reproduces these outcome-free rows:

| roots | true OR | lower-CI power | full pass power |
|---:|---:|---:|---:|
| 192 | 1.25 | 0.4912109375 | 0.4482421875 |
| 192 | 1.50 | 0.9267578125 | 0.9169921875 |
| 264 | 1.25 | 0.64453125 | 0.482421875 |
| 264 | 1.50 | 0.9765625 | 0.953125 |

Therefore the prospective design selects N192 and its grid MDE is OR1.50.
P0 must reproduce the complete seven-OR rows under the final code hash before
it may seal READY.

P0 seals exactly one:

- `READY_O3_EVENT_ACQUISITION`;
- `HOLD_O3_DATA_OR_POWER`;
- `KILL_O3_REPRESENTATION_PREFLIGHT`.

READY requires exact lineage/feature/model tests, coherent N at most 264,
20,500-root cost/storage feasibility, five exact families, fresh collision-free
streams, no heavy contention, healthy services/top three, and at least 120 GiB
free. READY authorizes the one frozen acquisition, not an outcome claim.

## 9. Mechanism And Capability Gates

Before fitting, the sealed 1,152-episode learning corpus must contain:

- at least 40 pair-specific safe-merge successes overall;
- at least six successes at each of T48, T96, and T192;
- at least four families with at least three successes each;
- at least 40 frozen failures and at least 40 true h40 censors;
- finite features/labels and at least two nonempty safe-merge time bins.

These thresholds use only event/support counts after all frozen learning
episodes complete. A miss seals `HOLD_O3_LABEL_SUPPORT`; it cannot alter the
root allocation, rollout count, exploration schedule, loss, or checkpoint,
and it is never a representation kill.

After the mandatory checkpoint is sealed, development opens once, then the
untouched mechanism panel opens once. Mechanism pass requires:

- common OR point at least 1.25 and 95-percent root-bootstrap lower bound
  above 1.00;
- action-changing activity on at least 20 percent of roots and two percent of
  option decisions, across at least four families;
- zero illegal actions;
- survival noninferior by -2 percentage points;
- anchor preservation noninferior by -1 percentage point;
- mean empty count noninferior by -0.5 cells;
- decision-latency p95 at most 100 ms and full h40 runtime at most 2.0 times
  control;
- no family above 40 percent of weighted lift.

Target, alignment, stage, and stream-block signs are descriptive. A powered
mechanism failure kills this exact O3 model; support scarcity remains HOLD.

A mechanism pass alone authorizes one 512-root fresh paired normal-start
development assay. The complete treatment invokes the highest eligible pair,
controls every option move through termination, and uses the incumbent only
outside options under the exact lifecycle in section 5. Let `D` be the
root-paired treatment-minus-control difference in
`log1p(score_minus_starter)` and let `[L,U]` be its 95-percent whole-root
bootstrap interval. The permissive development screen requires
`mean(D)>0`, `L>=log(0.95)=-0.05129329438755058`, and
`U>=log(1.05)=0.04879016416943204`. It also requires P3072 no worse than
-2 percentage points, survival no worse than -2 points, lower decile no worse
than -5 percent, zero illegal actions, option activity at least five percent,
and no corner/anchor block. Passing development is not a capability claim.

Only a development pass opens the 2,560-root confirmation. Its co-primary
gates are paired log-score 95-percent lower bound above zero and P3072 common
OR 95-percent lower bound above 1.00. The sealed historical aggregate design
has worst-case OR-1.50 power 0.84766 and paired-log-score 80-percent MDE
6.755 percent. P6144, P95, maxima, raw/winsorized score, lower tail, runtime,
survival, air, anchor, and option activity are mandatory safeguards or
diagnostics. Promotion additionally requires a provenance-valid treatment
maximum above paired control and protected record 263670.

No acquisition, rollout, development, mechanism, or training result is
dashboard-eligible. Only a clean independent confirmation may update the
incumbent or dashboard.

## 10. Governance And Retention

Every phase has a separate output namespace, zero-work open marker, immutable
terminal result, exact command, implementation/test/config hashes, source and
stream manifests, resume identity, and fail-closed integrity checks. Partial
results never alter a gate. One heavy process is allowed. Disk hard floor is
100 GiB and target is 120 GiB. Ports 8765/8770, advisor, dashboard record
263670, and protected top three 263670/261369/258561 must remain healthy.

No cleanup is permitted without a reviewed deletion manifest. All historical
locks and spent evidence remain protected.

# O1 Charter Amendment A1: Pair, Action, Partition, And Power Exactness

Date: 2026-07-26

This amendment is authoritative over
`O1_GOAL_CONDITIONED_GEOMETRY_OPTION_CHARTER.md` at SHA-256
`d6ea7fb6f0ff547cbc84486d723c90fb4603900004dc181dff1e02e58622bdb4`.
The base charter remains immutable. No O1 candidate replay content, rollout,
label, fit, prediction, action, score, or policy outcome was opened before A1.

## A1. Pair-Specific Merge Provenance

Every base swipe used by O1 geometry carries immutable source-coordinate tags
through the exact simulator line-motion algorithm:

1. Each nonzero input cell begins with the singleton tag containing its
   coordinate.
2. Moving a tile moves its tag unchanged.
3. Merging tiles unions their tags at the merged destination.
4. The same `moved_into` and `merged_into` guards as
   `advance_line_toward_start` are applied.
5. The tagged output values and insertion slots must exactly equal
   `simulate_base_move`; any mismatch fails P0.

A candidate pair is merged by an action only when one output tile has value
`2*T` and its provenance tag contains both exact coordinates of that selected
pair. Merging another pair at the same tile scale does not make the candidate
pair merge-ready.

The action is a safe pair merge only when:

- the candidate pair merged;
- the fixed starter remains at `(0,0)`;
- the pre-spawn afterstate has at least two empty cells.

`merge_ready` is pair-specific and true only when at least one legal action is
a safe pair merge.

## A2. Safe Success And Option Failure

The exact air-safe predicate at every observed full simulator state is
`empty_count >= 2`. The exact anchor-safe predicate is that a non-null starter
still occupies `(0,0)`, or true when no starter exists.

Any increase in the normalized count of `2*T` above its option-root count is
success for every requested next-stage goal, provided the resulting full state
is anchor-safe and air-safe. Otherwise:

- goals 1 through 3 succeed only when the same target scale reaches the
  requested stage or better and the full state is anchor-safe and air-safe;
- goal 4 succeeds only through the safe target merge above.

Unsafe touching, adjacency, merge readiness, or target merge never counts as
success. An option root itself must be anchor-safe, have at least two empty
cells, and have at least two legal actions.

Failure precedes censoring and occurs on terminal state, anchor loss, no legal
action, target count below two before safe merge success, or air loss
`empty_count < 2`. Horizons remain 10, 20, and 40 from one h40 path.

## A3. Action-Conditioned Scalar Distribution

The model is evaluated once for each legal candidate action.

The action-conditioned spatial tensor has 16 channels:

1. empty;
2. fixed starter;
3-9. clipped relative-rank bins;
10. selected pair;
11. support tile `T/2`;
12-15. broadcast goal one-hot;
16. insertion-slot mask for this candidate action's exact pre-spawn base move.

The global vector has the original ordered 24 fields followed by an ordered
one-hot candidate action in simulator order `up, down, left, right`, for width
28.

The network is:

- `Conv2d(16,32,3,padding=1)`, `GroupNorm(4,32)`, ReLU;
- the same two frozen 32-channel residual blocks;
- flatten and concatenate 28 globals;
- `Linear(540,128)`, ReLU;
- `Linear(128,20)`.

The first five logits retain the mutually exclusive event categories. The
remaining 15 logits retain the three successor-stage distributions. A legal
action's lexicographic score is computed from its own scalar head. Illegal
actions are never forwarded and receive score negative infinity.

## A4. Exact Labels, Masks, And Rollout Partitions

Only `train` roots may generate learning rollouts.

For every decision at offset `t` in a self-generated option trajectory, store
the exact state, fixed goal contract, chosen legal action, and remaining
horizon. Its five-class event label is:

1. safe success in relative moves 1-10;
2. safe success in 11-20;
3. safe success in 21-40;
4. failure before safe success;
5. still live and unsuccessful at the remaining h40 censor boundary.

Exactly one class is active. Decisions with fewer than 40 remaining moves use
their actual remaining boundary; a success is assigned by its relative
decision time, failure remains failure, and a live boundary is censoring.

Successor classes are `separated`, `diagonal_touching`, `adjacent`,
`merge_ready`, and `merged_success`. At each relative h10/h20/h40 checkpoint,
the auxiliary row is included only if that checkpoint was observed or safe
success occurred earlier. A failure before a checkpoint masks that auxiliary
head out. A safe success at or before the checkpoint maps to
`merged_success`; otherwise it maps to the exact pair stage then observed.

Root weight is one, split equally over that root's trajectories, decisions,
and included heads. Each genuine family receives equal total loss weight;
within-family roots divide that weight equally. No action-count, episode
length, or pair multiplicity can increase a root's total weight.

Round 1 collection uses uniform legal action exploration from every train root,
with policy-stream uniform variates and lowest-enum boundary ties. It does not
use the untrained network. After the fixed five round-1 training epochs,
rounds 2, 3, and 4 use the current network with epsilon
`0.15, 0.10, 0.05`, respectively. Five fixed cumulative-buffer epochs follow
each round.

Development roots are opened exactly once after the final round-4 checkpoint
is hash-bound. They report the frozen mechanism diagnostics but cannot select
a checkpoint, feature, optimizer, threshold, or policy. Untouched test roots
remain unopened until the final checkpoint, train manifest, and development
report are all immutable and hash-bound.

## A5. Genuine Family Classifier

The classifier first obtains `replay_behavior_family(replay, path)` and the
lowercased replay policy plus path text. It then applies these ordered rules:

1. `corner2_lineage` or exact policy prefix `corner2`:
   `g1r_corner2`.
2. `expectimax_baseline` or exact policy prefix `expectimax2`:
   `g1r_expectimax2`.
3. text containing `replaycal` or `replay_cal`:
   `g1r_replaycal`.
4. exact QD-v2 policy bundle identifier
   `g1r_qd_static_archive_oneply_v2_terminal_schema`:
   that same family.
5. `phaseblend_incumbent_lineage`, `phaseblend_cheap_lineage`,
   `legacy_ntuple_lineage`, `td_student_lineage`, `ntuple`,
   `train_td:*`, or text containing `student1`, `parent_mc1000`, or
   `phaseblend`: `g1r_parent_mc1000`.
6. Every other source is `ineligible_unverified_family`.

Rules 1-5 are bound to the immutable 64-state action panel SHA-256
`b8862aa3c8eaf6278fc078fb3e03aa7222a01930673cfee497738c74e81eff9d`,
pilot-v1 preflight file SHA-256
`f78288b3f47bda6aa6d15c2157fd79f7b3d0685f0367d8b9964f5dc73981ea91`,
its action-audit SHA-256
`f78184001df46b9eab4e71a7e620fb9247c9a05b88613846fc22f1879512eab4`,
and QD-v2 admission-result file SHA-256
`27bcb3328a02d6dc5094dcc5a8e52b8f27d2f3e4ea7b92f5c1a8153bc1326a8e`.
The parent, student, and incumbent aliases remain one family.

## A6. Deterministic Partition Allocator

The cross-product power stratum is `(starting_stage, scale_band)`, where scale
bands are `early={48,96,192}`, `mid={384,768}`, and `late={1536}`.

After one state per ancestry is frozen, allocation uses only root, family,
stage, scale band, and SHA hashes:

1. Test target is the smallest power-passing N rounded to a multiple of 12.
   Each of the 12 stage/scale cells receives exactly N/12 roots.
2. Within each cell, families are visited in lexical order rotated by
   `int(SHA256("O1-family-rotation"|cell)[:8],16) mod F`.
3. Roots within a cell/family are ordered by
   `SHA256("O1-test-v1"|cell|family|root)`.
4. Add in round-robin family order while no family's final quota can exceed
   `floor(0.40*N)`. Failure to fill any cell fails test feasibility.
5. Remove test roots. Allocate exactly 80 development roots, 20 per starting
   stage, using the same procedure and key prefix `O1-dev-v1`; require at
   least four families and no family above 32 roots.
6. Remove development roots. Allocate exactly 240 training roots, 60 per
   starting stage, using prefix `O1-train-v1`; require at least four families
   and no family above 96 roots.
7. Extra roots remain `inventory_only`.

There is no fallback reassignment, stage pooling, family relabeling, or
post-outcome balancing.

## A7. Power Uses The Frozen Common-OR Estimator

Power strata are the exact 12 allocator cells above. In each simulated design:

- roots are allocated equally across all 12 strata;
- each root's base probability is drawn from the frozen beta distribution,
  multiplied by fixed stage factors `{0:0.50,1:0.75,2:1.00,3:1.50}` and
  scale factors `{early:1.25,mid:1.00,late:0.75}`, then clipped to
  `[0.002,0.80]`;
- the treatment probability applies the candidate common odds shift;
- eight paired arm outcomes use the frozen 0.50 shared-uniform coupling;
- the point estimator is the exact Mantel-Haenszel common odds ratio over the
  12 stratum tables with a 0.5 continuity correction only when a numerator or
  denominator cross-product sum is zero.

For each N/OR design, 4,096 whole-root synthetic datasets are generated in
batches under NumPy `PCG64(2026072601 + 1000*N + round(100*OR))`. The standard
error is the across-dataset standard deviation of log common OR, multiplied by
1.10. This is conservative relative to the asymptotic whole-root cluster
bootstrap under the same independent-root generative model. Power is the
fraction whose log-OR lower bound
`log(OR_hat) - 1.959963984540054 * inflated_SE` exceeds zero.

Candidate N values, all divisible by 12, are
`{144,192,264,384,516,768,1020,1536}`. MDE is the smallest passing OR in
`{1.25,1.50,1.75,2.00,2.50,3.00,4.00}`. P0 READY still requires at least
80-percent power at OR 1.50. The eventual mechanism gate uses the same
Mantel-Haenszel point estimator and a 10,000-repeat whole-ancestry bootstrap
with seed `2026072603`.

## A8. Marker-Before-Content State Machine

Future E0 simulator stream bases are reserved, not consumed, at:

- logical `77_000_000_000`;
- deck `78_000_000_000`;
- slot `79_000_000_000`;
- policy `80_000_000_000`.

P0 has two commands:

1. `prepare`: path-only source discovery and hashing, protected
   root/stream-manifest inventory, historical action-signature verification,
   implementation/test/schema hashes, requested-stream collision audit,
   free-disk/process/service/dashboard checks, and immutable
   `O1_P0_CONTENT_OPENED.json`. It opens no candidate replay JSON.
2. `scan`: requires that exact marker and rejects every hash, inventory,
   service, process, disk, or collision mismatch before opening candidate
   replay content. It writes one immutable P0 result and cannot rerun.

The marker binds the base charter, A1, implementation, tests and evidence,
feature/model schema, source-path inventory, exclusion sources, family
evidence, power contract, future stream set, output directory, one-process
nice-priority contract, disk, services, dashboard record, protected top three,
and explicit zero outcomes/training work. Candidate replay content may not be
opened before that marker exists.

Any post-marker integrity error seals
`KILL_O1_REPRESENTATION_PREFLIGHT`; insufficient natural support or power seals
`HOLD_O1_DATA_OR_POWER`. Only a clean support, partition, power, and integrity
pass seals `READY_O1_E0_PILOT`.

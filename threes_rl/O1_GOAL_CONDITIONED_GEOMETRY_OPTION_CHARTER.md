# O1 Goal-Conditioned Geometry Option Charter

Date: 2026-07-26

## Historical Locks

The exact depth-3 program is permanently killed, including C1, C2, K1-v1,
and every K1 successor. The compiled K1 kernel is retained only as
non-promotable engineering evidence. No C1/C2/K1 untouched state, gate,
stream, timing, value, or action may be opened or reused.

G3 marginal hazard, G4 conditional pairwise ranking, prior UCT/MCTS, broad
scalar value fitting, action-prior imitation, and human-action supervision
remain killed. O1 is a new closed-loop option-policy experiment and may not be
used to reinterpret any earlier result.

## Scientific Question

Can one scale-normalized, goal-conditioned spatial policy learn to preserve
anchor and air while repeatedly controlling the game toward the next
relational stage of the highest useful duplicate pair, and improve the
stage-appropriate h40 next-stage odds under full option exposure?

The option policy controls every move until success, failure, or its fixed
horizon. A one-action treatment followed by incumbent continuation is
forbidden as the primary instrument.

## Exact Geometry Contract

### Board normalization

1. The board is a 4 by 4 integer array.
2. If `starter_tile` is non-null and cell `(0, 0)` equals it, that cell is
   replaced by zero only for target selection and relational geometry.
3. The live simulator board is never mutated.
4. Eligible target scales are exactly
   `{48, 96, 192, 384, 768, 1536}`.
5. `target_tile T` is the largest eligible scale with at least two copies on
   the normalized board and `2*T < 12288`.
6. A state with no such target has no O1 option and is ineligible.

### Pair selection

For all unordered coordinate pairs containing `T`, compute:

- Manhattan distance `|dr| + |dc|`;
- Chebyshev distance `max(|dr|, |dc|)`;
- whether the pair is orthogonally adjacent;
- whether the pair is diagonally touching;
- whether at least one deterministic legal pre-spawn base move increases the
  normalized board count of `2*T`, decreases the count of `T` by at least two,
  leaves the starter at `(0, 0)`, and leaves at least two empty cells.

The pair's stage rank is:

0. `separated`: Chebyshev distance greater than one.
1. `diagonal_touching`: Chebyshev distance one and Manhattan distance two.
2. `adjacent`: Manhattan distance one without an anchor-safe, air-safe merge
   move.
3. `merge_ready`: the deterministic safe-merge predicate above is true.

Choose the pair with maximum stage rank, then minimum Manhattan distance, then
minimum Chebyshev distance, then lexicographically smallest ordered coordinate
pair. Coordinates within a pair are sorted lexicographically first. Every tie
is therefore deterministic.

### Goals and termination

The frozen goal IDs are:

- `1=touching_or_better` from stage 0;
- `2=adjacent_or_better` from stage 1;
- `3=merge_ready_or_merged` from stage 2;
- `4=merged` from stage 3.

The target scale and root counts of `T` and `2*T` are fixed for an option.
Stage goals 1 through 3 succeed when the best remaining pair at the same
target scale reaches at least the requested stage. Goal 4 succeeds when the
normalized count of `2*T` exceeds its root count.

An option fails before success if the simulator is terminal, the fixed starter
is absent from `(0, 0)`, no legal action exists, the count of `T` drops below
two before goal 4 succeeds, or a zero-empty state has at most one legal action.
Otherwise it is censored at the frozen horizon. Primary horizon is 40 moves;
10 and 20 moves are fixed secondary checkpoints from the same trajectory.

## Natural Corpus Contract

P0 may discover regular files named `replay.json` below `threes_rl/runs`.
Before replay content is opened it must hash and freeze the discovered path
inventory.

Allowed sources must be completed, simulator-valid, normal-reset machine
games with fresh root provenance. Human, partial, replay-start, continuation,
restart, synthetic, and current active-recorder sources are forbidden.

The following path-selected sources are forbidden before content access:

- any `top_games`, ranked replay, replay playlist, or score-bearing path;
- any `continuations`, `human_diagnostics`, or imported human path;
- every prior `forensics` branch directory, including G1-G4, S3, C1/C2, and
  K1;
- any source explicitly named by a protected untouched or confirmation
  manifest.

No final score, future milestone, recorded action, policy outcome, prior label,
or favorable geometry may participate in source or state selection.

For every eligible whole ancestry, P0 may select at most one state. It finds
all option-eligible live frames, then selects the global minimum of
`SHA256("O1-P0-state-v1" | ancestry | target_tile | stage | frame |
state_hash)`. One ancestry can therefore occupy only one scale/stage cell.

All source replay bytes, selected state bytes, exact preview/deck context,
starter, frame, ancestry, family, and state hashes are bound. Exact restoration
and legal-action agreement are mandatory.

### Historical exclusions

The root and stream exclusion union includes every available S3, G1/G1-R,
G2/G3/G4, C1/C2, K1, R1/R1b/A2, human, selector, MCTS/UCT, continuation, and
confirmation manifest. Root identifiers may be read from immutable manifests
for exclusion only. Protected state content and outcomes remain unopened.

### Genuine families and partitions

Family identity uses immutable action-signature evidence, not checkpoint names.
At least four genuine machine behavior families are required overall and in
the untouched mechanism test. No family may exceed 40 percent of roots in any
partition.

Whole ancestries are assigned before rollouts:

1. `untouched_test`: lowest deterministic family/stage-stratified hashes up to
   the power-required count;
2. `development`: next 20 percent of the remaining roots in each
   family/stage cell, with at least 80 roots total and 16 per starting stage;
3. `train`: all remaining admissible roots, with at least 240 roots total and
   40 per starting stage.

There must be zero root, source, state, or stream overlap. If an exact
partition cannot be constructed, P0 is not READY. Pooled raw counts do not
substitute for partition feasibility.

## Frozen P0 Power Contract

The future primary mechanism endpoint is paired h40 success at the root's
requested next stage under full option-policy exposure versus full frozen
incumbent exposure. Analysis is a strata-standardized common odds ratio over
starting stage and target scale with whole-ancestry cluster inference.

Power is computed outcome-free using:

- target common odds ratio `1.50`;
- control root probabilities from `Beta(alpha=1.6, beta=18.4)`, mean 0.08;
- treatment probability formed by an exact odds shift of each sampled root
  probability;
- eight paired common-random-number replicates per root;
- shared-uniform coupling probability 0.50 and otherwise independent arm
  uniforms;
- equal root weight and equal total starting-stage weight;
- exact integration over the beta root mixture using 256-node Gauss-Jacobi
  quadrature;
- exact first and second moments of each paired Bernoulli difference under the
  frozen coupling, including between-root heterogeneity and eight-repeat
  within-root variance;
- two-sided normal-power calculation for the equal-root mean paired-rate
  difference at alpha 0.05;
- candidate N grid `{128, 192, 256, 384, 512, 768, 1024}`;
- normal critical value `1.959963984540054`.

P0 records power at OR 1.50 and the 80-percent-power MDE over
`{1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00}`. The smallest N with power at
least 0.80 is required. Untouched test support must meet that N, include at
least 48 roots in every starting stage, include at least four genuine families,
and satisfy the 40-percent family cap. Underpowered late-stage evidence is a
HOLD, never a representation failure.

## Frozen O1 Model

PyTorch `2.12.1` on CPU is the sole toolchain. No dependency install,
architecture search, optimizer sweep, reward-weight sweep, or checkpoint
selection is allowed.

### Inputs

The spatial tensor has 19 ordered 4 by 4 channels:

1. empty;
2. fixed starter;
3-9. clipped tile-rank offset to `T` in
   `<=-4, -3, -2, -1, 0, +1, >=+2`;
10. selected-pair coordinates;
11. support tile `T/2`;
12-15. broadcast one-hot goal IDs 1 through 4;
16-19. deterministic eligible insertion-slot masks after legal base swipes in
   simulator order `up, down, left, right`; an illegal swipe emits an all-zero
   mask.

The 24 ordered global inputs are:

- visible preview one-hot: blue, red, gray, bonus;
- normalized remaining small-bag counts: red, blue, gray;
- normalized small position, total small seen, span position, large pending;
- normalized minimum, mean, and maximum visible bonus-candidate rank offset;
- current geometry-stage one-hot;
- empty count divided by 16;
- legal-action count divided by 4;
- distance to forced plus divided by 21;
- normalized target scale `log2(T/48)/5`;
- clipped maximum board rank relative to T divided by 4;
- anchor-integrity bit.

Missing bonus candidates emit zero for all three candidate summaries. Every
input must be finite. No source, family, root, frame, seed, score, future
outcome, recorded action, or wall time enters the model.

### Architecture

- `Conv2d(19, 32, 3, padding=1)`, `GroupNorm(4,32)`, ReLU;
- two fixed residual blocks, each
  `Conv2d(32,32,3,padding=1)`, `GroupNorm(4,32)`, ReLU,
  `Conv2d(32,32,3,padding=1)`, `GroupNorm(4,32)`, residual add, ReLU;
- flatten, concatenate 24 globals;
- `Linear(536,128)`, ReLU;
- one `Linear(128, 4*20)` output reshaped by action.

For each action, five logits represent success in moves 1-10, 11-20, 21-40,
failure, or censoring. Fifteen logits represent the five successor geometry
states at h10, h20, and h40.

### Learning and acting

The loss is root/family-equal weighted event-category cross entropy plus
`0.25/3` times each successor-stage cross entropy. Optimizer is AdamW,
learning rate `3e-4`, weight decay `1e-4`, batch size 256, gradient norm cap
1.0, deterministic seed `2026072602`.

E0, if authorized by P0, uses four closed-loop collection rounds. The current
network controls every move with epsilon-greedy exploration
`0.20, 0.15, 0.10, 0.05`; ties use lowest action enum after deterministic
policy-stream noise. Five fixed training epochs follow each round over the
cumulative replay buffer. The final round checkpoint is used; no development
checkpoint selection occurs.

Action score is lexicographic: highest predicted success probability by the
remaining horizon, then highest nonfailure probability, then lowest action
enum. Illegal actions are masked. Behavior and human actions are never labels.
Terminal game score is never a target.

## Conditional E0 Mechanism Gate

P0 READY permits exactly one separately marked E0 with newly reserved,
collision-free logical/deck/slot/policy streams. E0 remains non-promotable.

The untouched full-option comparison requires:

- common OR point estimate at least 1.50 and 95-percent ancestry-bootstrap
  lower bound above 1.00;
- positive point direction at every starting stage with at least 48 test roots;
- at least 20 percent of test roots where O1 changes one or more actions,
  across at least four genuine families;
- zero illegal actions;
- survival non-inferior within -2 percentage points;
- mean h40 empty count non-inferior within -0.5 cells;
- anchor preservation non-inferior within -1 percentage point;
- no single family contributes more than 40 percent of weighted success lift.

Stream-block signs are descriptive, not a conjunction gate. First-action
agreement is secondary only.

A mechanism pass authorizes only a new preregistered normal-start integration
development block. Promotion ultimately requires paired normal-start
score-minus-starter improvement with a positive 95-percent interval, high-tile
non-inferiority, max-score evidence, and one sealed root-disjoint confirmation.
No O1 development, option continuation, or training high may change the
dashboard.

## Decisions

P0 seals exactly one:

- `READY_O1_E0_PILOT`: representation, provenance, partition, support, power,
  services, storage, and implementation tests all pass;
- `HOLD_O1_DATA_OR_POWER`: representation is coherent but natural root support
  or OR 1.50 power is inadequate;
- `KILL_O1_REPRESENTATION_PREFLIGHT`: exact predicates, restoration,
  invariance, model schema, or integrity cannot be made coherent without
  changing this charter.

If P0 is HOLD, no rollout or fit runs. The next proposal must acquire genuinely
fresh, complete normal-start O1 roots under a new, balanced multi-family
contract or move to another materially different self-learning approach.

## Operations

One heavy process at a time. Free disk hard floor is 100 GiB and target is
120 GiB. P0 is read-only apart from compact manifests and reports. Services on
ports 8765 and 8770, advisor health, dashboard record 263670, and protected
top three 263670/261369/258561 must remain healthy. Cleanup requires a reviewed
manifest.

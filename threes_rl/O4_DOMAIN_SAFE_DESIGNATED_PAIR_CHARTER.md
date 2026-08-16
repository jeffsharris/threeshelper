# O4 Domain-Safe Designated-Pair Option Charter

Date: 2026-07-27

Status: authoritative outcome-free P0 contract. O4 is a new branch, not an
O3 repair. This charter authorizes only representation tests and one
zero-game availability/provenance/power preflight. Acquisition, option
rollouts, labels, fitting, policy evaluation, and promotion remain held.

## 1. Permanent O3 Boundary

O3 is permanently `KILL_O3_TRAINING_INTEGRITY`. Every O3 acquisition,
recovery, reseal, P0, training, marker, result, episode, metadata, attempt,
stream, root, and checkpoint artifact remains byte-protected.

The immutable O3 acquisition/recovery corpus is nevertheless the O4
fresh-to-science source pool. O4 may read, for each of its 20,500 completed
normal-start roots:

- root/support-candidate identity;
- the current hard-start board and exact simulator context needed to
  recompute O4 designated-pair eligibility; and
- the bound source replay hash.

O4 excludes all 320 roots selected into O3 train, development, or untouched
mechanism roles. It may read those selected root IDs and the 1,152 reserved O3
learning stream rows only to construct exact exclusion sets.

O4 never parses an O3 option-training episode array or metadata body, never
reads an O3 option label, recorded or generated action, event, outcome,
initial/model checkpoint, or policy prediction, and never reuses an O3
selected root or learning stream. The remaining 20,180 acquisition roots are
eligible for outcome-free O4 geometry screening without becoming reused O3
science.

## 2. Scientific Question

Can one domain-safe, scale-relative, action-conditioned relational option
model learn sustained h40 control of a designated equal-valued pair toward
its pair-specific safe merge while preserving anchor and air?

Training targets remain `T={48,96,192}`. Stage occupancy is descriptive only.
Human and behavior actions are never labels. Terminal score and future
milestone/max-tile outcomes are absent from representation, root selection,
training, and checkpoint choice.

## 3. Bounded Pair Geometry

For designated coordinates `a=(r0,c0)` and `b=(r1,c1)`, define eligible
blocker cells exactly:

1. Same row: cells strictly between the two columns.
2. Same column: cells strictly between the two rows.
3. Otherwise: every cell in the inclusive axis-aligned bounding rectangle,
   excluding exactly `a` and `b`.

`blocker_capacity` is the number of eligible cells and
`blocker_occupied` is the number whose board value is nonzero.
`blocker_density` is:

```text
0                                  when blocker_capacity == 0
blocker_occupied / blocker_capacity otherwise
```

Thus adjacent same-row/same-column pairs have capacity zero and density zero.
The definition is invariant to pair-coordinate order and always lies in
`[0,1]`. Pair selection changes from O3: among eligible target pairs choose
the lexicographic minimum of

```text
(
  0 if safe_merge_actions is nonempty else 1,
  manhattan + blocker_density,
  blocker_density,
  manhattan,
  chebyshev,
  coordinates
)
```

The eight successor targets, in order, are:

1. Manhattan distance divided by 6.
2. Chebyshev distance divided by 3.
3. Blocker density above.
4. Same-row indicator.
5. Same-column indicator.
6. Empty-cell count divided by 16.
7. Legal-action count divided by 4.
8. Exact two-descendant-lineage-live indicator.

Every transform is finite and in `[0,1]`. The policy-facing blocker input uses
the same blocker density, not clamped raw count. P0 must exhaustively enumerate
all 120 coordinate pairs and every binary occupancy pattern over their
eligible cells: 43,296 pair-pattern cases. It must also property-test exact
simulator afterstates and all legal transitions on crafted/random reachable
states. Any nonfinite or out-of-domain value is
`KILL_O4_REPRESENTATION_PREFLIGHT`.

## 4. Frozen Model And Learning Contract

O4 keeps one 102,557-parameter action-conditioned model:

- 16 cell tokens x 37 features;
- linear 37->64 token projection;
- two 64-wide, four-head transformer layers, feed-forward width 128,
  dropout zero;
- mean, max, and designated-pair pooling;
- 35 global features;
- 128-wide GELU/LayerNorm hidden layer;
- 29 outputs: five event logits plus three ordered eight-value successor
  heads at h10/h20/h40.

The event, lineage, safety, masking, chosen-action-only labels, action
ordering, optimizer, batch size, loss coefficients, deterministic checkpoint
discipline, and label-support gate remain as scientifically specified for O3,
except every blocker input/target is the O4 density above.

Exactly 192 train roots produce six trajectories each, totaling 1,152:

- round 1: two uniform-legal trajectories per root;
- round 2: two trajectories per root, epsilon 0.15;
- round 3: one trajectory per root, epsilon 0.10;
- round 4: one trajectory per root, epsilon 0.05.

Five cumulative-buffer epochs follow each round. AdamW learning rate is
`3e-4`, weight decay `1e-4`, batch 128, gradient clip 1.0, CPU, PyTorch
2.12.1, and seed `2026072804`. There is one mandatory round-4 checkpoint and
no sweep, alternate objective, calibration, sign flip, or checkpoint choice.

The pre-fit support gate remains: at least 40 successes overall; at least six
for each target; at least four families with at least three; at least 40
failures; at least 40 true h40 censors; finite arrays; and at least two
nonempty success-time bins. A miss is a data-support HOLD.

## 5. Fresh Root Universe And Exact Allocation

The five immutable collector families, in canonical order, are:

1. `o4_corner2`;
2. `o4_expectimax2`;
3. `o4_parent_mc1000`;
4. `o4_replaycal`;
5. `o4_qd_v2`.

Their policy/checkpoint identities and already sealed action-distinctness
signatures must reproduce without new action evaluation. P0 considers only
the immutable 20,500-root O3 acquisition/recovery union, less the exact 320
O3-selected roots. Those roots are complete natural normal-start machine
ancestries collected before any O3 option rollout. P0 may read current
hard-start board/context geometry, support-candidate identity, ancestry
identity, family identity, and replay hash only. It may not read final/future
score, milestone, max tile, recorded or generated action, event label, option
episode metadata, checkpoint, prediction, or policy outcome.

P0 neither generates nor reserves a replacement acquisition corpus. It must
allocate all 448 roles from this unselected immutable pool or seal
`HOLD_O4_DATA_SUPPORT` with zero new games.

Exactly one hard-start state may be selected from one whole ancestry. The
selected roles are mutually ancestry-disjoint and frozen before any rollout
or label:

- train: 192;
- development: 64;
- untouched mechanism: 192.

Family marginals are fixed:

```text
                         corner2 expectimax2 parent replaycal qd_v2
train                         39          39     38        38    38
development                   13          13     13        13    12
untouched mechanism           38          38     39        38    39
combined                      90          90     90        89    89
```

Target marginals are fixed:

```text
                         T48 T96 T192
train                     64  64   64
development               22  21   21
untouched mechanism       64  64   64
```

The unique minimum-spread family x target matrices are frozen as follows,
with rows in canonical family order and columns `T48,T96,T192`:

```text
train:
13 13 13
13 13 13
12 13 13
13 12 13
13 13 12

development:
4 4 5
4 5 4
5 4 4
5 4 4
4 4 4

untouched mechanism:
12 13 13
13 12 13
13 13 13
13 13 12
13 13 13
```

P0 visits cells in role, family, target order shown above. Within a cell it
visits candidates by
`SHA256("O4-P0-cell-v1"|role|family|target|root|frame|state_hash)` and takes
the first roots not already assigned anywhere. One root qualifying multiple
targets can be used only by the first frozen cell that claims it. There is no
backtracking, substitution, target relabeling, quota shrink, or adaptive
reallocation. Exact post-allocation role/family/target/cell marginals and 448
unique roots are mandatory; any miss is `HOLD_O4_DATA_SUPPORT`.

## 6. Fresh Learning And Evaluation Streams

P0 reserves but consumes none of these namespaces:

| purpose | logical | deck | slot | policy |
|---|---:|---:|---:|---:|
| six-trajectory learning | 129B | 130B | 131B | 132B |
| option development/mechanism | 133B | 134B | 135B | 136B |
| normal-start development | 137B | 138B | 139B | 140B |
| sealed confirmation | 141B | 142B | 143B | 144B |

There is no O4 acquisition stream block. Learning code is
`root_index*6 + trajectory_index`, with trajectory-to-round map
`0,1 -> round1`, `2,3 -> round2`, `4 -> round3`, `5 -> round4`.
Option development/test uses eight paired CRN repeats/root; arms share
logical/deck/slot IDs and have unique policy IDs. Normal-start development
and confirmation remain separately sealed future manifests. All requested
IDs must have zero intersection with the complete historical union and the
explicit 1,152-row O3 exclusion set.

## 7. Mechanism Power And Downstream Gates

The untouched mechanism design remains N=192, eight paired CRN repeats/root,
target `64/64/64`, and target x alignment strata. The primary is the exact
Mantel-Haenszel common odds ratio with whole-root bootstrap. At true OR 1.50,
the frozen pass is point OR at least 1.25 and 95-percent bootstrap lower bound
above 1.00. The outcome-free simulation must reproduce at least 80-percent
power and grid MDE OR1.50 under the previously frozen beta-root/coupling
model. Family, target, alignment, and stream-block signs are descriptive.
There is no one-move gate.

Only a mechanism PASS may open a separately sealed full-policy normal-start
development block. Only its PASS may open fresh root-disjoint confirmation.
Promotion requires confirmed paired score and high-tile/max-score capability
with no safety regression. O4 P0 authorizes none of these.

## 8. P0 Decisions And Governance

P0 seals exactly one:

- `READY_O4_DOMAIN_SAFE_OPTION_PREFLIGHT`;
- `HOLD_O4_DATA_SUPPORT`;
- `KILL_O4_REPRESENTATION_PREFLIGHT`.

READY requires exact 448-root allocation, five family identities, all frozen
marginals, exhaustive domain proof, N192 power, zero root/stream collision,
projected runtime/storage within frozen caps, one heavy process, more than
100 GiB free with 120 GiB target, healthy ports 8765/8770/advisor/dashboard,
and protected top three `263670/261369/258561`.

P0 has a separate zero-work marker and immutable terminal result. It generates
zero games, consumes zero streams, creates zero labels/models/rollouts/policy
outcomes, and makes no incumbent/dashboard change. Acquisition or training
requires a new research-lead authorization regardless of READY.

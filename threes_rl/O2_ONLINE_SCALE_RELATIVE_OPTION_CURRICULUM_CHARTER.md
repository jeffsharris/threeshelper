# O2 Online Scale-Relative Option Curriculum Charter

Date: 2026-07-26

Status: outcome-free preflight only. Pilot execution, corpus generation,
option rollouts, labels, fitting, policy outcomes, capability evaluation, and
promotion are held pending a separate authorization.

## 1. Historical Locks and Question

O1 P0 remains `HOLD_O1_DATA_OR_POWER`; its exclusions and immutable artifacts
may not be relaxed or reopened. Exact depth 3 remains permanently killed.
G3 transfer, human/partial/restart/continuation/synthetic sources, behavior
actions, and human actions are forbidden training or confirmation inputs.

O2 asks whether a closed-loop, machine-goal option learner trained on
prospectively generated normal-start states can improve safe duplicate-pair
progression and ultimately full-game score.

The representation is **scale-relative**, not exactly scale-equivariant:
O1's global input includes the absolute scalar `log2(T/48)/5`.

## 2. Frozen Collectors and Unconditional Roots

Exactly four genuine collector families are used, in this order:

1. `o2_corner2`: policy spec `corner2`;
2. `o2_expectimax2`: policy spec `expectimax2`;
3. `o2_parent_mc1000`: policy spec
   `ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest`;
4. `o2_qd_v2`: policy spec
   `g1r_qd_static_archive_oneply_v2_terminal_schema:threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema/policy`.

Their immutable 64-state signatures are respectively
`4be42141...7043`, `2ad642cd...b38`, `e43dc11f...2064`, and
`66da7d61...281`; every pair has previously passed the genuine-family
distinctness rule.

Every generated root is a fresh, complete, normal-start machine game.
All roots are retained unconditionally. Score, terminal milestone, final max
tile, action sequence, or favorable geometry may not select, rank, stop, or
delete a root. Compact completion/provenance rows are retained for every
attempt and a full replay is retained for every completed root.

## 3. Yield Pilot and Finite Collection

The yield pilot is separate from training and diagnostic-only:

- exactly `128` complete roots, `32` per family;
- one worker, nice `>=10`, deterministic family round robin, chunks `<=8`;
- hard bounds: six active hours, 3 GiB output, pause below 100 GiB free,
  target at least 120 GiB free;
- no partial support peeking and no within-family early stop.

Only current-state support fields may be read after all 128 roots complete:
target value, scale band, pair stage, family, root identity, frame identity,
anchor/air validity, and legal count. Reset/root score may be read solely for
fresh-root provenance. Final/future score, terminal milestone/max tile, and
recorded actions are not read or used.

For each root and each of the 12 `(stage, scale_band)` cells, record whether
at least one O1-eligible natural state exists. A deterministic bipartite
matching assigns each root to at most one cell, visiting cells in
`stage,scale` order and roots by
`SHA256("O2-pilot-support-v1"|family|root|cell)`.

Pilot support passes only if a disjoint matching fills exactly six roots in
every cell, at least three families appear in every cell, and no family
supplies more than three roots to a cell. This includes all four late/1536
stage cells. Six of 128 has a 90-percent Wilson lower bound above the
`16/640` rate required by the final test allocator. Failure seals
`HOLD_O2_DATA_SUPPORT`; it never kills the representation.

If and only if the pilot passes, one later authorization may generate the
fixed corpus:

- `640` new complete roots, `160` per family;
- exact partitions frozen by family-local stream order before content:
  `train_candidate=0..159`; allocation after support may select at most
  `32` train, `12` development, and `48` untouched-test roots per family;
- final targets: `128` train, `48` development, `192` untouched test;
  remaining roots are retained support inventory only.

The deterministic allocator opens support only after all 640 roots complete.
It allocates untouched test first: exactly 16 roots in each stage/scale cell,
at least three families per cell, no family above six in a cell, and exactly
48 roots per family. It then allocates development: 12 roots per stage and
12 per family. It then allocates training: 32 roots per stage and 32 per
family. A root is used once. Roots within a cell/family are ordered by
`SHA256("O2-corpus-v1"|partition|cell|family|root|frame|state_hash)`.

No allocation target, family, stage, scale, or cap can change after pilot
support is read. Failure to fill the frozen corpus seals
`HOLD_O2_DATA_SUPPORT`.

## 4. Streams and Partitions

All IDs below are reserved in the no-outcome preflight and must have zero
historical intersection:

| Purpose | logical | deck | slot | policy |
|---|---:|---:|---:|---:|
| 128-root yield pilot | 81B | 82B | 83B | 84B |
| 640-root corpus | 85B | 86B | 87B | 88B |
| option learning | 89B | 90B | 91B | 92B |
| option dev/test | 93B | 94B | 95B | 96B |
| normal-start development | 97B | 98B | 99B | 100B |
| sealed confirmation | 101B | 102B | 103B | 104B |

Collector code is `family_index*roots_per_family + game_index`. Learning code
is `2_000_000 + root_index*8 + round_index*2 + replicate`. Paired evaluation
code is `3_000_000 + root_index*8 + replicate`; logical/deck/slot IDs are
shared between arms and policy ID is `base + 2*code + arm`.

Normal-start development has 384 paired roots. Confirmation has 768 paired
roots. Their complete stream manifests are frozen before the yield pilot.
Whole roots and streams never cross pilot, corpus, train, development,
untouched test, normal-start development, or confirmation.

## 5. Frozen Geometry, Labels, and Model

O2 preserves O1 A1-A3 exactly:

- schema SHA-256
  `55dd298ea2bf40a24d8af641d852d5f9c09aff14b1b736a29e6b5a071563772c`;
- action-conditioned scalar forward pass;
- `113,780` parameters;
- one 20-logit head: five event categories plus five successor classes at
  h10/h20/h40;
- pair-specific tagged merge provenance;
- at least three pre-spawn empties and two post-spawn empties;
- actual successor stage at observed checkpoints; `merged_success` only for
  a real `2*T` count increase;
- masked unobserved post-termination auxiliary checkpoints.

No success/time head is added. Success time is represented only by the frozen
event classes `1-10`, `11-20`, `21-40`, failure, and censor.

Exactly one state is selected per training root. Training uses four rounds,
two trajectories per root per round, at most 40 moves: `128*4*2=1,024`
trajectories. Round 1 uses uniform legal actions. Rounds 2-4 use epsilon
`.15/.10/.05`. Five cumulative-buffer epochs follow each round.

Loss remains root/family-equal event cross entropy plus `0.25/3` times each
included successor cross entropy. Optimizer is AdamW, learning rate `3e-4`,
weight decay `1e-4`, batch 256, gradient cap 1.0, seed `2026072602`.
The round-4 checkpoint is mandatory; development cannot choose a checkpoint
or hyperparameter.

## 6. Complete Integrated Treatment

At every game decision with no active option:

1. compute O1 eligibility;
2. if ineligible, take the frozen incumbent depth-2 action and reconsider next
   move;
3. if eligible, freeze target `T`, root count of `2*T`, requested goal
   `min(root_stage+1,4)`, and a 40-move option clock.

During an active option, recompute the deterministic best current pair at the
fixed T after every transition. For every legal action, run its own scalar
forward pass. Choose lexicographically:

1. highest predicted success probability within the remaining horizon;
2. highest predicted nonfailure probability;
3. lowest action enum.

The option terminates on frozen safe success, failure, game terminal, or 40
moves. A live post-termination state is reconsidered immediately for a new
option. Thus treatment applies on every eligible move; it is never a
one-action wrapper. Control uses the incumbent on every move.

The untouched option test requires changed actions on at least 20 percent of
roots, at least two percent of all option decisions, and at least three
families. Lower activity seals `HOLD_O2_INACTIVE`, not a policy pass.

## 7. Gates and Power

### Option mechanism

Untouched test is 192 roots, 16 per stage/scale cell, eight paired CRN
replicates, and whole-root inference. Primary is the 12-stratum
Mantel-Haenszel common OR for safe requested-stage attainment by h40.
Frozen power is 0.959 at OR 1.50; MDE is OR 1.50.

Pass requires common-OR point `>=1.50`, 95-percent root-bootstrap lower bound
`>1.00`, positive stage directions where 48 roots exist, required activity,
zero illegal actions, survival within -2 pp, mean empties within -0.5,
anchor preservation within -1 pp, and no family above 40 percent of weighted
lift. Stream-block signs are descriptive.

### Normal-start capability

Only a mechanism pass may open 384 paired normal-start development roots.
Only its preregistered pass may open 768 sealed confirmation roots.

The confirmation score primary is equal-root mean paired
`log1p(max(score_minus_starter,0))`. Under frozen paired SD 0.90, N=768 has
80-percent MDE `exp((1.959964+0.841621)*0.90/sqrt(768))-1 = 9.53%`.
Raw paired score, 10-percent winsorized mean, median, lower decile, P90/P95,
and maximum are mandatory reports.

The milestone co-primary is the preregistered common OR for first non-starter
1536 across eight fixed stream blocks, powered at OR 1.50 under the
outcome-free beta-root model in the preflight. P3072 is a safeguard.

Confirmation promotion requires:

- score-primary 95-percent root-bootstrap lower bound above zero;
- milestone common-OR lower bound above 1.00;
- P3072 difference at least -2 pp;
- no material survival, anchor, lower-decile, or corner regression;
- confirmed treatment maximum above both the paired control maximum and the
  protected record 263670, with a provenance-valid replay;
- no integrity or concentration failure.

Acquisition games are never capability evidence. Only confirmation can update
the incumbent or dashboard.

## 8. Resource and Decision Contract

Historical metadata gives 9.20 seconds per complete acquisition game.
Pilot plus corpus is 768 games; at 2.5x safety the projection is 4.91 active
hours. Using the immutable maximum replay size 1,000,401 bytes and 1 MiB
summary allowance, the conservative 1.25x projection plus 512 MiB fixed
overhead is below 3 GiB.

One heavy process, one worker, nice `>=10`, six active hours, 3 GiB, and
100/120 GiB hard/target free-space rules are frozen for acquisition.
Training/mechanism E0, if later authorized, has separate hard limits of eight
active hours and 3 GiB.

Outcome-free preflight decisions:

- `READY_O2_YIELD_PILOT_PREFLIGHT`;
- `HOLD_O2_COST_OR_POWER`;
- `KILL_O2_PREFLIGHT_INTEGRITY`.

Pilot decisions, if later authorized:

- `READY_O2_CORPUS_COLLECTION`;
- `HOLD_O2_DATA_SUPPORT`;
- operational integrity HOLD.

Adequately powered model or policy failure kills the exact O2 configuration.
Support/cost scarcity is always HOLD. `PROMOTE=false` before confirmation.

# O5 Four-Family Domain-Safe P0 Charter

Date: 2026-07-27

Status: authoritative outcome-free P0 contract. O5 is a new four-family
source-feasibility branch. It is not an O4 rerun or repair. This charter
authorizes implementation/tests, one zero-work marker, and one read-only
current-state geometry/support/provenance/power preflight. It does not
authorize acquisition, labels, stream consumption, option rollouts, fitting,
policy outcomes, normal-start evaluation, incumbent changes, dashboard
changes, or promotion.

## 1. Permanent O3 And O4 Boundaries

O3 remains permanently killed. O5 may use the immutable 20,500-root O3
acquisition/recovery union only as a completed natural normal-start source
universe. It excludes all exact 320 roots selected into O3 science and may
read only:

- root, family, support-candidate, frame, and replay identity;
- the current support frame's board, visible preview, tile-cycle context,
  move count, and game-over bit; and
- immutable replay bytes solely for hash verification and that whitelisted
  current-state restoration.

O5 never parses any O3 option-training episode, metadata, label, action,
outcome, checkpoint, prediction, or selected-root replay body. It never reads
final/future score, final/future max tile, milestone, recorded move/action,
legal-action cache, or policy outcome fields.

O4 V1 and V2 code, tests, evidence, markers, audits, reservations, terminal
result, and documentation remain byte-protected. O4's exact five-family P0
is killed because every qualifying QD support root was already in the
protected O3-selected set, leaving zero untouched QD roots. O5 excludes QD
rather than relabeling or substituting it. O5 may import only
`threes_rl/o4_domain_safe_pair_option.py` at exact SHA-256
`95a4da48fb7550e87b09e1f1594cdbdc062a52c7df544b7445b5e58878c87f41`
and `threes_rl/o4_power_contract.py` at exact SHA-256
`16e2c26c9e1f2b176937f1a0546604b878d45875b4c29dbc83a441588f7fc5cd`
as immutable outcome-free operators. O4 stream IDs 129B through 144B are
spent reservations and must collide with no O5 stream.

## 2. Scientific Question

Can the exhaustively domain-safe designated-pair representation be supported
by a fresh-to-science, ancestry-disjoint four-family natural corpus while
retaining the already frozen sustained-policy N=192 mechanism design?

This P0 tests only representation exactness, source support, provenance,
allocation feasibility, stream availability, and prospective power. It does
not test learning, policy utility, score, or capability.

## 3. Immutable Representation

O5 reuses the exact O4 designated-pair feature/operator implementation
without source modification:

- 16 cell tokens x 37 inputs;
- 35 global inputs;
- action-conditioned 29-output dual head;
- exact 102,557 parameters;
- blocker density defined as occupied eligible cells divided by possible
  eligible cells, with adjacent/no-interior capacity zero mapped to zero;
- exact pair-selection first key
  `0 if safe_merge_actions is nonempty else 1`;
- eight successor transforms finite and bounded in `[0,1]`.

P0 must reproduce schema SHA-256
`60a83881d8e8275a4aa2d03df06815d65e5b247b16f36118009f42f2ce3098ba`,
all 120 coordinate pairs, all 43,296 relevant occupancy cases, density range
`[0,1]`, and parameter count 102,557. Focused tests must retain whitelist
sentinels and a bounded deterministic random-reachable exact-transition
property test with nonzero eligible-state and legal-transition counts.
No feature, architecture, objective, optimizer, sign, or calibration change
is permitted.

## 4. Genuine Families

The four families, in semantic order, are:

1. `o5_corner2`;
2. `o5_expectimax2`;
3. `o5_parent_mc1000`;
4. `o5_replaycal`.

They map one-to-one to the first four exact O3 family identities. P0 binds the
accepted 64-state signature and pairwise-distinctness audit without new
action evaluation or retiming. Semantic order is reconstructed from the
explicit prior `family_order` list and compared to the literal O5 tuple;
dictionary insertion order is never a gate. All six retained pairwise gates
must pass. Family signs in any later outcome analysis are descriptive, not
conjunctive.

## 5. Exact Root Allocation

Targets are `(48,96,192)`. Roles are `(train, development,
untouched_mechanism)`. Exactly one support state from one whole ancestry may
be assigned to exactly one role and target.

Role totals and family totals are:

```text
                         corner2 expectimax2 parent replaycal total
train                          48          48     48        48   192
development                    16          16     16        16    64
untouched mechanism            48          48     48        48   192
combined                      112         112    112       112   448
```

Every family therefore has exactly 25% of every role and of the combined
corpus.

The exact family x target matrices, with columns `T48,T96,T192`, are:

```text
train:
16 16 16
16 16 16
16 16 16
16 16 16

development:
6 5 5
5 6 5
5 5 6
6 5 5

untouched mechanism:
16 16 16
16 16 16
16 16 16
16 16 16
```

Role target marginals are train `64/64/64`, development `22/21/21`, and
untouched mechanism `64/64/64`. Combined target totals are `150/149/149`.

P0 visits cells in literal role, family, target order. Within each cell,
candidates are ordered by ascending

```text
SHA256(
  "O5-P0-cell-v1"|role|family|target|root|frame_index|state_hash
)
```

and the first still-unassigned roots are taken. There is no backtracking,
substitution, role reassignment, target relabeling, quota shrink, or
post-content rebalancing. Exact post-allocation matrices, 448 unique
ancestries, and zero overlap with the protected 320 are mandatory. A support
miss is `HOLD_O5_FOUR_FAMILY_DATA_SUPPORT`, never representation evidence.

## 6. Fresh Reserved Streams

P0 reserves and consumes zero rows from four fresh quartets:

| purpose | logical | deck | slot | policy |
|---|---:|---:|---:|---:|
| six-trajectory learning | 181B | 182B | 183B | 184B |
| option development/mechanism | 185B | 186B | 187B | 188B |
| normal-start development | 189B | 190B | 191B | 192B |
| sealed confirmation | 193B | 194B | 195B | 196B |

The exact row contract is:

- 1,152 learning rows: `192 roots * 6 trajectories`;
- 512 paired option-development rows: `64 roots * 8 repeats`;
- 1,536 paired untouched-mechanism rows: `192 roots * 8 repeats`;
- 512 paired normal-development rows;
- 2,560 paired confirmation rows;
- 6,272 rows total.

Learning code is `root_index*6 + trajectory_index`. Option development uses
codes 0 through 511. Untouched mechanism uses offset 1,000,000. Normal
development and confirmation each start at code zero in their separate
quartets. Paired arms share logical/deck/slot IDs and receive distinct policy
IDs `base+2*code` and `base+2*code+1`.

Before marker, only the deterministic requested-row hash may be computed.
After marker, collision audit must prove zero intersection with the complete
historical union, the exact 1,152 O3 learning rows, and every O4 V1/V2
reservation. O4 reservations are spent even though no O4 stream was
consumed. No stream may be consumed in P0.

## 7. Sustained-Policy Power

The untouched mechanism contract is unchanged scientifically:

- N=192 roots, target-balanced 64/64/64;
- eight paired common-random-number repeats per arm/root;
- sustained learned option policy versus sustained frozen-incumbent policy
  through h40;
- target x starting-alignment strata;
- Mantel-Haenszel common odds ratio primary;
- whole-ancestry bootstrap;
- at true OR1.50, pass point OR at least 1.25 and bootstrap lower bound above
  1.00;
- family, target, alignment, and stream-block signs descriptive.

The frozen outcome-free simulation must reproduce OR1.50 full-gate power at
least 80% and the same MDE grid. There is no one-move utility gate.

## 8. P0 Orchestration

Fresh files and identities are:

- charter: `threes_rl/O5_FOUR_FAMILY_DOMAIN_SAFE_P0_CHARTER.md`;
- runner: `threes_rl/o5_four_family_p0.py`;
- focused tests: `tests/test_rl_o5_four_family_p0.py`;
- test evidence:
  `threes_rl/runs/forensics/o5_four_family_domain_safe_p0_test_evidence_v1.json`;
- output:
  `threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1`;
- marker: `O5_P0_OPENED.json`;
- result: `O5_P0_RESULT.json`.

Tests and immutable test evidence must exist before `open`. `open` validates
all bound source/code/test hashes, zero prior O5 work, deterministic matrices
and stream rows, no heavy contender, nice at least 10, disk above 120 GiB
target and 100 GiB hard floor, services, dashboard record, and protected top
three. It writes exactly one immutable zero-work marker and exits before any
source-content scan.

`run` requires that exact marker, revalidates all bindings, and executes the
single permitted P0 scan once. It must seal one immutable terminal result and
may never rerun after a result exists.

Projected downstream active runtime remains 18 hours, projected incremental
storage 2.75 GiB, and hard storage cap 4 GiB. P0 itself is lightweight and
uses one nice>=10 process.

## 9. Terminal Decisions

Exactly one terminal decision is allowed:

1. `READY_O5_FOUR_FAMILY_DOMAIN_SAFE_PREFLIGHT` when every integrity,
   representation, exact-allocation, stream, power, process, disk, and
   service gate passes.
2. `HOLD_O5_FOUR_FAMILY_DATA_SUPPORT` when immutable source support cannot
   fill the exact matrix or an operational support bound fails without
   scientific corruption.
3. `KILL_O5_FOUR_FAMILY_INTEGRITY_OR_REPRESENTATION` when immutable
   identity, provenance, whitelist, domain, schema, policy-signature,
   collision, or representation integrity fails.

READY authorizes no training. A separately frozen execution charter and
research-lead authorization remain mandatory. P0 always records zero games,
consumed streams, labels, models, rollouts, policy outcomes, score/action
inspection, incumbent changes, and dashboard changes.

# O2 Online Scale-Relative Option Curriculum Charter Amendment A2

Date: 2026-07-26

Status: authoritative outcome-free amendment. It binds:

- base charter SHA-256
  `865f44c526d1859899b532e87dc4b99d031aa487705845b2c5412523b1997e12`;
- A1 SHA-256
  `79423c0fdce09cdb3b69b4e3bed4ddd5ec4d2dc7051cd2b3567f92c48e5e40af`.

Where this amendment conflicts with either prior document, A2 controls.
Pilot execution, corpus generation, labels, fitting, policy outcomes, and
promotion remain held.

## A1. Exact-Target Support Is Part Of Readiness

Scale bands remain useful allocator strata, but they may not stand in for
exact target levels. Every selected state has one exact frozen target
`T in {48,96,192,384,768,1536}` in addition to its starting stage and band.

The 640-root support-only allocator must satisfy all earlier A1 role, family,
stage, and band requirements plus these exact-target constraints.

### Untouched mechanism test: 192 roots

For each starting stage:

- the 16-root early cell contains at least four roots at each of
  `T=48,96,192`; the remaining at most four may be any early target;
- the 16-root mid cell contains exactly eight `T=384` roots and eight
  `T=768` roots;
- the 16-root late cell contains 16 `T=1536` roots.

Thus untouched test contains at least 32 independent `T=768` roots, exactly
eight in every starting stage. A mid-band mechanism result may not be called
upward transfer unless the separately reported `T=768` estimate is
directionally nonnegative. Its sign is descriptive because 32 roots are not
assumed powered.

### Development: 48 roots

For each starting stage, the 12 selected roots must include at least:

- one root for each of `T=48,96,192,384`;
- four roots at `T=768`;
- two roots at `T=1536`.

The remaining two roots per stage are allocated by the existing deterministic
support hash without outcome access. Development therefore contains at least
16 `T=768` roots, four per starting stage.

### Training: 128 roots

Across training, require at least:

- 16 roots at each of `T=48,96,192,384`;
- 24 roots at `T=768`;
- 16 roots at `T=1536`.

Additionally, each starting stage has at least four `T=768` and four
`T=1536` roots. No exact target supplies more than 40 percent of training
roots. Family/root balancing from A1 remains unchanged.

The allocator is deterministic and fail-closed. It visits exact-target
requirements in the order written above, roots by the existing A1
partition/state hash, then fills remaining band/stage/family quotas. It may
not relabel a target or use one root twice. Any exact-target, stage, family, or
band shortfall is `HOLD_O2_DATA_SUPPORT`, never a representation kill.

The 128-root yield pilot reports distinct whole-root support counts by
`family x starting_stage x exact_target`. In addition to the A1 84-root
band-cell matching, pilot readiness requires:

- at least seven support roots and at least three families for each of
  `T=48,96,192,384` overall;
- at least seven `T=768` support roots in every starting stage, with at least
  three families represented in each stage;
- the existing seven-root late-cell match, which already implies seven
  `T=1536` roots in every starting stage.

Pilot support may be overlapping for this availability report, but the A1
84-root band-cell match and every final corpus role remain one-root/one-state.

## A2. Normal-Start Activity Has No Collector-Family Label

Collector-family coverage remains required for the support corpus and the
option mechanism test, whose roots have immutable collector provenance.

Fresh paired normal-start development and confirmation roots have no collector
family identity. Their activity gate is instead:

- at least 20 percent of treatment roots invoke an option that changes one or
  more actions relative to the paired incumbent arm;
- changed option actions are at least two percent of all treatment option
  decisions;
- at least one treatment option decision occurs at every starting stage
  `0,1,2,3`;
- activity spans at least four exact target levels and must include `T=768`.

All activity is measured from the integrated treatment policy after the
checkpoint is frozen. No family label is inferred from a normal-start stream.

## A3. Authoritative Historical Calibration Paths

The preflight must bind and hash these exact paths, pair rows by
`(block,index,logical_seed,deck_stream_id,slot_stream_id)`, and reproduce all
A1 aggregates. A hash-only role label or copied constant is insufficient.

| role | exact path | SHA-256 |
|---|---|---|
| D0 incumbent | `threes_rl/runs/eval_artifacts/r1_baseline_incumbent_split_v1_d0_20260709/results.csv` | `240fca445d79b6f546e9dab5b62bfa4ae9531d1d31cc727c59bc04495376f4bb` |
| D0 comparator | `threes_rl/runs/eval_artifacts/r1_candidate_1000_split_v1_d0_20260709/results.csv` | `4c4db06b457b44155e11e5dd5476980f010f0fd65b9c545f561598bfd34a5d48` |
| D1 incumbent | `threes_rl/runs/eval_artifacts/r1_baseline_incumbent_split_v1_d1_20260709/results.csv` | `0226d60858f052597399c6ba3ed804cedc4769e09a393df201ef4fcdc6d17491` |
| D1 comparator | `threes_rl/runs/eval_artifacts/r1b_candidate_1000_split_v1_d1_20260709/results.csv` | `88b4b0151c6280dca30de4ed74ac4e2d22840b17565c02d66e16fe54dfcbd3d0` |
| D2 incumbent | `threes_rl/runs/eval_artifacts/r1b_baseline_incumbent_split_v1_d2_20260709/results.csv` | `d04a68d8f486f1b4c965e220427b4f753a828ee40ad4bdf746bb73893e161f8f` |
| D2 comparator | `threes_rl/runs/eval_artifacts/r1b_candidate_5000_split_v1_d2_20260709/results.csv` | `a81bfa2484218c09626ab65f303bef6f030cfe4b6e87995157eae700cc2652d8` |
| C incumbent | `threes_rl/runs/eval_artifacts/r1b_confirmation_incumbent_c_20260710/results.csv` | `474356da0bc5847acbba6723fcdc2e477c64d9894c75a47d5bd10f673a285383` |
| C comparator | `threes_rl/runs/eval_artifacts/r1b_confirmation_candidate_5000_c_20260710/results.csv` | `9c34ce8daa3926cff294bed0b094ced5421f458cdc5dfdc6ac16125924c1e564` |

The machine-readable calibration artifact must reproduce:

- D0-D2 pair count 768;
- D0-D2 incumbent/comparator/both P3072 counts `29/40/2`;
- D0-D2 shared-uniform coupling `0.017809776430466086`;
- D0-D2 paired log1p-difference SD `1.1167440698964322`;
- C pair count 512;
- C incumbent/comparator/both P3072 counts `21/21/3`;
- C gains/losses `18/18`;
- C shared-uniform coupling `0.10619726505673553`;
- C paired log1p-difference mean `0.02383` within absolute tolerance `5e-6`;
- C paired log1p-difference SD `1.18043` within absolute tolerance `5e-6`.

These are already-spent calibration data. They do not become O2 evidence, do
not select a model, and do not reopen any old promotion decision.

## A4. Confirmation Power Is A READY Gate

The O2 preflight must explicitly report the N2560 P3072 common-OR power table
for every frozen base-rate vector and coupling sensitivity. The scalar
`worst_case_or_1_50_power_at_n2560` is the minimum table value.

`READY_O2_YIELD_PILOT_PREFLIGHT` requires
`worst_case_or_1_50_power_at_n2560 >= 0.80`. A missing row, failed aggregate
reproduction, or lower power makes READY impossible:

- aggregate/hash/integrity failure:
  `KILL_O2_PREFLIGHT_INTEGRITY`;
- correctly computed but underpowered N2560 design:
  `HOLD_O2_COST_OR_POWER`.

No score-power calculation can satisfy this milestone requirement.

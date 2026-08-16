# O2 Online Scale-Relative Option Curriculum Charter Amendment A1

Date: 2026-07-26

Status: authoritative outcome-free amendment. This amendment binds the base
charter at SHA-256
`865f44c526d1859899b532e87dc4b99d031aa487705845b2c5412523b1997e12`.
Where this amendment conflicts with the base charter, this amendment controls.
Pilot execution, corpus generation, labels, fitting, policy outcomes, and
promotion remain held.

## A1. Exact Scale Bands And Cells

The representation remains scale-relative, not exactly scale-equivariant.
The three scale bands are literal target-tile sets:

- `early = {48, 96, 192}`;
- `mid = {384, 768}`;
- `late = {1536}`.

The four starting stages are the frozen O1 A3 geometry stages `0,1,2,3`.
The 12 cells are the Cartesian product ordered by stage first and then
`early,mid,late`.

## A2. Yield Pilot Correction

The pilot remains exactly 128 complete unconditional normal-start roots, 32
per collector family. Its deterministic one-root/one-cell matching must fill
**seven**, not six, roots in every cell:

- 84 matched roots total;
- at least three families in every cell;
- no family above three roots in a cell;
- each root used by at most one cell.

The bound is the lower endpoint of the two-sided 90-percent Wilson interval,
using `z=1.6448536269514722`. At `k=6,n=128`, the lower endpoint is
`0.024430227943622398`, which is below the required final rate `16/640=0.025`.
At `k=7,n=128`, it is `0.02991899584928971`, which clears the requirement.
The smallest passing integer is therefore seven.

Failure to obtain the 84-root matching is `HOLD_O2_DATA_SUPPORT`; it is not
representation evidence. The pilot is availability evidence only and can
never be called policy improvement.

## A3. Prospective Corpus And Partition Semantics

The later corpus, if separately authorized, is a prospective universe of 640
complete unconditional roots, exactly 160 per family. Before any game, each
root has immutable `(family_index, family_game_index, stream IDs)`. No root is
a train, development, or test root at generation time.

After all 640 roots complete, the allocator may open only current-state support
metadata and exact-state hashes. It allocates disjoint roles in this order:

1. untouched test: 192 roots, exactly 16 per cell and 48 per family, at least
   three families per cell, no family above six roots per cell;
2. development: 48 roots, exactly 12 per starting stage and 12 per family;
3. training: 128 roots, exactly 32 per starting stage and 32 per family;
4. support inventory: all remaining roots.

Opening support metadata for a future test root is not opening a test
trajectory, option label, action, model prediction, or policy outcome. No such
test information may be opened until the mandatory round-4 checkpoint and its
hash are sealed. Confirmation roots are not part of the 640-root corpus and
remain wholly unopened until a development pass.

Within an allocated root and its assigned cell, the one selected state is

`argmin SHA256("O2-state-v1"|partition|family|root|stage|scale_band|frame|state_hash)`.

Only exact O1-A3-eligible states in that assigned cell enter the argmin. Ties
are resolved by lower frame and then lexical `state_hash`. A root contributes
at most one selected state across all roles.

## A4. Exact 20-Logit Contract And Action Rule

One scalar action-conditioned forward pass emits exactly 20 logits:

- logits `0..4`: event categories `success_1_10`, `success_11_20`,
  `success_21_40`, `failure`, `censor`;
- logits `5..9`: five actual successor geometry classes at h10;
- logits `10..14`: the same five classes at h20;
- logits `15..19`: the same five classes at h40.

This is a `5 + 15` grouping, not 20 unrelated classes. There is no separate
success or time head.

For event softmax probabilities `(p1,p2,p3,pf,pc)` and integer remaining
option horizon `r in [1,40]`, define

`P_success(r) = p1*min(r,10)/10
              + p2*clip(r-10,0,10)/10
              + p3*clip(r-20,0,20)/20`.

Define `P_nonfailure = 1-pf`; censoring is nonfailure, not success. Illegal
actions are masked before scoring. Compare `P_success(r)` first and
`P_nonfailure` second in float64. Values whose absolute difference is at most
`1e-12` tie at that level. Final ties use the lowest action enum. No random
tie break enters integrated treatment.

The complete treatment remains closed-loop on every move: deterministic O1-A3
eligibility, pair and target selection, fixed target `T`, goal
`min(root_stage+1,4)`, 40-move clock, action-conditioned scoring on every
active move, frozen success/failure/terminal/horizon termination, and immediate
reconsideration after a live termination. Outside an active eligible option,
the treatment arm uses the frozen incumbent and rechecks eligibility next
move. The control arm uses the frozen incumbent for every move.

## A5. O2-Specific Mechanism Power

The O1 power table is not reused as evidence. The O2 preflight must run and
seal a fresh simulation for the exact O2 full-option endpoint:

- safe requested-stage attainment by h40;
- 12 exact `(starting_stage,scale_band)` strata;
- 192 roots, 16 per stratum;
- eight CRN replicates per arm/root;
- whole-root, strata-standardized Mantel-Haenszel common OR;
- root-stratified bootstrap confidence interval;
- stream-block signs descriptive only.

The prospective base-probability design is explicitly an outcome-free design
assumption, not an observed O2 rate: root probabilities are drawn from
`Beta(1.6,18.4)`, multiplied by stage factors
`{0:0.50,1:0.75,2:1.00,3:1.50}` and scale factors
`{early:1.25,mid:1.00,late:0.75}`, then clipped to `[0.002,0.80]`. Treatment
applies an OR shift. CRN shared-uniform coupling is `0.50`.

The preflight must report power at OR `1.25,1.50,1.75,2.00` and the smallest
grid MDE with at least 80-percent power. It may not declare the mechanism
design powered until that new artifact exists. Failure at the fixed 192-root
design is `HOLD_O2_COST_OR_POWER`.

The mechanism gate uses the pooled common OR and root-bootstrap interval.
Starting-stage and stream-block directions are descriptive. A stage may block
only for clear material harm: its 95-percent root-bootstrap upper confidence
bound is below OR `0.80`. There is no every-stage-positive conjunction.

## A6. Historical Capability Calibration

O2 normal-start capability uses a different power model from the option
mechanism. The following already-open historical development files are bound
only for outcome-free base-rate, pairing, and variance calibration:

| role | file SHA-256 |
|---|---|
| D0 incumbent | `240fca445d79b6f546e9dab5b62bfa4ae9531d1d31cc727c59bc04495376f4bb` |
| D0 comparator | `4c4db06b457b44155e11e5dd5476980f010f0fd65b9c545f561598bfd34a5d48` |
| D1 incumbent | `0226d60858f052597399c6ba3ed804cedc4769e09a393df201ef4fcdc6d17491` |
| D1 comparator | `88b4b0151c6280dca30de4ed74ac4e2d22840b17565c02d66e16fe54dfcbd3d0` |
| D2 incumbent | `d04a68d8f486f1b4c965e220427b4f753a828ee40ad4bdf746bb73893e161f8f` |
| D2 comparator | `a81bfa2484218c09626ab65f303bef6f030cfe4b6e87995157eae700cc2652d8` |

Across these 768 paired roots, the frozen aggregate is:

- incumbent P3072 `29/768 = 0.037760416666666664`;
- comparator P3072 `40/768 = 0.052083333333333336`;
- both arms P3072 `2/768`;
- mixture shared-uniform coupling estimate `0.017809776430466086`;
- paired log1p score-difference SD `1.1167440698964322`.

The spent R1b confirmation files are bound for a conservative variance and
pairing sensitivity check only, not effectiveness evidence or candidate
selection:

- incumbent file SHA-256
  `474356da0bc5847acbba6723fcdc2e477c64d9894c75a47d5bd10f673a285383`;
- comparator file SHA-256
  `9c34ce8daa3926cff294bed0b094ced5421f458cdc5dfdc6ac16125924c1e564`;
- supplied aggregate paired log1p SD `1.18043`;
- incumbent P3072 `21/512 = 0.041015625`;
- both arms P3072 `21/512`, with `18` gains and `18` losses, implying three
  both-success roots and shared-uniform coupling `0.10619726505673553`.

No C row, outcome, action, or score is reopened by O2. The preflight verifies
file hashes and uses only these frozen aggregates. The conservative score SD
is the maximum, `1.18043`. The conservative milestone power is the minimum
over control base-rate vectors from D0-D2 and C and couplings
`{0,0.017809776430466086,0.10619726505673553}`.

## A7. Normal-Start Endpoints, Power, And Gates

Normal-start development is exactly 384 paired roots. Confirmation is exactly
2,560 fresh paired roots. Both use eight equal stream strata, shared
logical/deck/slot tapes, arm-unique policy streams, and whole-root inference.

Score primary is mean paired
`log1p(max(score_minus_starter,0))`. With SD `1.18043`, normal-approximation
80-percent MDEs are:

- development N384: `18.3840%`;
- confirmation N2560: `6.7545%`.

These calculations concern the score primary only. They are not tail,
maximum-score, P3072, or P6144 power claims.

The capability milestone co-primary is non-starter P3072, analyzed as an
eight-stream-stratum common OR with a root-stratified bootstrap. The preflight
must simulate the exact paired binary endpoint under the frozen historical
base-rate vectors and coupling sensitivity above, report power at OR
`1.25,1.50,1.75,2.00`, and report the 80-percent MDE at N384 and N2560.
Confirmation is coherent only if worst-case OR1.50 power is at least 80
percent at N2560. Otherwise preflight seals `HOLD_O2_COST_OR_POWER`. P6144 and
maximum score remain mandatory tail reports but are not called powered.

Development passes and may open confirmation only when all hold:

- mean paired log1p score is positive and its 95-percent root-bootstrap lower
  bound is above zero;
- P3072 difference is at least `-0.02` and P3072 common-OR point is at least
  `1.00`;
- option activity is at least 20 percent of roots and two percent of option
  decisions across at least three collector families;
- zero illegal actions;
- survival difference at least `-0.02`;
- mean-empty difference at least `-0.5`;
- anchor-preservation difference at least `-0.01`;
- no material lower-decile or frozen corner/catastrophic-tail safeguard.

Confirmation promotion requires all development safeguards plus:

- score-primary 95-percent root-bootstrap lower bound above zero;
- P3072 common-OR point at least `1.50` and lower bound above `1.00`;
- treatment maximum above the paired control maximum and protected record
  `263670`, with a provenance-valid normal-start replay;
- mandatory raw score, winsorized mean, median, lower decile, P90, P95, P99,
  P3072, P6144, maximum, survival, air, anchor, and corner reporting.

A continuous score result never substitutes for the P3072/tail gate.
Acquisition and option-state mechanism results never count as normal-start
capability.

## A8. Exact Stream Row Counts And CRN Semantics

All stream namespaces in the base charter remain fixed. The no-outcome
preflight must materialize and verify:

| purpose | roots/tapes | arm rows |
|---|---:|---:|
| yield pilot | 128 | 128 |
| prospective corpus | 640 | 640 |
| option learning | `128*4*2 = 1,024` | 1,024 |
| option development | `48*8 = 384` | 768 |
| untouched mechanism test | `192*8 = 1,536` | 3,072 |
| normal-start development | 384 | 768 |
| sealed confirmation | 2,560 | 5,120 |

Trajectory codes are exact:

- yield pilot: `family_index*32 + family_game_index`;
- corpus: `family_index*160 + family_game_index`;
- learning: `2_000_000 + root_index*8 + round_index*2 + replicate`;
- option development: `3_000_000 + root_index*8 + replicate`;
- untouched mechanism test: `4_000_000 + root_index*8 + replicate`;
- normal-start development: `5_000_000 + root_index`;
- sealed confirmation: `6_000_000 + root_index`.

Within each purpose, logical/deck/slot IDs are `purpose_base + code`. A
single-arm policy ID is `policy_base + code`; a paired policy ID is
`policy_base + 2*code + arm`, where control is arm zero and treatment is arm
one.

For every paired tape, logical/deck/slot IDs are exactly equal between arms
and policy IDs are distinct. Across distinct trajectory codes, all four IDs
are unique. Learning and collector rows have one arm. Every namespace must
have zero intersection with the complete historical collision union before
any execution.

## A9. Finite Cost And Storage Gates

The immutable historical complete-game rate is
`17655.126695 / 1920 = 9.195378486979167` seconds/game. The immutable h40
path-rate proxy is bound to G3-v2 preflight file SHA-256
`052985f7e5c13797df43bfd074602169ff5c85618dd0f3db549720fec95f7d66`
and equals `42820.82137607305 / 5072 = 8.442590965314087`
seconds/path. Projections apply a 2.5x safety factor to option paths and a
conservative `1.0x incumbent + 2.5x treatment` factor to paired full games.

| phase | projected active time | hard active-time cap | incremental storage cap |
|---|---:|---:|---:|
| pilot + corpus, 768 games | 4.905 h | 6 h | 3 GiB |
| option learning, 1,024 paths | 6.004 h | 8 h | 3 GiB |
| option development, 768 paths | 4.503 h | 6 h | 2 GiB |
| mechanism test, 3,072 paths | 18.011 h | 24 h | 3 GiB |
| normal-start development, 384 pairs | 3.433 h | 8 h | 3 GiB |
| confirmation, 2,560 pairs | 22.887 h | 30 h | 4 GiB |

The acquisition storage projection remains
`ceil(1.25*768*(1,000,401 + 1 MiB) + 512 MiB) =
2,503,888,832 bytes = 2.332 GiB`.

Every phase remains separately authorized, one worker/process at nice at
least 10, with 100 GiB hard and 120 GiB target free-space bounds. The
outcome-free preflight must fail `HOLD_O2_COST_OR_POWER` if any projection
exceeds its frozen cap.

## A10. Preflight Envelope

The no-game O2 preflight binds the base charter and this amendment, all source
hashes and frozen aggregates above, O1 A1-A3 schema, collector identities,
stream and partition manifests, exact power simulation code/seeds/repeats,
cost formulas, service/storage checks, and zero-work attestations.

Its only decisions remain:

- `READY_O2_YIELD_PILOT_PREFLIGHT`;
- `HOLD_O2_COST_OR_POWER`;
- `KILL_O2_PREFLIGHT_INTEGRITY`.

READY authorizes no pilot by itself. `PROMOTE=false`.

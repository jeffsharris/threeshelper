# O2 Online Scale-Relative Option Curriculum Charter Amendment A3

Date: 2026-07-26

Status: authoritative outcome-free scientific correction. It binds:

- base charter SHA-256
  `865f44c526d1859899b532e87dc4b99d031aa487705845b2c5412523b1997e12`;
- A1 SHA-256
  `79423c0fdce09cdb3b69b4e3bed4ddd5ec4d2dc7051cd2b3567f92c48e5e40af`;
- A2 SHA-256
  `610e564815f6c51244d474034a3f58865e30fefb3703bbc3e062b276984bb9fb`.

A3 supersedes A1/A2 wherever they require 768 or 1536 training, powered 1536
mechanism cells, or the old 12-band-cell mechanism design. The historical
calibration paths, stream rules, model, full-policy semantics, normal-start
N2560 capability gate, resource locks, and integrity requirements remain in
force. No pilot, rollout, label, fit, or policy outcome is authorized.

## A1. Scientific Transfer Contract

O2 is lower-to-higher scale transfer:

- training targets are exactly `T={48,96,192,384}`;
- `T=768` is development and untouched mechanism transfer only;
- `T=1536` is sealed descriptive transfer availability and full-game
  capability evidence only.

No `T=768` or `T=1536` trajectory may enter training, standardization,
checkpoint selection, thresholding, or optimizer state. No 1536 mechanism
sample count is required to train or test lower-to-768 transfer.

## A2. Exact 20-Cell Pilot

The 128-root yield pilot remains 32 unconditional complete normal-start roots
per collector family. It audits 20 exact support cells:

- 16 lower cells:
  `(starting_stage,T)` for stages `0,1,2,3` and
  `T in {48,96,192,384}`;
- four transfer cells:
  `(starting_stage,T=768)` for stages `0,1,2,3`.

The pilot performs one deterministic root-disjoint support matching:

- four roots in every lower cell, at least three families, no family above
  two roots in a cell;
- seven roots in every 768 cell, at least three families, no family above
  three roots in a cell;
- 92 matched roots total, each root used once.

Root order within a cell is
`SHA256("O2-pilot-exact-v1"|stage|T|family|root|frame|state_hash)`.
Cells are visited first by `T=768,384,192,96,48`, then by stage `0,1,2,3`;
this conservative order gives the scarce transfer cells first claim while
using no outcome.

The two-sided 90-percent Wilson lower endpoint at `k=4,n=128` is
`0.014105329342481491`, above the lower-cell final need `8/640=0.0125`.
At `k=7,n=128` it is `0.02991899584928971`, above the transfer final need
`16/640=0.025`.

The pilot separately reports any natural `T=1536` support by stage/family.
It has no 1536 minimum and does not allocate those roots. Zero 1536 support
does not block the pilot or lower-to-768 program.

Any failure of the 92-root exact matching is `HOLD_O2_DATA_SUPPORT`. The pilot
is availability evidence only.

## A3. Prospective 640-Root Allocation

The 640 unconditional roots remain a role-free prospective universe until
all games complete and support-only allocation opens. The final disjoint roles
are:

### Training: 128 roots

- exactly eight roots in each of the 16 lower `(stage,T)` cells;
- exactly 32 roots at each `T=48,96,192,384`;
- exactly 32 roots per collector family;
- at least three families in every cell and no family above three roots in a
  cell;
- zero `T=768` and zero `T=1536`.

### Development: 48 roots

- exactly two roots in each of the 16 lower cells: 32 lower roots;
- exactly four roots in each of the four `T=768` stage cells: 16 transfer
  roots;
- exactly 12 roots per collector family overall;
- each 768 cell contains at least three families and no family above two
  roots;
- zero `T=1536`.

Development cannot select a checkpoint, feature, threshold, or hyperparameter.
The round-4 checkpoint remains mandatory and is hash-sealed before any
development trajectory opens.

### Untouched mechanism test: 192 roots

- exactly eight roots in each of the 16 lower cells: 128 lower roots;
- exactly 16 roots in each of the four `T=768` stage cells: 64 transfer
  roots;
- exactly 48 roots per collector family overall;
- at least three families in every cell;
- no family above three roots in a lower cell or six roots in a 768 cell;
- zero required `T=1536`.

The allocator visits untouched test, development, and training in that order.
Within each requirement it uses the A1 state argmin and the A3 cell order.
Family totals may be balanced across lower cells to offset a genuine 768
supplier imbalance, but every per-cell family cap and overall exact family
total is hard. There is no root reuse, target relabeling, family aliasing, or
outcome input.

Remaining roots are support inventory. From unused roots only, the allocator
may seal at most 16 natural `T=1536` states, at most four per starting stage
and one per whole root, ordered by
`SHA256("O2-1536-descriptive-v1"|stage|family|root|frame|state_hash)`.
This panel remains unopened until the round-4 checkpoint is sealed. Its size,
including zero, is descriptive and never a readiness gate.

Any required lower/768 cell or family shortfall is `HOLD_O2_DATA_SUPPORT`.

## A4. Corrected Mechanism Estimands And Power

The O2 preflight computes fresh power for three exact full-option estimands:

1. lower-scale common OR over 16 `(stage,T)` strata, N128 roots, eight CRN
   replicates per arm/root;
2. 768-transfer common OR over four starting-stage strata, N64 roots, eight
   CRN replicates per arm/root;
3. pooled common OR over all 20 strata, N192 roots, eight CRN replicates per
   arm/root.

The safe requested-stage h40 endpoint, whole-root bootstrap, CRN coupling
`0.50`, and `Beta(1.6,18.4)` root base remain frozen. Target factors replace
the obsolete band factors:

- `T48:1.30`;
- `T96:1.15`;
- `T192:1.00`;
- `T384:0.85`;
- `T768:0.70`.

Starting-stage factors remain `{0:0.50,1:0.75,2:1.00,3:1.50}` and
probabilities are clipped to `[0.002,0.80]`.

For each estimand, report full-gate power at true OR
`1.25,1.50,1.75,2.00` and the 80-percent grid MDE. A simulated mechanism
replicate passes its statistical gate only when the root-bootstrap lower
bound exceeds OR 1.00 and the point estimate is at least OR 1.25. This avoids
the incoherent requirement that a true OR1.50 draw must estimate at least
1.50, which would cap power near 50 percent.

Preflight readiness requires at least 80-percent full-gate power at true
OR1.50 for the lower and pooled estimands. It reports 768-transfer power and
MDE but does not require 80-percent OR1.50 power for the N64 transfer slice.
An underpowered transfer slice is explicitly diagnostic; it cannot become a
powered claim.

The future mechanism gate requires:

- pooled point OR at least 1.25 and 95-percent lower bound above 1.00;
- lower-scale point OR at least 1.25 and lower bound above 1.00;
- 768-transfer point OR above 1.00;
- no 768-transfer material harm, defined as its 95-percent upper bound below
  OR 0.80;
- all frozen activity, survival, air, anchor, illegality, and concentration
  safeguards.

Starting-stage, family, target, and stream-block signs are descriptive except
for the explicit transfer-harm rule. No T1536 mechanism power or pass is
claimed.

## A5. Normal-Start Capability Remains The High-Scale Gate

Fresh normal-start development remains N384 paired roots. Sealed confirmation
remains N2560 paired roots. Non-starter P3072 is the high-scale milestone
co-primary; P6144 and maximum score are mandatory unpowered tail reports.

The preflight must still reproduce the A2 historical calibration and explicitly
show worst-case N2560 power for the exact paired P3072 common-OR gate. The
simulated full milestone gate at true OR1.50 is:

- P3072 common-OR 95-percent root-bootstrap lower bound above 1.00;
- point common OR at least 1.25.

`READY_O2_YIELD_PILOT_PREFLIGHT` requires at least 80-percent worst-case power
for this full N2560 gate. Confirmation promotion uses the same point/lower
threshold, plus the score, maximum, P3072 non-inferiority, P6144 reporting,
survival, air, anchor, lower-tail, corner, provenance, and integrity gates.

Normal-start activity uses A2's stage/target rule and never invents a
collector-family identity.

## A6. Corrected Stream Counts

The A1 stream namespaces and trajectory-code rules remain fixed. Exact rows
remain:

- pilot collection: 128;
- corpus collection: 640;
- training: 1,024 single-arm h40 paths;
- option development: 768 arm rows;
- untouched mechanism test: 3,072 arm rows;
- normal-start development: 768 arm rows;
- confirmation: 5,120 arm rows.

Development and test mechanism rows now reference the exact A3 lower/768
partition, not the obsolete 12 band cells. CRN and collision rules are
unchanged.

## A7. Decision Semantics

No 1536 support minimum enters preflight READY. Outcomes remain unopened.

- coherent sources, power, cost, streams, services, hashes, and integrity:
  `READY_O2_YIELD_PILOT_PREFLIGHT`;
- inadequate prospective lower/768 support after a future pilot:
  `HOLD_O2_DATA_SUPPORT`;
- correctly computed but inadequate preflight power/cost:
  `HOLD_O2_COST_OR_POWER`;
- provenance, hash, stream, schema, or aggregate failure:
  `KILL_O2_PREFLIGHT_INTEGRITY`.

READY authorizes no pilot by itself. `PROMOTE=false`.

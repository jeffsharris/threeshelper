# O2 Online Scale-Relative Option Curriculum Charter Amendment A4

Date: 2026-07-26

Status: authoritative outcome-free pilot arithmetic correction. It binds:

- base charter SHA-256
  `865f44c526d1859899b532e87dc4b99d031aa487705845b2c5412523b1997e12`;
- A1 SHA-256
  `79423c0fdce09cdb3b69b4e3bed4ddd5ec4d2dc7051cd2b3567f92c48e5e40af`;
- A2 SHA-256
  `610e564815f6c51244d474034a3f58865e30fefb3703bbc3e062b276984bb9fb`;
- A3 SHA-256
  `f19e2c9a458621722f722e84d5f51ff57804f48592c2580efdb4d1525d8472fe`.

A4 supersedes only A3's pilot yield arithmetic. The lower-to-768 scientific
design, final disjoint corpus allocator, model, streams, power, costs,
normal-start gates, and all historical locks remain unchanged. No pilot,
rollout, label, fit, action evaluation, or policy outcome is authorized.

## A1. Final Disjoint Demand

The final 640-root corpus uses each root once across all roles.

For every lower `(starting_stage,T)` cell:

- training needs 8 roots;
- development needs 2 roots;
- untouched test needs 8 roots;
- total disjoint demand is 18 roots;
- required corpus rate is `18/640 = 0.028125`.

For every `T=768` starting-stage cell:

- development needs 4 roots;
- untouched test needs 16 roots;
- total disjoint demand is 20 roots;
- required corpus rate is `20/640 = 0.03125`.

With the two-sided 90-percent Wilson lower bound and
`z=1.6448536269514722`:

- `k=7,n=128` gives `0.02991899584928971`, the smallest lower-cell count
  above `0.028125`;
- `k=8,n=128` gives `0.03557167190355302`, the smallest transfer-cell count
  above `0.03125`.

A fully root-disjoint pilot matching would require
`7*16 + 8*4 = 144` roots and is impossible at pilot N128. The pilot therefore
has two separately reported, jointly required layers.

## A2. Layer One: Root-Disjoint Structural Matching

The deterministic structural matching remains exactly as A3 specified:

- four roots in each of 16 lower cells;
- seven roots in each of four `T=768` cells;
- 92 matched roots total;
- each root used once;
- at least three families per cell;
- no family above two roots in a lower cell or three roots in a transfer cell;
- A3 cell and root ordering unchanged.

This layer proves that support is not wholly concentrated in the same roots
or one family. Its quotas are structural and are not called Wilson-qualified
yield estimates.

## A3. Layer Two: Overlapping Whole-Root Availability

Independently of structural assignment, count each complete pilot root once
within every exact cell it naturally supports. A root may appear in several
different cell availability counts, but never more than once in the same
cell.

Pilot availability requires:

- at least seven distinct whole roots in every lower cell;
- at least eight distinct whole roots in every `T=768` cell;
- at least three collector families represented in every cell;
- no family supplies more than three availability roots in any cell.

These overlapping counts provide the Wilson yield evidence. They do not
allocate roots, inflate a disjoint sample size, or guarantee the later corpus.
The final 640-root allocator remains one-root/one-state across train,
development, test, and inventory and must independently satisfy every frozen
A3 quota.

## A4. Pilot Decision

The yield pilot passes only if both layers pass after all 128 complete,
unconditionally retained roots are available:

1. the 92-root disjoint structural match;
2. all 20 overlapping Wilson-qualified availability counts and family rules.

Failure of either layer is `HOLD_O2_DATA_SUPPORT`, never representation or
policy evidence. Natural `T=1536` support remains descriptive with no minimum.
Acquisition remains non-capability evidence.

## A5. Preflight/Test Binding

The no-game preflight design manifest and focused tests must represent both
layers explicitly. They must assert:

- structural quotas sum to 92 and are root-disjoint by contract;
- availability minima are 7 for each lower cell and 8 for each transfer cell;
- lower and transfer final rates are exactly `18/640` and `20/640`;
- the corresponding Wilson lower bounds strictly exceed those rates;
- the impossible 144-root disjoint interpretation is rejected;
- final corpus role quotas still sum to the A3 disjoint 128/48/192 design.

No preflight result may be sealed without binding this amendment.

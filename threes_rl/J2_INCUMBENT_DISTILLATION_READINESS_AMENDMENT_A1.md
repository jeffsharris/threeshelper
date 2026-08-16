# J2 Incumbent Distillation Readiness Amendment A1

Status: frozen outcome-free readiness reseal; no J2 execution is authorized.

## 1. Scope and immutable parents

This amendment is additive to the sealed
`J2_INCUMBENT_DISTILLED_JOINT_POLICY_VALUE_CHARTER.md`. It preserves the
original J2 readiness package, exact-teacher feasibility pilot V1, and
exact-teacher feasibility pilot V2 byte-for-byte. It binds the complete
original readiness package and the V2 terminal, retention, central,
sensitivity, synchronous, power, inventory, marker, preflight, test-evidence,
and 16 round-manifest identities.

The sole scientific-authority change is:

- distillation validation and closed-loop teacher fidelity increase from
  2,048 to exactly 6,144 fresh paired roots.

Architecture, initialization, eight distillation epochs, optimizer, root-equal
loss, behavior-cloning mechanism gates, closed-loop fidelity gates, the
rounds-1-through-16 teacher KL schedule, 64-by-256 PPO, development,
confirmation, H0, and every historical HOLD/KILL remain unchanged. This
amendment cannot query a teacher, reserve or consume a stream, run a game,
write a label, step an optimizer, create a checkpoint, read a policy outcome,
or alter the incumbent, dashboard, or top three.

## 2. Derived authority

One frozen stage table is authoritative:

| Stage | Rows or pairs | Game arms | Stream prefixes |
| --- | ---: | ---: | --- |
| Teacher behavior cloning | 8,192 | 8,192 | 227B-230B |
| Distillation validation | 6,144 | 12,288 | 231B-235B |
| On-policy training | 16,384 | 16,384 | 236B-239B |
| Development | 896 | 1,792 | 240B-244B |
| Confirmation | 4,480 | 8,960 | 245B-249B |

Code must derive, never hand-copy, these totals:

- 36,096 whole-root rows or pairs;
- 47,616 complete-game arms;
- 155,904 unique streams;
- 14,336 pre-PPO teacher roots: 8,192 BC plus 6,144 validation controls;
- 4,096 on-policy teacher roots in PPO rounds 1-16; and
- 18,432 total exact-teacher root equivalents.

The bases remain exactly those in the parent charter. Validation intervals end
at offset 6,143; all other stage intervals are unchanged. The original J2 rows
were prospective, unopened, unreserved, and unconsumed. A1 may therefore
supersede their content-blind authority without consuming them. A1 uses
`j2-a1-root-v1` and `j2-a1-ancestry-v1` commitments over exact stage, row, and
stream identities, so every A1 root and ancestry identity is distinct from the
parent commitments.

The authority audit must prove all row, root, ancestry, and stream identities
unique; paired arms share logical/deck/slot and have distinct policy streams;
stage root and ancestry sets are disjoint; prefixes 227B-249B do not collide
with spent prefixes through 226B or engineering-only pilot prefixes 250B-255B;
and every row remains unopened, unreserved, and unconsumed.

## 3. Frozen power

Validation power uses the parent implementation byte-for-byte except
`N=6,144`: eight equal strata, control rates 0.02/0.04/0.08/0.15, couplings
0/0.05/0.10, 768 datasets per cell, 199 within-stratum whole-root bootstraps
per dataset, seed rule `2026072821`, 0.5 edge correction, NumPy linear 0.025
quantile, point OR at least 0.90, lower 95% OR above 0.50, and required
worst-cell power at least 0.80.

The exact V2 pilot row is binding: worst-cell power
`0.8059895833333334`, Monte Carlo standard error
`0.014269101547515112`, and full-report SHA-256
`157f90f6185fe7a08548a140b10d0f582c351d7d7b8abff53be78ff4ee91e28b`.
A1 must independently recompute that report and match the row exactly.

Score fidelity retains paired log-score SD 1.25, the 0.97 point floor, 0.90
lower-95% floor, and the parent normal paired-score method. At N=6,144 its
80%-power MDE is derived by code; no margin or method may change.

## 4. Measured cost and storage

No ideal scaling is admissible. The V2 observed central eight-process p99,
`0.1316514358320273` seconds per call, is the frozen pre-PPO admission
statistic. With 14,336 roots, 512 calls per root, eight workers, the sealed
optimizer fixture, and a 25% margin, the projection must fit 72 hours. Aggregate
throughput remains descriptive and a consistency check.

Online rounds 1-16 use the V2 synchronous observed throughput
`408.1622875147186` calls/second for exactly 4,096 roots times 512 calls,
added to the inherited bounded J1 PPO projection with the same 25% margin.
The exact measured contemporaneous parent-plus-workers RSS and conservative
independent peak sum remain bound. Both distillation and PPO must fit 24 GiB.

Storage is recomputed from the parent synthetic maximum-shape formula with
6,144 validation pair blobs. The central 512-move projection is conjunctive.
The frozen 5,000-move projection remains mandatory and descriptive, never a
favorable adaptation or a conjunctive veto.

## 5. Family limitation and safeguard

The V2 inventory contained exactly 139 `low_air`, 4,861
`low_constrained`, and zero `mid_progression` or `upper_progression` states.
It establishes the frozen engineering p99 and synchronous orchestration gates,
but it does not establish all-family cost invariance.

Any future complete natural teacher inventory must retain every root and pass
the unchanged parent four-family support gates before a distilled checkpoint
can become authoritative:

- each family has at least 1,024 natural states from at least 256 distinct
  validation roots;
- no natural family exceeds 0.70 of validation states;
- the deterministic capped mechanism inventory has no family above 0.40; and
- all existing overall/per-family accuracy and value gates remain unchanged.

A shortfall is a prospective data-support HOLD. It cannot change counts,
filter roots, weaken gates, or reinterpret the V2 timing evidence.

## 6. Reseal and decision

The A1 runner exposes only `audit-zero-work`, `write-test-evidence`, and
`prepare`. It binds exact parent and pilot identities, writes a fresh
content-blind authority, recomputes power and projections, and seals one
create-once readiness lock, result, and retention manifest.

Decision precedence is:

1. identity, schema, arithmetic, authority, or zero-work defect:
   `KILL_J2_A1_READINESS_INTEGRITY`;
2. power, central runtime/storage, measured-memory, family-safeguard contract,
   or operational shortfall:
   `HOLD_J2_A1_INCUMBENT_DISTILLATION_PREFLIGHT`;
3. all gates pass:
   `READY_J2_A1_INCUMBENT_DISTILLATION_PREFLIGHT`.

READY permits only research-lead review and a separately frozen distillation
execution-surface/phase-lock proposal. It does not authorize teacher data,
labels, training, fidelity evaluation, PPO, development, confirmation, or
promotion.

Terminal status is explicit:

- CONTINUE: research-lead review of the A1 readiness reseal;
- HOLD: every J2 scientific or execution action;
- KILL: historical kills only unless A1 seals an integrity KILL; and
- PROMOTE: false.

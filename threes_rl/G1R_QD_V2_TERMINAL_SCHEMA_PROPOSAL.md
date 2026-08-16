# G1-R QD V2 Terminal-Schema Proposal

Date drafted: 2026-07-25

Status: proposal only. This document does not authorize implementation, archive
construction, lock preparation, panel actions, timing, game generation, labels,
model fitting, continuation outcomes, policy evaluation, or dashboard changes.

## Trigger And Boundary

QD-v1 is permanently closed as `KILL_QD_V1_EXECUTION`. Its one authorized
admission wrote the immutable opened marker and then stopped during the frozen
reference-signature stage with:

`ValueError: Descriptor requires a live state, legal=0`

This is a schema-coverage defect. A legal base move followed by an exact visible
tile insertion can produce a valid terminal afterstate whose legal-action count
is zero. QD-v1 froze that coordinate to `1..4` and therefore could not
descriptorize every outcome required by its own exact spawn expectation.

QD-v2 is a fresh predeclared engineering revision, not a v1 recovery, rerun,
threshold adaptation, or scientific reinterpretation. V1's directory, lock,
archive, policy, marker, HOLD artifact, and unobserved partial computation remain
untouched. V2 gets a new schema, rebuilt archive, policy bundle, execution lock,
one-shot assay, output directory, and immutable hashes before any action.

## Sole Semantic Change

The ordered 14-coordinate descriptor remains identical to the authoritative v1
proposal except for coordinate 8:

- v1: legal-action count `1..4`;
- v2: legal-action count `0..4`.

The mixed-distance denominator for legal-action count is exactly `4`. Its
contribution is therefore:

`abs(left_legal_count - right_legal_count) / 4`

All values `0,1,2,3,4` are valid. Values outside that set remain hard schema
errors. A descriptor with legal count zero is terminal but otherwise uses the
same board, starter-removal, target/support, monotonicity, anchor, preview, and
pending definitions as v1.

The mixed metric remains an unweighted sum of seven categorical Hamming terms
and seven normalized ordinal terms. The ordinal denominators in v2 are,
in descriptor order:

`6, 4, 4, 4, 4, 3, 3`

for built/second Manhattan distance, support components, target/support
adjacency, empty-count bin, legal-action count, top-row violations, and
left-column violations. No other metric, rank, weight, tie rule, missing rule,
or category changes.

## Fresh Identifiers

V2 must use these identifiers, never v1 identifiers:

- behavior family: `g1r_qd_static_archive_oneply_v2_terminal_schema`;
- descriptor schema: `g1r_qd_descriptor_v2_terminal_legal0`;
- archive payload: `g1r_qd_archive_v2_terminal_legal0`;
- archive source manifest: `g1r_qd_archive_sources_v2_terminal_legal0`;
- policy bundle: `g1r_qd_policy_v2_terminal_legal0`;
- execution lock/runner: `g1r_qd_admission_v2_terminal_schema`;
- output directory:
  `threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema`.

Shared immutable inputs retain and re-bind their actual existing hashes when
their content is unchanged. This includes the frozen panel, A2 inventory, parent
checkpoint, incumbent, simulator/evaluation sources, QD-v1 proposal/failure
evidence, and unused stream namespaces. V2 must never copy, rewrite, pad, or
otherwise perturb a shared input merely to manufacture a different hash.

Only newly created v2 artifacts require new identifiers and naturally fresh
hashes: the v2 proposal/charter, implementation and focused tests, descriptor
schema, selected-source manifest generated under the v2 identity, archive and
cell table, policy bundle, execution lock, opened marker, terminal result, and
output directory. Every v2 lock must bind both sets explicitly: existing hashes
for unchanged shared inputs and new hashes for v2-owned artifacts.

## Frozen Inheritance From V1

Everything below is inherited unchanged from the authoritative amended v1
proposal:

- the root-capped A2 natural-state source inventory;
- one state per excluded ancestry by the same deterministic SHA argmin;
- exact replay hash, canonical fresh-root provenance, and state round-trip
  validation;
- starter removal, target/second-tile ties, support components and adjacency,
  monotonicity, anchor, preview, and pending formulas;
- archive occupancy, nearest-cell lookup, lexicographic tie resolution, and
  immutability;
- exact visible-preview value and legal insertion-slot enumeration;
- parent-MC1000 expected quality;
- static-archive novelty and equal ordinal quality/novelty rank sum;
- action ties by quality, novelty, then `up, down, left, right`;
- the immutable 32 pre1536 + 32 pre3072 panel;
- reference-signature identity checks and the `2%` overall plus nonzero
  per-stratum pairwise distinctness gates;
- five-pass interleaved one-process timing schedule and every absolute/relative
  threshold;
- reserved but unused `45B/46B/47B/48B` stream namespaces;
- nice, disk, service, ports, protected top-three, dashboard, process, and
  collision gates;
- atomic staging promotion, opened-marker one-shot sealing, terminal error
  sealing, and never-rerun semantics;
- zero labels, rollouts, learned bonuses, human actions, outcomes, score
  inspection, or dashboard eligibility at admission.

The QD-v1 promoted directory and failed staging directory are read-only
historical evidence and are excluded from all v2 writes.

## Mandatory Pre-Lock Tests

No v2 archive or execution lock may be prepared until all focused tests and the
broader G1/S3 regression set pass under frozen implementation/test hashes.

Focused coverage must include:

1. crafted valid terminal afterstates with legal count `0`;
2. crafted live afterstates with each legal count `1,2,3,4`;
3. exact coordinate order, finiteness, range rejection, and legal-count
   denominator `4`;
4. descriptor schema hash changes if the legal-count domain, denominator,
   name, order, formula, or normalization mask changes;
5. exhaustive enumeration of every legal base action, every exact visible
   preview value, and every legal insertion slot on all 64 frozen panel states;
6. for every hypothetical insertion outcome above, exact agreement between
   descriptor legal count and `len(sim.legal_actions(outcome_state))`, including
   both terminal and live outcomes;
7. the same exhaustive insertion-outcome audit over every selected root-capped
   archive source before lock sealing;
8. no mutation of root state, preview, cycle/deck state, archive, or parent
   model during descriptorization;
9. deterministic archive reconstruction, cell counts, nearest-cell ties,
   action ranking, save/load, and schema/hash rejection;
10. v2 loading rejects every v1 descriptor, archive, policy, and lock identifier;
11. staging failure cannot create the final directory; opened-marker and
    post-open failure behavior remain irreversible;
12. the accepted four reference signatures still exactly match the immutable
    pilot-v1 signature hashes before QD distinctness is interpreted.

The exhaustive audits are descriptor/exactness tests only. They may not retain,
inspect, or compare candidate action selections, action values, timing, scores,
or downstream outcomes.

## Conditional Execution Sequence

This proposal does not authorize these steps. If oversight later authorizes
them, they must occur in order:

1. implement only the terminal-schema change and tests;
2. review and accept fresh implementation/test hashes;
3. build the root-capped v2 archive and execution lock once through staging;
4. review and accept the promoted v2 lock;
5. open exactly one v2 action/timing admission;
6. seal exactly one of `KILL_QD_ALIAS`, `KILL_QD_COST`,
   `READY_QD_FAMILY_ADMISSION`, or `HOLD_QD_ADMISSION_ERROR`;
7. return to oversight before any acquisition.

No acquisition pilot, generated game, all-action h40 label, fit, continuation,
score outcome, incumbent update, or dashboard change is authorized by this
proposal, even if a future v2 admission passes.

## Decision

`PROPOSE_QD_V2_TERMINAL_SCHEMA_ONLY`

G1-R acquisition and `PROMOTE` remain held. The revision exists solely to make
the unchanged frozen QD calculation total over every simulator-valid exact
insertion outcome.

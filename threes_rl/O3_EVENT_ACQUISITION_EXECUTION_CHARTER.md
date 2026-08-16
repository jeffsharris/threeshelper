# O3 Event Acquisition Execution Charter

Date: 2026-07-26

Status: executable only after the immutable O3 P0 result is
`READY_O3_EVENT_ACQUISITION`. This charter implements section 6 of
`O3_EVENT_CONDITIONED_DESIGNATED_PAIR_CHARTER.md`; it does not change its
scientific contract.

## 1. Immutable Inputs

The execution marker binds the O3 course charter, O3 representation and power
sources, P0 runner/tests/test evidence, and every immutable P0 marker/result,
stream, partition, policy, power, and collision artifact by file and canonical
payload hash. It also binds the current acquisition runner/tests/test evidence,
the exact five policy/checkpoint payload manifests, simulator/evaluator/replay/
provenance sources, output directory, command, resource limits, and jobs=1.

The only O2 access permitted is the already-sealed aggregate evidence inside
the P0 result. No O2 support rows, replay paths, states, pair coordinates, or
source replay content may be parsed.

## 2. Frozen Collection

Collect exactly 20,500 complete fresh normal-start games, 4,100 each in this
order:

1. `o3_corner2`;
2. `o3_expectimax2`;
3. `o3_parent_mc1000`;
4. `o3_replaycal`;
5. `o3_qd_v2`.

Use only the P0 acquisition rows in the 105B/106B/107B/108B namespaces. A
chunk is one game from each family at the same family-game index, evaluated
in the listed order. There is one worker and one heavy process at nice >=10.
Every replay and completion/provenance row is retained unconditionally. No
score, milestone, maximum tile, action, geometry, or support field may be used
for filtering, stopping, family allocation, or analysis during collection.

An append-only attempt ledger writes a deterministic opened event before every
dispatch and exactly one terminal event afterward. Charged evaluator runtime
is persisted before replay/completion writes. Same-marker recovery validates
and completes retained replay/completion evidence, then evaluates only the
missing frozen row. No replay or completion is overwritten.

Hard bounds are 144 active evaluator hours, 28 GiB incremental output, and
100 GiB free disk, with 120 GiB the operating target. Disk, services,
dashboard/top-three truth, and process contention are checked before opening,
before each chunk, and at terminal. Immutable inputs and the complete stream
collision audit are checked before opening, again before the first game, and
at terminal.

## 3. Completion Barrier And Support Scan

Replay content remains sealed until all 20,500 unique completions, streams,
fresh ancestries, retained replay hashes, and attempt-ledger events pass.
After that barrier, the scanner may access current simulator state fields only.
It never accesses recorded actions or final/future score, milestone, maximum
tile, or terminal outcome fields. Reset/root score fields may be read only by
the frozen fresh-root provenance validator.

For each whole ancestry and exact target T in 48/96/192, retain at most one
hard-start state. It must be live, anchor-safe, have at least two empty cells
and two legal actions, contain the canonical designated T pair, and have zero
pair-specific safe-merge actions. The candidate key is exactly:

`SHA256("O3-event-root-v1"|role|target|family|root|frame|state_hash)`.

Within each role, allocate targets rare-first 192,96,48. For target index j
and role index i, visit families in cyclic order starting at `(i+j) mod 5`;
take the next unused-root candidate in candidate-hash order from each family
until the target quota is filled or a complete cycle makes no progress.

Required counts are:

- train: 48/29/19 for T48/T96/T192;
- development: 16/10/6;
- untouched mechanism: 96/58/38.

Each role must contain at least four families, no family above 40 percent, and
each represented family at least 4/2/8 roots for train/development/mechanism.
Stage is descriptive only.

## 4. Decisions And Sealing

The open command writes only `O3_ACQUISITION_OPENED.json` and exits. The
execute command requires that exact marker and is resumption-safe while no
terminal result exists.

The terminal result is exactly one of:

- `READY_O3_OPTION_TRAINING`;
- `HOLD_O3_DATA_OR_POWER`;
- `HOLD_O3_ACQUISITION_INTEGRITY`.

READY requires every frozen collection, provenance, stream, resource, and
allocation gate. A support shortfall is HOLD, never representation failure.
An operational or integrity fault is fail-closed HOLD. No acquisition result
is a policy outcome or dashboard-eligible capability claim.

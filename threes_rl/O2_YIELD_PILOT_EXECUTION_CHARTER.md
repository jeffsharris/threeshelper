# O2 Yield-Pilot Execution Charter

Date: 2026-07-26

Status: authoritative one-shot execution contract for the authorized O2
128-root yield pilot. This charter authorizes no corpus expansion, option
rollout, label generation, model fit, policy evaluation, or promotion.

## 1. Immutable Scientific Inputs

This execution binds:

- O2 base charter SHA-256
  `865f44c526d1859899b532e87dc4b99d031aa487705845b2c5412523b1997e12`;
- A1 SHA-256
  `79423c0fdce09cdb3b69b4e3bed4ddd5ec4d2dc7051cd2b3567f92c48e5e40af`;
- A2 SHA-256
  `610e564815f6c51244d474034a3f58865e30fefb3703bbc3e062b276984bb9fb`;
- A3 SHA-256
  `f19e2c9a458621722f722e84d5f51ff57804f48592c2580efdb4d1525d8472fe`;
- A4 SHA-256
  `1095462bbef0759ce8c56573727a645d3859469f5b50e1231e907c3cd8a479b3`;
- sealed O2 preflight result file/payload SHA-256
  `09c8b19cea4b992f3b77feab71b978c717e88f4dbf0547a5789e54759ae430fc` /
  `60ffe3905a032e26e1a32b7f248e4f138cf56f09c0d49b44b59802e824fa5f5b`;
- sealed design-manifest file/payload SHA-256
  `d921ae156503f697d9d689f39bc664a83cb88ca21e9bb01d5fdb34653de30480` /
  `287780188a284044adffb366791524ea2da3daa04d5db903b84bb91b2f4b5c00`;
- sealed stream-manifest file/payload/row SHA-256
  `ff0a67f98bc68ebfee5b3700b533f0a8b0e7e8f628f22cbf840cc91d7e6eedbc` /
  `fb29eb96e8f4cd90b5783d4dadae496ffc450d373685e36542243af6dfdf14bf` /
  `b4bcc661d0502df3b75178b496a75e3b19770d8e288698ac8a585b15e7d8d836`.

The execution marker must additionally bind the exact execution-charter,
runner, focused-test, test-evidence, policy/checkpoint, O1 geometry/schema,
simulator/evaluator, and service-audit hashes current after tests pass.

## 2. Output And One-Shot State Machine

The sole output directory is:

`threes_rl/runs/forensics/o2_yield_pilot_v1`

An `open` command performs all zero-game identity, collision, process,
priority, storage, disk, service, dashboard, and top-three checks. Only if all
pass may it atomically create the directory and immutable
`O2_YIELD_PILOT_EXECUTION_OPENED.json`. The command exits immediately and
creates no replay, completion row, stream-consumption record, support record,
or terminal result.

The marker binds the exact later `execute` command. The execute command
requires that exact marker and refuses any source, policy, manifest, command,
jobs, output-directory, or marker mismatch. An interrupted process may resume
only under the same marker and command. A terminal result makes the execution
immutable and non-rerunnable.

## 3. Frozen Collection

The pilot is exactly 128 complete, naturally terminal, fresh normal-start
machine roots:

1. `o2_corner2`, 32 roots;
2. `o2_expectimax2`, 32 roots;
3. `o2_parent_mc1000`, 32 roots;
4. `o2_qd_v2`, 32 roots.

Policies, signatures, checkpoint payloads, and family order are exactly those
bound by the sealed O2 preflight. Starter tile is 1536 and evaluator
`max_moves` is 5000; a nonterminal max-move truncation is an integrity fault,
not a complete root.

The pilot uses only the 128 `purpose=pilot` rows in the sealed stream
manifest:

- logical base 81,000,000,000;
- deck base 82,000,000,000;
- slot base 83,000,000,000;
- policy base 84,000,000,000.

There is one worker and one heavy process at nice at least 10. Collection uses
eight rounds. Each round runs one four-game chunk for each family in frozen
family order, so chunks are strictly below eight and every family advances
equally. There is no support scan, score/milestone inspection, or within-family
early stop.

Every completed replay is retained unconditionally. Replay filenames depend
only on family index and game index. Completion rows contain only frozen
identity, stream IDs, source path/hash, fresh-root provenance, and completion
integrity. They contain no score, max tile, recorded action, selected
geometry, or policy comparison. Existing final score/max-tile/action fields
inside the opaque evaluator replay are never read or used.

Before each root is dispatched, an append-only attempt ledger records its
family/game, four stream IDs, chunk index, deterministic attempt index and
immutable attempt ID with status `opened`. The same attempt ID is paired with
`completed`, `completed_recovered`, or `interrupted_no_replay`. A retry is
permitted only after the prior attempt is explicitly closed as interrupted,
and its attempt index must be the next contiguous integer. Existing replay
and completion hashes are revalidated on resume; an orphan replay is recovered
without reevaluation. Evaluator wall time is durably charged immediately after
the evaluator returns, before any replay or completion write. Recomputed work
may therefore be conservatively double-counted, but spent evaluator time may
not be lost.

Hard gates are:

- active runtime below six hours;
- output below three GiB;
- free disk at least 100 GiB, with 120 GiB target;
- healthy ports 8765 and 8770, advisor, dashboard record 263670, and protected
  top three 263670/261369/258561;
- no competing heavy process;
- exact stream and ancestry uniqueness.

The active-runtime gate is checked before every chunk and once again after all
collection and before support scanning. The attempt ledger, all 128 completion
rows, all replay hashes, and exact one-final-completion-per-root accounting must
pass before support content can be opened.

## 4. Post-Completion Support Scan

Support content remains inaccessible until all 128 completion rows and replay
hashes pass. The scanner then reads only:

- board, preview, tile-cycle, move count, game-over flag, and legal mask;
- root/family/frame identity;
- current anchor/air validity, legal-action count, target pair, and stage.

Reset/root score may be read only by the frozen provenance helper to establish
fresh reset origin. Final/future score, final/future milestone or max tile,
recorded actions, policy outcomes, and favorable replay ranking are forbidden.

For each live frame and exact target `T in {48,96,192,384,768,1536}`, the
state is eligible when:

- O1-A3 anchor and air predicates both pass;
- at least two legal actions exist;
- `geometry(board, starter_tile, fixed_target=T)` returns a pair.

The resulting exact pair stage is 0, 1, 2, or 3. For each
`(root,stage,T)`, retain one compact current-state support row chosen by:

`argmin SHA256("O2-pilot-exact-v1"|stage|T|family|root|frame|state_hash)`,

then lower frame index, then lexical state hash. State hashes exclude score
and action fields.

Natural `T=1536` support is descriptive only. At most 16 descriptive rows,
at most four per stage and one per root, are retained in the compact report by
the frozen A3 hash order. Its presence or absence never changes readiness.

## 5. A4 Layer One: Structural Matching

Cells are ordered by `T=768,384,192,96,48`, then stage `0,1,2,3`.
Required quotas are seven roots in each T768 cell and four roots in every
lower cell, for 92 distinct roots.

Feasibility is solved once with SciPy 1.17.1 `scipy.optimize.milp`, using
binary root-cell assignment and family-presence variables:

- each root is assigned to at most one cell;
- each cell fills its exact quota;
- at least three families appear in every cell;
- no family exceeds three roots in a T768 cell or two in a lower cell.

Variables are ordered by frozen cell order, candidate selection hash, family,
and root. The fixed objective minimizes normalized candidate-order rank plus a
strictly smaller variable-index tiebreak. Presolve is enabled; integrality and
all constraints are rechecked exactly after rounding. There is no solver,
objective, or matching fallback. Solver error is an operational integrity
HOLD; valid infeasibility is `HOLD_O2_DATA_SUPPORT`.

## 6. A4 Layer Two: Overlapping Availability

For each exact cell and family, roots are ordered by candidate selection hash,
then root. At most the first three roots from one family are credited in that
cell. The same root may be credited in different cells but only once within a
cell.

Every lower cell requires at least seven credited distinct roots; every T768
cell requires at least eight. Every cell requires at least three represented
families. Raw natural root counts and capped credited counts are both reported.
The exact two-sided 90-percent Wilson lower bounds must exceed `18/640` and
`20/640`, respectively.

The 92-root structural layer and all 20 availability cells must both pass.
The 144-slot fully disjoint Wilson interpretation remains forbidden.

## 7. Decisions

After all integrity, completion, resource, and support checks:

- `READY_O2_CORPUS_COLLECTION`: both A4 layers pass;
- `HOLD_O2_DATA_SUPPORT`: integrity passes but either support layer fails;
- `HOLD_O2_PILOT_OPERATIONAL_INTEGRITY`: collection, source, solver,
  stream, ancestry, service, storage, process, or sealing integrity fails.

READY authorizes no 640-root corpus by itself. Every result remains
non-promotable and dashboard-ineligible. `KILL=false`, `PROMOTE=false`.

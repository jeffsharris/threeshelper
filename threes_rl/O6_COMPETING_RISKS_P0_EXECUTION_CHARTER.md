# O6 Competing-Risks P0 Execution Charter

Date: 2026-07-27

Status: execution-surface preparation only. This charter does not authorize
test evidence, a preflight lock, an execution marker, corpus scanning, stream
reservation, power execution, labels, training, or policy outcomes.

## 1. Immutable Scientific Parent

This execution surface implements, without amendment, the accepted O6 source
preparation:

- `O6_COMPETING_RISKS_P0_CHARTER.md`
  SHA-256
  `2ee1e4273866f7f40376fb584e908f5a0e10e70446e2540f36bf320ac0edbb11`;
- `o6_competing_risks_p0.py`
  SHA-256
  `c1a1d0a22fa185672e62f0b712d79d8bd01d76e04cebe04bca78b45a7c092dd6`;
- `tests/test_rl_o6_competing_risks_p0.py`
  SHA-256
  `3d7cbe8f20149f3b21305e8306762f8ede78f2227094f8848dc5b6f383ba0b34`.

The preparation files are immutable and are not imported as a source scanner,
stream writer, label generator, trainer, or power executor. O6 cannot use any
O5 checkpoint, selected root, task stream, episode, label, prediction, or
model output. O3/O4/O5 protected bodies remain parse-forbidden.

## 2. Separate Identity And State Machine

New source paths are:

- `threes_rl/O6_COMPETING_RISKS_P0_EXECUTION_CHARTER.md`;
- `threes_rl/o6_competing_risks_p0_execute.py`;
- `tests/test_rl_o6_competing_risks_p0_execute.py`.

The sole output namespace is
`threes_rl/runs/forensics/o6_competing_risks_p0_execution_v1`.
Exact output names are:

1. `O6_P0_EXECUTION_TEST_EVIDENCE.json`;
2. `O6_P0_EXECUTION_PREFLIGHT_LOCK.json`;
3. `O6_P0_EXECUTION_PREFLIGHT_RESULT.json`;
4. `O6_P0_EXECUTION_OPENED.json`;
5. `O6_P0_PROTECTED_INVENTORY.json`;
6. `O6_P0_EXCLUSION_UNION.json`;
7. `O6_P0_SOURCE_INVENTORY.json`;
8. `O6_P0_CANDIDATE_ROOTS.json`;
9. `O6_P0_SELECTED_ROOTS.json`;
10. `O6_P0_STREAM_RESERVATION.json`;
11. `O6_P0_COLLISION_AUDIT.json`;
12. `O6_P0_POWER_PROGRESS.sqlite3`;
13. `O6_P0_POWER_TABLE.json`;
14. `O6_P0_OPERATIONAL_AUDIT.json`;
15. `O6_P0_RUNTIME.json`;
16. `O6_P0_RESULT.json`.

The exact command states are:

- `audit-zero-work`: read-only; validates immutable source identities,
  schemas, dependencies, output absence, services, process ownership, and
  disk. It writes nothing.
- `write-test-evidence`: future authorization only; writes item 1 once.
- `seal-preflight`: future authorization only; requires item 1 and writes
  items 2-3 once without scanning corpus content.
- `open`: future authorization only; validates the lock, inventories and
  hashes candidate/protected files without parsing payloads, and atomically
  writes item 4 containing that byte inventory.
- `execute`: future authorization only; requires the exact existing marker,
  parses only marker-bound permitted payloads, and writes items 5-16.

No command may skip a state. A terminal result forbids every rerun. An
interrupted `execute` may resume only with the same output path, lock, marker,
source hashes, command, jobs, and nice setting.

## 3. Byte Inventory Before Content

The future `open` command freezes one canonical byte inventory before the first
JSON/JSONL/CSV payload is parsed. It resolves and rejects symlinks, aliases,
paths outside the repository, and duplicate canonical paths.

Protected discovery uses every `.json`, `.jsonl`, and `.csv` under:

- `threes_rl/runs/eval_manifests`;
- `threes_rl/runs/eval_artifacts`;
- `threes_rl/runs/forensics`;
- `threes_rl/runs/continuations`;
- `threes_rl/runs/replays`;
- `threes_rl/runs/human_diagnostics`;
- direct children of `threes_rl/runs/dashboard`.

Candidate replay discovery is the deduplicated union of:

- `threes_rl/runs/**/replay.json`;
- `threes_rl/runs/**/source_replays/*.json`;
- `threes_rl/runs/replays/**/*.json`.

O3/O5 episode/checkpoint directories, all human diagnostics, continuation
bodies, and non-governance replay bodies are hash-only. The marker records
path, file SHA-256, size, classification, and whether byte stability is
required. Dashboard live summaries are classified but are not immutable.
Every other inventory row must match at execute and terminal.

## 4. Protected Exclusion Union

Governance filenames contain one of the exact preparation tokens:
`manifest`, `lock`, `marker`, `opened`, `result`, `seal`, `audit`,
`selection`, `selected`, `roots`, `streams`, `collision`, `retention`,
`attempt`, `completion`, `completed`, `runtime`, `task`, `config`, or
`preflight`.

Only these fields may enter the exclusion union:
`ancestry`, `ancestry_id`, `root`, `root_cluster`,
`source_replay_sha256`, `replay_sha256`, `state_sha1`, `state_sha256`,
`logical_seed`, `deck_stream_id`, `slot_stream_id`, and
`policy_stream_id`. JSON/JSONL/CSV traversal may read those fields and
container structure only. Unknown root/stream-bearing keys, malformed
governance data, missing hash-only identity companions, or inventory drift
seal `HOLD_O6_DATA_PREFLIGHT`.

Candidate ancestry, root, replay hash, state hash, and every source stream must
have zero intersection with the union. All O3/O4/O5 reservations and selected
roots, sealed confirmation roots, top-three sources, human sources, prior
selector/MCTS/reachability/first-action/continuation sources, and historical
locks are therefore excluded.

## 5. Natural Source And Family Contract

Only complete normal-start machine replays are eligible. The first frame must
reproduce a fresh reset using board/preview/cycle/move-count fields without
reading score. The last frame must have `game_over=true`. Human, restart,
replay-start, continuation, synthetic, playlist, partial, source-selected,
and score-filtered sources are rejected by provenance fields or path class.

Canonical ancestry is `fresh:{seed}:{starter_tile}`. Conflicting direct-root
fields, duplicate ancestry with inconsistent policy identity, or any protected
intersection is a data HOLD. Replay copies are reduced by:

`SHA256("O6-source-copy-v1"|ancestry|replay_sha256|resolved_path)`.

Exact family order and accepted policy specs are:

1. `o6_corner2`: `corner2`;
2. `o6_expectimax2`: `expectimax2`;
3. `o6_parent_mc1000`:
   `ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest`;
4. `o6_replaycal`:
   `ntuple_expectimax2:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest`.

At least one of `acquisition_policy_spec`, `policy_spec`,
`source_policy_spec`, or `policy` must match exactly. Conflicting recognized
specs reject the source. The accepted four action-signature identities and
six pairwise gates are hash-bound, not recomputed with actions or timing.

## 6. Candidate And Partition Allocation

P0 may inspect only each permitted frame's current board, preview, exact
tile-cycle context, move count, game-over flag, frame index, source identity,
and policy spec. It never accesses frame action/move, score, future milestone,
max-tile history, terminal score, or policy outcome.

For T48/T96/T192, exact designated-pair selection, anchor/air safety,
no-immediate-safe-merge, at least two empties, at least two legal actions,
lineage initialization, every-legal-action feature-domain, and simulator
round-trip checks must pass. Board values stay in their native domain;
normalized features/targets/probabilities must be finite in `[0,1]`.

Each ancestry contributes only the minimum:

`SHA256("O6-P0-root-v1"|ancestry|state_hash|frame_index|target|pair_coords)`.

For each candidate N in `192/256/384/512`, train/development/untouched counts
are exactly those in the accepted preparation charter. Each role is exactly
25% per family. T48/T96/T192 differ by at most one. Each family-target cell is
split aligned/unaligned with floor-half counts; an odd remainder goes to
aligned iff `(role_index + family_index + target_index) mod 2 == 0`, otherwise
to unaligned. Allocation order is role, family, target, alignment, then the
candidate SHA. There is no backtracking, substitution, reassignment, or
post-content quota change.

## 7. Stream Reservation And Collision

The exact proposed windows remain 197B-212B in the accepted four logical,
deck, slot, and policy quartets. P0 writes only 16 half-open one-million-ID
range rows. It does not consume an ID.

Every historical identity extracted from the marker-bound inventory must fall
outside every range. Policy IDs are globally distinct by range. Later CRN
row-level sharing is not defined or authorized here. Any range collision or
unclassified identity source seals `HOLD_O6_DATA_PREFLIGHT`; windows cannot
move.

## 8. Exact Prospective Power

The future execution evaluates all:

- N in `192/256/384/512`;
- true OR in `1.25/1.35/1.50/1.75/2.00`;
- ICC in `0.05/0.15/0.25`;
- exactly 4,096 datasets per cell;
- exactly 4,096 stratified whole-root bootstraps per dataset;
- exactly eight paired CRN replicates per arm/root.

The base rate is exactly `188/1152`. Root probabilities use the frozen beta
mean/ICC parameterization; treatment applies the exact odds shift. One uniform
is shared by paired arms/replicate. Strata are the frozen target/alignment
counts from Section 6. The common OR uses Haldane-Anscombe `+0.5` in every
2x2 stratum cell. The lower bound is NumPy linear 2.5% quantile over all 4,096
bootstrap ORs. Pass is point OR `>=1.25` and lower95 `>1.0`.

Execution batches exactly 16 datasets and 64 bootstraps. A PCG64 generator is
initialized once per cell from the accepted cell seed. After each 16-dataset
batch, one SQLite transaction inserts all 16 immutable results and updates
the exact JSON generator state. A crash before commit replays the whole
deterministic batch; a crash after commit resumes at the next batch. Unique
keys are `(N,true_OR,ICC,dataset_index)`. Config/schema/marker hashes live in
SQLite metadata. Duplicate, missing, out-of-order, nonfinite, reduced-count,
or RNG-state drift is `KILL_O6_P0_INTEGRITY`.

`O6_P0_RUNTIME.json` is the sole mutable orchestration journal. It is
marker-bound, atomically replaced, self-hashed, and charges each phase before
the next phase starts. An interrupted open phase is charged through resume
wall time, which may conservatively overcount downtime but cannot undercount
work. The 36-hour cap applies to this total across inventory validation,
identity parsing, source scanning, allocation, collision checking, and power.

All 60 cells must close, yielding exactly 245,760 dataset rows and
1,006,632,960 bootstrap estimates. Power is the minimum full-pass fraction
over ICC for each N/OR. Select the smallest N with OR1.50 power `>=0.80`, then
the smallest grid OR with power `>=0.80` at that N. No passing N is
`HOLD_O6_DATA_PREFLIGHT`. No approximation, analytic CI, reduced bootstrap,
or alternate random stream is permitted.

## 9. Resources, Services, And Restart

Frozen execution settings are jobs `1`, nice at least `10`, one heavy process,
36 active hours, output below `4 GiB`, disk hard minimum `100 GiB` and target
`120 GiB`. The conservative dry-run projection is `1.5 GiB` and `24` active
worker hours. The arithmetic workload is 338,228,674,560 root-index draws.

Process auditing uses PID-only `pgrep` results for frozen heavy module
patterns; it never records full command lines or human-session content. Ports
8765/8770 must accept local connections. Recorder health may read only
top-level `status` and advisor readiness; active-session rows/content are
ignored. Protected top-three scores remain `263670/261369/258561`.

Operational, service, disk, source-support, provenance, family, collision, or
power-readiness failures are `HOLD_O6_DATA_PREFLIGHT`. Immutable source,
marker, schema, simulator-domain, SQLite, count, or deterministic-resume
corruption is `KILL_O6_P0_INTEGRITY`. KILL precedence is first, then HOLD,
then `READY_O6_COMPETING_RISKS_P0`.

READY authorizes only a separately frozen label/training charter. It does not
authorize labels, training, policy evaluation, incumbent change, dashboard
change, or promotion.

`CONTINUE=execution-surface review only`; `HOLD=all commands except the
read-only zero-work audit`; `KILL=false`; `PROMOTE=false`.

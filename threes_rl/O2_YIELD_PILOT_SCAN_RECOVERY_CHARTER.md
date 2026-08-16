# O2 Yield-Pilot Scan-Only Recovery Charter

Date: 2026-07-26

Status: authoritative scan-only recovery for the immutable O2 pilot operational
HOLD. The original marker, result, attempts, completions, runtime, and 128
replays remain authoritative and byte-locked. This recovery cannot rerun a
game, consume a stream, evaluate a policy, inspect score/action/max-tile
outcomes, collect a corpus, train a model, or change the incumbent/dashboard.

## 1. Immutable Source Evidence

The recovery marker binds exact file and canonical payload hashes for:

- original execution charter, runner, tests, and test evidence;
- original execution marker and terminal
  `HOLD_O2_PILOT_OPERATIONAL_INTEGRITY` result;
- original append-only attempts, completions, and runtime files;
- all 128 completion-referenced replay paths, byte sizes, and SHA-256 hashes.

The opening audit parses only compact attempt/completion/runtime metadata.
Replay files are byte-hashed but their JSON/state content remains unopened.
The original `O2_PILOT_SUPPORT.json` must be absent.

Opening requires exactly 128 completion rows and 256 attempt events, 32 roots
per frozen family, 128 unique ancestries, 128 unique replay paths and hashes,
exact frozen stream identities, 32 completed chunks, 128 charged games,
runtime below six hours, output below three GiB, free disk at least 100 GiB,
healthy services, and protected top three `263670/261369/258561`.

## 2. Failure Reproduction And Adapter

The raw completion schema contains `game_index` and deliberately does not
contain `family_game_index`. Calling the immutable original `_stream_key` on a
raw completion must reproduce `KeyError("family_game_index")`.

The only repair is a dedicated metadata adapter that copies
`game_index -> family_game_index` in an in-memory audit view. It must reject a
row that lacks `game_index`, already contains `family_game_index`, or changes
any other field. The adapted rows must pass the immutable original attempt
ledger audit. No replay, support, geometry, MILP, stream, or policy semantics
change.

## 3. One-Shot Recovery State Machine

The sole output namespace is:

`threes_rl/runs/forensics/o2_yield_pilot_scan_recovery_v1`

An `open` command creates only immutable
`O2_SCAN_RECOVERY_OPENED.json` after every source, schema, operational, and
zero-content check passes. It binds the exact later execute command, charter,
runner, tests/evidence, source manifest, adapter audit, and original evidence.
The marker is logged before execution.

The `execute` command requires that exact marker and recomputes every binding.
It is one-shot: an existing terminal result rejects execution. Any
post-marker fault seals `HOLD_O2_SCAN_RECOVERY_INTEGRITY`; it never edits or
retries the original pilot.

## 4. Exact Recovered Support Scan

After marker validation, and only then, execution loads the existing 128
replays and invokes the immutable original:

`o2_yield_pilot.support_analysis(completions)`

This preserves exact O1-A3 geometry, target/stage predicates, deterministic
root-state selection, A4 SciPy MILP ordering/objective/rounding checks,
structural quotas/family caps, overlapping availability/Wilson rules, and
descriptive-only T1536 handling.

Allowed semantic reads are current board/preview/cycle/move/legal/provenance
fields. Reset/root score may be read only by the frozen fresh-root provenance
helper. Final/future score, milestone/max tile, recorded action, and policy
outcome fields are forbidden.

## 5. Terminal Decisions

- `READY_O2_CORPUS_COLLECTION`: exact recovered A4 support passes.
- `HOLD_O2_DATA_SUPPORT`: exact recovered A4 support is insufficient.
- `HOLD_O2_SCAN_RECOVERY_INTEGRITY`: source, adapter, provenance, scan, MILP,
  service, storage, or sealing integrity fails.

Every terminal state holds corpus collection, rollouts, labels, fitting,
policy evaluation, confirmation, incumbent/dashboard changes, and promotion
for research-lead review. `KILL=false` and `PROMOTE=false`.

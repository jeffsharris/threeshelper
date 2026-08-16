# G1-R QD Family Admission Execution Charter

Date: 2026-07-25

Status: authorized for QD admission only under proposal SHA-256
`e9a72c659ae43302a3f646614c1a5e1c09daf8c696f6d5d1d7d5d150a50bf880`.

## Boundary

This charter authorizes:

- one exact implementation of `g1r_qd_static_archive_oneply_v1`;
- focused tests and broader G1/S3 regressions;
- one root-capped, source-validated A2 static archive;
- one immutable execution lock;
- exactly one outcome-free 64-state action-distinctness and latency admission.

It forbids normal-start game generation, acquisition pilot execution,
continuations, all-action h40 labels, model fitting, score inspection, human
action use, incumbent changes, and dashboard changes.

## Immutable Inputs

- Proposal SHA-256:
  `e9a72c659ae43302a3f646614c1a5e1c09daf8c696f6d5d1d7d5d150a50bf880`.
- Original G1-R pilot-v1 preflight SHA-256:
  `f78288b3f47bda6aa6d15c2157fd79f7b3d0685f0367d8b9964f5dc73981ea91`.
- Pilot-v1 panel SHA-256:
  `b8862aa3c8eaf6278fc078fb3e03aa7222a01930673cfee497738c74e81eff9d`.
- A2 source inventory:
  `runs/forensics/r15a_context_a1/r15a_natural_state_inventory_a1_20260711.json`.
- Parent MC1000 checkpoint:
  `runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest`.
- Output directory:
  `runs/forensics/g1r_qd_admission_v1`, resolved and bound in the lock.

The lock must hash the proposal, this charter, implementation, focused test,
A2 inventory and selected root-capped source manifest, complete archive cell
table, full parent checkpoint payload, simulator/evaluator/policy sources,
incumbent file, original preflight, and panel.

## Archive Selection

Use only `selected_records` from the frozen A2 inventory with fresh root origin
and starter 1536. Collapse by canonical `root_cluster`. Select one record per
root by lexicographic argmin of:

`SHA256(canonical_json(["G1R-QD-archive-state-v1", root_cluster, record_id, state_sha1]))`.

`state_sha1` is recomputed from the complete payload before selection. Every
selected replay hash, exact frame, payload, simulator round trip, and canonical
root must validate. The compact selected-source manifest and sorted
`(descriptor_cell,count)` table are immutable and separately hashed. Copies do
not increase counts.

All descriptor, mixed-distance, archive-count, nearest-cell, spawn-expectation,
rank, tie, scarcity, yield-projection, and latency formulas are exactly those
in the authoritative proposal. There is no variant or sweep.

For one-ply post-insertion descriptor outcomes, preview category and
`large_pending` are copied from the live root state. The contract stops after
inserting the currently visible tile and does not sample a next preview. Parent
quality is `NtupleValue.value(post_insertion_board)` for that exact outcome.

## Reserved but Unused Streams

Admission reserves and collision-checks, but never consumes:

- logical `45_000_000_000 + game`;
- deck `46_000_000_000 + game`;
- slot `47_000_000_000 + game`;
- policy `48_000_000_000 + game`;
- game range `0..11,999`.

Any historical collision blocks the lock. Admission records
`streams_consumed=false`.

## Operational Gate

Preparation and admission require:

- output directory identity exact;
- one process, admission timing worker count exactly one;
- process nice value at least 10, inherited by children;
- no other heavy Threes training/evaluation/acquisition process;
- free disk above 100 GiB and target 120 GiB reported;
- ports 8765 and 8770 healthy without reading active sessions;
- dashboard record exactly 263670 and protected top-three files present;
- original G1-R preflight byte-identical;
- all source, parent, code, panel, archive, and lock hashes current.

The preparation command performs no policy action. The admission command
revalidates the lock before its first action and writes no partial action or
timing artifact.

## One-Shot Admission

The four admitted reference components are:

1. `g1r_corner2`;
2. `g1r_expectimax2`;
3. `g1r_parent_mc1000`;
4. `g1r_replaycal`.

On the immutable 32 pre1536 + 32 pre3072 panel, compute one deterministic
action signature for the QD candidate and each reference. QD must disagree
with every reference at least 2% overall and at least once in both strata.
Record exact ties and input nonmutation.

Run the proposal's one-warmup/five-pass interleaved timing schedule exactly once
against the full frozen incumbent. Use one process and no worker pool. Record
all per-call nanoseconds, distributions, ratios, call-order parity, and
thermal/power availability.

Decision precedence:

1. `KILL_QD_ALIAS` if any pairwise action gate fails.
2. `KILL_QD_COST` if action gates pass but any absolute or relative latency
   gate fails.
3. `READY_QD_FAMILY_ADMISSION` only if every action, latency, exactness,
   provenance, service, process, storage, and serialization check passes.

Every decision artifact must state zero games, labels, models, continuations,
score outcomes, and dashboard changes. Even READY returns to oversight and
does not authorize an acquisition pilot.

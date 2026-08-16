# G1-R QD-v2 Supplemental Storage Admission Audit Charter

Date frozen: 2026-07-25

## Status And Scope

This charter authorizes one outcome-free, metadata-and-hash-only supplemental
storage audit for the already sealed QD-v2 action admission. It does not reopen,
rerun, modify, or backfill the one-shot action assay.

The authoritative immutable admission inputs are:

- `runs/forensics/g1r_qd_admission_v2_terminal_schema/ADMISSION_OPENED.json`
  with file SHA-256
  `11b21137303fa4cfd258dfe3ff536b227c24fa4cb7db727ca376b970418c5135`.
- `runs/forensics/g1r_qd_admission_v2_terminal_schema/admission_result.json`
  with file SHA-256
  `27bcb3328a02d6dc5094dcc5a8e52b8f27d2f3e4ea7b92f5c1a8153bc1326a8e`
  and canonical payload SHA-256
  `0eced74a61279613661e82d3f696c8e1d34d5256548194b155ad2db3d3ed38e2`.
- The admission-open timestamp and inventory cutoff are exactly
  `2026-07-25T10:26:56-0700`.

The action-family decision remains `READY_QD_FAMILY_ADMISSION`. Acquisition
remains `HOLD_QD_STORAGE_REPORTING_DEFECT` until this supplemental gate is
sealed. No candidate/reference action, stream, game, label, model,
continuation, score, or policy outcome may be generated or inspected.

## Replay Inventory Contract

Scan recursively beneath `threes_rl/runs` using filesystem metadata only.
Include every existing regular file whose basename is exactly `replay.json`
and whose modification time is strictly earlier than the frozen admission-open
cutoff.

Exclude only files beneath these two exact promoted directory roots:

- `threes_rl/runs/forensics/g1r_qd_admission_v1`
- `threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema`

Do not exclude staging directories, policies, families, scores, terminal
states, or favorable/unfavorable file sizes. Do not open or parse any
`replay.json`. For every included file, record:

- workspace-relative POSIX path;
- logical byte size from `stat.st_size`;
- integer `mtime_ns`;
- UTC ISO-8601 mtime;
- `predates_cutoff=true`;
- SHA-256 of the raw file bytes.

The inventory order is ascending path. The top-ten report is ordered by byte
size descending, then path ascending. Let `M` be the largest included logical
byte size; ties choose the lexicographically first path. The audit fails closed
if the eligible inventory is empty, a listed path disappears or ceases to be a
regular file during hashing, or size/mtime metadata changes between the
pre-hash and post-hash `stat` calls.

Write the immutable inventory to:

`threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema/QD_V2_STORAGE_REPLAY_INVENTORY.json`

## Directory Size And Projection

Let `B` be the exact current logical byte size of the immutable QD-v2 admission
directory immediately before either supplemental artifact is written. Compute
`B` as the sum of `stat.st_size` for every regular file recursively beneath:

`threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema`

The inventory artifact and supplemental audit artifact are therefore not
included in `B`. Record the regular-file count used for `B`.

Use these exact binary units:

- `1 MiB = 1,048,576 bytes`
- `1 GiB = 1,073,741,824 bytes`
- `4 GiB = 4,294,967,296 bytes`

Compute:

`replay_plus_summary_bytes_per_game = M + 1 MiB`

`first_120_game_increment_bytes = 120 * (M + 1 MiB)`

`pre_overhead_total_bytes = B + first_120_game_increment_bytes`

`P = ceil(1.25 * pre_overhead_total_bytes)`

The `1 MiB` per game conservatively covers compact summaries and manifests.
The `25%` multiplier covers filesystem and serialization overhead.

The supplemental storage gate passes if and only if both conditions hold:

- `P < 4 GiB` (strict inequality);
- current free disk is `> 120 GiB` (strict inequality), measured after the
  inventory has been written and before the terminal audit is sealed.

Report `B`, its regular-file count, `M`, every formula term, `P` in bytes and
GiB, headroom to `4 GiB`, current free bytes/GiB, replay count, maximum
path/hash/bytes, top ten entries, and the immutable inventory file and canonical
payload hashes.

## Operational Truth

Before sealing the terminal audit:

- verify the admission marker/result file hashes above are unchanged;
- verify the v1 marker/HOLD evidence is unchanged;
- verify ports `8765` and `8770`, advisor status, dashboard record `263670`,
  and protected top-three truth;
- record current free disk and the exact service/dashboard evidence;
- assert no supplemental inventory or audit artifact existed before this
  one-shot audit;
- assert zero new actions, timing assays, games, consumed streams, labels,
  fitted models, continuations, score inspection, policy outcomes, incumbent
  changes, and dashboard changes.

Write exactly one immutable terminal artifact:

`threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema/QD_V2_STORAGE_ADMISSION_AUDIT.json`

Its only supplemental decision is:

- `READY_QD_STORAGE_ADMISSION` if every inventory, integrity, service, and
  storage check passes; or
- `KILL_QD_STORAGE_ADMISSION` otherwise.

The terminal artifact records its canonical payload SHA-256, calculated over
the payload before adding the self-hash field. Both supplemental artifacts are
written atomically and must never be overwritten. The original marker and
admission result remain byte-for-byte unchanged.

Regardless of the supplemental decision, acquisition, G1-R, incumbent
promotion, and dashboard promotion remain held for oversight.

# Artifact Retention

The training workspace should retain scientific evidence without keeping every
full model snapshot indefinitely.

## 2026-07-26 K1 Support Audit / Exact Depth-3 KILL

- Preserve `K1_SUPPORT_AUDIT_CHARTER.md`, `k1_support_audit.py`, and
  `tests/test_rl_k1_support_audit.py` at SHA-256
  `a091900a...2e27`, `f04b9934...d2da`, and `e08ed580...3836`.
  Py-compile, `10/10` focused tests, and `122` relevant regressions passed;
  four obsolete pre-execution artifact-absence assertions were deselected.
- Preserve the complete compact
  `runs/forensics/k1_support_audit_v1` directory and
  `K1_SUPPORT_AUDIT.json` at file SHA-256 `536fd76d...3111`.
  Its embedded `949bd408...fc3f` payload hash is a preserved serialization
  defect, not the authoritative integrity seal.
- Preserve
  `K1_SUPPORT_AUDIT_V2_SERIALIZATION_AMENDMENT.md`,
  `k1_support_audit_seal_v2.py`, and
  `tests/test_rl_k1_support_audit_seal_v2.py` at SHA-256
  `6e58aeb4...dd72`, `0e171a15...793c`, and `62c765a8...e2b4`.
  Py-compile and `16` applicable tests passed.
- Preserve the one-file
  `runs/forensics/k1_support_audit_v2` directory. The authoritative v2
  file/payload SHA-256 are `9b4da740...3e09` /
  `3a27d2c1...b5e5`; its unchanged scientific-payload SHA-256 is
  `171c0b09...833b`. V2 was generated solely from the immutable v1 JSON and
  reopened zero replays.
- The result binds all 108 completion rows, 24 retained replay/state pairs,
  source hashes, whole-root/stream integrity, immutable family-admission
  evidence, and unopened C2/K1 gate status. It contains only support counts,
  root/frame/state-hash identities, and feasibility evidence.
- Observed roots with at least one qualifying state were `4/11/9`; all 24
  observable roots actually had at least four states. The remaining `84`
  roots lack immutable replay content and may never be imputed. Alternative
  expectimax2/QD-v2 signatures have zero K1-compatible immutable trigger-root
  support.
- Decision is permanently `KILL_EXACT_DEPTH3_PROGRAM`. K1-v1, its marker, and
  73B-76B streams remain spent. Do not rerun, reacquire, compile, retime, or
  repurpose the sealed C2/K1 untouched gates.
- The audit generated no game, stream, compilation, timing, depth-3 value,
  score/action outcome, label, model, incumbent artifact, dashboard artifact,
  or promotion. Free disk remained `149.260 GiB`; no cleanup is warranted.

## 2026-07-26 K1 Compiled-Kernel Corpus HOLD Retention

- Preserve `K1_COMPILED_EXACT_KERNEL_CHARTER.md`, amendments A1-A5,
  `k1_exact_kernel.c`, `k1_compiled_kernel.py`, `k1_engineering.py`, and
  `tests/test_rl_k1_compiled_kernel.py`. Final native/wrapper/runner/test
  SHA-256 are `e47af279...b41e4f`, `fe2ee430...c5af8`,
  `e4b360f4...29615c`, and `ae180434...ad71a`.
- Preserve all K1 design preflights, superseded test-evidence files, and the
  three failed zero-work staging directories. They document the offline
  linker repair and the conservative stream-reservation/collision
  classification repairs; they contain no fresh game or timing evidence.
- Preserve the complete sealed
  `runs/forensics/k1_compiled_kernel_v1` directory byte-for-byte. It is
  approximately `14 MiB` and contains `62` pre-terminal files totaling
  `14,298,243` bytes under artifact-manifest SHA-256
  `39e03d71...2208`.
- Retain preflight lock `257ac50a...8a6a`, marker `889aa515...c7e0`,
  terminal `157d73f7...e7a6`, completion rows `c99685e2...2021`, build
  manifest `27bdf707...37b`, and native library `24b20d06...ff56`.
  The preflight/marker/terminal payload SHA-256 values are
  `8fde35a1...ae3`, `0e87dd45...db19`, and `3907532f...2287`.
- The one-shot acquisition completed 108 unique fresh roots and unique
  streams but retained only `4/11/9` trigger-qualified roots for corner2,
  parent MC1000, and replaycal against the frozen `12/12/12` requirement.
  Preserve every retained source replay/state record because no promoted
  corpus manifest was created.
- Decision is `HOLD_K1_ENGINEERING_FAULT` at corpus construction. The fresh
  exactness/runtime gate never opened, so this is not kernel correctness,
  runtime, or policy evidence. The run may not be resumed or rerun under its
  spent marker.
- No policy outcome, h10/h20/h40 result, score inspection, label, training
  model, incumbent artifact, dashboard artifact, or promotion was produced.
  Terminal free disk was `149.280 GiB`; no cleanup is warranted.

## 2026-07-26 C2 Cost-Admission Terminal Retention

- Preserve `C2_COST_ADMISSION_EXECUTION_CHARTER.md`,
  `c2_cost_admission.py`, and `tests/test_rl_c2_cost_admission.py` at
  SHA-256 `d274c680...a7f4`, `436f2956...7be1`, and
  `a36c3219...1b11`. Retain test evidence payload
  `dd7006a1...e098`; py-compile, `19/19` focused tests, and `272` applicable
  historical regressions passed.
- Preserve the failed preflight staging directory
  `runs/forensics/c2_cost_admission_v1.staging.93590` as compact engineering
  evidence of the repaired historical signature-key mismatch. It contains no
  game or timing work and must not be mistaken for the promoted run.
- Preserve the complete sealed
  `runs/forensics/c2_cost_admission_v1` directory byte-for-byte. It is
  approximately `18 MiB` and contains the fresh source corpus, exact selected
  states, fit/validation timing rows, cost model, validation report, marker,
  and terminal result. Do not delete the untouched-gate states; they document
  that the gate remained sealed.
- Retain preflight lock `1f7dd905...1b4f`, marker `9b606f88...4817`,
  corpus manifest `2be3db6c...3653`, model `80aafb29...09299`, validation
  report `5e2a8441...0ae0`, and terminal result `ac1e3b49...cb9f`.
  Their canonical payload hashes are respectively `a8db1b03...56c6`,
  `15d70810...85a9`, `817f34f7...6aea`, model identity
  `f91adeb4...5cf2`, `28fe6584...5abf`, and `8acc25e4...a285`.
- Decision is permanently `KILL_C2_COST_ADMISSION`: conservative root
  coverage was `87.5%` versus the frozen `90%` safety floor. The untouched
  48-state runtime gate and all policy-outcome work remained unopened.
- Terminal free disk was `149.340 GiB`; no cleanup is warranted. The C2
  evidence is not dashboard-eligible and does not modify the incumbent or
  protected top-three replays.

## 2026-07-26 G4 Conditional-Pairwise Preflight Retention

- Retain `G4_CONDITIONAL_PAIRWISE_ACTION_RANKER_CHARTER.md`,
  `g4_conditional_pairwise.py`, and
  `tests/test_rl_g4_conditional_pairwise.py` at SHA-256
  `765992cc...247e`, `d7fef45b...2c27`, and `67d2b91a...0416`.
  Py-compile, `12/12` focused tests, and `107/107` applicable regressions
  passed. The one deselected historical assertion requires an authorized,
  already-existing G3-v2 output directory to remain absent.
- Preserve the complete compact
  `runs/forensics/g4_conditional_pairwise_v1` directory byte-for-byte. It is
  approximately `3.7 MiB` and contains only the immutable preflight, pair
  audit, fail-closed untouched inventory, and unused future stream manifest.
  No diagnostic marker or model exists.
- Retain `G4_PREFLIGHT.json` at file/payload SHA-256
  `bad6ca95...5db` / `4c0bc125...a40e`,
  `G4_PAIR_MANIFEST.json` at file SHA-256 `5acad327...02a`,
  `G4_FUTURE_STREAM_MANIFEST.json` at `d264d0b4...c86f`, and
  `G4_UNTOUCHED_INVENTORY.json` at `16cf3a33...5d6a`. The exact pair dataset
  identity is `ade1040d...484d`.
- The decision is `KILL_G4_PAIRWISE_INFEASIBLE` for this exact v1 preflight.
  All scientific/data integrity gates passed, but only `8` informative
  pre1536 development roots were available against the frozen floor of `12`.
  The seal also contains a sandbox-local operational-health failure; a
  subsequent read-only host audit confirmed services/process health, but the
  immutable seal is not rerun or edited. The conditional-pairwise model was
  not fit and its mechanism remains untested.
- Retain the aggregate support evidence: `727` spent discordant pairs
  (`552 train`, `175 development`), with `146/39` informative roots and zero
  G3 transfer access. Preserve the G3 ordinary database at its existing
  location/hash; G4 does not duplicate it.
- The prospective 61B/62B/63B/64B stream manifest contains `512` roots and
  `2,048` unique IDs, zero collisions against `8,950` historical sources, and
  zero consumed streams. It is planning evidence only and does not authorize
  evaluation.
- No simulation, new label, transfer path/prediction, fitted model, policy
  evaluation, score analysis, incumbent artifact, dashboard artifact, or
  promotion was produced. Free disk remained about `149.42 GiB`; no cleanup is
  warranted.

## 2026-07-26 G4-v2 Spent Cross-Fit Terminal Retention

- Retain `G4_V2_SPENT_CROSSFIT_AMENDMENT.md`,
  `g4_conditional_pairwise_v2.py`, and
  `tests/test_rl_g4_conditional_pairwise_v2.py` at SHA-256
  `080d7bf3...bca`, `f9113297...681`, and `d51c63b0...f3f`.
  Py-compile, `14/14` focused tests, and `121/121` applicable regressions
  passed. Retain test evidence at file/payload SHA-256
  `92185c2b...8df` / `a6633109...828`.
- Preserve the complete compact
  `runs/forensics/g4_conditional_pairwise_v2` directory byte-for-byte. It is
  approximately `500 KiB` and contains the immutable prefit lock, fold
  manifest, execution marker, five fold models, OOF predictions, and terminal
  result.
- Retain the prefit lock at file/payload SHA-256 `9b969505...6e87` /
  `01be9aa5...700d`, fold manifest at `18ddd4fd...9312`, execution marker at
  `8b9f8e6a...034`, OOF predictions at file/payload SHA-256
  `d9febb07...3828` / `932c7634...92ea`, and terminal result at file/payload
  SHA-256 `0426b562...09c3` / `e0ebebe4...0569`.
- Retain all five fold model array/meta files. Their array/meta SHA-256 pairs
  are `26397002...2ac` / `9bf70d72...07a`,
  `45d080b0...202` / `24b84833...90e`,
  `2af0b1dd...a95` / `78008b1a...848`,
  `95dc554d...a8b` / `d313d913...4fa`, and
  `977bab41...943` / `bb04d5f0...922`. These are the sole exact attribution
  artifacts for the frozen five-fold mechanism test.
- The decision is permanently `KILL_G4_PAIRWISE_MECHANISM`: primary
  root-direction concordance was `0.5189`, 95% whole-root bootstrap interval
  `[0.4514, 0.5838]`, versus the frozen material floor `0.6059`.
  Pre1536 direction was `0.4778`, and only one of three support-eligible
  families exceeded chance.
- This directory contains no simulation, new label, transfer path/prediction,
  score analysis, policy construction/evaluation, incumbent artifact, or
  dashboard artifact. G3 transfer access remained zero. Free disk remained
  about `149.4 GiB`; no cleanup is warranted.

## 2026-07-25 G3 E0 label/fit preflight v1 KILL and v2 READY

- Retain the base E0 charter and compact-manifest v2 amendment at SHA-256
  `78c7a836...0591` and `1b0594d5...15fe`. Retain runner, v1/v2 preflight,
  and focused-test sources at `19d74a31...8b5`, `bb1c09b7...c627`,
  `05da1916...dd0e`, `cd611547...87b1`, and `754f8d6b...5829`.
- Retain test evidence at `0d4511eb...6ca9` for v1 and
  `8f062d3e...87e9` for v2. The latter binds py-compile, `39/39` focused
  checks, and `113/113` frozen regressions.
- Preserve the entire failed
  `runs/forensics/g3_e0_label_fit_v1` directory (`12 MiB`) byte-for-byte.
  Its immutable lock file/payload SHA-256 are `73eee861...fa94` /
  `af4dcda1...e45c`. The sole failure was the compact-record adapter's
  `KeyError: 'state'`; it is a spent zero-outcome engineering false positive
  and may never be rerun, edited, or used for execution.
- Preserve the complete corrected
  `runs/forensics/g3_e0_label_fit_v2` directory (`11 MiB`) byte-for-byte.
  Decision is `READY_G3_E0_LABEL_FIT_EXECUTION` with terminal execution HOLD.
  Lock file/payload SHA-256 are `fde44d08...2ef7` /
  `c7ec0d0d...e2bb`; record/stream/task manifest SHA-256 are
  `90a4f55f...9d5`, `e40b7dd3...a05`, and `087fd68c...bd2f`.
- The v2 directory is compact manifest/readiness evidence only. It contains
  zero label database, model checkpoint, prediction, transfer outcome, policy
  evaluation, score inspection, incumbent artifact, or dashboard artifact.
  No cleanup is warranted.
- The reused `57B/58B/59B/60B` reservation remains unconsumed. Preserve all
  source replays, incumbent checkpoint components, G3-v1/v2 manifests, and G2
  transfer sources bound by the lock. Post-seal free disk was `150.294 GiB`.

## 2026-07-25 G3 E0 v3 orchestration READY / spent open HOLD

- Preserve the v3 open/resume amendment at SHA-256
  `d201e4a1...38ab`. Preserve orchestration runner, preflight, focused tests,
  and test evidence at `db512549...8392`, `4672e769...3890`,
  `969acc53...f802`, and `8c7c6de8...18d5`.
- Preserve the complete compact
  `runs/forensics/g3_e0_label_fit_v3` directory byte-for-byte. Its preflight
  decision remains `READY_G3_E0_V3_EXECUTION`, with terminal status
  `HOLD_G3_AFTER_E0_V3_PREFLIGHT_SEAL`. Preflight file/payload SHA-256 are
  `ac4f4e74...57b3` / `25f3686d...5af7`.
- Preserve `HOLD_G3_E0_V3_OPEN_REVALIDATION.json` at file/payload SHA-256
  `f1f69132...378d` / `a2d06ce9...3e6d`. It records the fail-closed live
  dashboard source-list hash mismatch before marker creation. V3 is spent and
  may not be retried or executed.
- V3 record/task/stream manifest SHA-256 are
  `90a4f55f...9d5`, `087fd68c...bd2f`, and `e40b7dd3...a05`, exactly
  matching v2. Do not deduplicate these copies while v3 execution is pending:
  their paths and hashes are directly bound by the future marker contract.
- Preserve the v2 READY directory unchanged as the source and historical
  orchestration-HOLD evidence. It has no marker or work artifact and may not be
  executed. V3 is a separate preflight identity, not a rewrite of v2.
- The v3 directory contains only three scientific manifests, one preflight
  lock, and the compact engineering HOLD. It contains no execution marker,
  database, label, model, checkpoint, prediction, scientific outcome, score,
  incumbent artifact, or dashboard artifact.
- Preserve every source replay and incumbent component bound by the 715-record
  restore and 8,927-source collision audits. The 57B/58B/59B/60B reservation
  remains unconsumed. Post-seal free disk was `149.876 GiB`; no cleanup is
  warranted.

## 2026-07-25 G3 E0 v4 collision preflight READY / execution hold

- Retain the v4 collision amendment, runner, preflight, tests, and test
  evidence at SHA-256 `cbae9910...a628`, `4c564023...4ee0`,
  `bca7c32a...aba`, `04f67861...27a7`, and `b9d38372...1f14b`.
  Test evidence binds py-compile, `17/17` focused, `66/66` applicable G3,
  and `113/113` frozen regression checks.
- Preserve the complete
  `runs/forensics/g3_e0_label_fit_v4` directory byte-for-byte. Its preflight
  decision is `READY_G3_E0_V4_EXECUTION`, terminal status
  `HOLD_G3_AFTER_E0_V4_PREFLIGHT_SEAL`. Lock file/payload SHA-256 are
  `fdc801f5...a9e7` / `8937c359...8c12`.
- Preserve `E0_COLLISION_SOURCE_MANIFEST.json` at file/payload SHA-256
  `c62b2d04...28d1` / `f99b6448...20f`. It binds `8,926` immutable
  external sources (inventory SHA `e6a43f52...aca20`), exactly the two live
  dashboard summaries, `17` inherited internal sources at preflight, requested
  stream contract `fbbfd9fc...fc7f`, and zero requested-stream collisions.
- Preserve the byte-identical record/task/stream manifests at
  `90a4f55f...9d5`, `087fd68c...bd2f`, and `e40b7dd3...a05`, plus all
  source replays and incumbent components bound by the lock. The
  57B/58B/59B/60B reservation remains unconsumed.
- At the v4 preflight seal, it contained no execution marker, database, path,
  label, model, checkpoint, prediction, transfer outcome, policy outcome,
  score inspection, incumbent artifact, or dashboard artifact. Seal-time free
  disk was `149.706 GiB`; the later separately authorized execution is
  recorded below.

## 2026-07-25 G3 E0 v4 terminal predictive KILL retention

- Preserve the complete sealed
  `runs/forensics/g3_e0_label_fit_v4` directory byte-for-byte. The directory is
  `21,104,381` bytes and contains the preflight evidence, immutable execution
  marker, complete ordinary-label database, one frozen model, checkpoint seal,
  and terminal result.
- Retain `G3_E0_EXECUTION_OPENED.json` at file/payload SHA-256
  `91b38116...a76b` / `72ebce43...3331`,
  `G3_E0_CHECKPOINT_SEALED.json` at `a9ac34d3...e58b` /
  `6d44ca18...9568`, and `G3_E0_TERMINAL_RESULT.json` at
  `e7ca390f...ddd1` / `a70457df...5bfa`.
- Retain `ordinary_labels.sqlite3` at SHA-256 `d0954a91...820`; it is the
  sole E0 label corpus and contains exactly `4,846` completed ordinary paths
  (`3,902 train`, `944 development`). Retain model arrays/meta at
  `bc5d9830...6fcb` / `ffb9980a...a8e6` and the checkpoint-embedded ordinary
  report, canonical SHA-256 `7965bea1...dc8`.
- The exact terminal decision is `KILL_G3_BOOTSTRAP_PREDICTIVE`. Preserve this
  as evidence that the scale-transfer model improved marginal hazard losses
  but failed legal-action ranking and family robustness. It is not capability
  evidence and is never dashboard-eligible.
- No transfer prediction seal or `transfer_labels.sqlite3` exists; all `226`
  transfer paths and transfer outcomes remained unopened. No E1 artifact,
  reranker, normal-start evaluation, score claim, incumbent payload, or
  dashboard artifact was created.
- Final free disk was `149.554 GiB`; no cleanup is warranted. Keep all
  v1-v4, G2, G1, S3, QD, incumbent, and protected replay evidence unchanged
  pending course review.

## 2026-07-25 G3 V2 integrity preflight READY / execution hold

- Retain the v2 integrity amendment, implementation, tests, and test evidence
  at SHA-256 `c60895f9...6aef`, `b8488f2e...e526`,
  `9e81e2be...50a3`, and `4ec431b3...0d8`. Test evidence binds py-compile,
  `17/17` focused tests, and `113/113` relevant regressions.
- Retain the complete compact
  `runs/forensics/g3_scale_transfer_bootstrap_preflight_v2` directory
  (`3.5 MiB`) byte-for-byte. Its decision is
  `READY_G3_V2_BOOTSTRAP_LABELS`, with terminal HOLD on execution.
- Preflight file/payload SHA-256 are `052985f7...7d66` /
  `18539079...93ab`. Corrected untouchedness file/payload SHA-256 are
  `8b2fd76e...51ed` / `f9e7079f...ba7e`; panel-input-binding file/payload
  SHA-256 are `374fe010...2ca3` / `c6a19325...688b`.
- Preserve the exact 65-file panel binding table and corrected scan evidence:
  all G2 inputs and reused v1 evidence reproduce, true external matches are
  zero, and no broad namespace exclusion is used.
- Preserve the E0/E1 staged-cost report. E0 is `5,072` paths and E1 is
  `15,216`; both remain unauthorized. READY is not label, fit, policy,
  capability, or promotion evidence and is never dashboard-eligible.
- V2 generated zero game, consumed stream, label, label value, fit, transfer
  outcome, candidate action, continuation, policy outcome, incumbent artifact,
  or dashboard point. There is nothing to prune. Post-seal free disk was
  `150.382 GiB`.

## 2026-07-25 G3 V1 bootstrap preflight terminal hold

- Preserve both frozen contract files:
  `G3_SCALE_TRANSFER_BOOTSTRAP_CHARTER.md` SHA-256
  `e216aa50...45fc1` and
  `G3_SCALE_TRANSFER_BOOTSTRAP_CHARTER_AMENDMENT_A1.md` SHA-256
  `baba7200...2b94`.
- Preserve implementation/test/test-evidence SHA-256
  `27ac9cc6...5234`, `d9c60a84...a14a`, and `a8f6a176...bd21`.
- Preserve the complete sealed
  `runs/forensics/g3_scale_transfer_bootstrap_preflight_v1` directory
  (`15 MiB`) byte-for-byte. Its terminal decision is
  `KILL_G3_PREFLIGHT_INTEGRITY`; preflight file/payload SHA-256 are
  `0cd19d5a...d77a` / `4513530c...06ad`.
- Preserve record and stream manifests at SHA-256
  `938e903f...b9ec` / `bdbe5621...5744`. They are the sole exact
  outcome-free accounting of `20,288` required paths and the unconsumed
  `57B/58B/59B/60B` reservation.
- Do not edit or rerun this v1. Its false-positive untouchedness failure is
  useful engineering evidence: all reported matches are inside the exact G2
  source directory that its glob intended to exclude. The separately retained
  v2 evidence above is the only correction.
- Preserve the entire G2 acquisition directory and its 32 replay/state source
  pairs. G3 created zero labels, fits, outcomes, or derivative payloads, so
  there is nothing to prune.
- Post-seal free disk was `150.319 GiB`; no cleanup is required.

## Keep

- Every checkpoint referenced by `current_incumbent_policy.txt`.
- The latest checkpoint for a promoted policy and its direct training parents.
- Configs, summaries, metrics, manifests, source replay JSON, and experiment-log
  conclusions for completed experiments.
- Provenance-safe human, fresh-root, and rare-event frontier trajectories.
- Small diagnostic artifacts needed to reproduce reported comparisons.

## Prune

- `checkpoint_game_*` directories after a completed run has a valid `latest`
  checkpoint, unless an intermediate checkpoint has a documented result.
- Learned tables for variants explicitly marked killed or not promotion-worthy
  after their config, summary, metrics, and decision have been retained.
- Test scratch runs and generated HTML when they become material storage users
  and equivalent JSON remains available.

Before pruning, verify that the artifact is not being written, is not referenced
by the incumbent policy, and is not the sole copy of a source replay or result.
Prefer keeping at least 100 GiB of free disk space during active research.

## 2026-07-07 Cleanup

- Removed 42 completed-run `checkpoint_game_*` snapshots while retaining each
  run's `latest` checkpoint and evidence artifacts: about 63.5 GiB recovered.
- Removed about 31.7 GiB of learned tables from 12 explicitly rejected big6,
  suffix-calibration, and capacity-probe variants. Their run directories,
  configs, summaries, metrics, and logged conclusions remain.
- Workspace usage fell from about 145 GiB to 50 GiB. Free disk space increased
  from about 53 GiB to 148 GiB.
- Verified that all four checkpoint components referenced by
  `current_incumbent_policy.txt` still contain `meta.json`.

For exploratory training, use no periodic checkpoints when practical. For long
runs that need recovery points, keep only the latest recovery checkpoint plus
the final `latest` model after completion.

## Replay Retention Audit

Use the dry-run replay audit before deleting replay artifacts:

```bash
.venv/bin/python -m threes_rl.replay_retention_audit \
  --out threes_rl/runs/dashboard/replay_retention_audit.json \
  --print-summary
```

The audit protects the global top-three normal-start full-game replays using
the same eligibility rules as the dashboard. Continuation, milestone,
diagnostic, and human-diagnostic replays are reported separately as scientific
evidence and are not prune candidates in this audit. The report is
non-destructive; any deletion still requires explicit review.

Stable copies of the current global top-three normal-start replay set live in:

```bash
threes_rl/runs/replays/top3/index.html
```

The dashboard watcher syncs this playlist automatically when the global
top-three signature changes. You can also regenerate it manually with:

```bash
.venv/bin/python -m threes_rl.top_replay_playlist
```

Treat `threes_rl/runs/replays/top3/` as protected replay evidence. The sync
step rewrites current stable copies but intentionally does not delete old
ranked directories; stale-copy pruning remains an explicit retention-review
task.

## 2026-07-08 Support-Frontier Cleanup

- Removed 134 large reproducible support-accumulation frontier archive files:
  full `records.json`, full `transitions.json`, and duplicated
  `support_accumulation_frontier.json` payloads over 10 MB.
- Kept every `summary.json`, `depth_rows.json`, `target_records.json`, HTML
  summary, source state-record pool, and experiment-log/report conclusion.
- Verified before pruning that every affected run directory retained the small
  evidence files above and that no frontier process was still writing.
- Recovered about 17.37 GiB. `threes_rl/runs/forensics/support_accumulation_frontier`
  fell from about 18 GiB to 195 MiB; `threes_rl/runs` fell from about 67 GiB
  to 50 GiB. Free disk after cleanup was about 133 GiB.
- After the later adjacent-384 prospective screen, removed four additional
  full archive/transition payload sets while retaining `summary.json`,
  `depth_rows.json`, `target_records.json`, and HTML summaries. This recovered
  about 1.07 GiB; no large files remain in those four run directories.

## 2026-07-09 Restart Preflight Cleanup

- Wrote the deletion manifest before removal at
  `runs/forensics/restart_preflight/r0_preflight_20260709/preflight_and_deletion_manifest.json`
  with a CSV companion.
- Verified no training/evaluation writer was active; only the dashboard watcher
  was running. All four incumbent components, the parent MC1000 checkpoint,
  protected top-three replays, retained source replays, and evaluation manifests
  were excluded.
- Removed only `.npy` learned-table payloads from seven explicitly rejected or
  non-promoted models: four failed phase4 action-label corrections, the failed
  phase4-corner3 label correction, the historical failed 80-game
  phase4-corner3 TD model, and the killed exact-rung reachability sidecar.
- Retained configs, summaries, metrics, progress artifacts, metadata, source and
  top-game replays, compact paired/continuation evidence, and the logged
  conclusions for every affected run.
- Reclaimed `26,197,880,832` bytes. `threes_rl/runs` fell from about `51 GiB`
  to `26 GiB`; free disk rose from `126 GiB` to `150 GiB`.
- Active policy: target at least `120 GiB` free and pause heavy training below
  `100 GiB`. R1 uses `checkpoint_every=0` and retains latest/final only, with
  checkpoint disk delta reported at every gate.

## 2026-07-09 R1 Harm-Stop Retention

- The stopped R1 run retains one `latest` checkpoint plus config, metrics,
  summaries, progress, top replays, and compact audits. It has no periodic
  `checkpoint_game_*` snapshots and occupies about `3.1 GiB`.
- Compact failure evidence is retained at
  `runs/forensics/restart_program/r1_pilot_1000_failure_audit_20260709.json`
  (`16 KiB`), with the validated D0 phase-state sample beside it (`184 KiB`).
- The parent-only D0 attribution artifact occupies `12 KiB`. No D1 or C
  candidate artifact was created.
- Post-audit `threes_rl/runs` usage is about `30 GiB`; free disk is `149 GiB`,
  above the `120 GiB` target. No further pruning is needed before review.

## 2026-07-09 R1b Pilot Retention

- R1b retains one `latest` residual-composite checkpoint plus compact identity,
  100-episode, 1,000-episode, D1, config, metrics, and replay evidence. It has no
  periodic checkpoint snapshots and occupies about `3.1 GiB`.
- The four incumbent components are referenced read-only rather than copied
  into R1b. Their file-stat fingerprints were unchanged after 1,000 episodes.
- Original R1 has now served its attribution purpose and its 3.1 GiB learned
  tables are eligible for a future deletion-manifest review. Keep its compact
  D0 audit, config, metrics, summary, and conclusions regardless. No deletion
  was performed because current free space is `144 GiB`, above target, and the
  proposed R1b continuation is still awaiting authorization.
- No R1b D2 or C artifact exists. D0 is retained as pre-update identity
  evidence; D1 is retained as the sole 1,000-episode development gate.

## 2026-07-09 R1b D2 Retention Review

- R1b stopped exactly at 5,000 and retains only `latest`; there are no periodic
  snapshots. The fixed-size checkpoint is `3,334,477,906` bytes, a five-byte
  metadata increase from 1,000. Its run directory occupies about `3.1 GiB`.
- The complete candidate D2 evidence, including 512 paired rows, analysis,
  hashes, and 46 fixed worst-case replay captures, occupies about `17 MiB`.
  C remains sealed and no C candidate artifact exists.
- The killed original R1 checkpoint also occupies about `3.1 GiB`. Its tables
  are still eligible for pruning, but it remains the sole full attribution
  snapshot contrasting bare-parent and incumbent-residual initialization.
  Retain it through the separate pre-C decision; compact evidence alone will
  be sufficient after that decision is closed.
- Total `threes_rl/runs` usage is about `33 GiB` and free disk is `143.9 GiB`,
  above the `120 GiB` target. No destructive cleanup was performed.

## 2026-07-10 R1b Confirmation Retention

- Retain the exact incumbent and R1b C CSVs, summaries, paired analysis,
  pre/postflight hashes, baseline/confirmation locks, fixed tail audit, and its
  46 compact replay captures. They are the sole untouched-confirmation record.
- R1b failed confirmation and was not promoted. The original killed R1 and R1b
  learned tables remain retained pending the required research review; do not
  prune attribution checkpoints while the branch interpretation is unresolved.
- No dashboard, incumbent, or protected top-three replay artifact changed.
  Storage remains above the `120 GiB` target, so no deletion was performed.

## 2026-07-10 Human H0 Retention

- Retain all ten completed exact human replay/session pairs and the excluded
  active-session record under `datasets/human_play`; the dataset is about
  `6.5 MiB` and is the sole source for this independent behavioral family.
- Retain the hashed corpus manifest, frozen 48-root manifest, H0 stream/action
  manifest, preregistration, preflight hashes, compact result CSV/analysis, and
  fixed two-case replay audit. Do not retain all continuation trajectories.
- The active H0 forensic directory was about `1.0 MiB` at this checkpoint.
  `threes_rl/runs` remains about `33 GiB`, with `146 GiB` free, above the
  `120 GiB` target. No deletion or checkpoint pruning was needed.
- Human normal-start and restart-play artifacts are always development-only
  and explicitly dashboard-record-ineligible. They cannot replace protected
  confirmed normal-start replays.
- Retain the single frame-286 API smoke only as compact recorder provenance
  evidence. It is tagged `calibration-discard`, contains one frame and no
  played action, and is excluded from every H0 outcome analysis.
- Final H0 action-conditioned artifacts occupy about `23 MiB`; free disk is
  `146 GiB`. Retain `results.csv`, `analysis.json`, `summary.json`, the frozen
  manifest/preregistration locks, and `fixed_replay_audit.json`. Their final
  hashes are respectively `37bbba01...aaade4c`, `5fec9c51...cb106e`, and
  `7b95da50...aa83de` for results, analysis, and replay audit.
- The append-only `checkpoint.jsonl` is the reproducible task-level source for
  the compact CSV and remains small enough to retain. No trajectory expansion,
  deletion, or cleanup is needed.
- Final H2 context artifacts occupy about `2.7 MiB`; the complete human H0/H2
  forensic directory is about `26 MiB`, with `146 GiB` free.
- Retain the H2 preregistration lock, frozen manifest, compact checkpoint/CSV,
  analysis, summary, and fixed replay audit. Final hashes are
  `d2e082ca...1c186` for results, `b0a83719...a6a9` for analysis, and
  `ba84c3cf...59d9` for the replay audit.
- No full continuation corpus, model checkpoint, or policy evaluation was
  created. No cleanup is needed, and all dashboard/protected artifacts remain
  unchanged.

## 2026-07-11 R1.5a Preflight Retention

- Retain the frozen preregistration, `R15A_PREFLIGHT_STOP_GO.json`, model config,
  and natural-state inventory. They are the sole evidence for the source-
  readiness failure and contain no generated labels or fitted parameters.
- The R1.5a forensic directory is about `4.3 MiB`; total runs remain about
  `33 GiB`, with roughly `145 GiB` free. No pruning is needed.
- The inventory contains compact selected natural states and source hashes only;
  underlying replay sources remain protected as provenance evidence. Synthetic
  H2 swaps remain in their existing diagnostic partition.
- No heavy output, periodic checkpoint, label corpus, fit checkpoint, policy
  evaluation, dashboard point, or replay playlist change was created.

## 2026-07-11 R1.5a Amendment A1 Retention

- Retain `R15A_AMENDMENT_A1_20260711.md`, the A1 inventory implementation and
  focused test, the complete amended natural-state inventory, and
  `R15A_A1_PREFLIGHT_STOP_GO.json`. Together they are the sole evidence for the
  weighted-ESS readiness stop and contain no rollout or model outcomes.
- Preserve the original R1.5a preregistration/inventory/stop artifact unchanged;
  A1 is a new namespace and does not overwrite the first failed preflight.
- The A1 forensic directory is about `7.1 MiB`; total runs remain about
  `33 GiB`, with about `145 GiB` free. No cleanup is needed.
- Underlying natural replay sources remain provenance-protected. No label
  corpus, model table, policy evaluation, normal-start replay, dashboard point,
  or periodic checkpoint was created, so there is no new heavy payload to
  prune.

## 2026-07-11 R1.5a Amendment A2 Retention

- Retain `R15A_AMENDMENT_A2_20260711.md` and
  `runs/forensics/r15a_context_a2/R15A_A2_READINESS_LOCK.json`. They are thin
  immutable references to the A1 source manifest and do not duplicate selected
  state payloads.
- A2 creates no new replay, label, model, or policy artifact at the readiness
  lock. Existing source manifests and both prior failed readiness artifacts
  remain protected.
- Retain the A2 execution lock, execution preflight, natural label manifest,
  and synthetic diagnostic manifest. During generation retain only append-only
  compact checkpoints, final label sufficient statistics, summaries, and the
  capped replay audits; never retain every trajectory as a replay.
- Natural labeling produced about `58 MiB`: retain `checkpoint.jsonl` as the
  resumable source, deterministic `labels.jsonl`, integrity summary, and the
  24-path fixed replay audit. No expanded replay set or periodic model
  checkpoint exists. Free disk after labeling is about `142 GiB`.
- Retain the fitting-ineligible synthetic label manifest/corpus, exact final
  model pair, fit summary, and `OFFLINE_GATE.json` as the complete R1.5a failure
  record. The entire A2 forensic directory is about `70 MiB`; no cleanup is
  needed. No policy eval, normal-start replay, or dashboard artifact exists.
- Retain the compact R2a preregistration, implementation/tests, and 64-root
  prescreen manifest. The R2a directory is about `340 KiB`; no rollout corpus
  exists because runtime failed before stream freezing.
- Retain `ASSISTED_HUMAN_INVENTORY_20260711.json` and its referenced completed
  replay/session pairs. Active sessions remain untouched and excluded. Total
  runs remain about `33 GiB`, with about `141 GiB` free; no cleanup is needed.
- Retain the C1 preregistration, preflight lock, frozen three-way engineering
  corpus, reference profiles, equivalence snapshots, and each coherent retained
  optimization benchmark. C1 generates no replay or score corpus.

## 2026-07-12 C1 Final Retention

- Retain the compact C1 corpus/lock, reference profile, frozen equivalence
  reference, iterative/persistent/batched/grouped/byte-key benchmarks, one-shot
  runtime gate, and `C1_STOP_GO.json`. Together they preserve the full bounded
  optimization sequence and the failed p99 decision without gameplay outcomes.
- The C1 forensic directory is under `2 MiB`; no C1 payload warrants pruning.
  No tables, replay trajectories, policy checkpoints, or periodic snapshots
  were generated.
- Runs remain about `33 GiB` with about `140 GiB` free, above both the `100 GiB`
  floor and `120 GiB` target. Incumbent components, R1b/C evidence, protected
  replays, R1.5a labels/models, R2a evidence, and human provenance remain
  protected and unchanged.
- Retain `C1_TAIL_MECHANISM_AUDIT.md` and
  `C2_COST_ADMISSION_PROPOSAL.md` as compact explanatory/proposal artifacts.
  They reuse existing traces, generate no payloads, and do not reopen C1.

## 2026-07-25 S3 Outcome-Free Hold Retention

- Retain the immutable S3 charter, outcome-free preflight implementation,
  focused tests, complete power tables, and all three audit generations:
  `S3_POWER_PREFLIGHT_V1_INCOMPLETE_EXCLUSION.json` documents the original
  ancestry-parser miss; `S3_POWER_PREFLIGHT.json` is the corrected scan with
  the coherence-reporting defect; and `S3_POWER_PREFLIGHT_V2_SEALED.json` is
  the authoritative repaired decision artifact.
- Retain `S3_PROVENANCE_SEAL.json` as the pre-repair supplementary snapshot and
  `S3_PROVENANCE_SEAL_V2.json` as authoritative. V2 preserves the complete
  `2,610`-root exclusion union with source categories and all `142` surviving
  compact candidate records from `133` roots, without duplicating state
  payloads.
- S3 generated no treatment outcome, continuation trajectory, policy
  checkpoint, replay, or dashboard artifact. The entire S3 forensic directory
  is about `760 KiB`; nothing warrants pruning.
- At seal time, `threes_rl/runs` was about `33 GiB` and free disk was
  `152.88 GiB`. The `100 GiB` floor and `120 GiB` target both pass. All prior
  incumbent, confirmation, C1, human, and provenance evidence remains
  protected and unchanged.

## 2026-07-25 G1/G1-R Outcome-Free Hold Retention

- Retain every G1 V1-V4 provisional/superseded power artifact and the
  authoritative
  `G1_EXISTING_CORPUS_PREFLIGHT_V5_AUTHORITATIVE.json`. Together they preserve
  the corrected beta-heterogeneity calibration, complete attempted power rows,
  zero-root retained-corpus result, and immutable 864-root acquisition target.
- Retain `G1R_NATURAL_ROOT_ACQUISITION_CHARTER.md`,
  `g1r_acquire.py`, its focused test, and the complete pilot-v1 preflight.
  The 3.8 MiB pilot-v1 directory contains only the immutable 4 MiB policy/
  history/action preflight and compact stop/go record.
- The preflight itself remains byte-for-byte unchanged at SHA-256
  `f78288b3...81ea91`; the stop/go is SHA-256 `4370efb6...797f1`.
  Preserve its complete `8,825`-source collision manifest and policy-table
  hashes despite their verbosity: they are the sole proof that the family hold
  occurred after artifact, stream, storage, service, and exactness checks.
- G1-R generated zero games, retained source replays, labels, models, policy
  outcomes, score outcomes, checkpoints, or dashboard artifacts. Nothing is
  eligible for pruning.
- Retain the initial QD proposal hash `63032ef7...1a513` in the experiment
  record as superseded proposal evidence. Retain the authoritative exact-
  contract proposal at SHA-256 `e9a72c65...bf880`. It remains proposal-only,
  creates no executable artifact or storage burden, and does not authorize a
  new family.
- Free disk at the hold was `152.806 GiB`; the dashboard record and protected
  top-three replays remain unchanged.

## 2026-07-25 QD Admission Lock And Terminal Hold Retention

- Retain the accepted implementation/test hashes, immutable execution lock,
  `489`-root archive, `489`-source provenance manifest, policy bundle, and
  preserved first preparation failure. They are the complete evidence chain
  from proposal through preparation.
- Retain `ADMISSION_OPENED.json` (file SHA-256 `f1faadcf...1f6e`) and
  `HOLD_QD_ADMISSION_ERROR.json` (file SHA-256 `205229ce...068b`) permanently.
  The marker proves the one-shot panel was opened; the terminal artifact proves
  it stopped in `reference_action_signatures` on `legal=0`.
- Never delete the marker and never rerun admission under this lock. No complete
  action-signature, pairwise, timing, thermal-after, exactness, or postflight
  artifact exists.
- The branch generated no games, labels, fitted models, continuation outcomes,
  score outcomes, checkpoint payloads, or dashboard artifacts. Nothing from
  this terminal HOLD is dashboard-eligible or safe to reinterpret as a policy
  result.
- Oversight's authoritative terminal decision is `KILL_QD_V1_EXECUTION`.
  Preserve the complete promoted `g1r_qd_admission_v1` directory byte-for-byte,
  including marker SHA-256 `f1faadcf...1f6e`, HOLD file SHA-256
  `205229ce...068b`, and HOLD payload SHA-256 `6bc74c73...cba0`.
- A future terminal-schema v2 must use a separate output directory, rebuilt
  archive, policy bundle, lock, versions, and hashes. It must never overwrite,
  repair, resume from, or prune v1 evidence. Retain the proposal at SHA-256
  `9a95f0b9...89cf`; retain pre-audit draft SHA-256 `b63e70d5...ea66` as
  superseded history.
- Retain the reviewed v2 implementation/test/charter hashes
  `191c612d...c51b`, `2e078039...da00`, and `d032c35c...81a0`, plus descriptor
  schema `a8cd1e15...a2c2`. They are implementation/test evidence only.
- V2 created no run directory, archive, policy payload, lock, marker, result,
  replay, label, model, continuation, score outcome, or dashboard artifact.
  This statement is superseded only by the preparation-only artifacts below.
- Retain the promoted v2 execution lock (file SHA-256 `1f48822f...ff4a`,
  payload SHA-256 `f6f6287b...af74`), its `489`-root archive/source manifests,
  and policy bundle as immutable preparation evidence. No action marker or
  result exists.
- V2 still has no generated replay, label, fitted model, continuation, score
  outcome, incumbent artifact, or dashboard artifact. Nothing in the v2
  preparation directory is capability evidence or eligible for pruning before
  the action-admission decision.
- Retain v2 `ADMISSION_OPENED.json` (SHA-256 `11b21137...5135`) and
  `admission_result.json` (SHA-256 `27bcb332...6a8e`, payload
  `0eced74a...38e2`) permanently with the lock/archive/policy chain. The
  one-shot lock is spent and may never be rerun.
- The result admits QD-v2 as a distinct action family only. It generated no
  acquisition replay, game, label, model, continuation, score outcome,
  incumbent artifact, or dashboard artifact. The 4.40 MiB admission directory
  is compact evidence and should not be pruned.
- Retain the supplemental storage charter
  `G1R_QD_V2_STORAGE_AUDIT_CHARTER.md` at SHA-256 `dd51e274...03070`,
  metadata-only runner at SHA-256 `96b57cd6...1dd1`, immutable replay inventory
  at file/payload SHA-256 `0dd9e2d4...e552` / `d5ebf1ba...aa03`, and immutable
  storage audit at file/payload SHA-256 `0bdef1de...037f` /
  `2864146d...bd78`.
- The inventory preserves metadata and raw-byte hashes for `3,242` qualifying
  pre-marker `replay.json` files but does not copy or parse them. It is compact
  provenance evidence and must not be regenerated or pruned while QD-v2
  acquisition remains under review.
- Supplemental decision `READY_QD_STORAGE_ADMISSION` is storage evidence only:
  `P=313,094,177` bytes (`0.29159 GiB`) versus the strict `4 GiB` cap with
  `152.714 GiB` free. It does not authorize acquisition, games, labels, models,
  outcomes, incumbent changes, or dashboard changes.
- Retain pilot-v2 QD5 charter/runner/tests at SHA-256
  `1f58d73b...e003`, `f1950260...d776`, and `85be02eb...a8de`.
  Also retain superseded charter SHA-256 `06ae8fa2...aebf` as documented
  provenance for the cross-stratum ancestry-uniqueness amendment.
- Retain all three pilot-v2 test-evidence files. The first two are superseded
  transcription evidence (`59b6fae9...64aa`, `a0080aae...67f0`); the
  authoritative third is file SHA-256 `c0804d7c...e393`, payload
  `74512d05...91c3`.
- Retain
  `runs/forensics/g1r_acquisition/pilot_v2_qd5/preflight_lock.json`
  permanently at file/payload SHA-256 `0d50edaa...22ad` /
  `1a0ca85b...7e67`. It is the sole promoted pilot-v2 artifact and binds the
  five policies, QD seals, tests, 100-row stream manifest, complete collision
  union, services, storage, and zero-work contract.
- The preflight lock is `3,658,296` bytes and contains no replay, game, label,
  model, continuation, score outcome, stream-consumption, or dashboard data.
  Do not prune, overwrite, or treat it as pilot execution/capability evidence.
- Retain the complete sealed pilot-v2 QD5 directory after its one authorized
  `100`-game acquisition run. In particular, retain
  `PILOT_V2_EXECUTION_OPENED.json` at file/payload SHA-256
  `a2fbd4a3...54d3` / `327517d1...2c2b` and `PILOT_V2_SEAL.json` at
  file/payload SHA-256 `75a11648...e57` / `b9588d39...8ecb`.
- Retain compact summary/completion/runtime evidence at SHA-256
  `512f65b9...bd90`, `b31912af...2d3b`, and `17853d41...bc03`, plus all `23`
  root-capped source replays required to reconstruct the selected states.
  Retained source/state verification passed `23/23`.
- The directory is about `11 MiB` on disk and contains no fitted model,
  action-label corpus, h40 outcome, continuation result, checkpoint, or
  dashboard artifact. No cleanup is warranted.
- Decision is `HOLD_G1R_AFTER_PILOT_V2_QD5_SEAL` because the frozen Wilson
  projection yields only `27` pre3072 roots against the required `432`.
  Preserve this as acquisition-feasibility evidence; it is not solver
  performance evidence and cannot update the incumbent or dashboard.

## 2026-07-25 G2 Outcome-Free Preflight Retention

- Retain `G2_SCALE_EQUIVARIANT_RELATIONAL_HAZARD_PROPOSAL.md` at SHA-256
  `43b413c1...7099`, implementation/preflight/test sources at
  `9ffaa45d...af8a`, `b5feebe5...2559`, and `3eca4551...9911`, and the exact
  schema identity `6af0cd51...340e`.
- Retain
  `runs/forensics/g2_scale_equivariant_relational_hazard_test_evidence.json`
  at SHA-256 `6319a294...6b25`. It records `13/13` focused, `76/76` relevant
  regression, and `89/89` combined passing tests.
- Retain the complete compact
  `runs/forensics/g2_scale_equivariant_relational_hazard` directory. Its
  immutable preflight file/payload SHA-256 are `2e1084f2...05cc` /
  `4d6fef61...55c9`; root-manifest file/payload SHA-256 are
  `60d514ed...a2ca` / `15ecb9d5...31ce`. The directory is approximately
  `2.3 MiB`; no cleanup is warranted.
- Preserve the source-manifest identity `71158f36...5349`, feature-row identity
  `bc98e816...73c75`, and explicit protected-overlap disclosures. They establish
  that all `98` natural pre3072 roots overlap prior protected evidence and that
  no untouched transfer root was available.
- The decision is `HOLD_G2_DATA_OR_POWER`, caused by transfer-data scarcity.
  The feature representation and prospective power calculation passed. These
  files are data-readiness and representation evidence only, not capability
  evidence and never dashboard-eligible.
- G2 produced no replay, generated game, stream, label corpus, rollout,
  h10/h20/h40 outcome, fitted model, checkpoint, candidate policy action,
  continuation result, score inspection, incumbent artifact, or dashboard
  artifact. Preserve all prior G1-R, QD, S3, R1b/C, and incumbent evidence
  unchanged.

## 2026-07-25 G2 Fresh Transfer Acquisition Preflight Retention

- Retain `G2_FRESH_TRANSFER_ACQUISITION_CHARTER.md`,
  `g2_fresh_transfer_acquire.py`, and
  `tests/test_rl_g2_fresh_transfer_acquire.py` at SHA-256
  `bebfbbce...526d`, `66ce0dea...7c23`, and `597a5a6e...6c40`.
- Retain test evidence at file SHA-256 `f20f7bfb...4316`; it records
  `20/20` focused and `85/85` established regressions.
- Retain the failed engineering staging directory
  `runs/forensics/g2_fresh_transfer_acquisition_v1.staging.11437` and its
  `PREFLIGHT_FAILURE.json` at SHA-256 `716fa96e...385e`. It is a 4-KiB
  fail-closed record of a panel-field reader mismatch and contains no game,
  stream use, label, or outcome.
- Retain the complete promoted
  `runs/forensics/g2_fresh_transfer_acquisition_v1` directory. Its immutable
  preflight lock file/payload SHA-256 are `5250e54d...cf44` /
  `18d9e851...5993`; the directory is approximately `4.2 MiB`.
- Preserve policy-lock `74c08c0c...f812`, action-audit
  `7217161f...31c4`, stream-manifest `8c5aefd3...c047`, and derived embedded
  historical-source-list `eead7a4e...c756` identities. The lock binds all
  policy/checkpoint/source artifacts and `8,859` historical collision sources.
- Decision `READY_G2_FRESH_TRANSFER_ACQUISITION` is source-acquisition
  readiness only. The reserved `53B/54B/55B/56B` streams are unused and no
  execution is authorized. This is not capability evidence and is never
  dashboard-eligible.
- No generated replay, game, consumed stream, label corpus, rollout,
  h10/h20/h40 outcome, fitted model, checkpoint, score/policy outcome
  inspection, continuation, incumbent artifact, or dashboard artifact was
  produced. No cleanup is warranted.

## 2026-07-25 G2 Fresh Transfer Acquisition Terminal Retention

- Retain the complete sealed
  `runs/forensics/g2_fresh_transfer_acquisition_v1` directory. It is
  approximately `14 MiB` and contains the immutable preflight, marker,
  completion/runtime evidence, terminal result, and `32` qualifying
  replay/state pairs.
- Retain `G2_TRANSFER_ACQUISITION_OPENED.json` at file/payload SHA-256
  `0c54dddf...03d7` / `6cc5c76d...c673` and
  `G2_TRANSFER_ACQUISITION_RESULT.json` at file/payload SHA-256
  `7b862377...ca74` / `a464287e...cee4`. The one-shot execution is spent and
  may not be rerun under this marker.
- Retain completion rows `f97bb0ef...6859`, runtime state
  `d9807b22...a98b`, runner summary `29f973b9...4465`, and source manifest
  `e689accb...aba9`. These establish exact `640/640/640` completion and
  `12/1/19` retained-root counts.
- Retain all `64` qualifying-source files. Exact replay/state reconstruction,
  root uniqueness, and protected-overlap checks passed for all `32` roots.
  Nothing may be pruned while G2 course review is pending.
- Decision `HOLD_G2_FRESH_TRANSFER_ACQUISITION` is a source-yield failure, not
  policy or solver performance evidence. No acquired game or retained source
  is dashboard-eligible.
- The run created only acquisition replays/states and compact integrity
  evidence. It created no label corpus, fitted model, h10/h20/h40 outcome,
  continuation, policy evaluation, checkpoint, incumbent artifact, or
  dashboard artifact. No score/action outcome analysis occurred.

## 2026-07-26 O2 Outcome-Free Preflight Retention

- Retain the O2 base charter and A1-A4 amendments at SHA-256
  `865f44c...7e12`, `79423c0f...e40af`, `610e5648...b9fb`,
  `f19e2c9a...472fe`, and `1095462b...79b3`. A4 is the authoritative pilot
  arithmetic correction; earlier amendments remain immutable evidence.
- Retain runner/tests at SHA-256 `99e61f55...5fda` /
  `00d98c67...5a47` and test evidence at file/payload SHA-256
  `dc043d5b...f777` / `e96cea6b...315d`. The evidence records py-compile,
  `19/19` focused tests, and `137/137` applicable regressions.
- Retain the complete compact
  `runs/forensics/o2_online_option_preflight_v1` directory. It is about
  `7 MiB`; no cleanup is warranted. Result file/payload SHA-256 are
  `09c8b19c...30fc` / `60ffe390...5f5b`.
- Preserve design, stream, collision-source, calibration, and power artifacts
  at file SHA-256 `d921ae15...0480`, `ff0a67f9...edbc`,
  `9a6a2275...5ed`, `cc9dc157...b7ff`, and `ca40b4e2...7d56`.
  They bind the corrected two-layer pilot, 640-root future allocator, 11,520
  unused stream rows, 9,101-source collision union, historical aggregate
  calibration, and preregistered power/MDE table.
- Decision `READY_O2_YIELD_PILOT_PREFLIGHT` is design/power readiness only.
  It does not authorize the 128-game yield pilot, option learning, mechanism
  or normal-start policy evaluation, confirmation, or promotion.
- The directory contains no generated game, consumed stream, rollout, label,
  fitted model, candidate action, policy/score outcome, incumbent artifact, or
  dashboard artifact. Dashboard record/top-three remain
  `263670/261369/258561`.

## 2026-07-26 O2 Yield Pilot Terminal Retention

- Retain the complete immutable
  `runs/forensics/o2_yield_pilot_v1` directory (about `28 MiB`). It contains
  the one-shot marker, `128` unconditional fresh-root replays, append-only
  attempt/completion/runtime evidence, and the fail-closed terminal result.
  No file in this directory is eligible for cleanup or in-place repair.
- Retain marker file/payload SHA-256
  `bbce7dd41c84ea5c6e1985a70529f284515ae6dec79c405581ce383c4a3c6457` /
  `23e704420d369c3b3410a8feb4b4620896a1ad96d90a5cc3df0128073234dc4b`
  and result file/payload SHA-256
  `f443a76392f09052179d8a9b458dd2d3ff615072c6d48024fafc5ef11b9ce576` /
  `0f3de62667ac13734ec45118de4179e40f8aaf819b0f570847de285e415333d2`.
- Retain attempt, completion, and runtime files at SHA-256
  `33e9062e748495cb2ac7f02be2be223b19abc875de6ea0624e1327fe63750948`,
  `de99abc9b096aaa6f12606a4a345417cfdadd7c16b8ed7a66ede4d09ab829eae`,
  and
  `4cc9ab89222a13ad39405ea75cb43abc7f0141fe875faf49b8ccfedb3a259138`.
  They establish `128/128` complete ancestry-unique roots, equal `32`-root
  family counts, zero retries, and `419.663s` charged evaluator runtime.
- Decision `HOLD_O2_PILOT_OPERATIONAL_INTEGRITY` arose from a post-collection
  completion-field mismatch before support scanning. `O2_PILOT_SUPPORT.json`
  is absent. The retained games are acquisition/provenance evidence only and
  contain no opened yield or policy result.
- No label corpus, option rollout, fitted model, policy evaluation, incumbent
  artifact, or dashboard artifact was created. Free disk remains about
  `149 GiB`; no cleanup is warranted.

## 2026-07-26 O2 Scan-Only Recovery Retention

- Preserve the original `runs/forensics/o2_yield_pilot_v1` directory
  byte-for-byte. Its operational HOLD, marker, result, attempts, completions,
  runtime evidence, and 128 retained replays remain authoritative and spent;
  the recovery does not amend or replace them.
- Retain `O2_YIELD_PILOT_SCAN_RECOVERY_CHARTER.md`,
  `o2_yield_pilot_scan_recovery.py`, and
  `tests/test_rl_o2_yield_pilot_scan_recovery.py` at SHA-256
  `a7630d4f37c7bde6164d3c3b5f7d9280c4371e268dda482e1b361da4f6197af0`,
  `aed1cee3e60c63dc5d32130f57043c92ad4121e170fac48303fbeeb96b85fae2`,
  and
  `646e6c1161f1d677c1efceb1e3460146529530a89c4b2cd2f92792056dd55b53`.
  Retain test evidence at file/payload SHA-256
  `f988e0f5f530040096a3530100cadd8e209ad8ab4af139d3f0f851924ba1be42` /
  `3deedb0e7c2f109133edf13f856c8a759434b61c6d8ff3305a6a3429ecfe000c`.
- Retain the complete sealed
  `runs/forensics/o2_yield_pilot_scan_recovery_v1` directory. Marker
  file/payload SHA-256 are
  `40b4e5f8aae5a5dc280b6dbf9871c1bd1a4a444173bff7b6c18cafbe98d84228` /
  `ca529ef7f4cb3c791af1c4bddec778a2cf14993366209ce345ae47927f9193f1`;
  terminal result file/payload SHA-256 are
  `4b74063061b072360094dc0069397439326e200362d07f5c00d58185c8e260cf` /
  `fff9450e4d35b8723258cb595890de921b7696f37899cfacd0478f19bce68a0f`;
  recovered-support file/payload SHA-256 are
  `a956d13d1366dc3ca343e84a49145367d7edab0d63c7b5b00a75aa090d64a1f9` /
  `7a7bbfce3a1cf7d611ca200d4a0dc3c1852295eda9c8ea95acf3abdf33e0b0f9`.
- `HOLD_O2_DATA_SUPPORT` records failure of both frozen A4 support layers,
  including zero `T768` cells and zero descriptive `T1536` roots. It is
  natural-support evidence only, not a representation, policy, score, or
  capability result and is never dashboard-eligible.
- Recovery created no game, consumed stream, corpus, label, model, rollout,
  policy evaluation, incumbent artifact, or dashboard artifact. Free disk
  remained `148.979 GiB`; no cleanup is warranted.

## 2026-07-26 O3 P0 Retention

- Preserve `O3_EVENT_CONDITIONED_DESIGNATED_PAIR_CHARTER.md`,
  `o3_designated_pair_option.py`, `o3_power_contract.py`,
  `o3_p0_preflight.py`, and both O3 focused-test files at their P0-bound
  hashes.
- Preserve `runs/forensics/o3_event_option_p0_test_evidence.json` and the
  complete `runs/forensics/o3_event_option_p0_v1` directory. Marker/result
  file SHA-256 are
  `4b8133ea1e8f237debbbdb90bb682214d04560de4756e0e6e639c4df9e6e63d1`
  and
  `9ced80be3e2a784372f50fd2a99b0b41bdcc98920820796daa03a8db1640ced5`.
  The stream/partition/policy/power/collision manifests are the immutable
  source of truth for every later O3 phase.
- P0 used 13 MiB and left `148.897 GiB` free. It created no game, consumed
  stream, label, model, outcome, incumbent artifact, or dashboard point; no
  cleanup is warranted.

## 2026-07-27 O3 Acquisition HOLD Retention

- Preserve the complete
  `runs/forensics/o3_event_acquisition_v1` directory byte-for-byte. It
  contains the immutable marker and terminal HOLD, `18,990` unconditional
  fresh-root replays, append-only attempt/completion ledgers, and charged
  runtime evidence. No file is eligible for cleanup, repair, or in-place
  continuation.
- Retain marker file/payload SHA-256
  `fcf9e275444ab1cce3b11855b54f84236ba2cb2ba96aa861c444830a99371145` /
  `c8fc61d249478b879c694fb166ae2e4f3d206765d74f0abd889844fd609db404`
  and result file/payload SHA-256
  `f7a967b936894a3d626055e366dc899d4efc52e9ad4b791b8c9a95d6e7fc791a` /
  `677b99bddc124cba128b46031dc67559db5845a86e3013464084712f88e2f9ff`.
  Retain completion, attempt, and runtime-state files at SHA-256
  `7b121f99945a15692b0719c0adebd6f30eb2fbb17c0c362d2e792ea79546aeb5`,
  `fbf59e5f1e614667f7e804c5ded4bcdde356f0f9c56a8a743b26aadd0488a4f3`,
  and
  `e70de8b6531f9c5650007048fd8c9f25e688323ad0c709db29b57ed017f64804`.
- `HOLD_O3_ACQUISITION_INTEGRITY` records a transient heavy-process guard
  failure at `18,990/20,500`, not a support or scientific result. No
  `O3_SUPPORT_SCAN.json` or `O3_SELECTED_ROOTS.json` exists, and the retained
  replays must not be scanned or repurposed without a separately frozen
  course decision.
- The directory is about `4.4 GiB`; post-stop free disk was `143.367 GiB`.
  No cleanup is warranted. Labels, models, rollouts, policy outcomes,
  incumbent artifacts, and dashboard artifacts remain absent.

## 2026-07-27 O3 Acquisition Recovery Preflight Retention

- Preserve `O3_EVENT_ACQUISITION_RECOVERY_CHARTER.md`,
  `o3_event_acquire_recovery.py`, and
  `tests/test_rl_o3_event_acquire_recovery.py` at SHA-256
  `079f377b445c0bc0ceca1273b8a2043eb6abbc0c2071e3d4863c5c57d536c2c7`,
  `b67be9537e7728855d63006f12503038d3414fc600a9adc08809869ab8e64525`,
  and
  `bde138b9233003a028c7c6e3a7e383f845ff70dc3c7ff4f83e919a8b8f9f4ee8`.
  Retain test evidence at file/payload SHA-256
  `8b7abb4be9677bf5eaf967addb65f068d11bf82d2bba0b8cb2ae57988a394e64` /
  `a6b8bb6f404dff7fc407c51b2363ddbbfffa42f74d3de2658125a52963f610a0`.
- Retain the complete new
  `runs/forensics/o3_event_acquisition_recovery_v1` namespace. Marker
  file/payload SHA-256 are
  `a2c27ba9e7b0d772db4a57eb765bfa4540ff8c3a59568a03a30637cc8a6855ac` /
  `0a90923b154b6654ddaa3473d9ef96c26c89851d32d734e368c21b924efa4de7`;
  zero-game preflight result file/payload SHA-256 are
  `6cfbb55fb8f3ccefcc8a214ed4c83b8eac33a82e59e318dace0832175ef8a24f` /
  `1dfb52d6c90e9ca3b2ac7d0970909a6b5e5f659043be9ad5cfe8f157955705f8`.
- Source-audit, complement, collision, ownership, and process-guard artifacts
  are protected provenance evidence. The source audit hashes but does not
  parse all `18,990` original replays. The complement contains only the exact
  `1,510` never-run P0 rows.
- The recovery namespace is additive and separate. Never append to, edit,
  delete, or reinterpret `o3_event_acquisition_v1`; no cleanup is warranted.

## 2026-07-27 O3 Recovery Terminal and Integrity Reseal Retention

- Preserve the complete
  `runs/forensics/o3_event_acquisition_recovery_v1` directory byte-for-byte.
  In particular retain union, support, selected, and terminal-result
  file/payload identities:
  - union:
    `02ea2c5be8823de775f56b7267f9c8371d26efc53897115b25733f8ef4527311` /
    `cec88701a1754f1064d639dae09cd6856ee18ce9399865338ebed7107f672d94`;
  - support:
    `4c71513e6a3a2778bb8d1db0ba08f8ff5a1f0d6edc82ee1208b7458593059d27` /
    `27ae3a6aca5f1de71ee18df193c0663a83579d3aeba65cd864065cfff594e25a`;
  - selected file:
    `9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049`
    with embedded/post-JSON hashes
    `c6c8b1a35cc63f4c1c1fdc98579f1ae0859a84c5eef7203306000223ac9c61a5` /
    `d9600cf420d947826c812b88225633b78a889f94f94ce39270dd71bc11b12f0e`;
  - terminal HOLD:
    `962da52b83b8746c006a9ef5fbe1fdd34f43e9c7bf97d9b6ff48f2a42019c23a` /
    `a679d512d6ce44bf5fd4ecd8249d15625c59f342e64796a6d5eb894396224ad0`.
- Preserve the complete
  `runs/forensics/o3_selected_integrity_reseal_v2` directory and its
  immutable fail-closed envelope, file/payload SHA-256
  `f466cae4e298edfc25499a90a78bfb6d6e037e2d065be72eb0de498cf9b31d57` /
  `58b55acb66033092dad5e789421d4cb60adfe960ccf25e1a6ef277e81141357d`.
  Preserve the V2 amendment, runner, and tests as the exact failed-attempt
  source surface. The intended test-evidence file was never created.
- The recovery union/support artifacts are valid completed scientific-source
  evidence, while both terminal decisions remain authoritative engineering
  HOLDs. Do not edit, overwrite, rerun, or reinterpret either namespace.
  No training, mechanism evaluation, or capability evaluation may consume
  the selected roots without a new integrity course decision.
- No artifact in either namespace is eligible for cleanup. The reseal added
  no game, stream, label, model, rollout, policy outcome, incumbent artifact,
  or dashboard artifact.

## 2026-07-27 O3 Selected Integrity V3 Retention

- Preserve `O3_SELECTED_INTEGRITY_RESEAL_AMENDMENT_V3.md`,
  `o3_selected_integrity_reseal_v3.py`, and
  `tests/test_rl_o3_selected_integrity_reseal_v3.py` at SHA-256
  `2ba2779cf47a59bd489471886ca7f5ae8994d6bce5dd392131c24652ffbc9c16`,
  `2770b02515fa1834c25415cbbbd7b949362db284456f33498a794a3e1fb1ab21`,
  and
  `d6558b25a5d72c56401e6129c2cb8227afb39dafb45086766749ee4ce093fe33`.
- Preserve V3 test evidence at file/payload SHA-256
  `6608d39605d38727fb81e85208a6e2e7fc5be14eb04c3fba2624d9f2d131a906` /
  `aef78c400e86bcc35ad15aa2d9937eeb55769bfca5ae2146570f57fc6270188c`.
  Preserve the complete
  `runs/forensics/o3_selected_integrity_reseal_v3` directory and terminal
  envelope at file/payload SHA-256
  `5bb80bc02597ea934c02f8ebd07eaf0158623232f88ea0408532cdc0039e6696` /
  `622ebf6361527be7283fd51c7a7acff99aa8125b06c76dbc4ee8a801faf3904d`.
- V3 is the authoritative selected-root integrity envelope. The original
  acquisition, recovery, and V2 HOLDs remain protected historical evidence
  and are not superseded or edited. None of these artifacts is eligible for
  cleanup.
- V3 created no new scientific data. Downstream O3 artifacts must bind this
  envelope before consuming the selected roots.

## 2026-07-27 O3 Option-Training Preflight Retention

- Preserve the O3 option-training charter, runner, and focused tests at
  SHA-256
  `5d382e68e3ec306ead500be492d3b512b28298c32235ebc9a6074ffb82c09d58`,
  `e9baa328f2091e8f51ed287f774b862314af08e828eb2fe032dda4409e06504d`,
  and
  `cfe8a2b4e090d2b3e73b550adeeedb2a0058eab4f40d8e7eb4af28c99d815a9e`.
  Preserve immutable test evidence file/payload SHA-256
  `c86fac6b04ee0a128208d2d6ae28773d994f7ed12d47bf2ee212b8674fa7f502` /
  `7475da0251066fca779aeafc4c93715dff7ab44221f108acf675c5c8264c44d9`.
- Preserve the seven compact artifacts currently in
  `runs/forensics/o3_option_training_v1`, especially the READY lock
  file/payload
  `b2b355ba08dcc7716d53e90b2a8a1f94fa6a674f97586bb7d45f9d45d8256dd2` /
  `75b7af3245617f33d29bcac8d3cab40de6589b99b2815153f2239e73e9e8d334`
  and preflight result file/payload
  `118f937ae247e3cb3510e78b512b0f240b1129ad20e7de756f94dacc2e7708d6` /
  `e1a6c2f4483bca39a29d1375bc3dcb33ec36055135900e1064b40286d7944fde`.
- This preflight contains no consumed stream, episode, label, model, policy
  outcome, development-content read, or untouched-content read. No listed
  artifact is eligible for cleanup.
- Preserve the immutable zero-label training marker at file/payload SHA-256
  `e00033d12e74c7c1f5a61fc4bfdc31c3a26e466978ea7a4da86b13bc7d624d13` /
  `2061319205b4c95b8d80c8886a813f4c59f5840ab0889a066283721716c24816`.
  It is the sole authority for any same-command training execution or
  resume; never edit, replace, or recreate it.

## 2026-07-27 O3 Option-Training KILL Retention

- Preserve the complete
  `runs/forensics/o3_option_training_v1` directory byte-for-byte. The
  authoritative terminal result file/payload SHA-256 are
  `943fefaf4cc2dfcbc50a670119caafc279056ba2d4c492be6680b075ddc32c67` /
  `11ab686df92f75f2d9f3b7e206a3c4ae05e1a92aacf09bea6c45b4dd77c6599b`.
- Retain all 123 committed episode array/metadata pairs, the append-only
  attempts ledger SHA-256
  `1fcecac5b34edb346caa3ede2cf12b75923a6106ea556e8c9533fbe7c2fc0592`,
  runtime-state SHA-256
  `4c9000890bf9c51974f1abd0c8bba2ac4abd7dc85bd18875ec26353925df08ee`,
  and initial untrained checkpoint SHA-256
  `1b22b243c46b9d4c4336516b860e8c097ccdf9675a2d7a59b6bd49d3ab4c25fc`.
- No artifact in this namespace is eligible for cleanup or downstream
  training/evaluation reuse without a new explicit research decision.
  Never resume, repair, overwrite, or reinterpret the exact killed run.
- The subsequent permanent-close ruling is stricter: preserve every O3 byte,
  but never inspect or reuse episode arrays, episode metadata, labels,
  actions, outcomes, the initial checkpoint, selected O3 roots, or O3
  learning streams in future science. Aggregate counts and immutable hashes
  may be retained for governance only.

## 2026-07-27 O4 P0 Retention

- Preserve the complete
  `runs/forensics/o4_domain_safe_p0_v1` directory byte-for-byte. In
  particular retain its marker file/payload SHA-256
  `7f84bbd9679b9d6294a0530b47b5ba01749426191a1a3f509bf38a48114723b6` /
  `854822ffb6bd6b23cae646c684475293e9f65c8b35a86267c5616f39f2d55679`
  and engineering-HOLD file/payload SHA-256
  `17be1eb2c5ecf0be1a7331779e5eab7cc3159eb760d50d4f4b7aacdf395332e8` /
  `af14ae1a94fd335d6a6728f0ab7d62077a47f328290e36b1aa1e2dc2d0c8e2ca`.
- Preserve the complete
  `runs/forensics/o4_domain_safe_p0_v2` directory byte-for-byte. Its marker
  file/payload SHA-256 are
  `9d9f032f61fa637941d677e788dcb7d2dcec70179a7ea9a2fafe128af73336da` /
  `645edb5464a9a258c8a0f9b6b563f595e3a7bf4d5c35ff35d901ecbcc2195f4b`;
  terminal file/payload SHA-256 are
  `897cac07ce2625f5616690f0a4611e11948e6ca58a55b828ee43f92b493893cd` /
  `ed0102032291a8396ffabccdddc657e57779ef2623e21427b49c1ed344d87eac`.
  Retain every bound source, replay-hash, collision, policy, domain, power,
  stream, and selection audit in that directory.
- Preserve the O4 charter, domain-safe representation, power contract, V1
  runner/tests, V2 serialization amendment/runner/tests, and both immutable
  test-evidence artifacts. No O4 P0 artifact is eligible for cleanup,
  mutation, rerun, or downstream scientific reuse without a new explicit
  research decision.
- O4 P0 produced no fresh game, consumed stream, label, model, policy
  outcome, incumbent artifact, or dashboard artifact.

## 2026-07-27 O5 Four-Family P0 Retention

- Preserve `O5_FOUR_FAMILY_DOMAIN_SAFE_P0_CHARTER.md`,
  `o5_four_family_p0.py`, and
  `tests/test_rl_o5_four_family_p0.py` at SHA-256
  `39d598f44c1b6478c927b09dfdd400b4b0991ce1e78e6e97c1b5131de6b3b7dd`,
  `f0ffcc17578581b6e4783e63beef28e59ffab16676ddbd84126127fab47bcff6`,
  and
  `8f54c464e4ada8ca9e175770756a932278c0c7f22330eae64956d568ea3c7e6b`.
- Preserve immutable test evidence at file/payload SHA-256
  `7a3e0edf7a3b1aabfe775e1a14a101b90fb18aa4153fed4518eca88224e39447` /
  `22fe9b0ca21c9ba10f2060115d20bdd405ce637a11fa7e7d6a9570e8abb2162c`.
- Preserve the complete
  `runs/forensics/o5_four_family_domain_safe_p0_v1` directory byte-for-byte.
  Marker file/payload SHA-256 are
  `902df97928d2b393c8819887717c213b831f8321ac0270a0761633737b668c13` /
  `12cc4b06cb2e9fae81bd569d6681ad15a5ed512152679d201b793d51693512dc`;
  READY result file/payload SHA-256 are
  `b2ca5368dd6f29debfd0fb0e4c86005c9bae7b92d736ebc5750c5ec71f97a96f` /
  `1707c2982e62a29787b69dae9f6e31c9a042162f0573203ab6f2f38d9d3b7fe1`.
- Retain every O5 source-hash, selection, stream, collision, domain, power,
  and policy audit. The selected-root manifest SHA-256 is
  `05850e87eaa03010e06c27b548d04d22bf22768dfa94d62d0a2a1cba96d20612`;
  the reserved-stream manifest SHA-256 is
  `a536eb66a4afc73822bcf0448afe7f8229531c40866555191d9d41bb39048302`.
- O5 P0 created no fresh game, consumed stream, label, model, rollout,
  policy outcome, incumbent artifact, or dashboard artifact. No listed O5
  artifact is eligible for cleanup, mutation, or execution reuse without a
  separately frozen and authorized training charter.

## 2026-07-27 O5 V2 Training Retention

- Preserve the V2 charter, runner, tests, and immutable test evidence at
  SHA-256
  `0b274979e388f5e0297c17d85264193caf2186794024b14d500f504ff7a7aede`,
  `37e0a20d2437f09ef7efe1073573f7f53c4ed8ae0267560192e7892164e956ea`,
  `5006d11984ce927d72729ef065ad3bfd3772750303d62208f37e4872e5ca7e27`,
  and
  `d42d871824952cc429a030a38be87f5a2d551472fae07dff98aae17f9c937cff`.
- Preserve the complete
  `runs/forensics/o5_domain_safe_training_v2` namespace byte-for-byte.
  Marker file/payload SHA-256 are
  `534151b8514336db2fe8d5946c8c66acb42ec1b0931a6d41c9ccf66ed9578cd8` /
  `c4a811be16b5c71ffcf1c5d53c8698bbe44cb18b65f937dcc5681302b4834ba9`;
  terminal HOLD file/payload SHA-256 are
  `74ac4ca9f375ff93e2fed5dfa5c2154a7b4fcc682654539e05cc67cc4a515e05` /
  `686c34218b0cb06c2411dc4e3ee072587d36ee347088684a71d5d9ec29c866be`.
- Retain all `1,152` episode files, task and fit ledgers, runtime state,
  manifests, source/collision audits, and aggregate support report. Episode
  manifest SHA-256 is
  `03e11f4942dc251ed6145d8762108bace984d63414d65b235e81fb7bce0db9ee`;
  task-ledger SHA-256 is
  `aeb3b876e258c1f5158068932ef788b1caf740e567076313869b379d45816782`;
  aggregate support file/payload SHA-256 are
  `bd905b62f05c95c42dc36336f9133d6d80044687476be396161d43ada10a7a94` /
  `45d513007291d3e33cc902b7e97cbaf0cf58319b6f27c186fb745857f0544267`.
- Retain all four provisional checkpoints, but never use any downstream.
  They are explicitly non-authoritative and quarantined by file/payload
  SHA-256
  `96a5336f3a9c37dad56447ceedf9481cd39fe0d6f896effa5f47b07b9c461ece` /
  `6658cef4e6a9a111ae1b2cabf5970ce70bd77449470c39f47e44209ac57a4054`.
  No checkpoint-authority artifact exists.
- The exact V2 run may not be resumed, retried, retuned, threshold-relaxed,
  or reinterpreted. Development/untouched roots remain unopened and all
  downstream evaluation remains held. No artifact in this namespace is
  eligible for cleanup without a reviewed retention decision.

## 2026-07-27 O6 P0 Source-Preparation Retention

- Preserve `O6_COMPETING_RISKS_P0_CHARTER.md`,
  `o6_competing_risks_p0.py`, and
  `tests/test_rl_o6_competing_risks_p0.py` byte-for-byte at SHA-256
  `2ee1e4273866f7f40376fb584e908f5a0e10e70446e2540f36bf320ac0edbb11`,
  `c1a1d0a22fa185672e62f0b712d79d8bd01d76e04cebe04bca78b45a7c092dd6`,
  and
  `3d7cbe8f20149f3b21305e8306762f8ede78f2227094f8848dc5b6f383ba0b34`.
- The accepted preparation audit matched all `18` dependencies and `26` core
  governance identities without parsing protected bodies. Preserve the
  frozen risk/source/protected/power schema contracts and the exact
  `4096 x 4096` batched workload specification.
- No preparation test-evidence file, output directory, marker, source
  inventory, selection, stream manifest, power artifact, label, model, or
  outcome was created. Any future execution artifact must live under the
  separately frozen O6 P0 execution namespace and may not modify these three
  preparation files.

## 2026-07-27 O6 Staged Execution-Surface Retention

- Preserve the complete directory
  `threes_rl/runs/forensics/o6_competing_risks_p0_execution_v1` with exactly
  four immutable staged files. Test-evidence file/payload SHA-256 are
  `4f7d5b90091dfa23a1d7c674148b6f2fb18e5f61360d5a81f8e68de9a50537ae` /
  `59b86d7bccb973caa1af29d5a3dd95540463cf2ef538f7aa163add5b72dc9084`.
  Preflight-lock file/payload SHA-256 are
  `74afff9908f04484a857af551df2f6538fd77ed27a6fa6fbfc2a2bd2e6502ff4` /
  `a71f066fffae412e780b696a88593b6c941632cba17f0274eab12f85a19ce1b2`.
  Preflight-result file/payload SHA-256 are
  `d7bae4fb50a0e29a717256f06dda434c8d6f34b0459a1205e51066a93c1356f1` /
  `4ddf35563d451862c3d630662691cc89468625c92036ac2ee9353eb72655bdb1`.
  Opened-marker file/payload SHA-256 are
  `bcb5bc559e1023ed0cc71478dd9751b58d0a679bbd0d359363acc81d9c1fd025` /
  `132dafabae870b977a151cd1e477970254b7a69d5a884ba24621879bfe626bd8`.
- Preserve the marker's canonical `30,900`-row byte inventory at SHA-256
  `c88e5420c2b0e446a0e77f8ea32de57e7b4173e0fc5a84edec02d53e90b5de6c`.
  It is content-blind: no candidate, replay, governance, protected, human,
  O3, or O5 payload was parsed.
- Do not edit, replace, regenerate, or remove these files. No execute-time
  exclusion, source, candidate, selection, stream, collision, power,
  operational, runtime, or terminal artifact exists. No execution may begin
  without a new research-lead authorization under this exact marker.

## 2026-07-27 O6 One-Shot Terminal Retention

- Preserve the complete O6 execution directory byte-for-byte. In addition to
  its four staged files, retain:
  `O6_P0_PROTECTED_INVENTORY.json` at file/payload SHA-256
  `ec29d02a435846c1627ed00ce90c32549caafd5214355d7fe70bb26aa7eeb4ba` /
  `332fb56d185b158600fa9f546ffb2249bf2fe69a3fe043c81dbbc3f1d2c8c88d`;
  `O6_P0_RUNTIME.json` at
  `b05b615e652177e6a581a10dbcecc786f9156cae74b845049e8665eda30300a3` /
  `9bd2910a79630c4685e37f13b660132f778db66437ba5d12913998487d581c3c`;
  and `O6_P0_RESULT.json` at
  `4cc27d5ec374eeaf5f14189977a36b9e99ab4411606cdc78f5d94be50a3376a4` /
  `2f37353ea88b71afba4fc81866f6709992a2af861918430ee7630358314a7bc4`.
- The terminal `HOLD_O6_DATA_PREFLIGHT` is authoritative for this one-shot.
  Do not rerun/resume the marker, reconstruct the unsealed transient
  exclusion-key list, edit thresholds or identity rules, or reinterpret the
  HOLD as representation/policy evidence.
- No exclusion-union, source-inventory, candidate-root, selected-root,
  stream-reservation, collision-audit, power, terminal-operational-audit,
  label, model, checkpoint, or policy-outcome artifact exists. Preserve that
  absence as part of the evidence.

## 2026-07-27 J1 Proposal and Readiness Retention

- Preserve `J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md` at SHA-256
  `26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2`.
- Preserve `J1_IMPLEMENTATION_READINESS_AUDIT.json` at file/payload SHA-256
  `f3e4e8029e159a1db7767164e1623d2e166b139be319d6077d61d7d107a44042` /
  `5b6b9a2383296f82b6547bbd46ddc892b486e4b89f4c325aa88f9c8b15944f99`.
- These are outcome-free course artifacts. They reserve no stream and contain
  no game, label, checkpoint, model output, policy outcome, human-session
  content, incumbent change, or dashboard change.
- Retain the explicit implementation HOLD: current `train_ppo.py` is not a J1
  execution runner. No 213B-226B stream may be reserved or consumed until a
  separately hashed runner, GAE/resume tests, compact denylist, synthetic
  runtime/storage projection, and zero-work preflight all pass.

## 2026-07-27 J1 Implementation Preflight Retention

- Preserve `J1_IMPLEMENTATION_PREFLIGHT_CHARTER.md`,
  `j1_joint_policy_value.py`, and
  `tests/test_rl_j1_joint_policy_value.py` at SHA-256
  `7f87bc29c5764ccb290b25558f1cfe999083e9fddb089ea652cac9d0b92ab137`,
  `55d9e3206c2905509466c4962006e6cf3426f76647af6d2e60afe674b80c9bfe`,
  and
  `e6b169f2d629021f96315380a3cf0ff6eece94a30e5027b1ace4d741499fbfa4`.
- Preserve the complete
  `threes_rl/runs/forensics/j1_implementation_preflight_v1` directory with
  exactly five immutable files. Test-evidence file/payload SHA-256 are
  `aceab517c4fffc52fe1827468b8408484c0f9ddade594e5200e025d71239137f` /
  `686b1e58daa937076704eec5ebd84b3af6bf2a47d8ec41875fe4901cf5dc988e`.
  Denylist file/payload SHA-256 are
  `0a7be318ebe5281a11ded38f3bbde29745ccb7c3a969585de1788df468fbd763` /
  `22731c89df661419d7ca2bcffdb86240f2ad8974b00e765dd715cf8f4e675add`.
  Projection file/payload SHA-256 are
  `e023fe04239ceb2d317ab0e26979033db3c2a5c93d4a5016168de442fc97e401` /
  `1aaba01b73d53ad10252f0c59c238c8274a9e8f8066a8f3f03f3c0587c6bef0b`.
  Lock file/payload SHA-256 are
  `42d1f8d3d6b7bfd62c173a3147ce1eb7dff465aaa92271e7af6bc5fb3c533825` /
  `e465cec348f987af4c77f062a0e8f8bfa968ddc4ff460b40ba829915791622da`.
  HOLD result file/payload SHA-256 are
  `339e3ef6dcf8c5b3eb1951204d08b97b94b3c4816f993d58509b9b341dc364b1` /
  `4d21a092e584d9419a47bef384de164cfc9a8590268a67abefa35afb6b573ce2`.
- The result is an outcome-free runtime-cost HOLD. Preserve every threshold,
  fixture, timing distribution, 5,000-move sensitivity, denylist identity,
  semantic/resume check, service audit, and zero-work counter byte-for-byte.
  Do not retime, relax, overwrite, or reinterpret this preflight without a
  new research-lead course decision.
- No J1 marker, reserved/consumed stream, scientific game, label, scientific
  optimizer step, checkpoint, development/confirmation content, model/policy
  outcome, human-session content, incumbent artifact, dashboard artifact, or
  promotion evidence exists. Preserve those absences as part of the boundary.

## 2026-07-27 J1a Cost/Power Amendment Retention

- Preserve `J1A_OUTCOME_FREE_COST_POWER_AMENDMENT.md`,
  `j1a_cost_power_preflight.py`, and
  `tests/test_rl_j1a_cost_power_preflight.py` at SHA-256
  `d738a55bb438ee87d59d2433466e813cfd0a9fb5f041cbc3cc807d4bbafa2e11`,
  `27ffb3825d60bd8ca4ec0646f976e325c2a7c5f00a077aea3803544531fe6a98`,
  and
  `898f25aa4ed109db2c9fc27b4bba9d7e9641dc57834e4e02d7a8242df195eb59`.
- Preserve the complete
  `threes_rl/runs/forensics/j1a_cost_power_amendment_v1` directory with
  exactly four immutable files. Test-evidence file/payload SHA-256 are
  `8d052459dd8c914b0c3b68609d113b3f1bc7d8bbb5ec5412635bb5affe306edc` /
  `a5ffd778c0cfce00d58429a287e7a813d4e429ea5e862217c9ca32a56fa24597`.
  Arithmetic file/payload SHA-256 are
  `957159dcbfe4ee95be9c2abd2ab2d99a4cd49ce611895bdd3c55ff5ce4fcf9b0` /
  `b1d13d49db07fa59afd995640c5d063f8bc9776122ead554bb53856543fd21b6`.
  Lock file/payload SHA-256 are
  `7ed37c9bf1c6ec0fe7e74f36ef4cde8ab5e3bdd8ae1a7d9e1e065e32a21b852e` /
  `b84228d9e5587682fad0cca91e0e5349076ab70674cf0412205712fa05e37850`.
  READY result file/payload SHA-256 are
  `4ecda2a1101011437c912d884dfb5acecf7e586b87c4646c63354c4ecc5403ef` /
  `abe17a53c1af2b182a488d4fc05b060a214b106652c04462453ad01e75ed9471`.
- The terminal is an outcome-free arithmetic readiness result. Preserve the
  exact parent-method reproductions, amended power cells, fixture-derived
  central and 5,000-move cost projections, 91% headroom rule, prospective
  stream-prefix proof, service/disk/process audit, and zero-work counters.
  Do not retime, alter counts, reserve a stream, or create an execution marker
  under this namespace.
- Preserve the absence of every game, scientific label, optimizer step,
  checkpoint, development/confirmation content read, policy/score outcome,
  human-session read, incumbent/dashboard change, and promotion artifact.

## 2026-07-27 J1 Execution-Surface Readiness Retention

- Preserve `J1_EXECUTION_SURFACE_CHARTER.md`,
  `j1_execution_surface.py`, and
  `tests/test_rl_j1_execution_surface.py` at SHA-256
  `468cc181c32a934fcbc64bb4cadc22758bd0fc46870f0f120f9ac6008ddb696a`,
  `d4367d95aba05ec592310008bae21e7de90905fa1268601dd60cc8fcb2b6f2bd`,
  and
  `cb696e4502d61abd7a24d5781d7c15e2dd8a0ffed538480ecbd2a27434a339cf`.
- Preserve the complete
  `threes_rl/runs/forensics/j1_execution_surface_readiness_v1` directory with
  exactly six immutable files. Test-evidence file/payload SHA-256 are
  `465d1c4a00f91e3e614cd496ad3260236ecbb3106dc0b69e3a12a38380ff89b6` /
  `3982fb9bace0fe1ac73610804445df19c53bcf0944bb1ba81a70ec9cdc3738d7`.
  Schema file/payload are
  `0c9bd38e5cbccd840bbc4aed575b6e1dd95aa9516ed1f9431b87ff5f93d13730` /
  `d58082c26ecdcca641531c71198307bc65b997e5c27dd647008b96e6ca6ac681`.
  Prospective-manifest file/payload are
  `2aee68a08325cdbc5e42153942079c1375163f2b88217bf407e64fd95f096dce` /
  `de0046a2121138659dd2fd0bb46a48081d80842c5d24334d1a683dbf0a9a7093`.
  Runtime/storage-projection file/payload are
  `92dfc49a8f0830a4b39c627d9257e4a20b4ca504019c455b3b2b1eb05a959f20` /
  `60e9697e82409e5ea930b7b07d2ab042ca3b28ecebff4bc6c2058f8b04e9f6ce`.
  Lock file/payload are
  `e7f648eb04d7d197a9a2391352f82af5df6a12f7868ced8c8e9559703adb9fdc` /
  `70c83f640632ec034b346cda355c875f79cc002409474d537ac67a6ab7c975cc`.
  READY-result file/payload are
  `ba3e9d67c64b89cf583c2ad1778b073a6a702c003bf1a895c164d6f9f984d4f6` /
  `af5525b35ec5d5c0deab88d1ec00d8215fbb4dc14abb2aaa8dc9aa70b27d556c`.
- Preserve create-once artifact semantics, bounded dispatcher and engine
  routing, ownership/recovery and stream-lineage evidence, all authenticated
  resume/retirement rules, the exact regression accounting, the central and
  5,000-move projections, operational audit, and zero-work counters
  byte-for-byte.
- The separate `threes_rl/runs/forensics/j1_execution_v1` root remains absent.
  Do not create a phase lock, marker, manifest, owner, reservation,
  consumption, game, label, optimizer state, checkpoint, evaluation result,
  or retention artifact without a new research-lead authorization.

## 2026-07-27 J1 Training Start HOLD Retention

- The earlier absence statement is superseded only by the authorized training
  start. Preserve the complete
  `threes_rl/runs/forensics/j1_execution_v1` directory byte-for-byte at its
  authoritative `HOLD_J1_OPERATIONAL` terminal. Never retry, resume, replace,
  append to, clean, reinterpret, or reuse its consumed streams.
- Preserve terminal file/payload SHA-256
  `21092fb34631eb0eaf48811caa814ff4d05abbb23c9bc5add85eefd93a8959d3` /
  `9bcc81d217141fdfa801d1fca606c356720e4ac5c0e2a26f9d1ab688ca93dbcf`
  and retention file/payload SHA-256
  `dc339aafdbe32859d07c591a36c9088afa53f5be30412f3340049ca18994ceb0` /
  `11cc89c6a6fe41ff74c472e3fa0b61d179e1cedfa4755cc4f13fe7ced44018b2`.
  Preserve retained-file inventory SHA-256
  `7233c65745a9ae7258dbb165b60f4ae55c1cf60376819b80bb9e0be17d677471`.
- Preserve the genesis-only commit boundary and exact zero counts for
  completed roots, attempt events, optimizer steps, and round aggregates.
  Preserve the operational diagnosis (`deterministic=true`, intra-op `1`,
  inter-op `12` versus required `1`) as outcome-free engineering evidence.
  No checkpoint or scientific outcome exists.

## 2026-07-27 J1b Operational-Repair Readiness Retention

- Preserve `J1B_OPERATIONAL_REPAIR_PREFLIGHT_CHARTER.md`,
  `J1B_OPERATIONAL_REPAIR_PREFLIGHT_AMENDMENT_A1.md`,
  `j1b_operational_repair_preflight.py`, and
  `tests/test_rl_j1b_operational_repair_preflight.py` at SHA-256
  `a426801fc3015051ea51517e925a7d1c2e556718e2551ee480b802c8a7422cc1`,
  `64de3de37bff6a08bd95da217dc52d2f4bb58fbf99d28bede263a44d0aa2eb9c`,
  `7d73565c510dfe74b87ec362c05f8928e15a65cb8af5494b5ad9fe5f4c30ca5f`,
  and
  `f7e55b71f7954fcbdd4db61c1693d773b8ea106684ea19ad19998be15f4dbaff`.
- Preserve the pre-A1 historical evidence bytes at file/payload SHA-256
  `d2f6333bd4fdbe584fbf231141a24c01256dcc9ebe0f57c2691e19a8f046bddf` /
  `b462c0b46afaa478caeb66c622799eb1e7a533673439a89fe0e60650a448e25e`
  in `j1b_operational_repair_preseal_history_v1`. They document only the
  corrected CLI exit-status defect and are not authoritative readiness.
- Preserve all nine immutable files in
  `threes_rl/runs/forensics/j1b_operational_repair_readiness_v1`.
  Test-evidence, root-cause, denylist, prospective-manifest, runtime-audit,
  projection, and schema file/payload identities are respectively:
  `1b42acb9e88f46d0e20910ea0f6d7ae590418c3f268f8a469b96cc974025ee68` /
  `ef126d9c892ee4081b0390779ec50f0d290fbb9380eb6c1e26038d9df5228760`;
  `692290bfdd741b7c7442dccfe67d48cf347f69fe9be05ae51bb71f9a15bc438d` /
  `4a192fb02203149eaa640139b65996bcd65c14aa5a010273d7e6451b01f7f17c`;
  `1d36c79bae091a8b6b05ce69a2de19b43241ec091423d3a9d1bc3002ec229704` /
  `83cf3201ffba5d4080ca76bb15d906a3b4cfe5df3eb32efe4be2088546b9d8f5`;
  `2bb0b2385360f2d06c019fdbac11cb58515629ab4f5fcf321624f499a07329f9` /
  `f85a7624b2e8052d0b451bde9bf792181e08e055406fb5837232655a48f8a8a8`;
  `1b85bfd78d6658c979ca9007782dd119e3307258d8f25f87896d138da6da078f` /
  `4a8abe1edb9d59bfbf07124caa8a5d1e9a0fa51e74aa5e2b116650d43c427f01`;
  `ef1a83390978843ce0fccd2961066a2c34a1b1117936fad7247650ea48574d18` /
  `21da2409f07c86beb63fc54af964bb3533e668f6b69dcf820472cf567cb80857`;
  `862795836243ee69670bb8333ea5d2f58b0394048cb86c0aaea476b9fba04456` /
  `2f5c3c924dbf2195e8a29b3b1b99ea75d6061d5b159048c7d5a7b71c3495f430`.
- Preserve lock file/payload
  `b8b5377370f0e9e04739aae582604ce85f38bd1ddf84b5312a2cf12406f38814` /
  `ef0c1adce5f948a238e81911ab034d84ed297c2b2570d58481fb2906ef2e7e3b`
  and READY result file/payload
  `108038d15b222afd00c07c9801b460fb4687bfe0a9e8a4fb54a59e58e8907ec6` /
  `5d56b2c3cec39c16590a20f8acf8f10c60db7739e5161a653ea45a779204ba5e`.
- Preserve the exact fresh 16,384-root manifest, collision proof, clean-process
  runtime ordering, original genesis/root-cause evidence, tests, projection,
  operational checks, and all-zero scientific counters. Keep
  `threes_rl/runs/forensics/j1b_execution_v1` absent until a separate
  research-lead authorization.

## 2026-07-27 J1b Training-Only Execution-Surface Readiness Retention

- Preserve `J1B_TRAINING_EXECUTION_SURFACE_CHARTER.md`,
  `j1b_training_execution_surface.py`, and
  `tests/test_rl_j1b_training_execution_surface.py` byte-for-byte at SHA-256
  `aeb458781e206f8f16002ffaa311d782b26fdb4076211155a6230b9835e29858`,
  `c586d41f752cff7aa7c36c911008ca72ce147139fedd7586a03e627471282bc5`,
  and
  `86159c76a42c54c47d30e75b92f988773a6c6da580e8bb6b01de0f2a944a516e`.
- Preserve the complete six-file
  `threes_rl/runs/forensics/j1b_training_execution_surface_readiness_v1`
  namespace. File/payload SHA-256 pairs are: test evidence
  `aa8a78db1b2a740a2fbcdb183049f47e5d6231a1e85b26d51fa98adc4c68c590` /
  `da57014e6bb8cbda4ecaed6c9f61f8092f33667a642c9da588ee496f71449e2a`;
  input bindings
  `76284190098c72fc7b591f459c5feabe4ef24d514dc9d79f15bba8b8a9e665c1` /
  `eed9cc97b4e4f83662401303f30556d248e01203a208db2087bf5ff0a95add20`;
  schema
  `9a7f63ed20311d15353c6d051dfc6882e43f5f96bf973143cd1a005a20b2f761` /
  `336f1c4942d709bafe8d8fe9477cb7a418fce77eb30ec314aeb371100f0ef05d`;
  projection
  `0efc23c4abd3c8723d7567e1647f7f8a059f25278a90aea589678a0f11a3fc90` /
  `dcc8562a37ed87ee0f2334d4fcb631126ccb5532ce5b5f9d5284c571fec8c109`;
  readiness lock
  `adeae9ce6f9056914da48b79096ee7143a559a2d4e97c02cbe622eff7b0eb79e` /
  `e559d197f299d2ddf62d8d7736c8fa5a6256c90248ed658f9f24c3459c5b11fe`;
  READY result
  `3403a9d70e73e38eca7a372bd7db08b855051f1c409b621ebb7a391c45d96213` /
  `84fc2adf7d5204ed1dd1002799fe250575657bb937badc194abaa1a02217b3d2`.
- Preserve the exact 16,384-row fresh source binding, runtime-before-owner
  ordering, bounded-engine-only call graph, owner/reclaim and stream lineage,
  32-tick and bounded-I/O contracts, crash-resume tests, terminal/retention
  behavior, source/parent/spent-J1 tamper tests, projection, operational audit,
  and all-zero counters.
- Keep `threes_rl/runs/forensics/j1b_execution_v1` absent. No J1b phase lock,
  marker, materialized root manifest, owner, reservation, consumption,
  genesis, game, transition, label, optimizer step, checkpoint, holdout read,
  outcome, human-session read, incumbent/dashboard change, or promotion may
  be created without a separate research-lead authorization.

## 2026-07-27 J1b Open-Failure External Retention

- Supersede only the earlier statement that `j1b_execution_v1` is absent.
  Preserve its exact three existing files byte-for-byte: phase lock
  `ac12b9f21977a3adcd61ef5f0d8ba60b058306dcc05fdfed423d2ca77c17a0ce`,
  phase-lock result
  `6a2f63dc8875db394333ac901a919466a6a432083e29feba32ba8917f3ee9bcf`,
  and zero-work marker
  `e99099b87aa6417b4200ee236ef2b770d1524d11b26a878e9f3bf0d749a54cff`.
  Never retry, resume, overwrite, append to, or clean this spent namespace.
- Preserve
  `J1B_OPEN_FAILURE_TERMINALIZATION_CHARTER.md`,
  `j1b_open_failure_terminalize.py`, and its focused tests at SHA-256
  `62c7466dde4e9022723abc3279583db64d6e7ba6017e91fb9ede99aaa2204946`,
  `1396fd2f4d8080970ab9575f56c0bb39f511214ec8f2dd2a0f11534964ec6c71`,
  and
  `8ada63aeae9b85d9edd71810f7973f75de9edb986ee3c3d09339a5b254ec919d`.
- Preserve all three files in
  `threes_rl/runs/forensics/j1b_open_failure_terminal_v1`: test evidence
  file/payload
  `38261821557d3a49dee81bb7cad02fa2c91ac058aef168cbe39abfa52caf7155` /
  `46e589fc139bfaa8e4475cdfa678fb9b6cb914646777a5836db981f043215fc2`;
  terminal
  `2f9cdfacb04a064b67785ab9bb00cac7d3d46bd057912b40ac4c06db0a0ed122` /
  `1cf98c5676b23c6168be4feef4d3e3a4ffeb98fb90f028af515f1646eb5e2369`;
  retention
  `28738328f724a544ee92fc7992ef8f256f0886c2e138a234a863ec0fe55c5f67` /
  `f88941a61da8909bd852d180892ce6c22d84c8ef4749b2110f9f2d58db8dd37a`.
- Preserve the authoritative all-zero materialization, ownership, stream,
  commit, game, optimizer, checkpoint, outcome, holdout, human-session,
  incumbent, dashboard, and promotion counters. The J1b-declared training
  stream ranges are spent for future allocation despite zero consumption.

## 2026-07-27 J1c Training-Only Readiness Retention

- Preserve `J1C_TRAINING_EXECUTION_ORCHESTRATION_REPAIR_CHARTER.md`,
  `j1c_training_execution_surface.py`, and
  `tests/test_rl_j1c_training_execution_surface.py` byte-for-byte at SHA-256
  `e352262614a7c3c46c53811c727599f9926f6cbd579b99732c6802c8c41462dd`,
  `f50b475ed00efcfb0fa2ac5b4e4a11b0587ec17c4e8e404bad08be8f4f8c990d`,
  and
  `4ff0a2253cd23059d33404b5d3f0829309dc1565657547f6112e9c3d268dc86d`.
- Preserve all eight files in
  `threes_rl/runs/forensics/j1c_training_execution_surface_readiness_v1`.
  File/payload SHA-256 pairs are: prospective manifest
  `135fec4c75db8871e20ab3988471f75538a399e982573bf4a108afc569fe08b7` /
  `c0d7953aa158297d5b515f6b6e4613b6fc22acc059e8e0aa0ff8904ee7e3546d`;
  compact stream authority
  `8aff7f07827cfe796a07646215362f1d37e502f64a43b9b7142b53288b7041f6` /
  `30439292fceeeae4832f5259c62e8a954f3103de7246875c353dfb8cea138016`;
  test evidence
  `7a467bab2bcf457501413b64ed9e3dc41a36e881eece7200e0cf66806f38457a` /
  `c68c87a3361ceacec8d991c513589edb1c0d727e2183c37713ce54e9478f3048`;
  input bindings
  `c230a951d4c44c4e6d67f59c0405bb78c1b8f4ef980408de126bc8b770b83fef` /
  `30f911740e8dc99a62cec63439ec522d4d28fa5db3cd645fea2267c8cfe787c8`;
  schema
  `54a23f1bf627573652f6e1455be612d3a36e417543b23efa223a3bea8edc36fd` /
  `ab2deb5d5f09e5e65a738e96306bf05e21017dca6737fa607c5f14b92bad5d0d`;
  projection
  `c6fb7a5d426a3fa32c0affa4d34b937845a7cdb02e22dcf2baff657d3540879d` /
  `60ab68e30b48ee5261b9ea8a661da887b6dad8b8b466f970ae58e03d6b811c17`;
  readiness lock
  `a95712126796dcd91a82885aa1990a77e725064970ba34bc1f31306de8ef2368` /
  `15701d62e5ddc7fca7e38702d78ddd54fa7aefbbd6bdf49ea130c2e224f78ef4`;
  READY result
  `908c1570f972a612e02815811a9885162a89f9a1e87ea2b081f2801dab7419bd` /
  `e13c27501231af3ea72de1dbea8ac2e9b485b7763e24fb3394391ec326ef37a3`.
- Preserve the exact fresh ranges, root commitment, canonical rows/root set,
  JSON-native exact-byte writer, clean-process real-audit roundtrip,
  training-only bounded dispatcher, parent scientific identities, all tests,
  operational/projection evidence, and J1c-labelled all-zero work counters.
- Keep `threes_rl/runs/forensics/j1c_execution_v1` absent. No J1c phase lock,
  marker, materialized manifest, owner, reservation, consumption, genesis,
  game, transition, label, optimizer step, checkpoint, development or
  confirmation read, outcome, human-session read, incumbent/dashboard
  change, or promotion may occur without separate research-lead
  authorization.

## 2026-07-27 J1c Training Terminal Retention

- Supersede only the earlier statement that `j1c_execution_v1` is absent.
  Preserve the complete
  `threes_rl/runs/forensics/j1c_execution_v1/training` namespace
  byte-for-byte. Never retry, resume, overwrite, retune, reinterpret, or use
  its streams or quarantined checkpoint.
- Preserve phase-lock file/payload
  `e22b251e17d289575f03503c7374139f3036c23c4216f77f0e7c19c47a88a5de` /
  `281140be80474fe091d950c0ecea4dfd8ea3361bc90b6e8740617f39692f2b0f`;
  phase-lock-result
  `a57fbc12bc7cc9dc16d29ce35fd0d51bceef703bd23bc7c9d2d84cb720f5836f` /
  `e5c8366661c47709975bfb3e034bdfe496dd7ebf071a5cc7ecb399607f699ba5`;
  marker
  `1c118ecdef095774d20572ddff71cd403fc0a473826ccee491e2fd0fa18bacc2` /
  `0d39429fd95c52882594d98420dc954b5fda722ff9ff641e3c2da20a8c3a59a8`;
  and root manifest
  `0d975d72384bc6ea92ceebcad16ed447aed4c2c666fcbef947369c50b382de33` /
  `071e06646dd2b7d39d9792dde18c1f80cbe9bfdc19620737f680bd940eab975e`.
- Preserve reservation
  `90a6e67cd462ded225853a6fa8ec1cd82965cfc1f86c5bac5d322ba8605615ff` /
  `bdf0ecd28a628273f55199a15d650bdf18fe98755b9727d7a747cb83eb3abc73`;
  consumption
  `6a4671c2c2e92f58b300aa311b3c36d82135abd21c2453f58ae65571400aeb36` /
  `37f4367d15038a8faef712e3390fd8a9c06ba007b64eaef900dea2a8c0eb281d`;
  owner ledger
  `c69ea70c4551de20195e73ee183d854f098926bf08eb2c86b6caaf8c78bf855b` /
  `2efa7f5c924ab697e2c487f97a7116f204b3ce4244dff156659dd19b1981ec09`;
  and final commit-head file/payload
  `50e4c2868eb224067d32efb90eaa6ff1053b0c2ee4c57d5f1e51af34bbd4080e` /
  `d0111348453205ed6ab0683d2fce0a943068f80ec5588b11b91e701a71001e36`.
- Preserve the quarantined round-64 checkpoint byte-for-byte at SHA-256
  `053e6a87441114595e09b9fc6a0f7ce71e5acd23c9757da3aadb96df18468c79`.
  It is not an authoritative candidate and may not authorize development,
  confirmation, evaluation, promotion, or future initialization.
- Preserve terminal file/payload
  `7ec4fe7627a129dbb7227fcb88df87ab46ee87479d381011103511ec8f2ca414` /
  `c71dac534755add014a0debe6418f75b591df264f1bb096fb5d50fd253d8ce4f`
  and retention file/payload
  `8946669ffeba05626ee863f4e2df8536e267920d771eb088ebf84253a2059532` /
  `01a6f95b79c10f343489fa2e2add086001e89691c7c4edf6b6a1b19f8ec66409`.
  The retention inventory binds 18,835 files, `4,299,100,371` bytes, and SHA
  `65e8e32cadf6246f3157b6c06da2c63e7d669692196fa59bfee13d768d4a3fe3`.
- Preserve the exact 16,384 roots, 65,536 spent stream IDs, 321-record commit
  chain, 2,479 completed attempts, zero abandoned attempts, runtime ledger,
  rolling state, root blobs, retirement manifests, and all source/readiness
  parents. Development, confirmation, assisted sessions, incumbent,
  dashboard, top-three, and promotion artifacts remain unopened or unchanged.

## 2026-07-28 J1d V2 Readiness Retention

- Preserve the J1d V1 charter, runner, tests, and all four files in
  `threes_rl/runs/forensics/j1d_metric_authentication_readiness_v1`
  byte-for-byte as pre-correction evidence. Preserve test-evidence file/payload
  `d2957b138a734dd3185a1b5f09eba782a7632027cdd06b0cb0ebd4042e7f064c` /
  `4e0a6068289472478fddf199c86a803b1dae53b6f55fbb5e32bab0566ab0b3da`.
- Preserve V2 charter/runner/tests at
  `f3d42d6f4d908c723756e140fc2ba424378f280a18dc99a50b585e59478cd07c`,
  `6ee656ae0288877560df5a6a140777bf341f8a34dbe554c61ebe2812e6147a3d`,
  and
  `9148d70b3d8c8c55b27a75829ef2e5b4df142e124d6265b639667049b4ac5868`.
- Preserve all nine files in
  `threes_rl/runs/forensics/j1d_metric_authentication_readiness_v2`.
  File/payload identities are: manifest
  `c2be5faf37d9e2619c0bd57d12a64248738e6b4c8bda1802931898a63e18b1e0` /
  `f6da9b35674a08c21b53c476692cd7073e492289a8cec8d687ceaa45afaf092d`;
  stream authority
  `4e8e1661ab04c3d87c5819e0112d27b8213f65539c2ea9b955fa6a1a47fca867` /
  `90e98195d0be7a50d38c0c00e681c120c9a8300c5d6510f836809088fb2b7c6e`;
  root cause
  `4c6b51462138e0fd88e9a926be860528889ed584ac870f48b48296b4174a1e47` /
  `aa9b114642e62c8424a6bc87bbb6cc1335f91d46a03dc7fe7a6ccb6597a6b86e`;
  test evidence
  `b70755957444b39f97c24985c0ed5393abee1a8b54b2e43763aa2381810265bc` /
  `af6add4c249f8092ea5a3743301d302bff79011935ac4bed1f6ea3e3f7e4f08f`;
  input bindings
  `95975c89dac4cdad527b599645566bcfddf01b175450a9380614a03199a35918` /
  `64484fe39ec3da178774703bf6aa9dc70e8fbe16d51a97d8d3a63a51a21479d9`;
  schema
  `d960c55fb42f2c403db2d5f9b8b2380e2c005a8d19f2e62923721b2dfa56e328` /
  `bfc72eb81f53790acbcfff0e33901e7280f76407206a8298e855e263343f166e`;
  projection
  `98090b122a754d849df0fa49006a79e414cff74e04b4ddbd7372248f0c7b7fcd` /
  `9ad94e24acf7632590ddc8e3ee2a4bbad9ebb597a0bdc73639ef0f91440b2640`;
  readiness lock
  `60587f40512555dadab5cc09a0e9802039754034f427a6084b11b7d8146627c7` /
  `bbca2deb85cee5abea8fcbe89d9917797c7bf20655b3590b9ba66468f422f7b5`;
  and result
  `b891a0d63fd0c532387a64dc719ec20f27dcf15c84aeeb3a094470030076449b` /
  `f4af39cc6c2f54e3e79aef76a73fdddd141979faa684794beb7deb59291f3693`.
- Preserve canonical rows/root commitment/root set
  `7bef7fd71403bbb26ffe3fe8293e6745e8aa3bb585dcaff53f06bdd2b36cb7a1`,
  `13359e460b2956b94e328900c077b2a1d9aef12b00a9f2c882f780abefe0ce47`,
  and
  `4d86d6e934d1982c0efde2407ef3b205464dc3063db82d0b338dda1bd16f97e0`.
  The 49,152..65,535 prospective stream offsets remain unreserved and
  unconsumed at the readiness boundary.
- Keep `threes_rl/runs/forensics/j1d_execution_v1` absent until separately
  authorized. Do not create a phase lock, marker, manifest, owner, reservation,
  consumption, genesis, game, transition, label, optimizer step, checkpoint,
  development or confirmation read, outcome, assisted-session read,
  incumbent/dashboard change, or promotion without that authorization.

## 2026-07-28 J1d V2 Training Terminal Retention

- Supersede only the earlier statement that `j1d_execution_v1` is absent.
  Preserve the complete
  `threes_rl/runs/forensics/j1d_execution_v1/training` namespace
  byte-for-byte. Never retry, resume, overwrite, reinterpret, or use its
  quarantined checkpoint.
- Preserve phase-lock file/payload
  `dfbe49bae84e0cfbb2759ef64ae46c55021430339792776a6a983a8210bf052c` /
  `1627d7b7dff536d0cf722472a309631975449a8d04de7c18fd1b9b5e31034fb1`;
  phase-lock-result
  `5ee4ee0b8166a7060518c2d5ee9ff1c8dac3c378be837aebd73b7a2c5ac3ac7d` /
  `ae7eca01a3b92bcbbd2d19ea62c2040d9d9f0e4e28170e3124af793059329c2f`;
  marker
  `c564f7fd5cc80909dbee2c53390f3a6270439aa8f8a8d969c3fa35d6814a023f` /
  `631c34174a034d3625a1fd40e792e6462b1133c9a5f3959ee326d63be337844c`;
  and root manifest
  `134f402ff92800e2b9053bca8ca176ceea64b686f6cd3ba9efc02243f946d45c` /
  `87f764e8fadbff029aae24b4666086de01b3c9d8168a7f07331eb49dc473fbe0`.
- Preserve reservation
  `c6ed80500b2e003165562e8a8e71200ef12e41286fb125e7fd3abbd469b33a69` /
  `f730564f217d11040761a51a7feb1c272788b897b2be045e3b5c9c7ee3c03a2f`;
  consumption
  `34f9622ee549f039bcbea9e0490800416961072c349e86891b392963cbb76622` /
  `cc338f3b402df6d80897aae7e16ad02e05e573dbaab7b9a59b60633583c92d15`;
  owner ledger
  `a55ccc5ecbd4ecdd25d7c1b61a790046c8244eeaa33496350c3839e4b7489289` /
  `27a7b441f88b6fbb23698f229284e3ba98295a63b9f7afe11344b0c16149dcc0`;
  and final commit-head file/payload
  `a7f5ff94db795f0c7e36893f4ef4e562004821bd88e915601766389a7f53bf60` /
  `e48d85eb1674137b963cb9b8e8d9609f333f71f3259c8755c9dfad7b1f6ce8d3`.
- Preserve training-sanity file/payload
  `2faba052c943552c37dc3fe36fd82cd44e1a74ca4f8b29023e64464ffc8167e8` /
  `2bba7980191d66d2721556456c7ea65ce426aefa02cb1d5d6d508c4ff49e906b`.
  Preserve the round-64 checkpoint at file/payload SHA
  `cde85c1ca62b9bd045d680ec980ec25e58ae6e7e083b7ccbac1e239cfbb1a41e` /
  `8f2fc16bb7bcdf5b9b5437d95b4b487fc8029d3aef5817494bc4b54223975d60`.
  It is quarantined, non-authoritative, and unusable for all future science.
- Preserve terminal file/payload
  `9ab0c76142aa70041a5f0540abbc3f9b77ac197599f607a646b2952368f13e1a` /
  `e37a32ec2d0ef1df78d804689ee8f529e5cc78bb627b34fbc8728b7840366fb6`
  and retention file/payload
  `5fe222bfc3e1681ee3b1cb98db71e2a0b90017c869947329ca49df084ed65518` /
  `39ff3a6f028f7b27ddb775270959b2f0964e7bab8c649e20942f7296e8dbfe2c`.
  The retention inventory binds 19,092 files, `5,093,140,624` bytes, and SHA
  `c8e7dcaa47b40b59ae538306080a9536bdfb46104dadc49a5e660ad81807a815`.
- Preserve all 16,384 root blobs, 65,536 spent stream IDs, 385 commit units,
  780 optimizer steps, 64 canonical metric commits, 2,322 completed attempts,
  zero abandoned attempts, runtime/ownership journals, retirement manifests,
  and all source/readiness parents. Development, confirmation, assisted
  sessions, incumbent, dashboard, top-three, and promotion artifacts remain
  unopened or unchanged.

## 2026-07-28 J2 Outcome-Free Readiness Retention

- Preserve the charter, runner, and focused tests byte-for-byte at
  `3cf410a4da9418c9e06164ac077d3e389f77720d056dfe25ced2a4a2a052163b`,
  `9ecd658ea69968feb605d0e0a9e4e621b73ac01619536e45c0cdf69b7bc3b15f`,
  and
  `24736fa56702c46b24d515716d7a6365dadb49b20622f333bead39d3105ebdb2`.
- Preserve all ten immutable files in
  `threes_rl/runs/forensics/j2_incumbent_distillation_readiness_v1`.
  File/payload identities are: test evidence
  `32ad0836ff55501bdc3f78bc49d58a44d89ebc4544c7e721c4c7b7f991cd6e53` /
  `2ae7371b2a946eb52b0942387acd0f599e42065b996f04f09d53a7506c7c82cd`;
  input bindings
  `06656be5428ce57cc29960988fdbcdc720844ff82dc2c45007f4e33366170416` /
  `651d66fc6328e35f1932e937dd819eec037c8f84ab11b994ca47815b1e5184a0`;
  prospective authority
  `cea6f129e0dbb5309d67d554a74ddb8965e6c5586efb36f570363d7d370707f8` /
  `631ed382950a30dd51790ad94cfb9fb56b78f9330c87d794d17977e9d14690d6`;
  protected stream authority
  `b9e806e13c28d33f0edabe756ed06b49c7c5e880bd8370de99b007c0bc9d28db` /
  `51fa8c173049b01a3fff19860968de2bc4d09521f5cc3980ab0da9ab4add40e6`;
  teacher provenance
  `824aa8988136d81a00d81dd4899b9985aedbbb213260d3a2e94c4e7dc931840a` /
  `a8d355bd056bdd31f860a668d4e86a0898866192b39cf0665d348db33ac02768`;
  model schema
  `ac976d7c392b211bd3791ef218ec6da42b55aabe9a5e0643043ce4704200d056` /
  `679ee9a64c53f8dc821dfe173bc8b0c1ee807d0bb7e808a5342cb5cb71182867`;
  power/feasibility
  `b210be7d16d27d1cb4fc419f38952aeafdd4d3939497f9b4d06df5e9f65ef43f` /
  `4bf3f3fdf32d0929bc2dd0dc5389b3c2772b18677a68f73d4b2c17c3c141b64c`;
  runtime/storage projection
  `f59740b3f3d6f15697033f769267b233640978dd5947beb203f2f62de5643f68` /
  `45d3aa109244792165ffa9e9a9ac231b063814386ea4d905668dc3adb3da44d7`;
  readiness lock
  `c3f08429b625369263b75a5724b3abfdf2487d6a9fd2414897c7aaca7fd74488` /
  `a4683de92f833c4f33451b9f73acc0214566ab2d28d45a4e95e49a6d07372c8e`;
  and readiness result
  `8c24be58bb6a30cd2cf302f17894b69e131f3b3c6092a4e71801c6b0f2f96eab` /
  `4110e987eed93a0b50cf8dfc3978469f316039edfe03ae22549daf464ddf04de`.
- Preserve the decision `HOLD_J2_INCUMBENT_DISTILLATION_PREFLIGHT`.
  All prospective 227B-249B rows remain content-blind and unreserved;
  no future J2 execution namespace exists. Do not create a marker, reserve or
  consume a stream, query the teacher, generate a game or label, step an
  optimizer, create a checkpoint, open development/confirmation, read a human
  session, or alter incumbent/dashboard/top-three without a new
  research-lead course decision.

## 2026-07-28 J2A1 Execution-Surface Readiness Retention

- Preserve the charter, runner, and focused tests byte-for-byte at
  `dbe3470f67229c086f514de20efdd2daf074329df81ca66611895fecabef8f61`,
  `b5435d6d5d0999b035220a6763646ee133b23f06e79f45456d9c5af083dfe8c1`,
  and
  `bb1c8fffa52dea332032447f60426addfbd0acaf2bc5453feb8620856062889d`.
- Preserve all eight immutable files in
  `threes_rl/runs/forensics/j2a1_distillation_fidelity_execution_surface_readiness_v1`.
  File/payload identities are: test evidence
  `dc0e42d0e11524fc7cdcd354841065d98c8a9ea1a7715badf9d1257e193dc105` /
  `5d081e8336780df537cc9254f055deb80e270bce6df5ed4a757007998bcef291`;
  input bindings
  `427b9260c1844d3701fcbf648b9ea5ce7a5c8e546bcc1fcefca213d4819a45a3` /
  `537d16c3334653b5cce20315f31a42fd9cd8679839bdfea7677212cc1da5ca1c`;
  authority
  `e21019dcd8deea7f7ebc31adccd01239e8ef991585a3988b445c6f5ea12c65cb` /
  `05ba18673bc6d5b2265c718df856fb1ade2080d942aaacc8e14d6f3f8655a344`;
  schema
  `d7e73f5dbc8b0e048e505e99b3bd345e4138e7dbd8b72377a7711ab1b755ef73` /
  `b677865fac4628d37cc2caf04eebfab5c68e050e84a8e75eb8e6a27016b4bddd`;
  projection
  `9b78e744690f4df0c7f0e04512ab16a15f45528947ab554762db110d57d987bc` /
  `e63cdb50f96f27eeba9168af2c82cb235a302a26c9e0e7fb10d29366c9061f15`;
  readiness lock
  `1aefee84417c4dda5f17f0309b7b5fd18e2f7a418635f8dc16a90e9c5503da13` /
  `09318a12c1f1789753f416a37e91d5b59cea5a507dcdfff475486ff13dba61d6`;
  readiness result
  `a90d1600502264d42315e0806d7665be679e06111aacc006dc193e88baa97d22` /
  `e118a58de2357ff4d870283cd46a2e23696aad42303427f31dc2e6befc3b9861`;
  and retention
  `dd258ffbded154ec299d5b48368cd21c9d35e585464079d57009a0e960eb28eb` /
  `ea339915bb05f604bf9cafc7e319120354df803acbe8834e8c88ff88442db854`.
- Preserve decision `READY_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE`
  as zero-work readiness only. Keep
  `threes_rl/runs/forensics/j2a1_distillation_fidelity_execution_v1`
  absent until separately authorized. Do not create a phase lock, marker,
  materialized authority, owner, reservation, consumption, genesis, teacher
  process/query, game, label, optimizer step, checkpoint, fidelity result,
  PPO/development/confirmation read, human-session read, incumbent/dashboard
  change, or promotion without that authorization.

## 2026-07-28 J2A1 Authorization/HOLD Retention

- Preserve the sole authorization artifact at
  `threes_rl/runs/forensics/j2a1_distillation_fidelity_execution_authorization_v1/J2A1_DISTILLATION_FIDELITY_EXECUTION_AUTHORIZATION.json`
  byte-for-byte. Its file/payload SHAs are
  `29ea95388165250b7b7f7db909698ec853101f85bf62e81445e23540879e576f` /
  `b2f5792e16be8ee8e08109fdf894f6243ee7dae840706656ff349da0e71c277b`.
- Preserve the accepted execution charter/runner/tests and eight readiness
  artifacts unchanged. The failed pre-phase invocation created no execution
  artifact; keep
  `threes_rl/runs/forensics/j2a1_distillation_fidelity_execution_v1`
  absent.
- Do not retry the existing authorization or alter the frozen guard. A future
  attempt requires a separately reviewed, separately versioned orchestration
  repair and fresh authorization. All teacher, game, label, optimizer,
  checkpoint, fidelity, PPO, development, confirmation, human-session,
  incumbent, dashboard, and promotion work remains forbidden.

## 2026-07-28 J2A1 V2 Readiness Retention

- Preserve V1 and its spent authorization exactly as previously sealed. V1
  remains forbidden for execution reuse.
- Preserve V2 charter/runner/tests byte-for-byte at
  `d9c5382d803c606c29415fc020fa7d63762dfcb053232d1ac904f21827d74dd4`,
  `044a67bf9b34b311787e3e7de246c4ce62a33f4f8ae47d211f6a76dd231a22f3`,
  and
  `b211bfac0bb2e18c87dddcd72a0c8e7f1a0c3cbd76fee92572133aefa7abd95d`.
- Preserve all eight files in
  `threes_rl/runs/forensics/j2a1_distillation_fidelity_execution_surface_readiness_v2`.
  File/payload identities are: test evidence
  `8c3225ba74a66ea0a8817c3623a368b10430031e29afbb5784cf92ae374795c9` /
  `a8197ed0a442f314d9da74c310c7add8f7b8b77172a52c95385a7648eb16748e`;
  input bindings
  `0936290b049bcd24a580f13a24823a2e32e4997411ccee2670e2ee87df8678c1` /
  `58c7f8c7ef2a59b1501a8b00c21271aa80582fbbbd7ab365445e8b6186fd8155`;
  authority
  `b48b4254b0272283542902710f86aa9d39aad136a523ef33753e610c2fb401e6` /
  `ece84c85385d88c00507b9180d412a6892c98491541320b25afbe3528b104b29`;
  schema
  `f72f8ad12f3023b8f4215975c03b1d1e4b2c1157712b0e75bb57a5c61fa1ab7d` /
  `792238fc213103a0bc150ee6b4cfef4fc924b953fbe32e191c6700792e8d87c2`;
  projection
  `19187c197541729227140cbc70fa54c3c7f824feccc2b0bf978ea7ba599ccd90` /
  `49fc21aa45d5c07147b33af4db5a78dd19646d6d708cc20a9c709ba41180a717`;
  lock
  `259df7e65be1e9cf73e93424cc40d4dadb6e27f87593abbcaa9d577e14d49702` /
  `fd2ed8ba9713d799420d2eafa264b4ff3185384811c68fbeed680a548d8fab31`;
  result
  `c445d7ab1478b22b7bb7d74e06533566e519a4b400ced5db09555833fd3ad045` /
  `567b5c58e89a66e0cc0040515f3646c0195263f40882586c05c8bb74844dcedd`;
  and retention
  `e2b8bea7a7570b1268339d68cb063b11d8e84d2e3535f4a10b352b1c0590d068` /
  `48568dc9313de9dc71826c722999bcce4e2e5d7605534d2eaa9eb4493cb5675a`.
- Keep both
  `j2a1_distillation_fidelity_execution_authorization_v2` and
  `j2a1_distillation_fidelity_execution_v2` absent until separately
  authorized. Do not reserve/consume streams or create any teacher, game,
  label, optimizer, checkpoint, fidelity, PPO, development, confirmation,
  human-session, incumbent, dashboard, or promotion artifact.

## 2026-07-28 J2A1 V2 Execution HOLD Retention

- Preserve the sole V2 authorization artifact byte-for-byte at file/payload
  SHA
  `8787804d85e22d6720b6428feaa2d9122424c620c977becf1d106d88fc58e68c` /
  `5e21b9ae764a023c567a9cfc8e495b6569322c253893986205e3ef98421b090a`.
- Preserve the complete spent execution namespace
  `threes_rl/runs/forensics/j2a1_distillation_fidelity_execution_v2`
  byte-for-byte. The authoritative terminal evidence, retention, and terminal
  file/payload identities are
  `6a855bb18ca73cfef3dc465a3885e88901a317bbb08ce7624f2fb726438fdc7c` /
  `304da0e20042485e5e913d65e99cce81c93b613ad98b68f491085c772f5eeb5d`,
  `93f3a5ac0e155b16af84fc06165cc4e23cbd4184b10b96cc77dc9870b1c315ac` /
  `d0f637b76345694ded4679afb1fe6740a55065a135263398db59a9c0df3cad74`,
  and
  `13dd5c3a8eeb79d03149da0fa99a19aee3e6a657109e7fe4104a149d5d02ca6b` /
  `c3ad1135034b33a6118d3239f88e94a760eaa9afe98a9ca589bd70e351ce91a3`.
- Retain all 3,048 completed teacher-root blobs, both append-only ledgers, all
  chain artifacts, and the full reservation/consumption authority. Retention
  passes for 3,058 inventoried files and 1,782,523,714 bytes with canonical
  inventory SHA
  `54baf47f3a3c0ba72e60b4f74d9351ce8fffc42de2ded93a8593ae35c38e7642`.
- Decision `HOLD_J2A1_V2_DISTILLATION_OPERATIONAL` is authoritative. Do not
  retry, resume, overwrite, reinterpret, or reuse any V2 root, stream,
  teacher trajectory, label-bearing body, ledger, or partial scientific
  artifact without a separately frozen research-lead decision. No checkpoint
  is authoritative. PPO, development, confirmation, promotion, human-session
  reads, and incumbent/dashboard changes remain forbidden.

## 2026-07-30 J2A1 V3 Recovery-Readiness Retention

- Preserve the V3 charter, runner, and tests byte-for-byte at
  `4638ffbcc67806742a1683d4aeec39a9669055d2051c18768a5ea5cd68aa216e`,
  `c4bbb5b79a6a8df17e4b97663f49e0a5a2db06fe8d7b5e8bb483aa67ff3c8c43`,
  and
  `577dfe0fa434cbb8d95e6e780d5396e692ae63679e57a6ce64cf4214ab9b7705`.
- Preserve all nine files in
  `threes_rl/runs/forensics/`
  `j2a1_distillation_fidelity_recovery_readiness_v3`.
  File/payload identities are: test evidence
  `a45fe7b7e1eaf564e55c91ab9b0e25488fd958020fb9dc3138077df7294d808b` /
  `0ad095a211fb69eb12892faa9f9e2047317b3cd3367a933dba476e1797766d3e`;
  input bindings
  `c6678f3bae0b7d85c4caddf09e44acd35932050a28b2da4afe67fe940a8531cb` /
  `5a87f7d02a74f6493bc15683fa7f5a410b3fa89dd57866e7ea82a2820f51a275`;
  V2 integrity
  `b4527df82c154e8831965050ef5eb6a5a124d5ed6e383f5402254d50fb556d58` /
  `8c5a2e5c8de7f0687084957cccc42e04e6dcf5da17127a4aa36b547698be0059`;
  recovery authority
  `ca6f1bd99a9c6d3654e4af04227a1aad0f1d4d012f0eb3cedea1f3e405523691` /
  `13ca1ac368dbf679d39ba6d563deaa759dd8ed167f251641e95f062820c72a4b`;
  wall projection
  `0f949e4e84495354f567ae91a8e848b113dbe7625bcc19b302a0c77351eb0f99` /
  `c544b2e9285a60be7d654e34235a4ff211c58fc8b2d8e3b9d5aa37bbf6acda72`;
  schema
  `bd9169c544cf9d1f212a4995c0727e6ca101bf18260ae603407f4062f0d35c4b` /
  `29f7a421c640a927f36da9c2947399b518e6da5cb24b8945e65e6322801dac3c`;
  lock
  `0f8d4f916b9672dfbe2844595952053c81f153dd6b2d01bd8c30486204bb0153` /
  `d6bb27e29bda678afdbb82d5cbfa342338c2dcc8ff053fe09d159a3432879ca6`;
  result
  `23199ead16dce7ac87ea7d955bba5c913be632f624fa8771fc01a07669ab33ae` /
  `66ef7008eb71e3ebf088d908170f510e823a884adbb07e35daeae35b17a8cc56`;
  and retention
  `26e07603590d39e5402e6f95f35efc94933c83d68931e42a0b1de4b9f49c3246` /
  `2c4072d9a6771b89728336b57062c4d0b1ef06de77248241a05c35dabde9ee4a`.
- Continue preserving the entire spent V2 execution namespace byte-for-byte,
  including all 3,048 root blobs and its sole reservation/consumption
  authority. The future V3 execution namespace remains absent. No cleanup,
  collector, teacher query, label, scientific read, optimizer, checkpoint,
  evaluation, human-session read, incumbent/dashboard change, or promotion is
  authorized by this readiness seal.

## 2026-07-30 J2A1 V3 Recovery Execution-Surface Retention

- Preserve the execution-surface charter, runner, and tests byte-for-byte at
  `674ed0e1c67df0cbc8645a2190a5632ce70c9cddc5922ad0325a9e53d14c481c`,
  `611dc428a3f940ff1db15ae58e960bab27ab7307c36393bd23a7400e9da12c02`,
  and
  `1a1dbb1039b9dd7d57d8d9a88f7cd81dfdd68240bd16b23357df6f5c5eb01df4`.
- Preserve all nine files in
  `threes_rl/runs/forensics/`
  `j2a1_distillation_fidelity_recovery_execution_surface_readiness_v3`.
  File/payload identities are: test evidence
  `9a46fe4abeb4cac94302ae2ad746d83ce2da9d5748a377db5f90ecd6d0e83b99` /
  `66d77e3cc0d217d8a1059736913271211512526e8d149e672251de24c0efa0ae`;
  input bindings
  `b2343a23023441331f86ded68426cf5babe3e73bc7f860e9c029faafbeb46e72` /
  `aa1a427e240d769cfd8dee530d439c8ed0ad235f4814c57761488ba71bd7a58f`;
  authority audit
  `fbdd57e403c614c916e3317aefba5ce3ae37c3e03aaa61b7189f307c7ed84069` /
  `5a018f630fc2ff7d85911fe00138e461c67c8d29b54f04dfb7d340ad1317294d`;
  schema
  `d3dd83fef73131b84b8ea7668e73d94e5b3dab564c7312f4840bf822f9ed65c4` /
  `dbdaeb5f16fb14c7016e06306967931265b9d25c23b2eff5001e0a9ca5704e9b`;
  projection
  `39dba009a461ae512121b02a206afbd89f058b38df93786831a7831533f6df4c` /
  `c39647126dd9aac79c813f6e61515ef49f0e459785d3052be39624de2f9d250e`;
  state-machine audit
  `035c61e79e9d191684d04885c19f4cefe595c34070b24918186fef96e5ef4959` /
  `2bc15d93284c19fe140148f0df26c933961495fa22fba1e9361a4fc6dc637b0a`;
  readiness lock
  `ba44650eaead39de45465ff6a785d7a30aaf9c5740294b2516e70354287691ef` /
  `8d4d87baa92ea04730435d2798d9b9bed0088bbd757443913d3fc9eaafb0bea3`;
  readiness result
  `3bac460ad19a32b249b199eec66d6aa7cc9f27be83eb2c4842412868e81ac610` /
  `c2626a14d8b86613d05c0934e4735171cc3242f7841308914a669c3666cb7bb9`;
  and retention
  `7f9def1579f2414dcbda7002ee5f7519daa86ae86986206d1a4dbbe7348a701c` /
  `55dde2fe501267adb635d48849144ffca046d9e40f29e81ceb45acfcb488eeb7`.
- Continue preserving the full spent V2 namespace and all nine V3 recovery
  preflight artifacts byte-for-byte. The 3,048 V2 roots remain body-unread,
  the frozen 11,288-root unfinished authority remains unchanged, and the sole
  V2 reservation/consumption remains the only stream authority.
- The future authorization and execution namespaces must remain absent until
  a separate research-lead decision. No phase lock, marker, materialization,
  owner, collector, new reservation/consumption, teacher query, label, game,
  optimizer step, checkpoint, family/mechanism/fidelity read, PPO,
  development, confirmation, human-session read, incumbent/dashboard change,
  promotion, or cleanup is authorized by this readiness seal.

## 2026-07-30 J2A1 V3A1 HOLD Retention

- Preserve the V3A1 amendment, runner, and tests byte-for-byte at
  `397db39026b5eb42d6f1ed633f11de0d1c773ebadcdd69d8e9cce2d7811f9c5f`,
  `64cc16d99c9366d8c968486e8ee159b9ca1326ea1de1fac2ec44c56cea65cbeb`,
  and
  `ae4c689bee1c852b024cfa8451eaedf2e9ca160a4135189abd4c48b50dde6012`.
- Preserve all ten files in
  `threes_rl/runs/forensics/`
  `j2a1_distillation_fidelity_recovery_execution_surface_readiness_v3a1`.
  File/payload identities are: test evidence
  `8513f90bcec408b0927c25c6d550db7babcfe4b637a0ed1855b9e889fdb06312` /
  `fa4ce8e52f1bd5922f697ae284d5dd27a000a40c9fc9b3ae06f0400744e23eb9`;
  parent HOLD binding
  `10b2bd10f7be5228d6b74b85de9f2aa9b62b185dcc172a7a39bb09e364bd2ce5` /
  `61ccd1577a2cb5b5820dc325160bcd14fb212cac24f34dbb241d85dcfc792c16`;
  input bindings
  `84f54048ac76aa0d222b2b8d676d20fd434e447fe07fd54640bb620f4e06cf8c` /
  `af5d7f0ce89f716d243e65b911ea5b67bc91918a7226392a369fb8046a7287c3`;
  chronology audit
  `e2209835c1a6907f64556e1291ee468da334d7f777ec08aa9b03c4b173d46eea` /
  `eaf76ed13376d10290d87e953e8ebb10154393733040bf2a0382a0077de79201`;
  headroom audit
  `71317860515fae1b416c3ea4cfa5604c501c40794332a8083a7bddcb8d62b5fc` /
  `1fa040a157030dbc795dcc170a570b29bb8bb42a43c17480302298756b565a34`;
  no-delete review proposal
  `bd29e699afbafb13468a6de8f5fddcada8e933490d5f1c6ca6c2938c33a357c2` /
  `2a63b1f53086140e1d2422e55edfd13be4d805c1188c0426e0c2c2f22c51cdea`;
  schema
  `0ecd0c54833751a517c7054e56908721e3835a2c253be3be95e0390c38337399` /
  `142c7648d28ab10dc3afe53c73abc318bac8ac8328b4ffb8ef041d148731181f`;
  lock
  `eba9434e05fd9dcad42b423c4e4e88ccba48e901019563613dd1fda7f141e06a` /
  `43ccc9ff3830a3c6442916994aaf86457cc5ce07720ec0f3238494b0d38e452d`;
  result
  `cb06f22c90b0df34a5eded5fe90fab25d875cd1ff7cd8e85db474075c21a1fa8` /
  `5082df80e4c93d02416b01faa659bdc94b7ad547094a622852313e614a2c3798`;
  and retention
  `5dd3973c3930c30fd531ba7b67d75535ec5e7192586e234856e60b19cf51a500` /
  `5ef75defb4e622013e5fd6a9abc8238f9701a07fb31b194fd0b8b2a9c33fc442`.
- Preserve every V2, V3 preflight, V3 readiness, and V3A1 file
  byte-for-byte. The V3 and V3A1 authorization/execution namespaces remain
  absent. The review proposal contains no candidate deletions and authorizes
  no cleanup.
- No protected evidence may be moved or deleted without a separate reviewed
  manifest. No recovery execution, stream event, scientific read, teacher
  query, label, game, optimizer step, checkpoint, PPO, development,
  confirmation, human-session read, incumbent/dashboard change, or promotion
  is authorized.

## 2026-08-15 Handoff Cleanup

- The user requested a reviewed cleanup before handing the program to a new
  training agent. The apply manifest is retained at
  `runs/forensics/storage_cleanup_20260815/preflight_and_deletion_manifest.json`
  with a CSV companion.
- Removed exactly `6,668,898,048` bytes across 798 `.npy` learned-table files
  from the killed original R1 and permanently unpromoted R1b runs. Both were
  outside the active incumbent and protected replay set. Their configs,
  metrics, summaries, audits, metadata, top-game replays, and confirmation
  evidence remain.
- Preserved every pre-existing forensic artifact, all four incumbent model
  components, protected top-three replays, source replays, human provenance,
  and evaluation manifests.
- Removed only disposable caches/scratch files and an explicitly approved
  recovery-only Git ref in addition to those model tables. No user process or
  retained service was stopped.
- Free disk increased from `109.94 GiB` to `124.95 GiB`. This cleanup does not
  alter or reopen any scientific HOLD/KILL decision.

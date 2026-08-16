# Threes RL Current Decision Ledger

Last updated: 2026-07-26

This file is the short-form guardrail for active research decisions. The
experiment log keeps the full history; this ledger records the current branch
status so old ideas do not get revived after context shifts.

## Active Branch: Exact Depth-3 Program KILL

- C1 remains permanently `FAIL_STOP_C1` and byte-locked. C2 remains
  permanently `KILL_C2_COST_ADMISSION`; its untouched 48-state runtime gate
  remains unopened. K1 is a separate native C11 exact leaf/transition kernel,
  not a reinterpretation or modification of either branch.
- K1 froze the native/wrapper/final-runner/final-test sources at SHA-256
  `e47af279...b41e4f`, `fe2ee430...c5af8`, `e4b360f4...29615c`, and
  `ae180434...ad71a`. Focused tests passed `26/26`; broader
  simulator/search/provenance regressions passed `307` with two known
  artifact-availability exclusions.
- Amendments A4/A5 preserved two failed zero-work preflight attempts,
  permanently spent the 69B-72B reservations, moved K1 to 73B-76B, and
  corrected only the hash-bound self-declaration classification. No external
  collision source was excluded.
- The final zero-work preflight sealed `READY_K1_ENGINEERING_EXECUTION`.
  Preflight lock file/payload SHA-256 are `257ac50a...8a6a` /
  `8fde35a1...ae3`; marker file/payload SHA-256 are `889aa515...c7e0` /
  `0e87dd45...db19`. They bound 108 fresh games, three genuine families,
  9,038 external immutable collision sources, and a frozen 36-root/144-state
  corpus plan.
- The one-shot run completed all `108/108` fresh normal-start games under
  unique streams and ancestries. It retained `4/11/9` trigger-qualified roots
  and `16/44/36` selected states for corner2, parent MC1000, and replaycal.
  The frozen corpus required exactly `12` roots and `48` states per family.
- K1 therefore sealed `HOLD_K1_ENGINEERING_FAULT` at corpus construction,
  before creating the corpus manifest or opening any fresh exactness/runtime
  gate. Terminal file/payload SHA-256 are `157d73f7...e7a6` /
  `3907532f...2287`. This is a fresh-corpus coverage/design hold, not evidence
  that the compiled kernel is inexact or too slow.
- The compiled engineering library and all source replays/state records are
  retained in the 14-MiB sealed directory. No policy outcome, h10/h20/h40
  outcome, score inspection, label, training model, incumbent update, or
  dashboard change occurred. Free disk is about `149.28 GiB`; dashboard
  record/top three remain `263670/261369/258561`.
- The separately frozen support audit then opened only the 24 retained K1
  replays. Charter/runner/test SHA-256 are `a091900a...2e27`,
  `f04b9934...d2da`, and `e08ed580...3836`; `10/10` focused and `122`
  relevant regression tests passed.
- Every observable retained root had at least four qualifying states. Exact
  roots with at least 1/2/3/4 states were therefore `4/4/4/4` for corner2,
  `11/11/11/11` for parent MC1000, and `9/9/9/9` for replaycal. Qualifying
  counts ranged from `5-47`, `13-122`, and `21-131` respectively.
- The remaining `32/25/27` roots have no immutable replay content and stay
  unobservable; they cannot be imputed as one-state roots. The observed
  one-state design remains `4/11/9`, below 12 in every family, with a maximum
  family share of `45.83%`.
- Immutable signature evidence establishes at least five genuine policy
  families, including expectimax2 and QD-v2, but neither alternative has any
  immutable natural root bound to the unchanged K1 trigger predicate.
  Distinctness therefore cannot repair support scarcity.
- The support audit sealed `KILL_EXACT_DEPTH3_PROGRAM`. Its immutable v1 file
  remains at SHA-256 `536fd76d...3111`; its embedded
  `949bd408...fc3f` payload hash is preserved as a known serialization defect
  because integer histogram keys became strings on JSON reload.
- A serialization-only v2 envelope was built from that v1 JSON without
  reopening any replay or recomputing a support statistic. Amendment/runner/
  test SHA-256 are `6e58aeb4...dd72`, `0e171a15...793c`, and
  `62c765a8...e2b4`; `16` applicable tests passed. The authoritative v2
  file/payload SHA-256 are `9b4da740...3e09` /
  `3a27d2c1...b5e5`, and its unchanged scientific-payload SHA-256 is
  `171c0b09...833b`.
- C2 untouched timing and K1 fresh exactness/runtime partitions both remained
  unopened. No K2 proposal or execution is authorized.
- Current states: `CONTINUE=none`; `HOLD` compilation, timing, acquisition,
  policy evaluation, human training ground, and promotion; `KILL` C1, exact
  C2, G3, G4, K1-v1, and the exact depth-3 program permanently;
  `PROMOTE=false`.

## Preserved C2 Cost Admission KILL

- C1 remains permanently `FAIL_STOP_C1`; its exact optimized depth-3 operator
  was used only as the immutable C2 equivalence oracle and was not modified,
  retuned, or rerun as C1.
- C2 froze one nonnegative monotone cost model, one feature dictionary, one
  deterministic admission rule, and one fresh three-family normal-start
  engineering corpus. Charter/runner/test SHA-256 are `d274c680...a7f4`,
  `436f2956...7be1`, and `a36c3219...1b11`.
- The zero-timing preflight and one-shot marker sealed at file/payload
  SHA-256 `1f7dd905...1b4f` / `a8db1b03...56c6` and
  `9b606f88...4817` / `15d70810...85a9`. All `216` games completed and
  yielded exactly `36` fresh disjoint roots and `144` states, partitioned
  before timing as `18/6/12` roots for fit/validation/untouched gate with
  equal representation from all three genuine families.
- The frozen model passed solver, deterministic-refit, finite/nonnegative,
  depth-2 exactness, rank, error, activity, family breadth, and admitted-state
  absolute/relative safety checks. Validation root Spearman was `0.600`;
  mean absolute error was `0.1235`; p90 absolute error was `0.1764`; activity
  was `100%` across all three families.
- C2 failed its conservative coverage gate: the upper bound covered `87.5%`
  of validation roots, below the frozen `90%` requirement. It therefore
  sealed `KILL_C2_COST_ADMISSION` at engineering validation. Terminal
  file/payload SHA-256 are `ac1e3b49...cb9f` / `8acc25e4...a285`.
- The untouched `12`-root/`48`-state runtime gate was never opened. No policy
  evaluation, h10/h20/h40 outcome, label, training model, incumbent update, or
  dashboard change occurred. The dashboard remains `263670` with protected
  top three `263670/261369/258561`.
- Current states: `CONTINUE=none`; `HOLD` policy evaluation, any C2 successor,
  human training ground, and confirmation; `KILL` C1, exact C2 cost admission,
  G3, and G4 permanently; `PROMOTE=false`.

## Preserved G4 V2 Mechanism KILL

- G3 remains permanently `KILL_G3_BOOTSTRAP_PREDICTIVE`. Its ordinary
  marginal-hazard model may not be rerun, extended, reinterpreted, or used as a
  reranker. The 32 transfer roots, transfer predictions, and 226 transfer paths
  remain unopened and forbidden.
- Preserve exact G4-v1 permanently at `KILL_G4_PAIRWISE_INFEASIBLE`.
  Its pre-fit support stop remains valid historical evidence and was not
  rerun, weakened, or reinterpreted.
- G4-v2 used only the exact `727` already-opened ordinary discordant pairs in
  one ancestry-disjoint five-fold cross-fit. The amendment/implementation/test
  SHA-256 are `080d7bf3...bca`, `f9113297...681`, and
  `d51c63b0...f3f`; the immutable marker is `8b9f8e6a...034`.
- The outcome-free support gate passed with `185` informative roots overall,
  `162` pre768, `45` pre1536, and three support-eligible families with
  `154/14/12` roots. All `352` ordinary roots were assigned to folds before
  outcomes. The material 80%-power point floor was frozen at
  `0.6058598392`.
- The one cross-fit sealed `KILL_G4_PAIRWISE_MECHANISM`. Overall
  root-direction concordance was `0.5189`, with 95% whole-root bootstrap
  interval `[0.4514, 0.5838]`, so the lower bound did not exceed chance and
  the point estimate missed the material floor. Continuous root concordance
  was `0.5065`.
- Scale transfer failed directionally: pre768 root direction was `0.5247`,
  while pre1536 was `0.4778`. Only the incumbent lineage exceeded chance among
  the three support-eligible families; corner2 was `0.2500` and legacy learned
  was `0.4583`.
- All five fixed models converged, gradients were below `1e-4`, predictions
  were finite, save/load reproduced exactly, all `727` pairs and `185` roots
  received one OOF prediction, and fold ancestry leakage was zero. Therefore
  this is an adequately supported mechanism KILL, not an integrity or
  underpower HOLD.
- Terminal result file/payload SHA-256 are `0426b562...09c3` /
  `e0ebebe4...0569`; OOF prediction file/payload SHA-256 are
  `d9febb07...3828` / `932c7634...92ea`. The complete compact evidence bundle
  is approximately `500 KiB`.
- G3 transfer access, simulations, new labels, score inspection, policy
  construction/evaluation, incumbent changes, and dashboard changes remained
  exactly zero. Services were healthy, free disk remained `149.406 GiB`, the
  dashboard record stayed `263670`, and protected top three stayed
  `263670/261369/258561`.
- Current states: `CONTINUE=none`; `HOLD` fresh acquisition, labels, all policy
  work, C2, human training ground, and confirmation; `KILL` G3, G4-v1, and the
  G4 conditional-pairwise mechanism permanently; `PROMOTE=false`.

## Preserved G3 E0 V4 / Predictive KILL

- The exact v4 E0 execution sealed `KILL_G3_BOOTSTRAP_PREDICTIVE` at the
  ordinary-development gate. Terminal file/payload SHA-256 are
  `e7ca390f...ddd1` / `a70457df...5bfa`; checkpoint-seal file/payload
  SHA-256 are `a9ac34d3...e58b` / `6d44ca18...9568`.
- All `4,846` authorized ordinary paths completed (`3,902 train`,
  `944 development`). The one frozen 64-feature L2 model converged, calibration
  and numerical-stability checks passed, and the checkpoint was sealed before
  any transfer access.
- Aggregate prediction improved: development log-loss improvement was
  `+0.12308` with 95% root-bootstrap CI `[+0.06546,+0.18771]`; Brier
  improvement was `+0.04425`, CI `[+0.02008,+0.07102]`. Both `pre768` and
  `pre1536` point improvements were positive, and ECE was `0.04292`.
- The action-selection gate failed decisively. Legal-action rank correlation
  was `-0.11302` overall, `-0.08847` at `pre768`, and `-0.08070` at
  `pre1536`; its 95% root-bootstrap CI was `[-0.32366,+0.10105]`.
  The `legacy_learned_lineage` family also regressed on both log loss
  (`-0.06039`) and Brier (`-0.02318`) across five roots, failing frozen family
  robustness. Positive marginal hazard prediction therefore did not transport
  into useful action ordering.
- The outcome-free activity step, all `226` transfer paths, and all transfer
  outcomes remained unopened. There is no transfer database or prediction
  seal. Weak N=32 evidence played no role in the KILL.
- Marker file/payload SHA-256 are `91b38116...a76b` /
  `72ebce43...3331`; ordinary database SHA-256 is `d0954a91...820`.
  Runtime from marker to terminal was `14,157 s` (`3:55:57`); the complete
  directory is `21,104,381` bytes and free disk remains `149.554 GiB`.
- E0 is non-promotable and this exact G3 bootstrap candidate is killed. E1,
  transfer labeling, reranker construction, normal-start evaluation, C2,
  human training ground, incumbent/dashboard changes, and `PROMOTE` remain
  HOLD pending course review. Dashboard record/top three remain
  `263670/261369/258561`.

### Preserved V4 Preflight And Execution Identity

- The narrow collision-contract correction sealed
  `READY_G3_E0_V4_EXECUTION`, terminal status
  `HOLD_G3_AFTER_E0_V4_PREFLIGHT_SEAL`. This historical zero-outcome
  readiness seal was later opened and executed exactly once as recorded above.
- V4 classifies every collision-bearing history source explicitly:
  `8,926` byte-stable immutable external sources, exactly `2` live generated
  dashboard summaries, and `17` inherited internal reservation sources at
  preflight. The immutable inventory SHA-256 is `e6a43f52...aca20`;
  requested stream-contract SHA-256 is `fbbfd9fc...fc7f`.
- Both exact live paths remain in the collision union but are not byte-locked:
  `dashboard.json` and `score_trends.json`. Every requested
  57B/58B/59B/60B collision set is empty. Any immutable mutation, new
  collision-bearing source, unclassified alias/symlink, missing live path, or
  actual collision fails closed.
- Amendment/runner/preflight/test SHA-256 are
  `cbae9910...a628`, `4c564023...4ee0`, `bca7c32a...aba`, and
  `04f67861...27a7`. Py-compile, `17/17` focused, `66/66` applicable G3,
  and `113/113` frozen regression checks pass.
- V4 preflight lock file/payload SHA-256 are `fdc801f5...a9e7` /
  `8937c359...8c12`; collision-manifest file/payload SHA-256 are
  `c62b2d04...28d1` / `f99b6448...20f`. Record/task/stream manifests remain
  byte-identical at `90a4f55f...9d5`, `087fd68c...bd2f`, and
  `e40b7dd3...a05`.
- Seal-time free disk was `149.706 GiB`; one nice-10 process, no contender,
  ports/advisor, dashboard record `263670`, and protected top three
  `263670/261369/258561` all passed.
- At the v4 preflight seal, no execution marker, consumed stream, path, label,
  database, model, prediction, transfer outcome, policy outcome, score
  inspection, incumbent change, or dashboard change existed. Its later
  separately authorized terminal artifacts are recorded above.

### Preserved V3 Live-Source Hash Hold

- The separately authorized exact v3 `open` command failed closed before
  marker creation as
  `HOLD_G3_E0_V3_OPEN_REVALIDATION_DYNAMIC_SOURCE_HASH`. The scientific
  `execute` command was not launched.
- Every substantive open gate passed: bound files, manifests, incumbent,
  services/process/disk/nice, source count `8,927`, and all requested stream
  collision sets. Only the full matched-source SHA changed from
  `e06d563a...dad3` to `48c40c8c...70a1`.
- Root cause is deterministic lifecycle coupling: required live dashboard
  files `dashboard.json` and `score_trends.json` were rewritten after preflight
  and participate in the seed-bearing historical source-list hash. Keeping the
  dashboard healthy therefore invalidates the byte-level source-union lock
  despite unchanged source count and zero requested-stream collisions.
- Immutable HOLD file/payload SHA-256 are `f1f69132...378d` /
  `a2d06ce9...3e6d`. No execution marker, stream, label database/path/value,
  model, checkpoint, prediction, transfer outcome, or scientific terminal
  result exists. A fresh narrow orchestration/preflight correction is required;
  v3 may not be retried or weakened in place.

- V2 execution is explicitly
  `HOLD_G3_E0_EXECUTION_ORCHESTRATION_MISMATCH`. Its READY preflight remains
  valid zero-work evidence, but its runner could not both pre-seal/log a marker
  and resume under that same marker. No v2 marker, stream, database, label,
  model, prediction, or outcome was opened.
- Froze the narrow v3 open/resume amendment at SHA-256
  `d201e4a1...38ab`. It changes orchestration only: a dedicated `open` command
  validates, seals, and exits; `execute` requires that exact marker and supports
  marker-bound database/checkpoint/prediction resume. Runtime monkeypatching is
  absent. The scientific runner remains frozen at `19d74a31...8b5`.
- V3 orchestration runner/preflight/test SHA-256 are
  `db512549...8392`, `4672e769...3890`, and `969acc53...f802`.
  Test evidence SHA-256 `8c7c6de8...18d5` binds py-compile, `11/11` focused
  checks, `49/49` applicable G3 checks, and `113/113` frozen regressions. One
  preserved v2 pre-seal freshness assertion was correctly deselected because
  the immutable READY v2 directory now exists; v3 directly verifies it remains
  zero-work.
- The single no-outcome v3 preflight sealed
  `READY_G3_E0_V3_EXECUTION`, terminal status
  `HOLD_G3_AFTER_E0_V3_PREFLIGHT_SEAL`. Lock file/payload SHA-256 are
  `ac4f4e74...57b3` / `25f3686d...5af7`.
- Record, task, and stream manifests are byte-identical to v2 at
  `90a4f55f...9d5`, `087fd68c...bd2f`, and `e40b7dd3...a05`.
  All `683` ordinary records from `352` source replays and all `32` transfer
  records revalidated, with `384` unique roots and zero source/provenance/state
  failures.
- E0 remains exactly `5,072` all-legal-action h40 paths:
  `3,902 train`, `944 development`, and `226 transfer diagnostic`, using
  replicates `0-1`. Reserved `57B/58B/59B/60B` streams remain unconsumed and
  collision-free against `8,927` non-excluded historical sources, external
  source-list SHA-256 `e06d563a...dad3`.
- The unchanged projection is `11.895h` on one nice-10 worker and
  `46,940,160` bytes. Seal-time free disk was `149.876 GiB`; no competing
  heavy process, ports/advisor, dashboard record `263670`, and protected top
  three `263670/261369/258561` all passed.
- The v3 preflight READY remains preserved readiness evidence, but its exact
  open lock is now spent by the fail-closed attempt. E0 labels/fits, E1,
  transfer outcomes, reranker work, normal-start evaluation, C2, human
  training-ground work, and `PROMOTE` remain HOLD. Exact forbidden work remains
  zero.

### Preserved E0 V2 Ready / Orchestration Hold

- Preserve `runs/forensics/g3_e0_label_fit_v2` byte-for-byte. Its preflight
  file/payload SHA-256 remain `fde44d08...2ef7` /
  `c7ec0d0d...e2bb`; its scientific manifests are the exact v3 inputs above.
- V2 is not a scientific failure and may not be executed, edited, or
  reinterpreted. Its localized orchestration mismatch is corrected only in the
  separately hashed v3 branch.

### Preserved E0 V1 Compact-Manifest Integrity Kill

- The first E0 preflight is permanently
  `KILL_G3_E0_PREFLIGHT_INTEGRITY`, file/payload SHA-256
  `73eee861...fa94` / `af4dcda1...e45c`. Its complete directory remains
  byte-for-byte preserved and must never be executed or overwritten.
- Its sole failure was an outcome-free schema-adapter false positive: the
  older validator expected an embedded `state`, while the authoritative G3
  compact records intentionally retain only exact source replay/frame/hash
  pointers. It therefore emitted `KeyError: 'state'` for all ordinary rows.
  The same sealed run independently reproduced all `683` feature digests, and
  every non-adapter gate passed.
- V1 generated zero labels, models, predictions, or outcomes. The v2
  correction changed only source-pointer validation and output identity; no
  corpus, stream, model, threshold, cost, or scientific gate changed.

## Preserved G3 Scale-Transfer Bootstrap V2 / READY

- The separately versioned integrity correction sealed
  `READY_G3_V2_BOOTSTRAP_LABELS`, with terminal status
  `HOLD_G3_AFTER_V2_BOOTSTRAP_PREFLIGHT_SEAL`. This is label-readiness only:
  E0, E1, fitting, transfer outcomes, reranker work, normal-start evaluation,
  C2, and `PROMOTE` remain HOLD.
- The narrow v2 integrity amendment SHA-256 is `c60895f9...6aef`.
  Implementation/test/test-evidence SHA-256 are
  `b8488f2e...e526`, `9e81e2be...50a3`, and `4ec431b3...0d8`.
  Py-compile, `17/17` focused tests, and all `113/113` frozen regressions
  passed.
- The immutable v2 preflight file/payload SHA-256 are
  `052985f7...7d66` / `18539079...93ab`. Corrected untouchedness file/payload
  SHA-256 are `8b2fd76e...51ed` / `f9e7079f...ba7e`; panel-input-binding
  file/payload SHA-256 are `374fe010...2ca3` / `c6a19325...688b`.
- The corrected live scan reports exactly zero true external root/state-token
  matches. It separately reports `34` matched files inside the exact sealed G2
  input namespace and `2` exact hash-bound v1 manifest files. No broad
  `forensics` exclusion is possible. All `65` panel inputs
  (result plus 32 replay/state pairs) reproduce exactly, binding
  `9,010,761` bytes with zero failures.
- The exact v1 record/stream manifests remain unchanged and are reused only by
  file and canonical-payload hash. Coverage remains `0/20,288` compatible
  h40 paths; no label value was read. The reused `57B/58B/59B/60B` reservation
  is still unconsumed and collision-free against `8,927` historical sources,
  source-list SHA-256 `6c434d89...5ea5`.
- Nonadaptive cost governance is now explicit. E0 is all legal arms at
  replicates `0-1`: `5,072` paths, `11.895h`, `46,940,160` bytes. E1 is all
  same arms at replicates `2-7`: `15,216` paths, `35.684h`, `77,905,920`
  bytes. They sum exactly to the unchanged eight-replicate projection of
  `47.579h` and `124,846,080` bytes. Neither stage is authorized; any future
  E0 needs a separately frozen execution charter and any E0 fit is
  non-promotable.
- N=32 sensitivity remains unchanged at an 80%-power grid MDE of OR `4.0`.
  Free disk was `150.382 GiB`; process, service, dashboard record `263670`,
  and protected top three `263670/261369/258561` all passed.
- Zero games, consumed streams, labels, label values, fitted models, transfer
  outcomes, candidate actions, rerankers, policy outcomes, continuations,
  score inspection, incumbent changes, or dashboard changes occurred.

### Preserved G3 V1 Integrity Kill

- Research-lead decision `KILL_G2_ROUTINE_REACQUISITION` is now explicit.
  The spent G2 acquisition remains exactly `640/640/640` games with
  `12/1/19` clean transfer roots. Do not rerun, extend, substitute collectors,
  reallocate quotas, or interpret acquisition yield as solver quality.
- G3's outcome-free scientific charter was frozen before any label-value or
  model-outcome inspection. Charter SHA-256 is `e216aa50...45fc1`.
  A separately preserved A1 amendment, SHA-256 `baba7200...2b94`, corrects one
  pre-outcome stream-formula contradiction so all legal actions within a
  root/replicate share the same CRN tapes. No other contract changed.
- The G3 preflight implementation/test SHA-256 are
  `27ac9cc6...5234` / `d9c60a84...a14a`; test-evidence SHA-256 is
  `a8f6a176...bd21`. `15/15` focused tests and `113/113` established
  G2/G1/S3/provenance regressions passed.
- One no-outcome preflight sealed
  `KILL_G3_PREFLIGHT_INTEGRITY`, with terminal status
  `HOLD_G3_AFTER_BOOTSTRAP_PREFLIGHT_SEAL`. Preflight file SHA-256 is
  `0cd19d5a...d77a`; canonical payload SHA-256 is
  `4513530c...06ad`. Record/stream manifest file SHA-256 are
  `938e903f...b9ec` / `bdbe5621...5744`.
- The sole failed check is an implementation false positive in
  `transfer_panel_untouched`: the ripgrep exclusion glob did not exclude the
  sealed G2 source directory. Every reported match is inside
  `runs/forensics/g2_fresh_transfer_acquisition_v1` itself
  (`G2_TRANSFER_ACQUISITION_RESULT.json`, `completion_rows.jsonl`, or one of
  the exact retained state files). All `32` transfer sources and all `683`
  ordinary selected records independently revalidated with zero source,
  state, feature, or provenance failures. There is no external match in the
  sealed list, but the terminal KILL remains authoritative and may not be
  repaired by rerunning this one-shot v1.
- Outcome-free label coverage is otherwise fully specified: `20,288` required
  h40 CRN paths, with `15,608` train, `3,776` development, and `904`
  transfer-diagnostic paths. Exactly `0` legacy paths have the required G2
  root/state/stream/continuation sidecar, so every path is missing.
- The conservative one-worker projection is `47.579 h` and `0.1163 GiB`.
  Disk was `150.319 GiB`; all incumbent payload, stream-collision, process,
  service, dashboard `263670`, and protected-top-three checks passed.
- The N=32 transfer panel has only `21.62%` prospective full-gate power for
  policy OR `1.75`; the frozen-grid 80%-power MDE is OR `4.0`. Any later null
  would therefore be a HOLD below sensitivity, never a utility failure.
- Zero games, consumed streams, labels, label values, fitted models, transfer
  outcomes, candidate actions, rerankers, policy outcomes, continuations,
  score inspection, incumbent changes, or dashboard changes occurred.
- V1 remains permanently killed and spent. Its correction exists only in the
  separately hashed v2 evidence above; do not edit or rerun v1.

### Preserved G2 Fresh Transfer Acquisition

- The exact marker-bound run completed all `1,920` scheduled fresh normal-start
  games: `640` each for corner2, expectimax2, and the current phaseblend
  incumbent. The terminal result is `HOLD_G2_FRESH_TRANSFER_ACQUISITION`;
  explicit terminal status is
  `HOLD_G2_AFTER_FRESH_TRANSFER_ACQUISITION_SEAL`.
- Retained natural `pre3072_transfer` roots were only `12/1/19`, versus frozen
  independent quotas `32/32/32`. No quota was reallocated and every family
  reached its exact cap. This confirms that routine acquisition under these
  three policies cannot fill the clean transfer design.
- Immutable open-marker file/payload SHA-256 are
  `0c54dddf...03d7` / `6cc5c76d...c673`; terminal-result file/payload SHA-256
  are `7b862377...ca74` / `a464287e...cee4`.
- All `32` retained roots are unique, their replay/state artifacts reconstruct
  exactly with zero failures, and protected-root overlap is zero. Retained
  source-manifest SHA-256 is `e689accb...aba9`. The sealed aggregate
  `retained_source_integrity=false` is caused solely by its bundled
  `retained_count_96` quota check; the independent `sources_exact`,
  `roots_unique`, and `prior_root_seed_overlap_zero` checks all pass.
- Stream manifest `8c5aefd3...c047` stayed collision-free against historical
  union `eead7a4e...c756`. Completion/runtime/runner-summary SHA-256 are
  `f97bb0ef...6859`, `d9807b22...a98b`, and `29f973b9...4465`.
- Active runtime was `17,655.13 s` (`4.90h`) in `320` chunks; output was
  `14,279,693` bytes, free disk `150.336 GiB`, and all service,
  dashboard/top-three, process, runtime, storage, and hard-disk checks passed.
- No per-game score or chosen action was inspected, filtered, summarized, or
  reported. Zero labels, models, h10/h20/h40 outcomes, policy outcomes,
  continuations, incumbent changes, or dashboard changes occurred.
- Acquisition execution is closed. Labels/fitting, policy evaluation, C2, and
  `PROMOTE` remain HOLD pending research-lead review.

### Preflight Record

- The authoritative scientific status remains `HOLD_G2_DATA_OR_POWER`: the
  scale-equivariant representation and prospective power passed, but no
  untouched `1536->3072` transfer roots existed.
- A separate fresh-source acquisition design is now sealed
  `READY_G2_FRESH_TRANSFER_ACQUISITION`. It targets any first natural
  `pre3072_transfer` state in completed fresh normal-start machine games; it
  does not revive killed G1-R exact-rung continuation and is not a policy
  comparison.
- Frozen charter/runner/test SHA-256 are `bebfbbce...526d`,
  `66ce0dea...7c23`, and `597a5a6e...6c40`. `20/20` focused tests and `85/85`
  established G2/G1/provenance regressions pass.
- The exact ordered collectors are corner2, hand-built expectimax2, and the
  current depth-2 phaseblend incumbent. Immutable signatures reproduced at
  `4be42141...7043`, `2ad642cd...4b38`, and `868a6337...1ccb`; all three
  pairwise rates and both stratum rates match the sealed pilot-v1 evidence.
  Parent/student remain collapsed into the incumbent alias component.
- The immutable no-game lock is
  `runs/forensics/g2_fresh_transfer_acquisition_v1/preflight_lock.json`, file
  SHA-256 `5250e54d...cf44`, canonical payload SHA-256
  `18d9e851...5993`. Policy-lock and signature-audit SHA-256 are
  `74c08c0c...f812` and `7217161f...31c4`.
- It freezes independent quotas `32/32/32`, caps `640` games per family, one
  worker, nice `>=10`, deterministic round-robin chunks of at most six, and
  fresh `53B/54B/55B/56B` stream bases. The `1,920`-row manifest SHA-256 is
  `8c5aefd3...c047`; it has zero collisions across `8,859` protected sources,
  whose embedded source-list SHA-256 is `eead7a4e...c756`.
- Conservative worst-case projection is `0.25768 GiB` and `10,811.97` active
  seconds, both below frozen `4 GiB`/`12h` limits. Free disk was
  `152.242 GiB`; no heavy contention and all service/dashboard/top-three
  checks passed.
- A first staging attempt failed before policy actions because its reader
  expected a nonexistent panel field. Preserve
  `g2_fresh_transfer_acquisition_v1.staging.11437/PREFLIGHT_FAILURE.json`,
  SHA-256 `716fa96e...385e`, as engineering provenance. The repaired focused
  test binds the actual immutable panel payload hash; no game or stream was
  opened in either attempt.
- READY authorizes only a later separately approved acquisition execution.
  Current status is HOLD execution. Zero games, streams, labels, rollouts,
  h10/h20/h40 outcomes, fitted models, score/policy outcome inspection,
  continuations, incumbent changes, or dashboard changes occurred.

### G2 Scientific Hold

- Research-lead decision: `KILL_G1R_EXACT_RUNG_CONTINUATION`. The sealed QD5
  pilot remains `HOLD_G1R_AFTER_PILOT_V2_QD5_SEAL`; do not buy its projected
  `27/432` pre3072 shortfall with routine acquisition. This kills only the
  exact two-rung acquisition continuation, not the solver or relational-hazard
  research.
- The authoritative G2 proposal was frozen before corpus statistics at
  SHA-256 `43b413c1...7099`. It defines one 64-column, orientation- and
  target-scale-equivariant relational afterstate representation and one fixed
  regularized discrete-time logistic hazard model. Earlier transitions
  `384->768` and `768->1536` are training scales; `1536->3072` is untouched
  transfer only.
- Implementation/schema/test SHA-256 are `9ffaa45d...af8a`,
  `6af0cd51...340e`, and `3eca4551...9911`. `13/13` focused tests and `76/76`
  relevant geometry/provenance/G1/S3 regressions passed.
- Immutable outcome-free preflight:
  `runs/forensics/g2_scale_equivariant_relational_hazard/G2_PREFLIGHT.json`,
  file SHA-256 `2e1084f2...05cc`, canonical payload SHA-256
  `4d6fef61...55c9`. Root manifest file/payload SHA-256 are
  `60d514ed...a2ca` / `15ecb9d5...31ce`.
- Representation and integrity passed: exact finite bounded feature schema,
  crafted scale/orientation invariance, no state or RNG mutation, `1,557`
  source hashes, zero provenance/restore failures, and zero cross-partition
  root overlap.
- Natural availability is broad at earlier scales but unavailable for clean
  upward transfer: `485` roots across five families, with `485` pre768, `463`
  pre1536, and `98` pre3072 roots. Every pre3072 root overlaps protected prior
  evidence, leaving `0` untouched transfer roots. Proposed ordinary partitions
  contain `283` train and `69` development roots; transfer contains `0`.
- The frozen prospective calculation is sufficiently powered in principle:
  minimum viable `N=96` gives `92.67%` target-event power for OR `1.75`.
  Therefore the authoritative decision is `HOLD_G2_DATA_OR_POWER` because of
  uncontaminated transfer-data scarcity, not representation or prospective
  power failure.
- No G2 game, stream, action label, rollout, h10/h20/h40 outcome, model fit,
  candidate action, score inspection, continuation, incumbent change, or
  dashboard point was created. Labels and fitting remain held. C2 and S3
  outcomes remain held; dashboard record remains `263,670`.

### Preserved Prior Branch State

- S3 is closed as `HOLD_UNDERPOWERED_PREFLIGHT`, with zero S3 policy outcomes.
  The authoritative artifact is
  `runs/forensics/s3_full_policy/S3_POWER_PREFLIGHT_V2_SEALED.json`.
- S3 found `133` eligible pre-1536 roots but only `9` pre-3072 roots. The
  largest family supplied `114/133` and `7/9`, respectively. The best frozen
  design (`N=192, R=8`) had `48.22%` power for common OR `1.50`; its 80%-power
  MDE was OR `1.84375`.
- Root count, family concentration, and power failed independently. Compact
  storage and disk headroom passed. Do not weaken exclusions, reuse roots, run
  S3 outcomes, or reinterpret the hold as a utility failure.
- C2 remains `HOLD`: exact depth-3 utility is still unknown.
- The retained-corpus G1 V5 audit found zero eligible roots after all historical
  exclusions, so G1-R opened as fresh normal-start acquisition only.
- G1-R pilot-v1 stopped outcome-free as `HOLD_G1R_FAMILY_SCARCITY`. Its fixed
  `32+32` action panel collapses six nominal specs into four genuine families:
  corner2, hand-built expectimax2, replay-cal, and one
  parent/student1/incumbent alias component. The last component had zero
  pre3072 disagreements pairwise and cannot be counted three times.
- Immutable preflight:
  `runs/forensics/g1r_acquisition/pilot_v1/preflight_lock_pilot_v1.json`,
  file SHA-256 `f78288b3...81ea91`, payload SHA-256
  `b21e4c41...09e44`. All other checks passed, including zero collisions
  across `8,825` matched historical sources and `152.806 GiB` free.
- Zero G1-R games, labels, models, or policy/score outcomes were created. Do
  not run the pilot, weaken the `2%`/per-stratum alias rule, rename policies,
  or treat checkpoint hashes as behavioral families.
- `G1R_QUALITY_DIVERSITY_FAMILY_PROPOSAL.md` initial
  SHA-256 `63032ef7...1a513` is retained as superseded. The amended proposal is
  SHA-256 `e9a72c65...bf880` and freezes exact descriptor/tie/missing rules,
  mixed categorical/ordinal distance, archive counts, 12,000-game Wilson yield
  projection, and absolute action latency.
- QD implementation and focused tests passed review at SHA-256
  `cb6dd06f...77b3` and `e43c9252...cb0c`. One preparation-only invocation
  atomically promoted
  `runs/forensics/g1r_qd_admission_v1/execution_lock.json`, file SHA-256
  `ed0a89e7...b6e2`, payload SHA-256 `36858d1a...962f`.
- The frozen archive has `489` validated fresh-root ancestries from `489`
  distinct source replays in `489` descriptor cells. Reserved streams are
  unused and collision-free over `8,827` matched historical sources; the
  process, service, dashboard/top-three, nice, and `152.761 GiB` disk checks
  passed.
- The one authorized `run-admission` invocation wrote immutable
  `ADMISSION_OPENED.json`, then terminated during
  `reference_action_signatures` with `Descriptor requires a live state,
  legal=0`. Terminal decision: `HOLD_QD_ADMISSION_ERROR`.
- Marker file SHA-256 is `f1faadcf...1f6e`, marker payload SHA-256 is
  `0d40fa89...1ce0`; terminal HOLD file SHA-256 is
  `205229ce...068b`, payload SHA-256 is `6bc74c73...cba0`.
- The assay is spent and must never be rerun under this lock. It produced no
  complete candidate/reference signatures, pairwise rates, timing samples,
  exactness/determinism decision, thermal-after record, or postflight service
  record. This is an engineering HOLD, not an alias/cost/scientific result.
- Authoritative terminal decision: `KILL_QD_V1_EXECUTION`. Preserve the entire
  promoted v1 directory byte-for-byte. Do not recover partial actions, modify
  the descriptor or archive in place, remove the marker, or open acquisition.
- Root cause is a frozen schema omission: exact visible-tile insertion can
  produce a terminal afterstate with zero legal actions, while v1 admitted only
  legal counts `1..4` and hard-failed on `0`.
- `G1R_QD_V2_TERMINAL_SCHEMA_PROPOSAL.md` is a separate proposal-only revision.
  It may change only the legal-count domain to `0..4`, its mixed-distance
  denominator to `4`, and all version/output identifiers and hashes. No v2
  implementation, lock, panel action, timing, or acquisition is authorized.
  Authoritative proposal SHA-256: `9a95f0b9...89cf`.
- The separate v2 runner, focused test surface, and execution charter are now
  implemented at SHA-256 `191c612d...c51b`, `2e078039...da00`, and
  `d032c35c...81a0`. Descriptor-schema SHA-256 is `a8cd1e15...a2c2`.
- `py_compile`, `25/25` focused tests, and `51/51` broader G1/S3 regressions
  pass. Descriptor-only enumeration ran twice over every exact insertion
  outcome for all `64` panel states and all `489` deterministic selected roots
  without changing state or deck/slot RNG streams.
- Panel legal-count coverage was `0:10, 1:0, 2:47, 3:165, 4:495`; archive-root
  coverage was `0:83, 1:0, 2:428, 3:1360, 4:3898`. Crafted tests cover every
  count `0..4`, including count `1`, and the exhaustive natural sets contain
  both terminal and live outcomes.
- Decision: implementation/test surface PASS only. The v2 output directory is
  now replaced by one authorized preparation-only PASS.
- Exactly one fresh staging preparation atomically promoted
  `runs/forensics/g1r_qd_admission_v2_terminal_schema/execution_lock.json`.
  File SHA-256 is `1f48822f...ff4a`; canonical payload SHA-256 is
  `f6f6287b...af74`.
- The v2 archive has `489` validated canonical fresh roots from `489` distinct
  replays, `489` unique state hashes, and `489` occupied cells. Source-manifest,
  archive, and policy metadata hashes are fresh v2 identities. The unchanged
  cell-table content correctly retains SHA-256 `7eeb9eb2...4ecf`.
- Collision audit passed over `8,829` historical sources with no use or
  collision in reserved `45B/46B/47B/48B` ranges. Nice was `19`, no competing
  heavy process existed, free disk was `152.735 GiB`, and ports/advisor,
  dashboard `263,670`, and protected top three passed.
- Decision: `READY_QD_ADMISSION_LOCK` only. No `ADMISSION_OPENED`, real
  candidate/reference action, pairwise signature, timing, stream consumption,
  game, label, model, continuation, score outcome, incumbent change, or
  dashboard change existed before the one authorized action admission.
- The one-shot v2 action admission is now sealed `READY_QD_FAMILY_ADMISSION`.
  Marker file SHA-256 is `11b21137...5135`, marker payload SHA-256
  `aa7e517a...86e2`; result file SHA-256 is `27bcb332...6a8e`, result payload
  SHA-256 `0eced74a...38e2`.
- Candidate disagreement passed against corner2 (`64.06%`), expectimax2
  (`56.25%`), parent MC1000 (`56.25%`), and replay-cal (`54.69%`), with
  nonzero disagreement in both frozen strata. Candidate signature
  `66da7d61...d281` is distinct from all four reference components and therefore
  adds one genuine action family on this panel.
- Candidate timing passed all gates: median `5.792 ms`, p90 `14.863 ms`, p99
  `23.857 ms`, max `24.251 ms`; relative median/p90 were `0.0207x/0.0175x`
  the incumbent. All `320` repeats were deterministic; order was balanced
  `160/160`.
- Exactness, state nonmutation, spawn-probability, reference-signature,
  save/reload, thermal, service, dashboard, and free-disk checks passed.
  The sealed result lacks a separate numeric conditional-pilot storage
  projection field; actual admission storage was about `4.40 MiB`, and the
  preregistered pilot cap remains `4 GiB`. Do not backfill the result.
- Oversight classified that omission as `HOLD_QD_STORAGE_REPORTING_DEFECT` and
  authorized one metadata-only supplemental audit. The frozen charter is
  `G1R_QD_V2_STORAGE_AUDIT_CHARTER.md`, SHA-256 `dd51e274...03070`; the action
  marker/result remain immutable.
- The supplemental replay inventory includes every qualifying pre-marker
  regular file named `replay.json` under `runs`, without parsing replay JSON:
  `3,242` files. Inventory file SHA-256 is `0dd9e2d4...e552`; canonical payload
  SHA-256 is `d5ebf1ba...aa03`. The maximum file was `1,000,401` bytes.
- Supplemental decision: `READY_QD_STORAGE_ADMISSION`. Baseline admission
  storage `B` was `4,598,101` logical bytes; the frozen first-120-game
  projection is `313,094,177` bytes (`0.29159 GiB`), leaving
  `3,981,873,119` bytes below the `4 GiB` cap. Free disk was `152.714 GiB`.
  Audit file SHA-256 is `0bdef1de...037f`; canonical payload SHA-256 is
  `2864146d...bd78`.
- All storage, sealed-input, cutoff, service, dashboard-record, and protected
  top-three checks passed. The v2 marker/result and v1 marker/HOLD hashes
  remained unchanged. The supplemental audit generated zero actions, timing
  assays, games, streams, labels, models, continuations, score/policy outcome
  inspection, incumbent changes, or dashboard changes.
- Decision: QD-v2 is admitted as a fifth behavior family only. Acquisition,
  games, labels, outcomes, incumbent/dashboard changes, G1-R, and `PROMOTE`
  remain held pending oversight even though the supplemental storage gate
  passed.
- A separate five-family pilot-v2 acquisition surface is now implemented
  without modifying `g1r_acquire.py`, its tests, or pilot-v1. The authoritative
  charter SHA-256 is `1f58d73b...e003`; pre-amendment charter
  `06ae8fa2...aebf` is retained as superseded. The amendment requires one
  global cross-stratum state per ancestry and
  `k1536 + k3072 = kany` per family in the yield projection.
- Runner/test SHA-256 are `f1950260...d776` and `85be02eb...a8de`.
  `19/19` focused tests and `93/93` phase-applicable G1/S3/QD regressions pass.
  Two immutable QD-v2 pre-lock absence guards are explicitly deselected because
  the authoritative QD admission directory now exists; neither sealed test is
  modified.
- Immutable no-game preflight decision:
  `READY_G1R_PILOT_V2_QD5_PREFLIGHT`. Lock file SHA-256 is
  `0d50edaa...22ad`; canonical payload SHA-256 is `1a0ca85b...7e67`.
- The lock freezes exactly corner2, expectimax2, parent MC1000, replay-cal, and
  QD-v2 in that order. All five accepted action signatures, zero tie counts,
  and all ten pairwise overall/pre1536/pre3072 rates reproduced exactly without
  retiming. Signature-audit SHA-256 is `cc747bea...35d`.
- The future pilot manifest has exactly `100` rows (`20` per family), stream
  bases `49B/50B/51B/52B`, manifest SHA-256 `fae883a1...dc68`, and zero
  collisions across `8,831` matched historical sources; union SHA-256 is
  `5df05131...d53f`. No requested stream was consumed.
- Split reset/round-trip, one worker, nice `10`, no competing heavy process,
  `152.714 GiB` free, `0.29159 GiB` conservative projected storage,
  ports/advisor, dashboard `263,670`, and protected top three all passed.
  The promoted directory contains only the immutable preflight lock.
- Decision: pilot execution, games, labels, models, continuations, score/policy
  outcomes, G1 fitting, C2, incumbent/dashboard changes, and `PROMOTE` remain
  held for oversight after the authorized pilot seal.
- The one authorized pilot completed exactly `100` normal-start acquisition
  games, `20` per frozen family. Immutable execution marker file/payload
  SHA-256 are `a2fbd4a3...54d3` / `327517d1...2c2b`; terminal seal
  file/payload SHA-256 are `75a11648...e57` / `b9588d39...8ecb`.
- Terminal status is `HOLD_G1R_AFTER_PILOT_V2_QD5_SEAL`; the runner's
  outcome-free feasibility decision is `HOLD_G1R_YIELD_PROJECTION`.
  No continuation acquisition is authorized.
- The global cross-stratum root cap retained `23` unique ancestries from `24`
  eligible records and reproduced exactly. Per-family
  `k1536 + k3072 = kany` passed:
  corner2 `6+0=6`, expectimax2 `0+0=0`, parent MC1000 `7+1=8`,
  replay-cal `9+0=9`, and QD-v2 `0+0=0`.
- The frozen 90% Wilson projection passes pooled pre1536 (`1,568 >= 432`) and
  any-rung (`1,664 >= 864`) readiness, but projects only `27` pre3072 roots
  versus the required `432`. This is a source-yield HOLD, not policy evidence
  and not a solver failure.
- Root uniqueness, retained-state reconstruction (`23/23`), exact manifest
  completion, stream integrity/collision checks, `12h/4GiB/100GiB` bounds,
  and postflight services/dashboard/top-three checks all passed. Active runtime
  was `563.12 s`; output was `7,807,304` bytes; free disk was `152.679 GiB`.
- Zero labels, fitted models, h40 outcomes, continuations, score or policy
  outcome analysis, incumbent changes, or dashboard changes occurred.
- Incumbent and dashboard remain unchanged at `263,670`. Human actions remain
  deferred mechanism evidence, never labels.

## R1.5a Amendment A2: KILLED OFFLINE

- `R15A_AMENDMENT_A2_20260711.md` is the final pre-outcome data-readiness
  amendment. It preserves the original and A1 hold artifacts unchanged and
  changes only weighted train ESS minimum from `120` to `100`.
- Observed weighted train ESS is `110.18`; all other A1 source, family,
  holdout, context, provenance, and overlap rules passed. Decision: `READY_A2`.
- Compact labels and the exact two-model offline gate completed. Ancestry MAE,
  family robustness, concentration, and synthetic-context gates failed, so the
  exact context residual is permanently killed without policy evaluation.
- One heavy job at a time; keep at least `100 GiB` free and target `120 GiB`.
  The dashboard record remains `263,670`.
- Natural labels completed and passed integrity: `24,576` tasks, `73,728`
  horizon rows, exact 16 replicates for all `1,536` states, 24 deterministic
  replay audits, zero terminal-bootstrap/source/cell failures. These are
  retained failure evidence, not authorization for another fit.

## R2a Adaptive Expectimax: KILLED PRESCREEN

- R1.5a/A2 is permanently killed before policy evaluation. Ancestry-heldout
  MAE worsened under both root and family weighting; family robustness,
  concentration, and synthetic context-contrast gates failed.
- The frozen R2a configuration used depth 2 normally and incumbent-leaf depth 3
  only for built-768/1536 low-empty or <=2% margin decisions, chance limit 8,
  and a deterministic 2,048-node budget.
- Its root-capped heldout prescreen passed activity/diversity but failed runtime
  at median `10.03x` and p90 `16.42x`. No paired continuation was authorized.
- R1.5a models are never policy candidates. C remains spent; dashboard remains
  `263,670`.

## Current Program State: C1 FAILED / COURSE-CHANGE HOLD

- R2a is permanently killed at prescreen. It changed `23/64` heldout actions
  across three families, but runtime was median `10.03x`, p90 `16.42x`, max
  `38.91x`; frozen limits were `3x/5x`. No continuation outcomes were run.
- C1 preserved the exact calculation and reduced profile-split runtime to
  median `2.94x`, p90 `4.42x`. Its one-shot untouched gate passed exactness,
  median (`2.51x`), p90 (`3.89x`), and max (`11.58x`) but failed the frozen p99
  limit: `10.63x` versus `8x`. Decision: `FAIL_STOP_C1`.
- R2b was not opened. Additional cache variants, node-budget tuning, and R2a
  outcomes remain unauthorized. The bounded hotspot is state-dependent exact
  n-tuple leaf work, not chance expansion or budget cutoffs.
- H3 remains held: the latest completed-session inventory has two substantial
  assisted ancestries and needs at least three. Active/partial games remain
  excluded from analysis.
- Do not revive R1.5a, R2a, imitation, UCT/MCTS, broad self-play fitting, or
  search-budget sweeps. Incumbent and dashboard remain unchanged.

## C1 Exact Search Optimization: FAILED RUNTIME GATE

- C1 is engineering-only against the immutable killed-R2a calculation. Frozen
  corpus has 12 profile, 24 equivalence, and 48 untouched runtime-gate roots;
  all are triggered, root-disjoint, R2a-disjoint, and each split has at least
  three families.
- The retained implementation batches grouped n-tuple features, memoizes exact
  board-only leaf results across chance contexts, and uses exact byte board
  keys. All 24 equivalence roots and all 48 gate roots matched values/actions.
- Runtime pass required median <=3x, p90 <=5x, p99 <=8x, max <=12x. Observed:
  `2.51x / 3.89x / 10.63x / 11.58x`; p99 failed, so C1 is closed.
- Evidence: `runs/forensics/c1_search/C1_RUNTIME_GATE.json` and
  `C1_STOP_GO.json`. No score outcome, continuation, normal-start policy block,
  or dashboard point was created. Dashboard remains `263,670`.
- Read-only tail audit did not alter this decision. The failed p99 was primarily
  a denominator tail: optimized absolute times were not outliers, but depth-2
  baselines were exceptionally cheap. Tail-specific counters were not recorded;
  independent profile analogs support distinct-leaf search expansion as the
  secondary mechanism.
- `C2_COST_ADMISSION_PROPOSAL.md` is proposal-only and not active. Ranked next
  options are deterministic cost admission on a fresh corpus, a compiled exact
  leaf kernel, or abandoning exact depth 3. H3 remains held pending a third
  substantial completed assisted ancestry.

## R1.5a Amendment A1 HOLD_DATA_A1

- The authorized A1 amendment is frozen in
  `R15A_AMENDMENT_A1_20260711.md`. The original preregistration, inventory, and
  `HOLD_DATA` lock remain immutable historical evidence.
- A1 removed the hard family root-count deletion and retained `359` train roots
  plus `92` ancestry-holdout roots. Corner2 remains an untouched `39`-root
  whole-family holdout; human remains a `13`-root diagnostic partition.
- All amended readiness rules passed except one: train weighted effective
  ancestry count is `110.18`, below the frozen minimum `120`. Unweighted train
  ESS is `352.82`; all four stages, context marginals, provenance checks, and
  root-overlap checks pass.
- Frozen train effective family weights are phaseblend incumbent `40.0%`,
  legacy learned `34.29%`, expectimax `11.43%`, cheap phaseblend `11.43%`, and
  random `2.86%`. The ESS miss follows from these frozen masses and small
  minority-family root pools.
- Decision: `HOLD_DATA_A1`. Artifact:
  `runs/forensics/r15a_context_a1/R15A_A1_PREFLIGHT_STOP_GO.json`. No labels,
  fitted model, policy evaluation, normal-start block, or dashboard point was
  created.
- Active program state is stopped for a new course-change decision. Do not
  alter the A1 weights or threshold, generate labels, fit models, or launch a
  policy block from this manifest without an explicit pre-outcome amendment.
  C remains spent; R1b remains unpromoted; H0 imitation remains killed.

## R1.5a Original Preflight HOLD_DATA

- Frozen preregistration: `R15A_OFFLINE_PREREGISTRATION.md`. Primary target is
  the h40 simulator-consistent residual correction
  `score_0:40 + V_inc(live_s40) - V_inc(s0)`, with terminal bootstrap zero.
- Read-only inventory scanned `1,555` retained replay files, retained `643`
  deduplicated natural normal-start replays and `165,987` valid states, removed
  `871` replay copies, excluded `41` non-natural artifacts, and found zero
  state/provenance failures.
- Whole-family holdout is all `39` corner2 ancestries. Human is a separate
  `11`-ancestry diagnostic partition. Thirty root-seed clusters shared with
  corner2 were removed from fit/ancestry partitions before selection.
- The preregistered 40% fit-family cap is the blocker. It removed `280` train
  and `83` ancestry-holdout ancestries, leaving train `81` roots (required
  `100`) and ancestry holdout `10` (required `25`). Decision: `HOLD_DATA`.
- Train still has five families, effective ancestry count `77.94`, all four
  stages, all plus/pending/empties marginals, and largest family share `39.5%`.
  Family holdout is strong (`39` roots, ESS `38.78`). The shortfall is genuine
  independent fit/ancestry-holdout diversity, not context or provenance parsing.
- Joint context-cell coverage is sparse: train `64/192`, ancestry holdout
  `36/192`, family holdout `54/192`; ancestry holdout has only one selected
  late-stage state. Do not infer readiness from marginal-bin coverage alone.
- Equal-capacity model engineering passes: board-only and context modes each
  have `3,796` parameters, identical deterministic hidden initialization,
  exact zero residual identity, context masking, save/load, schema rejection,
  and state/nonmutation tests.
- Stop/go artifact:
  `runs/forensics/r15a_context/R15A_PREFLIGHT_STOP_GO.json`. No label stream,
  label corpus, model fit, policy evaluation, or dashboard artifact exists.
- This hold was superseded only by the explicitly frozen A1 amendment. Do not
  otherwise relax the family cap,
  repartition roots, resurrect pseudo-families, or proceed to labels from this
  manifest. R1.5a B-E and R2 are held.

## Human H0/H2 Results / R1.5a Proposal Hold

- R1b failed sealed C and is permanently unpromoted. Never rerun C, inspect it
  incrementally, tune against it, or reinterpret the development result as
  confirmation. The incumbent and dashboard record remain unchanged.
- Completed work is the preregistered human H0 action-conditioned diagnostic in
  `HUMAN_H0_PREREGISTRATION.md`. Human games are development/mechanism data,
  never held-out policy evidence.
- Frozen corpus: ten completed exact-simulator games retained, six substantial
  independent ancestries used for H0, one selected 3072 success and five
  built-768 failures. The empty active session is excluded.
- Frozen roots: eight correlated success-window frames and 40 deterministic
  failure-ancestry geometry references. The references are not exchangeable
  matched controls; their match distance is retained continuously.
- Primary human-minus-incumbent statistics use only the 18 roots where the
  frozen actions differ: four success-window roots and 14 failure references
  distributed `5/1/4/3/1` across all five failure ancestries. Same-action roots
  are uninformative structural zeroes and enter all-action rankings only.
- H0 uses every legal forced first action, 64 replicates per root, and one h40
  path supplying h10/h20/h40 checkpoints. Arms share deck streams and mapped
  slot uniforms; all new logical/deck/slot/policy IDs are disjoint from prior
  development and sealed-C identifiers.
- H0 is complete with `9,600/9,600` tasks and `28,800` horizon rows.
- H0 decision: `KILL DIRECT HUMAN-ACTION SUPERVISION`. On four informative
  success-window roots, recorded actions produced two positive and two negative
  h40 score contrasts, mean score `-2,536.75`, and first-1536 difference
  `-3.52 pp`. Failure-reference clusters were neutral on score (`+36.05`, CI
  `[-1,942.65,+1,969.84]`) and slightly negative on first 1536 (`-0.04 pp`).
- Across all 18 disagreement roots, h40 score was `-748.33`, CI
  `[-2,114.69,+1,543.18]`; first 1536 was `-0.87 pp`; survival was `+0.35 pp`.
  Anchor preservation regressed `-3.56 pp`, ancestry CI
  `[-6.98,-1.18] pp`. Both A/B score and milestone blocks were negative.
- Frame 286 does not rescue the branch. Human right exceeded incumbent up by
  only `10.78` h40 score while all legal actions reached both 1536 and 3072 in
  `64/64` replicates; up had slightly higher survival.
- H1 is not authorized because H0 failed. Human action agreement is not a
  label; no imitation, action prior, deeper-search teacher, or policy fitting
  may follow from H0.
- Active conditional path is H2: one bounded, preregistered preview/cycle
  context-sensitivity diagnostic using human roots as state coverage only.
- H2 is complete: `CONTEXT_MATERIAL`. With board/current preview fixed,
  cycle context flipped `3/48` selected actions but shifted normalized top-two
  margin by at least `1%` in `34/48`. H20 score median absolute effect was
  `3,400.59`, above permutation null95 `2,155.43`, with stable A/B signs in
  `39/46` informative cases.
- High-plus versus zero-plus signed effects expose a tradeoff, not a free
  bonus: mean h20 score `+5,988.02`, first1536 `+5.73 pp`, first3072
  `+4.95 pp`, survival `-8.98 pp`, and anchor preservation `-10.55 pp`.
- Continue only to the proposed R1.5a offline prediction gate in
  `HUMAN_H2_CONTEXT_RESIDUAL_PROPOSAL.md`. No fitting has been authorized.
  Require a separate preregistration/stop-go note before generating a training
  corpus or model. Full policy training, normal-start evaluation, sealed-C
  reuse, and dashboard changes remain held.
- Recorder engineering now supports post-game `good`, `mistakes`, or
  `calibration-discard` metadata without modifying frames, plus catalog-backed
  exact-state human restart continuations. Restart outputs are explicitly
  dashboard-ineligible.

## Governing Restart Decision

The value-first restart proposal now supersedes the earlier phase-0-oracle and
human-data hold as the default branch.

- R0 gate: `PASS` on 2026-07-09.
- Gate artifact: `runs/forensics/restart_program/r0_gate_20260709.json`.
- R1 outcome: `HARM STOP` at the 1,000-episode D0 boundary. The exact
  `n=8`, `alpha=0.001`, 50/50 configuration is killed and must not continue to
  5,000.
- Preregistration: `R1_PREREGISTRATION.md`.
- Frozen restart manifest: `139,149` states; unique ancestries by stage are
  `490`, `485`, `108`, and `105`.
- Frozen split-stream normal-start blocks: D0 `64`, D1 `192`, untouched C
  `512`.
- Incumbent D0+D1 baseline: mean score-minus-starter `21,689.33`, median
  `12,343.5`, lower decile `4,933.5`, P3072 `4.30%`, mean moves `185.10`.
- D1 and C were not inspected for the candidate. C remains sealed.
- The 100-episode correctness smoke passed: exact 50/50 starts, every stage
  touched, later-stage promotions `126,002 / 5,520 / 53,302`, finite payloads,
  exact reload predictions, no mask mutation, and no periodic checkpoints.
  The same run reached 1,000 and stopped at its first policy gate.
- D0 candidate versus incumbent: paired mean score-minus-starter `-7,400.77`,
  95% CI `[-16,356.89, -230.47]`; P3072 difference `-4.69 pp`; lower-decile
  estimate `-2,409.6`. This satisfies both frozen harm rules.
- Bounded attribution audit:
  `runs/forensics/restart_program/r1_pilot_1000_failure_audit_20260709.json`.
  The bare parent was already `-8,448.84` behind the incumbent on D0, while
  the trained candidate was `+1,048.08` versus that parent, CI
  `[-1,798.75, +3,988.72]`. The untrained promotion wrapper was exactly
  parent-equivalent. This changes the diagnosis but not the kill decision.
- Boundary, promotion, and sampler plumbing were not implicated: zero phase
  mismatches across `139,149` restart records, exact wrapper values and depth-2
  actions, exact 500/500 starts, broad ancestry visits, finite payloads, and
  plausible TC scales.
- Leading diagnosis: R1 started from a bare MC1000 leaf that was materially
  weaker than the blended incumbent. TD made a small, uncertain recovery but
  did not close that inherited policy gap or preserve the lower tail.
- Active candidate: R1b, preregistered in `R1B_PREREGISTRATION.md`. It is a new
  frozen-incumbent policy-evaluation candidate, not a continuation of R1.
- R1b total leaf is the exact frozen incumbent plus a zero-initialized promoted
  phase4 residual. Only residual arrays update; all trajectories come from the
  full frozen depth-2 incumbent; n=8, alpha=.001, TC, exact 50/50 starts, and
  the existing ancestry manifest remain fixed.
- R1b must match all 64 incumbent D0 outcomes exactly before episode 1. D0 is
  identity-only. The 100-episode checkpoint is correctness-only; untouched D1
  is the sole 1,000-episode policy gate. C remains sealed.
- R1b pre-update D0 identity: `PASS`, with `0/64` outcome mismatches, zero
  residual entries, zero promotions, and all frozen arrays read-only.
- R1b 100-episode correctness smoke: `PASS`. Exact 50/50 starts, all four
  stages touched, residual promotions `229,196 / 9,092 / 131,584`, unchanged
  incumbent fingerprints, finite payloads, exact reload, and `146 GiB` free.
  The same run is authorized to continue to exactly 1,000 total episodes and
  then evaluate once on untouched D1.
- R1b 1,000-episode D1 pilot: continuation rule `PASS`, promotion `HOLD`.
  Paired mean `+1,425.98`, CI `[-4,051.67, +6,731.73]`; median `+4,093.5`;
  lower decile `+1,289.7`; moves `+7.20`; P3072 tied at `8/192`.
- Changed seeds: `97` candidate wins, `90` losses, `5` ties; P3072 had `8`
  gains and `8` losses. Mean without the largest gain remains `+616.63`.
- No harm fired and the CI contains the preregistered `+1,059.57` useful gain,
  so R1b is not killed. Its score CI crosses zero, so it is not promoted.
- The 5,000/D2 gate in `R1B_NEXT_GATE_PROPOSAL.md` is authorized. D2 is frozen
  at 512 games in `runs/eval_manifests/r1b_split_streams_d2_512_20260709.json`.
  Its deck, slot, policy, and logical IDs have zero collisions with D0, D1, or
  sealed C; audit: `r1b_split_streams_d2_512_disjointness_20260709.json`.
- Cache the incumbent D2 baseline, then continue the exact R1b run to 5,000
  with no intermediate evaluation. Evaluate the candidate once on D2. C and
  all other branches remain held.
- The incumbent D2 cache is complete and hash-locked at
  `runs/eval_artifacts/r1b_baseline_incumbent_split_v1_d2_20260709`:
  mean score-minus-starter `20,567.11`, median `12,312`, P3072 `18/512`, and
  mean moves `187.78`. This baseline was completed before resumed training.
- Before candidate D2 outcomes, substantial coverage saturation is frozen as
  every stage retaining at least `85%` of its final 5,000 touched features from
  the 1,000 boundary. Material lower-decile harm is a wholly negative 95%
  bootstrap interval or a point drop at most `-2,056.71`.
- HOLD R1.5, R2, phase-0 labels, MCTS/UCT, unrelated sidecars, milestone
  classifiers, preview fitting, D0 reuse, C, and dashboard promotion. Do not
  turn R1b into a parent/mixture/promotion sweep.
- R1b stopped permanently at exactly 5,000 for this gate. Checkpoint audit is
  a full `PASS`; touched-feature coverage grew `2.25x / 2.52x / 2.49x / 2.88x`
  from 1,000 and is not saturated under the frozen 85% rule.
- R1b D2 result: `PROMOTION CANDIDATE PASS / PRE-C HOLD`. Paired score lift
  `+4,506.57`, CI `[+1,176.73, +7,894.16]`; P3072 `32/512` versus `18/512`,
  difference `+2.73 pp`, CI `[+0.20 pp, +5.47 pp]`; median `+4,123.5`, lower
  decile `+1,455`, moves `+12.25`.
- Changed games: `276` wins / `235` losses / `1` tie; P3072 had `30` gains and
  `16` losses. Fixed tail audit passed with an improving below-P5 rate and zero
  candidate-only terminal anchor/max-displacement failures.
- Next authorized status is HOLD. Before C can be considered, the separate
  held-out h10/h20/h40 pre-promotion diagnostic must be explicitly authorized
  and reviewed. Do not inspect C, retrain, update the incumbent/dashboard, or
  start R1.5/R2 meanwhile. See `R1B_PRE_C_DECISION.md`.
- The leakage audit subsequently passed on `21` globally unsampled ancestries:
  no R1b restart/normal-seed or D0-D2/C identifier collisions, exact simulator
  preview/cycle round-trip, and outcome-independent current-state selection.
- The frozen 16-repeat h10/h20/h40 diagnostic completed `NEUTRAL`. At h40 both
  arms reached first built 1536 in `30/336` paths; score difference `-324.08`,
  CI `[-2,495.47, +1,761.65]`; survival `+0.60 pp`, CI
  `[-5.95 pp, +6.25 pp]`. No pre-C harm block fired. At h20 R1b was `12/336`
  versus `9/336`, with both stream blocks positive.
- Interpretation: the small failure-only slice does not localize the D2 gain,
  but it does not contradict it. D2's direct positive P3072 endpoint is the
  stronger level-up result. Recommend authorizing exactly one sealed-C
  confirmation evaluation. C remains unopened until explicit authorization.
- C was subsequently authorized and opened exactly once per arm. Preflight and
  postflight content hashes matched for the manifest, current incumbent file,
  all four incumbent components, and the exact 5,000-episode R1b checkpoint.
- C result: incumbent `21,131.63` versus R1b `21,919.80` mean
  score-minus-starter. Paired lift `+788.18`, CI
  `[-2,412.96, +4,021.11]`; median `+453`; moves `+3.89`; P3072 tied
  `21/512` with `18` gains and `18` losses.
- Lower-decile difference was `-113.7`, CI `[-1,593.66, +1,467.09]`; the
  fixed 23-case corner/tail audit did not block. These secondary results do not
  rescue the primary criterion.
- Final decision: `CONFIRMATION FAILED / NO PROMOTION`. The paired score CI
  crosses zero, so the current phaseblend incumbent remains active. Do not
  rerun C or reinterpret the positive D2 result. HOLD retraining, R1.5, R2,
  dashboard/incumbent changes, replay promotion, and all unrelated work for
  review.

## Current Status

The first direct endpoint selective-rollout gate on fresh-root one-`768`
failure states failed promotion on 2026-07-08:

- Corpus: `24` fresh one-`768` failure roots from the 2520:2620 pool.
- Endpoint: h40 `reached_1536`, meaning `max_tile_excl_starter >= 1536`.
- Result: selected-vs-incumbent mean lift `-1.04 pp`, root-cluster CI
  `[-2.60 pp, 0.00 pp]`; promotion screen failed.
- Do not use `raw_one_1536` for this gate; the fixed starter tile already makes
  that true.

The larger pooled direct endpoint gate over source-diverse one-`768` failure
states also failed promotion on 2026-07-08:

- Corpus: `191` fresh root-capped one-`1536` / one-raw-`768` near-failure
  records from `2020:2720`.
- Gate: `48` roots, h40 `reached_1536`, pilot `2` repeats/action,
  eval `4` repeats/action x `2` independent blocks.
- Pilot endpoint hits were extremely sparse: only `2` action rows hit in
  `185` pilot action rows, so the gate changed only `2 / 48` cases.
- Result: selected-vs-incumbent mean lift `+0.78 pp`, root-cluster CI
  `[0.0 pp, +2.34 pp]`; block `0` positive, block `1` exactly zero.
- Promotion screen failed because the CI touches zero and all blocks were not
  positive.
- Do not rerun this h40/pilot2/eval4x2 configuration over mostly one-`768`
  states.

A later matched pre-duplicate-`1536` selector screen remains diagnostic only:

- Air-survival top vs root-matched random on `9` fresh roots had a mean
  top-minus-random root-rate difference of `+9.72 pp`, but the bootstrap CI
  was `[-17.0 pp, +36.5 pp]` and only `4/9` roots were positive.
- A downstream non-near duplicate-`1536` -> near-adjacent `1536` diagnostic was
  easy locally (`271/396` hits, `29/29` cases with any hit), but represented
  only `3` selected roots.
- Conclusion: useful mechanism evidence, not selector or policy promotion.

The first new-search-family MCTS/UCT benchmark also failed promotion:

- Corpus: the same fixed `24` fresh one-`768` failure roots from the failed h40
  direct gate.
- Target: h40 `reached_1536`.
- Mean-value MCTS, `64` sims depth `12`: target lift `0.00 pp`, root-cluster
  CI `[-2.60 pp, +3.13 pp]`; score lift `-82.5`.
- Robust-child MCTS, `64` sims depth `12`: target lift `+0.52 pp`,
  root-cluster CI `[-2.08 pp, +3.65 pp]`; score/survival intervals crossed
  zero.
- Target-directed UCT, `128` sims depth `20`, target bonus `100000`, no leaf
  value: target lift `-1.56 pp`, CI `[-4.69 pp, +1.04 pp]`; score and survival
  worsened on mean.
- Conclusion: budgeted UCT is feasible but not a policy improvement on this
  hard corpus, even when the tree reward directly targets first-`1536`.

## Active Work

Active work is specification and source-diversity readiness only, not policy
promotion:

- Do not collect more incumbent self-play or run downstream ladder diagnostics
  while waiting for external human data.
- Do not run more MCTS/UCT budget, exploration-constant, or reward-objective
  sweeps from the current corpus.
- The human/tracker pipeline is available and verified:
  `python -m threes_rl.human_diagnostics_pipeline --events-jsonl ... --policy-file threes_rl/current_incumbent_policy.txt`.
- That pipeline now writes imported replays, high-board reservoirs,
  pre-promotion transition windows, human-root support-ladder windows, optional
  policy-agreement diagnostics, and optional no-label top-two scan artifacts.
- Batch intake is available for the watch inbox:
  `python -m threes_rl.human_diagnostics_batch --run`.
  The batch command skips current sessions, writes an inbox JSON/HTML report,
  tracks non-starter `1536`/`3072` intake targets from replay frames, and is
  readiness plumbing only.
- No local `events.jsonl` human/tracker sessions are currently present. The next
  informative input is strong human/tracker data, ideally at least five
  independent games reaching non-starter `1536` and one or more reaching
  `3072`.
- Human data is the fastest source-diversity injection, but not the only
  theoretical unblock. A self-play-only branch may reopen only through the
  frozen phase-0 oracle spec in `PHASE0_ORACLE_GATE_SPEC.md`.
- Phase-0 execution is currently held. The dry-run audit found `29`
  root-capped h40 first-non-starter-`1536` roots (`12` success, `17` failure)
  but only one behavior family, `phaseblend_incumbent_lineage`, so the corpus
  fails diversity requirements.
- A retained replay inventory then found enough existing normal-start replay
  coverage to satisfy the frozen diversity rule via downsampling:
  `598` extractable h40 roots across `10` behavior families, including
  `176` success roots and `422` matched failures. The raw pile is not ready
  because `phaseblend_incumbent_lineage` is `437 / 598` roots (`73.1%`), but a
  retained subset can be ready with `0` new roots, for example all non-largest
  roots plus a cap of `161` phaseblend roots (`322` roots total, largest share
  `50%`). Conservative no-incumbent options also exist, such as
  `corner2_lineage` + `expectimax_baseline` at `49 + 49` roots.
- If all current phaseblend roots were kept, the new-root burden would be
  large: `corner2_lineage +356`, `expectimax_baseline +388`,
  `phaseblend_cheap_lineage +427`, `td_student_lineage +429`, `ntuple +432`,
  and any zero-root/new family `+437`. Prefer retained downsampling over new
  acquisition unless we explicitly require human roots.
- Allowed work before execution: revise/freeze spec text and run read-only
  corpus/power audits and retained-candidate manifest construction. Do not run
  rollout labels, search, fitting, or normal-start evaluation for phase 0 until
  the selected corpus manifest satisfies the frozen requirements.

Standing promotion rule for any future gate or selector:

- At least `20` independent roots, or a clearly labeled smaller diagnostic if
  used only for mechanism discovery.
- Positive paired direct-endpoint lift across independent seed blocks.
- Root-cluster bootstrap interval excludes zero.
- Pilot/action-selection seeds must be disjoint from evaluation seeds.
- Score/survival non-inferiority margins must be predeclared.
- Only after that: run normal-start paired evaluation. No dashboard capability
  promotion before normal-start evidence.

## Killed For Policy-Facing Work

These branches should not be restarted without new direct-endpoint evidence:

- `supportstock` support-preservation rank/profile iteration.
- `second768` archive objective promotion.
- `regen768` post-`3072` archive objective promotion.
- Adjacent/duplicate-`384` and other hand-designed support-rank variants.
- Further post-`3072` ladder chaining from the current correlated pools.
- Success-path selectors as policy evidence.
- The current h40 `reached_1536` selective gate configuration on the 2520:2620
  one-`768` failure pool.
- The pooled h40 `reached_1536` selective gate configuration on the 2020:2720
  one-`768` failure pool.
- Incumbent-leaf MCTS/UCT first-action selectors tested on the fixed hard
  24-root corpus:
  - `64` simulations, depth `12`, root action by mean value.
  - `64` simulations, depth `12`, root action by visit count.
- Target-directed MCTS/UCT on the same hard corpus:
  - `128` simulations, depth `20`, target bonus `100000`, no n-tuple leaf
    value.
- Further MCTS/UCT budget, exploration, and reward sweeps over the current
  incumbent-generated hard corpus.

Reason: recent causal labels showed dense immediate support gain but zero
downstream raw2/first-`1536` lift, and heldout/root-diverse screens did not
replicate enough independent root conversion. The new MCTS benchmarks likewise
failed to improve direct h40 first-`1536` probability on the fixed hard corpus.

## Held

- Broad value fitting, action-prior fitting, sidecars, and n-tuple capacity
  probes.
- Normal-start capability claims from continuation or selected-state artifacts.
- Dashboard high-score promotion from anything except real normal-start games.

## Current Incumbent

`ntuple_phaseblend_expectimax2` with:

- Parent MC1000 table.
- Student1 blend weight `0.25` on all phases.
- Replay calibration weight `0.05` from mid phase.
- Old default action-label sidecar weight `0.10` on endgame.

This remains the actor for acquisition and diagnostics until a direct
selective-rollout gate passes the promotion rule above.

## 2026-07-26 O1 Goal-Conditioned Option P0

**Decision: `HOLD_O1_DATA_OR_POWER`.**

- The exact-depth-3 program remains permanently killed. O1 is a separate
  closed-loop, machine-goal option-policy branch and does not reinterpret any
  C1/C2/K1 result.
- O1 representation, A3 pair-specific/safe merge semantics, A4 power design,
  provenance, stream, service, and integrity checks passed.
- The powered untouched design is coherent at `N=192`: OR-1.50 power `0.959`,
  48 roots per starting stage, MDE OR `1.50`. The valid N=144 power result is
  retained but structurally ineligible.
- Existing-corpus support is zero under the frozen exclusions. All `3,242`
  retained replay files are in prior forensics, continuation, diagnostic,
  rank/score, or score/playlist-selected namespaces. No candidate replay
  content qualified for parsing and no root entered train/dev/test.
- O1 is held for clean natural-data availability, not killed as a
  representation. E0, labels, rollouts, fits, policy outcomes, and all future
  77B-80B streams remain unopened.
- The proposal-only next course is the online normal-start scale-equivariant
  option curriculum in `O2_ONLINE_OPTION_CURRICULUM_PROPOSAL.md` at SHA-256
  `642aa15b6e3ae6e10487b7a4d3686d54302ec4584848c2d8355d7b900550348b`.
  O2 execution requires a separate frozen authorization.
- `CONTINUE=none`; `HOLD` O1 E0, O2 execution, human training ground,
  normal-start policy evaluation, and promotion; `KILL` exact depth 3;
  `PROMOTE=false`.

## 2026-07-26 O2 Outcome-Free Curriculum Preflight

**Decision: `READY_O2_YIELD_PILOT_PREFLIGHT`.**

- O2 remains a prospective normal-start, scale-relative option-curriculum
  branch. It does not reuse O1's empty retained-root corpus and does not
  reinterpret O1, C1/C2/K1, G3/G4, or human evidence.
- Amendment A4 corrects the pilot arithmetic without changing the scientific
  design. The 128-root pilot now has two jointly required support layers:
  a 92-root disjoint structural match (`4` roots in each lower cell and `7`
  in each `T=768` cell), plus overlapping whole-root availability of at least
  `7` lower and `8` transfer roots per cell. The impossible 144-root disjoint
  Wilson interpretation is explicitly rejected.
- The later corpus allocator remains disjoint: `128` lower-scale train roots,
  `48` development roots, and `192` untouched mechanism-test roots. Training
  remains strictly `T={48,96,192,384}`; `T=768` is development/test transfer
  and `T=1536` is descriptive only.
- Prospective power is coherent for the frozen primary gates: OR-1.50 power is
  `0.91699` for the lower estimand and `0.94922` pooled. The `N=64` transfer
  panel is intentionally underpowered (`0.51465`, MDE grid OR `1.75`) and
  cannot independently promote or kill the branch. Worst-case OR-1.50 P3072
  power at the sealed `N=2560` confirmation design is `0.84766`.
- Four genuine collector families, 11,520 reserved stream rows, zero
  collisions over 9,101 historical sources, exact historical calibration,
  storage/runtime projections, nice-10 operation, services, dashboard record,
  and protected top three all passed.
- No pilot game, stream, rollout, label, fit, candidate action, policy outcome,
  score outcome, incumbent change, or dashboard change occurred.
- `CONTINUE=none`; `HOLD` O2 yield-pilot execution pending separate
  authorization, all option training and policy evaluation, human training
  ground, incumbent/dashboard changes, and promotion; permanent historical
  KILL/HOLD decisions remain unchanged; `PROMOTE=false`.

## 2026-07-26 O2 Yield Pilot Terminal

**Decision: `HOLD_O2_PILOT_OPERATIONAL_INTEGRITY`.**

- The authorized one-shot pilot completed all `128` fresh normal-start roots:
  exactly `32` each for corner2, expectimax2, parent MC1000, and QD-v2.
  All roots are ancestry-unique, all replays are retained unconditionally, and
  the append-only ledger contains exactly `128 opened + 128 completed` events
  with zero retries.
- The post-collection attempt audit stopped on an engineering field-name
  mismatch: it requested `family_game_index` from completion rows, whose frozen
  field is `game_index`. The six-hour post-collection guard passed first;
  charged evaluator time was `419.663s`, output was about `28 MiB`, and free
  disk remained about `149 GiB`.
- The failure occurred before `support_analysis`. No support artifact exists,
  no A4 structural/availability cells or yield outcomes were opened, and this
  is not `HOLD_O2_DATA_SUPPORT`, a representation failure, or a policy result.
  The immutable one-shot marker/result may not be rerun or edited.
- Labels, fits, option rollouts, policy/score outcome analysis, incumbent
  changes, and dashboard changes remain zero. Services and protected top three
  `263670/261369/258561` remain healthy.
- `CONTINUE=none`; `HOLD` O2 support decision, corpus collection, training,
  mechanism/policy evaluation, confirmation, human training ground, and
  promotion; `KILL=false`; `PROMOTE=false`.

## 2026-07-26 O2 Scan-Only Recovery

**Decision: `HOLD_O2_DATA_SUPPORT`.**

- The original `HOLD_O2_PILOT_OPERATIONAL_INTEGRITY` remains authoritative
  and immutable. A separate recovery namespace reproduced the exact
  `KeyError('family_game_index')`, adapted only `game_index` for the audit,
  and scanned the existing 128 retained replays once under the frozen A4
  support semantics. No game was rerun.
- Source and attempt integrity passed: `128` unique ancestries/replay
  paths/replay hashes, `32` roots per family, `128 opened + 128 completed`
  attempt events, zero retries, and exact stream identity.
- Both A4 support layers failed. The disjoint structural MILP stopped on
  `cell_candidate_shortfall`; overlapping availability passed only `7/20`
  cells. Credited roots by stage 0/1/2/3 were: `T768 0/0/0/0`,
  `T384 4/4/0/3`, `T192 5/5/0/9`, `T96 9/9/0/9`, and
  `T48 9/9/2/9`. The descriptive `T1536` count was zero.
- This is a prospective natural-data support HOLD, not a representation or
  policy failure. No corpus expansion, option rollout, label, fit, policy
  evaluation, score/action/max-tile inspection, incumbent change, or
  dashboard change occurred.
- `CONTINUE=none`; `HOLD` corpus collection, training, mechanism/policy
  evaluation, confirmation, human training ground, and promotion;
  `KILL=false`; `PROMOTE=false`.

## 2026-07-26 O3 Event-Conditioned Option P0

**Decision: `READY_O3_EVENT_ACQUISITION`.**

- The outcome-free P0 reproduced the frozen 102,557-parameter schema, all
  five behavior-family signatures and pairwise distinctness, and the exact
  N192 mechanism power contract. OR1.50 full-gate power is `0.9169921875`;
  the frozen grid MDE is OR1.50.
- The nonadaptive universe contains 20,500 planned roots, exactly 4,100 per
  family, assigned before content to 5,020 train, 1,675 development, and
  13,805 untouched-mechanism roots. The 26,516-row future stream contract is
  internally valid and collision-free against 9,109 historical metadata
  sources.
- O2 entered only through ledger/log aggregates and a byte hash of the sealed
  support JSON. No O2 support row or replay body was parsed. P0 generated zero
  games, streams, labels, fits, policy outcomes, or score observations.
- `CONTINUE=one frozen O3 acquisition`; `HOLD` training, mechanism,
  development, confirmation, human training ground, and promotion until
  their preceding gates pass; `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O3 Acquisition Terminal

**Decision: `HOLD_O3_ACQUISITION_INTEGRITY`.**

- The one-shot acquisition stopped at its frozen operational guard after
  `18,990/20,500` complete roots because a competing heavy Python/Threes
  process was detected. The runner sealed the HOLD; it may not be resumed or
  reinterpreted under the existing marker.
- Data written before the stop are internally complete and balanced:
  `3,798` unique roots per family, `18,990` unique ancestries/replay hashes,
  and `18,990 opened + 18,990 completed` attempt events with zero retries.
  Train and development roles completed; untouched-mechanism acquisition is
  `12,295/13,805`. The remaining `1,510` roots were not generated.
- The stop occurred before any replay support scan. It is an operational
  integrity HOLD, not a data-support, representation, policy, or capability
  result. No selected-root, label, model, rollout, mechanism outcome, policy
  evaluation, score/action inspection, incumbent change, or dashboard change
  exists.
- Runtime was `27.623h`, output was `4,702,190,939` bytes, free disk remained
  `143.367 GiB`, services were healthy, and protected top three remained
  `263670/261369/258561`.
- `CONTINUE=none`; `HOLD` acquisition retry and every downstream O3 gate
  pending research-lead review; `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O3 Acquisition Recovery Preflight

**Decision: `READY_O3_ACQUISITION_RECOVERY`.**

- The original acquisition remains permanently sealed. The recovery may run
  only the exact P0-minus-completion complement: `1,510`
  untouched-mechanism roots, exactly `302` per frozen family.
- Metadata and byte-only audits reproduced `18,990` original roots,
  `3,798/family`, unique ancestry/replay hashes, all `37,980` attempt events,
  zero retries, and unchanged source hashes without parsing any original
  replay body.
- Exclusive ownership, attributed process guards, collision freedom, all
  five collector/signature locks, nice `10`, services, protected top three,
  `143.302 GiB` free disk, and the combined storage projection passed.
- Marker and preflight sealed with zero games, streams, support reads, labels,
  fits, policy outcomes, or dashboard/incumbent changes.
- `CONTINUE=one exact 1,510-root recovery`; `HOLD` every downstream O3 gate
  until union and support gates pass; `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O3 Recovery Terminal and V2 Integrity Reseal

**Decision: `HOLD_O3_SELECTED_INTEGRITY_RESEAL`.**

- The recovery itself completed all `1,510` missing roots and formed the
  exact original `20,500`-root universe. Union and support checks passed:
  `4,100` roots per family, roles `5,020/1,675/13,805`, unique ancestry and
  replay hashes, zero drift, and `12,922` support candidates from `7,607`
  roots.
- Root selection also passed scientifically with no deficits and exact
  `96/32/192` train/development/untouched allocation. The recovery terminal
  HOLD arose only because integer keys in six count maps became strings
  during JSON serialization, so the embedded pre-serialization selected
  hash did not validate after reload.
- A separately authorized V2 integrity reseal proved the six-path coercion
  mechanism in `16/16` focused and `156/156` applicable regression tests.
  Its CLI then failed closed before test-evidence creation because the
  repeatable `--command` option shared the subcommand parser destination.
  The immutable V2 envelope is
  `HOLD_O3_SELECTED_INTEGRITY_RESEAL`, file/payload SHA-256
  `f466cae4e298edfc25499a90a78bfb6d6e037e2d065be72eb0de498cf9b31d57` /
  `58b55acb66033092dad5e789421d4cb60adfe960ccf25e1a6ef277e81141357d`.
- This is an engineering orchestration HOLD, not a support, representation,
  model, policy, or capability result. The original recovery HOLD and all
  source artifacts remain immutable.
- `CONTINUE=none`; `HOLD` integrity retry and all O3 training/mechanism/
  capability gates pending research-lead review; `KILL=false`;
  `PROMOTE=false`.

## 2026-07-27 O3 Selected-Root Integrity V3

**Decision: `READY_O3_OPTION_TRAINING_INTEGRITY_RESEALED_V3`.**

- V3 preserved the original acquisition, recovery, and V2 terminal HOLDs
  byte-for-byte. It wrote immutable test evidence first, then exactly one
  terminal envelope in a new namespace.
- The envelope binds all four recovery JSON identities and all immutable V2
  identities. It proves the selected artifact mismatch is exhausted by JSON
  key coercion at exactly six frozen count-map paths; restoring only those
  keys reproduces the embedded pre-serialization hash.
- Test evidence file/payload SHA-256 are
  `6608d39605d38727fb81e85208a6e2e7fc5be14eb04c3fba2624d9f2d131a906` /
  `aef78c400e86bcc35ad15aa2d9937eeb55769bfca5ae2146570f57fc6270188c`;
  envelope file/payload SHA-256 are
  `5bb80bc02597ea934c02f8ebd07eaf0158623232f88ea0408532cdc0039e6696` /
  `622ebf6361527be7283fd51c7a7acff99aa8125b06c76dbc4ee8a801faf3904d`.
- This is an integrity READY only. It authorizes the already frozen O3 label
  preparation sequence but is not model, mechanism, policy, score, or
  promotion evidence.
- `CONTINUE=O3 label preparation`; `HOLD` mechanism, development,
  confirmation, and promotion until their preceding gates pass;
  `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O3 Option-Training Preflight

**Decision: `READY_O3_OPTION_TRAINING_EXECUTION`.**

- A separately hashed charter/runner/test surface passed py-compile,
  `20/20` focused tests, and `199/199` applicable regressions. Tests cover
  exact routing, 96-root/1,152-task accounting, deterministic episode and
  checkpoint behavior, orphan-array resume, attempt multiplicity, sealed
  holdouts, and operational HOLD versus integrity KILL.
- Preflight lock file/payload SHA-256 are
  `b2b355ba08dcc7716d53e90b2a8a1f94fa6a674f97586bb7d45f9d45d8256dd2` /
  `75b7af3245617f33d29bcac8d3cab40de6589b99b2815153f2239e73e9e8d334`.
  It binds 96 restored train roots, 32 development and 192 untouched
  hash-only roots, 320 unique ancestries, and 1,152 collision-free learning
  rows.
- Operational and service gates passed at nice `10`, `142.662 GiB` free,
  with no heavy contention and unchanged dashboard/top-three truth.
  Development and untouched replay content remain unopened; no episode,
  stream, label, model, or policy outcome exists.
- `CONTINUE=one zero-label open`; `HOLD=training execution until the marker
  is independently logged and verified`; `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O3 Option-Training Open

**Decision: `READY_O3_OPTION_TRAINING_OPENED`.**

- Immutable marker file/payload SHA-256 are
  `e00033d12e74c7c1f5a61fc4bfdc31c3a26e466978ea7a4da86b13bc7d624d13` /
  `2061319205b4c95b8d80c8886a813f4c59f5840ab0889a066283721716c24816`.
  It binds the exact READY lock and frozen execute command.
- The marker was the sole new artifact and attests zero episode, consumed
  stream, label, model, policy outcome, development read, or untouched read.
  Process, disk, services, dashboard, and top-three checks passed.
- `CONTINUE=one frozen training execution`; `HOLD=development/mechanism/
  capability work until its preceding gates pass`; `KILL=false`;
  `PROMOTE=false`.

## 2026-07-27 O3 Option-Training Terminal

**Decision: `KILL_O3_TRAINING_INTEGRITY`.**

- The immutable one-shot result sealed at file/payload SHA-256
  `943fefaf4cc2dfcbc50a670119caafc279056ba2d4c492be6680b075ddc32c67` /
  `11ab686df92f75f2d9f3b7e206a3c4ae05e1a92aacf09bea6c45b4dd77c6599b`
  after the frozen normalized successor-geometry contract produced a value
  outside `[0,1]`.
- The exact run committed 123/1,152 episode artifacts and opened task 124.
  It performed no fit and sealed no trained checkpoint or support report.
  Development and untouched replay content remained unopened; no mechanism
  or normal-start policy evaluation occurred.
- This is a representation/serialization integrity KILL for exact
  O3-training v1, not policy-utility or capability evidence. Partial labels
  are retained but uninspected. The run may not be edited, resumed, repaired,
  or reinterpreted.
- Services/top-three remain healthy and disk is `143 GiB` free.
- `CONTINUE=none pending course change`; `HOLD=all downstream O3 work`;
  `KILL=true`; `PROMOTE=false`.
- Research-lead terminal ruling: O3 is permanently closed. Its 123 completed
  and one opened task artifacts, selected roots, learning streams, episode
  bodies, labels, actions, outcomes, and initial checkpoint are prohibited
  from future scientific reuse or inspection. O4 is a separate outcome-free
  branch, not an O3 repair.

## 2026-07-27 O4 Domain-Safe P0

**Decision: `KILL_O4_REPRESENTATION_PREFLIGHT` for exact V2.**

- V1 remains an immutable zero-content engineering HOLD. V2 sealed its
  outcome-free terminal result at file/payload SHA-256
  `897cac07ce2625f5616690f0a4611e11948e6ca58a55b828ee43f92b493893cd` /
  `ed0102032291a8396ffabccdddc657e57779ef2623e21427b49c1ed344d87eac`.
- Representation-domain proof, source identities, whitelist restoration,
  schema/tests, stream contract, N192 OR1.50 power, services, disk, and
  zero-work gates passed. The raw source upper bound independently failed
  the frozen 448-root allocation: `o4_qd_v2` had zero eligible support roots,
  so no candidate replay body or scientific outcome was opened.
- Terminal KILL precedence was triggered by two fail-closed audit flags.
  Read-only attribution shows exact accepted policy signatures/pairwise
  gates were intact but reconstructed-map ordering failed the family-order
  check; the collision list was the preserved V1 zero-work reservation, not
  consumed O4 or O3 streams. This attribution does not alter or reopen the
  authoritative result.
- O4 has generated zero games, consumed streams, labels, models, policy
  outcomes, score/action outcome reads, incumbent changes, or dashboard
  claims.
- `CONTINUE=none`; `HOLD=all O4 downstream work and any successor pending
  research-lead review`; `KILL=true` for exact O4 P0 V2;
  `PROMOTE=false`.

## 2026-07-27 O5 Four-Family Domain-Safe P0

**Decision: `READY_O5_FOUR_FAMILY_DOMAIN_SAFE_PREFLIGHT`.**

- O5 is a new four-family source-feasibility branch, not an O4 rerun. O4
  remains killed and immutable; QD is excluded without replacement because
  it has zero untouched qualifying support after the protected O3-root
  exclusion.
- The marker-bound outcome-free scan passed every immutable identity,
  representation, source, whitelist, semantic policy-order, allocation,
  stream, collision, power, process, disk, and service gate. Terminal
  file/payload SHA-256 are
  `b2ca5368dd6f29debfd0fb0e4c86005c9bae7b92d736ebc5750c5ec71f97a96f` /
  `1707c2982e62a29787b69dae9f6e31c9a042162f0573203ab6f2f38d9d3b7fe1`.
- Exactly 448 unique ancestries were frozen with zero protected-root overlap:
  train/development/untouched `192/64/192`, each family exactly `112`, and
  T48/T96/T192 exactly `150/149/149`. Every frozen cell passed with no
  deficit or backtracking.
- All four accepted policy signatures and six pairwise gates reproduced.
  The fresh 6,272-row 181B-196B reservation has zero collision with the
  complete historical scan, exact 1,152 O3 learning rows, and all 6,272
  spent O4 reservations.
- The unchanged domain-safe representation and N192 sustained-policy power
  contract passed: 102,557 parameters, exact schema/domain proof, OR1.50
  power `0.912109375`, and grid MDE OR1.50.
- O5 P0 generated no game, consumed stream, label, model, rollout, policy
  outcome, score/action inspection, incumbent change, or dashboard claim.
- `CONTINUE=only a separately frozen O5 training execution charter`;
  `HOLD=training, mechanism, normal-start development, confirmation, and
  promotion pending separate authorization`; `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O5 V2 Adaptive Training

**Decision: `HOLD_O5_TRAINING_DATA_SUPPORT`.**

- The accepted V2 marker-bound run completed all `1,152` frozen trajectories
  and all four adaptive cumulative fits without integrity or operational
  failure. Terminal result file/payload SHA-256 are
  `74ac4ca9f375ff93e2fed5dfa5c2154a7b4fcc682654539e05cc67cc4a515e05` /
  `686c34218b0cb06c2411dc4e3ee072587d36ee347088684a71d5d9ec29c866be`.
- Aggregate support passed success, failure, target, family, finite-domain,
  and success-bin gates. It failed only the preregistered true-h40-censor
  gate: `0` observed versus `>=40` required. The exact V2 one-shot is spent
  and may not be retried, swept, threshold-relaxed, or reinterpreted.
- Every R1-R4 checkpoint is sealed non-authoritative and unusable under
  quarantine file/payload SHA-256
  `96a5336f3a9c37dad56447ceedf9481cd39fe0d6f896effa5f47b07b9c461ece` /
  `6658cef4e6a9a111ae1b2cabf5970ce70bd77449470c39f47e44209ac57a4054`.
  No candidate checkpoint or checkpoint-authority artifact exists.
- Development and untouched roots remained hash-only and unopened. No
  mechanism, normal-start, confirmation, incumbent, dashboard, top-three, or
  promotion evaluation occurred. The HOLD is data-support evidence only,
  not a policy-utility or representation KILL.
- `CONTINUE=none pending research-lead course decision`;
  `HOLD=all downstream O5 work and all checkpoint use`; `KILL=false`;
  `PROMOTE=false`.

## 2026-07-27 O5 Support-Mechanism Audit / O6 Proposal

**Finding: `O5_CENSOR_MISS_IS_COMPLETE_COMPETING_EVENT_RESOLUTION`.**

- The source and aggregate-only audit SHA-256 is
  `80de4e1ad8cbf17fe5bdda10874b38219df0deb0c0928752f27a8fe691ec9a76`.
  O5's true-h40 counter is consistent with its loop: censoring occurs only
  after 40 consecutive live transitions. All 1,152 episodes instead reached a
  competing absorbing event (`188` success, `964` failure), so zero censor
  mass is a population-resolution fact rather than a demonstrated code bug.
- This does not reopen O5. Its exact run remains spent, its checkpoints remain
  quarantined, and no individual episode, checkpoint, holdout, or outcome was
  inspected.
- A fresh O6 competing-risks proposal is recorded at SHA-256
  `a3ff3bebc7251cbb6dd60acb5c594cdd9ab427cebb667fb56b8ab1b04ddfc770`.
  It requires a new outcome-free P0, whole-ancestry partitions, balanced
  genuine families/targets, fresh collision-free streams, and exact
  root-cluster common-OR power/MDE before any labels. Administrative censoring
  contributes survival likelihood when present but has no artificial minimum.
- `CONTINUE=O6 proposal review only`; `HOLD=all O6 execution and all O5
  downstream work`; `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O6 P0 Source-Preparation Seal

**Decision: `CONTINUE_O6_P0_PREPARATION_SEAL_AND_EXECUTION_SURFACE_ONLY`.**

- Preserve the accepted O6 source-preparation charter/runner/tests at SHA-256
  `2ee1e4273866f7f40376fb584e908f5a0e10e70446e2540f36bf320ac0edbb11`,
  `c1a1d0a22fa185672e62f0b712d79d8bd01d76e04cebe04bca78b45a7c092dd6`,
  and
  `3d7cbe8f20149f3b21305e8306762f8ede78f2227094f8848dc5b6f383ba0b34`.
  Py-compile, `32/32` focused tests, and `362` applicable regressions passed;
  eight named historical artifact-state/exhaustive checks were deselected and
  remain preserved rather than rewritten.
- The source-preparation audit passed `18/18` dependencies and `26/26` core
  governance identities. It froze, but did not execute, `60` power cells,
  `245,760` datasets, and `1,006,632,960` whole-root bootstraps.
- There is no O6 marker, source scan, selected root, reserved or consumed
  stream, power result, label, model, checkpoint, development/untouched read,
  or policy outcome. The accepted preparation runner remains read-only.
- `CONTINUE=only a separate marker-bound execution surface and read-only
  zero-work preflight`; `HOLD=all P0 execution and downstream science`;
  `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O6 Staged Content-Blind Opening

**Boundary: `CONTINUE_O6_TEST_EVIDENCE_PREFLIGHT_AND_OPEN_ONLY` completed;
`HOLD` before `execute`.**

- Accepted source identities remain exact. Py-compile, `53/53` focused tests,
  and `415` applicable regressions with eight documented deselections
  reproduced before any artifact was opened.
- Test-evidence file/payload:
  `4f7d5b90091dfa23a1d7c674148b6f2fb18e5f61360d5a81f8e68de9a50537ae` /
  `59b86d7bccb973caa1af29d5a3dd95540463cf2ef538f7aa163add5b72dc9084`.
  Preflight-lock file/payload:
  `74afff9908f04484a857af551df2f6538fd77ed27a6fa6fbfc2a2bd2e6502ff4` /
  `a71f066fffae412e780b696a88593b6c941632cba17f0274eab12f85a19ce1b2`.
  READY preflight-result file/payload:
  `d7bae4fb50a0e29a717256f06dda434c8d6f34b0459a1205e51066a93c1356f1` /
  `4ddf35563d451862c3d630662691cc89468625c92036ac2ee9353eb72655bdb1`.
- The single content-blind marker file/payload:
  `bcb5bc559e1023ed0cc71478dd9751b58d0a679bbd0d359363acc81d9c1fd025` /
  `132dafabae870b977a151cd1e477970254b7a69d5a884ba24621879bfe626bd8`.
  Its canonical byte-inventory SHA-256 is
  `c88e5420c2b0e446a0e77f8ea32de57e7b4173e0fc5a84edec02d53e90b5de6c`
  over `30,900` rows, with `content_parsed=false`.
- The marker records zero collision scan, selection, reservation,
  power, label, training, and policy-outcome work. `execute` remains
  unauthorized.
- `CONTINUE=research-lead marker review`; `HOLD=O6 execute, labels,
  training, and outcomes`; `KILL=false`; `PROMOTE=false`.

## 2026-07-27 O6 One-Shot P0 Terminal

**Decision: `HOLD_O6_DATA_PREFLIGHT` at `protected_exclusion`.**

- The exact one-shot command passed marker/source validation and the
  content-blind protected-inventory gate, then stopped before candidate scan
  because the frozen exclusion union failed its unknown-identity-key
  fail-closed check. No exclusion artifact/key list was sealed, and none was
  reconstructed after terminal.
- Protected-inventory file/payload:
  `ec29d02a435846c1627ed00ce90c32549caafd5214355d7fe70bb26aa7eeb4ba` /
  `332fb56d185b158600fa9f546ffb2249bf2fe69a3fe043c81dbbc3f1d2c8c88d`.
  Runtime file/payload:
  `b05b615e652177e6a581a10dbcecc786f9156cae74b845049e8665eda30300a3` /
  `9bd2910a79630c4685e37f13b660132f778db66437ba5d12913998487d581c3c`.
  Result file/payload:
  `4cc27d5ec374eeaf5f14189977a36b9e99ab4411606cdc78f5d94be50a3376a4` /
  `2f37353ea88b71afba4fc81866f6709992a2af861918430ee7630358314a7bc4`.
- Charged runtime was `56.07639694213867` seconds with zero resumes.
  Services, disk, process ownership, dashboard, and top-three checks remained
  healthy. All scientific work and downstream evaluation counters remain
  zero.
- `CONTINUE=research-lead review/course decision`; `HOLD=O6 P0, labels,
  training, and outcomes`; `KILL=false`; `PROMOTE=false`.

## 2026-07-27 J1 Joint Policy/Value Course Boundary

**Decision: `CONTINUE_J1_IMPLEMENTATION_PREFLIGHT`; execution is
`HOLD_J1_IMPLEMENTATION_NOT_READY`.**

- Further O6 continuation/retry is permanently killed as a course. Preserve
  `HOLD_O6_DATA_PREFLIGHT` as the authoritative operational terminal and do
  not reinterpret it scientifically. O5 checkpoints remain quarantined.
- J1 proposal SHA-256:
  `26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2`.
  Readiness-audit file/payload SHA-256:
  `f3e4e8029e159a1db7767164e1623d2e166b139be319d6077d61d7d107a44042` /
  `5b6b9a2383296f82b6547bbd46ddc892b486e4b89f4c325aa88f9c8b15944f99`.
- Selected one from-scratch 411,656-parameter joint policy/value PPO model
  after comparing exactly two existing conservative paths. J1 uses true
  normal starts (`starter_tile=None`), full actor control, dense score delta
  with an exact final-score-minus-start identity, and fixed machine-only
  auxiliaries. It uses no O5/O3 weight, episode body, behavior action, or human
  label.
- Current `train_ppo.py` is only a source pattern. GAE terminal masking,
  complete-root/root-equal batching, and bit-identical environment/RNG resume
  must pass focused tests before implementation readiness.
- Frozen prospective design: 16,384 train roots; 1,024 paired development
  roots; 5,120 paired confirmation roots; unconsumed 213B-226B namespaces.
  Confirmation is powered for a 7% paired score lift (0.972129) and P1536
  OR 1.50 over a prospective 2%-15% base-rate envelope (worst 0.885417).
  Extreme score statistics are descriptive and cannot veto the powered gate.
- A synthetic/fixture-only runtime and storage projection plus compact
  protected-ID denylist is mandatory in the next zero-work preflight.
- `CONTINUE=runner/tests/denylist/zero-work preflight after review`;
  `HOLD=all streams, games, training, checkpoints, evaluations, incumbent and
  dashboard work`; `KILL=O6 continuation/retry and prior permanent kills`;
  `PROMOTE=false`.

## 2026-07-27 J1 Implementation Readiness Terminal

**Decision: `HOLD_J1_IMPLEMENTATION_PREFLIGHT`.**

- The separate J1 charter/runner/tests are frozen at SHA-256
  `7f87bc29c5764ccb290b25558f1cfe999083e9fddb089ea652cac9d0b92ab137`,
  `55d9e3206c2905509466c4962006e6cf3426f76647af6d2e60afe674b80c9bfe`,
  and
  `e6b169f2d629021f96315380a3cf0ff6eece94a30e5027b1ace4d741499fbfa4`.
  Py-compile, `36/36` focused tests, and `697` applicable regressions passed;
  one fixture-data skip and 13 stale historical artifact-state checks are
  documented in immutable test evidence.
- Every immutable and semantic gate passed: 411,656 parameters, true normal
  starts, transition-t GAE, actual clipped PPO with root-equal weighting,
  deterministic four-epoch updates, dense-objective telescoping, legal/finite
  checks, and bit-identical resume at all five frozen boundaries. The compact
  denylist passed for 15 root/corpus and 14 stream manifests; 213B-226B remain
  unreserved and unconsumed.
- Storage and training-runtime projections passed. The sole failed gate is
  paired-evaluation runtime after the frozen 25% margin:
  development `24.730037h > 24h`, confirmation
  `123.650186h > 120h`. This is an outcome-free cost HOLD, not an integrity
  or scientific failure. No threshold, workload, or fixture was adapted.
- Immutable lock file/payload:
  `42d1f8d3d6b7bfd62c173a3147ce1eb7dff465aaa92271e7af6bc5fb3c533825` /
  `e465cec348f987af4c77f062a0e8f8bfa968ddc4ff460b40ba829915791622da`.
  Terminal file/payload:
  `339e3ef6dcf8c5b3eb1951204d08b97b94b3c4816f993d58509b9b341dc364b1` /
  `4d21a092e584d9419a47bef384de164cfc9a8590268a67abefa35afb6b573ce2`.
- No execution marker exists. Zero J1 streams, games, scientific labels,
  scientific optimizer steps, checkpoints, development/confirmation reads,
  policy/score outcomes, human-session reads, incumbent changes, dashboard
  changes, or promotion evidence were created.
- `CONTINUE=research-lead review of this exact readiness HOLD`;
  `HOLD=all J1 execution/science`; `KILL=historical kills unchanged and J1 is
  not killed`; `PROMOTE=false`.

## 2026-07-27 J1a Cost/Power Amendment Terminal

**Decision: `READY_J1A_COST_POWER_AMENDMENT`; execution remains HOLD.**

- The parent `HOLD_J1_IMPLEMENTATION_PREFLIGHT` remains authoritative and
  unchanged. J1a changes only prospective evaluation counts to 896
  development pairs and 4,480 confirmation pairs; the 16,384-root learning
  design is unchanged.
- The preserved progression-power implementation and exact J1 seed contract
  reproduced the published N=1,024 and N=5,120 cells. At amended
  confirmation N=4,480, 7% score-lift power is `0.9518340090`, score MDE is
  `0.0537137790`, worst-case OR1.50 progression power is `0.84375`, and the
  progression grid MDE is `1.50`. Every frozen power gate passed.
- Reused fixture projections passed without retiming. Development and
  confirmation require `21.638783h/24h` and `108.193913h/120h` after margin,
  each `90.161594%` of cap and below the 91% ceiling. The 5,000-move
  sensitivities remain over the evaluation caps and descriptive.
- Immutable lock file/payload:
  `7ed37c9bf1c6ec0fe7e74f36ef4cde8ab5e3bdd8ae1a7d9e1e065e32a21b852e` /
  `b84228d9e5587682fad0cca91e0e5349076ab70674cf0412205712fa05e37850`.
  Terminal file/payload:
  `4ecda2a1101011437c912d884dfb5acecf7e586b87c4646c63354c4ecc5403ef` /
  `abe17a53c1af2b182a488d4fc05b060a214b106652c04462453ad01e75ed9471`.
- No execution surface, marker, reservation, game, label, optimizer step,
  checkpoint, holdout read, or outcome exists. `CONTINUE=research-lead review
  and a separately frozen J1 execution surface`; `HOLD=all J1/J1a execution
  and science`; `KILL=historical kills unchanged and J1/J1a are not killed`;
  `PROMOTE=false`.

## 2026-07-27 J1 Execution-Surface Readiness Terminal

**Decision: `READY_J1_EXECUTION_SURFACE`; all scientific execution remains
HOLD.**

- The separate execution charter/runner/tests are frozen at SHA-256
  `468cc181c32a934fcbc64bb4cadc22758bd0fc46870f0f120f9ac6008ddb696a`,
  `d4367d95aba05ec592310008bae21e7de90905fa1268601dd60cc8fcb2b6f2bd`,
  and
  `cb696e4502d61abd7a24d5781d7c15e2dd8a0ffed538480ecbd2a27434a339cf`.
  Py-compile, `97/97` focused tests, immutable parent `36/36` and `18/18`
  suites, and `697` applicable regressions passed; one fixture-data skip and
  the exact inherited 13 historical-state deselections are sealed.
- The production state machine now has create-once immutable phase artifacts,
  bounded-engine-only dispatch, explicit owner/reclaim and stream evidence,
  marker/manifest/predecessor/candidate lineage barriers, deterministic
  bounded resume, indexed journals, linear I/O, recoverable current-round
  retirement, and idempotent terminal retention. Concurrent open proved
  exactly one marker wins and its bytes cannot change.
- Central 25%-margin projections pass: training `3.309264h/17.482115 GiB`,
  development `21.677866h/0.061394 GiB`, confirmation
  `108.389329h/0.199181 GiB`. Both evaluation phases use `90.324441%` of
  their runtime caps. The 5,000-move sensitivity remains diagnostic and fails
  training storage/file/fsync caps; it does not weaken the live integrity
  boundary.
- Immutable lock file/payload:
  `e7f648eb04d7d197a9a2391352f82af5df6a12f7868ced8c8e9559703adb9fdc` /
  `70c83f640632ec034b346cda355c875f79cc002409474d537ac67a6ab7c975cc`.
  READY result file/payload:
  `ba3e9d67c64b89cf583c2ad1778b073a6a702c003bf1a895c164d6f9f984d4f6` /
  `af5525b35ec5d5c0deab88d1ec00d8215fbb4dc14abb2aaa8dc9aa70b27d556c`.
- The execution root is absent. No phase lock, marker, stream
  reservation/consumption, scientific game, label, optimizer step,
  checkpoint, holdout read, policy/score outcome, human-session read,
  incumbent/dashboard change, or promotion exists.
- `CONTINUE=research-lead review and separately authorized training
  lock/open`; `HOLD=all J1 scientific execution`; `KILL=historical kills
  unchanged and J1/J1a are not killed`; `PROMOTE=false`.

## 2026-07-27 J1 Training Start Terminal

**Decision: `HOLD_J1_OPERATIONAL`; the original execution is permanently
spent.**

- Terminal file/payload:
  `21092fb34631eb0eaf48811caa814ff4d05abbb23c9bc5add85eefd93a8959d3` /
  `9bcc81d217141fdfa801d1fca606c356720e4ac5c0e2a26f9d1ab688ca93dbcf`.
  Retention file/payload:
  `dc339aafdbe32859d07c591a36c9088afa53f5be30412f3340049ca18994ceb0` /
  `11cc89c6a6fe41ff74c472e3fa0b61d179e1cedfa4755cc4f13fe7ced44018b2`.
  Retained-file inventory:
  `7233c65745a9ae7258dbb165b60f4ae55c1cf60376819b80bb9e0be17d677471`.
- The authenticated boundary is genesis sequence zero with zero completed
  roots, zero attempt events, zero optimizer steps, and zero round aggregates.
  The exact 213B-216B stream rows are formally consumed and cannot be reused.
- Root cause is deterministic runtime orchestration: frozen initialization
  produced deterministic algorithms `true`, intra-op `1`, inter-op `12`,
  while the immutable guard requires inter-op `1`. No learning result, policy
  result, checkpoint, or outcome was produced.
- `CONTINUE=J1b operational-repair preflight only`; `HOLD=original J1
  execution and all J1b scientific phases`; `KILL=historical only, J1/J1b not
  scientifically killed`; `PROMOTE=false`.

## 2026-07-27 J1b Operational-Repair Readiness Terminal

**Decision: `READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT`; execution remains
HOLD.**

- The J1b charter/A1/runner/tests are frozen at SHA-256
  `a426801fc3015051ea51517e925a7d1c2e556718e2551ee480b802c8a7422cc1`,
  `64de3de37bff6a08bd95da217dc52d2f4bb58fbf99d28bede263a44d0aa2eb9c`,
  `7d73565c510dfe74b87ec362c05f8928e15a65cb8af5494b5ad9fe5f4c30ca5f`,
  and
  `f7e55b71f7954fcbdd4db61c1693d773b8ea106684ea19ad19998be15f4dbaff`.
  Tests passed `23/23`, `97/97`, `36/36`, `18/18`, and `697` applicable
  regressions with the frozen 13 deselections and one documented skip.
- The clean process established Torch inter-op/intra-op `1/1` and
  deterministic algorithms before parent import and passed the exact first
  operational guard. The initial 411,656-parameter model hash matched the
  legacy genesis probe exactly. No guard was weakened.
- The fresh training manifest has 16,384 unique roots/ancestries and exactly
  the next 213B-216B contiguous stream prefixes; zero compact-denylist,
  original-consumed, internal-role, root, or ancestry collisions were found.
- Immutable lock file/payload:
  `b8b5377370f0e9e04739aae582604ce85f38bd1ddf84b5312a2cf12406f38814` /
  `ef0c1adce5f948a238e81911ab034d84ed297c2b2570d58481fb2906ef2e7e3b`.
  READY result file/payload:
  `108038d15b222afd00c07c9801b460fb4687bfe0a9e8a4fb54a59e58e8907ec6` /
  `5d56b2c3cec39c16590a20f8acf8f10c60db7739e5161a653ea45a779204ba5e`.
- The J1b execution root is absent and all scientific counters remain zero.
  `CONTINUE=research-lead review and separately authorized J1b training
  surface`; `HOLD=all J1b execution/science and all development/confirmation`;
  `KILL=historical only, J1/J1b not scientifically killed`; `PROMOTE=false`.

## 2026-07-27 J1b Training-Only Execution-Surface Readiness

**Decision: `READY_J1B_TRAINING_EXECUTION_SURFACE`; phase activation and all
science remain HOLD.**

- The immutable charter/runner/tests are
  `aeb458781e206f8f16002ffaa311d782b26fdb4076211155a6230b9835e29858`,
  `c586d41f752cff7aa7c36c911008ca72ce147139fedd7586a03e627471282bc5`,
  and
  `86159c76a42c54c47d30e75b92f988773a6c6da580e8bb6b01de0f2a944a516e`.
  The training-only dispatcher establishes and verifies Torch `1/1`
  inter-op/intra-op plus deterministic mode before parent import and the
  unchanged first guard before any scientific artifact. It routes only to
  `execute_training_engine_bounded`; development, confirmation, promotion,
  legacy engines, alternate seed, and restart surfaces are absent.
- Tests passed `29/29`, parent `23/23`, `97/97`, `36/36`, `18/18`, and broad
  `697` with one expected skip and 13 documented deselections. The central
  projection passes at `3.350931h`, `17.521178 GiB`, `35,705` files, and
  `158,665` fsyncs after margin. Operational checks passed at nice `10`,
  `141.550488 GiB` free, healthy services, no competing heavy process, opaque
  human sessions, and unchanged protected top three.
- Immutable lock file/payload:
  `adeae9ce6f9056914da48b79096ee7143a559a2d4e97c02cbe622eff7b0eb79e` /
  `e559d197f299d2ddf62d8d7736c8fa5a6256c90248ed658f9f24c3459c5b11fe`.
  READY result file/payload:
  `3403a9d70e73e38eca7a372bd7db08b855051f1c409b621ebb7a391c45d96213` /
  `84fc2adf7d5204ed1dd1002799fe250575657bb937badc194abaa1a02217b3d2`.
- The exact fresh 16,384-row source remains unmaterialized and
  `j1b_execution_v1` remains absent. Every scientific/work counter is zero.
  `CONTINUE=research-lead review`; `HOLD=phase lock/open/materialize/execute,
  all J1b science, development, and confirmation`; `KILL=historical only,
  J1b not killed`; `PROMOTE=false`.

## 2026-07-27 J1b External Open-Failure Terminal

**Decision: `HOLD_J1B_OPEN_SERIALIZATION_INTEGRITY`; J1b is spent
operationally and all science remains HOLD.**

- Preserve the exact three-file `j1b_execution_v1` boundary at phase-lock,
  phase-lock-result, and marker SHA-256
  `ac12b9f21977a3adcd61ef5f0d8ba60b058306dcc05fdfed423d2ca77c17a0ce`,
  `6a2f63dc8875db394333ac901a919466a6a432083e29feba32ba8917f3ee9bcf`,
  and
  `e99099b87aa6417b4200ee236ef2b770d1524d11b26a878e9f3bf0d749a54cff`.
  Its valid marker payload is
  `c9e48e972a59f699627bfaa949854930672a8c45a6c671be591e175522a107e4`.
- The external terminal/retention file SHA-256 values are
  `2f9cdfacb04a064b67785ab9bb00cac7d3d46bd057912b40ac4c06db0a0ed122`
  and
  `28738328f724a544ee92fc7992ef8f256f0886c2e138a234a863ec0fe55c5f67`.
  The tuple-to-list raw-equality failure is an outcome-free serialization
  defect only. All materialization and scientific counters are exactly zero.
- `CONTINUE=J1c orchestration-only readiness construction`; `HOLD=all J1b
  retry plus all J1/J1c science`; `KILL=historical only and J1 remains
  scientifically live`; `PROMOTE=false`.

## 2026-07-27 J1c Training-Only Orchestration Readiness

**Decision: `READY_J1C_TRAINING_EXECUTION_SURFACE`; phase activation and all
science remain HOLD.**

- The immutable J1c charter/runner/tests are
  `e352262614a7c3c46c53811c727599f9926f6cbd579b99732c6802c8c41462dd`,
  `f50b475ed00efcfb0fa2ac5b4e4a11b0587ec17c4e8e404bad08be8f4f8c990d`,
  and
  `4ff0a2253cd23059d33404b5d3f0829309dc1565657547f6112e9c3d268dc86d`.
  JSON-native normalization and exact post-write serialized-byte comparison
  repair the J1b tuple/list defect without changing scientific semantics.
- The exact fresh 16,384-root source uses the next disjoint 213B-216B ranges.
  Manifest file/payload are
  `135fec4c75db8871e20ab3988471f75538a399e982573bf4a108afc569fe08b7` /
  `c0d7953aa158297d5b515f6b6e4613b6fc22acc059e8e0aa0ff8904ee7e3546d`;
  compact stream authority file/payload are
  `8aff7f07827cfe796a07646215362f1d37e502f64a43b9b7142b53288b7041f6` /
  `30439292fceeeae4832f5259c62e8a954f3103de7246875c353dfb8cea138016`.
  J1b-declared ranges remain spent and no stream is reserved or consumed.
- Final verification passed focused `37/37`, external terminalization `10/10`,
  parent surfaces `28`, `17`, `97`, `36`, and `18`, broad regressions `697`,
  clean-process marker roundtrip `1/1`, and miniature full-chain `2/2`, with
  only 20 exact documented historical/namespace-state deselections and one
  fixture-data skip. Operational and projection gates pass.
- Readiness lock file/payload:
  `a95712126796dcd91a82885aa1990a77e725064970ba34bc1f31306de8ef2368` /
  `15701d62e5ddc7fca7e38702d78ddd54fa7aefbbd6bdf49ea130c2e224f78ef4`.
  READY result file/payload:
  `908c1570f972a612e02815811a9885162a89f9a1e87ea2b081f2801dab7419bd` /
  `e13c27501231af3ea72de1dbea8ac2e9b485b7763e24fb3394391ec326ef37a3`.
- `j1c_execution_v1` is absent and all prospective counters are exact zero
  under J1c-labelled keys. `CONTINUE=research-lead review`; `HOLD=all J1c
  phase lock/open/materialize/execute and science, plus all J1b retry`;
  `KILL=historical only and J1/J1c not killed`; `PROMOTE=false`.

## 2026-07-27 J1c Training Integrity Terminal

**Decision: `KILL_J1C_TRAINING_INTEGRITY`; all downstream phases remain
HOLD.**

- The exact production training run consumed the sealed 65,536 J1c stream
  IDs and completed all 16,384 normal-start roots and all 64 frozen round
  checkpoints. The 321-record commit chain verifies recursively from genesis;
  2,479 attempts started and finished with zero abandoned, including 856
  optimizer units.
- The mandatory training-sanity terminal could not authenticate its round
  metric evidence and failed closed with
  `Training round metric evidence is not authenticated`. No sanity PASS or
  scientific learning conclusion was produced. The round-64 checkpoint SHA
  `053e6a87441114595e09b9fc6a0f7ce71e5acd23c9757da3aadb96df18468c79`
  is quarantined and non-authoritative.
- Terminal file/payload:
  `7ec4fe7627a129dbb7227fcb88df87ab46ee87479d381011103511ec8f2ca414` /
  `c71dac534755add014a0debe6418f75b591df264f1bb096fb5d50fd253d8ce4f`.
  Retention file/payload:
  `8946669ffeba05626ee863f4e2df8536e267920d771eb088ebf84253a2059532` /
  `01a6f95b79c10f343489fa2e2add086001e89691c7c4edf6b6a1b19f8ec66409`.
  Preserve all 18,835 retained files and `4,299,100,371` sealed bytes.
- Development and confirmation remain unopened; human sessions stayed opaque;
  incumbent, dashboard, and top three are unchanged.
  `CONTINUE=research-lead review`; `HOLD=all J1c reuse and every downstream
  phase`; `KILL=exact J1c execution integrity, not a scientific verdict on
  J1`; `PROMOTE=false`.

## 2026-07-28 J1d V2 Metric-Authentication Readiness

**Decision: `READY_J1D_V2_METRIC_AUTHENTICATION_PREFLIGHT`; all science
remains HOLD pending explicit execution authorization.**

- J1c remains permanently `KILL_J1C_TRAINING_INTEGRITY`; its artifacts,
  streams, checkpoint, and metrics remain unusable. J1d V1 is immutable
  pre-correction evidence and sealed no readiness terminal.
- J1d V2 makes one orchestration-only repair: published round aggregates must
  equal the canonical projection from authenticated per-root rows exactly, and
  published/canonical projection hashes must be identical. The inherited
  scientific `1e-12` tolerance and all learning semantics are unchanged.
- The computed parent production-cast fixture reproduces a
  `1.324547654890651e-08` direct-global versus per-root-normalized discrepancy
  for unequal lengths `[1,7,31]`. Rehashed scalar and sequence one-ULP
  mutations fail closed.
- Charter/runner/tests are
  `f3d42d6f4d908c723756e140fc2ba424378f280a18dc99a50b585e59478cd07c`,
  `6ee656ae0288877560df5a6a140777bf341f8a34dbe554c61ebe2812e6147a3d`,
  and
  `9148d70b3d8c8c55b27a75829ef2e5b4df142e124d6265b639667049b4ac5868`.
  Readiness lock/result file hashes are
  `60587f40512555dadab5cc09a0e9802039754034f427a6084b11b7d8146627c7`
  and
  `b891a0d63fd0c532387a64dc719ec20f27dcf15c84aeeb3a094470030076449b`.
- The exact 16,384-root prospective manifest and 65,536 stream IDs use the
  unused offsets 49,152..65,535 and have zero collision. At this readiness
  boundary no phase lock, marker, materialization, owner, reservation,
  consumption, genesis, game, transition, label, optimizer step, checkpoint,
  holdout read, outcome, human-session read, incumbent/dashboard change, or
  promotion occurred.
- Nice-10 focused, parent, broad, 64-round synthetic, computed mechanism,
  one-ULP tamper, crash-resume, service, process, storage, and source audits
  all pass. `j1d_execution_v1` is absent.
  `CONTINUE=research-lead review`; `HOLD=all J1d training, development,
  confirmation, and promotion`; `KILL=J1c exact execution plus historical
  kills, J1 hypothesis remains live`; `PROMOTE=false`.

## 2026-07-28 J1d V2 Training Sanity Terminal

**Decision: `HOLD_J1_LEARNING_SANITY`; development, confirmation, evaluation,
checkpoint use, and promotion remain HOLD.**

- The training-only dispatcher completed all 16,384 unique normal-start roots,
  all 780 expected optimizer steps, and all 64 frozen rounds under the exact
  fresh 49,152..65,535 stream offsets. The 65,536 stream IDs were reserved
  and consumed once with zero collision.
- The 385-unit recursive commit chain, 64 canonical metric publications,
  manifest/root partitions, and save/load checks all pass. Runtime accounting
  is 2,322 started and finished attempts, zero abandoned, and
  `506.507513s` charged active work.
- Frozen scientific sanity passed score direction and entropy, but failed
  value MSE versus the zero baseline and achieved only one of the required
  two auxiliary Brier improvements. This is a clean sanity miss, not an
  integrity or operational fault.
- The round-64 checkpoint SHA
  `cde85c1ca62b9bd045d680ec980ec25e58ae6e7e083b7ccbac1e239cfbb1a41e`
  is quarantined and non-authoritative. It may not authorize development,
  confirmation, evaluation, promotion, or future initialization.
- Terminal file/payload:
  `9ab0c76142aa70041a5f0540abbc3f9b77ac197599f607a646b2952368f13e1a` /
  `e37a32ec2d0ef1df78d804689ee8f529e5cc78bb627b34fbc8728b7840366fb6`.
  Retention file/payload:
  `5fe222bfc3e1681ee3b1cb98db71e2a0b90017c869947329ca49df084ed65518` /
  `39ff3a6f028f7b27ddb775270959b2f0964e7bab8c649e20942f7296e8dbfe2c`.
- Development and confirmation remain unopened; human sessions stayed
  opaque; incumbent, dashboard, and top three are unchanged.
  `CONTINUE=research-lead review`; `HOLD=all J1d downstream work and
  checkpoint use`; `KILL=J1c exact execution plus historical kills`;
  `PROMOTE=false`.

## 2026-07-28 J2 Incumbent-Distillation Readiness

**Decision: `HOLD_J2_INCUMBENT_DISTILLATION_PREFLIGHT`; all J2 science and
execution remain HOLD.**

- J2 remains a live hypothesis, but the frozen N=2,048 sustained full-policy
  fidelity design is not admissible: worst-case P1536 common-OR power is
  `0.643229`, below `0.80`.
- Required real eight-process incumbent throughput/memory evidence and
  synchronous rounds-1-16 online teacher-query evidence are absent. Central
  ideal projections fit, but ideal scaling is explicitly not evidence.
- The model/schedule contract is otherwise coherent: 410,117 parameters,
  ordinary BC on 8,192 teacher roots, 2,048 paired full-policy fidelity roots,
  and 16,384 PPO roots. The KL anchor is positive through round 16 and zero
  from round 17, requiring exactly 4,096 online teacher-query roots.
- Integrity, source/provenance, 135,424-stream collision, zero-work,
  service/process/disk, and test gates all pass. No J2 marker, reservation,
  consumption, teacher query, label, game, optimizer step, checkpoint,
  holdout read, policy outcome, human-session read, incumbent/dashboard
  change, or promotion occurred.
- Readiness lock/result file hashes are
  `c3f08429b625369263b75a5724b3abfdf2487d6a9fd2414897c7aaca7fd74488`
  and
  `8c24be58bb6a30cd2cf302f17894b69e131f3b3c6092a4e71801c6b0f2f96eab`.
  `CONTINUE=research-lead review`; `HOLD=all J2 execution and science`;
  `KILL=J1c exact execution, J1d checkpoint reuse, and historical kills;
  J2 not killed`; `PROMOTE=false`.

## 2026-07-28 J2A1 Distillation/Fidelity Execution-Surface Readiness

**Decision: `READY_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE`;
`execution_authorized=false`, and all scientific work remains HOLD.**

- The frozen source hashes are charter
  `dbe3470f67229c086f514de20efdd2daf074329df81ca66611895fecabef8f61`,
  runner
  `b5435d6d5d0999b035220a6763646ee133b23f06e79f45456d9c5af083dfe8c1`,
  and tests
  `bb1c8fffa52dea332032447f60426addfbd0acaf2bc5453feb8620856062889d`.
  Final nice-10 evidence is 1,160 passed, zero failed, one expected skip, and
  14 documented stale-state deselections.
- The active A1 authority is exact: 8,192 BC roots, 6,144 validation pairs,
  14,336 teacher roots, 20,480 game arms, 63,488 streams, unique roots and
  ancestries, and eight equal 768-pair strata. Central projected cost after
  margin is `45.43873303943208h / 20.41377067565918 GiB`; both frozen caps
  pass.
- Readiness lock/result file SHAs are
  `1aefee84417c4dda5f17f0309b7b5fd18e2f7a418635f8dc16a90e9c5503da13`
  and
  `a90d1600502264d42315e0806d7665be679e06111aacc006dc193e88baa97d22`;
  retention file SHA is
  `dd258ffbded154ec299d5b48368cd21c9d35e585464079d57009a0e960eb28eb`.
  All parent, source, authority, schema, projection, operation, and retention
  checks pass.
- The future execution root is absent and every real-work counter is zero.
  `CONTINUE=research-lead review of future phase-lock authorization`;
  `HOLD=all J2A1 execution, PPO, development, confirmation, and promotion`;
  `KILL=historical kills only`; `PROMOTE=false`.

## 2026-07-28 J2A1 Pre-Phase Operational HOLD

**Decision: `HOLD_J2A1_DISTILLATION_OPERATIONAL`; no scientific work
started.**

- Scientific authorization file/payload SHA:
  `29ea95388165250b7b7f7db909698ec853101f85bf62e81445e23540879e576f` /
  `b2f5792e16be8ee8e08109fdf894f6243ee7dae840706656ff349da0e71c277b`.
- `seal-phase-lock` failed closed because the hash-bound execution guard
  compared an unchanged top-three tuple directly to a list. All values and
  every other operational check passed.
- No phase lock or later artifact exists, the execution root is absent, and
  every scientific counter remains zero. The accepted execution surface was
  not changed or retried.
  `CONTINUE=research-lead review of a versioned orchestration repair`;
  `HOLD=all J2A1 execution and downstream phases`; `KILL=historical only`;
  `PROMOTE=false`.

## 2026-07-28 J2A1 V2 Execution-Surface Readiness

**Decision: `READY_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_V2`;
`execution_authorized=false`.**

- V1 remains preserved and killed for execution reuse. V2 makes only the
  exact list/tuple-to-three-integer-tuple top-three comparison repair.
- Charter/runner/tests SHAs are
  `d9c5382d803c606c29415fc020fa7d63762dfcb053232d1ac904f21827d74dd4`,
  `044a67bf9b34b311787e3e7de246c4ce62a33f4f8ae47d211f6a76dd231a22f3`,
  and
  `b211bfac0bb2e18c87dddcd72a0c8e7f1a0c3cbd76fee92572133aefa7abd95d`.
  Nice-10 evidence is 1,234 passed, zero failed, one expected skip, and 14
  documented deselections.
- Readiness lock/result/retention file SHAs are
  `259df7e65be1e9cf73e93424cc40d4dadb6e27f87593abbcaa9d577e14d49702`,
  `c445d7ab1478b22b7bb7d74e06533566e519a4b400ced5db09555833fd3ad045`,
  and
  `e2b8bea7a7570b1268339d68cb063b11d8e84d2e3535f4a10b352b1c0590d068`.
  Parent, V1 HOLD, authority, projection, operation, and zero-work checks pass.
- Counts/costs are unchanged and both future V2 namespaces are absent.
  `CONTINUE=research-lead review of a future V2 authorization`;
  `HOLD=all J2A1 execution, PPO, development, confirmation, and promotion`;
  `KILL=V1 execution reuse plus historical kills`; `PROMOTE=false`.

## 2026-07-28 J2A1 V2 Production Terminal

**Decision: `HOLD_J2A1_V2_DISTILLATION_OPERATIONAL`; J2A1 V2 is spent and
all downstream work remains HOLD.**

- The one authorized V2 dispatcher sealed exact authorization, lock, marker,
  and materialized identities, then reserved and consumed the frozen
  14,336-root / 63,488-stream authority under one marker.
- Stage A closed exactly 3,048 teacher-root attempts with zero abandoned or
  open attempts. The charged active runtime reached
  `259763.24813699722` seconds (`72.1564578158` hours), so the frozen 72-hour
  final operational guard sealed a clean HOLD.
- No family-support gate, optimizer step, checkpoint, mechanism gate, student
  arm, fidelity result, PPO, development, confirmation, human-session read,
  incumbent change, dashboard change, or promotion occurred. There is no
  authoritative checkpoint or successor-review authority.
- Authoritative terminal and retention file SHAs are
  `13dd5c3a8eeb79d03149da0fa99a19aee3e6a657109e7fe4104a149d5d02ca6b`
  and
  `93f3a5ac0e155b16af84fc06165cc4e23cbd4184b10b96cc77dc9870b1c315ac`;
  both reload and chain verification pass. Retention covers 3,058 files and
  1,782,523,714 bytes.
- Final operations remain healthy with `127.70 GiB` free, no competing heavy
  process, protected top three unchanged, and human-session content unread.
  `CONTINUE=false pending research-lead decision`;
  `HOLD=J2A1 V2 plus PPO/development/confirmation/promotion`;
  `KILL=false for V2, with V1 reuse and historical kills unchanged`;
  `PROMOTE=false`.

## 2026-07-30 J2A1 V3 Recovery Readiness

**Decision: `READY_J2A1_V3_DISTILLATION_RECOVERY_PREFLIGHT`;
`execution_authorized=false`.**

- Preserve V2 permanently at
  `HOLD_J2A1_V2_DISTILLATION_OPERATIONAL`. V3 binds its exact 3,048
  completions, 6,096 attempt events, 14,336-root authority, 63,488 streams,
  and retained bytes without opening scientific root content.
- The frozen unfinished authority is exactly 11,288 roots at SHA
  `dca4de9005bede7e710ce004ade443aef5a0eda3c28f3994157a136bde0d34a9`.
  Existing V2 stream consumption is reused as authority; no new reservation,
  consumption, replacement, filtering, or duplicate work is authorized.
- The 72-hour cap is elapsed wall time. Observed V2 wall time is
  `9.05785490612189h`; projected total is `42.602824125381694h`, or
  `50.989066430196644h` with the frozen 25% remaining-work margin.
- Result/retention file SHAs are
  `23199ead16dce7ac87ea7d955bba5c913be632f624fa8771fc01a07669ab33ae`
  and
  `26e07603590d39e5402e6f95f35efc94933c83d68931e42a0b1de4b9f49c3246`.
  All scientific and future-execution counters remain zero.
  `CONTINUE=separate research-lead review`;
  `HOLD=all V3 execution and all scientific/downstream work`;
  `KILL=historical locks plus V1 reuse`;
  `PROMOTE=false`.

## 2026-07-30 J2A1 V3A1 Reproducibility and Headroom HOLD

**Decision: `HOLD_J2A1_V3A1_RECOVERY_EXECUTION_HEADROOM`;
`execution_authorized=false`.**

- Preserve V3 as spent and unexecuted at
  `HOLD_J2A1_V3_RECOVERY_EXECUTION_SURFACE_REPRODUCIBILITY`. The exact 41/1
  parent result is a chronology defect only; all immutable identities,
  authority, no-double-consumption, projection, service, and zero-work checks
  remain valid.
- V3A1 source SHAs are amendment
  `397db39026b5eb42d6f1ed633f11de0d1c773ebadcdd69d8e9cce2d7811f9c5f`,
  runner
  `64cc16d99c9366d8c968486e8ee159b9ca1326ea1de1fac2ec44c56cea65cbeb`,
  and tests
  `ae4c689bee1c852b024cfa8451eaedf2e9ca160a4135189abd4c48b50dde6012`.
  The sealed package is post-seal reproducible at 43/43 focused and
  365 passing inherited/applicable tests with only the prior eight
  deselections.
- Projected incremental use is `20,270,813,374` bytes. The sealed projected
  free space is `102.481632 GiB`, but cushion above the hard floor is only
  `2.481632 GiB`; `2.518368 GiB` more free space is required before the
  5-GiB launch-cushion gate can pass.
- Lock/result/retention file SHAs are
  `eba9434e05fd9dcad42b423c4e4e88ccba48e901019563613dd1fda7f141e06a`,
  `cb06f22c90b0df34a5eded5fe90fab25d875cd1ff7cd8e85db474075c21a1fa8`,
  and
  `5dd3973c3930c30fd531ba7b67d75535ec5e7192586e234856e60b19cf51a500`.
  No cleanup candidate or deletion is authorized.
- V3 and V3A1 future authorization/execution namespaces remain absent. No
  phase lock, marker, owner, stream event, teacher query, label, game,
  optimizer step, checkpoint, scientific read, cleanup, or human-session read
  occurred.
  `CONTINUE=false pending separate headroom review`;
  `HOLD=all recovery execution and downstream science`;
  `KILL=historical locks and V1 reuse only`;
  `PROMOTE=false`.

## 2026-07-30 J2A1 V3 Recovery Execution-Surface Readiness

**Decision: `READY_J2A1_V3_RECOVERY_EXECUTION_SURFACE`;
`execution_authorized=false`.**

- The immutable source SHAs are charter
  `674ed0e1c67df0cbc8645a2190a5632ce70c9cddc5922ad0325a9e53d14c481c`,
  runner
  `611dc428a3f940ff1db15ae58e960bab27ab7307c36393bd23a7400e9da12c02`,
  and tests
  `1a1dbb1039b9dd7d57d8d9a88f7cd81dfdd68240bd16b23357df6f5c5eb01df4`.
  Final evidence records py-compile PASS and 364/364 passing tests at nice 10,
  with eight exact spent-V2 chronology/state deselections.
- The surface binds 3,048 byte-authenticated V2 completions and exactly
  11,288 frozen unfinished roots, reuses the sole existing
  reservation/consumption authority, and bars Stage B until the exact
  14,336-root union passes. Readiness performed zero body deserializations,
  teacher queries, labels, games, optimizer steps, checkpoints, reservations,
  consumptions, family/mechanism/fidelity reads, or human-session reads.
- Readiness lock/result/retention file SHAs are
  `ba44650eaead39de45465ff6a785d7a30aaf9c5740294b2516e70354287691ef`,
  `3bac460ad19a32b249b199eec66d6aa7cc9f27be83eb2c4842412868e81ac610`,
  and
  `7f9def1579f2414dcbda7002ee5f7519daa86ae86986206d1a4dbbe7348a701c`.
  Post-seal package and operational audits pass; projected total Stage A wall
  is `42.602824125381694h` point / `50.989066430196644h` conservative and
  peak retained storage is `20.53877067565918 GiB`.
- No scientific authorization, phase lock, marker, materialization, owner,
  collector, or future execution namespace exists.
  `CONTINUE=separate research-lead review`;
  `HOLD=V3 execution and all scientific/downstream work`;
  `KILL=historical locks plus V1 reuse`;
  `PROMOTE=false`.

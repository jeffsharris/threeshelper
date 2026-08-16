# G3 E0 Label/Fit Charter Amendment V3: Open/Resume Orchestration

Date frozen: 2026-07-25

Status: authoritative narrow engineering amendment for one fresh zero-outcome
preflight. E0 execution, E1, labels, fits, predictions, transfer outcomes,
policy evaluation, and promotion remain unauthorized until a separate message.

## Preserved V2 Ready State

The v2 preflight remains immutable READY evidence:

- output:
  `threes_rl/runs/forensics/g3_e0_label_fit_v2`;
- preflight lock file SHA-256:
  `fde44d089b3d59e97e417d101ce1fbfeedc51874de4b74a1bc4ca0583ae62ef7`;
- preflight lock payload SHA-256:
  `c7ec0d0d1feab5dc1fd3564d63d11dd624a8354763661ead45dcb5e87446e2bb`;
- base charter SHA-256:
  `78c7a83601f71de46e0ea53db98023eef12fe16d2f024362d33fd710c82d0591`;
- v2 integrity amendment SHA-256:
  `1b0594d5c0cb55b7c5e11d24c64fa0f82d33952f6e4d4eab8eb5bdf429815fe7`;
- scientific runner SHA-256:
  `19d74a319459d75619f515fd9cdea03a126e1270046fb8e12ae367d43b2cc8b5`;
- record/task/stream manifest file SHA-256:
  `90a4f55ff29f51c0d6ac35375650258188b6961debd6cbcc546382762547d9d5`,
  `087fd68c71421c8402360a1c096b476cb1bf494de7d8c8f025e7e699bf97bd2f`,
  and `e40b7dd3744dd0df04f621034894656568991291c17490e27e8c3a93e189ea05`;
- test evidence SHA-256:
  `8f062d3ef9f36371a2e25465f32c5a3bccd82d507739f20b6eb9b39b42d787e9`.

V2 has no execution-open marker, label database, checkpoint, prediction,
terminal result, consumed stream, label path, fitted model, or opened outcome.
It may not be executed, edited, deleted, or reused as an output directory.

## Localized Orchestration Mismatch

The v2 scientific runner seals its marker inside `execute` and rejects an
already-existing marker. Its CLI records normalized Python argv rather than the
exact authorized shell command. It therefore cannot both:

1. seal and hash a marker before execution starts; and
2. resume under that same pre-existing marker.

This is a pre-stream engineering mismatch. It is not scientific evidence and
does not change any task, stream, feature, model, threshold, or gate.

## Fresh V3 Identity

V3 uses:

- orchestration runner:
  `threes_rl/g3_e0_label_fit_v3.py`;
- zero-outcome preflight:
  `threes_rl/g3_e0_preflight_v3.py`;
- focused tests:
  `tests/test_rl_g3_e0_label_fit_v3.py`;
- test evidence:
  `threes_rl/runs/forensics/g3_e0_label_fit_v3_test_evidence.json`;
- fresh output:
  `threes_rl/runs/forensics/g3_e0_label_fit_v3`.

The three scientific manifests in v3 must be byte-identical copies of the v2
record, task, and stream manifests. Their file and canonical payload hashes
must reproduce exactly. The reserved 57B/58B/59B/60B streams remain
unconsumed.

## Open-Only Contract

The exact future open command is:

`zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python -m threes_rl.g3_e0_label_fit_v3 open --out-dir threes_rl/runs/forensics/g3_e0_label_fit_v3 --preflight-lock threes_rl/runs/forensics/g3_e0_label_fit_v3/preflight_lock.json --jobs 1'`

It must:

1. require the fresh READY v3 preflight, jobs `1`, nice at least `10`, no
   existing marker or terminal, and zero database/model/prediction work;
2. revalidate every bound implementation, test, evidence, manifest, incumbent,
   process, service, disk, dashboard, top-three, and stream-collision lock;
3. atomically write exactly one immutable `G3_E0_EXECUTION_OPENED.json`;
4. bind directly in that marker the exact open and execution commands; base,
   v2, and v3 charter hashes; scientific and orchestration runner hashes;
   focused test/evidence hashes; preflight file and canonical payload hashes;
   record/task/stream file and canonical payload hashes; incumbent identity;
   683 ordinary records, 32 transfer roots, 5,072 paths split
   3,902/944/226; replicates `0,1`; jobs/nice and runtime/storage/service
   bounds; fresh output path; current health; and zero pre-open work;
5. exit successfully without opening a stream or creating any other work
   artifact.

The marker file and canonical payload hashes must be recorded in
`EXPERIMENT_LOG.md` before the execution command starts.

## Execute/Resume Contract

The exact future execution command is:

`zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python -m threes_rl.g3_e0_label_fit_v3 execute --out-dir threes_rl/runs/forensics/g3_e0_label_fit_v3 --preflight-lock threes_rl/runs/forensics/g3_e0_label_fit_v3/preflight_lock.json --jobs 1'`

Execution must reject a missing, noncanonical, mismatched, or wrong-directory
marker before any task, stream, database, model, or prediction access. It must
revalidate the marker against the current immutable lock, commands, code,
tests, manifests, incumbent, and operational bounds.

Interruption may resume only with the exact same marker, lock, output, command,
jobs, and hashes. Completed label rows are hash-verified and skipped. A sealed
checkpoint is loaded and verified rather than refit. A sealed transfer
prediction artifact is loaded and verified rather than recomputed. Transfer
labels remain inaccessible until the checkpoint and prediction seals exist.
A terminal result is immutable and rejects further execution.

## Unchanged Scientific Contract

V3 delegates all simulator rollouts, feature construction, task ordering,
event/censor arithmetic, weighting, fitting, calibration, prediction,
bootstrap, activity checks, scientific decisions, and terminal evidence to the
frozen scientific runner. The following remain byte-identical and semantically
unchanged:

- 683 ordinary records, 352 ordinary ancestries, and 32 transfer roots;
- train/development/transfer assignments and family caps;
- 5,072 all-legal-action paths and replicates `0,1`;
- shared CRN tapes and 57B/58B/59B/60B stream IDs;
- G2 64-feature schema and fixed L2 logistic model;
- development-only calibration and all READY/HOLD/KILL thresholds;
- frozen incumbent continuation policy;
- checkpoint-before-transfer barrier;
- one worker, nice `10`, 18-hour, 4-GiB, 100-GiB hard and 120-GiB target
  constraints;
- non-promotable E0 status and held E1/policy/dashboard work.

No task, label, model, action, outcome, score, or dashboard datum may be opened
by the v3 preflight.

## Zero-Outcome Preflight Gate

The authorized v3 preflight may only implement and test orchestration, copy and
hash the frozen manifests, reconstruct records for integrity, audit unconsumed
streams and operational health, and seal one of:

- `READY_G3_E0_V3_EXECUTION`;
- `HOLD_G3_E0_V3_ORCHESTRATION`;
- `KILL_G3_E0_V3_INTEGRITY`.

READY requires focused tests for open-only zero-work behavior, exact command
binding, missing/mismatched marker rejection, same-marker resume, immutable
terminal behavior, and no transfer access before checkpoint, plus the frozen
G3/G2/G1/S3/provenance/service regressions. READY authorizes only a later
separate execution message.

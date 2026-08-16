# G3 E0 Label/Fit Charter Amendment V4: Collision Source Classification

Date frozen: 2026-07-25

Status: authoritative narrow engineering amendment for one fresh zero-outcome
preflight. E0 execution, E1, labels, fits, predictions, transfer outcomes,
policy evaluation, and promotion remain unauthorized until a separate message.

## Preserved V3 Hold

The v3 no-outcome preflight remains immutable READY evidence:

- preflight lock file SHA-256:
  `ac4f4e7478ea51c088925528fa62dd3020b412f9898d9f87857e70dd3ea657b3`;
- preflight lock payload SHA-256:
  `25f3686d05057935ca8a57f9e9a5668249323ff0a61768b2f238c4603dc05af7`;
- v3 orchestration amendment SHA-256:
  `d201e4a187e77a643d555499f76d8dca6c76eab584c1f3d5efaced8de98d38ab`;
- v3 runner SHA-256:
  `db5125498812f0561368b97a70c51dc9ebcff3fb556cc84db16c3ee13ba48392`;
- v3 test evidence file/payload SHA-256:
  `8c7c6de84393967e229ace31446f450a8357f627a61dbf5e6193000096bd18d5` /
  `e2c7e0129db7a6814d667a4807b7f8ac5fc694ece1543a2b495982d6fc69ba21`.

The exact v3 open command failed before marker creation because its full
historical source-list SHA included required live dashboard summaries. Source
count remained `8,927`; all requested stream collision sets remained empty;
only the byte-level source-list SHA changed after the dashboard watcher rewrote
two files. The immutable v3 HOLD file/payload SHA-256 are
`f1f69132787ceeea5fa32ff964287edda791d7c206b3460df949de469417378d` /
`a2d06ce98897c1c719c9999d952ff5186efa9ccc9ca273eab46a8c364c253e6d`.

V3 created no execution marker, consumed stream, label database/path/value,
model, checkpoint, prediction, transfer outcome, policy outcome, score
inspection, incumbent change, or dashboard change. It may not be retried,
edited, or used as an execution output.

## Sole V4 Change

V4 changes only the historical collision source contract. All scientific
records, tasks, streams, simulator semantics, model, calibration, gates,
checkpoint barrier, commands sequence, and operational limits remain
unchanged.

Every regular `.json`, `.jsonl`, or `.csv` under `threes_rl/runs` from which
the existing history scanner extracts at least one logical seed, seed alias,
deck stream, slot stream, or policy stream is classified exactly once.

### Immutable External Sources

Every collision-bearing source outside the exact inherited/self namespaces and
outside the two live paths below is `immutable_external`.

The preflight seals, per source:

- normalized workspace-relative path;
- SHA-256;
- byte size;
- extracted field names and value counts;
- classification `immutable_external`.

The ordered row list, canonical payload, and file are hashed. At open time,
all immutable rows must reproduce exactly. A missing source, new source,
changed path, changed SHA, changed byte size, or changed extracted-count row
fails closed.

### Live Generated Sources

Exactly these paths are `live_generated_dashboard`:

- `threes_rl/runs/dashboard/dashboard.json`;
- `threes_rl/runs/dashboard/score_trends.json`.

No other basename, sibling, alias, symlink, or dashboard file is live by
implication. The sealed contract binds the two normalized paths, classification
reason, and exact count `2`. Their byte hashes and mtimes are descriptive only
because the required dashboard watcher rewrites them. At preflight and open,
both files must exist, remain regular files under those exact paths, be scanned
for current collision-bearing values, and contribute to the actual requested
stream collision test. A missing live path or any new unclassified source
fails closed.

### Inherited Internal Reservation Namespaces

Exact inherited/self directories are classified by normalized path and reason
as `inherited_internal_reservation`. They contain immutable copies or locks of
the same unconsumed 57B/58B/59B/60B reservation and are excluded only from the
external-collision union:

- G3 bootstrap preflight v1;
- G3 bootstrap preflight v2;
- E0 preflight v1;
- E0 preflight v2;
- E0 preflight v3;
- fresh E0 preflight v4.

Directory identities and reasons are sealed. No broad `forensics`, `runs`,
dashboard, prefix, glob, or symlink exclusion is allowed. Internal files are
still scanned and reported by count/classification, but intentional copies of
the requested reservation do not count as historical collisions.

## Requested Stream Set

The requested set is derived only from the byte-identical frozen v3 task
manifest. The contract seals:

- all `5,072` ordered task rows;
- each task key and its logical/deck/slot/policy IDs;
- ordered-row canonical SHA-256;
- unique value count and canonical SHA-256 for each stream kind;
- exact task, partition, and replicate counts.

At preflight and open, current values from every
`immutable_external` and `live_generated_dashboard` source are unioned. Seed
aliases are included exactly as in the existing audit. The intersection with
every requested stream kind must be empty. Source-inventory validity and
zero-collision validity are separate checks; both must pass.

## Fresh V4 Identity

V4 uses:

- orchestration runner:
  `threes_rl/g3_e0_label_fit_v4.py`;
- no-outcome preflight:
  `threes_rl/g3_e0_preflight_v4.py`;
- focused tests:
  `tests/test_rl_g3_e0_label_fit_v4.py`;
- test evidence:
  `threes_rl/runs/forensics/g3_e0_label_fit_v4_test_evidence.json`;
- fresh output:
  `threes_rl/runs/forensics/g3_e0_label_fit_v4`;
- classified collision manifest:
  `E0_COLLISION_SOURCE_MANIFEST.json`.

The record, task, and stream manifests must remain byte-identical to v3 and v2.
The 57B/58B/59B/60B streams remain unconsumed.

The exact future open command is:

`zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python -m threes_rl.g3_e0_label_fit_v4 open --out-dir threes_rl/runs/forensics/g3_e0_label_fit_v4 --preflight-lock threes_rl/runs/forensics/g3_e0_label_fit_v4/preflight_lock.json --jobs 1'`

The exact future execution command is:

`zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python -m threes_rl.g3_e0_label_fit_v4 execute --out-dir threes_rl/runs/forensics/g3_e0_label_fit_v4 --preflight-lock threes_rl/runs/forensics/g3_e0_label_fit_v4/preflight_lock.json --jobs 1'`

V4 inherits the explicit v3 open-only marker and resume state machine. No
runtime monkeypatching is permitted.

## Unchanged Scientific And Operational Contract

The following remain exact:

- frozen scientific runner SHA-256
  `19d74a319459d75619f515fd9cdea03a126e1270046fb8e12ae367d43b2cc8b5`;
- `683` ordinary records, `32` transfer roots, and `384` unique roots;
- `5,072` paths split `3,902/944/226`, replicates `0-1`;
- G2 64-feature L2 logistic model and development-only calibration;
- frozen incumbent continuation and all READY/HOLD/KILL thresholds;
- checkpoint-before-transfer and transfer activity floor `6/32`;
- one worker, nice `10`, 18-hour, 4-GiB, 100-GiB hard and 120-GiB target
  limits;
- required healthy dashboard port `8765`, advisor port `8770`, dashboard
  record `263670`, and protected top three `263670/261369/258561`;
- non-promotable E0 and held downstream work.

## Zero-Outcome V4 Gate

The authorized preflight may implement and test classification, copy and hash
the frozen manifests, restore sources, audit current collisions and services,
and seal exactly one:

- `READY_G3_E0_V4_EXECUTION`;
- `HOLD_G3_E0_V4_ORCHESTRATION`;
- `KILL_G3_E0_V4_INTEGRITY`.

Focused adversarial tests must prove:

1. content-only rewrites of the two exact live dashboard files do not invalidate
   an otherwise identical zero-collision inventory;
2. an immutable source mutation fails;
3. a new unclassified collision-bearing file fails;
4. an actual requested stream collision in a live or immutable source fails;
5. missing, aliased, symlinked, or reclassified live paths fail;
6. open-only, marker, resume, terminal, and transfer-barrier behavior remains
   unchanged.

READY requires a separate execution authorization. The v4 preflight may not
create a marker, consume a stream, generate a path or label, fit a model, make
a prediction, or inspect an outcome.

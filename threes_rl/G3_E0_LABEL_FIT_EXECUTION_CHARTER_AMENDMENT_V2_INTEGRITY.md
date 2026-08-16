# G3 E0 Label/Fit Charter Amendment V2: Compact-Manifest Integrity

Date frozen: 2026-07-25

Status: authoritative narrow engineering amendment for one new no-outcome
preflight. The base E0 charter remains scientifically unchanged. E0 execution,
E1, labels, fits, predictions, transfer outcomes, policy evaluation, and
promotion remain unauthorized.

## Preserved V1 Failure

The first E0 preflight is immutable and spent:

- output directory:
  `threes_rl/runs/forensics/g3_e0_label_fit_v1`;
- preflight lock file SHA-256:
  `73eee861d1ba964e4e9c2d24bffab29835f25fe2d435e7f9c665868a046dfa94`;
- preflight lock canonical payload SHA-256:
  `af4dcda1a2346e32450e7d37c6b989ae4b8ff139c625aca380d7fe424ceae45c`;
- decision: `KILL_G3_E0_PREFLIGHT_INTEGRITY`;
- base charter SHA-256:
  `78c7a83601f71de46e0ea53db98023eef12fe16d2f024362d33fd710c82d0591`;
- runner SHA-256:
  `19d74a319459d75619f515fd9cdea03a126e1270046fb8e12ae367d43b2cc8b5`;
- v1 preflight implementation SHA-256:
  `bb1c09b71060c1be1b6fb4bcfd03eb762950ea34d537b77f181f578c0b5cc627`;
- focused test SHA-256:
  `cd611547e86c6a65acc570440f4c94616549f2a419c439e3df669712a64687b1`;
- test-evidence SHA-256:
  `0d4511eb4422f79304fa2dc17f54112e1f8cf3f0769fa086296c2ded59196ca9`.

V1 generated zero games, consumed zero streams, generated zero label paths,
opened zero label values or outcomes, fit zero scientific models, made zero
predictions, and changed neither incumbent nor dashboard. Its output directory
may not be edited, deleted, reused, or executed.

## Localized False Positive

The immutable G3 record manifest is intentionally compact. Its ordinary records
bind `source_replay`, `source_replay_sha256`, `source_frame_index`,
`root_cluster`, `state_sha1`, starter, legal actions, and frozen feature digest,
but do not embed a `state` object.

The v1 E0 preflight incorrectly passed these compact rows to the earlier
`validate_ordinary_records` adapter, which expects the pre-compaction G2 root
manifest's embedded `state`. It therefore reported `KeyError: 'state'` for all
683 ordinary records. Independently in the same sealed preflight,
source-pointer restoration reproduced all 683 frozen feature digests with zero
failure. All transfer, stream, schema, incumbent, cost, storage, process,
service, disk, and test gates passed.

This is an engineering schema-adapter false positive, not scientific evidence
and not permission to reinterpret or overwrite v1.

## Sole V2 Integrity Change

V2 validates each ordinary compact record directly from its immutable source
pointer:

1. hash the exact source replay and match `source_replay_sha256`;
2. require genuine direct normal-start provenance and reset invariants;
3. require canonical ancestry to equal `root_cluster`;
4. select exactly one physical replay frame matching
   `source_frame_index`;
5. restore the complete simulator state from that frame;
6. require the restored `state_sha1`, starter, legal action IDs/names, and
   frozen 64-feature digest to match the compact record;
7. require feature extraction to leave state, deck RNG, and slot RNG
   unchanged.

There is no embedded-state comparison because the authoritative compact record
contains no embedded state. Transfer validation remains unchanged.

The new preflight implementation and tests use fresh v2 identities and output:

- implementation: `threes_rl/g3_e0_preflight_v2.py`;
- focused tests: `tests/test_rl_g3_e0_preflight_v2.py`;
- test evidence:
  `threes_rl/runs/forensics/g3_e0_label_fit_v2_test_evidence.json`;
- output: `threes_rl/runs/forensics/g3_e0_label_fit_v2`.

The v2 collision audit excludes only the exact immutable, unconsumed G3-v1,
G3-v2, and failed E0-v1 preflight directories. Every excluded file remains
hash-bound and separately reported. Any requested stream match elsewhere fails.

## Unchanged Contract

Every scientific and operational choice in the base charter remains exact:

- same 683 ordinary records, 352 ordinary ancestries, and 32 transfer roots;
- same train/development/transfer partitions and family assignments;
- same all-legal-action breadth-first E0 paths and replicates 0 and 1;
- same unconsumed 57B/58B/59B/60B shared CRN reservation;
- same event/censoring arithmetic, G2 64-feature schema, L2 logistic model,
  calibration, weights, bootstrap, activity, and READY/HOLD/KILL gates;
- same frozen incumbent continuation policy and artifact hashes;
- same 5,072-path, 11.895-hour, 44.8-MiB projection;
- same one-worker nice-10, 18-hour, 4-GiB, 100-GiB hard and 120-GiB target
  constraints;
- same non-promotable E0 status and held E1/policy/dashboard work.

If v2 integrity fails for any reason beyond this exact compact-manifest adapter,
it seals and stops. No further source, model, threshold, cost, or exclusion
amendment is authorized by this file.

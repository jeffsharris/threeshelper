# G3 Scale-Transfer Bootstrap Charter Amendment V2 Integrity

Date frozen: 2026-07-25

Status: authoritative outcome-free amendment for the separate
`g3_scale_transfer_bootstrap_preflight_v2` integrity preflight.

The following G3-v1 evidence remains authoritative, spent, and byte-for-byte
immutable:

- charter SHA-256
  `e216aa50737afee0d439e060cc9b1e1f24d2f552af4c3f0c8944470ff7a45fc1`;
- stream amendment A1 SHA-256
  `baba72003934ef55a48383704e4c6b5738787d561a0d81f0ea09383f64122b94`;
- implementation SHA-256
  `27ac9cc6a5d4ee7449650ef0d886395233bea1e77ff7a7213e9142a614215234`;
- tests SHA-256
  `d9c60a840d95c85cc180128641b09920d2e7214f09ccb00f711a8e0c061aa14a`;
- test-evidence SHA-256
  `a8f6a1768a293395c9a5583857ef77188ff41421a95a3cfabdfa62315a3dbd21`;
- sealed preflight file SHA-256
  `0cd19d5a3d390df4f9d0165e72f20a130799ceafb0cb408e5274b8924a82d77a`.

Its `KILL_G3_PREFLIGHT_INTEGRITY` decision is never rerun, overwritten, or
reinterpreted. The observed failure was an outcome-free engineering false
positive: the v1 token search reported files inside the exact immutable G2
transfer-source directory as external contamination because its exclusion
glob did not match paths rooted at `threes_rl/runs`.

No label value, score, chosen action, future milestone, fit, transfer outcome,
policy outcome, or continuation was opened before this amendment.

## Sole Integrity Semantic Change

V2 inherits every G2/G3 representation, partition, model, root, family, label,
CRN, endpoint, power, cost, and transfer-diagnostic rule from the original
charter and A1. It changes only classification of token matches in the
transfer-panel untouchedness audit.

The scan root is the exact resolved directory:

`threes_rl/runs`

Only these exact namespaces may be classified as non-external:

1. immutable G2 panel-input namespace:
   `threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1`;
2. immutable G3-v2 output namespace:
   `threes_rl/runs/forensics/g3_scale_transfer_bootstrap_preflight_v2`;
3. immutable G3-v2 staging namespace:
   `threes_rl/runs/forensics/g3_scale_transfer_bootstrap_preflight_v2.staging`.

There is no wildcard, prefix, sibling, or broad `forensics/**` exclusion.
Classification is fail-closed:

- a match is internal only when its absolute lexical path is normalized,
  contains no `.` or `..` alias segment, has no symlink-mediated component,
  and is strictly inside one of the exact namespaces above;
- a prefix lookalike, sibling directory, outside copy, symlink alias, relative
  alias, or normalization escape is external contamination even when it
  resolves to an internal file;
- every root or state token match outside the exact namespaces fails the
  untouchedness gate;
- excluded G2-input matches, excluded G3-self matches, and true external
  matches are reported separately;
- every excluded regular file that matches is hashed;
- every replay and state file used by the 32-root panel, plus the sealed G2
  acquisition result that binds those files, is independently path-checked,
  byte-size checked where recorded, SHA-256 checked, and listed in an
  immutable panel-input binding table.

V2 reuses the validated v1 record and stream manifests only after their file
and canonical-payload hashes reproduce exactly:

- record-manifest file SHA-256
  `938e903f8d2fefb072af84ac19baf4977e4f4d93bf72e8af7acc174b6974b9ec`;
- record-manifest canonical payload SHA-256
  `a78e2fd51ee20a7aeb23c71d9930c33561844357920f4808eeeaff653d49f759`;
- stream-manifest file SHA-256
  `bdbe562167f304327e52f0593f0958753e8afa949a7b38e15b357492faea5744`;
- stream-manifest canonical payload SHA-256
  `c2afc3c6fa26c1106a480c58189d9a9b4f9dcf99ac8b506d890ff3c330278caa`.

V2 does not regenerate, alter, or replace either manifest.

## Nonadaptive Staged-Cost Governance

The final scientific label contract remains exactly eight CRN replicates for
every frozen record and every legal first-action arm. This amendment only
decomposes that fixed work for possible future execution governance:

- `E0`: replicates `0` and `1` for every ordinary and transfer record and
  every legal action, root breadth first;
- `E1`: replicates `2` through `7` for every same record and legal action,
  completing the frozen total of eight.

The v1 manifest fixes `20,288` total h40 action paths. Therefore:

- E0 contains exactly `5,072` paths:
  `3,902 train`, `944 development`, `226 transfer_diagnostic`;
- E1 contains exactly `15,216` paths:
  `11,706 train`, `2,832 development`, `678 transfer_diagnostic`.

The v2 preflight recomputes stage path counts from the immutable stream
manifest and fails if they differ. It reports one-worker nice-10 time using
the unchanged conservative seconds-per-path estimate. Storage uses the
unchanged selected bytes-per-path and multiplier: E0 receives the single
fixed base allocation; E1 is the additional path payload needed to complete
eight replicates. E0 plus E1 must equal the unchanged final cost projection.

E0 and E1 are both unauthorized by this preflight. Any future E0 execution
requires a separate immutable charter. A provisional E0 fit, if separately
authorized, is explicitly non-promotable. E1 may open only under a prospective
E0 gate frozen before E0 outcomes. That gate may authorize all of E1 or none
of it; it may not select roots, actions, families, scales, or records using E0
outcomes.

## V2 Outcome-Free Decision

V2 recomputes only the corrected external untouchedness audit, immutable
panel-input bindings, stream collisions, disk, services, process health,
dashboard/top-three truth, label-coverage metadata, staged/final cost, and
prospective power. It reads no label or score/action outcome and fits nothing.

The separate output is:

`threes_rl/runs/forensics/g3_scale_transfer_bootstrap_preflight_v2`

The preflight seals exactly one:

- `READY_G3_V2_BOOTSTRAP_LABELS`;
- `HOLD_G3_V2_LABEL_COVERAGE_OR_COST`;
- `KILL_G3_V2_PREFLIGHT_INTEGRITY`.

READY authorizes only a later separately frozen E0 label-execution charter.
It does not authorize E0 or E1 labels, fitting, transfer outcomes, reranker
construction, normal-start evaluation, C2, human-training-ground work,
incumbent change, dashboard change, or promotion.

# G1-R QD V2 Admission Execution Charter

Date frozen: 2026-07-25

Status: implementation and descriptor-exactness tests only. This charter does
not authorize v2 archive construction, lock preparation, candidate action
evaluation, reference signatures, pairwise comparison, timing, acquisition,
game generation, labels, model fitting, continuation outcomes, score
inspection, incumbent changes, or dashboard changes.

Authoritative proposal:
`G1R_QD_V2_TERMINAL_SCHEMA_PROPOSAL.md`, SHA-256
`9a95f0b9a60c1825b9f083321c0bfd4048290e2e60893bdff5e9b0bac98789cf`.

## Immutable V1 Boundary

QD-v1 is `KILL_QD_V1_EXECUTION` and may never be modified or rerun. Preserve:

- execution-lock file SHA-256
  `ed0a89e77baadfaf685c09960b00701bf61e64f6c434941fa2f26cdeb56eb6e2`;
- opened-marker file SHA-256
  `f1faadcf2152b28b0254f36402de4568be4eb056c7dd56c52bbcd51c17d51f6e`;
- terminal-HOLD file SHA-256
  `205229ce77a34b68ff3fdc31ee0bb83bc917671ae3f9efdb5f5eeb91c0b7068b`;
- terminal-HOLD payload SHA-256
  `6bc74c736269ba6d426c4c7ec6ec4307ef60cfbd86d300a7173714bd5424cba0`.

The complete promoted and failed-staging v1 directories are read-only evidence.
No v2 path may point inside them.

## V2-Owned Surface

- runner: `threes_rl/g1r_qd_admission_v2.py`;
- focused tests: `tests/test_rl_g1r_qd_admission_v2.py`;
- descriptor: `g1r_qd_descriptor_v2_terminal_legal0`;
- archive: `g1r_qd_archive_v2_terminal_legal0`;
- source manifest: `g1r_qd_archive_sources_v2_terminal_legal0`;
- policy: `g1r_qd_policy_v2_terminal_legal0`;
- family: `g1r_qd_static_archive_oneply_v2_terminal_schema`;
- runner/lock: `g1r_qd_admission_v2_terminal_schema`;
- future output:
  `threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema`.

The future output directory must remain absent during this charter's authorized
work.

## Sole Semantic Delta

Descriptor coordinate 8 accepts exactly legal-action counts `0..4`. Its mixed
distance is absolute difference divided by `4`. All other descriptor formulas,
categorical/ordinal roles, objective, tie rules, source selection, archive
semantics, parent value, reference gates, timing, streams, provenance, service,
storage, process, staging, one-shot sealing, and no-label rules are inherited
literally from QD-v1.

## Shared Immutable Inputs

Unchanged shared inputs retain and re-bind their actual hashes. They are never
copied or perturbed to manufacture new identities:

- pilot-v1 panel payload SHA-256
  `b8862aa3c8eaf6278fc078fb3e03aa7222a01930673cfee497738c74e81eff9d`;
- pilot-v1 preflight file SHA-256
  `f78288b3f47bda6aa6d15c2157fd79f7b3d0685f0367d8b9964f5dc73981ea91`;
- A2 inventory SHA-256
  `8604778696164fdabd5ab653c933b0b543ca1d20a8fde1d78b6e7da2994d794a`;
- parent checkpoint manifest SHA-256
  `005fe0d7d1fa8d46f2ba78a99a826d8f5bde5ed0413efedcdbbd5f1f845fe8d3`;
- incumbent policy-file SHA-256
  `d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4`;
- reserved, unused logical/deck/slot/policy namespaces beginning at
  `45B/46B/47B/48B`.

Simulator/evaluation source hashes must be read and reported after tests. No
test may alter a shared input.

## Descriptor-Only Test Gate

Before any future v2 lock, all of the following must pass:

1. `py_compile` for the v2 runner and focused test;
2. crafted afterstates with legal counts `0,1,2,3,4`;
3. domain, normalization, schema-hash, archive, policy, and cross-version load
   rejection;
4. every exact visible-preview insertion outcome for all 64 frozen panel
   states;
5. every exact visible-preview insertion outcome for all 489 deterministic
   selected A2 archive roots;
6. exact equality between descriptor coordinate 8 and
   `len(sim.legal_actions(outcome_state))`;
7. nonzero terminal and live outcome coverage, reported by source set and legal
   count;
8. root state/context no-mutation and deterministic repeated enumeration;
9. inherited archive, save/load, staging, one-shot, signature-lock, process,
   and latency-report unit tests;
10. the broader G1/S3 regression set.

Enumeration may record only descriptor/exactness fields needed for the test. It
must not call QD action ranking, parent values, reference policies, timing,
scores, or continuations, and must not create a persistent outcome artifact.

## Stop

After tests, report implementation/test/charter/schema hashes, test counts,
coverage by source set and legal count, unchanged v1 hashes, and absent v2
output. Then HOLD for oversight. No routine passing test authorizes lock
preparation or action evaluation.

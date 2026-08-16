# J2A1 V3A1 Post-Seal Reproducibility Amendment

Status: outcome-free chronology and launch-headroom readiness only.
No scientific authorization, phase lock, marker, materialization, owner,
stream event, collector, teacher query, label, game, optimizer step,
checkpoint, scientific read, cleanup, or downstream work is authorized.

## 1. Parent HOLD and immutable scope

`READY_J2A1_V3_RECOVERY_EXECUTION_SURFACE` is preserved byte-for-byte but is
spent and operationally held for execution reuse. Its sealed focused test
suite reproduces `41 passed / 1 failed` after readiness sealing. The sole
failure is
`test_zero_work_and_future_namespaces_are_absent`: it calls the pre-prepare
`audit_zero_work(...)` against the now-populated readiness directory.

This is a chronology/orchestration defect only. It is not scientific evidence
and does not alter any V2/V3 authority, stream, wall-time, storage, source,
metric, gate, or decision semantics.

The immutable V3 source identities are:

- charter
  `674ed0e1c67df0cbc8645a2190a5632ce70c9cddc5922ad0325a9e53d14c481c`;
- runner
  `611dc428a3f940ff1db15ae58e960bab27ab7307c36393bd23a7400e9da12c02`;
- tests
  `1a1dbb1039b9dd7d57d8d9a88f7cd81dfdd68240bd16b23357df6f5c5eb01df4`.

All nine files in
`threes_rl/runs/forensics/`
`j2a1_distillation_fidelity_recovery_execution_surface_readiness_v3`
remain immutable:

- test evidence file/payload
  `9a46fe4abeb4cac94302ae2ad746d83ce2da9d5748a377db5f90ecd6d0e83b99` /
  `66d77e3cc0d217d8a1059736913271211512526e8d149e672251de24c0efa0ae`;
- input bindings
  `b2343a23023441331f86ded68426cf5babe3e73bc7f860e9c029faafbeb46e72` /
  `aa1a427e240d769cfd8dee530d439c8ed0ad235f4814c57761488ba71bd7a58f`;
- authority audit
  `fbdd57e403c614c916e3317aefba5ce3ae37c3e03aaa61b7189f307c7ed84069` /
  `5a018f630fc2ff7d85911fe00138e461c67c8d29b54f04dfb7d340ad1317294d`;
- schema
  `d3dd83fef73131b84b8ea7668e73d94e5b3dab564c7312f4840bf822f9ed65c4` /
  `dbdaeb5f16fb14c7016e06306967931265b9d25c23b2eff5001e0a9ca5704e9b`;
- projection
  `39dba009a461ae512121b02a206afbd89f058b38df93786831a7831533f6df4c` /
  `c39647126dd9aac79c813f6e61515ef49f0e459785d3052be39624de2f9d250e`;
- state-machine audit
  `035c61e79e9d191684d04885c19f4cefe595c34070b24918186fef96e5ef4959` /
  `2bc15d93284c19fe140148f0df26c933961495fa22fba1e9361a4fc6dc637b0a`;
- readiness lock
  `ba44650eaead39de45465ff6a785d7a30aaf9c5740294b2516e70354287691ef` /
  `8d4d87baa92ea04730435d2798d9b9bed0088bbd757443913d3fc9eaafb0bea3`;
- readiness result
  `3bac460ad19a32b249b199eec66d6aa7cc9f27be83eb2c4842412868e81ac610` /
  `c2626a14d8b86613d05c0934e4735171cc3242f7841308914a669c3666cb7bb9`;
- retention
  `7f9def1579f2414dcbda7002ee5f7519daa86ae86986206d1a4dbbe7348a701c` /
  `55dde2fe501267adb635d48849144ffca046d9e40f29e81ceb45acfcb488eeb7`.

V3A1 binds the accepted V3 preflight, all V2 evidence, the exact
3,048/11,288 authority split, completed/unfinished identities, and the sole
existing reservation/consumption authority through the immutable V3 package.
It never opens a root body.

## 2. Chronology repair

The pre-prepare audit has exactly one valid readiness state: the V3A1
readiness namespace does not exist. It also requires all future V3 and V3A1
authorization/execution namespaces absent and every scientific-work counter
zero.

The post-seal audit has exactly one valid parent state: the V3 readiness
namespace contains the immutable nine-file package listed in Section 1. It:

1. requires the exact file set with no symlink or extra file;
2. verifies every file SHA and self-hashed payload;
3. verifies source identities, predecessor identities, and the canonical
   retention inventory;
4. re-runs the V3 package loader;
5. requires all future V3 and V3A1 authorization/execution namespaces absent;
6. requires every V3 and V3A1 scientific-work counter exactly zero.

The V3A1 focused suite tests the pre-prepare audit on a fresh fixture namespace
and the post-seal audit against the authoritative V3 package. It never calls
the parent pre-prepare audit against a sealed package. The immutable V3
focused failure remains bound as parent HOLD evidence rather than edited,
deselected, or reinterpreted.

## 3. Launch headroom

The frozen combined peak projection is `22,053,337,088` bytes. The already
retained V2 execution footprint is exactly `1,782,523,714` bytes and is
already reflected in current free disk. Therefore the prospective incremental
charge is exactly:

`22,053,337,088 - 1,782,523,714 = 20,270,813,374 bytes`.

At readiness sealing, V3A1 records current free bytes from the repository
filesystem and computes:

- `projected_peak_free_bytes = current_free_bytes - 20,270,813,374`;
- `hard_floor_bytes = 100 * 2^30`;
- `projected_floor_cushion_bytes =
  projected_peak_free_bytes - hard_floor_bytes`.

Launch admission requires both:

1. `projected_peak_free_bytes >= 100 GiB`;
2. `projected_floor_cushion_bytes >= 5 GiB`.

The second rule is a mandatory launch cushion, not a descriptive target.
No rounding, current-free-only check, ideal scaling, cleanup assumption, or
double-counting of the V2 footprint is allowed.

If the exact calculation is unavailable, projected free is below 100 GiB, or
the cushion is below 5 GiB, V3A1 seals
`HOLD_J2A1_V3A1_RECOVERY_EXECUTION_HEADROOM`. It also seals a create-once
review proposal containing exact required bytes, protected paths, and zero
authorized deletions. It performs no cleanup. Protected evidence may be
deleted or moved only under a later separately reviewed manifest.

## 4. Separate files and namespaces

The V3A1 source files are:

- `threes_rl/`
  `J2A1_DISTILLATION_FIDELITY_V3A1_POSTSEAL_REPRODUCIBILITY_AMENDMENT.md`;
- `threes_rl/`
  `j2a1_distillation_fidelity_recovery_execution_surface_v3a1.py`;
- `tests/`
  `test_rl_j2a1_distillation_fidelity_recovery_execution_surface_v3a1.py`.

The readiness namespace is
`threes_rl/runs/forensics/`
`j2a1_distillation_fidelity_recovery_execution_surface_readiness_v3a1`.
Future V3A1 authorization and execution namespaces are
`j2a1_distillation_fidelity_recovery_authorization_v3a1` and
`j2a1_distillation_fidelity_recovery_v3a1`; both remain absent.

The public CLI is exactly:

1. `audit-pre-prepare`;
2. `audit-post-seal`;
3. `write-test-evidence`;
4. `prepare-readiness`.

There is no V3A1 authorization, phase-lock, open, materialize, execute,
cleanup, PPO, development, confirmation, or promotion command.

## 5. Readiness artifacts and decision

The create-once V3A1 readiness package contains exactly:

1. test evidence;
2. parent V3 HOLD binding;
3. immutable input bindings;
4. post-seal chronology audit;
5. launch-headroom audit;
6. cleanup/headroom review proposal;
7. readiness schema;
8. readiness lock;
9. readiness result;
10. retention.

Source, parent, package, retained-inventory, future-namespace, service,
one-heavy-job, nice-10, disk, top-three, and human-opacity checks are binding.
All scientific counters remain zero.

Readiness precedence is:

1. immutable identity/integrity failure:
   `KILL_J2A1_V3A1_RECOVERY_EXECUTION_SURFACE_INTEGRITY`;
2. exact headroom calculation unavailable or either headroom gate misses:
   `HOLD_J2A1_V3A1_RECOVERY_EXECUTION_HEADROOM`;
3. all checks pass:
   `READY_J2A1_V3A1_RECOVERY_EXECUTION_SURFACE`.

READY would permit only separate research-lead review. This amendment itself
never authorizes execution.

`CONTINUE=V3A1 outcome-free reproducibility/headroom readiness`;
`HOLD=all V3/V3A1 execution, inherited scientific stages, PPO, development,
confirmation, promotion, alternate branches, and human training`;
`KILL=historical locks and V1 execution reuse`;
`PROMOTE=false`.

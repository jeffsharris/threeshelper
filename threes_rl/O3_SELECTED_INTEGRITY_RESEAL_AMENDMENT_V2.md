# O3 Selected-Root Integrity Reseal Amendment V2

Status: authorized integrity repair only. The immutable
`o3_event_acquisition_recovery_v1` terminal decision remains
`HOLD_O3_ACQUISITION_RECOVERY_INTEGRITY`.

## Scope

This amendment permits one separately namespaced, no-recompute integrity
envelope. It may read only these four existing JSON artifacts:

- `O3_RECOVERY_UNION_MANIFEST.json`
- `O3_RECOVERY_SUPPORT_SCAN.json`
- `O3_RECOVERY_SELECTED_ROOTS.json`
- `O3_RECOVERY_RESULT.json`

It may not read a replay, rerun support extraction or allocation, generate a
stream, label, rollout, model, action, score, max-tile result, or policy
outcome. The original recovery directory is immutable.

## Frozen Input Identities

| Artifact | File SHA-256 | Embedded payload SHA-256 |
| --- | --- | --- |
| union | `02ea2c5be8823de775f56b7267f9c8371d26efc53897115b25733f8ef4527311` | `cec88701a1754f1064d639dae09cd6856ee18ce9399865338ebed7107f672d94` |
| support | `4c71513e6a3a2778bb8d1db0ba08f8ff5a1f0d6edc82ee1208b7458593059d27` | `27ae3a6aca5f1de71ee18df193c0663a83579d3aeba65cd864065cfff594e25a` |
| selected | `9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049` | pre-serialization `c6c8b1a35cc63f4c1c1fdc98579f1ae0859a84c5eef7203306000223ac9c61a5`; post-JSON canonical body `d9600cf420d947826c812b88225633b78a889f94f94ce39270dd71bc11b12f0e` |
| terminal HOLD | `962da52b83b8746c006a9ef5fbe1fdd34f43e9c7bf97d9b6ff48f2a42019c23a` | `a679d512d6ce44bf5fd4ecd8249d15625c59f342e64796a6d5eb894396224ad0` |

The union must remain a passing exact 20,500-root union with 4,100 roots per
family, role counts 5,020/1,675/13,805, unique ancestries and replay hashes,
and zero role or stream drift. The support audit must remain passing with
12,922 candidates from 7,607 roots. The selected payload must remain passing,
have no deficits, and contain exactly 96 train, 32 development, and 192
untouched-mechanism roots with all five-family and family-cap checks passing.
The terminal result must retain its exact HOLD decision and exact selected-file
self-hash error.

## Exhaustive Serialization Proof

After removing `selected_payload_sha256`, the post-JSON body must have
numeric-string dictionary keys at exactly these six paths and nowhere else:

- `per_role.train.target_counts`
- `per_role.train.descriptive_stage_counts`
- `per_role.development.target_counts`
- `per_role.development.descriptive_stage_counts`
- `per_role.untouched_mechanism.target_counts`
- `per_role.untouched_mechanism.descriptive_stage_counts`

Every key at each path must be a base-10 nonnegative integer string.
Converting only those keys back to integers must reproduce
`c6c8b1a35cc63f4c1c1fdc98579f1ae0859a84c5eef7203306000223ac9c61a5`
exactly. The untouched post-JSON body must reproduce
`d9600cf420d947826c812b88225633b78a889f94f94ce39270dd71bc11b12f0e`.
Any missing path, extra numeric-string-key path, nonnumeric key, key collision,
changed input identity, failed scientific check, or nonempty deficit fails
closed.

## Output and Decision

The separate output is
`threes_rl/runs/forensics/o3_selected_integrity_reseal_v2/O3_SELECTED_INTEGRITY_RESEAL_V2.json`.
It embeds the exact post-JSON selected scientific payload, records both
scientific hashes and the coercion proof, binds this amendment, runner, tests,
and test evidence, and uses a JSON-round-trip-stable self hash.

The one terminal decision is:

- `READY_O3_OPTION_TRAINING_INTEGRITY_RESEALED` when every frozen check passes;
  or
- `HOLD_O3_SELECTED_INTEGRITY_RESEAL` on any integrity or operational error.

READY repairs only the serialization envelope. It does not alter or reinterpret
the original HOLD and creates no new scientific evidence by itself.

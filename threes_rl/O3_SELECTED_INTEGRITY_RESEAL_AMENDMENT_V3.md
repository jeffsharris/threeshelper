# O3 Selected-Root Integrity Reseal Amendment V3

Status: authorized no-recompute engineering repair. The original acquisition,
recovery, and V2 reseal namespaces and their terminal HOLD decisions remain
immutable and authoritative historical evidence.

## Scope

V3 may read only:

- the sealed recovery union, support, selected, and terminal-result JSON;
- the immutable V2 amendment, runner, tests, and terminal envelope; and
- its own amendment, runner, tests, and test evidence.

No replay may be parsed. No support candidate or allocation may be recomputed.
No stream, label, rollout, model, candidate action, score, max tile, or policy
outcome may be opened before V3 is READY.

## Frozen History

The four recovery inputs retain the file/payload identities frozen by V2:

- union: `02ea2c5be8823de775f56b7267f9c8371d26efc53897115b25733f8ef4527311` /
  `cec88701a1754f1064d639dae09cd6856ee18ce9399865338ebed7107f672d94`
- support: `4c71513e6a3a2778bb8d1db0ba08f8ff5a1f0d6edc82ee1208b7458593059d27` /
  `27ae3a6aca5f1de71ee18df193c0663a83579d3aeba65cd864065cfff594e25a`
- selected: file
  `9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049`,
  pre-serialization payload
  `c6c8b1a35cc63f4c1c1fdc98579f1ae0859a84c5eef7203306000223ac9c61a5`,
  and post-JSON canonical body
  `d9600cf420d947826c812b88225633b78a889f94f94ce39270dd71bc11b12f0e`
- recovery HOLD: `962da52b83b8746c006a9ef5fbe1fdd34f43e9c7bf97d9b6ff48f2a42019c23a` /
  `a679d512d6ce44bf5fd4ecd8249d15625c59f342e64796a6d5eb894396224ad0`

V2 identities are:

- amendment `380a13c9472d25edfe32a5b9f979c365514a125b9f74cb9ce98702413ffa78c8`
- runner `4e4c17c9f059c3c4e1679bde0a3811cfd85bbca953bf491c7cbc9d9c06f5cab6`
- tests `d681baa7e5f80de982f74849d05b6963292e7401bb9cd4070c69e8a54a790db6`
- terminal envelope file/payload
  `f466cae4e298edfc25499a90a78bfb6d6e037e2d065be72eb0de498cf9b31d57` /
  `58b55acb66033092dad5e789421d4cb60adfe960ccf25e1a6ef277e81141357d`

V2 remains `HOLD_O3_SELECTED_INTEGRITY_RESEAL` because its evidence-writing
CLI entered the sealing branch before creating test evidence.

## Serialization Contract

V3 preserves the V2 scientific verifier and exact proof: numeric-string keys
occur at exactly
`per_role.{train,development,untouched_mechanism}.{target_counts,descriptive_stage_counts}`.
Converting only those six maps' keys back to integers must reproduce the
selected pre-serialization payload hash exactly. Every aggregate scientific
check, count, family cap, and empty-deficit check remains unchanged.

## Orchestration Repair

The parser uses `subcommand` only for `write-test-evidence` versus `seal`.
Repeatable command records use the distinct destination
`recorded_commands`. Tests must prove that evidence routing cannot call seal,
that missing evidence seals HOLD, and that test evidence and the terminal
envelope survive JSON reload with their self hashes intact.

Test evidence is written first to
`threes_rl/runs/forensics/o3_selected_integrity_reseal_v3_test_evidence.json`.
Only after its identity verifies may exactly one terminal envelope be created
at
`threes_rl/runs/forensics/o3_selected_integrity_reseal_v3/O3_SELECTED_INTEGRITY_RESEAL_V3.json`.

The terminal decision is exactly
`READY_O3_OPTION_TRAINING_INTEGRITY_RESEALED_V3` or
`HOLD_O3_SELECTED_INTEGRITY_RESEAL_V3`. READY repairs serialization integrity
only; it is not new scientific evidence or promotion evidence.

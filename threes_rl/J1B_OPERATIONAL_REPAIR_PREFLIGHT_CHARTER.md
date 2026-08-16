# J1b deterministic Torch-runtime repair preflight charter

Status: frozen before J1b implementation.

This charter opens only an outcome-free J1b readiness surface. It does not
authorize a J1b phase lock, marker, stream reservation, stream consumption,
normal-start game, optimizer step, checkpoint, development, confirmation, or
promotion.

## 1. Parent and spent execution boundary

The J1 and J1a scientific contract remains byte-identical. J1b may import and
call the reviewed parent implementation, but it may not edit or reinterpret:

- `threes_rl/J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md`
- `threes_rl/J1A_OUTCOME_FREE_COST_POWER_AMENDMENT.md`
- `threes_rl/J1_EXECUTION_SURFACE_CHARTER.md`
- `threes_rl/j1_joint_policy_value.py`
- `threes_rl/j1_execution_surface.py`
- any accepted parent test or readiness artifact

The accepted execution-surface source identities are:

- charter:
  `468cc181c32a934fcbc64bb4cadc22758bd0fc46870f0f120f9ac6008ddb696a`
- runner:
  `d4367d95aba05ec592310008bae21e7de90905fa1268601dd60cc8fcb2b6f2bd`
- tests:
  `cb696e4502d61abd7a24d5781d7c15e2dd8a0ffed538480ecbd2a27434a339cf`

All six files in
`threes_rl/runs/forensics/j1_execution_surface_readiness_v1` are immutable.
The READY lock/result file identities are respectively
`e7f648eb04d7d197a9a2391352f82af5df6a12f7868ced8c8e9559703adb9fdc`
and
`ba3e9d67c64b89cf583c2ad1778b073a6a702c003bf1a895c164d6f9f984d4f6`.

`threes_rl/runs/forensics/j1_execution_v1` is permanently spent at
`HOLD_J1_OPERATIONAL` and must remain byte-for-byte unchanged. Its terminal
file/payload identities are
`21092fb34631eb0eaf48811caa814ff4d05abbb23c9bc5add85eefd93a8959d3` /
`9bcc81d217141fdfa801d1fca606c356720e4ac5c0e2a26f9d1ab688ca93dbcf`.
Its retention file/payload identities are
`dc339aafdbe32859d07c591a36c9088afa53f5be30412f3340049ca18994ceb0` /
`11cc89c6a6fe41ff74c472e3fa0b61d179e1cedfa4755cc4f13fe7ced44018b2`,
with file-inventory SHA
`7233c65745a9ae7258dbb165b60f4ae55c1cf60376819b80bb9e0be17d677471`.

The authenticated terminal evidence must reproduce genesis sequence 0,
zero completed roots, zero started/finished/abandoned attempts, zero optimizer
steps, and zero round aggregates. The operational cause must reproduce as:
deterministic algorithms enabled, intra-op threads 1, and inter-op threads 12
after frozen model initialization, while the unchanged guard requires inter-op
threads 1. This is not a scientific result.

## 2. Sole repair

In a fresh J1b scientific process, before importing the parent execution
surface and before any owner, reservation, consumption, genesis, game, or
optimizer work:

1. import PyTorch;
2. call `torch.set_num_interop_threads(1)`;
3. call `torch.set_num_threads(1)`;
4. call `torch.use_deterministic_algorithms(True)`;
5. verify exact values `interop=1`, `intraop=1`, and deterministic algorithms
   enabled;
6. only then import the parent J1 modules and initialize the frozen model and
   optimizer;
7. run the unchanged first parent operational guard and require it to pass.

An inability to establish these values is a fail-closed operational HOLD before
owner acquisition or any stream reservation/consumption. The inter-op guard may
not be removed, weakened, bypassed, or treated as advisory. No other runtime,
model, optimizer, seed, or scientific semantic may change.

The J1b preflight runner must be import-light: importing it may not import
PyTorch or a parent J1 execution module. The clean-subprocess production-path
probe is the authority for ordering.

## 3. Frozen scientific identity

J1b preserves exactly:

- a from-scratch 411,656-parameter model and the original initialization seed;
- 16,384 independent complete normal-start roots, 64 rounds by 256 roots,
  16 synchronous environments, and `starter_tile=None`;
- dense score-delta telescoping, fixed auxiliary labels/coefficient,
  root-equal clipped PPO, four epochs, minibatch 4,096, the exact learning-rate
  schedule, and only the round-64 candidate;
- the unchanged train-only sanity gate;
- 72 active hours, 24 GiB, disk hard floor above 100 GiB and target above
  120 GiB;
- one process, one heavy job, `nice>=10`, one Torch intra-op thread, one Torch
  inter-op thread, and deterministic algorithms;
- all crash-resume, bounded-I/O, retirement, ownership, service, lineage,
  development, and confirmation contracts from the parent surface.

No J1b outcome can be generated in this preflight.

## 4. Fresh training partition

The original training rows consumed these inclusive prefixes:

- logical: `213000000000..213000016383`
- deck: `214000000000..214000016383`
- slot: `215000000000..215000016383`
- candidate policy: `216000000000..216000016383`

J1b freezes the next contiguous, equal-size prefixes in the same prospective
namespace families:

- logical: `213000016384..213000032767`
- deck: `214000016384..214000032767`
- slot: `215000016384..215000032767`
- candidate policy: `216000016384..216000032767`

There are exactly 16,384 rows. Row `i` uses offset `i` in all four prefixes,
has block `i mod 8`, `starter_tile=None`, one candidate arm, and no control
policy stream. Every row, root, ancestry, and role identity is unique. Root IDs
are derived from a new immutable J1b training root-commitment payload and the
accepted marker-bound parent formula. No model seed changes.

A compact immutable denylist must bind all accepted historical intervals, the
entire original consumed J1 partition, and the new J1b partition. The new
partition must have zero intersection with every denied interval and no
internal role collision. No broad schema-heterogeneous payload parser is
permitted.

## 5. Separate files and commands

New source paths:

- `threes_rl/J1B_OPERATIONAL_REPAIR_PREFLIGHT_CHARTER.md`
- `threes_rl/j1b_operational_repair_preflight.py`
- `tests/test_rl_j1b_operational_repair_preflight.py`

Readiness namespace:
`threes_rl/runs/forensics/j1b_operational_repair_readiness_v1`

Future execution namespace, which must remain absent in this turn:
`threes_rl/runs/forensics/j1b_execution_v1`

The readiness package consists of:

- `J1B_TEST_EVIDENCE.json`
- `J1B_GENESIS_ROOT_CAUSE_AUDIT.json`
- `J1B_PROTECTED_STREAM_DENYLIST.json`
- `J1B_PROSPECTIVE_TRAINING_MANIFEST.json`
- `J1B_RUNTIME_ORCHESTRATION_AUDIT.json`
- `J1B_RUNTIME_STORAGE_PROJECTION.json`
- `J1B_SCHEMA.json`
- `J1B_READINESS_LOCK.json`
- `J1B_READINESS_RESULT.json`

Only test-evidence writing and outcome-free `prepare` commands are permitted.
The runner contains no J1b phase-lock, open, materialize, or execute command in
this version.

## 6. Tests and operational gates

Before readiness sealing:

- a clean nice-10 subprocess must prove the production entry ordering, frozen
  model identity, exact Torch settings, and first unchanged parent operational
  guard PASS before any owner/reservation/consumption path can run;
- a forced configuration failure must stop before owner, reservation,
  consumption, genesis, model initialization, game, or optimizer callbacks;
- parent identities and all original J1 execution files must rehash exactly;
- the genesis-only and root-cause evidence must reproduce;
- the fresh row/root/ancestry/stream partition and denylist collision proof must
  pass;
- py_compile, focused J1b tests, immutable J1/J1a/J1 execution-surface suites,
  and applicable non-scientific regressions must pass;
- no competing heavy process, healthy ports 8765/8770/advisor/dashboard,
  protected top three `263670/261369/258561`, and disk above the hard floor
  must be recorded without reading human-session content;
- the parent runtime/storage projection is reused byte-for-byte and remains
  within its frozen central caps.

## 7. Decision and zero-work boundary

The readiness result seals exactly one decision:

- `READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT`
- `HOLD_J1B_OPERATIONAL_REPAIR_PREFLIGHT`
- `KILL_J1B_PREFLIGHT_INTEGRITY`

READY authorizes only research-lead review and a later separately authorized
J1b training phase surface. Mutable service, process, disk, or inability-to-set
runtime faults HOLD. Immutable identity, collision, schema, or false-evidence
faults KILL this exact J1b preflight.

At terminal, all of the following remain zero: J1b phase locks, markers,
owners, reservations, consumptions, genesis commits, games, labels, scientific
optimizer steps, checkpoints, development/confirmation reads, policy/score
outcomes, human-session reads, incumbent changes, dashboard changes, and
promotion.

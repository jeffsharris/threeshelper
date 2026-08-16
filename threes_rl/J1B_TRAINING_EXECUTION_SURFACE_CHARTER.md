# J1b Training-Only Execution-Surface Charter

Status: frozen before implementation; outcome-free execution-surface readiness only.

## 1. Scope and preserved evidence

J1b is a narrow operational wrapper around the unchanged J1 scientific
training contract. Its sole semantic repair is to establish and verify the
deterministic PyTorch runtime in a fresh process before any ownership, stream,
commit, game, or optimizer artifact can exist:

- `torch.set_num_interop_threads(1)`;
- `torch.set_num_threads(1)`;
- `torch.use_deterministic_algorithms(True)`;
- exact postcondition checks for inter-op `1`, intra-op `1`, and deterministic
  algorithms enabled.

The original J1 execution at
`threes_rl/runs/forensics/j1_execution_v1` remains the authoritative immutable
`HOLD_J1_OPERATIONAL`. J1, J1a, the J1 execution-surface readiness package,
the J1b operational-repair charter/A1/source/tests, the pre-A1 historical
evidence, and the complete sealed J1b operational-repair readiness package are
immutable inputs. J1b never edits, reruns, resumes, or reuses the spent J1
execution or its consumed streams.

This charter does not change the 411,656-parameter model, initialization seed,
optimizer, PPO losses, root-equal weighting, dense score-delta return,
auxiliaries, legal masking, schedule, learning sanity gate, 64 rounds by 256
roots, 16 synchronous environments, four epochs, minibatch 4,096, 5,000-move
integrity ceiling, storage/runtime gates, or one-candidate rule. It does not
authorize development, confirmation, promotion, human access, or dashboard or
incumbent mutation.

## 2. Namespaces and immutable inputs

Implementation/readiness paths:

- charter:
  `threes_rl/J1B_TRAINING_EXECUTION_SURFACE_CHARTER.md`;
- runner:
  `threes_rl/j1b_training_execution_surface.py`;
- focused tests:
  `tests/test_rl_j1b_training_execution_surface.py`;
- readiness namespace:
  `threes_rl/runs/forensics/j1b_training_execution_surface_readiness_v1`.

The future scientific execution root is exactly:

`/Users/jeffharris/code/threeshelper/threes_rl/runs/forensics/j1b_execution_v1`

It must remain absent throughout construction, testing, projection, and
readiness sealing. Miniature fixtures use temporary roots only and carry
`execution_mode=miniature_fixture`, `scientific_authority=false`.

The readiness lock binds, by absolute path and SHA-256:

1. this charter, the final runner, and final focused tests;
2. J1b operational-repair charter SHA
   `a426801fc3015051ea51517e925a7d1c2e556718e2551ee480b802c8a7422cc1`,
   A1 SHA
   `64de3de37bff6a08bd95da217dc52d2f4bb58fbf99d28bede263a44d0aa2eb9c`,
   runner SHA
   `7d73565c510dfe74b87ec362c05f8928e15a65cb8af5494b5ad9fe5f4c30ca5f`,
   and tests SHA
   `f7e55b71f7954fcbdd4db61c1693d773b8ea106684ea19ad19998be15f4dbaff`;
3. J1b operational-repair lock/result file SHAs
   `b8b5377370f0e9e04739aae582604ce85f38bd1ddf84b5312a2cf12406f38814`
   and
   `108038d15b222afd00c07c9801b460fb4687bfe0a9e8a4fb54a59e58e8907ec6`;
4. every artifact identity embedded in that lock, including test evidence,
   root-cause audit, protected denylist, prospective manifest, runtime audit,
   projection, schema, parent J1/J1a sources/readiness artifacts, and the pre-A1
   evidence;
5. the complete spent-J1 execution file inventory embedded in the J1b lock,
   including terminal file SHA
   `21092fb34631eb0eaf48811caa814ff4d05abbb23c9bc5add85eefd93a8959d3`
   and retention file SHA
   `dc339aafdbe32859d07c591a36c9088afa53f5be30412f3340049ca18994ceb0`;
6. the parent bounded training engine source identity and the parent model,
   simulator, environment, feature, and policy dependencies already frozen by
   the accepted readiness chain.

Any missing, moved, changed, malformed, or non-READY immutable input fails
closed.

## 3. Exact fresh training manifest

Materialization copies, without regeneration or reinterpretation, the exact
16,384 ordered rows from
`J1B_PROSPECTIVE_TRAINING_MANIFEST.json` in the accepted J1b readiness
namespace. It binds:

- source file SHA
  `2bb0b2385360f2d06c019fdbac11cb58515629ab4f5fcf321624f499a07329f9`;
- source payload SHA
  `f85a7624b2e8052d0b451bde9bf792181e08e055406fb5837232655a48f8a8a8`;
- outer canonical rows SHA
  `4d28217d402d8b0e67e5465c90e433556f7f79adb0aedaf9c682c4defabfb170`;
- marker root-commitment payload SHA
  `fe8f67395dc2fced9d2b0f86c6990f66563681b868b04707d8771a8a4fe85d12`;
- exact unique root/ancestry set SHA
  `3a44d95d25c3b979d8e94dfbdb7c59b7e1b25dea891f8a31dd4f36040156ba55`.

The rows use `starter_tile=null`, one root per whole ancestry, and exactly these
fresh ranges:

- logical `213000016384..213000032767`;
- deck `214000016384..214000032767`;
- slot `215000016384..215000032767`;
- candidate policy `216000016384..216000032767`.

All four roles are unique for all rows. The materialized manifest is
create-once, byte-stable, and proves exact row order, row commitments, stream
sets, roots, ancestries, counts, and collision/denylist binding. There is no
substitution, filtering, repartitioning, or root regeneration.

## 4. Public production dispatcher

The runner is import-light through argument parsing: its module import and
parser use only the Python standard library. Its public CLI exposes exactly:

1. `seal-phase-lock`;
2. `open`;
3. `materialize`;
4. `execute`.

All commands are training-only. `--phase`, development, confirmation,
promotion, outcome-inspection, restart, alternate seed, and arbitrary
scientific hooks do not exist. Jobs must equal one. The exact execution root
and readiness namespace are fixed and revalidated.

The phase lock binds every readiness/source/history/spent-J1/manifest identity,
the exact four commands, the parent bounded-engine identity, and
`execution_mode=scientific`. `open` creates only a create-once activation
marker. `materialize` creates only the exact sealed root manifest. Each verb
reloads and rehashes all predecessor artifacts from disk. Concurrent immutable
creation is `O_EXCL` create-once with collision byte verification; no immutable
artifact is replaced.

Readiness construction is a separate, outcome-free Python function with no
public production verb. It may write only the readiness namespace and refuses
to run if the future execution root exists.

## 5. Mandatory execute ordering

In a fresh process, `execute` performs these steps in this order:

1. parse and validate arguments without importing Torch or parent J1 modules;
2. import Torch and set inter-op `1`, intra-op `1`, deterministic algorithms
   true;
3. verify those exact runtime postconditions;
4. import the immutable parent J1 modules;
5. initialize/load the exact frozen model and optimizer and verify parameter,
   finite-state, optimizer-binding, and deterministic save/load identities;
6. run the unchanged parent first operational guard with active seconds zero,
   target disk required, jobs one, nice at least ten, one heavy process,
   healthy services, protected dashboard/top-three, free disk above 100 GiB
   and target above 120 GiB, and output below cap;
7. only after all prior steps pass may it acquire or verifiably reclaim
   ownership, create stream reservation/consumption records, establish genesis,
   or perform scientific work.

Runtime-configuration or first-guard failure raises an operational HOLD before
owner, reservation, consumption, genesis, game, transition, optimizer, or
checkpoint artifacts. The wrapper does not reproduce the spent parent ordering
defect.

## 6. Scientific engine and durability

Scientific execution routes only to the immutable parent
`execute_training_engine_bounded` with its exact scientific
`TrainingEngineConfig()`. The legacy verbose engine and all fixture hooks are
unreachable in scientific mode. The wrapper may not transform rows or alter
the bounded engine configuration.

After the pre-owner gate, ownership, dead-owner recovery, stream
reservation/consumption, authenticated genesis, rolling slots, indexed commit
chain, runtime ledger, output accountant, 32-tick collection cadence,
append-once transition chunks and root blobs, ephemeral round batches,
retirement intents/recovery, bounded abandoned-unit charging, model/Adam
durability, exact resume, operational guards, storage/file/fsync caps, and
terminal full-chain audits remain the parent contracts unchanged.

Crash resume is permitted only in the same phase, marker, lock, command,
manifest, runner/engine, and execution mode. A live or mismatched owner fails
closed. Consumption opened by a predecessor owner is reused only through the
authenticated ownership recovery chain. There is no scientific restart after
a clean terminal or sanity miss.

## 7. Terminal and retention

At a complete round-64 boundary, the wrapper uses the parent authenticated
training report and frozen training-sanity decision. The exact round-64
checkpoint is authoritative only for `READY_J1_TRAINING_SANITY`; otherwise it
is quarantined. Clean sanity failure HOLDs. Immutable identity, model,
checkpoint, transition, optimizer, or commit corruption KILLs integrity.
Operational/resource failure HOLDs without scientific reinterpretation.

The wrapper terminal and retention bind its own charter/runner/tests/readiness,
the parent bounded engine, exact phase artifacts, stream opener and current
owner/recovery chain, checkpoint/sanity identities, runtime/storage/I/O
evidence, and `scientific_authority=true`. Terminal finalization and retention
are idempotent. Existing valid terminal plus absent retention repairs retention;
existing retention is reverified. No development or confirmation artifact is
created.

Possible training terminals are exactly:

- `READY_J1_TRAINING_SANITY`;
- the frozen clean training-sanity HOLD from the parent decision;
- `HOLD_J1B_OPERATIONAL`;
- `KILL_J1B_TRAINING_INTEGRITY`.

All terminals retain `PROMOTE=false`, `incumbent_changed=false`, and
`dashboard_changed=false`.

## 8. Readiness tests and projection

Before sealing readiness, focused tests must cover:

- standard-library-only module import and exact four-verb CLI;
- clean-subprocess runtime setup and successful first guard before ownership;
- runtime setup and first-guard failures leaving zero owner, reservation,
  consumption, genesis, game, optimizer, and checkpoint artifacts;
- create-once phase lock/open/materialize and concurrent-open stability;
- exact READY/source/A1/pre-A1/spent-J1/fresh-manifest binding and tamper
  rejection;
- exact materialized row/root/stream identities and no regeneration;
- scientific routing only to the bounded engine; legacy and phase escalation
  rejection;
- same-contract dead-owner/crash recovery and terminal/retention idempotence;
- miniature uninterrupted versus interrupted/resumed collection, update, and
  checkpoint equivalence through the wrapper;
- no development, confirmation, promotion, or outcome-inspection command.

The outcome-free projection uses accepted parent bounded-fixture measurements
without retiming scientific work, adds wrapper artifact/process overhead, and
covers 16,384 roots, 64 model/Adam commit states, three rolling slots/orphans,
current-round chunks and batch only, journals, retirement manifests,
checkpoint/metadata, file/fsync counts, and 25% margin. It must pass 72 active
hours, 24 GiB retained/peak, file/fsync caps, and the 100/120 GiB disk gates.
A miniature full chain is synthetic/fixture-only and cannot create scientific
artifacts.

Required test surfaces are focused J1b, immutable parent suites with accepted
counts `23/97/36/18`, and applicable non-scientific regressions. Any
deselection is named exactly.

## 9. Readiness decision and zero-work boundary

The readiness package records exact source/test commands and hashes, projection,
dependency and operational audits, immutable input bindings, and counters:

- phase locks, markers, materialized scientific manifests: zero;
- reservations and consumed streams: zero;
- games, transitions, labels, optimizer steps, checkpoints: zero;
- policy/score outcomes and holdout reads: zero;
- human-session reads: zero;
- incumbent/dashboard/top-three changes and promotion: zero.

It seals exactly one:

- `READY_J1B_TRAINING_EXECUTION_SURFACE`;
- `HOLD_J1B_TRAINING_EXECUTION_SURFACE`;
- `KILL_J1B_TRAINING_EXECUTION_SURFACE_INTEGRITY`.

READY authorizes only later research-lead review and a separately authorized
training phase lock/open/materialize/execute sequence. This turn ends with
`CONTINUE=research-lead review`, `HOLD=all J1b scientific execution`,
historical KILLs unchanged, and `PROMOTE=false`.

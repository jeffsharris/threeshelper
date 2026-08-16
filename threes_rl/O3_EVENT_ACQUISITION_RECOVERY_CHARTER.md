# O3 Event Acquisition Recovery Charter

Status: frozen recovery contract; original O3 acquisition remains permanently
sealed at `HOLD_O3_ACQUISITION_INTEGRITY`.

## 1. Scope

This recovery may generate only the `1,510` planned roots absent from the
original frozen `20,500`-root O3 P0 acquisition universe. It is not a resume,
repair, or reinterpretation of
`runs/forensics/o3_event_acquisition_v1`.

The original directory, marker, result, runner, attempts, completions, runtime,
and `18,990` retained replays are immutable inputs. The recovery lives only in
`runs/forensics/o3_event_acquisition_recovery_v1`.

## 2. Mechanical Complement

The recovery universe is the set difference between:

1. acquisition-purpose rows in the immutable O3 P0 stream manifest; and
2. `(family, game_index)` keys in the immutable original completion JSONL.

The result must contain exactly `302` rows for each frozen family, `1,510`
total, all with role `untouched_mechanism`. In the current immutable evidence
these are family-local game indices `3798..4099`, but the executable contract
derives and verifies that fact rather than assuming it.

No original replay JSON body may be parsed before the recovery union seal.
Original replay files are verified only by path, byte size, and SHA-256 against
completion metadata. No action, score, max-tile, milestone, geometry, frame, or
outcome field may be opened.

## 3. Frozen Collectors And Streams

Family order, policy specs, checkpoints, action signatures, root IDs, roles,
and logical/deck/slot/policy stream IDs are inherited byte-for-byte from O3
P0. No new or substitute stream is allowed. The recovery runs one root per
family in deterministic game-index order, one worker, at nice `>=10`.

Every planned root is retained unconditionally. There is no content-based
selection, family reallocation, early stop, or regeneration of a completed
root.

## 4. Zero-Game Preflight

Before a stream is consumed, the recovery writes:

- an atomic ownership-lock file;
- append-only structured process-audit evidence;
- `O3_RECOVERY_OPENED.json`; and
- `O3_RECOVERY_PREFLIGHT_RESULT.json`.

The preflight binds the recovery charter, runner, tests, test evidence, P0
artifacts, original acquisition artifacts, all original replay byte hashes,
collector policies/signatures, dependency sources, complement manifest,
collision inventory, exact commands, output directory, and resource limits.

`READY_O3_ACQUISITION_RECOVERY` requires:

- exact immutable original marker/result/file and payload hashes;
- `18,990` unique completion roots and replay hashes;
- exactly `3,798` completions per family;
- exactly `18,990 opened + 18,990 completed` attempt events and zero retries;
- complete train/development roles;
- exact `1,510`-row, `302`-per-family untouched-mechanism complement;
- zero requested-stream collisions outside the immutable P0 reservation;
- exact five-family policy/signature reproduction;
- no competing heavy Python/Threes process;
- exclusive process ownership, nice `>=10`, one worker;
- healthy ports `8765/8770`, advisor/dashboard, and protected top three;
- free disk `>100 GiB`, target `>120 GiB`; and
- projected original-plus-recovery storage below `28 GiB`.

Any failed check seals `HOLD_O3_ACQUISITION_RECOVERY_PREFLIGHT` with zero
games. READY authorizes only the exact marker-bound recovery command.

## 5. Process Ownership And Guards

The output namespace contains one atomically created ownership file. The
execution process must hold a nonblocking exclusive OS file lock for its full
lifetime. A second process cannot acquire or resume the same recovery.

Every preflight and execution guard appends a structured record containing:

- current PID and ancestor chain;
- every Python/Threes candidate PID, PPID, command, and classification;
- allowed dashboard/recorder processes;
- disallowed candidates;
- nice, disk, output bytes, services, and check results.

The dashboard and recorder remain allowed and uninterrupted. Any disallowed
candidate or resource/service failure seals a terminal operational HOLD.

## 6. Recovery Execution

The frozen execution processes only missing rows. Before dispatch it appends
an immutable attempt-open event. Charged evaluator runtime is persisted before
replay/completion writes. Each retained replay and completion is written once.
There are no hidden retries. An interruption may resume only under the same
marker and command, using existing attempt/replay evidence; any guard failure
requires a new research-lead decision.

Hard limits are:

- jobs `=1`;
- nice `>=10`;
- active recovery runtime `<=18h`;
- recovery output `<6 GiB`;
- combined original plus recovery output `<28 GiB`;
- free disk `>100 GiB` at every guard.

## 7. Content-Blind Union Seal

After `1,510/1,510` recovery completions, and before opening support content,
write `O3_RECOVERY_UNION_MANIFEST.json`. The union must prove:

- exact membership in the original P0 `20,500` roots;
- no duplicate or omitted `(family, game_index)` key;
- `4,100` roots per family;
- roles `5,020 train / 1,675 development / 13,805 untouched_mechanism`;
- unique ancestry and replay hashes across all `20,500`;
- exact attempt accounting for both source directories;
- no stream collision or role drift;
- unchanged hashes for every original top-level artifact and replay file.

Failure seals `HOLD_O3_ACQUISITION_UNION_INTEGRITY`. It is operational, not
scientific.

## 8. Support And Downstream Gates

Only after the immutable union seal passes may the existing frozen O3 support
scanner and allocator open replay state content. Their predicates, target
counts, one-state-per-ancestry rule, family limits, and decision rules remain
unchanged.

- Support pass: `READY_O3_OPTION_TRAINING`.
- Support shortfall: `HOLD_O3_DATA_OR_POWER`, never a representation kill.

If support passes, later phases must use the already frozen O3 scientific
contract: one `102,557`-parameter model, fixed training/checkpoint rule,
untouched `N=192` sustained-policy h40 common-OR mechanism test, then only on
PASS full-trajectory normal-start development and sealed confirmation.
No sweep, alternate objective, sign flip, one-move utility gate, incumbent
change, dashboard claim, or promotion is permitted before confirmation.

## 9. Decisions

- `CONTINUE`: only the next frozen phase after a passing gate.
- `HOLD`: any preflight, ownership, service, storage, union, or data-support
  failure.
- `KILL`: only a later adequately powered scientific failure under its frozen
  rule.
- `PROMOTE`: false unless independent normal-start confirmation passes.

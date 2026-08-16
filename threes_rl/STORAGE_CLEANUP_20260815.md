# RL Workspace Storage Cleanup

Cleanup date: 2026-08-15

This cleanup prepared the workspace for a new training course without changing
the incumbent, reopening a held branch, or deleting protected scientific
evidence.

## Result

| Measure | Before | After |
| --- | ---: | ---: |
| Free disk | 109.94 GiB | 124.95 GiB |
| Exact free-space gain |  | 15,741,536 KiB (15.01 GiB) |
| `.git` | about 8.5 GiB | 305,764 KiB (298.6 MiB) |
| `threes_rl/runs` | 51,193,076 KiB | 44,674,420 KiB (42.61 GiB) |
| `threes_rl/runs/forensics` | protected | 19,781,544 KiB (18.87 GiB) |

The final free-space level is above both the 100-GiB hard floor and the
120-GiB preferred research target. This cleanup does not retroactively change
the authoritative `HOLD_J2A1_V3A1_RECOVERY_EXECUTION_HEADROOM` decision or
authorize that spent recovery course.

## Removed

- Python bytecode caches, pytest caches, `.DS_Store` files, completed scratch
  run directories matching the established `test_*_tmp*`/`td_*_tmp*`
  conventions, and two runaway service error logs. The logs belong to retained
  services and may grow again if their launch wrappers retry an already-bound
  port.
- One obsolete Codex turn-diff recovery ref. It retained an 81,135-object
  snapshot of the pre-commit worktree, including ignored run artifacts. The
  user explicitly approved deleting this recovery-only ref after all source
  was committed and pushed. The remaining reachable Git history was repacked.
- Git's abandoned 44-KiB temporary object.
- Exactly 798 `.npy` learned-table files from two scientifically closed runs:
  - `td_phase4_promoted_balanced_restart_r1_v2_20260709` (original killed R1)
  - `td_phase4_incumbent_residual_r1b_v1_20260709` (R1b failed sealed
    confirmation and was permanently unpromoted)

The table deletion reclaimed exactly `6,668,898,048` bytes (6.21 GiB). The
reviewed apply manifest is retained at
`runs/forensics/storage_cleanup_20260815/preflight_and_deletion_manifest.json`,
with its CSV companion. Both run directories still contain their compact
configs, metrics, summaries, audits, metadata, and replay evidence.

## Preserved

- Every pre-existing file under `threes_rl/runs/forensics/`.
- All four model components named by `current_incumbent_policy.txt`; each still
  has its `latest/meta.json`.
- Protected top-three replays, evaluation manifests, human provenance, source
  replays, and all held/killed branch governance artifacts.
- The local `.venv`, so the next agent can run the harness immediately.
- Dashboard/recorder services and user applications. No user process was
  killed; the process audit found no stale RL training worker consuming RAM.

The `threes_rl/runs` tree remains intentionally local and Git-ignored. Its
remaining 42.61 GiB is mostly protected forensics plus incumbent components and
historical model/evaluation evidence that was not unambiguously disposable.

## Verification

- `git fsck --full --no-progress`: PASS. Only harmless dangling trees from
  prior worktree snapshots were reported.
- `git count-objects -vH`: one 18.25-MiB pack, 280.14 MiB loose objects, zero
  garbage.
- Cleanup preflight: two eligible rows, no active-process, incumbent, or
  protected-replay references; applied bytes equal eligible bytes exactly.
- Core simulator/evaluator selection: `102 passed`.
- Current V3A1 focused suite: `43 passed`.
- `compileall` over `threes_rl` and `tests`: PASS.
- The unfiltered historical suite completed `1804 passed, 69 failed, 1
  skipped`; the failures are chronology/state assertions against later sealed
  local artifacts and sandbox-only nice/service probes, not core simulator
  regressions. Use the artifact-independent selections in
  `RL_PROGRAM_HANDOFF.md` for a fresh course.

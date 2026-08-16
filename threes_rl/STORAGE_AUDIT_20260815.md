# Threes Storage and Memory Audit

Audit date: 2026-08-15

This audit measures the current checkout after the handoff cleanup. Directory
figures use allocated filesystem blocks from `du -sk`; format figures use
logical file bytes and therefore differ slightly. GiB means 1,073,741,824
bytes.

## Headline

The data volume currently has about `124.95 GiB` free. The complete Threes
checkout occupies `43.50 GiB`, of which `42.61 GiB` is `threes_rl/runs`.

| Checkout area | GiB | Share of checkout | Purpose |
| --- | ---: | ---: | --- |
| `threes_rl/runs` | 42.605 | 97.9% | Models, replays, executions, and evidence |
| `.venv` | 0.571 | 1.3% | Reproducible Python environment |
| `.git` | 0.292 | 0.7% | Reachable source history after repack |
| Source, tests, datasets, and other files | about 0.031 | 0.1% | Code and small inputs |
| **Total checkout** | **43.499** | **100%** | |

The source code itself is not the storage problem: the 315 files under
`threes_rl` outside `runs` total only about 9.6 MiB. The local datasets tree is
about 13.0 MiB.

## Where `runs` Is Spent

| Category | GiB | Share of `runs` | Disposition |
| --- | ---: | ---: | --- |
| Protected forensic executions/evidence | 18.865 | 44.3% | Preserve unless a new reviewed retention decision says otherwise |
| Other historical model/training runs | 16.111 | 37.8% | Main future cleanup-review pool |
| Current incumbent's four model components | 6.105 | 14.3% | Required to run the protected incumbent |
| Evaluation, continuations, replay, dashboard, and operations | 1.524 | 3.6% | Mixed reproducibility and generated output |
| **Total** | **42.605** | **100%** | |

The incumbent's `6.105 GiB` is split across the exact paths in
`current_incumbent_policy.txt`:

| Component | GiB |
| --- | ---: |
| Replay calibration sidecar | 3.051 |
| Endgame action-label sidecar | 2.034 |
| Student n-step/temporal-coherence table | 0.763 |
| MC1000 n-tuple parent | 0.257 |

The largest non-incumbent model directories are the late-reservoir model
(`3.052 GiB`), protected-late model (`3.052 GiB`), parent-init model
(`2.036 GiB`), reachability-reservoir model (`1.526 GiB`), and several
roughly `0.763 GiB` student/reachability variants. They are not live incumbent
dependencies, but some remain provenance or direct-parent evidence. Treat the
`16.111 GiB` category as a review pool, not a blanket deletion list.

## Protected Forensics

`runs/forensics` occupies `18.865 GiB` across 70,812 files. Most of it is five
spent scientific authorities:

| Forensic group | GiB | Share of forensics | Files / meaning |
| --- | ---: | ---: | --- |
| O3 acquisition plus recovery | 4.814 | 25.5% | 20,519 files; immutable root/replay corpus |
| J1d full PPO execution | 4.784 | 25.4% | 19,093 files; clean execution, learning-sanity HOLD |
| J1c full PPO execution | 4.043 | 21.4% | 18,836 files; integrity-killed execution |
| J2A1 partial teacher collection | 1.667 | 8.8% | 3,060 files; 3,048 completed protected roots |
| Rare-event and post-3072 frontiers | 1.458 | 7.7% | 978 files; high-tile source/evaluation evidence |
| All other forensic programs | 2.099 | 11.1% | Manifests, labels, audits, and smaller branches |
| **Total** | **18.865** | **100%** | **70,812 files** |

J1c, J1d, O3, and J2A1 account for most of both forensic bytes and inode count.
They are scientifically spent but intentionally retained because later gates
bind their byte identities. They should be archived or pruned only through a
new explicit retention decision, not ordinary housekeeping.

## What the Bytes Are

Across `runs`, logical file contents are approximately:

| Format | Logical GiB | Typical content |
| --- | ---: | --- |
| `.npy` | 22.11 | N-tuple learned tables and training arrays |
| `.bin` | 10.34 | Authenticated root/transition blobs and bounded execution state |
| `.json` | 9.33 | Replays, complete-root records, manifests, and audit payloads |
| All other formats | about 0.67 | HTML, JSONL, PyTorch checkpoints, logs, NPZ, CSV, and SQLite |

Outside forensics, almost all space is `.npy` model tables (`22.11 GiB`).
Inside forensics, almost all space is `.bin` (`10.34 GiB`) and `.json`
(`8.20 GiB`) evidence.

## Why Free Space Fell From About 150 GiB

The retention ledger records a useful comparable point on 2026-07-09: after a
large rejected-model cleanup, `runs` was about `26 GiB` and free disk was about
`150 GiB`. Today:

- `runs` is `42.61 GiB`, a net increase of about `16.61 GiB` from that point;
- free disk is `124.95 GiB`, about `25.05 GiB` below the recorded point;
- therefore Threes run growth explains roughly two thirds of the decline;
- the remaining roughly `8.4 GiB` is outside this checkout or normal APFS/OS
  variation and cannot be attributed to Threes data from this repository.

Immediately before the 2026-08-15 handoff cleanup, free disk had reached
`109.94 GiB`, `runs` was about `48.82 GiB`, and `.git` had temporarily grown to
about `8.5 GiB` because a recovery ref captured ignored run data. The cleanup
recovered about `15.01 GiB`, returning free space above the 120-GiB target.
See `STORAGE_CLEANUP_20260815.md` for the exact actions.

## Live RAM

No training worker is running. The three retained services use about 267 MiB
RSS in total:

| Process | RSS |
| --- | ---: |
| Dashboard watcher | 41.2 MiB |
| Static dashboard HTTP server | 10.8 MiB |
| Human-play recorder server | 215.1 MiB |

These services were left running intentionally. Model and replay files on disk
are not resident RAM unless a policy, evaluator, or training process loads
them.

## Cleanup Priorities

1. The `16.111 GiB` historical-model category is the best next review target.
   Verify each run against `ARTIFACT_RETENTION.md`, preserve compact evidence,
   and delete only learned arrays from explicitly closed, non-incumbent runs.
2. Generated HTML and duplicate continuation/evaluation outputs offer a much
   smaller `1.524 GiB` pool. Run the replay-retention audit before pruning.
3. Keep the current incumbent's `6.105 GiB` together unless replacing the
   incumbent and its reproducibility contract.
4. Treat the `18.865 GiB` forensic tree as protected. External immutable
   archival could reclaim local disk, but simple deletion would break many
   historical hash and chronology audits.

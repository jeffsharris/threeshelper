# O6 Competing-Risks P0 Charter

Date: 2026-07-27

Status: source preparation only. No O6 marker, source scan, root selection,
stream reservation, power execution, label, model, or policy outcome is
authorized by this charter.

## 1. Scientific Boundary

O6 asks whether a fresh scale-relative, action-conditioned designated-pair
controller can increase safe pair-specific merge probability by h40 relative
to competing absorbing failure under sustained control.

O6 is not an O5 repair or retry. O5 checkpoints, selected roots, task streams,
episodes, labels, and model outputs are forbidden. O3/O4 selected roots,
streams, bodies, labels, models, and outcomes are also forbidden. Human actions
are never labels.

The primary h40 endpoint is safe pair-specific merge versus no safe merge.
Competing failure and administrative h40 survival are both non-success for the
common-odds-ratio mechanism endpoint. Training uses a discrete-time
competing-risks likelihood with:

1. safe pair-specific merge;
2. third-party merge, lineage loss, terminal state, anchor/air violation, or
   no legal action;
3. remaining live risk mass.

Administrative h40 censoring contributes survival likelihood when present but
is not a fitted event class and has no minimum-count gate.

## 2. Natural Source Contract

Only completed natural normal-start machine ancestries are eligible. Human,
partial, restart, continuation, synthetic, playlist, top-score-selected, and
score-filtered sources are forbidden.

P0 may inspect only current board, visible preview, exact deck/cycle context,
move count, game-over flag, source provenance, and machine policy family.
Final/future score, future milestone, max-tile history, recorded action, and
policy outcome are forbidden.

One root is selected per whole ancestry using the minimum canonical SHA-256:

`SHA256("O6-P0-root-v1"|ancestry|state_hash|frame_index|target|pair_coords)`.

Eligibility requires a live exact simulator state, safe anchor, at least two
empty cells, at least two legal actions, target T in `{48,96,192}`, one
deterministically designated equal-valued pair, and no pair-specific safe merge
action at the root. Static geometry is support evidence only and must never be
described as proof of future h40 survival.

## 3. Protected Exclusion Union

Before candidate content opens, P0 must seal a complete protected-identity
inventory and exclusion-union hash. The union includes:

- exact O3/O4/O5 selected-root and stream governance manifests;
- every O3/O4/O5 reservation, marker, result, and historical lock;
- every prior development, untouched, sealed-confirmation, selector, MCTS,
  first-action, reachability, continuation, and human root/stream manifest;
- protected dashboard top-three replay identities;
- all human, restart, partial, continuation, synthetic, and playlist sources.

The inventory algorithm is frozen:

1. Resolve and hash every exact governance path and every file matching the
   frozen protected discovery roots/patterns:
   every `**/*.json`, `**/*.jsonl`, and `**/*.csv` file under
   `threes_rl/runs/eval_manifests`, `eval_artifacts`, `forensics`,
   `continuations`, `replays`, and `human_diagnostics`, plus every
   `*.json`, `*.jsonl`, and `*.csv` file directly under
   `threes_rl/runs/dashboard`.
2. Reject symlinks, path aliases, files outside the repository, duplicate
   canonical paths, unknown root/stream-bearing schemas, and inventory drift.
3. Classify only filenames containing one of
   `manifest`, `lock`, `marker`, `opened`, `result`, `seal`, `audit`,
   `selection`, `selected`, `roots`, `streams`, `collision`, or `retention`
   as governance data. `attempt`, `completion`, `completed`, `runtime`,
   `task`, `config`, and `preflight` are also governance tokens. From those
   files, whitelist only
   root/ancestry/replay-hash/state-hash and logical/deck/slot/policy stream
   identity fields. Never read score, action, model output, or outcome fields.
4. Exclude a candidate if any ancestry, root, source replay hash, state hash,
   or stream identity intersects the union.
5. Files under `threes_rl/runs/human_diagnostics`,
   `threes_rl/runs/forensics/o3_option_training_v1/episodes`,
   `threes_rl/runs/forensics/o3_option_training_v1/checkpoints`,
   `threes_rl/runs/forensics/o5_domain_safe_training_v2/episodes`, and
   `threes_rl/runs/forensics/o5_domain_safe_training_v2/checkpoints` are
   forbidden bodies: hash their inventory identity when required, exclude
   them wholesale, and never parse them. Non-governance replay and
   continuation bodies are likewise hash-only and forbidden. Every hash-only
   body must have its root/stream identities represented in classified
   governance data; a missing identity companion is an unknown schema and
   forces HOLD.

Zero overlap is mandatory. Any unclassified historical lock or confirmation
source seals `HOLD_O6_DATA_PREFLIGHT`.

## 4. Families And Partitions

Exact semantic family order and immutable action signatures are:

1. `o6_corner2`:
   `4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043`
2. `o6_expectimax2`:
   `2ad642cdca7739cc73af4f570de5054c422815f9a7d8f93a2619921b46b74b38`
3. `o6_parent_mc1000`:
   `e43dc11f3220557d7f9aef228db96dc6f06f49b26300d5a4128ea00bf8ba2064`
4. `o6_replaycal`:
   `e07c566b55d86a889ab7ca54d01c00c9b6cdf808fdb1627f70596bd829fdeab3`

The accepted action-only policy-lock identity is
`6b0384d9fedfc8f560853a050c28750194ec9c9d3d36cf2d9d7fd47a9a423ea0`.
P0 must reproduce all four signatures and all six pairwise distinctness gates
without rollout outcomes or timing.

Whole ancestries are partitioned before labels into train, development, and
untouched mechanism roles. One ancestry can occur in only one role. For the
power-selected untouched N, exact role counts are:

| Untouched N | Train | Development | Untouched |
| ---: | ---: | ---: | ---: |
| 192 | 384 | 96 | 192 |
| 256 | 512 | 128 | 256 |
| 384 | 768 | 192 | 384 |
| 512 | 1024 | 256 | 512 |

Each role is exactly balanced across the four families, giving 25% family
share. T48/T96/T192 marginals differ by at most one. A frozen 4x3 cell matrix
uses a common floor plus the lexicographic remainder pattern in the runner;
no post-content backtracking, substitution, or quota relaxation is allowed.

## 5. Outcome-Free Reachability

P0 reports counts by role candidate pool, family, target, aligned versus
unaligned designated pair, Manhattan/Chebyshev distance, blocker-density bin,
empties, legal-action count, preview category, and pending/deck-cycle bins.
It reports holes and concentration without using future states or outcomes.

Every selected state must pass exact restore, pair/lineage initialization,
feature-domain, legal transition, and save/load round-trip tests. Every
normalized model input, blocker density, successor-geometry target, risk
indicator, hazard probability, survival probability, and normalized time
coordinate must be finite and in `[0,1]`. Exact board tile values, target
values, move counts, coordinates, and categorical identifiers retain their
native domains and are not subject to the `[0,1]` constraint. The
competing-risk source contract must map only `success`, `failure`, and `live`;
administrative h40 censoring is generated only by 40 live transitions.

For an option trajectory, transition indices are exactly `t=1..40`. While the
pair is at risk, each transition contributes one row with ordered fields
`t`, `time_fraction=t/40`, `safe_merge_event`, `competing_failure_event`, and
`live_after_transition`. The three indicators are one-hot. `success` maps to
`(1,0,0)`, `failure` maps to `(0,1,0)`, and `live` maps to `(0,0,1)`.
The first success or failure is absorbing and emits the final row; no later
rows exist. A trajectory with 40 live rows is an administrative h40 censor,
represented by its 40 live likelihood rows plus an audit flag, never a third
event target. A shorter all-live trajectory is incomplete and invalid.
Round-trip serialization must reproduce every row and the absorbing/censor
identity exactly. The frozen event-time bands for the support gate are safe
merge at `1..10`, `11..20`, and `21..40`; at least two must be occupied.

## 6. Prospective Power And MDE

The untouched mechanism design uses eight paired CRN replicates per arm/root.
The treatment controls every move through h40; the frozen incumbent controls
every move in control. Primary strata are
`T48/T96/T192 x aligned/unaligned`. Family and stream-block signs are
descriptive.

For each `N in {192,256,384,512}` and OR in
`{1.25,1.35,1.50,1.75,2.00}`, run exactly 4,096 simulated datasets. Each
dataset uses exactly 4,096 whole-root stratified bootstrap replicates.
Simulation uses:

- control safe-merge base rate `188/1152`;
- beta-binomial root-propensity ICC sensitivity
  `rho in {0.05,0.15,0.25}`;
- treatment probability obtained by the exact odds shift;
- one shared uniform per paired arm/replicate;
- deterministic seed `2026072906 + 100000*N + 1000*round(100*OR)
  + 10*round(100*rho)`.

The estimator is the continuity-corrected strata-standardized
Mantel-Haenszel common odds ratio. The pass event is point OR at least 1.25
and whole-root bootstrap lower 95% bound above 1.0. Power is the minimum
full-pass rate across the ICC grid.

The future power implementation must evaluate all `4 x 5 x 3 = 60` cells,
all `4,096` datasets per cell, and all `4,096` whole-root bootstraps per
dataset. It must batch exactly 16 datasets and 64 bootstraps for memory
scheduling while preserving the same deterministic random draws and exact
estimator. No dataset, bootstrap, root, stratum, or ICC cell may be sampled,
approximated, analytically substituted, or dropped. P0 source preparation may
only report the exact workload and batch counts; it may not execute any power
dataset or bootstrap.

Freeze the smallest N with at least 80% power at true OR1.50. Report the
smallest MDE grid OR with at least 80% power at that N. If no N through 512
passes, seal `HOLD_O6_DATA_PREFLIGHT` before labels.

## 7. Fresh Stream Namespace

Proposed, not yet reserved, one-million-ID windows are:

| Purpose | Logical | Deck | Slot | Policy |
| --- | ---: | ---: | ---: | ---: |
| label learning | 197B | 198B | 199B | 200B |
| mechanism | 201B | 202B | 203B | 204B |
| normal development | 205B | 206B | 207B | 208B |
| confirmation | 209B | 210B | 211B | 212B |

P0 must prove every requested window collision-free against the complete
historical union before writing any stream-row manifest. A collision causes
HOLD; the windows may not be moved after inspection.

## 8. Pre-Fit Support Gate

Any later label execution must close all frozen tasks before exposing aggregate
support. Before fitting, require:

- at least 40 safe merges;
- at least 40 competing failures;
- at least 6 safe merges at each of T48/T96/T192;
- at least 3 safe merges in each represented family;
- finite, domain-valid risk rows;
- at least two occupied event-time bands.

There is no administrative-censor quota. A clean miss is data-support HOLD.
It cannot authorize threshold changes or checkpoint use.

## 9. Governance

One heavy job at a time, nice at least 10, free disk above 100 GiB with a
120 GiB target, output below a separately frozen cap, and healthy ports
8765/8770/advisor/dashboard/top-three are mandatory.

Future P0 decisions are exactly:

- `READY_O6_COMPETING_RISKS_P0`;
- `HOLD_O6_DATA_PREFLIGHT`;
- `KILL_O6_P0_INTEGRITY`.

READY could authorize only a separately frozen label/training charter. It
cannot authorize labels, training, policy evaluation, incumbent changes, or
promotion.

`CONTINUE=source preparation and research-lead review only`;
`HOLD=all O6 execution`; `KILL=false`; `PROMOTE=false`.

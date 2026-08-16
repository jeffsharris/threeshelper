# R1b Incumbent-Residual Multi-Stage TD Preregistration

Frozen: 2026-07-09 before R1b initialization or training.

## Rationale

Original R1 remains killed. Its bounded audit showed that the bare MC1000
parent was already `-8,448.84` behind the full incumbent on D0, while the
trained R1 leaf recovered an uncertain `+1,048.08` versus that parent. R1b is
a new candidate that isolates staged residual value learning without discarding
the incumbent's validated blend and phase-gated corrections.

## Candidate

- Total leaf: `V_total(s) = V_incumbent_frozen(s) + R_phase4(s)`.
- `V_incumbent_frozen` is exactly the policy in `current_incumbent_policy.txt`,
  including parent MC1000, student1 blend, replay calibration from mid onward,
  and the endgame action-label correction. Every component is read-only.
- `R_phase4` uses default n-tuples and four phase4 residual stages. Every
  residual entry and TC accumulator starts at zero. First training access in a
  later stage copies the exact residual weight and TC state from the preceding
  residual stage.
- Only residual tables update. TD errors and bootstraps use `V_total`.
- Frozen target: n-step `n=8`, TC, `alpha=0.001`.
- Trajectory policy: the full frozen depth-2 incumbent for every normal and
  restart episode. The residual never feeds back into action generation.
- Starts: exact alternating 50% normal and 50% restart using
  `runs/forensics/restart_manifests/r1_phase4_ancestry_balanced_v1_20260709.json`.
- Restart sampling remains stage uniform, ancestry uniform within stage, then
  state uniform within ancestry. Outcome provenance is reporting only.
- Seed `82000000`; `max_moves=5000`; compact latest-only storage;
  `actor_generation_jobs=8`. Parallel workers only generate independent frozen
  actor trajectories; residual targets and updates are applied in game-index
  order, so worker count does not change the learning sequence.
- Run: `td_phase4_incumbent_residual_r1b_v1_20260709`.

## Identity Gate

Before episode 1, save the zero-residual composite and evaluate all 64 frozen
D0 streams under depth-2 expectimax. Every score, move count, max tile, and
stream field must match the incumbent D0 CSV exactly. Residual arrays and TC
state must be exactly zero, masks untouched, and frozen component arrays
read-only. D0 is identity verification only and will not be reused for a policy
decision.

## Progressive Gates

1. Correctness smoke: 100 total episodes. Require exact pre-update identity,
   all four stages visited and touched, nonzero residual promotion, finite
   values/tables, no frozen-component mutation, exact 50/50 starts, plausible
   ancestry reports, exact save/load, and no periodic checkpoints. This is not
   a policy gate.
2. Pilot: continue the same run to 1,000 total episodes only after the smoke
   passes. Evaluate once on untouched D1, 192 paired normal-start games.
3. Do not inspect C. Do not continue to 5,000 without a new recorded gate.

## D1 Decision

- Primary endpoint: paired final score-minus-starter difference versus the
  frozen incumbent.
- Secondary: P3072, median, lower decile, and moves/survival.
- Harm: upper 95% paired-score CI below zero, point estimate no greater than
  `-10%` of the D1 incumbent mean, or clear catastrophic lower-tail/corner
  failure.
- Practically useful pilot gain: `+5%` of D1 incumbent mean, or
  `+1,059.57` score-minus-starter. Continue past 1,000 only if there is no harm
  and the paired-score interval still contains that gain. Otherwise kill R1b.
- D1 cannot promote the incumbent. C remains sealed, and any next training
  boundary requires a new recorded decision.

R1.5, R2, D0 reuse, C, dashboard changes, labels, sidecars, MCTS/UCT, and all
unrelated branches remain held.

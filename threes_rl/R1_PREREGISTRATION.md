# R1 Multi-Stage Restart TD V2 Preregistration

Frozen: 2026-07-09 before R1 training.

## Candidate

- One candidate only: `default` n-tuples, `phase4`, fixed top-left `1536` starter.
- Stage 0 prior: `threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest`.
- Later stages: copy exact weight plus both temporal-coherence accumulators from
  the immediately previous stage on first training access. Evaluation fallback
  is read-only.
- Target: n-step TD with `n=8`, temporal coherence, `alpha=0.001`.
- Rationale: `n=8` is the common supported target throughout the positive
  student sequence and the prior staged run. `alpha=0.001` is the gentlest
  staged/restart setting already exercised; this run tests promotion and
  ancestry balance rather than another step-size comparison.
- Actor during training: the candidate's own fast one-ply n-tuple action
  chooser. No sidecars, labels, milestone model, UCT, MCTS, risk split, preview
  context, or external actor.
- Starts: deterministic alternating 50% normal reset and 50% restart.
- Restart manifest:
  `threes_rl/runs/forensics/restart_manifests/r1_phase4_ancestry_balanced_v1_20260709.json`.
- Restart selection order: uniform eligible stage, then uniform root ancestry,
  then uniform state within that ancestry. Outcome provenance is reporting only.
- Training seed: `81000000`. `max_moves=5000`, `checkpoint_every=0`, latest only.

## Checkpoints

- Correctness smoke: 100 total episodes. No policy claim.
- Pilot: 1,000 total episodes, evaluate on D0 only.
- Medium: 5,000 total episodes, evaluate on D0+D1.
- Final development: 20,000 total episodes only if medium passes.
- Untouched confirmation C is evaluated once only if final development passes.

## Evaluation

- Frozen stream manifest:
  `threes_rl/runs/eval_manifests/r1_split_streams_d0_64_d1_192_c_512_20260709.json`.
- D0: 64 paired normal-start games.
- D1: 192 additional paired normal-start games.
- C: 512 paired normal-start games, frozen and untouched until confirmation.
- Both arms use the depth-2 expectimax actor. They share deck and slot-uniform
  stream IDs, with independent generator instances; slot uniforms map over each
  arm's legal insertion positions. Policy tie-breaking uses its own recorded
  stream.
- Primary endpoint: paired final score-minus-starter difference.
- Secondary: `P(non-starter max >=3072)`, median score-minus-starter,
  lower-decile score-minus-starter, and moves/survival.
- Diagnostics only: fixed h10/h20/h40 score gain and first-nonstarter-1536
  hazard on retained roots; high score is never a promotion endpoint.
- Intervals: seed/root-level paired bootstrap. Development and confirmation are
  analyzed separately.

## Frozen Decisions

- Harm: upper 95% paired-score CI below zero, or point estimate no greater than
  negative 10% of incumbent mean, or a clear new catastrophic lower-tail/corner
  failure.
- D0 continue: no harm and the effect is positive, or D0's score CI still
  contains a positive 5% incumbent-mean gain.
- Medium futility: only after promoted-feature/stage coverage stabilizes, stop
  if the score CI excludes a positive 3% incumbent-mean gain and P3072 has no
  useful signal.
- Development promotion candidate: paired-score 95% CI excludes zero and P3072
  difference is at least negative 2 percentage points, with no material
  lower-decile regression.
- Final promotion: the same criteria pass on untouched C. No promotion from D0,
  continuation starts, or high score.

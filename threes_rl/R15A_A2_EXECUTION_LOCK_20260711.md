# R1.5a/A2 Label, Fit, And Offline Execution Lock

Status: frozen before the first retained label rollout. This document resolves
implementation details left implicit by the original preregistration without
changing its scientific endpoints or gates.

## Natural Label Corpus

- Frozen manifest:
  `runs/forensics/r15a_context_a2/R15A_A2_LABEL_MANIFEST.json`, SHA-256
  `75c9fabadedc8e35bccd782ba5581cbce2eedced07a523893abcf02d0b2217eb`.
- Exactly 1,536 ordinary states and 16 replicates per state in A/B blocks of
  eight: 24,576 h40 trajectories. One path supplies h10/h20/h40 rows.
- The simulator-valid state value used in the target is the frozen incumbent
  composite one-ply post-spawn value: best legal merge score plus the complete
  phase-gated composite afterstate leaf. It is evaluated at root and each live
  endpoint; terminal endpoint leaf is zero.
- Return bins apply to the multi-step residual target, not raw score.
- Twenty-four deterministic replay audits are partition-balanced. Results are
  append-only and resumable; no full trajectory corpus is retained.

## Exact Optimization

- Both models use the immutable `ContextResidualModel` schema SHA-256
  `80e8a1aebcc68dab446da3b7b3bda2e899d45017eea49d0a0af8f42676530648`.
- Adam uses `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, learning rate `0.001`,
  and coupled L2 weight decay `0.0001` on all four parameter arrays.
- Batch size is 256 for exactly 200 epochs. Each epoch visits every train
  replicate once in a deterministic seed-20260711 permutation. A2 state loss
  weights are divided equally over the 16 replicates and applied without
  resampling or duplication.
- The h40 target is standardized using the A2-weighted train mean and standard
  deviation. The final epoch is the sole checkpoint.

## Synthetic Context Diagnostic

The 12 human H2 boards x four frozen donor-cycle pairs form 48 paired
same-board context cases. They are diagnostic-only and never enter fitting,
ordinary holdout metrics, checkpoint selection, or policy claims.

- Generate 16 new A/B paired replicates per case under a disjoint A2 synthetic
  stream namespace. Low/high arms share deck, slot, and policy stream IDs.
- Use the same h40 multi-step residual target and frozen incumbent continuation
  as the natural corpus.
- Primary contrast is predicted versus empirical high-minus-low expected
  target. Require Spearman correlation at least 0.25 and non-tie sign accuracy
  at least 65%.
- Opportunity direction pools expected-target, first-1536, and first-3072
  contrasts. Risk direction pools survival and anchor-preservation contrasts.
  Each pool independently requires non-tie sign accuracy at least 65% and
  median per-head Spearman correlation at least 0.25.
- Bootstrap and concentration units remain the six human ancestries. Passing
  this diagnostic cannot substitute for either ordinary holdout.

## Ordinary Offline Reporting

- Aggregate each state's 16 replicates before primary MAE and calibration
  reporting; average states within canonical roots.
- On ancestry holdout, report both natural root-balanced and equal-family
  metrics. The context-minus-board MAE improvement 95% ancestry bootstrap
  interval must exclude zero under both.
- Corner2 whole-family holdout point improvement must be positive. Binary Brier
  regression may not exceed 0.01 and ECE regression may not exceed 0.02 on
  either ordinary holdout under either reporting.
- Within ancestry holdout, every represented family improvement must be
  nonnegative and a majority strictly positive. Positive aggregate improvement
  may not be more than 40% attributable to one root, family, stage, or frozen
  context bin.

Any failed or ambiguous requirement kills R1.5a before policy evaluation and
opens only the preauthorized R2a branch.

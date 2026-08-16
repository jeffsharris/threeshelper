# Proposed R1b 5,000-Episode Gate

Status: completed 2026-07-09. D2 promotion-candidate gate passed; pre-C hold
is recorded in `R1B_PRE_C_DECISION.md`.

## Evidence At 1,000

- Untouched D1 paired mean score-minus-starter: `+1,425.98`, 95% CI
  `[-4,051.67, +6,731.73]` over 192 games.
- Median difference `+4,093.5`; lower-decile difference `+1,289.7`; moves
  `+7.20`; P3072 exactly tied at `8/192` in each arm.
- Candidate wins/losses/ties: `97 / 90 / 5`. P3072 transitions were `8` gains
  and `8` losses. Removing the largest positive seed leaves mean `+616.63`, so
  the point estimate is not carried by one game, but both tails contain changes
  above 100k and the interval remains wide.
- The frozen pilot rule passes: no harm, and the score interval contains the
  preregistered useful gain `+1,059.57`. This is not promotion evidence because
  the CI crosses zero.

## Proposed Training Boundary

- Continue the exact same run from 1,000 to 5,000 total episodes.
- Change no model, target, alpha, actor, start mixture, restart sampler, seed,
  worker semantics, or checkpoint policy.
- Keep the frozen incumbent actor and residual-only ordered updates.
- Evaluate no intermediate checkpoint. Keep latest-only storage.

## Proposed Evaluation Block

- Before training, freeze a new `D2` block of 512 split-stream normal-start
  games, disjoint from D0, D1, and sealed C.
- Prove deck, slot, policy, and logical stream IDs are disjoint from every
  existing block before training. Reading C stream metadata for this proof does
  not open or evaluate C outcomes.
- Cache the incumbent D2 baseline before candidate evaluation.
- Evaluate the 5,000 candidate exactly once on D2. D0 and D1 are descriptive
  history only; C remains sealed.

## Proposed Decision Rule

- Harm: unchanged from R1b, applied to standalone D2.
- Futility: only after residual feature coverage is compared with 1,000. Stop
  if coverage has stabilized, P3072 has no useful signal, and the upper paired
  score CI excludes a `+3%` incumbent-mean gain.
- Freeze substantial coverage saturation before seeing candidate D2 outcomes:
  every stage must have at least `85%` of its 5,000-episode touched entries
  already represented at 1,000 episodes. Report the per-stage retained share;
  one unsaturated stage makes the aggregate coverage flag false.
- Continue: no harm and either a positive point estimate or an interval that
  still contains the `+3%` gain, with no material lower-decile regression.
- Promotion candidate: D2 paired-score 95% CI excludes zero, P3072 difference
  is at least `-2 pp`, and lower decile has no material regression. This would
  authorize a separate confirmation decision, not automatic use of C.
- Review new corner/tail failures before declaring a promotion candidate.
- Freeze that review before candidate outcomes: inspect the 12 largest paired
  losses plus up to 12 new crossings below the incumbent's fixed D2 fifth
  percentile. A new lower-tail-rate increase over `2 pp` with a positive 95%
  interval blocks the gate. In the fixed replay set, three candidate-only
  terminal top-left anchor losses or three candidate-only terminal maximum-tile
  displacements trigger a corner-mechanism hold for review.
- Freeze material lower-decile regression before seeing candidate D2 outcomes:
  the lower-decile difference has a wholly negative 95% bootstrap interval, or
  its point estimate is at most `-10%` of the frozen D2 incumbent mean
  (`-2,056.71` score-minus-starter). Either condition blocks promotion.
- Before any future C decision, run the existing held-out h10/h20/h40
  pre-promotion diagnostic to test the congested level-up window.
- Any extension beyond 5,000 or any opening of C requires another recorded
  authorization.
- If D2 crosses zero after coverage is substantially saturated, apply futility
  once and stop. Do not create another development block. A negative point
  estimate or material tail regression kills R1b.

Estimated cost from measured throughput: about 3.5 hours of training plus
roughly 45-55 minutes each for the incumbent and candidate D2 evaluations with
eight workers.

## Outcome

- Training stopped exactly at 5,000. Integrity audit: full `PASS`.
- D2 paired score lift `+4,506.57`, 95% CI
  `[+1,176.73, +7,894.16]`.
- P3072 difference `+2.73 pp`, 95% CI `[+0.20 pp, +5.47 pp]`.
- Lower decile improved by `+1,455`; the fixed tail/corner audit did not block.
- Decision: `PROMOTION CANDIDATE PASS / PRE-C HOLD`. No incumbent, C, or
  dashboard change at this gate.

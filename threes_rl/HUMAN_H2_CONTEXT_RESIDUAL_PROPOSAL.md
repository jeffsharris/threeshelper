# H2 Context-Residual Research Proposal

Status: proposal only. No fitting or policy evaluation is authorized by this
document.

## Decision

H2 found a material preview/tile-cycle representation gap. Continue the
context branch to a bounded offline prediction gate, but do not train a
policy-facing residual until that gate is separately preregistered and passes.

Direct human-action supervision remains killed. Human states are context-rich
coverage, not action labels or held-out evaluation.

## Evidence

- Preview-only control: exact supported preview changes altered the incumbent
  root action in `98/269` variants (`36.4%`) and on at least one variant for
  `27/48` boards. Depth-2 search already captures this visible-preview effect.
- Cycle-only search: with board and current small preview held fixed, high-plus
  versus zero-plus exact cycles changed the selected action in only `3/48`
  cases, but changed the normalized top-two margin by at least `1%` in `34/48`
  (`70.8%`).
- Cycle-only h20 score: median absolute paired effect `3,400.59`, versus a
  replicate-level arm-permutation null 95th percentile of `2,155.43`.
  `46/48` cases exceeded 500 points and A/B signs agreed in `39/46` (`84.8%`).
- The signed high-plus contrast was `+5,988.02` score on average; A/B were both
  positive at `+4,268.41/+7,707.63`. Effects were positive in `45/48` cases
  and positive on average in all six target ancestries.
- The same context increased first-1536 by `+5.73 pp` and first-3072 by
  `+4.95 pp`, but reduced survival by `-8.98 pp` and anchor preservation by
  `-10.55 pp`. Both blocks shared these directions. Milestone and survival
  absolute-effect stability did not independently pass the frozen H2 gate.

Interpretation: context changes the value/risk tradeoff beyond the current
visible tile. A single scalar board-only leaf necessarily averages together
states with materially different short-horizon score opportunity and death
risk. H2 does not show that any particular context bonus improves play.

## Proposed Model

Preserve the complete frozen incumbent and add a small zero-initialized,
stage-aware context residual at leaf states:

`V_total(s) = V_incumbent(board) + R_context(board_summary, mechanics_context)`

Use one shared compact model with four phase4 output heads, not a new n-tuple
table crossed with context. Candidate inputs are explicit mechanics, so the
model does not relearn the simulator:

- visible preview one-hot and exact bonus candidate/value probabilities;
- `P(plus next)` after consuming the visible preview;
- full `P(value | plus)` and joint next-value probabilities;
- small-bag red/blue/gray probabilities after the visible preview;
- small-bag position and remaining counts;
- total-small count, span position, pending status, and distance to forced plus;
- current maximum and legal bonus window;
- compact board interaction summaries: phase4 stage, empties, legal count,
  corner/anchor status, top-edge ranks, support score mass, and built maximum.

Outputs must remain distribution-aware because H2 exposed opposing score and
survival effects:

- expected score-return residual used by search;
- score quantiles or fixed score-return bins;
- h10/h20/h40 survival and first-1536/3072 auxiliary hazards;
- anchor-preservation auxiliary diagnostic.

The auxiliary heads diagnose calibration and regularize representation; they
must not be converted into an unvalidated hand-weighted search reward.

## R1.5a Offline Gate

Before any policy-facing training, freeze one context dataset and one model
configuration.

1. Source simulator-valid states from exact normal-start and restart manifests,
   ancestry-capped and balanced over phase4 stage, plus probability, pending
   status, bag position, empties, and source family. Human roots may be included
   in development only.
2. Hold out complete root ancestries and at least one behavioral family. Never
   split correlated frames across train/test.
3. Generate incumbent-fixed split-stream continuations with next-preview
   expansion enabled. Preserve empirical score distributions and hazard labels;
   do not reduce the dataset to recorded actions.
4. Fit only the zero-initialized compact context residual. Frozen incumbent
   arrays and actions used for data generation may not mutate.
5. Evaluate first on same-board context contrasts, including a source-disjoint
   version of H2. Require prediction of both the positive score/milestone effect
   and the negative survival/anchor effect without ancestry concentration.

Suggested offline pass criteria, to be justified and frozen in the experiment
preregistration:

- zero-residual identity and save/load tests pass exactly;
- held-out expected-score error improves over a board/stage-only residual;
- held-out same-board context contrast has positive rank correlation and
  calibrated sign across independent stream blocks;
- survival and milestone Brier/calibration metrics improve over context-blind
  base rates;
- leave-one-ancestry and leave-one-family-out effects do not collapse.

If this gate fails, kill context fitting without a normal-start policy run.

## Policy Gate After Offline Success

Only after a separate stop/go note may the residual enter depth-2 expectimax.
The first candidate must preserve exact incumbent behavior at zero residual,
use a fresh development stream block, and pass paired score plus lower-tail,
P3072, survival, and corner safeguards. Sealed C is permanently unavailable;
freeze a new confirmation block only after a decisive development pass.

No hyperparameter sweep, human imitation loss, hand-weighted plus bonus, actor
feedback loop, dashboard update, or normal-start capability claim belongs in
R1.5a.

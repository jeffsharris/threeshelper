# Proposal To Restart The Threes Solver Research Program

Status date: 2026-07-09

Outcome update: R0 passed. The exact R1 candidate stopped for harm at the
1,000-episode D0 gate and is killed. A bounded audit found that the bare parent
MC1000 prior was already `-8,448.84` mean score-minus-starter behind the blended
incumbent on D0; the trained candidate recovered `+1,048.08` versus the parent
but remained `-7,400.77` behind the incumbent. Stage boundaries, promotion,
save/load, and the 50/50 ancestry-balanced sampler passed their checks. R1.5
and R2 are held pending review of
`runs/forensics/restart_program/r1_pilot_1000_failure_audit_20260709.json`.

R1b authorization update: the bounded audit justifies one new conservative
candidate, preregistered in `R1B_PREREGISTRATION.md`. R1b keeps the complete
incumbent leaf frozen and learns only a zero-initialized promoted phase4
residual under trajectories from the frozen incumbent policy. Original R1
remains killed. D0 is used once for exact pre-update identity only; untouched
D1 is the 1,000-episode policy gate; C remains sealed.

R1b pilot update: the 1,000-episode candidate passed the D1 continuation rule
without earning promotion. Paired mean was `+1,425.98`, CI
`[-4,051.67, +6,731.73]`, with P3072 exactly tied and positive median and
lower-decile estimates. The proposed 5,000 boundary and disjoint D2 gate are in
`R1B_NEXT_GATE_PROPOSAL.md` and require explicit authorization before launch.

This proposal synthesizes the existing research record, the external review of
`RESEARCH_PROGRAM_HANDOFF.md`, and a fresh audit of the current implementation.
It defines what should be reopened, what remains killed, and the sequence of
experiments required before returning to a broad self-play program.

## Executive Decision

Restart the research program under a **value-first, literature-parity** plan.

The first branch should not be human imitation, another MCTS sweep, or the full
phase-0 oracle gate. It should be a controlled reconstruction of multi-stage
n-tuple TD with restart-distribution training, using the existing simulator and
value infrastructure but correcting the most important gaps from the published
2048/Threes recipe:

1. retain genuinely separate stage tables;
2. add per-feature weight promotion between stages;
3. train with an ancestry-balanced mixture of normal starts and stage-balanced
   restart states;
4. preserve a strictly normal-start paired evaluation;
5. evaluate with a dense continuous score endpoint as the primary statistic;
6. test preview/tile-cycle context as a separate ablation rather than silently
   leaving the leaf state partially observed.

If the staged leaf improves normal-start score or non-starter `3072` frequency,
use it under a structurally improved, node-budgeted adaptive expectimax teacher.
Only after search demonstrates a stable action advantage should the program
consider iterative policy/value distillation or an AlphaZero-style loop.

Human/tracker data remains useful but opportunistic. The restart does not wait
for it.

## Why This Changes The Previous Roadmap

The prior program correctly concluded that the current incumbent-generated
frontier corpus is too correlated for policy claims. That conclusion remains
valid. What changes is the distinction between **training support** and
**evaluation support**:

- Value learning may train from replay starts, near misses, death windows, and
  novelty archives even when those starts are correlated or not reachable by
  the current actor.
- Promotion claims must still come from fresh normal starts or preregistered
  held-out root ancestries.

The previous roadmap applied excellent evaluation hygiene, but at times treated
the lack of source-diverse frontier roots as a prerequisite for training at all.
That is too restrictive for staged value learning. Restart distributions are a
way to teach a late-stage value function, not evidence that the end-to-end actor
can reach or exploit those states.

## What The External Review Got Right

### Multi-Stage TD Is Directly Relevant

Published multi-stage TD work reported a large improvement for Threes: a
multi-stage system reached `6144` in `7.83%` of games versus `0.45%` for its
single-stage TD comparator. The result makes multi-stage value learning a much
stronger prior than another hand-designed support selector.

### Restart Training Is Legitimate For Value Learning

Carousel/restart shaping is compatible with strict normal-start evaluation.
The policy does not need to reach every training state before the value function
can learn from it. The risk is not using restart states; the risk is mistaking
restart performance for normal-start capability.

### Expectimax Is A Better Planning Prior Than Low-Budget UCT Here

The failed UCT experiments used only 64 to 128 simulations in a wide stochastic
tree. Existing 2048-family results favor n-tuple values under expectimax, and
the current Threes code already has exact chance expansion, preview-aware state
keys, adaptive depth triggers, caches, and chance budgeting. A stronger leaf
plus better search budgeting is a more natural next teacher than another UCT
variant.

### The Existing Binary Frontier Gates Are Underpowered For Small Effects

A binary h40 milestone on 24 to 48 roots is useful for rejecting large claims,
but it is poorly matched to a real per-decision gain of one to three percentage
points. The `+0.78 pp` pooled result with CI `[0.00, +2.34]` should remain a
failed promotion, but not proof that every related mechanism has zero value.

### Preview And Deck State Deserve Explicit Attention

The search tree conditions on the current preview and expands future preview
outcomes. However, the n-tuple value API is board-only, and one-step TD targets
call transition expansion with `include_next_preview=False`. The learned leaf
therefore averages over information in the preview and tile-cycle state that is
available to the real decision process.

## Corrections To The External Review

The proposed directions are strong, but they are not entirely absent from the
history.

### Separate Staged Tables Already Exist

`StagedNtupleValue` supports four max-tile phases:

- `early_lt384`;
- `mid_384_768`;
- `late_1536`;
- `endgame_3072p`.

It also supports a 12-way phase-by-corner-risk mode. These are genuinely
separate tables, not only scalar phase blends.

What is missing is the key literature-style behavior of **per-feature weight
promotion**. The current conversion from a single table clones the same parent
into every stage once. It does not copy a feature's latest value from stage
`s-1` when that feature is first encountered in stage `s`.

### Restart-Distribution TD Has Been Tried, But Not Decisively

The strongest relevant prior run used:

- `phase4_corner3` staging;
- 80 training games;
- 60% replay-start probability;
- 2,141 stored frames from only 13 high-game replays;
- n-step TD with temporal coherence.

It looked excellent on restart-heavy training trajectories but lost badly on a
20-seed normal-start screen (`-14172.45` paired mean score-minus-starter).

That is important negative evidence against the exact prior configuration. It
is not a decisive test of multi-stage restart TD because the run was tiny,
highly ancestry-correlated, over-fragmented into 12 stages, and lacked weight
promotion. The catastrophic transfer failure also indicates that restart
sampling must preserve normal-start corner invariants.

### Adaptive Expectimax Already Exists And Is Too Slow As Implemented

The current adaptive depth-2-to-3 policy took roughly 4.4 times the runtime of
plain depth 2 on a one-seed comparison for a small score gain. A representative
chance-budget approximation was faster at its smallest budget but
underperformed the incumbent. Later depth-3 continuation attempts could not
finish even tiny screens promptly.

Therefore, deeper expectimax should not simply be switched on. It requires
structural work: node/time budgets, iterative deepening, faithful stochastic
sampling, and stronger transposition reuse.

## Confirmed Implementation Gaps

### 1. No Per-Feature Stage Weight Promotion

Current staged initialization either allocates independent zero/optimistic
tables or clones a base model into every stage. Add explicit initialization
masks so the first access to feature `i` in stage `s` copies the current weight
and temporal-coherence accumulators from stage `s-1`.

This must happen at feature access, not only when the whole stage object is
created. Otherwise later learning in an earlier stage cannot transfer to
previously unseen features in the next stage.

### 2. The Leaf Value Is Preview/Tile-Cycle Blind

`NtupleValue.value(board)` and `StagedNtupleValue.value(board)` receive only a
board. The search layer handles preview and cycle state above the leaf, but the
leaf cannot distinguish identical boards with different known upcoming tiles
or deck positions.

### 3. One-Step TD Marginalizes The Next Preview

`expected_afterstate_target(...)` calls
`transition_outcomes(..., include_next_preview=False)`. This is internally
consistent with a board-only leaf but prevents learning a context-conditioned
value even though replay and simulator states contain the required fields.

### 4. Paired Evaluation Does Not Yet Expose Independent Exogenous Streams

For low-variance action comparisons, tile/deck randomness should be coupled
after actions diverge, while spawn-position randomness should not be forced to
represent the same board location. The simulator currently presents one RNG
interface. A rigorous paired evaluator should split at least:

- tile-cycle / preview / bonus-value stream;
- spawn-slot stream;
- policy exploration stream.

The paired estimand should share the deck stream and independently transform
slot draws against each action's legal insertion set.

## Restart Hypotheses

### H1: Stage Specialization Is Undertrained, Not Disproven

A four-stage n-tuple model with weight promotion and enough balanced training
will improve late-board value accuracy without sacrificing normal-start corner
discipline.

### H2: Restart Training Can Repair Rare-State Coverage

An ancestry-balanced restart mixture can teach the late stages from existing
high-board and near-failure states while a substantial normal-start fraction
prevents continuation-only overfitting.

### H3: Preview/Cycle Context Explains Residual Leaf Error

Among identical or similar boards, known preview and deck-position features
carry enough information to improve value calibration and action ranking,
especially in the congested 10/20/40-move pre-promotion window.

### H4: A Better Leaf Makes Adaptive Expectimax Viable

Once the leaf is better calibrated in late stages, a node-budgeted expectimax
teacher will need fewer deep expansions to find useful action changes.

### H5: Dense Paired Endpoints Reveal Small Improvements Earlier

Paired score gain, fixed-horizon score, and time-to-milestone summaries will
provide more power than a binary h40 event alone while preserving milestone
probability as a key secondary outcome.

## Program Structure

The restart has four phases. Each phase has an explicit deliverable and gate.
No phase automatically authorizes the next.

## Phase R0: Literature-Parity And Evaluation Foundations

Purpose: make the first staged-TD experiment scientifically interpretable.

### R0.1 Implement Per-Feature Weight Promotion

Requirements:

- explicit feature-initialization masks per stage/table;
- first access in stage `s` copies weight from stage `s-1`;
- temporal-coherence state is copied or deliberately reset, with the choice
  documented;
- save/load round trips preserve initialization masks;
- tests cover promotion before and after the previous stage changes;
- no stage promotion across incompatible pattern sets.

### R0.2 Build An Ancestry-Balanced Restart Sampler

Use existing retained normal-start replay reservoirs, near-miss windows, and
death windows. Training records may be replay starts, but sampling must avoid
letting one game contribute thousands of near-identical updates.

Sampler policy:

- sample stage first;
- sample root ancestry second;
- sample one state/window within that ancestry third;
- cap repeated use of one ancestry within a batch or training interval;
- report effective roots and visits per stage;
- retain both success and failure/near-miss states;
- preserve exact preview and tile-cycle state from the replay.

### R0.3 Split Exogenous RNG Streams For Paired Evaluation

Implement deterministic stream IDs for deck, spawn slot, and policy
exploration. For paired policies/actions:

- share the deck stream;
- use the same slot uniform draw but map it independently over each legal slot
  set;
- never claim state-identical coupling after trajectories diverge;
- record stream IDs in artifacts.

This is the prerequisite for more sensitive group-sequential comparisons.

### R0.4 Freeze Baselines

Before training, rerun or reconstruct the incumbent on the planned paired
normal-start seed/deck blocks. Freeze:

- primary: paired final score-minus-starter difference;
- secondary: median, lower-tail score, moves/survival, and
  `P(max_tile_excl_starter >= 3072)`;
- diagnostic: h10/h20/h40 first-nonstarter-`1536` hazard and high score.

High score is never a standalone promotion endpoint.

### R1b Confirmation Outcome

R1b passed D2 but failed untouched C on the frozen primary criterion. C paired
score lift was `+788.18`, 95% CI `[-2,412.96, +4,021.11]`; P3072 tied at
`21/512`. No lower-tail or corner safeguard fired. The candidate is not
promoted and must not be rerun on C. Further roadmap phases remain held for a
new review decision.

### R0 Gate

Proceed only if weight promotion tests pass, restart sampling reports ancestry
counts correctly, and paired deck coupling reproduces identical outcomes for
identical policies.

## Phase R1: Multi-Stage Restart TD V2

Status: stopped at 1,000 episodes under the preregistered harm rule. Do not
continue this exact configuration to 5,000 or reuse D1/C for it.

Purpose: test the external review's highest-value hypothesis with minimal
confounding.

### Candidate Definition

Use:

- existing `default` n-tuple pattern set;
- four-stage `phase4` mode, not the 12-way corner-risk mode;
- current parent MC1000 checkpoint as the initial stage-0 prior;
- per-feature weight promotion;
- temporal coherence;
- existing afterstate TD or n-step target chosen before the run;
- a `50%` normal-start / `50%` restart-start mixture;
- stage- and ancestry-balanced restart sampling;
- the existing fast one-ply n-tuple actor during training;
- no action labels, milestone labels, sidecars, or MCTS.

Why four stages: the 12-stage prior divided a tiny dataset too aggressively.
The first restart should test the main multi-stage hypothesis, not phase/risk
interactions.

### Training Schedule

Use a progressive single-run schedule with resumable checkpoints rather than a
hyperparameter grid:

1. correctness smoke: enough episodes to touch and verify every stage;
2. pilot checkpoint: evaluate whether normal-start behavior has catastrophically
   regressed;
3. medium checkpoint: continue only if transfer is non-harmful;
4. final checkpoint: scale only the same preregistered run.

The previous 80-game attempt is too small to be the benchmark for this method.
The exact episode counts should be set from measured games/second and table
coverage, with stage feature-visit saturation reported at every checkpoint.

### Primary R1 Evaluation

Drop the staged table directly under the current depth-2 expectimax actor. Run
a paired normal-start comparison against the frozen incumbent using shared deck
streams.

Primary endpoint:

- paired final score-minus-starter difference.

Secondary endpoints:

- `P(non-starter 3072)` difference;
- median and lower-decile score difference;
- survival/move-count difference;
- fixed h40 score gain and first-`1536` hazard on retained diagnostic roots.

Use root/seed-level bootstrap intervals. Do not select the checkpoint on the
final test block; use a separate development block for checkpoint decisions.

### Group-Sequential Stop Rules

- **Harm stop:** candidate has a clearly negative paired score effect or
  materially worse `P(3072)` on a development checkpoint.
- **Futility stop:** stage-table coverage has stabilized but paired score and
  milestone effects remain near zero with intervals too narrow to contain a
  practically useful gain.
- **Continue:** point estimates are positive or plausibly useful without lower
  tail/corner failures.
- **Promotion candidate:** paired mean score improves with an interval excluding
  zero and `P(3072)` is non-inferior; confirm on an untouched larger block.

Do not promote from a 20-seed screen.

### R1 Interpretation

- If R1 passes, the program has direct evidence that value representation and
  restart coverage were the main bottlenecks.
- If R1 is neutral, proceed to the preview/cycle ablation before abandoning
  staged value learning.
- If R1 is clearly harmful after adequate coverage, kill this exact
  weight-promotion/restart configuration and restore the frontier-data/search
  diagnosis as the leading explanation.

## Phase R1.5: Preview And Tile-Cycle Value Ablation

Purpose: test the code-level partial-observability concern separately from the
main staging experiment.

### Start With A Predictive Audit

On held-out replay ancestries, measure whether preview/cycle features reduce
value error or improve legal-action ranking beyond board and stage alone.

Candidate context features:

- current preview kind/value or bonus-candidate summary;
- `small_pos` and a coarse remaining-small-tile count;
- `span_small_pos`;
- `large_pending`;
- stage and empty-count bucket.

Avoid multiplying every n-tuple table by the full context state initially.
Start with a small additive context correction or a late-stage-only context
split. The ablation must show held-out value or action-ranking gain before the
context representation is added to the actor.

### Training Semantics

If a context-conditioned value is introduced:

- value APIs must accept the full `SimState` or a frozen context object;
- TD transition expansion must include the next-preview distribution;
- cache keys must include all context used by the leaf;
- board-only caches may remain only around helpers that are mathematically
  context independent;
- stage and context contributions must be separately logged.

### R1.5 Gate

Keep the context model only if it improves held-out root-grouped value/action
ranking and then improves paired normal-start or frontier results without
material runtime regression.

## Phase R2: Node-Budgeted Adaptive Expectimax Teacher

Purpose: build the stronger policy-improvement operator that UCT failed to
provide.

### Search Design

Build on the existing preview-aware expectimax, but replace naive fixed-depth
deepening and biased representative-quantile chance pruning with:

- iterative deepening under a fixed node or time budget;
- transposition tables keyed by board, preview, tile-cycle state, and remaining
  depth/budget as appropriate;
- adaptive budget allocation triggered by low empty count, small action margin,
  or bonus preview;
- probability-faithful chance sampling or progressive widening;
- sequential racing across root actions using uncertainty bounds;
- the best R1/R1.5 staged value as leaf;
- deterministic reproducibility from explicit search RNG streams.

The teacher may search deeper in congested states where the branching factor
has already contracted. It should not pay depth-5 cost on ordinary easy moves.

### R2 Oracle Gate

Use the frozen phase-0 corpus rules, but update the statistics:

- pilot samples choose and freeze the teacher action;
- evaluation samples are fully independent;
- primary dense endpoint: paired h40 score gain or a preregistered continuous
  milestone-hazard integral;
- key secondary: first-nonstarter-`1536` probability by h10/h20/h40;
- score and survival non-inferiority;
- group-sequential root acquisition into the hundreds when cheap enough;
- leave-one-root and leave-one-family-out robustness.

The old MCTS configurations remain killed. The old binary selective-rollout
gate remains a failed promotion, not a reason to reject this new teacher.

### R2 Gate

Only a teacher with stable paired advantage and acceptable runtime may create
training targets or a runtime policy wrapper.

## Phase R3: Iterative Self-Play And Novelty Curriculum

Purpose: convert a proven teacher into a repeatable improvement loop.

This phase begins only after R1 or R2 produces a promoted candidate.

### Data Mixture

Maintain three explicit streams:

1. normal-start self-play for end-to-end competence;
2. ancestry-balanced restart training for stage coverage;
3. novelty/archive starts for rare support geometry.

A Go-Explore-style archive may retain cells keyed by a compact board signature,
for example stage, empty-count bucket, support-tile counts, corner topology,
preview/cycle bucket, and score band. Archive novelty is a data-acquisition
mechanism, not a reward or promotion metric.

### AlphaZero-Style Loop

If the R2 teacher passes:

1. generate search-improved action distributions and empirical outcomes;
2. fit or update the staged policy/value representation;
3. regenerate fresh normal-start and archive trajectories;
4. gate the new actor against the previous incumbent on paired normal starts;
5. promote only after the predefined normal-start rule passes.

The search teacher must remain stronger than the student. If distillation
catches up and search no longer adds value, improve the teacher before
continuing the loop.

## Human Data Policy

Keep the existing import pipeline live, but do not gate the restart on human
games.

Use human/tracker sessions for:

- novel root and board-geometry discovery;
- restart/archive seeding;
- held-out qualitative error analysis;
- testing whether human trajectories occupy support regions absent from
  self-play.

Do not treat recorded human actions as optimal labels. Any policy-facing use
still requires action-conditioned simulator evaluation.

## Revised Statistical Policy

### Continuous Primary Endpoints

For policy comparisons, prefer paired continuous score quantities:

- final score-minus-starter for normal starts;
- score gain at a fixed horizon for frontier roots;
- time-to-milestone or a preregistered milestone-hazard integral.

Retain binary `P(1536)`, `P(3072)`, and `P(6144)` because they express the real
capability goal, but do not make every early kill depend on a low-power binary
test.

### Common Randomness

Share tile/deck streams, not physical spawn positions. A uniform slot variate
may be shared, but each trajectory maps it into its own legal insertion set.
Once states diverge, positional identity is not meaningful.

### Effective Sample Size

- Training updates may reuse restart states, with ancestry-balanced sampling
  and explicit visit counts.
- Statistical intervals cluster by normal-start seed/root ancestry.
- Frames and repeated continuations do not count as independent samples.
- Multiple checkpoints from one behavioral lineage do not create source-family
  diversity.

### Sequential Testing

Use preregistered development checkpoints for harm/futility decisions and one
untouched confirmation block. Do not repeatedly inspect and stop on the same
final seed block without an alpha-spending or confidence-sequence design.

## What Is Reopened

- Four-stage n-tuple TD with literature-style weight promotion.
- Restart-distribution value training with ancestry-balanced sampling.
- Preview/tile-cycle-aware value as a controlled ablation.
- Structurally redesigned adaptive expectimax after leaf improvement.
- Dense paired endpoints and group-sequential evaluation.
- Novelty/archive state generation for training support.

## What Remains Killed Or Held

### Remains Killed

- The exact 80-game `phase4_corner3` replay-start checkpoint as a policy
  candidate.
- Existing `supportstock`, `second768`, and `regen768` policy-promotion
  variants.
- Existing mostly-one-`768` h40 selective-rollout configuration.
- Existing 64/128-simulation incumbent-leaf and target-directed UCT variants.
- More hyperparameter sweeps of those same operators.

### Remains Held

- Broad action-label sidecar fitting.
- Human imitation.
- Generic PPO from scratch.
- Dashboard promotion from continuation or restart states.
- Full AlphaZero-style distillation until a teacher passes R2.
- Heavy deep-search runs before the R2 structural search changes are benchmarked.

## First Executable Experiment

After R0 verification, run exactly one R1 candidate:

> Four-stage `phase4` n-tuple TD with per-feature weight promotion, temporal
> coherence, a 50/50 normal-start and ancestry-balanced restart mixture, and no
> auxiliary labels; evaluate it as the leaf under the current depth-2
> expectimax actor against the frozen incumbent on paired normal-start decks.

This experiment directly tests whether late-stage value representation and
training support are the bottleneck. It is stronger than repeating the prior
80-game run because it changes the mechanisms most likely responsible for that
run's failure while preserving a single interpretable hypothesis.

## Decision Tree

1. **R1 clearly improves normal-start score and preserves/improves P(3072):**
   promote the staged leaf candidate and begin R1.5/R2.
2. **R1 is neutral but well covered:** run the preview/cycle ablation before
   abandoning staged value learning.
3. **R1 is harmful or overfits restarts again:** kill the exact R1
   configuration; inspect stage boundaries, promotion semantics, and
   normal/restart mixture once, then stop rather than sweep.
4. **R1.5 shows preview/cycle value:** integrate the smallest effective context
   representation and reevaluate.
5. **R2 expectimax teacher passes:** begin iterative search/value self-play.
6. **R2 fails despite a better leaf:** prioritize novelty archives and human
   roots; do not return to UCT or proxy labels by default.

## Required Artifacts For Every Phase

- frozen config and code revision;
- source replay/root provenance summary;
- training ancestry visit counts by stage;
- stage-table coverage and promoted-feature counts;
- development versus confirmation seed manifests;
- paired deck/slot stream manifest;
- full score and milestone summaries;
- root-clustered intervals;
- explicit continue/kill/promote decision in `EXPERIMENT_LOG.md` and
  `CURRENT_DECISION_LEDGER.md`.

## References

- Kun-Hao Yeh et al., [Multi-Stage Temporal Difference Learning for 2048-like
  Games](https://arxiv.org/abs/1606.07374). The paper reports multi-stage TD
  improvements for both 2048 and Threes.
- Wojciech Jaskowski, [Mastering 2048 with Delayed Temporal Coherence Learning,
  Multi-Stage Weight Promotion, Redundant Encoding and Carousel
  Shaping](https://arxiv.org/abs/1604.05085). Relevant mechanisms include
  per-stage weight promotion, carousel restart shaping, temporal coherence, and
  deeper/time-budgeted expectimax with transposition tables.

## Bottom Line

The research program should restart, but not by discarding its methodological
discipline. The strongest new evidence is that the project already built much
of the right machinery but tested an incomplete, undertrained version of the
most relevant published approach.

The immediate bet is therefore not "more data" in the abstract. It is a better
late-stage value function trained where late-stage decisions occur, while
normal-start paired evaluation remains the only promotion authority. A better
leaf then gives adaptive expectimax, novelty curricula, and eventual
AlphaZero-style self-play a much firmer foundation.

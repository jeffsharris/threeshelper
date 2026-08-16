# Threes Solver Research Program Handoff

> Historical snapshot (2026-07-09). Use `RL_PROGRAM_HANDOFF.md` for the current
> program state and next-agent instructions.

Status date: 2026-07-09

This document is a self-contained briefing for an external ML researcher or
model asked to critique the research program and recommend the next step. The
authoritative detailed history remains in `EXPERIMENT_LOG.md`; the short-form
active decisions are in `CURRENT_DECISION_LEDGER.md`.

## Executive Summary

The project has a high-confidence simulator, a competent n-tuple/expectimax
incumbent, reproducible evaluation and replay tooling, and extensive diagnostics
of high-board failure states. The best observed normal-start score is `263670`
(`204621` excluding the fixed starter tile), but no normal-start evaluation has
created a non-starter `6144`.

The research is stuck at a data-and-policy-improvement bottleneck, not an
engineering bottleneck. The difficult decisions occur in congested boards
roughly 10, 20, and 40 moves before creating the next major tile. Existing
incumbent self-play rarely reaches those transitions, and the resulting state
corpus is dominated by a small number of related game ancestries and one policy
family. Local board-shape diagnostics often look promising, but direct paired
rollout gates on independent roots have not shown a stable action advantage.

The currently tested selective-rollout and MCTS/UCT operators failed their
promotion rules. More sweeps of those exact operators are killed. Broad policy
fitting is held because the available labels are sparse, unstable, and too
correlated to justify learning from them.

Two credible routes remain:

1. Acquire source-diverse high-board human/tracker games and use them to seed
   provenance-safe pre-milestone roots.
2. Build a materially different AlphaZero-style improvement loop, beginning
   with a bounded empirical oracle test that must prove its search teacher is
   stronger than the incumbent before any policy or value fitting.

Human data is the fastest practical source-diversity injection. It is not the
only theoretically valid route, and the negative MCTS results do not establish
that self-play is impossible. A 2026-07-09 retained replay inventory found that
existing normal-start replay artifacts can supply a phase-0-compliant
multi-family corpus if the overrepresented incumbent lineage is downsampled;
new acquisition is not required merely to get a second behavior family.

## Goal And Evaluation Semantics

The goal is superhuman normal-start play in Threes, ultimately maximizing final
board score rather than optimizing a hand-designed proxy.

Important evaluation details:

- The simulated game starts with a fixed `1536` starter tile in the top-left.
- Report score and max tile both with and without that starter contribution.
- A raw `>=1536` board test is not a valid milestone because the starter already
  satisfies it. Use `max_tile_excl_starter` or an explicitly defined raw tile
  count/geometry.
- Continuation-start and selected-state experiments are diagnostics. They do
  not establish normal-start capability.
- High score is useful tail evidence but is not, by itself, a promotion metric.
  Mean, median, survival, milestone probability, and paired-seed comparisons
  remain necessary.

## Engineering And Simulator Status

The simulator and evaluation substrate are mature enough for research:

- Move mechanics matched the prior move oracle on 20,000 random boards plus
  adversarial cases in all four directions with zero disagreements.
- 50,000 simulated non-terminal transitions passed tracker validation.
- The tile schedule matched the tracker `TileCycle` in lock-step tests.
- 270/270 replayable observed single-step moves reproduced exactly.
- Seeded environments are deterministic under the same action sequence.
- The terminal `6144 + 6144 -> 12288` rule and scoring are implemented.
- Evaluation supports normal starts, continuation roots, paired random seeds,
  replay capture, milestone retention, death forensics, and progress charts.
- Direct rollout and MCTS gates support checkpoint/resume.
- A dashboard and protected top-three replay playlist are live.
- Human/tracker replay import and 10/20/40 pre-promotion window extraction are
  implemented and tested.

Residual simulator risk is concentrated in rare start/schedule edge cases
inherited from the reverse-engineered tracker, not in ordinary move mechanics.

## Current Incumbent And Observed Capability

The frozen incumbent is `ntuple_phaseblend_expectimax2` composed of:

- parent MC1000 n-tuple table;
- student1 blend weight `0.25` in all phases;
- replay-calibration weight `0.05` from midgame;
- action-label sidecar weight `0.10` in endgame.

Representative normal-start evidence:

| Evidence | Result | Interpretation |
| --- | ---: | --- |
| Best observed game | `263670` | Real normal-start tail result; not a robust mean-performance claim |
| Best score excluding starter | `204621` | Removes the fixed `1536` starter contribution |
| Protected global top three | `263670`, `261369`, `258561` | Strong tail evidence from multiple runs |
| 100-game seeds `1450:1550` | mean excl. starter `20649.81`, median `16342.5`, `P(3072 excl. starter)=0.03` | Representative incumbent block |
| 100-game seeds `1820:1920` | mean excl. starter `22195.74`, median `18237`, `P(3072 excl. starter)=0.04` | Another independent seed block |
| Latest 200-game tail-hunt block | mean excl. starter `18830.76`, median `12069`, `P(3072)=0.025` | Acquisition run, not a promotion comparison |
| Normal-start non-starter `6144` | `0` observed in reported suites | Central unsolved capability gap |

The high score is substantially better than the earliest random, greedy,
expectimax-2, imitation, and simple TD baselines. However, the score distribution
is heavy-tailed, and the current evidence does not show reliable creation of the
next high tile across normal starts.

## Research Program So Far

### 1. Classical Search And Imitation

- Random, greedy, depth-2 expectimax, and a small depth-3 expectimax probe were
  established as baselines.
- Behavior cloning from expectimax improved over greedy with larger datasets,
  but remained weaker than the search teacher.
- More training accuracy did not monotonically improve gameplay; later epochs
  could regress despite better imitation loss.
- Exact deeper expectimax was too slow to use naively for large-scale labels.

Conclusion: imitation is useful for bootstrapping but does not solve policy
improvement when the teacher itself is limited.

### 2. PPO And Direct Sparse RL

- Sparse final-score PPO from scratch was stable initially but did not beat
  greedy and later collapsed toward shorter games.
- The terminal objective is semantically correct but produces weak credit
  assignment for rare high-board transitions.

Conclusion: do not resume generic PPO from scratch without a stronger warm
start, curriculum, or value target.

### 3. N-Tuple TD, Actor Bootstrapping, And Phase Blends

- N-tuple afterstate learning improved survival and provided a cheap learned
  leaf evaluator.
- Training from stronger `corner2` actor trajectories was better than short
  pure self-play.
- Learning rate mattered materially; a cooler MC update outperformed a hotter
  one on the same actor data.
- Student blending, replay calibration, and an endgame action-label sidecar
  produced the current incumbent and strong tail games.

Conclusion: the n-tuple/expectimax hybrid is the best actor available, but its
own trajectories are now a narrow training distribution rather than a reliable
source of further improvement.

### 4. Rare-Event And Support-Ladder Diagnostics

The program mapped a plausible high-board support ladder involving duplicate
or adjacent `384`, a second raw `768`, duplicate `1536`, near-adjacent/adjacent
`1536`, a second `3072`, and eventual `6144`.

Important local observations:

- Once selected states already contain non-near duplicate `1536`, conversion
  to near-adjacent `1536` can be locally easy (`271/396` hits), but that result
  represents only three roots.
- In a pooled continuation diagnostic, adjacent `1536 -> second 3072` and
  `second 3072 -> 6144` were very high-probability transitions, but represented
  only two root ancestries and selected continuation states.
- The difficult part appears upstream: creating source-diverse support states
  and making the correct actions before the board becomes fully congested.

Conclusion: the ladder is a useful mechanism hypothesis, not policy evidence.
High local conversion rates after conditioning on rare success states must not
be presented as normal-start capability.

### 5. Matched Selectors And Direct Rollout Gates

The program progressively tightened methodology from observational rankings to
root-matched controls and finally direct paired endpoint tests.

Key results:

| Experiment | Result | Decision |
| --- | --- | --- |
| Air-survival selector, 9 roots | top-minus-random `+9.72 pp`, CI `[-17.0, +36.5]`, only `4/9` roots positive | Diagnostic only |
| Direct h40 gate, 24 fresh one-`768` roots | `-1.04 pp`, CI `[-2.60, 0.00]` | Failed |
| Pooled direct h40 gate, 48 roots | `+0.78 pp`, CI `[0.00, +2.34]`; only `2/48` actions changed | Failed |

The pooled pilot had only two target-hitting action rows among 185 action rows.
Most source states contained only one raw `768`; the only positive evaluation
root was one of the rare richer-support states.

Conclusion: the first-action decision from mostly one-`768` failure states is
too late or too weakly labeled for the tested selector. The missing support rung
must be generated earlier or supplied by a different source.

### 6. Stochastic MCTS/UCT

A resumable stochastic UCT first-action gate was tested on the same fixed 24
fresh hard roots.

| Search variant | Direct h40 milestone lift | Other effects | Decision |
| --- | ---: | --- | --- |
| 24 simulations, depth 8 pilot | `+4.17 pp`, CI `[0.00, +9.38]` | score CI crossed zero; one seed block zero | Failed pilot |
| Mean-value, 64 simulations, depth 12 | `0.00 pp`, CI `[-2.60, +3.13]` | mean score `-82.5` | Failed |
| Visit-count, 64 simulations, depth 12 | `+0.52 pp`, CI `[-2.08, +3.65]` | score/survival CIs crossed zero | Failed |
| Target-directed, 128 simulations, depth 20 | `-1.56 pp`, CI `[-4.69, +1.04]` | mean score `-520.8`, survival `-6.25 pp` | Failed |

Conclusion: the tested search teacher is not stronger than the incumbent on
this corpus. The pilot effect did not survive stronger evaluation. Further
budget, exploration, or reward sweeps of this same operator are killed.

This does not test a full AlphaZero loop. The experiments used a frozen
incumbent leaf evaluator or a sparse hand-directed target and did not jointly
improve a policy/value model across iterations.

## What We Have Learned

### Reliable Findings

1. The simulator and reproducibility infrastructure are not the limiting
   factors.
2. Search-guided n-tuple learning and phase-specific blending can produce a
   capable actor and rare strong games.
3. The major performance gap is concentrated in rare, congested pre-promotion
   windows rather than ordinary early-game survival.
4. Frame count is a misleading sample-size measure. Root ancestry and policy
   family determine effective sample size.
5. Success-path selection, observational AUC, policy agreement, or local state
   conversion cannot substitute for action-conditioned causal evaluation.
6. Pilot seeds used to select an action must be separate from evaluation seeds;
   otherwise winner's-curse leakage exaggerates lift.
7. Continuation-start evidence is valuable for mechanism discovery but cannot
   justify a normal-start or dashboard promotion.
8. Existing incumbent self-play mostly reproduces incumbent failure modes.
   More of the same data has low expected value.
9. The tested MCTS operator changes actions but does not improve the direct
   endpoint. Search activity is not equivalent to policy improvement.
10. The project needs a stronger improvement operator or a new source of
    frontier states before more fitting is warranted.

### Hypotheses, Not Established Facts

- Rich duplicate/adjacent `384` and second-`768` support may be the missing
  upstream rung.
- Human expert games may contain qualitatively different ways of entering the
  congested level-up window.
- A distributional, horizon-conditioned value model may be more useful than a
  scalar expected-score leaf value.
- An AlphaZero-style iterative loop may work if, and only if, search first
  demonstrates measurable improvement over the actor.

## Where The Program Is Stuck

### Data Support

- High-board successes are rare under normal-start incumbent play.
- Many stored frames descend from a small number of roots.
- The corpus is dominated by one actor family, so nominal source diversity is
  weaker than it appears.
- The state-record dry-run audit found only one behavior family, but a broader
  retained replay inventory found `598` extractable h40 first-non-starter
  `1536` roots across `10` behavior families: `176` success roots and `422`
  matched failures.
- The raw retained replay pile still fails the frozen `50%` family-share rule:
  `phaseblend_incumbent_lineage` contributes `437 / 598` roots. However,
  downsampled retained subsets can satisfy the rule with `0` new roots; the
  largest option uses all non-largest roots plus `161` capped phaseblend roots
  (`322` roots total), and a stricter no-incumbent option uses
  `corner2_lineage` plus `expectimax_baseline` at `49 + 49` roots.
- Top-game and success-window selection introduce selection bias.
- There are currently no imported human/tracker sessions in
  `datasets/human_watch/`.

### Labels And Credit Assignment

- Direct 10/20/40-move milestone events are sparse.
- Short-horizon geometric proxies are easier to label but have not transferred
  reliably to later milestones.
- Recorded-action agreement cannot tell whether another action was better.
- Scalar score targets underrepresent the rare upper tail and can trade away
  survival or milestone probability.

### Policy Improvement

- The current selective-rollout teacher usually ties the incumbent because the
  pilot rarely observes the endpoint.
- The tested MCTS/UCT family is not a better first-action teacher.
- Fitting on unstable labels risks distilling noise or a proxy rather than an
  improved policy.
- No candidate has passed the direct frontier gate required before normal-start
  evaluation.

## Current Kill, Hold, And Promotion Rules

### Killed Unless New Direct Evidence Appears

- `supportstock`, `second768`, and `regen768` archive-objective promotion.
- Adjacent/duplicate-`384` hand-ranked variants on the existing pools.
- Success-path selectors as policy evidence.
- Further downstream ladder chaining from the existing correlated pools.
- The tested h40 selective-rollout gates on mostly one-`768` states.
- The tested incumbent-leaf and target-directed MCTS/UCT configurations.
- More hyperparameter sweeps of the same search/data operator.

### Held

- Broad value fitting, action-prior fitting, sidecars, and capacity probes.
- Generic PPO or imitation scaling without a better teacher.
- Normal-start claims from replay starts or selected continuation states.
- Dashboard promotion from anything other than real normal-start evidence.

### Standing Promotion Ladder

1. Establish provenance-safe, source-diverse roots.
2. Generate or acquire the missing frontier support rung.
3. Run paired action-conditioned 10/20/40-move gates on held-out root
   ancestries.
4. Require positive lift in every independent seed block and a root-cluster
   confidence interval excluding zero.
5. Require no material score or survival regression using predeclared margins.
6. Only then fit a policy/value model.
7. Run paired normal-start evaluation before changing the incumbent.
8. Promote dashboard capability only from normal-start evidence.

## Possible Future Directions

### Direction A: Human/Tracker Frontier Injection

This is the fastest practical next step.

Target intake:

- at least five independent games reaching a non-starter `1536`;
- at least one game reaching `3072`;
- preferably multiple players or policy styles, not repeated attempts from one
  person alone.

The existing pipeline can import `events.jsonl`, create replay artifacts, and
extract 10/20/40-move transition and support-ladder windows. Human actions
should be treated as candidate actions, not ground truth. The causal test must
still compare legal first actions under paired simulator randomness.

Primary risk: five games may still be too little or too homogeneous. Root
ancestry, player identity, and game provenance must remain explicit.

### Direction B: AlphaZero-Style Phase-0 Oracle Gate

This is the most principled self-play-only reboot, but it must not be confused
with another MCTS sweep.

The frozen pre-execution design is now in
`threes_rl/PHASE0_ORACLE_GATE_SPEC.md`. In short:

- one sentinel direct endpoint: h40 creation of the first non-starter `1536`;
- h10/h20, score, and survival reported secondarily on the same roots;
- pilot common-random-number seeds choose and freeze the oracle action;
- fully independent preregistered evaluation seed blocks score the frozen
  action against the incumbent;
- phase 0 uses empirical rollout outcome distributions only, with no learned
  distributional value model;
- score and survival non-inferiority margins are frozen before rollout;
- the old weak "30% changed roots nonnegative" concentration guard is replaced
  by leave-one-root-out and leave-one-behavior-family-out robustness.

The read-only corpus audit is implemented in
`threes_rl.phase0_oracle_corpus_audit`. The first dry run over retained
pre-milestone window artifacts found:

- `23,793` source records from `21` files;
- `1,132` h40 first-non-starter-`1536` candidate records;
- `29` root-capped fresh ancestries;
- `12` success roots and `17` matched-failure roots;
- only `1` behavior family, `phaseblend_incumbent_lineage`, with `100%` of
  roots.

Conclusion: the retained corpus is not ready for phase-0 rollout execution.
It has enough root count and both outcome strata, but it fails behavioral
family diversity. Do not run labels/search until new behavior-family support
is available or the diversity rule is explicitly revised before execution.

### Direction C: Automated Frontier Curriculum

If human data is unavailable, build a source-diverse self-play curriculum rather
than collecting more undifferentiated incumbent games:

- retain at most one 10/20/40 window per milestone and game ancestry;
- include near-miss and death windows, not only successes;
- generate games from multiple policy snapshots or deliberately perturbed
  policy families;
- use exploration or quality-diversity objectives to reach distinct support
  geometries;
- preserve a substantial normal-start fraction so continuation competence does
  not replace end-to-end competence;
- keep curriculum starts out of the final normal-start test set.

Primary risk: restart-state curricula can create a solver that is excellent at
continuations but cannot reach those states itself.

### Direction D: Distributional Multi-Horizon Value Learning

Only after a teacher or oracle passes a direct gate, fit value targets that
match the actual decision problem:

- score distribution or quantiles, not only expected score;
- survival probability;
- `P(milestone within 10/20/40 moves)`;
- optional terminal or next-major-tile hazards;
- action-value uncertainty for search allocation.

Train/validation/test splits must be grouped by original game ancestry and,
where possible, policy family. Evaluate calibration and action ranking, but use
paired simulator rollouts as the promotion endpoint.

### Direction E: Alternative Planning Operators

The failure of one UCT family leaves room for genuinely different operators:

- chance-sampled expectimax with sequential racing across legal actions;
- best-first or beam search over stochastic outcomes;
- cross-entropy or quality-diversity search over short action sequences;
- model-predictive control with a calibrated distributional leaf value;
- a Gumbel/regularized policy-improvement operator once a policy/value model
  exists.

Each should face the same small direct endpoint gate before broad training.
Novelty in implementation is not enough; the teacher must beat the incumbent.

## Recommended Immediate Decision

Do not resume broad training yet.

The highest-signal immediate work is to choose between:

- collecting/importing source-diverse human frontier games; or
- specifying and auditing the phase-0 empirical oracle corpus before running
  any expensive search.

The first executable experiment should be small enough to kill quickly and
strong enough to answer one question: can a new teacher reliably select a
better first action in the h40 pre-`1536` window on held-out ancestries?

## Questions For The External Reviewer

Please opine specifically on:

1. Is the diagnosis correct that the limiting factor is the policy-improvement
   operator and frontier-state support, rather than model capacity?
2. Is h40 first non-starter `1536` the right sentinel endpoint, or should the
   first oracle gate target an earlier support milestone?
3. What stochastic planning operator is most likely to outperform the current
   actor under a bounded compute budget?
4. How should common randomness be coupled across actions when legal spawn
   locations diverge after the first move?
5. What score and survival non-inferiority margins should be preregistered?
6. Is 20 independent ancestries sufficient for a phase-0 kill decision, and
   what sequential testing design would reduce compute without bias?
7. How would you create a self-play-only frontier curriculum without producing
   continuation-state overfitting?
8. When should a distributional value model be introduced, and what target
   parameterization would best handle the heavy score tail?
9. Which killed branches, if any, deserve reconsideration only after new roots
   or a stronger teacher exist?
10. What is the single experiment with the best expected information gain per
    unit compute?

## Repository Guide

Read these files in this order:

1. `threes_rl/RESEARCH_PROGRAM_HANDOFF.md` - this summary.
2. `threes_rl/CURRENT_DECISION_LEDGER.md` - authoritative branch status.
3. `threes_rl/EXPERIMENT_LOG.md` - full chronological evidence trail.
4. `threes_rl/current_incumbent_policy.txt` - exact incumbent composition.
5. `threes_rl/RESULTS.md` - raw evaluation records.
6. `threes_rl/ML_FINDINGS.md` - earlier research findings; useful but partly
   superseded by the ledger and later experiment log.
7. `RL_SPEC.md` - environment and objective specification.
8. `threes_rl/SETUP.md` - commands and operational handoff.
9. `threes_rl/ARTIFACT_RETENTION.md` - what was retained or pruned.

Key implementation modules:

- `threes_rl/sim.py`
- `threes_rl/eval.py`
- `threes_rl/expectimax.py`
- `threes_rl/ntuple.py`
- `threes_rl/selective_rollout_gate.py`
- `threes_rl/mcts_rollout_gate.py`
- `threes_rl/human_diagnostics_pipeline.py`
- `threes_rl/transition_window_reservoir.py`
- `threes_rl/support_ladder_window_reservoir.py`

Key retained artifacts:

- current dashboard: `threes_rl/runs/dashboard/index.html`
- protected top-three replays: `threes_rl/runs/replays/top3/index.html`
- direct pooled gate report under
  `threes_rl/runs/forensics/selective_rollout_gate/`
- MCTS gate reports under `threes_rl/runs/forensics/mcts_rollout_gate/`
- human inbox: `datasets/human_watch/`

## Final Perspective

The program has made real progress: it moved from a simulator and weak learned
baselines to a strong hybrid actor and, more importantly, developed enough
methodological discipline to reject attractive but non-causal frontier signals.
The current pause is not evidence that Threes cannot be solved by self-play. It
is evidence that the present actor, data distribution, and search teacher do not
form a reliable policy-improvement loop.

The next phase should be judged by information gain, not experiment count. A
new branch earns compute only by introducing genuinely new frontier support or
by proving that its teacher selects better actions under leakage-safe paired
evaluation.

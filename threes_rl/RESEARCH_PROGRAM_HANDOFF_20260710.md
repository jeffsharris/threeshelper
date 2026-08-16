# Threes Solver Research Program: Current Handoff

Status date: 2026-07-10

Purpose: give an external ML researcher or agent enough context to critique the
program and recommend the next experiment without needing the full repository
history. The detailed source of truth is `EXPERIMENT_LOG.md`; frozen decisions
are summarized in `CURRENT_DECISION_LEDGER.md`.

## Executive Summary

The project has a high-confidence Threes simulator, a capable
n-tuple/expectimax incumbent, reproducible paired evaluation, replay and
provenance tracking, and unusually strict promotion hygiene. Engineering is not
the current bottleneck.

The incumbent's best eligible normal-start score is `263,670` (`204,621`
excluding the fixed starter tile). No reported normal-start suite has produced
a non-starter `6144`. The central gameplay weakness remains the congested
10/20/40-move window before creating the next high tile.

The latest research branch tested literature-inspired multi-stage n-tuple TD
with restart-distribution training. A conservative residual version, R1b,
looked strongly positive on a 512-game development block, D2, but failed to
confirm on one untouched 512-game block, C. Its confirmation point estimate was
positive, but the paired confidence interval crossed zero and P3072 was exactly
tied. R1b is therefore not promoted. C is spent and must not be rerun or used
for tuning.

Current decision: `CONFIRMATION FAILED / NO PROMOTION / HOLD FOR REVIEW`.

The most useful next task is not more training. It is to decide which genuinely
new hypothesis has the best expected information gain:

1. preview/tile-cycle-aware staged value learning;
2. a stronger adaptive expectimax teacher after improving the leaf;
3. source-diverse frontier/restart generation, including human games or a
   novelty archive;
4. a richer value target, such as distributional score or horizon-conditioned
   milestone hazards.

## Objective And Evaluation Semantics

The objective is superhuman normal-start play and ultimately maximum final
score, not performance on selected continuation states.

Important game-specific details:

- The simulated board starts with a fixed `1536` tile in the top-left.
- Score and maximum tile must be reported both raw and excluding that starter.
- Raw `max_tile >= 1536` is not a valid milestone because it is true at move
  zero. Use `max_tile_excl_starter` or an explicit tile-count/geometry event.
- Continuation starts, replay starts, selected high-board roots, and synthetic
  descendants are training or diagnostic tools. They are not normal-start
  capability evidence.
- High score is tail evidence, not a promotion endpoint. Primary decisions use
  paired normal-start score with survival, lower-tail, milestone, and board
  geometry safeguards.
- The user's central concern is accurate play during the congested 10, 20, and
  40 moves before creating the next high tile, such as `1536 -> 3072`.

## Simulator And Research Infrastructure

The simulator is mature enough to support research claims:

- Move mechanics matched a prior move oracle on 20,000 random plus adversarial
  boards with zero disagreements.
- 50,000 non-terminal transitions passed tracker validation.
- The `TileCycle` schedule matched the tracker in lock-step tests.
- 270/270 observed single-step moves replayed exactly.
- Seeded environments are deterministic under the same action sequence.
- The `6144 + 6144 -> 12288` rule and scoring are implemented.
- Evaluation supports split deck/slot/policy RNG streams, paired logical seeds,
  frozen manifests, replay capture, milestone retention, and tail audits.
- Restart data records canonical ancestry, source policy family, root frame,
  preview, tile-cycle state, and provenance.
- The dashboard separates record-eligible normal-start results from training,
  development, diagnostic, continuation, and failed-confirmation results.

Residual simulator risk is concentrated in rare tracker-derived schedule edge
cases, not ordinary move mechanics.

## Current Incumbent

The frozen incumbent is `ntuple_phaseblend_expectimax2` with:

- parent MC1000 n-tuple value;
- student1 sidecar at weight `0.25` in all phases;
- replay-calibration sidecar at weight `0.05` from midgame;
- action-label sidecar at weight `0.10` in endgame;
- depth-2 expectimax action selection.

Representative capability:

| Measure | Result |
| --- | ---: |
| Best eligible normal-start score | `263,670` |
| Best excluding starter | `204,621` |
| Protected top three | `263,670`, `261,369`, `258,561` |
| Typical P(non-starter 3072) | roughly `2.5%` to `4%` across cited blocks |
| Confirmed normal-start non-starter 6144 | `0` |

The distribution is very heavy-tailed. Small blocks can be dominated by a few
successful 3072 games, which is why paired continuous endpoints and untouched
confirmation are essential.

## Research Arc

### Early Baselines

- Random, greedy, depth-2 expectimax, and small depth-3 probes established the
  initial ladder.
- Behavior cloning improved over greedy but remained below its search teacher.
- Better imitation loss did not monotonically improve gameplay.
- Sparse final-score PPO from scratch did not beat greedy and later collapsed
  toward shorter games.

Interpretation: sparse end rewards and a limited teacher do not provide enough
credit assignment for rare high-board transitions.

### N-Tuple TD And Hybrid Search

- N-tuple afterstate learning materially improved survival and supplied a
  cheap leaf value.
- Stronger actor trajectories, lower learning rates, phase blends, replay
  calibration, and a sparse endgame sidecar produced the incumbent.
- Incumbent self-play eventually became a narrow data distribution that mostly
  reproduces incumbent strengths and failure modes.

Interpretation: the hybrid architecture works, but naive self-distillation has
reached diminishing returns.

### Frontier And Level-Up Diagnostics

The program mapped a plausible support ladder involving duplicate/adjacent
`384`, a second `768`, duplicate `1536`, adjacent `1536`, a second `3072`, and
eventual `6144`.

Selected continuation states sometimes showed high downstream conversion, but
those results came from very few root ancestries. Direct first-action h40 gates
on broader roots were neutral or negative. A pooled gate produced only
`+0.78 pp`, CI `[0.00,+2.34]`, with only two changed actions among 48 roots.

Interpretation: the hard part is reaching source-diverse support states and
making earlier congestion-management decisions. Conditioning on rare success
states makes the downstream problem look easier than normal-start play.

### MCTS/UCT Branch

Stochastic UCT variants at 64 to 128 simulations did not produce a stable
action advantage over the incumbent. The largest pilot effect disappeared
under stronger evaluation, and a target-directed variant regressed score and
survival.

Decision: further sweeps of that exact low-budget UCT operator are killed.
This does not prove that AlphaZero-style improvement is impossible; it shows
that this search teacher was not stronger than the actor.

## Value-First Restart Program

An external review identified three underused ideas from the 2048/Threes
literature:

1. separate stage tables rather than one blended table;
2. restart-distribution TD for late-stage value support;
3. deeper adaptive expectimax rather than low-budget UCT.

It also identified a partial-observability gap: search conditions on preview
and cycle state, but the learned n-tuple leaf is board-only, and historical TD
transition generation used `include_next_preview=False`.

### R0: Infrastructure

R0 passed:

- true per-feature stage weight promotion;
- ancestry-balanced restart sampling;
- split deck/slot/policy RNG streams;
- frozen paired normal-start development and confirmation blocks.

These changes are retained.

### Original R1: Bare Multi-Stage Parent

R1 used four stage tables with the parent MC1000 value as the stage-0 prior.
It was stopped at 1,000 episodes because it remained far below the full blended
incumbent on D0.

- Parent gap versus incumbent: `-8,448.84` mean score excluding starter.
- Training recovered `+1,048.08` versus the parent.
- Candidate still trailed the incumbent by `-7,400.77`.

Interpretation: this was primarily a comparator-design failure. The candidate
discarded valuable incumbent components before learning enough to replace
them. Original R1 is permanently killed; the broader multi-stage hypothesis is
not disproven.

### R1b: Frozen Incumbent Plus Stage Residual

R1b froze the complete incumbent leaf and learned a zero-initialized four-stage
residual under fixed incumbent trajectories.

Training specification:

- exact 50/50 normal-start and ancestry-balanced restart starts;
- default n-tuple patterns;
- temporal coherence;
- fixed incumbent trajectory actor;
- 5,000 episodes;
- no side experiments or hyperparameter sweeps.

D1 at 1,000 episodes was directionally positive but uncertain:

- paired score `+1,425.98`, CI `[-4,051.67,+6,731.73]`;
- P3072 tied;
- positive median and lower-decile point estimates.

D2 at 5,000 episodes looked strongly positive on 512 paired normal starts:

- paired mean score `+4,506.57`, CI `[+1,176.73,+7,894.16]`;
- median and moves had positive intervals;
- P3072 `+2.73 pp`, CI `[+0.20,+5.47]`;
- no material lower-tail or corner block.

This made R1b a promotion candidate, not an incumbent.

### Clean Pre-C Level-Up Diagnostic

A frozen, outcome-independent diagnostic used 21 restart-training-unsampled
ancestries and paired h10/h20/h40 rollouts.

- h10 first-1536: tied `1/336` per arm.
- h20: `9/336` incumbent versus `12/336` R1b, `+0.89 pp`, CI
  `[-0.60,+2.38]`.
- h40: tied `30/336` per arm.
- h40 score difference `-324.08`, CI `[-2,495.47,+1,761.65]`.
- h40 survival `+0.60 pp`, CI `[-5.95,+6.25]`.

Decision: neutral/no block. This did not independently validate improved
pre-promotion play. The corpus was small and 19/21 roots came from the
phaseblend lineage.

## Final Sealed Confirmation

Exactly one evaluation was run on untouched C with 512 paired normal starts.
No retraining, partial peeking, repeated C run, or configuration change
occurred. Manifest, stream, checkpoint, and incumbent hashes matched preflight.

| Policy | Mean excl. starter | Median | P3072 | Mean moves | High excl. starter |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent | `21,131.63` | `12,144` | `4.10%` | `186.46` | `204,573` |
| R1b | `21,919.80` | `12,597` | `4.10%` | `190.35` | `206,679` |

Paired results:

- mean score `+788.18`, 95% CI `[-2,412.96,+4,021.11]`;
- median `+453`, CI `[-2,208.04,+5,781.19]`;
- lower decile `-113.7`, CI `[-1,593.66,+1,467.09]`;
- moves `+3.89`, CI `[-4.70,+12.62]`;
- P3072 exactly `0.00 pp`, CI `[-2.34,+2.34]`, with `21/512` in both arms;
- 265 candidate wins, 247 losses;
- symmetric trimming kept the mean positive, ranging from `+402` to `+793`;
- tail and corner safeguards passed.

The preregistered primary rule required the paired score CI to exclude zero.
It did not. Therefore:

`CONFIRMATION FAILED / NO PROMOTION / HOLD FOR REVIEW`.

The raw candidate high score of `265,728` is diagnostic only and is explicitly
ineligible for the dashboard record. The eligible record remains `263,670`.

## Scientific Interpretation Of R1b

The defensible interpretation is narrower than either "R1b works" or "staged
TD does not work":

- R1b may have a small positive average effect; most robust point estimates
  are positive.
- The effect was not large or stable enough to meet the frozen confirmation
  rule.
- D2's strong P3072 and score lift did not replicate on C.
- The D2-to-C attenuation could be ordinary heavy-tail variance, block
  heterogeneity, development winner's curse, or a real state-distribution
  interaction.
- C cannot now be used to choose hyperparameters, subgroups, or a revised
  endpoint. Any analysis of C is explanatory and hypothesis-generating only.
- This result rejects promotion of this exact checkpoint. It does not reject
  multi-stage TD, restart training, or context-aware value learning as classes.

## Reliable Lessons

1. The simulator and reproducibility substrate are not the limiting factor.
2. Search-guided n-tuple learning can produce a capable actor and rare strong
   games.
3. The remaining gap is concentrated in rare congested level-up windows.
4. Frames are not independent samples; root ancestry and policy family govern
   effective sample size.
5. Observational AUC, action agreement, selected success windows, and local
   continuation conversion do not establish action advantage.
6. Shared exogenous streams materially improve paired evaluation, but score
   remains heavy-tailed because trajectories diverge.
7. Replay/restart states are legitimate for value training but not for
   normal-start promotion claims.
8. Low-budget UCT was not a stronger teacher than the incumbent.
9. A strong development result can disappear on an untouched block. Promotion
   discipline prevented a false incumbent update here.
10. The learned leaf ignores preview/cycle information available to search;
    this remains a concrete representational hypothesis, not yet a result.

## Where The Program Is Stuck

### Effect Size And Variance

Useful changes may be only one to three percentage points per rare decision,
while final score is dominated by a small number of high-tile trajectories.
Even 512 paired games left a wide score interval.

### High-Board Training Support

Normal-start incumbent play rarely reaches the decisive windows. Restart
training supplies states, but many roots still share ancestry and policy
family. The clean pre-C corpus was especially lineage-concentrated.

### State Representation

Board-only n-tuples average over preview and deck-cycle information that the
agent actually observes. A value function can therefore assign the same leaf
value to states with different near-term tile hazards.

### Improvement Operator

The incumbent is stronger than simple imitation, generic PPO, the bare staged
parent, and tested low-budget UCT. The program lacks a teacher or target that
has repeatedly demonstrated causal action improvement in the congested window.

### End-To-End Transfer

Local continuation success and development-block gains have not reliably
translated into confirmed normal-start improvement.

## Branches That Should Remain Killed Or Held

- Generic PPO from scratch with sparse terminal reward.
- More sweeps of the tested 64-128 simulation UCT variants.
- Rerunning original R1's bare-parent comparator.
- Rerunning or extending sealed C.
- Promoting R1b from its isolated high score or positive point estimate.
- Broad policy fitting from sparse, correlated milestone labels.
- Dashboard promotion from training, restart, continuation, diagnostic,
  development-only, or failed-confirmation games.

## Credible Future Directions

### 1. Existing-Output D2 Versus C Mechanism Audit

Use only already generated artifacts to characterize, not salvage, R1b:

- compare paired-difference distributions and concentration by seed/block;
- quantify how much each result depends on P3072 gains and losses;
- compare baseline-difficulty strata without selecting a favorable subgroup;
- inspect the frozen tail cases for recurring stage or geometry differences;
- document hypotheses before choosing the next candidate.

This audit must not produce a retroactive promotion rule or tune against C.

### 2. R1.5: Preview/Tile-Cycle-Aware Value

Test the clearest representational gap as a new preregistered candidate:

- preserve the full incumbent baseline;
- condition a small residual on preview category and coarse cycle position;
- generate correct next-preview targets rather than silently marginalizing;
- use stage/ancestry-balanced restart training;
- establish exact zero-residual identity before training;
- evaluate on fresh development streams, never C.

This is the current highest-priority model hypothesis because it uses
information already observed by the search policy and directly targets leaf
aliasing.

### 3. Better Multi-Stage Representation

The four-stage residual may be too coarse. Candidate ideas include stage
boundaries tied to support geometry as well as max tile, more expressive tuple
patterns for congested boards, or separate residual capacity for the 40-move
pre-promotion regime. Any added capacity needs stage coverage and promoted-
feature audits.

### 4. Adaptive Expectimax

Once a leaf improves on fresh normal-start evidence, use node- or time-budgeted
adaptive expectimax:

- deepen when empty count is low or top-tile support is congested;
- use transposition caching and controlled chance sampling;
- compare actions with shared deck streams on held-out roots;
- prove the teacher improves score/hazard before distillation.

Do not make deeper search the default while its leaf remains unvalidated.

### 5. Distributional Or Horizon-Conditioned Targets

A scalar expected-score leaf may trade away rare milestone probability. A
compact alternative is a shared representation with heads for score
distribution, survival, and 10/20/40-move milestone hazard. Promotion should
still use simulator rollouts, not head accuracy.

### 6. Source-Diverse Frontier Generation

Use a novelty archive or Go-Explore-style restart pool keyed by board signature,
support geometry, preview, and cycle position. Human/tracker games are valuable
as independent roots, but primarily for state support and evaluation seeding,
not imitation labels. Do not gate all progress on human data collection.

## Recommended Next Decision Sequence

1. Keep R1b unpromoted and preserve its checkpoint and all C evidence.
2. Perform one lightweight, preregistered-as-descriptive D2/C attenuation audit
   using existing outputs only.
3. Choose exactly one new hypothesis. Current recommendation: context-aware
   staged residual value, not another search or hyperparameter sweep.
4. Freeze fresh development and confirmation streams before training.
5. Use a continuous paired score endpoint with P3072, lower-tail, and
   corner/geometry safeguards.
6. Keep the difficult h10/h20/h40 pre-promotion behavior as a diagnostic, but
   do not require a tiny binary root gate to detect all useful effects.
7. Promote only after untouched normal-start confirmation.
8. If the context-aware leaf is neutral, move to source-diverse novelty
   restarts or richer value targets before revisiting adaptive search.

## Questions For The Reviewing Agent

1. What is the most plausible explanation for D2's strong result and C's weak
   positive result, given the paired outcomes and heavy tail?
2. Is preview/cycle-conditioned residual value the highest-information next
   experiment, or is another bottleneck more likely?
3. What is the smallest context representation that avoids leaf aliasing
   without exploding n-tuple sample complexity?
4. Should stage boundaries remain max-tile based, or incorporate support
   geometry and congestion?
5. What evaluation design has enough power for a plausible 1-3% improvement
   without consuming another large fixed block unnecessarily?
6. Should the next primary endpoint remain paired mean score, use a robust or
   transformed score, or combine score with a preregistered milestone-hazard
   integral?
7. At what point should adaptive expectimax or an AlphaZero-style loop be
   reopened?
8. What evidence would falsify the value-representation hypothesis quickly?

## Key Artifacts

- `threes_rl/EXPERIMENT_LOG.md`: full chronological record.
- `threes_rl/CURRENT_DECISION_LEDGER.md`: frozen decisions and active holds.
- `threes_rl/RESEARCH_RESTART_PROPOSAL.md`: value-first restart rationale.
- `threes_rl/R1B_PREREGISTRATION.md`: R1b specification.
- `threes_rl/R1B_PRE_C_DIAGNOSTIC_SPEC.md`: level-up diagnostic rules.
- `threes_rl/R1B_PRE_C_DECISION.md`: pre-confirmation decision.
- `threes_rl/runs/eval_artifacts/r1b_confirmation_candidate_5000_c_20260710/confirmation_lock.json`:
  final frozen decision.
- `threes_rl/runs/eval_artifacts/r1b_confirmation_candidate_5000_c_20260710/paired_vs_incumbent.json`:
  C paired metrics.
- `threes_rl/runs/eval_artifacts/r1b_confirmation_candidate_5000_c_20260710/tail_sensitivity.json`:
  robust tail checks.
- `threes_rl/runs/eval_artifacts/r1b_confirmation_candidate_5000_c_20260710/tail_audit/summary.json`:
  corner/tail replay audit.

## Operational Status

- Research is held; no training or evaluation job is active.
- Dashboard generation is live. A flat chart means no new research artifacts
  were produced during the hold, not that the watcher stopped.
- Eligible dashboard record: `263,670`.
- Protected top-three scores: `263,670`, `261,369`, `258,561`.
- Run artifacts occupy about `33 GiB`; approximately `146 GiB` remains free.
- Preserve the incumbent, R1b checkpoint, C evidence, provenance sources, and
  protected top-three replays during cleanup.

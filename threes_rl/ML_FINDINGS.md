# Threes RL Research Findings And Next Directions

This report summarizes the current RL work so a stronger model or a future
agent can pick up the research without rediscovering the same facts. The short
version: the simulator appears solid, behavior cloning from expectimax2 gives a
real but modest learned-policy gain, sparse PPO from scratch is not yet useful,
and the next meaningful work should focus on exact-model search plus learned
afterstate values rather than simply adding RAM.

## 2026-07-05 Update: Research Plan V2 Supersedes The Imitation-First Plan

`threes_rl/RESEARCH_PLAN_V2.md` reframed the problem around the key corrected
metric: score excluding the free 1536 starter tile and max tile excluding that
starter. On that metric, imitation from `expectimax2` is not merely below human
level; the teacher itself almost never builds a meaningful high tile beyond the
starter. The old DAgger recommendation below is therefore historical context,
not the current first choice.

Current best corrected-start baseline:

- `corner2`, seeds 1000:1200: mean score 74323.59, mean score-minus-starter
  15274.59, mean moves 163.785.
- It built a non-starter 3072 in 3/200 games and reached non-starter 6144 in
  0/200.
- Top replays and chart:
  `threes_rl/runs/eval_artifacts/corner2_1000_1200_fixed_starter/`.

TD n-tuple status:

- `threes_rl/ntuple.py` and `threes_rl/train_td.py` now implement an afterstate
  value table, exact expected-spawn action targets via
  `ThreesSim.transition_outcomes()`, checkpointing, progress charts, and
  top-three replay retention.
- Pure self-play smoke:
  `td_default_expected_500_init3000_a005_20260705`: held-out seeds 1000:1050
  reached mean score 65842.98, mean score-minus-starter 6793.98, mean moves
  100.44, high score 88443, and no non-starter 1536s.
- Learned-leaf search:
  `ntuple_expectimax2:td_default_expected_500_init3000.../latest` on seeds
  1000:1010 reached mean score-minus-starter 9837 and high score 81852. Search
  helps the value table, but the self-play table is still weak.
- Corner2 Monte Carlo bootstrap:
  `td_default_corner2_mc_50_init3000_a005_20260705` trains the n-tuple table on
  actual remaining returns from 50 `corner2` actor games. The training run
  retained a 194271-score / non-starter-3072 replay. The resulting
  `ntuple_expectimax2` probe on seeds 1000:1010 reached mean
  score-minus-starter 13832.7 and high score 87888.
- Larger corner2-MC bootstrap:
  200 actor games with `alpha=0.05` made learned-search worse, but the same 200
  actor games with `alpha=0.01` produced the current best learned-search probe:
  seeds 1000:1050 mean score 77139.78, mean score-minus-starter 18090.78, high
  score 205719, and 2/50 held-out games with a non-starter 3072. The full
  200-seed combined eval settled to mean score-minus-starter 15251.805, high
  score 205719, and 3/200 non-starter 3072 games, effectively tying `corner2`
  rather than clearly beating it.
- Interpretation: bootstrapping from strong trajectories is the right direction,
  but update rate matters and the current n-tuple/search setup has not yet
  broken past the `corner2` ceiling. The next likely upgrade is either a larger
  lower-alpha corner2-MC value dataset with better weighting for rare high-return
  trajectories, or search profiling/caching so adaptive depth becomes feasible.

## Current State

The repository now contains a pure simulated Threes environment:

```text
threes_rl/sim.py              exact game simulator, NumPy + stdlib
threes_rl/env.py              Gymnasium wrapper
threes_rl/obs.py              observation encoders
threes_rl/baselines.py        random and greedy baselines
threes_rl/expectimax.py       bag-aware depth-limited expectimax
threes_rl/train_imitation.py  behavior cloning from expert policies
threes_rl/train_ppo.py        masked PPO
threes_rl/eval.py             deterministic evaluation harness
threes_rl/bench.py            throughput benchmark
```

The simulator does not use screen capture or iPhone Mirroring. It tracks the
internal board, preview, bag, bonus schedule, legal moves, random spawn slot,
and final scoring directly. This is the right architecture for fast training.

The current local best learned checkpoint is:

```text
threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/checkpoint_epoch_25.pt
```

The dataset used to train the current local best is generated locally:

```text
threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/dataset.npz
```

The earlier 200k dataset/checkpoint baseline is committed:

```text
threes_rl/runs/imitation_expectimax2_200k_w8_e30/dataset.npz
threes_rl/runs/imitation_expectimax2_200k_w8_e30/checkpoint_epoch_20.pt
```

## Rule And Simulator Confidence

The simulator has high confidence on the known game rules:

- Move mechanics match `state_hunt.simulate_base_move` on 20,000 random boards
  plus adversarial cases in all four directions.
- Simulated transitions validated against the tracker for 50,000 non-terminal
  random steps.
- The tile schedule matches `window_stream.TileCycle` in lock-step tests.
- 270/270 replayable single-step observed moves were reproduced exactly.
- Gymnasium API checks pass.
- Two seeded envs following the same random legal policy match for 1,000
  steps.
- The terminal merge rule is implemented: `6144 + 6144 -> 12288`, no spawn,
  immediate game over.
- `score_tile(12288) == 1594323`.

Residual simulator risks:

- The fresh-game start model is empirical. It uses exactly 8 small board tiles
  plus one `starter_tile`, defaulting to 1536 because that was observed on the
  recorded device. The starter tile is fixed in the top-left corner, and the 8
  small board tiles are sampled uniformly among the remaining cells.
- The bonus tile schedule is modeled from the tracker reverse-engineering and
  verified against `TileCycle`; if the tracker model is wrong in a rare edge
  case, the RL simulator will inherit that error.

## Objective And Scoring

The training and evaluation objective should remain final board score.

Important scoring facts:

- Empty, 1, and 2 tiles score 0.
- 3 scores 3, 6 scores 9, 12 scores 27, and so on.
- 6144 scores 531441.
- 12288 scores 1594323.
- Final score is the sum of tile scores, not the sum of face values.
- A policy may rationally delay merging two 6144 tiles if it can increase the
  rest of the board first, because the terminal merge ends the game.

The current reward mode defaults to sparse final score:

```text
0 for non-terminal legal moves
final board score at terminal state
```

That is semantically correct for the real objective, but it is hard for PPO
from scratch.

## Baselines And Results

Fixed 200-seed eval suite: seeds `1000:1200`, default starter tile 1536.

| Policy | Mean Score | Median | P90 | Mean Moves |
| --- | ---: | ---: | ---: | ---: |
| random | 59600.21 | 59220.00 | 60009 | 30.40 |
| greedy | 60304.29 | 59488.50 | 61677 | 40.44 |
| expectimax2 | 66460.14 | 65814.00 | 73905 | 107.95 |
| expectimax3 probe, 5 seeds | 76954.80 | 79929.00 | 81306 | 142.20 |
| imitation expectimax2 30k | 60848.28 | 59610.00 | 65838 | 47.37 |
| imitation expectimax2 200k epoch 20 | 61826.34 | 61048.50 | 66237 | 60.81 |
| imitation expectimax2 400k epoch 25 | 62154.87 | 61236.00 | 66786 | 63.42 |

Wider 1000-seed sanity check, seeds `1000:2000`:

| Policy | Mean Score | Median | P90 | Mean Moves |
| --- | ---: | ---: | ---: | ---: |
| greedy | 60224.63 | 59475.00 | 61644 | 39.02 |
| imitation expectimax2 200k epoch 20 | 61489.74 | 60354.00 | 66072 | 59.97 |
| imitation expectimax2 400k epoch 25 | 61814.26 | 60661.50 | 66573 | 62.12 |

Interpretation:

- The learned policy is now reliably better than greedy.
- It is still materially weaker than expectimax2.
- Expectimax3 looks much stronger, but exact eval measured roughly 40 seconds
  per game on the current machine, so it is too slow to use naively.
- Current scores are heavily influenced by the 1536 starter tile. Max-tile
  threshold metrics up through `>=1536` are therefore not informative.
- The listed learned-policy checkpoints were trained before the starter tile
  was fixed to the top-left corner, so treat them as pre-fix baselines until
  retrained.

## Training Findings

### Behavior cloning works, but only partially

Behavior cloning from expectimax2 produced the best learned policy so far.

Observed progression:

- Expectimax2 imitation, 10k samples: 59.60 percent training accuracy,
  60221.66 mean score.
- Expectimax2 imitation, 30k samples: 62.06 percent training accuracy,
  60848.28 mean score.
- Expectimax2 imitation, 100k samples, 30 epochs: 61385.97 mean score.
- Expectimax2 imitation, 100k samples, 60 epochs: training accuracy rose to
  about 90 percent, but eval regressed to 60896.19 mean score.
- Expectimax2 imitation, 200k samples, epoch 20: best current learned score,
  61826.34 mean score.
- Expectimax2 imitation, 400k samples, epoch 25: current local best learned
  score, 62154.87 mean score.
- Expectimax2 imitation, 200k samples, epoch 30: lower than epoch 20.

Takeaway: more supervised accuracy is not the same as better gameplay. The
agent likely suffers from behavior-cloning distribution shift. Once it takes a
suboptimal move, it reaches board states that are underrepresented in the
expert dataset, and local expert-action accuracy no longer guarantees good
rollout value.

### Sparse PPO from scratch is not currently promising

Sparse final-score PPO from scratch was stable initially but did not beat
greedy. Longer resume collapsed toward shorter games. PPO fine-tuning from an
imitation checkpoint was stable but did not beat the best imitation-only model
in the experiments run so far.

This does not mean PPO is impossible. It means the current PPO setup is not
yet the highest-leverage path. Likely missing pieces:

- stronger warm starts
- better reward shaping or value targets
- curriculum
- more robust policy architecture
- offline/on-policy data aggregation

### Label generation is CPU-bound

The expensive part is expectimax label generation, not neural network training
and not RAM.

Measured throughput:

- Serial expectimax2 labels: roughly 82-88 samples/sec.
- Parallel expectimax2 labels with 8 workers: roughly 470-517 samples/sec
  after warmup.

The 200k dataset was only about 5.5 MB compressed. The best checkpoint is
about 4.7 MB. RAM is not currently the limiting resource.

Use the more capable Mac if it has more or faster CPU cores. Extra RAM alone
will not materially speed up the current pipeline.

## Why I Think The Current Policy Is Limited

The learned policy is a feed-forward imitation model trained on expert states.
It does not search at inference time and does not explicitly estimate future
chance outcomes. Expectimax2, by contrast, uses the exact simulator, chance
nodes, and a hand-built leaf evaluator at every decision.

Failure modes implied by the results:

- The policy imitates individual actions better with more epochs, but rollout
  value can fall. This is classic covariate shift.
- The policy may learn shallow correlations from the expert dataset without
  preserving the long-horizon board-shaping strategy.
- The current observation is rich enough for Markov state, but the MLP may be
  too weak or poorly structured to capture board geometry and merge planning.
- The expert itself is imperfect: expectimax2 is much better than greedy, but
  the expectimax3 probe suggests depth matters a lot.
- The eval suite currently starts with a 1536 tile, so the early game is not
  the hard part. The policy needs to preserve structure around high-value
  tiles and avoid late-board traps.

## Highest-Leverage Next Experiments

### 1. DAgger-style data aggregation

This is probably the best next step.

Current behavior cloning trains on states visited by the expert. Instead:

1. Start with the best imitation checkpoint.
2. Roll it out for many games.
3. At each visited state, query expectimax2 for the expert action.
4. Add those states to the dataset.
5. Retrain or fine-tune.
6. Repeat for several rounds.

Why this should help:

- It directly addresses distribution shift.
- It labels the model's own mistake states.
- It is cheaper than jumping straight to expectimax3 everywhere.

Implementation sketch:

```text
train_dagger.py
  load policy checkpoint
  for each round:
    collect N states from policy rollouts
    label actions with expectimax2
    append to dataset.npz
    train imitation model for M epochs
    eval checkpoints on seeds 1000:1200
```

Important detail: collect states across the full game, not just early moves.
The current learned policy improves mean moves, but late-game decisions matter
most for score.

### 2. Mix expert-state and policy-state data

Pure DAgger data may overrepresent bad states if the policy is weak. A robust
mix:

```text
50 percent existing expectimax2 states
50 percent policy-rollout states labeled by expectimax2
```

Track eval after each round rather than relying on training accuracy.

### 3. Use expectimax3 selectively

Full expectimax3 labeling is expensive, but it may be useful selectively:

- Label only high-value or late-game states with expectimax3.
- Label only states where expectimax2 and the current policy disagree.
- Label only states near low-empty-cell danger zones.
- Generate a smaller "gold" validation/eval set of expectimax3 decisions.

This can improve target quality without making all data generation 20x slower.

### 4. Improve the policy architecture

The current network is a plain MLP:

```text
obs -> Linear(282, 512) -> ReLU -> Linear(512, 512) -> ReLU -> policy/value
```

Possible improvements:

- Encode the 4x4 board with a small convolutional or residual network.
- Use separate embeddings for tile ranks instead of raw normalized values.
- Preserve spatial structure rather than flattening everything immediately.
- Add explicit feature channels for legal moves, preview type, bag counts, and
  bonus candidates.
- Train a value head against expectimax leaf value or rollout return.

The observation contains enough state, but the architecture may not be making
the right inductive bias easy.

### 5. Train value estimation, not only action imitation

Expert actions are useful but lossy. A value model could be trained from:

- final returns of expert rollouts
- expectimax2 state values
- expectimax3 values on a smaller subset
- Monte Carlo returns from policy rollouts

Then use:

- policy + value auxiliary loss
- value-guided greedy action selection
- shallow neural expectimax where the learned value replaces the hand-built
  leaf evaluator

This may narrow the gap to search while keeping inference fast.

### 6. Search with a learned leaf evaluator

A promising hybrid:

```text
depth-1 or depth-2 exact search
learned value network at leaves
exact chance model for spawn slots and bonus values
```

This could outperform the pure neural policy while remaining much faster than
expectimax3. The simulator already has `transition_outcomes`, so the search
surface exists.

### 7. Rethink PPO reward shaping

Sparse final score is correct but hard. Alternatives to test carefully:

- final score plus a small per-merge score delta
- log score delta
- empty-cell bonus
- monotonicity/corner regularization
- survival reward only as a curriculum, not final objective
- terminal final-score reward preserved as the main signal

Risk: shaping can teach behavior that scores well on proxies but not on final
score. Every shaped run should be judged by deterministic final-score eval.

### 8. Evaluate more than mean score

Mean score is the headline metric, but debugging needs richer metrics:

- score distribution percentiles
- move count distribution
- empty-cell count over time
- max tile excluding the starter tile
- number of avoidable illegal/no-op attempts, if any
- late-game board entropy or trapped high-tile patterns
- action agreement with expectimax2 by move number and board density

The eval harness already writes per-seed CSVs. A small analysis script could
summarize failure clusters.

## Concrete Next Run Recommendation

If continuing on a more capable machine, I would not start with PPO. I would
start with DAgger.

Recommended first new implementation:

```text
threes_rl/train_dagger.py
```

Recommended first DAgger experiment:

```bash
.venv/bin/python -m threes_rl.train_dagger \
  --run-name dagger_ex2_rounds3_50k \
  --base-checkpoint threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/checkpoint_epoch_25.pt \
  --expert expectimax2 \
  --rounds 3 \
  --samples-per-round 50000 \
  --replay-dataset threes_rl/runs/imitation_expectimax2_200k_w8_e30/dataset.npz \
  --replay-ratio 0.5 \
  --epochs-per-round 10 \
  --batch-size 1024 \
  --workers <physical-cores> \
  --checkpoint-every 5 \
  --device cpu
```

Evaluate after each round:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ppo:threes_rl/runs/dagger_ex2_rounds3_50k/checkpoint_round_<R>_epoch_<E>.pt \
  --seeds 1000:1200 \
  --no-append
```

Success criterion for this phase:

```text
Beat 62154.87 mean score on seeds 1000:1200.
Beat 61814.26 mean score on seeds 1000:2000.
Move materially closer to expectimax2's 66460.14 on seeds 1000:1200.
```

If DAgger does not beat the current best after a few rounds, switch to model
architecture and value-learning work before generating huge amounts of the same
kind of data.

## Concrete Larger Dataset Recommendation

If the next agent wants a simpler scale-up before DAgger, run:

```bash
.venv/bin/python -m threes_rl.train_imitation \
  --run-name imitation_expectimax2_400k_wN_e30 \
  --expert expectimax2 \
  --samples 400000 \
  --epochs 30 \
  --batch-size 1024 \
  --workers <physical-cores> \
  --chunk-size 5000 \
  --device cpu \
  --save-full-dataset \
  --checkpoint-every 5
```

Evaluate epochs 10, 15, 20, 25, and 30. Do not assume the final epoch is best.

My expectation: it may improve modestly, but likely still trails expectimax2
because the core limitation is distribution shift and non-search inference.

## Code Improvements Worth Doing Soon

These are not prerequisites, but they would make research faster:

- Add `train_dagger.py`.
- Add an eval-analysis script that reads per-seed CSVs and compares policies
  seed-by-seed.
- Add checkpoint metadata recording eval score, not only training loss.
- Add `--hidden-dim` and architecture flags to `train_imitation.py`.
- Add policy agreement diagnostics: learned policy vs expectimax2 on held-out
  states by move bucket.
- Add a "max tile excluding starter" eval metric.
- Add multiprocessing support to expectation-value data generation if training
  a value net.
- Add optional `torch.compile` or MPS probes only after correctness is stable.

## Open Questions For A Stronger Model

Questions worth asking explicitly:

- Is DAgger enough to close most of the gap to expectimax2?
- Should the policy be trained to imitate actions, values, or action-value
  margins?
- Can we distill expectimax2 into a shallow search plus neural leaf evaluator
  more effectively than into a pure policy?
- What board representation best captures Threes geometry?
- Are there hand-engineered features from the expectimax evaluator that should
  become observation channels?
- Does the 1536 starter-tile setting hide early-game weaknesses that would
  matter on other devices?
- Can a learned policy discover the strategic value of delaying a terminal
  6144 merge when there is still safe score to harvest?

## Bottom Line

The simulator and environment are ready for serious iteration. The current
best learned model is useful and reproducibly above greedy, but it is not yet
"really, really good" relative to the search baseline. The most promising next
research step is DAgger or another data-aggregation method that labels the
learned policy's own visited states. The more capable machine should help
because expert labeling is CPU-bound; additional RAM is not currently the
critical resource.

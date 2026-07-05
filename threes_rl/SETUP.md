# Threes RL Setup And Remote Handoff

This document is the operational handoff for running the Threes RL simulator,
evaluation harness, imitation trainer, and PPO trainer on another machine.

The repository already has a GitHub remote:

```bash
git@github.com:jeffsharris/threeshelper.git
```

As of this handoff, GitHub reports the repository visibility as `PUBLIC`, so it
was left public per the user request. If a future copy of this repo is made
private, no code changes are needed.

## What To Clone

On the target machine:

```bash
git clone git@github.com:jeffsharris/threeshelper.git
cd threeshelper
git checkout master
```

If SSH is not configured on the target machine:

```bash
git clone https://github.com/jeffsharris/threeshelper.git
cd threeshelper
git checkout master
```

The RL code and current best artifacts live under `threes_rl/`.

Important committed artifacts:

```text
threes_rl/runs/imitation_expectimax2_200k_w8_e30/dataset.npz
threes_rl/runs/imitation_expectimax2_200k_w8_e30/checkpoint_epoch_20.pt
threes_rl/runs/imitation_expectimax2_200k_w8_e30/config.json
threes_rl/runs/imitation_expectimax2_200k_w8_e30/imitation_metrics.jsonl
```

The checkpoint above is the current best learned policy from this research
pass. Other epoch checkpoints and generated eval CSVs are ignored by default.

## Python Environment

This repo uses `uv` and a lockfile.

From the repo root:

```bash
uv venv
uv pip sync requirements.lock.txt
```

If you want to call Python explicitly without activating the environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

If you activate it:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Expected major RL dependencies are already in `requirements.in` and
`requirements.lock.txt`:

```text
gymnasium
torch
numpy
```

The core simulator itself is intentionally lightweight: `threes_rl/sim.py`
depends only on NumPy and the Python standard library.

## Verify The Checkout

Run the full test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Current expected result:

```text
Ran 43 tests
OK
```

The RL-specific coverage includes:

```bash
.venv/bin/python -m unittest \
  tests.test_rl_sim_rules \
  tests.test_rl_sim_schedule \
  tests.test_rl_sim_replay \
  tests.test_rl_env_api \
  tests.test_rl_train_imitation \
  -v
```

## Simulator Smoke Checks

Benchmark the simulator, Gymnasium wrapper, and expectimax:

```bash
.venv/bin/python -m threes_rl.bench
```

Recent measurements on the current machine:

```json
{
  "raw_sim_steps_per_s": 11785.95989593385,
  "env_steps_per_s": 9528.070462461455,
  "expectimax_d2_moves_per_s": 80.38857694052804
}
```

The exact numbers will vary by CPU, but large regressions are suspicious.

## Evaluate Existing Policies

Evaluate random:

```bash
.venv/bin/python -m threes_rl.eval --policy random --seeds 1000:1200 --no-append
```

Evaluate greedy:

```bash
.venv/bin/python -m threes_rl.eval --policy greedy --seeds 1000:1200 --no-append
```

Evaluate expectimax depth 2:

```bash
.venv/bin/python -m threes_rl.eval --policy expectimax2 --seeds 1000:1200 --no-append
```

Evaluate the current best learned policy:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ppo:threes_rl/runs/imitation_expectimax2_200k_w8_e30/checkpoint_epoch_20.pt \
  --seeds 1000:1200 \
  --no-append
```

Expected current best learned result on the fixed 200-seed suite:

```json
{
  "games": 200,
  "mean_score": 61826.34,
  "median_score": 61048.5,
  "p90_score": 66237,
  "mean_moves": 60.81
}
```

Wider sanity check:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ppo:threes_rl/runs/imitation_expectimax2_200k_w8_e30/checkpoint_epoch_20.pt \
  --seeds 1000:2000 \
  --no-append
```

Recent result:

```text
learned mean score: 61489.74
greedy mean score: 60224.63
```

## Train Imitation From Cached Data

The committed 200k expert dataset can be reused without regenerating labels:

```bash
.venv/bin/python -m threes_rl.train_imitation \
  --run-name imitation_expectimax2_200k_retry \
  --expert expectimax2 \
  --samples 200000 \
  --epochs 30 \
  --batch-size 1024 \
  --workers 1 \
  --dataset-path threes_rl/runs/imitation_expectimax2_200k_w8_e30/dataset.npz \
  --device cpu \
  --checkpoint-every 5
```

This is useful for testing architecture changes, optimizer changes, or epoch
selection without paying the expectimax labeling cost again.

## Generate A Larger Expert Dataset

Expectimax data generation is CPU-bound. More RAM is not currently the main
bottleneck; more or faster CPU cores help.

On a larger CPU machine, use the number of physical performance cores for
`--workers` if known. A reasonable first run on a beefier machine:

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

Replace `<physical-cores>` with a concrete number, for example `8`, `10`, or
`12`.

On the current machine, 8 workers produced roughly 470-517 expectimax2 labels
per second after worker warmup. Serial generation was roughly 82-88 labels per
second.

Evaluate every saved epoch checkpoint:

```bash
for epoch in 5 10 15 20 25 30; do
  echo "epoch=$epoch"
  .venv/bin/python -m threes_rl.eval \
    --policy ppo:threes_rl/runs/imitation_expectimax2_400k_wN_e30/checkpoint_epoch_${epoch}.pt \
    --seeds 1000:1200 \
    --no-append
done
```

The best-scoring checkpoint may not be the final epoch. In the 200k run,
epoch 20 beat epoch 30.

## PPO Training

PPO exists and writes checkpoints, but current evidence says sparse
final-score PPO from scratch is not yet competitive. Treat PPO as experimental
until a stronger warm start or reward design is added.

Smoke run:

```bash
.venv/bin/python -m threes_rl.train_ppo \
  --run-name smoke \
  --total-steps 512 \
  --num-envs 4 \
  --rollout-steps 32 \
  --minibatch-size 64 \
  --update-epochs 1 \
  --checkpoint-interval 512 \
  --device cpu
```

Warm-start style run from an imitation checkpoint:

```bash
.venv/bin/python -m threes_rl.train_ppo \
  --run-name ppo_from_imitation_probe \
  --resume threes_rl/runs/imitation_expectimax2_200k_w8_e30/checkpoint_epoch_20.pt \
  --total-steps 200000 \
  --num-envs 64 \
  --rollout-steps 128 \
  --minibatch-size 4096 \
  --update-epochs 4 \
  --device cpu
```

Evaluate:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ppo:threes_rl/runs/ppo_from_imitation_probe/latest.pt \
  --seeds 1000:1200 \
  --no-append
```

## Files Worth Reading First

Read in this order:

```text
RL_SPEC.md
threes_rl/ML_FINDINGS.md
threes_rl/PROGRESS.md
threes_rl/RESULTS.md
threes_rl/sim.py
threes_rl/env.py
threes_rl/train_imitation.py
threes_rl/eval.py
```

The simulator was intentionally validated against the existing tracker rather
than independently invented. The most important verification tests are in:

```text
tests/test_rl_sim_rules.py
tests/test_rl_sim_schedule.py
tests/test_rl_sim_replay.py
```

## Operational Notes

- The environment models the game internally and does not need iPhone
  Mirroring for RL training.
- The default start includes the observed real-device `1536` starter tile.
- Because of that starter tile, max-tile distribution thresholds up through
  `>=1536` are not very informative in the current eval reports.
- The training/eval objective is final board score, not tile face-value sum.
- `6144 + 6144 -> 12288` is legal, immediately terminal, and scores 1594323
  for that tile.
- Spawn location after a legal move is uniform over eligible empty trailing
  edge slots.
- Generated eval CSV files are intentionally ignored by git.
- Generated extra epoch checkpoints and full datasets are ignored by default;
  force-add only selected artifacts worth preserving.

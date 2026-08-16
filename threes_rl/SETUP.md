# Threes RL Setup And Remote Handoff

> Current research status is in `RL_PROGRAM_HANDOFF.md`. This setup guide keeps
> older command examples and baseline context; it is not the active scientific
> decision ledger.

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

The checkpoint above is the best committed learned-policy baseline from the
first research pass. Other epoch checkpoints and generated eval CSVs are ignored
by default.

Current local best after the 2026-07-05 400k run, if the generated artifacts are
present:

```text
threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/dataset.npz
threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/checkpoint_epoch_25.pt
```

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
Ran 61 tests
OK (skipped=1)
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

Write eval artifacts, including progress charts, retained top-game replays,
and death forensics for the median and worst five games:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy corner2 \
  --seeds 1000:1050 \
  --progress-every 10 \
  --no-append \
  --artifact-dir threes_rl/runs/eval_artifacts/corner2_1000_1050_probe \
  --keep-top-games 3 \
  --charts
```

Artifact-producing evals write:

```text
summary.json
progress.csv
progress.html
top_games/
death_forensics.json
death_forensics.html
```

Evaluate the current local best learned policy:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ppo:threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/checkpoint_epoch_25.pt \
  --seeds 1000:1200 \
  --no-append
```

Expected current local best learned result on the fixed 200-seed suite:

```json
{
  "games": 200,
  "mean_score": 62154.87,
  "median_score": 61236.0,
  "p90_score": 66786,
  "mean_moves": 63.42
}
```

Wider sanity check:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ppo:threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/checkpoint_epoch_25.pt \
  --seeds 1000:2000 \
  --no-append
```

Recent result:

```text
learned mean score: 61814.26
greedy mean score: 60224.63
```

## Train And Evaluate TD N-Tuple Afterstate Models

The revised research path uses a NumPy n-tuple afterstate value table rather
than torch. The default pattern set is roughly 270 MB per checkpoint and is a
good fit for a higher-RAM machine.

Run the current best smoke setting:

```bash
.venv/bin/python -m threes_rl.train_td \
  --run-name td_default_expected_500_init3000_a005_20260705 \
  --games 500 \
  --pattern-set default \
  --alpha 0.05 \
  --epsilon 0.02 \
  --init-total 3000 \
  --progress-every 100 \
  --checkpoint-every 250 \
  --keep-top-games 3 \
  --seed 20260705
```

This writes:

```text
threes_rl/runs/<run-name>/latest/
threes_rl/runs/<run-name>/progress.csv
threes_rl/runs/<run-name>/progress.html
threes_rl/runs/<run-name>/top_games/
```

Evaluate the resulting checkpoint with replay/chart artifacts:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ntuple:threes_rl/runs/td_default_expected_500_init3000_a005_20260705/latest \
  --seeds 1000:1050 \
  --progress-every 10 \
  --no-append \
  --artifact-dir threes_rl/runs/eval_artifacts/ntuple_td_default_expected_500_init3000_1000_1050 \
  --keep-top-games 3 \
  --charts
```

Recent held-out result:

```json
{
  "games": 50,
  "high_score": 88443,
  "mean_score": 65842.98,
  "mean_score_minus_starter": 6793.98,
  "mean_moves": 100.44,
  "p_max_tile_excl_starter_ge_1536": 0.0
}
```

Interpretation: the corrected TD learner is useful infrastructure and optimistic
initialization improves survival, but this run is still far below `corner2` and
does not yet build a non-starter 1536.

Bootstrap the value table from a stronger actor:

```bash
.venv/bin/python -m threes_rl.train_td \
  --run-name td_default_corner2_mc_200_init3000_a001_20260705 \
  --games 200 \
  --pattern-set default \
  --alpha 0.01 \
  --init-total 3000 \
  --actor-policy corner2 \
  --target-mode mc \
  --progress-every 25 \
  --checkpoint-every 100 \
  --keep-top-games 3 \
  --seed 20260705
```

Recent result:

```json
{
  "games": 200,
  "high_score": 194271,
  "mean_score": 74121.345,
  "mean_score_minus_starter": 15072.345,
  "p_max_tile_excl_starter_ge_3072": 0.015
}
```

Evaluate a learned-leaf search policy:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_200_init3000_a001_20260705/latest \
  --seeds 1000:1200 \
  --progress-every 10 \
  --no-append \
  --artifact-dir threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1200_full \
  --keep-top-games 3 \
  --charts
```

Recent result:

```json
{
  "games": 200,
  "high_score": 205719,
  "mean_score": 74300.805,
  "mean_score_minus_starter": 15251.805,
  "p_max_tile_excl_starter_ge_1536": 0.015,
  "p_max_tile_excl_starter_ge_3072": 0.015
}
```

The recent 200-seed result was produced by combining an existing `1000:1050`
eval with a later `1050:1200` eval to avoid redoing slow searched games. A
single direct `1000:1200` run should produce the same deterministic result.

The `ntuple_expectimax2a:<checkpoint>` adaptive variant exists but was too slow
in the first probe; profile it before using it for wider eval.

## Student-Actor Value Iteration

The July 6 steers shifted the main loop from "fit more `corner2` data" to a
student-actor cycle:

1. Evaluate the current learned-search checkpoint against `corner2` on paired
   seeds.
2. If it is competitive, use that learned-search policy as the next actor.
3. Refit the value table with lower-variance n-step targets and TC updates.
4. Gate the new checkpoint on paired eval before promotion.

The current best candidate checkpoint is:

```text
threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest
```

Train a next-iteration value table from that student actor:

```bash
.venv/bin/python -m threes_rl.train_td \
  --run-name td_default_student1_nstep_tc_20260706 \
  --games 100 \
  --pattern-set default \
  --init-total 3000 \
  --alpha 0.02 \
  --actor-policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest \
  --target-mode nstep \
  --n-step 8 \
  --use-tc \
  --progress-every 10 \
  --checkpoint-every 50 \
  --keep-top-games 3
```

Add replay-derived late starts when useful. This example samples 30% of
episodes from frames in the current best replay where the max tile excluding
the free starter is at least 768:

```bash
.venv/bin/python -m threes_rl.train_td \
  --run-name td_default_student1_nstep_tc_late_20260706 \
  --games 100 \
  --pattern-set default \
  --init-total 3000 \
  --alpha 0.02 \
  --actor-policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest \
  --target-mode nstep \
  --n-step 8 \
  --use-tc \
  --start-state-replay threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_1000_a0005_1000_1050/top_games/rank_01_score_246774_seed_1014/replay.json \
  --start-state-prob 0.3 \
  --start-state-min-tile 768 \
  --progress-every 10 \
  --checkpoint-every 50 \
  --keep-top-games 3
```

Use the larger six-tuple value function on high-RAM machines:

```bash
.venv/bin/python -m threes_rl.train_td \
  --run-name td_big6_student1_nstep_tc_20260706 \
  --games 100 \
  --pattern-set big6 \
  --init-total 3000 \
  --alpha 0.02 \
  --actor-policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest \
  --target-mode nstep \
  --n-step 8 \
  --use-tc \
  --progress-every 10 \
  --checkpoint-every 50 \
  --keep-top-games 3
```

Cycle through starter states during training to exercise the early game as
well as the fixed 1536-start game:

```bash
.venv/bin/python -m threes_rl.train_td \
  --run-name td_default_curriculum_nstep_tc_20260706 \
  --games 1000 \
  --pattern-set default \
  --init-total 3000 \
  --alpha 0.005 \
  --actor-policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest \
  --target-mode nstep \
  --n-step 8 \
  --use-tc \
  --starter-curriculum none,96,384,1536 \
  --progress-every 50 \
  --checkpoint-every 250 \
  --keep-top-games 3
```

For evaluation, `--starter` can be a comma-separated mixed suite. The summary
will include a `by_starter` block, and retained top-game manifests record the
starter used for each replay:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest \
  --seeds 1000:1050 \
  --starter none,96,384,1536 \
  --progress-every 20 \
  --no-append \
  --keep-top-games 3 \
  --charts
```

Blend a proven parent value table with a sidecar/student value table at
expectimax leaves:

```bash
.venv/bin/python -m threes_rl.eval \
  --policy ntuple_blend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25 \
  --seeds 1000:1050 \
  --progress-every 5 \
  --no-append \
  --artifact-dir threes_rl/runs/eval_artifacts/ntuple_blend_parent_mc1000_student50_w025_1000_1050 \
  --keep-top-games 3 \
  --charts
```

The blend weight is the sidecar proportion:
`0.0 == parent only`, `1.0 == sidecar only`.

Compare paired eval CSVs:

```bash
.venv/bin/python -m threes_rl.compare_eval \
  --candidate threes_rl/runs/eval/ntuple_expectimax2_threes_rl_runs_td_default_corner2_mc_1000_init3000_a0005_20260706_latest_1000_1049.csv \
  --baseline threes_rl/runs/eval/corner2_1000_1199.csv \
  --metric score_minus_starter
```

## Live Research Dashboard

Build the dashboard once:

```bash
.venv/bin/python -m threes_rl.dashboard --refresh-seconds 15
```

Keep it refreshing while experiments run:

```bash
.venv/bin/python -m threes_rl.dashboard --watch --interval 10 --refresh-seconds 15
```

Serve the repo locally and open:

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory /Users/jeffharris/code/threeshelper
```

Then visit:

```text
http://127.0.0.1:8765/threes_rl/runs/dashboard/index.html
```

The dashboard scans run summaries and progress CSVs, shows the best-so-far
high score over artifact time, marks major technique shifts or discontinuities
with hover details, and separately tracks high/mean/median score-minus-starter
over time.

## Record Human Simulator Games

Start the exact-simulator browser interface:

```bash
.venv/bin/python -m threes_rl.human_play_server --host 127.0.0.1 --port 8770
```

Then visit:

```text
http://127.0.0.1:8770/
```

Accepted arrow-key moves are written immediately under
`datasets/human_play/<session_id>/replay.json`. Each frame includes the board,
preview, tile-cycle state, legal mask, action, inserted tile and slot, decision
time, split RNG stream IDs, and explicit human/root provenance. Finished games
also receive a browser replay at `replay.html`.

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
- The default start includes the observed real-device `1536` starter tile in
  the top-left corner.
- Because of that starter tile, max-tile distribution thresholds up through
  `>=1536` are not very informative in the current eval reports.
- Checkpoints listed above were trained before the top-left starter fix and
  should be treated as pre-fix baselines until retrained.
- The training/eval objective is final board score, not tile face-value sum.
- `6144 + 6144 -> 12288` is legal, immediately terminal, and scores 1594323
  for that tile.
- Spawn location after a legal move is uniform over eligible empty trailing
  edge slots.
- Generated eval CSV files are intentionally ignored by git.
- Generated extra epoch checkpoints and full datasets are ignored by default;
  force-add only selected artifacts worth preserving.

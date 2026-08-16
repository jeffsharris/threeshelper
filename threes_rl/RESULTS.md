# Threes RL Results

Note: older baseline sections in this file were generated before the simulator
was corrected to place the `1536` starter tile in the top-left corner. Sections
that explicitly say "corrected starter" or use the `*_fixed_starter` /
`*_expected_*` run names are post-fix results.

## Eval: corner2 corrected starter

Command: `python -m threes_rl.eval --policy corner2 --seeds 1000:1200 --progress-every 20 --no-append --artifact-dir threes_rl/runs/eval_artifacts/corner2_1000_1200_fixed_starter --keep-top-games 3 --charts`

```json
{
  "games": 200,
  "high_score": 204675,
  "high_score_minus_starter": 145626,
  "mean_score": 74323.59,
  "mean_score_minus_starter": 15274.59,
  "median_score": 69145.5,
  "median_score_minus_starter": 10096.5,
  "mean_moves": 163.785,
  "p_max_tile_excl_starter_ge_1536": 0.015,
  "p_max_tile_excl_starter_ge_3072": 0.015,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained replays:

- `threes_rl/runs/eval_artifacts/corner2_1000_1200_fixed_starter/top_games/rank_01_score_204675_seed_1013/replay.html`
- `threes_rl/runs/eval_artifacts/corner2_1000_1200_fixed_starter/top_games/rank_02_score_193581_seed_1032/replay.html`
- `threes_rl/runs/eval_artifacts/corner2_1000_1200_fixed_starter/top_games/rank_03_score_180456_seed_1049/replay.html`

Progress chart: `threes_rl/runs/eval_artifacts/corner2_1000_1200_fixed_starter/progress.html`

## TD n-tuple afterstate learner: expected-spawn correction

Implemented `threes_rl/ntuple.py` / `threes_rl/train_td.py` so move selection
and TD updates use the exact simulator chance model:

- `expected_afterstate_target()` enumerates spawn slots, preview value/candidate
  probabilities, and next-preview probabilities through
  `ThreesSim.transition_outcomes()`.
- `choose_action()` now scores legal moves by immediate merge score plus the
  expected post-spawn/future afterstate target.
- `train_td` updates the current afterstate directly against that Bellman target.
- `--init-total` initializes the total board value across active n-tuple
  features, avoiding accidental hundreds-of-times-too-large optimistic values.

### Train: td_default_expected_500_init3000_a005_20260705

Command: `python -m threes_rl.train_td --run-name td_default_expected_500_init3000_a005_20260705 --games 500 --pattern-set default --alpha 0.05 --epsilon 0.02 --init-total 3000 --progress-every 100 --checkpoint-every 250 --keep-top-games 3 --seed 20260705`

```json
{
  "games": 500,
  "high_score": 86520,
  "high_score_minus_starter": 27471,
  "mean_score": 63906.816,
  "mean_score_minus_starter": 4857.816,
  "median_score": 62458.5,
  "median_score_minus_starter": 3409.5,
  "mean_moves": 88.462,
  "p_max_tile_excl_starter_ge_1536": 0.0,
  "p_max_tile_excl_starter_ge_3072": 0.0,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained training replays:

- `threes_rl/runs/td_default_expected_500_init3000_a005_20260705/top_games/rank_01_score_86520_seed_369261752/replay.html`
- `threes_rl/runs/td_default_expected_500_init3000_a005_20260705/top_games/rank_02_score_81459_seed_284261497/replay.html`
- `threes_rl/runs/td_default_expected_500_init3000_a005_20260705/top_games/rank_03_score_81375_seed_404261857/replay.html`

Progress chart: `threes_rl/runs/td_default_expected_500_init3000_a005_20260705/progress.html`

### Eval: ntuple td_default_expected_500_init3000

Command: `python -m threes_rl.eval --policy ntuple:threes_rl/runs/td_default_expected_500_init3000_a005_20260705/latest --seeds 1000:1050 --progress-every 10 --no-append --artifact-dir threes_rl/runs/eval_artifacts/ntuple_td_default_expected_500_init3000_1000_1050 --keep-top-games 3 --charts`

```json
{
  "games": 50,
  "high_score": 88443,
  "high_score_minus_starter": 29394,
  "mean_score": 65842.98,
  "mean_score_minus_starter": 6793.98,
  "median_score": 63952.5,
  "median_score_minus_starter": 4903.5,
  "mean_moves": 100.44,
  "p_max_tile_excl_starter_ge_1536": 0.0,
  "p_max_tile_excl_starter_ge_3072": 0.0,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained eval replays:

- `threes_rl/runs/eval_artifacts/ntuple_td_default_expected_500_init3000_1000_1050/top_games/rank_01_score_88443_seed_1031/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_td_default_expected_500_init3000_1000_1050/top_games/rank_02_score_80712_seed_1017/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_td_default_expected_500_init3000_1000_1050/top_games/rank_03_score_80052_seed_1040/replay.html`

Progress chart: `threes_rl/runs/eval_artifacts/ntuple_td_default_expected_500_init3000_1000_1050/progress.html`

Interpretation: optimistic TD self-play is now learning longer survival than the
zero-init smoke (`mean_moves` 100.44 held-out after 500 games vs 45.42 after the
100-game zero-init checkpoint), but it still has no non-starter 1536 and remains
far below `corner2`. The next performance step should bootstrap the value model
from stronger actors or add learned-leaf expectimax rather than relying on short
pure self-play from a near-empty table.

## Learned-leaf expectimax and corner2 bootstrap

Implemented `NtupleExpectimaxPolicy` in `threes_rl/expectimax.py`.

- Policy specs: `ntuple_expectimax2:<checkpoint>` and
  `ntuple_expectimax2a:<checkpoint>`.
- The learned-search policy adds exact `score_delta` at simulator chance nodes,
  then uses the n-tuple afterstate table for remaining future value.
- The adaptive `2a` variant was probed but interrupted after roughly 90 seconds
  without completing the first game; it needs profiling before regular use.
- The searched eval hot path was optimized after the full-suite attempt exposed
  wasteful fallback scoring: `score_board()` now avoids calling `score_tile()`
  when a tile is already in `SCORE_BY_VALUE`.

### Eval: ntuple_expectimax2 on self-play TD-500 checkpoint

Command: `python -m threes_rl.eval --policy ntuple_expectimax2:threes_rl/runs/td_default_expected_500_init3000_a005_20260705/latest --seeds 1000:1010 --progress-every 2 --no-append --artifact-dir threes_rl/runs/eval_artifacts/ntuple_expectimax2_td500_init3000_1000_1010 --keep-top-games 3 --charts`

```json
{
  "games": 10,
  "high_score": 81852,
  "high_score_minus_starter": 22803,
  "mean_score": 68886,
  "mean_score_minus_starter": 9837,
  "median_score": 68344.5,
  "median_score_minus_starter": 9295.5,
  "mean_moves": 129.2,
  "p_max_tile_excl_starter_ge_1536": 0.0,
  "p_max_tile_excl_starter_ge_3072": 0.0,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained replays:

- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_td500_init3000_1000_1010/top_games/rank_01_score_81852_seed_1003/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_td500_init3000_1000_1010/top_games/rank_02_score_81591_seed_1009/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_td500_init3000_1000_1010/top_games/rank_03_score_68883_seed_1008/replay.html`

Progress chart: `threes_rl/runs/eval_artifacts/ntuple_expectimax2_td500_init3000_1000_1010/progress.html`

### Train: td_default_corner2_mc_50_init3000_a005_20260705

Command: `python -m threes_rl.train_td --run-name td_default_corner2_mc_50_init3000_a005_20260705 --games 50 --pattern-set default --alpha 0.05 --init-total 3000 --actor-policy corner2 --target-mode mc --progress-every 10 --checkpoint-every 25 --keep-top-games 3 --seed 20260705`

```json
{
  "games": 50,
  "high_score": 194271,
  "high_score_minus_starter": 135222,
  "mean_score": 74017.5,
  "mean_score_minus_starter": 14968.5,
  "median_score": 69258.0,
  "median_score_minus_starter": 10209.0,
  "mean_moves": 155.5,
  "p_max_tile_excl_starter_ge_1536": 0.02,
  "p_max_tile_excl_starter_ge_3072": 0.02,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained training replays:

- `threes_rl/runs/td_default_corner2_mc_50_init3000_a005_20260705/top_games/rank_01_score_194271_seed_29260732/replay.html`
- `threes_rl/runs/td_default_corner2_mc_50_init3000_a005_20260705/top_games/rank_02_score_88440_seed_35260750/replay.html`
- `threes_rl/runs/td_default_corner2_mc_50_init3000_a005_20260705/top_games/rank_03_score_88179_seed_62260831/replay.html`

Progress chart: `threes_rl/runs/td_default_corner2_mc_50_init3000_a005_20260705/progress.html`

### Eval: ntuple_expectimax2 on corner2-MC checkpoint

Command: `python -m threes_rl.eval --policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_50_init3000_a005_20260705/latest --seeds 1000:1010 --progress-every 2 --no-append --artifact-dir threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_50_1000_1010 --keep-top-games 3 --charts`

```json
{
  "games": 10,
  "high_score": 87888,
  "high_score_minus_starter": 28839,
  "mean_score": 72881.7,
  "mean_score_minus_starter": 13832.7,
  "median_score": 69036.0,
  "median_score_minus_starter": 9987.0,
  "mean_moves": 159.3,
  "p_max_tile_excl_starter_ge_1536": 0.0,
  "p_max_tile_excl_starter_ge_3072": 0.0,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained eval replays:

- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_50_1000_1010/top_games/rank_01_score_87888_seed_1002/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_50_1000_1010/top_games/rank_02_score_82614_seed_1003/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_50_1000_1010/top_games/rank_03_score_80235_seed_1000/replay.html`

Progress chart: `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_50_1000_1010/progress.html`

### Train: td_default_corner2_mc_200_init3000_a005_20260705

Command: `python -m threes_rl.train_td --run-name td_default_corner2_mc_200_init3000_a005_20260705 --games 200 --pattern-set default --alpha 0.05 --init-total 3000 --actor-policy corner2 --target-mode mc --progress-every 25 --checkpoint-every 100 --keep-top-games 3 --seed 20260705`

```json
{
  "games": 200,
  "high_score": 194271,
  "high_score_minus_starter": 135222,
  "mean_score": 74121.345,
  "mean_score_minus_starter": 15072.345,
  "median_score": 69643.5,
  "median_score_minus_starter": 10594.5,
  "mean_moves": 162.285,
  "p_max_tile_excl_starter_ge_1536": 0.015,
  "p_max_tile_excl_starter_ge_3072": 0.015,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained training replays:

- `threes_rl/runs/td_default_corner2_mc_200_init3000_a005_20260705/top_games/rank_01_score_194271_seed_29260732/replay.html`
- `threes_rl/runs/td_default_corner2_mc_200_init3000_a005_20260705/top_games/rank_02_score_184224_seed_112260981/replay.html`
- `threes_rl/runs/td_default_corner2_mc_200_init3000_a005_20260705/top_games/rank_03_score_180777_seed_187261206/replay.html`

### Eval: ntuple_expectimax2 on corner2-MC-200 alpha 0.05

Command: `python -m threes_rl.eval --policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_200_init3000_a005_20260705/latest --seeds 1000:1020 --progress-every 5 --no-append --artifact-dir threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_1000_1020 --keep-top-games 3 --charts`

```json
{
  "games": 20,
  "high_score": 88407,
  "high_score_minus_starter": 29358,
  "mean_score": 68084.25,
  "mean_score_minus_starter": 9035.25,
  "median_score": 65046.0,
  "median_score_minus_starter": 5997.0,
  "mean_moves": 111.85,
  "p_max_tile_excl_starter_ge_1536": 0.0,
  "p_max_tile_excl_starter_ge_3072": 0.0,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Interpretation: simply adding more corner2-MC data with `alpha=0.05` made the
learned-search policy worse than the 50-game checkpoint, suggesting the MC
updates were too aggressive for mixed-quality trajectories.

### Train: td_default_corner2_mc_200_init3000_a001_20260705

Command: `python -m threes_rl.train_td --run-name td_default_corner2_mc_200_init3000_a001_20260705 --games 200 --pattern-set default --alpha 0.01 --init-total 3000 --actor-policy corner2 --target-mode mc --progress-every 25 --checkpoint-every 100 --keep-top-games 3 --seed 20260705`

```json
{
  "games": 200,
  "high_score": 194271,
  "high_score_minus_starter": 135222,
  "mean_score": 74121.345,
  "mean_score_minus_starter": 15072.345,
  "median_score": 69643.5,
  "median_score_minus_starter": 10594.5,
  "mean_moves": 162.285,
  "p_max_tile_excl_starter_ge_1536": 0.015,
  "p_max_tile_excl_starter_ge_3072": 0.015,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained training replays:

- `threes_rl/runs/td_default_corner2_mc_200_init3000_a001_20260705/top_games/rank_01_score_194271_seed_29260732/replay.html`
- `threes_rl/runs/td_default_corner2_mc_200_init3000_a001_20260705/top_games/rank_02_score_184224_seed_112260981/replay.html`
- `threes_rl/runs/td_default_corner2_mc_200_init3000_a001_20260705/top_games/rank_03_score_180777_seed_187261206/replay.html`

Progress chart: `threes_rl/runs/td_default_corner2_mc_200_init3000_a001_20260705/progress.html`

### Eval: ntuple_expectimax2 on corner2-MC-200 alpha 0.01

Command: `python -m threes_rl.eval --policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_200_init3000_a001_20260705/latest --seeds 1000:1050 --progress-every 10 --no-append --artifact-dir threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1050 --keep-top-games 3 --charts`

```json
{
  "games": 50,
  "high_score": 205719,
  "high_score_minus_starter": 146670,
  "mean_score": 77139.78,
  "mean_score_minus_starter": 18090.78,
  "median_score": 69819.0,
  "median_score_minus_starter": 10770.0,
  "mean_moves": 169.14,
  "p_max_tile_excl_starter_ge_1536": 0.04,
  "p_max_tile_excl_starter_ge_3072": 0.04,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained eval replays:

- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1050/top_games/rank_01_score_205719_seed_1049/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1050/top_games/rank_02_score_181056_seed_1012/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1050/top_games/rank_03_score_92118_seed_1036/replay.html`

Progress chart: `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1050/progress.html`

### Eval: ntuple_expectimax2 on corner2-MC-200 alpha 0.01, full suite

Combined from:

- `python -m threes_rl.eval --policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_200_init3000_a001_20260705/latest --seeds 1000:1050 --progress-every 10 --no-append --artifact-dir threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1050 --keep-top-games 3 --charts`
- `python -m threes_rl.eval --policy ntuple_expectimax2:threes_rl/runs/td_default_corner2_mc_200_init3000_a001_20260705/latest --seeds 1050:1200 --progress-every 10 --no-append --artifact-dir threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1050_1200_fastscore --keep-top-games 3 --charts`

Combined artifact dir:
`threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1200_full/`

```json
{
  "games": 200,
  "high_score": 205719,
  "high_score_minus_starter": 146670,
  "mean_score": 74300.805,
  "mean_score_minus_starter": 15251.805,
  "median_score": 69543.0,
  "median_score_minus_starter": 10494.0,
  "mean_moves": 169.46,
  "p_max_tile_excl_starter_ge_1536": 0.015,
  "p_max_tile_excl_starter_ge_3072": 0.015,
  "p_max_tile_excl_starter_ge_6144": 0.0
}
```

Top retained eval replays:

- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1200_full/top_games/rank_01_score_205719_seed_1049/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1200_full/top_games/rank_02_score_181056_seed_1012/replay.html`
- `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1200_full/top_games/rank_03_score_180846_seed_1132/replay.html`

Progress chart:
`threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1200_full/progress.html`

Interpretation: the lower-alpha 200-game corner2-MC checkpoint is now the best
learned-search result, but on the full 200-seed suite it is effectively a tie
with `corner2`, not a clear win: mean score-minus-starter is 15251.805 versus
`corner2`'s 15274.59, while high score is slightly higher (205719 vs 204675).
Both policies built non-starter 3072 in 3/200 games and reached 6144 in 0/200.

## Bench

Command: `python -m threes_rl.bench`

```json
{
  "env_steps_per_s": 9528.070462461455,
  "expectimax_d2_moves_per_s": 80.38857694052804,
  "raw_sim_steps_per_s": 11785.95989593385
}
```

## Eval: random

Command: `python -m threes_rl.eval --policy random --seeds 1000:1200`

```json
{
  "games": 200,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.0,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "mean_moves": 30.4,
  "mean_score": 59600.205,
  "median_score": 59220.0,
  "p90_score": 60009
}
```

## Eval: greedy

Command: `python -m threes_rl.eval --policy greedy --seeds 1000:1200`

```json
{
  "games": 200,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.0,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "mean_moves": 40.44,
  "mean_score": 60304.29,
  "median_score": 59488.5,
  "p90_score": 61677
}
```

## Eval: expectimax2

Command: `python -m threes_rl.eval --policy expectimax2 --seeds 1000:1200`

```json
{
  "games": 200,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.0,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "mean_moves": 107.95,
  "mean_score": 66460.14,
  "median_score": 65814.0,
  "p90_score": 73905
}
```

## Eval: expectimax3 probe

Command: `python -m threes_rl.eval --policy expectimax3 --seeds 1000:1005 --progress-every 1 --no-append`

```json
{
  "games": 5,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.0,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "mean_moves": 142.2,
  "mean_score": 76954.8,
  "median_score": 79929,
  "p90_score": 81306
}
```

## Eval: learned imitation_expectimax2_30k

Command: `python -m threes_rl.eval --policy ppo:threes_rl/runs/imitation_expectimax2_30k/latest.pt --seeds 1000:1200 --no-append`

```json
{
  "games": 200,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.0,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "mean_moves": 47.365,
  "mean_score": 60848.28,
  "median_score": 59610.0,
  "p90_score": 65838
}
```

## Eval: learned imitation_expectimax2_200k_epoch20

Command: `python -m threes_rl.eval --policy ppo:threes_rl/runs/imitation_expectimax2_200k_w8_e30/checkpoint_epoch_20.pt --seeds 1000:1200 --no-append`

```json
{
  "games": 200,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.0,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "mean_moves": 60.81,
  "mean_score": 61826.34,
  "median_score": 61048.5,
  "p90_score": 66237
}
```

## Eval: learned imitation_expectimax2_400k_w10_e30_20260705_epoch25

Command: `python -m threes_rl.eval --policy ppo:threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/checkpoint_epoch_25.pt --seeds 1000:1200 --no-append`

```json
{
  "games": 200,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.0,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "mean_moves": 63.42,
  "mean_score": 62154.87,
  "median_score": 61236.0,
  "p90_score": 66786
}
```

Best learned checkpoint so far: `threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/checkpoint_epoch_25.pt`.

## Wider sanity check

- Greedy, seeds 1000:2000: mean score 60224.63, mean moves 39.02.
- Learned `imitation_expectimax2_200k_epoch20`, seeds 1000:2000: mean score 61489.74, mean moves 59.97.
- Learned `imitation_expectimax2_400k_w10_e30_20260705_epoch25`, seeds 1000:2000: mean score 61814.26, mean moves 62.12.

## Bench

Command: `python -m threes_rl.bench`

```json
{
  "env_steps_per_s": 16000.39403485069,
  "expectimax_d2_moves_per_s": 134.29416330664034,
  "raw_sim_steps_per_s": 20755.304199486756
}
```

## Eval: ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 1450:1550 --starter 1536 --progress-every 5 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708 --keep-top-games 50 --charts`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/death_forensics.json"
  },
  "games": 100,
  "high_score": 206778,
  "high_score_minus_starter": 147729,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.03,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.03,
    ">=192": 1.0,
    ">=3072": 0.03,
    ">=384": 0.93,
    ">=6144": 0.0,
    ">=768": 0.47
  },
  "mean_moves": 179.02,
  "mean_score": 79698.81,
  "mean_score_minus_starter": 20649.81,
  "median_moves": 173.5,
  "median_score": 75391.5,
  "median_score_minus_starter": 16342.5,
  "p90_moves": 253,
  "p90_score": 87951,
  "p90_score_minus_starter": 28902,
  "p_max_tile_excl_starter_ge_1536": 0.03,
  "p_max_tile_excl_starter_ge_3072": 0.03,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_01_score_206778_seed_1532_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_01_score_206778_seed_1532_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 434,
      "rank": 1,
      "score": 206778,
      "score_minus_starter": 147729,
      "seed": 1532,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_02_score_204729_seed_1519_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_02_score_204729_seed_1519_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 253,
      "rank": 2,
      "score": 204729,
      "score_minus_starter": 145680,
      "seed": 1519,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_03_score_185469_seed_1545_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_03_score_185469_seed_1545_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 430,
      "rank": 3,
      "score": 185469,
      "score_minus_starter": 126420,
      "seed": 1545,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_04_score_99813_seed_1528_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_04_score_99813_seed_1528_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 221,
      "rank": 4,
      "score": 99813,
      "score_minus_starter": 40764,
      "seed": 1528,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_05_score_95235_seed_1501_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_05_score_95235_seed_1501_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 307,
      "rank": 5,
      "score": 95235,
      "score_minus_starter": 36186,
      "seed": 1501,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_06_score_95217_seed_1456_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_06_score_95217_seed_1456_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 199,
      "rank": 6,
      "score": 95217,
      "score_minus_starter": 36168,
      "seed": 1456,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_07_score_92976_seed_1537_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_07_score_92976_seed_1537_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 286,
      "rank": 7,
      "score": 92976,
      "score_minus_starter": 33927,
      "seed": 1537,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_08_score_88905_seed_1548_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_08_score_88905_seed_1548_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 347,
      "rank": 8,
      "score": 88905,
      "score_minus_starter": 29856,
      "seed": 1548,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_09_score_88764_seed_1484_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_09_score_88764_seed_1484_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 151,
      "rank": 9,
      "score": 88764,
      "score_minus_starter": 29715,
      "seed": 1484,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_10_score_88422_seed_1543_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_10_score_88422_seed_1543_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 228,
      "rank": 10,
      "score": 88422,
      "score_minus_starter": 29373,
      "seed": 1543,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_11_score_87951_seed_1478_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_11_score_87951_seed_1478_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 175,
      "rank": 11,
      "score": 87951,
      "score_minus_starter": 28902,
      "seed": 1478,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_12_score_87876_seed_1450_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_12_score_87876_seed_1450_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 248,
      "rank": 12,
      "score": 87876,
      "score_minus_starter": 28827,
      "seed": 1450,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_13_score_87852_seed_1489_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_13_score_87852_seed_1489_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 167,
      "rank": 13,
      "score": 87852,
      "score_minus_starter": 28803,
      "seed": 1489,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_14_score_87630_seed_1504_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_14_score_87630_seed_1504_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 272,
      "rank": 14,
      "score": 87630,
      "score_minus_starter": 28581,
      "seed": 1504,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_15_score_87228_seed_1515_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_15_score_87228_seed_1515_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 208,
      "rank": 15,
      "score": 87228,
      "score_minus_starter": 28179,
      "seed": 1515,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_16_score_87075_seed_1459_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_16_score_87075_seed_1459_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 267,
      "rank": 16,
      "score": 87075,
      "score_minus_starter": 28026,
      "seed": 1459,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_17_score_86799_seed_1479_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_17_score_86799_seed_1479_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 189,
      "rank": 17,
      "score": 86799,
      "score_minus_starter": 27750,
      "seed": 1479,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_18_score_86511_seed_1505_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_18_score_86511_seed_1505_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 202,
      "rank": 18,
      "score": 86511,
      "score_minus_starter": 27462,
      "seed": 1505,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_19_score_86469_seed_1546_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_19_score_86469_seed_1546_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 231,
      "rank": 19,
      "score": 86469,
      "score_minus_starter": 27420,
      "seed": 1546,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_20_score_85170_seed_1510_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_20_score_85170_seed_1510_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 190,
      "rank": 20,
      "score": 85170,
      "score_minus_starter": 26121,
      "seed": 1510,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_21_score_84501_seed_1453_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_21_score_84501_seed_1453_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 151,
      "rank": 21,
      "score": 84501,
      "score_minus_starter": 25452,
      "seed": 1453,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_22_score_84468_seed_1547_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_22_score_84468_seed_1547_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 189,
      "rank": 22,
      "score": 84468,
      "score_minus_starter": 25419,
      "seed": 1547,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_23_score_83349_seed_1460_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_23_score_83349_seed_1460_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 124,
      "rank": 23,
      "score": 83349,
      "score_minus_starter": 24300,
      "seed": 1460,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_24_score_82923_seed_1488_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_24_score_82923_seed_1488_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 185,
      "rank": 24,
      "score": 82923,
      "score_minus_starter": 23874,
      "seed": 1488,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_25_score_82704_seed_1511_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_25_score_82704_seed_1511_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 203,
      "rank": 25,
      "score": 82704,
      "score_minus_starter": 23655,
      "seed": 1511,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_26_score_82683_seed_1496_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_26_score_82683_seed_1496_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 202,
      "rank": 26,
      "score": 82683,
      "score_minus_starter": 23634,
      "seed": 1496,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_27_score_82662_seed_1470_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_27_score_82662_seed_1470_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 121,
      "rank": 27,
      "score": 82662,
      "score_minus_starter": 23613,
      "seed": 1470,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_28_score_82572_seed_1465_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_28_score_82572_seed_1465_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 273,
      "rank": 28,
      "score": 82572,
      "score_minus_starter": 23523,
      "seed": 1465,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_29_score_82428_seed_1500_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_29_score_82428_seed_1500_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 191,
      "rank": 29,
      "score": 82428,
      "score_minus_starter": 23379,
      "seed": 1500,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_30_score_82335_seed_1480_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_30_score_82335_seed_1480_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 132,
      "rank": 30,
      "score": 82335,
      "score_minus_starter": 23286,
      "seed": 1480,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_31_score_82212_seed_1518_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_31_score_82212_seed_1518_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 325,
      "rank": 31,
      "score": 82212,
      "score_minus_starter": 23163,
      "seed": 1518,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_32_score_82128_seed_1498_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_32_score_82128_seed_1498_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 179,
      "rank": 32,
      "score": 82128,
      "score_minus_starter": 23079,
      "seed": 1498,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_33_score_82116_seed_1535_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_33_score_82116_seed_1535_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 341,
      "rank": 33,
      "score": 82116,
      "score_minus_starter": 23067,
      "seed": 1535,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_34_score_82104_seed_1531_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_34_score_82104_seed_1531_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 234,
      "rank": 34,
      "score": 82104,
      "score_minus_starter": 23055,
      "seed": 1531,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_35_score_82047_seed_1517_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_35_score_82047_seed_1517_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 189,
      "rank": 35,
      "score": 82047,
      "score_minus_starter": 22998,
      "seed": 1517,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_36_score_82005_seed_1458_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_36_score_82005_seed_1458_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 250,
      "rank": 36,
      "score": 82005,
      "score_minus_starter": 22956,
      "seed": 1458,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_37_score_81546_seed_1464_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_37_score_81546_seed_1464_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 196,
      "rank": 37,
      "score": 81546,
      "score_minus_starter": 22497,
      "seed": 1464,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_38_score_81327_seed_1493_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_38_score_81327_seed_1493_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 214,
      "rank": 38,
      "score": 81327,
      "score_minus_starter": 22278,
      "seed": 1493,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_39_score_80868_seed_1482_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_39_score_80868_seed_1482_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 243,
      "rank": 39,
      "score": 80868,
      "score_minus_starter": 21819,
      "seed": 1482,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_40_score_80766_seed_1525_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_40_score_80766_seed_1525_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 141,
      "rank": 40,
      "score": 80766,
      "score_minus_starter": 21717,
      "seed": 1525,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_41_score_80742_seed_1520_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_41_score_80742_seed_1520_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 206,
      "rank": 41,
      "score": 80742,
      "score_minus_starter": 21693,
      "seed": 1520,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_42_score_80709_seed_1487_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_42_score_80709_seed_1487_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 218,
      "rank": 42,
      "score": 80709,
      "score_minus_starter": 21660,
      "seed": 1487,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_43_score_80097_seed_1502_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_43_score_80097_seed_1502_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 190,
      "rank": 43,
      "score": 80097,
      "score_minus_starter": 21048,
      "seed": 1502,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_44_score_80070_seed_1454_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_44_score_80070_seed_1454_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 215,
      "rank": 44,
      "score": 80070,
      "score_minus_starter": 21021,
      "seed": 1454,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_45_score_79974_seed_1524_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_45_score_79974_seed_1524_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 234,
      "rank": 45,
      "score": 79974,
      "score_minus_starter": 20925,
      "seed": 1524,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_46_score_79896_seed_1451_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_46_score_79896_seed_1451_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 108,
      "rank": 46,
      "score": 79896,
      "score_minus_starter": 20847,
      "seed": 1451,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_47_score_79635_seed_1533_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_47_score_79635_seed_1533_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 122,
      "rank": 47,
      "score": 79635,
      "score_minus_starter": 20586,
      "seed": 1533,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_48_score_77352_seed_1508_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_48_score_77352_seed_1508_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 384,
      "moves": 130,
      "rank": 48,
      "score": 77352,
      "score_minus_starter": 18303,
      "seed": 1508,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_49_score_76893_seed_1509_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_49_score_76893_seed_1509_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 384,
      "moves": 141,
      "rank": 49,
      "score": 76893,
      "score_minus_starter": 17844,
      "seed": 1509,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_50_score_75456_seed_1538_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_1450_1550_keep50_20260708/top_games/rank_50_score_75456_seed_1538_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 384,
      "moves": 121,
      "rank": 50,
      "score": 75456,
      "score_minus_starter": 16407,
      "seed": 1538,
      "starter_tile": 1536
    }
  ]
}
```

## Eval: ntuple_phaseblend_expectimax1b:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax1b:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 1600:1620 --starter 1536 --progress-every 5 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708 --keep-top-games 10 --keep-milestone-games 3072 --keep-milestone-limit 0 --charts`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/death_forensics.json"
  },
  "games": 20,
  "high_score": 243540,
  "high_score_minus_starter": 184491,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.05,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.05,
    ">=192": 1.0,
    ">=3072": 0.05,
    ">=384": 0.95,
    ">=6144": 0.0,
    ">=768": 0.55
  },
  "mean_moves": 180.1,
  "mean_score": 85328.4,
  "mean_score_minus_starter": 26279.4,
  "median_moves": 169.0,
  "median_score": 80140.5,
  "median_score_minus_starter": 21091.5,
  "milestone_games": {
    "max_games": 0,
    "qualified_games": 1,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/milestone_games/ge_3072/seed_1618_score_243540_starter_1536/replay.html",
        "index": 1,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/milestone_games/ge_3072/seed_1618_score_243540_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 431,
        "score": 243540,
        "score_minus_starter": 184491,
        "seed": 1618,
        "starter_tile": 1536
      }
    ],
    "threshold": 3072
  },
  "p90_moves": 244,
  "p90_score": 86511,
  "p90_score_minus_starter": 27462,
  "p_max_tile_excl_starter_ge_1536": 0.05,
  "p_max_tile_excl_starter_ge_3072": 0.05,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_01_score_243540_seed_1618_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_01_score_243540_seed_1618_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 431,
      "rank": 1,
      "score": 243540,
      "score_minus_starter": 184491,
      "seed": 1618,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_02_score_89310_seed_1609_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_02_score_89310_seed_1609_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 168,
      "rank": 2,
      "score": 89310,
      "score_minus_starter": 30261,
      "seed": 1609,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_03_score_86511_seed_1601_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_03_score_86511_seed_1601_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 244,
      "rank": 3,
      "score": 86511,
      "score_minus_starter": 27462,
      "seed": 1601,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_04_score_86349_seed_1612_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_04_score_86349_seed_1612_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 195,
      "rank": 4,
      "score": 86349,
      "score_minus_starter": 27300,
      "seed": 1612,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_05_score_84336_seed_1605_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_05_score_84336_seed_1605_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 176,
      "rank": 5,
      "score": 84336,
      "score_minus_starter": 25287,
      "seed": 1605,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_06_score_82560_seed_1606_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_06_score_82560_seed_1606_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 181,
      "rank": 6,
      "score": 82560,
      "score_minus_starter": 23511,
      "seed": 1606,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_07_score_82506_seed_1610_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_07_score_82506_seed_1610_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 266,
      "rank": 7,
      "score": 82506,
      "score_minus_starter": 23457,
      "seed": 1610,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_08_score_82137_seed_1613_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_08_score_82137_seed_1613_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 162,
      "rank": 8,
      "score": 82137,
      "score_minus_starter": 23088,
      "seed": 1613,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_09_score_81978_seed_1614_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_09_score_81978_seed_1614_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 145,
      "rank": 9,
      "score": 81978,
      "score_minus_starter": 22929,
      "seed": 1614,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_10_score_80418_seed_1607_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend1b_incumbent_tailhunt_1600_1620_keep10_20260708/top_games/rank_10_score_80418_seed_1607_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 182,
      "rank": 10,
      "score": 80418,
      "score_minus_starter": 21369,
      "seed": 1607,
      "starter_tile": 1536
    }
  ]
}
```

## Eval: ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 1820:1920 --starter 1536 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708 --keep-top-games 3 --keep-milestone-games 1536 --keep-milestone-limit 30 --keep-pre-milestone-failures 3072 --keep-pre-milestone-min 1536 --keep-pre-milestone-limit 30 --checkpoint-results --charts --jobs 4 --progress-every 10`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/death_forensics.json"
  },
  "games": 100,
  "high_score": 186402,
  "high_score_minus_starter": 127353,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.04,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.04,
    ">=192": 1.0,
    ">=3072": 0.04,
    ">=384": 0.9,
    ">=6144": 0.0,
    ">=768": 0.49
  },
  "mean_moves": 202.8,
  "mean_score": 81244.74,
  "mean_score_minus_starter": 22195.74,
  "median_moves": 198.0,
  "median_score": 77286.0,
  "median_score_minus_starter": 18237.0,
  "milestone_games": {
    "max_games": 30,
    "qualified_games": 4,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/milestone_games/ge_1536/seed_1833_score_180393_starter_1536/replay.html",
        "index": 1,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/milestone_games/ge_1536/seed_1833_score_180393_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 252,
        "score": 180393,
        "score_minus_starter": 121344,
        "seed": 1833,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/milestone_games/ge_1536/seed_1884_score_186402_starter_1536/replay.html",
        "index": 2,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/milestone_games/ge_1536/seed_1884_score_186402_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 373,
        "score": 186402,
        "score_minus_starter": 127353,
        "seed": 1884,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/milestone_games/ge_1536/seed_1916_score_179766_starter_1536/replay.html",
        "index": 3,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/milestone_games/ge_1536/seed_1916_score_179766_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 278,
        "score": 179766,
        "score_minus_starter": 120717,
        "seed": 1916,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/milestone_games/ge_1536/seed_1919_score_180540_starter_1536/replay.html",
        "index": 4,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/milestone_games/ge_1536/seed_1919_score_180540_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 263,
        "score": 180540,
        "score_minus_starter": 121491,
        "seed": 1919,
        "starter_tile": 1536
      }
    ],
    "threshold": 1536
  },
  "p90_moves": 285,
  "p90_score": 88695,
  "p90_score_minus_starter": 29646,
  "p_max_tile_excl_starter_ge_1536": 0.04,
  "p_max_tile_excl_starter_ge_3072": 0.04,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "pre_milestone_failure_games": {
    "max_games": 30,
    "min_tile": 1536,
    "qualified_games": 0,
    "replays": [],
    "retained_games": 0,
    "threshold": 3072
  },
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/top_games/rank_01_score_186402_seed_1884_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/top_games/rank_01_score_186402_seed_1884_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 373,
      "rank": 1,
      "score": 186402,
      "score_minus_starter": 127353,
      "seed": 1884,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/top_games/rank_02_score_180540_seed_1919_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/top_games/rank_02_score_180540_seed_1919_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 263,
      "rank": 2,
      "score": 180540,
      "score_minus_starter": 121491,
      "seed": 1919,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/top_games/rank_03_score_180393_seed_1833_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1820_1920_20260708/top_games/rank_03_score_180393_seed_1833_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 252,
      "rank": 3,
      "score": 180393,
      "score_minus_starter": 121344,
      "seed": 1833,
      "starter_tile": 1536
    }
  ]
}
```

## Eval: ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 1920:2020 --starter 1536 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708 --keep-top-games 3 --keep-milestone-games 1536 --keep-milestone-limit 30 --keep-pre-milestone-failures 3072 --keep-pre-milestone-min 1536 --keep-pre-milestone-limit 30 --checkpoint-results --charts --jobs 6 --progress-every 10`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/death_forensics.json"
  },
  "games": 100,
  "high_score": 205413,
  "high_score_minus_starter": 146364,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.03,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.03,
    ">=192": 0.98,
    ">=3072": 0.03,
    ">=384": 0.83,
    ">=6144": 0.0,
    ">=768": 0.37
  },
  "mean_moves": 178.27,
  "mean_score": 77391.48,
  "mean_score_minus_starter": 18342.48,
  "median_moves": 177.0,
  "median_score": 70413.0,
  "median_score_minus_starter": 11364.0,
  "milestone_games": {
    "max_games": 30,
    "qualified_games": 3,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/milestone_games/ge_1536/seed_1926_score_205413_starter_1536/replay.html",
        "index": 1,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/milestone_games/ge_1536/seed_1926_score_205413_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 349,
        "score": 205413,
        "score_minus_starter": 146364,
        "seed": 1926,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/milestone_games/ge_1536/seed_1970_score_183129_starter_1536/replay.html",
        "index": 2,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/milestone_games/ge_1536/seed_1970_score_183129_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 261,
        "score": 183129,
        "score_minus_starter": 124080,
        "seed": 1970,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/milestone_games/ge_1536/seed_1975_score_186462_starter_1536/replay.html",
        "index": 3,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/milestone_games/ge_1536/seed_1975_score_186462_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 369,
        "score": 186462,
        "score_minus_starter": 127413,
        "seed": 1975,
        "starter_tile": 1536
      }
    ],
    "threshold": 1536
  },
  "p90_moves": 258,
  "p90_score": 88665,
  "p90_score_minus_starter": 29616,
  "p_max_tile_excl_starter_ge_1536": 0.03,
  "p_max_tile_excl_starter_ge_3072": 0.03,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "pre_milestone_failure_games": {
    "max_games": 30,
    "min_tile": 1536,
    "qualified_games": 0,
    "replays": [],
    "retained_games": 0,
    "threshold": 3072
  },
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/top_games/rank_01_score_205413_seed_1926_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/top_games/rank_01_score_205413_seed_1926_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 349,
      "rank": 1,
      "score": 205413,
      "score_minus_starter": 146364,
      "seed": 1926,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/top_games/rank_02_score_186462_seed_1975_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/top_games/rank_02_score_186462_seed_1975_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 369,
      "rank": 2,
      "score": 186462,
      "score_minus_starter": 127413,
      "seed": 1975,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/top_games/rank_03_score_183129_seed_1970_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_failures_1920_2020_20260708/top_games/rank_03_score_183129_seed_1970_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 261,
      "rank": 3,
      "score": 183129,
      "score_minus_starter": 124080,
      "seed": 1970,
      "starter_tile": 1536
    }
  ]
}
```

## Eval: ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 2020:2120 --starter 1536 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708 --keep-top-games 3 --keep-milestone-games 1536 --keep-milestone-limit 30 --keep-pre-milestone-failures 1536 --keep-pre-milestone-min 768 --keep-pre-milestone-limit 30 --checkpoint-results --charts --jobs 8 --progress-every 10`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/death_forensics.json"
  },
  "games": 100,
  "high_score": 204057,
  "high_score_minus_starter": 145008,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.07,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.07,
    ">=192": 0.99,
    ">=3072": 0.07,
    ">=384": 0.84,
    ">=6144": 0.0,
    ">=768": 0.43
  },
  "mean_moves": 184.61,
  "mean_score": 82080.39,
  "mean_score_minus_starter": 23031.39,
  "median_moves": 175.5,
  "median_score": 71113.5,
  "median_score_minus_starter": 12064.5,
  "milestone_games": {
    "max_games": 30,
    "qualified_games": 7,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2028_score_194580_starter_1536/replay.html",
        "index": 1,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2028_score_194580_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 240,
        "score": 194580,
        "score_minus_starter": 135531,
        "seed": 2028,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2033_score_186948_starter_1536/replay.html",
        "index": 2,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2033_score_186948_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 293,
        "score": 186948,
        "score_minus_starter": 127899,
        "seed": 2033,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2037_score_182826_starter_1536/replay.html",
        "index": 3,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2037_score_182826_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 359,
        "score": 182826,
        "score_minus_starter": 123777,
        "seed": 2037,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2081_score_204057_starter_1536/replay.html",
        "index": 4,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2081_score_204057_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 354,
        "score": 204057,
        "score_minus_starter": 145008,
        "seed": 2081,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2097_score_199671_starter_1536/replay.html",
        "index": 5,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2097_score_199671_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 393,
        "score": 199671,
        "score_minus_starter": 140622,
        "seed": 2097,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2103_score_184989_starter_1536/replay.html",
        "index": 6,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2103_score_184989_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 351,
        "score": 184989,
        "score_minus_starter": 125940,
        "seed": 2103,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2109_score_177888_starter_1536/replay.html",
        "index": 7,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/milestone_games/ge_1536/seed_2109_score_177888_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 279,
        "score": 177888,
        "score_minus_starter": 118839,
        "seed": 2109,
        "starter_tile": 1536
      }
    ],
    "threshold": 1536
  },
  "p90_moves": 293,
  "p90_score": 88716,
  "p90_score_minus_starter": 29667,
  "p_max_tile_excl_starter_ge_1536": 0.07,
  "p_max_tile_excl_starter_ge_3072": 0.07,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "pre_milestone_failure_games": {
    "max_games": 30,
    "min_tile": 768,
    "qualified_games": 36,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_01_score_99294_seed_2104_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_01_score_99294_seed_2104_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 208,
        "rank": 1,
        "score": 99294,
        "score_minus_starter": 40245,
        "seed": 2104,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_02_score_89148_seed_2088_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_02_score_89148_seed_2088_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 250,
        "rank": 2,
        "score": 89148,
        "score_minus_starter": 30099,
        "seed": 2088,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_03_score_89079_seed_2042_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_03_score_89079_seed_2042_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 188,
        "rank": 3,
        "score": 89079,
        "score_minus_starter": 30030,
        "seed": 2042,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_04_score_88716_seed_2076_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_04_score_88716_seed_2076_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 293,
        "rank": 4,
        "score": 88716,
        "score_minus_starter": 29667,
        "seed": 2076,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_05_score_88572_seed_2094_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_05_score_88572_seed_2094_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 353,
        "rank": 5,
        "score": 88572,
        "score_minus_starter": 29523,
        "seed": 2094,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_06_score_88401_seed_2032_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_06_score_88401_seed_2032_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 282,
        "rank": 6,
        "score": 88401,
        "score_minus_starter": 29352,
        "seed": 2032,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_07_score_88170_seed_2065_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_07_score_88170_seed_2065_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 236,
        "rank": 7,
        "score": 88170,
        "score_minus_starter": 29121,
        "seed": 2065,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_08_score_87450_seed_2050_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_08_score_87450_seed_2050_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 254,
        "rank": 8,
        "score": 87450,
        "score_minus_starter": 28401,
        "seed": 2050,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_09_score_86721_seed_2118_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_09_score_86721_seed_2118_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 194,
        "rank": 9,
        "score": 86721,
        "score_minus_starter": 27672,
        "seed": 2118,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_10_score_86601_seed_2077_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_10_score_86601_seed_2077_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 218,
        "rank": 10,
        "score": 86601,
        "score_minus_starter": 27552,
        "seed": 2077,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_11_score_85935_seed_2098_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_11_score_85935_seed_2098_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 231,
        "rank": 11,
        "score": 85935,
        "score_minus_starter": 26886,
        "seed": 2098,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_12_score_84444_seed_2047_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_12_score_84444_seed_2047_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 330,
        "rank": 12,
        "score": 84444,
        "score_minus_starter": 25395,
        "seed": 2047,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_13_score_84210_seed_2079_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_13_score_84210_seed_2079_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 224,
        "rank": 13,
        "score": 84210,
        "score_minus_starter": 25161,
        "seed": 2079,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_14_score_84162_seed_2044_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_14_score_84162_seed_2044_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 331,
        "rank": 14,
        "score": 84162,
        "score_minus_starter": 25113,
        "seed": 2044,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_15_score_84132_seed_2061_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_15_score_84132_seed_2061_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 220,
        "rank": 15,
        "score": 84132,
        "score_minus_starter": 25083,
        "seed": 2061,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_16_score_84057_seed_2059_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_16_score_84057_seed_2059_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 176,
        "rank": 16,
        "score": 84057,
        "score_minus_starter": 25008,
        "seed": 2059,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_17_score_84048_seed_2043_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_17_score_84048_seed_2043_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 299,
        "rank": 17,
        "score": 84048,
        "score_minus_starter": 24999,
        "seed": 2043,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_18_score_83526_seed_2034_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_18_score_83526_seed_2034_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 175,
        "rank": 18,
        "score": 83526,
        "score_minus_starter": 24477,
        "seed": 2034,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_19_score_82995_seed_2087_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_19_score_82995_seed_2087_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 225,
        "rank": 19,
        "score": 82995,
        "score_minus_starter": 23946,
        "seed": 2087,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_20_score_82521_seed_2035_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_20_score_82521_seed_2035_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 227,
        "rank": 20,
        "score": 82521,
        "score_minus_starter": 23472,
        "seed": 2035,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_21_score_82131_seed_2096_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_21_score_82131_seed_2096_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 172,
        "rank": 21,
        "score": 82131,
        "score_minus_starter": 23082,
        "seed": 2096,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_22_score_81891_seed_2095_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_22_score_81891_seed_2095_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 219,
        "rank": 22,
        "score": 81891,
        "score_minus_starter": 22842,
        "seed": 2095,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_23_score_81873_seed_2051_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_23_score_81873_seed_2051_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 164,
        "rank": 23,
        "score": 81873,
        "score_minus_starter": 22824,
        "seed": 2051,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_24_score_81777_seed_2090_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_24_score_81777_seed_2090_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 184,
        "rank": 24,
        "score": 81777,
        "score_minus_starter": 22728,
        "seed": 2090,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_25_score_81582_seed_2101_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_25_score_81582_seed_2101_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 186,
        "rank": 25,
        "score": 81582,
        "score_minus_starter": 22533,
        "seed": 2101,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_26_score_81579_seed_2107_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_26_score_81579_seed_2107_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 277,
        "rank": 26,
        "score": 81579,
        "score_minus_starter": 22530,
        "seed": 2107,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_27_score_81501_seed_2024_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_27_score_81501_seed_2024_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 249,
        "rank": 27,
        "score": 81501,
        "score_minus_starter": 22452,
        "seed": 2024,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_28_score_81183_seed_2046_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_28_score_81183_seed_2046_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 165,
        "rank": 28,
        "score": 81183,
        "score_minus_starter": 22134,
        "seed": 2046,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_29_score_80835_seed_2082_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_29_score_80835_seed_2082_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 224,
        "rank": 29,
        "score": 80835,
        "score_minus_starter": 21786,
        "seed": 2082,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_30_score_80835_seed_2093_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/diagnostic_games/pre_1536_min_768/rank_30_score_80835_seed_2093_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 198,
        "rank": 30,
        "score": 80835,
        "score_minus_starter": 21786,
        "seed": 2093,
        "starter_tile": 1536
      }
    ],
    "retained_games": 30,
    "threshold": 1536
  },
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/top_games/rank_01_score_204057_seed_2081_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/top_games/rank_01_score_204057_seed_2081_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 354,
      "rank": 1,
      "score": 204057,
      "score_minus_starter": 145008,
      "seed": 2081,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/top_games/rank_02_score_199671_seed_2097_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/top_games/rank_02_score_199671_seed_2097_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 393,
      "rank": 2,
      "score": 199671,
      "score_minus_starter": 140622,
      "seed": 2097,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/top_games/rank_03_score_194580_seed_2028_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2020_2120_20260708/top_games/rank_03_score_194580_seed_2028_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 240,
      "rank": 3,
      "score": 194580,
      "score_minus_starter": 135531,
      "seed": 2028,
      "starter_tile": 1536
    }
  ]
}
```

## Eval: ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 2120:2220 --starter 1536 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708 --keep-top-games 3 --keep-milestone-games 1536 --keep-milestone-limit 30 --keep-pre-milestone-failures 1536 --keep-pre-milestone-min 768 --keep-pre-milestone-limit 30 --checkpoint-results --charts --jobs 8 --progress-every 10`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/death_forensics.json"
  },
  "games": 100,
  "high_score": 226143,
  "high_score_minus_starter": 167094,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.05,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.05,
    ">=192": 0.99,
    ">=3072": 0.05,
    ">=384": 0.89,
    ">=6144": 0.0,
    ">=768": 0.37
  },
  "mean_moves": 181.44,
  "mean_score": 80597.79,
  "mean_score_minus_starter": 21548.79,
  "median_moves": 163.0,
  "median_score": 69864.0,
  "median_score_minus_starter": 10815.0,
  "milestone_games": {
    "max_games": 30,
    "qualified_games": 5,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2124_score_200190_starter_1536/replay.html",
        "index": 1,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2124_score_200190_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 412,
        "score": 200190,
        "score_minus_starter": 141141,
        "seed": 2124,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2127_score_200175_starter_1536/replay.html",
        "index": 2,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2127_score_200175_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 284,
        "score": 200175,
        "score_minus_starter": 141126,
        "seed": 2127,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2141_score_199581_starter_1536/replay.html",
        "index": 3,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2141_score_199581_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 439,
        "score": 199581,
        "score_minus_starter": 140532,
        "seed": 2141,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2164_score_226143_starter_1536/replay.html",
        "index": 4,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2164_score_226143_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 449,
        "score": 226143,
        "score_minus_starter": 167094,
        "seed": 2164,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2181_score_198351_starter_1536/replay.html",
        "index": 5,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/milestone_games/ge_1536/seed_2181_score_198351_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 325,
        "score": 198351,
        "score_minus_starter": 139302,
        "seed": 2181,
        "starter_tile": 1536
      }
    ],
    "threshold": 1536
  },
  "p90_moves": 280,
  "p90_score": 89106,
  "p90_score_minus_starter": 30057,
  "p_max_tile_excl_starter_ge_1536": 0.05,
  "p_max_tile_excl_starter_ge_3072": 0.05,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "pre_milestone_failure_games": {
    "max_games": 30,
    "min_tile": 768,
    "qualified_games": 32,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_01_score_103092_seed_2125_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_01_score_103092_seed_2125_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 240,
        "rank": 1,
        "score": 103092,
        "score_minus_starter": 44043,
        "seed": 2125,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_02_score_90978_seed_2182_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_02_score_90978_seed_2182_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 157,
        "rank": 2,
        "score": 90978,
        "score_minus_starter": 31929,
        "seed": 2182,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_03_score_90636_seed_2200_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_03_score_90636_seed_2200_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 175,
        "rank": 3,
        "score": 90636,
        "score_minus_starter": 31587,
        "seed": 2200,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_04_score_90090_seed_2156_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_04_score_90090_seed_2156_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 247,
        "rank": 4,
        "score": 90090,
        "score_minus_starter": 31041,
        "seed": 2156,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_05_score_89172_seed_2159_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_05_score_89172_seed_2159_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 283,
        "rank": 5,
        "score": 89172,
        "score_minus_starter": 30123,
        "seed": 2159,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_06_score_89106_seed_2199_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_06_score_89106_seed_2199_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 244,
        "rank": 6,
        "score": 89106,
        "score_minus_starter": 30057,
        "seed": 2199,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_07_score_88581_seed_2205_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_07_score_88581_seed_2205_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 208,
        "rank": 7,
        "score": 88581,
        "score_minus_starter": 29532,
        "seed": 2205,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_08_score_88473_seed_2207_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_08_score_88473_seed_2207_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 221,
        "rank": 8,
        "score": 88473,
        "score_minus_starter": 29424,
        "seed": 2207,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_09_score_88311_seed_2129_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_09_score_88311_seed_2129_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 201,
        "rank": 9,
        "score": 88311,
        "score_minus_starter": 29262,
        "seed": 2129,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_10_score_88002_seed_2170_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_10_score_88002_seed_2170_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 323,
        "rank": 10,
        "score": 88002,
        "score_minus_starter": 28953,
        "seed": 2170,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_11_score_87999_seed_2174_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_11_score_87999_seed_2174_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 326,
        "rank": 11,
        "score": 87999,
        "score_minus_starter": 28950,
        "seed": 2174,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_12_score_87534_seed_2161_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_12_score_87534_seed_2161_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 239,
        "rank": 12,
        "score": 87534,
        "score_minus_starter": 28485,
        "seed": 2161,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_13_score_86256_seed_2192_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_13_score_86256_seed_2192_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 293,
        "rank": 13,
        "score": 86256,
        "score_minus_starter": 27207,
        "seed": 2192,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_14_score_85947_seed_2157_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_14_score_85947_seed_2157_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 154,
        "rank": 14,
        "score": 85947,
        "score_minus_starter": 26898,
        "seed": 2157,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_15_score_85866_seed_2122_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_15_score_85866_seed_2122_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 211,
        "rank": 15,
        "score": 85866,
        "score_minus_starter": 26817,
        "seed": 2122,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_16_score_84318_seed_2176_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_16_score_84318_seed_2176_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 153,
        "rank": 16,
        "score": 84318,
        "score_minus_starter": 25269,
        "seed": 2176,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_17_score_84291_seed_2175_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_17_score_84291_seed_2175_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 264,
        "rank": 17,
        "score": 84291,
        "score_minus_starter": 25242,
        "seed": 2175,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_18_score_84210_seed_2158_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_18_score_84210_seed_2158_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 199,
        "rank": 18,
        "score": 84210,
        "score_minus_starter": 25161,
        "seed": 2158,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_19_score_83787_seed_2123_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_19_score_83787_seed_2123_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 256,
        "rank": 19,
        "score": 83787,
        "score_minus_starter": 24738,
        "seed": 2123,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_20_score_83754_seed_2217_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_20_score_83754_seed_2217_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 261,
        "rank": 20,
        "score": 83754,
        "score_minus_starter": 24705,
        "seed": 2217,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_21_score_83091_seed_2180_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_21_score_83091_seed_2180_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 215,
        "rank": 21,
        "score": 83091,
        "score_minus_starter": 24042,
        "seed": 2180,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_22_score_82800_seed_2146_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_22_score_82800_seed_2146_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 227,
        "rank": 22,
        "score": 82800,
        "score_minus_starter": 23751,
        "seed": 2146,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_23_score_82317_seed_2218_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_23_score_82317_seed_2218_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 285,
        "rank": 23,
        "score": 82317,
        "score_minus_starter": 23268,
        "seed": 2218,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_24_score_82062_seed_2150_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_24_score_82062_seed_2150_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 233,
        "rank": 24,
        "score": 82062,
        "score_minus_starter": 23013,
        "seed": 2150,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_25_score_81615_seed_2153_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_25_score_81615_seed_2153_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 200,
        "rank": 25,
        "score": 81615,
        "score_minus_starter": 22566,
        "seed": 2153,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_26_score_81612_seed_2160_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_26_score_81612_seed_2160_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 224,
        "rank": 26,
        "score": 81612,
        "score_minus_starter": 22563,
        "seed": 2160,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_27_score_81429_seed_2173_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_27_score_81429_seed_2173_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 271,
        "rank": 27,
        "score": 81429,
        "score_minus_starter": 22380,
        "seed": 2173,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_28_score_80910_seed_2171_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_28_score_80910_seed_2171_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 203,
        "rank": 28,
        "score": 80910,
        "score_minus_starter": 21861,
        "seed": 2171,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_29_score_80139_seed_2215_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_29_score_80139_seed_2215_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 86,
        "rank": 29,
        "score": 80139,
        "score_minus_starter": 21090,
        "seed": 2215,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_30_score_80010_seed_2179_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/diagnostic_games/pre_1536_min_768/rank_30_score_80010_seed_2179_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 154,
        "rank": 30,
        "score": 80010,
        "score_minus_starter": 20961,
        "seed": 2179,
        "starter_tile": 1536
      }
    ],
    "retained_games": 30,
    "threshold": 1536
  },
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/top_games/rank_01_score_226143_seed_2164_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/top_games/rank_01_score_226143_seed_2164_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 449,
      "rank": 1,
      "score": 226143,
      "score_minus_starter": 167094,
      "seed": 2164,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/top_games/rank_02_score_200190_seed_2124_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/top_games/rank_02_score_200190_seed_2124_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 412,
      "rank": 2,
      "score": 200190,
      "score_minus_starter": 141141,
      "seed": 2124,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/top_games/rank_03_score_200175_seed_2127_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2120_2220_20260708/top_games/rank_03_score_200175_seed_2127_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 284,
      "rank": 3,
      "score": 200175,
      "score_minus_starter": 141126,
      "seed": 2127,
      "starter_tile": 1536
    }
  ]
}
```

## Eval: ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 2220:2320 --starter 1536 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708 --keep-top-games 3 --keep-milestone-games 1536 --keep-milestone-limit 30 --keep-pre-milestone-failures 1536 --keep-pre-milestone-min 768 --keep-pre-milestone-limit 30 --checkpoint-results --charts --jobs 8 --progress-every 10`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/death_forensics.json"
  },
  "games": 100,
  "high_score": 188808,
  "high_score_minus_starter": 129759,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.03,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.03,
    ">=192": 0.98,
    ">=3072": 0.03,
    ">=384": 0.9,
    ">=6144": 0.0,
    ">=768": 0.48
  },
  "mean_moves": 187.24,
  "mean_score": 80072.58,
  "mean_score_minus_starter": 21023.58,
  "median_moves": 177.0,
  "median_score": 76095.0,
  "median_score_minus_starter": 17046.0,
  "milestone_games": {
    "max_games": 30,
    "qualified_games": 3,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/milestone_games/ge_1536/seed_2257_score_188808_starter_1536/replay.html",
        "index": 1,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/milestone_games/ge_1536/seed_2257_score_188808_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 421,
        "score": 188808,
        "score_minus_starter": 129759,
        "seed": 2257,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/milestone_games/ge_1536/seed_2271_score_180741_starter_1536/replay.html",
        "index": 2,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/milestone_games/ge_1536/seed_2271_score_180741_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 319,
        "score": 180741,
        "score_minus_starter": 121692,
        "seed": 2271,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/milestone_games/ge_1536/seed_2300_score_187113_starter_1536/replay.html",
        "index": 3,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/milestone_games/ge_1536/seed_2300_score_187113_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 294,
        "score": 187113,
        "score_minus_starter": 128064,
        "seed": 2300,
        "starter_tile": 1536
      }
    ],
    "threshold": 1536
  },
  "p90_moves": 292,
  "p90_score": 88752,
  "p90_score_minus_starter": 29703,
  "p_max_tile_excl_starter_ge_1536": 0.03,
  "p_max_tile_excl_starter_ge_3072": 0.03,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "pre_milestone_failure_games": {
    "max_games": 30,
    "min_tile": 768,
    "qualified_games": 45,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_01_score_101847_seed_2306_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_01_score_101847_seed_2306_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 252,
        "rank": 1,
        "score": 101847,
        "score_minus_starter": 42798,
        "seed": 2306,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_02_score_90798_seed_2287_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_02_score_90798_seed_2287_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 244,
        "rank": 2,
        "score": 90798,
        "score_minus_starter": 31749,
        "seed": 2287,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_03_score_90777_seed_2220_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_03_score_90777_seed_2220_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 322,
        "rank": 3,
        "score": 90777,
        "score_minus_starter": 31728,
        "seed": 2220,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_04_score_90738_seed_2240_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_04_score_90738_seed_2240_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 220,
        "rank": 4,
        "score": 90738,
        "score_minus_starter": 31689,
        "seed": 2240,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_05_score_90078_seed_2319_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_05_score_90078_seed_2319_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 225,
        "rank": 5,
        "score": 90078,
        "score_minus_starter": 31029,
        "seed": 2319,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_06_score_89304_seed_2296_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_06_score_89304_seed_2296_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 360,
        "rank": 6,
        "score": 89304,
        "score_minus_starter": 30255,
        "seed": 2296,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_07_score_88773_seed_2269_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_07_score_88773_seed_2269_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 332,
        "rank": 7,
        "score": 88773,
        "score_minus_starter": 29724,
        "seed": 2269,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_08_score_88752_seed_2270_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_08_score_88752_seed_2270_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 207,
        "rank": 8,
        "score": 88752,
        "score_minus_starter": 29703,
        "seed": 2270,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_09_score_88671_seed_2248_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_09_score_88671_seed_2248_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 307,
        "rank": 9,
        "score": 88671,
        "score_minus_starter": 29622,
        "seed": 2248,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_10_score_88590_seed_2241_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_10_score_88590_seed_2241_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 200,
        "rank": 10,
        "score": 88590,
        "score_minus_starter": 29541,
        "seed": 2241,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_11_score_88587_seed_2284_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_11_score_88587_seed_2284_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 234,
        "rank": 11,
        "score": 88587,
        "score_minus_starter": 29538,
        "seed": 2284,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_12_score_88425_seed_2268_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_12_score_88425_seed_2268_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 136,
        "rank": 12,
        "score": 88425,
        "score_minus_starter": 29376,
        "seed": 2268,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_13_score_88368_seed_2310_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_13_score_88368_seed_2310_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 214,
        "rank": 13,
        "score": 88368,
        "score_minus_starter": 29319,
        "seed": 2310,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_14_score_88146_seed_2245_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_14_score_88146_seed_2245_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 240,
        "rank": 14,
        "score": 88146,
        "score_minus_starter": 29097,
        "seed": 2245,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_15_score_88029_seed_2259_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_15_score_88029_seed_2259_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 246,
        "rank": 15,
        "score": 88029,
        "score_minus_starter": 28980,
        "seed": 2259,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_16_score_87939_seed_2311_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_16_score_87939_seed_2311_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 207,
        "rank": 16,
        "score": 87939,
        "score_minus_starter": 28890,
        "seed": 2311,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_17_score_87282_seed_2232_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_17_score_87282_seed_2232_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 363,
        "rank": 17,
        "score": 87282,
        "score_minus_starter": 28233,
        "seed": 2232,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_18_score_86982_seed_2239_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_18_score_86982_seed_2239_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 313,
        "rank": 18,
        "score": 86982,
        "score_minus_starter": 27933,
        "seed": 2239,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_19_score_86946_seed_2289_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_19_score_86946_seed_2289_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 250,
        "rank": 19,
        "score": 86946,
        "score_minus_starter": 27897,
        "seed": 2289,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_20_score_86937_seed_2233_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_20_score_86937_seed_2233_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 224,
        "rank": 20,
        "score": 86937,
        "score_minus_starter": 27888,
        "seed": 2233,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_21_score_86553_seed_2244_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_21_score_86553_seed_2244_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 284,
        "rank": 21,
        "score": 86553,
        "score_minus_starter": 27504,
        "seed": 2244,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_22_score_86451_seed_2286_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_22_score_86451_seed_2286_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 250,
        "rank": 22,
        "score": 86451,
        "score_minus_starter": 27402,
        "seed": 2286,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_23_score_86418_seed_2295_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_23_score_86418_seed_2295_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 266,
        "rank": 23,
        "score": 86418,
        "score_minus_starter": 27369,
        "seed": 2295,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_24_score_86337_seed_2314_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_24_score_86337_seed_2314_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 179,
        "rank": 24,
        "score": 86337,
        "score_minus_starter": 27288,
        "seed": 2314,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_25_score_85752_seed_2229_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_25_score_85752_seed_2229_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 194,
        "rank": 25,
        "score": 85752,
        "score_minus_starter": 26703,
        "seed": 2229,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_26_score_85752_seed_2226_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_26_score_85752_seed_2226_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 193,
        "rank": 26,
        "score": 85752,
        "score_minus_starter": 26703,
        "seed": 2226,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_27_score_85716_seed_2293_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_27_score_85716_seed_2293_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 221,
        "rank": 27,
        "score": 85716,
        "score_minus_starter": 26667,
        "seed": 2293,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_28_score_84702_seed_2301_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_28_score_84702_seed_2301_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 263,
        "rank": 28,
        "score": 84702,
        "score_minus_starter": 25653,
        "seed": 2301,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_29_score_84228_seed_2250_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_29_score_84228_seed_2250_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 186,
        "rank": 29,
        "score": 84228,
        "score_minus_starter": 25179,
        "seed": 2250,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_30_score_84216_seed_2278_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/diagnostic_games/pre_1536_min_768/rank_30_score_84216_seed_2278_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 277,
        "rank": 30,
        "score": 84216,
        "score_minus_starter": 25167,
        "seed": 2278,
        "starter_tile": 1536
      }
    ],
    "retained_games": 30,
    "threshold": 1536
  },
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/top_games/rank_01_score_188808_seed_2257_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/top_games/rank_01_score_188808_seed_2257_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 421,
      "rank": 1,
      "score": 188808,
      "score_minus_starter": 129759,
      "seed": 2257,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/top_games/rank_02_score_187113_seed_2300_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/top_games/rank_02_score_187113_seed_2300_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 294,
      "rank": 2,
      "score": 187113,
      "score_minus_starter": 128064,
      "seed": 2300,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/top_games/rank_03_score_180741_seed_2271_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2220_2320_20260708/top_games/rank_03_score_180741_seed_2271_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 319,
      "rank": 3,
      "score": 180741,
      "score_minus_starter": 121692,
      "seed": 2271,
      "starter_tile": 1536
    }
  ]
}
```

## Eval: ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 2320:2420 --starter 1536 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full --keep-top-games 3 --keep-milestone-games 1536 --keep-milestone-limit 30 --keep-pre-milestone-failures 1536 --keep-pre-milestone-min 768 --keep-pre-milestone-limit 30 --progress-every 10 --checkpoint-results --charts --jobs 8`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/death_forensics.json"
  },
  "games": 100,
  "high_score": 213012,
  "high_score_minus_starter": 153963,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.02,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.02,
    ">=192": 0.99,
    ">=3072": 0.02,
    ">=384": 0.8,
    ">=6144": 0.0,
    ">=768": 0.36
  },
  "mean_moves": 182.97,
  "mean_score": 76489.71,
  "mean_score_minus_starter": 17440.71,
  "median_moves": 174.5,
  "median_score": 69414.0,
  "median_score_minus_starter": 10365.0,
  "milestone_games": {
    "max_games": 30,
    "qualified_games": 2,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/milestone_games/ge_1536/seed_2391_score_213012_starter_1536/replay.html",
        "index": 1,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/milestone_games/ge_1536/seed_2391_score_213012_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 463,
        "score": 213012,
        "score_minus_starter": 153963,
        "seed": 2391,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/milestone_games/ge_1536/seed_2415_score_187809_starter_1536/replay.html",
        "index": 2,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/milestone_games/ge_1536/seed_2415_score_187809_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 396,
        "score": 187809,
        "score_minus_starter": 128760,
        "seed": 2415,
        "starter_tile": 1536
      }
    ],
    "threshold": 1536
  },
  "p90_moves": 260,
  "p90_score": 89400,
  "p90_score_minus_starter": 30351,
  "p_max_tile_excl_starter_ge_1536": 0.02,
  "p_max_tile_excl_starter_ge_3072": 0.02,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "pre_milestone_failure_games": {
    "max_games": 30,
    "min_tile": 768,
    "qualified_games": 34,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_01_score_95169_seed_2364_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_01_score_95169_seed_2364_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 209,
        "rank": 1,
        "score": 95169,
        "score_minus_starter": 36120,
        "seed": 2364,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_02_score_93048_seed_2395_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_02_score_93048_seed_2395_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 253,
        "rank": 2,
        "score": 93048,
        "score_minus_starter": 33999,
        "seed": 2395,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_03_score_92223_seed_2362_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_03_score_92223_seed_2362_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 242,
        "rank": 3,
        "score": 92223,
        "score_minus_starter": 33174,
        "seed": 2362,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_04_score_90807_seed_2385_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_04_score_90807_seed_2385_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 357,
        "rank": 4,
        "score": 90807,
        "score_minus_starter": 31758,
        "seed": 2385,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_05_score_90087_seed_2339_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_05_score_90087_seed_2339_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 240,
        "rank": 5,
        "score": 90087,
        "score_minus_starter": 31038,
        "seed": 2339,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_06_score_90000_seed_2390_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_06_score_90000_seed_2390_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 245,
        "rank": 6,
        "score": 90000,
        "score_minus_starter": 30951,
        "seed": 2390,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_07_score_89613_seed_2331_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_07_score_89613_seed_2331_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 217,
        "rank": 7,
        "score": 89613,
        "score_minus_starter": 30564,
        "seed": 2331,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_08_score_89529_seed_2388_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_08_score_89529_seed_2388_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 220,
        "rank": 8,
        "score": 89529,
        "score_minus_starter": 30480,
        "seed": 2388,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_09_score_89400_seed_2402_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_09_score_89400_seed_2402_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 298,
        "rank": 9,
        "score": 89400,
        "score_minus_starter": 30351,
        "seed": 2402,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_10_score_88692_seed_2325_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_10_score_88692_seed_2325_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 197,
        "rank": 10,
        "score": 88692,
        "score_minus_starter": 29643,
        "seed": 2325,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_11_score_88527_seed_2409_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_11_score_88527_seed_2409_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 283,
        "rank": 11,
        "score": 88527,
        "score_minus_starter": 29478,
        "seed": 2409,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_12_score_88380_seed_2358_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_12_score_88380_seed_2358_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 277,
        "rank": 12,
        "score": 88380,
        "score_minus_starter": 29331,
        "seed": 2358,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_13_score_88131_seed_2378_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_13_score_88131_seed_2378_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 197,
        "rank": 13,
        "score": 88131,
        "score_minus_starter": 29082,
        "seed": 2378,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_14_score_87705_seed_2419_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_14_score_87705_seed_2419_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 297,
        "rank": 14,
        "score": 87705,
        "score_minus_starter": 28656,
        "seed": 2419,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_15_score_87648_seed_2417_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_15_score_87648_seed_2417_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 330,
        "rank": 15,
        "score": 87648,
        "score_minus_starter": 28599,
        "seed": 2417,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_16_score_86940_seed_2333_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_16_score_86940_seed_2333_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 174,
        "rank": 16,
        "score": 86940,
        "score_minus_starter": 27891,
        "seed": 2333,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_17_score_86667_seed_2354_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_17_score_86667_seed_2354_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 217,
        "rank": 17,
        "score": 86667,
        "score_minus_starter": 27618,
        "seed": 2354,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_18_score_86502_seed_2330_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_18_score_86502_seed_2330_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 202,
        "rank": 18,
        "score": 86502,
        "score_minus_starter": 27453,
        "seed": 2330,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_19_score_86298_seed_2405_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_19_score_86298_seed_2405_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 314,
        "rank": 19,
        "score": 86298,
        "score_minus_starter": 27249,
        "seed": 2405,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_20_score_86238_seed_2413_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_20_score_86238_seed_2413_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 304,
        "rank": 20,
        "score": 86238,
        "score_minus_starter": 27189,
        "seed": 2413,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_21_score_85563_seed_2324_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_21_score_85563_seed_2324_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 223,
        "rank": 21,
        "score": 85563,
        "score_minus_starter": 26514,
        "seed": 2324,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_22_score_83718_seed_2407_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_22_score_83718_seed_2407_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 259,
        "rank": 22,
        "score": 83718,
        "score_minus_starter": 24669,
        "seed": 2407,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_23_score_82866_seed_2346_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_23_score_82866_seed_2346_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 228,
        "rank": 23,
        "score": 82866,
        "score_minus_starter": 23817,
        "seed": 2346,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_24_score_82842_seed_2374_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_24_score_82842_seed_2374_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 192,
        "rank": 24,
        "score": 82842,
        "score_minus_starter": 23793,
        "seed": 2374,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_25_score_82779_seed_2361_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_25_score_82779_seed_2361_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 159,
        "rank": 25,
        "score": 82779,
        "score_minus_starter": 23730,
        "seed": 2361,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_26_score_82551_seed_2379_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_26_score_82551_seed_2379_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 190,
        "rank": 26,
        "score": 82551,
        "score_minus_starter": 23502,
        "seed": 2379,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_27_score_82323_seed_2400_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_27_score_82323_seed_2400_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 251,
        "rank": 27,
        "score": 82323,
        "score_minus_starter": 23274,
        "seed": 2400,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_28_score_82191_seed_2363_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_28_score_82191_seed_2363_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 229,
        "rank": 28,
        "score": 82191,
        "score_minus_starter": 23142,
        "seed": 2363,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_29_score_81579_seed_2411_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_29_score_81579_seed_2411_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 203,
        "rank": 29,
        "score": 81579,
        "score_minus_starter": 22530,
        "seed": 2411,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_30_score_81099_seed_2401_starter_1536/replay.html",
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/diagnostic_games/pre_1536_min_768/rank_30_score_81099_seed_2401_starter_1536/replay.json",
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "moves": 182,
        "rank": 30,
        "score": 81099,
        "score_minus_starter": 22050,
        "seed": 2401,
        "starter_tile": 1536
      }
    ],
    "retained_games": 30,
    "threshold": 1536
  },
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/top_games/rank_01_score_213012_seed_2391_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/top_games/rank_01_score_213012_seed_2391_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 463,
      "rank": 1,
      "score": 213012,
      "score_minus_starter": 153963,
      "seed": 2391,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/top_games/rank_02_score_187809_seed_2415_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/top_games/rank_02_score_187809_seed_2415_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 396,
      "rank": 2,
      "score": 187809,
      "score_minus_starter": 128760,
      "seed": 2415,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/top_games/rank_03_score_95169_seed_2364_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_milestone1536_nearfail_2320_2420_20260708_full/top_games/rank_03_score_95169_seed_2364_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 209,
      "rank": 3,
      "score": 95169,
      "score_minus_starter": 36120,
      "seed": 2364,
      "starter_tile": 1536
    }
  ]
}
```

## Eval: ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame

Command: `python -m threes_rl.eval --policy ntuple_phaseblend_expectimax2:threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest:threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest:0.25:all:threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest:0.05:mid:threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest:0.10:endgame --seeds 2720:2820 --starter 1536 --max-moves 5000 --artifact-dir threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708 --keep-top-games 3 --keep-milestone-games 3072 --keep-milestone-limit 0 --keep-pre-milestone-failures 3072 --keep-pre-milestone-min 1536 --keep-pre-milestone-limit 10 --checkpoint-results --charts --progress-every 10 --jobs 8`

```json
{
  "death_forensics": {
    "cases": 6,
    "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/death_forensics.html",
    "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/death_forensics.json"
  },
  "games": 100,
  "high_score": 193044,
  "high_score_minus_starter": 133995,
  "max_tile_dist": {
    ">=12288": 0.0,
    ">=1536": 1.0,
    ">=192": 1.0,
    ">=3072": 0.02,
    ">=384": 1.0,
    ">=6144": 0.0,
    ">=768": 1.0
  },
  "max_tile_excl_starter_dist": {
    ">=12288": 0.0,
    ">=1536": 0.02,
    ">=192": 1.0,
    ">=3072": 0.02,
    ">=384": 0.96,
    ">=6144": 0.0,
    ">=768": 0.43
  },
  "mean_moves": 187.02,
  "mean_score": 78258.12,
  "mean_score_minus_starter": 19209.12,
  "median_moves": 182.0,
  "median_score": 71856.0,
  "median_score_minus_starter": 12807.0,
  "milestone_games": {
    "max_games": 0,
    "qualified_games": 2,
    "replays": [
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/milestone_games/ge_3072/seed_2730_score_187035_starter_1536/replay.html",
        "index": 1,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/milestone_games/ge_3072/seed_2730_score_187035_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 392,
        "score": 187035,
        "score_minus_starter": 127986,
        "seed": 2730,
        "starter_tile": 1536
      },
      {
        "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/milestone_games/ge_3072/seed_2787_score_193044_starter_1536/replay.html",
        "index": 2,
        "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/milestone_games/ge_3072/seed_2787_score_193044_starter_1536/replay.json",
        "max_tile": 3072,
        "max_tile_excl_starter": 3072,
        "moves": 249,
        "score": 193044,
        "score_minus_starter": 133995,
        "seed": 2787,
        "starter_tile": 1536
      }
    ],
    "threshold": 3072
  },
  "p90_moves": 265,
  "p90_score": 88599,
  "p90_score_minus_starter": 29550,
  "p_max_tile_excl_starter_ge_1536": 0.02,
  "p_max_tile_excl_starter_ge_3072": 0.02,
  "p_max_tile_excl_starter_ge_6144": 0.0,
  "pre_milestone_failure_games": {
    "max_games": 10,
    "min_tile": 1536,
    "qualified_games": 0,
    "replays": [],
    "retained_games": 0,
    "threshold": 3072
  },
  "top_games": [
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/top_games/rank_01_score_193044_seed_2787_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/top_games/rank_01_score_193044_seed_2787_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 249,
      "rank": 1,
      "score": 193044,
      "score_minus_starter": 133995,
      "seed": 2787,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/top_games/rank_02_score_187035_seed_2730_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/top_games/rank_02_score_187035_seed_2730_starter_1536/replay.json",
      "max_tile": 3072,
      "max_tile_excl_starter": 3072,
      "moves": 392,
      "rank": 2,
      "score": 187035,
      "score_minus_starter": 127986,
      "seed": 2730,
      "starter_tile": 1536
    },
    {
      "html": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/top_games/rank_03_score_90825_seed_2720_starter_1536/replay.html",
      "json": "threes_rl/runs/eval_artifacts/ntuple_phaseblend_incumbent_tailhunt_2720_2820_keep3_milestone3072_20260708/top_games/rank_03_score_90825_seed_2720_starter_1536/replay.json",
      "max_tile": 1536,
      "max_tile_excl_starter": 768,
      "moves": 184,
      "rank": 3,
      "score": 90825,
      "score_minus_starter": 31776,
      "seed": 2720,
      "starter_tile": 1536
    }
  ]
}
```

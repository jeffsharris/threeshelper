# Threes RL Results

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

Best learned checkpoint so far: `threes_rl/runs/imitation_expectimax2_30k/latest.pt`.

## Eval: ppo:threes_rl/runs/ppo_final_200k/latest.pt

Command: `python -m threes_rl.eval --policy ppo:threes_rl/runs/ppo_final_200k/latest.pt --seeds 1000:1200`

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
  "mean_moves": 28.005,
  "mean_score": 59589.96,
  "median_score": 59182.5,
  "p90_score": 60024
}
```

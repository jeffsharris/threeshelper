# Threes RL Progress

See `threes_rl/SETUP.md` for clone/setup/run instructions,
`threes_rl/ML_FINDINGS.md` for the research summary and next-experiment
recommendations, and `threes_rl/EXPERIMENT_LOG.md` for the running experiment
trail.

## 2026-07-05

- M0 core simulator implemented in `threes_rl/sim.py`.
  - Move oracle equivalence: 20,000 random boards plus adversarial cases x 4 directions, zero disagreements with `state_hunt.simulate_base_move`.
  - Simulated transition validation: 50,000 non-terminal steps, zero invalid transitions via `state_hunt.validate_transition`.
  - Terminal rule: `6144 + 6144 -> 12288`, game ends without a spawn, `score_tile(12288) == 1594323`.
- M1 schedule and replay validation implemented.
  - Tile-cycle lock-step: 100 episodes, at least 1,000 checked updates, exact `TileCycle.snapshot()` matches.
  - Replay validation: 270/270 replayable valid single-step observed moves reproduced exactly, 100.00%.
- M2 environment and benchmark implemented.
  - Gymnasium API check passes.
  - Determinism test: two envs with the same seed and same random-legal policy match for 1,000 steps.
  - Benchmark command `python -m threes_rl.bench`: raw sim 11616.54 steps/s, env 9254.09 steps/s, expectimax-d2 65.47 moves/s.
- PPO smoke run completed.
  - Command: `python -m threes_rl.train_ppo --run-name smoke --total-steps 512 --num-envs 4 --rollout-steps 32 --minibatch-size 64 --update-epochs 1 --checkpoint-interval 512 --device cpu`
  - Latest checkpoint: `threes_rl/runs/smoke/latest.pt`.
  - This verifies checkpoint writing only; it is not a meaningful policy.
- Classical baseline status.
  - Random 200 seeds: mean score 59600.21, mean moves 30.40.
  - Greedy 200 seeds: mean score 60304.29, mean moves 40.44.
  - Expectimax-d2 200 seeds: mean score 66460.14, mean moves 107.95.
  - Expectimax-d3 probe 5 seeds: mean score 76954.80, mean moves 142.20. Full 50-seed d3 was deferred because exact d3 evaluation measured roughly 40 seconds/game on this machine.
- Learned policy status.
  - Sparse final-score PPO from scratch was stable at first but did not beat greedy; after longer resume it collapsed toward shorter games, so it is not the best path without a stronger warm start.
  - Greedy imitation 200k samples reached 63.29% training accuracy and 60109.77 mean score on the fixed 200-seed eval, slightly below the greedy expert.
  - Expectimax-d2 imitation 10k samples reached 59.60% training accuracy and 60221.66 mean score.
  - Expectimax-d2 imitation 30k samples reached 62.06% training accuracy and 60848.28 mean score on the fixed 200-seed eval, beating greedy by 543.99 mean score.
  - Expectimax-d2 imitation 100k samples improved with more epochs up to 61385.97 mean score at 30 epochs, then regressed to 60896.19 at 60 epochs despite higher training accuracy.
  - Expectimax-d2 imitation 200k samples with epoch checkpoints peaked at epoch 20: mean score 61826.34, mean moves 60.81 on the fixed 200-seed eval.
  - Wider 1000-seed sanity check: greedy mean score 60224.63, best learned checkpoint mean score 61489.74.
  - Expectimax-d2 imitation 400k samples on the 12-logical-CPU machine with 10 workers generated labels at about 959 samples/s. Epoch 25 was best among saved checkpoints: mean score 62154.87, mean moves 63.42 on the fixed 200-seed eval.
  - Wider 1000-seed sanity check for the 400k epoch-25 checkpoint: mean score 61814.26, mean moves 62.12.
  - Best learned checkpoint so far: `threes_rl/runs/imitation_expectimax2_400k_w10_e30_20260705/checkpoint_epoch_25.pt`.
- Simulator correction.
  - The real game always starts the 1536 starter tile in the top-left corner.
  - `ThreesSim.reset()` now fixes `starter_tile` at row 0, col 0 and samples the 8 small starting tiles from the remaining 15 cells.
  - Previous training/eval results should be treated as pre-fix baselines until rerun.
- Corrected-start search baseline.
  - Added `corner2` / `corner3`, a corner-aware expectimax evaluator that keeps the old `expectimax2` baseline intact.
  - `corner2`, seeds 1000:1200: mean score 74323.59, mean score-minus-starter 15274.59, mean moves 163.785.
  - It built a non-starter 3072 in 3/200 games (`P(max tile excl. starter >= 3072) == 0.015`) but still reached 6144 in 0/200 games.
  - Top-three replay artifacts and progress chart live under `threes_rl/runs/eval_artifacts/corner2_1000_1200_fixed_starter/`.
- TD n-tuple afterstate learner.
  - Added `threes_rl/ntuple.py` and `threes_rl/train_td.py` with default n-tuple tables, checkpointing, progress charts, and top-three replay retention.
  - Corrected the TD action/update logic to use `ThreesSim.transition_outcomes()` for exact expected spawn slots, preview value/candidate probabilities, and next-preview probabilities.
  - Added `--init-total` so optimistic initialization is specified as total board value rather than per-feature table value.
  - 100-game zero-init smoke (`td_default_expected_100_a005_20260705`): held-out seeds 1000:1050 mean score-minus-starter 1588.14, mean moves 45.42, high score 68685.
  - 100-game optimistic smoke (`td_default_expected_100_init3000_a005_20260705`): held-out seeds 1000:1050 mean score-minus-starter 4441.62, mean moves 79.36, high score 81321.
  - 500-game optimistic run (`td_default_expected_500_init3000_a005_20260705`): training high score 86520, mean score-minus-starter 4857.816, mean moves 88.462; top-three training replays live under `threes_rl/runs/td_default_expected_500_init3000_a005_20260705/top_games/`.
  - 500-game checkpoint eval, held-out seeds 1000:1050: mean score 65842.98, mean score-minus-starter 6793.98, mean moves 100.44, high score 88443, `P(max tile excl. starter >= 1536) == 0.0`.
  - Interpretation: optimistic TD is learning survival and non-starter 768s, but short pure self-play still does not produce the non-starter 1536/3072 signal needed for the 6144 goal.
- Learned-leaf search and actor bootstrap.
  - Added `NtupleExpectimaxPolicy` with specs `ntuple_expectimax2:<checkpoint>` and `ntuple_expectimax2a:<checkpoint>`.
  - The learned-search policy adds exact simulator `score_delta` at chance branches and uses the n-tuple table only as remaining future value.
  - `ntuple_expectimax2` on the self-play TD-500 checkpoint, seeds 1000:1010: mean score 68886, mean score-minus-starter 9837, mean moves 129.2, high score 81852.
  - Added `train_td --actor-policy <policy> --target-mode mc`, which trains n-tuple afterstates on actual remaining returns from stronger actor trajectories while preserving checkpoints, charts, and top-three replays.
  - `td_default_corner2_mc_50_init3000_a005_20260705`: 50 `corner2` actor games, training high score 194271 with non-starter 3072, mean score-minus-starter 14968.5. Top-three training replays live under `threes_rl/runs/td_default_corner2_mc_50_init3000_a005_20260705/top_games/`.
  - `ntuple_expectimax2` on the corner2-MC checkpoint, seeds 1000:1010: mean score 72881.7, mean score-minus-starter 13832.7, mean moves 159.3, high score 87888. This is better than the self-play value checkpoint but still below the 200-seed `corner2` baseline and has no held-out non-starter 1536.
  - Scaling corner2-MC to 200 games with `alpha=0.05` produced the same strong actor data but a worse learned-search policy: seeds 1000:1020 mean score-minus-starter 9035.25. This suggests the online MC updates were too hot.
  - Re-running the same 200 actor games with `alpha=0.01` produced the current best learned-search result: `ntuple_expectimax2`, seeds 1000:1050, mean score 77139.78, mean score-minus-starter 18090.78, high score 205719, `P(max tile excl. starter >= 3072) == 0.04`.
  - Top-three eval replays for that result live under `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1050/top_games/`; the best held-out replay is seed 1049 with score 205719 and non-starter 3072.
  - Full-suite combined eval, seeds 1000:1200: mean score 74300.805, mean score-minus-starter 15251.805, high score 205719, `P(max tile excl. starter >= 3072) == 0.015`, `P(max tile excl. starter >= 6144) == 0.0`.
  - Interpretation after full eval: this learned-search policy is essentially tied with `corner2` on mean score-minus-starter (15251.805 vs 15274.59), with a slightly higher high score (205719 vs 204675), and the same non-starter 3072 rate (3/200).
  - Combined full-suite top-three replays and progress chart live under `threes_rl/runs/eval_artifacts/ntuple_expectimax2_corner2_mc_200_a001_1000_1200_full/`.
  - Optimized a searched-eval hot path in `score_board()` so cached score-table values no longer call `score_tile()` unnecessarily.
  - `ntuple_expectimax2a` adaptive depth was interrupted after roughly 90 seconds without completing the first game; it needs profiling/caching before regular eval.
- Handoff / Mac Mini resume notes.
  - Current bottleneck for better learned policies is expectimax-d2 data generation. Serial generation measured roughly 82-88 expert samples/s; the parallel trainer reached roughly 470-517 samples/s here with 8 workers.
  - Extra RAM alone is not currently important. A beefier machine helps if it has more or faster CPU cores for expectimax label generation.
  - To scale the best path on another machine: run `python -m threes_rl.train_imitation --run-name imitation_expectimax2_400k_wN_e30 --expert expectimax2 --samples 400000 --epochs 30 --batch-size 1024 --workers <physical-cores> --chunk-size 5000 --device cpu --save-full-dataset --checkpoint-every 5`.
  - To reuse the saved 200k dataset locally: run `python -m threes_rl.train_imitation --run-name <run> --expert expectimax2 --samples 200000 --epochs <epochs> --batch-size 1024 --workers 1 --dataset-path threes_rl/runs/imitation_expectimax2_200k_w8_e30/dataset.npz --device cpu --checkpoint-every 5`.
  - Evaluate with `python -m threes_rl.eval --policy ppo:<checkpoint> --seeds 1000:1200 --no-append`.
  - PPO can resume from any `latest.pt` with `python -m threes_rl.train_ppo --run-name <run> --resume <checkpoint> ...`, but the current evidence favors larger expectimax imitation before more PPO.

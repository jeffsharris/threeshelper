# Threes RL Progress

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
  - Best learned checkpoint so far: `threes_rl/runs/imitation_expectimax2_200k_w8_e30/checkpoint_epoch_20.pt`.
- Handoff / Mac Mini resume notes.
  - Current bottleneck for better learned policies is expectimax-d2 data generation. Serial generation measured roughly 82-88 expert samples/s; the parallel trainer reached roughly 470-517 samples/s here with 8 workers.
  - Extra RAM alone is not currently important. A beefier machine helps if it has more or faster CPU cores for expectimax label generation.
  - To scale the best path on another machine: run `python -m threes_rl.train_imitation --run-name imitation_expectimax2_400k_wN_e30 --expert expectimax2 --samples 400000 --epochs 30 --batch-size 1024 --workers <physical-cores> --chunk-size 5000 --device cpu --save-full-dataset --checkpoint-every 5`.
  - To reuse the saved 200k dataset locally: run `python -m threes_rl.train_imitation --run-name <run> --expert expectimax2 --samples 200000 --epochs <epochs> --batch-size 1024 --workers 1 --dataset-path threes_rl/runs/imitation_expectimax2_200k_w8_e30/dataset.npz --device cpu --checkpoint-every 5`.
  - Evaluate with `python -m threes_rl.eval --policy ppo:<checkpoint> --seeds 1000:1200 --no-append`.
  - PPO can resume from any `latest.pt` with `python -m threes_rl.train_ppo --run-name <run> --resume <checkpoint> ...`, but the current evidence favors larger expectimax imitation before more PPO.

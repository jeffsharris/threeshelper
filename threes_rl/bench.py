"""Throughput benchmarks for simulator, env, and expectimax."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from threes_rl.baselines import RandomPolicy
from threes_rl.env import ThreesEnv
from threes_rl.expectimax import ExpectimaxPolicy
from threes_rl.sim import ThreesSim


def bench_raw_sim(steps: int = 100_000) -> float:
    sim = ThreesSim(np.random.default_rng(1))
    policy = RandomPolicy()
    rng = np.random.default_rng(2)
    state = sim.reset()
    done_steps = 0
    start = time.perf_counter()
    while done_steps < steps:
        if state.game_over:
            state = sim.reset()
        action = policy(state, sim, rng)
        state, info = sim.step(state, action)
        if info.moved:
            done_steps += 1
    elapsed = time.perf_counter() - start
    return done_steps / elapsed


def bench_env(steps: int = 50_000) -> float:
    env = ThreesEnv(seed=3)
    rng = np.random.default_rng(4)
    _obs, info = env.reset(seed=3)
    done_steps = 0
    start = time.perf_counter()
    while done_steps < steps:
        legal = np.flatnonzero(info["legal_mask"])
        if len(legal) == 0:
            _obs, info = env.reset()
            continue
        action = int(legal[int(rng.integers(len(legal)))])
        _obs, _reward, terminated, _truncated, info = env.step(action)
        done_steps += 1
        if terminated:
            _obs, info = env.reset()
    elapsed = time.perf_counter() - start
    return done_steps / elapsed


def bench_expectimax(moves: int = 50) -> float:
    sim = ThreesSim(np.random.default_rng(5))
    policy = ExpectimaxPolicy(depth=2)
    rng = np.random.default_rng(6)
    state = sim.reset()
    done_moves = 0
    start = time.perf_counter()
    while done_moves < moves:
        if state.game_over:
            state = sim.reset()
        action = policy(state, sim, rng)
        state, info = sim.step(state, action)
        if info.moved:
            done_moves += 1
    elapsed = time.perf_counter() - start
    return done_moves / elapsed


def main() -> None:
    results = {
        "raw_sim_steps_per_s": bench_raw_sim(),
        "env_steps_per_s": bench_env(),
        "expectimax_d2_moves_per_s": bench_expectimax(),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    path = Path("threes_rl/RESULTS.md")
    with path.open("a") as fh:
        fh.write("\n## Bench\n\n")
        fh.write("Command: `python -m threes_rl.bench`\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(results, indent=2, sort_keys=True))
        fh.write("\n```\n")


if __name__ == "__main__":
    main()

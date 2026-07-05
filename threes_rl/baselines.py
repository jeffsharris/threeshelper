"""Baseline policies shared by evaluation and benchmarks."""

from __future__ import annotations

import numpy as np

from threes_rl.sim import SimState, ThreesSim, score_board, simulate_base_move


class RandomPolicy:
    name = "random"

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        legal = sim.legal_actions(state)
        if not legal:
            return 0
        return int(legal[int(rng.integers(len(legal)))])


class GreedyPolicy:
    name = "greedy"

    def __init__(self, empty_weight: float = 1.0) -> None:
        self.empty_weight = float(empty_weight)

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        legal = sim.legal_actions(state)
        if not legal:
            return 0
        before_score = score_board(state.board)
        best_score = None
        best_actions: list[int] = []
        for action in legal:
            shifted, _eligible = simulate_base_move(state.board, action)
            value = (score_board(shifted) - before_score) + self.empty_weight * int(np.count_nonzero(shifted == 0))
            if best_score is None or value > best_score:
                best_score = value
                best_actions = [action]
            elif value == best_score:
                best_actions.append(action)
        return int(best_actions[int(rng.integers(len(best_actions)))])

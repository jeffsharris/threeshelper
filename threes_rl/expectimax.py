"""Depth-limited expectimax over the exact simulator chance model."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from threes_rl.sim import SCORE_BY_VALUE, SimState, ThreesSim, rank_for_value, score_board, simulate_base_move

GRADIENTS = (
    (16, 15, 14, 13, 9, 10, 11, 12, 8, 7, 6, 5, 1, 2, 3, 4),
    (13, 14, 15, 16, 12, 11, 10, 9, 5, 6, 7, 8, 4, 3, 2, 1),
    (1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 11, 12, 16, 15, 14, 13),
    (4, 3, 2, 1, 5, 6, 7, 8, 12, 11, 10, 9, 13, 14, 15, 16),
)
CORNER_INDICES = (0, 3, 12, 15)


def _state_key(state: SimState, depth: int) -> tuple:
    counts = tuple(sorted((str(k), int(v)) for k, v in state.small_counts.items()))
    preview_key = (state.preview.kind, state.preview.value, state.preview.candidates)
    return (
        tuple(int(v) for v in state.board.reshape(-1)),
        preview_key,
        counts,
        int(state.small_pos),
        int(state.small_seen_total),
        int(state.span_small_pos),
        bool(state.large_pending),
        int(state.max_tile),
        int(depth),
    )


class ExpectimaxPolicy:
    def __init__(self, depth: int = 2, empty_weight: float = 2.0, monotonicity_weight: float = 0.25) -> None:
        self.depth = int(depth)
        self.empty_weight = float(empty_weight)
        self.monotonicity_weight = float(monotonicity_weight)
        self.name = f"expectimax{self.depth}"
        self._cache: dict[tuple, float] = {}
        self._action_cache: dict[tuple, float] = {}
        self._eval_cache: dict[tuple[int, ...], float] = {}

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        legal = sim.legal_actions(state)
        if not legal:
            return 0
        self._cache.clear()
        self._action_cache.clear()
        self._eval_cache.clear()
        best_value: Optional[float] = None
        best_actions: list[int] = []
        for action in legal:
            value = self._action_value(state, sim, action, self.depth)
            if best_value is None or value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)
        return int(best_actions[int(rng.integers(len(best_actions)))])

    def _action_value(self, state: SimState, sim: ThreesSim, action: int, depth: int) -> float:
        cache_key = (_state_key(state, depth), int(action))
        cached = self._action_cache.get(cache_key)
        if cached is not None:
            return cached
        if depth <= 1:
            value = self._leaf_action_value(state, sim, action)
            self._action_cache[cache_key] = value
            return value
        outcomes = sim.transition_outcomes(state, action, include_info=False)
        if not outcomes:
            return -1e18
        total = 0.0
        for probability, next_state, _info in outcomes:
            total += probability * self._value(next_state, sim, depth - 1)
        self._action_cache[cache_key] = total
        return total

    def _leaf_action_value(self, state: SimState, sim: ThreesSim, action: int) -> float:
        shifted, eligible_positions = simulate_base_move(state.board, action)
        if not eligible_positions:
            return -1e18
        if bool(np.any(shifted == 12288)):
            return self.evaluate_board(shifted)
        insert_options = sim._insert_value_options(state.preview)
        total = 0.0
        probability = 1.0 / (len(eligible_positions) * len(insert_options))
        for pos in eligible_positions:
            for inserted_value, value_prob in insert_options:
                board_after = shifted.copy()
                board_after[pos] = inserted_value
                total += (probability * len(insert_options) * value_prob) * self.evaluate_board(board_after)
        return total

    def _value(self, state: SimState, sim: ThreesSim, depth: int) -> float:
        if depth <= 0 or state.game_over:
            return self.evaluate(state)
        key = _state_key(state, depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        legal = sim.legal_actions(state)
        if not legal:
            value = self.evaluate(state)
        else:
            value = max(self._action_value(state, sim, action, depth) for action in legal)
        self._cache[key] = value
        return value

    def evaluate(self, state: SimState) -> float:
        return self.evaluate_board(state.board)

    def evaluate_board(self, board: np.ndarray) -> float:
        key = tuple(int(v) for v in board.reshape(-1))
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached
        value = float(sum(SCORE_BY_VALUE.get(v, score_board([[v]])) for v in key))
        value += self.empty_weight * sum(1 for value in key if value == 0)
        value += self.monotonicity_weight * _monotonicity_bonus_key(key)
        self._eval_cache[key] = value
        return value


@lru_cache(maxsize=4096)
def _rank_cached(value: int) -> int:
    return rank_for_value(value)


def _monotonicity_bonus_key(values: tuple[int, ...]) -> float:
    ranks = tuple(_rank_cached(value) for value in values)
    gradient_score = max(sum(rank * weight for rank, weight in zip(ranks, gradient)) for gradient in GRADIENTS)
    max_tile = max(values)
    corner_bonus = 0.0
    if max_tile > 0 and any(values[idx] == max_tile for idx in CORNER_INDICES):
        corner_bonus = 25.0
    return float(gradient_score + corner_bonus)

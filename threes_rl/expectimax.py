"""Depth-limited expectimax over the exact simulator chance model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from threes_rl.ntuple import (
    PHASE4_NAMES,
    NtupleValue,
    corner_risk_bucket_for_board,
    max_tile_excluding_free_starter,
    phase4_index_for_board,
)
from threes_rl.sim import (
    DOWN,
    LEFT,
    RIGHT,
    SCORE_BY_VALUE,
    TERMINAL_TILE,
    UP,
    Preview,
    SimState,
    StepInfo,
    ThreesSim,
    board_max_tile,
    rank_for_value,
    score_board,
    simulate_base_move,
)

GRADIENTS = (
    (16, 15, 14, 13, 9, 10, 11, 12, 8, 7, 6, 5, 1, 2, 3, 4),
    (13, 14, 15, 16, 12, 11, 10, 9, 5, 6, 7, 8, 4, 3, 2, 1),
    (1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 11, 12, 16, 15, 14, 13),
    (4, 3, 2, 1, 5, 6, 7, 8, 12, 11, 10, 9, 13, 14, 15, 16),
)
CORNER_INDICES = (0, 3, 12, 15)
TOP_LEFT_SNAKES = (
    (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12),
    (0, 4, 8, 12, 13, 9, 5, 1, 2, 6, 10, 14, 15, 11, 7, 3),
)
ORTHOGONAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gate_label(gate: int | str) -> str:
    if isinstance(gate, int):
        return PHASE4_NAMES[int(gate)]
    return str(gate).replace("=", "_").replace("/", "_")


def _gate_active(board: np.ndarray, gate: int | str) -> bool:
    if isinstance(gate, int):
        return phase4_index_for_board(board, starter_tile=1536) >= int(gate)
    if gate == "all":
        return True
    if isinstance(gate, str) and gate.startswith("risk="):
        return corner_risk_bucket_for_board(board, starter_tile=1536) == gate.split("=", 1)[1]
    if isinstance(gate, str) and gate.startswith("stratum="):
        phase, risk = gate.split("=", 1)[1].split("/", 1)
        return (
            PHASE4_NAMES[phase4_index_for_board(board, starter_tile=1536)] == phase
            and corner_risk_bucket_for_board(board, starter_tile=1536) == risk
        )
    raise ValueError(f"Unsupported blend gate: {gate}")


def _validate_gate(gate: int | str) -> None:
    if isinstance(gate, int) and 0 <= gate < len(PHASE4_NAMES):
        return
    if gate == "all":
        return
    if isinstance(gate, str) and (gate.startswith("risk=") or gate.startswith("stratum=")):
        return
    raise ValueError(f"Unsupported phase-gated blend gate: {gate}")


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
    def __init__(
        self,
        depth: int = 2,
        empty_weight: float = 2.0,
        monotonicity_weight: float = 0.25,
        corner_weight: float = 0.0,
        name: str | None = None,
    ) -> None:
        self.depth = int(depth)
        self.empty_weight = float(empty_weight)
        self.monotonicity_weight = float(monotonicity_weight)
        self.corner_weight = float(corner_weight)
        self.name = name or f"expectimax{self.depth}"
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
        value = 0.0
        for tile_value in key:
            cached_score = SCORE_BY_VALUE.get(tile_value)
            value += float(cached_score if cached_score is not None else score_board([[tile_value]]))
        value += self.empty_weight * sum(1 for value in key if value == 0)
        value += self.monotonicity_weight * _monotonicity_bonus_key(key)
        if self.corner_weight:
            value += self.corner_weight * _top_left_strategy_bonus_key(key)
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


def _top_left_strategy_bonus_key(values: tuple[int, ...]) -> float:
    ranks = tuple(_rank_cached(value) for value in values)
    max_rank = max(ranks)
    anchor_rank = ranks[0]
    empty_count = sum(1 for value in values if value == 0)

    bonus = 0.0
    if anchor_rank == max_rank:
        bonus += 500.0 * max_rank
    else:
        bonus -= 2000.0 * (max_rank - anchor_rank)

    # Reward a descending shape away from the top-left anchor. This is a soft
    # strategy prior, not a legality rule.
    monotone_penalty = 0
    for r in range(4):
        row = ranks[r * 4 : r * 4 + 4]
        for left, right in zip(row, row[1:]):
            if right > left:
                monotone_penalty += right - left
    for c in range(4):
        col = [ranks[r * 4 + c] for r in range(4)]
        for top, bottom in zip(col, col[1:]):
            if bottom > top:
                monotone_penalty += bottom - top
    bonus -= 120.0 * monotone_penalty

    merge_potential = 0
    smoothness_penalty = 0
    for r in range(4):
        for c in range(4):
            idx = r * 4 + c
            rank = ranks[idx]
            if rank == 0:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr >= 4 or nc >= 4:
                    continue
                other = ranks[nr * 4 + nc]
                if other == 0:
                    continue
                if other == rank:
                    merge_potential += rank
                else:
                    smoothness_penalty += abs(rank - other)
    bonus += 60.0 * merge_potential
    bonus -= 12.0 * smoothness_penalty
    bonus += 150.0 * empty_count
    return float(bonus)


def _board_without_free_starter(board: np.ndarray, starter_tile: int | None) -> np.ndarray:
    working = np.asarray(board, dtype=np.int32).copy()
    if starter_tile is None:
        return working
    matches = np.argwhere(working == int(starter_tile))
    if len(matches) == 0:
        return working
    match_idx = 0
    for idx, (row, col) in enumerate(matches):
        if int(row) == 0 and int(col) == 0:
            match_idx = idx
            break
    row, col = matches[match_idx]
    working[int(row), int(col)] = 0
    return working


def _snake_inversions_key(values: tuple[int, ...], path: tuple[int, ...]) -> int:
    ranks = tuple(_rank_cached(value) for value in values)
    inversions = 0
    previous = ranks[path[0]]
    for idx in path[1:]:
        current = ranks[idx]
        if current > previous:
            inversions += current - previous
        previous = current
    return int(inversions)


def _has_adjacent_value(board: np.ndarray, positions: list[tuple[int, int]], target: int) -> bool:
    if target <= 0:
        return False
    arr = np.asarray(board, dtype=np.int32)
    for row, col in positions:
        for dr, dc in ORTHOGONAL_OFFSETS:
            nr = int(row) + dr
            nc = int(col) + dc
            if 0 <= nr < 4 and 0 <= nc < 4 and int(arr[nr, nc]) == int(target):
                return True
    return False


def high_tile_geometry_score(
    board: np.ndarray,
    *,
    starter_tile: int | None = 1536,
    min_tile: int = 1536,
) -> float:
    """Small shape score for merge-ready high-tile geometry.

    The score is deliberately unitless and modest; callers decide its scale via
    a weight. It rewards keeping the built max near the top-left snake path and
    near same/half-max support, while penalizing stranded large tiles.
    """

    masked = _board_without_free_starter(board, starter_tile)
    max_tile = max_tile_excluding_free_starter(masked, None)
    if max_tile < int(min_tile):
        return 0.0
    positions = [tuple(int(v) for v in pos) for pos in np.argwhere(masked == int(max_tile))]
    if not positions:
        return 0.0
    max_rank = _rank_cached(int(max_tile))
    best_distance = min(row + col for row, col in positions)
    values_key = tuple(int(v) for v in masked.reshape(-1))
    snake_inversions = min(_snake_inversions_key(values_key, path) for path in TOP_LEFT_SNAKES)
    empty_count = int(np.count_nonzero(masked == 0))
    has_same = _has_adjacent_value(masked, positions, int(max_tile))
    has_half = _has_adjacent_value(masked, positions, int(max_tile // 2))

    score = 0.0
    score += float(max_rank) * (8.0 - 4.0 * float(best_distance))
    if (0, 0) in positions:
        score += 10.0 * float(max_rank)
    if has_same:
        score += 8.0 * float(max_rank)
    if has_half:
        score += 4.0 * float(max_rank)
    if empty_count <= 1 and not (has_same or has_half):
        score -= 4.0 * float(max_rank)
    score -= 2.0 * float(snake_inversions)
    return float(score)


class CornerExpectimaxPolicy(ExpectimaxPolicy):
    def __init__(self, depth: int = 2) -> None:
        super().__init__(
            depth=depth,
            empty_weight=350.0,
            monotonicity_weight=2.0,
            corner_weight=1.0,
            name=f"corner{depth}",
        )


class NtupleExpectimaxPolicy:
    """Expectimax search whose leaf value is a TD n-tuple afterstate table.

    The hand-built expectimax evaluator scores absolute board states. The
    n-tuple table is different: it estimates future score deltas from an
    afterstate. This policy therefore adds actual simulator score deltas at
    every chance branch and only asks the table for remaining future value.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        depth: int = 2,
        adaptive: bool = False,
        max_depth: int | None = None,
        empty_trigger: int = 2,
        chance_limit: int | None = None,
        suffix: str | None = None,
        blend_checkpoint: Path | None = None,
        blend_weight: float = 0.0,
        blend_specs: list[tuple[Path, float]] | None = None,
        phase_blend_specs: list[tuple[Path, float, int | str]] | None = None,
        bonus_specs: list[tuple[Path, float, int | str]] | None = None,
        tie_margin: float = 0.0,
        tie_breaker: str | None = None,
        ensemble_mode: str = "blend",
        geometry_weight: float = 0.0,
        geometry_min_tile: int = 1536,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.value_model = NtupleValue.load(self.checkpoint, mmap_mode="r")
        self.ensemble_mode = ensemble_mode
        if self.ensemble_mode not in ("blend", "additive", "max"):
            raise ValueError(f"Unsupported ensemble_mode: {self.ensemble_mode}")
        if blend_specs is not None and blend_checkpoint is not None:
            raise ValueError("Use either blend_specs or blend_checkpoint, not both")
        if blend_specs is not None:
            raw_blend_specs = [(Path(path), float(weight)) for path, weight in blend_specs]
        elif blend_checkpoint is not None:
            raw_blend_specs = [(Path(blend_checkpoint), float(blend_weight))]
        elif blend_weight != 0.0:
            raise ValueError("blend_checkpoint is required when blend_weight is nonzero")
        else:
            raw_blend_specs = []
        raw_phase_blend_specs = (
            [
                (Path(path), float(weight), int(gate) if isinstance(gate, int) else str(gate))
                for path, weight, gate in phase_blend_specs
            ]
            if phase_blend_specs is not None
            else []
        )
        raw_bonus_specs = (
            [
                (Path(path), float(weight), int(gate) if isinstance(gate, int) else str(gate))
                for path, weight, gate in bonus_specs
            ]
            if bonus_specs is not None
            else []
        )
        if raw_phase_blend_specs and self.ensemble_mode == "max":
            raise ValueError("phase-gated blend specs are only supported for blend/additive ensembles")
        total_blend_weight = sum(weight for _path, weight in raw_blend_specs)
        total_phase_blend_weight = sum(weight for _path, weight, _gate in raw_phase_blend_specs)
        for _path, weight in raw_blend_specs:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"blend weights must be between 0 and 1, got {weight}")
        for _path, weight, gate in raw_phase_blend_specs:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"phase-gated blend weights must be between 0 and 1, got {weight}")
            _validate_gate(gate)
        for _path, _weight, gate in raw_bonus_specs:
            _validate_gate(gate)
        if self.ensemble_mode == "blend" and total_blend_weight + total_phase_blend_weight > 1.0:
            raise ValueError(f"blend weights must sum to at most 1, got {total_blend_weight + total_phase_blend_weight}")
        self.base_weight = 1.0 - total_blend_weight if self.ensemble_mode == "blend" else 1.0
        self.blend_specs = tuple(raw_blend_specs)
        self.blend_models = tuple((path, NtupleValue.load(path, mmap_mode="r"), weight) for path, weight in self.blend_specs)
        self.phase_blend_specs = tuple(raw_phase_blend_specs)
        self.phase_blend_models = tuple(
            (path, NtupleValue.load(path, mmap_mode="r"), weight, min_phase)
            for path, weight, min_phase in self.phase_blend_specs
        )
        self.bonus_specs = tuple(raw_bonus_specs)
        self.bonus_models = tuple(
            (path, NtupleValue.load(path, mmap_mode="r"), weight, gate)
            for path, weight, gate in self.bonus_specs
        )
        self.blend_checkpoint = self.blend_specs[0][0] if self.blend_specs else None
        self.blend_weight = self.blend_specs[0][1] if self.blend_specs else 0.0
        self.blend_model = self.blend_models[0][1] if self.blend_models else None
        self.depth = int(depth)
        self.adaptive = bool(adaptive)
        self.max_depth = int(max_depth) if max_depth is not None else self.depth + (1 if self.adaptive else 0)
        self.empty_trigger = int(empty_trigger)
        self.chance_limit = None if chance_limit is None else max(1, int(chance_limit))
        self.tie_margin = float(tie_margin)
        self.tie_breaker = tie_breaker
        self.geometry_weight = float(geometry_weight)
        self.geometry_min_tile = int(geometry_min_tile)
        if self.tie_margin < 0.0:
            raise ValueError(f"tie_margin must be nonnegative, got {self.tie_margin}")
        if self.tie_breaker not in (None, "up_left"):
            raise ValueError(f"Unsupported tie_breaker: {self.tie_breaker}")
        if self.geometry_min_tile < 1:
            raise ValueError(f"geometry_min_tile must be positive, got {self.geometry_min_tile}")
        name_suffix = suffix if suffix is not None else ("a" if self.adaptive else "")
        if self.ensemble_mode == "max":
            blend_suffix = "".join(f":max:{path}" for path, _model, _weight in self.blend_models)
        elif self.ensemble_mode == "additive":
            blend_suffix = "".join(f":add:{path}:{weight:g}" for path, _model, weight in self.blend_models)
            blend_suffix += "".join(
                f":phaseadd:{path}:{weight:g}:{_gate_label(gate)}"
                for path, _model, weight, gate in self.phase_blend_models
            )
        else:
            blend_suffix = "".join(f":blend:{path}:{weight:g}" for path, _model, weight in self.blend_models)
            blend_suffix += "".join(
                f":phaseblend:{path}:{weight:g}:{_gate_label(gate)}"
                for path, _model, weight, gate in self.phase_blend_models
            )
        bonus_suffix = "".join(
            f":bonus:{path}:{weight:g}:{_gate_label(gate)}"
            for path, _model, weight, gate in self.bonus_models
        )
        tie_suffix = "" if self.tie_breaker is None else f":tie:{self.tie_breaker}:{self.tie_margin:g}"
        geometry_suffix = (
            ""
            if self.geometry_weight == 0.0
            else f":geometry:{self.geometry_weight:g}:{self.geometry_min_tile}"
        )
        self.name = f"ntuple_expectimax{self.depth}{name_suffix}:{self.checkpoint}{blend_suffix}{bonus_suffix}{tie_suffix}{geometry_suffix}"
        self._cache: dict[tuple, float] = {}
        self._action_cache: dict[tuple, float] = {}
        self._afterstate_cache: dict[tuple[int, ...], float] = {}
        self._post_spawn_cache: dict[tuple, float] = {}
        self._score_cache: dict[tuple[int, ...], int] = {}
        self._legal_cache: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._base_move_cache: dict[tuple[tuple[int, ...], int], tuple[np.ndarray, tuple[tuple[int, int], ...]]] = {}

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        action_values = self.action_values(state, sim)
        if not action_values:
            return 0
        return self._select_action(action_values, rng)

    def action_values(self, state: SimState, sim: ThreesSim) -> list[tuple[int, float]]:
        self._cache.clear()
        self._action_cache.clear()
        self._afterstate_cache.clear()
        self._post_spawn_cache.clear()
        self._score_cache.clear()
        self._legal_cache.clear()
        self._base_move_cache.clear()
        legal = self._legal_actions(state, sim)
        if not legal:
            return []
        depth = self._root_depth(state)
        action_values: list[tuple[int, float]] = []
        for action in legal:
            value = self._action_value(state, sim, action, depth)
            action_values.append((int(action), float(value)))
        return action_values

    def _select_action(self, action_values: list[tuple[int, float]], rng: np.random.Generator) -> int:
        best_value = max(value for _action, value in action_values)
        if self.tie_breaker is None:
            best_actions = [action for action, value in action_values if value == best_value]
            return int(best_actions[int(rng.integers(len(best_actions)))])

        candidates = [
            (action, value)
            for action, value in action_values
            if value >= best_value - self.tie_margin
        ]
        if self.tie_breaker == "up_left":
            priority = {UP: 0, LEFT: 1, DOWN: 2, RIGHT: 3}
            best_priority = min(priority.get(action, 99) for action, _value in candidates)
            best_actions = [action for action, _value in candidates if priority.get(action, 99) == best_priority]
            return int(best_actions[int(rng.integers(len(best_actions)))])
        raise AssertionError(f"Unhandled tie_breaker: {self.tie_breaker}")

    def _root_depth(self, state: SimState) -> int:
        depth = self.depth
        empty_count = int(np.count_nonzero(state.board == 0))
        bonus_tactical = state.preview.kind == "bonus" and empty_count <= max(4, self.empty_trigger)
        cramped_tactical = empty_count <= self.empty_trigger
        if self.adaptive and (bonus_tactical or cramped_tactical):
            depth += 1
        return min(depth, self.max_depth)

    def _afterstate_value(self, board: np.ndarray) -> float:
        key = tuple(int(v) for v in np.asarray(board, dtype=np.int32).reshape(-1))
        cached = self._afterstate_cache.get(key)
        if cached is not None:
            return cached
        base_value = float(self.value_model.value(board))
        if self.ensemble_mode == "max":
            value = max([base_value, *(float(model.value(board)) for _path, model, _weight in self.blend_models)])
        elif self.ensemble_mode == "additive":
            value = base_value
            for _path, model, weight in self.blend_models:
                if weight > 0.0:
                    value += weight * float(model.value(board))
            for _model, weight in [
                (model, weight)
                for _path, model, weight, gate in self.phase_blend_models
                if _gate_active(board, gate)
            ]:
                if weight > 0.0:
                    value += weight * float(_model.value(board))
        else:
            active_phase_models = [
                (model, weight)
                for _path, model, weight, gate in self.phase_blend_models
                if _gate_active(board, gate)
            ]
            active_phase_weight = sum(weight for _model, weight in active_phase_models)
            value = (self.base_weight - active_phase_weight) * base_value
            for _path, model, weight in self.blend_models:
                if weight > 0.0:
                    value += weight * float(model.value(board))
            for model, weight in active_phase_models:
                if weight > 0.0:
                    value += weight * float(model.value(board))
        if self.geometry_weight != 0.0:
            value += self.geometry_weight * high_tile_geometry_score(
                board,
                starter_tile=1536,
                min_tile=self.geometry_min_tile,
            )
        for _path, model, weight, gate in self.bonus_models:
            if weight != 0.0 and _gate_active(board, gate):
                value += float(weight) * float(model.value(board))
        self._afterstate_cache[key] = value
        return value

    def _score_board(self, board: np.ndarray) -> int:
        key = tuple(int(v) for v in np.asarray(board, dtype=np.int32).reshape(-1))
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        score = score_board(board)
        self._score_cache[key] = score
        return score

    def _base_move(self, board: np.ndarray, action: int) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        board_key = tuple(int(v) for v in np.asarray(board, dtype=np.int32).reshape(-1))
        key = (board_key, int(action))
        cached = self._base_move_cache.get(key)
        if cached is not None:
            return cached
        shifted, eligible = simulate_base_move(board, int(action))
        value = (shifted, tuple((int(row), int(col)) for row, col in eligible))
        self._base_move_cache[key] = value
        return value

    def _legal_actions(self, state: SimState, sim: ThreesSim) -> tuple[int, ...]:
        if state.game_over:
            return ()
        key = tuple(int(v) for v in np.asarray(state.board, dtype=np.int32).reshape(-1))
        cached = self._legal_cache.get(key)
        if cached is not None:
            return cached
        if int(np.max(state.board, initial=0)) >= TERMINAL_TILE:
            legal = ()
        else:
            legal = tuple(
                action
                for action in range(4)
                if self._base_move(state.board, action)[1]
            )
        self._legal_cache[key] = legal
        return legal

    def _post_spawn_state_value(self, state: SimState, sim: ThreesSim) -> float:
        if state.game_over:
            return 0.0
        # This leaf helper only uses the board: legal moves, merge deltas, and
        # the afterstate value are all preview/cycle independent.
        key = tuple(int(v) for v in np.asarray(state.board, dtype=np.int32).reshape(-1))
        cached = self._post_spawn_cache.get(key)
        if cached is not None:
            return cached
        legal = self._legal_actions(state, sim)
        if not legal:
            self._post_spawn_cache[key] = 0.0
            return 0.0
        before_score = self._score_board(state.board)
        best: float | None = None
        for action in legal:
            shifted, eligible = self._base_move(state.board, action)
            if not eligible:
                continue
            merge_delta = self._score_board(shifted) - before_score
            value = float(merge_delta + self._afterstate_value(shifted))
            if best is None or value > best:
                best = value
        value = 0.0 if best is None else float(best)
        self._post_spawn_cache[key] = value
        return value

    def _transition_outcomes(
        self,
        state: SimState,
        sim: ThreesSim,
        action: int,
        *,
        include_next_preview: bool,
    ) -> list[tuple[float, SimState, StepInfo]]:
        if state.game_over:
            return []
        before_score = self._score_board(state.board)
        shifted, eligible_positions = self._base_move(state.board, int(action))
        if not eligible_positions:
            return []

        merge_delta = self._score_board(shifted) - before_score
        if bool(np.any(shifted == TERMINAL_TILE)):
            next_state = SimState(
                board=shifted,
                preview=state.preview,
                small_counts=state.small_counts.copy(),
                small_pos=int(state.small_pos),
                small_seen_total=int(state.small_seen_total),
                span_small_pos=int(state.span_small_pos),
                large_pending=bool(state.large_pending),
                max_tile=TERMINAL_TILE,
                move_count=int(state.move_count) + 1,
                game_over=True,
            )
            info = StepInfo(
                True,
                None,
                None,
                int(merge_delta),
                int(self._score_board(shifted) - before_score),
                list(eligible_positions),
                True,
            )
            return [(1.0, next_state, info)]

        insert_options = sim._insert_value_options(state.preview)
        slot_prob = 1.0 / len(eligible_positions)
        consumed = sim._consume_preview(
            state.small_counts,
            state.small_pos,
            state.small_seen_total,
            state.span_small_pos,
            state.large_pending,
            state.preview.label,
        )
        outcomes: list[tuple[float, SimState, StepInfo]] = []
        for pos in eligible_positions:
            for inserted_value, value_prob in insert_options:
                board_after = shifted.copy()
                board_after[pos] = int(inserted_value)
                max_tile = board_max_tile(board_after)
                score_delta = self._score_board(board_after) - before_score
                if include_next_preview:
                    preview_options = sim.preview_options(
                        consumed[0],
                        consumed[1],
                        consumed[2],
                        consumed[3],
                        consumed[4],
                        max_tile,
                    )
                else:
                    preview_options = [(Preview("ignored", None), 1.0)]
                for preview_option in preview_options:
                    if hasattr(preview_option, "preview"):
                        preview = preview_option.preview
                        probability = float(preview_option.probability)
                    else:
                        preview, probability = preview_option
                    next_state = SimState(
                        board=board_after.copy(),
                        preview=preview,
                        small_counts=consumed[0].copy(),
                        small_pos=int(consumed[1]),
                        small_seen_total=int(consumed[2]),
                        span_small_pos=int(consumed[3]),
                        large_pending=bool(consumed[4]),
                        max_tile=int(max_tile),
                        move_count=int(state.move_count) + 1,
                        game_over=False,
                    )
                    info = StepInfo(
                        True,
                        int(inserted_value),
                        (int(pos[0]), int(pos[1])),
                        int(merge_delta),
                        int(score_delta),
                        list(eligible_positions),
                        False,
                    )
                    outcomes.append((slot_prob * float(value_prob) * probability, next_state, info))
        return outcomes

    def _action_value(self, state: SimState, sim: ThreesSim, action: int, depth: int) -> float:
        cache_key = (_state_key(state, depth), int(action))
        cached = self._action_cache.get(cache_key)
        if cached is not None:
            return cached
        outcomes = self._transition_outcomes(state, sim, action, include_next_preview=depth > 1)
        if not outcomes:
            return -1e18
        outcomes = self._budgeted_outcomes(outcomes, sim, depth)
        total = 0.0
        for probability, next_state, info in outcomes:
            if depth <= 1:
                future = self._post_spawn_state_value(next_state, sim)
            else:
                future = self._value(next_state, sim, depth - 1)
            total += float(probability) * (float(info.score_delta) + future)
        self._action_cache[cache_key] = total
        return total

    def _budgeted_outcomes(self, outcomes: list[tuple[float, SimState, object]], sim: ThreesSim, depth: int) -> list[tuple[float, SimState, object]]:
        if self.chance_limit is None or depth <= 1 or len(outcomes) <= self.chance_limit:
            return outcomes

        scored: list[tuple[float, int, float, SimState, object]] = []
        for idx, (probability, next_state, info) in enumerate(outcomes):
            quick_value = float(info.score_delta) + self._post_spawn_state_value(next_state, sim)
            scored.append((quick_value, idx, float(probability), next_state, info))
        scored.sort(key=lambda item: (item[0], item[1]))

        bucket_count = min(self.chance_limit, len(scored))
        chosen: list[tuple[float, int, float, SimState, object]] = []
        for bucket in range(bucket_count):
            # Pick representative quantiles from bad-to-good quick outcomes.
            idx = round(bucket * (len(scored) - 1) / max(1, bucket_count - 1))
            chosen.append(scored[int(idx)])

        probability = 1.0 / float(len(chosen))
        return [(probability, next_state, info) for _quick, _idx, _old_probability, next_state, info in chosen]

    def _value(self, state: SimState, sim: ThreesSim, depth: int) -> float:
        if state.game_over:
            return 0.0
        legal = self._legal_actions(state, sim)
        if not legal:
            return 0.0
        if depth <= 0:
            return self._post_spawn_state_value(state, sim)
        key = _state_key(state, depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = max(self._action_value(state, sim, action, depth) for action in legal)
        self._cache[key] = value
        return value

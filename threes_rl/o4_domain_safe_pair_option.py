"""Domain-safe O4 designated-pair geometry, features, and model schema."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from threes_rl.o1_geometry_option import (
    air_safe,
    anchor_safe,
    pair_safe_merge_actions,
)
from threes_rl.o3_designated_pair_option import (
    DecisionTargets,
    LineageMove,
    advance_lineage_base,
    apply_spawn_to_lineage,
    balanced_valid_row_weight,
    build_decision_targets as _build_decision_targets,
    canonical_json_hash,
    initial_lineage,
    lineage_integrity,
    transition_status,
)
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim


VERSION = "o4_domain_safe_designated_pair_v1"
TRAIN_TARGETS = (48, 96, 192)
INTEGRATED_TARGETS = (48, 96, 192, 384, 768)
MIN_SAFE_EMPTIES = 2
MIN_SAFE_PRESPAWN_EMPTIES = 3
OPTION_HORIZON = 40
CHECKPOINTS = (10, 20, 40)
LINEAGE_A = 1
LINEAGE_B = 2
LINEAGE_MERGED = LINEAGE_A | LINEAGE_B
RANK_CATEGORY_NAMES = (
    "empty",
    "small1",
    "small2",
    "small3",
    "relative_le_minus4",
    "relative_minus3",
    "relative_minus2",
    "relative_minus1",
    "relative_zero",
    "relative_plus1",
    "relative_ge_plus2",
)
TOKEN_WIDTH = 37
GLOBAL_WIDTH = 35
EVENT_WIDTH = 5
GEOMETRY_WIDTH = 8
OUTPUT_WIDTH = EVENT_WIDTH + len(CHECKPOINTS) * GEOMETRY_WIDTH
EVENT_CLASS_NAMES = (
    "safe_merge_moves_1_10",
    "safe_merge_moves_11_20",
    "safe_merge_moves_21_40",
    "failure",
    "censor_at_40",
)
GEOMETRY_NAMES = (
    "manhattan",
    "chebyshev",
    "blocker_density",
    "same_row",
    "same_column",
    "empty_count",
    "legal_count",
    "two_descendant_lineage_intact",
)


@dataclass(frozen=True)
class PairBlockerGeometry:
    eligible_cells: tuple[tuple[int, int], ...]
    occupied: int
    capacity: int
    density: float


@dataclass(frozen=True)
class DesignatedPair:
    target: int
    coordinates: tuple[tuple[int, int], tuple[int, int]]
    manhattan: int
    chebyshev: int
    blocker_occupied: int
    blocker_capacity: int
    blocker_density: float
    same_row: bool
    same_column: bool
    clear_line: bool
    safe_merge_actions: tuple[int, ...]


def normalized_board(
    board: Sequence[Sequence[int]] | np.ndarray,
    starter_tile: int | None,
) -> np.ndarray:
    result = np.asarray(board, dtype=np.int32).copy()
    if result.shape != (4, 4):
        raise ValueError(f"Expected 4x4 board, got {result.shape}")
    if starter_tile is not None and int(result[0, 0]) == int(starter_tile):
        result[0, 0] = 0
    return result


def _canonical_pair(
    pair: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    coordinates = tuple(
        sorted((int(value[0]), int(value[1])) for value in pair)
    )
    if len(coordinates) != 2 or coordinates[0] == coordinates[1]:
        raise ValueError("A pair requires two distinct coordinates")
    if any(not 0 <= value < 4 for coordinate in coordinates for value in coordinate):
        raise ValueError("Pair coordinates must lie on the 4x4 board")
    return coordinates  # type: ignore[return-value]


def eligible_blocker_cells(
    pair: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    (r0, c0), (r1, c1) = _canonical_pair(pair)
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return tuple((r0, column) for column in range(lo + 1, hi))
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return tuple((row, c0) for row in range(lo + 1, hi))
    rlo, rhi = sorted((r0, r1))
    clo, chi = sorted((c0, c1))
    endpoints = {(r0, c0), (r1, c1)}
    return tuple(
        (row, column)
        for row in range(rlo, rhi + 1)
        for column in range(clo, chi + 1)
        if (row, column) not in endpoints
    )


def blocker_geometry(
    board: Sequence[Sequence[int]] | np.ndarray,
    pair: Sequence[Sequence[int]],
) -> PairBlockerGeometry:
    arr = np.asarray(board, dtype=np.int32)
    if arr.shape != (4, 4):
        raise ValueError("Blocker geometry requires a 4x4 board")
    cells = eligible_blocker_cells(pair)
    capacity = len(cells)
    occupied = sum(int(arr[cell] != 0) for cell in cells)
    density = 0.0 if capacity == 0 else occupied / float(capacity)
    if not (
        0 <= occupied <= capacity
        and math.isfinite(density)
        and 0.0 <= density <= 1.0
    ):
        raise RuntimeError("O4 blocker density escaped its exact domain")
    return PairBlockerGeometry(
        eligible_cells=cells,
        occupied=occupied,
        capacity=capacity,
        density=density,
    )


def _ordered_pairs(
    coordinates: Iterable[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ordered = sorted(coordinates)
    return [
        (ordered[left], ordered[right])
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    ]


def _pair_row(
    board: np.ndarray,
    starter_tile: int | None,
    target: int,
    coordinates: tuple[tuple[int, int], tuple[int, int]],
) -> DesignatedPair:
    (r0, c0), (r1, c1) = coordinates
    blockers = blocker_geometry(board, coordinates)
    same_row = r0 == r1
    same_column = c0 == c1
    safe_actions = pair_safe_merge_actions(
        board,
        coordinates,
        target,
        starter_tile,
    )
    return DesignatedPair(
        target=int(target),
        coordinates=coordinates,
        manhattan=abs(r0 - r1) + abs(c0 - c1),
        chebyshev=max(abs(r0 - r1), abs(c0 - c1)),
        blocker_occupied=blockers.occupied,
        blocker_capacity=blockers.capacity,
        blocker_density=blockers.density,
        same_row=same_row,
        same_column=same_column,
        clear_line=bool(
            (same_row or same_column) and blockers.occupied == 0
        ),
        safe_merge_actions=tuple(int(action) for action in safe_actions),
    )


def select_designated_pair(
    board: Sequence[Sequence[int]] | np.ndarray,
    starter_tile: int | None,
    *,
    requested_target: int | None = None,
    allowed_targets: Sequence[int] = TRAIN_TARGETS,
) -> DesignatedPair | None:
    arr = np.asarray(board, dtype=np.int32)
    working = normalized_board(arr, starter_tile)
    frozen_targets = tuple(int(value) for value in allowed_targets)
    if requested_target is None:
        available = [
            target
            for target in frozen_targets
            if int(np.count_nonzero(working == target)) >= 2
        ]
        if not available:
            return None
        target = max(available)
    else:
        target = int(requested_target)
        if (
            target not in frozen_targets
            or int(np.count_nonzero(working == target)) < 2
        ):
            return None
    coordinates = [
        (int(row), int(column))
        for row, column in np.argwhere(working == target)
    ]
    rows = [
        _pair_row(arr, starter_tile, target, pair)
        for pair in _ordered_pairs(coordinates)
    ]
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            0 if row.safe_merge_actions else 1,
            row.manhattan + row.blocker_density,
            row.blocker_density,
            row.manhattan,
            row.chebyshev,
            row.coordinates,
        ),
    )


def _rank_offset(value: int, target: int) -> int:
    if value <= 0:
        raise ValueError("Rank offset requires a positive tile")
    exponent = math.log2(float(value) / float(target))
    if abs(exponent - round(exponent)) > 1e-9:
        raise ValueError(f"Non-rank tile value {value} for target {target}")
    return int(round(exponent))


def rank_category(value: int, target: int) -> int:
    value = int(value)
    if value == 0:
        return 0
    if value in (1, 2, 3):
        return value
    offset = _rank_offset(value, int(target))
    if offset <= -4:
        return 4
    if offset == -3:
        return 5
    if offset == -2:
        return 6
    if offset == -1:
        return 7
    if offset == 0:
        return 8
    if offset == 1:
        return 9
    return 10


def _bounded_ratio(value: int | float, maximum: int | float, name: str) -> float:
    number = float(value)
    limit = float(maximum)
    if not math.isfinite(number) or not 0.0 <= number <= limit:
        raise ValueError(f"{name} is outside 0..{limit}")
    return number / limit


def _relative_unit(value: int, target: int) -> float:
    clipped = max(-4, min(4, _rank_offset(int(value), int(target))))
    return (float(clipped) + 4.0) / 8.0


def _preview_name(state: SimState) -> str:
    return "bonus" if state.preview.kind == "bonus" else str(state.preview.kind)


def successor_geometry(
    state: SimState,
    sim: ThreesSim,
    *,
    lineage: Sequence[Sequence[int]] | np.ndarray,
    target: int,
) -> np.ndarray:
    marks = np.asarray(lineage, dtype=np.uint8)
    if lineage_integrity(marks) != "live":
        raise ValueError("O4 successor geometry requires live lineage")
    a = np.argwhere((marks & LINEAGE_A) != 0)
    b = np.argwhere((marks & LINEAGE_B) != 0)
    if a.shape != (1, 2) or b.shape != (1, 2):
        raise ValueError("O4 successor geometry requires one A and one B")
    coordinates = _canonical_pair((a[0], b[0]))
    pair = _pair_row(state.board, None, int(target), coordinates)
    values = np.asarray(
        (
            pair.manhattan / 6.0,
            pair.chebyshev / 3.0,
            pair.blocker_density,
            float(pair.same_row),
            float(pair.same_column),
            np.count_nonzero(state.board == 0) / 16.0,
            len(sim.legal_actions(state)) / 4.0,
            1.0,
        ),
        dtype=np.float32,
    )
    if (
        values.shape != (GEOMETRY_WIDTH,)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("O4 successor geometry escaped [0,1]")
    return values


def option_features(
    state: SimState,
    sim: ThreesSim,
    *,
    starter_tile: int | None,
    pair: DesignatedPair,
    lineage: Sequence[Sequence[int]] | np.ndarray,
    action: int,
) -> tuple[np.ndarray, np.ndarray]:
    legal = tuple(int(value) for value in sim.legal_actions(state))
    if int(action) not in legal:
        raise ValueError("O4 features require a legal candidate action")
    board_before = state.board.copy()
    lineage_before = np.asarray(lineage, dtype=np.uint8).copy()
    cycle_before = sim.tile_cycle_snapshot(state)
    deck_rng_before = json.dumps(
        sim.deck_rng.bit_generator.state,
        sort_keys=True,
    )
    slot_rng_before = json.dumps(
        sim.slot_rng.bit_generator.state,
        sort_keys=True,
    )
    after = advance_lineage_base(state.board, lineage_before, int(action))

    tokens = np.zeros((16, TOKEN_WIDTH), dtype=np.float32)
    for row in range(4):
        for column in range(4):
            index = row * 4 + column
            before_category = rank_category(
                int(state.board[row, column]),
                pair.target,
            )
            after_category = rank_category(
                int(after.board[row, column]),
                pair.target,
            )
            tokens[index, before_category] = 1.0
            tokens[index, 11 + after_category] = 1.0
            mark_before = int(lineage_before[row, column])
            tokens[index, 22] = float(bool(mark_before & LINEAGE_A))
            tokens[index, 23] = float(bool(mark_before & LINEAGE_B))
            mark_after = int(after.lineage[row, column])
            tokens[index, 24] = float(mark_after == LINEAGE_A)
            tokens[index, 25] = float(mark_after == LINEAGE_B)
            tokens[index, 26] = float(mark_after == LINEAGE_MERGED)
            tokens[index, 27] = float(
                starter_tile is not None
                and (row, column) == (0, 0)
                and int(state.board[row, column]) == int(starter_tile)
            )
            tokens[index, 28] = float((row, column) in after.eligible_slots)
            tokens[index, 29 + row] = 1.0
            tokens[index, 33 + column] = 1.0

    preview_name = _preview_name(state)
    global_values: list[float] = [
        float(preview_name == name)
        for name in ("blue", "red", "gray", "bonus")
    ]
    global_values.extend(
        _bounded_ratio(
            state.small_counts.get(name, 0),
            4,
            f"small_counts[{name}]",
        )
        for name in ("red", "blue", "gray")
    )
    global_values.extend(
        (
            _bounded_ratio(state.small_pos, 12, "small_pos"),
            _bounded_ratio(
                min(int(state.small_seen_total), 256),
                256,
                "small_seen_total",
            ),
            _bounded_ratio(
                min(int(state.span_small_pos), 21),
                21,
                "span_small_pos",
            ),
            float(state.large_pending),
        )
    )
    candidates = [
        _relative_unit(int(value), pair.target)
        for value in state.preview.candidates
    ]
    if candidates:
        global_values.extend(
            (
                min(candidates),
                sum(candidates) / len(candidates),
                max(candidates),
            )
        )
    else:
        global_values.extend((0.5, 0.5, 0.5))
    global_values.extend(
        (
            np.count_nonzero(state.board == 0) / 16.0,
            np.count_nonzero(after.board == 0) / 16.0,
            len(legal) / 4.0,
        )
    )
    safe_until = sim.safe_smalls_until_large_possible(state)
    global_values.append(
        1.0 if safe_until is None else min(1.0, float(safe_until) / 21.0)
    )
    working = normalized_board(state.board, starter_tile)
    maximum_offset = max(
        (
            _rank_offset(int(value), pair.target)
            for value in working.reshape(-1)
            if int(value) > 3
        ),
        default=-4,
    )
    global_values.append(
        (float(max(-4, min(4, maximum_offset))) + 4.0) / 8.0
    )
    global_values.extend(
        (
            pair.manhattan / 6.0,
            pair.chebyshev / 3.0,
            pair.blocker_density,
            float(pair.same_row),
            float(pair.same_column),
            float(pair.clear_line),
            float(pair.manhattan == 1),
            float(pair.chebyshev == 1 and pair.manhattan == 2),
        )
    )
    global_values.extend(
        float(int(action) == candidate) for candidate in range(4)
    )
    global_values.extend(
        (
            float(after.event == "designated_pair_merged"),
            float(after.event == "third_party_merge"),
            float(anchor_safe(after.board, starter_tile)),
            float(
                int(np.count_nonzero(after.board == 0))
                >= MIN_SAFE_PRESPAWN_EMPTIES
            ),
        )
    )
    globals_array = np.asarray(global_values, dtype=np.float32)
    if tokens.shape != (16, TOKEN_WIDTH):
        raise RuntimeError(f"O4 token shape mismatch: {tokens.shape}")
    if globals_array.shape != (GLOBAL_WIDTH,):
        raise RuntimeError(f"O4 global shape mismatch: {globals_array.shape}")
    if (
        not np.isfinite(tokens).all()
        or not np.isfinite(globals_array).all()
        or np.any(tokens < 0.0)
        or np.any(tokens > 1.0)
        or np.any(globals_array < 0.0)
        or np.any(globals_array > 1.0)
    ):
        raise RuntimeError("O4 model inputs must be finite and in [0,1]")
    if not np.array_equal(state.board, board_before):
        raise RuntimeError("O4 features mutated the simulator board")
    if not np.array_equal(np.asarray(lineage), lineage_before):
        raise RuntimeError("O4 features mutated lineage")
    if sim.tile_cycle_snapshot(state) != cycle_before:
        raise RuntimeError("O4 features mutated tile-cycle state")
    if json.dumps(sim.deck_rng.bit_generator.state, sort_keys=True) != deck_rng_before:
        raise RuntimeError("O4 features consumed deck RNG")
    if json.dumps(sim.slot_rng.bit_generator.state, sort_keys=True) != slot_rng_before:
        raise RuntimeError("O4 features consumed slot RNG")
    return tokens, globals_array


def build_decision_targets(
    *,
    decision_move: int,
    terminal_move: int,
    terminal_status: str,
    live_geometry_by_move: dict[int, Sequence[float]],
) -> DecisionTargets:
    targets = _build_decision_targets(
        decision_move=decision_move,
        terminal_move=terminal_move,
        terminal_status=terminal_status,
        live_geometry_by_move=live_geometry_by_move,
    )
    if (
        not np.isfinite(targets.geometry).all()
        or np.any(targets.geometry < 0.0)
        or np.any(targets.geometry > 1.0)
    ):
        raise ValueError("O4 decision geometry escaped [0,1]")
    return targets


def root_option_eligible(
    state: SimState,
    sim: ThreesSim,
    starter_tile: int | None,
    *,
    allowed_targets: Sequence[int] = TRAIN_TARGETS,
) -> bool:
    if (
        state.game_over
        or not anchor_safe(state.board, starter_tile)
        or not air_safe(state.board)
        or len(sim.legal_actions(state)) < 2
    ):
        return False
    pair = select_designated_pair(
        state.board,
        starter_tile,
        allowed_targets=allowed_targets,
    )
    return pair is not None and not pair.safe_merge_actions


def action_order_components(
    output: Sequence[float] | np.ndarray,
    *,
    remaining_horizon: int,
) -> tuple[float, float, float]:
    values = np.asarray(output, dtype=np.float64)
    if values.shape != (OUTPUT_WIDTH,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {OUTPUT_WIDTH} finite action outputs")
    remaining = int(remaining_horizon)
    if not 1 <= remaining <= OPTION_HORIZON:
        raise ValueError("remaining_horizon must be in 1..40")
    shifted = values[:EVENT_WIDTH] - float(np.max(values[:EVENT_WIDTH]))
    probabilities = np.exp(shifted)
    probabilities /= float(np.sum(probabilities))
    included_bins = 1 + int(remaining >= 11) + int(remaining >= 21)
    safe_probability = float(np.sum(probabilities[:included_bins]))
    nonfailure_probability = float(1.0 - probabilities[3])
    h10 = 1.0 / (1.0 + np.exp(-values[EVENT_WIDTH : EVENT_WIDTH + 8]))
    successor_potential = (
        -0.45 * float(h10[0])
        - 0.20 * float(h10[1])
        - 0.20 * float(h10[2])
        + 0.05 * max(float(h10[3]), float(h10[4]))
        + 0.05 * float(h10[5])
        + 0.05 * float(h10[6])
    )
    return safe_probability, nonfailure_probability, successor_potential


def choose_option_action(
    outputs_by_action: dict[int, Sequence[float] | np.ndarray],
    *,
    remaining_horizon: int,
    safe_merge_actions: Sequence[int] = (),
    tie_tolerance: float = 1e-12,
) -> int:
    actions = tuple(sorted(int(action) for action in outputs_by_action))
    if not actions:
        raise ValueError("At least one legal action output is required")
    safe = tuple(sorted(int(action) for action in safe_merge_actions))
    if safe:
        if any(action not in outputs_by_action for action in safe):
            raise ValueError("Safe-merge action is not legal")
        return safe[0]
    tolerance = float(tie_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    components = {
        action: action_order_components(
            outputs_by_action[action],
            remaining_horizon=remaining_horizon,
        )
        for action in actions
    }
    best = actions[0]
    for action in actions[1:]:
        for candidate, incumbent in zip(
            components[action],
            components[best],
            strict=True,
        ):
            if candidate > incumbent + tolerance:
                best = action
                break
            if candidate < incumbent - tolerance:
                break
    return best


class O4DesignatedPairNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_input = nn.Linear(TOKEN_WIDTH, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.hidden = nn.Sequential(
            nn.Linear(64 * 3 + GLOBAL_WIDTH, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        self.output = nn.Linear(128, OUTPUT_WIDTH)

    def forward(
        self,
        tokens: torch.Tensor,
        global_values: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(torch.nn.functional.gelu(self.token_input(tokens)))
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        designated_mask = torch.clamp(
            tokens[:, :, 22] + tokens[:, :, 23],
            min=0.0,
            max=1.0,
        )
        denominator = designated_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        designated_pool = (
            encoded * designated_mask.unsqueeze(-1)
        ).sum(dim=1) / denominator
        combined = torch.cat(
            (mean_pool, max_pool, designated_pool, global_values),
            dim=1,
        )
        return self.output(self.hidden(combined))


def exhaustive_blocker_domain_proof() -> dict[str, Any]:
    coordinates = tuple(product(range(4), repeat=2))
    cases = 0
    capacity_counts: dict[int, int] = {}
    for pair in combinations(coordinates, 2):
        cells = eligible_blocker_cells(pair)
        capacity_counts[len(cells)] = capacity_counts.get(len(cells), 0) + 1
        for occupancy_mask in range(1 << len(cells)):
            board = np.zeros((4, 4), dtype=np.int32)
            board[pair[0]] = 48
            board[pair[1]] = 48
            for index, cell in enumerate(cells):
                if occupancy_mask & (1 << index):
                    board[cell] = 1
            geometry = blocker_geometry(board, pair)
            if geometry.occupied != occupancy_mask.bit_count():
                raise RuntimeError("O4 exhaustive occupied-count mismatch")
            if geometry.capacity != len(cells):
                raise RuntimeError("O4 exhaustive capacity mismatch")
            if not 0.0 <= geometry.density <= 1.0:
                raise RuntimeError("O4 exhaustive density domain failure")
            cases += 1
    if cases != 43_296:
        raise RuntimeError(f"O4 exhaustive case count changed: {cases}")
    return {
        "coordinate_pairs": 120,
        "occupancy_cases": cases,
        "capacity_pair_counts": {
            str(key): value for key, value in sorted(capacity_counts.items())
        },
        "minimum_density": 0.0,
        "maximum_density": 1.0,
        "passes": True,
    }


def schema_manifest() -> dict[str, Any]:
    return {
        "version": VERSION,
        "train_targets": list(TRAIN_TARGETS),
        "integrated_targets": list(INTEGRATED_TARGETS),
        "token_width": TOKEN_WIDTH,
        "global_width": GLOBAL_WIDTH,
        "event_width": EVENT_WIDTH,
        "geometry_width": GEOMETRY_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "geometry_names": list(GEOMETRY_NAMES),
        "blocker_eligible_cells": {
            "same_row": "strictly_between_columns",
            "same_column": "strictly_between_rows",
            "unaligned": "inclusive_bounding_rectangle_minus_endpoints",
            "coordinate_order": "canonical_lexicographic",
        },
        "blocker_density": {
            "numerator": "nonzero_eligible_cells",
            "denominator": "eligible_cell_count",
            "zero_capacity": 0.0,
            "domain": [0.0, 1.0],
            "used_by_policy_input": True,
            "used_by_auxiliary_target": True,
        },
        "all_model_inputs_domain": [0.0, 1.0],
        "relative_context_mapping": "(clip(offset,-4,4)+4)/8",
        "pair_selection": (
            "min(0_if_safe_else_1,manhattan+blocker_density,blocker_density,"
            "manhattan,chebyshev,lex_pair)"
        ),
        "successor_transforms": (
            "manhattan/6",
            "chebyshev/3",
            "blocker_density",
            "same_row",
            "same_column",
            "empties/16",
            "legal_actions/4",
            "live_lineage",
        ),
        "training_schedule": {
            "roots": 192,
            "trajectories_per_root": 6,
            "round_trajectories": [2, 2, 1, 1],
            "round_epsilon": [1.0, 0.15, 0.10, 0.05],
            "epochs_per_round": 5,
            "seed": 2026072804,
        },
        "architecture": {
            "token_projection": "linear37x64-gelu",
            "transformer_layers": 2,
            "attention_heads": 4,
            "feedforward_width": 128,
            "dropout": 0.0,
            "pooling": ["mean", "max", "designated"],
            "hidden": "linear227x128-gelu-layernorm",
            "output": "linear128x29",
        },
        "stage_labels_used": False,
        "score_or_human_action_input": False,
    }


def schema_sha256() -> str:
    return canonical_json_hash(schema_manifest())


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in O4DesignatedPairNet().parameters())


__all__ = [
    "CHECKPOINTS",
    "DesignatedPair",
    "EVENT_WIDTH",
    "GEOMETRY_WIDTH",
    "INTEGRATED_TARGETS",
    "LineageMove",
    "O4DesignatedPairNet",
    "OPTION_HORIZON",
    "OUTPUT_WIDTH",
    "PairBlockerGeometry",
    "TRAIN_TARGETS",
    "advance_lineage_base",
    "apply_spawn_to_lineage",
    "balanced_valid_row_weight",
    "blocker_geometry",
    "build_decision_targets",
    "choose_option_action",
    "eligible_blocker_cells",
    "exhaustive_blocker_domain_proof",
    "initial_lineage",
    "lineage_integrity",
    "option_features",
    "parameter_count",
    "root_option_eligible",
    "schema_manifest",
    "schema_sha256",
    "select_designated_pair",
    "successor_geometry",
    "transition_status",
]

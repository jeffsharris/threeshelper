"""Frozen geometry and model schema for the O1 closed-loop option policy."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from threes_rl.sim import (
    DIRECTION_NAMES,
    SimState,
    ThreesSim,
    can_merge,
    merge_value,
    simulate_base_move,
)


VERSION = "o1_goal_conditioned_geometry_option_v1_a3"
TARGET_TILES = (48, 96, 192, 384, 768, 1536)
STAGE_NAMES = (
    "separated",
    "diagonal_touching",
    "adjacent",
    "merge_ready",
)
SUCCESSOR_NAMES = (*STAGE_NAMES, "merged_success")
GOAL_NAMES = (
    "touching_or_better",
    "adjacent_or_better",
    "merge_ready_or_merged",
    "merged",
)
MIN_SAFE_EMPTIES = 2
MIN_SAFE_PRESPAWN_EMPTIES = 3
SPATIAL_CHANNELS = 16
GLOBAL_WIDTH = 28
OUTPUT_WIDTH = 20


@dataclass(frozen=True)
class PairGeometry:
    target_tile: int
    pair: tuple[tuple[int, int], tuple[int, int]]
    stage: int
    manhattan: int
    chebyshev: int
    safe_merge_actions: tuple[int, ...]

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage]


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


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


def anchor_safe(
    board: Sequence[Sequence[int]] | np.ndarray,
    starter_tile: int | None,
) -> bool:
    arr = np.asarray(board, dtype=np.int32)
    return starter_tile is None or int(arr[0, 0]) == int(starter_tile)


def air_safe(board: Sequence[Sequence[int]] | np.ndarray) -> bool:
    return int(np.count_nonzero(np.asarray(board) == 0)) >= MIN_SAFE_EMPTIES


def prespawn_air_safe(board: Sequence[Sequence[int]] | np.ndarray) -> bool:
    return (
        int(np.count_nonzero(np.asarray(board) == 0))
        >= MIN_SAFE_PRESPAWN_EMPTIES
    )


def target_tile(
    board: Sequence[Sequence[int]] | np.ndarray,
    starter_tile: int | None,
) -> int | None:
    working = normalized_board(board, starter_tile)
    eligible = [
        value
        for value in TARGET_TILES
        if int(np.count_nonzero(working == value)) >= 2
    ]
    return max(eligible, default=None)


def _advance_tagged_line(
    values: list[int],
    tags: list[frozenset[tuple[int, int]]],
) -> tuple[list[int], list[frozenset[tuple[int, int]]]]:
    cells = list(values)
    origins = list(tags)
    moved_into = [False] * len(cells)
    merged_into = [False] * len(cells)
    for idx in range(1, len(cells)):
        value = int(cells[idx])
        if value == 0:
            continue
        if cells[idx - 1] == 0:
            cells[idx - 1] = value
            cells[idx] = 0
            origins[idx - 1] = origins[idx]
            origins[idx] = frozenset()
            moved_into[idx - 1] = True
            merged_into[idx - 1] = False
        elif (
            can_merge(int(cells[idx - 1]), value)
            and not moved_into[idx - 1]
            and not merged_into[idx - 1]
        ):
            cells[idx - 1] = merge_value(int(cells[idx - 1]), value)
            cells[idx] = 0
            origins[idx - 1] = origins[idx - 1] | origins[idx]
            origins[idx] = frozenset()
            moved_into[idx - 1] = False
            merged_into[idx - 1] = True
    return cells, origins


def tagged_base_move(
    board: Sequence[Sequence[int]] | np.ndarray,
    action: int,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[tuple[int, int], frozenset[tuple[int, int]]]]:
    arr = np.asarray(board, dtype=np.int32)
    direction = DIRECTION_NAMES[int(action)]
    values = arr.T.tolist() if direction in ("up", "down") else arr.tolist()
    coordinates = (
        [[(row, col) for row in range(4)] for col in range(4)]
        if direction in ("up", "down")
        else [[(row, col) for col in range(4)] for row in range(4)]
    )
    output_values: list[list[int]] = []
    output_tags: list[list[frozenset[tuple[int, int]]]] = []
    for line_values, line_coords in zip(values, coordinates, strict=True):
        tags = [
            frozenset({coord}) if int(value) else frozenset()
            for value, coord in zip(line_values, line_coords, strict=True)
        ]
        if direction in ("right", "down"):
            shifted, shifted_tags = _advance_tagged_line(
                list(reversed(line_values)),
                list(reversed(tags)),
            )
            shifted = list(reversed(shifted))
            shifted_tags = list(reversed(shifted_tags))
        else:
            shifted, shifted_tags = _advance_tagged_line(line_values, tags)
        output_values.append(shifted)
        output_tags.append(shifted_tags)

    result = np.asarray(output_values, dtype=np.int32)
    if direction in ("up", "down"):
        result = result.T.copy()
    expected, eligible = simulate_base_move(arr, action)
    if not np.array_equal(result, expected):
        raise RuntimeError("Tagged move diverged from simulator base move")

    tag_by_output: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for line_index, line_tags in enumerate(output_tags):
        for position, tag in enumerate(line_tags):
            coord = (
                (position, line_index)
                if direction in ("up", "down")
                else (line_index, position)
            )
            if tag:
                tag_by_output[coord] = tag
    return result, eligible, tag_by_output


def pair_safe_merge_actions(
    board: Sequence[Sequence[int]] | np.ndarray,
    pair: tuple[tuple[int, int], tuple[int, int]],
    target: int,
    starter_tile: int | None,
) -> tuple[int, ...]:
    selected = frozenset(pair)
    actions: list[int] = []
    for action in range(len(DIRECTION_NAMES)):
        shifted, eligible, tags = tagged_base_move(board, action)
        if not eligible:
            continue
        merged_selected = any(
            int(shifted[coord]) == 2 * int(target)
            and selected.issubset(provenance)
            for coord, provenance in tags.items()
        )
        if (
            merged_selected
            and anchor_safe(shifted, starter_tile)
            and prespawn_air_safe(shifted)
        ):
            actions.append(action)
    return tuple(actions)


def _ordered_pairs(
    coordinates: Iterable[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    values = sorted(coordinates)
    return [
        (values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]


def geometry(
    board: Sequence[Sequence[int]] | np.ndarray,
    starter_tile: int | None,
    *,
    fixed_target: int | None = None,
) -> PairGeometry | None:
    working = normalized_board(board, starter_tile)
    target = fixed_target if fixed_target is not None else target_tile(board, starter_tile)
    if target is None:
        return None
    coordinates = [
        (int(row), int(col))
        for row, col in np.argwhere(working == int(target))
    ]
    if len(coordinates) < 2:
        return None

    rows: list[PairGeometry] = []
    for pair in _ordered_pairs(coordinates):
        (r0, c0), (r1, c1) = pair
        manhattan = abs(r0 - r1) + abs(c0 - c1)
        chebyshev = max(abs(r0 - r1), abs(c0 - c1))
        safe_actions = pair_safe_merge_actions(
            board,
            pair,
            int(target),
            starter_tile,
        )
        if safe_actions:
            stage = 3
        elif manhattan == 1:
            stage = 2
        elif chebyshev == 1 and manhattan == 2:
            stage = 1
        else:
            stage = 0
        rows.append(
            PairGeometry(
                int(target),
                pair,
                stage,
                manhattan,
                chebyshev,
                safe_actions,
            )
        )
    return min(
        rows,
        key=lambda row: (
            -row.stage,
            row.manhattan,
            row.chebyshev,
            row.pair,
        ),
    )


def root_goal(pair_geometry: PairGeometry) -> int:
    return min(4, pair_geometry.stage + 1)


def root_option_eligible(
    state: SimState,
    sim: ThreesSim,
    starter_tile: int | None,
) -> bool:
    return (
        not state.game_over
        and anchor_safe(state.board, starter_tile)
        and air_safe(state.board)
        and len(sim.legal_actions(state)) >= 2
        and geometry(state.board, starter_tile) is not None
    )


def option_status(
    state: SimState,
    sim: ThreesSim,
    *,
    starter_tile: int | None,
    target: int,
    requested_goal: int,
    root_double_count: int,
) -> str:
    working = normalized_board(state.board, starter_tile)
    safe = anchor_safe(state.board, starter_tile) and air_safe(state.board)
    merged = int(np.count_nonzero(working == 2 * target)) > root_double_count
    if safe and merged:
        return "success"
    current = geometry(state.board, starter_tile, fixed_target=target)
    if (
        safe
        and requested_goal <= 3
        and current is not None
        and current.stage >= requested_goal
    ):
        return "success"
    if (
        state.game_over
        or not anchor_safe(state.board, starter_tile)
        or not air_safe(state.board)
        or not sim.legal_actions(state)
        or int(np.count_nonzero(working == target)) < 2
    ):
        return "failure"
    return "live"


def _rank_offset(value: int, target: int) -> int:
    if value <= 2:
        return -8
    ratio = float(value) / float(target)
    exponent = math.log2(ratio)
    if abs(exponent - round(exponent)) > 1e-9:
        raise ValueError(f"Non-rank tile value {value} for target {target}")
    return int(round(exponent))


def _preview_name(state: SimState) -> str:
    return "bonus" if state.preview.kind == "bonus" else state.preview.kind


def option_features(
    state: SimState,
    sim: ThreesSim,
    *,
    starter_tile: int | None,
    pair_geometry: PairGeometry,
    requested_goal: int,
    action: int,
) -> tuple[np.ndarray, np.ndarray]:
    if action not in sim.legal_actions(state):
        raise ValueError("O1 features require a legal candidate action")
    if requested_goal not in (1, 2, 3, 4):
        raise ValueError("requested_goal must be 1..4")

    board_before = state.board.copy()
    cycle_before = sim.tile_cycle_snapshot(state)
    target = int(pair_geometry.target_tile)
    spatial = np.zeros((SPATIAL_CHANNELS, 4, 4), dtype=np.float32)
    working = normalized_board(state.board, starter_tile)
    spatial[0] = (state.board == 0).astype(np.float32)
    if starter_tile is not None and int(state.board[0, 0]) == int(starter_tile):
        spatial[1, 0, 0] = 1.0
    for row in range(4):
        for col in range(4):
            value = int(working[row, col])
            if value <= 0:
                continue
            offset = _rank_offset(value, target)
            bucket = (
                0 if offset <= -4 else
                1 if offset == -3 else
                2 if offset == -2 else
                3 if offset == -1 else
                4 if offset == 0 else
                5 if offset == 1 else
                6
            )
            spatial[2 + bucket, row, col] = 1.0
    for coord in pair_geometry.pair:
        spatial[9, coord[0], coord[1]] = 1.0
    spatial[10] = (working == target // 2).astype(np.float32)
    spatial[10 + requested_goal] = 1.0
    _shifted, eligible = simulate_base_move(state.board, action)
    for row, col in eligible:
        spatial[15, row, col] = 1.0

    global_values: list[float] = []
    preview_name = _preview_name(state)
    global_values.extend(float(preview_name == name) for name in ("blue", "red", "gray", "bonus"))
    global_values.extend(
        float(state.small_counts.get(name, 0)) / 4.0
        for name in ("red", "blue", "gray")
    )
    global_values.extend(
        (
            float(state.small_pos) / 12.0,
            min(1.0, float(state.small_seen_total) / 256.0),
            min(1.0, float(state.span_small_pos) / 21.0),
            float(state.large_pending),
        )
    )
    candidate_offsets = [
        max(-4, min(2, _rank_offset(int(value), target)))
        for value in state.preview.candidates
    ]
    if candidate_offsets:
        global_values.extend(
            (
                float(min(candidate_offsets)) / 4.0,
                float(sum(candidate_offsets) / len(candidate_offsets)) / 4.0,
                float(max(candidate_offsets)) / 4.0,
            )
        )
    else:
        global_values.extend((0.0, 0.0, 0.0))
    global_values.extend(
        float(pair_geometry.stage == stage) for stage in range(4)
    )
    safe_until = sim.safe_smalls_until_large_possible(state)
    global_values.extend(
        (
            float(np.count_nonzero(state.board == 0)) / 16.0,
            float(len(sim.legal_actions(state))) / 4.0,
            1.0 if safe_until is None else min(1.0, float(safe_until) / 21.0),
            math.log2(float(target) / 48.0) / 5.0,
            max(
                -1.0,
                min(
                    1.0,
                    float(
                        max(
                            (
                                _rank_offset(int(value), target)
                                for value in working.reshape(-1)
                                if int(value) > 0
                            ),
                            default=-4,
                        )
                    )
                    / 4.0,
                ),
            ),
            float(anchor_safe(state.board, starter_tile)),
        )
    )
    global_values.extend(float(action == candidate) for candidate in range(4))
    global_array = np.asarray(global_values, dtype=np.float32)
    if global_array.shape != (GLOBAL_WIDTH,):
        raise RuntimeError(f"O1 global width mismatch: {global_array.shape}")
    if not np.isfinite(spatial).all() or not np.isfinite(global_array).all():
        raise RuntimeError("O1 features must be finite")
    if not np.array_equal(state.board, board_before):
        raise RuntimeError("O1 feature extraction mutated board")
    if sim.tile_cycle_snapshot(state) != cycle_before:
        raise RuntimeError("O1 feature extraction mutated tile cycle")
    return spatial, global_array


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, 32)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = torch.relu(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return torch.relu(value + residual)


class O1OptionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(SPATIAL_CHANNELS, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.hidden = nn.Sequential(nn.Linear(540, 128), nn.ReLU())
        self.output = nn.Linear(128, OUTPUT_WIDTH)

    def forward(
        self,
        spatial: torch.Tensor,
        global_values: torch.Tensor,
    ) -> torch.Tensor:
        value = self.blocks(self.stem(spatial))
        value = value.reshape(value.shape[0], -1)
        value = torch.cat((value, global_values), dim=1)
        return self.output(self.hidden(value))


def schema_manifest() -> dict[str, Any]:
    return {
        "version": VERSION,
        "target_tiles": list(TARGET_TILES),
        "stage_names": list(STAGE_NAMES),
        "goal_names": list(GOAL_NAMES),
        "successor_names": list(SUCCESSOR_NAMES),
        "minimum_safe_empties": MIN_SAFE_EMPTIES,
        "minimum_safe_prespawn_empties": MIN_SAFE_PRESPAWN_EMPTIES,
        "spatial_channels": SPATIAL_CHANNELS,
        "global_width": GLOBAL_WIDTH,
        "output_width": OUTPUT_WIDTH,
        "action_order": list(DIRECTION_NAMES),
        "action_conditioned_forward": True,
        "pair_specific_tagged_merge": True,
        "architecture": {
            "stem": "conv16x32k3p1-groupnorm4-relu",
            "residual_blocks": 2,
            "hidden": "linear540x128-relu",
            "output": "linear128x20",
        },
    }


def schema_sha256() -> str:
    return canonical_json_hash(schema_manifest())

"""Exact O3 designated-pair lineage, features, and frozen model schema."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from threes_rl.o1_geometry_option import (
    air_safe,
    anchor_safe,
    pair_safe_merge_actions,
)
from threes_rl.sim import (
    DIRECTION_NAMES,
    SimState,
    ThreesSim,
    can_merge,
    merge_value,
    simulate_base_move,
)


VERSION = "o3_event_conditioned_designated_pair_v1"
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
    "blockers",
    "same_row",
    "same_column",
    "empty_count",
    "legal_count",
    "two_descendant_lineage_intact",
)


@dataclass(frozen=True)
class DesignatedPair:
    target: int
    coordinates: tuple[tuple[int, int], tuple[int, int]]
    manhattan: int
    chebyshev: int
    blocker_count: int
    same_row: bool
    same_column: bool
    clear_line: bool
    safe_merge_actions: tuple[int, ...]


@dataclass(frozen=True)
class LineageMove:
    board: np.ndarray
    eligible_slots: tuple[tuple[int, int], ...]
    lineage: np.ndarray
    event: str


@dataclass(frozen=True)
class DecisionTargets:
    event_class: int | None
    event_mask: bool
    geometry: np.ndarray
    geometry_mask: np.ndarray


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


def _ordered_pairs(
    coordinates: Iterable[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ordered = sorted(coordinates)
    return [
        (ordered[left], ordered[right])
        for left in range(len(ordered))
        for right in range(left + 1, len(ordered))
    ]


def pair_blocker_count(
    board: Sequence[Sequence[int]] | np.ndarray,
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> int:
    arr = np.asarray(board, dtype=np.int32)
    (r0, c0), (r1, c1) = pair
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return int(np.count_nonzero(arr[r0, lo + 1 : hi]))
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return int(np.count_nonzero(arr[lo + 1 : hi, c0]))
    rlo, rhi = sorted((r0, r1))
    clo, chi = sorted((c0, c1))
    occupied = int(np.count_nonzero(arr[rlo : rhi + 1, clo : chi + 1]))
    return max(0, occupied - 2)


def _pair_row(
    board: np.ndarray,
    starter_tile: int | None,
    target: int,
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> DesignatedPair:
    (r0, c0), (r1, c1) = pair
    manhattan = abs(r0 - r1) + abs(c0 - c1)
    chebyshev = max(abs(r0 - r1), abs(c0 - c1))
    blockers = pair_blocker_count(board, pair)
    same_row = r0 == r1
    same_column = c0 == c1
    safe_actions = pair_safe_merge_actions(
        board,
        pair,
        target,
        starter_tile,
    )
    return DesignatedPair(
        target=int(target),
        coordinates=pair,
        manhattan=int(manhattan),
        chebyshev=int(chebyshev),
        blocker_count=int(blockers),
        same_row=same_row,
        same_column=same_column,
        clear_line=bool((same_row or same_column) and blockers == 0),
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
    if requested_target is None:
        eligible_targets = [
            int(target)
            for target in allowed_targets
            if int(np.count_nonzero(working == int(target))) >= 2
        ]
        if not eligible_targets:
            return None
        target = max(eligible_targets)
    else:
        target = int(requested_target)
        if target not in tuple(int(value) for value in allowed_targets):
            return None
        if int(np.count_nonzero(working == target)) < 2:
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
            not bool(row.safe_merge_actions),
            row.manhattan + row.blocker_count,
            row.blocker_count,
            row.manhattan,
            row.chebyshev,
            row.coordinates,
        ),
    )


def initial_lineage(pair: DesignatedPair) -> np.ndarray:
    lineage = np.zeros((4, 4), dtype=np.uint8)
    lineage[pair.coordinates[0]] = LINEAGE_A
    lineage[pair.coordinates[1]] = LINEAGE_B
    return lineage


def _advance_lineage_line(
    values: list[int],
    lineages: list[int],
) -> tuple[list[int], list[int], bool, bool]:
    cells = list(values)
    marks = list(lineages)
    moved_into = [False] * len(cells)
    merged_into = [False] * len(cells)
    pair_merged = False
    third_party_merged = False
    for index in range(1, len(cells)):
        value = int(cells[index])
        if value == 0:
            continue
        if cells[index - 1] == 0:
            cells[index - 1] = value
            cells[index] = 0
            marks[index - 1] = int(marks[index])
            marks[index] = 0
            moved_into[index - 1] = True
            merged_into[index - 1] = False
        elif (
            can_merge(int(cells[index - 1]), value)
            and not moved_into[index - 1]
            and not merged_into[index - 1]
        ):
            left_mark = int(marks[index - 1])
            right_mark = int(marks[index])
            union = left_mark | right_mark
            if union == LINEAGE_MERGED and left_mark and right_mark:
                pair_merged = True
            elif bool(left_mark) != bool(right_mark):
                third_party_merged = True
            cells[index - 1] = merge_value(int(cells[index - 1]), value)
            cells[index] = 0
            marks[index - 1] = union
            marks[index] = 0
            moved_into[index - 1] = False
            merged_into[index - 1] = True
    return cells, marks, pair_merged, third_party_merged


def advance_lineage_base(
    board: Sequence[Sequence[int]] | np.ndarray,
    lineage: Sequence[Sequence[int]] | np.ndarray,
    action: int,
) -> LineageMove:
    arr = np.asarray(board, dtype=np.int32)
    marks = np.asarray(lineage, dtype=np.uint8)
    if arr.shape != (4, 4) or marks.shape != (4, 4):
        raise ValueError("Board and lineage must both be 4x4")
    if np.any((marks < 0) | (marks > LINEAGE_MERGED)):
        raise ValueError("Lineage values must be in 0..3")
    direction = DIRECTION_NAMES[int(action)]
    values = arr.T.tolist() if direction in ("up", "down") else arr.tolist()
    lineage_values = (
        marks.T.tolist() if direction in ("up", "down") else marks.tolist()
    )
    output_values: list[list[int]] = []
    output_lineages: list[list[int]] = []
    pair_merged = False
    third_party_merged = False
    for line_values, line_marks in zip(
        values,
        lineage_values,
        strict=True,
    ):
        if direction in ("right", "down"):
            shifted, shifted_marks, pair_hit, third_hit = (
                _advance_lineage_line(
                    list(reversed(line_values)),
                    list(reversed(line_marks)),
                )
            )
            shifted.reverse()
            shifted_marks.reverse()
        else:
            shifted, shifted_marks, pair_hit, third_hit = (
                _advance_lineage_line(line_values, line_marks)
            )
        output_values.append(shifted)
        output_lineages.append(shifted_marks)
        pair_merged = pair_merged or pair_hit
        third_party_merged = third_party_merged or third_hit

    result = np.asarray(output_values, dtype=np.int32)
    result_lineage = np.asarray(output_lineages, dtype=np.uint8)
    if direction in ("up", "down"):
        result = result.T.copy()
        result_lineage = result_lineage.T.copy()
    expected, eligible = simulate_base_move(arr, int(action))
    if not np.array_equal(result, expected):
        raise RuntimeError("O3 lineage move diverged from simulator")
    event = (
        "designated_pair_merged"
        if pair_merged
        else "third_party_merge"
        if third_party_merged
        else "live"
    )
    return LineageMove(
        board=result,
        eligible_slots=tuple((int(r), int(c)) for r, c in eligible),
        lineage=result_lineage,
        event=event,
    )


def apply_spawn_to_lineage(
    lineage: Sequence[Sequence[int]] | np.ndarray,
    insertion_slot: tuple[int, int],
) -> np.ndarray:
    result = np.asarray(lineage, dtype=np.uint8).copy()
    row, column = (int(insertion_slot[0]), int(insertion_slot[1]))
    if int(result[row, column]) != 0:
        raise ValueError("Spawn insertion cannot overwrite designated lineage")
    result[row, column] = 0
    return result


def lineage_integrity(lineage: Sequence[Sequence[int]] | np.ndarray) -> str:
    marks = np.asarray(lineage, dtype=np.uint8)
    merged_cells = int(np.count_nonzero(marks == LINEAGE_MERGED))
    a_cells = int(np.count_nonzero((marks & LINEAGE_A) != 0))
    b_cells = int(np.count_nonzero((marks & LINEAGE_B) != 0))
    if merged_cells == 1 and a_cells == 1 and b_cells == 1:
        return "merged"
    if merged_cells == 0 and a_cells == 1 and b_cells == 1:
        return "live"
    return "invalid"


def transition_status(
    state: SimState,
    sim: ThreesSim,
    *,
    starter_tile: int | None,
    lineage: Sequence[Sequence[int]] | np.ndarray,
    base_event: str,
) -> str:
    safe = anchor_safe(state.board, starter_tile) and air_safe(state.board)
    integrity = lineage_integrity(lineage)
    if base_event == "designated_pair_merged" and safe and integrity == "merged":
        return "success"
    if (
        base_event in ("designated_pair_merged", "third_party_merge")
        or integrity != "live"
        or state.game_over
        or not safe
        or not sim.legal_actions(state)
    ):
        return "failure"
    return "live"


def build_decision_targets(
    *,
    decision_move: int,
    terminal_move: int,
    terminal_status: str,
    live_geometry_by_move: dict[int, Sequence[float]],
) -> DecisionTargets:
    """Build non-fictional targets relative to one queried decision."""
    decision = int(decision_move)
    terminal = int(terminal_move)
    if not 0 <= decision < terminal <= OPTION_HORIZON:
        raise ValueError("Expected 0 <= decision_move < terminal_move <= 40")
    if terminal_status not in {"success", "failure", "censor"}:
        raise ValueError(f"Unknown terminal status: {terminal_status}")

    relative_terminal = terminal - decision
    event_class: int | None = None
    event_mask = False
    if terminal_status == "success":
        event_class = (
            0
            if relative_terminal <= 10
            else 1
            if relative_terminal <= 20
            else 2
        )
        event_mask = True
    elif terminal_status == "failure":
        event_class = 3
        event_mask = True
    elif relative_terminal == OPTION_HORIZON:
        event_class = 4
        event_mask = True

    geometry = np.zeros(
        (len(CHECKPOINTS), GEOMETRY_WIDTH),
        dtype=np.float32,
    )
    geometry_mask = np.zeros((len(CHECKPOINTS),), dtype=np.bool_)
    for checkpoint_index, offset in enumerate(CHECKPOINTS):
        absolute_move = decision + int(offset)
        if absolute_move > terminal or (
            absolute_move == terminal and terminal_status != "censor"
        ):
            continue
        values = live_geometry_by_move.get(absolute_move)
        if values is None:
            continue
        row = np.asarray(values, dtype=np.float32)
        if row.shape != (GEOMETRY_WIDTH,) or not np.isfinite(row).all():
            raise ValueError("Geometry checkpoint must be eight finite values")
        geometry[checkpoint_index] = row
        geometry_mask[checkpoint_index] = True
    return DecisionTargets(
        event_class=event_class,
        event_mask=event_mask,
        geometry=geometry,
        geometry_mask=geometry_mask,
    )


def balanced_valid_row_weight(
    *,
    represented_family_count: int,
    roots_in_family: int,
    trajectories_per_root: int,
    valid_rows_in_trajectory: int,
) -> float:
    values = (
        int(represented_family_count),
        int(roots_in_family),
        int(trajectories_per_root),
        int(valid_rows_in_trajectory),
    )
    if any(value <= 0 for value in values):
        raise ValueError("All balancing denominators must be positive")
    family_count, root_count, trajectory_count, row_count = values
    return 1.0 / (
        float(family_count)
        * float(root_count)
        * float(trajectory_count)
        * float(row_count)
    )


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
    safe_merge_probability = float(np.sum(probabilities[:included_bins]))
    nonfailure_probability = float(1.0 - probabilities[3])

    h10 = 1.0 / (1.0 + np.exp(-values[EVENT_WIDTH : EVENT_WIDTH + 8]))
    same_line = max(float(h10[3]), float(h10[4]))
    successor_potential = (
        -0.45 * float(h10[0])
        - 0.20 * float(h10[1])
        - 0.20 * float(h10[2])
        + 0.05 * same_line
        + 0.05 * float(h10[5])
        + 0.05 * float(h10[6])
    )
    return (
        safe_merge_probability,
        nonfailure_probability,
        successor_potential,
    )


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
    if any(action < 0 or action >= len(DIRECTION_NAMES) for action in actions):
        raise ValueError("Action keys must be simulator action enums")
    safe = tuple(sorted(int(action) for action in safe_merge_actions))
    if safe:
        if any(action not in outputs_by_action for action in safe):
            raise ValueError("Safe-merge actions must be legal queried actions")
        return safe[0]

    tolerance = float(tie_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    components = {
        action: action_order_components(
            outputs_by_action[action],
            remaining_horizon=remaining_horizon,
        )
        for action in actions
    }
    best = actions[0]
    for action in actions[1:]:
        for candidate_value, best_value in zip(
            components[action],
            components[best],
            strict=True,
        ):
            if candidate_value > best_value + tolerance:
                best = action
                break
            if candidate_value < best_value - tolerance:
                break
    return best


def _rank_offset(value: int, target: int) -> int:
    if value <= 0:
        raise ValueError("Rank offset requires a positive tile")
    ratio = float(value) / float(target)
    exponent = math.log2(ratio)
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


def _preview_name(state: SimState) -> str:
    return "bonus" if state.preview.kind == "bonus" else str(state.preview.kind)


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
        raise ValueError("O3 features require a legal candidate action")
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
            tokens[index, rank_category(int(state.board[row, column]), pair.target)] = 1.0
            tokens[
                index,
                11 + rank_category(int(after.board[row, column]), pair.target),
            ] = 1.0
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

    global_values: list[float] = []
    preview_name = _preview_name(state)
    global_values.extend(
        float(preview_name == name)
        for name in ("blue", "red", "gray", "bonus")
    )
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
        max(-4, min(4, _rank_offset(int(value), pair.target)))
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
        (
            float(np.count_nonzero(state.board == 0)) / 16.0,
            float(np.count_nonzero(after.board == 0)) / 16.0,
            float(len(legal)) / 4.0,
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
    global_values.append(max(-1.0, min(1.0, float(maximum_offset) / 4.0)))
    global_values.extend(
        (
            min(1.0, float(pair.manhattan) / 6.0),
            min(1.0, float(pair.chebyshev) / 3.0),
            min(1.0, float(pair.blocker_count) / 6.0),
            float(pair.same_row),
            float(pair.same_column),
            float(pair.clear_line),
            float(pair.manhattan == 1),
            float(pair.chebyshev == 1 and pair.manhattan == 2),
        )
    )
    global_values.extend(float(int(action) == candidate) for candidate in range(4))
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
        raise RuntimeError(f"O3 token shape mismatch: {tokens.shape}")
    if globals_array.shape != (GLOBAL_WIDTH,):
        raise RuntimeError(f"O3 global shape mismatch: {globals_array.shape}")
    if not np.isfinite(tokens).all() or not np.isfinite(globals_array).all():
        raise RuntimeError("O3 features must be finite")
    if not np.array_equal(state.board, board_before):
        raise RuntimeError("O3 features mutated the simulator board")
    if not np.array_equal(np.asarray(lineage), lineage_before):
        raise RuntimeError("O3 features mutated lineage")
    if sim.tile_cycle_snapshot(state) != cycle_before:
        raise RuntimeError("O3 features mutated tile-cycle state")
    if json.dumps(sim.deck_rng.bit_generator.state, sort_keys=True) != deck_rng_before:
        raise RuntimeError("O3 features consumed deck RNG")
    if json.dumps(sim.slot_rng.bit_generator.state, sort_keys=True) != slot_rng_before:
        raise RuntimeError("O3 features consumed slot RNG")
    return tokens, globals_array


class O3DesignatedPairNet(nn.Module):
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


def schema_manifest() -> dict[str, Any]:
    return {
        "version": VERSION,
        "train_targets": list(TRAIN_TARGETS),
        "integrated_targets": list(INTEGRATED_TARGETS),
        "incumbent_delegated_targets": "target>=1536",
        "minimum_safe_empties": MIN_SAFE_EMPTIES,
        "minimum_safe_prespawn_empties": MIN_SAFE_PRESPAWN_EMPTIES,
        "hard_start": "canonical_pair_has_zero_safe_merge_actions",
        "option_horizon": OPTION_HORIZON,
        "rank_categories": list(RANK_CATEGORY_NAMES),
        "token_width": TOKEN_WIDTH,
        "global_width": GLOBAL_WIDTH,
        "event_width": EVENT_WIDTH,
        "event_class_names": list(EVENT_CLASS_NAMES),
        "geometry_width": GEOMETRY_WIDTH,
        "checkpoints": list(CHECKPOINTS),
        "geometry_names": list(GEOMETRY_NAMES),
        "output_width": OUTPUT_WIDTH,
        "simulator_action_order": list(DIRECTION_NAMES),
        "pair_selection": (
            "min(no_safe_merge,manhattan+blockers,blockers,"
            "manhattan,chebyshev,lex_pair)"
        ),
        "lineage_events": (
            "designated_pair_merged",
            "third_party_merge",
            "live",
        ),
        "decision_label_semantics": {
            "unit": "chosen_action_only",
            "offset_origin": "queried_decision",
            "event_rows": (
                "observed_success_or_failure_valid;"
                "censor_valid_only_with_40_future_moves;"
                "short_live_followup_masked"
            ),
            "geometry_rows": (
                "offsets_10_20_40_only_when_live_at_exact_offset"
            ),
            "weight": "1/(families*family_roots*12*valid_rows)",
        },
        "action_ranking": {
            "safe_probability": (
                "sum softmax event bins with lower endpoint <= remaining"
            ),
            "nonfailure_probability": "1-softmax_failure",
            "h10_potential": (
                "-.45*manhattan-.20*chebyshev-.20*blockers"
                "+.05*max(same_row,same_column)+.05*empties+.05*legal"
            ),
            "tie_tolerance": 1e-12,
            "final_tie": "lowest_action_enum",
            "immediate_safe_merge": "lowest_safe_action_enum",
        },
        "lifecycle": {
            "activation_targets_descending": [768, 384, 192, 96, 48],
            "persist_pair_and_lineage": True,
            "termination": "success_or_failure_or_40_moves",
            "cooldown_complete_incumbent_moves": 1,
        },
        "stage_labels_used": False,
        "absolute_target_scalar_used": False,
        "architecture": {
            "token_projection": "linear37x64-gelu",
            "transformer_layers": 2,
            "attention_heads": 4,
            "feedforward_width": 128,
            "dropout": 0.0,
            "pooling": ("mean", "max", "designated"),
            "hidden": "linear227x128-gelu-layernorm",
            "output": "linear128x29",
        },
    }


def schema_sha256() -> str:
    return canonical_json_hash(schema_manifest())

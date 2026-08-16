"""Frozen G1 relational and positional action-afterstate feature scaffolds."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.context_residual import context_metadata
from threes_rl.ntuple import phase4_index_for_board
from threes_rl.sim import (
    DIRECTION_NAMES,
    SimState,
    ThreesSim,
    rank_for_value,
    simulate_base_move,
)


SCHEMA_VERSION = "g1_action_hazard_v1"
FEATURE_WIDTH = 64
MODEL_PARAMETER_COUNT = FEATURE_WIDTH + 1
STRATA = ("pre1536", "pre3072")
TARGET_TILE = {"pre1536": 768, "pre3072": 1536}


SHARED_FEATURE_NAMES = (
    *(f"action_{name}" for name in DIRECTION_NAMES),
    *(f"stage_{index}" for index in range(4)),
    "stratum_pre1536",
    "stratum_pre3072",
    "empty_count",
    "legal_afterstate_action_count",
    "insertion_lane_count",
    "preview_is_small",
    "preview_is_bonus",
    "preview_value_rank",
    "preview_candidate_count",
    "probability_plus_next",
    "expected_plus_value_rank",
    "small_bag_entropy",
    "small_bag_position",
    "span_position",
    "large_pending",
    "distance_to_forced_plus",
)

RELATIONAL_SPECIFIC_FEATURE_NAMES = (
    "target_count",
    "target_duplicate_exists",
    "target_pair_manhattan",
    "target_pair_chebyshev",
    "target_pair_same_row",
    "target_pair_same_column",
    "target_pair_blockers",
    "target_pair_clear_line",
    "target_pair_merge_cost",
    "target_pair_starter_manhattan",
    "target_pair_starter_chebyshev",
    "target_pair_same_row_as_starter",
    "target_pair_same_column_as_starter",
    "target_pair_row_delta_from_starter",
    "target_pair_column_delta_from_starter",
    "support_half_count",
    "support_quarter_count",
    "support_components",
    "support_adjacent_to_target",
    "support_pair_adjacent",
    "support_min_manhattan",
    "support_min_chebyshev",
    "support_rank_mass",
    "support_half_adjacent_to_target",
    "support_quarter_adjacent_to_target",
    "target_neighborhood_occupied",
    "target_neighborhood_rank_mass",
    "top_row_monotonic_violations",
    "left_column_monotonic_violations",
    "high_tile_displacement",
    "top_edge_target_count",
    "starter_present_top_left",
    "anchor_integrity",
    "action_breaks_anchor",
    "insertion_min_target_manhattan",
    "insertion_min_support_manhattan",
    "insertion_min_starter_manhattan",
    "insertion_target_axis_alignment",
    "interaction_duplicate_clear_line",
    "interaction_support_adjacency_empties",
)

POSITIONAL_SPECIFIC_FEATURE_NAMES = (
    *(f"cell_rank_{index}" for index in range(16)),
    *(f"cell_target_{index}" for index in range(16)),
    *(f"row_rank_sum_{index}" for index in range(4)),
    *(f"column_rank_sum_{index}" for index in range(4)),
)

RELATIONAL_FEATURE_NAMES = (
    *SHARED_FEATURE_NAMES,
    *RELATIONAL_SPECIFIC_FEATURE_NAMES,
)
POSITIONAL_FEATURE_NAMES = (
    *SHARED_FEATURE_NAMES,
    *POSITIONAL_SPECIFIC_FEATURE_NAMES,
)

if len(RELATIONAL_FEATURE_NAMES) != FEATURE_WIDTH:
    raise AssertionError(f"Relational schema width: {len(RELATIONAL_FEATURE_NAMES)}")
if len(POSITIONAL_FEATURE_NAMES) != FEATURE_WIDTH:
    raise AssertionError(f"Positional schema width: {len(POSITIONAL_FEATURE_NAMES)}")

SHARED_FORMULAS = {
    **{
        f"action_{name}": f"1[action=={name}]"
        for name in DIRECTION_NAMES
    },
    **{
        f"stage_{index}": f"1[phase4_index(afterstate)=={index}]"
        for index in range(4)
    },
    "stratum_pre1536": "1[stratum==pre1536]",
    "stratum_pre3072": "1[stratum==pre3072]",
    "empty_count": "count(afterstate==0)/16",
    "legal_afterstate_action_count": "count legal base moves from afterstate/4",
    "insertion_lane_count": "legal insertion slots produced by forced base move/4",
    "preview_is_small": "1[visible preview kind != bonus]",
    "preview_is_bonus": "1[visible preview kind == bonus]",
    "preview_value_rank": "small preview rank or mean visible bonus-candidate rank, divided by 14",
    "preview_candidate_count": "visible bonus candidate count/8",
    "probability_plus_next": "exact simulator P(next preview is bonus) after visible preview consumption",
    "expected_plus_value_rank": "sum P(value|plus)*rank(value)/14",
    "small_bag_entropy": "entropy(post-visible small-bag probabilities)/log(3)",
    "small_bag_position": "post-visible small_pos/12",
    "span_position": "post-visible span_small_pos/20",
    "large_pending": "1[post-visible large_pending]",
    "distance_to_forced_plus": "min(1,post-visible distance_to_forced_plus/21)",
}

RELATIONAL_FORMULAS = {
    **SHARED_FORMULAS,
    "target_count": "min(count(afterstate==target),4)/4",
    "target_duplicate_exists": "1[count(afterstate==target)>=2]",
    "target_pair_manhattan": "best_pair_manhattan/6; missing=1",
    "target_pair_chebyshev": "best_pair_chebyshev/3; missing=1",
    "target_pair_same_row": "1[best_pair shares row]; missing=0",
    "target_pair_same_column": "1[best_pair shares column]; missing=0",
    "target_pair_blockers": "min(best_pair_nonzero_between,3)/3; unaligned count=manhattan-1; missing=0",
    "target_pair_clear_line": "1[best_pair aligned and zero blockers]; missing=0",
    "target_pair_merge_cost": "min(1,(best_pair_manhattan+blockers)/6); missing=1",
    "target_pair_starter_manhattan": "min pair-cell Manhattan to (0,0)/6; missing=1",
    "target_pair_starter_chebyshev": "min pair-cell Chebyshev to (0,0)/3; missing=1",
    "target_pair_same_row_as_starter": "1[any pair cell row==0]; missing=0",
    "target_pair_same_column_as_starter": "1[any pair cell column==0]; missing=0",
    "target_pair_row_delta_from_starter": "mean best-pair row/3; missing=1",
    "target_pair_column_delta_from_starter": "mean best-pair column/3; missing=1",
    "support_half_count": "min(count(afterstate==target/2),4)/4",
    "support_quarter_count": "min(count(afterstate==target/4),4)/4",
    "support_components": "orthogonal components among target/2 and target/4 positions/8",
    "support_adjacent_to_target": "1[any support orthogonally adjacent to any target]",
    "support_pair_adjacent": "1[any two support cells orthogonally adjacent]",
    "support_min_manhattan": "minimum support-to-target Manhattan/6; missing side=1",
    "support_min_chebyshev": "minimum support-to-target Chebyshev/3; missing side=1",
    "support_rank_mass": "min(1,sum rank(support cells)/128)",
    "support_half_adjacent_to_target": "1[any target/2 orthogonally adjacent to target]",
    "support_quarter_adjacent_to_target": "1[any target/4 orthogonally adjacent to target]",
    "target_neighborhood_occupied": "occupied unique orthogonal target neighbors excluding target cells/neighborhood size; empty neighborhood=0",
    "target_neighborhood_rank_mass": "min(1,sum rank(unique orthogonal target neighbors excluding target cells)/64)",
    "top_row_monotonic_violations": "count adjacent rank increases left-to-right on top row/3",
    "left_column_monotonic_violations": "count adjacent rank increases top-to-bottom on left column/3",
    "high_tile_displacement": "count cells >=target with row>0 and column>0, divided by 16",
    "top_edge_target_count": "count(afterstate[0,:]==target)/4",
    "starter_present_top_left": "1[afterstate[0,0]==starter_tile]",
    "anchor_integrity": "mean(1[afterstate[0,0]==starter_tile],1[top_row_monotonic_violations==0],1[left_column_monotonic_violations==0])",
    "action_breaks_anchor": "1[root starter at (0,0) and afterstate starter not at (0,0)]",
    "insertion_min_target_manhattan": "minimum forced-insertion-slot to target-cell Manhattan/6; missing target=1",
    "insertion_min_support_manhattan": "minimum forced-insertion-slot to target/2-or-target/4 Manhattan/6; missing support=1",
    "insertion_min_starter_manhattan": "minimum forced-insertion-slot Manhattan to fixed starter position (0,0)/6",
    "insertion_target_axis_alignment": "fraction forced-insertion slots sharing row or column with any target cell",
    "interaction_duplicate_clear_line": "target_duplicate_exists*target_pair_clear_line",
    "interaction_support_adjacency_empties": "support_adjacent_to_target*empty_count",
}

POSITIONAL_FORMULAS = {
    **SHARED_FORMULAS,
    **{
        f"cell_rank_{index}": f"rank(afterstate.flat[{index}])/14"
        for index in range(16)
    },
    **{
        f"cell_target_{index}": f"1[afterstate.flat[{index}]==stage_target]"
        for index in range(16)
    },
    **{
        f"row_rank_sum_{index}": f"mean normalized ranks in afterstate row {index}"
        for index in range(4)
    },
    **{
        f"column_rank_sum_{index}": f"mean normalized ranks in afterstate column {index}"
        for index in range(4)
    },
}

SHARED_BINARY_FEATURES = {
    *(f"action_{name}" for name in DIRECTION_NAMES),
    *(f"stage_{index}" for index in range(4)),
    "stratum_pre1536",
    "stratum_pre3072",
    "preview_is_small",
    "preview_is_bonus",
    "large_pending",
}
RELATIONAL_BINARY_FEATURES = {
    *SHARED_BINARY_FEATURES,
    "target_duplicate_exists",
    "target_pair_same_row",
    "target_pair_same_column",
    "target_pair_clear_line",
    "target_pair_same_row_as_starter",
    "target_pair_same_column_as_starter",
    "support_adjacent_to_target",
    "support_pair_adjacent",
    "support_half_adjacent_to_target",
    "support_quarter_adjacent_to_target",
    "starter_present_top_left",
    "action_breaks_anchor",
    "interaction_duplicate_clear_line",
}
POSITIONAL_BINARY_FEATURES = {
    *SHARED_BINARY_FEATURES,
    *(f"cell_target_{index}" for index in range(16)),
}

if set(RELATIONAL_FORMULAS) != set(RELATIONAL_FEATURE_NAMES):
    raise AssertionError("Relational formulas do not exactly match feature names")
if set(POSITIONAL_FORMULAS) != set(POSITIONAL_FEATURE_NAMES):
    raise AssertionError("Positional formulas do not exactly match feature names")


def schema_payload(mode: str) -> dict[str, Any]:
    names = feature_names(mode)
    formulas = (
        RELATIONAL_FORMULAS if mode == "relational" else POSITIONAL_FORMULAS
    )
    binary = (
        RELATIONAL_BINARY_FEATURES
        if mode == "relational"
        else POSITIONAL_BINARY_FEATURES
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "width": FEATURE_WIDTH,
        "parameter_count_including_intercept": MODEL_PARAMETER_COUNT,
        "afterstate_timing": (
            "deterministic legal base move before stochastic tile insertion"
        ),
        "target_by_stratum": TARGET_TILE,
        "best_target_pair_tiebreak": (
            "minimum (Manhattan, Chebyshev, lexicographic left position, "
            "lexicographic right position)"
        ),
        "starter_reference_position": [0, 0],
        "normalization": (
            "binary columns are not normalized; every other column uses "
            "train-only root/action-weighted mean and std; constant columns "
            "are masked to zero with scale 1"
        ),
        "column_partition": {
            "shared_count": len(SHARED_FEATURE_NAMES),
            "shared_names": list(SHARED_FEATURE_NAMES),
            "representation_specific_count": FEATURE_WIDTH
            - len(SHARED_FEATURE_NAMES),
            "representation_specific_names": list(
                RELATIONAL_SPECIFIC_FEATURE_NAMES
                if mode == "relational"
                else POSITIONAL_SPECIFIC_FEATURE_NAMES
            ),
        },
        "features": [
            {
                "index": index,
                "name": name,
                "formula": formulas[name],
                "normalize_train": name not in binary,
            }
            for index, name in enumerate(names)
        ],
    }


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def schema_sha256(mode: str) -> str:
    return payload_sha256(schema_payload(mode))


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def combined_schema_manifest() -> dict[str, Any]:
    relational = schema_payload("relational")
    positional = schema_payload("positional")
    return {
        "version": SCHEMA_VERSION,
        "implementation_source": str(Path(__file__)),
        "implementation_sha256": implementation_sha256(),
        "shared_feature_count": len(SHARED_FEATURE_NAMES),
        "shared_feature_names": list(SHARED_FEATURE_NAMES),
        "shared_formulas_identical": all(
            RELATIONAL_FORMULAS[name] == POSITIONAL_FORMULAS[name]
            for name in SHARED_FEATURE_NAMES
        ),
        "relational": {
            "sha256": schema_sha256("relational"),
            "schema": relational,
        },
        "positional": {
            "sha256": schema_sha256("positional"),
            "schema": positional,
        },
    }


def root_equal_action_weights(
    legal_action_counts: list[int],
) -> list[np.ndarray]:
    if not legal_action_counts or any(count <= 0 for count in legal_action_counts):
        raise ValueError("Every G1 root must have at least one legal action")
    root_weight = 1.0 / len(legal_action_counts)
    return [
        np.full(count, root_weight / count, dtype=np.float64)
        for count in legal_action_counts
    ]


def family_balanced_action_weights(
    families: list[str],
    legal_action_counts: list[int],
) -> list[np.ndarray]:
    if len(families) != len(legal_action_counts) or not families:
        raise ValueError("G1 families and legal-action counts must align")
    if any(count <= 0 for count in legal_action_counts):
        raise ValueError("Every G1 root must have at least one legal action")
    family_names = sorted(set(families))
    family_counts = {
        family: families.count(family) for family in family_names
    }
    family_weight = 1.0 / len(family_names)
    return [
        np.full(
            count,
            family_weight / family_counts[family] / count,
            dtype=np.float64,
        )
        for family, count in zip(families, legal_action_counts)
    ]


def feature_names(mode: str) -> tuple[str, ...]:
    if mode == "relational":
        return tuple(RELATIONAL_FEATURE_NAMES)
    if mode == "positional":
        return tuple(POSITIONAL_FEATURE_NAMES)
    raise ValueError(f"Unsupported G1 feature mode: {mode}")


def _positions(board: np.ndarray, value: int) -> list[tuple[int, int]]:
    return [
        (int(row), int(column))
        for row, column in np.argwhere(board == int(value))
    ]


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _chebyshev(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def _best_pair(
    positions: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    pairs = [
        (left, right)
        for left_index, left in enumerate(positions)
        for right in positions[left_index + 1 :]
    ]
    if not pairs:
        return None
    return min(
        pairs,
        key=lambda pair: (
            _manhattan(*pair),
            _chebyshev(*pair),
            pair,
        ),
    )


def _blockers(
    board: np.ndarray,
    pair: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, bool]:
    if pair is None:
        return 0, False
    left, right = pair
    if left[0] == right[0]:
        low, high = sorted((left[1], right[1]))
        values = board[left[0], low + 1 : high]
    elif left[1] == right[1]:
        low, high = sorted((left[0], right[0]))
        values = board[low + 1 : high, left[1]]
    else:
        return _manhattan(left, right) - 1, False
    count = int(np.count_nonzero(values))
    return count, count == 0


def _component_count(positions: list[tuple[int, int]]) -> int:
    remaining = set(positions)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            row, column = stack.pop()
            adjacent = {
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            }
            found = remaining.intersection(adjacent)
            remaining.difference_update(found)
            stack.extend(found)
    return components


def _any_adjacent(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> bool:
    return any(_manhattan(a, b) == 1 for a in left for b in right if a != b)


def _minimum_distance(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
    metric: Any,
    *,
    missing: int,
) -> int:
    return min(
        (metric(a, b) for a in left for b in right if a != b),
        default=missing,
    )


def _monotonic_violations(values: np.ndarray) -> int:
    ranks = [rank_for_value(int(value)) for value in values]
    return sum(ranks[index + 1] > ranks[index] for index in range(3))


def _afterstate_legal_count(board: np.ndarray) -> int:
    return sum(bool(simulate_base_move(board, action)[1]) for action in range(4))


def _pair_starter_features(
    pair: tuple[tuple[int, int], tuple[int, int]] | None,
) -> list[float]:
    if pair is None:
        return [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    row_center = (pair[0][0] + pair[1][0]) / 2.0
    column_center = (pair[0][1] + pair[1][1]) / 2.0
    return [
        min(_manhattan(position, (0, 0)) for position in pair) / 6.0,
        min(_chebyshev(position, (0, 0)) for position in pair) / 3.0,
        float(any(position[0] == 0 for position in pair)),
        float(any(position[1] == 0 for position in pair)),
        row_center / 3.0,
        column_center / 3.0,
    ]


def _shared_features(
    state: SimState,
    sim: ThreesSim,
    action: int,
    stratum: str,
    after: np.ndarray,
    insertion_positions: list[tuple[int, int]],
    starter_tile: int | None,
) -> np.ndarray:
    metadata = context_metadata(state, sim, starter_tile=starter_tile)
    plus_distribution = metadata["next_plus_value_conditional"]
    expected_plus_rank = sum(
        float(probability) * rank_for_value(int(value))
        for value, probability in plus_distribution.items()
    )
    bag_probabilities = metadata["post_visible_bag_probabilities"]
    entropy = -sum(
        float(probability) * math.log(max(float(probability), 1e-12))
        for probability in bag_probabilities.values()
    ) / math.log(3.0)
    if state.preview.kind == "bonus" and state.preview.candidates:
        preview_rank = sum(
            rank_for_value(int(value)) for value in state.preview.candidates
        ) / len(state.preview.candidates)
    else:
        preview_rank = rank_for_value(int(state.preview.value or 0))

    stage = int(phase4_index_for_board(after, starter_tile=starter_tile))
    action_one_hot = [float(index == int(action)) for index in range(4)]
    stage_one_hot = [float(index == stage) for index in range(4)]
    stratum_one_hot = [float(stratum == name) for name in STRATA]
    values = [
        *action_one_hot,
        *stage_one_hot,
        *stratum_one_hot,
        float(np.count_nonzero(after == 0)) / 16.0,
        _afterstate_legal_count(after) / 4.0,
        len(insertion_positions) / 4.0,
        float(state.preview.kind != "bonus"),
        float(state.preview.kind == "bonus"),
        preview_rank / 14.0,
        len(state.preview.candidates) / 8.0,
        float(metadata["p_plus_next"]),
        expected_plus_rank / 14.0,
        entropy,
        int(metadata["post_visible_small_pos"]) / 12.0,
        int(metadata["post_visible_span_small_pos"]) / 20.0,
        float(metadata["post_visible_large_pending"]),
        min(1.0, int(metadata["distance_to_forced_plus"]) / 21.0),
    ]
    shared = np.asarray(values, dtype=np.float64)
    if shared.shape != (len(SHARED_FEATURE_NAMES),):
        raise AssertionError(f"Invalid shared feature width: {shared.shape}")
    return shared


def _relational_features(
    state: SimState,
    sim: ThreesSim,
    action: int,
    stratum: str,
    starter_tile: int | None,
) -> np.ndarray:
    after, insertion_positions = simulate_base_move(state.board, action)
    if not insertion_positions:
        raise ValueError(f"Illegal action for G1 features: {action}")
    shared = _shared_features(
        state,
        sim,
        action,
        stratum,
        after,
        insertion_positions,
        starter_tile,
    )
    target = TARGET_TILE[stratum]
    target_positions = _positions(after, target)
    target_pair = _best_pair(target_positions)
    target_manhattan = (
        _manhattan(*target_pair) if target_pair is not None else 6
    )
    target_chebyshev = (
        _chebyshev(*target_pair) if target_pair is not None else 3
    )
    blockers, clear_line = _blockers(after, target_pair)

    half_positions = _positions(after, target // 2)
    quarter_positions = _positions(after, target // 4)
    support_positions = half_positions + quarter_positions
    support_target_adjacent = _any_adjacent(
        support_positions,
        target_positions,
    )
    support_pair_adjacent = _any_adjacent(
        support_positions,
        support_positions,
    )

    neighborhood: set[tuple[int, int]] = set()
    for row, column in target_positions:
        for candidate in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= candidate[0] < 4
                and 0 <= candidate[1] < 4
                and candidate not in target_positions
            ):
                neighborhood.add(candidate)
    neighborhood_values = [
        int(after[position]) for position in sorted(neighborhood)
    ]

    top_violations = _monotonic_violations(after[0])
    left_violations = _monotonic_violations(after[:, 0])
    starter_present = float(
        starter_tile is not None and int(after[0, 0]) == int(starter_tile)
    )
    anchor_integrity = (
        starter_present
        + float(top_violations == 0)
        + float(left_violations == 0)
    ) / 3.0
    action_breaks_anchor = float(
        starter_tile is not None
        and int(state.board[0, 0]) == int(starter_tile)
        and int(after[0, 0]) != int(starter_tile)
    )

    target_duplicate = float(len(target_positions) >= 2)
    target_same_row = float(
        target_pair is not None and target_pair[0][0] == target_pair[1][0]
    )
    target_same_column = float(
        target_pair is not None and target_pair[0][1] == target_pair[1][1]
    )
    empty_fraction = float(np.count_nonzero(after == 0)) / 16.0
    congestion = (
        float(sum(value > 0 for value in neighborhood_values))
        / max(1, len(neighborhood_values))
    )
    support_adjacency = float(support_target_adjacent)
    insertion_target_alignment = (
        sum(
            any(
                slot[0] == target_position[0]
                or slot[1] == target_position[1]
                for target_position in target_positions
            )
            for slot in insertion_positions
        )
        / len(insertion_positions)
    )

    specific = [
        min(len(target_positions), 4) / 4.0,
        target_duplicate,
        target_manhattan / 6.0,
        target_chebyshev / 3.0,
        target_same_row,
        target_same_column,
        min(blockers, 3) / 3.0,
        float(clear_line),
        min(1.0, (target_manhattan + blockers) / 6.0),
        *_pair_starter_features(target_pair),
        min(len(half_positions), 4) / 4.0,
        min(len(quarter_positions), 4) / 4.0,
        min(_component_count(support_positions), 8) / 8.0,
        support_adjacency,
        float(support_pair_adjacent),
        _minimum_distance(
            support_positions,
            target_positions,
            _manhattan,
            missing=6,
        )
        / 6.0,
        _minimum_distance(
            support_positions,
            target_positions,
            _chebyshev,
            missing=3,
        )
        / 3.0,
        min(
            1.0,
            sum(rank_for_value(int(after[position])) for position in support_positions)
            / 128.0,
        ),
        float(_any_adjacent(half_positions, target_positions)),
        float(_any_adjacent(quarter_positions, target_positions)),
        congestion,
        min(
            1.0,
            sum(rank_for_value(value) for value in neighborhood_values) / 64.0,
        ),
        top_violations / 3.0,
        left_violations / 3.0,
        float(
            np.count_nonzero(
                (after >= target)
                & np.asarray(
                    [
                        [False, False, False, False],
                        [False, True, True, True],
                        [False, True, True, True],
                        [False, True, True, True],
                    ]
                )
            )
        )
        / 16.0,
        float(np.count_nonzero(after[0] == target)) / 4.0,
        starter_present,
        anchor_integrity,
        action_breaks_anchor,
        _minimum_distance(
            insertion_positions,
            target_positions,
            _manhattan,
            missing=6,
        )
        / 6.0,
        _minimum_distance(
            insertion_positions,
            support_positions,
            _manhattan,
            missing=6,
        )
        / 6.0,
        min(_manhattan(position, (0, 0)) for position in insertion_positions)
        / 6.0,
        insertion_target_alignment,
        target_duplicate * float(clear_line),
        support_adjacency * empty_fraction,
    ]
    features = np.concatenate((shared, np.asarray(specific, dtype=np.float64)))
    if features.shape != (FEATURE_WIDTH,) or not np.all(np.isfinite(features)):
        raise AssertionError(f"Invalid relational features: {features.shape}")
    return features


def _positional_features(
    state: SimState,
    sim: ThreesSim,
    action: int,
    stratum: str,
    starter_tile: int | None,
) -> np.ndarray:
    after, insertion_positions = simulate_base_move(state.board, action)
    if not insertion_positions:
        raise ValueError(f"Illegal action for G1 features: {action}")
    shared = _shared_features(
        state,
        sim,
        action,
        stratum,
        after,
        insertion_positions,
        starter_tile,
    )
    target = TARGET_TILE[stratum]
    ranks = np.asarray(
        [rank_for_value(int(value)) / 14.0 for value in after.reshape(-1)],
        dtype=np.float64,
    )
    targets = (after.reshape(-1) == target).astype(np.float64)
    rank_board = ranks.reshape(4, 4)
    features = np.concatenate(
        (
            shared,
            ranks,
            targets,
            np.sum(rank_board, axis=1) / 4.0,
            np.sum(rank_board, axis=0) / 4.0,
        )
    )
    if features.shape != (FEATURE_WIDTH,) or not np.all(np.isfinite(features)):
        raise AssertionError(f"Invalid positional features: {features.shape}")
    return features


def encode_action_afterstate(
    state: SimState,
    sim: ThreesSim,
    *,
    action: int,
    stratum: str,
    mode: str,
    starter_tile: int | None = 1536,
) -> np.ndarray:
    if stratum not in STRATA:
        raise ValueError(f"Unsupported G1 stratum: {stratum}")
    if int(action) not in sim.legal_actions(state):
        raise ValueError(f"Action {action} is not legal")
    if mode == "relational":
        return _relational_features(
            state,
            sim,
            int(action),
            stratum,
            starter_tile,
        )
    if mode == "positional":
        return _positional_features(
            state,
            sim,
            int(action),
            stratum,
            starter_tile,
        )
    raise ValueError(f"Unsupported G1 feature mode: {mode}")


@dataclass
class G1LogisticModel:
    mode: str
    weights: np.ndarray
    intercept: float
    calibration_intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    @classmethod
    def zero(cls, mode: str) -> "G1LogisticModel":
        feature_names(mode)
        return cls(
            mode=mode,
            weights=np.zeros(FEATURE_WIDTH, dtype=np.float64),
            intercept=0.0,
            calibration_intercept=0.0,
            feature_mean=np.zeros(FEATURE_WIDTH, dtype=np.float64),
            feature_scale=np.ones(FEATURE_WIDTH, dtype=np.float64),
        )

    @property
    def parameter_count(self) -> int:
        return int(self.weights.size + 1)

    def predict_probability(self, features: np.ndarray) -> float:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (FEATURE_WIDTH,):
            raise ValueError(f"Expected {FEATURE_WIDTH} features")
        normalized = (values - self.feature_mean) / self.feature_scale
        logit = (
            float(normalized @ self.weights)
            + float(self.intercept)
            + float(self.calibration_intercept)
        )
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": schema_sha256(self.mode),
            "mode": self.mode,
            "feature_width": FEATURE_WIDTH,
            "parameter_count": self.parameter_count,
            "intercept": float(self.intercept),
            "calibration_intercept": float(self.calibration_intercept),
        }
        (directory / "meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        np.savez_compressed(
            directory / "arrays.npz",
            weights=self.weights,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
        )

    @classmethod
    def load(cls, directory: Path, *, expected_mode: str) -> "G1LogisticModel":
        metadata = json.loads((directory / "meta.json").read_text())
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Incompatible G1 model schema version")
        if metadata.get("mode") != expected_mode:
            raise ValueError("Incompatible G1 model feature mode")
        if metadata.get("schema_sha256") != schema_sha256(expected_mode):
            raise ValueError("Incompatible G1 model feature schema")
        if int(metadata.get("feature_width", -1)) != FEATURE_WIDTH:
            raise ValueError("Invalid G1 model feature width")
        if int(metadata.get("parameter_count", -1)) != MODEL_PARAMETER_COUNT:
            raise ValueError("Invalid G1 model parameter count")
        arrays = np.load(directory / "arrays.npz")
        model = cls(
            mode=expected_mode,
            weights=np.asarray(arrays["weights"], dtype=np.float64),
            intercept=float(metadata["intercept"]),
            calibration_intercept=float(metadata["calibration_intercept"]),
            feature_mean=np.asarray(arrays["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(arrays["feature_scale"], dtype=np.float64),
        )
        if (
            model.weights.shape != (FEATURE_WIDTH,)
            or model.feature_mean.shape != (FEATURE_WIDTH,)
            or model.feature_scale.shape != (FEATURE_WIDTH,)
        ):
            raise ValueError("Invalid G1 model array shape")
        if (
            not np.all(np.isfinite(model.weights))
            or not np.all(np.isfinite(model.feature_mean))
            or not np.all(np.isfinite(model.feature_scale))
            or not math.isfinite(model.intercept)
            or not math.isfinite(model.calibration_intercept)
        ):
            raise ValueError("Nonfinite G1 model payload")
        if np.any(model.feature_scale <= 0.0):
            raise ValueError("Invalid G1 model feature scale")
        return model

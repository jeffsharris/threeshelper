"""Scale-equivariant relational afterstate features for the G2 hazard preflight."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from threes_rl.sim import (
    DOWN,
    LARGE_DELAY_PREVIEWS,
    LARGE_SPAN_SMALLS,
    LEFT,
    RIGHT,
    SMALL_BAG_SIZE,
    UP,
    SimState,
    ThreesSim,
    simulate_base_move,
)


VERSION = "g2_scale_equivariant_relational_hazard_schema_v1"
FEATURE_WIDTH = 64
HORIZONS = (10, 20, 40)
TARGETS = (768, 1536, 3072)
ACTION_TRANSPOSE = {UP: LEFT, DOWN: RIGHT, LEFT: UP, RIGHT: DOWN}
STARTER_POSITION = (0, 0)

FEATURE_NAMES = (
    "h10",
    "h20",
    "h40",
    "action_up",
    "action_down",
    "action_left",
    "action_right",
    "empty_fraction",
    "legal_mobility_fraction",
    "moved_cell_fraction",
    "merge_count_fraction",
    "insertion_lane_fraction",
    "preview_is_small",
    "preview_is_bonus",
    "preview_relative_rank",
    "preview_candidate_fraction",
    "p_plus_next",
    "large_pending",
    "distance_to_forced_plus",
    "small_bag_entropy",
    "small_bag_position",
    "span_position",
    "parent_count_fraction",
    "child_count_fraction",
    "grandchild_count_fraction",
    "parent_pair_exists",
    "parent_pair_chebyshev",
    "parent_pair_manhattan",
    "parent_pair_same_row",
    "parent_pair_same_column",
    "parent_pair_diagonal_touch",
    "parent_pair_line_clear",
    "parent_pair_blockers",
    "parent_pair_action_aligned",
    "parent_pair_min_anchor_distance",
    "parent_pair_max_anchor_distance",
    "parent_pair_axis_imbalance",
    "support_graph_node_fraction",
    "support_graph_component_fraction",
    "support_graph_edge4_fraction",
    "support_graph_diagonal_edge_fraction",
    "support_parent_adj4_fraction",
    "support_parent_adj8_fraction",
    "parent_neighborhood_occupied_fraction",
    "parent_neighborhood_support_fraction",
    "top_row_monotonic_violations",
    "left_column_monotonic_violations",
    "anchor_integrity",
    "high_tile_displacement_fraction",
    "insertion_near_parent_fraction",
    "insertion_near_support_fraction",
    "insertion_anchor_distance",
    "clear_merge_path_fraction",
    "blocked_merge_path_fraction",
    "support_ladder_gap",
    "max_relative_rank",
    "second_relative_rank",
    "largest_parent_component_fraction",
    "largest_support_component_fraction",
    "board_relative_rank_mean",
    "board_relative_rank_spread",
    "parent_orientation_faces_anchor",
    "support_mass_fraction",
    "legal_mobility_delta",
)

if len(FEATURE_NAMES) != FEATURE_WIDTH or len(set(FEATURE_NAMES)) != FEATURE_WIDTH:
    raise AssertionError("G2 feature names must be exactly 64 unique columns")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _tile_level(value: int) -> int:
    value = int(value)
    if value < 3:
        raise ValueError(f"Tile {value} has no power-of-two Threes level")
    ratio = value // 3
    if value % 3 or ratio <= 0 or ratio & (ratio - 1):
        raise ValueError(f"Invalid Threes tile value: {value}")
    return ratio.bit_length() - 1


def _relative_code(value: int, target: int, *, starter: bool = False) -> int:
    if starter:
        return 9
    value = int(value)
    if value == 0:
        return -10
    if value == 1:
        return -9
    if value == 2:
        return -8
    return int(np.clip(_tile_level(value) - _tile_level(target), -6, 1))


def _relative_rank(value: int, target: int, *, starter: bool = False) -> float:
    if starter or int(value) <= 2:
        return 0.0
    delta = int(np.clip(_tile_level(int(value)) - _tile_level(target), -6, 1))
    return (delta + 6.0) / 7.0


def _board_codes(
    board: np.ndarray,
    target: int,
    starter_tile: int | None,
) -> tuple[int, ...]:
    values: list[int] = []
    for row in range(4):
        for column in range(4):
            is_starter = (
                starter_tile is not None
                and (row, column) == STARTER_POSITION
                and int(board[row, column]) == int(starter_tile)
            )
            values.append(
                _relative_code(int(board[row, column]), target, starter=is_starter)
            )
    return tuple(values)


def canonical_orientation(
    board: np.ndarray,
    action: int,
    target: int,
    starter_tile: int | None,
) -> tuple[np.ndarray, int, bool]:
    """Return the identity/transpose canonical board and action."""
    board = np.asarray(board, dtype=np.int32)
    identity_key = (_board_codes(board, target, starter_tile), int(action))
    transposed = board.T.copy()
    transposed_action = ACTION_TRANSPOSE[int(action)]
    transpose_key = (
        _board_codes(transposed, target, starter_tile),
        int(transposed_action),
    )
    if transpose_key < identity_key:
        return transposed, transposed_action, True
    return board.copy(), int(action), False


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
    positions: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    ordered = sorted(set(positions))
    if len(ordered) < 2:
        return None
    return min(
        combinations(ordered, 2),
        key=lambda pair: (
            _chebyshev(pair[0], pair[1]),
            _manhattan(pair[0], pair[1]),
            pair[0][0],
            pair[0][1],
            pair[1][0],
            pair[1][1],
        ),
    )


def _line_blockers(
    board: np.ndarray,
    pair: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, bool]:
    if pair is None:
        return 0, False
    left, right = pair
    if left[0] == right[0]:
        low, high = sorted((left[1], right[1]))
        between = board[left[0], low + 1 : high]
    elif left[1] == right[1]:
        low, high = sorted((left[0], right[0]))
        between = board[low + 1 : high, left[1]]
    else:
        return 0, False
    blockers = int(np.count_nonzero(between))
    return blockers, blockers == 0


def _component_sizes(
    positions: Iterable[tuple[int, int]],
    *,
    diagonal: bool = False,
) -> list[int]:
    remaining = set(positions)
    sizes: list[int] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 0
        while stack:
            row, column = stack.pop()
            size += 1
            neighbors = {
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            }
            if diagonal:
                neighbors.update(
                    {
                        (row - 1, column - 1),
                        (row - 1, column + 1),
                        (row + 1, column - 1),
                        (row + 1, column + 1),
                    }
                )
            found = remaining.intersection(neighbors)
            remaining.difference_update(found)
            stack.extend(found)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _edge_count(
    positions: Iterable[tuple[int, int]],
    *,
    distance: int,
) -> int:
    return sum(
        _chebyshev(left, right) == distance
        for left, right in combinations(sorted(set(positions)), 2)
    )


def _cross_adjacency_fraction(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
    *,
    diagonal: bool,
) -> float:
    pairs = [(a, b) for a in left for b in right if a != b]
    if not pairs:
        return 0.0
    if diagonal:
        hits = sum(_chebyshev(a, b) == 1 for a, b in pairs)
    else:
        hits = sum(_manhattan(a, b) == 1 for a, b in pairs)
    return hits / len(pairs)


def _relative_line_violations(
    values: np.ndarray,
    target: int,
    starter_tile: int | None,
    *,
    starter_axis: bool,
) -> int:
    ranks: list[int] = []
    for index, value in enumerate(values.tolist()):
        is_starter = (
            starter_axis
            and index == 0
            and starter_tile is not None
            and int(value) == int(starter_tile)
        )
        ranks.append(_relative_code(int(value), target, starter=is_starter))
    return sum(ranks[index + 1] > ranks[index] for index in range(3))


def _legal_count(board: np.ndarray) -> int:
    return sum(bool(simulate_base_move(board, action)[1]) for action in range(4))


def _transpose_positions(
    positions: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [(column, row) for row, column in positions]


def _preview_relative_rank(state: SimState, target: int) -> float:
    if state.preview.kind == "bonus":
        values = list(state.preview.candidates)
    else:
        values = [int(state.preview.value or 0)]
    if not values:
        return 0.0
    return float(np.mean([_relative_rank(value, target) for value in values]))


def _plus_probability(state: SimState, sim: ThreesSim) -> float:
    return sum(
        float(option.probability)
        for option in sim.preview_options(
            state.small_counts,
            state.small_pos,
            state.small_seen_total,
            state.span_small_pos,
            state.large_pending,
            state.max_tile,
        )
        if option.preview.kind == "bonus"
    )


def _entropy(counts: dict[str, int]) -> float:
    total = sum(max(0, int(value)) for value in counts.values())
    if total <= 0:
        return 0.0
    probabilities = [
        max(0, int(value)) / total
        for value in counts.values()
        if int(value) > 0
    ]
    return -sum(p * math.log(p) for p in probabilities) / math.log(3.0)


def _path_counts(
    board: np.ndarray,
    positions: list[tuple[int, int]],
) -> tuple[int, int]:
    clear = 0
    blocked = 0
    for pair in combinations(sorted(set(positions)), 2):
        if pair[0][0] != pair[1][0] and pair[0][1] != pair[1][1]:
            continue
        blockers, is_clear = _line_blockers(board, pair)
        if is_clear:
            clear += 1
        elif blockers:
            blocked += 1
    return clear, blocked


def _schema_formulas() -> tuple[str, ...]:
    return (
        "1[horizon==10]",
        "1[horizon==20]",
        "1[horizon==40]",
        "1[canonical_action==up]",
        "1[canonical_action==down]",
        "1[canonical_action==left]",
        "1[canonical_action==right]",
        "count(after==0)/16",
        "count_legal_base_moves(after)/4",
        "count(before!=after)/16",
        "clip(count_nonzero(before)-count_nonzero(after),0,4)/4",
        "len(canonical_insertion_slots)/4",
        "1[visible_preview.kind!=bonus]",
        "1[visible_preview.kind==bonus]",
        "mean(relative_rank(visible_preview_values,T)); empty=0",
        "len(visible_preview.candidates)/8",
        "sum(next_preview_probability(kind==bonus))",
        "1[tile_cycle.large_pending]",
        "clip(safe_smalls_until_large_possible,0,21)/21; missing=1",
        "entropy(remaining_small_bag)/log(3)",
        "tile_cycle.small_pos/12",
        "tile_cycle.span_small_pos/20",
        "min(count(after==T/2),4)/4",
        "min(count(after==T/4),4)/4",
        "min(count(after==T/8),4)/4",
        "1[at least two T/2 cells]",
        "best_parent_pair_chebyshev/3; missing=1",
        "best_parent_pair_manhattan/6; missing=1",
        "1[best_parent_pair shares row]",
        "1[best_parent_pair shares column]",
        "1[best_parent_pair Chebyshev=1 and Manhattan=2]",
        "1[best_parent_pair aligned and has zero occupied between]",
        "min(best_parent_pair_occupied_between,3)/3; missing=0",
        "1[row pair and action left/right or column pair and action up/down]",
        "min_parent_pair_manhattan_to_anchor/6; missing=1",
        "max_parent_pair_manhattan_to_anchor/6; missing=1",
        "abs(pair_mean_row-pair_mean_col)/3; missing=0",
        "count(unique T/2,T/4,T/8 positions)/16",
        "four_neighbor_component_count(support_graph)/16",
        "four_neighbor_edge_count(support_graph)/24",
        "(eight_neighbor_edges-four_neighbor_edges)/18",
        "four_neighbor_edges(parent,lower_support)/possible_cross_pairs",
        "eight_neighbor_edges(parent,lower_support)/possible_cross_pairs",
        "occupied unique four-neighborhood cells around parents/neighbor_count",
        "support-valued unique four-neighborhood cells around parents/neighbor_count",
        "top_row_relative_rank_increases/3",
        "left_column_relative_rank_increases/3",
        "mean(starter_at_00,top_violations_zero,left_violations_zero)",
        "count(nonstarter after>=T/2 outside top row and left column)/16",
        "fraction insertion slots Manhattan<=1 from a parent",
        "fraction insertion slots Manhattan<=1 from any lower support",
        "mean insertion-slot Manhattan distance to anchor/6",
        "min(clear aligned T/2 pair paths,6)/6",
        "min(blocked aligned T/2 pair paths,6)/6",
        "missing levels among T/2,T/4,T/8 divided by 3",
        "largest nonstarter nonempty relative_rank(value,T); missing=0",
        "second-largest nonstarter nonempty relative_rank(value,T); missing=0",
        "largest four-neighbor T/2 component/16",
        "largest four-neighbor T/2,T/4,T/8 component/16",
        "mean nonstarter nonempty relative_rank(value,T); missing=0",
        "2*stddev nonstarter nonempty relative_rank(value,T), clipped to 1",
        "1[row pair and action left or column pair and action up]",
        "sum(4*count(T/2)+2*count(T/4)+count(T/8))/64",
        "clip((legal_after-legal_before+4)/8,0,1)",
    )


def schema_manifest() -> dict[str, Any]:
    formulas = _schema_formulas()
    if len(formulas) != FEATURE_WIDTH:
        raise AssertionError("G2 formula manifest width mismatch")
    columns = [
        {
            "index": index,
            "name": name,
            "formula": formulas[index],
            "domain": "[0,1]",
            "missing": "formula-specific deterministic convention",
            "train_standardize": name
            not in {
                "h10",
                "h20",
                "h40",
                "action_up",
                "action_down",
                "action_left",
                "action_right",
                "preview_is_small",
                "preview_is_bonus",
                "large_pending",
                "parent_pair_exists",
                "parent_pair_same_row",
                "parent_pair_same_column",
                "parent_pair_diagonal_touch",
                "parent_pair_line_clear",
                "parent_pair_action_aligned",
                "anchor_integrity",
                "parent_orientation_faces_anchor",
            },
        }
        for index, name in enumerate(FEATURE_NAMES)
    ]
    return {
        "version": VERSION,
        "width": FEATURE_WIDTH,
        "columns": columns,
        "tile_level": "level(v)=log2(v/3) for v>=3",
        "relative_rank": "clip(level(v)-level(T),-6,+1), encoded as (delta+6)/7",
        "support_levels": {"parent": "T/2", "child": "T/4", "grandchild": "T/8"},
        "pair_tie": (
            "lexicographic(Chebyshev,Manhattan,row1,col1,row2,col2) after "
            "coordinate sorting"
        ),
        "graph_connectivity": "four-neighbor components; diagonal edges separate",
        "orientation": (
            "lexicographic minimum of identity and main-diagonal transpose; "
            "ties choose identity; actions map up<->left and down<->right"
        ),
        "rng_contract": "deterministic post-swipe/pre-spawn; consume no RNG",
    }


def schema_sha256() -> str:
    return hashlib.sha256(_canonical_json(schema_manifest())).hexdigest()


def feature_vector(
    state: SimState,
    sim: ThreesSim,
    action: int,
    *,
    target: int,
    horizon: int,
    starter_tile: int | None = 1536,
) -> np.ndarray:
    if int(target) not in TARGETS:
        raise ValueError(f"Unsupported G2 target scale: {target}")
    if int(horizon) not in HORIZONS:
        raise ValueError(f"Unsupported G2 horizon: {horizon}")
    if int(action) not in (UP, DOWN, LEFT, RIGHT):
        raise ValueError(f"Unsupported action: {action}")

    before_raw = np.asarray(state.board, dtype=np.int32)
    after_raw, insertion_raw = simulate_base_move(before_raw, int(action))
    if not insertion_raw:
        raise ValueError(f"Illegal action for G2 features: {action}")

    _canonical_before, canonical_action, transposed = canonical_orientation(
        before_raw,
        int(action),
        int(target),
        starter_tile,
    )
    before = before_raw.T.copy() if transposed else before_raw.copy()
    after = after_raw.T.copy() if transposed else after_raw.copy()
    insertion = (
        _transpose_positions(insertion_raw) if transposed else list(insertion_raw)
    )

    parent_value = int(target) // 2
    child_value = int(target) // 4
    grandchild_value = int(target) // 8
    parents = _positions(after, parent_value)
    children = _positions(after, child_value)
    grandchildren = _positions(after, grandchild_value)
    lower_support = children + grandchildren
    support = parents + lower_support
    pair = _best_pair(parents)
    blockers, line_clear = _line_blockers(after, pair)

    top_violations = _relative_line_violations(
        after[0],
        int(target),
        starter_tile,
        starter_axis=True,
    )
    left_violations = _relative_line_violations(
        after[:, 0],
        int(target),
        starter_tile,
        starter_axis=True,
    )
    starter_present = float(
        starter_tile is not None and int(after[0, 0]) == int(starter_tile)
    )

    if pair is None:
        pair_cheb = 3
        pair_manh = 6
        pair_same_row = False
        pair_same_column = False
        pair_diagonal = False
        pair_min_anchor = 6
        pair_max_anchor = 6
        pair_axis_imbalance = 0.0
    else:
        pair_cheb = _chebyshev(pair[0], pair[1])
        pair_manh = _manhattan(pair[0], pair[1])
        pair_same_row = pair[0][0] == pair[1][0]
        pair_same_column = pair[0][1] == pair[1][1]
        pair_diagonal = pair_cheb == 1 and pair_manh == 2
        anchor_distances = [_manhattan(position, STARTER_POSITION) for position in pair]
        pair_min_anchor = min(anchor_distances)
        pair_max_anchor = max(anchor_distances)
        pair_axis_imbalance = abs(
            (pair[0][0] + pair[1][0]) / 2.0
            - (pair[0][1] + pair[1][1]) / 2.0
        )

    pair_action_aligned = (
        (pair_same_row and canonical_action in (LEFT, RIGHT))
        or (pair_same_column and canonical_action in (UP, DOWN))
    )
    pair_faces_anchor = (
        (pair_same_row and canonical_action == LEFT)
        or (pair_same_column and canonical_action == UP)
    )

    support_unique = sorted(set(support))
    support_components = _component_sizes(support_unique)
    parent_components = _component_sizes(parents)
    edge4 = sum(
        _manhattan(left, right) == 1
        for left, right in combinations(support_unique, 2)
    )
    edge8 = _edge_count(support_unique, distance=1)

    neighborhood: set[tuple[int, int]] = set()
    for row, column in parents:
        for candidate in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= candidate[0] < 4
                and 0 <= candidate[1] < 4
                and candidate not in parents
            ):
                neighborhood.add(candidate)
    neighborhood_values = [int(after[position]) for position in sorted(neighborhood)]

    insertion_near_parent = (
        sum(
            any(_manhattan(slot, parent) <= 1 for parent in parents)
            for slot in insertion
        )
        / len(insertion)
        if parents
        else 0.0
    )
    insertion_near_support = (
        sum(
            any(_manhattan(slot, node) <= 1 for node in lower_support)
            for slot in insertion
        )
        / len(insertion)
        if lower_support
        else 0.0
    )

    clear_paths, blocked_paths = _path_counts(after, parents)
    missing_levels = sum(
        not positions for positions in (parents, children, grandchildren)
    )

    relative_values: list[float] = []
    for row in range(4):
        for column in range(4):
            value = int(after[row, column])
            if value <= 0:
                continue
            if (
                starter_tile is not None
                and (row, column) == STARTER_POSITION
                and value == int(starter_tile)
            ):
                continue
            relative_values.append(_relative_rank(value, int(target)))
    sorted_relative = sorted(relative_values, reverse=True)

    legal_before = _legal_count(before)
    legal_after = _legal_count(after)
    safe_smalls = sim.safe_smalls_until_large_possible(state)
    preview_candidate_count = (
        len(state.preview.candidates) if state.preview.kind == "bonus" else 0
    )

    moved_fraction = float(np.count_nonzero(before != after)) / 16.0
    merge_count = max(
        0,
        int(np.count_nonzero(before)) - int(np.count_nonzero(after)),
    )
    high_displaced = 0
    for row in range(1, 4):
        for column in range(1, 4):
            value = int(after[row, column])
            if value >= parent_value and not (
                starter_tile is not None
                and (row, column) == STARTER_POSITION
                and value == int(starter_tile)
            ):
                high_displaced += 1

    values = [
        float(horizon == 10),
        float(horizon == 20),
        float(horizon == 40),
        float(canonical_action == UP),
        float(canonical_action == DOWN),
        float(canonical_action == LEFT),
        float(canonical_action == RIGHT),
        float(np.count_nonzero(after == 0)) / 16.0,
        legal_after / 4.0,
        moved_fraction,
        min(merge_count, 4) / 4.0,
        len(insertion) / 4.0,
        float(state.preview.kind != "bonus"),
        float(state.preview.kind == "bonus"),
        _preview_relative_rank(state, int(target)),
        preview_candidate_count / 8.0,
        _plus_probability(state, sim),
        float(state.large_pending),
        1.0 if safe_smalls is None else min(21, int(safe_smalls)) / 21.0,
        _entropy(state.small_counts),
        min(SMALL_BAG_SIZE, max(0, int(state.small_pos))) / SMALL_BAG_SIZE,
        min(LARGE_SPAN_SMALLS, max(0, int(state.span_small_pos)))
        / LARGE_SPAN_SMALLS,
        min(len(parents), 4) / 4.0,
        min(len(children), 4) / 4.0,
        min(len(grandchildren), 4) / 4.0,
        float(pair is not None),
        pair_cheb / 3.0,
        pair_manh / 6.0,
        float(pair_same_row),
        float(pair_same_column),
        float(pair_diagonal),
        float(line_clear),
        min(blockers, 3) / 3.0,
        float(pair_action_aligned),
        pair_min_anchor / 6.0,
        pair_max_anchor / 6.0,
        pair_axis_imbalance / 3.0,
        len(support_unique) / 16.0,
        len(support_components) / 16.0,
        edge4 / 24.0,
        max(0, edge8 - edge4) / 18.0,
        _cross_adjacency_fraction(parents, lower_support, diagonal=False),
        _cross_adjacency_fraction(parents, lower_support, diagonal=True),
        (
            sum(value > 0 for value in neighborhood_values)
            / len(neighborhood_values)
            if neighborhood_values
            else 0.0
        ),
        (
            sum(value in {parent_value, child_value, grandchild_value} for value in neighborhood_values)
            / len(neighborhood_values)
            if neighborhood_values
            else 0.0
        ),
        top_violations / 3.0,
        left_violations / 3.0,
        (
            starter_present
            + float(top_violations == 0)
            + float(left_violations == 0)
        )
        / 3.0,
        high_displaced / 16.0,
        insertion_near_parent,
        insertion_near_support,
        float(np.mean([_manhattan(slot, STARTER_POSITION) for slot in insertion]))
        / 6.0,
        min(clear_paths, 6) / 6.0,
        min(blocked_paths, 6) / 6.0,
        missing_levels / 3.0,
        sorted_relative[0] if sorted_relative else 0.0,
        sorted_relative[1] if len(sorted_relative) > 1 else 0.0,
        (parent_components[0] if parent_components else 0) / 16.0,
        (support_components[0] if support_components else 0) / 16.0,
        float(np.mean(relative_values)) if relative_values else 0.0,
        min(1.0, 2.0 * float(np.std(relative_values)))
        if relative_values
        else 0.0,
        float(pair_faces_anchor),
        min(
            64,
            4 * len(parents) + 2 * len(children) + len(grandchildren),
        )
        / 64.0,
        (legal_after - legal_before + 4) / 8.0,
    ]
    features = np.asarray([_clip01(value) for value in values], dtype=np.float64)
    if features.shape != (FEATURE_WIDTH,):
        raise AssertionError(f"Invalid G2 feature width: {features.shape}")
    if not np.all(np.isfinite(features)):
        raise AssertionError("G2 features must be finite")
    return features

"""Search prefixes that accumulate raw 768 material before first raw 1536."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.rare_event_frontier import FrontierCase, load_cases, load_records, parse_root_origins, select_diverse_cases
from threes_rl.record_replay import state_payload
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board
from threes_rl.support_ladder_window_reservoir import raw_ladder_features
from threes_rl.train_td import copy_state

SUPPORTED_TARGETS = (
    "raw_one_192_with_one_768_no_1536",
    "raw_one_384_with_one_768_no_1536",
    "raw_two_384_with_one_768_no_1536",
    "raw_adjacent_384_with_one_768_no_1536",
    "raw_two_768_no_1536",
    "raw_adjacent_768_no_1536",
    "raw_three_768_no_1536",
    "raw_four_768_no_1536",
    "raw_four_adjacent_768_no_1536",
    "raw_duplicate_1536",
)
SUPPORTED_RANK_PROFILES = ("default", "buildable", "bridge", "second768", "supportstock")


@dataclass
class AccumulationNode:
    case: FrontierCase
    state: SimState
    depth: int
    path: list[str]
    seed_path: list[int]
    start_score: int
    score_delta: int
    parent_id: str | None
    node_id: str


def _state_digest(state: SimState) -> str:
    raw = json.dumps(
        {
            "board": np.asarray(state.board, dtype=int).tolist(),
            "preview": state.preview.label,
            "cycle": {
                "small_counts": sorted((str(key), int(value)) for key, value in state.small_counts.items()),
                "small_pos": int(state.small_pos),
                "small_seen_total": int(state.small_seen_total),
                "span_small_pos": int(state.span_small_pos),
                "large_pending": bool(state.large_pending),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _tile_positions(board: np.ndarray, value: int) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in pos) for pos in np.argwhere(np.asarray(board, dtype=np.int32) == int(value))]


def _has_adjacent_pair(board: np.ndarray, value: int) -> bool:
    arr = np.asarray(board, dtype=np.int32)
    positions = np.argwhere(arr == int(value))
    position_set = {tuple(int(v) for v in pos) for pos in positions}
    for row, col in position_set:
        if (row + 1, col) in position_set or (row, col + 1) in position_set:
            return True
    return False


def _support_geometry(board: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(board, dtype=np.int32)
    positions = _tile_positions(arr, 768)
    position_set = set(positions)
    adjacent_pairs = 0
    air_neighbors: set[tuple[int, int]] = set()
    visited: set[tuple[int, int]] = set()
    component_sizes: list[int] = []
    for row, col in positions:
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nrow = row + drow
            ncol = col + dcol
            if not (0 <= nrow < arr.shape[0] and 0 <= ncol < arr.shape[1]):
                continue
            if (nrow, ncol) in position_set and (nrow, ncol) > (row, col):
                adjacent_pairs += 1
            if int(arr[nrow, ncol]) == 0:
                air_neighbors.add((nrow, ncol))
    for start in positions:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            row, col = stack.pop()
            size += 1
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (row + drow, col + dcol)
                if nxt in position_set and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        component_sizes.append(size)
    return {
        "raw_768_adjacent_pairs": int(adjacent_pairs),
        "raw_768_components": int(len(component_sizes)),
        "raw_768_max_component": int(max(component_sizes, default=0)),
        "raw_768_air_neighbors": int(len(air_neighbors)),
        "raw_768_edge_count": int(sum(row in {0, arr.shape[0] - 1} or col in {0, arr.shape[1] - 1} for row, col in positions)),
        "raw_768_corner_count": int(
            sum(
                (row, col)
                in {
                    (0, 0),
                    (0, arr.shape[1] - 1),
                    (arr.shape[0] - 1, 0),
                    (arr.shape[0] - 1, arr.shape[1] - 1),
                }
                for row, col in positions
            )
        ),
    }


def _support_material(board: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(board, dtype=np.int32)
    return {
        "raw_count_384": int(np.count_nonzero(arr == 384)),
        "raw_count_192": int(np.count_nonzero(arr == 192)),
        "raw_count_96": int(np.count_nonzero(arr == 96)),
        "raw_count_48": int(np.count_nonzero(arr == 48)),
        "raw_has_adjacent_384": _has_adjacent_pair(arr, 384),
        "raw_has_adjacent_192": _has_adjacent_pair(arr, 192),
        "raw_has_adjacent_96": _has_adjacent_pair(arr, 96),
        "raw_has_adjacent_48": _has_adjacent_pair(arr, 48),
    }


def _raw(state: SimState, starter_tile: int | None) -> dict[str, Any]:
    raw = raw_ladder_features(state.board, starter_tile)
    return {
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_count_3072": int(raw["raw_count_3072"]),
        "masked_count_1536": int(raw["masked_count_1536"]),
        "masked_count_3072": int(raw["masked_count_3072"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "masked_highest_duplicate_tile": int(raw["masked_highest_duplicate_tile"]),
        "masked_highest_adjacent_pair_tile": int(raw["masked_highest_adjacent_pair_tile"]),
        "empty_count": int(np.count_nonzero(np.asarray(state.board, dtype=np.int32) == 0)),
        **_support_material(state.board),
        **_support_geometry(state.board),
    }


def _target_hit(state: SimState, starter_tile: int | None, target: str) -> bool:
    raw = _raw(state, starter_tile)
    if target == "raw_duplicate_1536":
        return int(raw["masked_count_1536"]) >= 1 or int(raw["raw_count_1536"]) >= 2
    if int(raw["masked_count_1536"]) != 0:
        return False
    if target == "raw_one_192_with_one_768_no_1536":
        return int(raw["raw_count_768"]) == 1 and int(raw["raw_count_192"]) >= 1
    if target == "raw_one_384_with_one_768_no_1536":
        return int(raw["raw_count_768"]) == 1 and int(raw["raw_count_384"]) >= 1
    if target == "raw_two_384_with_one_768_no_1536":
        return int(raw["raw_count_768"]) == 1 and int(raw["raw_count_384"]) >= 2
    if target == "raw_adjacent_384_with_one_768_no_1536":
        return (
            int(raw["raw_count_768"]) == 1
            and int(raw["raw_count_384"]) >= 2
            and bool(raw["raw_has_adjacent_384"])
        )
    if target == "raw_two_768_no_1536":
        return int(raw["raw_count_768"]) >= 2
    if target == "raw_adjacent_768_no_1536":
        return bool(raw["raw_has_adjacent_768"])
    if target == "raw_three_768_no_1536":
        return int(raw["raw_count_768"]) >= 3
    if target == "raw_four_768_no_1536":
        return int(raw["raw_count_768"]) >= 4
    if target == "raw_four_adjacent_768_no_1536":
        return int(raw["raw_count_768"]) >= 4 and bool(raw["raw_has_adjacent_768"])
    raise ValueError(f"Unsupported target: {target}")


def _node_id(case: FrontierCase, state: SimState, *, depth: int, path: list[str]) -> str:
    return safe_name(
        f"accum_{case.id}_d{int(depth)}_{'_'.join(path) or 'start'}_{_state_digest(state)}",
        max_length=160,
    )


def _start_node(case: FrontierCase) -> AccumulationNode:
    state = copy_state(case.state)
    return AccumulationNode(
        case=case,
        state=state,
        depth=0,
        path=[],
        seed_path=[],
        start_score=int(score_board(state.board)),
        score_delta=0,
        parent_id=None,
        node_id=_node_id(case, state, depth=0, path=[]),
    )


def _archive_key(node: AccumulationNode) -> tuple[str, int, int, int, int, int]:
    raw = _raw(node.state, node.case.starter_tile)
    sim = ThreesSim(np.random.default_rng(0), starter_tile=node.case.starter_tile)
    return (
        str(node.case.ancestry_key or node.case.id),
        int(raw["raw_count_768"]),
        int(bool(raw["raw_has_adjacent_768"])),
        int(raw["empty_count"]) // 2,
        len(sim.legal_actions(node.state)),
        int(node.depth),
    )


def _score_node(node: AccumulationNode, *, rank_profile: str = "default") -> tuple[float, ...]:
    if rank_profile not in SUPPORTED_RANK_PROFILES:
        raise ValueError(f"Unsupported rank profile: {rank_profile}")
    raw = _raw(node.state, node.case.starter_tile)
    sim = ThreesSim(np.random.default_rng(0), starter_tile=node.case.starter_tile)
    if rank_profile == "buildable":
        return (
            float(raw["raw_count_768"]),
            float(raw["raw_768_components"]),
            -float(raw["raw_768_adjacent_pairs"]),
            float(raw["raw_768_air_neighbors"]) / 8.0,
            float(raw["empty_count"]),
            float(len(sim.legal_actions(node.state))),
            -float(raw["raw_768_max_component"]),
            float(node.depth) / 20.0,
            float(node.score_delta) / 200000.0,
        )
    if rank_profile == "bridge":
        raw_count = int(raw["raw_count_768"])
        has_adjacent = bool(raw["raw_has_adjacent_768"])
        bridge_level = float(raw_count)
        if raw_count >= 3 and has_adjacent:
            bridge_level += 1.5
        elif raw_count >= 4:
            bridge_level += 0.25
        return (
            bridge_level,
            float(has_adjacent),
            float(raw["raw_768_air_neighbors"]) / 8.0,
            float(len(sim.legal_actions(node.state))) / 4.0,
            float(raw["empty_count"]),
            -float(max(0, int(raw["raw_768_max_component"]) - 2)),
            -float(max(0, int(raw["raw_768_adjacent_pairs"]) - 1)),
            float(node.depth) / 20.0,
            float(node.score_delta) / 200000.0,
        )
    if rank_profile == "second768":
        sim = ThreesSim(np.random.default_rng(0), starter_tile=node.case.starter_tile)
        highest_adjacent_support = min(int(raw["raw_highest_adjacent_pair_tile"]), 384)
        highest_duplicate_support = min(int(raw["raw_highest_duplicate_tile"]), 384)
        return (
            float(raw["raw_count_768"]),
            float(bool(raw["raw_has_adjacent_768"])),
            float(highest_adjacent_support) / 384.0,
            float(highest_duplicate_support) / 384.0,
            float(raw["raw_count_384"]) / 4.0,
            float(raw["raw_count_192"]) / 4.0,
            float(bool(raw["raw_has_adjacent_384"])),
            float(bool(raw["raw_has_adjacent_192"])),
            float(len(sim.legal_actions(node.state))) / 4.0,
            float(raw["empty_count"]) / 8.0,
            float(node.depth) / 20.0,
            float(node.score_delta) / 200000.0,
        )
    if rank_profile == "supportstock":
        sim = ThreesSim(np.random.default_rng(0), starter_tile=node.case.starter_tile)
        highest_duplicate_support = min(int(raw["raw_highest_duplicate_tile"]), 384)
        highest_adjacent_support = min(int(raw["raw_highest_adjacent_pair_tile"]), 384)
        material_score = (
            float(raw["raw_count_384"])
            + 0.50 * float(raw["raw_count_192"])
            + 0.25 * float(raw["raw_count_96"])
            + 0.125 * float(raw["raw_count_48"])
        )
        adjacent_stock = (
            float(bool(raw["raw_has_adjacent_384"]))
            + 0.50 * float(bool(raw["raw_has_adjacent_192"]))
            + 0.25 * float(bool(raw["raw_has_adjacent_96"]))
            + 0.125 * float(bool(raw["raw_has_adjacent_48"]))
        )
        return (
            float(raw["raw_count_768"]),
            float(material_score) / 4.0,
            float(highest_duplicate_support) / 384.0,
            float(highest_adjacent_support) / 384.0,
            float(adjacent_stock) / 2.0,
            float(len(sim.legal_actions(node.state))) / 4.0,
            float(raw["empty_count"]) / 8.0,
            -float(max(0, int(raw["raw_768_max_component"]) - 1)),
            float(node.depth) / 20.0,
            float(node.score_delta) / 200000.0,
        )
    return (
        float(raw["raw_count_768"]),
        float(raw["empty_count"]),
        float(len(sim.legal_actions(node.state))),
        float(bool(raw["raw_has_adjacent_768"])),
        float(node.depth),
        float(node.score_delta) / 200000.0,
    )


def _root_balance_key(node: AccumulationNode) -> str:
    return str(node.case.root_seed or node.case.ancestry_key or node.case.root_replay or node.case.id)


def _append_cell_capped(
    selected: list[AccumulationNode],
    grouped: dict[tuple[str, int, int, int, int, int], list[AccumulationNode]],
    seen_ids: set[str],
    candidates: list[AccumulationNode],
    *,
    max_nodes: int,
    max_per_cell: int,
    rank_profile: str,
) -> None:
    for node in sorted(candidates, key=lambda node: _score_node(node, rank_profile=rank_profile), reverse=True):
        if len(selected) >= int(max_nodes):
            return
        if node.node_id in seen_ids:
            continue
        key = _archive_key(node)
        if len(grouped[key]) >= int(max_per_cell):
            continue
        grouped[key].append(node)
        seen_ids.add(node.node_id)
        selected.append(node)


def select_archive_nodes(
    candidates: list[AccumulationNode],
    *,
    max_nodes: int,
    max_per_cell: int,
    balance_by_root: bool = False,
    rank_profile: str = "default",
) -> list[AccumulationNode]:
    if not candidates:
        return []
    max_nodes = max(0, int(max_nodes))
    if max_nodes <= 0:
        return []
    grouped: dict[tuple[str, int, int, int, int, int], list[AccumulationNode]] = defaultdict(list)
    seen_ids: set[str] = set()
    selected: list[AccumulationNode] = []
    if balance_by_root:
        by_root: dict[str, list[AccumulationNode]] = defaultdict(list)
        for node in candidates:
            by_root[_root_balance_key(node)].append(node)
        root_order = sorted(
            by_root,
            key=lambda root: _score_node(
                max(by_root[root], key=lambda node: _score_node(node, rank_profile=rank_profile)),
                rank_profile=rank_profile,
            ),
            reverse=True,
        )
        per_root = max(1, max_nodes // max(1, len(root_order)))
        for root in root_order:
            _append_cell_capped(
                selected,
                grouped,
                seen_ids,
                by_root[root],
                max_nodes=min(max_nodes, len(selected) + per_root),
                max_per_cell=max_per_cell,
                rank_profile=rank_profile,
            )
            if len(selected) >= max_nodes:
                break
    _append_cell_capped(
        selected,
        grouped,
        seen_ids,
        candidates,
        max_nodes=max_nodes,
        max_per_cell=max_per_cell,
        rank_profile=rank_profile,
    )
    selected.sort(key=lambda node: _score_node(node, rank_profile=rank_profile), reverse=True)
    return selected[:max_nodes]


def _record_for_node(node: AccumulationNode, *, rank_profile: str = "default") -> dict[str, Any]:
    sim = ThreesSim(np.random.default_rng(node.case.source_seed or 0), starter_tile=node.case.starter_tile)
    raw = _raw(node.state, node.case.starter_tile)
    legal = sim.legal_actions(node.state)
    return {
        "id": node.node_id,
        "kind": "support_accumulation_frontier_state",
        "state": state_payload(node.state, sim),
        "starter_tile": node.case.starter_tile,
        "source_replay": node.case.source_replay,
        "source_seed": node.case.source_seed,
        "source_frame_index": node.case.source_frame_index,
        "source_policy": node.case.source_policy,
        "root_origin": node.case.root_origin,
        "root_replay": node.case.root_replay,
        "root_seed": node.case.root_seed,
        "root_frame_index": node.case.root_frame_index,
        "root_policy": node.case.root_policy,
        "root_policy_family": node.case.root_policy_family,
        "ancestry_key": node.case.ancestry_key,
        "score": int(score_board(node.state.board)),
        "score_delta": int(node.score_delta),
        "move_count": int(node.state.move_count),
        "features": {
            **raw,
            "depth": int(node.depth),
            "legal_count": int(len(legal)),
            "score_delta": int(node.score_delta),
            "path_length": int(len(node.path)),
        },
        "frontier": {
            "parent_id": node.parent_id,
            "depth": int(node.depth),
            "path": list(node.path),
            "seed_path": [int(seed) for seed in node.seed_path],
            "archive_key": list(_archive_key(node)),
            "rank_profile": str(rank_profile),
            "rank_tuple": list(_score_node(node, rank_profile=rank_profile)),
        },
    }


def _step_seed(base_seed: int, *, depth: int, node_index: int, action: int, repeat: int) -> int:
    return int(base_seed) + int(depth) * 1_000_003 + int(node_index) * 10_007 + int(action) * 503 + int(repeat) * 97


def expand_frontier(
    starts: list[FrontierCase],
    *,
    target: str,
    max_depth: int,
    repeats_per_action: int,
    min_raw_768: int,
    max_nodes_per_depth: int,
    max_per_cell: int,
    balance_by_root: bool,
    rank_profile: str,
    seed: int,
    progress_every: int,
    saturate_target_roots: bool = False,
) -> tuple[list[AccumulationNode], list[AccumulationNode], list[dict[str, Any]], list[dict[str, Any]]]:
    frontier = [_start_node(case) for case in starts]
    archive: list[AccumulationNode] = list(frontier)
    target_nodes: list[AccumulationNode] = [
        node for node in frontier if _target_hit(node.state, node.case.starter_tile, target)
    ]
    saturated_roots: set[str] = set()
    if saturate_target_roots:
        saturated_roots.update(_root_balance_key(node) for node in target_nodes)
    depth_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    expanded = 0
    for depth in range(1, int(max_depth) + 1):
        candidates: list[AccumulationNode] = []
        stats: Counter[str] = Counter()
        for node_index, node in enumerate(frontier):
            root_key = _root_balance_key(node)
            if saturate_target_roots and root_key in saturated_roots:
                stats["saturated_root"] += 1
                continue
            sim_for_legal = ThreesSim(np.random.default_rng(seed + depth + node_index), starter_tile=node.case.starter_tile)
            legal_actions = sim_for_legal.legal_actions(node.state)
            if not legal_actions:
                stats["no_legal"] += 1
                continue
            for action in legal_actions:
                if saturate_target_roots and root_key in saturated_roots:
                    break
                for repeat in range(int(repeats_per_action)):
                    if saturate_target_roots and root_key in saturated_roots:
                        break
                    step_seed = _step_seed(seed, depth=depth, node_index=node_index, action=int(action), repeat=repeat)
                    sim = ThreesSim(np.random.default_rng(step_seed), starter_tile=node.case.starter_tile)
                    next_state, info = sim.step(node.state, int(action))
                    expanded += 1
                    if not info.moved:
                        stats["not_moved"] += 1
                        continue
                    raw = _raw(next_state, node.case.starter_tile)
                    hit = _target_hit(next_state, node.case.starter_tile, target)
                    if next_state.game_over:
                        outcome = "terminal"
                    elif hit:
                        outcome = "target"
                    elif int(raw["masked_count_1536"]) >= 1:
                        outcome = "converted_to_1536"
                    elif int(raw["raw_count_768"]) < int(min_raw_768):
                        outcome = "below_min_768"
                    else:
                        outcome = "archived"
                    stats[outcome] += 1
                    transitions.append(
                        {
                            "parent_id": node.node_id,
                            "case_id": node.case.id,
                            "root_key": root_key,
                            "depth": int(depth),
                            "action": DIRECTION_NAMES[int(action)],
                            "repeat_index": int(repeat),
                            "seed": int(step_seed),
                            "outcome": outcome,
                            "score_delta": int(score_board(next_state.board) - node.start_score),
                            **raw,
                        }
                    )
                    if outcome in {"archived", "target"}:
                        path = node.path + [DIRECTION_NAMES[int(action)]]
                        seed_path = node.seed_path + [int(step_seed)]
                        child = AccumulationNode(
                            case=node.case,
                            state=copy_state(next_state),
                            depth=int(depth),
                            path=path,
                            seed_path=seed_path,
                            start_score=node.start_score,
                            score_delta=int(score_board(next_state.board) - node.start_score),
                            parent_id=node.node_id,
                            node_id=_node_id(node.case, next_state, depth=depth, path=path),
                        )
                        if outcome == "target":
                            target_nodes.append(child)
                            if saturate_target_roots:
                                saturated_roots.add(root_key)
                        if not (saturate_target_roots and outcome == "target"):
                            candidates.append(child)
            if progress_every > 0 and expanded % int(progress_every) == 0:
                print(
                    "accum_frontier "
                    f"depth={depth} expanded={expanded} "
                    f"candidates={len(candidates)} "
                    f"targets={stats['target']} "
                    f"converted={stats['converted_to_1536']} "
                    f"saturated_roots={len(saturated_roots)}",
                    flush=True,
                )
        selected = select_archive_nodes(
            candidates,
            max_nodes=max_nodes_per_depth,
            max_per_cell=max_per_cell,
            balance_by_root=balance_by_root,
            rank_profile=rank_profile,
        )
        archive.extend(selected)
        depth_rows.append(
            {
                "depth": int(depth),
                "frontier_in": len(frontier),
                "candidate_archived": len(candidates),
                "frontier_out": len(selected),
                "frontier_out_root_seeds": len({node.case.root_seed for node in selected if node.case.root_seed is not None}),
                **{key: int(value) for key, value in sorted(stats.items())},
            }
        )
        frontier = selected
        if not frontier:
            break
    unique_targets: dict[str, AccumulationNode] = {}
    for node in sorted(target_nodes, key=lambda node: _score_node(node, rank_profile=rank_profile), reverse=True):
        unique_targets.setdefault(node.node_id, node)
    return archive, list(unique_targets.values()), depth_rows, transitions


def summarize(
    *,
    source_records: int,
    cases_total: int,
    selected_cases: list[FrontierCase],
    rejected: dict[str, int],
    archive: list[AccumulationNode],
    target_nodes: list[AccumulationNode],
    depth_rows: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    target: str,
    max_depth: int,
    repeats_per_action: int,
    min_raw_768: int,
    max_nodes_per_depth: int,
    max_per_cell: int,
    balance_by_root: bool,
    rank_profile: str,
    saturate_target_roots: bool,
) -> dict[str, Any]:
    deepest = max((node.depth for node in archive), default=0)
    archived_by_depth = Counter(int(node.depth) for node in archive)
    target_by_depth = Counter(int(node.depth) for node in target_nodes)
    archive_raw_counts = Counter(int(_raw(node.state, node.case.starter_tile)["raw_count_768"]) for node in archive)
    archive_root_seeds = Counter(str(node.case.root_seed) for node in archive if node.case.root_seed is not None)
    target_root_seeds = Counter(str(node.case.root_seed) for node in target_nodes if node.case.root_seed is not None)
    outcomes = Counter(str(row.get("outcome")) for row in transitions)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target": str(target),
        "source_records": int(source_records),
        "cases_total": int(cases_total),
        "cases_selected": len(selected_cases),
        "rejected": rejected,
        "max_depth": int(max_depth),
        "repeats_per_action": int(repeats_per_action),
        "min_raw_768": int(min_raw_768),
        "max_nodes_per_depth": int(max_nodes_per_depth),
        "max_per_cell": int(max_per_cell),
        "balance_by_root": bool(balance_by_root),
        "rank_profile": str(rank_profile),
        "saturate_target_roots": bool(saturate_target_roots),
        "archive_records": len(archive),
        "target_records": len(target_nodes),
        "deepest_archived_depth": int(deepest),
        "archived_by_depth": {str(key): int(value) for key, value in sorted(archived_by_depth.items())},
        "target_by_depth": {str(key): int(value) for key, value in sorted(target_by_depth.items())},
        "archive_raw_count_768": {str(key): int(value) for key, value in sorted(archive_raw_counts.items())},
        "archive_root_seeds": {str(key): int(value) for key, value in sorted(archive_root_seeds.items())},
        "target_root_seeds": {str(key): int(value) for key, value in sorted(target_root_seeds.items())},
        "target_saturated_root_seeds": sorted(str(key) for key in target_root_seeds) if saturate_target_roots else [],
        "transitions": len(transitions),
        "transition_outcomes": dict(outcomes),
        "converted_to_1536_transitions": int(outcomes.get("converted_to_1536", 0)),
        "below_min_768_transitions": int(outcomes.get("below_min_768", 0)),
        "terminal_transitions": int(outcomes.get("terminal", 0)),
        "unique_root_seeds": len({case.root_seed for case in selected_cases if case.root_seed is not None}),
        "target_unique_root_seeds": len({node.case.root_seed for node in target_nodes if node.case.root_seed is not None}),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    depth_rows = payload.get("depth_rows", [])
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    depth_html = "".join(
        "<tr>"
        f"<td>{cell(row.get('depth'))}</td>"
        f"<td>{cell(row.get('frontier_in'))}</td>"
        f"<td>{cell(row.get('candidate_archived'))}</td>"
        f"<td>{cell(row.get('frontier_out'))}</td>"
        f"<td>{cell(row.get('target', 0))}</td>"
        f"<td>{cell(row.get('converted_to_1536', 0))}</td>"
        f"<td>{cell(row.get('below_min_768', 0))}</td>"
        f"<td>{cell(row.get('terminal', 0))}</td>"
        "</tr>"
        for row in depth_rows
    )
    top_records = sorted(
        [record for record in records if isinstance(record, dict)],
        key=lambda row: tuple(row.get("frontier", {}).get("rank_tuple", [])),
        reverse=True,
    )[:80]
    record_html = "".join(
        "<tr>"
        f"<td>{cell((row.get('features') or {}).get('depth'))}</td>"
        f"<td>{cell(row.get('root_seed'))}</td>"
        f"<td>{cell(row.get('move_count'))}</td>"
        f"<td>{cell((row.get('features') or {}).get('empty_count'))}</td>"
        f"<td>{cell((row.get('features') or {}).get('raw_count_768'))}</td>"
        f"<td>{cell((row.get('features') or {}).get('raw_has_adjacent_768'))}</td>"
        f"<td>{cell((row.get('features') or {}).get('raw_count_1536'))}</td>"
        f"<td>{cell(' '.join((row.get('frontier') or {}).get('path', [])))}</td>"
        "</tr>"
        for row in top_records
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support Accumulation Frontier</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1180px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card, .panel {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; margin-top:12px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; vertical-align:top; }}
    th:last-child, td:last-child {{ text-align:left; overflow-wrap:anywhere; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; margin:0; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support Accumulation Frontier</h1>
    <p class="muted">Archived prefixes that keep raw 768 material alive while avoiding premature raw 1536 creation.</p>
    <section class="cards">
      <div class="card"><div class="label">Cases</div><div class="value">{cell(summary.get('cases_selected', 0))}</div></div>
      <div class="card"><div class="label">Archive</div><div class="value">{cell(summary.get('archive_records', 0))}</div></div>
      <div class="card"><div class="label">Deepest</div><div class="value">{cell(summary.get('deepest_archived_depth', 0))}</div></div>
      <div class="card"><div class="label">Targets</div><div class="value">{cell(summary.get('target_records', 0))}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top:14px;">
      <h2>Depths</h2>
      <table><thead><tr><th>Depth</th><th>In</th><th>Archived</th><th>Out</th><th>Targets</th><th>1536</th><th>Low 768</th><th>Terminal</th></tr></thead><tbody>{depth_html}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;">
      <h2>Top Archived States</h2>
      <table><thead><tr><th>Depth</th><th>Seed</th><th>Move</th><th>Air</th><th>768s</th><th>Adj</th><th>1536s</th><th>Path</th></tr></thead><tbody>{record_html}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_frontier(
    *,
    records_json: list[Path],
    target: str,
    max_depth: int,
    repeats_per_action: int,
    max_starts: int,
    min_start_raw_768: int,
    min_raw_768: int,
    max_nodes_per_depth: int,
    max_per_cell: int,
    seed: int,
    root_origins: set[str],
    default_starter_tile: int | None,
    require_no_start_1536: bool,
    out_dir: Path,
    progress_every: int = 200,
    balance_by_root: bool = False,
    rank_profile: str = "default",
    saturate_target_roots: bool = False,
    compact_output: bool = False,
) -> dict[str, Any]:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    records = load_records(records_json)
    cases, rejected = load_cases(records, default_starter_tile=default_starter_tile, root_origins=root_origins)
    filtered: list[FrontierCase] = []
    for case in cases:
        if int(case.raw.get("raw_count_768", 0)) < int(min_start_raw_768):
            rejected["below_start_min_raw_768"] = rejected.get("below_start_min_raw_768", 0) + 1
            continue
        if require_no_start_1536 and int(case.raw.get("masked_count_1536", 0)) > 0:
            rejected["start_has_1536"] = rejected.get("start_has_1536", 0) + 1
            continue
        filtered.append(case)
    selected = select_diverse_cases(filtered, max_starts=max_starts, seed=seed)
    if not selected:
        raise ValueError("No start cases matched the requested filters")
    archive, target_nodes, depth_rows, transitions = expand_frontier(
        selected,
        target=target,
        max_depth=max_depth,
        repeats_per_action=repeats_per_action,
        min_raw_768=min_raw_768,
        max_nodes_per_depth=max_nodes_per_depth,
        max_per_cell=max_per_cell,
        balance_by_root=balance_by_root,
        rank_profile=rank_profile,
        seed=seed,
        progress_every=progress_every,
        saturate_target_roots=saturate_target_roots,
    )
    archive_records = [_record_for_node(node, rank_profile=rank_profile) for node in archive]
    target_records = [_record_for_node(node, rank_profile=rank_profile) for node in target_nodes]
    summary = summarize(
        source_records=len(records),
        cases_total=len(filtered),
        selected_cases=selected,
        rejected=rejected,
        archive=archive,
        target_nodes=target_nodes,
        depth_rows=depth_rows,
        transitions=transitions,
        target=target,
        max_depth=max_depth,
        repeats_per_action=repeats_per_action,
        min_raw_768=min_raw_768,
        max_nodes_per_depth=max_nodes_per_depth,
        max_per_cell=max_per_cell,
        balance_by_root=balance_by_root,
        rank_profile=rank_profile,
        saturate_target_roots=saturate_target_roots,
    )
    payload = {
        "version": 1,
        "kind": "support_accumulation_frontier",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "compact_output": bool(compact_output),
        "records_json": [str(path) for path in records_json],
        "summary": summary,
        "depth_rows": depth_rows,
        "transitions": [] if compact_output else transitions,
        "records": [] if compact_output else archive_records,
        "target_records": target_records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "support_accumulation_frontier.json")
    payload["records_json"] = None if compact_output else str(out_dir / "records.json")
    payload["summary_json"] = str(out_dir / "summary.json")
    payload["transitions_json"] = None if compact_output else str(out_dir / "transitions.json")
    payload["target_records_json"] = str(out_dir / "target_records.json")
    payload["html"] = str(out_dir / "support_accumulation_frontier.html")
    write_json(out_dir / "support_accumulation_frontier.json", payload)
    if not compact_output:
        write_json(out_dir / "records.json", archive_records)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "depth_rows.json", depth_rows)
    if not compact_output:
        write_json(out_dir / "transitions.json", transitions)
    write_json(out_dir / "target_records.json", target_records)
    write_html(out_dir / "support_accumulation_frontier.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--target", choices=SUPPORTED_TARGETS, default="raw_three_768_no_1536")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--repeats-per-action", type=int, default=4)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--min-start-raw-768", type=int, default=2)
    parser.add_argument("--min-raw-768", type=int, default=2)
    parser.add_argument("--max-nodes-per-depth", type=int, default=256)
    parser.add_argument("--max-per-cell", type=int, default=3)
    parser.add_argument("--balance-by-root", action="store_true")
    parser.add_argument(
        "--saturate-target-roots",
        action="store_true",
        help="Stop expanding a root after its first target hit; useful for root-conversion acquisition screens.",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Write summaries and target states without full archive records or transition logs.",
    )
    parser.add_argument("--rank-profile", choices=SUPPORTED_RANK_PROFILES, default="default")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--allow-start-1536", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("threes_rl/runs/forensics/support_accumulation_frontier/latest"),
    )
    args = parser.parse_args()
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    payload = run_frontier(
        records_json=args.records_json,
        target=args.target,
        max_depth=args.max_depth,
        repeats_per_action=args.repeats_per_action,
        max_starts=args.max_starts,
        min_start_raw_768=args.min_start_raw_768,
        min_raw_768=args.min_raw_768,
        max_nodes_per_depth=args.max_nodes_per_depth,
        max_per_cell=args.max_per_cell,
        balance_by_root=bool(args.balance_by_root),
        rank_profile=args.rank_profile,
        seed=args.seed,
        root_origins=parse_root_origins(args.root_origin),
        default_starter_tile=default_starter,
        require_no_start_1536=not bool(args.allow_start_1536),
        out_dir=args.out_dir,
        progress_every=args.progress_every,
        saturate_target_roots=bool(args.saturate_target_roots),
        compact_output=bool(args.compact_output),
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()

"""Branch/archive frontier for the first-3072 to duplicate-1536 gap."""

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
from threes_rl.support_ladder_window_reservoir import milestone_specs, raw_ladder_features
from threes_rl.swing_label import state_features
from threes_rl.train_td import copy_state

SUPPORTED_TARGETS = (
    "raw_duplicate_768",
    "raw_adjacent_768",
    "raw_three_768_no_1536",
    "raw_four_768_no_1536",
    "raw_four_adjacent_768_no_1536",
    "raw_one_1536",
    "raw_adjacent_768_with_1536",
    "raw_duplicate_1536",
    "raw_near_adjacent_1536",
    "raw_adjacent_1536",
    "second_3072",
)
SUPPORTED_RANK_PROFILES = ("support", "near1536", "regen768")


@dataclass
class Post3072Node:
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


def _support_geometry(board: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(board, dtype=np.int32)
    positions = _tile_positions(arr, 768)
    position_set = set(positions)
    adjacent_pairs = 0
    air_neighbors: set[tuple[int, int]] = set()
    component_sizes: list[int] = []
    visited: set[tuple[int, int]] = set()
    for row, col in positions:
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nrow = row + drow
            ncol = col + dcol
            if not (0 <= nrow < 4 and 0 <= ncol < 4):
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
    }


def _has_adjacent_pair(board: np.ndarray, value: int) -> bool:
    positions = set(_tile_positions(board, value))
    for row, col in positions:
        if (row + 1, col) in positions or (row, col + 1) in positions:
            return True
    return False


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
        "raw_count_6144": int(raw["raw_count_6144"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "raw_has_diagonal_touch_1536": bool(raw["raw_has_diagonal_touch_1536"]),
        "raw_has_near_adjacent_1536": bool(raw["raw_has_near_adjacent_1536"]),
        "raw_min_pair_distance_1536": raw["raw_min_pair_distance_1536"],
        "raw_min_pair_chebyshev_1536": raw["raw_min_pair_chebyshev_1536"],
        "masked_max_tile_excl_starter": int(raw["masked_max_tile_excl_starter"]),
        "masked_count_3072": int(raw["masked_count_3072"]),
        "empty_count": int(np.count_nonzero(np.asarray(state.board, dtype=np.int32) == 0)),
        **_support_geometry(state.board),
        **_support_material(state.board),
    }


def _target_hit(state: SimState, starter_tile: int | None, target: str) -> bool:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    return bool(milestone_specs()[target].predicate(raw_ladder_features(state.board, starter_tile)))


def _first_3072_ready(case: FrontierCase) -> bool:
    return int(case.raw.get("raw_count_3072", 0)) >= 1 or int(case.raw.get("masked_max_tile_excl_starter", 0)) >= 3072


def _node_id(case: FrontierCase, state: SimState, *, depth: int, path: list[str]) -> str:
    return safe_name(f"post3072_{case.id}_d{int(depth)}_{'_'.join(path) or 'start'}_{_state_digest(state)}", max_length=160)


def _start_node(case: FrontierCase) -> Post3072Node:
    state = copy_state(case.state)
    return Post3072Node(
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


def _archive_key(node: Post3072Node) -> tuple[str, int, int, int, int, int, int]:
    raw = _raw(node.state, node.case.starter_tile)
    sim = ThreesSim(np.random.default_rng(0), starter_tile=node.case.starter_tile)
    return (
        str(node.case.ancestry_key or node.case.root_seed or node.case.id),
        int(raw["raw_count_1536"]),
        int(raw["raw_count_768"]),
        int(bool(raw["raw_has_adjacent_768"])),
        int(bool(raw["raw_has_near_adjacent_1536"])),
        int(raw["empty_count"]) // 2,
        len(sim.legal_actions(node.state)),
    )


def _distance_score(raw: dict[str, Any]) -> float:
    cheb = raw.get("raw_min_pair_chebyshev_1536")
    if cheb is None:
        return 0.0
    return -float(cheb)


def _score_node(node: Post3072Node, *, rank_profile: str = "support") -> tuple[float, ...]:
    if rank_profile not in SUPPORTED_RANK_PROFILES:
        raise ValueError(f"Unsupported rank profile: {rank_profile}")
    raw = _raw(node.state, node.case.starter_tile)
    sim = ThreesSim(np.random.default_rng(0), starter_tile=node.case.starter_tile)
    legal_count = len(sim.legal_actions(node.state))
    base = (
        float(raw["raw_count_1536"]) * 10.0,
        float(raw["raw_count_768"]),
        float(bool(raw["raw_has_adjacent_768"])),
        float(raw["raw_768_air_neighbors"]) / 8.0,
        float(raw["empty_count"]),
        float(legal_count),
        float(node.score_delta) / 200000.0,
    )
    if rank_profile == "near1536":
        return (
            float(bool(raw["raw_has_adjacent_1536"])) * 40.0,
            float(bool(raw["raw_has_near_adjacent_1536"])) * 20.0,
            float(raw["raw_count_1536"]) * 10.0,
            _distance_score(raw),
            float(raw["raw_count_768"]),
            float(bool(raw["raw_has_adjacent_768"])),
            float(raw["empty_count"]),
            float(legal_count),
            float(node.score_delta) / 200000.0,
        )
    if rank_profile == "regen768":
        support_stock = (
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
        duplicate_support = min(float(raw["raw_highest_duplicate_tile"]), 384.0) / 384.0
        adjacent_support = min(float(raw["raw_highest_adjacent_pair_tile"]), 384.0) / 384.0
        return (
            float(raw["raw_count_768"]),
            float(bool(raw["raw_has_adjacent_768"])),
            support_stock / 4.0,
            adjacent_stock / 2.0,
            duplicate_support,
            adjacent_support,
            float(raw["raw_768_air_neighbors"]) / 8.0,
            float(raw["empty_count"]),
            float(legal_count),
            -float(max(0, int(raw["raw_count_1536"]) - 1)),
            float(node.score_delta) / 200000.0,
        )
    return base


def select_archive_nodes(
    candidates: list[Post3072Node],
    *,
    max_nodes: int,
    max_per_cell: int,
    balance_by_root: bool = False,
    rank_profile: str = "support",
) -> list[Post3072Node]:
    if not candidates or max_nodes <= 0:
        return []
    grouped: dict[tuple[str, int, int, int, int, int, int], list[Post3072Node]] = defaultdict(list)
    seen_ids: set[str] = set()
    selected: list[Post3072Node] = []

    def append_from(nodes: list[Post3072Node], limit: int) -> None:
        for node in sorted(nodes, key=lambda item: _score_node(item, rank_profile=rank_profile), reverse=True):
            if len(selected) >= limit:
                return
            if node.node_id in seen_ids:
                continue
            key = _archive_key(node)
            if len(grouped[key]) >= int(max_per_cell):
                continue
            grouped[key].append(node)
            seen_ids.add(node.node_id)
            selected.append(node)

    if balance_by_root:
        by_root: dict[str, list[Post3072Node]] = defaultdict(list)
        for node in candidates:
            by_root[str(node.case.root_seed or node.case.ancestry_key or node.case.id)].append(node)
        per_root = max(1, int(max_nodes) // max(1, len(by_root)))
        for root in sorted(
            by_root,
            key=lambda key: _score_node(max(by_root[key], key=lambda item: _score_node(item, rank_profile=rank_profile)), rank_profile=rank_profile),
            reverse=True,
        ):
            append_from(by_root[root], min(int(max_nodes), len(selected) + per_root))
    append_from(candidates, int(max_nodes))
    selected.sort(key=lambda item: _score_node(item, rank_profile=rank_profile), reverse=True)
    return selected[: int(max_nodes)]


def _root_balance_key(node: Post3072Node) -> str:
    return str(node.case.root_seed or node.case.ancestry_key or node.case.root_replay or node.case.id)


def _record_for_node(node: Post3072Node, *, rank_profile: str) -> dict[str, Any]:
    sim = ThreesSim(np.random.default_rng(node.case.source_seed or 0), starter_tile=node.case.starter_tile)
    raw = _raw(node.state, node.case.starter_tile)
    features = state_features(node.state, sim, node.case.starter_tile)
    legal = sim.legal_actions(node.state)
    return {
        "id": node.node_id,
        "kind": "post3072_frontier_state",
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
        "max_tile_excl_starter": int(features["max_tile_excl_starter"]),
        "features": {
            **features,
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
) -> tuple[list[Post3072Node], list[Post3072Node], list[dict[str, Any]], list[dict[str, Any]]]:
    frontier = [_start_node(case) for case in starts]
    archive: list[Post3072Node] = list(frontier)
    target_nodes = [node for node in frontier if _target_hit(node.state, node.case.starter_tile, target)]
    saturated_roots: set[str] = set()
    if saturate_target_roots:
        saturated_roots.update(_root_balance_key(node) for node in target_nodes)
    depth_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    expanded = 0
    for depth in range(1, int(max_depth) + 1):
        candidates: list[Post3072Node] = []
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
                    elif int(raw["raw_count_3072"]) < 1 and int(raw["masked_max_tile_excl_starter"]) < 3072:
                        outcome = "lost_first_3072"
                    elif int(raw["raw_count_768"]) < int(min_raw_768) and int(raw["raw_count_1536"]) < 2:
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
                        child = Post3072Node(
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
                        candidates.append(child)
                        if outcome == "target":
                            target_nodes.append(child)
                            if saturate_target_roots:
                                saturated_roots.add(root_key)
            if progress_every > 0 and expanded % int(progress_every) == 0:
                print(
                    "post3072_frontier "
                    f"depth={depth} expanded={expanded} candidates={len(candidates)} "
                    f"targets={stats['target']} saturated_roots={len(saturated_roots)}",
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
    unique_targets: dict[str, Post3072Node] = {}
    for node in sorted(target_nodes, key=lambda item: _score_node(item, rank_profile=rank_profile), reverse=True):
        unique_targets.setdefault(node.node_id, node)
    return archive, list(unique_targets.values()), depth_rows, transitions


def summarize(
    *,
    source_records: int,
    cases_total: int,
    selected_cases: list[FrontierCase],
    rejected: dict[str, int],
    archive: list[Post3072Node],
    target_nodes: list[Post3072Node],
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
    archive_raw = [_raw(node.state, node.case.starter_tile) for node in archive]
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
        "deepest_archived_depth": int(max((node.depth for node in archive), default=0)),
        "archived_by_depth": {str(key): int(value) for key, value in sorted(Counter(int(node.depth) for node in archive).items())},
        "target_by_depth": {str(key): int(value) for key, value in sorted(Counter(int(node.depth) for node in target_nodes).items())},
        "archive_raw_count_768": {str(key): int(value) for key, value in sorted(Counter(int(raw["raw_count_768"]) for raw in archive_raw).items())},
        "archive_raw_count_1536": {str(key): int(value) for key, value in sorted(Counter(int(raw["raw_count_1536"]) for raw in archive_raw).items())},
        "archive_near_adjacent_1536": int(sum(1 for raw in archive_raw if raw["raw_has_near_adjacent_1536"])),
        "archive_adjacent_1536": int(sum(1 for raw in archive_raw if raw["raw_has_adjacent_1536"])),
        "archive_root_seeds": {
            str(key): int(value)
            for key, value in sorted(Counter(str(node.case.root_seed) for node in archive if node.case.root_seed is not None).items())
        },
        "target_root_seeds": {
            str(key): int(value)
            for key, value in sorted(Counter(str(node.case.root_seed) for node in target_nodes if node.case.root_seed is not None).items())
        },
        "target_saturated_root_seeds": sorted(
            str(key) for key in {node.case.root_seed for node in target_nodes if node.case.root_seed is not None}
        )
        if saturate_target_roots
        else [],
        "transitions": len(transitions),
        "transition_outcomes": dict(outcomes),
        "terminal_transitions": int(outcomes.get("terminal", 0)),
        "below_min_768_transitions": int(outcomes.get("below_min_768", 0)),
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
        f"<td>{cell((row.get('features') or {}).get('raw_count_1536'))}</td>"
        f"<td>{cell((row.get('features') or {}).get('raw_has_near_adjacent_1536'))}</td>"
        f"<td>{cell(' '.join((row.get('frontier') or {}).get('path', [])))}</td>"
        "</tr>"
        for row in top_records
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Post-3072 Frontier</title>
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
    <h1>Post-3072 Frontier</h1>
    <p class="muted">Archived simulator-valid descendants after first 3072, ranked for duplicate/near-adjacent 1536 support.</p>
    <section class="cards">
      <div class="card"><div class="label">Cases</div><div class="value">{cell(summary.get('cases_selected', 0))}</div></div>
      <div class="card"><div class="label">Archive</div><div class="value">{cell(summary.get('archive_records', 0))}</div></div>
      <div class="card"><div class="label">Deepest</div><div class="value">{cell(summary.get('deepest_archived_depth', 0))}</div></div>
      <div class="card"><div class="label">Targets</div><div class="value">{cell(summary.get('target_records', 0))}</div></div>
    </section>
    <section class="panel"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
    <section class="panel" style="margin-top:14px;">
      <h2>Depths</h2>
      <table><thead><tr><th>Depth</th><th>In</th><th>Candidates</th><th>Out</th><th>Targets</th><th>Low 768</th><th>Terminal</th></tr></thead><tbody>{depth_html}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;">
      <h2>Top Archived States</h2>
      <table><thead><tr><th>Depth</th><th>Seed</th><th>Move</th><th>Air</th><th>768s</th><th>1536s</th><th>Near</th><th>Path</th></tr></thead><tbody>{record_html}</tbody></table>
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
    out_dir: Path,
    progress_every: int = 200,
    balance_by_root: bool = False,
    rank_profile: str = "support",
    saturate_target_roots: bool = False,
    compact_output: bool = False,
    omit_transitions: bool = False,
) -> dict[str, Any]:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    records = load_records(records_json)
    cases, rejected = load_cases(records, default_starter_tile=default_starter_tile, root_origins=root_origins)
    filtered: list[FrontierCase] = []
    for case in cases:
        if not _first_3072_ready(case):
            rejected["below_first_3072"] = rejected.get("below_first_3072", 0) + 1
            continue
        if int(case.raw.get("raw_count_768", 0)) < int(min_start_raw_768) and not _target_hit(case.state, case.starter_tile, target):
            rejected["below_start_min_raw_768"] = rejected.get("below_start_min_raw_768", 0) + 1
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
        "kind": "post3072_frontier",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "compact_output": bool(compact_output),
        "omit_transitions": bool(omit_transitions or compact_output),
        "records_json": [str(path) for path in records_json],
        "summary": summary,
        "depth_rows": depth_rows,
        "transitions": [] if (compact_output or omit_transitions) else transitions,
        "records": [] if compact_output else archive_records,
        "target_records": target_records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(out_dir / "post3072_frontier.json")
    payload["records_json"] = None if compact_output else str(out_dir / "records.json")
    payload["summary_json"] = str(out_dir / "summary.json")
    payload["transitions_json"] = None if (compact_output or omit_transitions) else str(out_dir / "transitions.json")
    payload["target_records_json"] = str(out_dir / "target_records.json")
    payload["html"] = str(out_dir / "post3072_frontier.html")
    write_json(out_dir / "post3072_frontier.json", payload)
    if not compact_output:
        write_json(out_dir / "records.json", archive_records)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "depth_rows.json", depth_rows)
    if not compact_output and not omit_transitions:
        write_json(out_dir / "transitions.json", transitions)
    write_json(out_dir / "target_records.json", target_records)
    write_html(out_dir / "post3072_frontier.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--target", choices=SUPPORTED_TARGETS, default="raw_duplicate_1536")
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--repeats-per-action", type=int, default=4)
    parser.add_argument("--max-starts", type=int, default=0)
    parser.add_argument("--min-start-raw-768", type=int, default=1)
    parser.add_argument("--min-raw-768", type=int, default=1)
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
    parser.add_argument(
        "--omit-transitions",
        action="store_true",
        help="Write archive/target records but skip the full transition log.",
    )
    parser.add_argument("--rank-profile", choices=SUPPORTED_RANK_PROFILES, default="support")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--root-origin", help="Comma-separated root origins; defaults to fresh,human.")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/post3072_frontier/latest"))
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
        saturate_target_roots=bool(args.saturate_target_roots),
        compact_output=bool(args.compact_output),
        omit_transitions=bool(args.omit_transitions),
        seed=args.seed,
        root_origins=parse_root_origins(args.root_origin),
        default_starter_tile=default_starter,
        out_dir=args.out_dir,
        progress_every=args.progress_every,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"target_records={payload['target_records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()

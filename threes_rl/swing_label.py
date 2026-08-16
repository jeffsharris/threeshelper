"""Find and label high-leverage near-tie action states.

This module is intentionally diagnostic. It keeps the actor fixed, samples
states where another policy disagrees with the frozen actor under a low
action-value margin, then labels only the actor's top two actions with
common-random-number continuations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.eval import make_policy, max_tile_excluding_initial_starter, parse_seed_range, starter_baseline_score
from threes_rl.expectimax import high_tile_geometry_score
from threes_rl.record_replay import state_payload
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import (
    DIRECTION_NAMES,
    SimState,
    ThreesSim,
    direction_index,
    rank_for_value,
    score_board,
    simulate_base_move,
)

POLICY_RNG_OFFSET = 1_000_003
COMPARISON_RNG_OFFSET = 2_000_003
ROLLOUT_POLICY_RNG_OFFSET = 3_000_003
_WORKER_POLICY: object | None = None
PHASE_BUCKETS = ("early_lt384", "mid_384_768", "late_1536", "endgame_3072p")
CORNER_RISK_BUCKETS = ("low_corner_risk", "medium_corner_risk", "high_corner_risk")


def clear_policy_caches(policy: object) -> None:
    for cache_name in (
        "_cache",
        "_action_cache",
        "_afterstate_cache",
        "_post_spawn_cache",
        "_score_cache",
        "_legal_cache",
        "_eval_cache",
    ):
        cache = getattr(policy, cache_name, None)
        if hasattr(cache, "clear"):
            cache.clear()


def _action_value_rows(policy: object, state: SimState, sim: ThreesSim) -> list[dict[str, Any]]:
    if hasattr(policy, "action_values"):
        clear_policy_caches(policy)
        rows = [
            {
                "action": int(action),
                "name": DIRECTION_NAMES[int(action)],
                "value": float(value),
            }
            for action, value in policy.action_values(state, sim)
        ]
        rows.sort(key=lambda row: (-float(row["value"]), str(row["name"])))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows
    if not hasattr(policy, "_action_value"):
        return []
    clear_policy_caches(policy)
    depth = policy._root_depth(state) if hasattr(policy, "_root_depth") else getattr(policy, "depth", 1)
    rows = []
    for action in sim.legal_actions(state):
        rows.append(
            {
                "action": int(action),
                "name": DIRECTION_NAMES[int(action)],
                "value": float(policy._action_value(state, sim, int(action), int(depth))),
            }
        )
    rows.sort(key=lambda row: (-float(row["value"]), str(row["name"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def action_value_cache_key(policy_spec: str, state: SimState, starter_tile: int | None) -> str:
    preview_key = {
        "kind": state.preview.kind,
        "value": state.preview.value,
        "candidates": list(state.preview.candidates),
    }
    cycle_key = {
        "small_counts": {str(key): int(value) for key, value in sorted(state.small_counts.items())},
        "small_pos": int(state.small_pos),
        "small_seen_total": int(state.small_seen_total),
        "span_small_pos": int(state.span_small_pos),
        "large_pending": bool(state.large_pending),
        "max_tile": int(state.max_tile),
    }
    raw = json.dumps(
        {
            "version": 1,
            "policy": policy_spec,
            "starter_tile": starter_tile,
            "board": [int(value) for value in np.asarray(state.board, dtype=np.int32).reshape(-1)],
            "preview": preview_key,
            "cycle": cycle_key,
            "move_count": int(state.move_count),
            "game_over": bool(state.game_over),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _copy_action_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _cached_action_value_rows(
    policy: object,
    policy_spec: str,
    state: SimState,
    sim: ThreesSim,
    cache: dict[str, list[dict[str, Any]]] | None,
    stats: Counter[str] | None,
) -> list[dict[str, Any]]:
    if cache is None:
        return _action_value_rows(policy, state, sim)
    key = action_value_cache_key(policy_spec, state, getattr(sim, "starter_tile", None))
    cached = cache.get(key)
    if cached is not None:
        if stats is not None:
            stats["action_value_cache_hits"] += 1
        return _copy_action_rows(cached)
    rows = _action_value_rows(policy, state, sim)
    cache[key] = _copy_action_rows(rows)
    if stats is not None:
        stats["action_value_cache_misses"] += 1
    return rows


def load_action_value_cache(path: Path | None) -> dict[str, list[dict[str, Any]]] | None:
    if path is None:
        return None
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    raw_entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(raw_entries, dict):
        raw_entries = payload if isinstance(payload, dict) else {}
    entries: dict[str, list[dict[str, Any]]] = {}
    for key, value in raw_entries.items():
        rows = value.get("rows") if isinstance(value, dict) else value
        if isinstance(rows, list):
            entries[str(key)] = [dict(row) for row in rows if isinstance(row, dict)]
    return entries


def write_action_value_cache(path: Path, cache: dict[str, list[dict[str, Any]]]) -> None:
    payload = {
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "entries": {key: {"rows": rows} for key, rows in sorted(cache.items())},
    }
    write_json(path, payload)


def state_top_two_key(state: SimState, top_two_actions: list[str]) -> str:
    preview_key = (state.preview.kind, state.preview.value, state.preview.candidates)
    cycle_key = (
        tuple(sorted((str(key), int(value)) for key, value in state.small_counts.items())),
        int(state.small_pos),
        int(state.small_seen_total),
        int(state.span_small_pos),
        bool(state.large_pending),
        int(state.max_tile),
        int(state.move_count),
    )
    return json.dumps(
        {
            "board": [int(value) for value in np.asarray(state.board, dtype=np.int32).reshape(-1)],
            "preview": preview_key,
            "cycle": cycle_key,
            "top_two": list(top_two_actions),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def normalized_margin(rows: list[dict[str, Any]]) -> tuple[float, float]:
    if len(rows) < 2:
        return 0.0, 0.0
    margin = float(rows[0]["value"]) - float(rows[1]["value"])
    scale = max(1.0, abs(float(rows[0]["value"])), abs(float(rows[1]["value"])))
    return margin, margin / scale


def select_action_from_rows(policy: object, rows: list[dict[str, Any]], rng: np.random.Generator) -> int:
    action_values = [(int(row["action"]), float(row["value"])) for row in rows]
    if hasattr(policy, "_select_action"):
        return int(policy._select_action(action_values, rng))
    best_value = max(value for _action, value in action_values)
    best_actions = [action for action, value in action_values if value == best_value]
    return int(best_actions[int(rng.integers(len(best_actions)))])


def phase_bucket(state: SimState, starter_tile: int | None) -> str:
    built_max = max_tile_excluding_initial_starter(state.board, starter_tile)
    if built_max < 384:
        return "early_lt384"
    if built_max < 1536:
        return "mid_384_768"
    if built_max < 3072:
        return "late_1536"
    return "endgame_3072p"


def corner_risk_bucket(state: SimState, starter_tile: int | None) -> str:
    board = np.asarray(state.board, dtype=np.int32)
    top_left = int(board[0, 0])
    board_max = int(board.max(initial=0))
    empty_count = int(np.count_nonzero(board == 0))
    built_max = max_tile_excluding_initial_starter(board, starter_tile)

    risk = 0
    if built_max >= 384 and top_left != board_max:
        risk += 2
    if empty_count <= 2:
        risk += 2
    elif empty_count <= 4:
        risk += 1
    if state.preview.kind == "bonus" or state.large_pending:
        risk += 1
    if starter_tile is not None and top_left not in (0, int(starter_tile), board_max):
        risk += 1

    if risk >= 3:
        return "high_corner_risk"
    if risk >= 1:
        return "medium_corner_risk"
    return "low_corner_risk"


def state_features(state: SimState, sim: ThreesSim, starter_tile: int | None) -> dict[str, Any]:
    board = np.asarray(state.board, dtype=np.int32)
    safe_smalls = sim.safe_smalls_until_large_possible(state)
    return {
        "phase": phase_bucket(state, starter_tile),
        "corner_risk": corner_risk_bucket(state, starter_tile),
        "stratum": f"{phase_bucket(state, starter_tile)}/{corner_risk_bucket(state, starter_tile)}",
        "empty_count": int(np.count_nonzero(board == 0)),
        "top_left": int(board[0, 0]),
        "max_tile": int(board.max(initial=0)),
        "max_tile_excl_starter": int(max_tile_excluding_initial_starter(board, starter_tile)),
        "score_minus_starter": int(score_board(board) - starter_baseline_score(starter_tile)),
        "preview": state.preview.label,
        "large_pending": bool(state.large_pending),
        "safe_smalls_until_large_possible": safe_smalls,
    }


def sample_scope_key(first_per: str, seed: int, comparison_spec: str, features: dict[str, Any]) -> tuple[Any, ...] | None:
    if first_per == "none":
        return None
    if first_per == "seed":
        return (int(seed),)
    if first_per == "seed-phase":
        return (int(seed), str(features.get("phase")))
    if first_per == "seed-stratum":
        return (int(seed), str(features.get("stratum")))
    if first_per == "seed-policy":
        return (int(seed), comparison_spec)
    if first_per == "seed-policy-phase":
        return (int(seed), comparison_spec, str(features.get("phase")))
    if first_per == "seed-policy-stratum":
        return (int(seed), comparison_spec, str(features.get("stratum")))
    raise ValueError(f"Unsupported first_per mode: {first_per}")


def candidate_scope_key(
    first_per: str,
    seed: int,
    comparison_spec: str,
    features: dict[str, Any],
    source_replay: str | None,
) -> tuple[Any, ...] | None:
    if first_per == "replay":
        return ("replay", source_replay if source_replay is not None else int(seed))
    if first_per == "replay-phase":
        return ("replay", source_replay if source_replay is not None else int(seed), str(features.get("phase")))
    if first_per == "replay-stratum":
        return ("replay", source_replay if source_replay is not None else int(seed), str(features.get("stratum")))
    return sample_scope_key(first_per, seed, comparison_spec, features)


def _filter_accepts(features: dict[str, Any], phase_filter: set[str] | None, corner_risk_filter: set[str] | None) -> str | None:
    if phase_filter is not None and str(features.get("phase")) not in phase_filter:
        return "phase_filter"
    if corner_risk_filter is not None and str(features.get("corner_risk")) not in corner_risk_filter:
        return "corner_risk_filter"
    return None


def _step_with_fallback(sim: ThreesSim, state: SimState, action: int) -> SimState:
    next_state, info = sim.step(state, int(action))
    if info.moved:
        return next_state
    legal = sim.legal_actions(state)
    if not legal:
        return state
    next_state, _info = sim.step(state, int(legal[0]))
    return next_state


def _anchor_safe_actions(state: SimState, sim: ThreesSim, min_anchor_tile: int) -> tuple[int, set[int]]:
    anchor = int(state.board[0, 0])
    if anchor < int(min_anchor_tile):
        return anchor, set()
    safe: set[int] = set()
    for action in sim.legal_actions(state):
        shifted, eligible = simulate_base_move(state.board, int(action))
        if eligible and int(shifted[0, 0]) >= anchor:
            safe.add(int(action))
    return anchor, safe


def _normalized_value_gap(left_value: float, right_value: float) -> tuple[float, float]:
    gap = float(left_value) - float(right_value)
    scale = max(1.0, abs(float(left_value)), abs(float(right_value)))
    return gap, abs(gap) / scale


def _anchor_risk_candidate(
    *,
    base_policy: object,
    base_policy_spec: str,
    state: SimState,
    sim: ThreesSim,
    rows: list[dict[str, Any]],
    seed: int,
    starter_tile: int | None,
    actor_action: int,
    margin_threshold: float,
    min_anchor_tile: int,
    source_replay: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if len(rows) < 2:
        return None, "no_action_values"
    anchor, safe_actions = _anchor_safe_actions(state, sim, min_anchor_tile)
    if anchor < int(min_anchor_tile):
        return None, "anchor_below_min"
    if not safe_actions:
        return None, "no_safe_action"
    if int(actor_action) in safe_actions:
        return None, "actor_safe"
    row_by_action = {int(row["action"]): row for row in rows}
    actor_row = row_by_action.get(int(actor_action))
    if actor_row is None:
        return None, "actor_no_value"
    safe_rows = [row for row in rows if int(row["action"]) in safe_actions]
    if not safe_rows:
        return None, "no_safe_value"
    challenger_row = safe_rows[0]
    margin, norm = _normalized_value_gap(float(actor_row["value"]), float(challenger_row["value"]))
    if norm > float(margin_threshold):
        return None, "margin_filter"
    actor_name = DIRECTION_NAMES[int(actor_action)]
    challenger_name = str(challenger_row["name"])
    features = state_features(state, sim, starter_tile)
    stratum = str(features["stratum"])
    sample_id = safe_name(
        f"anchor_seed{int(seed)}_move{int(state.move_count)}_{actor_name}_vs_{challenger_name}_{stratum}",
        max_length=96,
    )
    sample = {
        "id": sample_id,
        "sample_mode": "anchor-risk",
        "seed": int(seed),
        "starter_tile": starter_tile,
        "move_count": int(state.move_count),
        "source_replay": source_replay,
        "base_policy": base_policy_spec,
        "comparison_policy": "anchor_safe",
        "base_action": actor_name,
        "comparison_action": challenger_name,
        "top_two_actions": [actor_name, challenger_name],
        "top_two_values": [float(actor_row["value"]), float(challenger_row["value"])],
        "action_values": rows,
        "margin": float(margin),
        "normalized_margin": float(norm),
        "features": features,
        "anchor": {
            "min_tile": int(min_anchor_tile),
            "top_left_value": int(anchor),
            "safe_actions": [DIRECTION_NAMES[action] for action in sorted(safe_actions)],
            "unsafe_actor_action": actor_name,
            "best_safe_action": challenger_name,
        },
        "state": state_payload(state, sim),
    }
    return sample, "accepted"


def _geometry_action_scores(
    state: SimState,
    sim: ThreesSim,
    *,
    starter_tile: int | None,
    geometry_min_tile: int,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for action in sim.legal_actions(state):
        shifted, eligible_positions = simulate_base_move(state.board, int(action))
        if not eligible_positions:
            continue
        scores.append(
            {
                "action": int(action),
                "name": DIRECTION_NAMES[int(action)],
                "geometry_score": float(
                    high_tile_geometry_score(
                        shifted,
                        starter_tile=starter_tile,
                        min_tile=int(geometry_min_tile),
                    )
                ),
                "eligible_positions": [[int(row), int(col)] for row, col in eligible_positions],
            }
        )
    scores.sort(key=lambda row: (-float(row["geometry_score"]), str(row["name"])))
    for rank, row in enumerate(scores, start=1):
        row["geometry_rank"] = rank
    return scores


def _geometry_risk_candidate(
    *,
    base_policy_spec: str,
    state: SimState,
    sim: ThreesSim,
    rows: list[dict[str, Any]],
    seed: int,
    starter_tile: int | None,
    actor_action: int,
    margin_threshold: float,
    geometry_min_tile: int,
    geometry_min_delta: float,
    source_replay: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if len(rows) < 2:
        return None, "no_action_values"
    features = state_features(state, sim, starter_tile)
    if int(features["max_tile_excl_starter"]) < int(geometry_min_tile):
        return None, "geometry_below_min"
    row_by_action = {int(row["action"]): row for row in rows}
    actor_row = row_by_action.get(int(actor_action))
    if actor_row is None:
        return None, "actor_no_value"
    action_scores = _geometry_action_scores(
        state,
        sim,
        starter_tile=starter_tile,
        geometry_min_tile=geometry_min_tile,
    )
    if not action_scores:
        return None, "no_geometry_actions"
    score_by_action = {int(row["action"]): row for row in action_scores}
    actor_score_row = score_by_action.get(int(actor_action))
    if actor_score_row is None:
        return None, "actor_no_geometry"
    actor_geometry = float(actor_score_row["geometry_score"])
    challenger_score_row = action_scores[0]
    if int(challenger_score_row["action"]) == int(actor_action):
        return None, "actor_geometry_best"
    geometry_delta = float(challenger_score_row["geometry_score"]) - actor_geometry
    if geometry_delta < float(geometry_min_delta):
        return None, "geometry_delta_filter"
    challenger_row = row_by_action.get(int(challenger_score_row["action"]))
    if challenger_row is None:
        return None, "no_geometry_value"
    margin, norm = _normalized_value_gap(float(actor_row["value"]), float(challenger_row["value"]))
    if norm > float(margin_threshold):
        return None, "margin_filter"

    actor_name = DIRECTION_NAMES[int(actor_action)]
    challenger_name = str(challenger_row["name"])
    stratum = str(features["stratum"])
    sample_id = safe_name(
        f"geometry_seed{int(seed)}_move{int(state.move_count)}_{actor_name}_vs_{challenger_name}_{stratum}",
        max_length=96,
    )
    sample = {
        "id": sample_id,
        "sample_mode": "geometry-risk",
        "seed": int(seed),
        "starter_tile": starter_tile,
        "move_count": int(state.move_count),
        "source_replay": source_replay,
        "base_policy": base_policy_spec,
        "comparison_policy": "geometry_best",
        "base_action": actor_name,
        "comparison_action": challenger_name,
        "top_two_actions": [actor_name, challenger_name],
        "top_two_values": [float(actor_row["value"]), float(challenger_row["value"])],
        "action_values": rows,
        "margin": float(margin),
        "normalized_margin": float(norm),
        "features": features,
        "geometry": {
            "min_tile": int(geometry_min_tile),
            "min_delta": float(geometry_min_delta),
            "actor_score": float(actor_geometry),
            "challenger_score": float(challenger_score_row["geometry_score"]),
            "geometry_delta": float(geometry_delta),
            "best_geometry_action": challenger_name,
            "action_scores": action_scores,
        },
        "state": state_payload(state, sim),
    }
    return sample, "accepted"


def _mask_free_starter(board: np.ndarray, starter_tile: int | None) -> np.ndarray:
    masked = np.asarray(board, dtype=np.int32).copy()
    if starter_tile is None:
        return masked
    matches = np.argwhere(masked == int(starter_tile))
    if len(matches) == 0:
        return masked
    match_idx = 0
    for idx, (row, col) in enumerate(matches):
        if int(row) == 0 and int(col) == 0:
            match_idx = idx
            break
    row, col = matches[match_idx]
    masked[int(row), int(col)] = 0
    return masked


def _tile_positions(board: np.ndarray, value: int) -> list[tuple[int, int]]:
    if int(value) <= 0:
        return []
    return [tuple(int(v) for v in pos) for pos in np.argwhere(board == int(value))]


def _adjacent_positions(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1])) == 1


def _has_adjacent_pair(positions: list[tuple[int, int]]) -> bool:
    for idx, left in enumerate(positions):
        for right in positions[idx + 1 :]:
            if _adjacent_positions(left, right):
                return True
    return False


def _highest_duplicate_tile_for_support(board: np.ndarray) -> int:
    counts = Counter(int(value) for value in board.reshape(-1) if int(value) > 0)
    return max((value for value, count in counts.items() if count >= 2), default=0)


def _highest_adjacent_pair_tile_for_support(board: np.ndarray) -> int:
    best = 0
    for value in sorted({int(value) for value in board.reshape(-1) if int(value) > 0}, reverse=True):
        if value <= best:
            continue
        if _has_adjacent_pair(_tile_positions(board, value)):
            return int(value)
    return int(best)


def _min_pair_distance(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int | None:
    if not left or not right:
        return None
    return min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in left for b in right)


def _min_top_left_distance(positions: list[tuple[int, int]]) -> int | None:
    if not positions:
        return None
    return min(row + col for row, col in positions)


def support_chain_features(
    board: np.ndarray,
    *,
    starter_tile: int | None,
    support_min_tile: int,
    target_min_tile: int,
    mask_starter: bool = True,
) -> dict[str, Any]:
    """Score whether a board is building support for the next max-tile copy."""

    support_board = _mask_free_starter(board, starter_tile) if mask_starter else np.asarray(board, dtype=np.int32)
    built_max = int(support_board.max(initial=0))
    if built_max < int(target_min_tile):
        return {
            "score": 0.0,
            "built_max": built_max,
            "target_support_tile": 0,
            "highest_duplicate_tile": _highest_duplicate_tile_for_support(support_board),
            "highest_adjacent_pair_tile": _highest_adjacent_pair_tile_for_support(support_board),
            "scored_duplicate_tile": 0,
            "scored_adjacent_pair_tile": 0,
            "count_built_max": int(np.count_nonzero(support_board == built_max)) if built_max else 0,
            "count_target_support": 0,
            "target_support_has_duplicate": False,
            "target_support_has_adjacent_pair": False,
            "target_support_adjacent_to_max": False,
            "min_target_support_to_max_distance": None,
            "min_target_support_to_top_left_distance": None,
            "empty_count": int(np.count_nonzero(support_board == 0)),
            "high_support_count": int(sum(1 for value in support_board.reshape(-1) if int(value) >= int(support_min_tile))),
            "mask_starter": bool(mask_starter),
        }

    target_support = int(built_max // 2)
    max_positions = _tile_positions(support_board, built_max)
    target_positions = _tile_positions(support_board, target_support)
    highest_duplicate = _highest_duplicate_tile_for_support(support_board)
    highest_adjacent = _highest_adjacent_pair_tile_for_support(support_board)
    count_built_max = int(np.count_nonzero(support_board == built_max))
    count_target = len(target_positions)
    target_has_duplicate = count_target >= 2
    target_has_adjacent_pair = _has_adjacent_pair(target_positions)
    support_to_max_distance = _min_pair_distance(target_positions, max_positions)
    support_to_top_left_distance = _min_top_left_distance(target_positions)
    support_adjacent_to_max = support_to_max_distance == 1
    empty_count = int(np.count_nonzero(support_board == 0))
    support_floor = int(support_min_tile)
    scored_duplicate = highest_duplicate if highest_duplicate >= support_floor else 0
    scored_adjacent = highest_adjacent if highest_adjacent >= support_floor else 0

    target_rank = rank_for_value(target_support)
    max_rank = rank_for_value(built_max)
    duplicate_rank = rank_for_value(scored_duplicate)
    adjacent_rank = rank_for_value(scored_adjacent)

    score = 0.0
    score += 90.0 * float(adjacent_rank)
    score += 45.0 * float(duplicate_rank)
    score += 275.0 * float(max(0, count_built_max - 1)) * float(max_rank)
    score += 30.0 * float(min(count_target, 4)) * float(target_rank)
    if target_has_duplicate:
        score += 110.0 * float(target_rank)
    if target_has_adjacent_pair:
        score += 180.0 * float(target_rank)
    if support_adjacent_to_max:
        score += 130.0 * float(target_rank)
    if support_to_max_distance is not None:
        score += 18.0 * float(target_rank) * max(0.0, 5.0 - float(support_to_max_distance))
    if support_to_top_left_distance is not None:
        score += 10.0 * float(target_rank) * max(0.0, 6.0 - float(support_to_top_left_distance))
    score += 2.0 * float(empty_count)

    high_support_count = int(sum(1 for value in support_board.reshape(-1) if int(value) >= support_floor))
    return {
        "score": float(score),
        "built_max": int(built_max),
        "target_support_tile": int(target_support),
        "highest_duplicate_tile": int(highest_duplicate),
        "highest_adjacent_pair_tile": int(highest_adjacent),
        "scored_duplicate_tile": int(scored_duplicate),
        "scored_adjacent_pair_tile": int(scored_adjacent),
        "count_built_max": int(count_built_max),
        "count_target_support": int(count_target),
        "target_support_has_duplicate": bool(target_has_duplicate),
        "target_support_has_adjacent_pair": bool(target_has_adjacent_pair),
        "target_support_adjacent_to_max": bool(support_adjacent_to_max),
        "min_target_support_to_max_distance": None
        if support_to_max_distance is None
        else int(support_to_max_distance),
        "min_target_support_to_top_left_distance": None
        if support_to_top_left_distance is None
        else int(support_to_top_left_distance),
        "empty_count": int(empty_count),
        "high_support_count": int(high_support_count),
        "mask_starter": bool(mask_starter),
    }


def _support_chain_action_scores(
    state: SimState,
    sim: ThreesSim,
    *,
    starter_tile: int | None,
    support_min_tile: int,
    support_target_min_tile: int,
    support_mask_starter: bool,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for action in sim.legal_actions(state):
        shifted, eligible_positions = simulate_base_move(state.board, int(action))
        if not eligible_positions:
            continue
        support = support_chain_features(
            shifted,
            starter_tile=starter_tile,
            support_min_tile=support_min_tile,
            target_min_tile=support_target_min_tile,
            mask_starter=support_mask_starter,
        )
        scores.append(
            {
                "action": int(action),
                "name": DIRECTION_NAMES[int(action)],
                "support_score": float(support["score"]),
                "support": support,
                "eligible_positions": [[int(row), int(col)] for row, col in eligible_positions],
            }
        )
    scores.sort(key=lambda row: (-float(row["support_score"]), str(row["name"])))
    for rank, row in enumerate(scores, start=1):
        row["support_rank"] = rank
    return scores


def _support_chain_risk_candidate(
    *,
    base_policy_spec: str,
    state: SimState,
    sim: ThreesSim,
    rows: list[dict[str, Any]],
    seed: int,
    starter_tile: int | None,
    actor_action: int,
    margin_threshold: float,
    support_min_tile: int,
    support_target_min_tile: int,
    support_mask_starter: bool,
    support_min_delta: float,
    source_replay: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    if len(rows) < 2:
        return None, "no_action_values"
    features = state_features(state, sim, starter_tile)
    if int(features["max_tile_excl_starter"]) < int(support_target_min_tile):
        return None, "support_below_min"
    row_by_action = {int(row["action"]): row for row in rows}
    actor_row = row_by_action.get(int(actor_action))
    if actor_row is None:
        return None, "actor_no_value"
    action_scores = _support_chain_action_scores(
        state,
        sim,
        starter_tile=starter_tile,
        support_min_tile=support_min_tile,
        support_target_min_tile=support_target_min_tile,
        support_mask_starter=support_mask_starter,
    )
    if not action_scores:
        return None, "no_support_actions"
    score_by_action = {int(row["action"]): row for row in action_scores}
    actor_score_row = score_by_action.get(int(actor_action))
    if actor_score_row is None:
        return None, "actor_no_support"
    actor_support = float(actor_score_row["support_score"])
    challenger_score_row = action_scores[0]
    if int(challenger_score_row["action"]) == int(actor_action):
        return None, "actor_support_best"
    support_delta = float(challenger_score_row["support_score"]) - actor_support
    if support_delta < float(support_min_delta):
        return None, "support_delta_filter"
    challenger_row = row_by_action.get(int(challenger_score_row["action"]))
    if challenger_row is None:
        return None, "no_support_value"
    margin, norm = _normalized_value_gap(float(actor_row["value"]), float(challenger_row["value"]))
    if norm > float(margin_threshold):
        return None, "margin_filter"

    actor_name = DIRECTION_NAMES[int(actor_action)]
    challenger_name = str(challenger_row["name"])
    stratum = str(features["stratum"])
    sample_id = safe_name(
        f"support_seed{int(seed)}_move{int(state.move_count)}_{actor_name}_vs_{challenger_name}_{stratum}",
        max_length=96,
    )
    sample = {
        "id": sample_id,
        "sample_mode": "support-chain-risk",
        "seed": int(seed),
        "starter_tile": starter_tile,
        "move_count": int(state.move_count),
        "source_replay": source_replay,
        "base_policy": base_policy_spec,
        "comparison_policy": "support_chain_best",
        "base_action": actor_name,
        "comparison_action": challenger_name,
        "top_two_actions": [actor_name, challenger_name],
        "top_two_values": [float(actor_row["value"]), float(challenger_row["value"])],
        "action_values": rows,
        "margin": float(margin),
        "normalized_margin": float(norm),
        "features": features,
        "support_chain": {
            "support_min_tile": int(support_min_tile),
            "target_min_tile": int(support_target_min_tile),
            "mask_starter": bool(support_mask_starter),
            "min_delta": float(support_min_delta),
            "actor_score": float(actor_support),
            "challenger_score": float(challenger_score_row["support_score"]),
            "support_delta": float(support_delta),
            "best_support_action": challenger_name,
            "action_scores": action_scores,
        },
        "state": state_payload(state, sim),
    }
    return sample, "accepted"


def _accept_candidate_sample(
    *,
    sample: dict[str, Any],
    comparison_spec: str,
    first_per: str,
    stratum_counts: Counter[str],
    accepted_scopes: set[tuple[Any, ...]],
    seen_state_top_two: set[str],
    max_samples: int,
    max_per_stratum: int,
    current_sample_count: int,
    rejected: Counter[str],
    phase_filter: set[str] | None,
    corner_risk_filter: set[str] | None,
) -> bool:
    features = sample.get("features", {})
    if not isinstance(features, dict):
        rejected["bad_features"] += 1
        return False
    filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
    if filter_reason is not None:
        rejected[filter_reason] += 1
        return False
    source_replay = sample.get("source_replay")
    scope_key = candidate_scope_key(
        first_per,
        int(sample["seed"]),
        comparison_spec,
        features,
        str(source_replay) if source_replay is not None else None,
    )
    if scope_key is not None and scope_key in accepted_scopes:
        rejected["scope_seen"] += 1
        return False
    stratum = str(features["stratum"])
    if stratum_counts[stratum] >= max_per_stratum:
        rejected["bucket_full"] += 1
        return False
    if current_sample_count >= max_samples:
        rejected["global_full"] += 1
        return False
    state = state_from_payload(sample["state"])
    state_key = state_top_two_key(state, [str(name) for name in sample["top_two_actions"]])
    if state_key in seen_state_top_two:
        rejected["duplicate_state"] += 1
        return False
    seen_state_top_two.add(state_key)
    stratum_counts[stratum] += 1
    if scope_key is not None:
        accepted_scopes.add(scope_key)
    return True


def collect_anchor_risk_states(
    *,
    base_policy: object,
    base_policy_spec: str,
    seeds: Iterable[int],
    starter_tile: int | None,
    max_moves: int,
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-stratum",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    min_anchor_tile: int = 1536,
    progress_every_seeds: int = 0,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_moves = 0
    anchor_moves = 0
    unsafe_actor_moves = 0
    low_margin_moves = 0
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()

    seed_list = list(seeds)
    for seed_idx, seed in enumerate(seed_list, start=1):
        sim = ThreesSim(np.random.default_rng(int(seed)), starter_tile=starter_tile)
        state = sim.reset()
        base_rng = np.random.default_rng(int(seed) + POLICY_RNG_OFFSET)
        for _move_idx in range(max_moves):
            if state.game_over or not sim.legal_actions(state):
                break
            scanned_moves += 1
            rows = _action_value_rows(base_policy, state, sim)
            if len(rows) < 2:
                actor_action = int(base_policy(state, sim, base_rng))
            else:
                actor_action = select_action_from_rows(base_policy, rows, base_rng)
            anchor, safe_actions = _anchor_safe_actions(state, sim, min_anchor_tile)
            if anchor >= int(min_anchor_tile):
                anchor_moves += 1
                if actor_action not in safe_actions:
                    unsafe_actor_moves += 1
            sample, reason = _anchor_risk_candidate(
                base_policy=base_policy,
                base_policy_spec=base_policy_spec,
                state=state,
                sim=sim,
                rows=rows,
                seed=int(seed),
                starter_tile=starter_tile,
                actor_action=actor_action,
                margin_threshold=margin_threshold,
                min_anchor_tile=min_anchor_tile,
            )
            if sample is None:
                rejected[reason] += 1
            else:
                low_margin_moves += 1
                if _accept_candidate_sample(
                    sample=sample,
                    comparison_spec="anchor_safe",
                    first_per=first_per,
                    stratum_counts=stratum_counts,
                    accepted_scopes=accepted_scopes,
                    seen_state_top_two=seen_state_top_two,
                    max_samples=max_samples,
                    max_per_stratum=max_per_stratum,
                    current_sample_count=len(samples),
                    rejected=rejected,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                ):
                    samples.append(sample)
                    if len(samples) >= max_samples:
                        break
            state = _step_with_fallback(sim, state, actor_action)
        if len(samples) >= max_samples:
            break
        if progress_every_seeds > 0 and seed_idx % int(progress_every_seeds) == 0:
            print(
                "anchor_scan_progress "
                f"seeds={seed_idx}/{len(seed_list)} "
                f"scanned_moves={scanned_moves} "
                f"anchor_moves={anchor_moves} "
                f"unsafe_actor={unsafe_actor_moves} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    return {
        "samples": samples,
        "scan_stats": {
            "source": "actor",
            "sample_mode": "anchor-risk",
            "scanned_moves": int(scanned_moves),
            "anchor_moves": int(anchor_moves),
            "unsafe_actor_moves": int(unsafe_actor_moves),
            "unsafe_actor_move_rate": float(unsafe_actor_moves / anchor_moves) if anchor_moves else 0.0,
            "low_margin_moves": int(low_margin_moves),
            "low_margin_move_rate": float(low_margin_moves / scanned_moves) if scanned_moves else 0.0,
            "accepted_samples": len(samples),
            "accepted_scopes": len(accepted_scopes),
            "strata": dict(stratum_counts),
            "rejected": dict(rejected),
        },
    }


def collect_anchor_risk_states_from_replays(
    *,
    base_policy: object,
    base_policy_spec: str,
    replay_paths: Iterable[Path],
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-stratum",
    replay_base_action: str = "recorded",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    min_anchor_tile: int = 1536,
    max_replays: int = 0,
    replay_start_index: int = 0,
    progress_every_replays: int = 0,
    action_value_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if replay_base_action not in ("recorded", "policy"):
        raise ValueError(f"Unsupported replay_base_action {replay_base_action!r}")
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_moves = 0
    anchor_moves = 0
    unsafe_actor_moves = 0
    low_margin_moves = 0
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    all_paths = list(replay_paths)
    start_index = max(0, int(replay_start_index))
    candidate_paths = all_paths[start_index:]
    paths = candidate_paths[: int(max_replays)] if int(max_replays) > 0 else candidate_paths

    for replay_idx, replay_path in enumerate(paths, start=1):
        replay = json.loads(Path(replay_path).read_text())
        seed = int(replay.get("seed", replay_idx))
        starter_tile = replay.get("starter_tile", 1536)
        frames = replay.get("frames", [])
        if not isinstance(frames, list):
            rejected["bad_replay"] += 1
            continue
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        base_rng = np.random.default_rng(seed + POLICY_RNG_OFFSET)

        for before_idx in range(max(0, len(frames) - 1)):
            before_frame = frames[before_idx]
            after_frame = frames[before_idx + 1]
            if not isinstance(before_frame, dict) or not isinstance(after_frame, dict):
                continue
            state_payload_obj = before_frame.get("state")
            if not isinstance(state_payload_obj, dict):
                continue
            state = state_from_payload(state_payload_obj)
            if state.game_over or not sim.legal_actions(state):
                continue
            scanned_moves += 1
            features = state_features(state, sim, starter_tile)
            filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
            if filter_reason is not None:
                rejected[filter_reason] += 1
                continue
            rows = _cached_action_value_rows(
                base_policy,
                base_policy_spec,
                state,
                sim,
                action_value_cache,
                cache_stats,
            )
            if len(rows) < 2:
                rejected["no_action_values"] += 1
                continue
            if replay_base_action == "policy":
                actor_action = select_action_from_rows(base_policy, rows, base_rng)
            else:
                move = after_frame.get("move")
                if not isinstance(move, dict) or move.get("action") is None:
                    rejected["bad_action"] += 1
                    continue
                try:
                    actor_action = direction_index(str(move["action"]))
                except ValueError:
                    rejected["bad_action"] += 1
                    continue
            anchor, safe_actions = _anchor_safe_actions(state, sim, min_anchor_tile)
            if anchor >= int(min_anchor_tile):
                anchor_moves += 1
                if int(actor_action) not in safe_actions:
                    unsafe_actor_moves += 1
            sample, reason = _anchor_risk_candidate(
                base_policy=base_policy,
                base_policy_spec=base_policy_spec,
                state=state,
                sim=sim,
                rows=rows,
                seed=seed,
                starter_tile=starter_tile,
                actor_action=int(actor_action),
                margin_threshold=margin_threshold,
                min_anchor_tile=min_anchor_tile,
                source_replay=str(replay_path),
            )
            if sample is None:
                rejected[reason] += 1
                continue
            low_margin_moves += 1
            if _accept_candidate_sample(
                sample=sample,
                comparison_spec="anchor_safe",
                first_per=first_per,
                stratum_counts=stratum_counts,
                accepted_scopes=accepted_scopes,
                seen_state_top_two=seen_state_top_two,
                max_samples=max_samples,
                max_per_stratum=max_per_stratum,
                current_sample_count=len(samples),
                rejected=rejected,
                phase_filter=phase_filter,
                corner_risk_filter=corner_risk_filter,
            ):
                samples.append(sample)
                if len(samples) >= max_samples:
                    break
        if len(samples) >= max_samples:
            break
        if progress_every_replays > 0 and replay_idx % int(progress_every_replays) == 0:
            print(
                "anchor_replay_scan_progress "
                f"replays={replay_idx}/{len(paths)} "
                f"scanned_moves={scanned_moves} "
                f"anchor_moves={anchor_moves} "
                f"unsafe_actor={unsafe_actor_moves} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    scan_stats = {
        "source": "replay",
        "sample_mode": "anchor-risk",
        "replays": len(paths),
        "available_replays": len(all_paths),
        "replay_start_index": int(start_index),
        "scanned_replay_paths": [str(path) for path in paths],
        "scanned_moves": int(scanned_moves),
        "anchor_moves": int(anchor_moves),
        "unsafe_actor_moves": int(unsafe_actor_moves),
        "unsafe_actor_move_rate": float(unsafe_actor_moves / anchor_moves) if anchor_moves else 0.0,
        "low_margin_moves": int(low_margin_moves),
        "low_margin_move_rate": float(low_margin_moves / scanned_moves) if scanned_moves else 0.0,
        "accepted_samples": len(samples),
        "accepted_scopes": len(accepted_scopes),
        "strata": dict(stratum_counts),
        "rejected": dict(rejected),
    }
    if action_value_cache is not None:
        scan_stats.update(
            {
                "action_value_cache_hits": int(cache_stats["action_value_cache_hits"]),
                "action_value_cache_misses": int(cache_stats["action_value_cache_misses"]),
                "action_value_cache_entries": len(action_value_cache),
            }
        )

    return {
        "samples": samples,
        "scan_stats": scan_stats,
    }


def collect_anchor_risk_states_from_state_records(
    *,
    base_policy: object,
    base_policy_spec: str,
    state_records: Iterable[dict[str, Any]],
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-stratum",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    min_anchor_tile: int = 1536,
    max_state_records: int = 0,
    state_start_index: int = 0,
    progress_every_state_records: int = 0,
    action_value_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_records = 0
    anchor_records = 0
    unsafe_actor_records = 0
    low_margin_records = 0
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    all_records, records = _state_record_slice(
        state_records,
        start_index=state_start_index,
        max_records=max_state_records,
    )

    for record_idx, record in enumerate(records, start=1):
        state_payload_obj = record.get("state")
        if not isinstance(state_payload_obj, dict):
            rejected["bad_state_record"] += 1
            continue
        seed = _record_seed(record, record_idx)
        starter_tile = _record_starter_tile(record)
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        base_rng = np.random.default_rng(seed + POLICY_RNG_OFFSET + record_idx * 7919)
        state = state_from_payload(state_payload_obj)
        if state.game_over or not sim.legal_actions(state):
            rejected["terminal_or_no_legal"] += 1
            continue
        scanned_records += 1
        features = state_features(state, sim, starter_tile)
        filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
        if filter_reason is not None:
            rejected[filter_reason] += 1
            continue
        rows = _cached_action_value_rows(
            base_policy,
            base_policy_spec,
            state,
            sim,
            action_value_cache,
            cache_stats,
        )
        if len(rows) < 2:
            rejected["no_action_values"] += 1
            continue
        actor_action = select_action_from_rows(base_policy, rows, base_rng)
        anchor, safe_actions = _anchor_safe_actions(state, sim, min_anchor_tile)
        if anchor >= int(min_anchor_tile):
            anchor_records += 1
            if int(actor_action) not in safe_actions:
                unsafe_actor_records += 1
        sample, reason = _anchor_risk_candidate(
            base_policy=base_policy,
            base_policy_spec=base_policy_spec,
            state=state,
            sim=sim,
            rows=rows,
            seed=seed,
            starter_tile=starter_tile,
            actor_action=int(actor_action),
            margin_threshold=margin_threshold,
            min_anchor_tile=min_anchor_tile,
            source_replay=_record_source_replay(record),
        )
        if sample is None:
            rejected[reason] += 1
            continue
        _annotate_sample_from_state_record(sample, record)
        low_margin_records += 1
        if _accept_candidate_sample(
            sample=sample,
            comparison_spec="anchor_safe",
            first_per=first_per,
            stratum_counts=stratum_counts,
            accepted_scopes=accepted_scopes,
            seen_state_top_two=seen_state_top_two,
            max_samples=max_samples,
            max_per_stratum=max_per_stratum,
            current_sample_count=len(samples),
            rejected=rejected,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
        ):
            samples.append(sample)
            if len(samples) >= max_samples:
                break
        if progress_every_state_records > 0 and record_idx % int(progress_every_state_records) == 0:
            print(
                "anchor_state_scan_progress "
                f"records={record_idx}/{len(records)} "
                f"scanned_records={scanned_records} "
                f"anchor_records={anchor_records} "
                f"unsafe_actor={unsafe_actor_records} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    scan_stats = {
        "source": "state_records",
        "sample_mode": "anchor-risk",
        "state_records": len(records),
        "available_state_records": len(all_records),
        "state_start_index": int(max(0, state_start_index)),
        "scanned_records": int(scanned_records),
        "anchor_records": int(anchor_records),
        "unsafe_actor_records": int(unsafe_actor_records),
        "unsafe_actor_record_rate": float(unsafe_actor_records / anchor_records) if anchor_records else 0.0,
        "low_margin_records": int(low_margin_records),
        "low_margin_record_rate": float(low_margin_records / scanned_records) if scanned_records else 0.0,
        "accepted_samples": len(samples),
        "accepted_scopes": len(accepted_scopes),
        "strata": dict(stratum_counts),
        "rejected": dict(rejected),
    }
    if action_value_cache is not None:
        scan_stats.update(
            {
                "action_value_cache_hits": int(cache_stats["action_value_cache_hits"]),
                "action_value_cache_misses": int(cache_stats["action_value_cache_misses"]),
                "action_value_cache_entries": len(action_value_cache),
            }
        )

    return {
        "samples": samples,
        "scan_stats": scan_stats,
    }


def collect_geometry_risk_states_from_replays(
    *,
    base_policy: object,
    base_policy_spec: str,
    replay_paths: Iterable[Path],
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-stratum",
    replay_base_action: str = "recorded",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    geometry_min_tile: int = 1536,
    geometry_min_delta: float = 1.0,
    max_replays: int = 0,
    replay_start_index: int = 0,
    progress_every_replays: int = 0,
    action_value_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if replay_base_action not in ("recorded", "policy"):
        raise ValueError(f"Unsupported replay_base_action {replay_base_action!r}")
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_moves = 0
    geometry_moves = 0
    geometry_disagreements = 0
    low_margin_moves = 0
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    all_paths = list(replay_paths)
    start_index = max(0, int(replay_start_index))
    candidate_paths = all_paths[start_index:]
    paths = candidate_paths[: int(max_replays)] if int(max_replays) > 0 else candidate_paths

    for replay_idx, replay_path in enumerate(paths, start=1):
        replay = json.loads(Path(replay_path).read_text())
        seed = int(replay.get("seed", replay_idx))
        starter_tile = replay.get("starter_tile", 1536)
        frames = replay.get("frames", [])
        if not isinstance(frames, list):
            rejected["bad_replay"] += 1
            continue
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        base_rng = np.random.default_rng(seed + POLICY_RNG_OFFSET)

        for before_idx in range(max(0, len(frames) - 1)):
            before_frame = frames[before_idx]
            after_frame = frames[before_idx + 1]
            if not isinstance(before_frame, dict) or not isinstance(after_frame, dict):
                continue
            state_payload_obj = before_frame.get("state")
            if not isinstance(state_payload_obj, dict):
                continue
            state = state_from_payload(state_payload_obj)
            if state.game_over or not sim.legal_actions(state):
                continue
            scanned_moves += 1
            features = state_features(state, sim, starter_tile)
            filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
            if filter_reason is not None:
                rejected[filter_reason] += 1
                continue
            if int(features["max_tile_excl_starter"]) >= int(geometry_min_tile):
                geometry_moves += 1
            rows = _cached_action_value_rows(
                base_policy,
                base_policy_spec,
                state,
                sim,
                action_value_cache,
                cache_stats,
            )
            if len(rows) < 2:
                rejected["no_action_values"] += 1
                continue
            if replay_base_action == "policy":
                actor_action = select_action_from_rows(base_policy, rows, base_rng)
            else:
                move = after_frame.get("move")
                if not isinstance(move, dict) or move.get("action") is None:
                    rejected["bad_action"] += 1
                    continue
                try:
                    actor_action = direction_index(str(move["action"]))
                except ValueError:
                    rejected["bad_action"] += 1
                    continue
            sample, reason = _geometry_risk_candidate(
                base_policy_spec=base_policy_spec,
                state=state,
                sim=sim,
                rows=rows,
                seed=seed,
                starter_tile=starter_tile,
                actor_action=int(actor_action),
                margin_threshold=margin_threshold,
                geometry_min_tile=geometry_min_tile,
                geometry_min_delta=geometry_min_delta,
                source_replay=str(replay_path),
            )
            if sample is None:
                rejected[reason] += 1
                if reason == "margin_filter":
                    geometry_disagreements += 1
                continue
            geometry_disagreements += 1
            low_margin_moves += 1
            if _accept_candidate_sample(
                sample=sample,
                comparison_spec="geometry_best",
                first_per=first_per,
                stratum_counts=stratum_counts,
                accepted_scopes=accepted_scopes,
                seen_state_top_two=seen_state_top_two,
                max_samples=max_samples,
                max_per_stratum=max_per_stratum,
                current_sample_count=len(samples),
                rejected=rejected,
                phase_filter=phase_filter,
                corner_risk_filter=corner_risk_filter,
            ):
                samples.append(sample)
                if len(samples) >= max_samples:
                    break
        if len(samples) >= max_samples:
            break
        if progress_every_replays > 0 and replay_idx % int(progress_every_replays) == 0:
            print(
                "geometry_replay_scan_progress "
                f"replays={replay_idx}/{len(paths)} "
                f"scanned_moves={scanned_moves} "
                f"geometry_moves={geometry_moves} "
                f"geometry_disagreements={geometry_disagreements} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    scan_stats = {
        "source": "replay",
        "sample_mode": "geometry-risk",
        "replays": len(paths),
        "available_replays": len(all_paths),
        "replay_start_index": int(start_index),
        "scanned_replay_paths": [str(path) for path in paths],
        "scanned_moves": int(scanned_moves),
        "geometry_min_tile": int(geometry_min_tile),
        "geometry_min_delta": float(geometry_min_delta),
        "geometry_moves": int(geometry_moves),
        "geometry_move_rate": float(geometry_moves / scanned_moves) if scanned_moves else 0.0,
        "geometry_disagreements": int(geometry_disagreements),
        "geometry_disagreement_rate": float(geometry_disagreements / geometry_moves) if geometry_moves else 0.0,
        "low_margin_moves": int(low_margin_moves),
        "low_margin_move_rate": float(low_margin_moves / scanned_moves) if scanned_moves else 0.0,
        "accepted_samples": len(samples),
        "accepted_scopes": len(accepted_scopes),
        "strata": dict(stratum_counts),
        "rejected": dict(rejected),
    }
    if action_value_cache is not None:
        scan_stats.update(
            {
                "action_value_cache_hits": int(cache_stats["action_value_cache_hits"]),
                "action_value_cache_misses": int(cache_stats["action_value_cache_misses"]),
                "action_value_cache_entries": len(action_value_cache),
            }
        )

    return {
        "samples": samples,
        "scan_stats": scan_stats,
    }


def collect_geometry_risk_states_from_state_records(
    *,
    base_policy: object,
    base_policy_spec: str,
    state_records: Iterable[dict[str, Any]],
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-stratum",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    geometry_min_tile: int = 1536,
    geometry_min_delta: float = 1.0,
    max_state_records: int = 0,
    state_start_index: int = 0,
    progress_every_state_records: int = 0,
    action_value_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_records = 0
    geometry_records = 0
    geometry_disagreements = 0
    low_margin_records = 0
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    all_records, records = _state_record_slice(
        state_records,
        start_index=state_start_index,
        max_records=max_state_records,
    )

    for record_idx, record in enumerate(records, start=1):
        state_payload_obj = record.get("state")
        if not isinstance(state_payload_obj, dict):
            rejected["bad_state_record"] += 1
            continue
        seed = _record_seed(record, record_idx)
        starter_tile = _record_starter_tile(record)
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        base_rng = np.random.default_rng(seed + POLICY_RNG_OFFSET + record_idx * 7919)
        state = state_from_payload(state_payload_obj)
        if state.game_over or not sim.legal_actions(state):
            rejected["terminal_or_no_legal"] += 1
            continue
        scanned_records += 1
        features = state_features(state, sim, starter_tile)
        filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
        if filter_reason is not None:
            rejected[filter_reason] += 1
            continue
        if int(features["max_tile_excl_starter"]) >= int(geometry_min_tile):
            geometry_records += 1
        rows = _cached_action_value_rows(
            base_policy,
            base_policy_spec,
            state,
            sim,
            action_value_cache,
            cache_stats,
        )
        if len(rows) < 2:
            rejected["no_action_values"] += 1
            continue
        actor_action = select_action_from_rows(base_policy, rows, base_rng)
        sample, reason = _geometry_risk_candidate(
            base_policy_spec=base_policy_spec,
            state=state,
            sim=sim,
            rows=rows,
            seed=seed,
            starter_tile=starter_tile,
            actor_action=int(actor_action),
            margin_threshold=margin_threshold,
            geometry_min_tile=geometry_min_tile,
            geometry_min_delta=geometry_min_delta,
            source_replay=_record_source_replay(record),
        )
        if sample is None:
            rejected[reason] += 1
            if reason == "margin_filter":
                geometry_disagreements += 1
            continue
        _annotate_sample_from_state_record(sample, record)
        geometry_disagreements += 1
        low_margin_records += 1
        if _accept_candidate_sample(
            sample=sample,
            comparison_spec="geometry_best",
            first_per=first_per,
            stratum_counts=stratum_counts,
            accepted_scopes=accepted_scopes,
            seen_state_top_two=seen_state_top_two,
            max_samples=max_samples,
            max_per_stratum=max_per_stratum,
            current_sample_count=len(samples),
            rejected=rejected,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
        ):
            samples.append(sample)
            if len(samples) >= max_samples:
                break
        if progress_every_state_records > 0 and record_idx % int(progress_every_state_records) == 0:
            print(
                "geometry_state_scan_progress "
                f"records={record_idx}/{len(records)} "
                f"scanned_records={scanned_records} "
                f"geometry_records={geometry_records} "
                f"geometry_disagreements={geometry_disagreements} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    scan_stats = {
        "source": "state_records",
        "sample_mode": "geometry-risk",
        "state_records": len(records),
        "available_state_records": len(all_records),
        "state_start_index": int(max(0, state_start_index)),
        "scanned_records": int(scanned_records),
        "geometry_min_tile": int(geometry_min_tile),
        "geometry_min_delta": float(geometry_min_delta),
        "geometry_records": int(geometry_records),
        "geometry_record_rate": float(geometry_records / scanned_records) if scanned_records else 0.0,
        "geometry_disagreements": int(geometry_disagreements),
        "geometry_disagreement_rate": float(geometry_disagreements / geometry_records) if geometry_records else 0.0,
        "low_margin_records": int(low_margin_records),
        "low_margin_record_rate": float(low_margin_records / scanned_records) if scanned_records else 0.0,
        "accepted_samples": len(samples),
        "accepted_scopes": len(accepted_scopes),
        "strata": dict(stratum_counts),
        "rejected": dict(rejected),
    }
    if action_value_cache is not None:
        scan_stats.update(
            {
                "action_value_cache_hits": int(cache_stats["action_value_cache_hits"]),
                "action_value_cache_misses": int(cache_stats["action_value_cache_misses"]),
                "action_value_cache_entries": len(action_value_cache),
            }
        )

    return {
        "samples": samples,
        "scan_stats": scan_stats,
    }


def collect_support_chain_risk_states_from_replays(
    *,
    base_policy: object,
    base_policy_spec: str,
    replay_paths: Iterable[Path],
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-stratum",
    replay_base_action: str = "recorded",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    support_min_tile: int = 768,
    support_target_min_tile: int = 3072,
    support_mask_starter: bool = True,
    support_min_delta: float = 50.0,
    max_replays: int = 0,
    replay_start_index: int = 0,
    progress_every_replays: int = 0,
    action_value_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if replay_base_action not in ("recorded", "policy"):
        raise ValueError(f"Unsupported replay_base_action {replay_base_action!r}")
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_moves = 0
    support_moves = 0
    support_disagreements = 0
    low_margin_moves = 0
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    all_paths = list(replay_paths)
    start_index = max(0, int(replay_start_index))
    candidate_paths = all_paths[start_index:]
    paths = candidate_paths[: int(max_replays)] if int(max_replays) > 0 else candidate_paths

    for replay_idx, replay_path in enumerate(paths, start=1):
        replay = json.loads(Path(replay_path).read_text())
        seed = int(replay.get("seed", replay_idx))
        starter_tile = replay.get("starter_tile", 1536)
        frames = replay.get("frames", [])
        if not isinstance(frames, list):
            rejected["bad_replay"] += 1
            continue
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        base_rng = np.random.default_rng(seed + POLICY_RNG_OFFSET)

        for before_idx in range(max(0, len(frames) - 1)):
            before_frame = frames[before_idx]
            after_frame = frames[before_idx + 1]
            if not isinstance(before_frame, dict) or not isinstance(after_frame, dict):
                continue
            state_payload_obj = before_frame.get("state")
            if not isinstance(state_payload_obj, dict):
                continue
            state = state_from_payload(state_payload_obj)
            if state.game_over or not sim.legal_actions(state):
                continue
            scanned_moves += 1
            features = state_features(state, sim, starter_tile)
            filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
            if filter_reason is not None:
                rejected[filter_reason] += 1
                continue
            if int(features["max_tile_excl_starter"]) >= int(support_target_min_tile):
                support_moves += 1
            rows = _cached_action_value_rows(
                base_policy,
                base_policy_spec,
                state,
                sim,
                action_value_cache,
                cache_stats,
            )
            if len(rows) < 2:
                rejected["no_action_values"] += 1
                continue
            if replay_base_action == "policy":
                actor_action = select_action_from_rows(base_policy, rows, base_rng)
            else:
                move = after_frame.get("move")
                if not isinstance(move, dict) or move.get("action") is None:
                    rejected["bad_action"] += 1
                    continue
                try:
                    actor_action = direction_index(str(move["action"]))
                except ValueError:
                    rejected["bad_action"] += 1
                    continue
            sample, reason = _support_chain_risk_candidate(
                base_policy_spec=base_policy_spec,
                state=state,
                sim=sim,
                rows=rows,
                seed=seed,
                starter_tile=starter_tile,
                actor_action=int(actor_action),
                margin_threshold=margin_threshold,
                support_min_tile=support_min_tile,
                support_target_min_tile=support_target_min_tile,
                support_mask_starter=support_mask_starter,
                support_min_delta=support_min_delta,
                source_replay=str(replay_path),
            )
            if sample is None:
                rejected[reason] += 1
                if reason == "margin_filter":
                    support_disagreements += 1
                continue
            support_disagreements += 1
            low_margin_moves += 1
            if _accept_candidate_sample(
                sample=sample,
                comparison_spec="support_chain_best",
                first_per=first_per,
                stratum_counts=stratum_counts,
                accepted_scopes=accepted_scopes,
                seen_state_top_two=seen_state_top_two,
                max_samples=max_samples,
                max_per_stratum=max_per_stratum,
                current_sample_count=len(samples),
                rejected=rejected,
                phase_filter=phase_filter,
                corner_risk_filter=corner_risk_filter,
            ):
                samples.append(sample)
                if len(samples) >= max_samples:
                    break
        if len(samples) >= max_samples:
            break
        if progress_every_replays > 0 and replay_idx % int(progress_every_replays) == 0:
            print(
                "support_replay_scan_progress "
                f"replays={replay_idx}/{len(paths)} "
                f"scanned_moves={scanned_moves} "
                f"support_moves={support_moves} "
                f"support_disagreements={support_disagreements} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    scan_stats = {
        "source": "replay",
        "sample_mode": "support-chain-risk",
        "replays": len(paths),
        "available_replays": len(all_paths),
        "replay_start_index": int(start_index),
        "scanned_replay_paths": [str(path) for path in paths],
        "scanned_moves": int(scanned_moves),
        "support_min_tile": int(support_min_tile),
        "support_target_min_tile": int(support_target_min_tile),
        "support_mask_starter": bool(support_mask_starter),
        "support_min_delta": float(support_min_delta),
        "support_moves": int(support_moves),
        "support_move_rate": float(support_moves / scanned_moves) if scanned_moves else 0.0,
        "support_disagreements": int(support_disagreements),
        "support_disagreement_rate": float(support_disagreements / support_moves) if support_moves else 0.0,
        "low_margin_moves": int(low_margin_moves),
        "low_margin_move_rate": float(low_margin_moves / scanned_moves) if scanned_moves else 0.0,
        "accepted_samples": len(samples),
        "accepted_scopes": len(accepted_scopes),
        "strata": dict(stratum_counts),
        "rejected": dict(rejected),
    }
    if action_value_cache is not None:
        scan_stats.update(
            {
                "action_value_cache_hits": int(cache_stats["action_value_cache_hits"]),
                "action_value_cache_misses": int(cache_stats["action_value_cache_misses"]),
                "action_value_cache_entries": len(action_value_cache),
            }
        )

    return {
        "samples": samples,
        "scan_stats": scan_stats,
    }


def collect_support_chain_risk_states_from_state_records(
    *,
    base_policy: object,
    base_policy_spec: str,
    state_records: Iterable[dict[str, Any]],
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-stratum",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    support_min_tile: int = 768,
    support_target_min_tile: int = 3072,
    support_mask_starter: bool = True,
    support_min_delta: float = 50.0,
    max_state_records: int = 0,
    state_start_index: int = 0,
    progress_every_state_records: int = 0,
    action_value_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_records = 0
    support_records = 0
    support_disagreements = 0
    low_margin_records = 0
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    all_records, records = _state_record_slice(
        state_records,
        start_index=state_start_index,
        max_records=max_state_records,
    )

    for record_idx, record in enumerate(records, start=1):
        state_payload_obj = record.get("state")
        if not isinstance(state_payload_obj, dict):
            rejected["bad_state_record"] += 1
            continue
        seed = _record_seed(record, record_idx)
        starter_tile = _record_starter_tile(record)
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        base_rng = np.random.default_rng(seed + POLICY_RNG_OFFSET + record_idx * 7919)
        state = state_from_payload(state_payload_obj)
        if state.game_over or not sim.legal_actions(state):
            rejected["terminal_or_no_legal"] += 1
            continue
        scanned_records += 1
        features = state_features(state, sim, starter_tile)
        filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
        if filter_reason is not None:
            rejected[filter_reason] += 1
            continue
        if int(features["max_tile_excl_starter"]) >= int(support_target_min_tile):
            support_records += 1
        rows = _cached_action_value_rows(
            base_policy,
            base_policy_spec,
            state,
            sim,
            action_value_cache,
            cache_stats,
        )
        if len(rows) < 2:
            rejected["no_action_values"] += 1
            continue
        actor_action = select_action_from_rows(base_policy, rows, base_rng)
        sample, reason = _support_chain_risk_candidate(
            base_policy_spec=base_policy_spec,
            state=state,
            sim=sim,
            rows=rows,
            seed=seed,
            starter_tile=starter_tile,
            actor_action=int(actor_action),
            margin_threshold=margin_threshold,
            support_min_tile=support_min_tile,
            support_target_min_tile=support_target_min_tile,
            support_mask_starter=support_mask_starter,
            support_min_delta=support_min_delta,
            source_replay=_record_source_replay(record),
        )
        if sample is None:
            rejected[reason] += 1
            if reason == "margin_filter":
                support_disagreements += 1
            continue
        _annotate_sample_from_state_record(sample, record)
        support_disagreements += 1
        low_margin_records += 1
        if _accept_candidate_sample(
            sample=sample,
            comparison_spec="support_chain_best",
            first_per=first_per,
            stratum_counts=stratum_counts,
            accepted_scopes=accepted_scopes,
            seen_state_top_two=seen_state_top_two,
            max_samples=max_samples,
            max_per_stratum=max_per_stratum,
            current_sample_count=len(samples),
            rejected=rejected,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
        ):
            samples.append(sample)
            if len(samples) >= max_samples:
                break
        if progress_every_state_records > 0 and record_idx % int(progress_every_state_records) == 0:
            print(
                "support_state_scan_progress "
                f"records={record_idx}/{len(records)} "
                f"scanned_records={scanned_records} "
                f"support_records={support_records} "
                f"support_disagreements={support_disagreements} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    scan_stats = {
        "source": "state_records",
        "sample_mode": "support-chain-risk",
        "state_records": len(records),
        "available_state_records": len(all_records),
        "state_start_index": int(max(0, state_start_index)),
        "scanned_records": int(scanned_records),
        "support_min_tile": int(support_min_tile),
        "support_target_min_tile": int(support_target_min_tile),
        "support_mask_starter": bool(support_mask_starter),
        "support_min_delta": float(support_min_delta),
        "support_records": int(support_records),
        "support_record_rate": float(support_records / scanned_records) if scanned_records else 0.0,
        "support_disagreements": int(support_disagreements),
        "support_disagreement_rate": float(support_disagreements / support_records) if support_records else 0.0,
        "low_margin_records": int(low_margin_records),
        "low_margin_record_rate": float(low_margin_records / scanned_records) if scanned_records else 0.0,
        "accepted_samples": len(samples),
        "accepted_scopes": len(accepted_scopes),
        "strata": dict(stratum_counts),
        "rejected": dict(rejected),
    }
    if action_value_cache is not None:
        scan_stats.update(
            {
                "action_value_cache_hits": int(cache_stats["action_value_cache_hits"]),
                "action_value_cache_misses": int(cache_stats["action_value_cache_misses"]),
                "action_value_cache_entries": len(action_value_cache),
            }
        )

    return {
        "samples": samples,
        "scan_stats": scan_stats,
    }


def collect_top_two_states_from_state_records(
    *,
    base_policy: object,
    base_policy_spec: str,
    state_records: Iterable[dict[str, Any]],
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-stratum",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    max_state_records: int = 0,
    state_start_index: int = 0,
    progress_every_state_records: int = 0,
    action_value_cache: dict[str, list[dict[str, Any]]] | None = None,
    min_top_value: float = 0.0,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_records = 0
    low_margin_records = 0
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    all_records, records = _state_record_slice(
        state_records,
        start_index=state_start_index,
        max_records=max_state_records,
    )

    for record_idx, record in enumerate(records, start=1):
        state_payload_obj = record.get("state")
        if not isinstance(state_payload_obj, dict):
            rejected["bad_state_record"] += 1
            continue
        seed = _record_seed(record, record_idx)
        starter_tile = _record_starter_tile(record)
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        state = state_from_payload(state_payload_obj)
        if state.game_over or not sim.legal_actions(state):
            rejected["terminal_or_no_legal"] += 1
            continue
        scanned_records += 1
        features = state_features(state, sim, starter_tile)
        filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
        if filter_reason is not None:
            rejected[filter_reason] += 1
            continue
        rows = _cached_action_value_rows(
            base_policy,
            base_policy_spec,
            state,
            sim,
            action_value_cache,
            cache_stats,
        )
        if len(rows) < 2:
            rejected["no_action_values"] += 1
            continue
        margin, norm = normalized_margin(rows)
        if norm > float(margin_threshold):
            rejected["margin_filter"] += 1
            continue
        top_value = float(rows[0]["value"])
        if top_value < float(min_top_value):
            rejected["top_value_filter"] += 1
            continue
        low_margin_records += 1
        top_two = rows[:2]
        top_two_names = [str(row["name"]) for row in top_two]
        sample = {
            "id": safe_name(
                f"top2_seed{int(seed)}_move{int(state.move_count)}_{top_two_names[0]}_vs_{top_two_names[1]}_{features['stratum']}",
                max_length=96,
            ),
            "sample_mode": "top-two",
            "seed": int(seed),
            "starter_tile": starter_tile,
            "move_count": int(state.move_count),
            "source_replay": _record_source_replay(record),
            "base_policy": base_policy_spec,
            "comparison_policy": "top_two_second",
            "base_action": top_two_names[0],
            "comparison_action": top_two_names[1],
            "top_two_actions": top_two_names,
            "top_two_values": [float(row["value"]) for row in top_two],
            "action_values": rows,
            "margin": float(margin),
            "normalized_margin": float(norm),
            "features": features,
            "state": state_payload_obj,
        }
        _annotate_sample_from_state_record(sample, record)
        if _accept_candidate_sample(
            sample=sample,
            comparison_spec="top_two_second",
            first_per=first_per,
            stratum_counts=stratum_counts,
            accepted_scopes=accepted_scopes,
            seen_state_top_two=seen_state_top_two,
            max_samples=max_samples,
            max_per_stratum=max_per_stratum,
            current_sample_count=len(samples),
            rejected=rejected,
            phase_filter=phase_filter,
            corner_risk_filter=corner_risk_filter,
        ):
            samples.append(sample)
            if len(samples) >= max_samples:
                break
        if progress_every_state_records > 0 and record_idx % int(progress_every_state_records) == 0:
            print(
                "top_two_state_scan_progress "
                f"records={record_idx}/{len(records)} "
                f"scanned_records={scanned_records} "
                f"low_margin={low_margin_records} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    scan_stats = {
        "source": "state_records",
        "sample_mode": "top-two",
        "state_records": len(records),
        "available_state_records": len(all_records),
        "state_start_index": int(max(0, state_start_index)),
        "scanned_records": int(scanned_records),
        "low_margin_records": int(low_margin_records),
        "low_margin_record_rate": float(low_margin_records / scanned_records) if scanned_records else 0.0,
        "min_top_value": float(min_top_value),
        "accepted_samples": len(samples),
        "accepted_scopes": len(accepted_scopes),
        "strata": dict(stratum_counts),
        "rejected": dict(rejected),
    }
    if action_value_cache is not None:
        scan_stats.update(
            {
                "action_value_cache_hits": int(cache_stats["action_value_cache_hits"]),
                "action_value_cache_misses": int(cache_stats["action_value_cache_misses"]),
                "action_value_cache_entries": len(action_value_cache),
            }
        )

    return {
        "samples": samples,
        "scan_stats": scan_stats,
    }


def collect_swing_states(
    *,
    base_policy: object,
    base_policy_spec: str,
    comparison_policies: list[tuple[str, object]],
    seeds: Iterable[int],
    starter_tile: int | None,
    max_moves: int,
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-policy",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    progress_every_seeds: int = 0,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_moves = 0
    low_margin_moves = 0
    disagreements = 0
    accepted_seed_policy: set[tuple[int, str]] = set()
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()

    seed_list = list(seeds)
    for seed_idx, seed in enumerate(seed_list, start=1):
        sim = ThreesSim(np.random.default_rng(int(seed)), starter_tile=starter_tile)
        state = sim.reset()
        base_rng = np.random.default_rng(int(seed) + POLICY_RNG_OFFSET)
        comparison_rngs = {
            spec: np.random.default_rng(int(seed) + COMPARISON_RNG_OFFSET + idx * 7919)
            for idx, (spec, _policy) in enumerate(comparison_policies)
        }
        for _move_idx in range(max_moves):
            if state.game_over or not sim.legal_actions(state):
                break
            scanned_moves += 1
            rows = _action_value_rows(base_policy, state, sim)
            if not rows:
                base_action = int(base_policy(state, sim, base_rng))
            else:
                base_action = select_action_from_rows(base_policy, rows, base_rng)
            if len(rows) >= 2:
                _margin, norm = normalized_margin(rows)
                if norm <= margin_threshold:
                    low_margin_moves += 1
                    for comparison_spec, comparison_policy in comparison_policies:
                        comparison_action = int(comparison_policy(state, sim, comparison_rngs[comparison_spec]))
                        if comparison_action == base_action:
                            continue
                        disagreements += 1
                        features = state_features(state, sim, starter_tile)
                        filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
                        if filter_reason is not None:
                            rejected[filter_reason] += 1
                            continue
                        scope_key = sample_scope_key(first_per, int(seed), comparison_spec, features)
                        if scope_key is not None and scope_key in accepted_scopes:
                            rejected["scope_seen"] += 1
                            continue
                        stratum = str(features["stratum"])
                        if stratum_counts[stratum] >= max_per_stratum:
                            rejected["bucket_full"] += 1
                            continue
                        if len(samples) >= max_samples:
                            rejected["global_full"] += 1
                            continue
                        top_two = rows[:2]
                        top_two_names = [str(row["name"]) for row in top_two]
                        state_key = state_top_two_key(state, top_two_names)
                        if state_key in seen_state_top_two:
                            rejected["duplicate_state"] += 1
                            continue
                        sample_id = safe_name(
                            f"{comparison_spec}_seed{int(seed)}_move{int(state.move_count)}_{stratum}",
                            max_length=96,
                        )
                        sample = {
                            "id": sample_id,
                            "seed": int(seed),
                            "starter_tile": starter_tile,
                            "move_count": int(state.move_count),
                            "base_policy": base_policy_spec,
                            "comparison_policy": comparison_spec,
                            "base_action": DIRECTION_NAMES[base_action],
                            "comparison_action": DIRECTION_NAMES[comparison_action],
                            "top_two_actions": top_two_names,
                            "top_two_values": [float(row["value"]) for row in top_two],
                            "action_values": rows,
                            "margin": float(_margin),
                            "normalized_margin": float(norm),
                            "features": features,
                            "state": state_payload(state, sim),
                        }
                        samples.append(sample)
                        seen_state_top_two.add(state_key)
                        stratum_counts[stratum] += 1
                        if scope_key is not None:
                            accepted_scopes.add(scope_key)
                        accepted_seed_policy.add((int(seed), comparison_spec))
                        if len(samples) >= max_samples:
                            break
                    if len(samples) >= max_samples:
                        break
            state = _step_with_fallback(sim, state, base_action)
        if len(samples) >= max_samples:
            break
        if progress_every_seeds > 0 and seed_idx % int(progress_every_seeds) == 0:
            print(
                "scan_progress "
                f"seeds={seed_idx}/{len(seed_list)} "
                f"scanned_moves={scanned_moves} "
                f"low_margin={low_margin_moves} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    return {
        "samples": samples,
        "scan_stats": {
            "scanned_moves": int(scanned_moves),
            "low_margin_moves": int(low_margin_moves),
            "low_margin_move_rate": float(low_margin_moves / scanned_moves) if scanned_moves else 0.0,
            "disagreements": int(disagreements),
            "accepted_samples": len(samples),
            "accepted_seed_policy_pairs": len(accepted_seed_policy),
            "accepted_scopes": len(accepted_scopes),
            "strata": dict(stratum_counts),
            "rejected": dict(rejected),
        },
    }


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _glob_paths(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path().glob(pattern))
    return sorted(path for path in paths if path.is_file())


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)


def _load_state_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records: object
    if isinstance(payload, dict):
        records = payload.get("records", [])
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain a records list")
    out: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        state_payload_obj = record.get("state")
        if not isinstance(state_payload_obj, dict):
            continue
        out.append(
            {
                **record,
                "_source_json": str(path),
                "_record_index": int(idx),
            }
        )
    return out


def _load_state_records_from_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_load_state_records(Path(path)))
    return records


def _record_source_replay(record: dict[str, Any]) -> str:
    source = record.get("source_replay") or record.get("replay") or record.get("_source_json")
    return str(source)


def _record_frame_index(record: dict[str, Any]) -> int:
    return _int_or_default(
        record.get("source_frame_index", record.get("frame_index", record.get("_record_index", 0))),
        0,
    )


def _record_seed(record: dict[str, Any], fallback: int) -> int:
    return _int_or_default(record.get("seed", record.get("source_seed")), fallback)


def _record_starter_tile(record: dict[str, Any], default: int | None = 1536) -> int | None:
    value = record.get("starter_tile", default)
    if value is None:
        return None
    return int(value)


def _annotate_sample_from_state_record(sample: dict[str, Any], record: dict[str, Any]) -> None:
    sample["source_json"] = record.get("_source_json")
    sample["source_record_index"] = int(record.get("_record_index", 0))
    sample["source_record_id"] = record.get("id")
    sample["source_frame_index"] = _record_frame_index(record)
    for key in (
        "kind",
        "target_tile",
        "outcome",
        "control_tile",
        "moves_to_promotion",
        "moves_to_terminal",
        "window_start_position",
        "window_end_position",
        "promotion_frame_position",
        "terminal_frame_position",
        "source_next_action",
    ):
        if key in record:
            sample[key] = record.get(key)


def _state_record_slice(
    state_records: Iterable[dict[str, Any]],
    *,
    start_index: int,
    max_records: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_records = list(state_records)
    start = max(0, int(start_index))
    candidates = all_records[start:]
    records = candidates[: int(max_records)] if int(max_records) > 0 else candidates
    return all_records, records


def collect_swing_states_from_replays(
    *,
    base_policy: object,
    base_policy_spec: str,
    comparison_policies: list[tuple[str, object]],
    replay_paths: Iterable[Path],
    margin_threshold: float,
    max_samples: int,
    max_per_stratum: int,
    first_per: str = "seed-policy",
    replay_base_action: str = "recorded",
    phase_filter: set[str] | None = None,
    corner_risk_filter: set[str] | None = None,
    max_replays: int = 0,
    replay_start_index: int = 0,
    progress_every_replays: int = 0,
    action_value_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if replay_base_action not in ("recorded", "policy"):
        raise ValueError(f"Unsupported replay_base_action {replay_base_action!r}")
    samples: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    scanned_moves = 0
    low_margin_moves = 0
    disagreements = 0
    accepted_seed_policy: set[tuple[int, str]] = set()
    accepted_scopes: set[tuple[Any, ...]] = set()
    seen_state_top_two: set[str] = set()
    rejected: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    all_paths = list(replay_paths)
    start_index = max(0, int(replay_start_index))
    candidate_paths = all_paths[start_index:]
    paths = candidate_paths[: int(max_replays)] if int(max_replays) > 0 else candidate_paths

    for replay_idx, replay_path in enumerate(paths, start=1):
        replay = json.loads(Path(replay_path).read_text())
        seed = int(replay.get("seed", replay_idx))
        starter_tile = replay.get("starter_tile", 1536)
        frames = replay.get("frames", [])
        if not isinstance(frames, list):
            rejected["bad_replay"] += 1
            continue
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        comparison_rngs = {
            spec: np.random.default_rng(seed + COMPARISON_RNG_OFFSET + idx * 7919)
            for idx, (spec, _policy) in enumerate(comparison_policies)
        }
        base_rng = np.random.default_rng(seed + POLICY_RNG_OFFSET)

        for before_idx in range(max(0, len(frames) - 1)):
            before_frame = frames[before_idx]
            after_frame = frames[before_idx + 1]
            if not isinstance(before_frame, dict) or not isinstance(after_frame, dict):
                continue
            state_payload_obj = before_frame.get("state")
            if not isinstance(state_payload_obj, dict):
                continue
            move = after_frame.get("move")
            if not isinstance(move, dict) or move.get("action") is None:
                continue
            state = state_from_payload(state_payload_obj)
            if state.game_over or not sim.legal_actions(state):
                continue
            scanned_moves += 1
            features = state_features(state, sim, starter_tile)
            filter_reason = _filter_accepts(features, phase_filter, corner_risk_filter)
            if filter_reason is not None:
                rejected[filter_reason] += 1
                continue
            rows = _cached_action_value_rows(
                base_policy,
                base_policy_spec,
                state,
                sim,
                action_value_cache,
                cache_stats,
            )
            if len(rows) < 2:
                continue
            margin, norm = normalized_margin(rows)
            if norm > margin_threshold:
                continue
            low_margin_moves += 1
            if replay_base_action == "policy":
                base_action = select_action_from_rows(base_policy, rows, base_rng)
                base_action_name = DIRECTION_NAMES[base_action]
            else:
                base_action_name = str(move["action"])
                try:
                    base_action = direction_index(base_action_name)
                except ValueError:
                    rejected["bad_action"] += 1
                    continue
            top_two = rows[:2]
            top_two_names = [str(row["name"]) for row in top_two]
            if base_action_name not in top_two_names:
                rejected["actor_not_top_two"] += 1
                continue
            for comparison_spec, comparison_policy in comparison_policies:
                comparison_action = int(comparison_policy(state, sim, comparison_rngs[comparison_spec]))
                comparison_action_name = DIRECTION_NAMES[comparison_action]
                if comparison_action == base_action:
                    continue
                disagreements += 1
                if first_per in ("replay", "replay-phase", "replay-stratum"):
                    scope_key = candidate_scope_key(first_per, seed, comparison_spec, features, str(replay_path))
                else:
                    seed_scope_key = sample_scope_key(first_per, seed, comparison_spec, features)
                    scope_key = ("replay", str(replay_path), *seed_scope_key) if seed_scope_key is not None else None
                if scope_key is not None:
                    if scope_key in accepted_scopes:
                        rejected["scope_seen"] += 1
                        continue
                stratum = str(features["stratum"])
                if stratum_counts[stratum] >= max_per_stratum:
                    rejected["bucket_full"] += 1
                    continue
                if len(samples) >= max_samples:
                    rejected["global_full"] += 1
                    continue
                state_key = state_top_two_key(state, top_two_names)
                if state_key in seen_state_top_two:
                    rejected["duplicate_state"] += 1
                    continue
                sample_id = safe_name(
                    f"replay_seed{seed}_move{int(state.move_count)}_{comparison_spec}_{stratum}",
                    max_length=96,
                )
                sample = {
                    "id": sample_id,
                    "seed": seed,
                    "starter_tile": starter_tile,
                    "move_count": int(state.move_count),
                    "source_replay": str(replay_path),
                    "base_policy": base_policy_spec,
                    "comparison_policy": comparison_spec,
                    "base_action": base_action_name,
                    "comparison_action": comparison_action_name,
                    "top_two_actions": top_two_names,
                    "top_two_values": [float(row["value"]) for row in top_two],
                    "action_values": rows,
                    "margin": float(margin),
                    "normalized_margin": float(norm),
                    "features": features,
                    "state": state_payload_obj,
                }
                samples.append(sample)
                seen_state_top_two.add(state_key)
                stratum_counts[stratum] += 1
                if scope_key is not None:
                    accepted_scopes.add(scope_key)
                accepted_seed_policy.add((seed, comparison_spec))
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break
        if progress_every_replays > 0 and replay_idx % int(progress_every_replays) == 0:
            print(
                "replay_scan_progress "
                f"replays={replay_idx}/{len(paths)} "
                f"scanned_moves={scanned_moves} "
                f"low_margin={low_margin_moves} "
                f"accepted={len(samples)} "
                f"strata={dict(stratum_counts)}",
                flush=True,
            )

    scan_stats = {
        "source": "replay",
        "replays": len(paths),
        "available_replays": len(all_paths),
        "replay_start_index": int(start_index),
        "scanned_replay_paths": [str(path) for path in paths],
        "scanned_moves": int(scanned_moves),
        "low_margin_moves": int(low_margin_moves),
        "low_margin_move_rate": float(low_margin_moves / scanned_moves) if scanned_moves else 0.0,
        "disagreements": int(disagreements),
        "accepted_samples": len(samples),
        "accepted_seed_policy_pairs": len(accepted_seed_policy),
        "accepted_scopes": len(accepted_scopes),
        "strata": dict(stratum_counts),
        "rejected": dict(rejected),
    }
    if action_value_cache is not None:
        scan_stats.update(
            {
                "action_value_cache_hits": int(cache_stats["action_value_cache_hits"]),
                "action_value_cache_misses": int(cache_stats["action_value_cache_misses"]),
                "action_value_cache_entries": len(action_value_cache),
            }
        )

    return {
        "samples": samples,
        "scan_stats": scan_stats,
    }


def _clone_state(state: SimState) -> SimState:
    return SimState(
        board=np.asarray(state.board, dtype=np.int32).copy(),
        preview=state.preview,
        small_counts=state.small_counts.copy(),
        small_pos=int(state.small_pos),
        small_seen_total=int(state.small_seen_total),
        span_small_pos=int(state.span_small_pos),
        large_pending=bool(state.large_pending),
        max_tile=int(state.max_tile),
        move_count=int(state.move_count),
        game_over=bool(state.game_over),
    )


def rollout_action(
    *,
    state: SimState,
    first_action: int,
    policy: object,
    starter_tile: int | None,
    horizons: tuple[int, ...],
    sim_seed: int,
    policy_seed: int,
) -> dict[int, int]:
    metrics = rollout_action_metrics(
        state=state,
        first_action=first_action,
        policy=policy,
        starter_tile=starter_tile,
        horizons=horizons,
        sim_seed=sim_seed,
        policy_seed=policy_seed,
    )
    return {int(horizon): int(row["score_delta"]) for horizon, row in metrics.items()}


def rollout_action_metrics(
    *,
    state: SimState,
    first_action: int,
    policy: object,
    starter_tile: int | None,
    horizons: tuple[int, ...],
    sim_seed: int,
    policy_seed: int,
    target_tile: int | None = None,
) -> dict[int, dict[str, Any]]:
    sim = ThreesSim(np.random.default_rng(int(sim_seed)), starter_tile=starter_tile)
    policy_rng = np.random.default_rng(int(policy_seed))
    current = _clone_state(state)
    start_score = score_board(current.board)
    horizon_set = set(int(horizon) for horizon in horizons)
    max_horizon = max(horizon_set)
    values: dict[int, dict[str, Any]] = {}

    def current_metrics() -> dict[str, Any]:
        built_max = max_tile_excluding_initial_starter(current.board, starter_tile)
        row: dict[str, Any] = {
            "score_delta": int(score_board(current.board) - start_score),
            "max_tile": int(np.asarray(current.board, dtype=np.int32).max(initial=0)),
            "max_tile_excl_starter": int(built_max),
        }
        if target_tile is not None:
            row["target_tile"] = int(target_tile)
            row["target_reached"] = bool(int(built_max) >= int(target_tile))
        return row

    for step_idx in range(max_horizon):
        if current.game_over or not sim.legal_actions(current):
            break
        action = int(first_action) if step_idx == 0 else int(policy(current, sim, policy_rng))
        current = _step_with_fallback(sim, current, action)
        completed = step_idx + 1
        if completed in horizon_set:
            values[completed] = current_metrics()
    final_metrics = current_metrics()
    for horizon in horizons:
        values.setdefault(int(horizon), dict(final_metrics))
    return values


def _bootstrap_winner_fraction(diffs: list[float], winner_sign: int, rng: np.random.Generator, resamples: int = 200) -> float:
    if not diffs or winner_sign == 0:
        return 0.5
    arr = np.asarray(diffs, dtype=np.float64)
    kept = 0
    for _ in range(int(resamples)):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        sign = 1 if float(sample.mean()) > 0 else -1 if float(sample.mean()) < 0 else 0
        if sign == winner_sign:
            kept += 1
    return kept / float(resamples)


def _sample_offset(sample: dict[str, Any]) -> int:
    digest = hashlib.sha1(str(sample.get("id", "")).encode("utf-8")).hexdigest()
    return int(digest[:10], 16) % 1_000_000


def _top_two_names(sample: dict[str, Any]) -> tuple[str, str]:
    if len(sample.get("top_two_actions", [])) < 2:
        raise ValueError(f"Sample {sample.get('id')} does not have two top actions")
    return tuple(str(name) for name in sample["top_two_actions"][:2])  # type: ignore[return-value]


def _empty_by_action(names: tuple[str, str], horizons: tuple[int, ...]) -> dict[str, dict[int, list[int]]]:
    return {
        name: {int(horizon): [] for horizon in horizons}
        for name in names
    }


def _empty_metrics_by_action(names: tuple[str, str], horizons: tuple[int, ...]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    return {
        name: {int(horizon): [] for horizon in horizons}
        for name in names
    }


def _score_delta_from_metric(value: object) -> int:
    if isinstance(value, dict):
        return int(value.get("score_delta", 0))
    return int(value)


def _metric_from_value(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"score_delta": int(value)}


def _score_by_action_from_metrics(
    metrics_by_action: dict[str, dict[int, list[dict[str, Any]]]],
    horizons: tuple[int, ...],
) -> dict[str, dict[int, list[int]]]:
    return {
        action_name: {
            int(horizon): [_score_delta_from_metric(value) for value in horizon_values.get(int(horizon), [])]
            for horizon in horizons
        }
        for action_name, horizon_values in metrics_by_action.items()
    }


def _merge_by_action(target: dict[str, dict[int, list[Any]]], chunk: dict[str, dict[int, list[Any]]]) -> None:
    for action_name, horizon_values in chunk.items():
        for horizon, values in horizon_values.items():
            target[action_name][int(horizon)].extend(_metric_from_value(value) for value in values)


def label_sample_repeats(
    sample: dict[str, Any],
    *,
    policy: object,
    repeat_indices: Iterable[int],
    horizons: tuple[int, ...],
    label_seed: int,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    state = state_from_payload(sample["state"])
    names = _top_two_names(sample)
    actions = tuple(direction_index(name) for name in names)
    starter_tile = sample.get("starter_tile")
    target_tile = sample.get("target_tile")
    if target_tile is not None:
        target_tile = int(target_tile)
    by_action = _empty_metrics_by_action(names, horizons)

    sample_offset = _sample_offset(sample)
    for repeat_idx in repeat_indices:
        sim_seed = int(label_seed) + sample_offset + repeat_idx * 10_007
        policy_seed = sim_seed + ROLLOUT_POLICY_RNG_OFFSET
        for action, name in zip(actions, names):
            metrics = rollout_action_metrics(
                state=state,
                first_action=action,
                policy=policy,
                starter_tile=starter_tile,
                horizons=horizons,
                sim_seed=sim_seed,
                policy_seed=policy_seed,
                target_tile=target_tile,
            )
            for horizon, row in metrics.items():
                by_action[name][int(horizon)].append(dict(row))
    return by_action


def finish_label_sample(
    sample: dict[str, Any],
    *,
    by_action: dict[str, dict[int, list[int]]],
    metrics_by_action: dict[str, dict[int, list[dict[str, Any]]]] | None = None,
    repeats: int,
    horizons: tuple[int, ...],
    stability_threshold: float,
) -> dict[str, Any]:
    names = _top_two_names(sample)
    target_tile = sample.get("target_tile")
    if target_tile is not None:
        target_tile = int(target_tile)

    horizon_rows = []
    bootstrap_rng = np.random.default_rng(_sample_offset(sample) + 99_991)
    for horizon in horizons:
        left = by_action[names[0]][int(horizon)]
        right = by_action[names[1]][int(horizon)]
        diffs = [float(a - b) for a, b in zip(left, right)]
        mean_left = mean(left) if left else 0.0
        mean_right = mean(right) if right else 0.0
        if mean_left > mean_right:
            winner = names[0]
            winner_sign = 1
        elif mean_right > mean_left:
            winner = names[1]
            winner_sign = -1
        else:
            winner = "tie"
            winner_sign = 0
        paired_win_rate = (
            sum(1 for diff in diffs if (diff > 0 if winner_sign == 1 else diff < 0))
            / float(len(diffs))
            if diffs and winner_sign != 0
            else 0.5
        )
        row = {
            "horizon": int(horizon),
            "winner": winner,
            "mean_delta": {names[0]: float(mean_left), names[1]: float(mean_right)},
            "median_delta": {
                names[0]: float(median(left)) if left else 0.0,
                names[1]: float(median(right)) if right else 0.0,
            },
            "mean_diff_first_minus_second": float(mean_left - mean_right),
            "paired_win_rate": float(paired_win_rate),
            "bootstrap_winner_fraction": float(
                _bootstrap_winner_fraction(diffs, winner_sign, bootstrap_rng)
            ),
        }
        if target_tile is not None and metrics_by_action is not None:
            promotion_rates: dict[str, float] = {}
            promotion_counts: dict[str, int] = {}
            mean_max_tiles: dict[str, float] = {}
            max_tiles: dict[str, int] = {}
            for name in names:
                metrics = metrics_by_action[name][int(horizon)]
                hits = [1 if bool(item.get("target_reached")) else 0 for item in metrics]
                max_values = [int(item.get("max_tile_excl_starter", 0)) for item in metrics]
                promotion_counts[name] = int(sum(hits))
                promotion_rates[name] = float(sum(hits) / len(hits)) if hits else 0.0
                mean_max_tiles[name] = float(mean(max_values)) if max_values else 0.0
                max_tiles[name] = int(max(max_values)) if max_values else 0
            first_rate = promotion_rates[names[0]]
            second_rate = promotion_rates[names[1]]
            if first_rate > second_rate:
                promotion_winner = names[0]
            elif second_rate > first_rate:
                promotion_winner = names[1]
            else:
                promotion_winner = "tie"
            row.update(
                {
                    "target_tile": int(target_tile),
                    "promotion_count": promotion_counts,
                    "promotion_rate": promotion_rates,
                    "promotion_rate_diff_first_minus_second": float(first_rate - second_rate),
                    "promotion_winner": promotion_winner,
                    "mean_max_tile_excl_starter": mean_max_tiles,
                    "max_tile_excl_starter": max_tiles,
                }
            )
        horizon_rows.append(row)

    non_tie_winners = [row["winner"] for row in horizon_rows if row["winner"] != "tie"]
    same_winner = len(set(non_tie_winners)) == 1 and len(non_tie_winners) == len(horizon_rows)
    min_bootstrap = min((float(row["bootstrap_winner_fraction"]) for row in horizon_rows), default=0.0)
    stable = bool(same_winner and min_bootstrap >= stability_threshold)
    stable_winner = str(non_tie_winners[0]) if stable else None
    max_horizon_row = horizon_rows[-1]
    max_winner = str(max_horizon_row["winner"])
    base_action = str(sample.get("base_action"))
    mean_deltas = max_horizon_row["mean_delta"]
    oracle_winner = stable_winner or (max_winner if max_winner != "tie" else None)
    oracle_regret = None
    if oracle_winner in mean_deltas and base_action in mean_deltas:
        oracle_regret = max(0.0, float(mean_deltas[oracle_winner]) - float(mean_deltas[base_action]))

    return {
        **sample,
        "label": {
            "repeats": int(repeats),
            "horizons": [int(horizon) for horizon in horizons],
            "actions": list(names),
            "by_action": {
                action_name: {
                    str(horizon): values
                    for horizon, values in horizon_values.items()
                }
                for action_name, horizon_values in by_action.items()
            },
            "horizon_results": horizon_rows,
            "same_winner_across_horizons": bool(same_winner),
            "stable": stable,
            "stable_winner": stable_winner,
            "min_bootstrap_winner_fraction": float(min_bootstrap),
            "oracle_winner": oracle_winner,
            "oracle_regret_at_max_horizon": oracle_regret,
            **_promotion_summary_for_label(sample, max_horizon_row),
        },
    }


def _promotion_summary_for_label(sample: dict[str, Any], max_horizon_row: dict[str, Any]) -> dict[str, Any]:
    target_tile = sample.get("target_tile")
    if target_tile is None or "promotion_rate" not in max_horizon_row:
        return {}
    base_action = str(sample.get("base_action"))
    promotion_rate = max_horizon_row.get("promotion_rate", {})
    if not isinstance(promotion_rate, dict):
        return {}
    winner = str(max_horizon_row.get("promotion_winner"))
    base_rate = float(promotion_rate.get(base_action, 0.0))
    winner_rate = float(promotion_rate.get(winner, 0.0)) if winner != "tie" else base_rate
    return {
        "target_tile": int(target_tile),
        "promotion_winner_at_max_horizon": winner,
        "promotion_rate_at_max_horizon": promotion_rate,
        "promotion_count_at_max_horizon": max_horizon_row.get("promotion_count", {}),
        "promotion_rate_gain_vs_base_at_max_horizon": float(max(0.0, winner_rate - base_rate)),
    }


def label_sample(
    sample: dict[str, Any],
    *,
    policy: object,
    repeats: int,
    horizons: tuple[int, ...],
    label_seed: int,
    stability_threshold: float,
) -> dict[str, Any]:
    metrics_by_action = label_sample_repeats(
        sample,
        policy=policy,
        repeat_indices=range(int(repeats)),
        horizons=horizons,
        label_seed=label_seed,
    )
    by_action = _score_by_action_from_metrics(metrics_by_action, horizons)
    return finish_label_sample(
        sample,
        by_action=by_action,
        metrics_by_action=metrics_by_action,
        repeats=repeats,
        horizons=horizons,
        stability_threshold=stability_threshold,
    )


def label_corpus(
    samples: list[dict[str, Any]],
    *,
    policy: object,
    repeats: int,
    horizons: tuple[int, ...],
    label_seed: int,
    stability_threshold: float,
) -> list[dict[str, Any]]:
    return [
        label_sample(
            sample,
            policy=policy,
            repeats=repeats,
            horizons=horizons,
            label_seed=int(label_seed) + idx * 1_000_003,
            stability_threshold=stability_threshold,
        )
        for idx, sample in enumerate(samples)
    ]


def _label_progress_config(policy_spec: str, horizons: tuple[int, ...], label_seed: int) -> dict[str, Any]:
    return {
        "policy": policy_spec,
        "horizons": [int(horizon) for horizon in horizons],
        "label_seed": int(label_seed),
    }


def _label_sample_progress_key(sample: dict[str, Any], sample_idx: int) -> str:
    raw = json.dumps(
        {
            "sample_idx": int(sample_idx),
            "id": sample.get("id"),
            "seed": sample.get("seed"),
            "move_count": sample.get("move_count"),
            "base_action": sample.get("base_action"),
            "comparison_action": sample.get("comparison_action"),
            "top_two_actions": sample.get("top_two_actions"),
            "state": sample.get("state"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_label_progress(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": dict(config),
            "samples": {},
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a label progress object")
    existing_config = payload.get("config")
    if existing_config != config:
        raise ValueError(f"{path} was created for a different label configuration")
    samples = payload.get("samples")
    if not isinstance(samples, dict):
        payload["samples"] = {}
    return payload


def _repeat_record_complete(record: object, names: tuple[str, str], horizons: tuple[int, ...]) -> bool:
    if not isinstance(record, dict):
        return False
    for name in names:
        action_values = record.get(name)
        if not isinstance(action_values, dict):
            return False
        for horizon in horizons:
            if str(int(horizon)) not in action_values:
                return False
    return True


def _completed_repeat_indices(
    repeat_results: dict[str, Any],
    names: tuple[str, str],
    horizons: tuple[int, ...],
    repeats: int,
) -> set[int]:
    return {
        int(idx)
        for idx in range(int(repeats))
        if _repeat_record_complete(repeat_results.get(str(idx)), names, horizons)
    }


def _store_repeat_chunk(
    repeat_results: dict[str, Any],
    repeat_indices: list[int],
    chunk: dict[str, dict[int, list[Any]]],
) -> None:
    for pos, repeat_idx in enumerate(repeat_indices):
        record: dict[str, dict[str, Any]] = {}
        for action_name, horizon_values in chunk.items():
            record[action_name] = {}
            for horizon, values in horizon_values.items():
                record[action_name][str(int(horizon))] = _metric_from_value(values[pos])
        repeat_results[str(int(repeat_idx))] = record


def _by_action_from_repeat_results(
    sample: dict[str, Any],
    repeat_results: dict[str, Any],
    repeats: int,
    horizons: tuple[int, ...],
) -> dict[str, dict[int, list[int]]]:
    names = _top_two_names(sample)
    by_action = _empty_by_action(names, horizons)
    for repeat_idx in range(int(repeats)):
        record = repeat_results.get(str(repeat_idx))
        if not _repeat_record_complete(record, names, horizons):
            raise ValueError(f"Sample {sample.get('id')} is missing repeat {repeat_idx}")
        assert isinstance(record, dict)
        for name in names:
            action_values = record[name]
            for horizon in horizons:
                by_action[name][int(horizon)].append(_score_delta_from_metric(action_values[str(int(horizon))]))
    return by_action


def _metrics_by_action_from_repeat_results(
    sample: dict[str, Any],
    repeat_results: dict[str, Any],
    repeats: int,
    horizons: tuple[int, ...],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    names = _top_two_names(sample)
    by_action = _empty_metrics_by_action(names, horizons)
    for repeat_idx in range(int(repeats)):
        record = repeat_results.get(str(repeat_idx))
        if not _repeat_record_complete(record, names, horizons):
            raise ValueError(f"Sample {sample.get('id')} is missing repeat {repeat_idx}")
        assert isinstance(record, dict)
        for name in names:
            action_values = record[name]
            for horizon in horizons:
                by_action[name][int(horizon)].append(_metric_from_value(action_values[str(int(horizon))]))
    return by_action


def label_corpus_resumable(
    samples: list[dict[str, Any]],
    *,
    policy: object,
    policy_spec: str,
    repeats: int,
    horizons: tuple[int, ...],
    label_seed: int,
    stability_threshold: float,
    progress_path: Path,
    repeat_chunk_size: int = 1,
    progress_every_chunks: int = 0,
) -> list[dict[str, Any]]:
    config = _label_progress_config(policy_spec, horizons, label_seed)
    progress = _load_label_progress(progress_path, config)
    progress_samples = progress.setdefault("samples", {})
    if not isinstance(progress_samples, dict):
        raise ValueError(f"{progress_path} has invalid label samples")

    labels: list[dict[str, Any]] = []
    chunk_size = max(1, int(repeat_chunk_size))
    completed_chunks = 0
    for sample_idx, sample in enumerate(samples):
        names = _top_two_names(sample)
        key = _label_sample_progress_key(sample, sample_idx)
        entry = progress_samples.setdefault(
            key,
            {
                "id": sample.get("id"),
                "sample_idx": int(sample_idx),
                "repeat_results": {},
            },
        )
        if not isinstance(entry, dict):
            raise ValueError(f"{progress_path} has invalid progress entry for {sample.get('id')}")
        repeat_results = entry.setdefault("repeat_results", {})
        if not isinstance(repeat_results, dict):
            raise ValueError(f"{progress_path} has invalid repeat results for {sample.get('id')}")

        sample_label_seed = int(label_seed) + sample_idx * 1_000_003
        completed = _completed_repeat_indices(repeat_results, names, horizons, repeats)
        for repeat_start in range(0, int(repeats), chunk_size):
            repeat_stop = min(int(repeats), repeat_start + chunk_size)
            repeat_indices = [
                idx
                for idx in range(repeat_start, repeat_stop)
                if idx not in completed
            ]
            if not repeat_indices:
                continue
            chunk = label_sample_repeats(
                sample,
                policy=policy,
                repeat_indices=repeat_indices,
                horizons=horizons,
                label_seed=sample_label_seed,
            )
            _store_repeat_chunk(repeat_results, repeat_indices, chunk)
            completed.update(repeat_indices)
            entry["completed_repeats"] = sorted(int(idx) for idx in completed)
            entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            progress["updated_at"] = entry["updated_at"]
            write_json(progress_path, progress)
            completed_chunks += 1
            if progress_every_chunks > 0 and completed_chunks % int(progress_every_chunks) == 0:
                print(
                    "label_checkpoint_progress "
                    f"samples={sample_idx + 1}/{len(samples)} "
                    f"sample_id={sample.get('id')} "
                    f"completed_repeats={len(completed)}/{int(repeats)} "
                    f"chunks={completed_chunks} "
                    f"path={progress_path}",
                    flush=True,
                )

        metrics_by_action = _metrics_by_action_from_repeat_results(sample, repeat_results, repeats, horizons)
        by_action = _score_by_action_from_metrics(metrics_by_action, horizons)
        labels.append(
            finish_label_sample(
                sample,
                by_action=by_action,
                metrics_by_action=metrics_by_action,
                repeats=repeats,
                horizons=horizons,
                stability_threshold=stability_threshold,
            )
        )

    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(progress_path, progress)
    return labels


def _init_label_worker(policy_spec: str) -> None:
    global _WORKER_POLICY
    _WORKER_POLICY = make_policy(policy_spec)


def _label_repeat_chunk_worker(args: tuple[int, dict[str, Any], list[int], tuple[int, ...], int]) -> tuple[int, list[int], dict[str, dict[int, list[dict[str, Any]]]]]:
    sample_idx, sample, repeat_indices, horizons, label_seed = args
    if _WORKER_POLICY is None:
        raise RuntimeError("Label worker policy was not initialized")
    return sample_idx, list(repeat_indices), label_sample_repeats(
        sample,
        policy=_WORKER_POLICY,
        repeat_indices=[int(idx) for idx in repeat_indices],
        horizons=horizons,
        label_seed=label_seed,
    )


def label_corpus_parallel(
    samples: list[dict[str, Any]],
    *,
    policy_spec: str,
    repeats: int,
    horizons: tuple[int, ...],
    label_seed: int,
    stability_threshold: float,
    workers: int,
    repeat_chunk_size: int = 4,
    progress_every_chunks: int = 0,
) -> list[dict[str, Any]]:
    chunk_size = max(1, int(repeat_chunk_size))
    sample_label_seeds = [int(label_seed) + idx * 1_000_003 for idx, _sample in enumerate(samples)]
    merged = [
        _empty_metrics_by_action(_top_two_names(sample), horizons)
        for sample in samples
    ]
    jobs = []
    for idx, sample in enumerate(samples):
        for repeat_start in range(0, int(repeats), chunk_size):
            jobs.append(
                (
                    idx,
                    sample,
                    list(range(repeat_start, min(int(repeats), repeat_start + chunk_size))),
                    horizons,
                    sample_label_seeds[idx],
                )
            )
    expected_chunks_by_sample = [
        (int(repeats) + chunk_size - 1) // chunk_size
        for _sample in samples
    ]
    completed_chunks_by_sample = [0 for _sample in samples]
    completed_chunks = 0
    completed_samples = 0
    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        initializer=_init_label_worker,
        initargs=(policy_spec,),
    ) as executor:
        for sample_idx, _repeat_indices, chunk in executor.map(_label_repeat_chunk_worker, jobs):
            idx = int(sample_idx)
            _merge_by_action(merged[idx], chunk)
            completed_chunks += 1
            completed_chunks_by_sample[idx] += 1
            if completed_chunks_by_sample[idx] == expected_chunks_by_sample[idx]:
                completed_samples += 1
            if progress_every_chunks > 0 and completed_chunks % int(progress_every_chunks) == 0:
                print(
                    "label_parallel_progress "
                    f"chunks={completed_chunks}/{len(jobs)} "
                    f"samples={completed_samples}/{len(samples)}",
                    flush=True,
                )
    return [
        finish_label_sample(
            sample,
            by_action=_score_by_action_from_metrics(merged[idx], horizons),
            metrics_by_action=merged[idx],
            repeats=repeats,
            horizons=horizons,
            stability_threshold=stability_threshold,
        )
        for idx, sample in enumerate(samples)
    ]


def label_corpus_resumable_parallel(
    samples: list[dict[str, Any]],
    *,
    policy_spec: str,
    repeats: int,
    horizons: tuple[int, ...],
    label_seed: int,
    stability_threshold: float,
    progress_path: Path,
    workers: int,
    repeat_chunk_size: int = 4,
    progress_every_chunks: int = 0,
) -> list[dict[str, Any]]:
    config = _label_progress_config(policy_spec, horizons, label_seed)
    progress = _load_label_progress(progress_path, config)
    progress_samples = progress.setdefault("samples", {})
    if not isinstance(progress_samples, dict):
        raise ValueError(f"{progress_path} has invalid label samples")

    chunk_size = max(1, int(repeat_chunk_size))
    jobs: list[tuple[int, dict[str, Any], list[int], tuple[int, ...], int]] = []
    sample_keys: list[str] = []
    completed_by_sample: list[set[int]] = []
    for sample_idx, sample in enumerate(samples):
        names = _top_two_names(sample)
        key = _label_sample_progress_key(sample, sample_idx)
        sample_keys.append(key)
        entry = progress_samples.setdefault(
            key,
            {
                "id": sample.get("id"),
                "sample_idx": int(sample_idx),
                "repeat_results": {},
            },
        )
        if not isinstance(entry, dict):
            raise ValueError(f"{progress_path} has invalid progress entry for {sample.get('id')}")
        repeat_results = entry.setdefault("repeat_results", {})
        if not isinstance(repeat_results, dict):
            raise ValueError(f"{progress_path} has invalid repeat results for {sample.get('id')}")
        completed = _completed_repeat_indices(repeat_results, names, horizons, repeats)
        completed_by_sample.append(set(completed))
        sample_label_seed = int(label_seed) + sample_idx * 1_000_003
        for repeat_start in range(0, int(repeats), chunk_size):
            repeat_stop = min(int(repeats), repeat_start + chunk_size)
            repeat_indices = [
                idx
                for idx in range(repeat_start, repeat_stop)
                if idx not in completed
            ]
            if repeat_indices:
                jobs.append((sample_idx, sample, repeat_indices, horizons, sample_label_seed))

    completed_chunks = 0
    completed_samples = sum(1 for completed in completed_by_sample if len(completed) >= int(repeats))
    if jobs:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(
            max_workers=max(1, int(workers)),
            initializer=_init_label_worker,
            initargs=(policy_spec,),
        ) as executor:
            futures = [executor.submit(_label_repeat_chunk_worker, job) for job in jobs]
            for future in as_completed(futures):
                sample_idx, repeat_indices, chunk = future.result()
                idx = int(sample_idx)
                key = sample_keys[idx]
                entry = progress_samples[key]
                if not isinstance(entry, dict):
                    raise ValueError(f"{progress_path} has invalid progress entry for sample {idx}")
                repeat_results = entry.setdefault("repeat_results", {})
                if not isinstance(repeat_results, dict):
                    raise ValueError(f"{progress_path} has invalid repeat results for sample {idx}")
                was_complete = len(completed_by_sample[idx]) >= int(repeats)
                _store_repeat_chunk(repeat_results, [int(value) for value in repeat_indices], chunk)
                completed_by_sample[idx].update(int(value) for value in repeat_indices)
                entry["completed_repeats"] = sorted(int(value) for value in completed_by_sample[idx])
                entry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                progress["updated_at"] = entry["updated_at"]
                write_json(progress_path, progress)
                completed_chunks += 1
                if not was_complete and len(completed_by_sample[idx]) >= int(repeats):
                    completed_samples += 1
                if progress_every_chunks > 0 and completed_chunks % int(progress_every_chunks) == 0:
                    print(
                        "label_checkpoint_parallel_progress "
                        f"chunks={completed_chunks}/{len(jobs)} "
                        f"samples={completed_samples}/{len(samples)} "
                        f"path={progress_path}",
                        flush=True,
                    )

    labels: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        key = sample_keys[sample_idx]
        entry = progress_samples.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"{progress_path} is missing progress entry for {sample.get('id')}")
        repeat_results = entry.get("repeat_results")
        if not isinstance(repeat_results, dict):
            raise ValueError(f"{progress_path} has invalid repeat results for {sample.get('id')}")
        metrics_by_action = _metrics_by_action_from_repeat_results(sample, repeat_results, repeats, horizons)
        by_action = _score_by_action_from_metrics(metrics_by_action, horizons)
        labels.append(
            finish_label_sample(
                sample,
                by_action=by_action,
                metrics_by_action=metrics_by_action,
                repeats=repeats,
                horizons=horizons,
                stability_threshold=stability_threshold,
            )
        )

    progress["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(progress_path, progress)
    return labels


def summarize_labeled(labels: list[dict[str, Any]], scan_stats: dict[str, Any]) -> dict[str, Any]:
    stable = [item for item in labels if item.get("label", {}).get("stable")]
    horizon_consistent = [
        item
        for item in labels
        if item.get("label", {}).get("same_winner_across_horizons")
    ]
    regrets = [
        float(item["label"]["oracle_regret_at_max_horizon"])
        for item in labels
        if item.get("label", {}).get("oracle_regret_at_max_horizon") is not None
    ]
    positive_regrets = [value for value in regrets if value > 0.0]
    stable_positive_regrets = [
        float(item["label"]["oracle_regret_at_max_horizon"])
        for item in stable
        if float(item.get("label", {}).get("oracle_regret_at_max_horizon") or 0.0) > 0.0
    ]
    stable_oracle_disagreements = [
        item
        for item in stable
        if item.get("label", {}).get("stable_winner") is not None
        and str(item.get("label", {}).get("stable_winner")) != str(item.get("base_action"))
    ]
    winners = Counter(str(item.get("label", {}).get("stable_winner")) for item in stable)
    promotion_labels = [
        item
        for item in labels
        if item.get("label", {}).get("promotion_rate_at_max_horizon") is not None
    ]
    promotion_positive_gains = [
        float(item.get("label", {}).get("promotion_rate_gain_vs_base_at_max_horizon") or 0.0)
        for item in promotion_labels
        if float(item.get("label", {}).get("promotion_rate_gain_vs_base_at_max_horizon") or 0.0) > 0.0
    ]
    stable_promotion_positive_gains = [
        float(item.get("label", {}).get("promotion_rate_gain_vs_base_at_max_horizon") or 0.0)
        for item in stable
        if item.get("label", {}).get("promotion_rate_at_max_horizon") is not None
        and float(item.get("label", {}).get("promotion_rate_gain_vs_base_at_max_horizon") or 0.0) > 0.0
    ]
    promotion_hit_labels = 0
    for item in promotion_labels:
        rates = item.get("label", {}).get("promotion_rate_at_max_horizon", {})
        if isinstance(rates, dict) and any(float(value) > 0.0 for value in rates.values()):
            promotion_hit_labels += 1
    by_stratum: dict[str, dict[str, Any]] = {}
    for item in labels:
        features = item.get("features", {})
        stratum = str(features.get("stratum", "unknown")) if isinstance(features, dict) else "unknown"
        bucket = by_stratum.setdefault(
            stratum,
            {
                "labels": 0,
                "stable_labels": 0,
                "positive_oracle_regrets": 0,
                "mean_positive_oracle_regret": 0.0,
                "max_positive_oracle_regret": 0.0,
                "_positive_regrets": [],
            },
        )
        bucket["labels"] += 1
        if item.get("label", {}).get("stable"):
            bucket["stable_labels"] += 1
        regret = item.get("label", {}).get("oracle_regret_at_max_horizon")
        if regret is not None and float(regret) > 0.0:
            bucket["positive_oracle_regrets"] += 1
            bucket["_positive_regrets"].append(float(regret))
    for bucket in by_stratum.values():
        bucket["stable_label_rate"] = float(bucket["stable_labels"] / bucket["labels"]) if bucket["labels"] else 0.0
        positive = bucket.pop("_positive_regrets")
        if positive:
            bucket["mean_positive_oracle_regret"] = float(mean(positive))
            bucket["max_positive_oracle_regret"] = float(max(positive))
    return {
        "labels": len(labels),
        "horizon_consistent_labels": len(horizon_consistent),
        "horizon_consistent_label_rate": float(len(horizon_consistent) / len(labels)) if labels else 0.0,
        "stable_labels": len(stable),
        "stable_label_rate": float(len(stable) / len(labels)) if labels else 0.0,
        "stable_winners": dict(winners),
        "stable_oracle_disagreements": len(stable_oracle_disagreements),
        "stable_oracle_disagreement_rate": float(len(stable_oracle_disagreements) / len(stable)) if stable else 0.0,
        "oracle_positive_regrets": len(positive_regrets),
        "mean_oracle_regret": float(mean(regrets)) if regrets else 0.0,
        "median_oracle_regret": float(median(regrets)) if regrets else 0.0,
        "mean_positive_oracle_regret": float(mean(positive_regrets)) if positive_regrets else 0.0,
        "median_positive_oracle_regret": float(median(positive_regrets)) if positive_regrets else 0.0,
        "max_positive_oracle_regret": float(max(positive_regrets)) if positive_regrets else 0.0,
        "stable_positive_oracle_regrets": len(stable_positive_regrets),
        "stable_mean_positive_oracle_regret": float(mean(stable_positive_regrets)) if stable_positive_regrets else 0.0,
        "stable_max_positive_oracle_regret": float(max(stable_positive_regrets)) if stable_positive_regrets else 0.0,
        "promotion_labels": len(promotion_labels),
        "promotion_hit_labels": int(promotion_hit_labels),
        "promotion_positive_gains": len(promotion_positive_gains),
        "mean_promotion_rate_gain_vs_base": float(mean(promotion_positive_gains)) if promotion_positive_gains else 0.0,
        "max_promotion_rate_gain_vs_base": float(max(promotion_positive_gains)) if promotion_positive_gains else 0.0,
        "stable_promotion_positive_gains": len(stable_promotion_positive_gains),
        "stable_mean_promotion_rate_gain_vs_base": float(mean(stable_promotion_positive_gains))
        if stable_promotion_positive_gains
        else 0.0,
        "stable_max_promotion_rate_gain_vs_base": float(max(stable_promotion_positive_gains))
        if stable_promotion_positive_gains
        else 0.0,
        "by_stratum": by_stratum,
        "scan_stats": scan_stats,
    }


def write_report_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    labels = payload.get("labels", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for item in labels if isinstance(labels, list) else []:
        label = item.get("label", {}) if isinstance(item, dict) else {}
        features = item.get("features", {}) if isinstance(item, dict) else {}
        rows.append(
            "<tr>"
            f"<td>{cell(item.get('seed'))}</td>"
            f"<td>{cell(item.get('move_count'))}</td>"
            f"<td>{cell(features.get('stratum'))}</td>"
            f"<td>{cell(item.get('base_action'))} / {cell(item.get('comparison_action'))}</td>"
            f"<td>{cell(item.get('top_two_actions'))}</td>"
            f"<td>{float(item.get('normalized_margin', 0.0)):.6f}</td>"
            f"<td>{cell(label.get('stable_winner'))}</td>"
            f"<td>{cell(label.get('stable'))}</td>"
            f"<td>{float(label.get('oracle_regret_at_max_horizon') or 0.0):.1f}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Swing-State Labels</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101113; --panel: #191d21; --line: #364047; --ink: #f1f5f0; --muted: #a9b3ad; --gold: #e9bd4a; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 36px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 18px 0 8px; font-size: 17px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 4px; font-size: 23px; font-weight: 800; font-variant-numeric: tabular-nums; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4), th:nth-child(5), td:nth-child(5) {{ text-align: left; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; color: var(--muted); }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <main>
    <h1>Swing-State Label Report</h1>
    <p class="muted">Frozen actor, first low-normalized-margin disagreements, top-two CRN continuation labels.</p>
    <section class="cards">
      <div class="card"><div class="label">Labels</div><div class="value">{cell(summary.get('labels', 0))}</div></div>
      <div class="card"><div class="label">Stable Rate</div><div class="value">{float(summary.get('stable_label_rate', 0.0)):.0%}</div></div>
      <div class="card"><div class="label">Stable Actor Flips</div><div class="value">{cell(summary.get('stable_oracle_disagreements', 0))}</div></div>
      <div class="card"><div class="label">Stable Mean Regret</div><div class="value">{float(summary.get('stable_mean_positive_oracle_regret', 0.0)):.0f}</div></div>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Samples</h2>
      <table><thead><tr><th>Seed</th><th>Move</th><th>Stratum</th><th>Base / Other</th><th>Top Two</th><th>Norm Margin</th><th>Stable Winner</th><th>Stable</th><th>Regret</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Summary JSON</h2>
      <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def parse_starter(text: str) -> int | None:
    value = text.strip().lower()
    return None if value == "none" else int(value)


def parse_horizons(text: str) -> tuple[int, ...]:
    horizons = tuple(sorted({int(part) for part in text.split(",") if part.strip()}))
    if not horizons:
        raise ValueError("At least one horizon is required")
    return horizons


def parse_filter_values(values: list[str] | None, *, allowed: tuple[str, ...], aliases: dict[str, str], label: str) -> set[str] | None:
    if not values:
        return None
    parsed: set[str] = set()
    for value in values:
        for part in value.split(","):
            normalized = part.strip().lower()
            if not normalized:
                continue
            normalized = aliases.get(normalized, normalized)
            if normalized not in allowed:
                raise ValueError(f"Unsupported {label} filter {part!r}; allowed values are {', '.join(allowed)}")
            parsed.add(normalized)
    return parsed or None


def load_samples_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text())
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raw_samples = payload.get("labels")
    if not isinstance(raw_samples, list):
        raise ValueError(f"{path} does not contain a samples or labels list")
    samples = [item for item in raw_samples if isinstance(item, dict)]
    summary = payload.get("summary")
    scan_stats = summary.get("scan_stats") if isinstance(summary, dict) else None
    if not isinstance(scan_stats, dict):
        scan_stats = {
            "scanned_moves": None,
            "low_margin_moves": None,
            "low_margin_move_rate": None,
            "disagreements": None,
            "accepted_samples": len(samples),
            "accepted_seed_policy_pairs": None,
            "strata": dict(Counter(str(item.get("features", {}).get("stratum", "unknown")) for item in samples)),
            "rejected": {},
        }
    metadata = {
        "source_json": str(path),
        "source_base_policy": payload.get("base_policy"),
        "source_comparison_policies": payload.get("comparison_policies"),
        "source_seeds": payload.get("seeds"),
        "source_margin_threshold": payload.get("margin_threshold"),
        "source_max_samples": payload.get("max_samples"),
        "source_max_per_stratum": payload.get("max_per_stratum"),
    }
    return samples, dict(scan_stats), metadata


def run_swing_labeling(args: argparse.Namespace) -> dict[str, Any]:
    base_policy = make_policy(args.base_policy)
    seeds = parse_seed_range(args.seeds)
    starter_tile = parse_starter(args.starter)
    phase_filter = parse_filter_values(
        args.phase_filter,
        allowed=PHASE_BUCKETS,
        aliases={
            "early": "early_lt384",
            "mid": "mid_384_768",
            "middle": "mid_384_768",
            "late": "late_1536",
            "endgame": "endgame_3072p",
        },
        label="phase",
    )
    corner_risk_filter = parse_filter_values(
        args.corner_risk_filter,
        allowed=CORNER_RISK_BUCKETS,
        aliases={
            "low": "low_corner_risk",
            "medium": "medium_corner_risk",
            "med": "medium_corner_risk",
            "high": "high_corner_risk",
        },
        label="corner risk",
    )
    source_metadata: dict[str, Any] = {}
    action_value_cache = load_action_value_cache(args.action_value_cache) if args.samples_json is None else None
    if args.samples_json is not None:
        samples, scan_stats, source_metadata = load_samples_json(args.samples_json)
        corpus = {"samples": samples, "scan_stats": scan_stats}
    else:
        replay_paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
        state_record_paths = _flatten_paths(args.state_json) + _glob_paths(args.state_glob)
        if replay_paths and state_record_paths:
            raise ValueError("Use replay inputs or state-record inputs, not both")
        state_records = _load_state_records_from_paths(state_record_paths) if state_record_paths else []
        if args.sample_mode == "anchor-risk":
            if state_records:
                corpus = collect_anchor_risk_states_from_state_records(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    state_records=state_records,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    min_anchor_tile=args.anchor_min_tile,
                    max_state_records=args.max_state_records,
                    state_start_index=args.state_start_index,
                    progress_every_state_records=args.progress_every_state_records,
                    action_value_cache=action_value_cache,
                )
            elif replay_paths:
                corpus = collect_anchor_risk_states_from_replays(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    replay_paths=replay_paths,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    replay_base_action=args.replay_base_action,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    min_anchor_tile=args.anchor_min_tile,
                    max_replays=args.max_replays,
                    replay_start_index=args.replay_start_index,
                    progress_every_replays=args.progress_every_replays,
                    action_value_cache=action_value_cache,
                )
            else:
                corpus = collect_anchor_risk_states(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    seeds=seeds,
                    starter_tile=starter_tile,
                    max_moves=args.max_moves,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    min_anchor_tile=args.anchor_min_tile,
                    progress_every_seeds=args.progress_every_seeds,
                )
        elif args.sample_mode == "geometry-risk":
            if state_records:
                corpus = collect_geometry_risk_states_from_state_records(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    state_records=state_records,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    geometry_min_tile=args.geometry_min_tile,
                    geometry_min_delta=args.geometry_min_delta,
                    max_state_records=args.max_state_records,
                    state_start_index=args.state_start_index,
                    progress_every_state_records=args.progress_every_state_records,
                    action_value_cache=action_value_cache,
                )
            else:
                if not replay_paths:
                    raise ValueError("--sample-mode geometry-risk requires replay inputs or --state-json/--state-glob")
                corpus = collect_geometry_risk_states_from_replays(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    replay_paths=replay_paths,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    replay_base_action=args.replay_base_action,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    geometry_min_tile=args.geometry_min_tile,
                    geometry_min_delta=args.geometry_min_delta,
                    max_replays=args.max_replays,
                    replay_start_index=args.replay_start_index,
                    progress_every_replays=args.progress_every_replays,
                    action_value_cache=action_value_cache,
                )
        elif args.sample_mode == "support-chain-risk":
            if state_records:
                corpus = collect_support_chain_risk_states_from_state_records(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    state_records=state_records,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    support_min_tile=args.support_min_tile,
                    support_target_min_tile=args.support_target_min_tile,
                    support_mask_starter=args.support_mask_mode == "masked",
                    support_min_delta=args.support_min_delta,
                    max_state_records=args.max_state_records,
                    state_start_index=args.state_start_index,
                    progress_every_state_records=args.progress_every_state_records,
                    action_value_cache=action_value_cache,
                )
            else:
                if not replay_paths:
                    raise ValueError("--sample-mode support-chain-risk requires replay inputs or --state-json/--state-glob")
                corpus = collect_support_chain_risk_states_from_replays(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    replay_paths=replay_paths,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    replay_base_action=args.replay_base_action,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    support_min_tile=args.support_min_tile,
                    support_target_min_tile=args.support_target_min_tile,
                    support_mask_starter=args.support_mask_mode == "masked",
                    support_min_delta=args.support_min_delta,
                    max_replays=args.max_replays,
                    replay_start_index=args.replay_start_index,
                    progress_every_replays=args.progress_every_replays,
                    action_value_cache=action_value_cache,
                )
        elif args.sample_mode == "top-two":
            if not state_records:
                raise ValueError("--sample-mode top-two requires --state-json/--state-glob inputs")
            corpus = collect_top_two_states_from_state_records(
                base_policy=base_policy,
                base_policy_spec=args.base_policy,
                state_records=state_records,
                margin_threshold=args.margin_threshold,
                max_samples=args.max_samples,
                max_per_stratum=args.max_per_stratum,
                first_per=args.first_per,
                phase_filter=phase_filter,
                corner_risk_filter=corner_risk_filter,
                max_state_records=args.max_state_records,
                state_start_index=args.state_start_index,
                progress_every_state_records=args.progress_every_state_records,
                action_value_cache=action_value_cache,
                min_top_value=args.min_top_value,
            )
        else:
            if not args.comparison_policy:
                raise ValueError("--comparison-policy is required unless --samples-json is provided")
            comparison_policies = [(spec, make_policy(spec)) for spec in args.comparison_policy]
            if state_records:
                raise ValueError("--state-json scans currently support anchor-risk, geometry-risk, support-chain-risk, and top-two sample modes")
            if replay_paths:
                corpus = collect_swing_states_from_replays(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    comparison_policies=comparison_policies,
                    replay_paths=replay_paths,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    replay_base_action=args.replay_base_action,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    max_replays=args.max_replays,
                    replay_start_index=args.replay_start_index,
                    progress_every_replays=args.progress_every_replays,
                    action_value_cache=action_value_cache,
                )
            else:
                corpus = collect_swing_states(
                    base_policy=base_policy,
                    base_policy_spec=args.base_policy,
                    comparison_policies=comparison_policies,
                    seeds=seeds,
                    starter_tile=starter_tile,
                    max_moves=args.max_moves,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.max_samples,
                    max_per_stratum=args.max_per_stratum,
                    first_per=args.first_per,
                    phase_filter=phase_filter,
                    corner_risk_filter=corner_risk_filter,
                    progress_every_seeds=args.progress_every_seeds,
                )
    if args.action_value_cache is not None and action_value_cache is not None:
        write_action_value_cache(args.action_value_cache, action_value_cache)
    horizons = parse_horizons(args.horizons)
    label_progress_path = args.label_progress_json
    if label_progress_path is None and args.checkpoint_labels:
        label_progress_path = args.out_dir / "label_progress.json"
    if args.no_label:
        labels = []
    elif label_progress_path is not None:
        if args.workers > 1 and len(corpus["samples"]) > 1:
            labels = label_corpus_resumable_parallel(
                corpus["samples"],
                policy_spec=args.base_policy,
                repeats=args.label_repeats,
                horizons=horizons,
                label_seed=args.label_seed,
                stability_threshold=args.stability_threshold,
                progress_path=label_progress_path,
                workers=args.workers,
                repeat_chunk_size=args.repeat_chunk_size,
                progress_every_chunks=args.progress_every_label_chunks,
            )
        else:
            labels = label_corpus_resumable(
                corpus["samples"],
                policy=base_policy,
                policy_spec=args.base_policy,
                repeats=args.label_repeats,
                horizons=horizons,
                label_seed=args.label_seed,
                stability_threshold=args.stability_threshold,
                progress_path=label_progress_path,
                repeat_chunk_size=args.repeat_chunk_size,
                progress_every_chunks=args.progress_every_label_chunks,
            )
    elif args.workers > 1 and len(corpus["samples"]) > 1:
        labels = label_corpus_parallel(
            corpus["samples"],
            policy_spec=args.base_policy,
            repeats=args.label_repeats,
            horizons=horizons,
            label_seed=args.label_seed,
            stability_threshold=args.stability_threshold,
            workers=args.workers,
            repeat_chunk_size=args.repeat_chunk_size,
            progress_every_chunks=args.progress_every_label_chunks,
        )
    else:
        labels = label_corpus(
            corpus["samples"],
            policy=base_policy,
            repeats=args.label_repeats,
            horizons=horizons,
            label_seed=args.label_seed,
            stability_threshold=args.stability_threshold,
        )
    summary = summarize_labeled(labels, corpus["scan_stats"])
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sample_mode": args.sample_mode,
        "base_policy": args.base_policy,
        "comparison_policies": (
            args.comparison_policy
            or source_metadata.get("source_comparison_policies")
            or (
                ["anchor_safe"]
                if args.sample_mode == "anchor-risk"
                else ["geometry_best"]
                if args.sample_mode == "geometry-risk"
                else ["support_chain_best"]
                if args.sample_mode == "support-chain-risk"
                else ["top_two_second"]
                if args.sample_mode == "top-two"
                else []
            )
        ),
        "seeds": args.seeds,
        "starter_tile": starter_tile,
        "max_moves": args.max_moves,
        "margin_threshold": args.margin_threshold,
        "max_samples": args.max_samples,
        "max_per_stratum": args.max_per_stratum,
        "max_replays": args.max_replays,
        "replay_start_index": args.replay_start_index,
        "action_value_cache": str(args.action_value_cache) if args.action_value_cache is not None else None,
        "first_per": args.first_per,
        "anchor_min_tile": args.anchor_min_tile,
        "geometry_min_tile": args.geometry_min_tile,
        "geometry_min_delta": args.geometry_min_delta,
        "support_min_tile": args.support_min_tile,
        "support_target_min_tile": args.support_target_min_tile,
        "support_mask_mode": args.support_mask_mode,
        "support_min_delta": args.support_min_delta,
        "max_state_records": args.max_state_records,
        "state_start_index": args.state_start_index,
        "min_top_value": args.min_top_value,
        "replay_base_action": args.replay_base_action,
        "phase_filter": sorted(phase_filter) if phase_filter is not None else None,
        "corner_risk_filter": sorted(corner_risk_filter) if corner_risk_filter is not None else None,
        "label_repeats": args.label_repeats,
        "checkpoint_labels": bool(args.checkpoint_labels or args.label_progress_json is not None),
        "label_progress_json": str(label_progress_path) if label_progress_path is not None else None,
        "horizons": list(horizons),
        "stability_threshold": args.stability_threshold,
        "workers": args.workers,
        "repeat_chunk_size": args.repeat_chunk_size,
        "progress_every_label_chunks": args.progress_every_label_chunks,
        "replay_json": [str(path) for path in _flatten_paths(args.replay_json)],
        "replay_glob": list(args.replay_glob or []),
        "resolved_replay_paths": [str(path) for path in replay_paths] if args.samples_json is None else [],
        "state_json": [str(path) for path in _flatten_paths(args.state_json)],
        "state_glob": list(args.state_glob or []),
        "resolved_state_record_paths": [str(path) for path in state_record_paths] if args.samples_json is None else [],
        "samples": corpus["samples"],
        "labels": labels,
        "summary": summary,
        **source_metadata,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "swing_labels.json"
    html_path = args.out_dir / "swing_labels.html"
    write_json(json_path, payload)
    write_report_html(html_path, payload)
    payload["json"] = str(json_path)
    payload["html"] = str(html_path)
    write_json(json_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", required=True)
    parser.add_argument(
        "--sample-mode",
        choices=["disagreement", "anchor-risk", "geometry-risk", "support-chain-risk", "top-two"],
        default="disagreement",
        help="Collect policy-disagreement, top-left-anchor-risk, high-tile-geometry-risk, support-chain-risk, or direct top-two samples.",
    )
    parser.add_argument("--comparison-policy", action="append")
    parser.add_argument("--samples-json", type=Path, help="Reuse samples from a previous swing_labels.json instead of rescanning.")
    parser.add_argument(
        "--replay-json",
        type=Path,
        nargs="+",
        action="append",
        help="One or more replay JSONs to scan instead of generating fresh actor rollouts.",
    )
    parser.add_argument(
        "--replay-glob",
        action="append",
        help="Glob pattern for replay JSONs to scan; useful for large top-game pools.",
    )
    parser.add_argument(
        "--state-json",
        type=Path,
        nargs="+",
        action="append",
        help="Reservoir/state-record JSONs to scan; records must contain a state payload.",
    )
    parser.add_argument(
        "--state-glob",
        action="append",
        help="Glob pattern for reservoir/state-record JSONs to scan.",
    )
    parser.add_argument("--seeds", default="2000:2060")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--margin-threshold", type=float, default=0.002)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--max-per-stratum", type=int, default=4)
    parser.add_argument("--max-replays", type=int, default=0, help="For replay scans, scan at most this many resolved replay files; 0 means all.")
    parser.add_argument(
        "--replay-start-index",
        type=int,
        default=0,
        help="For replay scans, skip this many resolved replay files before applying --max-replays.",
    )
    parser.add_argument(
        "--max-state-records",
        type=int,
        default=0,
        help="For state-record scans, scan at most this many resolved records; 0 means all.",
    )
    parser.add_argument(
        "--state-start-index",
        type=int,
        default=0,
        help="For state-record scans, skip this many records before applying --max-state-records.",
    )
    parser.add_argument(
        "--action-value-cache",
        type=Path,
        help="Optional JSON cache for replay-scan action values keyed by policy and exact simulator state.",
    )
    parser.add_argument(
        "--first-per",
        choices=[
            "seed",
            "seed-phase",
            "seed-stratum",
            "seed-policy",
            "seed-policy-phase",
            "seed-policy-stratum",
            "replay",
            "replay-phase",
            "replay-stratum",
            "none",
        ],
        default="seed-policy",
        help="Limit accepted samples to the first qualifying disagreement for each scope.",
    )
    parser.add_argument("--anchor-min-tile", type=int, default=1536)
    parser.add_argument(
        "--min-top-value",
        type=float,
        default=0.0,
        help="For top-two state-record scans, require the actor's best root value to be at least this value.",
    )
    parser.add_argument("--geometry-min-tile", type=int, default=1536)
    parser.add_argument(
        "--geometry-min-delta",
        type=float,
        default=1.0,
        help="Minimum afterstate geometry-score improvement over the actor action for geometry-risk samples.",
    )
    parser.add_argument(
        "--support-min-tile",
        type=int,
        default=768,
        help="Minimum high tile counted as support-chain material for support-chain-risk samples.",
    )
    parser.add_argument(
        "--support-target-min-tile",
        type=int,
        default=3072,
        help="Minimum built max tile before support-chain-risk samples are considered.",
    )
    parser.add_argument(
        "--support-mask-mode",
        choices=["masked", "raw"],
        default="masked",
        help="Whether support-chain-risk afterstate scoring masks the free starter tile or uses raw board counts.",
    )
    parser.add_argument(
        "--support-min-delta",
        type=float,
        default=50.0,
        help="Minimum support-chain score improvement over the actor action for support-chain-risk samples.",
    )
    parser.add_argument(
        "--replay-base-action",
        choices=["recorded", "policy"],
        default="recorded",
        help="For replay scans, compare against the recorded move or the frozen base policy's move.",
    )
    parser.add_argument(
        "--phase-filter",
        action="append",
        help="Comma-separated phase buckets to accept; aliases include early, mid, late, endgame.",
    )
    parser.add_argument(
        "--corner-risk-filter",
        action="append",
        help="Comma-separated corner-risk buckets to accept; aliases include low, medium, high.",
    )
    parser.add_argument("--progress-every-seeds", type=int, default=0)
    parser.add_argument("--progress-every-replays", type=int, default=0)
    parser.add_argument("--progress-every-state-records", type=int, default=0)
    parser.add_argument("--label-repeats", type=int, default=16)
    parser.add_argument("--horizons", default="32,64")
    parser.add_argument("--label-seed", type=int, default=20260706)
    parser.add_argument("--stability-threshold", type=float, default=0.70)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeat-chunk-size", type=int, default=4)
    parser.add_argument(
        "--progress-every-label-chunks",
        type=int,
        default=0,
        help="For checkpointed labels, print progress after this many newly written repeat chunks; 0 disables.",
    )
    parser.add_argument(
        "--checkpoint-labels",
        action="store_true",
        help="Checkpoint single-process continuation labels to <out-dir>/label_progress.json after each repeat chunk.",
    )
    parser.add_argument(
        "--label-progress-json",
        type=Path,
        help="Explicit JSON path for resumable single-process continuation-label progress.",
    )
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("threes_rl/runs/forensics/swing_labels/latest"),
    )
    args = parser.parse_args()

    payload = run_swing_labeling(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()

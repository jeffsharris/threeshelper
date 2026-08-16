"""Audit features that distinguish transition-window success and failure states."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.run_artifacts import write_json
from threes_rl.sim import rank_for_value
from threes_rl.swing_label import support_chain_features


BOARD_CELLS = tuple((row, col) for row in range(4) for col in range(4))
POSITIONAL_TILE_VALUES = (768, 1536, 3072)
CELL_RANK_FEATURES = tuple(f"cell_{row}{col}_rank" for row, col in BOARD_CELLS)
CELL_TILE_FEATURES = tuple(
    f"cell_{row}{col}_is_{tile}"
    for tile in POSITIONAL_TILE_VALUES
    for row, col in BOARD_CELLS
)
LINE_MERGE_FEATURES = (
    "raw_768_same_line_pair_count",
    "raw_768_line_blocker_min",
    "raw_768_clear_merge_pair_count",
    "raw_768_min_clear_pair_distance",
    "raw_768_clear_target_min_distance_to_1536",
    "raw_768_clear_target_adjacent_to_1536",
    "raw_1536_same_line_pair_count",
    "raw_1536_line_blocker_min",
    "raw_1536_clear_merge_pair_count",
    "raw_1536_min_clear_pair_distance",
    "raw_1536_clear_target_min_distance_to_3072",
    "raw_1536_clear_target_adjacent_to_3072",
)

BASE_NUMERIC_FEATURES = (
    "score_minus_starter",
    "move_count",
    "empty_count",
    "legal_count",
    "safe_smalls_until_large_possible",
    "large_pending",
    "top_left",
    "top_left_is_max",
    "count_3072",
    "count_1536",
    "count_768",
    "count_ge_768",
    "support_score",
    "support_count_target",
    "support_adjacent_pair",
    "support_adjacent_to_max",
    "support_distance_to_max",
    "support_distance_to_top_left",
    "support_high_count",
    "raw_count_768",
    "raw_count_1536",
    "raw_highest_duplicate_tile",
    "raw_highest_adjacent_pair_tile",
    "raw_has_adjacent_768",
    "raw_has_adjacent_1536",
    "raw_768_adjacent_pairs",
    "raw_768_components",
    "raw_768_max_component",
    "raw_768_air_neighbors",
    "raw_768_edge_count",
    "raw_768_corner_count",
    "raw_768_min_pair_distance",
    "raw_768_min_distance_to_1536",
    "raw_768_adjacent_to_1536",
    "raw_768_min_distance_to_3072",
    "raw_1536_min_pair_distance",
    "raw_1536_min_distance_to_3072",
    "raw_1536_adjacent_to_3072",
    "max_tile_row",
    "max_tile_col",
    "max_tile_corner_distance",
    "corner_count_ge_768",
)
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + LINE_MERGE_FEATURES + CELL_RANK_FEATURES + CELL_TILE_FEATURES

CATEGORICAL_FEATURES = ("preview", "corner_risk", "source_action")
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _records_from_path(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} does not contain records[]")
    return [record for record in records if isinstance(record, dict)]


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for path in paths:
        for idx, record in enumerate(_records_from_path(Path(path))):
            loaded.append({**record, "_source_json": str(path), "_record_index": int(idx)})
    return loaded


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_float(value: object) -> float:
    return 1.0 if bool(value) else 0.0


def _has_adjacent_pair(board: np.ndarray, value: int) -> bool:
    arr = np.asarray(board, dtype=np.int32)
    positions = [tuple(int(v) for v in pos) for pos in np.argwhere(arr == int(value))]
    for idx, left in enumerate(positions):
        for right in positions[idx + 1 :]:
            if abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1:
                return True
    return False


def _positions(board: np.ndarray, value: int) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in pos) for pos in np.argwhere(np.asarray(board, dtype=np.int32) == int(value))]


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def _min_pair_distance(board: np.ndarray, value: int) -> int:
    positions = _positions(board, value)
    if len(positions) < 2:
        return 9
    return min(_manhattan(left, right) for idx, left in enumerate(positions) for right in positions[idx + 1 :])


def _min_distance_between_values(board: np.ndarray, left_value: int, right_value: int) -> int:
    left_positions = _positions(board, left_value)
    right_positions = _positions(board, right_value)
    if not left_positions or not right_positions:
        return 9
    return min(_manhattan(left, right) for left in left_positions for right in right_positions)


def _adjacent_between_values(board: np.ndarray, left_value: int, right_value: int) -> float:
    return _bool_float(_min_distance_between_values(board, left_value, right_value) == 1)


def _max_tile_position(board: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(board, dtype=np.int32)
    max_value = int(arr.max(initial=0))
    positions = _positions(arr, max_value)
    if not positions:
        return (0, 0)
    return min(positions)


def _corner_count_ge(board: np.ndarray, threshold: int) -> int:
    arr = np.asarray(board, dtype=np.int32)
    corners = (arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1])
    return int(sum(int(value) >= int(threshold) for value in corners))


def _cells_between(left: tuple[int, int], right: tuple[int, int]) -> list[tuple[int, int]]:
    if left[0] == right[0]:
        row = int(left[0])
        start, end = sorted((int(left[1]), int(right[1])))
        return [(row, col) for col in range(start + 1, end)]
    if left[1] == right[1]:
        col = int(left[1])
        start, end = sorted((int(left[0]), int(right[0])))
        return [(row, col) for row in range(start + 1, end)]
    return []


def _same_line(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return int(left[0]) == int(right[0]) or int(left[1]) == int(right[1])


def _line_merge_stats(board: np.ndarray, value: int, support_value: int) -> dict[str, float]:
    arr = np.asarray(board, dtype=np.int32)
    positions = _positions(arr, value)
    support_positions = _positions(arr, support_value)
    pair_count = 0
    min_blockers = 9
    clear_count = 0
    min_clear_distance = 9
    min_target_distance = 9
    clear_target_adjacent = 0.0
    for idx, left in enumerate(positions):
        for right in positions[idx + 1 :]:
            if not _same_line(left, right):
                continue
            pair_count += 1
            between = _cells_between(left, right)
            blockers = sum(int(arr[row, col]) != 0 for row, col in between)
            min_blockers = min(min_blockers, blockers)
            if blockers != 0:
                continue
            clear_count += 1
            min_clear_distance = min(min_clear_distance, _manhattan(left, right))
            if support_positions:
                endpoint_distance = min(
                    _manhattan(endpoint, support)
                    for endpoint in (left, right)
                    for support in support_positions
                )
                min_target_distance = min(min_target_distance, endpoint_distance)
                if endpoint_distance == 1:
                    clear_target_adjacent = 1.0
    return {
        f"raw_{value}_same_line_pair_count": float(pair_count),
        f"raw_{value}_line_blocker_min": float(min_blockers),
        f"raw_{value}_clear_merge_pair_count": float(clear_count),
        f"raw_{value}_min_clear_pair_distance": float(min_clear_distance),
        f"raw_{value}_clear_target_min_distance_to_{support_value}": float(min_target_distance),
        f"raw_{value}_clear_target_adjacent_to_{support_value}": float(clear_target_adjacent),
    }


def _cell_rank(value: int) -> float:
    try:
        return float(rank_for_value(int(value)))
    except ValueError:
        return 0.0


def _positional_features(board: np.ndarray) -> dict[str, float]:
    arr = np.asarray(board, dtype=np.int32)
    features: dict[str, float] = {}
    for row, col in BOARD_CELLS:
        value = int(arr[row, col])
        features[f"cell_{row}{col}_rank"] = _cell_rank(value)
        for tile in POSITIONAL_TILE_VALUES:
            features[f"cell_{row}{col}_is_{tile}"] = _bool_float(value == int(tile))
    return features


def _support_transition_features(board: np.ndarray) -> dict[str, float]:
    features = {}
    features.update(_line_merge_stats(board, 768, 1536))
    features.update(_line_merge_stats(board, 1536, 3072))
    features.update(_positional_features(board))
    return features


def _highest_duplicate_tile(board: np.ndarray) -> int:
    counts = Counter(int(value) for value in np.asarray(board, dtype=np.int32).reshape(-1) if int(value) > 0)
    return max((value for value, count in counts.items() if count >= 2), default=0)


def _highest_adjacent_pair_tile(board: np.ndarray) -> int:
    arr = np.asarray(board, dtype=np.int32)
    values = sorted({int(value) for value in arr.reshape(-1) if int(value) > 0}, reverse=True)
    for value in values:
        if _has_adjacent_pair(arr, value):
            return int(value)
    return 0


def _raw_ladder_feature(record: dict[str, Any], raw_source: dict[str, Any], board: np.ndarray, key: str) -> float:
    if raw_source.get(key) is not None:
        return _float(raw_source.get(key))
    if record.get(key) is not None:
        return _float(record.get(key))
    if key == "raw_count_768":
        return _float(np.count_nonzero(board == 768))
    if key == "raw_count_1536":
        return _float(np.count_nonzero(board == 1536))
    if key == "raw_highest_duplicate_tile":
        return _float(_highest_duplicate_tile(board))
    if key == "raw_highest_adjacent_pair_tile":
        return _float(_highest_adjacent_pair_tile(board))
    if key == "raw_has_adjacent_768":
        return _bool_float(_has_adjacent_pair(board, 768))
    if key == "raw_has_adjacent_1536":
        return _bool_float(_has_adjacent_pair(board, 1536))
    return 0.0


def _group_key_for_record(record: dict[str, Any], source_replay: str, *, group_by: str) -> str:
    mode = str(group_by or "auto")
    if mode == "auto":
        return str(record.get("source_group", record.get("group_key", source_replay)))
    if mode == "source-group":
        return str(record.get("source_group", record.get("group_key", source_replay)))
    if mode == "source-replay":
        return source_replay
    if mode == "original-replay":
        replay = record.get("original_source_replay", record.get("source_replay", source_replay))
        seed = record.get("original_source_seed", record.get("source_seed", record.get("seed")))
        return json.dumps(
            {"original_source_replay": str(replay), "original_source_seed": seed},
            sort_keys=True,
            separators=(",", ":"),
        )
    raise ValueError(f"Unsupported group_by mode: {group_by}")


def row_from_record(
    record: dict[str, Any],
    *,
    require_outcome: bool = True,
    group_by: str = "auto",
) -> dict[str, Any] | None:
    outcome = str(record.get("outcome", "")).lower()
    has_outcome = outcome in ("success", "failure")
    if require_outcome and not has_outcome:
        return None
    state_payload = record.get("state")
    if not isinstance(state_payload, dict):
        return None
    state = state_from_payload(state_payload)
    board = np.asarray(state.board, dtype=np.int32)
    features = record.get("features") if isinstance(record.get("features"), dict) else {}
    starter_tile = record.get("starter_tile")
    starter = None if starter_tile is None else int(starter_tile)
    support = support_chain_features(
        board,
        starter_tile=starter,
        support_min_tile=768,
        target_min_tile=3072,
    )
    raw_source = record.get("features") if isinstance(record.get("features"), dict) else record
    board_max = int(board.max(initial=0))
    max_row, max_col = _max_tile_position(board)
    support_transition = _support_transition_features(board)
    source_replay = str(record.get("source_replay", record.get("_source_json", "unknown")))
    group_key = _group_key_for_record(record, source_replay, group_by=group_by)
    row = {
        "id": record.get("id"),
        "source_replay": source_replay,
        "group_key": group_key,
        "source_seed": record.get("source_seed", record.get("seed")),
        "source_frame_index": record.get("source_frame_index"),
        "outcome": outcome if has_outcome else "unlabeled",
        "y": 1 if outcome == "success" else 0,
        "target_tile": record.get("target_tile"),
        "moves_to_promotion": record.get("moves_to_promotion"),
        "moves_to_terminal": record.get("moves_to_terminal"),
        "score_minus_starter": _float(features.get("score_minus_starter", record.get("score_minus_starter"))),
        "move_count": _float(record.get("move_count", state.move_count)),
        "empty_count": _float(features.get("empty_count", int(np.count_nonzero(board == 0)))),
        "legal_count": _float(record.get("legal_count", len(state_payload.get("legal_actions", [])))),
        "safe_smalls_until_large_possible": _float(features.get("safe_smalls_until_large_possible")),
        "large_pending": _bool_float(features.get("large_pending", state.large_pending)),
        "top_left": _float(features.get("top_left", int(board[0, 0]))),
        "top_left_is_max": _bool_float(int(board[0, 0]) == board_max),
        "count_3072": _float(np.count_nonzero(board == 3072)),
        "count_1536": _float(np.count_nonzero(board == 1536)),
        "count_768": _float(np.count_nonzero(board == 768)),
        "count_ge_768": _float(np.count_nonzero(board >= 768)),
        "support_score": _float(support.get("score")),
        "support_count_target": _float(support.get("count_target_support")),
        "support_adjacent_pair": _bool_float(support.get("target_support_has_adjacent_pair")),
        "support_adjacent_to_max": _bool_float(support.get("target_support_adjacent_to_max")),
        "support_distance_to_max": _float(support.get("min_target_support_to_max_distance"), 9.0),
        "support_distance_to_top_left": _float(support.get("min_target_support_to_top_left_distance"), 9.0),
        "support_high_count": _float(support.get("high_support_count")),
        "raw_count_768": _raw_ladder_feature(record, raw_source, board, "raw_count_768"),
        "raw_count_1536": _raw_ladder_feature(record, raw_source, board, "raw_count_1536"),
        "raw_highest_duplicate_tile": _raw_ladder_feature(record, raw_source, board, "raw_highest_duplicate_tile"),
        "raw_highest_adjacent_pair_tile": _raw_ladder_feature(record, raw_source, board, "raw_highest_adjacent_pair_tile"),
        "raw_has_adjacent_768": _raw_ladder_feature(record, raw_source, board, "raw_has_adjacent_768"),
        "raw_has_adjacent_1536": _raw_ladder_feature(record, raw_source, board, "raw_has_adjacent_1536"),
        "raw_768_adjacent_pairs": _float(features.get("raw_768_adjacent_pairs", record.get("raw_768_adjacent_pairs"))),
        "raw_768_components": _float(features.get("raw_768_components", record.get("raw_768_components"))),
        "raw_768_max_component": _float(features.get("raw_768_max_component", record.get("raw_768_max_component"))),
        "raw_768_air_neighbors": _float(features.get("raw_768_air_neighbors", record.get("raw_768_air_neighbors"))),
        "raw_768_edge_count": _float(features.get("raw_768_edge_count", record.get("raw_768_edge_count"))),
        "raw_768_corner_count": _float(features.get("raw_768_corner_count", record.get("raw_768_corner_count"))),
        "raw_768_min_pair_distance": _float(_min_pair_distance(board, 768)),
        "raw_768_min_distance_to_1536": _float(_min_distance_between_values(board, 768, 1536)),
        "raw_768_adjacent_to_1536": _adjacent_between_values(board, 768, 1536),
        "raw_768_min_distance_to_3072": _float(_min_distance_between_values(board, 768, 3072)),
        "raw_1536_min_pair_distance": _float(_min_pair_distance(board, 1536)),
        "raw_1536_min_distance_to_3072": _float(_min_distance_between_values(board, 1536, 3072)),
        "raw_1536_adjacent_to_3072": _adjacent_between_values(board, 1536, 3072),
        "max_tile_row": _float(max_row),
        "max_tile_col": _float(max_col),
        "max_tile_corner_distance": _float(_manhattan((max_row, max_col), (0, 0))),
        "corner_count_ge_768": _float(_corner_count_ge(board, 768)),
        **support_transition,
        "preview": str(features.get("preview", state.preview.label)),
        "corner_risk": str(features.get("corner_risk", record.get("corner_risk", "unknown"))),
        "stratum": str(features.get("stratum", record.get("stratum", "unknown"))),
        "source_action": str(record.get("source_next_action", "unknown")),
    }
    return row


def build_rows(
    records: Iterable[dict[str, Any]],
    *,
    target_tile: int | None,
    group_by: str = "auto",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if target_tile is not None and int(record.get("target_tile", -1)) != int(target_tile):
            continue
        row = row_from_record(record, group_by=group_by)
        if row is not None:
            rows.append(row)
    return rows


def build_unlabeled_rows(
    records: Iterable[dict[str, Any]],
    *,
    target_tile: int | None = None,
    group_by: str = "auto",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if target_tile is not None and record.get("target_tile") is not None:
            try:
                if int(record.get("target_tile", -1)) != int(target_tile):
                    continue
            except (TypeError, ValueError):
                continue
        row = row_from_record(record, require_outcome=False, group_by=group_by)
        if row is not None:
            rows.append(row)
    return rows


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(median(float(value) for value in values)),
        "std": float(arr.std()),
    }


def feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if int(row["y"]) == 1]
    failure = [row for row in rows if int(row["y"]) == 0]
    numeric: list[dict[str, Any]] = []
    for feature in NUMERIC_FEATURES:
        s_values = [_float(row.get(feature)) for row in success]
        f_values = [_float(row.get(feature)) for row in failure]
        s_stats = _stats(s_values)
        f_stats = _stats(f_values)
        pooled = math.sqrt((s_stats["std"] ** 2 + f_stats["std"] ** 2) / 2.0)
        numeric.append(
            {
                "feature": feature,
                "success": s_stats,
                "failure": f_stats,
                "mean_diff_success_minus_failure": float(s_stats["mean"] - f_stats["mean"]),
                "standardized_diff": float((s_stats["mean"] - f_stats["mean"]) / pooled) if pooled > 1e-9 else 0.0,
            }
        )
    numeric.sort(key=lambda row: abs(float(row["standardized_diff"])), reverse=True)

    categorical: dict[str, list[dict[str, Any]]] = {}
    for feature in CATEGORICAL_FEATURES:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            value = str(row.get(feature, "unknown"))
            counts[value]["total"] += 1
            counts[value]["success"] += int(row["y"])
        categorical[feature] = [
            {
                "value": value,
                "records": int(counter["total"]),
                "successes": int(counter["success"]),
                "success_rate": float(counter["success"] / counter["total"]) if counter["total"] else 0.0,
            }
            for value, counter in sorted(counts.items(), key=lambda item: (-item[1]["total"], item[0]))
        ]
    return {"numeric": numeric, "categorical": categorical}


def _one_hot_maps(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        feature: sorted({str(row.get(feature, "unknown")) for row in rows})
        for feature in CATEGORICAL_FEATURES
    }


def _design_matrix(
    rows: list[dict[str, Any]],
    *,
    one_hot: dict[str, list[str]],
    mean_vec: np.ndarray | None = None,
    std_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    raw_cols: list[list[float]] = []
    names: list[str] = []
    for feature in NUMERIC_FEATURES:
        raw_cols.append([_float(row.get(feature)) for row in rows])
        names.append(feature)
    for feature, values in one_hot.items():
        for value in values:
            raw_cols.append([1.0 if str(row.get(feature, "unknown")) == value else 0.0 for row in rows])
            names.append(f"{feature}={value}")
    if not raw_cols:
        x = np.zeros((len(rows), 0), dtype=np.float64)
    else:
        x = np.asarray(raw_cols, dtype=np.float64).T
    if mean_vec is None:
        mean_vec = x.mean(axis=0) if x.size else np.zeros(0, dtype=np.float64)
    if std_vec is None:
        std_vec = x.std(axis=0) if x.size else np.ones(len(mean_vec), dtype=np.float64)
    std_safe = np.where(std_vec < 1e-6, 1.0, std_vec)
    x_std = (x - mean_vec) / std_safe if x.size else x
    x_bias = np.concatenate([np.ones((len(rows), 1), dtype=np.float64), x_std], axis=1)
    y = np.asarray([int(row["y"]) for row in rows], dtype=np.float64)
    return x_bias, y, ["bias", *names], mean_vec, std_safe


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, steps: int = 900, lr: float = 0.08, l2: float = 0.02) -> np.ndarray:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(int(steps)):
        logits = np.clip(x @ weights, -30.0, 30.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = (x.T @ (pred - y)) / max(1, len(y))
        grad[1:] += float(l2) * weights[1:]
        weights -= float(lr) * grad
    return weights


def _auc(y: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, y), key=lambda item: item[0])
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return 0.0
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for j in range(idx, end):
            if pairs[j][1] == 1:
                rank_sum += avg_rank
        idx = end
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def grouped_logistic_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(row.get("group_key", row.get("source_replay"))) for row in rows})
    one_hot = _one_hot_maps(rows)
    predictions: list[dict[str, Any]] = []
    skipped = 0
    for group in groups:
        train = [row for row in rows if str(row.get("group_key", row.get("source_replay"))) != group]
        test = [row for row in rows if str(row.get("group_key", row.get("source_replay"))) == group]
        if len({int(row["y"]) for row in train}) < 2 or not test:
            skipped += len(test)
            continue
        x_train, y_train, names, mean_vec, std_vec = _design_matrix(train, one_hot=one_hot)
        weights = _fit_logistic(x_train, y_train)
        x_test, y_test, _names, _mean, _std = _design_matrix(test, one_hot=one_hot, mean_vec=mean_vec, std_vec=std_vec)
        probs = 1.0 / (1.0 + np.exp(-np.clip(x_test @ weights, -30.0, 30.0)))
        for row, y_value, prob in zip(test, y_test, probs):
            predictions.append(
                {
                    "id": row.get("id"),
                    "source_replay": row.get("source_replay"),
                    "group_key": row.get("group_key"),
                    "source_seed": row.get("source_seed"),
                    "outcome": row.get("outcome"),
                    "y": int(y_value),
                    "prob_success": float(prob),
                }
            )
    y_true = [int(row["y"]) for row in predictions]
    scores = [float(row["prob_success"]) for row in predictions]
    acc = (
        sum((score >= 0.5) == bool(y) for score, y in zip(scores, y_true)) / len(y_true)
        if y_true
        else 0.0
    )

    x_all, y_all, names, _mean, _std = _design_matrix(rows, one_hot=one_hot)
    weights = _fit_logistic(x_all, y_all) if len({int(value) for value in y_all}) == 2 else np.zeros(x_all.shape[1])
    top_weights = [
        {"feature": name, "weight": float(weight)}
        for name, weight in sorted(zip(names, weights), key=lambda item: abs(float(item[1])), reverse=True)[:16]
    ]
    return {
        "groups": len(groups),
        "predictions": len(predictions),
        "skipped_records": int(skipped),
        "accuracy_at_0_5": float(acc),
        "auc": _auc(y_true, scores) if y_true else 0.0,
        "mean_predicted_success": float(mean(scores)) if scores else 0.0,
        "top_weights_full_fit": top_weights,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(int(row["y"]) for row in rows)
    failures = len(rows) - successes
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "records": len(rows),
        "successes": int(successes),
        "failures": int(failures),
        "success_rate": float(successes / len(rows)) if rows else 0.0,
        "source_replays": len({str(row.get("source_replay")) for row in rows}),
        "source_groups": len({str(row.get("group_key", row.get("source_replay"))) for row in rows}),
        "by_stratum": dict(Counter(str(row.get("stratum")) for row in rows)),
        "by_outcome": dict(Counter(str(row.get("outcome")) for row in rows)),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    feature_stats = payload.get("feature_summary", {})
    numeric = feature_stats.get("numeric", []) if isinstance(feature_stats, dict) else []
    probe = payload.get("grouped_logistic_probe", {})

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for item in numeric[:16] if isinstance(numeric, list) else []:
        rows.append(
            "<tr>"
            f"<td>{cell(item.get('feature'))}</td>"
            f"<td>{float((item.get('success') or {}).get('mean', 0.0)):.3f}</td>"
            f"<td>{float((item.get('failure') or {}).get('mean', 0.0)):.3f}</td>"
            f"<td>{float(item.get('mean_diff_success_minus_failure', 0.0)):.3f}</td>"
            f"<td>{float(item.get('standardized_diff', 0.0)):.3f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Transition Reachability Audit</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Transition Reachability Audit</h1>
    <p class="muted">Success-vs-failure feature audit with source-replay-disjoint logistic probing.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Success Rate</div><div class="value">{float(summary.get('success_rate', 0.0)):.0%}</div></div>
      <div class="card"><div class="label">LOSO AUC</div><div class="value">{float(probe.get('auc', 0.0)):.3f}</div></div>
      <div class="card"><div class="label">LOSO Accuracy</div><div class="value">{float(probe.get('accuracy_at_0_5', 0.0)):.0%}</div></div>
    </section>
    <section class="panel">
      <table><thead><tr><th>Feature</th><th>Success Mean</th><th>Failure Mean</th><th>Diff</th><th>Std Diff</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;"><pre>{escape(json.dumps(probe, indent=2, sort_keys=True))}</pre></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records([path for group in args.state_json for path in group])
    target_tile = None if bool(args.no_target_filter) else args.target_tile
    group_by = getattr(args, "group_by", "auto")
    rows = build_rows(records, target_tile=target_tile, group_by=group_by)
    feature_stats = feature_summary(rows)
    probe = grouped_logistic_probe(rows)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_json": [str(path) for group in args.state_json for path in group],
        "target_tile": target_tile,
        "group_by": group_by,
        "summary": summarize(rows),
        "feature_summary": feature_stats,
        "grouped_logistic_probe": probe,
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "transition_reachability_audit.json")
    payload["html"] = str(args.out_dir / "transition_reachability_audit.html")
    write_json(args.out_dir / "transition_reachability_audit.json", payload)
    write_html(args.out_dir / "transition_reachability_audit.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--target-tile", type=int, default=6144)
    parser.add_argument(
        "--no-target-filter",
        action="store_true",
        help="Audit all outcome-labeled records even when target_tile is absent or heterogeneous.",
    )
    parser.add_argument(
        "--group-by",
        choices=("auto", "source-group", "source-replay", "original-replay"),
        default="auto",
        help="Grouping key for held-out logistic validation.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/reachability/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps(payload["grouped_logistic_probe"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()

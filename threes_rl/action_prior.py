"""Action-conditioned top-two correction policy.

The value sidecar experiments try to reshape the whole board evaluator from a
small number of action labels. This module keeps the problem narrower: train a
pairwise model that predicts whether one first action should beat another from
the current state, then allow a policy wrapper to override only the actor's
top-two choice.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from threes_rl.expectimax import high_tile_geometry_score
from threes_rl.ntuple import PHASE4_NAMES, max_tile_excluding_free_starter, phase4_index_for_board
from threes_rl.run_artifacts import write_json, write_progress_csv
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, direction_index, rank_for_value, score_board, simulate_base_move
from threes_rl.train_action_label_calibration import confidence_regret_weight
from threes_rl.train_td import state_from_replay_payload

ORTHOGONAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class ActionPriorTrainConfig:
    run_name: str
    endgame_label_json: list[str] = field(default_factory=list)
    swing_label_json: list[str] = field(default_factory=list)
    stable_only: bool = True
    confidence_threshold: float = 0.70
    phase_filter: list[str] | None = None
    corner_risk_filter: list[str] | None = None
    label_weight_mode: str = "confidence_regret"
    starter_tile: int | None = 1536
    epochs: int = 400
    lr: float = 0.05
    l2: float = 0.001
    seed: int = 20260706
    progress_every: int = 200


@dataclass
class PairExample:
    group_id: str
    first_action: str
    second_action: str
    target: float
    features: np.ndarray
    weight: float
    phase: str
    corner_risk: str
    base_action: str
    winner: str
    oracle_regret: float


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def parse_starter(text: str) -> int | None:
    value = text.strip().lower()
    return None if value == "none" else int(value)


def parse_optional_filter(text: str | None) -> list[str] | None:
    if text is None or not text.strip():
        return None
    aliases = {
        "early": "early_lt384",
        "mid": "mid_384_768",
        "middle": "mid_384_768",
        "late": "late_1536",
        "endgame": "endgame_3072p",
        "low": "low_corner_risk",
        "medium": "medium_corner_risk",
        "med": "medium_corner_risk",
        "high": "high_corner_risk",
    }
    values: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        raw = part.strip().lower()
        if not raw:
            continue
        normalized = aliases.get(raw, raw)
        if normalized not in seen:
            values.append(normalized)
            seen.add(normalized)
    return values or None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _phase_for_board(board: np.ndarray, starter_tile: int | None) -> str:
    return PHASE4_NAMES[phase4_index_for_board(board, starter_tile=starter_tile)]


def _corner_risk_bucket_for_state(state: SimState, starter_tile: int | None) -> str:
    board = np.asarray(state.board, dtype=np.int32)
    top_left = int(board[0, 0])
    board_max = int(board.max(initial=0))
    empty_count = int(np.count_nonzero(board == 0))
    built_max = max_tile_excluding_free_starter(board, starter_tile)

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


def stratum_for_state(state: SimState, starter_tile: int | None) -> str:
    return f"{_phase_for_board(state.board, starter_tile)}/{_corner_risk_bucket_for_state(state, starter_tile)}"


def _rank(value: int) -> float:
    return float(rank_for_value(int(value))) / 13.0


def _board_shape_features(board: np.ndarray, starter_tile: int | None) -> dict[str, float]:
    ranks = [rank_for_value(int(value)) for value in np.asarray(board, dtype=np.int32).reshape(-1)]
    max_rank = max(ranks) if ranks else 0
    top_left_rank = ranks[0] if ranks else 0
    empty_count = sum(1 for value in np.asarray(board).reshape(-1) if int(value) == 0)
    merge_potential = 0
    smoothness = 0
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
                    smoothness += abs(rank - other)
    built_max = max_tile_excluding_free_starter(board, starter_tile)
    return {
        "empty_count": float(empty_count) / 16.0,
        "top_left_rank": float(top_left_rank) / 13.0,
        "max_rank": float(max_rank) / 13.0,
        "built_max_rank": _rank(built_max),
        "anchor_is_max": 1.0 if top_left_rank == max_rank else 0.0,
        "anchor_is_built_max": 1.0 if int(np.asarray(board)[0, 0]) == int(built_max) and built_max > 0 else 0.0,
        "monotone_penalty": float(monotone_penalty) / 64.0,
        "merge_potential": float(merge_potential) / 64.0,
        "smoothness": float(smoothness) / 128.0,
    }


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


def _built_max_support_features(board: np.ndarray, starter_tile: int | None) -> dict[str, float]:
    masked = _board_without_free_starter(board, starter_tile)
    built_max = int(masked.max(initial=0))
    if built_max <= 0:
        return {
            "built_max_distance_top_left": 1.0,
            "built_max_top_left": 0.0,
            "built_max_top_or_left_edge": 0.0,
            "built_max_has_same_neighbor": 0.0,
            "built_max_has_half_neighbor": 0.0,
            "built_max_stranded": 0.0,
        }
    positions = [tuple(int(v) for v in pos) for pos in np.argwhere(masked == built_max)]
    min_distance = min(int(row) + int(col) for row, col in positions)
    top_left = any(row == 0 and col == 0 for row, col in positions)
    top_or_left = any(row == 0 or col == 0 for row, col in positions)
    has_same = _has_adjacent_value(masked, positions, built_max)
    has_half = _has_adjacent_value(masked, positions, built_max // 2)
    stranded = built_max >= 1536 and not top_left and not has_same and not has_half
    return {
        "built_max_distance_top_left": float(min_distance) / 6.0,
        "built_max_top_left": 1.0 if top_left else 0.0,
        "built_max_top_or_left_edge": 1.0 if top_or_left else 0.0,
        "built_max_has_same_neighbor": 1.0 if has_same else 0.0,
        "built_max_has_half_neighbor": 1.0 if has_half else 0.0,
        "built_max_stranded": 1.0 if stranded else 0.0,
    }


def action_feature_names() -> list[str]:
    names = [f"action_{name}" for name in DIRECTION_NAMES]
    names.extend(f"cell_{idx}_rank" for idx in range(16))
    names.extend(
        [
            "merge_delta_log",
            "empty_count",
            "top_left_rank",
            "max_rank",
            "built_max_rank",
            "anchor_is_max",
            "anchor_is_built_max",
            "monotone_penalty",
            "merge_potential",
            "smoothness",
            "high_tile_geometry_score",
            "built_max_distance_top_left",
            "built_max_top_left",
            "built_max_top_or_left_edge",
            "built_max_has_same_neighbor",
            "built_max_has_half_neighbor",
            "built_max_stranded",
            "preview_blue",
            "preview_red",
            "preview_gray",
            "preview_bonus",
            "large_pending",
        ]
    )
    return names


FEATURE_NAMES = action_feature_names()


def action_features(state: SimState, action: int | str, starter_tile: int | None = 1536) -> np.ndarray:
    action_idx = direction_index(action)
    afterstate, eligible = simulate_base_move(state.board, int(action_idx))
    if not eligible:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    before_score = score_board(state.board)
    after_score = score_board(afterstate)
    values: list[float] = []
    values.extend(1.0 if idx == int(action_idx) else 0.0 for idx in range(len(DIRECTION_NAMES)))
    values.extend(_rank(int(value)) for value in np.asarray(afterstate, dtype=np.int32).reshape(-1))
    shape = _board_shape_features(afterstate, starter_tile)
    values.append(math.log1p(max(0, int(after_score - before_score))) / 14.0)
    values.extend(shape[name] for name in (
        "empty_count",
        "top_left_rank",
        "max_rank",
        "built_max_rank",
        "anchor_is_max",
        "anchor_is_built_max",
        "monotone_penalty",
        "merge_potential",
        "smoothness",
    ))
    values.append(
        float(
            high_tile_geometry_score(
                afterstate,
                starter_tile=starter_tile,
                min_tile=1536,
            )
        )
        / 256.0
    )
    support = _built_max_support_features(afterstate, starter_tile)
    values.extend(
        support[name]
        for name in (
            "built_max_distance_top_left",
            "built_max_top_left",
            "built_max_top_or_left_edge",
            "built_max_has_same_neighbor",
            "built_max_has_half_neighbor",
            "built_max_stranded",
        )
    )
    preview = state.preview.label
    values.extend(
        [
            1.0 if preview == "blue" else 0.0,
            1.0 if preview == "red" else 0.0,
            1.0 if preview == "gray" else 0.0,
            1.0 if state.preview.kind == "bonus" else 0.0,
            1.0 if state.large_pending else 0.0,
        ]
    )
    return np.asarray(values, dtype=np.float64)


def pair_features(state: SimState, first_action: int | str, second_action: int | str, starter_tile: int | None = 1536) -> np.ndarray:
    return action_features(state, first_action, starter_tile) - action_features(state, second_action, starter_tile)


def _state_payload_from_replay_frame(replay_path: Path, frame_index: int) -> dict[str, Any] | None:
    replay = json.loads(replay_path.read_text())
    frames = replay.get("frames", [])
    if not isinstance(frames, list):
        return None
    for frame in frames:
        if isinstance(frame, dict) and int(frame.get("index", -1)) == int(frame_index):
            state_payload = frame.get("state")
            return state_payload if isinstance(state_payload, dict) else None
    if 0 <= int(frame_index) < len(frames) and isinstance(frames[int(frame_index)], dict):
        state_payload = frames[int(frame_index)].get("state")
        return state_payload if isinstance(state_payload, dict) else None
    return None


def _accept_label(
    *,
    stable: bool,
    confidence: float,
    phase: str,
    corner_risk: str,
    phase_filter: set[str] | None,
    corner_risk_filter: set[str] | None,
    stable_only: bool,
    confidence_threshold: float,
) -> bool:
    if stable_only and not stable:
        return False
    if confidence < confidence_threshold:
        return False
    if phase_filter is not None and phase not in phase_filter:
        return False
    if corner_risk_filter is not None and corner_risk not in corner_risk_filter:
        return False
    return True


def _weight_for_label(
    *,
    stable: bool,
    confidence: float,
    regret: float,
    winner: str,
    base_action: str,
    p6144: float,
    mode: str,
) -> tuple[float, str]:
    if mode == "uniform":
        return 1.0, "uniform"
    if mode != "confidence_regret":
        raise ValueError(f"Unsupported label_weight_mode: {mode}")
    return confidence_regret_weight(
        stable=stable,
        confidence=confidence,
        regret=regret,
        winner=winner,
        base_action=base_action,
        p6144=p6144,
    )


def _examples_from_state_label(
    *,
    group_id: str,
    state: SimState,
    action_names: list[str],
    winner: str,
    base_action: str,
    stable: bool,
    confidence: float,
    regret: float,
    p6144: float,
    phase: str,
    corner_risk: str,
    label_weight_mode: str,
    starter_tile: int | None,
) -> list[PairExample]:
    if winner not in action_names:
        return []
    group_weight, _reason = _weight_for_label(
        stable=stable,
        confidence=confidence,
        regret=regret,
        winner=winner,
        base_action=base_action,
        p6144=p6144,
        mode=label_weight_mode,
    )
    examples: list[PairExample] = []
    for other in action_names:
        if other == winner:
            continue
        forward = pair_features(state, winner, other, starter_tile)
        reverse = -forward
        for first, second, target, features in (
            (winner, other, 1.0, forward),
            (other, winner, 0.0, reverse),
        ):
            examples.append(
                PairExample(
                    group_id=group_id,
                    first_action=first,
                    second_action=second,
                    target=float(target),
                    features=features,
                    weight=float(group_weight),
                    phase=phase,
                    corner_risk=corner_risk,
                    base_action=base_action,
                    winner=winner,
                    oracle_regret=float(regret),
                )
            )
    return examples


def examples_from_endgame_label_file(path: Path, config: ActionPriorTrainConfig) -> list[PairExample]:
    payload = json.loads(path.read_text())
    phase_filter = set(config.phase_filter) if config.phase_filter else None
    corner_filter = set(config.corner_risk_filter) if config.corner_risk_filter else None
    examples: list[PairExample] = []
    state_cache: dict[tuple[str, int], dict[str, Any] | None] = {}
    for item in payload.get("labels", []):
        if not isinstance(item, dict):
            continue
        winner = str(item.get("winner") or "")
        if not winner or winner == "tie":
            continue
        source_replay = item.get("source_replay")
        frame_index = item.get("source_frame_index")
        if not isinstance(source_replay, str) or frame_index is None:
            continue
        cache_key = (source_replay, int(frame_index))
        if cache_key not in state_cache:
            state_cache[cache_key] = _state_payload_from_replay_frame(Path(source_replay), int(frame_index))
        state_payload = state_cache[cache_key]
        if state_payload is None:
            continue
        state = state_from_replay_payload(state_payload)
        features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
        phase = str(features.get("phase") or _phase_for_board(state.board, config.starter_tile))
        corner_risk = str(features.get("corner_risk") or _corner_risk_bucket_for_state(state, config.starter_tile))
        confidence = _safe_float(item.get("bootstrap_winner_fraction"), 0.5)
        stable = bool(item.get("stable"))
        if not _accept_label(
            stable=stable,
            confidence=confidence,
            phase=phase,
            corner_risk=corner_risk,
            phase_filter=phase_filter,
            corner_risk_filter=corner_filter,
            stable_only=config.stable_only,
            confidence_threshold=config.confidence_threshold,
        ):
            continue
        action_names = [str(row.get("action")) for row in item.get("action_results", []) if isinstance(row, dict) and row.get("action")]
        examples.extend(
            _examples_from_state_label(
                group_id=str(item.get("id", f"{path}:{len(examples)}")),
                state=state,
                action_names=action_names,
                winner=winner,
                base_action=str(item.get("base_action") or ""),
                stable=stable,
                confidence=confidence,
                regret=_safe_float(item.get("oracle_regret"), 0.0),
                p6144=_safe_float(item.get("winner_p6144"), 0.0),
                phase=phase,
                corner_risk=corner_risk,
                label_weight_mode=config.label_weight_mode,
                starter_tile=config.starter_tile,
            )
        )
    return examples


def examples_from_swing_label_file(path: Path, config: ActionPriorTrainConfig) -> list[PairExample]:
    payload = json.loads(path.read_text())
    phase_filter = set(config.phase_filter) if config.phase_filter else None
    corner_filter = set(config.corner_risk_filter) if config.corner_risk_filter else None
    examples: list[PairExample] = []
    for item in payload.get("labels", []):
        if not isinstance(item, dict) or not isinstance(item.get("state"), dict):
            continue
        label = item.get("label", {})
        if not isinstance(label, dict):
            continue
        winner = str(label.get("oracle_winner") or label.get("stable_winner") or "")
        if not winner or winner == "tie":
            continue
        state = state_from_replay_payload(item["state"])
        features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
        phase = str(features.get("phase") or _phase_for_board(state.board, config.starter_tile))
        corner_risk = str(features.get("corner_risk") or _corner_risk_bucket_for_state(state, config.starter_tile))
        confidence = _safe_float(label.get("min_bootstrap_winner_fraction"), 0.5)
        stable = bool(label.get("stable"))
        if not _accept_label(
            stable=stable,
            confidence=confidence,
            phase=phase,
            corner_risk=corner_risk,
            phase_filter=phase_filter,
            corner_risk_filter=corner_filter,
            stable_only=config.stable_only,
            confidence_threshold=config.confidence_threshold,
        ):
            continue
        action_names = [str(action) for action in label.get("actions", [])]
        if not action_names:
            action_names = [str(action) for action in label.get("by_action", {}).keys()]
        examples.extend(
            _examples_from_state_label(
                group_id=str(item.get("id", f"{path}:{len(examples)}")),
                state=state,
                action_names=action_names,
                winner=winner,
                base_action=str(item.get("base_action") or ""),
                stable=stable,
                confidence=confidence,
                regret=_safe_float(label.get("oracle_regret_at_max_horizon"), 0.0),
                p6144=0.0,
                phase=phase,
                corner_risk=corner_risk,
                label_weight_mode=config.label_weight_mode,
                starter_tile=config.starter_tile,
            )
        )
    return examples


def load_pair_examples(config: ActionPriorTrainConfig) -> list[PairExample]:
    examples: list[PairExample] = []
    for text_path in config.endgame_label_json:
        examples.extend(examples_from_endgame_label_file(Path(text_path), config))
    for text_path in config.swing_label_json:
        examples.extend(examples_from_swing_label_file(Path(text_path), config))
    return examples


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class ActionPriorModel:
    def __init__(
        self,
        weights: np.ndarray,
        bias: float,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        *,
        feature_names: list[str] | None = None,
        starter_tile: int | None = 1536,
    ) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_std = np.asarray(feature_std, dtype=np.float64)
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.starter_tile = starter_tile
        self._feature_indices = [
            FEATURE_NAMES.index(name) if name in FEATURE_NAMES else None
            for name in self.feature_names
        ]

    def _align_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64)
        if len(arr) == len(self.weights) and len(self.feature_names) == len(self.weights):
            if self.feature_names == FEATURE_NAMES[: len(self.feature_names)]:
                return arr
        aligned = np.zeros(len(self.feature_names), dtype=np.float64)
        for idx, source_idx in enumerate(self._feature_indices):
            if source_idx is not None and source_idx < len(arr):
                aligned[idx] = arr[source_idx]
        return aligned

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        aligned = self._align_features(features)
        return (aligned - self.feature_mean) / self.feature_std

    def logit(self, state: SimState, first_action: int | str, second_action: int | str) -> float:
        features = pair_features(state, first_action, second_action, self.starter_tile)
        return float(np.dot(self.weights, self._standardize(features)) + self.bias)

    def probability(self, state: SimState, first_action: int | str, second_action: int | str) -> float:
        return _sigmoid(self.logit(state, first_action, second_action))

    def to_json(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "feature_names": self.feature_names,
            "weights": [float(value) for value in self.weights],
            "bias": float(self.bias),
            "feature_mean": [float(value) for value in self.feature_mean],
            "feature_std": [float(value) for value in self.feature_std],
            "starter_tile": self.starter_tile,
        }
        if extra:
            payload.update(extra)
        return payload

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        write_json(path / "model.json", self.to_json(extra))

    @classmethod
    def load(cls, path: Path) -> "ActionPriorModel":
        model_path = path / "model.json" if path.is_dir() else path
        payload = json.loads(model_path.read_text())
        return cls(
            np.asarray(payload["weights"], dtype=np.float64),
            float(payload.get("bias", 0.0)),
            np.asarray(payload["feature_mean"], dtype=np.float64),
            np.asarray(payload["feature_std"], dtype=np.float64),
            feature_names=list(payload.get("feature_names", FEATURE_NAMES)),
            starter_tile=payload.get("starter_tile", 1536),
        )


def train_model(config: ActionPriorTrainConfig) -> tuple[ActionPriorModel, list[PairExample], dict[str, Any]]:
    examples = load_pair_examples(config)
    if not examples:
        raise ValueError("No pairwise action-prior examples were loaded")
    x_raw = np.vstack([example.features for example in examples]).astype(np.float64)
    y = np.asarray([example.target for example in examples], dtype=np.float64)
    sample_weight = np.asarray([example.weight for example in examples], dtype=np.float64)
    feature_mean = x_raw.mean(axis=0)
    feature_std = x_raw.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    x = (x_raw - feature_mean) / feature_std
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    rng = np.random.default_rng(int(config.seed))
    order = np.arange(len(examples), dtype=np.int32)
    losses: list[float] = []
    progress_rows: list[dict[str, object]] = []
    updates = 0
    start_time = time.perf_counter()
    for epoch in range(1, int(config.epochs) + 1):
        rng.shuffle(order)
        for idx_value in order:
            idx = int(idx_value)
            logit = float(np.dot(weights, x[idx]) + bias)
            pred = _sigmoid(logit)
            target = y[idx]
            weight = sample_weight[idx]
            loss = -weight * (target * math.log(max(pred, 1e-12)) + (1.0 - target) * math.log(max(1.0 - pred, 1e-12)))
            losses.append(float(loss))
            grad = weight * (pred - target)
            weights -= float(config.lr) * (grad * x[idx] + float(config.l2) * weights)
            bias -= float(config.lr) * grad
            updates += 1
            if config.progress_every > 0 and updates % int(config.progress_every) == 0:
                progress_rows.append(
                    {
                        "updates": int(updates),
                        "epoch": int(epoch),
                        "elapsed_s": time.perf_counter() - start_time,
                        "mean_loss_recent": float(mean(losses[-int(config.progress_every) :])),
                    }
                )
    model = ActionPriorModel(weights, bias, feature_mean, feature_std, starter_tile=config.starter_tile)
    summary = summarize_model(model, examples)
    summary.update(
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updates": int(updates),
            "epochs": int(config.epochs),
            "pair_examples": len(examples),
            "groups": len({example.group_id for example in examples}),
            "mean_loss": float(mean(losses)) if losses else 0.0,
            "label_weight_mode": config.label_weight_mode,
            "weight_mean": float(mean(float(example.weight) for example in examples)),
            "weight_max": float(max(float(example.weight) for example in examples)),
            "phase_counts": dict(Counter(example.phase for example in examples if example.target == 1.0)),
            "corner_risk_counts": dict(Counter(example.corner_risk for example in examples if example.target == 1.0)),
        }
    )
    summary["progress_rows"] = progress_rows
    return model, examples, summary


def summarize_model(model: ActionPriorModel, examples: list[PairExample]) -> dict[str, Any]:
    correct = 0
    weighted_correct = 0.0
    weight_total = 0.0
    base_groups = {}
    for example in examples:
        logit = float(np.dot(model.weights, model._standardize(example.features)) + model.bias)
        pred = 1.0 if _sigmoid(logit) >= 0.5 else 0.0
        if pred == example.target:
            correct += 1
            weighted_correct += float(example.weight)
        weight_total += float(example.weight)
        if example.target == 1.0 and example.second_action == example.base_action:
            base_groups[example.group_id] = _sigmoid(logit)
    corrective_probs = [
        prob
        for example in examples
        for group_id, prob in base_groups.items()
        if example.group_id == group_id and example.winner != example.base_action
    ]
    return {
        "pair_accuracy": float(correct / len(examples)) if examples else 0.0,
        "weighted_pair_accuracy": float(weighted_correct / weight_total) if weight_total else 0.0,
        "winner_vs_base_groups": len(base_groups),
        "winner_vs_base_mean_probability": float(mean(base_groups.values())) if base_groups else 0.0,
        "corrective_winner_vs_base_mean_probability": float(mean(corrective_probs)) if corrective_probs else 0.0,
        "corrective_groups": len({example.group_id for example in examples if example.target == 1.0 and example.winner != example.base_action}),
    }


def calibrate_action_prior(config: ActionPriorTrainConfig) -> Path:
    run_dir = Path("threes_rl/runs") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(config))
    model, _examples, summary = train_model(config)
    progress_rows = list(summary.pop("progress_rows", []))
    if progress_rows:
        write_progress_csv(run_dir / "progress.csv", progress_rows)
    write_json(run_dir / "summary.json", summary)
    model.save(
        run_dir / "latest",
        extra={
            "action_prior_config": asdict(config),
            "summary": summary,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"latest_checkpoint={run_dir / 'latest'}", flush=True)
    return run_dir / "latest"


@dataclass(frozen=True)
class ActionPriorPolicyConfig:
    checkpoint: str
    min_probability: float = 0.70
    max_base_margin: float = 0.01
    max_override_rate: float = 0.02
    stratum: str | None = "endgame_3072p/medium_corner_risk"
    starter_tile: int | None = 1536


def _normalized_margin(action_values: list[tuple[int, float]]) -> tuple[float, float]:
    if len(action_values) < 2:
        return 0.0, 0.0
    ordered = sorted(action_values, key=lambda item: (-float(item[1]), int(item[0])))
    margin = float(ordered[0][1]) - float(ordered[1][1])
    scale = max(1.0, abs(float(ordered[0][1])), abs(float(ordered[1][1])))
    return margin, margin / scale


def _select_from_values(base_policy: object, action_values: list[tuple[int, float]], rng: np.random.Generator) -> int:
    if hasattr(base_policy, "_select_action"):
        return int(base_policy._select_action(action_values, rng))
    best_value = max(value for _action, value in action_values)
    best_actions = [action for action, value in action_values if value == best_value]
    return int(best_actions[int(rng.integers(len(best_actions)))])


class ActionPriorPolicy:
    def __init__(self, base_policy: object, base_spec: str, config: ActionPriorPolicyConfig) -> None:
        self.base_policy = base_policy
        self.base_spec = base_spec
        self.config = config
        self.model = ActionPriorModel.load(Path(config.checkpoint))
        self.name = (
            f"action_prior|{base_spec}|checkpoint={config.checkpoint}|prob={config.min_probability:g}"
            f"|max_base_margin={config.max_base_margin:g}|max_rate={config.max_override_rate:g}"
            f"|stratum={config.stratum or 'all'}"
        )
        self._decision_count = 0
        self._eligible_count = 0
        self._override_count = 0
        self._skip_counts: dict[str, int] = {
            "no_action_values": 0,
            "not_enough_actions": 0,
            "stratum": 0,
            "base_margin": 0,
            "rate": 0,
            "low_probability": 0,
            "model_prefers_base": 0,
        }

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        self._decision_count += 1
        if not hasattr(self.base_policy, "action_values"):
            self._skip_counts["no_action_values"] += 1
            return int(self.base_policy(state, sim, rng))
        action_values = list(self.base_policy.action_values(state, sim))
        if len(action_values) < 2:
            self._skip_counts["not_enough_actions"] += 1
            if action_values:
                return int(action_values[0][0])
            return int(self.base_policy(state, sim, rng))
        base_action = _select_from_values(self.base_policy, action_values, rng)
        ordered = sorted(action_values, key=lambda item: (-float(item[1]), int(item[0])))
        top_two = [int(ordered[0][0]), int(ordered[1][0])]
        if int(base_action) not in top_two:
            top_two[-1] = int(base_action)
        if self.config.stratum and self.config.stratum.lower() != "all":
            if stratum_for_state(state, self.config.starter_tile) != self.config.stratum:
                self._skip_counts["stratum"] += 1
                return int(base_action)
        _margin, normalized = _normalized_margin(ordered)
        if normalized > self.config.max_base_margin:
            self._skip_counts["base_margin"] += 1
            return int(base_action)
        self._eligible_count += 1
        allowed = max(1, int(self._decision_count * self.config.max_override_rate))
        if self._override_count >= allowed:
            self._skip_counts["rate"] += 1
            return int(base_action)
        alternatives = [action for action in top_two if int(action) != int(base_action)]
        if not alternatives:
            self._skip_counts["model_prefers_base"] += 1
            return int(base_action)
        alt = int(alternatives[0])
        probability = self.model.probability(state, DIRECTION_NAMES[alt], DIRECTION_NAMES[int(base_action)])
        if probability < self.config.min_probability:
            reverse_probability = self.model.probability(state, DIRECTION_NAMES[int(base_action)], DIRECTION_NAMES[alt])
            if reverse_probability >= probability:
                self._skip_counts["model_prefers_base"] += 1
            else:
                self._skip_counts["low_probability"] += 1
            return int(base_action)
        self._override_count += 1
        return int(alt)

    def summary_stats(self) -> dict[str, Any]:
        return {
            "base_policy": self.base_spec,
            "config": {
                "checkpoint": self.config.checkpoint,
                "min_probability": float(self.config.min_probability),
                "max_base_margin": float(self.config.max_base_margin),
                "max_override_rate": float(self.config.max_override_rate),
                "stratum": self.config.stratum,
                "starter_tile": self.config.starter_tile,
            },
            "decisions": int(self._decision_count),
            "eligible": int(self._eligible_count),
            "overrides": int(self._override_count),
            "skip_counts": dict(self._skip_counts),
        }


def parse_action_prior_spec(spec: str, make_policy_fn) -> ActionPriorPolicy:
    parts = spec.split("|")
    if len(parts) < 3 or parts[0] != "action_prior":
        raise ValueError(f"Unsupported action prior spec: {spec}")
    base_spec = parts[1]
    options: dict[str, str] = {}
    for part in parts[2:]:
        if "=" not in part:
            raise ValueError(f"Action prior option must be key=value: {part}")
        key, value = part.split("=", 1)
        options[key.strip().lower()] = value.strip()
    checkpoint = options.get("checkpoint")
    if not checkpoint:
        raise ValueError("action_prior specs require checkpoint=<path>")
    stratum = options.get("stratum", "endgame_3072p/medium_corner_risk")
    if stratum.lower() == "all":
        stratum = None
    config = ActionPriorPolicyConfig(
        checkpoint=checkpoint,
        min_probability=float(options.get("prob", options.get("min_probability", 0.70))),
        max_base_margin=float(options.get("max_base_margin", options.get("margin", 0.01))),
        max_override_rate=float(options.get("max_rate", options.get("max_override_rate", 0.02))),
        stratum=stratum,
        starter_tile=None if options.get("starter", "1536").lower() == "none" else int(options.get("starter", "1536")),
    )
    return ActionPriorPolicy(make_policy_fn(base_spec), base_spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an action-conditioned top-two correction prior.")
    parser.add_argument("--run-name", default=f"action_prior_{int(time.time())}")
    parser.add_argument("--endgame-label-json", type=Path, nargs="+", action="append")
    parser.add_argument("--swing-label-json", type=Path, nargs="+", action="append")
    parser.add_argument("--stable-only", action="store_true", default=True)
    parser.add_argument("--include-unstable", dest="stable_only", action="store_false")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--phase-filter")
    parser.add_argument("--corner-risk-filter")
    parser.add_argument("--label-weight-mode", choices=["uniform", "confidence_regret"], default="confidence_regret")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    endgame_paths = _flatten_paths(args.endgame_label_json)
    swing_paths = _flatten_paths(args.swing_label_json)
    if not endgame_paths and not swing_paths:
        raise ValueError("Pass at least one --endgame-label-json or --swing-label-json")
    config = ActionPriorTrainConfig(
        run_name=args.run_name,
        endgame_label_json=[str(path) for path in endgame_paths],
        swing_label_json=[str(path) for path in swing_paths],
        stable_only=bool(args.stable_only),
        confidence_threshold=float(args.confidence_threshold),
        phase_filter=parse_optional_filter(args.phase_filter),
        corner_risk_filter=parse_optional_filter(args.corner_risk_filter),
        label_weight_mode=args.label_weight_mode,
        starter_tile=parse_starter(args.starter),
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed),
        progress_every=int(args.progress_every),
    )
    calibrate_action_prior(config)


if __name__ == "__main__":
    main()

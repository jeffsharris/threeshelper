"""Equal-capacity identity-initialized context residual scaffolding for R1.5a."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.geometry_forensics import board_without_free_starter
from threes_rl.ntuple import PHASE4_NAMES, phase4_index_for_board
from threes_rl.sim import SimState, ThreesSim, rank_for_value, score_tile


MODEL_VERSION = "r15a_context_residual_v1"
MODEL_MODES = {"board_stage_only", "board_plus_context"}
INPUT_WIDTH = 64
BOARD_WIDTH = 32
CONTEXT_WIDTH = 32
HIDDEN_WIDTH = 32
TARGET_HORIZON = 40
BONUS_VALUES = (6, 12, 24, 48, 96, 192, 384, 768)
RETURN_BIN_EDGES = (0.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0, 32_000.0, 64_000.0, float("inf"))
OUTPUT_NAMES = (
    "expected_return_residual",
    *(f"return_bin_{index}" for index in range(len(RETURN_BIN_EDGES) - 1)),
    "survival_logit",
    "first_1536_logit",
    "first_3072_logit",
    "anchor_preserved_logit",
)


def _schema_payload() -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "input_width": INPUT_WIDTH,
        "board_width": BOARD_WIDTH,
        "context_width": CONTEXT_WIDTH,
        "hidden_width": HIDDEN_WIDTH,
        "primary_target_horizon": TARGET_HORIZON,
        "primary_target": (
            "score_accumulated_0_to_H + frozen_incumbent_leaf(live_s_H) "
            "- frozen_incumbent_leaf(s_0); terminal bootstrap=0"
        ),
        "stage_names": list(PHASE4_NAMES),
        "bonus_values": list(BONUS_VALUES),
        "return_bin_edges": ["inf" if np.isinf(value) else value for value in RETURN_BIN_EDGES],
        "output_names": list(OUTPUT_NAMES),
    }


def schema_sha256() -> str:
    raw = json.dumps(_schema_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _probability_map_after_visible(state: SimState, sim: ThreesSim) -> dict[str, Any]:
    counts, small_pos, small_seen_total, span_small_pos, large_pending = sim._consume_preview(
        state.small_counts,
        state.small_pos,
        state.small_seen_total,
        state.span_small_pos,
        state.large_pending,
        state.preview.label,
    )
    options = sim.preview_options(
        counts,
        small_pos,
        small_seen_total,
        span_small_pos,
        large_pending,
        state.max_tile,
    )
    small_joint = {"red": 0.0, "blue": 0.0, "gray": 0.0}
    plus_value_joint = {value: 0.0 for value in BONUS_VALUES}
    p_plus = 0.0
    for option in options:
        probability = float(option.probability)
        if option.preview.kind == "bonus":
            p_plus += probability
            candidates = tuple(int(value) for value in option.preview.candidates)
            if candidates:
                for value in candidates:
                    if value in plus_value_joint:
                        plus_value_joint[value] += probability / len(candidates)
        elif option.preview.kind in small_joint:
            small_joint[option.preview.kind] += probability
    plus_conditional = {
        value: (plus_value_joint[value] / p_plus if p_plus > 0.0 else 0.0)
        for value in BONUS_VALUES
    }
    remaining_small = max(1, sum(int(value) for value in counts.values()))
    bag_probabilities = {
        kind: int(counts.get(kind, 0)) / remaining_small
        for kind in ("red", "blue", "gray")
    }
    if small_seen_total < 21:
        distance_to_forced_plus = 21 - int(small_seen_total)
    elif large_pending:
        distance_to_forced_plus = max(0, 20 - int(span_small_pos))
    else:
        distance_to_forced_plus = max(0, 20 - int(span_small_pos)) + 1
    return {
        "post_visible_small_counts": counts,
        "post_visible_small_pos": int(small_pos),
        "post_visible_small_seen_total": int(small_seen_total),
        "post_visible_span_small_pos": int(span_small_pos),
        "post_visible_large_pending": bool(large_pending),
        "p_plus_next": float(p_plus),
        "next_small_joint": small_joint,
        "next_plus_value_joint": plus_value_joint,
        "next_plus_value_conditional": plus_conditional,
        "post_visible_bag_probabilities": bag_probabilities,
        "distance_to_forced_plus": int(distance_to_forced_plus),
    }


def context_metadata(state: SimState, sim: ThreesSim, starter_tile: int | None = 1536) -> dict[str, Any]:
    mechanics = _probability_map_after_visible(state, sim)
    board = np.asarray(state.board, dtype=np.int32)
    masked = board_without_free_starter(board, starter_tile)
    built_max = int(max_tile_excluding_initial_starter(board, starter_tile))
    support_values = [int(value) for value in masked.reshape(-1) if 0 < int(value) < max(384, built_max)]
    top_ranks = [rank_for_value(int(value)) for value in board[0]]
    return {
        **mechanics,
        "phase4_stage_index": int(phase4_index_for_board(board, starter_tile=starter_tile)),
        "phase4_stage": PHASE4_NAMES[phase4_index_for_board(board, starter_tile=starter_tile)],
        "visible_preview_kind": state.preview.kind,
        "visible_preview_candidates": list(state.preview.candidates),
        "empty_count": int(np.count_nonzero(board == 0)),
        "legal_count": len(sim.legal_actions(state)),
        "anchor_value": int(board[0, 0]),
        "top_edge_ranks": top_ranks,
        "top_edge_descending": all(top_ranks[index] >= top_ranks[index + 1] for index in range(3)),
        "support_score_mass": int(sum(score_tile(value) for value in support_values)),
        "built_max": built_max,
    }


def _board_features(state: SimState, sim: ThreesSim, starter_tile: int | None) -> np.ndarray:
    board = np.asarray(state.board, dtype=np.int32)
    ranks = np.asarray([rank_for_value(int(value)) / 14.0 for value in board.reshape(-1)], dtype=np.float64)
    stage = int(phase4_index_for_board(board, starter_tile=starter_tile))
    stage_one_hot = np.zeros(4, dtype=np.float64)
    stage_one_hot[stage] = 1.0
    built_max = int(max_tile_excluding_initial_starter(board, starter_tile))
    top_ranks = [rank_for_value(int(value)) for value in board[0]]
    masked = board_without_free_starter(board, starter_tile)
    support_values = [int(value) for value in masked.reshape(-1) if 0 < int(value) < max(384, built_max)]
    max_positions = np.argwhere(masked == built_max) if built_max > 0 else np.empty((0, 2), dtype=np.int64)
    max_manhattan = min((int(row) + int(col) for row, col in max_positions), default=0)
    extra = np.asarray(
        [
            np.count_nonzero(board == 0) / 16.0,
            len(sim.legal_actions(state)) / 4.0,
            rank_for_value(int(board[0, 0])) / 14.0,
            rank_for_value(built_max) / 14.0,
            *(rank / 14.0 for rank in top_ranks),
            float(np.sum(ranks)) / 16.0,
            float(all(top_ranks[index] >= top_ranks[index + 1] for index in range(3))),
            min(1.0, sum(score_tile(value) for value in support_values) / 100_000.0),
            max_manhattan / 6.0,
        ],
        dtype=np.float64,
    )
    features = np.concatenate((ranks, stage_one_hot, extra))
    if features.shape != (BOARD_WIDTH,):
        raise AssertionError(f"Board feature width mismatch: {features.shape}")
    return features


def _context_features(state: SimState, sim: ThreesSim) -> np.ndarray:
    metadata = _probability_map_after_visible(state, sim)
    preview_one_hot = np.zeros(4, dtype=np.float64)
    preview_one_hot[("red", "blue", "gray", "bonus").index(state.preview.kind)] = 1.0
    visible_candidates = np.zeros(len(BONUS_VALUES), dtype=np.float64)
    if state.preview.kind == "bonus" and state.preview.candidates:
        for value in state.preview.candidates:
            if int(value) in BONUS_VALUES:
                visible_candidates[BONUS_VALUES.index(int(value))] = 1.0 / len(state.preview.candidates)
    plus_conditional = np.asarray(
        [metadata["next_plus_value_conditional"][value] for value in BONUS_VALUES],
        dtype=np.float64,
    )
    small_joint = metadata["next_small_joint"]
    bag = metadata["post_visible_bag_probabilities"]
    tail = np.asarray(
        [
            float(metadata["p_plus_next"]),
            float(small_joint["red"]),
            float(small_joint["blue"]),
            float(small_joint["gray"]),
            float(bag["red"]),
            float(bag["blue"]),
            float(bag["gray"]),
            int(metadata["post_visible_small_pos"]) / 12.0,
            min(1.0, int(metadata["post_visible_small_seen_total"]) / 400.0),
            int(metadata["post_visible_span_small_pos"]) / 20.0,
            float(metadata["post_visible_large_pending"]),
            min(1.0, int(metadata["distance_to_forced_plus"]) / 21.0),
        ],
        dtype=np.float64,
    )
    features = np.concatenate((preview_one_hot, visible_candidates, plus_conditional, tail))
    if features.shape != (CONTEXT_WIDTH,):
        raise AssertionError(f"Context feature width mismatch: {features.shape}")
    return features


def encode_state(
    state: SimState,
    sim: ThreesSim,
    *,
    mode: str,
    starter_tile: int | None = 1536,
) -> np.ndarray:
    if mode not in MODEL_MODES:
        raise ValueError(f"Unsupported context residual mode: {mode}")
    board = _board_features(state, sim, starter_tile)
    context = _context_features(state, sim)
    if mode == "board_stage_only":
        context = np.zeros_like(context)
    encoded = np.concatenate((board, context))
    if encoded.shape != (INPUT_WIDTH,):
        raise AssertionError(f"Input feature width mismatch: {encoded.shape}")
    return encoded


class ContextResidualModel:
    """Untrained equal-capacity residual model with exact zero-output identity."""

    def __init__(
        self,
        *,
        mode: str,
        seed: int = 20260711,
        w1: np.ndarray | None = None,
        b1: np.ndarray | None = None,
        w2: np.ndarray | None = None,
        b2: np.ndarray | None = None,
    ) -> None:
        if mode not in MODEL_MODES:
            raise ValueError(f"Unsupported context residual mode: {mode}")
        self.mode = mode
        self.seed = int(seed)
        output_width = len(PHASE4_NAMES) * len(OUTPUT_NAMES)
        rng = np.random.default_rng(self.seed)
        self.w1 = (
            rng.normal(0.0, 1.0 / np.sqrt(INPUT_WIDTH), size=(INPUT_WIDTH, HIDDEN_WIDTH))
            if w1 is None else np.asarray(w1, dtype=np.float64).copy()
        )
        self.b1 = np.zeros(HIDDEN_WIDTH, dtype=np.float64) if b1 is None else np.asarray(b1, dtype=np.float64).copy()
        self.w2 = np.zeros((HIDDEN_WIDTH, output_width), dtype=np.float64) if w2 is None else np.asarray(w2, dtype=np.float64).copy()
        self.b2 = np.zeros(output_width, dtype=np.float64) if b2 is None else np.asarray(b2, dtype=np.float64).copy()
        expected_shapes = {
            "w1": (INPUT_WIDTH, HIDDEN_WIDTH),
            "b1": (HIDDEN_WIDTH,),
            "w2": (HIDDEN_WIDTH, output_width),
            "b2": (output_width,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"Incompatible {name} shape: {getattr(self, name).shape}, expected {shape}")

    @property
    def parameter_count(self) -> int:
        return int(sum(array.size for array in (self.w1, self.b1, self.w2, self.b2)))

    def predict(self, state: SimState, sim: ThreesSim, *, starter_tile: int | None = 1536) -> dict[str, float]:
        encoded = encode_state(state, sim, mode=self.mode, starter_tile=starter_tile)
        hidden = np.tanh(encoded @ self.w1 + self.b1)
        raw = (hidden @ self.w2 + self.b2).reshape(len(PHASE4_NAMES), len(OUTPUT_NAMES))
        stage = int(phase4_index_for_board(state.board, starter_tile=starter_tile))
        return {name: float(raw[stage, index]) for index, name in enumerate(OUTPUT_NAMES)}

    def residual_value(self, state: SimState, sim: ThreesSim, *, starter_tile: int | None = 1536) -> float:
        return self.predict(state, sim, starter_tile=starter_tile)["expected_return_residual"]

    def total_value(
        self,
        frozen_incumbent_value: float,
        state: SimState,
        sim: ThreesSim,
        *,
        starter_tile: int | None = 1536,
    ) -> float:
        return float(frozen_incumbent_value + self.residual_value(state, sim, starter_tile=starter_tile))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "w1.npy", self.w1)
        np.save(path / "b1.npy", self.b1)
        np.save(path / "w2.npy", self.w2)
        np.save(path / "b2.npy", self.b2)
        meta = {
            **_schema_payload(),
            "schema_sha256": schema_sha256(),
            "mode": self.mode,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
            "zero_output_identity": bool(np.all(self.w2 == 0.0) and np.all(self.b2 == 0.0)),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "ContextResidualModel":
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("model_version") != MODEL_VERSION or meta.get("schema_sha256") != schema_sha256():
            raise ValueError("Incompatible context residual feature/target schema")
        model = cls(
            mode=str(meta["mode"]),
            seed=int(meta["seed"]),
            w1=np.load(path / "w1.npy"),
            b1=np.load(path / "b1.npy"),
            w2=np.load(path / "w2.npy"),
            b2=np.load(path / "b2.npy"),
        )
        if model.parameter_count != int(meta["parameter_count"]):
            raise ValueError("Context residual parameter count mismatch")
        return model

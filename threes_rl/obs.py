"""Observation encoders for Threes RL agents."""

from __future__ import annotations

import numpy as np

from threes_rl.sim import SimState, ThreesSim, rank_for_value

NUM_RANKS = 16
PREVIEW_KINDS = ("blue", "red", "gray", "bonus")


def observation_size(encoder: str = "full") -> int:
    if encoder == "full":
        return 4 * 4 * NUM_RANKS + len(PREVIEW_KINDS) + NUM_RANKS + 6
    if encoder == "board_only":
        return 4 * 4 * NUM_RANKS + len(PREVIEW_KINDS) + NUM_RANKS
    raise ValueError(f"Unsupported observation encoder: {encoder}")


def encode_observation(state: SimState, sim: ThreesSim, encoder: str = "full") -> np.ndarray:
    board = np.zeros((4, 4, NUM_RANKS), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            rank = min(NUM_RANKS - 1, rank_for_value(int(state.board[r, c])))
            board[r, c, rank] = 1.0

    preview_kind = np.zeros(len(PREVIEW_KINDS), dtype=np.float32)
    preview_kind[PREVIEW_KINDS.index(state.preview.kind)] = 1.0

    candidates = np.zeros(NUM_RANKS, dtype=np.float32)
    if state.preview.kind == "bonus":
        for value in state.preview.candidates:
            candidates[min(NUM_RANKS - 1, rank_for_value(value))] = 1.0
    elif state.preview.value is not None:
        candidates[min(NUM_RANKS - 1, rank_for_value(state.preview.value))] = 1.0

    parts = [board.reshape(-1), preview_kind, candidates]
    if encoder == "full":
        safe_smalls = sim.safe_smalls_until_large_possible(state)
        if safe_smalls is None:
            safe_norm = 1.0
        else:
            safe_norm = min(1.0, float(safe_smalls) / 21.0)
        extras = np.asarray(
            [
                state.small_counts.get("red", 0) / 4.0,
                state.small_counts.get("blue", 0) / 4.0,
                state.small_counts.get("gray", 0) / 4.0,
                min(1.0, state.span_small_pos / 20.0),
                1.0 if state.large_pending else 0.0,
                safe_norm,
            ],
            dtype=np.float32,
        )
        parts.append(extras)
    elif encoder != "board_only":
        raise ValueError(f"Unsupported observation encoder: {encoder}")

    return np.concatenate(parts).astype(np.float32, copy=False)

"""Gymnasium wrapper for the Threes simulator."""

from __future__ import annotations

from math import log1p
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from threes_rl.obs import encode_observation, observation_size
from threes_rl.sim import SimState, ThreesSim, score_board

ILLEGAL_PENALTY = -1.0


class ThreesEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        starter_tile: Optional[int] = 1536,
        obs_encoder: str = "full",
        reward_mode: str = "final_score",
    ) -> None:
        super().__init__()
        self.starter_tile = starter_tile
        self.obs_encoder = obs_encoder
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)
        self.sim = ThreesSim(self.rng, starter_tile=starter_tile)
        self.state: Optional[SimState] = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size(obs_encoder),),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.sim = ThreesSim(self.rng, starter_tile=self.starter_tile)
        self.state = self.sim.reset()
        return self._obs(), self._info(self.state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        before = self.state
        next_state, step_info = self.sim.step(before, int(action))
        if not step_info.moved:
            info = self._info(before)
            info["moved"] = False
            info["illegal_action"] = True
            return self._obs(), ILLEGAL_PENALTY, False, False, info

        self.state = next_state
        reward = self._reward(before, next_state, step_info)
        terminated = bool(next_state.game_over)
        info = self._info(next_state)
        info["moved"] = True
        info["illegal_action"] = False
        info["inserted_value"] = step_info.inserted_value
        info["inserted_pos"] = step_info.inserted_pos
        info["merge_score_delta"] = step_info.merge_score_delta
        info["score_delta"] = step_info.score_delta
        info["terminal_merge"] = step_info.terminal_merge
        if terminated:
            info["final_score"] = score_board(next_state.board)
        return self._obs(), reward, terminated, False, info

    def _reward(self, before: SimState, after: SimState, step_info: Any) -> float:
        if self.reward_mode == "final_score":
            return float(score_board(after.board) if after.game_over else 0.0)
        if self.reward_mode == "score_delta":
            return float(step_info.score_delta)
        if self.reward_mode == "merge_delta":
            return float(step_info.merge_score_delta)
        if self.reward_mode == "log_score_delta":
            return float(log1p(max(0, step_info.score_delta)))
        if self.reward_mode == "survival":
            return 1.0
        raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")

    def _obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Call reset() before requesting observations.")
        return encode_observation(self.state, self.sim, self.obs_encoder)

    def _info(self, state: SimState) -> dict[str, Any]:
        info: dict[str, Any] = {
            "legal_mask": self.sim.legal_mask(state),
            "score": score_board(state.board),
            "max_tile": state.max_tile,
            "move_count": state.move_count,
            "preview": state.preview,
            "tile_cycle_snapshot": self.sim.tile_cycle_snapshot(state),
        }
        if state.game_over:
            info["final_score"] = score_board(state.board)
        return info

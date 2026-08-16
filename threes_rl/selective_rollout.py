"""Selective top-two continuation re-search policy wrappers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

import numpy as np

from threes_rl.ntuple import PHASE4_NAMES, max_tile_excluding_free_starter, phase4_index_for_board
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board
from threes_rl.train_td import copy_state

ROLLOUT_POLICY_RNG_OFFSET = 3_000_003


@dataclass(frozen=True)
class RolloutGateConfig:
    margin_threshold: float = 0.002
    repeats: int = 16
    horizon: int = 32
    max_gate_rate: float = 0.02
    stratum: str = "endgame_3072p/medium_corner_risk"
    starter_tile: int | None = 1536


def _state_key(state: SimState, top_two_actions: tuple[int, int]) -> str:
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
            "top_two": [int(action) for action in top_two_actions],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _seed_for_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _normalized_margin(action_values: list[tuple[int, float]]) -> tuple[float, float]:
    if len(action_values) < 2:
        return 0.0, 0.0
    ordered = sorted(action_values, key=lambda item: (-float(item[1]), int(item[0])))
    margin = float(ordered[0][1]) - float(ordered[1][1])
    scale = max(1.0, abs(float(ordered[0][1])), abs(float(ordered[1][1])))
    return margin, margin / scale


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


def _stratum_for_state(state: SimState, starter_tile: int | None) -> str:
    phase = PHASE4_NAMES[phase4_index_for_board(state.board, starter_tile=starter_tile)]
    risk = _corner_risk_bucket_for_state(state, starter_tile)
    return f"{phase}/{risk}"


def _select_from_values(base_policy: object, action_values: list[tuple[int, float]], rng: np.random.Generator) -> int:
    if hasattr(base_policy, "_select_action"):
        return int(base_policy._select_action(action_values, rng))
    best_value = max(value for _action, value in action_values)
    best_actions = [action for action, value in action_values if value == best_value]
    return int(best_actions[int(rng.integers(len(best_actions)))])


def _rollout_action(
    *,
    base_policy: object,
    state: SimState,
    first_action: int,
    starter_tile: int | None,
    horizon: int,
    sim_seed: int,
) -> dict[str, float | int | bool]:
    sim = ThreesSim(np.random.default_rng(int(sim_seed)), starter_tile=starter_tile)
    policy_rng = np.random.default_rng(int(sim_seed) + ROLLOUT_POLICY_RNG_OFFSET)
    current = copy_state(state)
    start_score = int(score_board(current.board))
    start_move_count = int(current.move_count)

    next_state, info = sim.step(current, int(first_action))
    if not info.moved:
        return {
            "score_delta": -1_000_000_000,
            "moves_delta": 0,
            "max_tile_excl_starter": max_tile_excluding_free_starter(current.board, starter_tile),
            "advanced_to_6144": False,
        }
    current = next_state

    while not current.game_over and current.move_count - start_move_count < int(horizon):
        legal = sim.legal_actions(current)
        if not legal:
            break
        action = int(base_policy(current, sim, policy_rng))
        next_state, info = sim.step(current, action)
        if not info.moved:
            break
        current = next_state

    final_score = int(score_board(current.board))
    max_tile_excl_starter = max_tile_excluding_free_starter(current.board, starter_tile)
    return {
        "score_delta": int(final_score - start_score),
        "moves_delta": int(current.move_count - start_move_count),
        "max_tile_excl_starter": int(max_tile_excl_starter),
        "advanced_to_6144": bool(max_tile_excl_starter >= 6144),
    }


class SelectiveRolloutPolicy:
    """Wrap a learned search policy with a rare top-two rollout re-ranker."""

    def __init__(self, base_policy: object, base_spec: str, config: RolloutGateConfig | None = None) -> None:
        self.base_policy = base_policy
        self.base_spec = base_spec
        self.config = config or RolloutGateConfig()
        if self.config.repeats <= 0:
            raise ValueError("rollout repeats must be positive")
        if self.config.horizon <= 0:
            raise ValueError("rollout horizon must be positive")
        if self.config.margin_threshold < 0.0:
            raise ValueError("margin threshold must be nonnegative")
        if self.config.max_gate_rate < 0.0:
            raise ValueError("max gate rate must be nonnegative")
        self.name = (
            f"selective_rollout|{base_spec}|margin={self.config.margin_threshold:g}"
            f"|repeats={self.config.repeats}|horizon={self.config.horizon}"
            f"|max_rate={self.config.max_gate_rate:g}|stratum={self.config.stratum}"
        )
        self._decision_count = 0
        self._eligible_count = 0
        self._gate_count = 0
        self._override_count = 0
        self._cache_hits = 0
        self._rollout_count = 0
        self._skip_counts: dict[str, int] = {
            "no_action_values": 0,
            "not_enough_actions": 0,
            "stratum": 0,
            "margin": 0,
            "rate": 0,
        }
        self._cache: dict[str, int] = {}

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
        top_two = (int(ordered[0][0]), int(ordered[1][0]))
        if _stratum_for_state(state, self.config.starter_tile) != self.config.stratum:
            self._skip_counts["stratum"] += 1
            return int(base_action)

        _margin, normalized = _normalized_margin(ordered)
        if normalized > self.config.margin_threshold:
            self._skip_counts["margin"] += 1
            return int(base_action)

        self._eligible_count += 1
        allowed = max(1, int(self._decision_count * self.config.max_gate_rate))
        if self._gate_count >= allowed:
            self._skip_counts["rate"] += 1
            return int(base_action)

        cache_key = _state_key(state, top_two)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return int(cached)

        chosen = self._rollout_choice(state, top_two)
        self._cache[cache_key] = int(chosen)
        self._gate_count += 1
        if int(chosen) != int(base_action):
            self._override_count += 1
        return int(chosen)

    def _rollout_choice(self, state: SimState, top_two: tuple[int, int]) -> int:
        key = _state_key(state, top_two)
        base_seed = _seed_for_key(key)
        rows = []
        for action in top_two:
            repeats = []
            for repeat_idx in range(int(self.config.repeats)):
                seed = base_seed + repeat_idx * 1009
                repeats.append(
                    _rollout_action(
                        base_policy=self.base_policy,
                        state=state,
                        first_action=int(action),
                        starter_tile=self.config.starter_tile,
                        horizon=int(self.config.horizon),
                        sim_seed=int(seed),
                    )
                )
            self._rollout_count += len(repeats)
            rows.append(
                {
                    "action": int(action),
                    "p6144": sum(bool(item["advanced_to_6144"]) for item in repeats) / float(len(repeats)),
                    "mean_delta": float(mean(int(item["score_delta"]) for item in repeats)),
                    "median_delta": float(median(int(item["score_delta"]) for item in repeats)),
                }
            )
        winner = max(
            rows,
            key=lambda row: (
                float(row["p6144"]),
                float(row["mean_delta"]),
                float(row["median_delta"]),
                -int(row["action"]),
            ),
        )
        return int(winner["action"])

    def summary_stats(self) -> dict[str, Any]:
        return {
            "base_policy": self.base_spec,
            "config": {
                "margin_threshold": float(self.config.margin_threshold),
                "repeats": int(self.config.repeats),
                "horizon": int(self.config.horizon),
                "max_gate_rate": float(self.config.max_gate_rate),
                "stratum": self.config.stratum,
                "starter_tile": self.config.starter_tile,
            },
            "decisions": int(self._decision_count),
            "eligible": int(self._eligible_count),
            "gated": int(self._gate_count),
            "overrides": int(self._override_count),
            "cache_hits": int(self._cache_hits),
            "rollouts": int(self._rollout_count),
            "skip_counts": dict(self._skip_counts),
        }


def parse_selective_rollout_spec(spec: str, make_policy_fn) -> SelectiveRolloutPolicy:
    parts = spec.split("|")
    if len(parts) < 2 or parts[0] != "selective_rollout":
        raise ValueError(f"Unsupported selective rollout spec: {spec}")
    base_spec = parts[1]
    options: dict[str, str] = {}
    for part in parts[2:]:
        if "=" not in part:
            raise ValueError(f"Selective rollout option must be key=value: {part}")
        key, value = part.split("=", 1)
        options[key.strip().lower()] = value.strip()
    config = RolloutGateConfig(
        margin_threshold=float(options.get("margin", options.get("margin_threshold", 0.002))),
        repeats=int(options.get("repeats", 16)),
        horizon=int(options.get("horizon", 32)),
        max_gate_rate=float(options.get("max_rate", options.get("max_gate_rate", 0.02))),
        stratum=options.get("stratum", "endgame_3072p/medium_corner_risk"),
        starter_tile=None if options.get("starter", "1536").lower() == "none" else int(options.get("starter", "1536")),
    )
    return SelectiveRolloutPolicy(make_policy_fn(base_spec), base_spec, config)

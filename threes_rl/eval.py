"""Evaluate Threes policies on deterministic seed suites."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import sys
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from threes_rl.baselines import GreedyPolicy, RandomPolicy
from threes_rl.expectimax import CornerExpectimaxPolicy, ExpectimaxPolicy, NtupleExpectimaxPolicy
from threes_rl.ntuple import PHASE4_NAMES, NtuplePolicy, NtupleValue
from threes_rl.obs import encode_observation
from threes_rl.record_replay import preview_payload, record_replay_for_policy, state_payload
from threes_rl.run_artifacts import (
    safe_name,
    write_json,
    write_milestone_replays,
    write_pre_milestone_failure_replays,
    write_progress_chart,
    write_progress_csv,
    write_top_replays,
)
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, score_board, score_tile, simulate_base_move


@dataclass
class GameResult:
    seed: int
    score: int
    score_minus_starter: int
    moves: int
    max_tile: int
    max_tile_excl_starter: int
    terminal_tile: bool
    starter_tile: int | None = 1536


@dataclass(frozen=True)
class EvalJob:
    index: int
    seed: int
    starter_tile: int | None
    stream_ids: "EvalStreamIds | None" = None


@dataclass(frozen=True)
class EvalStreamIds:
    deck_stream_id: int
    slot_stream_id: int
    policy_stream_id: int
    evaluator_version: str = "split_exogenous_v1"

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluator_version": self.evaluator_version,
            "deck_stream_id": int(self.deck_stream_id),
            "slot_stream_id": int(self.slot_stream_id),
            "policy_stream_id": int(self.policy_stream_id),
        }


@dataclass
class EvalJobOutput:
    index: int
    result: GameResult
    replay: dict[str, Any] | None


_WORKER_POLICY: Any | None = None
_WORKER_POLICY_NAME = ""
_WORKER_MAX_MOVES = 5000
_WORKER_CAPTURE_REPLAY = False


class PpoPolicy:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        import torch

        from threes_rl.train_ppo import ActorCritic

        payload = torch.load(checkpoint, map_location=device)
        config = payload["config"]
        self.device = torch.device(device)
        self.obs_encoder = config.get("obs_encoder", "full")
        self.model = ActorCritic(int(config["obs_dim"])).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.name = f"ppo:{checkpoint}"

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        import torch

        obs = encode_observation(state, sim, self.obs_encoder)
        mask = sim.legal_mask(state)
        with torch.no_grad():
            logits, _value = self.model(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            logits = logits.squeeze(0)
            logits[~torch.as_tensor(mask, dtype=torch.bool, device=self.device)] = -1e9
            return int(torch.argmax(logits).item())


class AnchorGuardPolicy:
    """Filter or penalize root actions that immediately dislodge a large top-left anchor."""

    def __init__(self, base_policy, *, min_tile: int = 1536, penalty: float | None = None) -> None:
        self.base_policy = base_policy
        self.min_tile = int(min_tile)
        self.penalty = None if penalty is None else float(penalty)
        base_name = getattr(base_policy, "name", base_policy.__class__.__name__)
        mode = "guard" if self.penalty is None else f"penalty{self.penalty:g}"
        self.name = f"anchor_{mode}|min{self.min_tile}|{base_name}"

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        safe_actions = self._safe_actions(state, sim)
        if not safe_actions:
            return int(self.base_policy(state, sim, rng))
        if hasattr(self.base_policy, "action_values"):
            action_values = self.base_policy.action_values(state, sim)
            if self.penalty is None:
                filtered = [(action, value) for action, value in action_values if action in safe_actions]
                if filtered:
                    return self._select_best(filtered, rng)
            else:
                adjusted = [
                    (action, value if action in safe_actions else value - self.penalty)
                    for action, value in action_values
                ]
                if adjusted:
                    return self._select_best(adjusted, rng)
        action = int(self.base_policy(state, sim, rng))
        if action in safe_actions:
            return action
        return int(safe_actions[int(rng.integers(len(safe_actions)))])

    def _safe_actions(self, state: SimState, sim: ThreesSim) -> set[int]:
        anchor = int(state.board[0, 0])
        if anchor < self.min_tile:
            return set()
        safe: set[int] = set()
        for action in sim.legal_actions(state):
            shifted, eligible = simulate_base_move(state.board, int(action))
            if eligible and int(shifted[0, 0]) >= anchor:
                safe.add(int(action))
        return safe

    @staticmethod
    def _select_best(action_values: list[tuple[int, float]], rng: np.random.Generator) -> int:
        best_value = max(value for _action, value in action_values)
        best_actions = [action for action, value in action_values if value == best_value]
        return int(best_actions[int(rng.integers(len(best_actions)))])


def parse_seed_range(text: str) -> list[int]:
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(part) for part in text.split(",") if part]


def parse_starter_values(text: str) -> list[int | None]:
    values: list[int | None] = []
    for part in text.split(","):
        value = part.strip().lower()
        if not value:
            continue
        values.append(None if value == "none" else int(value))
    if not values:
        raise ValueError("At least one starter value is required")
    return values


def starter_label(starter_tile: int | None) -> str:
    return "none" if starter_tile is None else str(int(starter_tile))


def replay_key(seed: int, starter_tile: int | None) -> tuple[int, str]:
    return (int(seed), starter_label(starter_tile))


def parse_ntuple_expectimax_options(prefix: str, stem: str) -> tuple[int, bool, int | None, str | None]:
    depth_text = prefix.removeprefix(stem)
    budgeted = depth_text.endswith("b")
    if budgeted:
        depth_text = depth_text[:-1]
    adaptive = depth_text.endswith("a")
    if adaptive:
        depth_text = depth_text[:-1]
    depth = int(depth_text) if depth_text else 2
    return depth, adaptive or budgeted, 12 if budgeted else None, "b" if budgeted else None


def parse_phase_gate(text: str) -> int | str | None:
    normalized = text.strip().lower()
    if normalized == "all":
        return None
    risk_aliases = {
        "risk_low": "low_corner_risk",
        "low_risk": "low_corner_risk",
        "low_corner_risk": "low_corner_risk",
        "risk_medium": "medium_corner_risk",
        "medium_risk": "medium_corner_risk",
        "med_risk": "medium_corner_risk",
        "medium_corner_risk": "medium_corner_risk",
        "risk_high": "high_corner_risk",
        "high_risk": "high_corner_risk",
        "high_corner_risk": "high_corner_risk",
    }
    if normalized in risk_aliases:
        return f"risk={risk_aliases[normalized]}"
    aliases = {
        "early": "early_lt384",
        "mid": "mid_384_768",
        "middle": "mid_384_768",
        "late": "late_1536",
        "endgame": "endgame_3072p",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized.isdigit():
        index = int(normalized)
        if 0 <= index < len(PHASE4_NAMES):
            return index
    if normalized in PHASE4_NAMES:
        return PHASE4_NAMES.index(normalized)
    if "/" in normalized:
        phase_text, risk_text = normalized.split("/", 1)
        phase = aliases.get(phase_text, phase_text)
        risk = risk_aliases.get(risk_text, risk_text)
        if phase in PHASE4_NAMES and risk in risk_aliases.values():
            return f"stratum={phase}/{risk}"
    raise ValueError(f"Unsupported phase gate: {text}")


def make_policy(spec: str):
    if spec.startswith("anchor_guard|"):
        return AnchorGuardPolicy(make_policy(spec.split("|", 1)[1]))
    if spec.startswith("anchor_penalty|"):
        _prefix, penalty_text, inner_spec = spec.split("|", 2)
        return AnchorGuardPolicy(make_policy(inner_spec), penalty=float(penalty_text))
    if spec.startswith("geometry_bonus|"):
        _prefix, weight_text, min_tile_text, inner_spec = spec.split("|", 3)
        min_tile = int(min_tile_text)
        if min_tile < 1:
            raise ValueError(f"geometry_bonus min tile must be positive, got {min_tile}")
        policy = make_policy(inner_spec)
        if not isinstance(policy, NtupleExpectimaxPolicy):
            raise ValueError("geometry_bonus can only wrap an ntuple expectimax policy")
        policy.geometry_weight = float(weight_text)
        policy.geometry_min_tile = min_tile
        policy.name = f"geometry_bonus|{policy.geometry_weight:g}|{policy.geometry_min_tile}|{policy.name}"
        policy._afterstate_cache.clear()
        return policy
    if spec.startswith("value_bonus|"):
        _prefix, checkpoint_text, weight_text, gate_text, inner_spec = spec.split("|", 4)
        policy = make_policy(inner_spec)
        if not isinstance(policy, NtupleExpectimaxPolicy):
            raise ValueError("value_bonus can only wrap an ntuple expectimax policy")
        gate = parse_phase_gate(gate_text)
        bonus_gate: int | str = "all" if gate is None else gate
        bonus_path = Path(checkpoint_text)
        bonus_weight = float(weight_text)
        policy.bonus_specs = (*policy.bonus_specs, (bonus_path, bonus_weight, bonus_gate))
        policy.bonus_models = (
            *policy.bonus_models,
            (bonus_path, NtupleValue.load(bonus_path, mmap_mode="r"), bonus_weight, bonus_gate),
        )
        policy.name = f"value_bonus|{bonus_path}|{bonus_weight:g}|{gate_text}|{policy.name}"
        policy._afterstate_cache.clear()
        return policy
    if spec.startswith("action_prior|"):
        from threes_rl.action_prior import parse_action_prior_spec

        return parse_action_prior_spec(spec, make_policy)
    if spec.startswith("selective_rollout|"):
        from threes_rl.selective_rollout import parse_selective_rollout_spec

        return parse_selective_rollout_spec(spec, make_policy)
    if spec == "random":
        return RandomPolicy()
    if spec == "greedy":
        return GreedyPolicy()
    if spec == "expectimax2":
        return ExpectimaxPolicy(depth=2)
    if spec == "expectimax3":
        return ExpectimaxPolicy(depth=3)
    if spec in ("corner2", "expectimax2_corner"):
        return CornerExpectimaxPolicy(depth=2)
    if spec in ("corner3", "expectimax3_corner"):
        return CornerExpectimaxPolicy(depth=3)
    if spec.startswith("ntuple_maxblend_expectimax"):
        parts = spec.split(":")
        prefix = parts[0]
        if len(parts) < 3:
            raise ValueError("ntuple_maxblend_expectimax specs require <base>:<sidecar> after the prefix")
        depth, adaptive, chance_limit, suffix = parse_ntuple_expectimax_options(prefix, "ntuple_maxblend_expectimax")
        return NtupleExpectimaxPolicy(
            Path(parts[1]),
            depth=depth,
            adaptive=adaptive,
            chance_limit=chance_limit,
            suffix=suffix,
            blend_specs=[(Path(part), 0.0) for part in parts[2:]],
            ensemble_mode="max",
        )
    if spec.startswith("ntuple_blend_tiebreak_expectimax"):
        prefix, base_checkpoint, blend_checkpoint, weight_text, margin_text = spec.split(":", 4)
        depth, adaptive, chance_limit, suffix = parse_ntuple_expectimax_options(prefix, "ntuple_blend_tiebreak_expectimax")
        return NtupleExpectimaxPolicy(
            Path(base_checkpoint),
            depth=depth,
            adaptive=adaptive,
            chance_limit=chance_limit,
            suffix=suffix,
            blend_checkpoint=Path(blend_checkpoint),
            blend_weight=float(weight_text),
            tie_margin=float(margin_text),
            tie_breaker="up_left",
        )
    if spec.startswith("ntuple_multiblend_expectimax"):
        parts = spec.split(":")
        prefix = parts[0]
        if len(parts) < 4 or (len(parts) - 2) % 2 != 0:
            raise ValueError(
                "ntuple_multiblend_expectimax specs require "
                "<base>:<sidecar>:<weight> pairs after the prefix"
            )
        depth, adaptive, chance_limit, suffix = parse_ntuple_expectimax_options(prefix, "ntuple_multiblend_expectimax")
        blend_specs = [(Path(parts[idx]), float(parts[idx + 1])) for idx in range(2, len(parts), 2)]
        return NtupleExpectimaxPolicy(
            Path(parts[1]),
            depth=depth,
            adaptive=adaptive,
            chance_limit=chance_limit,
            suffix=suffix,
            blend_specs=blend_specs,
        )
    if spec.startswith("ntuple_additive_phaseblend_expectimax"):
        parts = spec.split(":")
        prefix = parts[0]
        if len(parts) < 5 or (len(parts) - 2) % 3 != 0:
            raise ValueError(
                "ntuple_additive_phaseblend_expectimax specs require "
                "<base>:<sidecar>:<weight>:<phase> triples after the prefix"
            )
        depth, adaptive, chance_limit, suffix = parse_ntuple_expectimax_options(prefix, "ntuple_additive_phaseblend_expectimax")
        blend_specs = []
        phase_blend_specs = []
        for idx in range(2, len(parts), 3):
            path = Path(parts[idx])
            weight = float(parts[idx + 1])
            phase_gate = parse_phase_gate(parts[idx + 2])
            if phase_gate is None:
                blend_specs.append((path, weight))
            else:
                phase_blend_specs.append((path, weight, phase_gate))
        return NtupleExpectimaxPolicy(
            Path(parts[1]),
            depth=depth,
            adaptive=adaptive,
            chance_limit=chance_limit,
            suffix=suffix,
            blend_specs=blend_specs,
            phase_blend_specs=phase_blend_specs,
            ensemble_mode="additive",
        )
    if spec.startswith("ntuple_phaseblend_expectimax"):
        parts = spec.split(":")
        prefix = parts[0]
        if len(parts) < 5 or (len(parts) - 2) % 3 != 0:
            raise ValueError(
                "ntuple_phaseblend_expectimax specs require "
                "<base>:<sidecar>:<weight>:<phase> triples after the prefix"
            )
        depth, adaptive, chance_limit, suffix = parse_ntuple_expectimax_options(prefix, "ntuple_phaseblend_expectimax")
        blend_specs = []
        phase_blend_specs = []
        for idx in range(2, len(parts), 3):
            path = Path(parts[idx])
            weight = float(parts[idx + 1])
            phase_gate = parse_phase_gate(parts[idx + 2])
            if phase_gate is None:
                blend_specs.append((path, weight))
            else:
                phase_blend_specs.append((path, weight, phase_gate))
        return NtupleExpectimaxPolicy(
            Path(parts[1]),
            depth=depth,
            adaptive=adaptive,
            chance_limit=chance_limit,
            suffix=suffix,
            blend_specs=blend_specs,
            phase_blend_specs=phase_blend_specs,
        )
    if spec.startswith("ntuple_blend_expectimax"):
        prefix, base_checkpoint, blend_checkpoint, weight_text = spec.split(":", 3)
        depth, adaptive, chance_limit, suffix = parse_ntuple_expectimax_options(prefix, "ntuple_blend_expectimax")
        return NtupleExpectimaxPolicy(
            Path(base_checkpoint),
            depth=depth,
            adaptive=adaptive,
            chance_limit=chance_limit,
            suffix=suffix,
            blend_checkpoint=Path(blend_checkpoint),
            blend_weight=float(weight_text),
        )
    if spec.startswith("ntuple_expectimax"):
        prefix, checkpoint = spec.split(":", 1)
        depth, adaptive, chance_limit, suffix = parse_ntuple_expectimax_options(prefix, "ntuple_expectimax")
        return NtupleExpectimaxPolicy(
            Path(checkpoint),
            depth=depth,
            adaptive=adaptive,
            chance_limit=chance_limit,
            suffix=suffix,
        )
    if spec.startswith("ntuple:"):
        return NtuplePolicy(Path(spec.split(":", 1)[1]))
    if spec.startswith("ppo:"):
        return PpoPolicy(Path(spec.split(":", 1)[1]))
    raise ValueError(f"Unsupported policy: {spec}")


def starter_baseline_score(starter_tile: int | None) -> int:
    return 0 if starter_tile is None else score_tile(starter_tile)


def max_tile_excluding_initial_starter(board: np.ndarray, starter_tile: int | None) -> int:
    arr = np.asarray(board, dtype=np.int32)
    if starter_tile is None:
        return int(arr.max(initial=0))
    working = arr.copy()
    matches = np.argwhere(working == int(starter_tile))
    if len(matches):
        # The starter tile may move; exclude exactly one free starter tile
        # wherever it currently sits. If it is still top-left, prefer that one.
        match_idx = 0
        for idx, (row, col) in enumerate(matches):
            if int(row) == 0 and int(col) == 0:
                match_idx = idx
                break
        row, col = matches[match_idx]
        working[int(row), int(col)] = 0
    return int(working.max(initial=0))


def run_game(policy, seed: int, starter_tile: int | None, max_moves: int) -> GameResult:
    result, _replay = run_game_with_optional_replay(
        policy,
        policy_name="",
        seed=seed,
        starter_tile=starter_tile,
        max_moves=max_moves,
        capture_replay=False,
    )
    return result


def run_game_with_optional_replay(
    policy,
    *,
    policy_name: str,
    seed: int,
    starter_tile: int | None,
    max_moves: int,
    capture_replay: bool,
    stream_ids: EvalStreamIds | None = None,
) -> tuple[GameResult, dict[str, Any] | None]:
    if stream_ids is None:
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
        policy_rng = np.random.default_rng(seed + 1_000_003)
    else:
        if stream_ids.evaluator_version != "split_exogenous_v1":
            raise ValueError(f"Unsupported evaluator version: {stream_ids.evaluator_version}")
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=stream_ids.deck_stream_id,
            slot_stream_id=stream_ids.slot_stream_id,
            starter_tile=starter_tile,
        )
        policy_rng = np.random.default_rng(int(stream_ids.policy_stream_id))
    state = sim.reset()
    frames: list[dict[str, Any]] | None = None
    if capture_replay:
        frames = [
            {
                "index": 0,
                "state": state_payload(state, sim),
                "move": None,
            }
        ]
    while not state.game_over and state.move_count < max_moves:
        before = state
        action = int(policy(state, sim, policy_rng))
        state, info = sim.step(before, action)
        if not info.moved:
            legal = sim.legal_actions(before)
            if not legal:
                break
            action = int(legal[0])
            state, info = sim.step(before, action)
        if frames is not None:
            frames.append(
                {
                    "index": len(frames),
                    "state": state_payload(state, sim),
                    "move": {
                        "action": DIRECTION_NAMES[action],
                        "preview_used": preview_payload(before),
                        "inserted_value": info.inserted_value,
                        "inserted_pos": list(info.inserted_pos) if info.inserted_pos is not None else None,
                        "eligible_positions": [list(pos) for pos in info.eligible_positions],
                        "merge_score_delta": int(info.merge_score_delta),
                        "score_delta": int(info.score_delta),
                        "terminal_merge": bool(info.terminal_merge),
                        "score_before": int(score_board(before.board)),
                        "score_after": int(score_board(state.board)),
                        "max_tile_before": int(before.max_tile),
                        "max_tile_after": int(state.max_tile),
                    },
                }
            )
    score = score_board(state.board)
    result = GameResult(
        seed=seed,
        score=score,
        score_minus_starter=score - starter_baseline_score(starter_tile),
        moves=state.move_count,
        max_tile=state.max_tile,
        max_tile_excl_starter=max_tile_excluding_initial_starter(state.board, starter_tile),
        terminal_tile=bool(np.any(state.board == 12288)),
        starter_tile=starter_tile,
    )
    replay = None
    if frames is not None:
        replay = {
            "policy": policy_name,
            "seed": int(seed),
            "starter_tile": starter_tile,
            "max_moves": int(max_moves),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rng_streams": (
                {
                    **sim.stream_metadata(),
                    "policy_stream_id": int(stream_ids.policy_stream_id),
                }
                if stream_ids is not None
                else sim.stream_metadata()
            ),
            "final_score": int(score),
            "final_moves": int(state.move_count),
            "final_max_tile": int(state.max_tile),
            "game_over": bool(state.game_over),
            "frames": frames,
        }
    return result, replay


def _init_eval_worker(policy_name: str, max_moves: int, capture_replay: bool) -> None:
    global _WORKER_POLICY, _WORKER_POLICY_NAME, _WORKER_MAX_MOVES, _WORKER_CAPTURE_REPLAY
    _WORKER_POLICY_NAME = policy_name
    _WORKER_MAX_MOVES = int(max_moves)
    _WORKER_CAPTURE_REPLAY = bool(capture_replay)
    _WORKER_POLICY = make_policy(policy_name)


def _run_eval_job_worker(job: EvalJob) -> EvalJobOutput:
    if _WORKER_POLICY is None:
        raise RuntimeError("eval worker was not initialized")
    result, replay = run_game_with_optional_replay(
        _WORKER_POLICY,
        policy_name=_WORKER_POLICY_NAME,
        seed=job.seed,
        starter_tile=job.starter_tile,
        max_moves=_WORKER_MAX_MOVES,
        capture_replay=_WORKER_CAPTURE_REPLAY,
        stream_ids=job.stream_ids,
    )
    return EvalJobOutput(index=job.index, result=result, replay=replay)


def iter_eval_job_outputs(
    *,
    policy,
    policy_name: str,
    eval_jobs: list[EvalJob],
    max_moves: int,
    capture_replay: bool,
    jobs: int = 1,
) -> Iterable[EvalJobOutput]:
    worker_count = max(1, int(jobs))
    if worker_count == 1:
        for job in eval_jobs:
            result, replay = run_game_with_optional_replay(
                policy,
                policy_name=policy_name,
                seed=job.seed,
                starter_tile=job.starter_tile,
                max_moves=max_moves,
                capture_replay=capture_replay,
                stream_ids=job.stream_ids,
            )
            yield EvalJobOutput(index=job.index, result=result, replay=replay)
        return

    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_eval_worker,
            initargs=(policy_name, int(max_moves), bool(capture_replay)),
        )
    except PermissionError as exc:
        print(f"warning: parallel eval unavailable ({exc}); falling back to serial", file=sys.stderr, flush=True)
        yield from iter_eval_job_outputs(
            policy=policy,
            policy_name=policy_name,
            eval_jobs=eval_jobs,
            max_moves=max_moves,
            capture_replay=capture_replay,
            jobs=1,
        )
        return

    with executor:
        future_to_index = {executor.submit(_run_eval_job_worker, job): job.index for job in eval_jobs}
        for future in concurrent.futures.as_completed(future_to_index):
            yield future.result()


def _p(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        raise ValueError("No values to summarize")
    idx = min(len(sorted_values) - 1, int(fraction * (len(sorted_values) - 1)))
    return sorted_values[idx]


def summarize(results: list[GameResult], *, include_by_starter: bool = True) -> dict[str, object]:
    scores = sorted(result.score for result in results)
    if not scores:
        raise ValueError("No results to summarize")
    score_minus = sorted(result.score_minus_starter for result in results)
    moves = sorted(result.moves for result in results)
    thresholds = [192, 384, 768, 1536, 3072, 6144, 12288]
    max_tile_dist = {f">={threshold}": sum(1 for result in results if result.max_tile >= threshold) / len(results) for threshold in thresholds}
    max_tile_excl_starter_dist = {
        f">={threshold}": sum(1 for result in results if result.max_tile_excl_starter >= threshold) / len(results)
        for threshold in thresholds
    }
    payload: dict[str, object] = {
        "games": len(results),
        "high_score": max(scores),
        "mean_score": mean(scores),
        "median_score": median(scores),
        "p90_score": _p(scores, 0.9),
        "high_score_minus_starter": max(score_minus),
        "mean_score_minus_starter": mean(score_minus),
        "median_score_minus_starter": median(score_minus),
        "p90_score_minus_starter": _p(score_minus, 0.9),
        "mean_moves": mean(result.moves for result in results),
        "median_moves": median(moves),
        "p90_moves": _p(moves, 0.9),
        "max_tile_dist": max_tile_dist,
        "max_tile_excl_starter_dist": max_tile_excl_starter_dist,
        "p_max_tile_excl_starter_ge_1536": max_tile_excl_starter_dist[">=1536"],
        "p_max_tile_excl_starter_ge_3072": max_tile_excl_starter_dist[">=3072"],
        "p_max_tile_excl_starter_ge_6144": max_tile_excl_starter_dist[">=6144"],
    }
    if include_by_starter:
        starter_values = sorted({result.starter_tile for result in results}, key=lambda value: -1 if value is None else int(value))
        if len(starter_values) > 1:
            payload["by_starter"] = {
                starter_label(starter): summarize(
                    [result for result in results if result.starter_tile == starter],
                    include_by_starter=False,
                )
                for starter in starter_values
            }
    return payload


def progress_row(results: list[GameResult]) -> dict[str, object]:
    summary = summarize(results)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "games": summary["games"],
        "high_score": summary["high_score"],
        "mean_score": summary["mean_score"],
        "median_score": summary["median_score"],
        "high_score_minus_starter": summary["high_score_minus_starter"],
        "mean_score_minus_starter": summary["mean_score_minus_starter"],
        "median_score_minus_starter": summary["median_score_minus_starter"],
        "mean_moves": summary["mean_moves"],
        "p_max_tile_excl_starter_ge_1536": summary["p_max_tile_excl_starter_ge_1536"],
        "p_max_tile_excl_starter_ge_3072": summary["p_max_tile_excl_starter_ge_3072"],
        "p_max_tile_excl_starter_ge_6144": summary["p_max_tile_excl_starter_ge_6144"],
    }


def write_results_csv(path: Path, results: list[GameResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "starter_tile",
                "score",
                "score_minus_starter",
                "moves",
                "max_tile",
                "max_tile_excl_starter",
                "terminal_tile",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def selected_death_forensics_results(results: list[GameResult], worst_n: int = 5) -> list[tuple[list[str], GameResult]]:
    if not results:
        return []
    ordered = sorted(results, key=lambda result: (result.score_minus_starter, result.score, result.moves))
    selected: dict[tuple[int, str], tuple[list[str], GameResult]] = {}
    for idx, result in enumerate(ordered[: max(0, worst_n)], start=1):
        selected[replay_key(result.seed, result.starter_tile)] = ([f"worst_{idx}"], result)
    median_result = ordered[(len(ordered) - 1) // 2]
    key = replay_key(median_result.seed, median_result.starter_tile)
    if key in selected:
        selected[key][0].append("median")
    else:
        selected[key] = (["median"], median_result)
    return list(selected.values())


def _board_without_free_starter(board: np.ndarray, starter_tile: int | None) -> np.ndarray:
    working = np.asarray(board, dtype=np.int32).copy()
    if starter_tile is None:
        return working
    matches = np.argwhere(working == int(starter_tile))
    if len(matches):
        match_idx = 0
        for idx, (row, col) in enumerate(matches):
            if int(row) == 0 and int(col) == 0:
                match_idx = idx
                break
        row, col = matches[match_idx]
        working[int(row), int(col)] = 0
    return working


def classify_death(final_state: dict[str, Any], last_moves: list[dict[str, Any]], starter_tile: int | None) -> dict[str, object]:
    board = np.asarray(final_state.get("board", []), dtype=np.int32)
    legal_actions = list(final_state.get("legal_actions", []))
    labels: list[str] = []
    notes: list[str] = []

    if final_state.get("game_over") and not legal_actions:
        labels.append("no_legal_moves")
        notes.append("Final state has no legal actions.")
    elif legal_actions:
        labels.append("max_moves_cap")
        notes.append("Game stopped before a terminal board, likely due to max-moves cap.")

    if board.shape == (4, 4):
        board_wo_starter = _board_without_free_starter(board, starter_tile)
        nonstarter_max = int(board_wo_starter.max(initial=0))
        if nonstarter_max >= 384:
            max_positions = {tuple(int(v) for v in pos) for pos in np.argwhere(board_wo_starter == nonstarter_max)}
            if (0, 0) not in max_positions:
                labels.append("corner_trap")
                notes.append(f"Largest built tile {nonstarter_max} is away from the top-left anchor.")
        elif legal_actions == [] and int(board[0, 0]) not in (0, int(board.max(initial=0))):
            labels.append("corner_trap")
            notes.append("Terminal board does not preserve the largest tile in the top-left corner.")

    preview = final_state.get("preview") or {}
    recent_large_inserts = [
        move
        for move in last_moves
        if isinstance(move.get("inserted_value"), int) and int(move.get("inserted_value", 0)) >= 6
    ]
    if preview.get("kind") == "bonus" or preview.get("candidates") or len(recent_large_inserts) >= 3:
        labels.append("bonus_clog")
        if preview.get("kind") == "bonus" or preview.get("candidates"):
            notes.append("Final preview is a bonus/large-candidate tile.")
        if len(recent_large_inserts) >= 3:
            notes.append(f"Last {len(last_moves)} moves include {len(recent_large_inserts)} large-tile insertions.")

    cycle = final_state.get("tile_cycle") or {}
    counts = cycle.get("small_counts") or {}
    if counts:
        remaining = {str(key): int(value) for key, value in counts.items()}
        if sum(remaining.values()) <= 2 or any(value == 0 for value in remaining.values()):
            labels.append("bag_starvation")
            notes.append(f"Small-tile bag is sparse at death: {remaining}.")

    if not labels:
        labels.append("unclassified")
        notes.append("No first-pass heuristic matched; inspect final board and last moves.")

    deduped_labels = list(dict.fromkeys(labels))
    return {"labels": deduped_labels, "notes": notes}


def forensics_case_from_replay(roles: list[str], result: GameResult, replay: dict[str, Any]) -> dict[str, object]:
    frames = list(replay.get("frames", []))
    final_frame = frames[-1] if frames else {}
    final_state = final_frame.get("state", {})
    last_moves: list[dict[str, Any]] = []
    for frame in frames[-20:]:
        move = frame.get("move")
        if not move:
            continue
        last_moves.append(
            {
                "index": int(frame.get("index", 0)),
                "action": move.get("action"),
                "preview_used": move.get("preview_used"),
                "inserted_value": move.get("inserted_value"),
                "inserted_pos": move.get("inserted_pos"),
                "score_delta": move.get("score_delta"),
                "score_after": move.get("score_after"),
                "max_tile_after": move.get("max_tile_after"),
            }
        )
    classification = classify_death(final_state, last_moves, result.starter_tile)
    return {
        "roles": roles,
        "seed": int(result.seed),
        "starter_tile": result.starter_tile,
        "score": int(result.score),
        "score_minus_starter": int(result.score_minus_starter),
        "moves": int(result.moves),
        "max_tile": int(result.max_tile),
        "max_tile_excl_starter": int(result.max_tile_excl_starter),
        "terminal_tile": bool(result.terminal_tile),
        "classification": classification,
        "final_board": final_state.get("board"),
        "final_preview": final_state.get("preview"),
        "legal_actions": final_state.get("legal_actions", []),
        "tile_cycle": final_state.get("tile_cycle"),
        "last_moves": last_moves,
    }


def write_death_forensics_html(path: Path, payload: dict[str, object]) -> None:
    cases = payload.get("cases", [])

    def board_html(board: object) -> str:
        if not isinstance(board, list):
            return ""
        rows = []
        for row in board:
            cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
            rows.append(f"<tr>{cells}</tr>")
        return "<table class=\"board\">" + "".join(rows) + "</table>"

    def moves_html(moves: object) -> str:
        if not isinstance(moves, list) or not moves:
            return "<p class=\"muted\">No recorded moves.</p>"
        rows = []
        for move in moves:
            preview = move.get("preview_used") if isinstance(move, dict) else None
            preview_text = preview.get("label") if isinstance(preview, dict) else ""
            rows.append(
                "<tr>"
                f"<td>{escape(str(move.get('index', '')))}</td>"
                f"<td>{escape(str(move.get('action', '')))}</td>"
                f"<td>{escape(str(preview_text))}</td>"
                f"<td>{escape(str(move.get('inserted_value', '')))}</td>"
                f"<td>{escape(str(move.get('score_delta', '')))}</td>"
                f"<td>{escape(str(move.get('score_after', '')))}</td>"
                "</tr>"
            )
        return (
            "<table><thead><tr><th>#</th><th>Action</th><th>Preview</th><th>Insert</th>"
            "<th>Delta</th><th>Score</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    case_html = []
    iter_cases = cases if isinstance(cases, list) else []
    for case in iter_cases:
        if not isinstance(case, dict):
            continue
        classification = case.get("classification", {})
        labels = ", ".join(classification.get("labels", [])) if isinstance(classification, dict) else ""
        notes = classification.get("notes", []) if isinstance(classification, dict) else []
        note_html = "".join(f"<li>{escape(str(note))}</li>" for note in notes)
        case_html.append(
            "<section class=\"case\">"
            f"<h2>{escape('/'.join(case.get('roles', [])))} seed {escape(str(case.get('seed')))} starter {escape(str(case.get('starter_tile')))}</h2>"
            f"<p class=\"muted\">score-minus-starter {escape(str(case.get('score_minus_starter')))} / "
            f"moves {escape(str(case.get('moves')))} / max excl starter {escape(str(case.get('max_tile_excl_starter')))}</p>"
            f"<p><strong>{escape(labels)}</strong></p>"
            f"<ul>{note_html}</ul>"
            f"{board_html(case.get('final_board'))}"
            f"{moves_html(case.get('last_moves'))}"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Death Forensics</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101113; --panel: #191d21; --line: #364047; --ink: #f1f5f0; --muted: #a9b3ad; --gold: #e9bd4a; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1080px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 36px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    h2 {{ margin: 0 0 6px; font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .case {{ margin-top: 16px; padding: 14px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-variant-numeric: tabular-nums; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .board {{ width: auto; margin-top: 10px; }}
    .board td {{ width: 58px; height: 42px; text-align: center; border: 1px solid var(--line); background: #22282e; font-weight: 700; }}
    strong {{ color: var(--gold); }}
  </style>
</head>
<body>
  <main>
    <h1>Death Forensics</h1>
    <p class="muted">Median and worst games by score-minus-starter. Classifications are first-pass heuristics; final boards and last moves are the source of truth.</p>
    {''.join(case_html)}
  </main>
</body>
</html>
"""
    path.write_text(html)


def write_death_forensics(
    *,
    run_dir: Path,
    results: list[GameResult],
    policy,
    policy_name: str,
    max_moves: int,
    replays_by_seed: dict[object, dict[str, object]] | None = None,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for roles, result in selected_death_forensics_results(results):
        key = replay_key(result.seed, result.starter_tile)
        replay = None if replays_by_seed is None else replays_by_seed.get(key) or replays_by_seed.get(int(result.seed))
        if replay is None:
            replay = record_replay_for_policy(policy, policy_name, int(result.seed), result.starter_tile, max_moves)
        cases.append(forensics_case_from_replay(roles, result, replay))
    payload: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": policy_name,
        "cases": cases,
    }
    json_path = run_dir / "death_forensics.json"
    html_path = run_dir / "death_forensics.html"
    write_json(json_path, payload)
    write_death_forensics_html(html_path, payload)
    return {"json": str(json_path), "html": str(html_path), "cases": len(cases)}


def append_results(policy_name: str, command: str, summary: dict[str, object]) -> None:
    path = Path("threes_rl/RESULTS.md")
    with path.open("a") as fh:
        fh.write(f"\n## Eval: {policy_name}\n\n")
        fh.write(f"Command: `{command}`\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(summary, indent=2, sort_keys=True))
        fh.write("\n```\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        required=True,
        help=(
            "random, greedy, expectimax2/3, corner2/3, ntuple:<checkpoint>, "
            "ntuple_expectimax2:<checkpoint>, ntuple_expectimax2a/b:<checkpoint>, "
            "ntuple_blend_expectimax2:<base>:<sidecar>:<weight>, "
            "ntuple_blend_tiebreak_expectimax2:<base>:<sidecar>:<weight>:<margin>, "
            "ntuple_maxblend_expectimax2:<base>:<sidecar>..., "
            "ntuple_multiblend_expectimax2:<base>:<sidecar>:<weight>..., "
            "ntuple_phaseblend_expectimax2:<base>:<sidecar>:<weight>:<phase>..., "
            "geometry_bonus|<weight>|<min_tile>|<ntuple-expectimax-spec>, or ppo:<checkpoint>"
        ),
    )
    parser.add_argument("--seeds", default="1000:1200")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--no-append", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, help="Directory for summary, progress chart, and retained top-game replays.")
    parser.add_argument("--keep-top-games", type=int, default=0, help="Retain replay artifacts for the top N games by final score.")
    parser.add_argument(
        "--keep-milestone-games",
        type=int,
        default=0,
        help="Retain replay artifacts for every game whose max tile excluding the starter reaches this threshold.",
    )
    parser.add_argument(
        "--keep-milestone-limit",
        type=int,
        default=0,
        help="Optional cap on retained milestone games; 0 keeps all qualifying games.",
    )
    parser.add_argument(
        "--keep-pre-milestone-failures",
        type=int,
        default=0,
        help=(
            "Retain replay artifacts for the highest-scoring games that reached "
            "--keep-pre-milestone-min but stayed below this threshold."
        ),
    )
    parser.add_argument(
        "--keep-pre-milestone-min",
        type=int,
        default=1536,
        help="Minimum max tile excluding starter for --keep-pre-milestone-failures.",
    )
    parser.add_argument(
        "--keep-pre-milestone-limit",
        type=int,
        default=3,
        help="Cap on retained pre-milestone failure games; 0 keeps all qualifying games.",
    )
    parser.add_argument(
        "--checkpoint-results",
        action="store_true",
        help="Periodically write partial_results.csv and progress.csv to the artifact directory during evaluation.",
    )
    parser.add_argument("--charts", action="store_true", help="Write progress.csv and progress.html into the artifact directory.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for seed evaluation. Each worker loads the policy once.",
    )
    args = parser.parse_args()

    starters = parse_starter_values(args.starter)
    seeds = parse_seed_range(args.seeds)
    starter_slug = "_".join(starter_label(starter) for starter in starters)
    artifact_dir = args.artifact_dir
    if artifact_dir is None and (
        args.keep_top_games > 0
        or args.keep_milestone_games > 0
        or args.keep_pre_milestone_failures > 0
        or args.charts
    ):
        artifact_dir = Path("threes_rl/runs/eval_artifacts") / f"{safe_name(args.policy)}_starter_{safe_name(starter_slug)}_{seeds[0]}_{seeds[-1]}"
    if artifact_dir is not None and args.checkpoint_results:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    policy = make_policy(args.policy)
    capture_replays = (
        args.keep_top_games > 0
        or args.keep_milestone_games > 0
        or args.keep_pre_milestone_failures > 0
    )
    captured_replays: dict[object, dict[str, object]] | None = {} if capture_replays else None
    progress_rows: list[dict[str, object]] = []
    eval_jobs = [
        EvalJob(index=idx, seed=seed, starter_tile=starter)
        for idx, (starter, seed) in enumerate((starter, seed) for starter in starters for seed in seeds)
    ]
    completed_results: list[GameResult] = []
    results_by_index: dict[int, GameResult] = {}
    for output in iter_eval_job_outputs(
        policy=policy,
        policy_name=args.policy,
        eval_jobs=eval_jobs,
        max_moves=args.max_moves,
        capture_replay=capture_replays,
        jobs=args.jobs,
    ):
        results_by_index[output.index] = output.result
        completed_results.append(output.result)
        if captured_replays is not None and output.replay is not None:
            captured_replays[replay_key(output.result.seed, output.result.starter_tile)] = output.replay
        idx = len(completed_results)
        if args.progress_every and idx % args.progress_every == 0:
            partial = summarize(completed_results)
            progress_rows.append(progress_row(completed_results))
            if artifact_dir is not None and args.checkpoint_results:
                write_results_csv(artifact_dir / "partial_results.csv", completed_results)
                write_progress_csv(artifact_dir / "progress.csv", progress_rows)
                write_json(artifact_dir / "partial_summary.json", partial)
                if args.keep_milestone_games > 0:
                    write_milestone_replays(
                        run_dir=artifact_dir,
                        results=completed_results,
                        policy=policy,
                        policy_name=args.policy,
                        starter_tile=starters[0],
                        max_moves=args.max_moves,
                        threshold=args.keep_milestone_games,
                        max_games=args.keep_milestone_limit,
                        replays_by_seed=captured_replays,
                    )
                if args.keep_pre_milestone_failures > 0:
                    write_pre_milestone_failure_replays(
                        run_dir=artifact_dir,
                        results=completed_results,
                        policy=policy,
                        policy_name=args.policy,
                        starter_tile=starters[0],
                        max_moves=args.max_moves,
                        min_tile=args.keep_pre_milestone_min,
                        threshold=args.keep_pre_milestone_failures,
                        max_games=args.keep_pre_milestone_limit,
                        replays_by_seed=captured_replays,
                    )
            print(
                f"progress {idx}/{len(eval_jobs)} "
                f"mean_score={partial['mean_score']:.2f} "
                f"mean_score_minus_starter={partial['mean_score_minus_starter']:.2f} "
                f"mean_moves={partial['mean_moves']:.2f}",
                flush=True,
            )
    results = [results_by_index[idx] for idx in range(len(eval_jobs))]
    summary = summarize(results)
    if not progress_rows or progress_rows[-1].get("games") != len(results):
        progress_rows.append(progress_row(results))

    out_dir = Path("threes_rl/runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{safe_name(args.policy)}_starter_{safe_name(starter_slug)}_{seeds[0]}_{seeds[-1]}.csv"
    write_results_csv(csv_path, results)
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if args.checkpoint_results:
            write_results_csv(artifact_dir / "partial_results.csv", results)
        write_progress_csv(artifact_dir / "progress.csv", progress_rows)
        write_progress_chart(
            artifact_dir / "progress.html",
            progress_rows,
            f"{args.policy} starters {starter_slug} seeds {seeds[0]}:{seeds[-1] + 1}",
        )
        if args.keep_top_games > 0:
            top_manifest = write_top_replays(
                run_dir=artifact_dir,
                results=results,
                policy=policy,
                policy_name=args.policy,
                starter_tile=starters[0],
                max_moves=args.max_moves,
                top_n=args.keep_top_games,
                replays_by_seed=captured_replays,
            )
            summary = {**summary, "top_games": top_manifest}
        if args.keep_milestone_games > 0:
            milestone_manifest = write_milestone_replays(
                run_dir=artifact_dir,
                results=results,
                policy=policy,
                policy_name=args.policy,
                starter_tile=starters[0],
                max_moves=args.max_moves,
                threshold=args.keep_milestone_games,
                max_games=args.keep_milestone_limit,
                replays_by_seed=captured_replays,
            )
            summary = {**summary, "milestone_games": milestone_manifest}
        if args.keep_pre_milestone_failures > 0:
            pre_milestone_failure_manifest = write_pre_milestone_failure_replays(
                run_dir=artifact_dir,
                results=results,
                policy=policy,
                policy_name=args.policy,
                starter_tile=starters[0],
                max_moves=args.max_moves,
                min_tile=args.keep_pre_milestone_min,
                threshold=args.keep_pre_milestone_failures,
                max_games=args.keep_pre_milestone_limit,
                replays_by_seed=captured_replays,
            )
            summary = {**summary, "pre_milestone_failure_games": pre_milestone_failure_manifest}
        death_forensics = write_death_forensics(
            run_dir=artifact_dir,
            results=results,
            policy=policy,
            policy_name=args.policy,
            max_moves=args.max_moves,
            replays_by_seed=captured_replays,
        )
        summary = {**summary, "death_forensics": death_forensics}
        write_json(artifact_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"per_seed_csv={csv_path}")
    if artifact_dir is not None:
        print(f"artifact_dir={artifact_dir}")
    if not args.no_append:
        append_results(args.policy, "python -m threes_rl.eval " + " ".join(_quote_args()), summary)


def _quote_args() -> list[str]:
    import sys

    return sys.argv[1:]


if __name__ == "__main__":
    main()

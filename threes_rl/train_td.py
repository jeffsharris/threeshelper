"""Train a TD afterstate n-tuple value function by self-play."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from threes_rl.eval import (
    GameResult,
    max_tile_excluding_initial_starter,
    parse_starter_values,
    progress_row,
    starter_baseline_score,
    summarize,
)
from threes_rl.ntuple import (
    PHASE4_NAMES,
    NtupleValue,
    ResidualStagedNtupleValue,
    SYMMETRIES,
    StagedNtupleValue,
    choose_action,
    expected_afterstate_target,
    patterns_for_set,
    phase4_index_for_board,
)
from threes_rl.record_replay import preview_payload, state_payload, write_html
from threes_rl.replay_provenance import (
    ORIGIN_FRESH,
    ORIGIN_REPLAY_START,
    ORIGIN_UNKNOWN,
    direct_root_fields,
    provenance_fields_from_record,
    replay_provenance,
)
from threes_rl.run_artifacts import write_json, write_progress_chart, write_progress_csv
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, preview_from_label, score_board, simulate_base_move


@dataclass
class TDConfig:
    run_name: str
    games: int = 10_000
    pattern_set: str = "default"
    stage_mode: str = "none"
    lazy_stages: bool = False
    alpha: float = 0.1
    epsilon: float = 0.0
    init: float = 0.0
    init_total: float | None = None
    seed: int = 1
    starter_tile: int | None = 1536
    starter_tiles: list[int | None] | None = None
    max_moves: int = 5000
    continuation_max_moves: int | None = None
    progress_every: int = 100
    checkpoint_every: int = 1000
    keep_top_games: int = 3
    actor_policy: str | None = None
    target_mode: str = "td"
    n_step: int = 8
    use_tc: bool = False
    start_state_replays: list[str] = field(default_factory=list)
    start_state_prob: float = 0.0
    start_state_min_tile: int = 768
    start_state_sample_mode: str = "flat"
    train_phase_filter: list[str] | None = None
    stage_weight_promotion: bool = False
    promotion_copy_tc: bool = True
    exact_start_mix: bool = False
    frozen_incumbent_policy: str | None = None
    actor_generation_jobs: int = 1


@dataclass
class EpisodeRecord:
    result: GameResult
    replay: dict[str, Any]
    mean_abs_td_error: float
    updates_applied: int = 0
    updates_skipped: int = 0
    sampled_start: "StartStateRecord | None" = None
    deferred_nstep_afterstates: list[tuple[np.ndarray, int, bool]] = field(default_factory=list)


@dataclass
class FixedActorEpisodeJob:
    game_index: int
    sampled_start: "StartStateRecord | None"
    policy_rng_state: dict[str, Any]


_FIXED_ACTOR_POLICY = None
_FIXED_ACTOR_CONFIG: TDConfig | None = None


START_STATE_PHASE_NAMES = PHASE4_NAMES


@dataclass
class StartStateRecord:
    state: SimState
    starter_tile: int | None
    source_replay: str | None = None
    source_seed: int | None = None
    source_frame_index: int | None = None
    source_policy: str | None = None
    source_origin: str = ORIGIN_UNKNOWN
    root_origin: str = ORIGIN_UNKNOWN
    root_replay: str | None = None
    root_seed: int | None = None
    root_frame_index: int | None = None
    root_move_count: int | None = None
    root_score: int | None = None
    root_policy: str | None = None
    root_policy_family: str | None = None
    ancestry_key: str | None = None
    behavior_family: str | None = None
    trajectory_outcome: str | None = None
    record_id: str | None = None


def start_state_phase_index(state: SimState, starter_tile: int | None) -> int:
    built_max = max_tile_excluding_initial_starter(state.board, starter_tile)
    if built_max < 384:
        return 0
    if built_max < 1536:
        return 1
    if built_max < 3072:
        return 2
    return 3


@dataclass
class StartStateReservoir:
    states: list[SimState | StartStateRecord]
    starter_tile: int | None
    sample_mode: str = "flat"

    def __post_init__(self) -> None:
        if self.sample_mode not in ("flat", "phase_balanced", "ancestry_balanced"):
            raise ValueError(f"Unsupported start state sample mode: {self.sample_mode}")
        self.records: list[StartStateRecord] = [
            item
            if isinstance(item, StartStateRecord)
            else StartStateRecord(state=item, starter_tile=self.starter_tile)
            for item in self.states
        ]
        self.phase_buckets: dict[int, list[StartStateRecord]] = {idx: [] for idx in range(len(START_STATE_PHASE_NAMES))}
        for record in self.records:
            self.phase_buckets[start_state_phase_index(record.state, record.starter_tile)].append(record)
        self.phase_ancestry_buckets: dict[int, dict[str, list[StartStateRecord]]] = {
            idx: defaultdict(list) for idx in range(len(START_STATE_PHASE_NAMES))
        }
        for record in self.records:
            stage_idx = start_state_phase_index(record.state, record.starter_tile)
            ancestry = record.ancestry_key or f"source:{record.source_replay}:{record.source_seed}"
            self.phase_ancestry_buckets[stage_idx][ancestry].append(record)
        self.sample_count = 0
        self.stage_visit_counts: Counter[int] = Counter()
        self.ancestry_visit_counts: dict[int, Counter[str]] = defaultdict(Counter)
        self.outcome_visit_counts: Counter[str] = Counter()
        self.family_visit_counts: Counter[str] = Counter()

    def sample(self, rng: np.random.Generator) -> SimState | None:
        record = self.sample_record(rng)
        return None if record is None else record.state

    def sample_record(self, rng: np.random.Generator) -> StartStateRecord | None:
        if not self.records:
            return None
        if self.sample_mode == "flat":
            record = self.records[int(rng.integers(len(self.records)))]
        else:
            non_empty = [idx for idx, records in self.phase_buckets.items() if records]
            phase_idx = non_empty[int(rng.integers(len(non_empty)))]
            if self.sample_mode == "phase_balanced":
                records = self.phase_buckets[phase_idx]
                record = records[int(rng.integers(len(records)))]
            else:
                ancestry_buckets = self.phase_ancestry_buckets[phase_idx]
                ancestries = sorted(ancestry_buckets)
                ancestry = ancestries[int(rng.integers(len(ancestries)))]
                records = ancestry_buckets[ancestry]
                record = records[int(rng.integers(len(records)))]
        stage_idx = start_state_phase_index(record.state, record.starter_tile)
        ancestry = record.ancestry_key or f"source:{record.source_replay}:{record.source_seed}"
        self.sample_count += 1
        self.stage_visit_counts[stage_idx] += 1
        self.ancestry_visit_counts[stage_idx][ancestry] += 1
        self.outcome_visit_counts[record.trajectory_outcome or "unknown"] += 1
        self.family_visit_counts[record.behavior_family or record.root_policy_family or "unknown"] += 1
        return record

    def summary(self) -> dict[str, object]:
        return {
            "sample_mode": self.sample_mode,
            "states": len(self.records),
            "phase_buckets": {
                START_STATE_PHASE_NAMES[idx]: len(records)
                for idx, records in self.phase_buckets.items()
                if records
            },
            "unique_ancestries_by_stage": {
                START_STATE_PHASE_NAMES[idx]: len(ancestries)
                for idx, ancestries in self.phase_ancestry_buckets.items()
                if ancestries
            },
        }

    def sampling_summary(self) -> dict[str, object]:
        stages: dict[str, object] = {}
        scarcity_flags: list[str] = []
        for idx, eligible_ancestries in self.phase_ancestry_buckets.items():
            if not eligible_ancestries:
                continue
            visits = self.ancestry_visit_counts[idx]
            total = sum(visits.values())
            probabilities = [count / total for count in visits.values()] if total else []
            entropy = -sum(prob * math.log(prob) for prob in probabilities if prob > 0.0)
            unique_eligible = len(eligible_ancestries)
            if unique_eligible < 20:
                scarcity_flags.append(START_STATE_PHASE_NAMES[idx])
            stages[START_STATE_PHASE_NAMES[idx]] = {
                "restart_episodes": int(self.stage_visit_counts[idx]),
                "eligible_ancestries": unique_eligible,
                "unique_ancestries_visited": len(visits),
                "max_ancestry_share": max(probabilities, default=0.0),
                "ancestry_entropy": float(entropy),
                "effective_ancestry_count": float(math.exp(entropy)) if probabilities else 0.0,
            }
        return {
            "restart_episodes": int(self.sample_count),
            "stages": stages,
            "success_failure_provenance_counts": dict(sorted(self.outcome_visit_counts.items())),
            "policy_family_counts": dict(sorted(self.family_visit_counts.items())),
            "scarcity_flag_lt20_ancestries": scarcity_flags,
        }

    def restore_sampling_metrics(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            if str(row.get("start_type", "")) != "restart":
                continue
            stage_name = str(row.get("restart_stage", ""))
            if stage_name not in START_STATE_PHASE_NAMES:
                continue
            stage_idx = START_STATE_PHASE_NAMES.index(stage_name)
            ancestry = str(row.get("restart_ancestry") or "unknown")
            self.sample_count += 1
            self.stage_visit_counts[stage_idx] += 1
            self.ancestry_visit_counts[stage_idx][ancestry] += 1
            self.outcome_visit_counts[str(row.get("restart_outcome") or "unknown")] += 1
            self.family_visit_counts[str(row.get("restart_behavior_family") or "unknown")] += 1


def starter_options(config: TDConfig) -> list[int | None]:
    return list(config.starter_tiles) if config.starter_tiles else [config.starter_tile]


def starter_for_game(config: TDConfig, game_index: int) -> int | None:
    options = starter_options(config)
    return options[(int(game_index) - 1) % len(options)]


def parse_phase_filter(text: str | None) -> list[str] | None:
    if text is None or not text.strip():
        return None
    aliases = {
        "early": "early_lt384",
        "mid": "mid_384_768",
        "middle": "mid_384_768",
        "late": "late_1536",
        "endgame": "endgame_3072p",
    }
    phases: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        normalized = aliases.get(part.strip().lower(), part.strip().lower())
        if not normalized:
            continue
        if normalized not in PHASE4_NAMES:
            raise ValueError(f"Unsupported train phase filter: {part}")
        if normalized not in seen:
            phases.append(normalized)
            seen.add(normalized)
    return phases or None


def should_update_afterstate(config: TDConfig, board: np.ndarray, starter_tile: int | None) -> bool:
    if not config.train_phase_filter:
        return True
    phase_idx = phase4_index_for_board(board, starter_tile=starter_tile)
    return PHASE4_NAMES[phase_idx] in set(config.train_phase_filter)


def update_value(value_model: NtupleValue, board: np.ndarray, target: float, alpha: float, use_tc: bool) -> float:
    if use_tc:
        return value_model.update_tc(board, target, alpha)
    return value_model.update(board, target, alpha)


def copy_state(state: SimState) -> SimState:
    return SimState(
        board=state.board.copy(),
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


def state_from_replay_payload(payload: dict[str, Any]) -> SimState:
    preview_payload = payload["preview"]
    if preview_payload["kind"] == "bonus":
        preview = preview_from_label("large_candidates", preview_payload.get("candidates", ()))
    else:
        preview = preview_from_label(str(preview_payload["kind"]))
    cycle = payload["tile_cycle"]
    return SimState(
        board=np.asarray(payload["board"], dtype=np.int32),
        preview=preview,
        small_counts={str(k): int(v) for k, v in cycle["small_counts"].items()},
        small_pos=int(cycle["small_pos"]),
        small_seen_total=int(cycle["small_seen_total"]),
        span_small_pos=int(cycle["span_small_pos"]),
        large_pending=bool(cycle["large_pending"]),
        max_tile=int(cycle["max_tile"]),
        move_count=int(payload["move_count"]),
        game_over=bool(payload["game_over"]),
    )


def _starter_from_record(record: dict[str, Any], fallback: int | None) -> int | None:
    value = record.get("starter_tile", fallback)
    if value is None:
        return None
    return int(value)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_from_replay_frame(
    replay: dict[str, Any],
    replay_path: Path,
    *,
    frame_index: int,
) -> dict[str, Any]:
    provenance = replay_provenance(replay, replay_path)
    return {
        "source_origin": provenance["replay_origin"],
        "source_replay": str(replay_path),
        "source_seed": _int_or_none(replay.get("seed")),
        "source_frame_index": int(frame_index),
        "source_policy": replay.get("policy"),
        "root_origin": provenance["root_origin"],
        "root_replay": provenance["root_replay"],
        "root_seed": provenance["root_seed"],
        "root_frame_index": provenance["root_frame_index"],
        "root_move_count": provenance["root_move_count"],
        "root_score": provenance["root_score"],
        "root_policy": provenance["root_policy"],
        "root_policy_family": provenance["root_policy_family"],
        "ancestry_key": provenance["ancestry_key"],
    }


def _iter_start_state_payloads(
    payload: object,
    fallback_starter: int | None,
    source_path: Path,
) -> list[tuple[dict[str, Any], int | None, dict[str, Any]]]:
    out: list[tuple[dict[str, Any], int | None, dict[str, Any]]] = []
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        for record in payload["records"]:
            if not isinstance(record, dict):
                continue
            state_payload_dict = record.get("state")
            if isinstance(state_payload_dict, dict):
                metadata = provenance_fields_from_record(record, source_path)
                metadata.update(
                    {
                        "behavior_family": record.get("behavior_family"),
                        "trajectory_outcome": record.get("trajectory_outcome"),
                        "record_id": record.get("record_id"),
                    }
                )
                out.append(
                    (
                        state_payload_dict,
                        _starter_from_record(record, fallback_starter),
                        metadata,
                    )
                )
        return out
    if isinstance(payload, list):
        for record in payload:
            if not isinstance(record, dict):
                continue
            state_payload_dict = record.get("state")
            if isinstance(state_payload_dict, dict):
                metadata = provenance_fields_from_record(record, source_path)
                metadata.update(
                    {
                        "behavior_family": record.get("behavior_family"),
                        "trajectory_outcome": record.get("trajectory_outcome"),
                        "record_id": record.get("record_id"),
                    }
                )
                out.append(
                    (
                        state_payload_dict,
                        _starter_from_record(record, fallback_starter),
                        metadata,
                    )
                )
        return out
    if isinstance(payload, dict):
        replay_starter = _starter_from_record(payload, fallback_starter)
        frames = payload.get("frames", [])
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                state_payload_dict = frame.get("state")
                if isinstance(state_payload_dict, dict):
                    frame_index = int(frame.get("index", len(out)))
                    out.append(
                        (
                            state_payload_dict,
                            replay_starter,
                            _metadata_from_replay_frame(payload, source_path, frame_index=frame_index),
                        )
                    )
    return out


def load_start_state_records(paths: list[str], min_tile: int, starter_tile: int | None) -> list[StartStateRecord]:
    records: list[StartStateRecord] = []
    for text_path in paths:
        source_path = Path(text_path)
        payload = json.loads(source_path.read_text())
        for state_payload_dict, payload_starter, metadata in _iter_start_state_payloads(payload, starter_tile, source_path):
            if state_payload_dict.get("game_over"):
                continue
            state = state_from_replay_payload(state_payload_dict)
            if max_tile_excluding_initial_starter(state.board, payload_starter) >= int(min_tile):
                records.append(
                    StartStateRecord(
                        state=state,
                        starter_tile=payload_starter,
                        source_replay=metadata.get("source_replay"),
                        source_seed=_int_or_none(metadata.get("source_seed")),
                        source_frame_index=_int_or_none(metadata.get("source_frame_index")),
                        source_policy=metadata.get("source_policy"),
                        source_origin=str(metadata.get("source_origin", ORIGIN_UNKNOWN)),
                        root_origin=str(metadata.get("root_origin", ORIGIN_UNKNOWN)),
                        root_replay=metadata.get("root_replay"),
                        root_seed=_int_or_none(metadata.get("root_seed")),
                        root_frame_index=_int_or_none(metadata.get("root_frame_index")),
                        root_move_count=_int_or_none(metadata.get("root_move_count")),
                        root_score=_int_or_none(metadata.get("root_score")),
                        root_policy=metadata.get("root_policy"),
                        root_policy_family=metadata.get("root_policy_family"),
                        ancestry_key=metadata.get("ancestry_key"),
                        behavior_family=metadata.get("behavior_family"),
                        trajectory_outcome=metadata.get("trajectory_outcome"),
                        record_id=metadata.get("record_id"),
                    )
                )
    return records


def load_start_states(paths: list[str], min_tile: int, starter_tile: int | None) -> list[SimState]:
    return [record.state for record in load_start_state_records(paths, min_tile, starter_tile)]


def replay_header(config: TDConfig, game_seed: int, starter_tile: int | None) -> dict[str, Any]:
    return {
        "policy": f"train_td:{config.run_name}",
        "seed": int(game_seed),
        "starter_tile": starter_tile,
        "max_moves": int(config.max_moves),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training_config": asdict(config),
    }


def replay_start_provenance_fields(sample: StartStateRecord) -> dict[str, Any]:
    return {
        "replay_origin": ORIGIN_REPLAY_START,
        "source_origin": sample.source_origin,
        "source_replay": sample.source_replay,
        "source_seed": sample.source_seed,
        "source_frame_index": sample.source_frame_index,
        "source_move_count": int(sample.state.move_count),
        "source_score": int(score_board(sample.state.board)),
        "source_policy": sample.source_policy,
        "root_origin": sample.root_origin,
        "root_replay": sample.root_replay,
        "root_seed": sample.root_seed,
        "root_frame_index": sample.root_frame_index,
        "root_move_count": sample.root_move_count,
        "root_score": sample.root_score,
        "root_policy": sample.root_policy,
        "root_policy_family": sample.root_policy_family,
        "ancestry_key": sample.ancestry_key,
    }


def play_episode(
    value_model: NtupleValue | None,
    config: TDConfig,
    game_index: int,
    actor_policy=None,
    start_state_reservoir: StartStateReservoir | None = None,
    sampled_start_override: StartStateRecord | None = None,
    start_choice_override: bool | None = None,
    policy_rng_state_override: dict[str, Any] | None = None,
    defer_nstep_updates: bool = False,
) -> EpisodeRecord:
    game_seed = int(config.seed + 1_000_003 * game_index)
    episode_starter = starter_for_game(config, game_index)
    sim = ThreesSim(np.random.default_rng(game_seed), starter_tile=episode_starter)
    policy_rng = np.random.default_rng(game_seed + 37)
    if policy_rng_state_override is not None:
        policy_rng.bit_generator.state = policy_rng_state_override
    if start_choice_override is not None:
        choose_restart = bool(start_choice_override)
    elif start_state_reservoir is None or float(config.start_state_prob) <= 0.0:
        choose_restart = False
    elif config.exact_start_mix:
        if abs(float(config.start_state_prob) - 0.5) > 1e-12:
            raise ValueError("exact_start_mix currently requires start_state_prob=0.5")
        choose_restart = game_index % 2 == 0
    else:
        choose_restart = float(policy_rng.random()) < float(config.start_state_prob)
    if start_choice_override is not None:
        sampled_record = sampled_start_override if choose_restart else None
        state = copy_state(sampled_record.state) if sampled_record is not None else sim.reset()
        used_replay_start = sampled_record is not None
    elif (
        start_state_reservoir is not None
        and float(config.start_state_prob) > 0.0
        and choose_restart
    ):
        sampled_record = start_state_reservoir.sample_record(policy_rng)
        state = copy_state(sampled_record.state) if sampled_record is not None else sim.reset()
        used_replay_start = sampled_record is not None
    else:
        sampled_record = None
        state = sim.reset()
        used_replay_start = False
    move_limit = int(config.max_moves)
    if used_replay_start and config.continuation_max_moves is not None:
        move_limit = int(state.move_count) + max(0, int(config.continuation_max_moves))
    frames: list[dict[str, Any]] = [{"index": 0, "state": state_payload(state, sim), "move": None}]

    td_errors: list[float] = []
    mc_afterstates: list[tuple[np.ndarray, int, bool]] = []
    nstep_afterstates: list[tuple[np.ndarray, int, bool]] = []
    updates_applied = 0
    updates_skipped = 0

    while not state.game_over and state.move_count < move_limit:
        before = state
        if actor_policy is None:
            if value_model is None:
                raise ValueError("A value model is required when no actor policy is supplied")
            action, afterstate = choose_action(value_model, before, sim, policy_rng, epsilon=config.epsilon)
        else:
            action = int(actor_policy(before, sim, policy_rng))
            afterstate, _eligible = simulate_base_move(before.board, action)
        target: float | None = None
        if config.target_mode == "td":
            if value_model is None:
                raise ValueError("A value model is required for one-step TD targets")
            target, _target_afterstate = expected_afterstate_target(value_model, before, sim, action)
        state, info = sim.step(before, action)
        if not info.moved:
            legal = sim.legal_actions(before)
            if not legal:
                break
            action = int(legal[int(policy_rng.integers(len(legal)))])

            afterstate, _eligible = simulate_base_move(before.board, action)
            if config.target_mode == "td":
                if value_model is None:
                    raise ValueError("A value model is required for one-step TD targets")
                target, _target_afterstate = expected_afterstate_target(value_model, before, sim, action)
            state, info = sim.step(before, action)
            if not info.moved:
                break

        if config.target_mode == "td":
            if target is None:
                raise RuntimeError("TD target was not computed")
            if should_update_afterstate(config, afterstate, episode_starter):
                td_errors.append(abs(update_value(value_model, afterstate, float(target), config.alpha, config.use_tc)))
                updates_applied += 1
            else:
                updates_skipped += 1
        elif config.target_mode == "mc":
            eligible_update = should_update_afterstate(config, afterstate, episode_starter)
            mc_afterstates.append((afterstate.copy(), score_board(afterstate), eligible_update))
        elif config.target_mode == "nstep":
            eligible_update = should_update_afterstate(config, afterstate, episode_starter)
            nstep_afterstates.append((afterstate.copy(), score_board(afterstate), eligible_update))
        else:
            raise ValueError(f"Unsupported target mode: {config.target_mode}")

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

        if state.game_over:
            break

    score = score_board(state.board)
    if config.target_mode == "mc":
        if value_model is None:
            raise ValueError("A value model is required for MC updates")
        for afterstate, afterstate_score, eligible_update in mc_afterstates:
            if not eligible_update:
                updates_skipped += 1
                continue
            target = float(score - afterstate_score)
            td_errors.append(abs(update_value(value_model, afterstate, target, config.alpha, config.use_tc)))
            updates_applied += 1
    elif config.target_mode == "nstep":
        if not defer_nstep_updates:
            if value_model is None:
                raise ValueError("A value model is required for n-step updates")
            td_errors, updates_applied, updates_skipped = apply_nstep_updates(
                value_model,
                nstep_afterstates,
                score,
                config,
            )

    result = GameResult(
        seed=game_seed,
        score=score,
        score_minus_starter=score - starter_baseline_score(episode_starter),
        moves=state.move_count,
        max_tile=state.max_tile,
        max_tile_excl_starter=max_tile_excluding_initial_starter(state.board, episode_starter),
        terminal_tile=bool(np.any(state.board == 12288)),
        starter_tile=episode_starter,
    )
    header = replay_header(config, game_seed, episode_starter)
    if sampled_record is not None:
        header.update(replay_start_provenance_fields(sampled_record))
    else:
        header.update(
            direct_root_fields(
                origin=ORIGIN_FRESH,
                seed=int(game_seed),
                policy=f"train_td:{config.run_name}",
                first_score=int(frames[0]["state"]["score"]),
            )
        )
    replay = {
        **header,
        "final_score": int(score),
        "final_moves": int(state.move_count),
        "final_max_tile": int(state.max_tile),
        "final_max_tile_excl_starter": int(result.max_tile_excl_starter),
        "game_over": bool(state.game_over),
        "frames": frames,
    }
    return EpisodeRecord(
        result=result,
        replay=replay,
        mean_abs_td_error=float(mean(td_errors)) if td_errors else 0.0,
        updates_applied=updates_applied,
        updates_skipped=updates_skipped,
        sampled_start=sampled_record,
        deferred_nstep_afterstates=nstep_afterstates if defer_nstep_updates else [],
    )


def apply_nstep_updates(
    value_model: NtupleValue,
    nstep_afterstates: list[tuple[np.ndarray, int, bool]],
    final_score: int,
    config: TDConfig,
) -> tuple[list[float], int, int]:
    n = max(1, int(config.n_step))
    scores = [afterstate_score for _afterstate, afterstate_score, _eligible_update in nstep_afterstates]
    bootstrap_values = [
        float(value_model.value(afterstate))
        for afterstate, _afterstate_score, _eligible_update in nstep_afterstates
    ]
    errors: list[float] = []
    updates_applied = 0
    updates_skipped = 0
    for idx, (afterstate, afterstate_score, eligible_update) in enumerate(nstep_afterstates):
        if not eligible_update:
            updates_skipped += 1
            continue
        bootstrap_idx = idx + n
        if bootstrap_idx < len(nstep_afterstates):
            _bootstrap_board, bootstrap_score, _eligible_update = nstep_afterstates[bootstrap_idx]
            target = float(bootstrap_score - afterstate_score + bootstrap_values[bootstrap_idx])
        else:
            target = float(final_score - scores[idx])
        errors.append(abs(update_value(value_model, afterstate, target, config.alpha, config.use_tc)))
        updates_applied += 1
    return errors, updates_applied, updates_skipped


def _init_fixed_actor_worker(config: TDConfig) -> None:
    global _FIXED_ACTOR_POLICY, _FIXED_ACTOR_CONFIG
    from threes_rl.eval import make_policy

    if config.actor_policy is None:
        raise ValueError("Fixed-actor worker requires actor_policy")
    _FIXED_ACTOR_CONFIG = config
    _FIXED_ACTOR_POLICY = make_policy(config.actor_policy)


def _run_fixed_actor_episode_worker(job: FixedActorEpisodeJob) -> tuple[int, EpisodeRecord]:
    if _FIXED_ACTOR_POLICY is None or _FIXED_ACTOR_CONFIG is None:
        raise RuntimeError("Fixed-actor training worker was not initialized")
    episode = play_episode(
        None,
        _FIXED_ACTOR_CONFIG,
        job.game_index,
        actor_policy=_FIXED_ACTOR_POLICY,
        sampled_start_override=job.sampled_start,
        start_choice_override=job.sampled_start is not None,
        policy_rng_state_override=job.policy_rng_state,
        defer_nstep_updates=True,
    )
    return job.game_index, episode


def prepare_fixed_actor_jobs(
    config: TDConfig,
    game_indices: range,
    start_state_reservoir: StartStateReservoir | None,
) -> list[FixedActorEpisodeJob]:
    jobs: list[FixedActorEpisodeJob] = []
    for game_index in game_indices:
        game_seed = int(config.seed + 1_000_003 * game_index)
        policy_rng = np.random.default_rng(game_seed + 37)
        if start_state_reservoir is None or float(config.start_state_prob) <= 0.0:
            choose_restart = False
        elif config.exact_start_mix:
            choose_restart = game_index % 2 == 0
        else:
            choose_restart = float(policy_rng.random()) < float(config.start_state_prob)
        sampled_start = start_state_reservoir.sample_record(policy_rng) if choose_restart else None
        jobs.append(
            FixedActorEpisodeJob(
                game_index=game_index,
                sampled_start=sampled_start,
                policy_rng_state=policy_rng.bit_generator.state,
            )
        )
    return jobs


def iter_fixed_actor_episodes(
    config: TDConfig,
    jobs: list[FixedActorEpisodeJob],
) -> Any:
    worker_count = max(1, int(config.actor_generation_jobs))
    if worker_count == 1:
        if config.actor_policy is None:
            raise ValueError("Fixed actor generation requires actor_policy")
        from threes_rl.eval import make_policy

        actor_policy = make_policy(config.actor_policy)
        for job in jobs:
            yield job.game_index, play_episode(
                None,
                config,
                job.game_index,
                actor_policy=actor_policy,
                sampled_start_override=job.sampled_start,
                start_choice_override=job.sampled_start is not None,
                policy_rng_state_override=job.policy_rng_state,
                defer_nstep_updates=True,
            )
        return
    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_fixed_actor_worker,
            initargs=(config,),
        )
    except PermissionError as exc:
        print(f"warning: parallel fixed-actor generation unavailable ({exc}); falling back to serial", file=sys.stderr, flush=True)
        fallback = TDConfig(**{**asdict(config), "actor_generation_jobs": 1})
        yield from iter_fixed_actor_episodes(fallback, jobs)
        return
    with executor:
        for result in executor.map(_run_fixed_actor_episode_worker, jobs, chunksize=1):
            yield result


def write_top_training_replays(run_dir: Path, top_records: list[EpisodeRecord]) -> list[dict[str, object]]:
    top_dir = run_dir / "top_games"
    if top_dir.exists():
        shutil.rmtree(top_dir)
    top_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for rank, record in enumerate(top_records, start=1):
        result = record.result
        game_dir = top_dir / f"rank_{rank:02d}_score_{result.score}_seed_{result.seed}"
        game_dir.mkdir(parents=True, exist_ok=True)
        json_path = game_dir / "replay.json"
        html_path = game_dir / "replay.html"
        write_json(json_path, record.replay)
        write_html(html_path, record.replay)
        manifest.append(
            {
                "rank": rank,
                "seed": int(result.seed),
                "starter_tile": result.starter_tile,
                "score": int(result.score),
                "score_minus_starter": int(result.score_minus_starter),
                "moves": int(result.moves),
                "max_tile": int(result.max_tile),
                "max_tile_excl_starter": int(result.max_tile_excl_starter),
                "html": str(html_path),
                "json": str(json_path),
            }
        )
    write_json(top_dir / "manifest.json", manifest)
    return manifest


def maybe_track_top(top_records: list[EpisodeRecord], record: EpisodeRecord, keep: int) -> list[EpisodeRecord]:
    if keep <= 0:
        return []
    top_records.append(record)
    top_records.sort(key=lambda item: (item.result.score, item.result.moves), reverse=True)
    return top_records[:keep]


def save_checkpoint(value_model: NtupleValue, run_dir: Path, name: str, config: TDConfig, games_completed: int) -> None:
    value_model.save(
        run_dir / name,
        extra_meta={
            "training_config": asdict(config),
            "games_completed": int(games_completed),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def _metric_value(row: dict[str, str], key: str, default: str = "0") -> str:
    value = row.get(key)
    return default if value in (None, "") else str(value)


def load_existing_metrics(run_dir: Path) -> tuple[list[GameResult], list[dict[str, object]]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return [], []
    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    results: list[GameResult] = []
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        starter_text = raw.get("starter_tile")
        starter_tile = None if starter_text in (None, "", "None") else int(starter_text)
        results.append(
            GameResult(
                seed=int(_metric_value(raw, "seed")),
                score=int(float(_metric_value(raw, "score"))),
                score_minus_starter=int(float(_metric_value(raw, "score_minus_starter"))),
                moves=int(float(_metric_value(raw, "moves"))),
                max_tile=int(float(_metric_value(raw, "max_tile"))),
                max_tile_excl_starter=int(float(_metric_value(raw, "max_tile_excl_starter"))),
                terminal_tile=_metric_value(raw, "terminal_tile", "False").lower() == "true",
                starter_tile=starter_tile,
            )
        )
        row: dict[str, object] = dict(raw)
        for key in (
            "game",
            "seed",
            "score",
            "score_minus_starter",
            "moves",
            "max_tile",
            "max_tile_excl_starter",
            "updates_applied",
            "updates_skipped",
        ):
            if raw.get(key) not in (None, ""):
                row[key] = int(float(str(raw[key])))
        if raw.get("mean_abs_td_error") not in (None, ""):
            row["mean_abs_td_error"] = float(str(raw["mean_abs_td_error"]))
        row["starter_tile"] = starter_tile
        row["terminal_tile"] = _metric_value(raw, "terminal_tile", "False").lower() == "true"
        rows.append(row)
    return results, rows


def load_existing_progress(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "progress.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_existing_top_records(run_dir: Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for replay_path in sorted((run_dir / "top_games").glob("*/replay.json")):
        try:
            replay = json.loads(replay_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        starter_value = replay.get("starter_tile", 1536)
        starter_tile = None if starter_value is None else int(starter_value)
        score = int(replay.get("final_score", 0))
        final_board = np.asarray(replay.get("frames", [{}])[-1].get("state", {}).get("board", np.zeros((4, 4))), dtype=np.int32)
        result = GameResult(
            seed=int(replay.get("seed", 0)),
            score=score,
            score_minus_starter=score - starter_baseline_score(starter_tile),
            moves=int(replay.get("final_moves", 0)),
            max_tile=int(replay.get("final_max_tile", 0)),
            max_tile_excl_starter=max_tile_excluding_initial_starter(final_board, starter_tile),
            terminal_tile=bool(np.any(final_board == 12288)),
            starter_tile=starter_tile,
        )
        records.append(EpisodeRecord(result=result, replay=replay, mean_abs_td_error=0.0))
    return records


def allocated_stage_summary(value_model: NtupleValue) -> dict[str, object] | None:
    staged_model = (
        value_model.residual
        if isinstance(value_model, ResidualStagedNtupleValue)
        else value_model
    )
    if not isinstance(staged_model, StagedNtupleValue):
        return None
    allocated = [
        {
            "index": idx,
            "name": value_model.stage_names[idx],
        }
        for idx, stage in enumerate(staged_model.stages)
        if stage is not None
    ]
    summary = {
        "stage_mode": staged_model.stage_mode,
        "allocated_count": len(allocated),
        "total_stages": len(staged_model.stages),
        "allocated": allocated,
    }
    summary["stage_metrics"] = value_model.stage_metrics()
    return summary


def create_value_model(config: TDConfig) -> NtupleValue:
    if config.frozen_incumbent_policy is not None:
        from threes_rl.eval import make_policy
        from threes_rl.expectimax import NtupleExpectimaxPolicy

        if config.stage_mode != "phase4" or not config.stage_weight_promotion:
            raise ValueError("Frozen-incumbent residual training requires phase4 stage promotion")
        if config.init != 0.0 or config.init_total is not None:
            raise ValueError("Frozen-incumbent residual entries must initialize to exactly zero")
        if config.actor_policy != config.frozen_incumbent_policy:
            raise ValueError("Residual training actor must be the exact frozen incumbent policy")
        policy = make_policy(config.frozen_incumbent_policy)
        if not isinstance(policy, NtupleExpectimaxPolicy) or policy.depth != 2:
            raise ValueError("Frozen incumbent must be a depth-2 n-tuple expectimax policy")
        if policy.ensemble_mode != "blend" or policy.bonus_specs or policy.geometry_weight != 0.0:
            raise ValueError("Residual composite supports the frozen incumbent blend without bonuses or geometry")
        return ResidualStagedNtupleValue.from_frozen_blend(
            frozen_policy_spec=config.frozen_incumbent_policy,
            base_checkpoint=policy.checkpoint,
            blend_specs=list(policy.blend_specs),
            phase_blend_specs=list(policy.phase_blend_specs),
            pattern_set=config.pattern_set,
            starter_tile=starter_options(config)[0],
        )  # type: ignore[return-value]
    init = float(config.init)
    if config.init_total is not None:
        feature_count = len(patterns_for_set(config.pattern_set)) * len(SYMMETRIES)
        init = float(config.init_total) / float(feature_count)
    if config.stage_mode == "none":
        if config.stage_weight_promotion:
            raise ValueError("Stage weight promotion requires a staged model and a parent checkpoint")
        return NtupleValue.from_pattern_set(config.pattern_set, init=init)
    if config.stage_mode in ("phase4", "phase4_corner3"):
        if config.stage_weight_promotion:
            raise ValueError("Stage weight promotion requires --resume from the frozen parent model")
        return StagedNtupleValue.from_pattern_set(
            config.pattern_set,
            init=init,
            stage_mode=config.stage_mode,
            starter_tile=starter_options(config)[0],
            lazy=bool(config.lazy_stages),
        )  # type: ignore[return-value]
    raise ValueError(f"Unsupported stage_mode: {config.stage_mode}")


def train(config: TDConfig, resume: Path | None = None) -> Path:
    if config.target_mode not in ("td", "mc", "nstep"):
        raise ValueError(f"Unsupported target mode: {config.target_mode}")
    run_dir = Path("threes_rl/runs") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))
    games_completed = 0
    if resume is not None:
        resume_meta = json.loads((resume / "meta.json").read_text())
        resume_config = resume_meta.get("training_config")
        resume_run_name = resume_config.get("run_name") if isinstance(resume_config, dict) else None
        games_completed = (
            int(resume_meta.get("games_completed", 0))
            if resume_run_name == config.run_name
            else 0
        )
        value_model = NtupleValue.load(resume)
        if config.frozen_incumbent_policy is not None:
            if not isinstance(value_model, ResidualStagedNtupleValue):
                raise ValueError("Frozen-incumbent residual run must resume a residual composite checkpoint")
            if value_model.frozen_policy_spec != config.frozen_incumbent_policy:
                raise ValueError("Resume checkpoint frozen incumbent does not match the configured policy")
        elif config.stage_mode != "none" and not isinstance(value_model, StagedNtupleValue):
            value_model = StagedNtupleValue.from_base_model(
                value_model,
                stage_mode=config.stage_mode,
                starter_tile=starter_options(config)[0],
                promotion_enabled=bool(config.stage_weight_promotion),
                promotion_copy_tc=True,
            )  # type: ignore[assignment]
        elif isinstance(value_model, StagedNtupleValue):
            if bool(config.stage_weight_promotion) != bool(value_model.promotion_enabled):
                raise ValueError(
                    "Resume checkpoint promotion mode does not match stage_weight_promotion config"
                )
    else:
        value_model = create_value_model(config)
    if value_model.pattern_set != config.pattern_set:
        raise ValueError(
            f"Resume checkpoint pattern set {value_model.pattern_set!r} does not match config {config.pattern_set!r}"
        )
    if config.stage_weight_promotion and not config.use_tc:
        raise ValueError("The frozen R1 promotion configuration requires temporal coherence")
    if config.frozen_incumbent_policy is not None and config.actor_policy != config.frozen_incumbent_policy:
        raise ValueError("Residual training action generation must remain incumbent-fixed")
    if config.actor_generation_jobs > 1 and config.frozen_incumbent_policy is None:
        raise ValueError("Parallel actor generation is restricted to frozen-incumbent residual training")
    if config.actor_generation_jobs > 1 and config.target_mode != "nstep":
        raise ValueError("Parallel fixed-actor generation currently requires n-step targets")
    if config.use_tc:
        value_model.enable_temporal_coherence()
    if games_completed > config.games:
        raise ValueError(
            f"Resume checkpoint already has {games_completed} games, beyond requested total {config.games}"
        )
    actor_policy = None
    if config.actor_policy and config.actor_generation_jobs <= 1:
        from threes_rl.eval import make_policy

        actor_policy = make_policy(config.actor_policy)
    start_states = load_start_state_records(config.start_state_replays, config.start_state_min_tile, starter_options(config)[0])
    start_state_reservoir = (
        StartStateReservoir(
            start_states,
            starter_tile=starter_options(config)[0],
            sample_mode=config.start_state_sample_mode,
        )
        if start_states
        else None
    )
    if start_states:
        print(
            json.dumps(
                {
                    "loaded_start_states": len(start_states),
                    "start_state_prob": config.start_state_prob,
                    "start_state_min_tile": config.start_state_min_tile,
                    "start_state_sample_mode": config.start_state_sample_mode,
                "start_state_phase_buckets": start_state_reservoir.summary()["phase_buckets"]
                    if start_state_reservoir is not None
                    else {},
                    "start_state_unique_ancestries": start_state_reservoir.summary().get("unique_ancestries_by_stage", {})
                    if start_state_reservoir is not None
                    else {},
                },
                sort_keys=True,
            ),
            flush=True,
        )

    results, metrics_rows = load_existing_metrics(run_dir) if games_completed else ([], [])
    if games_completed and len(results) != games_completed:
        raise ValueError(
            f"Resume checkpoint reports {games_completed} games but metrics.csv has {len(results)} rows"
        )
    progress_rows = load_existing_progress(run_dir) if games_completed else []
    top_records = load_existing_top_records(run_dir) if games_completed else []
    if start_state_reservoir is not None and metrics_rows:
        start_state_reservoir.restore_sampling_metrics(metrics_rows)
    start_time = time.perf_counter()

    def episode_source():
        if config.actor_generation_jobs <= 1:
            for source_game_idx in range(games_completed + 1, config.games + 1):
                yield source_game_idx, play_episode(
                    value_model,
                    config,
                    source_game_idx,
                    actor_policy=actor_policy,
                    start_state_reservoir=start_state_reservoir,
                )
            return
        batch_size = max(1, int(config.progress_every) if config.progress_every > 0 else 100)
        first = games_completed + 1
        while first <= config.games:
            stop = min(config.games + 1, first + batch_size)
            fixed_jobs = prepare_fixed_actor_jobs(config, range(first, stop), start_state_reservoir)
            yield from iter_fixed_actor_episodes(config, fixed_jobs)
            first = stop

    for game_idx, episode in episode_source():
        if episode.deferred_nstep_afterstates:
            errors, updates_applied, updates_skipped = apply_nstep_updates(
                value_model,
                episode.deferred_nstep_afterstates,
                episode.result.score,
                config,
            )
            episode.mean_abs_td_error = float(mean(errors)) if errors else 0.0
            episode.updates_applied = updates_applied
            episode.updates_skipped = updates_skipped
            episode.deferred_nstep_afterstates = []
        results.append(episode.result)
        top_records = maybe_track_top(top_records, episode, config.keep_top_games)

        metrics_rows.append(
            {
                "game": game_idx,
                "seed": episode.result.seed,
                "starter_tile": episode.result.starter_tile,
                "score": episode.result.score,
                "score_minus_starter": episode.result.score_minus_starter,
                "moves": episode.result.moves,
                "max_tile": episode.result.max_tile,
                "max_tile_excl_starter": episode.result.max_tile_excl_starter,
                "terminal_tile": episode.result.terminal_tile,
                "mean_abs_td_error": episode.mean_abs_td_error,
                "updates_applied": episode.updates_applied,
                "updates_skipped": episode.updates_skipped,
                "replay_origin": episode.replay.get("replay_origin"),
                "root_origin": episode.replay.get("root_origin"),
                "root_replay": episode.replay.get("root_replay"),
                "start_type": "restart" if episode.sampled_start is not None else "normal",
                "restart_stage": (
                    START_STATE_PHASE_NAMES[
                        start_state_phase_index(episode.sampled_start.state, episode.sampled_start.starter_tile)
                    ]
                    if episode.sampled_start is not None
                    else None
                ),
                "restart_ancestry": episode.sampled_start.ancestry_key if episode.sampled_start is not None else None,
                "restart_outcome": episode.sampled_start.trajectory_outcome if episode.sampled_start is not None else None,
                "restart_behavior_family": episode.sampled_start.behavior_family if episode.sampled_start is not None else None,
            }
        )

        if config.progress_every > 0 and game_idx % config.progress_every == 0:
            row = progress_row(results)
            row["elapsed_s"] = time.perf_counter() - start_time
            row["games_per_s"] = (game_idx - games_completed) / max(1e-9, float(row["elapsed_s"]))
            row["mean_abs_td_error_recent"] = float(mean(metric["mean_abs_td_error"] for metric in metrics_rows[-config.progress_every :]))
            row["updates_applied_recent"] = int(sum(metric["updates_applied"] for metric in metrics_rows[-config.progress_every :]))
            row["updates_skipped_recent"] = int(sum(metric["updates_skipped"] for metric in metrics_rows[-config.progress_every :]))
            progress_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            write_progress_csv(run_dir / "progress.csv", progress_rows)
            write_progress_chart(run_dir / "progress.html", progress_rows, f"TD {config.run_name}")
            write_json(run_dir / "summary.json", summarize(results))
            write_json(
                run_dir / "training_diagnostics.json",
                {
                    "games_completed": game_idx,
                    "restart_sampling": start_state_reservoir.sampling_summary()
                    if start_state_reservoir is not None
                    else None,
                    "stage_metrics": value_model.stage_metrics()
                    if isinstance(value_model, (StagedNtupleValue, ResidualStagedNtupleValue))
                    else None,
                },
            )
            write_top_training_replays(run_dir, top_records)

        if config.checkpoint_every > 0 and game_idx % config.checkpoint_every == 0:
            save_checkpoint(value_model, run_dir, f"checkpoint_game_{game_idx}", config, game_idx)
            save_checkpoint(value_model, run_dir, "latest", config, game_idx)

    if not progress_rows or progress_rows[-1].get("games") != len(results):
        progress_rows.append(progress_row(results))
    write_progress_csv(run_dir / "progress.csv", progress_rows)
    write_progress_chart(run_dir / "progress.html", progress_rows, f"TD {config.run_name}")
    write_progress_csv(run_dir / "metrics.csv", metrics_rows)
    summary = summarize(results)
    summary["updates_applied"] = int(sum(metric["updates_applied"] for metric in metrics_rows))
    summary["updates_skipped"] = int(sum(metric["updates_skipped"] for metric in metrics_rows))
    summary["train_phase_filter"] = list(config.train_phase_filter) if config.train_phase_filter else None
    normal_results = [result for result, metric in zip(results, metrics_rows) if metric["start_type"] == "normal"]
    restart_results = [result for result, metric in zip(results, metrics_rows) if metric["start_type"] == "restart"]
    summary["normal_start_training"] = summarize(normal_results) if normal_results else {"games": 0}
    summary["restart_start_training"] = summarize(restart_results) if restart_results else {"games": 0}
    summary["restart_sampling"] = (
        start_state_reservoir.sampling_summary() if start_state_reservoir is not None else None
    )
    stage_summary = allocated_stage_summary(value_model)
    if stage_summary is not None:
        summary["allocated_stages"] = stage_summary
    summary["top_games"] = write_top_training_replays(run_dir, top_records)
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "training_diagnostics.json",
        {
            "games_completed": config.games,
            "restart_sampling": summary["restart_sampling"],
            "stage_metrics": value_model.stage_metrics()
            if isinstance(value_model, (StagedNtupleValue, ResidualStagedNtupleValue))
            else None,
        },
    )
    save_checkpoint(value_model, run_dir, "latest", config, config.games)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"latest_checkpoint={run_dir / 'latest'}", flush=True)
    return run_dir / "latest"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"td_{int(time.time())}")
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--pattern-set", choices=["tiny", "small", "default", "big6"], default="default")
    parser.add_argument(
        "--stage-mode",
        choices=["none", "phase4", "phase4_corner3"],
        default="none",
        help="Optional board-phase-conditioned value tables.",
    )
    parser.add_argument(
        "--lazy-stages",
        action="store_true",
        help="For staged value tables, allocate each phase/risk table on first update instead of eagerly.",
    )
    parser.add_argument(
        "--stage-weight-promotion",
        action="store_true",
        help="Initialize stage 0 from --resume and lazily promote exact feature weights plus TC state to later stages.",
    )
    parser.add_argument(
        "--frozen-incumbent-policy-file",
        type=Path,
        help="Build a zero residual over the exact frozen depth-2 incumbent leaf and keep it as the actor.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--init", type=float, default=0.0, help="Initial value per lookup-table entry.")
    parser.add_argument("--init-total", type=float, help="Initial total board value, divided across active n-tuple features.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--starter", default="1536")
    parser.add_argument(
        "--starter-curriculum",
        help="Comma-separated starter values to cycle through during training, e.g. none,96,384,1536.",
    )
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument(
        "--continuation-max-moves",
        type=int,
        help="For replay-start episodes, cap moves relative to the sampled frame instead of by absolute move_count.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--keep-top-games", type=int, default=3)
    parser.add_argument("--actor-policy", help="Optional policy spec to generate trajectories, e.g. corner2.")
    parser.add_argument(
        "--actor-generation-jobs",
        type=int,
        default=1,
        help="Parallel workers for frozen-policy trajectory generation; updates remain game-index ordered.",
    )
    parser.add_argument("--target-mode", choices=["td", "mc", "nstep"], default="td", help="Use one-step TD, Monte Carlo returns, or n-step bootstrapped returns.")
    parser.add_argument("--n-step", type=int, default=8, help="Bootstrap horizon for --target-mode nstep.")
    parser.add_argument("--use-tc", action="store_true", help="Use temporal-coherence per-feature step scaling.")
    parser.add_argument("--start-state-replay", action="append", default=[], help="Replay JSON to sample high-board episode starts from. Can be passed multiple times.")
    parser.add_argument("--start-state-prob", type=float, default=0.0, help="Probability of starting an episode from the replay reservoir.")
    parser.add_argument(
        "--exact-start-mix",
        action="store_true",
        help="Use a deterministic alternating 50/50 normal/restart schedule; requires --start-state-prob 0.5.",
    )
    parser.add_argument("--start-state-min-tile", type=int, default=768, help="Minimum max tile excluding the starter for replay reservoir states.")
    parser.add_argument(
        "--start-state-sample-mode",
        choices=["flat", "phase_balanced", "ancestry_balanced"],
        default="flat",
        help="Sample restart states flat, by phase, or by phase then root ancestry then state.",
    )
    parser.add_argument(
        "--train-phase-filter",
        help="Comma-separated afterstate phase names to update, e.g. late,endgame. Other phases are played but not trained.",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    starter_tile = parse_starter_values(args.starter)[0]
    starter_tiles = parse_starter_values(args.starter_curriculum) if args.starter_curriculum else None
    frozen_incumbent_policy = None
    if args.frozen_incumbent_policy_file is not None:
        frozen_incumbent_policy = next(
            line.strip()
            for line in args.frozen_incumbent_policy_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if args.actor_policy is not None and args.actor_policy != frozen_incumbent_policy:
            parser.error("--actor-policy must match --frozen-incumbent-policy-file exactly")
    config = TDConfig(
        run_name=args.run_name,
        games=args.games,
        pattern_set=args.pattern_set,
        stage_mode=args.stage_mode,
        lazy_stages=args.lazy_stages,
        alpha=args.alpha,
        epsilon=args.epsilon,
        init=args.init,
        init_total=args.init_total,
        seed=args.seed,
        starter_tile=starter_tile,
        starter_tiles=starter_tiles,
        max_moves=args.max_moves,
        continuation_max_moves=args.continuation_max_moves,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        keep_top_games=args.keep_top_games,
        actor_policy=frozen_incumbent_policy or args.actor_policy,
        target_mode=args.target_mode,
        n_step=args.n_step,
        use_tc=args.use_tc,
        start_state_replays=args.start_state_replay,
        start_state_prob=args.start_state_prob,
        start_state_min_tile=args.start_state_min_tile,
        start_state_sample_mode=args.start_state_sample_mode,
        train_phase_filter=parse_phase_filter(args.train_phase_filter),
        stage_weight_promotion=args.stage_weight_promotion,
        promotion_copy_tc=True,
        exact_start_mix=args.exact_start_mix,
        frozen_incumbent_policy=frozen_incumbent_policy,
        actor_generation_jobs=args.actor_generation_jobs,
    )
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()

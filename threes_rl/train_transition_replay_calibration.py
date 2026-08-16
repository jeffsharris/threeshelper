"""Calibrate n-tuple values from replay suffixes referenced by state records."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from threes_rl.ntuple import (
    PHASE4_NAMES,
    NtupleValue,
    SYMMETRIES,
    StagedNtupleValue,
    patterns_for_set,
    phase4_index_for_board,
)
from threes_rl.run_artifacts import write_json, write_progress_csv
from threes_rl.sim import direction_index, score_board, simulate_base_move
from threes_rl.train_replay_calibration import final_score_for_replay
from threes_rl.train_td import parse_phase_filter, state_from_replay_payload
from threes_rl.transition_reachability_audit import load_records, row_from_record
from threes_rl.transition_reachability_score import (
    _parse_equal_feature_filters,
    _parse_max_feature_filters,
    _parse_min_feature_filters,
    _passes_equal_feature_filters,
    _passes_max_feature_filters,
    _passes_min_feature_filters,
)


@dataclass
class TransitionReplayCalibrationConfig:
    run_name: str
    records_json: list[str] = field(default_factory=list)
    epochs: int = 1
    pattern_set: str = "default"
    stage_mode: str = "none"
    alpha: float = 0.001
    use_tc: bool = False
    init: float = 0.0
    init_total: float | None = None
    starter_tile: int | None = 1536
    train_phase_filter: list[str] | None = None
    target_tile: int | None = None
    max_suffix_moves: int = 0
    success_weight: float = 1.0
    failure_weight: float = 1.0
    shuffle: bool = True
    seed: int = 1
    max_updates: int = 0
    progress_every: int = 1000
    dedupe_starts: bool = False
    candidate_min_feature: dict[str, float] = field(default_factory=dict)
    candidate_max_feature: dict[str, float] = field(default_factory=dict)
    candidate_feature_equals: dict[str, str] = field(default_factory=dict)


@dataclass
class SuffixExample:
    afterstate: np.ndarray
    target: float
    weight: float
    phase: str
    outcome: str | None
    source_replay: str
    source_seed: int | None
    source_frame_index: int
    move_count: int


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def create_value_model(config: TransitionReplayCalibrationConfig) -> NtupleValue:
    init = float(config.init)
    if config.init_total is not None:
        feature_count = len(patterns_for_set(config.pattern_set)) * len(SYMMETRIES)
        init = float(config.init_total) / float(feature_count)
    if config.stage_mode == "none":
        return NtupleValue.from_pattern_set(config.pattern_set, init=init)
    if config.stage_mode in ("phase4", "phase4_corner3"):
        return StagedNtupleValue.from_pattern_set(
            config.pattern_set,
            init=init,
            stage_mode=config.stage_mode,
            starter_tile=config.starter_tile,
        )  # type: ignore[return-value]
    raise ValueError(f"Unsupported stage_mode: {config.stage_mode}")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _outcome(record: dict[str, Any]) -> str | None:
    raw = record.get("outcome")
    if raw in ("success", "failure"):
        return str(raw)
    return None


def _weight_for_outcome(outcome: str | None, config: TransitionReplayCalibrationConfig) -> float:
    if outcome == "success":
        return float(config.success_weight)
    if outcome == "failure":
        return float(config.failure_weight)
    return 1.0


def _passes_filters(record: dict[str, Any], config: TransitionReplayCalibrationConfig) -> bool:
    if config.target_tile is not None and _as_int(record.get("target_tile")) != int(config.target_tile):
        return False
    row = row_from_record(record, require_outcome=False)
    if row is None:
        return False
    return (
        _passes_min_feature_filters(row, config.candidate_min_feature)
        and _passes_max_feature_filters(row, config.candidate_max_feature)
        and _passes_equal_feature_filters(row, config.candidate_feature_equals)
    )


def _frame_position(frames: list[Any], frame_index: int) -> int | None:
    for pos, frame in enumerate(frames):
        if isinstance(frame, dict) and _as_int(frame.get("index", pos)) == int(frame_index):
            return pos
    if 0 <= int(frame_index) < len(frames):
        return int(frame_index)
    return None


def _record_source(record: dict[str, Any]) -> tuple[Path | None, int | None, int | None]:
    source_replay = record.get("source_replay")
    source_frame = _as_int(record.get("source_frame_index", record.get("frame_index")))
    source_seed = _as_int(record.get("source_seed", record.get("seed")))
    if source_replay is None or source_frame is None:
        return None, None, source_seed
    return Path(str(source_replay)), int(source_frame), source_seed


def examples_from_record(
    record: dict[str, Any],
    *,
    config: TransitionReplayCalibrationConfig,
    replay_cache: dict[Path, dict[str, Any]],
    rejected: Counter[str],
) -> list[SuffixExample]:
    if not _passes_filters(record, config):
        rejected["record_filter"] += 1
        return []
    source_replay, source_frame_index, source_seed = _record_source(record)
    if source_replay is None or source_frame_index is None:
        rejected["missing_source"] += 1
        return []
    if not source_replay.exists():
        rejected["missing_replay"] += 1
        return []
    replay = replay_cache.get(source_replay)
    if replay is None:
        replay = json.loads(source_replay.read_text())
        replay_cache[source_replay] = replay
    frames = replay.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        rejected["bad_replay_frames"] += 1
        return []
    start_pos = _frame_position(frames, int(source_frame_index))
    if start_pos is None or start_pos >= len(frames) - 1:
        rejected["missing_start_frame"] += 1
        return []
    try:
        final_score = final_score_for_replay(replay)
    except ValueError:
        rejected["missing_final_score"] += 1
        return []

    start_state_payload = frames[start_pos].get("state") if isinstance(frames[start_pos], dict) else None
    if not isinstance(start_state_payload, dict):
        rejected["bad_start_state"] += 1
        return []
    start_state = state_from_replay_payload(start_state_payload)
    start_move_count = int(start_state.move_count)
    phase_filter = set(config.train_phase_filter) if config.train_phase_filter else None
    weight = _weight_for_outcome(_outcome(record), config)
    outcome = _outcome(record)

    examples: list[SuffixExample] = []
    for before_pos in range(start_pos, len(frames) - 1):
        before_frame = frames[before_pos]
        after_frame = frames[before_pos + 1]
        if not isinstance(before_frame, dict) or not isinstance(after_frame, dict):
            continue
        state_payload = before_frame.get("state")
        move = after_frame.get("move")
        if not isinstance(state_payload, dict) or not isinstance(move, dict) or move.get("action") is None:
            continue
        state = state_from_replay_payload(state_payload)
        if config.max_suffix_moves > 0 and int(state.move_count) - start_move_count >= int(config.max_suffix_moves):
            break
        try:
            action = direction_index(str(move["action"]))
        except ValueError:
            continue
        afterstate, eligible = simulate_base_move(state.board, action)
        if not eligible:
            continue
        phase = PHASE4_NAMES[phase4_index_for_board(afterstate, starter_tile=config.starter_tile)]
        if phase_filter is not None and phase not in phase_filter:
            continue
        target = float(final_score - score_board(afterstate))
        examples.append(
            SuffixExample(
                afterstate=np.asarray(afterstate, dtype=np.int32),
                target=target,
                weight=weight,
                phase=phase,
                outcome=outcome,
                source_replay=str(source_replay),
                source_seed=source_seed,
                source_frame_index=int(source_frame_index),
                move_count=int(state.move_count),
            )
        )
    if not examples:
        rejected["no_examples"] += 1
    return examples


def load_examples(config: TransitionReplayCalibrationConfig) -> tuple[list[SuffixExample], dict[str, int]]:
    records = load_records([Path(path) for path in config.records_json])
    replay_cache: dict[Path, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    examples: list[SuffixExample] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        source_replay, source_frame_index, _source_seed = _record_source(record)
        if config.dedupe_starts and source_replay is not None and source_frame_index is not None:
            key = (str(source_replay), int(source_frame_index))
            if key in seen:
                rejected["duplicate_start"] += 1
                continue
            seen.add(key)
        examples.extend(examples_from_record(record, config=config, replay_cache=replay_cache, rejected=rejected))
    return examples, dict(rejected)


def update_value(value_model: NtupleValue, board: np.ndarray, target: float, alpha: float, use_tc: bool) -> float:
    if use_tc:
        return value_model.update_tc(board, target, alpha)
    return value_model.update(board, target, alpha)


def calibrate(config: TransitionReplayCalibrationConfig, resume: Path | None = None) -> Path:
    run_dir = Path("threes_rl/runs") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(config))
    value_model = NtupleValue.load(resume) if resume is not None else create_value_model(config)
    if config.stage_mode != "none" and not isinstance(value_model, StagedNtupleValue):
        value_model = StagedNtupleValue.from_base_model(
            value_model,
            stage_mode=config.stage_mode,
            starter_tile=config.starter_tile,
        )  # type: ignore[assignment]

    examples, rejected = load_examples(config)
    if not examples:
        raise ValueError("No replay-suffix examples were loaded")

    rng = np.random.default_rng(int(config.seed))
    order = np.arange(len(examples), dtype=np.int64)
    progress_rows: list[dict[str, object]] = []
    errors: list[float] = []
    updates = 0
    start_time = time.perf_counter()
    phase_counts: Counter[str] = Counter(example.phase for example in examples)
    outcome_counts: Counter[str] = Counter(example.outcome or "unknown" for example in examples)
    source_replays = {example.source_replay for example in examples}
    start_keys = {(example.source_replay, int(example.source_frame_index)) for example in examples}

    for epoch in range(1, int(config.epochs) + 1):
        if config.shuffle:
            rng.shuffle(order)
        for example_idx in order:
            if config.max_updates > 0 and updates >= int(config.max_updates):
                break
            example = examples[int(example_idx)]
            effective_alpha = float(config.alpha) * max(0.0, float(example.weight))
            errors.append(abs(update_value(value_model, example.afterstate, example.target, effective_alpha, config.use_tc)))
            updates += 1
            if config.progress_every > 0 and updates % int(config.progress_every) == 0:
                progress_rows.append(
                    {
                        "updates": updates,
                        "epoch": epoch,
                        "elapsed_s": time.perf_counter() - start_time,
                        "mean_abs_error_recent": float(mean(errors[-int(config.progress_every) :])),
                    }
                )
                print(json.dumps(progress_rows[-1], sort_keys=True), flush=True)
                write_progress_csv(run_dir / "progress.csv", progress_rows)
        if config.max_updates > 0 and updates >= int(config.max_updates):
            break

    if not progress_rows or progress_rows[-1].get("updates") != updates:
        progress_rows.append(
            {
                "updates": updates,
                "epoch": min(int(config.epochs), max(1, int(config.epochs))),
                "elapsed_s": time.perf_counter() - start_time,
                "mean_abs_error_recent": float(mean(errors[-max(1, min(len(errors), int(config.progress_every) or len(errors))) :])),
            }
        )
    write_progress_csv(run_dir / "progress.csv", progress_rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updates": int(updates),
        "examples": len(examples),
        "epochs": int(config.epochs),
        "mean_abs_error": float(mean(errors)) if errors else 0.0,
        "phase_counts": dict(phase_counts),
        "outcome_counts": dict(outcome_counts),
        "source_replays": len(source_replays),
        "source_starts": len(start_keys),
        "rejected": rejected,
        "train_phase_filter": list(config.train_phase_filter) if config.train_phase_filter else None,
        "source_records": list(config.records_json),
    }
    write_json(run_dir / "summary.json", summary)
    value_model.save(
        run_dir / "latest",
        extra_meta={
            "transition_replay_calibration_config": asdict(config),
            "updates_completed": int(updates),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"latest_checkpoint={run_dir / 'latest'}", flush=True)
    return run_dir / "latest"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"transition_replay_calibration_{int(time.time())}")
    parser.add_argument("--records-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pattern-set", choices=["tiny", "small", "default", "big6"], default="default")
    parser.add_argument("--stage-mode", choices=["none", "phase4", "phase4_corner3"], default="none")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--use-tc", action="store_true")
    parser.add_argument("--init", type=float, default=0.0)
    parser.add_argument("--init-total", type=float)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--train-phase-filter")
    parser.add_argument("--target-tile", type=int)
    parser.add_argument("--max-suffix-moves", type=int, default=0)
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--failure-weight", type=float, default=1.0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--dedupe-starts", action="store_true")
    parser.add_argument("--candidate-min-feature", action="append", default=[])
    parser.add_argument("--candidate-max-feature", action="append", default=[])
    parser.add_argument("--candidate-feature-equals", action="append", default=[])
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    starter = args.starter.strip().lower()
    starter_tile = None if starter == "none" else int(starter)
    paths = _flatten_paths(args.records_json)
    config = TransitionReplayCalibrationConfig(
        run_name=args.run_name,
        records_json=[str(path) for path in paths],
        epochs=args.epochs,
        pattern_set=args.pattern_set,
        stage_mode=args.stage_mode,
        alpha=args.alpha,
        use_tc=args.use_tc,
        init=args.init,
        init_total=args.init_total,
        starter_tile=starter_tile,
        train_phase_filter=parse_phase_filter(args.train_phase_filter),
        target_tile=args.target_tile,
        max_suffix_moves=args.max_suffix_moves,
        success_weight=args.success_weight,
        failure_weight=args.failure_weight,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        max_updates=args.max_updates,
        progress_every=args.progress_every,
        dedupe_starts=args.dedupe_starts,
        candidate_min_feature=_parse_min_feature_filters(args.candidate_min_feature),
        candidate_max_feature=_parse_max_feature_filters(args.candidate_max_feature),
        candidate_feature_equals=_parse_equal_feature_filters(args.candidate_feature_equals),
    )
    calibrate(config, resume=args.resume)


if __name__ == "__main__":
    main()

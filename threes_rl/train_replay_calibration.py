"""Calibrate n-tuple values from recorded high-score replay trajectories."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable

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
from threes_rl.train_td import parse_phase_filter, state_from_replay_payload


@dataclass
class ReplayCalibrationConfig:
    run_name: str
    replay_json: list[str] = field(default_factory=list)
    epochs: int = 1
    pattern_set: str = "default"
    stage_mode: str = "none"
    alpha: float = 0.001
    use_tc: bool = False
    init: float = 0.0
    init_total: float | None = None
    starter_tile: int | None = 1536
    train_phase_filter: list[str] | None = None
    max_updates: int = 0
    progress_every: int = 1000


@dataclass
class ReplayExample:
    afterstate: np.ndarray
    target: float
    phase: str
    replay: str
    seed: int | None
    move_count: int


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def create_value_model(config: ReplayCalibrationConfig) -> NtupleValue:
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


def final_score_for_replay(replay: dict[str, object]) -> int:
    final_score = replay.get("final_score")
    if isinstance(final_score, int):
        return int(final_score)
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay is missing final_score and frames")
    final_state = frames[-1].get("state") if isinstance(frames[-1], dict) else None
    if not isinstance(final_state, dict):
        raise ValueError("Replay final frame is missing a state")
    return int(score_board(np.asarray(final_state["board"], dtype=np.int32)))


def replay_examples(
    replay_path: Path,
    *,
    starter_tile: int | None,
    phase_filter: set[str] | None = None,
) -> list[ReplayExample]:
    replay = json.loads(replay_path.read_text())
    frames = replay.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"{replay_path} has no frames list")
    final_score = final_score_for_replay(replay)
    seed = replay.get("seed")
    seed_value = int(seed) if isinstance(seed, int) else None
    examples: list[ReplayExample] = []
    for before_idx in range(max(0, len(frames) - 1)):
        before_frame = frames[before_idx]
        after_frame = frames[before_idx + 1]
        if not isinstance(before_frame, dict) or not isinstance(after_frame, dict):
            continue
        state_payload = before_frame.get("state")
        move = after_frame.get("move")
        if not isinstance(state_payload, dict) or not isinstance(move, dict) or move.get("action") is None:
            continue
        state = state_from_replay_payload(state_payload)
        try:
            action = direction_index(str(move["action"]))
        except ValueError:
            continue
        afterstate, eligible = simulate_base_move(state.board, action)
        if not eligible:
            continue
        phase = PHASE4_NAMES[phase4_index_for_board(afterstate, starter_tile=starter_tile)]
        if phase_filter is not None and phase not in phase_filter:
            continue
        target = float(final_score - score_board(afterstate))
        examples.append(
            ReplayExample(
                afterstate=np.asarray(afterstate, dtype=np.int32),
                target=target,
                phase=phase,
                replay=str(replay_path),
                seed=seed_value,
                move_count=int(state.move_count),
            )
        )
    return examples


def load_examples(paths: Iterable[Path], *, starter_tile: int | None, phase_filter: set[str] | None) -> list[ReplayExample]:
    examples: list[ReplayExample] = []
    for path in paths:
        examples.extend(replay_examples(path, starter_tile=starter_tile, phase_filter=phase_filter))
    return examples


def update_value(value_model: NtupleValue, board: np.ndarray, target: float, alpha: float, use_tc: bool) -> float:
    if use_tc:
        return value_model.update_tc(board, target, alpha)
    return value_model.update(board, target, alpha)


def calibrate(config: ReplayCalibrationConfig, resume: Path | None = None) -> Path:
    run_dir = Path("threes_rl/runs") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))
    value_model = NtupleValue.load(resume) if resume is not None else create_value_model(config)
    if config.stage_mode != "none" and not isinstance(value_model, StagedNtupleValue):
        value_model = StagedNtupleValue.from_base_model(
            value_model,
            stage_mode=config.stage_mode,
            starter_tile=config.starter_tile,
        )  # type: ignore[assignment]

    phase_filter = set(config.train_phase_filter) if config.train_phase_filter else None
    examples = load_examples(
        [Path(path) for path in config.replay_json],
        starter_tile=config.starter_tile,
        phase_filter=phase_filter,
    )
    if not examples:
        raise ValueError("No replay examples were loaded")

    progress_rows: list[dict[str, object]] = []
    errors: list[float] = []
    updates = 0
    start_time = time.perf_counter()
    phase_counts: Counter[str] = Counter(example.phase for example in examples)
    for epoch in range(1, int(config.epochs) + 1):
        for example in examples:
            if config.max_updates > 0 and updates >= int(config.max_updates):
                break
            errors.append(abs(update_value(value_model, example.afterstate, example.target, config.alpha, config.use_tc)))
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
        "train_phase_filter": list(config.train_phase_filter) if config.train_phase_filter else None,
        "source_replays": list(config.replay_json),
    }
    write_json(run_dir / "summary.json", summary)
    value_model.save(
        run_dir / "latest",
        extra_meta={
            "replay_calibration_config": asdict(config),
            "updates_completed": int(updates),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"latest_checkpoint={run_dir / 'latest'}", flush=True)
    return run_dir / "latest"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"replay_calibration_{int(time.time())}")
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pattern-set", choices=["tiny", "small", "default", "big6"], default="default")
    parser.add_argument("--stage-mode", choices=["none", "phase4", "phase4_corner3"], default="none")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--use-tc", action="store_true")
    parser.add_argument("--init", type=float, default=0.0)
    parser.add_argument("--init-total", type=float)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--train-phase-filter")
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    starter = args.starter.strip().lower()
    starter_tile = None if starter == "none" else int(starter)
    replay_paths = _flatten_paths(args.replay_json)
    config = ReplayCalibrationConfig(
        run_name=args.run_name,
        replay_json=[str(path) for path in replay_paths],
        epochs=args.epochs,
        pattern_set=args.pattern_set,
        stage_mode=args.stage_mode,
        alpha=args.alpha,
        use_tc=args.use_tc,
        init=args.init,
        init_total=args.init_total,
        starter_tile=starter_tile,
        train_phase_filter=parse_phase_filter(args.train_phase_filter),
        max_updates=args.max_updates,
        progress_every=args.progress_every,
    )
    calibrate(config, resume=args.resume)


if __name__ == "__main__":
    main()

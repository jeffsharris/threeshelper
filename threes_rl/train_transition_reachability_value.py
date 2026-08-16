"""Train an n-tuple sidecar on transition-window reachability labels."""

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
    NtupleValue,
    SYMMETRIES,
    StagedNtupleValue,
    patterns_for_set,
)
from threes_rl.run_artifacts import write_json, write_progress_csv
from threes_rl.train_td import parse_phase_filter, state_from_replay_payload
from threes_rl.transition_reachability_score import (
    _parse_equal_feature_filters,
    _parse_max_feature_filters,
    _parse_min_feature_filters,
    _passes_equal_feature_filters,
    _passes_max_feature_filters,
    _passes_min_feature_filters,
)
from threes_rl.transition_reachability_audit import load_records, row_from_record


@dataclass
class ReachabilityValueConfig:
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
    success_target: float = 250_000.0
    failure_target: float = 0.0
    success_weight: float = 1.0
    failure_weight: float = 1.0
    shuffle: bool = True
    seed: int = 1
    max_updates: int = 0
    progress_every: int = 1000
    candidate_min_feature: dict[str, float] = field(default_factory=dict)
    candidate_max_feature: dict[str, float] = field(default_factory=dict)
    candidate_feature_equals: dict[str, str] = field(default_factory=dict)


@dataclass
class ReachabilityExample:
    board: np.ndarray
    target: float
    weight: float
    outcome: str
    phase: str | None
    source_replay: str | None
    source_seed: int | None
    source_frame_index: int | None


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def create_value_model(config: ReachabilityValueConfig) -> NtupleValue:
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
    if raw == "success":
        return "success"
    if raw == "failure":
        return "failure"
    return None


def _passes_filters(record: dict[str, Any], config: ReachabilityValueConfig) -> bool:
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


def examples_from_records(records: list[dict[str, Any]], config: ReachabilityValueConfig) -> list[ReachabilityExample]:
    phase_filter = set(config.train_phase_filter) if config.train_phase_filter else None
    examples: list[ReachabilityExample] = []
    for record in records:
        outcome = _outcome(record)
        if outcome is None or not _passes_filters(record, config):
            continue
        phase = record.get("phase")
        phase_name = str(phase) if phase is not None else None
        if phase_filter is not None and phase_name not in phase_filter:
            continue
        state_payload = record.get("state")
        if not isinstance(state_payload, dict):
            continue
        state = state_from_replay_payload(state_payload)
        if outcome == "success":
            target = float(config.success_target)
            weight = float(config.success_weight)
        else:
            target = float(config.failure_target)
            weight = float(config.failure_weight)
        examples.append(
            ReachabilityExample(
                board=np.asarray(state.board, dtype=np.int32),
                target=target,
                weight=weight,
                outcome=outcome,
                phase=phase_name,
                source_replay=str(record.get("source_replay")) if record.get("source_replay") is not None else None,
                source_seed=_as_int(record.get("source_seed")),
                source_frame_index=_as_int(record.get("source_frame_index")),
            )
        )
    return examples


def load_examples(config: ReachabilityValueConfig) -> list[ReachabilityExample]:
    records = load_records([Path(path) for path in config.records_json])
    return examples_from_records(records, config)


def update_value(value_model: NtupleValue, board: np.ndarray, target: float, alpha: float, use_tc: bool) -> float:
    if use_tc:
        return value_model.update_tc(board, target, alpha)
    return value_model.update(board, target, alpha)


def train(config: ReachabilityValueConfig, resume: Path | None = None) -> Path:
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

    examples = load_examples(config)
    if not examples:
        raise ValueError("No reachability examples were loaded")
    if len({example.outcome for example in examples}) < 2:
        raise ValueError("Reachability training requires both success and failure examples")

    rng = np.random.default_rng(int(config.seed))
    progress_rows: list[dict[str, object]] = []
    errors: list[float] = []
    updates = 0
    start_time = time.perf_counter()
    outcome_counts: Counter[str] = Counter(example.outcome for example in examples)
    phase_counts: Counter[str] = Counter(example.phase or "unknown" for example in examples)
    order = np.arange(len(examples), dtype=np.int64)
    for epoch in range(1, int(config.epochs) + 1):
        if config.shuffle:
            rng.shuffle(order)
        for example_idx in order:
            if config.max_updates > 0 and updates >= int(config.max_updates):
                break
            example = examples[int(example_idx)]
            effective_alpha = float(config.alpha) * max(0.0, float(example.weight))
            errors.append(abs(update_value(value_model, example.board, example.target, effective_alpha, config.use_tc)))
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
        "outcome_counts": dict(outcome_counts),
        "phase_counts": dict(phase_counts),
        "train_phase_filter": list(config.train_phase_filter) if config.train_phase_filter else None,
        "source_records": list(config.records_json),
    }
    write_json(run_dir / "summary.json", summary)
    value_model.save(
        run_dir / "latest",
        extra_meta={
            "transition_reachability_value_config": asdict(config),
            "updates_completed": int(updates),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"latest_checkpoint={run_dir / 'latest'}", flush=True)
    return run_dir / "latest"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"transition_reachability_value_{int(time.time())}")
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
    parser.add_argument("--success-target", type=float, default=250_000.0)
    parser.add_argument("--failure-target", type=float, default=0.0)
    parser.add_argument("--success-weight", type=float, default=1.0)
    parser.add_argument("--failure-weight", type=float, default=1.0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--candidate-min-feature", action="append", default=[])
    parser.add_argument("--candidate-max-feature", action="append", default=[])
    parser.add_argument("--candidate-feature-equals", action="append", default=[])
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    starter = args.starter.strip().lower()
    starter_tile = None if starter == "none" else int(starter)
    paths = _flatten_paths(args.records_json)
    config = ReachabilityValueConfig(
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
        success_target=args.success_target,
        failure_target=args.failure_target,
        success_weight=args.success_weight,
        failure_weight=args.failure_weight,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        max_updates=args.max_updates,
        progress_every=args.progress_every,
        candidate_min_feature=_parse_min_feature_filters(args.candidate_min_feature),
        candidate_max_feature=_parse_max_feature_filters(args.candidate_max_feature),
        candidate_feature_equals=_parse_equal_feature_filters(args.candidate_feature_equals),
    )
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()

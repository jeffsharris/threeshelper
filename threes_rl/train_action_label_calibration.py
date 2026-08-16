"""Train an n-tuple correction from continuation action labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from threes_rl.ntuple import (
    PHASE4_NAMES,
    NtupleValue,
    StagedNtupleValue,
    phase4_index_for_board,
)
from threes_rl.run_artifacts import write_json, write_progress_csv
from threes_rl.sim import direction_index, score_board, simulate_base_move
from threes_rl.train_td import parse_phase_filter, state_from_replay_payload


@dataclass
class ActionLabelCalibrationConfig:
    run_name: str
    swing_label_json: list[str] = field(default_factory=list)
    endgame_label_json: list[str] = field(default_factory=list)
    epochs: int = 20
    pattern_set: str = "default"
    stage_mode: str = "phase4"
    alpha: float = 0.01
    use_tc: bool = True
    lazy_stages: bool = False
    init: float = 0.0
    starter_tile: int | None = 1536
    target_mode: str = "centered_afterstate"
    horizon: int = 64
    train_phase_filter: list[str] | None = None
    corner_risk_filter: list[str] | None = None
    label_weight_mode: str = "uniform"
    progress_every: int = 100
    seed: int = 20260706


@dataclass
class ActionLabelExample:
    group_id: str
    action: str
    afterstate: np.ndarray
    raw_afterstate_target: float
    target: float
    source: str
    phase: str
    corner_risk: str | None
    weight: float = 1.0
    weight_reason: str = "uniform"


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
    values: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        normalized = part.strip().lower()
        if not normalized:
            continue
        if normalized not in seen:
            values.append(normalized)
            seen.add(normalized)
    return values or None


def create_value_model(config: ActionLabelCalibrationConfig) -> NtupleValue:
    if config.stage_mode == "none":
        return NtupleValue.from_pattern_set(config.pattern_set, init=config.init)
    if config.stage_mode in ("phase4", "phase4_corner3"):
        return StagedNtupleValue.from_pattern_set(
            config.pattern_set,
            init=config.init,
            stage_mode=config.stage_mode,
            starter_tile=config.starter_tile,
            lazy=config.lazy_stages,
        )  # type: ignore[return-value]
    raise ValueError(f"Unsupported stage_mode: {config.stage_mode}")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def confidence_regret_weight(
    *,
    stable: bool,
    confidence: float,
    regret: float,
    winner: str | None,
    base_action: str | None,
    p6144: float = 0.0,
) -> tuple[float, str]:
    """Return a group weight for noisy action labels.

    Stable corrective labels should dominate. No-regret/base-action labels still
    provide scale, but should not wash out rare tail-correction signal.
    """
    correction = bool(winner and base_action and winner != base_action and regret > 0.0)
    confidence_term = _clamp((float(confidence) - 0.5) / 0.5, 0.0, 1.0)
    weight = 0.25 + 1.5 * confidence_term
    if stable:
        weight += 0.5
    if correction:
        weight += _clamp(float(regret) / 20_000.0, 0.0, 2.0)
    else:
        weight *= 0.35
    if p6144 > 0:
        weight += _clamp(float(p6144) * 2.0, 0.0, 1.0)
    weight = _clamp(weight, 0.1, 4.0)
    reason = (
        "confidence_regret:"
        f"stable={bool(stable)},"
        f"confidence={float(confidence):.3f},"
        f"correction={correction},"
        f"regret={float(regret):.1f},"
        f"p6144={float(p6144):.3f}"
    )
    return weight, reason


def _label_weight_from_swing_item(item: dict[str, Any], label: dict[str, Any], mode: str) -> tuple[float, str]:
    if mode == "uniform":
        return 1.0, "uniform"
    if mode != "confidence_regret":
        raise ValueError(f"Unsupported label_weight_mode: {mode}")
    return confidence_regret_weight(
        stable=bool(label.get("stable")),
        confidence=_as_float(label.get("min_bootstrap_winner_fraction"), 0.5),
        regret=_as_float(label.get("oracle_regret_at_max_horizon"), 0.0),
        winner=str(label.get("oracle_winner") or label.get("stable_winner") or "") or None,
        base_action=str(item.get("base_action")) if item.get("base_action") is not None else None,
    )


def _label_weight_from_endgame_item(item: dict[str, Any], mode: str) -> tuple[float, str]:
    if mode == "uniform":
        return 1.0, "uniform"
    if mode != "confidence_regret":
        raise ValueError(f"Unsupported label_weight_mode: {mode}")
    return confidence_regret_weight(
        stable=bool(item.get("stable")),
        confidence=_as_float(item.get("bootstrap_winner_fraction"), 0.5),
        regret=_as_float(item.get("oracle_regret"), 0.0),
        winner=str(item.get("winner")) if item.get("winner") is not None else None,
        base_action=str(item.get("base_action")) if item.get("base_action") is not None else None,
        p6144=_as_float(item.get("winner_p6144"), 0.0),
    )


def _phase_for_afterstate(afterstate: np.ndarray, starter_tile: int | None) -> str:
    return PHASE4_NAMES[phase4_index_for_board(afterstate, starter_tile=starter_tile)]


def _afterstate_target_for_action(state_payload: dict[str, Any], action_name: str, raw_delta: float) -> tuple[np.ndarray, float] | None:
    state = state_from_replay_payload(state_payload)
    try:
        action = direction_index(action_name)
    except ValueError:
        return None
    afterstate, eligible = simulate_base_move(state.board, action)
    if not eligible:
        return None
    merge_delta = score_board(afterstate) - score_board(state.board)
    return np.asarray(afterstate, dtype=np.int32), float(raw_delta) - float(merge_delta)


def _center_group(
    rows: list[tuple[str, np.ndarray, float, str, str, str | None]],
    *,
    target_mode: str,
    base_action: str | None,
    group_weight: float = 1.0,
    weight_reason: str = "uniform",
) -> list[ActionLabelExample]:
    if not rows:
        return []
    if target_mode == "raw_afterstate":
        center = 0.0
    elif target_mode == "centered_afterstate":
        center = float(mean(row[2] for row in rows))
    elif target_mode == "base_centered_afterstate":
        base_rows = [row for row in rows if row[0] == base_action]
        center = float(base_rows[0][2]) if base_rows else float(mean(row[2] for row in rows))
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    group_id = rows[0][3]
    return [
        ActionLabelExample(
            group_id=group_id,
            action=action,
            afterstate=afterstate,
            raw_afterstate_target=float(raw_target),
            target=float(raw_target - center),
            source=source,
            phase=phase,
            corner_risk=corner_risk,
            weight=float(group_weight),
            weight_reason=weight_reason,
        )
        for action, afterstate, raw_target, group_id, source, phase, corner_risk in rows
    ]


def examples_from_swing_label_file(
    path: Path,
    *,
    horizon: int,
    target_mode: str,
    starter_tile: int | None,
    label_weight_mode: str = "uniform",
) -> list[ActionLabelExample]:
    payload = json.loads(path.read_text())
    examples: list[ActionLabelExample] = []
    for item in payload.get("labels", []):
        if not isinstance(item, dict) or not isinstance(item.get("state"), dict):
            continue
        label = item.get("label", {})
        if not isinstance(label, dict):
            continue
        group_weight, weight_reason = _label_weight_from_swing_item(item, label, label_weight_mode)
        by_action = label.get("by_action", {})
        if not isinstance(by_action, dict):
            continue
        horizons = [int(value) for value in label.get("horizons", []) if isinstance(value, int)]
        chosen_horizon = int(horizon) if str(horizon) in {str(value) for value in horizons} else max(horizons or [horizon])
        features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
        rows: list[tuple[str, np.ndarray, float, str, str, str | None]] = []
        for action_name, action_payload in by_action.items():
            if not isinstance(action_payload, dict):
                continue
            values = action_payload.get(str(chosen_horizon))
            if not isinstance(values, list) or not values:
                continue
            converted = _afterstate_target_for_action(item["state"], str(action_name), float(mean(float(v) for v in values)))
            if converted is None:
                continue
            afterstate, raw_afterstate_target = converted
            phase = _phase_for_afterstate(afterstate, starter_tile)
            rows.append(
                (
                    str(action_name),
                    afterstate,
                    raw_afterstate_target,
                    str(item.get("id", f"{path}:{len(examples)}")),
                    str(path),
                    phase,
                    str(features.get("corner_risk")) if features.get("corner_risk") is not None else None,
                )
            )
        examples.extend(
            _center_group(
                rows,
                target_mode=target_mode,
                base_action=str(item.get("base_action")),
                group_weight=group_weight,
                weight_reason=weight_reason,
            )
        )
    return examples


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


def examples_from_endgame_label_file(
    path: Path,
    *,
    target_mode: str,
    starter_tile: int | None,
    label_weight_mode: str = "uniform",
) -> list[ActionLabelExample]:
    payload = json.loads(path.read_text())
    examples: list[ActionLabelExample] = []
    state_cache: dict[tuple[str, int], dict[str, Any] | None] = {}
    for item in payload.get("labels", []):
        if not isinstance(item, dict):
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
        features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
        group_weight, weight_reason = _label_weight_from_endgame_item(item, label_weight_mode)
        rows: list[tuple[str, np.ndarray, float, str, str, str | None]] = []
        for action_result in item.get("action_results", []):
            if not isinstance(action_result, dict):
                continue
            action_name = str(action_result.get("action"))
            mean_delta = action_result.get("mean_delta")
            if mean_delta is None:
                continue
            converted = _afterstate_target_for_action(state_payload, action_name, float(mean_delta))
            if converted is None:
                continue
            afterstate, raw_afterstate_target = converted
            phase = _phase_for_afterstate(afterstate, starter_tile)
            rows.append(
                (
                    action_name,
                    afterstate,
                    raw_afterstate_target,
                    str(item.get("id", f"{path}:{len(examples)}")),
                    str(path),
                    phase,
                    str(features.get("corner_risk")) if features.get("corner_risk") is not None else None,
                )
            )
        examples.extend(
            _center_group(
                rows,
                target_mode=target_mode,
                base_action=str(item.get("base_action")),
                group_weight=group_weight,
                weight_reason=weight_reason,
            )
        )
    return examples


def filter_examples(
    examples: list[ActionLabelExample],
    *,
    phase_filter: set[str] | None,
    corner_risk_filter: set[str] | None,
) -> list[ActionLabelExample]:
    out = []
    for example in examples:
        if phase_filter is not None and example.phase not in phase_filter:
            continue
        if corner_risk_filter is not None and example.corner_risk not in corner_risk_filter:
            continue
        out.append(example)
    return out


def load_examples(config: ActionLabelCalibrationConfig) -> list[ActionLabelExample]:
    examples: list[ActionLabelExample] = []
    for text_path in config.swing_label_json:
        examples.extend(
            examples_from_swing_label_file(
                Path(text_path),
                horizon=config.horizon,
                target_mode=config.target_mode,
                starter_tile=config.starter_tile,
                label_weight_mode=config.label_weight_mode,
            )
        )
    for text_path in config.endgame_label_json:
        examples.extend(
            examples_from_endgame_label_file(
                Path(text_path),
                target_mode=config.target_mode,
                starter_tile=config.starter_tile,
                label_weight_mode=config.label_weight_mode,
            )
        )
    phase_filter = set(config.train_phase_filter) if config.train_phase_filter else None
    corner_filter = set(config.corner_risk_filter) if config.corner_risk_filter else None
    return filter_examples(examples, phase_filter=phase_filter, corner_risk_filter=corner_filter)


def update_value(value_model: NtupleValue, board: np.ndarray, target: float, alpha: float, use_tc: bool) -> float:
    if use_tc:
        return value_model.update_tc(board, target, alpha)
    return value_model.update(board, target, alpha)


def preference_accuracy(value_model: NtupleValue, examples: list[ActionLabelExample]) -> dict[str, float | int]:
    groups: dict[str, list[ActionLabelExample]] = defaultdict(list)
    for example in examples:
        groups[example.group_id].append(example)
    total = 0
    correct = 0
    tied = 0
    for group_examples in groups.values():
        if len(group_examples) < 2:
            continue
        target_best = max(example.target for example in group_examples)
        target_winners = {example.action for example in group_examples if example.target == target_best}
        pred_values = {example.action: float(value_model.value(example.afterstate)) for example in group_examples}
        pred_best = max(pred_values.values())
        pred_winners = {action for action, value in pred_values.items() if value == pred_best}
        total += 1
        if pred_winners & target_winners:
            correct += 1
        if len(pred_winners) > 1:
            tied += 1
    return {
        "preference_groups": int(total),
        "preference_accuracy": float(correct / total) if total else 0.0,
        "prediction_tie_groups": int(tied),
    }


def calibrate(config: ActionLabelCalibrationConfig) -> Path:
    run_dir = Path("threes_rl/runs") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(config))
    value_model = create_value_model(config)
    examples = load_examples(config)
    if not examples:
        raise ValueError("No action-label examples were loaded")

    rng = np.random.default_rng(int(config.seed))
    order = np.arange(len(examples), dtype=np.int32)
    progress_rows: list[dict[str, object]] = []
    errors: list[float] = []
    updates = 0
    start_time = time.perf_counter()
    for epoch in range(1, int(config.epochs) + 1):
        rng.shuffle(order)
        for idx in order:
            example = examples[int(idx)]
            error = update_value(
                value_model,
                example.afterstate,
                example.target,
                config.alpha * float(example.weight),
                config.use_tc,
            )
            errors.append(abs(float(error)))
            updates += 1
            if config.progress_every > 0 and updates % int(config.progress_every) == 0:
                row = {
                    "updates": updates,
                    "epoch": epoch,
                    "elapsed_s": time.perf_counter() - start_time,
                    "mean_abs_error_recent": float(mean(errors[-int(config.progress_every) :])),
                }
                progress_rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                write_progress_csv(run_dir / "progress.csv", progress_rows)

    if not progress_rows or progress_rows[-1].get("updates") != updates:
        progress_rows.append(
            {
                "updates": updates,
                "epoch": int(config.epochs),
                "elapsed_s": time.perf_counter() - start_time,
                "mean_abs_error_recent": float(mean(errors[-max(1, min(len(errors), int(config.progress_every) or len(errors))) :])),
            }
        )
    write_progress_csv(run_dir / "progress.csv", progress_rows)

    phase_counts = Counter(example.phase for example in examples)
    corner_counts = Counter(str(example.corner_risk) for example in examples)
    targets = [example.target for example in examples]
    raw_targets = [example.raw_afterstate_target for example in examples]
    weights = [float(example.weight) for example in examples]
    predictions = [float(value_model.value(example.afterstate)) for example in examples]
    abs_errors = [abs(target - pred) for target, pred in zip(targets, predictions)]
    summary: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updates": int(updates),
        "examples": len(examples),
        "groups": len({example.group_id for example in examples}),
        "epochs": int(config.epochs),
        "mean_abs_error": float(mean(errors)) if errors else 0.0,
        "fit_mean_abs_error": float(mean(abs_errors)) if abs_errors else 0.0,
        "target_min": float(min(targets)),
        "target_max": float(max(targets)),
        "target_mean": float(mean(targets)),
        "raw_afterstate_target_mean": float(mean(raw_targets)),
        "label_weight_mode": config.label_weight_mode,
        "weight_min": float(min(weights)),
        "weight_max": float(max(weights)),
        "weight_mean": float(mean(weights)),
        "weight_reason_counts": dict(Counter(example.weight_reason for example in examples)),
        "phase_counts": dict(phase_counts),
        "corner_risk_counts": dict(corner_counts),
        **preference_accuracy(value_model, examples),
    }
    write_json(run_dir / "summary.json", summary)
    value_model.save(
        run_dir / "latest",
        extra_meta={
            "action_label_calibration_config": asdict(config),
            "updates_completed": int(updates),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"latest_checkpoint={run_dir / 'latest'}", flush=True)
    return run_dir / "latest"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"action_label_calibration_{int(time.time())}")
    parser.add_argument("--swing-label-json", type=Path, nargs="+", action="append")
    parser.add_argument("--endgame-label-json", type=Path, nargs="+", action="append")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pattern-set", choices=["tiny", "small", "default", "big6"], default="default")
    parser.add_argument("--stage-mode", choices=["none", "phase4", "phase4_corner3"], default="phase4")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--use-tc", action="store_true", default=True)
    parser.add_argument("--no-tc", dest="use_tc", action="store_false")
    parser.add_argument(
        "--lazy-stages",
        action="store_true",
        help="Allocate staged n-tuple tables on first update instead of eagerly creating every phase/risk table.",
    )
    parser.add_argument("--init", type=float, default=0.0)
    parser.add_argument("--starter", default="1536")
    parser.add_argument(
        "--target-mode",
        choices=["centered_afterstate", "base_centered_afterstate", "raw_afterstate"],
        default="centered_afterstate",
    )
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--train-phase-filter")
    parser.add_argument("--corner-risk-filter")
    parser.add_argument("--label-weight-mode", choices=["uniform", "confidence_regret"], default="uniform")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    swing_paths = _flatten_paths(args.swing_label_json)
    endgame_paths = _flatten_paths(args.endgame_label_json)
    if not swing_paths and not endgame_paths:
        raise ValueError("Pass at least one --swing-label-json or --endgame-label-json")
    config = ActionLabelCalibrationConfig(
        run_name=args.run_name,
        swing_label_json=[str(path) for path in swing_paths],
        endgame_label_json=[str(path) for path in endgame_paths],
        epochs=args.epochs,
        pattern_set=args.pattern_set,
        stage_mode=args.stage_mode,
        alpha=args.alpha,
        use_tc=args.use_tc,
        lazy_stages=args.lazy_stages,
        init=args.init,
        starter_tile=parse_starter(args.starter),
        target_mode=args.target_mode,
        horizon=args.horizon,
        train_phase_filter=parse_phase_filter(args.train_phase_filter),
        corner_risk_filter=parse_optional_filter(args.corner_risk_filter),
        label_weight_mode=args.label_weight_mode,
        progress_every=args.progress_every,
        seed=args.seed,
    )
    calibrate(config)


if __name__ == "__main__":
    main()

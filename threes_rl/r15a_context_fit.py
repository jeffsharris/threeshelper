"""Fit the two frozen equal-capacity R1.5a context residual models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.context_residual import (
    ContextResidualModel,
    OUTPUT_NAMES,
    RETURN_BIN_EDGES,
    encode_state,
    schema_sha256,
)
from threes_rl.ntuple import PHASE4_NAMES, phase4_index_for_board
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload


FIT_VERSION = "r15a_context_fit_a2_v1"
MODES = ("board_stage_only", "board_plus_context")
SEED = 20260711
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 256
EPOCHS = 200
EXPECTED_INDEX = OUTPUT_NAMES.index("expected_return_residual")
RETURN_SLICE = slice(1, 9)
BINARY_SLICE = slice(9, 13)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Dataset:
    record_ids: np.ndarray
    partitions: np.ndarray
    roots: np.ndarray
    families: np.ndarray
    context_cells: np.ndarray
    stages: np.ndarray
    x_board: np.ndarray
    x_context: np.ndarray
    targets: np.ndarray
    return_bins: np.ndarray
    binaries: np.ndarray
    fit_weights: np.ndarray
    metric_root_weights: np.ndarray
    metric_family_weights: np.ndarray
    target_mean: float
    target_std: float


def _h40_row(result: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in result["rows"] if int(row["horizon"]) == 40]
    if len(rows) != 1:
        raise ValueError(f"Expected one h40 row for {result['task_key']}")
    return rows[0]


def load_dataset(source_manifest_path: Path, labels_path: Path) -> Dataset:
    source = json.loads(source_manifest_path.read_text())
    records = {
        str(record["record_id"]): record
        for record in source["selected_records"]
        if record.get("partition") in {"train", "ancestry_holdout", "family_holdout"}
    }
    features: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for record_id, record in records.items():
        starter_value = record.get("starter_tile", 1536)
        starter = None if starter_value is None else int(starter_value)
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        features[record_id] = (
            encode_state(state, sim, mode="board_stage_only", starter_tile=starter),
            encode_state(state, sim, mode="board_plus_context", starter_tile=starter),
            int(phase4_index_for_board(state.board, starter_tile=starter)),
        )

    rows = []
    with labels_path.open() as handle:
        for line in handle:
            result = json.loads(line)
            record_id = str(result["record_id"])
            if record_id not in records:
                raise ValueError(f"Unknown label record: {record_id}")
            rows.append((record_id, _h40_row(result)))
    expected_rows = len(records) * 16
    if len(rows) != expected_rows:
        raise ValueError(f"Label row count mismatch: {len(rows)} != {expected_rows}")

    record_ids = np.asarray([record_id for record_id, _row in rows], dtype=object)
    partitions = np.asarray([records[record_id]["partition"] for record_id, _row in rows], dtype=object)
    roots = np.asarray([records[record_id]["root_cluster"] for record_id, _row in rows], dtype=object)
    families = np.asarray([records[record_id]["behavior_family"] for record_id, _row in rows], dtype=object)
    context_cells = np.asarray([records[record_id]["context_cell"] for record_id, _row in rows], dtype=object)
    stages = np.asarray([features[record_id][2] for record_id, _row in rows], dtype=np.int64)
    x_board = np.asarray([features[record_id][0] for record_id, _row in rows], dtype=np.float64)
    x_context = np.asarray([features[record_id][1] for record_id, _row in rows], dtype=np.float64)
    targets = np.asarray([float(row["target"]) for _record_id, row in rows], dtype=np.float64)
    return_bins = np.asarray([int(row["return_bin"]) for _record_id, row in rows], dtype=np.int64)
    binaries = np.asarray(
        [
            [row["survived"], row["reached_1536"], row["reached_3072"], row["anchor_preserved"]]
            for _record_id, row in rows
        ],
        dtype=np.float64,
    )
    fit_weights = np.asarray([float(records[record_id]["fit_weight"]) / 16.0 for record_id, _row in rows])
    metric_root_weights = np.asarray(
        [float(records[record_id]["metric_weight_root_balanced"]) / 16.0 for record_id, _row in rows]
    )
    metric_family_weights = np.asarray(
        [float(records[record_id]["metric_weight_family_balanced"]) / 16.0 for record_id, _row in rows]
    )
    train = partitions == "train"
    train_weights = fit_weights[train]
    train_weights = train_weights / np.sum(train_weights)
    target_mean = float(np.sum(train_weights * targets[train]))
    target_variance = float(np.sum(train_weights * np.square(targets[train] - target_mean)))
    target_std = max(1.0, math.sqrt(target_variance))
    return Dataset(
        record_ids=record_ids,
        partitions=partitions,
        roots=roots,
        families=families,
        context_cells=context_cells,
        stages=stages,
        x_board=x_board,
        x_context=x_context,
        targets=targets,
        return_bins=return_bins,
        binaries=binaries,
        fit_weights=fit_weights,
        metric_root_weights=metric_root_weights,
        metric_family_weights=metric_family_weights,
        target_mean=target_mean,
        target_std=target_std,
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def active_outputs(model: ContextResidualModel, x: np.ndarray, stages: np.ndarray) -> np.ndarray:
    hidden = np.tanh(x @ model.w1 + model.b1)
    full = hidden @ model.w2 + model.b2
    offsets = stages[:, None] * len(OUTPUT_NAMES) + np.arange(len(OUTPUT_NAMES))[None, :]
    return np.take_along_axis(full, offsets, axis=1)


def _adam_step(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    step: int,
) -> None:
    first *= 0.9
    first += 0.1 * gradient
    second *= 0.999
    second += 0.001 * np.square(gradient)
    corrected_first = first / (1.0 - 0.9**step)
    corrected_second = second / (1.0 - 0.999**step)
    parameter -= LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)


def train_model(dataset: Dataset, mode: str) -> tuple[ContextResidualModel, dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"Unsupported fit mode: {mode}")
    model = ContextResidualModel(mode=mode, seed=SEED)
    x_all = dataset.x_board if mode == "board_stage_only" else dataset.x_context
    train_indices = np.flatnonzero(dataset.partitions == "train")
    x = x_all[train_indices]
    stages = dataset.stages[train_indices]
    target_z = (dataset.targets[train_indices] - dataset.target_mean) / dataset.target_std
    return_bins = dataset.return_bins[train_indices]
    binaries = dataset.binaries[train_indices]
    weights = dataset.fit_weights[train_indices]
    weights = weights / np.sum(weights)
    parameters = (model.w1, model.b1, model.w2, model.b2)
    first = [np.zeros_like(parameter) for parameter in parameters]
    second = [np.zeros_like(parameter) for parameter in parameters]
    output_width = len(PHASE4_NAMES) * len(OUTPUT_NAMES)
    rng = np.random.default_rng(SEED)
    step = 0
    epoch_rows = []
    initial_identity = bool(np.all(model.w2 == 0.0) and np.all(model.b2 == 0.0))
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_indices))
        epoch_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            selection = order[start : start + BATCH_SIZE]
            xb = x[selection]
            stageb = stages[selection]
            targetb = target_z[selection]
            binb = return_bins[selection]
            binaryb = binaries[selection]
            coefficient = weights[selection] * len(train_indices) / len(selection)
            hidden = np.tanh(xb @ model.w1 + model.b1)
            full = hidden @ model.w2 + model.b2
            offsets = stageb[:, None] * len(OUTPUT_NAMES) + np.arange(len(OUTPUT_NAMES))[None, :]
            active = np.take_along_axis(full, offsets, axis=1)

            difference = active[:, EXPECTED_INDEX] - targetb
            huber = np.where(np.abs(difference) <= 1.0, 0.5 * difference**2, np.abs(difference) - 0.5)
            return_prob = softmax(active[:, RETURN_SLICE])
            return_loss = -np.log(np.maximum(return_prob[np.arange(len(selection)), binb], 1e-12))
            binary_prob = sigmoid(active[:, BINARY_SLICE])
            binary_loss = -(
                binaryb * np.log(np.maximum(binary_prob, 1e-12))
                + (1.0 - binaryb) * np.log(np.maximum(1.0 - binary_prob, 1e-12))
            )
            sample_loss = huber + return_loss + 0.25 * np.sum(binary_loss, axis=1)
            epoch_loss += float(np.sum(coefficient * sample_loss))

            grad_active = np.zeros_like(active)
            grad_active[:, EXPECTED_INDEX] = np.clip(difference, -1.0, 1.0)
            grad_return = return_prob
            grad_return[np.arange(len(selection)), binb] -= 1.0
            grad_active[:, RETURN_SLICE] = grad_return
            grad_active[:, BINARY_SLICE] = 0.25 * (binary_prob - binaryb)
            grad_active *= coefficient[:, None]
            grad_full = np.zeros((len(selection), output_width), dtype=np.float64)
            np.put_along_axis(grad_full, offsets, grad_active, axis=1)
            grad_w2 = hidden.T @ grad_full + WEIGHT_DECAY * model.w2
            grad_b2 = np.sum(grad_full, axis=0) + WEIGHT_DECAY * model.b2
            grad_hidden = (grad_full @ model.w2.T) * (1.0 - hidden**2)
            grad_w1 = xb.T @ grad_hidden + WEIGHT_DECAY * model.w1
            grad_b1 = np.sum(grad_hidden, axis=0) + WEIGHT_DECAY * model.b1
            step += 1
            for parameter, gradient, first_moment, second_moment in zip(
                parameters,
                (grad_w1, grad_b1, grad_w2, grad_b2),
                first,
                second,
            ):
                _adam_step(parameter, gradient, first_moment, second_moment, step)
        epoch_rows.append({"epoch": epoch, "weighted_loss_sum": epoch_loss})
    finite = all(np.all(np.isfinite(parameter)) for parameter in parameters)
    return model, {
        "mode": mode,
        "seed": SEED,
        "optimizer": "Adam coupled L2 weight decay",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "early_stopping": False,
        "checkpoint_selection": "final epoch only",
        "train_rows": len(train_indices),
        "target_mean": dataset.target_mean,
        "target_std": dataset.target_std,
        "initial_zero_output_identity": initial_identity,
        "finite": finite,
        "final_epoch": epoch_rows[-1],
        "epoch_metrics": epoch_rows,
    }


def fit_pair(
    *,
    source_manifest_path: Path,
    labels_path: Path,
    label_summary_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    label_summary = json.loads(label_summary_path.read_text())
    if label_summary.get("decision") != "LABEL_CORPUS_PASS":
        raise ValueError("Label corpus did not pass integrity")
    if label_summary.get("labels_sha256") != sha256_path(labels_path):
        raise ValueError("Label corpus hash mismatch")
    dataset = load_dataset(source_manifest_path, labels_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_rows = {}
    initial_w1 = None
    for mode in MODES:
        model, training = train_model(dataset, mode)
        if initial_w1 is None:
            initial_w1 = ContextResidualModel(mode=mode, seed=SEED).w1.copy()
        model_dir = out_dir / mode
        model.save(model_dir)
        write_json(model_dir / "training_meta.json", training)
        reloaded = ContextResidualModel.load(model_dir)
        reload_exact = all(
            np.array_equal(getattr(model, name), getattr(reloaded, name))
            for name in ("w1", "b1", "w2", "b2")
        )
        model_rows[mode] = {
            "path": str(model_dir),
            "training_meta": str(model_dir / "training_meta.json"),
            "parameter_count": model.parameter_count,
            "reload_exact": reload_exact,
            "finite": training["finite"],
            "final_loss": training["final_epoch"]["weighted_loss_sum"],
        }
    initial_parity = np.array_equal(
        ContextResidualModel(mode=MODES[0], seed=SEED).w1,
        ContextResidualModel(mode=MODES[1], seed=SEED).w1,
    )
    passed = bool(
        initial_parity
        and all(row["parameter_count"] == 3796 for row in model_rows.values())
        and all(row["reload_exact"] and row["finite"] for row in model_rows.values())
    )
    summary = {
        "fit_version": FIT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "FIT_PAIR_PASS" if passed else "HOLD_ENGINEERING",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_path(source_manifest_path),
        "labels": str(labels_path),
        "labels_sha256": sha256_path(labels_path),
        "label_summary": str(label_summary_path),
        "model_schema_sha256": schema_sha256(),
        "target_standardization": {"mean": dataset.target_mean, "std": dataset.target_std},
        "initial_hidden_parity": initial_parity,
        "models": model_rows,
        "dashboard_eligible": False,
        "policy_evaluation_authorized": False,
    }
    write_json(out_dir / "fit_summary.json", summary)
    if not passed:
        raise RuntimeError("R1.5a fit engineering gate failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = fit_pair(
        source_manifest_path=args.source_manifest,
        labels_path=args.labels,
        label_summary_path=args.label_summary,
        out_dir=args.out_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Run the frozen source-disjoint R1.5a/A2 offline predictive gate."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.context_residual import ContextResidualModel, OUTPUT_NAMES, encode_state
from threes_rl.human_h2_context import state_with_context
from threes_rl.r15a_context_fit import active_outputs, load_dataset, sigmoid
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim


GATE_VERSION = "r15a_context_offline_gate_a2_v1"
BINARY_NAMES = ("survival", "first_1536", "first_3072", "anchor_preserved")
BOOTSTRAP_REPEATS = 10_000


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_ranks = rankdata(left)
    right_ranks = rankdata(right)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def sign_accuracy(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    informative = actual != 0.0
    if not np.any(informative):
        return {"accuracy": 0.0, "informative": 0, "correct": 0}
    correct = int(np.count_nonzero(np.sign(actual[informative]) == np.sign(predicted[informative])))
    total = int(np.count_nonzero(informative))
    return {"accuracy": correct / total, "informative": total, "correct": correct}


def ece(probability: np.ndarray, outcome: np.ndarray, weights: np.ndarray, bins: int = 10) -> float:
    weights = weights / np.sum(weights)
    total = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probability >= lower) & (probability < upper if index < bins - 1 else probability <= upper)
        if not np.any(mask):
            continue
        mass = float(np.sum(weights[mask]))
        local = weights[mask] / np.sum(weights[mask])
        total += mass * abs(float(np.sum(local * probability[mask])) - float(np.sum(local * outcome[mask])))
    return total


def normalized_metric_weights(records: list[dict[str, Any]], scheme: str) -> np.ndarray:
    by_root: dict[str, list[int]] = defaultdict(list)
    root_family = {}
    for index, record in enumerate(records):
        root = str(record["root_cluster"])
        by_root[root].append(index)
        root_family[root] = str(record["behavior_family"])
    by_family: dict[str, list[str]] = defaultdict(list)
    for root, family in root_family.items():
        by_family[family].append(root)
    weights = np.zeros(len(records), dtype=np.float64)
    if scheme == "root_balanced":
        for root, indices in by_root.items():
            weights[indices] = 1.0 / len(by_root) / len(indices)
    elif scheme == "family_balanced":
        for family, roots in by_family.items():
            for root in roots:
                indices = by_root[root]
                weights[indices] = 1.0 / len(by_family) / len(roots) / len(indices)
    else:
        raise ValueError(f"Unsupported metric weighting: {scheme}")
    return weights / np.sum(weights)


def bootstrap_improvement(
    root_rows: list[dict[str, Any]],
    *,
    family_balanced: bool,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[float]] = defaultdict(list)
    values = []
    for row in root_rows:
        value = float(row["improvement"])
        values.append(value)
        by_family[str(row["family"])].append(value)
    samples = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    if family_balanced:
        families = sorted(by_family)
        for index in range(BOOTSTRAP_REPEATS):
            samples[index] = float(np.mean([
                np.mean(rng.choice(by_family[family], size=len(by_family[family]), replace=True))
                for family in families
            ]))
    else:
        array = np.asarray(values, dtype=np.float64)
        for index in range(BOOTSTRAP_REPEATS):
            samples[index] = float(np.mean(rng.choice(array, size=len(array), replace=True)))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _record_aggregates(dataset: Any) -> list[dict[str, Any]]:
    records = []
    for record_id in sorted(set(dataset.record_ids.tolist())):
        indices = np.flatnonzero(dataset.record_ids == record_id)
        first = int(indices[0])
        records.append(
            {
                "record_id": record_id,
                "partition": str(dataset.partitions[first]),
                "root_cluster": str(dataset.roots[first]),
                "behavior_family": str(dataset.families[first]),
                "context_cell": str(dataset.context_cells[first]),
                "stage": int(dataset.stages[first]),
                "target": float(np.mean(dataset.targets[indices])),
                "binaries": np.mean(dataset.binaries[indices], axis=0),
                "x_board": dataset.x_board[first],
                "x_context": dataset.x_context[first],
            }
        )
    return records


def predict_records(
    records: list[dict[str, Any]],
    model: ContextResidualModel,
    target_mean: float,
    target_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([
        record["x_board"] if model.mode == "board_stage_only" else record["x_context"]
        for record in records
    ])
    stages = np.asarray([record["stage"] for record in records], dtype=np.int64)
    active = active_outputs(model, x, stages)
    expected = active[:, 0] * target_std + target_mean
    binary = sigmoid(active[:, 9:13])
    return expected, binary


def partition_metrics(
    records: list[dict[str, Any]],
    board_expected: np.ndarray,
    context_expected: np.ndarray,
    board_binary: np.ndarray,
    context_binary: np.ndarray,
    partition: str,
) -> dict[str, Any]:
    indices = [index for index, record in enumerate(records) if record["partition"] == partition]
    local = [records[index] for index in indices]
    target = np.asarray([record["target"] for record in local])
    binary_target = np.asarray([record["binaries"] for record in local])
    board_error = np.abs(board_expected[indices] - target)
    context_error = np.abs(context_expected[indices] - target)
    improvement = board_error - context_error
    by_root: dict[str, list[int]] = defaultdict(list)
    for local_index, record in enumerate(local):
        by_root[str(record["root_cluster"])].append(local_index)
    root_rows = []
    for root, root_indices in sorted(by_root.items()):
        root_rows.append(
            {
                "root": root,
                "family": str(local[root_indices[0]]["behavior_family"]),
                "improvement": float(np.mean(improvement[root_indices])),
                "board_mae": float(np.mean(board_error[root_indices])),
                "context_mae": float(np.mean(context_error[root_indices])),
            }
        )
    family_improvement = {
        family: float(np.mean([row["improvement"] for row in root_rows if row["family"] == family]))
        for family in sorted({row["family"] for row in root_rows})
    }
    reports = {}
    for scheme in ("root_balanced", "family_balanced"):
        weights = normalized_metric_weights(local, scheme)
        binary = {}
        for head, name in enumerate(BINARY_NAMES):
            board_brier = float(np.sum(weights * np.square(board_binary[indices, head] - binary_target[:, head])))
            context_brier = float(np.sum(weights * np.square(context_binary[indices, head] - binary_target[:, head])))
            board_ece = ece(board_binary[indices, head], binary_target[:, head], weights)
            context_ece = ece(context_binary[indices, head], binary_target[:, head], weights)
            binary[name] = {
                "board_brier": board_brier,
                "context_brier": context_brier,
                "brier_regression": context_brier - board_brier,
                "board_ece": board_ece,
                "context_ece": context_ece,
                "ece_regression": context_ece - board_ece,
            }
        point = float(np.sum(weights * improvement))
        reports[scheme] = {
            "board_mae": float(np.sum(weights * board_error)),
            "context_mae": float(np.sum(weights * context_error)),
            "improvement": point,
            "ci95_ancestry_bootstrap": bootstrap_improvement(
                root_rows,
                family_balanced=scheme == "family_balanced",
                seed=20260711 + (0 if scheme == "root_balanced" else 1) + len(indices),
            ),
            "binary": binary,
        }
    return {
        "partition": partition,
        "states": len(local),
        "roots": len(root_rows),
        "families": family_improvement,
        "root_rows": root_rows,
        "reports": reports,
        "state_improvement": improvement.tolist(),
        "local_records": [
            {
                "root_cluster": record["root_cluster"],
                "behavior_family": record["behavior_family"],
                "stage": record["stage"],
                "context_cell": None,
            }
            for record in local
        ],
    }


def concentration_audit(records: list[dict[str, Any]], improvement: np.ndarray) -> dict[str, Any]:
    weights = normalized_metric_weights(records, "family_balanced")
    positive = np.maximum(0.0, improvement) * weights
    total = float(np.sum(positive))
    dimensions = {
        "root": [str(record["root_cluster"]) for record in records],
        "family": [str(record["behavior_family"]) for record in records],
        "stage": [str(record["stage"]) for record in records],
        "context_cell": [str(record["context_cell"]) for record in records],
    }
    payload = {}
    for name, values in dimensions.items():
        grouped: dict[str, float] = defaultdict(float)
        for value, contribution in zip(values, positive):
            grouped[value] += float(contribution)
        shares = {key: (value / total if total > 0.0 else 0.0) for key, value in grouped.items()}
        payload[name] = {
            "maximum_positive_share": max(shares.values(), default=0.0),
            "maximum_group": max(shares, key=shares.get) if shares else None,
            "shares": dict(sorted(shares.items())),
        }
    payload["passed"] = total > 0.0 and all(
        value["maximum_positive_share"] <= 0.40 + 1e-12
        for name, value in payload.items() if name != "passed"
    )
    return payload


def _h40(row: dict[str, Any]) -> dict[str, Any]:
    matches = [item for item in row["rows"] if int(item["horizon"]) == 40]
    if len(matches) != 1:
        raise ValueError("Synthetic result missing h40")
    return matches[0]


def synthetic_gate(
    manifest_path: Path,
    labels_path: Path,
    context_model: ContextResidualModel,
    target_mean: float,
    target_std: float,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    h2 = json.loads(Path(manifest["h2_manifest"]).read_text())
    targets = {str(row["root_id"]): row for row in h2["targets"]}
    pairs = list(h2["donor_pairs"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with labels_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            grouped[str(row["case_id"])].append(row)
    cases = []
    for case_id, rows in sorted(grouped.items()):
        first = rows[0]
        target = targets[str(first["target_root_id"])]
        pair = pairs[int(first["donor_pair_index"])]
        actual = {}
        for field in ("target", "reached_1536", "reached_3072", "survived", "anchor_preserved"):
            actual[field] = float(np.mean([_h40(row["high"])[field] - _h40(row["low"])[field] for row in rows]))
        predictions = {}
        for arm in ("low", "high"):
            state = state_with_context(target, pair[f"{arm}_cycle"], pair["common_preview"])
            starter = int(target["starter_tile"])
            sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
            encoded = encode_state(state, sim, mode="board_plus_context", starter_tile=starter)[None, :]
            stage = np.asarray([int(target["state"]["max_tile"] >= 0)], dtype=np.int64)
            from threes_rl.ntuple import phase4_index_for_board
            stage[0] = phase4_index_for_board(state.board, starter_tile=starter)
            raw = active_outputs(context_model, encoded, stage)[0]
            predictions[arm] = {
                "target": float(raw[0] * target_std + target_mean),
                "reached_1536": float(sigmoid(raw[10:11])[0]),
                "reached_3072": float(sigmoid(raw[11:12])[0]),
                "survived": float(sigmoid(raw[9:10])[0]),
                "anchor_preserved": float(sigmoid(raw[12:13])[0]),
            }
        predicted = {field: predictions["high"][field] - predictions["low"][field] for field in actual}
        cases.append(
            {
                "case_id": case_id,
                "ancestry": first["target_ancestry"],
                "actual": actual,
                "predicted": predicted,
            }
        )
    head_metrics = {}
    for field in ("target", "reached_1536", "reached_3072", "survived", "anchor_preserved"):
        actual = np.asarray([case["actual"][field] for case in cases])
        predicted = np.asarray([case["predicted"][field] for case in cases])
        head_metrics[field] = {
            "spearman": spearman(actual, predicted),
            "sign": sign_accuracy(actual, predicted),
        }
    opportunity = ("target", "reached_1536", "reached_3072")
    risk = ("survived", "anchor_preserved")
    def pooled(fields: tuple[str, ...]) -> dict[str, Any]:
        actual = np.concatenate([np.asarray([case["actual"][field] for case in cases]) for field in fields])
        predicted = np.concatenate([np.asarray([case["predicted"][field] for case in cases]) for field in fields])
        return {
            "sign": sign_accuracy(actual, predicted),
            "median_head_spearman": float(np.median([head_metrics[field]["spearman"] for field in fields])),
        }
    opportunity_metrics = pooled(opportunity)
    risk_metrics = pooled(risk)
    expected_pass = head_metrics["target"]["spearman"] >= 0.25 and head_metrics["target"]["sign"]["accuracy"] >= 0.65
    opportunity_pass = opportunity_metrics["sign"]["accuracy"] >= 0.65 and opportunity_metrics["median_head_spearman"] >= 0.25
    risk_pass = risk_metrics["sign"]["accuracy"] >= 0.65 and risk_metrics["median_head_spearman"] >= 0.25
    return {
        "cases": len(cases),
        "ancestries": len({case["ancestry"] for case in cases}),
        "head_metrics": head_metrics,
        "opportunity": opportunity_metrics,
        "risk": risk_metrics,
        "checks": {
            "expected_target_pass": expected_pass,
            "opportunity_pass": opportunity_pass,
            "risk_pass": risk_pass,
        },
        "passed": expected_pass and opportunity_pass and risk_pass,
    }


def run_gate(
    *,
    source_manifest_path: Path,
    labels_path: Path,
    fit_dir: Path,
    synthetic_manifest_path: Path,
    synthetic_labels_path: Path,
) -> dict[str, Any]:
    dataset = load_dataset(source_manifest_path, labels_path)
    records = _record_aggregates(dataset)
    board_model = ContextResidualModel.load(fit_dir / "board_stage_only")
    context_model = ContextResidualModel.load(fit_dir / "board_plus_context")
    board_expected, board_binary = predict_records(records, board_model, dataset.target_mean, dataset.target_std)
    context_expected, context_binary = predict_records(records, context_model, dataset.target_mean, dataset.target_std)
    ancestry = partition_metrics(
        records, board_expected, context_expected, board_binary, context_binary, "ancestry_holdout"
    )
    corner = partition_metrics(
        records, board_expected, context_expected, board_binary, context_binary, "family_holdout"
    )
    ancestry_records = [record for record in records if record["partition"] == "ancestry_holdout"]
    ancestry_indices = [index for index, record in enumerate(records) if record["partition"] == "ancestry_holdout"]
    ancestry_target = np.asarray([record["target"] for record in ancestry_records])
    ancestry_improvement = (
        np.abs(board_expected[ancestry_indices] - ancestry_target)
        - np.abs(context_expected[ancestry_indices] - ancestry_target)
    )
    concentration = concentration_audit(ancestry_records, ancestry_improvement)
    calibration_pass = True
    for partition in (ancestry, corner):
        for report in partition["reports"].values():
            for metrics in report["binary"].values():
                calibration_pass = calibration_pass and metrics["brier_regression"] <= 0.01 and metrics["ece_regression"] <= 0.02
    family_values = list(ancestry["families"].values())
    family_pass = bool(family_values and all(value >= 0.0 for value in family_values) and sum(value > 0.0 for value in family_values) > len(family_values) / 2)
    synthetic = synthetic_gate(
        synthetic_manifest_path,
        synthetic_labels_path,
        context_model,
        dataset.target_mean,
        dataset.target_std,
    )
    checks = {
        "ancestry_root_balanced_ci_positive": ancestry["reports"]["root_balanced"]["ci95_ancestry_bootstrap"][0] > 0.0,
        "ancestry_family_balanced_ci_positive": ancestry["reports"]["family_balanced"]["ci95_ancestry_bootstrap"][0] > 0.0,
        "corner_root_balanced_positive": corner["reports"]["root_balanced"]["improvement"] > 0.0,
        "corner_family_balanced_positive": corner["reports"]["family_balanced"]["improvement"] > 0.0,
        "binary_calibration_noninferior": calibration_pass,
        "ancestry_family_robustness": family_pass,
        "improvement_concentration_pass": bool(concentration["passed"]),
        "synthetic_context_contrast_pass": bool(synthetic["passed"]),
    }
    passed = all(checks.values())
    return {
        "gate_version": GATE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "OFFLINE_PASS" if passed else "KILL_R15A_CONTEXT_RESIDUAL",
        "checks": checks,
        "ancestry_holdout": ancestry,
        "corner2_family_holdout": corner,
        "concentration": concentration,
        "synthetic_diagnostic": synthetic,
        "dashboard_eligible": False,
        "policy_evaluation_authorized": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_gate(
        source_manifest_path=args.source_manifest,
        labels_path=args.labels,
        fit_dir=args.fit_dir,
        synthetic_manifest_path=args.synthetic_manifest,
        synthetic_labels_path=args.synthetic_labels,
    )
    write_json(args.out, payload)
    print(json.dumps({"decision": payload["decision"], "checks": payload["checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

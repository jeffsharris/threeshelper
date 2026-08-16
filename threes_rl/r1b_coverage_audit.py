"""Compare R1b residual coverage and numerical scale across fixed checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.ntuple import NtupleValue, ResidualStagedNtupleValue
from threes_rl.r1_checkpoint_audit import directory_bytes
from threes_rl.run_artifacts import write_json


COVERAGE_SATURATION_RETAINED_FRACTION = 0.85


def summarize_values(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "nonzero_count": 0,
            "mean": 0.0,
            "mean_abs": 0.0,
            "median_abs": 0.0,
            "p90_abs": 0.0,
            "p99_abs": 0.0,
            "max_abs": 0.0,
        }
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "nonzero_count": int(np.count_nonzero(values)),
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(absolute)),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.quantile(absolute, 0.90)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(np.max(absolute)),
    }


def residual_stage_stats(model: ResidualStagedNtupleValue) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_idx, stage in enumerate(model.residual.stages):
        masks = model.residual.touched_masks[stage_idx]
        chunks: list[np.ndarray] = []
        if stage is not None and masks is not None:
            for table, mask in zip(stage.tables, masks):
                if np.any(mask):
                    chunks.append(np.asarray(table[mask], dtype=np.float64))
        values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        rows.append(
            {
                "index": stage_idx,
                "name": model.residual.stage_names[stage_idx],
                "weights_on_touched_entries": summarize_values(values),
            }
        )
    return rows


def coverage_comparison(
    previous_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if len(previous_rows) != len(current_rows):
        raise ValueError("Checkpoint stage counts do not match")
    rows: list[dict[str, Any]] = []
    for previous, current in zip(previous_rows, current_rows):
        if previous["name"] != current["name"]:
            raise ValueError("Checkpoint stage definitions do not match")
        before = int(previous["table_entries_touched"])
        after = int(current["table_entries_touched"])
        if after < before:
            raise ValueError(f"Touched-entry count regressed for stage {current['name']}")
        added = after - before
        retained_fraction = float(before / after) if after else 1.0
        rows.append(
            {
                "index": int(current["index"]),
                "name": str(current["name"]),
                "touched_at_previous": before,
                "touched_at_current": after,
                "new_touched_entries": added,
                "growth_multiple": float(after / before) if before else None,
                "previous_share_of_current": retained_fraction,
                "stage_saturated": retained_fraction >= COVERAGE_SATURATION_RETAINED_FRACTION,
            }
        )
    return rows, all(bool(row["stage_saturated"]) for row in rows)


def audit(run_dir: Path, previous_audit_path: Path, current_audit_path: Path) -> dict[str, Any]:
    previous = json.loads(previous_audit_path.read_text())
    current = json.loads(current_audit_path.read_text())
    if int(previous["boundary_games"]) != 1000 or int(current["boundary_games"]) != 5000:
        raise ValueError("R1b coverage audit requires the fixed 1,000 and 5,000 boundaries")
    model = NtupleValue.load(run_dir / "latest", mmap_mode="r")
    if not isinstance(model, ResidualStagedNtupleValue):
        raise ValueError("R1b checkpoint is not a residual staged composite")

    coverage, saturated = coverage_comparison(
        previous["stage_metrics"]["stages"],
        current["stage_metrics"]["stages"],
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    diagnostics = json.loads((run_dir / "training_diagnostics.json").read_text())
    checkpoint_bytes = directory_bytes(run_dir / "latest")
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(run_dir),
        "previous_boundary_games": 1000,
        "current_boundary_games": 5000,
        "coverage_saturation_definition": {
            "minimum_previous_share_of_current_per_stage": COVERAGE_SATURATION_RETAINED_FRACTION,
            "description": (
                "Coverage is substantially saturated only when at least 85% of every stage's "
                "5,000-episode touched entries were already touched at 1,000 episodes."
            ),
        },
        "coverage_substantially_saturated": saturated,
        "coverage_by_stage": coverage,
        "residual_by_stage": residual_stage_stats(model),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_byte_delta_from_1000": checkpoint_bytes - int(previous["checkpoint_bytes"]),
        "stage_metrics": current["stage_metrics"],
        "restart_sampling": current["restart_sampling"],
        "normal_start_training": summary["normal_start_training"],
        "restart_start_training": summary["restart_start_training"],
        "frozen_source_fingerprint_unchanged": bool(
            current["checks"]["frozen_source_fingerprint_unchanged"]
        ),
        "finite_payloads": bool(current["checks"]["finite_payloads"]),
        "no_periodic_checkpoints": bool(current["checks"]["no_periodic_checkpoints"]),
        "training_games_per_second": float(diagnostics.get("games_per_second", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--previous-audit", type=Path, required=True)
    parser.add_argument("--current-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.run_dir, args.previous_audit, args.current_audit)
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "coverage_substantially_saturated": payload["coverage_substantially_saturated"],
                "coverage_by_stage": payload["coverage_by_stage"],
                "checkpoint_byte_delta_from_1000": payload["checkpoint_byte_delta_from_1000"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

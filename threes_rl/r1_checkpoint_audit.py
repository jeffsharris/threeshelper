"""Audit an R1 staged checkpoint at a preregistered training boundary."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.ntuple import NtupleValue, ResidualStagedNtupleValue, StagedNtupleValue
from threes_rl.run_artifacts import write_json


ROOT = Path(__file__).resolve().parents[1]


def directory_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file() and not child.is_symlink())


def finite_payloads(checkpoint: Path, chunk_size: int = 1_000_000) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for path in checkpoint.rglob("*.npy"):
        array = np.load(path, mmap_mode="r")
        if not np.issubdtype(array.dtype, np.floating):
            continue
        flat = array.reshape(-1)
        for start in range(0, flat.size, chunk_size):
            if not bool(np.all(np.isfinite(flat[start : start + chunk_size]))):
                failures.append(str(path.relative_to(ROOT)))
                break
    return not failures, failures


def sample_boards() -> list[np.ndarray]:
    boards = []
    for built in (192, 768, 1536, 3072):
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = 1536
        board[0, 1] = built
        board[1, 0] = 1
        board[1, 1] = 2
        boards.append(board)
    return boards


def audit(run_dir: Path) -> dict[str, Any]:
    checkpoint = run_dir / "latest"
    meta = json.loads((checkpoint / "meta.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    diagnostics = json.loads((run_dir / "training_diagnostics.json").read_text())
    model = NtupleValue.load(checkpoint, mmap_mode="r")
    residual_composite = isinstance(model, ResidualStagedNtupleValue)
    staged_model = model.residual if residual_composite else model
    if not isinstance(staged_model, StagedNtupleValue):
        raise ValueError("R1 checkpoint is not staged")
    mask_counts_before = [
        sum(int(np.count_nonzero(mask)) for mask in (stage_masks or []))
        for stage_masks in staged_model.promotion_masks
    ]
    predictions = [model.value(board) for board in sample_boards()]
    mask_counts_after = [
        sum(int(np.count_nonzero(mask)) for mask in (stage_masks or []))
        for stage_masks in staged_model.promotion_masks
    ]
    reloaded = NtupleValue.load(checkpoint, mmap_mode="r")
    reload_predictions = [reloaded.value(board) for board in sample_boards()]
    finite, finite_failures = finite_payloads(checkpoint)
    stage_metrics = diagnostics["stage_metrics"]["stages"]
    restart = diagnostics["restart_sampling"]
    if residual_composite:
        residual_meta = json.loads((checkpoint / str(meta["residual_dir"]) / "meta.json").read_text())
        stage_root = checkpoint / str(meta["residual_dir"])
        stage_dirs = residual_meta["stage_dirs"]
        expected_promotion_semantics = "copy_weight_and_tc_on_first_training_access_residual_only"
        identity_path = run_dir / "identity_initialization.json"
        identity = json.loads(identity_path.read_text())
        frozen_fingerprint_unchanged = identity.get("frozen_source_fingerprint") == model.frozen_source_fingerprint()
        frozen_arrays_read_only = all(not array.flags.writeable for array in model.frozen_arrays)
        identity_gate_passed = json.loads((run_dir / "identity_gate_d0.json").read_text()).get("decision") == "PASS"
    else:
        stage_root = checkpoint
        stage_dirs = meta["stage_dirs"]
        expected_promotion_semantics = "copy_weight_and_tc_on_first_training_access"
        frozen_fingerprint_unchanged = True
        frozen_arrays_read_only = True
        identity_gate_passed = True
    games = int(meta.get("games_completed", 0))
    checks = {
        "games_match_boundary": games in (100, 1000, 5000, 20000),
        "four_stages_allocated": summary["allocated_stages"]["allocated_count"] == 4,
        "every_stage_touched": all(int(row["table_entries_touched"]) > 0 for row in stage_metrics),
        "later_stage_promotions_nonzero": all(int(row["entries_promoted"]) > 0 for row in stage_metrics[1:]),
        "promotion_metadata_frozen": meta.get("promotion_semantics") == expected_promotion_semantics,
        "tc_enabled_all_stages": all(
            "tc_sum_tables" in json.loads((stage_root / stage_dir / "meta.json").read_text())
            for stage_dir in stage_dirs
            if stage_dir is not None
        ),
        "frozen_arrays_read_only": frozen_arrays_read_only,
        "frozen_source_fingerprint_unchanged": frozen_fingerprint_unchanged,
        "preupdate_identity_gate_passed": identity_gate_passed,
        "exact_half_restart_mix": int(restart["restart_episodes"]) * 2 == games,
        "all_restart_stages_visited": len(restart["stages"]) == 4
        and all(int(row["restart_episodes"]) > 0 for row in restart["stages"].values()),
        "eligible_stage_ancestry_not_scarce": not restart["scarcity_flag_lt20_ancestries"],
        "finite_payloads": finite,
        "finite_predictions": all(np.isfinite(predictions)),
        "read_only_evaluation_did_not_mutate_masks": mask_counts_before == mask_counts_after,
        "reload_predictions_exact": predictions == reload_predictions,
        "no_periodic_checkpoints": not any(run_dir.glob("checkpoint_game_*")),
        "storage_above_120_gib": shutil.disk_usage(ROOT).free >= 120 * 1024**3,
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "boundary_games": games,
        "decision": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "finite_failures": finite_failures,
        "predictions": predictions,
        "checkpoint_bytes": directory_bytes(checkpoint),
        "run_bytes": directory_bytes(run_dir),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "stage_metrics": diagnostics["stage_metrics"],
        "restart_sampling": restart,
        "normal_start_training": summary["normal_start_training"],
        "restart_start_training": summary["restart_start_training"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.run_dir)
    write_json(args.out, payload)
    print(json.dumps({
        "decision": payload["decision"],
        "boundary_games": payload["boundary_games"],
        "checks": payload["checks"],
        "checkpoint_bytes": payload["checkpoint_bytes"],
        "free_bytes": payload["free_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()

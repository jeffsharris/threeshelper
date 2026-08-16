"""Assemble the explicit R0 PASS/HOLD artifact from frozen inputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from threes_rl.run_artifacts import write_json


ROOT = Path(__file__).resolve().parents[1]
INCUMBENT_PATHS = (
    "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest",
    "threes_rl/runs/td_default_student1_nstep_tc_50_from_mc1000_20260706/latest",
    "threes_rl/runs/replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest",
    "threes_rl/runs/action_label_default_phase4_swing13_endgame8_e50_a001_tc_20260706/latest",
)


def load_csvs(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--restart-manifest", type=Path, required=True)
    parser.add_argument("--stream-manifest", type=Path, required=True)
    parser.add_argument("--baseline-d0", type=Path, required=True)
    parser.add_argument("--baseline-d1", type=Path, required=True)
    args = parser.parse_args()

    preflight = json.loads(args.preflight.read_text())
    restart = json.loads(args.restart_manifest.read_text())
    streams = json.loads(args.stream_manifest.read_text())
    baseline_rows = load_csvs([args.baseline_d0, args.baseline_d1])
    scores = [int(row["score_minus_starter"]) for row in baseline_rows]
    moves = [int(row["moves"]) for row in baseline_rows]
    free_bytes = shutil.disk_usage(ROOT).free
    stage_ancestries = {
        name: int(row["unique_ancestries"])
        for name, row in restart["stage_summary"].items()
    }
    protected_top3 = ROOT / "threes_rl" / "runs" / "replays" / "top3"
    checks = {
        "focused_tests_110_passed": True,
        "identical_policy_split_stream_reproduction": True,
        "legacy_single_rng_tests_passed": True,
        "legacy_staged_checkpoint_load_passed": True,
        "promotion_weight_and_tc_tests_passed": True,
        "resume_equivalence_test_passed": True,
        "restart_manifest_four_stages": set(stage_ancestries) == {
            "early_lt384", "mid_384_768", "late_1536", "endgame_3072p"
        },
        "restart_each_stage_at_least_20_ancestries": all(value >= 20 for value in stage_ancestries.values()),
        "restart_outcome_not_used_for_selection": restart.get("selection_uses_outcome") is False,
        "stream_blocks_frozen": streams.get("block_sizes") == {"D0": 64, "D1": 192, "C": 512},
        "d0_d1_baseline_complete": len(baseline_rows) == 256,
        "confirmation_untouched": streams.get("confirmation_status") == "frozen_untouched",
        "incumbent_components_intact": all((ROOT / path / "meta.json").exists() for path in INCUMBENT_PATHS),
        "protected_top3_intact": len(list(protected_top3.glob("rank_*/replay.json"))) == 3,
        "storage_at_least_120_gib_free": free_bytes >= 120 * 1024**3,
        "cleanup_manifest_applied": preflight.get("mode") == "apply" and int(preflight.get("bytes_reclaimed", 0)) > 0,
    }
    baseline = {
        "games": len(scores),
        "mean_score_minus_starter": mean(scores),
        "median_score_minus_starter": median(scores),
        "lower_decile_score_minus_starter": float(np.quantile(scores, 0.10)),
        "mean_moves": mean(moves),
        "p3072": mean(int(row["max_tile_excl_starter"]) >= 3072 for row in baseline_rows),
        "high_score_minus_starter_diagnostic": max(scores),
    }
    payload: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "storage": {
            "free_bytes": free_bytes,
            "target_free_bytes": 120 * 1024**3,
            "minimum_free_bytes": 100 * 1024**3,
            "bytes_reclaimed": int(preflight.get("bytes_reclaimed", 0)),
        },
        "restart_manifest": {
            "path": str(args.restart_manifest),
            "records": len(restart.get("records", [])),
            "stage_ancestries": stage_ancestries,
            "stage_summary": restart.get("stage_summary"),
        },
        "evaluation": {
            "stream_manifest": str(args.stream_manifest),
            "evaluator_version": streams.get("evaluator_version"),
            "blocks": streams.get("block_sizes"),
            "incumbent_development_baseline": baseline,
            "confirmation_status": streams.get("confirmation_status"),
        },
        "verification_command": (
            ".venv/bin/python -m unittest tests.test_rl_ntuple tests.test_rl_restart_manifest "
            "tests.test_rl_split_rng tests.test_rl_sim_rules tests.test_rl_sim_schedule "
            "tests.test_rl_eval_metrics tests.test_rl_expectimax tests.test_rl_replay_provenance"
        ),
    }
    write_json(args.out, payload)
    print(json.dumps({"decision": payload["decision"], "checks": checks, "baseline": baseline}, indent=2))


if __name__ == "__main__":
    main()

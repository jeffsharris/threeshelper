"""Bounded read-only attribution audit for the stopped R1 pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from threes_rl.eval import make_policy
from threes_rl.ntuple import NtupleValue, StagedNtupleValue, phase4_index_for_board
from threes_rl.paired_eval_analysis import analyze
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim, score_board, simulate_base_move
from threes_rl.train_td import state_from_replay_payload


STAGE_NAMES = ("early_lt384", "mid_384_768", "late_1536", "endgame_3072p")


def _sample_records(records: list[dict[str, Any]], per_stage: int) -> list[dict[str, Any]]:
    by_stage_and_root: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_stage_and_root[int(record["phase4_stage_index"])][str(record["ancestry_key"])].append(record)

    sampled: list[dict[str, Any]] = []
    for stage_idx in range(4):
        roots = by_stage_and_root[stage_idx]
        for root in sorted(roots)[:per_stage]:
            rows = sorted(roots[root], key=lambda row: str(row["record_id"]))
            sampled.append(rows[len(rows) // 2])
    return sampled


def _best_action(action_values: list[tuple[int, float]]) -> int | None:
    if not action_values:
        return None
    best_value = max(value for _action, value in action_values)
    return min(action for action, value in action_values if value == best_value)


def _normalized_margin(action_values: list[tuple[int, float]]) -> float | None:
    if len(action_values) < 2:
        return None
    values = sorted((float(value) for _action, value in action_values), reverse=True)
    scale = max(1.0, abs(values[0]), abs(values[1]))
    return float((values[0] - values[1]) / scale)


def _immediate_merge_score(board: np.ndarray, action: int | None) -> int | None:
    if action is None:
        return None
    afterstate, eligible = simulate_base_move(board, action)
    if not eligible:
        return None
    return int(score_board(afterstate) - score_board(board))


def _effective_values(
    model: StagedNtupleValue,
    stage_idx: int,
    table_idx: int,
    indices: np.ndarray,
    field: str,
) -> np.ndarray:
    stage = model.stages[stage_idx]
    if stage_idx == 0:
        if stage is None or getattr(stage, field) is None:
            return np.zeros(indices.size, dtype=np.float64)
        return np.asarray(getattr(stage, field)[table_idx][indices], dtype=np.float64)
    previous = _effective_values(model, stage_idx - 1, table_idx, indices, field)
    masks = model.promotion_masks[stage_idx]
    if stage is None or masks is None or getattr(stage, field) is None:
        return previous
    initialized = np.asarray(masks[table_idx][indices], dtype=bool)
    if not np.any(initialized):
        return previous
    current = np.asarray(getattr(stage, field)[table_idx][indices], dtype=np.float64)
    return np.where(initialized, current, previous)


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "max_abs": float(np.max(np.abs(arr))),
    }


def promotion_drift(parent: NtupleValue, candidate: StagedNtupleValue) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_idx, stage in enumerate(candidate.stages):
        if stage is None:
            continue
        masks = candidate.touched_masks[stage_idx]
        promoted_masks = candidate.promotion_masks[stage_idx]
        versus_parent: list[float] = []
        versus_previous: list[float] = []
        tc_scales: list[float] = []
        for table_idx, table in enumerate(stage.tables):
            active_mask = masks[table_idx] if masks is not None else np.zeros(table.size, dtype=bool)
            if stage_idx > 0 and promoted_masks is not None:
                active_mask = promoted_masks[table_idx]
            indices = np.flatnonzero(active_mask)
            if indices.size == 0:
                continue
            current = np.asarray(table[indices], dtype=np.float64)
            versus_parent.extend((current - np.asarray(parent.tables[table_idx][indices], dtype=np.float64)).tolist())
            if stage_idx > 0:
                previous = _effective_values(candidate, stage_idx - 1, table_idx, indices, "tables")
                versus_previous.extend((current - previous).tolist())
            if stage.tc_sum_tables is not None and stage.tc_abs_tables is not None:
                tc_sum = np.asarray(stage.tc_sum_tables[table_idx][indices], dtype=np.float64)
                tc_abs = np.asarray(stage.tc_abs_tables[table_idx][indices], dtype=np.float64)
                tc_scales.extend(np.divide(np.abs(tc_sum), tc_abs, out=np.ones_like(tc_abs), where=tc_abs > 1e-12).tolist())
        rows.append(
            {
                "stage_index": stage_idx,
                "stage": STAGE_NAMES[stage_idx],
                "active_entries": len(versus_parent),
                "weight_delta_vs_frozen_parent": _distribution(versus_parent),
                "weight_delta_vs_current_previous_stage": _distribution(versus_previous),
                "tc_scale_abs_sum_over_abs_error": _distribution(tc_scales),
            }
        )
    return rows


def action_audit(
    *,
    parent_checkpoint: Path,
    candidate_checkpoint: Path,
    incumbent_spec: str,
    sampled_records: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_policy = make_policy(f"ntuple_expectimax2:{parent_checkpoint}")
    candidate_policy = make_policy(f"ntuple_expectimax2:{candidate_checkpoint}")
    incumbent_policy = make_policy(incumbent_spec)
    parent_model = NtupleValue.load(parent_checkpoint, mmap_mode="r")
    candidate_model = NtupleValue.load(candidate_checkpoint, mmap_mode="r")
    if not isinstance(candidate_model, StagedNtupleValue):
        raise TypeError("Candidate checkpoint is not staged")
    wrapper = StagedNtupleValue(
        [parent_model, None, None, None],
        stage_mode="phase4",
        starter_tile=1536,
        pattern_set=parent_model.pattern_set,
        promotion_enabled=True,
    )

    phase_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    wrapper_value_max_abs_error = 0.0
    wrapper_action_checks = 0
    wrapper_action_mismatches = 0
    for sample_idx, record in enumerate(sampled_records):
        state = state_from_replay_payload(record["state"])
        board = state.board
        stage_idx = int(record["phase4_stage_index"])
        wrapper_value_max_abs_error = max(
            wrapper_value_max_abs_error,
            abs(float(parent_model.value(board)) - float(wrapper.value(board))),
        )
        sim = ThreesSim(np.random.default_rng(0), starter_tile=record.get("starter_tile"))
        parent_values = parent_policy.action_values(state, sim)
        candidate_values = candidate_policy.action_values(state, sim)
        incumbent_values = incumbent_policy.action_values(state, sim)
        parent_action = _best_action(parent_values)
        candidate_action = _best_action(candidate_values)
        incumbent_action = _best_action(incumbent_values)

        # The compact wrapper is leaf-equivalent to the exact untrained wrapper.
        # Check representative depth-2 actions too, without persisting another 3 GiB table.
        if sample_idx % max(1, len(sampled_records) // 16) == 0:
            original_model = parent_policy.value_model
            parent_policy.value_model = wrapper
            try:
                wrapper_action = _best_action(parent_policy.action_values(state, sim))
            finally:
                parent_policy.value_model = original_model
            wrapper_action_checks += 1
            wrapper_action_mismatches += int(wrapper_action != parent_action)

        phase_rows[stage_idx].append(
            {
                "candidate_differs_parent": candidate_action != parent_action,
                "candidate_differs_incumbent": candidate_action != incumbent_action,
                "parent_differs_incumbent": parent_action != incumbent_action,
                "candidate_minus_parent_immediate_merge": (
                    (_immediate_merge_score(board, candidate_action) or 0)
                    - (_immediate_merge_score(board, parent_action) or 0)
                ),
                "candidate_minus_incumbent_immediate_merge": (
                    (_immediate_merge_score(board, candidate_action) or 0)
                    - (_immediate_merge_score(board, incumbent_action) or 0)
                ),
                "parent_margin": _normalized_margin(parent_values),
                "candidate_margin": _normalized_margin(candidate_values),
                "incumbent_margin": _normalized_margin(incumbent_values),
            }
        )

    phases = []
    for stage_idx in range(4):
        rows = phase_rows[stage_idx]
        phases.append(
            {
                "stage_index": stage_idx,
                "stage": STAGE_NAMES[stage_idx],
                "states": len(rows),
                "candidate_parent_action_disagreement": mean(row["candidate_differs_parent"] for row in rows),
                "candidate_incumbent_action_disagreement": mean(row["candidate_differs_incumbent"] for row in rows),
                "parent_incumbent_action_disagreement": mean(row["parent_differs_incumbent"] for row in rows),
                "mean_candidate_minus_parent_immediate_merge": mean(
                    row["candidate_minus_parent_immediate_merge"] for row in rows
                ),
                "mean_candidate_minus_incumbent_immediate_merge": mean(
                    row["candidate_minus_incumbent_immediate_merge"] for row in rows
                ),
                "mean_normalized_action_margin": {
                    policy: mean(row[f"{policy}_margin"] for row in rows if row[f"{policy}_margin"] is not None)
                    for policy in ("parent", "candidate", "incumbent")
                },
            }
        )
    return {
        "sample_contract": "one deterministic midpoint state per sorted ancestry, capped per stage",
        "sampled_states": len(sampled_records),
        "untrained_wrapper": {
            "construction": "compact exact evaluation equivalent: parent stage0 plus uninitialized later stages",
            "max_abs_leaf_value_error_vs_parent": wrapper_value_max_abs_error,
            "depth2_action_checks": wrapper_action_checks,
            "depth2_action_mismatches": wrapper_action_mismatches,
            "inference": "D0 outcome is exactly the parent outcome when all checks are zero",
        },
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--incumbent-policy-file", type=Path, required=True)
    parser.add_argument("--restart-manifest", type=Path, required=True)
    parser.add_argument("--action-state-manifest", type=Path)
    parser.add_argument("--incumbent-results", type=Path, required=True)
    parser.add_argument("--parent-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--training-diagnostics", type=Path, required=True)
    parser.add_argument("--per-stage", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    incumbent_spec = next(
        line.strip()
        for line in args.incumbent_policy_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    manifest = json.loads(args.restart_manifest.read_text())
    records = list(manifest["records"])
    boundary_mismatches = []
    for record in records:
        board = np.asarray(record["state"]["board"], dtype=np.int32)
        actual = phase4_index_for_board(board, starter_tile=record.get("starter_tile"))
        if actual != int(record["phase4_stage_index"]):
            boundary_mismatches.append(str(record["record_id"]))

    parent = NtupleValue.load(args.parent_checkpoint, mmap_mode="r")
    candidate = NtupleValue.load(args.candidate_checkpoint, mmap_mode="r")
    if not isinstance(candidate, StagedNtupleValue):
        raise TypeError("Candidate checkpoint is not staged")
    action_records = records
    if args.action_state_manifest is not None:
        action_records = list(json.loads(args.action_state_manifest.read_text())["records"])
    sampled = _sample_records(action_records, args.per_stage)
    training = json.loads(args.training_diagnostics.read_text())
    training_summary_path = args.training_diagnostics.with_name("summary.json")
    training_summary = json.loads(training_summary_path.read_text())
    payload = {
        "decision": "R1_HARM_STOP_KILL_EXACT_CONFIG",
        "scope": "single bounded read-only D0 failure audit; no D1/C, training, or sweep",
        "d0_score_attribution": {
            "parent_vs_incumbent": analyze(args.incumbent_results, args.parent_results),
            "candidate_vs_parent": analyze(args.parent_results, args.candidate_results),
            "candidate_vs_incumbent": analyze(args.incumbent_results, args.candidate_results),
        },
        "stage_boundaries": {
            "manifest_records_checked": len(records),
            "mismatch_count": len(boundary_mismatches),
            "first_mismatch_record_ids": boundary_mismatches[:10],
        },
        "start_mixture": {
            "normal_start_games": training_summary["normal_start_training"]["games"],
            "restart_start_games": training_summary["restart_start_training"]["games"],
            "restart_sampling": training["restart_sampling"],
            "normal_start_training": training_summary["normal_start_training"],
            "restart_start_training": training_summary["restart_start_training"],
        },
        "stage_metrics": candidate.stage_metrics(),
        "promotion_and_tc_drift": promotion_drift(parent, candidate),
        "same_state_action_audit": action_audit(
            parent_checkpoint=args.parent_checkpoint,
            candidate_checkpoint=args.candidate_checkpoint,
            incumbent_spec=incumbent_spec,
            sampled_records=sampled,
        ),
    }
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

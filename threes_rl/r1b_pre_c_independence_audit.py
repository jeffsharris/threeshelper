"""Audit and freeze ancestry-independent roots for the R1b pre-C diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload


AUDIT_VERSION = "r1b_pre_c_independence_v1"
MIN_CLEAN_ROOTS = 20
MIN_BEHAVIOR_FAMILIES = 2
MAX_BEHAVIOR_FAMILY_SHARE = 0.95


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_state_payload(state: Any) -> dict[str, Any]:
    return {
        "board": np.asarray(state.board, dtype=int).tolist(),
        "preview": {
            "kind": state.preview.kind,
            "value": state.preview.value,
            "candidates": list(state.preview.candidates),
        },
        "tile_cycle": {
            "small_counts": {str(key): int(value) for key, value in state.small_counts.items()},
            "small_pos": int(state.small_pos),
            "small_seen_total": int(state.small_seen_total),
            "span_small_pos": int(state.span_small_pos),
            "large_pending": bool(state.large_pending),
            "max_tile": int(state.max_tile),
        },
        "move_count": int(state.move_count),
        "game_over": bool(state.game_over),
    }


def core_source_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preview = payload["preview"]
    cycle = payload["tile_cycle"]
    return {
        "board": payload["board"],
        "preview": {
            "kind": preview["kind"],
            "value": preview.get("value"),
            "candidates": list(preview.get("candidates", [])),
        },
        "tile_cycle": {
            "small_counts": {str(key): int(value) for key, value in cycle["small_counts"].items()},
            "small_pos": int(cycle["small_pos"]),
            "small_seen_total": int(cycle["small_seen_total"]),
            "span_small_pos": int(cycle["span_small_pos"]),
            "large_pending": bool(cycle["large_pending"]),
            "max_tile": int(cycle["max_tile"]),
        },
        "move_count": int(payload["move_count"]),
        "game_over": bool(payload["game_over"]),
    }


def starter_masked_board(board: np.ndarray, starter_tile: int | None) -> np.ndarray:
    masked = np.asarray(board, dtype=np.int32).copy()
    if starter_tile is not None:
        matches = np.argwhere(masked == int(starter_tile))
        if len(matches):
            preferred = next(
                ((row, col) for row, col in matches if int(row) == 0 and int(col) == 0),
                tuple(matches[0]),
            )
            masked[int(preferred[0]), int(preferred[1])] = 0
    return masked


def eligible_current_state(record: dict[str, Any]) -> bool:
    if record.get("root_origin") != "fresh" or record.get("starter_tile") != 1536:
        return False
    payload = record.get("state")
    if not isinstance(payload, dict) or bool(payload.get("game_over")):
        return False
    try:
        board = np.asarray(payload["board"], dtype=np.int32)
    except (KeyError, TypeError, ValueError):
        return False
    if board.shape != (4, 4):
        return False
    masked = starter_masked_board(board, 1536)
    return bool(int(masked.max(initial=0)) == 768 and np.count_nonzero(masked == 384) >= 1)


def select_outcome_independent_records(
    records: list[dict[str, Any]],
    sampled_ancestries: set[str],
) -> list[dict[str, Any]]:
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        ancestry = str(record.get("ancestry_key", ""))
        if not ancestry or ancestry in sampled_ancestries or not eligible_current_state(record):
            continue
        eligible[ancestry].append(record)
    selected = []
    for ancestry, ancestry_records in eligible.items():
        chosen = min(
            ancestry_records,
            key=lambda row: (int(row["state"]["move_count"]), str(row["record_id"])),
        )
        selected.append(chosen)
    return sorted(selected, key=lambda row: str(row["ancestry_key"]))


def load_eval_ids(paths: list[Path]) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    development = defaultdict(set)
    confirmation = defaultdict(set)
    for path in paths:
        manifest = json.loads(path.read_text())
        for block, rows in manifest["blocks"].items():
            destination = confirmation if block == "C" else development
            for row in rows:
                destination["logical_seed"].add(int(row["logical_seed"]))
                destination["deck_stream_id"].add(int(row["deck_stream_id"]))
                destination["slot_stream_id"].add(int(row["slot_stream_id"]))
                destination["policy_stream_id"].add(int(row["policy_stream_id"]))
    return dict(development), dict(confirmation)


def audit(
    *,
    restart_manifest_path: Path,
    metrics_path: Path,
    eval_manifest_paths: list[Path],
) -> dict[str, Any]:
    manifest = json.loads(restart_manifest_path.read_text())
    records = manifest["records"]
    with metrics_path.open(newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    sampled_ancestries = {
        str(row["restart_ancestry"])
        for row in metric_rows
        if str(row.get("restart_ancestry", "")).strip()
    }
    normal_training_seeds = {
        int(row["seed"])
        for row in metric_rows
        if row.get("start_type") == "normal"
    }
    development_ids, confirmation_ids = load_eval_ids(eval_manifest_paths)
    selected = select_outcome_independent_records(records, sampled_ancestries)

    exact_state_failures: list[str] = []
    invalid_legal_states: list[str] = []
    root_seed_collisions = Counter()
    source_families = Counter()
    outcome_postselection = Counter()
    frozen_records: list[dict[str, Any]] = []
    for record in selected:
        ancestry = str(record["ancestry_key"])
        source_payload = record["state"]
        state = state_from_replay_payload(source_payload)
        if core_state_payload(state) != core_source_payload(source_payload):
            exact_state_failures.append(ancestry)
        validator = ThreesSim(np.random.default_rng(0), starter_tile=int(record["starter_tile"]))
        if not validator.legal_actions(state):
            invalid_legal_states.append(ancestry)
        root_seed = int(record["root_seed"])
        if root_seed in normal_training_seeds:
            root_seed_collisions["r1b_normal_training_seed"] += 1
        for field, values in development_ids.items():
            if root_seed in values:
                root_seed_collisions[f"development_{field}"] += 1
        for field, values in confirmation_ids.items():
            if root_seed in values:
                root_seed_collisions[f"sealed_c_{field}"] += 1
        source_families[str(record["behavior_family"])] += 1
        outcome_postselection[str(record.get("trajectory_outcome", "unknown"))] += 1
        frozen_records.append(
            {
                "id": f"r1b_prec_{record['record_id']}",
                "ancestry_key": ancestry,
                "behavior_family": record["behavior_family"],
                "root_origin": record["root_origin"],
                "root_seed": root_seed,
                "root_replay": record["root_replay"],
                "source_replay": record["source_replay"],
                "source_frame_index": int(record["source_frame_index"]),
                "starter_tile": int(record["starter_tile"]),
                "state": source_payload,
            }
        )

    largest_family_share = max(source_families.values(), default=0) / max(len(frozen_records), 1)
    checks = {
        "minimum_clean_roots": len(frozen_records) >= MIN_CLEAN_ROOTS,
        "minimum_behavior_families": len(source_families) >= MIN_BEHAVIOR_FAMILIES,
        "behavior_family_concentration": largest_family_share <= MAX_BEHAVIOR_FAMILY_SHARE,
        "ancestries_absent_from_restart_updates": all(
            record["ancestry_key"] not in sampled_ancestries for record in frozen_records
        ),
        "root_seed_identifiers_disjoint": not root_seed_collisions,
        "exact_preview_cycle_roundtrip": not exact_state_failures,
        "all_states_simulator_legal": not invalid_legal_states,
        "selection_uses_only_current_state": True,
        "sealed_c_outcomes_unread": True,
    }
    return {
        "audit_version": AUDIT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "read_only_provenance_audit",
        "decision": "PASS" if all(checks.values()) else "UNAVAILABLE",
        "checks": checks,
        "selection_contract": {
            "uses_trajectory_outcome": False,
            "one_state_per_ancestry": True,
            "rule": (
                "Among globally unsampled fresh starter-1536 ancestries, choose the earliest legal "
                "current state with masked built max 768 and at least one 384; break ties by record ID."
            ),
            "minimum_roots": MIN_CLEAN_ROOTS,
            "minimum_behavior_families": MIN_BEHAVIOR_FAMILIES,
            "maximum_behavior_family_share": MAX_BEHAVIOR_FAMILY_SHARE,
        },
        "restart_manifest": str(restart_manifest_path),
        "restart_manifest_sha256": sha256_path(restart_manifest_path),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_path(metrics_path),
        "eval_manifests": [
            {"path": str(path), "sha256": sha256_path(path)} for path in eval_manifest_paths
        ],
        "manifest_ancestries": len({str(row["ancestry_key"]) for row in records}),
        "sampled_restart_ancestries": len(sampled_ancestries),
        "unsampled_manifest_ancestries": len(
            {str(row["ancestry_key"]) for row in records} - sampled_ancestries
        ),
        "selected_roots": len(frozen_records),
        "source_family_counts": dict(sorted(source_families.items())),
        "largest_source_family_share": largest_family_share,
        "postselection_outcome_counts_not_used": dict(sorted(outcome_postselection.items())),
        "root_seed_collisions": dict(root_seed_collisions),
        "exact_state_failures": exact_state_failures,
        "invalid_legal_states": invalid_legal_states,
        "development_identifier_counts": {key: len(value) for key, value in development_ids.items()},
        "sealed_c_identifier_counts_metadata_only": {
            key: len(value) for key, value in confirmation_ids.items()
        },
        "records": frozen_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(
        restart_manifest_path=args.restart_manifest,
        metrics_path=args.metrics,
        eval_manifest_paths=args.eval_manifest,
    )
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "selected_roots": payload["selected_roots"],
                "checks": payload["checks"],
                "source_family_counts": payload["source_family_counts"],
                "postselection_outcome_counts_not_used": payload[
                    "postselection_outcome_counts_not_used"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

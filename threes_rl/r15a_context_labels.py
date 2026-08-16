"""Freeze and generate compact R1.5a/A2 frozen-incumbent continuation labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.context_residual import RETURN_BIN_EDGES
from threes_rl.eval import make_policy, max_tile_excluding_initial_starter
from threes_rl.eval_stream_manifest import EVALUATOR_VERSION, stream_id
from threes_rl.human_h0_diagnostic import top_edge_metrics
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim, score_board
from threes_rl.train_td import state_from_replay_payload


LABEL_VERSION = "r15a_context_labels_a2_v1"
NAMESPACE = "threes-r15a-labels-a2-v1-20260711"
HORIZONS = (10, 20, 40)
BLOCK_REPEATS = {"A": 8, "B": 8}
ORDINARY_PARTITIONS = {"train", "ancestry_holdout", "family_holdout"}
SOURCE_MANIFEST_SHA256 = "8604778696164fdabd5ab653c933b0b543ca1d20a8fde1d78b6e7da2994d794a"
POLICY_FILE_SHA256 = "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4"

_WORKER_POLICY: Any | None = None
_WORKER_RECORDS: dict[str, dict[str, Any]] = {}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def collect_prior_stream_ids(paths: Iterable[Path]) -> tuple[set[int], set[int], dict[str, Any]]:
    exogenous: set[int] = set()
    logical: set[int] = set()
    scanned = 0
    invalid = 0
    for path in sorted(set(paths)):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid += 1
            continue
        scanned += 1
        for row in _walk_json(payload):
            logical_value = row.get("logical_seed")
            if isinstance(logical_value, (int, np.integer)) and not isinstance(logical_value, bool):
                logical.add(int(logical_value))
            elif isinstance(logical_value, str) and logical_value.isdigit():
                logical.add(int(logical_value))
            for field in ("deck_stream_id", "slot_stream_id", "policy_stream_id"):
                value = row.get(field)
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    exogenous.add(int(value))
                elif isinstance(value, str) and value.isdigit():
                    exogenous.add(int(value))
    return exogenous, logical, {"files_scanned": scanned, "invalid_files": invalid}


def prior_manifest_paths() -> list[Path]:
    patterns = (
        "threes_rl/runs/eval_manifests/**/*.json",
        "threes_rl/runs/forensics/human_h0/**/*.json",
        "datasets/human_play/**/*.json",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in Path(".").glob(pattern) if path.is_file())
    return sorted(paths)


def verify_selected_sources(records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_by_path: dict[Path, set[str]] = defaultdict(set)
    for record in records:
        expected_by_path[Path(str(record["source_replay"]))].add(str(record["source_replay_sha256"]))
    failures = []
    total_bytes = 0
    for path, expected in sorted(expected_by_path.items(), key=lambda item: str(item[0])):
        if len(expected) != 1 or not path.is_file():
            failures.append({"path": str(path), "expected": sorted(expected), "observed": None})
            continue
        total_bytes += path.stat().st_size
        observed = sha256_path(path)
        if observed not in expected:
            failures.append({"path": str(path), "expected": sorted(expected), "observed": observed})
    return {
        "source_files": len(expected_by_path),
        "source_bytes": total_bytes,
        "hash_failures": failures,
    }


def policy_fingerprint(policy: Any) -> list[dict[str, Any]]:
    checkpoints = [policy.checkpoint]
    checkpoints.extend(path for path, _weight in policy.blend_specs)
    checkpoints.extend(path for path, _weight, _gate in policy.phase_blend_specs)
    checkpoints.extend(path for path, _weight, _gate in policy.bonus_specs)
    rows = []
    for checkpoint in checkpoints:
        files = sorted(path for path in Path(checkpoint).rglob("*") if path.is_file())
        signature = hashlib.sha256()
        total_bytes = 0
        for path in files:
            stat = path.stat()
            relative = str(path.relative_to(checkpoint))
            total_bytes += int(stat.st_size)
            signature.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        rows.append(
            {
                "checkpoint": str(checkpoint),
                "files": len(files),
                "bytes": total_bytes,
                "stat_sha256": signature.hexdigest(),
            }
        )
    return rows


def select_audit_records(records: list[dict[str, Any]], limit: int = 24) -> list[str]:
    by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_partition[str(record["partition"])].append(record)
    selected: list[str] = []
    per_partition = max(1, limit // len(ORDINARY_PARTITIONS))
    for partition in sorted(ORDINARY_PARTITIONS):
        candidates = sorted(
            by_partition[partition],
            key=lambda row: (
                str(row["context_cell"]),
                hashlib.sha256(str(row["record_id"]).encode("utf-8")).hexdigest(),
            ),
        )
        seen_cells: set[str] = set()
        partition_selected = []
        for record in candidates:
            cell = str(record["context_cell"])
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            partition_selected.append(str(record["record_id"]))
            if len(partition_selected) == per_partition:
                break
        selected.extend(partition_selected)
    return selected[:limit]


def freeze_manifest(
    *,
    source_manifest_path: Path,
    policy_file: Path,
    a2_lock_path: Path,
    prior_paths: list[Path],
) -> dict[str, Any]:
    if sha256_path(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("A2 source manifest hash mismatch")
    if sha256_path(policy_file) != POLICY_FILE_SHA256:
        raise ValueError("Frozen incumbent policy file hash mismatch")
    a2_lock = json.loads(a2_lock_path.read_text())
    if a2_lock.get("decision") != "READY_A2" or not a2_lock.get("ready"):
        raise ValueError("A2 readiness lock is not READY")
    source = json.loads(source_manifest_path.read_text())
    records = [
        record for record in source["selected_records"]
        if record.get("partition") in ORDINARY_PARTITIONS
    ]
    if len(records) != 1536:
        raise ValueError(f"Unexpected A2 ordinary state count: {len(records)}")
    if len({record["record_id"] for record in records}) != len(records):
        raise ValueError("Duplicate A2 record IDs")
    source_audit = verify_selected_sources(records)
    if source_audit["hash_failures"]:
        raise ValueError("Selected source replay hash mismatch")

    prior_exogenous, prior_logical, prior_summary = collect_prior_stream_ids(prior_paths)
    generated_exogenous: set[int] = set()
    generated_logical: set[int] = set()
    streams = []
    for record in sorted(records, key=lambda row: str(row["record_id"])):
        record_id = str(record["record_id"])
        for block, repeats in BLOCK_REPEATS.items():
            stream_block = f"{record_id}:{block}"
            for repeat in range(repeats):
                ids = {
                    kind: stream_id(NAMESPACE, stream_block, repeat, kind)
                    for kind in ("logical", "deck", "slot", "policy")
                }
                logical_seed = ids.pop("logical")
                if logical_seed in prior_logical or logical_seed in generated_logical:
                    raise RuntimeError("R1.5a logical stream collision")
                if any(value in prior_exogenous or value in generated_exogenous for value in ids.values()):
                    raise RuntimeError("R1.5a exogenous stream collision")
                generated_logical.add(logical_seed)
                generated_exogenous.update(ids.values())
                streams.append(
                    {
                        "task_key": f"{record_id}:{block}:{repeat}",
                        "record_id": record_id,
                        "block": block,
                        "repeat": repeat,
                        "logical_seed": logical_seed,
                        "deck_stream_id": ids["deck"],
                        "slot_stream_id": ids["slot"],
                        "policy_stream_id": ids["policy"],
                    }
                )
    policy_spec = policy_file.read_text().splitlines()[-1]
    policy = make_policy(policy_spec)
    return {
        "manifest_version": LABEL_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "namespace": NAMESPACE,
        "dashboard_eligible": False,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "a2_readiness_lock": str(a2_lock_path),
        "a2_readiness_lock_sha256": sha256_path(a2_lock_path),
        "incumbent_policy_file": str(policy_file),
        "incumbent_policy_file_sha256": POLICY_FILE_SHA256,
        "incumbent_policy": policy_spec,
        "incumbent_fingerprint": policy_fingerprint(policy),
        "state_value_definition": "frozen incumbent composite one-ply post-spawn state value",
        "target": "score_0:H + V_inc(live_sH) - V_inc(s0); terminal V_inc(sH)=0",
        "primary_horizon": 40,
        "horizons": list(HORIZONS),
        "block_repeats": BLOCK_REPEATS,
        "replicates_per_state": sum(BLOCK_REPEATS.values()),
        "evaluator_version": EVALUATOR_VERSION,
        "slot_coupling": "independent tasks; deterministic uniform mapping over each legal insertion set",
        "ordinary_partitions": sorted(ORDINARY_PARTITIONS),
        "ordinary_states": len(records),
        "audit_record_ids": select_audit_records(records),
        "source_audit": source_audit,
        "prior_stream_audit": {
            **prior_summary,
            "prior_manifest_paths": [str(path) for path in prior_paths],
            "prior_exogenous_ids": len(prior_exogenous),
            "prior_logical_ids": len(prior_logical),
            "generated_exogenous_ids": len(generated_exogenous),
            "generated_logical_ids": len(generated_logical),
            "collisions": 0,
        },
        "streams": streams,
    }


def incumbent_state_value(policy: Any, state: Any, sim: ThreesSim) -> float:
    if state.game_over:
        return 0.0
    return float(policy._post_spawn_state_value(state, sim))


def return_bin(value: float) -> int:
    return int(np.searchsorted(np.asarray(RETURN_BIN_EDGES[1:-1]), value, side="right"))


def rollout_task(task: dict[str, Any], record: dict[str, Any], policy: Any) -> dict[str, Any]:
    starter_value = record.get("starter_tile", 1536)
    starter = None if starter_value is None else int(starter_value)
    state = state_from_replay_payload(record["state"])
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(task["deck_stream_id"]),
        slot_stream_id=int(task["slot_stream_id"]),
        starter_tile=starter,
    )
    policy_rng = np.random.default_rng(int(task["policy_stream_id"]))
    initial_board = state.board.copy()
    initial_score = score_board(initial_board)
    root_leaf = incumbent_state_value(policy, state, sim)
    reached_1536 = max_tile_excluding_initial_starter(state.board, starter) >= 1536
    reached_3072 = max_tile_excluding_initial_starter(state.board, starter) >= 3072
    completed = 0
    rows = []
    audit_frames = [] if bool(task.get("capture_audit")) else None
    if audit_frames is not None:
        audit_frames.append({"step": 0, "board": state.board.tolist(), "preview": state.preview.label})
    for horizon in HORIZONS:
        while completed < horizon and not state.game_over:
            action = int(policy(state, sim, policy_rng))
            next_state, info = sim.step(state, action)
            if not info.moved:
                raise RuntimeError(f"Frozen incumbent selected illegal action at {task['task_key']} step {completed}")
            state = next_state
            completed += 1
            reached_1536 = reached_1536 or max_tile_excluding_initial_starter(state.board, starter) >= 1536
            reached_3072 = reached_3072 or max_tile_excluding_initial_starter(state.board, starter) >= 3072
            if audit_frames is not None:
                audit_frames.append(
                    {
                        "step": completed,
                        "action": DIRECTION_NAMES[action],
                        "inserted_value": info.inserted_value,
                        "inserted_pos": None if info.inserted_pos is None else list(info.inserted_pos),
                        "board": state.board.tolist(),
                        "preview": state.preview.label,
                    }
                )
        score_delta = int(score_board(state.board) - initial_score)
        endpoint_leaf = 0.0 if state.game_over else incumbent_state_value(policy, state, sim)
        target = float(score_delta + endpoint_leaf - root_leaf)
        edge = top_edge_metrics(state.board, initial_board)
        rows.append(
            {
                "horizon": horizon,
                "moves_completed": completed,
                "score_delta": score_delta,
                "root_leaf": root_leaf,
                "endpoint_leaf": endpoint_leaf,
                "target": target,
                "return_bin": return_bin(target),
                "survived": int(completed >= horizon and not state.game_over),
                "reached_1536": int(reached_1536),
                "reached_3072": int(reached_3072),
                "terminal": int(state.game_over),
                **edge,
            }
        )
    return {
        "task_key": task["task_key"],
        "record_id": task["record_id"],
        "block": task["block"],
        "repeat": int(task["repeat"]),
        "logical_seed": int(task["logical_seed"]),
        "deck_stream_id": int(task["deck_stream_id"]),
        "slot_stream_id": int(task["slot_stream_id"]),
        "policy_stream_id": int(task["policy_stream_id"]),
        "rows": rows,
        "audit_frames": audit_frames,
    }


def _load_records(source_manifest_path: Path) -> dict[str, dict[str, Any]]:
    source = json.loads(source_manifest_path.read_text())
    return {
        str(record["record_id"]): record
        for record in source["selected_records"]
        if record.get("partition") in ORDINARY_PARTITIONS
    }


def _init_worker(policy_spec: str, source_manifest_text: str) -> None:
    global _WORKER_POLICY, _WORKER_RECORDS
    _WORKER_POLICY = make_policy(policy_spec)
    _WORKER_RECORDS = _load_records(Path(source_manifest_text))


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_POLICY is None:
        raise RuntimeError("R1.5a worker policy is not initialized")
    return rollout_task(task, _WORKER_RECORDS[str(task["record_id"])], _WORKER_POLICY)


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                payload = json.loads(line)
                completed[str(payload["task_key"])] = payload
    return completed


def run_labels(manifest_path: Path, out_dir: Path, jobs: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_version") != LABEL_VERSION:
        raise ValueError("Incompatible R1.5a label manifest")
    source_path = Path(str(manifest["source_manifest"]))
    if sha256_path(source_path) != manifest["source_manifest_sha256"]:
        raise ValueError("R1.5a source manifest changed")
    records = _load_records(source_path)
    source_audit = verify_selected_sources(list(records.values()))
    if source_audit["hash_failures"]:
        raise ValueError("R1.5a source replay changed")
    tasks = list(manifest["streams"])
    audit_records = set(manifest["audit_record_ids"])
    for task in tasks:
        task["capture_audit"] = (
            task["record_id"] in audit_records and task["block"] == "A" and int(task["repeat"]) == 0
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.jsonl"
    completed = _load_checkpoint(checkpoint_path)
    pending = [task for task in tasks if task["task_key"] not in completed]
    started = time.perf_counter()
    worker_count = max(1, int(jobs))
    if pending and worker_count == 1:
        policy = make_policy(str(manifest["incumbent_policy"]))
        with checkpoint_path.open("a") as handle:
            for index, task in enumerate(pending, start=1):
                result = rollout_task(task, records[str(task["record_id"])], policy)
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["task_key"]] = result
                if index % 100 == 0:
                    print(json.dumps({"completed_new": index, "pending_total": len(pending)}), flush=True)
    elif pending:
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(str(manifest["incumbent_policy"]), str(source_path)),
            )
        except PermissionError as exc:
            print(f"warning: R1.5a parallelism unavailable ({exc}); falling back to serial", file=sys.stderr)
            return run_labels(manifest_path, out_dir, 1)
        with executor, checkpoint_path.open("a") as handle:
            futures = {executor.submit(_worker, task): task["task_key"] for task in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["task_key"]] = result
                if index % 100 == 0:
                    print(json.dumps({"completed_new": index, "pending_total": len(pending)}), flush=True)
    elapsed = time.perf_counter() - started
    if len(completed) != len(tasks):
        raise RuntimeError(f"R1.5a labels incomplete: {len(completed)}/{len(tasks)}")

    labels_path = out_dir / "labels.jsonl"
    with labels_path.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(completed[task["task_key"]], separators=(",", ":")) + "\n")
    audits = [
        completed[task["task_key"]]
        for task in tasks if completed[task["task_key"]].get("audit_frames") is not None
    ]
    write_json(out_dir / "fixed_replay_audit.json", {"audits": audits})

    per_record = Counter(str(result["record_id"]) for result in completed.values())
    terminal_bootstrap_errors = sum(
        row["terminal"] and float(row["endpoint_leaf"]) != 0.0
        for result in completed.values() for row in result["rows"]
    )
    missing_context_cells = sorted(
        {
            (str(record["partition"]), str(record["context_cell"]))
            for record in records.values()
        }
        - {
            (str(records[record_id]["partition"]), str(records[record_id]["context_cell"]))
            for record_id in per_record
        }
    )
    deterministic_audit_mismatches = []
    audit_policy = make_policy(str(manifest["incumbent_policy"]))
    for task in tasks:
        if not task["capture_audit"]:
            continue
        replayed = rollout_task(task, records[str(task["record_id"])], audit_policy)
        if replayed != completed[task["task_key"]]:
            deterministic_audit_mismatches.append(task["task_key"])
    post_source_audit = verify_selected_sources(list(records.values()))
    integrity = {
        "expected_tasks": len(tasks),
        "completed_tasks": len(completed),
        "expected_horizon_rows": len(tasks) * len(HORIZONS),
        "completed_horizon_rows": sum(len(result["rows"]) for result in completed.values()),
        "records_with_exact_16_replicates": sum(value == 16 for value in per_record.values()),
        "records_total": len(records),
        "terminal_bootstrap_errors": int(terminal_bootstrap_errors),
        "missing_partition_context_cells": [list(value) for value in missing_context_cells],
        "deterministic_audit_paths": len(audits),
        "deterministic_audit_mismatches": deterministic_audit_mismatches,
        "source_hash_failures_before": source_audit["hash_failures"],
        "source_hash_failures_after": post_source_audit["hash_failures"],
    }
    passed = bool(
        integrity["expected_tasks"] == integrity["completed_tasks"]
        and integrity["expected_horizon_rows"] == integrity["completed_horizon_rows"]
        and integrity["records_with_exact_16_replicates"] == integrity["records_total"]
        and not terminal_bootstrap_errors
        and not missing_context_cells
        and len(audits) == len(manifest["audit_record_ids"])
        and not deterministic_audit_mismatches
        and not post_source_audit["hash_failures"]
    )
    summary = {
        "decision": "LABEL_CORPUS_PASS" if passed else "HOLD_INTEGRITY",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "labels": str(labels_path),
        "labels_sha256": sha256_path(labels_path),
        "elapsed_s_this_invocation": elapsed,
        "resumed_tasks": len(tasks) - len(pending),
        "new_tasks": len(pending),
        "jobs": worker_count,
        "integrity": integrity,
        "dashboard_eligible": False,
    }
    write_json(out_dir / "summary.json", summary)
    if not passed:
        raise RuntimeError("R1.5a label integrity gate failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--source-manifest", type=Path, required=True)
    freeze_parser.add_argument("--policy-file", type=Path, required=True)
    freeze_parser.add_argument("--a2-lock", type=Path, required=True)
    freeze_parser.add_argument("--out", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.command == "freeze":
        payload = freeze_manifest(
            source_manifest_path=args.source_manifest,
            policy_file=args.policy_file,
            a2_lock_path=args.a2_lock,
            prior_paths=prior_manifest_paths(),
        )
        write_json(args.out, payload)
        printable = {
            "out": str(args.out),
            "states": payload["ordinary_states"],
            "streams": len(payload["streams"]),
            "audit_paths": len(payload["audit_record_ids"]),
            "prior_stream_audit": payload["prior_stream_audit"],
        }
    else:
        payload = run_labels(args.manifest, args.out_dir, args.jobs)
        printable = payload
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

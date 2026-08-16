"""Generate diagnostic-only h40 labels for frozen H2 same-board context swaps."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.eval import make_policy, max_tile_excluding_initial_starter
from threes_rl.eval_stream_manifest import EVALUATOR_VERSION, stream_id
from threes_rl.human_h0_diagnostic import top_edge_metrics
from threes_rl.human_h2_context import state_with_context
from threes_rl.r15a_context_labels import (
    HORIZONS,
    collect_prior_stream_ids,
    incumbent_state_value,
    prior_manifest_paths,
    return_bin,
    sha256_path,
)
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim, score_board


VERSION = "r15a_context_synthetic_labels_a2_v1"
NAMESPACE = "threes-r15a-synthetic-a2-v1-20260711"
BLOCK_REPEATS = {"A": 8, "B": 8}
H2_MANIFEST_SHA256 = "c76a6a19ed86022d1573943c1bdc320ca60281d6872cd9321261bfb18e1d46ba"

_WORKER_POLICY: Any | None = None
_WORKER_TARGETS: dict[str, dict[str, Any]] = {}
_WORKER_PAIRS: list[dict[str, Any]] = []


def freeze_manifest(h2_manifest_path: Path, natural_manifest_path: Path) -> dict[str, Any]:
    observed_h2_hash = sha256_path(h2_manifest_path)
    if observed_h2_hash != H2_MANIFEST_SHA256:
        raise ValueError(f"Frozen H2 manifest hash mismatch: {observed_h2_hash}")
    natural = json.loads(natural_manifest_path.read_text())
    h2 = json.loads(h2_manifest_path.read_text())
    prior_paths = [*prior_manifest_paths(), natural_manifest_path]
    prior_exogenous, prior_logical, prior_summary = collect_prior_stream_ids(prior_paths)
    generated_exogenous: set[int] = set()
    generated_logical: set[int] = set()
    streams = []
    for target in sorted(h2["targets"], key=lambda row: str(row["root_id"])):
        for pair_index in range(len(h2["donor_pairs"])):
            case_id = f"{target['root_id']}:pair{pair_index}"
            for block, repeats in BLOCK_REPEATS.items():
                for repeat in range(repeats):
                    ids = {
                        kind: stream_id(NAMESPACE, f"{case_id}:{block}", repeat, kind)
                        for kind in ("logical", "deck", "slot", "policy")
                    }
                    logical_seed = ids.pop("logical")
                    if logical_seed in prior_logical or logical_seed in generated_logical:
                        raise RuntimeError("Synthetic logical stream collision")
                    if any(value in prior_exogenous or value in generated_exogenous for value in ids.values()):
                        raise RuntimeError("Synthetic exogenous stream collision")
                    generated_logical.add(logical_seed)
                    generated_exogenous.update(ids.values())
                    streams.append(
                        {
                            "task_key": f"{case_id}:{block}:{repeat}",
                            "case_id": case_id,
                            "target_root_id": target["root_id"],
                            "donor_pair_index": pair_index,
                            "block": block,
                            "repeat": repeat,
                            "logical_seed": logical_seed,
                            "deck_stream_id": ids["deck"],
                            "slot_stream_id": ids["slot"],
                            "policy_stream_id": ids["policy"],
                        }
                    )
    return {
        "manifest_version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "namespace": NAMESPACE,
        "dashboard_eligible": False,
        "fitting_eligible": False,
        "ordinary_holdout_eligible": False,
        "h2_manifest": str(h2_manifest_path),
        "h2_manifest_sha256": observed_h2_hash,
        "natural_label_manifest": str(natural_manifest_path),
        "natural_label_manifest_sha256": sha256_path(natural_manifest_path),
        "incumbent_policy": natural["incumbent_policy"],
        "incumbent_fingerprint": natural["incumbent_fingerprint"],
        "horizons": list(HORIZONS),
        "block_repeats": BLOCK_REPEATS,
        "cases": len(h2["targets"]) * len(h2["donor_pairs"]),
        "paired_replicates_per_case": sum(BLOCK_REPEATS.values()),
        "evaluator_version": EVALUATOR_VERSION,
        "coupling": "low/high arms share deck, slot, and policy stream IDs",
        "prior_stream_audit": {
            **prior_summary,
            "prior_exogenous_ids": len(prior_exogenous),
            "prior_logical_ids": len(prior_logical),
            "generated_exogenous_ids": len(generated_exogenous),
            "generated_logical_ids": len(generated_logical),
            "collisions": 0,
        },
        "audit_case_ids": sorted({stream["case_id"] for stream in streams})[:6],
        "streams": streams,
    }


def rollout_arm(task: dict[str, Any], target: dict[str, Any], pair: dict[str, Any], arm: str, policy: Any) -> dict[str, Any]:
    state = state_with_context(target, pair[f"{arm}_cycle"], pair["common_preview"])
    starter = int(target["starter_tile"])
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(task["deck_stream_id"]),
        slot_stream_id=int(task["slot_stream_id"]),
        starter_tile=starter,
    )
    rng = np.random.default_rng(int(task["policy_stream_id"]))
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
            action = int(policy(state, sim, rng))
            state, info = sim.step(state, action)
            if not info.moved:
                raise RuntimeError(f"Illegal synthetic action at {task['task_key']} {arm} step {completed}")
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
        target_value = float(score_delta + endpoint_leaf - root_leaf)
        rows.append(
            {
                "horizon": horizon,
                "moves_completed": completed,
                "score_delta": score_delta,
                "root_leaf": root_leaf,
                "endpoint_leaf": endpoint_leaf,
                "target": target_value,
                "return_bin": return_bin(target_value),
                "survived": int(completed >= horizon and not state.game_over),
                "reached_1536": int(reached_1536),
                "reached_3072": int(reached_3072),
                "terminal": int(state.game_over),
                **top_edge_metrics(state.board, initial_board),
            }
        )
    return {"arm": arm, "rows": rows, "audit_frames": audit_frames}


def rollout_task(task: dict[str, Any], target: dict[str, Any], pair: dict[str, Any], policy: Any) -> dict[str, Any]:
    return {
        **{key: task[key] for key in (
            "task_key", "case_id", "target_root_id", "donor_pair_index", "block", "repeat",
            "logical_seed", "deck_stream_id", "slot_stream_id", "policy_stream_id",
        )},
        "target_ancestry": target["ancestry_cluster"],
        "donor_ancestry": pair["donor_ancestry"],
        "common_preview": pair["common_preview"],
        "low": rollout_arm(task, target, pair, "low", policy),
        "high": rollout_arm(task, target, pair, "high", policy),
    }


def _init_worker(policy_spec: str, h2_manifest_path: str) -> None:
    global _WORKER_POLICY, _WORKER_TARGETS, _WORKER_PAIRS
    _WORKER_POLICY = make_policy(policy_spec)
    h2 = json.loads(Path(h2_manifest_path).read_text())
    _WORKER_TARGETS = {str(row["root_id"]): row for row in h2["targets"]}
    _WORKER_PAIRS = list(h2["donor_pairs"])


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_POLICY is None:
        raise RuntimeError("Synthetic worker policy is not initialized")
    return rollout_task(
        task,
        _WORKER_TARGETS[str(task["target_root_id"])],
        _WORKER_PAIRS[int(task["donor_pair_index"])],
        _WORKER_POLICY,
    )


def run(manifest_path: Path, out_dir: Path, jobs: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    h2_path = Path(str(manifest["h2_manifest"]))
    if sha256_path(h2_path) != manifest["h2_manifest_sha256"]:
        raise ValueError("Synthetic source manifest changed")
    h2 = json.loads(h2_path.read_text())
    targets = {str(row["root_id"]): row for row in h2["targets"]}
    pairs = list(h2["donor_pairs"])
    tasks = list(manifest["streams"])
    audit_cases = set(manifest["audit_case_ids"])
    for task in tasks:
        task["capture_audit"] = task["case_id"] in audit_cases and task["block"] == "A" and int(task["repeat"]) == 0
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                completed[str(row["task_key"])] = row
    pending = [task for task in tasks if task["task_key"] not in completed]
    started = time.perf_counter()
    worker_count = max(1, int(jobs))
    if pending and worker_count == 1:
        policy = make_policy(str(manifest["incumbent_policy"]))
        with checkpoint_path.open("a") as handle:
            for index, task in enumerate(pending, start=1):
                result = rollout_task(task, targets[task["target_root_id"]], pairs[int(task["donor_pair_index"])], policy)
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["task_key"]] = result
                if index % 50 == 0:
                    print(json.dumps({"completed_new": index, "pending_total": len(pending)}), flush=True)
    elif pending:
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(str(manifest["incumbent_policy"]), str(h2_path)),
            )
        except PermissionError as exc:
            print(f"warning: synthetic parallelism unavailable ({exc}); falling back to serial", file=sys.stderr)
            return run(manifest_path, out_dir, 1)
        with executor, checkpoint_path.open("a") as handle:
            futures = {executor.submit(_worker, task): task["task_key"] for task in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["task_key"]] = result
                if index % 50 == 0:
                    print(json.dumps({"completed_new": index, "pending_total": len(pending)}), flush=True)
    if len(completed) != len(tasks):
        raise RuntimeError("Synthetic label corpus incomplete")
    elapsed = time.perf_counter() - started
    labels_path = out_dir / "labels.jsonl"
    with labels_path.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(completed[task["task_key"]], separators=(",", ":")) + "\n")
    audits = [row for row in completed.values() if row["low"]["audit_frames"] is not None]
    write_json(out_dir / "fixed_replay_audit.json", {"audits": audits})
    terminal_errors = sum(
        row["terminal"] and float(row["endpoint_leaf"]) != 0.0
        for result in completed.values() for arm in ("low", "high") for row in result[arm]["rows"]
    )
    summary = {
        "decision": "SYNTHETIC_LABEL_PASS" if terminal_errors == 0 else "HOLD_INTEGRITY",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "labels": str(labels_path),
        "labels_sha256": sha256_path(labels_path),
        "tasks": len(tasks),
        "trajectories": len(tasks) * 2,
        "horizon_rows": len(tasks) * 2 * len(HORIZONS),
        "terminal_bootstrap_errors": int(terminal_errors),
        "audit_cases": len(audits),
        "elapsed_s_this_invocation": elapsed,
        "resumed_tasks": len(tasks) - len(pending),
        "dashboard_eligible": False,
        "fitting_eligible": False,
    }
    write_json(out_dir / "summary.json", summary)
    if terminal_errors:
        raise RuntimeError("Synthetic terminal bootstrap integrity failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--h2-manifest", type=Path, required=True)
    freeze_parser.add_argument("--natural-manifest", type=Path, required=True)
    freeze_parser.add_argument("--out", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.command == "freeze":
        payload = freeze_manifest(args.h2_manifest, args.natural_manifest)
        write_json(args.out, payload)
        printable = {"out": str(args.out), "cases": payload["cases"], "streams": len(payload["streams"]), "audit": payload["prior_stream_audit"]}
    else:
        payload = run(args.manifest, args.out_dir, args.jobs)
        printable = payload
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

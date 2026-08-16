"""Freeze, run, and analyze all-action human H0 continuations."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from threes_rl.eval import make_policy, max_tile_excluding_initial_starter
from threes_rl.eval_stream_manifest import stream_id
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim, rank_for_value, score_board
from threes_rl.train_td import state_from_replay_payload


HORIZONS = (10, 20, 40)
BLOCK_REPEATS = {"A": 32, "B": 32}
NAMESPACE = "threes-human-h0-20260710"
LOGICAL_SEED_START = 7_000_000
_WORKER_POLICY: Any | None = None


def _manifest_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("blocks"), dict):
        return [row for rows in payload["blocks"].values() for row in rows]
    return list(payload.get("jobs", payload.get("streams", [])))


def root_incumbent_action(root: dict[str, Any], policy: Any) -> tuple[int, dict[str, int]]:
    root_hash = hashlib.sha1(str(root["root_id"]).encode("utf-8")).hexdigest()[:16]
    ids = {
        "deck_stream_id": stream_id(NAMESPACE, root_hash, 0, "root_deck"),
        "slot_stream_id": stream_id(NAMESPACE, root_hash, 0, "root_slot"),
        "policy_stream_id": stream_id(NAMESPACE, root_hash, 0, "root_action"),
    }
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=ids["deck_stream_id"],
        slot_stream_id=ids["slot_stream_id"],
        starter_tile=int(root["starter_tile"]),
    )
    state = state_from_replay_payload(root["state"])
    return int(policy(state, sim, np.random.default_rng(ids["policy_stream_id"]))), ids


def freeze_manifest(
    roots_path: Path,
    incumbent_policy_spec: str,
    prior_manifest_paths: list[Path],
    corpus_manifest_path: Path,
) -> dict[str, Any]:
    root_manifest = json.loads(roots_path.read_text())
    if root_manifest.get("decision") != "PASS":
        raise ValueError("Human H0 roots did not pass")
    corpus = json.loads(corpus_manifest_path.read_text())
    prior_ids: set[int] = set()
    prior_logical: set[int] = set()
    for path in prior_manifest_paths:
        payload = json.loads(path.read_text())
        for row in _manifest_rows(payload):
            if row.get("logical_seed") is not None:
                prior_logical.add(int(row["logical_seed"]))
            for field in ("deck_stream_id", "slot_stream_id", "policy_stream_id"):
                if row.get(field) is not None:
                    prior_ids.add(int(row[field]))
    for game in corpus["games"]:
        prior_logical.add(int(game["logical_seed"]))
        prior_ids.add(int(game["deck_stream_id"]))
        prior_ids.add(int(game["slot_stream_id"]))

    policy = make_policy(incumbent_policy_spec)
    roots: list[dict[str, Any]] = []
    root_action_ids: set[int] = set()
    for root in root_manifest["roots"]:
        legal = [DIRECTION_NAMES.index(action) for action in root["state"]["legal_actions"]]
        incumbent_action, action_streams = root_incumbent_action(root, policy)
        if any(value in prior_ids or value in root_action_ids for value in action_streams.values()):
            raise RuntimeError("H0 root-action stream collision")
        root_action_ids.update(action_streams.values())
        recorded_action = DIRECTION_NAMES.index(str(root["recorded_action"]))
        if incumbent_action not in legal or recorded_action not in legal:
            raise ValueError(f"Frozen action is illegal at root {root['root_id']}")
        roots.append(
            {
                **root,
                "legal_action_indices": legal,
                "incumbent_action": DIRECTION_NAMES[incumbent_action],
                "incumbent_action_index": incumbent_action,
                "incumbent_action_streams": action_streams,
                "recorded_action_index": recorded_action,
                "recorded_disagrees": recorded_action != incumbent_action,
            }
        )

    streams: list[dict[str, Any]] = []
    generated_ids: set[int] = set()
    generated_logical: set[int] = set()
    logical_seed = LOGICAL_SEED_START
    for root in roots:
        root_hash = hashlib.sha1(str(root["root_id"]).encode("utf-8")).hexdigest()[:16]
        for block, repeats in BLOCK_REPEATS.items():
            for repeat in range(repeats):
                stream_block = f"{block}:{root_hash}"
                ids = {
                    kind: stream_id(NAMESPACE, stream_block, repeat, kind)
                    for kind in ("deck", "slot", "policy")
                }
                if logical_seed in prior_logical or logical_seed in generated_logical:
                    raise RuntimeError("H0 logical seed collision")
                if any(
                    value in prior_ids or value in root_action_ids or value in generated_ids
                    for value in ids.values()
                ):
                    raise RuntimeError("H0 exogenous stream collision")
                generated_logical.add(logical_seed)
                generated_ids.update(ids.values())
                streams.append(
                    {
                        "root_id": root["root_id"],
                        "block": block,
                        "repeat": repeat,
                        "logical_seed": logical_seed,
                        "deck_stream_id": ids["deck"],
                        "slot_stream_id": ids["slot"],
                        "policy_stream_id": ids["policy"],
                    }
                )
                logical_seed += 1

    frame286 = next(root for root in roots if root["role"] == "success_window" and root["source_frame_index"] == 286)
    matched_failure = min(
        (
            root
            for root in roots
            if root["role"] == "failure_control" and root["success_offset"] == 3
        ),
        key=lambda root: (float(root["match_distance"]), str(root["root_id"])),
    )
    return {
        "manifest_version": "human_h0_action_conditioned_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "namespace": NAMESPACE,
        "incumbent_policy": incumbent_policy_spec,
        "horizons": list(HORIZONS),
        "block_repeats": BLOCK_REPEATS,
        "replicates_per_root_action": sum(BLOCK_REPEATS.values()),
        "continuation_policy": "frozen incumbent after forced first action",
        "selection_uses_incumbent_values": False,
        "action_agreement_is_label": False,
        "dashboard_eligible": False,
        "roots_path": str(roots_path),
        "corpus_manifest": str(corpus_manifest_path),
        "prior_manifests_checked": [str(path) for path in prior_manifest_paths],
        "stream_disjointness": {
            "prior_logical_ids": len(prior_logical),
            "prior_exogenous_ids": len(prior_ids),
            "generated_logical_ids": len(generated_logical),
            "generated_exogenous_ids": len(generated_ids),
            "root_action_stream_ids": len(root_action_ids),
            "collisions": 0,
            "action_arm_coupling": "one shared stream triplet per root/replicate",
        },
        "audit_root_ids": [frame286["root_id"], matched_failure["root_id"]],
        "roots": roots,
        "streams": streams,
    }


def top_edge_metrics(board: np.ndarray, initial_board: np.ndarray) -> dict[str, Any]:
    ranks = [rank_for_value(int(value)) for value in board[0]]
    initial_ranks = [rank_for_value(int(value)) for value in initial_board[0]]
    descending = all(ranks[index] >= ranks[index + 1] for index in range(3))
    initial_anchor = int(initial_board[0, 0])
    return {
        "empty_count": int(np.count_nonzero(board == 0)),
        "anchor_preserved": int(initial_anchor <= 0 or int(board[0, 0]) >= initial_anchor),
        "top_edge_rank_mass_delta": int(sum(ranks) - sum(initial_ranks)),
        "top_edge_descending": int(descending),
    }


def rollout_task(task: dict[str, Any], policy: Any) -> dict[str, Any]:
    root = task["root"]
    stream = task["stream"]
    forced_action = int(task["action"])
    starter = int(root["starter_tile"])
    state = state_from_replay_payload(root["state"])
    initial_board = state.board.copy()
    initial_score = score_board(initial_board)
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(stream["deck_stream_id"]),
        slot_stream_id=int(stream["slot_stream_id"]),
        starter_tile=starter,
    )
    policy_rng = np.random.default_rng(int(stream["policy_stream_id"]))
    reached_1536 = max_tile_excluding_initial_starter(state.board, starter) >= 1536
    reached_3072 = max_tile_excluding_initial_starter(state.board, starter) >= 3072
    completed = 0
    rows: list[dict[str, Any]] = []
    audit_frames = [] if task["capture_audit"] else None
    if audit_frames is not None:
        audit_frames.append({"step": 0, "board": state.board.tolist(), "action": None})
    for horizon in HORIZONS:
        while completed < horizon and not state.game_over:
            before = state
            action = forced_action if completed == 0 else int(policy(before, sim, policy_rng))
            state, info = sim.step(before, action)
            if not info.moved:
                raise RuntimeError(f"Illegal H0 action {action} at {root['root_id']} step {completed}")
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
                    }
                )
        edge = top_edge_metrics(state.board, initial_board)
        rows.append(
            {
                "root_id": root["root_id"],
                "ancestry_cluster": root["ancestry_cluster"],
                "role": root["role"],
                "success_offset": int(root["success_offset"]),
                "source_frame_index": int(root["source_frame_index"]),
                "recorded_action": root["recorded_action"],
                "incumbent_action": root["incumbent_action"],
                "forced_action": DIRECTION_NAMES[forced_action],
                "block": stream["block"],
                "repeat": int(stream["repeat"]),
                "logical_seed": int(stream["logical_seed"]),
                "deck_stream_id": int(stream["deck_stream_id"]),
                "slot_stream_id": int(stream["slot_stream_id"]),
                "policy_stream_id": int(stream["policy_stream_id"]),
                "horizon": horizon,
                "score_delta": int(score_board(state.board) - initial_score),
                "survived": int(completed >= horizon and not state.game_over),
                "reached_1536": int(reached_1536),
                "reached_3072": int(reached_3072),
                **edge,
            }
        )
    return {"task_key": task["task_key"], "rows": rows, "audit_frames": audit_frames}


def _init_worker(policy_spec: str) -> None:
    global _WORKER_POLICY
    _WORKER_POLICY = make_policy(policy_spec)


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_POLICY is None:
        raise RuntimeError("H0 worker policy is not initialized")
    return rollout_task(task, _WORKER_POLICY)


def build_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    roots = {root["root_id"]: root for root in manifest["roots"]}
    audit_roots = set(manifest["audit_root_ids"])
    tasks = []
    for stream in manifest["streams"]:
        root = roots[stream["root_id"]]
        for action in root["legal_action_indices"]:
            task_key = f"{root['root_id']}:{stream['block']}:{stream['repeat']}:{action}"
            tasks.append(
                {
                    "task_key": task_key,
                    "root": root,
                    "stream": stream,
                    "action": int(action),
                    "capture_audit": root["root_id"] in audit_roots and int(stream["repeat"]) == 0 and stream["block"] == "A",
                }
            )
    return tasks


def run(manifest_path: Path, out_dir: Path, jobs: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    tasks = build_tasks(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open() as handle:
            for line in handle:
                payload = json.loads(line)
                completed[payload["task_key"]] = payload
    pending = [task for task in tasks if task["task_key"] not in completed]
    started = time.perf_counter()
    worker_count = max(1, int(jobs))
    if pending and worker_count == 1:
        policy = make_policy(manifest["incumbent_policy"])
        with checkpoint_path.open("a") as handle:
            for task in pending:
                result = rollout_task(task, policy)
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["task_key"]] = result
    elif pending:
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(manifest["incumbent_policy"],),
            )
        except PermissionError as exc:
            print(f"warning: H0 parallelism unavailable ({exc}); falling back to serial", file=sys.stderr)
            return run(manifest_path, out_dir, 1)
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
        raise RuntimeError(f"H0 incomplete: {len(completed)}/{len(tasks)} tasks")
    rows = [row for task in tasks for row in completed[task["task_key"]]["rows"]]
    results_path = out_dir / "results.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audits = [
        {"task_key": task["task_key"], "frames": completed[task["task_key"]]["audit_frames"]}
        for task in tasks
        if completed[task["task_key"]]["audit_frames"] is not None
    ]
    write_json(out_dir / "fixed_replay_audit.json", {"audits": audits})
    summary = {
        "manifest": str(manifest_path),
        "tasks": len(tasks),
        "rows": len(rows),
        "roots": len(manifest["roots"]),
        "ancestry_clusters": len({root["ancestry_cluster"] for root in manifest["roots"]}),
        "elapsed_s_this_invocation": elapsed,
        "resumed_tasks": len(tasks) - len(pending),
        "results": str(results_path),
        "dashboard_eligible": False,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def bootstrap(values: list[float], seed: int, repeats: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        samples[index] = float(np.mean(array[rng.integers(len(array), size=len(array))]))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def analyze(manifest_path: Path, results_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    roots = {root["root_id"]: root for root in manifest["roots"]}
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {
        (row["root_id"], row["block"], int(row["repeat"]), row["forced_action"], int(row["horizon"])): row
        for row in rows
    }
    fields = (
        "score_delta", "survived", "reached_1536", "reached_3072",
        "empty_count", "anchor_preserved", "top_edge_rank_mass_delta", "top_edge_descending",
    )
    comparisons: dict[str, Any] = {}
    root_metric_values: dict[tuple[str, int, str], float] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        horizon_payload = {}
        for field_index, field in enumerate(fields):
            per_root = {}
            block_diffs = {}
            informative_roots = {
                root_id: root for root_id, root in roots.items() if root["recorded_disagrees"]
            }
            for root_id, root in informative_roots.items():
                differences = []
                by_block = defaultdict(list)
                for block, repeats in BLOCK_REPEATS.items():
                    for repeat in range(repeats):
                        recorded_key = (root_id, block, repeat, root["recorded_action"], horizon)
                        incumbent_key = (root_id, block, repeat, root["incumbent_action"], horizon)
                        difference = float(keyed[recorded_key][field]) - float(keyed[incumbent_key][field])
                        differences.append(difference)
                        by_block[block].append(difference)
                per_root[root_id] = mean(differences)
                root_metric_values[(root_id, horizon, field)] = per_root[root_id]
                for block, values in by_block.items():
                    block_diffs.setdefault(block, []).extend(values)
            cluster_values = {}
            for cluster in sorted({root["ancestry_cluster"] for root in informative_roots.values()}):
                cluster_values[cluster] = mean(
                    value for root_id, value in per_root.items()
                    if informative_roots[root_id]["ancestry_cluster"] == cluster
                )
            all_differences = list(per_root.values())
            horizon_payload[field] = {
                "difference": mean(all_differences),
                "ci95_ancestry_bootstrap": bootstrap(
                    list(cluster_values.values()), seed=20260710 + horizon_index * 20 + field_index
                ),
                "block_differences": {block: mean(values) for block, values in block_diffs.items()},
                "cluster_differences": cluster_values,
                "root_wins": sum(value > 0 for value in all_differences),
                "root_losses": sum(value < 0 for value in all_differences),
                "root_ties": sum(value == 0 for value in all_differences),
            }
        comparisons[f"h{horizon}"] = horizon_payload

    success_disagreements = [
        root for root in roots.values() if root["role"] == "success_window" and root["recorded_disagrees"]
    ]
    success_score_values = [root_metric_values[(root["root_id"], 40, "score_delta")] for root in success_disagreements]
    success_milestone_values = [root_metric_values[(root["root_id"], 40, "reached_1536")] for root in success_disagreements]
    failure_clusters = sorted({root["ancestry_cluster"] for root in roots.values() if root["role"] == "failure_control"})
    failure_score = []
    failure_milestone = []
    failure_disagreement_counts = {}
    for cluster in failure_clusters:
        cluster_roots = [
            root for root in roots.values()
            if root["ancestry_cluster"] == cluster and root["recorded_disagrees"]
        ]
        failure_disagreement_counts[cluster] = len(cluster_roots)
        if cluster_roots:
            failure_score.append(mean(root_metric_values[(root["root_id"], 40, "score_delta")] for root in cluster_roots))
            failure_milestone.append(mean(root_metric_values[(root["root_id"], 40, "reached_1536")] for root in cluster_roots))

    h40 = comparisons["h40"]
    success_positive = sum(value > 0 for value in success_score_values)
    success_negative = sum(value < 0 for value in success_score_values)
    failure_score_mean = mean(failure_score)
    failure_milestone_mean = mean(failure_milestone)
    score_ci = h40["score_delta"]["ci95_ancestry_bootstrap"]
    milestone_lift = h40["reached_1536"]["difference"]
    survival_lift = h40["survived"]["difference"]
    failure_generalization_available = bool(
        len(failure_disagreement_counts) == 5
        and all(count > 0 for count in failure_disagreement_counts.values())
    )
    continue_gate = bool(
        success_positive >= 4
        and success_positive > success_negative
        and failure_generalization_available
        and failure_score_mean >= 0.0
        and failure_milestone_mean >= 0.0
        and (score_ci[0] > 0.0 or milestone_lift >= 0.05)
        and survival_lift >= -0.02
    )
    failure_score_ci = bootstrap(failure_score, seed=20260810)
    kill_gate = bool(
        (mean(success_score_values) < 0.0 and mean(success_milestone_values) <= 0.0)
        or failure_score_ci[1] < 0.0
        or (survival_lift < -0.02 and h40["survived"]["ci95_ancestry_bootstrap"][1] < 0.0)
    )
    decision = "CONTINUE" if continue_gate else "KILL" if kill_gate else "HOLD"

    rankings = []
    for root_id, root in roots.items():
        action_rows = []
        for action in root["state"]["legal_actions"]:
            action_keys = [
                key for key in keyed if key[0] == root_id and key[3] == action and key[4] == 40
            ]
            action_rows.append(
                {
                    "action": action,
                    "reached_3072": mean(float(keyed[key]["reached_3072"]) for key in action_keys),
                    "reached_1536": mean(float(keyed[key]["reached_1536"]) for key in action_keys),
                    "score_delta": mean(float(keyed[key]["score_delta"]) for key in action_keys),
                    "survived": mean(float(keyed[key]["survived"]) for key in action_keys),
                }
            )
        ordered = sorted(
            action_rows,
            key=lambda row: (row["reached_3072"], row["reached_1536"], row["score_delta"], row["survived"]),
            reverse=True,
        )
        rankings.append(
            {
                "root_id": root_id,
                "ancestry_cluster": root["ancestry_cluster"],
                "role": root["role"],
                "source_frame_index": root["source_frame_index"],
                "recorded_action": root["recorded_action"],
                "incumbent_action": root["incumbent_action"],
                "recorded_rank": next(index for index, row in enumerate(ordered, 1) if row["action"] == root["recorded_action"]),
                "incumbent_rank": next(index for index, row in enumerate(ordered, 1) if row["action"] == root["incumbent_action"]),
                "actions": ordered,
            }
        )
    frame286 = next(row for row in rankings if row["role"] == "success_window" and row["source_frame_index"] == 286)
    failure_distances = np.asarray(
        [
            float(root["match_distance"])
            for root in roots.values() if root["role"] == "failure_control"
        ],
        dtype=np.float64,
    )
    same_action_by_ancestry = {
        cluster: sorted(
            root["root_id"]
            for root in roots.values()
            if root["ancestry_cluster"] == cluster and not root["recorded_disagrees"]
        )
        for cluster in sorted({root["ancestry_cluster"] for root in roots.values()})
    }
    return {
        "decision": decision,
        "roots": len(roots),
        "ancestry_clusters": 6,
        "comparisons": comparisons,
        "comparison_scope": {
            "informative_disagreement_roots": sum(root["recorded_disagrees"] for root in roots.values()),
            "same_action_structural_zero_roots": sum(not root["recorded_disagrees"] for root in roots.values()),
            "same_action_root_ids_by_ancestry": same_action_by_ancestry,
            "failure_geometry_reference_match_distance": {
                "values": failure_distances.tolist(),
                "median": float(np.median(failure_distances)),
                "p90": float(np.quantile(failure_distances, 0.9)),
                "max": float(np.max(failure_distances)),
                "roots_over_1_0": int(np.count_nonzero(failure_distances > 1.0)),
            },
        },
        "gate": {
            "continue": continue_gate,
            "kill": kill_gate,
            "success_disagreement_roots": len(success_disagreements),
            "success_positive_score_roots": success_positive,
            "success_negative_score_roots": success_negative,
            "success_mean_score": mean(success_score_values),
            "success_mean_first1536": mean(success_milestone_values),
            "failure_cluster_mean_score": failure_score_mean,
            "failure_cluster_score_ci95": failure_score_ci,
            "failure_cluster_mean_first1536": failure_milestone_mean,
            "failure_disagreement_counts": failure_disagreement_counts,
            "failure_generalization_available": failure_generalization_available,
        },
        "frame286_case_study": frame286,
        "all_action_rankings": rankings,
        "dashboard_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--roots", type=Path, required=True)
    freeze_parser.add_argument("--corpus", type=Path, required=True)
    freeze_parser.add_argument("--incumbent-policy-file", type=Path, required=True)
    freeze_parser.add_argument("--prior-manifest", type=Path, action="append", default=[])
    freeze_parser.add_argument("--out", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--jobs", type=int, default=8)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--manifest", type=Path, required=True)
    analyze_parser.add_argument("--results", type=Path, required=True)
    analyze_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        policy_spec = args.incumbent_policy_file.read_text().splitlines()[-1]
        payload = freeze_manifest(args.roots, policy_spec, args.prior_manifest, args.corpus)
        write_json(args.out, payload)
    elif args.command == "run":
        payload = run(args.manifest, args.out_dir, args.jobs)
    else:
        payload = analyze(args.manifest, args.results)
        write_json(args.out, payload)
    if args.command == "freeze":
        printable = {
            "roots": len(payload["roots"]),
            "streams": len(payload["streams"]),
            "disagreements": sum(root["recorded_disagrees"] for root in payload["roots"]),
            "stream_disjointness": payload["stream_disjointness"],
            "audit_root_ids": payload["audit_root_ids"],
        }
    elif args.command == "analyze":
        printable = {
            "decision": payload["decision"],
            "gate": payload["gate"],
            "h40": payload["comparisons"]["h40"],
            "frame286_case_study": payload["frame286_case_study"],
        }
    else:
        printable = payload
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Freeze, run, and analyze the R1b held-out congested-window diagnostic."""

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

from threes_rl.eval import EvalStreamIds, make_policy, max_tile_excluding_initial_starter
from threes_rl.eval_stream_manifest import stream_id
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim, score_board
from threes_rl.train_td import state_from_replay_payload


HORIZONS = (10, 20, 40)
BLOCK_REPEATS = {"A": 8, "B": 8}
NAMESPACE = "threes-r1b-pre-c-20260710"
LOGICAL_SEED_START = 6_000_000
_WORKER_POLICY: Any | None = None
_WORKER_POLICY_SPEC = ""


def freeze_manifest(
    independence_audit_path: Path,
    eval_manifest_paths: list[Path],
) -> dict[str, Any]:
    audit = json.loads(independence_audit_path.read_text())
    if audit.get("decision") != "PASS":
        raise ValueError("Independence audit did not pass")
    prior_ids: set[int] = set()
    for path in eval_manifest_paths:
        manifest = json.loads(path.read_text())
        for rows in manifest["blocks"].values():
            for row in rows:
                prior_ids.update(
                    int(row[field])
                    for field in ("deck_stream_id", "slot_stream_id", "policy_stream_id")
                )
    jobs: list[dict[str, Any]] = []
    generated_ids: set[int] = set()
    logical_seed = LOGICAL_SEED_START
    for root_index, root in enumerate(audit["records"]):
        root_key = hashlib.sha1(str(root["ancestry_key"]).encode("utf-8")).hexdigest()[:16]
        for block, repeats in BLOCK_REPEATS.items():
            for repeat in range(repeats):
                stream_block = f"{block}:{root_key}"
                ids = {
                    kind: stream_id(NAMESPACE, stream_block, repeat, kind)
                    for kind in ("deck", "slot", "policy")
                }
                if any(value in prior_ids or value in generated_ids for value in ids.values()):
                    raise RuntimeError("Pre-C diagnostic RNG stream collision")
                generated_ids.update(ids.values())
                jobs.append(
                    {
                        "job_index": len(jobs),
                        "root_index": root_index,
                        "root_id": root["id"],
                        "ancestry_key": root["ancestry_key"],
                        "behavior_family": root["behavior_family"],
                        "block": block,
                        "repeat": repeat,
                        "logical_seed": logical_seed,
                        "starter_tile": root["starter_tile"],
                        "deck_stream_id": ids["deck"],
                        "slot_stream_id": ids["slot"],
                        "policy_stream_id": ids["policy"],
                        "state": root["state"],
                    }
                )
                logical_seed += 1
    return {
        "manifest_version": "r1b_pre_c_diagnostic_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "namespace": NAMESPACE,
        "horizons": list(HORIZONS),
        "block_repeats": BLOCK_REPEATS,
        "roots": len(audit["records"]),
        "jobs_per_arm": len(jobs),
        "independence_audit": str(independence_audit_path),
        "selection_uses_outcome": False,
        "sealed_c_outcomes_read": False,
        "prior_eval_manifests_checked": [str(path) for path in eval_manifest_paths],
        "stream_disjointness": {
            "prior_stream_ids": len(prior_ids),
            "generated_stream_ids": len(generated_ids),
            "collisions": 0,
        },
        "jobs": jobs,
    }


def _init_worker(policy_spec: str) -> None:
    global _WORKER_POLICY, _WORKER_POLICY_SPEC
    _WORKER_POLICY_SPEC = policy_spec
    _WORKER_POLICY = make_policy(policy_spec)


def rollout_job(job: dict[str, Any], policy: Any) -> list[dict[str, Any]]:
    starter_tile = int(job["starter_tile"])
    state = state_from_replay_payload(job["state"])
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(job["deck_stream_id"]),
        slot_stream_id=int(job["slot_stream_id"]),
        starter_tile=starter_tile,
    )
    policy_rng = np.random.default_rng(int(job["policy_stream_id"]))
    start_score = score_board(state.board)
    reached_1536 = max_tile_excluding_initial_starter(state.board, starter_tile) >= 1536
    rows: list[dict[str, Any]] = []
    completed_steps = 0
    for horizon in HORIZONS:
        while completed_steps < horizon and not state.game_over:
            before = state
            action = int(policy(before, sim, policy_rng))
            state, info = sim.step(before, action)
            if not info.moved:
                legal = sim.legal_actions(before)
                if not legal:
                    break
                state, info = sim.step(before, int(legal[0]))
            completed_steps += 1
            reached_1536 = reached_1536 or (
                max_tile_excluding_initial_starter(state.board, starter_tile) >= 1536
            )
        rows.append(
            {
                "job_index": int(job["job_index"]),
                "root_index": int(job["root_index"]),
                "root_id": str(job["root_id"]),
                "ancestry_key": str(job["ancestry_key"]),
                "behavior_family": str(job["behavior_family"]),
                "block": str(job["block"]),
                "repeat": int(job["repeat"]),
                "logical_seed": int(job["logical_seed"]),
                "deck_stream_id": int(job["deck_stream_id"]),
                "slot_stream_id": int(job["slot_stream_id"]),
                "policy_stream_id": int(job["policy_stream_id"]),
                "horizon": horizon,
                "score_gain": int(score_board(state.board) - start_score),
                "survived": int(completed_steps >= horizon and not state.game_over),
                "reached_1536": int(reached_1536),
                "completed_steps": completed_steps,
            }
        )
    return rows


def _worker(job: dict[str, Any]) -> list[dict[str, Any]]:
    if _WORKER_POLICY is None:
        raise RuntimeError("Diagnostic worker policy is not initialized")
    return rollout_job(job, _WORKER_POLICY)


def run_arm(policy_spec: str, manifest_path: Path, out_dir: Path, jobs: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    work = manifest["jobs"]
    started = time.perf_counter()
    outputs: dict[int, list[dict[str, Any]]] = {}
    worker_count = max(1, int(jobs))
    if worker_count == 1:
        policy = make_policy(policy_spec)
        for job in work:
            outputs[int(job["job_index"])] = rollout_job(job, policy)
    else:
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(policy_spec,),
            )
        except PermissionError as exc:
            print(f"warning: parallel diagnostic unavailable ({exc}); falling back to serial", file=sys.stderr)
            return run_arm(policy_spec, manifest_path, out_dir, 1)
        with executor:
            futures = {executor.submit(_worker, job): int(job["job_index"]) for job in work}
            for future in concurrent.futures.as_completed(futures):
                outputs[futures[future]] = future.result()
    rows = [row for index in range(len(work)) for row in outputs[index]]
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    results_path = out_dir / "results.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    elapsed = time.perf_counter() - started
    summary = {
        "policy": policy_spec,
        "manifest": str(manifest_path),
        "roots": int(manifest["roots"]),
        "jobs": len(work),
        "rows": len(rows),
        "elapsed_s": elapsed,
        "jobs_per_s": len(work) / max(elapsed, 1e-9),
        "results": str(results_path),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def read_rows(path: Path) -> dict[tuple[str, str, int, int], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["root_id"], row["block"], int(row["repeat"]), int(row["horizon"])): row
        for row in rows
    }


def root_bootstrap(values: list[float], seed: int, repeats: int = 10_000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        samples[index] = float(np.mean(array[rng.integers(len(array), size=len(array))]))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def analyze(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = read_rows(baseline_path)
    candidate = read_rows(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("Diagnostic arm rows do not match")
    for key in baseline:
        for field in ("logical_seed", "deck_stream_id", "slot_stream_id", "policy_stream_id"):
            if baseline[key][field] != candidate[key][field]:
                raise ValueError(f"Paired stream mismatch for {key}: {field}")
    roots = sorted({key[0] for key in baseline})
    metrics: dict[str, Any] = {}
    metric_fields = ("reached_1536", "score_gain", "survived")
    for horizon_index, horizon in enumerate(HORIZONS):
        horizon_rows: dict[str, Any] = {}
        for metric_index, field in enumerate(metric_fields):
            root_differences = []
            for root in roots:
                keys = [key for key in baseline if key[0] == root and key[3] == horizon]
                root_differences.append(
                    mean(float(candidate[key][field]) - float(baseline[key][field]) for key in keys)
                )
            block_estimates = {}
            for block in BLOCK_REPEATS:
                keys = [key for key in baseline if key[1] == block and key[3] == horizon]
                block_estimates[block] = mean(
                    float(candidate[key][field]) - float(baseline[key][field]) for key in keys
                )
            keys = [key for key in baseline if key[3] == horizon]
            horizon_rows[field] = {
                "baseline": mean(float(baseline[key][field]) for key in keys),
                "candidate": mean(float(candidate[key][field]) for key in keys),
                "difference": mean(root_differences),
                "ci95_root_bootstrap": root_bootstrap(
                    root_differences, seed=20260710 + horizon_index * 10 + metric_index
                ),
                "block_differences": block_estimates,
                "root_wins": sum(value > 0 for value in root_differences),
                "root_losses": sum(value < 0 for value in root_differences),
                "root_ties": sum(value == 0 for value in root_differences),
            }
        metrics[f"h{horizon}"] = horizon_rows
    h40 = metrics["h40"]
    blocks_pre_c = any(h40[field]["ci95_root_bootstrap"][1] < 0.0 for field in metric_fields)
    supports_pre_c = bool(
        not blocks_pre_c
        and h40["reached_1536"]["difference"] > 0.0
        and h40["score_gain"]["difference"] >= 0.0
        and h40["survived"]["difference"] >= -0.02
    )
    decision = "BLOCKS_PRE_C" if blocks_pre_c else "SUPPORTS_PRE_C" if supports_pre_c else "NEUTRAL"
    return {
        "decision": decision,
        "paired_roots": len(roots),
        "repeats_per_root": sum(BLOCK_REPEATS.values()),
        "metrics": metrics,
        "frozen_interpretation": {
            "blocks_pre_c": blocks_pre_c,
            "supports_pre_c": supports_pre_c,
            "h40_survival_noninferiority": -0.02,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--independence-audit", type=Path, required=True)
    freeze_parser.add_argument("--eval-manifest", type=Path, action="append", required=True)
    freeze_parser.add_argument("--out", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--policy", required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--jobs", type=int, default=4)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--baseline", type=Path, required=True)
    analyze_parser.add_argument("--candidate", type=Path, required=True)
    analyze_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        payload = freeze_manifest(args.independence_audit, args.eval_manifest)
        write_json(args.out, payload)
    elif args.command == "run":
        payload = run_arm(args.policy, args.manifest, args.out_dir, args.jobs)
    else:
        payload = analyze(args.baseline, args.candidate)
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

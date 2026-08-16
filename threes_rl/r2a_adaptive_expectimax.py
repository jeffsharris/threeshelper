"""Freeze, run, and analyze the single preregistered R2a adaptive-depth gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from threes_rl.eval import make_policy, max_tile_excluding_initial_starter
from threes_rl.eval_stream_manifest import EVALUATOR_VERSION, stream_id
from threes_rl.expectimax import NtupleExpectimaxPolicy, _state_key
from threes_rl.human_h0_diagnostic import top_edge_metrics
from threes_rl.r15a_context_labels import collect_prior_stream_ids, prior_manifest_paths, sha256_path
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim, score_board
from threes_rl.train_td import state_from_replay_payload


VERSION = "r2a_adaptive_expectimax_v1"
NAMESPACE = "threes-r2a-adaptive-v1-20260711"
SOURCE_MANIFEST_SHA256 = "8604778696164fdabd5ab653c933b0b543ca1d20a8fde1d78b6e7da2994d794a"
POLICY_FILE_SHA256 = "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4"
NODE_BUDGET = 2048
CHANCE_LIMIT = 8
EMPTY_TRIGGER = 3
MARGIN_TRIGGER = 0.02
MAX_ROOTS = 64
MAX_FAMILY_ROOTS = 32
HORIZONS = (10, 20, 40)
BLOCK_REPEATS = {"A": 8, "B": 8}

_WORKER_POLICY: Any | None = None
_WORKER_ROOTS: dict[str, dict[str, Any]] = {}


class NodeBudgetedNtuplePolicy(NtupleExpectimaxPolicy):
    """Depth-3 incumbent leaf with deterministic value-node budget fallback."""

    def __init__(self, *args: Any, node_budget: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.node_budget = int(node_budget)
        self.expanded_value_nodes = 0
        self.budget_cutoffs = 0

    def action_values(self, state: Any, sim: ThreesSim) -> list[tuple[int, float]]:
        self.expanded_value_nodes = 0
        self.budget_cutoffs = 0
        return super().action_values(state, sim)

    def _value(self, state: Any, sim: ThreesSim, depth: int) -> float:
        if state.game_over or depth <= 0:
            return super()._value(state, sim, depth)
        key = _state_key(state, depth)
        if key not in self._cache:
            if self.expanded_value_nodes >= self.node_budget:
                self.budget_cutoffs += 1
                return self._post_spawn_state_value(state, sim)
            self.expanded_value_nodes += 1
        return super()._value(state, sim, depth)


def clone_depth3(base: NtupleExpectimaxPolicy) -> NodeBudgetedNtuplePolicy:
    return NodeBudgetedNtuplePolicy(
        base.checkpoint,
        depth=3,
        chance_limit=CHANCE_LIMIT,
        blend_specs=list(base.blend_specs),
        phase_blend_specs=list(base.phase_blend_specs),
        bonus_specs=list(base.bonus_specs),
        tie_margin=base.tie_margin,
        tie_breaker=base.tie_breaker,
        ensemble_mode=base.ensemble_mode,
        geometry_weight=base.geometry_weight,
        geometry_min_tile=base.geometry_min_tile,
        node_budget=NODE_BUDGET,
    )


def normalized_margin(action_values: list[tuple[int, float]]) -> float:
    if len(action_values) < 2:
        return 1.0
    values = sorted((float(value) for _action, value in action_values), reverse=True)
    return float((values[0] - values[1]) / max(1.0, abs(values[0])))


def choose_action(policy: NtupleExpectimaxPolicy, values: list[tuple[int, float]], seed: int) -> int:
    return int(policy._select_action(values, np.random.default_rng(seed)))


def milestone_for_built_max(built_max: int) -> int | None:
    if 768 <= built_max < 1536:
        return 1536
    if 1536 <= built_max < 3072:
        return 3072
    return None


def recorded_promotion_offset(record: dict[str, Any], milestone: int) -> int | None:
    replay = json.loads(Path(str(record["source_replay"])).read_text())
    start_index = int(record["source_frame_index"])
    starter_value = record.get("starter_tile", 1536)
    starter = None if starter_value is None else int(starter_value)
    for frame in replay.get("frames", []):
        frame_index = int(frame.get("index", -1))
        if frame_index <= start_index or frame_index > start_index + 40:
            continue
        state = frame.get("state") or {}
        board = np.asarray(state.get("board", []), dtype=np.int32)
        if board.shape == (4, 4) and max_tile_excluding_initial_starter(board, starter) >= milestone:
            return frame_index - start_index
    return None


def _candidate_priority(record: dict[str, Any]) -> tuple[Any, ...]:
    metadata = record["context_metadata"]
    offset = record.get("recorded_promotion_offset")
    return (
        0 if offset is not None else 1,
        abs(int(offset or 40) - 20),
        int(metadata["empty_count"]),
        hashlib.sha256(str(record["record_id"]).encode("utf-8")).hexdigest(),
    )


def _stratified_select(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        role = "promotion_window" if record["recorded_promotion_offset"] is not None else "control"
        groups[(str(record["behavior_family"]), int(record["target_milestone"]), role)].append(record)
    for rows in groups.values():
        rows.sort(key=lambda row: (
            int(row["context_metadata"]["empty_count"]),
            float(row["incumbent_margin"]),
            hashlib.sha256(str(row["record_id"]).encode()).hexdigest(),
        ))
    selected = []
    family_counts: Counter[str] = Counter()
    ordered_groups = sorted(groups)
    while len(selected) < MAX_ROOTS:
        added = False
        for group in ordered_groups:
            rows = groups[group]
            if not rows:
                continue
            family = group[0]
            if family_counts[family] >= MAX_FAMILY_ROOTS:
                continue
            selected.append(rows.pop(0))
            family_counts[family] += 1
            added = True
            if len(selected) == MAX_ROOTS:
                break
        if not added:
            break
    return selected


def freeze_roots(source_manifest_path: Path, policy_file: Path) -> dict[str, Any]:
    if sha256_path(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("R2a source manifest changed")
    if sha256_path(policy_file) != POLICY_FILE_SHA256:
        raise ValueError("R2a incumbent policy changed")
    source = json.loads(source_manifest_path.read_text())
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source["selected_records"]:
        if record.get("partition") not in {"ancestry_holdout", "family_holdout"}:
            continue
        built_max = int(record["context_metadata"]["built_max"])
        milestone = milestone_for_built_max(built_max)
        if milestone is None:
            continue
        row = dict(record)
        row["target_milestone"] = milestone
        row["recorded_promotion_offset"] = recorded_promotion_offset(record, milestone)
        by_root[str(record["root_cluster"])].append(row)
    root_candidates = [min(rows, key=_candidate_priority) for rows in by_root.values()]

    policy_spec = policy_file.read_text().splitlines()[-1]
    base = make_policy(policy_spec)
    if not isinstance(base, NtupleExpectimaxPolicy):
        raise TypeError("R2a incumbent is not n-tuple expectimax")
    for record in root_candidates:
        starter = int(record.get("starter_tile", 1536))
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        started = time.perf_counter()
        values = base.action_values(state, sim)
        record["incumbent_runtime_s"] = time.perf_counter() - started
        record["incumbent_action_values"] = {DIRECTION_NAMES[action]: value for action, value in values}
        record["incumbent_margin"] = normalized_margin(values)
    selected = _stratified_select(root_candidates)
    deep = clone_depth3(base)
    for record in selected:
        starter = int(record.get("starter_tile", 1536))
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter)
        selection_seed = stream_id(NAMESPACE, str(record["root_cluster"]), 0, "selection")
        base_values = [(DIRECTION_NAMES.index(name), float(value)) for name, value in record["incumbent_action_values"].items()]
        base_action = choose_action(base, base_values, selection_seed)
        empties = int(record["context_metadata"]["empty_count"])
        built_max = int(record["context_metadata"]["built_max"])
        trigger = built_max >= 768 and built_max < 3072 and (empties <= EMPTY_TRIGGER or float(record["incumbent_margin"]) <= MARGIN_TRIGGER)
        if trigger:
            started = time.perf_counter()
            deep_values = deep.action_values(state, sim)
            adaptive_runtime = time.perf_counter() - started
            adaptive_action = choose_action(deep, deep_values, selection_seed)
            deep_payload = {DIRECTION_NAMES[action]: value for action, value in deep_values}
            nodes = deep.expanded_value_nodes
            cutoffs = deep.budget_cutoffs
        else:
            adaptive_runtime = float(record["incumbent_runtime_s"])
            adaptive_action = base_action
            deep_payload = record["incumbent_action_values"]
            nodes = 0
            cutoffs = 0
        record.update(
            {
                "selection_seed": selection_seed,
                "triggered": trigger,
                "trigger_reasons": {
                    "low_empty": empties <= EMPTY_TRIGGER,
                    "low_margin": float(record["incumbent_margin"]) <= MARGIN_TRIGGER,
                },
                "incumbent_action_index": base_action,
                "incumbent_action": DIRECTION_NAMES[base_action],
                "adaptive_action_index": adaptive_action,
                "adaptive_action": DIRECTION_NAMES[adaptive_action],
                "adaptive_action_values": deep_payload,
                "adaptive_runtime_s": adaptive_runtime,
                "runtime_ratio": adaptive_runtime / max(1e-9, float(record["incumbent_runtime_s"])),
                "expanded_value_nodes": nodes,
                "budget_cutoffs": cutoffs,
                "action_changed": adaptive_action != base_action,
                "role": "promotion_window" if record["recorded_promotion_offset"] is not None else "control",
            }
        )
    ratios = [float(row["runtime_ratio"]) for row in selected]
    changed = [row for row in selected if row["action_changed"]]
    p90_ratio = float(np.quantile(ratios, 0.9)) if ratios else float("inf")
    checks = {
        "min_40_roots": len(selected) >= 40,
        "min_3_families": len({row["behavior_family"] for row in selected}) >= 3,
        "min_8_changed_actions": len(changed) >= 8,
        "min_10pct_activity": len(changed) / max(1, len(selected)) >= 0.10,
        "changed_min_3_families": len({row["behavior_family"] for row in changed}) >= 3,
        "median_runtime_max_3x": median(ratios) <= 3.0 if ratios else False,
        "p90_runtime_max_5x": p90_ratio <= 5.0,
    }
    return {
        "manifest_version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PRESCREEN_PASS" if all(checks.values()) else "KILL_R2A_PRESCREEN",
        "dashboard_eligible": False,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "incumbent_policy_file": str(policy_file),
        "incumbent_policy_file_sha256": POLICY_FILE_SHA256,
        "incumbent_policy": policy_spec,
        "config": {
            "depth": 3,
            "node_budget": NODE_BUDGET,
            "chance_limit": CHANCE_LIMIT,
            "empty_trigger": EMPTY_TRIGGER,
            "margin_trigger": MARGIN_TRIGGER,
            "built_max_min": 768,
            "built_max_max_exclusive": 3072,
        },
        "checks": checks,
        "summary": {
            "roots": len(selected),
            "families": dict(Counter(str(row["behavior_family"]) for row in selected)),
            "roles": dict(Counter(str(row["role"]) for row in selected)),
            "milestones": dict(Counter(str(row["target_milestone"]) for row in selected)),
            "triggered": sum(bool(row["triggered"]) for row in selected),
            "changed": len(changed),
            "changed_families": dict(Counter(str(row["behavior_family"]) for row in changed)),
            "median_runtime_ratio": median(ratios) if ratios else None,
            "p90_runtime_ratio": p90_ratio,
            "max_runtime_ratio": max(ratios, default=None),
            "median_nodes": median([int(row["expanded_value_nodes"]) for row in selected]),
            "total_budget_cutoffs": sum(int(row["budget_cutoffs"]) for row in selected),
        },
        "roots": selected,
    }


def freeze_streams(root_manifest_path: Path) -> dict[str, Any]:
    roots = json.loads(root_manifest_path.read_text())
    if roots.get("decision") != "PRESCREEN_PASS":
        raise ValueError("R2a prescreen did not pass")
    prior_paths = [*prior_manifest_paths(), root_manifest_path]
    prior_exogenous, prior_logical, prior_summary = collect_prior_stream_ids(prior_paths)
    generated_exogenous: set[int] = set()
    generated_logical: set[int] = set()
    streams = []
    for root in roots["roots"]:
        root_id = str(root["record_id"])
        for block, repeats in BLOCK_REPEATS.items():
            for repeat in range(repeats):
                ids = {kind: stream_id(NAMESPACE, f"{root_id}:{block}", repeat, kind) for kind in ("logical", "deck", "slot", "policy")}
                logical = ids.pop("logical")
                if logical in prior_logical or logical in generated_logical:
                    raise RuntimeError("R2a logical collision")
                if any(value in prior_exogenous or value in generated_exogenous for value in ids.values()):
                    raise RuntimeError("R2a stream collision")
                generated_logical.add(logical)
                generated_exogenous.update(ids.values())
                streams.append(
                    {
                        "task_key": f"{root_id}:{block}:{repeat}",
                        "record_id": root_id,
                        "block": block,
                        "repeat": repeat,
                        "logical_seed": logical,
                        "deck_stream_id": ids["deck"],
                        "slot_stream_id": ids["slot"],
                        "policy_stream_id": ids["policy"],
                    }
                )
    return {
        "manifest_version": "r2a_paired_streams_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root_manifest": str(root_manifest_path),
        "root_manifest_sha256": sha256_path(root_manifest_path),
        "incumbent_policy": roots["incumbent_policy"],
        "horizons": list(HORIZONS),
        "block_repeats": BLOCK_REPEATS,
        "evaluator_version": EVALUATOR_VERSION,
        "coupling": "incumbent/adaptive first-action arms share deck, slot, and policy IDs",
        "dashboard_eligible": False,
        "stream_audit": {
            **prior_summary,
            "prior_exogenous_ids": len(prior_exogenous),
            "prior_logical_ids": len(prior_logical),
            "generated_exogenous_ids": len(generated_exogenous),
            "generated_logical_ids": len(generated_logical),
            "collisions": 0,
        },
        "audit_record_ids": sorted(str(root["record_id"]) for root in roots["roots"])[:6],
        "streams": streams,
    }


def rollout_arm(task: dict[str, Any], root: dict[str, Any], action: int, policy: Any) -> dict[str, Any]:
    starter = int(root.get("starter_tile", 1536))
    state = state_from_replay_payload(root["state"])
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(task["deck_stream_id"]),
        slot_stream_id=int(task["slot_stream_id"]),
        starter_tile=starter,
    )
    rng = np.random.default_rng(int(task["policy_stream_id"]))
    initial_board = state.board.copy()
    initial_score = score_board(initial_board)
    target = int(root["target_milestone"])
    reached = max_tile_excluding_initial_starter(state.board, starter) >= target
    completed = 0
    rows = []
    audit_frames = [] if bool(task.get("capture_audit")) else None
    if audit_frames is not None:
        audit_frames.append({"step": 0, "board": state.board.tolist(), "preview": state.preview.label})
    for horizon in HORIZONS:
        while completed < horizon and not state.game_over:
            selected = int(action) if completed == 0 else int(policy(state, sim, rng))
            state, info = sim.step(state, selected)
            if not info.moved:
                raise RuntimeError(f"Illegal R2a action at {task['task_key']} step {completed}")
            completed += 1
            reached = reached or max_tile_excluding_initial_starter(state.board, starter) >= target
            if audit_frames is not None:
                audit_frames.append(
                    {
                        "step": completed,
                        "action": DIRECTION_NAMES[selected],
                        "inserted_value": info.inserted_value,
                        "inserted_pos": None if info.inserted_pos is None else list(info.inserted_pos),
                        "board": state.board.tolist(),
                    }
                )
        rows.append(
            {
                "horizon": horizon,
                "score_delta": int(score_board(state.board) - initial_score),
                "reached_target": int(reached),
                "survived": int(completed >= horizon and not state.game_over),
                "moves_completed": completed,
                **top_edge_metrics(state.board, initial_board),
            }
        )
    return {"rows": rows, "audit_frames": audit_frames}


def rollout_task(task: dict[str, Any], root: dict[str, Any], policy: Any) -> dict[str, Any]:
    incumbent_action = int(root["incumbent_action_index"])
    adaptive_action = int(root["adaptive_action_index"])
    incumbent = rollout_arm(task, root, incumbent_action, policy)
    adaptive = incumbent if adaptive_action == incumbent_action else rollout_arm(task, root, adaptive_action, policy)
    return {
        **{key: task[key] for key in (
            "task_key", "record_id", "block", "repeat", "logical_seed",
            "deck_stream_id", "slot_stream_id", "policy_stream_id",
        )},
        "root_cluster": root["root_cluster"],
        "behavior_family": root["behavior_family"],
        "role": root["role"],
        "target_milestone": root["target_milestone"],
        "action_changed": root["action_changed"],
        "incumbent": incumbent,
        "adaptive": adaptive,
    }


def _init_worker(policy_spec: str, root_manifest_path: str) -> None:
    global _WORKER_POLICY, _WORKER_ROOTS
    _WORKER_POLICY = make_policy(policy_spec)
    payload = json.loads(Path(root_manifest_path).read_text())
    _WORKER_ROOTS = {str(root["record_id"]): root for root in payload["roots"]}


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_POLICY is None:
        raise RuntimeError("R2a worker is not initialized")
    return rollout_task(task, _WORKER_ROOTS[str(task["record_id"])], _WORKER_POLICY)


def run(manifest_path: Path, out_dir: Path, jobs: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    roots_path = Path(manifest["root_manifest"])
    if sha256_path(roots_path) != manifest["root_manifest_sha256"]:
        raise ValueError("R2a root manifest changed")
    roots_payload = json.loads(roots_path.read_text())
    roots = {str(root["record_id"]): root for root in roots_payload["roots"]}
    tasks = list(manifest["streams"])
    audit_ids = set(manifest["audit_record_ids"])
    for task in tasks:
        task["capture_audit"] = task["record_id"] in audit_ids and task["block"] == "A" and int(task["repeat"]) == 0
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.jsonl"
    completed = {}
    if checkpoint.exists():
        with checkpoint.open() as handle:
            for line in handle:
                row = json.loads(line)
                completed[row["task_key"]] = row
    pending = [task for task in tasks if task["task_key"] not in completed]
    started = time.perf_counter()
    worker_count = max(1, int(jobs))
    if pending and worker_count == 1:
        policy = make_policy(manifest["incumbent_policy"])
        with checkpoint.open("a") as handle:
            for task in pending:
                result = rollout_task(task, roots[task["record_id"]], policy)
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["task_key"]] = result
    elif pending:
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_worker,
                initargs=(manifest["incumbent_policy"], str(roots_path)),
            )
        except PermissionError as exc:
            print(f"warning: R2a parallelism unavailable ({exc}); falling back to serial", file=sys.stderr)
            return run(manifest_path, out_dir, 1)
        with executor, checkpoint.open("a") as handle:
            futures = {executor.submit(_worker, task): task["task_key"] for task in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["task_key"]] = result
                if index % 50 == 0:
                    print(json.dumps({"completed_new": index, "pending_total": len(pending)}), flush=True)
    if len(completed) != len(tasks):
        raise RuntimeError("R2a continuation corpus incomplete")
    results_path = out_dir / "results.jsonl"
    with results_path.open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(completed[task["task_key"]], separators=(",", ":")) + "\n")
    audits = [row for row in completed.values() if row["incumbent"]["audit_frames"] is not None]
    write_json(out_dir / "fixed_replay_audit.json", {"audits": audits})
    summary = {
        "decision": "R2A_CONTINUATIONS_PASS",
        "tasks": len(tasks),
        "roots": len(roots),
        "results": str(results_path),
        "results_sha256": sha256_path(results_path),
        "elapsed_s_this_invocation": time.perf_counter() - started,
        "resumed_tasks": len(tasks) - len(pending),
        "audit_paths": len(audits),
        "dashboard_eligible": False,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def bootstrap(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(10_000, dtype=np.float64)
    for index in range(len(samples)):
        samples[index] = float(np.mean(rng.choice(array, size=len(array), replace=True)))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def analyze(root_manifest_path: Path, results_path: Path) -> dict[str, Any]:
    root_manifest = json.loads(root_manifest_path.read_text())
    roots = {str(root["record_id"]): root for root in root_manifest["roots"]}
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    fields = ("score_delta", "reached_target", "survived", "anchor_preserved", "empty_count", "moves_completed")
    comparisons = {}
    root_h40 = {}
    for horizon in HORIZONS:
        horizon_payload = {}
        for field_index, field in enumerate(fields):
            by_root: dict[str, list[float]] = defaultdict(list)
            by_block: dict[str, list[float]] = defaultdict(list)
            for result in results:
                incumbent = next(row for row in result["incumbent"]["rows"] if row["horizon"] == horizon)
                adaptive = next(row for row in result["adaptive"]["rows"] if row["horizon"] == horizon)
                difference = float(adaptive[field]) - float(incumbent[field])
                by_root[str(result["record_id"])].append(difference)
                by_block[str(result["block"])].append(difference)
            root_values = {root: float(np.mean(values)) for root, values in by_root.items()}
            if horizon == 40:
                for root, value in root_values.items():
                    root_h40.setdefault(root, {})[field] = value
            values = list(root_values.values())
            horizon_payload[field] = {
                "difference": float(np.mean(values)),
                "ci95_root_bootstrap": bootstrap(values, 20260711 + horizon * 10 + field_index),
                "block_differences": {block: float(np.mean(values)) for block, values in by_block.items()},
                "root_wins": sum(value > 0 for value in values),
                "root_losses": sum(value < 0 for value in values),
                "root_ties": sum(value == 0 for value in values),
            }
        comparisons[f"h{horizon}"] = horizon_payload
    changed = [root for root in roots.values() if root["action_changed"]]
    catastrophic = [
        root_id for root_id, metrics in root_h40.items()
        if metrics["score_delta"] <= -20_000.0
        and (metrics["survived"] <= -0.10 or metrics["anchor_preserved"] <= -0.20)
    ]
    h40 = comparisons["h40"]
    score_pass = h40["score_delta"]["ci95_root_bootstrap"][0] > 0.0
    milestone_pass = h40["reached_target"]["difference"] >= 0.03 and h40["reached_target"]["ci95_root_bootstrap"][0] >= 0.0
    checks = {
        "min_8_changed_roots": len(changed) >= 8,
        "changed_min_3_families": len({root["behavior_family"] for root in changed}) >= 3,
        "primary_score_or_milestone": score_pass or milestone_pass,
        "both_blocks_score_positive": all(value > 0.0 for value in h40["score_delta"]["block_differences"].values()),
        "both_blocks_milestone_nonnegative": all(value >= 0.0 for value in h40["reached_target"]["block_differences"].values()),
        "survival_noninferior": h40["survived"]["difference"] >= -0.02,
        "anchor_noninferior": h40["anchor_preserved"]["difference"] >= -0.03,
        "no_catastrophic_roots": not catastrophic,
    }
    passed = all(checks.values())
    return {
        "decision": "R2A_OFFLINE_PASS" if passed else "KILL_R2A",
        "checks": checks,
        "comparisons": comparisons,
        "changed_roots": len(changed),
        "changed_families": dict(Counter(str(root["behavior_family"]) for root in changed)),
        "catastrophic_root_ids": catastrophic,
        "dashboard_eligible": False,
        "normal_start_development_authorized": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    roots_parser = subparsers.add_parser("freeze-roots")
    roots_parser.add_argument("--source-manifest", type=Path, required=True)
    roots_parser.add_argument("--policy-file", type=Path, required=True)
    roots_parser.add_argument("--out", type=Path, required=True)
    streams_parser = subparsers.add_parser("freeze-streams")
    streams_parser.add_argument("--roots", type=Path, required=True)
    streams_parser.add_argument("--out", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--jobs", type=int, default=8)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--roots", type=Path, required=True)
    analyze_parser.add_argument("--results", type=Path, required=True)
    analyze_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-roots":
        payload = freeze_roots(args.source_manifest, args.policy_file)
        write_json(args.out, payload)
        printable = {"decision": payload["decision"], "checks": payload["checks"], "summary": payload["summary"]}
    elif args.command == "freeze-streams":
        payload = freeze_streams(args.roots)
        write_json(args.out, payload)
        printable = {"streams": len(payload["streams"]), "audit": payload["stream_audit"]}
    elif args.command == "run":
        payload = run(args.manifest, args.out_dir, args.jobs)
        printable = payload
    else:
        payload = analyze(args.roots, args.results)
        write_json(args.out, payload)
        printable = {"decision": payload["decision"], "checks": payload["checks"], "h40": payload["comparisons"]["h40"]}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

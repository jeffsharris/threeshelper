"""Freeze, run, and analyze the bounded H2 preview/cycle context diagnostic."""

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
from threes_rl.sim import DIRECTION_NAMES, Preview, ThreesSim, preview_from_label, score_board
from threes_rl.train_td import state_from_replay_payload


HORIZONS = (10, 20)
BLOCK_REPEATS = {"A": 8, "B": 8}
TARGET_OFFSETS = {20, 3}
DONOR_PAIRS = 4
NAMESPACE = "threes-human-h2-context-20260711"
LOGICAL_SEED_START = 8_000_000
_WORKER_POLICY: Any | None = None


def _manifest_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("blocks"), dict):
        return [row for rows in payload["blocks"].values() for row in rows]
    return list(payload.get("jobs", payload.get("streams", [])))


def plus_probability(root: dict[str, Any]) -> float:
    return float(root["match_features"]["plus_probability"])


def cycle_payload(root: dict[str, Any]) -> dict[str, Any]:
    return dict(root["state"]["tile_cycle"])


def cycle_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_counts = left["small_counts"]
    right_counts = right["small_counts"]
    terms = [
        (int(left["small_pos"]) - int(right["small_pos"])) / 12.0,
        (int(left["span_small_pos"]) - int(right["span_small_pos"])) / 21.0,
        (int(left["small_seen_total"]) - int(right["small_seen_total"])) / 400.0,
        float(bool(left["large_pending"])) - float(bool(right["large_pending"])),
    ]
    terms.extend(
        (int(left_counts.get(kind, 0)) - int(right_counts.get(kind, 0))) / 4.0
        for kind in ("red", "blue", "gray")
    )
    return float(np.sqrt(sum(value * value for value in terms)))


def preview_options_for_cycle(cycle: dict[str, Any]) -> list[tuple[Preview, float]]:
    sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=1536)
    return [
        (option.preview, float(option.probability))
        for option in sim.preview_options(
            {str(key): int(value) for key, value in cycle["small_counts"].items()},
            int(cycle["small_pos"]),
            int(cycle["small_seen_total"]),
            int(cycle["span_small_pos"]),
            bool(cycle["large_pending"]),
            int(cycle["max_tile"]),
        )
        if float(option.probability) > 0.0
    ]


def small_support(cycle: dict[str, Any]) -> dict[str, float]:
    return {
        preview.kind: probability
        for preview, probability in preview_options_for_cycle(cycle)
        if preview.kind in {"red", "blue", "gray"}
    }


def select_donor_pairs(roots: list[dict[str, Any]], count: int = DONOR_PAIRS) -> list[dict[str, Any]]:
    by_ancestry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root in roots:
        by_ancestry[str(root["ancestry_cluster"])].append(root)
    candidates = []
    for ancestry, ancestry_roots in by_ancestry.items():
        positive = [root for root in ancestry_roots if plus_probability(root) > 0.0]
        zero = [root for root in ancestry_roots if plus_probability(root) == 0.0]
        if not positive or not zero:
            continue
        high = sorted(positive, key=lambda root: (-plus_probability(root), str(root["root_id"])))[0]
        high_cycle = cycle_payload(high)
        low_candidates = sorted(
            zero,
            key=lambda root: (cycle_distance(high_cycle, cycle_payload(root)), str(root["root_id"])),
        )
        selected = None
        for low in low_candidates:
            shared = set(small_support(high_cycle)) & set(small_support(cycle_payload(low)))
            if shared:
                selected = low
                break
        if selected is None:
            continue
        low_cycle = cycle_payload(selected)
        high_support = small_support(high_cycle)
        low_support = small_support(low_cycle)
        shared = sorted(set(high_support) & set(low_support))
        common_preview = sorted(
            shared,
            key=lambda kind: (-min(high_support[kind], low_support[kind]), kind),
        )[0]
        candidates.append(
            {
                "donor_ancestry": ancestry,
                "high_root_id": high["root_id"],
                "low_root_id": selected["root_id"],
                "high_plus_probability": plus_probability(high),
                "low_plus_probability": 0.0,
                "cycle_distance": cycle_distance(high_cycle, low_cycle),
                "common_preview": common_preview,
                "common_preview_support": {
                    "high": high_support[common_preview],
                    "low": low_support[common_preview],
                },
                "high_cycle": high_cycle,
                "low_cycle": low_cycle,
            }
        )
    candidates.sort(key=lambda row: (-float(row["high_plus_probability"]), str(row["donor_ancestry"])))
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible cycle donor ancestries; need {count}")
    return candidates[:count]


def state_with_context(target: dict[str, Any], cycle: dict[str, Any], preview_kind: str):
    sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=int(target["starter_tile"]))
    preview = preview_from_label(preview_kind)
    supported = {
        option.kind
        for option, probability in preview_options_for_cycle(cycle)
        if probability > 0.0
    }
    if preview.kind not in supported:
        raise ValueError(f"Preview {preview.kind} is unsupported by transplanted cycle")
    return sim.state_from_snapshot(
        target["state"]["board"],
        preview,
        (
            {str(key): int(value) for key, value in cycle["small_counts"].items()},
            int(cycle["small_pos"]),
            int(cycle["small_seen_total"]),
            int(cycle["span_small_pos"]),
            bool(cycle["large_pending"]),
            int(cycle["max_tile"]),
        ),
        move_count=int(target["state"]["move_count"]),
    )


def action_value_payload(policy: Any, state: Any, starter_tile: int) -> dict[str, Any]:
    sim = ThreesSim.from_stream_ids(deck_stream_id=3, slot_stream_id=4, starter_tile=starter_tile)
    action_values = [(int(action), float(value)) for action, value in policy.action_values(state, sim)]
    ordered = sorted(action_values, key=lambda row: (-row[1], row[0]))
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else ordered[0]
    scale = max(1.0, abs(best[1]))
    return {
        "selected_action": DIRECTION_NAMES[best[0]],
        "selected_action_index": best[0],
        "best_value": best[1],
        "top_two_margin": best[1] - second[1],
        "normalized_top_two_margin": (best[1] - second[1]) / scale,
        "action_values": {DIRECTION_NAMES[action]: value for action, value in action_values},
    }


def freeze_manifest(
    roots_path: Path,
    incumbent_policy_spec: str,
    prior_manifest_paths: list[Path],
) -> dict[str, Any]:
    source = json.loads(roots_path.read_text())
    if source.get("decision") != "PASS":
        raise ValueError("H2 source root manifest did not pass")
    roots = list(source["roots"])
    targets = sorted(
        (root for root in roots if int(root["success_offset"]) in TARGET_OFFSETS),
        key=lambda root: (str(root["ancestry_cluster"]), -int(root["success_offset"])),
    )
    target_counts: dict[str, int] = defaultdict(int)
    for target in targets:
        target_counts[str(target["ancestry_cluster"])] += 1
    if len(targets) != 12 or sorted(target_counts.values()) != [2] * 6:
        raise ValueError(f"H2 target balance failed: {dict(target_counts)}")
    donor_pairs = select_donor_pairs(roots)
    policy = make_policy(incumbent_policy_spec)

    preview_sensitivity = []
    for root in roots:
        observed = state_from_replay_payload(root["state"])
        observed_values = action_value_payload(policy, observed, int(root["starter_tile"]))
        variants = []
        for preview, probability in preview_options_for_cycle(cycle_payload(root)):
            sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=int(root["starter_tile"]))
            cycle = cycle_payload(root)
            variant = sim.state_from_snapshot(
                root["state"]["board"],
                preview,
                (
                    {str(key): int(value) for key, value in cycle["small_counts"].items()},
                    int(cycle["small_pos"]),
                    int(cycle["small_seen_total"]),
                    int(cycle["span_small_pos"]),
                    bool(cycle["large_pending"]),
                    int(cycle["max_tile"]),
                ),
                move_count=int(root["state"]["move_count"]),
            )
            variants.append(
                {
                    "preview_kind": preview.kind,
                    "preview_value": preview.value,
                    "preview_candidates": list(preview.candidates),
                    "support_probability": probability,
                    **action_value_payload(policy, variant, int(root["starter_tile"])),
                }
            )
        preview_sensitivity.append(
            {
                "root_id": root["root_id"],
                "ancestry_cluster": root["ancestry_cluster"],
                "role": root["role"],
                "observed_preview": root["state"]["preview"],
                "observed_action_values": observed_values,
                "variants": variants,
            }
        )

    cycle_search = []
    for target in targets:
        for pair_index, pair in enumerate(donor_pairs):
            low = state_with_context(target, pair["low_cycle"], pair["common_preview"])
            high = state_with_context(target, pair["high_cycle"], pair["common_preview"])
            if not np.array_equal(low.board, high.board) or low.preview != high.preview:
                raise RuntimeError("H2 cycle arms do not share board/current preview")
            low_values = action_value_payload(policy, low, int(target["starter_tile"]))
            high_values = action_value_payload(policy, high, int(target["starter_tile"]))
            cycle_search.append(
                {
                    "case_id": f"{target['root_id']}:donor{pair_index}",
                    "target_root_id": target["root_id"],
                    "target_ancestry": target["ancestry_cluster"],
                    "success_offset": int(target["success_offset"]),
                    "donor_pair_index": pair_index,
                    "donor_ancestry": pair["donor_ancestry"],
                    "common_preview": pair["common_preview"],
                    "low": low_values,
                    "high": high_values,
                    "action_flip": low_values["selected_action"] != high_values["selected_action"],
                    "normalized_margin_change": abs(
                        float(high_values["normalized_top_two_margin"])
                        - float(low_values["normalized_top_two_margin"])
                    ),
                }
            )

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
    streams = []
    generated_ids: set[int] = set()
    generated_logical: set[int] = set()
    logical_seed = LOGICAL_SEED_START
    for target in targets:
        for pair_index, _pair in enumerate(donor_pairs):
            case_hash = hashlib.sha1(f"{target['root_id']}:{pair_index}".encode()).hexdigest()[:16]
            for block, repeats in BLOCK_REPEATS.items():
                for repeat in range(repeats):
                    ids = {
                        kind: stream_id(NAMESPACE, f"{block}:{case_hash}", repeat, kind)
                        for kind in ("deck", "slot", "policy")
                    }
                    if logical_seed in prior_logical or logical_seed in generated_logical:
                        raise RuntimeError("H2 logical stream collision")
                    if any(value in prior_ids or value in generated_ids for value in ids.values()):
                        raise RuntimeError("H2 exogenous stream collision")
                    generated_logical.add(logical_seed)
                    generated_ids.update(ids.values())
                    streams.append(
                        {
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
                    logical_seed += 1
    return {
        "manifest_version": "human_h2_context_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "namespace": NAMESPACE,
        "dashboard_eligible": False,
        "policy_training_authorized": False,
        "incumbent_policy": incumbent_policy_spec,
        "roots_path": str(roots_path),
        "prior_manifests_checked": [str(path) for path in prior_manifest_paths],
        "horizons": list(HORIZONS),
        "block_repeats": BLOCK_REPEATS,
        "targets": targets,
        "donor_pairs": donor_pairs,
        "preview_sensitivity": preview_sensitivity,
        "cycle_search": cycle_search,
        "streams": streams,
        "stream_disjointness": {
            "prior_logical_ids": len(prior_logical),
            "prior_exogenous_ids": len(prior_ids),
            "generated_logical_ids": len(generated_logical),
            "generated_exogenous_ids": len(generated_ids),
            "collisions": 0,
        },
        "checks": {
            "target_roots": len(targets),
            "target_ancestries": len(target_counts),
            "roots_per_ancestry": dict(target_counts),
            "donor_pairs": len(donor_pairs),
            "cycle_cases": len(cycle_search),
            "stream_rows": len(streams),
            "rollout_tasks": len(streams) * 2,
            "metric_rows": len(streams) * 2 * len(HORIZONS),
        },
    }


def rollout_task(task: dict[str, Any], policy: Any) -> dict[str, Any]:
    target = task["target"]
    pair = task["pair"]
    stream = task["stream"]
    arm = task["arm"]
    state = state_with_context(target, pair[f"{arm}_cycle"], pair["common_preview"])
    starter = int(target["starter_tile"])
    initial_board = state.board.copy()
    initial_score = score_board(initial_board)
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(stream["deck_stream_id"]),
        slot_stream_id=int(stream["slot_stream_id"]),
        starter_tile=starter,
    )
    rng = np.random.default_rng(int(stream["policy_stream_id"]))
    completed = 0
    reached_1536 = max_tile_excluding_initial_starter(state.board, starter) >= 1536
    reached_3072 = max_tile_excluding_initial_starter(state.board, starter) >= 3072
    rows = []
    audit_frames = [] if task["capture_audit"] else None
    if audit_frames is not None:
        audit_frames.append({"step": 0, "board": state.board.tolist(), "preview": state.preview.kind})
    for horizon in HORIZONS:
        while completed < horizon and not state.game_over:
            action = int(policy(state, sim, rng))
            state, info = sim.step(state, action)
            if not info.moved:
                raise RuntimeError(f"Illegal H2 continuation action at {task['task_key']}")
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
        rows.append(
            {
                "target_root_id": target["root_id"],
                "target_ancestry": target["ancestry_cluster"],
                "success_offset": int(target["success_offset"]),
                "donor_pair_index": int(task["donor_pair_index"]),
                "donor_ancestry": pair["donor_ancestry"],
                "arm": arm,
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
                "empty_count": int(np.count_nonzero(state.board == 0)),
                "anchor_preserved": int(int(state.board[0, 0]) >= int(initial_board[0, 0])),
            }
        )
    return {"task_key": task["task_key"], "rows": rows, "audit_frames": audit_frames}


def _init_worker(policy_spec: str) -> None:
    global _WORKER_POLICY
    _WORKER_POLICY = make_policy(policy_spec)


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_POLICY is None:
        raise RuntimeError("H2 worker policy is not initialized")
    return rollout_task(task, _WORKER_POLICY)


def build_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    targets = {root["root_id"]: root for root in manifest["targets"]}
    tasks = []
    for stream in manifest["streams"]:
        target = targets[stream["target_root_id"]]
        pair_index = int(stream["donor_pair_index"])
        pair = manifest["donor_pairs"][pair_index]
        for arm in ("low", "high"):
            key = f"{target['root_id']}:{pair_index}:{stream['block']}:{stream['repeat']}:{arm}"
            tasks.append(
                {
                    "task_key": key,
                    "target": target,
                    "pair": pair,
                    "donor_pair_index": pair_index,
                    "stream": stream,
                    "arm": arm,
                    "capture_audit": (
                        target["root_id"] == manifest["targets"][0]["root_id"]
                        and pair_index == 0
                        and stream["block"] == "A"
                        and int(stream["repeat"]) == 0
                    ),
                }
            )
    return tasks


def run(manifest_path: Path, out_dir: Path, jobs: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    tasks = build_tasks(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.jsonl"
    completed = {}
    if checkpoint_path.exists():
        with checkpoint_path.open() as handle:
            for line in handle:
                payload = json.loads(line)
                completed[payload["task_key"]] = payload
    pending = [task for task in tasks if task["task_key"] not in completed]
    started = time.perf_counter()
    if pending:
        try:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max(1, int(jobs)),
                initializer=_init_worker,
                initargs=(manifest["incumbent_policy"],),
            )
        except PermissionError as exc:
            print(f"warning: H2 parallelism unavailable ({exc})", file=sys.stderr)
            raise
        with executor, checkpoint_path.open("a") as handle:
            futures = {executor.submit(_worker, task): task["task_key"] for task in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
                handle.flush()
                completed[result["task_key"]] = result
                if index % 100 == 0:
                    print(json.dumps({"completed_new": index, "pending_total": len(pending)}), flush=True)
    if len(completed) != len(tasks):
        raise RuntimeError(f"H2 incomplete: {len(completed)}/{len(tasks)}")
    rows = [row for task in tasks for row in completed[task["task_key"]]["rows"]]
    results_path = out_dir / "results.csv"
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audits = [
        {"task_key": task["task_key"], "frames": completed[task["task_key"]]["audit_frames"]}
        for task in tasks if completed[task["task_key"]]["audit_frames"] is not None
    ]
    write_json(out_dir / "fixed_replay_audit.json", {"audits": audits})
    summary = {
        "manifest": str(manifest_path),
        "tasks": len(tasks),
        "rows": len(rows),
        "elapsed_s_this_invocation": time.perf_counter() - started,
        "resumed_tasks": len(tasks) - len(pending),
        "results": str(results_path),
        "dashboard_eligible": False,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def permutation_null95(case_differences: list[list[float]], seed: int, repeats: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    stats = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        null_means = [
            abs(float(np.mean(np.asarray(values) * rng.choice((-1.0, 1.0), size=len(values)))))
            for values in case_differences
        ]
        stats[index] = float(np.median(null_means))
    return float(np.quantile(stats, 0.95))


def analyze(manifest_path: Path, results_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {
        (
            row["target_root_id"], int(row["donor_pair_index"]), row["block"],
            int(row["repeat"]), row["arm"], int(row["horizon"]),
        ): row
        for row in rows
    }
    cases = [(target["root_id"], pair_index) for target in manifest["targets"] for pair_index in range(4)]
    target_ancestry = {target["root_id"]: target["ancestry_cluster"] for target in manifest["targets"]}
    thresholds = {"score_delta": (500.0, 1000.0), "reached_1536": (0.02, 0.05), "survived": (0.02, 0.05)}
    endpoints = {}
    for endpoint_index, (field, (informative_threshold, material_threshold)) in enumerate(thresholds.items()):
        case_payloads = []
        case_differences = []
        for target_id, pair_index in cases:
            differences = []
            block_means = {}
            for block, repeats in BLOCK_REPEATS.items():
                block_values = []
                for repeat in range(repeats):
                    low = keyed[(target_id, pair_index, block, repeat, "low", 20)]
                    high = keyed[(target_id, pair_index, block, repeat, "high", 20)]
                    difference = float(high[field]) - float(low[field])
                    differences.append(difference)
                    block_values.append(difference)
                block_means[block] = mean(block_values)
            combined = mean(differences)
            case_differences.append(differences)
            case_payloads.append(
                {
                    "target_root_id": target_id,
                    "target_ancestry": target_ancestry[target_id],
                    "donor_pair_index": pair_index,
                    "difference": combined,
                    "absolute_difference": abs(combined),
                    "block_differences": block_means,
                    "informative": abs(combined) >= informative_threshold,
                    "block_sign_stable": block_means["A"] * block_means["B"] > 0.0,
                }
            )
        informative = [row for row in case_payloads if row["informative"]]
        stable_rate = (
            mean(float(row["block_sign_stable"]) for row in informative) if informative else 0.0
        )
        median_absolute = float(np.median([row["absolute_difference"] for row in case_payloads]))
        null95 = permutation_null95(case_differences, seed=20260711 + endpoint_index)
        material = bool(
            len(informative) >= 12
            and stable_rate >= 0.70
            and median_absolute >= material_threshold
            and null95 < median_absolute
        )
        ancestry = {}
        for cluster in sorted(set(target_ancestry.values())):
            values = [row["difference"] for row in case_payloads if row["target_ancestry"] == cluster]
            ancestry[cluster] = {
                "mean_difference": mean(values),
                "median_absolute_difference": float(np.median(np.abs(values))),
            }
        endpoints[field] = {
            "median_absolute_h20_difference": median_absolute,
            "permutation_null95": null95,
            "informative_pairs": len(informative),
            "informative_block_sign_stability": stable_rate,
            "material": material,
            "by_ancestry": ancestry,
            "cases": case_payloads,
        }

    cycle_search = manifest["cycle_search"]
    action_flip_rate = mean(float(row["action_flip"]) for row in cycle_search)
    margin_change_rate = mean(float(row["normalized_margin_change"] >= 0.01) for row in cycle_search)
    decision_sensitive = action_flip_rate >= 0.15 or margin_change_rate >= 0.25
    outcome_sensitive = any(payload["material"] for payload in endpoints.values())
    decision = "CONTEXT_MATERIAL" if decision_sensitive and outcome_sensitive else "CONTEXT_WEAK_OR_INCONCLUSIVE"

    preview_rows = manifest["preview_sensitivity"]
    preview_variant_count = 0
    preview_action_flips = 0
    roots_with_flip = 0
    for root in preview_rows:
        observed_action = root["observed_action_values"]["selected_action"]
        root_flips = 0
        for variant in root["variants"]:
            preview_variant_count += 1
            if variant["selected_action"] != observed_action:
                preview_action_flips += 1
                root_flips += 1
        roots_with_flip += int(root_flips > 0)
    return {
        "decision": decision,
        "preview_only_control": {
            "roots": len(preview_rows),
            "supported_variants": preview_variant_count,
            "action_flip_variants": preview_action_flips,
            "action_flip_rate": preview_action_flips / preview_variant_count,
            "roots_with_any_action_flip": roots_with_flip,
        },
        "cycle_only_search": {
            "cases": len(cycle_search),
            "action_flips": sum(row["action_flip"] for row in cycle_search),
            "action_flip_rate": action_flip_rate,
            "normalized_margin_change_ge_1pct": sum(row["normalized_margin_change"] >= 0.01 for row in cycle_search),
            "normalized_margin_change_ge_1pct_rate": margin_change_rate,
            "decision_sensitive": decision_sensitive,
        },
        "cycle_only_h20": endpoints,
        "outcome_sensitive": outcome_sensitive,
        "dashboard_eligible": False,
        "policy_training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--roots", type=Path, required=True)
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
        policy_spec = next(
            line.strip()
            for line in args.incumbent_policy_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        payload = freeze_manifest(args.roots, policy_spec, args.prior_manifest)
        write_json(args.out, payload)
    elif args.command == "run":
        payload = run(args.manifest, args.out_dir, args.jobs)
    else:
        payload = analyze(args.manifest, args.results)
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

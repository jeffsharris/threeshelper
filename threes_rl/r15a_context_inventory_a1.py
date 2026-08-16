"""Amendment A1 natural-state inventory with family-capped loss weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from threes_rl.context_residual import context_metadata, schema_sha256
from threes_rl.phase0_replay_coverage_inventory import replay_action_signature
from threes_rl.r15a_context_inventory import (
    DEFAULT_REPLAY_GLOBS,
    FAMILY_HOLDOUT,
    MAX_FAMILY_SHARE,
    MAX_STATES_PER_ANCESTRY,
    PARTITION_STATE_CAPS,
    coalesced_behavior_family,
    context_bins,
    context_cell,
    coverage_summary,
    deterministic_key,
    glob_replays,
    round_robin_states,
)
from threes_rl.replay_provenance import GENUINE_ROOT_ORIGINS, replay_provenance
from threes_rl.restart_manifest import canonical_ancestry_id, state_signature
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload


A1_VERSION = "r15a_context_inventory_a1_v1"
A1_LABEL_NAMESPACE = "threes-r15a-labels-a1-v1-20260711"


def family_stratified_split(root_clusters: list[str]) -> tuple[set[str], set[str]]:
    ordered = sorted(root_clusters, key=lambda root: deterministic_key("A1-holdout", root))
    if len(ordered) >= 5:
        holdout_count = int(math.ceil(0.20 * len(ordered)))
    elif len(ordered) >= 2:
        holdout_count = 1
    else:
        holdout_count = 0
    return set(ordered[holdout_count:]), set(ordered[:holdout_count])


def waterfill_family_masses(root_counts: dict[str, int], cap: float = MAX_FAMILY_SHARE) -> dict[str, float]:
    if not root_counts or any(count <= 0 for count in root_counts.values()):
        raise ValueError("Family root counts must be positive")
    remaining = set(root_counts)
    remaining_mass = 1.0
    masses: dict[str, float] = {}
    while remaining:
        total_roots = sum(root_counts[family] for family in remaining)
        proposed = {
            family: remaining_mass * root_counts[family] / total_roots
            for family in remaining
        }
        over = sorted(family for family, mass in proposed.items() if mass > cap + 1e-15)
        if not over:
            masses.update(proposed)
            break
        for family in over:
            masses[family] = float(cap)
            remaining.remove(family)
            remaining_mass -= float(cap)
        if remaining_mass < -1e-12:
            raise ValueError("Family cap is infeasible")
        if not remaining and remaining_mass > 1e-9:
            raise ValueError("Family cap is infeasible for represented families")
    total = sum(masses.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Family masses do not sum to one: {total}")
    return dict(sorted(masses.items()))


def assign_partition_weights(records: list[dict[str, Any]], partition: str) -> dict[str, Any]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root_family: dict[str, str] = {}
    for record in records:
        root = str(record["root_cluster"])
        by_root[root].append(record)
        root_family[root] = str(record["behavior_family"])
    family_roots: dict[str, list[str]] = defaultdict(list)
    for root, family in root_family.items():
        family_roots[family].append(root)
    family_counts = {family: len(roots) for family, roots in family_roots.items()}
    fit_family_masses = (
        waterfill_family_masses(family_counts)
        if partition == "train" else {family: count / len(by_root) for family, count in family_counts.items()}
    )
    equal_family_masses = {family: 1.0 / len(family_roots) for family in family_roots}
    natural_root_mass = 1.0 / len(by_root) if by_root else 0.0
    root_fit_weights: dict[str, float] = {}
    for family, roots in family_roots.items():
        roots.sort()
        for root in roots:
            root_fit_weights[root] = fit_family_masses[family] / len(roots)
            state_count = len(by_root[root])
            for record in by_root[root]:
                record["fit_weight"] = root_fit_weights[root] / state_count
                record["metric_weight_root_balanced"] = natural_root_mass / state_count
                record["metric_weight_family_balanced"] = (
                    equal_family_masses[family] / len(roots) / state_count
                )
    weighted_ess = (
        1.0 / sum(weight * weight for weight in root_fit_weights.values())
        if root_fit_weights else 0.0
    )
    return {
        "raw_root_family_counts": dict(sorted(family_counts.items())),
        "raw_root_family_shares": {
            family: count / len(by_root) for family, count in sorted(family_counts.items())
        },
        "effective_fit_family_weights": fit_family_masses,
        "maximum_effective_fit_family_weight": max(fit_family_masses.values(), default=0.0),
        "weighted_effective_ancestry_count": weighted_ess,
        "root_weight_min": min(root_fit_weights.values(), default=0.0),
        "root_weight_max": max(root_fit_weights.values(), default=0.0),
        "state_fit_weight_sum": sum(float(record["fit_weight"]) for record in records),
        "state_root_balanced_metric_weight_sum": sum(
            float(record["metric_weight_root_balanced"]) for record in records
        ),
        "state_family_balanced_metric_weight_sum": sum(
            float(record["metric_weight_family_balanced"]) for record in records
        ),
    }


def context_holes(records: list[dict[str, Any]]) -> dict[str, Any]:
    stages = ("early_lt384", "mid_384_768", "late_1536", "endgame_3072p")
    plus = ("zero", "lt_0.10", "0.10_0.25", "ge_0.25")
    preview = ("small", "bonus")
    pending = ("not_pending", "pending")
    empties = ("0_1", "2_3", "4_plus")
    possible = {
        "|".join((stage, plus_bin, preview_bin, pending_bin, empties_bin))
        for stage in stages
        for plus_bin in plus
        for preview_bin in preview
        for pending_bin in pending
        for empties_bin in empties
    }
    observed = {str(record["context_cell"]) for record in records}
    return {
        "possible_frozen_cells": len(possible),
        "observed_cells": len(observed),
        "missing_cells": len(possible - observed),
        "missing_cell_examples": sorted(possible - observed)[:30],
    }


def build_inventory_a1(paths: list[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    replay_signatures: set[tuple[str, str]] = set()
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    ancestry_family: dict[str, str] = {}
    ancestry_cluster: dict[str, str] = {}
    source_families: Counter[str] = Counter()

    for path in paths:
        counts["source_files_scanned"] += 1
        try:
            replay = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            counts["invalid_replay_json"] += 1
            continue
        if not isinstance(replay, dict):
            counts["invalid_replay_payload"] += 1
            continue
        provenance = replay_provenance(replay, path)
        if (
            provenance.get("replay_origin") not in GENUINE_ROOT_ORIGINS
            or provenance.get("root_origin") not in GENUINE_ROOT_ORIGINS
            or not provenance.get("replay_reset_invariant")
        ):
            counts["not_natural_normal_start"] += 1
            continue
        family = coalesced_behavior_family(replay, path)
        replay_key = (family, replay_action_signature(replay))
        if replay_key in replay_signatures:
            counts["duplicate_replay_copy"] += 1
            continue
        replay_signatures.add(replay_key)
        counts["natural_replays"] += 1
        source_families[family] += 1
        root_cluster = canonical_ancestry_id(replay, path)
        ancestry = f"{family}:{root_cluster}"
        ancestry_family[ancestry] = family
        ancestry_cluster[ancestry] = root_cluster
        starter_value = replay.get("starter_tile", 1536)
        starter_tile = None if starter_value is None else int(starter_value)
        validator = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=starter_tile)
        frames = replay.get("frames")
        if not isinstance(frames, list):
            counts["missing_frames"] += 1
            continue
        replay_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        seen_state_signatures: set[str] = set()
        for fallback_index, frame in enumerate(frames):
            if not isinstance(frame, dict) or not isinstance(frame.get("state"), dict):
                counts["invalid_frame"] += 1
                continue
            state_payload = frame["state"]
            if bool(state_payload.get("game_over")):
                continue
            try:
                state = state_from_replay_payload(state_payload)
            except (KeyError, TypeError, ValueError):
                counts["invalid_state_restore"] += 1
                continue
            if not validator.legal_actions(state):
                counts["state_without_legal_action"] += 1
                continue
            signature = state_signature(state_payload, starter_tile)
            if signature in seen_state_signatures:
                counts["duplicate_state_within_ancestry"] += 1
                continue
            seen_state_signatures.add(signature)
            metadata = context_metadata(state, validator, starter_tile)
            bins = context_bins(metadata)
            cell = context_cell(bins)
            frame_index = int(frame.get("index", fallback_index))
            record_id = deterministic_key("A1", ancestry, frame_index, signature)[:20]
            record = {
                "record_id": record_id,
                "state": state_payload,
                "starter_tile": starter_tile,
                "source_replay": str(path),
                "source_replay_sha256": replay_hash,
                "source_frame_index": frame_index,
                "source_seed": replay.get("seed"),
                "source_policy": replay.get("policy"),
                "root_origin": provenance.get("root_origin"),
                "root_seed": provenance.get("root_seed"),
                "root_replay": provenance.get("root_replay") or str(path),
                "root_cluster": root_cluster,
                "ancestry_key": ancestry,
                "behavior_family": family,
                "context_bins": bins,
                "context_cell": cell,
                "context_metadata": metadata,
            }
            key = (ancestry, cell)
            existing = candidates.get(key)
            if existing is None or deterministic_key("A1-cell", record_id) < deterministic_key("A1-cell", existing["record_id"]):
                candidates[key] = record
            counts["valid_natural_states"] += 1

    candidate_by_ancestry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates.values():
        candidate_by_ancestry[str(record["ancestry_key"])].append(record)
    ancestries_by_cluster: dict[str, list[str]] = defaultdict(list)
    for ancestry in candidate_by_ancestry:
        ancestries_by_cluster[ancestry_cluster[ancestry]].append(ancestry)

    representative_by_cluster: dict[str, str] = {}
    excluded_aliases = 0
    for root_cluster, ancestries in ancestries_by_cluster.items():
        corner = [ancestry for ancestry in ancestries if ancestry_family[ancestry] == FAMILY_HOLDOUT]
        human = [ancestry for ancestry in ancestries if ancestry_family[ancestry] == "human_observed"]
        eligible = corner or human or ancestries
        representative = sorted(
            eligible,
            key=lambda ancestry: (
                -len(candidate_by_ancestry[ancestry]),
                ancestry_family[ancestry],
                ancestry,
            ),
        )[0]
        representative_by_cluster[root_cluster] = representative
        excluded_aliases += len(ancestries) - 1

    raw_partitions: dict[str, set[str]] = defaultdict(set)
    family_clusters: dict[str, list[str]] = defaultdict(list)
    for root_cluster, ancestry in representative_by_cluster.items():
        family = ancestry_family[ancestry]
        if family == FAMILY_HOLDOUT:
            raw_partitions["family_holdout"].add(ancestry)
        elif family == "human_observed":
            raw_partitions["human_diagnostic"].add(ancestry)
        else:
            family_clusters[family].append(root_cluster)
    split_counts = {}
    for family, root_clusters in sorted(family_clusters.items()):
        train_clusters, holdout_clusters = family_stratified_split(root_clusters)
        split_counts[family] = {
            "available_roots": len(root_clusters),
            "train_roots": len(train_clusters),
            "ancestry_holdout_roots": len(holdout_clusters),
        }
        for root_cluster in train_clusters:
            raw_partitions["train"].add(representative_by_cluster[root_cluster])
        for root_cluster in holdout_clusters:
            raw_partitions["ancestry_holdout"].add(representative_by_cluster[root_cluster])

    selected_records: list[dict[str, Any]] = []
    partition_summary = {}
    root_sets: dict[str, set[str]] = {}
    for partition in ("train", "ancestry_holdout", "family_holdout", "human_diagnostic"):
        records = round_robin_states(
            candidate_by_ancestry,
            raw_partitions.get(partition, set()),
            PARTITION_STATE_CAPS[partition],
        )
        for record in records:
            record["partition"] = partition
        weight_summary = assign_partition_weights(records, partition)
        selected_records.extend(records)
        partition_summary[partition] = {
            **coverage_summary(records),
            **weight_summary,
            "context_holes": context_holes(records),
        }
        root_sets[partition] = {str(record["root_cluster"]) for record in records}

    overlap = {}
    ordinary = ("train", "ancestry_holdout", "family_holdout")
    for index, left in enumerate(ordinary):
        for right in ordinary[index + 1 :]:
            shared = sorted(root_sets[left] & root_sets[right])
            overlap[f"{left}__{right}"] = shared

    train = partition_summary["train"]
    ancestry_holdout = partition_summary["ancestry_holdout"]
    family_holdout = partition_summary["family_holdout"]
    combined_holdout_records = [
        record for record in selected_records if record["partition"] in {"ancestry_holdout", "family_holdout"}
    ]
    combined_holdout = coverage_summary(combined_holdout_records)
    train_plus = train["root_counts"]["plus_bin"]
    holdout_plus = combined_holdout["root_counts"]["plus_bin"]
    readiness = {
        "train_min_150_ancestries": train["unique_ancestries"] >= 150,
        "train_min_4_families": len(train["root_family_counts"]) >= 4,
        "train_unweighted_ess_min_120": train["effective_ancestry_count"] >= 120.0,
        "train_weighted_ess_min_120": train["weighted_effective_ancestry_count"] >= 120.0,
        "train_max_effective_family_weight_0.40": train["maximum_effective_fit_family_weight"] <= MAX_FAMILY_SHARE + 1e-12,
        "ancestry_holdout_min_25_roots": ancestry_holdout["unique_ancestries"] >= 25,
        "ancestry_holdout_min_3_families": len(ancestry_holdout["root_family_counts"]) >= 3,
        "ancestry_holdout_ess_min_20": ancestry_holdout["effective_ancestry_count"] >= 20.0,
        "family_holdout_min_20_roots": family_holdout["unique_ancestries"] >= 20,
        "train_all_phase4_stages": len(train["root_counts"]["stage"]) == 4,
        "holdouts_all_phase4_stages": len(combined_holdout["root_counts"]["stage"]) == 4,
        "train_plus_bins_min_20_roots": all(train_plus.get(key, 0) >= 20 for key in ("zero", "lt_0.10", "0.10_0.25", "ge_0.25")),
        "holdout_plus_bins_min_5_roots": all(holdout_plus.get(key, 0) >= 5 for key in ("zero", "lt_0.10", "0.10_0.25", "ge_0.25")),
        "train_pending_strata_min_20_roots": len(train["root_counts"]["pending"]) == 2 and all(value >= 20 for value in train["root_counts"]["pending"].values()),
        "train_empties_bins_min_20_roots": len(train["root_counts"]["empties_bin"]) == 3 and all(value >= 20 for value in train["root_counts"]["empties_bin"].values()),
        "exact_state_provenance_failures_zero": not any(counts[key] for key in ("invalid_state_restore", "state_without_legal_action")),
        "cross_partition_root_overlap_zero": not any(overlap.values()),
    }
    ordinary_states = sum(
        1 for record in selected_records if record["partition"] in ordinary
    )
    trajectories = ordinary_states * 16
    estimated_wall_seconds = trajectories / (9600.0 / 6120.47811758297)
    estimated_storage_bytes = int(trajectories * (23 * 1024 * 1024 / 9600.0))
    selected_records.sort(key=lambda row: (str(row["partition"]), str(row["root_cluster"]), str(row["record_id"])))
    return {
        "inventory_version": A1_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "read_only_a1_no_labels_no_fitting",
        "amendment": "threes_rl/R15A_AMENDMENT_A1_20260711.md",
        "original_hold_data_lock": "threes_rl/runs/forensics/r15a_context/R15A_PREFLIGHT_STOP_GO.json",
        "selection_uses_outcome": False,
        "dashboard_eligible": False,
        "label_namespace_if_ready": A1_LABEL_NAMESPACE,
        "model_schema_sha256": schema_sha256(),
        "counts": dict(counts),
        "source_behavior_family_replays": dict(sorted(source_families.items())),
        "cross_family_root_aliases_removed": excluded_aliases,
        "family_stratified_split_counts": split_counts,
        "partitions": partition_summary,
        "combined_nonhuman_holdouts": combined_holdout,
        "cross_partition_root_overlap": overlap,
        "readiness_checks": readiness,
        "ready": all(readiness.values()),
        "synthetic_diagnostic_partition": {
            "manifest": "threes_rl/runs/eval_manifests/human_h2_context_12roots_4pairs_r16_20260711.json",
            "ordinary_metrics_eligible": False,
            "fitting_eligible": False,
        },
        "future_label_estimate": {
            "ordinary_states": ordinary_states,
            "replicates_per_state": 16,
            "horizon": 40,
            "trajectories": trajectories,
            "maximum_actor_decisions": trajectories * 40,
            "estimated_wall_seconds_at_measured_h0_throughput": estimated_wall_seconds,
            "estimated_wall_hours_at_measured_h0_throughput": estimated_wall_seconds / 3600.0,
            "estimated_compact_storage_bytes": estimated_storage_bytes,
        },
        "selected_records": selected_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = glob_replays(args.replay_glob or DEFAULT_REPLAY_GLOBS)
    payload = build_inventory_a1(paths)
    payload["source_paths"] = [str(path) for path in paths]
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "ready": payload["ready"],
                "readiness_checks": payload["readiness_checks"],
                "partitions": {
                    key: {
                        "states": value["states"],
                        "roots": value["unique_ancestries"],
                        "ess": value["effective_ancestry_count"],
                        "weighted_ess": value["weighted_effective_ancestry_count"],
                        "families": value["root_family_counts"],
                        "effective_family_weights": value["effective_fit_family_weights"],
                    }
                    for key, value in payload["partitions"].items()
                },
                "future_label_estimate": payload["future_label_estimate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

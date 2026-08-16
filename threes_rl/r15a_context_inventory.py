"""Read-only natural-state inventory and frozen partition manifest for R1.5a."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.context_residual import context_metadata, schema_sha256
from threes_rl.phase0_replay_coverage_inventory import replay_action_signature
from threes_rl.replay_provenance import GENUINE_ROOT_ORIGINS, replay_provenance
from threes_rl.restart_manifest import canonical_ancestry_id, replay_behavior_family, state_signature
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload


INVENTORY_VERSION = "r15a_context_inventory_v1"
FAMILY_HOLDOUT = "corner2_lineage"
MAX_FAMILY_SHARE = 0.40
MAX_STATES_PER_ANCESTRY = 8
PARTITION_STATE_CAPS = {
    "train": 1024,
    "ancestry_holdout": 256,
    "family_holdout": 256,
    "human_diagnostic": 64,
}
DEFAULT_REPLAY_GLOBS = (
    "threes_rl/runs/eval_artifacts/**/replay.json",
    "threes_rl/runs/*/top_games/**/replay.json",
    "threes_rl/runs/replays/**/replay.json",
    "datasets/human_play/**/replay.json",
)


def glob_replays(patterns: Iterable[str]) -> list[Path]:
    seen: set[str] = set()
    paths = []
    for pattern in patterns:
        for text in glob.glob(pattern, recursive=True):
            path = Path(text)
            key = str(path.resolve(strict=False))
            if key not in seen and path.is_file():
                seen.add(key)
                paths.append(path)
    return sorted(paths, key=str)


def coalesced_behavior_family(replay: dict[str, Any], path: Path) -> str:
    family = replay_behavior_family(replay, path)
    if family == "human_observed":
        return family
    if family == "td_student_lineage" or family == "ntuple" or family.startswith("train_td:"):
        return "legacy_learned_lineage"
    if family.startswith("ntuple"):
        return "legacy_ntuple_lineage"
    if family in {
        "phaseblend_incumbent_lineage",
        "phaseblend_cheap_lineage",
        "corner2_lineage",
        "expectimax_baseline",
        "random",
    }:
        return family
    return "legacy_learned_lineage"


def plus_bin(value: float) -> str:
    if value <= 0.0:
        return "zero"
    if value < 0.10:
        return "lt_0.10"
    if value < 0.25:
        return "0.10_0.25"
    return "ge_0.25"


def empties_bin(value: int) -> str:
    if value <= 1:
        return "0_1"
    if value <= 3:
        return "2_3"
    return "4_plus"


def bag_bin(value: int) -> str:
    if value <= 3:
        return "0_3"
    if value <= 7:
        return "4_7"
    return "8_11"


def context_bins(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": str(metadata["phase4_stage"]),
        "plus_bin": plus_bin(float(metadata["p_plus_next"])),
        "preview_bin": "bonus" if metadata["visible_preview_kind"] == "bonus" else "small",
        "pending": "pending" if metadata["post_visible_large_pending"] else "not_pending",
        "empties_bin": empties_bin(int(metadata["empty_count"])),
        "bag_bin": bag_bin(int(metadata["post_visible_small_pos"])),
    }


def context_cell(bins: dict[str, Any]) -> str:
    return "|".join(
        str(bins[key]) for key in ("stage", "plus_bin", "preview_bin", "pending", "empties_bin")
    )


def deterministic_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def ancestry_partition(family: str, ancestry: str) -> str:
    if family == "human_observed":
        return "human_diagnostic"
    if family == FAMILY_HOLDOUT:
        return "family_holdout"
    bucket = int(hashlib.sha256(ancestry.encode("utf-8")).hexdigest()[:16], 16) % 5
    return "ancestry_holdout" if bucket == 0 else "train"


def cap_family_share(ancestries: set[str], ancestry_family: dict[str, str]) -> tuple[set[str], list[str]]:
    selected = set(ancestries)
    removed: list[str] = []
    while selected:
        counts = Counter(ancestry_family[ancestry] for ancestry in selected)
        family, count = counts.most_common(1)[0]
        if count / len(selected) <= MAX_FAMILY_SHARE:
            break
        candidates = sorted(
            (ancestry for ancestry in selected if ancestry_family[ancestry] == family),
            key=lambda ancestry: deterministic_key("family-cap", ancestry),
            reverse=True,
        )
        selected.remove(candidates[0])
        removed.append(candidates[0])
    return selected, removed


def round_robin_states(
    candidate_by_ancestry: dict[str, list[dict[str, Any]]],
    selected_ancestries: set[str],
    cap: int,
) -> list[dict[str, Any]]:
    queues: dict[str, list[dict[str, Any]]] = {}
    cell_frequency = Counter(
        record["context_cell"]
        for ancestry in selected_ancestries
        for record in candidate_by_ancestry.get(ancestry, [])
    )
    for ancestry in selected_ancestries:
        records = sorted(
            candidate_by_ancestry.get(ancestry, []),
            key=lambda row: (
                cell_frequency[row["context_cell"]],
                deterministic_key("state-order", row["record_id"]),
            ),
        )[:MAX_STATES_PER_ANCESTRY]
        if records:
            queues[ancestry] = records
    ancestry_order = sorted(
        queues,
        key=lambda ancestry: (
            queues[ancestry][0]["behavior_family"],
            deterministic_key("ancestry-order", ancestry),
        ),
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < cap:
        progressed = False
        for ancestry in ancestry_order:
            queue = queues.get(ancestry, [])
            if queue:
                selected.append(queue.pop(0))
                progressed = True
                if len(selected) >= cap:
                    break
        if not progressed:
            break
    return selected


def effective_ancestry_count(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    counts = Counter(str(record["ancestry_key"]) for record in records)
    total = sum(counts.values())
    return float(1.0 / sum((count / total) ** 2 for count in counts.values()))


def coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    roots = {str(record["ancestry_key"]) for record in records}
    state_counts = {
        field: dict(sorted(Counter(str(record["context_bins"][field]) for record in records).items()))
        for field in ("stage", "plus_bin", "preview_bin", "pending", "empties_bin", "bag_bin")
    }
    root_counts = {}
    for field in ("stage", "plus_bin", "preview_bin", "pending", "empties_bin", "bag_bin"):
        groups: dict[str, set[str]] = defaultdict(set)
        for record in records:
            groups[str(record["context_bins"][field])].add(str(record["ancestry_key"]))
        root_counts[field] = {key: len(value) for key, value in sorted(groups.items())}
    families = Counter(str(record["behavior_family"]) for record in records)
    root_families: dict[str, set[str]] = defaultdict(set)
    for record in records:
        root_families[str(record["behavior_family"])].add(str(record["ancestry_key"]))
    return {
        "states": len(records),
        "unique_ancestries": len(roots),
        "effective_ancestry_count": effective_ancestry_count(records),
        "state_counts": state_counts,
        "root_counts": root_counts,
        "state_family_counts": dict(sorted(families.items())),
        "root_family_counts": {key: len(value) for key, value in sorted(root_families.items())},
        "context_cells": len({str(record["context_cell"]) for record in records}),
    }


def build_inventory(paths: list[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    replay_signatures: set[tuple[str, str]] = set()
    root_clusters_by_holdout: set[str] = set()
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    ancestry_family: dict[str, str] = {}
    ancestry_root_cluster: dict[str, str] = {}
    all_state_counts: Counter[str] = Counter()
    all_root_bins: dict[str, set[str]] = defaultdict(set)
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
        signature_key = (family, replay_action_signature(replay))
        if signature_key in replay_signatures:
            counts["duplicate_replay_copy"] += 1
            continue
        replay_signatures.add(signature_key)
        counts["natural_replays"] += 1
        source_families[family] += 1
        ancestry = f"{family}:{canonical_ancestry_id(replay, path)}"
        root_cluster = canonical_ancestry_id(replay, path)
        ancestry_family[ancestry] = family
        ancestry_root_cluster[ancestry] = root_cluster
        if family == FAMILY_HOLDOUT:
            root_clusters_by_holdout.add(root_cluster)
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
            all_state_counts[f"stage:{bins['stage']}"] += 1
            all_state_counts[f"plus:{bins['plus_bin']}"] += 1
            all_state_counts[f"pending:{bins['pending']}"] += 1
            all_state_counts[f"empties:{bins['empties_bin']}"] += 1
            all_root_bins[f"stage:{bins['stage']}"] .add(ancestry)
            all_root_bins[f"plus:{bins['plus_bin']}"] .add(ancestry)
            all_root_bins[f"pending:{bins['pending']}"] .add(ancestry)
            all_root_bins[f"empties:{bins['empties_bin']}"] .add(ancestry)
            frame_index = int(frame.get("index", fallback_index))
            record_id = deterministic_key(ancestry, frame_index, signature)[:20]
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
            if existing is None or deterministic_key("cell-state", record_id) < deterministic_key("cell-state", existing["record_id"]):
                candidates[key] = record
            counts["valid_natural_states"] += 1

    candidate_by_ancestry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates.values():
        candidate_by_ancestry[str(record["ancestry_key"])].append(record)

    raw_partitions: dict[str, set[str]] = defaultdict(set)
    heldout_collision_ancestries = []
    for ancestry in candidate_by_ancestry:
        family = ancestry_family[ancestry]
        partition = ancestry_partition(family, ancestry)
        if partition in {"train", "ancestry_holdout"} and ancestry_root_cluster[ancestry] in root_clusters_by_holdout:
            heldout_collision_ancestries.append(ancestry)
            continue
        raw_partitions[partition].add(ancestry)

    capped_partitions: dict[str, set[str]] = {}
    cap_removed: dict[str, list[str]] = {}
    for partition, ancestries in raw_partitions.items():
        if partition in {"train", "ancestry_holdout"}:
            selected, removed = cap_family_share(ancestries, ancestry_family)
            capped_partitions[partition] = selected
            cap_removed[partition] = removed
        else:
            capped_partitions[partition] = set(ancestries)
            cap_removed[partition] = []

    selected_records: list[dict[str, Any]] = []
    partition_summary = {}
    for partition in ("train", "ancestry_holdout", "family_holdout", "human_diagnostic"):
        records = round_robin_states(
            candidate_by_ancestry,
            capped_partitions.get(partition, set()),
            PARTITION_STATE_CAPS[partition],
        )
        for record in records:
            record["partition"] = partition
        selected_records.extend(records)
        partition_summary[partition] = coverage_summary(records)
        partition_summary[partition]["family_cap_removed_ancestries"] = len(cap_removed.get(partition, []))

    train_summary = partition_summary["train"]
    ancestry_summary = partition_summary["ancestry_holdout"]
    family_summary = partition_summary["family_holdout"]
    combined_holdout = [
        record for record in selected_records if record["partition"] in {"ancestry_holdout", "family_holdout"}
    ]
    combined_summary = coverage_summary(combined_holdout)
    train_root_families = train_summary["root_family_counts"]
    largest_train_share = (
        max(train_root_families.values()) / train_summary["unique_ancestries"]
        if train_summary["unique_ancestries"] else 0.0
    )
    train_plus = train_summary["root_counts"]["plus_bin"]
    holdout_plus = combined_summary["root_counts"]["plus_bin"]
    readiness = {
        "train_min_100_ancestries": train_summary["unique_ancestries"] >= 100,
        "train_min_3_families": len(train_root_families) >= 3,
        "train_max_family_share_0.40": largest_train_share <= MAX_FAMILY_SHARE,
        "ancestry_holdout_min_25_roots": ancestry_summary["unique_ancestries"] >= 25,
        "family_holdout_min_20_roots": family_summary["unique_ancestries"] >= 20,
        "train_all_phase4_stages": len(train_summary["root_counts"]["stage"]) == 4,
        "holdouts_all_phase4_stages": len(combined_summary["root_counts"]["stage"]) == 4,
        "train_plus_bins_min_20_roots": all(train_plus.get(key, 0) >= 20 for key in ("zero", "lt_0.10", "0.10_0.25", "ge_0.25")),
        "holdout_plus_bins_min_5_roots": all(holdout_plus.get(key, 0) >= 5 for key in ("zero", "lt_0.10", "0.10_0.25", "ge_0.25")),
        "train_pending_strata_min_20_roots": all(value >= 20 for value in train_summary["root_counts"]["pending"].values()) and len(train_summary["root_counts"]["pending"]) == 2,
        "train_empties_bins_min_20_roots": all(value >= 20 for value in train_summary["root_counts"]["empties_bin"].values()) and len(train_summary["root_counts"]["empties_bin"]) == 3,
        "exact_state_provenance_failures_zero": not any(
            counts[key]
            for key in ("invalid_state_restore", "state_without_legal_action")
        ),
    }
    ordinary_states = sum(
        1 for record in selected_records if record["partition"] in {"train", "ancestry_holdout", "family_holdout"}
    )
    trajectories = ordinary_states * 16
    estimated_wall_seconds = trajectories / (9600.0 / 6120.47811758297)
    estimated_storage_bytes = int(trajectories * (23 * 1024 * 1024 / 9600.0))
    selected_records.sort(key=lambda row: (str(row["partition"]), str(row["ancestry_key"]), str(row["record_id"])))
    return {
        "inventory_version": INVENTORY_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "read_only_preflight_no_labels_no_fitting",
        "selection_uses_outcome": False,
        "dashboard_eligible": False,
        "model_schema_sha256": schema_sha256(),
        "family_holdout": FAMILY_HOLDOUT,
        "max_family_share": MAX_FAMILY_SHARE,
        "partition_state_caps": PARTITION_STATE_CAPS,
        "max_states_per_ancestry": MAX_STATES_PER_ANCESTRY,
        "counts": dict(counts),
        "source_behavior_family_replays": dict(sorted(source_families.items())),
        "all_natural_state_counts": dict(sorted(all_state_counts.items())),
        "all_natural_root_counts": {key: len(value) for key, value in sorted(all_root_bins.items())},
        "heldout_root_cluster_collisions_removed": len(heldout_collision_ancestries),
        "partitions": partition_summary,
        "combined_nonhuman_holdouts": combined_summary,
        "largest_train_family_share": largest_train_share,
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
    payload = build_inventory(paths)
    payload["source_paths"] = [str(path) for path in paths]
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "source_files": len(paths),
                "ready": payload["ready"],
                "readiness_checks": payload["readiness_checks"],
                "partitions": {
                    key: {
                        "states": value["states"],
                        "unique_ancestries": value["unique_ancestries"],
                        "root_family_counts": value["root_family_counts"],
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

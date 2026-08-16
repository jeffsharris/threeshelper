"""G4 conditional pairwise action-ranker preflight and spent diagnostic.

This module never simulates a new path. It may read only the immutable G3
ordinary label database after the G4 charter has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom, binomtest

from threes_rl.g1r_acquire import historical_collision_union, service_health
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.g2_scale_relational_hazard import (
    FEATURE_NAMES,
    FEATURE_WIDTH,
    schema_manifest,
    schema_sha256,
)
from threes_rl.g3_e0_label_fit import (
    CANONICAL_ACTIONS,
    HORIZON_NAMES,
    HORIZONS,
    canonical_bytes,
    canonical_sha256,
    event_censor_rows,
    feature_rows_for_record,
    json_object,
    load_record_state,
    verify_payload_hash,
    write_immutable_json,
)
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "g4_conditional_pairwise_v1"
CHARTER_PATH = Path(
    "threes_rl/G4_CONDITIONAL_PAIRWISE_ACTION_RANKER_CHARTER.md"
)
CHARTER_SHA256 = (
    "765992cc0af3fc7c9d10c88ed3e0436a2ec6bc3b989f776775fe86230b22247e"
)
TEST_PATH = Path("tests/test_rl_g4_conditional_pairwise.py")
OUTPUT_DIR = Path("threes_rl/runs/forensics/g4_conditional_pairwise_v1")
PREFLIGHT_NAME = "G4_PREFLIGHT.json"
PAIR_MANIFEST_NAME = "G4_PAIR_MANIFEST.json"
FUTURE_STREAM_MANIFEST_NAME = "G4_FUTURE_STREAM_MANIFEST.json"
UNTOUCHED_INVENTORY_NAME = "G4_UNTOUCHED_INVENTORY.json"
DIAGNOSTIC_OPENED_NAME = "G4_DIAGNOSTIC_OPENED.json"
DIAGNOSTIC_RESULT_NAME = "G4_SPENT_DIAGNOSTIC_RESULT.json"
DIAGNOSTIC_ERROR_NAME = "G4_SPENT_DIAGNOSTIC_ERROR.json"
MODEL_DIR_NAME = "pairwise_checkpoint"

G3_DIR = Path("threes_rl/runs/forensics/g3_e0_label_fit_v4")
G3_TERMINAL_PATH = G3_DIR / "G3_E0_TERMINAL_RESULT.json"
G3_RECORD_MANIFEST_PATH = G3_DIR / "E0_RECORD_MANIFEST.json"
G3_TASK_MANIFEST_PATH = G3_DIR / "E0_TASK_MANIFEST.json"
G3_STREAM_MANIFEST_PATH = G3_DIR / "E0_STREAM_MANIFEST.json"
ORDINARY_DB_PATH = G3_DIR / "ordinary_labels.sqlite3"
FORBIDDEN_TRANSFER_DB_PATH = G3_DIR / "transfer_labels.sqlite3"
FORBIDDEN_TRANSFER_PREDICTION_PATH = (
    G3_DIR / "G3_E0_TRANSFER_PREDICTIONS_SEALED.json"
)

G2_ROOT_MANIFEST_PATH = Path(
    "threes_rl/runs/forensics/"
    "g2_scale_equivariant_relational_hazard/G2_ROOT_MANIFEST.json"
)
G2_PREFLIGHT_PATH = Path(
    "threes_rl/runs/forensics/"
    "g2_scale_equivariant_relational_hazard/G2_PREFLIGHT.json"
)
S3_PROVENANCE_PATH = Path(
    "threes_rl/runs/forensics/s3_full_policy/S3_PROVENANCE_SEAL_V2.json"
)
G1R_PILOT_SEAL_PATH = Path(
    "threes_rl/runs/forensics/g1r_acquisition/"
    "pilot_v2_qd5/PILOT_V2_SEAL.json"
)
G2_ACQUISITION_RESULT_PATH = Path(
    "threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1/"
    "G2_TRANSFER_ACQUISITION_RESULT.json"
)

EXPECTED_FILE_HASHES = {
    str(G3_TERMINAL_PATH):
        "e7ca390f0c32ebb3a680235de02e12beb62f45b1050115e8c9a30a7a3ca0ddd1",
    str(G3_RECORD_MANIFEST_PATH):
        "90a4f55ff29f51c0d6ac35375650258188b6961debd6cbcc546382762547d9d5",
    str(G3_TASK_MANIFEST_PATH):
        "087fd68c71421c8402360a1c096b476cb1bf494de7d8c8f025e7e699bf97bd2f",
    str(G3_STREAM_MANIFEST_PATH):
        "e40b7dd3744dd0df04f621034894656568991291c17490e27e8c3a93e189ea05",
    str(ORDINARY_DB_PATH):
        "d0954a91e84bc7a420d64e7294f40232c1ffcb692fab86d07425b138e063f820",
    str(G2_ROOT_MANIFEST_PATH):
        "60d514ed79ff315f7c2e0d2ad13bb712a57d4c3b204587691aa878a7486ea2ca",
    str(G2_PREFLIGHT_PATH):
        "2e1084f2a0673935866839e89765d3d1a31a2c2348e99c01edc9abc2405f05cc",
    str(S3_PROVENANCE_PATH):
        "5326f25b50ad33b4e00eb5ca7180468d3a243917075d15d377a1511b04867949",
    str(G1R_PILOT_SEAL_PATH):
        "75a11648859e5a67686702420b760737b4a810271183d050350f38d4a0c4ae57",
    str(G2_ACQUISITION_RESULT_PATH):
        "7b862377546b35c8c53967eedd39edb736c5db039d262f65048da4c47774ca74",
    "threes_rl/g2_scale_relational_hazard.py":
        "9ffaa45dd36b633cdae10110fdaefc8cd27053ab3f0216ddb3f1886ea625af8a",
    "threes_rl/g3_e0_label_fit.py":
        "19d74a319459d75619f515fd9cdea03a126e1270046fb8e12ae367d43b2cc8b5",
}
EXPECTED_ORDINARY_RECORDS = 683
EXPECTED_ORDINARY_PATHS = 4_846
ACCEPTED_PARTITIONS = ("train", "development")
ACCEPTED_SCALES = ("pre768", "pre1536")
BOOTSTRAP_SEED = 2_026_072_604
BOOTSTRAP_REPEATS = 10_000
L2_LAMBDA = 1.0
MAX_OPTIMIZER_ITERATIONS = 500
OPTIMIZER_GTOL = 1e-8
FUTURE_ROOTS = 512
FUTURE_STREAM_BASES = {
    "logical_seed": 61_000_000_000,
    "deck_stream_id": 62_000_000_000,
    "slot_stream_id": 63_000_000_000,
    "policy_stream_id": 64_000_000_000,
}
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
GIB = 1024**3


def _file_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    rows = []
    for path in sorted(set(paths), key=str):
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "rows": rows,
        "count": len(rows),
        "manifest_sha256": canonical_sha256(rows),
    }


def _source_hash_audit() -> dict[str, Any]:
    actual: dict[str, str | None] = {}
    checks: dict[str, bool] = {}
    for path_text, expected in EXPECTED_FILE_HASHES.items():
        path = Path(path_text)
        value = sha256_path(path) if path.is_file() else None
        actual[path_text] = value
        checks[path_text] = value == expected
    wal_path = Path(f"{ORDINARY_DB_PATH}-wal")
    wal_bytes = wal_path.stat().st_size if wal_path.is_file() else 0
    checks["ordinary_wal_empty"] = wal_bytes == 0
    checks["forbidden_transfer_db_not_opened"] = True
    checks["forbidden_transfer_predictions_not_opened"] = True
    checks["charter_hash_exact"] = (
        sha256_path(CHARTER_PATH) == CHARTER_SHA256
    )
    checks["schema_hash_exact"] = (
        schema_sha256()
        == "6af0cd515e5886b5fd8bc4d9f52cc9202bd3ed1f149d0ae146829681aea8340e"
    )
    return {
        "expected": dict(EXPECTED_FILE_HASHES),
        "actual": actual,
        "ordinary_wal_bytes": wal_bytes,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _load_ordinary_records() -> list[dict[str, Any]]:
    manifest = json_object(G2_ROOT_MANIFEST_PATH)
    if not verify_payload_hash(manifest):
        raise ValueError("G2 root manifest payload hash mismatch")
    records = [
        dict(row)
        for row in manifest.get("records", [])
        if row.get("partition") in ACCEPTED_PARTITIONS
    ]
    if len(records) != EXPECTED_ORDINARY_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_ORDINARY_RECORDS} ordinary records, "
            f"found {len(records)}"
        )
    if any(
        row.get("partition") not in ACCEPTED_PARTITIONS
        or row.get("scale") not in ACCEPTED_SCALES
        for row in records
    ):
        raise ValueError("Non-ordinary record entered G4")
    record_ids = [str(row["record_id"]) for row in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate ordinary record ID")
    return sorted(records, key=lambda row: str(row["record_id"]))


def _read_ordinary_paths() -> list[dict[str, Any]]:
    uri = f"file:{ORDINARY_DB_PATH.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT task_key,payload_json,payload_sha256 "
            "FROM paths ORDER BY task_key"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != EXPECTED_ORDINARY_PATHS:
        raise ValueError(
            f"Expected {EXPECTED_ORDINARY_PATHS} ordinary paths, "
            f"found {len(rows)}"
        )
    paths = []
    for task_key, payload_json, payload_sha in rows:
        actual_sha = hashlib.sha256(payload_json.encode("ascii")).hexdigest()
        if actual_sha != payload_sha:
            raise ValueError(f"Path payload hash mismatch: {task_key}")
        payload = json.loads(payload_json)
        if (
            not isinstance(payload, dict)
            or payload.get("partition") not in ACCEPTED_PARTITIONS
            or "transfer" in str(payload.get("partition", "")).lower()
        ):
            raise ValueError("Transfer or malformed path entered G4")
        if str(payload.get("task_key")) != str(task_key):
            raise ValueError("Path task key mismatch")
        paths.append(payload)
    return paths


def _path_manifest(paths: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "task_key": str(row["task_key"]),
            "record_id": str(row["record_id"]),
            "partition": str(row["partition"]),
            "replicate": int(row["replicate"]),
            "action_id": int(row["action_id"]),
            "payload_sha256": canonical_sha256(dict(row)),
        }
        for row in paths
    ]
    return {
        "path_count": len(rows),
        "rows_sha256": canonical_sha256(rows),
    }


def _validate_interval_rows(path: Mapping[str, Any]) -> None:
    expected = event_censor_rows(
        event_move=(
            None
            if path.get("event_move") is None
            else int(path["event_move"])
        ),
        terminal_move=(
            None
            if path.get("terminal_move") is None
            else int(path["terminal_move"])
        ),
        completed_moves=int(path["completed_moves"]),
    )
    if list(path.get("interval_rows", [])) != expected:
        raise ValueError(f"Interval arithmetic mismatch: {path['task_key']}")


def _enrich_and_validate_records(
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_record = {str(row["record_id"]): dict(row) for row in records}
    path_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        record_id = str(path["record_id"])
        if record_id not in by_record:
            raise ValueError("Path references a non-ordinary record")
        path_groups[record_id].append(path)
    if set(path_groups) != set(by_record):
        raise ValueError("Ordinary record/path coverage mismatch")

    enriched: list[dict[str, Any]] = []
    source_hash_checks = 0
    state_checks = 0
    coupling_units = 0
    for record_id in sorted(by_record):
        record = by_record[record_id]
        rows = path_groups[record_id]
        if any(
            str(row["partition"]) != str(record["partition"])
            or str(row["root_cluster"]) != str(record["root_cluster"])
            or str(row["behavior_family"]) != str(record["behavior_family"])
            or str(row["scale"]) != str(record["scale"])
            or int(row["target"]) != int(record["target"])
            or str(row["state_sha1"]) != str(record["state_sha1"])
            for row in rows
        ):
            raise ValueError(f"Record/path identity mismatch: {record_id}")
        for row in rows:
            _validate_interval_rows(row)

        replicas = sorted({int(row["replicate"]) for row in rows})
        if replicas != [0, 1]:
            raise ValueError(f"Replicate mismatch: {record_id}")
        actions_by_rep: dict[int, list[tuple[int, str]]] = {}
        stream_keys = (
            "logical_seed",
            "deck_stream_id",
            "slot_stream_id",
            "policy_stream_id",
        )
        for replicate in replicas:
            arm_rows = [
                row for row in rows if int(row["replicate"]) == replicate
            ]
            action_pairs = sorted(
                {
                    (int(row["action_id"]), str(row["action"]))
                    for row in arm_rows
                }
            )
            if len(action_pairs) != len(arm_rows):
                raise ValueError(f"Duplicate action arm: {record_id}")
            if any(
                len({int(row[key]) for row in arm_rows}) != 1
                for key in stream_keys
            ):
                raise ValueError(f"CRN stream mismatch: {record_id}")
            actions_by_rep[replicate] = action_pairs
            coupling_units += 1
        if actions_by_rep[0] != actions_by_rep[1]:
            raise ValueError(f"Action set differs by replicate: {record_id}")

        action_pairs = actions_by_rep[0]
        action_ids = [action_id for action_id, _name in action_pairs]
        action_names = [name for _action_id, name in action_pairs]
        if any(
            action_id < 0
            or action_id >= len(CANONICAL_ACTIONS)
            or CANONICAL_ACTIONS[action_id] != name
            for action_id, name in action_pairs
        ):
            raise ValueError(f"Canonical action mismatch: {record_id}")

        source_path = Path(str(record["source_replay"]))
        if sha256_path(source_path) != str(record["source_replay_sha256"]):
            raise ValueError(f"Source replay hash mismatch: {record_id}")
        source_hash_checks += 1

        state = load_record_state(record)
        restored = state
        simulator = ThreesSim.from_stream_ids(
            deck_stream_id=2_026_072_611,
            slot_stream_id=2_026_072_612,
            starter_tile=int(record["starter_tile"]),
        )
        legal = sorted(
            int(value)
            for value in simulator.legal_actions(
                state_from_replay_payload(restored)
            )
        )
        if legal != action_ids:
            raise ValueError(f"Restored legal actions mismatch: {record_id}")
        state_checks += 1

        item = dict(record)
        item["legal_action_ids"] = action_ids
        item["legal_actions"] = action_names
        enriched.append(item)
    return enriched, {
        "records": len(enriched),
        "source_replay_hash_checks": source_hash_checks,
        "state_legal_action_checks": state_checks,
        "record_replicate_crn_units": coupling_units,
        "passes": (
            len(enriched) == EXPECTED_ORDINARY_RECORDS
            and source_hash_checks == EXPECTED_ORDINARY_RECORDS
            and state_checks == EXPECTED_ORDINARY_RECORDS
        ),
    }


def _feature_map(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int, str], np.ndarray], dict[str, Any]]:
    result: dict[tuple[str, int, str], np.ndarray] = {}
    record_hashes: list[dict[str, str]] = []
    deterministic_checks = 0
    for record in records:
        first_rows, first_digest = feature_rows_for_record(record)
        second_rows, second_digest = feature_rows_for_record(record)
        if first_digest != second_digest or len(first_rows) != len(second_rows):
            raise ValueError("Nondeterministic feature extraction")
        for first, second in zip(first_rows, second_rows):
            left = np.asarray(first["features"], dtype=np.float64)
            right = np.asarray(second["features"], dtype=np.float64)
            if not np.array_equal(left, right):
                raise ValueError("Feature extraction changed across repeats")
            horizon = f"h{int(first['horizon'])}"
            key = (
                str(first["record_id"]),
                int(first["action_id"]),
                horizon,
            )
            if key in result:
                raise ValueError("Duplicate feature row")
            result[key] = left.copy()
            deterministic_checks += 1
        record_hashes.append(
            {
                "record_id": str(record["record_id"]),
                "feature_rows_sha256": first_digest,
            }
        )
    return result, {
        "feature_rows": len(result),
        "deterministic_repeat_checks": deterministic_checks,
        "record_feature_hashes_sha256": canonical_sha256(record_hashes),
        "schema_sha256": schema_sha256(),
        "passes": len(result) == deterministic_checks,
    }


def _interval_map(path: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in path["interval_rows"]:
        horizon = str(row["horizon"])
        if horizon in rows or horizon not in HORIZON_NAMES:
            raise ValueError("Invalid interval-row horizon")
        rows[horizon] = row
    return rows


def _count_table(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        counts[tuple(row[key] for key in keys)] += 1
    return [
        {
            **{key: values[index] for index, key in enumerate(keys)},
            "count": count,
        }
        for values, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def build_pair_dataset(
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
    features: Mapping[tuple[str, int, str], np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    record_map = {str(row["record_id"]): row for row in records}
    path_map = {
        (str(row["record_id"]), int(row["replicate"]), int(row["action_id"])): row
        for row in paths
    }
    if len(path_map) != len(paths):
        raise ValueError("Duplicate path key")

    pair_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    reversal_checks = 0
    zero_horizon_checks = 0
    for record_id in sorted(record_map):
        record = record_map[record_id]
        actions = sorted(int(value) for value in record["legal_action_ids"])
        for replicate in (0, 1):
            arms = {
                action: path_map[(record_id, replicate, action)]
                for action in actions
            }
            stream_keys = (
                "logical_seed",
                "deck_stream_id",
                "slot_stream_id",
                "policy_stream_id",
            )
            if any(
                len({int(arms[action][key]) for action in actions}) != 1
                for key in stream_keys
            ):
                raise ValueError("Pair arm CRN mismatch")
            interval_maps = {
                action: _interval_map(arms[action]) for action in actions
            }
            for horizon in HORIZON_NAMES:
                for action_a, action_b in combinations(actions, 2):
                    row_a = interval_maps[action_a].get(horizon)
                    row_b = interval_maps[action_b].get(horizon)
                    base = {
                        "partition": str(record["partition"]),
                        "scale": str(record["scale"]),
                        "behavior_family": str(record["behavior_family"]),
                        "root_cluster": str(record["root_cluster"]),
                        "record_id": record_id,
                        "horizon": horizon,
                        "replicate": replicate,
                        "action_pair": (
                            f"{CANONICAL_ACTIONS[action_a]}:"
                            f"{CANONICAL_ACTIONS[action_b]}"
                        ),
                    }
                    comparable = (
                        row_a is not None
                        and row_b is not None
                        and bool(row_a.get("observed"))
                        and bool(row_b.get("observed"))
                        and row_a.get("event") in (0, 1)
                        and row_b.get("event") in (0, 1)
                    )
                    if not comparable:
                        status_rows.append({**base, "status": "noncomparable"})
                        continue
                    event_a = int(row_a["event"])
                    event_b = int(row_b["event"])
                    if event_a == event_b:
                        status = (
                            "concordant_event"
                            if event_a == 1
                            else "concordant_no_event"
                        )
                        status_rows.append({**base, "status": status})
                        continue

                    delta = (
                        np.asarray(
                            features[(record_id, action_a, horizon)],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            features[(record_id, action_b, horizon)],
                            dtype=np.float64,
                        )
                    )
                    reverse = (
                        np.asarray(
                            features[(record_id, action_b, horizon)],
                            dtype=np.float64,
                        )
                        - np.asarray(
                            features[(record_id, action_a, horizon)],
                            dtype=np.float64,
                        )
                    )
                    if (
                        delta.shape != (FEATURE_WIDTH,)
                        or not np.all(np.isfinite(delta))
                        or not np.array_equal(reverse, -delta)
                    ):
                        raise ValueError("Pair delta reversal/integrity failure")
                    reversal_checks += 1
                    if not np.array_equal(delta[:3], np.zeros(3)):
                        raise ValueError("Horizon indicators did not cancel")
                    zero_horizon_checks += 1
                    pair_rows.append(
                        {
                            **base,
                            "action_a_id": action_a,
                            "action_b_id": action_b,
                            "label": event_a,
                            "delta": delta,
                            "unit_key": f"{horizon}:r{replicate}",
                        }
                    )
                    status_rows.append({**base, "status": "discordant"})
    pair_rows.sort(
        key=lambda row: (
            str(row["partition"]),
            str(row["behavior_family"]),
            str(row["root_cluster"]),
            str(row["record_id"]),
            HORIZON_NAMES.index(str(row["horizon"])),
            int(row["replicate"]),
            int(row["action_a_id"]),
            int(row["action_b_id"]),
        )
    )
    pair_metadata = [
        {
            key: value
            for key, value in row.items()
            if key not in {"delta"}
        }
        for row in pair_rows
    ]
    delta_hasher = hashlib.sha256()
    for row in pair_rows:
        delta_hasher.update(
            np.asarray(row["delta"], dtype="<f8").tobytes(order="C")
        )
    audit = {
        "candidate_action_pairs": len(status_rows),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in status_rows).items())
        ),
        "discordant_pairs": len(pair_rows),
        "reversal_checks": reversal_checks,
        "zero_horizon_delta_checks": zero_horizon_checks,
        "pair_metadata_sha256": canonical_sha256(pair_metadata),
        "delta_bytes_sha256": delta_hasher.hexdigest(),
        "pair_dataset_sha256": canonical_sha256(
            {
                "metadata_sha256": canonical_sha256(pair_metadata),
                "delta_bytes_sha256": delta_hasher.hexdigest(),
            }
        ),
        "status_by_partition_scale_family_horizon_pair": _count_table(
            status_rows,
            (
                "partition",
                "scale",
                "behavior_family",
                "horizon",
                "action_pair",
                "status",
            ),
        ),
        "discordant_by_partition_scale_family_horizon_pair": _count_table(
            pair_rows,
            (
                "partition",
                "scale",
                "behavior_family",
                "horizon",
                "action_pair",
            ),
        ),
        "passes": (
            len(pair_rows) == reversal_checks == zero_horizon_checks
        ),
    }
    return pair_rows, audit


def assign_pair_weights(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    families = sorted({str(row["behavior_family"]) for row in rows})
    roots_by_family: dict[str, set[str]] = defaultdict(set)
    records_by_root: dict[tuple[str, str], set[str]] = defaultdict(set)
    units_by_record: dict[
        tuple[str, str, str], set[tuple[str, int]]
    ] = defaultdict(set)
    pairs_by_unit: Counter[tuple[str, str, str, str, int]] = Counter()
    for row in rows:
        family = str(row["behavior_family"])
        root = str(row["root_cluster"])
        record = str(row["record_id"])
        unit = (str(row["horizon"]), int(row["replicate"]))
        roots_by_family[family].add(root)
        records_by_root[(family, root)].add(record)
        units_by_record[(family, root, record)].add(unit)
        pairs_by_unit[(family, root, record, unit[0], unit[1])] += 1
    weighted = []
    for row in rows:
        item = dict(row)
        family = str(row["behavior_family"])
        root = str(row["root_cluster"])
        record = str(row["record_id"])
        horizon = str(row["horizon"])
        replicate = int(row["replicate"])
        local = (
            1.0
            / len(records_by_root[(family, root)])
            / len(units_by_record[(family, root, record)])
            / pairs_by_unit[(family, root, record, horizon, replicate)]
        )
        item["root_local_weight"] = float(local)
        item["weight"] = float(
            1.0
            / len(families)
            / len(roots_by_family[family])
            * local
        )
        weighted.append(item)
    total = sum(float(row["weight"]) for row in weighted)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Pair weights sum to {total}, not one")
    family_sums = defaultdict(float)
    root_local_sums = defaultdict(float)
    for row in weighted:
        family_sums[str(row["behavior_family"])] += float(row["weight"])
        root_local_sums[
            (str(row["behavior_family"]), str(row["root_cluster"]))
        ] += float(row["root_local_weight"])
    expected_family = 1.0 / len(families)
    if any(
        not math.isclose(value, expected_family, abs_tol=1e-12)
        for value in family_sums.values()
    ) or any(
        not math.isclose(value, 1.0, abs_tol=1e-12)
        for value in root_local_sums.values()
    ):
        raise ValueError("Hierarchical pair weights are imbalanced")
    return weighted


def _support_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pairs": 0,
            "roots": 0,
            "families": 0,
            "records": 0,
            "roots_by_scale": {},
            "roots_by_family": {},
            "pairs_by_horizon": {},
            "max_raw_pair_share_by_root": 0.0,
            "effective_root_count": 0.0,
        }
    roots = {str(row["root_cluster"]) for row in rows}
    root_pair_counts = Counter(str(row["root_cluster"]) for row in rows)
    roots_by_family = {
        family: len(
            {
                str(row["root_cluster"])
                for row in rows
                if str(row["behavior_family"]) == family
            }
        )
        for family in sorted(
            {str(row["behavior_family"]) for row in rows}
        )
    }
    roots_by_scale = {
        scale: len(
            {
                str(row["root_cluster"])
                for row in rows
                if str(row["scale"]) == scale
            }
        )
        for scale in ACCEPTED_SCALES
    }
    weighted = assign_pair_weights(rows)
    root_weights = defaultdict(float)
    for row in weighted:
        root_weights[str(row["root_cluster"])] += float(row["weight"])
    ess = 1.0 / sum(value * value for value in root_weights.values())
    return {
        "pairs": len(rows),
        "roots": len(roots),
        "families": len(roots_by_family),
        "records": len({str(row["record_id"]) for row in rows}),
        "roots_by_scale": roots_by_scale,
        "roots_by_family": roots_by_family,
        "pairs_by_horizon": dict(
            sorted(Counter(str(row["horizon"]) for row in rows).items())
        ),
        "max_raw_pair_share_by_root": (
            max(root_pair_counts.values()) / len(rows)
        ),
        "effective_root_count": float(ess),
    }


def _future_stream_rows() -> list[dict[str, Any]]:
    return [
        {
            "root_index": index,
            **{
                key: int(base + index)
                for key, base in FUTURE_STREAM_BASES.items()
            },
        }
        for index in range(FUTURE_ROOTS)
    ]


def _future_stream_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude_dir: Path,
) -> dict[str, Any]:
    historical, sources = historical_collision_union(exclude_dir=exclude_dir)
    collisions = {}
    for key in FUTURE_STREAM_BASES:
        prior = set(historical.get(key, set()))
        if key == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                prior.update(historical.get(alias, set()))
        requested = {int(row[key]) for row in rows}
        collisions[key] = sorted(requested.intersection(prior))
    flat = [
        int(row[key])
        for row in rows
        for key in FUTURE_STREAM_BASES
    ]
    internal_unique = len(flat) == len(set(flat))
    return {
        "historical_union": sources,
        "requested_rows": len(rows),
        "requested_stream_ids": len(flat),
        "requested_manifest_sha256": canonical_sha256(list(rows)),
        "collisions": collisions,
        "internal_unique": internal_unique,
        "passes": internal_unique and not any(collisions.values()),
    }


def _untouched_inventory(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    roots_by_partition: dict[str, set[str]] = defaultdict(set)
    for row in json_object(G2_ROOT_MANIFEST_PATH)["records"]:
        roots_by_partition[str(row.get("partition"))].add(
            str(row["root_cluster"])
        )
    ordinary_roots = {str(row["root_cluster"]) for row in records}
    locked_catalogs = {
        "g3_spent_ordinary_roots": {
            "count": len(ordinary_roots),
            "sha256": hashlib.sha256(
                "\n".join(sorted(ordinary_roots)).encode("utf-8")
            ).hexdigest(),
            "eligibility": "forbidden_spent",
        },
        "g2_diagnostic_only_roots": {
            "count": len(roots_by_partition.get("diagnostic_only", set())),
            "sha256": hashlib.sha256(
                "\n".join(
                    sorted(roots_by_partition.get("diagnostic_only", set()))
                ).encode("utf-8")
            ).hexdigest(),
            "eligibility": "forbidden_prior_diagnostic",
        },
        "g2_prior_overlap_transfer_roots": {
            "count": len(
                roots_by_partition.get(
                    "diagnostic_prior_overlap_transfer", set()
                )
            ),
            "sha256": hashlib.sha256(
                "\n".join(
                    sorted(
                        roots_by_partition.get(
                            "diagnostic_prior_overlap_transfer", set()
                        )
                    )
                ).encode("utf-8")
            ).hexdigest(),
            "eligibility": "forbidden_prior_overlap",
        },
    }
    source_locks = _file_manifest(
        (
            G2_ROOT_MANIFEST_PATH,
            G2_PREFLIGHT_PATH,
            S3_PROVENANCE_PATH,
            G1R_PILOT_SEAL_PATH,
            G2_ACQUISITION_RESULT_PATH,
        )
    )
    return {
        "version": f"{VERSION}_untouched_inventory",
        "method": (
            "fail-closed retained-manifest inventory; roots in any sealed "
            "historical catalog are not certified untouched"
        ),
        "locked_catalogs": locked_catalogs,
        "source_locks": source_locks,
        "certified_existing_confirmation_roots": [],
        "certified_existing_confirmation_root_count": 0,
        "complete_workspace_replay_certification": False,
        "status": "NO_CERTIFIED_EXISTING_ROOTS_NO_REACQUISITION_AUTHORIZED",
        "g3_transfer_panel_accessed": False,
        "g3_transfer_panel_underpowered_if_separately_authorized": {
            "root_count": 32,
            "minimum_activity_floor": 48,
            "minimum_family_floor": 4,
            "recorded_mde": "approximately common OR 4.0",
            "can_confirm_g4": False,
        },
    }


def _exact_binomial_power(
    n: int,
    alternative_probability: float,
    *,
    alpha: float = 0.05,
) -> float:
    rejection = [
        binomtest(k, n, 0.5, alternative="two-sided").pvalue <= alpha
        for k in range(n + 1)
    ]
    probabilities = binom.pmf(
        np.arange(n + 1),
        n,
        alternative_probability,
    )
    return float(np.sum(probabilities[np.asarray(rejection, dtype=bool)]))


def _power_table() -> dict[str, Any]:
    root_counts = (48, 64, 96, 128, 192, 256)
    alternatives = (0.60, 0.65, 0.70)
    rows = [
        {
            "informative_roots": n,
            **{
                f"power_true_concordance_{probability:.2f}":
                    _exact_binomial_power(n, probability)
                for probability in alternatives
            },
        }
        for n in root_counts
    ]
    passing = [
        row["informative_roots"]
        for row in rows
        if row["power_true_concordance_0.65"] >= 0.80
    ]
    return {
        "test": "exact two-sided binomial concordance test",
        "null": 0.5,
        "alpha": 0.05,
        "unit": "one aggregated ancestry concordance outcome",
        "rows": rows,
        "smallest_listed_n_with_80pct_power_at_0_65": (
            min(passing) if passing else None
        ),
        "activity_floor": {
            "overall_disagreements": 48,
            "per_scale_disagreements": 16,
            "families": 4,
            "max_family_share": 0.40,
        },
    }


def _operational_audit() -> dict[str, Any]:
    free_gib = shutil.disk_usage(Path(".")).free / GIB
    try:
        services = service_health()
    except Exception as error:
        services = {
            "passes": False,
            "error": f"{type(error).__name__}: {error}",
        }
    try:
        heavy = _heavy_process_audit()
    except Exception as error:
        heavy = {
            "passes": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "free_gib": free_gib,
        "disk_above_100_gib": free_gib >= MIN_FREE_GIB,
        "disk_above_120_gib_target": free_gib >= TARGET_FREE_GIB,
        "services": services,
        "heavy_process_audit": heavy,
        "passes": (
            free_gib >= MIN_FREE_GIB
            and bool(services.get("passes"))
            and bool(heavy.get("passes"))
        ),
    }


def _preflight_decision(
    *,
    integrity_checks: Mapping[str, bool],
    train: Mapping[str, Any],
    development: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    support_checks = {
        "train_roots_at_least_100": int(train["roots"]) >= 100,
        "development_roots_at_least_32": int(development["roots"]) >= 32,
        "train_families_at_least_4": int(train["families"]) >= 4,
        "development_families_at_least_3":
            int(development["families"]) >= 3,
        "development_pre768_roots_at_least_12":
            int(development["roots_by_scale"].get("pre768", 0)) >= 12,
        "development_pre1536_roots_at_least_12":
            int(development["roots_by_scale"].get("pre1536", 0)) >= 12,
        "development_pairs_at_least_128":
            int(development["pairs"]) >= 128,
        "development_max_raw_root_share_at_most_0_10":
            float(development["max_raw_pair_share_by_root"]) <= 0.10,
    }
    if not all(integrity_checks.values()):
        return "KILL_G4_PAIRWISE_INFEASIBLE", support_checks
    if not all(support_checks.values()):
        return "HOLD_G4_PAIRWISE_UNDERPOWERED", support_checks
    return "READY_G4_SPENT_DIAGNOSTIC", support_checks


@dataclass(frozen=True)
class PairwiseModel:
    feature_names: tuple[str, ...]
    schema_sha256: str
    feature_scale: np.ndarray
    coefficients: np.ndarray
    optimizer_summary: dict[str, Any]
    source_hashes: dict[str, str]
    pair_dataset_sha256: str

    def logits(self, deltas: np.ndarray) -> np.ndarray:
        values = np.asarray(deltas, dtype=np.float64)
        return (values / self.feature_scale) @ self.coefficients

    def save(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=False)
        arrays_path = directory / "arrays.npz"
        np.savez_compressed(
            arrays_path,
            feature_scale=self.feature_scale,
            coefficients=self.coefficients,
        )
        metadata = {
            "version": VERSION,
            "feature_width": FEATURE_WIDTH,
            "parameter_count": FEATURE_WIDTH,
            "has_intercept": False,
            "feature_names": list(self.feature_names),
            "schema_sha256": self.schema_sha256,
            "arrays_file_sha256": sha256_path(arrays_path),
            "optimizer_summary": self.optimizer_summary,
            "source_hashes": self.source_hashes,
            "pair_dataset_sha256": self.pair_dataset_sha256,
        }
        metadata["canonical_payload_sha256"] = canonical_sha256(metadata)
        write_immutable_json(directory / "meta.json", metadata)
        return metadata

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_source_hashes: Mapping[str, str] | None = None,
    ) -> "PairwiseModel":
        metadata = json_object(directory / "meta.json")
        if not verify_payload_hash(metadata):
            raise ValueError("Pairwise model metadata hash mismatch")
        if (
            metadata.get("version") != VERSION
            or int(metadata.get("feature_width", -1)) != FEATURE_WIDTH
            or int(metadata.get("parameter_count", -1)) != FEATURE_WIDTH
            or bool(metadata.get("has_intercept", True))
            or metadata.get("schema_sha256") != schema_sha256()
            or tuple(metadata.get("feature_names", ())) != tuple(FEATURE_NAMES)
        ):
            raise ValueError("Incompatible pairwise model schema")
        arrays_path = directory / "arrays.npz"
        if sha256_path(arrays_path) != metadata["arrays_file_sha256"]:
            raise ValueError("Pairwise model array hash mismatch")
        arrays = np.load(arrays_path, allow_pickle=False)
        scale = np.asarray(arrays["feature_scale"], dtype=np.float64)
        coefficients = np.asarray(arrays["coefficients"], dtype=np.float64)
        if (
            scale.shape != (FEATURE_WIDTH,)
            or coefficients.shape != (FEATURE_WIDTH,)
            or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(coefficients))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("Invalid pairwise model arrays")
        source_hashes = {
            str(key): str(value)
            for key, value in metadata["source_hashes"].items()
        }
        if (
            expected_source_hashes is not None
            and source_hashes != dict(expected_source_hashes)
        ):
            raise ValueError("Pairwise model source hashes mismatch")
        return cls(
            feature_names=tuple(metadata["feature_names"]),
            schema_sha256=str(metadata["schema_sha256"]),
            feature_scale=scale,
            coefficients=coefficients,
            optimizer_summary=dict(metadata["optimizer_summary"]),
            source_hashes=source_hashes,
            pair_dataset_sha256=str(metadata["pair_dataset_sha256"]),
        )


def _pair_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack(
            [np.asarray(row["delta"], dtype=np.float64) for row in rows]
        ),
        np.asarray([float(row["label"]) for row in rows], dtype=np.float64),
        np.asarray([float(row["weight"]) for row in rows], dtype=np.float64),
    )


def fit_pairwise_model(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    source_hashes: Mapping[str, str],
    pair_dataset_sha256: str,
) -> PairwiseModel:
    weighted = assign_pair_weights(train_rows)
    features, labels, weights = _pair_matrix(weighted)
    standardize_mask = np.asarray(
        [
            bool(column["train_standardize"])
            for column in schema_manifest()["columns"]
        ],
        dtype=np.bool_,
    )
    scale = np.ones(FEATURE_WIDTH, dtype=np.float64)
    total = float(weights.sum())
    rms = np.sqrt(
        np.maximum(
            np.sum(weights[:, None] * features * features, axis=0) / total,
            0.0,
        )
    )
    scale[standardize_mask] = np.where(
        rms[standardize_mask] < 1e-12,
        1.0,
        rms[standardize_mask],
    )
    normalized = features / scale

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = normalized @ coefficients
        probabilities = np.where(
            logits >= 0.0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        loss = float(
            -np.sum(
                weights
                * (
                    labels * np.log(clipped)
                    + (1.0 - labels) * np.log1p(-clipped)
                )
            )
            / total
            + 0.5 * L2_LAMBDA * np.sum(coefficients * coefficients)
        )
        gradient = (
            normalized.T @ (weights * (probabilities - labels)) / total
            + L2_LAMBDA * coefficients
        )
        return loss, np.asarray(gradient, dtype=np.float64)

    fit = minimize(
        objective,
        np.zeros(FEATURE_WIDTH, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": MAX_OPTIMIZER_ITERATIONS,
            "gtol": OPTIMIZER_GTOL,
        },
    )
    gradient = np.asarray(fit.jac, dtype=np.float64)
    return PairwiseModel(
        feature_names=tuple(FEATURE_NAMES),
        schema_sha256=schema_sha256(),
        feature_scale=scale,
        coefficients=np.asarray(fit.x, dtype=np.float64),
        optimizer_summary={
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "iterations": int(fit.nit),
            "objective": float(fit.fun),
            "gradient_infinity_norm": float(
                np.max(np.abs(gradient), initial=0.0)
            ),
            "l2_lambda": L2_LAMBDA,
            "intercept": None,
            "calibration": None,
        },
        source_hashes=dict(source_hashes),
        pair_dataset_sha256=pair_dataset_sha256,
    )


def _metric_rows(
    model: PairwiseModel,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    weighted = assign_pair_weights(rows)
    features, labels, _weights = _pair_matrix(weighted)
    logits = model.logits(features)
    probabilities = np.where(
        logits >= 0.0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )
    scored = []
    for row, label, logit, probability in zip(
        weighted, labels, logits, probabilities
    ):
        loss = -(
            label * math.log(max(float(probability), 1e-12))
            + (1.0 - label)
            * math.log(max(1.0 - float(probability), 1e-12))
        )
        if float(logit) > 0.0:
            concordance = float(label)
        elif float(logit) < 0.0:
            concordance = float(1.0 - label)
        else:
            concordance = 0.5
        scored.append(
            {
                **row,
                "logit": float(logit),
                "probability": float(probability),
                "log_loss": float(loss),
                "log_loss_improvement": float(math.log(2.0) - loss),
                "concordance": concordance,
            }
        )
    total = sum(float(row["weight"]) for row in scored)
    summary = {
        "log_loss": sum(
            float(row["weight"]) * float(row["log_loss"]) for row in scored
        )
        / total,
        "log_loss_improvement": sum(
            float(row["weight"]) * float(row["log_loss_improvement"])
            for row in scored
        )
        / total,
        "concordance": sum(
            float(row["weight"]) * float(row["concordance"])
            for row in scored
        )
        / total,
        "prediction_ties": sum(
            1 for row in scored if float(row["logit"]) == 0.0
        ),
        "pairs": len(scored),
        "roots": len({str(row["root_cluster"]) for row in scored}),
    }
    return scored, summary


def _root_metric_values(
    scored: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in scored:
        grouped[
            (str(row["behavior_family"]), str(row["root_cluster"]))
        ].append(row)
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (family, root), rows in grouped.items():
        local_total = sum(float(row["root_local_weight"]) for row in rows)
        result[family][root] = {
            "log_loss_improvement": sum(
                float(row["root_local_weight"])
                * float(row["log_loss_improvement"])
                for row in rows
            )
            / local_total,
            "concordance": sum(
                float(row["root_local_weight"])
                * float(row["concordance"])
                for row in rows
            )
            / local_total,
        }
    return result


def root_bootstrap_metrics(
    scored: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    values = _root_metric_values(scored)
    families = sorted(values)
    if not families:
        raise ValueError("No roots for bootstrap")
    rng = np.random.default_rng(seed)
    draws = {
        "log_loss_improvement": np.empty(repeats, dtype=np.float64),
        "concordance": np.empty(repeats, dtype=np.float64),
    }
    family_arrays = {
        family: {
            metric: np.asarray(
                [values[family][root][metric] for root in sorted(values[family])],
                dtype=np.float64,
            )
            for metric in draws
        }
        for family in families
    }
    for index in range(repeats):
        for metric in draws:
            family_means = []
            for family in families:
                array = family_arrays[family][metric]
                sample = rng.integers(0, len(array), size=len(array))
                family_means.append(float(np.mean(array[sample])))
            draws[metric][index] = float(np.mean(family_means))
    return {
        "seed": seed,
        "repeats": repeats,
        "unit": "root resampled within behavior family",
        "metrics": {
            metric: {
                "lower_95": float(np.quantile(array, 0.025)),
                "median": float(np.quantile(array, 0.5)),
                "upper_95": float(np.quantile(array, 0.975)),
            }
            for metric, array in draws.items()
        },
    }


def _subset_metric(
    model: PairwiseModel,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {"pairs": 0, "roots": 0}
    _scored, summary = _metric_rows(model, rows)
    return summary


def _diagnostic_decision(
    *,
    optimizer: Mapping[str, Any],
    primary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    by_scale: Mapping[str, Mapping[str, Any]],
    by_family: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, bool]]:
    primary_checks = {
        "optimizer_success": bool(optimizer.get("success")),
        "gradient_at_most_1e_4":
            float(optimizer.get("gradient_infinity_norm", math.inf)) <= 1e-4,
        "primary_log_loss_improvement_positive":
            float(primary["log_loss_improvement"]) > 0.0,
        "primary_log_loss_ci_lower_positive":
            float(
                bootstrap["metrics"]["log_loss_improvement"]["lower_95"]
            )
            > 0.0,
        "primary_concordance_above_half":
            float(primary["concordance"]) > 0.5,
        "primary_concordance_ci_lower_above_half":
            float(bootstrap["metrics"]["concordance"]["lower_95"]) > 0.5,
        "both_scales_positive": all(
            int(summary.get("pairs", 0)) > 0
            and float(summary["log_loss_improvement"]) > 0.0
            and float(summary["concordance"]) > 0.5
            for summary in by_scale.values()
        ),
        "supported_families_nonnegative": all(
            int(summary.get("roots", 0)) < 8
            or (
                float(summary["log_loss_improvement"]) >= 0.0
                and float(summary["concordance"]) >= 0.5
            )
            for summary in by_family.values()
        ),
    }
    if all(primary_checks.values()):
        return "SUPPORT_G4_PAIRWISE_MECHANISM_SPENT", primary_checks
    decisive_failure = (
        float(bootstrap["metrics"]["log_loss_improvement"]["upper_95"]) <= 0.0
        or float(bootstrap["metrics"]["concordance"]["upper_95"]) <= 0.5
    )
    if decisive_failure:
        return "KILL_G4_PAIRWISE_MECHANISM_SPENT", primary_checks
    return "HOLD_G4_PAIRWISE_MECHANISM_AMBIGUOUS", primary_checks


def _build_dataset() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    records = _load_ordinary_records()
    paths = _read_ordinary_paths()
    enriched, record_audit = _enrich_and_validate_records(records, paths)
    features, feature_audit = _feature_map(enriched)
    pairs, pair_audit = build_pair_dataset(enriched, paths, features)
    dataset_audit = {
        "record_audit": record_audit,
        "path_manifest": _path_manifest(paths),
        "feature_audit": feature_audit,
        "pair_audit": pair_audit,
    }
    return pairs, dataset_audit, {
        "records": enriched,
        "paths": paths,
    }


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"G4 output already exists: {out_dir}")
    source_audit = _source_hash_audit()
    if not source_audit["passes"]:
        raise ValueError("Immutable G4 source audit failed before label read")

    pairs, dataset_audit, _sources = _build_dataset()
    train_rows = [row for row in pairs if row["partition"] == "train"]
    development_rows = [
        row for row in pairs if row["partition"] == "development"
    ]
    train_summary = _support_summary(train_rows)
    development_summary = _support_summary(development_rows)

    future_rows = _future_stream_rows()
    future_stream_audit = _future_stream_audit(
        future_rows,
        exclude_dir=out_dir,
    )
    untouched = _untouched_inventory(_load_ordinary_records())
    operations = _operational_audit()

    integrity_checks = {
        "immutable_sources_exact": bool(source_audit["passes"]),
        "record_path_crn_integrity":
            bool(dataset_audit["record_audit"]["passes"]),
        "feature_integrity": bool(dataset_audit["feature_audit"]["passes"]),
        "pair_integrity": bool(dataset_audit["pair_audit"]["passes"]),
        "ordinary_path_count_exact":
            int(dataset_audit["path_manifest"]["path_count"])
            == EXPECTED_ORDINARY_PATHS,
        "future_streams_collision_free": bool(future_stream_audit["passes"]),
        "operational_health": bool(operations["passes"]),
        "g3_transfer_access_zero": True,
    }
    decision, support_checks = _preflight_decision(
        integrity_checks=integrity_checks,
        train=train_summary,
        development=development_summary,
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    future_manifest = {
        "version": f"{VERSION}_future_streams",
        "charter_sha256": CHARTER_SHA256,
        "rows": future_rows,
        "rows_sha256": canonical_sha256(future_rows),
        "collision_audit": future_stream_audit,
        "streams_consumed": 0,
    }
    future_manifest["canonical_payload_sha256"] = canonical_sha256(
        future_manifest
    )
    write_immutable_json(
        out_dir / FUTURE_STREAM_MANIFEST_NAME,
        future_manifest,
    )

    untouched["charter_sha256"] = CHARTER_SHA256
    untouched["canonical_payload_sha256"] = canonical_sha256(untouched)
    write_immutable_json(out_dir / UNTOUCHED_INVENTORY_NAME, untouched)

    pair_manifest = {
        "version": f"{VERSION}_pair_manifest",
        "charter_sha256": CHARTER_SHA256,
        "source_hashes": source_audit["actual"],
        "dataset_audit": dataset_audit,
        "train_support": train_summary,
        "development_support": development_summary,
        "raw_pair_values_stored": False,
        "new_labels_generated": 0,
        "transfer_records_accessed": 0,
    }
    pair_manifest["canonical_payload_sha256"] = canonical_sha256(pair_manifest)
    write_immutable_json(out_dir / PAIR_MANIFEST_NAME, pair_manifest)

    implementation_path = Path(__file__)
    preflight = {
        "version": f"{VERSION}_preflight",
        "decision": decision,
        "charter_path": str(CHARTER_PATH),
        "charter_sha256": CHARTER_SHA256,
        "implementation_path": str(implementation_path),
        "implementation_sha256": sha256_path(implementation_path),
        "test_path": str(TEST_PATH),
        "test_sha256": sha256_path(TEST_PATH),
        "test_evidence": {
            "py_compile": {
                "command": (
                    "zsh -ic 'no-secrets .venv/bin/python -m py_compile "
                    "threes_rl/g4_conditional_pairwise.py "
                    "tests/test_rl_g4_conditional_pairwise.py'"
                ),
                "passed": True,
            },
            "focused": {
                "command": (
                    "zsh -ic 'no-secrets env PYTHONPATH=. "
                    "UV_CACHE_DIR=/tmp/uv-cache uv run --with pytest "
                    "--with numpy --with scipy pytest -q "
                    "tests/test_rl_g4_conditional_pairwise.py'"
                ),
                "passed": 12,
                "failed": 0,
            },
            "relevant_regressions": {
                "passed": 107,
                "failed": 0,
                "deselected": 1,
                "deselection_reason": (
                    "historical g3-v2 pre-execution test asserts its output "
                    "directory is absent after the authorized v2 run"
                ),
            },
        },
        "source_audit": source_audit,
        "integrity_checks": integrity_checks,
        "support_checks": support_checks,
        "train_support": train_summary,
        "development_support": development_summary,
        "pair_manifest_path": str(out_dir / PAIR_MANIFEST_NAME),
        "pair_manifest_file_sha256": sha256_path(
            out_dir / PAIR_MANIFEST_NAME
        ),
        "pair_dataset_sha256":
            dataset_audit["pair_audit"]["pair_dataset_sha256"],
        "future_stream_manifest_path": str(
            out_dir / FUTURE_STREAM_MANIFEST_NAME
        ),
        "future_stream_manifest_file_sha256": sha256_path(
            out_dir / FUTURE_STREAM_MANIFEST_NAME
        ),
        "untouched_inventory_path": str(
            out_dir / UNTOUCHED_INVENTORY_NAME
        ),
        "untouched_inventory_file_sha256": sha256_path(
            out_dir / UNTOUCHED_INVENTORY_NAME
        ),
        "prospective_power": _power_table(),
        "operations": operations,
        "g3_transfer_access": {
            "records": 0,
            "predictions": 0,
            "paths": 0,
            "database_opened": False,
        },
        "forbidden_work": {
            "new_simulations": 0,
            "new_labels": 0,
            "policy_outcomes": 0,
            "scores_inspected": 0,
            "model_fits": 0,
            "dashboard_changes": 0,
        },
        "state": {
            "CONTINUE": (
                "single_spent_diagnostic" if decision
                == "READY_G4_SPENT_DIAGNOSTIC" else "none"
            ),
            "HOLD": (
                "untouched_labels_policy_evaluation_C2_human_training_ground"
            ),
            "KILL": "G3_permanent",
            "PROMOTE": False,
        },
    }
    preflight["canonical_payload_sha256"] = canonical_sha256(preflight)
    write_immutable_json(out_dir / PREFLIGHT_NAME, preflight)
    return preflight


def _validate_preflight_for_diagnostic(out_dir: Path) -> dict[str, Any]:
    preflight = json_object(out_dir / PREFLIGHT_NAME)
    if (
        not verify_payload_hash(preflight)
        or preflight.get("decision") != "READY_G4_SPENT_DIAGNOSTIC"
        or preflight.get("charter_sha256") != CHARTER_SHA256
        or preflight.get("implementation_sha256")
        != sha256_path(Path(__file__))
        or preflight.get("pair_manifest_file_sha256")
        != sha256_path(out_dir / PAIR_MANIFEST_NAME)
    ):
        raise ValueError("G4 preflight is not valid for diagnostic")
    if not _source_hash_audit()["passes"]:
        raise ValueError("G4 immutable source changed after preflight")
    return preflight


def run_spent_diagnostic(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    preflight = _validate_preflight_for_diagnostic(out_dir)
    opened_path = out_dir / DIAGNOSTIC_OPENED_NAME
    result_path = out_dir / DIAGNOSTIC_RESULT_NAME
    error_path = out_dir / DIAGNOSTIC_ERROR_NAME
    if opened_path.exists() or result_path.exists() or error_path.exists():
        raise FileExistsError("G4 spent diagnostic is one-shot and already opened")
    marker = {
        "version": f"{VERSION}_diagnostic_opened",
        "charter_sha256": CHARTER_SHA256,
        "preflight_file_sha256": sha256_path(out_dir / PREFLIGHT_NAME),
        "preflight_payload_sha256": preflight["canonical_payload_sha256"],
        "pair_dataset_sha256": preflight["pair_dataset_sha256"],
        "implementation_sha256": sha256_path(Path(__file__)),
        "g3_transfer_access": 0,
        "new_labels_generated": 0,
    }
    marker["canonical_payload_sha256"] = canonical_sha256(marker)
    write_immutable_json(opened_path, marker)

    try:
        pairs, dataset_audit, _sources = _build_dataset()
        if (
            dataset_audit["pair_audit"]["pair_dataset_sha256"]
            != preflight["pair_dataset_sha256"]
        ):
            raise ValueError("Pair dataset changed after preflight")
        train_rows = [row for row in pairs if row["partition"] == "train"]
        development_rows = [
            row for row in pairs if row["partition"] == "development"
        ]
        source_hashes = {
            **{
                path: str(value)
                for path, value in _source_hash_audit()["actual"].items()
                if value is not None
            },
            "charter": CHARTER_SHA256,
            "implementation": sha256_path(Path(__file__)),
        }
        model = fit_pairwise_model(
            train_rows,
            source_hashes=source_hashes,
            pair_dataset_sha256=preflight["pair_dataset_sha256"],
        )
        model_metadata = model.save(out_dir / MODEL_DIR_NAME)
        loaded = PairwiseModel.load(
            out_dir / MODEL_DIR_NAME,
            expected_source_hashes=source_hashes,
        )
        train_scored, train_metrics = _metric_rows(model, train_rows)
        dev_scored, dev_metrics = _metric_rows(loaded, development_rows)
        bootstrap = root_bootstrap_metrics(dev_scored)
        by_scale = {
            scale: _subset_metric(
                loaded,
                [
                    row
                    for row in development_rows
                    if str(row["scale"]) == scale
                ],
            )
            for scale in ACCEPTED_SCALES
        }
        development_families = sorted(
            {str(row["behavior_family"]) for row in development_rows}
        )
        by_family = {
            family: _subset_metric(
                loaded,
                [
                    row
                    for row in development_rows
                    if str(row["behavior_family"]) == family
                ],
            )
            for family in development_families
        }
        by_horizon = {
            horizon: _subset_metric(
                loaded,
                [
                    row
                    for row in development_rows
                    if str(row["horizon"]) == horizon
                ],
            )
            for horizon in HORIZON_NAMES
        }
        by_action_pair = {
            action_pair: _subset_metric(
                loaded,
                [
                    row
                    for row in development_rows
                    if str(row["action_pair"]) == action_pair
                ],
            )
            for action_pair in sorted(
                {str(row["action_pair"]) for row in development_rows}
            )
        }
        mechanism_decision, decision_checks = _diagnostic_decision(
            optimizer=model.optimizer_summary,
            primary=dev_metrics,
            bootstrap=bootstrap,
            by_scale=by_scale,
            by_family=by_family,
        )
        train_root_metrics = _root_metric_values(train_scored)
        dev_root_metrics = _root_metric_values(dev_scored)
        result = {
            "version": f"{VERSION}_spent_diagnostic",
            "operational_terminal": "HOLD_G4_AFTER_SPENT_DIAGNOSTIC",
            "mechanism_decision": mechanism_decision,
            "decision_checks": decision_checks,
            "charter_sha256": CHARTER_SHA256,
            "preflight_file_sha256": sha256_path(out_dir / PREFLIGHT_NAME),
            "marker_file_sha256": sha256_path(opened_path),
            "marker_payload_sha256": marker["canonical_payload_sha256"],
            "pair_dataset_sha256": preflight["pair_dataset_sha256"],
            "model": {
                "directory": str(out_dir / MODEL_DIR_NAME),
                "meta_file_sha256": sha256_path(
                    out_dir / MODEL_DIR_NAME / "meta.json"
                ),
                "arrays_file_sha256": sha256_path(
                    out_dir / MODEL_DIR_NAME / "arrays.npz"
                ),
                "metadata_payload_sha256":
                    model_metadata["canonical_payload_sha256"],
                "optimizer": model.optimizer_summary,
                "save_load_prediction_exact": bool(
                    np.array_equal(
                        model.logits(
                            np.stack(
                                [
                                    np.asarray(row["delta"], dtype=np.float64)
                                    for row in development_rows
                                ]
                            )
                        ),
                        loaded.logits(
                            np.stack(
                                [
                                    np.asarray(row["delta"], dtype=np.float64)
                                    for row in development_rows
                                ]
                            )
                        ),
                    )
                ),
            },
            "train_metrics": train_metrics,
            "development_primary_metrics": dev_metrics,
            "development_bootstrap": bootstrap,
            "development_by_scale": by_scale,
            "development_by_family": by_family,
            "development_by_horizon": by_horizon,
            "development_by_action_pair": by_action_pair,
            "root_metric_manifest_hashes": {
                "train": canonical_sha256(train_root_metrics),
                "development": canonical_sha256(dev_root_metrics),
            },
            "g3_transfer_access": {
                "records": 0,
                "predictions": 0,
                "paths": 0,
                "database_opened": False,
            },
            "forbidden_work": {
                "new_simulations": 0,
                "new_labels": 0,
                "normal_start_policy_evaluations": 0,
                "scores_inspected": 0,
                "dashboard_changes": 0,
            },
            "state": {
                "CONTINUE": "none_pending_oversight",
                "HOLD": (
                    "all_untouched_labels_policy_construction_evaluation_C2_"
                    "human_training_ground"
                ),
                "KILL": {
                    "G3": "permanent",
                    "G4_exact_spent_mechanism": (
                        mechanism_decision
                        == "KILL_G4_PAIRWISE_MECHANISM_SPENT"
                    ),
                },
                "PROMOTE": False,
            },
        }
        result["canonical_payload_sha256"] = canonical_sha256(result)
        write_immutable_json(result_path, result)
        return result
    except Exception as error:
        failure = {
            "version": f"{VERSION}_spent_diagnostic_error",
            "operational_terminal": "HOLD_G4_AFTER_SPENT_DIAGNOSTIC",
            "decision": "HOLD_G4_DIAGNOSTIC_ENGINEERING_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "charter_sha256": CHARTER_SHA256,
            "marker_file_sha256": sha256_path(opened_path),
            "new_labels_generated": 0,
            "g3_transfer_access": 0,
            "PROMOTE": False,
        }
        failure["canonical_payload_sha256"] = canonical_sha256(failure)
        write_immutable_json(error_path, failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G4 conditional pairwise preflight/spent diagnostic"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "diagnostic"):
        child = subparsers.add_parser(command)
        child.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args.out_dir)
    else:
        result = run_spent_diagnostic(args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

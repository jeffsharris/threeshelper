"""One-shot, outcome-blind 128-root O2 yield pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import optimize, sparse

from threes_rl import g1r_acquire as history
from threes_rl import g1r_acquire_v2_qd5 as family_source
from threes_rl import o2_online_option_preflight as preflight
from threes_rl.eval import EvalJob, EvalStreamIds, iter_eval_job_outputs
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.o1_geometry_option import air_safe, anchor_safe, geometry
from threes_rl.replay_provenance import (
    ORIGIN_FRESH,
    direct_root_fields,
    replay_provenance,
)
from threes_rl.restart_manifest import canonical_ancestry_id, state_signature
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "o2_yield_pilot_v1"
ROOT = Path("threes_rl/runs")
OUTPUT_DIR = ROOT / "forensics/o2_yield_pilot_v1"
CHARTER_PATH = Path("threes_rl/O2_YIELD_PILOT_EXECUTION_CHARTER.md")
RUNNER_PATH = Path("threes_rl/o2_yield_pilot.py")
TEST_PATH = Path("tests/test_rl_o2_yield_pilot.py")
TEST_EVIDENCE_PATH = ROOT / "forensics/o2_yield_pilot_test_evidence.json"
PREFLIGHT_RESULT_PATH = (
    ROOT / "forensics/o2_online_option_preflight_v1/O2_PREFLIGHT_RESULT.json"
)
PREFLIGHT_DESIGN_PATH = (
    ROOT / "forensics/o2_online_option_preflight_v1/O2_DESIGN_MANIFEST.json"
)
PREFLIGHT_STREAM_PATH = (
    ROOT / "forensics/o2_online_option_preflight_v1/O2_STREAM_MANIFEST.json"
)
PREFLIGHT_DIR = ROOT / "forensics/o2_online_option_preflight_v1"
MARKER_PATH = OUTPUT_DIR / "O2_YIELD_PILOT_EXECUTION_OPENED.json"
ATTEMPT_PATH = OUTPUT_DIR / "attempts.jsonl"
COMPLETION_PATH = OUTPUT_DIR / "completed_games.jsonl"
RUNTIME_PATH = OUTPUT_DIR / "runtime_state.json"
REPLAY_DIR = OUTPUT_DIR / "source_replays"
SUPPORT_PATH = OUTPUT_DIR / "O2_PILOT_SUPPORT.json"
RESULT_PATH = OUTPUT_DIR / "O2_YIELD_PILOT_RESULT.json"

CHARTER_SHA256 = (
    "4c015f0238ae7e6fe545cf497b58576890a733656995fe0f6507767e2e3391b3"
)
PREFLIGHT_RESULT_FILE_SHA256 = (
    "09c8b19cea4b992f3b77feab71b978c717e88f4dbf0547a5789e54759ae430fc"
)
PREFLIGHT_RESULT_PAYLOAD_SHA256 = (
    "60ffe3905a032e26e1a32b7f248e4f138cf56f09c0d49b44b59802e824fa5f5b"
)
PREFLIGHT_DESIGN_FILE_SHA256 = (
    "d921ae156503f697d9d689f39bc664a83cb88ca21e9bb01d5fdb34653de30480"
)
PREFLIGHT_DESIGN_PAYLOAD_SHA256 = (
    "287780188a284044adffb366791524ea2da3daa04d5db903b84bb91b2f4b5c00"
)
PREFLIGHT_STREAM_FILE_SHA256 = (
    "ff0a67f98bc68ebfee5b3700b533f0a8b0e7e8f628f22cbf840cc91d7e6eedbc"
)
PREFLIGHT_STREAM_PAYLOAD_SHA256 = (
    "fb29eb96e8f4cd90b5783d4dadae496ffc450d373685e36542243af6dfdf14bf"
)
PREFLIGHT_STREAM_ROW_SHA256 = (
    "b4bcc661d0502df3b75178b496a75e3b19770d8e288698ac8a585b15e7d8d836"
)

FAMILY_ORDER = tuple(row[0] for row in preflight.FAMILIES)
HISTORICAL_FAMILIES = {
    row[0]: row[1] for row in preflight.FAMILIES
}
POLICY_SPECS = {row[0]: row[2] for row in preflight.FAMILIES}
EXPECTED_SIGNATURES = {row[1]: row[3] for row in preflight.FAMILIES}
TARGET_ORDER = (768, 384, 192, 96, 48)
SCAN_TARGETS = (*TARGET_ORDER, 1536)
STAGE_ORDER = (0, 1, 2, 3)
PILOT_ROOTS_PER_FAMILY = 32
TOTAL_ROOTS = 128
CHUNK_SIZE = 4
FROZEN_JOBS = 1
STARTER_TILE = 1536
MAX_MOVES = 5000
MINIMUM_NICE = 10
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
ACTIVE_RUNTIME_LIMIT = 6 * 3600
OUTPUT_BYTE_LIMIT = 3 * 1024**3
EXPECTED_TOP_THREE = (263670, 261369, 258561)
STRUCTURAL_QUOTA = {target: (7 if target == 768 else 4) for target in TARGET_ORDER}
STRUCTURAL_FAMILY_CAP = {
    target: (3 if target == 768 else 2) for target in TARGET_ORDER
}
AVAILABILITY_MINIMUM = {
    target: (8 if target == 768 else 7) for target in TARGET_ORDER
}
AVAILABILITY_FINAL_DEMAND = {
    target: (20 if target == 768 else 18) for target in TARGET_ORDER
}
AVAILABILITY_FAMILY_CAP = 3

EXECUTE_COMMAND = (
    "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. .venv/bin/python "
    "-m threes_rl.o2_yield_pilot execute --out-dir "
    "threes_rl/runs/forensics/o2_yield_pilot_v1 --jobs 1'"
)

DEPENDENCY_SOURCE_PATHS = (
    Path("threes_rl/action_prior.py"),
    Path("threes_rl/eval.py"),
    Path("threes_rl/expectimax.py"),
    Path("threes_rl/g1r_acquire.py"),
    Path("threes_rl/g1r_acquire_v2_qd5.py"),
    Path("threes_rl/g1r_qd_admission_v2.py"),
    Path("threes_rl/ntuple.py"),
    Path("threes_rl/o1_geometry_option.py"),
    Path("threes_rl/o2_online_option_preflight.py"),
    Path("threes_rl/record_replay.py"),
    Path("threes_rl/replay_provenance.py"),
    Path("threes_rl/restart_manifest.py"),
    Path("threes_rl/run_artifacts.py"),
    Path("threes_rl/sim.py"),
    Path("threes_rl/train_td.py"),
)


def canonical_json_hash(value: Any) -> str:
    return preflight.canonical_json_hash(value)


def sha256_path(path: Path) -> str:
    return preflight.sha256_path(path)


def _atomic_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    preflight.write_immutable_json(path, payload)


def _payload_identity(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    checks = {
        "file_sha256": sha256_path(path) == expected_file_sha256,
        "payload_sha256": payload.get("canonical_payload_sha256")
        == expected_payload_sha256,
        "payload_valid": preflight.verify_payload_hash(payload),
    }
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "canonical_payload_sha256": payload.get("canonical_payload_sha256"),
        "checks": checks,
        "passes": all(checks.values()),
    }


def immutable_input_audit() -> dict[str, Any]:
    rows = {
        "charter": {
            "path": str(CHARTER_PATH),
            "expected": CHARTER_SHA256,
            "actual": sha256_path(CHARTER_PATH),
        },
        "base_charter": {
            "path": str(preflight.CHARTER_PATH),
            "expected": preflight.CHARTER_SHA256,
            "actual": sha256_path(preflight.CHARTER_PATH),
        },
        "amendment_a1": {
            "path": str(preflight.A1_PATH),
            "expected": preflight.A1_SHA256,
            "actual": sha256_path(preflight.A1_PATH),
        },
        "amendment_a2": {
            "path": str(preflight.A2_PATH),
            "expected": preflight.A2_SHA256,
            "actual": sha256_path(preflight.A2_PATH),
        },
        "amendment_a3": {
            "path": str(preflight.A3_PATH),
            "expected": preflight.A3_SHA256,
            "actual": sha256_path(preflight.A3_PATH),
        },
        "amendment_a4": {
            "path": str(preflight.A4_PATH),
            "expected": preflight.A4_SHA256,
            "actual": sha256_path(preflight.A4_PATH),
        },
    }
    for row in rows.values():
        row["passes"] = row["actual"] == row["expected"]
    sealed = {
        "result": _payload_identity(
            PREFLIGHT_RESULT_PATH,
            expected_file_sha256=PREFLIGHT_RESULT_FILE_SHA256,
            expected_payload_sha256=PREFLIGHT_RESULT_PAYLOAD_SHA256,
        ),
        "design": _payload_identity(
            PREFLIGHT_DESIGN_PATH,
            expected_file_sha256=PREFLIGHT_DESIGN_FILE_SHA256,
            expected_payload_sha256=PREFLIGHT_DESIGN_PAYLOAD_SHA256,
        ),
        "stream": _payload_identity(
            PREFLIGHT_STREAM_PATH,
            expected_file_sha256=PREFLIGHT_STREAM_FILE_SHA256,
            expected_payload_sha256=PREFLIGHT_STREAM_PAYLOAD_SHA256,
        ),
    }
    result = json.loads(PREFLIGHT_RESULT_PATH.read_text())
    design = json.loads(PREFLIGHT_DESIGN_PATH.read_text())
    stream = json.loads(PREFLIGHT_STREAM_PATH.read_text())
    checks = {
        "document_hashes": all(row["passes"] for row in rows.values()),
        "sealed_payloads": all(row["passes"] for row in sealed.values()),
        "preflight_ready": result.get("decision")
        == "READY_O2_YIELD_PILOT_PREFLIGHT",
        "pilot_not_previously_authorized": result.get(
            "pilot_execution_authorized"
        )
        is False,
        "a4_design": design.get("version", "").endswith("_a4")
        and bool(design.get("passes")),
        "stream_rows_exact": stream.get("row_manifest_sha256")
        == PREFLIGHT_STREAM_ROW_SHA256,
        "streams_unconsumed": stream.get("streams_consumed") == 0,
    }
    return {
        "documents": rows,
        "sealed": sealed,
        "checks": checks,
        "passes": all(checks.values()),
    }


def dependency_manifest() -> dict[str, Any]:
    source_rows = []
    for path in DEPENDENCY_SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"O2 bound dependency missing: {path}")
        source_rows.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_path(path),
            }
        )
    source_rows.sort(key=lambda row: row["path"])
    o1_contract = preflight.bound_document_audit()
    family_evidence = preflight.family_evidence_audit()
    pilot_lock = json.loads(preflight.PILOT_V2_LOCK_PATH.read_text())
    policy_lock = pilot_lock["policy_lock"]
    body = {
        "version": f"{VERSION}_dependency_manifest",
        "sources": source_rows,
        "o1_contract": o1_contract,
        "policy_identity": {
            "pilot_lock_path": str(preflight.PILOT_V2_LOCK_PATH),
            "pilot_lock_file_sha256": sha256_path(
                preflight.PILOT_V2_LOCK_PATH
            ),
            "policy_lock_sha256": policy_lock["policy_lock_sha256"],
            "policy_lock_manifest_sha256": canonical_json_hash(policy_lock),
            "family_evidence_sha256": canonical_json_hash(family_evidence),
        },
        "checks": {
            "o1_contract_exact": bool(o1_contract.get("passes")),
            "family_policy_checkpoint_evidence_exact": bool(
                family_evidence.get("passes")
            ),
        },
    }
    body["passes"] = all(body["checks"].values())
    return {
        **body,
        "manifest_sha256": canonical_json_hash(body),
    }


def pilot_stream_rows() -> list[dict[str, Any]]:
    payload = json.loads(PREFLIGHT_STREAM_PATH.read_text())
    if not preflight.verify_payload_hash(payload):
        raise ValueError("O2 preflight stream payload invalid")
    rows = [
        dict(row) for row in payload["rows"] if row.get("purpose") == "pilot"
    ]
    rows.sort(key=lambda row: (int(row["family_index"]), int(row["family_game_index"])))
    checks = {
        "exact_count": len(rows) == TOTAL_ROOTS,
        "family_order": tuple(
            sorted(
                {str(row["family"]) for row in rows},
                key=FAMILY_ORDER.index,
            )
        )
        == FAMILY_ORDER,
        "equal_family_counts": Counter(str(row["family"]) for row in rows)
        == {family: PILOT_ROOTS_PER_FAMILY for family in FAMILY_ORDER},
        "logical_81b": all(
            81_000_000_000 <= int(row["logical_seed"]) < 82_000_000_000
            for row in rows
        ),
        "deck_82b": all(
            82_000_000_000 <= int(row["deck_stream_id"]) < 83_000_000_000
            for row in rows
        ),
        "slot_83b": all(
            83_000_000_000 <= int(row["slot_stream_id"]) < 84_000_000_000
            for row in rows
        ),
        "policy_84b": all(
            84_000_000_000 <= int(row["policy_stream_id"]) < 85_000_000_000
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Frozen O2 pilot stream rows invalid: {checks}")
    return rows


def round_robin_chunks(
    rows: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    if PILOT_ROOTS_PER_FAMILY % CHUNK_SIZE:
        raise ValueError("O2 roots per family must divide evenly into chunks")
    indexed = {
        (str(row["family"]), int(row["family_game_index"])): dict(row)
        for row in rows
    }
    chunks = []
    for start in range(0, PILOT_ROOTS_PER_FAMILY, CHUNK_SIZE):
        for family in FAMILY_ORDER:
            chunk = [
                indexed[(family, index)]
                for index in range(start, start + CHUNK_SIZE)
            ]
            chunks.append(chunk)
    expected_chunks = len(FAMILY_ORDER) * (
        PILOT_ROOTS_PER_FAMILY // CHUNK_SIZE
    )
    if len(chunks) != expected_chunks or any(
        len(chunk) != CHUNK_SIZE for chunk in chunks
    ):
        raise ValueError("O2 round-robin chunk construction failed")
    return chunks


ATTEMPT_TERMINAL_STATUSES = (
    "completed",
    "completed_recovered",
    "interrupted_no_replay",
)
ATTEMPT_EVENT_FIELDS = {
    "version",
    "attempt_id",
    "attempt_index",
    "chunk_index",
    "family",
    "family_index",
    "family_game_index",
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
    "status",
}


def _stream_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["family"]), int(row["family_game_index"])


def _chunk_index_by_key(
    chunks: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for chunk_index, chunk in enumerate(chunks):
        for row in chunk:
            key = _stream_key(row)
            if key in result:
                raise ValueError(f"O2 stream row appears in two chunks: {key}")
            result[key] = chunk_index
    return result


def _attempt_identity(
    row: Mapping[str, Any],
    *,
    chunk_index: int,
    attempt_index: int,
) -> dict[str, Any]:
    identity = {
        "version": f"{VERSION}_attempt",
        "attempt_index": int(attempt_index),
        "chunk_index": int(chunk_index),
        "family": str(row["family"]),
        "family_index": int(row["family_index"]),
        "family_game_index": int(row["family_game_index"]),
        "logical_seed": int(row["logical_seed"]),
        "deck_stream_id": int(row["deck_stream_id"]),
        "slot_stream_id": int(row["slot_stream_id"]),
        "policy_stream_id": int(row["policy_stream_id"]),
    }
    identity["attempt_id"] = canonical_json_hash(
        {"namespace": "O2-yield-pilot-attempt-v1", **identity}
    )
    return identity


def _load_attempt_ledger(
    rows: Sequence[Mapping[str, Any]],
    chunks: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    row_by_key = {_stream_key(row): row for row in rows}
    chunk_by_key = _chunk_index_by_key(chunks)
    attempts: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    events = []
    if ATTEMPT_PATH.is_file():
        with ATTEMPT_PATH.open() as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    seen_ids: dict[str, tuple[str, int]] = {}
    for event in events:
        if set(event) != ATTEMPT_EVENT_FIELDS:
            raise ValueError("O2 attempt event has an incompatible schema")
        key = _stream_key(event)
        if key not in row_by_key:
            raise ValueError(f"O2 attempt is outside frozen rows: {key}")
        attempt_index = int(event["attempt_index"])
        expected = _attempt_identity(
            row_by_key[key],
            chunk_index=chunk_by_key[key],
            attempt_index=attempt_index,
        )
        if any(event[field] != value for field, value in expected.items()):
            raise ValueError(f"O2 attempt identity mismatch: {key}")
        attempt_id = str(event["attempt_id"])
        previous_key = seen_ids.setdefault(attempt_id, key)
        if previous_key != key:
            raise ValueError("O2 attempt ID reused across stream rows")
        while len(attempts[key]) <= attempt_index:
            attempts[key].append({"identity": None, "statuses": []})
        attempt = attempts[key][attempt_index]
        if attempt["identity"] is None:
            attempt["identity"] = expected
        attempt["statuses"].append(str(event["status"]))
    for key, rows_attempts in attempts.items():
        for attempt_index, attempt in enumerate(rows_attempts):
            if attempt["identity"] is None:
                raise ValueError(f"O2 attempt indices are not contiguous: {key}")
            statuses = attempt["statuses"]
            valid = statuses == ["opened"] or (
                len(statuses) == 2
                and statuses[0] == "opened"
                and statuses[1] in ATTEMPT_TERMINAL_STATUSES
            )
            if not valid:
                raise ValueError(
                    f"O2 attempt status sequence is invalid: {key} {statuses}"
                )
            if attempt_index < len(rows_attempts) - 1 and statuses != [
                "opened",
                "interrupted_no_replay",
            ]:
                raise ValueError(
                    f"O2 retry hides a non-interrupted prior attempt: {key}"
                )
    return dict(attempts)


def _append_attempt_status(
    attempt: dict[str, Any],
    status: str,
) -> None:
    statuses = attempt["statuses"]
    if status == "opened":
        if statuses:
            raise ValueError("O2 attempt opened twice")
    elif statuses != ["opened"] or status not in ATTEMPT_TERMINAL_STATUSES:
        raise ValueError("O2 attempt terminal status is not paired to one open")
    event = {**attempt["identity"], "status": status}
    history._append_jsonl_row(ATTEMPT_PATH, event)
    statuses.append(status)


def _open_attempt(
    attempts: dict[tuple[str, int], list[dict[str, Any]]],
    row: Mapping[str, Any],
    *,
    chunk_index: int,
) -> dict[str, Any]:
    key = _stream_key(row)
    rows_attempts = attempts.setdefault(key, [])
    if rows_attempts and rows_attempts[-1]["statuses"] != [
        "opened",
        "interrupted_no_replay",
    ]:
        raise ValueError(f"O2 attempted an unapproved retry: {key}")
    attempt = {
        "identity": _attempt_identity(
            row,
            chunk_index=chunk_index,
            attempt_index=len(rows_attempts),
        ),
        "statuses": [],
    }
    rows_attempts.append(attempt)
    _append_attempt_status(attempt, "opened")
    return attempt


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def collision_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    found: dict[str, set[int]] = defaultdict(set)
    matched = []
    excluded = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        if _is_within(path, PREFLIGHT_DIR) or _is_within(path, out_dir):
            excluded.append(
                {
                    "path": str(path),
                    "sha256": sha256_path(path),
                    "bytes": int(path.stat().st_size),
                    "classification": (
                        "immutable_o2_preflight_reservation"
                        if _is_within(path, PREFLIGHT_DIR)
                        else "current_o2_pilot_namespace"
                    ),
                }
            )
            continue
        values = history._scan_history_file(path)
        if not values:
            continue
        for key, items in values.items():
            found[key].update(items)
        matched.append(
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "bytes": int(path.stat().st_size),
                "counts": {
                    key: len(items) for key, items in sorted(values.items())
                },
            }
        )
    collisions = {}
    for field in preflight.STREAM_FIELDS:
        requested = {int(row[field]) for row in rows}
        historical = set(found.get(field, set()))
        if field == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                historical.update(found.get(alias, set()))
        collisions[field] = sorted(requested.intersection(historical))
    requested_ids = [
        int(row[field]) for row in rows for field in preflight.STREAM_FIELDS
    ]
    checks = {
        "requested_ids_unique": len(requested_ids) == len(set(requested_ids)),
        "zero_external_collisions": not any(collisions.values()),
        "preflight_reservation_excluded": any(
            row["classification"] == "immutable_o2_preflight_reservation"
            for row in excluded
        ),
    }
    return {
        "collisions": collisions,
        "matched_source_count": len(matched),
        "matched_sources_sha256": canonical_json_hash(matched),
        "excluded_source_count": len(excluded),
        "excluded_sources_sha256": canonical_json_hash(excluded),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not preflight.verify_payload_hash(payload):
        raise ValueError("O2 pilot test evidence payload mismatch")
    expected = {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("O2 pilot test evidence source identity mismatch")
    if not payload.get("passes"):
        raise ValueError("O2 pilot test evidence is not passing")
    return payload


def seal_test_evidence(
    *,
    focused_passed: int,
    regression_passed: int,
    commands: Sequence[str],
) -> dict[str, Any]:
    if focused_passed <= 0 or regression_passed <= 0 or not commands:
        raise ValueError("Passing test counts and commands are required")
    payload = {
        "version": "o2_yield_pilot_test_evidence_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "focused_passed": int(focused_passed),
        "regression_passed": int(regression_passed),
        "commands": list(commands),
        "passes": True,
        "zero_work": {
            "games": 0,
            "streams": 0,
            "support_scans": 0,
            "labels": 0,
            "model_fits": 0,
            "policy_outcomes": 0,
        },
    }
    _atomic_new_json(TEST_EVIDENCE_PATH, payload)
    return json.loads(TEST_EVIDENCE_PATH.read_text())


def operational_audit(*, out_dir: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    free_gib = disk.free / 1024**3
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    service = history.service_health()
    heavy = _heavy_process_audit()
    used = (
        history._directory_bytes(out_dir)
        if out_dir.exists()
        else 0
    )
    checks = {
        "nice_at_least_10": nice >= MINIMUM_NICE,
        "no_competing_heavy_process": bool(heavy.get("passes")),
        "free_disk_hard": free_gib >= MIN_FREE_GIB,
        "free_disk_target": free_gib >= TARGET_FREE_GIB,
        "services_healthy": bool(service.get("passes")),
        "top_three_exact": tuple(service.get("dashboard_top_scores", [])[:3])
        == EXPECTED_TOP_THREE,
        "output_below_limit": used < OUTPUT_BYTE_LIMIT,
    }
    return {
        "nice": nice,
        "free_bytes": disk.free,
        "free_gib": free_gib,
        "output_bytes": used,
        "service_health": service,
        "heavy_process_audit": heavy,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _marker_bindings(
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current_dependencies = (
        dict(dependencies) if dependencies is not None else dependency_manifest()
    )
    return {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "preflight_result_file_sha256": sha256_path(PREFLIGHT_RESULT_PATH),
        "preflight_design_file_sha256": sha256_path(PREFLIGHT_DESIGN_PATH),
        "preflight_stream_file_sha256": sha256_path(PREFLIGHT_STREAM_PATH),
        "dependency_manifest_sha256": current_dependencies[
            "manifest_sha256"
        ],
    }


def open_execution(*, out_dir: Path = OUTPUT_DIR, jobs: int = FROZEN_JOBS) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O2 pilot output directory mismatch")
    if jobs != FROZEN_JOBS:
        raise ValueError("O2 pilot requires jobs=1")
    if out_dir.exists():
        raise FileExistsError(f"O2 pilot output already exists: {out_dir}")
    immutable = immutable_input_audit()
    tests = _load_test_evidence()
    rows = pilot_stream_rows()
    families = preflight.family_evidence_audit()
    dependencies = dependency_manifest()
    collision = collision_audit(rows, out_dir=out_dir)
    operations = operational_audit(out_dir=out_dir)
    chunks = round_robin_chunks(rows)
    checks = {
        "immutable_inputs": immutable["passes"],
        "tests": tests["passes"],
        "family_signatures_and_payloads": families["passes"],
        "dependency_and_policy_bindings": dependencies["passes"],
        "stream_collisions": collision["passes"],
        "operations": operations["passes"],
        "exact_128_rows": len(rows) == TOTAL_ROOTS,
        "exact_round_robin": len(chunks) == 32
        and all(len(chunk) == CHUNK_SIZE for chunk in chunks),
        "output_absent": not out_dir.exists(),
        "no_prior_work": not any(
            path.exists()
            for path in (
                MARKER_PATH,
                ATTEMPT_PATH,
                COMPLETION_PATH,
                RUNTIME_PATH,
                REPLAY_DIR,
                SUPPORT_PATH,
                RESULT_PATH,
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"O2 pilot open checks failed: {checks}")
    marker = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "OPENED_O2_YIELD_PILOT",
        "bound_out_dir": str(out_dir.resolve()),
        "bound_execute_command": EXECUTE_COMMAND,
        "jobs": jobs,
        "family_order": list(FAMILY_ORDER),
        "games_per_family": PILOT_ROOTS_PER_FAMILY,
        "total_games": TOTAL_ROOTS,
        "chunk_size": CHUNK_SIZE,
        "active_runtime_limit": ACTIVE_RUNTIME_LIMIT,
        "output_byte_limit": OUTPUT_BYTE_LIMIT,
        "minimum_free_gib": MIN_FREE_GIB,
        "target_free_gib": TARGET_FREE_GIB,
        "stream_rows_sha256": canonical_json_hash(rows),
        "stream_rows": rows,
        "bindings": _marker_bindings(dependencies),
        "dependency_manifest": dependencies,
        "immutable_input_audit": immutable,
        "family_evidence": families,
        "collision_audit": collision,
        "operational_audit": operations,
        "checks": checks,
        "zero_work": {
            "games": 0,
            "streams": 0,
            "attempt_rows": 0,
            "completion_rows": 0,
            "replays": 0,
            "support_scans": 0,
            "labels": 0,
            "model_fits": 0,
            "policy_outcomes": 0,
            "dashboard_changes": 0,
        },
    }
    out_dir.mkdir(parents=True)
    _atomic_new_json(MARKER_PATH, marker)
    return json.loads(MARKER_PATH.read_text())


def _load_marker(*, out_dir: Path, jobs: int) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O2 pilot output directory mismatch")
    if jobs != FROZEN_JOBS:
        raise ValueError("O2 pilot jobs mismatch")
    if RESULT_PATH.exists():
        raise FileExistsError("O2 pilot terminal result already exists")
    if not MARKER_PATH.is_file():
        raise FileNotFoundError("O2 pilot execution marker missing")
    marker = json.loads(MARKER_PATH.read_text())
    if not preflight.verify_payload_hash(marker):
        raise ValueError("O2 pilot marker payload mismatch")
    current_dependencies = dependency_manifest()
    current_families = preflight.family_evidence_audit()
    checks = {
        "version": marker.get("version") == VERSION,
        "decision": marker.get("decision") == "OPENED_O2_YIELD_PILOT",
        "out_dir": marker.get("bound_out_dir") == str(out_dir.resolve()),
        "command": marker.get("bound_execute_command") == EXECUTE_COMMAND,
        "jobs": marker.get("jobs") == jobs,
        "bindings": marker.get("bindings")
        == _marker_bindings(current_dependencies),
        "dependencies": marker.get("dependency_manifest")
        == current_dependencies,
        "family_evidence": marker.get("family_evidence")
        == current_families,
        "stream_rows": marker.get("stream_rows") == pilot_stream_rows(),
        "stream_hash": marker.get("stream_rows_sha256")
        == canonical_json_hash(pilot_stream_rows()),
    }
    if not all(checks.values()):
        raise ValueError(f"O2 pilot marker binding mismatch: {checks}")
    return marker


def _runtime_state() -> dict[str, Any]:
    if RUNTIME_PATH.is_file():
        return json.loads(RUNTIME_PATH.read_text())
    return {
        "version": VERSION,
        "active_runtime_seconds": 0.0,
        "evaluation_batches_charged": 0,
        "games_evaluated_charged": 0,
        "chunks_completed": 0,
        "games_completed": 0,
    }


def _load_completions() -> dict[tuple[str, int], dict[str, Any]]:
    rows = history._load_completed(COMPLETION_PATH)
    allowed = {
        (str(row["family"]), int(row["family_game_index"]))
        for row in pilot_stream_rows()
    }
    if not set(rows).issubset(allowed):
        raise ValueError("O2 completion contains a non-frozen row")
    return rows


def _verify_existing_completions(
    completions: Mapping[tuple[str, int], Mapping[str, Any]],
    stream_rows: Sequence[Mapping[str, Any]],
) -> None:
    by_key = {
        (str(row["family"]), int(row["family_game_index"])): row
        for row in stream_rows
    }
    for key, completion in completions.items():
        if key not in by_key:
            raise ValueError(f"O2 completion key is outside frozen rows: {key}")
        replay_path = Path(str(completion["source_replay"]))
        if (
            not replay_path.is_file()
            or sha256_path(replay_path)
            != str(completion["source_replay_sha256"])
        ):
            raise ValueError(f"O2 completed replay changed: {replay_path}")
        replay = json.loads(replay_path.read_text())
        reconstructed = _completion_from_replay(
            replay,
            replay_path=replay_path,
            stream_row=by_key[key],
        )
        if reconstructed != dict(completion):
            raise ValueError(f"O2 completion row mismatch on resume: {key}")


def _reconcile_attempts_and_replays(
    completions: dict[tuple[str, int], dict[str, Any]],
    stream_rows: Sequence[Mapping[str, Any]],
    attempts: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    for row in stream_rows:
        key = _stream_key(row)
        rows_attempts = attempts.get(key, [])
        latest = rows_attempts[-1] if rows_attempts else None
        if key in completions:
            if latest is None:
                raise ValueError(
                    f"O2 completion has no immutable attempt evidence: {key}"
                )
            if latest["statuses"] == ["opened"]:
                _append_attempt_status(latest, "completed_recovered")
            elif latest["statuses"] not in (
                ["opened", "completed"],
                ["opened", "completed_recovered"],
            ):
                raise ValueError(
                    f"O2 completion follows an invalid attempt state: {key}"
                )
            continue
        replay_path = _replay_path(row)
        if not replay_path.exists():
            if latest is not None and latest["statuses"] == ["opened"]:
                _append_attempt_status(latest, "interrupted_no_replay")
            continue
        if latest is None or latest["statuses"] != ["opened"]:
            raise ValueError(
                f"O2 orphan replay has no open attempt evidence: {key}"
            )
        replay = json.loads(replay_path.read_text())
        completion = _completion_from_replay(
            replay,
            replay_path=replay_path,
            stream_row=row,
        )
        history._append_jsonl_row(COMPLETION_PATH, completion)
        completions[key] = completion
        _append_attempt_status(latest, "completed_recovered")


def _replay_path(stream_row: Mapping[str, Any]) -> Path:
    return REPLAY_DIR / (
        f"{int(stream_row['family_index']):02d}_"
        f"{stream_row['family']}_"
        f"{int(stream_row['family_game_index']):03d}.json"
    )


def _completion_from_replay(
    replay: dict[str, Any],
    *,
    replay_path: Path,
    stream_row: Mapping[str, Any],
) -> dict[str, Any]:
    if replay.get("game_over") is not True:
        raise ValueError("O2 pilot replay is not naturally terminal")
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("O2 pilot replay has no frames")
    final_state = frames[-1].get("state") if isinstance(frames[-1], dict) else None
    if not isinstance(final_state, dict) or final_state.get("game_over") is not True:
        raise ValueError("O2 pilot final frame is not terminal")
    provenance = replay_provenance(replay, replay_path)
    expected_seed = int(stream_row["logical_seed"])
    checks = {
        "fresh_replay": provenance.get("replay_origin") == ORIGIN_FRESH,
        "fresh_root": provenance.get("root_origin") == ORIGIN_FRESH,
        "reset_invariant": bool(provenance.get("replay_reset_invariant")),
        "root_seed": int(provenance.get("root_seed")) == expected_seed,
        "replay_seed": int(replay.get("seed")) == expected_seed,
        "starter": int(replay.get("starter_tile")) == STARTER_TILE,
        "dashboard_ineligible": replay.get("dashboard_eligible") is False,
    }
    streams = replay.get("rng_streams")
    if not isinstance(streams, dict):
        raise ValueError("O2 pilot replay lacks split stream metadata")
    checks["deck_stream"] = int(streams.get("deck_stream_id")) == int(
        stream_row["deck_stream_id"]
    )
    checks["slot_stream"] = int(streams.get("slot_stream_id")) == int(
        stream_row["slot_stream_id"]
    )
    checks["policy_stream"] = int(streams.get("policy_stream_id")) == int(
        stream_row["policy_stream_id"]
    )
    if not all(checks.values()):
        raise ValueError(f"O2 pilot replay provenance mismatch: {checks}")
    return {
        "family": str(stream_row["family"]),
        "nominal_family": str(stream_row["family"]),
        "family_index": int(stream_row["family_index"]),
        "game_index": int(stream_row["family_game_index"]),
        "logical_seed": expected_seed,
        "deck_stream_id": int(stream_row["deck_stream_id"]),
        "slot_stream_id": int(stream_row["slot_stream_id"]),
        "policy_stream_id": int(stream_row["policy_stream_id"]),
        "source_replay": str(replay_path),
        "source_replay_sha256": sha256_path(replay_path),
        "root_cluster": canonical_ancestry_id(replay, replay_path),
        "complete": True,
        "dashboard_eligible": False,
    }


def _store_output(
    output: Any,
    *,
    stream_row: Mapping[str, Any],
) -> dict[str, Any]:
    replay = output.replay
    if not isinstance(replay, dict):
        raise ValueError("O2 pilot evaluator omitted replay capture")
    replay_path = _replay_path(stream_row)
    if replay_path.exists():
        raise FileExistsError(f"O2 pilot replay already exists: {replay_path}")
    replay.update(
        direct_root_fields(
            origin=ORIGIN_FRESH,
            seed=int(stream_row["logical_seed"]),
            policy=str(stream_row["family"]),
            replay_path=replay_path,
            first_score=None,
        )
    )
    replay["behavior_family"] = str(stream_row["family"])
    replay["nominal_family"] = str(stream_row["family"])
    replay["acquisition_policy_spec"] = POLICY_SPECS[str(stream_row["family"])]
    replay["o2_yield_pilot"] = True
    replay["dashboard_eligible"] = False
    _atomic_new_json(replay_path, replay)
    stored = json.loads(replay_path.read_text())
    return _completion_from_replay(
        stored,
        replay_path=replay_path,
        stream_row=stream_row,
    )


def _guard_execution(runtime: Mapping[str, Any]) -> None:
    operations = operational_audit(out_dir=OUTPUT_DIR)
    checks = dict(operations["checks"])
    checks["active_runtime"] = (
        float(runtime["active_runtime_seconds"]) < ACTIVE_RUNTIME_LIMIT
    )
    if not all(checks.values()):
        raise history.AcquisitionPause(
            "HOLD_O2_PILOT_OPERATIONAL_INTEGRITY",
            f"O2 pilot operational guard failed: {checks}",
        )


def collect_all(marker: Mapping[str, Any], *, jobs: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in marker["stream_rows"]]
    chunks = round_robin_chunks(rows)
    completions = _load_completions()
    _verify_existing_completions(completions, rows)
    attempts = _load_attempt_ledger(rows, chunks)
    _reconcile_attempts_and_replays(completions, rows, attempts)
    runtime = _runtime_state()
    REPLAY_DIR.mkdir(exist_ok=True)
    policies: dict[str, Any] = {}

    for chunk_index, chunk in enumerate(chunks):
        pending = [
            row
            for row in chunk
            if (str(row["family"]), int(row["family_game_index"]))
            not in completions
        ]
        if not pending:
            continue
        _guard_execution(runtime)
        family = str(chunk[0]["family"])
        if any(str(row["family"]) != family for row in chunk):
            raise ValueError("O2 pilot chunk mixed families")
        if family not in policies:
            policies[family] = family_source.load_policy(
                HISTORICAL_FAMILIES[family],
                POLICY_SPECS[family],
            )
        attempt_by_key = {
            _stream_key(row): _open_attempt(
                attempts,
                row,
                chunk_index=chunk_index,
            )
            for row in pending
        }
        jobs_rows = [
            EvalJob(
                index=index,
                seed=int(row["logical_seed"]),
                starter_tile=STARTER_TILE,
                stream_ids=EvalStreamIds(
                    deck_stream_id=int(row["deck_stream_id"]),
                    slot_stream_id=int(row["slot_stream_id"]),
                    policy_stream_id=int(row["policy_stream_id"]),
                ),
            )
            for index, row in enumerate(pending)
        ]
        started = time.perf_counter()
        try:
            outputs = list(
                iter_eval_job_outputs(
                    policy=policies[family],
                    policy_name=POLICY_SPECS[family],
                    eval_jobs=jobs_rows,
                    max_moves=MAX_MOVES,
                    capture_replay=True,
                    jobs=jobs,
                )
            )
        finally:
            elapsed = time.perf_counter() - started
            runtime["active_runtime_seconds"] += elapsed
            runtime["evaluation_batches_charged"] = (
                int(runtime.get("evaluation_batches_charged", 0)) + 1
            )
            runtime["games_evaluated_charged"] = (
                int(runtime.get("games_evaluated_charged", 0)) + len(pending)
            )
            # Charge evaluator work before replay writes so a same-marker
            # recovery may overcount, but can never lose already-spent time.
            write_json(RUNTIME_PATH, runtime)
        if len(outputs) != len(pending):
            raise ValueError("O2 pilot evaluator returned incomplete chunk")
        for output in sorted(outputs, key=lambda value: value.index):
            row = pending[int(output.index)]
            completion = _store_output(output, stream_row=row)
            history._append_jsonl_row(COMPLETION_PATH, completion)
            key = _stream_key(row)
            completions[key] = completion
            _append_attempt_status(attempt_by_key[key], "completed")
        runtime["chunks_completed"] = int(runtime["chunks_completed"]) + 1
        runtime["games_completed"] = len(completions)
        write_json(RUNTIME_PATH, runtime)

    expected = {
        (str(row["family"]), int(row["family_game_index"])) for row in rows
    }
    if set(completions) != expected:
        raise ValueError("O2 pilot did not complete the exact frozen manifest")
    result = [completions[key] for key in sorted(expected)]
    roots = [str(row["root_cluster"]) for row in result]
    checks = {
        "exact_128": len(result) == TOTAL_ROOTS,
        "equal_families": Counter(row["family"] for row in result)
        == {family: PILOT_ROOTS_PER_FAMILY for family in FAMILY_ORDER},
        "unique_ancestries": len(roots) == len(set(roots)),
        "all_complete": all(row["complete"] for row in result),
        "all_replays_retained": all(
            Path(row["source_replay"]).is_file() for row in result
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"O2 pilot completion integrity failed: {checks}")
    return result


def attempt_ledger_audit(
    rows: Sequence[Mapping[str, Any]],
    completions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    chunks = round_robin_chunks(rows)
    attempts = _load_attempt_ledger(rows, chunks)
    expected = {_stream_key(row) for row in rows}
    completion_keys = {_stream_key(row) for row in completions}
    statuses = Counter()
    opened = 0
    final_completed = 0
    for key, rows_attempts in attempts.items():
        opened += len(rows_attempts)
        for attempt in rows_attempts:
            statuses.update(attempt["statuses"])
        if rows_attempts and rows_attempts[-1]["statuses"] in (
            ["opened", "completed"],
            ["opened", "completed_recovered"],
        ):
            final_completed += 1
    checks = {
        "every_frozen_root_attempted": set(attempts) == expected,
        "every_frozen_root_completed": completion_keys == expected,
        "one_final_completion_per_root": final_completed == len(expected),
        "all_attempts_paired": all(
            len(attempt["statuses"]) == 2
            for rows_attempts in attempts.values()
            for attempt in rows_attempts
        ),
        "no_hidden_retries": opened
        == statuses["completed"]
        + statuses["completed_recovered"]
        + statuses["interrupted_no_replay"],
    }
    return {
        "attempt_rows_path": str(ATTEMPT_PATH),
        "attempt_rows_file_sha256": sha256_path(ATTEMPT_PATH),
        "attempt_events": sum(statuses.values()),
        "attempts_opened": opened,
        "retries": opened - len(expected),
        "status_counts": dict(sorted(statuses.items())),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _selection_hash(
    *,
    stage: int,
    target: int,
    family: str,
    root: str,
    frame: int,
    state_hash: str,
) -> str:
    return hashlib.sha256(
        "|".join(
            (
                "O2-pilot-exact-v1",
                str(stage),
                str(target),
                family,
                root,
                str(frame),
                state_hash,
            )
        ).encode("utf-8")
    ).hexdigest()


def scan_support(
    completions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(completions) != TOTAL_ROOTS:
        raise ValueError("O2 support scan requires all 128 completed roots")
    best: dict[tuple[str, int, int], dict[str, Any]] = {}
    provenance_rows = []
    seen_roots = set()
    frames_scanned = 0
    for completion in sorted(
        completions,
        key=lambda row: (int(row["family_index"]), int(row["game_index"])),
    ):
        path = Path(str(completion["source_replay"]))
        if not path.is_file() or sha256_path(path) != completion[
            "source_replay_sha256"
        ]:
            raise ValueError(f"O2 retained replay changed: {path}")
        replay = json.loads(path.read_text())
        provenance = replay_provenance(replay, path)
        root = canonical_ancestry_id(replay, path)
        family = str(completion["family"])
        checks = {
            "root_matches": root == completion["root_cluster"],
            "fresh_replay": provenance.get("replay_origin") == ORIGIN_FRESH,
            "fresh_root": provenance.get("root_origin") == ORIGIN_FRESH,
            "reset": bool(provenance.get("replay_reset_invariant")),
            "seed": int(provenance.get("root_seed"))
            == int(completion["logical_seed"]),
        }
        if not all(checks.values()) or root in seen_roots:
            raise ValueError(f"O2 support provenance failed: {checks}")
        seen_roots.add(root)
        provenance_rows.append(
            {
                "root_cluster": root,
                "family": family,
                "source_replay": str(path),
                "source_replay_sha256": completion["source_replay_sha256"],
                "checks": checks,
            }
        )
        validator = ThreesSim.from_stream_ids(
            deck_stream_id=int(completion["deck_stream_id"]),
            slot_stream_id=int(completion["slot_stream_id"]),
            starter_tile=STARTER_TILE,
        )
        frames = replay.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"O2 replay has no frames: {path}")
        for fallback, frame in enumerate(frames):
            if not isinstance(frame, dict):
                raise ValueError(f"O2 malformed frame: {path}:{fallback}")
            payload = frame.get("state")
            if not isinstance(payload, dict):
                raise ValueError(f"O2 malformed state: {path}:{fallback}")
            if bool(payload.get("game_over")):
                continue
            state = state_from_replay_payload(payload)
            legal = validator.legal_actions(state)
            expected_names = [DIRECTION_NAMES[action] for action in legal]
            if payload.get("legal_actions") != expected_names:
                raise ValueError(f"O2 legal-action mismatch: {path}:{fallback}")
            if not (
                anchor_safe(state.board, STARTER_TILE)
                and air_safe(state.board)
                and len(legal) >= 2
            ):
                continue
            frame_index = int(frame.get("index", fallback))
            state_hash = state_signature(payload, STARTER_TILE)
            frames_scanned += 1
            for target in SCAN_TARGETS:
                pair = geometry(
                    state.board,
                    STARTER_TILE,
                    fixed_target=target,
                )
                if pair is None:
                    continue
                selection = _selection_hash(
                    stage=pair.stage,
                    target=target,
                    family=family,
                    root=root,
                    frame=frame_index,
                    state_hash=state_hash,
                )
                row = {
                    "root_cluster": root,
                    "family": family,
                    "family_index": int(completion["family_index"]),
                    "target": target,
                    "stage": int(pair.stage),
                    "frame_index": frame_index,
                    "state_sha1": state_hash,
                    "selection_sha256": selection,
                    "pair": [list(coord) for coord in pair.pair],
                    "safe_merge_actions": list(pair.safe_merge_actions),
                    "anchor_safe": True,
                    "air_safe": True,
                    "empty_count": int(np.count_nonzero(state.board == 0)),
                    "legal_count": len(legal),
                    "source_replay": str(path),
                    "source_replay_sha256": completion[
                        "source_replay_sha256"
                    ],
                }
                key = (root, int(pair.stage), target)
                prior = best.get(key)
                candidate_key = (
                    selection,
                    frame_index,
                    state_hash,
                )
                if prior is None or candidate_key < (
                    prior["selection_sha256"],
                    int(prior["frame_index"]),
                    prior["state_sha1"],
                ):
                    best[key] = row
    candidates = sorted(
        best.values(),
        key=lambda row: (
            TARGET_ORDER.index(row["target"])
            if row["target"] in TARGET_ORDER
            else len(TARGET_ORDER),
            row["stage"],
            row["selection_sha256"],
            row["root_cluster"],
        ),
    )
    audit = {
        "completion_roots": len(completions),
        "unique_roots": len(seen_roots),
        "frames_scanned": frames_scanned,
        "candidate_rows": len(candidates),
        "provenance_rows": provenance_rows,
        "provenance_manifest_sha256": canonical_json_hash(provenance_rows),
        "score_field_access": (
            "reset/root score may be read only inside replay_provenance; "
            "final/future score and score-derived selection are not read"
        ),
        "recorded_actions_accessed": False,
        "final_or_future_milestone_fields_accessed": False,
        "policy_outcomes_compared": False,
        "passes": len(seen_roots) == TOTAL_ROOTS,
    }
    return candidates, audit


def _cell_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["target"]), int(row["stage"])


def _cell_order() -> list[tuple[int, int]]:
    return [(target, stage) for target in TARGET_ORDER for stage in STAGE_ORDER]


def structural_objective(edge_count: int, presence_count: int) -> np.ndarray:
    if edge_count < 0 or presence_count < 0:
        raise ValueError("Structural objective counts must be nonnegative")
    objective = np.zeros(edge_count + presence_count, dtype=float)
    if edge_count:
        indices = np.arange(edge_count, dtype=float)
        objective[:edge_count] = (indices + 1.0) / (edge_count + 1.0)
        objective[:edge_count] += indices / (edge_count + 1.0) ** 3
    if presence_count:
        objective[edge_count:] = 1e-9 * np.arange(
            1, presence_count + 1, dtype=float
        )
    return objective


def solve_structural_match(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = _cell_order()
    eligible = [
        dict(row)
        for row in candidates
        if int(row["target"]) in TARGET_ORDER
    ]
    edges = sorted(
        eligible,
        key=lambda row: (
            cells.index(_cell_key(row)),
            row["selection_sha256"],
            FAMILY_ORDER.index(str(row["family"])),
            str(row["root_cluster"]),
        ),
    )
    roots = sorted({str(row["root_cluster"]) for row in edges})
    edge_count = len(edges)
    y_keys = [
        (cell, family) for cell in cells for family in FAMILY_ORDER
    ]
    y_offset = edge_count
    variable_count = edge_count + len(y_keys)
    if any(
        sum(_cell_key(row) == cell for row in edges)
        < STRUCTURAL_QUOTA[cell[0]]
        for cell in cells
    ):
        return {
            "feasible": False,
            "reason": "cell_candidate_shortfall",
            "selected": [],
            "solver": "scipy.optimize.milp-1.17.1",
            "passes": False,
        }

    rows_data: list[dict[int, float]] = []
    lower = []
    upper = []

    def add_constraint(
        coefficients: Mapping[int, float],
        minimum: float,
        maximum: float,
    ) -> None:
        rows_data.append(dict(coefficients))
        lower.append(minimum)
        upper.append(maximum)

    for root in roots:
        add_constraint(
            {
                index: 1.0
                for index, row in enumerate(edges)
                if row["root_cluster"] == root
            },
            0.0,
            1.0,
        )
    for cell in cells:
        quota = STRUCTURAL_QUOTA[cell[0]]
        add_constraint(
            {
                index: 1.0
                for index, row in enumerate(edges)
                if _cell_key(row) == cell
            },
            float(quota),
            float(quota),
        )
        y_indices = []
        for family in FAMILY_ORDER:
            y_index = y_offset + y_keys.index((cell, family))
            y_indices.append(y_index)
            x_indices = [
                index
                for index, row in enumerate(edges)
                if _cell_key(row) == cell and row["family"] == family
            ]
            cap = STRUCTURAL_FAMILY_CAP[cell[0]]
            add_constraint(
                {
                    **{index: 1.0 for index in x_indices},
                    y_index: -float(cap),
                },
                -math.inf,
                0.0,
            )
            add_constraint(
                {
                    **{index: -1.0 for index in x_indices},
                    y_index: 1.0,
                },
                -math.inf,
                0.0,
            )
        add_constraint(
            {index: 1.0 for index in y_indices},
            3.0,
            math.inf,
        )

    matrix = sparse.lil_matrix((len(rows_data), variable_count), dtype=float)
    for row_index, coefficients in enumerate(rows_data):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
    objective = structural_objective(edge_count, len(y_keys))
    constraints = optimize.LinearConstraint(
        matrix.tocsr(),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )

    def run_solver() -> optimize.OptimizeResult:
        return optimize.milp(
            c=objective,
            integrality=np.ones(variable_count, dtype=np.int8),
            bounds=optimize.Bounds(
                np.zeros(variable_count),
                np.ones(variable_count),
            ),
            constraints=constraints,
            options={"presolve": True},
        )

    first = run_solver()
    if first.status == 2:
        return {
            "feasible": False,
            "reason": "milp_infeasible",
            "solver": "scipy.optimize.milp-1.17.1",
            "solver_status": int(first.status),
            "solver_message": str(first.message),
            "selected": [],
            "passes": False,
        }
    if not first.success or first.x is None:
        raise RuntimeError(
            f"O2 structural MILP failed: {first.status}:{first.message}"
        )
    second = run_solver()
    if not second.success or second.x is None:
        raise RuntimeError("O2 structural MILP repeat failed")
    first_all_bits = tuple(int(value >= 0.5) for value in first.x)
    second_all_bits = tuple(int(value >= 0.5) for value in second.x)
    if first_all_bits != second_all_bits:
        raise RuntimeError("O2 structural MILP was not deterministic")
    first_bits = first_all_bits[:edge_count]
    rounded = np.asarray(first_all_bits, dtype=float)
    constraint_values = np.asarray(matrix.tocsr() @ rounded).reshape(-1)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    lower_ok = np.logical_or(
        ~np.isfinite(lower_array),
        constraint_values >= lower_array,
    )
    upper_ok = np.logical_or(
        ~np.isfinite(upper_array),
        constraint_values <= upper_array,
    )
    post_rounding_constraints_exact = bool(
        np.all(lower_ok) and np.all(upper_ok)
    )
    max_binary_residual = float(
        np.max(
            np.minimum(
                np.abs(np.asarray(first.x, dtype=float)),
                np.abs(np.asarray(first.x, dtype=float) - 1.0),
            )
        )
    )
    if not post_rounding_constraints_exact or max_binary_residual > 1e-7:
        raise RuntimeError("O2 structural MILP post-rounding audit failed")
    selected = [
        edges[index] for index, bit in enumerate(first_bits) if bit
    ]
    roots_selected = [str(row["root_cluster"]) for row in selected]
    cell_reports = []
    for cell in cells:
        rows = [row for row in selected if _cell_key(row) == cell]
        family_counts = Counter(str(row["family"]) for row in rows)
        quota = STRUCTURAL_QUOTA[cell[0]]
        cap = STRUCTURAL_FAMILY_CAP[cell[0]]
        cell_reports.append(
            {
                "target": cell[0],
                "stage": cell[1],
                "quota": quota,
                "selected_count": len(rows),
                "family_counts": dict(sorted(family_counts.items())),
                "minimum_families": 3,
                "maximum_per_family": cap,
                "passes": len(rows) == quota
                and len(family_counts) >= 3
                and max(family_counts.values(), default=0) <= cap,
            }
        )
    checks = {
        "selected_92": len(selected) == 92,
        "root_disjoint": len(roots_selected) == len(set(roots_selected)),
        "all_cells": all(row["passes"] for row in cell_reports),
        "repeat_deterministic": first_all_bits == second_all_bits,
        "post_rounding_constraints_exact": post_rounding_constraints_exact,
        "binary_residual_below_1e_7": max_binary_residual <= 1e-7,
    }
    return {
        "feasible": all(checks.values()),
        "reason": "matched" if all(checks.values()) else "postsolve_check_failed",
        "solver": "scipy.optimize.milp-1.17.1",
        "solver_status": int(first.status),
        "solver_message": str(first.message),
        "objective": float(first.fun),
        "objective_sha256": canonical_json_hash(objective.tolist()),
        "candidate_order_sha256": canonical_json_hash(
            [
                {
                    "root_cluster": row["root_cluster"],
                    "family": row["family"],
                    "target": row["target"],
                    "stage": row["stage"],
                    "selection_sha256": row["selection_sha256"],
                }
                for row in edges
            ]
        ),
        "max_binary_residual": max_binary_residual,
        "post_rounding_constraint_count": len(rows_data),
        "candidate_edges": edge_count,
        "selected": selected,
        "selected_manifest_sha256": canonical_json_hash(selected),
        "cell_reports": cell_reports,
        "checks": checks,
        "passes": all(checks.values()),
    }


def availability_report(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cells = []
    for target, stage in _cell_order():
        rows = sorted(
            (
                dict(row)
                for row in candidates
                if int(row["target"]) == target
                and int(row["stage"]) == stage
            ),
            key=lambda row: (
                FAMILY_ORDER.index(str(row["family"])),
                row["selection_sha256"],
                str(row["root_cluster"]),
            ),
        )
        raw_by_family = Counter(str(row["family"]) for row in rows)
        credited = []
        for family in FAMILY_ORDER:
            family_rows = sorted(
                (row for row in rows if row["family"] == family),
                key=lambda row: (
                    row["selection_sha256"],
                    str(row["root_cluster"]),
                ),
            )
            credited.extend(family_rows[:AVAILABILITY_FAMILY_CAP])
        credited_roots = {str(row["root_cluster"]) for row in credited}
        credited_by_family = Counter(str(row["family"]) for row in credited)
        minimum = AVAILABILITY_MINIMUM[target]
        final_demand = AVAILABILITY_FINAL_DEMAND[target]
        wilson = preflight.wilson_lower(minimum, TOTAL_ROOTS)
        required_rate = final_demand / 640
        checks = {
            "minimum_roots": len(credited_roots) >= minimum,
            "minimum_families": len(credited_by_family) >= 3,
            "family_cap": max(credited_by_family.values(), default=0)
            <= AVAILABILITY_FAMILY_CAP,
            "wilson": wilson > required_rate,
            "one_credit_per_root": len(credited_roots) == len(credited),
        }
        cells.append(
            {
                "target": target,
                "stage": stage,
                "raw_distinct_roots": len(
                    {str(row["root_cluster"]) for row in rows}
                ),
                "raw_family_counts": dict(sorted(raw_by_family.items())),
                "credited_distinct_roots": len(credited_roots),
                "credited_family_counts": dict(
                    sorted(credited_by_family.items())
                ),
                "credited_root_ids": sorted(credited_roots),
                "minimum_distinct_roots": minimum,
                "final_disjoint_demand": final_demand,
                "final_rate_required": required_rate,
                "wilson_lower_at_minimum": wilson,
                "checks": checks,
                "passes": all(checks.values()),
            }
        )
    checks = {
        "twenty_cells": len(cells) == 20,
        "all_cells": all(row["passes"] for row in cells),
        "fully_disjoint_144_rejected": 16 * 7 + 4 * 8 > TOTAL_ROOTS,
    }
    return {
        "cells": cells,
        "checks": checks,
        "passes": all(checks.values()),
    }


def descriptive_1536(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in candidates if int(row["target"]) == 1536]
    raw_counts = {
        str(stage): dict(
            sorted(
                Counter(
                    str(row["family"])
                    for row in rows
                    if int(row["stage"]) == stage
                ).items()
            )
        )
        for stage in STAGE_ORDER
    }
    selected = []
    used_roots = set()
    for stage in STAGE_ORDER:
        available = sorted(
            (row for row in rows if int(row["stage"]) == stage),
            key=lambda row: (
                hashlib.sha256(
                    "|".join(
                        (
                            "O2-1536-descriptive-v1",
                            str(stage),
                            str(row["family"]),
                            str(row["root_cluster"]),
                            str(row["frame_index"]),
                            str(row["state_sha1"]),
                        )
                    ).encode("utf-8")
                ).hexdigest(),
                str(row["root_cluster"]),
            ),
        )
        for row in available:
            if row["root_cluster"] in used_roots:
                continue
            selected.append(row)
            used_roots.add(row["root_cluster"])
            if sum(int(value["stage"]) == stage for value in selected) == 4:
                break
    return {
        "readiness_gate": False,
        "raw_family_counts_by_stage": raw_counts,
        "selected": selected,
        "selected_count": len(selected),
        "selected_manifest_sha256": canonical_json_hash(selected),
        "checks": {
            "maximum_16": len(selected) <= 16,
            "maximum_4_per_stage": all(
                sum(int(row["stage"]) == stage for row in selected) <= 4
                for stage in STAGE_ORDER
            ),
            "one_per_root": len(used_roots) == len(selected),
        },
    }


def support_analysis(
    completions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    roots = [str(row.get("root_cluster")) for row in completions]
    if len(completions) != TOTAL_ROOTS or len(set(roots)) != TOTAL_ROOTS:
        raise ValueError(
            "O2 support analysis requires exactly 128 unique completions"
        )
    candidates, scan = scan_support(completions)
    structural = solve_structural_match(candidates)
    availability = availability_report(candidates)
    descriptive = descriptive_1536(candidates)
    support_passes = structural["passes"] and availability["passes"]
    return {
        "version": VERSION,
        "candidate_rows": candidates,
        "candidate_manifest_sha256": canonical_json_hash(candidates),
        "scan_audit": scan,
        "structural_layer": structural,
        "availability_layer": availability,
        "descriptive_1536": descriptive,
        "support_passes": support_passes,
        "decision": (
            "READY_O2_CORPUS_COLLECTION"
            if support_passes
            else "HOLD_O2_DATA_SUPPORT"
        ),
    }


def _terminal_zero_forbidden_work() -> dict[str, Any]:
    return {
        "corpus_games_beyond_pilot": 0,
        "option_rollouts": 0,
        "labels": 0,
        "model_fits": 0,
        "policy_outcome_evaluations": 0,
        "score_outcomes_inspected": 0,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
    }


def execute(*, out_dir: Path = OUTPUT_DIR, jobs: int = FROZEN_JOBS) -> dict[str, Any]:
    marker = _load_marker(out_dir=out_dir, jobs=jobs)
    try:
        if not immutable_input_audit()["passes"]:
            raise ValueError("O2 immutable inputs changed before execution")
        if not preflight.family_evidence_audit()["passes"]:
            raise ValueError("O2 family evidence changed before execution")
        collision = collision_audit(marker["stream_rows"], out_dir=out_dir)
        if not collision["passes"]:
            raise ValueError("O2 pilot stream collision appeared")
        _guard_execution(_runtime_state())
        completions = collect_all(marker, jobs=jobs)
        _guard_execution(_runtime_state())
        attempts = attempt_ledger_audit(
            marker["stream_rows"],
            completions,
        )
        if not attempts["passes"]:
            raise ValueError("O2 pilot attempt ledger integrity failed")
        support = support_analysis(completions)
        _atomic_new_json(SUPPORT_PATH, support)
        terminal_operations = operational_audit(out_dir=out_dir)
        if not terminal_operations["passes"]:
            raise ValueError("O2 terminal operational audit failed")
        completion_rows = [
            dict(row)
            for row in sorted(
                completions,
                key=lambda row: (
                    int(row["family_index"]),
                    int(row["game_index"]),
                ),
            )
        ]
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": support["decision"],
            "terminal_status": "HOLD_O2_AFTER_YIELD_PILOT_SEAL",
            "continue": (
                "O2_CORPUS_COLLECTION_REQUIRES_SEPARATE_AUTHORIZATION"
                if support["decision"] == "READY_O2_CORPUS_COLLECTION"
                else "NONE"
            ),
            "hold": [
                "corpus_collection",
                "option_rollouts",
                "labels",
                "model_fit",
                "policy_evaluation",
                "promotion",
            ],
            "kill": False,
            "promote": False,
            "marker": preflight.artifact_identity(MARKER_PATH),
            "attempt_ledger": attempts,
            "support": preflight.artifact_identity(SUPPORT_PATH),
            "completion_rows": len(completion_rows),
            "completion_manifest_sha256": canonical_json_hash(completion_rows),
            "games_by_family": dict(
                sorted(Counter(row["family"] for row in completion_rows).items())
            ),
            "unique_ancestries": len(
                {row["root_cluster"] for row in completion_rows}
            ),
            "runtime": _runtime_state(),
            "output_bytes": history._directory_bytes(out_dir),
            "collision_audit": collision,
            "terminal_operations": terminal_operations,
            "support_decision": support["decision"],
            "zero_forbidden_work": _terminal_zero_forbidden_work(),
            "dashboard_eligible": False,
        }
        _atomic_new_json(RESULT_PATH, result)
        return json.loads(RESULT_PATH.read_text())
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        if RESULT_PATH.exists():
            raise
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_O2_PILOT_OPERATIONAL_INTEGRITY",
            "terminal_status": "HOLD_O2_AFTER_YIELD_PILOT_SEAL",
            "continue": "NONE",
            "hold": [
                "pilot_retry",
                "corpus_collection",
                "option_rollouts",
                "labels",
                "model_fit",
                "policy_evaluation",
                "promotion",
            ],
            "kill": False,
            "promote": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "marker": preflight.artifact_identity(MARKER_PATH),
            "attempt_rows": (
                sum(
                    1
                    for line in ATTEMPT_PATH.read_text().splitlines()
                    if line.strip()
                )
                if ATTEMPT_PATH.is_file()
                else 0
            ),
            "completion_rows": len(_load_completions()),
            "runtime": _runtime_state(),
            "output_bytes": (
                history._directory_bytes(out_dir) if out_dir.exists() else 0
            ),
            "zero_forbidden_work": _terminal_zero_forbidden_work(),
            "dashboard_eligible": False,
        }
        _atomic_new_json(RESULT_PATH, result)
        return json.loads(RESULT_PATH.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tests = subparsers.add_parser("seal-test-evidence")
    tests.add_argument("--focused-passed", type=int, required=True)
    tests.add_argument("--regression-passed", type=int, required=True)
    tests.add_argument(
        "--test-command",
        action="append",
        dest="test_commands",
        required=True,
    )

    open_parser = subparsers.add_parser("open")
    open_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    open_parser.add_argument("--jobs", type=int, default=FROZEN_JOBS)

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    execute_parser.add_argument("--jobs", type=int, default=FROZEN_JOBS)

    args = parser.parse_args()
    if args.command == "seal-test-evidence":
        result = seal_test_evidence(
            focused_passed=args.focused_passed,
            regression_passed=args.regression_passed,
            commands=args.test_commands,
        )
    elif args.command == "open":
        result = open_execution(out_dir=args.out_dir, jobs=args.jobs)
    else:
        result = execute(out_dir=args.out_dir, jobs=args.jobs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

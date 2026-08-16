"""Zero-outcome preflight for the G3 E0 v3 open/resume correction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from threes_rl import g3_e0_label_fit as scientific
from threes_rl import g3_e0_label_fit_v3 as runner
from threes_rl import g3_e0_preflight as v1
from threes_rl import g3_e0_preflight_v2 as v2
from threes_rl import g3_scale_transfer_bootstrap_preflight as g3v1
from threes_rl.s3_power_preflight import sha256_path


VERSION = "g3_e0_label_fit_preflight_v3"
IMPLEMENTATION_PATH = Path("threes_rl/g3_e0_preflight_v3.py")
TEST_PATH = Path("tests/test_rl_g3_e0_label_fit_v3.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/g3_e0_label_fit_v3_test_evidence.json"
)
OUTPUT_DIR = runner.OUTPUT_DIR
V2_OUTPUT_DIR = v2.OUTPUT_DIR
V2_LOCK_PATH = V2_OUTPUT_DIR / "preflight_lock.json"
V2_LOCK_FILE_SHA256 = (
    "fde44d089b3d59e97e417d101ce1fbfeedc51874de4b74a1bc4ca0583ae62ef7"
)
V2_LOCK_PAYLOAD_SHA256 = (
    "c7ec0d0d1feab5dc1fd3564d63d11dd624a8354763661ead45dcb5e87446e2bb"
)
V2_MANIFEST_HASHES = {
    "E0_RECORD_MANIFEST.json":
        "90a4f55ff29f51c0d6ac35375650258188b6961debd6cbcc546382762547d9d5",
    "E0_TASK_MANIFEST.json":
        "087fd68c71421c8402360a1c096b476cb1bf494de7d8c8f025e7e699bf97bd2f",
    "E0_STREAM_MANIFEST.json":
        "e40b7dd3744dd0df04f621034894656568991291c17490e27e8c3a93e189ea05",
}
V2_ALLOWED_FILES = set(V2_MANIFEST_HASHES) | {"preflight_lock.json"}
COLLISION_EXCLUSIONS = (
    v1.V1_OUTPUT_DIR,
    v1.V2_OUTPUT_DIR,
    v2.FAILED_V1_OUTPUT_DIR,
    V2_OUTPUT_DIR,
    OUTPUT_DIR,
)


def _locked_rows(
    locks: Mapping[Path, str],
) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    for path, expected in locks.items():
        actual = sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    return rows, bool(rows) and all(row["matches"] for row in rows)


def _v2_zero_work_audit() -> dict[str, Any]:
    files = sorted(
        str(path.relative_to(V2_OUTPUT_DIR))
        for path in V2_OUTPUT_DIR.rglob("*")
        if path.is_file()
    )
    checks = {
        "v2_directory_exists": V2_OUTPUT_DIR.is_dir(),
        "only_preflight_files": set(files) == V2_ALLOWED_FILES,
        "marker_absent":
            not (V2_OUTPUT_DIR / scientific.OPEN_MARKER_NAME).exists(),
        "ordinary_database_absent":
            not (V2_OUTPUT_DIR / scientific.ORDINARY_DB_NAME).exists(),
        "transfer_database_absent":
            not (V2_OUTPUT_DIR / scientific.TRANSFER_DB_NAME).exists(),
        "checkpoint_absent":
            not (V2_OUTPUT_DIR / scientific.MODEL_DIR_NAME).exists(),
        "prediction_absent":
            not (
                V2_OUTPUT_DIR / scientific.PREDICTION_SEAL_NAME
            ).exists(),
        "terminal_absent":
            not (V2_OUTPUT_DIR / scientific.TERMINAL_RESULT_NAME).exists(),
    }
    return {
        "files": files,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _test_evidence_audit() -> dict[str, Any]:
    if not TEST_EVIDENCE_PATH.is_file():
        return {
            "path": str(TEST_EVIDENCE_PATH),
            "passes": False,
            "reason": "missing",
        }
    evidence = scientific.json_object(TEST_EVIDENCE_PATH)
    rows = []
    for row in evidence.get("bound_files", []):
        path = Path(str(row["path"]))
        actual = sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expected_sha256": str(row["sha256"]),
                "actual_sha256": actual,
                "matches": actual == row["sha256"],
            }
        )
    checks = {
        "version_exact":
            evidence.get("version") == "g3_e0_v3_test_evidence_v1",
        "tests_pass": evidence.get("passes") is True,
        "bound_files_exact": bool(rows) and all(
            row["matches"] for row in rows
        ),
        "focused_tests_present":
            int(evidence.get("focused_tests_passed", 0)) >= 1,
        "regression_tests_present":
            int(evidence.get("regression_tests_passed", 0)) >= 113,
        "zero_forbidden_work":
            evidence.get("zero_forbidden_work", {}).get(
                "label_paths_generated"
            )
            == 0
            and evidence.get("zero_forbidden_work", {}).get(
                "models_fit"
            )
            == 0
            and evidence.get("zero_forbidden_work", {}).get(
                "outcomes_opened"
            )
            == 0,
    }
    return {
        "path": str(TEST_EVIDENCE_PATH),
        "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "payload_sha256": evidence.get("canonical_payload_sha256"),
        "rows": rows,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _input_audit() -> dict[str, Any]:
    locks = {
        scientific.CHARTER_PATH: scientific.CHARTER_SHA256,
        runner.V2_AMENDMENT_PATH: runner.V2_AMENDMENT_SHA256,
        runner.AMENDMENT_PATH: runner.AMENDMENT_SHA256,
        runner.SCIENTIFIC_RUNNER_PATH: runner.SCIENTIFIC_RUNNER_SHA256,
        V2_LOCK_PATH: V2_LOCK_FILE_SHA256,
    }
    for name, expected in V2_MANIFEST_HASHES.items():
        locks[V2_OUTPUT_DIR / name] = expected
    rows, exact = _locked_rows(locks)
    v2_lock = scientific.json_object(V2_LOCK_PATH)
    checks = {
        "locked_files_exact": exact,
        "v2_payload_exact":
            scientific.verify_payload_hash(v2_lock)
            and v2_lock.get("canonical_payload_sha256")
            == V2_LOCK_PAYLOAD_SHA256,
        "v2_ready":
            v2_lock.get("decision")
            == "READY_G3_E0_LABEL_FIT_EXECUTION",
        "v2_zero_work": _v2_zero_work_audit()["passes"],
    }
    return {
        "rows": rows,
        "v2_zero_work": _v2_zero_work_audit(),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _manifest_payloads() -> dict[str, dict[str, Any]]:
    payloads = {}
    for name, expected in V2_MANIFEST_HASHES.items():
        path = V2_OUTPUT_DIR / name
        if sha256_path(path) != expected:
            raise ValueError(f"Frozen v2 manifest changed: {name}")
        payload = scientific.json_object(path)
        if not scientific.verify_payload_hash(payload):
            raise ValueError(f"Frozen v2 manifest payload changed: {name}")
        payloads[name] = payload
    return payloads


def _record_and_task_audit(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = [
        dict(row)
        for row in payloads["E0_RECORD_MANIFEST.json"]["records"]
    ]
    tasks = [
        dict(row)
        for row in payloads["E0_TASK_MANIFEST.json"]["tasks"]
    ]
    records_audit = v2.validate_records_v2(records)
    coupling = scientific.task_coupling_audit(tasks)
    by_partition = dict(
        sorted(Counter(str(task["partition"]) for task in tasks).items())
    )
    roots = {str(record["root_cluster"]) for record in records}
    checks = {
        "record_integrity": records_audit["passes"],
        "record_count": len(records) == v1.EXPECTED_RECORDS,
        "root_count": len(roots) == 384,
        "task_count": len(tasks) == v1.EXPECTED_E0_PATHS,
        "partition_counts":
            by_partition == v1.EXPECTED_E0_BY_PARTITION,
        "replicates":
            {int(task["replicate"]) for task in tasks}
            == set(scientific.E0_REPLICATES),
        "coupling": coupling["passes"],
    }
    return {
        "records": records_audit,
        "tasks": len(tasks),
        "roots": len(roots),
        "by_partition": by_partition,
        "coupling": coupling,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _incumbent_audit(v2_lock: Mapping[str, Any]) -> dict[str, Any]:
    transfer_lock = scientific.json_object(
        Path(str(v2_lock["transfer_preflight_lock_path"]))
    )
    return g3v1._verify_incumbent_artifacts(transfer_lock)


def _collision_audit(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    tasks = [
        dict(row)
        for row in payloads["E0_TASK_MANIFEST.json"]["tasks"]
    ]
    audit = v1.historical_collision_audit(
        tasks,
        excluded_exact_directories=COLLISION_EXCLUSIONS,
    )
    expected_exclusions = {
        str(path.resolve()) for path in COLLISION_EXCLUSIONS
    }
    checks = {
        "collisions_zero": audit["passes"],
        "exclusions_exact":
            set(audit["excluded_exact_directories"])
            == expected_exclusions,
    }
    return {
        "audit": audit,
        "checks": checks,
        "passes": all(checks.values()),
    }


def build_preflight_payload(
    *,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    inputs = _input_audit()
    payloads = _manifest_payloads()
    structure = _record_and_task_audit(payloads)
    collision = _collision_audit(payloads)
    v2_lock = scientific.json_object(V2_LOCK_PATH)
    incumbent = _incumbent_audit(v2_lock)
    tests = _test_evidence_audit()
    disk = v1.disk_audit(v1.WORKSPACE_ROOT)
    operations = v1.process_and_service_audit()
    runner_hash = sha256_path(runner.RUNNER_PATH)
    preflight_hash = sha256_path(IMPLEMENTATION_PATH)
    test_hash = sha256_path(TEST_PATH)
    incumbent_hash = scientific.canonical_sha256(incumbent)
    cost = dict(v2_lock["cost_projection"])
    cost_checks = {
        "paths_exact": int(cost["paths"]) == v1.EXPECTED_E0_PATHS,
        "runtime_within_18h":
            float(cost["projected_runtime_hours"])
            <= v1.MAX_PROJECTED_RUNTIME_HOURS,
        "storage_below_4gib":
            int(cost["projected_incremental_bytes"])
            < v1.MAX_INCREMENTAL_BYTES,
    }
    integrity_checks = {
        "v2_and_charters_exact": inputs["passes"],
        "manifests_and_sources_exact": structure["passes"],
        "streams_unconsumed_and_collision_free": collision["passes"],
        "incumbent_exact": incumbent["passes"],
        "scientific_schema_exact":
            scientific.schema_sha256() == scientific.SCHEMA_SHA256,
        "focused_and_regression_tests": tests["passes"],
        "runner_hash_available": bool(runner_hash),
        "preflight_hash_available": bool(preflight_hash),
        "test_hash_available": bool(test_hash),
    }
    readiness_checks = {
        "cost_contract": all(cost_checks.values()),
        "disk_above_100gib": disk["above_hard_minimum"],
        "disk_above_120gib_target": disk["above_target"],
        "services_process_nice": operations["passes"],
        "fresh_output": not out_dir.exists(),
    }
    if not all(integrity_checks.values()):
        decision = "KILL_G3_E0_V3_INTEGRITY"
    elif all(readiness_checks.values()):
        decision = "READY_G3_E0_V3_EXECUTION"
    else:
        decision = "HOLD_G3_E0_V3_ORCHESTRATION"

    bound_files = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in inputs["rows"]
    ]
    bound_files.extend(
        [
            {"path": str(runner.RUNNER_PATH), "sha256": runner_hash},
            {"path": str(IMPLEMENTATION_PATH), "sha256": preflight_hash},
            {"path": str(TEST_PATH), "sha256": test_hash},
            {
                "path": str(TEST_EVIDENCE_PATH),
                "sha256": tests.get("file_sha256"),
            },
        ]
    )
    unique_bound = {
        str(row["path"]): str(row["sha256"]) for row in bound_files
    }
    record_payload = payloads["E0_RECORD_MANIFEST.json"]
    task_payload = payloads["E0_TASK_MANIFEST.json"]
    stream_payload = payloads["E0_STREAM_MANIFEST.json"]
    lock = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "terminal_status": "HOLD_G3_AFTER_E0_V3_PREFLIGHT_SEAL",
        "out_dir_resolved": str(out_dir.resolve()),
        "open_command": runner.OPEN_COMMAND,
        "execution_command": runner.EXECUTE_COMMAND,
        "jobs": 1,
        "nice": scientific.MIN_NICE,
        "maximum_active_hours": 18.0,
        "maximum_output_bytes": scientific.MAX_OUTPUT_BYTES,
        "minimum_free_gib": scientific.MIN_FREE_GIB,
        "target_free_gib": 120.0,
        "required_services": [
            "dashboard:8765",
            "advisor:8770",
            "dashboard_record:263670",
            "protected_top_three:263670/261369/258561",
        ],
        "base_charter_sha256": scientific.CHARTER_SHA256,
        "v2_amendment_sha256": runner.V2_AMENDMENT_SHA256,
        "v3_orchestration_amendment_sha256": runner.AMENDMENT_SHA256,
        "scientific_runner_sha256": runner.SCIENTIFIC_RUNNER_SHA256,
        "orchestration_runner_sha256": runner_hash,
        "preflight_implementation_sha256": preflight_hash,
        "focused_test_sha256": test_hash,
        "test_evidence_file_sha256": tests.get("file_sha256"),
        "test_evidence_payload_sha256": tests.get("payload_sha256"),
        "bound_files": [
            {"path": path, "sha256": sha}
            for path, sha in sorted(unique_bound.items())
        ],
        "record_manifest_name": "E0_RECORD_MANIFEST.json",
        "record_manifest_file_sha256":
            V2_MANIFEST_HASHES["E0_RECORD_MANIFEST.json"],
        "record_manifest_payload_sha256":
            record_payload["canonical_payload_sha256"],
        "task_manifest_name": "E0_TASK_MANIFEST.json",
        "task_manifest_file_sha256":
            V2_MANIFEST_HASHES["E0_TASK_MANIFEST.json"],
        "task_manifest_payload_sha256":
            task_payload["canonical_payload_sha256"],
        "stream_manifest_name": "E0_STREAM_MANIFEST.json",
        "stream_manifest_file_sha256":
            V2_MANIFEST_HASHES["E0_STREAM_MANIFEST.json"],
        "stream_manifest_payload_sha256":
            stream_payload["canonical_payload_sha256"],
        "ordinary_record_count": v1.EXPECTED_ORDINARY_RECORDS,
        "transfer_record_count": v1.EXPECTED_TRANSFER_RECORDS,
        "total_path_count": v1.EXPECTED_E0_PATHS,
        "path_counts": dict(v1.EXPECTED_E0_BY_PARTITION),
        "ordinary_path_count":
            v1.EXPECTED_E0_BY_PARTITION["train"]
            + v1.EXPECTED_E0_BY_PARTITION["development"],
        "transfer_path_count":
            v1.EXPECTED_E0_BY_PARTITION["transfer_diagnostic"],
        "replicates": list(scientific.E0_REPLICATES),
        "incumbent_policy_spec": incumbent["incumbent_policy_spec"],
        "incumbent_artifact_audit": incumbent,
        "incumbent_artifact_audit_sha256": incumbent_hash,
        "transfer_preflight_lock_path":
            str(v2_lock["transfer_preflight_lock_path"]),
        "stream_collision_excluded_directories": [
            str(path.resolve()) for path in COLLISION_EXCLUSIONS
        ],
        "stream_collision_lock": {
            "matched_source_count":
                collision["audit"]["matched_source_count"],
            "matched_sources_sha256":
                collision["audit"]["matched_sources_sha256"],
            "excluded_source_count":
                collision["audit"]["excluded_source_count"],
            "excluded_sources_sha256":
                collision["audit"]["excluded_sources_sha256"],
        },
        "input_audit": inputs,
        "structure_audit": structure,
        "stream_collision_audit": collision,
        "test_evidence_audit": tests,
        "cost_projection": cost,
        "cost_checks": cost_checks,
        "disk": disk,
        "operations": operations,
        "integrity_checks": integrity_checks,
        "readiness_checks": readiness_checks,
        "zero_forbidden_work": {
            "new_games": 0,
            "streams_consumed": 0,
            "label_paths_generated": 0,
            "label_values_opened": False,
            "scientific_models_fit": 0,
            "checkpoint_predictions": 0,
            "transfer_outcomes_opened": 0,
            "policy_outcomes": 0,
            "score_inspection": False,
            "incumbent_changed": False,
            "dashboard_changed": False,
        },
        "e0_execution_authorized":
            decision == "READY_G3_E0_V3_EXECUTION",
        "e1_authorized": False,
        "policy_evaluation_authorized": False,
        "promotion_authorized": False,
        "dashboard_eligible": False,
    }
    return lock, payloads


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    staging = out_dir.with_name(out_dir.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        lock, payloads = build_preflight_payload(out_dir=out_dir)
        for name in V2_MANIFEST_HASHES:
            shutil.copyfile(V2_OUTPUT_DIR / name, staging / name)
            if sha256_path(staging / name) != V2_MANIFEST_HASHES[name]:
                raise ValueError(f"Byte-identical manifest copy failed: {name}")
        lock = scientific.payload_with_hash(lock)
        scientific.write_immutable_json(
            staging / "preflight_lock.json", lock
        )
        staging.replace(out_dir)
        return lock
    except Exception as error:
        failure = scientific.payload_with_hash(
            {
                "version": VERSION,
                "decision": "KILL_G3_E0_V3_INTEGRITY",
                "error": f"{type(error).__name__}: {error}",
                "zero_forbidden_work": {
                    "streams_consumed": 0,
                    "label_paths_generated": 0,
                    "models_fit": 0,
                    "predictions": 0,
                    "outcomes_opened": 0,
                },
            }
        )
        scientific.write_immutable_json(
            staging / "PREFLIGHT_FAILURE.json", failure
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if os.nice(0) < scientific.MIN_NICE:
        os.nice(scientific.MIN_NICE - os.nice(0))
    result = run_preflight(args.out_dir)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "canonical_payload_sha256":
                    result["canonical_payload_sha256"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

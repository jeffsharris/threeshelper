"""No-outcome preflight for the G3 E0 breadth-first label/fit screen."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from threes_rl import g1r_acquire as g1r
from threes_rl import g3_scale_transfer_bootstrap_preflight as g3v1
from threes_rl import g3_scale_transfer_bootstrap_preflight_v2 as g3v2
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.g2_scale_relational_hazard import schema_sha256
from threes_rl.g3_e0_label_fit import (
    CHARTER_PATH,
    CHARTER_SHA256,
    E0_REPLICATES,
    OUTPUT_DIR,
    SCHEMA_SHA256,
    VERSION as RUNNER_VERSION,
    build_e0_tasks,
    canonical_sha256,
    feature_rows_for_record,
    json_object,
    payload_with_hash,
    task_coupling_audit,
    verify_payload_hash,
    write_immutable_json,
)
from threes_rl.s3_power_preflight import sha256_path


VERSION = "g3_e0_label_fit_preflight_v1"
IMPLEMENTATION_PATH = Path("threes_rl/g3_e0_label_fit.py")
PREFLIGHT_PATH = Path("threes_rl/g3_e0_preflight.py")
TEST_PATH = Path("tests/test_rl_g3_e0_label_fit.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/g3_e0_label_fit_v1_test_evidence.json"
)
V1_OUTPUT_DIR = g3v2.V1_OUTPUT_DIR
V2_OUTPUT_DIR = g3v2.OUTPUT_DIR
V1_RECORD_MANIFEST_PATH = g3v2.V1_RECORD_MANIFEST_PATH
V1_STREAM_MANIFEST_PATH = g3v2.V1_STREAM_MANIFEST_PATH
V2_PREFLIGHT_PATH = V2_OUTPUT_DIR / "G3_V2_BOOTSTRAP_PREFLIGHT.json"
V2_UNTOUCHED_AUDIT_PATH = (
    V2_OUTPUT_DIR / "G3_V2_CORRECTED_UNTOUCHEDNESS_AUDIT.json"
)
G2_TRANSFER_LOCK_PATH = g3v1.TRANSFER_PREFLIGHT_LOCK_PATH
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = Path("threes_rl/runs")
EXPECTED_RECORDS = 715
EXPECTED_ORDINARY_RECORDS = 683
EXPECTED_TRANSFER_RECORDS = 32
EXPECTED_E0_PATHS = 5_072
EXPECTED_E0_BY_PARTITION = {
    "train": 3_902,
    "development": 944,
    "transfer_diagnostic": 226,
}
V2_PREFLIGHT_FILE_SHA256 = (
    "052985f7e5c13797df43bfd074602169ff5c85618dd0f3db549720fec95f7d66"
)
V2_PREFLIGHT_PAYLOAD_SHA256 = (
    "185390797d66d9997faa801292a89a8a8b24967304204fec42b8fbb1852a93ab"
)
V2_UNTOUCHED_FILE_SHA256 = (
    "8b2fd76e593db0b6af998aa5f49092a5dea285754111b891d3aed85ad80e51ed"
)
V1_RECORD_FILE_SHA256 = (
    "938e903f8d2fefb072af84ac19baf4977e4f4d93bf72e8af7acc174b6974b9ec"
)
V1_RECORD_PAYLOAD_SHA256 = (
    "a78e2fd51ee20a7aeb23c71d9930c33561844357920f4808eeeaff653d49f759"
)
V1_STREAM_FILE_SHA256 = (
    "bdbe562167f304327e52f0593f0958753e8afa949a7b38e15b357492faea5744"
)
V1_STREAM_PAYLOAD_SHA256 = (
    "c2afc3c6fa26c1106a480c58189d9a9b4f9dcf99ac8b506d890ff3c330278caa"
)
MIN_NICE = 10
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
MAX_PROJECTED_RUNTIME_HOURS = 18.0
MAX_INCREMENTAL_BYTES = 4 * 1024**3
FUTURE_COMMAND = (
    "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
    ".venv/bin/python -m threes_rl.g3_e0_label_fit execute "
    "--out-dir threes_rl/runs/forensics/g3_e0_label_fit_v1 "
    "--preflight-lock "
    "threes_rl/runs/forensics/g3_e0_label_fit_v1/preflight_lock.json "
    "--jobs 1'"
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
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    return rows, all(row["matches"] for row in rows)


def upstream_input_audit() -> dict[str, Any]:
    locks = {
        CHARTER_PATH: CHARTER_SHA256,
        V2_PREFLIGHT_PATH: V2_PREFLIGHT_FILE_SHA256,
        V2_UNTOUCHED_AUDIT_PATH: V2_UNTOUCHED_FILE_SHA256,
        V1_RECORD_MANIFEST_PATH: V1_RECORD_FILE_SHA256,
        V1_STREAM_MANIFEST_PATH: V1_STREAM_FILE_SHA256,
        g3v1.CHARTER_PATH: g3v1.CHARTER_SHA256,
        g3v1.AMENDMENT_PATH: g3v1.AMENDMENT_SHA256,
        g3v2.AMENDMENT_PATH: g3v2.AMENDMENT_SHA256,
    }
    rows, files_exact = _locked_rows(locks)
    v2_payload = json_object(V2_PREFLIGHT_PATH)
    record_payload = json_object(V1_RECORD_MANIFEST_PATH)
    stream_payload = json_object(V1_STREAM_MANIFEST_PATH)
    checks = {
        "files_exact": files_exact,
        "v2_payload_exact":
            v2_payload.get("canonical_payload_sha256")
            == V2_PREFLIGHT_PAYLOAD_SHA256
            and verify_payload_hash(v2_payload),
        "v2_ready":
            v2_payload.get("decision") == "READY_G3_V2_BOOTSTRAP_LABELS",
        "v2_zero_forbidden_work":
            v2_payload.get("zero_forbidden_work", {}).get("new_labels") == 0
            and v2_payload.get("zero_forbidden_work", {}).get("models_fit") == 0
            and v2_payload.get("zero_forbidden_work", {}).get(
                "transfer_outcomes_opened"
            )
            == 0,
        "record_payload_exact":
            record_payload.get("canonical_payload_sha256")
            == V1_RECORD_PAYLOAD_SHA256
            and verify_payload_hash(record_payload),
        "stream_payload_exact":
            stream_payload.get("canonical_payload_sha256")
            == V1_STREAM_PAYLOAD_SHA256
            and verify_payload_hash(stream_payload),
        "schema_exact": schema_sha256() == SCHEMA_SHA256,
    }
    return {
        "rows": rows,
        "v2_preflight_payload_sha256": V2_PREFLIGHT_PAYLOAD_SHA256,
        "v1_record_payload_sha256": V1_RECORD_PAYLOAD_SHA256,
        "v1_stream_payload_sha256": V1_STREAM_PAYLOAD_SHA256,
        "checks": checks,
        "passes": all(checks.values()),
    }


def validate_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordinary_source = [
        dict(row)
        for row in records
        if row["partition"] != "transfer_diagnostic"
    ]
    transfer_source = [
        dict(row)
        for row in records
        if row["partition"] == "transfer_diagnostic"
    ]
    ordinary_validated, ordinary_audit = g3v1.validate_ordinary_records(
        ordinary_source
    )
    transfer_result = json_object(g3v1.TRANSFER_RESULT_PATH)
    transfer_sources = g3v1._transfer_sources(transfer_result)
    transfer_validated, transfer_audit = g3v1.validate_transfer_records(
        transfer_sources
    )
    validated = ordinary_validated + transfer_validated
    by_record = {str(row["record_id"]): row for row in records}
    validated_by_record = {
        str(row["record_id"]): row for row in validated
    }
    feature_failures = []
    action_failures = []
    for record_id, record in by_record.items():
        check = validated_by_record.get(record_id)
        if check is None:
            action_failures.append(
                {"record_id": record_id, "reason": "not_validated"}
            )
            continue
        if (
            list(check["legal_actions"]) != list(record["legal_actions"])
            or list(check["legal_action_ids"])
            != list(record["legal_action_ids"])
        ):
            action_failures.append(
                {"record_id": record_id, "reason": "legal_actions_changed"}
            )
        try:
            _rows, feature_sha = feature_rows_for_record(record)
            if feature_sha != str(record["feature_rows_sha256"]):
                feature_failures.append(
                    {
                        "record_id": record_id,
                        "expected": record["feature_rows_sha256"],
                        "actual": feature_sha,
                    }
                )
        except Exception as error:
            feature_failures.append(
                {
                    "record_id": record_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    roots = [str(row["root_cluster"]) for row in records]
    checks = {
        "record_count_exact": len(records) == EXPECTED_RECORDS,
        "ordinary_count_exact":
            len(ordinary_source) == EXPECTED_ORDINARY_RECORDS,
        "transfer_count_exact":
            len(transfer_source) == EXPECTED_TRANSFER_RECORDS,
        "ordinary_sources_exact": ordinary_audit["passes"],
        "transfer_sources_exact": transfer_audit["passes"],
        "all_records_validated": len(validated_by_record) == len(records),
        "legal_actions_exact": not action_failures,
        "feature_hashes_exact": not feature_failures,
        "ordinary_transfer_root_overlap_zero": not (
            {
                str(row["root_cluster"]) for row in ordinary_source
            }
            & {str(row["root_cluster"]) for row in transfer_source}
        ),
        "expected_root_count":
            len(set(roots)) == 384,
    }
    return {
        "ordinary": ordinary_audit,
        "transfer": transfer_audit,
        "feature_failures": feature_failures,
        "action_failures": action_failures,
        "record_counts": dict(
            sorted(Counter(str(row["partition"]) for row in records).items())
        ),
        "root_count": len(set(roots)),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _is_under(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def historical_collision_audit(
    tasks: Sequence[Mapping[str, Any]],
    *,
    excluded_exact_directories: Sequence[Path],
) -> dict[str, Any]:
    found: dict[str, set[int]] = defaultdict(set)
    sources = []
    excluded_sources = []
    for path in sorted(RUNS_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        values = g1r._scan_history_file(path)
        if not values:
            continue
        row = {
            "path": str(path),
            "sha256": sha256_path(path),
            "byte_size": path.stat().st_size,
            "counts": {
                key: len(items) for key, items in sorted(values.items())
            },
        }
        if _is_under(path, excluded_exact_directories):
            excluded_sources.append(row)
            continue
        sources.append(row)
        for key, items in values.items():
            found[key].update(items)

    collisions: dict[str, list[int]] = {}
    for key in (
        "logical_seed",
        "deck_stream_id",
        "slot_stream_id",
        "policy_stream_id",
    ):
        prior_values = set(found.get(key, set()))
        if key == "logical_seed":
            for alias in (
                "seed",
                "root_seed",
                "source_seed",
                "fresh_root_seed",
            ):
                prior_values.update(found.get(alias, set()))
        requested = {int(task[key]) for task in tasks}
        collisions[key] = sorted(requested.intersection(prior_values))
    checks = {
        "excluded_directories_distinct":
            len({path.resolve() for path in excluded_exact_directories})
            == len(excluded_exact_directories),
        "historical_collisions_zero": not any(collisions.values()),
    }
    return {
        "scan_root": str(RUNS_ROOT),
        "excluded_exact_directories": [
            str(path.resolve()) for path in excluded_exact_directories
        ],
        "matched_source_count": len(sources),
        "matched_sources_sha256": canonical_sha256(sources),
        "excluded_source_count": len(excluded_sources),
        "excluded_sources_sha256": canonical_sha256(excluded_sources),
        "value_counts": {
            key: len(values) for key, values in sorted(found.items())
        },
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
    }


def test_evidence_audit() -> dict[str, Any]:
    if not TEST_EVIDENCE_PATH.is_file():
        return {
            "path": str(TEST_EVIDENCE_PATH),
            "passes": False,
            "reason": "missing",
        }
    evidence = json_object(TEST_EVIDENCE_PATH)
    rows = []
    for row in evidence.get("bound_files", []):
        path = Path(str(row["path"]))
        actual = sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expected_sha256": row["sha256"],
                "actual_sha256": actual,
                "matches": actual == row["sha256"],
            }
        )
    checks = {
        "version_exact":
            evidence.get("version") == "g3_e0_test_evidence_v1",
        "tests_pass": evidence.get("passes") is True,
        "bound_files_present": bool(rows),
        "bound_files_exact": bool(rows) and all(row["matches"] for row in rows),
        "scientific_labels_generated_zero":
            evidence.get("scientific_labels_generated") == 0,
        "scientific_models_fit_zero":
            evidence.get("scientific_models_fit") == 0,
    }
    return {
        "path": str(TEST_EVIDENCE_PATH),
        "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "rows": rows,
        "checks": checks,
        "passes": all(checks.values()),
    }


def disk_audit(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    return {
        "free_bytes": usage.free,
        "free_gib": free_gib,
        "minimum_free_gib": MIN_FREE_GIB,
        "target_free_gib": TARGET_FREE_GIB,
        "above_hard_minimum": free_gib >= MIN_FREE_GIB,
        "above_target": free_gib >= TARGET_FREE_GIB,
    }


def process_and_service_audit() -> dict[str, Any]:
    services = g1r.service_health()
    heavy = _heavy_process_audit()
    current_nice = os.nice(0)
    return {
        "services": services,
        "heavy_process": heavy,
        "current_nice": current_nice,
        "passes":
            services["passes"]
            and heavy["passes"]
            and current_nice >= MIN_NICE,
    }


def build_preflight_payload(
    *,
    out_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if schema_sha256() != SCHEMA_SHA256:
        raise ValueError("G2 feature schema changed")
    upstream = upstream_input_audit()
    v2_preflight, record_manifest, stream_manifest, reuse = (
        g3v2.load_v1_manifests()
    )
    if sha256_path(V2_PREFLIGHT_PATH) != V2_PREFLIGHT_FILE_SHA256:
        raise ValueError("G3-v2 READY seal changed")
    if v2_preflight.get("decision") != "KILL_G3_PREFLIGHT_INTEGRITY":
        # load_v1_manifests returns the spent v1 preflight by design.
        raise ValueError("G3-v1 spent decision changed")
    records = [dict(row) for row in record_manifest["records"]]
    all_stream_rows = [dict(row) for row in stream_manifest["rows"]]
    tasks = build_e0_tasks(records, all_stream_rows)
    record_audit = validate_records(records)
    coupling = task_coupling_audit(tasks)
    collision = historical_collision_audit(
        tasks,
        excluded_exact_directories=(V1_OUTPUT_DIR, V2_OUTPUT_DIR),
    )
    by_partition = dict(
        sorted(Counter(str(task["partition"]) for task in tasks).items())
    )
    task_checks = {
        "task_count_exact": len(tasks) == EXPECTED_E0_PATHS,
        "partition_counts_exact": by_partition == EXPECTED_E0_BY_PARTITION,
        "replicates_exact":
            {int(task["replicate"]) for task in tasks}
            == set(E0_REPLICATES),
        "stream_coupling_exact": coupling["passes"],
        "collision_audit_passes": collision["passes"],
        "collision_exclusions_exact":
            set(collision["excluded_exact_directories"])
            == {
                str(V1_OUTPUT_DIR.resolve()),
                str(V2_OUTPUT_DIR.resolve()),
            },
    }

    transfer_lock = json_object(G2_TRANSFER_LOCK_PATH)
    incumbent = g3v1._verify_incumbent_artifacts(transfer_lock)
    v2_seal = json_object(V2_PREFLIGHT_PATH)
    stage_cost = v2_seal["staged_cost_decomposition"]["stage_costs"]["E0"]
    cost_checks = {
        "path_count_exact": int(stage_cost["paths"]) == EXPECTED_E0_PATHS,
        "runtime_at_most_18h":
            float(stage_cost["projected_runtime_hours"])
            <= MAX_PROJECTED_RUNTIME_HOURS,
        "storage_below_4gib":
            int(stage_cost["projected_incremental_bytes"])
            < MAX_INCREMENTAL_BYTES,
    }
    disk = disk_audit(WORKSPACE_ROOT)
    operations = process_and_service_audit()
    tests = test_evidence_audit()

    integrity_checks = {
        "upstream_inputs_exact": upstream["passes"],
        "v1_manifest_reuse_exact": reuse["passes"],
        "record_reconstruction_exact": record_audit["passes"],
        "task_contract_exact": all(task_checks.values()),
        "incumbent_payloads_exact": incumbent["passes"],
        "feature_schema_exact": schema_sha256() == SCHEMA_SHA256,
        "focused_and_regression_tests_pass": tests["passes"],
    }
    readiness_checks = {
        "cost_contract_passes": all(cost_checks.values()),
        "disk_above_100gib": disk["above_hard_minimum"],
        "disk_above_120gib_target": disk["above_target"],
        "services_process_and_nice_pass": operations["passes"],
        "output_dir_fresh": not out_dir.exists(),
        "no_existing_e0_work": not any(
            path.exists()
            for path in (
                out_dir / "ordinary_labels.sqlite3",
                out_dir / "transfer_labels.sqlite3",
                out_dir / "checkpoint",
                out_dir / "G3_E0_EXECUTION_OPENED.json",
            )
        ),
    }
    if not all(integrity_checks.values()):
        decision = "KILL_G3_E0_PREFLIGHT_INTEGRITY"
    elif all(readiness_checks.values()):
        decision = "READY_G3_E0_LABEL_FIT_EXECUTION"
    else:
        decision = "HOLD_G3_E0_COST_OR_COVERAGE"

    compact_tasks = []
    for task in tasks:
        compact = {
            key: value
            for key, value in task.items()
            if key != "record"
        }
        compact_tasks.append(compact)
    e0_records = payload_with_hash(
        {
            "version": "g3_e0_records_v1",
            "source_record_manifest_file_sha256": V1_RECORD_FILE_SHA256,
            "source_record_manifest_payload_sha256":
                V1_RECORD_PAYLOAD_SHA256,
            "records": records,
            "records_sha256": canonical_sha256(records),
            "label_values_opened": False,
            "models_fit": 0,
        }
    )
    e0_streams = payload_with_hash(
        {
            "version": "g3_e0_streams_v1",
            "source_stream_manifest_file_sha256": V1_STREAM_FILE_SHA256,
            "source_stream_manifest_payload_sha256":
                V1_STREAM_PAYLOAD_SHA256,
            "replicates": list(E0_REPLICATES),
            "rows": compact_tasks,
            "rows_sha256": canonical_sha256(compact_tasks),
            "streams_consumed": 0,
        }
    )
    task_records = {str(row["record_id"]): row for row in records}
    full_tasks = [
        {**task, "record": task_records[str(task["record_id"])]}
        for task in compact_tasks
    ]
    e0_tasks = payload_with_hash(
        {
            "version": "g3_e0_tasks_v1",
            "records_payload_sha256":
                e0_records["canonical_payload_sha256"],
            "streams_payload_sha256":
                e0_streams["canonical_payload_sha256"],
            "tasks": full_tasks,
            "tasks_sha256": canonical_sha256(full_tasks),
            "label_values_opened": False,
        }
    )

    lock = {
        "version": VERSION,
        "runner_version": RUNNER_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "terminal_status": "HOLD_G3_AFTER_E0_PREFLIGHT_SEAL",
        "out_dir_resolved": str(out_dir.resolve()),
        "future_command": FUTURE_COMMAND,
        "jobs": 1,
        "nice": MIN_NICE,
        "charter": {
            "path": str(CHARTER_PATH),
            "sha256": CHARTER_SHA256,
        },
        "implementation": {
            "runner_path": str(IMPLEMENTATION_PATH),
            "runner_sha256": sha256_path(IMPLEMENTATION_PATH),
            "preflight_path": str(PREFLIGHT_PATH),
            "preflight_sha256": sha256_path(PREFLIGHT_PATH),
            "test_path": str(TEST_PATH),
            "test_sha256": sha256_path(TEST_PATH),
            "test_evidence": tests,
        },
        "upstream_input_audit": upstream,
        "record_audit": record_audit,
        "task_audit": {
            "tasks": len(tasks),
            "by_partition": by_partition,
            "coupling": coupling,
            "collision": collision,
            "checks": task_checks,
            "passes": all(task_checks.values()),
        },
        "incumbent_artifact_audit": incumbent,
        "incumbent_policy_spec": incumbent["incumbent_policy_spec"],
        "cost_projection": stage_cost,
        "cost_checks": cost_checks,
        "disk": disk,
        "operations": operations,
        "integrity_checks": integrity_checks,
        "readiness_checks": readiness_checks,
        "ordinary_path_count":
            EXPECTED_E0_BY_PARTITION["train"]
            + EXPECTED_E0_BY_PARTITION["development"],
        "transfer_path_count":
            EXPECTED_E0_BY_PARTITION["transfer_diagnostic"],
        "record_manifest_name": "E0_RECORD_MANIFEST.json",
        "stream_manifest_name": "E0_STREAM_MANIFEST.json",
        "task_manifest_name": "E0_TASK_MANIFEST.json",
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
            decision == "READY_G3_E0_LABEL_FIT_EXECUTION",
        "e1_authorized": False,
        "policy_evaluation_authorized": False,
        "promotion_authorized": False,
        "dashboard_eligible": False,
    }
    return lock, e0_records, e0_streams, e0_tasks


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    staging = out_dir.with_name(out_dir.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        lock, records, streams, tasks = build_preflight_payload(
            out_dir=out_dir
        )
        write_immutable_json(staging / "E0_RECORD_MANIFEST.json", records)
        write_immutable_json(staging / "E0_STREAM_MANIFEST.json", streams)
        write_immutable_json(staging / "E0_TASK_MANIFEST.json", tasks)
        lock["record_manifest_file_sha256"] = sha256_path(
            staging / "E0_RECORD_MANIFEST.json"
        )
        lock["stream_manifest_file_sha256"] = sha256_path(
            staging / "E0_STREAM_MANIFEST.json"
        )
        lock["task_manifest_file_sha256"] = sha256_path(
            staging / "E0_TASK_MANIFEST.json"
        )
        lock = payload_with_hash(lock)
        write_immutable_json(staging / "preflight_lock.json", lock)
        staging.replace(out_dir)
        return lock
    except Exception as error:
        failure = payload_with_hash(
            {
                "version": VERSION,
                "decision": "KILL_G3_E0_PREFLIGHT_INTEGRITY",
                "error": f"{type(error).__name__}: {error}",
                "zero_forbidden_work": {
                    "new_games": 0,
                    "streams_consumed": 0,
                    "label_paths_generated": 0,
                    "label_values_opened": False,
                    "scientific_models_fit": 0,
                    "transfer_outcomes_opened": 0,
                    "policy_outcomes": 0,
                    "score_inspection": False,
                },
            }
        )
        write_immutable_json(staging / "PREFLIGHT_FAILURE.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if os.nice(0) < MIN_NICE:
        os.nice(MIN_NICE - os.nice(0))
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

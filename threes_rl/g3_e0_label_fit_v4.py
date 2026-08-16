"""G3 E0 v4 open/resume orchestration with classified collision sources.

The ``open`` command seals the execution marker and exits without touching a
label stream. The ``execute`` command requires that exact marker and runs the
unchanged G3 E0 scientific computations through an explicit resumable state
machine.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from threes_rl import g1r_acquire as g1r
from threes_rl import g3_e0_label_fit as core
from threes_rl import g3_e0_preflight as preflight_core
from threes_rl import g3_scale_transfer_bootstrap_preflight as g3v1
from threes_rl.s3_power_preflight import sha256_path


VERSION = "g3_e0_label_fit_v4"
AMENDMENT_PATH = Path(
    "threes_rl/"
    "G3_E0_LABEL_FIT_EXECUTION_CHARTER_AMENDMENT_V4_COLLISION_AUDIT.md"
)
AMENDMENT_SHA256 = (
    "cbae99106e2d559b628140553012d7144ef6bc729012cd0a87c30d48f3a6287c"
)
V3_AMENDMENT_PATH = Path(
    "threes_rl/"
    "G3_E0_LABEL_FIT_EXECUTION_CHARTER_AMENDMENT_V3_ORCHESTRATION.md"
)
V3_AMENDMENT_SHA256 = (
    "d201e4a187e77a643d555499f76d8dca6c76eab584c1f3d5efaced8de98d38ab"
)
V2_AMENDMENT_PATH = Path(
    "threes_rl/G3_E0_LABEL_FIT_EXECUTION_CHARTER_AMENDMENT_V2_INTEGRITY.md"
)
V2_AMENDMENT_SHA256 = (
    "1b0594d5c0cb55b7c5e11d24c64fa0f82d33952f6e4d4eab8eb5bdf429815fe7"
)
SCIENTIFIC_RUNNER_PATH = Path("threes_rl/g3_e0_label_fit.py")
SCIENTIFIC_RUNNER_SHA256 = (
    "19d74a319459d75619f515fd9cdea03a126e1270046fb8e12ae367d43b2cc8b5"
)
RUNNER_PATH = Path("threes_rl/g3_e0_label_fit_v4.py")
OUTPUT_DIR = Path("threes_rl/runs/forensics/g3_e0_label_fit_v4")
OPEN_MARKER_NAME = core.OPEN_MARKER_NAME
TERMINAL_RESULT_NAME = core.TERMINAL_RESULT_NAME
COLLISION_MANIFEST_NAME = "E0_COLLISION_SOURCE_MANIFEST.json"
PREFLIGHT_DECISION = "READY_G3_E0_V4_EXECUTION"
OPEN_COMMAND = (
    "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
    ".venv/bin/python -m threes_rl.g3_e0_label_fit_v4 open "
    "--out-dir threes_rl/runs/forensics/g3_e0_label_fit_v4 "
    "--preflight-lock "
    "threes_rl/runs/forensics/g3_e0_label_fit_v4/preflight_lock.json "
    "--jobs 1'"
)
EXECUTE_COMMAND = (
    "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
    ".venv/bin/python -m threes_rl.g3_e0_label_fit_v4 execute "
    "--out-dir threes_rl/runs/forensics/g3_e0_label_fit_v4 "
    "--preflight-lock "
    "threes_rl/runs/forensics/g3_e0_label_fit_v4/preflight_lock.json "
    "--jobs 1'"
)
ALLOWED_BASE_FILES = {
    "E0_RECORD_MANIFEST.json",
    "E0_STREAM_MANIFEST.json",
    "E0_TASK_MANIFEST.json",
    COLLISION_MANIFEST_NAME,
    "preflight_lock.json",
}
WORK_ARTIFACTS = {
    OPEN_MARKER_NAME,
    TERMINAL_RESULT_NAME,
    core.ORDINARY_DB_NAME,
    core.TRANSFER_DB_NAME,
    core.MODEL_DIR_NAME,
    core.CHECKPOINT_SEAL_NAME,
    core.PREDICTION_SEAL_NAME,
}
LIVE_SOURCE_PATHS = (
    Path("threes_rl/runs/dashboard/dashboard.json"),
    Path("threes_rl/runs/dashboard/score_trends.json"),
)
STREAM_KEYS = (
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
)
SEED_ALIASES = (
    "seed",
    "root_seed",
    "source_seed",
    "fresh_root_seed",
)


def _load_lock(
    preflight_lock: Path,
    *,
    out_dir: Path,
    jobs: int,
) -> dict[str, Any]:
    if jobs != 1:
        raise ValueError("Frozen G3 E0 v4 execution requires jobs=1")
    lock = core.json_object(preflight_lock)
    if not core.verify_payload_hash(lock):
        raise ValueError("Preflight lock payload hash mismatch")
    if lock.get("decision") != PREFLIGHT_DECISION:
        raise PermissionError("V4 preflight did not authorize execution")
    if Path(str(lock["out_dir_resolved"])).resolve() != out_dir.resolve():
        raise ValueError("Preflight output directory mismatch")
    if int(lock.get("jobs", -1)) != jobs:
        raise ValueError("Preflight jobs mismatch")
    if str(lock.get("open_command")) != OPEN_COMMAND:
        raise ValueError("Preflight open command mismatch")
    if str(lock.get("execution_command")) != EXECUTE_COMMAND:
        raise ValueError("Preflight execution command mismatch")
    return lock


def _verify_bound_files(lock: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for row in lock.get("bound_files", []):
        path = Path(str(row["path"]))
        expected = str(row["sha256"])
        actual = sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    checks = {
        "rows_present": bool(rows),
        "all_files_exact": bool(rows) and all(row["matches"] for row in rows),
        "scientific_runner_exact":
            sha256_path(SCIENTIFIC_RUNNER_PATH)
            == SCIENTIFIC_RUNNER_SHA256,
        "orchestration_runner_exact":
            sha256_path(RUNNER_PATH)
            == lock.get("orchestration_runner_sha256"),
        "base_charter_exact":
            sha256_path(core.CHARTER_PATH) == core.CHARTER_SHA256,
        "v2_amendment_exact":
            sha256_path(V2_AMENDMENT_PATH) == V2_AMENDMENT_SHA256,
        "v3_amendment_exact":
            sha256_path(V3_AMENDMENT_PATH) == V3_AMENDMENT_SHA256,
        "v4_amendment_exact":
            sha256_path(AMENDMENT_PATH) == AMENDMENT_SHA256,
    }
    return {
        "rows": rows,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _manifest_audit(
    lock: Mapping[str, Any],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    bindings = {}
    checks = {}
    for role, name_key, file_key, payload_key in (
        (
            "record",
            "record_manifest_name",
            "record_manifest_file_sha256",
            "record_manifest_payload_sha256",
        ),
        (
            "task",
            "task_manifest_name",
            "task_manifest_file_sha256",
            "task_manifest_payload_sha256",
        ),
        (
            "stream",
            "stream_manifest_name",
            "stream_manifest_file_sha256",
            "stream_manifest_payload_sha256",
        ),
    ):
        path = out_dir / str(lock[name_key])
        payload = core.json_object(path)
        row = {
            "path": str(path),
            "file_sha256": sha256_path(path),
            "expected_file_sha256": str(lock[file_key]),
            "payload_sha256": payload.get("canonical_payload_sha256"),
            "expected_payload_sha256": str(lock[payload_key]),
            "payload_valid": core.verify_payload_hash(payload),
        }
        bindings[role] = row
        checks[f"{role}_file_exact"] = (
            row["file_sha256"] == row["expected_file_sha256"]
        )
        checks[f"{role}_payload_exact"] = (
            row["payload_valid"]
            and row["payload_sha256"] == row["expected_payload_sha256"]
        )
    return {
        "bindings": bindings,
        "checks": checks,
        "passes": all(checks.values()),
    }


def requested_stream_contract(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        {
            "task_key": str(task["task_key"]),
            **{key: int(task[key]) for key in STREAM_KEYS},
        }
        for task in tasks
    ]
    unique = {
        key: sorted({int(row[key]) for row in rows})
        for key in STREAM_KEYS
    }
    return {
        "task_rows": len(rows),
        "task_rows_sha256": core.canonical_sha256(rows),
        "unique": {
            key: {
                "count": len(values),
                "sha256": core.canonical_sha256(values),
            }
            for key, values in unique.items()
        },
        "all_values_sha256": core.canonical_sha256(unique),
    }


def _is_under_exact_directories(
    path: Path,
    directories: Sequence[Path],
) -> bool:
    resolved = path.resolve()
    for directory in directories:
        try:
            resolved.relative_to(directory.resolve())
            return True
        except ValueError:
            continue
    return False


def _normalized_path_text(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(
            resolved.relative_to(preflight_core.WORKSPACE_ROOT.resolve())
        )
    except ValueError:
        return str(resolved)


def _source_row(
    path: Path,
    values: Mapping[str, set[int]],
    *,
    classification: str,
    byte_stability_required: bool,
) -> dict[str, Any]:
    return {
        "path": _normalized_path_text(path),
        "classification": classification,
        "byte_stability_required": byte_stability_required,
        "sha256": sha256_path(path),
        "byte_size": path.stat().st_size,
        "counts": {
            key: len(items) for key, items in sorted(values.items())
        },
    }


def scan_classified_collision_sources(
    *,
    excluded_directories: Sequence[Path],
    live_source_paths: Sequence[Path] = LIVE_SOURCE_PATHS,
    scan_root: Path = preflight_core.RUNS_ROOT,
) -> dict[str, Any]:
    if len({path.resolve() for path in excluded_directories}) != len(
        excluded_directories
    ):
        raise ValueError("Collision exclusion directories must be distinct")
    expected_live = {
        path.resolve(): path for path in live_source_paths
    }
    immutable_rows = []
    live_rows = []
    internal_rows = []
    symlink_rows = []
    external_values: dict[str, set[int]] = defaultdict(set)
    for path in sorted(scan_root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in {".json", ".jsonl", ".csv"}
        ):
            continue
        is_exact_live = path.resolve() in expected_live
        values = g1r._scan_history_file(path)
        if path.is_symlink():
            if is_exact_live or values:
                symlink_rows.append(_normalized_path_text(path))
            continue
        if is_exact_live:
            row = _source_row(
                path,
                values,
                classification="live_generated_dashboard",
                byte_stability_required=False,
            )
            live_rows.append(row)
            for key, items in values.items():
                external_values[key].update(items)
        elif not values:
            continue
        elif _is_under_exact_directories(path, excluded_directories):
            internal_rows.append(
                _source_row(
                    path,
                    values,
                    classification="inherited_internal_reservation",
                    byte_stability_required=False,
                )
            )
        else:
            row = _source_row(
                path,
                values,
                classification="immutable_external",
                byte_stability_required=True,
            )
            immutable_rows.append(row)
            for key, items in values.items():
                external_values[key].update(items)
    return {
        "immutable_sources": immutable_rows,
        "live_sources": live_rows,
        "internal_sources": internal_rows,
        "symlink_sources": symlink_rows,
        "external_values": {
            key: sorted(values)
            for key, values in sorted(external_values.items())
        },
        "excluded_directories": [
            str(path.resolve()) for path in excluded_directories
        ],
    }


def collision_intersections(
    tasks: Sequence[Mapping[str, Any]],
    external_values: Mapping[str, Sequence[int]],
) -> dict[str, list[int]]:
    collisions = {}
    for key in STREAM_KEYS:
        prior = set(int(value) for value in external_values.get(key, []))
        if key == "logical_seed":
            for alias in SEED_ALIASES:
                prior.update(
                    int(value)
                    for value in external_values.get(alias, [])
                )
        requested = {int(task[key]) for task in tasks}
        collisions[key] = sorted(requested.intersection(prior))
    return collisions


def build_collision_source_manifest(
    tasks: Sequence[Mapping[str, Any]],
    *,
    excluded_directories: Sequence[Path],
    live_source_paths: Sequence[Path] = LIVE_SOURCE_PATHS,
    scan_root: Path = preflight_core.RUNS_ROOT,
) -> dict[str, Any]:
    scan = scan_classified_collision_sources(
        excluded_directories=excluded_directories,
        live_source_paths=live_source_paths,
        scan_root=scan_root,
    )
    collisions = collision_intersections(tasks, scan["external_values"])
    live_contract = [
        {
            "path": _normalized_path_text(path),
            "classification": "live_generated_dashboard",
            "reason":
                "Required watcher-generated dashboard summary; values are "
                "rescanned for collisions but bytes are not stability-locked.",
        }
        for path in live_source_paths
    ]
    internal_contract = [
        {
            "path": str(path.resolve()),
            "classification": "inherited_internal_reservation",
            "reason":
                "Exact inherited/self namespace containing immutable copies "
                "of the same unconsumed requested reservation.",
        }
        for path in excluded_directories
    ]
    observed_live_paths = sorted(
        str(row["path"]) for row in scan["live_sources"]
    )
    expected_live_paths = sorted(
        _normalized_path_text(path)
        for path in live_source_paths
    )
    checks = {
        "requested_task_count_exact": len(tasks) == 5_072,
        "live_paths_exact": observed_live_paths == expected_live_paths,
        "live_count_exact": len(scan["live_sources"]) == 2,
        "no_symlink_sources": not scan["symlink_sources"],
        "zero_collisions": not any(collisions.values()),
    }
    return core.payload_with_hash(
        {
            "version": "g3_e0_v4_collision_sources_v1",
            "requested_stream_contract":
                requested_stream_contract(tasks),
            "immutable_sources": scan["immutable_sources"],
            "immutable_source_count": len(scan["immutable_sources"]),
            "immutable_inventory_sha256":
                core.canonical_sha256(scan["immutable_sources"]),
            "live_source_contract": live_contract,
            "live_sources_observed": scan["live_sources"],
            "live_source_count": len(scan["live_sources"]),
            "internal_namespace_contract": internal_contract,
            "internal_source_count": len(scan["internal_sources"]),
            "internal_sources_sha256":
                core.canonical_sha256(scan["internal_sources"]),
            "collisions": collisions,
            "checks": checks,
            "passes": all(checks.values()),
            "outcomes_opened": False,
        }
    )


def revalidate_collision_source_manifest(
    manifest: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    *,
    excluded_directories: Sequence[Path],
    live_source_paths: Sequence[Path] = LIVE_SOURCE_PATHS,
    scan_root: Path = preflight_core.RUNS_ROOT,
) -> dict[str, Any]:
    if not core.verify_payload_hash(manifest):
        raise ValueError("Collision source manifest payload hash mismatch")
    scan = scan_classified_collision_sources(
        excluded_directories=excluded_directories,
        live_source_paths=live_source_paths,
        scan_root=scan_root,
    )
    collisions = collision_intersections(tasks, scan["external_values"])
    requested = requested_stream_contract(tasks)
    expected_live_paths = sorted(
        str(row["path"]) for row in manifest["live_source_contract"]
    )
    actual_live_paths = sorted(
        str(row["path"]) for row in scan["live_sources"]
    )
    expected_internal = {
        str(row["path"]) for row in manifest["internal_namespace_contract"]
    }
    actual_internal = {
        str(path.resolve()) for path in excluded_directories
    }
    current_immutable_sha = core.canonical_sha256(
        scan["immutable_sources"]
    )
    checks = {
        "requested_stream_contract_exact":
            requested == manifest["requested_stream_contract"],
        "immutable_source_count_exact":
            len(scan["immutable_sources"])
            == int(manifest["immutable_source_count"]),
        "immutable_inventory_exact":
            current_immutable_sha
            == str(manifest["immutable_inventory_sha256"])
            and scan["immutable_sources"]
            == manifest["immutable_sources"],
        "live_paths_exact":
            actual_live_paths == expected_live_paths,
        "live_count_exact":
            len(scan["live_sources"])
            == int(manifest["live_source_count"])
            == 2,
        "internal_namespaces_exact":
            actual_internal == expected_internal,
        "no_symlink_sources": not scan["symlink_sources"],
        "zero_collisions": not any(collisions.values()),
    }
    return {
        "requested_stream_contract": requested,
        "immutable_source_count": len(scan["immutable_sources"]),
        "immutable_inventory_sha256": current_immutable_sha,
        "live_sources": scan["live_sources"],
        "live_source_count": len(scan["live_sources"]),
        "internal_source_count": len(scan["internal_sources"]),
        "internal_sources_sha256":
            core.canonical_sha256(scan["internal_sources"]),
        "symlink_sources": scan["symlink_sources"],
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _stream_collision_audit(
    lock: Mapping[str, Any],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    task_payload = core.json_object(
        out_dir / str(lock["task_manifest_name"])
    )
    tasks = [dict(task) for task in task_payload["tasks"]]
    manifest_path = out_dir / str(lock["collision_manifest_name"])
    if sha256_path(manifest_path) != str(
        lock["collision_manifest_file_sha256"]
    ):
        raise ValueError("Collision source manifest file hash mismatch")
    manifest = core.json_object(manifest_path)
    if manifest.get("canonical_payload_sha256") != str(
        lock["collision_manifest_payload_sha256"]
    ):
        raise ValueError("Collision source manifest payload identity changed")
    excluded = tuple(
        Path(str(path))
        for path in lock["stream_collision_excluded_directories"]
    )
    return revalidate_collision_source_manifest(
        manifest,
        tasks,
        excluded_directories=excluded,
        live_source_paths=LIVE_SOURCE_PATHS,
    )


def _incumbent_audit(lock: Mapping[str, Any]) -> dict[str, Any]:
    transfer_lock = core.json_object(
        Path(str(lock["transfer_preflight_lock_path"]))
    )
    audit = g3v1._verify_incumbent_artifacts(transfer_lock)
    checks = {
        "incumbent_payloads_exact": audit["passes"],
        "incumbent_audit_sha256_exact":
            core.canonical_sha256(audit)
            == str(lock["incumbent_artifact_audit_sha256"]),
        "incumbent_policy_spec_exact":
            audit["incumbent_policy_spec"]
            == str(lock["incumbent_policy_spec"]),
    }
    return {
        "audit": audit,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _operational_audit(out_dir: Path) -> dict[str, Any]:
    disk = preflight_core.disk_audit(out_dir)
    operations = preflight_core.process_and_service_audit()
    checks = {
        "disk_above_hard_minimum": disk["above_hard_minimum"],
        "disk_above_target": disk["above_target"],
        "services_process_and_nice": operations["passes"],
    }
    return {
        "disk": disk,
        "operations": operations,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _immutable_contract(
    lock: Mapping[str, Any],
    *,
    preflight_lock: Path,
    out_dir: Path,
    jobs: int,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "out_dir_resolved": str(out_dir.resolve()),
        "open_command": OPEN_COMMAND,
        "execution_command": EXECUTE_COMMAND,
        "base_charter_sha256": core.CHARTER_SHA256,
        "v2_amendment_sha256": V2_AMENDMENT_SHA256,
        "v3_orchestration_amendment_sha256": V3_AMENDMENT_SHA256,
        "v4_collision_amendment_sha256": AMENDMENT_SHA256,
        "scientific_runner_sha256": SCIENTIFIC_RUNNER_SHA256,
        "orchestration_runner_sha256": sha256_path(RUNNER_PATH),
        "focused_test_sha256": lock["focused_test_sha256"],
        "test_evidence_file_sha256": lock["test_evidence_file_sha256"],
        "preflight_lock_file_sha256": sha256_path(preflight_lock),
        "preflight_lock_payload_sha256":
            lock["canonical_payload_sha256"],
        "record_manifest_file_sha256":
            lock["record_manifest_file_sha256"],
        "record_manifest_payload_sha256":
            lock["record_manifest_payload_sha256"],
        "task_manifest_file_sha256":
            lock["task_manifest_file_sha256"],
        "task_manifest_payload_sha256":
            lock["task_manifest_payload_sha256"],
        "stream_manifest_file_sha256":
            lock["stream_manifest_file_sha256"],
        "stream_manifest_payload_sha256":
            lock["stream_manifest_payload_sha256"],
        "incumbent_policy_spec": lock["incumbent_policy_spec"],
        "incumbent_artifact_audit_sha256":
            lock["incumbent_artifact_audit_sha256"],
        "ordinary_records": int(lock["ordinary_record_count"]),
        "transfer_roots": int(lock["transfer_record_count"]),
        "total_paths": int(lock["total_path_count"]),
        "train_paths": int(lock["path_counts"]["train"]),
        "development_paths": int(lock["path_counts"]["development"]),
        "transfer_paths": int(lock["path_counts"]["transfer_diagnostic"]),
        "replicates": [int(value) for value in lock["replicates"]],
        "jobs": jobs,
        "minimum_nice": int(lock["nice"]),
        "maximum_active_hours": float(lock["maximum_active_hours"]),
        "maximum_output_bytes": int(lock["maximum_output_bytes"]),
        "minimum_free_gib": float(lock["minimum_free_gib"]),
        "target_free_gib": float(lock["target_free_gib"]),
        "required_services": list(lock["required_services"]),
        "collision_manifest_name": lock["collision_manifest_name"],
        "collision_manifest_file_sha256":
            lock["collision_manifest_file_sha256"],
        "collision_manifest_payload_sha256":
            lock["collision_manifest_payload_sha256"],
        "requested_stream_contract_sha256":
            lock["requested_stream_contract_sha256"],
        "immutable_inventory_sha256":
            lock["immutable_inventory_sha256"],
        "live_source_paths": list(lock["live_source_paths"]),
        "live_source_count": int(lock["live_source_count"]),
        "stream_collision_excluded_directories": list(
            lock["stream_collision_excluded_directories"]
        ),
    }


def _assert_zero_work(out_dir: Path) -> None:
    present = sorted(
        path.name for path in out_dir.iterdir()
        if path.name in WORK_ARTIFACTS
    )
    unexpected = sorted(
        path.name for path in out_dir.iterdir()
        if path.name not in ALLOWED_BASE_FILES
    )
    if present or unexpected:
        raise FileExistsError(
            f"V4 output is not zero-work: present={present}, "
            f"unexpected={unexpected}"
        )


def open_execution(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
) -> dict[str, Any]:
    """Seal the immutable marker and exit before any scientific work."""

    lock = _load_lock(preflight_lock, out_dir=out_dir, jobs=jobs)
    _assert_zero_work(out_dir)
    files = _verify_bound_files(lock)
    manifests = _manifest_audit(lock, out_dir=out_dir)
    incumbent = _incumbent_audit(lock)
    collisions = _stream_collision_audit(lock, out_dir=out_dir)
    health = _operational_audit(out_dir)
    if not (
        files["passes"]
        and manifests["passes"]
        and incumbent["passes"]
        and collisions["passes"]
        and health["passes"]
    ):
        raise RuntimeError("V4 open-only revalidation failed")
    contract = _immutable_contract(
        lock,
        preflight_lock=preflight_lock,
        out_dir=out_dir,
        jobs=jobs,
    )
    payload = core.payload_with_hash(
        {
            "version": VERSION,
            "decision": "OPENED_G3_E0_V4_EXECUTION",
            "opened_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "contract": contract,
            "open_health": health,
            "bound_file_audit": files,
            "manifest_audit": manifests,
            "incumbent_audit": incumbent,
            "stream_collision_audit": collisions,
            "zero_work_before_open": {
                "streams_consumed": 0,
                "label_paths_generated": 0,
                "label_databases": 0,
                "models_fit": 0,
                "checkpoint_predictions": 0,
                "transfer_outcomes_opened": 0,
                "policy_outcomes": 0,
                "dashboard_changed": False,
            },
            "e0_non_promotable": True,
        }
    )
    core.write_immutable_json(out_dir / OPEN_MARKER_NAME, payload)
    return payload


def validate_execution_marker(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
    revalidate_operations: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the one marker explicitly before any scientific access."""

    lock = _load_lock(preflight_lock, out_dir=out_dir, jobs=jobs)
    marker_path = out_dir / OPEN_MARKER_NAME
    if not marker_path.is_file():
        raise FileNotFoundError("Missing G3 E0 v4 execution marker")
    if (out_dir / TERMINAL_RESULT_NAME).exists():
        raise FileExistsError("G3 E0 v4 terminal result is immutable")
    marker = core.json_object(marker_path)
    if not core.verify_payload_hash(marker):
        raise ValueError("Execution marker payload hash mismatch")
    expected = _immutable_contract(
        lock,
        preflight_lock=preflight_lock,
        out_dir=out_dir,
        jobs=jobs,
    )
    if marker.get("version") != VERSION:
        raise ValueError("Execution marker version mismatch")
    if marker.get("decision") != "OPENED_G3_E0_V4_EXECUTION":
        raise ValueError("Execution marker decision mismatch")
    if marker.get("contract") != expected:
        raise ValueError("Execution marker contract mismatch")
    files = _verify_bound_files(lock)
    manifests = _manifest_audit(lock, out_dir=out_dir)
    incumbent = _incumbent_audit(lock)
    if (
        not files["passes"]
        or not manifests["passes"]
        or not incumbent["passes"]
    ):
        raise RuntimeError("Execution marker bound files changed")
    current = {
        "bound_files": files,
        "manifests": manifests,
        "incumbent": incumbent,
    }
    if revalidate_operations:
        collisions = _stream_collision_audit(lock, out_dir=out_dir)
        health = _operational_audit(out_dir)
        if not collisions["passes"] or not health["passes"]:
            raise RuntimeError("Execution marker operational recheck failed")
        current["stream_collisions"] = collisions
        current["health"] = health
    return lock, {
        "marker": marker,
        "marker_file_sha256": sha256_path(marker_path),
        "current": current,
    }


def _load_checkpoint(
    *,
    out_dir: Path,
    preflight_lock: Path,
    marker_file_sha256: str,
) -> tuple[core.G3HazardModel, dict[str, Any]]:
    checkpoint_path = out_dir / core.CHECKPOINT_SEAL_NAME
    checkpoint = core.json_object(checkpoint_path)
    if not core.verify_payload_hash(checkpoint):
        raise ValueError("Checkpoint seal payload hash mismatch")
    expected_sources = {
        "preflight_lock": sha256_path(preflight_lock),
        "ordinary_labels": sha256_path(out_dir / core.ORDINARY_DB_NAME),
    }
    model = core.G3HazardModel.load(
        out_dir / core.MODEL_DIR_NAME,
        expected_source_hashes=expected_sources,
    )
    checks = {
        "model_meta_exact":
            sha256_path(out_dir / core.MODEL_DIR_NAME / "meta.json")
            == checkpoint["model_meta_sha256"],
        "model_arrays_exact":
            sha256_path(out_dir / core.MODEL_DIR_NAME / "arrays.npz")
            == checkpoint["model_arrays_sha256"],
        "ordinary_labels_exact":
            sha256_path(out_dir / core.ORDINARY_DB_NAME)
            == checkpoint["ordinary_labels_sha256"],
        "execution_marker_exact":
            checkpoint.get("execution_marker_file_sha256")
            == marker_file_sha256,
    }
    if not all(checks.values()):
        raise ValueError("Checkpoint seal artifact mismatch")
    return model, checkpoint


def _seal_terminal_error(
    out_dir: Path,
    *,
    stage: str,
    error: BaseException,
    marker_file_sha256: str,
) -> dict[str, Any]:
    payload = core.payload_with_hash(
        {
            "version": VERSION,
            "decision": "KILL_G3_BOOTSTRAP_PREDICTIVE",
            "stage": stage,
            "error": f"{type(error).__name__}: {error}",
            "execution_marker_file_sha256": marker_file_sha256,
            "promotable": False,
            "policy_evaluation_authorized": False,
            "dashboard_eligible": False,
        }
    )
    core.write_immutable_json(out_dir / TERMINAL_RESULT_NAME, payload)
    return payload


def _seal_terminal_decision(
    out_dir: Path,
    *,
    decision: str,
    stage: str,
    evidence: Mapping[str, Any],
    marker_file_sha256: str,
) -> dict[str, Any]:
    if decision not in {
        "READY_G3_E1_COMPLETION",
        "HOLD_G3_E0_UNDERPOWERED_TRANSFER",
        "KILL_G3_BOOTSTRAP_PREDICTIVE",
    }:
        raise ValueError(f"Unsupported E0 terminal decision: {decision}")
    payload = core.payload_with_hash(
        {
            "version": VERSION,
            "decision": decision,
            "stage": stage,
            "evidence": {
                **dict(evidence),
                "execution_marker_file_sha256": marker_file_sha256,
            },
            "e0_non_promotable": True,
            "e1_authorized": decision == "READY_G3_E1_COMPLETION",
            "policy_evaluation_authorized": False,
            "promotion_authorized": False,
            "dashboard_eligible": False,
        }
    )
    core.write_immutable_json(out_dir / TERMINAL_RESULT_NAME, payload)
    return payload


def _execute_pipeline(
    *,
    out_dir: Path,
    preflight_lock: Path,
    lock: Mapping[str, Any],
    marker_file_sha256: str,
) -> dict[str, Any]:
    """Run or resume the frozen core pipeline after marker validation."""

    stage = "ordinary_labels"
    try:
        records_manifest = core.json_object(
            out_dir / str(lock["record_manifest_name"])
        )
        task_manifest = core.json_object(
            out_dir / str(lock["task_manifest_name"])
        )
        records = records_manifest["records"]
        ordinary_tasks = [
            task for task in task_manifest["tasks"]
            if task["partition"] != "transfer_diagnostic"
        ]
        transfer_tasks = [
            task for task in task_manifest["tasks"]
            if task["partition"] == "transfer_diagnostic"
        ]
        policy = core.make_policy(str(lock["incumbent_policy_spec"]))
        scientific_identity = core.execution_identity(lock)
        store_identity = {
            **scientific_identity,
            "partition": "ordinary",
            "tasks_sha256": core.canonical_sha256(ordinary_tasks),
            "execution_marker_file_sha256": marker_file_sha256,
        }
        with core.LabelStore(
            out_dir / core.ORDINARY_DB_NAME,
            identity=store_identity,
        ) as store:
            completed = store.completed_keys()
            pending = [
                task for task in ordinary_tasks
                if task["task_key"] not in completed
            ]
            for offset in range(0, len(pending), core.MAX_CHUNK_SIZE):
                chunk = [
                    core.rollout_label(task, policy)
                    for task in pending[offset:offset + core.MAX_CHUNK_SIZE]
                ]
                store.insert_chunk(chunk)
                if core._directory_size(out_dir) >= core.MAX_OUTPUT_BYTES:
                    raise RuntimeError("E0 output exceeded 4 GiB")
                stat = os.statvfs(out_dir)
                if stat.f_bavail * stat.f_frsize / 1024**3 < core.MIN_FREE_GIB:
                    raise RuntimeError("Free disk fell below 100 GiB")
            ordinary_paths = store.rows()
        if len(ordinary_paths) != int(lock["ordinary_path_count"]):
            raise RuntimeError("Ordinary label completion mismatch")

        stage = "ordinary_fit"
        checkpoint_path = out_dir / core.CHECKPOINT_SEAL_NAME
        if checkpoint_path.exists():
            model, checkpoint = _load_checkpoint(
                out_dir=out_dir,
                preflight_lock=preflight_lock,
                marker_file_sha256=marker_file_sha256,
            )
            ordinary_decision = str(checkpoint["ordinary_decision"])
        else:
            train_records = [
                row for row in records if row["partition"] == "train"
            ]
            dev_records = [
                row for row in records
                if row["partition"] == "development"
            ]
            train_paths = [
                row for row in ordinary_paths
                if row["partition"] == "train"
            ]
            dev_paths = [
                row for row in ordinary_paths
                if row["partition"] == "development"
            ]
            train_rows = core.aggregate_grouped_rows(
                train_records, train_paths, family_balanced=True
            )
            dev_rows = core.aggregate_grouped_rows(
                dev_records, dev_paths, family_balanced=False
            )
            expected_sources = {
                "preflight_lock": sha256_path(preflight_lock),
                "ordinary_labels":
                    sha256_path(out_dir / core.ORDINARY_DB_NAME),
            }
            model_dir = out_dir / core.MODEL_DIR_NAME
            if model_dir.exists():
                model = core.G3HazardModel.load(
                    model_dir,
                    expected_source_hashes=expected_sources,
                )
            else:
                model = core.fit_hazard_model(
                    train_rows,
                    dev_rows,
                    source_hashes=expected_sources,
                )
                model.save(model_dir)
            stability = core.model_stability_audit(model)
            stage = "ordinary_evaluation"
            ordinary_report = core.predictive_report(
                dev_records,
                dev_paths,
                model,
                bootstrap_seed=core.DEV_BOOTSTRAP_SEED,
            )
            ordinary_decision = core.ordinary_gate_decision(
                ordinary_report,
                integrity_passes=True,
                model_stable=stability["passes"],
            )
            checkpoint = core.payload_with_hash(
                {
                    "version": VERSION,
                    "model_meta_sha256": sha256_path(
                        out_dir / core.MODEL_DIR_NAME / "meta.json"
                    ),
                    "model_arrays_sha256": sha256_path(
                        out_dir / core.MODEL_DIR_NAME / "arrays.npz"
                    ),
                    "ordinary_labels_sha256": sha256_path(
                        out_dir / core.ORDINARY_DB_NAME
                    ),
                    "model_stability": stability,
                    "ordinary_report": ordinary_report,
                    "ordinary_decision": ordinary_decision,
                    "execution_marker_file_sha256": marker_file_sha256,
                    "promotable": False,
                }
            )
            core.write_immutable_json(checkpoint_path, checkpoint)
        if ordinary_decision != "READY_G3_E0_ORDINARY_PREDICTIVE":
            return _seal_terminal_decision(
                out_dir,
                decision="KILL_G3_BOOTSTRAP_PREDICTIVE",
                stage="ordinary_evaluation",
                evidence={
                    "checkpoint_file_sha256":
                        sha256_path(checkpoint_path),
                    "transfer_predictions_opened": False,
                    "transfer_labels_opened": False,
                },
                marker_file_sha256=marker_file_sha256,
            )

        stage = "transfer_predictions"
        prediction_path = out_dir / core.PREDICTION_SEAL_NAME
        if prediction_path.exists():
            prediction_seal = core.json_object(prediction_path)
            if not core.verify_payload_hash(prediction_seal):
                raise ValueError("Transfer prediction payload mismatch")
            if (
                prediction_seal.get("checkpoint_file_sha256")
                != sha256_path(checkpoint_path)
            ):
                raise ValueError("Transfer prediction checkpoint mismatch")
            if (
                prediction_seal.get("execution_marker_file_sha256")
                != marker_file_sha256
            ):
                raise ValueError("Transfer prediction marker mismatch")
            predictions = prediction_seal["predictions"]
            if (
                core.canonical_sha256(predictions)
                != prediction_seal["predictions_sha256"]
            ):
                raise ValueError("Transfer predictions changed")
            activity = dict(prediction_seal["activity"])
        else:
            transfer_records = [
                row for row in records
                if row["partition"] == "transfer_diagnostic"
            ]
            predictions, activity = core.predict_transfer_actions(
                transfer_records,
                transfer_tasks,
                model,
                policy,
            )
            prediction_seal = core.payload_with_hash(
                {
                    "version": VERSION,
                    "checkpoint_file_sha256":
                        sha256_path(checkpoint_path),
                    "predictions": predictions,
                    "predictions_sha256":
                        core.canonical_sha256(predictions),
                    "activity": activity,
                    "transfer_label_values_opened": False,
                    "execution_marker_file_sha256": marker_file_sha256,
                    "promotable": False,
                }
            )
            core.write_immutable_json(prediction_path, prediction_seal)
        activity_passes = (
            int(activity["roots"]) >= 6
            and int(activity["corner2"]) >= 1
            and int(activity["incumbent"]) >= 1
        )
        if not activity_passes:
            return _seal_terminal_decision(
                out_dir,
                decision="HOLD_G3_E0_UNDERPOWERED_TRANSFER",
                stage="transfer_activity",
                evidence={
                    "checkpoint_file_sha256":
                        sha256_path(checkpoint_path),
                    "prediction_file_sha256":
                        sha256_path(prediction_path),
                    "activity": activity,
                    "transfer_labels_opened": False,
                },
                marker_file_sha256=marker_file_sha256,
            )

        stage = "transfer_labels"
        transfer_identity = {
            **scientific_identity,
            "partition": "transfer_diagnostic",
            "tasks_sha256": core.canonical_sha256(transfer_tasks),
            "checkpoint_sha256": sha256_path(checkpoint_path),
            "prediction_sha256": sha256_path(prediction_path),
            "execution_marker_file_sha256": marker_file_sha256,
        }
        with core.LabelStore(
            out_dir / core.TRANSFER_DB_NAME,
            identity=transfer_identity,
            transfer=True,
            checkpoint_seal=checkpoint_path,
            prediction_seal=prediction_path,
        ) as store:
            completed = store.completed_keys()
            pending = [
                task for task in transfer_tasks
                if task["task_key"] not in completed
            ]
            for offset in range(0, len(pending), core.MAX_CHUNK_SIZE):
                chunk = [
                    core.rollout_label(task, policy)
                    for task in pending[offset:offset + core.MAX_CHUNK_SIZE]
                ]
                store.insert_chunk(chunk)
            transfer_paths = store.rows()
        if len(transfer_paths) != int(lock["transfer_path_count"]):
            raise RuntimeError("Transfer label completion mismatch")

        stage = "transfer_evaluation"
        transfer_records = [
            row for row in records
            if row["partition"] == "transfer_diagnostic"
        ]
        transfer_report = core.predictive_report(
            transfer_records,
            transfer_paths,
            model,
            bootstrap_seed=core.TRANSFER_BOOTSTRAP_SEED,
        )
        decision = core.transfer_gate_decision(
            transfer_report,
            activity=activity,
            integrity_passes=True,
        )
        return _seal_terminal_decision(
            out_dir,
            decision=decision,
            stage="transfer_evaluation",
            evidence={
                "checkpoint_file_sha256": sha256_path(checkpoint_path),
                "prediction_file_sha256": sha256_path(prediction_path),
                "transfer_labels_sha256":
                    sha256_path(out_dir / core.TRANSFER_DB_NAME),
                "activity": activity,
                "transfer_report": transfer_report,
                "n32_mde_or": 4.0,
            },
            marker_file_sha256=marker_file_sha256,
        )
    except Exception as error:
        if not (out_dir / TERMINAL_RESULT_NAME).exists():
            _seal_terminal_error(
                out_dir,
                stage=stage,
                error=error,
                marker_file_sha256=marker_file_sha256,
            )
        raise


def execute(
    *,
    out_dir: Path,
    preflight_lock: Path,
    jobs: int,
) -> dict[str, Any]:
    """Validate the existing marker, then run or resume the frozen pipeline."""

    lock, marker_audit = validate_execution_marker(
        out_dir=out_dir,
        preflight_lock=preflight_lock,
        jobs=jobs,
        revalidate_operations=True,
    )
    return _execute_pipeline(
        out_dir=out_dir,
        preflight_lock=preflight_lock,
        lock=lock,
        marker_file_sha256=marker_audit["marker_file_sha256"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("open", "execute"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--out-dir", type=Path, required=True)
        command_parser.add_argument(
            "--preflight-lock", type=Path, required=True
        )
        command_parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    if os.nice(0) < core.MIN_NICE:
        os.nice(core.MIN_NICE - os.nice(0))
    if args.command == "open":
        result = open_execution(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    else:
        result = execute(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

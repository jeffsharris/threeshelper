"""Marker-bound O6 competing-risks P0 execution surface.

Only ``audit-zero-work`` is authorized at the current research boundary.
The remaining commands are deliberately separate, fail-closed state-machine
steps for a later authorization.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import sqlite3
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from threes_rl import o6_competing_risks_p0 as prep


VERSION = "o6_competing_risks_p0_execution_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]

CHARTER_PATH = Path("threes_rl/O6_COMPETING_RISKS_P0_EXECUTION_CHARTER.md")
RUNNER_PATH = Path("threes_rl/o6_competing_risks_p0_execute.py")
TEST_PATH = Path("tests/test_rl_o6_competing_risks_p0_execute.py")
OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/o6_competing_risks_p0_execution_v1"
)

TEST_EVIDENCE_NAME = "O6_P0_EXECUTION_TEST_EVIDENCE.json"
PREFLIGHT_LOCK_NAME = "O6_P0_EXECUTION_PREFLIGHT_LOCK.json"
PREFLIGHT_RESULT_NAME = "O6_P0_EXECUTION_PREFLIGHT_RESULT.json"
OPENED_NAME = "O6_P0_EXECUTION_OPENED.json"
PROTECTED_INVENTORY_NAME = "O6_P0_PROTECTED_INVENTORY.json"
EXCLUSION_UNION_NAME = "O6_P0_EXCLUSION_UNION.json"
SOURCE_INVENTORY_NAME = "O6_P0_SOURCE_INVENTORY.json"
CANDIDATE_ROOTS_NAME = "O6_P0_CANDIDATE_ROOTS.json"
SELECTED_ROOTS_NAME = "O6_P0_SELECTED_ROOTS.json"
STREAM_RESERVATION_NAME = "O6_P0_STREAM_RESERVATION.json"
COLLISION_AUDIT_NAME = "O6_P0_COLLISION_AUDIT.json"
POWER_DB_NAME = "O6_P0_POWER_PROGRESS.sqlite3"
POWER_TABLE_NAME = "O6_P0_POWER_TABLE.json"
OPERATIONAL_AUDIT_NAME = "O6_P0_OPERATIONAL_AUDIT.json"
RUNTIME_NAME = "O6_P0_RUNTIME.json"
RESULT_NAME = "O6_P0_RESULT.json"

OUTPUT_NAMES = (
    TEST_EVIDENCE_NAME,
    PREFLIGHT_LOCK_NAME,
    PREFLIGHT_RESULT_NAME,
    OPENED_NAME,
    PROTECTED_INVENTORY_NAME,
    EXCLUSION_UNION_NAME,
    SOURCE_INVENTORY_NAME,
    CANDIDATE_ROOTS_NAME,
    SELECTED_ROOTS_NAME,
    STREAM_RESERVATION_NAME,
    COLLISION_AUDIT_NAME,
    POWER_DB_NAME,
    POWER_TABLE_NAME,
    OPERATIONAL_AUDIT_NAME,
    RUNTIME_NAME,
    RESULT_NAME,
)

PARENT_PREPARATION_SHA256 = {
    str(prep.CHARTER_PATH): (
        "2ee1e4273866f7f40376fb584e908f5a0e10e70446e2540f36bf320ac0edbb11"
    ),
    str(prep.RUNNER_PATH): (
        "c1a1d0a22fa185672e62f0b712d79d8bd01d76e04cebe04bca78b45a7c092dd6"
    ),
    str(prep.TEST_PATH): (
        "3d7cbe8f20149f3b21305e8306762f8ede78f2227094f8848dc5b6f383ba0b34"
    ),
}

FAMILY_ORDER = prep.FAMILY_ORDER
FAMILY_SIGNATURES = prep.FAMILY_SIGNATURES
TARGET_ORDER = prep.TARGET_ORDER
ROLE_ORDER = prep.ROLE_ORDER
FAMILY_POLICY_SPECS = {
    "o6_corner2": "corner2",
    "o6_expectimax2": "expectimax2",
    "o6_parent_mc1000": (
        "ntuple_expectimax2:threes_rl/runs/"
        "td_default_corner2_mc_1000_init3000_a0005_20260706/latest"
    ),
    "o6_replaycal": (
        "ntuple_expectimax2:threes_rl/runs/"
        "replay_cal_phase4_late_midlate_top13_e3_a001_tc_20260706/latest"
    ),
}
POLICY_SPEC_FIELDS = (
    "acquisition_policy_spec",
    "policy_spec",
    "source_policy_spec",
    "policy",
)

CANDIDATE_PATTERNS = (
    "threes_rl/runs/**/replay.json",
    "threes_rl/runs/**/source_replays/*.json",
    "threes_rl/runs/replays/**/*.json",
)
PROTECTED_PATTERNS = prep.PROTECTED_DISCOVERY_PATTERNS
LIVE_PATHS = frozenset(prep.LIVE_SUMMARY_PATHS)
GOVERNANCE_TOKENS = prep.GOVERNANCE_FILENAME_TOKENS
IDENTITY_FIELDS = prep.PROTECTED_IDENTITY_FIELDS
STREAM_FIELDS = prep.STREAM_FIELDS

MIN_NICE = 10
MAX_ACTIVE_HOURS = 36.0
MAX_OUTPUT_BYTES = 4 * 1024**3
PROJECTED_OUTPUT_BYTES = int(1.5 * 1024**3)
PROJECTED_ACTIVE_HOURS = 24.0
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
TOP_THREE = (263_670, 261_369, 258_561)
INCUMBENT_POLICY_SHA256 = (
    "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4"
)
SERVICE_PORTS = (8765, 8770)
POWER_DATASET_BATCH_SIZE = 16
POWER_BOOTSTRAP_BATCH_SIZE = 64

HEAVY_MODULE_PATTERNS = (
    "threes_rl.eval",
    "threes_rl.train",
    "threes_rl.o3_",
    "threes_rl.o4_",
    "threes_rl.o5_",
    "threes_rl.o6_competing_risks_p0_execute",
    "threes_rl.g3_",
    "threes_rl.c1_",
    "threes_rl.c2_",
    "threes_rl.k1_",
)


class O6ExecutionError(RuntimeError):
    """Base fail-closed execution error."""


class O6DataHold(O6ExecutionError):
    """Outcome-free data, cost, or operational HOLD."""


class O6IntegrityKill(O6ExecutionError):
    """Immutable identity, schema, or deterministic-state KILL."""


def canonical_json_hash(value: Any) -> str:
    return prep.canonical_json_hash(value)


def sha256_path(path: str | Path) -> str:
    return prep.sha256_path(path)


def _repo_path(path: str | Path) -> Path:
    return REPO_ROOT / Path(path)


def _normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _payload_with_hash(
    payload: Mapping[str, Any],
    field: str = "payload_sha256",
) -> dict[str, Any]:
    normalized = _normalize(dict(payload))
    normalized.pop(field, None)
    normalized[field] = canonical_json_hash(normalized)
    return normalized


def _verify_payload_hash(
    payload: Mapping[str, Any],
    field: str = "payload_sha256",
) -> bool:
    expected = payload.get(field)
    if not isinstance(expected, str):
        return False
    body = dict(payload)
    body.pop(field)
    return canonical_json_hash(body) == expected


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str = "payload_sha256",
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Immutable O6 artifact exists: {path}")
    body = _payload_with_hash(payload, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            json.dump(body, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "payload_sha256": body[field],
    }


def _load_hashed_json(
    path: Path,
    *,
    field: str = "payload_sha256",
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict) or not _verify_payload_hash(payload, field):
        raise O6IntegrityKill(f"Invalid self-hash in {path}")
    return payload


def _write_mutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str = "payload_sha256",
) -> None:
    body = _payload_with_hash(payload, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="ascii") as handle:
            json.dump(body, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
        return True
    except ValueError:
        return False


def _source_identity() -> dict[str, Any]:
    paths = {
        "parent_charter": prep.CHARTER_PATH,
        "parent_runner": prep.RUNNER_PATH,
        "parent_tests": prep.TEST_PATH,
        "execution_charter": CHARTER_PATH,
        "execution_runner": RUNNER_PATH,
        "execution_tests": TEST_PATH,
    }
    return {
        name: {
            "path": str(path),
            "sha256": (
                sha256_path(_repo_path(path))
                if _repo_path(path).is_file()
                else None
            ),
            "exists": _repo_path(path).is_file(),
        }
        for name, path in paths.items()
    }


def accepted_parent_audit() -> dict[str, Any]:
    rows = {}
    for path, expected in PARENT_PREPARATION_SHA256.items():
        absolute = _repo_path(path)
        actual = sha256_path(absolute) if absolute.is_file() else None
        rows[path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
    checks = {
        "parent_files_exact": all(row["matches"] for row in rows.values()),
        "risk_schema_exact": (
            prep.RISK_SCHEMA_SHA256 == prep.EXPECTED_RISK_SCHEMA_SHA256
        ),
        "source_schema_exact": (
            prep.SOURCE_STATE_SCHEMA_SHA256
            == prep.EXPECTED_SOURCE_STATE_SCHEMA_SHA256
        ),
        "protected_contract_exact": (
            prep.PROTECTED_CONTRACT_SHA256
            == prep.EXPECTED_PROTECTED_CONTRACT_SHA256
        ),
        "power_contract_exact": (
            prep.POWER_CONTRACT_SHA256
            == prep.EXPECTED_POWER_CONTRACT_SHA256
        ),
    }
    return {
        "files": rows,
        "risk_schema_sha256": prep.RISK_SCHEMA_SHA256,
        "source_state_schema_sha256": prep.SOURCE_STATE_SCHEMA_SHA256,
        "protected_contract_sha256": prep.PROTECTED_CONTRACT_SHA256,
        "power_contract_sha256": prep.POWER_CONTRACT_SHA256,
        "checks": checks,
        "passes": all(checks.values()),
    }


def current_nice() -> int:
    return int(os.getpriority(os.PRIO_PROCESS, 0))


def free_disk_gib() -> float:
    return shutil.disk_usage(REPO_ROOT).free / 1024**3


def _socket_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout):
            return True
    except OSError:
        return False


def recorder_health() -> dict[str, Any]:
    url = "http://127.0.0.1:8770/api/health"
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        return {
            "url": url,
            "status": None,
            "advisor_ready": None,
            "error_type": type(error).__name__,
            "active_session_content_read": False,
            "passes": False,
        }
    advisor = payload.get("advisor")
    if isinstance(advisor, Mapping):
        advisor_ready = (
            bool(advisor.get("ready"))
            if "ready" in advisor
            else advisor.get("status") == "ready"
        )
        policy_sha256 = advisor.get("policy_file_sha256")
    else:
        advisor_ready = bool(payload.get("advisor_ready"))
        policy_sha256 = payload.get("advisor_policy_sha256")
    return {
        "url": url,
        "status": payload.get("status"),
        "advisor_ready": advisor_ready,
        "advisor_policy_sha256": policy_sha256,
        "advisor_policy_exact": policy_sha256 == INCUMBENT_POLICY_SHA256,
        "active_session_content_read": False,
        "passes": (
            payload.get("status") == "ok"
            and bool(advisor_ready)
            and policy_sha256 == INCUMBENT_POLICY_SHA256
        ),
    }


def service_audit() -> dict[str, Any]:
    ports = {str(port): _socket_open(port) for port in SERVICE_PORTS}
    recorder = recorder_health()
    dashboard_path = _repo_path("threes_rl/runs/dashboard/dashboard.json")
    try:
        dashboard_payload = json.loads(
            dashboard_path.read_text(encoding="utf-8")
        )
        dashboard_scores = tuple(
            int(row["score"])
            for row in dashboard_payload["global_top_replays"][:3]
        )
        dashboard_best = int(dashboard_payload["best_high_score"])
        dashboard = {
            "path": str(dashboard_path),
            "best_high_score": dashboard_best,
            "top_three": dashboard_scores,
            "passes": (
                dashboard_best == TOP_THREE[0]
                and dashboard_scores == TOP_THREE
            ),
        }
    except Exception as error:
        dashboard = {
            "path": str(dashboard_path),
            "best_high_score": None,
            "top_three": (),
            "error_type": type(error).__name__,
            "passes": False,
        }
    checks = {
        "ports_open": all(ports.values()),
        "recorder_status_and_advisor_ready": recorder["passes"],
        "active_session_content_unread": (
            not recorder["active_session_content_read"]
        ),
        "dashboard_top_three_exact": dashboard["passes"],
    }
    return {
        "ports": ports,
        "recorder": recorder,
        "dashboard": dashboard,
        "protected_top_three": TOP_THREE,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _ancestor_pids(pid: int | None = None) -> set[int]:
    current = int(pid or os.getpid())
    ancestors = {current}
    while current > 1:
        try:
            output = subprocess.run(
                ("ps", "-o", "ppid=", "-p", str(current)),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            parent = int(output)
        except (OSError, subprocess.SubprocessError, ValueError):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def heavy_process_audit() -> dict[str, Any]:
    ancestors = _ancestor_pids()
    candidates: dict[int, set[str]] = defaultdict(set)
    for pattern in HEAVY_MODULE_PATTERNS:
        result = subprocess.run(
            ("pgrep", "-f", pattern),
            check=False,
            capture_output=True,
            text=True,
        )
        for token in result.stdout.split():
            try:
                candidates[int(token)].add(pattern)
            except ValueError:
                continue
    unrelated = {
        pid: sorted(patterns)
        for pid, patterns in candidates.items()
        if pid not in ancestors
    }
    return {
        "method": "PID-only pgrep; command lines and sessions not recorded",
        "ancestor_pids": sorted(ancestors),
        "candidate_pids": sorted(candidates),
        "unrelated_candidate_pids": sorted(unrelated),
        "matched_patterns_by_unrelated_pid": unrelated,
        "passes": not unrelated,
    }


def operational_audit(*, output_dir: Path = _repo_path(OUTPUT_DIR)) -> dict[str, Any]:
    disk = free_disk_gib()
    nice = current_nice()
    process = heavy_process_audit()
    services = service_audit()
    output_bytes = directory_bytes(output_dir)
    checks = {
        "nice_at_least_10": nice >= MIN_NICE,
        "one_heavy_job": process["passes"],
        "free_disk_above_100_gib": disk > MIN_FREE_GIB,
        "target_120_gib_met": disk > TARGET_FREE_GIB,
        "output_below_4_gib": output_bytes < MAX_OUTPUT_BYTES,
        "services_healthy": services["passes"],
    }
    return {
        "nice": nice,
        "free_gib": disk,
        "output_bytes": output_bytes,
        "process": process,
        "services": services,
        "checks": checks,
        "passes": all(checks.values()),
    }


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )


def output_absence_audit(
    *,
    output_dir: Path = _repo_path(OUTPUT_DIR),
) -> dict[str, Any]:
    expected = {name: output_dir / name for name in OUTPUT_NAMES}
    present = sorted(name for name, path in expected.items() if path.exists())
    unknown = []
    if output_dir.exists():
        unknown = sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file() and path.name not in OUTPUT_NAMES
        )
    checks = {
        "output_directory_absent": not output_dir.exists(),
        "all_execution_artifacts_absent": not present,
        "no_unknown_output_files": not unknown,
    }
    return {
        "output_dir": str(output_dir),
        "present": present,
        "unknown": unknown,
        "checks": checks,
        "passes": all(checks.values()),
    }


def power_workload_contract() -> dict[str, Any]:
    parent = prep.power_workload_estimate()
    checks = {
        "sixty_cells": parent["cells"] == 60,
        "datasets_exact": parent["datasets"] == 245_760,
        "bootstraps_exact": (
            parent["whole_root_bootstraps"] == 1_006_632_960
        ),
        "root_draws_exact": (
            parent["whole_root_index_draws"] == 338_228_674_560
        ),
        "dataset_batch_16": POWER_DATASET_BATCH_SIZE == 16,
        "bootstrap_batch_64": POWER_BOOTSTRAP_BATCH_SIZE == 64,
        "no_power_executed": not parent["power_executed"],
    }
    return {
        **parent,
        "projected_output_bytes": PROJECTED_OUTPUT_BYTES,
        "projected_active_hours": PROJECTED_ACTIVE_HOURS,
        "hard_output_bytes": MAX_OUTPUT_BYTES,
        "hard_active_hours": MAX_ACTIVE_HOURS,
        "checks": checks,
        "passes": all(checks.values()),
    }


def zero_work_preflight(
    *,
    output_dir: Path = _repo_path(OUTPUT_DIR),
) -> dict[str, Any]:
    parent = accepted_parent_audit()
    dependencies = prep.dependency_audit(repo_root=REPO_ROOT)
    governance = prep.protected_source_audit(repo_root=REPO_ROOT)
    family = prep.family_contract_audit()
    matrices = {
        str(n): prep.role_matrix_contract(n)
        for n in prep.POWER_ROOT_COUNTS
    }
    streams = prep.stream_window_contract()
    workload = power_workload_contract()
    absence = output_absence_audit(output_dir=output_dir)
    operational = operational_audit(output_dir=output_dir)
    source_identity = _source_identity()
    checks = {
        "accepted_parent_exact": parent["passes"],
        "execution_sources_exist": all(
            row["exists"] for row in source_identity.values()
        ),
        "dependencies_exact": dependencies["passes"],
        "protected_governance_exact": governance["passes"],
        "family_contract_exact": family["passes"],
        "all_role_matrices_exact": all(
            row["passes"] for row in matrices.values()
        ),
        "stream_contract_frozen_unreserved": streams["passes"],
        "power_workload_exact_unexecuted": workload["passes"],
        "output_absent": absence["passes"],
        "operational_ready": operational["passes"],
        "corpus_source_scan_zero": True,
        "historical_collision_scan_zero": True,
        "root_content_reads_zero": True,
        "root_selection_zero": True,
        "stream_reservations_zero": True,
        "power_datasets_zero": True,
        "labels_training_outcomes_zero": True,
    }
    return {
        "version": VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_identity": source_identity,
        "accepted_parent": parent,
        "dependency_audit": dependencies,
        "protected_governance_audit": governance,
        "family_contract": family,
        "role_matrix_contracts": matrices,
        "stream_contract": streams,
        "power_workload": workload,
        "output_absence": absence,
        "operational": operational,
        "checks": checks,
        "passes": all(checks.values()),
        "decision": (
            "READY_O6_P0_EXECUTION_SURFACE_REVIEW"
            if all(checks.values())
            else "HOLD_O6_P0_EXECUTION_SURFACE_REVIEW"
        ),
        "forbidden_work": {
            "execution_markers": 0,
            "corpus_source_scans": 0,
            "candidate_root_content_reads": 0,
            "historical_collision_scans": 0,
            "root_selections": 0,
            "stream_reservations": 0,
            "stream_consumption": 0,
            "power_datasets": 0,
            "root_bootstraps": 0,
            "labels": 0,
            "training_steps": 0,
            "checkpoints": 0,
            "development_or_untouched_reads": 0,
            "policy_outcomes": 0,
        },
    }


def write_test_evidence(
    *,
    focused_passed: int,
    regressions_passed: int,
    deselections: Sequence[str],
    commands: Sequence[str],
    output_dir: Path = _repo_path(OUTPUT_DIR),
) -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_test_evidence",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_identity": _source_identity(),
        "focused_passed": int(focused_passed),
        "regressions_passed": int(regressions_passed),
        "documented_deselections": list(deselections),
        "commands": list(commands),
        "zero_work": {
            "markers": 0,
            "source_scans": 0,
            "root_content_reads": 0,
            "collisions_scanned": 0,
            "streams_reserved": 0,
            "power_datasets": 0,
            "labels": 0,
            "training_steps": 0,
            "outcomes": 0,
        },
        "passes": focused_passed > 0 and regressions_passed > 0,
    }
    return _write_immutable_json(
        output_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def seal_preflight(
    *,
    output_dir: Path = _repo_path(OUTPUT_DIR),
) -> dict[str, Any]:
    evidence_path = output_dir / TEST_EVIDENCE_NAME
    evidence = _load_hashed_json(
        evidence_path,
        field="test_evidence_payload_sha256",
    )
    if not evidence.get("passes"):
        raise O6IntegrityKill("O6 execution test evidence did not pass")
    existing_entries = {
        path.name for path in output_dir.iterdir()
    }
    if existing_entries != {TEST_EVIDENCE_NAME}:
        raise O6IntegrityKill(
            f"O6 preflight namespace differs: {sorted(existing_entries)}"
        )
    if any((output_dir / name).exists() for name in OUTPUT_NAMES[1:]):
        raise O6IntegrityKill("O6 execution namespace is not preflight-clean")
    audit = zero_work_preflight(output_dir=Path("__o6_absent_probe__"))
    if not audit["passes"]:
        raise O6DataHold("O6 execution zero-work preflight did not pass")
    command = (
        "nice -n 10 env PYTHONPATH=. .venv/bin/python -m "
        "threes_rl.o6_competing_risks_p0_execute execute "
        f"--out-dir {OUTPUT_DIR} --jobs 1"
    )
    lock_payload = {
        "version": f"{VERSION}_preflight_lock",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bound_output_dir": str(output_dir.resolve()),
        "source_identity": _source_identity(),
        "test_evidence": {
            "file_sha256": sha256_path(evidence_path),
            "payload_sha256": evidence["test_evidence_payload_sha256"],
        },
        "accepted_parent": accepted_parent_audit(),
        "dependency_audit": prep.dependency_audit(repo_root=REPO_ROOT),
        "family_contract": prep.family_contract_audit(),
        "role_matrix_contracts": {
            str(n): prep.role_matrix_contract(n)
            for n in prep.POWER_ROOT_COUNTS
        },
        "stream_contract": prep.stream_window_contract(),
        "power_workload": power_workload_contract(),
        "jobs": 1,
        "minimum_nice": MIN_NICE,
        "max_active_hours": MAX_ACTIVE_HOURS,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "min_free_gib": MIN_FREE_GIB,
        "target_free_gib": TARGET_FREE_GIB,
        "execute_command": command,
        "zero_work": True,
        "passes": True,
    }
    lock = _write_immutable_json(
        output_dir / PREFLIGHT_LOCK_NAME,
        lock_payload,
        field="preflight_lock_payload_sha256",
    )
    result_payload = {
        "version": f"{VERSION}_preflight_result",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "READY_O6_P0_EXECUTION",
        "preflight_lock_file_sha256": lock["file_sha256"],
        "preflight_lock_payload_sha256": lock["payload_sha256"],
        "test_evidence_file_sha256": sha256_path(evidence_path),
        "zero_work": True,
        "passes": True,
    }
    result = _write_immutable_json(
        output_dir / PREFLIGHT_RESULT_NAME,
        result_payload,
        field="preflight_result_payload_sha256",
    )
    return {"lock": lock, "result": result}


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _is_under(path: Path, relative: str | Path) -> bool:
    try:
        path.resolve().relative_to(_repo_path(relative).resolve())
        return True
    except ValueError:
        return False


def _inventory_class(path: Path, candidate_paths: set[Path]) -> str:
    relative = _relative(path)
    if relative in LIVE_PATHS:
        return "live_summary"
    if any(_is_under(path, prefix) for prefix in prep.FORBIDDEN_BODY_PREFIXES):
        return "protected_hash_only_body"
    if _is_under(path, "threes_rl/runs/human_diagnostics"):
        return "protected_hash_only_human"
    if _is_under(path, "threes_rl/runs/continuations"):
        return "protected_hash_only_continuation"
    if _is_under(path, "threes_rl/runs/replays/top3"):
        return "protected_hash_only_top_three"
    if path.resolve() in candidate_paths:
        return "candidate_replay"
    lowered = path.name.lower()
    if any(token in lowered for token in GOVERNANCE_TOKENS):
        return "protected_governance_identity"
    return "protected_hash_only_other"


def build_byte_inventory(
    *,
    output_dir: Path = _repo_path(OUTPUT_DIR),
) -> dict[str, Any]:
    candidate_raw = [
        path
        for pattern in CANDIDATE_PATTERNS
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    ]
    protected_raw = [
        path
        for pattern in PROTECTED_PATTERNS
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    ]
    for path in candidate_raw + protected_raw:
        if path.is_symlink():
            raise O6DataHold(f"Symlink inventory path is forbidden: {path}")
    candidates = {path.resolve() for path in candidate_raw}
    canonical_to_raw: dict[Path, set[Path]] = defaultdict(set)
    for path in candidate_raw + protected_raw:
        canonical_to_raw[path.resolve()].add(path.absolute())
    aliases = {
        canonical: paths
        for canonical, paths in canonical_to_raw.items()
        if len(paths) > 1
    }
    if aliases:
        raise O6DataHold(
            "Canonical inventory aliases are forbidden: "
            + ",".join(str(path) for path in sorted(aliases, key=str))
        )
    inode_to_paths: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in canonical_to_raw:
        stat = path.stat()
        inode_to_paths[(int(stat.st_dev), int(stat.st_ino))].append(path)
    hardlink_aliases = [
        paths for paths in inode_to_paths.values() if len(paths) > 1
    ]
    if hardlink_aliases:
        raise O6DataHold("Hard-link inventory aliases are forbidden")
    all_paths = sorted(canonical_to_raw, key=str)
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for resolved in all_paths:
        if _is_under(resolved, output_dir):
            continue
        original = Path(resolved)
        if (
            not original.is_file()
            or original.is_symlink()
            or not _inside_repo(original)
            or resolved in seen
        ):
            raise O6DataHold(f"Unsafe or duplicate inventory path: {original}")
        seen.add(resolved)
        classification = _inventory_class(original, candidates)
        rows.append(
            {
                "path": _relative(original),
                "sha256": sha256_path(original),
                "bytes": int(original.stat().st_size),
                "classification": classification,
                "byte_stable": classification != "live_summary",
            }
        )
    classifications = Counter(row["classification"] for row in rows)
    checks = {
        "paths_unique": len(rows) == len({row["path"] for row in rows}),
        "all_paths_inside_repo": all(
            _inside_repo(_repo_path(row["path"])) for row in rows
        ),
        "all_paths_regular_nonsymlink": all(
            _repo_path(row["path"]).is_file()
            and not _repo_path(row["path"]).is_symlink()
            for row in rows
        ),
        "live_paths_only_named_summaries": all(
            row["path"] in LIVE_PATHS
            for row in rows
            if not row["byte_stable"]
        ),
        "candidate_rows_present": classifications["candidate_replay"] > 0,
        "payloads_unparsed": True,
    }
    return {
        "version": f"{VERSION}_byte_inventory",
        "rows": rows,
        "row_count": len(rows),
        "classification_counts": dict(sorted(classifications.items())),
        "inventory_sha256": canonical_json_hash(rows),
        "checks": checks,
        "passes": all(checks.values()),
        "payloads_parsed": False,
    }


def validate_byte_inventory(
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    rows = inventory.get("rows")
    if not isinstance(rows, list):
        raise O6IntegrityKill("Marker byte inventory rows are malformed")
    for row in rows:
        if not isinstance(row, Mapping):
            failures.append("non_mapping_row")
            continue
        path = _repo_path(str(row.get("path")))
        if (
            not _inside_repo(path)
            or path.is_symlink()
            or not path.is_file()
        ):
            failures.append(f"unsafe_or_missing:{row.get('path')}")
            continue
        if bool(row.get("byte_stable")):
            actual = sha256_path(path)
            if actual != row.get("sha256"):
                failures.append(f"immutable_hash_drift:{row.get('path')}")
    checks = {
        "inventory_hash_exact": (
            inventory.get("inventory_sha256") == canonical_json_hash(rows)
        ),
        "zero_immutable_failures": not failures,
        "live_summaries_not_byte_bound": all(
            bool(row.get("byte_stable"))
            or str(row.get("path")) in LIVE_PATHS
            for row in rows
        ),
    }
    return {
        "row_count": len(rows),
        "failures": failures,
        "checks": checks,
        "passes": all(checks.values()),
    }


def compare_inventory_to_current(
    inventory: Mapping[str, Any],
    *,
    output_dir: Path = _repo_path(OUTPUT_DIR),
) -> dict[str, Any]:
    current = build_byte_inventory(output_dir=output_dir)
    frozen_rows = {
        str(row["path"]): row for row in inventory.get("rows", [])
    }
    current_rows = {
        str(row["path"]): row for row in current.get("rows", [])
    }
    new_paths = sorted(set(current_rows) - set(frozen_rows))
    missing_paths = sorted(set(frozen_rows) - set(current_rows))
    classification_drift = sorted(
        path
        for path in set(frozen_rows) & set(current_rows)
        if frozen_rows[path]["classification"]
        != current_rows[path]["classification"]
    )
    immutable_hash_drift = sorted(
        path
        for path in set(frozen_rows) & set(current_rows)
        if bool(frozen_rows[path]["byte_stable"])
        and frozen_rows[path]["sha256"] != current_rows[path]["sha256"]
    )
    checks = {
        "zero_new_paths": not new_paths,
        "zero_missing_paths": not missing_paths,
        "zero_classification_drift": not classification_drift,
        "zero_immutable_hash_drift": not immutable_hash_drift,
        "live_rewrites_allowed_only_for_named_paths": all(
            bool(row["byte_stable"]) or path in LIVE_PATHS
            for path, row in frozen_rows.items()
        ),
    }
    return {
        "new_paths": new_paths,
        "missing_paths": missing_paths,
        "classification_drift": classification_drift,
        "immutable_hash_drift": immutable_hash_drift,
        "frozen_inventory_sha256": inventory.get("inventory_sha256"),
        "current_inventory_sha256": current.get("inventory_sha256"),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _validate_preflight_lock(
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock_path = output_dir / PREFLIGHT_LOCK_NAME
    result_path = output_dir / PREFLIGHT_RESULT_NAME
    evidence_path = output_dir / TEST_EVIDENCE_NAME
    lock = _load_hashed_json(
        lock_path,
        field="preflight_lock_payload_sha256",
    )
    result = _load_hashed_json(
        result_path,
        field="preflight_result_payload_sha256",
    )
    evidence = _load_hashed_json(
        evidence_path,
        field="test_evidence_payload_sha256",
    )
    checks = {
        "lock_version_exact": lock.get("version")
        == f"{VERSION}_preflight_lock",
        "bound_output_exact": lock.get("bound_output_dir")
        == str(output_dir.resolve()),
        "source_identity_exact": lock.get("source_identity")
        == _source_identity(),
        "lock_result_identity_exact": (
            result.get("preflight_lock_file_sha256")
            == sha256_path(lock_path)
            and result.get("preflight_lock_payload_sha256")
            == lock["preflight_lock_payload_sha256"]
        ),
        "test_evidence_identity_exact": (
            lock.get("test_evidence", {}).get("file_sha256")
            == sha256_path(evidence_path)
            and lock.get("test_evidence", {}).get("payload_sha256")
            == evidence["test_evidence_payload_sha256"]
        ),
        "ready_decision_exact": (
            result.get("decision") == "READY_O6_P0_EXECUTION"
        ),
        "jobs_exact": lock.get("jobs") == 1,
        "minimum_nice_exact": lock.get("minimum_nice") == MIN_NICE,
    }
    if not all(checks.values()):
        raise O6IntegrityKill(f"O6 preflight binding mismatch: {checks}")
    return lock, result, evidence


def open_execution(
    *,
    output_dir: Path = _repo_path(OUTPUT_DIR),
    jobs: int = 1,
) -> dict[str, Any]:
    if jobs != 1:
        raise O6IntegrityKill("O6 execution jobs must equal one")
    if (output_dir / OPENED_NAME).exists() or (
        output_dir / RESULT_NAME
    ).exists():
        raise O6IntegrityKill("O6 execution marker/result already exists")
    lock, result, evidence = _validate_preflight_lock(
        output_dir=output_dir
    )
    allowed = {
        TEST_EVIDENCE_NAME,
        PREFLIGHT_LOCK_NAME,
        PREFLIGHT_RESULT_NAME,
    }
    actual_entries = [
        path for path in output_dir.iterdir() if not path.name.startswith(".")
    ]
    actual = {path.name for path in actual_entries}
    if actual != allowed:
        raise O6IntegrityKill(
            f"O6 pre-open namespace differs: {sorted(actual)}"
        )
    if any(
        not path.is_file() or path.is_symlink()
        for path in actual_entries
    ):
        raise O6IntegrityKill("O6 pre-open namespace has unsafe entries")
    operational = operational_audit(output_dir=output_dir)
    if not operational["passes"]:
        raise O6DataHold("O6 open operational audit failed")
    inventory = build_byte_inventory(output_dir=output_dir)
    if not inventory["passes"]:
        raise O6DataHold("O6 byte inventory failed")
    marker_payload = {
        "version": f"{VERSION}_opened",
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bound_output_dir": str(output_dir.resolve()),
        "source_identity": _source_identity(),
        "preflight_lock_file_sha256": sha256_path(
            output_dir / PREFLIGHT_LOCK_NAME
        ),
        "preflight_lock_payload_sha256": lock[
            "preflight_lock_payload_sha256"
        ],
        "preflight_result_file_sha256": sha256_path(
            output_dir / PREFLIGHT_RESULT_NAME
        ),
        "preflight_result_payload_sha256": result[
            "preflight_result_payload_sha256"
        ],
        "test_evidence_file_sha256": sha256_path(
            output_dir / TEST_EVIDENCE_NAME
        ),
        "test_evidence_payload_sha256": evidence[
            "test_evidence_payload_sha256"
        ],
        "execute_command": lock["execute_command"],
        "jobs": 1,
        "minimum_nice": MIN_NICE,
        "byte_inventory": inventory,
        "operational": operational,
        "content_parsed": False,
        "historical_collision_scan_executed": False,
        "root_selection_executed": False,
        "stream_reservation_executed": False,
        "power_datasets_executed": 0,
        "labels": 0,
        "training_steps": 0,
        "policy_outcomes": 0,
    }
    return _write_immutable_json(
        output_dir / OPENED_NAME,
        marker_payload,
        field="opened_payload_sha256",
    )


def _load_marker(output_dir: Path) -> dict[str, Any]:
    marker = _load_hashed_json(
        output_dir / OPENED_NAME,
        field="opened_payload_sha256",
    )
    lock, result, evidence = _validate_preflight_lock(
        output_dir=output_dir
    )
    checks = {
        "marker_version_exact": marker.get("version")
        == f"{VERSION}_opened",
        "output_exact": marker.get("bound_output_dir")
        == str(output_dir.resolve()),
        "source_identity_exact": marker.get("source_identity")
        == _source_identity(),
        "lock_file_exact": marker.get("preflight_lock_file_sha256")
        == sha256_path(output_dir / PREFLIGHT_LOCK_NAME),
        "lock_payload_exact": marker.get("preflight_lock_payload_sha256")
        == lock["preflight_lock_payload_sha256"],
        "result_payload_exact": (
            marker.get("preflight_result_payload_sha256")
            == result["preflight_result_payload_sha256"]
        ),
        "evidence_payload_exact": (
            marker.get("test_evidence_payload_sha256")
            == evidence["test_evidence_payload_sha256"]
        ),
        "inventory_exact": validate_byte_inventory(
            marker.get("byte_inventory", {})
        )["passes"],
        "jobs_exact": marker.get("jobs") == 1,
    }
    if not all(checks.values()):
        raise O6IntegrityKill(f"O6 marker binding mismatch: {checks}")
    return marker


def _identity_like_key(key: str) -> bool:
    lowered = key.lower()
    return (
        "ancestry" in lowered
        or "root_cluster" in lowered
        or lowered in {"root", "root_id"}
        or ("stream" in lowered and ("id" in lowered or "seed" in lowered))
        or lowered == "logical_seed"
        or (
            ("replay" in lowered or "state" in lowered)
            and ("sha" in lowered or "hash" in lowered)
        )
    )


def _collect_json_identities(
    value: Any,
    *,
    identities: dict[str, set[str | int]],
    unknown_keys: set[str],
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in IDENTITY_FIELDS:
                if isinstance(child, (str, int)):
                    identities[key].add(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, (str, int)):
                            identities[key].add(item)
            elif _identity_like_key(key):
                unknown_keys.add(key)
            _collect_json_identities(
                child,
                identities=identities,
                unknown_keys=unknown_keys,
            )
    elif isinstance(value, list):
        for child in value:
            _collect_json_identities(
                child,
                identities=identities,
                unknown_keys=unknown_keys,
            )


def _parse_identity_file(
    path: Path,
) -> tuple[dict[str, set[str | int]], set[str]]:
    identities = {field: set() for field in IDENTITY_FIELDS}
    unknown: set[str] = set()
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        _collect_json_identities(
            payload,
            identities=identities,
            unknown_keys=unknown,
        )
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise O6DataHold(
                        f"Non-object JSONL row: {path}:{line_number}"
                    )
                _collect_json_identities(
                    payload,
                    identities=identities,
                    unknown_keys=unknown,
                )
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or ():
                if field not in IDENTITY_FIELDS and _identity_like_key(field):
                    unknown.add(field)
            for row in reader:
                for field in IDENTITY_FIELDS:
                    value = row.get(field)
                    if value not in (None, ""):
                        if field in STREAM_FIELDS or field == "logical_seed":
                            try:
                                identities[field].add(int(value))
                            except ValueError:
                                raise O6DataHold(
                                    f"Noninteger stream identity: {path}:{field}"
                                )
                        else:
                            identities[field].add(str(value))
    else:
        raise O6DataHold(f"Unsupported governance format: {path}")
    return identities, unknown


def build_exclusion_union(
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    union = {field: set() for field in IDENTITY_FIELDS}
    source_rows: list[dict[str, Any]] = []
    unknown: set[str] = set()
    parsed = 0
    for row in inventory.get("rows", []):
        classification = str(row["classification"])
        if classification != "protected_governance_identity":
            continue
        path = _repo_path(str(row["path"]))
        expected_sha = row.get("sha256")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or sha256_path(path) != expected_sha
        ):
            raise O6IntegrityKill(
                f"Governance source hash is absent or changed: {path}"
            )
        identities, local_unknown = _parse_identity_file(path)
        parsed += 1
        for field, values in identities.items():
            union[field].update(values)
        unknown.update(local_unknown)
        source_rows.append(
                {
                    "path": str(row["path"]),
                    "sha256": expected_sha,
                "identity_counts": {
                    field: len(values)
                    for field, values in identities.items()
                    if values
                },
            }
        )
    serialized = {
        field: sorted(values, key=lambda value: (str(type(value)), str(value)))
        for field, values in union.items()
    }
    checks = {
        "zero_unknown_identity_keys": not unknown,
        "at_least_one_governance_source": parsed > 0,
        "forbidden_bodies_unparsed": True,
        "human_bodies_unparsed": True,
    }
    return {
        "version": f"{VERSION}_exclusion_union",
        "source_rows": source_rows,
        "source_count": parsed,
        "unknown_identity_keys": sorted(unknown),
        "identities": serialized,
        "union_sha256": canonical_json_hash(serialized),
        "checks": checks,
        "passes": all(checks.values()),
    }


def whitelisted_sim_state(
    payload: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from threes_rl.sim import SimState, preview_from_label

    board = np.asarray(payload["board"], dtype=np.int32)
    if board.shape != (4, 4):
        raise O6DataHold(f"O6 source board shape changed: {board.shape}")
    if np.any(board < 0):
        raise O6DataHold("O6 source board contains negative values")
    preview_payload = payload["preview"]
    if not isinstance(preview_payload, Mapping):
        raise O6DataHold("O6 source preview must be an object")
    preview_kind = str(preview_payload["kind"])
    if preview_kind == "bonus":
        candidates = tuple(
            int(value) for value in preview_payload.get("candidates", ())
        )
        preview = preview_from_label("large_candidates", candidates)
    else:
        candidates = ()
        preview = preview_from_label(preview_kind)
    cycle = payload["tile_cycle"]
    if not isinstance(cycle, Mapping):
        raise O6DataHold("O6 source tile_cycle must be an object")
    raw_counts = cycle["small_counts"]
    if not isinstance(raw_counts, Mapping):
        raise O6DataHold("O6 small_counts must be an object")
    small_counts = {
        name: int(raw_counts[name]) for name in ("red", "blue", "gray")
    }
    state = SimState(
        board=board.copy(),
        preview=preview,
        small_counts=small_counts,
        small_pos=int(cycle["small_pos"]),
        small_seen_total=int(cycle["small_seen_total"]),
        span_small_pos=int(cycle["span_small_pos"]),
        large_pending=bool(cycle["large_pending"]),
        max_tile=int(np.max(board)),
        move_count=int(payload["move_count"]),
        game_over=bool(payload["game_over"]),
    )
    identity = {
        "board": board.tolist(),
        "preview": {
            "kind": preview_kind,
            "candidates": list(candidates),
        },
        "tile_cycle": {
            "small_counts": small_counts,
            "small_pos": state.small_pos,
            "small_seen_total": state.small_seen_total,
            "span_small_pos": state.span_small_pos,
            "large_pending": state.large_pending,
        },
        "move_count": state.move_count,
        "game_over": state.game_over,
    }
    return state, identity


def _source_policy_spec(replay: Mapping[str, Any]) -> str:
    observed = [
        str(replay[field])
        for field in POLICY_SPEC_FIELDS
        if replay.get(field) not in (None, "")
    ]
    root_policy = replay.get("root_policy")
    if root_policy not in (None, ""):
        observed.append(str(root_policy))
    unique = tuple(dict.fromkeys(observed))
    if len(unique) != 1:
        raise O6DataHold(f"Conflicting or missing policy specs: {unique}")
    return unique[0]


def _family_for_policy_spec(policy_spec: str) -> str:
    matches = [
        family
        for family, expected in FAMILY_POLICY_SPECS.items()
        if policy_spec == expected
    ]
    if len(matches) != 1:
        raise O6DataHold(f"Unrecognized O6 policy spec: {policy_spec}")
    return matches[0]


def _stream_ids(replay: Mapping[str, Any]) -> dict[str, int]:
    streams = replay.get("rng_streams")
    if not isinstance(streams, Mapping):
        raise O6DataHold("O6 source replay lacks split RNG streams")
    result = {
        "logical_seed": int(replay["seed"]),
        "deck_stream_id": int(streams["deck_stream_id"]),
        "slot_stream_id": int(streams["slot_stream_id"]),
        "policy_stream_id": int(streams["policy_stream_id"]),
    }
    return result


def _normal_start_identity(
    replay: Mapping[str, Any],
    *,
    first_state: Any,
    first_identity: Mapping[str, Any],
    streams: Mapping[str, int],
) -> dict[str, Any]:
    from threes_rl.sim import ThreesSim

    starter = replay.get("starter_tile")
    starter_tile = None if starter is None else int(starter)
    seed = int(replay["seed"])
    if replay.get("replay_origin") != "fresh":
        raise O6DataHold("O6 replay_origin is not fresh")
    if replay.get("root_origin") != "fresh":
        raise O6DataHold("O6 root_origin is not fresh")
    if int(replay.get("root_seed")) != seed:
        raise O6DataHold("O6 root seed differs from replay seed")
    if int(replay.get("root_frame_index")) != 0:
        raise O6DataHold("O6 root frame is not zero")
    if int(replay.get("root_move_count")) != 0:
        raise O6DataHold("O6 root move count is not zero")
    simulator = ThreesSim.from_stream_ids(
        deck_stream_id=int(streams["deck_stream_id"]),
        slot_stream_id=int(streams["slot_stream_id"]),
        starter_tile=starter_tile,
    )
    reset = simulator.reset()
    reset_identity = {
        "board": reset.board.tolist(),
        "preview": {
            "kind": reset.preview.kind,
            "candidates": list(reset.preview.candidates),
        },
        "tile_cycle": {
            "small_counts": reset.small_counts,
            "small_pos": reset.small_pos,
            "small_seen_total": reset.small_seen_total,
            "span_small_pos": reset.span_small_pos,
            "large_pending": reset.large_pending,
        },
        "move_count": reset.move_count,
        "game_over": reset.game_over,
    }
    if reset_identity != dict(first_identity):
        raise O6DataHold("O6 first frame does not reproduce split-stream reset")
    if first_state.move_count != 0 or first_state.game_over:
        raise O6DataHold("O6 first state is not a live reset")
    return {
        "seed": seed,
        "starter_tile": starter_tile,
        "ancestry": f"fresh:{seed}:{starter_tile}",
        "reset_identity_sha256": canonical_json_hash(reset_identity),
    }


def _protected_intersections(
    *,
    ancestry: str,
    family: str,
    replay_sha256: str,
    streams: Mapping[str, int],
    exclusion: Mapping[str, Any],
) -> dict[str, list[Any]]:
    identities = exclusion.get("identities", {})
    ancestry_aliases = {
        ancestry,
        f"{family}:{ancestry}",
    }
    intersections: dict[str, list[Any]] = {}
    for field in ("ancestry", "ancestry_id", "root", "root_cluster"):
        protected = {str(value) for value in identities.get(field, ())}
        overlap = sorted(ancestry_aliases & protected)
        if overlap:
            intersections[field] = overlap
    for field in ("source_replay_sha256", "replay_sha256"):
        if replay_sha256 in {
            str(value) for value in identities.get(field, ())
        }:
            intersections[field] = [replay_sha256]
    for field in STREAM_FIELDS:
        protected = {
            int(value)
            for value in identities.get(field, ())
            if str(value).lstrip("-").isdigit()
        }
        if int(streams[field]) in protected:
            intersections[field] = [int(streams[field])]
    return intersections


def _feature_domain_audit(
    *,
    state: Any,
    simulator: Any,
    starter_tile: int | None,
    pair: Any,
) -> dict[str, Any]:
    from threes_rl.o4_domain_safe_pair_option import (
        initial_lineage,
        option_features,
    )

    lineage = initial_lineage(pair.coordinates)
    board_before = state.board.copy()
    deck_before = json.dumps(
        simulator.deck_rng.bit_generator.state,
        sort_keys=True,
    )
    slot_before = json.dumps(
        simulator.slot_rng.bit_generator.state,
        sort_keys=True,
    )
    rows = []
    for action in simulator.legal_actions(state):
        tokens, globals_ = option_features(
            state,
            simulator,
            starter_tile=starter_tile,
            pair=pair,
            lineage=lineage,
            action=int(action),
        )
        rows.append(
            {
                "action": int(action),
                "tokens_shape": list(tokens.shape),
                "globals_shape": list(globals_.shape),
                "finite": bool(
                    np.isfinite(tokens).all() and np.isfinite(globals_).all()
                ),
                "bounded": bool(
                    np.all((tokens >= 0.0) & (tokens <= 1.0))
                    and np.all((globals_ >= 0.0) & (globals_ <= 1.0))
                ),
            }
        )
    unchanged = (
        np.array_equal(state.board, board_before)
        and json.dumps(
            simulator.deck_rng.bit_generator.state,
            sort_keys=True,
        )
        == deck_before
        and json.dumps(
            simulator.slot_rng.bit_generator.state,
            sort_keys=True,
        )
        == slot_before
    )
    checks = {
        "all_legal_actions_covered": len(rows)
        == len(simulator.legal_actions(state)),
        "all_features_finite": all(row["finite"] for row in rows),
        "all_normalized_features_bounded": all(
            row["bounded"] for row in rows
        ),
        "state_and_rng_unmutated": unchanged,
    }
    return {
        "action_rows": rows,
        "checks": checks,
        "passes": all(checks.values()),
    }


def candidate_rows_from_replay(
    replay: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    exclusion: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from threes_rl.o4_domain_safe_pair_option import (
        TRAIN_TARGETS,
        root_option_eligible,
        select_designated_pair,
    )
    from threes_rl.sim import ThreesSim

    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        raise O6DataHold("O6 replay has no frames")
    first = frames[0]
    last = frames[-1]
    if not isinstance(first, Mapping) or not isinstance(last, Mapping):
        raise O6DataHold("O6 replay frame structure is malformed")
    first_payload = first.get("state")
    last_payload = last.get("state")
    if not isinstance(first_payload, Mapping) or not isinstance(
        last_payload,
        Mapping,
    ):
        raise O6DataHold("O6 replay state payload is malformed")
    first_state, first_identity = whitelisted_sim_state(first_payload)
    last_state, _ = whitelisted_sim_state(last_payload)
    if not last_state.game_over:
        raise O6DataHold("O6 replay is not complete")
    streams = _stream_ids(replay)
    normal = _normal_start_identity(
        replay,
        first_state=first_state,
        first_identity=first_identity,
        streams=streams,
    )
    policy_spec = _source_policy_spec(replay)
    family = _family_for_policy_spec(policy_spec)
    if replay.get("root_policy") not in (None, policy_spec):
        raise O6DataHold("O6 root policy differs from source policy")
    intersections = _protected_intersections(
        ancestry=normal["ancestry"],
        family=family,
        replay_sha256=source_sha256,
        streams=streams,
        exclusion=exclusion,
    )
    if intersections:
        raise O6DataHold(
            f"O6 source intersects protected evidence: {intersections}"
        )
    simulator = ThreesSim.from_stream_ids(
        deck_stream_id=streams["deck_stream_id"],
        slot_stream_id=streams["slot_stream_id"],
        starter_tile=normal["starter_tile"],
    )
    candidates: list[dict[str, Any]] = []
    frames_checked = 0
    for fallback, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise O6DataHold("O6 frame is not an object")
        payload = frame.get("state")
        if not isinstance(payload, Mapping):
            raise O6DataHold("O6 frame state is not an object")
        state, identity = whitelisted_sim_state(payload)
        frame_index = int(frame.get("index", fallback))
        if state.game_over:
            continue
        frames_checked += 1
        for target in TRAIN_TARGETS:
            pair = select_designated_pair(
                state.board,
                normal["starter_tile"],
                requested_target=int(target),
                allowed_targets=TRAIN_TARGETS,
            )
            if pair is None:
                continue
            if not root_option_eligible(
                state,
                simulator,
                normal["starter_tile"],
                allowed_targets=(int(target),),
            ):
                continue
            feature_audit = _feature_domain_audit(
                state=state,
                simulator=simulator,
                starter_tile=normal["starter_tile"],
                pair=pair,
            )
            if not feature_audit["passes"]:
                raise O6IntegrityKill("O6 feature-domain audit failed")
            state_hash = canonical_json_hash(identity)
            alignment = (
                "aligned"
                if pair.same_row or pair.same_column
                else "unaligned"
            )
            row = {
                "ancestry": normal["ancestry"],
                "family": family,
                "policy_spec": policy_spec,
                "target": int(target),
                "alignment": alignment,
                "source_replay": _relative(source_path),
                "source_replay_sha256": source_sha256,
                "frame_index": frame_index,
                "state_hash": state_hash,
                "pair_coords": [
                    list(coordinate) for coordinate in pair.coordinates
                ],
                **streams,
                "current_state_fields_only": True,
                "recorded_action_read": False,
                "score_or_max_tile_read": False,
                "future_outcome_read": False,
            }
            row["selection_key"] = prep.candidate_selection_key(row)
            candidates.append(row)
    report = {
        "source_path": _relative(source_path),
        "source_sha256": source_sha256,
        "ancestry": normal["ancestry"],
        "family": family,
        "policy_spec": policy_spec,
        "frames_checked": frames_checked,
        "candidate_count": len(candidates),
        "reset_identity_sha256": normal["reset_identity_sha256"],
        "protected_intersections": intersections,
        "complete": True,
        "current_state_geometry_only": True,
        "final_score_read": False,
        "recorded_action_read": False,
        "policy_outcome_read": False,
    }
    return candidates, report


def scan_candidate_sources(
    inventory: Mapping[str, Any],
    exclusion: Mapping[str, Any],
) -> dict[str, Any]:
    all_candidates: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    protected_hash_skips: list[dict[str, str]] = []
    hygiene_losses: list[dict[str, str]] = []
    protected_replay_hashes = {
        str(value)
        for field in ("source_replay_sha256", "replay_sha256")
        for value in exclusion.get("identities", {}).get(field, ())
    }
    for row in inventory.get("rows", []):
        if row.get("classification") != "candidate_replay":
            continue
        if str(row["sha256"]) in protected_replay_hashes:
            protected_hash_skips.append(
                {
                    "path": str(row["path"]),
                    "sha256": str(row["sha256"]),
                }
            )
            continue
        path = _repo_path(str(row["path"]))
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(replay, Mapping):
                raise O6DataHold("candidate replay is not an object")
            candidates, report = candidate_rows_from_replay(
                replay,
                source_path=path,
                source_sha256=str(row["sha256"]),
                exclusion=exclusion,
            )
        except O6IntegrityKill:
            raise
        except Exception as error:
            hygiene_losses.append(
                {
                    "path": str(row["path"]),
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
            continue
        all_candidates.extend(candidates)
        source_reports.append(report)
    reports_by_ancestry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in source_reports:
        reports_by_ancestry[str(report["ancestry"])].append(report)
    chosen_sources: dict[str, dict[str, Any]] = {}
    source_conflicts: list[dict[str, Any]] = []
    for ancestry, reports in sorted(reports_by_ancestry.items()):
        families = {str(row["family"]) for row in reports}
        policies = {str(row["policy_spec"]) for row in reports}
        if len(families) != 1 or len(policies) != 1:
            source_conflicts.append(
                {
                    "ancestry": ancestry,
                    "families": sorted(families),
                    "policy_specs": sorted(policies),
                }
            )
            continue
        for row in reports:
            material = (
                f"O6-source-copy-v1|{ancestry}|"
                f"{row['source_sha256']}|{row['source_path']}"
            )
            row["source_copy_key"] = hashlib.sha256(
                material.encode("ascii")
            ).hexdigest()
        chosen_sources[ancestry] = min(
            reports,
            key=lambda row: row["source_copy_key"],
        )
    chosen_paths = {
        str(row["source_path"]) for row in chosen_sources.values()
    }
    candidates_from_chosen_sources = [
        row
        for row in all_candidates
        if str(row["source_replay"]) in chosen_paths
    ]
    deduped = prep.dedupe_one_candidate_per_ancestry(
        candidates_from_chosen_sources
    )
    family_counts = Counter(row["family"] for row in deduped)
    target_counts = Counter(int(row["target"]) for row in deduped)
    alignment_counts = Counter(row["alignment"] for row in deduped)
    checks = {
        "one_candidate_per_ancestry": len(deduped)
        == len({row["ancestry"] for row in deduped}),
        "only_frozen_families": set(family_counts).issubset(FAMILY_ORDER),
        "only_frozen_targets": set(target_counts).issubset(TARGET_ORDER),
        "only_frozen_alignment": set(alignment_counts).issubset(
            {"aligned", "unaligned"}
        ),
        "zero_source_policy_conflicts": not source_conflicts,
        "one_source_copy_per_ancestry": len(chosen_sources)
        == len(reports_by_ancestry),
        "forbidden_fields_unread": True,
    }
    return {
        "source_reports": source_reports,
        "source_count": len(source_reports),
        "source_ancestry_count": len(reports_by_ancestry),
        "chosen_source_count": len(chosen_sources),
        "chosen_sources": sorted(
            chosen_sources.values(),
            key=lambda row: row["source_copy_key"],
        ),
        "source_conflicts": source_conflicts,
        "protected_hash_skips": protected_hash_skips,
        "protected_hash_skip_count": len(protected_hash_skips),
        "hygiene_losses": hygiene_losses,
        "hygiene_loss_count": len(hygiene_losses),
        "raw_candidate_count": len(all_candidates),
        "chosen_source_candidate_count": len(
            candidates_from_chosen_sources
        ),
        "deduped_candidates": deduped,
        "deduped_candidate_count": len(deduped),
        "family_counts": dict(family_counts),
        "target_counts": dict(target_counts),
        "alignment_counts": dict(alignment_counts),
        "candidate_sha256": canonical_json_hash(deduped),
        "checks": checks,
        "passes": all(checks.values()),
        "protected_replay_bodies_parsed": False,
    }


def allocation_cell_quotas(untouched_n: int) -> list[dict[str, Any]]:
    contract = prep.role_matrix_contract(untouched_n)
    rows: list[dict[str, Any]] = []
    for role_index, role in enumerate(ROLE_ORDER):
        matrix = contract["matrices"][role]
        for family_index, family in enumerate(FAMILY_ORDER):
            for target_index, target in enumerate(TARGET_ORDER):
                count = int(matrix[family_index][target_index])
                aligned = count // 2
                if count % 2 and (
                    role_index + family_index + target_index
                ) % 2 == 0:
                    aligned += 1
                rows.append(
                    {
                        "role": role,
                        "family": family,
                        "target": int(target),
                        "alignment": "aligned",
                        "required": aligned,
                    }
                )
                rows.append(
                    {
                        "role": role,
                        "family": family,
                        "target": int(target),
                        "alignment": "unaligned",
                        "required": count - aligned,
                    }
                )
    if sum(row["required"] for row in rows) != sum(
        prep.ROLE_COUNTS_BY_UNTOUCHED_N[untouched_n].values()
    ):
        raise O6IntegrityKill("O6 aligned allocation quota total changed")
    return rows


def allocate_candidate_design(
    rows: Sequence[Mapping[str, Any]],
    *,
    untouched_n: int,
) -> dict[str, Any]:
    deduped = prep.dedupe_one_candidate_per_ancestry(rows)
    cells: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in deduped:
        row = dict(raw)
        key = (
            str(row["family"]),
            int(row["target"]),
            str(row["alignment"]),
        )
        cells[key].append(row)
    for values in cells.values():
        values.sort(key=lambda row: row["selection_key"])
    offsets: Counter[tuple[str, int, str]] = Counter()
    selected: list[dict[str, Any]] = []
    deficits: list[dict[str, Any]] = []
    quotas = allocation_cell_quotas(untouched_n)
    for quota in quotas:
        key = (
            quota["family"],
            quota["target"],
            quota["alignment"],
        )
        start = offsets[key]
        stop = start + int(quota["required"])
        picked = cells.get(key, [])[start:stop]
        if len(picked) != quota["required"]:
            deficits.append(
                {
                    **quota,
                    "available_remaining": max(
                        0,
                        len(cells.get(key, [])) - start,
                    ),
                }
            )
        selected.extend(
            {**row, "role": quota["role"]} for row in picked
        )
        offsets[key] = stop
    integrity = prep.partition_integrity(
        selected,
        expected_role_counts=prep.ROLE_COUNTS_BY_UNTOUCHED_N[untouched_n],
    )
    checks = {
        "zero_deficits": not deficits,
        "partition_integrity": integrity["passes"],
        "exact_selected_total": len(selected)
        == sum(prep.ROLE_COUNTS_BY_UNTOUCHED_N[untouched_n].values()),
        "ancestry_unique": len(selected)
        == len({row["ancestry"] for row in selected}),
    }
    return {
        "untouched_n": untouched_n,
        "quotas": quotas,
        "selected": selected,
        "deficits": deficits,
        "partition_integrity": integrity,
        "selection_sha256": canonical_json_hash(selected),
        "checks": checks,
        "passes": all(checks.values()),
    }


def stream_reservation_contract() -> dict[str, Any]:
    parent = prep.stream_window_contract()
    rows = [
        {
            "purpose": row["purpose"],
            "field": row["field"],
            "start": int(row["start"]),
            "stop_exclusive": int(row["stop_exclusive"]),
        }
        for row in parent["ranges"]
    ]
    return {
        "version": f"{VERSION}_stream_reservation",
        "ranges": rows,
        "range_count": len(rows),
        "ranges_sha256": canonical_json_hash(rows),
        "streams_consumed": 0,
        "passes": parent["passes"],
    }


def historical_stream_collision_audit(
    exclusion: Mapping[str, Any],
    reservation: Mapping[str, Any],
) -> dict[str, Any]:
    identities = exclusion.get("identities", {})
    collisions: list[dict[str, Any]] = []
    for row in reservation["ranges"]:
        field = str(row["field"])
        historical = identities.get(field, ())
        for raw in historical:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                raise O6DataHold(
                    f"Historical stream value is not integral: {field}"
                )
            if int(row["start"]) <= value < int(row["stop_exclusive"]):
                collisions.append(
                    {
                        "field": field,
                        "value": value,
                        "purpose": row["purpose"],
                    }
                )
    checks = {
        "sixteen_ranges": reservation["range_count"] == 16,
        "zero_historical_collisions": not collisions,
        "streams_not_consumed": reservation["streams_consumed"] == 0,
    }
    return {
        "historical_union_sha256": exclusion["union_sha256"],
        "reservation_sha256": reservation["ranges_sha256"],
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
    }


def stratified_bootstrap_indices(
    rng: np.random.Generator,
    strata: np.ndarray,
    count: int,
) -> np.ndarray:
    strata_array = np.asarray(strata, dtype=np.int64)
    if strata_array.ndim != 1 or strata_array.size == 0 or count <= 0:
        raise O6IntegrityKill("Invalid stratified bootstrap request")
    result = np.empty((int(count), strata_array.size), dtype=np.int64)
    for stratum in sorted(set(strata_array.tolist())):
        positions = np.flatnonzero(strata_array == stratum)
        if positions.size == 0:
            raise O6IntegrityKill("Empty power stratum")
        choices = rng.integers(
            0,
            positions.size,
            size=(int(count), positions.size),
        )
        result[:, positions] = positions[choices]
    return result


def power_strata_for_design(n: int) -> np.ndarray:
    if n not in prep.POWER_ROOT_COUNTS:
        raise O6IntegrityKill("Power N is outside the frozen grid")
    counts = Counter()
    for row in allocation_cell_quotas(n):
        if row["role"] != "untouched_mechanism":
            continue
        label = f"T{row['target']}_{row['alignment']}"
        if label not in prep.POWER_STRATA:
            raise O6IntegrityKill(f"Unknown O6 power stratum: {label}")
        counts[label] += int(row["required"])
    values: list[int] = []
    for index, label in enumerate(prep.POWER_STRATA):
        values.extend([index] * counts[label])
    result = np.asarray(values, dtype=np.int64)
    if (
        result.shape != (n,)
        or set(counts) != set(prep.POWER_STRATA)
        or sum(counts.values()) != n
    ):
        raise O6IntegrityKill("O6 power strata do not match allocation")
    return result


def vectorized_common_odds_ratio(
    control: np.ndarray,
    treatment: np.ndarray,
    strata: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    control_array = np.asarray(control, dtype=np.int8)
    treatment_array = np.asarray(treatment, dtype=np.int8)
    strata_array = np.asarray(strata, dtype=np.int64)
    index_array = np.asarray(indices, dtype=np.int64)
    if (
        control_array.shape != treatment_array.shape
        or control_array.ndim != 2
        or strata_array.shape != (control_array.shape[0],)
        or index_array.ndim != 2
        or index_array.shape[1] != control_array.shape[0]
        or np.any(index_array < 0)
        or np.any(index_array >= control_array.shape[0])
    ):
        raise O6IntegrityKill("Invalid vectorized common-OR shapes")
    sampled_control = control_array[index_array]
    sampled_treatment = treatment_array[index_array]
    numerator = np.zeros(index_array.shape[0], dtype=np.float64)
    denominator = np.zeros(index_array.shape[0], dtype=np.float64)
    for stratum in sorted(set(strata_array.tolist())):
        positions = np.flatnonzero(strata_array == stratum)
        a = sampled_treatment[:, positions, :].sum(axis=(1, 2)) + 0.5
        b = (
            sampled_treatment[:, positions, :].shape[1]
            * sampled_treatment.shape[2]
            - sampled_treatment[:, positions, :].sum(axis=(1, 2))
            + 0.5
        )
        c = sampled_control[:, positions, :].sum(axis=(1, 2)) + 0.5
        d = (
            sampled_control[:, positions, :].shape[1]
            * sampled_control.shape[2]
            - sampled_control[:, positions, :].sum(axis=(1, 2))
            + 0.5
        )
        total = a + b + c + d
        numerator += a * d / total
        denominator += b * c / total
    values = numerator / denominator
    if (
        not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or np.any(denominator <= 0.0)
    ):
        raise O6IntegrityKill("Vectorized common OR is invalid")
    return values


def simulate_power_dataset_batch(
    *,
    rng: np.random.Generator,
    n: int,
    true_or: float,
    rho: float,
    dataset_start: int,
    dataset_count: int = POWER_DATASET_BATCH_SIZE,
    bootstrap_count: int = prep.POWER_BOOTSTRAPS,
    bootstrap_batch_size: int = POWER_BOOTSTRAP_BATCH_SIZE,
) -> list[dict[str, Any]]:
    if n not in prep.POWER_ROOT_COUNTS:
        raise O6IntegrityKill("Power N is outside the frozen grid")
    if true_or not in prep.POWER_OR_GRID or rho not in prep.POWER_RHO_GRID:
        raise O6IntegrityKill("Power cell is outside the frozen grid")
    if dataset_count <= 0 or bootstrap_count <= 0:
        raise O6IntegrityKill("Power draw count must be positive")
    if bootstrap_count % bootstrap_batch_size:
        raise O6IntegrityKill("Bootstrap count must divide into exact batches")
    alpha, beta = prep.beta_parameters(prep.POWER_BASE_RATE, rho)
    probabilities = rng.beta(alpha, beta, size=(dataset_count, n))
    shifted = prep.odds_shift(probabilities, true_or)
    uniforms = rng.random(
        (dataset_count, n, prep.POWER_REPEATS_PER_ARM)
    )
    control = uniforms < probabilities[..., None]
    treatment = uniforms < shifted[..., None]
    strata = power_strata_for_design(n)
    rows: list[dict[str, Any]] = []
    identity_indices = np.arange(n, dtype=np.int64)[None, :]
    for offset in range(dataset_count):
        point = float(
            vectorized_common_odds_ratio(
                control[offset],
                treatment[offset],
                strata,
                identity_indices,
            )[0]
        )
        bootstrap_values = np.empty(bootstrap_count, dtype=np.float64)
        for start in range(0, bootstrap_count, bootstrap_batch_size):
            indices = stratified_bootstrap_indices(
                rng,
                strata,
                bootstrap_batch_size,
            )
            bootstrap_values[start : start + bootstrap_batch_size] = (
                vectorized_common_odds_ratio(
                    control[offset],
                    treatment[offset],
                    strata,
                    indices,
                )
            )
        lower = float(
            np.quantile(bootstrap_values, 0.025, method="linear")
        )
        if not math.isfinite(point) or not math.isfinite(lower):
            raise O6IntegrityKill("Nonfinite power result")
        rows.append(
            {
                "dataset_index": dataset_start + offset,
                "point_or": point,
                "lower95": lower,
                "passes": (
                    point >= prep.POWER_POINT_GATE
                    and lower > prep.POWER_LOWER_GATE
                ),
            }
        )
    return rows


def power_config_payload() -> dict[str, Any]:
    return {
        "version": f"{VERSION}_power",
        "root_counts": prep.POWER_ROOT_COUNTS,
        "or_grid": prep.POWER_OR_GRID,
        "rho_grid": prep.POWER_RHO_GRID,
        "datasets": prep.POWER_DATASETS,
        "bootstraps": prep.POWER_BOOTSTRAPS,
        "dataset_batch_size": POWER_DATASET_BATCH_SIZE,
        "bootstrap_batch_size": POWER_BOOTSTRAP_BATCH_SIZE,
        "repeats_per_arm": prep.POWER_REPEATS_PER_ARM,
        "base_rate": prep.POWER_BASE_RATE,
        "point_gate": prep.POWER_POINT_GATE,
        "lower_gate": prep.POWER_LOWER_GATE,
        "required_power": prep.POWER_REQUIRED,
        "quantile": "numpy-linear-0.025",
        "correction": "Haldane-Anscombe +0.5 every stratum cell",
        "root_bootstrap": "target/alignment-stratified whole-root",
        "stratum_counts_by_n": {
            str(n): dict(
                Counter(
                    prep.POWER_STRATA[int(index)]
                    for index in power_strata_for_design(n)
                )
            )
            for n in prep.POWER_ROOT_COUNTS
        },
        "rng": "numpy-PCG64 one generator per cell",
        "seed_function": "accepted O6 preparation power_seed",
    }


def _power_cell_key(n: int, true_or: float, rho: float) -> str:
    return f"N{int(n)}_OR{true_or:.2f}_ICC{rho:.2f}"


def initialize_power_database(
    path: Path,
    *,
    marker_sha256: str,
) -> None:
    config = power_config_payload()
    config_hash = canonical_json_hash(config)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS progress ("
            "cell_key TEXT PRIMARY KEY, n INTEGER NOT NULL, "
            "true_or REAL NOT NULL, rho REAL NOT NULL, "
            "next_dataset INTEGER NOT NULL, rng_state TEXT NOT NULL, "
            "batch_opened_wall REAL, active_seconds REAL NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            "cell_key TEXT NOT NULL, dataset_index INTEGER NOT NULL, "
            "point_or REAL NOT NULL, lower95 REAL NOT NULL, "
            "passes INTEGER NOT NULL, "
            "PRIMARY KEY(cell_key, dataset_index))"
        )
        expected = {
            "version": VERSION,
            "config_sha256": config_hash,
            "marker_sha256": marker_sha256,
        }
        observed = dict(
            connection.execute("SELECT key, value FROM metadata").fetchall()
        )
        if observed and observed != expected:
            raise O6IntegrityKill("O6 power database metadata drift")
        if not observed:
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES(?, ?)",
                sorted(expected.items()),
            )
        for n in prep.POWER_ROOT_COUNTS:
            for true_or in prep.POWER_OR_GRID:
                for rho in prep.POWER_RHO_GRID:
                    key = _power_cell_key(n, true_or, rho)
                    existing = connection.execute(
                        "SELECT cell_key FROM progress WHERE cell_key=?",
                        (key,),
                    ).fetchone()
                    if existing is None:
                        rng = np.random.Generator(
                            np.random.PCG64(
                                prep.power_seed(n, true_or, rho)
                            )
                        )
                        connection.execute(
                            "INSERT INTO progress("
                            "cell_key,n,true_or,rho,next_dataset,rng_state,"
                            "batch_opened_wall,active_seconds"
                            ") VALUES(?,?,?,?,?,?,NULL,0.0)",
                            (
                                key,
                                n,
                                true_or,
                                rho,
                                0,
                                json.dumps(
                                    rng.bit_generator.state,
                                    sort_keys=True,
                                ),
                            ),
                        )
        connection.commit()
    finally:
        connection.close()


def audit_power_database(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    expected = {
        _power_cell_key(n, true_or, rho): (n, true_or, rho)
        for n in prep.POWER_ROOT_COUNTS
        for true_or in prep.POWER_OR_GRID
        for rho in prep.POWER_RHO_GRID
    }
    progress_rows = connection.execute(
        "SELECT cell_key,n,true_or,rho,next_dataset,rng_state,"
        "batch_opened_wall,active_seconds FROM progress ORDER BY cell_key"
    ).fetchall()
    observed_keys = {str(row[0]) for row in progress_rows}
    result_keys = {
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT cell_key FROM results"
        ).fetchall()
    }
    identity_failures: list[str] = []
    progress_failures: list[str] = []
    result_failures: list[str] = []
    progress_by_key: dict[str, int] = {}

    for row in progress_rows:
        (
            key,
            n,
            true_or,
            rho,
            next_dataset,
            rng_state_json,
            batch_opened_wall,
            active_seconds,
        ) = row
        key = str(key)
        expected_identity = expected.get(key)
        if expected_identity is None or (
            int(n),
            float(true_or),
            float(rho),
        ) != expected_identity:
            identity_failures.append(key)
            continue
        next_dataset = int(next_dataset)
        progress_by_key[key] = next_dataset
        if (
            next_dataset < 0
            or next_dataset > prep.POWER_DATASETS
            or next_dataset % POWER_DATASET_BATCH_SIZE
        ):
            progress_failures.append(f"{key}:next_dataset")
        if not math.isfinite(float(active_seconds)) or float(
            active_seconds
        ) < 0.0:
            progress_failures.append(f"{key}:active_seconds")
        if batch_opened_wall is not None and not math.isfinite(
            float(batch_opened_wall)
        ):
            progress_failures.append(f"{key}:batch_opened_wall")
        try:
            rng_state = json.loads(str(rng_state_json))
            bit_generator = np.random.PCG64()
            bit_generator.state = rng_state
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            progress_failures.append(f"{key}:rng_state")

    aggregates = {
        str(row[0]): row[1:]
        for row in connection.execute(
            "SELECT cell_key,COUNT(*),MIN(dataset_index),"
            "MAX(dataset_index),SUM(dataset_index) "
            "FROM results GROUP BY cell_key"
        ).fetchall()
    }
    for key, next_dataset in progress_by_key.items():
        count, minimum, maximum, index_sum = aggregates.get(
            key,
            (0, None, None, None),
        )
        count = int(count)
        expected_sum = next_dataset * (next_dataset - 1) // 2
        if count != next_dataset:
            result_failures.append(f"{key}:count")
        if next_dataset == 0:
            if any(value is not None for value in (minimum, maximum, index_sum)):
                result_failures.append(f"{key}:empty_index_domain")
        elif (
            minimum is None
            or maximum is None
            or index_sum is None
            or int(minimum) != 0
            or int(maximum) != next_dataset - 1
            or int(index_sum) != expected_sum
        ):
            result_failures.append(f"{key}:index_domain")

    invalid_value_count = 0
    for dataset_index, point_or, lower95, passes in connection.execute(
        "SELECT dataset_index,point_or,lower95,passes FROM results"
    ):
        try:
            valid = (
                isinstance(dataset_index, int)
                and math.isfinite(float(point_or))
                and float(point_or) > 0.0
                and math.isfinite(float(lower95))
                and float(lower95) > 0.0
                and isinstance(passes, int)
                and passes in (0, 1)
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalid_value_count += 1

    checks = {
        "exact_sixty_progress_cells": (
            len(progress_rows) == 60
            and observed_keys == set(expected)
        ),
        "progress_identity_exact": not identity_failures,
        "progress_state_valid": not progress_failures,
        "result_cells_known": result_keys <= set(expected),
        "result_indices_match_progress": not result_failures,
        "result_values_finite_and_valid": invalid_value_count == 0,
    }
    return {
        "progress_row_count": len(progress_rows),
        "result_row_count": int(
            connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        ),
        "identity_failures": sorted(identity_failures),
        "progress_failures": sorted(progress_failures),
        "result_failures": sorted(result_failures),
        "unknown_result_cells": sorted(result_keys - set(expected)),
        "invalid_result_value_count": invalid_value_count,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _resume_pending_runtime(
    connection: sqlite3.Connection,
    cell_key: str,
) -> float:
    row = connection.execute(
        "SELECT batch_opened_wall, active_seconds "
        "FROM progress WHERE cell_key=?",
        (cell_key,),
    ).fetchone()
    if row is None:
        raise O6IntegrityKill("Missing O6 power progress row")
    opened, active = row
    if opened is None:
        return float(active)
    charged = float(active) + max(0.0, time.time() - float(opened))
    connection.execute(
        "UPDATE progress SET active_seconds=?, batch_opened_wall=NULL "
        "WHERE cell_key=?",
        (charged, cell_key),
    )
    connection.commit()
    return charged


def total_power_active_seconds(
    connection: sqlite3.Connection,
) -> float:
    now = time.time()
    total = 0.0
    for active, opened in connection.execute(
        "SELECT active_seconds,batch_opened_wall FROM progress"
    ):
        total += float(active)
        if opened is not None:
            total += max(0.0, now - float(opened))
    return total


def run_power_database(
    path: Path,
    *,
    marker_sha256: str,
    guard: Any,
) -> dict[str, Any]:
    initialize_power_database(path, marker_sha256=marker_sha256)
    connection = sqlite3.connect(path)
    try:
        initial_integrity = audit_power_database(connection)
        if not initial_integrity["passes"]:
            raise O6IntegrityKill("O6 power database resume integrity failed")
        for n in prep.POWER_ROOT_COUNTS:
            for true_or in prep.POWER_OR_GRID:
                for rho in prep.POWER_RHO_GRID:
                    key = _power_cell_key(n, true_or, rho)
                    active = _resume_pending_runtime(connection, key)
                    while True:
                        row = connection.execute(
                            "SELECT next_dataset,rng_state,active_seconds "
                            "FROM progress WHERE cell_key=?",
                            (key,),
                        ).fetchone()
                        if row is None:
                            raise O6IntegrityKill(
                                "Missing O6 power progress row"
                            )
                        next_dataset, rng_state_json, active = row
                        next_dataset = int(next_dataset)
                        if next_dataset == prep.POWER_DATASETS:
                            break
                        if (
                            next_dataset < 0
                            or next_dataset % POWER_DATASET_BATCH_SIZE
                            or next_dataset + POWER_DATASET_BATCH_SIZE
                            > prep.POWER_DATASETS
                        ):
                            raise O6IntegrityKill(
                                "O6 power progress index drift"
                            )
                        if (
                            total_power_active_seconds(connection)
                            > MAX_ACTIVE_HOURS * 3600.0
                        ):
                            raise O6DataHold(
                                "O6 power active runtime exceeded 36 hours"
                            )
                        guard()
                        opened = time.time()
                        connection.execute(
                            "UPDATE progress SET batch_opened_wall=? "
                            "WHERE cell_key=?",
                            (opened, key),
                        )
                        connection.commit()
                        bit_generator = np.random.PCG64()
                        bit_generator.state = json.loads(rng_state_json)
                        rng = np.random.Generator(bit_generator)
                        rows = simulate_power_dataset_batch(
                            rng=rng,
                            n=n,
                            true_or=true_or,
                            rho=rho,
                            dataset_start=next_dataset,
                        )
                        elapsed = max(0.0, time.time() - opened)
                        new_state = json.dumps(
                            rng.bit_generator.state,
                            sort_keys=True,
                        )
                        with connection:
                            connection.executemany(
                                "INSERT INTO results("
                                "cell_key,dataset_index,point_or,lower95,passes"
                                ") VALUES(?,?,?,?,?)",
                                [
                                    (
                                        key,
                                        row["dataset_index"],
                                        row["point_or"],
                                        row["lower95"],
                                        int(row["passes"]),
                                    )
                                    for row in rows
                                ],
                            )
                            connection.execute(
                                "UPDATE progress SET next_dataset=?,"
                                "rng_state=?,batch_opened_wall=NULL,"
                                "active_seconds=active_seconds+? "
                                "WHERE cell_key=?",
                                (
                                    next_dataset
                                    + POWER_DATASET_BATCH_SIZE,
                                    new_state,
                                    elapsed,
                                    key,
                                ),
                            )
                        if (
                            total_power_active_seconds(connection)
                            > MAX_ACTIVE_HOURS * 3600.0
                        ):
                            raise O6DataHold(
                                "O6 final power batch exceeded 36 hours"
                            )
        terminal_integrity = audit_power_database(connection)
        if not terminal_integrity["passes"]:
            raise O6IntegrityKill("O6 power database terminal integrity failed")
        return summarize_power_database(connection)
    finally:
        connection.close()


def summarize_power_database(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    integrity = audit_power_database(connection)
    rows: list[dict[str, Any]] = []
    for n in prep.POWER_ROOT_COUNTS:
        for true_or in prep.POWER_OR_GRID:
            for rho in prep.POWER_RHO_GRID:
                key = _power_cell_key(n, true_or, rho)
                count, passing = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(passes),0) "
                    "FROM results WHERE cell_key=?",
                    (key,),
                ).fetchone()
                rows.append(
                    {
                        "n": n,
                        "true_or": true_or,
                        "rho": rho,
                        "dataset_count": int(count),
                        "passing_count": int(passing),
                        "full_pass_power": (
                            int(passing) / prep.POWER_DATASETS
                            if int(count) == prep.POWER_DATASETS
                            else None
                        ),
                    }
                )
    total = sum(row["dataset_count"] for row in rows)
    complete = all(
        row["dataset_count"] == prep.POWER_DATASETS for row in rows
    )
    if complete:
        selection = prep.select_power_design(rows)
    else:
        selection = {
            "selected_n": None,
            "mde_grid_or": None,
            "ready": False,
        }
    checks = {
        "database_integrity": integrity["passes"],
        "sixty_cells": len(rows) == 60,
        "all_cells_complete": complete,
        "total_dataset_rows_exact": total == 245_760,
        "no_reduced_draw_count": all(
            row["dataset_count"] in (0, prep.POWER_DATASETS)
            or row["dataset_count"] % POWER_DATASET_BATCH_SIZE == 0
            for row in rows
        ),
    }
    return {
        "config": power_config_payload(),
        "config_sha256": canonical_json_hash(power_config_payload()),
        "rows": rows,
        "dataset_row_count": total,
        "total_active_seconds": total_power_active_seconds(connection),
        "selection": selection,
        "database_integrity": integrity,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _write_or_validate_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    expected = _payload_with_hash(payload, field)
    if path.exists():
        observed = _load_hashed_json(path, field=field)
        if observed != expected:
            raise O6IntegrityKill(f"Resumed O6 artifact differs: {path}")
        return {
            "path": str(path),
            "file_sha256": sha256_path(path),
            "payload_sha256": observed[field],
            "resumed": True,
        }
    artifact = _write_immutable_json(path, payload, field=field)
    return {**artifact, "resumed": False}


def _runtime_now_seconds(
    payload: Mapping[str, Any],
    *,
    now: float | None = None,
) -> float:
    total = float(payload["active_seconds"])
    opened = payload.get("phase_opened_wall")
    if opened is not None:
        current = time.time() if now is None else float(now)
        total += max(0.0, current - float(opened))
    return total


def begin_or_resume_runtime(
    *,
    output_dir: Path,
    marker: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    path = output_dir / RUNTIME_NAME
    now = time.time()
    if path.exists():
        payload = _load_hashed_json(
            path,
            field="runtime_payload_sha256",
        )
        if (
            payload.get("version") != f"{VERSION}_runtime"
            or payload.get("marker_payload_sha256")
            != marker["opened_payload_sha256"]
        ):
            raise O6IntegrityKill("O6 runtime journal identity changed")
        charged = _runtime_now_seconds(payload, now=now)
        payload["active_seconds"] = charged
        payload["resume_count"] = int(payload["resume_count"]) + 1
        payload["phase"] = str(phase)
        payload["phase_opened_wall"] = now
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        payload.pop("runtime_payload_sha256", None)
    else:
        payload = {
            "version": f"{VERSION}_runtime",
            "marker_payload_sha256": marker["opened_payload_sha256"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "active_seconds": 0.0,
            "resume_count": 0,
            "phase": str(phase),
            "phase_opened_wall": now,
            "phase_history": [],
        }
    if float(payload["active_seconds"]) > MAX_ACTIVE_HOURS * 3600.0:
        raise O6DataHold("O6 total execution runtime exceeded 36 hours")
    _write_mutable_json(
        path,
        payload,
        field="runtime_payload_sha256",
    )
    return _load_hashed_json(path, field="runtime_payload_sha256")


def checkpoint_runtime(
    *,
    output_dir: Path,
    marker: Mapping[str, Any],
    next_phase: str | None,
    enforce_cap: bool = True,
) -> dict[str, Any]:
    path = output_dir / RUNTIME_NAME
    payload = _load_hashed_json(
        path,
        field="runtime_payload_sha256",
    )
    if payload.get("marker_payload_sha256") != marker[
        "opened_payload_sha256"
    ]:
        raise O6IntegrityKill("O6 runtime marker binding changed")
    now = time.time()
    charged = _runtime_now_seconds(payload, now=now)
    history = list(payload.get("phase_history", []))
    history.append(
        {
            "phase": payload.get("phase"),
            "closed_active_seconds": charged,
        }
    )
    payload.update(
        {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "active_seconds": charged,
            "phase": next_phase,
            "phase_opened_wall": now if next_phase is not None else None,
            "phase_history": history,
        }
    )
    payload.pop("runtime_payload_sha256", None)
    _write_mutable_json(
        path,
        payload,
        field="runtime_payload_sha256",
    )
    observed = _load_hashed_json(path, field="runtime_payload_sha256")
    if (
        enforce_cap
        and _runtime_now_seconds(observed) > MAX_ACTIVE_HOURS * 3600.0
    ):
        raise O6DataHold("O6 total execution runtime exceeded 36 hours")
    return observed


def _execution_guard(
    *,
    output_dir: Path,
    marker: Mapping[str, Any],
) -> dict[str, Any]:
    if (output_dir / RESULT_NAME).exists():
        raise O6IntegrityKill("O6 terminal result already exists")
    current_marker = _load_marker(output_dir)
    if (
        current_marker["opened_payload_sha256"]
        != marker["opened_payload_sha256"]
    ):
        raise O6IntegrityKill("O6 marker changed during execution")
    audit = operational_audit(output_dir=output_dir)
    if not audit["passes"]:
        raise O6DataHold("O6 operational guard failed")
    runtime_path = output_dir / RUNTIME_NAME
    if runtime_path.exists():
        runtime = _load_hashed_json(
            runtime_path,
            field="runtime_payload_sha256",
        )
        if (
            runtime.get("marker_payload_sha256")
            != marker["opened_payload_sha256"]
        ):
            raise O6IntegrityKill("O6 runtime marker binding changed")
        if _runtime_now_seconds(runtime) > MAX_ACTIVE_HOURS * 3600.0:
            raise O6DataHold(
                "O6 total execution runtime exceeded 36 hours"
            )
    if directory_bytes(output_dir) + PROJECTED_OUTPUT_BYTES > MAX_OUTPUT_BYTES:
        raise O6DataHold("O6 projected output exceeds four GiB")
    return audit


def _terminal_result(
    *,
    output_dir: Path,
    marker: Mapping[str, Any],
    decision: str,
    stage: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_result",
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "stage": stage,
        "marker_file_sha256": sha256_path(output_dir / OPENED_NAME),
        "marker_payload_sha256": marker["opened_payload_sha256"],
        "details": dict(details),
        "forbidden_work": {
            "labels": 0,
            "training_steps": 0,
            "checkpoints": 0,
            "development_or_untouched_outcomes": 0,
            "policy_outcomes": 0,
            "incumbent_changes": 0,
            "dashboard_changes": 0,
            "human_session_reads": 0,
        },
        "promotion_eligible": False,
    }
    runtime_path = output_dir / RUNTIME_NAME
    if runtime_path.is_file():
        runtime = _load_hashed_json(
            runtime_path,
            field="runtime_payload_sha256",
        )
        payload["runtime"] = {
            "file_sha256": sha256_path(runtime_path),
            "payload_sha256": runtime["runtime_payload_sha256"],
            "active_seconds": runtime["active_seconds"],
            "resume_count": runtime["resume_count"],
            "phase": runtime["phase"],
        }
    return _write_immutable_json(
        output_dir / RESULT_NAME,
        payload,
        field="result_payload_sha256",
    )


def execute(
    *,
    output_dir: Path = _repo_path(OUTPUT_DIR),
    jobs: int = 1,
) -> dict[str, Any]:
    if jobs != 1:
        raise O6IntegrityKill("O6 execution jobs must equal one")
    marker = _load_marker(output_dir)
    marker_path = output_dir / OPENED_NAME
    with marker_path.open("rb") as marker_handle:
        try:
            fcntl.flock(
                marker_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise O6DataHold("Another O6 execution owns the marker") from error
        stage = "marker_validation"
        try:
            begin_or_resume_runtime(
                output_dir=output_dir,
                marker=marker,
                phase=stage,
            )
            operational = _execution_guard(
                output_dir=output_dir,
                marker=marker,
            )
            inventory = marker["byte_inventory"]
            current_inventory = compare_inventory_to_current(
                inventory,
                output_dir=output_dir,
            )
            if current_inventory["immutable_hash_drift"]:
                raise O6IntegrityKill(
                    "O6 immutable source changed after marker"
                )
            if not current_inventory["passes"]:
                raise O6DataHold(
                    "O6 source inventory changed after marker"
                )
            checkpoint_runtime(
                output_dir=output_dir,
                marker=marker,
                next_phase="protected_exclusion",
            )
            _write_or_validate_json(
                output_dir / PROTECTED_INVENTORY_NAME,
                {
                    "version": f"{VERSION}_protected_inventory",
                    "marker_payload_sha256": marker[
                        "opened_payload_sha256"
                    ],
                    "inventory": inventory,
                    "passes": validate_byte_inventory(inventory)["passes"],
                },
                field="protected_inventory_payload_sha256",
            )

            stage = "protected_exclusion"
            exclusion = build_exclusion_union(inventory)
            if not exclusion["passes"]:
                raise O6DataHold("O6 protected exclusion union failed")
            _write_or_validate_json(
                output_dir / EXCLUSION_UNION_NAME,
                exclusion,
                field="exclusion_union_payload_sha256",
            )

            stage = "candidate_scan"
            checkpoint_runtime(
                output_dir=output_dir,
                marker=marker,
                next_phase=stage,
            )
            scan = scan_candidate_sources(inventory, exclusion)
            if not scan["passes"]:
                raise O6IntegrityKill("O6 candidate scanner integrity failed")
            source_payload = {
                key: value
                for key, value in scan.items()
                if key not in {"deduped_candidates"}
            }
            _write_or_validate_json(
                output_dir / SOURCE_INVENTORY_NAME,
                source_payload,
                field="source_inventory_payload_sha256",
            )
            candidate_payload = {
                "version": f"{VERSION}_candidate_roots",
                "rows": scan["deduped_candidates"],
                "row_count": scan["deduped_candidate_count"],
                "rows_sha256": scan["candidate_sha256"],
                "one_root_per_ancestry": True,
                "passes": scan["passes"],
            }
            _write_or_validate_json(
                output_dir / CANDIDATE_ROOTS_NAME,
                candidate_payload,
                field="candidate_roots_payload_sha256",
            )

            stage = "allocation"
            checkpoint_runtime(
                output_dir=output_dir,
                marker=marker,
                next_phase=stage,
            )
            designs = {
                str(n): allocate_candidate_design(
                    scan["deduped_candidates"],
                    untouched_n=n,
                )
                for n in prep.POWER_ROOT_COUNTS
            }
            selected_payload = {
                "version": f"{VERSION}_selected_roots",
                "designs": designs,
                "ready_n": [
                    n
                    for n in prep.POWER_ROOT_COUNTS
                    if designs[str(n)]["passes"]
                ],
                "passes": any(
                    designs[str(n)]["passes"]
                    for n in prep.POWER_ROOT_COUNTS
                ),
            }
            _write_or_validate_json(
                output_dir / SELECTED_ROOTS_NAME,
                selected_payload,
                field="selected_roots_payload_sha256",
            )
            if not selected_payload["passes"]:
                raise O6DataHold(
                    "O6 natural source support cannot fill any frozen design"
                )

            stage = "stream_collision"
            checkpoint_runtime(
                output_dir=output_dir,
                marker=marker,
                next_phase=stage,
            )
            reservation = stream_reservation_contract()
            _write_or_validate_json(
                output_dir / STREAM_RESERVATION_NAME,
                reservation,
                field="stream_reservation_payload_sha256",
            )
            collision = historical_stream_collision_audit(
                exclusion,
                reservation,
            )
            _write_or_validate_json(
                output_dir / COLLISION_AUDIT_NAME,
                collision,
                field="collision_audit_payload_sha256",
            )
            if not collision["passes"]:
                raise O6DataHold("O6 frozen stream ranges collide")

            stage = "power"
            checkpoint_runtime(
                output_dir=output_dir,
                marker=marker,
                next_phase=stage,
            )
            marker_sha = sha256_path(marker_path)
            power = run_power_database(
                output_dir / POWER_DB_NAME,
                marker_sha256=marker_sha,
                guard=lambda: _execution_guard(
                    output_dir=output_dir,
                    marker=marker,
                ),
            )
            _write_or_validate_json(
                output_dir / POWER_TABLE_NAME,
                power,
                field="power_table_payload_sha256",
            )
            if not power["passes"]:
                raise O6IntegrityKill("O6 exact power execution incomplete")
            selected_n = power["selection"]["selected_n"]
            if selected_n is None:
                raise O6DataHold("O6 OR1.50 power has no passing N")
            if not designs[str(selected_n)]["passes"]:
                raise O6DataHold(
                    "O6 power-selected N lacks exact natural root support"
                )

            stage = "terminal_audit"
            checkpoint_runtime(
                output_dir=output_dir,
                marker=marker,
                next_phase=stage,
            )
            terminal_inventory = compare_inventory_to_current(
                inventory,
                output_dir=output_dir,
            )
            if terminal_inventory["immutable_hash_drift"]:
                raise O6IntegrityKill(
                    "O6 immutable source changed before terminal"
                )
            if not terminal_inventory["passes"]:
                raise O6DataHold(
                    "O6 source inventory changed before terminal"
                )
            operational = _execution_guard(
                output_dir=output_dir,
                marker=marker,
            )
            _write_or_validate_json(
                output_dir / OPERATIONAL_AUDIT_NAME,
                {
                    "version": f"{VERSION}_operational",
                    "audit": operational,
                    "projected_output_bytes": PROJECTED_OUTPUT_BYTES,
                    "actual_output_bytes": directory_bytes(output_dir),
                    "passes": operational["passes"],
                },
                field="operational_audit_payload_sha256",
            )
            checkpoint_runtime(
                output_dir=output_dir,
                marker=marker,
                next_phase=None,
            )
            return _terminal_result(
                output_dir=output_dir,
                marker=marker,
                decision="READY_O6_COMPETING_RISKS_P0",
                stage=stage,
                details={
                    "selected_n": selected_n,
                    "mde_grid_or": power["selection"]["mde_grid_or"],
                    "selected_roots_payload_sha256": sha256_path(
                        output_dir / SELECTED_ROOTS_NAME
                    ),
                    "power_table_file_sha256": sha256_path(
                        output_dir / POWER_TABLE_NAME
                    ),
                    "collision_audit_file_sha256": sha256_path(
                        output_dir / COLLISION_AUDIT_NAME
                    ),
                    "ready_authorizes_labels": False,
                },
            )
        except O6DataHold as error:
            try:
                checkpoint_runtime(
                    output_dir=output_dir,
                    marker=marker,
                    next_phase=None,
                    enforce_cap=False,
                )
            except Exception as runtime_error:
                return _terminal_result(
                    output_dir=output_dir,
                    marker=marker,
                    decision="KILL_O6_P0_INTEGRITY",
                    stage="runtime_terminalization",
                    details={
                        "error_type": type(runtime_error).__name__,
                        "error": str(runtime_error),
                        "prior_hold": str(error),
                    },
                )
            return _terminal_result(
                output_dir=output_dir,
                marker=marker,
                decision="HOLD_O6_DATA_PREFLIGHT",
                stage=stage,
                details={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except Exception as error:
            try:
                checkpoint_runtime(
                    output_dir=output_dir,
                    marker=marker,
                    next_phase=None,
                    enforce_cap=False,
                )
            except Exception:
                pass
            return _terminal_result(
                output_dir=output_dir,
                marker=marker,
                decision="KILL_O6_P0_INTEGRITY",
                stage=stage,
                details={
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
    )
    audit = subparsers.add_parser("audit-zero-work")
    audit.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)

    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence.add_argument("--focused", type=int, required=True)
    evidence.add_argument("--regressions", type=int, required=True)
    evidence.add_argument(
        "--deselection",
        action="append",
        default=[],
    )
    evidence.add_argument(
        "--recorded-command",
        action="append",
        default=[],
    )
    for name in ("seal-preflight", "open", "execute"):
        child = subparsers.add_parser(name)
        child.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
        if name in {"open", "execute"}:
            child.add_argument("--jobs", type=int, default=1)
    return parser


def dispatch(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    output_dir = _repo_path(args.out_dir)
    if output_dir.resolve() != _repo_path(OUTPUT_DIR).resolve():
        raise O6IntegrityKill("O6 execution output directory changed")
    if args.subcommand == "audit-zero-work":
        return zero_work_preflight(output_dir=output_dir)
    if args.subcommand == "write-test-evidence":
        return write_test_evidence(
            focused_passed=args.focused,
            regressions_passed=args.regressions,
            deselections=args.deselection,
            commands=args.recorded_command,
            output_dir=output_dir,
        )
    if args.subcommand == "seal-preflight":
        return seal_preflight(output_dir=output_dir)
    if args.subcommand == "open":
        return open_execution(output_dir=output_dir, jobs=args.jobs)
    if args.subcommand == "execute":
        return execute(output_dir=output_dir, jobs=args.jobs)
    raise O6IntegrityKill(f"Unknown O6 command: {args.subcommand}")


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(dispatch(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

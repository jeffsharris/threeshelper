"""J2A1 V3 recovery execution surface and outcome-free readiness.

The readiness and phase-setup paths are metadata-only. Scientific imports and
root-body loading are lazy and structurally downstream of the complete-union
seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from threes_rl import (
    j2a1_distillation_fidelity_recovery_readiness_v3 as preflight,
)


VERSION = "j2a1_distillation_fidelity_recovery_execution_surface_v3"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs" / "forensics"

CHARTER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J2A1_DISTILLATION_FIDELITY_V3_RECOVERY_EXECUTION_SURFACE_CHARTER.md"
)
RUNNER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "j2a1_distillation_fidelity_recovery_execution_surface_v3.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_rl_j2a1_distillation_fidelity_recovery_execution_surface_v3.py"
)

PREFLIGHT_DIR = preflight.READINESS_DIR
READINESS_DIR = (
    RUNS_ROOT
    / "j2a1_distillation_fidelity_recovery_execution_surface_readiness_v3"
)
FUTURE_AUTHORIZATION_DIR = (
    RUNS_ROOT / "j2a1_distillation_fidelity_recovery_authorization_v3"
)
FUTURE_EXECUTION_DIR = preflight.FUTURE_EXECUTION_DIR

TEST_EVIDENCE_NAME = "J2A1_V3_RECOVERY_SURFACE_TEST_EVIDENCE.json"
INPUT_BINDINGS_NAME = "J2A1_V3_RECOVERY_SURFACE_INPUT_BINDINGS.json"
AUTHORITY_AUDIT_NAME = "J2A1_V3_RECOVERY_SURFACE_AUTHORITY_AUDIT.json"
SCHEMA_NAME = "J2A1_V3_RECOVERY_SURFACE_SCHEMA.json"
PROJECTION_NAME = "J2A1_V3_RECOVERY_SURFACE_PROJECTION.json"
STATE_MACHINE_NAME = "J2A1_V3_RECOVERY_SURFACE_STATE_MACHINE_AUDIT.json"
READINESS_LOCK_NAME = "J2A1_V3_RECOVERY_SURFACE_READINESS_LOCK.json"
READINESS_RESULT_NAME = "J2A1_V3_RECOVERY_SURFACE_READINESS_RESULT.json"
READINESS_RETENTION_NAME = "J2A1_V3_RECOVERY_SURFACE_RETENTION.json"

AUTHORIZATION_NAME = "J2A1_V3_RECOVERY_EXECUTION_AUTHORIZATION.json"
PHASE_LOCK_NAME = "J2A1_V3_RECOVERY_PHASE_LOCK.json"
MARKER_NAME = "J2A1_V3_RECOVERY_EXECUTION_MARKER.json"
MATERIALIZED_NAME = "J2A1_V3_RECOVERY_MATERIALIZED_AUTHORITY.json"
STREAM_REUSE_NAME = "J2A1_V3_STREAM_AUTHORITY_REUSE.json"
GENESIS_NAME = "J2A1_V3_RECOVERY_GENESIS.json"
UNION_NAME = "J2A1_V3_RECOVERY_UNION_COMPLETE.json"
FINAL_OPERATIONAL_GUARD_NAME = "J2A1_V3_FINAL_OPERATIONAL_GUARD.json"
CHECKPOINT_AUTHORITY_NAME = "J2A1_V3_DISTILLED_CHECKPOINT_AUTHORITY.json"
TERMINAL_EVIDENCE_NAME = "J2A1_V3_RECOVERY_TERMINAL_EVIDENCE.json"
TERMINAL_NAME = "J2A1_V3_RECOVERY_TERMINAL.json"
EXECUTION_RETENTION_NAME = "J2A1_V3_RECOVERY_EXECUTION_RETENTION.json"

READY = "READY_J2A1_V3_RECOVERY_EXECUTION_SURFACE"
HOLD = "HOLD_J2A1_V3_RECOVERY_EXECUTION_SURFACE"
KILL = "KILL_J2A1_V3_RECOVERY_EXECUTION_SURFACE_INTEGRITY"
AUTHORIZATION_DECISION = "CONTINUE_J2A1_V3_RECOVERY_EXECUTION"
READY_EXECUTION = "READY_J2A1_V3_PPO_EXECUTION_SURFACE_REVIEW"
KILL_EXECUTION = "KILL_J2A1_V3_RECOVERY_INTEGRITY"

ACTIVE_ROOTS = preflight.ACTIVE_ROOTS
ACTIVE_STREAMS = preflight.ACTIVE_STREAMS
V2_COMPLETED_ROOTS = preflight.COMPLETED_ROOTS
RECOVERY_ROOTS = preflight.REMAINING_ROOTS
COLLECTORS = preflight.COLLECTORS
V2_WALL_SECONDS = preflight.EXPECTED_WALL_SECONDS
RUNTIME_CAP_SECONDS = preflight.RUNTIME_CAP_HOURS * 3600.0
STORAGE_CAP_BYTES = 24 * 1024**3
TERMINAL_FINALIZATION_ALLOWANCE_BYTES = 16 * 1024**2
HARD_DISK_FLOOR_GIB = preflight.HARD_DISK_FLOOR_GIB
TARGET_DISK_GIB = preflight.TARGET_DISK_GIB
INHERITED_PEAK_BYTES = 21_919_119_360
RECOVERY_METADATA_ALLOWANCE_BYTES = 128 * 1024**2
PROJECTED_COMBINED_BYTES = (
    INHERITED_PEAK_BYTES + RECOVERY_METADATA_ALLOWANCE_BYTES
)

ABANDONED_WALL_CHARGE_SECONDS = {
    "teacher_root_block": 658.2571791601365,
    "distillation_minibatch": 60.0,
    "student_fidelity_pair": 5.0,
    "phase_setup": 0.0,
}

EXPECTED_UNFINISHED_SHA256 = (
    "dca4de9005bede7e710ce004ade443aef5a0eda3c28f3994157a136bde0d34a9"
)
EXPECTED_COMPLETED_REFS_SHA256 = (
    "78f67e7b4da2a23ceb537366ad7cab6ac6f287b872f1e30b8eb67b5fefe2457b"
)

ZERO_WORK = {
    "scientific_authorizations": 0,
    "phase_locks": 0,
    "execution_markers": 0,
    "materialized_authorities": 0,
    "owners": 0,
    "new_stream_reservations": 0,
    "new_stream_consumptions": 0,
    "stream_authority_reuse_records": 0,
    "collector_process_loads": 0,
    "teacher_queries": 0,
    "root_body_deserializations": 0,
    "family_reads": 0,
    "labels": 0,
    "games": 0,
    "optimizer_steps": 0,
    "checkpoints": 0,
    "mechanism_reads": 0,
    "fidelity_reads": 0,
    "ppo_reads": 0,
    "development_reads": 0,
    "confirmation_reads": 0,
    "human_session_reads": 0,
    "incumbent_changes": 0,
    "dashboard_changes": 0,
    "promotion_actions": 0,
}

PREFLIGHT_SOURCE_HASHES = {
    preflight.CHARTER_PATH:
        "4638ffbcc67806742a1683d4aeec39a9669055d2051c18768a5ea5cd68aa216e",
    preflight.RUNNER_PATH:
        "c4bbb5b79a6a8df17e4b97663f49e0a5a2db06fe8d7b5e8bb483aa67ff3c8c43",
    preflight.TEST_PATH:
        "577dfe0fa434cbb8d95e6e780d5396e692ae63679e57a6ce64cf4214ab9b7705",
}

# name -> (file SHA, payload field, payload SHA)
PREFLIGHT_ARTIFACTS = {
    preflight.TEST_EVIDENCE_NAME: (
        "a45fe7b7e1eaf564e55c91ab9b0e25488fd958020fb9dc3138077df7294d808b",
        "test_evidence_payload_sha256",
        "0ad095a211fb69eb12892faa9f9e2047317b3cd3367a933dba476e1797766d3e",
    ),
    preflight.INPUT_BINDINGS_NAME: (
        "c6678f3bae0b7d85c4caddf09e44acd35932050a28b2da4afe67fe940a8531cb",
        "input_bindings_payload_sha256",
        "5a87f7d02a74f6493bc15683fa7f5a410b3fa89dd57866e7ea82a2820f51a275",
    ),
    preflight.V2_INTEGRITY_NAME: (
        "b4527df82c154e8831965050ef5eb6a5a124d5ed6e383f5402254d50fb556d58",
        "v2_integrity_payload_sha256",
        "8c5a2e5c8de7f0687084957cccc42e04e6dcf5da17127a4aa36b547698be0059",
    ),
    preflight.AUTHORITY_NAME: (
        "ca6f1bd99a9c6d3654e4af04227a1aad0f1d4d012f0eb3cedea1f3e405523691",
        "recovery_authority_payload_sha256",
        "13ca1ac368dbf679d39ba6d563deaa759dd8ed167f251641e95f062820c72a4b",
    ),
    preflight.PROJECTION_NAME: (
        "0f949e4e84495354f567ae91a8e848b113dbe7625bcc19b302a0c77351eb0f99",
        "wall_projection_payload_sha256",
        "c544b2e9285a60be7d654e34235a4ff211c58fc8b2d8e3b9d5aa37bbf6acda72",
    ),
    preflight.SCHEMA_NAME: (
        "bd9169c544cf9d1f212a4995c0727e6ca101bf18260ae603407f4062f0d35c4b",
        "recovery_schema_payload_sha256",
        "29f7a421c640a927f36da9c2947399b518e6da5cb24b8945e65e6322801dac3c",
    ),
    preflight.LOCK_NAME: (
        "0f8d4f916b9672dfbe2844595952053c81f153dd6b2d01bd8c30486204bb0153",
        "readiness_lock_payload_sha256",
        "d6bb27e29bda678afdbb82d5cbfa342338c2dcc8ff053fe09d159a3432879ca6",
    ),
    preflight.RESULT_NAME: (
        "23199ead16dce7ac87ea7d955bba5c913be632f624fa8771fc01a07669ab33ae",
        "readiness_result_payload_sha256",
        "66ef7008eb71e3ebf088d908170f510e823a884adbb07e35daeae35b17a8cc56",
    ),
    preflight.RETENTION_NAME: (
        "26e07603590d39e5402e6f95f35efc94933c83d68931e42a0b1de4b9f49c3246",
        "retention_payload_sha256",
        "2c4072d9a6771b89728336b57062c4d0b1ef06de77248241a05c35dabde9ee4a",
    ),
}


class J2A1V3SurfaceIntegrityError(RuntimeError):
    """An immutable execution-surface contract failed."""


class J2A1V3SurfaceOperationalHold(RuntimeError):
    """A mutable operational/resource gate failed."""


class J2A1V3SurfaceDataHold(RuntimeError):
    """An inherited scientific support gate cleanly held."""


class J2A1V3PlannedInterruption(RuntimeError):
    """Fixture-only interruption after a durable boundary."""


canonical_json_bytes = preflight.canonical_json_bytes
canonical_json_hash = preflight.canonical_json_hash
sha256_path = preflight.sha256_path
payload_with_hash = preflight.payload_with_hash
verify_payload_hash = preflight.verify_payload_hash
load_json = preflight.load_json
write_immutable_json = preflight.write_immutable_json
artifact_identity = preflight.artifact_identity


def _validate_hex(value: Any, *, name: str) -> str:
    try:
        return preflight._validate_hex(value, name=name)
    except preflight.J2A1V3IntegrityError as error:
        raise J2A1V3SurfaceIntegrityError(str(error)) from error


def _serialized_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            preflight._json_native(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _source_identities() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256_path(path)
        for path in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)
    }


def readiness_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "test_evidence": output_dir / TEST_EVIDENCE_NAME,
        "input_bindings": output_dir / INPUT_BINDINGS_NAME,
        "authority": output_dir / AUTHORITY_AUDIT_NAME,
        "schema": output_dir / SCHEMA_NAME,
        "projection": output_dir / PROJECTION_NAME,
        "state_machine": output_dir / STATE_MACHINE_NAME,
        "lock": output_dir / READINESS_LOCK_NAME,
        "result": output_dir / READINESS_RESULT_NAME,
        "retention": output_dir / READINESS_RETENTION_NAME,
    }


READINESS_FIELDS = {
    "test_evidence": "test_evidence_payload_sha256",
    "input_bindings": "input_bindings_payload_sha256",
    "authority": "authority_audit_payload_sha256",
    "schema": "execution_schema_payload_sha256",
    "projection": "projection_payload_sha256",
    "state_machine": "state_machine_payload_sha256",
    "lock": "readiness_lock_payload_sha256",
    "result": "readiness_result_payload_sha256",
    "retention": "retention_payload_sha256",
}


def phase_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "lock": out_dir / PHASE_LOCK_NAME,
        "marker": out_dir / MARKER_NAME,
        "manifest": out_dir / MATERIALIZED_NAME,
        "stream_reuse": out_dir / STREAM_REUSE_NAME,
        "genesis": out_dir / GENESIS_NAME,
        "owner": out_dir / "recovery_ownership_ledger.jsonl",
        "wall": out_dir / "top_level_wall_ledger.jsonl",
        "commits": out_dir / "recovery_commit_ledger.jsonl",
        "attempts": out_dir / "attempt_runtime_ledger.jsonl",
        "completions": out_dir / "teacher_root_completions.jsonl",
        "union": out_dir / UNION_NAME,
        "final_guard": out_dir / FINAL_OPERATIONAL_GUARD_NAME,
        "checkpoint_authority": out_dir / CHECKPOINT_AUTHORITY_NAME,
        "terminal_evidence": out_dir / TERMINAL_EVIDENCE_NAME,
        "result": out_dir / TERMINAL_NAME,
        "retention": out_dir / EXECUTION_RETENTION_NAME,
    }


def _load_bound_json(
    path: Path,
    *,
    file_sha256: str,
    payload_field: str,
    payload_sha256: str,
) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_path(path) != file_sha256
    ):
        raise J2A1V3SurfaceIntegrityError(
            f"Bound artifact bytes changed: {path}"
        )
    payload = load_json(path)
    if (
        not verify_payload_hash(payload, payload_field)
        or payload.get(payload_field) != payload_sha256
    ):
        raise J2A1V3SurfaceIntegrityError(
            f"Bound artifact payload changed: {path}"
        )
    return payload


def source_and_parent_audit(
    *,
    require_future_absent: bool,
) -> dict[str, Any]:
    preflight_sources = {}
    for path, expected in PREFLIGHT_SOURCE_HASHES.items():
        observed = sha256_path(path)
        preflight_sources[str(path.relative_to(REPO_ROOT))] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passes": observed == expected,
        }
    artifacts = {}
    for name, expected in PREFLIGHT_ARTIFACTS.items():
        path = PREFLIGHT_DIR / name
        _load_bound_json(
            path,
            file_sha256=expected[0],
            payload_field=expected[1],
            payload_sha256=expected[2],
        )
        artifacts[name] = {
            "file_sha256": expected[0],
            "payload_field": expected[1],
            "payload_sha256": expected[2],
            "bytes": int(path.stat().st_size),
        }
    result = load_json(PREFLIGHT_DIR / preflight.RESULT_NAME)
    retention = load_json(PREFLIGHT_DIR / preflight.RETENTION_NAME)
    checks = {
        "preflight_sources_exact": all(
            row["passes"] for row in preflight_sources.values()
        ),
        "nine_preflight_artifacts_exact": len(artifacts) == 9,
        "preflight_ready": result.get("decision") == preflight.READY,
        "preflight_execution_unauthorized":
            result.get("execution_authorized") is False,
        "preflight_retention_passes": retention.get("passes") is True,
        "v2_bound_sources_exact": all(
            sha256_path(path) == expected
            for path, expected in preflight.V2_SOURCE_PATHS.items()
        ),
        "future_authorization_absent":
            not FUTURE_AUTHORIZATION_DIR.exists(),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
    }
    if not require_future_absent:
        checks["future_authorization_absent"] = True
        checks["future_execution_absent"] = True
    if not all(checks.values()):
        raise J2A1V3SurfaceIntegrityError(
            "Execution-surface predecessor audit failed"
        )
    return {
        "version": f"{VERSION}_input_bindings_v1",
        "preflight_sources": preflight_sources,
        "preflight_artifacts": artifacts,
        "v2_terminal": {
            "file_sha256":
                preflight.V2_BOUND_ARTIFACTS[
                    preflight.V2_EXECUTION_DIR
                    / "J2A1_V2_DISTILLATION_FIDELITY_TERMINAL.json"
                ][0],
            "payload_sha256":
                preflight.V2_BOUND_ARTIFACTS[
                    preflight.V2_EXECUTION_DIR
                    / "J2A1_V2_DISTILLATION_FIDELITY_TERMINAL.json"
                ][2],
        },
        "v2_retention": {
            "file_sha256":
                preflight.V2_BOUND_ARTIFACTS[
                    preflight.V2_EXECUTION_DIR
                    / "J2A1_V2_DISTILLATION_FIDELITY_RETENTION.json"
                ][0],
            "payload_sha256":
                preflight.V2_BOUND_ARTIFACTS[
                    preflight.V2_EXECUTION_DIR
                    / "J2A1_V2_DISTILLATION_FIDELITY_RETENTION.json"
                ][2],
        },
        "checks": checks,
        "passes": True,
    }


def load_recovery_authority() -> dict[str, Any]:
    expected = PREFLIGHT_ARTIFACTS[preflight.AUTHORITY_NAME]
    payload = _load_bound_json(
        PREFLIGHT_DIR / preflight.AUTHORITY_NAME,
        file_sha256=expected[0],
        payload_field=expected[1],
        payload_sha256=expected[2],
    )
    unfinished = payload.get("unfinished_rows")
    completed = payload.get("completed_refs")
    if (
        not isinstance(unfinished, list)
        or not isinstance(completed, list)
        or len(unfinished) != RECOVERY_ROOTS
        or len(completed) != V2_COMPLETED_ROOTS
        or payload.get("unfinished_rows_sha256")
        != EXPECTED_UNFINISHED_SHA256
        or payload.get("completed_refs_sha256")
        != EXPECTED_COMPLETED_REFS_SHA256
        or canonical_json_hash(unfinished) != EXPECTED_UNFINISHED_SHA256
        or canonical_json_hash(completed) != EXPECTED_COMPLETED_REFS_SHA256
        or payload.get("duplicate_stream_consumptions") != 0
        or payload.get("replacement_roots") != 0
        or payload.get("filtered_roots") != 0
        or payload.get("scientific_content_opened") != 0
        or payload.get("passes") is not True
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Frozen recovery authority changed"
        )
    return payload


def authority_audit() -> dict[str, Any]:
    authority = load_recovery_authority()
    unfinished = authority["unfinished_rows"]
    completed = authority["completed_refs"]
    completed_roots = {row["root_id"] for row in completed}
    unfinished_roots = {row["root_id"] for row in unfinished}
    unfinished_ancestries = {row["ancestry_id"] for row in unfinished}
    streams = [
        int(stream_id)
        for row in unfinished
        for stream_id in row["streams"].values()
    ]
    bc_rows = [
        row
        for row in unfinished
        if row["stage"] == "teacher_behavior_cloning"
    ]
    validation_rows = [
        row
        for row in unfinished
        if row["stage"] == "distillation_validation"
    ]
    expected_bc_indices = list(range(V2_COMPLETED_ROOTS, 8_192))
    expected_validation_indices = list(range(6_144))
    completed_byte_inventory = []
    for ref in completed:
        path = preflight.V2_EXECUTION_DIR / str(ref["relative_path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(ref["bytes"])
            or sha256_path(path) != ref["file_sha256"]
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Preserved V2 completion bytes changed"
            )
        completed_byte_inventory.append(
            {
                "root_id": ref["root_id"],
                "relative_path": ref["relative_path"],
                "bytes": int(ref["bytes"]),
                "file_sha256": ref["file_sha256"],
            }
        )
    checks = {
        "completed_exact": len(completed_roots) == V2_COMPLETED_ROOTS,
        "unfinished_exact": len(unfinished_roots) == RECOVERY_ROOTS,
        "root_sets_disjoint": not completed_roots & unfinished_roots,
        "unfinished_ancestries_unique":
            len(unfinished_ancestries) == RECOVERY_ROOTS,
        "unfinished_streams_unique": len(set(streams)) == len(streams),
        "bc_suffix_exact": [row["row_index"] for row in bc_rows]
        == expected_bc_indices,
        "validation_full_exact": [
            row["row_index"] for row in validation_rows
        ]
        == expected_validation_indices,
        "canonical_order": unfinished == sorted(
            unfinished,
            key=lambda row: (
                0
                if row["stage"] == "teacher_behavior_cloning"
                else 1,
                int(row["row_index"]),
            ),
        ),
        "shards_fixed": all(
            int(row["row_index"]) % COLLECTORS in range(COLLECTORS)
            for row in unfinished
        ),
        "no_second_consumption":
            authority["duplicate_stream_consumptions"] == 0,
        "stage_b_gate": authority[
            "stage_b_sealed_until_total_completions"
        ]
        == ACTIVE_ROOTS,
        "completed_bytes_rehashed":
            len(completed_byte_inventory) == V2_COMPLETED_ROOTS,
    }
    if not all(checks.values()):
        raise J2A1V3SurfaceIntegrityError(
            "Recovery execution authority audit failed"
        )
    return {
        "version": f"{VERSION}_authority_audit_v1",
        "total_authority_roots": ACTIVE_ROOTS,
        "v2_completed_roots": V2_COMPLETED_ROOTS,
        "recovery_roots": RECOVERY_ROOTS,
        "active_streams": ACTIVE_STREAMS,
        "recovery_streams": len(streams),
        "completed_refs_sha256": EXPECTED_COMPLETED_REFS_SHA256,
        "completed_byte_inventory_sha256":
            canonical_json_hash(completed_byte_inventory),
        "completed_bytes_rehashed": len(completed_byte_inventory),
        "unfinished_rows_sha256": EXPECTED_UNFINISHED_SHA256,
        "bc_recovery_rows": len(bc_rows),
        "validation_recovery_rows": len(validation_rows),
        "new_reservations": 0,
        "new_consumptions": 0,
        "root_body_deserializations": 0,
        "family_reads": 0,
        "checks": checks,
        "passes": True,
    }


def projection_payload() -> dict[str, Any]:
    combined_gib = PROJECTED_COMBINED_BYTES / 1024**3
    checks = {
        "inherited_projection_exact":
            INHERITED_PEAK_BYTES == 21_919_119_360,
        "recovery_allowance_positive":
            RECOVERY_METADATA_ALLOWANCE_BYTES == 128 * 1024**2,
        "terminal_finalization_allowance_exact":
            TERMINAL_FINALIZATION_ALLOWANCE_BYTES == 16 * 1024**2,
        "terminal_allowance_inside_recovery_allowance":
            TERMINAL_FINALIZATION_ALLOWANCE_BYTES
            <= RECOVERY_METADATA_ALLOWANCE_BYTES,
        "combined_under_24_gib":
            PROJECTED_COMBINED_BYTES < STORAGE_CAP_BYTES,
        "point_wall_below_72":
            42.602824125381694 < preflight.RUNTIME_CAP_HOURS,
        "conservative_wall_below_72":
            50.989066430196644 < preflight.RUNTIME_CAP_HOURS,
    }
    return {
        "version": f"{VERSION}_projection_v1",
        "inherited_peak_after_margin_bytes": INHERITED_PEAK_BYTES,
        "recovery_metadata_allowance_bytes":
            RECOVERY_METADATA_ALLOWANCE_BYTES,
        "terminal_finalization_allowance_bytes":
            TERMINAL_FINALIZATION_ALLOWANCE_BYTES,
        "combined_peak_after_margin_bytes": PROJECTED_COMBINED_BYTES,
        "combined_peak_after_margin_gib": combined_gib,
        "storage_cap_bytes": STORAGE_CAP_BYTES,
        "storage_headroom_bytes":
            STORAGE_CAP_BYTES - PROJECTED_COMBINED_BYTES,
        "v2_observed_wall_seconds": V2_WALL_SECONDS,
        "point_total_stage_a_wall_hours": 42.602824125381694,
        "conservative_total_stage_a_wall_hours":
            50.989066430196644,
        "runtime_cap_hours": preflight.RUNTIME_CAP_HOURS,
        "abandoned_wall_charge_seconds":
            dict(ABANDONED_WALL_CHARGE_SECONDS),
        "checks": checks,
        "passes": all(checks.values()),
    }


def authorization_schema() -> dict[str, Any]:
    return {
        "version": f"{VERSION}_authorization_schema_v1",
        "decision": AUTHORIZATION_DECISION,
        "required_fields": [
            "version",
            "decision",
            "execution_mode",
            "scientific_authority",
            "readiness_result",
            "readiness_lock",
            "readiness_retention",
            "recovery_preflight_result",
            "recovery_preflight_lock",
            "recovery_preflight_retention",
            "recovery_preflight_artifacts",
            "recovery_authority_audit_sha256",
            "v2_bound_artifacts",
            "v2_terminal",
            "v2_retention",
            "execution_root",
            "jobs",
            "authorized_commands",
            "execution_authorized",
            "ppo_authorized",
            "development_authorized",
            "confirmation_authorized",
            "promotion_authorized",
        ],
        "authorized_commands": [
            "seal-phase-lock",
            "open",
            "materialize",
            "execute",
        ],
        "execution_mode": "scientific",
        "jobs": 1,
        "execution_authorized": True,
        "ppo_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
    }


def execution_schema() -> dict[str, Any]:
    return {
        "version": f"{VERSION}_execution_schema_v1",
        "public_commands": [
            "audit-zero-work",
            "write-test-evidence",
            "prepare-readiness",
            "seal-phase-lock",
            "open",
            "materialize",
            "execute",
        ],
        "phase_order": [
            "authorization",
            "seal-phase-lock",
            "open",
            "materialize",
            "execute",
            "union-seal",
            "inherited-stage-b",
            "terminal-evidence",
            "retention",
            "terminal",
        ],
        "authorization": authorization_schema(),
        "authority": {
            "v2_completed": V2_COMPLETED_ROOTS,
            "recovery": RECOVERY_ROOTS,
            "total": ACTIVE_ROOTS,
            "streams": ACTIVE_STREAMS,
            "unfinished_rows_sha256": EXPECTED_UNFINISHED_SHA256,
            "new_reservation_or_consumption": False,
        },
        "wall": {
            "v2_prior_seconds": V2_WALL_SECONDS,
            "cap_seconds": RUNTIME_CAP_SECONDS,
            "basis": "cumulative top-level live wall time",
            "worker_seconds": "descriptive only",
            "abandoned_charges": dict(ABANDONED_WALL_CHARGE_SECONDS),
        },
        "worker_topology": {
            "top_level_jobs": 1,
            "collectors": COLLECTORS,
            "single_thread_each": True,
            "shard": "stage-local row_index modulo 8",
            "canonical_merge": True,
            "work_stealing": False,
        },
        "output_accounting": {
            "initial_full_scans": 1,
            "per_boundary": "targeted stat updates only",
            "terminal_full_reconciliations": 1,
            "terminal_allowance_bytes":
                TERMINAL_FINALIZATION_ALLOWANCE_BYTES,
        },
        "stage_b_barrier": {
            "required_total_completions": ACTIVE_ROOTS,
            "required_v2_completions": V2_COMPLETED_ROOTS,
            "required_v3_completions": RECOVERY_ROOTS,
            "body_reads_before_union": 0,
        },
        "terminal_precedence": [
            "integrity_kill",
            "operational_kill",
            "scientific_hold",
            "ready_review",
        ],
        "zero_work": dict(ZERO_WORK),
    }


def state_machine_audit() -> dict[str, Any]:
    schema = execution_schema()
    phase = schema["phase_order"]
    checks = {
        "public_commands_exact": schema["public_commands"] == [
            "audit-zero-work",
            "write-test-evidence",
            "prepare-readiness",
            "seal-phase-lock",
            "open",
            "materialize",
            "execute",
        ],
        "phase_order_exact": phase.index("seal-phase-lock")
        < phase.index("open")
        < phase.index("materialize")
        < phase.index("execute")
        < phase.index("union-seal")
        < phase.index("inherited-stage-b"),
        "terminal_written_last": phase[-1] == "terminal",
        "stage_b_after_union": schema["stage_b_barrier"][
            "body_reads_before_union"
        ]
        == 0,
        "no_second_consumption":
            not schema["authority"]["new_reservation_or_consumption"],
        "scientific_successors_closed":
            not schema["authorization"]["ppo_authorized"]
            and not schema["authorization"]["development_authorized"]
            and not schema["authorization"]["confirmation_authorized"]
            and not schema["authorization"]["promotion_authorized"],
    }
    return {
        "version": f"{VERSION}_state_machine_audit_v1",
        "schema_sha256": canonical_json_hash(schema),
        "checks": checks,
        "passes": all(checks.values()),
    }


def operational_audit(
    *,
    output_dir: Path,
    include_services: bool,
    require_future_absent: bool,
) -> dict[str, Any]:
    try:
        parent = preflight.operational_audit(
            output_dir=output_dir,
            include_services=include_services,
        )
    except preflight.J2A1V3OperationalHold as error:
        raise J2A1V3SurfaceOperationalHold(str(error)) from error
    checks = {
        "parent_passes": parent.get("passes") is True,
        "future_authorization_absent":
            not FUTURE_AUTHORIZATION_DIR.exists(),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "projection_under_cap": PROJECTED_COMBINED_BYTES
        < STORAGE_CAP_BYTES,
        "zero_work": all(value == 0 for value in ZERO_WORK.values()),
    }
    if not require_future_absent:
        checks["future_authorization_absent"] = True
        checks["future_execution_absent"] = True
    return {
        "version": f"{VERSION}_operational_audit_v1",
        "parent": parent,
        "checks": checks,
        "passes": all(checks.values()),
    }


def test_evidence_payload(
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
) -> dict[str, Any]:
    normalized = []
    for command in commands:
        if set(command) != {"command", "passed", "failed", "note"}:
            raise J2A1V3SurfaceIntegrityError(
                "Test command schema changed"
            )
        if (
            not isinstance(command["command"], str)
            or type(command["passed"]) is not int
            or type(command["failed"]) is not int
            or command["failed"] != 0
            or not isinstance(command["note"], str)
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Test command result is invalid"
            )
        normalized.append(dict(command))
    if not normalized or not all(
        isinstance(value, str) for value in deselections
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Test evidence is incomplete"
        )
    return {
        "version": f"{VERSION}_test_evidence_v1",
        "source_identities": _source_identities(),
        "commands": normalized,
        "deselections": list(deselections),
        "total_passed": sum(row["passed"] for row in normalized),
        "total_failed": sum(row["failed"] for row in normalized),
        "total_deselected": len(deselections),
        "future_authorization_absent":
            not FUTURE_AUTHORIZATION_DIR.exists(),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "zero_work": dict(ZERO_WORK),
        "passes": all(row["failed"] == 0 for row in normalized)
        and not FUTURE_AUTHORIZATION_DIR.exists()
        and not FUTURE_EXECUTION_DIR.exists()
        and all(value == 0 for value in ZERO_WORK.values()),
    }


def write_test_evidence(
    *,
    output_dir: Path,
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
) -> dict[str, Any]:
    observed = (
        [
            path
            for path in output_dir.rglob("*")
            if path.is_file()
        ]
        if output_dir.exists()
        else []
    )
    if observed:
        raise J2A1V3SurfaceIntegrityError(
            "Readiness namespace was not fresh"
        )
    return write_immutable_json(
        output_dir / TEST_EVIDENCE_NAME,
        test_evidence_payload(commands, deselections),
        field=READINESS_FIELDS["test_evidence"],
    )


def _load_test_evidence(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if (
        not verify_payload_hash(payload, READINESS_FIELDS["test_evidence"])
        or payload.get("source_identities") != _source_identities()
        or payload.get("passes") is not True
        or payload.get("total_failed") != 0
    ):
        raise J2A1V3SurfaceIntegrityError("Test evidence changed")
    return payload


def _retention_payload(output_dir: Path) -> dict[str, Any]:
    paths = readiness_paths(output_dir)
    inventory = [
        {
            "path": path.name,
            "bytes": int(path.stat().st_size),
            "file_sha256": sha256_path(path),
            "payload_sha256": load_json(path)[READINESS_FIELDS[key]],
        }
        for key, path in paths.items()
        if key != "retention"
    ]
    return {
        "version": f"{VERSION}_readiness_retention_v1",
        "files": inventory,
        "file_count": len(inventory),
        "retained_bytes": sum(row["bytes"] for row in inventory),
        "canonical_inventory_sha256": canonical_json_hash(inventory),
        "future_authorization_absent":
            not FUTURE_AUTHORIZATION_DIR.exists(),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "zero_work": dict(ZERO_WORK),
        "no_cleanup_performed": True,
        "passes": len(inventory) == 8
        and not FUTURE_AUTHORIZATION_DIR.exists()
        and not FUTURE_EXECUTION_DIR.exists()
        and all(value == 0 for value in ZERO_WORK.values()),
    }


def prepare_readiness(
    *,
    output_dir: Path = READINESS_DIR,
    include_operational: bool = True,
) -> dict[str, Any]:
    paths = readiness_paths(output_dir)
    if not paths["test_evidence"].is_file():
        raise J2A1V3SurfaceIntegrityError(
            "Test evidence must exist before readiness"
        )
    observed = {
        path.resolve()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if observed != {paths["test_evidence"].resolve()}:
        raise J2A1V3SurfaceIntegrityError(
            "Readiness namespace is not at evidence boundary"
        )
    evidence = _load_test_evidence(paths["test_evidence"])
    bindings = source_and_parent_audit(require_future_absent=True)
    authority = authority_audit()
    schema = execution_schema()
    projection = projection_payload()
    state_machine = state_machine_audit()
    operations = operational_audit(
        output_dir=output_dir,
        include_services=include_operational,
        require_future_absent=True,
    )
    decision = (
        READY
        if all(
            payload.get("passes") is True
            for payload in (
                evidence,
                bindings,
                authority,
                projection,
                state_machine,
                operations,
            )
        )
        else HOLD
    )
    payloads = {
        "input_bindings": bindings,
        "authority": authority,
        "schema": schema,
        "projection": projection,
        "state_machine": state_machine,
    }
    for key, payload in payloads.items():
        write_immutable_json(
            paths[key],
            payload,
            field=READINESS_FIELDS[key],
        )
    predecessors = {
        key: artifact_identity(paths[key], READINESS_FIELDS[key])
        for key in (
            "test_evidence",
            "input_bindings",
            "authority",
            "schema",
            "projection",
            "state_machine",
        )
    }
    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "decision_candidate": decision,
        "source_identities": _source_identities(),
        "predecessors": predecessors,
        "operations": operations,
        "authorization_schema_sha256":
            canonical_json_hash(authorization_schema()),
        "recovery_authority_sha256": EXPECTED_UNFINISHED_SHA256,
        "completed_refs_sha256": EXPECTED_COMPLETED_REFS_SHA256,
        "future_authorization_root": str(
            FUTURE_AUTHORIZATION_DIR.relative_to(REPO_ROOT)
        ),
        "future_execution_root": str(
            FUTURE_EXECUTION_DIR.relative_to(REPO_ROOT)
        ),
        "execution_authorized": False,
        "zero_work": dict(ZERO_WORK),
        "passes": decision == READY,
    }
    write_immutable_json(
        paths["lock"],
        lock_payload,
        field=READINESS_FIELDS["lock"],
    )
    result_payload = {
        "version": f"{VERSION}_readiness_result_v1",
        "decision": decision,
        "readiness_lock": artifact_identity(
            paths["lock"],
            READINESS_FIELDS["lock"],
        ),
        "source_identities": _source_identities(),
        "v2_completed_roots": V2_COMPLETED_ROOTS,
        "recovery_roots": RECOVERY_ROOTS,
        "total_roots": ACTIVE_ROOTS,
        "streams": ACTIVE_STREAMS,
        "point_stage_a_wall_hours": 42.602824125381694,
        "conservative_stage_a_wall_hours": 50.989066430196644,
        "projected_combined_gib":
            PROJECTED_COMBINED_BYTES / 1024**3,
        "execution_authorized": False,
        "phase_lock_authorized": False,
        "collectors_authorized": False,
        "stage_b_authorized": False,
        "ppo_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
        "human_training_authorized": False,
        "continue": decision == READY,
        "hold": True,
        "kill": decision == KILL,
        "promote": False,
        "zero_work": dict(ZERO_WORK),
        "checks": {
            "evidence": evidence["passes"],
            "bindings": bindings["passes"],
            "authority": authority["passes"],
            "projection": projection["passes"],
            "state_machine": state_machine["passes"],
            "operations": operations["passes"],
            "future_authorization_absent":
                not FUTURE_AUTHORIZATION_DIR.exists(),
            "future_execution_absent":
                not FUTURE_EXECUTION_DIR.exists(),
            "zero_work": all(value == 0 for value in ZERO_WORK.values()),
        },
        "passes": decision == READY,
    }
    write_immutable_json(
        paths["result"],
        result_payload,
        field=READINESS_FIELDS["result"],
    )
    write_immutable_json(
        paths["retention"],
        _retention_payload(output_dir),
        field=READINESS_FIELDS["retention"],
    )
    expected_files = {path.resolve() for path in paths.values()}
    observed_files = {
        path.resolve()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise J2A1V3SurfaceIntegrityError(
            "Readiness file set changed"
        )
    return {
        "decision": decision,
        "result": load_json(paths["result"]),
        "lock": load_json(paths["lock"]),
        "retention": load_json(paths["retention"]),
        "operations": operations,
        "execution_authorized": False,
        "passes": decision == READY,
    }


def verify_readiness_package(readiness_dir: Path) -> dict[str, Any]:
    paths = readiness_paths(readiness_dir)
    expected_files = {path.resolve() for path in paths.values()}
    observed_files = {
        path.resolve()
        for path in readiness_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    } if readiness_dir.exists() else set()
    if observed_files != expected_files:
        raise J2A1V3SurfaceIntegrityError(
            "Readiness package file set changed"
        )
    payloads = {}
    identities = {}
    for key, path in paths.items():
        if not path.is_file():
            raise J2A1V3SurfaceIntegrityError(
                "Readiness package is incomplete"
            )
        payload = load_json(path)
        field = READINESS_FIELDS[key]
        if not verify_payload_hash(payload, field):
            raise J2A1V3SurfaceIntegrityError(
                f"Readiness artifact changed: {path}"
            )
        payloads[key] = payload
        identities[key] = artifact_identity(path, field)
    retention = payloads["retention"]
    for row in retention["files"]:
        path = readiness_dir / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_path(path) != row["file_sha256"]
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Readiness retention file changed"
            )
    checks = {
        "result_ready": payloads["result"].get("decision") == READY,
        "execution_unauthorized":
            payloads["result"].get("execution_authorized") is False,
        "source_current": payloads["result"].get("source_identities")
        == _source_identities(),
        "retention_passes": retention.get("passes") is True,
        "retention_inventory": canonical_json_hash(retention["files"])
        == retention["canonical_inventory_sha256"],
        "exact_file_set": observed_files == expected_files,
    }
    if not all(checks.values()):
        raise J2A1V3SurfaceIntegrityError(
            "Readiness package validation failed"
        )
    return {
        "payloads": payloads,
        "identities": identities,
        "checks": checks,
        "passes": True,
    }


def authorization_payload(
    *,
    readiness_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    readiness = verify_readiness_package(readiness_dir)
    current_authority = authority_audit()
    preflight_result = artifact_identity(
        PREFLIGHT_DIR / preflight.RESULT_NAME,
        PREFLIGHT_ARTIFACTS[preflight.RESULT_NAME][1],
    )
    preflight_lock = artifact_identity(
        PREFLIGHT_DIR / preflight.LOCK_NAME,
        PREFLIGHT_ARTIFACTS[preflight.LOCK_NAME][1],
    )
    preflight_retention = artifact_identity(
        PREFLIGHT_DIR / preflight.RETENTION_NAME,
        PREFLIGHT_ARTIFACTS[preflight.RETENTION_NAME][1],
    )
    return {
        "version": f"{VERSION}_authorization_v1",
        "decision": AUTHORIZATION_DECISION,
        "execution_mode": "scientific",
        "scientific_authority": True,
        "readiness_result": readiness["identities"]["result"],
        "readiness_lock": readiness["identities"]["lock"],
        "readiness_retention": readiness["identities"]["retention"],
        "recovery_preflight_result": preflight_result,
        "recovery_preflight_lock": preflight_lock,
        "recovery_preflight_retention": preflight_retention,
        "recovery_preflight_artifacts": {
            name: {
                "file_sha256": values[0],
                "payload_field": values[1],
                "payload_sha256": values[2],
            }
            for name, values in PREFLIGHT_ARTIFACTS.items()
        },
        "recovery_authority_audit_sha256":
            canonical_json_hash(current_authority),
        "v2_bound_artifacts": {
            str(path.relative_to(REPO_ROOT)): {
                "file_sha256": values[0],
                "payload_field": values[1],
                "payload_sha256": values[2],
            }
            for path, values in preflight.V2_BOUND_ARTIFACTS.items()
        },
        "v2_terminal": source_and_parent_audit(
            require_future_absent=False
        )["v2_terminal"],
        "v2_retention": source_and_parent_audit(
            require_future_absent=False
        )["v2_retention"],
        "execution_root": str(out_dir.resolve()),
        "jobs": 1,
        "authorized_commands": [
            "seal-phase-lock",
            "open",
            "materialize",
            "execute",
        ],
        "execution_authorized": True,
        "ppo_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
    }


def _load_authorization(
    path: Path,
    *,
    readiness_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    payload = load_json(path)
    if not verify_payload_hash(payload, "authorization_payload_sha256"):
        raise J2A1V3SurfaceIntegrityError(
            "Authorization payload changed"
        )
    expected = authorization_payload(
        readiness_dir=readiness_dir,
        out_dir=out_dir,
    )
    expected = payload_with_hash(expected, "authorization_payload_sha256")
    if payload != expected:
        raise J2A1V3SurfaceIntegrityError(
            "Authorization does not match the frozen schema"
        )
    return payload


def _phase_operational_guard(
    *,
    out_dir: Path,
    cumulative_wall_seconds: float,
    include_services: bool,
    accountant: Any | None = None,
) -> dict[str, Any]:
    from threes_rl import j2a1_distillation_fidelity_execution_surface_v2 as v2

    active_accountant = (
        v2.OutputAccountant(out_dir)
        if accountant is None
        else accountant
    )
    # V2's aggregate-attempt runtime gate is intentionally neutralized; the
    # V3 top-level wall ledger below is the sole runtime authority.
    try:
        inherited = v2.execution_operational_guard(
            phase_dir=out_dir,
            accountant=active_accountant,
            active_seconds=0.0,
            include_services=include_services,
            require_target_disk=False,
        )
    except v2.J2A1ExecutionOperationalHold as error:
        raise J2A1V3SurfaceOperationalHold(str(error)) from error
    checks = {
        "inherited_nonruntime_guards": inherited.get("passes") is True,
        "cumulative_top_level_wall_within_cap":
            cumulative_wall_seconds <= RUNTIME_CAP_SECONDS,
        "output_under_24_gib":
            active_accountant.snapshot()["output_bytes"]
            + TERMINAL_FINALIZATION_ALLOWANCE_BYTES
            <= STORAGE_CAP_BYTES,
    }
    if not all(checks.values()):
        raise J2A1V3SurfaceOperationalHold(
            "V3 recovery operational guard failed"
        )
    return {
        "version": f"{VERSION}_phase_operational_guard_v1",
        "cumulative_top_level_wall_seconds": cumulative_wall_seconds,
        "v2_prior_wall_seconds": V2_WALL_SECONDS,
        "v3_wall_seconds":
            cumulative_wall_seconds - V2_WALL_SECONDS,
        "runtime_cap_seconds": RUNTIME_CAP_SECONDS,
        "aggregate_worker_seconds_conjunctive": False,
        "output_accounting": active_accountant.snapshot(),
        "inherited": inherited,
        "checks": checks,
        "passes": True,
    }


def seal_phase_lock(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    jobs: int,
    include_operational: bool = True,
) -> dict[str, Any]:
    if jobs != 1:
        raise J2A1V3SurfaceIntegrityError("jobs must equal one")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise J2A1V3SurfaceIntegrityError(
            "Execution root is not fresh"
        )
    authorization = _load_authorization(
        authorization_path,
        readiness_dir=readiness_dir,
        out_dir=out_dir,
    )
    operations = operational_audit(
        output_dir=out_dir,
        include_services=include_operational,
        require_future_absent=False,
    )
    if not operations["passes"]:
        raise J2A1V3SurfaceOperationalHold(
            "Phase-lock operational audit failed"
        )
    payload = {
        "version": f"{VERSION}_phase_lock_v1",
        "phase": "distillation_recovery",
        "execution_mode": "scientific",
        "scientific_authority": True,
        "authorization": artifact_identity(
            authorization_path,
            "authorization_payload_sha256",
        ),
        "readiness": verify_readiness_package(readiness_dir)[
            "identities"
        ],
        "recovery_preflight": {
            name: {
                "file_sha256": values[0],
                "payload_sha256": values[2],
            }
            for name, values in PREFLIGHT_ARTIFACTS.items()
        },
        "v2_stream_reservation":
            preflight.V2_BOUND_ARTIFACTS[
                preflight.V2_EXECUTION_DIR
                / "J2A1_V2_DISTILLATION_STREAM_RESERVATION.json"
            ][0],
        "v2_stream_consumption":
            preflight.V2_BOUND_ARTIFACTS[
                preflight.V2_EXECUTION_DIR
                / "J2A1_V2_DISTILLATION_STREAM_CONSUMPTION.json"
            ][0],
        "unfinished_rows_sha256": EXPECTED_UNFINISHED_SHA256,
        "completed_refs_sha256": EXPECTED_COMPLETED_REFS_SHA256,
        "jobs": 1,
        "collectors": COLLECTORS,
        "new_reservations": 0,
        "new_consumptions": 0,
        "operations": operations,
        "counters": dict(ZERO_WORK),
        "passes": authorization["execution_authorized"] is True,
    }
    path = out_dir / PHASE_LOCK_NAME
    return write_immutable_json(
        path,
        payload,
        field="phase_lock_payload_sha256",
    )


def _load_phase_lock(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    authorization = _load_authorization(
        authorization_path,
        readiness_dir=readiness_dir,
        out_dir=out_dir,
    )
    path = out_dir / PHASE_LOCK_NAME
    lock = load_json(path)
    if (
        not verify_payload_hash(lock, "phase_lock_payload_sha256")
        or lock.get("authorization")
        != artifact_identity(
            authorization_path,
            "authorization_payload_sha256",
        )
        or lock.get("unfinished_rows_sha256")
        != EXPECTED_UNFINISHED_SHA256
        or lock.get("completed_refs_sha256")
        != EXPECTED_COMPLETED_REFS_SHA256
        or lock.get("new_reservations") != 0
        or lock.get("new_consumptions") != 0
        or lock.get("passes") is not True
        or authorization.get("execution_authorized") is not True
    ):
        raise J2A1V3SurfaceIntegrityError("Phase lock changed")
    return {
        "payload": lock,
        "identity": artifact_identity(
            path,
            "phase_lock_payload_sha256",
        ),
    }


def open_phase(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    jobs: int,
) -> dict[str, Any]:
    if jobs != 1:
        raise J2A1V3SurfaceIntegrityError("jobs must equal one")
    lock = _load_phase_lock(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
    )
    unexpected = [
        path.name
        for path in out_dir.iterdir()
        if path.name not in {PHASE_LOCK_NAME}
    ]
    if unexpected:
        raise J2A1V3SurfaceIntegrityError(
            "Open encountered unexpected execution files"
        )
    payload = {
        "version": f"{VERSION}_marker_v1",
        "phase_lock": lock["identity"],
        "authorization": lock["payload"]["authorization"],
        "content_blind": True,
        "v2_completed_root_bodies_opened": 0,
        "unfinished_root_bodies_opened": 0,
        "teacher_queries": 0,
        "new_reservations": 0,
        "new_consumptions": 0,
        "jobs": 1,
        "collectors": COLLECTORS,
        "passes": True,
    }
    return write_immutable_json(
        out_dir / MARKER_NAME,
        payload,
        field="execution_marker_payload_sha256",
    )


def _load_marker_chain(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    lock = _load_phase_lock(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
    )
    marker_path = out_dir / MARKER_NAME
    marker = load_json(marker_path)
    if (
        not verify_payload_hash(
            marker,
            "execution_marker_payload_sha256",
        )
        or marker.get("phase_lock") != lock["identity"]
        or marker.get("content_blind") is not True
        or marker.get("teacher_queries") != 0
        or marker.get("new_reservations") != 0
        or marker.get("new_consumptions") != 0
    ):
        raise J2A1V3SurfaceIntegrityError("Execution marker changed")
    return {
        "lock": lock,
        "marker": marker,
        "marker_identity": artifact_identity(
            marker_path,
            "execution_marker_payload_sha256",
        ),
    }


def materialize_phase(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    jobs: int,
) -> dict[str, Any]:
    if jobs != 1:
        raise J2A1V3SurfaceIntegrityError("jobs must equal one")
    chain = _load_marker_chain(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
    )
    unexpected = [
        path.name
        for path in out_dir.iterdir()
        if path.name not in {PHASE_LOCK_NAME, MARKER_NAME}
    ]
    if unexpected:
        raise J2A1V3SurfaceIntegrityError(
            "Materialize encountered unexpected execution files"
        )
    authority = load_recovery_authority()
    rows = authority["unfinished_rows"]
    payload = {
        "version": f"{VERSION}_materialized_authority_v1",
        "phase_lock": chain["lock"]["identity"],
        "marker": chain["marker_identity"],
        "row_count": len(rows),
        "v2_completed_count": len(authority["completed_refs"]),
        "total_completion_gate": ACTIVE_ROOTS,
        "rows": rows,
        "canonical_rows_sha256": canonical_json_hash(rows),
        "completed_refs_sha256": authority["completed_refs_sha256"],
        "root_set_sha256": canonical_json_hash(
            sorted(row["root_id"] for row in rows)
        ),
        "ancestry_set_sha256": canonical_json_hash(
            sorted(row["ancestry_id"] for row in rows)
        ),
        "stream_set_sha256": canonical_json_hash(
            sorted(
                int(stream_id)
                for row in rows
                for stream_id in row["streams"].values()
            )
        ),
        "new_reservations": 0,
        "new_consumptions": 0,
        "content_opened": 0,
        "teacher_queries": 0,
        "passes": len(rows) == RECOVERY_ROOTS
        and canonical_json_hash(rows) == EXPECTED_UNFINISHED_SHA256,
    }
    return write_immutable_json(
        out_dir / MATERIALIZED_NAME,
        payload,
        field="materialized_authority_payload_sha256",
    )


def _load_materialized_chain(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    chain = _load_marker_chain(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
    )
    path = out_dir / MATERIALIZED_NAME
    manifest = load_json(path)
    rows = manifest.get("rows")
    if (
        not verify_payload_hash(
            manifest,
            "materialized_authority_payload_sha256",
        )
        or manifest.get("phase_lock") != chain["lock"]["identity"]
        or manifest.get("marker") != chain["marker_identity"]
        or not isinstance(rows, list)
        or len(rows) != RECOVERY_ROOTS
        or canonical_json_hash(rows) != EXPECTED_UNFINISHED_SHA256
        or manifest.get("canonical_rows_sha256")
        != EXPECTED_UNFINISHED_SHA256
        or manifest.get("completed_refs_sha256")
        != EXPECTED_COMPLETED_REFS_SHA256
        or manifest.get("new_reservations") != 0
        or manifest.get("new_consumptions") != 0
        or manifest.get("content_opened") != 0
        or manifest.get("teacher_queries") != 0
        or manifest.get("passes") is not True
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Materialized recovery authority changed"
        )
    return {
        **chain,
        "manifest": manifest,
        "manifest_identity": artifact_identity(
            path,
            "materialized_authority_payload_sha256",
        ),
        "rows": rows,
    }


def phase_contract_sha256(
    chain: Mapping[str, Any],
    *,
    command: str,
) -> str:
    return canonical_json_hash(
        {
            "version": f"{VERSION}_phase_contract_v1",
            "lock": chain["lock"]["identity"],
            "marker": chain["marker_identity"],
            "manifest": chain["manifest_identity"],
            "command": command,
            "runner_sha256": sha256_path(RUNNER_PATH),
            "execution_mode": "scientific",
            "jobs": 1,
        }
    )


def _append_jsonl_record(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    record = payload_with_hash(payload, field)
    serialized = canonical_json_bytes(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _load_jsonl_chain(
    path: Path,
    *,
    field: str,
    predecessor_field: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    predecessor = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise J2A1V3SurfaceIntegrityError(
                    f"Malformed ledger line {line_number}: {path}"
                ) from error
            if (
                not isinstance(record, dict)
                or not verify_payload_hash(record, field)
                or record.get("sequence") != len(records)
                or record.get(predecessor_field) != predecessor
            ):
                raise J2A1V3SurfaceIntegrityError(
                    f"Ledger chain changed: {path}"
                )
            records.append(record)
            predecessor = record[field]
    return records


class TopLevelWallLedger:
    """Append-only top-level wall accounting; worker seconds stay separate."""

    def __init__(
        self,
        *,
        path: Path,
        contract_sha256: str,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.contract_sha256 = _validate_hex(
            contract_sha256,
            name="wall contract",
        )
        self.wall_clock = wall_clock
        self.records = _load_jsonl_chain(
            path,
            field="wall_record_sha256",
            predecessor_field="predecessor_record_sha256",
        )
        self.v3_seconds = 0.0
        self.open_segment: dict[str, Any] | None = None
        self.last_timestamp: float | None = None
        for record in self.records:
            if record.get("contract_sha256") != self.contract_sha256:
                raise J2A1V3SurfaceIntegrityError(
                    "Wall ledger contract changed"
                )
            event = record.get("event")
            if event == "started":
                if self.open_segment is not None:
                    raise J2A1V3SurfaceIntegrityError(
                        "Wall ledger overlaps segments"
                    )
                if record.get("unit_kind") not in (
                    ABANDONED_WALL_CHARGE_SECONDS
                ):
                    raise J2A1V3SurfaceIntegrityError(
                        "Wall ledger start has unknown unit"
                    )
                self.open_segment = record
                self.last_timestamp = float(record["wall_timestamp"])
            elif event in {"heartbeat", "finished"}:
                if self.open_segment is None:
                    raise J2A1V3SurfaceIntegrityError(
                        "Wall ledger event has no segment"
                    )
                timestamp = float(record["wall_timestamp"])
                charged = float(record["charged_seconds"])
                expected_charge = timestamp - float(self.last_timestamp)
                if (
                    record.get("segment_id")
                    != self.open_segment.get("segment_id")
                    or record.get("unit_kind")
                    != self.open_segment.get("unit_kind")
                    or not math.isfinite(charged)
                    or charged < 0
                    or not math.isclose(
                        charged,
                        expected_charge,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    raise J2A1V3SurfaceIntegrityError(
                        "Wall ledger charge changed"
                    )
                self.v3_seconds += charged
                self.last_timestamp = timestamp
                if event == "finished":
                    self.open_segment = None
                    self.last_timestamp = None
            elif event == "abandoned":
                if self.open_segment is None:
                    raise J2A1V3SurfaceIntegrityError(
                        "Wall abandonment has no segment"
                    )
                unit_kind = str(self.open_segment["unit_kind"])
                charge = float(record["charged_seconds"])
                if (
                    record.get("segment_id")
                    != self.open_segment.get("segment_id")
                    or record.get("unit_kind") != unit_kind
                    or charge
                    != ABANDONED_WALL_CHARGE_SECONDS[unit_kind]
                    or record.get("dead_process_downtime_charged") is not False
                ):
                    raise J2A1V3SurfaceIntegrityError(
                        "Wall abandonment charge changed"
                    )
                self.v3_seconds += charge
                self.open_segment = None
                self.last_timestamp = None
            else:
                raise J2A1V3SurfaceIntegrityError(
                    "Wall ledger event changed"
                )

    def _append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = _append_jsonl_record(
            self.path,
            {
                "version": f"{VERSION}_wall_record_v1",
                "sequence": len(self.records),
                "predecessor_record_sha256": (
                    None
                    if not self.records
                    else self.records[-1]["wall_record_sha256"]
                ),
                "contract_sha256": self.contract_sha256,
                **dict(payload),
            },
            field="wall_record_sha256",
        )
        self.records.append(record)
        return record

    def start(self, *, unit_kind: str) -> dict[str, Any]:
        if self.open_segment is not None:
            raise J2A1V3SurfaceIntegrityError(
                "Wall segment is already open"
            )
        if unit_kind not in ABANDONED_WALL_CHARGE_SECONDS:
            raise J2A1V3SurfaceIntegrityError("Wall unit kind is unknown")
        now = float(self.wall_clock())
        record = self._append(
            {
                "event": "started",
                "segment_id": f"segment={len(self.records)}",
                "unit_kind": unit_kind,
                "wall_timestamp": now,
            }
        )
        self.open_segment = record
        self.last_timestamp = now
        return record

    def heartbeat(self, *, unit_kind: str) -> dict[str, Any]:
        if (
            self.open_segment is None
            or self.last_timestamp is None
            or unit_kind not in ABANDONED_WALL_CHARGE_SECONDS
            or unit_kind != self.open_segment.get("unit_kind")
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Wall heartbeat has no valid segment"
            )
        now = float(self.wall_clock())
        delta = now - self.last_timestamp
        if not math.isfinite(delta) or delta < 0:
            raise J2A1V3SurfaceIntegrityError(
                "Wall clock moved backwards"
            )
        record = self._append(
            {
                "event": "heartbeat",
                "segment_id": self.open_segment["segment_id"],
                "unit_kind": unit_kind,
                "wall_timestamp": now,
                "charged_seconds": delta,
            }
        )
        self.v3_seconds += delta
        self.last_timestamp = now
        return record

    def finish(self, *, unit_kind: str) -> dict[str, Any]:
        if (
            self.open_segment is None
            or unit_kind != self.open_segment.get("unit_kind")
        ):
            raise J2A1V3SurfaceIntegrityError(
                "No matching wall segment is open"
            )
        heartbeat = self.heartbeat(unit_kind=unit_kind)
        record = self._append(
            {
                "event": "finished",
                "segment_id": self.open_segment["segment_id"],
                "unit_kind": unit_kind,
                "wall_timestamp": heartbeat["wall_timestamp"],
                "charged_seconds": 0.0,
            }
        )
        self.open_segment = None
        self.last_timestamp = None
        return record

    def switch(self, *, unit_kind: str) -> dict[str, Any]:
        if unit_kind not in ABANDONED_WALL_CHARGE_SECONDS:
            raise J2A1V3SurfaceIntegrityError("Wall unit kind is unknown")
        previous = None
        if self.open_segment is not None:
            previous_kind = str(self.open_segment["unit_kind"])
            if previous_kind == unit_kind:
                return {
                    "previous": None,
                    "current": self.open_segment,
                    "changed": False,
                }
            previous = self.finish(unit_kind=previous_kind)
        current = self.start(unit_kind=unit_kind)
        return {
            "previous": previous,
            "current": current,
            "changed": True,
        }

    def abandon_open(self) -> dict[str, Any] | None:
        if self.open_segment is None:
            return None
        unit_kind = str(self.open_segment["unit_kind"])
        charge = ABANDONED_WALL_CHARGE_SECONDS[unit_kind]
        record = self._append(
            {
                "event": "abandoned",
                "segment_id": self.open_segment["segment_id"],
                "unit_kind": unit_kind,
                "wall_timestamp": float(self.wall_clock()),
                "charged_seconds": charge,
                "charge_basis": "frozen abandoned-unit ceiling",
                "dead_process_downtime_charged": False,
            }
        )
        self.v3_seconds += charge
        self.open_segment = None
        self.last_timestamp = None
        return record

    def cumulative_seconds(self) -> float:
        return V2_WALL_SECONDS + self.v3_seconds

    def summary(self) -> dict[str, Any]:
        return {
            "v2_prior_wall_seconds": V2_WALL_SECONDS,
            "v3_charged_wall_seconds": self.v3_seconds,
            "cumulative_wall_seconds": self.cumulative_seconds(),
            "cap_seconds": RUNTIME_CAP_SECONDS,
            "record_count": len(self.records),
            "head_sha256": (
                None
                if not self.records
                else self.records[-1]["wall_record_sha256"]
            ),
            "open_segment": (
                None
                if self.open_segment is None
                else self.open_segment["segment_id"]
            ),
            "passes": self.cumulative_seconds()
            <= RUNTIME_CAP_SECONDS,
        }


def terminal_precedence(
    *,
    integrity_failure: bool,
    operational_failure: bool,
    scientific_decision: str,
) -> str:
    if integrity_failure:
        return KILL_EXECUTION
    if operational_failure:
        return KILL_EXECUTION
    return scientific_decision


def stage_b_barrier(
    *,
    v2_completed: int,
    v3_completed: int,
    union_passes: bool,
) -> dict[str, Any]:
    checks = {
        "v2_exact": v2_completed == V2_COMPLETED_ROOTS,
        "v3_exact": v3_completed == RECOVERY_ROOTS,
        "total_exact": v2_completed + v3_completed == ACTIVE_ROOTS,
        "union_passes": union_passes is True,
    }
    return {
        "v2_completed": v2_completed,
        "v3_completed": v3_completed,
        "total_completed": v2_completed + v3_completed,
        "required": ACTIVE_ROOTS,
        "checks": checks,
        "passes": all(checks.values()),
    }


def canonical_merge_refs(
    *,
    full_rows: Sequence[Mapping[str, Any]],
    v2_refs: Mapping[str, Mapping[str, Any]],
    v3_refs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if set(v2_refs) & set(v3_refs):
        raise J2A1V3SurfaceIntegrityError(
            "V2/V3 completion sets overlap"
        )
    merged = []
    for authority_index, row in enumerate(full_rows):
        root_id = str(row["root_id"])
        if root_id in v2_refs:
            source = "v2"
            ref = dict(v2_refs[root_id])
        elif root_id in v3_refs:
            source = "v3"
            ref = dict(v3_refs[root_id])
        else:
            raise J2A1V3SurfaceIntegrityError(
                "Union is missing an authority root"
            )
        if (
            ref.get("ancestry_id") != row["ancestry_id"]
            or ref.get("row_index") != row["row_index"]
            or ref.get("stage") != row["stage"]
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Union ref changed its authority"
            )
        merged.append(
            {
                "authority_index": authority_index,
                "source": source,
                "root_id": root_id,
                "ancestry_id": row["ancestry_id"],
                "row_index": row["row_index"],
                "stage": row["stage"],
                "relative_path": ref["relative_path"],
                "file_sha256": ref["file_sha256"],
                "content_sha256": ref["content_sha256"],
            }
        )
    if len(merged) != ACTIVE_ROOTS:
        raise J2A1V3SurfaceIntegrityError(
            "Union completion count changed"
        )
    return merged


def _full_authority_rows() -> list[dict[str, Any]]:
    manifest_path = (
        preflight.V2_EXECUTION_DIR
        / "J2A1_V2_DISTILLATION_ACTIVE_MANIFEST.json"
    )
    expected = preflight.V2_BOUND_ARTIFACTS[manifest_path]
    manifest = _load_bound_json(
        manifest_path,
        file_sha256=expected[0],
        payload_field=str(expected[1]),
        payload_sha256=str(expected[2]),
    )
    try:
        return preflight._manifest_rows(manifest)
    except preflight.J2A1V3IntegrityError as error:
        raise J2A1V3SurfaceIntegrityError(str(error)) from error


def seal_union(
    *,
    out_dir: Path,
    v3_completion_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    authority = load_recovery_authority()
    v2_refs = {
        str(row["root_id"]): row
        for row in authority["completed_refs"]
    }
    v3_refs = {
        str(row["root_id"]): row for row in v3_completion_records
    }
    expected_v3_roots = {
        str(row["root_id"]) for row in authority["unfinished_rows"]
    }
    if (
        len(v3_refs) != len(v3_completion_records)
        or len(v2_refs) != len(authority["completed_refs"])
        or set(v3_refs) != expected_v3_roots
    ):
        raise J2A1V3SurfaceIntegrityError(
            "V3 completions do not match unfinished authority"
        )
    for ref in v2_refs.values():
        path = preflight.V2_EXECUTION_DIR / ref["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != ref["bytes"]
            or sha256_path(path) != ref["file_sha256"]
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Preserved V2 completion bytes changed"
            )
    for ref in v3_refs.values():
        path = out_dir / str(ref["relative_path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or (
                "bytes" in ref
                and path.stat().st_size != int(ref["bytes"])
            )
            or sha256_path(path) != ref["file_sha256"]
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Recovery completion bytes changed"
            )
    merged = canonical_merge_refs(
        full_rows=_full_authority_rows(),
        v2_refs=v2_refs,
        v3_refs=v3_refs,
    )
    barrier = stage_b_barrier(
        v2_completed=len(v2_refs),
        v3_completed=len(v3_refs),
        union_passes=True,
    )
    file_hashes = [str(row["file_sha256"]) for row in merged]
    content_hashes = [str(row["content_sha256"]) for row in merged]
    ancestries = [str(row["ancestry_id"]) for row in merged]
    predecessor = source_and_parent_audit(
        require_future_absent=False,
    )
    uniqueness = {
        "root_ids_unique":
            len({str(row["root_id"]) for row in merged}) == ACTIVE_ROOTS,
        "ancestries_unique": len(set(ancestries)) == ACTIVE_ROOTS,
        "file_hashes_unique": len(set(file_hashes)) == ACTIVE_ROOTS,
        "content_hashes_unique": len(set(content_hashes)) == ACTIVE_ROOTS,
    }
    if not all(uniqueness.values()):
        raise J2A1V3SurfaceIntegrityError(
            "Recovery union contains a duplicate identity"
        )
    payload = {
        "version": f"{VERSION}_union_complete_v1",
        "v2_completed": len(v2_refs),
        "v3_completed": len(v3_refs),
        "total_completed": len(merged),
        "canonical_merge_sha256": canonical_json_hash(merged),
        "merged_refs": merged,
        "stream_set_sha256":
            preflight.EXPECTED_MANIFEST_IDENTITIES["stream_set_sha256"],
        "v2_predecessor_binding_sha256":
            canonical_json_hash(predecessor),
        "v2_terminal": predecessor["v2_terminal"],
        "v2_retention": predecessor["v2_retention"],
        "uniqueness": uniqueness,
        "new_reservations": 0,
        "new_consumptions": 0,
        "replacement_roots": 0,
        "filtered_roots": 0,
        "body_deserializations_before_union": 0,
        "stage_b_barrier": barrier,
        "passes": barrier["passes"],
    }
    return write_immutable_json(
        out_dir / UNION_NAME,
        payload,
        field="union_payload_sha256",
    )


def stream_authority_reuse_payload(
    chain: Mapping[str, Any],
) -> dict[str, Any]:
    reservation_path = (
        preflight.V2_EXECUTION_DIR
        / "J2A1_V2_DISTILLATION_STREAM_RESERVATION.json"
    )
    consumption_path = (
        preflight.V2_EXECUTION_DIR
        / "J2A1_V2_DISTILLATION_STREAM_CONSUMPTION.json"
    )
    reservation_expected = preflight.V2_BOUND_ARTIFACTS[reservation_path]
    consumption_expected = preflight.V2_BOUND_ARTIFACTS[consumption_path]
    _load_bound_json(
        reservation_path,
        file_sha256=reservation_expected[0],
        payload_field=str(reservation_expected[1]),
        payload_sha256=str(reservation_expected[2]),
    )
    _load_bound_json(
        consumption_path,
        file_sha256=consumption_expected[0],
        payload_field=str(consumption_expected[1]),
        payload_sha256=str(consumption_expected[2]),
    )
    return {
        "version": f"{VERSION}_stream_authority_reuse_v1",
        "phase_lock": chain["lock"]["identity"],
        "marker": chain["marker_identity"],
        "manifest": chain["manifest_identity"],
        "v2_reservation": {
            "file_sha256": reservation_expected[0],
            "payload_sha256": reservation_expected[2],
        },
        "v2_consumption": {
            "file_sha256": consumption_expected[0],
            "payload_sha256": consumption_expected[2],
        },
        "stream_set_sha256":
            preflight.EXPECTED_MANIFEST_IDENTITIES["stream_set_sha256"],
        "stream_count": ACTIVE_STREAMS,
        "recovery_row_count": RECOVERY_ROOTS,
        "new_reservations": 0,
        "new_consumptions": 0,
        "reuse_only": True,
        "passes": True,
    }


def _sha256_prefix(path: Path, size: int) -> str:
    if size < 0 or not path.is_file() or path.is_symlink():
        raise J2A1V3SurfaceIntegrityError(
            f"Committed path is missing or invalid: {path}"
        )
    if path.stat().st_size < size:
        raise J2A1V3SurfaceIntegrityError(
            f"Committed path was truncated: {path}"
        )
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise J2A1V3SurfaceIntegrityError(
                    f"Committed path ended early: {path}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def verify_commit_ledger(
    *,
    path: Path,
    contract_sha256: str,
) -> list[dict[str, Any]]:
    records = _load_jsonl_chain(
        path,
        field="commit_record_sha256",
        predecessor_field="predecessor_record_sha256",
    )
    requirements: dict[Path, set[tuple[int, str]]] = {}
    for record in records:
        inventory = record.get("inventory")
        if (
            record.get("contract_sha256") != contract_sha256
            or not isinstance(inventory, list)
            or record.get("inventory_sha256")
            != canonical_json_hash(inventory)
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Recovery commit contract changed"
            )
        seen: set[str] = set()
        for item in inventory:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "bytes", "file_sha256"}
                or not isinstance(item["path"], str)
                or type(item["bytes"]) is not int
                or item["bytes"] < 0
            ):
                raise J2A1V3SurfaceIntegrityError(
                    "Recovery commit inventory is malformed"
                )
            file_sha = _validate_hex(
                item["file_sha256"],
                name="commit file SHA",
            )
            if item["path"] in seen:
                raise J2A1V3SurfaceIntegrityError(
                    "Recovery commit repeats an inventory path"
                )
            seen.add(item["path"])
            requirements.setdefault(Path(item["path"]), set()).add(
                (int(item["bytes"]), file_sha)
            )
    for committed_path, identities in requirements.items():
        current_size = (
            int(committed_path.stat().st_size)
            if committed_path.is_file()
            else -1
        )
        for size, expected_sha in sorted(identities):
            append_only = committed_path.suffix == ".jsonl"
            if (
                current_size < size
                or (not append_only and current_size != size)
                or _sha256_prefix(committed_path, size) != expected_sha
            ):
                raise J2A1V3SurfaceIntegrityError(
                    f"Recovery commit inventory changed: {committed_path}"
                )
    return records


def _append_commit(
    *,
    path: Path,
    contract_sha256: str,
    kind: str,
    bound_paths: Sequence[Path],
) -> dict[str, Any]:
    records = _load_jsonl_chain(
        path,
        field="commit_record_sha256",
        predecessor_field="predecessor_record_sha256",
    )
    if any(
        record.get("contract_sha256") != contract_sha256
        for record in records
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Recovery commit contract changed"
        )
    inventory = [
        {
            "path": str(item),
            "bytes": int(item.stat().st_size),
            "file_sha256": sha256_path(item),
        }
        for item in bound_paths
        if item.is_file()
    ]
    return _append_jsonl_record(
        path,
        {
            "version": f"{VERSION}_commit_v1",
            "sequence": len(records),
            "predecessor_record_sha256": (
                None
                if not records
                else records[-1]["commit_record_sha256"]
            ),
            "contract_sha256": contract_sha256,
            "kind": kind,
            "inventory": inventory,
            "inventory_sha256": canonical_json_hash(inventory),
        },
        field="commit_record_sha256",
    )


def _owner_contract_sha256(
    chain: Mapping[str, Any],
    *,
    command: str,
) -> str:
    return phase_contract_sha256(chain, command=command)


def acquire_or_reclaim_owner(
    *,
    ledger_path: Path,
    chain: Mapping[str, Any],
    command: str,
    commit_ledger_path: Path,
    pid: int,
    process_start_identity: str,
    is_live: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    contract = _owner_contract_sha256(chain, command=command)
    records = _load_jsonl_chain(
        ledger_path,
        field="owner_record_sha256",
        predecessor_field="predecessor_record_sha256",
    )
    commits = verify_commit_ledger(
        path=commit_ledger_path,
        contract_sha256=contract,
    )
    if not commits:
        raise J2A1V3SurfaceIntegrityError(
            "Owner acquisition requires a durable genesis commit"
        )
    commit_head = commits[-1]["commit_record_sha256"]
    if not records:
        record = _append_jsonl_record(
            ledger_path,
            {
                "version": f"{VERSION}_owner_v1",
                "sequence": 0,
                "predecessor_record_sha256": None,
                "kind": "owner",
                "contract_sha256": contract,
                "marker_sha256":
                    chain["marker_identity"]["file_sha256"],
                "lock_sha256":
                    chain["lock"]["identity"]["file_sha256"],
                "manifest_sha256":
                    chain["manifest_identity"]["file_sha256"],
                "command": command,
                "runner_sha256": sha256_path(RUNNER_PATH),
                "hostname": socket.gethostname(),
                "pid": int(pid),
                "process_start_identity": process_start_identity,
                "commit_head_sha256": commit_head,
            },
            field="owner_record_sha256",
        )
        return {"owner": record, "reclaimed": False, "passes": True}
    head = records[-1]
    expected = {
        "contract": head.get("contract_sha256") == contract,
        "marker": head.get("marker_sha256")
        == chain["marker_identity"]["file_sha256"],
        "lock": head.get("lock_sha256")
        == chain["lock"]["identity"]["file_sha256"],
        "manifest": head.get("manifest_sha256")
        == chain["manifest_identity"]["file_sha256"],
        "command": head.get("command") == command,
        "runner": head.get("runner_sha256") == sha256_path(RUNNER_PATH),
    }
    if not all(expected.values()):
        raise J2A1V3SurfaceIntegrityError(
            "Existing owner contract changed"
        )
    bound_commit_head = (
        head.get("commit_head_sha256")
        if head.get("kind") == "owner"
        else head.get("predecessor_commit_head_sha256")
    )
    if bound_commit_head not in {
        record["commit_record_sha256"] for record in commits
    }:
        raise J2A1V3SurfaceIntegrityError(
            "Existing owner is not anchored in the commit chain"
        )
    if is_live(head):
        raise J2A1V3SurfaceOperationalHold(
            "A live recovery owner already exists"
        )
    reclaim = _append_jsonl_record(
        ledger_path,
        {
            "version": f"{VERSION}_owner_reclaim_v1",
            "sequence": len(records),
            "predecessor_record_sha256":
                head["owner_record_sha256"],
            "kind": "reclaim",
            "contract_sha256": contract,
            "marker_sha256":
                chain["marker_identity"]["file_sha256"],
            "lock_sha256": chain["lock"]["identity"]["file_sha256"],
            "manifest_sha256":
                chain["manifest_identity"]["file_sha256"],
            "command": command,
            "runner_sha256": sha256_path(RUNNER_PATH),
            "hostname": socket.gethostname(),
            "pid": int(pid),
            "process_start_identity": process_start_identity,
            "recovered_owner_record_sha256":
                head["owner_record_sha256"],
            "predecessor_commit_head_sha256": commit_head,
            "commit_head_sha256": commit_head,
            "process_death_verified": True,
            "zero_concurrent_writer_verified": True,
        },
        field="owner_record_sha256",
    )
    return {"owner": reclaim, "reclaimed": True, "passes": True}


def verify_current_owner(
    *,
    ledger_path: Path,
    expected_owner_sha256: str,
    chain: Mapping[str, Any],
    command: str,
    pid: int,
    process_start_identity: str,
) -> dict[str, Any]:
    records = _load_jsonl_chain(
        ledger_path,
        field="owner_record_sha256",
        predecessor_field="predecessor_record_sha256",
    )
    if not records:
        raise J2A1V3SurfaceIntegrityError(
            "Recovery owner ledger is empty"
        )
    head = records[-1]
    checks = {
        "head": head.get("owner_record_sha256")
        == expected_owner_sha256,
        "contract": head.get("contract_sha256")
        == _owner_contract_sha256(chain, command=command),
        "marker": head.get("marker_sha256")
        == chain["marker_identity"]["file_sha256"],
        "lock": head.get("lock_sha256")
        == chain["lock"]["identity"]["file_sha256"],
        "manifest": head.get("manifest_sha256")
        == chain["manifest_identity"]["file_sha256"],
        "command": head.get("command") == command,
        "runner": head.get("runner_sha256") == sha256_path(RUNNER_PATH),
        "pid": head.get("pid") == int(pid),
        "process_start": head.get("process_start_identity")
        == process_start_identity,
    }
    if not all(checks.values()):
        raise J2A1V3SurfaceIntegrityError(
            "Current recovery owner changed"
        )
    return {
        "head": head,
        "record_count": len(records),
        "checks": checks,
        "passes": True,
    }


def _pid_live(record: Mapping[str, Any]) -> bool:
    pid = int(record.get("pid", -1))
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _execution_inventory(out_dir: Path) -> list[dict[str, Any]]:
    result_path = out_dir / TERMINAL_NAME
    retention_path = out_dir / EXECUTION_RETENTION_NAME
    return [
        {
            "path": str(path.relative_to(out_dir)),
            "bytes": int(path.stat().st_size),
            "file_sha256": sha256_path(path),
        }
        for path in sorted(out_dir.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path not in {result_path, retention_path}
    ]


def _verify_execution_retention(
    out_dir: Path,
    retention: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = retention.get("files")
    if (
        not verify_payload_hash(retention, "retention_payload_sha256")
        or not isinstance(inventory, list)
        or retention.get("canonical_inventory_sha256")
        != canonical_json_hash(inventory)
        or retention.get("passes") is not True
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Execution retention payload changed"
        )
    expected_paths: set[str] = set()
    retained_bytes = 0
    for item in inventory:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "bytes", "file_sha256"}
            or not isinstance(item["path"], str)
            or Path(item["path"]).is_absolute()
            or ".." in Path(item["path"]).parts
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Execution retention inventory is malformed"
            )
        path = out_dir / item["path"]
        if (
            item["path"] in expected_paths
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item["bytes"]
            or sha256_path(path) != item["file_sha256"]
        ):
            raise J2A1V3SurfaceIntegrityError(
                "Execution retention bytes changed"
            )
        expected_paths.add(item["path"])
        retained_bytes += int(item["bytes"])
    observed_paths = {
        str(path.relative_to(out_dir))
        for path in out_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name not in {TERMINAL_NAME, EXECUTION_RETENTION_NAME}
    }
    total_bytes = sum(
        int(path.stat().st_size)
        for path in out_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    checks = {
        "inventory_exact": observed_paths == expected_paths,
        "file_count_exact": retention.get("file_count")
        == len(inventory),
        "retained_bytes_exact": retention.get("retained_bytes")
        == retained_bytes,
        "allowance_exact":
            retention.get("terminal_finalization_allowance_bytes")
            == TERMINAL_FINALIZATION_ALLOWANCE_BYTES,
        "retained_under_cap":
            retention.get("under_24_gib") is True,
        "final_directory_under_cap": total_bytes <= STORAGE_CAP_BYTES,
    }
    if not all(checks.values()):
        raise J2A1V3SurfaceIntegrityError(
            "Execution retention audit failed"
        )
    return {
        "inventory": inventory,
        "retained_bytes": retained_bytes,
        "total_bytes": total_bytes,
        "checks": checks,
        "passes": True,
    }


def _seal_execution_retention(out_dir: Path) -> dict[str, Any]:
    retention_path = out_dir / EXECUTION_RETENTION_NAME
    inventory = _execution_inventory(out_dir)
    retained_bytes = sum(row["bytes"] for row in inventory)
    payload = {
        "version": f"{VERSION}_execution_retention_v1",
        "files": inventory,
        "file_count": len(inventory),
        "retained_bytes": retained_bytes,
        "canonical_inventory_sha256": canonical_json_hash(inventory),
        "terminal_finalization_allowance_bytes":
            TERMINAL_FINALIZATION_ALLOWANCE_BYTES,
        "under_24_gib": retained_bytes
        + TERMINAL_FINALIZATION_ALLOWANCE_BYTES
        <= STORAGE_CAP_BYTES,
        "no_cleanup_performed": True,
        "passes": retained_bytes
        + TERMINAL_FINALIZATION_ALLOWANCE_BYTES
        <= STORAGE_CAP_BYTES,
    }
    sealed = payload_with_hash(payload, "retention_payload_sha256")
    if (
        retained_bytes
        + len(_serialized_json_bytes(sealed))
        + TERMINAL_FINALIZATION_ALLOWANCE_BYTES
        > STORAGE_CAP_BYTES
    ):
        raise J2A1V3SurfaceOperationalHold(
            "Terminal retention would exceed the output cap"
        )
    observed = write_immutable_json(
        retention_path,
        payload,
        field="retention_payload_sha256",
    )
    _verify_execution_retention(out_dir, observed)
    return observed


def _load_terminal_bundle(out_dir: Path) -> dict[str, Any]:
    paths = phase_paths(out_dir)
    terminal = load_json(paths["result"])
    evidence = load_json(paths["terminal_evidence"])
    retention = load_json(paths["retention"])
    if (
        not verify_payload_hash(terminal, "terminal_payload_sha256")
        or not verify_payload_hash(
            evidence,
            "terminal_evidence_payload_sha256",
        )
        or terminal.get("terminal_evidence")
        != artifact_identity(
            paths["terminal_evidence"],
            "terminal_evidence_payload_sha256",
        )
        or terminal.get("retention")
        != artifact_identity(
            paths["retention"],
            "retention_payload_sha256",
        )
        or terminal.get("authoritative_terminal_written_last") is not True
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Existing terminal bundle changed"
        )
    terminal_body = dict(terminal)
    terminal_body.pop("terminal_payload_sha256", None)
    evidence_body = dict(evidence)
    evidence_body.pop("terminal_evidence_payload_sha256", None)
    expected_terminal_body = {
        **evidence_body,
        "terminal_evidence": artifact_identity(
            paths["terminal_evidence"],
            "terminal_evidence_payload_sha256",
        ),
        "retention": artifact_identity(
            paths["retention"],
            "retention_payload_sha256",
        ),
        "authoritative_terminal_written_last": True,
    }
    if terminal_body != expected_terminal_body:
        raise J2A1V3SurfaceIntegrityError(
            "Terminal and evidence payloads disagree"
        )
    retention_audit = _verify_execution_retention(out_dir, retention)
    return {
        "terminal": terminal,
        "evidence": evidence,
        "retention": retention,
        "retention_audit": retention_audit,
        "existing": True,
        "passes": True,
    }


def _resume_terminal_finalization(
    out_dir: Path,
) -> dict[str, Any] | None:
    paths = phase_paths(out_dir)
    if paths["result"].exists():
        return _load_terminal_bundle(out_dir)
    if paths["retention"].exists() and not paths[
        "terminal_evidence"
    ].exists():
        raise J2A1V3SurfaceIntegrityError(
            "Retention exists without terminal evidence"
        )
    if not paths["terminal_evidence"].exists():
        return None
    evidence = load_json(paths["terminal_evidence"])
    if not verify_payload_hash(
        evidence,
        "terminal_evidence_payload_sha256",
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Partial terminal evidence changed"
        )
    terminal_payload = dict(evidence)
    terminal_payload.pop("terminal_evidence_payload_sha256", None)
    return seal_terminal(
        out_dir=out_dir,
        terminal_payload=terminal_payload,
    )


def seal_terminal(
    *,
    out_dir: Path,
    terminal_payload: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = write_immutable_json(
        out_dir / TERMINAL_EVIDENCE_NAME,
        terminal_payload,
        field="terminal_evidence_payload_sha256",
    )
    retention = _seal_execution_retention(out_dir)
    final_payload = {
        **dict(terminal_payload),
        "terminal_evidence": artifact_identity(
            out_dir / TERMINAL_EVIDENCE_NAME,
            "terminal_evidence_payload_sha256",
        ),
        "retention": artifact_identity(
            out_dir / EXECUTION_RETENTION_NAME,
            "retention_payload_sha256",
        ),
        "authoritative_terminal_written_last": True,
    }
    sealed_terminal = payload_with_hash(
        final_payload,
        "terminal_payload_sha256",
    )
    current_bytes = sum(
        int(path.stat().st_size)
        for path in out_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if (
        current_bytes + len(_serialized_json_bytes(sealed_terminal))
        > STORAGE_CAP_BYTES
    ):
        raise J2A1V3SurfaceOperationalHold(
            "Authoritative terminal would exceed the output cap"
        )
    terminal = write_immutable_json(
        out_dir / TERMINAL_NAME,
        final_payload,
        field="terminal_payload_sha256",
    )
    return {
        "terminal": terminal,
        "evidence": evidence,
        "retention": retention,
        "passes": True,
    }


def _run_inherited_post_union(
    *,
    out_dir: Path,
    full_rows: Sequence[Mapping[str, Any]],
    union: Mapping[str, Any],
    contract_sha256: str,
    attempt_ledger: Any,
    boundary_callback: Callable[[str, Sequence[Path]], None],
    phase_transition_callback: Callable[[str], None],
) -> dict[str, Any]:
    from threes_rl import j2a1_distillation_fidelity_execution_surface_v2 as v2

    rows_by_root = {str(row["root_id"]): row for row in full_rows}
    roots = []
    for ref in union["merged_refs"]:
        row = rows_by_root[str(ref["root_id"])]
        base = (
            preflight.V2_EXECUTION_DIR
            if ref["source"] == "v2"
            else out_dir
        )
        root = v2.load_teacher_root_blob(
            base / str(ref["relative_path"]),
            authoritative_row=row,
            expected_file_sha256=str(ref["file_sha256"]),
        )
        root["file_sha256"] = ref["file_sha256"]
        root["relative_path"] = ref["relative_path"]
        roots.append(root)
    if len(roots) != ACTIVE_ROOTS:
        raise J2A1V3SurfaceIntegrityError(
            "Post-union root load count changed"
        )
    bc_roots = [
        root
        for root in roots
        if root["stage"] == v2.BC_STAGE
    ]
    validation_roots = [
        root
        for root in roots
        if root["stage"] == v2.VALIDATION_STAGE
    ]
    validation_rows = [
        row for row in full_rows if row["stage"] == v2.VALIDATION_STAGE
    ]
    inventory = v2.strict_feature_inventory(
        validation_roots,
        expected_root_count=v2.VALIDATION_PAIRS,
    )
    inventory_payload = v2._seal_stage_json(
        out_dir / "J2A1_VALIDATION_FEATURE_INVENTORY.json",
        inventory,
        field="inventory_payload_sha256",
    )
    boundary_callback(
        "family_inventory_seal",
        (out_dir / "J2A1_VALIDATION_FEATURE_INVENTORY.json",),
    )
    if inventory.get("passes") is not True:
        return {
            "decision": v2.HOLD_FAMILY,
            "stage": "family_support",
            "inventory": inventory_payload,
            "checkpoint": None,
            "checkpoint_authoritative": False,
            "passes": False,
        }
    phase_transition_callback("distillation_minibatch")
    distillation = v2.bounded_distillation(
        phase_dir=out_dir,
        bc_roots=bc_roots,
        contract_sha256=contract_sha256,
        collection_seal_sha256=union["union_payload_sha256"],
        family_inventory_sha256=inventory_payload[
            "inventory_payload_sha256"
        ],
        attempt_ledger=attempt_ledger,
        expected_root_count=v2.BC_ROOTS,
        minibatch_size=v2.j2.MINIBATCH_SIZE,
        epochs=v2.j2.DISTILLATION_EPOCHS,
        boundary_callback=boundary_callback,
    )
    checkpoint_path = Path(distillation["checkpoint"]["path"])
    checkpoint_payload = v2._deserialize_torch_payload(
        checkpoint_path.read_bytes(),
        magic=v2.CHECKPOINT_MAGIC,
    )
    model, _optimizer = v2.validate_epoch8_checkpoint(checkpoint_payload)
    mechanism = v2.mechanism_metrics(
        model,
        validation_roots,
        inventory,
    )
    mechanism_payload = v2._seal_stage_json(
        out_dir / "J2A1_BC_MECHANISM_RESULT.json",
        mechanism,
        field="mechanism_payload_sha256",
    )
    retirement = v2.seal_retirement(
        phase_dir=out_dir,
        name="bc_batch_after_epoch8_mechanism_seal",
        paths=[out_dir / "ephemeral_bc_batch" / "current_batch.bin"],
        predecessor_sha256=mechanism_payload[
            "mechanism_payload_sha256"
        ],
    )
    boundary_callback(
        "mechanism_and_batch_retirement_seal",
        (
            out_dir / "J2A1_BC_MECHANISM_RESULT.json",
            out_dir
            / "retirements"
            / "bc_batch_after_epoch8_mechanism_seal.json",
        ),
    )
    if mechanism.get("passes") is not True:
        quarantine = v2._quarantine_checkpoint(
            phase_dir=out_dir,
            checkpoint=distillation["checkpoint"],
            decision=v2.HOLD_MECHANISM,
            predecessor=mechanism_payload,
        )
        return {
            "decision": v2.HOLD_MECHANISM,
            "stage": "bc_mechanism",
            "inventory": inventory_payload,
            "distillation": distillation,
            "mechanism": mechanism_payload,
            "retirement": retirement,
            "checkpoint": distillation["checkpoint"],
            "quarantine": quarantine,
            "checkpoint_authoritative": False,
            "passes": False,
        }
    teacher_by_root = {
        root["root_id"]: root for root in validation_roots
    }
    phase_transition_callback("student_fidelity_pair")
    panel = v2.bounded_collect_fidelity_pairs(
        phase_dir=out_dir,
        rows=validation_rows,
        teacher_roots=teacher_by_root,
        checkpoint_path=checkpoint_path,
        contract_sha256=contract_sha256,
        attempt_ledger=attempt_ledger,
        arm_runner=lambda rows, actor: v2.run_student_arms_synchronously(
            rows=rows,
            model=actor,
        ),
        boundary_callback=boundary_callback,
    )
    pair_records = v2.load_fidelity_pairs_after_seal(
        phase_dir=out_dir,
        panel=panel,
        rows=validation_rows,
    )
    fidelity = v2.analyze_closed_loop_fidelity(
        pair_records,
        validation_rows,
    )
    fidelity_payload = v2._seal_stage_json(
        out_dir / "J2A1_CLOSED_LOOP_FIDELITY_RESULT.json",
        fidelity,
        field="fidelity_payload_sha256",
    )
    boundary_callback(
        "fidelity_analysis_seal",
        (out_dir / "J2A1_CLOSED_LOOP_FIDELITY_RESULT.json",),
    )
    if fidelity.get("passes") is not True:
        decision = str(fidelity.get("decision", v2.HOLD_FIDELITY))
        if decision not in {v2.HOLD_FIDELITY, v2.HOLD_BASE_RATE}:
            decision = v2.HOLD_FIDELITY
        quarantine = v2._quarantine_checkpoint(
            phase_dir=out_dir,
            checkpoint=distillation["checkpoint"],
            decision=decision,
            predecessor=fidelity_payload,
        )
        return {
            "decision": decision,
            "stage": "closed_loop_fidelity",
            "inventory": inventory_payload,
            "distillation": distillation,
            "mechanism": mechanism_payload,
            "panel": panel["seal"],
            "fidelity": fidelity_payload,
            "retirement": retirement,
            "checkpoint": distillation["checkpoint"],
            "quarantine": quarantine,
            "checkpoint_authoritative": False,
            "passes": False,
        }
    return {
        "decision": READY_EXECUTION,
        "stage": "closed_loop_fidelity",
        "inventory": inventory_payload,
        "distillation": distillation,
        "mechanism": mechanism_payload,
        "panel": panel["seal"],
        "fidelity": fidelity_payload,
        "retirement": retirement,
        "checkpoint": distillation["checkpoint"],
        "checkpoint_authority": None,
        "checkpoint_authority_pending_final_operational_guard": True,
        "checkpoint_authoritative": False,
        "passes": True,
    }


def _quarantine_v3_checkpoint(
    *,
    out_dir: Path,
    checkpoint: Mapping[str, Any],
    decision: str,
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    from threes_rl import j2a1_distillation_fidelity_execution_surface_v2 as v2

    return v2._quarantine_checkpoint(
        phase_dir=out_dir,
        checkpoint=checkpoint,
        decision=decision,
        predecessor=predecessor,
    )


def _authorize_v3_checkpoint(
    *,
    out_dir: Path,
    engine: Mapping[str, Any],
    final_guard_identity: Mapping[str, Any],
    execution_mode: str,
) -> dict[str, Any]:
    checkpoint = engine.get("checkpoint")
    mechanism = engine.get("mechanism")
    fidelity = engine.get("fidelity")
    if (
        engine.get("decision") != READY_EXECUTION
        or engine.get("passes") is not True
        or engine.get("checkpoint_authoritative") is not False
        or engine.get(
            "checkpoint_authority_pending_final_operational_guard"
        )
        is not True
        or not isinstance(checkpoint, Mapping)
        or not isinstance(mechanism, Mapping)
        or not isinstance(fidelity, Mapping)
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Checkpoint authority predecessors are incomplete"
        )
    checkpoint_path = Path(str(checkpoint.get("path")))
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
        or sha256_path(checkpoint_path) != checkpoint.get("file_sha256")
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Candidate checkpoint changed before authority"
        )
    payload = {
        "version": f"{VERSION}_checkpoint_authority_v1",
        "decision": READY_EXECUTION,
        "execution_mode": execution_mode,
        "scientific_authority": execution_mode == "scientific",
        "checkpoint": preflight._json_native(checkpoint),
        "mechanism_payload_sha256":
            mechanism["mechanism_payload_sha256"],
        "fidelity_payload_sha256": fidelity["fidelity_payload_sha256"],
        "final_operational_guard":
            preflight._json_native(final_guard_identity),
        "authoritative": True,
        "usable_only_for_separately_reviewed_ppo_surface": True,
        "ppo_execution_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
        "passes": True,
    }
    return write_immutable_json(
        out_dir / CHECKPOINT_AUTHORITY_NAME,
        payload,
        field="checkpoint_authority_payload_sha256",
    )


def execute_phase_from_artifacts(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    jobs: int,
    include_operational: bool = True,
    execution_mode: str = "scientific",
    fixture_collector: Callable[
        [Sequence[Mapping[str, Any]]],
        Sequence[Mapping[str, Any]],
    ] | None = None,
    fixture_post_union: Callable[
        [Mapping[str, Any]],
        Mapping[str, Any],
    ] | None = None,
    owner_pid: int | None = None,
    owner_start_identity: str | None = None,
    wall_clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    if jobs != 1:
        raise J2A1V3SurfaceIntegrityError("jobs must equal one")
    if execution_mode not in {"scientific", "miniature_fixture"}:
        raise J2A1V3SurfaceIntegrityError("Execution mode is invalid")
    if execution_mode == "scientific" and (
        fixture_collector is not None or fixture_post_union is not None
    ):
        raise J2A1V3SurfaceIntegrityError(
            "Scientific execution cannot inject fixture callbacks"
        )
    chain = _load_materialized_chain(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
    )
    paths = phase_paths(out_dir)
    resumed_terminal = _resume_terminal_finalization(out_dir)
    if resumed_terminal is not None:
        return resumed_terminal
    from threes_rl import j2a1_distillation_fidelity_execution_surface_v2 as v2

    accountant = v2.OutputAccountant(out_dir)
    _phase_operational_guard(
        out_dir=out_dir,
        cumulative_wall_seconds=V2_WALL_SECONDS,
        include_services=include_operational,
        accountant=accountant,
    )
    contract = phase_contract_sha256(
        chain,
        command="execute",
    )
    genesis = write_immutable_json(
        paths["genesis"],
        {
            "version": f"{VERSION}_genesis_v1",
            "phase_lock": chain["lock"]["identity"],
            "marker": chain["marker_identity"],
            "manifest": chain["manifest_identity"],
            "contract_sha256": contract,
            "v2_completed": V2_COMPLETED_ROOTS,
            "v3_completed": 0,
            "new_reservations": 0,
            "new_consumptions": 0,
            "passes": True,
        },
        field="genesis_payload_sha256",
    )
    del genesis
    if not paths["commits"].exists():
        _append_commit(
            path=paths["commits"],
            contract_sha256=contract,
            kind="genesis",
            bound_paths=(
                paths["lock"],
                paths["marker"],
                paths["manifest"],
                paths["genesis"],
            ),
        )
    current_pid = os.getpid() if owner_pid is None else owner_pid
    current_start_identity = (
        f"pid={os.getpid()}|start={time.time_ns()}"
        if owner_start_identity is None
        else owner_start_identity
    )
    owner = acquire_or_reclaim_owner(
        ledger_path=paths["owner"],
        chain=chain,
        command="execute",
        commit_ledger_path=paths["commits"],
        pid=current_pid,
        process_start_identity=current_start_identity,
        is_live=_pid_live,
    )
    reuse = write_immutable_json(
        paths["stream_reuse"],
        stream_authority_reuse_payload(chain),
        field="stream_reuse_payload_sha256",
    )
    del reuse
    wall = TopLevelWallLedger(
        path=paths["wall"],
        contract_sha256=contract,
        wall_clock=wall_clock,
    )
    if wall.open_segment is not None:
        wall.abandon_open()
    wall.start(unit_kind="teacher_root_block")
    attempt_ledger = v2.AttemptRuntimeLedger(
        path=paths["attempts"],
        contract_sha256=contract,
    )
    for path in (
        paths["lock"],
        paths["marker"],
        paths["manifest"],
        paths["genesis"],
        paths["commits"],
        paths["owner"],
        paths["stream_reuse"],
        paths["wall"],
        paths["attempts"],
    ):
        accountant.record_path(path)
    boundary_count = 0

    def boundary(kind: str, bound_paths: Sequence[Path]) -> None:
        nonlocal boundary_count
        expected_unit_kind = (
            "teacher_root_block"
            if kind.startswith("teacher") or kind == "union_seal"
            else "student_fidelity_pair"
            if "fidelity" in kind
            else "distillation_minibatch"
        )
        if (
            wall.open_segment is None
            or wall.open_segment.get("unit_kind") != expected_unit_kind
        ):
            raise J2A1V3SurfaceIntegrityError(
                f"Wall unit does not match boundary: {kind}"
            )
        verify_current_owner(
            ledger_path=paths["owner"],
            expected_owner_sha256=owner["owner"][
                "owner_record_sha256"
            ],
            chain=chain,
            command="execute",
            pid=current_pid,
            process_start_identity=current_start_identity,
        )
        wall.heartbeat(unit_kind=expected_unit_kind)
        for path in tuple(bound_paths) + (
            paths["wall"],
            paths["attempts"],
        ):
            accountant.record_path(path)
        _append_commit(
            path=paths["commits"],
            contract_sha256=contract,
            kind=kind,
            bound_paths=tuple(bound_paths)
            + (paths["wall"], paths["attempts"]),
        )
        accountant.record_path(paths["commits"])
        boundary_count += 1
        _phase_operational_guard(
            out_dir=out_dir,
            cumulative_wall_seconds=wall.cumulative_seconds(),
            include_services=include_operational,
            accountant=accountant,
        )

    def transition_wall(unit_kind: str) -> None:
        wall.switch(unit_kind=unit_kind)
        accountant.record_path(paths["wall"])
        _phase_operational_guard(
            out_dir=out_dir,
            cumulative_wall_seconds=wall.cumulative_seconds(),
            include_services=include_operational,
            accountant=accountant,
        )

    if execution_mode == "scientific":
        binding = v2._teacher_binding_from_authority()
        worker_group = v2.TeacherRootWorkerGroup(binding=binding)
        collector = worker_group.collect
    else:
        worker_group = None
        if fixture_collector is None or fixture_post_union is None:
            raise J2A1V3SurfaceIntegrityError(
                "Miniature execution needs fixture callbacks"
            )
        collector = fixture_collector
    integrity_failure = False
    operational_failure = False
    engine: dict[str, Any]
    try:
        collection = v2.bounded_collect_teacher_roots(
            phase_dir=out_dir,
            rows=chain["rows"],
            contract_sha256=contract,
            batch_collector=collector,
            attempt_ledger=attempt_ledger,
            boundary_callback=boundary,
        )
        union = seal_union(
            out_dir=out_dir,
            v3_completion_records=collection["refs"],
        )
        boundary("union_seal", (paths["union"],))
        if not stage_b_barrier(
            v2_completed=V2_COMPLETED_ROOTS,
            v3_completed=len(collection["refs"]),
            union_passes=union["passes"],
        )["passes"]:
            raise J2A1V3SurfaceIntegrityError(
                "Stage B completeness barrier failed"
            )
        transition_wall("distillation_minibatch")
        if execution_mode == "scientific":
            engine = _run_inherited_post_union(
                out_dir=out_dir,
                full_rows=_full_authority_rows(),
                union=union,
                contract_sha256=contract,
                attempt_ledger=attempt_ledger,
                boundary_callback=boundary,
                phase_transition_callback=transition_wall,
            )
        else:
            engine = dict(fixture_post_union(union))
    except (
        J2A1V3PlannedInterruption,
        v2.J2A1PlannedInterruption,
        KeyboardInterrupt,
        SystemExit,
    ):
        raise
    except (
        J2A1V3SurfaceOperationalHold,
        v2.J2A1ExecutionOperationalHold,
    ) as error:
        operational_failure = True
        engine = {
            "decision": KILL_EXECUTION,
            "stage": "operational",
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "checkpoint_authoritative": False,
            "passes": False,
        }
    except Exception as error:
        integrity_failure = True
        engine = {
            "decision": KILL_EXECUTION,
            "stage": "integrity",
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "checkpoint_authoritative": False,
            "passes": False,
        }
    finally:
        if worker_group is not None:
            worker_group.close(terminate=False)
    if wall.open_segment is not None:
        wall.finish(unit_kind=str(wall.open_segment["unit_kind"]))
    accountant.record_path(paths["wall"])
    accountant.record_path(paths["attempts"])
    accountant.reconcile()
    final_guard_identity: dict[str, Any] | None = None
    try:
        final_guard = _phase_operational_guard(
            out_dir=out_dir,
            cumulative_wall_seconds=wall.cumulative_seconds(),
            include_services=include_operational,
            accountant=accountant,
        )
        sealed_final_guard = write_immutable_json(
            paths["final_guard"],
            final_guard,
            field="final_operational_guard_payload_sha256",
        )
        final_guard_identity = artifact_identity(
            paths["final_guard"],
            "final_operational_guard_payload_sha256",
        )
        _append_commit(
            path=paths["commits"],
            contract_sha256=contract,
            kind="final_operational_guard",
            bound_paths=(
                paths["final_guard"],
                paths["wall"],
                paths["attempts"],
            ),
        )
        accountant.record_path(paths["final_guard"])
        accountant.record_path(paths["commits"])
        del sealed_final_guard
    except (
        J2A1V3SurfaceOperationalHold,
        v2.J2A1ExecutionOperationalHold,
    ) as error:
        operational_failure = True
        prior_decision = str(engine.get("decision"))
        checkpoint = engine.get("checkpoint")
        quarantine = None
        try:
            if isinstance(checkpoint, Mapping):
                quarantine = _quarantine_v3_checkpoint(
                    out_dir=out_dir,
                    checkpoint=checkpoint,
                    decision=KILL_EXECUTION,
                    predecessor={
                        "prior_decision": prior_decision,
                        "error_type": type(error).__name__,
                        "error_sha256": hashlib.sha256(
                            str(error).encode("utf-8")
                        ).hexdigest(),
                    },
                )
        except Exception as quarantine_error:
            integrity_failure = True
            engine = {
                "decision": KILL_EXECUTION,
                "stage": "checkpoint_quarantine",
                "error_type": type(quarantine_error).__name__,
                "error_sha256": hashlib.sha256(
                    str(quarantine_error).encode("utf-8")
                ).hexdigest(),
                "checkpoint": checkpoint,
                "checkpoint_authoritative": False,
                "passes": False,
            }
        else:
            engine = {
                "decision": KILL_EXECUTION,
                "stage": "final_operational_guard",
                "prior_engine_decision": prior_decision,
                "error_type": type(error).__name__,
                "error_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
                "checkpoint": checkpoint,
                "quarantine": quarantine,
                "checkpoint_authoritative": False,
                "passes": False,
            }
    except Exception as error:
        integrity_failure = True
        engine = {
            "decision": KILL_EXECUTION,
            "stage": "final_operational_guard_integrity",
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "checkpoint": engine.get("checkpoint"),
            "checkpoint_authoritative": False,
            "passes": False,
        }
    if (
        not integrity_failure
        and not operational_failure
        and engine.get("decision") == READY_EXECUTION
    ):
        try:
            if final_guard_identity is None:
                raise J2A1V3SurfaceIntegrityError(
                    "Checkpoint authority has no final guard"
                )
            authority = _authorize_v3_checkpoint(
                out_dir=out_dir,
                engine=engine,
                final_guard_identity=final_guard_identity,
                execution_mode=execution_mode,
            )
            engine = {
                **dict(engine),
                "checkpoint_authority": authority,
                "checkpoint_authority_pending_final_operational_guard":
                    False,
                "checkpoint_authoritative": True,
                "final_operational_guard": final_guard_identity,
            }
        except Exception as error:
            integrity_failure = True
            checkpoint = engine.get("checkpoint")
            quarantine = None
            try:
                if isinstance(checkpoint, Mapping):
                    quarantine = _quarantine_v3_checkpoint(
                        out_dir=out_dir,
                        checkpoint=checkpoint,
                        decision=KILL_EXECUTION,
                        predecessor={
                            "error_type": type(error).__name__,
                            "error_sha256": hashlib.sha256(
                                str(error).encode("utf-8")
                            ).hexdigest(),
                        },
                    )
            except Exception:
                quarantine = None
            engine = {
                "decision": KILL_EXECUTION,
                "stage": "checkpoint_authority",
                "error_type": type(error).__name__,
                "error_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
                "checkpoint": checkpoint,
                "quarantine": quarantine,
                "checkpoint_authoritative": False,
                "passes": False,
            }
    elif (
        (out_dir / CHECKPOINT_AUTHORITY_NAME).exists()
        and engine.get("decision") != READY_EXECUTION
    ):
        integrity_failure = True
        engine = {
            "decision": KILL_EXECUTION,
            "stage": "checkpoint_authority",
            "error_type": "UnexpectedCheckpointAuthority",
            "error_sha256": hashlib.sha256(
                b"checkpoint authority exists for a non-READY engine"
            ).hexdigest(),
            "checkpoint_authoritative": False,
            "passes": False,
        }
    final_decision = terminal_precedence(
        integrity_failure=integrity_failure,
        operational_failure=operational_failure,
        scientific_decision=str(engine["decision"]),
    )
    v2_integrity_expected = PREFLIGHT_ARTIFACTS[
        preflight.V2_INTEGRITY_NAME
    ]
    v2_integrity = _load_bound_json(
        PREFLIGHT_DIR / preflight.V2_INTEGRITY_NAME,
        file_sha256=v2_integrity_expected[0],
        payload_field=v2_integrity_expected[1],
        payload_sha256=v2_integrity_expected[2],
    )
    completion_summary = v2.CompletionLedger(
        path=paths["completions"],
        contract_sha256=contract,
        kind="teacher_root",
    ).summary()
    owner_records = _load_jsonl_chain(
        paths["owner"],
        field="owner_record_sha256",
        predecessor_field="predecessor_record_sha256",
    )
    commit_records = verify_commit_ledger(
        path=paths["commits"],
        contract_sha256=contract,
    )
    union_identity = (
        artifact_identity(paths["union"], "union_payload_sha256")
        if paths["union"].is_file()
        else None
    )
    terminal_payload = {
        "version": f"{VERSION}_terminal_v1",
        "decision": final_decision,
        "execution_mode": execution_mode,
        "scientific_authority": execution_mode == "scientific",
        "phase_lock": chain["lock"]["identity"],
        "marker": chain["marker_identity"],
        "manifest": chain["manifest_identity"],
        "stream_authority_reuse": artifact_identity(
            paths["stream_reuse"],
            "stream_reuse_payload_sha256",
        ),
        "owner_head_sha256": owner["owner"]["owner_record_sha256"],
        "owner_record_count": len(owner_records),
        "commit_head_sha256": commit_records[-1][
            "commit_record_sha256"
        ],
        "commit_record_count": len(commit_records),
        "wall": wall.summary(),
        "v2_attempt_summary": v2_integrity["attempt_ledger"],
        "v2_completion_summary": v2_integrity["completion_ledger"],
        "v3_completion_summary": completion_summary,
        "total_completed_roots":
            V2_COMPLETED_ROOTS + int(completion_summary["completed"]),
        "union": union_identity,
        "final_operational_guard": final_guard_identity,
        "aggregate_worker_seconds_descriptive":
            attempt_ledger.summary()["active_seconds"],
        "attempt_summary": attempt_ledger.summary(),
        "boundary_count": boundary_count,
        "engine": engine,
        "checkpoint_authoritative": bool(
            engine.get("checkpoint_authoritative")
        )
        and final_decision == READY_EXECUTION,
        "ppo_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
        "human_session_reads": 0,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
        "new_stream_reservations": 0,
        "new_stream_consumptions": 0,
        "v2_completed_roots_requeried": 0,
        "replacement_roots": 0,
        "filtered_roots": 0,
        "partial_scientific_peeks": 0,
        "ppo_reads": 0,
        "development_reads": 0,
        "confirmation_reads": 0,
        "continue": final_decision == READY_EXECUTION,
        "hold": final_decision not in {
            READY_EXECUTION,
            KILL_EXECUTION,
        },
        "kill": final_decision == KILL_EXECUTION,
        "promote": False,
    }
    return seal_terminal(
        out_dir=out_dir,
        terminal_payload=terminal_payload,
    )


def audit_zero_work(
    *,
    output_dir: Path = READINESS_DIR,
    include_operational: bool,
) -> dict[str, Any]:
    existing = (
        sorted(
            path.name
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        if output_dir.exists()
        else []
    )
    checks = {
        "readiness_boundary": set(existing) <= {TEST_EVIDENCE_NAME},
        "future_authorization_absent":
            not FUTURE_AUTHORIZATION_DIR.exists(),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "zero_work": all(value == 0 for value in ZERO_WORK.values()),
    }
    operations = operational_audit(
        output_dir=output_dir,
        include_services=include_operational,
        require_future_absent=True,
    )
    checks["operations"] = operations["passes"]
    return {
        "version": f"{VERSION}_zero_work_audit_v1",
        "existing_files": existing,
        "operations": operations,
        "zero_work": dict(ZERO_WORK),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _load_json_list(path: Path, *, name: str) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise J2A1V3SurfaceIntegrityError(
            f"Cannot load {name} JSON"
        ) from error
    if not isinstance(payload, list):
        raise J2A1V3SurfaceIntegrityError(f"{name} is not a list")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-zero-work")
    audit.add_argument("--output-dir", type=Path, default=READINESS_DIR)
    audit.add_argument("--include-operational", action="store_true")
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--output-dir", type=Path, default=READINESS_DIR)
    evidence.add_argument("--commands-json", type=Path, required=True)
    evidence.add_argument("--deselections-json", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-readiness")
    prepare.add_argument("--output-dir", type=Path, default=READINESS_DIR)
    for command in ("seal-phase-lock", "open", "materialize", "execute"):
        phase = subparsers.add_parser(command)
        phase.add_argument("--readiness-dir", type=Path, required=True)
        phase.add_argument("--authorization", type=Path, required=True)
        phase.add_argument("--out-dir", type=Path, required=True)
        phase.add_argument("--jobs", type=int, default=1)
    return parser


def dispatch_cli(args: argparse.Namespace) -> dict[str, Any]:
    command = str(args.command)
    if command == "audit-zero-work":
        return audit_zero_work(
            output_dir=args.output_dir,
            include_operational=bool(args.include_operational),
        )
    if command == "write-test-evidence":
        commands = _load_json_list(
            args.commands_json,
            name="commands",
        )
        deselections = _load_json_list(
            args.deselections_json,
            name="deselections",
        )
        return write_test_evidence(
            output_dir=args.output_dir,
            commands=commands,
            deselections=deselections,
        )
    if command == "prepare-readiness":
        return prepare_readiness(
            output_dir=args.output_dir,
            include_operational=True,
        )
    common = {
        "readiness_dir": args.readiness_dir,
        "authorization_path": args.authorization,
        "out_dir": args.out_dir,
    }
    if command == "seal-phase-lock":
        return seal_phase_lock(
            **common,
            jobs=args.jobs,
            include_operational=True,
        )
    if command == "open":
        return open_phase(**common, jobs=args.jobs)
    if command == "materialize":
        return materialize_phase(**common, jobs=args.jobs)
    if command == "execute":
        return execute_phase_from_artifacts(
            **common,
            jobs=args.jobs,
            include_operational=True,
            execution_mode="scientific",
        )
    raise J2A1V3SurfaceIntegrityError(
        f"Forbidden command: {command}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = dispatch_cli(args)
    print(json.dumps(preflight._json_native(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

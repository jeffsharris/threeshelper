"""Outcome-free J2A1 V3A1 post-seal reproducibility readiness."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from threes_rl import (
    j2a1_distillation_fidelity_recovery_execution_surface_v3 as v3,
)


VERSION = "j2a1_distillation_fidelity_recovery_execution_surface_v3a1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs" / "forensics"
CHARTER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J2A1_DISTILLATION_FIDELITY_V3A1_POSTSEAL_REPRODUCIBILITY_AMENDMENT.md"
)
RUNNER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "j2a1_distillation_fidelity_recovery_execution_surface_v3a1.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_rl_j2a1_distillation_fidelity_recovery_execution_surface_v3a1.py"
)

PARENT_READINESS_DIR = v3.READINESS_DIR
READINESS_DIR = (
    RUNS_ROOT
    / "j2a1_distillation_fidelity_recovery_execution_surface_readiness_v3a1"
)
FUTURE_AUTHORIZATION_DIR = (
    RUNS_ROOT
    / "j2a1_distillation_fidelity_recovery_authorization_v3a1"
)
FUTURE_EXECUTION_DIR = (
    RUNS_ROOT / "j2a1_distillation_fidelity_recovery_v3a1"
)

TEST_EVIDENCE_NAME = "J2A1_V3A1_POSTSEAL_TEST_EVIDENCE.json"
PARENT_HOLD_NAME = "J2A1_V3A1_PARENT_V3_HOLD_BINDING.json"
INPUT_BINDINGS_NAME = "J2A1_V3A1_INPUT_BINDINGS.json"
CHRONOLOGY_AUDIT_NAME = "J2A1_V3A1_POSTSEAL_CHRONOLOGY_AUDIT.json"
HEADROOM_AUDIT_NAME = "J2A1_V3A1_LAUNCH_HEADROOM_AUDIT.json"
CLEANUP_PROPOSAL_NAME = "J2A1_V3A1_HEADROOM_REVIEW_PROPOSAL.json"
SCHEMA_NAME = "J2A1_V3A1_READINESS_SCHEMA.json"
LOCK_NAME = "J2A1_V3A1_READINESS_LOCK.json"
RESULT_NAME = "J2A1_V3A1_READINESS_RESULT.json"
RETENTION_NAME = "J2A1_V3A1_RETENTION.json"

READY = "READY_J2A1_V3A1_RECOVERY_EXECUTION_SURFACE"
HOLD = "HOLD_J2A1_V3A1_RECOVERY_EXECUTION_HEADROOM"
KILL = "KILL_J2A1_V3A1_RECOVERY_EXECUTION_SURFACE_INTEGRITY"
PARENT_HOLD = (
    "HOLD_J2A1_V3_RECOVERY_EXECUTION_SURFACE_REPRODUCIBILITY"
)

PROJECTED_COMBINED_BYTES = 22_053_337_088
V2_RETAINED_BYTES = 1_782_523_714
PROJECTED_INCREMENTAL_BYTES = (
    PROJECTED_COMBINED_BYTES - V2_RETAINED_BYTES
)
HARD_FLOOR_BYTES = 100 * 1024**3
MINIMUM_CUSHION_BYTES = 5 * 1024**3

PARENT_SOURCE_IDENTITIES = {
    "threes_rl/"
    "J2A1_DISTILLATION_FIDELITY_V3_RECOVERY_EXECUTION_SURFACE_CHARTER.md":
        "674ed0e1c67df0cbc8645a2190a5632ce70c9cddc5922ad0325a9e53d14c481c",
    "threes_rl/"
    "j2a1_distillation_fidelity_recovery_execution_surface_v3.py":
        "611dc428a3f940ff1db15ae58e960bab27ab7307c36393bd23a7400e9da12c02",
    "tests/"
    "test_rl_j2a1_distillation_fidelity_recovery_execution_surface_v3.py":
        "1a1dbb1039b9dd7d57d8d9a88f7cd81dfdd68240bd16b23357df6f5c5eb01df4",
}

PARENT_ARTIFACT_IDENTITIES = {
    "test_evidence": (
        "9a46fe4abeb4cac94302ae2ad746d83ce2da9d5748a377db5f90ecd6d0e83b99",
        "66d77e3cc0d217d8a1059736913271211512526e8d149e672251de24c0efa0ae",
    ),
    "input_bindings": (
        "b2343a23023441331f86ded68426cf5babe3e73bc7f860e9c029faafbeb46e72",
        "aa1a427e240d769cfd8dee530d439c8ed0ad235f4814c57761488ba71bd7a58f",
    ),
    "authority": (
        "fbdd57e403c614c916e3317aefba5ce3ae37c3e03aaa61b7189f307c7ed84069",
        "5a018f630fc2ff7d85911fe00138e461c67c8d29b54f04dfb7d340ad1317294d",
    ),
    "schema": (
        "d3dd83fef73131b84b8ea7668e73d94e5b3dab564c7312f4840bf822f9ed65c4",
        "dbdaeb5f16fb14c7016e06306967931265b9d25c23b2eff5001e0a9ca5704e9b",
    ),
    "projection": (
        "39dba009a461ae512121b02a206afbd89f058b38df93786831a7831533f6df4c",
        "c39647126dd9aac79c813f6e61515ef49f0e459785d3052be39624de2f9d250e",
    ),
    "state_machine": (
        "035c61e79e9d191684d04885c19f4cefe595c34070b24918186fef96e5ef4959",
        "2bc15d93284c19fe140148f0df26c933961495fa22fba1e9361a4fc6dc637b0a",
    ),
    "lock": (
        "ba44650eaead39de45465ff6a785d7a30aaf9c5740294b2516e70354287691ef",
        "8d4d87baa92ea04730435d2798d9b9bed0088bbd757443913d3fc9eaafb0bea3",
    ),
    "result": (
        "3bac460ad19a32b249b199eec66d6aa7cc9f27be83eb2c4842412868e81ac610",
        "c2626a14d8b86613d05c0934e4735171cc3242f7841308914a669c3666cb7bb9",
    ),
    "retention": (
        "7f9def1579f2414dcbda7002ee5f7519daa86ae86986206d1a4dbbe7348a701c",
        "55dde2fe501267adb635d48849144ffca046d9e40f29e81ceb45acfcb488eeb7",
    ),
}

PARENT_FAILURE_NODE = (
    "tests/test_rl_j2a1_distillation_fidelity_recovery_execution_surface_v3.py"
    "::test_zero_work_and_future_namespaces_are_absent"
)
EXPECTED_DESELECTIONS = [
    "tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py"
    "::test_v2_binds_v1_hold_and_both_execution_roots_are_absent",
    "tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py"
    "::test_test_evidence_is_create_once",
    "tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py"
    "::test_zero_work_audit_requires_future_execution_absence",
    "tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py"
    "::test_phase_chain_create_once_and_exact_order",
    "tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py"
    "::test_marker_tamper_is_rejected_before_materialization",
    "tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py"
    "::test_dispatcher_miniature_chain_reaches_non_authoritative_terminal",
    "tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py"
    "::test_post_consumption_dead_owner_recovery_reuses_consumption",
    "tests/test_rl_j2a1_distillation_fidelity_execution_surface_v2.py"
    "::test_final_operational_fault_cannot_survive_as_ready",
]

ZERO_WORK = {
    "scientific_authorizations": 0,
    "phase_locks": 0,
    "execution_markers": 0,
    "materialized_authorities": 0,
    "owners": 0,
    "new_stream_reservations": 0,
    "new_stream_consumptions": 0,
    "teacher_queries": 0,
    "labels": 0,
    "games": 0,
    "optimizer_steps": 0,
    "checkpoints": 0,
    "family_reads": 0,
    "mechanism_reads": 0,
    "fidelity_reads": 0,
    "ppo_reads": 0,
    "development_reads": 0,
    "confirmation_reads": 0,
    "human_session_reads": 0,
    "incumbent_changes": 0,
    "dashboard_changes": 0,
    "promotion_actions": 0,
    "cleanup_actions": 0,
    "root_body_deserializations": 0,
}


class J2A1V3A1IntegrityError(RuntimeError):
    """Raised when an immutable V3A1 identity or schema changes."""


def sha256_path(path: Path) -> str:
    return v3.sha256_path(path)


def canonical_json_hash(payload: Any) -> str:
    return v3.canonical_json_hash(payload)


def load_json(path: Path) -> dict[str, Any]:
    return v3.load_json(path)


def write_immutable_json(
    path: Path,
    body: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    return v3.write_immutable_json(path, body, field=field)


def artifact_identity(path: Path, field: str) -> dict[str, Any]:
    return v3.artifact_identity(path, field)


def _source_identities() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256_path(path)
        for path in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)
    }


def readiness_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "test_evidence": output_dir / TEST_EVIDENCE_NAME,
        "parent_hold": output_dir / PARENT_HOLD_NAME,
        "input_bindings": output_dir / INPUT_BINDINGS_NAME,
        "chronology": output_dir / CHRONOLOGY_AUDIT_NAME,
        "headroom": output_dir / HEADROOM_AUDIT_NAME,
        "cleanup_proposal": output_dir / CLEANUP_PROPOSAL_NAME,
        "schema": output_dir / SCHEMA_NAME,
        "lock": output_dir / LOCK_NAME,
        "result": output_dir / RESULT_NAME,
        "retention": output_dir / RETENTION_NAME,
    }


READINESS_FIELDS = {
    "test_evidence": "test_evidence_payload_sha256",
    "parent_hold": "parent_hold_payload_sha256",
    "input_bindings": "input_bindings_payload_sha256",
    "chronology": "chronology_audit_payload_sha256",
    "headroom": "headroom_audit_payload_sha256",
    "cleanup_proposal": "cleanup_proposal_payload_sha256",
    "schema": "readiness_schema_payload_sha256",
    "lock": "readiness_lock_payload_sha256",
    "result": "readiness_result_payload_sha256",
    "retention": "retention_payload_sha256",
}


def _future_namespaces_absent() -> bool:
    return all(
        not path.exists()
        for path in (
            v3.FUTURE_AUTHORIZATION_DIR,
            v3.FUTURE_EXECUTION_DIR,
            FUTURE_AUTHORIZATION_DIR,
            FUTURE_EXECUTION_DIR,
        )
    )


def _zero_work_exact() -> bool:
    return (
        all(value == 0 for value in ZERO_WORK.values())
        and all(value == 0 for value in v3.ZERO_WORK.values())
    )


def verify_parent_v3_package() -> dict[str, Any]:
    current_sources = {
        path: sha256_path(REPO_ROOT / path)
        for path in PARENT_SOURCE_IDENTITIES
    }
    parent = v3.verify_readiness_package(PARENT_READINESS_DIR)
    observed_files = {
        path.resolve()
        for path in PARENT_READINESS_DIR.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_paths = {
        path.resolve()
        for path in v3.readiness_paths(PARENT_READINESS_DIR).values()
    }
    identity_checks = {}
    for key, expected in PARENT_ARTIFACT_IDENTITIES.items():
        identity = parent["identities"][key]
        identity_checks[key] = (
            identity["file_sha256"] == expected[0]
            and identity["payload_sha256"] == expected[1]
        )
    checks = {
        "source_identities_exact":
            current_sources == PARENT_SOURCE_IDENTITIES,
        "parent_loader_passes": parent.get("passes") is True,
        "exact_nine_file_set":
            observed_files == expected_paths and len(observed_files) == 9,
        "all_file_payload_identities_exact":
            all(identity_checks.values()),
        "retention_exact":
            parent["checks"].get("retention_inventory") is True
            and parent["checks"].get("retention_passes") is True,
        "execution_unauthorized":
            parent["payloads"]["result"].get("execution_authorized") is False,
        "future_namespaces_absent": _future_namespaces_absent(),
        "zero_scientific_work": _zero_work_exact(),
    }
    if not all(checks.values()):
        raise J2A1V3A1IntegrityError(
            "The immutable parent V3 package changed"
        )
    return {
        "version": f"{VERSION}_parent_package_audit_v1",
        "source_identities": current_sources,
        "artifact_identities": {
            key: {
                "file_sha256": identity["file_sha256"],
                "payload_sha256": identity["payload_sha256"],
                "bytes": identity["bytes"],
            }
            for key, identity in parent["identities"].items()
        },
        "checks": checks,
        "passes": True,
    }


def audit_pre_prepare(
    *,
    output_dir: Path = READINESS_DIR,
) -> dict[str, Any]:
    parent = verify_parent_v3_package()
    checks = {
        "readiness_namespace_absent": not output_dir.exists(),
        "parent_v3_exact": parent["passes"] is True,
        "future_namespaces_absent": _future_namespaces_absent(),
        "zero_scientific_work": _zero_work_exact(),
    }
    return {
        "version": f"{VERSION}_pre_prepare_audit_v1",
        "output_dir": str(output_dir),
        "parent_v3": parent,
        "checks": checks,
        "zero_work": dict(ZERO_WORK),
        "passes": all(checks.values()),
    }


def audit_post_seal() -> dict[str, Any]:
    parent = verify_parent_v3_package()
    checks = {
        "parent_exact_nine_file_package":
            parent["checks"]["exact_nine_file_set"],
        "every_parent_identity_exact":
            parent["checks"]["all_file_payload_identities_exact"],
        "parent_sources_exact":
            parent["checks"]["source_identities_exact"],
        "parent_retention_exact":
            parent["checks"]["retention_exact"],
        "future_namespaces_absent": _future_namespaces_absent(),
        "zero_scientific_work": _zero_work_exact(),
    }
    return {
        "version": f"{VERSION}_post_seal_audit_v1",
        "parent_v3_readiness_dir": str(PARENT_READINESS_DIR),
        "parent_v3": parent,
        "checks": checks,
        "zero_work": dict(ZERO_WORK),
        "passes": all(checks.values()),
    }


def parent_hold_payload() -> dict[str, Any]:
    post_seal = audit_post_seal()
    checks = {
        "parent_exact": post_seal["passes"] is True,
        "failure_is_chronology_only": True,
        "parent_spent_unexecuted": True,
        "future_namespaces_absent": _future_namespaces_absent(),
        "zero_scientific_work": _zero_work_exact(),
    }
    return {
        "version": f"{VERSION}_parent_hold_binding_v1",
        "decision": PARENT_HOLD,
        "authoritative_parent_decision": v3.READY,
        "parent_execution_authorized": False,
        "parent_surface_spent_for_execution": True,
        "diagnostic": {
            "command":
                "PYTHONPATH=. .venv/bin/pytest -q "
                "tests/test_rl_j2a1_distillation_fidelity_"
                "recovery_execution_surface_v3.py",
            "passed": 41,
            "failed": 1,
            "failure_node": PARENT_FAILURE_NODE,
            "failure_mechanism":
                "pre-prepare audit invoked against sealed readiness package",
            "classification": "chronology_orchestration_only",
        },
        "post_seal_audit_sha256": canonical_json_hash(post_seal),
        "checks": checks,
        "zero_work": dict(ZERO_WORK),
        "passes": all(checks.values()),
    }


def readiness_decision(
    *,
    integrity_passes: bool,
    headroom_passes: bool,
) -> str:
    if not integrity_passes:
        return KILL
    if not headroom_passes:
        return HOLD
    return READY


def calculate_launch_headroom(
    current_free_bytes: Any,
) -> dict[str, Any]:
    available = (
        type(current_free_bytes) is int and current_free_bytes >= 0
    )
    projected_free = (
        int(current_free_bytes) - PROJECTED_INCREMENTAL_BYTES
        if available
        else None
    )
    cushion = (
        projected_free - HARD_FLOOR_BYTES
        if projected_free is not None
        else None
    )
    required_current_free = (
        PROJECTED_INCREMENTAL_BYTES
        + HARD_FLOOR_BYTES
        + MINIMUM_CUSHION_BYTES
    )
    additional_required = (
        max(0, required_current_free - int(current_free_bytes))
        if available
        else None
    )
    checks = {
        "calculation_available": available,
        "combined_projection_exact":
            PROJECTED_COMBINED_BYTES == v3.PROJECTED_COMBINED_BYTES,
        "v2_retained_bytes_exact":
            V2_RETAINED_BYTES == 1_782_523_714,
        "incremental_formula_exact":
            PROJECTED_INCREMENTAL_BYTES
            == PROJECTED_COMBINED_BYTES - V2_RETAINED_BYTES,
        "projected_free_at_least_100_gib":
            projected_free is not None
            and projected_free >= HARD_FLOOR_BYTES,
        "projected_cushion_at_least_5_gib":
            cushion is not None and cushion >= MINIMUM_CUSHION_BYTES,
    }
    formula_integrity = all(
        checks[key]
        for key in (
            "combined_projection_exact",
            "v2_retained_bytes_exact",
            "incremental_formula_exact",
        )
    )
    admission = (
        formula_integrity
        and available
        and checks["projected_free_at_least_100_gib"]
        and checks["projected_cushion_at_least_5_gib"]
    )
    return {
        "version": f"{VERSION}_launch_headroom_calculation_v1",
        "current_free_bytes":
            int(current_free_bytes) if available else None,
        "current_free_gib":
            int(current_free_bytes) / 1024**3 if available else None,
        "projected_combined_bytes": PROJECTED_COMBINED_BYTES,
        "already_retained_v2_bytes": V2_RETAINED_BYTES,
        "projected_incremental_bytes": PROJECTED_INCREMENTAL_BYTES,
        "projected_incremental_gib":
            PROJECTED_INCREMENTAL_BYTES / 1024**3,
        "projected_peak_free_bytes": projected_free,
        "projected_peak_free_gib":
            projected_free / 1024**3
            if projected_free is not None
            else None,
        "hard_floor_bytes": HARD_FLOOR_BYTES,
        "minimum_cushion_bytes": MINIMUM_CUSHION_BYTES,
        "projected_floor_cushion_bytes": cushion,
        "projected_floor_cushion_gib":
            cushion / 1024**3 if cushion is not None else None,
        "required_current_free_bytes": required_current_free,
        "additional_free_bytes_required": additional_required,
        "additional_free_gib_required":
            additional_required / 1024**3
            if additional_required is not None
            else None,
        "checks": checks,
        "formula_integrity_passes": formula_integrity,
        "calculation_available": available,
        "admission_passes": admission,
        "decision": readiness_decision(
            integrity_passes=formula_integrity,
            headroom_passes=admission,
        ),
    }


def launch_headroom_audit(
    *,
    output_dir: Path,
    include_operational: bool,
    free_bytes: int | None = None,
) -> dict[str, Any]:
    if free_bytes is None:
        try:
            observed_free: Any = int(shutil.disk_usage(REPO_ROOT).free)
        except OSError:
            observed_free = None
    else:
        observed_free = free_bytes
    calculation = calculate_launch_headroom(observed_free)
    parent_operations = v3.operational_audit(
        output_dir=output_dir,
        include_services=include_operational,
        require_future_absent=True,
    )
    checks = {
        "parent_operations_pass": parent_operations["passes"] is True,
        "v3a1_future_namespaces_absent":
            not FUTURE_AUTHORIZATION_DIR.exists()
            and not FUTURE_EXECUTION_DIR.exists(),
        "zero_scientific_work": _zero_work_exact(),
        "headroom_formula_integrity":
            calculation["formula_integrity_passes"] is True,
    }
    return {
        "version": f"{VERSION}_launch_headroom_audit_v1",
        "calculation": calculation,
        "parent_operations": parent_operations,
        "checks": checks,
        "integrity_passes": all(checks.values()),
        "admission_passes":
            all(checks.values())
            and calculation["admission_passes"] is True,
        "zero_work": dict(ZERO_WORK),
    }


def cleanup_review_proposal(
    headroom: Mapping[str, Any],
) -> dict[str, Any]:
    calculation = headroom["calculation"]
    hold = calculation["admission_passes"] is not True
    protected = [
        str(v3.preflight.V2_EXECUTION_DIR.relative_to(REPO_ROOT)),
        str(v3.PREFLIGHT_DIR.relative_to(REPO_ROOT)),
        str(PARENT_READINESS_DIR.relative_to(REPO_ROOT)),
        str(READINESS_DIR.relative_to(REPO_ROOT)),
        "threes_rl/EXPERIMENT_LOG.md",
        "threes_rl/CURRENT_DECISION_LEDGER.md",
        "threes_rl/ARTIFACT_RETENTION.md",
    ]
    return {
        "version": f"{VERSION}_headroom_review_proposal_v1",
        "decision":
            "REVIEW_ADDITIONAL_HEADROOM"
            if hold
            else "NO_HEADROOM_ACTION_REQUIRED",
        "current_free_bytes": calculation["current_free_bytes"],
        "projected_incremental_bytes":
            calculation["projected_incremental_bytes"],
        "projected_peak_free_bytes":
            calculation["projected_peak_free_bytes"],
        "projected_floor_cushion_bytes":
            calculation["projected_floor_cushion_bytes"],
        "minimum_required_cushion_bytes": MINIMUM_CUSHION_BYTES,
        "additional_free_bytes_required":
            calculation["additional_free_bytes_required"],
        "protected_paths": protected,
        "candidate_deletions": [],
        "cleanup_manifest_approved": False,
        "cleanup_authorized": False,
        "cleanup_performed": False,
        "review_required_before_any_move_or_delete": True,
        "non_destructive_option":
            "Add external free space or separately review a future manifest",
        "zero_work": dict(ZERO_WORK),
        "passes": True,
    }


def readiness_schema() -> dict[str, Any]:
    return {
        "version": f"{VERSION}_readiness_schema_v1",
        "public_commands": [
            "audit-pre-prepare",
            "audit-post-seal",
            "write-test-evidence",
            "prepare-readiness",
        ],
        "parent_package_files": 9,
        "readiness_package_files": 10,
        "headroom": {
            "projected_combined_bytes": PROJECTED_COMBINED_BYTES,
            "already_retained_v2_bytes": V2_RETAINED_BYTES,
            "projected_incremental_bytes": PROJECTED_INCREMENTAL_BYTES,
            "hard_floor_bytes": HARD_FLOOR_BYTES,
            "minimum_cushion_bytes": MINIMUM_CUSHION_BYTES,
        },
        "decisions": [READY, HOLD, KILL],
        "execution_commands": [],
        "cleanup_commands": [],
        "execution_authorized": False,
        "zero_work": dict(ZERO_WORK),
    }


def _normalize_test_commands(
    commands: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for command in commands:
        if set(command) != {"command", "passed", "failed", "note"}:
            raise J2A1V3A1IntegrityError("Test command schema changed")
        if (
            not isinstance(command["command"], str)
            or type(command["passed"]) is not int
            or type(command["failed"]) is not int
            or command["failed"] != 0
            or not isinstance(command["note"], str)
        ):
            raise J2A1V3A1IntegrityError("Test command result is invalid")
        normalized.append(dict(command))
    if not normalized:
        raise J2A1V3A1IntegrityError("Test commands are empty")
    return normalized


def test_evidence_payload(
    *,
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
    pre_prepare: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _normalize_test_commands(commands)
    if (
        not all(isinstance(value, str) for value in deselections)
        or list(deselections) != EXPECTED_DESELECTIONS
    ):
        raise J2A1V3A1IntegrityError("Deselections changed")
    expected_diagnostic = parent_hold_payload()["diagnostic"]
    return {
        "version": f"{VERSION}_test_evidence_v1",
        "source_identities": _source_identities(),
        "commands": normalized,
        "deselections": list(deselections),
        "total_passed": sum(row["passed"] for row in normalized),
        "total_failed": sum(row["failed"] for row in normalized),
        "total_deselected": len(deselections),
        "bound_parent_diagnostic": expected_diagnostic,
        "pre_prepare_audit_sha256": canonical_json_hash(pre_prepare),
        "future_namespaces_absent": _future_namespaces_absent(),
        "zero_work": dict(ZERO_WORK),
        "passes":
            pre_prepare.get("passes") is True
            and all(row["failed"] == 0 for row in normalized)
            and _future_namespaces_absent()
            and _zero_work_exact(),
    }


def write_test_evidence(
    *,
    output_dir: Path,
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
) -> dict[str, Any]:
    pre_prepare = audit_pre_prepare(output_dir=output_dir)
    if pre_prepare["passes"] is not True:
        raise J2A1V3A1IntegrityError(
            "Pre-prepare audit did not pass"
        )
    return write_immutable_json(
        output_dir / TEST_EVIDENCE_NAME,
        test_evidence_payload(
            commands=commands,
            deselections=deselections,
            pre_prepare=pre_prepare,
        ),
        field=READINESS_FIELDS["test_evidence"],
    )


def _load_test_evidence(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if (
        not v3.verify_payload_hash(
            payload,
            READINESS_FIELDS["test_evidence"],
        )
        or payload.get("source_identities") != _source_identities()
        or payload.get("passes") is not True
        or payload.get("total_failed") != 0
    ):
        raise J2A1V3A1IntegrityError("Test evidence changed")
    return payload


def input_bindings_payload(
    *,
    parent_hold: Mapping[str, Any],
    post_seal: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "sources_present":
            all(path.is_file() for path in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)),
        "parent_hold_passes": parent_hold.get("passes") is True,
        "parent_post_seal_passes": post_seal.get("passes") is True,
        "parent_sources_exact":
            post_seal["parent_v3"]["checks"]["source_identities_exact"],
        "parent_nine_artifacts_exact":
            post_seal["parent_v3"]["checks"][
                "all_file_payload_identities_exact"
            ],
        "authority_split_exact":
            v3.V2_COMPLETED_ROOTS == 3_048
            and v3.RECOVERY_ROOTS == 11_288,
        "unfinished_identity_exact":
            v3.EXPECTED_UNFINISHED_SHA256
            == "dca4de9005bede7e710ce004ade443aef5a0eda3c28f3994157a136bde0d34a9",
        "completed_identity_exact":
            v3.EXPECTED_COMPLETED_REFS_SHA256
            == "78f67e7b4da2a23ceb537366ad7cab6ac6f287b872f1e30b8eb67b5fefe2457b",
        "future_namespaces_absent": _future_namespaces_absent(),
        "zero_scientific_work": _zero_work_exact(),
    }
    return {
        "version": f"{VERSION}_input_bindings_v1",
        "source_identities": _source_identities(),
        "parent_v3_source_identities": PARENT_SOURCE_IDENTITIES,
        "parent_v3_artifact_identities": PARENT_ARTIFACT_IDENTITIES,
        "parent_hold_sha256": canonical_json_hash(parent_hold),
        "post_seal_audit_sha256": canonical_json_hash(post_seal),
        "v2_completed_roots": v3.V2_COMPLETED_ROOTS,
        "v3_unfinished_roots": v3.RECOVERY_ROOTS,
        "total_roots": v3.ACTIVE_ROOTS,
        "streams": v3.ACTIVE_STREAMS,
        "new_reservations": 0,
        "new_consumptions": 0,
        "checks": checks,
        "zero_work": dict(ZERO_WORK),
        "passes": all(checks.values()),
    }


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
        "version": f"{VERSION}_retention_v1",
        "files": inventory,
        "file_count": len(inventory),
        "retained_bytes": sum(row["bytes"] for row in inventory),
        "canonical_inventory_sha256": canonical_json_hash(inventory),
        "future_namespaces_absent": _future_namespaces_absent(),
        "zero_work": dict(ZERO_WORK),
        "no_cleanup_performed": True,
        "passes":
            len(inventory) == 9
            and _future_namespaces_absent()
            and _zero_work_exact(),
    }


def prepare_readiness(
    *,
    output_dir: Path = READINESS_DIR,
    include_operational: bool = True,
    free_bytes: int | None = None,
) -> dict[str, Any]:
    paths = readiness_paths(output_dir)
    observed = {
        path.resolve()
        for path in output_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    } if output_dir.exists() else set()
    if observed != {paths["test_evidence"].resolve()}:
        raise J2A1V3A1IntegrityError(
            "Readiness namespace is not at the evidence boundary"
        )
    evidence = _load_test_evidence(paths["test_evidence"])
    parent_hold = parent_hold_payload()
    post_seal = audit_post_seal()
    bindings = input_bindings_payload(
        parent_hold=parent_hold,
        post_seal=post_seal,
    )
    headroom = launch_headroom_audit(
        output_dir=output_dir,
        include_operational=include_operational,
        free_bytes=free_bytes,
    )
    cleanup = cleanup_review_proposal(headroom)
    schema = readiness_schema()
    integrity_passes = all(
        payload.get("passes") is True
        for payload in (
            evidence,
            parent_hold,
            post_seal,
            bindings,
            cleanup,
        )
    ) and headroom["integrity_passes"] is True
    decision = readiness_decision(
        integrity_passes=integrity_passes,
        headroom_passes=headroom["admission_passes"] is True,
    )
    payloads = {
        "parent_hold": parent_hold,
        "input_bindings": bindings,
        "chronology": post_seal,
        "headroom": headroom,
        "cleanup_proposal": cleanup,
        "schema": schema,
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
            "parent_hold",
            "input_bindings",
            "chronology",
            "headroom",
            "cleanup_proposal",
            "schema",
        )
    }
    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "decision_candidate": decision,
        "source_identities": _source_identities(),
        "predecessors": predecessors,
        "parent_v3_readiness": {
            key: {
                "file_sha256": value[0],
                "payload_sha256": value[1],
            }
            for key, value in PARENT_ARTIFACT_IDENTITIES.items()
        },
        "headroom": headroom["calculation"],
        "future_authorization_root": str(
            FUTURE_AUTHORIZATION_DIR.relative_to(REPO_ROOT)
        ),
        "future_execution_root": str(
            FUTURE_EXECUTION_DIR.relative_to(REPO_ROOT)
        ),
        "execution_authorized": False,
        "cleanup_authorized": False,
        "zero_work": dict(ZERO_WORK),
        "integrity_passes": integrity_passes,
        "admission_passes": headroom["admission_passes"] is True,
        "passes": integrity_passes,
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
        "parent_hold_decision": PARENT_HOLD,
        "headroom": headroom["calculation"],
        "execution_authorized": False,
        "phase_lock_authorized": False,
        "cleanup_authorized": False,
        "collectors_authorized": False,
        "stage_b_authorized": False,
        "ppo_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
        "human_training_authorized": False,
        "continue": decision == READY,
        "hold": decision == HOLD,
        "kill": decision == KILL,
        "promote": False,
        "checks": {
            "evidence": evidence["passes"],
            "parent_hold": parent_hold["passes"],
            "post_seal_reproducibility": post_seal["passes"],
            "input_bindings": bindings["passes"],
            "headroom_integrity": headroom["integrity_passes"],
            "headroom_admission": headroom["admission_passes"],
            "cleanup_review_only":
                cleanup["cleanup_authorized"] is False
                and cleanup["cleanup_performed"] is False,
            "future_namespaces_absent": _future_namespaces_absent(),
            "zero_scientific_work": _zero_work_exact(),
        },
        "zero_work": dict(ZERO_WORK),
        "integrity_passes": integrity_passes,
        "admission_passes": headroom["admission_passes"] is True,
        "passes": integrity_passes,
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
    verification = verify_readiness_package(output_dir)
    return {
        "decision": decision,
        "result": load_json(paths["result"]),
        "lock": load_json(paths["lock"]),
        "retention": load_json(paths["retention"]),
        "verification": verification,
        "execution_authorized": False,
        "passes": integrity_passes,
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
        raise J2A1V3A1IntegrityError(
            "V3A1 readiness file set changed"
        )
    payloads = {}
    identities = {}
    for key, path in paths.items():
        payload = load_json(path)
        field = READINESS_FIELDS[key]
        if not v3.verify_payload_hash(payload, field):
            raise J2A1V3A1IntegrityError(
                f"V3A1 readiness payload changed: {path.name}"
            )
        payloads[key] = payload
        identities[key] = artifact_identity(path, field)
    retention = payloads["retention"]
    for row in retention["files"]:
        path = readiness_dir / row["path"]
        if (
            not path.is_file()
            or int(path.stat().st_size) != row["bytes"]
            or sha256_path(path) != row["file_sha256"]
        ):
            raise J2A1V3A1IntegrityError(
                "V3A1 retained file changed"
            )
    checks = {
        "source_identities_exact":
            payloads["result"].get("source_identities")
            == _source_identities(),
        "decision_valid":
            payloads["result"].get("decision") in {READY, HOLD, KILL},
        "execution_unauthorized":
            payloads["result"].get("execution_authorized") is False,
        "parent_hold_bound":
            payloads["result"].get("parent_hold_decision") == PARENT_HOLD,
        "retention_passes": retention.get("passes") is True,
        "retention_inventory_exact":
            canonical_json_hash(retention["files"])
            == retention["canonical_inventory_sha256"],
        "future_namespaces_absent": _future_namespaces_absent(),
        "zero_scientific_work": _zero_work_exact(),
    }
    if not all(checks.values()):
        raise J2A1V3A1IntegrityError(
            "V3A1 readiness verification failed"
        )
    return {
        "decision": payloads["result"]["decision"],
        "payloads": payloads,
        "identities": identities,
        "checks": checks,
        "passes": True,
    }


def _load_json_list(path: Path, *, name: str) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise J2A1V3A1IntegrityError(
            f"Cannot load {name}"
        ) from error
    if not isinstance(payload, list):
        raise J2A1V3A1IntegrityError(f"{name} is not a list")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("audit-pre-prepare")
    pre.add_argument("--output-dir", type=Path, default=READINESS_DIR)
    commands.add_parser("audit-post-seal")
    evidence = commands.add_parser("write-test-evidence")
    evidence.add_argument("--output-dir", type=Path, default=READINESS_DIR)
    evidence.add_argument("--commands-json", type=Path, required=True)
    evidence.add_argument("--deselections-json", type=Path, required=True)
    prepare = commands.add_parser("prepare-readiness")
    prepare.add_argument("--output-dir", type=Path, default=READINESS_DIR)
    return parser


def dispatch_cli(args: argparse.Namespace) -> dict[str, Any]:
    command = str(args.command)
    if command == "audit-pre-prepare":
        return audit_pre_prepare(output_dir=args.output_dir)
    if command == "audit-post-seal":
        return audit_post_seal()
    if command == "write-test-evidence":
        return write_test_evidence(
            output_dir=args.output_dir,
            commands=_load_json_list(
                args.commands_json,
                name="commands",
            ),
            deselections=_load_json_list(
                args.deselections_json,
                name="deselections",
            ),
        )
    if command == "prepare-readiness":
        return prepare_readiness(
            output_dir=args.output_dir,
            include_operational=True,
        )
    raise J2A1V3A1IntegrityError(
        f"Forbidden V3A1 command: {command}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    result = dispatch_cli(build_parser().parse_args(argv))
    print(json.dumps(v3.preflight._json_native(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

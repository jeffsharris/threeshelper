"""J2 A1 distillation and closed-loop fidelity execution surface.

Readiness commands are outcome-free. Scientific phase commands are present
for review but require a separately sealed authorization that does not exist
in this construction turn.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import queue
import socket
import statistics
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from threes_rl import j2_incumbent_distillation_readiness as j2


VERSION = "j2a1_distillation_fidelity_execution_surface_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs"
CHARTER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_CHARTER.md"
)
RUNNER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "j2a1_distillation_fidelity_execution_surface.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_rl_j2a1_distillation_fidelity_execution_surface.py"
)
A1_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j2_incumbent_distillation_readiness_amendment_a1"
)
PARENT_READINESS_DIR = (
    RUNS_ROOT / "forensics" / "j2_incumbent_distillation_readiness_v1"
)
V2_PILOT_DIR = (
    RUNS_ROOT / "forensics" / "j2_exact_teacher_feasibility_pilot_v2"
)
READINESS_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j2a1_distillation_fidelity_execution_surface_readiness_v1"
)
FUTURE_EXECUTION_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j2a1_distillation_fidelity_execution_v1"
)

TEST_EVIDENCE_NAME = "J2A1_EXECUTION_SURFACE_TEST_EVIDENCE.json"
INPUT_BINDINGS_NAME = "J2A1_EXECUTION_SURFACE_INPUT_BINDINGS.json"
AUTHORITY_AUDIT_NAME = "J2A1_EXECUTION_SURFACE_AUTHORITY_AUDIT.json"
SCHEMA_NAME = "J2A1_EXECUTION_SURFACE_SCHEMA.json"
PROJECTION_NAME = "J2A1_EXECUTION_SURFACE_PROJECTION.json"
READINESS_LOCK_NAME = "J2A1_EXECUTION_SURFACE_READINESS_LOCK.json"
READINESS_RESULT_NAME = "J2A1_EXECUTION_SURFACE_READINESS_RESULT.json"
RETENTION_NAME = "J2A1_EXECUTION_SURFACE_RETENTION.json"

PHASE_LOCK_NAME = "J2A1_DISTILLATION_PHASE_LOCK.json"
OPEN_MARKER_NAME = "J2A1_DISTILLATION_EXECUTION_MARKER.json"
MATERIALIZED_MANIFEST_NAME = "J2A1_DISTILLATION_ACTIVE_MANIFEST.json"
RESERVATION_NAME = "J2A1_DISTILLATION_STREAM_RESERVATION.json"
CONSUMPTION_NAME = "J2A1_DISTILLATION_STREAM_CONSUMPTION.json"
GENESIS_NAME = "J2A1_DISTILLATION_GENESIS.json"
FINAL_OPERATIONAL_GUARD_NAME = (
    "J2A1_DISTILLATION_FINAL_OPERATIONAL_GUARD.json"
)
TERMINAL_EVIDENCE_NAME = "J2A1_DISTILLATION_FIDELITY_TERMINAL_EVIDENCE.json"
TERMINAL_NAME = "J2A1_DISTILLATION_FIDELITY_TERMINAL.json"
EXECUTION_RETENTION_NAME = "J2A1_DISTILLATION_FIDELITY_RETENTION.json"

READY = "READY_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE"
HOLD = "HOLD_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE"
KILL = "KILL_J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_INTEGRITY"

READY_EXECUTION = "READY_J2A1_PPO_EXECUTION_SURFACE_REVIEW"
HOLD_FAMILY = "HOLD_J2A1_FAMILY_DATA_SUPPORT"
HOLD_MECHANISM = "HOLD_J2A1_BC_MECHANISM"
HOLD_FIDELITY = "HOLD_J2A1_CLOSED_LOOP_FIDELITY"
HOLD_BASE_RATE = "HOLD_J2A1_FIDELITY_INCONCLUSIVE_LOW_BASE_RATE"
HOLD_OPERATIONAL = "HOLD_J2A1_DISTILLATION_OPERATIONAL"
KILL_EXECUTION = "KILL_J2A1_DISTILLATION_FIDELITY_INTEGRITY"

BC_STAGE = "teacher_behavior_cloning"
VALIDATION_STAGE = "distillation_validation"
ACTIVE_STAGES = (BC_STAGE, VALIDATION_STAGE)
BC_ROOTS = 8_192
VALIDATION_PAIRS = 6_144
ACTIVE_ROOTS = BC_ROOTS + VALIDATION_PAIRS
ACTIVE_ARMS = BC_ROOTS + 2 * VALIDATION_PAIRS
ACTIVE_STREAMS = 4 * BC_ROOTS + 5 * VALIDATION_PAIRS
SHARDS = 8
VALIDATION_STRATA = 8
VALIDATION_PAIRS_PER_STRATUM = VALIDATION_PAIRS // VALIDATION_STRATA
MAX_MOVES = 5_000

BOOTSTRAP_REPLICATES = 4_096
SCORE_BOOTSTRAP_SEED = 2_026_072_831
PROGRESSION_BOOTSTRAP_SEED = 2_026_072_832
BOOTSTRAP_QUANTILES = (0.025, 0.975)
QUANTILE_METHOD = "linear"

FEATURE_FAMILIES = (
    "low_air",
    "low_constrained",
    "mid_progression",
    "upper_progression",
)
MIN_STATES_PER_FAMILY = 1_024
MIN_ROOTS_PER_FAMILY = 256
NATURAL_MAX_SHARE = 0.70
CAPPED_MAX_SHARE = 0.40

RUNTIME_CAP_HOURS = 72.0
STORAGE_CAP_BYTES = 24 * 1024**3
HARD_DISK_FLOOR_GIB = 100.0
TARGET_DISK_GIB = 120.0
SAFETY_MULTIPLIER = 1.25
PLANNING_MOVES = 512
SENSITIVITY_MOVES = 5_000
ROOT_BLOB_BYTES_PER_TRANSITION = 1_519
BC_BATCH_BYTES_PER_ROW = 1_261
PAIR_BLOB_BYTES = 24_576
ACTIVE_CHUNK_BYTES_PER_TRANSITION = 1_526
FIXED_OVERHEAD_BYTES = 896 * 1024**2
TEACHER_P99_SECONDS = 0.1316514358320273
OPTIMIZER_FIXTURE_HOURS = 0.03300427754720052
INHERITED_ADMIN_HOURS = 3.309263890690274
ACTOR_BATCH16_P99_SECONDS = 0.00009085019119083881
SIM_TRANSITION_P99_SECONDS = 0.00004372494295239449

DISTILLATION_AUTHORIZATION_DECISION = (
    "CONTINUE_J2A1_DISTILLATION_FIDELITY_EXECUTION"
)

ZERO_WORK = {
    "phase_locks": 0,
    "execution_markers": 0,
    "manifests_materialized": 0,
    "streams_reserved": 0,
    "streams_consumed": 0,
    "teacher_process_loads": 0,
    "teacher_queries": 0,
    "teacher_action_labels": 0,
    "normal_start_games": 0,
    "scientific_optimizer_steps": 0,
    "scientific_checkpoints": 0,
    "validation_teacher_content_reads": 0,
    "validation_student_arms": 0,
    "partial_fidelity_outcome_reads": 0,
    "fidelity_outcome_reads": 0,
    "ppo_content_reads": 0,
    "development_content_reads": 0,
    "confirmation_content_reads": 0,
    "human_session_reads": 0,
    "incumbent_changes": 0,
    "dashboard_changes": 0,
    "promotion_actions": 0,
}

EXPECTED_SOURCE_HASHES = {
    "threes_rl/J2_INCUMBENT_DISTILLATION_READINESS_AMENDMENT_A1.md":
        "371e16088a4cbe3a7a3c5e6668fdd13424cad439f1a8f7e85b4dc7c120573e6a",
    "threes_rl/j2_incumbent_distillation_readiness_amendment_a1.py":
        "80b4e0d88bbfa25b494d3c0f5783c1996ee057405e2277152107821265eb5a7d",
    "tests/test_rl_j2_incumbent_distillation_readiness_amendment_a1.py":
        "bbfb98088431c0421fd82d5faa067736783ed262c5e841772eb9af09d1616db6",
    "threes_rl/j2_incumbent_distillation_readiness.py":
        "9ecd658ea69968feb605d0e0a9e4e621b73ac01619536e45c0cdf69b7bc3b15f",
    "threes_rl/J2_EXACT_TEACHER_ENGINEERING_FEASIBILITY_PILOT_V2_CHARTER.md":
        "f612695a69a1914a29eb0b9c60932680576085d2da55a9cd32663d3f99b1277f",
    "threes_rl/j2_exact_teacher_feasibility_pilot_v2.py":
        "e72e96907ac356116ee6b764a6d2fe32d9ccf1339010418d1ade9c3660bee3f8",
    "tests/test_rl_j2_exact_teacher_feasibility_pilot_v2.py":
        "f92e46135a52d000782d8ae6c4d36ba4d9d3b1179a56438d952a94cf8852c4df",
}

EXPECTED_A1_ARTIFACTS = {
    "J2A1_FAMILY_SUPPORT_SAFEGUARD.json": (
        "ca42403d6bbd1fb5ff41ed8195914b3ff69f87a0e7e44f5d2c554965bee65303",
        "family_safeguard_payload_sha256",
        "f8c46b011d7359f2e27e20c39d8f767528b3c6f5cf5581526b094bdecf265462",
    ),
    "J2A1_INPUT_BINDINGS.json": (
        "67efadb9bce4b3a00c4a264a400e39f1bd73190fc18472c90a26fa76e297cf59",
        "input_bindings_payload_sha256",
        "14eede8aa0e2d72b6460a4c38666f9b864e97e4e31cbf20391913bd3833bf5e8",
    ),
    "J2A1_POWER.json": (
        "38d590badc528906e6c7437f56b2f5348a24d2cc160dfe34e73eb4660c820b08",
        "power_payload_sha256",
        "65a0c1ae9c7bdfb078e61ef383c5569f7a6167bdb4fa5f11edcfead23529fa37",
    ),
    "J2A1_PROSPECTIVE_AUTHORITY.json": (
        "0b421aa2da3e1f88d9b47b4d0dc26f7e696ca8e5ec6e8e43abd4052c6f7b2b94",
        "prospective_authority_payload_sha256",
        "4c120dbc5830385b59ac10cc0fff3505f3a39da197cd312d5f3bf2a5126f6c5f",
    ),
    "J2A1_READINESS_LOCK.json": (
        "19b5b469d711a02cb04c3eb4d772a7375f5bccd1a8069fa19bc61e92d92a654b",
        "readiness_lock_payload_sha256",
        "d3e7f1bb4490d2a4ff398c365d6864c939ffbe3b83e4e29660c9468fe1a06c10",
    ),
    "J2A1_READINESS_RESULT.json": (
        "0cfccc9fdc0cd7310200b31d1ec65890153541799a30a95808f92028b114e804",
        "readiness_result_payload_sha256",
        "d37783ba9fed257f24c0d1888fd749c2c102e2ee399e5511d2839d201c0c0b52",
    ),
    "J2A1_RETENTION.json": (
        "a9932178a5c62758e35c668a5f44e8415f7c66bb1c8abcda10a303a65ff5adb6",
        "retention_payload_sha256",
        "d08b968c0f5205ebbb5c0ff6828798ab3f9ada69a64b68ef43ab67eada29da54",
    ),
    "J2A1_RUNTIME_STORAGE_PROJECTION.json": (
        "1dc56a86f94d42dae03a555005e602e1e4794fe5aaea3b07eae0b731cb68820c",
        "projection_payload_sha256",
        "a8ead8f025debdbe03f423852a1102508de780a376caf1c2bcf4cb83e71d1dd5",
    ),
    "J2A1_TEST_EVIDENCE.json": (
        "2338f0c151bcb5c9f1c5d152952c9b6aa78428dfa01cee1cbb1784ba7ec4e8cb",
        "test_evidence_payload_sha256",
        "8ef724070deee64ad4e091f4ae3fcc36e1696a194d955079c83708505d613154",
    ),
}

EXPECTED_PARENT_TEACHER_PROVENANCE = (
    "824aa8988136d81a00d81dd4899b9985aedbbb213260d3a2e94c4e7dc931840a",
    "teacher_provenance_payload_sha256",
    "a8d355bd056bdd31f860a668d4e86a0898866192b39cf0665d348db33ac02768",
)
EXPECTED_V2_TERMINAL = (
    "3ee2b204307bb96489ffd0fc3ff5c6c0cef488d6b5cfe986c4940f808354fcd9",
    "terminal_payload_sha256",
    "8b98a0ec9892b615dd5072849b9fc655f7d043c7a257d90619ddbf35ad925089",
)
EXPECTED_V2_RETENTION = (
    "6fe6563d6d676bf93455f0f3060ae3d851bf4b87b0c440216c52c703d0ff53a0",
    "retention_payload_sha256",
    "e8b9e6365a449689f0b485790ea9f3e4a27d1351562b0d387cd113c6db4702d1",
)


class J2A1ExecutionIntegrityError(RuntimeError):
    """An immutable identity, chronology, or numerical contract failed."""


class J2A1ExecutionOperationalHold(RuntimeError):
    """A mutable process, service, runtime, disk, or ownership gate failed."""


class J2A1ExecutionDataHold(RuntimeError):
    """A frozen feature, mechanism, or fidelity support gate failed."""


class J2A1PlannedInterruption(RuntimeError):
    """A fixture-only interruption immediately after a durable boundary."""


def sha256_path(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        j2.json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = j2.json_native(dict(payload))
    if not isinstance(body, dict):
        raise J2A1ExecutionIntegrityError("Payload is not a JSON object")
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    observed = body.pop(field, None)
    return isinstance(observed, str) and observed == canonical_json_hash(body)


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    try:
        return j2.write_immutable_json(path, payload, field=field)
    except j2.J2ReadinessIntegrityError as error:
        raise J2A1ExecutionIntegrityError(str(error)) from error


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise J2A1ExecutionIntegrityError(
            f"Cannot load immutable JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise J2A1ExecutionIntegrityError(
            f"Immutable JSON is not an object: {path}"
        )
    return payload


def load_bound_json(
    path: Path,
    *,
    file_sha256: str,
    payload_field: str,
    payload_sha256: str,
) -> dict[str, Any]:
    if not path.is_file() or sha256_path(path) != file_sha256:
        raise J2A1ExecutionIntegrityError(
            f"Immutable file identity changed: {path}"
        )
    payload = load_json(path)
    if (
        not verify_payload_hash(payload, payload_field)
        or payload.get(payload_field) != payload_sha256
    ):
        raise J2A1ExecutionIntegrityError(
            f"Immutable payload identity changed: {path}"
        )
    return payload


def artifact_identity(path: Path, payload_field: str) -> dict[str, Any]:
    payload = load_json(path)
    if not verify_payload_hash(payload, payload_field):
        raise J2A1ExecutionIntegrityError(
            f"Artifact payload is invalid: {path}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "file_sha256": sha256_path(path),
        "payload_field": payload_field,
        "payload_sha256": payload[payload_field],
    }


def _validate_hex(value: Any, *, name: str) -> str:
    text = str(value)
    if len(text) != 64:
        raise J2A1ExecutionIntegrityError(f"{name} is not SHA-256")
    try:
        int(text, 16)
    except ValueError as error:
        raise J2A1ExecutionIntegrityError(
            f"{name} is not hexadecimal"
        ) from error
    return text


def _stream_prefix(value: int) -> int:
    return int(value) // 1_000_000_000


def expected_active_rows() -> list[dict[str, Any]]:
    authority_path = A1_DIR / "J2A1_PROSPECTIVE_AUTHORITY.json"
    specification = EXPECTED_A1_ARTIFACTS[authority_path.name]
    payload = load_bound_json(
        authority_path,
        file_sha256=specification[0],
        payload_field=specification[1],
        payload_sha256=specification[2],
    )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise J2A1ExecutionIntegrityError("A1 authority rows are missing")
    active = [
        j2.json_native(row)
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("stage")) in ACTIVE_STAGES
    ]
    if not all(isinstance(row, dict) for row in active):
        raise J2A1ExecutionIntegrityError("Active A1 row is malformed")
    return [dict(row) for row in active]


def authority_audit() -> dict[str, Any]:
    rows = expected_active_rows()
    stage_rows = {
        stage: [row for row in rows if row["stage"] == stage]
        for stage in ACTIVE_STAGES
    }
    expected_counts = {
        BC_STAGE: BC_ROOTS,
        VALIDATION_STAGE: VALIDATION_PAIRS,
    }
    stream_keys = {
        BC_STAGE: (
            "logical_stream_id",
            "deck_stream_id",
            "slot_stream_id",
            "teacher_policy_stream_id",
        ),
        VALIDATION_STAGE: (
            "logical_stream_id",
            "deck_stream_id",
            "slot_stream_id",
            "student_policy_stream_id",
            "teacher_policy_stream_id",
        ),
    }
    bases = {
        BC_STAGE: (227, 228, 229, 230),
        VALIDATION_STAGE: (231, 232, 233, 234, 235),
    }
    all_streams: list[int] = []
    roots: list[str] = []
    ancestries: list[str] = []
    stage_checks: dict[str, bool] = {}
    for stage in ACTIVE_STAGES:
        candidates = stage_rows[stage]
        count = expected_counts[stage]
        stage_checks[f"{stage}_count_exact"] = len(candidates) == count
        stage_checks[f"{stage}_indices_exact"] = [
            int(row["row_index"]) for row in candidates
        ] == list(range(count))
        keys = stream_keys[stage]
        prefixes = bases[stage]
        row_stream_checks = []
        for row in candidates:
            index = int(row["row_index"])
            streams = row.get("streams")
            if (
                not isinstance(streams, Mapping)
                or set(streams) != set(keys)
                or len(streams) != len(keys)
            ):
                row_stream_checks.append(False)
                continue
            values = [int(streams[key]) for key in keys]
            row_stream_checks.append(
                all(
                    value == prefix * 1_000_000_000 + index
                    for value, prefix in zip(values, prefixes)
                )
            )
            all_streams.extend(values)
            roots.append(_validate_hex(row["root_id"], name="root_id"))
            ancestries.append(
                _validate_hex(row["ancestry_id"], name="ancestry_id")
            )
        stage_checks[f"{stage}_stream_rows_exact"] = all(row_stream_checks)
        stage_checks[f"{stage}_fresh_flags_zero"] = all(
            row.get("reserved") is False
            and row.get("consumed") is False
            and row.get("content_opened") is False
            for row in candidates
        )
    validation_strata = Counter(
        int(row["row_index"]) % VALIDATION_STRATA
        for row in stage_rows[VALIDATION_STAGE]
    )
    checks = {
        **stage_checks,
        "active_root_count_exact": len(rows) == ACTIVE_ROOTS,
        "active_arm_count_exact": ACTIVE_ARMS == 20_480,
        "active_stream_count_exact": len(all_streams) == ACTIVE_STREAMS,
        "streams_unique": len(set(all_streams)) == ACTIVE_STREAMS,
        "roots_unique": len(set(roots)) == ACTIVE_ROOTS,
        "ancestries_unique": len(set(ancestries)) == ACTIVE_ROOTS,
        "fixed_shard_ownership": all(
            int(row["row_index"]) % SHARDS in range(SHARDS)
            for row in rows
        ),
        "validation_eight_equal_strata": validation_strata
        == Counter(
            {
                index: VALIDATION_PAIRS_PER_STRATUM
                for index in range(VALIDATION_STRATA)
            }
        ),
        "prefixes_exact": Counter(map(_stream_prefix, all_streams))
        == Counter(
            {
                227: BC_ROOTS,
                228: BC_ROOTS,
                229: BC_ROOTS,
                230: BC_ROOTS,
                231: VALIDATION_PAIRS,
                232: VALIDATION_PAIRS,
                233: VALIDATION_PAIRS,
                234: VALIDATION_PAIRS,
                235: VALIDATION_PAIRS,
            }
        ),
    }
    return {
        "version": f"{VERSION}_authority_audit_v1",
        "active_stages": list(ACTIVE_STAGES),
        "stage_counts": {
            stage: len(stage_rows[stage]) for stage in ACTIVE_STAGES
        },
        "active_root_rows": len(rows),
        "active_game_arms": ACTIVE_ARMS,
        "active_unique_streams": len(set(all_streams)),
        "canonical_rows_sha256": canonical_json_hash(rows),
        "root_set_sha256": canonical_json_hash(sorted(roots)),
        "ancestry_set_sha256": canonical_json_hash(sorted(ancestries)),
        "stream_set_sha256": canonical_json_hash(sorted(all_streams)),
        "validation_stratum_counts": {
            str(key): value for key, value in sorted(validation_strata.items())
        },
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": dict(ZERO_WORK),
    }


def source_and_parent_audit(
    *,
    require_future_execution_absent: bool,
) -> dict[str, Any]:
    local_sources = {
        relative: sha256_path(REPO_ROOT / relative)
        for relative in EXPECTED_SOURCE_HASHES
    }
    source_checks = {
        relative: local_sources[relative] == expected
        for relative, expected in EXPECTED_SOURCE_HASHES.items()
    }
    a1_artifacts = {}
    for name, (file_sha, field, payload_sha) in EXPECTED_A1_ARTIFACTS.items():
        path = A1_DIR / name
        load_bound_json(
            path,
            file_sha256=file_sha,
            payload_field=field,
            payload_sha256=payload_sha,
        )
        a1_artifacts[name] = artifact_identity(path, field)
    a1_result = load_json(A1_DIR / "J2A1_READINESS_RESULT.json")
    a1_retention = load_json(A1_DIR / "J2A1_RETENTION.json")
    teacher_path = PARENT_READINESS_DIR / "J2_TEACHER_PROVENANCE.json"
    teacher = load_bound_json(
        teacher_path,
        file_sha256=EXPECTED_PARENT_TEACHER_PROVENANCE[0],
        payload_field=EXPECTED_PARENT_TEACHER_PROVENANCE[1],
        payload_sha256=EXPECTED_PARENT_TEACHER_PROVENANCE[2],
    )
    v2_terminal_path = V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_TERMINAL_RESULT.json"
    v2_terminal = load_bound_json(
        v2_terminal_path,
        file_sha256=EXPECTED_V2_TERMINAL[0],
        payload_field=EXPECTED_V2_TERMINAL[1],
        payload_sha256=EXPECTED_V2_TERMINAL[2],
    )
    v2_retention_path = V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_RETENTION.json"
    v2_retention = load_bound_json(
        v2_retention_path,
        file_sha256=EXPECTED_V2_RETENTION[0],
        payload_field=EXPECTED_V2_RETENTION[1],
        payload_sha256=EXPECTED_V2_RETENTION[2],
    )
    teacher_source_checks = {
        relative: (
            sha256_path(REPO_ROOT / relative)
            == binding["expected_sha256"]
            == binding["observed_sha256"]
        )
        for relative, binding in teacher["implementation_sources"].items()
    }
    checks = {
        "all_parent_sources_exact": all(source_checks.values()),
        "all_a1_artifacts_exact": len(a1_artifacts)
        == len(EXPECTED_A1_ARTIFACTS),
        "a1_ready_exact": a1_result.get("decision")
        == "READY_J2_A1_INCUMBENT_DISTILLATION_PREFLIGHT",
        "a1_execution_not_authorized": a1_result.get(
            "execution_authorized"
        )
        is False,
        "a1_retention_passes": a1_retention.get("passes") is True,
        "teacher_kind_exact": teacher.get("teacher_kind")
        == "exact protected composite software incumbent",
        "teacher_binding_exact": teacher.get("passes") is True
        and teacher.get("human_session_content_read") is False
        and teacher.get("replay_payload_parsed") is False,
        "teacher_implementation_sources_exact": all(
            teacher_source_checks.values()
        ),
        "v2_pilot_terminal_exact": (
            v2_terminal.get("decision")
            == "READY_J2_FEASIBILITY_AMENDMENT_PREFLIGHT_V2"
            and v2_terminal.get("integrity_passes") is True
            and v2_terminal.get("separate_decisions", {}).get(
                "real_eight_process_pretraining_throughput_memory"
            )
            == "PASS"
            and v2_terminal.get("separate_decisions", {}).get(
                "synchronous_16_round_orchestration"
            )
            == "PASS"
            and v2_terminal.get("separate_decisions", {}).get(
                "powered_validation_n_recommendation",
                {},
            ).get("smallest_grid_n_at_least_080")
            == 6_144
        ),
        "v2_pilot_retention_exact": v2_retention.get("passes") is True,
        "future_execution_absent": (
            not FUTURE_EXECUTION_DIR.exists()
            if require_future_execution_absent
            else True
        ),
    }
    return {
        "version": f"{VERSION}_source_parent_audit_v1",
        "local_source_hashes": local_sources,
        "source_checks": source_checks,
        "a1_artifacts": a1_artifacts,
        "teacher_provenance": artifact_identity(
            teacher_path,
            EXPECTED_PARENT_TEACHER_PROVENANCE[1],
        ),
        "teacher_incumbent_binding_sha256": teacher[
            "incumbent_binding"
        ]["incumbent_binding_sha256"],
        "teacher_implementation_source_checks": teacher_source_checks,
        "v2_terminal": artifact_identity(
            v2_terminal_path,
            EXPECTED_V2_TERMINAL[1],
        ),
        "v2_retention": artifact_identity(
            v2_retention_path,
            EXPECTED_V2_RETENTION[1],
        ),
        "checks": checks,
        "passes": all(checks.values()),
    }


def execution_schema() -> dict[str, Any]:
    schema = {
        "version": f"{VERSION}_schema_v1",
        "parent_model_schema": j2.model_schema(),
        "active_authority": {
            "stages": list(ACTIVE_STAGES),
            "bc_roots": BC_ROOTS,
            "validation_pairs": VALIDATION_PAIRS,
            "active_rows": ACTIVE_ROOTS,
            "game_arms": ACTIVE_ARMS,
            "unique_streams": ACTIVE_STREAMS,
            "shards": SHARDS,
        },
        "stage_order": [
            "teacher_collection",
            "family_support",
            "distillation",
            "bc_mechanism",
            "closed_loop_fidelity",
        ],
        "public_readiness_commands": [
            "audit-zero-work",
            "write-test-evidence",
            "prepare-readiness",
        ],
        "public_scientific_commands": [
            "seal-phase-lock",
            "open",
            "materialize",
            "execute",
        ],
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "score_seed": SCORE_BOOTSTRAP_SEED,
            "progression_seed": PROGRESSION_BOOTSTRAP_SEED,
            "score_method": "global paired-root resampling",
            "progression_method":
                "independent within-stratum whole-root resampling",
            "strata": VALIDATION_STRATA,
            "pairs_per_stratum": VALIDATION_PAIRS_PER_STRATUM,
            "quantiles": list(BOOTSTRAP_QUANTILES),
            "quantile_method": QUANTILE_METHOD,
        },
        "family_gate": {
            "families": list(FEATURE_FAMILIES),
            "minimum_states_each": MIN_STATES_PER_FAMILY,
            "minimum_roots_each": MIN_ROOTS_PER_FAMILY,
            "natural_max_share_strictly_below": NATURAL_MAX_SHARE,
            "capped_max_share_strictly_below": CAPPED_MAX_SHARE,
        },
        "retention": {
            "complete_teacher_roots": "immutable_write_once",
            "pair_results": "immutable_write_once",
            "transition_chunks": "current_collection_ephemeral",
            "bc_batch": "current_distillation_ephemeral",
            "rolling_slots": 2,
            "maximum_orphans": 1,
        },
        "scientific_execution_authorized": False,
    }
    schema["schema_sha256"] = canonical_json_hash(schema)
    return schema


def model_schema_sha256() -> str:
    return canonical_json_hash(j2.model_schema())


def _scientific_projection(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(
                tensor.numpy().tobytes(order="C")
            ).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _scientific_projection(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_projection(child) for child in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise J2A1ExecutionIntegrityError(
                "Scientific payload contains a nonfinite float"
            )
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise J2A1ExecutionIntegrityError(
        f"Unsupported scientific payload type: {type(value).__name__}"
    )


def scientific_hash(value: Any) -> str:
    return canonical_json_hash(_scientific_projection(value))


ROOT_BLOB_MAGIC = b"J2A1ROOT1\n"
CHECKPOINT_MAGIC = b"J2A1CKPT1\n"


def _serialize_torch_payload(
    payload: Mapping[str, Any],
    *,
    magic: bytes,
) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    body = buffer.getvalue()
    digest = hashlib.sha256(body).hexdigest().encode("ascii")
    return magic + digest + b"\n" + body


def _deserialize_torch_payload(
    serialized: bytes,
    *,
    magic: bytes,
) -> dict[str, Any]:
    if not serialized.startswith(magic):
        raise J2A1ExecutionIntegrityError("Binary artifact magic changed")
    remainder = serialized[len(magic) :]
    try:
        digest, body = remainder.split(b"\n", 1)
    except ValueError as error:
        raise J2A1ExecutionIntegrityError(
            "Binary artifact header is truncated"
        ) from error
    if hashlib.sha256(body).hexdigest().encode("ascii") != digest:
        raise J2A1ExecutionIntegrityError("Binary artifact body changed")
    payload = torch.load(
        io.BytesIO(body),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise J2A1ExecutionIntegrityError(
            "Binary artifact payload is not a mapping"
        )
    return payload


def _create_once_binary(
    path: Path,
    serialized: bytes,
    *,
    validator: Callable[[bytes], Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    validator(serialized)
    expected_sha = hashlib.sha256(serialized).hexdigest()
    if path.exists():
        observed = path.read_bytes()
        if observed != serialized:
            raise J2A1ExecutionIntegrityError(
                f"Immutable binary collision changed bytes: {path}"
            )
        validator(observed)
        return expected_sha
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            observed = path.read_bytes()
            if observed != serialized:
                raise J2A1ExecutionIntegrityError(
                    f"Concurrent immutable binary mismatch: {path}"
                ) from error
            validator(observed)
            return expected_sha
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    observed = path.read_bytes()
    if observed != serialized or hashlib.sha256(observed).hexdigest() != (
        expected_sha
    ):
        raise J2A1ExecutionIntegrityError(
            f"Immutable binary post-write mismatch: {path}"
        )
    validator(observed)
    return expected_sha


def _as_observation(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if (
        array.shape != (j2.OBSERVATION_WIDTH,)
        or not np.isfinite(array).all()
    ):
        raise J2A1ExecutionIntegrityError(
            "Teacher transition observation is malformed"
        )
    return array


def _as_legal_mask(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != (j2.ACTION_COUNT,) or not array.any():
        raise J2A1ExecutionIntegrityError(
            "Teacher transition legal mask is malformed"
        )
    return array


def _as_board(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (4, 4)
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
        or np.any(array < 0)
    ):
        raise J2A1ExecutionIntegrityError(
            "Teacher transition board is malformed"
        )
    return array.astype(np.int32, copy=False)


def teacher_root_content_hash(record: Mapping[str, Any]) -> str:
    body = dict(record)
    body.pop("root_content_sha256", None)
    return scientific_hash(body)


def validate_teacher_root_record(
    record: Mapping[str, Any],
    *,
    authoritative_row: Mapping[str, Any],
) -> dict[str, Any]:
    if record.get("version") != f"{VERSION}_teacher_root_v1":
        raise J2A1ExecutionIntegrityError(
            "Teacher root version changed"
        )
    row = j2.json_native(record.get("row"))
    expected_row = j2.json_native(authoritative_row)
    if row != expected_row:
        raise J2A1ExecutionIntegrityError(
            "Teacher root row changed from immutable authority"
        )
    stage = str(record.get("stage"))
    if stage not in ACTIVE_STAGES or stage != expected_row["stage"]:
        raise J2A1ExecutionIntegrityError("Teacher root stage changed")
    root_id = _validate_hex(record.get("root_id"), name="root_id")
    ancestry_id = _validate_hex(
        record.get("ancestry_id"),
        name="ancestry_id",
    )
    if (
        root_id != expected_row["root_id"]
        or ancestry_id != expected_row["ancestry_id"]
    ):
        raise J2A1ExecutionIntegrityError(
            "Teacher root or ancestry identity changed"
        )
    if int(record.get("shard", -1)) != (
        int(expected_row["row_index"]) % SHARDS
    ):
        raise J2A1ExecutionIntegrityError(
            "Teacher root crossed its fixed shard"
        )
    if (
        record.get("normal_start") is not True
        or record.get("starter_tile", object()) is not None
        or record.get("natural_terminal") is not True
    ):
        raise J2A1ExecutionIntegrityError(
            "Teacher root is not a complete natural normal-start game"
        )
    transitions = record.get("transitions")
    if not isinstance(transitions, (list, tuple)) or not transitions:
        raise J2A1ExecutionIntegrityError(
            "Teacher root transitions are missing"
        )
    if len(transitions) > MAX_MOVES:
        raise J2A1ExecutionIntegrityError(
            "Teacher root reached the 5000-move integrity boundary"
        )
    start_score = int(record.get("start_score", -1))
    final_score = int(record.get("final_score", -1))
    final_max_tile = int(record.get("final_max_tile", -1))
    policy_latency_seconds = float(
        record.get("policy_latency_seconds", math.nan)
    )
    survival = float(record.get("survival", math.nan))
    if start_score < 0 or final_score < start_score:
        raise J2A1ExecutionIntegrityError(
            "Teacher root scores are malformed"
        )
    if (
        final_max_tile < 0
        or not math.isfinite(policy_latency_seconds)
        or policy_latency_seconds < 0.0
        or not math.isfinite(survival)
        or survival < 0.0
    ):
        raise J2A1ExecutionIntegrityError(
            "Teacher root safeguard metrics are malformed"
        )
    deltas: list[int] = []
    normalized_rows = []
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            raise J2A1ExecutionIntegrityError(
                "Teacher transition is not a mapping"
            )
        if int(transition.get("transition_index", -1)) != index:
            raise J2A1ExecutionIntegrityError(
                "Teacher transition order changed"
            )
        observation = _as_observation(transition.get("observation"))
        legal_mask = _as_legal_mask(transition.get("legal_mask"))
        board = _as_board(transition.get("board"))
        action = int(transition.get("teacher_action", -1))
        if action < 0 or action >= j2.ACTION_COUNT or not legal_mask[action]:
            raise J2A1ExecutionIntegrityError(
                "Teacher supplied an illegal action"
            )
        current_score = int(transition.get("current_score", -1))
        delta = int(transition.get("score_delta", -1))
        if current_score < start_score or delta < 0:
            raise J2A1ExecutionIntegrityError(
                "Teacher dense score row is malformed"
            )
        if index == 0 and current_score != start_score:
            raise J2A1ExecutionIntegrityError(
                "Teacher root start score changed"
            )
        if index > 0:
            previous = normalized_rows[-1]
            if current_score != (
                int(previous["current_score"])
                + int(previous["score_delta"])
            ):
                raise J2A1ExecutionIntegrityError(
                    "Teacher score deltas do not form a chain"
                )
        deltas.append(delta)
        normalized_rows.append(
            {
                "transition_index": index,
                "observation": observation,
                "legal_mask": legal_mask,
                "teacher_action": action,
                "current_score": current_score,
                "score_delta": delta,
                "board": board,
                "feature_family": j2.feature_family(board),
            }
        )
    if start_score + sum(deltas) != final_score:
        raise J2A1ExecutionIntegrityError(
            "Teacher root dense score deltas do not telescope"
        )
    suffix = 0
    for transition in reversed(normalized_rows):
        suffix += int(transition["score_delta"])
        target = j2.value_target(
            current_score=int(transition["current_score"]),
            final_score=final_score,
            remaining_score_deltas=[
                int(value["score_delta"])
                for value in normalized_rows[
                    int(transition["transition_index"]) :
                ]
            ],
        )
        if not math.isclose(target, 1e-5 * suffix, abs_tol=0.0, rel_tol=0.0):
            raise J2A1ExecutionIntegrityError(
                "Teacher value target changed from dense telescope"
            )
        transition["value_target"] = target
    if record.get("root_content_sha256") != teacher_root_content_hash(record):
        raise J2A1ExecutionIntegrityError(
            "Teacher root content hash changed"
        )
    return {
        "row": expected_row,
        "root_id": root_id,
        "ancestry_id": ancestry_id,
        "stage": stage,
        "shard": int(record["shard"]),
        "start_score": start_score,
        "final_score": final_score,
        "final_max_tile": final_max_tile,
        "policy_latency_seconds": policy_latency_seconds,
        "survival": survival,
        "transitions": normalized_rows,
        "transition_count": len(normalized_rows),
        "root_content_sha256": record["root_content_sha256"],
        "passes": True,
    }


def seal_teacher_root_record(
    record: Mapping[str, Any],
    *,
    authoritative_row: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(record)
    body.pop("root_content_sha256", None)
    body["root_content_sha256"] = teacher_root_content_hash(body)
    validate_teacher_root_record(
        body,
        authoritative_row=authoritative_row,
    )
    return body


def write_teacher_root_blob(
    path: Path,
    record: Mapping[str, Any],
    *,
    authoritative_row: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = seal_teacher_root_record(
        record,
        authoritative_row=authoritative_row,
    )
    serialized = _serialize_torch_payload(sealed, magic=ROOT_BLOB_MAGIC)
    file_sha = _create_once_binary(
        path,
        serialized,
        validator=lambda value: _deserialize_torch_payload(
            value,
            magic=ROOT_BLOB_MAGIC,
        ),
    )
    observed = _deserialize_torch_payload(
        path.read_bytes(),
        magic=ROOT_BLOB_MAGIC,
    )
    validate_teacher_root_record(
        observed,
        authoritative_row=authoritative_row,
    )
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "file_sha256": file_sha,
        "root_content_sha256": sealed["root_content_sha256"],
        "root_id": sealed["root_id"],
        "ancestry_id": sealed["ancestry_id"],
        "stage": sealed["stage"],
        "row_index": int(authoritative_row["row_index"]),
    }


def load_teacher_root_blob(
    path: Path,
    *,
    authoritative_row: Mapping[str, Any],
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise J2A1ExecutionIntegrityError(
            f"Teacher root blob is missing: {path}"
        )
    if (
        expected_file_sha256 is not None
        and sha256_path(path) != expected_file_sha256
    ):
        raise J2A1ExecutionIntegrityError(
            "Teacher root blob file hash changed"
        )
    record = _deserialize_torch_payload(
        path.read_bytes(),
        magic=ROOT_BLOB_MAGIC,
    )
    return validate_teacher_root_record(
        record,
        authoritative_row=authoritative_row,
    )


def strict_feature_inventory(
    roots: Sequence[Mapping[str, Any]],
    *,
    expected_root_count: int = VALIDATION_PAIRS,
) -> dict[str, Any]:
    if len(roots) != expected_root_count:
        raise J2A1ExecutionIntegrityError(
            "Validation root count is incomplete before family inventory"
        )
    root_ids = [str(root["root_id"]) for root in roots]
    if len(set(root_ids)) != len(root_ids):
        raise J2A1ExecutionIntegrityError(
            "Validation family inventory repeats a root"
        )
    rows: list[dict[str, Any]] = []
    quartiles: Counter[str] = Counter()
    for root in sorted(roots, key=lambda value: str(value["root_id"])):
        transitions = list(root["transitions"])
        if not transitions:
            raise J2A1ExecutionIntegrityError(
                "Validation family inventory has an empty root"
            )
        for offset, transition in enumerate(transitions):
            family = j2.feature_family(transition["board"])
            quartile = min(3, (4 * offset) // len(transitions))
            quartiles[f"q{quartile + 1}"] += 1
            rows.append(
                {
                    "root_id": str(root["root_id"]),
                    "transition_index": int(
                        transition["transition_index"]
                    ),
                    "family": family,
                    "trajectory_quartile": quartile + 1,
                }
            )
    rows.sort(
        key=lambda row: (
            row["family"],
            row["root_id"],
            row["transition_index"],
        )
    )
    natural_counts = Counter(str(row["family"]) for row in rows)
    roots_by_family = {
        family: {
            str(row["root_id"])
            for row in rows
            if row["family"] == family
        }
        for family in FEATURE_FAMILIES
    }
    k = min(
        (natural_counts.get(family, 0) for family in FEATURE_FAMILIES),
        default=0,
    )
    capped_rows = [
        row
        for family in FEATURE_FAMILIES
        for row in [
            candidate for candidate in rows if candidate["family"] == family
        ][:k]
    ]
    capped_counts = Counter(str(row["family"]) for row in capped_rows)
    natural_total = len(rows)
    capped_total = len(capped_rows)
    natural_frequencies = {
        family: (
            natural_counts.get(family, 0) / natural_total
            if natural_total
            else 0.0
        )
        for family in FEATURE_FAMILIES
    }
    capped_frequencies = {
        family: (
            capped_counts.get(family, 0) / capped_total
            if capped_total
            else 0.0
        )
        for family in FEATURE_FAMILIES
    }
    checks = {
        "all_complete_roots_retained": len(set(root_ids))
        == expected_root_count,
        "all_natural_states_retained": natural_total
        == sum(len(root["transitions"]) for root in roots),
        "minimum_1024_states_each": all(
            natural_counts.get(family, 0) >= MIN_STATES_PER_FAMILY
            for family in FEATURE_FAMILIES
        ),
        "minimum_256_roots_each": all(
            len(roots_by_family[family]) >= MIN_ROOTS_PER_FAMILY
            for family in FEATURE_FAMILIES
        ),
        "natural_max_share_strictly_below_070": max(
            natural_frequencies.values(),
            default=1.0,
        )
        < NATURAL_MAX_SHARE,
        "capped_inventory_nonempty_and_four_family": k > 0
        and all(
            capped_counts.get(family, 0) == k
            for family in FEATURE_FAMILIES
        ),
        "capped_max_share_strictly_below_040": max(
            capped_frequencies.values(),
            default=1.0,
        )
        < CAPPED_MAX_SHARE,
    }
    return {
        "version": f"{VERSION}_strict_feature_inventory_v1",
        "validation_root_count": len(roots),
        "natural_state_count": natural_total,
        "natural_family_counts": {
            family: natural_counts.get(family, 0)
            for family in FEATURE_FAMILIES
        },
        "natural_family_frequencies": natural_frequencies,
        "natural_family_root_counts": {
            family: len(roots_by_family[family])
            for family in FEATURE_FAMILIES
        },
        "capped_k_per_family": k,
        "capped_state_count": capped_total,
        "capped_family_counts": {
            family: capped_counts.get(family, 0)
            for family in FEATURE_FAMILIES
        },
        "capped_family_frequencies": capped_frequencies,
        "trajectory_quartile_counts": dict(sorted(quartiles.items())),
        "natural_inventory_sha256": canonical_json_hash(rows),
        "capped_inventory_sha256": canonical_json_hash(capped_rows),
        "capped_refs": [
            {
                "root_id": row["root_id"],
                "transition_index": row["transition_index"],
                "family": row["family"],
            }
            for row in capped_rows
        ],
        "checks": checks,
        "passes": all(checks.values()),
        "decision": (
            "READY_J2A1_DISTILLATION_OPTIMIZER"
            if all(checks.values())
            else HOLD_FAMILY
        ),
    }


def build_distillation_batch(
    roots: Sequence[Mapping[str, Any]],
    *,
    expected_root_count: int = BC_ROOTS,
) -> j2.DistillationBatch:
    if len(roots) != expected_root_count:
        raise J2A1ExecutionIntegrityError(
            "BC root count is incomplete before optimizer construction"
        )
    ordered = sorted(roots, key=lambda root: str(root["root_id"]))
    root_ids = [str(root["root_id"]) for root in ordered]
    if len(set(root_ids)) != len(root_ids):
        raise J2A1ExecutionIntegrityError("BC roots are duplicated")
    lengths = [len(root["transitions"]) for root in ordered]
    if any(length <= 0 for length in lengths):
        raise J2A1ExecutionIntegrityError("BC root is empty")
    observations = np.stack(
        [
            _as_observation(transition["observation"])
            for root in ordered
            for transition in root["transitions"]
        ],
        axis=0,
    )
    legal_masks = np.stack(
        [
            _as_legal_mask(transition["legal_mask"])
            for root in ordered
            for transition in root["transitions"]
        ],
        axis=0,
    )
    actions = np.asarray(
        [
            int(transition["teacher_action"])
            for root in ordered
            for transition in root["transitions"]
        ],
        dtype=np.int64,
    )
    targets = np.asarray(
        [
            float(transition["value_target"])
            for root in ordered
            for transition in root["transitions"]
        ],
        dtype=np.float32,
    )
    weights = j2.root_equal_weights(lengths).astype(np.float32)
    repeated_ids = tuple(
        root_id
        for root_id, length in zip(root_ids, lengths)
        for _ in range(length)
    )
    batch = j2.DistillationBatch(
        observations=torch.from_numpy(observations),
        legal_masks=torch.from_numpy(legal_masks),
        teacher_actions=torch.from_numpy(actions),
        value_targets=torch.from_numpy(targets),
        row_weights=torch.from_numpy(weights),
        root_ids=repeated_ids,
    )
    j2.validate_distillation_batch(batch)
    return batch


def _root_equal_mean(
    values: Sequence[float],
    root_ids: Sequence[str],
) -> float:
    if not values or len(values) != len(root_ids):
        raise J2A1ExecutionIntegrityError(
            "Root-equal metric rows are malformed"
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for root_id, value in zip(root_ids, values):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise J2A1ExecutionIntegrityError(
                "Root-equal metric is nonfinite"
            )
        grouped[str(root_id)].append(numeric)
    return float(
        statistics.fmean(
            statistics.fmean(group)
            for _, group in sorted(grouped.items())
        )
    )


def mechanism_metrics(
    model: j2.J2ActorCritic,
    validation_roots: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    if inventory.get("passes") is not True:
        raise J2A1ExecutionDataHold(
            "Family support must pass before mechanism analysis"
        )
    by_ref: dict[tuple[str, int], Mapping[str, Any]] = {}
    all_rows = []
    for root in validation_roots:
        root_id = str(root["root_id"])
        for transition in root["transitions"]:
            key = (root_id, int(transition["transition_index"]))
            if key in by_ref:
                raise J2A1ExecutionIntegrityError(
                    "Validation transition identity is duplicated"
                )
            by_ref[key] = transition
            all_rows.append((root_id, transition))
    model.eval()
    accuracies = []
    losses = []
    value_errors = []
    zero_errors = []
    selected_actions = []
    root_ids = []
    with torch.no_grad():
        for root_id, transition in all_rows:
            observation = torch.from_numpy(
                _as_observation(transition["observation"])
            ).unsqueeze(0)
            legal = torch.from_numpy(
                _as_legal_mask(transition["legal_mask"])
            ).unsqueeze(0)
            logits, value = model(observation)
            legal_logits = j2.masked_logits(logits, legal)
            action = int(torch.argmax(legal_logits, dim=1)[0])
            target_action = int(transition["teacher_action"])
            target_value = float(transition["value_target"])
            loss = torch.nn.functional.cross_entropy(
                legal_logits,
                torch.tensor([target_action], dtype=torch.int64),
            )
            selected_actions.append(action)
            accuracies.append(float(action == target_action))
            losses.append(float(loss))
            value_errors.append(float((float(value[0]) - target_value) ** 2))
            zero_errors.append(float(target_value**2))
            root_ids.append(root_id)
    illegal = sum(
        not _as_legal_mask(transition["legal_mask"])[action]
        for action, (_, transition) in zip(selected_actions, all_rows)
    )
    family_accuracies = {}
    for family in FEATURE_FAMILIES:
        refs = [
            (str(row["root_id"]), int(row["transition_index"]))
            for row in inventory["capped_refs"]
            if row["family"] == family
        ]
        family_values = []
        family_roots = []
        for ref in refs:
            transition = by_ref.get(ref)
            if transition is None:
                raise J2A1ExecutionIntegrityError(
                    "Capped inventory reference is missing"
                )
            observation = torch.from_numpy(
                _as_observation(transition["observation"])
            ).unsqueeze(0)
            legal = torch.from_numpy(
                _as_legal_mask(transition["legal_mask"])
            ).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(observation)
            selected = int(torch.argmax(j2.masked_logits(logits, legal), dim=1))
            family_values.append(
                float(selected == int(transition["teacher_action"]))
            )
            family_roots.append(ref[0])
        family_accuracies[family] = _root_equal_mean(
            family_values,
            family_roots,
        )
    metrics = {
        "overall_root_equal_accuracy": _root_equal_mean(
            accuracies,
            root_ids,
        ),
        "family_accuracies": family_accuracies,
        "policy_loss": _root_equal_mean(losses, root_ids),
        "value_mse": _root_equal_mean(value_errors, root_ids),
        "zero_value_mse": _root_equal_mean(zero_errors, root_ids),
        "illegal_teacher_or_student_actions": int(illegal),
        "validation_root_count": len(validation_roots),
        "validation_transition_count": len(all_rows),
        "validation_rows_sha256": scientific_hash(
            [
                {
                    "root_id": root_id,
                    "transition_index": transition["transition_index"],
                    "teacher_action": transition["teacher_action"],
                    "value_target": transition["value_target"],
                }
                for root_id, transition in all_rows
            ]
        ),
        "inventory_sha256": inventory["natural_inventory_sha256"],
        "capped_inventory_sha256": inventory["capped_inventory_sha256"],
    }
    gate = j2.bc_mechanism_gate(
        overall_root_equal_accuracy=metrics[
            "overall_root_equal_accuracy"
        ],
        family_accuracies=family_accuracies,
        policy_loss=metrics["policy_loss"],
        value_mse=metrics["value_mse"],
        zero_value_mse=metrics["zero_value_mse"],
        illegal_teacher_actions=metrics[
            "illegal_teacher_or_student_actions"
        ],
        inventory=inventory,
    )
    return {
        "version": f"{VERSION}_mechanism_metrics_v1",
        "metrics": metrics,
        "gate_checks": gate["checks"],
        "passes": bool(gate["passes"]),
        "decision": (
            "READY_J2A1_CLOSED_LOOP_FIDELITY"
            if gate["passes"]
            else HOLD_MECHANISM
        ),
    }


def _mh_log_or(
    treatment_success: np.ndarray,
    control_success: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    a = treatment_success.astype(np.float64)
    b = totals - a
    c = control_success.astype(np.float64)
    d = totals - c
    n = a + b + c + d
    numerator = np.sum(a * d / n, axis=-1)
    denominator = np.sum(b * c / n, axis=-1)
    zero = (numerator <= 0.0) | (denominator <= 0.0)
    if np.any(zero):
        az = a[zero] + 0.5
        bz = b[zero] + 0.5
        cz = c[zero] + 0.5
        dz = d[zero] + 0.5
        nz = az + bz + cz + dz
        numerator[zero] = np.sum(az * dz / nz, axis=-1)
        denominator[zero] = np.sum(bz * cz / nz, axis=-1)
    return np.log(numerator / denominator)


def _score_bootstrap_bounds(
    differences: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SCORE_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if differences.ndim != 1 or differences.size < 1:
        raise J2A1ExecutionIntegrityError(
            "Score bootstrap differences are malformed"
        )
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    chunk = 256
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        indices = rng.integers(
            0,
            differences.size,
            size=(count, differences.size),
        )
        values[start : start + count] = differences[indices].mean(axis=1)
    return (
        float(
            np.quantile(
                values,
                BOOTSTRAP_QUANTILES[0],
                method=QUANTILE_METHOD,
            )
        ),
        float(
            np.quantile(
                values,
                BOOTSTRAP_QUANTILES[1],
                method=QUANTILE_METHOD,
            )
        ),
    )


def _progression_bootstrap_bounds(
    student: np.ndarray,
    teacher: np.ndarray,
    strata: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = PROGRESSION_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if not (
        student.shape == teacher.shape == strata.shape
        and student.ndim == 1
    ):
        raise J2A1ExecutionIntegrityError(
            "Progression bootstrap rows are malformed"
        )
    rng = np.random.default_rng(seed)
    student_counts = []
    teacher_counts = []
    totals = []
    for stratum in range(VALIDATION_STRATA):
        mask = strata == stratum
        student_block = student[mask].astype(np.int64)
        teacher_block = teacher[mask].astype(np.int64)
        roots = int(student_block.size)
        if roots < 1:
            raise J2A1ExecutionIntegrityError(
                "Progression bootstrap stratum is empty"
            )
        indices = rng.integers(0, roots, size=(replicates, roots))
        student_counts.append(student_block[indices].sum(axis=1))
        teacher_counts.append(teacher_block[indices].sum(axis=1))
        totals.append(roots)
    student_matrix = np.stack(student_counts, axis=1)
    teacher_matrix = np.stack(teacher_counts, axis=1)
    total_matrix = np.broadcast_to(
        np.asarray(totals, dtype=np.float64),
        student_matrix.shape,
    )
    values = _mh_log_or(student_matrix, teacher_matrix, total_matrix)
    return (
        float(
            np.quantile(
                values,
                BOOTSTRAP_QUANTILES[0],
                method=QUANTILE_METHOD,
            )
        ),
        float(
            np.quantile(
                values,
                BOOTSTRAP_QUANTILES[1],
                method=QUANTILE_METHOD,
            )
        ),
    )


def validate_complete_pair_records(
    pair_records: Sequence[Mapping[str, Any]],
    authoritative_rows: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int = VALIDATION_PAIRS,
) -> list[dict[str, Any]]:
    if (
        len(pair_records) != expected_pairs
        or len(authoritative_rows) != expected_pairs
    ):
        raise J2A1ExecutionIntegrityError(
            "Closed-loop pair panel is incomplete"
        )
    expected = {
        str(row["root_id"]): j2.json_native(row)
        for row in authoritative_rows
    }
    if len(expected) != expected_pairs:
        raise J2A1ExecutionIntegrityError(
            "Closed-loop authority repeats a root"
        )
    normalized = []
    seen = set()
    for record in pair_records:
        root_id = str(record.get("root_id"))
        if root_id in seen or root_id not in expected:
            raise J2A1ExecutionIntegrityError(
                "Closed-loop pair root is duplicate or unknown"
            )
        seen.add(root_id)
        if j2.json_native(record.get("row")) != expected[root_id]:
            raise J2A1ExecutionIntegrityError(
                "Closed-loop pair row changed from authority"
            )
        if record.get("pair_complete") is not True:
            raise J2A1ExecutionIntegrityError(
                "Closed-loop pair is partial"
            )
        arms = {}
        for arm_name in ("student", "teacher"):
            arm = record.get(arm_name)
            if not isinstance(arm, Mapping):
                raise J2A1ExecutionIntegrityError(
                    "Closed-loop pair arm is missing"
                )
            start_score = int(arm.get("start_score", -1))
            final_score = int(arm.get("final_score", -1))
            max_tile = int(arm.get("max_tile", -1))
            moves = int(arm.get("moves", -1))
            latency = float(arm.get("latency_seconds", math.nan))
            survival = float(arm.get("survival", math.nan))
            illegal_actions = int(arm.get("illegal_actions", -1))
            if (
                start_score < 0
                or final_score < start_score
                or max_tile < 0
                or moves < 0
                or not math.isfinite(latency)
                or latency < 0.0
                or not math.isfinite(survival)
                or illegal_actions < 0
            ):
                raise J2A1ExecutionIntegrityError(
                    "Closed-loop pair arm metric is malformed"
                )
            arms[arm_name] = {
                "start_score": start_score,
                "final_score": final_score,
                "max_tile": max_tile,
                "moves": moves,
                "latency_seconds": latency,
                "survival": survival,
                "illegal_actions": illegal_actions,
                "arm_file_sha256": _validate_hex(
                    arm.get("arm_file_sha256"),
                    name=f"{arm_name}_arm_file_sha256",
                ),
            }
        normalized.append(
            {
                "root_id": root_id,
                "row": expected[root_id],
                "stratum": int(expected[root_id]["row_index"])
                % VALIDATION_STRATA,
                **arms,
            }
        )
    if seen != set(expected):
        raise J2A1ExecutionIntegrityError(
            "Closed-loop pair panel omitted a root"
        )
    return sorted(
        normalized,
        key=lambda row: int(row["row"]["row_index"]),
    )


def analyze_closed_loop_fidelity(
    pair_records: Sequence[Mapping[str, Any]],
    authoritative_rows: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int = VALIDATION_PAIRS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    score_seed: int = SCORE_BOOTSTRAP_SEED,
    progression_seed: int = PROGRESSION_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if (
        bootstrap_replicates != BOOTSTRAP_REPLICATES
        or score_seed != SCORE_BOOTSTRAP_SEED
        or progression_seed != PROGRESSION_BOOTSTRAP_SEED
    ):
        raise J2A1ExecutionIntegrityError(
            "Closed-loop bootstrap contract changed"
        )
    rows = validate_complete_pair_records(
        pair_records,
        authoritative_rows,
        expected_pairs=expected_pairs,
    )
    score_differences = np.asarray(
        [
            math.log1p(
                max(
                    int(row["student"]["final_score"])
                    - int(row["student"]["start_score"]),
                    0,
                )
            )
            - math.log1p(
                max(
                    int(row["teacher"]["final_score"])
                    - int(row["teacher"]["start_score"]),
                    0,
                )
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    score_point = float(score_differences.mean())
    score_lower, score_upper = _score_bootstrap_bounds(
        score_differences,
        replicates=bootstrap_replicates,
        seed=score_seed,
    )
    student_p1536 = np.asarray(
        [int(row["student"]["max_tile"]) >= 1536 for row in rows],
        dtype=np.int64,
    )
    teacher_p1536 = np.asarray(
        [int(row["teacher"]["max_tile"]) >= 1536 for row in rows],
        dtype=np.int64,
    )
    strata = np.asarray([int(row["stratum"]) for row in rows], dtype=np.int64)
    student_counts = np.asarray(
        [
            int(student_p1536[strata == stratum].sum())
            for stratum in range(VALIDATION_STRATA)
        ],
        dtype=np.float64,
    )[None, :]
    teacher_counts = np.asarray(
        [
            int(teacher_p1536[strata == stratum].sum())
            for stratum in range(VALIDATION_STRATA)
        ],
        dtype=np.float64,
    )[None, :]
    totals = np.full(
        (1, VALIDATION_STRATA),
        expected_pairs // VALIDATION_STRATA,
        dtype=np.float64,
    )
    progression_log_or = float(
        _mh_log_or(student_counts, teacher_counts, totals)[0]
    )
    progression_lower, progression_upper = _progression_bootstrap_bounds(
        student_p1536,
        teacher_p1536,
        strata,
        replicates=bootstrap_replicates,
        seed=progression_seed,
    )
    control_rate = float(teacher_p1536.mean())
    point_or = math.exp(progression_log_or)
    lower_or = math.exp(progression_lower)
    upper_or = math.exp(progression_upper)
    stratum_counts = [
        int(np.sum(strata == stratum))
        for stratum in range(VALIDATION_STRATA)
    ]
    illegal = sum(
        int(row["student"]["illegal_actions"]) for row in rows
    )
    finite_latency_survival = all(
        math.isfinite(float(row[arm][field]))
        for row in rows
        for arm in ("student", "teacher")
        for field in ("latency_seconds", "survival")
    )
    low_base_rate = control_rate < 0.02
    checks = {
        "all_6144_pairs_retained": expected_pairs == VALIDATION_PAIRS
        and len(rows) == VALIDATION_PAIRS,
        "eight_equal_strata": stratum_counts
        == [VALIDATION_PAIRS_PER_STRATUM] * VALIDATION_STRATA,
        "score_point_above_log_097": score_point > math.log(0.97),
        "score_lower_above_log_090": score_lower > math.log(0.90),
        "p1536_control_rate_at_least_002": not low_base_rate,
        "progression_point_at_least_090": point_or >= 0.90,
        "progression_lower_above_050": lower_or > 0.50,
        "illegal_student_actions_zero": illegal == 0,
        "latency_and_survival_finite": finite_latency_survival,
        "bootstrap_contract_exact": bootstrap_replicates
        == BOOTSTRAP_REPLICATES
        and score_seed == SCORE_BOOTSTRAP_SEED
        and progression_seed == PROGRESSION_BOOTSTRAP_SEED,
    }
    if low_base_rate:
        decision = HOLD_BASE_RATE
    elif all(checks.values()):
        decision = READY_EXECUTION
    else:
        decision = HOLD_FIDELITY
    descriptive = {
        "student_final_score_max": max(
            int(row["student"]["final_score"]) for row in rows
        ),
        "teacher_final_score_max": max(
            int(row["teacher"]["final_score"]) for row in rows
        ),
        "student_final_score_p95": float(
            np.quantile(
                [row["student"]["final_score"] for row in rows],
                0.95,
                method=QUANTILE_METHOD,
            )
        ),
        "student_final_score_p99": float(
            np.quantile(
                [row["student"]["final_score"] for row in rows],
                0.99,
                method=QUANTILE_METHOD,
            )
        ),
        "stratum_score_signs": [
            float(score_differences[strata == stratum].mean())
            for stratum in range(VALIDATION_STRATA)
        ],
    }
    return {
        "version": f"{VERSION}_closed_loop_fidelity_v1",
        "decision": decision,
        "passes": decision == READY_EXECUTION,
        "pair_count": len(rows),
        "panel_sha256": scientific_hash(rows),
        "score": {
            "point": score_point,
            "lower_95": score_lower,
            "upper_95": score_upper,
        },
        "p1536": {
            "teacher_control_rate": control_rate,
            "common_or_point": point_or,
            "common_or_lower_95": lower_or,
            "common_or_upper_95": upper_or,
            "student_stratum_successes": student_counts[0].astype(int).tolist(),
            "teacher_stratum_successes": teacher_counts[0].astype(int).tolist(),
        },
        "bootstrap": execution_schema()["bootstrap"],
        "checks": checks,
        "descriptive_safeguards": descriptive,
        "family_and_stratum_signs_conjunctive": False,
        "first_action_gate_used": False,
        "full_policy_sustained_exposure": True,
    }


def runtime_storage_projection() -> dict[str, Any]:
    teacher_transitions = ACTIVE_ROOTS * PLANNING_MOVES
    bc_rows = BC_ROOTS * PLANNING_MOVES
    student_calls = VALIDATION_PAIRS * PLANNING_MOVES
    teacher_root_blob_bytes = (
        teacher_transitions * ROOT_BLOB_BYTES_PER_TRANSITION
    )
    ephemeral_bc_batch_bytes = bc_rows * BC_BATCH_BYTES_PER_ROW
    final_pair_blob_bytes = VALIDATION_PAIRS * PAIR_BLOB_BYTES
    active_chunk_bytes = (
        SHARDS * PLANNING_MOVES * ACTIVE_CHUNK_BYTES_PER_TRANSITION
    )
    before_margin = (
        teacher_root_blob_bytes
        + ephemeral_bc_batch_bytes
        + final_pair_blob_bytes
        + active_chunk_bytes
        + FIXED_OVERHEAD_BYTES
    )
    after_margin = int(before_margin * SAFETY_MULTIPLIER)
    teacher_calls = ACTIVE_ROOTS * PLANNING_MOVES
    teacher_hours = (
        teacher_calls * TEACHER_P99_SECONDS / SHARDS / 3600.0
        + OPTIMIZER_FIXTURE_HOURS
    ) * SAFETY_MULTIPLIER
    student_hours = (
        student_calls
        * (
            ACTOR_BATCH16_P99_SECONDS
            + SIM_TRANSITION_P99_SECONDS
        )
        / 3600.0
        * SAFETY_MULTIPLIER
    )
    total_hours = teacher_hours + student_hours + INHERITED_ADMIN_HOURS
    sensitivity_teacher_hours = (
        ACTIVE_ROOTS
        * SENSITIVITY_MOVES
        * TEACHER_P99_SECONDS
        / SHARDS
        / 3600.0
        + OPTIMIZER_FIXTURE_HOURS
    ) * SAFETY_MULTIPLIER
    sensitivity_storage = int(
        (
            ACTIVE_ROOTS
            * SENSITIVITY_MOVES
            * ROOT_BLOB_BYTES_PER_TRANSITION
            + BC_ROOTS
            * SENSITIVITY_MOVES
            * BC_BATCH_BYTES_PER_ROW
            + final_pair_blob_bytes
            + SHARDS
            * SENSITIVITY_MOVES
            * ACTIVE_CHUNK_BYTES_PER_TRANSITION
            + FIXED_OVERHEAD_BYTES
        )
        * SAFETY_MULTIPLIER
    )
    optimizer_rows = bc_rows
    minibatches_per_epoch = math.ceil(
        optimizer_rows / j2.MINIBATCH_SIZE
    )
    optimizer_steps = j2.DISTILLATION_EPOCHS * minibatches_per_epoch
    charged_units = ACTIVE_ROOTS + optimizer_steps + VALIDATION_PAIRS
    checks = {
        "central_peak_within_24gib": after_margin <= STORAGE_CAP_BYTES,
        "central_runtime_within_72h": total_hours <= RUNTIME_CAP_HOURS,
        "teacher_runtime_uses_measured_p99_divided_by_eight": math.isclose(
            teacher_hours,
            41.98247721555496,
            abs_tol=1e-12,
            rel_tol=0.0,
        ),
        "root_and_batch_duplication_counted_at_peak": (
            teacher_root_blob_bytes > 0
            and ephemeral_bc_batch_bytes > 0
        ),
        "transition_chunks_current_stage_only": active_chunk_bytes
        == SHARDS
        * PLANNING_MOVES
        * ACTIVE_CHUNK_BYTES_PER_TRANSITION,
        "ephemeral_batch_current_stage_only": ephemeral_bc_batch_bytes
        == BC_ROOTS * PLANNING_MOVES * BC_BATCH_BYTES_PER_ROW,
        "active_stream_count_exact": ACTIVE_STREAMS == 63_488,
        "charged_unit_count_derived": charged_units
        == ACTIVE_ROOTS + optimizer_steps + VALIDATION_PAIRS,
        "sensitivity_reported_not_substituted": SENSITIVITY_MOVES == 5_000,
    }
    return {
        "version": f"{VERSION}_runtime_storage_projection_v1",
        "central_512_moves": {
            "teacher_transitions": teacher_transitions,
            "bc_tensor_rows": bc_rows,
            "student_fidelity_calls": student_calls,
            "teacher_root_blob_bytes": teacher_root_blob_bytes,
            "ephemeral_bc_batch_bytes": ephemeral_bc_batch_bytes,
            "final_pair_blob_bytes": final_pair_blob_bytes,
            "active_chunk_bytes": active_chunk_bytes,
            "fixed_overhead_bytes": FIXED_OVERHEAD_BYTES,
            "peak_before_margin_bytes": before_margin,
            "peak_after_25pct_margin_bytes": after_margin,
            "peak_after_25pct_margin_gib": after_margin / 1024**3,
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "storage_headroom_bytes": STORAGE_CAP_BYTES - after_margin,
            "teacher_runtime_hours_after_margin": teacher_hours,
            "student_runtime_hours_after_margin": student_hours,
            "inherited_admin_hours": INHERITED_ADMIN_HOURS,
            "total_runtime_hours_after_margin": total_hours,
            "runtime_cap_hours": RUNTIME_CAP_HOURS,
            "runtime_headroom_hours": RUNTIME_CAP_HOURS - total_hours,
            "optimizer_rows": optimizer_rows,
            "minibatches_per_epoch": minibatches_per_epoch,
            "optimizer_steps": optimizer_steps,
            "charged_units": charged_units,
            "attempt_journal_events": 2 * charged_units,
            "projected_immutable_root_files": ACTIVE_ROOTS,
            "projected_immutable_pair_files": VALIDATION_PAIRS,
            "rolling_slots": 2,
            "maximum_orphans": 1,
        },
        "sensitivity_5000_moves": {
            "diagnostic_not_conjunctive": True,
            "teacher_runtime_hours_after_margin": sensitivity_teacher_hours,
            "peak_storage_after_margin_bytes": sensitivity_storage,
            "runtime_fits_72h": sensitivity_teacher_hours
            <= RUNTIME_CAP_HOURS,
            "storage_fits_24gib": sensitivity_storage
            <= STORAGE_CAP_BYTES,
        },
        "method": {
            "teacher_runtime":
                "V2 measured steady-state p99 / eight fixed workers",
            "student_runtime":
                "sealed actor-batch16 p99 plus simulator-transition p99",
            "storage":
                "historical roots + current-stage chunks + one ephemeral BC "
                "batch + final pairs + fixed durability envelope",
            "safety_multiplier": SAFETY_MULTIPLIER,
        },
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": dict(ZERO_WORK),
    }


def operational_audit(
    *,
    output_dir: Path,
    include_services: bool,
) -> dict[str, Any]:
    if include_services:
        configure_deterministic_runtime()
        parent = j2.operational_audit(output_dir=output_dir)
        checks = {
            "parent_operations_pass": parent.get("passes") is True,
            "nice_at_least_10": int(parent["parent_operational"]["nice"])
            >= 10,
            "free_disk_above_100_gib": float(
                parent["parent_operational"]["free_disk_gib"]
            )
            > HARD_DISK_FLOOR_GIB,
            "target_120_gib_met": float(
                parent["parent_operational"]["free_disk_gib"]
            )
            > TARGET_DISK_GIB,
            "services_healthy": parent["parent_operational"]["services"].get(
                "passes"
            )
            is True,
            "one_heavy_job": parent["parent_operational"]["process"].get(
                "passes"
            )
            is True,
            "torch_intra_inter_one": parent["torch_runtime"][
                "intra_op_threads"
            ]
            == 1
            and parent["torch_runtime"]["inter_op_threads"] == 1,
            "torch_deterministic": parent["torch_runtime"][
                "deterministic_algorithms"
            ]
            is True,
            "dashboard_top_three_exact": parent["parent_operational"][
                "services"
            ]["dashboard"]["top_three"]
            is not None
            and list(
                parent["parent_operational"]["services"]["dashboard"][
                    "top_three"
                ]
            )
            == [263670, 261369, 258561],
            "human_session_content_unread": parent.get(
                "human_session_content_read"
            )
            is False,
        }
    else:
        parent = {
            "passes": True,
            "fixture_only": True,
            "human_session_content_read": False,
        }
        checks = {
            "parent_operations_pass": True,
            "nice_at_least_10": True,
            "free_disk_above_100_gib": True,
            "target_120_gib_met": True,
            "services_healthy": True,
            "one_heavy_job": True,
            "torch_intra_inter_one": True,
            "torch_deterministic": True,
            "dashboard_top_three_exact": True,
            "human_session_content_unread": True,
        }
    return {
        "version": f"{VERSION}_operational_audit_v1",
        "parent": parent,
        "checks": checks,
        "passes": all(checks.values()),
    }


def readiness_namespace_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "test_evidence": output_dir / TEST_EVIDENCE_NAME,
        "input_bindings": output_dir / INPUT_BINDINGS_NAME,
        "authority": output_dir / AUTHORITY_AUDIT_NAME,
        "schema": output_dir / SCHEMA_NAME,
        "projection": output_dir / PROJECTION_NAME,
        "lock": output_dir / READINESS_LOCK_NAME,
        "result": output_dir / READINESS_RESULT_NAME,
        "retention": output_dir / RETENTION_NAME,
    }


def audit_zero_work(
    *,
    output_dir: Path = READINESS_DIR,
    include_operational: bool,
    allowed_files: Sequence[str] = (),
) -> dict[str, Any]:
    entries = (
        sorted(path.name for path in output_dir.iterdir())
        if output_dir.exists()
        else []
    )
    allowed = set(allowed_files)
    checks = {
        "namespace_has_only_allowed_files": set(entries) <= allowed,
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "all_work_counters_zero": all(
            int(value) == 0 for value in ZERO_WORK.values()
        ),
        "no_scientific_artifact_names": not any(
            name in entries
            for name in (
                PHASE_LOCK_NAME,
                OPEN_MARKER_NAME,
                MATERIALIZED_MANIFEST_NAME,
                RESERVATION_NAME,
                CONSUMPTION_NAME,
                GENESIS_NAME,
                TERMINAL_NAME,
            )
        ),
    }
    operations = operational_audit(
        output_dir=output_dir,
        include_services=include_operational,
    )
    checks["operations_pass"] = operations["passes"]
    return {
        "version": f"{VERSION}_zero_work_audit_v1",
        "output_dir": str(output_dir.resolve()),
        "entries": entries,
        "allowed_files": sorted(allowed),
        "future_execution_dir": str(FUTURE_EXECUTION_DIR.resolve()),
        "zero_work": dict(ZERO_WORK),
        "operational": operations,
        "checks": checks,
        "passes": all(checks.values()),
    }


def test_evidence_payload(
    *,
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
) -> dict[str, Any]:
    normalized_commands = [j2.json_native(row) for row in commands]
    if not normalized_commands or not all(
        isinstance(row, dict) for row in normalized_commands
    ):
        raise J2A1ExecutionIntegrityError(
            "Test evidence commands are missing"
        )
    source_identities = {
        str(path.relative_to(REPO_ROOT)): sha256_path(path)
        for path in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)
    }
    checks = {
        "all_commands_passed": all(
            int(row.get("failed", -1)) == 0
            and int(row.get("passed", 0)) > 0
            for row in normalized_commands
        ),
        "commands_unique": len(
            {str(row.get("command")) for row in normalized_commands}
        )
        == len(normalized_commands),
        "source_files_exist": all(
            path.is_file() for path in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)
        ),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "zero_work_exact": all(value == 0 for value in ZERO_WORK.values()),
    }
    return {
        "version": f"{VERSION}_test_evidence_v1",
        "source_identities": source_identities,
        "commands": normalized_commands,
        "deselections": list(deselections),
        "totals": {
            "passed": sum(int(row["passed"]) for row in normalized_commands),
            "failed": sum(int(row["failed"]) for row in normalized_commands),
            "deselected": sum(
                int(row.get("deselected", 0))
                for row in normalized_commands
            ),
        },
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": dict(ZERO_WORK),
    }


def write_test_evidence(
    *,
    output_dir: Path,
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
) -> dict[str, Any]:
    before = audit_zero_work(
        output_dir=output_dir,
        include_operational=False,
        allowed_files=(),
    )
    if not before["passes"]:
        raise J2A1ExecutionIntegrityError(
            "Readiness namespace was not fresh before test evidence"
        )
    payload = test_evidence_payload(
        commands=commands,
        deselections=deselections,
    )
    if not payload["passes"]:
        raise J2A1ExecutionIntegrityError(
            "Frozen test evidence did not pass"
        )
    return write_immutable_json(
        output_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def _readiness_inputs(
    *,
    output_dir: Path,
    require_future_execution_absent: bool,
) -> dict[str, Any]:
    paths = readiness_namespace_paths(output_dir)
    evidence = load_json(paths["test_evidence"])
    if not verify_payload_hash(
        evidence,
        "test_evidence_payload_sha256",
    ) or evidence.get("passes") is not True:
        raise J2A1ExecutionIntegrityError(
            "Readiness test evidence is invalid"
        )
    current_sources = {
        str(path.relative_to(REPO_ROOT)): sha256_path(path)
        for path in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)
    }
    if evidence.get("source_identities") != current_sources:
        raise J2A1ExecutionIntegrityError(
            "Readiness source changed after tests"
        )
    sources = source_and_parent_audit(
        require_future_execution_absent=require_future_execution_absent,
    )
    authority = authority_audit()
    schema = execution_schema()
    projection = runtime_storage_projection()
    if not sources["passes"] or not authority["passes"]:
        raise J2A1ExecutionIntegrityError(
            "Immutable source or authority audit failed"
        )
    return {
        "test_evidence": evidence,
        "test_evidence_identity": artifact_identity(
            paths["test_evidence"],
            "test_evidence_payload_sha256",
        ),
        "sources": sources,
        "authority": authority,
        "schema": schema,
        "projection": projection,
    }


def _readiness_decision(
    *,
    integrity: Mapping[str, bool],
    feasibility: Mapping[str, bool],
    operations: Mapping[str, bool],
) -> dict[str, Any]:
    if not all(integrity.values()):
        decision = KILL
    elif not all(feasibility.values()) or not all(operations.values()):
        decision = HOLD
    else:
        decision = READY
    return {
        "decision": decision,
        "integrity_checks": dict(integrity),
        "feasibility_checks": dict(feasibility),
        "operational_checks": dict(operations),
        "passes": decision == READY,
        "execution_authorized": False,
    }


def prepare_readiness(
    *,
    output_dir: Path = READINESS_DIR,
    include_operational: bool = True,
) -> dict[str, Any]:
    paths = readiness_namespace_paths(output_dir)
    before = audit_zero_work(
        output_dir=output_dir,
        include_operational=include_operational,
        allowed_files=(TEST_EVIDENCE_NAME,),
    )
    inputs = _readiness_inputs(
        output_dir=output_dir,
        require_future_execution_absent=True,
    )
    operations = before["operational"]
    authority_payload = payload_with_hash(
        inputs["authority"],
        "authority_audit_payload_sha256",
    )
    schema_payload = payload_with_hash(
        inputs["schema"],
        "execution_schema_payload_sha256",
    )
    projection_payload = payload_with_hash(
        inputs["projection"],
        "projection_payload_sha256",
    )
    source_payload = payload_with_hash(
        {
            "version": f"{VERSION}_input_bindings_v1",
            "sources_and_parents": inputs["sources"],
            "test_evidence": inputs["test_evidence_identity"],
            "zero_work_before_prepare": before,
            "zero_work": dict(ZERO_WORK),
        },
        "input_bindings_payload_sha256",
    )
    write_immutable_json(
        paths["authority"],
        authority_payload,
        field="authority_audit_payload_sha256",
    )
    write_immutable_json(
        paths["schema"],
        schema_payload,
        field="execution_schema_payload_sha256",
    )
    write_immutable_json(
        paths["projection"],
        projection_payload,
        field="projection_payload_sha256",
    )
    write_immutable_json(
        paths["input_bindings"],
        source_payload,
        field="input_bindings_payload_sha256",
    )
    predecessor_fields = {
        "test_evidence": "test_evidence_payload_sha256",
        "input_bindings": "input_bindings_payload_sha256",
        "authority": "authority_audit_payload_sha256",
        "schema": "execution_schema_payload_sha256",
        "projection": "projection_payload_sha256",
    }
    predecessor_identities = {
        key: artifact_identity(paths[key], field)
        for key, field in predecessor_fields.items()
    }
    integrity = {
        "zero_work_before_prepare": before["passes"],
        "sources_and_parents_exact": inputs["sources"]["passes"],
        "authority_exact": inputs["authority"]["passes"],
        "model_parameter_count_exact": j2.parameter_count()
        == j2.EXPECTED_PARAMETER_COUNT,
        "schema_no_auxiliary_heads": inputs["schema"][
            "parent_model_schema"
        ]["auxiliary_heads"]
        == [],
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
    }
    feasibility = {
        "runtime_storage_projection_passes": inputs["projection"]["passes"],
        "central_runtime_within_cap": inputs["projection"]["checks"][
            "central_runtime_within_72h"
        ],
        "central_storage_within_cap": inputs["projection"]["checks"][
            "central_peak_within_24gib"
        ],
    }
    operation_checks = {
        key: bool(value) for key, value in operations["checks"].items()
    }
    disposition = _readiness_decision(
        integrity=integrity,
        feasibility=feasibility,
        operations=operation_checks,
    )
    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "predecessors": predecessor_identities,
        "source_identities": inputs["test_evidence"]["source_identities"],
        "active_authority_identity": {
            "canonical_rows_sha256": inputs["authority"][
                "canonical_rows_sha256"
            ],
            "root_set_sha256": inputs["authority"]["root_set_sha256"],
            "ancestry_set_sha256": inputs["authority"][
                "ancestry_set_sha256"
            ],
            "stream_set_sha256": inputs["authority"]["stream_set_sha256"],
            "active_rows": ACTIVE_ROOTS,
            "active_streams": ACTIVE_STREAMS,
        },
        "bootstrap_contract": inputs["schema"]["bootstrap"],
        "future_execution_dir": str(FUTURE_EXECUTION_DIR.resolve()),
        "scientific_execution_authorized": False,
        "zero_work": dict(ZERO_WORK),
    }
    write_immutable_json(
        paths["lock"],
        lock_payload,
        field="readiness_lock_payload_sha256",
    )
    result_payload = {
        "version": f"{VERSION}_readiness_result_v1",
        **disposition,
        "readiness_lock": artifact_identity(
            paths["lock"],
            "readiness_lock_payload_sha256",
        ),
        "active_counts": {
            "bc_roots": BC_ROOTS,
            "validation_pairs": VALIDATION_PAIRS,
            "teacher_roots": ACTIVE_ROOTS,
            "student_fidelity_arms": VALIDATION_PAIRS,
            "game_arms": ACTIVE_ARMS,
            "unique_streams": ACTIVE_STREAMS,
        },
        "runtime_storage": {
            "central_runtime_hours_after_margin": inputs["projection"][
                "central_512_moves"
            ]["total_runtime_hours_after_margin"],
            "central_peak_bytes_after_margin": inputs["projection"][
                "central_512_moves"
            ]["peak_after_25pct_margin_bytes"],
            "storage_headroom_bytes": inputs["projection"][
                "central_512_moves"
            ]["storage_headroom_bytes"],
        },
        "zero_work": dict(ZERO_WORK),
        "next_authority": (
            "research-lead review of future phase lock only"
            if disposition["decision"] == READY
            else "none"
        ),
        "continue": disposition["decision"] == READY,
        "hold": True,
        "kill": disposition["decision"] == KILL,
        "promote": False,
    }
    write_immutable_json(
        paths["result"],
        result_payload,
        field="readiness_result_payload_sha256",
    )
    retention_predecessors = [
        {
            "path": path.name,
            "bytes": int(path.stat().st_size),
            "file_sha256": sha256_path(path),
        }
        for path in (
            paths["test_evidence"],
            paths["input_bindings"],
            paths["authority"],
            paths["schema"],
            paths["projection"],
            paths["lock"],
            paths["result"],
        )
    ]
    retention_payload = {
        "version": f"{VERSION}_readiness_retention_v1",
        "predecessors": retention_predecessors,
        "canonical_inventory_sha256": canonical_json_hash(
            retention_predecessors
        ),
        "predecessor_file_count": len(retention_predecessors),
        "predecessor_bytes": sum(
            row["bytes"] for row in retention_predecessors
        ),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "zero_work": dict(ZERO_WORK),
        "passes": not FUTURE_EXECUTION_DIR.exists(),
    }
    write_immutable_json(
        paths["retention"],
        retention_payload,
        field="retention_payload_sha256",
    )
    observed = load_json(paths["result"])
    if not verify_payload_hash(observed, "readiness_result_payload_sha256"):
        raise J2A1ExecutionIntegrityError(
            "Readiness result changed after sealing"
        )
    return observed


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != content:
        raise J2A1ExecutionIntegrityError(
            f"Atomic replacement changed bytes: {path}"
        )


def _atomic_replace_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    body = payload_with_hash(payload, field)
    content = (
        json.dumps(body, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_replace_bytes(path, content)
    observed = load_json(path)
    if not verify_payload_hash(observed, field) or observed != body:
        raise J2A1ExecutionIntegrityError(
            f"Atomic JSON replacement changed payload: {path}"
        )
    return observed


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_start_identity(pid: int) -> str:
    try:
        completed = os.popen(f"ps -o lstart= -p {int(pid)}").read().strip()
    except OSError as error:
        raise J2A1ExecutionOperationalHold(
            "Process start identity is unavailable"
        ) from error
    if not completed:
        raise J2A1ExecutionOperationalHold(
            "Process start identity is unavailable"
        )
    return completed


class OwnershipLedger:
    """Append-only phase owner/recovery records bound to one contract."""

    def __init__(
        self,
        *,
        path: Path,
        contract_sha256: str,
    ) -> None:
        self.path = path
        self.contract_sha256 = _validate_hex(
            contract_sha256,
            name="ownership contract",
        )
        self.records = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        predecessor = None
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise J2A1ExecutionIntegrityError(
                    f"Ownership line {line_number} is malformed"
                ) from error
            if not verify_payload_hash(record, "owner_record_sha256"):
                raise J2A1ExecutionIntegrityError(
                    f"Ownership line {line_number} changed"
                )
            checks = {
                "sequence": int(record.get("sequence", -1))
                == len(records),
                "predecessor": record.get("predecessor_record_sha256")
                == predecessor,
                "contract": record.get("contract_sha256")
                == self.contract_sha256,
                "kind": record.get("kind") in {"owner", "recovery"},
            }
            if not all(checks.values()):
                raise J2A1ExecutionIntegrityError(
                    f"Ownership line {line_number} is inconsistent"
                )
            records.append(record)
            predecessor = record["owner_record_sha256"]
        if records and records[-1]["kind"] != "owner":
            raise J2A1ExecutionIntegrityError(
                "Ownership ledger does not end in an owner"
            )
        return records

    def _append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = payload_with_hash(
            {
                "version": f"{VERSION}_owner_record_v1",
                "sequence": len(self.records),
                "predecessor_record_sha256": (
                    None
                    if not self.records
                    else self.records[-1]["owner_record_sha256"]
                ),
                "contract_sha256": self.contract_sha256,
                **dict(payload),
            },
            "owner_record_sha256",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        return record

    def acquire(
        self,
        *,
        marker_sha256: str,
        lock_sha256: str,
        manifest_sha256: str,
        command: str,
        execution_mode: str,
        pid: int | None = None,
        start_identity: str | None = None,
    ) -> dict[str, Any]:
        if self.records:
            raise J2A1ExecutionOperationalHold(
                "Phase owner already exists"
            )
        owner_pid = os.getpid() if pid is None else int(pid)
        identity = (
            process_start_identity(owner_pid)
            if start_identity is None
            else str(start_identity)
        )
        return self._append(
            {
                "kind": "owner",
                "marker_sha256": _validate_hex(
                    marker_sha256,
                    name="owner marker",
                ),
                "lock_sha256": _validate_hex(lock_sha256, name="owner lock"),
                "manifest_sha256": _validate_hex(
                    manifest_sha256,
                    name="owner manifest",
                ),
                "runner_sha256": sha256_path(RUNNER_PATH),
                "command": command,
                "execution_mode": execution_mode,
                "hostname": socket.gethostname(),
                "pid": owner_pid,
                "process_start_identity": identity,
            }
        )

    def verify(
        self,
        *,
        marker_sha256: str,
        lock_sha256: str,
        manifest_sha256: str,
        command: str,
        execution_mode: str,
        require_current_pid: bool,
    ) -> dict[str, Any]:
        if not self.records or self.records[-1]["kind"] != "owner":
            raise J2A1ExecutionIntegrityError(
                "Phase ownership is missing"
            )
        owner = self.records[-1]
        checks = {
            "marker": owner.get("marker_sha256") == marker_sha256,
            "lock": owner.get("lock_sha256") == lock_sha256,
            "manifest": owner.get("manifest_sha256") == manifest_sha256,
            "runner": owner.get("runner_sha256") == sha256_path(RUNNER_PATH),
            "command": owner.get("command") == command,
            "execution_mode": owner.get("execution_mode") == execution_mode,
            "owner_live": _pid_alive(int(owner.get("pid", -1))),
            "current_pid": (
                int(owner.get("pid", -1)) == os.getpid()
                if require_current_pid
                else True
            ),
        }
        if not all(checks.values()):
            raise J2A1ExecutionOperationalHold(
                "Phase ownership changed"
            )
        return {"owner": owner, "checks": checks, "passes": True}

    def reclaim_dead(
        self,
        *,
        marker_sha256: str,
        lock_sha256: str,
        manifest_sha256: str,
        command: str,
        execution_mode: str,
        boundary: Mapping[str, Any],
        pid: int | None = None,
        start_identity: str | None = None,
    ) -> dict[str, Any]:
        if not self.records or self.records[-1]["kind"] != "owner":
            raise J2A1ExecutionIntegrityError(
                "Dead-owner reclaim has no predecessor"
            )
        previous = self.records[-1]
        if _pid_alive(int(previous["pid"])):
            raise J2A1ExecutionOperationalHold(
                "Live owner cannot be reclaimed"
            )
        for key, expected in (
            ("marker_sha256", marker_sha256),
            ("lock_sha256", lock_sha256),
            ("manifest_sha256", manifest_sha256),
            ("command", command),
            ("execution_mode", execution_mode),
        ):
            if previous.get(key) != expected:
                raise J2A1ExecutionIntegrityError(
                    f"Dead-owner reclaim changed {key}"
                )
        if (
            boundary.get("passes") is not True
            or not verify_payload_hash(
                boundary,
                "genesis_payload_sha256",
            )
            and not verify_payload_hash(
                boundary,
                "boundary_payload_sha256",
            )
        ):
            raise J2A1ExecutionIntegrityError(
                "Dead-owner reclaim boundary is invalid"
            )
        old_owner_sha = previous["owner_record_sha256"]
        recovery = self._append(
            {
                "kind": "recovery",
                "old_owner_record_sha256": old_owner_sha,
                "process_death_verified": True,
                "zero_concurrent_writer": True,
                "boundary_sha256": scientific_hash(boundary),
                "new_pid": os.getpid() if pid is None else int(pid),
            }
        )
        owner_pid = os.getpid() if pid is None else int(pid)
        identity = (
            process_start_identity(owner_pid)
            if start_identity is None
            else str(start_identity)
        )
        owner = self._append(
            {
                "kind": "owner",
                "marker_sha256": marker_sha256,
                "lock_sha256": lock_sha256,
                "manifest_sha256": manifest_sha256,
                "runner_sha256": sha256_path(RUNNER_PATH),
                "command": command,
                "execution_mode": execution_mode,
                "hostname": socket.gethostname(),
                "pid": owner_pid,
                "process_start_identity": identity,
                "recovery_record_sha256": recovery["owner_record_sha256"],
                "recovered_owner_record_sha256": old_owner_sha,
            }
        )
        return {"recovery": recovery, "owner": owner, "passes": True}


ABANDONED_UNIT_CHARGE_SECONDS = {
    "teacher_root": TEACHER_P99_SECONDS * MAX_MOVES,
    "distillation_minibatch": 60.0,
    "student_arm": 5.0,
}


class AttemptRuntimeLedger:
    """One-scan append-only attempts with O(1) counters and bounded recovery."""

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
            name="attempt contract",
        )
        self.wall_clock = wall_clock
        self.records = self._load()
        self.open_starts: dict[str, dict[str, Any]] = {}
        self.started = 0
        self.finished = 0
        self.abandoned = 0
        self.active_seconds = 0.0
        self.completed_units: set[str] = set()
        self.attempts_by_unit: Counter[str] = Counter()
        for record in self.records:
            event = record["event"]
            if event == "started":
                self.started += 1
                self.open_starts[str(record["attempt_id"])] = record
                self.attempts_by_unit[str(record["unit_id"])] += 1
            elif event in {"finished", "abandoned"}:
                start = self.open_starts.get(str(record["attempt_id"]))
                if start is None:
                    raise J2A1ExecutionIntegrityError(
                        "Attempt closure has no open start"
                    )
                if record["start_sha256"] != start[
                    "attempt_record_sha256"
                ]:
                    raise J2A1ExecutionIntegrityError(
                        "Attempt closure changed its start"
                    )
                self.active_seconds += float(record["charged_seconds"])
                if event == "finished":
                    self.finished += 1
                    self.completed_units.add(str(record["unit_id"]))
                else:
                    self.abandoned += 1
                self.open_starts.pop(str(record["attempt_id"]))
        for attempt_id in sorted(list(self.open_starts)):
            self._close_abandoned(attempt_id)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        predecessor = None
        open_records: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise J2A1ExecutionIntegrityError(
                    f"Attempt line {line_number} is malformed"
                ) from error
            if not verify_payload_hash(record, "attempt_record_sha256"):
                raise J2A1ExecutionIntegrityError(
                    f"Attempt line {line_number} changed"
                )
            checks = {
                "sequence": int(record.get("sequence", -1))
                == len(records),
                "predecessor": record.get("predecessor_record_sha256")
                == predecessor,
                "contract": record.get("contract_sha256")
                == self.contract_sha256,
                "event": record.get("event")
                in {"started", "finished", "abandoned"},
            }
            if not all(checks.values()):
                raise J2A1ExecutionIntegrityError(
                    f"Attempt line {line_number} is inconsistent"
                )
            if record["event"] == "started":
                attempt_id = str(record.get("attempt_id"))
                if attempt_id in open_records:
                    raise J2A1ExecutionIntegrityError(
                        "Attempt ledger repeats an open attempt"
                    )
                open_records[attempt_id] = record
            else:
                attempt_id = str(record.get("attempt_id"))
                open_record = open_records.get(attempt_id)
                if (
                    open_record is None
                    or record.get("start_sha256")
                    != open_record["attempt_record_sha256"]
                    or record.get("unit_id") != open_record["unit_id"]
                ):
                    raise J2A1ExecutionIntegrityError(
                        "Attempt closure is malformed"
                    )
                open_records.pop(attempt_id)
            records.append(record)
            predecessor = record["attempt_record_sha256"]
        return records

    def _append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = payload_with_hash(
            {
                "version": f"{VERSION}_attempt_record_v1",
                "sequence": len(self.records),
                "predecessor_record_sha256": (
                    None
                    if not self.records
                    else self.records[-1]["attempt_record_sha256"]
                ),
                "contract_sha256": self.contract_sha256,
                **dict(payload),
            },
            "attempt_record_sha256",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        return record

    def _close_abandoned(self, attempt_id: str) -> dict[str, Any]:
        start = self.open_starts.get(attempt_id)
        if start is None:
            raise J2A1ExecutionIntegrityError(
                "No attempt is open for abandonment"
            )
        unit_type = str(start["unit_type"])
        if unit_type not in ABANDONED_UNIT_CHARGE_SECONDS:
            raise J2A1ExecutionIntegrityError(
                "Open attempt unit type is unknown"
            )
        record = self._append(
            {
                "event": "abandoned",
                "unit_id": start["unit_id"],
                "unit_type": unit_type,
                "attempt_id": start["attempt_id"],
                "start_sha256": start["attempt_record_sha256"],
                "wall_ended_at": float(self.wall_clock()),
                "charged_seconds": ABANDONED_UNIT_CHARGE_SECONDS[
                    unit_type
                ],
                "charge_basis":
                    "fixed preregistered abandoned-unit ceiling",
            }
        )
        self.abandoned += 1
        self.active_seconds += float(record["charged_seconds"])
        self.open_starts.pop(attempt_id)
        return record

    def begin(self, *, unit_id: str, unit_type: str) -> dict[str, Any]:
        if any(
            start["unit_id"] == unit_id
            for start in self.open_starts.values()
        ):
            raise J2A1ExecutionIntegrityError(
                "Attempt ledger already has this unit open"
            )
        if unit_id in self.completed_units:
            raise J2A1ExecutionIntegrityError(
                "Completed unit cannot be regenerated"
            )
        if unit_type not in ABANDONED_UNIT_CHARGE_SECONDS:
            raise J2A1ExecutionIntegrityError(
                "Attempt unit type is unknown"
            )
        ordinal = self.attempts_by_unit[unit_id]
        record = self._append(
            {
                "event": "started",
                "unit_id": unit_id,
                "unit_type": unit_type,
                "attempt_id": f"{unit_id}|attempt={ordinal}",
                "wall_started_at": float(self.wall_clock()),
            }
        )
        self.started += 1
        self.attempts_by_unit[unit_id] += 1
        self.open_starts[str(record["attempt_id"])] = record
        return record

    def finish(
        self,
        start: Mapping[str, Any],
        *,
        output_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        opened = self.open_starts.get(str(start.get("attempt_id")))
        if opened is None or start.get("attempt_record_sha256") != opened.get(
            "attempt_record_sha256"
        ):
            raise J2A1ExecutionIntegrityError(
                "Attempt finish changed its open start"
            )
        ended = float(self.wall_clock())
        started = float(start["wall_started_at"])
        if not math.isfinite(ended) or ended < started:
            raise J2A1ExecutionIntegrityError(
                "Attempt runtime clock moved backwards"
            )
        record = self._append(
            {
                "event": "finished",
                "unit_id": start["unit_id"],
                "unit_type": start["unit_type"],
                "attempt_id": start["attempt_id"],
                "start_sha256": start["attempt_record_sha256"],
                "wall_ended_at": ended,
                "charged_seconds": ended - started,
                "output_identity": j2.json_native(output_identity),
            }
        )
        self.finished += 1
        self.active_seconds += float(record["charged_seconds"])
        self.completed_units.add(str(start["unit_id"]))
        self.open_starts.pop(str(start["attempt_id"]))
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "active_seconds": self.active_seconds,
            "attempts_started": self.started,
            "attempts_finished": self.finished,
            "attempts_abandoned": self.abandoned,
            "open_attempt_ids": sorted(self.open_starts),
            "completed_unit_count": len(self.completed_units),
            "record_count": len(self.records),
            "head_sha256": (
                None
                if not self.records
                else self.records[-1]["attempt_record_sha256"]
            ),
            "passes": True,
        }


def _compact_updater_state_hash(state: Mapping[str, Any]) -> str:
    snapshot = state.get("snapshot_bytes")
    if not isinstance(snapshot, bytes):
        raise J2A1ExecutionIntegrityError(
            "Compact updater snapshot bytes are missing"
        )
    projection = {
        key: value
        for key, value in state.items()
        if key != "snapshot_bytes"
    }
    projection["snapshot_bytes_sha256"] = hashlib.sha256(
        snapshot
    ).hexdigest()
    projection["snapshot_bytes"] = len(snapshot)
    return canonical_json_hash(projection)


class CompactUpdaterStore:
    """Two-slot optimizer state with one immutable batch identity."""

    def __init__(
        self,
        *,
        directory: Path,
        contract_sha256: str,
    ) -> None:
        self.directory = directory
        self.contract_sha256 = _validate_hex(
            contract_sha256,
            name="updater contract",
        )
        self.journal_path = directory / "updater_journal.jsonl"
        self.head_path = directory / "updater_head.json"
        self.records = self._load_records()
        self.current = self._recover_current()

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        records = []
        predecessor = None
        for line_number, line in enumerate(
            self.journal_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                continue
            record = json.loads(line)
            if not verify_payload_hash(record, "updater_record_sha256"):
                raise J2A1ExecutionIntegrityError(
                    f"Updater journal line {line_number} changed"
                )
            if (
                int(record.get("sequence", -1)) != len(records)
                or record.get("predecessor_record_sha256") != predecessor
                or record.get("contract_sha256") != self.contract_sha256
            ):
                raise J2A1ExecutionIntegrityError(
                    f"Updater journal line {line_number} is inconsistent"
                )
            records.append(record)
            predecessor = record["updater_record_sha256"]
        return records

    def _recover_current(self) -> dict[str, Any] | None:
        if not self.records:
            if self.head_path.exists():
                raise J2A1ExecutionIntegrityError(
                    "Updater head exists without journal"
                )
            return None
        latest = self.records[-1]
        latest_slot = self.directory / str(latest["slot_name"])
        if (
            not latest_slot.is_file()
            or sha256_path(latest_slot) != latest["slot_file_sha256"]
        ):
            raise J2A1ExecutionIntegrityError(
                "Latest updater slot changed from journal"
            )
        latest_payload = _deserialize_torch_payload(
            latest_slot.read_bytes(),
            magic=CHECKPOINT_MAGIC,
        )
        if _compact_updater_state_hash(latest_payload) != latest[
            "state_sha256"
        ]:
            raise J2A1ExecutionIntegrityError(
                "Latest updater state changed from journal"
            )
        if self.head_path.exists():
            head = load_json(self.head_path)
            if not verify_payload_hash(head, "updater_head_payload_sha256"):
                raise J2A1ExecutionIntegrityError(
                    "Updater head changed"
                )
            head_sequence = int(head["sequence"])
            if head_sequence > int(latest["sequence"]):
                raise J2A1ExecutionIntegrityError(
                    "Updater head is ahead of journal"
                )
            if head_sequence < int(latest["sequence"]) - 1:
                raise J2A1ExecutionIntegrityError(
                    "Updater has more than one orphan"
                )
        head = _atomic_replace_json(
            self.head_path,
            {
                "version": f"{VERSION}_updater_head_v1",
                "sequence": latest["sequence"],
                "updater_record_sha256": latest[
                    "updater_record_sha256"
                ],
                "slot_name": latest["slot_name"],
                "slot_file_sha256": latest["slot_file_sha256"],
                "contract_sha256": self.contract_sha256,
            },
            field="updater_head_payload_sha256",
        )
        return {
            "record": latest,
            "head": head,
            "snapshot_bytes": latest_payload["snapshot_bytes"],
        }

    def append(
        self,
        *,
        unit_id: str,
        snapshot_bytes: bytes,
        cursor: int,
        batch_identity: str,
        crash_stage: str | None = None,
    ) -> dict[str, Any]:
        if crash_stage not in {None, "after_slot", "after_journal"}:
            raise ValueError("Unknown updater crash stage")
        sequence = len(self.records)
        slot_name = f"updater_slot_{sequence % 2}.bin"
        slot_path = self.directory / slot_name
        state = {
            "version": f"{VERSION}_compact_updater_state_v1",
            "unit_id": unit_id,
            "cursor": int(cursor),
            "batch_identity": _validate_hex(
                batch_identity,
                name="updater batch identity",
            ),
            "snapshot_bytes": bytes(snapshot_bytes),
        }
        serialized = _serialize_torch_payload(
            state,
            magic=CHECKPOINT_MAGIC,
        )
        _atomic_replace_bytes(slot_path, serialized)
        slot_sha = sha256_path(slot_path)
        if crash_stage == "after_slot":
            raise RuntimeError("fixture crash after updater slot")
        record = payload_with_hash(
            {
                "version": f"{VERSION}_updater_record_v1",
                "sequence": sequence,
                "predecessor_record_sha256": (
                    None
                    if not self.records
                    else self.records[-1]["updater_record_sha256"]
                ),
                "contract_sha256": self.contract_sha256,
                "unit_id": unit_id,
                "cursor": int(cursor),
                "batch_identity": batch_identity,
                "slot_name": slot_name,
                "slot_file_sha256": slot_sha,
                "state_sha256": _compact_updater_state_hash(state),
            },
            "updater_record_sha256",
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("ab") as handle:
            handle.write(canonical_json_bytes(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        if crash_stage == "after_journal":
            raise RuntimeError("fixture crash after updater journal")
        head = _atomic_replace_json(
            self.head_path,
            {
                "version": f"{VERSION}_updater_head_v1",
                "sequence": sequence,
                "updater_record_sha256": record[
                    "updater_record_sha256"
                ],
                "slot_name": slot_name,
                "slot_file_sha256": slot_sha,
                "contract_sha256": self.contract_sha256,
            },
            field="updater_head_payload_sha256",
        )
        self.current = {
            "record": record,
            "head": head,
            "snapshot_bytes": bytes(snapshot_bytes),
        }
        return copy.deepcopy(self.current)


class CompletionLedger:
    """Append-only complete-root or pair references."""

    def __init__(
        self,
        *,
        path: Path,
        contract_sha256: str,
        kind: str,
    ) -> None:
        if kind not in {"teacher_root", "fidelity_pair"}:
            raise ValueError("Completion ledger kind is invalid")
        self.path = path
        self.contract_sha256 = _validate_hex(
            contract_sha256,
            name="completion contract",
        )
        self.kind = kind
        self.records = self._load()
        self.by_root = {
            str(record["root_id"]): record for record in self.records
        }
        if len(self.by_root) != len(self.records):
            raise J2A1ExecutionIntegrityError(
                "Completion ledger repeats a root"
            )

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        predecessor = None
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise J2A1ExecutionIntegrityError(
                    f"Completion line {line_number} is malformed"
                ) from error
            if not verify_payload_hash(
                record,
                "completion_record_sha256",
            ):
                raise J2A1ExecutionIntegrityError(
                    f"Completion line {line_number} changed"
                )
            if (
                int(record.get("sequence", -1)) != len(records)
                or record.get("predecessor_record_sha256") != predecessor
                or record.get("contract_sha256") != self.contract_sha256
                or record.get("kind") != self.kind
            ):
                raise J2A1ExecutionIntegrityError(
                    f"Completion line {line_number} is inconsistent"
                )
            _validate_hex(record.get("root_id"), name="completion root")
            _validate_hex(
                record.get("file_sha256"),
                name="completion file",
            )
            records.append(record)
            predecessor = record["completion_record_sha256"]
        return records

    def append(
        self,
        *,
        root_id: str,
        ancestry_id: str,
        row_index: int,
        stage: str,
        relative_path: str,
        file_sha256: str,
        content_sha256: str,
        recovered_orphan: bool,
    ) -> dict[str, Any]:
        if root_id in self.by_root:
            raise J2A1ExecutionIntegrityError(
                "Completed root cannot be closed twice"
            )
        record = payload_with_hash(
            {
                "version": f"{VERSION}_completion_record_v1",
                "sequence": len(self.records),
                "predecessor_record_sha256": (
                    None
                    if not self.records
                    else self.records[-1]["completion_record_sha256"]
                ),
                "contract_sha256": self.contract_sha256,
                "kind": self.kind,
                "root_id": _validate_hex(root_id, name="completion root"),
                "ancestry_id": _validate_hex(
                    ancestry_id,
                    name="completion ancestry",
                ),
                "row_index": int(row_index),
                "stage": stage,
                "relative_path": relative_path,
                "file_sha256": _validate_hex(
                    file_sha256,
                    name="completion file",
                ),
                "content_sha256": _validate_hex(
                    content_sha256,
                    name="completion content",
                ),
                "recovered_orphan": bool(recovered_orphan),
            },
            "completion_record_sha256",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        self.by_root[root_id] = record
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "completed": len(self.records),
            "root_set_sha256": canonical_json_hash(sorted(self.by_root)),
            "head_sha256": (
                None
                if not self.records
                else self.records[-1]["completion_record_sha256"]
            ),
            "passes": True,
        }


def _teacher_binding_from_authority() -> dict[str, Any]:
    path = PARENT_READINESS_DIR / "J2_TEACHER_PROVENANCE.json"
    payload = load_bound_json(
        path,
        file_sha256=EXPECTED_PARENT_TEACHER_PROVENANCE[0],
        payload_field=EXPECTED_PARENT_TEACHER_PROVENANCE[1],
        payload_sha256=EXPECTED_PARENT_TEACHER_PROVENANCE[2],
    )
    binding = payload.get("incumbent_binding")
    if not isinstance(binding, Mapping):
        raise J2A1ExecutionIntegrityError(
            "Teacher incumbent binding is missing"
        )
    return copy.deepcopy(dict(binding))


def _collect_teacher_root_with_policy(
    row: Mapping[str, Any],
    policy: Any,
) -> dict[str, Any]:
    from threes_rl.obs import encode_observation
    from threes_rl.sim import ThreesSim, score_board

    stage = str(row["stage"])
    if stage not in ACTIVE_STAGES:
        raise J2A1ExecutionIntegrityError(
            "Teacher worker received an inactive stage"
        )
    streams = row["streams"]
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(streams["deck_stream_id"]),
        slot_stream_id=int(streams["slot_stream_id"]),
        starter_tile=None,
    )
    if sim.starter_tile is not None:
        raise J2A1ExecutionIntegrityError(
            "Teacher collection did not use a normal start"
        )
    state = sim.reset()
    policy_rng = np.random.default_rng(
        int(streams["teacher_policy_stream_id"])
    )
    start_score = int(score_board(state.board))
    transitions = []
    policy_latency = 0.0
    for transition_index in range(MAX_MOVES):
        legal = sim.legal_actions(state)
        if state.game_over or not legal:
            break
        legal_mask = sim.legal_mask(state)
        current_score = int(score_board(state.board))
        started = time.perf_counter()
        selected = int(policy(state, sim, policy_rng))
        policy_latency += time.perf_counter() - started
        if selected not in legal:
            raise J2A1ExecutionIntegrityError(
                "Exact incumbent selected an illegal action"
            )
        next_state, info = sim.step(state, selected)
        if not info.moved:
            raise J2A1ExecutionIntegrityError(
                "Exact incumbent action did not move"
            )
        transitions.append(
            {
                "transition_index": transition_index,
                "observation": encode_observation(
                    state,
                    sim,
                    encoder="full",
                ),
                "legal_mask": legal_mask,
                "teacher_action": selected,
                "current_score": current_score,
                "score_delta": int(info.score_delta),
                "board": state.board.copy(),
            }
        )
        state = next_state
    else:
        raise J2A1ExecutionIntegrityError(
            "Teacher root reached the 5000-move integrity boundary"
        )
    if not state.game_over and sim.legal_actions(state):
        raise J2A1ExecutionIntegrityError(
            "Teacher root did not close naturally"
        )
    final_score = int(score_board(state.board))
    record = {
        "version": f"{VERSION}_teacher_root_v1",
        "row": j2.json_native(row),
        "root_id": row["root_id"],
        "ancestry_id": row["ancestry_id"],
        "stage": stage,
        "shard": int(row["row_index"]) % SHARDS,
        "normal_start": True,
        "starter_tile": None,
        "natural_terminal": True,
        "start_score": start_score,
        "final_score": final_score,
        "final_max_tile": int(np.max(state.board, initial=0)),
        "policy_latency_seconds": policy_latency,
        "survival": float(len(transitions)),
        "transitions": transitions,
    }
    return seal_teacher_root_record(record, authoritative_row=row)


def _teacher_root_worker(
    worker_id: int,
    command_queue: Any,
    result_queue: Any,
    expected_binding: Mapping[str, Any],
) -> None:
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        from threes_rl import j1_execution_surface as j1_execution

        policy = j1_execution.load_bound_incumbent_policy(
            expected_binding
        )
        result_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "pid": os.getpid(),
            }
        )
        while True:
            command = command_queue.get()
            if command.get("kind") == "stop":
                return
            if command.get("kind") != "collect":
                raise J2A1ExecutionIntegrityError(
                    "Teacher worker command is invalid"
                )
            row = command["row"]
            if int(row["row_index"]) % SHARDS != worker_id:
                raise J2A1ExecutionIntegrityError(
                    "Teacher row crossed fixed worker ownership"
                )
            record = _collect_teacher_root_with_policy(row, policy)
            result_queue.put(
                {
                    "kind": "root",
                    "worker_id": worker_id,
                    "root_id": row["root_id"],
                    "record": record,
                }
            )
    except BaseException as error:
        result_queue.put(
            {
                "kind": "error",
                "worker_id": worker_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback_sha256": hashlib.sha256(
                    traceback.format_exc().encode("utf-8")
                ).hexdigest(),
            }
        )


def validate_teacher_worker_results(
    results: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = {str(row["root_id"]): row for row in rows}
    if len(expected) != len(rows) or len(results) != len(rows):
        raise J2A1ExecutionIntegrityError(
            "Teacher worker result count changed"
        )
    observed = {}
    for result in results:
        if result.get("kind") != "root":
            raise J2A1ExecutionIntegrityError(
                "Teacher worker result kind changed"
            )
        root_id = str(result.get("root_id"))
        if root_id in observed or root_id not in expected:
            raise J2A1ExecutionIntegrityError(
                "Teacher worker result is duplicate or unknown"
            )
        row = expected[root_id]
        worker_id = int(result.get("worker_id", -1))
        if worker_id != int(row["row_index"]) % SHARDS:
            raise J2A1ExecutionIntegrityError(
                "Teacher worker result crossed its shard"
            )
        record = result.get("record")
        if not isinstance(record, Mapping):
            raise J2A1ExecutionIntegrityError(
                "Teacher worker root record is missing"
            )
        validate_teacher_root_record(
            record,
            authoritative_row=row,
        )
        observed[root_id] = dict(record)
    return [observed[str(row["root_id"])] for row in rows]


class TeacherRootWorkerGroup:
    """Eight fixed-shard exact-incumbent collectors."""

    def __init__(
        self,
        *,
        binding: Mapping[str, Any],
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.context = mp.get_context("spawn")
        self.result_queue = self.context.Queue()
        self.command_queues = [
            self.context.Queue() for _ in range(SHARDS)
        ]
        self.processes = [
            self.context.Process(
                target=_teacher_root_worker,
                args=(
                    worker_id,
                    self.command_queues[worker_id],
                    self.result_queue,
                    dict(binding),
                ),
                name=f"j2a1-teacher-root-{worker_id}",
            )
            for worker_id in range(SHARDS)
        ]
        self.timeout_seconds = float(timeout_seconds)
        for process in self.processes:
            process.start()
        ready = self._receive(SHARDS, kind="ready")
        if {int(row["worker_id"]) for row in ready} != set(range(SHARDS)):
            self.close(terminate=True)
            raise J2A1ExecutionIntegrityError(
                "Teacher worker readiness identities changed"
            )

    def _receive(self, count: int, *, kind: str) -> list[dict[str, Any]]:
        records = []
        deadline = time.monotonic() + self.timeout_seconds
        while len(records) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise J2A1ExecutionOperationalHold(
                    "Teacher workers timed out"
                )
            try:
                record = self.result_queue.get(
                    timeout=min(5.0, remaining)
                )
            except queue.Empty:
                dead = [
                    process.pid
                    for process in self.processes
                    if not process.is_alive()
                ]
                if dead:
                    raise J2A1ExecutionIntegrityError(
                        f"Teacher worker exited early: {dead}"
                    )
                continue
            if record.get("kind") == "error":
                raise J2A1ExecutionIntegrityError(
                    "Teacher worker failed: "
                    f"{record.get('error_type')}: "
                    f"{record.get('error_message')}"
                )
            if record.get("kind") != kind:
                raise J2A1ExecutionIntegrityError(
                    "Teacher worker returned a late or wrong-kind record"
                )
            records.append(record)
        return records

    def collect(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        shards = [int(row["row_index"]) % SHARDS for row in rows]
        if len(set(shards)) != len(shards):
            raise J2A1ExecutionIntegrityError(
                "Teacher batch repeats a shard"
            )
        for row, shard in zip(rows, shards):
            self.command_queues[shard].put(
                {"kind": "collect", "row": j2.json_native(row)}
            )
        return validate_teacher_worker_results(
            self._receive(len(rows), kind="root"),
            rows,
        )

    def close(self, *, terminate: bool = False) -> None:
        for worker_id, process in enumerate(self.processes):
            if process.is_alive() and not terminate:
                self.command_queues[worker_id].put({"kind": "stop"})
            elif process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=15.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    def __enter__(self) -> "TeacherRootWorkerGroup":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close(terminate=False)


def _root_blob_relative_path(row: Mapping[str, Any]) -> str:
    return (
        f"teacher_roots/{row['stage']}/"
        f"{int(row['row_index']):05d}_{row['root_id']}.bin"
    )


def bounded_collect_teacher_roots(
    *,
    phase_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    contract_sha256: str,
    batch_collector: Callable[
        [Sequence[Mapping[str, Any]]],
        Sequence[Mapping[str, Any]],
    ],
    attempt_ledger: AttemptRuntimeLedger,
    boundary_callback: Callable[[str, Sequence[Path]], None] | None = None,
    interrupt_after_completed: int | None = None,
) -> dict[str, Any]:
    ordered = sorted(
        [j2.json_native(row) for row in rows],
        key=lambda row: (
            ACTIVE_STAGES.index(str(row["stage"])),
            int(row["row_index"]),
        ),
    )
    if any(not isinstance(row, dict) for row in ordered):
        raise J2A1ExecutionIntegrityError(
            "Teacher collection row is malformed"
        )
    expected_roots = [str(row["root_id"]) for row in ordered]
    if len(set(expected_roots)) != len(expected_roots):
        raise J2A1ExecutionIntegrityError(
            "Teacher collection authority repeats a root"
        )
    ledger = CompletionLedger(
        path=phase_dir / "teacher_root_completions.jsonl",
        contract_sha256=contract_sha256,
        kind="teacher_root",
    )
    rows_by_root = {str(row["root_id"]): row for row in ordered}
    for root_id, completion in list(ledger.by_root.items()):
        row = rows_by_root.get(root_id)
        if row is None:
            raise J2A1ExecutionIntegrityError(
                "Teacher completion is outside authority"
            )
        path = phase_dir / str(completion["relative_path"])
        record = load_teacher_root_blob(
            path,
            authoritative_row=row,
            expected_file_sha256=str(completion["file_sha256"]),
        )
        if record["root_content_sha256"] != completion["content_sha256"]:
            raise J2A1ExecutionIntegrityError(
                "Teacher completion content changed"
            )
    for stage in ACTIVE_STAGES:
        stage_rows = [row for row in ordered if row["stage"] == stage]
        for start in range(0, len(stage_rows), SHARDS):
            block = stage_rows[start : start + SHARDS]
            missing = []
            committed_paths: list[Path] = []
            for row in block:
                root_id = str(row["root_id"])
                if root_id in ledger.by_root:
                    continue
                relative = _root_blob_relative_path(row)
                path = phase_dir / relative
                if path.exists():
                    record = load_teacher_root_blob(
                        path,
                        authoritative_row=row,
                    )
                    ledger.append(
                        root_id=root_id,
                        ancestry_id=str(row["ancestry_id"]),
                        row_index=int(row["row_index"]),
                        stage=str(row["stage"]),
                        relative_path=relative,
                        file_sha256=sha256_path(path),
                        content_sha256=record["root_content_sha256"],
                        recovered_orphan=True,
                    )
                    committed_paths.extend(
                        (path, ledger.path)
                    )
                    continue
                missing.append(row)
            if missing:
                starts = {
                    str(row["root_id"]): attempt_ledger.begin(
                        unit_id=f"teacher_root|{row['stage']}|{row['root_id']}",
                        unit_type="teacher_root",
                    )
                    for row in missing
                }
                collected = list(batch_collector(missing))
                if collected and all(
                    isinstance(value, Mapping)
                    and value.get("kind") == "root"
                    for value in collected
                ):
                    results = validate_teacher_worker_results(
                        collected,
                        missing,
                    )
                else:
                    if len(collected) != len(missing):
                        raise J2A1ExecutionIntegrityError(
                            "Teacher collector changed result count"
                        )
                    results = []
                    for row, record in zip(missing, collected):
                        if not isinstance(record, Mapping):
                            raise J2A1ExecutionIntegrityError(
                                "Teacher collector result is malformed"
                            )
                        surface_record = dict(record)
                        validate_teacher_root_record(
                            surface_record,
                            authoritative_row=row,
                        )
                        results.append(surface_record)
                for row, record in zip(missing, results):
                    relative = _root_blob_relative_path(row)
                    identity = write_teacher_root_blob(
                        phase_dir / relative,
                        record,
                        authoritative_row=row,
                    )
                    attempt_ledger.finish(
                        starts[str(row["root_id"])],
                        output_identity=identity,
                    )
                    ledger.append(
                        root_id=str(row["root_id"]),
                        ancestry_id=str(row["ancestry_id"]),
                        row_index=int(row["row_index"]),
                        stage=str(row["stage"]),
                        relative_path=relative,
                        file_sha256=identity["file_sha256"],
                        content_sha256=identity["root_content_sha256"],
                        recovered_orphan=False,
                    )
                    committed_paths.extend(
                        (
                            phase_dir / relative,
                            ledger.path,
                            attempt_ledger.path,
                        )
                    )
                    if (
                        interrupt_after_completed is not None
                        and len(ledger.records) >= interrupt_after_completed
                    ):
                        raise J2A1PlannedInterruption(
                            "fixture interruption after teacher root commit"
                        )
            if boundary_callback is not None and committed_paths:
                boundary_callback(
                    "teacher_root_block",
                    tuple(dict.fromkeys(committed_paths)),
                )
    if set(ledger.by_root) != set(expected_roots):
        raise J2A1ExecutionIntegrityError(
            "Teacher collection did not close every root"
        )
    refs = [
        ledger.by_root[root_id]
        for root_id in expected_roots
    ]
    scientific_refs = [
        {
            "root_id": ref["root_id"],
            "ancestry_id": ref["ancestry_id"],
            "row_index": ref["row_index"],
            "stage": ref["stage"],
            "relative_path": ref["relative_path"],
            "file_sha256": ref["file_sha256"],
            "content_sha256": ref["content_sha256"],
        }
        for ref in refs
    ]
    seal = {
        "version": f"{VERSION}_teacher_collection_complete_v1",
        "expected_roots": len(expected_roots),
        "bc_roots": sum(row["stage"] == BC_STAGE for row in ordered),
        "validation_teacher_roots": sum(
            row["stage"] == VALIDATION_STAGE for row in ordered
        ),
        "canonical_root_order_sha256": canonical_json_hash(expected_roots),
        "completion_refs_sha256": canonical_json_hash(scientific_refs),
        "scientific_content_excludes_recovery_bookkeeping": True,
        "passes": True,
    }
    seal_path = phase_dir / "J2A1_TEACHER_COLLECTION_COMPLETE.json"
    if seal_path.exists():
        observed = load_json(seal_path)
        expected = payload_with_hash(
            seal,
            "collection_payload_sha256",
        )
        if observed != expected:
            raise J2A1ExecutionIntegrityError(
                "Teacher collection seal changed"
            )
    else:
        write_immutable_json(
            seal_path,
            seal,
            field="collection_payload_sha256",
        )
    return {
        "seal": load_json(seal_path),
        "refs": refs,
        "ledger": ledger.summary(),
        "passes": True,
    }


def load_teacher_roots_from_refs(
    *,
    phase_dir: Path,
    refs: Sequence[Mapping[str, Any]],
    rows_by_root: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    roots = []
    for ref in refs:
        root_id = str(ref["root_id"])
        row = rows_by_root.get(root_id)
        if row is None:
            raise J2A1ExecutionIntegrityError(
                "Teacher root ref is outside authority"
            )
        record = load_teacher_root_blob(
            phase_dir / str(ref["relative_path"]),
            authoritative_row=row,
            expected_file_sha256=str(ref["file_sha256"]),
        )
        roots.append(record)
    return roots


def _distillation_batch_payload(
    batch: j2.DistillationBatch,
) -> dict[str, Any]:
    j2.validate_distillation_batch(batch)
    return {
        "version": f"{VERSION}_distillation_batch_v1",
        "batch_identity": j2.distillation_batch_identity(batch),
        "observations": batch.observations.detach().cpu().clone(),
        "legal_masks": batch.legal_masks.detach().cpu().clone(),
        "teacher_actions": batch.teacher_actions.detach().cpu().clone(),
        "value_targets": batch.value_targets.detach().cpu().clone(),
        "row_weights": batch.row_weights.detach().cpu().clone(),
        "root_ids": tuple(batch.root_ids),
    }


def _batch_from_payload(
    payload: Mapping[str, Any],
) -> j2.DistillationBatch:
    if payload.get("version") != f"{VERSION}_distillation_batch_v1":
        raise J2A1ExecutionIntegrityError(
            "Distillation batch version changed"
        )
    batch = j2.DistillationBatch(
        observations=payload["observations"].detach().cpu().clone(),
        legal_masks=payload["legal_masks"].detach().cpu().clone(),
        teacher_actions=payload["teacher_actions"].detach().cpu().clone(),
        value_targets=payload["value_targets"].detach().cpu().clone(),
        row_weights=payload["row_weights"].detach().cpu().clone(),
        root_ids=tuple(str(value) for value in payload["root_ids"]),
    )
    j2.validate_distillation_batch(batch)
    if payload.get("batch_identity") != j2.distillation_batch_identity(batch):
        raise J2A1ExecutionIntegrityError(
            "Distillation batch identity changed"
        )
    return batch


def write_distillation_batch_blob(
    path: Path,
    batch: j2.DistillationBatch,
) -> dict[str, Any]:
    payload = _distillation_batch_payload(batch)
    serialized = _serialize_torch_payload(
        payload,
        magic=CHECKPOINT_MAGIC,
    )
    file_sha = _create_once_binary(
        path,
        serialized,
        validator=lambda value: _batch_from_payload(
            _deserialize_torch_payload(value, magic=CHECKPOINT_MAGIC)
        ),
    )
    observed = _batch_from_payload(
        _deserialize_torch_payload(
            path.read_bytes(),
            magic=CHECKPOINT_MAGIC,
        )
    )
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "file_sha256": file_sha,
        "batch_identity": j2.distillation_batch_identity(observed),
        "row_count": observed.row_count(),
        "root_count": len(set(observed.root_ids)),
    }


def load_distillation_batch_blob(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
    expected_batch_identity: str | None = None,
) -> j2.DistillationBatch:
    if not path.is_file():
        raise J2A1ExecutionIntegrityError(
            "Distillation batch blob is missing"
        )
    if (
        expected_file_sha256 is not None
        and sha256_path(path) != expected_file_sha256
    ):
        raise J2A1ExecutionIntegrityError(
            "Distillation batch file changed"
        )
    batch = _batch_from_payload(
        _deserialize_torch_payload(
            path.read_bytes(),
            magic=CHECKPOINT_MAGIC,
        )
    )
    if (
        expected_batch_identity is not None
        and j2.distillation_batch_identity(batch)
        != expected_batch_identity
    ):
        raise J2A1ExecutionIntegrityError(
            "Distillation batch payload changed"
        )
    return batch


def _checkpoint_payload(
    *,
    updater: j2.DistillationUpdater,
    batch_identity: str,
    collection_seal_sha256: str,
    family_inventory_sha256: str,
) -> dict[str, Any]:
    if updater.cursor != len(updater.plan):
        raise J2A1ExecutionIntegrityError(
            "Distillation checkpoint preceded epoch eight"
        )
    if any(
        not torch.isfinite(value).all()
        for value in updater.model.state_dict().values()
    ):
        raise J2A1ExecutionIntegrityError(
            "Distillation checkpoint model is nonfinite"
        )
    return {
        "version": f"{VERSION}_epoch8_checkpoint_v1",
        "epoch": j2.DISTILLATION_EPOCHS,
        "parameter_count": j2.parameter_count(updater.model),
        "model_schema_sha256": model_schema_sha256(),
        "batch_identity": batch_identity,
        "plan_sha256": canonical_json_hash(updater.plan),
        "closed_step_ids_sha256": canonical_json_hash(
            updater.closed_step_ids
        ),
        "optimizer_step_count": updater.cursor,
        "collection_seal_sha256": _validate_hex(
            collection_seal_sha256,
            name="checkpoint collection seal",
        ),
        "family_inventory_sha256": _validate_hex(
            family_inventory_sha256,
            name="checkpoint family inventory",
        ),
        "model_state": copy.deepcopy(updater.model.state_dict()),
        "optimizer_state": copy.deepcopy(updater.optimizer.state_dict()),
        "authoritative": False,
        "quarantined_until_mechanism_and_fidelity_pass": True,
    }


def write_epoch8_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(payload)
    body["checkpoint_payload_sha256"] = scientific_hash(body)
    serialized = _serialize_torch_payload(body, magic=CHECKPOINT_MAGIC)
    file_sha = _create_once_binary(
        path,
        serialized,
        validator=lambda value: validate_epoch8_checkpoint(
            _deserialize_torch_payload(value, magic=CHECKPOINT_MAGIC)
        ),
    )
    observed = _deserialize_torch_payload(
        path.read_bytes(),
        magic=CHECKPOINT_MAGIC,
    )
    model, optimizer = validate_epoch8_checkpoint(observed)
    del model, optimizer
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "file_sha256": file_sha,
        "checkpoint_payload_sha256": body["checkpoint_payload_sha256"],
        "parameter_count": body["parameter_count"],
        "optimizer_step_count": body["optimizer_step_count"],
        "authoritative": False,
        "quarantined": True,
    }


def validate_epoch8_checkpoint(
    payload: Mapping[str, Any],
) -> tuple[j2.J2ActorCritic, torch.optim.Optimizer]:
    body = dict(payload)
    observed_hash = body.pop("checkpoint_payload_sha256", None)
    if observed_hash != scientific_hash(body):
        raise J2A1ExecutionIntegrityError(
            "Epoch-8 checkpoint payload changed"
        )
    if (
        body.get("version") != f"{VERSION}_epoch8_checkpoint_v1"
        or int(body.get("epoch", -1)) != j2.DISTILLATION_EPOCHS
        or int(body.get("parameter_count", -1))
        != j2.EXPECTED_PARAMETER_COUNT
        or body.get("model_schema_sha256") != model_schema_sha256()
        or body.get("authoritative") is not False
        or body.get("quarantined_until_mechanism_and_fidelity_pass")
        is not True
    ):
        raise J2A1ExecutionIntegrityError(
            "Epoch-8 checkpoint contract changed"
        )
    model, optimizer = j2.initialize_model_optimizer()
    try:
        model.load_state_dict(body["model_state"], strict=True)
        optimizer.load_state_dict(body["optimizer_state"])
    except (KeyError, RuntimeError, ValueError) as error:
        raise J2A1ExecutionIntegrityError(
            "Epoch-8 checkpoint state is malformed"
        ) from error
    if any(
        not torch.isfinite(value).all()
        for value in model.state_dict().values()
    ):
        raise J2A1ExecutionIntegrityError(
            "Epoch-8 checkpoint model is nonfinite"
        )
    model_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if model_ids != optimizer_ids:
        raise J2A1ExecutionIntegrityError(
            "Epoch-8 optimizer is not bound to its model"
        )
    return model, optimizer


def bounded_distillation(
    *,
    phase_dir: Path,
    bc_roots: Sequence[Mapping[str, Any]],
    contract_sha256: str,
    collection_seal_sha256: str,
    family_inventory_sha256: str,
    attempt_ledger: AttemptRuntimeLedger,
    expected_root_count: int = BC_ROOTS,
    minibatch_size: int = j2.MINIBATCH_SIZE,
    epochs: int = j2.DISTILLATION_EPOCHS,
    boundary_callback: Callable[[str, Sequence[Path]], None] | None = None,
    interrupt_after_committed_steps: int | None = None,
    interrupt_before_commit_step: int | None = None,
) -> dict[str, Any]:
    batch = build_distillation_batch(
        bc_roots,
        expected_root_count=expected_root_count,
    )
    batch_dir = phase_dir / "ephemeral_bc_batch"
    batch_path = batch_dir / "current_batch.bin"
    batch_identity = write_distillation_batch_blob(batch_path, batch)
    batch = load_distillation_batch_blob(
        batch_path,
        expected_file_sha256=batch_identity["file_sha256"],
        expected_batch_identity=batch_identity["batch_identity"],
    )
    store = CompactUpdaterStore(
        directory=phase_dir / "optimizer_resume",
        contract_sha256=contract_sha256,
    )
    if store.current is None:
        model, optimizer = j2.initialize_model_optimizer()
        updater = j2.DistillationUpdater(
            model,
            optimizer,
            batch,
            minibatch_size=minibatch_size,
            epochs=epochs,
        )
        store.append(
            unit_id="distillation_pre_update",
            snapshot_bytes=updater.snapshot_bytes(),
            cursor=0,
            batch_identity=batch_identity["batch_identity"],
        )
    else:
        updater = j2.DistillationUpdater.from_snapshot_bytes(
            store.current["snapshot_bytes"],
            batch,
            minibatch_size=minibatch_size,
            epochs=epochs,
        )
        if store.current["record"]["batch_identity"] != batch_identity[
            "batch_identity"
        ]:
            raise J2A1ExecutionIntegrityError(
                "Optimizer resume changed the immutable BC batch"
            )
    while updater.cursor < len(updater.plan):
        step = updater.plan[updater.cursor]
        unit_id = f"distillation_step|{step['step_id']}"
        started = attempt_ledger.begin(
            unit_id=unit_id,
            unit_type="distillation_minibatch",
        )
        report = updater.step()
        if (
            interrupt_before_commit_step is not None
            and updater.cursor == interrupt_before_commit_step
        ):
            raise J2A1PlannedInterruption(
                "fixture interruption before optimizer durable commit"
            )
        boundary = store.append(
            unit_id=unit_id,
            snapshot_bytes=updater.snapshot_bytes(),
            cursor=updater.cursor,
            batch_identity=batch_identity["batch_identity"],
        )
        attempt_ledger.finish(
            started,
            output_identity={
                "updater_record_sha256": boundary["record"][
                    "updater_record_sha256"
                ],
                "step_id": report["step_id"],
                "cursor": report["cursor"],
            },
        )
        if boundary_callback is not None:
            boundary_callback(
                "distillation_minibatch",
                tuple(
                    dict.fromkeys(
                        (
                            attempt_ledger.path,
                            store.journal_path,
                            store.head_path,
                            store.directory
                            / str(boundary["record"]["slot_name"]),
                            batch_path,
                        )
                    )
                ),
            )
        if (
            interrupt_after_committed_steps is not None
            and updater.cursor >= interrupt_after_committed_steps
        ):
            raise J2A1PlannedInterruption(
                "fixture interruption after optimizer durable commit"
            )
    checkpoint_path = phase_dir / "J2A1_EPOCH8_CHECKPOINT.bin"
    checkpoint = write_epoch8_checkpoint(
        checkpoint_path,
        _checkpoint_payload(
            updater=updater,
            batch_identity=batch_identity["batch_identity"],
            collection_seal_sha256=collection_seal_sha256,
            family_inventory_sha256=family_inventory_sha256,
        ),
    )
    return {
        "batch": batch_identity,
        "optimizer_steps": updater.cursor,
        "plan_sha256": canonical_json_hash(updater.plan),
        "closed_step_ids_sha256": canonical_json_hash(
            updater.closed_step_ids
        ),
        "checkpoint": checkpoint,
        "attempt_summary": attempt_ledger.summary(),
        "passes": True,
    }


def _student_arm(
    *,
    row: Mapping[str, Any],
    model: j2.J2ActorCritic,
) -> dict[str, Any]:
    from threes_rl.obs import encode_observation
    from threes_rl.sim import ThreesSim, score_board

    streams = row["streams"]
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(streams["deck_stream_id"]),
        slot_stream_id=int(streams["slot_stream_id"]),
        starter_tile=None,
    )
    state = sim.reset()
    start_score = int(score_board(state.board))
    latency = 0.0
    moves = 0
    model.eval()
    while not state.game_over:
        legal = sim.legal_actions(state)
        if not legal:
            break
        if moves >= MAX_MOVES:
            raise J2A1ExecutionIntegrityError(
                "Student arm reached the 5000-move integrity boundary"
            )
        observation = torch.from_numpy(
            encode_observation(state, sim, encoder="full")
        ).unsqueeze(0)
        legal_mask = torch.from_numpy(sim.legal_mask(state)).unsqueeze(0)
        started = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(observation)
            action = int(
                torch.argmax(
                    j2.masked_logits(logits, legal_mask),
                    dim=1,
                )[0]
            )
        latency += time.perf_counter() - started
        if action not in legal:
            raise J2A1ExecutionIntegrityError(
                "Student selected an illegal action"
            )
        state, info = sim.step(state, action)
        if not info.moved:
            raise J2A1ExecutionIntegrityError(
                "Student legal action did not move"
            )
        moves += 1
    return {
        "start_score": start_score,
        "final_score": int(score_board(state.board)),
        "max_tile": int(np.max(state.board, initial=0)),
        "moves": moves,
        "latency_seconds": latency,
        "survival": float(moves),
        "illegal_actions": 0,
        "policy_stream_id": int(streams["student_policy_stream_id"]),
    }


def run_student_arms_synchronously(
    *,
    rows: Sequence[Mapping[str, Any]],
    model: j2.J2ActorCritic,
) -> list[dict[str, Any]]:
    if not rows or len(rows) > 16:
        raise J2A1ExecutionIntegrityError(
            "Student synchronous batch must contain 1..16 roots"
        )
    from threes_rl.obs import encode_observation
    from threes_rl.sim import ThreesSim, score_board

    items = []
    for row in rows:
        streams = row["streams"]
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=int(streams["deck_stream_id"]),
            slot_stream_id=int(streams["slot_stream_id"]),
            starter_tile=None,
        )
        state = sim.reset()
        items.append(
            {
                "row": row,
                "sim": sim,
                "state": state,
                "start_score": int(score_board(state.board)),
                "moves": 0,
                "latency_seconds": 0.0,
            }
        )
    model.eval()
    while True:
        live: list[tuple[int, list[int]]] = []
        for index, item in enumerate(items):
            state = item["state"]
            sim = item["sim"]
            legal = sim.legal_actions(state)
            if state.game_over or not legal:
                continue
            if int(item["moves"]) >= MAX_MOVES:
                raise J2A1ExecutionIntegrityError(
                    "Student arm reached the 5000-move integrity boundary"
                )
            live.append((index, legal))
        if not live:
            break
        observations = torch.from_numpy(
            np.stack(
                [
                    encode_observation(
                        items[index]["state"],
                        items[index]["sim"],
                        encoder="full",
                    )
                    for index, _legal in live
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        )
        legal_masks = torch.from_numpy(
            np.stack(
                [
                    items[index]["sim"].legal_mask(
                        items[index]["state"]
                    )
                    for index, _legal in live
                ],
                axis=0,
            )
        )
        started = time.perf_counter()
        with torch.no_grad():
            logits, _ = model(observations)
            actions = torch.argmax(
                j2.masked_logits(logits, legal_masks),
                dim=1,
            ).detach().cpu().tolist()
        elapsed_per_root = (
            time.perf_counter() - started
        ) / len(live)
        for (index, legal), action_value in zip(live, actions):
            action = int(action_value)
            if action not in legal:
                raise J2A1ExecutionIntegrityError(
                    "Student selected an illegal action"
                )
            item = items[index]
            next_state, info = item["sim"].step(item["state"], action)
            if not info.moved:
                raise J2A1ExecutionIntegrityError(
                    "Student legal action did not move"
                )
            item["state"] = next_state
            item["moves"] = int(item["moves"]) + 1
            item["latency_seconds"] = (
                float(item["latency_seconds"]) + elapsed_per_root
            )
    return [
        {
            "start_score": int(item["start_score"]),
            "final_score": int(
                score_board(item["state"].board)
            ),
            "max_tile": int(
                np.max(item["state"].board, initial=0)
            ),
            "moves": int(item["moves"]),
            "latency_seconds": float(item["latency_seconds"]),
            "survival": float(item["moves"]),
            "illegal_actions": 0,
            "policy_stream_id": int(
                item["row"]["streams"]["student_policy_stream_id"]
            ),
        }
        for item in items
    ]


def _pair_content_hash(record: Mapping[str, Any]) -> str:
    body = dict(record)
    body.pop("pair_content_sha256", None)
    return scientific_hash(body)


def seal_pair_record(
    *,
    row: Mapping[str, Any],
    teacher_root: Mapping[str, Any],
    student_arm: Mapping[str, Any],
    student_checkpoint_sha256: str,
) -> dict[str, Any]:
    teacher_arm = {
        "start_score": int(teacher_root["start_score"]),
        "final_score": int(teacher_root["final_score"]),
        "max_tile": int(teacher_root["final_max_tile"]),
        "moves": len(teacher_root["transitions"]),
        "latency_seconds": float(
            teacher_root["policy_latency_seconds"]
        ),
        "survival": float(teacher_root["survival"]),
        "illegal_actions": 0,
        "arm_file_sha256": _validate_hex(
            teacher_root["file_sha256"],
            name="teacher arm file",
        ),
    }
    student_body = {
        key: student_arm[key]
        for key in (
            "start_score",
            "final_score",
            "max_tile",
            "moves",
            "latency_seconds",
            "survival",
            "illegal_actions",
        )
    }
    student_body["arm_file_sha256"] = scientific_hash(
        {
            "root_id": row["root_id"],
            "checkpoint": student_checkpoint_sha256,
            "student_arm": student_body,
        }
    )
    record = {
        "version": f"{VERSION}_fidelity_pair_v1",
        "root_id": row["root_id"],
        "ancestry_id": row["ancestry_id"],
        "row": j2.json_native(row),
        "student_checkpoint_sha256": _validate_hex(
            student_checkpoint_sha256,
            name="student checkpoint",
        ),
        "teacher": teacher_arm,
        "student": student_body,
        "pair_complete": True,
    }
    record["pair_content_sha256"] = _pair_content_hash(record)
    validate_complete_pair_records([record], [row], expected_pairs=1)
    return record


def _pair_relative_path(row: Mapping[str, Any]) -> str:
    return (
        "fidelity_pairs/"
        f"{int(row['row_index']):05d}_{row['root_id']}.json"
    )


def bounded_collect_fidelity_pairs(
    *,
    phase_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    teacher_roots: Mapping[str, Mapping[str, Any]],
    checkpoint_path: Path,
    contract_sha256: str,
    attempt_ledger: AttemptRuntimeLedger,
    arm_runner: Callable[
        [Sequence[Mapping[str, Any]], j2.J2ActorCritic],
        Sequence[Mapping[str, Any]],
    ] = lambda rows, model: run_student_arms_synchronously(
        rows=rows,
        model=model,
    ),
    boundary_callback: Callable[[str, Sequence[Path]], None] | None = None,
    interrupt_after_completed: int | None = None,
) -> dict[str, Any]:
    checkpoint_payload = _deserialize_torch_payload(
        checkpoint_path.read_bytes(),
        magic=CHECKPOINT_MAGIC,
    )
    model, _optimizer = validate_epoch8_checkpoint(checkpoint_payload)
    checkpoint_sha = sha256_path(checkpoint_path)
    ordered = sorted(rows, key=lambda row: int(row["row_index"]))
    ledger = CompletionLedger(
        path=phase_dir / "fidelity_pair_completions.jsonl",
        contract_sha256=contract_sha256,
        kind="fidelity_pair",
    )
    for root_id, completion in ledger.by_root.items():
        row = next(
            (value for value in ordered if value["root_id"] == root_id),
            None,
        )
        if row is None:
            raise J2A1ExecutionIntegrityError(
                "Fidelity completion is outside authority"
            )
        path = phase_dir / str(completion["relative_path"])
        if not path.is_file() or sha256_path(path) != completion[
            "file_sha256"
        ]:
            raise J2A1ExecutionIntegrityError(
                "Fidelity pair file changed"
            )
        record = load_json(path)
        if not verify_payload_hash(record, "pair_content_sha256"):
            raise J2A1ExecutionIntegrityError(
                "Fidelity pair payload changed"
            )
        validate_complete_pair_records([record], [row], expected_pairs=1)
    for start in range(0, len(ordered), 16):
        block = ordered[start : start + 16]
        missing = []
        committed_paths: list[Path] = []
        for row in block:
            root_id = str(row["root_id"])
            if root_id in ledger.by_root:
                continue
            relative = _pair_relative_path(row)
            path = phase_dir / relative
            if path.exists():
                record = load_json(path)
                if (
                    not verify_payload_hash(record, "pair_content_sha256")
                    or record.get("pair_content_sha256")
                    != _pair_content_hash(record)
                ):
                    raise J2A1ExecutionIntegrityError(
                        "Orphan fidelity pair changed"
                    )
                validate_complete_pair_records(
                    [record],
                    [row],
                    expected_pairs=1,
                )
                ledger.append(
                    root_id=root_id,
                    ancestry_id=str(row["ancestry_id"]),
                    row_index=int(row["row_index"]),
                    stage=VALIDATION_STAGE,
                    relative_path=relative,
                    file_sha256=sha256_path(path),
                    content_sha256=record["pair_content_sha256"],
                    recovered_orphan=True,
                )
                committed_paths.extend((path, ledger.path))
                continue
            missing.append(row)
        if not missing:
            if boundary_callback is not None and committed_paths:
                boundary_callback(
                    "fidelity_pair_block",
                    tuple(dict.fromkeys(committed_paths)),
                )
            continue
        starts = {
            str(row["root_id"]): attempt_ledger.begin(
                unit_id=f"student_arm|{row['root_id']}",
                unit_type="student_arm",
            )
            for row in missing
        }
        arms = list(arm_runner(missing, model))
        if len(arms) != len(missing):
            raise J2A1ExecutionIntegrityError(
                "Student arm runner changed result count"
            )
        for row, arm in zip(missing, arms):
            teacher = teacher_roots.get(str(row["root_id"]))
            if teacher is None:
                raise J2A1ExecutionIntegrityError(
                    "Fidelity teacher root is missing"
                )
            record = seal_pair_record(
                row=row,
                teacher_root=teacher,
                student_arm=arm,
                student_checkpoint_sha256=checkpoint_sha,
            )
            relative = _pair_relative_path(row)
            write_immutable_json(
                phase_dir / relative,
                record,
                field="pair_content_sha256",
            )
            identity = {
                "path": relative,
                "file_sha256": sha256_path(phase_dir / relative),
                "pair_content_sha256": record["pair_content_sha256"],
            }
            attempt_ledger.finish(
                starts[str(row["root_id"])],
                output_identity=identity,
            )
            ledger.append(
                root_id=str(row["root_id"]),
                ancestry_id=str(row["ancestry_id"]),
                row_index=int(row["row_index"]),
                stage=VALIDATION_STAGE,
                relative_path=relative,
                file_sha256=identity["file_sha256"],
                content_sha256=identity["pair_content_sha256"],
                recovered_orphan=False,
            )
            committed_paths.extend(
                (
                    phase_dir / relative,
                    ledger.path,
                    attempt_ledger.path,
                )
            )
            if (
                interrupt_after_completed is not None
                and len(ledger.records) >= interrupt_after_completed
            ):
                raise J2A1PlannedInterruption(
                    "fixture interruption after fidelity pair commit"
                )
        if boundary_callback is not None and committed_paths:
            boundary_callback(
                "fidelity_pair_block",
                tuple(dict.fromkeys(committed_paths)),
            )
    expected_ids = [str(row["root_id"]) for row in ordered]
    if set(ledger.by_root) != set(expected_ids):
        raise J2A1ExecutionIntegrityError(
            "Fidelity panel is incomplete"
        )
    refs = [ledger.by_root[root_id] for root_id in expected_ids]
    seal = {
        "version": f"{VERSION}_fidelity_panel_complete_v1",
        "pair_count": len(refs),
        "canonical_pair_order_sha256": canonical_json_hash(expected_ids),
        "pair_refs_sha256": canonical_json_hash(refs),
        "completion_ledger_head_sha256": ledger.summary()["head_sha256"],
        "checkpoint_file_sha256": checkpoint_sha,
        "partial_outcome_reads": 0,
        "passes": True,
    }
    seal_path = phase_dir / "J2A1_FIDELITY_PANEL_COMPLETE.json"
    if seal_path.exists():
        observed = load_json(seal_path)
        if observed != payload_with_hash(
            seal,
            "panel_payload_sha256",
        ):
            raise J2A1ExecutionIntegrityError(
                "Fidelity panel seal changed"
            )
    else:
        write_immutable_json(
            seal_path,
            seal,
            field="panel_payload_sha256",
        )
    return {
        "seal": load_json(seal_path),
        "refs": refs,
        "ledger": ledger.summary(),
        "passes": True,
    }


def load_fidelity_pairs_after_seal(
    *,
    phase_dir: Path,
    panel: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seal = panel.get("seal")
    refs = panel.get("refs")
    if (
        not isinstance(seal, Mapping)
        or not verify_payload_hash(seal, "panel_payload_sha256")
        or seal.get("passes") is not True
        or int(seal.get("partial_outcome_reads", -1)) != 0
        or not isinstance(refs, Sequence)
        or len(refs) != len(rows)
    ):
        raise J2A1ExecutionIntegrityError(
            "Fidelity outcomes cannot open before complete panel seal"
        )
    records = []
    for ref in refs:
        path = phase_dir / str(ref["relative_path"])
        if not path.is_file() or sha256_path(path) != ref["file_sha256"]:
            raise J2A1ExecutionIntegrityError(
                "Fidelity pair changed after panel seal"
            )
        record = load_json(path)
        if (
            not verify_payload_hash(record, "pair_content_sha256")
            or record["pair_content_sha256"] != ref["content_sha256"]
        ):
            raise J2A1ExecutionIntegrityError(
                "Fidelity pair content changed after panel seal"
            )
        records.append(record)
    validate_complete_pair_records(
        records,
        rows,
        expected_pairs=len(rows),
    )
    return records


def seal_retirement(
    *,
    phase_dir: Path,
    name: str,
    paths: Sequence[Path],
    predecessor_sha256: str,
    crash_after_manifest: bool = False,
    crash_after_deletions: int | None = None,
) -> dict[str, Any]:
    manifest_path = phase_dir / "retirements" / f"{name}.json"
    expected_rows = []
    for path in sorted(paths, key=lambda value: str(value)):
        try:
            relative = str(path.resolve().relative_to(phase_dir.resolve()))
        except ValueError as error:
            raise J2A1ExecutionIntegrityError(
                "Retirement path escapes phase directory"
            ) from error
        if path.exists():
            expected_rows.append(
                {
                    "relative_path": relative,
                    "file_sha256": sha256_path(path),
                    "bytes": int(path.stat().st_size),
                }
            )
        elif manifest_path.exists():
            continue
        else:
            raise J2A1ExecutionIntegrityError(
                "Retirement source is missing before intent"
            )
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if not verify_payload_hash(manifest, "retirement_payload_sha256"):
            raise J2A1ExecutionIntegrityError(
                "Retirement manifest changed"
            )
    else:
        manifest = write_immutable_json(
            manifest_path,
            {
                "version": f"{VERSION}_retirement_v1",
                "name": name,
                "predecessor_sha256": _validate_hex(
                    predecessor_sha256,
                    name="retirement predecessor",
                ),
                "files": expected_rows,
                "file_count": len(expected_rows),
                "bytes": sum(row["bytes"] for row in expected_rows),
                "inventory_sha256": canonical_json_hash(expected_rows),
            },
            field="retirement_payload_sha256",
        )
    if crash_after_manifest:
        raise J2A1PlannedInterruption(
            "fixture interruption after retirement manifest"
        )
    deleted = 0
    for row in manifest["files"]:
        path = phase_dir / str(row["relative_path"])
        if path.exists():
            if (
                sha256_path(path) != row["file_sha256"]
                or int(path.stat().st_size) != int(row["bytes"])
            ):
                raise J2A1ExecutionIntegrityError(
                    "Retirement source changed after intent"
                )
            path.unlink()
            deleted += 1
            if (
                crash_after_deletions is not None
                and deleted >= crash_after_deletions
            ):
                raise J2A1PlannedInterruption(
                    "fixture interruption during retirement"
                )
    if any(
        (phase_dir / str(row["relative_path"])).exists()
        for row in manifest["files"]
    ):
        raise J2A1ExecutionIntegrityError(
            "Retirement did not remove every intended file"
        )
    return {
        "manifest": manifest,
        "manifest_file_sha256": sha256_path(manifest_path),
        "all_sources_absent": True,
        "passes": True,
    }


class OutputAccountant:
    """One initial directory scan with O(1) targeted size updates."""

    def __init__(self, phase_dir: Path) -> None:
        self.phase_dir = phase_dir.resolve()
        self.full_scan_count = 1
        self.targeted_stat_count = 0
        self._sizes = {
            str(path.resolve()): int(path.stat().st_size)
            for path in phase_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        } if phase_dir.exists() else {}

    def _local(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.phase_dir)
        except ValueError as error:
            raise J2A1ExecutionIntegrityError(
                "Output accounting path escapes phase directory"
            ) from error
        return resolved

    def record_path(self, path: Path) -> None:
        resolved = self._local(path)
        if resolved.exists():
            self.targeted_stat_count += 1
            self._sizes[str(resolved)] = int(resolved.stat().st_size)
        else:
            self._sizes.pop(str(resolved), None)

    def retire_path(self, path: Path) -> None:
        self._sizes.pop(str(self._local(path)), None)

    def reconcile(self) -> dict[str, Any]:
        self.full_scan_count += 1
        self._sizes = {
            str(path.resolve()): int(path.stat().st_size)
            for path in self.phase_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        } if self.phase_dir.exists() else {}
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "output_bytes": sum(self._sizes.values()),
            "output_file_count": len(self._sizes),
            "full_scan_count": self.full_scan_count,
            "targeted_stat_count": self.targeted_stat_count,
            "passes": True,
        }


def configure_deterministic_runtime() -> dict[str, Any]:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise J2A1ExecutionOperationalHold(
                "Torch inter-op threads cannot be set to one"
            ) from error
    torch.use_deterministic_algorithms(True)
    checks = {
        "torch_intra_one": torch.get_num_threads() == 1,
        "torch_inter_one": torch.get_num_interop_threads() == 1,
        "torch_deterministic": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    if not all(checks.values()):
        raise J2A1ExecutionOperationalHold(
            "Deterministic Torch runtime guard failed"
        )
    return {
        "torch_version": torch.__version__,
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "checks": checks,
        "passes": True,
    }


def execution_operational_guard(
    *,
    phase_dir: Path,
    accountant: OutputAccountant,
    active_seconds: float,
    include_services: bool,
    require_target_disk: bool,
) -> dict[str, Any]:
    if include_services:
        from threes_rl import j1_joint_policy_value as j1

        free_gib = j1.free_disk_gib()
        process = j1.heavy_process_audit()
        services = j1.service_audit()
        nice = int(os.getpriority(os.PRIO_PROCESS, 0))
    else:
        free_gib = 1_000.0
        process = {"passes": True, "fixture_only": True}
        services = {
            "passes": True,
            "fixture_only": True,
            "dashboard": {
                "top_three": [263670, 261369, 258561],
            },
            "recorder": {"active_session_content_read": False},
        }
        nice = 10
    runtime = configure_deterministic_runtime()
    output = accountant.snapshot()
    checks = {
        "nice_at_least_10": nice >= 10,
        "one_heavy_job": process.get("passes") is True,
        "services_healthy": services.get("passes") is True,
        "human_sessions_opaque": services.get("recorder", {}).get(
            "active_session_content_read",
            False,
        )
        is False,
        "top_three_unchanged": services.get("dashboard", {}).get(
            "top_three"
        )
        == [263670, 261369, 258561],
        "free_disk_hard_floor": free_gib > HARD_DISK_FLOOR_GIB,
        "free_disk_target": (
            free_gib > TARGET_DISK_GIB if require_target_disk else True
        ),
        "active_runtime_within_cap": float(active_seconds)
        <= RUNTIME_CAP_HOURS * 3600.0,
        "output_within_cap": int(output["output_bytes"])
        <= STORAGE_CAP_BYTES,
        "torch_runtime_exact": runtime["passes"],
    }
    if not all(checks.values()):
        raise J2A1ExecutionOperationalHold(
            "J2A1 execution operational guard failed"
        )
    return {
        "version": f"{VERSION}_execution_operational_guard_v1",
        "free_disk_gib": free_gib,
        "nice": nice,
        "active_seconds": float(active_seconds),
        "runtime_cap_hours": RUNTIME_CAP_HOURS,
        "output": output,
        "process": process,
        "services": services,
        "torch": runtime,
        "checks": checks,
        "passes": True,
    }


def execution_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "lock": out_dir / PHASE_LOCK_NAME,
        "marker": out_dir / OPEN_MARKER_NAME,
        "manifest": out_dir / MATERIALIZED_MANIFEST_NAME,
        "reservation": out_dir / RESERVATION_NAME,
        "consumption": out_dir / CONSUMPTION_NAME,
        "genesis": out_dir / GENESIS_NAME,
        "owner": out_dir / "ownership_ledger.jsonl",
        "attempts": out_dir / "attempt_runtime_ledger.jsonl",
        "result": out_dir / TERMINAL_NAME,
        "retention": out_dir / EXECUTION_RETENTION_NAME,
    }


def _verify_readiness_package(readiness_dir: Path) -> dict[str, Any]:
    paths = readiness_namespace_paths(readiness_dir)
    required = set(paths)
    if not all(paths[key].is_file() for key in required):
        raise J2A1ExecutionIntegrityError(
            "Execution-surface readiness package is incomplete"
        )
    fields = {
        "test_evidence": "test_evidence_payload_sha256",
        "input_bindings": "input_bindings_payload_sha256",
        "authority": "authority_audit_payload_sha256",
        "schema": "execution_schema_payload_sha256",
        "projection": "projection_payload_sha256",
        "lock": "readiness_lock_payload_sha256",
        "result": "readiness_result_payload_sha256",
        "retention": "retention_payload_sha256",
    }
    payloads = {}
    identities = {}
    for key, field in fields.items():
        payload = load_json(paths[key])
        if not verify_payload_hash(payload, field):
            raise J2A1ExecutionIntegrityError(
                f"Readiness artifact changed: {paths[key]}"
            )
        payloads[key] = payload
        identities[key] = artifact_identity(paths[key], field)
    result = payloads["result"]
    lock = payloads["lock"]
    retention = payloads["retention"]
    checks = {
        "decision_ready": result.get("decision") == READY,
        "execution_not_yet_authorized": result.get(
            "execution_authorized"
        )
        is False,
        "lock_identity_exact": result.get("readiness_lock")
        == identities["lock"],
        "retention_passes": retention.get("passes") is True,
        "source_hashes_current": payloads["test_evidence"].get(
            "source_identities"
        )
        == {
            str(path.relative_to(REPO_ROOT)): sha256_path(path)
            for path in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)
        },
        "active_authority_exact": lock.get("active_authority_identity")
        == {
            "canonical_rows_sha256": authority_audit()[
                "canonical_rows_sha256"
            ],
            "root_set_sha256": authority_audit()["root_set_sha256"],
            "ancestry_set_sha256": authority_audit()[
                "ancestry_set_sha256"
            ],
            "stream_set_sha256": authority_audit()["stream_set_sha256"],
            "active_rows": ACTIVE_ROOTS,
            "active_streams": ACTIVE_STREAMS,
        },
        "bootstrap_contract_exact": lock.get("bootstrap_contract")
        == execution_schema()["bootstrap"],
    }
    if not all(checks.values()):
        raise J2A1ExecutionIntegrityError(
            "Execution-surface readiness package changed"
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
    execution_mode: str,
    scientific_authority: bool,
) -> dict[str, Any]:
    readiness = _verify_readiness_package(readiness_dir)
    return {
        "version": f"{VERSION}_execution_authorization_v1",
        "decision": DISTILLATION_AUTHORIZATION_DECISION,
        "execution_mode": execution_mode,
        "scientific_authority": bool(scientific_authority),
        "readiness_result": readiness["identities"]["result"],
        "readiness_lock": readiness["identities"]["lock"],
        "readiness_retention": readiness["identities"]["retention"],
        "execution_root": str(out_dir.resolve()),
        "jobs": 1,
        "authorized_commands": [
            "seal-phase-lock",
            "open",
            "materialize",
            "execute",
        ],
        "training_only": True,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
    }


def _load_authorization(
    path: Path,
    *,
    readiness_dir: Path,
    out_dir: Path,
    execution_mode: str,
) -> dict[str, Any]:
    payload = load_json(path)
    if not verify_payload_hash(
        payload,
        "authorization_payload_sha256",
    ):
        raise J2A1ExecutionIntegrityError(
            "Execution authorization payload changed"
        )
    readiness = _verify_readiness_package(readiness_dir)
    checks = {
        "decision": payload.get("decision")
        == DISTILLATION_AUTHORIZATION_DECISION,
        "mode": payload.get("execution_mode") == execution_mode,
        "scientific_authority": (
            payload.get("scientific_authority") is True
            if execution_mode == "scientific"
            else payload.get("scientific_authority") is False
        ),
        "readiness_result": payload.get("readiness_result")
        == readiness["identities"]["result"],
        "readiness_lock": payload.get("readiness_lock")
        == readiness["identities"]["lock"],
        "readiness_retention": payload.get("readiness_retention")
        == readiness["identities"]["retention"],
        "execution_root": payload.get("execution_root")
        == str(out_dir.resolve()),
        "jobs": int(payload.get("jobs", -1)) == 1,
        "training_only": payload.get("training_only") is True,
        "development_forbidden": payload.get("development_authorized")
        is False,
        "confirmation_forbidden": payload.get(
            "confirmation_authorized"
        )
        is False,
        "promotion_forbidden": payload.get("promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise J2A1ExecutionIntegrityError(
            "Execution authorization does not match readiness"
        )
    return {
        "payload": payload,
        "identity": artifact_identity(
            path,
            "authorization_payload_sha256",
        ),
        "readiness": readiness,
        "checks": checks,
        "passes": True,
    }


def _phase_contract_hash(
    *,
    lock: Mapping[str, Any],
    marker: Mapping[str, Any],
    manifest: Mapping[str, Any],
    command: str,
    execution_mode: str,
) -> str:
    return canonical_json_hash(
        {
            "lock_file_sha256": lock["file_sha256"],
            "lock_payload_sha256": lock["payload_sha256"],
            "marker_file_sha256": marker["file_sha256"],
            "marker_payload_sha256": marker["payload_sha256"],
            "manifest_file_sha256": manifest["file_sha256"],
            "manifest_payload_sha256": manifest["payload_sha256"],
            "command": command,
            "runner_sha256": sha256_path(RUNNER_PATH),
            "execution_mode": execution_mode,
        }
    )


def seal_phase_lock(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    jobs: int,
    execution_mode: str,
    include_operational: bool,
) -> dict[str, Any]:
    if jobs != 1:
        raise J2A1ExecutionIntegrityError("J2A1 jobs must equal one")
    if execution_mode not in {"scientific", "miniature_fixture"}:
        raise J2A1ExecutionIntegrityError(
            "J2A1 execution mode is invalid"
        )
    if execution_mode == "scientific" and out_dir.resolve() != (
        FUTURE_EXECUTION_DIR.resolve()
    ):
        raise J2A1ExecutionIntegrityError(
            "Scientific execution root changed"
        )
    if out_dir.exists():
        raise J2A1ExecutionIntegrityError(
            "Execution root must be absent before phase lock"
        )
    authorization = _load_authorization(
        authorization_path,
        readiness_dir=readiness_dir,
        out_dir=out_dir,
        execution_mode=execution_mode,
    )
    accountant = OutputAccountant(out_dir)
    operations = execution_operational_guard(
        phase_dir=out_dir,
        accountant=accountant,
        active_seconds=0.0,
        include_services=include_operational,
        require_target_disk=True,
    )
    authority = authority_audit()
    payload = {
        "version": f"{VERSION}_phase_lock_v1",
        "phase": "distillation_and_fidelity",
        "execution_mode": execution_mode,
        "scientific_authority": execution_mode == "scientific",
        "authorization": authorization["identity"],
        "readiness_result": authorization["readiness"]["identities"][
            "result"
        ],
        "readiness_lock": authorization["readiness"]["identities"]["lock"],
        "readiness_retention": authorization["readiness"]["identities"][
            "retention"
        ],
        "source_identities": authorization["readiness"]["payloads"][
            "test_evidence"
        ]["source_identities"],
        "authority_identity": {
            "canonical_rows_sha256": authority["canonical_rows_sha256"],
            "root_set_sha256": authority["root_set_sha256"],
            "ancestry_set_sha256": authority["ancestry_set_sha256"],
            "stream_set_sha256": authority["stream_set_sha256"],
            "active_rows": ACTIVE_ROOTS,
            "active_streams": ACTIVE_STREAMS,
        },
        "jobs": jobs,
        "stage_order": execution_schema()["stage_order"],
        "bootstrap_contract": execution_schema()["bootstrap"],
        "operations": operations,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "teacher_queries": 0,
        "games": 0,
        "optimizer_steps": 0,
    }
    return write_immutable_json(
        out_dir / PHASE_LOCK_NAME,
        payload,
        field="phase_lock_payload_sha256",
    )


def _load_phase_lock(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    execution_mode: str,
) -> dict[str, Any]:
    authorization = _load_authorization(
        authorization_path,
        readiness_dir=readiness_dir,
        out_dir=out_dir,
        execution_mode=execution_mode,
    )
    path = out_dir / PHASE_LOCK_NAME
    payload = load_json(path)
    if not verify_payload_hash(payload, "phase_lock_payload_sha256"):
        raise J2A1ExecutionIntegrityError("Phase lock changed")
    identity = artifact_identity(path, "phase_lock_payload_sha256")
    authority = authority_audit()
    checks = {
        "authorization": payload.get("authorization")
        == authorization["identity"],
        "readiness_result": payload.get("readiness_result")
        == authorization["readiness"]["identities"]["result"],
        "readiness_lock": payload.get("readiness_lock")
        == authorization["readiness"]["identities"]["lock"],
        "readiness_retention": payload.get("readiness_retention")
        == authorization["readiness"]["identities"]["retention"],
        "sources": payload.get("source_identities")
        == authorization["readiness"]["payloads"]["test_evidence"][
            "source_identities"
        ],
        "authority": payload.get("authority_identity")
        == {
            "canonical_rows_sha256": authority["canonical_rows_sha256"],
            "root_set_sha256": authority["root_set_sha256"],
            "ancestry_set_sha256": authority["ancestry_set_sha256"],
            "stream_set_sha256": authority["stream_set_sha256"],
            "active_rows": ACTIVE_ROOTS,
            "active_streams": ACTIVE_STREAMS,
        },
        "mode": payload.get("execution_mode") == execution_mode,
        "jobs": int(payload.get("jobs", -1)) == 1,
        "zero_before_open": all(
            int(payload.get(key, -1)) == 0
            for key in (
                "streams_reserved",
                "streams_consumed",
                "teacher_queries",
                "games",
                "optimizer_steps",
            )
        ),
    }
    if not all(checks.values()):
        raise J2A1ExecutionIntegrityError(
            "Phase lock no longer matches predecessors"
        )
    return {
        "payload": payload,
        "identity": identity,
        "authorization": authorization,
        "checks": checks,
        "passes": True,
    }


def open_phase(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    execution_mode: str,
    include_operational: bool,
) -> dict[str, Any]:
    lock = _load_phase_lock(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
        execution_mode=execution_mode,
    )
    if any(
        (out_dir / name).exists()
        for name in (
            MATERIALIZED_MANIFEST_NAME,
            RESERVATION_NAME,
            CONSUMPTION_NAME,
            GENESIS_NAME,
            TERMINAL_NAME,
        )
    ):
        raise J2A1ExecutionIntegrityError(
            "Open found premature execution artifacts"
        )
    operations = execution_operational_guard(
        phase_dir=out_dir,
        accountant=OutputAccountant(out_dir),
        active_seconds=0.0,
        include_services=include_operational,
        require_target_disk=True,
    )
    payload = {
        "version": f"{VERSION}_execution_marker_v1",
        "phase": "distillation_and_fidelity",
        "execution_mode": execution_mode,
        "scientific_authority": execution_mode == "scientific",
        "phase_lock": lock["identity"],
        "authorization": lock["authorization"]["identity"],
        "authority_identity": lock["payload"]["authority_identity"],
        "source_identities": lock["payload"]["source_identities"],
        "opened_at": time.time(),
        "hostname": socket.gethostname(),
        "command":
            "seal-phase-lock -> open -> materialize -> execute",
        "operations": operations,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "teacher_loads": 0,
        "teacher_queries": 0,
        "games": 0,
        "optimizer_steps": 0,
    }
    return write_immutable_json(
        out_dir / OPEN_MARKER_NAME,
        payload,
        field="execution_marker_payload_sha256",
    )


def _load_marker_and_lock(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    execution_mode: str,
) -> dict[str, Any]:
    lock = _load_phase_lock(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
        execution_mode=execution_mode,
    )
    marker_path = out_dir / OPEN_MARKER_NAME
    marker = load_json(marker_path)
    if not verify_payload_hash(
        marker,
        "execution_marker_payload_sha256",
    ):
        raise J2A1ExecutionIntegrityError("Execution marker changed")
    marker_identity = artifact_identity(
        marker_path,
        "execution_marker_payload_sha256",
    )
    checks = {
        "lock": marker.get("phase_lock") == lock["identity"],
        "authorization": marker.get("authorization")
        == lock["authorization"]["identity"],
        "authority": marker.get("authority_identity")
        == lock["payload"]["authority_identity"],
        "sources": marker.get("source_identities")
        == lock["payload"]["source_identities"],
        "mode": marker.get("execution_mode") == execution_mode,
        "zero_before_materialization": all(
            int(marker.get(key, -1)) == 0
            for key in (
                "streams_reserved",
                "streams_consumed",
                "teacher_loads",
                "teacher_queries",
                "games",
                "optimizer_steps",
            )
        ),
    }
    if not all(checks.values()):
        raise J2A1ExecutionIntegrityError(
            "Execution marker no longer matches phase lock"
        )
    return {
        "lock": lock,
        "marker": marker,
        "marker_identity": marker_identity,
        "checks": checks,
        "passes": True,
    }


def materialize_phase(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    execution_mode: str,
    include_operational: bool,
    rows_override: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    chain = _load_marker_and_lock(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
        execution_mode=execution_mode,
    )
    if rows_override is not None and execution_mode != "miniature_fixture":
        raise J2A1ExecutionIntegrityError(
            "Scientific materialization cannot override authority"
        )
    rows = (
        expected_active_rows()
        if rows_override is None
        else [j2.json_native(row) for row in rows_override]
    )
    if any(not isinstance(row, dict) for row in rows):
        raise J2A1ExecutionIntegrityError(
            "Materialized row is malformed"
        )
    if execution_mode == "scientific":
        audit = authority_audit()
        if (
            len(rows) != ACTIVE_ROOTS
            or canonical_json_hash(rows) != audit[
                "canonical_rows_sha256"
            ]
        ):
            raise J2A1ExecutionIntegrityError(
                "Scientific materialization changed A1 rows"
            )
    root_ids = [str(row["root_id"]) for row in rows]
    ancestry_ids = [str(row["ancestry_id"]) for row in rows]
    streams = [
        int(stream_id)
        for row in rows
        for stream_id in row["streams"].values()
    ]
    if (
        len(set(root_ids)) != len(root_ids)
        or len(set(ancestry_ids)) != len(ancestry_ids)
        or len(set(streams)) != len(streams)
    ):
        raise J2A1ExecutionIntegrityError(
            "Materialized authority has an identity collision"
        )
    operations = execution_operational_guard(
        phase_dir=out_dir,
        accountant=OutputAccountant(out_dir),
        active_seconds=0.0,
        include_services=include_operational,
        require_target_disk=True,
    )
    payload = {
        "version": f"{VERSION}_active_manifest_v1",
        "execution_mode": execution_mode,
        "marker": chain["marker_identity"],
        "phase_lock": chain["lock"]["identity"],
        "authority_identity": chain["marker"]["authority_identity"],
        "rows": rows,
        "row_count": len(rows),
        "canonical_rows_sha256": canonical_json_hash(rows),
        "root_set_sha256": canonical_json_hash(sorted(root_ids)),
        "ancestry_set_sha256": canonical_json_hash(sorted(ancestry_ids)),
        "stream_set_sha256": canonical_json_hash(sorted(streams)),
        "stream_count": len(streams),
        "stage_counts": dict(Counter(str(row["stage"]) for row in rows)),
        "operations": operations,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "content_opened": 0,
    }
    return write_immutable_json(
        out_dir / MATERIALIZED_MANIFEST_NAME,
        payload,
        field="active_manifest_payload_sha256",
    )


def _load_materialized_chain(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    execution_mode: str,
) -> dict[str, Any]:
    chain = _load_marker_and_lock(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
        execution_mode=execution_mode,
    )
    path = out_dir / MATERIALIZED_MANIFEST_NAME
    manifest = load_json(path)
    if not verify_payload_hash(
        manifest,
        "active_manifest_payload_sha256",
    ):
        raise J2A1ExecutionIntegrityError(
            "Materialized manifest changed"
        )
    identity = artifact_identity(
        path,
        "active_manifest_payload_sha256",
    )
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise J2A1ExecutionIntegrityError(
            "Materialized manifest rows are missing"
        )
    root_ids = [str(row["root_id"]) for row in rows]
    ancestries = [str(row["ancestry_id"]) for row in rows]
    streams = [
        int(value)
        for row in rows
        for value in row["streams"].values()
    ]
    checks = {
        "marker": manifest.get("marker") == chain["marker_identity"],
        "phase_lock": manifest.get("phase_lock")
        == chain["lock"]["identity"],
        "authority": manifest.get("authority_identity")
        == chain["marker"]["authority_identity"],
        "mode": manifest.get("execution_mode") == execution_mode,
        "row_hash": manifest.get("canonical_rows_sha256")
        == canonical_json_hash(rows),
        "root_hash": manifest.get("root_set_sha256")
        == canonical_json_hash(sorted(root_ids)),
        "ancestry_hash": manifest.get("ancestry_set_sha256")
        == canonical_json_hash(sorted(ancestries)),
        "stream_hash": manifest.get("stream_set_sha256")
        == canonical_json_hash(sorted(streams)),
        "counts": int(manifest.get("row_count", -1)) == len(rows)
        and int(manifest.get("stream_count", -1)) == len(streams),
        "sets_unique": len(set(root_ids)) == len(root_ids)
        and len(set(ancestries)) == len(ancestries)
        and len(set(streams)) == len(streams),
        "zero_before_execute": int(manifest.get("streams_reserved", -1))
        == 0
        and int(manifest.get("streams_consumed", -1)) == 0
        and int(manifest.get("content_opened", -1)) == 0,
    }
    if execution_mode == "scientific":
        audit = authority_audit()
        checks["scientific_rows_exact"] = (
            len(rows) == ACTIVE_ROOTS
            and canonical_json_hash(rows)
            == audit["canonical_rows_sha256"]
            and len(streams) == ACTIVE_STREAMS
        )
    if not all(checks.values()):
        raise J2A1ExecutionIntegrityError(
            "Materialized manifest no longer matches phase chain"
        )
    return {
        **chain,
        "manifest": manifest,
        "manifest_identity": identity,
        "rows": rows,
        "checks": checks,
        "passes": True,
    }


@dataclass(frozen=True)
class EngineConfig:
    execution_mode: str = "scientific"
    expected_bc_roots: int = BC_ROOTS
    expected_validation_pairs: int = VALIDATION_PAIRS
    distillation_epochs: int = j2.DISTILLATION_EPOCHS
    distillation_minibatch_size: int = j2.MINIBATCH_SIZE

    def validate(self) -> None:
        if self.execution_mode not in {"scientific", "miniature_fixture"}:
            raise J2A1ExecutionIntegrityError(
                "Engine execution mode is invalid"
            )
        if self.expected_bc_roots < 1 or self.expected_validation_pairs < 1:
            raise J2A1ExecutionIntegrityError(
                "Engine root counts are invalid"
            )
        if self.distillation_epochs != j2.DISTILLATION_EPOCHS:
            raise J2A1ExecutionIntegrityError(
                "Distillation epoch count changed"
            )
        if self.distillation_minibatch_size < 1:
            raise J2A1ExecutionIntegrityError(
                "Distillation minibatch size is invalid"
            )
        if self.execution_mode == "scientific" and (
            self.expected_bc_roots != BC_ROOTS
            or self.expected_validation_pairs != VALIDATION_PAIRS
            or self.distillation_minibatch_size != j2.MINIBATCH_SIZE
        ):
            raise J2A1ExecutionIntegrityError(
                "Scientific engine dimensions changed"
            )


def _seal_stage_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    if path.exists():
        observed = load_json(path)
        expected = payload_with_hash(payload, field)
        if observed != expected or not verify_payload_hash(observed, field):
            raise J2A1ExecutionIntegrityError(
                f"Stage artifact changed: {path}"
            )
        return observed
    return write_immutable_json(path, payload, field=field)


def _quarantine_checkpoint(
    *,
    phase_dir: Path,
    checkpoint: Mapping[str, Any],
    decision: str,
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    return _seal_stage_json(
        phase_dir / "J2A1_CHECKPOINT_QUARANTINE.json",
        {
            "version": f"{VERSION}_checkpoint_quarantine_v1",
            "decision": decision,
            "checkpoint": j2.json_native(checkpoint),
            "predecessor_sha256": scientific_hash(predecessor),
            "authoritative": False,
            "usable_for_ppo": False,
            "usable_for_development": False,
            "usable_for_confirmation": False,
            "passes": True,
        },
        field="quarantine_payload_sha256",
    )


def _authorize_checkpoint_after_final_guard(
    *,
    phase_dir: Path,
    engine: Mapping[str, Any],
    operational_guard_identity: Mapping[str, Any],
    execution_mode: str,
) -> dict[str, Any]:
    if (
        engine.get("decision") != READY_EXECUTION
        or engine.get("passes") is not True
        or engine.get("checkpoint_authoritative") is not False
        or engine.get(
            "checkpoint_authority_pending_final_operational_guard"
        )
        is not True
    ):
        raise J2A1ExecutionIntegrityError(
            "Checkpoint authority requested outside a passing final guard"
        )
    checkpoint = engine.get("checkpoint")
    mechanism = engine.get("mechanism")
    fidelity = engine.get("fidelity")
    if not all(
        isinstance(value, Mapping)
        for value in (checkpoint, mechanism, fidelity)
    ):
        raise J2A1ExecutionIntegrityError(
            "Checkpoint authority predecessors are incomplete"
        )
    checkpoint_path = Path(str(checkpoint["path"]))
    if (
        not checkpoint_path.is_file()
        or sha256_path(checkpoint_path) != checkpoint["file_sha256"]
    ):
        raise J2A1ExecutionIntegrityError(
            "Candidate checkpoint changed before authority"
        )
    return _seal_stage_json(
        phase_dir / "J2A1_DISTILLED_CHECKPOINT_AUTHORITY.json",
        {
            "version": f"{VERSION}_checkpoint_authority_v1",
            "decision": READY_EXECUTION,
            "execution_mode": execution_mode,
            "scientific_authority": execution_mode == "scientific",
            "checkpoint": j2.json_native(checkpoint),
            "mechanism_payload_sha256": mechanism[
                "mechanism_payload_sha256"
            ],
            "fidelity_payload_sha256": fidelity[
                "fidelity_payload_sha256"
            ],
            "final_operational_guard": j2.json_native(
                operational_guard_identity
            ),
            "authoritative": True,
            "usable_only_for_separately_reviewed_ppo_surface": True,
            "development_authorized": False,
            "confirmation_authorized": False,
            "promotion_authorized": False,
            "passes": True,
        },
        field="checkpoint_authority_payload_sha256",
    )


def execute_distillation_fidelity_engine(
    *,
    phase_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    contract_sha256: str,
    attempt_ledger: AttemptRuntimeLedger,
    config: EngineConfig,
    batch_collector: Callable[
        [Sequence[Mapping[str, Any]]],
        Sequence[Mapping[str, Any]],
    ],
    arm_runner: Callable[
        [Sequence[Mapping[str, Any]], j2.J2ActorCritic],
        Sequence[Mapping[str, Any]],
    ],
    family_inventory_fn: Callable[
        [Sequence[Mapping[str, Any]], int],
        Mapping[str, Any],
    ] | None = None,
    mechanism_fn: Callable[
        [
            j2.J2ActorCritic,
            Sequence[Mapping[str, Any]],
            Mapping[str, Any],
        ],
        Mapping[str, Any],
    ] | None = None,
    fidelity_fn: Callable[
        [
            Sequence[Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
        ],
        Mapping[str, Any],
    ] | None = None,
    boundary_callback: Callable[[str, Sequence[Path]], None] | None = None,
    interrupt_after_teacher_roots: int | None = None,
    interrupt_after_optimizer_steps: int | None = None,
    interrupt_before_optimizer_commit: int | None = None,
    interrupt_after_pairs: int | None = None,
) -> dict[str, Any]:
    config.validate()
    if config.execution_mode == "scientific" and any(
        value is not None
        for value in (family_inventory_fn, mechanism_fn, fidelity_fn)
    ):
        raise J2A1ExecutionIntegrityError(
            "Scientific engine cannot inject gate implementations"
        )
    bc_rows = [row for row in rows if row["stage"] == BC_STAGE]
    validation_rows = [
        row for row in rows if row["stage"] == VALIDATION_STAGE
    ]
    if (
        len(bc_rows) != config.expected_bc_roots
        or len(validation_rows) != config.expected_validation_pairs
    ):
        raise J2A1ExecutionIntegrityError(
            "Engine manifest stage counts changed"
        )
    collection = bounded_collect_teacher_roots(
        phase_dir=phase_dir,
        rows=rows,
        contract_sha256=contract_sha256,
        batch_collector=batch_collector,
        attempt_ledger=attempt_ledger,
        boundary_callback=boundary_callback,
        interrupt_after_completed=interrupt_after_teacher_roots,
    )
    if boundary_callback is not None:
        boundary_callback(
            "teacher_collection_seal",
            (
                phase_dir / "J2A1_TEACHER_COLLECTION_COMPLETE.json",
                phase_dir / "teacher_root_completions.jsonl",
                attempt_ledger.path,
            ),
        )
    rows_by_root = {str(row["root_id"]): row for row in rows}
    roots = load_teacher_roots_from_refs(
        phase_dir=phase_dir,
        refs=collection["refs"],
        rows_by_root=rows_by_root,
    )
    refs_by_root = {
        str(ref["root_id"]): ref for ref in collection["refs"]
    }
    for root in roots:
        ref = refs_by_root[root["root_id"]]
        root["file_sha256"] = ref["file_sha256"]
        root["relative_path"] = ref["relative_path"]
    bc_roots = [root for root in roots if root["stage"] == BC_STAGE]
    validation_roots = [
        root for root in roots if root["stage"] == VALIDATION_STAGE
    ]
    if family_inventory_fn is None:
        inventory = strict_feature_inventory(
            validation_roots,
            expected_root_count=config.expected_validation_pairs,
        )
    else:
        inventory = dict(
            family_inventory_fn(
                validation_roots,
                config.expected_validation_pairs,
            )
        )
    inventory_payload = _seal_stage_json(
        phase_dir / "J2A1_VALIDATION_FEATURE_INVENTORY.json",
        inventory,
        field="inventory_payload_sha256",
    )
    if boundary_callback is not None:
        boundary_callback(
            "family_inventory_seal",
            (phase_dir / "J2A1_VALIDATION_FEATURE_INVENTORY.json",),
        )
    if inventory.get("passes") is not True:
        return {
            "decision": HOLD_FAMILY,
            "stage": "family_support",
            "collection": collection["seal"],
            "inventory": inventory_payload,
            "checkpoint": None,
            "checkpoint_authoritative": False,
            "attempt_summary": attempt_ledger.summary(),
            "passes": False,
        }
    collection_payload_sha = collection["seal"][
        "collection_payload_sha256"
    ]
    distillation = bounded_distillation(
        phase_dir=phase_dir,
        bc_roots=bc_roots,
        contract_sha256=contract_sha256,
        collection_seal_sha256=collection_payload_sha,
        family_inventory_sha256=inventory_payload[
            "inventory_payload_sha256"
        ],
        attempt_ledger=attempt_ledger,
        expected_root_count=config.expected_bc_roots,
        minibatch_size=config.distillation_minibatch_size,
        epochs=config.distillation_epochs,
        boundary_callback=boundary_callback,
        interrupt_after_committed_steps=interrupt_after_optimizer_steps,
        interrupt_before_commit_step=interrupt_before_optimizer_commit,
    )
    if boundary_callback is not None:
        boundary_callback(
            "epoch8_checkpoint_seal",
            (
                Path(distillation["checkpoint"]["path"]),
                phase_dir / "optimizer_resume" / "updater_journal.jsonl",
                phase_dir / "optimizer_resume" / "updater_head.json",
                attempt_ledger.path,
            ),
        )
    checkpoint_path = Path(distillation["checkpoint"]["path"])
    checkpoint_payload = _deserialize_torch_payload(
        checkpoint_path.read_bytes(),
        magic=CHECKPOINT_MAGIC,
    )
    model, _optimizer = validate_epoch8_checkpoint(checkpoint_payload)
    if mechanism_fn is None:
        mechanism = mechanism_metrics(
            model,
            validation_roots,
            inventory,
        )
    else:
        mechanism = dict(
            mechanism_fn(model, validation_roots, inventory)
        )
    mechanism_payload = _seal_stage_json(
        phase_dir / "J2A1_BC_MECHANISM_RESULT.json",
        mechanism,
        field="mechanism_payload_sha256",
    )
    retirement = seal_retirement(
        phase_dir=phase_dir,
        name="bc_batch_after_epoch8_mechanism_seal",
        paths=[phase_dir / "ephemeral_bc_batch" / "current_batch.bin"],
        predecessor_sha256=mechanism_payload[
            "mechanism_payload_sha256"
        ],
    )
    if boundary_callback is not None:
        boundary_callback(
            "mechanism_and_batch_retirement_seal",
            (
                phase_dir / "J2A1_BC_MECHANISM_RESULT.json",
                phase_dir
                / "retirements"
                / "bc_batch_after_epoch8_mechanism_seal.json",
                phase_dir / "ephemeral_bc_batch" / "current_batch.bin",
            ),
        )
    if mechanism.get("passes") is not True:
        quarantine = _quarantine_checkpoint(
            phase_dir=phase_dir,
            checkpoint=distillation["checkpoint"],
            decision=HOLD_MECHANISM,
            predecessor=mechanism_payload,
        )
        return {
            "decision": HOLD_MECHANISM,
            "stage": "bc_mechanism",
            "collection": collection["seal"],
            "inventory": inventory_payload,
            "distillation": distillation,
            "mechanism": mechanism_payload,
            "retirement": retirement,
            "checkpoint": distillation["checkpoint"],
            "quarantine": quarantine,
            "checkpoint_authoritative": False,
            "attempt_summary": attempt_ledger.summary(),
            "passes": False,
        }
    teacher_by_root = {
        root["root_id"]: root for root in validation_roots
    }
    panel = bounded_collect_fidelity_pairs(
        phase_dir=phase_dir,
        rows=validation_rows,
        teacher_roots=teacher_by_root,
        checkpoint_path=checkpoint_path,
        contract_sha256=contract_sha256,
        attempt_ledger=attempt_ledger,
        arm_runner=arm_runner,
        boundary_callback=boundary_callback,
        interrupt_after_completed=interrupt_after_pairs,
    )
    if boundary_callback is not None:
        boundary_callback(
            "fidelity_panel_seal",
            (
                phase_dir / "J2A1_FIDELITY_PANEL_COMPLETE.json",
                phase_dir / "fidelity_pair_completions.jsonl",
                attempt_ledger.path,
            ),
        )
    pair_records = load_fidelity_pairs_after_seal(
        phase_dir=phase_dir,
        panel=panel,
        rows=validation_rows,
    )
    if fidelity_fn is None:
        fidelity = analyze_closed_loop_fidelity(
            pair_records,
            validation_rows,
        )
    else:
        fidelity = dict(fidelity_fn(pair_records, validation_rows))
    fidelity_payload = _seal_stage_json(
        phase_dir / "J2A1_CLOSED_LOOP_FIDELITY_RESULT.json",
        fidelity,
        field="fidelity_payload_sha256",
    )
    if boundary_callback is not None:
        boundary_callback(
            "fidelity_analysis_seal",
            (phase_dir / "J2A1_CLOSED_LOOP_FIDELITY_RESULT.json",),
        )
    if fidelity.get("passes") is not True:
        decision = str(fidelity.get("decision", HOLD_FIDELITY))
        if decision not in {HOLD_FIDELITY, HOLD_BASE_RATE}:
            decision = HOLD_FIDELITY
        quarantine = _quarantine_checkpoint(
            phase_dir=phase_dir,
            checkpoint=distillation["checkpoint"],
            decision=decision,
            predecessor=fidelity_payload,
        )
        return {
            "decision": decision,
            "stage": "closed_loop_fidelity",
            "collection": collection["seal"],
            "inventory": inventory_payload,
            "distillation": distillation,
            "mechanism": mechanism_payload,
            "panel": panel["seal"],
            "fidelity": fidelity_payload,
            "retirement": retirement,
            "checkpoint": distillation["checkpoint"],
            "quarantine": quarantine,
            "checkpoint_authoritative": False,
            "attempt_summary": attempt_ledger.summary(),
            "passes": False,
        }
    return {
        "decision": READY_EXECUTION,
        "stage": "closed_loop_fidelity",
        "collection": collection["seal"],
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
        "attempt_summary": attempt_ledger.summary(),
        "passes": True,
    }


def _reservation_payload(
    chain: Mapping[str, Any],
    *,
    owner_record_sha256: str,
) -> dict[str, Any]:
    rows = chain["rows"]
    streams = sorted(
        int(value)
        for row in rows
        for value in row["streams"].values()
    )
    return {
        "version": f"{VERSION}_stream_reservation_v1",
        "phase_lock": chain["lock"]["identity"],
        "marker": chain["marker_identity"],
        "manifest": chain["manifest_identity"],
        "owner_record_sha256": owner_record_sha256,
        "row_count": len(rows),
        "stream_count": len(streams),
        "stream_set_sha256": canonical_json_hash(streams),
        "root_set_sha256": chain["manifest"]["root_set_sha256"],
        "ancestry_set_sha256": chain["manifest"]["ancestry_set_sha256"],
        "validation_crn_pair_authority": (
            "one reservation per row; shared logical/deck/slot with "
            "distinct teacher/student policy streams"
        ),
        "reserved": True,
        "consumed": False,
    }


def _consumption_payload(
    chain: Mapping[str, Any],
    *,
    reservation_identity: Mapping[str, Any],
    opening_owner_record_sha256: str,
) -> dict[str, Any]:
    return {
        "version": f"{VERSION}_stream_consumption_v1",
        "phase_lock": chain["lock"]["identity"],
        "marker": chain["marker_identity"],
        "manifest": chain["manifest_identity"],
        "reservation": dict(reservation_identity),
        "opening_owner_record_sha256": opening_owner_record_sha256,
        "row_count": len(chain["rows"]),
        "stream_count": int(chain["manifest"]["stream_count"]),
        "stream_set_sha256": chain["manifest"]["stream_set_sha256"],
        "pair_row_consumption_semantics":
            "validation exogenous authority consumed once for both "
            "precommitted arms",
        "consumed": True,
    }


def _verify_existing_exact(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    observed = load_json(path)
    expected = payload_with_hash(payload, field)
    if observed != expected or not verify_payload_hash(observed, field):
        raise J2A1ExecutionIntegrityError(
            f"Existing phase artifact changed: {path}"
        )
    return observed


def _terminal_base(
    *,
    chain: Mapping[str, Any],
    engine: Mapping[str, Any],
    attempt_summary: Mapping[str, Any],
    reservation: Mapping[str, Any],
    consumption: Mapping[str, Any],
    ownership: OwnershipLedger,
    execution_mode: str,
) -> dict[str, Any]:
    decision = str(engine["decision"])
    integrity_kill = decision == KILL_EXECUTION
    checkpoint_authoritative = bool(
        engine.get("checkpoint_authoritative")
    ) and execution_mode == "scientific"
    return {
        "version": f"{VERSION}_terminal_v1",
        "decision": decision,
        "execution_mode": execution_mode,
        "scientific_authority": execution_mode == "scientific",
        "successor_review_authority": (
            decision == READY_EXECUTION and checkpoint_authoritative
        ),
        "phase_lock": chain["lock"]["identity"],
        "marker": chain["marker_identity"],
        "manifest": chain["manifest_identity"],
        "reservation": reservation,
        "consumption": consumption,
        "ownership_head_sha256": ownership.records[-1][
            "owner_record_sha256"
        ],
        "ownership_record_count": len(ownership.records),
        "engine": j2.json_native(engine),
        "attempt_summary": dict(attempt_summary),
        "integrity_passes": not integrity_kill,
        "checkpoint_authoritative": checkpoint_authoritative,
        "ppo_execution_authorized": False,
        "development_authorized": False,
        "confirmation_authorized": False,
        "promotion_authorized": False,
        "human_session_reads": 0,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
        "promote": False,
        "continue": (
            decision == READY_EXECUTION and checkpoint_authoritative
        ),
        "hold": decision != READY_EXECUTION,
        "kill": integrity_kill,
    }


def _execution_retention_inventory(
    out_dir: Path,
) -> list[dict[str, Any]]:
    retention_path = out_dir / EXECUTION_RETENTION_NAME
    terminal_path = out_dir / TERMINAL_NAME
    return [
        {
            "path": str(path.relative_to(out_dir)),
            "bytes": int(path.stat().st_size),
            "file_sha256": sha256_path(path),
        }
        for path in sorted(out_dir.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path != retention_path
        and path != terminal_path
    ]


def seal_execution_retention(out_dir: Path) -> dict[str, Any]:
    evidence_path = out_dir / TERMINAL_EVIDENCE_NAME
    evidence = load_json(evidence_path)
    if not verify_payload_hash(
        evidence,
        "terminal_evidence_payload_sha256",
    ):
        raise J2A1ExecutionIntegrityError(
            "Terminal evidence changed before retention"
        )
    inventory = _execution_retention_inventory(out_dir)
    payload = {
        "version": f"{VERSION}_execution_retention_v1",
        "terminal_evidence": artifact_identity(
            evidence_path,
            "terminal_evidence_payload_sha256",
        ),
        "files": inventory,
        "file_count": len(inventory),
        "retained_bytes": sum(row["bytes"] for row in inventory),
        "canonical_inventory_sha256": canonical_json_hash(inventory),
        "no_cleanup_performed": True,
        "checkpoint_authoritative": evidence[
            "checkpoint_authoritative"
        ],
        "development_unopened": True,
        "confirmation_unopened": True,
        "human_sessions_unread": evidence["human_session_reads"] == 0,
        "incumbent_unchanged": evidence["incumbent_changes"] == 0,
        "dashboard_unchanged": evidence["dashboard_changes"] == 0,
        "passes": True,
    }
    path = out_dir / EXECUTION_RETENTION_NAME
    if path.exists():
        return _verify_existing_exact(
            path,
            payload,
            field="retention_payload_sha256",
        )
    return write_immutable_json(
        path,
        payload,
        field="retention_payload_sha256",
    )


def _seal_terminal_and_retention(
    *,
    out_dir: Path,
    terminal_payload: Mapping[str, Any],
    accountant: OutputAccountant,
) -> dict[str, Any]:
    reserve = 16 * 1024**2
    if accountant.snapshot()["output_bytes"] + reserve > STORAGE_CAP_BYTES:
        raise J2A1ExecutionOperationalHold(
            "Terminal and retention allowance exceeds output cap"
        )
    evidence_path = out_dir / TERMINAL_EVIDENCE_NAME
    if evidence_path.exists():
        evidence = _verify_existing_exact(
            evidence_path,
            terminal_payload,
            field="terminal_evidence_payload_sha256",
        )
    else:
        evidence = write_immutable_json(
            evidence_path,
            terminal_payload,
            field="terminal_evidence_payload_sha256",
        )
    accountant.record_path(evidence_path)
    retention = seal_execution_retention(out_dir)
    retention_path = out_dir / EXECUTION_RETENTION_NAME
    accountant.record_path(retention_path)
    final_payload = {
        **dict(terminal_payload),
        "terminal_evidence": artifact_identity(
            evidence_path,
            "terminal_evidence_payload_sha256",
        ),
        "retention": artifact_identity(
            retention_path,
            "retention_payload_sha256",
        ),
        "authoritative_terminal_written_last": True,
    }
    staged = payload_with_hash(final_payload, "terminal_payload_sha256")
    staged_bytes = (
        json.dumps(staged, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    projected_final_bytes = (
        accountant.snapshot()["output_bytes"] + len(staged_bytes)
    )
    if projected_final_bytes > STORAGE_CAP_BYTES:
        raise J2A1ExecutionOperationalHold(
            "Staged authoritative terminal exceeds output cap"
        )
    for _ in range(4):
        final_payload["projected_final_output_bytes"] = (
            projected_final_bytes
        )
        staged = payload_with_hash(
            final_payload,
            "terminal_payload_sha256",
        )
        staged_bytes = (
            json.dumps(staged, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        updated = accountant.snapshot()["output_bytes"] + len(staged_bytes)
        if updated == projected_final_bytes:
            break
        projected_final_bytes = updated
    else:
        raise J2A1ExecutionIntegrityError(
            "Terminal output projection did not stabilize"
        )
    if projected_final_bytes > STORAGE_CAP_BYTES:
        raise J2A1ExecutionOperationalHold(
            "Final authoritative terminal exceeds output cap"
        )
    terminal_path = out_dir / TERMINAL_NAME
    if terminal_path.exists():
        terminal = _verify_existing_exact(
            terminal_path,
            final_payload,
            field="terminal_payload_sha256",
        )
    else:
        terminal = write_immutable_json(
            terminal_path,
            final_payload,
            field="terminal_payload_sha256",
        )
    accountant.record_path(terminal_path)
    final = accountant.snapshot()
    return {
        "terminal": terminal,
        "retention": retention,
        "final_output": final,
        "post_terminal_checks_performed": 0,
        "passes": True,
    }


def _load_existing_execution_terminal(
    *,
    out_dir: Path,
    chain: Mapping[str, Any],
    execution_mode: str,
    accountant: OutputAccountant,
) -> dict[str, Any] | None:
    path = out_dir / TERMINAL_NAME
    if not path.exists():
        return None
    terminal = load_json(path)
    if not verify_payload_hash(terminal, "terminal_payload_sha256"):
        raise J2A1ExecutionIntegrityError(
            "Existing execution terminal changed"
        )
    checks = {
        "mode": terminal.get("execution_mode") == execution_mode,
        "authority": terminal.get("scientific_authority")
        is (execution_mode == "scientific"),
        "lock": terminal.get("phase_lock") == chain["lock"]["identity"],
        "marker": terminal.get("marker") == chain["marker_identity"],
        "manifest": terminal.get("manifest")
        == chain["manifest_identity"],
        "no_successor_execution": terminal.get("ppo_execution_authorized")
        is False,
        "no_development": terminal.get("development_authorized") is False,
        "no_confirmation": terminal.get("confirmation_authorized") is False,
        "no_promotion": terminal.get("promotion_authorized") is False,
    }
    if not all(checks.values()):
        raise J2A1ExecutionIntegrityError(
            "Existing terminal does not match phase chain"
        )
    retention = seal_execution_retention(out_dir)
    retention_identity = artifact_identity(
        out_dir / EXECUTION_RETENTION_NAME,
        "retention_payload_sha256",
    )
    evidence_identity = artifact_identity(
        out_dir / TERMINAL_EVIDENCE_NAME,
        "terminal_evidence_payload_sha256",
    )
    if (
        terminal.get("retention") != retention_identity
        or terminal.get("terminal_evidence") != evidence_identity
        or terminal.get("authoritative_terminal_written_last") is not True
    ):
        raise J2A1ExecutionIntegrityError(
            "Existing terminal lost its evidence/retention binding"
        )
    accountant.record_path(path)
    accountant.record_path(out_dir / EXECUTION_RETENTION_NAME)
    return {
        "terminal": terminal,
        "retention": retention,
        "existing_terminal_reused": True,
        "checks": checks,
        "passes": True,
    }


def _load_or_write_genesis(
    *,
    out_dir: Path,
    chain: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_execution_genesis_v1",
        "phase_lock": chain["lock"]["identity"],
        "marker": chain["marker_identity"],
        "manifest": chain["manifest_identity"],
        "contract_sha256": contract_sha256,
        "sequence": 0,
        "completed_teacher_roots": 0,
        "optimizer_steps": 0,
        "completed_fidelity_pairs": 0,
        "attempts_started": 0,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "passes": True,
    }
    path = out_dir / GENESIS_NAME
    if path.exists():
        return _verify_existing_exact(
            path,
            payload,
            field="genesis_payload_sha256",
        )
    return write_immutable_json(
        path,
        payload,
        field="genesis_payload_sha256",
    )


def _owner_in_ledger(
    ownership: OwnershipLedger,
    owner_sha256: str,
) -> bool:
    return any(
        record.get("kind") == "owner"
        and record.get("owner_record_sha256") == owner_sha256
        for record in ownership.records
    )


def _load_or_write_reservation(
    *,
    out_dir: Path,
    chain: Mapping[str, Any],
    ownership: OwnershipLedger,
) -> dict[str, Any]:
    path = out_dir / RESERVATION_NAME
    current_owner = ownership.records[-1]["owner_record_sha256"]
    if path.exists():
        observed = load_json(path)
        if not verify_payload_hash(
            observed,
            "reservation_payload_sha256",
        ):
            raise J2A1ExecutionIntegrityError(
                "Stream reservation changed"
            )
        opening_owner = str(observed.get("owner_record_sha256"))
        if not _owner_in_ledger(ownership, opening_owner):
            raise J2A1ExecutionIntegrityError(
                "Stream reservation owner is not in ownership ancestry"
            )
        expected = _reservation_payload(
            chain,
            owner_record_sha256=opening_owner,
        )
        return _verify_existing_exact(
            path,
            expected,
            field="reservation_payload_sha256",
        )
    payload = _reservation_payload(
        chain,
        owner_record_sha256=current_owner,
    )
    return write_immutable_json(
        path,
        payload,
        field="reservation_payload_sha256",
    )


def _load_or_write_consumption(
    *,
    out_dir: Path,
    chain: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    ownership: OwnershipLedger,
) -> dict[str, Any]:
    path = out_dir / CONSUMPTION_NAME
    current_owner = ownership.records[-1]["owner_record_sha256"]
    if path.exists():
        observed = load_json(path)
        if not verify_payload_hash(
            observed,
            "consumption_payload_sha256",
        ):
            raise J2A1ExecutionIntegrityError(
                "Stream consumption changed"
            )
        opening_owner = str(
            observed.get("opening_owner_record_sha256")
        )
        if not _owner_in_ledger(ownership, opening_owner):
            raise J2A1ExecutionIntegrityError(
                "Stream consumption opener is not an ownership ancestor"
            )
        expected = _consumption_payload(
            chain,
            reservation_identity=reservation_identity,
            opening_owner_record_sha256=opening_owner,
        )
        return _verify_existing_exact(
            path,
            expected,
            field="consumption_payload_sha256",
        )
    payload = _consumption_payload(
        chain,
        reservation_identity=reservation_identity,
        opening_owner_record_sha256=current_owner,
    )
    return write_immutable_json(
        path,
        payload,
        field="consumption_payload_sha256",
    )


def execute_phase_from_artifacts(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    execution_mode: str,
    include_operational: bool,
    command: str,
    jobs: int,
    config: EngineConfig | None = None,
    batch_collector: Callable[
        [Sequence[Mapping[str, Any]]],
        Sequence[Mapping[str, Any]],
    ] | None = None,
    arm_runner: Callable[
        [Sequence[Mapping[str, Any]], j2.J2ActorCritic],
        Sequence[Mapping[str, Any]],
    ] | None = None,
    family_inventory_fn: Callable[
        [Sequence[Mapping[str, Any]], int],
        Mapping[str, Any],
    ] | None = None,
    mechanism_fn: Callable[
        [
            j2.J2ActorCritic,
            Sequence[Mapping[str, Any]],
            Mapping[str, Any],
        ],
        Mapping[str, Any],
    ] | None = None,
    fidelity_fn: Callable[
        [
            Sequence[Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
        ],
        Mapping[str, Any],
    ] | None = None,
    interrupt_stage: str | None = None,
    owner_pid: int | None = None,
    owner_start_identity: str | None = None,
) -> dict[str, Any]:
    if jobs != 1:
        raise J2A1ExecutionIntegrityError("J2A1 jobs must equal one")
    chain = _load_materialized_chain(
        readiness_dir=readiness_dir,
        authorization_path=authorization_path,
        out_dir=out_dir,
        execution_mode=execution_mode,
    )
    accountant = OutputAccountant(out_dir)
    existing = _load_existing_execution_terminal(
        out_dir=out_dir,
        chain=chain,
        execution_mode=execution_mode,
        accountant=accountant,
    )
    if existing is not None:
        return existing
    runtime = configure_deterministic_runtime()
    del runtime
    execution_operational_guard(
        phase_dir=out_dir,
        accountant=accountant,
        active_seconds=0.0,
        include_services=include_operational,
        require_target_disk=True,
    )
    contract_sha256 = _phase_contract_hash(
        lock={
            "file_sha256": chain["lock"]["identity"]["file_sha256"],
            "payload_sha256": chain["lock"]["identity"]["payload_sha256"],
        },
        marker={
            "file_sha256": chain["marker_identity"]["file_sha256"],
            "payload_sha256": chain["marker_identity"]["payload_sha256"],
        },
        manifest={
            "file_sha256": chain["manifest_identity"]["file_sha256"],
            "payload_sha256": chain["manifest_identity"]["payload_sha256"],
        },
        command=command,
        execution_mode=execution_mode,
    )
    genesis = _load_or_write_genesis(
        out_dir=out_dir,
        chain=chain,
        contract_sha256=contract_sha256,
    )
    accountant.record_path(out_dir / GENESIS_NAME)
    ownership = OwnershipLedger(
        path=out_dir / "ownership_ledger.jsonl",
        contract_sha256=contract_sha256,
    )
    if not ownership.records:
        ownership.acquire(
            marker_sha256=chain["marker_identity"]["file_sha256"],
            lock_sha256=chain["lock"]["identity"]["file_sha256"],
            manifest_sha256=chain["manifest_identity"]["file_sha256"],
            command=command,
            execution_mode=execution_mode,
            pid=owner_pid,
            start_identity=owner_start_identity,
        )
    else:
        last = ownership.records[-1]
        if _pid_alive(int(last["pid"])):
            ownership.verify(
                marker_sha256=chain["marker_identity"]["file_sha256"],
                lock_sha256=chain["lock"]["identity"]["file_sha256"],
                manifest_sha256=chain["manifest_identity"]["file_sha256"],
                command=command,
                execution_mode=execution_mode,
                require_current_pid=owner_pid is None,
            )
        else:
            ownership.reclaim_dead(
                marker_sha256=chain["marker_identity"]["file_sha256"],
                lock_sha256=chain["lock"]["identity"]["file_sha256"],
                manifest_sha256=chain["manifest_identity"]["file_sha256"],
                command=command,
                execution_mode=execution_mode,
                boundary=genesis,
                pid=owner_pid,
                start_identity=owner_start_identity,
            )
    accountant.record_path(out_dir / "ownership_ledger.jsonl")
    if interrupt_stage == "after_owner":
        raise J2A1PlannedInterruption(
            "fixture interruption after owner"
        )
    reservation = _load_or_write_reservation(
        out_dir=out_dir,
        chain=chain,
        ownership=ownership,
    )
    reservation_identity = artifact_identity(
        out_dir / RESERVATION_NAME,
        "reservation_payload_sha256",
    )
    accountant.record_path(out_dir / RESERVATION_NAME)
    if interrupt_stage == "after_reservation":
        raise J2A1PlannedInterruption(
            "fixture interruption after reservation"
        )
    consumption = _load_or_write_consumption(
        out_dir=out_dir,
        chain=chain,
        reservation_identity=reservation_identity,
        ownership=ownership,
    )
    consumption_identity = artifact_identity(
        out_dir / CONSUMPTION_NAME,
        "consumption_payload_sha256",
    )
    accountant.record_path(out_dir / CONSUMPTION_NAME)
    if interrupt_stage == "after_consumption":
        raise J2A1PlannedInterruption(
            "fixture interruption after consumption"
        )
    attempt_ledger = AttemptRuntimeLedger(
        path=out_dir / "attempt_runtime_ledger.jsonl",
        contract_sha256=contract_sha256,
    )
    effective_config = (
        EngineConfig(execution_mode=execution_mode)
        if config is None
        else config
    )
    effective_config.validate()
    if effective_config.execution_mode != execution_mode:
        raise J2A1ExecutionIntegrityError(
            "Engine config mode changed from phase mode"
        )
    if execution_mode == "scientific":
        if any(
            value is not None
            for value in (
                batch_collector,
                arm_runner,
                family_inventory_fn,
                mechanism_fn,
                fidelity_fn,
            )
        ):
            raise J2A1ExecutionIntegrityError(
                "Scientific dispatcher cannot inject execution callbacks"
            )
        binding = _teacher_binding_from_authority()
        worker_group = TeacherRootWorkerGroup(binding=binding)
        batch_collector = worker_group.collect
        arm_runner = lambda rows, model: run_student_arms_synchronously(
            rows=rows,
            model=model,
        )
    else:
        worker_group = None
        if batch_collector is None or arm_runner is None:
            raise J2A1ExecutionIntegrityError(
                "Miniature fixture callbacks are required"
            )
    boundary_state: dict[str, Any] = {
        "count": 0,
        "by_kind": Counter(),
        "latest_guard_sha256": None,
    }

    def boundary_guard(kind: str, paths: Sequence[Path]) -> None:
        for path in paths:
            if path.exists() and path.is_file():
                accountant.record_path(path)
        observed = execution_operational_guard(
            phase_dir=out_dir,
            accountant=accountant,
            active_seconds=float(
                attempt_ledger.summary()["active_seconds"]
            ),
            include_services=include_operational,
            require_target_disk=False,
        )
        boundary_state["count"] += 1
        boundary_state["by_kind"][kind] += 1
        boundary_state["latest_guard_sha256"] = scientific_hash(observed)

    try:
        engine = execute_distillation_fidelity_engine(
            phase_dir=out_dir,
            rows=chain["rows"],
            contract_sha256=contract_sha256,
            attempt_ledger=attempt_ledger,
            config=effective_config,
            batch_collector=batch_collector,
            arm_runner=arm_runner,
            family_inventory_fn=family_inventory_fn,
            mechanism_fn=mechanism_fn,
            fidelity_fn=fidelity_fn,
            boundary_callback=boundary_guard,
        )
    except J2A1PlannedInterruption:
        raise
    except J2A1ExecutionOperationalHold as error:
        engine = {
            "decision": HOLD_OPERATIONAL,
            "stage": "operational",
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "checkpoint_authoritative": False,
            "attempt_summary": attempt_ledger.summary(),
            "passes": False,
        }
    except J2A1ExecutionDataHold as error:
        engine = {
            "decision": HOLD,
            "stage": "data_support",
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "checkpoint_authoritative": False,
            "attempt_summary": attempt_ledger.summary(),
            "passes": False,
        }
    except BaseException as error:
        engine = {
            "decision": KILL_EXECUTION,
            "stage": "integrity",
            "error_type": type(error).__name__,
            "error_sha256": hashlib.sha256(
                str(error).encode("utf-8")
            ).hexdigest(),
            "checkpoint_authoritative": False,
            "attempt_summary": attempt_ledger.summary(),
            "passes": False,
        }
    finally:
        if worker_group is not None:
            worker_group.close(terminate=False)
    accountant.reconcile()
    final_guard: dict[str, Any] | None = None
    final_guard_error: J2A1ExecutionOperationalHold | None = None
    try:
        final_guard = execution_operational_guard(
            phase_dir=out_dir,
            accountant=accountant,
            active_seconds=float(
                attempt_ledger.summary()["active_seconds"]
            ),
            include_services=include_operational,
            require_target_disk=False,
        )
    except J2A1ExecutionOperationalHold as error:
        final_guard_error = error
    if final_guard_error is not None:
        if engine.get("decision") == KILL_EXECUTION:
            engine = {
                **dict(engine),
                "final_operational_guard_passed": False,
                "final_operational_error_type": type(
                    final_guard_error
                ).__name__,
                "final_operational_error_sha256": hashlib.sha256(
                    str(final_guard_error).encode("utf-8")
                ).hexdigest(),
            }
        else:
            prior_decision = str(engine.get("decision"))
            checkpoint = engine.get("checkpoint")
            quarantine = None
            if isinstance(checkpoint, Mapping):
                quarantine = _quarantine_checkpoint(
                    phase_dir=out_dir,
                    checkpoint=checkpoint,
                    decision=HOLD_OPERATIONAL,
                    predecessor={
                        "prior_decision": prior_decision,
                        "error_type": type(final_guard_error).__name__,
                        "error_sha256": hashlib.sha256(
                            str(final_guard_error).encode("utf-8")
                        ).hexdigest(),
                    },
                )
            engine = {
                "decision": HOLD_OPERATIONAL,
                "stage": "final_operational_guard",
                "prior_engine_decision": prior_decision,
                "error_type": type(final_guard_error).__name__,
                "error_sha256": hashlib.sha256(
                    str(final_guard_error).encode("utf-8")
                ).hexdigest(),
                "checkpoint": checkpoint,
                "quarantine": quarantine,
                "checkpoint_authoritative": False,
                "attempt_summary": attempt_ledger.summary(),
                "passes": False,
            }
    else:
        if final_guard is None:
            raise J2A1ExecutionIntegrityError(
                "Final operational guard produced no evidence"
            )
        final_guard_path = out_dir / FINAL_OPERATIONAL_GUARD_NAME
        if final_guard_path.exists():
            sealed_guard = load_json(final_guard_path)
            if (
                not verify_payload_hash(
                    sealed_guard,
                    "final_operational_guard_payload_sha256",
                )
                or sealed_guard.get("passes") is not True
            ):
                raise J2A1ExecutionIntegrityError(
                    "Sealed final operational guard changed"
                )
        else:
            sealed_guard = write_immutable_json(
                final_guard_path,
                final_guard,
                field="final_operational_guard_payload_sha256",
            )
        accountant.record_path(final_guard_path)
        final_guard_identity = artifact_identity(
            final_guard_path,
            "final_operational_guard_payload_sha256",
        )
        if engine.get("decision") == READY_EXECUTION:
            try:
                authority = _authorize_checkpoint_after_final_guard(
                    phase_dir=out_dir,
                    engine=engine,
                    operational_guard_identity=final_guard_identity,
                    execution_mode=execution_mode,
                )
                accountant.record_path(
                    out_dir / "J2A1_DISTILLED_CHECKPOINT_AUTHORITY.json"
                )
                engine = {
                    **dict(engine),
                    "checkpoint_authority": authority,
                    "checkpoint_authority_pending_final_operational_guard":
                        False,
                    "checkpoint_authoritative": True,
                    "final_operational_guard": final_guard_identity,
                }
            except BaseException as error:
                checkpoint = engine.get("checkpoint")
                quarantine = None
                if isinstance(checkpoint, Mapping):
                    quarantine = _quarantine_checkpoint(
                        phase_dir=out_dir,
                        checkpoint=checkpoint,
                        decision=KILL_EXECUTION,
                        predecessor={
                            "error_type": type(error).__name__,
                            "error_sha256": hashlib.sha256(
                                str(error).encode("utf-8")
                            ).hexdigest(),
                        },
                    )
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
                    "attempt_summary": attempt_ledger.summary(),
                    "passes": False,
                }
        elif (
            out_dir / "J2A1_DISTILLED_CHECKPOINT_AUTHORITY.json"
        ).exists():
            engine = {
                "decision": KILL_EXECUTION,
                "stage": "checkpoint_authority",
                "error_type": "UnexpectedCheckpointAuthority",
                "error_sha256": hashlib.sha256(
                    b"checkpoint authority exists for a non-READY engine"
                ).hexdigest(),
                "checkpoint_authoritative": False,
                "attempt_summary": attempt_ledger.summary(),
                "passes": False,
            }
    engine = {
        **dict(engine),
        "bounded_guard_accounting": {
            "guard_count": int(boundary_state["count"]),
            "by_kind": dict(boundary_state["by_kind"]),
            "latest_guard_sha256": boundary_state[
                "latest_guard_sha256"
            ],
        },
    }
    terminal_payload = _terminal_base(
        chain=chain,
        engine=engine,
        attempt_summary=attempt_ledger.summary(),
        reservation=reservation_identity,
        consumption=consumption_identity,
        ownership=ownership,
        execution_mode=execution_mode,
    )
    return _seal_terminal_and_retention(
        out_dir=out_dir,
        terminal_payload=terminal_payload,
        accountant=accountant,
    )


def _load_json_list(path: Path, *, name: str) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise J2A1ExecutionIntegrityError(
            f"{name} JSON cannot be loaded"
        ) from error
    if not isinstance(value, list):
        raise J2A1ExecutionIntegrityError(
            f"{name} JSON must contain a list"
        )
    return value


def canonical_execute_command(
    *,
    readiness_dir: Path,
    authorization_path: Path,
    out_dir: Path,
    jobs: int,
) -> str:
    return (
        "nice -n 10 env PYTHONPATH=. .venv/bin/python -m "
        "threes_rl.j2a1_distillation_fidelity_execution_surface "
        f"execute --readiness-dir {readiness_dir.resolve()} "
        f"--authorization {authorization_path.resolve()} "
        f"--out-dir {out_dir.resolve()} --jobs {int(jobs)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "J2A1 distillation/fidelity readiness and bounded "
            "training-only execution surface"
        )
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
    )
    audit = subparsers.add_parser("audit-zero-work")
    audit.add_argument("--output-dir", type=Path, default=READINESS_DIR)
    audit.add_argument(
        "--include-operational",
        action="store_true",
        help="Run read-only disk/process/service checks.",
    )

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
    command = str(args.subcommand)
    allowed = {
        "audit-zero-work",
        "write-test-evidence",
        "prepare-readiness",
        "seal-phase-lock",
        "open",
        "materialize",
        "execute",
    }
    if command not in allowed:
        raise J2A1ExecutionIntegrityError(
            f"Forbidden or unknown command: {command}"
        )
    if command == "audit-zero-work":
        return audit_zero_work(
            output_dir=args.output_dir,
            include_operational=bool(args.include_operational),
            allowed_files=(),
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
        if not all(isinstance(value, str) for value in deselections):
            raise J2A1ExecutionIntegrityError(
                "Deselections must be strings"
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
        "execution_mode": "scientific",
        "include_operational": True,
    }
    if command == "seal-phase-lock":
        return seal_phase_lock(
            **common,
            jobs=args.jobs,
        )
    if command == "open":
        if args.jobs != 1:
            raise J2A1ExecutionIntegrityError(
                "J2A1 jobs must equal one"
            )
        return open_phase(**common)
    if command == "materialize":
        if args.jobs != 1:
            raise J2A1ExecutionIntegrityError(
                "J2A1 jobs must equal one"
            )
        return materialize_phase(**common)
    if command == "execute":
        return execute_phase_from_artifacts(
            **common,
            command=canonical_execute_command(
                readiness_dir=args.readiness_dir,
                authorization_path=args.authorization,
                out_dir=args.out_dir,
                jobs=args.jobs,
            ),
            jobs=args.jobs,
        )
    raise J2A1ExecutionIntegrityError(
        f"Unreachable command routing state: {command}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = dispatch_cli(args)
    print(json.dumps(j2.json_native(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

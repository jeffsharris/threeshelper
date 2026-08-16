"""Outcome-free J2A1 V3 distillation recovery readiness.

This module deliberately has no teacher, simulator, model, optimizer, game,
or scientific execution imports. Retained V2 root artifacts are byte-hashed
only and are never deserialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


VERSION = "j2a1_distillation_fidelity_recovery_readiness_v3"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs" / "forensics"

CHARTER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J2A1_DISTILLATION_FIDELITY_V3_RECOVERY_READINESS_CHARTER.md"
)
RUNNER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "j2a1_distillation_fidelity_recovery_readiness_v3.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_rl_j2a1_distillation_fidelity_recovery_readiness_v3.py"
)

V2_SOURCE_PATHS = {
    REPO_ROOT
    / "threes_rl"
    / "J2A1_DISTILLATION_FIDELITY_EXECUTION_SURFACE_V2_CHARTER.md":
        "d9c5382d803c606c29415fc020fa7d63762dfcb053232d1ac904f21827d74dd4",
    REPO_ROOT
    / "threes_rl"
    / "j2a1_distillation_fidelity_execution_surface_v2.py":
        "044a67bf9b34b311787e3e7de246c4ce62a33f4f8ae47d211f6a76dd231a22f3",
    REPO_ROOT
    / "tests"
    / "test_rl_j2a1_distillation_fidelity_execution_surface_v2.py":
        "b211bfac0bb2e18c87dddcd72a0c8e7f1a0c3cbd76fee92572133aefa7abd95d",
}

V2_READINESS_DIR = (
    RUNS_ROOT
    / "j2a1_distillation_fidelity_execution_surface_readiness_v2"
)
V2_AUTHORIZATION_DIR = (
    RUNS_ROOT
    / "j2a1_distillation_fidelity_execution_authorization_v2"
)
V2_EXECUTION_DIR = (
    RUNS_ROOT / "j2a1_distillation_fidelity_execution_v2"
)
READINESS_DIR = (
    RUNS_ROOT / "j2a1_distillation_fidelity_recovery_readiness_v3"
)
FUTURE_EXECUTION_DIR = (
    RUNS_ROOT / "j2a1_distillation_fidelity_recovery_v3"
)

TEST_EVIDENCE_NAME = "J2A1_V3_RECOVERY_TEST_EVIDENCE.json"
INPUT_BINDINGS_NAME = "J2A1_V3_RECOVERY_INPUT_BINDINGS.json"
V2_INTEGRITY_NAME = "J2A1_V3_RECOVERY_V2_INTEGRITY_AUDIT.json"
AUTHORITY_NAME = "J2A1_V3_RECOVERY_AUTHORITY.json"
PROJECTION_NAME = "J2A1_V3_RECOVERY_WALL_PROJECTION.json"
SCHEMA_NAME = "J2A1_V3_RECOVERY_SCHEMA.json"
LOCK_NAME = "J2A1_V3_RECOVERY_READINESS_LOCK.json"
RESULT_NAME = "J2A1_V3_RECOVERY_READINESS_RESULT.json"
RETENTION_NAME = "J2A1_V3_RECOVERY_RETENTION.json"

READY = "READY_J2A1_V3_DISTILLATION_RECOVERY_PREFLIGHT"
HOLD = "HOLD_J2A1_V3_DISTILLATION_RECOVERY_PREFLIGHT"
KILL = "KILL_J2A1_V3_DISTILLATION_RECOVERY_PREFLIGHT_INTEGRITY"
V2_HOLD = "HOLD_J2A1_V2_DISTILLATION_OPERATIONAL"

ACTIVE_ROOTS = 14_336
ACTIVE_STREAMS = 63_488
COMPLETED_ROOTS = 3_048
REMAINING_ROOTS = ACTIVE_ROOTS - COMPLETED_ROOTS
ATTEMPT_RECORDS = 6_096
COLLECTORS = 8
RUNTIME_CAP_HOURS = 72.0
SAFETY_MULTIPLIER = 1.25
HARD_DISK_FLOOR_GIB = 100.0
TARGET_DISK_GIB = 120.0
EXPECTED_TOP_THREE = (263_670, 261_369, 258_561)

EXPECTED_WORKER_SECONDS = 259_763.24813699722
EXPECTED_EARLIEST_START = 1_785_246_546.567381
EXPECTED_LATEST_FINISH = 1_785_279_154.845043
EXPECTED_WALL_SECONDS = 32_608.277662038803

ZERO_WORK = {
    "recovery_phase_locks": 0,
    "recovery_markers": 0,
    "recovery_owners": 0,
    "recovery_stream_reservations": 0,
    "recovery_stream_consumptions": 0,
    "collector_process_loads": 0,
    "teacher_queries": 0,
    "root_body_deserializations": 0,
    "family_field_reads": 0,
    "teacher_action_labels": 0,
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

# path -> (file sha256, payload hash field or None, payload sha256 or None)
V2_BOUND_ARTIFACTS: dict[Path, tuple[str, str | None, str | None]] = {
    V2_READINESS_DIR / "J2A1_V2_EXECUTION_SURFACE_AUTHORITY_AUDIT.json": (
        "b48b4254b0272283542902710f86aa9d39aad136a523ef33753e610c2fb401e6",
        "authority_audit_payload_sha256",
        "ece84c85385d88c00507b9180d412a6892c98491541320b25afbe3528b104b29",
    ),
    V2_READINESS_DIR / "J2A1_V2_EXECUTION_SURFACE_INPUT_BINDINGS.json": (
        "0936290b049bcd24a580f13a24823a2e32e4997411ccee2670e2ee87df8678c1",
        "input_bindings_payload_sha256",
        "58c7f8c7ef2a59b1501a8b00c21271aa80582fbbbd7ab365445e8b6186fd8155",
    ),
    V2_READINESS_DIR / "J2A1_V2_EXECUTION_SURFACE_PROJECTION.json": (
        "19187c197541729227140cbc70fa54c3c7f824feccc2b0bf978ea7ba599ccd90",
        "projection_payload_sha256",
        "49fc21aa45d5c07147b33af4db5a78dd19646d6d708cc20a9c709ba41180a717",
    ),
    V2_READINESS_DIR / "J2A1_V2_EXECUTION_SURFACE_READINESS_LOCK.json": (
        "259df7e65be1e9cf73e93424cc40d4dadb6e27f87593abbcaa9d577e14d49702",
        "readiness_lock_payload_sha256",
        "fd2ed8ba9713d799420d2eafa264b4ff3185384811c68fbeed680a548d8fab31",
    ),
    V2_READINESS_DIR / "J2A1_V2_EXECUTION_SURFACE_READINESS_RESULT.json": (
        "c445d7ab1478b22b7bb7d74e06533566e519a4b400ced5db09555833fd3ad045",
        "readiness_result_payload_sha256",
        "567b5c58e89a66e0cc0040515f3646c0195263f40882586c05c8bb74844dcedd",
    ),
    V2_READINESS_DIR / "J2A1_V2_EXECUTION_SURFACE_RETENTION.json": (
        "e2b8bea7a7570b1268339d68cb063b11d8e84d2e3535f4a10b352b1c0590d068",
        "retention_payload_sha256",
        "48568dc9313de9dc71826c722999bcce4e2e5d7605534d2eaa9eb4493cb5675a",
    ),
    V2_READINESS_DIR / "J2A1_V2_EXECUTION_SURFACE_SCHEMA.json": (
        "f72f8ad12f3023b8f4215975c03b1d1e4b2c1157712b0e75bb57a5c61fa1ab7d",
        "execution_schema_payload_sha256",
        "792238fc213103a0bc150ee6b4cfef4fc924b953fbe32e191c6700792e8d87c2",
    ),
    V2_READINESS_DIR / "J2A1_V2_EXECUTION_SURFACE_TEST_EVIDENCE.json": (
        "8c3225ba74a66ea0a8817c3623a368b10430031e29afbb5784cf92ae374795c9",
        "test_evidence_payload_sha256",
        "a8197ed0a442f314d9da74c310c7add8f7b8b77172a52c95385a7648eb16748e",
    ),
    V2_AUTHORIZATION_DIR
    / "J2A1_V2_DISTILLATION_FIDELITY_EXECUTION_AUTHORIZATION.json": (
        "8787804d85e22d6720b6428feaa2d9122424c620c977becf1d106d88fc58e68c",
        "authorization_payload_sha256",
        "5e21b9ae764a023c567a9cfc8e495b6569322c253893986205e3ef98421b090a",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_PHASE_LOCK.json": (
        "7c5e7bbbaf326f05d1265b8d29e69b62595bdbbdf123fddf43ed9099309bc7bf",
        "phase_lock_payload_sha256",
        "affdb4b0597ce7286839044863f950eb34f8d3cb85f73a664870d86cffaa3d1f",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_EXECUTION_MARKER.json": (
        "e7adfe4de7bb750494cd8d19e8b2356d722f2f16796f367613b60ee82fed4179",
        "execution_marker_payload_sha256",
        "9430e2ff3d57221b0ab83f7b7788b9c618d6f6e7c10496da107105e836b87d07",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_ACTIVE_MANIFEST.json": (
        "7d91c3a1771654098c6e0fa80a0eb0bf4f3f4e1e7363fc7a3144932b454438d6",
        "active_manifest_payload_sha256",
        "d51d4b89d2c756c14cabde6ec5ebfef2f53d4965454ae27c613fecb6ef5ed6f0",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_STREAM_RESERVATION.json": (
        "ddd19bd0f1426e2ba5f92882d22b00038f7251934ab2a173cbc1c32e455fa9b4",
        "reservation_payload_sha256",
        "a6cf2235b847cd682de51113998db9e97bd2e92e1ac9f5fa6de307724b9803d5",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_STREAM_CONSUMPTION.json": (
        "b4cdbdd2dd9628e9cdd53c8d71699cde25d4b364a3441d5220255442d09c9e04",
        "consumption_payload_sha256",
        "1e4a92d0425e9f16d93efb1b8a6c4c2d9156afddd1bc1126f902241602aa8009",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_GENESIS.json": (
        "7804a4828ed30a1d0581d083b7118b418261641164bcc759ff30e6ddd9b7df8d",
        "genesis_payload_sha256",
        "5e3b730943cbae609b8543f61f1a1d39d53ace33a83ad6d0c84d1d17ec59f053",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_FIDELITY_TERMINAL_EVIDENCE.json": (
        "6a855bb18ca73cfef3dc465a3885e88901a317bbb08ce7624f2fb726438fdc7c",
        "terminal_evidence_payload_sha256",
        "304da0e20042485e5e913d65e99cce81c93b613ad98b68f491085c772f5eeb5d",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_FIDELITY_TERMINAL.json": (
        "13dd5c3a8eeb79d03149da0fa99a19aee3e6a657109e7fe4104a149d5d02ca6b",
        "terminal_payload_sha256",
        "c3ad1135034b33a6118d3239f88e94a760eaa9afe98a9ca589bd70e351ce91a3",
    ),
    V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_FIDELITY_RETENTION.json": (
        "93f3a5ac0e155b16af84fc06165cc4e23cbd4184b10b96cc77dc9870b1c315ac",
        "retention_payload_sha256",
        "d0f637b76345694ded4679afb1fe6740a55065a135263398db59a9c0df3cad74",
    ),
    V2_EXECUTION_DIR / "attempt_runtime_ledger.jsonl": (
        "06ec8ff51b35f722175f698edb3fadd3ae0d98c4b4e0bf77b8c4787ea24f5ca8",
        None,
        None,
    ),
    V2_EXECUTION_DIR / "teacher_root_completions.jsonl": (
        "7ca78f7090d6c3df1cbe3bb522a0cd12f7ba85e8c1559f138b6e840a31011acd",
        None,
        None,
    ),
    V2_EXECUTION_DIR / "ownership_ledger.jsonl": (
        "63bdfcafa0cbf5d80a9b486acf8cc7652b8bbef068c2a3bfcfa1c8a4fa9d19b2",
        None,
        None,
    ),
}

EXPECTED_MANIFEST_IDENTITIES = {
    "canonical_rows_sha256":
        "20af7bec1e4a9833d9af8bb73fb5dcce56380c7e3805baf7b842ab0e042c4fa3",
    "root_set_sha256":
        "85b3fe1de480d6a4b7a20ebf2fbb05a98e9b685474b674f59815a682dbc020b9",
    "ancestry_set_sha256":
        "da451be4cdc284e43a32286cc43443a3ef185ae07011cceeb58e872b388dc16a",
    "stream_set_sha256":
        "77c795f4f64fcb615a42140aa4bff23ecc3dc06eb2a9bb2671e2e6e9a70b81f3",
}

EXPECTED_RETENTION = {
    "file_count": 3_058,
    "retained_bytes": 1_782_523_714,
    "canonical_inventory_sha256":
        "54baf47f3a3c0ba72e60b4f74d9351ce8fffc42de2ded93a8593ae35c38e7642",
}


class J2A1V3IntegrityError(RuntimeError):
    """An immutable identity or outcome-free recovery contract failed."""


class J2A1V3OperationalHold(RuntimeError):
    """A mutable process, service, disk, or pace gate failed."""


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise J2A1V3IntegrityError("JSON payload contains nonfinite data")
        return float(value)
    raise J2A1V3IntegrityError(
        f"Value is not JSON-native: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_native(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = _json_native(dict(payload))
    if not isinstance(body, dict):
        raise J2A1V3IntegrityError("Payload is not an object")
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    observed = body.pop(field, None)
    return isinstance(observed, str) and observed == canonical_json_hash(body)


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise J2A1V3IntegrityError(
            f"Cannot load immutable JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise J2A1V3IntegrityError(f"JSON artifact is not an object: {path}")
    return payload


def _serialized_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_native(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    sealed = payload_with_hash(payload, field)
    serialized = _serialized_json(sealed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != serialized:
            raise J2A1V3IntegrityError(
                f"Immutable artifact already exists with other bytes: {path}"
            )
        observed = load_json(path)
        if observed != sealed or not verify_payload_hash(observed, field):
            raise J2A1V3IntegrityError(
                f"Existing immutable artifact is invalid: {path}"
            )
        return observed
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != serialized:
                raise J2A1V3IntegrityError(
                    f"Concurrent immutable write disagreed: {path}"
                )
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != serialized:
        raise J2A1V3IntegrityError(
            f"Immutable artifact bytes changed after write: {path}"
        )
    observed = load_json(path)
    if observed != sealed or not verify_payload_hash(observed, field):
        raise J2A1V3IntegrityError(
            f"Immutable artifact failed reload: {path}"
        )
    return observed


def artifact_identity(path: Path, payload_field: str) -> dict[str, Any]:
    payload = load_json(path)
    if not verify_payload_hash(payload, payload_field):
        raise J2A1V3IntegrityError(f"Payload identity changed: {path}")
    try:
        display_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display_path = str(path.resolve())
    return {
        "path": display_path,
        "bytes": int(path.stat().st_size),
        "file_sha256": sha256_path(path),
        "payload_field": payload_field,
        "payload_sha256": payload[payload_field],
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _validate_hex(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise J2A1V3IntegrityError(f"{name} is not a SHA-256 hex value")
    return value


def _strict_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    observed = set(payload)
    if observed != expected:
        raise J2A1V3IntegrityError(
            f"{context} keys changed: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _load_bound_artifact(
    path: Path,
    expected: tuple[str, str | None, str | None],
) -> dict[str, Any] | None:
    file_sha, field, payload_sha = expected
    if not path.is_file() or path.is_symlink():
        raise J2A1V3IntegrityError(f"Bound artifact is missing: {path}")
    if sha256_path(path) != file_sha:
        raise J2A1V3IntegrityError(f"Bound artifact bytes changed: {path}")
    if field is None:
        return None
    payload = load_json(path)
    if (
        not verify_payload_hash(payload, field)
        or payload.get(field) != payload_sha
    ):
        raise J2A1V3IntegrityError(f"Bound payload changed: {path}")
    return payload


def source_and_input_bindings() -> dict[str, Any]:
    source_checks = {
        str(path.relative_to(REPO_ROOT)): {
            "expected_sha256": expected,
            "observed_sha256": sha256_path(path),
            "passes": path.is_file() and sha256_path(path) == expected,
        }
        for path, expected in V2_SOURCE_PATHS.items()
    }
    artifacts: dict[str, Any] = {}
    for path, expected in V2_BOUND_ARTIFACTS.items():
        _load_bound_artifact(path, expected)
        artifacts[str(path.relative_to(REPO_ROOT))] = {
            "file_sha256": expected[0],
            "payload_field": expected[1],
            "payload_sha256": expected[2],
            "bytes": int(path.stat().st_size),
        }
    checks = {
        "v2_sources_exact": all(
            row["passes"] for row in source_checks.values()
        ),
        "v2_artifacts_exact": len(artifacts) == len(V2_BOUND_ARTIFACTS),
        "v2_execution_is_directory": V2_EXECUTION_DIR.is_dir(),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
    }
    if not all(checks.values()):
        raise J2A1V3IntegrityError("V2 source/input binding audit failed")
    return {
        "version": f"{VERSION}_input_bindings_v1",
        "v2_sources": source_checks,
        "v2_artifacts": artifacts,
        "future_execution_root": _display_path(FUTURE_EXECUTION_DIR),
        "checks": checks,
        "passes": True,
    }


ATTEMPT_START_KEYS = {
    "version",
    "sequence",
    "predecessor_record_sha256",
    "contract_sha256",
    "event",
    "unit_id",
    "unit_type",
    "attempt_id",
    "wall_started_at",
    "attempt_record_sha256",
}
ATTEMPT_FINISH_KEYS = {
    "version",
    "sequence",
    "predecessor_record_sha256",
    "contract_sha256",
    "event",
    "unit_id",
    "unit_type",
    "attempt_id",
    "start_sha256",
    "wall_ended_at",
    "charged_seconds",
    "output_identity",
    "attempt_record_sha256",
}
OUTPUT_IDENTITY_KEYS = {
    "root_id",
    "ancestry_id",
    "row_index",
    "stage",
    "path",
    "bytes",
    "file_sha256",
    "root_content_sha256",
}
COMPLETION_KEYS = {
    "version",
    "sequence",
    "predecessor_record_sha256",
    "contract_sha256",
    "kind",
    "root_id",
    "ancestry_id",
    "row_index",
    "stage",
    "relative_path",
    "file_sha256",
    "content_sha256",
    "recovered_orphan",
    "completion_record_sha256",
}
MANIFEST_ROW_KEYS = {
    "row_index",
    "root_id",
    "ancestry_id",
    "stage",
    "streams",
    "reserved",
    "consumed",
    "content_opened",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise J2A1V3IntegrityError(
                        f"JSONL line lacks terminator: {path}:{line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise J2A1V3IntegrityError(
                        f"Malformed JSONL: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise J2A1V3IntegrityError(
                        f"JSONL record is not an object: {path}:{line_number}"
                    )
                records.append(record)
    except OSError as error:
        raise J2A1V3IntegrityError(f"Cannot read ledger: {path}") from error
    return records


def audit_attempt_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != ATTEMPT_RECORDS:
        raise J2A1V3IntegrityError("Attempt record count changed")
    predecessor: str | None = None
    contract: str | None = None
    opens: dict[str, Mapping[str, Any]] = {}
    completed_units: set[str] = set()
    attempts_by_unit: Counter[str] = Counter()
    earliest = math.inf
    latest = -math.inf
    worker_seconds = 0.0
    starts = finishes = abandoned = 0
    outputs: dict[str, dict[str, Any]] = {}
    for sequence, source_record in enumerate(records):
        record = dict(source_record)
        event = record.get("event")
        expected_keys = (
            ATTEMPT_START_KEYS
            if event == "started"
            else ATTEMPT_FINISH_KEYS
            if event == "finished"
            else set()
        )
        if not expected_keys:
            raise J2A1V3IntegrityError(
                "V2 attempt ledger contains a non-finished event"
            )
        _strict_keys(record, expected_keys, context="attempt record")
        if not verify_payload_hash(record, "attempt_record_sha256"):
            raise J2A1V3IntegrityError("Attempt record hash changed")
        if (
            record["sequence"] != sequence
            or record["predecessor_record_sha256"] != predecessor
        ):
            raise J2A1V3IntegrityError("Attempt hash chain changed")
        current_contract = _validate_hex(
            record["contract_sha256"],
            name="attempt contract",
        )
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise J2A1V3IntegrityError("Attempt contract changed")
        unit_id = record["unit_id"]
        if not isinstance(unit_id, str):
            raise J2A1V3IntegrityError("Attempt unit is malformed")
        unit_parts = unit_id.split("|")
        if (
            len(unit_parts) != 3
            or unit_parts[0] != "teacher_root"
            or unit_parts[1]
            not in {
                "teacher_behavior_cloning",
                "distillation_validation",
            }
        ):
            raise J2A1V3IntegrityError("Attempt unit schema changed")
        root_id = _validate_hex(
            unit_parts[2],
            name="attempt root",
        )
        if record["unit_type"] != "teacher_root":
            raise J2A1V3IntegrityError("Attempt unit type changed")
        attempt_id = record["attempt_id"]
        if not isinstance(attempt_id, str):
            raise J2A1V3IntegrityError("Attempt identity is malformed")
        if event == "started":
            if (
                attempt_id in opens
                or root_id in completed_units
                or attempt_id != f"{unit_id}|attempt=0"
            ):
                raise J2A1V3IntegrityError(
                    "Attempt ledger contains a hidden retry"
                )
            attempts_by_unit[unit_id] += 1
            if attempts_by_unit[unit_id] != 1:
                raise J2A1V3IntegrityError(
                    "Attempt ledger repeats a root"
                )
            started = float(record["wall_started_at"])
            if not math.isfinite(started):
                raise J2A1V3IntegrityError("Attempt start is nonfinite")
            earliest = min(earliest, started)
            opens[attempt_id] = record
            starts += 1
        else:
            opened = opens.pop(attempt_id, None)
            if (
                opened is None
                or record["start_sha256"]
                != opened["attempt_record_sha256"]
                or unit_id != opened["unit_id"]
            ):
                raise J2A1V3IntegrityError(
                    "Attempt finish is not paired to its start"
                )
            ended = float(record["wall_ended_at"])
            charged = float(record["charged_seconds"])
            expected_charge = ended - float(opened["wall_started_at"])
            if (
                not math.isfinite(ended)
                or not math.isfinite(charged)
                or ended < float(opened["wall_started_at"])
                or not math.isclose(
                    charged,
                    expected_charge,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise J2A1V3IntegrityError(
                    "Attempt finish time accounting changed"
                )
            output = record["output_identity"]
            if not isinstance(output, Mapping):
                raise J2A1V3IntegrityError(
                    "Attempt output identity is malformed"
                )
            output = dict(output)
            _strict_keys(
                output,
                OUTPUT_IDENTITY_KEYS,
                context="attempt output identity",
            )
            if output["root_id"] != root_id:
                raise J2A1V3IntegrityError(
                    "Attempt output changed its root"
                )
            outputs[root_id] = output
            completed_units.add(root_id)
            worker_seconds += charged
            latest = max(latest, ended)
            finishes += 1
        predecessor = record["attempt_record_sha256"]
    if opens:
        raise J2A1V3IntegrityError("Attempt ledger has an open attempt")
    wall_seconds = latest - earliest
    checks = {
        "records_exact": len(records) == ATTEMPT_RECORDS,
        "starts_exact": starts == COMPLETED_ROOTS,
        "finishes_exact": finishes == COMPLETED_ROOTS,
        "abandoned_zero": abandoned == 0,
        "completed_units_exact": len(completed_units) == COMPLETED_ROOTS,
        "worker_seconds_exact": math.isclose(
            worker_seconds,
            EXPECTED_WORKER_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "earliest_start_exact": earliest == EXPECTED_EARLIEST_START,
        "latest_finish_exact": latest == EXPECTED_LATEST_FINISH,
        "wall_seconds_exact": wall_seconds == EXPECTED_WALL_SECONDS,
    }
    if not all(checks.values()):
        raise J2A1V3IntegrityError("Attempt ledger facts changed")
    return {
        "contract_sha256": contract,
        "head_sha256": predecessor,
        "record_count": len(records),
        "attempts_started": starts,
        "attempts_finished": finishes,
        "attempts_abandoned": abandoned,
        "open_attempts": 0,
        "completed_unit_count": len(completed_units),
        "aggregate_worker_seconds_descriptive": worker_seconds,
        "earliest_start": earliest,
        "latest_finish": latest,
        "top_level_wall_span_seconds": wall_seconds,
        "outputs_by_root": outputs,
        "checks": checks,
        "passes": True,
    }


def audit_completion_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != COMPLETED_ROOTS:
        raise J2A1V3IntegrityError("Completion record count changed")
    predecessor: str | None = None
    contract: str | None = None
    by_root: dict[str, dict[str, Any]] = {}
    ancestries: set[str] = set()
    paths: set[str] = set()
    file_hashes: set[str] = set()
    for sequence, source_record in enumerate(records):
        record = dict(source_record)
        _strict_keys(record, COMPLETION_KEYS, context="completion record")
        if not verify_payload_hash(record, "completion_record_sha256"):
            raise J2A1V3IntegrityError("Completion record hash changed")
        if (
            record["sequence"] != sequence
            or record["predecessor_record_sha256"] != predecessor
            or record["kind"] != "teacher_root"
            or record["recovered_orphan"] is not False
        ):
            raise J2A1V3IntegrityError("Completion chain changed")
        current_contract = _validate_hex(
            record["contract_sha256"],
            name="completion contract",
        )
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise J2A1V3IntegrityError("Completion contract changed")
        root_id = _validate_hex(record["root_id"], name="completion root")
        ancestry_id = _validate_hex(
            record["ancestry_id"],
            name="completion ancestry",
        )
        file_sha = _validate_hex(
            record["file_sha256"],
            name="completion file",
        )
        _validate_hex(
            record["content_sha256"],
            name="completion content identity",
        )
        relative_path = record["relative_path"]
        if (
            not isinstance(relative_path, str)
            or not relative_path.startswith("teacher_roots/")
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise J2A1V3IntegrityError(
                "Completion path leaves the root namespace"
            )
        if (
            root_id in by_root
            or ancestry_id in ancestries
            or relative_path in paths
            or file_sha in file_hashes
        ):
            raise J2A1V3IntegrityError(
                "Completion metadata repeats an identity"
            )
        by_root[root_id] = record
        ancestries.add(ancestry_id)
        paths.add(relative_path)
        file_hashes.add(file_sha)
        predecessor = record["completion_record_sha256"]
    return {
        "contract_sha256": contract,
        "head_sha256": predecessor,
        "completed": len(records),
        "by_root": by_root,
        "unique_ancestries": len(ancestries),
        "unique_paths": len(paths),
        "unique_file_hashes": len(file_hashes),
        "passes": True,
    }


def _manifest_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_rows = manifest.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != ACTIVE_ROOTS:
        raise J2A1V3IntegrityError("Active manifest row count changed")
    rows: list[dict[str, Any]] = []
    roots: set[str] = set()
    ancestries: set[str] = set()
    streams: set[int] = set()
    stage_row_keys: set[tuple[str, int]] = set()
    stage_counts: Counter[str] = Counter()
    for authority_index, source_row in enumerate(source_rows):
        if not isinstance(source_row, Mapping):
            raise J2A1V3IntegrityError("Active manifest row is malformed")
        row = dict(source_row)
        _strict_keys(row, MANIFEST_ROW_KEYS, context="manifest row")
        root_id = _validate_hex(row["root_id"], name="manifest root")
        ancestry_id = _validate_hex(
            row["ancestry_id"],
            name="manifest ancestry",
        )
        if root_id in roots or ancestry_id in ancestries:
            raise J2A1V3IntegrityError(
                "Manifest repeats a root or ancestry"
            )
        if row["stage"] not in {
            "teacher_behavior_cloning",
            "distillation_validation",
        }:
            raise J2A1V3IntegrityError("Manifest stage changed")
        if (
            type(row["row_index"]) is not int
            or row["row_index"] < 0
            or (row["stage"], row["row_index"]) in stage_row_keys
            or row["row_index"] != stage_counts[row["stage"]]
        ):
            raise J2A1V3IntegrityError(
                "Manifest stage-local row order changed"
            )
        stream_map = row["streams"]
        if not isinstance(stream_map, Mapping):
            raise J2A1V3IntegrityError("Manifest stream map is malformed")
        for stream_id in stream_map.values():
            if type(stream_id) is not int or stream_id in streams:
                raise J2A1V3IntegrityError(
                    "Manifest stream identity is invalid or duplicated"
                )
            streams.add(stream_id)
        if (
            row["reserved"] is not False
            or row["consumed"] is not False
            or row["content_opened"] is not False
        ):
            raise J2A1V3IntegrityError(
                "Manifest pre-science state changed"
            )
        roots.add(root_id)
        ancestries.add(ancestry_id)
        stage_row_keys.add((row["stage"], row["row_index"]))
        stage_counts[row["stage"]] += 1
        rows.append(row)
    if stage_counts != Counter(
        {
            "teacher_behavior_cloning": 8_192,
            "distillation_validation": 6_144,
        }
    ):
        raise J2A1V3IntegrityError("Manifest stage counts changed")
    computed = {
        "canonical_rows_sha256": canonical_json_hash(rows),
        "root_set_sha256": canonical_json_hash(sorted(roots)),
        "ancestry_set_sha256": canonical_json_hash(sorted(ancestries)),
        "stream_set_sha256": canonical_json_hash(sorted(streams)),
    }
    if (
        len(streams) != ACTIVE_STREAMS
        or computed != EXPECTED_MANIFEST_IDENTITIES
        or any(manifest.get(key) != value for key, value in computed.items())
    ):
        raise J2A1V3IntegrityError("Active manifest identity changed")
    return rows


def hash_only_retention_audit(
    retention: Mapping[str, Any],
    *,
    execution_dir: Path = V2_EXECUTION_DIR,
    hash_file: Callable[[str | Path], str] = sha256_path,
) -> dict[str, Any]:
    files = retention.get("files")
    if not isinstance(files, list):
        raise J2A1V3IntegrityError("V2 retention inventory is malformed")
    if (
        retention.get("file_count") != EXPECTED_RETENTION["file_count"]
        or retention.get("retained_bytes")
        != EXPECTED_RETENTION["retained_bytes"]
        or retention.get("canonical_inventory_sha256")
        != EXPECTED_RETENTION["canonical_inventory_sha256"]
        or canonical_json_hash(files)
        != EXPECTED_RETENTION["canonical_inventory_sha256"]
        or retention.get("passes") is not True
    ):
        raise J2A1V3IntegrityError("V2 retention inventory changed")
    root_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    total_root_bytes = 0
    for source_row in files:
        if not isinstance(source_row, Mapping):
            raise J2A1V3IntegrityError("Retention row is malformed")
        row = dict(source_row)
        _strict_keys(
            row,
            {"path", "bytes", "file_sha256"},
            context="retention row",
        )
        relative_path = row["path"]
        if not isinstance(relative_path, str):
            raise J2A1V3IntegrityError("Retention path is malformed")
        if not relative_path.startswith("teacher_roots/"):
            continue
        path = execution_dir / relative_path
        if (
            relative_path in seen_paths
            or row["file_sha256"] in seen_hashes
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or hash_file(path) != row["file_sha256"]
        ):
            raise J2A1V3IntegrityError(
                "Retained teacher-root byte identity changed"
            )
        seen_paths.add(relative_path)
        seen_hashes.add(row["file_sha256"])
        total_root_bytes += int(row["bytes"])
        root_rows.append(row)
    if len(root_rows) != COMPLETED_ROOTS:
        raise J2A1V3IntegrityError("Retained teacher-root count changed")
    return {
        "root_blob_count": len(root_rows),
        "root_blob_bytes": total_root_bytes,
        "unique_paths": len(seen_paths),
        "unique_file_hashes": len(seen_hashes),
        "root_blob_inventory_sha256": canonical_json_hash(root_rows),
        "root_rows": root_rows,
        "body_access": "streaming SHA-256 and byte count only",
        "root_body_deserializations": 0,
        "passes": True,
    }


def derive_recovery_authority(
    rows: Sequence[Mapping[str, Any]],
    completions_by_root: Mapping[str, Mapping[str, Any]],
    attempt_outputs_by_root: Mapping[str, Mapping[str, Any]],
    retention_root_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_root = {str(row["root_id"]): dict(row) for row in rows}
    if len(by_root) != len(rows):
        raise J2A1V3IntegrityError("Authority rows repeat a root")
    completed_roots = set(completions_by_root)
    if (
        completed_roots != set(attempt_outputs_by_root)
        or not completed_roots <= set(by_root)
        or len(completed_roots) != COMPLETED_ROOTS
    ):
        raise J2A1V3IntegrityError(
            "Completion/attempt/manifest root sets disagree"
        )
    retention_by_path = {
        str(row["path"]): dict(row) for row in retention_root_rows
    }
    completed_refs: list[dict[str, Any]] = []
    unfinished_rows: list[dict[str, Any]] = []
    for authority_index, row in enumerate(rows):
        root_id = str(row["root_id"])
        if root_id not in completed_roots:
            unfinished_rows.append(dict(row))
            continue
        completion = dict(completions_by_root[root_id])
        output = dict(attempt_outputs_by_root[root_id])
        retained = retention_by_path.get(str(completion["relative_path"]))
        output_path = output["path"]
        expected_relative_path = str(completion["relative_path"])
        output_path_matches = output_path == expected_relative_path
        if isinstance(output_path, str) and Path(output_path).is_absolute():
            output_path_matches = (
                Path(output_path).resolve()
                == (V2_EXECUTION_DIR / expected_relative_path).resolve()
            )
        checks = {
            "row_index": completion["row_index"] == row["row_index"],
            "ancestry": completion["ancestry_id"] == row["ancestry_id"],
            "stage": completion["stage"] == row["stage"],
            "retained": retained is not None,
            "retained_file": retained is not None
            and retained["file_sha256"] == completion["file_sha256"],
            "output_root": output["root_id"] == root_id,
            "output_ancestry": output["ancestry_id"] == row["ancestry_id"],
            "output_row": output["row_index"] == row["row_index"],
            "output_stage": output["stage"] == row["stage"],
            "output_path": output_path_matches,
            "output_file": output["file_sha256"]
            == completion["file_sha256"],
            "output_content": output["root_content_sha256"]
            == completion["content_sha256"],
            "output_bytes": retained is not None
            and output["bytes"] == retained["bytes"],
        }
        if not all(checks.values()):
            raise J2A1V3IntegrityError(
                "Completed root cross-binding changed"
            )
        completed_refs.append(
            {
                "authority_index": authority_index,
                "row_index": row["row_index"],
                "root_id": root_id,
                "ancestry_id": row["ancestry_id"],
                "stage": row["stage"],
                "relative_path": completion["relative_path"],
                "bytes": retained["bytes"],
                "file_sha256": completion["file_sha256"],
                "content_sha256": completion["content_sha256"],
                "completion_record_sha256":
                    completion["completion_record_sha256"],
                "attempt_output_identity_sha256":
                    canonical_json_hash(output),
            }
        )
    if (
        len(completed_refs) != COMPLETED_ROOTS
        or len(unfinished_rows) != REMAINING_ROOTS
        or {row["root_id"] for row in completed_refs}
        & {row["root_id"] for row in unfinished_rows}
    ):
        raise J2A1V3IntegrityError("Recovery set difference is inconsistent")
    unfinished_root_ids = {
        str(row["root_id"]) for row in unfinished_rows
    }
    authority_index_by_root = {
        str(row["root_id"]): index for index, row in enumerate(rows)
    }
    reconstructed = sorted(
        [
            {
                "authority_index": row["authority_index"],
                "root_id": row["root_id"],
            }
            for row in completed_refs
        ]
        + [
            {
                "authority_index": authority_index_by_root[
                    str(root_id)
                ],
                "root_id": root_id,
            }
            for root_id in unfinished_root_ids
        ],
        key=lambda row: row["authority_index"],
    )
    expected = [
        {"authority_index": index, "root_id": row["root_id"]}
        for index, row in enumerate(rows)
    ]
    if reconstructed != expected:
        raise J2A1V3IntegrityError(
            "Recovery partition does not reconstruct authority"
        )
    return {
        "version": f"{VERSION}_recovery_authority_v1",
        "v2_manifest_identity": EXPECTED_MANIFEST_IDENTITIES,
        "v2_stream_consumption": {
            "file_sha256":
                V2_BOUND_ARTIFACTS[
                    V2_EXECUTION_DIR
                    / "J2A1_V2_DISTILLATION_STREAM_CONSUMPTION.json"
                ][0],
            "payload_sha256":
                V2_BOUND_ARTIFACTS[
                    V2_EXECUTION_DIR
                    / "J2A1_V2_DISTILLATION_STREAM_CONSUMPTION.json"
                ][2],
        },
        "total_rows": len(rows),
        "total_streams": ACTIVE_STREAMS,
        "completed_rows": len(completed_refs),
        "unfinished_rows_count": len(unfinished_rows),
        "completed_refs_sha256": canonical_json_hash(completed_refs),
        "unfinished_rows_sha256": canonical_json_hash(unfinished_rows),
        "completed_root_set_sha256": canonical_json_hash(
            sorted(row["root_id"] for row in completed_refs)
        ),
        "unfinished_root_set_sha256": canonical_json_hash(
            sorted(row["root_id"] for row in unfinished_rows)
        ),
        "completed_refs": completed_refs,
        "unfinished_rows": unfinished_rows,
        "replacement_roots": 0,
        "filtered_roots": 0,
        "duplicate_stream_consumptions": 0,
        "stage_b_sealed_until_total_completions": ACTIVE_ROOTS,
        "scientific_content_opened": 0,
        "passes": True,
    }


def wall_clock_projection(
    *,
    completed_roots: int,
    total_roots: int,
    earliest_start: float,
    latest_finish: float,
    aggregate_worker_seconds: float,
) -> dict[str, Any]:
    if (
        type(completed_roots) is not int
        or type(total_roots) is not int
        or completed_roots <= 0
        or total_roots <= completed_roots
        or not all(
            math.isfinite(value)
            for value in (
                earliest_start,
                latest_finish,
                aggregate_worker_seconds,
            )
        )
        or latest_finish <= earliest_start
        or aggregate_worker_seconds < latest_finish - earliest_start
    ):
        raise J2A1V3IntegrityError(
            "Wall-clock projection inputs are invalid"
        )
    wall_seconds = latest_finish - earliest_start
    wall_hours = wall_seconds / 3600.0
    remaining = total_roots - completed_roots
    rate = completed_roots / wall_hours
    remaining_hours = remaining / rate
    projected_total = wall_hours + remaining_hours
    conservative_total = wall_hours + SAFETY_MULTIPLIER * remaining_hours
    remaining_cap = RUNTIME_CAP_HOURS - wall_hours
    required_rate = remaining / remaining_cap
    checks = {
        "wall_span_positive": wall_seconds > 0.0,
        "worker_seconds_descriptive_only":
            aggregate_worker_seconds > wall_seconds,
        "completed_exact": completed_roots == COMPLETED_ROOTS,
        "remaining_exact": remaining == REMAINING_ROOTS,
        "projected_total_below_cap": projected_total < RUNTIME_CAP_HOURS,
        "conservative_total_below_cap":
            conservative_total < RUNTIME_CAP_HOURS,
        "observed_rate_above_required": rate > required_rate,
    }
    return {
        "version": f"{VERSION}_wall_clock_projection_v1",
        "runtime_cap_basis": "top-level elapsed wall time only",
        "runtime_cap_hours": RUNTIME_CAP_HOURS,
        "aggregate_worker_seconds_descriptive": aggregate_worker_seconds,
        "aggregate_worker_hours_descriptive":
            aggregate_worker_seconds / 3600.0,
        "observed_wall_seconds": wall_seconds,
        "observed_wall_hours": wall_hours,
        "completed_roots": completed_roots,
        "remaining_roots": remaining,
        "observed_roots_per_wall_hour": rate,
        "projected_remaining_wall_hours": remaining_hours,
        "projected_total_stage_a_wall_hours": projected_total,
        "safety_multiplier_on_remaining": SAFETY_MULTIPLIER,
        "conservative_total_stage_a_wall_hours": conservative_total,
        "remaining_cap_hours": remaining_cap,
        "minimum_required_remaining_roots_per_wall_hour": required_rate,
        "adjudication_and_dead_process_downtime_charged": False,
        "checks": checks,
        "passes": all(checks.values()),
    }


def v2_integrity_and_authority_audit(
    *,
    hash_file: Callable[[str | Path], str] = sha256_path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load_bound_artifact(
        V2_EXECUTION_DIR / "J2A1_V2_DISTILLATION_ACTIVE_MANIFEST.json",
        V2_BOUND_ARTIFACTS[
            V2_EXECUTION_DIR
            / "J2A1_V2_DISTILLATION_ACTIVE_MANIFEST.json"
        ],
    )
    terminal = _load_bound_artifact(
        V2_EXECUTION_DIR
        / "J2A1_V2_DISTILLATION_FIDELITY_TERMINAL.json",
        V2_BOUND_ARTIFACTS[
            V2_EXECUTION_DIR
            / "J2A1_V2_DISTILLATION_FIDELITY_TERMINAL.json"
        ],
    )
    retention = _load_bound_artifact(
        V2_EXECUTION_DIR
        / "J2A1_V2_DISTILLATION_FIDELITY_RETENTION.json",
        V2_BOUND_ARTIFACTS[
            V2_EXECUTION_DIR
            / "J2A1_V2_DISTILLATION_FIDELITY_RETENTION.json"
        ],
    )
    if manifest is None or terminal is None or retention is None:
        raise J2A1V3IntegrityError("V2 JSON bindings are unavailable")
    rows = _manifest_rows(manifest)
    attempts = audit_attempt_records(
        _read_jsonl(V2_EXECUTION_DIR / "attempt_runtime_ledger.jsonl")
    )
    completions = audit_completion_records(
        _read_jsonl(V2_EXECUTION_DIR / "teacher_root_completions.jsonl")
    )
    retention_audit = hash_only_retention_audit(
        retention,
        hash_file=hash_file,
    )
    if attempts["contract_sha256"] != completions["contract_sha256"]:
        raise J2A1V3IntegrityError(
            "Attempt and completion contracts disagree"
        )
    attempt_summary = terminal.get("attempt_summary")
    if not isinstance(attempt_summary, Mapping):
        raise J2A1V3IntegrityError(
            "V2 terminal attempt summary is missing"
        )
    terminal_checks = {
        "decision": terminal.get("decision") == V2_HOLD,
        "scientific_authority": terminal.get("scientific_authority") is True,
        "checkpoint_non_authoritative":
            terminal.get("checkpoint_authoritative") is False,
        "ppo_closed": terminal.get("ppo_execution_authorized") is False,
        "development_closed":
            terminal.get("development_authorized") is False,
        "confirmation_closed":
            terminal.get("confirmation_authorized") is False,
        "attempt_records": attempt_summary.get("record_count")
        == ATTEMPT_RECORDS,
        "attempt_started": attempt_summary.get("attempts_started")
        == COMPLETED_ROOTS,
        "attempt_finished": attempt_summary.get("attempts_finished")
        == COMPLETED_ROOTS,
        "attempt_abandoned":
            attempt_summary.get("attempts_abandoned") == 0,
        "attempt_head": attempt_summary.get("head_sha256")
        == attempts["head_sha256"],
        "worker_seconds": attempt_summary.get("active_seconds")
        == attempts["aggregate_worker_seconds_descriptive"],
    }
    if not all(terminal_checks.values()):
        raise J2A1V3IntegrityError("V2 terminal boundary changed")
    authority = derive_recovery_authority(
        rows,
        completions["by_root"],
        attempts["outputs_by_root"],
        retention_audit["root_rows"],
    )
    projection = wall_clock_projection(
        completed_roots=COMPLETED_ROOTS,
        total_roots=ACTIVE_ROOTS,
        earliest_start=attempts["earliest_start"],
        latest_finish=attempts["latest_finish"],
        aggregate_worker_seconds=attempts[
            "aggregate_worker_seconds_descriptive"
        ],
    )
    integrity = {
        "version": f"{VERSION}_v2_integrity_audit_v1",
        "v2_terminal_decision": V2_HOLD,
        "v2_terminal_checks": terminal_checks,
        "attempt_ledger": {
            key: value
            for key, value in attempts.items()
            if key not in {"outputs_by_root", "checks"}
        },
        "attempt_checks": attempts["checks"],
        "completion_ledger": {
            key: value
            for key, value in completions.items()
            if key != "by_root"
        },
        "retention": {
            key: value
            for key, value in retention_audit.items()
            if key != "root_rows"
        },
        "manifest": {
            **EXPECTED_MANIFEST_IDENTITIES,
            "row_count": len(rows),
            "stream_count": ACTIVE_STREAMS,
        },
        "scientific_body_deserializations": 0,
        "family_fields_read": 0,
        "labels_read": 0,
        "outcomes_read": 0,
        "passes": True,
    }
    return integrity, authority, projection


def operational_audit(
    *,
    output_dir: Path,
    include_services: bool = True,
) -> dict[str, Any]:
    if include_services:
        from threes_rl import j1_joint_policy_value as j1

        free_gib = float(j1.free_disk_gib())
        process = j1.heavy_process_audit()
        services = j1.service_audit()
        nice = int(os.getpriority(os.PRIO_PROCESS, 0))
        top_three = services.get("dashboard", {}).get("top_three")
        top_three_native = (
            tuple(top_three)
            if type(top_three) in {list, tuple}
            and len(top_three) == 3
            and all(type(value) is int for value in top_three)
            else None
        )
        checks = {
            "nice_at_least_10": nice >= 10,
            "one_heavy_job": process.get("passes") is True,
            "free_disk_hard_floor": free_gib > HARD_DISK_FLOOR_GIB,
            "free_disk_target": free_gib > TARGET_DISK_GIB,
            "services_healthy": services.get("passes") is True,
            "top_three_exact": top_three_native == EXPECTED_TOP_THREE,
            "human_sessions_opaque": services.get(
                "recorder",
                {},
            ).get("active_session_content_read") is False,
        }
    else:
        free_gib = 140.0
        process = {"passes": True, "fixture": True}
        services = {
            "passes": True,
            "dashboard": {"top_three": list(EXPECTED_TOP_THREE)},
            "recorder": {"active_session_content_read": False},
        }
        nice = 10
        checks = {
            "nice_at_least_10": True,
            "one_heavy_job": True,
            "free_disk_hard_floor": True,
            "free_disk_target": True,
            "services_healthy": True,
            "top_three_exact": True,
            "human_sessions_opaque": True,
        }
    checks.update(
        {
            "future_execution_absent":
                not FUTURE_EXECUTION_DIR.exists(),
            "zero_work": all(value == 0 for value in ZERO_WORK.values()),
        }
    )
    return {
        "version": f"{VERSION}_operational_audit_v1",
        "output_dir": str(output_dir),
        "nice": nice,
        "free_disk_gib": free_gib,
        "process": process,
        "services": services,
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "zero_work": dict(ZERO_WORK),
        "checks": checks,
        "passes": all(checks.values()),
    }


def schema_payload() -> dict[str, Any]:
    return {
        "version": f"{VERSION}_schema_v1",
        "public_commands": [
            "audit-zero-work",
            "write-test-evidence",
            "prepare",
        ],
        "v2_counts": {
            "authority_roots": ACTIVE_ROOTS,
            "streams": ACTIVE_STREAMS,
            "completed_roots": COMPLETED_ROOTS,
            "remaining_roots": REMAINING_ROOTS,
            "attempt_records": ATTEMPT_RECORDS,
            "collectors": COLLECTORS,
        },
        "wall_cap": {
            "hours": RUNTIME_CAP_HOURS,
            "basis": "top-level elapsed wall time only",
            "aggregate_worker_seconds": "descriptive only",
            "safety_multiplier_on_projected_remaining":
                SAFETY_MULTIPLIER,
        },
        "read_boundary": {
            "json_governance_and_metadata_ledgers": True,
            "root_blob_access": "streaming byte hash and byte count only",
            "root_blob_deserialization": False,
            "family_fields": False,
            "labels": False,
            "actions": False,
            "scores": False,
            "outcomes": False,
            "teacher_import": False,
        },
        "future_execution": {
            "authorized": False,
            "one_top_level_heavy_job": True,
            "fixed_collectors": COLLECTORS,
            "nice_at_least": 10,
            "stage_b_completion_gate": ACTIVE_ROOTS,
            "duplicate_reservation_or_consumption": False,
        },
        "zero_work": dict(ZERO_WORK),
    }


def readiness_paths(readiness_dir: Path) -> dict[str, Path]:
    return {
        "test_evidence": readiness_dir / TEST_EVIDENCE_NAME,
        "input_bindings": readiness_dir / INPUT_BINDINGS_NAME,
        "v2_integrity": readiness_dir / V2_INTEGRITY_NAME,
        "authority": readiness_dir / AUTHORITY_NAME,
        "projection": readiness_dir / PROJECTION_NAME,
        "schema": readiness_dir / SCHEMA_NAME,
        "lock": readiness_dir / LOCK_NAME,
        "result": readiness_dir / RESULT_NAME,
        "retention": readiness_dir / RETENTION_NAME,
    }


READINESS_FIELDS = {
    "test_evidence": "test_evidence_payload_sha256",
    "input_bindings": "input_bindings_payload_sha256",
    "v2_integrity": "v2_integrity_payload_sha256",
    "authority": "recovery_authority_payload_sha256",
    "projection": "wall_projection_payload_sha256",
    "schema": "recovery_schema_payload_sha256",
    "lock": "readiness_lock_payload_sha256",
    "result": "readiness_result_payload_sha256",
    "retention": "retention_payload_sha256",
}


def _source_identities() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256_path(path)
        for path in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)
    }


def test_evidence_payload(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        if set(record) != {
            "command",
            "passed",
            "failed",
            "deselected",
            "note",
        }:
            raise J2A1V3IntegrityError(
                "Test evidence record schema changed"
            )
        if (
            not isinstance(record["command"], str)
            or type(record["passed"]) is not int
            or type(record["failed"]) is not int
            or type(record["deselected"]) is not int
            or not isinstance(record["note"], str)
            or record["passed"] < 0
            or record["failed"] != 0
            or record["deselected"] < 0
        ):
            raise J2A1V3IntegrityError("Test evidence record is invalid")
        normalized.append(dict(record))
    if not normalized:
        raise J2A1V3IntegrityError("Test evidence cannot be empty")
    return {
        "version": f"{VERSION}_test_evidence_v1",
        "source_identities": _source_identities(),
        "records": normalized,
        "total_passed": sum(record["passed"] for record in normalized),
        "total_failed": sum(record["failed"] for record in normalized),
        "total_deselected": sum(
            record["deselected"] for record in normalized
        ),
        "zero_work": dict(ZERO_WORK),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "passes": all(record["failed"] == 0 for record in normalized)
        and not FUTURE_EXECUTION_DIR.exists()
        and all(value == 0 for value in ZERO_WORK.values()),
    }


def write_test_evidence(
    readiness_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = test_evidence_payload(records)
    return write_immutable_json(
        readiness_dir / TEST_EVIDENCE_NAME,
        payload,
        field=READINESS_FIELDS["test_evidence"],
    )


def _load_test_evidence(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if (
        not verify_payload_hash(
            payload,
            READINESS_FIELDS["test_evidence"],
        )
        or payload.get("source_identities") != _source_identities()
        or payload.get("passes") is not True
        or payload.get("total_failed") != 0
    ):
        raise J2A1V3IntegrityError("Test evidence changed")
    return payload


def _retention_payload(readiness_dir: Path) -> dict[str, Any]:
    paths = readiness_paths(readiness_dir)
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
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "no_cleanup_performed": True,
        "zero_work": dict(ZERO_WORK),
        "passes": len(inventory) == 8
        and not FUTURE_EXECUTION_DIR.exists()
        and all(value == 0 for value in ZERO_WORK.values()),
    }


def prepare(
    readiness_dir: Path = READINESS_DIR,
    *,
    include_services: bool = True,
    hash_file: Callable[[str | Path], str] = sha256_path,
) -> dict[str, Any]:
    paths = readiness_paths(readiness_dir)
    if not paths["test_evidence"].is_file():
        raise J2A1V3IntegrityError(
            "Immutable test evidence must exist before prepare"
        )
    allowed_before = {paths["test_evidence"].resolve()}
    if readiness_dir.exists():
        observed_before = {
            path.resolve()
            for path in readiness_dir.rglob("*")
            if path.is_file()
        }
        if observed_before != allowed_before:
            raise J2A1V3IntegrityError(
                "Readiness namespace is not at the test-evidence boundary"
            )
    test_evidence = _load_test_evidence(paths["test_evidence"])
    bindings = source_and_input_bindings()
    integrity, authority, projection = v2_integrity_and_authority_audit(
        hash_file=hash_file
    )
    operations = operational_audit(
        output_dir=readiness_dir,
        include_services=include_services,
    )
    schema = schema_payload()
    if not projection["passes"] or not operations["passes"]:
        decision = HOLD
    else:
        decision = READY
    written = {
        "input_bindings": write_immutable_json(
            paths["input_bindings"],
            bindings,
            field=READINESS_FIELDS["input_bindings"],
        ),
        "v2_integrity": write_immutable_json(
            paths["v2_integrity"],
            integrity,
            field=READINESS_FIELDS["v2_integrity"],
        ),
        "authority": write_immutable_json(
            paths["authority"],
            authority,
            field=READINESS_FIELDS["authority"],
        ),
        "projection": write_immutable_json(
            paths["projection"],
            projection,
            field=READINESS_FIELDS["projection"],
        ),
        "schema": write_immutable_json(
            paths["schema"],
            schema,
            field=READINESS_FIELDS["schema"],
        ),
    }
    predecessor_identities = {
        "test_evidence": artifact_identity(
            paths["test_evidence"],
            READINESS_FIELDS["test_evidence"],
        ),
        **{
            key: artifact_identity(paths[key], READINESS_FIELDS[key])
            for key in written
        },
    }
    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "decision_candidate": decision,
        "source_identities": _source_identities(),
        "predecessors": predecessor_identities,
        "v2_terminal_decision": V2_HOLD,
        "v2_completed_roots": COMPLETED_ROOTS,
        "v2_attempt_records": ATTEMPT_RECORDS,
        "recovery_remaining_roots": REMAINING_ROOTS,
        "v2_streams": ACTIVE_STREAMS,
        "wall_cap_basis": "top-level elapsed wall time only",
        "aggregate_worker_seconds_conjunctive": False,
        "existing_v2_consumption_reused": True,
        "duplicate_reservation_or_consumption_authorized": False,
        "future_execution_root": _display_path(FUTURE_EXECUTION_DIR),
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "execution_authorized": False,
        "stage_b_open_authorized": False,
        "zero_work": dict(ZERO_WORK),
        "operations": operations,
        "passes": decision == READY,
    }
    lock = write_immutable_json(
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
        "v2_terminal_preserved": True,
        "completed_roots_preserved": COMPLETED_ROOTS,
        "remaining_roots_frozen": REMAINING_ROOTS,
        "attempt_records_bound": ATTEMPT_RECORDS,
        "aggregate_worker_seconds_descriptive":
            EXPECTED_WORKER_SECONDS,
        "observed_wall_seconds": EXPECTED_WALL_SECONDS,
        "projected_total_stage_a_wall_hours":
            projection["projected_total_stage_a_wall_hours"],
        "conservative_total_stage_a_wall_hours":
            projection["conservative_total_stage_a_wall_hours"],
        "execution_authorized": False,
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
            "test_evidence": test_evidence["passes"],
            "bindings": bindings["passes"],
            "v2_integrity": integrity["passes"],
            "authority": authority["passes"],
            "projection": projection["passes"],
            "operations": operations["passes"],
            "future_execution_absent":
                not FUTURE_EXECUTION_DIR.exists(),
            "zero_work": all(
                value == 0 for value in ZERO_WORK.values()
            ),
        },
        "passes": decision == READY,
    }
    result = write_immutable_json(
        paths["result"],
        result_payload,
        field=READINESS_FIELDS["result"],
    )
    retention = write_immutable_json(
        paths["retention"],
        _retention_payload(readiness_dir),
        field=READINESS_FIELDS["retention"],
    )
    observed_after = {
        path.resolve()
        for path in readiness_dir.rglob("*")
        if path.is_file()
    }
    expected_after = {path.resolve() for path in paths.values()}
    if observed_after != expected_after:
        raise J2A1V3IntegrityError(
            "Readiness namespace file set changed after seal"
        )
    return {
        "decision": decision,
        "lock": lock,
        "result": result,
        "retention": retention,
        "operations": operations,
        "projection": projection,
        "authority_summary": {
            "total_rows": authority["total_rows"],
            "completed_rows": authority["completed_rows"],
            "unfinished_rows": authority["unfinished_rows_count"],
            "total_streams": authority["total_streams"],
            "completed_refs_sha256": authority[
                "completed_refs_sha256"
            ],
            "unfinished_rows_sha256": authority[
                "unfinished_rows_sha256"
            ],
        },
        "execution_authorized": False,
        "passes": decision == READY,
    }


def recovery_contract_sha256(
    *,
    marker_sha256: str,
    authority_sha256: str,
    command: str,
) -> str:
    return canonical_json_hash(
        {
            "version": f"{VERSION}_owner_contract_v1",
            "marker_sha256": _validate_hex(
                marker_sha256,
                name="recovery marker",
            ),
            "authority_sha256": _validate_hex(
                authority_sha256,
                name="recovery authority",
            ),
            "command": command,
            "runner_sha256": sha256_path(RUNNER_PATH),
            "collectors": COLLECTORS,
        }
    )


def _owner_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    return payload_with_hash(payload, "owner_record_sha256")


def _read_owner_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = _read_jsonl(path)
    predecessor = None
    contract = None
    for sequence, record in enumerate(records):
        if not verify_payload_hash(record, "owner_record_sha256"):
            raise J2A1V3IntegrityError("Recovery owner record changed")
        if (
            record.get("sequence") != sequence
            or record.get("predecessor_record_sha256") != predecessor
            or record.get("kind") not in {"owner", "reclaim"}
        ):
            raise J2A1V3IntegrityError("Recovery ownership chain changed")
        if contract is None:
            contract = record.get("contract_sha256")
        elif record.get("contract_sha256") != contract:
            raise J2A1V3IntegrityError("Recovery owner contract changed")
        predecessor = record["owner_record_sha256"]
    return records


def _append_owner_record(path: Path, record: Mapping[str, Any]) -> None:
    serialized = canonical_json_bytes(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            raise J2A1V3OperationalHold(
                "Recovery owner was acquired concurrently"
            )
        with os.fdopen(fd, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        with path.open("ab") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())


def acquire_or_reclaim_recovery_owner(
    *,
    ledger_path: Path,
    marker_sha256: str,
    authority_sha256: str,
    command: str,
    pid: int,
    process_start_identity: str,
    is_live: Callable[[Mapping[str, Any]], bool],
    commit_head_sha256: str | None,
) -> dict[str, Any]:
    contract = recovery_contract_sha256(
        marker_sha256=marker_sha256,
        authority_sha256=authority_sha256,
        command=command,
    )
    records = _read_owner_ledger(ledger_path)
    if not records:
        record = _owner_record(
            {
                "version": f"{VERSION}_owner_v1",
                "sequence": 0,
                "predecessor_record_sha256": None,
                "kind": "owner",
                "contract_sha256": contract,
                "marker_sha256": marker_sha256,
                "authority_sha256": authority_sha256,
                "command": command,
                "runner_sha256": sha256_path(RUNNER_PATH),
                "hostname": socket.gethostname(),
                "pid": int(pid),
                "process_start_identity": process_start_identity,
                "commit_head_sha256": commit_head_sha256,
            }
        )
        _append_owner_record(ledger_path, record)
        return {"owner": record, "reclaimed": False, "passes": True}
    head = records[-1]
    if head.get("contract_sha256") != contract:
        raise J2A1V3IntegrityError(
            "Recovery owner belongs to another contract"
        )
    if is_live(head):
        raise J2A1V3OperationalHold(
            "A live recovery owner already exists"
        )
    if commit_head_sha256 is None:
        raise J2A1V3IntegrityError(
            "Dead-owner reclaim lacks an authenticated commit head"
        )
    _validate_hex(commit_head_sha256, name="recovery commit head")
    reclaim = _owner_record(
        {
            "version": f"{VERSION}_owner_reclaim_v1",
            "sequence": len(records),
            "predecessor_record_sha256": head["owner_record_sha256"],
            "kind": "reclaim",
            "contract_sha256": contract,
            "marker_sha256": marker_sha256,
            "authority_sha256": authority_sha256,
            "command": command,
            "runner_sha256": sha256_path(RUNNER_PATH),
            "hostname": socket.gethostname(),
            "pid": int(pid),
            "process_start_identity": process_start_identity,
            "recovered_owner_record_sha256":
                head["owner_record_sha256"],
            "predecessor_commit_head_sha256": commit_head_sha256,
            "process_death_verified": True,
            "zero_concurrent_writer_verified": True,
        }
    )
    _append_owner_record(ledger_path, reclaim)
    return {"owner": reclaim, "reclaimed": True, "passes": True}


def audit_zero_work(
    readiness_dir: Path = READINESS_DIR,
    *,
    include_services: bool = True,
) -> dict[str, Any]:
    existing = []
    if readiness_dir.exists():
        existing = sorted(
            str(path.relative_to(readiness_dir))
            for path in readiness_dir.rglob("*")
            if path.is_file()
        )
    allowed = {TEST_EVIDENCE_NAME}
    checks = {
        "future_execution_absent": not FUTURE_EXECUTION_DIR.exists(),
        "scientific_counters_zero": all(
            value == 0 for value in ZERO_WORK.values()
        ),
        "readiness_file_set_pre_prepare": set(existing) <= allowed,
        "v2_execution_preserved": V2_EXECUTION_DIR.is_dir(),
    }
    operations = operational_audit(
        output_dir=readiness_dir,
        include_services=include_services,
    )
    checks["operations"] = operations["passes"]
    return {
        "version": f"{VERSION}_zero_work_audit_v1",
        "existing_readiness_files": existing,
        "zero_work": dict(ZERO_WORK),
        "operations": operations,
        "checks": checks,
        "passes": all(checks.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readiness-dir",
        type=Path,
        default=READINESS_DIR,
    )
    subparsers = parser.add_subparsers(dest="verb", required=True)
    audit = subparsers.add_parser("audit-zero-work")
    audit.add_argument("--skip-services", action="store_true")
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument(
        "--record-json",
        action="append",
        required=True,
    )
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--skip-services", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verb == "audit-zero-work":
        result = audit_zero_work(
            args.readiness_dir,
            include_services=not args.skip_services,
        )
    elif args.verb == "write-test-evidence":
        records = []
        for encoded in args.record_json:
            try:
                record = json.loads(encoded)
            except json.JSONDecodeError as error:
                raise J2A1V3IntegrityError(
                    "Test-evidence record JSON is malformed"
                ) from error
            if not isinstance(record, dict):
                raise J2A1V3IntegrityError(
                    "Test-evidence record is not an object"
                )
            records.append(record)
        result = write_test_evidence(args.readiness_dir, records)
    elif args.verb == "prepare":
        result = prepare(
            args.readiness_dir,
            include_services=not args.skip_services,
        )
    else:
        raise J2A1V3IntegrityError("Forbidden CLI route")
    print(json.dumps(_json_native(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

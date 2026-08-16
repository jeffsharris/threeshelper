"""Training-only J1b execution surface.

The module stays standard-library-only through argument parsing.  Torch and
the immutable parent J1 implementation are imported only by the execute path,
after the selected command and its artifact contract have been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "j1b_training_execution_surface_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs"
CHARTER_PATH = (
    REPO_ROOT / "threes_rl" / "J1B_TRAINING_EXECUTION_SURFACE_CHARTER.md"
)
RUNNER_PATH = REPO_ROOT / "threes_rl" / "j1b_training_execution_surface.py"
TEST_PATH = REPO_ROOT / "tests" / "test_rl_j1b_training_execution_surface.py"
READINESS_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j1b_training_execution_surface_readiness_v1"
)
FUTURE_EXECUTION_ROOT = (
    RUNS_ROOT / "forensics" / "j1b_execution_v1"
)
J1B_PREFLIGHT_DIR = (
    RUNS_ROOT / "forensics" / "j1b_operational_repair_readiness_v1"
)
J1B_PRE_A1_HISTORY_PATH = (
    RUNS_ROOT
    / "forensics"
    / "j1b_operational_repair_preseal_history_v1"
    / "J1B_TEST_EVIDENCE_PRE_A1.json"
)
SPENT_J1_TRAINING_DIR = (
    RUNS_ROOT / "forensics" / "j1_execution_v1" / "training"
)

EXPECTED_CHARTER_SHA256 = (
    "aeb458781e206f8f16002ffaa311d782b26fdb4076211155a6230b9835e29858"
)
EXPECTED_J1B_SOURCE_IDENTITIES = {
    "threes_rl/J1B_OPERATIONAL_REPAIR_PREFLIGHT_CHARTER.md":
        "a426801fc3015051ea51517e925a7d1c2e556718e2551ee480b802c8a7422cc1",
    "threes_rl/J1B_OPERATIONAL_REPAIR_PREFLIGHT_AMENDMENT_A1.md":
        "64de3de37bff6a08bd95da217dc52d2f4bb58fbf99d28bede263a44d0aa2eb9c",
    "threes_rl/j1b_operational_repair_preflight.py":
        "7d73565c510dfe74b87ec362c05f8928e15a65cb8af5494b5ad9fe5f4c30ca5f",
    "tests/test_rl_j1b_operational_repair_preflight.py":
        "f7e55b71f7954fcbdd4db61c1693d773b8ea106684ea19ad19998be15f4dbaff",
}
EXPECTED_J1B_READINESS_FILES = {
    "J1B_READINESS_LOCK.json":
        "b8b5377370f0e9e04739aae582604ce85f38bd1ddf84b5312a2cf12406f38814",
    "J1B_READINESS_RESULT.json":
        "108038d15b222afd00c07c9801b460fb4687bfe0a9e8a4fb54a59e58e8907ec6",
}
EXPECTED_J1B_READINESS_PAYLOADS = {
    "J1B_READINESS_LOCK.json": (
        "readiness_lock_payload_sha256",
        "ef0c1adce5f948a238e81911ab034d84ed297c2b2570d58481fb2906ef2e7e3b",
    ),
    "J1B_READINESS_RESULT.json": (
        "readiness_result_payload_sha256",
        "5d56b2c3cec39c16590a20f8acf8f10c60db7739e5161a653ea45a779204ba5e",
    ),
}
EXPECTED_PRE_A1_FILE_SHA256 = (
    "d2f6333bd4fdbe584fbf231141a24c01256dcc9ebe0f57c2691e19a8f046bddf"
)
EXPECTED_PRE_A1_PAYLOAD_SHA256 = (
    "b462c0b46afaa478caeb66c622799eb1e7a533673439a89fe0e60650a448e25e"
)
EXPECTED_SOURCE_MANIFEST_FILE_SHA256 = (
    "2bb0b2385360f2d06c019fdbac11cb58515629ab4f5fcf321624f499a07329f9"
)
EXPECTED_SOURCE_MANIFEST_PAYLOAD_SHA256 = (
    "f85a7624b2e8052d0b451bde9bf792181e08e055406fb5837232655a48f8a8a8"
)
EXPECTED_CANONICAL_ROWS_SHA256 = (
    "4d28217d402d8b0e67e5465c90e433556f7f79adb0aedaf9c682c4defabfb170"
)
EXPECTED_ROOT_COMMITMENT_PAYLOAD_SHA256 = (
    "fe8f67395dc2fced9d2b0f86c6990f66563681b868b04707d8771a8a4fe85d12"
)
EXPECTED_ROOT_SET_SHA256 = (
    "3a44d95d25c3b979d8e94dfbdb7c59b7e1b25dea891f8a31dd4f36040156ba55"
)
EXPECTED_SPENT_TERMINAL_FILE_SHA256 = (
    "21092fb34631eb0eaf48811caa814ff4d05abbb23c9bc5add85eefd93a8959d3"
)
EXPECTED_SPENT_RETENTION_FILE_SHA256 = (
    "dc339aafdbe32859d07c591a36c9088afa53f5be30412f3340049ca18994ceb0"
)
EXPECTED_PARENT_ENGINE_SHA256 = (
    "d4367d95aba05ec592310008bae21e7de90905fa1268601dd60cc8fcb2b6f2bd"
)
EXPECTED_MODEL_SCHEMA_SHA256 = (
    "75919f80ed3550f27e1929cad355f2380e39058409456a125c86001f149d5351"
)
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "f5102689925a4db2cc3972a0a6d0e88943f87a48f9206ddf36bfb1d0f7c7b80e"
)
EXPECTED_PARAMETER_COUNT = 411_656
TRAIN_ROOTS = 16_384
STREAM_RANGES = {
    "logical_stream_id": (213_000_016_384, 213_000_032_767),
    "deck_stream_id": (214_000_016_384, 214_000_032_767),
    "slot_stream_id": (215_000_016_384, 215_000_032_767),
    "candidate_policy_stream_id": (
        216_000_016_384,
        216_000_032_767,
    ),
}

PUBLIC_COMMANDS = (
    "seal-phase-lock",
    "open",
    "materialize",
    "execute",
)
PHASE_DIR_NAME = "training"
PHASE_LOCK_NAME = "phase_lock.json"
PHASE_LOCK_RESULT_NAME = "phase_lock_result.json"
PHASE_MARKER_NAME = "execution_opened.json"
PHASE_MANIFEST_NAME = "root_manifest.json"
PHASE_OWNER_NAME = "writer_owner.json"
PHASE_RESERVATION_NAME = "stream_reservation.json"
PHASE_CONSUMPTION_NAME = "stream_consumption_opened.json"
PHASE_RESULT_NAME = "terminal_result.json"
PHASE_RETENTION_NAME = "retention_manifest.json"
COMMIT_HEAD_NAME = "commit_head.json"

TEST_EVIDENCE_NAME = "J1B_TRAINING_EXECUTION_TEST_EVIDENCE.json"
SCHEMA_NAME = "J1B_TRAINING_EXECUTION_SCHEMA.json"
PROJECTION_NAME = "J1B_TRAINING_EXECUTION_PROJECTION.json"
INPUT_BINDINGS_NAME = "J1B_TRAINING_EXECUTION_INPUT_BINDINGS.json"
READINESS_LOCK_NAME = "J1B_TRAINING_EXECUTION_READINESS_LOCK.json"
READINESS_RESULT_NAME = "J1B_TRAINING_EXECUTION_READINESS_RESULT.json"
READY_DECISION = "READY_J1B_TRAINING_EXECUTION_SURFACE"
HOLD_DECISION = "HOLD_J1B_TRAINING_EXECUTION_SURFACE"
KILL_DECISION = "KILL_J1B_TRAINING_EXECUTION_SURFACE_INTEGRITY"


class J1bSurfaceError(RuntimeError):
    """Base error for the J1b training-only surface."""


class J1bSurfaceIntegrityError(J1bSurfaceError):
    """Immutable identity or scientific execution integrity failed."""


class J1bSurfaceOperationalHold(J1bSurfaceError):
    """A mutable operational or resource condition failed."""


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
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
    body = dict(payload)
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    observed = body.pop(field, None)
    return (
        isinstance(observed, str)
        and observed == canonical_json_hash(body)
    )


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise J1bSurfaceIntegrityError(
            f"Expected JSON object: {path}"
        )
    return payload


def ordered_rows_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
    allow_existing_exact: bool = False,
) -> dict[str, Any]:
    body = payload_with_hash(payload, field)
    serialized = (
        json.dumps(body, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    reloaded = json.loads(serialized.decode("utf-8"))
    if not verify_payload_hash(reloaded, field):
        raise J1bSurfaceIntegrityError(
            f"JSON reload instability: {path}"
        )
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
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            observed = path.read_bytes()
            if observed != serialized:
                raise J1bSurfaceIntegrityError(
                    f"Immutable artifact collision changed bytes: {path}"
                ) from error
            payload_observed = json.loads(observed.decode("utf-8"))
            if not verify_payload_hash(payload_observed, field):
                raise J1bSurfaceIntegrityError(
                    f"Existing immutable artifact is invalid: {path}"
                ) from error
            if allow_existing_exact:
                return payload_observed
            raise FileExistsError(
                f"Immutable artifact already exists: {path}"
            ) from error
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)
    observed = load_json(path)
    if observed != body or not verify_payload_hash(observed, field):
        raise J1bSurfaceIntegrityError(
            f"Written immutable artifact changed: {path}"
        )
    return observed


def immutable_json_identity(
    path: Path,
    *,
    payload_field: str,
    decision: str | None = None,
) -> dict[str, Any]:
    payload = load_json(path)
    if not verify_payload_hash(payload, payload_field):
        raise J1bSurfaceIntegrityError(
            f"Immutable JSON payload is invalid: {path}"
        )
    if decision is not None and payload.get("decision") != decision:
        raise J1bSurfaceIntegrityError(
            f"Immutable JSON decision changed: {path}"
        )
    identity = {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_field": payload_field,
        "payload_sha256": payload[payload_field],
    }
    if decision is not None:
        identity["decision"] = decision
    return identity


def _assert_file_hash(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise J1bSurfaceIntegrityError(
            f"Required immutable file is missing: {path}"
        )
    observed = sha256_path(path)
    if observed != expected:
        raise J1bSurfaceIntegrityError(
            f"Immutable file changed: {path}"
        )


def _assert_json_identity(
    path: Path,
    *,
    file_sha256: str,
    payload_field: str,
    payload_sha256: str,
    decision: str | None = None,
) -> dict[str, Any]:
    _assert_file_hash(path, file_sha256)
    payload = load_json(path)
    if (
        not verify_payload_hash(payload, payload_field)
        or payload.get(payload_field) != payload_sha256
    ):
        raise J1bSurfaceIntegrityError(
            f"Immutable payload changed: {path}"
        )
    if decision is not None and payload.get("decision") != decision:
        raise J1bSurfaceIntegrityError(
            f"Immutable decision changed: {path}"
        )
    return payload


def phase_paths(execution_root: Path) -> dict[str, Path]:
    phase_dir = execution_root / PHASE_DIR_NAME
    return {
        "phase_dir": phase_dir,
        "lock": phase_dir / PHASE_LOCK_NAME,
        "lock_result": phase_dir / PHASE_LOCK_RESULT_NAME,
        "marker": phase_dir / PHASE_MARKER_NAME,
        "manifest": phase_dir / PHASE_MANIFEST_NAME,
        "owner": phase_dir / PHASE_OWNER_NAME,
        "reservation": phase_dir / PHASE_RESERVATION_NAME,
        "consumption": phase_dir / PHASE_CONSUMPTION_NAME,
        "result": phase_dir / PHASE_RESULT_NAME,
        "retention": phase_dir / PHASE_RETENTION_NAME,
        "commit_head": phase_dir / COMMIT_HEAD_NAME,
        "checkpoint": phase_dir / "round64_candidate_checkpoint.bin",
        "sanity": phase_dir / "training_sanity_result.json",
    }


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def bound_command(
    action: str,
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> str:
    if action not in PUBLIC_COMMANDS:
        raise ValueError(f"Unsupported J1b command: {action}")
    return shlex.join(
        [
            "nice",
            "-n",
            "10",
            "env",
            "PYTHONPATH=.",
            ".venv/bin/python",
            "-m",
            "threes_rl.j1b_training_execution_surface",
            action,
            "--execution-root",
            _relative_or_absolute(execution_root),
            "--readiness-dir",
            _relative_or_absolute(readiness_dir),
            "--jobs",
            "1",
        ]
    )


def _source_manifest_path() -> Path:
    return J1B_PREFLIGHT_DIR / "J1B_PROSPECTIVE_TRAINING_MANIFEST.json"


def _validate_source_manifest(
    payload: Mapping[str, Any],
    *,
    scientific: bool,
) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    commitment = payload.get("root_commitment", {})
    root_ids = [str(row.get("root_id", "")) for row in rows]
    ancestry_ids = [str(row.get("ancestry_id", "")) for row in rows]
    expected_count = TRAIN_ROOTS if scientific else len(rows)
    checks = {
        "payload_hash_valid": verify_payload_hash(
            payload,
            "prospective_manifest_payload_sha256",
        ),
        "passes_true": payload.get("passes") is True,
        "phase_training": payload.get("phase") == "training",
        "partition_train": payload.get("partition") == "train",
        "row_count_exact": len(rows) == expected_count,
        "row_indices_exact": [
            int(row.get("row_index", -1)) for row in rows
        ] == list(range(len(rows))),
        "root_ids_unique": len(set(root_ids)) == len(rows),
        "ancestries_unique": len(set(ancestry_ids)) == len(rows),
        "one_root_per_ancestry": root_ids == ancestry_ids,
        "starter_none": all(
            row.get("starter_tile") is None for row in rows
        ),
        "one_arm": all(int(row.get("arm_count", -1)) == 1 for row in rows),
        "canonical_rows_exact": (
            ordered_rows_hash(rows)
            == payload.get("canonical_rows_sha256")
        ),
        "root_set_exact": (
            canonical_json_hash(root_ids)
            == (
                EXPECTED_ROOT_SET_SHA256
                if scientific
                else canonical_json_hash(root_ids)
            )
        ),
    }
    if scientific:
        checks.update(
            {
                "source_payload_exact": (
                    payload.get("prospective_manifest_payload_sha256")
                    == EXPECTED_SOURCE_MANIFEST_PAYLOAD_SHA256
                ),
                "canonical_rows_bound": (
                    payload.get("canonical_rows_sha256")
                    == EXPECTED_CANONICAL_ROWS_SHA256
                ),
                "root_commitment_bound": (
                    commitment.get("marker_payload_sha256")
                    == EXPECTED_ROOT_COMMITMENT_PAYLOAD_SHA256
                    and verify_payload_hash(
                        commitment,
                        "marker_payload_sha256",
                    )
                ),
            }
        )
        for field, (start, end) in STREAM_RANGES.items():
            values = [int(row.get(field, -1)) for row in rows]
            checks[f"{field}_range_exact"] = (
                values == list(range(start, end + 1))
            )
        all_streams = [
            int(row[field])
            for row in rows
            for field in STREAM_RANGES
        ]
        checks["all_stream_roles_unique"] = (
            len(all_streams) == len(set(all_streams)) == TRAIN_ROOTS * 4
        )
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise J1bSurfaceIntegrityError(
            "J1b source manifest validation failed: "
            + ", ".join(failed)
        )
    return {
        "row_count": len(rows),
        "root_set_sha256": canonical_json_hash(root_ids),
        "ancestry_set_sha256": canonical_json_hash(ancestry_ids),
        "checks": checks,
        "passes": True,
    }


def build_materialized_manifest(
    source: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    scientific: bool,
) -> dict[str, Any]:
    validation = _validate_source_manifest(
        source,
        scientific=scientific,
    )
    rows = list(source["rows"])
    stream_roles = {}
    for field in STREAM_RANGES:
        values = [int(row[field]) for row in rows]
        stream_roles[field] = {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "ordered_sha256": canonical_json_hash(values),
        }
    payload = {
        "version": f"{VERSION}_training_root_manifest_v1",
        "phase": "training",
        "partition": "train",
        "rows": rows,
        "canonical_rows_sha256": source["canonical_rows_sha256"],
        "root_set_sha256": validation["root_set_sha256"],
        "ancestry_set_sha256": validation["ancestry_set_sha256"],
        "source_manifest_identity": dict(source_identity),
        "root_commitment": source.get("root_commitment"),
        "stream_roles": stream_roles,
        "checks": validation["checks"],
        "passes": True,
    }
    return payload_with_hash(payload, "root_manifest_payload_sha256")


def manifest_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not verify_payload_hash(payload, "root_manifest_payload_sha256")
        or payload.get("passes") is not True
    ):
        raise J1bSurfaceIntegrityError(
            "Materialized manifest payload is invalid"
        )
    return {
        "phase": "training",
        "row_count": len(payload["rows"]),
        "canonical_rows_sha256": payload["canonical_rows_sha256"],
        "root_set_sha256": payload["root_set_sha256"],
        "ancestry_set_sha256": payload["ancestry_set_sha256"],
        "payload_sha256": payload["root_manifest_payload_sha256"],
    }


def audit_authoritative_inputs(
    *,
    require_future_execution_absent: bool,
) -> dict[str, Any]:
    _assert_file_hash(CHARTER_PATH, EXPECTED_CHARTER_SHA256)
    for relative, expected in EXPECTED_J1B_SOURCE_IDENTITIES.items():
        _assert_file_hash(REPO_ROOT / relative, expected)

    lock_path = J1B_PREFLIGHT_DIR / "J1B_READINESS_LOCK.json"
    result_path = J1B_PREFLIGHT_DIR / "J1B_READINESS_RESULT.json"
    lock = _assert_json_identity(
        lock_path,
        file_sha256=EXPECTED_J1B_READINESS_FILES[
            "J1B_READINESS_LOCK.json"
        ],
        payload_field=EXPECTED_J1B_READINESS_PAYLOADS[
            "J1B_READINESS_LOCK.json"
        ][0],
        payload_sha256=EXPECTED_J1B_READINESS_PAYLOADS[
            "J1B_READINESS_LOCK.json"
        ][1],
        decision="READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
    )
    result = _assert_json_identity(
        result_path,
        file_sha256=EXPECTED_J1B_READINESS_FILES[
            "J1B_READINESS_RESULT.json"
        ],
        payload_field=EXPECTED_J1B_READINESS_PAYLOADS[
            "J1B_READINESS_RESULT.json"
        ][0],
        payload_sha256=EXPECTED_J1B_READINESS_PAYLOADS[
            "J1B_READINESS_RESULT.json"
        ][1],
        decision="READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
    )
    expected_lock_identity = immutable_json_identity(
        lock_path,
        payload_field="readiness_lock_payload_sha256",
    )
    if result.get("readiness_lock") != expected_lock_identity:
        raise J1bSurfaceIntegrityError(
            "J1b preflight result changed its lock identity"
        )

    checked_artifacts: dict[str, Any] = {}
    for name, identity in sorted(lock.get("artifacts", {}).items()):
        path = Path(str(identity.get("path", "")))
        payload = _assert_json_identity(
            path,
            file_sha256=str(identity.get("file_sha256", "")),
            payload_field=str(identity.get("payload_field", "")),
            payload_sha256=str(identity.get("payload_sha256", "")),
        )
        checked_artifacts[name] = {
            **dict(identity),
            "version": payload.get("version"),
        }

    parent_sources = dict(lock.get("parent_source_identities", {}))
    for relative, expected in sorted(parent_sources.items()):
        _assert_file_hash(REPO_ROOT / relative, str(expected))
    parent_readiness_dir = (
        RUNS_ROOT / "forensics" / "j1_execution_surface_readiness_v1"
    )
    parent_readiness = dict(
        lock.get("parent_readiness_identities", {})
    )
    for name, expected in sorted(parent_readiness.items()):
        _assert_file_hash(parent_readiness_dir / name, str(expected))

    history = dict(lock.get("pre_a1_historical_evidence", {}))
    _assert_json_identity(
        J1B_PRE_A1_HISTORY_PATH,
        file_sha256=EXPECTED_PRE_A1_FILE_SHA256,
        payload_field="test_evidence_payload_sha256",
        payload_sha256=EXPECTED_PRE_A1_PAYLOAD_SHA256,
    )
    if (
        history.get("file_sha256") != EXPECTED_PRE_A1_FILE_SHA256
        or history.get("payload_sha256")
        != EXPECTED_PRE_A1_PAYLOAD_SHA256
        or Path(str(history.get("path", ""))).resolve()
        != J1B_PRE_A1_HISTORY_PATH.resolve()
    ):
        raise J1bSurfaceIntegrityError(
            "J1b pre-A1 evidence binding changed"
        )

    spent = dict(lock.get("spent_j1_execution_identities", {}))
    for relative, expected in sorted(spent.items()):
        _assert_file_hash(SPENT_J1_TRAINING_DIR / relative, str(expected))
    if (
        spent.get("terminal_result.json")
        != EXPECTED_SPENT_TERMINAL_FILE_SHA256
        or spent.get("retention_manifest.json")
        != EXPECTED_SPENT_RETENTION_FILE_SHA256
    ):
        raise J1bSurfaceIntegrityError(
            "Spent J1 terminal or retention binding changed"
        )

    source_manifest_path = _source_manifest_path()
    _assert_file_hash(
        source_manifest_path,
        EXPECTED_SOURCE_MANIFEST_FILE_SHA256,
    )
    source_manifest = load_json(source_manifest_path)
    if (
        source_manifest.get("prospective_manifest_payload_sha256")
        != EXPECTED_SOURCE_MANIFEST_PAYLOAD_SHA256
    ):
        raise J1bSurfaceIntegrityError(
            "J1b source manifest payload changed"
        )
    source_validation = _validate_source_manifest(
        source_manifest,
        scientific=True,
    )
    checks = {
        "charter_exact": True,
        "j1b_source_identities_exact": True,
        "j1b_readiness_lock_exact": True,
        "j1b_readiness_result_exact": True,
        "j1b_readiness_artifacts_exact": len(checked_artifacts) == 7,
        "pre_a1_history_exact": True,
        "parent_sources_exact": (
            parent_sources.get("threes_rl/j1_execution_surface.py")
            == EXPECTED_PARENT_ENGINE_SHA256
        ),
        "parent_readiness_exact": len(parent_readiness) == 6,
        "spent_j1_inventory_exact": len(spent) == 14,
        "source_manifest_exact": source_validation["passes"],
        "future_execution_root_state": (
            not FUTURE_EXECUTION_ROOT.exists()
            if require_future_execution_absent
            else True
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise J1bSurfaceIntegrityError(
            "Authoritative input audit failed: " + ", ".join(failed)
        )
    identities = {
        "charter": {
            "path": str(CHARTER_PATH.resolve()),
            "file_sha256": EXPECTED_CHARTER_SHA256,
        },
        "j1b_preflight_lock": immutable_json_identity(
            lock_path,
            payload_field="readiness_lock_payload_sha256",
            decision="READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
        ),
        "j1b_preflight_result": immutable_json_identity(
            result_path,
            payload_field="readiness_result_payload_sha256",
            decision="READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
        ),
        "j1b_preflight_artifacts": checked_artifacts,
        "pre_a1_history": {
            "path": str(J1B_PRE_A1_HISTORY_PATH.resolve()),
            "file_sha256": EXPECTED_PRE_A1_FILE_SHA256,
            "payload_sha256": EXPECTED_PRE_A1_PAYLOAD_SHA256,
        },
        "parent_sources": parent_sources,
        "parent_readiness": parent_readiness,
        "spent_j1_execution": spent,
        "source_manifest": immutable_json_identity(
            source_manifest_path,
            payload_field="prospective_manifest_payload_sha256",
        ),
    }
    return {
        "identities": identities,
        "identities_sha256": canonical_json_hash(identities),
        "source_manifest_validation": source_validation,
        "checks": checks,
        "passes": True,
    }


def _readiness_paths(readiness_dir: Path) -> dict[str, Path]:
    return {
        "test_evidence": readiness_dir / TEST_EVIDENCE_NAME,
        "schema": readiness_dir / SCHEMA_NAME,
        "projection": readiness_dir / PROJECTION_NAME,
        "input_bindings": readiness_dir / INPUT_BINDINGS_NAME,
        "lock": readiness_dir / READINESS_LOCK_NAME,
        "result": readiness_dir / READINESS_RESULT_NAME,
    }


def load_ready_surface(readiness_dir: Path) -> dict[str, Any]:
    paths = _readiness_paths(readiness_dir)
    lock = load_json(paths["lock"])
    result = load_json(paths["result"])
    if (
        not verify_payload_hash(
            lock,
            "readiness_lock_payload_sha256",
        )
        or lock.get("decision") != READY_DECISION
    ):
        raise J1bSurfaceIntegrityError(
            "J1b execution-surface readiness lock is invalid"
        )
    if (
        not verify_payload_hash(
            result,
            "readiness_result_payload_sha256",
        )
        or result.get("decision") != READY_DECISION
    ):
        raise J1bSurfaceIntegrityError(
            "J1b execution-surface readiness result is invalid"
        )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="readiness_lock_payload_sha256",
        decision=READY_DECISION,
    )
    result_identity = immutable_json_identity(
        paths["result"],
        payload_field="readiness_result_payload_sha256",
        decision=READY_DECISION,
    )
    if result.get("readiness_lock_identity") != lock_identity:
        raise J1bSurfaceIntegrityError(
            "J1b surface result changed its readiness lock"
        )
    mode = str(lock.get("execution_mode", ""))
    if mode not in {"scientific", "miniature_fixture"}:
        raise J1bSurfaceIntegrityError(
            "J1b readiness execution mode is invalid"
        )
    expected_sources = {
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
    }
    for field, expected in expected_sources.items():
        if lock.get(field) != expected:
            raise J1bSurfaceIntegrityError(
                f"J1b readiness changed {field}"
            )
    artifact_fields = {
        "test_evidence": "test_evidence_payload_sha256",
        "schema": "schema_payload_sha256",
        "projection": "projection_payload_sha256",
        "input_bindings": "input_bindings_payload_sha256",
    }
    for name, payload_field in artifact_fields.items():
        observed = immutable_json_identity(
            paths[name],
            payload_field=payload_field,
        )
        if lock.get("artifacts", {}).get(name) != observed:
            raise J1bSurfaceIntegrityError(
                f"J1b readiness changed {name}"
            )
    expected_root = Path(str(lock.get("future_execution_root", "")))
    if mode == "scientific":
        if (
            expected_root.resolve() != FUTURE_EXECUTION_ROOT.resolve()
            or readiness_dir.resolve() != READINESS_DIR.resolve()
            or lock.get("scientific_authority") is not True
            or lock.get("fixture_only") is not False
        ):
            raise J1bSurfaceIntegrityError(
                "Scientific readiness paths or authority changed"
            )
        input_audit = audit_authoritative_inputs(
            require_future_execution_absent=False,
        )
        if (
            lock.get("authoritative_input_identities_sha256")
            != input_audit["identities_sha256"]
        ):
            raise J1bSurfaceIntegrityError(
                "Scientific readiness input binding changed"
            )
    else:
        if (
            lock.get("scientific_authority") is not False
            or lock.get("fixture_only") is not True
        ):
            raise J1bSurfaceIntegrityError(
                "Fixture readiness has scientific authority"
            )
    source_identity = dict(lock.get("source_manifest_identity", {}))
    source_path = Path(str(source_identity.get("path", "")))
    if (
        not source_path.is_file()
        or sha256_path(source_path)
        != source_identity.get("file_sha256")
    ):
        raise J1bSurfaceIntegrityError(
            "Readiness source manifest file changed"
        )
    source = load_json(source_path)
    payload_field = str(source_identity.get("payload_field", ""))
    if (
        not verify_payload_hash(source, payload_field)
        or source.get(payload_field)
        != source_identity.get("payload_sha256")
    ):
        raise J1bSurfaceIntegrityError(
            "Readiness source manifest payload changed"
        )
    _validate_source_manifest(
        source,
        scientific=mode == "scientific",
    )
    return {
        "paths": paths,
        "lock": lock,
        "result": result,
        "lock_identity": lock_identity,
        "result_identity": result_identity,
        "mode": mode,
        "future_execution_root": expected_root,
        "source_manifest": source,
        "source_manifest_identity": source_identity,
        "passes": True,
    }


def _phase_ready_decision() -> str:
    return "READY_J1B_TRAINING_PHASE_LOCK"


def _assert_execution_root(
    execution_root: Path,
    readiness: Mapping[str, Any],
) -> None:
    if (
        execution_root.resolve()
        != Path(readiness["future_execution_root"]).resolve()
    ):
        raise J1bSurfaceIntegrityError(
            "Execution root differs from immutable readiness"
        )


def seal_training_phase_lock(
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    readiness = load_ready_surface(readiness_dir)
    _assert_execution_root(execution_root, readiness)
    paths = phase_paths(execution_root)
    if execution_root.exists():
        present = sorted(
            str(path.relative_to(execution_root))
            for path in execution_root.rglob("*")
            if path.is_file()
        )
        if present:
            raise FileExistsError(
                "J1b phase lock requires a fresh execution root: "
                + ", ".join(present)
            )
    materialized = build_materialized_manifest(
        readiness["source_manifest"],
        source_identity=readiness["source_manifest_identity"],
        scientific=readiness["mode"] == "scientific",
    )
    commands = {
        action: bound_command(
            action,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
        for action in PUBLIC_COMMANDS
    }
    payload = {
        "version": f"{VERSION}_training_phase_lock_v1",
        "phase": "training",
        "decision": _phase_ready_decision(),
        "execution_mode": readiness["mode"],
        "scientific_authority": (
            readiness["mode"] == "scientific"
        ),
        "fixture_only": readiness["mode"] == "miniature_fixture",
        "readiness_lock_identity": readiness["lock_identity"],
        "readiness_result_identity": readiness["result_identity"],
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
        "bounded_engine": "execute_training_engine_bounded",
        "legacy_engine_unreachable": True,
        "training_only": True,
        "development_command_present": False,
        "confirmation_command_present": False,
        "promotion_command_present": False,
        "future_execution_root": str(execution_root.resolve()),
        "source_manifest_identity":
            readiness["source_manifest_identity"],
        "manifest_identity": manifest_identity(materialized),
        "commands": commands,
        "runtime_order": [
            "parse_arguments_standard_library_only",
            "configure_torch_interop_one",
            "configure_torch_intraop_one",
            "enable_deterministic_algorithms",
            "verify_runtime",
            "import_parent",
            "initialize_frozen_model_optimizer",
            "first_parent_operational_guard",
            "acquire_or_reclaim_owner",
            "reserve_streams",
            "open_stream_consumption",
            "bounded_engine_genesis_and_work",
        ],
        "first_guard_before_owner": True,
        "first_guard_before_reservation": True,
        "first_guard_before_consumption": True,
        "first_guard_before_genesis": True,
        "jobs": 1,
        "nice_minimum": 10,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "scientific_games": 0,
        "scientific_optimizer_steps": 0,
        "scientific_checkpoints": 0,
        "policy_or_score_outcomes": 0,
        "passes": True,
    }
    written_lock = write_immutable_json(
        paths["lock"],
        payload,
        field="phase_lock_payload_sha256",
    )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="phase_lock_payload_sha256",
        decision=_phase_ready_decision(),
    )
    result = write_immutable_json(
        paths["lock_result"],
        {
            "version": f"{VERSION}_training_phase_lock_result_v1",
            "phase": "training",
            "decision": _phase_ready_decision(),
            "phase_lock_identity": lock_identity,
            "readiness_lock_identity": readiness["lock_identity"],
            "readiness_result_identity": readiness["result_identity"],
            "streams_reserved": 0,
            "streams_consumed": 0,
            "scientific_work": 0,
            "passes": True,
        },
        field="phase_lock_result_payload_sha256",
    )
    return {
        "lock": written_lock,
        "lock_identity": lock_identity,
        "result": result,
        "result_identity": immutable_json_identity(
            paths["lock_result"],
            payload_field="phase_lock_result_payload_sha256",
            decision=_phase_ready_decision(),
        ),
        "passes": True,
    }


def load_training_phase_lock(
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    readiness = load_ready_surface(readiness_dir)
    _assert_execution_root(execution_root, readiness)
    paths = phase_paths(execution_root)
    lock = load_json(paths["lock"])
    lock_result = load_json(paths["lock_result"])
    if (
        not verify_payload_hash(lock, "phase_lock_payload_sha256")
        or lock.get("decision") != _phase_ready_decision()
        or lock.get("phase") != "training"
    ):
        raise J1bSurfaceIntegrityError(
            "J1b training phase lock is invalid"
        )
    if (
        not verify_payload_hash(
            lock_result,
            "phase_lock_result_payload_sha256",
        )
        or lock_result.get("decision") != _phase_ready_decision()
    ):
        raise J1bSurfaceIntegrityError(
            "J1b training phase lock result is invalid"
        )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="phase_lock_payload_sha256",
        decision=_phase_ready_decision(),
    )
    if lock_result.get("phase_lock_identity") != lock_identity:
        raise J1bSurfaceIntegrityError(
            "J1b lock result changed phase lock identity"
        )
    if (
        lock.get("readiness_lock_identity")
        != readiness["lock_identity"]
        or lock.get("readiness_result_identity")
        != readiness["result_identity"]
    ):
        raise J1bSurfaceIntegrityError(
            "J1b phase lock changed readiness identities"
        )
    for field, expected in {
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
    }.items():
        if lock.get(field) != expected:
            raise J1bSurfaceIntegrityError(
                f"J1b phase lock changed {field}"
            )
    expected_commands = {
        action: bound_command(
            action,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
        for action in PUBLIC_COMMANDS
    }
    checks = {
        "mode_exact": lock.get("execution_mode")
        == readiness["mode"],
        "authority_exact": lock.get("scientific_authority")
        is (readiness["mode"] == "scientific"),
        "bounded_engine_exact": (
            lock.get("bounded_engine")
            == "execute_training_engine_bounded"
        ),
        "legacy_unreachable": (
            lock.get("legacy_engine_unreachable") is True
        ),
        "training_only": lock.get("training_only") is True,
        "no_development": (
            lock.get("development_command_present") is False
        ),
        "no_confirmation": (
            lock.get("confirmation_command_present") is False
        ),
        "no_promotion": (
            lock.get("promotion_command_present") is False
        ),
        "commands_exact": lock.get("commands") == expected_commands,
        "source_manifest_exact": (
            lock.get("source_manifest_identity")
            == readiness["source_manifest_identity"]
        ),
    }
    expected_manifest = build_materialized_manifest(
        readiness["source_manifest"],
        source_identity=readiness["source_manifest_identity"],
        scientific=readiness["mode"] == "scientific",
    )
    checks["manifest_identity_exact"] = (
        lock.get("manifest_identity")
        == manifest_identity(expected_manifest)
    )
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise J1bSurfaceIntegrityError(
            "J1b phase lock validation failed: " + ", ".join(failed)
        )
    return {
        "paths": paths,
        "readiness": readiness,
        "lock": lock,
        "lock_result": lock_result,
        "lock_identity": lock_identity,
        "expected_manifest": expected_manifest,
        "commands": expected_commands,
        "checks": checks,
        "passes": True,
    }


def _fixture_open_audit() -> dict[str, Any]:
    return {
        "version": f"{VERSION}_fixture_open_audit_v1",
        "checks": {
            "fixture_only": True,
            "no_scientific_work": True,
        },
        "passes": True,
    }


def open_training_phase(
    *,
    execution_root: Path,
    readiness_dir: Path,
    opened_at: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    loaded = load_training_phase_lock(
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    paths = loaded["paths"]
    for name in (
        "marker",
        "manifest",
        "owner",
        "reservation",
        "consumption",
        "commit_head",
        "result",
        "checkpoint",
    ):
        if paths[name].exists():
            raise FileExistsError(
                f"J1b open requires unused artifact: {paths[name]}"
            )
    if loaded["readiness"]["mode"] == "scientific":
        from threes_rl import j1_joint_policy_value as j1

        operational = j1.operational_audit(
            output_dir=paths["phase_dir"]
        )
        if operational.get("passes") is not True:
            raise J1bSurfaceOperationalHold(
                "J1b training open operational audit failed"
            )
    else:
        operational = _fixture_open_audit()
    marker = {
        "version": f"{VERSION}_training_execution_opened_v1",
        "phase": "training",
        "execution_mode": loaded["readiness"]["mode"],
        "scientific_authority": (
            loaded["readiness"]["mode"] == "scientific"
        ),
        "phase_lock_identity": loaded["lock_identity"],
        "phase_lock_result_identity": immutable_json_identity(
            paths["lock_result"],
            payload_field="phase_lock_result_payload_sha256",
            decision=_phase_ready_decision(),
        ),
        "readiness_lock_identity":
            loaded["readiness"]["lock_identity"],
        "readiness_result_identity":
            loaded["readiness"]["result_identity"],
        "manifest_identity": loaded["lock"]["manifest_identity"],
        "root_commitment": loaded["readiness"][
            "source_manifest"
        ].get("root_commitment"),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
        "open_command": loaded["commands"]["open"],
        "materialize_command": loaded["commands"]["materialize"],
        "execute_command": loaded["commands"]["execute"],
        "bounded_engine": "execute_training_engine_bounded",
        "opened_at": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if opened_at is None
            else opened_at
        ),
        "hostname": socket.gethostname() if hostname is None else hostname,
        "operational_audit": operational,
        "marker_only_open": True,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "scientific_work": 0,
        "passes": True,
    }
    written = write_immutable_json(
        paths["marker"],
        marker,
        field="activation_marker_payload_sha256",
    )
    state = {
        name: paths[name].exists()
        for name in (
            "marker",
            "manifest",
            "owner",
            "reservation",
            "consumption",
            "commit_head",
            "result",
            "checkpoint",
        )
    }
    expected = {
        "marker": True,
        "manifest": False,
        "owner": False,
        "reservation": False,
        "consumption": False,
        "commit_head": False,
        "result": False,
        "checkpoint": False,
    }
    if state != expected:
        raise J1bSurfaceIntegrityError(
            "J1b open created work beyond the immutable marker"
        )
    return {
        "marker": written,
        "marker_identity": immutable_json_identity(
            paths["marker"],
            payload_field="activation_marker_payload_sha256",
        ),
        "created_after_open": state,
        "passes": True,
    }


def load_open_training_contract(
    *,
    execution_root: Path,
    readiness_dir: Path,
    require_manifest: bool,
) -> dict[str, Any]:
    loaded = load_training_phase_lock(
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    paths = loaded["paths"]
    marker = load_json(paths["marker"])
    if (
        not verify_payload_hash(
            marker,
            "activation_marker_payload_sha256",
        )
        or marker.get("phase_lock_identity")
        != loaded["lock_identity"]
        or marker.get("readiness_lock_identity")
        != loaded["readiness"]["lock_identity"]
        or marker.get("readiness_result_identity")
        != loaded["readiness"]["result_identity"]
        or marker.get("manifest_identity")
        != loaded["lock"]["manifest_identity"]
        or marker.get("execute_command")
        != loaded["commands"]["execute"]
        or marker.get("runner_file_sha256") != sha256_path(RUNNER_PATH)
        or marker.get("parent_bounded_engine_file_sha256")
        != EXPECTED_PARENT_ENGINE_SHA256
    ):
        raise J1bSurfaceIntegrityError(
            "J1b open marker changed its immutable contract"
        )
    marker_identity = immutable_json_identity(
        paths["marker"],
        payload_field="activation_marker_payload_sha256",
    )
    manifest = None
    materialized_identity = None
    if require_manifest:
        manifest = load_json(paths["manifest"])
        if manifest != loaded["expected_manifest"]:
            raise J1bSurfaceIntegrityError(
                "J1b materialized manifest changed sealed rows"
            )
        materialized_identity = {
            **manifest_identity(manifest),
            "path": str(paths["manifest"].resolve()),
            "file_sha256": sha256_path(paths["manifest"]),
        }
    return {
        **loaded,
        "marker": marker,
        "marker_identity": marker_identity,
        "manifest": manifest,
        "materialized_manifest_identity": materialized_identity,
    }


def materialize_training_manifest(
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    loaded = load_open_training_contract(
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        require_manifest=False,
    )
    paths = loaded["paths"]
    if paths["manifest"].exists():
        raise FileExistsError(
            f"J1b manifest already exists: {paths['manifest']}"
        )
    for name in (
        "owner",
        "reservation",
        "consumption",
        "commit_head",
        "result",
        "checkpoint",
    ):
        if paths[name].exists():
            raise J1bSurfaceIntegrityError(
                "J1b manifest materialization followed scientific work"
            )
    written = write_immutable_json(
        paths["manifest"],
        {
            key: value
            for key, value in loaded["expected_manifest"].items()
            if key != "root_manifest_payload_sha256"
        },
        field="root_manifest_payload_sha256",
    )
    if written != loaded["expected_manifest"]:
        raise J1bSurfaceIntegrityError(
            "J1b manifest materialization changed bytes"
        )
    return {
        "manifest": written,
        "manifest_identity": {
            **manifest_identity(written),
            "path": str(paths["manifest"].resolve()),
            "file_sha256": sha256_path(paths["manifest"]),
        },
        "streams_reserved": 0,
        "streams_consumed": 0,
        "scientific_work": 0,
        "passes": True,
    }


def _stream_inventory(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(manifest["rows"])
    roles: dict[str, Any] = {}
    all_ids: list[int] = []
    for field in STREAM_RANGES:
        values = [int(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise J1bSurfaceIntegrityError(
                f"J1b manifest duplicated {field}"
            )
        roles[field] = {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "ordered_sha256": canonical_json_hash(values),
        }
        all_ids.extend(values)
    checks = {
        "four_roles": len(roles) == 4,
        "all_ids_unique": len(all_ids) == len(set(all_ids)),
        "one_id_per_role_per_root": len(all_ids) == len(rows) * 4,
        "manifest_valid": manifest.get("passes") is True,
    }
    if not all(checks.values()):
        raise J1bSurfaceIntegrityError(
            "J1b stream inventory is invalid"
        )
    return {
        "row_count": len(rows),
        "stream_id_count": len(all_ids),
        "roles": roles,
        "all_stream_ids_sha256": canonical_json_hash(sorted(all_ids)),
        "checks": checks,
        "passes": True,
    }


def _seal_stream_reservation(
    *,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    paths = loaded["paths"]
    inventory = _stream_inventory(loaded["manifest"])
    source_lock = loaded["readiness"]["lock"]
    payload = {
        "version": f"{VERSION}_training_stream_reservation_v1",
        "phase": "training",
        "execution_mode": loaded["readiness"]["mode"],
        "phase_lock_identity": loaded["lock_identity"],
        "marker_identity": loaded["marker_identity"],
        "manifest_identity":
            loaded["materialized_manifest_identity"],
        "source_manifest_identity":
            loaded["readiness"]["source_manifest_identity"],
        "protected_denylist_identity": source_lock.get(
            "authoritative_j1b_preflight_artifacts",
            {},
        ).get("denylist"),
        "stream_inventory": inventory,
        "collision_contract": {
            "accepted_j1b_preflight_lock":
                loaded["readiness"]["lock"].get(
                    "authoritative_j1b_preflight_lock_identity"
                ),
            "accepted_j1b_input_binding_sha256":
                loaded["readiness"]["lock"].get(
                    "authoritative_input_identities_sha256"
                ),
            "exact_fresh_manifest": True,
            "no_regeneration": True,
            "no_substitution": True,
            "historical_collision_count": 0,
        },
        "execute_command": loaded["commands"]["execute"],
        "decision": "RESERVED_J1B_TRAINING_STREAMS",
        "streams_reserved": inventory["stream_id_count"],
        "streams_consumed": 0,
        "scientific_work_before_reservation": 0,
        "passes": True,
    }
    written = write_immutable_json(
        paths["reservation"],
        payload,
        field="stream_reservation_payload_sha256",
        allow_existing_exact=True,
    )
    return {
        "reservation": written,
        "identity": immutable_json_identity(
            paths["reservation"],
            payload_field="stream_reservation_payload_sha256",
        ),
        "passes": True,
    }


def _owner_is_ancestor(
    ledger: Mapping[str, Any],
    *,
    opener: str,
    current: str,
) -> bool:
    owners = {
        str(row.get("owner_record_sha256", ""))
        for row in ledger.get("owners", [])
    }
    if opener not in owners or current not in owners:
        return False
    links = {
        str(row.get("old_owner_sha256", "")):
            str(row.get("new_owner_sha256", ""))
        for row in ledger.get("recoveries", [])
    }
    cursor = opener
    seen: set[str] = set()
    while cursor != current:
        if cursor in seen or cursor not in links:
            return False
        seen.add(cursor)
        cursor = links[cursor]
    return True


def _seal_stream_consumption(
    *,
    loaded: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    paths = loaded["paths"]
    reservation = load_json(paths["reservation"])
    if (
        not verify_payload_hash(
            reservation,
            "stream_reservation_payload_sha256",
        )
        or reservation.get("passes") is not True
    ):
        raise J1bSurfaceIntegrityError(
            "J1b stream reservation is invalid"
        )
    owner = owner_audit["owner"]
    current_owner = str(owner["owner_record_sha256"])
    reservation_identity = immutable_json_identity(
        paths["reservation"],
        payload_field="stream_reservation_payload_sha256",
    )
    if paths["consumption"].exists():
        existing = load_json(paths["consumption"])
        if not verify_payload_hash(
            existing,
            "stream_consumption_payload_sha256",
        ):
            raise J1bSurfaceIntegrityError(
                "Existing J1b stream consumption is invalid"
            )
        opener = str(existing.get("owner_record_sha256", ""))
        checks = {
            "phase_lock": existing.get("phase_lock_identity")
            == loaded["lock_identity"],
            "marker": existing.get("marker_identity")
            == loaded["marker_identity"],
            "manifest": existing.get("manifest_identity")
            == loaded["materialized_manifest_identity"],
            "reservation": existing.get("reservation_identity")
            == reservation_identity,
            "command": existing.get("execute_command")
            == loaded["commands"]["execute"],
            "counts": (
                existing.get("streams_reserved")
                == reservation["streams_reserved"]
                and existing.get("streams_consumed")
                == reservation["streams_reserved"]
            ),
            "owner_ancestry": _owner_is_ancestor(
                owner_audit["ledger"],
                opener=opener,
                current=current_owner,
            ),
        }
        if not all(checks.values()):
            raise J1bSurfaceIntegrityError(
                "Recovered J1b stream consumption changed contract"
            )
        return {
            "consumption": existing,
            "identity": immutable_json_identity(
                paths["consumption"],
                payload_field="stream_consumption_payload_sha256",
            ),
            "opener_owner_record_sha256": opener,
            "current_owner_record_sha256": current_owner,
            "owner_recovery_chain_verified": True,
            "reused_existing_record": True,
            "passes": True,
        }
    payload = {
        "version": f"{VERSION}_training_stream_consumption_opened_v1",
        "phase": "training",
        "execution_mode": loaded["readiness"]["mode"],
        "phase_lock_identity": loaded["lock_identity"],
        "marker_identity": loaded["marker_identity"],
        "manifest_identity":
            loaded["materialized_manifest_identity"],
        "reservation_identity": reservation_identity,
        "owner_record_sha256": current_owner,
        "execute_command": loaded["commands"]["execute"],
        "consumption_scope": "exact full immutable J1b training manifest",
        "stream_inventory": reservation["stream_inventory"],
        "streams_reserved": reservation["streams_reserved"],
        "streams_consumed": reservation["streams_reserved"],
        "scientific_work_before_consumption_record": 0,
        "decision": "OPENED_J1B_TRAINING_STREAM_CONSUMPTION",
        "passes": True,
    }
    written = write_immutable_json(
        paths["consumption"],
        payload,
        field="stream_consumption_payload_sha256",
    )
    return {
        "consumption": written,
        "identity": immutable_json_identity(
            paths["consumption"],
            payload_field="stream_consumption_payload_sha256",
        ),
        "opener_owner_record_sha256": current_owner,
        "current_owner_record_sha256": current_owner,
        "owner_recovery_chain_verified": True,
        "reused_existing_record": False,
        "passes": True,
    }


def _acquire_or_reclaim_owner(
    *,
    parent: Any,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    paths = loaded["paths"]
    phase_dir = paths["phase_dir"]
    predecessor = (
        sha256_path(paths["commit_head"])
        if paths["commit_head"].is_file()
        else None
    )
    mode = loaded["readiness"]["mode"]
    if not paths["owner"].exists():
        parent.acquire_writer_owner(
            phase_dir=phase_dir,
            phase="training",
            marker_file_sha256=
                loaded["marker_identity"]["file_sha256"],
            phase_lock_file_sha256=
                loaded["lock_identity"]["file_sha256"],
            command=loaded["commands"]["execute"],
            predecessor_commit_head_sha256=predecessor,
            execution_mode=mode,
        )
    else:
        ledger = parent.load_json(paths["owner"])
        if not parent._verify_ownership_ledger(ledger):
            raise J1bSurfaceIntegrityError(
                "J1b ownership ledger is malformed"
            )
        head = ledger["owners"][-1]
        if int(head.get("pid", -1)) != os.getpid():
            if parent._pid_alive(int(head.get("pid", -1))):
                raise J1bSurfaceOperationalHold(
                    "A live J1b writer owns the training phase"
                )
            parent.reclaim_dead_writer_owner(
                phase_dir=phase_dir,
                phase="training",
                marker_file_sha256=
                    loaded["marker_identity"]["file_sha256"],
                phase_lock_file_sha256=
                    loaded["lock_identity"]["file_sha256"],
                command=loaded["commands"]["execute"],
                execution_mode=mode,
                contention_audit=(
                    {"passes": True, "fixture_only": True}
                    if mode == "miniature_fixture"
                    else None
                ),
            )
    return parent.verify_writer_owner(
        phase_dir=phase_dir,
        phase="training",
        marker_file_sha256=loaded["marker_identity"]["file_sha256"],
        phase_lock_file_sha256=loaded["lock_identity"]["file_sha256"],
        command=loaded["commands"]["execute"],
        execution_mode=mode,
    )


def _runtime_entrypoint(
    *,
    phase_dir: Path,
    after_guard: Any,
    fixture_guard: bool,
    fixture_guard_passes: bool = True,
) -> dict[str, Any]:
    from threes_rl import j1b_operational_repair_preflight as repair

    def model_initializer(parent: Any) -> tuple[Any, Any]:
        model, optimizer = parent.j1.initialize_model_optimizer()
        if (
            parent.j1.parameter_count(model) != EXPECTED_PARAMETER_COUNT
            or parent.j1.model_schema_sha256()
            != EXPECTED_MODEL_SCHEMA_SHA256
        ):
            raise J1bSurfaceIntegrityError(
                "Frozen J1 model identity changed"
            )
        parent.FrozenMinibatchUpdater._validate_optimizer_binding(
            model,
            optimizer,
        )
        parent.j1.assert_finite_model(model)
        return model, optimizer

    operational_audit = None
    if fixture_guard:
        operational_audit = (
            lambda parent, directory:
                (
                    parent.fixture_phase_operational_audit(
                        phase_dir=directory,
                        phase="training",
                        active_seconds=0.0,
                        require_target_disk=True,
                    )
                    if fixture_guard_passes
                    else {
                        "passes": False,
                        "fixture_forced_guard_failure": True,
                    }
                )
        )
    try:
        return repair.guarded_runtime_entrypoint(
            phase_dir=phase_dir,
            model_initializer=model_initializer,
            operational_audit=operational_audit,
            after_guard=after_guard,
        )
    except repair.J1bOperationalHold as error:
        raise J1bSurfaceOperationalHold(str(error)) from error


def _terminal_base(
    *,
    loaded: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
    engine_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "version": f"{VERSION}_training_terminal_result_v1",
        "phase": "training",
        "execution_mode": loaded["readiness"]["mode"],
        "scientific_authority": (
            loaded["readiness"]["mode"] == "scientific"
        ),
        "fixture_only": (
            loaded["readiness"]["mode"] == "miniature_fixture"
        ),
        "phase_lock_identity": loaded["lock_identity"],
        "phase_lock_result_identity": immutable_json_identity(
            loaded["paths"]["lock_result"],
            payload_field="phase_lock_result_payload_sha256",
            decision=_phase_ready_decision(),
        ),
        "marker_identity": loaded["marker_identity"],
        "manifest_identity":
            loaded["materialized_manifest_identity"],
        "readiness_lock_identity":
            loaded["readiness"]["lock_identity"],
        "readiness_result_identity":
            loaded["readiness"]["result_identity"],
        "source_manifest_identity":
            loaded["readiness"]["source_manifest_identity"],
        "stream_reservation_identity": dict(reservation_identity),
        "stream_consumption_identity": dict(consumption_identity),
        "ownership_ledger_identity": immutable_json_identity(
            loaded["paths"]["owner"],
            payload_field="ownership_payload_sha256",
        ),
        "owner_record_sha256": owner_audit["owner"][
            "owner_record_sha256"
        ],
        "owner_recovery_count": len(
            owner_audit["ledger"].get("recoveries", [])
        ),
        "bounded_engine": "execute_training_engine_bounded",
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
        "wrapper_runner_file_sha256": sha256_path(RUNNER_PATH),
        "wrapper_charter_file_sha256": sha256_path(CHARTER_PATH),
        "wrapper_test_file_sha256": sha256_path(TEST_PATH),
        "execute_command": loaded["commands"]["execute"],
        "resource_clock": (
            None
            if engine_result is None
            else engine_result.get("resource_clock")
        ),
        "output_accounting": (
            None
            if engine_result is None
            else engine_result.get("output_accounting")
        ),
        "commit_store_metrics": (
            None
            if engine_result is None
            else engine_result.get("commit_store_metrics")
        ),
        "rolling_store_metrics": (
            None
            if engine_result is None
            else engine_result.get("rolling_store_metrics")
        ),
        "runtime_ledger_metrics": (
            None
            if engine_result is None
            else engine_result.get("runtime_ledger_metrics")
        ),
        "io_metrics": (
            None
            if engine_result is None
            else engine_result.get("io_metrics")
        ),
        "incumbent_changed": False,
        "dashboard_changed": False,
        "promote": False,
        "development_opened": False,
        "confirmation_opened": False,
        "human_session_reads": 0,
    }


def _seal_scientific_training_terminal(
    *,
    parent: Any,
    loaded: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if engine_result.get("completed") is not True:
        raise J1bSurfaceIntegrityError(
            "J1b training engine did not complete"
        )
    state = engine_result["state"]
    boundary = engine_result["boundary"]
    if (
        state.get("engine_stage") != "complete"
        or int(state.get("round_number", -1)) != parent.ROUNDS
        or boundary.get("chain_audit", {}).get("passes") is not True
    ):
        raise J1bSurfaceIntegrityError(
            "J1b training terminal boundary is incomplete"
        )
    manifest = loaded["manifest"]
    model, optimizer = parent._load_model_optimizer_from_runtime(state)
    training_input = {
        "manifest_payload_sha256": manifest[
            "root_manifest_payload_sha256"
        ],
        "marker_file_sha256": loaded["marker_identity"][
            "file_sha256"
        ],
        "terminal_state_file_sha256": boundary["state_file_sha256"],
        "terminal_commit_head_payload_sha256": boundary[
            "commit_head_payload_sha256"
        ],
        "completed_root_ids_sha256": parent.j1.stable_hash(
            state["all_completed_root_ids"]
        ),
        "optimizer_step_ids_sha256": parent.j1.stable_hash(
            state["optimizer_step_ids"]
        ),
        "round_aggregates_sha256": parent.j1.stable_hash(
            state["round_aggregates"]
        ),
        "wrapper_runner_file_sha256": sha256_path(RUNNER_PATH),
        "j1b_readiness_result_payload_sha256":
            loaded["readiness"]["result"][
                "readiness_result_payload_sha256"
            ],
    }
    checkpoint_payload = parent.candidate_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        training_manifest_identity=parent.root_manifest_identity(
            manifest
        ),
        training_marker_file_sha256=
            loaded["marker_identity"]["file_sha256"],
        training_result_input_sha256=canonical_json_hash(
            training_input
        ),
    )
    checkpoint_identity = parent.write_candidate_checkpoint(
        loaded["paths"]["checkpoint"],
        checkpoint_payload,
    )
    checkpoint_identity["training_state_file_sha256"] = boundary[
        "state_file_sha256"
    ]
    report = {
        "manifest_root_ids": [
            str(row["root_id"]) for row in manifest["rows"]
        ],
        "completed_root_ids": list(state["all_completed_root_ids"]),
        "expected_optimizer_step_ids": list(
            state["expected_optimizer_step_ids"]
        ),
        "closed_optimizer_step_ids": list(
            state["optimizer_step_ids"]
        ),
        "rounds": list(state["round_aggregates"]),
        "authenticated_terminal_boundary": {
            "passes": boundary["passes"],
            "chain_audit_passes": boundary["chain_audit"]["passes"],
            "state_file_sha256": boundary["state_file_sha256"],
            "commit_head_file_sha256": boundary[
                "commit_head_file_sha256"
            ],
            "commit_head_payload_sha256": boundary[
                "commit_head_payload_sha256"
            ],
            "unit_ids_sha256": boundary["chain_audit"][
                "unit_ids_sha256"
            ],
        },
        "checkpoint_identity": checkpoint_identity,
    }
    sanity = parent.training_sanity_decision(report)
    sanity_payload = {
        **sanity,
        "training_input": training_input,
        "training_report_sha256": parent.j1.stable_hash(report),
        "checkpoint_authoritative": (
            sanity["decision"] == "READY_J1_TRAINING_SANITY"
        ),
        "checkpoint_quarantined": (
            sanity["decision"] != "READY_J1_TRAINING_SANITY"
        ),
        "wrapper_runner_file_sha256": sha256_path(RUNNER_PATH),
    }
    write_immutable_json(
        loaded["paths"]["sanity"],
        sanity_payload,
        field="training_sanity_payload_sha256",
        allow_existing_exact=True,
    )
    base = _terminal_base(
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=engine_result,
    )
    base.update(
        {
            "decision": sanity["decision"],
            "training_sanity_identity": immutable_json_identity(
                loaded["paths"]["sanity"],
                payload_field="training_sanity_payload_sha256",
                decision=sanity["decision"],
            ),
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_authoritative":
                sanity_payload["checkpoint_authoritative"],
            "checkpoint_quarantined":
                sanity_payload["checkpoint_quarantined"],
            "authenticated_terminal_boundary": report[
                "authenticated_terminal_boundary"
            ],
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
        allow_existing_exact=True,
    )


def _seal_fixture_training_terminal(
    *,
    loaded: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if engine_result.get("completed") is not True:
        raise J1bSurfaceIntegrityError(
            "Miniature bounded engine did not complete"
        )
    state = engine_result["state"]
    boundary = engine_result["boundary"]
    base = _terminal_base(
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=engine_result,
    )
    base.update(
        {
            "decision": "READY_J1B_MINIATURE_TRAINING_FIXTURE",
            "scientific_authority": False,
            "fixture_only": True,
            "checkpoint_authoritative": False,
            "checkpoint_quarantined": True,
            "completed_root_ids": list(
                state.get("all_completed_root_ids", [])
            ),
            "optimizer_step_ids": list(
                state.get("optimizer_step_ids", [])
            ),
            "round_aggregates_sha256": canonical_json_hash(
                state.get("round_aggregates", [])
            ),
            "terminal_state_file_sha256": boundary.get(
                "state_file_sha256"
            ),
            "terminal_commit_head_payload_sha256": boundary.get(
                "commit_head_payload_sha256"
            ),
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
        allow_existing_exact=True,
    )


def _seal_failure_terminal(
    *,
    loaded: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    operational = isinstance(
        error,
        J1bSurfaceOperationalHold,
    ) or error.__class__.__name__ in {
        "J1ExecutionOperationalHold",
    }
    decision = (
        "HOLD_J1B_OPERATIONAL"
        if operational
        else "KILL_J1B_TRAINING_INTEGRITY"
    )
    base = _terminal_base(
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=None,
    )
    base.update(
        {
            "decision": decision,
            "failure_class": (
                "operational" if operational else "integrity"
            ),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "partial_work_preserved": True,
            "checkpoint_authoritative": False,
            "checkpoint_quarantined": True,
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
        allow_existing_exact=True,
    )


def _load_terminal_result(path: Path) -> dict[str, Any]:
    result = load_json(path)
    if (
        not verify_payload_hash(
            result,
            "terminal_result_payload_sha256",
        )
        or result.get("wrapper_runner_file_sha256")
        != sha256_path(RUNNER_PATH)
        or result.get("wrapper_charter_file_sha256")
        != sha256_path(CHARTER_PATH)
        or result.get("parent_bounded_engine_file_sha256")
        != EXPECTED_PARENT_ENGINE_SHA256
        or result.get("bounded_engine")
        != "execute_training_engine_bounded"
        or result.get("promote") is not False
    ):
        raise J1bSurfaceIntegrityError(
            "Existing J1b terminal result is invalid"
        )
    return result


def _finalize_retention(
    *,
    parent: Any,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    result = _load_terminal_result(loaded["paths"]["result"])
    retention = parent.seal_phase_retention_manifest(
        execution_root=Path(
            loaded["lock"]["future_execution_root"]
        ),
        phase="training",
    )
    observed = immutable_json_identity(
        loaded["paths"]["retention"],
        payload_field="retention_payload_sha256",
    )
    if retention.get("retention_payload_sha256") != observed[
        "payload_sha256"
    ]:
        raise J1bSurfaceIntegrityError(
            "J1b retention seal changed after write"
        )
    return {
        "result": result,
        "result_identity": immutable_json_identity(
            loaded["paths"]["result"],
            payload_field="terminal_result_payload_sha256",
            decision=str(result["decision"]),
        ),
        "retention": retention,
        "retention_identity": observed,
        "passes": True,
    }


def execute_training_from_artifacts(
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    loaded = load_open_training_contract(
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        require_manifest=True,
    )
    mode = loaded["readiness"]["mode"]
    fixture_env_names = (
        "J1B_FIXTURE_RUNTIME_FAILURE",
        "J1B_FIXTURE_FIRST_GUARD_FAILURE",
        "J1B_FIXTURE_PRE_ENGINE_INTERRUPT",
        "J1B_FIXTURE_INTERRUPT_AFTER_BOUNDARY",
    )
    if mode == "scientific" and any(
        os.environ.get(name) for name in fixture_env_names
    ):
        raise J1bSurfaceIntegrityError(
            "Scientific execution rejects fixture controls"
        )
    if (
        mode == "miniature_fixture"
        and os.environ.get("J1B_FIXTURE_RUNTIME_FAILURE") == "1"
    ):
        raise J1bSurfaceOperationalHold(
            "Fixture runtime configuration failure before parent import"
        )

    def after_guard(
        parent: Any,
        initial_model: Any,
        _initial_optimizer: Any,
    ) -> dict[str, Any]:
        initial_state_sha256 = parent.j1.stable_hash(
            initial_model.state_dict()
        )
        if initial_state_sha256 != EXPECTED_INITIAL_MODEL_STATE_SHA256:
            raise J1bSurfaceIntegrityError(
                "Frozen initial J1 model state changed"
            )
        if loaded["paths"]["result"].exists():
            finalized = _finalize_retention(
                parent=parent,
                loaded=loaded,
            )
            return {
                "terminal": finalized,
                "terminal_already_sealed": True,
                "initial_model_state_sha256":
                    initial_state_sha256,
                "passes": True,
            }

        owner_audit = _acquire_or_reclaim_owner(
            parent=parent,
            loaded=loaded,
        )
        pre_engine_interrupt = (
            os.environ.get("J1B_FIXTURE_PRE_ENGINE_INTERRUPT")
            if mode == "miniature_fixture"
            else None
        )
        if pre_engine_interrupt == "after-owner":
            raise parent.J1ExecutionPlannedInterruption(
                "Fixture interruption after owner"
            )
        reservation = _seal_stream_reservation(loaded=loaded)
        if pre_engine_interrupt == "after-reservation":
            raise parent.J1ExecutionPlannedInterruption(
                "Fixture interruption after reservation"
            )
        consumption = _seal_stream_consumption(
            loaded=loaded,
            owner_audit=owner_audit,
        )
        if pre_engine_interrupt == "after-consumption":
            raise parent.J1ExecutionPlannedInterruption(
                "Fixture interruption after consumption"
            )

        if mode == "scientific":
            config = parent.TrainingEngineConfig()
            operational_audit_fn = None
            interrupt_after_boundary = None
        else:
            fixture = dict(
                loaded["readiness"]["lock"].get(
                    "fixture_engine_config",
                    {},
                )
            )
            config = parent.TrainingEngineConfig(
                rounds=int(fixture["rounds"]),
                roots_per_round=int(fixture["roots_per_round"]),
                env_count=int(fixture["env_count"]),
                minibatch_size=int(fixture["minibatch_size"]),
                max_moves=int(fixture["max_moves"]),
                execution_mode="miniature_fixture",
            )
            operational_audit_fn = parent.fixture_phase_operational_audit
            interrupt_after_boundary = os.environ.get(
                "J1B_FIXTURE_INTERRUPT_AFTER_BOUNDARY"
            )
        try:
            engine_result = parent.execute_training_engine_bounded(
                rows=loaded["manifest"]["rows"],
                phase_dir=loaded["paths"]["phase_dir"],
                marker_file_sha256=
                    loaded["marker_identity"]["file_sha256"],
                marker_payload_sha256=
                    loaded["marker_identity"]["payload_sha256"],
                phase_lock_file_sha256=
                    loaded["lock_identity"]["file_sha256"],
                manifest_file_sha256=
                    loaded["materialized_manifest_identity"][
                        "file_sha256"
                    ],
                manifest_payload_sha256=
                    loaded["materialized_manifest_identity"][
                        "payload_sha256"
                    ],
                command=loaded["commands"]["execute"],
                config=config,
                interrupt_after_boundary=interrupt_after_boundary,
                operational_audit_fn=operational_audit_fn,
            )
            if mode == "scientific":
                terminal = _seal_scientific_training_terminal(
                    parent=parent,
                    loaded=loaded,
                    engine_result=engine_result,
                    reservation_identity=reservation["identity"],
                    consumption_identity=consumption["identity"],
                    owner_audit=owner_audit,
                )
            else:
                terminal = _seal_fixture_training_terminal(
                    loaded=loaded,
                    engine_result=engine_result,
                    reservation_identity=reservation["identity"],
                    consumption_identity=consumption["identity"],
                    owner_audit=owner_audit,
                )
        except parent.J1ExecutionPlannedInterruption:
            raise
        except BaseException as error:
            terminal = _seal_failure_terminal(
                loaded=loaded,
                reservation_identity=reservation["identity"],
                consumption_identity=consumption["identity"],
                owner_audit=owner_audit,
                error=error,
            )
        finalized = _finalize_retention(
            parent=parent,
            loaded=loaded,
        )
        return {
            "terminal": finalized,
            "terminal_decision": terminal["decision"],
            "terminal_already_sealed": False,
            "initial_model_state_sha256": initial_state_sha256,
            "passes": True,
        }

    fixture_guard_failure = (
        mode == "miniature_fixture"
        and (
            os.environ.get("J1B_FIXTURE_FIRST_GUARD_FAILURE") == "1"
            or loaded["readiness"]["lock"].get(
                "fixture_first_guard_passes",
                True,
            )
            is not True
        )
    )
    runtime = _runtime_entrypoint(
        phase_dir=loaded["paths"]["phase_dir"],
        after_guard=after_guard,
        fixture_guard=mode == "miniature_fixture",
        fixture_guard_passes=not fixture_guard_failure,
    )
    after = runtime["after_guard"]
    return {
        "version": f"{VERSION}_execute_result_v1",
        "phase": "training",
        "execution_mode": mode,
        "runtime": runtime["runtime"],
        "first_operational_guard": runtime["operational_audit"],
        "ordering": runtime["ordering"],
        "terminal_decision": after["terminal"]["result"]["decision"],
        "terminal_result_identity":
            after["terminal"]["result_identity"],
        "retention_identity": after["terminal"]["retention_identity"],
        "terminal_already_sealed":
            after["terminal_already_sealed"],
        "initial_model_state_sha256":
            after["initial_model_state_sha256"],
        "development_opened": False,
        "confirmation_opened": False,
        "promote": False,
        "passes": True,
    }


def zero_work_counters() -> dict[str, int]:
    return {
        "j1b_training_phase_locks": 0,
        "j1b_training_markers": 0,
        "j1b_materialized_manifests": 0,
        "j1b_owners": 0,
        "j1b_streams_reserved": 0,
        "j1b_streams_consumed": 0,
        "j1b_genesis_commits": 0,
        "normal_start_games": 0,
        "scientific_transitions": 0,
        "scientific_labels": 0,
        "scientific_optimizer_steps": 0,
        "scientific_checkpoints": 0,
        "development_reads": 0,
        "confirmation_reads": 0,
        "policy_or_score_outcomes": 0,
        "human_session_reads": 0,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
        "promotion_actions": 0,
    }


def surface_schema() -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_schema_v1",
        "public_commands": list(PUBLIC_COMMANDS),
        "phase": "training",
        "development_surface_present": False,
        "confirmation_surface_present": False,
        "promotion_surface_present": False,
        "bounded_engine": "execute_training_engine_bounded",
        "legacy_engine_scientific_reachable": False,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_schema_sha256": EXPECTED_MODEL_SCHEMA_SHA256,
        "initial_model_state_sha256":
            EXPECTED_INITIAL_MODEL_STATE_SHA256,
        "root_count": TRAIN_ROOTS,
        "rounds": 64,
        "roots_per_round": 256,
        "env_count": 16,
        "epochs_per_round": 4,
        "minibatch_size": 4_096,
        "starter_tile": None,
        "runtime_contract": {
            "torch_interop_threads": 1,
            "torch_intraop_threads": 1,
            "deterministic_algorithms": True,
            "configuration_before_parent_import": True,
            "first_parent_guard_before_owner": True,
            "first_parent_guard_before_stream_reservation": True,
            "first_parent_guard_before_stream_consumption": True,
            "first_parent_guard_before_genesis": True,
        },
        "fresh_stream_ranges": {
            field: {
                "start": start,
                "end_inclusive": end,
                "rows": end - start + 1,
            }
            for field, (start, end) in STREAM_RANGES.items()
        },
        "terminal_decisions": [
            "READY_J1_TRAINING_SANITY",
            "HOLD_J1_LEARNING_SANITY",
            "HOLD_J1B_OPERATIONAL",
            "KILL_J1B_TRAINING_INTEGRITY",
        ],
        "promote": False,
    }
    return payload_with_hash(payload, "schema_payload_sha256")


def runtime_storage_projection() -> dict[str, Any]:
    parent_path = (
        RUNS_ROOT
        / "forensics"
        / "j1_execution_surface_readiness_v1"
        / "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json"
    )
    _assert_file_hash(
        parent_path,
        "92dfc49a8f0830a4b39c627d9257e4a20b4ca504019c455b3b2b1eb05a959f20",
    )
    parent = load_json(parent_path)
    if (
        not verify_payload_hash(parent, "projection_payload_sha256")
        or parent.get("passes") is not True
    ):
        raise J1bSurfaceIntegrityError(
            "Accepted parent projection is invalid"
        )
    central = parent["training"]["central"]
    parent_before_margin = int(
        central["storage"]["projected_before_margin_bytes"]
    )
    wrapper_storage_bytes = 32 * 1024**2
    before_margin = parent_before_margin + wrapper_storage_bytes
    with_margin = int(round(before_margin * 1.25))
    cap_bytes = 24 * 1024**3
    parent_runtime_before_margin_hours = (
        float(central["hours_with_25pct_margin"]) / 1.25
    )
    wrapper_runtime_seconds = 120.0
    runtime_with_margin_hours = (
        parent_runtime_before_margin_hours
        + wrapper_runtime_seconds / 3600.0
    ) * 1.25
    created_files = int(
        central["bounded_io"]["created_files"]
    ) + 16
    fsync_count = int(
        central["bounded_io"]["fsync_count"]
    ) + 32
    checks = {
        "parent_projection_file_exact": True,
        "parent_projection_payload_valid": True,
        "same_16384_root_workload": (
            int(central["bounded_io"]["transition_rows"])
            == 8_388_608
        ),
        "fixed_32_tick_cadence": (
            int(
                central["bounded_io"][
                    "fixed_collection_tick_cadence"
                ]
            )
            == 32
        ),
        "same_max_replay_ticks": (
            int(
                central["bounded_io"][
                    "maximum_replayed_collection_ticks"
                ]
            )
            == 32
        ),
        "same_retirement_contract": (
            parent["retirement_contract"][
                "transition_chunks_current_round_only"
            ]
            and parent["retirement_contract"][
                "round_batch_current_round_only"
            ]
            and parent["retirement_contract"][
                "idempotent_crash_window_recovery"
            ]
        ),
        "storage_with_margin_at_most_24gib": (
            with_margin <= cap_bytes
        ),
        "runtime_with_margin_at_most_72h": (
            runtime_with_margin_hours <= 72.0
        ),
        "file_count_within_50000": created_files <= 50_000,
        "fsync_count_within_200000": fsync_count <= 200_000,
        "sensitivity_reported": (
            parent["training"].get("sensitivity_5000_moves")
            is not None
        ),
        "sensitivity_not_conjunctive": (
            parent["training"]["sensitivity_5000_moves"].get(
                "diagnostic_not_conjunctive"
            )
            is True
        ),
        "fixed_25pct_margin": True,
    }
    payload = {
        "version": f"{VERSION}_runtime_storage_projection_v1",
        "method": (
            "accepted parent bounded-engine projection plus a fixed "
            "32-MiB J1b wrapper artifact envelope and 120-second "
            "fresh-process orchestration envelope; no retiming"
        ),
        "parent_projection_identity": {
            "path": str(parent_path.resolve()),
            "file_sha256": sha256_path(parent_path),
            "payload_sha256": parent["projection_payload_sha256"],
        },
        "training": {
            "root_count": TRAIN_ROOTS,
            "central_moves": 512,
            "parent_projected_before_margin_bytes":
                parent_before_margin,
            "wrapper_storage_envelope_bytes":
                wrapper_storage_bytes,
            "projected_before_margin_bytes": before_margin,
            "safety_multiplier": 1.25,
            "projected_with_margin_bytes": with_margin,
            "projected_with_margin_gib": with_margin / 1024**3,
            "storage_cap_bytes": cap_bytes,
            "storage_cap_gib": 24.0,
            "parent_runtime_before_margin_hours":
                parent_runtime_before_margin_hours,
            "wrapper_runtime_envelope_seconds":
                wrapper_runtime_seconds,
            "runtime_with_margin_hours":
                runtime_with_margin_hours,
            "runtime_cap_hours": 72.0,
            "created_files": created_files,
            "created_file_cap": 50_000,
            "fsync_count": fsync_count,
            "fsync_cap": 200_000,
            "current_round_chunks_only": True,
            "current_round_batch_only": True,
            "three_rolling_slots_or_orphan_envelope": True,
            "retirement_recovery": True,
            "bounded_abandoned_unit_charge": True,
        },
        "sensitivity_5000_moves": parent["training"][
            "sensitivity_5000_moves"
        ],
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": zero_work_counters(),
    }
    return payload_with_hash(payload, "projection_payload_sha256")


def write_test_evidence(
    *,
    readiness_dir: Path,
    commands: Sequence[Mapping[str, Any]],
    documented_deselections: Sequence[str],
    independent_reproduction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if FUTURE_EXECUTION_ROOT.exists():
        raise J1bSurfaceIntegrityError(
            "Future J1b execution root exists before test evidence"
        )
    expected_kinds = {
        "py_compile",
        "focused_j1b_surface",
        "parent_j1b_preflight",
        "parent_j1_execution_surface",
        "parent_j1_joint_policy_value",
        "parent_j1a_cost_power",
        "applicable_non_science_regressions",
        "miniature_full_chain",
    }
    observed_kinds = {str(row.get("kind")) for row in commands}
    checks = {
        "charter_exact": sha256_path(CHARTER_PATH)
        == EXPECTED_CHARTER_SHA256,
        "runner_present": RUNNER_PATH.is_file(),
        "tests_present": TEST_PATH.is_file(),
        "command_kinds_exact": observed_kinds == expected_kinds,
        "all_commands_passed": all(
            row.get("passed") is True
            and int(row.get("returncode", -1)) == 0
            for row in commands
        ),
        "future_execution_root_absent":
            not FUTURE_EXECUTION_ROOT.exists(),
        "no_scientific_work": all(
            value == 0 for value in zero_work_counters().values()
        ),
    }
    payload = {
        "version": f"{VERSION}_test_evidence_v1",
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "source_identities": {
            "charter_file_sha256": sha256_path(CHARTER_PATH),
            "runner_file_sha256": sha256_path(RUNNER_PATH),
            "test_file_sha256": sha256_path(TEST_PATH),
        },
        "commands": [dict(row) for row in commands],
        "documented_historical_artifact_state_deselections":
            list(documented_deselections),
        "independent_reproduction": (
            None
            if independent_reproduction is None
            else dict(independent_reproduction)
        ),
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": zero_work_counters(),
    }
    if not payload["passes"]:
        raise J1bSurfaceIntegrityError(
            "J1b training-surface test evidence gates failed"
        )
    return write_immutable_json(
        readiness_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def seal_readiness_package(
    *,
    readiness_dir: Path,
    operational_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if readiness_dir.resolve() != READINESS_DIR.resolve():
        raise J1bSurfaceIntegrityError(
            "Scientific readiness namespace changed"
        )
    if FUTURE_EXECUTION_ROOT.exists():
        raise J1bSurfaceIntegrityError(
            "Future J1b execution root exists before readiness seal"
        )
    paths = _readiness_paths(readiness_dir)
    if not paths["test_evidence"].is_file():
        raise J1bSurfaceIntegrityError(
            "J1b test evidence must precede readiness"
        )
    for name in ("schema", "projection", "input_bindings", "lock", "result"):
        if paths[name].exists():
            raise FileExistsError(
                f"J1b readiness artifact already exists: {paths[name]}"
            )
    input_audit = audit_authoritative_inputs(
        require_future_execution_absent=True,
    )
    input_bindings = write_immutable_json(
        paths["input_bindings"],
        {
            "version": f"{VERSION}_input_bindings_v1",
            "authoritative_input_audit": input_audit,
            "future_execution_root": str(
                FUTURE_EXECUTION_ROOT.resolve()
            ),
            "future_execution_root_absent": True,
            "protected_parent_artifacts_parsed_for_identity_only": True,
            "human_session_reads": 0,
            "passes": True,
        },
        field="input_bindings_payload_sha256",
    )
    schema = write_immutable_json(
        paths["schema"],
        {
            key: value
            for key, value in surface_schema().items()
            if key != "schema_payload_sha256"
        },
        field="schema_payload_sha256",
    )
    projection = runtime_storage_projection()
    write_immutable_json(
        paths["projection"],
        {
            key: value
            for key, value in projection.items()
            if key != "projection_payload_sha256"
        },
        field="projection_payload_sha256",
    )
    if operational_audit.get("passes") is not True:
        raise J1bSurfaceOperationalHold(
            "J1b readiness operational audit failed"
        )
    source_identity = immutable_json_identity(
        _source_manifest_path(),
        payload_field="prospective_manifest_payload_sha256",
    )
    artifacts = {
        "test_evidence": immutable_json_identity(
            paths["test_evidence"],
            payload_field="test_evidence_payload_sha256",
        ),
        "schema": immutable_json_identity(
            paths["schema"],
            payload_field="schema_payload_sha256",
        ),
        "projection": immutable_json_identity(
            paths["projection"],
            payload_field="projection_payload_sha256",
        ),
        "input_bindings": immutable_json_identity(
            paths["input_bindings"],
            payload_field="input_bindings_payload_sha256",
        ),
    }
    source_lock = load_json(
        J1B_PREFLIGHT_DIR / "J1B_READINESS_LOCK.json"
    )
    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "decision": READY_DECISION,
        "execution_mode": "scientific",
        "scientific_authority": True,
        "fixture_only": False,
        "future_execution_root": str(
            FUTURE_EXECUTION_ROOT.resolve()
        ),
        "future_execution_root_absent": True,
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "artifacts": artifacts,
        "source_manifest_identity": source_identity,
        "authoritative_input_identities_sha256":
            input_audit["identities_sha256"],
        "authoritative_j1b_preflight_lock_identity":
            input_audit["identities"]["j1b_preflight_lock"],
        "authoritative_j1b_preflight_result_identity":
            input_audit["identities"]["j1b_preflight_result"],
        "authoritative_j1b_preflight_artifacts":
            source_lock["artifacts"],
        "pre_a1_historical_evidence":
            input_audit["identities"]["pre_a1_history"],
        "spent_j1_execution_identities":
            input_audit["identities"]["spent_j1_execution"],
        "parent_source_identities":
            input_audit["identities"]["parent_sources"],
        "parent_readiness_identities":
            input_audit["identities"]["parent_readiness"],
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
        "manifest_contract": {
            "row_count": TRAIN_ROOTS,
            "canonical_rows_sha256":
                EXPECTED_CANONICAL_ROWS_SHA256,
            "root_commitment_payload_sha256":
                EXPECTED_ROOT_COMMITMENT_PAYLOAD_SHA256,
            "root_set_sha256": EXPECTED_ROOT_SET_SHA256,
            "stream_ranges": {
                field: {
                    "start": start,
                    "end_inclusive": end,
                }
                for field, (start, end) in STREAM_RANGES.items()
            },
        },
        "public_commands": list(PUBLIC_COMMANDS),
        "bounded_engine": "execute_training_engine_bounded",
        "operational_audit": dict(operational_audit),
        "zero_work": zero_work_counters(),
        "passes": True,
    }
    lock = write_immutable_json(
        paths["lock"],
        lock_payload,
        field="readiness_lock_payload_sha256",
    )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="readiness_lock_payload_sha256",
        decision=READY_DECISION,
    )
    result = write_immutable_json(
        paths["result"],
        {
            "version": f"{VERSION}_readiness_result_v1",
            "decision": READY_DECISION,
            "readiness_lock_identity": lock_identity,
            "artifacts": artifacts,
            "operational_audit": dict(operational_audit),
            "integrity_checks": {
                "source_and_history_exact": True,
                "fresh_manifest_exact": True,
                "runtime_order_repaired": True,
                "bounded_engine_only": True,
                "test_evidence_passes": True,
                "projection_passes": projection["passes"],
                "future_execution_root_absent": True,
                "zero_work": True,
            },
            "continue": "research-lead review",
            "hold": "all J1b scientific training execution",
            "kill": "historical kills unchanged; J1b not killed",
            "promote": False,
            "zero_work": zero_work_counters(),
            "passes": True,
        },
        field="readiness_result_payload_sha256",
    )
    return {
        "lock": lock,
        "lock_identity": lock_identity,
        "result": result,
        "result_identity": immutable_json_identity(
            paths["result"],
            payload_field="readiness_result_payload_sha256",
            decision=READY_DECISION,
        ),
        "artifacts": artifacts,
        "passes": True,
    }


def write_fixture_readiness(
    *,
    readiness_dir: Path,
    execution_root: Path,
    rows: Sequence[Mapping[str, Any]],
    engine_config: Mapping[str, int],
    first_guard_passes: bool = True,
) -> dict[str, Any]:
    readiness_dir.mkdir(parents=True, exist_ok=True)
    root_ids = [str(row["root_id"]) for row in rows]
    commitment = payload_with_hash(
        {
            "version": f"{VERSION}_fixture_root_commitment_v1",
            "phase": "training",
            "partition": "train",
            "row_count": len(rows),
        },
        "marker_payload_sha256",
    )
    source = payload_with_hash(
        {
            "version": f"{VERSION}_fixture_source_manifest_v1",
            "phase": "training",
            "partition": "train",
            "root_commitment": commitment,
            "rows": [dict(row) for row in rows],
            "canonical_rows_sha256": ordered_rows_hash(rows),
            "checks": {"fixture": True},
            "passes": True,
        },
        "prospective_manifest_payload_sha256",
    )
    source_path = readiness_dir / "FIXTURE_SOURCE_MANIFEST.json"
    write_immutable_json(
        source_path,
        {
            key: value
            for key, value in source.items()
            if key != "prospective_manifest_payload_sha256"
        },
        field="prospective_manifest_payload_sha256",
    )
    generic_artifacts = {
        "test_evidence": (
            TEST_EVIDENCE_NAME,
            "test_evidence_payload_sha256",
            {"fixture": True, "passes": True},
        ),
        "schema": (
            SCHEMA_NAME,
            "schema_payload_sha256",
            {
                "fixture": True,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "passes": True,
            },
        ),
        "projection": (
            PROJECTION_NAME,
            "projection_payload_sha256",
            {"fixture": True, "passes": True},
        ),
        "input_bindings": (
            INPUT_BINDINGS_NAME,
            "input_bindings_payload_sha256",
            {"fixture": True, "passes": True},
        ),
    }
    artifacts = {}
    for name, (filename, field, payload) in generic_artifacts.items():
        path = readiness_dir / filename
        write_immutable_json(path, payload, field=field)
        artifacts[name] = immutable_json_identity(
            path,
            payload_field=field,
        )
    lock = write_immutable_json(
        readiness_dir / READINESS_LOCK_NAME,
        {
            "version": f"{VERSION}_fixture_readiness_lock_v1",
            "decision": READY_DECISION,
            "execution_mode": "miniature_fixture",
            "scientific_authority": False,
            "fixture_only": True,
            "future_execution_root": str(execution_root.resolve()),
            "charter_file_sha256": sha256_path(CHARTER_PATH),
            "runner_file_sha256": sha256_path(RUNNER_PATH),
            "test_file_sha256": sha256_path(TEST_PATH),
            "artifacts": artifacts,
            "source_manifest_identity": immutable_json_identity(
                source_path,
                payload_field="prospective_manifest_payload_sha256",
            ),
            "fixture_engine_config": dict(engine_config),
            "fixture_first_guard_passes": first_guard_passes,
            "fixture_root_set_sha256": canonical_json_hash(root_ids),
            "public_commands": list(PUBLIC_COMMANDS),
            "passes": True,
        },
        field="readiness_lock_payload_sha256",
    )
    lock_identity = immutable_json_identity(
        readiness_dir / READINESS_LOCK_NAME,
        payload_field="readiness_lock_payload_sha256",
        decision=READY_DECISION,
    )
    result = write_immutable_json(
        readiness_dir / READINESS_RESULT_NAME,
        {
            "version": f"{VERSION}_fixture_readiness_result_v1",
            "decision": READY_DECISION,
            "readiness_lock_identity": lock_identity,
            "fixture_only": True,
            "scientific_authority": False,
            "passes": True,
        },
        field="readiness_result_payload_sha256",
    )
    return {
        "lock": lock,
        "result": result,
        "lock_identity": lock_identity,
        "passes": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="J1b training-only production dispatcher",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
    )
    for command in PUBLIC_COMMANDS:
        child = subparsers.add_parser(command)
        child.add_argument(
            "--execution-root",
            type=Path,
            required=True,
        )
        child.add_argument(
            "--readiness-dir",
            type=Path,
            required=True,
        )
        child.add_argument("--jobs", type=int, required=True)
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.jobs != 1:
        raise J1bSurfaceIntegrityError("J1b jobs must equal one")
    execution_root = args.execution_root.resolve()
    readiness_dir = args.readiness_dir.resolve()
    if args.subcommand == "seal-phase-lock":
        return seal_training_phase_lock(
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    if args.subcommand == "open":
        return open_training_phase(
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    if args.subcommand == "materialize":
        return materialize_training_manifest(
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    if args.subcommand == "execute":
        return execute_training_from_artifacts(
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    raise J1bSurfaceIntegrityError(
        f"Unsupported J1b command: {args.subcommand}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except BaseException as error:
        payload = {
            "version": f"{VERSION}_command_failure_v1",
            "command": args.subcommand,
            "error_type": type(error).__name__,
            "error": str(error),
            "passes": False,
        }
        print(json.dumps(payload, sort_keys=True))
        if error.__class__.__name__ == "J1ExecutionPlannedInterruption":
            return 75
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""External zero-work terminalization for the spent J1b open attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Iterable, Mapping, Sequence


VERSION = "j1b_open_failure_terminalize_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
CHARTER_PATH = (
    REPO_ROOT / "threes_rl" / "J1B_OPEN_FAILURE_TERMINALIZATION_CHARTER.md"
)
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = REPO_ROOT / "tests" / "test_rl_j1b_open_failure_terminalize.py"
ORIGINAL_ROOT = (
    REPO_ROOT / "threes_rl" / "runs" / "forensics" / "j1b_execution_v1"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "threes_rl"
    / "runs"
    / "forensics"
    / "j1b_open_failure_terminal_v1"
)

EXPECTED_ORIGINAL_FILES = {
    "training/phase_lock.json":
        "ac12b9f21977a3adcd61ef5f0d8ba60b058306dcc05fdfed423d2ca77c17a0ce",
    "training/phase_lock_result.json":
        "6a2f63dc8875db394333ac901a919466a6a432083e29feba32ba8917f3ee9bcf",
    "training/execution_opened.json":
        "e99099b87aa6417b4200ee236ef2b770d1524d11b26a878e9f3bf0d749a54cff",
}
EXPECTED_MARKER_PAYLOAD_SHA256 = (
    "c9e48e972a59f699627bfaa949854930"
    "672a8c45a6c671be591e175522a107e4"
)
EXPECTED_DECISION = "HOLD_J1B_OPEN_SERIALIZATION_INTEGRITY"

EVIDENCE_NAME = "J1B_OPEN_FAILURE_TEST_EVIDENCE.json"
TERMINAL_NAME = "J1B_OPEN_FAILURE_TERMINAL.json"
RETENTION_NAME = "J1B_OPEN_FAILURE_RETENTION.json"

ZERO_WORK = {
    "materialized_rows": 0,
    "materialized_manifests": 0,
    "owners": 0,
    "owner_recoveries": 0,
    "stream_reservations": 0,
    "stream_consumptions": 0,
    "commit_genesis": 0,
    "commits_after_genesis": 0,
    "completed_roots": 0,
    "active_roots": 0,
    "games": 0,
    "transitions": 0,
    "optimizer_steps": 0,
    "round_aggregates": 0,
    "checkpoints": 0,
    "policy_or_score_outcomes": 0,
    "development_reads": 0,
    "confirmation_reads": 0,
    "promotion_actions": 0,
    "incumbent_changes": 0,
    "dashboard_changes": 0,
    "human_session_reads": 0,
}


class J1bTerminalizationError(RuntimeError):
    """Fail-closed external terminalization error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def json_native(value: Any) -> Any:
    """Normalize a prospective payload through its exact JSON representation."""
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def canonical_json_bytes(value: Any) -> bytes:
    native = json_native(value)
    return json.dumps(
        native,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = dict(json_native(payload))
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(json_native(payload))
    observed = body.pop(field, None)
    return isinstance(observed, str) and observed == canonical_json_hash(body)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise J1bTerminalizationError(f"Expected JSON object: {path}")
    return value


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_immutable_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    field: str,
    allow_existing_exact: bool = False,
) -> dict[str, Any]:
    target = Path(path)
    body = payload_with_hash(payload, field)
    serialized = (
        json.dumps(body, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    reloaded = json.loads(serialized.decode("utf-8"))
    if reloaded != body or not verify_payload_hash(reloaded, field):
        raise J1bTerminalizationError(
            f"Prospective immutable JSON is unstable: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        observed = target.read_bytes()
        if observed != serialized:
            raise J1bTerminalizationError(
                f"Immutable artifact collision changed bytes: {target}"
            ) from error
        existing = json.loads(observed.decode("utf-8"))
        if not verify_payload_hash(existing, field):
            raise J1bTerminalizationError(
                f"Existing immutable artifact is invalid: {target}"
            ) from error
        if allow_existing_exact:
            return existing
        raise
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    observed_payload = load_json(target)
    if observed_payload != body or not verify_payload_hash(
        observed_payload, field
    ):
        raise J1bTerminalizationError(
            f"Written immutable artifact changed: {target}"
        )
    return observed_payload


def immutable_identity(
    path: str | Path,
    *,
    payload_field: str,
    decision: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    payload = load_json(target)
    if not verify_payload_hash(payload, payload_field):
        raise J1bTerminalizationError(
            f"Payload identity changed: {target}"
        )
    if decision is not None and payload.get("decision") != decision:
        raise J1bTerminalizationError(
            f"Decision identity changed: {target}"
        )
    return {
        "path": str(target.resolve()),
        "file_sha256": sha256_path(target),
        "payload_field": payload_field,
        "payload_sha256": payload[payload_field],
        **(
            {"decision": payload["decision"]}
            if decision is not None
            else {}
        ),
    }


def reproduce_tuple_list_defect() -> dict[str, Any]:
    prospective = {
        "operational_audit": {
            "services": {
                "dashboard": {
                    "top_three": (263670, 261369, 258561),
                }
            }
        }
    }
    serialized = json.dumps(prospective, sort_keys=True)
    reloaded = json.loads(serialized)
    checks = {
        "live_shape_uses_tuple": isinstance(
            prospective["operational_audit"]["services"]["dashboard"][
                "top_three"
            ],
            tuple,
        ),
        "json_reload_uses_list": isinstance(
            reloaded["operational_audit"]["services"]["dashboard"][
                "top_three"
            ],
            list,
        ),
        "raw_python_equality_fails": prospective != reloaded,
        "canonical_serialized_bytes_equal": (
            canonical_json_bytes(prospective)
            == canonical_json_bytes(reloaded)
        ),
        "json_native_normalization_equal": (
            json_native(prospective) == reloaded
        ),
    }
    return {
        "path": "operational_audit.services.dashboard.top_three",
        "prospective_type": "tuple",
        "reloaded_type": "list",
        "checks": checks,
        "passes": all(checks.values()),
    }


def audit_original_namespace(
    original_root: str | Path = ORIGINAL_ROOT,
    *,
    expected_files: Mapping[str, str] = EXPECTED_ORIGINAL_FILES,
    expected_marker_payload_sha256: str =
        EXPECTED_MARKER_PAYLOAD_SHA256,
) -> dict[str, Any]:
    root = Path(original_root)
    observed_paths = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )
    expected_paths = sorted(expected_files)
    identities = {
        relative: {
            "path": str((root / relative).resolve()),
            "file_sha256": sha256_path(root / relative),
        }
        for relative in expected_paths
        if (root / relative).is_file()
    }
    marker_path = root / "training" / "execution_opened.json"
    marker = load_json(marker_path) if marker_path.is_file() else {}
    marker_valid = (
        verify_payload_hash(marker, "activation_marker_payload_sha256")
        if marker
        else False
    )
    forbidden_names = (
        "root_manifest.json",
        "writer_owner.json",
        "stream_reservation.json",
        "stream_consumption_opened.json",
        "commit_head.json",
        "training_result.json",
        "round_64_candidate.pt",
        "retention.json",
    )
    forbidden_present = sorted(
        name
        for name in forbidden_names
        if (root / "training" / name).exists()
    )
    checks = {
        "root_exists": root.is_dir(),
        "exact_three_file_inventory": observed_paths == expected_paths,
        "all_file_hashes_exact": (
            len(identities) == len(expected_files)
            and all(
                identities[path]["file_sha256"] == expected_hash
                for path, expected_hash in expected_files.items()
            )
        ),
        "marker_payload_valid": marker_valid,
        "marker_payload_exact": (
            marker.get("activation_marker_payload_sha256")
            == expected_marker_payload_sha256
        ),
        "marker_is_zero_work": (
            marker.get("marker_only_open") is True
            and marker.get("streams_reserved") == 0
            and marker.get("streams_consumed") == 0
            and marker.get("scientific_work") == 0
        ),
        "forbidden_work_paths_absent": not forbidden_present,
    }
    return {
        "version": f"{VERSION}_original_namespace_audit_v1",
        "root": str(root.resolve()),
        "observed_paths": observed_paths,
        "expected_paths": expected_paths,
        "identities": identities,
        "marker_payload_sha256": marker.get(
            "activation_marker_payload_sha256"
        ),
        "forbidden_present": forbidden_present,
        "zero_work": dict(ZERO_WORK),
        "checks": checks,
        "passes": all(checks.values()),
    }


def write_test_evidence(
    *,
    out_dir: str | Path,
    commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = Path(out_dir)
    payload = {
        "version": f"{VERSION}_test_evidence_v1",
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "commands": [dict(json_native(command)) for command in commands],
        "all_commands_passed": bool(commands) and all(
            command.get("passes") is True for command in commands
        ),
        "scientific_work": 0,
        "passes": bool(commands) and all(
            command.get("passes") is True for command in commands
        ),
    }
    return write_immutable_json(
        output / EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def seal_external_terminal(
    *,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    original_root: str | Path = ORIGINAL_ROOT,
) -> dict[str, Any]:
    output = Path(out_dir)
    evidence_identity = immutable_identity(
        output / EVIDENCE_NAME,
        payload_field="test_evidence_payload_sha256",
    )
    evidence = load_json(output / EVIDENCE_NAME)
    source_checks = {
        "charter_exact": (
            evidence.get("charter_file_sha256")
            == sha256_path(CHARTER_PATH)
        ),
        "runner_exact": (
            evidence.get("runner_file_sha256")
            == sha256_path(RUNNER_PATH)
        ),
        "tests_exact": (
            evidence.get("test_file_sha256") == sha256_path(TEST_PATH)
        ),
        "tests_passed": evidence.get("passes") is True,
    }
    audit = audit_original_namespace(original_root)
    mechanism = reproduce_tuple_list_defect()
    checks = {
        **source_checks,
        "original_namespace_exact": audit["passes"],
        "defect_reproduced": mechanism["passes"],
        "zero_work_exact": all(value == 0 for value in ZERO_WORK.values()),
        "terminal_absent": not (output / TERMINAL_NAME).exists(),
        "retention_absent": not (output / RETENTION_NAME).exists(),
    }
    terminal_payload = {
        "version": f"{VERSION}_terminal_v1",
        "decision": EXPECTED_DECISION,
        "passes": all(checks.values()),
        "scientific_evidence": False,
        "j1_hypothesis_killed": False,
        "j1b_retry_authorized": False,
        "j1b_namespace_spent": True,
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "test_evidence_identity": evidence_identity,
        "original_namespace_audit": audit,
        "defect_mechanism": mechanism,
        "zero_work": dict(ZERO_WORK),
        "checks": checks,
        "sealed_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "hostname": socket.gethostname(),
        "status": {
            "CONTINUE": "J1c orchestration-only readiness",
            "HOLD": "all J1b retry and all J1/J1c science",
            "KILL": False,
            "PROMOTE": False,
        },
    }
    if not terminal_payload["passes"]:
        raise J1bTerminalizationError(
            "External J1b terminal checks did not all pass"
        )
    terminal = write_immutable_json(
        output / TERMINAL_NAME,
        terminal_payload,
        field="terminal_payload_sha256",
    )
    terminal_identity = immutable_identity(
        output / TERMINAL_NAME,
        payload_field="terminal_payload_sha256",
        decision=EXPECTED_DECISION,
    )
    retention_payload = {
        "version": f"{VERSION}_retention_v1",
        "decision": "PRESERVE_J1B_OPEN_FAILURE_EVIDENCE",
        "terminal_identity": terminal_identity,
        "test_evidence_identity": evidence_identity,
        "protected_original_files": audit["identities"],
        "protected_external_files": {
            EVIDENCE_NAME: evidence_identity,
            TERMINAL_NAME: terminal_identity,
        },
        "original_namespace_mutation_authorized": False,
        "cleanup_authorized": False,
        "passes": True,
    }
    retention = write_immutable_json(
        output / RETENTION_NAME,
        retention_payload,
        field="retention_payload_sha256",
    )
    return {
        "terminal": terminal,
        "terminal_identity": terminal_identity,
        "retention": retention,
        "retention_identity": immutable_identity(
            output / RETENTION_NAME,
            payload_field="retention_payload_sha256",
            decision="PRESERVE_J1B_OPEN_FAILURE_EVIDENCE",
        ),
        "passes": True,
    }


def _parse_recorded_command(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(
            "Recorded command must be a JSON object"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal the external zero-work J1b open failure"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    evidence.add_argument(
        "--recorded-command",
        action="append",
        type=_parse_recorded_command,
        required=True,
    )
    seal = subparsers.add_parser("seal")
    seal.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    seal.add_argument(
        "--original-root", type=Path, default=ORIGINAL_ROOT
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.subcommand == "write-test-evidence":
            result = write_test_evidence(
                out_dir=args.out_dir,
                commands=args.recorded_command,
            )
        else:
            result = seal_external_terminal(
                out_dir=args.out_dir,
                original_root=args.original_root,
            )
    except Exception as error:
        print(
            json.dumps(
                {
                    "version": f"{VERSION}_command_failure_v1",
                    "command": args.subcommand,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "passes": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

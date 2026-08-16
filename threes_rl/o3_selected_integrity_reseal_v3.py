"""V3 no-recompute integrity envelope for the sealed O3 selected roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from threes_rl import o3_selected_integrity_reseal_v2 as v2


VERSION = "o3_selected_integrity_reseal_v3"
READY = "READY_O3_OPTION_TRAINING_INTEGRITY_RESEALED_V3"
HOLD = "HOLD_O3_SELECTED_INTEGRITY_RESEAL_V3"

AMENDMENT_PATH = Path(
    "threes_rl/O3_SELECTED_INTEGRITY_RESEAL_AMENDMENT_V3.md"
)
RUNNER_PATH = Path("threes_rl/o3_selected_integrity_reseal_v3.py")
TEST_PATH = Path("tests/test_rl_o3_selected_integrity_reseal_v3.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/"
    "o3_selected_integrity_reseal_v3_test_evidence.json"
)
OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/o3_selected_integrity_reseal_v3"
)
ENVELOPE_PATH = OUTPUT_DIR / "O3_SELECTED_INTEGRITY_RESEAL_V3.json"

V2_BINDINGS = {
    "amendment": {
        "path": v2.AMENDMENT_PATH,
        "file_sha256":
            "380a13c9472d25edfe32a5b9f979c365514a125b9f74cb9ce98702413ffa78c8",
    },
    "runner": {
        "path": v2.RUNNER_PATH,
        "file_sha256":
            "4e4c17c9f059c3c4e1679bde0a3811cfd85bbca953bf491c7cbc9d9c06f5cab6",
    },
    "tests": {
        "path": v2.TEST_PATH,
        "file_sha256":
            "d681baa7e5f80de982f74849d05b6963292e7401bb9cd4070c69e8a54a790db6",
    },
    "terminal_envelope": {
        "path": v2.ENVELOPE_PATH,
        "file_sha256":
            "f466cae4e298edfc25499a90a78bfb6d6e037e2d065be72eb0de498cf9b31d57",
        "self_hash_field": "reseal_payload_sha256",
        "payload_sha256":
            "58b55acb66033092dad5e789421d4cb60adfe960ccf25e1a6ef277e81141357d",
    },
}
EXPECTED_V2_ERROR = (
    "[Errno 2] No such file or directory: "
    "'threes_rl/runs/forensics/"
    "o3_selected_integrity_reseal_v2_test_evidence.json'"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    embedded = body.pop(field, None)
    return isinstance(embedded, str) and embedded == canonical_json_hash(body)


def verify_v2_history(
    bindings: Mapping[str, Mapping[str, Any]] = V2_BINDINGS,
) -> dict[str, Any]:
    if set(bindings) != set(V2_BINDINGS):
        raise ValueError("V2 binding names mismatch")
    identities = {}
    for name, spec in bindings.items():
        path = Path(spec["path"])
        file_sha = sha256_path(path)
        if file_sha != spec["file_sha256"]:
            raise ValueError(f"V2 {name} file SHA mismatch")
        identity = {"path": str(path), "file_sha256": file_sha}
        if "self_hash_field" in spec:
            payload = json.loads(path.read_text())
            field = str(spec["self_hash_field"])
            if not verify_self_hash(payload, field):
                raise ValueError(f"V2 {name} self hash mismatch")
            if payload.get(field) != spec["payload_sha256"]:
                raise ValueError(f"V2 {name} payload SHA mismatch")
            if payload.get("decision") != v2.HOLD:
                raise ValueError("V2 terminal decision mismatch")
            if payload.get("error") != EXPECTED_V2_ERROR:
                raise ValueError("V2 terminal error mismatch")
            identity["payload_sha256"] = payload[field]
            identity["decision"] = payload["decision"]
            identity["error"] = payload["error"]
        identities[name] = identity
    return {
        "identities": identities,
        "v2_hold_preserved": True,
        "v2_test_evidence_absent": not v2.TEST_EVIDENCE_PATH.exists(),
    }


def build_test_evidence_payload(
    *,
    focused_tests_passed: int,
    regression_tests_passed: int,
    recorded_commands: Sequence[str],
) -> dict[str, Any]:
    if focused_tests_passed < 1 or regression_tests_passed < 1:
        raise ValueError("Passing test counts are required")
    if not recorded_commands:
        raise ValueError("At least one recorded command is required")
    payload = {
        "version": f"{VERSION}_test_evidence",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "amendment_sha256": sha256_path(AMENDMENT_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "focused_tests_passed": int(focused_tests_passed),
        "regression_tests_passed": int(regression_tests_passed),
        "recorded_commands": list(recorded_commands),
        "v2_history": verify_v2_history(),
        "zero_new_work": {
            "games": 0,
            "streams": 0,
            "labels": 0,
            "models": 0,
            "rollouts": 0,
            "candidate_actions": 0,
            "scores_or_max_tiles_inspected": 0,
            "policy_outcomes": 0,
        },
    }
    return payload_with_hash(payload, "test_evidence_payload_sha256")


def _atomic_write_file_once(
    path: Path,
    payload: Mapping[str, Any],
    self_hash_field: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    reloaded = json.loads(serialized)
    if not verify_self_hash(reloaded, self_hash_field):
        raise ValueError("Artifact fails JSON reload stability")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary exists: {temporary}")
    temporary.write_text(serialized + "\n")
    os.replace(temporary, path)
    written = json.loads(path.read_text())
    if not verify_self_hash(written, self_hash_field):
        raise ValueError("Written artifact self hash mismatch")


def write_test_evidence(
    *,
    focused_tests_passed: int,
    regression_tests_passed: int,
    recorded_commands: Sequence[str],
    path: Path = TEST_EVIDENCE_PATH,
) -> dict[str, Any]:
    payload = build_test_evidence_payload(
        focused_tests_passed=focused_tests_passed,
        regression_tests_passed=regression_tests_passed,
        recorded_commands=recorded_commands,
    )
    _atomic_write_file_once(
        path,
        payload,
        "test_evidence_payload_sha256",
    )
    return {
        "artifact": "test_evidence",
        "path": str(path),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload["test_evidence_payload_sha256"],
    }


def verify_test_evidence(path: Path = TEST_EVIDENCE_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise ValueError("V3 test evidence self hash mismatch")
    expected = {
        "amendment_sha256": sha256_path(AMENDMENT_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("V3 test evidence source identity mismatch")
    if payload.get("focused_tests_passed", 0) < 1:
        raise ValueError("V3 focused tests missing")
    if payload.get("regression_tests_passed", 0) < 1:
        raise ValueError("V3 regression tests missing")
    if set(payload.get("zero_new_work", {}).values()) != {0}:
        raise ValueError("V3 test evidence records forbidden work")
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload["test_evidence_payload_sha256"],
        "focused_tests_passed": payload["focused_tests_passed"],
        "regression_tests_passed": payload["regression_tests_passed"],
        "recorded_commands": list(payload["recorded_commands"]),
    }


def build_ready_envelope(
    evidence_path: Path = TEST_EVIDENCE_PATH,
) -> dict[str, Any]:
    v2_history = verify_v2_history()
    verified = v2.verify_frozen_inputs()
    proof = verified["coercion_proof"]
    if proof["post_json_sha256"] != v2.EXPECTED_SELECTED_POST_JSON_SHA256:
        raise ValueError("V3 selected post-JSON SHA mismatch")
    if (
        proof["pre_serialization_reproduction_sha256"]
        != v2.EXPECTED_INPUTS["selected"]["payload_sha256"]
    ):
        raise ValueError("V3 selected pre-serialization SHA mismatch")
    if set(proof["numeric_string_paths"]) != {
        ".".join(path) for path in v2.EXPECTED_COERCION_PATHS
    }:
        raise ValueError("V3 six-path coercion proof mismatch")

    payload = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": READY,
        "continue": True,
        "hold": False,
        "kill": False,
        "promote": False,
        "historical_holds": {
            "original_acquisition": "HOLD_O3_ACQUISITION_INTEGRITY",
            "recovery": "HOLD_O3_ACQUISITION_RECOVERY_INTEGRITY",
            "v2_reseal": v2.HOLD,
            "status": "preserved_immutable_and_authoritative",
        },
        "bindings": {
            "amendment": {
                "path": str(AMENDMENT_PATH),
                "sha256": sha256_path(AMENDMENT_PATH),
            },
            "runner": {
                "path": str(RUNNER_PATH),
                "sha256": sha256_path(RUNNER_PATH),
            },
            "tests": {
                "path": str(TEST_PATH),
                "sha256": sha256_path(TEST_PATH),
            },
            "test_evidence": verify_test_evidence(evidence_path),
            "v2_history": v2_history,
        },
        "input_identities": verified["identities"],
        "scientific_facts": verified["facts"],
        "selected_scientific_payload": proof["post_json_body"],
        "selected_post_json_scientific_payload_sha256":
            proof["post_json_sha256"],
        "selected_pre_serialization_reproduction_sha256":
            proof["pre_serialization_reproduction_sha256"],
        "serialization_proof": {
            "numeric_string_paths": proof["numeric_string_paths"],
            "path_proof": proof["path_proof"],
            "only_six_maps_coerced": True,
            "all_other_fields_unchanged": True,
            "defect_exhausted_by_json_key_coercion": True,
        },
        "reseal_scope": {
            "recovery_json_inputs_read": [
                str(v2.UNION_PATH),
                str(v2.SUPPORT_PATH),
                str(v2.SELECTED_PATH),
                str(v2.RESULT_PATH),
            ],
            "replay_files_read": 0,
            "support_candidates_recomputed": False,
            "allocation_recomputed": False,
            "new_scientific_work": False,
        },
        "zero_new_work": {
            "games": 0,
            "streams": 0,
            "labels": 0,
            "models": 0,
            "rollouts": 0,
            "candidate_actions": 0,
            "scores_or_max_tiles_inspected": 0,
            "policy_outcomes": 0,
            "incumbent_changes": 0,
            "dashboard_changes": 0,
        },
    }
    return payload_with_hash(payload, "v3_reseal_payload_sha256")


def build_hold_envelope(error: Exception) -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": HOLD,
        "continue": False,
        "hold": True,
        "kill": False,
        "promote": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "historical_holds": {
            "original_acquisition": "HOLD_O3_ACQUISITION_INTEGRITY",
            "recovery": "HOLD_O3_ACQUISITION_RECOVERY_INTEGRITY",
            "v2_reseal": v2.HOLD,
            "status": "preserved_immutable_and_authoritative",
        },
        "zero_new_work": {
            "games": 0,
            "streams": 0,
            "labels": 0,
            "models": 0,
            "rollouts": 0,
            "candidate_actions": 0,
            "scores_or_max_tiles_inspected": 0,
            "policy_outcomes": 0,
            "incumbent_changes": 0,
            "dashboard_changes": 0,
        },
    }
    return payload_with_hash(payload, "v3_reseal_payload_sha256")


def seal(
    *,
    output_path: Path = ENVELOPE_PATH,
    evidence_path: Path = TEST_EVIDENCE_PATH,
) -> dict[str, Any]:
    if output_path.exists() or output_path.parent.exists():
        raise FileExistsError(
            f"V3 terminal namespace already exists: {output_path.parent}"
        )
    try:
        payload = build_ready_envelope(evidence_path)
    except Exception as error:
        payload = build_hold_envelope(error)
    _atomic_write_file_once(output_path, payload, "v3_reseal_payload_sha256")
    return {
        "artifact": "terminal_envelope",
        "path": str(output_path),
        "file_sha256": sha256_path(output_path),
        "payload_sha256": payload["v3_reseal_payload_sha256"],
        "decision": payload["decision"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--focused", type=int, required=True)
    evidence.add_argument("--regressions", type=int, required=True)
    evidence.add_argument(
        "--recorded-command",
        dest="recorded_commands",
        action="append",
        required=True,
    )

    subparsers.add_parser("seal")
    return parser


def dispatch(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.subcommand == "write-test-evidence":
        return write_test_evidence(
            focused_tests_passed=args.focused,
            regression_tests_passed=args.regressions,
            recorded_commands=args.recorded_commands,
        )
    if args.subcommand == "seal":
        return seal()
    raise AssertionError(f"Unhandled subcommand: {args.subcommand}")


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(dispatch(argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

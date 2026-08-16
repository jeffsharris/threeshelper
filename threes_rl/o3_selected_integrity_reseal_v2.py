"""No-recompute integrity envelope for the sealed O3 selected-root artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "o3_selected_integrity_reseal_v2"
READY = "READY_O3_OPTION_TRAINING_INTEGRITY_RESEALED"
HOLD = "HOLD_O3_SELECTED_INTEGRITY_RESEAL"

RECOVERY_DIR = Path(
    "threes_rl/runs/forensics/o3_event_acquisition_recovery_v1"
)
UNION_PATH = RECOVERY_DIR / "O3_RECOVERY_UNION_MANIFEST.json"
SUPPORT_PATH = RECOVERY_DIR / "O3_RECOVERY_SUPPORT_SCAN.json"
SELECTED_PATH = RECOVERY_DIR / "O3_RECOVERY_SELECTED_ROOTS.json"
RESULT_PATH = RECOVERY_DIR / "O3_RECOVERY_RESULT.json"

AMENDMENT_PATH = Path(
    "threes_rl/O3_SELECTED_INTEGRITY_RESEAL_AMENDMENT_V2.md"
)
RUNNER_PATH = Path("threes_rl/o3_selected_integrity_reseal_v2.py")
TEST_PATH = Path("tests/test_rl_o3_selected_integrity_reseal_v2.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/"
    "o3_selected_integrity_reseal_v2_test_evidence.json"
)
OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/o3_selected_integrity_reseal_v2"
)
ENVELOPE_PATH = OUTPUT_DIR / "O3_SELECTED_INTEGRITY_RESEAL_V2.json"

EXPECTED_INPUTS = {
    "union": {
        "path": UNION_PATH,
        "file_sha256":
            "02ea2c5be8823de775f56b7267f9c8371d26efc53897115b25733f8ef4527311",
        "self_hash_field": "union_payload_sha256",
        "payload_sha256":
            "cec88701a1754f1064d639dae09cd6856ee18ce9399865338ebed7107f672d94",
    },
    "support": {
        "path": SUPPORT_PATH,
        "file_sha256":
            "4c71513e6a3a2778bb8d1db0ba08f8ff5a1f0d6edc82ee1208b7458593059d27",
        "self_hash_field": "support_payload_sha256",
        "payload_sha256":
            "27ae3a6aca5f1de71ee18df193c0663a83579d3aeba65cd864065cfff594e25a",
    },
    "selected": {
        "path": SELECTED_PATH,
        "file_sha256":
            "9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049",
        "self_hash_field": "selected_payload_sha256",
        "payload_sha256":
            "c6c8b1a35cc63f4c1c1fdc98579f1ae0859a84c5eef7203306000223ac9c61a5",
    },
    "result": {
        "path": RESULT_PATH,
        "file_sha256":
            "962da52b83b8746c006a9ef5fbe1fdd34f43e9c7bf97d9b6ff48f2a42019c23a",
        "self_hash_field": "result_payload_sha256",
        "payload_sha256":
            "a679d512d6ce44bf5fd4ecd8249d15625c59f342e64796a6d5eb894396224ad0",
    },
}

EXPECTED_SELECTED_POST_JSON_SHA256 = (
    "d9600cf420d947826c812b88225633b78a889f94f94ce39270dd71bc11b12f0e"
)
EXPECTED_RESULT_ERROR = (
    "Artifact self hash mismatch: "
    "threes_rl/runs/forensics/o3_event_acquisition_recovery_v1/"
    "O3_RECOVERY_SELECTED_ROOTS.json"
)
ROLES = ("train", "development", "untouched_mechanism")
COUNT_FIELDS = ("target_counts", "descriptive_stage_counts")
EXPECTED_COERCION_PATHS = frozenset(
    ("per_role", role, field)
    for role in ROLES
    for field in COUNT_FIELDS
)
EXPECTED_ROLE_COUNTS = {
    "train": 96,
    "development": 32,
    "untouched_mechanism": 192,
}
EXPECTED_UNION_ROLE_COUNTS = {
    "train": 5020,
    "development": 1675,
    "untouched_mechanism": 13805,
}
EXPECTED_FAMILIES = {
    "o3_corner2",
    "o3_expectimax2",
    "o3_parent_mc1000",
    "o3_qd_v2",
    "o3_replaycal",
}


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


def verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    embedded = body.pop(field, None)
    return isinstance(embedded, str) and embedded == canonical_json_hash(body)


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def _numeric_string_key_paths(
    value: Any,
    path: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    if isinstance(value, dict):
        if any(isinstance(key, str) and key.isdigit() for key in value):
            found.add(path)
        for key, child in value.items():
            found.update(_numeric_string_key_paths(child, path + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(
                _numeric_string_key_paths(child, path + (f"[{index}]",))
            )
    return found


def _mapping_at_path(
    payload: Mapping[str, Any],
    path: Sequence[str],
) -> dict[Any, Any]:
    value: Any = payload
    for component in path:
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"Missing coercion path: {'.'.join(path)}")
        value = value[component]
    if not isinstance(value, dict):
        raise ValueError(f"Coercion path is not a mapping: {'.'.join(path)}")
    return value


def prove_selected_key_coercion(
    selected_payload: Mapping[str, Any],
) -> dict[str, Any]:
    embedded = selected_payload.get("selected_payload_sha256")
    if embedded != EXPECTED_INPUTS["selected"]["payload_sha256"]:
        raise ValueError("Selected embedded payload SHA mismatch")

    post_json_body = copy.deepcopy(dict(selected_payload))
    post_json_body.pop("selected_payload_sha256", None)
    post_json_sha = canonical_json_hash(post_json_body)
    if post_json_sha != EXPECTED_SELECTED_POST_JSON_SHA256:
        raise ValueError("Selected post-JSON canonical SHA mismatch")

    numeric_paths = _numeric_string_key_paths(post_json_body)
    if numeric_paths != EXPECTED_COERCION_PATHS:
        missing = sorted(EXPECTED_COERCION_PATHS - numeric_paths)
        extra = sorted(numeric_paths - EXPECTED_COERCION_PATHS)
        raise ValueError(
            f"Unexpected numeric-string key paths; missing={missing}, extra={extra}"
        )

    reconstructed = copy.deepcopy(post_json_body)
    path_proof = {}
    for path in sorted(EXPECTED_COERCION_PATHS):
        mapping = _mapping_at_path(reconstructed, path)
        if not mapping:
            raise ValueError(f"Empty coercion mapping: {'.'.join(path)}")
        if not all(isinstance(key, str) and key.isdigit() for key in mapping):
            raise ValueError(f"Nonnumeric coercion key: {'.'.join(path)}")
        converted = {int(key): value for key, value in mapping.items()}
        if len(converted) != len(mapping):
            raise ValueError(f"Coercion key collision: {'.'.join(path)}")
        mapping.clear()
        mapping.update(converted)
        path_proof[".".join(path)] = {
            "post_json_keys": sorted(
                (str(key) for key in converted),
                key=int,
            ),
            "restored_integer_keys": sorted(converted),
        }

    reconstructed_sha = canonical_json_hash(reconstructed)
    if reconstructed_sha != embedded:
        raise ValueError("Selected pre-serialization SHA was not reproduced")

    return {
        "post_json_body": post_json_body,
        "post_json_sha256": post_json_sha,
        "pre_serialization_reproduction_sha256": reconstructed_sha,
        "numeric_string_paths": sorted(
            ".".join(path) for path in numeric_paths
        ),
        "path_proof": path_proof,
        "only_six_maps_coerced": True,
    }


def _load_and_verify_input(
    name: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(spec["path"])
    file_sha = sha256_path(path)
    if file_sha != spec["file_sha256"]:
        raise ValueError(f"{name} file SHA mismatch")
    payload = json.loads(path.read_text())
    embedded = payload.get(str(spec["self_hash_field"]))
    if embedded != spec["payload_sha256"]:
        raise ValueError(f"{name} embedded payload SHA mismatch")
    identity = {
        "path": str(path),
        "file_sha256": file_sha,
        "embedded_payload_sha256": embedded,
    }
    return payload, identity


def _verify_union(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not verify_self_hash(payload, "union_payload_sha256"):
        raise ValueError("Union self hash mismatch")
    checks = payload.get("checks")
    family_counts = payload.get("family_counts")
    membership = payload.get("membership")
    required = {
        "exact_20500",
        "exact_4100_per_family",
        "exact_p0_membership",
        "unique_ancestries",
        "unique_replay_hashes",
        "zero_role_drift",
        "zero_stream_drift",
    }
    if payload.get("passes") is not True:
        raise ValueError("Union passes is not true")
    if not isinstance(checks, dict) or not all(checks.get(key) for key in required):
        raise ValueError("Union scientific checks failed")
    if not isinstance(membership, list) or len(membership) != 20500:
        raise ValueError("Union membership count mismatch")
    if payload.get("role_counts") != EXPECTED_UNION_ROLE_COUNTS:
        raise ValueError("Union role counts mismatch")
    if (
        not isinstance(family_counts, dict)
        or set(family_counts) != EXPECTED_FAMILIES
        or set(family_counts.values()) != {4100}
    ):
        raise ValueError("Union family counts mismatch")
    if payload.get("unique_ancestries") != 20500:
        raise ValueError("Union ancestry count mismatch")
    if payload.get("unique_replay_hashes") != 20500:
        raise ValueError("Union replay hash count mismatch")
    if payload.get("role_drift") not in ([], {}):
        raise ValueError("Union role drift is nonempty")
    if payload.get("stream_drift") not in ([], {}):
        raise ValueError("Union stream drift is nonempty")
    return {
        "passes": True,
        "membership_count": len(membership),
        "family_counts": dict(family_counts),
        "role_counts": dict(payload["role_counts"]),
        "unique_ancestries": payload["unique_ancestries"],
        "unique_replay_hashes": payload["unique_replay_hashes"],
        "zero_role_drift": True,
        "zero_stream_drift": True,
    }


def _verify_support(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not verify_self_hash(payload, "support_payload_sha256"):
        raise ValueError("Support self hash mismatch")
    audit = payload.get("audit")
    candidate_rows = payload.get("candidate_rows")
    if not isinstance(audit, dict) or audit.get("passes") is not True:
        raise ValueError("Support audit does not pass")
    if audit.get("candidate_rows") != 12922:
        raise ValueError("Support candidate count mismatch")
    if audit.get("candidate_roots") != 7607:
        raise ValueError("Support root count mismatch")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != 12922:
        raise ValueError("Support candidate row payload mismatch")
    if audit.get("stage_used_only_descriptively") is not True:
        raise ValueError("Support stage semantics mismatch")
    if audit.get("recorded_actions_accessed") is not False:
        raise ValueError("Support recorded-action access mismatch")
    if (
        audit.get("final_or_future_milestone_or_max_fields_accessed")
        is not False
    ):
        raise ValueError("Support outcome-field access mismatch")
    return {
        "passes": True,
        "candidate_rows": 12922,
        "candidate_roots": 7607,
        "stage_descriptive_only": True,
    }


def _verify_selected(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("passes") is not True:
        raise ValueError("Selected payload does not pass")
    if payload.get("deficits") != []:
        raise ValueError("Selected deficits are nonempty")
    checks = payload.get("checks")
    selected = payload.get("selected")
    per_role = payload.get("per_role")
    if not isinstance(checks, dict) or not all(checks.values()):
        raise ValueError("Selected top-level checks failed")
    if not isinstance(selected, list) or len(selected) != 320:
        raise ValueError("Selected total count mismatch")
    if not isinstance(per_role, dict) or set(per_role) != set(ROLES):
        raise ValueError("Selected roles mismatch")

    aggregate = {}
    for role, expected_count in EXPECTED_ROLE_COUNTS.items():
        report = per_role[role]
        selected_count = report.get("selected")
        family_counts = report.get("family_counts")
        role_checks = report.get("checks")
        if report.get("passes") is not True:
            raise ValueError(f"Selected role does not pass: {role}")
        if selected_count != expected_count:
            raise ValueError(f"Selected role count mismatch: {role}")
        if not isinstance(role_checks, dict) or not all(role_checks.values()):
            raise ValueError(f"Selected role checks failed: {role}")
        if not isinstance(family_counts, dict):
            raise ValueError(f"Selected family counts missing: {role}")
        if set(family_counts) != EXPECTED_FAMILIES:
            raise ValueError(f"Selected family set mismatch: {role}")
        if sum(family_counts.values()) != expected_count:
            raise ValueError(f"Selected family total mismatch: {role}")
        if float(report.get("max_family_share", 1.0)) > 0.40:
            raise ValueError(f"Selected family cap failed: {role}")
        aggregate[role] = {
            "selected_count": selected_count,
            "family_counts": dict(family_counts),
            "max_family_share": report["max_family_share"],
            "target_counts": dict(report["target_counts"]),
            "descriptive_stage_counts":
                dict(report["descriptive_stage_counts"]),
            "checks": dict(role_checks),
        }
    return {
        "passes": True,
        "deficits": [],
        "selected_count": len(selected),
        "per_role": aggregate,
        "five_family_and_cap_checks_pass": True,
    }


def _verify_terminal_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not verify_self_hash(payload, "result_payload_sha256"):
        raise ValueError("Terminal result self hash mismatch")
    if payload.get("decision") != "HOLD_O3_ACQUISITION_RECOVERY_INTEGRITY":
        raise ValueError("Terminal result decision mismatch")
    if payload.get("error") != EXPECTED_RESULT_ERROR:
        raise ValueError("Terminal result error mismatch")
    return {
        "decision": payload["decision"],
        "error": payload["error"],
        "preserved_authoritative_hold": True,
    }


def verify_frozen_inputs(
    input_specs: Mapping[str, Mapping[str, Any]] = EXPECTED_INPUTS,
) -> dict[str, Any]:
    expected_names = {"union", "support", "selected", "result"}
    if set(input_specs) != expected_names:
        raise ValueError("Input specification names mismatch")

    payloads = {}
    identities = {}
    for name in ("union", "support", "selected", "result"):
        payloads[name], identities[name] = _load_and_verify_input(
            name,
            input_specs[name],
        )

    proof = prove_selected_key_coercion(payloads["selected"])
    facts = {
        "union": _verify_union(payloads["union"]),
        "support": _verify_support(payloads["support"]),
        "selected": _verify_selected(payloads["selected"]),
        "terminal_result": _verify_terminal_result(payloads["result"]),
    }
    return {
        "payloads": payloads,
        "identities": identities,
        "coercion_proof": proof,
        "facts": facts,
    }


def _test_evidence_identity() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise ValueError("Test evidence self hash mismatch")
    expected = {
        "amendment_sha256": sha256_path(AMENDMENT_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Test evidence source hash mismatch")
    if payload.get("focused_tests_passed", 0) < 1:
        raise ValueError("No focused tests recorded")
    if payload.get("regression_tests_passed", 0) < 1:
        raise ValueError("No regression tests recorded")
    return {
        "path": str(TEST_EVIDENCE_PATH),
        "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "payload_sha256": payload["test_evidence_payload_sha256"],
        "focused_tests_passed": payload["focused_tests_passed"],
        "regression_tests_passed": payload["regression_tests_passed"],
        "commands": list(payload["commands"]),
    }


def build_ready_envelope() -> dict[str, Any]:
    verified = verify_frozen_inputs()
    proof = verified["coercion_proof"]
    selected_body = proof["post_json_body"]
    payload = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": READY,
        "continue": True,
        "hold": False,
        "kill": False,
        "promote": False,
        "original_recovery": {
            "decision": "HOLD_O3_ACQUISITION_RECOVERY_INTEGRITY",
            "status": "preserved_immutable_and_authoritative",
            "directory": str(RECOVERY_DIR),
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
            "test_evidence": _test_evidence_identity(),
        },
        "input_identities": verified["identities"],
        "scientific_facts": verified["facts"],
        "selected_scientific_payload": selected_body,
        "selected_post_json_scientific_payload_sha256":
            proof["post_json_sha256"],
        "selected_pre_serialization_reproduction_sha256":
            proof["pre_serialization_reproduction_sha256"],
        "serialization_proof": {
            "numeric_string_paths": proof["numeric_string_paths"],
            "path_proof": proof["path_proof"],
            "only_six_maps_coerced": proof["only_six_maps_coerced"],
            "all_other_fields_unchanged": True,
            "defect_exhausted_by_json_key_coercion": True,
        },
        "reseal_scope": {
            "json_inputs_read": [
                str(UNION_PATH),
                str(SUPPORT_PATH),
                str(SELECTED_PATH),
                str(RESULT_PATH),
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
    return payload_with_hash(payload, "reseal_payload_sha256")


def build_hold_envelope(error: Exception) -> dict[str, Any]:
    identities = {}
    for name, spec in EXPECTED_INPUTS.items():
        path = Path(spec["path"])
        identities[name] = {
            "path": str(path),
            "file_sha256": sha256_path(path) if path.is_file() else None,
            "expected_file_sha256": spec["file_sha256"],
        }
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
        "original_recovery": {
            "decision": "HOLD_O3_ACQUISITION_RECOVERY_INTEGRITY",
            "status": "preserved_immutable_and_authoritative",
        },
        "observed_input_identities": identities,
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
    return payload_with_hash(payload, "reseal_payload_sha256")


def atomic_write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable reseal already exists: {path}")
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    reloaded = json.loads(serialized)
    if not verify_self_hash(reloaded, "reseal_payload_sha256"):
        raise ValueError("Reseal payload fails JSON round trip")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized + "\n")
    os.replace(temporary, path)
    written = json.loads(path.read_text())
    if not verify_self_hash(written, "reseal_payload_sha256"):
        raise ValueError("Written reseal payload self hash mismatch")


def write_test_evidence(
    *,
    focused_tests_passed: int,
    regression_tests_passed: int,
    commands: Sequence[str],
) -> dict[str, Any]:
    payload = payload_with_hash(
        {
            "version": f"{VERSION}_test_evidence",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "amendment_sha256": sha256_path(AMENDMENT_PATH),
            "runner_sha256": sha256_path(RUNNER_PATH),
            "tests_sha256": sha256_path(TEST_PATH),
            "focused_tests_passed": int(focused_tests_passed),
            "regression_tests_passed": int(regression_tests_passed),
            "commands": list(commands),
            "labels_generated": 0,
            "models_fit": 0,
            "policy_outcomes_opened": False,
        },
        "test_evidence_payload_sha256",
    )
    if TEST_EVIDENCE_PATH.exists():
        raise FileExistsError("Immutable test evidence already exists")
    TEST_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TEST_EVIDENCE_PATH.with_name(
        f".{TEST_EVIDENCE_PATH.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, TEST_EVIDENCE_PATH)
    return payload


def seal() -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Reseal output namespace exists: {OUTPUT_DIR}")
    try:
        payload = build_ready_envelope()
    except Exception as error:
        payload = build_hold_envelope(error)
    atomic_write_once(ENVELOPE_PATH, payload)
    return {
        "path": str(ENVELOPE_PATH),
        "file_sha256": sha256_path(ENVELOPE_PATH),
        "payload_sha256": payload["reseal_payload_sha256"],
        "decision": payload["decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--focused", type=int, required=True)
    evidence.add_argument("--regressions", type=int, required=True)
    evidence.add_argument("--command", action="append", required=True)

    subparsers.add_parser("seal")
    args = parser.parse_args()
    if args.command == "write-test-evidence":
        payload = write_test_evidence(
            focused_tests_passed=args.focused,
            regression_tests_passed=args.regressions,
            commands=args.command,
        )
        result = {
            "path": str(TEST_EVIDENCE_PATH),
            "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
            "payload_sha256": payload["test_evidence_payload_sha256"],
        }
    else:
        result = seal()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

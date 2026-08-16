"""Serialization-only integrity reseal for the completed K1 support audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


VERSION = "k1_support_audit_v2_serialization_reseal"
SOURCE_PATH = Path(
    "threes_rl/runs/forensics/k1_support_audit_v1/K1_SUPPORT_AUDIT.json"
)
OUTPUT_PATH = Path(
    "threes_rl/runs/forensics/k1_support_audit_v2/K1_SUPPORT_AUDIT_V2.json"
)
AMENDMENT_PATH = Path(
    "threes_rl/K1_SUPPORT_AUDIT_V2_SERIALIZATION_AMENDMENT.md"
)
IMPLEMENTATION_PATH = Path("threes_rl/k1_support_audit_seal_v2.py")
TEST_PATH = Path("tests/test_rl_k1_support_audit_seal_v2.py")

EXPECTED_SOURCE_FILE_SHA256 = (
    "536fd76da79791e873eecf1ea72c90ca2f26c9adc73f341c8fdb779517a73111"
)
EXPECTED_SOURCE_EMBEDDED_PAYLOAD_SHA256 = (
    "949bd408635a29e25ebdb6ca61b7723b7f68dca4f21d17d63be1b5e9de8efc3f"
)
EXPECTED_SOURCE_BODY_SHA256 = (
    "171c0b09ac6e92c7a36f1efb42932ba1a58ce1f917c310de470f5ffcfef7833b"
)
EXPECTED_DECISION = "KILL_EXACT_DEPTH3_PROGRAM"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def payload_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("canonical_payload_sha256", None)
    result["canonical_payload_sha256"] = canonical_json_hash(result)
    return result


def verify_payload_hash(payload: Mapping[str, Any]) -> bool:
    body = dict(payload)
    expected = body.pop("canonical_payload_sha256", None)
    return isinstance(expected, str) and canonical_json_hash(body) == expected


def load_frozen_scientific_payload(source_path: Path = SOURCE_PATH) -> dict[str, Any]:
    if sha256_path(source_path) != EXPECTED_SOURCE_FILE_SHA256:
        raise ValueError("K1 support audit v1 file hash mismatch")

    source = json.loads(source_path.read_text())
    embedded = source.pop("canonical_payload_sha256", None)
    if embedded != EXPECTED_SOURCE_EMBEDDED_PAYLOAD_SHA256:
        raise ValueError("K1 support audit v1 embedded payload hash mismatch")
    if canonical_json_hash(source) != EXPECTED_SOURCE_BODY_SHA256:
        raise ValueError("K1 support audit v1 canonical body hash mismatch")
    if source.get("decision") != EXPECTED_DECISION:
        raise ValueError("K1 support audit v1 decision mismatch")
    return source


def build_reseal_payload(
    *,
    source_path: Path = SOURCE_PATH,
    amendment_path: Path = AMENDMENT_PATH,
    implementation_path: Path = IMPLEMENTATION_PATH,
    test_path: Path = TEST_PATH,
) -> dict[str, Any]:
    scientific_payload = load_frozen_scientific_payload(source_path)
    payload = {
        "version": VERSION,
        "decision": EXPECTED_DECISION,
        "source_v1": {
            "path": str(source_path),
            "file_sha256": EXPECTED_SOURCE_FILE_SHA256,
            "embedded_payload_sha256":
                EXPECTED_SOURCE_EMBEDDED_PAYLOAD_SHA256,
            "post_json_scientific_body_sha256": EXPECTED_SOURCE_BODY_SHA256,
            "status": "preserved_superseded_serialization_defect",
        },
        "serialization_amendment": {
            "path": str(amendment_path),
            "sha256": sha256_path(amendment_path),
        },
        "reseal_implementation": {
            "path": str(implementation_path),
            "sha256": sha256_path(implementation_path),
        },
        "reseal_tests": {
            "path": str(test_path),
            "sha256": sha256_path(test_path),
        },
        "scientific_payload": scientific_payload,
        "scientific_payload_sha256": canonical_json_hash(scientific_payload),
        "scientific_fields_unchanged": True,
        "source_replays_reopened": 0,
        "support_statistics_recomputed": False,
        "forbidden_work": {
            "new_games": 0,
            "new_streams": 0,
            "compilations": 0,
            "timings": 0,
            "depth3_values_or_actions": 0,
            "score_or_recorded_action_reads": 0,
            "policy_outcomes": 0,
            "labels": 0,
            "models": 0,
            "incumbent_changes": 0,
            "dashboard_changes": 0,
        },
    }
    return payload_with_hash(payload)


def atomic_write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable reseal: {path}")

    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    reloaded = json.loads(serialized)
    if not verify_payload_hash(reloaded):
        raise ValueError("K1 support audit v2 payload fails JSON round trip")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized)
    os.replace(temporary, path)

    written = json.loads(path.read_text())
    if not verify_payload_hash(written):
        raise ValueError("Written K1 support audit v2 payload hash mismatch")


def main() -> None:
    payload = build_reseal_payload()
    atomic_write_once(OUTPUT_PATH, payload)
    result = {
        "path": str(OUTPUT_PATH),
        "file_sha256": sha256_path(OUTPUT_PATH),
        "canonical_payload_sha256": payload["canonical_payload_sha256"],
        "scientific_payload_sha256": payload["scientific_payload_sha256"],
        "decision": payload["decision"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

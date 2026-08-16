from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from threes_rl import j1b_open_failure_terminalize as terminalize


def _write_marker(path: Path, *, scientific_work: int = 0) -> dict:
    return terminalize.write_immutable_json(
        path,
        {
            "version": "fixture_marker_v1",
            "marker_only_open": True,
            "streams_reserved": 0,
            "streams_consumed": 0,
            "scientific_work": scientific_work,
        },
        field="activation_marker_payload_sha256",
    )


def _fixture_original(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    root = tmp_path / "original"
    training = root / "training"
    training.mkdir(parents=True)
    (training / "phase_lock.json").write_text(
        '{"fixture":"lock"}\n', encoding="utf-8"
    )
    (training / "phase_lock_result.json").write_text(
        '{"fixture":"result"}\n', encoding="utf-8"
    )
    marker = _write_marker(training / "execution_opened.json")
    expected = {
        str(path.relative_to(root)): terminalize.sha256_path(path)
        for path in training.iterdir()
        if path.is_file()
    }
    return (
        root,
        expected,
        marker["activation_marker_payload_sha256"],
    )


def test_json_native_reproduces_and_repairs_tuple_list_defect() -> None:
    report = terminalize.reproduce_tuple_list_defect()
    assert report["passes"]
    assert report["path"].endswith("dashboard.top_three")
    payload = {"top_three": (1, 2, 3)}
    native = terminalize.json_native(payload)
    assert payload != native
    assert native == {"top_three": [1, 2, 3]}
    assert (
        terminalize.canonical_json_bytes(payload)
        == terminalize.canonical_json_bytes(native)
    )


def test_immutable_json_is_json_native_and_create_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    payload = terminalize.write_immutable_json(
        path,
        {"top_three": (3, 2, 1)},
        field="payload_sha256",
    )
    assert payload["top_three"] == [3, 2, 1]
    assert terminalize.verify_payload_hash(payload, "payload_sha256")
    before = path.read_bytes()
    same = terminalize.write_immutable_json(
        path,
        {"top_three": [3, 2, 1]},
        field="payload_sha256",
        allow_existing_exact=True,
    )
    assert same == payload
    assert path.read_bytes() == before
    with pytest.raises(terminalize.J1bTerminalizationError):
        terminalize.write_immutable_json(
            path,
            {"top_three": [1, 2, 3]},
            field="payload_sha256",
            allow_existing_exact=True,
        )


def test_original_namespace_fixture_passes(tmp_path: Path) -> None:
    root, expected, marker_payload = _fixture_original(tmp_path)
    report = terminalize.audit_original_namespace(
        root,
        expected_files=expected,
        expected_marker_payload_sha256=marker_payload,
    )
    assert report["passes"]
    assert report["observed_paths"] == sorted(expected)
    assert all(value == 0 for value in report["zero_work"].values())


def test_extra_original_file_fails_closed(tmp_path: Path) -> None:
    root, expected, marker_payload = _fixture_original(tmp_path)
    (root / "training" / "root_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    report = terminalize.audit_original_namespace(
        root,
        expected_files=expected,
        expected_marker_payload_sha256=marker_payload,
    )
    assert not report["passes"]
    assert not report["checks"]["exact_three_file_inventory"]
    assert not report["checks"]["forbidden_work_paths_absent"]


def test_marker_tamper_fails_closed(tmp_path: Path) -> None:
    root, expected, marker_payload = _fixture_original(tmp_path)
    marker_path = root / "training" / "execution_opened.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["hostname"] = "tampered"
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = terminalize.audit_original_namespace(
        root,
        expected_files=expected,
        expected_marker_payload_sha256=marker_payload,
    )
    assert not report["passes"]
    assert not report["checks"]["all_file_hashes_exact"]
    assert not report["checks"]["marker_payload_valid"]


def test_nonzero_marker_work_fails_closed(tmp_path: Path) -> None:
    root, expected, _ = _fixture_original(tmp_path)
    marker_path = root / "training" / "execution_opened.json"
    marker_path.unlink()
    marker = _write_marker(marker_path, scientific_work=1)
    expected["training/execution_opened.json"] = (
        terminalize.sha256_path(marker_path)
    )
    report = terminalize.audit_original_namespace(
        root,
        expected_files=expected,
        expected_marker_payload_sha256=marker[
            "activation_marker_payload_sha256"
        ],
    )
    assert not report["passes"]
    assert not report["checks"]["marker_is_zero_work"]


def test_real_spent_namespace_audit_is_exact_and_read_only() -> None:
    before = {
        relative: terminalize.sha256_path(
            terminalize.ORIGINAL_ROOT / relative
        )
        for relative in terminalize.EXPECTED_ORIGINAL_FILES
    }
    report = terminalize.audit_original_namespace()
    after = {
        relative: terminalize.sha256_path(
            terminalize.ORIGINAL_ROOT / relative
        )
        for relative in terminalize.EXPECTED_ORIGINAL_FILES
    }
    assert report["passes"]
    assert before == after == terminalize.EXPECTED_ORIGINAL_FILES


def test_external_terminal_and_retention_seal_in_temp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "external"
    original_before = {
        relative: terminalize.sha256_path(
            terminalize.ORIGINAL_ROOT / relative
        )
        for relative in terminalize.EXPECTED_ORIGINAL_FILES
    }
    evidence = terminalize.write_test_evidence(
        out_dir=output,
        commands=[
            {
                "command": "synthetic focused fixture",
                "passed": 8,
                "failed": 0,
                "passes": True,
            }
        ],
    )
    result = terminalize.seal_external_terminal(out_dir=output)
    assert evidence["passes"]
    assert result["passes"]
    assert result["terminal"]["decision"] == (
        "HOLD_J1B_OPEN_SERIALIZATION_INTEGRITY"
    )
    assert result["retention"]["decision"] == (
        "PRESERVE_J1B_OPEN_FAILURE_EVIDENCE"
    )
    assert sorted(path.name for path in output.iterdir()) == sorted(
        (
            terminalize.EVIDENCE_NAME,
            terminalize.TERMINAL_NAME,
            terminalize.RETENTION_NAME,
        )
    )
    original_after = {
        relative: terminalize.sha256_path(
            terminalize.ORIGINAL_ROOT / relative
        )
        for relative in terminalize.EXPECTED_ORIGINAL_FILES
    }
    assert original_before == original_after


def test_evidence_tamper_blocks_terminal(tmp_path: Path) -> None:
    output = tmp_path / "external"
    terminalize.write_test_evidence(
        out_dir=output,
        commands=[
            {
                "command": "synthetic focused fixture",
                "passed": 1,
                "failed": 0,
                "passes": True,
            }
        ],
    )
    evidence_path = output / terminalize.EVIDENCE_NAME
    data = bytearray(evidence_path.read_bytes())
    data[data.index(b"scientific_work")] = ord("S")
    evidence_path.write_bytes(bytes(data))
    with pytest.raises(terminalize.J1bTerminalizationError):
        terminalize.seal_external_terminal(out_dir=output)


def test_payload_hash_matches_exact_canonical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    payload = terminalize.write_immutable_json(
        path,
        {"b": 2, "a": (1, 3)},
        field="payload_sha256",
    )
    body = dict(payload)
    observed = body.pop("payload_sha256")
    expected = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert observed == expected

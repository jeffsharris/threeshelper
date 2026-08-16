import copy
import json
from pathlib import Path

import pytest

from threes_rl import o3_selected_integrity_reseal_v2 as reseal


def _selected_payload() -> dict:
    return json.loads(reseal.SELECTED_PATH.read_text())


def _set_at_path(payload: dict, path: tuple[str, ...], value: object) -> None:
    current = payload
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = value


def test_frozen_input_files_have_exact_identities() -> None:
    for spec in reseal.EXPECTED_INPUTS.values():
        assert reseal.sha256_path(spec["path"]) == spec["file_sha256"]


def test_exact_six_path_coercion_reproduces_embedded_hash() -> None:
    payload = _selected_payload()
    proof = reseal.prove_selected_key_coercion(payload)
    assert proof["post_json_sha256"] == (
        reseal.EXPECTED_SELECTED_POST_JSON_SHA256
    )
    assert proof["pre_serialization_reproduction_sha256"] == (
        reseal.EXPECTED_INPUTS["selected"]["payload_sha256"]
    )
    assert set(proof["numeric_string_paths"]) == {
        ".".join(path) for path in reseal.EXPECTED_COERCION_PATHS
    }
    assert proof["only_six_maps_coerced"]


def test_extra_numeric_string_path_fails_closed() -> None:
    payload = _selected_payload()
    payload["unexpected"] = {"1": 2}
    body = copy.deepcopy(payload)
    body.pop("selected_payload_sha256")
    reseal.EXPECTED_SELECTED_POST_JSON_SHA256 = reseal.canonical_json_hash(body)
    try:
        with pytest.raises(ValueError, match="extra="):
            reseal.prove_selected_key_coercion(payload)
    finally:
        reseal.EXPECTED_SELECTED_POST_JSON_SHA256 = (
            "d9600cf420d947826c812b88225633b78a889f94f94ce39270dd71bc11b12f0e"
        )


def test_missing_coercion_path_fails_closed() -> None:
    payload = _selected_payload()
    del payload["per_role"]["train"]["target_counts"]
    body = copy.deepcopy(payload)
    body.pop("selected_payload_sha256")
    original = reseal.EXPECTED_SELECTED_POST_JSON_SHA256
    reseal.EXPECTED_SELECTED_POST_JSON_SHA256 = reseal.canonical_json_hash(body)
    try:
        with pytest.raises(ValueError, match="missing="):
            reseal.prove_selected_key_coercion(payload)
    finally:
        reseal.EXPECTED_SELECTED_POST_JSON_SHA256 = original


def test_nonnumeric_key_at_required_path_fails_closed() -> None:
    payload = _selected_payload()
    payload["per_role"]["train"]["target_counts"]["bad"] = 0
    body = copy.deepcopy(payload)
    body.pop("selected_payload_sha256")
    original = reseal.EXPECTED_SELECTED_POST_JSON_SHA256
    reseal.EXPECTED_SELECTED_POST_JSON_SHA256 = reseal.canonical_json_hash(body)
    try:
        with pytest.raises(ValueError, match="Nonnumeric coercion key"):
            reseal.prove_selected_key_coercion(payload)
    finally:
        reseal.EXPECTED_SELECTED_POST_JSON_SHA256 = original


@pytest.mark.parametrize("name", ["union", "support", "selected", "result"])
def test_changed_input_identity_fails_closed(
    tmp_path: Path,
    name: str,
) -> None:
    specs = copy.deepcopy(reseal.EXPECTED_INPUTS)
    original = Path(specs[name]["path"])
    changed = tmp_path / original.name
    changed.write_bytes(original.read_bytes() + b"\n")
    specs[name]["path"] = changed
    with pytest.raises(ValueError, match=f"{name} file SHA mismatch"):
        reseal.verify_frozen_inputs(specs)


def test_tampered_selected_scientific_check_fails_closed() -> None:
    payload = _selected_payload()
    payload["passes"] = False
    with pytest.raises(ValueError, match="post-JSON canonical SHA mismatch"):
        reseal.prove_selected_key_coercion(payload)


def test_verified_facts_match_frozen_aggregates() -> None:
    verified = reseal.verify_frozen_inputs()
    assert verified["facts"]["union"]["membership_count"] == 20500
    assert verified["facts"]["union"]["role_counts"] == {
        "train": 5020,
        "development": 1675,
        "untouched_mechanism": 13805,
    }
    assert verified["facts"]["support"] == {
        "passes": True,
        "candidate_rows": 12922,
        "candidate_roots": 7607,
        "stage_descriptive_only": True,
    }
    assert verified["facts"]["selected"]["selected_count"] == 320
    assert verified["facts"]["selected"]["deficits"] == []
    assert verified["facts"]["terminal_result"]["error"] == (
        reseal.EXPECTED_RESULT_ERROR
    )


def test_ready_envelope_is_json_reload_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = reseal.payload_with_hash(
        {
            "version": "test",
            "amendment_sha256": reseal.sha256_path(reseal.AMENDMENT_PATH),
            "runner_sha256": reseal.sha256_path(reseal.RUNNER_PATH),
            "tests_sha256": reseal.sha256_path(reseal.TEST_PATH),
            "focused_tests_passed": 1,
            "regression_tests_passed": 1,
            "commands": ["test"],
        },
        "test_evidence_payload_sha256",
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    monkeypatch.setattr(reseal, "TEST_EVIDENCE_PATH", evidence_path)
    payload = reseal.build_ready_envelope()
    reloaded = json.loads(json.dumps(payload, sort_keys=True))
    assert reseal.verify_self_hash(reloaded, "reseal_payload_sha256")
    assert reloaded["decision"] == reseal.READY
    assert reloaded["selected_post_json_scientific_payload_sha256"] == (
        reseal.EXPECTED_SELECTED_POST_JSON_SHA256
    )


def test_atomic_write_once_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "out" / "reseal.json"
    payload = reseal.payload_with_hash(
        {"decision": reseal.READY},
        "reseal_payload_sha256",
    )
    reseal.atomic_write_once(path, payload)
    assert reseal.verify_self_hash(
        json.loads(path.read_text()),
        "reseal_payload_sha256",
    )
    with pytest.raises(FileExistsError, match="already exists"):
        reseal.atomic_write_once(path, payload)


def test_hold_envelope_is_fail_closed_and_nonpromotable() -> None:
    payload = reseal.build_hold_envelope(ValueError("test failure"))
    assert payload["decision"] == reseal.HOLD
    assert payload["hold"]
    assert not payload["continue"]
    assert not payload["kill"]
    assert not payload["promote"]
    assert set(payload["zero_new_work"].values()) == {0}
    assert reseal.verify_self_hash(payload, "reseal_payload_sha256")


def test_output_namespace_is_separate_from_recovery() -> None:
    assert reseal.OUTPUT_DIR != reseal.RECOVERY_DIR
    assert reseal.RECOVERY_DIR not in reseal.OUTPUT_DIR.parents
    assert not reseal.OUTPUT_DIR.exists()


def test_only_four_json_inputs_are_authorized() -> None:
    assert {spec["path"] for spec in reseal.EXPECTED_INPUTS.values()} == {
        reseal.UNION_PATH,
        reseal.SUPPORT_PATH,
        reseal.SELECTED_PATH,
        reseal.RESULT_PATH,
    }

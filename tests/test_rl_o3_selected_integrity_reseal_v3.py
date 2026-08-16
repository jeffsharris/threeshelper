import copy
import json
from pathlib import Path

import pytest

from threes_rl import o3_selected_integrity_reseal_v2 as v2
from threes_rl import o3_selected_integrity_reseal_v3 as v3


def test_v2_history_is_exact_and_preserved() -> None:
    report = v3.verify_v2_history()
    assert report["v2_hold_preserved"]
    assert report["v2_test_evidence_absent"]
    terminal = report["identities"]["terminal_envelope"]
    assert terminal["decision"] == v2.HOLD
    assert terminal["payload_sha256"] == (
        v3.V2_BINDINGS["terminal_envelope"]["payload_sha256"]
    )
    assert terminal["error"] == v3.EXPECTED_V2_ERROR


def test_v2_binding_tamper_fails_closed(tmp_path: Path) -> None:
    bindings = copy.deepcopy(v3.V2_BINDINGS)
    changed = tmp_path / "changed_v2.py"
    changed.write_bytes(v2.RUNNER_PATH.read_bytes() + b"\n")
    bindings["runner"]["path"] = changed
    with pytest.raises(ValueError, match="V2 runner file SHA mismatch"):
        v3.verify_v2_history(bindings)


def test_six_path_scientific_proof_remains_exact() -> None:
    verified = v2.verify_frozen_inputs()
    proof = verified["coercion_proof"]
    assert proof["post_json_sha256"] == (
        v2.EXPECTED_SELECTED_POST_JSON_SHA256
    )
    assert proof["pre_serialization_reproduction_sha256"] == (
        v2.EXPECTED_INPUTS["selected"]["payload_sha256"]
    )
    assert set(proof["numeric_string_paths"]) == {
        ".".join(path) for path in v2.EXPECTED_COERCION_PATHS
    }


def test_parser_destinations_are_structurally_distinct() -> None:
    args = v3.build_parser().parse_args(
        [
            "write-test-evidence",
            "--focused",
            "17",
            "--regressions",
            "170",
            "--recorded-command",
            "focused",
            "--recorded-command",
            "regressions",
        ]
    )
    assert args.subcommand == "write-test-evidence"
    assert args.recorded_commands == ["focused", "regressions"]
    assert not hasattr(args, "command")


def test_evidence_argv_routing_cannot_enter_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_evidence(**kwargs: object) -> dict:
        calls.append(("evidence", kwargs))
        return {"artifact": "test_evidence"}

    def forbidden_seal() -> dict:
        raise AssertionError("seal must not be called")

    monkeypatch.setattr(v3, "write_test_evidence", fake_evidence)
    monkeypatch.setattr(v3, "seal", forbidden_seal)
    result = v3.dispatch(
        [
            "write-test-evidence",
            "--focused",
            "17",
            "--regressions",
            "170",
            "--recorded-command",
            "focused",
            "--recorded-command",
            "regressions",
        ]
    )
    assert result == {"artifact": "test_evidence"}
    assert [name for name, _ in calls] == ["evidence"]
    assert calls[0][1]["recorded_commands"] == ["focused", "regressions"]


def test_seal_argv_routing_cannot_write_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_evidence(**kwargs: object) -> dict:
        raise AssertionError("evidence must not be called")

    monkeypatch.setattr(v3, "write_test_evidence", forbidden_evidence)
    monkeypatch.setattr(
        v3,
        "seal",
        lambda: {"artifact": "terminal_envelope"},
    )
    assert v3.dispatch(["seal"]) == {"artifact": "terminal_envelope"}


def test_test_evidence_survives_json_reload() -> None:
    payload = v3.build_test_evidence_payload(
        focused_tests_passed=17,
        regression_tests_passed=170,
        recorded_commands=["focused", "regressions"],
    )
    reloaded = json.loads(json.dumps(payload, sort_keys=True))
    assert v3.verify_self_hash(
        reloaded,
        "test_evidence_payload_sha256",
    )
    assert set(reloaded["zero_new_work"].values()) == {0}


def test_write_test_evidence_is_separate_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    result = v3.write_test_evidence(
        focused_tests_passed=17,
        regression_tests_passed=170,
        recorded_commands=["focused", "regressions"],
        path=path,
    )
    assert result["artifact"] == "test_evidence"
    assert v3.verify_self_hash(
        json.loads(path.read_text()),
        "test_evidence_payload_sha256",
    )
    with pytest.raises(FileExistsError, match="already exists"):
        v3.write_test_evidence(
            focused_tests_passed=17,
            regression_tests_passed=170,
            recorded_commands=["focused"],
            path=path,
        )


def test_missing_evidence_seals_fail_closed_hold(tmp_path: Path) -> None:
    output = tmp_path / "terminal" / "result.json"
    result = v3.seal(
        output_path=output,
        evidence_path=tmp_path / "missing.json",
    )
    payload = json.loads(output.read_text())
    assert result["decision"] == v3.HOLD
    assert payload["decision"] == v3.HOLD
    assert payload["hold"]
    assert not payload["continue"]
    assert not payload["kill"]
    assert not payload["promote"]
    assert payload["error_type"] == "FileNotFoundError"
    assert set(payload["zero_new_work"].values()) == {0}
    assert v3.verify_self_hash(payload, "v3_reseal_payload_sha256")


def test_ready_envelope_survives_json_reload(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    v3.write_test_evidence(
        focused_tests_passed=17,
        regression_tests_passed=170,
        recorded_commands=["focused", "regressions"],
        path=evidence_path,
    )
    payload = v3.build_ready_envelope(evidence_path)
    reloaded = json.loads(json.dumps(payload, sort_keys=True))
    assert v3.verify_self_hash(reloaded, "v3_reseal_payload_sha256")
    assert reloaded["decision"] == v3.READY
    assert reloaded["serialization_proof"][
        "defect_exhausted_by_json_key_coercion"
    ]
    assert reloaded["selected_post_json_scientific_payload_sha256"] == (
        v2.EXPECTED_SELECTED_POST_JSON_SHA256
    )


def test_ready_envelope_binds_exact_scientific_facts(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    v3.write_test_evidence(
        focused_tests_passed=17,
        regression_tests_passed=170,
        recorded_commands=["focused"],
        path=evidence_path,
    )
    payload = v3.build_ready_envelope(evidence_path)
    assert payload["scientific_facts"]["union"]["membership_count"] == 20500
    assert payload["scientific_facts"]["support"]["candidate_rows"] == 12922
    assert payload["scientific_facts"]["support"]["candidate_roots"] == 7607
    assert payload["scientific_facts"]["selected"]["selected_count"] == 320
    assert payload["scientific_facts"]["selected"]["deficits"] == []
    assert payload["historical_holds"]["v2_reseal"] == v2.HOLD


def test_terminal_namespace_is_fresh_and_absent() -> None:
    assert v3.OUTPUT_DIR != v2.OUTPUT_DIR
    assert v3.TEST_EVIDENCE_PATH != v2.TEST_EVIDENCE_PATH
    assert not v3.OUTPUT_DIR.exists()
    assert not v3.TEST_EVIDENCE_PATH.exists()

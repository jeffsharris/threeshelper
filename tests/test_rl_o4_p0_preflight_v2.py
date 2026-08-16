from __future__ import annotations

import json
from pathlib import Path

import pytest

from threes_rl import o4_p0_preflight_v2 as v2


def test_json_normalization_equates_tuples_and_arrays() -> None:
    left = {"matrix": ((1, 2), (3, 4))}
    right = {"matrix": [[1, 2], [3, 4]]}
    assert v2._same_json(left, right)
    assert v2._normalize(left) == right
    assert not v2._same_json(left, {"matrix": [[1, 2], [4, 3]]})


def test_marker_reload_accepts_json_roundtrip_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "o4"
    out_dir.mkdir()
    bindings = {
        "role_family_target_counts": {
            "train": ((13, 13, 13), (13, 13, 13))
        }
    }
    monkeypatch.setattr(v2, "_bindings", lambda _out: bindings)
    v2.science._write_immutable_json(
        out_dir / v2.MARKER_NAME,
        {**bindings, "decision": "O4_P0_V2_OPENED_ZERO_WORK"},
        self_hash_field="opened_payload_sha256",
    )
    marker = v2._load_marker(out_dir)
    assert marker["role_family_target_counts"]["train"] == [
        [13, 13, 13],
        [13, 13, 13],
    ]
    monkeypatch.setattr(
        v2,
        "_bindings",
        lambda _out: {
            "role_family_target_counts": {
                "train": ((13, 13, 12), (13, 13, 13))
            }
        },
    )
    with pytest.raises(v2.science.SourceIntegrityError, match="binding"):
        v2._load_marker(out_dir)


def test_v1_hold_seals_only_from_marker_only_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    marker_path = v1_dir / v2.science.MARKER_NAME
    marker = v2.science._write_immutable_json(
        marker_path,
        {"decision": "O4_P0_OPENED_ZERO_WORK"},
        self_hash_field="opened_payload_sha256",
    )
    runner = tmp_path / "runner.py"
    runner.write_text("science")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    monkeypatch.setattr(v2.science, "OUTPUT_DIR", v1_dir)
    monkeypatch.setattr(v2.science, "RUNNER_PATH", runner)
    monkeypatch.setattr(v2.science, "TEST_EVIDENCE_PATH", evidence)
    monkeypatch.setattr(v2, "V1_MARKER_PATH", marker_path)
    monkeypatch.setattr(v2, "V1_HOLD_PATH", v1_dir / v2.V1_HOLD_NAME)
    monkeypatch.setattr(v2, "V1_MARKER_FILE_SHA256", v2.sha256_path(marker_path))
    monkeypatch.setattr(
        v2,
        "V1_MARKER_PAYLOAD_SHA256",
        marker["opened_payload_sha256"],
    )
    monkeypatch.setattr(v2, "V1_RUNNER_SHA256", v2.sha256_path(runner))
    monkeypatch.setattr(
        v2,
        "V1_TEST_EVIDENCE_FILE_SHA256",
        v2.sha256_path(evidence),
    )
    result = v2.seal_v1_engineering_hold()
    assert result["decision"] == "HOLD_O4_P0_V1_ORCHESTRATION"
    assert result["zero_work"]["source_replay_bodies_opened"] == 0
    assert set(path.name for path in v1_dir.iterdir()) == {
        v2.science.MARKER_NAME,
        v2.V1_HOLD_NAME,
    }


def test_v1_hold_rejects_any_content_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    (v1_dir / v2.science.MARKER_NAME).write_text("{}")
    (v1_dir / "unexpected.json").write_text("{}")
    monkeypatch.setattr(v2.science, "OUTPUT_DIR", v1_dir)
    monkeypatch.setattr(
        v2,
        "V1_MARKER_PATH",
        v1_dir / v2.science.MARKER_NAME,
    )
    monkeypatch.setattr(v2, "V1_HOLD_PATH", v1_dir / v2.V1_HOLD_NAME)
    with pytest.raises(v2.science.SourceIntegrityError, match="zero-content"):
        v2.seal_v1_engineering_hold()


def test_v2_amendment_changes_no_scientific_contract() -> None:
    assert v2.science.VERSION == "o4_domain_safe_p0_v1"
    assert v2.science.TOTAL_SELECTED_ROOTS == 448
    assert v2.science.ROLE_FAMILY_TARGET_COUNTS["train"][0] == (13, 13, 13)
    assert v2.science.STREAM_BASES["learning"]["logical_seed"] == 129_000_000_000
    assert v2.science.parameter_count() == 102_557


def test_v2_parser_routes_test_evidence_without_terminal_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []
    monkeypatch.setattr(
        v2,
        "seal_test_evidence",
        lambda **kwargs: called.append(("evidence", kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        v2,
        "run_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal route entered")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "o4",
            "seal-test-evidence",
            "--focused-passed",
            "1",
            "--regression-passed",
            "2",
        ],
    )
    v2.main()
    assert called and called[0][0] == "evidence"

from __future__ import annotations

import json
from pathlib import Path

import pytest

from threes_rl import o2_yield_pilot_scan_recovery as recovery


def _support_fixture(decision: str = "HOLD_O2_DATA_SUPPORT") -> dict:
    return {
        "version": recovery.pilot.VERSION,
        "decision": decision,
        "candidate_rows": [],
        "scan_audit": {"frames_scanned": 10},
        "structural_layer": {"passes": False},
        "availability_layer": {"passes": False},
        "descriptive_1536": {"selected_count": 0},
    }


def test_recovery_charter_and_original_evidence_hashes_are_exact() -> None:
    assert recovery.sha256_path(recovery.CHARTER_PATH) == recovery.CHARTER_SHA256
    audit = recovery.immutable_original_audit()
    assert audit["passes"]
    assert all(audit["checks"].values())
    assert not recovery.ORIGINAL_SUPPORT.exists()


def test_real_completion_schema_reproduces_keyerror_and_adapter_fixes_it() -> None:
    rows, sources = recovery.source_manifest_audit()
    assert sources["passes"]
    assert len(rows) == 128
    assert set(rows[0]) == recovery.COMPLETION_FIELDS
    with pytest.raises(KeyError) as error:
        recovery.pilot._stream_key(rows[0])
    assert error.value.args == ("family_game_index",)
    adapted = recovery.completion_adapter(rows[0])
    assert recovery.pilot._stream_key(adapted) == (
        rows[0]["family"],
        rows[0]["game_index"],
    )
    assert {
        key: value
        for key, value in adapted.items()
        if key != "family_game_index"
    } == rows[0]


def test_completion_adapter_rejects_missing_or_pre_adapted_schema() -> None:
    with pytest.raises(ValueError, match="raw v1"):
        recovery.completion_adapter({"family": "x"})
    with pytest.raises(ValueError, match="raw v1"):
        recovery.completion_adapter(
            {"game_index": 1, "family_game_index": 1}
        )


def test_real_source_attempt_and_runtime_audits_pass_without_support_read() -> None:
    rows, sources = recovery.source_manifest_audit()
    attempts = recovery.attempt_and_adapter_audit(rows)
    runtime = recovery.runtime_audit()
    assert sources["checks"] == {
        "exact_128_rows": True,
        "all_rows_exact": True,
        "equal_32_per_family": True,
        "unique_ancestries": True,
        "unique_replay_paths": True,
        "unique_replay_hashes": True,
        "exact_stream_manifest": True,
    }
    assert attempts["passes"]
    assert attempts["raw_error"] == "KeyError('family_game_index')"
    assert attempts["status_counts"] == {"completed": 128, "opened": 128}
    assert attempts["fixed_original_attempt_audit"]["retries"] == 0
    assert runtime["passes"]
    assert runtime["runtime"]["games_completed"] == 128
    assert not recovery.ORIGINAL_SUPPORT.exists()


def test_open_creates_only_zero_content_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "recovery"
    marker_path = out_dir / "marker.json"
    monkeypatch.setattr(recovery, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(recovery, "MARKER_PATH", marker_path)
    monkeypatch.setattr(recovery, "SUPPORT_PATH", out_dir / "support.json")
    monkeypatch.setattr(recovery, "RESULT_PATH", out_dir / "result.json")
    monkeypatch.setattr(
        recovery,
        "immutable_original_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        recovery,
        "source_manifest_audit",
        lambda: ([{"game_index": 0}], {"passes": True}),
    )
    monkeypatch.setattr(
        recovery,
        "attempt_and_adapter_audit",
        lambda rows: {"passes": True},
    )
    monkeypatch.setattr(recovery, "runtime_audit", lambda: {"passes": True})
    monkeypatch.setattr(
        recovery,
        "operational_audit",
        lambda out_dir: {"passes": True},
    )
    monkeypatch.setattr(
        recovery,
        "load_test_evidence",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        recovery,
        "current_bindings",
        lambda: {"fixture": "exact"},
    )
    marker = recovery.open_recovery(out_dir=out_dir)
    assert recovery.pilot.preflight.verify_payload_hash(marker)
    assert [path.name for path in out_dir.iterdir()] == ["marker.json"]
    assert all(value == 0 for value in marker["zero_content"].values())
    with pytest.raises(FileExistsError):
        recovery.open_recovery(out_dir=out_dir)


def test_load_marker_rejects_current_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "recovery"
    out_dir.mkdir()
    marker_path = out_dir / "marker.json"
    current = {"binding": {"source": "a"}}
    monkeypatch.setattr(recovery, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(recovery, "MARKER_PATH", marker_path)
    monkeypatch.setattr(recovery, "RESULT_PATH", out_dir / "result.json")
    monkeypatch.setattr(
        recovery,
        "source_manifest_audit",
        lambda: ([{"game_index": 0}], {"passes": True}),
    )
    monkeypatch.setattr(
        recovery,
        "attempt_and_adapter_audit",
        lambda rows: {"passes": True},
    )
    monkeypatch.setattr(recovery, "runtime_audit", lambda: {"passes": True})
    monkeypatch.setattr(
        recovery,
        "immutable_original_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        recovery,
        "current_bindings",
        lambda: current["binding"],
    )
    marker = {
        "version": recovery.VERSION,
        "decision": "OPENED_O2_SCAN_RECOVERY",
        "bound_out_dir": str(out_dir.resolve()),
        "bound_execute_command": recovery.EXECUTE_COMMAND,
        "bindings": {"source": "a"},
        "source_audit": {"passes": True},
        "attempt_and_adapter_audit": {"passes": True},
        "runtime_audit": {"passes": True},
    }
    recovery.write_immutable_json(marker_path, marker)
    assert recovery.load_marker(out_dir=out_dir)["decision"].startswith(
        "OPENED"
    )
    current["binding"] = {"source": "b"}
    with pytest.raises(ValueError, match="binding mismatch"):
        recovery.load_marker(out_dir=out_dir)


def test_execute_uses_exact_original_support_analysis_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "recovery"
    out_dir.mkdir()
    marker_path = out_dir / "marker.json"
    support_path = out_dir / "support.json"
    result_path = out_dir / "result.json"
    recovery.write_immutable_json(marker_path, {"fixture": True})
    completions = [{"game_index": index} for index in range(128)]
    calls = []
    monkeypatch.setattr(recovery, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(recovery, "MARKER_PATH", marker_path)
    monkeypatch.setattr(recovery, "SUPPORT_PATH", support_path)
    monkeypatch.setattr(recovery, "RESULT_PATH", result_path)
    monkeypatch.setattr(
        recovery,
        "load_marker",
        lambda out_dir: {"decision": "OPENED_O2_SCAN_RECOVERY"},
    )
    monkeypatch.setattr(
        recovery,
        "source_manifest_audit",
        lambda: (completions, {"passes": True}),
    )
    monkeypatch.setattr(
        recovery,
        "attempt_and_adapter_audit",
        lambda rows: {"passes": True},
    )
    monkeypatch.setattr(
        recovery,
        "operational_audit",
        lambda out_dir: {"passes": True},
    )

    def exact_support(rows: object) -> dict:
        calls.append(rows)
        return _support_fixture()

    monkeypatch.setattr(recovery.pilot, "support_analysis", exact_support)
    result = recovery.execute_recovery(out_dir=out_dir)
    assert result["decision"] == "HOLD_O2_DATA_SUPPORT"
    assert calls == [completions]
    assert support_path.is_file()
    assert result["original_decision_preserved"] == (
        "HOLD_O2_PILOT_OPERATIONAL_INTEGRITY"
    )
    assert all(value == 0 for value in result["zero_forbidden_work"].values())


def test_execute_seals_integrity_hold_without_retry_on_scan_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "recovery"
    out_dir.mkdir()
    marker_path = out_dir / "marker.json"
    result_path = out_dir / "result.json"
    recovery.write_immutable_json(marker_path, {"fixture": True})
    monkeypatch.setattr(recovery, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(recovery, "MARKER_PATH", marker_path)
    monkeypatch.setattr(recovery, "SUPPORT_PATH", out_dir / "support.json")
    monkeypatch.setattr(recovery, "RESULT_PATH", result_path)
    monkeypatch.setattr(
        recovery,
        "load_marker",
        lambda out_dir: {"decision": "OPENED_O2_SCAN_RECOVERY"},
    )
    monkeypatch.setattr(
        recovery,
        "source_manifest_audit",
        lambda: ([{}] * 128, {"passes": True}),
    )
    monkeypatch.setattr(
        recovery,
        "attempt_and_adapter_audit",
        lambda rows: {"passes": True},
    )
    monkeypatch.setattr(
        recovery,
        "operational_audit",
        lambda out_dir: {"passes": True},
    )
    monkeypatch.setattr(
        recovery.pilot,
        "support_analysis",
        lambda rows: (_ for _ in ()).throw(ValueError("scan failed")),
    )
    result = recovery.execute_recovery(out_dir=out_dir)
    assert result["decision"] == "HOLD_O2_SCAN_RECOVERY_INTEGRITY"
    assert result["error"] == "scan failed"
    with pytest.raises(FileExistsError):
        recovery.execute_recovery(out_dir=out_dir)


def test_original_pilot_artifacts_remain_immutable_and_recovery_absent() -> None:
    assert {
        path: recovery.sha256_path(path)
        for path in recovery.EXPECTED_ORIGINAL_HASHES
    } == recovery.EXPECTED_ORIGINAL_HASHES
    assert not recovery.ORIGINAL_SUPPORT.exists()
    assert not recovery.OUTPUT_DIR.exists()

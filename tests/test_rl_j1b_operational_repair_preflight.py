from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from threes_rl import j1b_operational_repair_preflight as j1b


@pytest.fixture(scope="module")
def fresh_manifest() -> dict:
    return j1b.prospective_training_manifest()


class FakeTorch:
    def __init__(self, *, fail_interop: bool = False) -> None:
        self.interop = 12
        self.intraop = 12
        self.deterministic = False
        self.fail_interop = fail_interop
        self.calls: list[str] = []

    def set_num_interop_threads(self, value: int) -> None:
        self.calls.append(f"set_interop:{value}")
        if self.fail_interop:
            raise RuntimeError("frozen failure")
        self.interop = value

    def set_num_threads(self, value: int) -> None:
        self.calls.append(f"set_intraop:{value}")
        self.intraop = value

    def use_deterministic_algorithms(self, value: bool) -> None:
        self.calls.append(f"deterministic:{value}")
        self.deterministic = value

    def get_num_interop_threads(self) -> int:
        return self.interop

    def get_num_threads(self) -> int:
        return self.intraop

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic


def _fake_parent(events: list[str]) -> SimpleNamespace:
    class FakeJ1:
        @staticmethod
        def initialize_model_optimizer():
            events.append("model_initialized")
            return object(), object()

    return SimpleNamespace(j1=FakeJ1())


def _fake_subprocess_runner(
    arguments: list[str],
    *,
    nice_10: bool,
    runtime_passes: bool = True,
) -> dict:
    if arguments[0] == "_root-cause-probe":
        payload = {
            "passes": True,
            "runtime": {
                "torch_num_interop_threads": 12,
                "torch_num_threads": 1,
                "deterministic_algorithms": True,
            },
            "zero_counts": {
                "completed_roots": 0,
                "attempts_started": 0,
                "attempts_finished": 0,
                "attempts_abandoned": 0,
                "optimizer_steps": 0,
                "round_aggregates": 0,
            },
        }
    else:
        operational_checks = {
            "services_healthy": runtime_passes,
            "one_heavy_process": True,
            "free_disk_hard_floor": True,
            "free_disk_target": True,
            "nice_at_least_10": True,
        }
        payload = {
            "passes": runtime_passes,
            "runtime": {
                "torch_num_interop_threads": 1,
                "torch_num_threads": 1,
                "deterministic_algorithms": True,
            },
            "checks": {
                "first_real_operational_guard_passed": runtime_passes,
            },
            "operational_audit": {
                "passes": runtime_passes,
                "checks": operational_checks,
            },
            "scientific_artifacts": {
                "owners": 0,
                "stream_reservations": 0,
                "stream_consumptions": 0,
                "genesis_commits": 0,
                "games": 0,
                "optimizer_steps": 0,
            },
        }
    return {
        "command": ["fixture", *arguments],
        "returncode": 0 if payload["passes"] else 2,
        "stdout": json.dumps(payload),
        "stderr": "",
        "payload": payload,
        "passes": payload["passes"],
        "nice_10": nice_10,
    }


def _write_fixture_evidence(readiness_dir: Path) -> dict:
    return j1b.write_test_evidence(
        readiness_dir=readiness_dir,
        py_compile_command="fixture py_compile",
        focused_command="fixture focused",
        focused_passed=1,
        parent_execution_command="fixture parent execution",
        parent_execution_passed=1,
        parent_j1_command="fixture parent j1",
        parent_j1_passed=1,
        parent_j1a_command="fixture parent j1a",
        parent_j1a_passed=1,
        applicable_command="fixture applicable",
        applicable_passed=1,
        documented_deselections=[],
    )


def test_module_import_is_torch_and_parent_light() -> None:
    code = (
        "import json,sys;"
        "import threes_rl.j1b_operational_repair_preflight;"
        "print(json.dumps({'torch':'torch' in sys.modules,"
        "'parent':'threes_rl.j1_execution_surface' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=j1b.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "torch": False,
        "parent": False,
    }


def test_runtime_configuration_order_precedes_parent_and_guard(
    tmp_path: Path,
) -> None:
    torch = FakeTorch()
    events: list[str] = []

    def parent_loader():
        events.append("parent_loaded")
        return _fake_parent(events)

    def audit(_parent, _phase_dir):
        events.append("guard")
        assert torch.interop == 1
        assert torch.intraop == 1
        assert torch.deterministic is True
        return {"passes": True}

    def after(_parent, _model, _optimizer):
        events.append("after_guard")
        return {"ok": True}

    result = j1b.guarded_runtime_entrypoint(
        phase_dir=tmp_path,
        torch_module=torch,
        parent_loader=parent_loader,
        operational_audit=audit,
        after_guard=after,
    )
    assert torch.calls == [
        "set_interop:1",
        "set_intraop:1",
        "deterministic:True",
    ]
    assert events == [
        "parent_loaded",
        "model_initialized",
        "guard",
        "after_guard",
    ]
    assert result["ordering"] == [
        "configure_torch_runtime",
        "import_parent",
        "initialize_frozen_model_optimizer",
        "first_unchanged_operational_guard",
        "guard_passed_before_scientific_artifacts",
        "after_guard_callback",
    ]


def test_configuration_failure_stops_before_parent_or_science(
    tmp_path: Path,
) -> None:
    torch = FakeTorch(fail_interop=True)
    events: list[str] = []
    with pytest.raises(j1b.J1bOperationalHold):
        j1b.guarded_runtime_entrypoint(
            phase_dir=tmp_path,
            torch_module=torch,
            parent_loader=lambda: events.append("parent"),
            operational_audit=lambda *_: events.append("audit"),
            after_guard=lambda *_: events.append("reservation"),
        )
    assert torch.calls == ["set_interop:1"]
    assert events == []


def test_guard_failure_stops_before_after_guard(tmp_path: Path) -> None:
    torch = FakeTorch()
    events: list[str] = []
    with pytest.raises(j1b.J1bOperationalHold):
        j1b.guarded_runtime_entrypoint(
            phase_dir=tmp_path,
            torch_module=torch,
            parent_loader=lambda: _fake_parent(events),
            operational_audit=lambda *_: {"passes": False},
            after_guard=lambda *_: events.append("reservation"),
        )
    assert events == ["model_initialized"]


def test_clean_subprocess_production_runtime_guard(tmp_path: Path) -> None:
    future_root = tmp_path / "future_execution"
    result = j1b._run_json_subprocess(
        [
            "_runtime-probe",
            "--phase-dir",
            str(tmp_path),
            "--future-execution-root",
            str(future_root),
        ],
        nice_10=True,
    )
    assert result["passes"], result
    payload = result["payload"]
    assert payload["runtime"] == {
        "torch_num_interop_threads": 1,
        "torch_num_threads": 1,
        "deterministic_algorithms": True,
        "checks": {
            "one_torch_interop_thread": True,
            "one_torch_intraop_thread": True,
            "deterministic_algorithms": True,
        },
        "passes": True,
    }
    assert payload["checks"]["first_real_operational_guard_passed"]
    assert not future_root.exists()


def test_legacy_root_cause_probe_is_genesis_zero_work() -> None:
    result = j1b._run_json_subprocess(
        ["_root-cause-probe"],
        nice_10=False,
    )
    assert result["passes"], result
    payload = result["payload"]
    assert payload["runtime"] == {
        "torch_num_interop_threads": 12,
        "torch_num_threads": 1,
        "deterministic_algorithms": True,
    }
    assert payload["zero_counts"] == {
        "completed_roots": 0,
        "attempts_started": 0,
        "attempts_finished": 0,
        "attempts_abandoned": 0,
        "optimizer_steps": 0,
        "round_aggregates": 0,
    }
    assert payload["commit_boundary"]["sequence"] == 0
    assert payload["commit_boundary"]["unit_id"] == "genesis"


def test_runtime_repair_preserves_initial_model_identity(
    tmp_path: Path,
) -> None:
    legacy = j1b._run_json_subprocess(
        ["_root-cause-probe"],
        nice_10=False,
    )
    repaired = j1b._run_json_subprocess(
        [
            "_runtime-probe",
            "--phase-dir",
            str(tmp_path),
            "--future-execution-root",
            str(tmp_path / "future"),
        ],
        nice_10=True,
    )
    assert legacy["passes"] and repaired["passes"]
    assert (
        legacy["payload"]["model"]["initial_state_sha256"]
        == repaired["payload"]["model"]["initial_state_sha256"]
    )
    assert legacy["payload"]["model"]["parameter_count"] == 411_656
    assert repaired["payload"]["model"]["parameter_count"] == 411_656


def test_parent_and_spent_execution_identities_are_exact() -> None:
    assert j1b.parent_identity_audit()["passes"]
    audit = j1b.original_execution_identity_audit()
    assert audit["passes"]
    assert len(audit["actual_files"]) == 14
    assert audit["terminal"]["decision"] == "HOLD_J1_OPERATIONAL"


def test_fresh_stream_rows_are_exact_next_contiguous_prefix() -> None:
    rows = list(j1b.iter_fresh_stream_rows())
    assert len(rows) == 16_384
    assert rows[0]["logical_stream_id"] == 213_000_016_384
    assert rows[-1]["logical_stream_id"] == 213_000_032_767
    assert rows[0]["deck_stream_id"] == 214_000_016_384
    assert rows[-1]["slot_stream_id"] == 215_000_032_767
    assert rows[-1]["candidate_policy_stream_id"] == 216_000_032_767
    assert all(row["starter_tile"] is None for row in rows)


def test_prospective_manifest_has_exact_root_ancestry_contract(
    fresh_manifest: dict,
) -> None:
    assert fresh_manifest["passes"]
    assert fresh_manifest["role_counts"]["roots"] == 16_384
    rows = fresh_manifest["rows"]
    assert len({row["root_id"] for row in rows}) == 16_384
    assert [row["root_id"] for row in rows] == [
        row["ancestry_id"] for row in rows
    ]
    assert j1b.verify_payload_hash(
        fresh_manifest,
        "prospective_manifest_payload_sha256",
    )


def test_root_identity_matches_accepted_parent_formula(
    fresh_manifest: dict,
) -> None:
    from threes_rl import j1_execution_surface as parent

    commitment = fresh_manifest["root_commitment"]
    for row in (
        fresh_manifest["rows"][0],
        fresh_manifest["rows"][8191],
        fresh_manifest["rows"][-1],
    ):
        assert row["root_id"] == parent.root_id_for_marker_commitment(
            commitment,
            row,
        )


def test_fresh_denylist_has_zero_interval_or_actual_collision(
    fresh_manifest: dict,
) -> None:
    denylist = j1b.protected_stream_denylist(fresh_manifest)
    assert denylist["passes"]
    assert denylist["collision_rows"] == []
    assert denylist["original_actual_intersections"] == {
        "logical": 0,
        "deck": 0,
        "slot": 0,
        "candidate_policy": 0,
    }
    assert all(
        row["end_inclusive"] - row["start"] + 1 == 16_384
        for row in denylist["fresh_intervals"]
    )


def test_projection_reuses_parent_byte_exact_without_retiming() -> None:
    projection = j1b.runtime_storage_projection()
    assert projection["passes"]
    assert projection["checks"]["no_retiming"]
    assert (
        projection["parent_projection"]["file_sha256"]
        == j1b.PARENT_READINESS_IDENTITIES[
            "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json"
        ]
    )
    assert (
        projection["training_central"]["storage"][
            "projected_with_margin_gib"
        ]
        < 24.0
    )
    assert projection["training_central"]["hours_with_25pct_margin"] < 72.0


def test_schema_contains_no_phase_execution_commands() -> None:
    schema = j1b.schema_payload()
    assert schema["future_commands_in_this_version"] == []
    assert schema["permitted_commands"] == [
        "write-test-evidence",
        "prepare",
    ]
    parser = j1b.build_parser()
    choices = next(
        action.choices
        for action in parser._actions
        if getattr(action, "dest", None) == "subcommand"
    )
    assert not {"seal-phase-lock", "open", "materialize", "execute"} & set(
        choices
    )


def test_test_evidence_is_create_once_and_zero_work(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness"
    evidence = _write_fixture_evidence(readiness)
    assert evidence["zero_work"] == j1b.ZERO_WORK
    assert all(value == 0 for value in evidence["zero_work"].values())
    with pytest.raises(FileExistsError):
        _write_fixture_evidence(readiness)


def test_test_evidence_cli_returns_success(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness"
    command = [
        sys.executable,
        "-m",
        "threes_rl.j1b_operational_repair_preflight",
        "write-test-evidence",
        "--readiness-dir",
        str(readiness),
        "--py-compile-command",
        "fixture py_compile",
        "--focused-command",
        "fixture focused",
        "--focused-passed",
        "1",
        "--parent-execution-command",
        "fixture parent execution",
        "--parent-execution-passed",
        "1",
        "--parent-j1-command",
        "fixture parent j1",
        "--parent-j1-passed",
        "1",
        "--parent-j1a-command",
        "fixture parent j1a",
        "--parent-j1a-passed",
        "1",
        "--applicable-command",
        "fixture applicable",
        "--applicable-passed",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=j1b.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["passes"] is True
    assert payload["zero_work"] == j1b.ZERO_WORK


def test_pre_a1_failed_status_evidence_is_preserved() -> None:
    audit = j1b.pre_a1_history_audit()
    assert audit["passes"]
    assert audit["file_sha256"] == j1b.PRE_A1_HISTORY_FILE_SHA256


def test_prepare_seals_ready_package_without_execution(
    tmp_path: Path,
) -> None:
    readiness = tmp_path / "readiness"
    future = tmp_path / "future"
    _write_fixture_evidence(readiness)
    result = j1b.prepare_readiness(
        readiness_dir=readiness,
        future_execution_root=future,
        subprocess_runner=_fake_subprocess_runner,
    )
    assert result["decision"] == "READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT"
    assert result["passes"]
    assert not future.exists()
    assert set(path.name for path in readiness.iterdir()) == {
        j1b.TEST_EVIDENCE_NAME,
        j1b.ROOT_CAUSE_AUDIT_NAME,
        j1b.DENYLIST_NAME,
        j1b.MANIFEST_NAME,
        j1b.RUNTIME_AUDIT_NAME,
        j1b.PROJECTION_NAME,
        j1b.SCHEMA_NAME,
        j1b.READINESS_LOCK_NAME,
        j1b.READINESS_RESULT_NAME,
    }
    terminal = j1b.load_json(readiness / j1b.READINESS_RESULT_NAME)
    assert terminal["zero_work"] == j1b.ZERO_WORK
    assert terminal["promote"] is False


def test_prepare_operational_failure_holds_before_execution(
    tmp_path: Path,
) -> None:
    readiness = tmp_path / "readiness"
    future = tmp_path / "future"
    _write_fixture_evidence(readiness)

    def failed_runner(arguments, *, nice_10):
        return _fake_subprocess_runner(
            arguments,
            nice_10=nice_10,
            runtime_passes=arguments[0] != "_runtime-probe",
        )

    result = j1b.prepare_readiness(
        readiness_dir=readiness,
        future_execution_root=future,
        subprocess_runner=failed_runner,
    )
    assert result["decision"] == "HOLD_J1B_OPERATIONAL_REPAIR_PREFLIGHT"
    assert not result["passes"]
    assert not future.exists()
    assert all(value == 0 for value in result["zero_work"].values())


def test_prepare_rejects_existing_future_execution_namespace(
    tmp_path: Path,
) -> None:
    readiness = tmp_path / "readiness"
    future = tmp_path / "future"
    _write_fixture_evidence(readiness)
    future.mkdir()
    with pytest.raises(j1b.J1bIntegrityError):
        j1b.prepare_readiness(
            readiness_dir=readiness,
            future_execution_root=future,
            subprocess_runner=_fake_subprocess_runner,
        )


def test_prepare_rejects_unexpected_readiness_artifact(
    tmp_path: Path,
) -> None:
    readiness = tmp_path / "readiness"
    _write_fixture_evidence(readiness)
    (readiness / "unexpected").write_text("no", encoding="utf-8")
    with pytest.raises(j1b.J1bIntegrityError):
        j1b.prepare_readiness(
            readiness_dir=readiness,
            future_execution_root=tmp_path / "future",
            subprocess_runner=_fake_subprocess_runner,
        )


def test_immutable_json_detects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    j1b.write_immutable_json(path, {"value": 1}, field="payload_sha256")
    path.write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(j1b.J1bIntegrityError):
        j1b.write_immutable_json(
            path,
            {"value": 1},
            field="payload_sha256",
        )


def test_original_execution_remains_byte_exact_after_all_fixture_work() -> None:
    assert j1b.original_execution_identity_audit()["passes"]

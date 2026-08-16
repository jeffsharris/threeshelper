from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from threes_rl import g3_e0_label_fit as core
from threes_rl import g3_e0_label_fit_v3 as v3
from threes_rl import g3_e0_preflight_v3 as preflight
from threes_rl.s3_power_preflight import sha256_path


def _write_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    value = core.payload_with_hash(payload)
    core.write_immutable_json(path, value)
    return value


def _pass_audit() -> dict[str, Any]:
    return {"passes": True, "checks": {"ok": True}}


def _lock_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    out_dir = tmp_path / "g3_e0_label_fit_v3"
    out_dir.mkdir()
    records = _write_payload(
        out_dir / "E0_RECORD_MANIFEST.json",
        {"version": "fixture-records", "records": []},
    )
    tasks = _write_payload(
        out_dir / "E0_TASK_MANIFEST.json",
        {"version": "fixture-tasks", "tasks": []},
    )
    streams = _write_payload(
        out_dir / "E0_STREAM_MANIFEST.json",
        {"version": "fixture-streams", "rows": []},
    )
    lock = core.payload_with_hash(
        {
            "version": preflight.VERSION,
            "decision": v3.PREFLIGHT_DECISION,
            "out_dir_resolved": str(out_dir.resolve()),
            "open_command": v3.OPEN_COMMAND,
            "execution_command": v3.EXECUTE_COMMAND,
            "jobs": 1,
            "nice": 10,
            "maximum_active_hours": 18.0,
            "maximum_output_bytes": core.MAX_OUTPUT_BYTES,
            "minimum_free_gib": 100.0,
            "target_free_gib": 120.0,
            "required_services": ["dashboard", "advisor"],
            "orchestration_runner_sha256": sha256_path(v3.RUNNER_PATH),
            "focused_test_sha256": "focused",
            "test_evidence_file_sha256": "evidence",
            "bound_files": [],
            "record_manifest_name": "E0_RECORD_MANIFEST.json",
            "record_manifest_file_sha256": sha256_path(
                out_dir / "E0_RECORD_MANIFEST.json"
            ),
            "record_manifest_payload_sha256":
                records["canonical_payload_sha256"],
            "task_manifest_name": "E0_TASK_MANIFEST.json",
            "task_manifest_file_sha256": sha256_path(
                out_dir / "E0_TASK_MANIFEST.json"
            ),
            "task_manifest_payload_sha256":
                tasks["canonical_payload_sha256"],
            "stream_manifest_name": "E0_STREAM_MANIFEST.json",
            "stream_manifest_file_sha256": sha256_path(
                out_dir / "E0_STREAM_MANIFEST.json"
            ),
            "stream_manifest_payload_sha256":
                streams["canonical_payload_sha256"],
            "incumbent_policy_spec": "fixture",
            "incumbent_artifact_audit_sha256": "incumbent",
            "transfer_preflight_lock_path": "fixture-transfer-lock.json",
            "ordinary_record_count": 683,
            "transfer_record_count": 32,
            "total_path_count": 5_072,
            "path_counts": {
                "train": 3_902,
                "development": 944,
                "transfer_diagnostic": 226,
            },
            "ordinary_path_count": 4_846,
            "transfer_path_count": 226,
            "replicates": [0, 1],
            "stream_collision_excluded_directories": [],
            "stream_collision_lock": {
                "matched_source_count": 0,
                "matched_sources_sha256": "sources",
                "excluded_source_count": 0,
                "excluded_sources_sha256": "excluded",
            },
            "canonical_payload_sha256_placeholder": False,
        }
    )
    lock_path = out_dir / "preflight_lock.json"
    core.write_immutable_json(lock_path, lock)
    return out_dir, lock_path, lock


def _patch_open_audits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v3, "_verify_bound_files", lambda _lock: _pass_audit())
    monkeypatch.setattr(
        v3,
        "_manifest_audit",
        lambda _lock, *, out_dir: _pass_audit(),
    )
    monkeypatch.setattr(v3, "_incumbent_audit", lambda _lock: _pass_audit())
    monkeypatch.setattr(
        v3,
        "_stream_collision_audit",
        lambda _lock, *, out_dir: _pass_audit(),
    )
    monkeypatch.setattr(
        v3,
        "_operational_audit",
        lambda _out_dir: _pass_audit(),
    )


def test_open_only_seals_exact_contract_and_creates_no_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir, lock_path, _lock = _lock_fixture(tmp_path)
    _patch_open_audits(monkeypatch)
    marker = v3.open_execution(
        out_dir=out_dir,
        preflight_lock=lock_path,
        jobs=1,
    )
    assert core.verify_payload_hash(marker)
    assert marker["contract"]["open_command"] == v3.OPEN_COMMAND
    assert marker["contract"]["execution_command"] == v3.EXECUTE_COMMAND
    assert marker["contract"]["ordinary_records"] == 683
    assert marker["contract"]["transfer_roots"] == 32
    assert marker["contract"]["total_paths"] == 5_072
    assert marker["contract"]["replicates"] == [0, 1]
    assert marker["zero_work_before_open"]["streams_consumed"] == 0
    assert marker["zero_work_before_open"]["label_paths_generated"] == 0
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(
        v3.ALLOWED_BASE_FILES | {v3.OPEN_MARKER_NAME}
    )


def test_open_only_is_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir, lock_path, _lock = _lock_fixture(tmp_path)
    _patch_open_audits(monkeypatch)
    v3.open_execution(
        out_dir=out_dir,
        preflight_lock=lock_path,
        jobs=1,
    )
    with pytest.raises(FileExistsError, match="not zero-work"):
        v3.open_execution(
            out_dir=out_dir,
            preflight_lock=lock_path,
            jobs=1,
        )


def test_marker_missing_and_jobs_mismatch_fail_before_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir, lock_path, _lock = _lock_fixture(tmp_path)
    called = False

    def forbidden(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(v3, "_execute_pipeline", forbidden)
    with pytest.raises(FileNotFoundError, match="Missing"):
        v3.execute(
            out_dir=out_dir,
            preflight_lock=lock_path,
            jobs=1,
        )
    with pytest.raises(ValueError, match="jobs=1"):
        v3.execute(
            out_dir=out_dir,
            preflight_lock=lock_path,
            jobs=2,
        )
    assert not called


def test_marker_contract_mismatch_rejected_before_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir, lock_path, _lock = _lock_fixture(tmp_path)
    _patch_open_audits(monkeypatch)
    marker = v3.open_execution(
        out_dir=out_dir,
        preflight_lock=lock_path,
        jobs=1,
    )
    marker["contract"]["execution_command"] = "changed"
    marker = core.payload_with_hash(
        {
            key: value
            for key, value in marker.items()
            if key != "canonical_payload_sha256"
        }
    )
    marker_path = out_dir / v3.OPEN_MARKER_NAME
    marker_path.unlink()
    core.write_immutable_json(marker_path, marker)
    with pytest.raises(ValueError, match="contract mismatch"):
        v3.execute(
            out_dir=out_dir,
            preflight_lock=lock_path,
            jobs=1,
        )


def test_same_marker_validates_twice_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir, lock_path, _lock = _lock_fixture(tmp_path)
    _patch_open_audits(monkeypatch)
    v3.open_execution(
        out_dir=out_dir,
        preflight_lock=lock_path,
        jobs=1,
    )
    marker_path = out_dir / v3.OPEN_MARKER_NAME
    before = sha256_path(marker_path)
    first_lock, first = v3.validate_execution_marker(
        out_dir=out_dir,
        preflight_lock=lock_path,
        jobs=1,
        revalidate_operations=False,
    )
    second_lock, second = v3.validate_execution_marker(
        out_dir=out_dir,
        preflight_lock=lock_path,
        jobs=1,
        revalidate_operations=False,
    )
    assert first_lock == second_lock
    assert first["marker_file_sha256"] == before
    assert second["marker_file_sha256"] == before
    assert sha256_path(marker_path) == before


def test_terminal_result_is_immutable_and_blocks_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir, lock_path, _lock = _lock_fixture(tmp_path)
    _patch_open_audits(monkeypatch)
    v3.open_execution(
        out_dir=out_dir,
        preflight_lock=lock_path,
        jobs=1,
    )
    terminal = _write_payload(
        out_dir / v3.TERMINAL_RESULT_NAME,
        {"version": v3.VERSION, "decision": "HOLD"},
    )
    terminal_hash = sha256_path(out_dir / v3.TERMINAL_RESULT_NAME)
    with pytest.raises(FileExistsError, match="immutable"):
        v3.execute(
            out_dir=out_dir,
            preflight_lock=lock_path,
            jobs=1,
        )
    assert core.verify_payload_hash(terminal)
    assert sha256_path(out_dir / v3.TERMINAL_RESULT_NAME) == terminal_hash


def test_label_store_same_identity_resume_and_transfer_barrier(
    tmp_path: Path,
) -> None:
    database = tmp_path / "labels.sqlite3"
    identity = {"marker": "same"}
    with core.LabelStore(database, identity=identity) as store:
        store.insert_chunk([{"task_key": "one", "value": 1}])
    with core.LabelStore(database, identity=identity) as store:
        assert store.completed_keys() == {"one"}
        store.insert_chunk([{"task_key": "one", "value": 1}])
        assert store.count() == 1
    with pytest.raises(PermissionError, match="checkpoint and prediction"):
        core.LabelStore(
            tmp_path / "transfer.sqlite3",
            identity={"marker": "same", "partition": "transfer"},
            transfer=True,
            checkpoint_seal=tmp_path / "missing-checkpoint.json",
            prediction_seal=tmp_path / "missing-prediction.json",
        )


def test_checkpoint_resume_requires_same_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "checkpoint-resume"
    model_dir = out_dir / core.MODEL_DIR_NAME
    model_dir.mkdir(parents=True)
    (model_dir / "meta.json").write_text("{}")
    (model_dir / "arrays.npz").write_bytes(b"arrays")
    (out_dir / core.ORDINARY_DB_NAME).write_bytes(b"labels")
    preflight_lock = out_dir / "preflight.json"
    preflight_lock.write_text("{}")
    checkpoint = _write_payload(
        out_dir / core.CHECKPOINT_SEAL_NAME,
        {
            "model_meta_sha256": sha256_path(model_dir / "meta.json"),
            "model_arrays_sha256": sha256_path(model_dir / "arrays.npz"),
            "ordinary_labels_sha256":
                sha256_path(out_dir / core.ORDINARY_DB_NAME),
            "execution_marker_file_sha256": "same-marker",
            "ordinary_decision": "READY_G3_E0_ORDINARY_PREDICTIVE",
        },
    )
    fake_model = object()
    monkeypatch.setattr(
        core.G3HazardModel,
        "load",
        lambda *_args, **_kwargs: fake_model,
    )
    model, loaded = v3._load_checkpoint(
        out_dir=out_dir,
        preflight_lock=preflight_lock,
        marker_file_sha256="same-marker",
    )
    assert model is fake_model
    assert loaded == checkpoint
    with pytest.raises(ValueError, match="artifact mismatch"):
        v3._load_checkpoint(
            out_dir=out_dir,
            preflight_lock=preflight_lock,
            marker_file_sha256="different-marker",
        )


def test_ordinary_kill_seals_checkpoint_before_any_transfer_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "pipeline"
    out_dir.mkdir()
    _write_payload(
        out_dir / "records.json",
        {"records": []},
    )
    _write_payload(
        out_dir / "tasks.json",
        {"tasks": []},
    )
    _write_payload(
        out_dir / "preflight.json",
        {"decision": "fixture"},
    )
    (out_dir / core.ORDINARY_DB_NAME).write_bytes(b"")
    lock = {
        "record_manifest_name": "records.json",
        "task_manifest_name": "tasks.json",
        "incumbent_policy_spec": "fixture",
        "canonical_payload_sha256": "preflight-payload",
        "task_manifest_file_sha256": "task-file",
        "stream_manifest_file_sha256": "stream-file",
        "ordinary_path_count": 0,
        "transfer_path_count": 0,
    }

    class FakeStore:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> "FakeStore":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def completed_keys(self) -> set[str]:
            return set()

        def insert_chunk(self, _rows: Any) -> None:
            raise AssertionError("No task should be generated")

        def rows(self) -> list[dict[str, Any]]:
            return []

    class FakeModel:
        def save(self, directory: Path) -> None:
            directory.mkdir()
            (directory / "meta.json").write_text("{}")
            (directory / "arrays.npz").write_bytes(b"arrays")

    monkeypatch.setattr(core, "LabelStore", FakeStore)
    monkeypatch.setattr(core, "make_policy", lambda _spec: object())
    monkeypatch.setattr(
        core, "aggregate_grouped_rows", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        core, "fit_hazard_model", lambda *_args, **_kwargs: FakeModel()
    )
    monkeypatch.setattr(
        core, "model_stability_audit", lambda _model: {"passes": True}
    )
    monkeypatch.setattr(
        core,
        "predictive_report",
        lambda *_args, **_kwargs: {"ordinary": True},
    )
    monkeypatch.setattr(
        core,
        "ordinary_gate_decision",
        lambda *_args, **_kwargs: "KILL_G3_BOOTSTRAP_PREDICTIVE",
    )
    monkeypatch.setattr(
        core,
        "predict_transfer_actions",
        lambda *_args, **_kwargs: pytest.fail(
            "Transfer predictions opened before checkpoint gate"
        ),
    )
    result = v3._execute_pipeline(
        out_dir=out_dir,
        preflight_lock=out_dir / "preflight.json",
        lock=lock,
        marker_file_sha256="marker",
    )
    assert result["decision"] == "KILL_G3_BOOTSTRAP_PREDICTIVE"
    assert (out_dir / core.CHECKPOINT_SEAL_NAME).is_file()
    assert not (out_dir / core.PREDICTION_SEAL_NAME).exists()
    assert not (out_dir / core.TRANSFER_DB_NAME).exists()


def test_state_machine_is_explicit_and_does_not_patch_core_marker() -> None:
    source = inspect.getsource(v3)
    assert "seal_execution_opened =" not in source
    assert "monkeypatch" not in source
    assert "def open_execution(" in source
    assert "def validate_execution_marker(" in source
    pipeline = inspect.getsource(v3._execute_pipeline)
    assert pipeline.index("checkpoint_path") < pipeline.index(
        "core.predict_transfer_actions"
    )
    assert pipeline.index("core.predict_transfer_actions") < pipeline.index(
        "core.TRANSFER_DB_NAME"
    )


def test_v2_ready_directory_remains_zero_work() -> None:
    audit = preflight._v2_zero_work_audit()
    assert audit["passes"], audit

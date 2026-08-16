from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from threes_rl import j1_execution_surface as parent
from threes_rl import j1b_training_execution_surface as surface


def _fixture_rows(count: int = 2) -> list[dict]:
    rows = []
    for index in range(count):
        root_id = hashlib.sha256(
            f"j1b-fixture-root-{index}".encode("ascii")
        ).hexdigest()
        row = {
            "phase": "training",
            "partition": "train",
            "row_index": index,
            "block": index % 8,
            "logical_stream_id": 9_100_000 + index,
            "deck_stream_id": 9_200_000 + index,
            "slot_stream_id": 9_300_000 + index,
            "candidate_policy_stream_id": 9_400_000 + index,
            "control_policy_stream_id": None,
            "arm_count": 1,
            "starter_tile": None,
            "root_id": root_id,
            "ancestry_id": root_id,
        }
        row["row_commitment_sha256"] = surface.canonical_json_hash(
            {
                key: value
                for key, value in row.items()
                if key not in {"root_id", "ancestry_id"}
            }
        )
        rows.append(row)
    return rows


def _fixture_chain(
    root: Path,
    *,
    first_guard_passes: bool = True,
) -> tuple[Path, Path]:
    readiness = root / "readiness"
    execution = root / "execution"
    surface.write_fixture_readiness(
        readiness_dir=readiness,
        execution_root=execution,
        rows=_fixture_rows(),
        engine_config={
            "rounds": 1,
            "roots_per_round": 2,
            "env_count": 2,
            "minibatch_size": 32,
            "max_moves": parent.MAX_MOVES,
        },
        first_guard_passes=first_guard_passes,
    )
    surface.seal_training_phase_lock(
        execution_root=execution,
        readiness_dir=readiness,
    )
    surface.open_training_phase(
        execution_root=execution,
        readiness_dir=readiness,
        opened_at="2026-07-27T00:00:00Z",
        hostname="fixture-host",
    )
    surface.materialize_training_manifest(
        execution_root=execution,
        readiness_dir=readiness,
    )
    return readiness, execution


def _cli(
    action: str,
    readiness: Path,
    execution: Path,
    *,
    env_add: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    if env_add:
        env.update(env_add)
    return subprocess.run(
        [
            "nice",
            "-n",
            "10",
            sys.executable,
            "-m",
            "threes_rl.j1b_training_execution_surface",
            action,
            "--execution-root",
            str(execution),
            "--readiness-dir",
            str(readiness),
            "--jobs",
            "1",
        ],
        cwd=surface.REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _json_stdout(
    completed: subprocess.CompletedProcess[str],
) -> dict:
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _assert_zero_post_materialization_work(execution: Path) -> None:
    paths = surface.phase_paths(execution)
    assert not paths["owner"].exists()
    assert not paths["reservation"].exists()
    assert not paths["consumption"].exists()
    assert not paths["commit_head"].exists()
    assert not paths["result"].exists()
    assert not paths["checkpoint"].exists()


def _terminal_state(execution: Path) -> dict:
    paths = surface.phase_paths(execution)
    result = surface.load_json(paths["result"])
    assert surface.verify_payload_hash(
        result,
        "terminal_result_payload_sha256",
    )
    boundary = parent.verify_commit_boundary(
        phase_dir=paths["phase_dir"],
        phase="training",
        marker_file_sha256=result["marker_identity"]["file_sha256"],
        phase_lock_file_sha256=result["phase_lock_identity"][
            "file_sha256"
        ],
        command=result["execute_command"],
        execution_mode="miniature_fixture",
    )
    return parent.load_atomic_binary(Path(boundary["state_path"]))


def _assert_states_bit_equal(left: dict, right: dict) -> None:
    for key in (
        "all_completed_root_ids",
        "optimizer_step_ids",
        "expected_optimizer_step_ids",
        "round_aggregates",
    ):
        assert parent.j1.stable_hash(left[key]) == parent.j1.stable_hash(
            right[key]
        )
    for key, tensor in left["model_state"].items():
        assert torch.equal(tensor, right["model_state"][key])
    assert parent.j1.stable_hash(left["optimizer_state"]) == (
        parent.j1.stable_hash(right["optimizer_state"])
    )


def test_charter_is_frozen() -> None:
    assert surface.sha256_path(surface.CHARTER_PATH) == (
        surface.EXPECTED_CHARTER_SHA256
    )


def test_module_import_is_torch_and_parent_light() -> None:
    code = (
        "import json,sys;"
        "import threes_rl.j1b_training_execution_surface as module;"
        "print(json.dumps({'torch':'torch' in sys.modules,"
        "'parent':'threes_rl.j1_execution_surface' in sys.modules,"
        "'commands':module.PUBLIC_COMMANDS}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=surface.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "torch": False,
        "parent": False,
        "commands": list(surface.PUBLIC_COMMANDS),
    }


def test_public_parser_is_training_only() -> None:
    parser = surface.build_parser()
    actions = parser._subparsers._group_actions[0].choices
    assert tuple(actions) == surface.PUBLIC_COMMANDS
    for forbidden in (
        "development",
        "confirmation",
        "promote",
        "restart",
        "inspect-outcomes",
    ):
        assert forbidden not in actions


def test_authoritative_j1b_inputs_reproduce_exactly() -> None:
    audit = surface.audit_authoritative_inputs(
        require_future_execution_absent=True,
    )
    assert audit["passes"]
    assert audit["source_manifest_validation"]["row_count"] == 16_384
    assert (
        audit["source_manifest_validation"]["root_set_sha256"]
        == surface.EXPECTED_ROOT_SET_SHA256
    )


def test_exact_source_manifest_materializes_without_regeneration() -> None:
    source = surface.load_json(surface._source_manifest_path())
    identity = surface.immutable_json_identity(
        surface._source_manifest_path(),
        payload_field="prospective_manifest_payload_sha256",
    )
    manifest = surface.build_materialized_manifest(
        source,
        source_identity=identity,
        scientific=True,
    )
    assert manifest["rows"] == source["rows"]
    assert len(manifest["rows"]) == 16_384
    assert manifest["canonical_rows_sha256"] == (
        surface.EXPECTED_CANONICAL_ROWS_SHA256
    )
    assert manifest["root_set_sha256"] == (
        surface.EXPECTED_ROOT_SET_SHA256
    )


def test_phase_lock_open_and_materialize_are_ordered(
    tmp_path: Path,
) -> None:
    readiness = tmp_path / "readiness"
    execution = tmp_path / "execution"
    surface.write_fixture_readiness(
        readiness_dir=readiness,
        execution_root=execution,
        rows=_fixture_rows(),
        engine_config={
            "rounds": 1,
            "roots_per_round": 2,
            "env_count": 2,
            "minibatch_size": 32,
            "max_moves": parent.MAX_MOVES,
        },
    )
    locked = surface.seal_training_phase_lock(
        execution_root=execution,
        readiness_dir=readiness,
    )
    assert locked["passes"]
    paths = surface.phase_paths(execution)
    assert paths["lock"].is_file()
    assert paths["lock_result"].is_file()
    assert not paths["marker"].exists()
    opened = surface.open_training_phase(
        execution_root=execution,
        readiness_dir=readiness,
        opened_at="2026-07-27T00:00:00Z",
        hostname="fixture-host",
    )
    assert opened["created_after_open"]["marker"]
    assert not paths["manifest"].exists()
    materialized = surface.materialize_training_manifest(
        execution_root=execution,
        readiness_dir=readiness,
    )
    assert materialized["manifest"]["rows"] == _fixture_rows()
    _assert_zero_post_materialization_work(execution)


def test_phase_lock_is_create_once(tmp_path: Path) -> None:
    readiness = tmp_path / "readiness"
    execution = tmp_path / "execution"
    surface.write_fixture_readiness(
        readiness_dir=readiness,
        execution_root=execution,
        rows=_fixture_rows(),
        engine_config={
            "rounds": 1,
            "roots_per_round": 2,
            "env_count": 2,
            "minibatch_size": 32,
            "max_moves": parent.MAX_MOVES,
        },
    )
    surface.seal_training_phase_lock(
        execution_root=execution,
        readiness_dir=readiness,
    )
    original = surface.phase_paths(execution)["lock"].read_bytes()
    with pytest.raises(FileExistsError):
        surface.seal_training_phase_lock(
            execution_root=execution,
            readiness_dir=readiness,
        )
    assert surface.phase_paths(execution)["lock"].read_bytes() == original


def test_concurrent_open_has_one_immutable_winner(
    tmp_path: Path,
) -> None:
    readiness = tmp_path / "readiness"
    execution = tmp_path / "execution"
    surface.write_fixture_readiness(
        readiness_dir=readiness,
        execution_root=execution,
        rows=_fixture_rows(),
        engine_config={
            "rounds": 1,
            "roots_per_round": 2,
            "env_count": 2,
            "minibatch_size": 32,
            "max_moves": parent.MAX_MOVES,
        },
    )
    surface.seal_training_phase_lock(
        execution_root=execution,
        readiness_dir=readiness,
    )

    def run() -> subprocess.CompletedProcess[str]:
        return _cli("open", readiness, execution)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sorted(result.returncode for result in results) == [0, 2]
    marker = surface.phase_paths(execution)["marker"]
    before = marker.read_bytes()
    assert surface.verify_payload_hash(
        surface.load_json(marker),
        "activation_marker_payload_sha256",
    )
    assert marker.read_bytes() == before


@pytest.mark.parametrize(
    "failure_env",
    [
        {"J1B_FIXTURE_RUNTIME_FAILURE": "1"},
        {"J1B_FIXTURE_FIRST_GUARD_FAILURE": "1"},
    ],
)
def test_runtime_or_first_guard_failure_precedes_owner(
    tmp_path: Path,
    failure_env: dict[str, str],
) -> None:
    readiness, execution = _fixture_chain(tmp_path)
    completed = _cli(
        "execute",
        readiness,
        execution,
        env_add=failure_env,
    )
    assert completed.returncode == 2
    assert _json_stdout(completed)["passes"] is False
    _assert_zero_post_materialization_work(execution)


def test_clean_subprocess_real_runtime_guard_passes(
    tmp_path: Path,
) -> None:
    future = tmp_path / "future"
    completed = subprocess.run(
        [
            "nice",
            "-n",
            "10",
            sys.executable,
            "-m",
            "threes_rl.j1b_operational_repair_preflight",
            "_runtime-probe",
            "--phase-dir",
            str(tmp_path),
            "--future-execution-root",
            str(future),
        ],
        cwd=surface.REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["passes"]
    assert payload["runtime"]["torch_num_interop_threads"] == 1
    assert payload["runtime"]["torch_num_threads"] == 1
    assert payload["checks"]["first_real_operational_guard_passed"]
    assert not future.exists()


def test_fixture_full_chain_routes_bounded_training_only(
    tmp_path: Path,
) -> None:
    readiness, execution = _fixture_chain(tmp_path)
    completed = _cli("execute", readiness, execution)
    assert completed.returncode == 0, completed.stderr
    payload = _json_stdout(completed)
    assert payload["terminal_decision"] == (
        "READY_J1B_MINIATURE_TRAINING_FIXTURE"
    )
    assert payload["ordering"][:5] == [
        "configure_torch_runtime",
        "import_parent",
        "initialize_frozen_model_optimizer",
        "first_unchanged_operational_guard",
        "guard_passed_before_scientific_artifacts",
    ]
    paths = surface.phase_paths(execution)
    result = surface.load_json(paths["result"])
    assert result["bounded_engine"] == "execute_training_engine_bounded"
    assert result["scientific_authority"] is False
    assert result["promote"] is False
    assert paths["retention"].is_file()
    assert not (execution / "development").exists()
    assert not (execution / "confirmation").exists()


@pytest.mark.parametrize(
    "boundary",
    ["after-owner", "after-reservation", "after-consumption"],
)
def test_pre_engine_crash_reclaims_same_contract_without_duplicate_streams(
    tmp_path: Path,
    boundary: str,
) -> None:
    readiness, execution = _fixture_chain(tmp_path)
    interrupted = _cli(
        "execute",
        readiness,
        execution,
        env_add={"J1B_FIXTURE_PRE_ENGINE_INTERRUPT": boundary},
    )
    assert interrupted.returncode == 75
    resumed = _cli("execute", readiness, execution)
    assert resumed.returncode == 0, resumed.stderr
    paths = surface.phase_paths(execution)
    reservation = surface.load_json(paths["reservation"])
    consumption = surface.load_json(paths["consumption"])
    owner = surface.load_json(paths["owner"])
    assert reservation["streams_reserved"] == 8
    assert consumption["streams_consumed"] == 8
    assert len(owner["recoveries"]) == 1
    opener = consumption["owner_record_sha256"]
    current = owner["owners"][-1]["owner_record_sha256"]
    assert surface._owner_is_ancestor(
        owner,
        opener=opener,
        current=current,
    )


def test_collection_update_checkpoint_resume_is_bit_exact(
    tmp_path: Path,
) -> None:
    baseline_readiness, baseline_execution = _fixture_chain(
        tmp_path / "baseline"
    )
    baseline = _cli(
        "execute",
        baseline_readiness,
        baseline_execution,
    )
    assert baseline.returncode == 0, baseline.stderr

    resumed_readiness, resumed_execution = _fixture_chain(
        tmp_path / "resumed"
    )
    for boundary in ("collection", "update", "checkpoint"):
        interrupted = _cli(
            "execute",
            resumed_readiness,
            resumed_execution,
            env_add={
                "J1B_FIXTURE_INTERRUPT_AFTER_BOUNDARY": boundary
            },
        )
        assert interrupted.returncode == 75, interrupted.stdout
    resumed = _cli(
        "execute",
        resumed_readiness,
        resumed_execution,
    )
    assert resumed.returncode == 0, resumed.stderr
    _assert_states_bit_equal(
        _terminal_state(baseline_execution),
        _terminal_state(resumed_execution),
    )


def test_existing_terminal_repairs_or_reverifies_retention(
    tmp_path: Path,
) -> None:
    readiness, execution = _fixture_chain(tmp_path)
    completed = _cli("execute", readiness, execution)
    assert completed.returncode == 0
    retention = surface.phase_paths(execution)["retention"]
    identity_before = surface.immutable_json_identity(
        retention,
        payload_field="retention_payload_sha256",
    )
    repeated = _cli("execute", readiness, execution)
    assert repeated.returncode == 0
    payload = _json_stdout(repeated)
    assert payload["terminal_already_sealed"] is True
    assert surface.immutable_json_identity(
        retention,
        payload_field="retention_payload_sha256",
    ) == identity_before


def test_changed_source_manifest_fails_closed(tmp_path: Path) -> None:
    readiness, execution = _fixture_chain(tmp_path)
    source_path = readiness / "FIXTURE_SOURCE_MANIFEST.json"
    source = surface.load_json(source_path)
    source["rows"][0]["logical_stream_id"] += 1
    source_path.write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.J1bSurfaceIntegrityError):
        surface.load_open_training_contract(
            execution_root=execution,
            readiness_dir=readiness,
            require_manifest=True,
        )


def test_authoritative_fresh_manifest_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "j1b_readiness"
    copied.mkdir()
    for name in (
        "J1B_READINESS_LOCK.json",
        "J1B_READINESS_RESULT.json",
        "J1B_PROSPECTIVE_TRAINING_MANIFEST.json",
    ):
        shutil.copy2(surface.J1B_PREFLIGHT_DIR / name, copied / name)
    monkeypatch.setattr(surface, "J1B_PREFLIGHT_DIR", copied)
    source = copied / "J1B_PROSPECTIVE_TRAINING_MANIFEST.json"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(surface.J1bSurfaceIntegrityError):
        surface.audit_authoritative_inputs(
            require_future_execution_absent=True,
        )


def test_spent_parent_inventory_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "spent_training"
    lock = surface.load_json(
        surface.J1B_PREFLIGHT_DIR / "J1B_READINESS_LOCK.json"
    )
    for relative in lock["spent_j1_execution_identities"]:
        source = surface.SPENT_J1_TRAINING_DIR / relative
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    monkeypatch.setattr(surface, "SPENT_J1_TRAINING_DIR", copied)
    target = copied / "terminal_result.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(surface.J1bSurfaceIntegrityError):
        surface.audit_authoritative_inputs(
            require_future_execution_absent=True,
        )


def test_readiness_result_tamper_fails_closed(tmp_path: Path) -> None:
    readiness, _execution = _fixture_chain(tmp_path)
    path = readiness / surface.READINESS_RESULT_NAME
    payload = surface.load_json(path)
    payload["readiness_lock_identity"]["file_sha256"] = "0" * 64
    payload = surface.payload_with_hash(
        payload,
        "readiness_result_payload_sha256",
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.J1bSurfaceIntegrityError):
        surface.load_ready_surface(readiness)


def test_changed_phase_lock_fails_closed(tmp_path: Path) -> None:
    readiness, execution = _fixture_chain(tmp_path)
    path = surface.phase_paths(execution)["lock"]
    payload = surface.load_json(path)
    payload["bounded_engine"] = "execute_training_engine"
    payload = surface.payload_with_hash(
        payload,
        "phase_lock_payload_sha256",
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.J1bSurfaceIntegrityError):
        surface.load_training_phase_lock(
            execution_root=execution,
            readiness_dir=readiness,
        )


def test_changed_materialized_manifest_fails_closed(
    tmp_path: Path,
) -> None:
    readiness, execution = _fixture_chain(tmp_path)
    path = surface.phase_paths(execution)["manifest"]
    payload = surface.load_json(path)
    payload["rows"][0]["root_id"] = "0" * 64
    payload = surface.payload_with_hash(
        payload,
        "root_manifest_payload_sha256",
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.J1bSurfaceIntegrityError):
        surface.load_open_training_contract(
            execution_root=execution,
            readiness_dir=readiness,
            require_manifest=True,
        )


def test_parent_runtime_validators_do_not_mutate_parent_namespace() -> None:
    before = set(vars(parent))
    model, optimizer = parent.j1.initialize_model_optimizer()
    parent.FrozenMinibatchUpdater._validate_optimizer_binding(
        model,
        optimizer,
    )
    parent.j1.assert_finite_model(model)
    after = set(vars(parent))
    assert before == after


def test_projection_preserves_parent_bounded_contract() -> None:
    projection = surface.runtime_storage_projection()
    assert projection["passes"]
    training = projection["training"]
    assert training["projected_with_margin_gib"] < 24.0
    assert training["runtime_with_margin_hours"] < 72.0
    assert training["created_files"] <= training["created_file_cap"]
    assert training["fsync_count"] <= training["fsync_cap"]
    assert projection["sensitivity_5000_moves"][
        "diagnostic_not_conjunctive"
    ]


def test_schema_has_no_downstream_surface() -> None:
    schema = surface.surface_schema()
    assert surface.verify_payload_hash(schema, "schema_payload_sha256")
    assert schema["public_commands"] == list(surface.PUBLIC_COMMANDS)
    assert schema["development_surface_present"] is False
    assert schema["confirmation_surface_present"] is False
    assert schema["promotion_surface_present"] is False


def test_scientific_fixture_controls_are_rejected_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(surface.RUNNER_PATH).read_text(encoding="utf-8")
    assert "execute_training_engine_bounded(" in source
    assert "parent.execute_training_engine(" not in source
    assert "execute_paired_evaluation_engine" not in source
    assert "--phase" not in source
    monkeypatch.setenv("J1B_FIXTURE_RUNTIME_FAILURE", "1")
    assert os.environ["J1B_FIXTURE_RUNTIME_FAILURE"] == "1"


def test_test_file_has_no_scientific_namespace_creation() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert str(surface.FUTURE_EXECUTION_ROOT) not in source
    assert "starter_tile" in source


def test_runner_can_be_loaded_without_invoking_cli() -> None:
    namespace = runpy.run_path(
        surface.RUNNER_PATH,
        run_name="j1b_surface_import_fixture",
    )
    assert namespace["PUBLIC_COMMANDS"] == surface.PUBLIC_COMMANDS
    assert "main" in namespace

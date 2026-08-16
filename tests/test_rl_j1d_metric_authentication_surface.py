from __future__ import annotations

import hashlib
import json
import math
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
from threes_rl import j1d_metric_authentication_surface as surface


def _fixture_rows(count: int = 2) -> list[dict]:
    rows = []
    for index in range(count):
        root_id = hashlib.sha256(
            f"j1d-fixture-root-{index}".encode("ascii")
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


def _fixture_evidence_commands() -> list[dict]:
    return [
        {
            "kind": kind,
            "command": f"fixture {kind}",
            "returncode": 0,
            "passed": True,
            "test_count": 1,
        }
        for kind in (
            "py_compile",
            "focused_j1d_surface",
            "j1b_terminalization",
            "parent_j1b_training_surface",
            "parent_j1b_preflight",
            "parent_j1_execution_surface",
            "parent_j1_joint_policy_value",
            "parent_j1a_cost_power",
            "clean_process_real_operational_roundtrip",
            "applicable_non_science_regressions",
            "miniature_full_chain",
            "synthetic_64_round_metric_authentication",
            "parent_j1c_training_surface",
        )
    ]


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
            "threes_rl.j1d_metric_authentication_surface",
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
        "import threes_rl.j1d_metric_authentication_surface as module;"
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


def _metric_round_fixture(
    *,
    round_number: int = 1,
    root_count: int = 3,
) -> dict:
    root_metrics = []
    for index in range(root_count):
        root_id = f"metric-root-{round_number:02d}-{index:03d}"
        root_metrics.append(
            {
                "root_id": root_id,
                "ancestry_id": root_id,
                "committed_record_sha256": hashlib.sha256(
                    f"record:{root_id}".encode()
                ).hexdigest(),
                "transition_content_sha256": hashlib.sha256(
                    f"transitions:{root_id}".encode()
                ).hexdigest(),
                "transition_rows": (1, 7, 31, 127)[index % 4],
                "log_score": (
                    1.1 if round_number >= 61 else 1.0
                ) + index * 1e-15,
                "legal_entropy_nats": (
                    0.20000000000000004 + index * 7e-16
                ),
                "value_mse": 0.20 + index * 3e-16,
                "zero_value_mse": 0.30 + index * 2e-16,
                "auxiliary_brier": [
                    0.10 + index * 1e-16,
                    0.20 + index * 2e-16,
                    0.40 + index * 3e-16,
                ],
                "auxiliary_prevalence": [
                    0.50 + index * 1e-16,
                    0.50 - index * 1e-16,
                    0.50,
                ],
            }
        )
    canonical = surface.canonical_root_equal_round_aggregates(
        root_metrics
    )
    return {
        "round": round_number,
        "root_ids": [row["root_id"] for row in root_metrics],
        "root_metrics": root_metrics,
        "root_metrics_sha256": parent.j1.stable_hash(root_metrics),
        "committed_records_sha256": hashlib.sha256(
            f"records:{round_number}".encode()
        ).hexdigest(),
        "transition_buffer_sha256": hashlib.sha256(
            f"buffer:{round_number}".encode()
        ).hexdigest(),
        "transition_rows": sum(
            int(row["transition_rows"]) for row in root_metrics
        ),
        **canonical,
    }


def test_historical_reduction_order_delta_fails_then_canonicalizes() -> None:
    row = _metric_round_fixture()
    legacy = json.loads(json.dumps(row))
    legacy["legal_entropy_nats"] += 2.62641797199592e-10
    legacy["auxiliary_brier"][0] += 4.422892815880708e-10
    legacy["auxiliary_prevalence_brier"][0] += (
        4.555544760864727e-10
    )
    canonical = surface.canonical_root_equal_round_aggregates(
        legacy["root_metrics"]
    )
    assert not math.isclose(
        legacy["legal_entropy_nats"],
        canonical["legal_entropy_nats"],
        rel_tol=0.0,
        abs_tol=surface.CANONICAL_METRIC_ABS_TOLERANCE,
    )
    assert not math.isclose(
        legacy["auxiliary_brier"][0],
        canonical["auxiliary_brier"][0],
        rel_tol=0.0,
        abs_tol=surface.CANONICAL_METRIC_ABS_TOLERANCE,
    )
    assert not math.isclose(
        legacy["auxiliary_prevalence_brier"][0],
        canonical["auxiliary_prevalence_brier"][0],
        rel_tol=0.0,
        abs_tol=surface.CANONICAL_METRIC_ABS_TOLERANCE,
    )
    repaired = surface.canonicalize_round_metric_row(legacy)
    audit = surface.validate_canonical_round_metric_row(repaired)
    assert audit["passes"]
    assert surface._metric_projection(repaired) == canonical
    assert [row["transition_rows"] for row in repaired["root_metrics"]] == [
        1,
        7,
        31,
    ]


def test_canonical_metric_single_bit_and_field_tamper_fail_closed() -> None:
    row = surface.canonicalize_round_metric_row(
        _metric_round_fixture()
    )
    bit_tamper = json.loads(json.dumps(row))
    bit_tamper["legal_entropy_nats"] = math.nextafter(
        float(bit_tamper["legal_entropy_nats"]),
        math.inf,
    )
    audit = surface.validate_canonical_round_metric_row(bit_tamper)
    assert not audit["passes"]
    assert not audit["checks"]["published_projection_hash_exact"]

    field_tamper = json.loads(json.dumps(row))
    field_tamper["root_metrics"][0]["auxiliary_brier"][1] += 1e-6
    field_tamper["root_metrics_sha256"] = parent.j1.stable_hash(
        field_tamper["root_metrics"]
    )
    audit = surface.validate_canonical_round_metric_row(field_tamper)
    assert not audit["passes"]
    assert not audit["checks"]["authentication_root_hash_exact"]


def _synthetic_64_round_report() -> dict:
    rounds = []
    root_ids = []
    for round_number in range(1, 65):
        row = _metric_round_fixture(
            round_number=round_number,
            root_count=256,
        )
        row = surface.canonicalize_round_metric_row(row)
        rounds.append(row)
        root_ids.extend(row["root_ids"])
    return {
        "manifest_root_ids": root_ids,
        "completed_root_ids": list(root_ids),
        "expected_optimizer_step_ids": ["step-a", "step-b"],
        "closed_optimizer_step_ids": ["step-a", "step-b"],
        "rounds": rounds,
        "authenticated_terminal_boundary": {
            "passes": True,
            "chain_audit_passes": True,
            "state_file_sha256": "e" * 64,
        },
        "checkpoint_identity": {
            "round": 64,
            "save_load_exact": True,
            "parameter_count": parent.j1.EXPECTED_PARAMETER_COUNT,
            "model_schema_sha256": parent.j1.model_schema_sha256(),
            "training_state_file_sha256": "e" * 64,
        },
    }


def test_real_shape_64_round_synthetic_authentication_is_ready() -> None:
    report = _synthetic_64_round_report()
    decision = surface.j1d_training_sanity_decision(
        parent=parent,
        report=report,
    )
    assert decision["decision"] == "READY_J1_TRAINING_SANITY"
    assert decision["j1d_metric_authentication"]["passes"]
    assert decision["j1d_metric_authentication"]["rounds"] == 64
    assert len(report["manifest_root_ids"]) == 16_384


def test_synthetic_round_substitution_duplicate_and_order_fail() -> None:
    report = _synthetic_64_round_report()
    report["rounds"][0]["root_ids"][0] = "substituted"
    with pytest.raises(parent.J1ExecutionIntegrityError):
        surface.j1d_training_sanity_decision(
            parent=parent,
            report=report,
        )

    report = _synthetic_64_round_report()
    report["rounds"][0]["root_ids"][1] = report["rounds"][0][
        "root_ids"
    ][0]
    with pytest.raises(parent.J1ExecutionIntegrityError):
        surface.j1d_training_sanity_decision(
            parent=parent,
            report=report,
        )

    report = _synthetic_64_round_report()
    report["rounds"][0]["root_ids"].reverse()
    report["rounds"][0]["root_metrics"].reverse()
    report["rounds"][0]["root_metrics_sha256"] = parent.j1.stable_hash(
        report["rounds"][0]["root_metrics"]
    )
    report["rounds"][0] = surface.canonicalize_round_metric_row(
        report["rounds"][0]
    )
    with pytest.raises(parent.J1ExecutionIntegrityError):
        surface.j1d_training_sanity_decision(
            parent=parent,
            report=report,
        )


def test_json_native_writer_roundtrips_tuple_as_exact_json_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "marker.json"
    written = surface.write_immutable_json(
        path,
        {
            "operational_audit": {
                "services": {
                    "dashboard": {
                        "top_three": (263670, 261369, 258561),
                    }
                }
            }
        },
        field="activation_marker_payload_sha256",
    )
    assert written["operational_audit"]["services"]["dashboard"][
        "top_three"
    ] == [263670, 261369, 258561]
    expected = (
        json.dumps(written, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert path.read_bytes() == expected
    assert surface.verify_payload_hash(
        surface.load_json(path),
        "activation_marker_payload_sha256",
    )


def test_post_write_equal_object_changed_bytes_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact.json"
    original_fsync_parent = surface._fsync_parent

    def mutate_equal_json(target: Path) -> None:
        payload = json.loads(target.read_text(encoding="utf-8"))
        target.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        original_fsync_parent(target)

    monkeypatch.setattr(surface, "_fsync_parent", mutate_equal_json)
    with pytest.raises(
        surface.J1dSurfaceIntegrityError,
        match="changed bytes",
    ):
        surface.write_immutable_json(
            path,
            {"tuple_value": (1, 2, 3)},
            field="payload_sha256",
        )


def test_authoritative_j1d_inputs_reproduce_exactly() -> None:
    audit = surface.audit_authoritative_inputs(
        require_future_execution_absent=True,
    )
    assert audit["passes"]
    assert audit["source_manifest_validation"]["row_count"] == 16_384
    assert (
        audit["source_manifest_validation"]["root_set_sha256"]
        == surface.EXPECTED_ROOT_SET_SHA256
    )
    assert audit["checks"]["spent_j1b_inventory_exact"]
    assert audit["checks"]["external_j1b_terminal_exact"]
    assert audit["checks"]["stream_authority_exact"]


def test_fresh_manifest_and_compact_authority_are_exact() -> None:
    manifest_path = surface._source_manifest_path()
    manifest = surface.load_json(manifest_path)
    authority_path = surface.READINESS_DIR / surface.STREAM_AUTHORITY_NAME
    authority = surface.load_json(authority_path)
    assert surface.sha256_path(manifest_path) == (
        surface.EXPECTED_SOURCE_MANIFEST_FILE_SHA256
    )
    assert manifest["prospective_manifest_payload_sha256"] == (
        surface.EXPECTED_SOURCE_MANIFEST_PAYLOAD_SHA256
    )
    assert manifest["canonical_rows_sha256"] == (
        surface.EXPECTED_CANONICAL_ROWS_SHA256
    )
    assert manifest["root_set_sha256"] == (
        surface.EXPECTED_ROOT_SET_SHA256
    )
    assert manifest["root_commitment"]["marker_payload_sha256"] == (
        surface.EXPECTED_ROOT_COMMITMENT_PAYLOAD_SHA256
    )
    assert surface.prospective_training_manifest() == manifest
    assert surface.sha256_path(authority_path) == (
        surface.EXPECTED_STREAM_AUTHORITY_FILE_SHA256
    )
    assert authority["stream_authority_payload_sha256"] == (
        surface.EXPECTED_STREAM_AUTHORITY_PAYLOAD_SHA256
    )
    assert authority["passes"]
    assert all(not values for values in authority["collisions"].values())
    for field, (start, end) in surface.STREAM_RANGES.items():
        values = [int(row[field]) for row in manifest["rows"]]
        assert values[0] == start
        assert values[-1] == end
        assert len(values) == 16_384


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
        {"J1D_FIXTURE_RUNTIME_FAILURE": "1"},
        {"J1D_FIXTURE_FIRST_GUARD_FAILURE": "1"},
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


def test_clean_subprocess_real_operational_audit_marker_roundtrip(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "real_operational_marker.json"
    code = (
        "import json;"
        "from pathlib import Path;"
        "from threes_rl import j1d_metric_authentication_surface as s;"
        f"r=s.real_operational_audit_marker_roundtrip(Path({str(marker)!r}));"
        "print(json.dumps(r,sort_keys=True))"
    )
    completed = subprocess.run(
        ["nice", "-n", "10", sys.executable, "-c", code],
        cwd=surface.REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["passes"]
    assert report["checks"]["pre_write_top_three_is_tuple"]
    assert report["checks"]["post_write_top_three_is_list"]
    assert report["checks"]["exact_written_bytes_reloaded"]
    payload = surface.load_json(marker)
    assert surface.verify_payload_hash(
        payload,
        "activation_marker_payload_sha256",
    )
    assert payload["operational_audit"]["services"]["dashboard"][
        "top_three"
    ] == [263670, 261369, 258561]


def test_fixture_full_chain_routes_bounded_training_only(
    tmp_path: Path,
) -> None:
    readiness, execution = _fixture_chain(tmp_path)
    completed = _cli("execute", readiness, execution)
    assert completed.returncode == 0, completed.stderr
    payload = _json_stdout(completed)
    assert payload["terminal_decision"] == (
        "READY_J1D_MINIATURE_TRAINING_FIXTURE"
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
    assert (
        result["bounded_engine"]
        == "execute_training_engine_bounded_j1d"
    )
    assert result["scientific_authority"] is False
    assert result["promote"] is False
    assert paths["retention"].is_file()
    assert not (execution / "development").exists()
    assert not (execution / "confirmation").exists()
    state = _terminal_state(execution)
    assert len(state["round_aggregates"]) == 1
    assert surface.validate_canonical_round_metric_row(
        state["round_aggregates"][0]
    )["passes"]


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
        env_add={"J1D_FIXTURE_PRE_ENGINE_INTERRUPT": boundary},
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
                "J1D_FIXTURE_INTERRUPT_AFTER_BOUNDARY": boundary
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


@pytest.mark.parametrize(
    "boundary",
    [
        "metric_authentication_precommit",
        "metric_authentication_postcommit",
    ],
)
def test_metric_authentication_commit_crash_resume_is_bit_exact(
    tmp_path: Path,
    boundary: str,
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
    interrupted = _cli(
        "execute",
        resumed_readiness,
        resumed_execution,
        env_add={
            "J1D_FIXTURE_INTERRUPT_AFTER_BOUNDARY": boundary,
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
    with pytest.raises(surface.J1dSurfaceIntegrityError):
        surface.load_open_training_contract(
            execution_root=execution,
            readiness_dir=readiness,
            require_manifest=True,
        )


def test_authoritative_fresh_manifest_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "j1d_readiness"
    copied.mkdir()
    for name in (
        surface.SOURCE_MANIFEST_NAME,
        surface.STREAM_AUTHORITY_NAME,
        surface.ROOT_CAUSE_NAME,
    ):
        shutil.copy2(surface.READINESS_DIR / name, copied / name)
    monkeypatch.setattr(surface, "READINESS_DIR", copied)
    source = copied / surface.SOURCE_MANIFEST_NAME
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(surface.J1dSurfaceIntegrityError):
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
    with pytest.raises(surface.J1dSurfaceIntegrityError):
        surface.audit_authoritative_inputs(
            require_future_execution_absent=True,
        )


def test_spent_j1b_inventory_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "spent_j1b"
    for relative in surface.EXPECTED_SPENT_J1B_FILES:
        source = surface.SPENT_J1B_EXECUTION_ROOT / relative
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    monkeypatch.setattr(surface, "SPENT_J1B_EXECUTION_ROOT", copied)
    target = copied / "training/execution_opened.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(surface.J1dSurfaceIntegrityError):
        surface.audit_authoritative_inputs(
            require_future_execution_absent=True,
        )


def test_external_j1b_terminal_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "external_j1b"
    copied.mkdir()
    for name in surface.EXPECTED_J1B_EXTERNAL_FILES:
        shutil.copy2(
            surface.J1B_EXTERNAL_TERMINAL_DIR / name,
            copied / name,
        )
    monkeypatch.setattr(surface, "J1B_EXTERNAL_TERMINAL_DIR", copied)
    target = copied / "J1B_OPEN_FAILURE_TERMINAL.json"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(surface.J1dSurfaceIntegrityError):
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
    with pytest.raises(surface.J1dSurfaceIntegrityError):
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
    with pytest.raises(surface.J1dSurfaceIntegrityError):
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
    with pytest.raises(surface.J1dSurfaceIntegrityError):
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


def test_scientific_readiness_package_seals_in_isolated_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    readiness.mkdir()
    for name in (
        surface.SOURCE_MANIFEST_NAME,
        surface.STREAM_AUTHORITY_NAME,
        surface.ROOT_CAUSE_NAME,
    ):
        shutil.copy2(surface.READINESS_DIR / name, readiness / name)
    future = tmp_path / "future_execution"
    monkeypatch.setattr(surface, "READINESS_DIR", readiness)
    monkeypatch.setattr(surface, "FUTURE_EXECUTION_ROOT", future)
    evidence = surface.write_test_evidence(
        readiness_dir=readiness,
        commands=_fixture_evidence_commands(),
        documented_deselections=["fixture stale state"],
    )
    assert evidence["passes"]
    sealed = surface.seal_readiness_package(
        readiness_dir=readiness,
        operational_audit={
            "version": "fixture_operational_audit_v1",
            "real_marker_roundtrip": {"passes": True},
            "passes": True,
        },
    )
    assert sealed["passes"]
    loaded = surface.load_ready_surface(readiness)
    assert loaded["mode"] == "scientific"
    assert loaded["lock"]["decision"] == surface.READY_DECISION
    assert not future.exists()
    assert sorted(path.name for path in readiness.iterdir()) == sorted(
        (
            surface.SOURCE_MANIFEST_NAME,
            surface.STREAM_AUTHORITY_NAME,
            surface.ROOT_CAUSE_NAME,
            surface.TEST_EVIDENCE_NAME,
            surface.SCHEMA_NAME,
            surface.PROJECTION_NAME,
            surface.INPUT_BINDINGS_NAME,
            surface.READINESS_LOCK_NAME,
            surface.READINESS_RESULT_NAME,
        )
    )


def test_schema_has_no_downstream_surface() -> None:
    schema = surface.surface_schema()
    assert surface.verify_payload_hash(schema, "schema_payload_sha256")
    assert schema["public_commands"] == list(surface.PUBLIC_COMMANDS)
    assert schema["development_surface_present"] is False
    assert schema["confirmation_surface_present"] is False
    assert schema["promotion_surface_present"] is False


def test_prospective_zero_work_counters_are_j1d_labelled() -> None:
    counters = surface.zero_work_counters()
    assert counters
    assert all(value == 0 for value in counters.values())
    assert not any(key.startswith("j1b_") for key in counters)
    assert {
        "j1d_training_phase_locks",
        "j1d_training_markers",
        "j1d_materialized_manifests",
        "j1d_owners",
        "j1d_streams_reserved",
        "j1d_streams_consumed",
        "j1d_genesis_commits",
    }.issubset(counters)


def test_scientific_fixture_controls_are_rejected_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(surface.RUNNER_PATH).read_text(encoding="utf-8")
    assert "execute_training_engine_bounded_j1d(" in source
    assert "parent.execute_training_engine(" not in source
    assert "execute_paired_evaluation_engine" not in source
    assert "--phase" not in source
    monkeypatch.setenv("J1D_FIXTURE_RUNTIME_FAILURE", "1")
    assert os.environ["J1D_FIXTURE_RUNTIME_FAILURE"] == "1"


def test_test_file_has_no_scientific_namespace_creation() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert str(surface.FUTURE_EXECUTION_ROOT) not in source
    assert "starter_tile" in source


def test_runner_can_be_loaded_without_invoking_cli() -> None:
    namespace = runpy.run_path(
        surface.RUNNER_PATH,
        run_name="j1d_surface_import_fixture",
    )
    assert namespace["PUBLIC_COMMANDS"] == surface.PUBLIC_COMMANDS
    assert "main" in namespace

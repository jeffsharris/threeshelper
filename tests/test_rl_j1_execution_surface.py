from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

from threes_rl import j1_execution_surface as surface
from threes_rl import j1_joint_policy_value as parent
from threes_rl import o2_online_option_preflight as accepted_power


def _module_namespace_identity() -> dict[str, tuple[int, str]]:
    result = {}
    for name, value in vars(parent).items():
        if name.startswith("__"):
            continue
        result[name] = (id(value), type(value).__qualname__)
    return result


def _runtime_payload() -> dict:
    torch_state = torch.get_rng_state()
    torch.manual_seed(123)
    try:
        model = parent.J1ActorCritic()
    finally:
        torch.set_rng_state(torch_state)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parent.FROZEN_CONFIG.learning_rate,
        eps=parent.FROZEN_CONFIG.adam_eps,
    )
    return {
        "version": "fixture",
        "runtime_payload_complete": True,
        "phase": "training",
        "marker_file_sha256": "a" * 64,
        "marker_payload_sha256": "b" * 64,
        "manifest_file_sha256": "c" * 64,
        "manifest_payload_sha256": "d" * 64,
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "round_number": 1,
        "collection_boundary": "pre_action",
        "next_manifest_row": 0,
        "active_roots": [],
        "completed_roots": [],
        "transition_buffer_path": None,
        "transition_buffer_sha256": None,
        "epoch_cursor": 0,
        "minibatch_cursor": 0,
        "optimizer_step_ids": [],
        "round_aggregates": [],
        "python_rng_state": None,
        "numpy_rng_state": None,
        "torch_rng_state": torch.get_rng_state().clone(),
        "resource_clock": {"active_seconds": 0.0},
        "output_bytes": 0,
    }


def _marker(phase: str, *, opened_at: str, host: str, command: str) -> dict:
    manifest = surface.materialize_root_manifest(phase=phase)
    lock = surface.payload_with_hash(
        {
            "version": "fixture",
            "phase": phase,
            "decision": f"READY_J1_{phase.upper()}_EXECUTION",
        },
        "phase_lock_payload_sha256",
    )
    return surface.build_phase_marker_payload(
        phase=phase,
        phase_lock=lock,
        phase_lock_file_sha256="e" * 64,
        manifest=manifest,
        command=command,
        opened_at=opened_at,
        hostname=host,
    )


def _commit_contract() -> dict[str, str]:
    return {
        "phase": "training",
        "marker_file_sha256": "1" * 64,
        "phase_lock_file_sha256": "2" * 64,
        "command": "fixture execute",
        "execution_mode": "miniature_fixture",
    }


def _rolling_contract(
    *,
    command: str = "fixture rolling execute",
) -> dict:
    return surface.rolling_resume_contract(
        phase="training",
        marker_file_sha256="a" * 64,
        marker_payload_sha256="b" * 64,
        phase_lock_file_sha256="c" * 64,
        manifest_file_sha256="d" * 64,
        manifest_payload_sha256="e" * 64,
        command=command,
        execution_mode="miniature_fixture",
    )


def _small_manifest_rows(phase: str, count: int) -> list[dict]:
    commitment = surface.phase_root_commitment(phase)
    rows = []
    for row in list(surface.iter_prospective_rows(phase))[:count]:
        root_id = surface.root_id_for_marker_commitment(commitment, row)
        rows.append(
            {
                **row,
                "root_id": root_id,
                "ancestry_id": root_id,
            }
        )
    return rows


def _write_fixture_training_terminal(execution_root: Path) -> dict:
    return surface.write_immutable_json(
        execution_root / "training" / surface.PHASE_RESULT_NAME,
        {
            "version": "fixture-training-terminal-v1",
            "phase": "training",
            "decision": "READY_J1_TRAINING_SANITY",
            "execution_mode": "miniature_fixture",
            "scientific_authority": False,
            "bounded_engine": "execute_training_engine_bounded",
        },
        field="terminal_result_payload_sha256",
    )


def test_concurrent_phase_open_creates_exactly_one_immutable_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root = tmp_path / "execution"
    readiness_dir = tmp_path / "readiness"
    paths = surface.phase_artifact_paths(
        execution_root=execution_root,
        phase="training",
    )
    lock = surface.payload_with_hash(
        {
            "version": "concurrent-open-fixture",
            "phase": "training",
            "decision": "READY_J1_TRAINING_EXECUTION",
            "execution_mode": "miniature_fixture",
            "bounded_engine": "execute_training_engine_bounded",
        },
        "phase_lock_payload_sha256",
    )
    loaded = {
        "paths": paths,
        "lock": lock,
        "lock_result": surface.write_immutable_json(
            paths["lock_result"],
            {
                "version": "concurrent-open-lock-result-fixture",
                "phase": "training",
                "decision": "READY_J1_TRAINING_EXECUTION",
            },
            field="phase_lock_result_payload_sha256",
        ),
        "lock_identity": {"file_sha256": "a" * 64},
        "commands": {
            "open": "fixture open",
            "materialize": "fixture materialize",
            "execute": "fixture execute",
        },
    }
    monkeypatch.setattr(
        surface,
        "_load_phase_lock_artifacts",
        lambda **_kwargs: loaded,
    )
    barrier = threading.Barrier(2)

    def operational_audit(*, output_dir: Path) -> dict:
        assert output_dir == paths["phase_dir"]
        barrier.wait(timeout=5.0)
        return {"passes": True, "fixture": "concurrent-open"}

    candidates = (
        ("2026-07-27T20:00:00Z", "fixture-host-a"),
        ("2026-07-27T20:00:01Z", "fixture-host-b"),
    )

    def open_candidate(candidate: tuple[str, str]) -> tuple[str, object]:
        opened_at, hostname = candidate
        try:
            result = surface.open_phase_from_artifacts(
                phase="training",
                execution_root=execution_root,
                readiness_dir=readiness_dir,
                execution_mode="miniature_fixture",
                operational_audit_fn=operational_audit,
                opened_at=opened_at,
                hostname=hostname,
            )
        except (FileExistsError, surface.J1ExecutionIntegrityError) as error:
            return ("collision", error)
        return ("winner", result)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(open_candidate, candidates))

    assert [status for status, _value in outcomes].count("winner") == 1
    assert [status for status, _value in outcomes].count("collision") == 1
    winning_bytes = paths["marker"].read_bytes()
    winning_sha256 = hashlib.sha256(winning_bytes).hexdigest()
    marker = json.loads(winning_bytes)
    assert (
        marker["activation_opened_at"],
        marker["activation_hostname"],
    ) in candidates
    assert surface.verify_payload_hash(
        marker,
        "activation_marker_payload_sha256",
    )

    changed = dict(marker)
    changed.pop("activation_marker_payload_sha256")
    changed["activation_hostname"] = "late-overwrite-attempt"
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.write_immutable_json(
            paths["marker"],
            changed,
            field="activation_marker_payload_sha256",
        )
    assert paths["marker"].read_bytes() == winning_bytes
    assert surface.sha256_path(paths["marker"]) == winning_sha256


@pytest.fixture(scope="module")
def miniature_training_records() -> list[dict]:
    model, _optimizer = parent.initialize_model_optimizer()
    session = surface.TrainingCollectionSession(
        rows=_small_manifest_rows("training", 3),
        model=model,
        env_count=2,
        max_moves=surface.MAX_MOVES,
    )
    return session.finish()


def _owner_ledger(
    phase_dir: Path,
    *,
    pid: int,
    start_identity: str,
    contract: dict[str, str],
) -> dict:
    owner = surface._new_owner_record(
        phase=contract["phase"],
        marker_file_sha256=contract["marker_file_sha256"],
        phase_lock_file_sha256=contract["phase_lock_file_sha256"],
        command=contract["command"],
        predecessor_commit_head_sha256=None,
        execution_mode=contract["execution_mode"],
        pid=pid,
        start_identity=start_identity,
    )
    ledger = surface.payload_with_hash(
        {
            "version": f"{surface.VERSION}_ownership_ledger_v1",
            "owners": [owner],
            "recoveries": [],
            "head_owner_sha256": owner["owner_record_sha256"],
        },
        "ownership_payload_sha256",
    )
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / surface.PHASE_OWNER_NAME).write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ledger


def test_parent_identities_and_zero_work_contract_are_exact() -> None:
    audit = surface.accepted_identity_audit()
    assert audit["passes"]
    assert surface.ZERO_WORK["execution_markers"] == 0
    assert surface.ZERO_WORK["j1_streams_consumed"] == 0
    assert surface.ZERO_WORK["scientific_optimizer_steps"] == 0


def test_readiness_and_runtime_validators_do_not_mutate_parent_module() -> None:
    before = _module_namespace_identity()
    surface.execution_schema()
    surface.accepted_identity_audit()
    surface.prospective_manifest()
    surface.validate_training_runtime_payload(_runtime_payload())
    after = _module_namespace_identity()
    assert before == after


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_model",
        "wrong_model_shape",
        "nonfinite_model",
        "missing_optimizer",
        "malformed_optimizer",
        "nonfinite_optimizer",
    ),
)
def test_runtime_tensor_validation_fails_closed(mutation: str) -> None:
    payload = _runtime_payload()
    if mutation == "missing_model":
        payload["model_state"].pop("policy.bias")
    elif mutation == "wrong_model_shape":
        payload["model_state"]["policy.bias"] = torch.zeros(5)
    elif mutation == "nonfinite_model":
        payload["model_state"]["policy.bias"][0] = math.nan
    elif mutation == "missing_optimizer":
        payload["optimizer_state"].pop("state")
    elif mutation == "malformed_optimizer":
        payload["optimizer_state"]["param_groups"] = {}
    elif mutation == "nonfinite_optimizer":
        payload["optimizer_state"]["state"] = {
            0: {"exp_avg": torch.tensor([math.inf])}
        }
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.validate_training_runtime_payload(payload)


def test_prospective_manifest_counts_prefixes_and_crn() -> None:
    payload = surface.prospective_manifest()
    assert payload["passes"]
    assert payload["counts"] == {
        "training_rows": 16_384,
        "development_pairs": 896,
        "confirmation_pairs": 4_480,
        "total_game_arms": 27_136,
    }
    assert payload["checks"]["all_amended_ranges_exact_parent_prefixes"]
    assert payload["checks"]["paired_crn_exact"]
    assert payload["checks"]["precommitted_root_sets_disjoint"]
    assert payload["streams_reserved"] == 0
    assert payload["streams_consumed"] == 0


def test_confirmation_roots_ignore_post_development_marker_variation() -> None:
    first = _marker(
        "confirmation",
        opened_at="2026-01-01T00:00:00Z",
        host="host-a",
        command="first command",
    )
    second = _marker(
        "confirmation",
        opened_at="2027-02-02T03:04:05Z",
        host="host-b",
        command="different command",
    )
    audit = surface.phase_marker_root_identity_audit(
        phase="confirmation",
        first_marker=first,
        second_marker=second,
        confirmation_access_evidence=surface.confirmation_access_audit(
            content_reads=0,
            streams_reserved=0,
            streams_consumed=0,
            evidence={"fixture": "content-blind"},
        ),
    )
    assert (
        first["activation_marker_payload_sha256"]
        != second["activation_marker_payload_sha256"]
    )
    assert audit["passes"]
    assert audit["checks"]["confirmation_content_reads_zero"]
    assert audit["checks"]["confirmation_streams_reserved_zero"]


def test_root_identity_exactly_matches_accepted_parent_formula() -> None:
    commitment = surface.phase_root_commitment("training")
    row = next(surface.iter_prospective_rows("training"))
    observed = surface.root_id_for_marker_commitment(commitment, row)
    expected = parent.canonical_json_hash(
        {
            "marker_payload_sha256": commitment["marker_payload_sha256"],
            "partition": "train",
            "row": 0,
            "logical_stream_id": row["logical_stream_id"],
            "deck_stream_id": row["deck_stream_id"],
            "slot_stream_id": row["slot_stream_id"],
        }
    )
    assert observed == expected


def test_root_commitment_changes_ids_but_activation_evidence_does_not() -> None:
    row = next(surface.iter_prospective_rows("confirmation"))
    commitment = surface.phase_root_commitment("confirmation")
    changed = dict(commitment)
    changed["phase_nonce"] = "f" * 64
    changed = surface.payload_with_hash(changed, "marker_payload_sha256")
    assert (
        surface.root_id_for_marker_commitment(commitment, row)
        != surface.root_id_for_marker_commitment(changed, row)
    )

    first = _marker(
        "confirmation",
        opened_at="2026-01-01T00:00:00Z",
        host="host-a",
        command="first command",
    )
    second = _marker(
        "confirmation",
        opened_at="2027-02-02T03:04:05Z",
        host="host-b",
        command="different command",
    )
    assert first["root_commitment"] == second["root_commitment"]
    assert (
        surface.materialize_root_manifest(
            phase="confirmation",
            marker_payload=first,
        )["rows"]
        == surface.materialize_root_manifest(
            phase="confirmation",
            marker_payload=second,
        )["rows"]
    )


def test_confirmation_identity_audit_requires_zero_access_evidence() -> None:
    first = _marker(
        "confirmation",
        opened_at="2026-01-01T00:00:00Z",
        host="host-a",
        command="first command",
    )
    second = _marker(
        "confirmation",
        opened_at="2026-01-02T00:00:00Z",
        host="host-b",
        command="second command",
    )
    assert not surface.phase_marker_root_identity_audit(
        phase="confirmation",
        first_marker=first,
        second_marker=second,
    )["passes"]
    nonzero = surface.confirmation_access_audit(
        content_reads=1,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"fixture": "nonzero"},
    )
    assert not surface.phase_marker_root_identity_audit(
        phase="confirmation",
        first_marker=first,
        second_marker=second,
        confirmation_access_evidence=nonzero,
    )["passes"]


def test_joint_evaluation_manifests_seal_before_development(
    tmp_path: Path,
) -> None:
    training = surface.materialize_root_manifest(phase="training")
    result = _write_fixture_training_terminal(tmp_path)
    access_path = tmp_path / "confirmation_access_before_joint_seal.json"
    access = surface.write_confirmation_access_audit(
        path=access_path,
        content_reads=0,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "fixture counters"},
    )
    sealed = surface.seal_joint_evaluation_manifests(
        execution_root=tmp_path,
        training_manifest=training,
        training_result=result,
        confirmation_access_audit_path=access_path,
    )
    assert sealed["passes"]
    assert sealed["seal"]["confirmation_access_audit"] == {
        "path": str(access_path.resolve()),
        "file_sha256": surface.sha256_path(access_path),
        "payload_sha256": access["confirmation_access_audit_sha256"],
        "confirmation_content_reads": 0,
        "confirmation_streams_reserved": 0,
        "confirmation_streams_consumed": 0,
    }
    assert sealed["seal"]["cross_phase_audit"]["phase_counts"] == {
        "training": 16_384,
        "development": 896,
        "confirmation": 4_480,
    }
    assert set(
        sealed["seal"]["incumbent_policy_binding"][
            "implementation_sources"
        ]
    ) == {
        "threes_rl/eval.py",
        "threes_rl/expectimax.py",
        "threes_rl/ntuple.py",
        "threes_rl/action_prior.py",
        "threes_rl/sim.py",
        "threes_rl/train_td.py",
        "threes_rl/obs.py",
        "threes_rl/env.py",
    }
    loaded = surface.load_precommitted_evaluation_manifest(
        execution_root=tmp_path,
        phase="confirmation",
    )
    assert loaded == sealed["confirmation"]


def test_joint_manifests_require_ready_training(tmp_path: Path) -> None:
    access_path = tmp_path / "confirmation_access_before_joint_seal.json"
    surface.write_confirmation_access_audit(
        path=access_path,
        content_reads=0,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "fixture counters"},
    )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.seal_joint_evaluation_manifests(
            execution_root=tmp_path,
            training_manifest=surface.materialize_root_manifest(
                phase="training"
            ),
            training_result={"decision": "HOLD_J1_LEARNING_SANITY"},
            confirmation_access_audit_path=access_path,
        )
    assert not (tmp_path / surface.PRECOMMITTED_MANIFEST_DIR).exists()


def test_joint_manifest_seal_rejects_nonzero_or_tampered_access(
    tmp_path: Path,
) -> None:
    training = surface.materialize_root_manifest(phase="training")
    result = _write_fixture_training_terminal(tmp_path)
    nonzero_path = tmp_path / "nonzero.json"
    surface.write_confirmation_access_audit(
        path=nonzero_path,
        content_reads=1,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "fixture counters"},
    )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.seal_joint_evaluation_manifests(
            execution_root=tmp_path,
            training_manifest=training,
            training_result=result,
            confirmation_access_audit_path=nonzero_path,
        )

    tampered_path = tmp_path / "tampered.json"
    audit = surface.confirmation_access_audit(
        content_reads=0,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "fixture counters"},
    )
    audit["evidence"] = {"source": "tampered"}
    tampered_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.seal_joint_evaluation_manifests(
            execution_root=tmp_path,
            training_manifest=training,
            training_result=result,
            confirmation_access_audit_path=tampered_path,
        )


def test_incumbent_binding_rejects_source_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = surface.incumbent_policy_binding()
    original = surface.sha256_path

    def changed(path, root=surface.REPO_ROOT):
        if Path(path).name == "expectimax.py":
            return "0" * 64
        return original(path, root)

    monkeypatch.setattr(surface, "sha256_path", changed)
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.load_bound_incumbent_policy(binding)


def _joint_seal_fixture() -> dict:
    confirmation = surface.root_manifest_identity(
        surface.materialize_root_manifest(phase="confirmation")
    )
    development = surface.root_manifest_identity(
        surface.materialize_root_manifest(phase="development")
    )
    return surface.payload_with_hash(
        {
            "version": "fixture-joint-seal",
            "development_manifest": development,
            "confirmation_manifest": confirmation,
        },
        "joint_manifest_seal_payload_sha256",
    )


def test_confirmation_lock_binds_exact_predevelopment_joint_seal() -> None:
    joint = _joint_seal_fixture()
    manifest_identity = surface.root_manifest_identity(
        surface.materialize_root_manifest(phase="confirmation")
    )
    predecessor = {
        "decision": "READY_J1_DEVELOPMENT_FULL_POLICY",
        "terminal_result_payload_sha256": "a" * 64,
        "joint_evaluation_manifest_seal_payload_sha256": joint[
            "joint_manifest_seal_payload_sha256"
        ],
    }
    lock = surface.build_phase_lock_payload(
        phase="confirmation",
        readiness_lock_identity={"file_sha256": "b" * 64},
        readiness_result_identity={
            "decision": "READY_J1_EXECUTION_SURFACE",
            "file_sha256": "c" * 64,
        },
        manifest_identity=manifest_identity,
        predecessor_result=predecessor,
        command="confirmation command",
        joint_manifest_seal=joint,
    )
    assert surface.verify_payload_hash(lock, "phase_lock_payload_sha256")
    assert (
        lock["joint_evaluation_manifest_seal_payload_sha256"]
        == joint["joint_manifest_seal_payload_sha256"]
    )


@pytest.mark.parametrize("fault", ("bad_hash", "wrong_manifest", "new_seal"))
def test_confirmation_lock_rejects_changed_joint_seal(fault: str) -> None:
    joint = _joint_seal_fixture()
    manifest_identity = surface.root_manifest_identity(
        surface.materialize_root_manifest(phase="confirmation")
    )
    predecessor = {
        "decision": "READY_J1_DEVELOPMENT_FULL_POLICY",
        "terminal_result_payload_sha256": "a" * 64,
        "joint_evaluation_manifest_seal_payload_sha256": joint[
            "joint_manifest_seal_payload_sha256"
        ],
    }
    changed = copy.deepcopy(joint)
    if fault == "bad_hash":
        changed["confirmation_manifest"]["row_count"] += 1
    elif fault == "wrong_manifest":
        changed["confirmation_manifest"]["canonical_rows_sha256"] = "f" * 64
        changed = surface.payload_with_hash(
            {
                key: value
                for key, value in changed.items()
                if key != "joint_manifest_seal_payload_sha256"
            },
            "joint_manifest_seal_payload_sha256",
        )
    else:
        changed["version"] = "fixture-joint-seal-new"
        changed = surface.payload_with_hash(
            {
                key: value
                for key, value in changed.items()
                if key != "joint_manifest_seal_payload_sha256"
            },
            "joint_manifest_seal_payload_sha256",
        )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.build_phase_lock_payload(
            phase="confirmation",
            readiness_lock_identity={"file_sha256": "b" * 64},
            readiness_result_identity={
                "decision": "READY_J1_EXECUTION_SURFACE",
                "file_sha256": "c" * 64,
            },
            manifest_identity=manifest_identity,
            predecessor_result=predecessor,
            command="confirmation command",
            joint_manifest_seal=changed,
        )


def test_cross_phase_duplicate_root_fails_closed() -> None:
    train = {
        "phase": "training",
        "rows": [{"root_id": "same", "ancestry_id": "same"}],
    }
    development = {
        "phase": "development",
        "rows": [{"root_id": "same", "ancestry_id": "same"}],
    }
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.validate_cross_phase_manifests([train, development])


@pytest.mark.parametrize(
    ("crash_stage", "committed_after_crash"),
    (
        ("after_state", False),
        ("after_record", False),
        ("after_head_record", False),
        ("after_pointer", True),
    ),
)
def test_transaction_commit_point_is_exact(
    tmp_path: Path,
    crash_stage: str,
    committed_after_crash: bool,
) -> None:
    contract = _commit_contract()
    surface.initialize_commit_store(
        phase_dir=tmp_path,
        **contract,
        initial_state={"fixture": 0},
    )
    with pytest.raises(RuntimeError):
        surface.commit_unit(
            phase_dir=tmp_path,
            **contract,
            unit_id="optimizer:round1:epoch0:minibatch0",
            post_state={"fixture": 1},
            journal_payload={"kind": "optimizer_step"},
            crash_stage=crash_stage,
        )
    boundary = surface.verify_commit_boundary(
        phase_dir=tmp_path,
        **contract,
    )
    assert (
        "optimizer:round1:epoch0:minibatch0"
        in boundary["state"]["committed_unit_ids"]
    ) is committed_after_crash
    if committed_after_crash:
        with pytest.raises(surface.J1ExecutionIntegrityError):
            surface.commit_unit(
                phase_dir=tmp_path,
                **contract,
                unit_id="optimizer:round1:epoch0:minibatch0",
                post_state={"fixture": 1},
                journal_payload={"kind": "optimizer_step"},
            )
    else:
        resumed = surface.commit_unit(
            phase_dir=tmp_path,
            **contract,
            unit_id="optimizer:round1:epoch0:minibatch0",
            post_state={"fixture": 1},
            journal_payload={"kind": "optimizer_step"},
        )
        assert resumed["sequence"] == 1
        assert resumed["state"]["committed_unit_ids"].count(
            "optimizer:round1:epoch0:minibatch0"
        ) == 1


def test_live_owner_reclaim_is_rejected(tmp_path: Path) -> None:
    contract = _commit_contract()
    surface.initialize_commit_store(
        phase_dir=tmp_path,
        **contract,
        initial_state={"fixture": 0},
    )
    _owner_ledger(
        tmp_path,
        pid=os.getpid(),
        start_identity="fixture-live",
        contract=contract,
    )
    with pytest.raises(surface.J1ExecutionOperationalHold):
        surface.reclaim_dead_writer_owner(
            phase_dir=tmp_path,
            **contract,
            pid_alive=lambda _pid: True,
            process_identity=lambda _pid: "fixture-live",
            contention_audit={"passes": True},
            new_pid=123,
            new_start_identity="new",
        )


def test_dead_same_contract_owner_reclaims_verified_boundary(
    tmp_path: Path,
) -> None:
    contract = _commit_contract()
    surface.initialize_commit_store(
        phase_dir=tmp_path,
        **contract,
        initial_state={"fixture": 0},
    )
    _owner_ledger(
        tmp_path,
        pid=999_999_999,
        start_identity="dead-owner-start",
        contract=contract,
    )
    recovered = surface.reclaim_dead_writer_owner(
        phase_dir=tmp_path,
        **contract,
        pid_alive=lambda _pid: False,
        process_identity=lambda _pid: None,
        contention_audit={"passes": True, "unrelated_candidate_pids": []},
        new_pid=123_456,
        new_start_identity="new-owner-start",
    )
    assert recovered["passes"]
    evidence = recovered["recovery"]["committed_boundary"]
    assert evidence["sequence"] == 0
    assert len(evidence["commit_head_file_sha256"]) == 64
    assert len(recovered["ledger"]["owners"]) == 2
    assert len(recovered["ledger"]["recoveries"]) == 1


@pytest.mark.parametrize("wrong_field", ("marker_file_sha256", "command"))
def test_dead_owner_wrong_contract_reclaim_is_rejected(
    tmp_path: Path,
    wrong_field: str,
) -> None:
    contract = _commit_contract()
    surface.initialize_commit_store(
        phase_dir=tmp_path,
        **contract,
        initial_state={"fixture": 0},
    )
    _owner_ledger(
        tmp_path,
        pid=999_999_998,
        start_identity="dead-owner-start",
        contract=contract,
    )
    changed = dict(contract)
    changed[wrong_field] = "wrong"
    with pytest.raises(surface.J1ExecutionOperationalHold):
        surface.reclaim_dead_writer_owner(
            phase_dir=tmp_path,
            **changed,
            pid_alive=lambda _pid: False,
            process_identity=lambda _pid: None,
            contention_audit={"passes": True},
            new_pid=123_457,
            new_start_identity="new-owner-start",
        )


@pytest.mark.parametrize("fault", ("missing", "tampered", "stale"))
def test_dead_owner_reclaim_rejects_bad_commit_head(
    tmp_path: Path,
    fault: str,
) -> None:
    contract = _commit_contract()
    surface.initialize_commit_store(
        phase_dir=tmp_path,
        **contract,
        initial_state={"fixture": 0},
    )
    _owner_ledger(
        tmp_path,
        pid=999_999_997,
        start_identity="dead-owner-start",
        contract=contract,
    )
    pointer = tmp_path / surface.COMMIT_HEAD_NAME
    if fault == "missing":
        pointer.unlink()
    elif fault == "tampered":
        pointer.write_bytes(pointer.read_bytes() + b"x")
    else:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        payload["sequence"] = 1
        payload["commit_head_pointer_sha256"] = surface.canonical_json_hash(
            {
                key: value
                for key, value in payload.items()
                if key != "commit_head_pointer_sha256"
            }
        )
        pointer.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.reclaim_dead_writer_owner(
            phase_dir=tmp_path,
            **contract,
            pid_alive=lambda _pid: False,
            process_identity=lambda _pid: None,
            contention_audit={"passes": True},
            new_pid=123_458,
            new_start_identity="new-owner-start",
        )


def test_binary_state_corruption_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "state.bin"
    surface.write_atomic_binary(path, {"tensor": torch.ones(2)})
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.load_atomic_binary(path)


@pytest.mark.parametrize("artifact_kind", ("state", "journal", "record"))
def test_full_commit_chain_detects_tampered_older_artifact(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    contract = _commit_contract()
    surface.initialize_commit_store(
        phase_dir=tmp_path,
        **contract,
        initial_state={"fixture": 0},
    )
    surface.commit_unit(
        phase_dir=tmp_path,
        **contract,
        unit_id="collection:0001",
        post_state={"fixture": 1},
        journal_payload={"kind": "collection"},
    )
    surface.commit_unit(
        phase_dir=tmp_path,
        **contract,
        unit_id="optimizer:0001",
        post_state={"fixture": 2},
        journal_payload={"kind": "optimizer"},
    )
    directory = {
        "state": surface.COMMIT_STATES_DIR,
        "journal": surface.COMMIT_JOURNALS_DIR,
        "record": surface.COMMIT_RECORDS_DIR,
    }[artifact_kind]
    candidates = sorted((tmp_path / directory).iterdir())
    older = candidates[0]
    older.write_bytes(older.read_bytes() + b"tampered")
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.verify_commit_boundary(
            phase_dir=tmp_path,
            **contract,
        )


@pytest.mark.parametrize("complete_flag", ("missing", False))
def test_scientific_training_commit_requires_complete_runtime_payload(
    tmp_path: Path,
    complete_flag: str | bool,
) -> None:
    contract = {
        **_commit_contract(),
        "execution_mode": "scientific",
    }
    surface.initialize_commit_store(
        phase_dir=tmp_path,
        **contract,
        initial_state={"fixture": 0},
    )
    runtime = _runtime_payload()
    if complete_flag == "missing":
        runtime.pop("runtime_payload_complete")
    else:
        runtime["runtime_payload_complete"] = False
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.commit_unit(
            phase_dir=tmp_path,
            **contract,
            unit_id="collection:scientific:0001",
            post_state=runtime,
            journal_payload={"kind": "collection"},
        )
    boundary = surface.verify_commit_boundary(
        phase_dir=tmp_path,
        **contract,
    )
    assert boundary["sequence"] == 0


def test_phase_order_blocks_premature_development_and_confirmation() -> None:
    readiness = {"decision": "READY_J1_EXECUTION_SURFACE"}
    access = surface.confirmation_access_audit(
        content_reads=0,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "fixture"},
    )
    development = surface.phase_order_barrier_audit(
        phase="development",
        readiness_result=readiness,
        training_result={"decision": "HOLD_J1_LEARNING_SANITY"},
        joint_manifest_seal=None,
        confirmation_access_audit=access,
    )
    confirmation = surface.phase_order_barrier_audit(
        phase="confirmation",
        readiness_result=readiness,
        training_result={"decision": "READY_J1_TRAINING_SANITY"},
        development_result={
            "decision": "HOLD_J1_DEVELOPMENT_INCONCLUSIVE"
        },
        joint_manifest_seal=None,
        confirmation_access_audit=access,
    )
    assert not development["passes"]
    assert not confirmation["passes"]


@pytest.mark.parametrize(
    ("content_reads", "reserved", "consumed"),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
)
def test_phase_order_rejects_nonzero_confirmation_access(
    content_reads: int,
    reserved: int,
    consumed: int,
) -> None:
    audit = surface.confirmation_access_audit(
        content_reads=content_reads,
        streams_reserved=reserved,
        streams_consumed=consumed,
        evidence={"source": "fixture"},
    )
    barrier = surface.phase_order_barrier_audit(
        phase="training",
        readiness_result={"decision": "READY_J1_EXECUTION_SURFACE"},
        confirmation_access_audit=audit,
    )
    assert not barrier["passes"]


@pytest.mark.parametrize(
    ("candidate_success", "control_success"),
    (
        ([0] * 8, [1] * 8),
        ([1] * 8, [0] * 8),
        ([2, 3, 4, 5, 2, 3, 4, 5], [1, 2, 3, 4, 1, 2, 3, 4]),
    ),
)
def test_common_or_exactly_matches_accepted_corrected_estimator(
    candidate_success: list[int],
    control_success: list[int],
) -> None:
    totals = np.full(8, 8, dtype=np.int64)
    candidate = []
    control = []
    blocks = []
    for block, (candidate_count, control_count) in enumerate(
        zip(candidate_success, control_success)
    ):
        candidate.extend([1] * candidate_count + [0] * (8 - candidate_count))
        control.extend([1] * control_count + [0] * (8 - control_count))
        blocks.extend([block] * 8)
    observed = surface._mantel_haenszel_or(
        np.asarray(candidate),
        np.asarray(control),
        np.asarray(blocks),
    )
    expected = float(
        np.exp(
            accepted_power._mh_log_or(
                np.asarray([candidate_success], dtype=np.float64),
                np.asarray([control_success], dtype=np.float64),
                np.asarray([totals], dtype=np.float64),
            )[0]
        )
    )
    assert observed == expected
    assert math.isfinite(observed)


@pytest.mark.parametrize(
    ("candidate_success", "control_success"),
    (
        ([0] * 8, [1] * 8),
        ([1] * 8, [0] * 8),
        ([2, 3, 4, 5, 2, 3, 4, 5], [1, 2, 3, 4, 1, 2, 3, 4]),
    ),
)
def test_progression_bootstrap_exactly_matches_accepted_within_stratum_method(
    candidate_success: list[int],
    control_success: list[int],
) -> None:
    candidate = []
    control = []
    blocks = []
    candidate_by_root = []
    control_by_root = []
    for block, (candidate_count, control_count) in enumerate(
        zip(candidate_success, control_success)
    ):
        candidate_block = np.asarray(
            [1] * candidate_count + [0] * (8 - candidate_count),
            dtype=np.int8,
        )
        control_block = np.asarray(
            [1] * control_count + [0] * (8 - control_count),
            dtype=np.int8,
        )
        candidate.extend(candidate_block.tolist())
        control.extend(control_block.tolist())
        blocks.extend([block] * 8)
        candidate_by_root.append(candidate_block.reshape(-1, 1))
        control_by_root.append(control_block.reshape(-1, 1))
    seed = 987_654
    repeats = 128
    observed = surface._accepted_progression_bootstrap_bounds(
        candidate=np.asarray(candidate),
        control=np.asarray(control),
        blocks=np.asarray(blocks),
        repeats=repeats,
        seed=seed,
    )
    expected_log = accepted_power._bootstrap_cluster_bounds(
        control_by_root,
        candidate_by_root,
        rng=np.random.default_rng(seed),
        bootstraps=repeats,
    )
    expected = tuple(float(np.exp(value)) for value in expected_log)
    assert observed == expected


def _paired_fixture_rows() -> list[dict]:
    rows = []
    for index in range(8):
        streams = {
            "logical_stream_id": 100 + index,
            "deck_stream_id": 200 + index,
            "slot_stream_id": 300 + index,
            "starter_tile": None,
        }
        rows.append(
            {
                "root_id": f"paired-{index}",
                "block": index,
                "candidate": {
                    **streams,
                    "policy_stream_id": 400 + index,
                    "start_score": 100,
                    "final_score": 90 if index == 0 else 120 + index,
                    "moves": 100,
                    "max_tile": 1536 if index % 2 == 0 else 768,
                    "decision_latencies_seconds": [0.01, 0.02],
                    "illegal_actions": 0,
                    "crashes": 0,
                },
                "control": {
                    **streams,
                    "policy_stream_id": 500 + index,
                    "start_score": 100,
                    "final_score": 100 if index == 0 else 115 + index,
                    "moves": 100,
                    "max_tile": 1536 if index % 3 == 0 else 768,
                    "decision_latencies_seconds": [0.02, 0.03],
                    "illegal_actions": 0,
                    "crashes": 0,
                },
            }
        )
    return rows


def test_score_estimand_clamps_final_minus_start_before_log1p() -> None:
    rows = _paired_fixture_rows()
    report = surface.analyze_paired_full_policy(
        rows,
        phase="development",
        bootstrap_repeats=16,
        fixture_mode=True,
    )
    expected = np.mean(
        [
            math.log1p(max(row["candidate"]["final_score"] - 100, 0))
            - math.log1p(max(row["control"]["final_score"] - 100, 0))
            for row in rows
        ]
    )
    assert report["score_log_difference"]["point"] == pytest.approx(expected)
    assert math.isfinite(report["p1536_common_or"]["point"])


def _training_sanity_report_fixture() -> dict:
    root_ids = [f"root-{index}" for index in range(surface.TRAIN_ROOTS)]
    rounds = []
    for round_number in range(1, surface.ROUNDS + 1):
        round_roots = root_ids[
            (round_number - 1) * surface.ROOTS_PER_ROUND :
            round_number * surface.ROOTS_PER_ROUND
        ]
        root_metrics = [
            {
                "root_id": root_id,
                "ancestry_id": root_id,
                "committed_record_sha256": hashlib.sha256(
                    f"record:{root_id}".encode()
                ).hexdigest(),
                "transition_content_sha256": hashlib.sha256(
                    f"transitions:{root_id}".encode()
                ).hexdigest(),
                "transition_rows": 3,
                "log_score": (
                    1.1 if round_number >= 61 else 1.0
                ),
                "legal_entropy_nats": 0.20,
                "value_mse": 0.20,
                "zero_value_mse": 0.30,
                "auxiliary_brier": [0.10, 0.20, 0.40],
                "auxiliary_prevalence": [0.50, 0.50, 0.50],
            }
            for root_id in round_roots
        ]
        rounds.append(
            {
                "round": round_number,
                "root_ids": round_roots,
                "root_metrics": root_metrics,
                "root_metrics_sha256": parent.stable_hash(root_metrics),
                "committed_records_sha256": hashlib.sha256(
                    f"records:{round_number}".encode()
                ).hexdigest(),
                "transition_buffer_sha256": hashlib.sha256(
                    f"buffer:{round_number}".encode()
                ).hexdigest(),
                "root_log_scores": [
                    metric["log_score"] for metric in root_metrics
                ],
                "legal_entropy_nats": 0.20,
                "value_mse": 0.20,
                "zero_value_mse": 0.30,
                "auxiliary_brier": [0.10, 0.20, 0.40],
                "auxiliary_prevalence_brier": [0.25, 0.25, 0.25],
            }
        )
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
            "parameter_count": parent.EXPECTED_PARAMETER_COUNT,
            "model_schema_sha256": parent.model_schema_sha256(),
            "training_state_file_sha256": "e" * 64,
        },
    }


def test_training_sanity_ready_hold_and_integrity_decisions() -> None:
    report = _training_sanity_report_fixture()
    assert (
        surface.training_sanity_decision(report)["decision"]
        == "READY_J1_TRAINING_SANITY"
    )
    hold = copy.deepcopy(report)
    final = hold["rounds"][-1]
    for metric in final["root_metrics"]:
        metric["legal_entropy_nats"] = 0.10
    final["root_metrics_sha256"] = parent.stable_hash(
        final["root_metrics"]
    )
    final["legal_entropy_nats"] = 0.10
    assert (
        surface.training_sanity_decision(hold)["decision"]
        == "HOLD_J1_LEARNING_SANITY"
    )
    killed = copy.deepcopy(report)
    killed["completed_root_ids"] = killed["completed_root_ids"][:-1]
    assert (
        surface.training_sanity_decision(killed)["decision"]
        == "KILL_J1_INTEGRITY"
    )


@pytest.mark.parametrize("fault", ("substitute", "duplicate", "wrong_round"))
def test_training_sanity_rejects_round_partition_tampering(
    fault: str,
) -> None:
    report = _training_sanity_report_fixture()
    if fault == "substitute":
        report["rounds"][0]["root_ids"][0] = "root-999"
        report["rounds"][0]["root_metrics"][0]["root_id"] = "root-999"
    elif fault == "duplicate":
        duplicate = report["rounds"][0]["root_ids"][0]
        report["rounds"][0]["root_ids"][1] = duplicate
        report["rounds"][0]["root_metrics"][1]["root_id"] = duplicate
    else:
        report["rounds"][0]["round"] = 2
    if fault != "wrong_round":
        report["rounds"][0]["root_metrics_sha256"] = parent.stable_hash(
            report["rounds"][0]["root_metrics"]
        )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.training_sanity_decision(report)


def test_actual_normal_start_collection_and_resume_are_identical() -> None:
    rows = _small_manifest_rows("training", 3)
    model, _optimizer = parent.initialize_model_optimizer()
    uninterrupted = surface.TrainingCollectionSession(
        rows=rows,
        model=model,
        env_count=2,
    ).finish()

    interrupted = surface.TrainingCollectionSession(
        rows=rows,
        model=model,
        env_count=2,
    )
    snapshot = None
    while not interrupted.is_complete():
        interrupted.step_tick()
        candidate = interrupted.snapshot()
        if (
            candidate["completed_records"]
            and candidate["active"]
        ):
            snapshot = candidate
            break
    assert snapshot is not None
    assert snapshot["completed_records"]
    assert snapshot["active"]
    restored = surface.TrainingCollectionSession.from_snapshot(
        surface.deserialize_binary_state(
            surface.serialize_binary_state(snapshot)
        ),
        rows=rows,
        model=model,
    )
    resumed = restored.finish()
    assert parent.stable_hash(resumed) == parent.stable_hash(uninterrupted)
    assert [row["root_id"] for row in resumed] == [
        row["root_id"] for row in rows
    ]
    assert all(row["natural_terminal"] for row in resumed)
    assert all(row["telescoping"]["passes"] for row in resumed)


@pytest.mark.parametrize("target", ("active", "completed"))
def test_training_resume_rejects_wrong_manifest_row_binding(
    target: str,
) -> None:
    rows = _small_manifest_rows("training", 3)
    model, _optimizer = parent.initialize_model_optimizer()
    session = surface.TrainingCollectionSession(
        rows=rows,
        model=model,
        env_count=2,
    )
    snapshot = None
    while not session.is_complete():
        session.step_tick()
        candidate = session.snapshot()
        if candidate["completed_records"] and candidate["active"]:
            snapshot = candidate
            break
    assert snapshot is not None
    changed = copy.deepcopy(snapshot)
    if target == "active":
        changed["active"][0]["row"]["deck_stream_id"] += 1
    else:
        changed["completed_records"][0]["source_manifest_row"][
            "slot_stream_id"
        ] += 1
        changed["completed_records_sha256"] = parent.stable_hash(
            changed["completed_records"]
        )
    body = dict(changed)
    body.pop("session_state_sha256")
    changed["session_state_sha256"] = parent.stable_hash(body)
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.TrainingCollectionSession.from_snapshot(
            changed,
            rows=rows,
            model=model,
        )


def test_training_snapshot_externalizes_completed_roots_once(
    tmp_path: Path,
) -> None:
    rows = _small_manifest_rows("training", 3)
    model, _optimizer = parent.initialize_model_optimizer()
    session = surface.TrainingCollectionSession(
        rows=rows,
        model=model,
        env_count=2,
    )
    snapshot = None
    while not session.is_complete():
        session.step_tick()
        candidate = session.snapshot(
            completed_blob_dir=tmp_path / surface.ROOT_BLOBS_DIR
        )
        if candidate["completed_record_refs"] and candidate["active"]:
            snapshot = candidate
            break
    assert snapshot is not None
    assert snapshot["completed_storage"] == "immutable_root_blobs"
    assert snapshot["completed_records"] == []
    assert len(snapshot["completed_record_refs"]) >= 1
    restored = surface.TrainingCollectionSession.from_snapshot(
        snapshot,
        rows=rows,
        model=model,
        completed_blob_dir=tmp_path / surface.ROOT_BLOBS_DIR,
    )
    assert restored.completed
    assert restored.active


def test_frozen_updater_rejects_mismatched_optimizer(
    miniature_training_records: list[dict],
) -> None:
    batch = surface.training_records_to_ppo_batch(
        miniature_training_records
    )
    model, _model_optimizer = parent.initialize_model_optimizer()
    other_model, other_optimizer = parent.initialize_model_optimizer()
    del other_model
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.FrozenMinibatchUpdater(
            model=model,
            optimizer=other_optimizer,
            batch=batch,
            round_number=1,
            minibatch_size=32,
        )


def test_frozen_updater_changes_actual_model_parameters(
    miniature_training_records: list[dict],
) -> None:
    batch = surface.training_records_to_ppo_batch(
        miniature_training_records
    )
    model, optimizer = parent.initialize_model_optimizer()
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    updater = surface.FrozenMinibatchUpdater(
        model=model,
        optimizer=optimizer,
        batch=batch,
        round_number=1,
        minibatch_size=32,
    )
    updater.step_once()
    assert any(
        not torch.equal(before[key], value)
        for key, value in model.state_dict().items()
    )


def test_frozen_updater_resume_is_bit_identical(
    miniature_training_records: list[dict],
) -> None:
    batch = surface.training_records_to_ppo_batch(
        miniature_training_records
    )
    baseline_model, baseline_optimizer = parent.initialize_model_optimizer()
    baseline = surface.FrozenMinibatchUpdater(
        model=baseline_model,
        optimizer=baseline_optimizer,
        batch=batch,
        round_number=1,
        minibatch_size=32,
    )
    baseline.finish()

    interrupted_model, interrupted_optimizer = (
        parent.initialize_model_optimizer()
    )
    interrupted = surface.FrozenMinibatchUpdater(
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        batch=batch,
        round_number=1,
        minibatch_size=32,
    )
    interrupted.step_once()
    restored = surface.FrozenMinibatchUpdater.from_snapshot(
        surface.deserialize_binary_state(
            surface.serialize_binary_state(interrupted.snapshot())
        ),
        minibatch_size=32,
    )
    restored.finish()
    assert restored.closed_step_ids == baseline.closed_step_ids
    for key, value in baseline.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])
    assert parent.stable_hash(baseline.optimizer.state_dict()) == (
        parent.stable_hash(restored.optimizer.state_dict())
    )


def test_real_updated_model_orphan_commit_reuses_exact_bytes(
    tmp_path: Path,
    miniature_training_records: list[dict],
) -> None:
    batch = surface.training_records_to_ppo_batch(
        miniature_training_records
    )
    model, optimizer = parent.initialize_model_optimizer()
    updater = surface.FrozenMinibatchUpdater(
        model=model,
        optimizer=optimizer,
        batch=batch,
        round_number=1,
        minibatch_size=32,
    )
    updater.step_once()
    runtime = _runtime_payload()
    runtime["model_state"] = copy.deepcopy(model.state_dict())
    runtime["optimizer_state"] = copy.deepcopy(optimizer.state_dict())
    runtime["optimizer_step_ids"] = list(updater.closed_step_ids)
    contract = {
        **_commit_contract(),
        "execution_mode": "scientific",
    }
    surface.initialize_commit_store(
        phase_dir=tmp_path,
        **contract,
        initial_state={"fixture": 0},
    )
    unit_id = "optimizer:actual-model:0001"
    with pytest.raises(RuntimeError):
        surface.commit_unit(
            phase_dir=tmp_path,
            **contract,
            unit_id=unit_id,
            post_state=runtime,
            journal_payload={"kind": "actual_ppo_update"},
            crash_stage="after_record",
        )
    paths = surface._commit_paths(
        tmp_path,
        sequence=1,
        unit_id=unit_id,
    )
    before = {
        key: path.read_bytes()
        for key, path in paths.items()
        if key in {"state", "journal", "record"}
    }
    resumed = surface.commit_unit(
        phase_dir=tmp_path,
        **contract,
        unit_id=unit_id,
        post_state=runtime,
        journal_payload={"kind": "actual_ppo_update"},
    )
    after = {
        key: path.read_bytes()
        for key, path in paths.items()
        if key in {"state", "journal", "record"}
    }
    assert before == after
    assert resumed["state"]["optimizer_step_ids"] == [
        updater.closed_step_ids[0]
    ]


def test_rolling_resume_orphan_reuse_and_bounded_linear_storage(
    tmp_path: Path,
) -> None:
    contract = _rolling_contract()
    first = {"step": 0, "buffer": torch.arange(128)}
    with pytest.raises(RuntimeError):
        surface.write_rolling_resume_boundary(
            root=tmp_path,
            contract=contract,
            unit_id="step-0",
            state=first,
            crash_stage="after_journal",
        )
    reused = surface.write_rolling_resume_boundary(
        root=tmp_path,
        contract=contract,
        unit_id="step-0",
        state=first,
    )
    assert reused["state"]["step"] == 0
    assert reused["journal_record_count"] == 1

    hypothetical_immutable_bytes = len(
        surface.serialize_binary_state(first)
    )
    largest_state_bytes = hypothetical_immutable_bytes
    for step in range(1, 41):
        payload = {
            "step": step,
            "buffer": torch.arange((step + 1) * 128),
        }
        serialized_bytes = len(surface.serialize_binary_state(payload))
        hypothetical_immutable_bytes += serialized_bytes
        largest_state_bytes = max(largest_state_bytes, serialized_bytes)
        surface.write_rolling_resume_boundary(
            root=tmp_path,
            contract=contract,
            unit_id=f"step-{step}",
            state=payload,
        )
    audit = surface.rolling_resume_storage_audit(
        tmp_path,
        contract=contract,
        planned_resume_boundaries=64,
        projected_journal_bytes_per_boundary=1_024,
        projected_root_blob_bytes=100_000,
        projected_epoch_commit_bytes=100_000,
        projected_checkpoint_bytes=10_000,
        projected_other_bytes=10_000,
        cap_gib=1.0,
    )
    assert audit["passes"]
    assert audit["slot_count"] == 2
    assert audit["journal_records"] == 41
    assert audit["slot_bytes"] <= 2 * largest_state_bytes
    assert audit["total_bytes"] < hypothetical_immutable_bytes


def test_rolling_resume_rejects_wrong_contract_or_current_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _rolling_contract()
    surface.write_rolling_resume_boundary(
        root=tmp_path,
        contract=contract,
        unit_id="step-0",
        state={"step": 0},
    )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.load_rolling_resume_boundary(
            tmp_path,
            contract=_rolling_contract(command="wrong command"),
        )
    original = surface.sha256_path

    def changed(path, root=surface.REPO_ROOT):
        if Path(path).resolve() == surface.RUNNER_PATH.resolve():
            return "0" * 64
        return original(path, root)

    monkeypatch.setattr(surface, "sha256_path", changed)
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.load_rolling_resume_boundary(
            tmp_path,
            contract=contract,
        )


def _mini_training_engine_run(
    phase_dir: Path,
    *,
    interrupt_after_boundary: str | None = None,
) -> dict:
    return surface.execute_training_engine(
        rows=_small_manifest_rows("training", 2),
        phase_dir=phase_dir,
        marker_file_sha256="1" * 64,
        marker_payload_sha256="2" * 64,
        phase_lock_file_sha256="3" * 64,
        manifest_file_sha256="4" * 64,
        manifest_payload_sha256="5" * 64,
        command="miniature training execute",
        config=surface.TrainingEngineConfig(
            rounds=1,
            roots_per_round=2,
            env_count=2,
            minibatch_size=32,
            max_moves=surface.MAX_MOVES,
            execution_mode="miniature_fixture",
        ),
        interrupt_after_boundary=interrupt_after_boundary,
    )


def test_training_engine_interrupted_resume_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    baseline = _mini_training_engine_run(tmp_path / "baseline")
    resumed_dir = tmp_path / "resumed"
    for boundary in ("collection", "update", "checkpoint"):
        with pytest.raises(surface.J1ExecutionPlannedInterruption):
            _mini_training_engine_run(
                resumed_dir,
                interrupt_after_boundary=boundary,
            )
    resumed = _mini_training_engine_run(resumed_dir)
    baseline_state = baseline["state"]
    resumed_state = resumed["state"]
    assert baseline_state["engine_stage"] == "complete"
    assert resumed_state["engine_stage"] == "complete"
    assert (
        baseline_state["optimizer_step_ids"]
        == resumed_state["optimizer_step_ids"]
    )
    assert (
        baseline_state["all_completed_root_ids"]
        == resumed_state["all_completed_root_ids"]
    )
    for key, value in baseline_state["model_state"].items():
        assert torch.equal(value, resumed_state["model_state"][key])
    assert parent.stable_hash(baseline_state["optimizer_state"]) == (
        parent.stable_hash(resumed_state["optimizer_state"])
    )
    assert parent.stable_hash(baseline_state["round_aggregates"]) == (
        parent.stable_hash(resumed_state["round_aggregates"])
    )


def test_round64_candidate_checkpoint_save_load_and_corruption(
    tmp_path: Path,
    miniature_training_records: list[dict],
) -> None:
    batch = surface.training_records_to_ppo_batch(
        miniature_training_records
    )
    model, optimizer = parent.initialize_model_optimizer()
    updater = surface.FrozenMinibatchUpdater(
        model=model,
        optimizer=optimizer,
        batch=batch,
        round_number=1,
        minibatch_size=32,
    )
    updater.step_once()
    payload = surface.candidate_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        training_manifest_identity={"payload_sha256": "a" * 64},
        training_marker_file_sha256="b" * 64,
        training_result_input_sha256="c" * 64,
    )
    path = tmp_path / "round64_candidate.bin"
    identity = surface.write_candidate_checkpoint(path, payload)
    policy = surface.load_authoritative_candidate_policy(
        checkpoint_identity=identity
    )
    assert callable(policy)
    corrupted = bytearray(path.read_bytes())
    corrupted[-1] ^= 1
    path.write_bytes(bytes(corrupted))
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.load_authoritative_candidate_policy(
            checkpoint_identity=identity
        )


def _first_legal_policy(state, sim, _rng) -> int:
    return min(sim.legal_actions(state))


def test_paired_full_policy_crn_identity_for_equal_policies() -> None:
    row = _small_manifest_rows("development", 1)[0]
    result = surface.execute_paired_full_policy_rows(
        rows=[row],
        candidate_policy=_first_legal_policy,
        control_policy=_first_legal_policy,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
    )[0]
    for key in (
        "start_score",
        "final_score",
        "moves",
        "max_tile",
        "terminal_state_sha256",
    ):
        assert result["candidate"][key] == result["control"][key]


def test_paired_evaluator_resumes_after_candidate_arm_without_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _small_manifest_rows("development", 1)[0]

    def clock():
        value = {"tick": 0}

        def read() -> float:
            value["tick"] += 1
            return value["tick"] * 0.001

        return read

    monkeypatch.setattr(surface.time, "perf_counter", clock())
    baseline = surface.PairedEvaluationSession(
        rows=[row],
        candidate_policy=_first_legal_policy,
        control_policy=_first_legal_policy,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
    ).finish()

    calls = {"candidate": 0, "control": 0}

    def candidate(state, sim, rng) -> int:
        calls["candidate"] += 1
        return _first_legal_policy(state, sim, rng)

    def control(state, sim, rng) -> int:
        calls["control"] += 1
        return _first_legal_policy(state, sim, rng)

    monkeypatch.setattr(surface.time, "perf_counter", clock())
    interrupted = surface.PairedEvaluationSession(
        rows=[row],
        candidate_policy=candidate,
        control_policy=control,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
    )
    assert interrupted.step_arm()["boundary"] == "candidate_arm_committed"
    candidate_calls_after_commit = calls["candidate"]
    restored = surface.PairedEvaluationSession.from_snapshot(
        surface.deserialize_binary_state(
            surface.serialize_binary_state(interrupted.snapshot())
        ),
        rows=[row],
        candidate_policy=candidate,
        control_policy=control,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
    )
    resumed = restored.finish()
    assert calls["candidate"] == candidate_calls_after_commit
    assert calls["control"] > 0
    assert parent.stable_hash(resumed) == parent.stable_hash(baseline)


def test_paired_resume_rejects_pending_arm_stream_drift() -> None:
    row = _small_manifest_rows("development", 1)[0]
    session = surface.PairedEvaluationSession(
        rows=[row],
        candidate_policy=_first_legal_policy,
        control_policy=_first_legal_policy,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
    )
    session.step_arm()
    changed = copy.deepcopy(session.snapshot())
    changed["pending_candidate"]["logical_stream_id"] += 1
    body = dict(changed)
    body.pop("session_state_sha256")
    changed["session_state_sha256"] = parent.stable_hash(body)
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.PairedEvaluationSession.from_snapshot(
            changed,
            rows=[row],
            candidate_policy=_first_legal_policy,
            control_policy=_first_legal_policy,
            candidate_policy_identity="candidate-fixture-sha",
            control_policy_identity="control-fixture-sha",
        )


def test_paired_snapshot_externalizes_completed_pairs_once(
    tmp_path: Path,
) -> None:
    row = _small_manifest_rows("development", 1)[0]
    session = surface.PairedEvaluationSession(
        rows=[row],
        candidate_policy=_first_legal_policy,
        control_policy=_first_legal_policy,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
    )
    session.step_arm()
    session.step_arm()
    blob_dir = tmp_path / surface.PAIR_BLOBS_DIR
    snapshot = session.snapshot(completed_blob_dir=blob_dir)
    assert snapshot["completed_storage"] == "immutable_pair_blobs"
    assert snapshot["completed_pairs"] == []
    assert len(snapshot["completed_pair_refs"]) == 1
    restored = surface.PairedEvaluationSession.from_snapshot(
        snapshot,
        rows=[row],
        candidate_policy=_first_legal_policy,
        control_policy=_first_legal_policy,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
        completed_blob_dir=blob_dir,
    )
    assert restored.is_complete()
    assert len(restored.completed_pairs) == 1


def _mini_paired_engine_run(
    phase_dir: Path,
    *,
    interrupt_after_boundary: str | None = None,
) -> dict:
    return surface.execute_paired_evaluation_engine(
        rows=_small_manifest_rows("development", 1),
        phase_dir=phase_dir,
        phase="development",
        marker_file_sha256="6" * 64,
        marker_payload_sha256="8" * 64,
        phase_lock_file_sha256="7" * 64,
        manifest_file_sha256="9" * 64,
        manifest_payload_sha256="a" * 64,
        command="miniature development execute",
        candidate_policy=_first_legal_policy,
        control_policy=_first_legal_policy,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
        execution_mode="miniature_fixture",
        interrupt_after_boundary=interrupt_after_boundary,
    )


def test_paired_engine_arm_boundary_resume_matches_uninterrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def clock():
        value = {"tick": 0}

        def read() -> float:
            value["tick"] += 1
            return value["tick"] * 0.001

        return read

    monkeypatch.setattr(surface.time, "perf_counter", clock())
    baseline = _mini_paired_engine_run(tmp_path / "baseline")

    resumed_dir = tmp_path / "resumed"
    monkeypatch.setattr(surface.time, "perf_counter", clock())
    with pytest.raises(surface.J1ExecutionPlannedInterruption):
        _mini_paired_engine_run(
            resumed_dir,
            interrupt_after_boundary="candidate_arm_committed",
        )
    with pytest.raises(surface.J1ExecutionPlannedInterruption):
        _mini_paired_engine_run(
            resumed_dir,
            interrupt_after_boundary="paired_root_committed",
        )
    resumed = _mini_paired_engine_run(resumed_dir)
    assert parent.stable_hash(resumed["rows"]) == parent.stable_hash(
        baseline["rows"]
    )


def _mini_bounded_training_engine_run(
    phase_dir: Path,
    *,
    interrupt_after_boundary: str | None = None,
    wall_clock=None,
) -> dict:
    return surface.execute_training_engine_bounded(
        rows=_small_manifest_rows("training", 2),
        phase_dir=phase_dir,
        marker_file_sha256="1" * 64,
        marker_payload_sha256="2" * 64,
        phase_lock_file_sha256="3" * 64,
        manifest_file_sha256="4" * 64,
        manifest_payload_sha256="5" * 64,
        command="bounded miniature training",
        config=surface.TrainingEngineConfig(
            rounds=1,
            roots_per_round=2,
            env_count=2,
            minibatch_size=32,
            max_moves=surface.MAX_MOVES,
            execution_mode="miniature_fixture",
        ),
        interrupt_after_boundary=interrupt_after_boundary,
        operational_audit_fn=surface.fixture_phase_operational_audit,
        wall_clock=wall_clock,
    )


def _assert_training_terminal_equivalent(
    left: dict,
    right: dict,
) -> None:
    left_state = left["state"]
    right_state = right["state"]
    assert left_state["engine_stage"] == right_state["engine_stage"] == "complete"
    for key in (
        "optimizer_step_ids",
        "all_completed_root_ids",
        "round_aggregates",
    ):
        assert parent.stable_hash(left_state[key]) == parent.stable_hash(
            right_state[key]
        )
    for key, tensor in left_state["model_state"].items():
        assert torch.equal(tensor, right_state["model_state"][key])
    assert parent.stable_hash(left_state["optimizer_state"]) == (
        parent.stable_hash(right_state["optimizer_state"])
    )


def test_bounded_training_engine_interrupted_resume_is_bit_exact(
    tmp_path: Path,
) -> None:
    baseline = _mini_bounded_training_engine_run(tmp_path / "baseline")
    resumed_dir = tmp_path / "resumed"
    for boundary in ("collection", "update", "checkpoint"):
        with pytest.raises(surface.J1ExecutionPlannedInterruption):
            _mini_bounded_training_engine_run(
                resumed_dir,
                interrupt_after_boundary=boundary,
            )
    resumed = _mini_bounded_training_engine_run(resumed_dir)
    _assert_training_terminal_equivalent(baseline, resumed)
    assert len(resumed["state"]["optimizer_step_ids"]) == 8
    assert resumed["resource_clock"]["attempts_abandoned"] == 0
    assert resumed["commit_store_metrics"]["full_chain_scan_count"] == 2
    assert baseline["commit_store_metrics"]["append_count"] == 5
    assert baseline["commit_store_metrics"]["unit_count"] == 6
    assert resumed["commit_store_metrics"]["unit_count"] == 6
    assert resumed["commit_store_metrics"][
        "committed_unit_prefix_copied"
    ] is False


def test_bounded_training_actual_io_is_linear_and_batch_cached(
    tmp_path: Path,
) -> None:
    baseline = _mini_bounded_training_engine_run(tmp_path / "baseline")
    metrics = baseline["io_metrics"]
    assert metrics["root_blob_writes"] == 2
    assert metrics["root_blob_reads"] == 4
    assert 0 < metrics["transition_chunk_writes"] < 200
    assert metrics["round_batch_writes"] == 1
    assert metrics["round_batch_loads"] == 0
    assert baseline["output_accounting"]["full_scan_count"] == 4

    resumed_dir = tmp_path / "resumed"
    with pytest.raises(surface.J1ExecutionPlannedInterruption):
        _mini_bounded_training_engine_run(
            resumed_dir,
            interrupt_after_boundary="update",
        )
    resumed = _mini_bounded_training_engine_run(resumed_dir)
    assert resumed["io_metrics"]["round_batch_loads"] == 1
    assert resumed["io_metrics"]["round_batch_bytes_read"] > 0


@pytest.mark.parametrize(
    "crash_boundary",
    [
        "transition_retirement_pre_apply",
        "transition_retirement_after_manifest",
        "transition_retirement_mid_delete",
        "batch_retirement_pre_apply",
        "batch_retirement_after_manifest",
        "batch_retirement_mid_delete",
    ],
)
def test_bounded_training_retirement_recovers_each_crash_window(
    tmp_path: Path,
    crash_boundary: str,
) -> None:
    phase_dir = tmp_path / crash_boundary
    with pytest.raises(surface.J1ExecutionPlannedInterruption):
        _mini_bounded_training_engine_run(
            phase_dir,
            interrupt_after_boundary=crash_boundary,
        )
    result = _mini_bounded_training_engine_run(phase_dir)
    assert result["completed"]
    assert not list(
        (phase_dir / surface.TRANSITION_CHUNKS_DIR).rglob("*.bin")
    )
    assert not list(
        (phase_dir / surface.ROUND_BATCHES_DIR).rglob("*.bin")
    )
    transition_manifests = list(
        (
            phase_dir / surface.TRANSITION_CHUNK_RETIREMENTS_DIR
        ).rglob("*.json")
    )
    batch_manifests = list(
        (
            phase_dir / surface.ROUND_BATCH_RETIREMENTS_DIR
        ).rglob("*.json")
    )
    assert len(transition_manifests) == 1
    assert len(batch_manifests) == 1
    transition = surface.load_json(transition_manifests[0])
    assert len(transition["files"]) < transition["transition_row_count"]
    assert transition["transition_chunk_count"] <= (
        surface.TRAINING_TRANSITION_FILE_CAP
    )


def _mini_bounded_paired_engine_run(
    phase_dir: Path,
    *,
    interrupt_after_boundary: str | None = None,
    wall_clock=None,
    row_count: int = 1,
    block_pairs: int = 1,
) -> dict:
    return surface.execute_paired_evaluation_engine_bounded(
        rows=_small_manifest_rows("development", row_count),
        phase_dir=phase_dir,
        phase="development",
        marker_file_sha256="6" * 64,
        marker_payload_sha256="8" * 64,
        phase_lock_file_sha256="7" * 64,
        manifest_file_sha256="9" * 64,
        manifest_payload_sha256="a" * 64,
        command="bounded miniature development",
        candidate_policy=_first_legal_policy,
        control_policy=_first_legal_policy,
        candidate_policy_identity="candidate-fixture-sha",
        control_policy_identity="control-fixture-sha",
        max_moves=surface.MAX_MOVES,
        interrupt_after_boundary=interrupt_after_boundary,
        execution_mode="miniature_fixture",
        block_pairs=block_pairs,
        operational_audit_fn=surface.fixture_phase_operational_audit,
        wall_clock=wall_clock,
    )


def test_bounded_paired_engine_resumes_candidate_and_preseal_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def perf_clock():
        value = {"tick": 0}

        def read() -> float:
            value["tick"] += 1
            return value["tick"] * 0.001

        return read

    monkeypatch.setattr(surface.time, "perf_counter", perf_clock())
    baseline = _mini_bounded_paired_engine_run(tmp_path / "baseline")
    resumed_dir = tmp_path / "resumed"
    monkeypatch.setattr(surface.time, "perf_counter", perf_clock())
    with pytest.raises(surface.J1ExecutionPlannedInterruption):
        _mini_bounded_paired_engine_run(
            resumed_dir,
            interrupt_after_boundary="candidate_arm_committed",
        )
    with pytest.raises(surface.J1ExecutionPlannedInterruption):
        _mini_bounded_paired_engine_run(
            resumed_dir,
            interrupt_after_boundary="paired_root_committed",
        )
    resumed = _mini_bounded_paired_engine_run(resumed_dir)
    assert parent.stable_hash(resumed["rows"]) == parent.stable_hash(
        baseline["rows"]
    )
    assert resumed["resource_clock"]["attempts_started"] == 2
    assert resumed["resource_clock"]["attempts_abandoned"] == 0
    assert resumed["commit_store_metrics"]["full_chain_scan_count"] == 2


def test_bounded_paired_actual_blob_io_scales_linearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(surface.time, "perf_counter", lambda: 1.0)
    two = _mini_bounded_paired_engine_run(
        tmp_path / "two",
        row_count=2,
        block_pairs=2,
    )
    four = _mini_bounded_paired_engine_run(
        tmp_path / "four",
        row_count=4,
        block_pairs=2,
    )
    assert two["io_metrics"]["pair_blob_writes"] == 2
    assert four["io_metrics"]["pair_blob_writes"] == 4
    assert two["io_metrics"]["pair_blob_terminal_reads"] == 2
    assert four["io_metrics"]["pair_blob_terminal_reads"] == 4
    assert (
        two["io_metrics"]["pair_blob_resume_reference_reads"]
        == four["io_metrics"]["pair_blob_resume_reference_reads"]
        == 0
    )
    assert four["io_metrics"]["pair_blob_bytes_written"] < (
        2.5 * two["io_metrics"]["pair_blob_bytes_written"]
    )
    assert four["io_metrics"]["pair_blob_bytes_read"] < (
        2.5 * two["io_metrics"]["pair_blob_bytes_read"]
    )


def test_indexed_commit_store_append_work_is_linear(
    tmp_path: Path,
) -> None:
    def run(root: Path, count: int) -> dict:
        accountant = surface.PhaseOutputAccountant(root)
        store = surface.IndexedCommitStore(
            phase_dir=root,
            **_commit_contract(),
            initial_state={"fixture": 0},
            output_accountant=accountant,
        )
        for index in range(count):
            store.commit(
                unit_id=f"unit-{index}",
                post_state={"fixture": index + 1},
                journal_payload={"fixture": index + 1},
            )
        return store.metrics()

    n = run(tmp_path / "n", 16)
    two_n = run(tmp_path / "two_n", 32)
    assert n["full_chain_scan_count"] == two_n["full_chain_scan_count"] == 1
    assert n["append_count"] == 16
    assert two_n["append_count"] == 32
    assert two_n["current_head_verification_count"] == (
        2 * n["current_head_verification_count"] - 1
    )
    assert two_n["current_head_verified_bytes"] < (
        2.25 * n["current_head_verified_bytes"]
    )


def test_abandoned_attempt_charge_is_bounded_not_downtime(
    tmp_path: Path,
) -> None:
    contract = _rolling_contract()

    def recovered_charge(root: Path, resume_after: float) -> float:
        started_clock = iter([0.0])
        first = surface.RuntimeChargeLedger(
            root=root,
            contract=contract,
            wall_clock=lambda: next(started_clock),
        )
        first.begin("round=1|collection_tick=1")
        second = surface.RuntimeChargeLedger(
            root=root,
            contract=contract,
            wall_clock=lambda: resume_after,
        )
        return second.summary()["active_seconds"]

    one_hour = recovered_charge(tmp_path / "one_hour", 3600.0)
    one_day = recovered_charge(tmp_path / "one_day", 86400.0)
    assert one_hour == one_day
    assert one_hour == surface.ABANDONED_ATTEMPT_CHARGE_SECONDS[
        "training_collection_tick_block"
    ]


def _write_fixture_readiness(
    readiness_dir: Path,
    execution_root: Path,
) -> None:
    lock = surface.write_immutable_json(
        readiness_dir / surface.READINESS_LOCK_NAME,
        {
            "version": "fixture-readiness-lock-v1",
            "decision": "READY_J1_EXECUTION_SURFACE",
            "bound_readiness_dir": str(readiness_dir.resolve()),
            "bound_execution_root": str(execution_root.resolve()),
            "charter_file_sha256": surface.sha256_path(
                surface.CHARTER_PATH
            ),
            "runner_file_sha256": surface.sha256_path(
                surface.RUNNER_PATH
            ),
            "test_file_sha256": surface.sha256_path(surface.TEST_PATH),
        },
        field="readiness_lock_payload_sha256",
    )
    lock_identity = surface.immutable_json_identity(
        readiness_dir / surface.READINESS_LOCK_NAME,
        payload_field="readiness_lock_payload_sha256",
    )
    surface.write_immutable_json(
        readiness_dir / surface.READINESS_RESULT_NAME,
        {
            "version": "fixture-readiness-result-v1",
            "decision": "READY_J1_EXECUTION_SURFACE",
            "readiness_lock": lock_identity,
            "passes": True,
        },
        field="readiness_result_payload_sha256",
    )
    assert lock["decision"] == "READY_J1_EXECUTION_SURFACE"


def _patch_tiny_dispatch_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = copy.deepcopy(surface.STREAMS)
    streams["training"]["rows"] = 2
    streams["development"]["rows"] = 1
    streams["confirmation"]["rows"] = 1
    monkeypatch.setattr(surface, "STREAMS", streams)
    monkeypatch.setattr(surface, "TRAIN_ROOTS", 2)
    monkeypatch.setattr(surface, "DEVELOPMENT_PAIRS", 1)
    monkeypatch.setattr(surface, "CONFIRMATION_PAIRS", 1)
    monkeypatch.setattr(surface, "TOTAL_GAME_ARMS", 6)


def _fixture_open_audit(*, output_dir: Path) -> dict:
    return {
        "output_dir": str(output_dir),
        "checks": {"miniature_fixture_only": True},
        "passes": True,
    }


def _fixture_wall_clock():
    value = {"time": 0.0}

    def read() -> float:
        value["time"] += 0.001
        return value["time"]

    return read


def _phase_fixture_hooks(
    phase: str,
    *,
    wall_clock,
) -> dict:
    common = {
        "open_operational_audit_fn": _fixture_open_audit,
        "opened_at": f"2026-07-27T22:00:0{surface.PHASES.index(phase)}Z",
        "hostname": "j1-dispatch-fixture",
        "operational_audit_fn": surface.fixture_phase_operational_audit,
        "wall_clock": wall_clock,
        "start_identity": f"fixture-owner-{phase}",
    }
    if phase == "training":
        return {
            **common,
            "training_config": surface.TrainingEngineConfig(
                rounds=1,
                roots_per_round=2,
                env_count=2,
                minibatch_size=32,
                max_moves=surface.MAX_MOVES,
                execution_mode="miniature_fixture",
            ),
        }
    return {
        **common,
        "candidate_policy": _first_legal_policy,
        "control_policy": _first_legal_policy,
        "candidate_policy_identity": "fixture-candidate-policy",
        "control_policy_identity": "fixture-control-policy",
        "pair_count": 1,
        "block_pairs": 1,
    }


def _prepare_tiny_training_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict]:
    _patch_tiny_dispatch_manifests(monkeypatch)
    execution_root = tmp_path / "execution"
    readiness_dir = tmp_path / "readiness"
    _write_fixture_readiness(readiness_dir, execution_root)
    hooks = _phase_fixture_hooks(
        "training",
        wall_clock=_fixture_wall_clock(),
    )
    for action in ("seal-phase-lock", "open", "materialize"):
        surface.dispatch_phase_command(
            action=action,
            phase="training",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=hooks,
        )
    return execution_root, readiness_dir, hooks


def test_full_scale_bounded_projection_passes_central_caps() -> None:
    projection = surface.full_scale_runtime_storage_projection()
    assert projection["passes"]
    assert projection["training"]["central"]["storage"][
        "projected_with_margin_gib"
    ] < 24.0
    assert projection["training"]["central"]["bounded_io"][
        "created_files"
    ] <= surface.TRAINING_OUTPUT_FILE_CAP
    assert projection["training"]["central"]["bounded_io"][
        "fsync_count"
    ] <= surface.TRAINING_FSYNC_CAP
    for phase in ("development", "confirmation"):
        central = projection["evaluation"][phase]["central"]
        assert central["runtime_at_most_91pct_cap"]
        assert central["storage"]["passes"]
    sensitivity = projection["training"]["sensitivity_5000_moves"]
    assert sensitivity["diagnostic_not_conjunctive"]
    assert not sensitivity["storage"]["passes"]
    assert not sensitivity["created_files_within_cap"]
    assert not sensitivity["fsync_count_within_cap"]


def test_readiness_prepare_is_zero_work_and_seals_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness_dir = tmp_path / "readiness"
    execution_root = tmp_path / "execution"
    monkeypatch.setattr(surface, "FOCUSED_TEST_COUNT", 84)
    monkeypatch.setattr(surface, "APPLICABLE_TEST_COUNT", 7)
    monkeypatch.setattr(
        surface,
        "APPLICABLE_TEST_COMMAND",
        "fixture applicable regressions",
    )
    monkeypatch.setattr(
        surface,
        "DOCUMENTED_HISTORICAL_STATE_DESELECTIONS",
        ("tests/fixture.py::test_historical_state",),
    )
    evidence = surface.write_execution_test_evidence(
        readiness_dir=readiness_dir,
        focused_command=surface.FOCUSED_TEST_COMMAND,
        focused_passed=84,
        parent_j1_command=surface.PARENT_TEST_COMMANDS[0][1],
        parent_j1_passed=surface.PARENT_TEST_COMMANDS[0][2],
        parent_j1a_command=surface.PARENT_TEST_COMMANDS[1][1],
        parent_j1a_passed=surface.PARENT_TEST_COMMANDS[1][2],
        applicable_command="fixture applicable regressions",
        applicable_passed=7,
        documented_deselections=[
            "tests/fixture.py::test_historical_state"
        ],
    )
    assert evidence["passes"]
    result = surface.prepare_execution_readiness(
        readiness_dir=readiness_dir,
        execution_root=execution_root,
        operational_audit_fn=lambda **_kwargs: {
            "checks": {"fixture_only": True},
            "passes": True,
        },
    )
    assert result["decision"] == "READY_J1_EXECUTION_SURFACE"
    assert not execution_root.exists()
    assert sorted(path.name for path in readiness_dir.iterdir()) == sorted(
        [
            surface.TEST_EVIDENCE_NAME,
            surface.SCHEMA_NAME,
            surface.MANIFEST_NAME,
            surface.RUNTIME_STORAGE_PROJECTION_NAME,
            surface.READINESS_LOCK_NAME,
            surface.READINESS_RESULT_NAME,
        ]
    )
    projection = surface.load_json(
        readiness_dir / surface.RUNTIME_STORAGE_PROJECTION_NAME
    )
    assert projection["passes"]
    assert projection["zero_work"] == surface.ZERO_WORK


def test_production_parser_exposes_only_frozen_phase_commands() -> None:
    parser = surface.build_parser()
    help_text = parser.format_help()
    for command in surface.PRODUCTION_COMMANDS:
        assert command in help_text
    assert "promote" not in help_text
    parsed = parser.parse_args(
        [
            "execute",
            "--phase",
            "training",
            "--execution-root",
            "/tmp/j1-execution-fixture",
            "--readiness-dir",
            "/tmp/j1-readiness-fixture",
            "--jobs",
            "1",
        ]
    )
    assert parsed.subcommand == "execute"
    assert parsed.phase == "training"
    assert parsed.jobs == 1


def test_dispatcher_routes_tiny_three_phase_chain_only_to_bounded_engines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_dispatch_manifests(monkeypatch)
    execution_root = tmp_path / "execution"
    readiness_dir = tmp_path / "readiness"
    _write_fixture_readiness(readiness_dir, execution_root)
    monkeypatch.setattr(
        surface,
        "execute_training_engine",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("verbose training engine was reached")
        ),
    )
    monkeypatch.setattr(
        surface,
        "execute_paired_evaluation_engine",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("verbose paired engine was reached")
        ),
    )
    clock = _fixture_wall_clock()

    with pytest.raises((FileNotFoundError, surface.J1ExecutionIntegrityError)):
        surface.dispatch_phase_command(
            action="open",
            phase="training",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=_phase_fixture_hooks(
                "training",
                wall_clock=clock,
            ),
        )

    training_hooks = _phase_fixture_hooks(
        "training",
        wall_clock=clock,
    )
    for action in ("seal-phase-lock", "open", "materialize"):
        surface.dispatch_phase_command(
            action=action,
            phase="training",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=training_hooks,
        )
    training = surface.dispatch_phase_command(
        action="execute",
        phase="training",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        execution_mode="miniature_fixture",
        fixture_hooks=training_hooks,
    )
    assert training["result"]["decision"] == "READY_J1_TRAINING_SANITY"
    assert training["result"]["scientific_authority"] is False
    assert training["result"]["bounded_engine"] == (
        "execute_training_engine_bounded"
    )

    access_path = tmp_path / "confirmation_access.json"
    surface.write_confirmation_access_audit(
        path=access_path,
        content_reads=0,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "dispatcher fixture"},
    )
    development_hooks = _phase_fixture_hooks(
        "development",
        wall_clock=clock,
    )
    surface.dispatch_phase_command(
        action="seal-phase-lock",
        phase="development",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        confirmation_access_audit_path=access_path,
        execution_mode="miniature_fixture",
        fixture_hooks=development_hooks,
    )
    precommitted_confirmation = (
        execution_root
        / surface.PRECOMMITTED_MANIFEST_DIR
        / "confirmation_root_manifest.json"
    )
    precommit_sha256 = surface.sha256_path(precommitted_confirmation)
    confirmation_paths = surface.phase_artifact_paths(
        execution_root=execution_root,
        phase="confirmation",
    )
    assert not confirmation_paths["marker"].exists()
    assert not confirmation_paths["manifest"].exists()
    assert not confirmation_paths["reservation"].exists()
    assert not confirmation_paths["consumption"].exists()
    for action in ("open", "materialize", "execute"):
        development = surface.dispatch_phase_command(
            action=action,
            phase="development",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=development_hooks,
        )
    assert development["result"]["decision"] == (
        "READY_J1_DEVELOPMENT_FULL_POLICY"
    )
    assert surface.sha256_path(precommitted_confirmation) == (
        precommit_sha256
    )
    confirmation_hooks = _phase_fixture_hooks(
        "confirmation",
        wall_clock=clock,
    )
    for action in (
        "seal-phase-lock",
        "open",
        "materialize",
        "execute",
    ):
        confirmation = surface.dispatch_phase_command(
            action=action,
            phase="confirmation",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=confirmation_hooks,
        )
    assert confirmation["result"]["decision"] == (
        "READY_J1_PROMOTION_REVIEW"
    )
    assert confirmation["result"]["scientific_authority"] is False
    assert confirmation["result"]["bounded_engine"] == (
        "execute_paired_evaluation_engine_bounded"
    )
    assert surface.sha256_path(precommitted_confirmation) == (
        precommit_sha256
    )


@pytest.mark.parametrize(
    "dispatch_boundary",
    ["owner", "reservation", "consumption"],
)
def test_dispatcher_recovers_dead_owner_before_commit_genesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dispatch_boundary: str,
) -> None:
    execution_root, readiness_dir, hooks = (
        _prepare_tiny_training_dispatch(tmp_path, monkeypatch)
    )
    interrupted_hooks = {
        **hooks,
        "interrupt_after_dispatch_boundary": dispatch_boundary,
    }
    with pytest.raises(surface.J1ExecutionPlannedInterruption):
        surface.dispatch_phase_command(
            action="execute",
            phase="training",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=interrupted_hooks,
        )
    paths = surface.phase_artifact_paths(
        execution_root=execution_root,
        phase="training",
    )
    assert not (paths["phase_dir"] / surface.COMMIT_HEAD_NAME).exists()
    consumption_bytes = (
        paths["consumption"].read_bytes()
        if paths["consumption"].exists()
        else None
    )
    loaded = surface._load_open_phase_contract(
        phase="training",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    recovered = surface.reclaim_dead_writer_owner(
        phase_dir=paths["phase_dir"],
        phase="training",
        marker_file_sha256=loaded["marker_identity"]["file_sha256"],
        phase_lock_file_sha256=loaded["lock_identity"]["file_sha256"],
        command=loaded["commands"]["execute"],
        execution_mode="miniature_fixture",
        pid_alive=lambda _pid: False,
        process_identity=lambda _pid: None,
        contention_audit={"passes": True, "fixture_only": True},
        new_pid=os.getpid(),
        new_start_identity=(
            f"recovered-owner-{dispatch_boundary}"
        ),
    )
    assert recovered["recovery"]["committed_boundary"]["mode"] == (
        "bootstrap_no_commit_head_v1"
    )
    terminal = surface.dispatch_phase_command(
        action="execute",
        phase="training",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        execution_mode="miniature_fixture",
        fixture_hooks=hooks,
    )
    assert terminal["result"]["decision"] == "READY_J1_TRAINING_SANITY"
    ledger = surface.load_json(paths["owner"])
    assert len(ledger["owners"]) == 2
    assert len(ledger["recoveries"]) == 1
    if consumption_bytes is not None:
        assert paths["consumption"].read_bytes() == consumption_bytes
        assert terminal["consumption"]["reused_existing_record"]
        assert terminal["consumption"]["owner_recovery_chain_verified"]
        assert terminal["consumption"][
            "opener_owner_record_sha256"
        ] == ledger["owners"][0]["owner_record_sha256"]
        assert terminal["consumption"][
            "current_owner_record_sha256"
        ] == ledger["owners"][1]["owner_record_sha256"]


def test_terminal_result_resume_repairs_missing_retention_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root, readiness_dir, hooks = (
        _prepare_tiny_training_dispatch(tmp_path, monkeypatch)
    )
    interrupted = {
        **hooks,
        "interrupt_after_terminal_boundary": "result",
    }
    with pytest.raises(surface.J1ExecutionPlannedInterruption):
        surface.dispatch_phase_command(
            action="execute",
            phase="training",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=interrupted,
        )
    paths = surface.phase_artifact_paths(
        execution_root=execution_root,
        phase="training",
    )
    result_bytes = paths["result"].read_bytes()
    assert not paths["retention"].exists()
    monkeypatch.setattr(
        surface,
        "execute_training_engine_bounded",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal resume reran the bounded engine")
        ),
    )
    resumed = surface.dispatch_phase_command(
        action="execute",
        phase="training",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        execution_mode="miniature_fixture",
        fixture_hooks=hooks,
    )
    assert resumed["terminal_already_sealed"]
    assert resumed["resumed_after_terminal"]
    assert paths["result"].read_bytes() == result_bytes
    assert paths["retention"].is_file()
    second = surface.dispatch_phase_command(
        action="execute",
        phase="training",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        execution_mode="miniature_fixture",
        fixture_hooks=hooks,
    )
    assert second["retention"] == resumed["retention"]


def test_fixture_ready_terminal_cannot_authorize_scientific_development(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root, readiness_dir, hooks = (
        _prepare_tiny_training_dispatch(tmp_path, monkeypatch)
    )
    surface.dispatch_phase_command(
        action="execute",
        phase="training",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        execution_mode="miniature_fixture",
        fixture_hooks=hooks,
    )
    access_path = tmp_path / "confirmation_access.json"
    surface.write_confirmation_access_audit(
        path=access_path,
        content_reads=0,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "cross-mode fixture"},
    )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.dispatch_phase_command(
            action="seal-phase-lock",
            phase="development",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            confirmation_access_audit_path=access_path,
            execution_mode="scientific",
        )
    assert not (
        execution_root / surface.PRECOMMITTED_MANIFEST_DIR
    ).exists()
    assert not surface.phase_artifact_paths(
        execution_root=execution_root,
        phase="development",
    )["lock"].exists()


def test_open_and_execute_reverify_predecessor_and_joint_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root, readiness_dir, hooks = (
        _prepare_tiny_training_dispatch(tmp_path, monkeypatch)
    )
    surface.dispatch_phase_command(
        action="execute",
        phase="training",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        execution_mode="miniature_fixture",
        fixture_hooks=hooks,
    )
    access_path = tmp_path / "confirmation_access.json"
    surface.write_confirmation_access_audit(
        path=access_path,
        content_reads=0,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "tamper fixture"},
    )
    development_hooks = _phase_fixture_hooks(
        "development",
        wall_clock=_fixture_wall_clock(),
    )
    surface.dispatch_phase_command(
        action="seal-phase-lock",
        phase="development",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        confirmation_access_audit_path=access_path,
        execution_mode="miniature_fixture",
        fixture_hooks=development_hooks,
    )
    training_result_path = surface.phase_artifact_paths(
        execution_root=execution_root,
        phase="training",
    )["result"]
    original_training = training_result_path.read_bytes()
    changed_training = surface.load_json(training_result_path)
    changed_training.pop("terminal_result_payload_sha256")
    changed_training["tampered_after_phase_lock"] = True
    changed_training = surface.payload_with_hash(
        changed_training,
        "terminal_result_payload_sha256",
    )
    training_result_path.write_text(
        json.dumps(changed_training, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.dispatch_phase_command(
            action="open",
            phase="development",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=development_hooks,
        )
    training_result_path.write_bytes(original_training)
    for action in ("open", "materialize"):
        surface.dispatch_phase_command(
            action=action,
            phase="development",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=development_hooks,
        )
    joint_path = (
        execution_root
        / surface.PRECOMMITTED_MANIFEST_DIR
        / surface.JOINT_MANIFEST_SEAL_NAME
    )
    changed_joint = surface.load_json(joint_path)
    changed_joint.pop("joint_manifest_seal_payload_sha256")
    changed_joint["tampered_before_execute"] = True
    changed_joint = surface.payload_with_hash(
        changed_joint,
        "joint_manifest_seal_payload_sha256",
    )
    joint_path.write_text(
        json.dumps(changed_joint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.dispatch_phase_command(
            action="execute",
            phase="development",
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            jobs=1,
            execution_mode="miniature_fixture",
            fixture_hooks=development_hooks,
        )


def test_unexpected_terminalization_error_seals_integrity_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root, readiness_dir, hooks = (
        _prepare_tiny_training_dispatch(tmp_path, monkeypatch)
    )

    def fail_terminal(**_kwargs):
        raise ValueError("deterministic terminalization fixture failure")

    monkeypatch.setattr(surface, "_seal_fixture_terminal", fail_terminal)
    terminal = surface.dispatch_phase_command(
        action="execute",
        phase="training",
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        jobs=1,
        execution_mode="miniature_fixture",
        fixture_hooks=hooks,
    )
    assert terminal["result"]["decision"] == "KILL_J1_INTEGRITY"
    assert terminal["result"]["failure_class"] == "integrity"
    assert terminal["result"]["error_type"] == "ValueError"
    assert terminal["result"]["scientific_authority"] is False
    assert terminal["retention"]["passes"]


def test_main_routes_scientific_phase_command_through_dispatcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed = {}

    def dispatch(**kwargs):
        observed.update(kwargs)
        return {"passes": True, "fixture": "cli-routing-only"}

    monkeypatch.setattr(surface, "dispatch_phase_command", dispatch)
    execution_root = tmp_path / "execution"
    readiness_dir = tmp_path / "readiness"
    result = surface.main(
        [
            "execute",
            "--phase",
            "training",
            "--execution-root",
            str(execution_root),
            "--readiness-dir",
            str(readiness_dir),
            "--jobs",
            "1",
        ]
    )
    assert result == 0
    assert observed == {
        "action": "execute",
        "phase": "training",
        "execution_root": execution_root,
        "readiness_dir": readiness_dir,
        "jobs": 1,
        "confirmation_access_audit_path": None,
        "execution_mode": "scientific",
    }
    assert json.loads(capsys.readouterr().out)["passes"]


def test_joint_sealed_candidate_checkpoint_rejects_preconfirmation_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_dispatch_manifests(monkeypatch)
    execution_root = tmp_path / "execution"
    training_paths = surface.phase_artifact_paths(
        execution_root=execution_root,
        phase="training",
    )
    training_manifest = surface.materialize_root_manifest(
        phase="training"
    )
    model, optimizer = parent.initialize_model_optimizer()
    checkpoint_payload = surface.candidate_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        training_manifest_identity=surface.root_manifest_identity(
            training_manifest
        ),
        training_marker_file_sha256="a" * 64,
        training_result_input_sha256="b" * 64,
    )
    checkpoint_identity = surface.write_candidate_checkpoint(
        training_paths["checkpoint"],
        checkpoint_payload,
    )
    training_result = surface.write_immutable_json(
        training_paths["result"],
        {
            "version": "scientific-lineage-fixture-v1",
            "phase": "training",
            "decision": "READY_J1_TRAINING_SANITY",
            "execution_mode": "scientific",
            "scientific_authority": True,
            "bounded_engine": "execute_training_engine_bounded",
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_authoritative": True,
            "checkpoint_quarantined": False,
        },
        field="terminal_result_payload_sha256",
    )
    access_path = tmp_path / "confirmation_access.json"
    surface.write_confirmation_access_audit(
        path=access_path,
        content_reads=0,
        streams_reserved=0,
        streams_consumed=0,
        evidence={"source": "candidate-lineage fixture"},
    )
    sealed = surface.seal_joint_evaluation_manifests(
        execution_root=execution_root,
        training_manifest=training_manifest,
        training_result=training_result,
        confirmation_access_audit_path=access_path,
    )
    assert sealed["passes"]
    changed = bytearray(training_paths["checkpoint"].read_bytes())
    changed[-1] ^= 1
    training_paths["checkpoint"].write_bytes(bytes(changed))
    with pytest.raises(surface.J1ExecutionIntegrityError):
        surface.verify_joint_candidate_lineage(
            execution_root=execution_root,
            expected_execution_mode="scientific",
        )

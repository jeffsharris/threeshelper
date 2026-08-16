from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
from pathlib import Path

import numpy as np
import pytest
import torch

from threes_rl import j2_incumbent_distillation_readiness as j2
from threes_rl import j2a1_distillation_fidelity_execution_surface as surface


ACTIVE_ROWS = surface.expected_active_rows()
BC_ROWS = [row for row in ACTIVE_ROWS if row["stage"] == surface.BC_STAGE]
VALIDATION_ROWS = [
    row for row in ACTIVE_ROWS if row["stage"] == surface.VALIDATION_STAGE
]


def _board(family: str) -> np.ndarray:
    board = np.full((4, 4), 3, dtype=np.int32)
    if family == "low_air":
        board.flat[:4] = 0
    elif family == "low_constrained":
        board.flat[0] = 0
        board.flat[-1] = 96
    elif family == "mid_progression":
        board.flat[0] = 0
        board.flat[-1] = 192
    elif family == "upper_progression":
        board.flat[0] = 0
        board.flat[-1] = 768
    else:
        raise ValueError(family)
    return board


def _raw_root(
    row: dict[str, object],
    *,
    families: tuple[str, ...] = ("low_air", "mid_progression"),
    deltas: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if deltas is None:
        deltas = tuple(3 for _ in families)
    assert len(deltas) == len(families)
    current = 0
    transitions = []
    for index, (family, delta) in enumerate(zip(families, deltas)):
        observation = np.zeros(j2.OBSERVATION_WIDTH, dtype=np.float32)
        observation[index % j2.OBSERVATION_WIDTH] = float(index + 1) / 10.0
        transitions.append(
            {
                "transition_index": index,
                "observation": observation,
                "legal_mask": np.asarray([True, True, False, False]),
                "teacher_action": index % 2,
                "current_score": current,
                "score_delta": delta,
                "board": _board(family),
            }
        )
        current += delta
    return surface.seal_teacher_root_record(
        {
            "version": f"{surface.VERSION}_teacher_root_v1",
            "row": copy.deepcopy(row),
            "root_id": row["root_id"],
            "ancestry_id": row["ancestry_id"],
            "stage": row["stage"],
            "shard": int(row["row_index"]) % surface.SHARDS,
            "normal_start": True,
            "starter_tile": None,
            "natural_terminal": True,
            "start_score": 0,
            "final_score": sum(deltas),
            "final_max_tile": int(max(_board(family).max() for family in families)),
            "policy_latency_seconds": 0.01,
            "survival": float(len(transitions)),
            "transitions": transitions,
        },
        authoritative_row=row,
    )


def _normalized_root(
    row: dict[str, object],
    **kwargs: object,
) -> dict[str, object]:
    raw = _raw_root(row, **kwargs)
    return surface.validate_teacher_root_record(
        raw,
        authoritative_row=row,
    )


def _fixture_collector(
    rows: list[dict[str, object]] | tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    return [_raw_root(dict(row)) for row in rows]


def _fixture_arm_runner(
    rows: list[dict[str, object]] | tuple[dict[str, object], ...],
    _model: j2.J2ActorCritic,
) -> list[dict[str, object]]:
    return [
        {
            "start_score": 0,
            "final_score": 12,
            "max_tile": 1536,
            "moves": 2,
            "latency_seconds": 0.001,
            "survival": 2.0,
            "illegal_actions": 0,
        }
        for _ in rows
    ]


def _passing_inventory(
    _roots: object,
    expected_root_count: int,
) -> dict[str, object]:
    return {
        "version": "miniature_inventory_v1",
        "validation_root_count": expected_root_count,
        "natural_inventory_sha256": "1" * 64,
        "capped_inventory_sha256": "2" * 64,
        "capped_refs": [],
        "checks": {"miniature_fixture": True},
        "decision": "READY_J2A1_DISTILLATION_OPTIMIZER",
        "passes": True,
    }


def _passing_mechanism(
    _model: object,
    _roots: object,
    _inventory: object,
) -> dict[str, object]:
    return {
        "version": "miniature_mechanism_v1",
        "checks": {"miniature_fixture": True},
        "decision": "READY_J2A1_CLOSED_LOOP_FIDELITY",
        "passes": True,
    }


def _passing_fidelity(
    _pairs: object,
    _rows: object,
) -> dict[str, object]:
    return {
        "version": "miniature_fidelity_v1",
        "checks": {"miniature_fixture": True},
        "decision": surface.READY_EXECUTION,
        "passes": True,
    }


def _mini_rows() -> list[dict[str, object]]:
    return [
        copy.deepcopy(BC_ROWS[0]),
        copy.deepcopy(BC_ROWS[1]),
        copy.deepcopy(VALIDATION_ROWS[0]),
        copy.deepcopy(VALIDATION_ROWS[1]),
    ]


def _mini_config() -> surface.EngineConfig:
    return surface.EngineConfig(
        execution_mode="miniature_fixture",
        expected_bc_roots=2,
        expected_validation_pairs=2,
        distillation_epochs=j2.DISTILLATION_EPOCHS,
        distillation_minibatch_size=2,
    )


def _attempt_ledger(
    directory: Path,
    *,
    contract: str = "a" * 64,
) -> surface.AttemptRuntimeLedger:
    return surface.AttemptRuntimeLedger(
        path=directory / "attempts.jsonl",
        contract_sha256=contract,
    )


def test_authority_uses_semantic_stream_keys_and_exact_counts() -> None:
    audit = surface.authority_audit()
    assert audit["passes"]
    assert audit["active_root_rows"] == 14_336
    assert audit["active_game_arms"] == 20_480
    assert audit["active_unique_streams"] == 63_488
    assert audit["checks"]["teacher_behavior_cloning_stream_rows_exact"]
    assert audit["checks"]["distillation_validation_stream_rows_exact"]


def test_authority_is_not_dictionary_insertion_order_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = surface.expected_active_rows()
    reversed_rows = []
    for row in rows:
        changed = copy.deepcopy(row)
        changed["streams"] = dict(reversed(list(changed["streams"].items())))
        reversed_rows.append(changed)
    monkeypatch.setattr(surface, "expected_active_rows", lambda: reversed_rows)
    assert surface.authority_audit()["passes"]


def test_execution_schema_and_model_are_exact_no_aux() -> None:
    schema = surface.execution_schema()
    assert j2.parameter_count() == 410_117
    assert schema["parent_model_schema"]["auxiliary_heads"] == []
    assert schema["active_authority"]["unique_streams"] == 63_488
    assert schema["bootstrap"] == {
        "replicates": 4_096,
        "score_seed": 2_026_072_831,
        "progression_seed": 2_026_072_832,
        "score_method": "global paired-root resampling",
        "progression_method": (
            "independent within-stratum whole-root resampling"
        ),
        "strata": 8,
        "pairs_per_stratum": 768,
        "quantiles": [0.025, 0.975],
        "quantile_method": "linear",
    }


def test_parser_exposes_only_frozen_commands() -> None:
    choices = surface.build_parser()._subparsers._group_actions[0].choices
    assert set(choices) == {
        "audit-zero-work",
        "write-test-evidence",
        "prepare-readiness",
        "seal-phase-lock",
        "open",
        "materialize",
        "execute",
    }
    assert not {"development", "confirmation", "promote"} & set(choices)


def test_dispatch_rejects_unknown_command() -> None:
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.dispatch_cli(argparse.Namespace(subcommand="development"))


def test_runtime_storage_projection_is_bounded_and_uses_margin() -> None:
    report = surface.runtime_storage_projection()
    assert report["passes"]
    central = report["central_512_moves"]
    assert central["peak_before_margin_bytes"] == 17_535_295_488
    assert central["peak_after_25pct_margin_bytes"] == 21_919_119_360
    assert central["teacher_runtime_hours_after_margin"] == pytest.approx(
        41.98247721555496
    )
    assert report["sensitivity_5000_moves"]["diagnostic_not_conjunctive"]


def test_teacher_root_value_targets_telescope_exactly() -> None:
    row = BC_ROWS[0]
    record = _raw_root(
        row,
        families=("low_air", "mid_progression", "upper_progression"),
        deltas=(3, 6, 12),
    )
    normalized = surface.validate_teacher_root_record(
        record,
        authoritative_row=row,
    )
    assert [row["value_target"] for row in normalized["transitions"]] == [
        pytest.approx(21e-5),
        pytest.approx(18e-5),
        pytest.approx(12e-5),
    ]


def test_illegal_teacher_action_fails_closed() -> None:
    row = BC_ROWS[0]
    record = _raw_root(row)
    record["transitions"][0]["legal_mask"] = np.asarray(
        [False, True, False, False]
    )
    record["root_content_sha256"] = surface.teacher_root_content_hash(record)
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.validate_teacher_root_record(record, authoritative_row=row)


def test_teacher_root_wrong_authority_fails_closed() -> None:
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.validate_teacher_root_record(
            _raw_root(BC_ROWS[0]),
            authoritative_row=BC_ROWS[1],
        )


def test_teacher_root_blob_reload_publishes_authenticated_targets(
    tmp_path: Path,
) -> None:
    row = BC_ROWS[0]
    path = tmp_path / "root.bin"
    identity = surface.write_teacher_root_blob(
        path,
        _raw_root(row, deltas=(3, 6)),
        authoritative_row=row,
    )
    loaded = surface.load_teacher_root_blob(
        path,
        authoritative_row=row,
        expected_file_sha256=identity["file_sha256"],
    )
    assert loaded["transitions"][0]["value_target"] == pytest.approx(9e-5)
    batch = surface.build_distillation_batch(
        [loaded],
        expected_root_count=1,
    )
    assert batch.value_targets.tolist() == pytest.approx([9e-5, 6e-5])


def test_teacher_root_blob_is_create_once(tmp_path: Path) -> None:
    row = BC_ROWS[0]
    path = tmp_path / "root.bin"
    first = surface.write_teacher_root_blob(
        path,
        _raw_root(row),
        authoritative_row=row,
    )
    second = surface.write_teacher_root_blob(
        path,
        _raw_root(row),
        authoritative_row=row,
    )
    assert first == second
    changed = _raw_root(row, deltas=(6, 6))
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.write_teacher_root_blob(
            path,
            changed,
            authoritative_row=row,
        )


def test_distillation_batch_root_equal_for_unequal_lengths() -> None:
    roots = [
        _normalized_root(BC_ROWS[0], families=("low_air",)),
        _normalized_root(
            BC_ROWS[1],
            families=(
                "low_air",
                "low_air",
                "mid_progression",
                "upper_progression",
                "low_constrained",
                "low_air",
                "mid_progression",
            ),
        ),
    ]
    batch = surface.build_distillation_batch(
        roots,
        expected_root_count=2,
    )
    totals: dict[str, float] = {}
    for root_id, weight in zip(batch.root_ids, batch.row_weights.tolist()):
        totals[root_id] = totals.get(root_id, 0.0) + weight
    assert totals == pytest.approx(
        {
            BC_ROWS[0]["root_id"]: 1.0,
            BC_ROWS[1]["root_id"]: 1.0,
        },
        abs=2e-7,
    )


def _balanced_family_roots() -> list[dict[str, object]]:
    families = tuple(
        family
        for family in surface.FEATURE_FAMILIES
        for _ in range(4)
    )
    return [
        {
            "root_id": row["root_id"],
            "transitions": [
                {
                    "transition_index": index,
                    "board": _board(family),
                }
                for index, family in enumerate(families)
            ],
        }
        for row in VALIDATION_ROWS[:256]
    ]


def test_strict_family_inventory_passes_exact_support_floor() -> None:
    report = surface.strict_feature_inventory(
        _balanced_family_roots(),
        expected_root_count=256,
    )
    assert report["passes"]
    assert report["natural_family_counts"] == {
        family: 1_024 for family in surface.FEATURE_FAMILIES
    }
    assert report["natural_family_root_counts"] == {
        family: 256 for family in surface.FEATURE_FAMILIES
    }
    assert set(report["trajectory_quartile_counts"]) == {
        "q1",
        "q2",
        "q3",
        "q4",
    }


def test_strict_family_inventory_holds_on_state_shortfall() -> None:
    roots = _balanced_family_roots()
    for root in roots:
        root["transitions"] = root["transitions"][:-1]
    report = surface.strict_feature_inventory(
        roots,
        expected_root_count=256,
    )
    assert not report["passes"]
    assert not report["checks"]["minimum_1024_states_each"]


def test_strict_family_inventory_holds_on_natural_concentration() -> None:
    roots = _balanced_family_roots()
    for root in roots:
        start = len(root["transitions"])
        root["transitions"].extend(
            {
                "transition_index": start + index,
                "board": _board("low_air"),
            }
            for index in range(48)
        )
    report = surface.strict_feature_inventory(
        roots,
        expected_root_count=256,
    )
    assert not report["passes"]
    assert not report["checks"]["natural_max_share_strictly_below_070"]
    assert report["checks"]["capped_max_share_strictly_below_040"]


@pytest.mark.parametrize("mode", ["duplicate", "missing", "cross_shard"])
def test_teacher_worker_result_integrity_fails_closed(mode: str) -> None:
    rows = [BC_ROWS[0], BC_ROWS[1]]
    results = [
        {
            "kind": "root",
            "worker_id": int(row["row_index"]) % surface.SHARDS,
            "root_id": row["root_id"],
            "record": _raw_root(row),
        }
        for row in rows
    ]
    if mode == "duplicate":
        results[1] = copy.deepcopy(results[0])
    elif mode == "missing":
        results.pop()
    else:
        results[0]["worker_id"] = 7
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.validate_teacher_worker_results(results, rows)


def test_teacher_worker_results_merge_in_authority_order() -> None:
    rows = [BC_ROWS[0], BC_ROWS[1]]
    results = [
        {
            "kind": "root",
            "worker_id": int(row["row_index"]) % surface.SHARDS,
            "root_id": row["root_id"],
            "record": _raw_root(row),
        }
        for row in reversed(rows)
    ]
    merged = surface.validate_teacher_worker_results(results, rows)
    assert [row["root_id"] for row in merged] == [
        row["root_id"] for row in rows
    ]


def test_attempt_ledger_charges_abandoned_work_by_unit_bound(
    tmp_path: Path,
) -> None:
    ledger = _attempt_ledger(tmp_path)
    ledger.begin(unit_id="root-a", unit_type="teacher_root")
    recovered = _attempt_ledger(tmp_path)
    summary = recovered.summary()
    assert summary["attempts_started"] == 1
    assert summary["attempts_abandoned"] == 1
    assert summary["active_seconds"] == pytest.approx(
        surface.ABANDONED_UNIT_CHARGE_SECONDS["teacher_root"]
    )


def test_completion_ledger_rejects_duplicate_root(tmp_path: Path) -> None:
    ledger = surface.CompletionLedger(
        path=tmp_path / "completions.jsonl",
        contract_sha256="b" * 64,
        kind="teacher_root",
    )
    row = BC_ROWS[0]
    kwargs = {
        "root_id": row["root_id"],
        "ancestry_id": row["ancestry_id"],
        "row_index": row["row_index"],
        "stage": row["stage"],
        "relative_path": "root.bin",
        "file_sha256": "c" * 64,
        "content_sha256": "d" * 64,
        "recovered_orphan": False,
    }
    ledger.append(**kwargs)
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        ledger.append(**kwargs)


def test_bounded_teacher_collection_resumes_without_duplicate(
    tmp_path: Path,
) -> None:
    rows = [BC_ROWS[0], BC_ROWS[1]]
    contract = "e" * 64
    with pytest.raises(surface.J2A1PlannedInterruption):
        surface.bounded_collect_teacher_roots(
            phase_dir=tmp_path,
            rows=rows,
            contract_sha256=contract,
            batch_collector=_fixture_collector,
            attempt_ledger=_attempt_ledger(tmp_path, contract=contract),
            interrupt_after_completed=1,
        )
    result = surface.bounded_collect_teacher_roots(
        phase_dir=tmp_path,
        rows=rows,
        contract_sha256=contract,
        batch_collector=_fixture_collector,
        attempt_ledger=_attempt_ledger(tmp_path, contract=contract),
    )
    assert result["ledger"]["completed"] == 2
    assert len({row["root_id"] for row in result["refs"]}) == 2
    attempts = _attempt_ledger(tmp_path, contract=contract).summary()
    assert attempts["attempts_abandoned"] == 1


@pytest.mark.parametrize("crash_stage", ["after_slot", "after_journal"])
def test_compact_updater_store_recovers_crash_sides(
    tmp_path: Path,
    crash_stage: str,
) -> None:
    directory = tmp_path / crash_stage
    store = surface.CompactUpdaterStore(
        directory=directory,
        contract_sha256="f" * 64,
    )
    with pytest.raises(RuntimeError):
        store.append(
            unit_id="unit-0",
            snapshot_bytes=b"snapshot",
            cursor=0,
            batch_identity="1" * 64,
            crash_stage=crash_stage,
        )
    if crash_stage == "after_slot":
        assert surface.CompactUpdaterStore(
            directory=directory,
            contract_sha256="f" * 64,
        ).current is None
    else:
        recovered = surface.CompactUpdaterStore(
            directory=directory,
            contract_sha256="f" * 64,
        )
        assert recovered.current["snapshot_bytes"] == b"snapshot"


def _run_mini_engine(
    directory: Path,
    *,
    interrupt_after_teacher_roots: int | None = None,
    interrupt_after_optimizer_steps: int | None = None,
    interrupt_before_optimizer_commit: int | None = None,
    interrupt_after_pairs: int | None = None,
) -> dict[str, object]:
    contract = "9" * 64
    return surface.execute_distillation_fidelity_engine(
        phase_dir=directory,
        rows=_mini_rows(),
        contract_sha256=contract,
        attempt_ledger=_attempt_ledger(directory, contract=contract),
        config=_mini_config(),
        batch_collector=_fixture_collector,
        arm_runner=_fixture_arm_runner,
        family_inventory_fn=_passing_inventory,
        mechanism_fn=_passing_mechanism,
        fidelity_fn=_passing_fidelity,
        interrupt_after_teacher_roots=interrupt_after_teacher_roots,
        interrupt_after_optimizer_steps=interrupt_after_optimizer_steps,
        interrupt_before_optimizer_commit=interrupt_before_optimizer_commit,
        interrupt_after_pairs=interrupt_after_pairs,
    )


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("teacher", 1),
        ("optimizer_after", 1),
        ("optimizer_before", 1),
        ("pair", 1),
    ],
)
def test_bounded_engine_interrupted_resume_matches_uninterrupted(
    tmp_path: Path,
    kind: str,
    value: int,
) -> None:
    uninterrupted = _run_mini_engine(tmp_path / "full")
    interrupted_dir = tmp_path / kind
    kwargs = {
        "interrupt_after_teacher_roots": None,
        "interrupt_after_optimizer_steps": None,
        "interrupt_before_optimizer_commit": None,
        "interrupt_after_pairs": None,
    }
    mapping = {
        "teacher": "interrupt_after_teacher_roots",
        "optimizer_after": "interrupt_after_optimizer_steps",
        "optimizer_before": "interrupt_before_optimizer_commit",
        "pair": "interrupt_after_pairs",
    }
    kwargs[mapping[kind]] = value
    with pytest.raises(surface.J2A1PlannedInterruption):
        _run_mini_engine(interrupted_dir, **kwargs)
    resumed = _run_mini_engine(interrupted_dir)
    assert resumed["decision"] == surface.READY_EXECUTION
    assert resumed["checkpoint"]["file_sha256"] == uninterrupted[
        "checkpoint"
    ]["file_sha256"]
    assert resumed["distillation"]["closed_step_ids_sha256"] == (
        uninterrupted["distillation"]["closed_step_ids_sha256"]
    )
    assert resumed["panel"]["pair_refs_sha256"] == uninterrupted[
        "panel"
    ]["pair_refs_sha256"]


def test_pair_record_and_completion_barrier(tmp_path: Path) -> None:
    row = VALIDATION_ROWS[0]
    root_path = tmp_path / "teacher.bin"
    identity = surface.write_teacher_root_blob(
        root_path,
        _raw_root(row),
        authoritative_row=row,
    )
    root = surface.load_teacher_root_blob(
        root_path,
        authoritative_row=row,
    )
    root["file_sha256"] = identity["file_sha256"]
    record = surface.seal_pair_record(
        row=row,
        teacher_root=root,
        student_arm=_fixture_arm_runner([row], j2.J2ActorCritic())[0],
        student_checkpoint_sha256="a" * 64,
    )
    assert surface.validate_complete_pair_records(
        [record],
        [row],
        expected_pairs=1,
    )[0]["student"]["final_score"] == 12
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.validate_complete_pair_records(
            [],
            [row],
            expected_pairs=1,
        )


def test_score_estimand_clamps_negative_score_minus_start() -> None:
    row = VALIDATION_ROWS[0]
    teacher = {
        "start_score": 2,
        "final_score": 2,
        "max_tile": 1536,
        "moves": 1,
        "latency_seconds": 0.1,
        "survival": 1.0,
        "illegal_actions": 0,
        "arm_file_sha256": "a" * 64,
    }
    student = {
        **teacher,
        "start_score": 5,
        "final_score": 5,
        "arm_file_sha256": "b" * 64,
    }
    pair = {
        "root_id": row["root_id"],
        "row": row,
        "pair_complete": True,
        "student": student,
        "teacher": teacher,
    }
    normalized = surface.validate_complete_pair_records(
        [pair],
        [row],
        expected_pairs=1,
    )
    difference = math.log1p(
        max(
            normalized[0]["student"]["final_score"]
            - normalized[0]["student"]["start_score"],
            0,
        )
    )
    assert difference == 0.0


def test_mh_correction_is_finite_at_zero_edges() -> None:
    totals = np.full((1, 8), 4.0)
    zero_numerator = surface._mh_log_or(
        np.zeros((1, 8)),
        np.ones((1, 8)),
        totals,
    )
    zero_denominator = surface._mh_log_or(
        np.full((1, 8), 4.0),
        np.zeros((1, 8)),
        totals,
    )
    assert np.isfinite(zero_numerator).all()
    assert np.isfinite(zero_denominator).all()


def test_bootstrap_contract_is_deterministic_and_fixed_stratum() -> None:
    student = np.tile(np.asarray([0, 1, 0, 1]), 8)
    teacher = np.tile(np.asarray([0, 0, 1, 0]), 8)
    strata = np.repeat(np.arange(8), 4)
    first = surface._progression_bootstrap_bounds(
        student,
        teacher,
        strata,
        replicates=31,
        seed=surface.PROGRESSION_BOOTSTRAP_SEED,
    )
    second = surface._progression_bootstrap_bounds(
        student,
        teacher,
        strata,
        replicates=31,
        seed=surface.PROGRESSION_BOOTSTRAP_SEED,
    )
    assert first == second


def test_retirement_is_idempotent_after_manifest_and_partial_delete(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "a.bin", tmp_path / "b.bin"]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]) * 4)
    with pytest.raises(surface.J2A1PlannedInterruption):
        surface.seal_retirement(
            phase_dir=tmp_path,
            name="fixture",
            paths=paths,
            predecessor_sha256="4" * 64,
            crash_after_manifest=True,
        )
    with pytest.raises(surface.J2A1PlannedInterruption):
        surface.seal_retirement(
            phase_dir=tmp_path,
            name="fixture",
            paths=paths,
            predecessor_sha256="4" * 64,
            crash_after_deletions=1,
        )
    result = surface.seal_retirement(
        phase_dir=tmp_path,
        name="fixture",
        paths=paths,
        predecessor_sha256="4" * 64,
    )
    assert result["all_sources_absent"]
    assert not any(path.exists() for path in paths)


def test_scientific_engine_rejects_gate_injection(tmp_path: Path) -> None:
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.execute_distillation_fidelity_engine(
            phase_dir=tmp_path,
            rows=_mini_rows(),
            contract_sha256="5" * 64,
            attempt_ledger=_attempt_ledger(tmp_path, contract="5" * 64),
            config=surface.EngineConfig(execution_mode="scientific"),
            batch_collector=_fixture_collector,
            arm_runner=_fixture_arm_runner,
            family_inventory_fn=_passing_inventory,
        )


def test_checkpoint_authority_requires_ready_engine(tmp_path: Path) -> None:
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface._authorize_checkpoint_after_final_guard(
            phase_dir=tmp_path,
            engine={
                "decision": surface.HOLD_MECHANISM,
                "passes": False,
                "checkpoint_authoritative": False,
            },
            operational_guard_identity={"file_sha256": "a" * 64},
            execution_mode="scientific",
        )


def test_terminal_fixture_cannot_gain_scientific_authority() -> None:
    terminal = surface._terminal_base(
        chain={
            "lock": {"identity": {"file_sha256": "a" * 64}},
            "marker_identity": {"file_sha256": "b" * 64},
            "manifest_identity": {"file_sha256": "c" * 64},
        },
        engine={
            "decision": surface.READY_EXECUTION,
            "checkpoint_authoritative": True,
        },
        attempt_summary={"attempts_started": 0},
        reservation={"file_sha256": "d" * 64},
        consumption={"file_sha256": "e" * 64},
        ownership=type(
            "FixtureOwnership",
            (),
            {
                "records": [
                    {"owner_record_sha256": "f" * 64},
                ]
            },
        )(),
        execution_mode="miniature_fixture",
    )
    assert not terminal["scientific_authority"]
    assert not terminal["checkpoint_authoritative"]
    assert not terminal["successor_review_authority"]
    assert not terminal["continue"]


def test_test_evidence_is_create_once(tmp_path: Path) -> None:
    commands = [{"command": "focused", "passed": 1, "failed": 0}]
    first = surface.write_test_evidence(
        output_dir=tmp_path,
        commands=commands,
        deselections=[],
    )
    assert first["passes"]
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.write_test_evidence(
            output_dir=tmp_path,
            commands=commands,
            deselections=[],
        )


def test_zero_work_audit_requires_future_execution_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future = tmp_path / "future"
    monkeypatch.setattr(surface, "FUTURE_EXECUTION_DIR", future)
    assert surface.audit_zero_work(
        output_dir=tmp_path / "readiness",
        include_operational=False,
    )["passes"]
    future.mkdir()
    assert not surface.audit_zero_work(
        output_dir=tmp_path / "readiness",
        include_operational=False,
    )["passes"]


def test_operational_audit_configures_runtime_and_accepts_tuple_top_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[bool] = []
    monkeypatch.setattr(
        surface,
        "configure_deterministic_runtime",
        lambda: configured.append(True),
    )
    monkeypatch.setattr(
        surface.j2,
        "operational_audit",
        lambda *, output_dir: {
            "passes": True,
            "human_session_content_read": False,
            "torch_runtime": {
                "intra_op_threads": 1,
                "inter_op_threads": 1,
                "deterministic_algorithms": True,
            },
            "parent_operational": {
                "nice": 10,
                "free_disk_gib": 140.0,
                "services": {
                    "passes": True,
                    "dashboard": {
                        "top_three": (263670, 261369, 258561),
                    },
                },
                "process": {"passes": True},
            },
        },
    )
    audit = surface.operational_audit(
        output_dir=tmp_path,
        include_services=True,
    )
    assert configured == [True]
    assert audit["checks"]["dashboard_top_three_exact"]
    assert audit["passes"]


def test_final_terminal_is_written_after_evidence_and_retention(
    tmp_path: Path,
) -> None:
    accountant = surface.OutputAccountant(tmp_path)
    payload = {
        "version": "fixture_terminal",
        "decision": surface.HOLD_FAMILY,
        "checkpoint_authoritative": False,
        "human_session_reads": 0,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
    }
    result = surface._seal_terminal_and_retention(
        out_dir=tmp_path,
        terminal_payload=payload,
        accountant=accountant,
    )
    terminal = result["terminal"]
    assert terminal["authoritative_terminal_written_last"]
    assert surface.verify_payload_hash(
        terminal,
        "terminal_payload_sha256",
    )
    assert result["post_terminal_checks_performed"] == 0


def _prepare_fixture_chain(
    base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    readiness = base / "readiness"
    execution = base / "execution"
    authorization = base / "authorization.json"
    future = base / "future_scientific_execution"
    monkeypatch.setattr(surface, "FUTURE_EXECUTION_DIR", future)
    monkeypatch.setattr(
        surface,
        "configure_deterministic_runtime",
        lambda: {
            "torch_version": torch.__version__,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "deterministic_algorithms": True,
            "checks": {
                "torch_intra_one": True,
                "torch_inter_one": True,
                "torch_deterministic": True,
            },
            "passes": True,
        },
    )
    surface.write_test_evidence(
        output_dir=readiness,
        commands=[
            {
                "command": "miniature phase-chain fixture",
                "passed": 1,
                "failed": 0,
            }
        ],
        deselections=[],
    )
    result = surface.prepare_readiness(
        output_dir=readiness,
        include_operational=False,
    )
    assert result["decision"] == surface.READY
    surface.write_immutable_json(
        authorization,
        surface.authorization_payload(
            readiness_dir=readiness,
            out_dir=execution,
            execution_mode="miniature_fixture",
            scientific_authority=False,
        ),
        field="authorization_payload_sha256",
    )
    surface.seal_phase_lock(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=execution,
        jobs=1,
        execution_mode="miniature_fixture",
        include_operational=False,
    )
    surface.open_phase(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=execution,
        execution_mode="miniature_fixture",
        include_operational=False,
    )
    surface.materialize_phase(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=execution,
        execution_mode="miniature_fixture",
        include_operational=False,
        rows_override=_mini_rows(),
    )
    return readiness, authorization, execution


def _execute_fixture_chain(
    *,
    readiness: Path,
    authorization: Path,
    execution: Path,
    interrupt_stage: str | None = None,
    owner_pid: int | None = None,
    owner_start_identity: str | None = None,
) -> dict[str, object]:
    return surface.execute_phase_from_artifacts(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=execution,
        execution_mode="miniature_fixture",
        include_operational=False,
        command="miniature bounded phase command",
        jobs=1,
        config=_mini_config(),
        batch_collector=_fixture_collector,
        arm_runner=_fixture_arm_runner,
        family_inventory_fn=_passing_inventory,
        mechanism_fn=_passing_mechanism,
        fidelity_fn=_passing_fidelity,
        interrupt_stage=interrupt_stage,
        owner_pid=owner_pid,
        owner_start_identity=owner_start_identity,
    )


def test_phase_chain_create_once_and_exact_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness, authorization, execution = _prepare_fixture_chain(
        tmp_path,
        monkeypatch,
    )
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.open_phase(
            readiness_dir=readiness,
            authorization_path=authorization,
            out_dir=execution,
            execution_mode="miniature_fixture",
            include_operational=False,
        )
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.materialize_phase(
            readiness_dir=readiness,
            authorization_path=authorization,
            out_dir=execution,
            execution_mode="miniature_fixture",
            include_operational=False,
            rows_override=[*_mini_rows(), copy.deepcopy(_mini_rows()[0])],
        )


def test_marker_tamper_is_rejected_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = tmp_path / "readiness"
    execution = tmp_path / "execution"
    authorization = tmp_path / "authorization.json"
    monkeypatch.setattr(
        surface,
        "FUTURE_EXECUTION_DIR",
        tmp_path / "future",
    )
    monkeypatch.setattr(
        surface,
        "configure_deterministic_runtime",
        lambda: {
            "passes": True,
            "checks": {},
            "torch_version": torch.__version__,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "deterministic_algorithms": True,
        },
    )
    surface.write_test_evidence(
        output_dir=readiness,
        commands=[{"command": "fixture", "passed": 1, "failed": 0}],
        deselections=[],
    )
    surface.prepare_readiness(
        output_dir=readiness,
        include_operational=False,
    )
    surface.write_immutable_json(
        authorization,
        surface.authorization_payload(
            readiness_dir=readiness,
            out_dir=execution,
            execution_mode="miniature_fixture",
            scientific_authority=False,
        ),
        field="authorization_payload_sha256",
    )
    surface.seal_phase_lock(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=execution,
        jobs=1,
        execution_mode="miniature_fixture",
        include_operational=False,
    )
    surface.open_phase(
        readiness_dir=readiness,
        authorization_path=authorization,
        out_dir=execution,
        execution_mode="miniature_fixture",
        include_operational=False,
    )
    marker = execution / surface.OPEN_MARKER_NAME
    body = json.loads(marker.read_text())
    body["streams_reserved"] = 1
    marker.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.materialize_phase(
            readiness_dir=readiness,
            authorization_path=authorization,
            out_dir=execution,
            execution_mode="miniature_fixture",
            include_operational=False,
            rows_override=_mini_rows(),
        )


def test_dispatcher_miniature_chain_reaches_non_authoritative_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness, authorization, execution = _prepare_fixture_chain(
        tmp_path,
        monkeypatch,
    )
    result = _execute_fixture_chain(
        readiness=readiness,
        authorization=authorization,
        execution=execution,
        owner_pid=os.getpid(),
        owner_start_identity="fixture-process",
    )
    terminal = result["terminal"]
    assert terminal["decision"] == surface.READY_EXECUTION
    assert terminal["execution_mode"] == "miniature_fixture"
    assert not terminal["checkpoint_authoritative"]
    assert not terminal["successor_review_authority"]
    assert not terminal["continue"]
    assert result["retention"]["passes"]
    assert not (execution / "development").exists()
    assert not (execution / "confirmation").exists()


def test_post_consumption_dead_owner_recovery_reuses_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness, authorization, execution = _prepare_fixture_chain(
        tmp_path,
        monkeypatch,
    )
    dead_pid = 999_999
    with pytest.raises(surface.J2A1PlannedInterruption):
        _execute_fixture_chain(
            readiness=readiness,
            authorization=authorization,
            execution=execution,
            interrupt_stage="after_consumption",
            owner_pid=dead_pid,
            owner_start_identity="dead-fixture-owner",
        )
    consumption_path = execution / surface.CONSUMPTION_NAME
    before = consumption_path.read_bytes()
    result = _execute_fixture_chain(
        readiness=readiness,
        authorization=authorization,
        execution=execution,
        owner_pid=os.getpid(),
        owner_start_identity="recovered-fixture-owner",
    )
    assert result["terminal"]["decision"] == surface.READY_EXECUTION
    assert consumption_path.read_bytes() == before
    ownership = surface.OwnershipLedger(
        path=execution / "ownership_ledger.jsonl",
        contract_sha256=surface._phase_contract_hash(
            lock={
                "file_sha256": result["terminal"]["phase_lock"][
                    "file_sha256"
                ],
                "payload_sha256": result["terminal"]["phase_lock"][
                    "payload_sha256"
                ],
            },
            marker={
                "file_sha256": result["terminal"]["marker"]["file_sha256"],
                "payload_sha256": result["terminal"]["marker"][
                    "payload_sha256"
                ],
            },
            manifest={
                "file_sha256": result["terminal"]["manifest"][
                    "file_sha256"
                ],
                "payload_sha256": result["terminal"]["manifest"][
                    "payload_sha256"
                ],
            },
            command="miniature bounded phase command",
            execution_mode="miniature_fixture",
        ),
    )
    assert [row["kind"] for row in ownership.records] == [
        "owner",
        "recovery",
        "owner",
    ]


def test_final_operational_fault_cannot_survive_as_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness, authorization, execution = _prepare_fixture_chain(
        tmp_path,
        monkeypatch,
    )
    original = surface.execution_operational_guard

    def fail_after_reconcile(**kwargs: object) -> dict[str, object]:
        accountant = kwargs["accountant"]
        if accountant.full_scan_count >= 2:
            raise surface.J2A1ExecutionOperationalHold(
                "fixture final storage fault"
            )
        return original(**kwargs)

    monkeypatch.setattr(
        surface,
        "execution_operational_guard",
        fail_after_reconcile,
    )
    result = _execute_fixture_chain(
        readiness=readiness,
        authorization=authorization,
        execution=execution,
        owner_pid=os.getpid(),
        owner_start_identity="fixture-process",
    )
    assert result["terminal"]["decision"] == surface.HOLD_OPERATIONAL
    assert not result["terminal"]["checkpoint_authoritative"]
    assert not result["terminal"]["successor_review_authority"]
    assert not (
        execution / "J2A1_DISTILLED_CHECKPOINT_AUTHORITY.json"
    ).exists()
    assert (execution / "J2A1_CHECKPOINT_QUARANTINE.json").is_file()


def test_terminal_cap_failure_occurs_before_authoritative_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(surface, "STORAGE_CAP_BYTES", 32)
    with pytest.raises(surface.J2A1ExecutionOperationalHold):
        surface._seal_terminal_and_retention(
            out_dir=tmp_path,
            terminal_payload={
                "checkpoint_authoritative": False,
                "human_session_reads": 0,
                "incumbent_changes": 0,
                "dashboard_changes": 0,
            },
            accountant=surface.OutputAccountant(tmp_path),
        )
    assert not (tmp_path / surface.TERMINAL_NAME).exists()


def test_family_hold_prevents_optimizer_and_student_arm(tmp_path: Path) -> None:
    arm_calls = 0

    def forbidden_arm(*_args: object, **_kwargs: object) -> object:
        nonlocal arm_calls
        arm_calls += 1
        raise AssertionError("student arm opened after family HOLD")

    result = surface.execute_distillation_fidelity_engine(
        phase_dir=tmp_path,
        rows=_mini_rows(),
        contract_sha256="6" * 64,
        attempt_ledger=_attempt_ledger(tmp_path, contract="6" * 64),
        config=_mini_config(),
        batch_collector=_fixture_collector,
        arm_runner=forbidden_arm,
        family_inventory_fn=lambda _roots, count: {
            "version": "fixture_family_hold",
            "validation_root_count": count,
            "checks": {"support": False},
            "passes": False,
            "decision": surface.HOLD_FAMILY,
        },
    )
    assert result["decision"] == surface.HOLD_FAMILY
    assert result["checkpoint"] is None
    assert arm_calls == 0
    assert not (tmp_path / "optimizer_resume").exists()


def test_mechanism_hold_quarantines_checkpoint_and_skips_pairs(
    tmp_path: Path,
) -> None:
    arm_calls = 0

    def forbidden_arm(*_args: object, **_kwargs: object) -> object:
        nonlocal arm_calls
        arm_calls += 1
        raise AssertionError("student arm opened after mechanism HOLD")

    result = surface.execute_distillation_fidelity_engine(
        phase_dir=tmp_path,
        rows=_mini_rows(),
        contract_sha256="7" * 64,
        attempt_ledger=_attempt_ledger(tmp_path, contract="7" * 64),
        config=_mini_config(),
        batch_collector=_fixture_collector,
        arm_runner=forbidden_arm,
        family_inventory_fn=_passing_inventory,
        mechanism_fn=lambda *_args: {
            "version": "fixture_mechanism_hold",
            "checks": {"accuracy": False},
            "passes": False,
            "decision": surface.HOLD_MECHANISM,
        },
    )
    assert result["decision"] == surface.HOLD_MECHANISM
    assert not result["checkpoint_authoritative"]
    assert result["quarantine"]["authoritative"] is False
    assert arm_calls == 0
    assert not (tmp_path / "fidelity_pairs").exists()


def test_fidelity_outcomes_cannot_open_before_complete_seal(
    tmp_path: Path,
) -> None:
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        surface.load_fidelity_pairs_after_seal(
            phase_dir=tmp_path,
            panel={
                "seal": {
                    "passes": True,
                    "partial_outcome_reads": 0,
                },
                "refs": [],
            },
            rows=[VALIDATION_ROWS[0]],
        )


def test_checkpoint_corruption_fails_closed(tmp_path: Path) -> None:
    result = _run_mini_engine(tmp_path)
    path = Path(result["checkpoint"]["path"])
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(
        (
            surface.J2A1ExecutionIntegrityError,
            RuntimeError,
            EOFError,
        )
    ):
        surface._deserialize_torch_payload(
            path.read_bytes(),
            magic=surface.CHECKPOINT_MAGIC,
        )


def test_worker_receive_distinguishes_dead_worker_and_timeout() -> None:
    class Process:
        def __init__(self, alive: bool, pid: int) -> None:
            self._alive = alive
            self.pid = pid

        def is_alive(self) -> bool:
            return self._alive

    dead = object.__new__(surface.TeacherRootWorkerGroup)
    dead.timeout_seconds = 0.01
    dead.result_queue = queue.Queue()
    dead.processes = [Process(False, 999_999)]
    with pytest.raises(surface.J2A1ExecutionIntegrityError):
        dead._receive(1, kind="root")

    live = object.__new__(surface.TeacherRootWorkerGroup)
    live.timeout_seconds = 0.01
    live.result_queue = queue.Queue()
    live.processes = [Process(True, os.getpid())]
    with pytest.raises(surface.J2A1ExecutionOperationalHold):
        live._receive(1, kind="root")


def test_source_and_authority_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = dict(surface.EXPECTED_SOURCE_HASHES)
    key = next(iter(changed))
    changed[key] = "0" * 64
    monkeypatch.setattr(surface, "EXPECTED_SOURCE_HASHES", changed)
    report = surface.source_and_parent_audit(
        require_future_execution_absent=True,
    )
    assert not report["passes"]
    assert not report["checks"]["all_parent_sources_exact"]


def test_batched_student_full_policy_is_deterministic_except_timing() -> None:
    model, _optimizer = j2.initialize_model_optimizer()
    for parameter in model.parameters():
        parameter.data.zero_()
    rows = [VALIDATION_ROWS[0], VALIDATION_ROWS[1]]
    first = surface.run_student_arms_synchronously(rows=rows, model=model)
    second = surface.run_student_arms_synchronously(rows=rows, model=model)
    projected = lambda arm: {
        key: arm[key]
        for key in (
            "start_score",
            "final_score",
            "max_tile",
            "moves",
            "survival",
            "illegal_actions",
            "policy_stream_id",
        )
    }
    assert [projected(arm) for arm in first] == [
        projected(arm) for arm in second
    ]
    assert all(arm["illegal_actions"] == 0 for arm in first)
    assert [arm["policy_stream_id"] for arm in first] == [
        row["streams"]["student_policy_stream_id"] for row in rows
    ]

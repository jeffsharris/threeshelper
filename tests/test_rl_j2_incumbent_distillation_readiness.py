from __future__ import annotations

import ast
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from threes_rl import j2_incumbent_distillation_readiness as j2
from threes_rl import o2_online_option_preflight as o2


def test_stage_table_derives_every_frozen_total() -> None:
    assert j2.derive_stage_totals() == {
        "prospective_rows_or_pairs": 32_000,
        "game_arms": 39_424,
        "unique_streams": 135_424,
        "pre_ppo_teacher_roots": 10_240,
        "online_teacher_roots": 4_096,
        "total_teacher_root_equivalents": 14_336,
    }
    assert j2.TOTAL_UNIQUE_STREAMS == (
        4 * 8_192
        + 5 * 2_048
        + 4 * 16_384
        + 5 * 896
        + 5 * 4_480
    )


def test_stage_table_drift_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = [dict(row) for row in j2.STAGE_TABLE]
    changed[0]["authority_rows"] = 8_193
    monkeypatch.setattr(j2, "STAGE_TABLE", tuple(changed))
    assert j2.derive_stage_totals() != j2.EXPECTED_STAGE_TOTALS
    assert not j2.prospective_authority()["checks"][
        "stage_table_totals_exact"
    ]


def test_prospective_authority_exact_and_collision_free() -> None:
    report = j2.prospective_authority()
    assert report["passes"]
    assert report["row_count"] == 32_000
    assert report["unique_stream_count"] == 135_424
    assert report["stage_counts"] == {
        "teacher_behavior_cloning": 8_192,
        "distillation_validation": 2_048,
        "on_policy_training": 16_384,
        "development": 896,
        "confirmation": 4_480,
    }
    assert report["checks"]["no_213b_226b_collision"]
    assert report["zero_work"]["streams_reserved"] == 0
    assert report["zero_work"]["streams_consumed"] == 0


def test_pair_stream_semantics_and_all_streams_unique() -> None:
    rows = j2.build_prospective_rows()
    validation = [
        row for row in rows if row["stage"] == "distillation_validation"
    ]
    assert len(validation) == 2_048
    first = validation[0]["streams"]
    assert set(first) == {
        "logical_stream_id",
        "deck_stream_id",
        "slot_stream_id",
        "student_policy_stream_id",
        "teacher_policy_stream_id",
    }
    assert first["student_policy_stream_id"] != first[
        "teacher_policy_stream_id"
    ]
    values = [
        value for row in rows for value in row["streams"].values()
    ]
    assert len(values) == len(set(values)) == 135_424


def test_protected_stream_authority_uses_compact_manifests() -> None:
    report = j2.protected_stream_authority()
    assert report["passes"]
    assert report["denied_namespace_prefixes"] == list(range(213, 227))
    assert report["checks"]["no_global_payload_parser"]
    assert report["streams_reserved"] == 0
    assert report["streams_consumed"] == 0


def test_model_schema_has_exact_no_aux_parameter_count() -> None:
    model = j2.J2ActorCritic()
    schema = j2.model_schema()
    assert j2.parameter_count(model) == 410_117
    assert schema["auxiliary_heads"] == []
    assert schema["auxiliary_losses"] == []
    assert set(schema["heads"]) == {"policy", "value"}


def test_masked_logits_fail_closed_and_never_choose_illegal() -> None:
    logits = torch.tensor([[100.0, 2.0, 3.0, 4.0]])
    legal = torch.tensor([[False, True, False, True]])
    masked = j2.masked_logits(logits, legal)
    assert torch.argmax(masked, dim=-1).item() == 3
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.masked_logits(logits, torch.zeros_like(legal))
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.masked_logits(
            torch.tensor([[math.nan, 0.0, 0.0, 0.0]]),
            torch.ones_like(legal),
        )


def test_root_equal_weights_equalize_unequal_roots() -> None:
    weights = j2.root_equal_weights((1, 7, 31))
    offsets = np.cumsum((0, 1, 7, 31))
    totals = [
        weights[offsets[index] : offsets[index + 1]].sum()
        for index in range(3)
    ]
    assert totals == pytest.approx([1.0, 1.0, 1.0], abs=1e-15)


def test_distillation_loss_applies_root_weights_to_both_heads() -> None:
    batch = j2.synthetic_distillation_batch(root_lengths=(1, 7))
    model, _optimizer = j2.initialize_model_optimizer()
    losses = j2.distillation_loss(model, batch)
    assert torch.isfinite(losses["total_loss"])
    assert losses["total_loss"].item() == pytest.approx(
        losses["policy_loss"].item()
        + 0.5 * losses["value_loss"].item(),
        abs=1e-7,
    )
    totals: dict[str, float] = {}
    for root_id, weight in zip(
        batch.root_ids,
        batch.row_weights.tolist(),
    ):
        totals[root_id] = totals.get(root_id, 0.0) + weight
    assert totals == pytest.approx({"root-0": 1.0, "root-1": 1.0})


def test_illegal_teacher_action_fails_closed() -> None:
    batch = j2.synthetic_distillation_batch()
    illegal = j2.DistillationBatch(
        observations=batch.observations,
        legal_masks=batch.legal_masks.clone(),
        teacher_actions=batch.teacher_actions.clone(),
        value_targets=batch.value_targets,
        row_weights=batch.row_weights,
        root_ids=batch.root_ids,
    )
    illegal.legal_masks[0, illegal.teacher_actions[0]] = False
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.validate_distillation_batch(illegal)


def test_distillation_plan_has_eight_epochs_and_final_short() -> None:
    audit = j2.distillation_plan_audit(
        4_097,
        epochs=8,
        minibatch_size=4_096,
    )
    assert audit["passes"]
    plan = j2.deterministic_distillation_plan(
        4_097,
        epochs=8,
        minibatch_size=4_096,
    )
    assert len(plan) == 16
    assert sum(row["final_short"] for row in plan) == 8


def test_distillation_resume_is_bit_exact() -> None:
    fixture = j2.synthetic_readiness_fixture()
    assert fixture["passes"]
    assert fixture["checks"]["resume_model_exact"]
    assert fixture["checks"]["resume_optimizer_exact"]


def test_mismatched_optimizer_fails_closed() -> None:
    batch = j2.synthetic_distillation_batch()
    model, _optimizer = j2.initialize_model_optimizer()
    other_model, wrong_optimizer = j2.initialize_model_optimizer()
    assert model is not other_model
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.DistillationUpdater(
            model,
            wrong_optimizer,
            batch,
            minibatch_size=4,
            epochs=1,
        )


@pytest.mark.parametrize(
    ("maximum", "empties", "expected"),
    [
        (96, 4, "low_air"),
        (96, 3, "low_constrained"),
        (192, 3, "mid_progression"),
        (767, 2, "mid_progression"),
        (768, 2, "upper_progression"),
    ],
)
def test_feature_family_is_current_state_only(
    maximum: int,
    empties: int,
    expected: str,
) -> None:
    board = np.ones((4, 4), dtype=np.int64) * 3
    board.flat[:empties] = 0
    board.flat[-1] = maximum
    assert j2.feature_family(board) == expected


def _board_for_family(family: str) -> np.ndarray:
    board = np.ones((4, 4), dtype=np.int64) * 3
    if family == "low_air":
        board.flat[:4] = 0
        board.flat[-1] = 96
    elif family == "low_constrained":
        board.flat[:3] = 0
        board.flat[-1] = 96
    elif family == "mid_progression":
        board.flat[:3] = 0
        board.flat[-1] = 384
    elif family == "upper_progression":
        board.flat[:3] = 0
        board.flat[-1] = 768
    else:
        raise AssertionError(family)
    return board


def test_feature_inventory_reports_natural_and_capped_without_dropping() -> None:
    rows = [
        {
            "root_id": f"{family}-{index % 256:03d}",
            "transition_index": index // 256,
            "board": _board_for_family(family),
        }
        for family in j2.FEATURE_FAMILIES
        for index in range(1_024)
    ]
    report = j2.feature_inventory(rows)
    assert report["passes"]
    assert report["natural_state_count"] == len(rows) == 4_096
    assert report["capped_k_per_family"] == 1_024
    assert report["capped_family_frequencies"] == pytest.approx(
        {family: 0.25 for family in j2.FEATURE_FAMILIES}
    )


def test_feature_inventory_support_shortfall_holds() -> None:
    rows = [
        {
            "root_id": f"root-{index}",
            "transition_index": 0,
            "board": _board_for_family("low_air"),
        }
        for index in range(10)
    ]
    report = j2.feature_inventory(rows)
    assert not report["passes"]
    gate = j2.bc_mechanism_gate(
        overall_root_equal_accuracy=1.0,
        family_accuracies={
            family: 1.0 for family in j2.FEATURE_FAMILIES
        },
        policy_loss=0.1,
        value_mse=0.1,
        zero_value_mse=0.2,
        illegal_teacher_actions=0,
        inventory=report,
    )
    assert gate["decision"] == "HOLD_J2_DISTILLATION_DATA_SUPPORT"


def test_value_target_is_true_objective_preserving() -> None:
    assert j2.value_target(
        current_score=10,
        final_score=31,
        remaining_score_deltas=(3, 6, 12),
    ) == pytest.approx(0.00021)
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.value_target(
            current_score=10,
            final_score=31,
            remaining_score_deltas=(3, 6),
        )


def test_teacher_kl_schedule_endpoints_and_zero_tail() -> None:
    assert j2.teacher_kl_coefficient(1) == pytest.approx(0.05)
    assert j2.teacher_kl_coefficient(16) == pytest.approx(0.003125)
    assert j2.teacher_kl_coefficient(17) == 0.0
    assert j2.teacher_kl_coefficient(64) == 0.0
    assert all(
        j2.teacher_kl_coefficient(round_number)
        >= j2.teacher_kl_coefficient(round_number + 1)
        for round_number in range(1, 64)
    )


def test_j2_ppo_loss_has_teacher_anchor_and_no_auxiliary() -> None:
    model, _optimizer = j2.initialize_model_optimizer()
    row_count = 8
    observations = torch.zeros((row_count, j2.OBSERVATION_WIDTH))
    legal = torch.ones((row_count, 4), dtype=torch.bool)
    with torch.no_grad():
        logits, _values = model(observations)
        distribution = torch.distributions.Categorical(
            logits=j2.masked_logits(logits, legal)
        )
        actions = torch.arange(row_count, dtype=torch.int64) % 4
        old = distribution.log_prob(actions)
    batch = j2.J2PPOBatch(
        observations=observations,
        legal_masks=legal,
        actions=actions,
        old_log_probabilities=old,
        advantages=torch.linspace(-1.0, 1.0, row_count),
        returns=torch.zeros(row_count),
        teacher_actions=torch.zeros(row_count, dtype=torch.int64),
        row_weights=torch.ones(row_count) / 4.0,
        root_ids=("a",) * 4 + ("b",) * 4,
    )
    round1 = j2.j2_ppo_loss(model, batch, round_number=1)
    post_anchor = j2.J2PPOBatch(
        observations=batch.observations,
        legal_masks=batch.legal_masks,
        actions=batch.actions,
        old_log_probabilities=batch.old_log_probabilities,
        advantages=batch.advantages,
        returns=batch.returns,
        teacher_actions=None,
        row_weights=batch.row_weights,
        root_ids=batch.root_ids,
    )
    round17 = j2.j2_ppo_loss(model, post_anchor, round_number=17)
    assert "auxiliary_loss" not in round1
    assert round1["teacher_kl_coefficient"].item() == pytest.approx(0.05)
    assert round17["teacher_kl_coefficient"].item() == 0.0
    assert round17["teacher_kl"].item() == 0.0
    assert round1["total_loss"].item() > round17["total_loss"].item()


def test_teacher_actions_required_through_16_and_forbidden_after() -> None:
    model, _optimizer = j2.initialize_model_optimizer()
    rows = 2
    observations = torch.zeros((rows, j2.OBSERVATION_WIDTH))
    legal = torch.ones((rows, 4), dtype=torch.bool)
    with torch.no_grad():
        logits, _values = model(observations)
        actions = torch.zeros(rows, dtype=torch.int64)
        old = torch.distributions.Categorical(
            logits=j2.masked_logits(logits, legal)
        ).log_prob(actions)

    def batch(teacher_actions: torch.Tensor | None) -> j2.J2PPOBatch:
        return j2.J2PPOBatch(
            observations=observations,
            legal_masks=legal,
            actions=actions,
            old_log_probabilities=old,
            advantages=torch.tensor([-1.0, 1.0]),
            returns=torch.zeros(rows),
            teacher_actions=teacher_actions,
            row_weights=torch.ones(rows),
            root_ids=("a", "b"),
        )

    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.j2_ppo_loss(model, batch(None), round_number=16)
    round16 = j2.j2_ppo_loss(
        model,
        batch(torch.zeros(rows, dtype=torch.int64)),
        round_number=16,
    )
    assert round16["teacher_kl_coefficient"].item() == pytest.approx(
        0.003125
    )
    round17 = j2.j2_ppo_loss(model, batch(None), round_number=17)
    assert round17["teacher_kl"].item() == 0.0
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.j2_ppo_loss(
            model,
            batch(torch.zeros(rows, dtype=torch.int64)),
            round_number=17,
        )
    assert j2.ONLINE_TEACHER_ROOTS == 16 * 256 == 4_096


def test_eight_shard_assignment_and_merge_are_exact() -> None:
    plan = j2.shard_plan(19)
    assert set(plan) == set(range(8))
    assert sorted(index for rows in plan.values() for index in rows) == list(
        range(19)
    )
    rows = [
        {
            "row_index": index,
            "shard": index % 8,
            "payload": {"value": index},
            "row_identity": j2.canonical_json_hash(
                {
                    "row_index": index,
                    "shard": index % 8,
                    "payload": {"value": index},
                }
            ),
        }
        for index in range(19)
    ]
    merged = j2.deterministic_shard_merge(rows[::-1])
    assert [row["row_index"] for row in merged] == list(range(19))
    rows[0]["shard"] = 7
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.deterministic_shard_merge(rows)


@pytest.mark.parametrize(
    ("treatment", "control", "totals"),
    [
        ([0] * 8, [1] * 8, [2] * 8),
        ([1] * 8, [0] * 8, [2] * 8),
        ([3, 2, 4, 1, 3, 2, 4, 1], [2, 3, 1, 4, 2, 3, 1, 4], [8] * 8),
    ],
)
def test_common_or_matches_accepted_edge_correction(
    treatment: list[int],
    control: list[int],
    totals: list[int],
) -> None:
    treatment_array = np.asarray([treatment])
    control_array = np.asarray([control])
    total_array = np.asarray([totals], dtype=np.float64)
    assert j2._mh_log_or(
        treatment_array,
        control_array,
        total_array,
    ) == pytest.approx(
        o2._mh_log_or(
            treatment_array,
            control_array,
            total_array,
        ),
        abs=1e-15,
    )


def test_within_stratum_bootstrap_matches_accepted_helper() -> None:
    cells = [
        np.asarray([10 + index, 2, 3, 17 - index])
        for index in range(8)
    ]
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    assert j2._bootstrap_binary_bounds(
        cells,
        32,
        rng=rng_a,
        bootstraps=31,
    ) == pytest.approx(
        o2._bootstrap_binary_bounds(
            cells,
            32,
            rng=rng_b,
            bootstraps=31,
        ),
        abs=1e-15,
    )


def test_common_or_power_is_deterministic_and_eight_stratum() -> None:
    first = j2.simulate_common_or_noninferiority_power(
        n_pairs=2_048,
        control_rate=0.04,
        coupling=0.05,
        datasets=16,
        bootstraps=11,
    )
    second = j2.simulate_common_or_noninferiority_power(
        n_pairs=2_048,
        control_rate=0.04,
        coupling=0.05,
        datasets=16,
        bootstraps=11,
    )
    assert first == second
    assert first["strata"] == 8
    assert first["bootstrap_method"].startswith(
        "eight independent within-stratum"
    )


def test_fidelity_score_power_reports_full_gate_not_ci_only() -> None:
    report = j2.score_fidelity_power()
    assert report["score_80pct_mde_percent"] == pytest.approx(
        8.045644927819161
    )
    assert report["equal_policy_10pct_ci_only_power"] == pytest.approx(
        0.9681657505636305
    )
    assert report["equal_policy_combined_gate_power"] == pytest.approx(
        0.8649301941261354
    )
    assert report["equal_policy_5pct_ci_only_power"] == pytest.approx(
        0.45900197536106296
    )


def test_cost_projection_keeps_feasibility_gates_separate() -> None:
    report = j2.runtime_storage_projection()
    assert report["integrity_passes"]
    assert not report["feasibility_passes"]
    assert not report["pretraining_sharding_evidence"]["accepted"]
    assert not report["online_teacher_query_evidence"]["accepted"]
    assert report["checks"][
        "synthetic_sharding_not_used_as_real_evidence"
    ]
    assert report["teacher_workload"]["pre_ppo"]["root_count"] == 10_240
    assert report["teacher_workload"]["on_policy_anchor"][
        "root_count"
    ] == 4_096
    assert report["teacher_workload"]["total_root_equivalents"] == 14_336
    sensitivity = report["sensitivity_5000_moves"]
    assert sensitivity["diagnostic_not_conjunctive"]
    assert sensitivity["moves_per_root"] == 5_000
    for phase in ("distillation", "on_policy_training"):
        assert (
            sensitivity[phase]["wall_hours_with_25pct_margin"] > 0.0
        )
        assert (
            sensitivity[phase]["storage_with_25pct_margin_bytes"] > 0
        )
        assert not sensitivity[phase]["runtime_fits_cap"]
        assert not sensitivity[phase]["storage_fits_cap"]


def test_teacher_provenance_hashes_replay_without_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_text

    def guarded(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() == j2.INCUMBENT_REPLAY_PATH.resolve():
            raise AssertionError("Replay body was parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    report = j2.teacher_provenance_audit()
    assert report["passes"]
    assert report["replay_payload_parsed"] is False
    assert report["checks"]["dashboard_association_exact"]


def test_parent_and_j1d_structural_identities_are_exact() -> None:
    report = j2.source_and_parent_audit()
    assert report["checks"]["all_parent_source_hashes_exact"]
    assert report["checks"]["all_parent_artifact_hashes_exact"]
    assert report["checks"]["j1d_terminal_is_clean_hold"]
    assert report["checks"][
        "j1d_checkpoint_identity_exact_and_quarantined"
    ]


def test_json_native_immutable_write_is_create_once(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = {"tuple": (1, 2), "nested": {"value": np.int64(3)}}
    written = j2.write_immutable_json(
        path,
        payload,
        field="payload_sha256",
    )
    assert written["tuple"] == [1, 2]
    exact = path.read_bytes()
    with pytest.raises(FileExistsError):
        j2.write_immutable_json(
            path,
            payload,
            field="payload_sha256",
        )
    assert path.read_bytes() == exact
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.write_immutable_json(
            path,
            {"tuple": (1, 9)},
            field="payload_sha256",
        )


def test_readiness_decision_scopes_hold_vs_kill() -> None:
    hold = j2.readiness_decision(
        integrity_checks={"identity": True},
        feasibility_checks={"real_sharding": False},
        operational_checks={"services": True},
    )
    assert hold["decision"] == j2.HOLD
    kill = j2.readiness_decision(
        integrity_checks={"identity": False},
        feasibility_checks={"real_sharding": False},
        operational_checks={"services": True},
    )
    assert kill["decision"] == j2.KILL
    ready = j2.readiness_decision(
        integrity_checks={"identity": True},
        feasibility_checks={"real_sharding": True},
        operational_checks={"services": True},
    )
    assert ready["decision"] == j2.READY


def test_zero_work_audit_has_no_marker_or_reservation(tmp_path: Path) -> None:
    output = tmp_path / "readiness"
    report = j2.audit_zero_work(
        output_dir=output,
        root=tmp_path,
        include_operational=False,
    )
    assert report["passes"]
    assert all(value == 0 for value in report["zero_work"].values())


def test_cli_has_only_three_readiness_verbs() -> None:
    parser = j2.build_parser()
    actions = [
        action
        for action in parser._actions
        if isinstance(action, __import__("argparse")._SubParsersAction)
    ]
    assert len(actions) == 1
    assert set(actions[0].choices) == {
        "audit-zero-work",
        "write-test-evidence",
        "prepare",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["execute"])
    with pytest.raises(SystemExit):
        parser.parse_args(["reserve"])


def test_ast_has_no_import_time_science_or_forbidden_cli() -> None:
    tree = ast.parse(j2.RUNNER_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "teacher_provenance_audit",
        "incumbent_policy_binding",
        "make_policy",
        "normal_start_sim",
        "ThreesSim",
        "J2ActorCritic",
        "initialize_model_optimizer",
        "prepare",
    }
    top_level_calls = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    top_level_calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    top_level_calls.add(child.func.attr)
    assert not (top_level_calls & forbidden)
    source = j2.RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden_verb in (
        '"execute"',
        '"reserve"',
        '"consume"',
        '"train"',
        '"evaluate"',
        '"promote"',
    ):
        assert f"add_parser({forbidden_verb}" not in source


def test_import_has_no_filesystem_or_science_side_effect(
    tmp_path: Path,
) -> None:
    script = (
        "import json, pathlib\n"
        "before=set(pathlib.Path('.').iterdir())\n"
        "import threes_rl.j2_incumbent_distillation_readiness as j\n"
        "after=set(pathlib.Path('.').iterdir())\n"
        "print(json.dumps({'created': sorted(str(p) for p in after-before),"
        "'zero': j.ZERO_WORK}))\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(j2.REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["created"] == []
    assert all(value == 0 for value in payload["zero"].values())


def test_prepare_rejects_reduced_power_workload(tmp_path: Path) -> None:
    with pytest.raises(j2.J2ReadinessIntegrityError):
        j2.prepare(
            output_dir=tmp_path,
            power_datasets=16,
            power_bootstraps=11,
        )

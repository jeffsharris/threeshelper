from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from threes_rl import j1_joint_policy_value as j1


def test_parent_identities_and_selected_model_are_exact():
    source = j1.source_identity()
    assert source["passes"]
    assert source["proposal"]["file_sha256"] == j1.EXPECTED_PROPOSAL_SHA256
    assert (
        source["readiness"]["file_sha256"]
        == j1.EXPECTED_READINESS_FILE_SHA256
    )
    assert (
        source["readiness"]["payload_sha256"]
        == j1.EXPECTED_READINESS_PAYLOAD_SHA256
    )
    assert j1.parameter_count() == 411_656
    assert j1.observation_size("full") == 282
    assert torch.__version__ == "2.12.1"


def test_transition_t_gae_matches_hand_fixture_and_rejects_shifted_done():
    fixture = j1.gae_contract_fixture()
    assert fixture["passes"]
    assert fixture["observed_advantages"] == pytest.approx(
        [2.43, 1.4, 3.1],
        abs=1e-12,
    )
    assert fixture["checks"]["done_t_plus_1_regression_would_fail"]


def test_normal_start_is_starter_none_for_every_role_and_roundtrips():
    for index, role in enumerate(("train", "development", "confirmation")):
        sim, state = j1.normal_start_sim(
            role=role,
            deck_stream_id=10_000 + index,
            slot_stream_id=20_000 + index,
        )
        assert sim.starter_tile is None
        assert int(state.board.max(initial=0)) < 48
        snapshot = j1.simulator_snapshot(sim, state)
        restored_sim, restored_state = j1.simulator_from_snapshot(snapshot)
        assert restored_sim.starter_tile is None
        assert j1.stable_hash(j1.state_snapshot(restored_state)) == j1.stable_hash(
            j1.state_snapshot(state)
        )
        assert j1.stable_hash(
            restored_sim.deck_rng.bit_generator.state
        ) == j1.stable_hash(sim.deck_rng.bit_generator.state)
        assert j1.stable_hash(
            restored_sim.slot_rng.bit_generator.state
        ) == j1.stable_hash(sim.slot_rng.bit_generator.state)


def test_complete_roots_are_required_and_root_weights_are_equal():
    roots = [
        j1.CompleteRoot(
            root_id="r1",
            ancestry_id="a1",
            partition="train",
            transitions=({"x": 1}, {"x": 2}),
            natural_terminal=True,
        ),
        j1.CompleteRoot(
            root_id="r2",
            ancestry_id="a2",
            partition="train",
            transitions=(
                {"x": 3},
                {"x": 4},
                {"x": 5},
                {"x": 6},
                {"x": 7},
            ),
            natural_terminal=True,
        ),
    ]
    flattened = j1.flatten_complete_roots(roots, expected_partition="train")
    assert flattened["per_root_weight"] == pytest.approx(
        {"r1": 1.0, "r2": 1.0},
        abs=2e-7,
    )
    bad = replace(roots[0], natural_terminal=False)
    with pytest.raises(j1.J1IntegrityError, match="Truncated"):
        j1.flatten_complete_roots(
            [bad, roots[1]],
            expected_partition="train",
        )
    crossed = replace(roots[1], ancestry_id="a1")
    with pytest.raises(j1.J1IntegrityError, match="Ancestry"):
        j1.flatten_complete_roots(
            [roots[0], crossed],
            expected_partition="train",
        )


def test_dense_score_delta_reward_telescopes_on_crafted_and_random_complete_games():
    crafted = j1.verify_dense_reward_telescoping(0, 18, [3, 6, 9])
    assert crafted["passes"]
    assert crafted["scaled_return"] == pytest.approx(18e-5)
    for seed in (31, 37):
        fixture = j1.dense_reward_complete_fixture(seed)
        assert fixture["passes"]
        assert fixture["natural_terminal"]
        assert not fixture["score_fields_retained"]
        assert not fixture["action_sequence_retained"]


def test_ppo_clipping_uses_old_logprob_legal_masks_and_all_weighted_terms():
    model, _optimizer = j1.initialize_model_optimizer()
    batch = j1.synthetic_complete_ppo_batch(model)
    with torch.no_grad():
        logits, _values, _auxiliary = model(batch.observations)
        new_logprob = torch.distributions.Categorical(
            logits=j1.masked_logits(logits, batch.legal_masks)
        ).log_prob(batch.actions)
    clipped_batch = replace(
        batch,
        old_log_probabilities=new_logprob - math.log(4.0),
    )
    normalized = torch.tensor(
        [1.0, -1.0, 1.0, -1.0, 0.0],
        dtype=torch.float32,
    )
    losses = j1.frozen_ppo_loss(
        model,
        clipped_batch,
        normalized_advantages=normalized,
    )
    ratio = torch.full_like(normalized, 4.0)
    expected_rows = torch.maximum(
        -normalized * ratio,
        -normalized
        * torch.clamp(
            ratio,
            1.0 - j1.FROZEN_CONFIG.clip_coef,
            1.0 + j1.FROZEN_CONFIG.clip_coef,
        ),
    )
    expected = torch.sum(batch.row_weights * expected_rows) / torch.sum(
        batch.row_weights
    )
    assert float(losses["policy_loss"].detach()) == pytest.approx(
        float(expected),
        abs=1e-6,
    )
    for name in (
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "auxiliary_loss",
        "approx_kl",
    ):
        assert torch.isfinite(losses[name])


def test_weighted_advantage_normalization_and_every_loss_share_root_weights():
    fixture = j1.ppo_contract_fixture()
    assert fixture["passes"]
    assert fixture["root_lengths"] == [2, 5]
    assert fixture["root_total_weights"] == pytest.approx(
        {"synthetic-root-0": 1.0, "synthetic-root-1": 1.0},
        abs=2e-7,
    )
    assert abs(fixture["weighted_normalized_advantage_mean"]) < 1e-6
    assert fixture["weighted_normalized_advantage_variance"] == pytest.approx(
        1.0,
        abs=1e-5,
    )
    assert len(set(fixture["component_weight_sha256"].values())) == 1


def test_global_weighted_minibatch_reductions_reconstruct_full_objective():
    model, _optimizer = j1.initialize_model_optimizer()
    batch = j1.synthetic_complete_ppo_batch(
        model,
        row_count=7,
        root_lengths=(2, 5),
    )
    normalized = j1.normalize_advantages_root_weighted(
        batch.advantages,
        batch.row_weights,
    )
    full = j1.frozen_ppo_loss(
        model,
        batch,
        normalized_advantages=normalized,
    )
    plan = j1.deterministic_epoch_minibatches(
        7,
        round_number=3,
        epochs=1,
        minibatch_size=3,
    )
    partial = []
    for row in plan:
        indices = torch.tensor(row["indices"], dtype=torch.int64)
        partial.append(
            j1.frozen_ppo_loss(
                model,
                batch.subset(indices),
                normalized_advantages=normalized[indices],
                global_weight_total=batch.row_weights.sum(),
                minibatches_per_epoch=len(plan),
            )
        )
    for component in (
        "policy_loss",
        "value_loss",
        "entropy",
        "auxiliary_loss",
        "total_loss",
    ):
        observed = sum(
            float(row[component].detach()) for row in partial
        ) / len(plan)
        assert observed == pytest.approx(
            float(full[component].detach()),
            abs=1e-6,
        )


def test_four_epoch_permutations_are_deterministic_and_retain_short_batch():
    np.random.seed(999)
    first = j1.ppo_schedule_audit(
        10,
        round_number=7,
        epochs=4,
        minibatch_size=4,
    )
    np.random.seed(1)
    second = j1.ppo_schedule_audit(
        10,
        round_number=7,
        epochs=4,
        minibatch_size=4,
    )
    assert first["passes"]
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["coverage_counts"] == {
        "0": 10,
        "1": 10,
        "2": 10,
        "3": 10,
    }
    assert first["short_minibatch_sizes"] == [2, 2, 2, 2]


def test_round_learning_rate_is_exact_and_zero_only_after_round_64():
    assert j1.round_learning_rate(1) == j1.FROZEN_CONFIG.learning_rate
    assert j1.round_learning_rate(64) == pytest.approx(
        j1.FROZEN_CONFIG.learning_rate / 64.0
    )
    assert j1.round_learning_rate(64, after_round=True) == 0.0
    with pytest.raises(ValueError):
        j1.round_learning_rate(0)


def test_actual_ppo_update_runs_four_epochs_and_clips_gradients():
    model, optimizer = j1.initialize_model_optimizer()
    batch = j1.synthetic_complete_ppo_batch(
        model,
        row_count=7,
        root_lengths=(2, 5),
    )
    before = j1.stable_hash(model.state_dict())
    report = j1.apply_frozen_ppo_update(
        model,
        optimizer,
        batch,
        round_number=1,
        epochs=4,
        minibatch_size=3,
        optimizer_step=True,
    )
    assert report["passes"]
    assert report["optimizer_steps"] == 12
    assert all(
        row["gradient_norm_before_clip"] >= 0.0
        for row in report["minibatches"]
    )
    assert j1.stable_hash(model.state_dict()) != before


@pytest.mark.parametrize("boundary", j1.RESUME_FIXTURE_BOUNDARIES)
def test_resume_is_bit_identical_at_every_frozen_boundary(tmp_path, boundary):
    report = j1.resume_equivalence_fixture(
        boundary,
        checkpoint_path=tmp_path / f"{boundary}.pt",
    )
    assert report["passes"]
    assert report["expected"] == report["observed"]
    assert report["expected"]["checkpoint_sealed"]
    if boundary in {"pre_update", "post_checkpoint"}:
        assert report["expected"]["update_report_sha256"] is not None


def test_resume_checkpoint_corruption_and_nonfinite_model_fail_closed(tmp_path):
    session = j1.ResumeFixtureSession()
    path = tmp_path / "resume.pt"
    j1.save_resume_fixture(path, session)
    raw = bytearray(path.read_bytes())
    raw[-17] ^= 0xFF
    path.write_bytes(raw)
    with pytest.raises(j1.J1IntegrityError, match="Corrupt|hash"):
        j1.load_resume_fixture(path)
    model, _optimizer = j1.initialize_model_optimizer()
    with torch.no_grad():
        next(model.parameters()).reshape(-1)[0] = float("nan")
    with pytest.raises(j1.J1IntegrityError, match="Nonfinite"):
        j1.assert_finite_model(model)


def test_legal_masking_is_finite_deterministic_and_never_selects_illegal():
    logits = torch.tensor(
        [[1.0, 1000.0, 1.0, 1.0], [3.0, 3.0, 2.0, 1.0]],
        dtype=torch.float32,
    )
    masks = torch.tensor(
        [[True, False, True, False], [True, True, False, False]],
        dtype=torch.bool,
    )
    actions = j1.deterministic_masked_actions(logits, masks)
    assert actions.tolist() == [0, 0]
    generator = torch.Generator().manual_seed(7)
    sampled = j1.sampled_masked_actions(
        logits.repeat_interleave(20, dim=0),
        masks.repeat_interleave(20, dim=0),
        generator=generator,
    )
    repeated_masks = masks.repeat_interleave(20, dim=0)
    assert torch.all(
        repeated_masks.gather(1, sampled.unsqueeze(1)).squeeze(1)
    )
    with pytest.raises(ValueError, match="at least one legal"):
        j1.masked_logits(
            torch.zeros((1, 4)),
            torch.zeros((1, 4), dtype=torch.bool),
        )


def test_model_save_load_payload_rejects_wrong_width_and_illegal_batch():
    model, _optimizer = j1.initialize_model_optimizer()
    batch = j1.synthetic_complete_ppo_batch(model)
    illegal = replace(
        batch,
        actions=torch.tensor([3, 1, 0, 0, 1], dtype=torch.int64),
    )
    with pytest.raises(j1.J1IntegrityError, match="illegal chosen action"):
        j1.validate_ppo_batch(illegal)
    wrong_width = replace(
        batch,
        observations=torch.zeros((5, 281), dtype=torch.float32),
    )
    with pytest.raises(j1.J1IntegrityError, match="observation shape"):
        j1.validate_ppo_batch(wrong_width)


def test_prospective_stream_contract_is_disjoint_and_collision_fails(monkeypatch):
    assert j1.prospective_stream_contract()["passes"]
    changed = copy.deepcopy(j1.PROSPECTIVE_STREAMS)
    changed["development"]["logical"] = changed["train"]["logical"]
    changed["development"]["rows"] = 1
    changed["train"]["rows"] = 1
    monkeypatch.setattr(j1, "PROSPECTIVE_STREAMS", changed)
    report = j1.prospective_stream_contract()
    assert not report["passes"]
    assert report["duplicate_stream_id_count"] == 1


def test_byte_only_denylist_detects_mutation_unknown_path_and_symlink(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    runs = repo / "threes_rl" / "runs"
    runs.mkdir(parents=True)
    protected = repo / "protected-root.bin"
    protected.write_bytes(b"opaque-root")
    stream = runs / "OPAQUE_STREAM_MANIFEST.bin"
    stream.write_bytes(b"not-json-and-never-parsed")
    monkeypatch.setattr(
        j1,
        "ROOT_MANIFEST_BINDINGS",
        {
            "protected-root.bin": hashlib.sha256(
                b"opaque-root"
            ).hexdigest()
        },
    )
    sealed = j1.build_protected_denylist(
        repo_root=repo,
        runs_root=runs,
    )
    assert sealed["passes"]
    assert not sealed["protected_payloads_parsed"]
    assert j1.verify_protected_denylist(
        sealed,
        repo_root=repo,
        runs_root=runs,
    )
    stream.write_bytes(b"mutated")
    assert not j1.verify_protected_denylist(
        sealed,
        repo_root=repo,
        runs_root=runs,
    )
    stream.write_bytes(b"not-json-and-never-parsed")
    (runs / "NEW_STREAM_MANIFEST.bin").write_bytes(b"new")
    assert not j1.verify_protected_denylist(
        sealed,
        repo_root=repo,
        runs_root=runs,
    )
    protected.unlink()
    protected.symlink_to(runs / "OPAQUE_STREAM_MANIFEST.bin")
    with pytest.raises(j1.J1IntegrityError, match="symlinked"):
        j1.build_protected_denylist(repo_root=repo, runs_root=runs)


def test_dependency_identity_mismatch_fails_closed(monkeypatch):
    changed = dict(j1.DEPENDENCY_BINDINGS)
    changed["threes_rl/sim.py"] = "0" * 64
    monkeypatch.setattr(j1, "DEPENDENCY_BINDINGS", changed)
    report = j1.source_identity()
    assert not report["passes"]
    assert not report["checks"]["dependencies_exact"]


def _fake_timing(seconds: float = 0.001):
    summary = {
        "count": 5,
        "median_seconds": seconds,
        "p90_seconds": seconds,
        "p99_seconds": seconds,
        "max_seconds": seconds,
    }
    return {
        "fixture_only": True,
        "game_roots_generated": 0,
        "policy_outcome_inspection": 0,
        "optimizer_steps": 0,
        "action_identities_retained": False,
        "actor_batch_size": 16,
        "update_batch_size": 4096,
        "actor_batch": dict(summary),
        "simulator_transition": dict(summary),
        "synthetic_forward_backward": dict(summary),
        "incumbent_fixed_state_action": dict(summary),
    }


def test_projection_accounts_all_arms_four_epochs_and_max_sensitivity_margin():
    report = j1.runtime_storage_projection(_fake_timing())
    assert report["passes"]
    assert report["checks"]["total_game_arms_exact"]
    assert (
        sum(
            row["complete_game_arms"]
            for row in report["phase_projections"].values()
        )
        == 28_672
    )
    training = report["phase_projections"]["training"]
    expected_passes = (
        j1.TRAIN_ROOTS
        * j1.PLANNING_MOVES
        * j1.FROZEN_CONFIG.epochs_per_round
    )
    expected_batches = math.ceil(
        expected_passes / j1.FROZEN_CONFIG.minibatch_size
    )
    assert training["update_seconds"] == pytest.approx(
        expected_batches * 0.001
    )
    for row in report["phase_projections"].values():
        assert (
            row[
                "contract_max_5000_move_sensitivity_hours_with_25pct_margin"
            ]
            == pytest.approx(
                row["contract_max_5000_move_sensitivity_hours"] * 1.25
            )
        )
        assert isinstance(
            row["contract_max_5000_move_sensitivity_runtime_passes"],
            bool,
        )


def test_projection_cost_miss_is_not_an_integrity_failure():
    report = j1.runtime_storage_projection(_fake_timing(seconds=1.0))
    assert report["integrity_passes"]
    assert not report["cost_passes"]
    assert not report["passes"]


def test_runtime_update_fixture_calls_actual_ppo_path_without_step(monkeypatch):
    called = []
    original = j1.apply_frozen_ppo_update

    def wrapped(*args, **kwargs):
        called.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(j1, "apply_frozen_ppo_update", wrapped)
    monkeypatch.setattr(
        j1,
        "_benchmark_calls",
        lambda call, **_kwargs: (
            call()
            or {
                "count": 1,
                "median_seconds": 0.001,
                "p90_seconds": 0.001,
                "p99_seconds": 0.001,
                "max_seconds": 0.001,
            }
        ),
    )

    class Policy:
        def __call__(self, state, sim, rng):
            return sim.legal_actions(state)[0]

    import threes_rl.eval as eval_module

    monkeypatch.setattr(eval_module, "make_policy", lambda _spec: Policy())
    report = j1.benchmark_projection_fixtures()
    assert report["optimizer_steps"] == 0
    assert called
    assert any(row["optimizer_step"] is False for row in called)
    assert any(row["minibatch_size"] == 4096 for row in called)


def test_zero_work_audit_reads_health_but_no_human_session_content(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        j1,
        "source_identity",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        j1,
        "operational_audit",
        lambda **_kwargs: {
            "passes": True,
            "services": {
                "recorder": {"active_session_content_read": False}
            },
        },
    )
    report = j1.audit_zero_work(output_dir=tmp_path / "absent")
    assert report["passes"]
    assert report["decision"] == "READY_J1_IMPLEMENTATION_PREFLIGHT_SURFACE"
    assert all(value == 0 for value in report["zero_work"].values())


def test_test_evidence_is_immutable_and_science_counters_stay_zero(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "evidence"
    monkeypatch.setattr(
        j1,
        "source_identity",
        lambda: {"passes": True, "identity": "fixed"},
    )
    artifact = j1.write_test_evidence(
        focused_passed=27,
        regressions_passed=400,
        deselections=["historical-state-a"],
        commands=["pytest focused", "pytest broad"],
        output_dir=output,
    )
    assert Path(artifact["path"]).is_file()
    payload = j1._load_hashed_json(
        Path(artifact["path"]),
        field="test_evidence_payload_sha256",
    )
    assert payload["zero_work"]["scientific_optimizer_steps"] == 0
    with pytest.raises(j1.J1IntegrityError, match="absent"):
        j1.write_test_evidence(
            focused_passed=27,
            regressions_passed=400,
            deselections=[],
            commands=[],
            output_dir=output,
        )


def _prepare_fixture(tmp_path, monkeypatch, *, operational=True, semantic=True):
    output = tmp_path / "preflight"
    monkeypatch.setattr(j1, "OUTPUT_DIR", output)
    identity = {"passes": True, "identity": "fixed"}
    monkeypatch.setattr(j1, "source_identity", lambda: identity)
    j1.write_immutable_json(
        output / j1.TEST_EVIDENCE_NAME,
        {
            "source_identity": identity,
            "passes": True,
            "zero_work": dict(j1.ZERO_WORK),
        },
        field="test_evidence_payload_sha256",
    )
    monkeypatch.setattr(
        j1,
        "operational_audit",
        lambda **_kwargs: {"passes": operational},
    )
    monkeypatch.setattr(
        j1,
        "build_protected_denylist",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        j1,
        "semantic_contract_audit",
        lambda **_kwargs: {"passes": semantic},
    )
    monkeypatch.setattr(
        j1,
        "runtime_storage_projection",
        lambda: {"passes": True},
    )
    return output


def test_prepare_seals_ready_without_marker_or_execution_command(
    tmp_path,
    monkeypatch,
):
    output = _prepare_fixture(tmp_path, monkeypatch)
    report = j1.prepare(output_dir=output)
    assert report["decision"] == "READY_J1_IMPLEMENTATION_PREFLIGHT"
    names = {path.name for path in output.iterdir()}
    assert names == {
        j1.TEST_EVIDENCE_NAME,
        j1.DENYLIST_NAME,
        j1.PROJECTION_NAME,
        j1.PREFLIGHT_LOCK_NAME,
        j1.PREFLIGHT_RESULT_NAME,
    }
    assert not any("marker" in name.lower() or "opened" in name.lower() for name in names)
    lock = j1._load_hashed_json(
        output / j1.PREFLIGHT_LOCK_NAME,
        field="preflight_lock_payload_sha256",
    )
    assert lock["marker_defined"] is False
    assert lock["execution_command_defined"] is False
    assert all(value == 0 for value in lock["zero_work"].values())


def test_prepare_classifies_operational_hold_and_integrity_kill(
    tmp_path,
    monkeypatch,
):
    hold_output = _prepare_fixture(
        tmp_path / "hold",
        monkeypatch,
        operational=False,
    )
    hold = j1.prepare(output_dir=hold_output)
    assert hold["decision"] == "HOLD_J1_IMPLEMENTATION_PREFLIGHT"

    cost_output = tmp_path / "cost" / "preflight"
    monkeypatch.setattr(j1, "OUTPUT_DIR", cost_output)
    identity = {"passes": True, "identity": "fixed"}
    monkeypatch.setattr(j1, "source_identity", lambda: identity)
    j1.write_immutable_json(
        cost_output / j1.TEST_EVIDENCE_NAME,
        {"source_identity": identity, "passes": True},
        field="test_evidence_payload_sha256",
    )
    monkeypatch.setattr(
        j1,
        "operational_audit",
        lambda **_kwargs: {"passes": True},
    )
    monkeypatch.setattr(
        j1,
        "build_protected_denylist",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        j1,
        "semantic_contract_audit",
        lambda **_kwargs: {"passes": True},
    )
    monkeypatch.setattr(
        j1,
        "runtime_storage_projection",
        lambda: {
            "passes": False,
            "integrity_passes": True,
            "cost_passes": False,
        },
    )
    cost_hold = j1.prepare(output_dir=cost_output)
    assert cost_hold["decision"] == "HOLD_J1_IMPLEMENTATION_PREFLIGHT"

    kill_output = tmp_path / "kill" / "preflight"
    monkeypatch.setattr(j1, "OUTPUT_DIR", kill_output)
    identity = {"passes": True, "identity": "fixed"}
    monkeypatch.setattr(j1, "source_identity", lambda: identity)
    j1.write_immutable_json(
        kill_output / j1.TEST_EVIDENCE_NAME,
        {"source_identity": identity, "passes": True},
        field="test_evidence_payload_sha256",
    )
    monkeypatch.setattr(
        j1,
        "operational_audit",
        lambda **_kwargs: {"passes": True},
    )
    monkeypatch.setattr(
        j1,
        "build_protected_denylist",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        j1,
        "semantic_contract_audit",
        lambda **_kwargs: {"passes": False},
    )
    monkeypatch.setattr(
        j1,
        "runtime_storage_projection",
        lambda: {"passes": True},
    )
    killed = j1.prepare(output_dir=kill_output)
    assert killed["decision"] == "KILL_J1_IMPLEMENTATION_INTEGRITY"


@pytest.mark.parametrize(
    "forbidden",
    ("open", "execute", "train", "evaluate", "reserve", "marker"),
)
def test_cli_has_no_scientific_or_marker_verb(forbidden):
    parser = j1.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([forbidden])


def test_runtime_is_single_thread_deterministic_cpu():
    model, _optimizer = j1.initialize_model_optimizer()
    assert next(model.parameters()).device.type == "cpu"
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()

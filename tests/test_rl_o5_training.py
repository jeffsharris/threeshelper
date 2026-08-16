import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from threes_rl import o3_option_training as training
from threes_rl.o3_designated_pair_option import (
    EVENT_WIDTH,
    GEOMETRY_WIDTH,
    O3DesignatedPairNet,
    build_decision_targets,
)
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def _state(board: list[list[int]]) -> SimState:
    return SimState(
        board=np.asarray(board, dtype=np.int32),
        preview=preview_from_label("blue"),
        small_counts={"red": 4, "blue": 3, "gray": 4},
        small_pos=1,
        small_seen_total=12,
        span_small_pos=3,
        large_pending=False,
        max_tile=max(max(row) for row in board),
        move_count=20,
        game_over=False,
    )


def test_training_config_is_exact() -> None:
    config = training.training_config()
    assert config["parameter_count"] == 102557
    assert config["schema_sha256"] == (
        "a1c2efa6bd980d32138fb6026c1a5109685db8f1630e1b5fa732b2c2eb983602"
    )
    assert config["episodes"] == 1152
    assert config["epsilon_by_round"] == [1.0, 0.15, 0.10, 0.05]
    assert config["score_target_used"] is False


def test_learning_manifest_has_exact_tasks_and_streams() -> None:
    rows = training._learning_rows()
    assert len(rows) == 1152
    assert (rows[0]["root_index"], rows[0]["round_index"], rows[0]["replicate"]) == (0, 0, 0)
    assert (rows[-1]["root_index"], rows[-1]["round_index"], rows[-1]["replicate"]) == (95, 3, 2)
    flat = [
        int(row[field])
        for row in rows
        for field in training.STREAM_FIELDS
    ]
    assert len(flat) == len(set(flat))


def test_selected_train_root_restores_exactly() -> None:
    rows = training._load_selected_rows()
    train = [row for row in rows if row["role"] == "train"]
    state, pair = training.restore_train_root(train[0])
    assert pair.target == int(train[0]["target"])
    assert [list(value) for value in pair.coordinates] == train[0]["pair"]
    assert not pair.safe_merge_actions
    assert state.board.shape == (4, 4)


def test_development_and_untouched_sources_are_hash_only_in_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = training._load_selected_rows()
    real_restore = training.restore_train_root
    restored_roles = []

    def tracked_restore(row: dict):
        restored_roles.append(row["role"])
        return real_restore(row)

    monkeypatch.setattr(training, "restore_train_root", tracked_restore)
    audit = training._source_audit(rows)
    assert set(restored_roles) == {"train"}
    assert audit["development_content_opened"] is False
    assert audit["untouched_content_opened"] is False
    assert audit["sealed_source_files_hashed"] == 224


def test_relative_event_and_geometry_masking() -> None:
    geometry = {
        10: np.zeros(GEOMETRY_WIDTH),
        20: np.ones(GEOMETRY_WIDTH),
        40: np.full(GEOMETRY_WIDTH, 0.5),
    }
    first = build_decision_targets(
        decision_move=0,
        terminal_move=40,
        terminal_status="censor",
        live_geometry_by_move=geometry,
    )
    later = build_decision_targets(
        decision_move=10,
        terminal_move=40,
        terminal_status="censor",
        live_geometry_by_move=geometry,
    )
    assert first.event_mask and first.event_class == 4
    assert first.geometry_mask.tolist() == [True, True, True]
    assert not later.event_mask
    assert later.geometry_mask.tolist() == [True, False, False]


def test_geometry_normalization_and_lineage_pair() -> None:
    state = _state(
        [
            [1536, 48, 0, 0],
            [0, 0, 0, 0],
            [48, 0, 3, 0],
            [0, 1, 2, 0],
        ]
    )
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=10,
        slot_stream_id=11,
        starter_tile=1536,
    )
    row = {
        "target": 48,
    }
    pair = training.select_designated_pair(
        state.board,
        1536,
        requested_target=48,
        allowed_targets=(48,),
    )
    assert pair is not None
    lineage = training.initial_lineage(pair)
    values = training._normalized_geometry(state, sim, lineage, row["target"])
    assert values.shape == (GEOMETRY_WIDTH,)
    assert np.isfinite(values).all()
    assert ((0.0 <= values) & (values <= 1.0)).all()


def test_generate_episode_is_deterministic_and_does_not_use_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = [
        row for row in training._load_selected_rows()
        if row["role"] == "train"
    ]
    task = training._learning_rows()[0]
    first_arrays, first_meta = training.generate_episode(
        root_row=roots[0],
        task=task,
        model=None,
    )
    second_arrays, second_meta = training.generate_episode(
        root_row=roots[0],
        task=task,
        model=None,
    )
    assert first_meta == second_meta
    assert all(
        np.array_equal(first_arrays[name], second_arrays[name])
        for name in first_arrays
    )
    assert first_arrays["tokens"].shape[1:] == (16, 37)
    assert first_arrays["globals"].shape[1:] == (35,)
    assert first_arrays["geometry"].shape[1:] == (3, 8)


def test_episode_write_reload_and_task_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "EPISODE_DIR", tmp_path / "episodes")
    roots = [
        row for row in training._load_selected_rows()
        if row["role"] == "train"
    ]
    task = training._learning_rows()[0]
    arrays, metadata = training.generate_episode(
        root_row=roots[0],
        task=task,
        model=None,
    )
    training._write_episode(task, arrays, metadata)
    loaded, loaded_meta = training._load_episode(task)
    assert loaded_meta["root_cluster"] == roots[0]["root_cluster"]
    assert all(np.array_equal(arrays[name], loaded[name]) for name in arrays)
    changed = dict(task)
    changed["root_index"] = 1
    with pytest.raises((FileNotFoundError, ValueError)):
        training._load_episode(changed)


def test_episode_array_orphan_resumes_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "EPISODE_DIR", tmp_path / "episodes")
    task = training._learning_rows()[0]
    arrays = {
        "tokens": np.zeros((1, 16, 37), dtype=np.float32),
        "globals": np.zeros((1, 35), dtype=np.float32),
        "actions": np.asarray([0], dtype=np.int8),
    }
    metadata = {
        "root_index": int(task["root_index"]),
        "round_index": int(task["round_index"]),
        "replicate": int(task["replicate"]),
        "root_cluster": "fixture",
    }
    real_write = training._write_immutable

    def crash_before_metadata(*args, **kwargs):
        raise RuntimeError("simulated crash after array commit")

    monkeypatch.setattr(training, "_write_immutable", crash_before_metadata)
    with pytest.raises(RuntimeError, match="simulated crash"):
        training._write_episode(task, arrays, metadata)
    array_path, metadata_path = training._episode_paths(task)
    assert array_path.exists()
    assert not metadata_path.exists()
    orphan_sha = training.sha256_path(array_path)

    monkeypatch.setattr(training, "_write_immutable", real_write)
    training._write_episode(task, arrays, metadata)
    assert training.sha256_path(array_path) == orphan_sha
    loaded, loaded_metadata = training._load_episode(task)
    assert training._arrays_equal(loaded, arrays)
    assert loaded_metadata["root_cluster"] == "fixture"


def test_attempt_ledger_rejects_duplicate_close_and_post_close_resume() -> None:
    task = training._learning_rows()[0]
    task_id = training._task_id(task)
    opened = {
        "task_id": task_id,
        "status": "opened",
        "root_index": int(task["root_index"]),
        "round_index": int(task["round_index"]),
        "replicate": int(task["replicate"]),
        "stream_ids": {
            field: int(task[field]) for field in training.STREAM_FIELDS
        },
    }
    completed = {
        "task_id": task_id,
        "status": "completed",
        "array_sha256": "a",
        "metadata_payload_sha256": "b",
    }
    states = training._validate_attempt_ledger(
        [task],
        [opened, {"task_id": task_id, "status": "resumed_same_stream"}, completed],
    )
    assert states[task_id] == {
        "opened": True,
        "completed": True,
        "resume_count": 1,
    }
    with pytest.raises(ValueError, match="Duplicate or unopened close"):
        training._validate_attempt_ledger(
            [task],
            [opened, completed, completed],
        )
    with pytest.raises(ValueError, match="Invalid attempt resume"):
        training._validate_attempt_ledger(
            [task],
            [
                opened,
                completed,
                {"task_id": task_id, "status": "resumed_same_stream"},
            ],
        )


def test_runtime_completion_reconciles_only_forward() -> None:
    runtime = {"completed_tasks": 2, "active_seconds": 3.0}
    assert training._reconcile_runtime_completions(runtime, 3)
    assert runtime["completed_tasks"] == 3
    assert not training._reconcile_runtime_completions(runtime, 3)
    with pytest.raises(ValueError, match="exceeds attempt closes"):
        training._reconcile_runtime_completions(runtime, 2)


def test_fit_round_is_deterministic_on_fixed_fixture() -> None:
    rng = np.random.default_rng(7)
    rows = 16
    arrays = {
        "tokens": rng.normal(size=(rows, 16, 37)).astype(np.float32),
        "globals": rng.normal(size=(rows, 35)).astype(np.float32),
        "event_class": rng.integers(0, 5, size=rows, dtype=np.int8),
        "event_mask": np.ones(rows, dtype=np.bool_),
        "event_weight": np.full(rows, 1.0 / rows, dtype=np.float32),
        "geometry": rng.random(size=(rows, 3, 8)).astype(np.float32),
        "geometry_mask": np.ones((rows, 3), dtype=np.bool_),
        "geometry_weight": np.full((rows, 3), 1.0 / rows, dtype=np.float32),
    }
    model_a, optimizer_a = training._initialize_training()
    model_b, optimizer_b = training._initialize_training()
    training.fit_round(
        model=model_a,
        optimizer=optimizer_a,
        arrays=arrays,
        round_number=1,
    )
    training.fit_round(
        model=model_b,
        optimizer=optimizer_b,
        arrays=arrays,
        round_number=1,
    )
    assert all(
        torch.equal(model_a.state_dict()[name], model_b.state_dict()[name])
        for name in model_a.state_dict()
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model, optimizer = training._initialize_training()
    path = tmp_path / "checkpoint.pt"
    digest = training._save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        round_number=0,
        config_sha256="config",
    )
    loaded, loaded_optimizer, round_number = training._load_checkpoint(
        path,
        config_sha256="config",
    )
    assert digest == training.sha256_path(path)
    assert round_number == 0
    assert loaded_optimizer.state_dict() == optimizer.state_dict()
    assert all(
        torch.equal(model.state_dict()[name], loaded.state_dict()[name])
        for name in model.state_dict()
    )


def test_weighting_equalizes_family_root_trajectory_rows() -> None:
    value = 1.0 / (5 * 20 * 12 * 7)
    assert math.isclose(value * 7 * 12 * 20, 1.0 / 5)


def test_support_gate_decisions_are_disjoint() -> None:
    passing = {
        "successes_at_least_40": True,
        "six_successes_each_target": True,
        "four_families_three_successes": True,
        "failures_at_least_40": True,
        "censors_at_least_40": True,
        "finite_arrays": True,
        "two_nonempty_success_bins": True,
    }
    assert all(passing.values())
    for key in passing:
        changed = dict(passing)
        changed[key] = False
        assert not all(changed.values())


def test_cli_destinations_and_routing_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = training.build_parser().parse_args(
        [
            "write-test-evidence",
            "--focused",
            "1",
            "--regressions",
            "2",
            "--recorded-command",
            "x",
        ]
    )
    assert args.subcommand == "write-test-evidence"
    assert args.recorded_commands == ["x"]
    monkeypatch.setattr(
        training,
        "write_test_evidence",
        lambda **kwargs: {"kind": "evidence", **kwargs},
    )
    result = training.dispatch(
        [
            "write-test-evidence",
            "--focused",
            "1",
            "--regressions",
            "2",
            "--recorded-command",
            "x",
        ]
    )
    assert result["kind"] == "evidence"
    monkeypatch.setattr(
        training,
        "prepare",
        lambda out_dir: {"kind": "prepare", "out_dir": out_dir},
    )
    monkeypatch.setattr(
        training,
        "open_execution",
        lambda: {"kind": "open"},
    )
    monkeypatch.setattr(
        training,
        "execute",
        lambda: {"kind": "execute"},
    )
    for command in ("prepare", "open", "execute"):
        routed = training.dispatch(
            [command, "--out-dir", str(training.OUTPUT_DIR)]
        )
        assert routed["kind"] == command


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            training.O3TrainingOperationalHold("disk"),
            "HOLD_O3_TRAINING_OPERATIONAL",
        ),
        (ValueError("identity"), "KILL_O3_TRAINING_INTEGRITY"),
    ],
)
def test_execute_seals_operational_hold_or_integrity_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected: str,
) -> None:
    marker_path = tmp_path / "marker.json"
    marker_path.write_text("{}\n")
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(training, "MARKER_PATH", marker_path)
    monkeypatch.setattr(training, "RESULT_PATH", result_path)
    monkeypatch.setattr(
        training,
        "_load_marker",
        lambda: {"marker_payload_sha256": "marker"},
    )

    def fail(_stage: str):
        raise error

    monkeypatch.setattr(training, "_require_operational", fail)
    result = training.execute()
    sealed = json.loads(result_path.read_text())
    assert result["decision"] == expected
    assert sealed["decision"] == expected
    assert sealed["hold"] is (expected == "HOLD_O3_TRAINING_OPERATIONAL")
    assert sealed["kill"] is (expected == "KILL_O3_TRAINING_INTEGRITY")


def test_open_rejects_existing_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "MARKER_PATH", tmp_path / "marker.json")
    monkeypatch.setattr(training, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(training, "ATTEMPT_PATH", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(training, "RUNTIME_PATH", tmp_path / "runtime.json")
    monkeypatch.setattr(training, "EPISODE_DIR", tmp_path / "episodes")
    monkeypatch.setattr(training, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    (tmp_path / "attempts.jsonl").write_text("{}\n")
    monkeypatch.setattr(training, "_load_preflight_lock", lambda: {})
    with pytest.raises(ValueError, match="work exists before marker"):
        training.open_execution()


def test_authoritative_output_and_evidence_are_absent() -> None:
    assert not training.OUTPUT_DIR.exists()
    assert not training.TEST_EVIDENCE_PATH.exists()

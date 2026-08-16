import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from threes_rl import o5_training_v2 as training
from threes_rl.o4_domain_safe_pair_option import (
    EVENT_WIDTH,
    GEOMETRY_WIDTH,
    O4DesignatedPairNet,
    select_designated_pair,
)
from threes_rl.sim import SimState, preview_from_label


def _state() -> SimState:
    board = np.asarray(
        [
            [1536, 48, 0, 0],
            [0, 0, 0, 0],
            [48, 0, 3, 0],
            [0, 1, 2, 0],
        ],
        dtype=np.int32,
    )
    return SimState(
        board=board,
        preview=preview_from_label("blue"),
        small_counts={"red": 4, "blue": 3, "gray": 4},
        small_pos=1,
        small_seen_total=12,
        span_small_pos=3,
        large_pending=False,
        max_tile=1536,
        move_count=20,
        game_over=False,
    )


def _root() -> dict:
    state = _state()
    pair = select_designated_pair(
        state.board,
        1536,
        requested_target=48,
        allowed_targets=(48,),
    )
    assert pair is not None and not pair.safe_merge_actions
    return {
        "role": "train",
        "root_cluster": "fixture-root",
        "family": "o5_corner2",
        "target": 48,
        "pair": [list(value) for value in pair.coordinates],
    }


def _fake_episode_arrays(rows: int = 2) -> dict[str, np.ndarray]:
    return {
        "tokens": np.zeros((rows, 16, 37), dtype=np.float32),
        "globals": np.zeros((rows, 35), dtype=np.float32),
        "actions": np.zeros(rows, dtype=np.int8),
        "decision_moves": np.arange(rows, dtype=np.int8),
        "event_target": np.tile(
            np.asarray([[1, 0, 0, 0, 0]], dtype=np.float32),
            (rows, 1),
        ),
        "event_mask": np.ones(rows, dtype=np.bool_),
        "geometry": np.zeros(
            (rows, 3, GEOMETRY_WIDTH),
            dtype=np.float32,
        ),
        "geometry_mask": np.ones((rows, 3), dtype=np.bool_),
    }


def _optimizer_step(
    model: O4DesignatedPairNet,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()


def test_v1_drafts_are_preserved_exactly() -> None:
    audit = training.v1_draft_audit()
    assert audit["passes"]
    assert audit["authoritative"] is False
    assert {
        name: row["sha256"] for name, row in audit["files"].items()
    } == training.EXPECTED_V1_DRAFT_SHA256


def test_config_freezes_adaptive_sequence() -> None:
    config = training.training_config()
    assert config["schema_sha256"] == training.EXPECTED_SCHEMA_SHA256
    assert config["parameter_count"] == 102557
    assert config["torch_version"] == "2.12.1"
    assert config["trajectories_by_round"] == [2, 2, 1, 1]
    assert config["epsilon_by_round"] == [1.0, 0.15, 0.10, 0.05]
    assert config["episodes"] == 1152
    assert config["checkpoint_authority_before_support"] is False


def test_p0_inputs_and_selected_manifest_are_exact() -> None:
    audit = training.p0_input_audit()
    assert audit["passes"]
    assert audit["checks"]["p0_ready"]
    rows = training.load_selected_rows()
    assert len(rows) == 448
    assert sum(row["role"] == "train" for row in rows) == 192
    assert sum(row["role"] == "development" for row in rows) == 64
    assert (
        sum(row["role"] == "untouched_mechanism" for row in rows)
        == 192
    )


def test_dependency_audit_enforces_literal_o4_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sha = training.sha256_path
    operator = Path("threes_rl/o4_domain_safe_pair_option.py")

    def changed(path: Path) -> str:
        if Path(path) == operator:
            return "0" * 64
        return real_sha(Path(path))

    monkeypatch.setattr(training, "sha256_path", changed)
    with pytest.raises(
        training.O5TrainingIntegrityError,
        match="Frozen O5 dependency changed",
    ):
        training.dependency_audit()


def test_dependency_inventory_covers_every_p0_bound_source() -> None:
    marker = json.loads(training.P0_FILES["marker"].read_text())
    p0_dependencies = marker["dependency_hashes"]
    configured_paths = {str(path) for path in training.DEPENDENCY_PATHS}
    assert set(p0_dependencies) <= configured_paths
    assert set(p0_dependencies) <= set(training.EXPECTED_DEPENDENCY_SHA256)


def test_learning_manifest_is_exact_192_by_six() -> None:
    rows = training.learning_rows()
    assert len(rows) == 1152
    counts = {}
    for row in rows:
        key = int(row["round_index"])
        counts[key] = counts.get(key, 0) + 1
    assert counts == {1: 384, 2: 384, 3: 192, 4: 192}
    per_root = {}
    for row in rows:
        per_root.setdefault(int(row["root_index"]), []).append(
            int(row["trajectory_index"])
        )
    assert len(per_root) == 192
    assert all(sorted(values) == list(range(6)) for values in per_root.values())
    for field in training.STREAM_FIELDS:
        values = [int(row[field]) for row in rows]
        assert len(values) == len(set(values))


class _ForbiddenRow(dict):
    def __init__(self, *args, forbidden=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden = set(forbidden)

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"forbidden access: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"forbidden access: {key}")
        return super().get(key, default)


def test_source_audit_opens_train_only_and_holdouts_are_hash_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "TRAIN_ROOTS", 1)
    monkeypatch.setattr(training, "DEVELOPMENT_ROOTS", 1)
    monkeypatch.setattr(training, "UNTOUCHED_ROOTS", 1)
    train = {
        "role": "train",
        "root_cluster": "train",
        "family": "o5_corner2",
        "target": 48,
        "source_replay": "train.json",
        "source_replay_sha256": "train",
        "frame_index": 1,
        "o5_whitelisted_state_sha256": "state",
    }
    forbidden = (
        "target",
        "pair",
        "frame_index",
        "legal_count",
        "blocker_density",
        "state_sha1",
    )
    development = _ForbiddenRow(
        {
            "role": "development",
            "root_cluster": "dev",
            "source_replay": "dev.json",
            "source_replay_sha256": "dev",
        },
        forbidden=forbidden,
    )
    untouched = _ForbiddenRow(
        {
            "role": "untouched_mechanism",
            "root_cluster": "test",
            "source_replay": "test.json",
            "source_replay_sha256": "test",
        },
        forbidden=forbidden,
    )
    pair = select_designated_pair(
        _state().board,
        1536,
        requested_target=48,
        allowed_targets=(48,),
    )
    assert pair is not None
    restored = []

    def restore(row):
        restored.append(row["role"])
        return _state(), pair

    monkeypatch.setattr(training, "restore_train_root", restore)
    monkeypatch.setattr(
        training,
        "sha256_path",
        lambda path: Path(path).stem,
    )
    audit = training.source_audit([train, development, untouched])
    assert audit["passes"]
    assert restored == ["train"]
    assert audit["checks"]["development_content_unopened"]
    assert audit["checks"]["untouched_content_unopened"]


def test_whitelist_does_not_access_forbidden_fields() -> None:
    payload = _ForbiddenRow(
        {
            "board": _state().board.tolist(),
            "preview": {"kind": "blue"},
            "tile_cycle": {
                "small_counts": {"red": 4, "blue": 3, "gray": 4},
                "small_pos": 1,
                "small_seen_total": 12,
                "span_small_pos": 3,
                "large_pending": False,
            },
            "move_count": 20,
            "game_over": False,
            "score": "forbidden",
            "max_tile": "forbidden",
            "legal_actions": "forbidden",
            "move": "forbidden",
            "action": "forbidden",
            "outcome": "forbidden",
        },
        forbidden=(
            "score",
            "max_tile",
            "legal_actions",
            "move",
            "action",
            "outcome",
        ),
    )
    state, _identity = training.p0.whitelisted_state_payload(payload)
    assert state.max_tile == 1536


def test_round_one_rejects_model_and_later_rounds_require_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root()
    state = _state()
    pair = select_designated_pair(
        state.board,
        1536,
        requested_target=48,
        allowed_targets=(48,),
    )
    assert pair is not None
    monkeypatch.setattr(
        training,
        "restore_train_root",
        lambda _row: (copy.deepcopy(state), pair),
    )
    rows = training.learning_rows()
    round_one = next(row for row in rows if row["round_index"] == 1)
    round_two = next(row for row in rows if row["round_index"] == 2)
    model, _optimizer = training.initialize_training()
    with pytest.raises(
        training.O5TrainingIntegrityError,
        match="R1 must be uniform",
    ):
        training.generate_episode(
            root_row=root,
            task=round_one,
            model=model,
            collection_model_round=0,
        )
    with pytest.raises(
        training.O5TrainingIntegrityError,
        match="require the prior-round model",
    ):
        training.generate_episode(
            root_row=root,
            task=round_two,
            model=None,
            collection_model_round=1,
        )


def test_round_two_uses_model_for_exploitation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    pair = select_designated_pair(
        state.board,
        1536,
        requested_target=48,
        allowed_targets=(48,),
    )
    assert pair is not None
    monkeypatch.setattr(
        training,
        "restore_train_root",
        lambda _row: (copy.deepcopy(state), pair),
    )
    calls = 0
    real_outputs = training._model_outputs

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_outputs(*args, **kwargs)

    monkeypatch.setattr(training, "_model_outputs", tracked)
    task = next(
        row for row in training.learning_rows()
        if row["round_index"] == 2
    )
    model, _optimizer = training.initialize_training()
    arrays, metadata = training.generate_episode(
        root_row=_root(),
        task=task,
        model=model,
        collection_model_round=1,
    )
    assert calls > 0
    assert metadata["collection_model_round"] == 1
    assert math.isclose(metadata["epsilon"], 0.15)
    for name in ("tokens", "globals", "event_target", "geometry"):
        assert np.isfinite(arrays[name]).all()
        assert ((0.0 <= arrays[name]) & (arrays[name] <= 1.0)).all()


def test_load_round_state_uses_immediate_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, optimizer = training.initialize_training()
    payload = {
        "post_fit_model_sha256": training.model_state_sha256(model),
        "post_fit_optimizer_sha256":
            training.optimizer_state_sha256(optimizer),
    }
    calls = []

    def load(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        return model, optimizer, payload

    monkeypatch.setattr(training, "_load_checkpoint", load)
    monkeypatch.setattr(
        training,
        "sha256_path",
        lambda path: f"sha:{Path(path).name}",
    )
    for round_number in (2, 3, 4):
        loaded = training.load_round_state(
            round_number,
            config_sha256="config",
        )
        assert loaded[2] is model
        assert loaded[3] == round_number - 1
    assert [name for name, _kwargs in calls] == [
        "round_1_provisional.pt",
        "round_2_provisional.pt",
        "round_3_provisional.pt",
    ]
    assert calls[0][1]["expected_predecessor_sha256"] is None
    assert calls[1][1]["expected_predecessor_sha256"].endswith(
        "round_1_provisional.pt"
    )


def test_optimizer_state_continues_across_round_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "CHECKPOINT_DIR", tmp_path)
    model, optimizer = training.initialize_training()
    _optimizer_step(model, optimizer)
    pre_model = training.model_state_sha256(model)
    pre_optimizer = training.optimizer_state_sha256(optimizer)
    first = training._save_checkpoint(
        training._checkpoint_path(1),
        model=model,
        optimizer=optimizer,
        round_number=1,
        config_sha256="config",
        predecessor_file_sha256=None,
        pre_fit_model_sha256=pre_model,
        pre_fit_optimizer_sha256=pre_optimizer,
    )
    loaded_model, loaded_optimizer, first_payload = (
        training._load_checkpoint(
            training._checkpoint_path(1),
            config_sha256="config",
            expected_round=1,
            expected_predecessor_sha256=None,
        )
    )
    assert (
        training.optimizer_state_sha256(loaded_optimizer)
        == first_payload["post_fit_optimizer_sha256"]
    )
    round_two_pre_model = training.model_state_sha256(loaded_model)
    round_two_pre_optimizer = training.optimizer_state_sha256(
        loaded_optimizer
    )
    _optimizer_step(loaded_model, loaded_optimizer)
    second = training._save_checkpoint(
        training._checkpoint_path(2),
        model=loaded_model,
        optimizer=loaded_optimizer,
        round_number=2,
        config_sha256="config",
        predecessor_file_sha256=first["file_sha256"],
        pre_fit_model_sha256=round_two_pre_model,
        pre_fit_optimizer_sha256=round_two_pre_optimizer,
    )
    payload = torch.load(
        training._checkpoint_path(2),
        map_location="cpu",
        weights_only=False,
    )
    assert payload["pre_fit_model_sha256"] == first[
        "post_fit_model_sha256"
    ]
    assert payload["pre_fit_optimizer_sha256"] == first[
        "post_fit_optimizer_sha256"
    ]
    assert second["optimizer_step_count"] > first["optimizer_step_count"]


def test_episode_is_one_atomic_artifact_and_cannot_double_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "EPISODE_DIR", tmp_path / "episodes")
    task = training.learning_rows()[0]
    arrays = _fake_episode_arrays()
    metadata = {
        "version": f"{training.VERSION}_episode",
        "task_id": training._task_id(task),
        "root_index": int(task["root_index"]),
        "round_index": int(task["round_index"]),
        "trajectory_index": int(task["trajectory_index"]),
        "root_cluster": "fixture",
        "family": "o5_corner2",
        "target": 48,
        "terminal_status": "success",
        "terminal_move": 2,
        "decision_rows": 2,
        "collection_model_round": 0,
        "epsilon": 1.0,
        "stream_ids": {
            field: int(task[field]) for field in training.STREAM_FIELDS
        },
        "score_or_behavior_action_label_used": False,
    }
    artifact_sha, metadata_sha = training._write_episode_atomic(
        task,
        arrays,
        metadata,
    )
    assert artifact_sha == training.sha256_path(training._episode_path(task))
    loaded, loaded_metadata = training._load_episode(task)
    assert training._arrays_equal(loaded, arrays)
    assert loaded_metadata["metadata_payload_sha256"] == metadata_sha
    task_id = training._task_id(task)
    opened = {
        "task_id": task_id,
        "status": "opened",
        "root_index": int(task["root_index"]),
        "round_index": int(task["round_index"]),
        "trajectory_index": int(task["trajectory_index"]),
        "stream_ids": {
            field: int(task[field]) for field in training.STREAM_FIELDS
        },
    }
    closed = {
        "task_id": task_id,
        "status": "completed",
        "artifact_sha256": artifact_sha,
    }
    with pytest.raises(
        training.O5TrainingIntegrityError,
        match="Duplicate/unopened",
    ):
        training.validate_attempt_ledger(
            [task],
            [opened, closed, closed],
        )


def test_fit_is_deterministic_with_soft_event_targets() -> None:
    rng = np.random.default_rng(7)
    rows = 8
    event_class = rng.integers(0, EVENT_WIDTH, size=rows)
    arrays = {
        "tokens": rng.random((rows, 16, 37), dtype=np.float32),
        "globals": rng.random((rows, 35), dtype=np.float32),
        "event_target": np.eye(EVENT_WIDTH, dtype=np.float32)[event_class],
        "event_mask": np.ones(rows, dtype=np.bool_),
        "event_weight": np.full(rows, 1.0 / rows, dtype=np.float32),
        "geometry": rng.random(
            (rows, 3, GEOMETRY_WIDTH),
            dtype=np.float32,
        ),
        "geometry_mask": np.ones((rows, 3), dtype=np.bool_),
        "geometry_weight": np.full(
            (rows, 3),
            1.0 / rows,
            dtype=np.float32,
        ),
    }
    first_model, first_optimizer = training.initialize_training()
    second_model, second_optimizer = training.initialize_training()
    training.fit_cumulative_round(
        model=first_model,
        optimizer=first_optimizer,
        arrays=arrays,
        round_number=1,
    )
    training.fit_cumulative_round(
        model=second_model,
        optimizer=second_optimizer,
        arrays=arrays,
        round_number=1,
    )
    assert training.model_state_sha256(
        first_model
    ) == training.model_state_sha256(second_model)
    assert training.optimizer_state_sha256(
        first_optimizer
    ) == training.optimizer_state_sha256(second_optimizer)


def test_checkpoint_quarantine_on_support_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training,
        "CHECKPOINT_QUARANTINE_PATH",
        tmp_path / "quarantine.json",
    )
    monkeypatch.setattr(
        training,
        "CHECKPOINT_AUTHORITY_PATH",
        tmp_path / "authority.json",
    )
    checkpoints = [
        {
            "round_number": number,
            "path": f"round-{number}.pt",
            "file_sha256": str(number) * 64,
            "authoritative": False,
        }
        for number in range(1, 5)
    ]
    payload = training.seal_checkpoint_disposition(
        support_passes=False,
        checkpoints=checkpoints,
    )
    assert payload["candidate_round"] is None
    assert all(
        row["quarantined"] and not row["usable_downstream"]
        for row in payload["checkpoints"]
    )
    assert not training.CHECKPOINT_AUTHORITY_PATH.exists()


def test_checkpoint_authority_names_only_round_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training,
        "CHECKPOINT_AUTHORITY_PATH",
        tmp_path / "authority.json",
    )
    monkeypatch.setattr(
        training,
        "CHECKPOINT_QUARANTINE_PATH",
        tmp_path / "quarantine.json",
    )
    checkpoints = [
        {
            "round_number": number,
            "path": f"round-{number}.pt",
            "file_sha256": str(number) * 64,
            "authoritative": False,
        }
        for number in range(1, 5)
    ]
    payload = training.seal_checkpoint_disposition(
        support_passes=True,
        checkpoints=checkpoints,
    )
    assert payload["candidate_round"] == 4
    assert payload["candidate"]["round_number"] == 4
    assert [
        row["round_number"]
        for row in payload["provisional_non_candidates"]
    ] == [1, 2, 3]
    assert not training.CHECKPOINT_QUARANTINE_PATH.exists()


def test_support_gate_thresholds_are_exact() -> None:
    passing = {
        "successes_at_least_40": True,
        "six_successes_each_target": True,
        "three_successes_each_family": True,
        "failures_at_least_40": True,
        "true_h40_censors_at_least_40": True,
        "finite_arrays": True,
        "two_nonempty_success_bins": True,
    }
    assert all(passing.values())
    for key in passing:
        failed = dict(passing)
        failed[key] = False
        assert not all(failed.values())


def test_cli_destinations_are_distinct_and_no_command_runs(
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


def test_v2_execution_artifacts_are_absent() -> None:
    assert not training.TEST_EVIDENCE_PATH.exists()
    assert not training.OUTPUT_DIR.exists()

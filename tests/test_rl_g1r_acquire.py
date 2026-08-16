from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

import threes_rl.g1r_acquire as g1r
from threes_rl.record_replay import state_payload
from threes_rl.sim import ThreesSim


def _payload(board: list[list[int]], move_count: int = 0) -> dict:
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=101,
        slot_stream_id=103,
        starter_tile=1536,
    )
    state = sim.reset()
    state.board = np.asarray(board, dtype=np.int32)
    state.max_tile = int(state.board.max())
    state.move_count = move_count
    state.game_over = not bool(sim.legal_actions(state))
    return state_payload(state, sim)


def _candidate(
    index: int,
    *,
    family: str,
    stratum: str,
    role: str,
) -> dict:
    return {
        "record_id": f"record-{index:06d}",
        "root_cluster": f"{family}:fresh:{index + 50_000_000}:1536",
        "root_seed": index + 50_000_000,
        "behavior_family": family,
        "stratum": stratum,
        "role": role,
        "source_frame_index": index,
        "state_sha1": f"{index:040x}",
        "source_replay": f"replay-{index}.json",
        "source_replay_sha256": f"{index:064x}",
        "state": {"board": [[0] * 4 for _ in range(4)]},
    }


def _write_valid_lock(
    path: Path,
    *,
    bound_out_dir: Path,
    frozen_jobs: int = 1,
    policy_lock_sha256: str = "policy-lock",
) -> dict:
    family, policy = g1r.policy_slate()[0]
    stream_row = g1r.requested_stream_manifest(
        1,
        representative_families=[family],
    )[0]
    lock = {
        "preflight_ready": True,
        "identity": g1r._lock_identity(),
        "bound_out_dir": str(bound_out_dir.resolve()),
        "frozen_jobs": frozen_jobs,
        "required_minimum_nice": 0,
        "policy_locks": {"policy_lock_sha256": policy_lock_sha256},
        "action_distinctness_audit": {
            "representative_by_nominal_family": {family: family}
        },
        "representative_families": [family],
        "stream_rows": [stream_row],
        "stream_manifest_sha256": g1r.canonical_json_hash([stream_row]),
        "games_per_genuine_family": 1,
    }
    lock["preflight_payload_sha256"] = g1r.canonical_json_hash(lock)
    path.write_text(json.dumps(lock))
    return lock


def test_stream_namespaces_are_disjoint_and_deterministic() -> None:
    first = g1r.stream_ids(0, 0)
    second = g1r.stream_ids(1, 0)
    assert first == g1r.stream_ids(0, 0)
    assert len(set(first.values())) == 4
    assert not set(first.values()).intersection(second.values())
    manifest = g1r.requested_stream_manifest(2)
    assert len(manifest) == len(g1r.policy_slate()) * 2
    assert len(
        {
            row[key]
            for row in manifest
            for key in g1r.STREAM_BASES
        }
    ) == len(manifest) * len(g1r.STREAM_BASES)


def test_stream_collision_checks_logical_against_all_ancestry_aliases(
    monkeypatch,
) -> None:
    row = g1r.requested_stream_manifest(1)[0]
    prior = {
        "fresh_root_seed": {row["logical_seed"]},
        "deck_stream_id": {row["deck_stream_id"]},
    }
    monkeypatch.setattr(
        g1r,
        "historical_collision_union",
        lambda **_kwargs: (
            prior,
            {"matched_source_count": 1, "matched_sources_sha256": "history"},
        ),
    )
    audit = g1r.stream_collision_audit([row])
    assert not audit["zero_collisions"]
    assert audit["collisions"]["logical_seed"] == [row["logical_seed"]]
    assert audit["collisions"]["deck_stream_id"] == [row["deck_stream_id"]]


def test_source_role_uses_exactly_next_40_frames() -> None:
    starter_only = {"state": {"board": [[1536, 0, 0, 0], [0] * 4, [0] * 4, [0] * 4]}}
    promoted = {
        "state": {
            "board": [[1536, 1536, 0, 0], [0] * 4, [0] * 4, [0] * 4]
        }
    }
    frames = [starter_only for _ in range(42)]
    frames[40] = promoted
    assert g1r.source_role(frames, 0, 1536) == "source_success_window"
    frames[40] = starter_only
    frames[41] = promoted
    assert g1r.source_role(frames, 0, 1536) == "source_control"


def test_candidate_extraction_uses_hash_argmin_and_round_trips(tmp_path) -> None:
    first = _payload(
        [[1536, 768, 192, 48], [96, 24, 6, 3], [1, 2, 0, 0], [0] * 4],
        10,
    )
    second = _payload(
        [[1536, 768, 384, 96], [48, 12, 6, 3], [1, 2, 0, 0], [0] * 4],
        11,
    )
    replay = {
        "seed": 17,
        "frames": [
            {"index": 10, "state": first, "move": "up"},
            {"index": 11, "state": second, "move": "left"},
        ],
    }
    replay_path = tmp_path / "replay.json"
    replay_path.write_text("{}")
    candidates = g1r.extract_candidates(
        replay,
        family="test_family",
        replay_path=replay_path,
    )
    assert len(candidates) == 1
    rows = []
    for index, payload in ((10, first), (11, second)):
        state_hash = g1r.state_signature(payload, 1536)
        rows.append(
            (
                g1r.deterministic_key(
                    "G1R-state-v1",
                    "test_family:fresh:17:1536",
                    "pre1536",
                    index,
                    state_hash,
                ),
                index,
            )
        )
    assert candidates[0]["source_frame_index"] == min(rows)[1]
    assert candidates[0]["stratum"] == "pre1536"


def test_policy_distinctness_collapses_near_aliases(monkeypatch) -> None:
    families = [family for family, _spec in g1r.policy_slate()]
    actions = {
        families[0]: [0] * 64,
        families[1]: [1] + [0] * 63,
        families[2]: [index % 2 for index in range(64)],
        families[3]: [(index // 2) % 2 for index in range(64)],
        families[4]: [index % 4 for index in range(64)],
        families[5]: [(index * 3) % 4 for index in range(64)],
    }

    class Policy:
        def __init__(self, family: str) -> None:
            self.family = family

    monkeypatch.setattr(
        g1r,
        "deterministic_policy_action",
        lambda policy, payload: {
            "action": actions[policy.family][payload["panel_index"]],
            "exact_tie_count": 1,
        },
    )
    panel = {
        "panel_sha256": "panel",
        "records": [
            {
                "stratum": "pre1536" if index < 32 else "pre3072",
                "state": {"panel_index": index},
            }
            for index in range(64)
        ],
    }
    audit = g1r.audit_policy_distinctness(
        {family: Policy(family) for family in families},
        panel,
    )
    assert audit["passes"]
    assert audit["genuine_family_count"] == 5
    assert audit["alias_components"][0] == families[:2]
    assert audit["representative_by_nominal_family"][families[1]] == families[0]


def test_partition_allocator_is_exact_root_capped_and_family_capped() -> None:
    candidates = []
    index = 0
    for family_index in range(5):
        family = f"family_{family_index}"
        for stratum in g1r.STRATA:
            for role in g1r.ROLES:
                for _ in range(60):
                    candidates.append(
                        _candidate(
                            index,
                            family=family,
                            stratum=stratum,
                            role=role,
                        )
                    )
                    index += 1
    result = g1r.allocate_partition_manifest(candidates)
    assert result["ready"], result["deficits"]
    assert len(result["assignments"]) == 864
    assert len(
        {row["root_cluster"] for row in result["assignments"]}
    ) == 864
    for partition, targets in g1r.PARTITION_TARGETS.items():
        report = result["per_partition"][partition]
        assert report["roots"] == sum(targets.values())
        assert report["stratum_counts"] == targets
        assert report["family_share_max"] <= 0.40
        assert min(report["role_cell_counts"].values()) >= 10


def test_partition_allocator_fails_closed_on_missing_role_cells() -> None:
    candidates = [
        _candidate(
            index,
            family=f"family_{index % 5}",
            stratum="pre1536" if index % 2 == 0 else "pre3072",
            role="source_control",
        )
        for index in range(1000)
    ]
    result = g1r.allocate_partition_manifest(candidates)
    assert not result["ready"]
    assert not result["structure_checks"]["all_role_cells_at_least_10"]
    assert any(row["role"] == "source_success_window" for row in result["deficits"])


def test_completed_checkpoint_resume_and_duplicate_rejection(tmp_path) -> None:
    checkpoint = tmp_path / "completed.jsonl"
    row = {
        "nominal_family": "family",
        "game_index": 4,
        "candidates": [],
    }
    g1r._append_jsonl_row(checkpoint, row)
    assert g1r._load_completed(checkpoint) == {("family", 4): row}
    g1r._append_jsonl_row(checkpoint, row)
    with pytest.raises(ValueError, match="Duplicate"):
        g1r._load_completed(checkpoint)


def test_completed_rows_must_be_exact_stream_and_policy_subset(tmp_path) -> None:
    stream_row = g1r.requested_stream_manifest(1)[0]
    checkpoint = tmp_path / "completed.jsonl"
    row = {
        "nominal_family": stream_row["nominal_family"],
        "genuine_family": stream_row["nominal_family"],
        "game_index": stream_row["game_index"],
        "logical_seed": stream_row["logical_seed"],
        "deck_stream_id": stream_row["deck_stream_id"],
        "slot_stream_id": stream_row["slot_stream_id"],
        "policy_stream_id": stream_row["policy_stream_id"],
        "policy_spec_sha256": g1r.hashlib.sha256(
            stream_row["policy"].encode()
        ).hexdigest(),
        "candidates": [],
    }
    g1r._append_jsonl_row(checkpoint, row)
    mapping = {stream_row["nominal_family"]: stream_row["nominal_family"]}
    audit = g1r.completed_rows_subset_audit(
        completed_path=checkpoint,
        stream_rows=[stream_row],
        representative_map=mapping,
    )
    assert audit["passes"]
    row["deck_stream_id"] += 1
    checkpoint.unlink()
    g1r._append_jsonl_row(checkpoint, row)
    audit = g1r.completed_rows_subset_audit(
        completed_path=checkpoint,
        stream_rows=[stream_row],
        representative_map=mapping,
    )
    assert not audit["passes"]
    assert audit["mismatches"][0]["fields"]["deck_stream_id"]


def test_execution_guard_pauses_on_disk_budget(monkeypatch, tmp_path) -> None:
    DiskUsage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        g1r.shutil,
        "disk_usage",
        lambda _path: DiskUsage(200 * 1024**3, 101 * 1024**3, 99 * 1024**3),
    )
    with pytest.raises(g1r.AcquisitionPause) as error:
        g1r._execution_guard(
            out_dir=tmp_path,
            lock={"games_per_genuine_family": 20},
            runtime={"active_runtime_seconds": 0},
        )
    assert error.value.decision == "HOLD_G1R_BUDGET"


def test_execution_guard_converts_service_exception_to_hold(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(g1r, "_directory_bytes", lambda _path: 0)
    monkeypatch.setattr(
        g1r,
        "service_health",
        lambda: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    with pytest.raises(g1r.AcquisitionPause) as error:
        g1r._execution_guard(
            out_dir=tmp_path,
            lock={"games_per_genuine_family": 20},
            runtime={"active_runtime_seconds": 0},
        )
    assert error.value.decision == "HOLD_G1R_SERVICE"
    assert "ConnectionError" in error.value.reason


@pytest.mark.parametrize(
    ("different_out_dir", "runtime_jobs", "message"),
    (
        (True, 1, "bound to"),
        (False, 2, "differs from frozen"),
    ),
)
def test_run_rejects_wrong_directory_or_jobs_before_eval(
    monkeypatch,
    tmp_path,
    different_out_dir,
    runtime_jobs,
    message,
) -> None:
    out_dir = tmp_path / "bound"
    out_dir.mkdir()
    lock_path = tmp_path / "lock.json"
    _write_valid_lock(lock_path, bound_out_dir=out_dir)
    called = False

    def forbidden_eval(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("eval must not run")

    monkeypatch.setattr(g1r, "iter_eval_job_outputs", forbidden_eval)
    actual_out = tmp_path / "other" if different_out_dir else out_dir
    with pytest.raises(ValueError, match=message):
        g1r.run_acquisition(
            out_dir=actual_out,
            preflight_lock=lock_path,
            jobs=runtime_jobs,
        )
    assert not called


def test_resume_rejects_changed_policy_lock_before_eval(monkeypatch, tmp_path) -> None:
    out_dir = tmp_path / "bound"
    out_dir.mkdir()
    lock_path = tmp_path / "lock.json"
    _write_valid_lock(
        lock_path,
        bound_out_dir=out_dir,
        policy_lock_sha256="frozen",
    )
    called = False
    monkeypatch.setattr(
        g1r,
        "load_and_lock_policies",
        lambda: ({"policy_lock_sha256": "changed"}, {}),
    )
    monkeypatch.setattr(
        g1r,
        "stream_collision_audit",
        lambda *_args, **_kwargs: {"zero_collisions": True},
    )

    def forbidden_eval(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("eval must not run")

    monkeypatch.setattr(g1r, "iter_eval_job_outputs", forbidden_eval)
    with pytest.raises(ValueError, match="resume integrity failed"):
        g1r.run_acquisition(
            out_dir=out_dir,
            preflight_lock=lock_path,
            jobs=1,
        )
    assert not called


def test_resume_rejects_new_stream_collision_before_eval(
    monkeypatch,
    tmp_path,
) -> None:
    out_dir = tmp_path / "bound"
    out_dir.mkdir()
    lock_path = tmp_path / "lock.json"
    _write_valid_lock(
        lock_path,
        bound_out_dir=out_dir,
        policy_lock_sha256="frozen",
    )
    called = False
    monkeypatch.setattr(
        g1r,
        "load_and_lock_policies",
        lambda: ({"policy_lock_sha256": "frozen"}, {}),
    )
    monkeypatch.setattr(
        g1r,
        "stream_collision_audit",
        lambda *_args, **_kwargs: {
            "zero_collisions": False,
            "collisions": {"deck_stream_id": [42_000_000_000]},
        },
    )

    def forbidden_eval(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("eval must not run")

    monkeypatch.setattr(g1r, "iter_eval_job_outputs", forbidden_eval)
    with pytest.raises(ValueError, match="resume integrity failed"):
        g1r.run_acquisition(
            out_dir=out_dir,
            preflight_lock=lock_path,
            jobs=1,
        )
    assert not called


def test_retained_source_verifier_fails_closed(
    tmp_path,
) -> None:
    payload = _payload(
        [[1536, 768, 192, 48], [96, 24, 6, 3], [1, 2, 0, 0], [0] * 4],
        10,
    )
    replay_path = tmp_path / "replay.json"
    replay = {
        "frames": [{"index": 10, "state": payload, "move": "up"}],
    }
    replay_path.write_text(json.dumps(replay))
    candidate = {
        "source_replay": str(replay_path),
        "source_replay_sha256": g1r.sha256_path(replay_path),
        "source_frame_index": 10,
        "state_sha1": g1r.state_signature(payload, 1536),
        "state": payload,
    }
    assert g1r.verify_retained_sources([candidate])["passes"]

    missing = dict(candidate, source_replay=str(tmp_path / "missing.json"))
    assert not g1r.verify_retained_sources([missing])["passes"]

    replay_path.write_text(json.dumps({"frames": []}))
    assert not g1r.verify_retained_sources([candidate])["passes"]

    replay_path.write_text(json.dumps(replay))
    changed_state = dict(candidate, state={"different": True})
    changed_state["source_replay_sha256"] = g1r.sha256_path(replay_path)
    assert not g1r.verify_retained_sources([changed_state])["passes"]

    wrong_frame = dict(candidate, source_frame_index=11)
    wrong_frame["source_replay_sha256"] = g1r.sha256_path(replay_path)
    assert not g1r.verify_retained_sources([wrong_frame])["passes"]


def test_write_summary_cannot_emit_ready_when_source_integrity_fails(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        g1r,
        "allocate_partition_manifest",
        lambda _candidates: {"ready": True},
    )
    monkeypatch.setattr(
        g1r,
        "verify_retained_sources",
        lambda _candidates: {
            "checked_replays": 0,
            "checked_states": 0,
            "failures": [{"reason": "hash_mismatch"}],
            "passes": False,
        },
    )
    monkeypatch.setattr(
        g1r,
        "stream_collision_audit",
        lambda *_args, **_kwargs: {"zero_collisions": True},
    )
    summary = g1r._write_summary(
        out_dir=tmp_path,
        lock={
            "preflight_payload_sha256": "lock",
            "representative_families": [],
            "stream_manifest_sha256": "streams",
        },
        rows=[],
        runtime={"active_runtime_seconds": 0.0, "chunks_completed": 0},
    )
    assert summary["decision"] == "HOLD_G1R_INTEGRITY"
    assert not (tmp_path / "ready_partition_manifest.json").exists()


def test_write_summary_cannot_emit_ready_after_new_terminal_collision(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        g1r,
        "allocate_partition_manifest",
        lambda _candidates: {"ready": True},
    )
    monkeypatch.setattr(
        g1r,
        "verify_retained_sources",
        lambda _candidates: {
            "checked_replays": 1,
            "checked_states": 1,
            "failures": [],
            "passes": True,
        },
    )
    monkeypatch.setattr(
        g1r,
        "stream_collision_audit",
        lambda *_args, **_kwargs: {
            "zero_collisions": False,
            "collisions": {"logical_seed": [41_000_000_000]},
        },
    )
    summary = g1r._write_summary(
        out_dir=tmp_path,
        lock={
            "preflight_payload_sha256": "lock",
            "representative_families": [],
            "stream_manifest_sha256": "streams",
            "stream_rows": [],
        },
        rows=[],
        runtime={"active_runtime_seconds": 0.0, "chunks_completed": 0},
    )
    assert summary["decision"] == "HOLD_G1R_INTEGRITY"
    assert not (tmp_path / "ready_partition_manifest.json").exists()


def test_preflight_lock_is_immutable_and_binds_identity(monkeypatch, tmp_path) -> None:
    families = [family for family, _spec in g1r.policy_slate()]
    policies = {family: object() for family in families}
    distinctness = {
        "passes": True,
        "representative_families": families,
        "representative_by_nominal_family": {
            family: family for family in families
        },
    }
    monkeypatch.setattr(
        g1r,
        "load_and_lock_policies",
        lambda: ({"policy_lock_sha256": "policies"}, policies),
    )
    monkeypatch.setattr(
        g1r,
        "build_distinctness_panel",
        lambda: {"panel_sha256": "panel", "records": []},
    )
    monkeypatch.setattr(
        g1r,
        "audit_policy_distinctness",
        lambda _policies, _panel: distinctness,
    )
    monkeypatch.setattr(
        g1r,
        "stream_collision_audit",
        lambda *_args, **_kwargs: {
            "zero_collisions": True,
            "collisions": {},
        },
    )
    monkeypatch.setattr(
        g1r,
        "service_health",
        lambda: {"passes": True, "checks": {}},
    )
    monkeypatch.setattr(
        g1r,
        "replay_roundtrip_fixture",
        lambda: {"passes": True},
    )
    lock = g1r.create_preflight_lock(
        out_dir=tmp_path,
        games_per_family=1,
        lock_name="test",
        frozen_jobs=1,
    )
    path = tmp_path / "preflight_lock_test.json"
    assert lock["preflight_ready"]
    assert lock["bound_out_dir"] == str(tmp_path.resolve())
    assert lock["frozen_jobs"] == 1
    assert lock["required_minimum_nice"] == 10
    assert g1r._validate_preflight_lock(path)["identity"] == g1r._lock_identity()
    with pytest.raises(FileExistsError, match="immutable"):
        g1r.create_preflight_lock(
            out_dir=tmp_path,
            games_per_family=1,
            lock_name="test",
            frozen_jobs=1,
        )

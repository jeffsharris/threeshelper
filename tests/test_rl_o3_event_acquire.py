from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from threes_rl import o3_event_acquire as acquire


def _row(
    family: str = "o3_corner2",
    family_index: int = 0,
    game_index: int = 0,
    role: str = "train",
) -> dict:
    code = family_index * acquire.ROOTS_PER_FAMILY + game_index
    return {
        "purpose": "acquisition",
        "family": family,
        "family_index": family_index,
        "game_index": game_index,
        "role": role,
        "planned_root_id": f"planned-{family_index}-{game_index}",
        "logical_seed": 105_000_000_000 + code,
        "deck_stream_id": 106_000_000_000 + code,
        "slot_stream_id": 107_000_000_000 + code,
        "policy_stream_id": 108_000_000_000 + code,
    }


def _candidate(
    *,
    role: str,
    target: int,
    family: str,
    order: int,
) -> dict:
    root = f"{role}-{target}-{family}-{order}"
    return {
        "root_cluster": root,
        "family": family,
        "family_index": acquire.FAMILY_ORDER.index(family),
        "game_index": order,
        "role": role,
        "target": target,
        "frame_index": order,
        "state_sha1": f"{order:040x}",
        "selection_sha256": f"{order:064x}",
        "pair": [[0, 0], [1, 1]],
        "pair_manhattan": 2,
        "pair_chebyshev": 1,
        "pair_blockers": 0,
        "descriptive_stage": 1,
        "empty_count": 4,
        "legal_count": 3,
        "source_replay": f"fixture/{root}.json",
        "source_replay_sha256": "a" * 64,
    }


def _feasible_candidates() -> list[dict]:
    rows = []
    order = 1
    for role in acquire.ROLE_ORDER:
        for target in acquire.TARGET_ORDER:
            quota = acquire.TARGET_COUNTS[role][target]
            for index in range(quota + len(acquire.FAMILY_ORDER)):
                family = acquire.FAMILY_ORDER[index % len(acquire.FAMILY_ORDER)]
                rows.append(
                    _candidate(
                        role=role,
                        target=target,
                        family=family,
                        order=order,
                    )
                )
                order += 1
    return rows


def test_p0_artifacts_and_acquisition_rows_are_exact() -> None:
    audit = acquire._sealed_artifact_audit()
    assert audit["passes"]
    rows = acquire.acquisition_rows()
    assert len(rows) == 20_500
    assert Counter(row["family"] for row in rows) == {
        family: 4_100 for family in acquire.FAMILY_ORDER
    }
    assert Counter(row["role"] for row in rows) == acquire.p0.ROLE_COUNTS
    assert acquire.canonical_json_hash(rows) == acquire.canonical_json_hash(
        acquire.p0.acquisition_rows()
    )


def test_round_robin_chunks_are_one_game_per_family() -> None:
    chunks = acquire.round_robin_chunks(acquire.acquisition_rows())
    assert len(chunks) == 4_100
    assert all(len(chunk) == 5 for chunk in chunks)
    for game_index, chunk in enumerate(chunks):
        assert [row["family"] for row in chunk] == list(acquire.FAMILY_ORDER)
        assert {row["game_index"] for row in chunk} == {game_index}


def test_candidate_hash_binds_every_frozen_field() -> None:
    base = acquire._candidate_hash(
        role="train",
        target=192,
        family="o3_corner2",
        root="root",
        frame=7,
        state_hash="a" * 40,
    )
    assert base == acquire._candidate_hash(
        role="train",
        target=192,
        family="o3_corner2",
        root="root",
        frame=7,
        state_hash="a" * 40,
    )
    variants = [
        {"role": "development"},
        {"target": 96},
        {"family": "o3_expectimax2"},
        {"root": "other"},
        {"frame": 8},
        {"state_hash": "b" * 40},
    ]
    for change in variants:
        args = {
            "role": "train",
            "target": 192,
            "family": "o3_corner2",
            "root": "root",
            "frame": 7,
            "state_hash": "a" * 40,
            **change,
        }
        assert acquire._candidate_hash(**args) != base


def test_allocator_is_repeat_deterministic_root_unique_and_balanced() -> None:
    candidates = _feasible_candidates()
    first = acquire.allocate_candidates(candidates)
    second = acquire.allocate_candidates(list(reversed(candidates)))
    assert first["passes"]
    assert second["passes"]
    assert first["selected_manifest_sha256"] == second[
        "selected_manifest_sha256"
    ]
    assert len(first["selected"]) == 320
    roots = [row["root_cluster"] for row in first["selected"]]
    assert len(roots) == len(set(roots))
    for role, report in first["per_role"].items():
        assert report["passes"], role
        assert report["max_family_share"] <= 0.40
        assert len(report["family_counts"]) >= 4


def test_allocator_holds_on_target_shortfall() -> None:
    candidates = [
        row
        for row in _feasible_candidates()
        if not (
            row["role"] == "development"
            and row["target"] == 192
        )
    ]
    report = acquire.allocate_candidates(candidates)
    assert not report["passes"]
    assert {
        (row["role"], row["target"])
        for row in report["deficits"]
    } == {("development", 192)}


def test_support_scan_fails_closed_before_completion_barrier() -> None:
    with pytest.raises(ValueError, match="20,500"):
        acquire.scan_support([])
    with pytest.raises(ValueError, match="20,500"):
        acquire.scan_support([{} for _ in range(20_499)])


def test_descriptive_stage_is_never_an_eligibility_gate() -> None:
    separated = SimpleNamespace(
        safe_merge_actions=(),
        manhattan=3,
        chebyshev=2,
    )
    diagonal = SimpleNamespace(
        safe_merge_actions=(),
        manhattan=2,
        chebyshev=1,
    )
    adjacent = SimpleNamespace(
        safe_merge_actions=(),
        manhattan=1,
        chebyshev=1,
    )
    ready = SimpleNamespace(
        safe_merge_actions=(0,),
        manhattan=1,
        chebyshev=1,
    )
    assert [acquire._descriptive_stage(row) for row in (
        separated,
        diagonal,
        adjacent,
        ready,
    )] == [0, 1, 2, 3]


def test_attempt_id_is_deterministic_and_stream_bound() -> None:
    row = _row()
    first = acquire._attempt_id(row, 0)
    assert first == acquire._attempt_id(dict(row), 0)
    assert first != acquire._attempt_id(row, 1)
    changed = dict(row)
    changed["slot_stream_id"] += 1
    assert first != acquire._attempt_id(changed, 0)


def test_partial_replay_recovery_never_evaluates_or_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row()
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_path = replay_dir / "root.json"
    replay_path.write_text('{"fixture":true}\n')
    completion_path = tmp_path / "completed.jsonl"
    attempt_path = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(acquire, "REPLAY_DIR", replay_dir)
    monkeypatch.setattr(acquire, "COMPLETION_PATH", completion_path)
    monkeypatch.setattr(acquire, "ATTEMPT_PATH", attempt_path)
    monkeypatch.setattr(acquire, "_replay_path", lambda _row: replay_path)
    monkeypatch.setattr(
        acquire,
        "_load_attempt_ledger",
        lambda _rows: {
            acquire._stream_key(row): [
                {
                    "attempt_number": 0,
                    "attempt_id": "id",
                    "statuses": ["opened"],
                }
            ]
        },
    )
    completion = {
        "family": row["family"],
        "game_index": row["game_index"],
        "source_replay": str(replay_path),
    }
    monkeypatch.setattr(
        acquire,
        "_completion_from_replay",
        lambda *_args, **_kwargs: completion,
    )
    events = []
    monkeypatch.setattr(
        acquire,
        "_append_attempt_event",
        lambda *_args, **kwargs: events.append(kwargs) or {},
    )
    completions: dict[tuple[str, int], dict] = {}
    attempts = acquire._recover_existing_evidence([row], completions)
    assert completions[acquire._stream_key(row)] == completion
    assert attempts[acquire._stream_key(row)][-1]["statuses"] == [
        "opened",
        "completed_recovered",
    ]
    assert events == [
        {
            "attempt_number": 0,
            "status": "completed_recovered",
            "chunk_index": 0,
        }
    ]
    assert len(completion_path.read_text().splitlines()) == 1


def test_interrupted_attempt_is_explicit_before_same_stream_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row()
    key = acquire._stream_key(row)
    attempts = {
        key: [
            {
                "attempt_number": 0,
                "attempt_id": "id0",
                "statuses": ["opened"],
            }
        ]
    }
    events = []

    def append(
        _row: dict,
        *,
        attempt_number: int,
        status: str,
        chunk_index: int,
    ) -> dict:
        event = {
            "attempt_id": f"id{attempt_number}",
            "attempt_number": attempt_number,
            "status": status,
            "chunk_index": chunk_index,
        }
        events.append(event)
        return event

    monkeypatch.setattr(acquire, "_append_attempt_event", append)
    monkeypatch.setattr(acquire, "_replay_path", lambda _row: Path("/missing"))
    number = acquire._open_attempt(
        row,
        chunk_index=3,
        attempts=attempts,
    )
    assert number == 1
    assert [event["status"] for event in events] == [
        "interrupted_no_replay",
        "opened",
    ]
    assert attempts[key][0]["statuses"] == [
        "opened",
        "interrupted_no_replay",
    ]
    assert attempts[key][1]["statuses"] == ["opened"]


def test_evaluator_runtime_is_persisted_before_replay_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row()
    replay_path = tmp_path / "replay.json"
    replay_path.write_text("{}\n")
    runtime_path = tmp_path / "runtime.json"
    monkeypatch.setattr(acquire, "FAMILY_ORDER", ("o3_corner2",))
    monkeypatch.setattr(acquire, "ROOTS_PER_FAMILY", 1)
    monkeypatch.setattr(acquire, "TOTAL_ROOTS", 1)
    monkeypatch.setattr(acquire, "CHUNK_SIZE", 1)
    monkeypatch.setattr(acquire, "RUNTIME_PATH", runtime_path)
    monkeypatch.setattr(acquire, "REPLAY_DIR", tmp_path)
    monkeypatch.setattr(acquire, "O3_TO_G1R", {"o3_corner2": "g1r_corner2"})
    monkeypatch.setattr(acquire, "POLICY_SPECS", {"o3_corner2": "corner2"})
    monkeypatch.setattr(acquire, "acquisition_rows", lambda: [row])
    monkeypatch.setattr(acquire, "_load_completions", lambda: {})
    monkeypatch.setattr(
        acquire, "_verify_existing_completions", lambda *_args: None
    )
    monkeypatch.setattr(
        acquire, "_recover_existing_evidence", lambda *_args: {}
    )
    monkeypatch.setattr(acquire, "_guard_execution", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        acquire.policy_source,
        "load_policy",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        acquire,
        "iter_eval_job_outputs",
        lambda **_kwargs: iter([SimpleNamespace(index=0, replay={})]),
    )
    monkeypatch.setattr(
        acquire.history,
        "_append_jsonl_row",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        acquire,
        "_append_attempt_event",
        lambda _row, **kwargs: {
            "attempt_id": "id",
            "attempt_number": kwargs["attempt_number"],
        },
    )

    def store(_output: object, *, stream_row: dict) -> dict:
        state = json.loads(runtime_path.read_text())
        assert state["games_evaluated_charged"] == 1
        assert state["evaluation_batches_charged"] == 1
        return {
            "family": stream_row["family"],
            "game_index": stream_row["game_index"],
            "root_cluster": "fresh:105000000000:1536",
            "complete": True,
            "source_replay": str(replay_path),
        }

    monkeypatch.setattr(acquire, "_store_output", store)
    result = acquire.collect_all({}, jobs=1)
    assert len(result) == 1
    assert json.loads(runtime_path.read_text())["games_completed"] == 1


def test_attempt_ledger_audit_rejects_hidden_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row()
    completion = {"family": row["family"], "game_index": 0}
    monkeypatch.setattr(
        acquire,
        "_load_attempt_ledger",
        lambda _rows: {
            acquire._stream_key(row): [
                {
                    "attempt_number": 0,
                    "attempt_id": "a",
                    "statuses": ["opened", "interrupted_no_replay"],
                },
                {
                    "attempt_number": 1,
                    "attempt_id": "b",
                    "statuses": ["opened"],
                },
            ]
        },
    )
    monkeypatch.setattr(acquire, "sha256_path", lambda _path: "a" * 64)
    report = acquire.attempt_ledger_audit([row], [completion])
    assert not report["passes"]
    assert not report["checks"]["all_attempts_paired"]


def test_open_writes_only_zero_work_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    marker_path = out_dir / "O3_ACQUISITION_OPENED.json"
    monkeypatch.setattr(acquire, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(acquire, "MARKER_PATH", marker_path)
    monkeypatch.setattr(
        acquire, "_sealed_artifact_audit", lambda: {"passes": True}
    )
    monkeypatch.setattr(
        acquire, "_load_test_evidence", lambda: {"passes": True}
    )
    monkeypatch.setattr(
        acquire,
        "_load_policies",
        lambda: ({}, {"passes": True}),
    )
    monkeypatch.setattr(acquire, "acquisition_rows", lambda: [_row()])
    monkeypatch.setattr(
        acquire, "collision_audit", lambda *_args, **_kwargs: {"passes": True}
    )
    monkeypatch.setattr(
        acquire,
        "_operational_audit",
        lambda _out: {"passes": True, "free_gib": 150.0},
    )
    monkeypatch.setattr(
        acquire,
        "_marker_identity",
        lambda _out: {"version": "fixture", "bound_out_dir": str(out_dir.resolve())},
    )
    monkeypatch.setattr(acquire.history, "current_nice", lambda: 10)
    marker = acquire.open_execution(out_dir=out_dir, jobs=1)
    assert marker["decision"] == "O3_ACQUISITION_OPENED_ZERO_WORK"
    assert marker["zero_work"]["games"] == 0
    assert [path.name for path in out_dir.iterdir()] == [marker_path.name]


def test_load_marker_rejects_current_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    marker_path = out_dir / "O3_ACQUISITION_OPENED.json"
    result_path = out_dir / "result.json"
    monkeypatch.setattr(acquire, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(acquire, "MARKER_PATH", marker_path)
    monkeypatch.setattr(acquire, "RESULT_PATH", result_path)
    acquire._write_immutable_json(
        marker_path,
        {"version": "old"},
        self_hash_field="opened_payload_sha256",
    )
    monkeypatch.setattr(
        acquire, "_marker_identity", lambda _out: {"version": "new"}
    )
    with pytest.raises(ValueError, match="binding changed"):
        acquire._load_marker(out_dir=out_dir, jobs=1)


def test_collision_audit_detects_external_collision_without_parsing_o2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    p0_dir = runs / "p0"
    p0_dir.mkdir()
    (p0_dir / "reserved.json").write_text(
        '{"logical_seed":105000000000}\n'
    )
    o2_dir = runs / "o2"
    o2_dir.mkdir()
    (o2_dir / "forbidden.json").write_text(
        '{"logical_seed":105000000000}\n'
    )
    out_dir = runs / "out"
    external = runs / "external.json"
    external.write_text('{"logical_seed":105000000000}\n')
    monkeypatch.setattr(acquire, "P0_DIR", p0_dir)
    monkeypatch.setattr(acquire, "O2_FORBIDDEN_DIRS", (o2_dir,))
    report = acquire.collision_audit(
        [_row()],
        out_dir=out_dir,
        scan_root=runs,
    )
    assert not report["passes"]
    assert report["collisions"]["logical_seed"] == [105000000000]
    external.write_text('{"logical_seed":999}\n')
    report = acquire.collision_audit(
        [_row()],
        out_dir=out_dir,
        scan_root=runs,
    )
    assert report["passes"]
    assert any(
        row["classification"] == "immutable_o2_content_forbidden_unread"
        for row in report["excluded_sources"]
    )


def test_completion_loader_uses_o3_family_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "completed.jsonl"
    row = {"family": "o3_corner2", "game_index": 7}
    path.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(acquire, "COMPLETION_PATH", path)
    assert acquire._load_completions() == {("o3_corner2", 7): row}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        acquire._load_completions()

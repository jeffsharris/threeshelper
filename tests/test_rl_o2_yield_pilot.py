from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from threes_rl import o2_yield_pilot as pilot


def _candidate(
    root: str,
    family: str,
    target: int,
    stage: int,
    order: int,
) -> dict:
    return {
        "root_cluster": root,
        "family": family,
        "family_index": pilot.FAMILY_ORDER.index(family),
        "target": target,
        "stage": stage,
        "frame_index": order,
        "state_sha1": f"{order:040x}",
        "selection_sha256": f"{order:064x}",
        "pair": [[0, 0], [0, 1]],
        "safe_merge_actions": [],
        "anchor_safe": True,
        "air_safe": True,
        "empty_count": 4,
        "legal_count": 3,
        "source_replay": f"fixture/{root}.json",
        "source_replay_sha256": "a" * 64,
    }


def _feasible_structural_candidates() -> list[dict]:
    rows = []
    order = 1
    for target, stage in pilot._cell_order():
        if target == 768:
            family_counts = (3, 2, 2, 0)
        else:
            family_counts = (2, 1, 1, 0)
        for family_index, count in enumerate(family_counts):
            family = pilot.FAMILY_ORDER[family_index]
            for local in range(count):
                rows.append(
                    _candidate(
                        f"root-{target}-{stage}-{family_index}-{local}",
                        family,
                        target,
                        stage,
                        order,
                    )
                )
                order += 1
    return rows


def _availability_candidates() -> list[dict]:
    rows = []
    order = 1
    for target, stage in pilot._cell_order():
        family_counts = (3, 3, 2, 0) if target == 768 else (3, 2, 2, 0)
        for family_index, count in enumerate(family_counts):
            family = pilot.FAMILY_ORDER[family_index]
            for local in range(count):
                rows.append(
                    _candidate(
                        f"availability-{target}-{stage}-{family_index}-{local}",
                        family,
                        target,
                        stage,
                        order,
                    )
                )
                order += 1
    return rows


def test_execution_charter_and_preflight_bindings_are_exact() -> None:
    assert pilot.sha256_path(pilot.CHARTER_PATH) == pilot.CHARTER_SHA256
    audit = pilot.immutable_input_audit()
    assert audit["passes"]
    assert all(audit["checks"].values())
    assert pilot.FAMILY_ORDER == (
        "o2_corner2",
        "o2_expectimax2",
        "o2_parent_mc1000",
        "o2_qd_v2",
    )
    bound_sources = {str(path) for path in pilot.DEPENDENCY_SOURCE_PATHS}
    assert {
        "threes_rl/o1_geometry_option.py",
        "threes_rl/sim.py",
        "threes_rl/eval.py",
        "threes_rl/replay_provenance.py",
        "threes_rl/g1r_acquire.py",
        "threes_rl/g1r_acquire_v2_qd5.py",
        "threes_rl/g1r_qd_admission_v2.py",
    }.issubset(bound_sources)


def test_pilot_stream_rows_are_exact_and_reserved() -> None:
    rows = pilot.pilot_stream_rows()
    assert len(rows) == 128
    assert Counter(row["family"] for row in rows) == {
        family: 32 for family in pilot.FAMILY_ORDER
    }
    assert len(
        {
            int(row[field])
            for row in rows
            for field in pilot.preflight.STREAM_FIELDS
        }
    ) == 512
    assert pilot.canonical_json_hash(rows)


def test_round_robin_chunks_are_four_games_and_family_balanced() -> None:
    chunks = pilot.round_robin_chunks(pilot.pilot_stream_rows())
    assert len(chunks) == 32
    assert all(len(chunk) == 4 for chunk in chunks)
    assert [chunk[0]["family"] for chunk in chunks[:8]] == [
        *pilot.FAMILY_ORDER,
        *pilot.FAMILY_ORDER,
    ]
    for chunk in chunks:
        assert len({row["family"] for row in chunk}) == 1
        indices = [int(row["family_game_index"]) for row in chunk]
        assert indices == list(range(indices[0], indices[0] + 4))


def test_structural_objective_coefficients_are_exact_and_deterministic() -> None:
    first = pilot.structural_objective(7, 3)
    second = pilot.structural_objective(7, 3)
    np.testing.assert_array_equal(first, second)
    expected_edges = np.asarray(
        [
            (index + 1) / 8 + index / 8**3
            for index in range(7)
        ],
        dtype=float,
    )
    np.testing.assert_allclose(first[:7], expected_edges, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        first[7:],
        np.asarray((1e-9, 2e-9, 3e-9)),
        rtol=0.0,
        atol=1e-24,
    )
    assert np.all(np.diff(first[:7]) > 0)
    with pytest.raises(ValueError):
        pilot.structural_objective(-1, 1)


def test_structural_milp_is_repeat_deterministic_and_rechecks_constraints() -> None:
    candidates = _feasible_structural_candidates()
    first = pilot.solve_structural_match(candidates)
    second = pilot.solve_structural_match(list(reversed(candidates)))
    assert first["passes"]
    assert second["passes"]
    assert first["selected_manifest_sha256"] == second[
        "selected_manifest_sha256"
    ]
    assert first["objective_sha256"] == second["objective_sha256"]
    assert first["candidate_order_sha256"] == second[
        "candidate_order_sha256"
    ]
    assert first["checks"]["repeat_deterministic"]
    assert first["checks"]["post_rounding_constraints_exact"]
    assert first["checks"]["binary_residual_below_1e_7"]
    assert first["post_rounding_constraint_count"] > 0
    assert first["max_binary_residual"] <= 1e-7
    assert len(first["selected"]) == 92
    roots = [row["root_cluster"] for row in first["selected"]]
    assert len(roots) == len(set(roots))
    assert all(row["passes"] for row in first["cell_reports"])


def test_structural_milp_fails_closed_on_cell_shortfall() -> None:
    candidates = _feasible_structural_candidates()
    candidates = [
        row
        for row in candidates
        if not (row["target"] == 768 and row["stage"] == 0)
    ]
    report = pilot.solve_structural_match(candidates)
    assert not report["passes"]
    assert report["reason"] == "cell_candidate_shortfall"


def test_availability_layer_applies_three_per_family_cap() -> None:
    candidates = _availability_candidates()
    report = pilot.availability_report(candidates)
    assert report["passes"]
    assert len(report["cells"]) == 20
    for cell in report["cells"]:
        assert cell["passes"]
        assert max(cell["credited_family_counts"].values()) <= 3
        expected = 8 if cell["target"] == 768 else 7
        assert cell["credited_distinct_roots"] == expected
        assert cell["minimum_distinct_roots"] == expected
        assert cell["wilson_lower_at_minimum"] > cell["final_rate_required"]


def test_availability_layer_holds_on_missing_transfer_root() -> None:
    candidates = _availability_candidates()
    target_rows = [
        row
        for row in candidates
        if row["target"] == 768 and row["stage"] == 0
    ]
    candidates.remove(target_rows[-1])
    report = pilot.availability_report(candidates)
    cell = next(
        row
        for row in report["cells"]
        if row["target"] == 768 and row["stage"] == 0
    )
    assert not cell["passes"]
    assert not cell["checks"]["minimum_roots"]
    assert not report["passes"]


def test_scan_support_fails_closed_before_all_128_completions() -> None:
    with pytest.raises(ValueError, match="all 128"):
        pilot.scan_support([])
    with pytest.raises(ValueError, match="all 128"):
        pilot.scan_support([{} for _ in range(127)])


def test_support_analysis_rejects_duplicate_roots_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_scan(_completions: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("support scan must remain sealed")

    monkeypatch.setattr(pilot, "scan_support", forbidden_scan)
    rows = [{"root_cluster": "same"} for _ in range(128)]
    with pytest.raises(ValueError, match="128 unique"):
        pilot.support_analysis(rows)
    assert not called


def test_descriptive_1536_is_capped_and_never_a_readiness_gate() -> None:
    candidates = []
    order = 1
    for stage in pilot.STAGE_ORDER:
        for index in range(10):
            family = pilot.FAMILY_ORDER[index % len(pilot.FAMILY_ORDER)]
            candidates.append(
                _candidate(
                    f"late-{stage}-{index}",
                    family,
                    1536,
                    stage,
                    order,
                )
            )
            order += 1
    report = pilot.descriptive_1536(candidates)
    assert not report["readiness_gate"]
    assert report["selected_count"] == 16
    assert all(report["checks"].values())


def test_open_creates_only_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "pilot"
    marker_path = out_dir / "O2_YIELD_PILOT_EXECUTION_OPENED.json"
    monkeypatch.setattr(pilot, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(pilot, "MARKER_PATH", marker_path)
    monkeypatch.setattr(pilot, "ATTEMPT_PATH", out_dir / "attempts.jsonl")
    monkeypatch.setattr(pilot, "COMPLETION_PATH", out_dir / "completed.jsonl")
    monkeypatch.setattr(pilot, "RUNTIME_PATH", out_dir / "runtime.json")
    monkeypatch.setattr(pilot, "REPLAY_DIR", out_dir / "replays")
    monkeypatch.setattr(pilot, "SUPPORT_PATH", out_dir / "support.json")
    monkeypatch.setattr(pilot, "RESULT_PATH", out_dir / "result.json")
    monkeypatch.setattr(pilot, "immutable_input_audit", lambda: {"passes": True})
    monkeypatch.setattr(pilot, "_load_test_evidence", lambda: {"passes": True})
    monkeypatch.setattr(
        pilot.preflight,
        "family_evidence_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        pilot,
        "dependency_manifest",
        lambda: {"passes": True, "manifest_sha256": "dependency"},
    )
    monkeypatch.setattr(
        pilot,
        "collision_audit",
        lambda rows, out_dir: {"passes": True},
    )
    monkeypatch.setattr(
        pilot,
        "operational_audit",
        lambda out_dir: {"passes": True},
    )
    monkeypatch.setattr(
        pilot,
        "_marker_bindings",
        lambda dependencies=None: {"fixture": "exact"},
    )
    marker = pilot.open_execution(out_dir=out_dir, jobs=1)
    assert pilot.preflight.verify_payload_hash(marker)
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "O2_YIELD_PILOT_EXECUTION_OPENED.json"
    ]
    assert marker["zero_work"] == {
        "games": 0,
        "streams": 0,
        "attempt_rows": 0,
        "completion_rows": 0,
        "replays": 0,
        "support_scans": 0,
        "labels": 0,
        "model_fits": 0,
        "policy_outcomes": 0,
        "dashboard_changes": 0,
    }
    with pytest.raises(FileExistsError):
        pilot.open_execution(out_dir=out_dir, jobs=1)


def test_load_marker_rejects_missing_and_wrong_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "pilot"
    out_dir.mkdir()
    monkeypatch.setattr(pilot, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(
        pilot,
        "MARKER_PATH",
        out_dir / "O2_YIELD_PILOT_EXECUTION_OPENED.json",
    )
    monkeypatch.setattr(pilot, "RESULT_PATH", out_dir / "result.json")
    with pytest.raises(ValueError, match="jobs"):
        pilot._load_marker(out_dir=out_dir, jobs=2)
    with pytest.raises(FileNotFoundError, match="marker"):
        pilot._load_marker(out_dir=out_dir, jobs=1)


def test_load_marker_rejects_dependency_or_family_evidence_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "pilot"
    out_dir.mkdir()
    marker_path = out_dir / "marker.json"
    dependency = {"passes": True, "manifest_sha256": "dependency-a"}
    family = {"passes": True, "signature": "family-a"}
    current = {"dependency": dependency, "family": family}
    monkeypatch.setattr(pilot, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(pilot, "MARKER_PATH", marker_path)
    monkeypatch.setattr(pilot, "RESULT_PATH", out_dir / "result.json")
    monkeypatch.setattr(pilot, "pilot_stream_rows", lambda: [])
    monkeypatch.setattr(
        pilot,
        "dependency_manifest",
        lambda: current["dependency"],
    )
    monkeypatch.setattr(
        pilot.preflight,
        "family_evidence_audit",
        lambda: current["family"],
    )
    monkeypatch.setattr(
        pilot,
        "_marker_bindings",
        lambda dependencies=None: {
            "dependency": dependencies["manifest_sha256"]
        },
    )
    marker = {
        "version": pilot.VERSION,
        "decision": "OPENED_O2_YIELD_PILOT",
        "bound_out_dir": str(out_dir.resolve()),
        "bound_execute_command": pilot.EXECUTE_COMMAND,
        "jobs": 1,
        "bindings": {"dependency": "dependency-a"},
        "dependency_manifest": dependency,
        "family_evidence": family,
        "stream_rows": [],
        "stream_rows_sha256": pilot.canonical_json_hash([]),
    }
    pilot._atomic_new_json(marker_path, marker)
    assert pilot._load_marker(out_dir=out_dir, jobs=1)["decision"].startswith(
        "OPENED"
    )
    current["dependency"] = {
        "passes": True,
        "manifest_sha256": "dependency-b",
    }
    with pytest.raises(ValueError, match="binding mismatch"):
        pilot._load_marker(out_dir=out_dir, jobs=1)
    current["dependency"] = dependency
    current["family"] = {"passes": True, "signature": "family-b"}
    with pytest.raises(ValueError, match="binding mismatch"):
        pilot._load_marker(out_dir=out_dir, jobs=1)


def test_partial_chunk_resume_evaluates_only_missing_remainder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = "o2_corner2"
    rows = [
        {
            "family": family,
            "family_index": 0,
            "family_game_index": index,
            "logical_seed": 81_000_000_000 + index,
            "deck_stream_id": 82_000_000_000 + index,
            "slot_stream_id": 83_000_000_000 + index,
            "policy_stream_id": 84_000_000_000 + index,
        }
        for index in range(4)
    ]
    existing = {}
    for index in range(2):
        replay = tmp_path / f"existing-{index}.json"
        replay.write_text("{}")
        existing[(family, index)] = {
            "family": family,
            "nominal_family": family,
            "family_index": 0,
            "game_index": index,
            "source_replay": str(replay),
            "root_cluster": f"root-{index}",
            "complete": True,
        }
    generated_indices: list[int] = []
    events: list[tuple[str, dict | None]] = []
    clock = iter((100.0, 103.5))

    class Output:
        def __init__(self, index: int) -> None:
            self.index = index

    def fake_outputs(**kwargs: object) -> list[Output]:
        eval_jobs = kwargs["eval_jobs"]
        assert isinstance(eval_jobs, list)
        generated_indices.extend(job.index for job in eval_jobs)
        events.append(("evaluated", None))
        return [Output(job.index) for job in eval_jobs]

    def fake_store(output: Output, *, stream_row: dict) -> dict:
        events.append(("store", None))
        replay = tmp_path / f"generated-{stream_row['family_game_index']}.json"
        replay.write_text("{}")
        return {
            "family": family,
            "nominal_family": family,
            "family_index": 0,
            "game_index": int(stream_row["family_game_index"]),
            "source_replay": str(replay),
            "root_cluster": f"root-{stream_row['family_game_index']}",
            "complete": True,
        }

    monkeypatch.setattr(pilot, "TOTAL_ROOTS", 4)
    monkeypatch.setattr(pilot, "PILOT_ROOTS_PER_FAMILY", 4)
    monkeypatch.setattr(pilot, "FAMILY_ORDER", (family,))
    monkeypatch.setattr(pilot, "REPLAY_DIR", tmp_path / "replays")
    monkeypatch.setattr(pilot, "ATTEMPT_PATH", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(pilot, "COMPLETION_PATH", tmp_path / "completed.jsonl")
    for row in rows[:2]:
        attempt = {
            "identity": pilot._attempt_identity(
                row,
                chunk_index=0,
                attempt_index=0,
            ),
            "statuses": [],
        }
        pilot._append_attempt_status(attempt, "opened")
        pilot._append_attempt_status(attempt, "completed")
    monkeypatch.setattr(pilot, "_load_completions", lambda: dict(existing))
    monkeypatch.setattr(
        pilot,
        "_verify_existing_completions",
        lambda completions, stream_rows: None,
    )
    monkeypatch.setattr(
        pilot,
        "_runtime_state",
        lambda: {
            "active_runtime_seconds": 0.0,
            "chunks_completed": 0,
            "games_completed": 2,
        },
    )
    monkeypatch.setattr(pilot, "_guard_execution", lambda runtime: None)
    monkeypatch.setattr(
        pilot.family_source,
        "load_policy",
        lambda historical, spec: object(),
    )
    monkeypatch.setattr(pilot, "iter_eval_job_outputs", fake_outputs)
    monkeypatch.setattr(pilot, "_store_output", fake_store)
    monkeypatch.setattr(pilot.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        pilot,
        "write_json",
        lambda path, value: events.append(("runtime", dict(value))),
    )
    completions = pilot.collect_all({"stream_rows": rows}, jobs=1)
    assert generated_indices == [0, 1]
    generated_rows = [
        json.loads(line)
        for line in pilot.COMPLETION_PATH.read_text().splitlines()
    ]
    assert [row["game_index"] for row in generated_rows] == [2, 3]
    assert len(completions) == 4
    assert len({row["root_cluster"] for row in completions}) == 4
    assert [event[0] for event in events[:3]] == [
        "evaluated",
        "runtime",
        "store",
    ]
    first_runtime = events[1][1]
    assert first_runtime is not None
    assert first_runtime["active_runtime_seconds"] == pytest.approx(3.5)
    assert first_runtime["evaluation_batches_charged"] == 1
    assert first_runtime["games_evaluated_charged"] == 2
    assert first_runtime["chunks_completed"] == 0


def test_orphaned_replay_is_recovered_without_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "family": "o2_corner2",
        "family_index": 0,
        "family_game_index": 0,
    }
    replay_path = tmp_path / "orphan.json"
    replay_path.write_text("{}")
    completions: dict[tuple[str, int], dict] = {}
    recovered = {
        "family": "o2_corner2",
        "nominal_family": "o2_corner2",
        "family_index": 0,
        "game_index": 0,
        "source_replay": str(replay_path),
        "root_cluster": "root",
        "complete": True,
    }
    monkeypatch.setattr(pilot, "ATTEMPT_PATH", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(pilot, "COMPLETION_PATH", tmp_path / "completed.jsonl")
    monkeypatch.setattr(pilot, "_replay_path", lambda stream_row: replay_path)
    monkeypatch.setattr(
        pilot,
        "_completion_from_replay",
        lambda replay, replay_path, stream_row: recovered,
    )
    attempt = {
        "identity": pilot._attempt_identity(
            {
                **row,
                "logical_seed": 1,
                "deck_stream_id": 2,
                "slot_stream_id": 3,
                "policy_stream_id": 4,
            },
            chunk_index=0,
            attempt_index=0,
        ),
        "statuses": [],
    }
    pilot._append_attempt_status(attempt, "opened")
    attempts = {("o2_corner2", 0): [attempt]}
    pilot._reconcile_attempts_and_replays(completions, [row], attempts)
    assert completions[("o2_corner2", 0)] == recovered
    assert attempt["statuses"] == ["opened", "completed_recovered"]
    appended = [
        json.loads(line)
        for line in pilot.COMPLETION_PATH.read_text().splitlines()
    ]
    assert appended == [recovered]


def test_attempt_ledger_records_interruption_and_retry_without_hidden_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "family": "o2_corner2",
        "family_index": 0,
        "family_game_index": 0,
        "logical_seed": 81_000_000_000,
        "deck_stream_id": 82_000_000_000,
        "slot_stream_id": 83_000_000_000,
        "policy_stream_id": 84_000_000_000,
    }
    monkeypatch.setattr(pilot, "FAMILY_ORDER", ("o2_corner2",))
    monkeypatch.setattr(pilot, "PILOT_ROOTS_PER_FAMILY", 1)
    monkeypatch.setattr(pilot, "CHUNK_SIZE", 1)
    monkeypatch.setattr(pilot, "ATTEMPT_PATH", tmp_path / "attempts.jsonl")
    attempts: dict[tuple[str, int], list[dict]] = {}
    first = pilot._open_attempt(attempts, row, chunk_index=0)
    pilot._append_attempt_status(first, "interrupted_no_replay")
    second = pilot._open_attempt(attempts, row, chunk_index=0)
    pilot._append_attempt_status(second, "completed")
    loaded = pilot._load_attempt_ledger([row], [[row]])
    assert [attempt["statuses"] for attempt in loaded[("o2_corner2", 0)]] == [
        ["opened", "interrupted_no_replay"],
        ["opened", "completed"],
    ]
    assert (
        loaded[("o2_corner2", 0)][0]["identity"]["attempt_id"]
        != loaded[("o2_corner2", 0)][1]["identity"]["attempt_id"]
    )
    completion = {
        "family": "o2_corner2",
        "family_game_index": 0,
    }
    audit = pilot.attempt_ledger_audit([row], [completion])
    assert audit["passes"]
    assert audit["retries"] == 1


def test_attempt_ledger_rejects_retry_after_completed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "family": "o2_corner2",
        "family_index": 0,
        "family_game_index": 0,
        "logical_seed": 81_000_000_000,
        "deck_stream_id": 82_000_000_000,
        "slot_stream_id": 83_000_000_000,
        "policy_stream_id": 84_000_000_000,
    }
    monkeypatch.setattr(pilot, "ATTEMPT_PATH", tmp_path / "attempts.jsonl")
    events = []
    for attempt_index in (0, 1):
        identity = pilot._attempt_identity(
            row,
            chunk_index=0,
            attempt_index=attempt_index,
        )
        events.extend(
            (
                {**identity, "status": "opened"},
                {**identity, "status": "completed"},
            )
        )
    pilot.ATTEMPT_PATH.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        )
    )
    with pytest.raises(ValueError, match="retry hides"):
        pilot._load_attempt_ledger([row], [[row]])


def test_post_collection_runtime_guard_blocks_support_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "pilot"
    out_dir.mkdir()
    result_path = out_dir / "result.json"
    marker_path = out_dir / "marker.json"
    pilot._atomic_new_json(marker_path, {"version": "fixture"})
    calls = 0
    support_called = False

    def guard(_runtime: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise pilot.history.AcquisitionPause(
                "HOLD_O2_PILOT_OPERATIONAL_INTEGRITY",
                "active runtime reached six hours",
            )

    def forbidden_support(_completions: object) -> object:
        nonlocal support_called
        support_called = True
        raise AssertionError("support must remain sealed")

    completions = [
        {"root_cluster": f"root-{index}"} for index in range(128)
    ]
    monkeypatch.setattr(pilot, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(pilot, "RESULT_PATH", result_path)
    monkeypatch.setattr(pilot, "MARKER_PATH", marker_path)
    monkeypatch.setattr(
        pilot,
        "_load_marker",
        lambda out_dir, jobs: {"stream_rows": []},
    )
    monkeypatch.setattr(pilot, "immutable_input_audit", lambda: {"passes": True})
    monkeypatch.setattr(
        pilot.preflight,
        "family_evidence_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        pilot,
        "collision_audit",
        lambda rows, out_dir: {"passes": True},
    )
    monkeypatch.setattr(
        pilot,
        "_runtime_state",
        lambda: {
            "active_runtime_seconds": pilot.ACTIVE_RUNTIME_LIMIT,
            "chunks_completed": 32,
            "games_completed": 128,
        },
    )
    monkeypatch.setattr(pilot, "_guard_execution", guard)
    monkeypatch.setattr(
        pilot,
        "collect_all",
        lambda marker, jobs: completions,
    )
    monkeypatch.setattr(pilot, "support_analysis", forbidden_support)
    monkeypatch.setattr(pilot, "_load_completions", lambda: {})
    monkeypatch.setattr(
        pilot.preflight,
        "artifact_identity",
        lambda path: {"path": str(path)},
    )
    result = pilot.execute(out_dir=out_dir, jobs=1)
    assert result["decision"] == "HOLD_O2_PILOT_OPERATIONAL_INTEGRITY"
    assert calls == 2
    assert not support_called


def test_selection_hash_changes_with_every_bound_field() -> None:
    baseline = pilot._selection_hash(
        stage=1,
        target=384,
        family=pilot.FAMILY_ORDER[0],
        root="root",
        frame=10,
        state_hash="state",
    )
    changes = (
        {"stage": 2},
        {"target": 768},
        {"family": pilot.FAMILY_ORDER[1]},
        {"root": "other"},
        {"frame": 11},
        {"state_hash": "other"},
    )
    base = {
        "stage": 1,
        "target": 384,
        "family": pilot.FAMILY_ORDER[0],
        "root": "root",
        "frame": 10,
        "state_hash": "state",
    }
    for change in changes:
        assert pilot._selection_hash(**{**base, **change}) != baseline


def test_forbidden_work_report_is_exactly_zero() -> None:
    assert pilot._terminal_zero_forbidden_work() == {
        "corpus_games_beyond_pilot": 0,
        "option_rollouts": 0,
        "labels": 0,
        "model_fits": 0,
        "policy_outcome_evaluations": 0,
        "score_outcomes_inspected": 0,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
    }

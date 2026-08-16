from __future__ import annotations

import argparse
import ast
import json
import math
import queue
import inspect
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from threes_rl import j1_execution_surface as j1_execution
from threes_rl import j2_exact_teacher_feasibility_pilot_v2 as pilot


def _parallel_report(
    *,
    p99: float = 0.02,
    calls_per_second: float = 100.0,
    count: int = pilot.CENTRAL_COUNT,
) -> dict[str, object]:
    return {
        "timing_summary": {
            "count": count,
            "median_seconds": p99 / 2,
            "p90_seconds": p99 * 0.9,
            "p99_seconds": p99,
            "max_seconds": p99,
            "mean_seconds": p99 / 2,
        },
        "calls_per_second": calls_per_second,
        "record_count": count,
        "wall_seconds": count / calls_per_second,
        "summed_worker_peak_rss_bytes": 800,
        "max_contemporaneous_parent_children_rss_bytes": 900,
    }


def _phase_projection_inputs(
    *,
    p99: float = 0.02,
    calls_per_second: float = 100.0,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    central = {
        "serial": {"parent_peak_rss_bytes": 100},
        "parallel_eight_process": _parallel_report(
            p99=p99,
            calls_per_second=calls_per_second,
        ),
    }
    sensitivity = {
        "serial": {"parent_peak_rss_bytes": 100},
        "parallel_eight_process": _parallel_report(
            p99=p99,
            calls_per_second=calls_per_second,
            count=pilot.INVENTORY_COUNT,
        ),
    }
    sync = {
        "calls_per_second": calls_per_second,
        "summed_worker_peak_rss_bytes": 800,
        "max_contemporaneous_parent_children_rss_bytes": 900,
    }
    return central, sensitivity, sync


def _worker_records(
    *,
    expected_indices: list[int],
    round_id: str = "round",
) -> list[dict[str, object]]:
    rows = []
    for worker_id in range(pilot.WORKERS):
        indices = [
            index
            for index in expected_indices
            if index % pilot.WORKERS == worker_id
        ]
        rows.append(
            {
                "kind": "result",
                "round_id": round_id,
                "worker_id": worker_id,
                "indices": indices,
                "actions": [index % 4 for index in indices],
                "timings": [0.001 for _index in indices],
                "cpu_seconds": 0.01,
                "peak_rss_bytes": 100 + worker_id,
                "warmup_calls": 0,
            }
        )
    return rows


def _ready_records() -> list[dict[str, object]]:
    return [
        {
            "worker_id": worker_id,
            "load_seconds": 0.1,
        }
        for worker_id in range(pilot.WORKERS)
    ]


def _all_mapping_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_all_mapping_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.extend(_all_mapping_keys(child))
    return keys


def test_prefix_contract_is_the_frozen_broad_distribution() -> None:
    assert pilot.PREFIX_MIN == 16
    assert pilot.PREFIX_SPAN == 160
    assert pilot.PREFIX_MULTIPLIER == 73
    assert pilot.PREFIX_OFFSET == 19
    rows = pilot.engineering_stream_rows()
    assert len(rows) == 5_000
    assert sorted(
        int(row["target_prefix_steps"]) for row in rows[:160]
    ) == list(range(16, 176))
    assert pilot.PREFIX_BANDS == (
        ("16-55", 16, 55),
        ("56-95", 56, 95),
        ("96-135", 96, 135),
        ("136-175", 136, 175),
    )
    charter = pilot.CHARTER_PATH.read_text(encoding="utf-8")
    assert "`16 + ((73*i + 19) mod 160)`" in charter
    assert "`8 + ((73*i + 19) mod 32)`" not in charter


def test_engineering_stream_authority_is_disjoint_and_exact() -> None:
    report = pilot.stream_authority_audit()
    assert report["passes"]
    assert report["row_count"] == 5_000
    assert report["intervals"] == {
        "deck": [253_000_000_000, 253_000_004_999],
        "slot": [254_000_000_000, 254_000_004_999],
        "exploration": [255_000_000_000, 255_000_004_999],
    }
    assert report["checks"]["no_spent_213b_226b_collision"]
    assert report["checks"]["no_j2_227b_249b_collision"]
    assert report["checks"]["no_v1_250b_252b_collision"]


def test_reachable_state_manifest_is_feature_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Sim:
        starter_tile = None

        def legal_actions(self, current: object) -> list[int]:
            return [] if getattr(current, "game_over") else [0]

        def step(self, current: object, action: int) -> tuple[object, object]:
            assert action == 0
            return (
                SimpleNamespace(
                    game_over=False,
                    board=[[0] * 4 for _ in range(4)],
                    step=int(getattr(current, "step")) + 1,
                ),
                SimpleNamespace(moved=True),
            )

    initial = SimpleNamespace(
        game_over=False,
        board=[[0] * 4 for _ in range(4)],
        step=0,
    )
    monkeypatch.setattr(
        pilot.j1,
        "normal_start_sim",
        lambda **_kwargs: (Sim(), initial),
    )
    monkeypatch.setattr(
        pilot.j1,
        "simulator_snapshot",
        lambda _sim, current: {
            "game_over": current.game_over,
            "board": current.board,
            "step": current.step,
        },
    )
    row = min(
        pilot.engineering_stream_rows(),
        key=lambda value: int(value["target_prefix_steps"]),
    )
    manifest, snapshot = pilot._state_from_engineering_row(row)
    assert manifest["target_prefix_steps"] == 16
    assert manifest["realized_prefix_steps"] == 16
    assert manifest["prefix_clamped"] is False
    assert manifest["state_sha256"] == pilot.canonical_hash(snapshot)
    assert set(manifest) == {
        "state_index",
        "root_id",
        "ancestry_id",
        "worker_id",
        "target_prefix_steps",
        "streams",
        "realized_prefix_steps",
        "prefix_clamped",
        "feature_family",
        "state_sha256",
        "legal_action_count",
    }
    pilot.assert_no_forbidden_retained_fields(manifest)


def test_natural_early_termination_retains_final_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(game_over=False, board=[[0] * 4 for _ in range(4)])
    terminal = SimpleNamespace(
        game_over=True,
        board=[[0] * 4 for _ in range(4)],
    )

    class Sim:
        starter_tile = None

        def legal_actions(self, current: object) -> list[int]:
            return [] if getattr(current, "game_over") else [0]

        def step(self, current: object, action: int) -> tuple[object, object]:
            assert action == 0
            return terminal, SimpleNamespace(moved=True)

    monkeypatch.setattr(
        pilot.j1,
        "normal_start_sim",
        lambda **_kwargs: (Sim(), state),
    )
    monkeypatch.setattr(
        pilot.j1,
        "simulator_snapshot",
        lambda _sim, current: {
            "game_over": current.game_over,
            "board": current.board,
        },
    )
    manifest, snapshot = pilot._state_from_engineering_row(
        {
            "state_index": 0,
            "root_id": "r",
            "ancestry_id": "a",
            "worker_id": 0,
            "target_prefix_steps": 16,
            "streams": {
                "deck_stream_id": 1,
                "slot_stream_id": 2,
                "exploration_policy_stream_id": 3,
            },
        }
    )
    assert manifest["realized_prefix_steps"] == 0
    assert manifest["prefix_clamped"] is True
    assert manifest["legal_action_count"] == 1
    assert snapshot["game_over"] is False
    assert "termination_reason" not in manifest
    assert "terminal_state" not in manifest


def test_duplicate_inventory_state_hash_fails_before_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pilot, "INVENTORY_COUNT", 2)
    rows = [
        {
            "state_index": index,
            "root_id": f"root-{index}",
            "ancestry_id": f"ancestry-{index}",
            "worker_id": index,
            "target_prefix_steps": 16,
            "streams": {
                "deck_stream_id": index,
                "slot_stream_id": 10 + index,
                "exploration_policy_stream_id": 20 + index,
            },
        }
        for index in range(2)
    ]
    monkeypatch.setattr(pilot, "engineering_stream_rows", lambda: rows)

    def duplicate(
        row: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        return (
            {
                **row,
                "realized_prefix_steps": 16,
                "prefix_clamped": False,
                "feature_family": "low_air",
                "state_sha256": "same",
                "legal_action_count": 2,
            },
            {"same": True},
        )

    monkeypatch.setattr(pilot, "_state_from_engineering_row", duplicate)
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="inventory failed",
    ):
        pilot.build_and_seal_inventory(output_dir=tmp_path)
    assert not (tmp_path / pilot.INVENTORY_NAME).exists()


def test_all_roots_are_retained_once_with_exact_clamp_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "state_index": index,
            "root_id": f"root-{index}",
            "ancestry_id": f"ancestry-{index}",
            "worker_id": index,
            "target_prefix_steps": target,
            "streams": {
                "deck_stream_id": 100 + index,
                "slot_stream_id": 200 + index,
                "exploration_policy_stream_id": 300 + index,
            },
        }
        for index, target in enumerate((16, 56, 96, 136))
    ]
    families = (
        "low_air",
        "low_constrained",
        "mid_progression",
        "upper_progression",
    )
    clamped = (False, True, False, True)
    calls: list[int] = []

    def synthesize(
        row: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        index = int(row["state_index"])
        calls.append(index)
        target = int(row["target_prefix_steps"])
        snapshot = {"state_index": index, "live": True}
        return (
            {
                **row,
                "realized_prefix_steps": target - int(clamped[index]),
                "prefix_clamped": clamped[index],
                "feature_family": families[index],
                "state_sha256": pilot.canonical_hash(snapshot),
                "legal_action_count": 2,
            },
            snapshot,
        )

    monkeypatch.setattr(pilot, "INVENTORY_COUNT", len(rows))
    monkeypatch.setattr(pilot, "engineering_stream_rows", lambda: rows)
    monkeypatch.setattr(pilot, "_state_from_engineering_row", synthesize)
    inventory, snapshots = pilot.build_and_seal_inventory(
        output_dir=tmp_path,
    )
    assert calls == [0, 1, 2, 3]
    assert len(snapshots) == len(rows)
    assert [row["root_id"] for row in inventory["rows"]] == [
        row["root_id"] for row in rows
    ]
    assert inventory["checks"]["all_planned_roots_retained_once"]
    assert inventory["checks"]["roots_replaced_zero"]
    assert inventory["checks"]["roots_dropped_zero"]
    assert inventory["checks"]["survival_conditioning_zero"]
    assert inventory["clamp_summary"]["total"] == {
        "root_count": 4,
        "clamped_root_count": 2,
        "clamping_rate": 0.5,
    }
    for band, expected_clamped in zip(
        ("16-55", "56-95", "96-135", "136-175"),
        (0, 1, 0, 1),
        strict=True,
    ):
        assert inventory["clamp_summary"]["by_target_prefix_band"][band] == {
            "root_count": 1,
            "clamped_root_count": expected_clamped,
            "clamping_rate": float(expected_clamped),
        }
    for family, expected_clamped in zip(
        families,
        (0, 1, 0, 1),
        strict=True,
    ):
        assert inventory["clamp_summary"]["by_feature_family"][family] == {
            "root_count": 1,
            "clamped_root_count": expected_clamped,
            "clamping_rate": float(expected_clamped),
        }


def test_root_substitution_fails_instead_of_replacing_clamped_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "state_index": index,
            "root_id": f"root-{index}",
            "ancestry_id": f"ancestry-{index}",
            "worker_id": index,
            "target_prefix_steps": 16,
            "streams": {
                "deck_stream_id": index,
                "slot_stream_id": 10 + index,
                "exploration_policy_stream_id": 20 + index,
            },
        }
        for index in range(2)
    ]

    def substitute(
        row: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        index = int(row["state_index"])
        snapshot = {"state_index": index}
        return (
            {
                **row,
                "root_id": (
                    "replacement" if index == 1 else row["root_id"]
                ),
                "realized_prefix_steps": 15 if index == 1 else 16,
                "prefix_clamped": index == 1,
                "feature_family": "low_air",
                "state_sha256": pilot.canonical_hash(snapshot),
                "legal_action_count": 2,
            },
            snapshot,
        )

    monkeypatch.setattr(pilot, "INVENTORY_COUNT", len(rows))
    monkeypatch.setattr(pilot, "engineering_stream_rows", lambda: rows)
    monkeypatch.setattr(pilot, "_state_from_engineering_row", substitute)
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="inventory failed",
    ):
        pilot.build_and_seal_inventory(output_dir=tmp_path)
    assert not (tmp_path / pilot.INVENTORY_NAME).exists()


def test_target_prefix_bands_are_total_and_fail_outside_contract() -> None:
    assert pilot.target_prefix_band(16) == "16-55"
    assert pilot.target_prefix_band(55) == "16-55"
    assert pilot.target_prefix_band(56) == "56-95"
    assert pilot.target_prefix_band(95) == "56-95"
    assert pilot.target_prefix_band(96) == "96-135"
    assert pilot.target_prefix_band(135) == "96-135"
    assert pilot.target_prefix_band(136) == "136-175"
    assert pilot.target_prefix_band(175) == "136-175"
    with pytest.raises(pilot.PilotIntegrityError):
        pilot.target_prefix_band(15)
    with pytest.raises(pilot.PilotIntegrityError):
        pilot.target_prefix_band(176)


def test_v1_history_is_hash_bound_and_immutable() -> None:
    audit = pilot.v1_history_audit()
    assert audit["passes"]
    assert audit["adjudication"] == "HOLD_J2_TEACHER_PILOT_V1_PREFLIGHT"
    assert audit["observed_files"] == [pilot.V1_TEST_EVIDENCE_NAME]
    assert audit["checks"]["v1_inventory_absent"]
    assert audit["checks"]["v1_lock_absent"]
    assert audit["checks"]["v1_result_absent"]
    assert audit["checks"]["v1_marker_absent"]
    assert audit["checks"]["v1_terminal_absent"]
    assert audit["checks"]["v1_retention_absent"]


def test_worker_command_loop_warms_once_then_measures_zero_warmups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_query(
        _policy: object,
        snapshot: dict[str, int],
        _stream: int,
    ) -> tuple[int, float]:
        calls.append(snapshot["index"])
        return snapshot["index"] % 4, 0.001

    monkeypatch.setattr(pilot, "_query_one", fake_query)
    commands: queue.Queue[dict[str, object]] = queue.Queue()
    results: queue.Queue[dict[str, object]] = queue.Queue()
    row = {
        "state_index": 0,
        "snapshot": {"index": 0},
        "policy_stream_id": 9,
    }
    commands.put(
        {
            "kind": "warmup",
            "round_id": "warmup",
            "rows": [row],
        }
    )
    commands.put(
        {
            "kind": "query",
            "round_id": "round-0",
            "rows": [row],
        }
    )
    commands.put(
        {
            "kind": "query",
            "round_id": "round-1",
            "rows": [row],
        }
    )
    commands.put({"kind": "stop"})
    pilot._serve_teacher_commands(
        worker_id=0,
        command_queue=commands,
        result_queue=results,
        policy=object(),
    )
    warmup = results.get_nowait()
    first = results.get_nowait()
    second = results.get_nowait()
    assert warmup["kind"] == "warmup_complete"
    assert warmup["warmup_calls"] == pilot.WARMUP_CALLS == 8
    assert warmup["actions_retained"] == 0
    assert first["warmup_calls"] == second["warmup_calls"] == 0
    assert len(calls) == 8 + 1 + 1


def test_worker_rejects_measurement_before_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot,
        "_query_one",
        lambda *_args: (0, 0.001),
    )
    commands: queue.Queue[dict[str, object]] = queue.Queue()
    commands.put(
        {
            "kind": "query",
            "round_id": "round-0",
            "rows": [
                {
                    "state_index": 0,
                    "snapshot": {},
                    "policy_stream_id": 1,
                }
            ],
        }
    )
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="preceded explicit warmup",
    ):
        pilot._serve_teacher_commands(
            worker_id=0,
            command_queue=commands,
            result_queue=queue.Queue(),
            policy=object(),
        )


def test_parallel_workload_places_warmup_before_measured_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Group:
        def __init__(self, _binding: object) -> None:
            self.lifetime_rss_evidence = None

        def warmup(self, *, rows: object) -> dict[str, object]:
            events.append("warmup")
            return {
                "warmup_calls_per_process": 8,
                "total_worker_warmup_calls": 64,
                "warmup_wall_seconds": 9.0,
                "actions_retained": 0,
            }

        def run_round(
            self,
            *,
            round_id: str,
            rows: object,
        ) -> dict[str, object]:
            events.append("measured")
            return {"round_id": round_id, "warmup_calls": 0}

        def close(self) -> None:
            events.append("close")
            self.lifetime_rss_evidence = {
                "maximum_contemporaneous_parent_children_rss_bytes": 900,
                "sample_count": 7,
                "covers_load_warmup_queries_and_shutdown": True,
            }

    monkeypatch.setattr(pilot, "TeacherWorkerGroup", Group)
    report = pilot.run_parallel_workload(
        binding={},
        rows=[{"state_index": index} for index in range(8)],
        round_id="central",
    )
    assert events == ["warmup", "measured", "close"]
    assert report["warmup_calls"] == 0
    assert report["warmup_evidence"]["total_worker_warmup_calls"] == 64


def test_worker_record_validation_is_canonical_and_warmup_free() -> None:
    indices = list(range(16))
    report = pilot.validate_worker_records(
        _worker_records(expected_indices=indices),
        expected_indices=indices,
        round_id="round",
        wall_seconds=1.0,
        ready_records=_ready_records(),
        startup_and_load_wall_seconds=2.0,
        dispatch_monotonic_ns=10,
        received_monotonic_ns=20,
        lifetime_contemporaneous_peak_rss_bytes=999,
        lifetime_rss_sample_count=3,
        measured_round_contemporaneous_peak_rss_bytes=777,
        measured_round_rss_sample_count=2,
    )
    assert report["actions"] == [index % 4 for index in indices]
    assert report["warmup_calls"] == 0
    assert report["record_count"] == 16


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "cross_round", "wrong_order", "illegal", "warmup"),
)
def test_worker_record_corruption_fails_closed(mutation: str) -> None:
    indices = list(range(16))
    records = _worker_records(expected_indices=indices)
    if mutation == "duplicate":
        records[1]["worker_id"] = 0
    elif mutation == "cross_round":
        records[0]["round_id"] = "late"
    elif mutation == "wrong_order":
        records[0]["indices"] = list(reversed(records[0]["indices"]))
    elif mutation == "illegal":
        records[0]["actions"][0] = 7
    else:
        records[0]["warmup_calls"] = 8
    with pytest.raises(pilot.PilotIntegrityError):
        pilot.validate_worker_records(
            records,
            expected_indices=indices,
            round_id="round",
            wall_seconds=1.0,
            ready_records=_ready_records(),
                startup_and_load_wall_seconds=2.0,
                dispatch_monotonic_ns=10,
                received_monotonic_ns=20,
                lifetime_contemporaneous_peak_rss_bytes=999,
                lifetime_rss_sample_count=3,
                measured_round_contemporaneous_peak_rss_bytes=777,
                measured_round_rss_sample_count=2,
        )


def test_public_cost_evidence_has_only_aggregate_actions_digest() -> None:
    serial = {
        "actions": [0, 1],
        "ordered_output_sha256": "a" * 64,
        "wall_seconds": 2.0,
        "record_count": 2,
        "warmup_calls": 8,
    }
    parallel = {
        "actions": [0, 1],
        "timings": [0.1, 0.1],
        "ordered_output_sha256": "a" * 64,
        "wall_seconds": 1.0,
        "record_count": 2,
        "warmup_calls": 0,
        "warmup_evidence": {
            "total_worker_warmup_calls": 64,
            "warmup_wall_seconds": 5.0,
        },
    }
    public = pilot._public_cost_result(
        workload="central",
        serial=serial,
        parallel=parallel,
        reference_digest=None,
    )
    assert public["passes"]
    assert public["warmup_calls_per_process"] == 8
    assert public["parallel_worker_warmup_calls_total"] == 64
    assert public["parallel_measured_warmup_calls"] == 0
    assert public["warmups_excluded_from_steady_wall"]
    keys = _all_mapping_keys(public)
    assert "action" not in keys
    assert "actions" not in keys
    assert "teacher_action" not in keys
    assert "teacher_actions" not in keys
    assert public["actions_retained"] == 0
    assert public["labels_retained"] == 0
    pilot.assert_no_forbidden_retained_fields(public)


def test_sync_public_contract_reports_one_warmup_barrier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pilot, "SYNC_ROUNDS", 2)
    monkeypatch.setattr(pilot, "SYNC_STATES_PER_ROUND", 8)
    monkeypatch.setattr(pilot, "SYNC_COUNT", 16)
    monkeypatch.setattr(
        j1_execution,
        "load_bound_incumbent_policy",
        lambda _binding: object(),
    )
    monkeypatch.setattr(
        pilot,
        "_query_one",
        lambda _policy, snapshot, _stream: (
            int(snapshot["index"]) % 4,
            0.001,
        ),
    )

    class Group:
        def __init__(self, _binding: object) -> None:
            self.startup_and_load_wall_seconds = 3.0
            self.ready = _ready_records()
            self.warmup_count = 0
            self.lifetime_rss_evidence = None

        def __enter__(self) -> "Group":
            return self

        def __exit__(self, *_args: object) -> None:
            self.lifetime_rss_evidence = {
                "maximum_contemporaneous_parent_children_rss_bytes": 1_100,
                "sample_count": 9,
                "covers_load_warmup_queries_and_shutdown": True,
            }
            return None

        def warmup(self, *, rows: object) -> dict[str, object]:
            self.warmup_count += 1
            return {
                "total_worker_warmup_calls": 64,
                "warmup_wall_seconds": 4.0,
            }

        def run_round(
            self,
            *,
            round_id: str,
            rows: list[dict[str, object]],
        ) -> dict[str, object]:
            assert self.warmup_count == 1
            actions = [
                int(row["snapshot"]["index"]) % 4 for row in rows
            ]
            return {
                "actions": actions,
                "ordered_output_sha256": pilot.canonical_hash(
                    [
                        (int(row["state_index"]), action)
                        for row, action in zip(rows, actions, strict=True)
                    ]
                ),
                "timings": [0.001] * len(rows),
                "timing_summary": pilot._timing_summary(
                    [0.001] * len(rows)
                ),
                "wall_seconds": 0.01,
                "child_cpu_seconds": 0.02,
                "dispatch_monotonic_ns": time.monotonic_ns(),
                "received_monotonic_ns": time.monotonic_ns(),
                "summed_worker_peak_rss_bytes": 800,
                "max_contemporaneous_parent_children_rss_bytes": 900,
                "warmup_calls": 0,
            }

    monkeypatch.setattr(pilot, "TeacherWorkerGroup", Group)
    rows = [
        {
            "state_index": index,
            "snapshot": {"index": index},
            "policy_stream_id": 1_000 + index,
        }
        for index in range(16)
    ]
    public = pilot.run_synchronous_orchestration(
        binding={},
        rows=rows,
        output_dir=tmp_path,
    )
    assert public["record_count"] == 16
    assert public["warmup_calls_per_process"] == 8
    assert public["total_sync_worker_warmup_calls"] == 64
    assert public["serial_reference_warmup_calls"] == 8
    assert public["measured_round_warmup_calls"] == [0, 0]
    assert public["all_measured_round_warmups_zero"]
    assert public["warmups_excluded_from_steady_wall"]


def test_central_p99_contract_is_frozen_before_timing() -> None:
    contract = pilot.central_p99_admission_contract()
    assert contract["total_teacher_calls"] == 10_240 * 512
    assert contract["worker_concurrency_divisor"] == 8
    assert contract["safety_multiplier"] == 1.25
    assert contract["runtime_cap_hours"] == 72.0
    expected = (
        (
            72.0 / 1.25 - pilot.OPTIMIZER_FIXTURE_HOURS
        )
        * 3600.0
        * 8
        / (10_240 * 512)
    )
    assert contract["maximum_admissible_p99_seconds"] == pytest.approx(
        expected,
        abs=1e-15,
    )


def test_central_projection_uses_p99_not_aggregate_throughput() -> None:
    contract = pilot.central_p99_admission_contract()
    slow_central, slow_sensitivity, slow_sync = _phase_projection_inputs(
        p99=0.02,
        calls_per_second=10.0,
    )
    fast_central, fast_sensitivity, fast_sync = _phase_projection_inputs(
        p99=0.02,
        calls_per_second=10_000.0,
    )
    slow = pilot._project_phase_costs(
        central_public=slow_central,
        sensitivity_public=slow_sensitivity,
        sync_public=slow_sync,
        preflight_available_memory_bytes=64 * 1024**3,
        output_bytes=0,
        admission_contract=contract,
    )
    fast = pilot._project_phase_costs(
        central_public=fast_central,
        sensitivity_public=fast_sensitivity,
        sync_public=fast_sync,
        preflight_available_memory_bytes=64 * 1024**3,
        output_bytes=0,
        admission_contract=contract,
    )
    assert slow["central"]["runtime_hours_with_25pct_margin"] == (
        fast["central"]["runtime_hours_with_25pct_margin"]
    )
    assert slow["central"]["observed_calls_per_second"] != (
        fast["central"]["observed_calls_per_second"]
    )
    assert slow["checks"][
        "aggregate_throughput_recomputes_from_measured_wall"
    ]
    assert fast["checks"][
        "aggregate_throughput_recomputes_from_measured_wall"
    ]


def test_central_p99_gate_and_observed_margin_are_exact() -> None:
    contract = pilot.central_p99_admission_contract()
    maximum = float(contract["maximum_admissible_p99_seconds"])
    under = _phase_projection_inputs(p99=maximum * 0.99)
    over = _phase_projection_inputs(p99=maximum * 1.01)
    under_report = pilot._project_phase_costs(
        central_public=under[0],
        sensitivity_public=under[1],
        sync_public=under[2],
        preflight_available_memory_bytes=64 * 1024**3,
        output_bytes=0,
        admission_contract=contract,
    )
    over_report = pilot._project_phase_costs(
        central_public=over[0],
        sensitivity_public=over[1],
        sync_public=over[2],
        preflight_available_memory_bytes=64 * 1024**3,
        output_bytes=0,
        admission_contract=contract,
    )
    assert under_report["checks"][
        "pretraining_p99_meets_derived_ceiling"
    ]
    assert under_report["central"]["observed_p99_margin_seconds"] > 0
    assert under_report["central"]["observed_p99_margin_ratio"] > 1
    assert not over_report["checks"][
        "pretraining_p99_meets_derived_ceiling"
    ]
    assert over_report["central"]["observed_p99_margin_seconds"] < 0


def test_projection_rejects_changed_preflight_p99_contract() -> None:
    central, sensitivity, sync = _phase_projection_inputs()
    changed = dict(pilot.central_p99_admission_contract())
    changed["worker_concurrency_divisor"] = 7
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="admission contract changed",
    ):
        pilot._project_phase_costs(
            central_public=central,
            sensitivity_public=sensitivity,
            sync_public=sync,
            preflight_available_memory_bytes=64 * 1024**3,
            output_bytes=0,
            admission_contract=changed,
        )


def test_open_marker_has_one_truthful_inventory_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pilot, "INVENTORY_COUNT", 3)
    inventory = {
        "version": "fixture",
        "rows": [{"state_index": index} for index in range(3)],
        "passes": True,
    }
    pilot.write_immutable(
        tmp_path / pilot.INVENTORY_NAME,
        inventory,
        field="inventory_payload_sha256",
    )
    inventory_identity = pilot.immutable_identity(
        tmp_path / pilot.INVENTORY_NAME,
        "inventory_payload_sha256",
    )
    pilot.write_immutable(
        tmp_path / pilot.TEST_EVIDENCE_NAME,
        {"version": "fixture"},
        field="test_evidence_payload_sha256",
    )
    sources = {
        "passes": True,
        "local_sources": {"runner": "r"},
        "teacher_binding": {"incumbent_binding_sha256": "t"},
    }
    lock = {
        "decision": pilot.READY_PREFLIGHT,
        "inventory": inventory_identity,
        "source_audit": sources,
        "execution_command": "fixture",
        "central_p99_admission_contract": (
            pilot.central_p99_admission_contract()
        ),
    }
    pilot.write_immutable(
        tmp_path / pilot.PREFLIGHT_LOCK_NAME,
        lock,
        field="preflight_lock_payload_sha256",
    )
    result = {
        "decision": pilot.READY_PREFLIGHT,
        "inventory": inventory_identity,
        "execution_authorized": True,
    }
    pilot.write_immutable(
        tmp_path / pilot.PREFLIGHT_RESULT_NAME,
        result,
        field="preflight_result_payload_sha256",
    )
    monkeypatch.setattr(pilot, "source_identity_audit", lambda: sources)
    monkeypatch.setattr(
        pilot,
        "stream_authority_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        pilot,
        "operational_audit",
        lambda **_kwargs: {"passes": True},
    )
    marker = pilot.open_execution(output_dir=tmp_path)
    assert marker["sealed_inventory_states_before_marker"] == 3
    assert marker["inventory_states_before_marker"] == 3
    assert marker["teacher_loads_before_marker"] == 0
    assert marker["teacher_queries_before_marker"] == 0


def test_power_grid_calls_only_the_accepted_helper_with_changed_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, int]] = []

    def fake_power(
        *,
        n_pairs: int,
        datasets: int,
        bootstraps: int,
    ) -> dict[str, object]:
        calls.append((n_pairs, datasets, bootstraps))
        power = (
            0.6432291666666666
            if n_pairs == 2_048
            else min(0.99, 0.7 + n_pairs / 20_000)
        )
        return {
            "worst_case_primary_power": power,
            "rows": [
                {
                    "primary_gate_power": power,
                    "monte_carlo_standard_error": 0.01,
                    "control_rate": 0.02,
                    "coupling": 0.0,
                }
            ],
        }

    monkeypatch.setattr(pilot.j2, "common_or_power_grid", fake_power)
    report = pilot.run_power_sizing()
    assert calls == [
        (n_pairs, 768, 199) for n_pairs in pilot.POWER_N_GRID
    ]
    assert report["checks"]["published_n2048_reproduced"]
    assert report["method"]["control_rates"] == list(
        pilot.j2.CONTROL_RATES
    )
    assert report["method"]["couplings"] == list(
        pilot.j2.PAIRING_COUPLINGS
    )


def test_forbidden_retention_fields_fail_closed() -> None:
    for key in (
        "action",
        "actions",
        "score",
        "final_score",
        "trajectory",
        "policy_outcome",
    ):
        with pytest.raises(pilot.PilotIntegrityError):
            pilot.assert_no_forbidden_retained_fields({key: 1})
    pilot.assert_no_forbidden_retained_fields(
        {
            "ordered_output_sha256": "a" * 64,
            "actions_retained": 0,
            "scores_retained": 0,
        }
    )


def test_cli_and_import_surface_have_no_training_command_or_side_effect() -> None:
    parser = pilot.build_parser()
    actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(actions) == 1
    assert set(actions[0].choices) == {
        "audit-zero-work",
        "write-test-evidence",
        "prepare",
        "open",
        "execute",
    }
    for forbidden in ("train", "distill", "evaluate", "promote", "reserve"):
        with pytest.raises(SystemExit):
            parser.parse_args([forbidden])
    tree = ast.parse(pilot.RUNNER_PATH.read_text(encoding="utf-8"))
    top_level_calls = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        top_level_calls.extend(
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        )
    names = {
        (
            call.func.id
            if isinstance(call.func, ast.Name)
            else call.func.attr
            if isinstance(call.func, ast.Attribute)
            else ""
        )
        for call in top_level_calls
    }
    assert not (
        names
        & {
            "load_bound_incumbent_policy",
            "normal_start_sim",
            "prepare",
            "open_execution",
            "execute",
        }
    )


def test_zero_work_audit_has_no_teacher_or_science_counters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pilot.j2, "FUTURE_EXECUTION_DIRS", ())
    report = pilot.zero_work_audit(
        output_dir=tmp_path / "absent",
        include_operational=False,
    )
    assert report["passes"]
    assert report["teacher_queries"] == 0
    assert report["actions_retained"] == 0
    assert report["labels"] == 0
    assert report["games"] == 0
    assert report["outcomes"] == 0
    assert report["j2_streams_reserved"] == 0
    assert report["j2_streams_consumed"] == 0


def test_dead_worker_and_timeout_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyQueue:
        def get(self, timeout: float) -> object:
            raise queue.Empty

    dead_group = object.__new__(pilot.TeacherWorkerGroup)
    dead_group.result_queue = EmptyQueue()
    dead_group.processes = [
        SimpleNamespace(pid=123, is_alive=lambda: False)
    ]
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="exited before result",
    ):
        dead_group._receive_many(1, expected_kind="result")

    live_group = object.__new__(pilot.TeacherWorkerGroup)
    live_group.result_queue = EmptyQueue()
    live_group.processes = [
        SimpleNamespace(pid=456, is_alive=lambda: True)
    ]
    monkeypatch.setattr(pilot, "QUERY_TIMEOUT_SECONDS", 0.001)
    with pytest.raises(
        pilot.PilotOperationalHold,
        match="timed out",
    ):
        live_group._receive_many(1, expected_kind="result")


def test_shuffled_arrival_passes_but_missing_state_and_intra_shard_shuffle_fail(
) -> None:
    indices = list(range(16))
    records = _worker_records(expected_indices=indices)
    report = pilot.validate_worker_records(
        list(reversed(records)),
        expected_indices=indices,
        round_id="round",
        wall_seconds=1.0,
        ready_records=_ready_records(),
        startup_and_load_wall_seconds=2.0,
        dispatch_monotonic_ns=10,
        received_monotonic_ns=20,
        lifetime_contemporaneous_peak_rss_bytes=999,
        lifetime_rss_sample_count=3,
        measured_round_contemporaneous_peak_rss_bytes=777,
        measured_round_rss_sample_count=2,
    )
    assert report["record_count"] == 16

    missing = _worker_records(expected_indices=indices)
    missing[0]["indices"] = [0]
    missing[0]["actions"] = [0]
    missing[0]["timings"] = [0.001]
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="membership changed",
    ):
        pilot.validate_worker_records(
            missing,
            expected_indices=indices,
            round_id="round",
            wall_seconds=1.0,
            ready_records=_ready_records(),
            startup_and_load_wall_seconds=2.0,
            dispatch_monotonic_ns=10,
            received_monotonic_ns=20,
            lifetime_contemporaneous_peak_rss_bytes=999,
            lifetime_rss_sample_count=3,
            measured_round_contemporaneous_peak_rss_bytes=777,
            measured_round_rss_sample_count=2,
        )

    shuffled = _worker_records(expected_indices=indices)
    shuffled[0]["indices"] = [8, 0]
    shuffled[0]["actions"] = [0, 0]
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="ownership/order",
    ):
        pilot.validate_worker_records(
            shuffled,
            expected_indices=indices,
            round_id="round",
            wall_seconds=1.0,
            ready_records=_ready_records(),
            startup_and_load_wall_seconds=2.0,
            dispatch_monotonic_ns=10,
            received_monotonic_ns=20,
            lifetime_contemporaneous_peak_rss_bytes=999,
            lifetime_rss_sample_count=3,
            measured_round_contemporaneous_peak_rss_bytes=777,
            measured_round_rss_sample_count=2,
        )


def test_late_and_cross_round_records_are_distinguished() -> None:
    indices = list(range(8))
    late = _worker_records(
        expected_indices=indices,
        round_id="sync-round-00",
    )
    with pytest.raises(pilot.PilotIntegrityError, match="Late worker"):
        pilot.validate_worker_records(
            late,
            expected_indices=indices,
            round_id="sync-round-01",
            wall_seconds=1.0,
            ready_records=_ready_records(),
            startup_and_load_wall_seconds=2.0,
            dispatch_monotonic_ns=10,
            received_monotonic_ns=20,
            lifetime_contemporaneous_peak_rss_bytes=999,
            lifetime_rss_sample_count=3,
            measured_round_contemporaneous_peak_rss_bytes=777,
            measured_round_rss_sample_count=2,
        )
    foreign = _worker_records(
        expected_indices=indices,
        round_id="foreign-round-00",
    )
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="Cross-round worker",
    ):
        pilot.validate_worker_records(
            foreign,
            expected_indices=indices,
            round_id="sync-round-01",
            wall_seconds=1.0,
            ready_records=_ready_records(),
            startup_and_load_wall_seconds=2.0,
            dispatch_monotonic_ns=10,
            received_monotonic_ns=20,
            lifetime_contemporaneous_peak_rss_bytes=999,
            lifetime_rss_sample_count=3,
            measured_round_contemporaneous_peak_rss_bytes=777,
            measured_round_rss_sample_count=2,
        )


def test_lifetime_rss_is_admission_evidence_and_zero_samples_fail() -> None:
    indices = list(range(8))
    report = pilot.validate_worker_records(
        _worker_records(expected_indices=indices),
        expected_indices=indices,
        round_id="round",
        wall_seconds=1.0,
        ready_records=_ready_records(),
        startup_and_load_wall_seconds=2.0,
        dispatch_monotonic_ns=10,
        received_monotonic_ns=20,
        lifetime_contemporaneous_peak_rss_bytes=1_500,
        lifetime_rss_sample_count=9,
        measured_round_contemporaneous_peak_rss_bytes=700,
        measured_round_rss_sample_count=2,
    )
    assert report[
        "max_contemporaneous_parent_children_rss_bytes"
    ] == 1_500
    assert report[
        "measured_round_contemporaneous_parent_children_rss_bytes"
    ] == 700
    assert report["summed_worker_peak_rss_bytes"] != 1_500
    for lifetime_samples, measured_samples in ((0, 2), (9, 0)):
        with pytest.raises(
            pilot.PilotIntegrityError,
            match="missing or started late",
        ):
            pilot.validate_worker_records(
                _worker_records(expected_indices=indices),
                expected_indices=indices,
                round_id="round",
                wall_seconds=1.0,
                ready_records=_ready_records(),
                startup_and_load_wall_seconds=2.0,
                dispatch_monotonic_ns=10,
                received_monotonic_ns=20,
                lifetime_contemporaneous_peak_rss_bytes=1_500,
                lifetime_rss_sample_count=lifetime_samples,
                measured_round_contemporaneous_peak_rss_bytes=700,
                measured_round_rss_sample_count=measured_samples,
            )


def test_projection_gates_contemporaneous_not_independent_peak_sum() -> None:
    central, sensitivity, sync = _phase_projection_inputs()
    central["parallel_eight_process"][
        "summed_worker_peak_rss_bytes"
    ] = 100 * 1024**3
    sensitivity["parallel_eight_process"][
        "summed_worker_peak_rss_bytes"
    ] = 100 * 1024**3
    sync["summed_worker_peak_rss_bytes"] = 100 * 1024**3
    report = pilot._project_phase_costs(
        central_public=central,
        sensitivity_public=sensitivity,
        sync_public=sync,
        preflight_available_memory_bytes=64 * 1024**3,
        output_bytes=0,
        admission_contract=pilot.central_p99_admission_contract(),
    )
    assert report["memory"][
        "summed_independent_worker_peak_rss_bytes"
    ] == 100 * 1024**3
    assert report["checks"][
        "contemporaneous_peak_memory_within_frozen_cap"
    ]
    central["parallel_eight_process"][
        "max_contemporaneous_parent_children_rss_bytes"
    ] = 25 * 1024**3
    report = pilot._project_phase_costs(
        central_public=central,
        sensitivity_public=sensitivity,
        sync_public=sync,
        preflight_available_memory_bytes=64 * 1024**3,
        output_bytes=0,
        admission_contract=pilot.central_p99_admission_contract(),
    )
    assert not report["checks"][
        "contemporaneous_peak_memory_within_frozen_cap"
    ]


def _write_open_fixture(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
) -> dict[str, object]:
    monkeypatch.setattr(pilot, "INVENTORY_COUNT", 3)
    inventory = {
        "version": "fixture",
        "rows": [{"state_index": index} for index in range(3)],
        "passes": True,
    }
    pilot.write_immutable(
        output_dir / pilot.INVENTORY_NAME,
        inventory,
        field="inventory_payload_sha256",
    )
    inventory_identity = pilot.immutable_identity(
        output_dir / pilot.INVENTORY_NAME,
        "inventory_payload_sha256",
    )
    pilot.write_immutable(
        output_dir / pilot.TEST_EVIDENCE_NAME,
        {"version": "fixture"},
        field="test_evidence_payload_sha256",
    )
    sources = {
        "passes": True,
        "local_sources": {"runner": "r"},
        "teacher_binding": {"incumbent_binding_sha256": "t"},
    }
    lock = {
        "decision": pilot.READY_PREFLIGHT,
        "inventory": inventory_identity,
        "source_audit": sources,
        "execution_command": "fixture",
        "central_p99_admission_contract": (
            pilot.central_p99_admission_contract()
        ),
    }
    pilot.write_immutable(
        output_dir / pilot.PREFLIGHT_LOCK_NAME,
        lock,
        field="preflight_lock_payload_sha256",
    )
    result = {
        "decision": pilot.READY_PREFLIGHT,
        "inventory": inventory_identity,
        "execution_authorized": True,
    }
    pilot.write_immutable(
        output_dir / pilot.PREFLIGHT_RESULT_NAME,
        result,
        field="preflight_result_payload_sha256",
    )
    monkeypatch.setattr(pilot, "source_identity_audit", lambda: sources)
    monkeypatch.setattr(
        pilot,
        "stream_authority_audit",
        lambda: {"passes": True},
    )
    monkeypatch.setattr(
        pilot,
        "operational_audit",
        lambda **_kwargs: {"passes": True},
    )
    return sources


def _rewrite_self_hashed_json(
    path: Path,
    *,
    field: str,
    mutate: object,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload.pop(field)
    payload[field] = pilot.canonical_hash(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_inventory_tamper_between_prepare_and_open_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_open_fixture(monkeypatch, tmp_path)
    _rewrite_self_hashed_json(
        tmp_path / pilot.INVENTORY_NAME,
        field="inventory_payload_sha256",
        mutate=lambda payload: payload["rows"][0].update(
            {"state_index": 99}
        ),
    )
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="identity changed",
    ):
        pilot.open_execution(output_dir=tmp_path)


def test_inventory_tamper_between_open_and_execute_fails_before_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_open_fixture(monkeypatch, tmp_path)
    pilot.open_execution(output_dir=tmp_path)
    _rewrite_self_hashed_json(
        tmp_path / pilot.INVENTORY_NAME,
        field="inventory_payload_sha256",
        mutate=lambda payload: payload["rows"][1].update(
            {"state_index": 88}
        ),
    )
    monkeypatch.setattr(
        pilot,
        "run_serial_workload",
        lambda **_kwargs: pytest.fail("teacher query path opened"),
    )
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="inventory identity changed",
    ):
        pilot._execute_one_shot(output_dir=tmp_path)


def test_source_drift_fails_at_open_and_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sources = _write_open_fixture(monkeypatch, tmp_path)
    drifted = {
        **sources,
        "local_sources": {"runner": "changed"},
    }
    monkeypatch.setattr(pilot, "source_identity_audit", lambda: drifted)
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="sources changed",
    ):
        pilot.open_execution(output_dir=tmp_path)

    second = tmp_path / "execute"
    sources = _write_open_fixture(monkeypatch, second)
    pilot.open_execution(output_dir=second)
    monkeypatch.setattr(pilot, "source_identity_audit", lambda: drifted)
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="Teacher/source identity changed",
    ):
        pilot._execute_one_shot(output_dir=second)


def test_aggregate_output_digest_tamper_fails_public_equality() -> None:
    serial = {
        "actions": [0, 1],
        "ordered_output_sha256": "a" * 64,
        "wall_seconds": 2.0,
        "record_count": 2,
        "warmup_calls": 8,
    }
    parallel = {
        "actions": [0, 1],
        "timings": [0.1, 0.1],
        "ordered_output_sha256": "b" * 64,
        "wall_seconds": 1.0,
        "record_count": 2,
        "warmup_calls": 0,
        "warmup_evidence": {
            "total_worker_warmup_calls": 64,
            "warmup_wall_seconds": 5.0,
        },
    }
    public = pilot._public_cost_result(
        workload="central",
        serial=serial,
        parallel=parallel,
        reference_digest=None,
    )
    assert not public["ordered_digest_equality"]
    assert not public["passes"]


@pytest.mark.parametrize(
    ("name", "field"),
    [
        (pilot.TEST_EVIDENCE_NAME, "test_evidence_payload_sha256"),
        (pilot.INVENTORY_NAME, "inventory_payload_sha256"),
        (pilot.PREFLIGHT_LOCK_NAME, "preflight_lock_payload_sha256"),
        (pilot.PREFLIGHT_RESULT_NAME, "preflight_result_payload_sha256"),
        (pilot.MARKER_NAME, "marker_payload_sha256"),
        (pilot.TERMINAL_NAME, "terminal_payload_sha256"),
    ],
)
def test_every_immutable_stage_artifact_is_create_once(
    tmp_path: Path,
    name: str,
    field: str,
) -> None:
    path = tmp_path / name
    pilot.write_immutable(path, {"version": "fixture"}, field=field)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        pilot.write_immutable(path, {"version": "fixture"}, field=field)
    assert path.read_bytes() == original
    with pytest.raises(pilot.j2.J2ReadinessIntegrityError):
        pilot.write_immutable(
            path,
            {"version": "changed"},
            field=field,
        )
    assert path.read_bytes() == original


def test_inventory_regeneration_is_restart_exact_and_tamper_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "state_index": 0,
        "root_id": "root",
        "ancestry_id": "ancestry",
        "worker_id": 0,
        "target_prefix_steps": 16,
        "streams": {},
        "realized_prefix_steps": 16,
        "prefix_clamped": False,
        "feature_family": "low_air",
        "state_sha256": pilot.canonical_hash({"state": 1}),
        "legal_action_count": 2,
    }
    monkeypatch.setattr(
        pilot,
        "_state_from_engineering_row",
        lambda _row: (dict(expected), {"state": 1}),
    )
    inventory = {
        "rows": [expected],
        "ordered_inventory_sha256": pilot.canonical_hash([expected]),
    }
    assert pilot.regenerate_inventory(inventory) == [{"state": 1}]
    assert pilot.regenerate_inventory(inventory) == [{"state": 1}]
    tampered = {**inventory, "rows": [{**expected, "root_id": "other"}]}
    with pytest.raises(pilot.PilotIntegrityError):
        pilot.regenerate_inventory(tampered)


def test_operational_contention_and_storage_shortfall_hold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    monkeypatch.setattr(
        pilot.j2,
        "operational_audit",
        lambda **_kwargs: {
            "checks": {
                "nice_at_least_10": True,
                "one_heavy_job": False,
                "services_healthy": True,
                "dashboard_top_three_exact": True,
                "human_session_content_unread": True,
            }
        },
    )
    monkeypatch.setattr(
        pilot.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=130 * 1024**3),
    )
    monkeypatch.setattr(pilot, "_physical_memory_bytes", lambda: 64)
    monkeypatch.setattr(pilot, "_available_memory_bytes", lambda: 32)
    operational = pilot.operational_audit(
        output_dir=tmp_path,
        include_namespace_absence=False,
    )
    assert not operational["passes"]
    assert not operational["checks"]["one_heavy_job"]

    central, sensitivity, sync = _phase_projection_inputs()
    projection = pilot._project_phase_costs(
        central_public=central,
        sensitivity_public=sensitivity,
        sync_public=sync,
        preflight_available_memory_bytes=64 * 1024**3,
        output_bytes=pilot.PILOT_OUTPUT_CAP_BYTES,
        admission_contract=pilot.central_p99_admission_contract(),
    )
    assert not projection["checks"][
        "projected_final_pilot_output_with_allowance_within_1gib"
    ]
    assert pilot.terminal_decision(
        integrity_pass=True,
        throughput_pass=False,
        synchronous_pass=True,
        power_pass=True,
    ) == pilot.HOLD_TERMINAL


def test_final_retained_output_cap_includes_terminal_and_retention() -> None:
    preterminal = 100
    within = pilot.final_output_cap_audit(
        preterminal_execution_delta_bytes=preterminal,
        final_execution_delta_bytes=(
            preterminal + pilot.FINAL_EVIDENCE_ALLOWANCE_BYTES
        ),
    )
    assert within["passes"]
    over_allowance = pilot.final_output_cap_audit(
        preterminal_execution_delta_bytes=preterminal,
        final_execution_delta_bytes=(
            preterminal + pilot.FINAL_EVIDENCE_ALLOWANCE_BYTES + 1
        ),
    )
    assert not over_allowance["passes"]
    over_cap = pilot.final_output_cap_audit(
        preterminal_execution_delta_bytes=preterminal,
        final_execution_delta_bytes=pilot.PILOT_OUTPUT_CAP_BYTES + 1,
    )
    assert not over_cap["passes"]


def test_terminal_decision_precedence_is_frozen() -> None:
    assert pilot.terminal_decision(
        integrity_pass=False,
        throughput_pass=True,
        synchronous_pass=True,
        power_pass=True,
    ) == pilot.KILL_TERMINAL
    for failed in ("throughput", "synchronous", "power"):
        gates = {
            "throughput": True,
            "synchronous": True,
            "power": True,
        }
        gates[failed] = False
        assert pilot.terminal_decision(
            integrity_pass=True,
            throughput_pass=gates["throughput"],
            synchronous_pass=gates["synchronous"],
            power_pass=gates["power"],
        ) == pilot.HOLD_TERMINAL
    assert pilot.terminal_decision(
        integrity_pass=True,
        throughput_pass=True,
        synchronous_pass=True,
        power_pass=True,
    ) == pilot.READY_TERMINAL
    assert "PREFLIGHT" in pilot.READY_TERMINAL


def test_query_accounting_is_exact_and_all_measured_warmups_are_zero() -> None:
    accounting = pilot.teacher_query_accounting()
    assert accounting == {
        "central_serial_measured": 512,
        "central_parallel_measured": 512,
        "sensitivity_serial_measured": 5_000,
        "sensitivity_parallel_measured": 5_000,
        "synchronous_serial_reference_measured": 4_096,
        "synchronous_parallel_measured": 4_096,
        "serial_warmups": 24,
        "parallel_worker_warmups": 192,
        "total": 19_432,
    }
    assert pilot.EXPECTED_TEACHER_QUERY_CALLS == 19_432


def test_missing_worker_result_fails_closed() -> None:
    indices = list(range(8))
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="result count changed",
    ):
        pilot.validate_worker_records(
            _worker_records(expected_indices=indices)[:-1],
            expected_indices=indices,
            round_id="round",
            wall_seconds=1.0,
            ready_records=_ready_records(),
            startup_and_load_wall_seconds=2.0,
            dispatch_monotonic_ns=10,
            received_monotonic_ns=20,
            lifetime_contemporaneous_peak_rss_bytes=999,
            lifetime_rss_sample_count=3,
            measured_round_contemporaneous_peak_rss_bytes=777,
            measured_round_rss_sample_count=2,
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            pilot.PilotIntegrityError("worker died"),
            pilot.KILL_TERMINAL,
        ),
        (
            pilot.PilotOperationalHold("worker timed out"),
            pilot.HOLD_TERMINAL,
        ),
    ],
)
def test_execute_seals_failure_terminal_and_forbids_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    expected: str,
) -> None:
    calls = 0

    def fail(*, output_dir: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(pilot, "_execute_one_shot", fail)
    terminal = pilot.execute(output_dir=tmp_path)
    assert terminal["decision"] == expected
    assert terminal["retry_authorized"] is False
    assert (tmp_path / pilot.RETENTION_NAME).is_file()
    terminal_bytes = (tmp_path / pilot.TERMINAL_NAME).read_bytes()
    retention_bytes = (tmp_path / pilot.RETENTION_NAME).read_bytes()
    repeated = pilot.execute(output_dir=tmp_path)
    assert repeated == terminal
    assert calls == 1
    assert (tmp_path / pilot.TERMINAL_NAME).read_bytes() == terminal_bytes
    assert (tmp_path / pilot.RETENTION_NAME).read_bytes() == retention_bytes


def test_output_allowance_is_sole_conjunctive_cap_before_terminal() -> None:
    source = inspect.getsource(pilot._execute_one_shot)
    assert "final_output_cap_audit(" not in source
    central, sensitivity, sync = _phase_projection_inputs()
    report = pilot._project_phase_costs(
        central_public=central,
        sensitivity_public=sensitivity,
        sync_public=sync,
        preflight_available_memory_bytes=64 * 1024**3,
        output_bytes=(
            pilot.PILOT_OUTPUT_CAP_BYTES
            - pilot.FINAL_EVIDENCE_ALLOWANCE_BYTES
            + 1
        ),
        admission_contract=pilot.central_p99_admission_contract(),
    )
    assert not report["checks"][
        "projected_final_pilot_output_with_allowance_within_1gib"
    ]
    assert pilot.terminal_decision(
        integrity_pass=True,
        throughput_pass=False,
        synchronous_pass=True,
        power_pass=True,
    ) == pilot.HOLD_TERMINAL


def test_existing_terminal_requires_valid_completed_retention(
    tmp_path: Path,
) -> None:
    pilot.write_immutable(
        tmp_path / pilot.TERMINAL_NAME,
        {
            "version": "fixture",
            "decision": pilot.READY_TERMINAL,
        },
        field="terminal_payload_sha256",
    )
    with pytest.raises(
        pilot.PilotIntegrityError,
        match="incomplete without retention",
    ):
        pilot.execute(output_dir=tmp_path)

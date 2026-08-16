from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from threes_rl import k1_engineering as k1e
from threes_rl.c1_search_optimization import clone_batched
from threes_rl.c2_cost_admission import incumbent_spec
from threes_rl.eval import make_policy
from threes_rl.k1_compiled_kernel import (
    ABI_VERSION,
    COMPILE_FLAGS,
    COMPILER,
    COMPILER_SHA256,
    NativeKernel,
    build_native_kernel,
    clone_k1,
    sha256_path,
)
from threes_rl.record_replay import state_payload
from threes_rl.sim import ThreesSim, score_board, simulate_base_move
from threes_rl.train_td import state_from_replay_payload


ROOT = Path(__file__).resolve().parents[1]
C2_CORPUS = ROOT / (
    "threes_rl/runs/forensics/c2_cost_admission_v1/C2_CORPUS_MANIFEST.json"
)
K1_OUTPUT = ROOT / "threes_rl/runs/forensics/k1_compiled_kernel_v1"


@pytest.fixture(scope="session")
def incumbent():
    return make_policy(incumbent_spec())


@pytest.fixture(scope="session")
def built_libraries(tmp_path_factory):
    folder = tmp_path_factory.mktemp("k1-build")
    target = folder / "libk1.dylib"
    first_copy = folder / "libk1_first_copy.dylib"
    first_manifest = build_native_kernel(target)
    shutil.copy2(target, first_copy)
    second_manifest = build_native_kernel(target)
    return target, first_copy, first_manifest, second_manifest


@pytest.fixture(scope="session")
def kernel(built_libraries, incumbent):
    return NativeKernel(built_libraries[0], incumbent)


def _spent_rows() -> list[dict]:
    payload = json.loads(C2_CORPUS.read_text())
    return [
        row
        for row in payload["states"]
        if row["partition"] in {"cost_fit", "engineering_validation"}
    ]


def _board(row: dict) -> np.ndarray:
    return np.asarray(row["state"]["board"], dtype=np.int32)


def _values_close(
    left: list[tuple[int, float]],
    right: list[tuple[int, float]],
    tolerance: float,
) -> bool:
    if [action for action, _value in left] != [
        action for action, _value in right
    ]:
        return False
    return all(
        abs(float(left_value) - float(right_value)) <= tolerance
        for (_left_action, left_value), (_right_action, right_value)
        in zip(left, right)
    )


def test_design_preflights_are_hashed_and_zero_work() -> None:
    for name in (
        "K1_TOOLCHAIN_DESIGN_PREFLIGHT.json",
        "K1_TOOLCHAIN_DESIGN_PREFLIGHT_A1.json",
        "K1_TOOLCHAIN_DESIGN_PREFLIGHT_A2.json",
        "K1_TOOLCHAIN_DESIGN_PREFLIGHT_A3.json",
        "K1_TOOLCHAIN_DESIGN_PREFLIGHT_A4.json",
        "K1_TOOLCHAIN_DESIGN_PREFLIGHT_A5.json",
    ):
        payload = json.loads(
            (ROOT / "threes_rl/runs/forensics" / name).read_text()
        )
        expected = payload.pop("canonical_payload_sha256")
        actual = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        assert actual == expected
    assert not K1_OUTPUT.exists()


def test_fresh_stream_manifest_is_exact_unique_and_balanced() -> None:
    rows = k1e.requested_stream_manifest()
    assert len(rows) == 108
    assert [row["behavior_family"] for row in rows[:36]] == [
        "k1_corner2"
    ] * 36
    assert [row["behavior_family"] for row in rows[36:72]] == [
        "k1_parent_mc1000"
    ] * 36
    assert [row["behavior_family"] for row in rows[72:]] == [
        "k1_replaycal"
    ] * 36
    values = [int(row[key]) for row in rows for key in k1e.STREAM_BASES]
    assert len(values) == len(set(values)) == 432
    assert rows[0]["logical_seed"] == 73_000_000_000
    assert rows[36]["logical_seed"] == 73_001_000_000
    assert rows[72]["logical_seed"] == 73_002_000_000


def test_internal_collision_allowlist_excludes_only_hash_bound_zero_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_root = tmp_path / "runs"
    scan_root.mkdir()
    internal = scan_root / "current_declaration.json"
    internal.write_text(json.dumps({
        "logical_seed": 73_000_000_000,
        "zero_work": {"games": 0, "streams": 0},
    }))
    external = scan_root / "external.json"
    external.write_text(json.dumps({"deck_stream_id": 74_000_000_000}))
    out_dir = scan_root / "current"
    out_dir.mkdir()
    monkeypatch.setattr(k1e, "COLLISION_SCAN_ROOT", scan_root)
    monkeypatch.setattr(
        k1e,
        "INTERNAL_COLLISION_ALLOWLIST",
        {internal: k1e.file_manifest(internal)["manifest_sha256"]},
    )
    audit = k1e._scan_collision_sources(out_dir=out_dir)
    assert 73_000_000_000 not in audit["prior"].get("logical_seed", set())
    assert 74_000_000_000 in audit["prior"]["deck_stream_id"]
    assert audit["internal_allowlist"]["passes"]


def test_internal_collision_allowlist_fails_on_mutation_and_new_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_root = tmp_path / "runs"
    scan_root.mkdir()
    internal = scan_root / "current_declaration.json"
    internal.write_text(json.dumps({
        "zero_work": {"games": 0},
        "active_base_sequence": [73_000_000_000],
    }))
    expected = k1e.file_manifest(internal)["manifest_sha256"]
    out_dir = scan_root / "current"
    out_dir.mkdir()
    monkeypatch.setattr(k1e, "COLLISION_SCAN_ROOT", scan_root)
    monkeypatch.setattr(
        k1e,
        "INTERNAL_COLLISION_ALLOWLIST",
        {internal: expected},
    )
    assert k1e._scan_collision_sources(out_dir=out_dir)[
        "internal_allowlist"
    ]["passes"]
    internal.write_text(json.dumps({
        "zero_work": {"games": 0},
        "active_base_sequence": [73_000_000_001],
    }))
    with pytest.raises(k1e.EngineeringFault, match="namespace changed"):
        k1e._scan_collision_sources(out_dir=out_dir)
    internal.write_text(json.dumps({
        "zero_work": {"games": 0},
        "active_base_sequence": [73_000_000_000],
    }))
    new_external = scan_root / "new_unclassified.json"
    new_external.write_text(json.dumps({"slot_stream_id": 75_000_000_000}))
    audit = k1e._scan_collision_sources(out_dir=out_dir)
    assert 75_000_000_000 in audit["prior"]["slot_stream_id"]


def test_corpus_plan_and_chunks_are_frozen_before_timing() -> None:
    plan = k1e.corpus_plan()
    assert plan["total_games"] == 108
    assert plan["required_roots_per_family"] == 12
    assert [row["roots_per_family"] for row in plan["partitions"]] == [
        4,
        4,
        4,
    ]
    assert plan["partition_before_timing"] is True
    assert k1e._partition_for_index(0) == "fresh_equivalence"
    assert k1e._partition_for_index(4) == "engineering_validation"
    assert k1e._partition_for_index(8) == "untouched_runtime_gate"
    with pytest.raises(ValueError):
        k1e._partition_for_index(12)
    chunks = k1e._acquisition_chunks(k1e.requested_stream_manifest())
    assert len(chunks) == 18
    assert all(1 <= len(chunk) <= 6 for chunk in chunks)
    assert [
        chunk[0]["behavior_family"] for chunk in chunks[:3]
    ] == [family for family, _spec in k1e.FAMILY_SLATE]


def test_temporal_state_selection_is_deterministic_and_trigger_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        {
            "root_ancestry": "fresh:73000000000:1536",
            "frame_index": index,
            "state": {"index": index},
            "state_sha256": f"{index:064x}",
            "empty_count": 3,
            "built_max": 768,
            "legal_actions": ["up", "left"],
        }
        for index in range(16)
    ]

    def run_once() -> list[dict]:
        by_index = iter(candidates)
        monkeypatch.setattr(
            k1e,
            "_frame_candidate",
            lambda frame, root: next(by_index),
        )
        monkeypatch.setattr(
            k1e,
            "_incumbent_metadata",
            lambda row, incumbent: {
                "incumbent_margin": 0.01,
                "trigger_reasons": {"low_empty": True, "low_margin": True},
                "incumbent_legal_actions": ["up", "left"],
            },
        )
        return k1e.extract_selected_states(
            {"frames": [{"index": index} for index in range(16)]},
            family="k1_corner2",
            stream_row=k1e.requested_stream_manifest()[0],
            incumbent=object(),
        )

    first = run_once()
    second = run_once()
    assert first == second
    assert [row["selection_bucket"] for row in first] == [0, 1, 2, 3]
    assert len({row["frame_index"] for row in first}) == 4
    assert all(any(row["trigger_reasons"].values()) for row in first)


def test_runtime_summary_and_gate_are_exactly_frozen() -> None:
    rows = []
    for index in range(48):
        family = k1e.FAMILY_SLATE[(index // 16) % 3][0]
        rows.append({
            "record_id": f"record-{index}",
            "root_ancestry": f"{family}-root-{index // 4}",
            "behavior_family": family,
            "baseline_median_seconds": 0.1,
            "compiled_median_seconds": 0.25,
            "compiled_over_depth2": 2.5,
            "repeat_exact": True,
            "compiled_activity": True,
        })
    summary = k1e._runtime_summary(rows)
    checks = k1e.runtime_gate_checks(
        summary,
        exactness_checks={"all": True},
    )
    assert all(checks.values())
    harmful = dict(summary)
    harmful["ratio_p99"] = 8.01
    assert not k1e.runtime_gate_checks(
        harmful,
        exactness_checks={"all": True},
    )["ratio_p99_le_8"]
    inactive = dict(summary)
    inactive["activity_fraction"] = 47 / 48
    assert not k1e.runtime_gate_checks(
        inactive,
        exactness_checks={"all": True},
    )["activity_100pct"]


def test_marker_is_zero_work_one_shot_and_command_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "k1"
    out.mkdir()
    lock_path = out / "K1_PREFLIGHT_LOCK.json"
    lock = k1e.payload_with_hash({
        "bound_out_dir": str(out.resolve()),
        "commands": {"execute": "exact execution command"},
        "jobs": 1,
    })
    k1e.atomic_write_json(lock_path, lock)
    monkeypatch.setattr(k1e, "OUTPUT_DIR", out)
    monkeypatch.setattr(k1e, "_revalidate_files", lambda out_dir, payload: None)
    monkeypatch.setattr(
        k1e,
        "_collision_revalidation",
        lambda out_dir, payload: {"passes": True, "checks": {}},
    )
    monkeypatch.setattr(
        k1e,
        "_operational_audit",
        lambda path: {"passes": True, "checks": {}},
    )
    opened = k1e.seal_execution_opened(
        out_dir=out,
        preflight_lock=lock_path,
        jobs=1,
    )
    assert all(value == 0 for value in opened["zero_work"].values())
    assert not (out / "libk1_exact.dylib").exists()
    marker = k1e._load_marker(out, lock)
    assert marker["execute_command"] == "exact execution command"
    with pytest.raises(FileExistsError):
        k1e.seal_execution_opened(
            out_dir=out,
            preflight_lock=lock_path,
            jobs=1,
        )


def test_marker_rejects_missing_and_mismatched_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(k1e, "OUTPUT_DIR", tmp_path)
    lock_path = tmp_path / "K1_PREFLIGHT_LOCK.json"
    lock = k1e.payload_with_hash({
        "bound_out_dir": str(tmp_path.resolve()),
        "commands": {"execute": "bound"},
        "jobs": 1,
    })
    k1e.atomic_write_json(lock_path, lock)
    with pytest.raises(k1e.EngineeringFault, match="missing"):
        k1e._load_marker(tmp_path, lock)
    marker = k1e.payload_with_hash({
        "execution_opened": True,
        "preflight_lock_file_sha256": sha256_path(lock_path),
        "preflight_lock_payload_sha256": lock["canonical_payload_sha256"],
        "execute_command": "wrong",
        "jobs": 1,
    })
    k1e.atomic_write_json(tmp_path / "K1_EXECUTION_OPENED.json", marker)
    with pytest.raises(k1e.EngineeringFault, match="mismatch"):
        k1e._load_marker(tmp_path, lock)


def test_completed_rows_fail_closed_on_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "completed.jsonl"
    row = {"behavior_family": "k1_corner2", "game_index": 0}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(k1e.EngineeringFault, match="Duplicate"):
        k1e._load_completed(path)


def test_output_namespace_is_still_fresh() -> None:
    assert not K1_OUTPUT.exists()


def test_toolchain_and_command_are_exact() -> None:
    assert sha256_path(COMPILER) == COMPILER_SHA256
    assert COMPILE_FLAGS == (
        "-O3",
        "-std=c11",
        "-fPIC",
        "-dynamiclib",
        "-isysroot",
        "/Applications/Xcode.app/Contents/Developer/Platforms/"
        "MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fno-associative-math",
        "-Wall",
        "-Wextra",
        "-Werror",
    )


def test_native_build_is_deterministic_and_loadable(built_libraries) -> None:
    first, second, first_manifest, second_manifest = built_libraries
    assert sha256_path(first) == sha256_path(second)
    assert first_manifest["library_sha256"] == second_manifest["library_sha256"]
    assert first_manifest["flags"] == second_manifest["flags"]


def test_native_abi_and_binding_are_frozen(kernel) -> None:
    assert int(kernel.library.k1_kernel_abi_version()) == ABI_VERSION
    assert kernel.binding_manifest["pattern_count"] == 21
    assert kernel.binding_manifest["table_count"] == 4 * 4 * 21
    assert len(kernel.binding_manifest["binding_sha256"]) == 64


@pytest.mark.parametrize("action", range(4))
def test_native_base_move_matches_python_on_spent_boards(kernel, action) -> None:
    for row in _spent_rows():
        board = _board(row)
        original = board.copy()
        native_board, native_eligible = kernel.base_move(board, action)
        python_board, python_eligible = simulate_base_move(board, action)
        assert np.array_equal(native_board, python_board)
        assert native_eligible == tuple(python_eligible)
        assert np.array_equal(board, original)


def test_native_base_move_matches_crafted_chain_and_terminal_cases(kernel) -> None:
    boards = (
        np.asarray(
            [[1, 2, 3, 3], [3, 3, 3, 0], [6, 6, 6, 6], [0, 0, 0, 0]],
            dtype=np.int32,
        ),
        np.asarray(
            [
                [12288, 12288, 0, 0],
                [6144, 6144, 0, 0],
                [1536, 768, 384, 192],
                [2, 1, 0, 0],
            ],
            dtype=np.int32,
        ),
    )
    for board in boards:
        for action in range(4):
            native = kernel.base_move(board, action)
            python = simulate_base_move(board, action)
            assert np.array_equal(native[0], python[0])
            assert native[1] == tuple(python[1])


def test_native_score_matches_python_and_rejects_invalid(kernel) -> None:
    for row in _spent_rows():
        assert kernel.score_board(_board(row)) == score_board(_board(row))
    with pytest.raises(ValueError):
        kernel.score_board(
            np.asarray(
                [[5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.int32,
            )
        )


def test_native_leaf_is_bit_exact_on_all_spent_c2_states(kernel) -> None:
    boards = [_board(row) for row in _spent_rows()]
    native = kernel.evaluate_many(boards)
    reference = kernel.reference_leaf.evaluate_many(boards)
    assert np.array_equal(native, reference)
    assert float(np.max(np.abs(native - reference), initial=0.0)) <= 1e-9


def test_native_leaf_matches_starter_removal_and_phase_edges(kernel) -> None:
    boards = [
        np.asarray(
            [
                [1536, 192, 96, 48],
                [24, 12, 6, 3],
                [2, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        np.asarray(
            [
                [768, 384, 192, 96],
                [48, 1536, 24, 12],
                [6, 3, 2, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        np.asarray(
            [
                [3072, 1536, 768, 384],
                [192, 96, 48, 24],
                [12, 6, 3, 2],
                [1, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    ]
    assert np.array_equal(
        kernel.evaluate_many(boards),
        kernel.reference_leaf.evaluate_many(boards),
    )


def test_native_calls_do_not_consume_sim_rng(kernel) -> None:
    row = _spent_rows()[0]
    board = _board(row)
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=123,
        slot_stream_id=456,
        starter_tile=1536,
    )
    deck_before = json.dumps(sim.deck_rng.bit_generator.state, sort_keys=True)
    slot_before = json.dumps(sim.slot_rng.bit_generator.state, sort_keys=True)
    kernel.evaluate_many([board])
    kernel.base_move(board, 0)
    kernel.score_board(board)
    assert json.dumps(sim.deck_rng.bit_generator.state, sort_keys=True) == deck_before
    assert json.dumps(sim.slot_rng.bit_generator.state, sort_keys=True) == slot_before


def test_fused_post_spawn_rows_match_python_primitives(kernel) -> None:
    for row in _spent_rows()[:12]:
        board = _board(row)
        before_score, native_rows = kernel.post_spawn_rows(board)
        assert before_score == score_board(board)
        native_by_action = {action: values for action, *values in native_rows}
        for action in range(4):
            shifted, eligible = simulate_base_move(board, action)
            if not eligible:
                assert action not in native_by_action
                continue
            native_shifted, native_eligible, after_score, leaf_value = (
                native_by_action[action]
            )
            assert np.array_equal(native_shifted, shifted)
            assert native_eligible == tuple(eligible)
            assert after_score == score_board(shifted)
            expected_leaf = float(
                kernel.reference_leaf.evaluate_many([shifted])[0]
            )
            assert leaf_value == expected_leaf


def test_compiled_search_matches_c1_on_spent_states(built_libraries, incumbent) -> None:
    reference = clone_batched(incumbent)
    compiled = clone_k1(incumbent, built_libraries[0])
    for row in _spent_rows()[:2]:
        state = state_from_replay_payload(row["state"])
        sim_reference = ThreesSim.from_stream_ids(
            deck_stream_id=1,
            slot_stream_id=2,
            starter_tile=1536,
        )
        sim_compiled = ThreesSim.from_stream_ids(
            deck_stream_id=1,
            slot_stream_id=2,
            starter_tile=1536,
        )
        reference.clear_decision_caches()
        compiled.clear_decision_caches()
        expected = reference.adaptive_values(state, sim_reference)
        actual = compiled.adaptive_values(state, sim_compiled)
        assert _values_close(expected["depth2"], actual["depth2"], 1e-8)
        assert _values_close(expected["depth3"], actual["depth3"], 1e-8)
        assert (
            actual["compiled_calls"]["leaf"]
            + actual["compiled_calls"]["post_spawn"]
        ) > 0
        assert actual["compiled_calls"]["base_move"] > 0
        assert state_payload(state, sim_reference) == state_payload(
            state,
            sim_compiled,
        )


def test_locked_c1_c2_artifacts_remain_unchanged() -> None:
    expected = {
        "threes_rl/c1_search_optimization.py":
            "c12852cc7dcc8211d8ecc47ccf8c5598d6055a5f12a9bcec497dc47715e0e789",
        "threes_rl/r2a_adaptive_expectimax.py":
            "ece2a1fc34ea759168d2722ca3a82a212649de97b47400a27b2d0b2055d6d4f6",
        "threes_rl/runs/forensics/c1_search/C1_RUNTIME_GATE.json":
            "2f76415d097d47da2749be58dbf3a16dd22d30d7b0670e3e874361af744c1a0f",
        "threes_rl/runs/forensics/c2_cost_admission_v1/"
        "C2_TERMINAL_RESULT.json":
            "ac1e3b490a6ab7d498cacfdd1157ce68020ebe8459e7b654ac487fa28eb3cb9f",
    }
    for relative, digest in expected.items():
        assert sha256_path(ROOT / relative) == digest

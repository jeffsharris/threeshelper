from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import threes_rl.g1r_qd_admission_v2 as qd
from threes_rl.ntuple import NtupleValue
from threes_rl.record_replay import state_payload
from threes_rl.sim import Preview, SimState, ThreesSim


def _state_unchecked(
    board: list[list[int]],
    *,
    preview: Preview | None = None,
    large_pending: bool = False,
) -> tuple[SimState, ThreesSim]:
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=101,
        slot_stream_id=103,
        starter_tile=1536,
    )
    state = SimState(
        board=np.asarray(board, dtype=np.int32),
        preview=preview or Preview("blue", 1),
        small_counts={"blue": 4, "red": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=large_pending,
        max_tile=max(max(row) for row in board),
        move_count=10,
        game_over=False,
    )
    return state, sim


def _state(
    board: list[list[int]],
    *,
    preview: Preview | None = None,
    large_pending: bool = False,
) -> tuple[SimState, ThreesSim]:
    state, sim = _state_unchecked(
        board,
        preview=preview,
        large_pending=large_pending,
    )
    assert sim.legal_actions(state)
    return state, sim


def test_starter_removal_and_equal_target_ties_are_row_major() -> None:
    state, sim = _state(
        [
            [1536, 768, 768, 0],
            [96, 48, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    descriptor = qd.board_descriptor(state, sim)
    assert descriptor[0] == 2
    assert descriptor[1] == 1
    assert descriptor[2] == 2
    assert descriptor[3] == 1
    assert descriptor[10] == 0

    moved_starter, moved_sim = _state(
        [
            [768, 1536, 768, 0],
            [96, 48, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    moved = qd.board_descriptor(moved_starter, moved_sim)
    assert moved[10] == 1
    assert moved[1:3] == (0, 2)


def test_missing_built_and_second_cells_use_frozen_conventions() -> None:
    only_starter, sim = _state(
        [
            [1536, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    descriptor = qd.board_descriptor(only_starter, sim)
    assert descriptor[0] == 0
    assert descriptor[1] == 16
    assert descriptor[2] == 16
    assert descriptor[3] == 6

    one_built, one_sim = _state(
        [
            [1536, 384, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    descriptor = qd.board_descriptor(one_built, one_sim)
    assert descriptor[1] == 1
    assert descriptor[2] == 16
    assert descriptor[3] == 6


def test_support_components_use_four_neighbors_and_edges_count_once() -> None:
    state, sim = _state(
        [
            [1536, 768, 384, 384],
            [192, 0, 384, 0],
            [0, 0, 96, 0],
            [96, 0, 0, 0],
        ]
    )
    descriptor = qd.board_descriptor(state, sim)
    # 384 is one component, 192 one, and the two diagonal 96s are separate.
    assert descriptor[4] == 4
    # Only target 768 -- 384 at (0,2) is a target/support board edge.
    assert descriptor[5] == 1


def test_monotonicity_and_anchor_integrity_are_exact() -> None:
    good, good_sim = _state(
        [
            [1536, 768, 384, 192],
            [768, 96, 48, 24],
            [384, 48, 12, 6],
            [192, 24, 3, 3],
        ]
    )
    good_descriptor = qd.board_descriptor(good, good_sim)
    assert good_descriptor[8:10] == (0, 0)
    assert good_descriptor[11] == 1

    bad, bad_sim = _state(
        [
            [1536, 3, 6, 12],
            [3, 0, 0, 0],
            [6, 0, 0, 0],
            [12, 0, 0, 0],
        ]
    )
    bad_descriptor = qd.board_descriptor(bad, bad_sim)
    assert bad_descriptor[8:10] == (2, 2)
    assert bad_descriptor[11] == 0


def test_preview_pending_and_invalid_preview_are_fail_closed() -> None:
    state, sim = _state(
        [
            [1536, 768, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        preview=Preview("bonus", None, (6, 12, 24)),
        large_pending=True,
    )
    descriptor = qd.board_descriptor(state, sim)
    assert descriptor[12:] == (3, 1)
    state.preview = Preview("unknown", None)
    with pytest.raises(ValueError, match="preview"):
        qd.board_descriptor(state, sim)


def test_terminal_schema_accepts_exact_legal_counts_zero_through_four() -> None:
    boards = {
        0: [
            [1536, 3, 6, 12],
            [6, 12, 24, 3],
            [12, 24, 3, 6],
            [24, 3, 6, 12],
        ],
        1: [
            [1536, 3, 6, 12],
            [6, 12, 24, 3],
            [12, 24, 3, 6],
            [0, 0, 0, 0],
        ],
        2: [
            [1536, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        3: [
            [1536, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 0, 0, 0],
        ],
        4: [
            [1536, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 3],
        ],
    }
    observed = {}
    for expected, board in boards.items():
        state, sim = _state_unchecked(board)
        assert len(sim.legal_actions(state)) == expected
        descriptor = qd.board_descriptor(state, sim)
        observed[expected] = descriptor[7]
    assert observed == {value: value for value in range(5)}


def test_terminal_schema_range_hash_and_distance_are_frozen() -> None:
    assert qd.DESCRIPTOR_SCHEMA_SHA256 == (
        "a8cd1e154c419d79056fb424e01dbec0627fabd40ffc130b1cd6ffa37d04a2c2"
    )
    assert qd.DESCRIPTOR_SCHEMA["version"] == (
        "g1r_qd_descriptor_v2_terminal_legal0"
    )
    assert qd.DESCRIPTOR_SCHEMA["legal_action_count_domain"] == list(range(5))
    assert qd.ORDINAL_DENOMINATORS[7] == 4.0
    mutations = []
    changed_domain = copy.deepcopy(qd.DESCRIPTOR_SCHEMA)
    changed_domain["legal_action_count_domain"] = [1, 2, 3, 4]
    mutations.append(changed_domain)
    changed_denominator = copy.deepcopy(qd.DESCRIPTOR_SCHEMA)
    changed_denominator["ordinal_denominators"]["7"] = 3.0
    mutations.append(changed_denominator)
    changed_version = copy.deepcopy(qd.DESCRIPTOR_SCHEMA)
    changed_version["version"] = "changed"
    mutations.append(changed_version)
    changed_name = copy.deepcopy(qd.DESCRIPTOR_SCHEMA)
    changed_name["ordered_columns"][7] = "changed_legal_count"
    mutations.append(changed_name)
    changed_order = copy.deepcopy(qd.DESCRIPTOR_SCHEMA)
    changed_order["ordered_columns"][7:9] = reversed(
        changed_order["ordered_columns"][7:9]
    )
    mutations.append(changed_order)
    changed_formula = copy.deepcopy(qd.DESCRIPTOR_SCHEMA)
    changed_formula["distance"] = "changed formula"
    mutations.append(changed_formula)
    changed_categorical_mask = copy.deepcopy(qd.DESCRIPTOR_SCHEMA)
    changed_categorical_mask["categorical_indices"] = [0, 1]
    mutations.append(changed_categorical_mask)
    changed_ordinal_mask = copy.deepcopy(qd.DESCRIPTOR_SCHEMA)
    changed_ordinal_mask["ordinal_denominators"].pop("7")
    mutations.append(changed_ordinal_mask)
    assert all(
        qd.canonical_json_hash(mutation) != qd.DESCRIPTOR_SCHEMA_SHA256
        for mutation in mutations
    )
    base = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    legal_four = list(base)
    legal_four[7] = 4
    assert qd.mixed_descriptor_distance(base, tuple(legal_four)) == 1.0
    with pytest.raises(ValueError, match="legal count outside"):
        qd.StaticArchive({tuple(list(base[:7]) + [-1] + list(base[8:])): 1})
    with pytest.raises(ValueError, match="legal count outside"):
        qd.StaticArchive({tuple(list(base[:7]) + [5] + list(base[8:])): 1})

    state, _sim = _state_unchecked(
        [
            [1536, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    class FiveActionSim:
        @staticmethod
        def legal_actions(_state):
            return [0, 1, 2, 3, 4]

    with pytest.raises(ValueError, match="outside 0..4"):
        qd.board_descriptor(state, FiveActionSim())


def _descriptor_coverage(
    payloads: list[dict],
    *,
    stream_offset: int,
) -> Counter:
    coverage: Counter = Counter()
    for index, payload in enumerate(payloads):
        state = qd.state_from_replay_payload(payload)
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=stream_offset + index * 2,
            slot_stream_id=stream_offset + index * 2 + 1,
            starter_tile=1536,
        )
        before = qd._state_fingerprint(state)
        deck_before = copy.deepcopy(sim.deck_rng.bit_generator.state)
        slot_before = copy.deepcopy(sim.slot_rng.bit_generator.state)
        rows = qd.enumerate_exact_insertion_descriptors(state, sim)
        middle = qd._state_fingerprint(state)
        deck_middle = copy.deepcopy(sim.deck_rng.bit_generator.state)
        slot_middle = copy.deepcopy(sim.slot_rng.bit_generator.state)
        repeated = qd.enumerate_exact_insertion_descriptors(state, sim)
        after = qd._state_fingerprint(state)
        assert rows == repeated
        assert before == middle == after
        assert deck_before == deck_middle == sim.deck_rng.bit_generator.state
        assert slot_before == slot_middle == sim.slot_rng.bit_generator.state
        assert rows
        for row in rows:
            assert row["descriptor"][7] == row["legal_count"]
            assert 0 <= row["legal_count"] <= 4
            coverage[row["legal_count"]] += 1
    return coverage


def test_all_frozen_panel_insertion_descriptors_are_total() -> None:
    assert not qd.OUTPUT_DIR.exists()
    panel = qd._panel_payload()
    assert len(panel["records"]) == 64
    coverage = _descriptor_coverage(
        [record["state"] for record in panel["records"]],
        stream_offset=700_000,
    )
    assert coverage[0] > 0
    assert sum(coverage[value] for value in range(1, 5)) > 0
    reported = {value: coverage[value] for value in range(5)}
    print("QD_V2_PANEL_COVERAGE=" + json.dumps(reported, sort_keys=True))
    assert not qd.OUTPUT_DIR.exists()


def test_all_selected_archive_root_insertion_descriptors_are_total() -> None:
    assert not qd.OUTPUT_DIR.exists()
    selected, scan = qd._selected_archive_records()
    assert scan["selected_roots"] == 489
    assert len(selected) == 489
    assert len({record["root_cluster"] for record in selected}) == 489
    coverage = _descriptor_coverage(
        [record["state"] for record in selected],
        stream_offset=800_000,
    )
    assert coverage[0] > 0
    assert sum(coverage[value] for value in range(1, 5)) > 0
    reported = {value: coverage[value] for value in range(5)}
    print("QD_V2_ARCHIVE_COVERAGE=" + json.dumps(reported, sort_keys=True))
    assert not qd.OUTPUT_DIR.exists()


def test_mixed_metric_uses_hamming_for_categories_and_normalized_ordinals() -> None:
    base = (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    different_cell = list(base)
    different_cell[1] = 15
    assert qd.mixed_descriptor_distance(base, tuple(different_cell)) == 1.0
    different_distance = list(base)
    different_distance[3] = 6
    assert qd.mixed_descriptor_distance(base, tuple(different_distance)) == 1.0

    all_different = list(base)
    for index in qd.CATEGORICAL_INDICES:
        all_different[index] = 1
    for index, denominator in qd.ORDINAL_DENOMINATORS.items():
        all_different[index] = base[index] + int(denominator)
    assert qd.mixed_descriptor_distance(base, tuple(all_different)) == 14.0


def test_archive_counts_and_nearest_tie_are_deterministic() -> None:
    left = (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    right = (0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    query = (0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    archive = qd.StaticArchive({left: 2, right: 3})
    distance, nearest = archive.nearest(query)
    assert distance == 1.0
    assert nearest == left
    assert archive.novelty(query) == pytest.approx(1.0 + 1.0 / 14.0)
    assert archive.novelty(left) == pytest.approx(1.0 / 3.0)
    reloaded = qd.StaticArchive.from_payload(archive.payload())
    assert reloaded.counts == archive.counts


def test_mocked_rank_and_tie_behavior_needs_no_panel_or_parent_value(
    monkeypatch,
) -> None:
    state, sim = _state(
        [
            [1536, 6, 3, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    descriptor = qd.board_descriptor(state, sim)
    policy = qd.StaticArchiveQDPolicy(
        archive=qd.StaticArchive({descriptor: 1}),
        parent_checkpoint=Path("unused"),
        parent_model=object(),
    )
    monkeypatch.setattr(
        policy,
        "action_values",
        lambda _state, _sim: {
            0: {"quality": 10.0, "novelty": 1.0, "spawn_probability": 1.0},
            1: {"quality": 9.0, "novelty": 2.0, "spawn_probability": 1.0},
            2: {"quality": 8.0, "novelty": 3.0, "spawn_probability": 1.0},
        },
    )
    decision = policy.decision(state, sim)
    assert decision["action"] == 0
    assert decision["tie_count_before_action_priority"] == 1

    monkeypatch.setattr(
        policy,
        "action_values",
        lambda _state, _sim: {
            1: {"quality": 10.0, "novelty": 1.0, "spawn_probability": 1.0},
            0: {"quality": 10.0, "novelty": 1.0, "spawn_probability": 1.0},
        },
    )
    decision = policy.decision(state, sim)
    assert decision["action"] == 0
    assert decision["tie_count_before_action_priority"] == 2


def test_policy_bundle_save_reload_is_exact_without_action_evaluation(
    tmp_path,
) -> None:
    state, sim = _state(
        [
            [1536, 6, 3, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    descriptor = qd.board_descriptor(state, sim)
    parent_dir = tmp_path / "parent"
    parent = NtupleValue(patterns=((0,),), init=0.0)
    parent.tables[0][:] = np.arange(len(parent.tables[0]), dtype=np.float32)
    parent.save(parent_dir)
    policy = qd.StaticArchiveQDPolicy(
        archive=qd.StaticArchive({descriptor: 1}),
        parent_checkpoint=parent_dir,
    )
    bundle = tmp_path / "policy"
    policy.save(bundle)
    reloaded = qd.StaticArchiveQDPolicy.load(bundle)
    assert reloaded.name == qd.FAMILY_NAME
    assert reloaded.archive.payload() == policy.archive.payload()
    assert reloaded.parent_checkpoint == parent_dir

    payload = json.loads((bundle / "archive.json").read_text())
    payload["cells"][0]["count"] += 1
    (bundle / "archive.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="archive hash"):
        qd.StaticArchiveQDPolicy.load(bundle)


def test_v1_v2_archive_and_policy_loads_are_cross_rejected(tmp_path) -> None:
    import threes_rl.g1r_qd_admission as v1

    state, sim = _state(
        [
            [1536, 6, 3, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    v1_archive = v1.StaticArchive({v1.board_descriptor(state, sim): 1})
    v2_archive = qd.StaticArchive({qd.board_descriptor(state, sim): 1})
    with pytest.raises(ValueError, match="version mismatch"):
        qd.StaticArchive.from_payload(v1_archive.payload())
    with pytest.raises(ValueError, match="descriptor schema mismatch"):
        v1.StaticArchive.from_payload(v2_archive.payload())

    parent_dir = tmp_path / "parent"
    parent = NtupleValue(patterns=((0,),), init=0.0)
    parent.save(parent_dir)
    v1_bundle = tmp_path / "v1_policy"
    v1.StaticArchiveQDPolicy(
        archive=v1_archive,
        parent_checkpoint=parent_dir,
    ).save(v1_bundle)
    with pytest.raises(ValueError, match="version mismatch"):
        qd.StaticArchiveQDPolicy.load(v1_bundle)

    v2_bundle = tmp_path / "v2_policy"
    qd.StaticArchiveQDPolicy(
        archive=v2_archive,
        parent_checkpoint=parent_dir,
    ).save(v2_bundle)
    with pytest.raises(ValueError, match="version mismatch"):
        v1.StaticArchiveQDPolicy.load(v2_bundle)


def test_v1_execution_lock_is_rejected_before_hash_or_service_work(
    monkeypatch,
    tmp_path,
) -> None:
    payload = {"version": "g1r_qd_admission_v1"}
    payload["lock_payload_sha256"] = qd.canonical_json_hash(payload)
    (tmp_path / "execution_lock.json").write_text(json.dumps(payload))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("identity/service work ran before version rejection")

    monkeypatch.setattr(qd, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(qd, "sha256_path", forbidden)
    monkeypatch.setattr(qd, "service_health", forbidden)
    monkeypatch.setattr(qd, "_heavy_process_audit", forbidden)
    with pytest.raises(ValueError, match="execution lock version mismatch"):
        qd._load_and_validate_lock(tmp_path)
    assert calls == []


def test_v1_sealed_artifact_hashes_remain_exact() -> None:
    expected = {
        "archive.json": (
            "f77777f47713ee6db48fdcffeb8c477532a2c1f935e0b92a322418a6419cce14"
        ),
        "archive_sources.json": (
            "a2211b5603e2a038c9edcaf490d4a5504b3b548e3e847626585c05f1edc1fdba"
        ),
        "execution_lock.json": (
            "ed0a89e77baadfaf685c09960b00701bf61e64f6c434941fa2f26cdeb56eb6e2"
        ),
        "ADMISSION_OPENED.json": (
            "f1faadcf2152b28b0254f36402de4568be4eb056c7dd56c52bbcd51c17d51f6e"
        ),
        "HOLD_QD_ADMISSION_ERROR.json": (
            "205229ce77a34b68ff3fdc31ee0bb83bc917671ae3f9efdb5f5eeb91c0b7068b"
        ),
        "policy/archive.json": (
            "7c6f6d9e38ec6e1652af33c9828e3278f54db3ca30ea3ace04d606cd4151ba1f"
        ),
        "policy/policy.json": (
            "d69fbd3f7ecdcc26097a46e3ac64d004abaf8f64711e629d83578171b6db0e79"
        ),
    }
    v1_dir = Path("threes_rl/runs/forensics/g1r_qd_admission_v1")
    assert {
        relative: qd.sha256_path(v1_dir / relative)
        for relative in expected
    } == expected


def test_root_capped_archive_selection_uses_hash_argmin(monkeypatch, tmp_path) -> None:
    state, _sim = _state(
        [
            [1536, 6, 3, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    payload = {
        "board": state.board.tolist(),
        "preview": {"kind": "blue", "value": 1, "candidates": []},
        "tile_cycle": {
            "small_counts": state.small_counts,
            "small_pos": state.small_pos,
            "small_seen_total": state.small_seen_total,
            "span_small_pos": state.span_small_pos,
            "large_pending": state.large_pending,
            "max_tile": state.max_tile,
        },
        "move_count": state.move_count,
        "game_over": False,
    }
    records = [
        {
            "root_origin": "fresh",
            "starter_tile": 1536,
            "root_cluster": "fresh:7:1536",
            "record_id": record_id,
            "state": payload,
            "source_replay": "unused",
            "source_replay_sha256": "unused",
            "source_frame_index": index,
        }
        for index, record_id in enumerate(("a", "b"))
    ]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"selected_records": records}))
    monkeypatch.setattr(qd, "DIAGNOSTIC_INVENTORY_PATH", inventory)
    selected, scan = qd._selected_archive_records()
    assert scan["selected_roots"] == 1
    assert len(selected) == 1
    expected = min(
        records,
        key=lambda record: qd.canonical_json_hash(
            [
                "G1R-QD-archive-state-v1",
                record["root_cluster"],
                record["record_id"],
                qd.state_signature(record["state"], 1536),
            ]
        ),
    )
    assert selected[0]["record_id"] == expected["record_id"]


def _fresh_replay(seed: int = 7) -> dict:
    reset_state = {
        "board": [
            [1536, 1, 2, 3],
            [1, 2, 3, 1],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        "preview": {"kind": "blue", "value": 1, "candidates": []},
        "tile_cycle": {
            "small_counts": {"blue": 0, "red": 0, "gray": 0},
            "small_pos": 8,
            "small_seen_total": 0,
            "span_small_pos": 0,
            "large_pending": False,
            "max_tile": 1536,
        },
        "move_count": 0,
        "score": 59058,
        "game_over": False,
    }
    return {
        "seed": seed,
        "starter_tile": 1536,
        "policy": "corner2",
        "frames": [{"index": 0, "state": reset_state}],
    }


def test_archive_provenance_rejects_canonical_root_mismatch(tmp_path) -> None:
    replay_path = tmp_path / "replay.json"
    replay = _fresh_replay(seed=7)
    replay_path.write_text(json.dumps(replay))
    valid = {
        "root_cluster": "fresh:7:1536",
        "root_origin": "fresh",
        "root_seed": 7,
        "starter_tile": 1536,
    }
    audit = qd._validate_archive_root_provenance(valid, replay, replay_path)
    assert audit["passes"]
    mismatched = dict(valid, root_cluster="fresh:8:1536", root_seed=8)
    with pytest.raises(ValueError, match="canonical ancestry mismatch"):
        qd._validate_archive_root_provenance(
            mismatched,
            replay,
            replay_path,
        )


def test_reference_signature_locks_match_immutable_preflight() -> None:
    preflight = json.loads(qd.PILOT_V1_PREFLIGHT_PATH.read_text())
    stored = preflight["action_distinctness_audit"][
        "action_signature_sha256"
    ]
    assert {
        family: stored[family]
        for family in qd.REFERENCE_FAMILIES
    } == qd.REFERENCE_ACTION_SIGNATURE_SHA256
    fake = {family: [0] * 64 for family in qd.REFERENCE_FAMILIES}
    with pytest.raises(ValueError, match="reference signatures changed"):
        qd._verify_reference_action_signatures(fake)


def test_admission_rejects_existing_opened_marker(tmp_path) -> None:
    (tmp_path / "ADMISSION_OPENED.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already opened"):
        qd.run_admission(tmp_path)


def test_post_open_failure_is_sealed_and_cannot_rerun(
    monkeypatch,
    tmp_path,
) -> None:
    state, sim = _state(
        [
            [1536, 6, 3, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    panel = {
        "panel_sha256": qd.PANEL_SHA256,
        "records": [
            {
                "stratum": "pre1536",
                "state": state_payload(state, sim),
            }
        ],
    }
    lock = {
        "lock_payload_sha256": "lock",
        "implementation_sha256": "implementation",
        "focused_test_sha256": "test",
        "charter_sha256": "charter",
    }

    class BrokenPolicy:
        def decision(self, _state, _sim):
            raise RuntimeError("sealed post-open failure")

    monkeypatch.setattr(qd, "_load_and_validate_lock", lambda _path: lock)
    monkeypatch.setattr(qd, "_panel_payload", lambda: panel)
    monkeypatch.setattr(
        qd.StaticArchiveQDPolicy,
        "load",
        lambda _path: BrokenPolicy(),
    )
    monkeypatch.setattr(
        qd,
        "policy_slate",
        lambda: [(family, family) for family in qd.REFERENCE_FAMILIES],
    )
    monkeypatch.setattr(qd, "make_policy", lambda _spec: object())
    monkeypatch.setattr(
        qd,
        "_verify_retained_archive_sources",
        lambda _path: {"passes": True},
    )
    incumbent = tmp_path / "incumbent.txt"
    incumbent.write_text("incumbent\n")
    monkeypatch.setattr(qd, "INCUMBENT_PATH", incumbent)

    result = qd.run_admission(tmp_path)
    assert result["decision"] == "HOLD_QD_ADMISSION_ERROR"
    assert result["stage"] == "reference_action_signatures"
    assert (tmp_path / "ADMISSION_OPENED.json").is_file()
    assert (tmp_path / "HOLD_QD_ADMISSION_ERROR.json").is_file()
    assert not (tmp_path / "admission_result.json").exists()
    with pytest.raises(FileExistsError, match="already opened"):
        qd.run_admission(tmp_path)


def test_failed_preparation_never_creates_final_directory(
    monkeypatch,
    tmp_path,
) -> None:
    final_dir = tmp_path / "final"
    proposal = tmp_path / "proposal.md"
    proposal.write_text("proposal")
    monkeypatch.setattr(qd, "OUTPUT_DIR", final_dir)
    monkeypatch.setattr(qd, "PROPOSAL_PATH", proposal)
    monkeypatch.setattr(qd, "PROPOSAL_SHA256", qd.sha256_path(proposal))
    monkeypatch.setattr(qd, "current_nice", lambda: 10)
    monkeypatch.setattr(
        qd,
        "_prepare_execution_lock_in_staging",
        lambda _staging, _final: (_ for _ in ()).throw(
            ValueError("source check failed")
        ),
    )
    with pytest.raises(ValueError, match="source check failed"):
        qd.prepare_execution_lock(final_dir)
    assert not final_dir.exists()
    staging = final_dir.with_name(f"{final_dir.name}.staging.{qd.os.getpid()}")
    failure = json.loads((staging / "PREPARATION_FAILED.json").read_text())
    assert failure["decision"] == "HOLD_QD_PREPARATION_ERROR"


def test_heavy_process_audit_ignores_current_parent_wrapper(
    monkeypatch,
) -> None:
    process_table = """
      1     0 /sbin/launchd
     50     1 zsh -ic no-secrets nice -n 10 .venv/bin/python -m threes_rl.g1r_qd_admission_v2 prepare-lock
    100    50 .venv/bin/python -m threes_rl.g1r_qd_admission_v2 prepare-lock
    200     1 .venv/bin/python -m threes_rl.dashboard
    201     1 .venv/bin/python -m threes_rl.human_play_server
    """
    monkeypatch.setattr(qd.os, "getpid", lambda: 100)
    monkeypatch.setattr(
        qd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=process_table),
    )
    audit = qd._heavy_process_audit()
    assert audit["passes"]
    assert audit["excluded_ancestor_pids"] == [1, 50, 100]
    assert audit["other_heavy_processes"] == []


def test_heavy_process_audit_rejects_nonancestor_sibling(
    monkeypatch,
) -> None:
    process_table = """
      1     0 /sbin/launchd
     50     1 zsh -ic no-secrets nice -n 10 .venv/bin/python -m threes_rl.g1r_qd_admission_v2 prepare-lock
    100    50 .venv/bin/python -m threes_rl.g1r_qd_admission_v2 prepare-lock
    300    50 .venv/bin/python -m threes_rl.train_td --episodes 5000
    """
    monkeypatch.setattr(qd.os, "getpid", lambda: 100)
    monkeypatch.setattr(
        qd.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=process_table),
    )
    audit = qd._heavy_process_audit()
    assert not audit["passes"]
    assert audit["other_heavy_processes"] == [
        {
            "pid": 300,
            "ppid": 50,
            "command": (
                ".venv/bin/python -m threes_rl.train_td --episodes 5000"
            ),
        }
    ]


def test_latency_report_uses_frozen_absolute_and_relative_gates() -> None:
    passing = qd.latency_gate(
        [50_000_000] * 10,
        [60_000_000] * 10,
    )
    assert passing["passes"]
    assert passing["candidate_ns"]["p99"] == 50_000_000

    absolute_failure = qd.latency_gate(
        [300_000_000] * 10,
        [400_000_000] * 10,
    )
    assert not absolute_failure["passes"]
    assert not absolute_failure["checks"]["absolute_median"]

    ratio_failure = qd.latency_gate(
        [50_000_000] * 10,
        [10_000_000] * 10,
    )
    assert not ratio_failure["passes"]
    assert not ratio_failure["checks"]["relative_median"]

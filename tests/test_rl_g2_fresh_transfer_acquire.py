from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from threes_rl import g2_fresh_transfer_acquire as acquire
from threes_rl.record_replay import state_payload
from threes_rl.replay_provenance import ORIGIN_FRESH, direct_root_fields
from threes_rl.sim import ThreesSim


def _fresh_completed_replay(seed: int = 101) -> dict:
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=901,
        slot_stream_id=902,
        starter_tile=acquire.STARTER_TILE,
    )
    initial = sim.reset()
    initial_payload = state_payload(initial, sim)
    board = np.array(
        [
            [1536, 1536, 3, 0],
            [768, 384, 192, 96],
            [48, 24, 12, 6],
            [3, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    transfer = replace(
        initial,
        board=board,
        max_tile=1536,
        move_count=12,
        game_over=False,
    )
    transfer_payload = state_payload(transfer, sim)
    terminal = replace(transfer, move_count=13, game_over=True)
    terminal_payload = state_payload(terminal, sim)
    replay = {
        "policy": "fixture",
        "seed": seed,
        "starter_tile": 1536,
        "game_over": True,
        "frames": [
            {"index": 0, "state": initial_payload, "move": None},
            {"index": 12, "state": transfer_payload, "move": {}},
            {"index": 13, "state": terminal_payload, "move": {}},
        ],
        **direct_root_fields(
            origin=ORIGIN_FRESH,
            seed=seed,
            policy="fixture",
            first_score=None,
        ),
    }
    return replay


def test_charter_and_three_family_contract() -> None:
    assert acquire.CHARTER_PATH.is_file()
    assert acquire.policy_slate()[0] == ("g2_transfer_corner2", "corner2")
    assert acquire.policy_slate()[1] == (
        "g2_transfer_expectimax2",
        "expectimax2",
    )
    assert [row[0] for row in acquire.policy_slate()] == list(
        acquire.EXPECTED_SIGNATURES
    )
    assert len(acquire.policy_slate()) == 3


def test_stream_manifest_is_exact_unique_1920_rows() -> None:
    rows = acquire.requested_stream_manifest()
    assert len(rows) == 1920
    assert {row["family"] for row in rows} == {
        family for family, _spec in acquire.policy_slate()
    }
    assert all(
        sum(row["family"] == family for row in rows) == 640
        for family, _spec in acquire.policy_slate()
    )
    values = [
        row[key] for row in rows for key in acquire.STREAM_BASES
    ]
    assert len(values) == len(set(values))


def test_round_robin_chunks_are_ordered_and_bounded() -> None:
    rows = acquire.requested_stream_manifest()
    chunks = acquire.round_robin_rows(rows, set(), {})
    assert all(1 <= len(chunk) <= 6 for chunk in chunks)
    first = [(row["family_index"], row["game_index"]) for row in chunks[0]]
    assert first == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def test_round_robin_does_not_reallocate_full_family_quota() -> None:
    rows = acquire.requested_stream_manifest()
    first_family = acquire.policy_slate()[0][0]
    chunks = acquire.round_robin_rows(
        rows,
        set(),
        {first_family: acquire.QUOTA_PER_FAMILY},
    )
    assert all(row["family"] != first_family for chunk in chunks for row in chunk)
    assert {
        row["family"] for chunk in chunks for row in chunk
    } == {family for family, _spec in acquire.policy_slate()[1:]}


def test_first_transfer_state_is_chronological_and_deterministic() -> None:
    replay = _fresh_completed_replay()
    later = copy.deepcopy(replay["frames"][1])
    later["index"] = 20
    replay["frames"].insert(2, later)
    before = copy.deepcopy(replay)
    left = acquire.extract_first_transfer_state(
        replay, family="fixture", expected_seed=101
    )
    right = acquire.extract_first_transfer_state(
        replay, family="fixture", expected_seed=101
    )
    assert left == right
    assert left is not None
    assert left["source_frame_index"] == 12
    assert left["source_physical_index"] == 1
    assert replay == before


def test_duplicate_frame_index_uses_physical_order() -> None:
    replay = _fresh_completed_replay()
    duplicate = copy.deepcopy(replay["frames"][1])
    duplicate["state"]["move_count"] = 99
    replay["frames"].insert(2, duplicate)
    selected = acquire.extract_first_transfer_state(
        replay, family="fixture", expected_seed=101
    )
    assert selected is not None
    assert selected["source_physical_index"] == 1
    assert selected["state"]["move_count"] == 12


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda replay: replay.update(game_over=False), "completed"),
        (lambda replay: replay.update(replay_origin="continuation"), "fresh"),
        (lambda replay: replay.update(root_seed=999), "fresh"),
        (lambda replay: replay.update(source_replay="parent.json"), "fresh"),
    ],
)
def test_transfer_extraction_rejects_nonfresh_or_incomplete(
    mutation, match: str
) -> None:
    replay = _fresh_completed_replay()
    mutation(replay)
    with pytest.raises(ValueError, match=match):
        acquire.extract_first_transfer_state(
            replay, family="fixture", expected_seed=101
        )


def test_transfer_extraction_returns_none_without_exact_scale() -> None:
    replay = _fresh_completed_replay()
    replay["frames"] = [replay["frames"][0], replay["frames"][-1]]
    assert (
        acquire.extract_first_transfer_state(
            replay, family="fixture", expected_seed=101
        )
        is None
    )


def test_stream_collision_audit_fails_on_prior_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = acquire.requested_stream_manifest()
    monkeypatch.setattr(
        acquire.base,
        "historical_collision_union",
        lambda **_kwargs: (
            {"deck_stream_id": {rows[0]["deck_stream_id"]}},
            [{"path": "fixture"}],
        ),
    )
    audit = acquire.stream_collision_audit(rows, exclude_dir=tmp_path)
    assert not audit["zero_collisions"]
    assert audit["collisions"]["deck_stream_id"] == [
        rows[0]["deck_stream_id"]
    ]


def test_signature_audit_requires_exact_frozen_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {"stratum": "pre1536", "state": {"id": index}}
        for index in range(32)
    ] + [
        {"stratum": "pre3072", "state": {"id": index + 32}}
        for index in range(32)
    ]
    by_family = {
        family: [index % 4 for index in range(64)]
        for family, _spec in acquire.policy_slate()
    }
    iterator = {
        family: iter(actions + actions)
        for family, actions in by_family.items()
    }
    monkeypatch.setattr(
        acquire.base,
        "deterministic_policy_action",
        lambda policy, _state: {
            "action": next(iterator[policy]),
            "exact_tie_count": 1,
        },
    )
    result = acquire.action_signature_audit(
        {family: family for family, _spec in acquire.policy_slate()},
        {"records": records},
    )
    assert not result["passes"]
    assert not result["checks"]["signature_hashes_exact"]


def test_immutable_signature_panel_loads_with_exact_hash() -> None:
    panel, source = acquire._load_signature_panel()
    assert len(panel["records"]) == 64
    assert panel["panel_sha256"] == source["panel_sha256"]
    assert source["file_sha256"] == acquire.PILOT_V1_LOCK_SHA256


def test_split_reset_fixture_is_exact_and_stream_free() -> None:
    fixture = acquire.split_reset_roundtrip_fixture()
    assert fixture["passes"]
    assert fixture["checks"] == {
        "split_reset_exact": True,
        "state_roundtrip_exact": True,
    }


def test_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "immutable.json"
    acquire._write_new_json_atomic(path, {"a": 1})
    with pytest.raises(FileExistsError):
        acquire._write_new_json_atomic(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_preflight_stages_then_promotes_without_game_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final = tmp_path / "final"
    monkeypatch.setattr(acquire, "OUTPUT_DIR", final)
    sentinel = {
        "version": acquire.VERSION,
        "decision": "READY_G2_FRESH_TRANSFER_ACQUISITION",
        "preflight_payload_sha256": "fixture",
    }

    def fake_prepare(staging: Path, final_dir: Path) -> dict:
        assert staging != final_dir
        assert not final_dir.exists()
        acquire._write_new_json_atomic(staging / "preflight_lock.json", sentinel)
        return sentinel

    monkeypatch.setattr(acquire, "_prepare_preflight_in_staging", fake_prepare)
    monkeypatch.setattr(
        acquire,
        "iter_eval_job_outputs",
        lambda **_kwargs: pytest.fail("preflight generated a game"),
    )
    result = acquire.prepare_preflight(final)
    assert result == sentinel
    assert (final / "preflight_lock.json").is_file()


def test_preflight_failure_is_retained_in_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final = tmp_path / "final"
    monkeypatch.setattr(acquire, "OUTPUT_DIR", final)
    monkeypatch.setattr(
        acquire,
        "_prepare_preflight_in_staging",
        lambda *_args: (_ for _ in ()).throw(ValueError("fixture failure")),
    )
    with pytest.raises(ValueError, match="fixture failure"):
        acquire.prepare_preflight(final)
    staging = list(tmp_path.glob("final.staging.*"))
    assert len(staging) == 1
    failure = json.loads((staging[0] / "PREFLIGHT_FAILURE.json").read_text())
    assert failure["decision"] == "KILL_G2_ACQUISITION_PREFLIGHT"
    assert failure["zero_games"] == 0


def test_validate_preflight_rejects_wrong_version(tmp_path: Path) -> None:
    payload = {
        "version": "wrong",
        "decision": "READY_G2_FRESH_TRANSFER_ACQUISITION",
        "bound_out_dir": str(tmp_path.resolve()),
    }
    payload["preflight_payload_sha256"] = acquire.canonical_json_hash(payload)
    path = tmp_path / "preflight_lock.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        acquire._validate_preflight(path)


def test_completion_row_omits_score_and_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    replay = _fresh_completed_replay()
    output = SimpleNamespace(replay=replay)
    row = {
        "family_index": 0,
        "game_index": 0,
        "logical_seed": 101,
        "deck_stream_id": 102,
        "slot_stream_id": 103,
        "policy_stream_id": 104,
    }
    monkeypatch.setattr(
        acquire,
        "extract_first_transfer_state",
        lambda *_args, **_kwargs: {
            "root_cluster": "fixture",
            "state": replay["frames"][1]["state"],
        },
    )
    compact = acquire._completion_row(
        output=output,
        stream_row=row,
        family="fixture",
        replay_dir=tmp_path,
        retained_count=0,
    )
    text = json.dumps(compact)
    assert "score" not in text
    assert "action" not in text
    assert compact["retained"]

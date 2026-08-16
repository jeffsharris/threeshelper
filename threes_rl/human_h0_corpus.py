"""Freeze and exactly validate the human-play corpus for H0 development."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.run_artifacts import write_json
from threes_rl.sim import DIRECTION_NAMES, ThreesSim, score_tile


CALIBRATION_IDS = {
    "human_20260710_202333_36328e22c2d8e08f",
    "human_20260710_203643_74d346ac3ee87152",
}
SUBSTANTIAL_IDS = {
    "human_20260710_215336_7975af939c096671",
    "human_20260710_220247_04855a408cd80b74",
    "human_20260710_220813_718f392f9012542f",
    "human_20260710_221354_5b0d4341fc2560a6",
    "human_20260710_222308_39af520df9f2e5cc",
    "human_20260710_222838_7fc7a34249b1cd70",
}
SUCCESS_ID = "human_20260710_222838_7fc7a34249b1cd70"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparable_state(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "move_count",
            "board",
            "score",
            "max_tile",
            "game_over",
            "preview",
            "legal_actions",
            "legal_mask",
            "tile_cycle",
        )
    }


def validate_replay(replay: dict[str, Any]) -> dict[str, Any]:
    frames = replay["frames"]
    stream = replay["stream_metadata"]
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(stream["deck_stream_id"]),
        slot_stream_id=int(stream["slot_stream_id"]),
        starter_tile=None if replay.get("starter_tile") is None else int(replay["starter_tile"]),
    )
    state = sim.reset()
    initial_actual = comparable_state(state_payload(state, sim))
    initial_expected = comparable_state(frames[0]["state"])
    mismatches: list[dict[str, Any]] = []
    if initial_actual != initial_expected:
        mismatches.append({"frame": 0, "kind": "initial_state"})
    for frame_index, frame in enumerate(frames[1:], start=1):
        move = frame["move"]
        action = int(move.get("action_index", DIRECTION_NAMES.index(str(move["action"]))))
        state, info = sim.step(state, action)
        expected_state = comparable_state(frame["state"])
        actual_state = comparable_state(state_payload(state, sim))
        if actual_state != expected_state:
            mismatches.append({"frame": frame_index, "kind": "state"})
        expected_move = {
            "action": str(move["action"]),
            "inserted_value": move.get("inserted_value"),
            "inserted_pos": move.get("inserted_pos"),
            "eligible_positions": move.get("eligible_positions"),
            "merge_score_delta": int(move["merge_score_delta"]),
            "score_delta": int(move["score_delta"]),
            "terminal_merge": bool(move["terminal_merge"]),
        }
        actual_move = {
            "action": DIRECTION_NAMES[action],
            "inserted_value": info.inserted_value,
            "inserted_pos": list(info.inserted_pos) if info.inserted_pos is not None else None,
            "eligible_positions": [list(pos) for pos in info.eligible_positions],
            "merge_score_delta": int(info.merge_score_delta),
            "score_delta": int(info.score_delta),
            "terminal_merge": bool(info.terminal_merge),
        }
        if actual_move != expected_move:
            mismatches.append({"frame": frame_index, "kind": "move"})
    return {
        "frames": len(frames),
        "moves_replayed": max(0, len(frames) - 1),
        "exact": not mismatches,
        "mismatches": mismatches[:20],
    }


def classification(session_id: str, earned_score: int) -> tuple[str, bool]:
    if session_id in CALIBRATION_IDS:
        return "calibration_discard", False
    if session_id in SUBSTANTIAL_IDS:
        return ("substantial_success" if session_id == SUCCESS_ID else "substantial_failure"), True
    if earned_score < 10_000:
        return "failure_only", False
    raise ValueError(f"Completed game {session_id} is not covered by the frozen review classification")


def freeze(data_root: Path) -> dict[str, Any]:
    games: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    logical_ids: set[int] = set()
    deck_ids: set[int] = set()
    slot_ids: set[int] = set()
    collisions: list[dict[str, Any]] = []
    for replay_path in sorted(data_root.glob("*/replay.json")):
        replay = json.loads(replay_path.read_text())
        session_path = replay_path.with_name("session.json")
        session = json.loads(session_path.read_text())
        session_id = str(replay["session_id"])
        if not bool(replay.get("game_over")) or session.get("status") == "active":
            excluded.append({"session_id": session_id, "reason": "active_or_incomplete", "path": str(replay_path)})
            continue
        logical = int(replay["seed"])
        deck = int(replay["stream_metadata"]["deck_stream_id"])
        slot = int(replay["stream_metadata"]["slot_stream_id"])
        for kind, value, seen in (
            ("logical", logical, logical_ids),
            ("deck", deck, deck_ids),
            ("slot", slot, slot_ids),
        ):
            if value in seen:
                collisions.append({"session_id": session_id, "kind": kind, "id": value})
            seen.add(value)
        starter = replay.get("starter_tile")
        starter_score = 0 if starter is None else score_tile(int(starter))
        earned = int(replay["final_score"]) - starter_score
        tag, h0_primary = classification(session_id, earned)
        validation = validate_replay(replay)
        games.append(
            {
                "session_id": session_id,
                "replay_path": str(replay_path),
                "replay_sha256": sha256_path(replay_path),
                "session_path": str(session_path),
                "session_sha256": sha256_path(session_path),
                "logical_seed": logical,
                "deck_stream_id": deck,
                "slot_stream_id": slot,
                "starter_tile": starter,
                "final_score": int(replay["final_score"]),
                "earned_score": earned,
                "final_moves": int(replay["final_moves"]),
                "final_max_tile": int(replay["final_max_tile"]),
                "review_tag": tag,
                "h0_primary_ancestry": h0_primary,
                "positive_demonstration": session_id == SUCCESS_ID,
                "validation": validation,
            }
        )
    checks = {
        "ten_completed_games": len(games) == 10,
        "one_active_session_excluded": len(excluded) == 1,
        "six_substantial_h0_ancestries": sum(game["h0_primary_ancestry"] for game in games) == 6,
        "one_positive_demonstration": sum(game["positive_demonstration"] for game in games) == 1,
        "all_replays_exact": all(game["validation"]["exact"] for game in games),
        "stream_ids_independent": not collisions,
    }
    return {
        "manifest_version": "human_h0_corpus_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "classification_contract": {
            "source": "HUMAN_PLAY_REVIEW_20260710.md and explicit user steer",
            "quality_inference_from_score": False,
            "calibration_ids": sorted(CALIBRATION_IDS),
            "substantial_ids": sorted(SUBSTANTIAL_IDS),
            "success_id": SUCCESS_ID,
            "other_completed_sub_10k": "failure_only",
        },
        "identifier_counts": {
            "logical": len(logical_ids),
            "deck": len(deck_ids),
            "slot": len(slot_ids),
        },
        "collisions": collisions,
        "games": games,
        "excluded": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/human_play"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args.data_root)
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "checks": payload["checks"],
                "identifier_counts": payload["identifier_counts"],
                "games": [
                    {
                        "session_id": game["session_id"],
                        "earned_score": game["earned_score"],
                        "review_tag": game["review_tag"],
                        "exact": game["validation"]["exact"],
                    }
                    for game in payload["games"]
                ],
                "excluded": payload["excluded"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

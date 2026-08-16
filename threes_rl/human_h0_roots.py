"""Freeze success-window and current-state-matched human failure roots for H0."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.geometry_forensics import board_without_free_starter, geometry_features
from threes_rl.human_h0_corpus import SUCCESS_ID
from threes_rl.run_artifacts import write_json
from threes_rl.sim import ThreesSim, rank_for_value, score_tile
from threes_rl.train_td import state_from_replay_payload


SUCCESS_OFFSETS = (40, 30, 20, 15, 10, 5, 3, 1)
FAILURE_MIN_FRAME_GAP = 3


def plus_probability(state: Any) -> float:
    sim = ThreesSim(np.random.default_rng(0), starter_tile=1536)
    return float(
        sim._large_probability(
            state.small_seen_total,
            state.span_small_pos,
            state.large_pending,
            state.max_tile,
        )
    )


def state_match_features(payload: dict[str, Any], starter_tile: int | None) -> OrderedDict[str, float]:
    state = state_from_replay_payload(payload)
    board = np.asarray(payload["board"], dtype=np.int32)
    masked = board_without_free_starter(board, starter_tile)
    geometry = geometry_features(board, starter_tile)
    ranks = np.vectorize(rank_for_value)(board).astype(np.float64)
    support_values = [int(value) for value in masked.reshape(-1) if 0 < int(value) < 768]
    preview = state.preview
    small_total = max(1, sum(state.small_counts.values()))
    features: OrderedDict[str, float] = OrderedDict()
    features["score_progress"] = float(geometry["score_minus_starter"]) / 50_000.0
    features["move_progress"] = float(state.move_count) / 400.0
    features["support_score_mass"] = sum(score_tile(value) for value in support_values) / 60_000.0
    features["empty_fraction"] = float(geometry["empty_count"]) / 16.0
    features["snake_inversions"] = float(geometry["best_top_left_snake_inversions"]) / 24.0
    features["max_manhattan"] = float(geometry["manhattan_from_top_left"] or 0) / 6.0
    features["count_384"] = float(np.count_nonzero(masked == 384)) / 2.0
    features["count_192"] = float(np.count_nonzero(masked == 192)) / 3.0
    features["count_96"] = float(np.count_nonzero(masked == 96)) / 4.0
    for col in range(4):
        features[f"top_rank_{col}"] = float(ranks[0, col]) / 13.0
    for kind in ("red", "blue", "gray", "bonus"):
        features[f"preview_{kind}"] = float(preview.kind == kind)
    features["preview_rank"] = (
        float(rank_for_value(preview.value)) / 13.0 if preview.value is not None else 0.0
    )
    features["preview_candidate_mean_rank"] = (
        float(np.mean([rank_for_value(value) for value in preview.candidates])) / 13.0
        if preview.candidates
        else 0.0
    )
    features["plus_probability"] = plus_probability(state)
    features["small_pos"] = float(state.small_pos) / 12.0
    features["span_small_pos"] = float(state.span_small_pos) / 21.0
    features["large_pending"] = float(state.large_pending)
    for kind in ("red", "blue", "gray"):
        features[f"bag_{kind}"] = float(state.small_counts.get(kind, 0)) / small_total
    return features


def feature_distance(left: OrderedDict[str, float], right: OrderedDict[str, float]) -> float:
    if tuple(left) != tuple(right):
        raise ValueError("H0 feature definitions do not match")
    return float(np.sqrt(sum((left[key] - right[key]) ** 2 for key in left)))


def replay_frame_record(
    replay_path: Path,
    replay: dict[str, Any],
    frame_index: int,
    *,
    role: str,
    success_offset: int,
    matched_success_id: str | None,
    distance: float | None,
) -> dict[str, Any]:
    frames = replay["frames"]
    frame = frames[frame_index]
    next_move = frames[frame_index + 1]["move"]
    state = frame["state"]
    starter = replay.get("starter_tile", 1536)
    features = state_match_features(state, starter)
    root_id = hashlib.sha1(
        f"{replay['session_id']}:{frame_index}:{role}:{success_offset}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "root_id": f"human_h0_{root_id}",
        "ancestry_cluster": replay["session_id"],
        "role": role,
        "success_offset": int(success_offset),
        "matched_success_root": matched_success_id,
        "match_distance": distance,
        "source_replay": str(replay_path),
        "source_replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        "source_frame_index": frame_index,
        "source_move_count": int(state["move_count"]),
        "recorded_action": str(next_move["action"]),
        "starter_tile": starter,
        "state": state,
        "match_features": dict(features),
    }


def freeze(corpus_manifest_path: Path) -> dict[str, Any]:
    corpus = json.loads(corpus_manifest_path.read_text())
    if corpus.get("decision") != "PASS":
        raise ValueError("Human H0 corpus did not pass")
    game_by_id = {game["session_id"]: game for game in corpus["games"]}
    substantial = [game for game in corpus["games"] if game["h0_primary_ancestry"]]
    success_game = game_by_id[SUCCESS_ID]
    success_path = Path(success_game["replay_path"])
    success_replay = json.loads(success_path.read_text())
    starter = success_replay.get("starter_tile", 1536)
    promotion_frame = next(
        index
        for index, frame in enumerate(success_replay["frames"])
        if max_tile_excluding_initial_starter(np.asarray(frame["state"]["board"]), starter) >= 1536
    )
    first_3072_frame = next(
        index
        for index, frame in enumerate(success_replay["frames"])
        if max_tile_excluding_initial_starter(np.asarray(frame["state"]["board"]), starter) >= 3072
    )
    success_roots = [
        replay_frame_record(
            success_path,
            success_replay,
            promotion_frame - offset,
            role="success_window",
            success_offset=offset,
            matched_success_id=None,
            distance=None,
        )
        for offset in SUCCESS_OFFSETS
    ]

    failure_games = sorted(
        (game for game in substantial if game["review_tag"] == "substantial_failure"),
        key=lambda game: game["session_id"],
    )
    failure_roots: list[dict[str, Any]] = []
    for game in failure_games:
        replay_path = Path(game["replay_path"])
        replay = json.loads(replay_path.read_text())
        candidates = []
        for index, frame in enumerate(replay["frames"][:-1]):
            state = frame["state"]
            if bool(state.get("game_over")):
                continue
            built_max = max_tile_excluding_initial_starter(
                np.asarray(state["board"], dtype=np.int32), replay.get("starter_tile", 1536)
            )
            if built_max != 768:
                continue
            candidates.append((index, state_match_features(state, replay.get("starter_tile", 1536))))
        if not candidates:
            raise ValueError(f"No built-768 matching frames in {game['session_id']}")
        used: list[int] = []
        for success_root in success_roots:
            target = OrderedDict(success_root["match_features"])
            eligible = [
                (index, features)
                for index, features in candidates
                if all(abs(index - prior) >= FAILURE_MIN_FRAME_GAP for prior in used)
            ]
            if not eligible:
                raise ValueError(f"Insufficient decorrelated matching frames in {game['session_id']}")
            index, _features = min(
                eligible,
                key=lambda row: (feature_distance(target, row[1]), row[0]),
            )
            used.append(index)
            failure_roots.append(
                replay_frame_record(
                    replay_path,
                    replay,
                    index,
                    role="failure_control",
                    success_offset=int(success_root["success_offset"]),
                    matched_success_id=str(success_root["root_id"]),
                    distance=feature_distance(target, _features),
                )
            )

    roots = [*success_roots, *failure_roots]
    clusters = sorted({root["ancestry_cluster"] for root in roots})
    checks = {
        "success_offsets_exact": [root["success_offset"] for root in success_roots]
        == list(SUCCESS_OFFSETS),
        "frame_286_included": any(root["source_frame_index"] == 286 for root in success_roots),
        "promotion_frame_289": promotion_frame == 289,
        "first_3072_frame_290": first_3072_frame == 290,
        "six_ancestry_clusters": len(clusters) == 6,
        "eight_success_roots": len(success_roots) == 8,
        "forty_failure_roots": len(failure_roots) == 40,
        "all_roots_have_recorded_legal_action": all(
            root["recorded_action"] in root["state"]["legal_actions"] for root in roots
        ),
        "matching_uses_no_incumbent_values": True,
    }
    return {
        "manifest_version": "human_h0_roots_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "selection_contract": {
            "success_offsets": list(SUCCESS_OFFSETS),
            "success_selection": "outcome-selected offsets before first built 1536",
            "failure_selection": (
                "Per failure ancestry and success offset, greedy nearest current-state feature match "
                "among built-768 frames, minimum three-frame spacing, distance then frame tie-break."
            ),
            "failure_match_inputs": list(success_roots[0]["match_features"]),
            "uses_incumbent_action_values": False,
            "uses_rollout_outcomes": False,
            "ancestry_is_analysis_cluster": True,
        },
        "corpus_manifest": str(corpus_manifest_path),
        "promotion_frame": promotion_frame,
        "first_3072_frame": first_3072_frame,
        "ancestry_clusters": clusters,
        "roots": roots,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args.corpus_manifest)
    write_json(args.out, payload)
    roots = payload["roots"]
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "checks": payload["checks"],
                "roots": len(roots),
                "clusters": payload["ancestry_clusters"],
                "success_roots": [
                    {
                        "frame": root["source_frame_index"],
                        "offset": root["success_offset"],
                        "action": root["recorded_action"],
                    }
                    for root in roots
                    if root["role"] == "success_window"
                ],
                "failure_match_distance": {
                    "mean": float(np.mean([root["match_distance"] for root in roots if root["role"] == "failure_control"])),
                    "max": float(np.max([root["match_distance"] for root in roots if root["role"] == "failure_control"])),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Capture one compact incumbent state per game and phase from frozen D0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.eval import EvalJob, EvalStreamIds, iter_eval_job_outputs, make_policy
from threes_rl.ntuple import phase4_index_for_board
from threes_rl.run_artifacts import write_json
from threes_rl.split_eval import load_block


STAGE_NAMES = ("early_lt384", "mid_384_768", "late_1536", "endgame_3072p")


def _read_expected(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="") as handle:
        return {
            (str(row["block"]), int(row["index"])): row
            for row in csv.DictReader(handle)
        }


def capture(
    *,
    policy_spec: str,
    stream_manifest: Path,
    expected_results: Path,
    jobs: int,
) -> dict[str, Any]:
    manifest, stream_rows = load_block(stream_manifest, ["D0"])
    expected = _read_expected(expected_results)
    policy = make_policy(policy_spec)
    eval_jobs = [
        EvalJob(
            index=index,
            seed=int(row["logical_seed"]),
            starter_tile=None if row.get("starter_tile") is None else int(row["starter_tile"]),
            stream_ids=EvalStreamIds(
                deck_stream_id=int(row["deck_stream_id"]),
                slot_stream_id=int(row["slot_stream_id"]),
                policy_stream_id=int(row["policy_stream_id"]),
            ),
        )
        for index, row in enumerate(stream_rows)
    ]
    records: list[dict[str, Any]] = []
    validated_games = 0
    for output in iter_eval_job_outputs(
        policy=policy,
        policy_name=policy_spec,
        eval_jobs=eval_jobs,
        max_moves=5000,
        capture_replay=True,
        jobs=jobs,
    ):
        replay = output.replay
        if replay is None:
            raise RuntimeError("D0 capture worker did not return a replay")
        expected_row = expected[("D0", output.index)]
        result = output.result
        for field in ("score", "score_minus_starter", "moves", "max_tile", "max_tile_excl_starter"):
            if int(getattr(result, field)) != int(expected_row[field]):
                raise ValueError(f"D0 replay mismatch for index {output.index}, field {field}")
        validated_games += 1

        by_phase: dict[int, list[tuple[int, dict[str, Any], str | None]]] = {}
        frames = list(replay.get("frames", []))
        for frame_index in range(max(0, len(frames) - 1)):
            frame = frames[frame_index]
            next_frame = frames[frame_index + 1]
            state_payload = frame.get("state") if isinstance(frame, dict) else None
            move_payload = next_frame.get("move") if isinstance(next_frame, dict) else None
            if not isinstance(state_payload, dict) or not isinstance(move_payload, dict):
                continue
            board = np.asarray(state_payload.get("board"), dtype=np.int32)
            if board.shape != (4, 4):
                continue
            stage_idx = phase4_index_for_board(board, starter_tile=result.starter_tile)
            by_phase.setdefault(stage_idx, []).append(
                (frame_index, state_payload, str(move_payload.get("action")) if move_payload.get("action") else None)
            )
        for stage_idx, choices in sorted(by_phase.items()):
            frame_index, state_payload, action = choices[len(choices) // 2]
            records.append(
                {
                    "record_id": f"d0_{output.index:03d}_phase_{stage_idx}_frame_{frame_index}",
                    "ancestry_key": f"d0:{output.index}",
                    "phase4_stage_index": stage_idx,
                    "phase4_stage": STAGE_NAMES[stage_idx],
                    "starter_tile": result.starter_tile,
                    "state": state_payload,
                    "recorded_incumbent_action": action,
                    "block": "D0",
                    "index": output.index,
                    "logical_seed": result.seed,
                    "source_frame_index": frame_index,
                }
            )
    records.sort(key=lambda row: (int(row["phase4_stage_index"]), int(row["index"])))
    return {
        "manifest_version": "r1_d0_compact_phase_states_v1",
        "scope": "one incumbent-trajectory midpoint state per D0 game and reached phase",
        "policy": policy_spec,
        "stream_manifest": str(stream_manifest),
        "stream_manifest_version": manifest.get("manifest_version"),
        "validated_games": validated_games,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--stream-manifest", type=Path, required=True)
    parser.add_argument("--expected-results", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policy_spec = next(
        line.strip()
        for line in args.policy_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    payload = capture(
        policy_spec=policy_spec,
        stream_manifest=args.stream_manifest,
        expected_results=args.expected_results,
        jobs=args.jobs,
    )
    write_json(args.out, payload)
    print(json.dumps({"out": str(args.out), "validated_games": payload["validated_games"], "records": len(payload["records"])}, indent=2))


if __name__ == "__main__":
    main()

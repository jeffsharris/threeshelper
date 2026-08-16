"""Capture and compare fixed worst-case D2 replays for the R1b gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.eval import EvalJob, EvalStreamIds, iter_eval_job_outputs, make_policy
from threes_rl.paired_eval_analysis import bootstrap_ci, read_rows
from threes_rl.run_artifacts import write_json


LARGEST_LOSS_COUNT = 12
LOWER_TAIL_CROSSING_LIMIT = 12
LOWER_TAIL_RATE_MARGIN = 0.02
CORNER_MECHANISM_COUNT = 3


def replay_corner_metrics(replay: dict[str, Any]) -> dict[str, Any]:
    starter = replay.get("starter_tile")
    starter_value = 0 if starter is None else int(starter)
    boards: list[np.ndarray] = []
    moves: list[int] = []
    for frame in replay.get("frames", []):
        state = frame.get("state", {})
        board = np.asarray(state.get("board", []), dtype=np.int64)
        if board.shape != (4, 4):
            continue
        boards.append(board)
        moves.append(int(state.get("move_count", frame.get("index", len(moves)))))
    if not boards:
        raise ValueError("Captured replay has no valid boards")
    final = boards[-1]
    first_top_left_below_starter = next(
        (move for board, move in zip(boards, moves) if starter_value and int(board[0, 0]) < starter_value),
        None,
    )
    final_max = int(np.max(final))
    return {
        "first_top_left_below_starter_move": first_top_left_below_starter,
        "final_top_left": int(final[0, 0]),
        "final_top_left_below_starter": bool(starter_value and int(final[0, 0]) < starter_value),
        "final_max_tile": final_max,
        "final_max_at_top_left": bool(int(final[0, 0]) == final_max),
        "final_empty_count": int(np.count_nonzero(final == 0)),
        "final_board": final.tolist(),
    }


def select_cases(
    baseline_rows: dict[tuple[str, int], dict[str, str]],
    candidate_rows: dict[tuple[str, int], dict[str, str]],
) -> tuple[list[tuple[str, int]], float]:
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Baseline and candidate split-eval rows do not match")
    baseline_scores = np.asarray(
        [float(baseline_rows[key]["score_minus_starter"]) for key in sorted(baseline_rows)],
        dtype=np.float64,
    )
    fixed_p05 = float(np.quantile(baseline_scores, 0.05, method="linear"))
    ordered = sorted(
        baseline_rows,
        key=lambda key: (
            float(candidate_rows[key]["score_minus_starter"])
            - float(baseline_rows[key]["score_minus_starter"])
        ),
    )
    largest_losses = ordered[:LARGEST_LOSS_COUNT]
    crossings = [
        key
        for key in ordered
        if float(candidate_rows[key]["score_minus_starter"]) <= fixed_p05
        and float(baseline_rows[key]["score_minus_starter"]) > fixed_p05
    ][:LOWER_TAIL_CROSSING_LIMIT]
    return list(dict.fromkeys([*largest_losses, *crossings])), fixed_p05


def _tail_rate_difference(
    baseline: list[dict[str, str]],
    candidate: list[dict[str, str]],
    threshold: float,
) -> float:
    baseline_rate = np.mean([float(row["score_minus_starter"]) <= threshold for row in baseline])
    candidate_rate = np.mean([float(row["score_minus_starter"]) <= threshold for row in candidate])
    return float(candidate_rate - baseline_rate)


def _capture_arm(
    *,
    policy_spec: str,
    selected: list[tuple[str, int]],
    rows: dict[tuple[str, int], dict[str, str]],
    out_dir: Path,
    jobs: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    policy = make_policy(policy_spec)
    eval_jobs: list[EvalJob] = []
    for index, key in enumerate(selected):
        row = rows[key]
        eval_jobs.append(
            EvalJob(
                index=index,
                seed=int(row["logical_seed"]),
                starter_tile=None if row.get("starter_tile") in (None, "", "None") else int(row["starter_tile"]),
                stream_ids=EvalStreamIds(
                    deck_stream_id=int(row["deck_stream_id"]),
                    slot_stream_id=int(row["slot_stream_id"]),
                    policy_stream_id=int(row["policy_stream_id"]),
                ),
            )
        )
    captured: dict[tuple[str, int], dict[str, Any]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for output in iter_eval_job_outputs(
        policy=policy,
        policy_name=policy_spec,
        eval_jobs=eval_jobs,
        max_moves=5000,
        capture_replay=True,
        jobs=jobs,
    ):
        key = selected[output.index]
        replay = output.replay
        if replay is None:
            raise RuntimeError("Tail audit replay capture unexpectedly disabled")
        expected = rows[key]
        if (
            output.result.score_minus_starter != int(float(expected["score_minus_starter"]))
            or output.result.moves != int(float(expected["moves"]))
            or output.result.max_tile_excl_starter != int(float(expected["max_tile_excl_starter"]))
        ):
            raise ValueError(f"Captured replay does not reproduce frozen result row {key}")
        replay_path = out_dir / f"{key[0]}_{key[1]:04d}.json"
        write_json(replay_path, replay)
        captured[key] = {
            "replay": str(replay_path),
            "score_minus_starter": output.result.score_minus_starter,
            "moves": output.result.moves,
            "max_tile_excl_starter": output.result.max_tile_excl_starter,
            "corner": replay_corner_metrics(replay),
        }
    return captured


def audit(
    *,
    baseline_results: Path,
    candidate_results: Path,
    baseline_policy: str,
    candidate_policy: str,
    out_dir: Path,
    jobs: int,
) -> dict[str, Any]:
    baseline_map = read_rows(baseline_results)
    candidate_map = read_rows(candidate_results)
    selected, fixed_p05 = select_cases(baseline_map, candidate_map)
    ordered_keys = sorted(baseline_map)
    baseline_rows = [baseline_map[key] for key in ordered_keys]
    candidate_rows = [candidate_map[key] for key in ordered_keys]
    tail_stat = lambda base, cand: _tail_rate_difference(base, cand, fixed_p05)
    tail_rate_difference = tail_stat(baseline_rows, candidate_rows)
    tail_rate_ci = bootstrap_ci(baseline_rows, candidate_rows, tail_stat, seed=20260719)

    baseline_capture = _capture_arm(
        policy_spec=baseline_policy,
        selected=selected,
        rows=baseline_map,
        out_dir=out_dir / "baseline_replays",
        jobs=jobs,
    )
    candidate_capture = _capture_arm(
        policy_spec=candidate_policy,
        selected=selected,
        rows=candidate_map,
        out_dir=out_dir / "candidate_replays",
        jobs=jobs,
    )
    cases: list[dict[str, Any]] = []
    for rank, key in enumerate(selected, start=1):
        baseline = baseline_capture[key]
        candidate = candidate_capture[key]
        baseline_corner = baseline["corner"]
        candidate_corner = candidate["corner"]
        cases.append(
            {
                "rank": rank,
                "block": key[0],
                "index": key[1],
                "logical_seed": int(candidate_map[key]["logical_seed"]),
                "score_difference": int(candidate["score_minus_starter"] - baseline["score_minus_starter"]),
                "baseline": baseline,
                "candidate": candidate,
                "candidate_only_final_anchor_loss": bool(
                    candidate_corner["final_top_left_below_starter"]
                    and not baseline_corner["final_top_left_below_starter"]
                ),
                "candidate_only_terminal_max_displacement": bool(
                    not candidate_corner["final_max_at_top_left"]
                    and baseline_corner["final_max_at_top_left"]
                ),
            }
        )
    new_anchor_losses = sum(bool(case["candidate_only_final_anchor_loss"]) for case in cases)
    new_max_displacements = sum(bool(case["candidate_only_terminal_max_displacement"]) for case in cases)
    catastrophic_tail_rate_regression = bool(
        tail_rate_difference > LOWER_TAIL_RATE_MARGIN and tail_rate_ci[0] > 0.0
    )
    corner_mechanism_flag = bool(
        new_anchor_losses >= CORNER_MECHANISM_COUNT or new_max_displacements >= CORNER_MECHANISM_COUNT
    )
    return {
        "selection": {
            "largest_paired_losses": LARGEST_LOSS_COUNT,
            "additional_new_p05_crossings_limit": LOWER_TAIL_CROSSING_LIMIT,
            "selected_cases": len(selected),
            "fixed_incumbent_p05_threshold": fixed_p05,
        },
        "tail_rate": {
            "difference": tail_rate_difference,
            "ci95": tail_rate_ci,
            "material_margin": LOWER_TAIL_RATE_MARGIN,
            "catastrophic_regression": catastrophic_tail_rate_regression,
        },
        "corner_review": {
            "candidate_only_final_anchor_losses": new_anchor_losses,
            "candidate_only_terminal_max_displacements": new_max_displacements,
            "mechanism_count_threshold": CORNER_MECHANISM_COUNT,
            "corner_mechanism_flag": corner_mechanism_flag,
        },
        "gate_blocked": bool(catastrophic_tail_rate_regression or corner_mechanism_flag),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--baseline-policy", required=True)
    parser.add_argument("--candidate-policy", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    payload = audit(
        baseline_results=args.baseline_results,
        candidate_results=args.candidate_results,
        baseline_policy=args.baseline_policy,
        candidate_policy=args.candidate_policy,
        out_dir=args.out_dir,
        jobs=args.jobs,
    )
    write_json(args.out_dir / "summary.json", payload)
    print(json.dumps({key: payload[key] for key in ("selection", "tail_rate", "corner_review", "gate_blocked")}, indent=2))


if __name__ == "__main__":
    main()

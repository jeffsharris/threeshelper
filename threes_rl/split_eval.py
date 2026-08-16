"""Evaluate one policy on a frozen split-stream normal-start block."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from threes_rl.eval import EvalJob, EvalStreamIds, iter_eval_job_outputs, make_policy, summarize
from threes_rl.run_artifacts import write_json


def load_block(path: Path, block_names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    if manifest.get("evaluator_version") != "split_exogenous_v1":
        raise ValueError("Split evaluation requires evaluator version split_exogenous_v1")
    rows: list[dict[str, Any]] = []
    for block in block_names:
        block_rows = manifest.get("blocks", {}).get(block)
        if not isinstance(block_rows, list):
            raise ValueError(f"Missing stream block {block}")
        for row in block_rows:
            rows.append({**row, "block": block})
    return manifest, rows


def evaluate_split_block(
    *,
    policy_spec: str,
    manifest_path: Path,
    block_names: list[str],
    out_dir: Path,
    max_moves: int = 5000,
    jobs: int = 1,
) -> dict[str, Any]:
    manifest, stream_rows = load_block(manifest_path, block_names)
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
    started = time.perf_counter()
    by_index = {}
    for output in iter_eval_job_outputs(
        policy=policy,
        policy_name=policy_spec,
        eval_jobs=eval_jobs,
        max_moves=max_moves,
        capture_replay=False,
        jobs=jobs,
    ):
        by_index[output.index] = output.result
    elapsed = time.perf_counter() - started
    results = [by_index[index] for index in range(len(eval_jobs))]
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "block",
        "index",
        "logical_seed",
        "starter_tile",
        "deck_stream_id",
        "slot_stream_id",
        "policy_stream_id",
        "score",
        "score_minus_starter",
        "moves",
        "max_tile",
        "max_tile_excl_starter",
        "terminal_tile",
    ]
    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stream_row, result in zip(stream_rows, results):
            writer.writerow(
                {
                    **{field: stream_row.get(field) for field in fieldnames if field in stream_row},
                    "logical_seed": result.seed,
                    "starter_tile": result.starter_tile,
                    "score": result.score,
                    "score_minus_starter": result.score_minus_starter,
                    "moves": result.moves,
                    "max_tile": result.max_tile,
                    "max_tile_excl_starter": result.max_tile_excl_starter,
                    "terminal_tile": result.terminal_tile,
                }
            )
    summary = summarize(results)
    summary.update(
        {
            "policy": policy_spec,
            "blocks": block_names,
            "stream_manifest": str(manifest_path),
            "stream_manifest_version": manifest.get("manifest_version"),
            "evaluator_version": manifest.get("evaluator_version"),
            "elapsed_s": elapsed,
            "games_per_s": len(results) / max(elapsed, 1e-9),
            "results_csv": str(out_dir / "results.csv"),
        }
    )
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--stream-manifest", type=Path, required=True)
    parser.add_argument("--block", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    summary = evaluate_split_block(
        policy_spec=args.policy,
        manifest_path=args.stream_manifest,
        block_names=args.block,
        out_dir=args.out_dir,
        max_moves=args.max_moves,
        jobs=args.jobs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

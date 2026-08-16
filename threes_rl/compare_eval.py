"""Compare two deterministic eval CSVs seed-by-seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def load_rows(paths: list[Path]) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for path in paths:
        with path.open() as fh:
            for row in csv.DictReader(fh):
                seed = int(row["seed"])
                if seed in rows:
                    raise ValueError(f"Duplicate seed {seed} across input CSVs")
                rows[seed] = row
    if not rows:
        raise ValueError("No rows loaded")
    return rows


def metric_value(row: dict[str, str], metric: str) -> float:
    return float(row[metric])


def summarize(candidate_paths: list[Path], baseline_paths: list[Path], metric: str, top_n: int) -> dict[str, object]:
    candidate = load_rows(candidate_paths)
    baseline = load_rows(baseline_paths)
    seeds = sorted(set(candidate) & set(baseline))
    if not seeds:
        raise ValueError("No overlapping seeds")

    candidate_values = [metric_value(candidate[seed], metric) for seed in seeds]
    baseline_values = [metric_value(baseline[seed], metric) for seed in seeds]
    diffs = [candidate_value - baseline_value for candidate_value, baseline_value in zip(candidate_values, baseline_values)]
    ranked = sorted(
        (
            {
                "seed": seed,
                "candidate": candidate_value,
                "baseline": baseline_value,
                "diff": diff,
            }
            for seed, candidate_value, baseline_value, diff in zip(seeds, candidate_values, baseline_values, diffs)
        ),
        key=lambda row: (row["diff"], row["seed"]),
    )
    return {
        "metric": metric,
        "paired_seeds": len(seeds),
        "candidate_mean": mean(candidate_values),
        "baseline_mean": mean(baseline_values),
        "paired_mean_diff": mean(diffs),
        "candidate_median": median(candidate_values),
        "baseline_median": median(baseline_values),
        "paired_median_diff": median(diffs),
        "wins": sum(diff > 0 for diff in diffs),
        "losses": sum(diff < 0 for diff in diffs),
        "ties": sum(diff == 0 for diff in diffs),
        "largest_wins": list(reversed(ranked[-top_n:])),
        "largest_losses": ranked[:top_n],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, action="append")
    parser.add_argument("--baseline", type=Path, required=True, action="append")
    parser.add_argument("--metric", default="score_minus_starter")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    print(json.dumps(summarize(args.candidate, args.baseline, args.metric, args.top_n), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

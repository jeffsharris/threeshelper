"""Paired endpoint and bootstrap analysis for split-stream evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

import numpy as np

from threes_rl.run_artifacts import write_json


def read_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(str(row["block"]), int(row["index"])): row for row in rows}


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def bootstrap_ci(
    baseline: list[dict[str, str]],
    candidate: list[dict[str, str]],
    statistic: Callable[[list[dict[str, str]], list[dict[str, str]]], float],
    *,
    repeats: int = 10_000,
    seed: int = 20260709,
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(baseline)
    samples = np.empty(repeats, dtype=np.float64)
    for idx in range(repeats):
        picked = rng.integers(n, size=n)
        base_sample = [baseline[int(i)] for i in picked]
        candidate_sample = [candidate[int(i)] for i in picked]
        samples[idx] = statistic(base_sample, candidate_sample)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def paired_mean(field: str) -> Callable[[list[dict[str, str]], list[dict[str, str]]], float]:
    return lambda base, cand: mean(float(c[field]) - float(b[field]) for b, c in zip(base, cand))


def arm_stat_difference(field: str, fn: Callable[[list[float]], float]) -> Callable[[list[dict[str, str]], list[dict[str, str]]], float]:
    return lambda base, cand: fn([float(c[field]) for c in cand]) - fn([float(b[field]) for b in base])


def p3072_difference(base: list[dict[str, str]], cand: list[dict[str, str]]) -> float:
    return mean(float(c["max_tile_excl_starter"]) >= 3072 for c in cand) - mean(
        float(b["max_tile_excl_starter"]) >= 3072 for b in base
    )


def analyze(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline_map = read_rows(baseline_path)
    candidate_map = read_rows(candidate_path)
    if set(baseline_map) != set(candidate_map):
        raise ValueError("Baseline and candidate split-eval rows do not match")
    keys = sorted(baseline_map)
    baseline = [baseline_map[key] for key in keys]
    candidate = [candidate_map[key] for key in keys]
    for base, cand in zip(baseline, candidate):
        for stream in ("deck_stream_id", "slot_stream_id", "policy_stream_id"):
            if base[stream] != cand[stream]:
                raise ValueError(f"Mismatched paired stream {stream}")

    metrics = {
        "paired_mean_score_minus_starter": paired_mean("score_minus_starter"),
        "median_score_minus_starter_difference": arm_stat_difference("score_minus_starter", median),
        "lower_decile_score_minus_starter_difference": arm_stat_difference(
            "score_minus_starter", lambda values: percentile(values, 0.10)
        ),
        "paired_mean_moves": paired_mean("moves"),
        "p3072_difference": p3072_difference,
    }
    result_metrics = {}
    for offset, (name, statistic) in enumerate(metrics.items()):
        result_metrics[name] = {
            "estimate": statistic(baseline, candidate),
            "ci95": bootstrap_ci(baseline, candidate, statistic, seed=20260709 + offset),
        }
    baseline_mean = mean(float(row["score_minus_starter"]) for row in baseline)
    score = result_metrics["paired_mean_score_minus_starter"]
    p3072 = result_metrics["p3072_difference"]
    harm = bool(score["ci95"][1] < 0.0 or score["estimate"] <= -0.10 * baseline_mean)
    promotion_candidate = bool(score["ci95"][0] > 0.0 and p3072["estimate"] >= -0.02)
    seed_rows = []
    for key, base, cand in zip(keys, baseline, candidate):
        difference = float(cand["score_minus_starter"]) - float(base["score_minus_starter"])
        seed_rows.append(
            {
                "block": key[0],
                "index": key[1],
                "logical_seed": int(cand["logical_seed"]),
                "baseline_score_minus_starter": int(float(base["score_minus_starter"])),
                "candidate_score_minus_starter": int(float(cand["score_minus_starter"])),
                "difference": difference,
                "baseline_p3072": int(float(base["max_tile_excl_starter"]) >= 3072),
                "candidate_p3072": int(float(cand["max_tile_excl_starter"]) >= 3072),
            }
        )
    changed = [row for row in seed_rows if row["difference"] != 0.0]
    ordered_positive = sorted(seed_rows, key=lambda row: float(row["difference"]), reverse=True)
    ordered_negative = sorted(seed_rows, key=lambda row: float(row["difference"]))
    differences = [float(row["difference"]) for row in seed_rows]
    without_largest_gain = sorted(differences)[:-1] if len(differences) > 1 else differences
    without_largest_loss = sorted(differences)[1:] if len(differences) > 1 else differences
    changed_seed_analysis = {
        "changed_score_games": len(changed),
        "candidate_wins": sum(float(row["difference"]) > 0.0 for row in seed_rows),
        "candidate_losses": sum(float(row["difference"]) < 0.0 for row in seed_rows),
        "ties": sum(float(row["difference"]) == 0.0 for row in seed_rows),
        "p3072_gains": sum(row["baseline_p3072"] == 0 and row["candidate_p3072"] == 1 for row in seed_rows),
        "p3072_losses": sum(row["baseline_p3072"] == 1 and row["candidate_p3072"] == 0 for row in seed_rows),
        "mean_difference_without_largest_gain": mean(without_largest_gain),
        "mean_difference_without_largest_loss": mean(without_largest_loss),
        "largest_gains": ordered_positive[:5],
        "largest_losses": ordered_negative[:5],
    }
    return {
        "paired_games": len(keys),
        "blocks": sorted({key[0] for key in keys}),
        "baseline_mean_score_minus_starter": baseline_mean,
        "metrics": result_metrics,
        "changed_seed_analysis": changed_seed_analysis,
        "frozen_rules": {
            "harm": harm,
            "promotion_candidate": promotion_candidate,
            "p3072_noninferiority_margin": -0.02,
            "score_harm_fraction_of_incumbent_mean": -0.10,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.baseline, args.candidate)
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

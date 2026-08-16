"""Outcome-free O3 mechanism power contract."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


VERSION = "o3_event_power_contract_v1"
TARGETS = (48, 96, 192)
TARGET_PROPORTIONS = (0.50, 0.30, 0.20)
TARGET_FACTORS = (1.20, 1.00, 0.80)
ALIGNMENTS = ("aligned", "unaligned")
ALIGNMENT_FACTORS = (1.15, 0.85)
REPEATS = 8
SIMULATION_DRAWS = 1_024
BOOTSTRAP_REPLICATES = 399
POINT_GATE_ODDS_RATIO = 1.25
POWER_REQUIRED = 0.80
ROOT_CANDIDATES = (192, 264)
ODDS_RATIO_GRID = (1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00)
SEED = 2_026_072_704


def odds_shift(probability: np.ndarray, odds_ratio: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    odds = clipped / (1.0 - clipped)
    shifted = float(odds_ratio) * odds
    return shifted / (1.0 + shifted)


def target_counts(roots: int) -> tuple[int, int, int]:
    total = int(roots)
    if total <= 0:
        raise ValueError("roots must be positive")
    raw = np.asarray(TARGET_PROPORTIONS, dtype=np.float64) * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(np.sum(counts))
    order = sorted(
        range(len(TARGETS)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    result = tuple(int(value) for value in counts)
    if sum(result) != total:
        raise RuntimeError("Target allocation failed to preserve root count")
    return result  # type: ignore[return-value]


def stratum_counts(roots: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for target, count in zip(TARGETS, target_counts(roots), strict=True):
        aligned = (count + 1) // 2
        result[f"T{target}:aligned"] = aligned
        result[f"T{target}:unaligned"] = count - aligned
    if sum(result.values()) != int(roots):
        raise RuntimeError("Stratum allocation failed to preserve root count")
    return result


def mantel_haenszel_odds_ratio(
    cells: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    if not cells:
        raise ValueError("At least one nonempty stratum is required")
    numerator: np.ndarray | None = None
    denominator: np.ndarray | None = None
    for raw_a, raw_b, raw_c, raw_d in cells:
        arrays = [
            np.asarray(value, dtype=np.float64)
            for value in (raw_a, raw_b, raw_c, raw_d)
        ]
        if not all(value.shape == arrays[0].shape for value in arrays):
            raise ValueError("Mantel-Haenszel cells must share one shape")
        correction = np.logical_or.reduce(
            tuple(value == 0.0 for value in arrays)
        ).astype(np.float64) * 0.5
        a, b, c, d = (value + correction for value in arrays)
        total = a + b + c + d
        term_numerator = a * d / total
        term_denominator = b * c / total
        numerator = (
            term_numerator
            if numerator is None
            else numerator + term_numerator
        )
        denominator = (
            term_denominator
            if denominator is None
            else denominator + term_denominator
        )
    assert numerator is not None and denominator is not None
    return numerator / denominator


def _bootstrap_weights(
    rng: np.random.Generator,
    roots: int,
    replicates: int,
) -> np.ndarray:
    samples = rng.integers(0, roots, size=(replicates, roots))
    weights = np.zeros((replicates, roots), dtype=np.int16)
    for index, sample in enumerate(samples):
        weights[index] = np.bincount(sample, minlength=roots)
    return weights


def simulate_mechanism_power(
    roots: int,
    true_odds_ratio: float,
    *,
    draws: int = SIMULATION_DRAWS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    allocations = stratum_counts(roots)
    rng = np.random.default_rng(
        int(seed)
        + int(roots) * 10_000
        + int(round(float(true_odds_ratio) * 100))
    )
    weights = {
        stratum: _bootstrap_weights(rng, count, bootstrap_replicates)
        for stratum, count in allocations.items()
    }
    target_factor = {
        target: factor
        for target, factor in zip(TARGETS, TARGET_FACTORS, strict=True)
    }
    alignment_factor = dict(zip(ALIGNMENTS, ALIGNMENT_FACTORS, strict=True))

    passes = 0
    significant = 0
    point_estimates: list[float] = []
    chunk_size = 32
    for start in range(0, int(draws), chunk_size):
        chunk = min(chunk_size, int(draws) - start)
        point_cells = []
        bootstrap_cells = []
        for stratum, root_count in allocations.items():
            target_text, alignment = stratum.split(":", 1)
            target = int(target_text[1:])
            root_probability = rng.beta(
                1.5,
                28.5,
                size=(chunk, root_count),
            )
            root_probability = np.clip(
                root_probability
                * target_factor[target]
                * alignment_factor[alignment],
                0.005,
                0.50,
            )
            treatment_probability = odds_shift(
                root_probability,
                true_odds_ratio,
            )
            control_uniform = rng.random((chunk, root_count, REPEATS))
            independent_treatment_uniform = rng.random(
                (chunk, root_count, REPEATS)
            )
            coupled = rng.random((chunk, root_count, REPEATS)) < 0.50
            treatment_uniform = np.where(
                coupled,
                control_uniform,
                independent_treatment_uniform,
            )
            control = np.sum(
                control_uniform < root_probability[:, :, None],
                axis=2,
            ).astype(np.float64)
            treatment = np.sum(
                treatment_uniform < treatment_probability[:, :, None],
                axis=2,
            ).astype(np.float64)

            a = np.sum(treatment, axis=1)
            b = root_count * REPEATS - a
            c = np.sum(control, axis=1)
            d = root_count * REPEATS - c
            point_cells.append((a, b, c, d))

            bootstrap_a = treatment @ weights[stratum].T
            bootstrap_c = control @ weights[stratum].T
            trials = root_count * REPEATS
            bootstrap_cells.append(
                (
                    bootstrap_a,
                    trials - bootstrap_a,
                    bootstrap_c,
                    trials - bootstrap_c,
                )
            )

        point = mantel_haenszel_odds_ratio(point_cells)
        bootstrap = mantel_haenszel_odds_ratio(bootstrap_cells)
        lower = np.quantile(bootstrap, 0.025, axis=1)
        lower_pass = lower > 1.0
        significant += int(np.count_nonzero(lower_pass))
        passes += int(
            np.count_nonzero(
                lower_pass & (point >= POINT_GATE_ODDS_RATIO)
            )
        )
        point_estimates.extend(point.tolist())

    power = passes / float(draws)
    lower_power = significant / float(draws)
    return {
        "version": VERSION,
        "roots": int(roots),
        "stratum_counts": allocations,
        "repeats_per_arm_root": REPEATS,
        "true_common_odds_ratio": float(true_odds_ratio),
        "point_gate_odds_ratio": POINT_GATE_ODDS_RATIO,
        "simulation_draws": int(draws),
        "bootstrap_replicates": int(bootstrap_replicates),
        "power_lower_ci_gt_1": lower_power,
        "power_full_gate": power,
        "monte_carlo_standard_error": math.sqrt(
            max(power * (1.0 - power), 0.0) / float(draws)
        ),
        "median_estimated_odds_ratio": float(np.median(point_estimates)),
    }

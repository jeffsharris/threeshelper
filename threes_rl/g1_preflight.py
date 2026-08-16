"""Outcome-free existing-corpus, feature, model, and power preflight for G1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from threes_rl.g1_relational_hazard import (
    FEATURE_WIDTH,
    MODEL_PARAMETER_COUNT,
    combined_schema_manifest,
    family_balanced_action_weights,
    implementation_sha256,
    root_equal_action_weights,
    schema_sha256,
)
from threes_rl.run_artifacts import write_json
from threes_rl.r15a_context_inventory import deterministic_key
from threes_rl.s3_power_preflight import (
    HISTORICAL_LABELS_PATH,
    SOURCE_INVENTORY_PATH,
    STRATA,
    _json,
    historical_control_calibration,
    natural_root_candidates,
    odds_shift,
    sha256_path,
)


VERSION = "g1_existing_corpus_preflight_v1"
CHARTER_PATH = Path("threes_rl/G1_RELATIONAL_HAZARD_EXECUTION_CHARTER.md")
S3_SEAL_PATH = Path(
    "threes_rl/runs/forensics/s3_full_policy/S3_PROVENANCE_SEAL_V2.json"
)
A1_INVENTORY_PATH = Path(
    "threes_rl/runs/forensics/r15a_context_a1/"
    "r15a_natural_state_inventory_a1_20260711.json"
)
TEST_ROOT_DESIGNS = (512, 768, 1024, 1536, 2048, 3072)
REPEATS = 8
ASSUMED_ACTIVITY_FRACTION = 0.30
POWER_DRAWS = 10_000
BOOTSTRAP_SENSITIVITY_DRAWS = 1_000
BOOTSTRAP_REPLICATES = 199
TARGET_ODDS_RATIO = 1.50
POWER_REQUIRED = 0.80
POWER_CALIBRATION_LOG_TOLERANCE = 0.02
FAMILY_CAP = 0.40
FREE_DISK_MIN_BYTES = 100 * 1024**3
FREE_DISK_TARGET_BYTES = 120 * 1024**3
Z_975 = 1.959963984540054


def root_list_sha256(roots: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(roots)).encode()).hexdigest()


def exclusion_union() -> dict[str, Any]:
    s3 = _json(S3_SEAL_PATH)
    roots_by_source = {
        "S3_prior_exclusion_union": set(
            s3["excluded_roots"]["roots"]
        ),
        "S3_surviving_inventory": set(
            s3["surviving_inventory"]["roots"]
        ),
    }
    a1 = _json(A1_INVENTORY_PATH)
    roots_by_source["A2_source_inventory"] = {
        str(record["root_cluster"])
        for record in a1["selected_records"]
    }
    union = set().union(*roots_by_source.values())
    return {
        "roots": union,
        "roots_by_source": roots_by_source,
        "counts": {
            source: len(roots)
            for source, roots in sorted(roots_by_source.items())
        },
        "union_count": len(union),
        "union_sha256": root_list_sha256(union),
    }


def _cluster_influence(
    control: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    roots = control.shape[1]
    control_mean = np.clip(
        np.mean(control, axis=1),
        0.5 / (roots * REPEATS + 1.0),
        1.0 - 0.5 / (roots * REPEATS + 1.0),
    )
    treatment_mean = np.clip(
        np.mean(treatment, axis=1),
        0.5 / (roots * REPEATS + 1.0),
        1.0 - 0.5 / (roots * REPEATS + 1.0),
    )
    effect = np.log(treatment_mean / (1.0 - treatment_mean)) - np.log(
        control_mean / (1.0 - control_mean)
    )
    influence = (
        (treatment - treatment_mean[:, None])
        / (treatment_mean * (1.0 - treatment_mean))[:, None]
        - (control - control_mean[:, None])
        / (control_mean * (1.0 - control_mean))[:, None]
    )
    variance = np.var(influence, axis=1, ddof=1) / roots
    return effect, variance


def active_odds_ratio_for_common(
    calibration: dict[str, Any],
    activity_fraction: float,
    common_odds_ratio: float,
) -> float:
    if activity_fraction <= 0.0:
        return math.inf

    beta_samples = {}
    for index, stratum in enumerate(STRATA):
        details = calibration["strata"][stratum]
        rng = np.random.default_rng(20260725 + index)
        beta_samples[stratum] = rng.beta(
            float(details["beta_alpha"]),
            float(details["beta_beta"]),
            size=200_000,
        )

    def aggregate_log_or(active_odds_ratio: float) -> float:
        effect = 0.0
        for stratum in STRATA:
            base = float(
                calibration["strata"][stratum]["root_equal_base_rate"]
            )
            active = float(
                np.mean(
                    odds_shift(
                        beta_samples[stratum],
                        active_odds_ratio,
                    )
                )
            )
            treatment = (
                (1.0 - activity_fraction) * base
                + activity_fraction * active
            )
            effect += 0.5 * (
                math.log(treatment / (1.0 - treatment))
                - math.log(base / (1.0 - base))
            )
        return effect

    target = math.log(common_odds_ratio)
    low = 1.0
    high = 2.0
    while aggregate_log_or(high) < target and high < 1_000_000.0:
        high *= 2.0
    if aggregate_log_or(high) < target:
        raise ValueError("Cannot realize requested common OR at this activity")
    for _ in range(80):
        midpoint = math.sqrt(low * high)
        if aggregate_log_or(midpoint) >= target:
            high = midpoint
        else:
            low = midpoint
    return high


def simulate_activity_power(
    calibration: dict[str, Any],
    roots: int,
    *,
    activity_fraction: float,
    odds_ratio: float = TARGET_ODDS_RATIO,
    draws: int = POWER_DRAWS,
    seed: int = 20260725,
) -> dict[str, Any]:
    if roots % 2:
        raise ValueError("G1 test roots must split evenly across strata")
    if not 0.0 <= activity_fraction <= 1.0:
        raise ValueError("Activity fraction must be in [0,1]")
    per_stratum = roots // 2
    active = int(math.floor(per_stratum * activity_fraction))
    realized_activity = active / per_stratum
    if active == 0:
        return {
            "roots": roots,
            "roots_per_stratum": per_stratum,
            "repeats_per_action": REPEATS,
            "assumed_activity_fraction": activity_fraction,
            "realized_activity_fraction": 0.0,
            "active_roots_per_stratum": 0,
            "structural_zero_roots_per_stratum": per_stratum,
            "target_policy_common_odds_ratio": odds_ratio,
            "implied_active_root_odds_ratio": None,
            "draws": draws,
            "inference": "whole-root cluster influence normal interval",
            "power_lower_ci_gt_1": 0.0,
            "power_pass_point_and_ci": 0.0,
            "monte_carlo_standard_error": 0.0,
            "median_estimated_common_odds_ratio": 1.0,
        }
    active_odds_ratio = active_odds_ratio_for_common(
        calibration,
        realized_activity,
        odds_ratio,
    )
    rng = np.random.default_rng(seed + roots + int(activity_fraction * 10_000))
    passes = 0
    useful_passes = 0
    estimates: list[float] = []
    chunk_size = 100
    for start in range(0, draws, chunk_size):
        chunk = min(chunk_size, draws - start)
        common_effect = np.zeros(chunk, dtype=np.float64)
        common_variance = np.zeros(chunk, dtype=np.float64)
        for stratum in STRATA:
            details = calibration["strata"][stratum]
            probability = rng.beta(
                float(details["beta_alpha"]),
                float(details["beta_beta"]),
                size=(chunk, per_stratum),
            )
            control = rng.binomial(REPEATS, probability) / REPEATS
            treatment = control.copy()
            if active:
                shifted = odds_shift(
                    probability[:, :active],
                    active_odds_ratio,
                )
                treatment[:, :active] = (
                    rng.binomial(REPEATS, shifted) / REPEATS
                )
            effect, variance = _cluster_influence(control, treatment)
            common_effect += 0.5 * effect
            common_variance += 0.25 * variance
        lower = common_effect - Z_975 * np.sqrt(common_variance)
        significant = lower > 0.0
        passes += int(np.count_nonzero(significant))
        useful_passes += int(
            np.count_nonzero(
                significant & (np.exp(common_effect) >= 1.25)
            )
        )
        estimates.extend(common_effect.tolist())
    power = passes / draws
    median_common_or = float(np.exp(np.median(np.asarray(estimates))))
    calibration_log_error = abs(
        math.log(median_common_or) - math.log(odds_ratio)
    )
    return {
        "roots": roots,
        "roots_per_stratum": per_stratum,
        "repeats_per_action": REPEATS,
        "assumed_activity_fraction": activity_fraction,
        "realized_activity_fraction": realized_activity,
        "active_roots_per_stratum": active,
        "structural_zero_roots_per_stratum": per_stratum - active,
        "target_policy_common_odds_ratio": odds_ratio,
        "implied_active_root_odds_ratio": active_odds_ratio,
        "active_or_solver": (
            "fixed 200000-draw expectation under each frozen beta root "
            "distribution"
        ),
        "draws": draws,
        "inference": "whole-root cluster influence normal interval",
        "power_lower_ci_gt_1": power,
        "power_pass_point_and_ci": useful_passes / draws,
        "monte_carlo_standard_error": math.sqrt(
            max(power * (1.0 - power), 0.0) / draws
        ),
        "median_estimated_common_odds_ratio": median_common_or,
        "target_calibration_log_error": calibration_log_error,
        "target_calibration_tolerance": POWER_CALIBRATION_LOG_TOLERANCE,
        "target_calibration_pass": (
            calibration_log_error <= POWER_CALIBRATION_LOG_TOLERANCE
        ),
    }


def _bootstrap_weights(
    rng: np.random.Generator,
    roots: int,
) -> np.ndarray:
    samples = rng.integers(0, roots, size=(BOOTSTRAP_REPLICATES, roots))
    weights = np.zeros((BOOTSTRAP_REPLICATES, roots), dtype=np.float64)
    for index, row in enumerate(samples):
        weights[index] = np.bincount(row, minlength=roots) / roots
    return weights


def bootstrap_activity_sensitivity(
    calibration: dict[str, Any],
    roots: int,
    *,
    activity_fraction: float,
    odds_ratio: float = TARGET_ODDS_RATIO,
    draws: int = BOOTSTRAP_SENSITIVITY_DRAWS,
    seed: int = 20261725,
) -> dict[str, Any]:
    per_stratum = roots // 2
    active = int(math.floor(per_stratum * activity_fraction))
    realized_activity = active / per_stratum
    active_odds_ratio = active_odds_ratio_for_common(
        calibration,
        realized_activity,
        odds_ratio,
    )
    rng = np.random.default_rng(seed + roots)
    weights = {
        stratum: _bootstrap_weights(rng, per_stratum)
        for stratum in STRATA
    }
    passes = 0
    useful_passes = 0
    chunk_size = 10
    for start in range(0, draws, chunk_size):
        chunk = min(chunk_size, draws - start)
        common_effect = np.zeros(chunk, dtype=np.float64)
        boot_effect = np.zeros(
            (chunk, BOOTSTRAP_REPLICATES),
            dtype=np.float64,
        )
        for stratum in STRATA:
            details = calibration["strata"][stratum]
            probability = rng.beta(
                float(details["beta_alpha"]),
                float(details["beta_beta"]),
                size=(chunk, per_stratum),
            )
            control = rng.binomial(REPEATS, probability) / REPEATS
            treatment = control.copy()
            if active:
                shifted = odds_shift(
                    probability[:, :active],
                    active_odds_ratio,
                )
                treatment[:, :active] = (
                    rng.binomial(REPEATS, shifted) / REPEATS
                )
            control_mean = np.clip(
                np.mean(control, axis=1),
                0.5 / (per_stratum * REPEATS + 1.0),
                1.0 - 0.5 / (per_stratum * REPEATS + 1.0),
            )
            treatment_mean = np.clip(
                np.mean(treatment, axis=1),
                0.5 / (per_stratum * REPEATS + 1.0),
                1.0 - 0.5 / (per_stratum * REPEATS + 1.0),
            )
            common_effect += 0.5 * (
                np.log(treatment_mean / (1.0 - treatment_mean))
                - np.log(control_mean / (1.0 - control_mean))
            )
            control_boot = np.clip(
                control @ weights[stratum].T,
                0.5 / (per_stratum * REPEATS + 1.0),
                1.0 - 0.5 / (per_stratum * REPEATS + 1.0),
            )
            treatment_boot = np.clip(
                treatment @ weights[stratum].T,
                0.5 / (per_stratum * REPEATS + 1.0),
                1.0 - 0.5 / (per_stratum * REPEATS + 1.0),
            )
            boot_effect += 0.5 * (
                np.log(treatment_boot / (1.0 - treatment_boot))
                - np.log(control_boot / (1.0 - control_boot))
            )
        lower = np.quantile(boot_effect, 0.025, axis=1)
        significant = lower > 0.0
        passes += int(np.count_nonzero(significant))
        useful_passes += int(
            np.count_nonzero(
                significant & (np.exp(common_effect) >= 1.25)
            )
        )
    power = passes / draws
    return {
        "roots": roots,
        "draws": draws,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "realized_activity_fraction": realized_activity,
        "target_policy_common_odds_ratio": odds_ratio,
        "implied_active_root_odds_ratio": active_odds_ratio,
        "active_or_solver": (
            "fixed 200000-draw expectation under each frozen beta root "
            "distribution"
        ),
        "power_lower_ci_gt_1": power,
        "power_pass_point_and_ci": useful_passes / draws,
        "monte_carlo_standard_error": math.sqrt(
            max(power * (1.0 - power), 0.0) / draws
        ),
    }


def candidate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    roots = {str(row["root_cluster"]) for row in records}
    return {
        "records": len(records),
        "unique_roots": len(roots),
        "records_by_stratum": dict(
            sorted(Counter(row["stratum"] for row in records).items())
        ),
        "records_by_role": dict(
            sorted(Counter(row["role"] for row in records).items())
        ),
        "roots_by_family": {
            family: len(
                {
                    str(row["root_cluster"])
                    for row in records
                    if row["behavior_family"] == family
                }
            )
            for family in sorted(
                {str(row["behavior_family"]) for row in records}
            )
        },
        "roots_by_family_stratum": {
            stratum: {
                family: len(
                    {
                        str(row["root_cluster"])
                        for row in records
                        if row["stratum"] == stratum
                        and row["behavior_family"] == family
                    }
                )
                for family in sorted(
                    {
                        str(row["behavior_family"])
                        for row in records
                        if row["stratum"] == stratum
                    }
                )
            }
            for stratum in STRATA
        },
    }


def select_one_state_per_root(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_root: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_root.setdefault(str(record["root_cluster"]), []).append(record)
    selected = [
        min(
            rows,
            key=lambda row: deterministic_key(
                "G1-one-state-v1",
                root,
                row["record_id"],
                row["stratum"],
                row["role"],
            ),
        )
        for root, rows in sorted(by_root.items())
    ]
    selection_rows = [
        {
            "root_cluster": str(row["root_cluster"]),
            "record_id": str(row["record_id"]),
            "stratum": str(row["stratum"]),
            "role": str(row["role"]),
            "behavior_family": str(row["behavior_family"]),
            "state_sha1": str(row["state_sha1"]),
        }
        for row in selected
    ]
    selection_hash = hashlib.sha256(
        json.dumps(
            selection_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "records": selected,
        "selection_rows": selection_rows,
        "selection_sha256": selection_hash,
        "raw_record_count": len(records),
        "selected_root_count": len(selected),
        "multiple_record_roots_removed": len(records) - len(selected),
    }


def run_preflight() -> dict[str, Any]:
    source_inventory = _json(SOURCE_INVENTORY_PATH)
    calibration = historical_control_calibration(
        source_inventory,
        HISTORICAL_LABELS_PATH,
    )
    exclusions = exclusion_union()
    candidates = natural_root_candidates(
        source_inventory,
        exclusions["roots"],
    )
    root_selection = select_one_state_per_root(candidates["records"])
    summary = candidate_summary(root_selection["records"])

    power_rows = [
        simulate_activity_power(
            calibration,
            roots,
            activity_fraction=ASSUMED_ACTIVITY_FRACTION,
        )
        for roots in TEST_ROOT_DESIGNS
    ]
    power_design = None
    bootstrap_sensitivities = []
    for row in power_rows:
        if (
            row["power_lower_ci_gt_1"] < POWER_REQUIRED
            or row["power_pass_point_and_ci"] < POWER_REQUIRED
            or not row["target_calibration_pass"]
        ):
            continue
        sensitivity = bootstrap_activity_sensitivity(
            calibration,
            int(row["roots"]),
            activity_fraction=ASSUMED_ACTIVITY_FRACTION,
        )
        bootstrap_sensitivities.append(sensitivity)
        if (
            sensitivity["power_lower_ci_gt_1"] >= POWER_REQUIRED
            and sensitivity["power_pass_point_and_ci"] >= POWER_REQUIRED
        ):
            power_design = row
            break
    bootstrap_sensitivity = (
        bootstrap_sensitivities[-1] if bootstrap_sensitivities else None
    )
    power_ready = power_design is not None

    disk = shutil.disk_usage(Path("threes_rl/runs"))
    train_min = 256
    validation_min = 96
    test_min = int(power_design["roots"]) if power_ready else None
    roots_required = (
        train_min + validation_min + test_min
        if test_min is not None
        else None
    )
    family_counts = summary["roots_by_family"]
    max_family_share = (
        max(family_counts.values()) / summary["unique_roots"]
        if summary["unique_roots"]
        else 1.0
    )
    enough_for_partition_attempt = bool(
        roots_required is not None
        and summary["unique_roots"] >= roots_required
    )
    partition_manifest = None
    partition_status = (
        "not_constructed_insufficient_roots"
        if not enough_for_partition_attempt
        else "not_constructed_fail_closed_requires_balanced_allocator"
    )
    descriptive_pool = {
        "both_strata_present": all(
            summary["records_by_stratum"].get(stratum, 0) > 0
            for stratum in STRATA
        ),
        "both_source_roles_present": all(
            summary["records_by_role"].get(role, 0) > 0
            for role in ("source_success_window", "source_control")
        ),
        "overall_max_family_share": max_family_share,
        "at_least_five_families_overall": len(family_counts) >= 5,
    }
    restore_failure_count = sum(
        int(candidates["counts"].get(key, 0))
        for key in (
            "invalid_replay",
            "invalid_state_restore",
            "missing_frames",
        )
    )
    readiness = {
        "schema_widths_64": FEATURE_WIDTH == 64,
        "parameter_counts_65": MODEL_PARAMETER_COUNT == 65,
        "shared_feature_formulas_identical": combined_schema_manifest()[
            "shared_formulas_identical"
        ],
        "test_power_design_available": power_ready,
        "enough_total_roots_for_partition_attempt": enough_for_partition_attempt,
        "actual_partition_manifest_frozen": partition_manifest is not None,
        "train_validation_test_whole_root_disjoint": False,
        "ten_roots_each_stratum_role_cell_each_partition": False,
        "family_share_at_most_40pct_each_partition": False,
        "at_least_five_families_overall": False,
        "at_least_three_test_families": False,
        "at_least_48_test_roots_each_stratum": False,
        "zero_cross_partition_source_root_collisions": False,
        "zero_stream_collisions": False,
        "exact_restore_and_provenance_audits": restore_failure_count == 0,
        "zero_prior_overlap": all(
            str(row["root_cluster"]) not in exclusions["roots"]
            for row in candidates["records"]
        ),
        "free_disk_above_100gib": disk.free >= FREE_DISK_MIN_BYTES,
    }
    ready = all(readiness.values())
    decision = "READY_G1_PREFLIGHT" if ready else "HOLD_G1_DATA_PREFLIGHT"

    root_weights = root_equal_action_weights([2, 3, 4])
    family_weights = family_balanced_action_weights(
        ["family_a", "family_a", "family_b"],
        [2, 3, 4],
    )
    return {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "labels_generated": False,
        "test_outcomes_generated": False,
        "test_outcomes_inspected": False,
        "models_fit": False,
        "dashboard_eligible": False,
        "locks": {
            "charter": str(CHARTER_PATH),
            "charter_sha256": sha256_path(CHARTER_PATH),
            "feature_implementation_sha256": implementation_sha256(),
            "relational_schema_sha256": schema_sha256("relational"),
            "positional_schema_sha256": schema_sha256("positional"),
            "source_inventory": str(SOURCE_INVENTORY_PATH),
            "source_inventory_sha256": sha256_path(SOURCE_INVENTORY_PATH),
            "historical_control_labels_sha256": sha256_path(
                HISTORICAL_LABELS_PATH
            ),
            "s3_provenance_seal": str(S3_SEAL_PATH),
            "s3_provenance_seal_sha256": sha256_path(S3_SEAL_PATH),
            "a1_inventory_sha256": sha256_path(A1_INVENTORY_PATH),
        },
        "feature_model_contract": combined_schema_manifest(),
        "weighting_audit": {
            "root_equal_total_weights": [
                float(np.sum(row)) for row in root_weights
            ],
            "family_balanced_total_weights": [
                float(np.sum(row)) for row in family_weights
            ],
            "root_equal_sum": float(
                sum(np.sum(row) for row in root_weights)
            ),
            "family_balanced_sum": float(
                sum(np.sum(row) for row in family_weights)
            ),
        },
        "historical_control_calibration": calibration,
        "power_contract": {
            "test_root_designs": list(TEST_ROOT_DESIGNS),
            "repeats_per_legal_action": REPEATS,
            "assumed_minimum_activity_fraction": ASSUMED_ACTIVITY_FRACTION,
            "target_active_root_odds_ratio": TARGET_ODDS_RATIO,
            "required_power": POWER_REQUIRED,
            "target_calibration_log_tolerance": (
                POWER_CALIBRATION_LOG_TOLERANCE
            ),
            "power_rows": power_rows,
            "selected_power_design": power_design,
            "bootstrap_sensitivities_attempted": bootstrap_sensitivities,
            "selected_design_bootstrap_sensitivity": bootstrap_sensitivity,
            "power_ready": power_ready,
        },
        "exclusions": {
            "counts": exclusions["counts"],
            "union_count": exclusions["union_count"],
            "union_sha256": exclusions["union_sha256"],
        },
        "candidate_scan_counts": candidates["counts"],
        "one_state_per_ancestry_selection": {
            key: value
            for key, value in root_selection.items()
            if key != "records"
        },
        "candidate_summary": summary,
        "descriptive_pool_checks": descriptive_pool,
        "partition_manifest": partition_manifest,
        "partition_status": partition_status,
        "minimum_partition_roots": {
            "train": train_min,
            "validation": validation_min,
            "untouched_test": test_min,
            "total": roots_required,
        },
        "readiness_checks": readiness,
        "storage": {
            "free_bytes": disk.free,
            "free_gib": disk.free / (1024**3),
            "minimum_bytes": FREE_DISK_MIN_BYTES,
            "target_bytes": FREE_DISK_TARGET_BYTES,
            "minimum_pass": disk.free >= FREE_DISK_MIN_BYTES,
            "target_pass": disk.free >= FREE_DISK_TARGET_BYTES,
        },
        "outcome_free_next_step": (
            None
            if ready
            else {
                "branch": "G1-R",
                "action": (
                    "Freeze a separately hashed natural normal-start root "
                    "acquisition charter using the selected activity-aware "
                    "power design before generating replays."
                ),
                "prohibitions": (
                    "No G1 labels, test outcomes, model fit, relaxed exclusion, "
                    "or recycled prior root."
                ),
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_preflight()
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "candidate_summary": payload["candidate_summary"],
                "minimum_partition_roots": payload[
                    "minimum_partition_roots"
                ],
                "power_contract": {
                    "selected_power_design": payload["power_contract"][
                        "selected_power_design"
                    ],
                    "bootstrap_sensitivity": payload["power_contract"][
                        "selected_design_bootstrap_sensitivity"
                    ],
                    "power_ready": payload["power_contract"]["power_ready"],
                },
                "readiness_checks": payload["readiness_checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

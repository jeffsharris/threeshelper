from __future__ import annotations

from threes_rl.g1_preflight import (
    active_odds_ratio_for_common,
    exclusion_union,
    select_one_state_per_root,
    simulate_activity_power,
)


def _calibration() -> dict:
    return {
        "strata": {
            "pre1536": {
                "beta_alpha": 1.0,
                "beta_beta": 19.0,
                "root_equal_base_rate": 0.05,
            },
            "pre3072": {
                "beta_alpha": 9.0,
                "beta_beta": 1.0,
                "root_equal_base_rate": 0.90,
            },
        }
    }


def test_structural_zero_activity_has_zero_power() -> None:
    result = simulate_activity_power(
        _calibration(),
        64,
        activity_fraction=0.0,
        odds_ratio=2.0,
        draws=200,
    )
    assert result["active_roots_per_stratum"] == 0
    assert result["power_lower_ci_gt_1"] == 0.0


def test_common_or_solver_increases_active_effect_when_activity_is_sparse() -> None:
    low = simulate_activity_power(
        _calibration(),
        128,
        activity_fraction=0.25,
        odds_ratio=2.0,
        draws=500,
        seed=1,
    )
    high = simulate_activity_power(
        _calibration(),
        128,
        activity_fraction=1.0,
        odds_ratio=2.0,
        draws=500,
        seed=1,
    )
    assert low["implied_active_root_odds_ratio"] > high[
        "implied_active_root_odds_ratio"
    ]
    assert low["median_estimated_common_odds_ratio"] > 1.0
    assert high["median_estimated_common_odds_ratio"] > 1.0
    assert low["target_calibration_pass"] is True


def test_active_odds_ratio_recovers_common_or() -> None:
    calibration = _calibration()
    calibration["strata"]["pre1536"]["root_equal_base_rate"] = 0.05
    calibration["strata"]["pre3072"]["root_equal_base_rate"] = 0.90
    active = active_odds_ratio_for_common(calibration, 0.30, 1.50)
    assert active > 1.50


def test_g1_exclusions_include_s3_survivors() -> None:
    exclusions = exclusion_union()
    assert exclusions["counts"]["S3_surviving_inventory"] == 133
    assert exclusions["counts"]["S3_prior_exclusion_union"] == 2610
    assert exclusions["union_count"] >= 2610


def test_one_state_selection_is_deterministic_and_root_unique() -> None:
    records = [
        {
            "root_cluster": "fresh:1:1536",
            "record_id": "a",
            "stratum": "pre1536",
            "role": "source_control",
            "behavior_family": "family",
            "state_sha1": "1",
        },
        {
            "root_cluster": "fresh:1:1536",
            "record_id": "b",
            "stratum": "pre3072",
            "role": "source_success_window",
            "behavior_family": "family",
            "state_sha1": "2",
        },
        {
            "root_cluster": "fresh:2:1536",
            "record_id": "c",
            "stratum": "pre1536",
            "role": "source_control",
            "behavior_family": "family",
            "state_sha1": "3",
        },
    ]
    first = select_one_state_per_root(records)
    second = select_one_state_per_root(list(reversed(records)))
    assert first["selection_rows"] == second["selection_rows"]
    assert first["selection_sha256"] == second["selection_sha256"]
    assert len({row["root_cluster"] for row in first["records"]}) == 2

from __future__ import annotations

import numpy as np

from threes_rl.s3_power_preflight import (
    _scan_fresh_ids,
    beta_binomial_rho,
    independent_coherence_checks,
    odds_shift,
    standardized_log_odds,
    stratum_for_built_max,
)


def test_stratum_requires_exact_pre_milestone_rung() -> None:
    assert stratum_for_built_max(768) == "pre1536"
    assert stratum_for_built_max(1536) == "pre3072"
    assert stratum_for_built_max(384) is None
    assert stratum_for_built_max(3072) is None


def test_odds_shift_matches_requested_odds_ratio() -> None:
    base = np.asarray([0.05, 0.50, 0.95])
    shifted = odds_shift(base, 1.5)
    base_odds = base / (1.0 - base)
    shifted_odds = shifted / (1.0 - shifted)
    np.testing.assert_allclose(shifted_odds / base_odds, 1.5)


def test_beta_binomial_rho_is_bounded() -> None:
    means = np.asarray([0.0, 0.0, 0.25, 0.50])
    assert 0.0 <= beta_binomial_rho(means, 8) <= 0.99
    assert beta_binomial_rho(np.asarray([0.0, 0.0]), 8) == 0.0


def test_standardized_log_odds_uses_equal_stratum_weights() -> None:
    control = {
        "pre1536": np.asarray([0.1, 0.1]),
        "pre3072": np.asarray([0.8, 0.8]),
    }
    treatment = {
        "pre1536": np.asarray([0.2, 0.2]),
        "pre3072": np.asarray([0.9, 0.9]),
    }
    expected = 0.5 * (
        np.log((0.2 / 0.8) / (0.1 / 0.9))
        + np.log((0.9 / 0.1) / (0.8 / 0.2))
    )
    assert np.isclose(standardized_log_odds(control, treatment), expected)


def test_prior_gate_scanner_normalizes_old_ancestry_keys(tmp_path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(
        '{"ancestry_key":"root:fresh:runs/example/replay.json:2563:0",'
        '"root_seed":2581,"root_cluster":"fresh:2601:1536"}'
    )
    assert _scan_fresh_ids(path) == {
        "fresh:2563:1536",
        "fresh:2581:1536",
        "fresh:2601:1536",
    }


def test_coherence_flags_are_independent() -> None:
    power_rows = [
        {
            "power": {
                "1.50": {
                    "conditional_independent": {
                        "power_lower_ci_gt_1": 0.49,
                    }
                }
            },
            "runtime_storage": {"compact_storage_bytes": 1_000_000},
        }
    ]
    feasibility = {
        "designs": {
            "192": {
                "root_count_feasible": False,
                "available_pool_family_cap_pass": False,
                "roots_per_stratum_required": 96,
                "feasible": False,
            }
        }
    }
    checks = independent_coherence_checks(
        power_rows,
        feasibility,
        free_disk_bytes=130 * 1024**3,
        selected_design=None,
    )
    assert checks == {
        "power_or_1_50_ge_80pct": False,
        "root_count_at_least_48_each_stratum": False,
        "available_pool_family_cap_40pct": False,
        "joint_root_family_selection_feasible": False,
        "compact_storage_below_10gib": True,
        "free_disk_above_100gib": True,
        "selected_coherent_design": False,
    }


def test_coherence_storage_failure_does_not_alias_other_checks() -> None:
    power_rows = [
        {
            "power": {
                "1.50": {
                    "conditional_independent": {
                        "power_lower_ci_gt_1": 0.81,
                    }
                }
            },
            "runtime_storage": {"compact_storage_bytes": 11 * 1024**3},
        }
    ]
    feasibility = {
        "designs": {
            "192": {
                "root_count_feasible": True,
                "available_pool_family_cap_pass": True,
                "roots_per_stratum_required": 96,
                "feasible": True,
            }
        }
    }
    checks = independent_coherence_checks(
        power_rows,
        feasibility,
        free_disk_bytes=90 * 1024**3,
        selected_design=None,
    )
    assert checks["power_or_1_50_ge_80pct"] is True
    assert checks["root_count_at_least_48_each_stratum"] is True
    assert checks["available_pool_family_cap_40pct"] is True
    assert checks["joint_root_family_selection_feasible"] is True
    assert checks["compact_storage_below_10gib"] is False
    assert checks["free_disk_above_100gib"] is False

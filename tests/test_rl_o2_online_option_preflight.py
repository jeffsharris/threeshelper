from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from threes_rl import o2_online_option_preflight as o2


def test_bound_documents_include_authoritative_a4() -> None:
    expected = {
        o2.CHARTER_PATH: o2.CHARTER_SHA256,
        o2.A1_PATH: o2.A1_SHA256,
        o2.A2_PATH: o2.A2_SHA256,
        o2.A3_PATH: o2.A3_SHA256,
        o2.A4_PATH: o2.A4_SHA256,
        o2.O1_GEOMETRY_PATH: o2.O1_GEOMETRY_SHA256,
        o2.PILOT_V2_LOCK_PATH: o2.PILOT_V2_LOCK_SHA256,
        o2.G3_COST_PATH: o2.G3_COST_SHA256,
    }
    assert o2.VERSION.endswith("_a4")
    for path, digest in expected.items():
        assert o2.sha256_path(path) == digest
    audit = o2.bound_document_audit()
    assert audit["passes"]
    assert all(audit["checks"].values())


def test_a4_wilson_minima_match_final_disjoint_demand() -> None:
    lower_rate = 18 / 640
    transfer_rate = 20 / 640
    assert o2.wilson_lower(6, 128) < lower_rate
    assert o2.wilson_lower(7, 128) > lower_rate
    assert o2.wilson_lower(7, 128) < transfer_rate
    assert o2.wilson_lower(8, 128) > transfer_rate
    assert math.isclose(
        o2.wilson_lower(7, 128),
        0.02991899584928971,
        rel_tol=0.0,
        abs_tol=1e-16,
    )
    assert math.isclose(
        o2.wilson_lower(8, 128),
        0.03557167190355302,
        rel_tol=0.0,
        abs_tol=1e-16,
    )


def test_a4_design_has_distinct_structural_and_availability_layers() -> None:
    design = o2.design_manifest()
    structural = design["pilot_structural_cells"]
    availability = design["pilot_availability_cells"]

    assert design["version"].endswith("_a4")
    assert design["passes"]
    assert all(design["checks"].values())
    assert len(structural) == len(availability) == 20
    assert sum(row["quota"] for row in structural) == 92
    assert Counter(
        (
            "transfer" if row["target"] == 768 else "lower",
            row["quota"],
        )
        for row in structural
    ) == {("lower", 4): 16, ("transfer", 7): 4}
    assert all(row["root_disjoint"] for row in structural)
    assert not any(row["wilson_yield_claim"] for row in structural)

    assert Counter(
        (
            "transfer" if row["target"] == 768 else "lower",
            row["minimum_distinct_whole_roots"],
        )
        for row in availability
    ) == {("lower", 7): 16, ("transfer", 8): 4}
    assert Counter(
        (
            "transfer" if row["target"] == 768 else "lower",
            row["final_disjoint_demand"],
        )
        for row in availability
    ) == {("lower", 18): 16, ("transfer", 20): 4}
    assert all(
        row["wilson_lower"] > row["final_rate_required"]
        for row in availability
    )
    assert all(row["root_may_support_other_cells"] for row in availability)
    assert all(
        row["root_counted_at_most_once_in_this_cell"]
        for row in availability
    )


def test_a4_rejects_fully_disjoint_wilson_pilot_interpretation() -> None:
    design = o2.design_manifest()
    contract = design["pilot_contract"]
    assert contract["complete_unconditionally_retained_roots"] == 128
    assert contract["fully_disjoint_wilson_slots"] == 144
    assert not contract["fully_disjoint_wilson_interpretation_allowed"]
    assert contract["decision_requires_both_layers"]
    assert 16 * 7 + 4 * 8 > 128


def test_final_allocator_remains_disjoint_128_48_192() -> None:
    design = o2.design_manifest()
    train = design["train_cells"]
    development = design["development_cells"]
    test = design["untouched_test_cells"]
    assert sum(row["quota"] for row in train) == 128
    assert sum(row["quota"] for row in development) == 48
    assert sum(row["quota"] for row in test) == 192
    assert {row["target"] for row in train} == {48, 96, 192, 384}
    assert sum(
        row["quota"] for row in development if row["target"] == 768
    ) == 16
    assert sum(row["quota"] for row in test if row["target"] == 768) == 64
    assert not any(
        row["target"] == 1536 for row in train + development + test
    )
    roots = design["prospective_corpus_roots"]
    assert len(roots) == 640
    assert Counter(row["family"] for row in roots) == {
        family[0]: 160 for family in o2.FAMILIES
    }


def test_family_signature_and_policy_artifact_lock_is_exact() -> None:
    audit = o2.family_evidence_audit()
    assert audit["passes"]
    assert all(audit["checks"].values())
    assert audit["family_order"] == [row[0] for row in o2.FAMILIES]
    assert len(audit["pairwise"]) == 6
    assert all(row["passes"] for row in audit["pairwise"])


def test_stream_manifest_counts_and_crn_pairing() -> None:
    rows = o2.stream_rows()
    audit = o2.internal_stream_audit(rows)
    assert audit["passes"]
    assert audit["purpose_counts"] == {
        "confirmation": 5120,
        "corpus": 640,
        "learning": 1024,
        "mechanism": 3840,
        "normal_development": 768,
        "pilot": 128,
    }
    assert audit["row_count"] == 11_520

    mechanism = [
        row
        for row in rows
        if row["purpose"] == "mechanism"
        and row["trajectory_code"] == 3_000_000
    ]
    assert len(mechanism) == 2
    assert len({row["logical_seed"] for row in mechanism}) == 1
    assert len({row["deck_stream_id"] for row in mechanism}) == 1
    assert len({row["slot_stream_id"] for row in mechanism}) == 1
    assert len({row["policy_stream_id"] for row in mechanism}) == 2


def test_stream_audit_fails_on_policy_id_reuse() -> None:
    rows = o2.stream_rows()
    rows[1]["policy_stream_id"] = rows[0]["policy_stream_id"]
    audit = o2.internal_stream_audit(rows)
    assert not audit["passes"]
    assert not audit["checks"]["policy_ids_globally_unique"]


def test_historical_calibration_reproduces_frozen_aggregates() -> None:
    report = o2.historical_calibration()
    assert report["passes"]
    assert all(report["checks"].values())
    development = report["development_d0_d2"]
    confirmation = report["spent_confirmation_sensitivity"]
    assert (
        development["pairs"],
        development["control_p3072_count"],
        development["treatment_p3072_count"],
        development["both_p3072_count"],
    ) == (768, 29, 40, 2)
    assert (
        confirmation["pairs"],
        confirmation["control_p3072_count"],
        confirmation["treatment_p3072_count"],
        confirmation["both_p3072_count"],
    ) == (512, 21, 21, 3)
    assert math.isclose(
        report["conservative_score_sd"],
        1.1804313028078002,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_mechanism_power_simulation_is_deterministic_on_small_fixture() -> None:
    first = o2.simulate_mechanism_power(
        "lower", 1.5, designs=4, bootstraps=19
    )
    second = o2.simulate_mechanism_power(
        "lower", 1.5, designs=4, bootstraps=19
    )
    assert first == second
    assert first["n_roots"] == 128
    assert first["strata"] == 16
    assert 0.0 <= first["full_gate_power"] <= 1.0


def test_capability_power_simulation_is_deterministic_on_small_fixture() -> None:
    base_rates = [0.03, 0.04, 0.05, 0.02, 0.03, 0.04, 0.05, 0.02]
    first = o2.simulate_capability_power(
        n_roots=384,
        odds_ratio=1.5,
        base_rates=base_rates,
        coupling=0.1,
        calibration_name="D0_D2",
        designs=4,
        bootstraps=19,
    )
    second = o2.simulate_capability_power(
        n_roots=384,
        odds_ratio=1.5,
        base_rates=base_rates,
        coupling=0.1,
        calibration_name="D0_D2",
        designs=4,
        bootstraps=19,
    )
    assert first == second
    assert first["n_roots"] == 384
    assert first["roots_per_stream_stratum"] == 48
    assert 0.0 <= first["full_gate_power"] <= 1.0


def test_resource_projection_and_score_mde_contract() -> None:
    report = o2.resource_projection()
    assert report["passes"]
    assert all(report["checks"].values())
    assert (
        report["phases"]["pilot_plus_corpus"]["projected_bytes"]
        == 2_503_888_832
    )
    sd = 1.1804313028078002
    expected_768 = math.exp(
        (1.959963984540054 + 0.8416212335729143)
        * sd
        / math.sqrt(768)
    ) - 1
    assert math.isclose(
        expected_768,
        0.12674611027876592,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


@pytest.mark.parametrize(
    ("integrity", "power", "resources", "operations", "expected"),
    (
        (
            False,
            True,
            True,
            True,
            "KILL_O2_PREFLIGHT_INTEGRITY",
        ),
        (
            True,
            False,
            True,
            True,
            "HOLD_O2_COST_OR_POWER",
        ),
        (
            True,
            True,
            False,
            True,
            "HOLD_O2_COST_OR_POWER",
        ),
        (
            True,
            True,
            True,
            False,
            "HOLD_O2_COST_OR_POWER",
        ),
        (
            True,
            True,
            True,
            True,
            "READY_O2_YIELD_PILOT_PREFLIGHT",
        ),
    ),
)
def test_decision_is_fail_closed(
    integrity: bool,
    power: bool,
    resources: bool,
    operations: bool,
    expected: str,
) -> None:
    assert (
        o2.decide(
            integrity_passes=integrity,
            power_passes=power,
            resource_passes=resources,
            operational_passes=operations,
        )
        == expected
    )


def test_immutable_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "sealed.json"
    o2.write_immutable_json(path, {"version": "test", "value": 1})
    payload = json.loads(path.read_text())
    assert o2.verify_payload_hash(payload)
    with pytest.raises(FileExistsError):
        o2.write_immutable_json(path, {"version": "changed"})


def test_wilson_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        o2.wilson_lower(-1, 128)
    with pytest.raises(ValueError):
        o2.wilson_lower(129, 128)
    with pytest.raises(ValueError):
        o2.wilson_lower(0, 0)

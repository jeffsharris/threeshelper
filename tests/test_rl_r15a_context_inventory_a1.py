from __future__ import annotations

import math

from threes_rl.r15a_context_inventory_a1 import (
    assign_partition_weights,
    family_stratified_split,
    waterfill_family_masses,
)


def _records(family_counts: dict[str, int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for family, count in family_counts.items():
        for index in range(count):
            root = f"{family}-root-{index}"
            for state_index in range(1 + index % 3):
                records.append(
                    {
                        "record_id": f"{root}-state-{state_index}",
                        "root_cluster": root,
                        "behavior_family": family,
                    }
                )
    return records


def test_family_stratified_split_is_deterministic_and_holds_out_twenty_percent() -> None:
    roots = [f"root-{index}" for index in range(10)]
    train, holdout = family_stratified_split(roots)
    reversed_train, reversed_holdout = family_stratified_split(list(reversed(roots)))

    assert len(train) == 8
    assert len(holdout) == 2
    assert train.isdisjoint(holdout)
    assert train | holdout == set(roots)
    assert (train, holdout) == (reversed_train, reversed_holdout)


def test_family_stratified_split_keeps_singleton_in_training() -> None:
    train, holdout = family_stratified_split(["only-root"])

    assert train == {"only-root"}
    assert holdout == set()


def test_waterfill_caps_dominant_family_without_duplicating_roots() -> None:
    masses = waterfill_family_masses(
        {
            "phaseblend": 300,
            "legacy_td": 40,
            "expectimax": 30,
            "cheap": 10,
            "random": 3,
        }
    )

    assert math.isclose(sum(masses.values()), 1.0, abs_tol=1e-12)
    assert math.isclose(masses["phaseblend"], 0.40, abs_tol=1e-12)
    assert max(masses.values()) <= 0.40 + 1e-12
    assert masses["legacy_td"] > masses["expectimax"] > masses["cheap"] > masses["random"]


def test_partition_weights_balance_roots_and_states_exactly() -> None:
    records = _records(
        {
            "phaseblend": 30,
            "legacy_td": 4,
            "expectimax": 3,
            "cheap": 2,
            "random": 1,
        }
    )

    summary = assign_partition_weights(records, "train")

    assert math.isclose(summary["state_fit_weight_sum"], 1.0, abs_tol=1e-12)
    assert math.isclose(summary["state_root_balanced_metric_weight_sum"], 1.0, abs_tol=1e-12)
    assert math.isclose(summary["state_family_balanced_metric_weight_sum"], 1.0, abs_tol=1e-12)
    assert summary["maximum_effective_fit_family_weight"] <= 0.40 + 1e-12

    root_fit_mass: dict[str, float] = {}
    family_fit_mass: dict[str, float] = {}
    for record in records:
        root = str(record["root_cluster"])
        family = str(record["behavior_family"])
        weight = float(record["fit_weight"])
        root_fit_mass[root] = root_fit_mass.get(root, 0.0) + weight
        family_fit_mass[family] = family_fit_mass.get(family, 0.0) + weight

    assert math.isclose(family_fit_mass["phaseblend"], 0.40, abs_tol=1e-12)
    for family in {str(record["behavior_family"]) for record in records}:
        family_roots = [
            mass for root, mass in root_fit_mass.items() if root.startswith(f"{family}-root-")
        ]
        assert max(family_roots) - min(family_roots) < 1e-12

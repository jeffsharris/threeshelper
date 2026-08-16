from __future__ import annotations

import numpy as np

from threes_rl.c1_search_optimization import (
    _allocate_split,
    _prune_dict,
    _runtime_gate_checks,
    _values_close,
    deterministic_key,
)


def test_deterministic_key_is_stable() -> None:
    assert deterministic_key("a", 1) == deterministic_key("a", 1)
    assert deterministic_key("a", 1) != deterministic_key("a", 2)


def test_allocate_split_uses_all_available_families() -> None:
    by_family = {
        "large": [{"id": f"large-{index}"} for index in range(20)],
        "small_a": [{"id": f"a-{index}"} for index in range(5)],
        "small_b": [{"id": f"b-{index}"} for index in range(5)],
    }

    selected = _allocate_split(by_family, 12)

    assert len(selected) == 12
    assert any(row["id"].startswith("large") for row in selected)
    assert any(row["id"].startswith("a-") for row in selected)
    assert any(row["id"].startswith("b-") for row in selected)


def test_values_close_uses_frozen_relative_tolerance() -> None:
    assert _values_close({"up": 100.0}, [(0, 100.0 + 1e-8)])[0]
    assert not _values_close({"up": 100.0}, [(0, 100.0 + 1e-5)])[0]


def test_prune_dict_keeps_newest_insertions() -> None:
    values = {index: index for index in range(5)}
    _prune_dict(values, 3)
    assert values == {2: 2, 3: 3, 4: 4}


def test_runtime_gate_checks_frozen_thresholds() -> None:
    passing = np.asarray([2.0, 2.5, 3.0, 4.0])
    assert all(_runtime_gate_checks(passing, exact=True, deterministic=True).values())
    failing = np.asarray([2.0] * 8 + [9.0, 13.0])
    checks = _runtime_gate_checks(failing, exact=True, deterministic=True)
    assert not checks["p90_le_5x"]
    assert not checks["p99_le_8x"]
    assert not checks["max_le_12x"]

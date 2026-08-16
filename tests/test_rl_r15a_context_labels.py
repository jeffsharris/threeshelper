from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from threes_rl.r15a_context_labels import (
    collect_prior_stream_ids,
    return_bin,
    select_audit_records,
)


def test_return_bins_match_frozen_edges() -> None:
    assert return_bin(-10.0) == 0
    assert return_bin(0.0) == 0
    assert return_bin(999.9) == 0
    assert return_bin(1_000.0) == 1
    assert return_bin(63_999.0) == 6
    assert return_bin(64_000.0) == 7
    assert return_bin(float("inf")) == 7


def test_collect_prior_stream_ids_walks_nested_manifests(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "blocks": {
                    "D0": [
                        {
                            "logical_seed": 1,
                            "deck_stream_id": 2,
                            "slot_stream_id": 3,
                            "policy_stream_id": 4,
                        }
                    ]
                }
            }
        )
    )

    exogenous, logical, summary = collect_prior_stream_ids([path])

    assert exogenous == {2, 3, 4}
    assert logical == {1}
    assert summary == {"files_scanned": 1, "invalid_files": 0}


def test_collect_prior_stream_ids_ignores_summary_lists(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    path.write_text(
        json.dumps(
            {
                "deck_stream_id": [1, 2, 3],
                "slot_stream_id": {"min": 4, "max": 5},
                "policy_stream_id": None,
                "logical_seed": "summary",
            }
        )
    )

    exogenous, logical, summary = collect_prior_stream_ids([path])

    assert exogenous == set()
    assert logical == set()
    assert summary == {"files_scanned": 1, "invalid_files": 0}


def test_audit_selection_is_partition_balanced_and_deterministic() -> None:
    records = []
    for partition in ("train", "ancestry_holdout", "family_holdout"):
        for index in range(20):
            records.append(
                {
                    "partition": partition,
                    "context_cell": f"cell-{index}",
                    "record_id": f"{partition}-{index}",
                }
            )

    selected = select_audit_records(records)
    shuffled = select_audit_records(list(reversed(records)))

    assert selected == shuffled
    assert len(selected) == 24
    assert {record_id.split("-", 1)[0] for record_id in selected} == {
        "train",
        "ancestry_holdout",
        "family_holdout",
    }


def test_return_bin_is_monotonic() -> None:
    values = np.linspace(-1_000, 100_000, 1_000)
    bins = [return_bin(float(value)) for value in values]
    assert bins == sorted(bins)

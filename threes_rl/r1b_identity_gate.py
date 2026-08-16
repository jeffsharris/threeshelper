"""Verify exact D0 outcome identity for an untrained residual composite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from threes_rl.ntuple import NtupleValue, ResidualStagedNtupleValue
from threes_rl.run_artifacts import write_json


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def gate(*, checkpoint: Path, incumbent_results: Path, candidate_results: Path) -> dict[str, object]:
    incumbent = _rows(incumbent_results)
    candidate = _rows(candidate_results)
    fields = (
        "block",
        "index",
        "logical_seed",
        "starter_tile",
        "deck_stream_id",
        "slot_stream_id",
        "policy_stream_id",
        "score",
        "score_minus_starter",
        "moves",
        "max_tile",
        "max_tile_excl_starter",
        "terminal_tile",
    )
    mismatches = []
    for idx, (base, cand) in enumerate(zip(incumbent, candidate)):
        changed = [field for field in fields if base.get(field) != cand.get(field)]
        if changed:
            mismatches.append({"row": idx, "fields": changed})
    if len(incumbent) != len(candidate):
        mismatches.append({"row_count": [len(incumbent), len(candidate)]})

    model = NtupleValue.load(checkpoint, mmap_mode="r")
    if not isinstance(model, ResidualStagedNtupleValue):
        raise TypeError("Identity gate checkpoint is not a residual composite")
    residual_arrays = [
        array
        for stage in model.residual.stages
        for array in [*stage.tables, *(stage.tc_sum_tables or []), *(stage.tc_abs_tables or [])]
    ]
    residual_nonzero = int(sum(np.count_nonzero(array) for array in residual_arrays))
    promoted = int(sum(model.residual.promotion_counts))
    frozen_read_only = all(not array.flags.writeable for array in model.frozen_arrays)
    passed = not mismatches and len(candidate) == 64 and residual_nonzero == 0 and promoted == 0 and frozen_read_only
    return {
        "decision": "PASS" if passed else "HOLD",
        "games": len(candidate),
        "outcome_mismatches": len(mismatches),
        "first_mismatches": mismatches[:10],
        "residual_nonzero_entries": residual_nonzero,
        "promotion_count": promoted,
        "frozen_arrays_read_only": frozen_read_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--incumbent-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = gate(
        checkpoint=args.checkpoint,
        incumbent_results=args.incumbent_results,
        candidate_results=args.candidate_results,
    )
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

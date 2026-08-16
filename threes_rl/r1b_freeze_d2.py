"""Freeze the R1b D2 block and prove cross-manifest stream disjointness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from threes_rl.eval_stream_manifest import build_stream_manifest
from threes_rl.run_artifacts import write_json


STREAM_FIELDS = ("deck_stream_id", "slot_stream_id", "policy_stream_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(*, existing_path: Path, out_path: Path, audit_path: Path) -> dict[str, Any]:
    if out_path.exists() or audit_path.exists():
        raise FileExistsError("D2 freeze refuses to overwrite an existing manifest or audit")
    existing = json.loads(existing_path.read_text())
    d2 = build_stream_manifest(
        namespace="threes-r1b-d2-20260709",
        block_sizes={"D2": 512},
        starter_tile=1536,
        logical_seed_start=5_000_000,
    )
    d2["manifest_version"] = "r1b_d2_paired_normal_start_v1"
    d2["confirmation_status"] = "development_frozen_untouched"

    existing_by_field: dict[str, set[int]] = {field: set() for field in STREAM_FIELDS}
    existing_logical: set[int] = set()
    existing_counts: dict[str, int] = {}
    for block_name in ("D0", "D1", "C"):
        rows = list(existing["blocks"][block_name])
        existing_counts[block_name] = len(rows)
        for row in rows:
            existing_logical.add(int(row["logical_seed"]))
            for field in STREAM_FIELDS:
                existing_by_field[field].add(int(row[field]))

    d2_rows = list(d2["blocks"]["D2"])
    collisions = {
        field: sorted(
            int(row[field])
            for row in d2_rows
            if int(row[field]) in existing_by_field[field]
        )
        for field in STREAM_FIELDS
    }
    logical_collisions = sorted(
        int(row["logical_seed"])
        for row in d2_rows
        if int(row["logical_seed"]) in existing_logical
    )
    all_d2_ids = [int(row[field]) for row in d2_rows for field in STREAM_FIELDS]
    internal_unique = len(all_d2_ids) == len(set(all_d2_ids))
    passed = internal_unique and not logical_collisions and all(not values for values in collisions.values())
    audit = {
        "decision": "PASS" if passed else "HOLD",
        "existing_manifest": str(existing_path),
        "existing_manifest_sha256": _sha256(existing_path),
        "existing_block_counts": existing_counts,
        "c_outcomes_read": False,
        "d2_manifest": str(out_path),
        "d2_games": len(d2_rows),
        "d2_internal_stream_ids_unique": internal_unique,
        "cross_block_collisions": collisions,
        "logical_seed_collisions": logical_collisions,
    }
    if not passed:
        write_json(audit_path, audit)
        raise RuntimeError("D2 stream disjointness gate failed")
    write_json(out_path, d2)
    write_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(existing_path=args.existing_manifest, out_path=args.out, audit_path=args.audit)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

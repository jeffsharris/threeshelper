"""Write compact, outcome-free provenance supplements for the sealed S3 hold."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from threes_rl.run_artifacts import write_json
from threes_rl.s3_power_preflight import (
    SOURCE_INVENTORY_PATH,
    _json,
    natural_root_candidates,
    prior_exclusion_catalog,
    sha256_path,
)


VERSION = "s3_provenance_seal_v1"


def root_list_sha256(roots: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(roots)).encode("utf-8")).hexdigest()


def compact_candidate(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "state"
    }


def build_seal(preflight_path: Path) -> dict[str, Any]:
    preflight = _json(preflight_path)
    if preflight.get("decision") != "HOLD_UNDERPOWERED_PREFLIGHT":
        raise ValueError("S3 provenance may only seal the frozen underpowered hold")
    if preflight.get("outcomes_generated") or preflight.get(
        "treatment_outcomes_inspected"
    ):
        raise ValueError("Refusing to seal an S3 artifact with policy outcomes")

    source_inventory = _json(SOURCE_INVENTORY_PATH)
    exclusions = prior_exclusion_catalog()
    candidates = natural_root_candidates(
        source_inventory,
        exclusions["roots"],
    )
    roots = sorted(exclusions["roots"])
    records = [
        compact_candidate(record)
        for record in candidates["records"]
    ]
    candidate_roots = sorted(
        {str(record["root_cluster"]) for record in records}
    )
    return {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scientific_decision": "HOLD_UNDERPOWERED_PREFLIGHT",
        "outcomes_generated": False,
        "treatment_outcomes_inspected": False,
        "source_preflight": str(preflight_path),
        "source_preflight_sha256": sha256_path(preflight_path),
        "source_inventory": str(SOURCE_INVENTORY_PATH),
        "source_inventory_sha256": sha256_path(SOURCE_INVENTORY_PATH),
        "excluded_roots": {
            "count": len(roots),
            "sha256": root_list_sha256(roots),
            "roots": roots,
            "sources": exclusions["sources"],
            "counts": exclusions["counts"],
        },
        "surviving_inventory": {
            "record_count": len(records),
            "root_count": len(candidate_roots),
            "root_sha256": root_list_sha256(candidate_roots),
            "roots": candidate_roots,
            "scan_counts": candidates["counts"],
            "records": records,
        },
        "integrity": {
            "matches_preflight_exclusion_count": (
                len(roots)
                == int(preflight["exclusion_catalog"]["union_count"])
            ),
            "matches_preflight_exclusion_hash": (
                root_list_sha256(roots)
                == preflight["exclusion_catalog"]["root_list_sha256"]
            ),
            "matches_preflight_candidate_records": (
                len(records)
                == int(preflight["candidate_catalog"]["records"])
            ),
            "matches_preflight_candidate_roots": (
                len(candidate_roots)
                == int(preflight["candidate_catalog"]["unique_roots"])
            ),
            "state_payloads_omitted": all(
                "state" not in record for record in records
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_seal(args.preflight)
    if not all(payload["integrity"].values()):
        raise RuntimeError(f"S3 provenance seal mismatch: {payload['integrity']}")
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": sha256_path(args.out),
                "excluded_roots": payload["excluded_roots"]["count"],
                "surviving_roots": payload["surviving_inventory"]["root_count"],
                "surviving_records": payload["surviving_inventory"]["record_count"],
                "integrity": payload["integrity"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

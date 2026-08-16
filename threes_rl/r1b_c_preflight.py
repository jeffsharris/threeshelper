"""Freeze hashes, stream checks, and thresholds before opening R1b confirmation C."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from threes_rl.run_artifacts import write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> dict[str, Any]:
    files = sorted(child for child in path.rglob("*") if child.is_file() and not child.is_symlink())
    digest = hashlib.sha256()
    total_bytes = 0
    for child in files:
        relative = str(child.relative_to(path))
        size = child.stat().st_size
        content_hash = file_sha256(child)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {"sha256": digest.hexdigest(), "files": len(files), "bytes": total_bytes}


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def freeze(
    *,
    primary_manifest_path: Path,
    other_manifest_paths: list[Path],
    incumbent_policy_path: Path,
    incumbent_components: list[Path],
    candidate_checkpoint: Path,
) -> dict[str, Any]:
    primary = json.loads(primary_manifest_path.read_text())
    c_rows = primary["blocks"]["C"]
    all_stream_ids: set[int] = set()
    all_logical_ids: set[int] = set()
    collisions: list[dict[str, Any]] = []
    manifests = [primary_manifest_path, *other_manifest_paths]
    for path in manifests:
        manifest = primary if path == primary_manifest_path else json.loads(path.read_text())
        grouped_rows = manifest.get("blocks")
        if not isinstance(grouped_rows, dict):
            grouped_rows = {"diagnostic": manifest.get("jobs", [])}
        for block, rows in grouped_rows.items():
            for row in rows:
                logical = int(row["logical_seed"])
                if logical in all_logical_ids:
                    collisions.append({"path": str(path), "block": block, "kind": "logical", "id": logical})
                all_logical_ids.add(logical)
                for field in ("deck_stream_id", "slot_stream_id", "policy_stream_id"):
                    value = int(row[field])
                    if value in all_stream_ids:
                        collisions.append({"path": str(path), "block": block, "kind": field, "id": value})
                    all_stream_ids.add(value)

    candidate_meta = json.loads((candidate_checkpoint / "meta.json").read_text())
    checks = {
        "c_has_512_rows": len(c_rows) == 512,
        "zero_stream_or_logical_collisions": not collisions,
        "candidate_is_exact_5000_checkpoint": int(candidate_meta.get("games_completed", -1)) == 5000,
        "confirmation_status_was_sealed": primary.get("confirmation_status") == "frozen_untouched",
    }
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "PASS" if all(checks.values()) else "HOLD",
        "checks": checks,
        "collisions": collisions,
        "primary_manifest": {
            "path": str(primary_manifest_path),
            "sha256": file_sha256(primary_manifest_path),
            "c_rows_canonical_sha256": canonical_json_sha256(c_rows),
            "c_rows": len(c_rows),
        },
        "other_manifests": [
            {"path": str(path), "sha256": file_sha256(path)} for path in other_manifest_paths
        ],
        "identifier_counts": {
            "logical": len(all_logical_ids),
            "exogenous_stream": len(all_stream_ids),
        },
        "incumbent_policy_file": {
            "path": str(incumbent_policy_path),
            "sha256": file_sha256(incumbent_policy_path),
            "policy": incumbent_policy_path.read_text().splitlines()[-1],
        },
        "incumbent_components": [
            {"path": str(path), **tree_sha256(path)} for path in incumbent_components
        ],
        "candidate_checkpoint": {
            "path": str(candidate_checkpoint),
            **tree_sha256(candidate_checkpoint),
            "games_completed": int(candidate_meta["games_completed"]),
        },
        "frozen_decision_rules": {
            "promote_score": "paired score-minus-starter 95% bootstrap CI lower bound > 0",
            "p3072_noninferiority": -0.02,
            "material_lower_decile_regression": (
                "lower-decile difference CI upper < 0 OR point estimate <= -10% of frozen C incumbent mean"
            ),
            "tail_selection": (
                "12 largest paired losses plus up to 12 new crossings below frozen incumbent C P5"
            ),
            "catastrophic_tail_rate": (
                "below-P5 rate difference > +2 pp and 95% paired bootstrap CI lower > 0"
            ),
            "corner_mechanism": (
                ">=3 candidate-only final anchor losses OR >=3 candidate-only terminal max displacements"
            ),
            "high_score": "diagnostic_only",
        },
        "execution_contract": {
            "incumbent_runs": 1,
            "candidate_runs": 1,
            "partial_peeking": False,
            "retraining_or_config_change": False,
            "paired_evaluator": "split_exogenous_v1",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--other-manifest", type=Path, action="append", default=[])
    parser.add_argument("--incumbent-policy-file", type=Path, required=True)
    parser.add_argument("--incumbent-component", type=Path, action="append", required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(
        primary_manifest_path=args.primary_manifest,
        other_manifest_paths=args.other_manifest,
        incumbent_policy_path=args.incumbent_policy_file,
        incumbent_components=args.incumbent_component,
        candidate_checkpoint=args.candidate_checkpoint,
    )
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "checks": payload["checks"],
                "identifier_counts": payload["identifier_counts"],
                "primary_manifest": payload["primary_manifest"],
                "incumbent_components": payload["incumbent_components"],
                "candidate_checkpoint": payload["candidate_checkpoint"],
                "frozen_decision_rules": payload["frozen_decision_rules"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

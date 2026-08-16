"""Corrected outcome-free integrity preflight for the G3 bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from threes_rl import g1r_acquire as g1r
from threes_rl import g3_scale_transfer_bootstrap_preflight as v1
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.s3_power_preflight import sha256_path


VERSION = "g3_scale_transfer_bootstrap_preflight_v2"
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = Path(
    "threes_rl/G3_SCALE_TRANSFER_BOOTSTRAP_CHARTER_AMENDMENT_V2_INTEGRITY.md"
)
IMPLEMENTATION_PATH = Path(
    "threes_rl/g3_scale_transfer_bootstrap_preflight_v2.py"
)
TEST_PATH = Path("tests/test_rl_g3_scale_transfer_bootstrap_preflight_v2.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/"
    "g3_scale_transfer_bootstrap_preflight_v2_test_evidence.json"
)
OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/g3_scale_transfer_bootstrap_preflight_v2"
)
STAGING_DIR = Path(
    "threes_rl/runs/forensics/g3_scale_transfer_bootstrap_preflight_v2.staging"
)
V1_OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/g3_scale_transfer_bootstrap_preflight_v1"
)
V1_PREFLIGHT_PATH = V1_OUTPUT_DIR / "G3_BOOTSTRAP_PREFLIGHT.json"
V1_RECORD_MANIFEST_PATH = V1_OUTPUT_DIR / "G3_RECORD_MANIFEST.json"
V1_STREAM_MANIFEST_PATH = V1_OUTPUT_DIR / "G3_LABEL_STREAM_MANIFEST.json"

AMENDMENT_SHA256 = (
    "c60895f9f72c78d72481e0d3759f2a818c1165de5e2f8e0d1fe189dc85026aef"
)
V1_PREFLIGHT_FILE_SHA256 = (
    "0cd19d5a3d390df4f9d0165e72f20a130799ceafb0cb408e5274b8924a82d77a"
)
V1_PREFLIGHT_PAYLOAD_SHA256 = (
    "4513530c047292daf53ef0d7084db1384206aa1bf6403df22bc7edd14c8606ad"
)
V1_RECORD_FILE_SHA256 = (
    "938e903f8d2fefb072af84ac19baf4977e4f4d93bf72e8af7acc174b6974b9ec"
)
V1_RECORD_PAYLOAD_SHA256 = (
    "a78e2fd51ee20a7aeb23c71d9930c33561844357920f4808eeeaff653d49f759"
)
V1_STREAM_FILE_SHA256 = (
    "bdbe562167f304327e52f0593f0958753e8afa949a7b38e15b357492faea5744"
)
V1_STREAM_PAYLOAD_SHA256 = (
    "c2afc3c6fa26c1106a480c58189d9a9b4f9dcf99ac8b506d890ff3c330278caa"
)
V1_IMPLEMENTATION_SHA256 = (
    "27ac9cc6a5d4ee7449650ef0d886395233bea1e77ff7a7213e9142a614215234"
)
V1_TEST_SHA256 = (
    "d9c60a840d95c85cc180128641b09920d2e7214f09ccb00f711a8e0c061aa14a"
)
V1_TEST_EVIDENCE_SHA256 = (
    "a8f6a1768a293395c9a5583857ef77188ff41421a95a3cfabdfa62315a3dbd21"
)

V1_ARTIFACT_LOCKS = {
    v1.CHARTER_PATH: v1.CHARTER_SHA256,
    v1.AMENDMENT_PATH: v1.AMENDMENT_SHA256,
    v1.IMPLEMENTATION_PATH: V1_IMPLEMENTATION_SHA256,
    v1.TEST_PATH: V1_TEST_SHA256,
    v1.TEST_EVIDENCE_PATH: V1_TEST_EVIDENCE_SHA256,
    V1_PREFLIGHT_PATH: V1_PREFLIGHT_FILE_SHA256,
    V1_RECORD_MANIFEST_PATH: V1_RECORD_FILE_SHA256,
    V1_STREAM_MANIFEST_PATH: V1_STREAM_FILE_SHA256,
    AMENDMENT_PATH: AMENDMENT_SHA256,
}

TRANSFER_INPUT_NAMESPACE = v1.TRANSFER_RESULT_PATH.parent
RUNS_ROOT = v1.RUNS_ROOT
SELF_NAMESPACES = (OUTPUT_DIR, STAGING_DIR)
BOUND_V1_EVIDENCE = {
    V1_PREFLIGHT_PATH: V1_PREFLIGHT_FILE_SHA256,
    V1_RECORD_MANIFEST_PATH: V1_RECORD_FILE_SHA256,
    V1_STREAM_MANIFEST_PATH: V1_STREAM_FILE_SHA256,
}

EXPECTED_RECORDS = 715
EXPECTED_ORDINARY_RECORDS = 683
EXPECTED_TRANSFER_RECORDS = 32
EXPECTED_TOTAL_PATHS = 20_288
EXPECTED_E0_PATHS = 5_072
EXPECTED_E1_PATHS = 15_216
EXPECTED_STAGE_PATHS = {
    "E0": {
        "train": 3_902,
        "development": 944,
        "transfer_diagnostic": 226,
    },
    "E1": {
        "train": 11_706,
        "development": 2_832,
        "transfer_diagnostic": 678,
    },
}
MIN_FREE_GIB = v1.MIN_FREE_GIB
TARGET_FREE_GIB = v1.TARGET_FREE_GIB
MIN_NICE = v1.MIN_NICE


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _repo_abs(path: Path) -> Path:
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _payload_hash_matches(
    payload: dict[str, Any], field: str, expected: str
) -> bool:
    actual_field = payload.get(field)
    without_field = {key: value for key, value in payload.items() if key != field}
    return actual_field == expected and canonical_sha256(without_field) == expected


def _locked_file_rows(
    locks: Mapping[Path, str],
) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    for path, expected in locks.items():
        actual = sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    return rows, all(row["matches"] for row in rows)


def load_v1_manifests() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    preflight = _json(V1_PREFLIGHT_PATH)
    records = _json(V1_RECORD_MANIFEST_PATH)
    streams = _json(V1_STREAM_MANIFEST_PATH)
    checks = {
        "v1_preflight_file_hash":
            sha256_path(V1_PREFLIGHT_PATH) == V1_PREFLIGHT_FILE_SHA256,
        "v1_preflight_payload_hash":
            _payload_hash_matches(
                preflight,
                "canonical_payload_sha256",
                V1_PREFLIGHT_PAYLOAD_SHA256,
            ),
        "v1_decision_is_authoritative_kill":
            preflight.get("decision") == "KILL_G3_PREFLIGHT_INTEGRITY",
        "v1_zero_forbidden_work":
            preflight.get("zero_forbidden_work", {}).get("new_labels") == 0
            and preflight.get("zero_forbidden_work", {}).get(
                "models_fit"
            ) == 0
            and preflight.get("zero_forbidden_work", {}).get(
                "transfer_outcomes_opened"
            ) == 0,
        "record_file_hash":
            sha256_path(V1_RECORD_MANIFEST_PATH) == V1_RECORD_FILE_SHA256,
        "record_payload_hash":
            _payload_hash_matches(
                records,
                "canonical_payload_sha256",
                V1_RECORD_PAYLOAD_SHA256,
            ),
        "stream_file_hash":
            sha256_path(V1_STREAM_MANIFEST_PATH) == V1_STREAM_FILE_SHA256,
        "stream_payload_hash":
            _payload_hash_matches(
                streams,
                "canonical_payload_sha256",
                V1_STREAM_PAYLOAD_SHA256,
            ),
        "record_count_exact":
            len(records.get("records", [])) == EXPECTED_RECORDS,
        "stream_path_count_exact":
            len(streams.get("rows", [])) == EXPECTED_TOTAL_PATHS,
        "stream_replicates_exact":
            streams.get("replicates") == v1.REPLICATES,
        "streams_unconsumed": streams.get("streams_consumed") == 0,
        "record_outcomes_unopened":
            records.get("score_or_label_outcome_opened") is False,
    }
    audit = {
        "checks": checks,
        "passes": all(checks.values()),
        "v1_preflight_file_sha256": V1_PREFLIGHT_FILE_SHA256,
        "v1_preflight_payload_sha256": V1_PREFLIGHT_PAYLOAD_SHA256,
        "record_manifest_file_sha256": V1_RECORD_FILE_SHA256,
        "record_manifest_payload_sha256": V1_RECORD_PAYLOAD_SHA256,
        "stream_manifest_file_sha256": V1_STREAM_FILE_SHA256,
        "stream_manifest_payload_sha256": V1_STREAM_PAYLOAD_SHA256,
        "record_count": len(records.get("records", [])),
        "stream_path_count": len(streams.get("rows", [])),
    }
    return preflight, records, streams, audit


def _normalized_absolute_text(text: str) -> bool:
    return (
        os.path.isabs(text)
        and os.path.normpath(text) == text
        and "/./" not in text
        and "/../" not in text
        and not text.endswith("/.")
        and not text.endswith("/..")
    )


def _has_symlink_component(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    cursor = root
    if cursor.is_symlink():
        return True
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


def _path_integrity(
    raw_path: str,
    *,
    search_root: Path,
) -> tuple[Path | None, list[str]]:
    reasons = []
    if not _normalized_absolute_text(raw_path):
        reasons.append("path_not_normalized_absolute")
        return None, reasons
    path = Path(raw_path)
    root = search_root.resolve(strict=True)
    if not path.is_file():
        reasons.append("match_not_regular_file")
        return path, reasons
    try:
        path.relative_to(root)
    except ValueError:
        reasons.append("match_outside_scan_root")
    if _has_symlink_component(path, root):
        reasons.append("symlink_component")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        reasons.append("path_resolution_failed")
    else:
        if resolved != path:
            reasons.append("resolved_path_differs")
    return path, reasons


def classify_match_path(
    raw_path: str,
    *,
    search_root: Path,
    input_namespace: Path,
    self_namespaces: Sequence[Path],
    bound_evidence: Mapping[Path, str],
) -> dict[str, Any]:
    path, reasons = _path_integrity(raw_path, search_root=search_root)
    category = "external"
    expected_hash = None
    actual_hash = None
    if path is not None and path.is_file():
        actual_hash = sha256_path(path)
    if not reasons and path is not None:
        input_abs = _repo_abs(input_namespace).absolute()
        self_abs = [_repo_abs(item).absolute() for item in self_namespaces]
        evidence_abs = {
            _repo_abs(item).absolute(): expected
            for item, expected in bound_evidence.items()
        }
        if path.is_relative_to(input_abs):
            category = "excluded_input"
        elif any(path.is_relative_to(namespace) for namespace in self_abs):
            category = "excluded_self"
        elif path in evidence_abs:
            expected_hash = evidence_abs[path]
            if actual_hash == expected_hash:
                category = "bound_v1_evidence"
            else:
                reasons.append("bound_evidence_hash_mismatch")
        else:
            reasons.append("outside_exact_namespaces")
    return {
        "raw_path": raw_path,
        "path": str(path) if path is not None else None,
        "category": category,
        "reasons": reasons,
        "sha256": actual_hash,
        "expected_sha256": expected_hash,
    }


def _rg_token_matches(
    patterns: Iterable[str], search_root: Path
) -> list[str]:
    unique = sorted({str(pattern) for pattern in patterns if str(pattern)})
    if not unique:
        return []
    root = search_root.resolve(strict=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
        handle.write("\n".join(unique))
        handle.write("\n")
        handle.flush()
        result = subprocess.run(
            [
                "rg",
                "-l",
                "-F",
                "-f",
                handle.name,
                "--follow",
                str(root),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"Corrected token audit failed closed: {result.stderr.strip()}"
        )
    return sorted(line for line in result.stdout.splitlines() if line)


def _classified_scan(
    patterns: Iterable[str],
    *,
    search_root: Path,
    input_namespace: Path,
    self_namespaces: Sequence[Path],
    bound_evidence: Mapping[Path, str],
    match_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    matches = (
        list(match_paths)
        if match_paths is not None
        else _rg_token_matches(patterns, search_root)
    )
    rows = [
        classify_match_path(
            path,
            search_root=search_root,
            input_namespace=input_namespace,
            self_namespaces=self_namespaces,
            bound_evidence=bound_evidence,
        )
        for path in matches
    ]
    counts = Counter(row["category"] for row in rows)
    return {
        "raw_match_count": len(matches),
        "rows": rows,
        "category_counts": dict(sorted(counts.items())),
        "external_matches": [
            row for row in rows if row["category"] == "external"
        ],
        "passes": not any(row["category"] == "external" for row in rows),
    }


def corrected_transfer_untouched_audit(
    records: list[dict[str, Any]],
    *,
    search_root: Path = RUNS_ROOT,
    input_namespace: Path = TRANSFER_INPUT_NAMESPACE,
    self_namespaces: Sequence[Path] = SELF_NAMESPACES,
    bound_evidence: Mapping[Path, str] = BOUND_V1_EVIDENCE,
) -> dict[str, Any]:
    root_scan = _classified_scan(
        [row["root_cluster"] for row in records],
        search_root=search_root,
        input_namespace=input_namespace,
        self_namespaces=self_namespaces,
        bound_evidence=bound_evidence,
    )
    state_scan = _classified_scan(
        [row["state_sha1"] for row in records],
        search_root=search_root,
        input_namespace=input_namespace,
        self_namespaces=self_namespaces,
        bound_evidence=bound_evidence,
    )
    combined = {}
    for scan in (root_scan, state_scan):
        for row in scan["rows"]:
            combined[row["raw_path"]] = row
    category_paths: dict[str, list[str]] = defaultdict(list)
    category_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for path, row in sorted(combined.items()):
        category_paths[row["category"]].append(path)
        if row["sha256"]:
            category_hashes[row["category"]][path] = row["sha256"]
    checks = {
        "root_token_scan_has_no_external_match": root_scan["passes"],
        "state_token_scan_has_no_external_match": state_scan["passes"],
        "combined_external_matches_zero": not category_paths["external"],
        "no_broad_forensics_exclusion":
            tuple(self_namespaces) == SELF_NAMESPACES
            and input_namespace == TRANSFER_INPUT_NAMESPACE,
    }
    return {
        "version": "g3_v2_corrected_transfer_untouchedness_v1",
        "scan_root": str(_repo_abs(search_root).resolve(strict=True)),
        "exact_excluded_input_namespace":
            str(_repo_abs(input_namespace).absolute()),
        "exact_excluded_self_namespaces": [
            str(_repo_abs(path).absolute()) for path in self_namespaces
        ],
        "exact_bound_v1_evidence_files": {
            str(_repo_abs(path).absolute()): digest
            for path, digest in bound_evidence.items()
        },
        "root_token_scan": root_scan,
        "state_token_scan": state_scan,
        "excluded_input_matches": category_paths["excluded_input"],
        "excluded_input_match_hashes": category_hashes["excluded_input"],
        "excluded_self_matches": category_paths["excluded_self"],
        "excluded_self_match_hashes": category_hashes["excluded_self"],
        "bound_v1_evidence_matches": category_paths["bound_v1_evidence"],
        "bound_v1_evidence_match_hashes":
            category_hashes["bound_v1_evidence"],
        "true_external_matches": category_paths["external"],
        "true_external_match_hashes": category_hashes["external"],
        "checks": checks,
        "passes": all(checks.values()),
    }


def bind_transfer_panel_inputs(
    sources: list[dict[str, Any]],
    *,
    input_namespace: Path = TRANSFER_INPUT_NAMESPACE,
) -> dict[str, Any]:
    namespace_abs = _repo_abs(input_namespace).absolute()
    result_abs = _repo_abs(v1.TRANSFER_RESULT_PATH).absolute()
    rows = [
        {
            "role": "sealed_transfer_result",
            "path": str(result_abs),
            "expected_sha256": v1.TRANSFER_RESULT_FILE_SHA256,
            "actual_sha256":
                sha256_path(result_abs) if result_abs.is_file() else None,
            "byte_size":
                result_abs.stat().st_size if result_abs.is_file() else None,
        }
    ]
    for source in sources:
        for role, path_key, hash_key in (
            ("source_replay", "source_replay", "source_replay_sha256"),
            ("source_state", "source_state", "source_state_sha256"),
        ):
            raw = str(source[path_key])
            path = _repo_abs(Path(raw)).absolute()
            rows.append(
                {
                    "role": role,
                    "root_cluster": source["root_cluster"],
                    "path": str(path),
                    "source_path_text": raw,
                    "expected_sha256": source[hash_key],
                    "actual_sha256":
                        sha256_path(path) if path.is_file() else None,
                    "byte_size": path.stat().st_size if path.is_file() else None,
                }
            )
    failures = []
    for row in rows:
        path = Path(row["path"])
        raw = row.get("source_path_text")
        if raw is not None and os.path.normpath(raw) != raw:
            failures.append({"path": str(path), "reason": "source_path_alias"})
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        if not path.is_relative_to(namespace_abs):
            failures.append(
                {"path": str(path), "reason": "outside_exact_input_namespace"}
            )
        if _has_symlink_component(path, _repo_abs(RUNS_ROOT).resolve(strict=True)):
            failures.append(
                {"path": str(path), "reason": "symlink_component"}
            )
        if path.resolve(strict=True) != path:
            failures.append(
                {"path": str(path), "reason": "resolved_path_differs"}
            )
        if row["actual_sha256"] != row["expected_sha256"]:
            failures.append({"path": str(path), "reason": "hash_mismatch"})
    unique_paths = {row["path"] for row in rows}
    checks = {
        "source_count_exact": len(sources) == EXPECTED_TRANSFER_RECORDS,
        "one_result_plus_two_files_per_source":
            len(rows) == 1 + 2 * EXPECTED_TRANSFER_RECORDS,
        "all_bound_paths_unique": len(unique_paths) == len(rows),
        "all_files_exact": not failures,
    }
    return {
        "namespace": str(namespace_abs),
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
        "unique_file_count": len(unique_paths),
        "checked_bytes": sum(int(row["byte_size"] or 0) for row in rows),
        "failures": failures,
        "checks": checks,
        "passes": all(checks.values()),
    }


def stream_collision_audit_v2(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    audit = v1.stream_coupling_audit(rows, exclude_dir=V1_OUTPUT_DIR)
    audit["reused_unconsumed_reservation"] = {
        "excluded_exact_directory": str(_repo_abs(V1_OUTPUT_DIR).absolute()),
        "stream_manifest_file_sha256": V1_STREAM_FILE_SHA256,
        "stream_manifest_payload_sha256": V1_STREAM_PAYLOAD_SHA256,
        "streams_consumed": 0,
        "reason":
            "same immutable unused reservation; exclude exact v1 evidence only",
    }
    return audit


def staged_cost_decomposition(
    stream_rows: list[dict[str, Any]],
    *,
    final_cost: dict[str, Any],
) -> dict[str, Any]:
    stages = {
        "E0": {0, 1},
        "E1": {2, 3, 4, 5, 6, 7},
    }
    stage_rows = {
        stage: [row for row in stream_rows if int(row["replicate"]) in reps]
        for stage, reps in stages.items()
    }
    arm_replicates: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in stream_rows:
        arm_replicates[(str(row["record_id"]), str(row["action"]))].add(
            int(row["replicate"])
        )
    all_arms_complete = all(
        replicates == set(range(v1.REPLICATES))
        for replicates in arm_replicates.values()
    )
    by_partition = {
        stage: dict(
            sorted(Counter(row["partition"] for row in rows).items())
        )
        for stage, rows in stage_rows.items()
    }
    seconds_per_path = float(final_cost["conservative_seconds_per_path"])
    bytes_per_path = int(final_cost["selected_bytes_per_path"])
    multiplier = float(final_cost["storage_multiplier"])
    base_bytes = int(final_cost["base_bytes"])
    e0_bytes = math.ceil(
        multiplier * (base_bytes + len(stage_rows["E0"]) * bytes_per_path)
    )
    e1_bytes = math.ceil(
        multiplier * len(stage_rows["E1"]) * bytes_per_path
    )
    stage_costs = {
        "E0": {
            "replicates": [0, 1],
            "paths": len(stage_rows["E0"]),
            "paths_by_partition": by_partition["E0"],
            "projected_runtime_seconds":
                len(stage_rows["E0"]) * seconds_per_path,
            "projected_runtime_hours":
                len(stage_rows["E0"]) * seconds_per_path / 3600.0,
            "projected_incremental_bytes": e0_bytes,
            "projected_incremental_gib": e0_bytes / 1024**3,
            "single_base_allocation_included": True,
            "authorized": False,
            "fit_promotable": False,
        },
        "E1": {
            "replicates": [2, 3, 4, 5, 6, 7],
            "paths": len(stage_rows["E1"]),
            "paths_by_partition": by_partition["E1"],
            "projected_runtime_seconds":
                len(stage_rows["E1"]) * seconds_per_path,
            "projected_runtime_hours":
                len(stage_rows["E1"]) * seconds_per_path / 3600.0,
            "projected_incremental_bytes": e1_bytes,
            "projected_incremental_gib": e1_bytes / 1024**3,
            "single_base_allocation_included": False,
            "authorized": False,
            "requires_prospective_e0_gate": True,
        },
    }
    checks = {
        "all_arms_have_exact_replicates_0_through_7": all_arms_complete,
        "e0_path_count_exact":
            len(stage_rows["E0"]) == EXPECTED_E0_PATHS,
        "e1_path_count_exact":
            len(stage_rows["E1"]) == EXPECTED_E1_PATHS,
        "total_path_count_exact":
            len(stream_rows) == EXPECTED_TOTAL_PATHS,
        "partition_counts_exact":
            all(by_partition[stage] == expected
                for stage, expected in EXPECTED_STAGE_PATHS.items()),
        "stage_storage_sums_to_final":
            e0_bytes + e1_bytes
            == int(final_cost["projected_incremental_bytes"]),
        "stage_runtime_sums_to_final":
            math.isclose(
                stage_costs["E0"]["projected_runtime_seconds"]
                + stage_costs["E1"]["projected_runtime_seconds"],
                float(final_cost["projected_runtime_seconds"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
        "stages_unauthorized":
            not stage_costs["E0"]["authorized"]
            and not stage_costs["E1"]["authorized"],
    }
    return {
        "scientific_replicates_unchanged": v1.REPLICATES,
        "ordering":
            "E0 all roots/actions first; E1 all roots/actions or none",
        "stage_costs": stage_costs,
        "final_cost_projection": final_cost,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _disk_audit(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    return {
        "free_bytes": usage.free,
        "free_gib": free_gib,
        "minimum_free_gib": MIN_FREE_GIB,
        "target_free_gib": TARGET_FREE_GIB,
        "above_hard_minimum": free_gib >= MIN_FREE_GIB,
        "above_target": free_gib >= TARGET_FREE_GIB,
    }


def _test_evidence_audit() -> dict[str, Any]:
    if not TEST_EVIDENCE_PATH.is_file():
        return {
            "path": str(TEST_EVIDENCE_PATH),
            "passes": False,
            "reason": "missing",
        }
    payload = _json(TEST_EVIDENCE_PATH)
    rows = []
    for row in payload.get("bound_files", []):
        path = Path(str(row["path"]))
        actual = sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expected_sha256": row["sha256"],
                "actual_sha256": actual,
                "matches": actual == row["sha256"],
            }
        )
    checks = {
        "evidence_passes": payload.get("passes") is True,
        "version_exact":
            payload.get("version") == "g3_v2_preflight_test_evidence_v1",
        "bound_files_present": bool(rows),
        "bound_files_exact": bool(rows) and all(row["matches"] for row in rows),
    }
    return {
        "path": str(TEST_EVIDENCE_PATH),
        "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "rows": rows,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _input_hash_audit() -> dict[str, Any]:
    inherited = v1._input_hash_audit()
    rows, v1_artifacts_pass = _locked_file_rows(V1_ARTIFACT_LOCKS)
    checks = {
        "inherited_g2_v1_inputs_exact": inherited["passes"],
        "v1_and_v2_amendment_artifacts_exact": v1_artifacts_pass,
    }
    return {
        "inherited": inherited,
        "additional_rows": rows,
        "checks": checks,
        "passes": all(checks.values()),
    }


def build_preflight_payload() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    input_hashes = _input_hash_audit()
    preflight_v1, record_manifest, stream_manifest, reuse_audit = (
        load_v1_manifests()
    )
    records = record_manifest["records"]
    stream_rows = stream_manifest["rows"]
    transfer_records = [
        row for row in records
        if row["partition"] == "transfer_diagnostic"
    ]
    ordinary_records = [
        row for row in records
        if row["partition"] != "transfer_diagnostic"
    ]

    transfer_result = _json(v1.TRANSFER_RESULT_PATH)
    transfer_sources = v1._transfer_sources(transfer_result)
    panel_bindings = bind_transfer_panel_inputs(transfer_sources)
    untouched = corrected_transfer_untouched_audit(transfer_records)
    stream_audit = stream_collision_audit_v2(stream_rows)

    coverage = preflight_v1["label_coverage"]
    final_cost = v1.cost_projection(int(coverage["missing_h40_paths"]))
    staged_cost = staged_cost_decomposition(
        stream_rows, final_cost=final_cost
    )
    power = v1.power_audit_n32()
    power_matches_v1 = canonical_sha256(power) == canonical_sha256(
        preflight_v1["n32_power_mde"]
    )
    cost_matches_v1 = canonical_sha256(final_cost) == canonical_sha256(
        preflight_v1["cost_projection"]
    )

    disk = _disk_audit(WORKSPACE_ROOT)
    services = g1r.service_health()
    heavy = _heavy_process_audit()
    current_nice = os.nice(0)
    tests = _test_evidence_audit()

    integrity_checks = {
        "immutable_input_hashes": input_hashes["passes"],
        "v1_manifests_exact": reuse_audit["passes"],
        "ordinary_count_exact":
            len(ordinary_records) == EXPECTED_ORDINARY_RECORDS,
        "transfer_count_exact":
            len(transfer_records) == EXPECTED_TRANSFER_RECORDS,
        "panel_input_bindings_exact": panel_bindings["passes"],
        "corrected_external_matches_zero": untouched["passes"],
        "stream_coupling_and_collisions_pass": stream_audit["passes"],
        "staged_cost_contract_exact": staged_cost["passes"],
        "cost_recomputes_exactly": cost_matches_v1,
        "power_recomputes_exactly": power_matches_v1,
        "focused_and_regression_tests_pass": tests["passes"],
    }
    readiness_checks = {
        "compatible_paths_still_zero":
            coverage["compatible_existing_h40_paths"] == 0,
        "missing_paths_exact":
            coverage["missing_h40_paths"] == EXPECTED_TOTAL_PATHS,
        "projected_runtime_within_72h": final_cost["runtime_passes"],
        "projected_incremental_storage_below_4gib":
            final_cost["storage_passes"],
        "disk_above_100gib": disk["above_hard_minimum"],
        "services_dashboard_top_three_pass": services["passes"],
        "no_competing_heavy_process": heavy["passes"],
        "nice_at_least_10": current_nice >= MIN_NICE,
        "n32_mde_computed": power["mde_grid_or"] is not None,
    }
    if not all(integrity_checks.values()):
        decision = "KILL_G3_V2_PREFLIGHT_INTEGRITY"
    elif all(readiness_checks.values()):
        decision = "READY_G3_V2_BOOTSTRAP_LABELS"
    else:
        decision = "HOLD_G3_V2_LABEL_COVERAGE_OR_COST"

    untouched_payload = dict(untouched)
    untouched_payload["canonical_payload_sha256"] = canonical_sha256(
        untouched_payload
    )
    panel_payload = {
        "version": "g3_v2_panel_input_bindings_v1",
        "v1_record_manifest_file_sha256": V1_RECORD_FILE_SHA256,
        "v1_record_manifest_payload_sha256": V1_RECORD_PAYLOAD_SHA256,
        **panel_bindings,
        "outcomes_opened": False,
    }
    panel_payload["canonical_payload_sha256"] = canonical_sha256(panel_payload)

    payload = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "terminal_status": "HOLD_G3_AFTER_V2_BOOTSTRAP_PREFLIGHT_SEAL",
        "v1_authoritative_decision_preserved":
            "KILL_G3_PREFLIGHT_INTEGRITY",
        "charter": {
            "base_path": str(v1.CHARTER_PATH),
            "base_sha256": v1.CHARTER_SHA256,
            "a1_path": str(v1.AMENDMENT_PATH),
            "a1_sha256": v1.AMENDMENT_SHA256,
            "v2_integrity_amendment_path": str(AMENDMENT_PATH),
            "v2_integrity_amendment_sha256": AMENDMENT_SHA256,
        },
        "implementation": {
            "path": str(IMPLEMENTATION_PATH),
            "sha256": sha256_path(IMPLEMENTATION_PATH),
            "test_path": str(TEST_PATH),
            "test_sha256": sha256_path(TEST_PATH),
            "test_evidence": tests,
        },
        "input_hash_audit": input_hashes,
        "v1_manifest_reuse_audit": reuse_audit,
        "record_counts": {
            "ordinary": len(ordinary_records),
            "transfer_diagnostic": len(transfer_records),
            "total": len(records),
        },
        "corrected_transfer_untouched_audit_payload_sha256":
            untouched_payload["canonical_payload_sha256"],
        "panel_input_binding_payload_sha256":
            panel_payload["canonical_payload_sha256"],
        "stream_audit": stream_audit,
        "label_coverage_reused_from_exact_v1": coverage,
        "staged_cost_decomposition": staged_cost,
        "n32_power_mde": power,
        "cost_matches_v1": cost_matches_v1,
        "power_matches_v1": power_matches_v1,
        "integrity_checks": integrity_checks,
        "readiness_checks": readiness_checks,
        "disk": disk,
        "services": services,
        "heavy_process_audit": heavy,
        "current_nice": current_nice,
        "zero_forbidden_work": {
            "new_games": 0,
            "streams_consumed": 0,
            "new_labels": 0,
            "label_values_opened": False,
            "models_fit": 0,
            "transfer_outcomes_opened": 0,
            "candidate_actions": 0,
            "rerankers_built": 0,
            "policy_outcomes": 0,
            "continuations": 0,
            "score_inspection": False,
            "incumbent_changed": False,
            "dashboard_changed": False,
        },
        "e0_authorized": False,
        "e1_authorized": False,
        "fitting_authorized": False,
        "policy_evaluation_authorized": False,
        "promotion_authorized": False,
        "dashboard_eligible": False,
    }
    payload["canonical_payload_sha256"] = canonical_sha256(payload)
    return payload, untouched_payload, panel_payload


def run_preflight() -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_DIR}")
    if STAGING_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {STAGING_DIR}")
    STAGING_DIR.mkdir(parents=True, exist_ok=False)
    try:
        payload, untouched, panel = build_preflight_payload()
        _write_immutable_json(
            STAGING_DIR / "G3_V2_CORRECTED_UNTOUCHEDNESS_AUDIT.json",
            untouched,
        )
        _write_immutable_json(
            STAGING_DIR / "G3_V2_PANEL_INPUT_BINDINGS.json",
            panel,
        )
        payload["corrected_transfer_untouched_audit_file_sha256"] = (
            sha256_path(
                STAGING_DIR / "G3_V2_CORRECTED_UNTOUCHEDNESS_AUDIT.json"
            )
        )
        payload["panel_input_binding_file_sha256"] = sha256_path(
            STAGING_DIR / "G3_V2_PANEL_INPUT_BINDINGS.json"
        )
        payload["canonical_payload_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "canonical_payload_sha256"
            }
        )
        _write_immutable_json(
            STAGING_DIR / "G3_V2_BOOTSTRAP_PREFLIGHT.json",
            payload,
        )
        STAGING_DIR.replace(OUTPUT_DIR)
        return payload
    except Exception as error:
        failure = {
            "version": VERSION,
            "decision": "KILL_G3_V2_PREFLIGHT_INTEGRITY",
            "error": f"{type(error).__name__}: {error}",
            "zero_forbidden_work": {
                "new_games": 0,
                "streams_consumed": 0,
                "new_labels": 0,
                "label_values_opened": False,
                "models_fit": 0,
                "transfer_outcomes_opened": 0,
                "policy_outcomes": 0,
                "score_inspection": False,
            },
        }
        failure["canonical_payload_sha256"] = canonical_sha256(failure)
        _write_immutable_json(STAGING_DIR / "PREFLIGHT_FAILURE.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if os.nice(0) < MIN_NICE:
        os.nice(MIN_NICE - os.nice(0))
    payload = run_preflight()
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "canonical_payload_sha256":
                    payload["canonical_payload_sha256"],
                "out_dir": str(OUTPUT_DIR),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

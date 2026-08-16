"""One-shot, outcome-free O4 domain and fresh-to-science root preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from threes_rl import g1r_acquire as history
from threes_rl import g1r_acquire_v2_qd5 as qd5
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.o3_designated_pair_option import initial_lineage
from threes_rl.o4_domain_safe_pair_option import (
    O4DesignatedPairNet,
    exhaustive_blocker_domain_proof,
    option_features,
    parameter_count,
    root_option_eligible,
    schema_sha256,
    select_designated_pair,
)
from threes_rl.o4_power_contract import power_table
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import SimState, ThreesSim, preview_from_label


VERSION = "o4_domain_safe_p0_v1"
STARTER_TILE = 1536
ROOT = Path(".")
CHARTER_PATH = Path("threes_rl/O4_DOMAIN_SAFE_DESIGNATED_PAIR_CHARTER.md")
OPTION_PATH = Path("threes_rl/o4_domain_safe_pair_option.py")
POWER_PATH = Path("threes_rl/o4_power_contract.py")
RUNNER_PATH = Path("threes_rl/o4_p0_preflight.py")
OPTION_TEST_PATH = Path("tests/test_rl_o4_domain_safe_pair_option.py")
TEST_PATH = Path("tests/test_rl_o4_p0_preflight.py")
OUTPUT_DIR = Path("threes_rl/runs/forensics/o4_domain_safe_p0_v1")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/o4_domain_safe_p0_test_evidence.json"
)
MARKER_NAME = "O4_P0_OPENED.json"
RESULT_NAME = "O4_P0_RESULT.json"
SOURCE_NAME = "O4_P0_SOURCE_POOL.json"
SOURCE_REPLAY_NAME = "O4_P0_SOURCE_REPLAY_MANIFEST.json"
SELECTION_NAME = "O4_P0_SELECTED_ROOTS.json"
STREAM_NAME = "O4_P0_STREAM_MANIFEST.json"
COLLISION_NAME = "O4_P0_COLLISION_AUDIT.json"
POWER_NAME = "O4_P0_POWER_TABLE.json"
POLICY_NAME = "O4_P0_POLICY_AUDIT.json"
DOMAIN_NAME = "O4_P0_DOMAIN_PROOF.json"

O3_ACQUISITION_DIR = Path(
    "threes_rl/runs/forensics/o3_event_acquisition_v1"
)
O3_RECOVERY_DIR = Path(
    "threes_rl/runs/forensics/o3_event_acquisition_recovery_v1"
)
O3_OPTION_TRAINING_DIR = Path(
    "threes_rl/runs/forensics/o3_option_training_v1"
)
O3_UNION_PATH = O3_RECOVERY_DIR / "O3_RECOVERY_UNION_MANIFEST.json"
O3_SUPPORT_PATH = O3_RECOVERY_DIR / "O3_RECOVERY_SUPPORT_SCAN.json"
O3_SELECTED_PATH = O3_RECOVERY_DIR / "O3_RECOVERY_SELECTED_ROOTS.json"
O3_RECOVERY_RESULT_PATH = O3_RECOVERY_DIR / "O3_RECOVERY_RESULT.json"
O3_RESEAL_V3_PATH = Path(
    "threes_rl/runs/forensics/o3_selected_integrity_reseal_v3/"
    "O3_SELECTED_INTEGRITY_RESEAL_V3.json"
)
O3_POLICY_AUDIT_PATH = Path(
    "threes_rl/runs/forensics/o3_event_option_p0_v1/"
    "O3_P0_POLICY_AUDIT.json"
)
O3_STREAM_MANIFEST_PATH = Path(
    "threes_rl/runs/forensics/o3_event_option_p0_v1/"
    "O3_P0_STREAM_MANIFEST.json"
)

IMMUTABLE_INPUT_HASHES = {
    str(O3_UNION_PATH): (
        "02ea2c5be8823de775f56b7267f9c8371d26efc53897115b25733f8ef4527311"
    ),
    str(O3_SUPPORT_PATH): (
        "4c71513e6a3a2778bb8d1db0ba08f8ff5a1f0d6edc82ee1208b7458593059d27"
    ),
    str(O3_SELECTED_PATH): (
        "9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049"
    ),
    str(O3_RECOVERY_RESULT_PATH): (
        "962da52b83b8746c006a9ef5fbe1fdd34f43e9c7bf97d9b6ff48f2a42019c23a"
    ),
    str(O3_RESEAL_V3_PATH): (
        "5bb80bc02597ea934c02f8ebd07eaf0158623232f88ea0408532cdc0039e6696"
    ),
    str(O3_POLICY_AUDIT_PATH): (
        "2b498ce5bc22f54f6286e114f3212758e911a1ac7a651da2c3095db42dea0e60"
    ),
    str(O3_STREAM_MANIFEST_PATH): (
        "94e7b0dfe83e568b4e9686dd3ee44cc70739c0312349fe36a05bb6df80c77225"
    ),
}
IMMUTABLE_PAYLOAD_HASHES = {
    str(O3_UNION_PATH): (
        "cec88701a1754f1064d639dae09cd6856ee18ce9399865338ebed7107f672d94"
    ),
    str(O3_SUPPORT_PATH): (
        "27ae3a6aca5f1de71ee18df193c0663a83579d3aeba65cd864065cfff594e25a"
    ),
    str(O3_RECOVERY_RESULT_PATH): (
        "a679d512d6ce44bf5fd4ecd8249d15625c59f342e64796a6d5eb894396224ad0"
    ),
    str(O3_RESEAL_V3_PATH): (
        "622ebf6361527be7283fd51c7a7acff99aa8125b06c76dbc4ee8a801faf3904d"
    ),
    str(O3_POLICY_AUDIT_PATH): (
        "6c09df4c8e0e0d3720720e05d58cda8459dea9296d050b751db6b115705deb9c"
    ),
    str(O3_STREAM_MANIFEST_PATH): (
        "27e3200e88d31d4f38a921965b631f264aa43f0ef02cb380f41b0c04d8455d1b"
    ),
}
SELECTED_PRE_SERIALIZATION_SHA256 = (
    "c6c8b1a35cc63f4c1c1fdc98579f1ae0859a84c5eef7203306000223ac9c61a5"
)
SELECTED_POST_JSON_SHA256 = (
    "d9600cf420d947826c812b88225633b78a889f94f94ce39270dd71bc11b12f0e"
)

DEPENDENCY_PATHS = (
    Path("threes_rl/o1_geometry_option.py"),
    Path("threes_rl/o3_designated_pair_option.py"),
    Path("threes_rl/restart_manifest.py"),
    Path("threes_rl/train_td.py"),
    Path("threes_rl/sim.py"),
    Path("threes_rl/replay_provenance.py"),
    Path("threes_rl/g1r_acquire.py"),
    Path("threes_rl/g1r_acquire_v2_qd5.py"),
    Path("threes_rl/g1r_qd_admission_v2.py"),
)

O3_FAMILY_ORDER = (
    "o3_corner2",
    "o3_expectimax2",
    "o3_parent_mc1000",
    "o3_replaycal",
    "o3_qd_v2",
)
FAMILY_ORDER = (
    "o4_corner2",
    "o4_expectimax2",
    "o4_parent_mc1000",
    "o4_replaycal",
    "o4_qd_v2",
)
O3_TO_O4 = dict(zip(O3_FAMILY_ORDER, FAMILY_ORDER, strict=True))
ROLE_ORDER = ("train", "development", "untouched_mechanism")
TARGET_ORDER = (48, 96, 192)
ROLE_FAMILY_TARGET_COUNTS = {
    "train": (
        (13, 13, 13),
        (13, 13, 13),
        (12, 13, 13),
        (13, 12, 13),
        (13, 13, 12),
    ),
    "development": (
        (4, 4, 5),
        (4, 5, 4),
        (5, 4, 4),
        (5, 4, 4),
        (4, 4, 4),
    ),
    "untouched_mechanism": (
        (12, 13, 13),
        (13, 12, 13),
        (13, 13, 13),
        (13, 13, 12),
        (13, 13, 13),
    ),
}
ROLE_FAMILY_COUNTS = {
    "train": (39, 39, 38, 38, 38),
    "development": (13, 13, 13, 13, 12),
    "untouched_mechanism": (38, 38, 39, 38, 39),
}
ROLE_TARGET_COUNTS = {
    "train": (64, 64, 64),
    "development": (22, 21, 21),
    "untouched_mechanism": (64, 64, 64),
}
ROLE_COUNTS = {
    "train": 192,
    "development": 64,
    "untouched_mechanism": 192,
}
TOTAL_SELECTED_ROOTS = 448
O3_SELECTED_ROOTS = 320
O3_ACQUISITION_ROOTS = 20_500
O3_UNSELECTED_ROOTS = O3_ACQUISITION_ROOTS - O3_SELECTED_ROOTS
FROZEN_PARAMETER_COUNT = 102_557

STREAM_BASES = {
    "learning": {
        "logical_seed": 129_000_000_000,
        "deck_stream_id": 130_000_000_000,
        "slot_stream_id": 131_000_000_000,
        "policy_stream_id": 132_000_000_000,
    },
    "option": {
        "logical_seed": 133_000_000_000,
        "deck_stream_id": 134_000_000_000,
        "slot_stream_id": 135_000_000_000,
        "policy_stream_id": 136_000_000_000,
    },
    "normal_development": {
        "logical_seed": 137_000_000_000,
        "deck_stream_id": 138_000_000_000,
        "slot_stream_id": 139_000_000_000,
        "policy_stream_id": 140_000_000_000,
    },
    "confirmation": {
        "logical_seed": 141_000_000_000,
        "deck_stream_id": 142_000_000_000,
        "slot_stream_id": 143_000_000_000,
        "policy_stream_id": 144_000_000_000,
    },
}
STREAM_FIELDS = (
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
)
TRAJECTORIES_PER_TRAIN_ROOT = 6
OPTION_REPEATS = 8
NORMAL_DEVELOPMENT_ROOTS = 512
CONFIRMATION_ROOTS = 2_560
MINIMUM_NICE = 10
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
PROJECTED_ACTIVE_SECONDS = 18 * 3_600
PROJECTED_STORAGE_BYTES = int(2.75 * 1024**3)
STORAGE_CAP_BYTES = 4 * 1024**3


class SourceIntegrityError(RuntimeError):
    """An immutable source or representation identity failed."""


class OperationalHold(RuntimeError):
    """An operational gate failed without scientific interpretation."""


def canonical_json_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _write_immutable_json(
    path: Path,
    payload: dict[str, Any],
    *,
    self_hash_field: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    body = dict(payload)
    body[self_hash_field] = canonical_json_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic temporary already exists: {temporary}")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return body


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    embedded = body.pop(field, None)
    return isinstance(embedded, str) and embedded == canonical_json_hash(body)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _dependency_hashes() -> dict[str, str]:
    return {str(path): sha256_path(path) for path in DEPENDENCY_PATHS}


def _current_bindings() -> dict[str, Any]:
    return {
        "version": VERSION,
        "charter_sha256": sha256_path(CHARTER_PATH),
        "option_implementation_sha256": sha256_path(OPTION_PATH),
        "power_implementation_sha256": sha256_path(POWER_PATH),
        "p0_implementation_sha256": sha256_path(RUNNER_PATH),
        "option_tests_sha256": sha256_path(OPTION_TEST_PATH),
        "p0_tests_sha256": sha256_path(TEST_PATH),
        "schema_sha256": schema_sha256(),
        "parameter_count": parameter_count(),
        "dependency_hashes": _dependency_hashes(),
        "immutable_input_file_hashes": {
            path: sha256_path(Path(path))
            for path in IMMUTABLE_INPUT_HASHES
        },
        "family_order": list(FAMILY_ORDER),
        "role_family_target_counts": ROLE_FAMILY_TARGET_COUNTS,
        "stream_bases": STREAM_BASES,
    }


def validate_frozen_matrices() -> dict[str, Any]:
    family_totals = [0] * len(FAMILY_ORDER)
    target_totals = [0] * len(TARGET_ORDER)
    role_checks = {}
    for role in ROLE_ORDER:
        matrix = ROLE_FAMILY_TARGET_COUNTS[role]
        row_totals = tuple(sum(row) for row in matrix)
        column_totals = tuple(
            sum(matrix[row][column] for row in range(len(FAMILY_ORDER)))
            for column in range(len(TARGET_ORDER))
        )
        for family_index, value in enumerate(row_totals):
            family_totals[family_index] += value
        for target_index, value in enumerate(column_totals):
            target_totals[target_index] += value
        role_checks[role] = {
            "matrix_shape_exact": len(matrix) == len(FAMILY_ORDER)
            and all(len(row) == len(TARGET_ORDER) for row in matrix),
            "family_marginals_exact": row_totals == ROLE_FAMILY_COUNTS[role],
            "target_marginals_exact": column_totals == ROLE_TARGET_COUNTS[role],
            "role_count_exact": sum(row_totals) == ROLE_COUNTS[role],
        }
    checks = {
        "all_role_matrices_exact": all(
            all(values.values()) for values in role_checks.values()
        ),
        "total_roots_448": sum(ROLE_COUNTS.values()) == TOTAL_SELECTED_ROOTS,
        "combined_family_marginals": tuple(family_totals)
        == (90, 90, 90, 89, 89),
        "combined_target_marginals": tuple(target_totals) == (150, 149, 149),
    }
    return {
        "role_checks": role_checks,
        "combined_family_counts": dict(zip(FAMILY_ORDER, family_totals, strict=True)),
        "combined_target_counts": dict(zip(TARGET_ORDER, target_totals, strict=True)),
        "checks": checks,
        "passes": all(checks.values()),
    }


def immutable_input_audit() -> dict[str, Any]:
    files = {}
    for raw_path, expected in IMMUTABLE_INPUT_HASHES.items():
        path = Path(raw_path)
        actual = sha256_path(path)
        files[raw_path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
            "bytes": int(path.stat().st_size),
        }
    payloads = {}
    for raw_path, expected in IMMUTABLE_PAYLOAD_HASHES.items():
        payload = json.loads(Path(raw_path).read_text())
        field = next(
            (
                key
                for key in (
                    "union_payload_sha256",
                    "support_payload_sha256",
                    "result_payload_sha256",
                    "v3_reseal_payload_sha256",
                    "payload_sha256",
                )
                if key in payload
            ),
            None,
        )
        actual = payload.get(field) if field else None
        payloads[raw_path] = {
            "field": field,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
            "self_hash_valid": bool(field and _verify_self_hash(payload, field)),
        }
    checks = {
        "all_file_hashes_exact": all(row["matches"] for row in files.values()),
        "all_payload_hashes_exact": all(
            row["matches"] and row["self_hash_valid"]
            for row in payloads.values()
        ),
        "selected_original_file_exact": files[str(O3_SELECTED_PATH)]["matches"],
        "selected_v3_proof_exact": (
            payloads[str(O3_RESEAL_V3_PATH)]["matches"]
            and payloads[str(O3_RESEAL_V3_PATH)]["self_hash_valid"]
        ),
    }
    return {
        "files": files,
        "payloads": payloads,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _required_family_target_counts() -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for family_index, family in enumerate(FAMILY_ORDER):
        for target_index, target in enumerate(TARGET_ORDER):
            result[(family, target)] = sum(
                ROLE_FAMILY_TARGET_COUNTS[role][family_index][target_index]
                for role in ROLE_ORDER
            )
    return result


def source_pool_from_payloads(
    union: Mapping[str, Any],
    support: Mapping[str, Any],
    selected: Mapping[str, Any],
    reseal: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    membership = union.get("membership")
    candidate_rows = support.get("candidate_rows")
    selected_rows = selected.get("selected")
    if not isinstance(membership, list) or not isinstance(candidate_rows, list):
        raise SourceIntegrityError("O3 source manifests have invalid row containers")
    if not isinstance(selected_rows, list):
        raise SourceIntegrityError("O3 selected manifest has invalid rows")

    union_index: dict[str, dict[str, Any]] = {}
    for raw in membership:
        if not isinstance(raw, dict):
            raise SourceIntegrityError("O3 union contains a non-object row")
        root = str(raw.get("root_cluster"))
        if root in union_index:
            raise SourceIntegrityError(f"Duplicate O3 union root: {root}")
        union_index[root] = dict(raw)
    selected_roots = {str(row.get("root_cluster")) for row in selected_rows}
    if len(selected_roots) != len(selected_rows):
        raise SourceIntegrityError("O3 selected roots are not unique")

    reseal_checks = {
        "decision_ready": reseal.get("decision")
        == "READY_O3_OPTION_TRAINING_INTEGRITY_RESEALED_V3",
        "pre_serialization_sha_exact": reseal.get(
            "selected_pre_serialization_reproduction_sha256"
        )
        == SELECTED_PRE_SERIALIZATION_SHA256,
        "post_json_sha_exact": reseal.get(
            "selected_post_json_scientific_payload_sha256"
        )
        == SELECTED_POST_JSON_SHA256,
    }

    compact_candidates = []
    candidate_keys: set[tuple[str, int]] = set()
    source_identity_failures = []
    for raw in candidate_rows:
        if not isinstance(raw, dict):
            raise SourceIntegrityError("O3 support contains a non-object row")
        root = str(raw.get("root_cluster"))
        if root in selected_roots:
            continue
        union_row = union_index.get(root)
        if union_row is None:
            source_identity_failures.append(f"missing_union:{root}")
            continue
        family = str(raw.get("family"))
        mapped_family = O3_TO_O4.get(family)
        target = int(raw.get("target", -1))
        key = (root, target)
        if mapped_family is None or target not in TARGET_ORDER:
            source_identity_failures.append(f"bad_family_target:{root}")
            continue
        if key in candidate_keys:
            source_identity_failures.append(f"duplicate_root_target:{root}:T{target}")
            continue
        candidate_keys.add(key)
        checks = (
            str(union_row.get("family")) == family,
            str(union_row.get("source_replay")) == str(raw.get("source_replay")),
            str(union_row.get("source_replay_sha256"))
            == str(raw.get("source_replay_sha256")),
        )
        if not all(checks):
            source_identity_failures.append(f"union_support_mismatch:{root}:T{target}")
            continue
        compact_candidates.append(
            {
                "root_cluster": root,
                "family": mapped_family,
                "source_family": family,
                "target": target,
                "frame_index": int(raw.get("frame_index")),
                "state_sha1": str(raw.get("state_sha1")),
                "source_replay": str(raw.get("source_replay")),
                "source_replay_sha256": str(raw.get("source_replay_sha256")),
                "deck_stream_id": int(union_row.get("deck_stream_id")),
                "slot_stream_id": int(union_row.get("slot_stream_id")),
            }
        )

    by_family_target = {
        (family, target): {
            row["root_cluster"]
            for row in compact_candidates
            if row["family"] == family and row["target"] == target
        }
        for family in FAMILY_ORDER
        for target in TARGET_ORDER
    }
    by_family = {
        family: {
            row["root_cluster"]
            for row in compact_candidates
            if row["family"] == family
        }
        for family in FAMILY_ORDER
    }
    by_target = {
        target: {
            row["root_cluster"]
            for row in compact_candidates
            if row["target"] == target
        }
        for target in TARGET_ORDER
    }
    required_family_target = _required_family_target_counts()
    required_family = validate_frozen_matrices()["combined_family_counts"]
    required_target = validate_frozen_matrices()["combined_target_counts"]
    family_target_upper_bound = {
        f"{family}/T{target}": len(by_family_target[(family, target)])
        for family in FAMILY_ORDER
        for target in TARGET_ORDER
    }
    family_upper_bound = {
        family: len(by_family[family]) for family in FAMILY_ORDER
    }
    target_upper_bound = {
        f"T{target}": len(by_target[target]) for target in TARGET_ORDER
    }
    upper_bound_checks = {
        "family_target_cells_can_meet_combined_need": all(
            len(by_family_target[key]) >= required
            for key, required in required_family_target.items()
        ),
        "families_can_meet_combined_need": all(
            family_upper_bound[family] >= required_family[family]
            for family in FAMILY_ORDER
        ),
        "targets_can_meet_combined_need": all(
            len(by_target[target]) >= required_target[target]
            for target in TARGET_ORDER
        ),
    }
    checks = {
        "union_passes": union.get("passes") is True,
        "union_exact_20500": len(union_index) == O3_ACQUISITION_ROOTS,
        "selected_exact_320": len(selected_roots) == O3_SELECTED_ROOTS,
        "selected_subset_union": selected_roots.issubset(union_index),
        "unselected_exact_20180": len(set(union_index) - selected_roots)
        == O3_UNSELECTED_ROOTS,
        "support_audit_passes": support.get("audit", {}).get("passes") is True,
        "support_candidate_manifest_exact": support.get(
            "candidate_manifest_sha256"
        )
        == canonical_json_hash(candidate_rows),
        "selected_scientific_checks_pass": selected.get("passes") is True
        and not selected.get("deficits"),
        "v3_reseal_exact": all(reseal_checks.values()),
        "zero_source_identity_failures": not source_identity_failures,
        "all_selected_roots_excluded": not any(
            row["root_cluster"] in selected_roots for row in compact_candidates
        ),
    }
    report = {
        "union_roots": len(union_index),
        "selected_roots_excluded": len(selected_roots),
        "unselected_root_universe": len(set(union_index) - selected_roots),
        "source_candidate_rows_after_exclusion": len(compact_candidates),
        "source_candidate_roots_after_exclusion": len(
            {row["root_cluster"] for row in compact_candidates}
        ),
        "candidate_identity_sha256": canonical_json_hash(compact_candidates),
        "family_target_root_upper_bounds": family_target_upper_bound,
        "family_root_upper_bounds": family_upper_bound,
        "target_root_upper_bounds": target_upper_bound,
        "required_family_target_counts": {
            f"{family}/T{target}": value
            for (family, target), value in required_family_target.items()
        },
        "required_family_counts": required_family,
        "required_target_counts": {
            f"T{target}": value for target, value in required_target.items()
        },
        "upper_bound_checks": upper_bound_checks,
        "upper_bound_feasible": all(upper_bound_checks.values()),
        "source_identity_failures": source_identity_failures,
        "reseal_checks": reseal_checks,
        "checks": checks,
        "passes": all(checks.values()),
        "o3_option_training_bodies_read": False,
        "o3_selected_replay_bodies_read": False,
        "final_score_action_outcome_fields_read": False,
    }
    return report, compact_candidates, union_index


def load_source_pool() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    audit = immutable_input_audit()
    if not audit["passes"]:
        raise SourceIntegrityError("Immutable O3 source identities changed")
    union = json.loads(O3_UNION_PATH.read_text())
    support = json.loads(O3_SUPPORT_PATH.read_text())
    selected = json.loads(O3_SELECTED_PATH.read_text())
    reseal = json.loads(O3_RESEAL_V3_PATH.read_text())
    return source_pool_from_payloads(union, support, selected, reseal)


def verify_candidate_source_replays(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_by_root: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        root = str(row["root_cluster"])
        prior = rows_by_root.get(root)
        if prior is not None and (
            prior["source_replay"] != row["source_replay"]
            or prior["source_replay_sha256"] != row["source_replay_sha256"]
        ):
            raise SourceIntegrityError(f"Candidate source drift within root: {root}")
        rows_by_root[root] = row
    manifest = []
    failures = []
    allowed_dirs = (
        O3_ACQUISITION_DIR / "source_replays",
        O3_RECOVERY_DIR / "source_replays",
    )
    for root, row in sorted(rows_by_root.items()):
        path = Path(str(row["source_replay"]))
        allowed = any(_is_within(path, directory) for directory in allowed_dirs)
        if not allowed or _is_within(path, O3_OPTION_TRAINING_DIR):
            failures.append(f"forbidden_source_path:{root}:{path}")
            continue
        if path.is_symlink() or not path.is_file():
            failures.append(f"missing_or_symlink_source:{root}:{path}")
            continue
        actual = sha256_path(path)
        expected = str(row["source_replay_sha256"])
        if actual != expected:
            failures.append(f"source_hash_mismatch:{root}:{path}")
            continue
        manifest.append(
            {
                "root_cluster": root,
                "path": str(path),
                "sha256": actual,
                "bytes": int(path.stat().st_size),
            }
        )
    checks = {
        "all_candidate_sources_verified": len(manifest) == len(rows_by_root),
        "zero_source_failures": not failures,
        "only_acquisition_recovery_replays": all(
            any(
                _is_within(Path(row["path"]), directory)
                for directory in allowed_dirs
            )
            for row in manifest
        ),
        "option_training_namespace_unread": True,
    }
    return {
        "source_count": len(manifest),
        "total_bytes_hashed": sum(row["bytes"] for row in manifest),
        "rows": manifest,
        "manifest_sha256": canonical_json_hash(manifest),
        "failures": failures,
        "checks": checks,
        "passes": all(checks.values()),
        "replay_bodies_parsed": False,
    }


def _find_frame(replay: Mapping[str, Any], frame_index: int) -> Mapping[str, Any]:
    frames = replay.get("frames")
    if not isinstance(frames, list):
        raise SourceIntegrityError("Permitted source replay has no frame list")
    matches = [
        frame
        for fallback, frame in enumerate(frames)
        if isinstance(frame, dict)
        and int(frame.get("index", fallback)) == int(frame_index)
    ]
    if len(matches) != 1:
        raise SourceIntegrityError(
            f"Expected one support frame {frame_index}, found {len(matches)}"
        )
    return matches[0]


def whitelisted_state_payload(
    payload: Mapping[str, Any],
) -> tuple[SimState, dict[str, Any]]:
    board = np.asarray(payload["board"], dtype=np.int32)
    if board.shape != (4, 4):
        raise SourceIntegrityError(f"O4 source board shape changed: {board.shape}")
    preview_payload = payload["preview"]
    if not isinstance(preview_payload, Mapping):
        raise SourceIntegrityError("O4 source preview is not an object")
    preview_kind = str(preview_payload["kind"])
    if preview_kind == "bonus":
        candidates = tuple(
            int(value) for value in preview_payload.get("candidates", ())
        )
        preview = preview_from_label("large_candidates", candidates)
    else:
        candidates = ()
        preview = preview_from_label(preview_kind)
    cycle = payload["tile_cycle"]
    if not isinstance(cycle, Mapping):
        raise SourceIntegrityError("O4 source tile_cycle is not an object")
    raw_counts = cycle["small_counts"]
    if not isinstance(raw_counts, Mapping):
        raise SourceIntegrityError("O4 source small_counts is not an object")
    small_counts = {
        name: int(raw_counts[name]) for name in ("red", "blue", "gray")
    }
    state = SimState(
        board=board.copy(),
        preview=preview,
        small_counts=small_counts,
        small_pos=int(cycle["small_pos"]),
        small_seen_total=int(cycle["small_seen_total"]),
        span_small_pos=int(cycle["span_small_pos"]),
        large_pending=bool(cycle["large_pending"]),
        max_tile=int(np.max(board)),
        move_count=int(payload["move_count"]),
        game_over=bool(payload["game_over"]),
    )
    identity_payload = {
        "board": board.tolist(),
        "preview": {
            "kind": preview_kind,
            "candidates": list(candidates),
        },
        "tile_cycle": {
            "small_counts": small_counts,
            "small_pos": state.small_pos,
            "small_seen_total": state.small_seen_total,
            "span_small_pos": state.span_small_pos,
            "large_pending": state.large_pending,
        },
        "move_count": state.move_count,
        "game_over": state.game_over,
    }
    return state, identity_payload


def restore_o4_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_root[str(row["root_cluster"])].append(row)
    restored = []
    failures = []
    action_feature_rows = 0
    for root, rows in sorted(by_root.items()):
        first = rows[0]
        path = Path(str(first["source_replay"]))
        if sha256_path(path) != first["source_replay_sha256"]:
            raise SourceIntegrityError(f"O4 source changed before restore: {path}")
        replay = json.loads(path.read_text())
        for row in sorted(rows, key=lambda item: int(item["target"])):
            try:
                frame = _find_frame(replay, int(row["frame_index"]))
                payload = frame.get("state")
                if not isinstance(payload, Mapping):
                    raise SourceIntegrityError("Support frame state is missing")
                state, identity_payload = whitelisted_state_payload(payload)
                o4_state_sha256 = canonical_json_hash(identity_payload)
                simulator = ThreesSim.from_stream_ids(
                    deck_stream_id=int(row["deck_stream_id"]),
                    slot_stream_id=int(row["slot_stream_id"]),
                    starter_tile=STARTER_TILE,
                )
                target = int(row["target"])
                pair = select_designated_pair(
                    state.board,
                    STARTER_TILE,
                    requested_target=target,
                    allowed_targets=TARGET_ORDER,
                )
                if pair is None or pair.safe_merge_actions:
                    continue
                if not root_option_eligible(
                    state,
                    simulator,
                    STARTER_TILE,
                    allowed_targets=(target,),
                ):
                    continue
                lineage = initial_lineage(pair)
                legal = tuple(int(action) for action in simulator.legal_actions(state))
                for action in legal:
                    tokens, global_values = option_features(
                        state,
                        simulator,
                        starter_tile=STARTER_TILE,
                        pair=pair,
                        lineage=lineage,
                        action=action,
                    )
                    if not (
                        np.isfinite(tokens).all()
                        and np.isfinite(global_values).all()
                    ):
                        raise SourceIntegrityError("Nonfinite O4 source features")
                    action_feature_rows += 1
                restored.append(
                    {
                        **dict(row),
                        "o4_whitelisted_state_sha256": o4_state_sha256,
                        "pair": [list(value) for value in pair.coordinates],
                        "pair_manhattan": pair.manhattan,
                        "pair_chebyshev": pair.chebyshev,
                        "blocker_occupied": pair.blocker_occupied,
                        "blocker_capacity": pair.blocker_capacity,
                        "blocker_density": pair.blocker_density,
                        "legal_count": len(legal),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "root_cluster": root,
                        "target": int(row["target"]),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    report = {
        "source_roots_opened": len(by_root),
        "source_candidate_rows_opened": len(candidates),
        "o4_eligible_rows": len(restored),
        "o4_eligible_roots": len({row["root_cluster"] for row in restored}),
        "action_feature_rows_verified": action_feature_rows,
        "failures": failures,
        "passes": not failures,
        "only_current_support_frames_read": True,
    "final_score_action_outcome_fields_read": False,
        "o3_option_training_bodies_read": False,
    }
    return restored, report


def _cell_selection_hash(
    *,
    role: str,
    family: str,
    target: int,
    root: str,
    frame: int,
    state_hash: str,
) -> str:
    text = (
        f"O4-P0-cell-v1|{role}|{family}|{target}|"
        f"{root}|{frame}|{state_hash}"
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def allocate_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = []
    used_roots: set[str] = set()
    deficits = []
    for role in ROLE_ORDER:
        role_matrix = ROLE_FAMILY_TARGET_COUNTS[role]
        for family_index, family in enumerate(FAMILY_ORDER):
            for target_index, target in enumerate(TARGET_ORDER):
                required = int(role_matrix[family_index][target_index])
                queue = sorted(
                    (
                        {
                            **dict(row),
                            "selection_sha256": _cell_selection_hash(
                                role=role,
                                family=family,
                                target=target,
                                root=str(row["root_cluster"]),
                                frame=int(row["frame_index"]),
                                state_hash=str(row["state_sha1"]),
                            ),
                        }
                        for row in candidates
                        if row["family"] == family
                        and int(row["target"]) == target
                    ),
                    key=lambda row: (
                        row["selection_sha256"],
                        row["root_cluster"],
                        int(row["frame_index"]),
                        row["state_sha1"],
                    ),
                )
                claimed = []
                for row in queue:
                    root = str(row["root_cluster"])
                    if root in used_roots:
                        continue
                    claimed.append({**row, "role": role})
                    used_roots.add(root)
                    if len(claimed) == required:
                        break
                selected.extend(claimed)
                if len(claimed) != required:
                    deficits.append(
                        {
                            "role": role,
                            "family": family,
                            "target": target,
                            "required": required,
                            "selected": len(claimed),
                        }
                    )

    role_family_target_actual = {
        role: {
            family: {
                f"T{target}": sum(
                    row["role"] == role
                    and row["family"] == family
                    and int(row["target"]) == target
                    for row in selected
                )
                for target in TARGET_ORDER
            }
            for family in FAMILY_ORDER
        }
        for role in ROLE_ORDER
    }
    role_counts = Counter(str(row["role"]) for row in selected)
    family_counts = Counter(str(row["family"]) for row in selected)
    target_counts = Counter(int(row["target"]) for row in selected)
    checks = {
        "zero_deficits": not deficits,
        "exact_448_roots": len(selected) == TOTAL_SELECTED_ROOTS,
        "one_state_per_root": len(used_roots) == len(selected),
        "role_counts_exact": dict(role_counts) == ROLE_COUNTS,
        "family_counts_exact": dict(family_counts)
        == validate_frozen_matrices()["combined_family_counts"],
        "target_counts_exact": dict(target_counts)
        == validate_frozen_matrices()["combined_target_counts"],
        "every_cell_exact": all(
            role_family_target_actual[role][family][f"T{target}"]
            == ROLE_FAMILY_TARGET_COUNTS[role][family_index][target_index]
            for role in ROLE_ORDER
            for family_index, family in enumerate(FAMILY_ORDER)
            for target_index, target in enumerate(TARGET_ORDER)
        ),
        "deterministic_no_backtracking": True,
    }
    return {
        "selected": selected,
        "selected_manifest_sha256": canonical_json_hash(selected),
        "deficits": deficits,
        "role_counts": dict(role_counts),
        "family_counts": dict(family_counts),
        "target_counts": {f"T{key}": value for key, value in target_counts.items()},
        "role_family_target_counts": role_family_target_actual,
        "checks": checks,
        "passes": all(checks.values()),
    }


def support_and_allocation(
    source_report: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not source_report["upper_bound_feasible"]:
        return (
            {
                "scan_skipped": True,
                "reason": "raw_unselected_support_upper_bound_cannot_fill_frozen_matrix",
                "source_roots_opened": 0,
                "source_candidate_rows_opened": 0,
                "o4_eligible_rows": 0,
                "o4_eligible_roots": 0,
                "action_feature_rows_verified": 0,
                "failures": [],
                "passes": True,
                "o3_option_training_bodies_read": False,
                "final_score_action_outcome_fields_read": False,
            },
            {
                "selected": [],
                "selected_manifest_sha256": canonical_json_hash([]),
                "deficits": [
                    {
                        "scope": "upper_bound",
                        "failed_checks": [
                            key
                            for key, value in source_report[
                                "upper_bound_checks"
                            ].items()
                            if not value
                        ],
                    }
                ],
                "checks": {
                    "zero_deficits": False,
                    "exact_448_roots": False,
                    "one_state_per_root": True,
                    "deterministic_no_backtracking": True,
                },
                "passes": False,
                "allocation_attempted": False,
            },
        )
    restored, restore_report = restore_o4_candidates(candidates)
    if not restore_report["passes"]:
        raise SourceIntegrityError("Permitted O4 source restoration failed")
    allocation = allocate_candidates(restored)
    allocation["allocation_attempted"] = True
    return restore_report, allocation


def _single_arm_rows() -> list[dict[str, Any]]:
    bases = STREAM_BASES["learning"]
    rows = []
    for root_index in range(ROLE_COUNTS["train"]):
        for trajectory_index in range(TRAJECTORIES_PER_TRAIN_ROOT):
            code = root_index * TRAJECTORIES_PER_TRAIN_ROOT + trajectory_index
            round_index = 1 + int(trajectory_index >= 2) + int(
                trajectory_index >= 4
            ) + int(trajectory_index >= 5)
            rows.append(
                {
                    "purpose": "learning",
                    "root_index": root_index,
                    "trajectory_index": trajectory_index,
                    "round_index": round_index,
                    **{
                        field: base + code
                        for field, base in bases.items()
                    },
                }
            )
    return rows


def _paired_rows(
    *,
    purpose: str,
    roots: int,
    repeats: int,
    bases: Mapping[str, int],
    code_offset: int,
) -> list[dict[str, Any]]:
    rows = []
    for root_index in range(int(roots)):
        for replicate in range(int(repeats)):
            code = int(code_offset) + root_index * int(repeats) + replicate
            rows.append(
                {
                    "purpose": purpose,
                    "root_index": root_index,
                    "replicate": replicate,
                    "logical_seed": int(bases["logical_seed"]) + code,
                    "deck_stream_id": int(bases["deck_stream_id"]) + code,
                    "slot_stream_id": int(bases["slot_stream_id"]) + code,
                    "control_policy_stream_id": int(
                        bases["policy_stream_id"]
                    )
                    + 2 * code,
                    "treatment_policy_stream_id": int(
                        bases["policy_stream_id"]
                    )
                    + 2 * code
                    + 1,
                }
            )
    return rows


def future_stream_rows() -> list[dict[str, Any]]:
    rows = _single_arm_rows()
    rows.extend(
        _paired_rows(
            purpose="option_development",
            roots=ROLE_COUNTS["development"],
            repeats=OPTION_REPEATS,
            bases=STREAM_BASES["option"],
            code_offset=0,
        )
    )
    rows.extend(
        _paired_rows(
            purpose="option_untouched_mechanism",
            roots=ROLE_COUNTS["untouched_mechanism"],
            repeats=OPTION_REPEATS,
            bases=STREAM_BASES["option"],
            code_offset=1_000_000,
        )
    )
    rows.extend(
        _paired_rows(
            purpose="normal_development",
            roots=NORMAL_DEVELOPMENT_ROOTS,
            repeats=1,
            bases=STREAM_BASES["normal_development"],
            code_offset=0,
        )
    )
    rows.extend(
        _paired_rows(
            purpose="confirmation",
            roots=CONFIRMATION_ROOTS,
            repeats=1,
            bases=STREAM_BASES["confirmation"],
            code_offset=0,
        )
    )
    return rows


def _stream_sets(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, set[int]]:
    result = {field: set() for field in STREAM_FIELDS}
    for row in rows:
        for field in STREAM_FIELDS[:3]:
            result[field].add(int(row[field]))
        if "policy_stream_id" in row:
            result["policy_stream_id"].add(int(row["policy_stream_id"]))
        else:
            result["policy_stream_id"].add(
                int(row["control_policy_stream_id"])
            )
            result["policy_stream_id"].add(
                int(row["treatment_policy_stream_id"])
            )
    return result


def stream_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    purpose_counts = Counter(str(row["purpose"]) for row in rows)
    paired = [row for row in rows if "control_policy_stream_id" in row]
    sets = _stream_sets(rows)
    expected_rows = {
        "learning": 1_152,
        "option_development": 512,
        "option_untouched_mechanism": 1_536,
        "normal_development": 512,
        "confirmation": 2_560,
    }
    checks = {
        "purpose_counts_exact": dict(purpose_counts) == expected_rows,
        "no_acquisition_rows": "acquisition" not in purpose_counts,
        "learning_schedule_exact": all(
            row["round_index"]
            == (1 + int(row["trajectory_index"] >= 2)
                + int(row["trajectory_index"] >= 4)
                + int(row["trajectory_index"] >= 5))
            for row in rows
            if row["purpose"] == "learning"
        ),
        "paired_exogenous_unique": len(
            {
                (
                    row["logical_seed"],
                    row["deck_stream_id"],
                    row["slot_stream_id"],
                )
                for row in paired
            }
        )
        == len(paired),
        "paired_policy_ids_unique": len(
            [
                int(row[field])
                for row in paired
                for field in (
                    "control_policy_stream_id",
                    "treatment_policy_stream_id",
                )
            ]
        )
        == len(
            {
                int(row[field])
                for row in paired
                for field in (
                    "control_policy_stream_id",
                    "treatment_policy_stream_id",
                )
            }
        ),
        "all_stream_types_internally_unique": all(
            len(values)
            == (
                sum(
                    2 if "control_policy_stream_id" in row else 1
                    for row in rows
                )
                if field == "policy_stream_id"
                else len(rows)
            )
            for field, values in sets.items()
        ),
        "streams_reserved_not_consumed": True,
    }
    return {
        "row_count": len(rows),
        "purpose_counts": dict(purpose_counts),
        "stream_bases": STREAM_BASES,
        "stream_value_counts": {
            field: len(values) for field, values in sets.items()
        },
        "manifest_sha256": canonical_json_hash(list(rows)),
        "checks": checks,
        "passes": all(checks.values()),
        "streams_consumed": 0,
    }


def o3_learning_stream_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(O3_STREAM_MANIFEST_PATH.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SourceIntegrityError("O3 stream manifest rows are missing")
    learning = [dict(row) for row in rows if row.get("purpose") == "learning"]
    checks = {
        "file_hash_exact": sha256_path(O3_STREAM_MANIFEST_PATH)
        == IMMUTABLE_INPUT_HASHES[str(O3_STREAM_MANIFEST_PATH)],
        "payload_hash_exact": payload.get("payload_sha256")
        == IMMUTABLE_PAYLOAD_HASHES[str(O3_STREAM_MANIFEST_PATH)],
        "payload_self_hash_valid": _verify_self_hash(payload, "payload_sha256"),
        "exact_1152_learning_rows": len(learning) == 1_152,
        "learning_rows_unique": all(
            len({int(row[field]) for row in learning}) == len(learning)
            for field in STREAM_FIELDS
        ),
    }
    return (
        {
            "path": str(O3_STREAM_MANIFEST_PATH),
            "file_sha256": sha256_path(O3_STREAM_MANIFEST_PATH),
            "payload_sha256": payload.get("payload_sha256"),
            "learning_rows": len(learning),
            "learning_manifest_sha256": canonical_json_hash(learning),
            "checks": checks,
            "passes": all(checks.values()),
            "option_training_episode_or_metadata_bodies_read": False,
        },
        learning,
    )


def collision_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    scan_root: Path = Path("threes_rl/runs"),
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    requested = _stream_sets(rows)
    found: dict[str, set[int]] = defaultdict(set)
    matched = []
    excluded = []
    skip_directories = (
        O3_OPTION_TRAINING_DIR,
        O3_ACQUISITION_DIR / "source_replays",
        O3_RECOVERY_DIR / "source_replays",
    )
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        if _is_within(path, out_dir):
            continue
        classification = next(
            (
                (
                    "o3_option_training_body_unread"
                    if directory == O3_OPTION_TRAINING_DIR
                    else "o3_acquisition_replay_body_unread"
                )
                for directory in skip_directories
                if _is_within(path, directory)
            ),
            None,
        )
        if classification is not None:
            excluded.append(
                {
                    "path": str(path),
                    "classification": classification,
                    "bytes": int(path.stat().st_size),
                }
            )
            continue
        values = history._scan_history_file(path)
        if not values:
            continue
        for field, items in values.items():
            found[field].update(items)
        matched.append(
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "bytes": int(path.stat().st_size),
                "counts": {
                    field: len(items)
                    for field, items in sorted(values.items())
                },
            }
        )

    o3_audit, o3_learning = o3_learning_stream_audit()
    o3_sets = _stream_sets(o3_learning)
    for field, values in o3_sets.items():
        found[field].update(values)
    collisions = {}
    for field, requested_values in requested.items():
        prior = set(found.get(field, set()))
        if field == "logical_seed":
            for alias in (
                "seed",
                "root_seed",
                "source_seed",
                "fresh_root_seed",
            ):
                prior.update(found.get(alias, set()))
        collisions[field] = sorted(requested_values.intersection(prior))
    o3_direct_collisions = {
        field: sorted(requested[field].intersection(o3_sets[field]))
        for field in STREAM_FIELDS
    }
    checks = {
        "zero_historical_collisions": not any(collisions.values()),
        "exact_o3_learning_reservation_included": o3_audit["passes"],
        "zero_o3_learning_stream_collisions": not any(
            o3_direct_collisions.values()
        ),
        "o3_option_training_bodies_unread": any(
            row["classification"] == "o3_option_training_body_unread"
            for row in excluded
        ),
        "o3_acquisition_replay_bodies_unread_by_collision_scan": any(
            row["classification"] == "o3_acquisition_replay_body_unread"
            for row in excluded
        ),
    }
    return {
        "scan_root": str(scan_root),
        "matched_sources": matched,
        "matched_source_count": len(matched),
        "matched_sources_sha256": canonical_json_hash(matched),
        "excluded_sources": excluded,
        "excluded_source_count": len(excluded),
        "excluded_sources_sha256": canonical_json_hash(excluded),
        "o3_learning_stream_audit": o3_audit,
        "o3_direct_collisions": o3_direct_collisions,
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
    }


def policy_audit() -> dict[str, Any]:
    prior = json.loads(O3_POLICY_AUDIT_PATH.read_text())
    if not _verify_self_hash(prior, "payload_sha256"):
        raise SourceIntegrityError("O3 policy audit payload changed")
    current_lock, _loaded = qd5._policy_lock()
    signatures = {
        O3_TO_O4[family]: value
        for family, value in prior["signature_sha256"].items()
    }
    pairwise = [
        {
            **row,
            "left": O3_TO_O4[str(row["left"])],
            "right": O3_TO_O4[str(row["right"])],
        }
        for row in prior["pairwise"]
    ]
    checks = {
        "prior_file_hash_exact": sha256_path(O3_POLICY_AUDIT_PATH)
        == IMMUTABLE_INPUT_HASHES[str(O3_POLICY_AUDIT_PATH)],
        "prior_payload_hash_exact": prior["payload_sha256"]
        == IMMUTABLE_PAYLOAD_HASHES[str(O3_POLICY_AUDIT_PATH)],
        "prior_family_order_exact": tuple(prior["family_order"])
        == O3_FAMILY_ORDER,
        "o4_family_order_exact": tuple(signatures) == FAMILY_ORDER,
        "five_unique_signatures": len(set(signatures.values())) == 5,
        "all_pairwise_distinctness_gates_pass": all(
            row["passes"] for row in pairwise
        ),
        "current_policy_payload_exact": current_lock["policy_lock_sha256"]
        == prior["policy_lock_sha256"],
        "no_new_action_evaluation": True,
        "no_retiming": True,
    }
    return {
        "family_order": list(FAMILY_ORDER),
        "signatures": signatures,
        "pairwise": pairwise,
        "tie_state_counts": {
            O3_TO_O4[family]: count
            for family, count in prior["tie_state_counts"].items()
        },
        "policy_lock_sha256": current_lock["policy_lock_sha256"],
        "source_policy_audit": {
            "path": str(O3_POLICY_AUDIT_PATH),
            "file_sha256": sha256_path(O3_POLICY_AUDIT_PATH),
            "payload_sha256": prior["payload_sha256"],
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def domain_proof() -> dict[str, Any]:
    exhaustive = exhaustive_blocker_domain_proof()
    model = O4DesignatedPairNet()
    count = sum(parameter.numel() for parameter in model.parameters())
    checks = {
        "exhaustive_blocker_domain": exhaustive["passes"],
        "exact_120_coordinate_pairs": exhaustive["coordinate_pairs"] == 120,
        "exact_43296_occupancy_cases": exhaustive["occupancy_cases"] == 43_296,
        "density_range_0_1": exhaustive["minimum_density"] == 0.0
        and exhaustive["maximum_density"] == 1.0,
        "parameter_count_102557": count == FROZEN_PARAMETER_COUNT,
        "schema_finite_hash": len(schema_sha256()) == 64,
        "matrix_contract_exact": validate_frozen_matrices()["passes"],
    }
    return {
        "schema_sha256": schema_sha256(),
        "parameter_count": count,
        "exhaustive": exhaustive,
        "matrix_contract": validate_frozen_matrices(),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not _verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise SourceIntegrityError("O4 test evidence payload mismatch")
    expected = {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "option_implementation_sha256": sha256_path(OPTION_PATH),
        "power_implementation_sha256": sha256_path(POWER_PATH),
        "p0_implementation_sha256": sha256_path(RUNNER_PATH),
        "option_tests_sha256": sha256_path(OPTION_TEST_PATH),
        "p0_tests_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise SourceIntegrityError("O4 test evidence source identity mismatch")
    if not payload.get("passes"):
        raise SourceIntegrityError("O4 test evidence did not pass")
    return payload


def seal_test_evidence(
    *,
    focused_passed: int,
    regression_passed: int,
    commands: list[str],
) -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_test_evidence",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "option_implementation_sha256": sha256_path(OPTION_PATH),
        "power_implementation_sha256": sha256_path(POWER_PATH),
        "p0_implementation_sha256": sha256_path(RUNNER_PATH),
        "option_tests_sha256": sha256_path(OPTION_TEST_PATH),
        "p0_tests_sha256": sha256_path(TEST_PATH),
        "dependency_hashes": _dependency_hashes(),
        "focused_tests_passed": int(focused_passed),
        "regression_tests_passed": int(regression_passed),
        "commands": commands,
        "passes": True,
        "games_generated": 0,
        "streams_consumed": 0,
        "labels_generated": 0,
        "models_fit": 0,
        "outcomes_inspected": False,
        "o3_option_training_bodies_read": False,
    }
    return _write_immutable_json(
        TEST_EVIDENCE_PATH,
        payload,
        self_hash_field="test_evidence_payload_sha256",
    )


def _bound_commands(out_dir: Path) -> dict[str, str]:
    base = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.o4_p0_preflight"
    )
    suffix = f" --out-dir {out_dir}'"
    return {
        "open": f"{base} open{suffix}",
        "run": f"{base} run{suffix}",
    }


def _marker_identity(out_dir: Path) -> dict[str, Any]:
    evidence = _load_test_evidence()
    rows = future_stream_rows()
    return {
        **_current_bindings(),
        "bound_out_dir": str(out_dir.resolve()),
        "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "test_evidence_payload_sha256": evidence[
            "test_evidence_payload_sha256"
        ],
        "source_universe_roots": O3_ACQUISITION_ROOTS,
        "selected_o3_roots_excluded": O3_SELECTED_ROOTS,
        "o4_root_count": TOTAL_SELECTED_ROOTS,
        "future_stream_manifest_sha256": canonical_json_hash(rows),
        "future_stream_row_count": len(rows),
        "commands": _bound_commands(out_dir),
    }


def open_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O4 P0 output directory is immutable")
    if out_dir.exists():
        raise FileExistsError(f"O4 P0 namespace already exists: {out_dir}")
    evidence = _load_test_evidence()
    immutable = {
        path: {
            "expected": expected,
            "actual": sha256_path(Path(path)),
        }
        for path, expected in IMMUTABLE_INPUT_HASHES.items()
    }
    heavy = _heavy_process_audit()
    services = history.service_health()
    free_gib = shutil.disk_usage(out_dir.parent).free / 1024**3
    checks = {
        "test_evidence_exact": evidence["passes"],
        "immutable_input_file_hashes_exact": all(
            row["actual"] == row["expected"] for row in immutable.values()
        ),
        "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
        "no_competing_heavy_process": heavy["passes"],
        "free_disk_above_120_gib": free_gib > TARGET_FREE_GIB,
        "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
        "services_dashboard_top_three": services["passes"],
        "zero_prior_o4_namespace": True,
        "zero_games_streams_labels_models_outcomes": True,
    }
    if not all(checks.values()):
        raise OperationalHold(f"O4 P0 open checks failed: {checks}")
    marker = {
        **_marker_identity(out_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "O4_P0_OPENED_ZERO_WORK",
        "preopen": {
            "immutable_input_file_hashes": immutable,
            "heavy_process_audit": heavy,
            "service_health": services,
            "free_gib": free_gib,
            "nice": history.current_nice(),
        },
        "checks": checks,
        "zero_work": {
            "games": 0,
            "streams_consumed": 0,
            "source_replay_bodies_opened": 0,
            "o3_option_training_bodies_read": 0,
            "labels": 0,
            "models_fit": 0,
            "policy_outcomes": 0,
            "scores_or_actions_inspected": 0,
            "dashboard_changes": 0,
        },
    }
    return _write_immutable_json(
        out_dir / MARKER_NAME,
        marker,
        self_hash_field="opened_payload_sha256",
    )


def _load_marker(out_dir: Path) -> dict[str, Any]:
    path = out_dir / MARKER_NAME
    if not path.is_file():
        raise FileNotFoundError("O4 P0 marker is missing")
    marker = json.loads(path.read_text())
    if not _verify_self_hash(marker, "opened_payload_sha256"):
        raise SourceIntegrityError("O4 P0 marker self hash mismatch")
    expected = _marker_identity(out_dir)
    for key, value in expected.items():
        if marker.get(key) != value:
            raise SourceIntegrityError(f"O4 P0 marker binding mismatch: {key}")
    return marker


def _write_manifest(
    out_dir: Path,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    body = _write_immutable_json(
        out_dir / name,
        payload,
        self_hash_field="payload_sha256",
    )
    return {
        "path": str(out_dir / name),
        "file_sha256": sha256_path(out_dir / name),
        "payload_sha256": body["payload_sha256"],
    }


def _decision(
    *,
    integrity_checks: Mapping[str, bool],
    support_checks: Mapping[str, bool],
) -> str:
    if not all(integrity_checks.values()):
        return "KILL_O4_REPRESENTATION_PREFLIGHT"
    if not all(support_checks.values()):
        return "HOLD_O4_DATA_SUPPORT"
    return "READY_O4_DOMAIN_SAFE_OPTION_PREFLIGHT"


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    marker = _load_marker(out_dir)
    result_path = out_dir / RESULT_NAME
    if result_path.exists():
        raise FileExistsError("O4 P0 terminal result already exists")
    unexpected = {
        path.name for path in out_dir.iterdir() if path.name != MARKER_NAME
    }
    if unexpected:
        raise SourceIntegrityError(
            f"O4 P0 namespace contains unexpected work: {sorted(unexpected)}"
        )
    try:
        tests = _load_test_evidence()
        immutable = immutable_input_audit()
        domain = domain_proof()
        source_report, candidates, _union_index = load_source_pool()
        replay_sources = verify_candidate_source_replays(candidates)
        restore_report, allocation = support_and_allocation(
            source_report,
            candidates,
        )
        rows = future_stream_rows()
        streams = stream_contract(rows)
        collision = collision_audit(rows, out_dir=out_dir)
        power = power_table()
        policies = policy_audit()
        heavy = _heavy_process_audit()
        services = history.service_health()
        free_gib = shutil.disk_usage(out_dir).free / 1024**3

        artifacts = {
            "domain": _write_manifest(
                out_dir,
                DOMAIN_NAME,
                {
                    "version": f"{VERSION}_domain",
                    **domain,
                    "outcomes_opened": False,
                },
            ),
            "source_pool": _write_manifest(
                out_dir,
                SOURCE_NAME,
                {
                    "version": f"{VERSION}_source",
                    **source_report,
                    "geometry_restore": restore_report,
                },
            ),
            "source_replays": _write_manifest(
                out_dir,
                SOURCE_REPLAY_NAME,
                {
                    "version": f"{VERSION}_source_replays",
                    **replay_sources,
                },
            ),
            "selection": _write_manifest(
                out_dir,
                SELECTION_NAME,
                {
                    "version": f"{VERSION}_selection",
                    **allocation,
                    "labels_generated": 0,
                    "policy_outcomes_opened": False,
                },
            ),
            "streams": _write_manifest(
                out_dir,
                STREAM_NAME,
                {
                    "version": f"{VERSION}_streams",
                    "rows": rows,
                    "contract": streams,
                    "streams_consumed": 0,
                },
            ),
            "collision": _write_manifest(
                out_dir,
                COLLISION_NAME,
                {
                    "version": f"{VERSION}_collision",
                    **collision,
                },
            ),
            "power": _write_manifest(
                out_dir,
                POWER_NAME,
                {
                    "version": f"{VERSION}_power",
                    **power,
                    "outcomes_used": False,
                },
            ),
            "policies": _write_manifest(
                out_dir,
                POLICY_NAME,
                {
                    "version": f"{VERSION}_policies",
                    **policies,
                },
            ),
        }

        integrity_checks = {
            "immutable_source_identities": immutable["passes"],
            "domain_proof": domain["passes"],
            "source_pool_integrity": source_report["passes"],
            "source_replay_hashes": replay_sources["passes"],
            "permitted_geometry_restore": restore_report["passes"],
            "tests_exact": tests["passes"],
            "stream_contract": streams["passes"],
            "stream_collisions_zero": collision["passes"],
            "policy_identities_and_signatures": policies["passes"],
        }
        support_checks = {
            "raw_support_upper_bound_feasible": source_report[
                "upper_bound_feasible"
            ],
            "exact_448_allocation": allocation["passes"],
            "n192_or150_power": power["passes"],
            "one_heavy_job": heavy["passes"],
            "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
            "free_disk_above_120_gib": free_gib > TARGET_FREE_GIB,
            "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
            "services_dashboard_top_three": services["passes"],
            "projected_runtime_frozen": PROJECTED_ACTIVE_SECONDS
            == 18 * 3_600,
            "projected_storage_below_4_gib": PROJECTED_STORAGE_BYTES
            < STORAGE_CAP_BYTES,
            "zero_fresh_work": True,
        }
        decision = _decision(
            integrity_checks=integrity_checks,
            support_checks=support_checks,
        )
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "continue": (
                "O4 acquisition/training requires separate authorization"
                if decision == "READY_O4_DOMAIN_SAFE_OPTION_PREFLIGHT"
                else "NONE"
            ),
            "hold": [
                "o4_acquisition",
                "o4_training",
                "o4_mechanism_outcomes",
                "normal_start_development",
                "confirmation",
                "promotion",
            ],
            "kill": decision == "KILL_O4_REPRESENTATION_PREFLIGHT",
            "promote": False,
            "marker": {
                "path": str(out_dir / MARKER_NAME),
                "file_sha256": sha256_path(out_dir / MARKER_NAME),
                "payload_sha256": marker["opened_payload_sha256"],
            },
            "artifacts": artifacts,
            "summaries": {
                "source_universe": source_report["union_roots"],
                "selected_o3_roots_excluded": source_report[
                    "selected_roots_excluded"
                ],
                "unselected_source_universe": source_report[
                    "unselected_root_universe"
                ],
                "candidate_root_upper_bounds_by_family": source_report[
                    "family_root_upper_bounds"
                ],
                "candidate_root_upper_bounds_by_family_target": source_report[
                    "family_target_root_upper_bounds"
                ],
                "allocation_count": len(allocation["selected"]),
                "allocation_deficits": allocation["deficits"],
                "stream_rows_reserved": streams["row_count"],
                "o3_learning_rows_explicitly_checked": collision[
                    "o3_learning_stream_audit"
                ]["learning_rows"],
                "collision_source_count": collision["matched_source_count"],
                "power_n": power["selected_roots"],
                "power_or150": next(
                    row["power_full_gate"]
                    for row in power["rows"]
                    if row["true_common_odds_ratio"] == 1.50
                ),
                "grid_mde": power["grid_mde"],
                "schema_sha256": domain["schema_sha256"],
                "parameter_count": domain["parameter_count"],
                "family_signatures": policies["signatures"],
            },
            "integrity_checks": integrity_checks,
            "support_checks": support_checks,
            "process": {
                "nice": history.current_nice(),
                "heavy_process_audit": heavy,
                "free_gib": free_gib,
                "service_health": services,
                "projected_active_seconds": PROJECTED_ACTIVE_SECONDS,
                "projected_storage_bytes": PROJECTED_STORAGE_BYTES,
                "storage_cap_bytes": STORAGE_CAP_BYTES,
            },
            "zero_work": {
                "fresh_games": 0,
                "fresh_streams_consumed": 0,
                "o3_option_training_bodies_read": False,
                "o3_selected_replay_bodies_read": False,
                "final_score_action_outcome_fields_read": False,
                "labels": 0,
                "models_fit": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    except SourceIntegrityError as error:
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "KILL_O4_REPRESENTATION_PREFLIGHT",
            "continue": "NONE",
            "hold": ["all_o4_execution"],
            "kill": True,
            "promote": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "marker": {
                "path": str(out_dir / MARKER_NAME),
                "file_sha256": sha256_path(out_dir / MARKER_NAME),
                "payload_sha256": marker["opened_payload_sha256"],
            },
            "zero_work": {
                "fresh_games": 0,
                "fresh_streams_consumed": 0,
                "labels": 0,
                "models_fit": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    except Exception as error:
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_O4_DATA_SUPPORT",
            "continue": "NONE",
            "hold": ["all_o4_execution"],
            "kill": False,
            "promote": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "marker": {
                "path": str(out_dir / MARKER_NAME),
                "file_sha256": sha256_path(out_dir / MARKER_NAME),
                "payload_sha256": marker["opened_payload_sha256"],
            },
            "zero_work": {
                "fresh_games": 0,
                "fresh_streams_consumed": 0,
                "labels": 0,
                "models_fit": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    return _write_immutable_json(
        result_path,
        result,
        self_hash_field="result_payload_sha256",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("open", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence = subparsers.add_parser("seal-test-evidence")
    evidence.add_argument("--focused-passed", type=int, required=True)
    evidence.add_argument("--regression-passed", type=int, required=True)
    evidence.add_argument(
        "--test-command",
        dest="test_commands",
        action="append",
        default=[],
    )
    args = parser.parse_args()
    if args.command == "open":
        result = open_preflight(args.out_dir)
    elif args.command == "run":
        result = run_preflight(args.out_dir)
    else:
        result = seal_test_evidence(
            focused_passed=args.focused_passed,
            regression_passed=args.regression_passed,
            commands=list(args.test_commands),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

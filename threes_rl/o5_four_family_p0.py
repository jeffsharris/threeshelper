"""Outcome-free O5 four-family domain-safe P0."""

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
from threes_rl.o4_domain_safe_pair_option import (
    exhaustive_blocker_domain_proof,
    initial_lineage,
    option_features,
    parameter_count,
    root_option_eligible,
    schema_sha256,
    select_designated_pair,
)
from threes_rl.o4_power_contract import power_table
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import SimState, ThreesSim, preview_from_label


VERSION = "o5_four_family_domain_safe_p0_v1"
STARTER_TILE = 1536

CHARTER_PATH = Path("threes_rl/O5_FOUR_FAMILY_DOMAIN_SAFE_P0_CHARTER.md")
RUNNER_PATH = Path("threes_rl/o5_four_family_p0.py")
TEST_PATH = Path("tests/test_rl_o5_four_family_p0.py")
OPTION_PATH = Path("threes_rl/o4_domain_safe_pair_option.py")
POWER_PATH = Path("threes_rl/o4_power_contract.py")
OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1"
)
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/"
    "o5_four_family_domain_safe_p0_test_evidence_v1.json"
)

MARKER_NAME = "O5_P0_OPENED.json"
RESULT_NAME = "O5_P0_RESULT.json"
SOURCE_NAME = "O5_P0_SOURCE_POOL.json"
SOURCE_REPLAY_NAME = "O5_P0_SOURCE_REPLAY_MANIFEST.json"
SELECTION_NAME = "O5_P0_SELECTED_ROOTS.json"
STREAM_NAME = "O5_P0_STREAM_MANIFEST.json"
COLLISION_NAME = "O5_P0_COLLISION_AUDIT.json"
POWER_NAME = "O5_P0_POWER_TABLE.json"
POLICY_NAME = "O5_P0_POLICY_AUDIT.json"
DOMAIN_NAME = "O5_P0_DOMAIN_PROOF.json"

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

O4_V1_MARKER_PATH = Path(
    "threes_rl/runs/forensics/o4_domain_safe_p0_v1/O4_P0_OPENED.json"
)
O4_V1_HOLD_PATH = Path(
    "threes_rl/runs/forensics/o4_domain_safe_p0_v1/"
    "O4_P0_V1_ENGINEERING_HOLD.json"
)
O4_V2_MARKER_PATH = Path(
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/O4_P0_V2_OPENED.json"
)
O4_V2_RESULT_PATH = Path(
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/O4_P0_V2_RESULT.json"
)
O4_V2_STREAM_PATH = Path(
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/"
    "O4_P0_STREAM_MANIFEST.json"
)

IMMUTABLE_FILE_HASHES = {
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
    str(O4_V1_MARKER_PATH): (
        "7f84bbd9679b9d6294a0530b47b5ba01749426191a1a3f509bf38a48114723b6"
    ),
    str(O4_V1_HOLD_PATH): (
        "17be1eb2c5ecf0be1a7331779e5eab7cc3159eb760d50d4f4b7aacdf395332e8"
    ),
    str(O4_V2_MARKER_PATH): (
        "9d9f032f61fa637941d677e788dcb7d2dcec70179a7ea9a2fafe128af73336da"
    ),
    str(O4_V2_RESULT_PATH): (
        "897cac07ce2625f5616690f0a4611e11948e6ca58a55b828ee43f92b493893cd"
    ),
    str(O4_V2_STREAM_PATH): (
        "24c94fe8898847a6b54676aec6d5e78511bf687ec18dc5d410a9194a0bde6828"
    ),
    str(OPTION_PATH): (
        "95a4da48fb7550e87b09e1f1594cdbdc062a52c7df544b7445b5e58878c87f41"
    ),
    str(POWER_PATH): (
        "16e2c26c9e1f2b176937f1a0546604b878d45875b4c29dbc83a441588f7fc5cd"
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
O4_STREAM_MANIFEST_SHA256 = (
    "3d0108b091b1d7a42d2113e8c16a3651a653041d84d9999eb4a37edec3dc0ab0"
)
O4_STREAM_PAYLOAD_SHA256 = (
    "4e3202cd25d44e9dffa860ceff0dc81ed14672d7c3f5e2396b612e9a9b95a6a9"
)
O4_OPERATOR_SCHEMA_SHA256 = (
    "60a83881d8e8275a4aa2d03df06815d65e5b247b16f36118009f42f2ce3098ba"
)

DEPENDENCY_PATHS = (
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
INCLUDED_O3_FAMILIES = O3_FAMILY_ORDER[:4]
EXCLUDED_O3_FAMILY = "o3_qd_v2"
FAMILY_ORDER = (
    "o5_corner2",
    "o5_expectimax2",
    "o5_parent_mc1000",
    "o5_replaycal",
)
O3_TO_O5 = dict(zip(INCLUDED_O3_FAMILIES, FAMILY_ORDER, strict=True))
ROLE_ORDER = ("train", "development", "untouched_mechanism")
TARGET_ORDER = (48, 96, 192)
ROLE_FAMILY_TARGET_COUNTS = {
    "train": (
        (16, 16, 16),
        (16, 16, 16),
        (16, 16, 16),
        (16, 16, 16),
    ),
    "development": (
        (6, 5, 5),
        (5, 6, 5),
        (5, 5, 6),
        (6, 5, 5),
    ),
    "untouched_mechanism": (
        (16, 16, 16),
        (16, 16, 16),
        (16, 16, 16),
        (16, 16, 16),
    ),
}
ROLE_FAMILY_COUNTS = {
    "train": (48, 48, 48, 48),
    "development": (16, 16, 16, 16),
    "untouched_mechanism": (48, 48, 48, 48),
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
        "logical_seed": 181_000_000_000,
        "deck_stream_id": 182_000_000_000,
        "slot_stream_id": 183_000_000_000,
        "policy_stream_id": 184_000_000_000,
    },
    "option": {
        "logical_seed": 185_000_000_000,
        "deck_stream_id": 186_000_000_000,
        "slot_stream_id": 187_000_000_000,
        "policy_stream_id": 188_000_000_000,
    },
    "normal_development": {
        "logical_seed": 189_000_000_000,
        "deck_stream_id": 190_000_000_000,
        "slot_stream_id": 191_000_000_000,
        "policy_stream_id": 192_000_000_000,
    },
    "confirmation": {
        "logical_seed": 193_000_000_000,
        "deck_stream_id": 194_000_000_000,
        "slot_stream_id": 195_000_000_000,
        "policy_stream_id": 196_000_000_000,
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
    """Fail-closed immutable identity or representation error."""


class OperationalHold(RuntimeError):
    """Fail-closed operational hold."""


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    self_hash_field: str = "payload_sha256",
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Immutable artifact exists: {path}")
    serializable = _normalize(dict(payload))
    serializable[self_hash_field] = canonical_json_hash(
        {
            key: value
            for key, value in serializable.items()
            if key != self_hash_field
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="ascii") as handle:
        json.dump(
            serializable,
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "payload_sha256": serializable[self_hash_field],
    }


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    expected = payload.get(field)
    if not isinstance(expected, str):
        return False
    body = {key: value for key, value in payload.items() if key != field}
    return expected == canonical_json_hash(body)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def validate_frozen_matrices() -> dict[str, Any]:
    family_totals = [0] * len(FAMILY_ORDER)
    target_totals = [0] * len(TARGET_ORDER)
    role_checks: dict[str, dict[str, bool]] = {}
    for role in ROLE_ORDER:
        matrix = ROLE_FAMILY_TARGET_COUNTS[role]
        row_totals = tuple(sum(row) for row in matrix)
        column_totals = tuple(
            sum(matrix[row][column] for row in range(len(FAMILY_ORDER)))
            for column in range(len(TARGET_ORDER))
        )
        for index, value in enumerate(row_totals):
            family_totals[index] += value
        for index, value in enumerate(column_totals):
            target_totals[index] += value
        role_checks[role] = {
            "matrix_shape_exact": (
                len(matrix) == len(FAMILY_ORDER)
                and all(len(row) == len(TARGET_ORDER) for row in matrix)
            ),
            "family_marginals_exact": (
                row_totals == ROLE_FAMILY_COUNTS[role]
            ),
            "target_marginals_exact": (
                column_totals == ROLE_TARGET_COUNTS[role]
            ),
            "role_count_exact": sum(row_totals) == ROLE_COUNTS[role],
        }
    checks = {
        "all_role_matrices_exact": all(
            all(values.values()) for values in role_checks.values()
        ),
        "total_roots_448": sum(ROLE_COUNTS.values()) == TOTAL_SELECTED_ROOTS,
        "combined_family_marginals": tuple(family_totals)
        == (112, 112, 112, 112),
        "combined_target_marginals": tuple(target_totals) == (150, 149, 149),
        "family_share_exact_25_percent": all(
            value / TOTAL_SELECTED_ROOTS == 0.25
            for value in family_totals
        ),
    }
    return {
        "role_checks": role_checks,
        "combined_family_counts": dict(
            zip(FAMILY_ORDER, family_totals, strict=True)
        ),
        "combined_target_counts": dict(
            zip(TARGET_ORDER, target_totals, strict=True)
        ),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _dependency_hashes() -> dict[str, str]:
    return {str(path): sha256_path(path) for path in DEPENDENCY_PATHS}


def _current_bindings() -> dict[str, Any]:
    return _normalize(
        {
            "version": VERSION,
            "charter_sha256": sha256_path(CHARTER_PATH),
            "runner_sha256": sha256_path(RUNNER_PATH),
            "tests_sha256": sha256_path(TEST_PATH),
            "o4_operator_sha256": sha256_path(OPTION_PATH),
            "o4_power_sha256": sha256_path(POWER_PATH),
            "dependency_hashes": _dependency_hashes(),
            "immutable_file_hashes": IMMUTABLE_FILE_HASHES,
            "family_order": FAMILY_ORDER,
            "target_order": TARGET_ORDER,
            "role_order": ROLE_ORDER,
            "role_family_target_counts": ROLE_FAMILY_TARGET_COUNTS,
            "stream_bases": STREAM_BASES,
            "future_stream_row_count": len(future_stream_rows()),
            "future_stream_manifest_sha256": canonical_json_hash(
                future_stream_rows()
            ),
            "matrix_contract": validate_frozen_matrices(),
        }
    )


def immutable_input_audit(*, parse_payloads: bool) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for raw_path, expected in IMMUTABLE_FILE_HASHES.items():
        path = Path(raw_path)
        actual = sha256_path(path)
        files[raw_path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
            "bytes": int(path.stat().st_size),
        }
    payloads: dict[str, dict[str, Any]] = {}
    if parse_payloads:
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
                "self_hash_valid": bool(
                    field and _verify_self_hash(payload, field)
                ),
            }
    checks = {
        "all_file_hashes_exact": all(
            row["matches"] for row in files.values()
        ),
        "payload_parse_deferred_or_exact": (
            not parse_payloads
            or all(
                row["matches"] and row["self_hash_valid"]
                for row in payloads.values()
            )
        ),
        "o4_operator_exact": files[str(OPTION_PATH)]["matches"],
        "o4_power_exact": files[str(POWER_PATH)]["matches"],
        "o4_v1_v2_protected_exact": all(
            files[str(path)]["matches"]
            for path in (
                O4_V1_MARKER_PATH,
                O4_V1_HOLD_PATH,
                O4_V2_MARKER_PATH,
                O4_V2_RESULT_PATH,
                O4_V2_STREAM_PATH,
            )
        ),
    }
    return {
        "files": files,
        "payloads": payloads,
        "source_payloads_parsed": bool(parse_payloads),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _required_family_target_counts() -> dict[tuple[str, int], int]:
    return {
        (family, target): sum(
            ROLE_FAMILY_TARGET_COUNTS[role][family_index][target_index]
            for role in ROLE_ORDER
        )
        for family_index, family in enumerate(FAMILY_ORDER)
        for target_index, target in enumerate(TARGET_ORDER)
    }


def source_pool_from_payloads(
    union: Mapping[str, Any],
    support: Mapping[str, Any],
    selected: Mapping[str, Any],
    reseal: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    membership = union.get("membership")
    candidate_rows = support.get("candidate_rows")
    selected_rows = selected.get("selected")
    if not isinstance(membership, list) or not isinstance(candidate_rows, list):
        raise SourceIntegrityError("O3 source manifests have invalid rows")
    if not isinstance(selected_rows, list):
        raise SourceIntegrityError("O3 selected manifest has invalid rows")

    union_index: dict[str, dict[str, Any]] = {}
    for raw in membership:
        if not isinstance(raw, dict):
            raise SourceIntegrityError("O3 union has a non-object row")
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

    compact_candidates: list[dict[str, Any]] = []
    candidate_keys: set[tuple[str, int]] = set()
    source_identity_failures: list[str] = []
    excluded_qd_rows = 0
    excluded_qd_roots: set[str] = set()
    for raw in candidate_rows:
        if not isinstance(raw, dict):
            raise SourceIntegrityError("O3 support has a non-object row")
        root = str(raw.get("root_cluster"))
        if root in selected_roots:
            continue
        union_row = union_index.get(root)
        if union_row is None:
            source_identity_failures.append(f"missing_union:{root}")
            continue
        family = str(raw.get("family"))
        if family == EXCLUDED_O3_FAMILY:
            excluded_qd_rows += 1
            excluded_qd_roots.add(root)
            continue
        mapped_family = O3_TO_O5.get(family)
        target = int(raw.get("target", -1))
        key = (root, target)
        if mapped_family is None or target not in TARGET_ORDER:
            source_identity_failures.append(f"bad_family_target:{root}")
            continue
        if key in candidate_keys:
            source_identity_failures.append(
                f"duplicate_root_target:{root}:T{target}"
            )
            continue
        candidate_keys.add(key)
        identity_matches = (
            str(union_row.get("family")) == family,
            str(union_row.get("source_replay"))
            == str(raw.get("source_replay")),
            str(union_row.get("source_replay_sha256"))
            == str(raw.get("source_replay_sha256")),
        )
        if not all(identity_matches):
            source_identity_failures.append(
                f"union_support_mismatch:{root}:T{target}"
            )
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
                "source_replay_sha256": str(
                    raw.get("source_replay_sha256")
                ),
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
    matrix = validate_frozen_matrices()
    required_family = matrix["combined_family_counts"]
    required_target = matrix["combined_target_counts"]
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
        "unselected_exact_20180": (
            len(set(union_index) - selected_roots) == O3_UNSELECTED_ROOTS
        ),
        "support_audit_passes": (
            support.get("audit", {}).get("passes") is True
        ),
        "support_candidate_manifest_exact": support.get(
            "candidate_manifest_sha256"
        )
        == canonical_json_hash(candidate_rows),
        "selected_scientific_checks_pass": (
            selected.get("passes") is True and not selected.get("deficits")
        ),
        "v3_reseal_exact": all(reseal_checks.values()),
        "zero_source_identity_failures": not source_identity_failures,
        "all_selected_roots_excluded": not any(
            row["root_cluster"] in selected_roots
            for row in compact_candidates
        ),
        "qd_explicitly_excluded": not any(
            row["source_family"] == EXCLUDED_O3_FAMILY
            for row in compact_candidates
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
        "qd_candidate_rows_excluded": excluded_qd_rows,
        "qd_candidate_roots_excluded": len(excluded_qd_roots),
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
    return report, compact_candidates


def load_source_pool() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = immutable_input_audit(parse_payloads=True)
    if not audit["passes"]:
        raise SourceIntegrityError("Immutable O3/O4 source identities changed")
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
            raise SourceIntegrityError(
                f"Candidate source drift within root: {root}"
            )
        rows_by_root[root] = row
    manifest: list[dict[str, Any]] = []
    failures: list[str] = []
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


def _find_frame(
    replay: Mapping[str, Any],
    frame_index: int,
) -> Mapping[str, Any]:
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
        raise SourceIntegrityError(f"O5 source board shape changed: {board.shape}")
    preview_payload = payload["preview"]
    if not isinstance(preview_payload, Mapping):
        raise SourceIntegrityError("O5 source preview is not an object")
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
        raise SourceIntegrityError("O5 source tile_cycle is not an object")
    raw_counts = cycle["small_counts"]
    if not isinstance(raw_counts, Mapping):
        raise SourceIntegrityError("O5 source small_counts is not an object")
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
    identity = {
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
    return state, identity


def restore_o5_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_root[str(row["root_cluster"])].append(row)
    restored: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    action_feature_rows = 0
    for root, rows in sorted(by_root.items()):
        first = rows[0]
        path = Path(str(first["source_replay"]))
        if sha256_path(path) != first["source_replay_sha256"]:
            raise SourceIntegrityError(
                f"O5 source changed before restore: {path}"
            )
        replay = json.loads(path.read_text())
        for row in sorted(rows, key=lambda item: int(item["target"])):
            try:
                frame = _find_frame(replay, int(row["frame_index"]))
                payload = frame.get("state")
                if not isinstance(payload, Mapping):
                    raise SourceIntegrityError("Support frame state is missing")
                state, identity = whitelisted_state_payload(payload)
                state_sha256 = canonical_json_hash(identity)
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
                legal = tuple(
                    int(action) for action in simulator.legal_actions(state)
                )
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
                        and np.all((0.0 <= tokens) & (tokens <= 1.0))
                        and np.all(
                            (0.0 <= global_values)
                            & (global_values <= 1.0)
                        )
                    ):
                        raise SourceIntegrityError(
                            "Nonfinite/out-of-domain O5 source features"
                        )
                    action_feature_rows += 1
                restored.append(
                    {
                        **dict(row),
                        "o5_whitelisted_state_sha256": state_sha256,
                        "pair": [
                            list(coordinate) for coordinate in pair.coordinates
                        ],
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
        "o5_eligible_rows": len(restored),
        "o5_eligible_roots": len(
            {row["root_cluster"] for row in restored}
        ),
        "action_feature_rows_verified": action_feature_rows,
        "failures": failures,
        "passes": not failures,
        "only_current_support_frames_read": True,
        "final_score_action_outcome_fields_read": False,
        "o3_option_training_bodies_read": False,
        "o3_selected_replay_bodies_read": False,
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
        f"O5-P0-cell-v1|{role}|{family}|{target}|"
        f"{root}|{frame}|{state_hash}"
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def allocate_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    used_roots: set[str] = set()
    deficits: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        matrix = ROLE_FAMILY_TARGET_COUNTS[role]
        for family_index, family in enumerate(FAMILY_ORDER):
            for target_index, target in enumerate(TARGET_ORDER):
                required = int(matrix[family_index][target_index])
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
                claimed: list[dict[str, Any]] = []
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

    actual = {
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
    matrix = validate_frozen_matrices()
    checks = {
        "zero_deficits": not deficits,
        "exact_448_roots": len(selected) == TOTAL_SELECTED_ROOTS,
        "one_state_per_root": len(used_roots) == len(selected),
        "role_counts_exact": dict(role_counts) == ROLE_COUNTS,
        "family_counts_exact": (
            dict(family_counts) == matrix["combined_family_counts"]
        ),
        "target_counts_exact": (
            dict(target_counts) == matrix["combined_target_counts"]
        ),
        "every_cell_exact": all(
            actual[role][family][f"T{target}"]
            == ROLE_FAMILY_TARGET_COUNTS[role][family_index][target_index]
            for role in ROLE_ORDER
            for family_index, family in enumerate(FAMILY_ORDER)
            for target_index, target in enumerate(TARGET_ORDER)
        ),
        "deterministic_no_backtracking": True,
        "family_share_exact_25_percent": all(
            family_counts[family] == 112 for family in FAMILY_ORDER
        ),
    }
    return {
        "selected": selected,
        "selected_manifest_sha256": canonical_json_hash(selected),
        "deficits": deficits,
        "role_counts": dict(role_counts),
        "family_counts": dict(family_counts),
        "target_counts": {
            f"T{key}": value for key, value in target_counts.items()
        },
        "role_family_target_counts": actual,
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
                "reason": (
                    "raw_unselected_support_upper_bound_cannot_fill_"
                    "frozen_matrix"
                ),
                "source_roots_opened": 0,
                "source_candidate_rows_opened": 0,
                "o5_eligible_rows": 0,
                "o5_eligible_roots": 0,
                "action_feature_rows_verified": 0,
                "failures": [],
                "passes": True,
                "o3_option_training_bodies_read": False,
                "o3_selected_replay_bodies_read": False,
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
    restored, report = restore_o5_candidates(candidates)
    if not report["passes"]:
        raise SourceIntegrityError("Permitted O5 source restoration failed")
    allocation = allocate_candidates(restored)
    allocation["allocation_attempted"] = True
    return report, allocation


def _single_arm_rows() -> list[dict[str, Any]]:
    bases = STREAM_BASES["learning"]
    rows: list[dict[str, Any]] = []
    for root_index in range(ROLE_COUNTS["train"]):
        for trajectory_index in range(TRAJECTORIES_PER_TRAIN_ROOT):
            code = (
                root_index * TRAJECTORIES_PER_TRAIN_ROOT + trajectory_index
            )
            round_index = (
                1
                + int(trajectory_index >= 2)
                + int(trajectory_index >= 4)
                + int(trajectory_index >= 5)
            )
            rows.append(
                {
                    "purpose": "learning",
                    "root_index": root_index,
                    "trajectory_index": trajectory_index,
                    "round_index": round_index,
                    **{
                        field: int(base) + code
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
    rows: list[dict[str, Any]] = []
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
                    "control_policy_stream_id": (
                        int(bases["policy_stream_id"]) + 2 * code
                    ),
                    "treatment_policy_stream_id": (
                        int(bases["policy_stream_id"]) + 2 * code + 1
                    ),
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
    policy_row_count = sum(
        2 if "control_policy_stream_id" in row else 1 for row in rows
    )
    checks = {
        "purpose_counts_exact": dict(purpose_counts) == expected_rows,
        "total_rows_exact_6272": len(rows) == 6_272,
        "no_acquisition_rows": "acquisition" not in purpose_counts,
        "learning_schedule_exact": all(
            row["round_index"]
            == (
                1
                + int(row["trajectory_index"] >= 2)
                + int(row["trajectory_index"] >= 4)
                + int(row["trajectory_index"] >= 5)
            )
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
                policy_row_count
                if field == "policy_stream_id"
                else len(rows)
            )
            for field, values in sets.items()
        ),
        "stream_bases_exact_181b_196b": STREAM_BASES
        == {
            "learning": {
                "logical_seed": 181_000_000_000,
                "deck_stream_id": 182_000_000_000,
                "slot_stream_id": 183_000_000_000,
                "policy_stream_id": 184_000_000_000,
            },
            "option": {
                "logical_seed": 185_000_000_000,
                "deck_stream_id": 186_000_000_000,
                "slot_stream_id": 187_000_000_000,
                "policy_stream_id": 188_000_000_000,
            },
            "normal_development": {
                "logical_seed": 189_000_000_000,
                "deck_stream_id": 190_000_000_000,
                "slot_stream_id": 191_000_000_000,
                "policy_stream_id": 192_000_000_000,
            },
            "confirmation": {
                "logical_seed": 193_000_000_000,
                "deck_stream_id": 194_000_000_000,
                "slot_stream_id": 195_000_000_000,
                "policy_stream_id": 196_000_000_000,
            },
        },
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
        "file_hash_exact": (
            sha256_path(O3_STREAM_MANIFEST_PATH)
            == IMMUTABLE_FILE_HASHES[str(O3_STREAM_MANIFEST_PATH)]
        ),
        "payload_hash_exact": (
            payload.get("payload_sha256")
            == IMMUTABLE_PAYLOAD_HASHES[str(O3_STREAM_MANIFEST_PATH)]
        ),
        "payload_self_hash_valid": _verify_self_hash(
            payload, "payload_sha256"
        ),
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


def o4_reservation_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(O4_V2_STREAM_PATH.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SourceIntegrityError("O4 reservation rows are missing")
    v1_marker = json.loads(O4_V1_MARKER_PATH.read_text())
    v2_marker = json.loads(O4_V2_MARKER_PATH.read_text())
    checks = {
        "stream_file_hash_exact": (
            sha256_path(O4_V2_STREAM_PATH)
            == IMMUTABLE_FILE_HASHES[str(O4_V2_STREAM_PATH)]
        ),
        "stream_payload_hash_exact": (
            payload.get("payload_sha256") == O4_STREAM_PAYLOAD_SHA256
        ),
        "stream_payload_self_hash_valid": _verify_self_hash(
            payload, "payload_sha256"
        ),
        "exact_6272_rows": len(rows) == 6_272,
        "stream_manifest_exact": (
            canonical_json_hash(rows) == O4_STREAM_MANIFEST_SHA256
        ),
        "v1_reservation_exact": (
            v1_marker.get("future_stream_manifest_sha256")
            == O4_STREAM_MANIFEST_SHA256
            and int(v1_marker.get("future_stream_row_count", -1)) == 6_272
        ),
        "v2_reservation_exact": (
            v2_marker.get("future_stream_manifest_sha256")
            == O4_STREAM_MANIFEST_SHA256
            and int(v2_marker.get("future_stream_row_count", -1)) == 6_272
        ),
    }
    return (
        {
            "path": str(O4_V2_STREAM_PATH),
            "file_sha256": sha256_path(O4_V2_STREAM_PATH),
            "payload_sha256": payload.get("payload_sha256"),
            "reservation_rows": len(rows),
            "reservation_manifest_sha256": canonical_json_hash(rows),
            "v1_marker_file_sha256": sha256_path(O4_V1_MARKER_PATH),
            "v2_marker_file_sha256": sha256_path(O4_V2_MARKER_PATH),
            "checks": checks,
            "passes": all(checks.values()),
            "streams_consumed_by_o5": 0,
        },
        [dict(row) for row in rows],
    )


def collision_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    scan_root: Path = Path("threes_rl/runs"),
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    requested = _stream_sets(rows)
    found: dict[str, set[int]] = defaultdict(set)
    matched: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
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

    o3_audit, o3_rows = o3_learning_stream_audit()
    o4_audit, o4_rows = o4_reservation_audit()
    o3_sets = _stream_sets(o3_rows)
    o4_sets = _stream_sets(o4_rows)
    for prior_sets in (o3_sets, o4_sets):
        for field, values in prior_sets.items():
            found[field].update(values)
    collisions: dict[str, list[int]] = {}
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
    o3_direct = {
        field: sorted(requested[field].intersection(o3_sets[field]))
        for field in STREAM_FIELDS
    }
    o4_direct = {
        field: sorted(requested[field].intersection(o4_sets[field]))
        for field in STREAM_FIELDS
    }
    checks = {
        "zero_historical_collisions": not any(collisions.values()),
        "exact_o3_learning_reservation_included": o3_audit["passes"],
        "zero_o3_learning_stream_collisions": not any(o3_direct.values()),
        "exact_o4_v1_v2_reservations_included": o4_audit["passes"],
        "zero_o4_reservation_collisions": not any(o4_direct.values()),
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
        "o4_reservation_audit": o4_audit,
        "o3_direct_collisions": o3_direct,
        "o4_direct_collisions": o4_direct,
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
    }


def policy_audit() -> dict[str, Any]:
    prior = json.loads(O3_POLICY_AUDIT_PATH.read_text())
    if not _verify_self_hash(prior, "payload_sha256"):
        raise SourceIntegrityError("O3 policy audit payload changed")
    prior_order = tuple(str(value) for value in prior["family_order"])
    retained_order = tuple(
        family for family in prior_order if family in INCLUDED_O3_FAMILIES
    )
    semantic_order = tuple(O3_TO_O5[family] for family in retained_order)
    signatures = {
        O3_TO_O5[family]: str(prior["signature_sha256"][family])
        for family in retained_order
    }
    pairwise = [
        {
            **row,
            "left": O3_TO_O5[str(row["left"])],
            "right": O3_TO_O5[str(row["right"])],
        }
        for row in prior["pairwise"]
        if str(row["left"]) in INCLUDED_O3_FAMILIES
        and str(row["right"]) in INCLUDED_O3_FAMILIES
    ]
    current_lock, _loaded = qd5._policy_lock()
    checks = {
        "prior_file_hash_exact": (
            sha256_path(O3_POLICY_AUDIT_PATH)
            == IMMUTABLE_FILE_HASHES[str(O3_POLICY_AUDIT_PATH)]
        ),
        "prior_payload_hash_exact": (
            prior["payload_sha256"]
            == IMMUTABLE_PAYLOAD_HASHES[str(O3_POLICY_AUDIT_PATH)]
        ),
        "prior_family_order_exact": prior_order == O3_FAMILY_ORDER,
        "semantic_family_order_exact": semantic_order == FAMILY_ORDER,
        "semantic_keys_exact": tuple(signatures.keys()) == FAMILY_ORDER,
        "four_unique_signatures": len(set(signatures.values())) == 4,
        "exact_six_pairwise_rows": len(pairwise) == 6,
        "all_pairwise_distinctness_gates_pass": all(
            row["passes"] for row in pairwise
        ),
        "current_policy_payload_exact": (
            current_lock["policy_lock_sha256"]
            == prior["policy_lock_sha256"]
        ),
        "qd_excluded_without_substitution": (
            EXCLUDED_O3_FAMILY not in retained_order
            and "o5_qd_v2" not in signatures
        ),
        "no_new_action_evaluation": True,
        "no_retiming": True,
    }
    return {
        "family_order": list(FAMILY_ORDER),
        "source_family_order": list(retained_order),
        "signatures": signatures,
        "pairwise": pairwise,
        "tie_state_counts": {
            O3_TO_O5[family]: int(prior["tie_state_counts"][family])
            for family in retained_order
        },
        "policy_lock_sha256": current_lock["policy_lock_sha256"],
        "excluded_family": EXCLUDED_O3_FAMILY,
        "checks": checks,
        "passes": all(checks.values()),
    }


def domain_proof() -> dict[str, Any]:
    proof = exhaustive_blocker_domain_proof()
    checks = {
        "o4_operator_source_exact": (
            sha256_path(OPTION_PATH)
            == IMMUTABLE_FILE_HASHES[str(OPTION_PATH)]
        ),
        "o4_power_source_exact": (
            sha256_path(POWER_PATH)
            == IMMUTABLE_FILE_HASHES[str(POWER_PATH)]
        ),
        "schema_sha_exact": schema_sha256() == O4_OPERATOR_SCHEMA_SHA256,
        "parameter_count_exact": parameter_count() == FROZEN_PARAMETER_COUNT,
        "coordinate_pairs_exact": proof["coordinate_pairs"] == 120,
        "occupancy_cases_exact": proof["occupancy_cases"] == 43_296,
        "density_minimum_zero": proof["minimum_density"] == 0.0,
        "density_maximum_one": proof["maximum_density"] == 1.0,
        "exhaustive_proof_passes": proof["passes"],
    }
    return {
        "schema_sha256": schema_sha256(),
        "parameter_count": parameter_count(),
        "proof": proof,
        "checks": checks,
        "passes": all(checks.values()),
        "representation_tuned": False,
        "outcomes_opened": False,
    }


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not _verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise SourceIntegrityError("O5 test evidence self hash mismatch")
    expected = {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "o4_operator_sha256": sha256_path(OPTION_PATH),
        "o4_power_sha256": sha256_path(POWER_PATH),
        "dependency_hashes": _dependency_hashes(),
    }
    if any(_normalize(payload.get(key)) != _normalize(value) for key, value in expected.items()):
        raise SourceIntegrityError("O5 test evidence bindings changed")
    if not payload.get("passes"):
        raise SourceIntegrityError("O5 tests did not pass")
    if any(
        int(payload.get(field, -1)) != 0
        for field in (
            "source_content_opened",
            "games",
            "streams_consumed",
            "labels",
            "models_fit",
            "policy_outcomes",
        )
    ):
        raise SourceIntegrityError("O5 test evidence is not zero-work")
    return payload


def seal_test_evidence(
    *,
    focused_passed: int,
    regression_passed: int,
    commands: Sequence[str],
) -> dict[str, Any]:
    if focused_passed <= 0 or regression_passed <= 0:
        raise ValueError("O5 test counts must be positive")
    input_audit = immutable_input_audit(parse_payloads=False)
    if not input_audit["passes"]:
        raise SourceIntegrityError("Immutable inputs changed before tests")
    return _write_immutable_json(
        TEST_EVIDENCE_PATH,
        {
            "version": f"{VERSION}_test_evidence",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "charter_sha256": sha256_path(CHARTER_PATH),
            "runner_sha256": sha256_path(RUNNER_PATH),
            "tests_sha256": sha256_path(TEST_PATH),
            "o4_operator_sha256": sha256_path(OPTION_PATH),
            "o4_power_sha256": sha256_path(POWER_PATH),
            "dependency_hashes": _dependency_hashes(),
            "focused_tests_passed": int(focused_passed),
            "regression_tests_passed": int(regression_passed),
            "commands": list(commands),
            "passes": True,
            "source_content_opened": 0,
            "games": 0,
            "streams_consumed": 0,
            "labels": 0,
            "models_fit": 0,
            "policy_outcomes": 0,
        },
        self_hash_field="test_evidence_payload_sha256",
    )


def _commands(out_dir: Path) -> dict[str, str]:
    base = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.o5_four_family_p0"
    )
    suffix = f" --out-dir {out_dir}'"
    return {
        "open": f"{base} open{suffix}",
        "run": f"{base} run{suffix}",
    }


def _bindings(out_dir: Path) -> dict[str, Any]:
    evidence = _load_test_evidence()
    rows = future_stream_rows()
    return _normalize(
        {
            **_current_bindings(),
            "bound_out_dir": str(out_dir.resolve()),
            "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
            "test_evidence_payload_sha256": evidence[
                "test_evidence_payload_sha256"
            ],
            "source_universe_roots": O3_ACQUISITION_ROOTS,
            "selected_o3_roots_excluded": O3_SELECTED_ROOTS,
            "unselected_source_universe": O3_UNSELECTED_ROOTS,
            "o5_root_count": TOTAL_SELECTED_ROOTS,
            "future_stream_manifest_sha256": canonical_json_hash(rows),
            "future_stream_row_count": len(rows),
            "o4_reservation_manifest_sha256": O4_STREAM_MANIFEST_SHA256,
            "commands": _commands(out_dir),
        }
    )


def open_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O5 P0 output directory is immutable")
    if out_dir.exists():
        raise FileExistsError(f"O5 P0 namespace exists: {out_dir}")
    evidence = _load_test_evidence()
    immutable = immutable_input_audit(parse_payloads=False)
    rows = future_stream_rows()
    stream = stream_contract(rows)
    matrix = validate_frozen_matrices()
    heavy = _heavy_process_audit()
    services = history.service_health()
    free_gib = shutil.disk_usage(out_dir.parent).free / 1024**3
    checks = {
        "test_evidence_exact": evidence["passes"],
        "immutable_file_hashes_exact": immutable["passes"],
        "source_payloads_not_parsed": not immutable[
            "source_payloads_parsed"
        ],
        "matrix_contract_exact": matrix["passes"],
        "stream_contract_exact": stream["passes"],
        "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
        "no_competing_heavy_process": heavy["passes"],
        "free_disk_above_120_gib": free_gib > TARGET_FREE_GIB,
        "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
        "services_dashboard_top_three": services["passes"],
        "zero_prior_o5_namespace": True,
        "zero_games_streams_labels_models_outcomes": True,
    }
    if not all(checks.values()):
        raise OperationalHold(f"O5 P0 open failed: {checks}")
    marker = {
        **_bindings(out_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "O5_P0_OPENED_ZERO_WORK",
        "preopen": {
            "heavy_process_audit": heavy,
            "service_health": services,
            "free_gib": free_gib,
            "nice": history.current_nice(),
            "immutable_file_audit": immutable,
            "matrix_contract": matrix,
            "stream_contract": stream,
        },
        "checks": checks,
        "zero_work": {
            "source_payloads_parsed": 0,
            "source_replay_bodies_opened": 0,
            "games": 0,
            "streams_consumed": 0,
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
    marker_path = out_dir / MARKER_NAME
    marker = json.loads(marker_path.read_text())
    if not _verify_self_hash(marker, "opened_payload_sha256"):
        raise SourceIntegrityError("O5 marker self hash mismatch")
    if marker.get("decision") != "O5_P0_OPENED_ZERO_WORK":
        raise SourceIntegrityError("O5 marker decision changed")
    expected = _bindings(out_dir)
    for key, value in expected.items():
        if _normalize(marker.get(key)) != _normalize(value):
            raise SourceIntegrityError(f"O5 marker binding mismatch: {key}")
    if any(int(value) != 0 for value in marker["zero_work"].values()):
        raise SourceIntegrityError("O5 marker is not zero-work")
    return marker


def _write_manifest(
    out_dir: Path,
    name: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return _write_immutable_json(
        out_dir / name,
        payload,
        self_hash_field="payload_sha256",
    )


def _decision(
    *,
    integrity_checks: Mapping[str, bool],
    support_checks: Mapping[str, bool],
) -> str:
    if not all(integrity_checks.values()):
        return "KILL_O5_FOUR_FAMILY_INTEGRITY_OR_REPRESENTATION"
    if not all(support_checks.values()):
        return "HOLD_O5_FOUR_FAMILY_DATA_SUPPORT"
    return "READY_O5_FOUR_FAMILY_DOMAIN_SAFE_PREFLIGHT"


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("O5 P0 output directory is immutable")
    result_path = out_dir / RESULT_NAME
    if result_path.exists():
        raise FileExistsError(f"O5 terminal result exists: {result_path}")
    marker = _load_marker(out_dir)
    try:
        immutable = immutable_input_audit(parse_payloads=True)
        if not immutable["passes"]:
            raise SourceIntegrityError("Immutable source identities changed")
        tests = _load_test_evidence()
        domain = domain_proof()
        source_report, source_candidates = load_source_pool()
        replay_sources = verify_candidate_source_replays(source_candidates)
        restore_report, allocation = support_and_allocation(
            source_report,
            source_candidates,
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
            "projected_runtime_frozen": (
                PROJECTED_ACTIVE_SECONDS == 18 * 3_600
            ),
            "projected_storage_below_4_gib": (
                PROJECTED_STORAGE_BYTES < STORAGE_CAP_BYTES
            ),
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
                "O5 training requires a separate frozen execution charter"
                if decision
                == "READY_O5_FOUR_FAMILY_DOMAIN_SAFE_PREFLIGHT"
                else "NONE"
            ),
            "hold": [
                "o5_training",
                "o5_mechanism_outcomes",
                "normal_start_development",
                "confirmation",
                "promotion",
            ],
            "kill": decision
            == "KILL_O5_FOUR_FAMILY_INTEGRITY_OR_REPRESENTATION",
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
                "qd_candidate_rows_excluded": source_report[
                    "qd_candidate_rows_excluded"
                ],
                "qd_candidate_roots_excluded": source_report[
                    "qd_candidate_roots_excluded"
                ],
                "candidate_root_upper_bounds_by_family": source_report[
                    "family_root_upper_bounds"
                ],
                "candidate_root_upper_bounds_by_family_target": source_report[
                    "family_target_root_upper_bounds"
                ],
                "allocation_count": len(allocation["selected"]),
                "allocation_manifest_sha256": allocation[
                    "selected_manifest_sha256"
                ],
                "role_counts": allocation.get("role_counts", {}),
                "family_counts": allocation.get("family_counts", {}),
                "target_counts": allocation.get("target_counts", {}),
                "role_family_target_counts": allocation.get(
                    "role_family_target_counts", {}
                ),
                "allocation_deficits": allocation["deficits"],
                "stream_rows_reserved": streams["row_count"],
                "stream_manifest_sha256": streams["manifest_sha256"],
                "o3_learning_rows_explicitly_checked": collision[
                    "o3_learning_stream_audit"
                ]["learning_rows"],
                "o4_reservation_rows_explicitly_checked": collision[
                    "o4_reservation_audit"
                ]["reservation_rows"],
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
                "rollouts": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    except SourceIntegrityError as error:
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "KILL_O5_FOUR_FAMILY_INTEGRITY_OR_REPRESENTATION",
            "continue": "NONE",
            "hold": ["all_o5_execution"],
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
                "rollouts": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    except OperationalHold as error:
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_O5_FOUR_FAMILY_DATA_SUPPORT",
            "continue": "NONE",
            "hold": ["all_o5_execution"],
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
                "rollouts": 0,
                "policy_outcomes": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    except Exception as error:
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "KILL_O5_FOUR_FAMILY_INTEGRITY_OR_REPRESENTATION",
            "continue": "NONE",
            "hold": ["all_o5_execution"],
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
                "rollouts": 0,
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
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    for command in ("open", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence = subparsers.add_parser("seal-test-evidence")
    evidence.add_argument("--focused-passed", type=int, required=True)
    evidence.add_argument("--regression-passed", type=int, required=True)
    evidence.add_argument(
        "--recorded-command",
        action="append",
        default=[],
    )
    args = parser.parse_args()
    if args.subcommand == "seal-test-evidence":
        result = seal_test_evidence(
            focused_passed=args.focused_passed,
            regression_passed=args.regression_passed,
            commands=args.recorded_command,
        )
    elif args.subcommand == "open":
        result = open_preflight(args.out_dir)
    else:
        result = run_preflight(args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

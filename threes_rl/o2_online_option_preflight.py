"""Outcome-free design, power, stream, and service preflight for O2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from threes_rl import g1r_acquire as history
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.o1_geometry_option import (
    O1OptionNet,
    canonical_json_hash,
    schema_sha256,
)


VERSION = "o2_online_scale_relative_option_preflight_v1_a4"
ROOT = Path("threes_rl/runs")
OUTPUT_DIR = ROOT / "forensics/o2_online_option_preflight_v1"
CHARTER_PATH = Path(
    "threes_rl/O2_ONLINE_SCALE_RELATIVE_OPTION_CURRICULUM_CHARTER.md"
)
A1_PATH = Path(
    "threes_rl/"
    "O2_ONLINE_SCALE_RELATIVE_OPTION_CURRICULUM_CHARTER_AMENDMENT_A1.md"
)
A2_PATH = Path(
    "threes_rl/"
    "O2_ONLINE_SCALE_RELATIVE_OPTION_CURRICULUM_CHARTER_AMENDMENT_A2.md"
)
A3_PATH = Path(
    "threes_rl/"
    "O2_ONLINE_SCALE_RELATIVE_OPTION_CURRICULUM_CHARTER_AMENDMENT_A3.md"
)
A4_PATH = Path(
    "threes_rl/"
    "O2_ONLINE_SCALE_RELATIVE_OPTION_CURRICULUM_CHARTER_AMENDMENT_A4.md"
)
RUNNER_PATH = Path("threes_rl/o2_online_option_preflight.py")
TEST_PATH = Path("tests/test_rl_o2_online_option_preflight.py")
O1_GEOMETRY_PATH = Path("threes_rl/o1_geometry_option.py")
PILOT_V2_LOCK_PATH = (
    ROOT / "forensics/g1r_acquisition/pilot_v2_qd5/preflight_lock.json"
)
G3_COST_PATH = (
    ROOT
    / "forensics/g3_scale_transfer_bootstrap_preflight_v2/"
    "G3_V2_BOOTSTRAP_PREFLIGHT.json"
)
TEST_EVIDENCE_PATH = (
    ROOT / "forensics/o2_online_option_preflight_test_evidence.json"
)
STREAM_MANIFEST_PATH = OUTPUT_DIR / "O2_STREAM_MANIFEST.json"
DESIGN_MANIFEST_PATH = OUTPUT_DIR / "O2_DESIGN_MANIFEST.json"
COLLISION_MANIFEST_PATH = OUTPUT_DIR / "O2_COLLISION_SOURCE_MANIFEST.json"
CALIBRATION_PATH = OUTPUT_DIR / "O2_HISTORICAL_CALIBRATION.json"
POWER_PATH = OUTPUT_DIR / "O2_POWER_MDE.json"
RESULT_PATH = OUTPUT_DIR / "O2_PREFLIGHT_RESULT.json"

CHARTER_SHA256 = (
    "865f44c526d1859899b532e87dc4b99d031aa487705845b2c5412523b1997e12"
)
A1_SHA256 = (
    "79423c0fdce09cdb3b69b4e3bed4ddd5ec4d2dc7051cd2b3567f92c48e5e40af"
)
A2_SHA256 = (
    "610e564815f6c51244d474034a3f58865e30fefb3703bbc3e062b276984bb9fb"
)
A3_SHA256 = (
    "f19e2c9a458621722f722e84d5f51ff57804f48592c2580efdb4d1525d8472fe"
)
A4_SHA256 = (
    "1095462bbef0759ce8c56573727a645d3859469f5b50e1231e907c3cd8a479b3"
)
O1_GEOMETRY_SHA256 = (
    "c9ff99e49da6cd9a54a01ae6c402bd1634c7905cb898dae94a0ac4c554342f87"
)
O1_SCHEMA_SHA256 = (
    "55dd298ea2bf40a24d8af641d852d5f9c09aff14b1b736a29e6b5a071563772c"
)
O1_PARAMETER_COUNT = 113_780
PILOT_V2_LOCK_SHA256 = (
    "0d50edaae52e9a6f6291c4b397fd03c9d7d8651b28bb9dbd05b53c8718ee22ad"
)
PILOT_V2_ACTION_AUDIT_SHA256 = (
    "cc747bead64edfd3820f4547bc629e764339f3917e4e0a62ca71ba0979d0635d"
)
PILOT_V2_PANEL_SHA256 = (
    "b8862aa3c8eaf6278fc078fb3e03aa7222a01930673cfee497738c74e81eff9d"
)
PILOT_V2_POLICY_LOCK_SHA256 = (
    "6b0384d9fedfc8f560853a050c28750194ec9c9d3d36cf2d9d7fd47a9a423ea0"
)
G3_COST_SHA256 = (
    "052985f7e5c13797df43bfd074602169ff5c85618dd0f3db549720fec95f7d66"
)

FAMILIES = (
    (
        "o2_corner2",
        "g1r_corner2",
        "corner2",
        "4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043",
    ),
    (
        "o2_expectimax2",
        "g1r_expectimax2",
        "expectimax2",
        "2ad642cdca7739cc73af4f570de5054c422815f9a7d8f93a2619921b46b74b38",
    ),
    (
        "o2_parent_mc1000",
        "g1r_parent_mc1000",
        (
            "ntuple_expectimax2:"
            "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_"
            "20260706/latest"
        ),
        "e43dc11f3220557d7f9aef228db96dc6f06f49b26300d5a4128ea00bf8ba2064",
    ),
    (
        "o2_qd_v2",
        "g1r_qd_static_archive_oneply_v2_terminal_schema",
        (
            "g1r_qd_static_archive_oneply_v2_terminal_schema:"
            "threes_rl/runs/forensics/"
            "g1r_qd_admission_v2_terminal_schema/policy"
        ),
        "66da7d61c9178bd982cac492e397a0cfc4424d51f396fe1b290c4af7fe1cd281",
    ),
)

STREAM_BASES = {
    "pilot": (81_000_000_000, 82_000_000_000, 83_000_000_000, 84_000_000_000),
    "corpus": (85_000_000_000, 86_000_000_000, 87_000_000_000, 88_000_000_000),
    "learning": (
        89_000_000_000,
        90_000_000_000,
        91_000_000_000,
        92_000_000_000,
    ),
    "mechanism": (
        93_000_000_000,
        94_000_000_000,
        95_000_000_000,
        96_000_000_000,
    ),
    "normal_development": (
        97_000_000_000,
        98_000_000_000,
        99_000_000_000,
        100_000_000_000,
    ),
    "confirmation": (
        101_000_000_000,
        102_000_000_000,
        103_000_000_000,
        104_000_000_000,
    ),
}
STREAM_FIELDS = (
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
)

LOWER_TARGETS = (48, 96, 192, 384)
TRANSFER_TARGET = 768
STARTING_STAGES = (0, 1, 2, 3)
PILOT_ROOTS_PER_FAMILY = 32
CORPUS_ROOTS_PER_FAMILY = 160
TRAIN_ROOTS = 128
DEVELOPMENT_ROOTS = 48
TEST_ROOTS = 192
NORMAL_DEVELOPMENT_ROOTS = 384
CONFIRMATION_ROOTS = 2_560
MECHANISM_REPLICATES = 8

POWER_OR_GRID = (1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00)
POWER_REQUIRED = 0.80
POWER_POINT_FLOOR = 1.25
POWER_BOOTSTRAPS = 199
MECHANISM_POWER_DESIGNS = 1_024
CAPABILITY_POWER_DESIGNS = 768
MECHANISM_POWER_SEED = 2_026_072_620
CAPABILITY_POWER_SEED = 2_026_072_621
MECHANISM_ALPHA = 1.6
MECHANISM_BETA = 18.4
MECHANISM_COUPLING = 0.50
STAGE_FACTORS = (0.50, 0.75, 1.00, 1.50)
TARGET_FACTORS = {48: 1.30, 96: 1.15, 192: 1.00, 384: 0.85, 768: 0.70}

MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
EXPECTED_TOP_THREE = (263670, 261369, 258561)
ACQUISITION_GAME_SECONDS = 17_655.126695 / 1_920
H40_PATH_SECONDS = 42_820.82137607305 / 5_072
MAX_REPLAY_BYTES = 1_000_401
MIB = 1024**2
GIB = 1024**3

HISTORICAL_PAIRS = (
    (
        "D0",
        Path(
            "threes_rl/runs/eval_artifacts/"
            "r1_baseline_incumbent_split_v1_d0_20260709/results.csv"
        ),
        "240fca445d79b6f546e9dab5b62bfa4ae9531d1d31cc727c59bc04495376f4bb",
        Path(
            "threes_rl/runs/eval_artifacts/"
            "r1_candidate_1000_split_v1_d0_20260709/results.csv"
        ),
        "4c4db06b457b44155e11e5dd5476980f010f0fd65b9c545f561598bfd34a5d48",
    ),
    (
        "D1",
        Path(
            "threes_rl/runs/eval_artifacts/"
            "r1_baseline_incumbent_split_v1_d1_20260709/results.csv"
        ),
        "0226d60858f052597399c6ba3ed804cedc4769e09a393df201ef4fcdc6d17491",
        Path(
            "threes_rl/runs/eval_artifacts/"
            "r1b_candidate_1000_split_v1_d1_20260709/results.csv"
        ),
        "88b4b0151c6280dca30de4ed74ac4e2d22840b17565c02d66e16fe54dfcbd3d0",
    ),
    (
        "D2",
        Path(
            "threes_rl/runs/eval_artifacts/"
            "r1b_baseline_incumbent_split_v1_d2_20260709/results.csv"
        ),
        "d04a68d8f486f1b4c965e220427b4f753a828ee40ad4bdf746bb73893e161f8f",
        Path(
            "threes_rl/runs/eval_artifacts/"
            "r1b_candidate_5000_split_v1_d2_20260709/results.csv"
        ),
        "a81bfa2484218c09626ab65f303bef6f030cfe4b6e87995157eae700cc2652d8",
    ),
    (
        "C",
        Path(
            "threes_rl/runs/eval_artifacts/"
            "r1b_confirmation_incumbent_c_20260710/results.csv"
        ),
        "474356da0bc5847acbba6723fcdc2e477c64d9894c75a47d5bd10f673a285383",
        Path(
            "threes_rl/runs/eval_artifacts/"
            "r1b_confirmation_candidate_5000_c_20260710/results.csv"
        ),
        "9c34ce8daa3926cff294bed0b094ced5421f458cdc5dfdc6ac16125924c1e564",
    ),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("canonical_payload_sha256", None)
    result["canonical_payload_sha256"] = canonical_json_hash(result)
    return result


def verify_payload_hash(payload: Mapping[str, Any]) -> bool:
    body = dict(payload)
    expected = body.pop("canonical_payload_sha256", None)
    return isinstance(expected, str) and canonical_json_hash(body) == expected


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable O2 artifact: {path}")
    value = payload_with_hash(payload)
    serialized = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )
    if not verify_payload_hash(json.loads(serialized)):
        raise ValueError("O2 JSON failed pre-write payload verification")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    if not verify_payload_hash(json.loads(path.read_text())):
        raise ValueError("O2 JSON failed post-write payload verification")


def artifact_identity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "canonical_payload_sha256": payload.get("canonical_payload_sha256"),
        "payload_valid": verify_payload_hash(payload),
        "bytes": int(path.stat().st_size),
    }


def _validate_self_hashed_json(path: Path, field: str) -> bool:
    payload = json.loads(path.read_text())
    expected = payload.pop(field, None)
    return isinstance(expected, str) and canonical_json_hash(payload) == expected


def bound_document_audit() -> dict[str, Any]:
    expected = {
        CHARTER_PATH: CHARTER_SHA256,
        A1_PATH: A1_SHA256,
        A2_PATH: A2_SHA256,
        A3_PATH: A3_SHA256,
        A4_PATH: A4_SHA256,
        O1_GEOMETRY_PATH: O1_GEOMETRY_SHA256,
        PILOT_V2_LOCK_PATH: PILOT_V2_LOCK_SHA256,
        G3_COST_PATH: G3_COST_SHA256,
    }
    rows = [
        {
            "path": str(path),
            "expected_sha256": digest,
            "actual_sha256": sha256_path(path),
            "passes": sha256_path(path) == digest,
        }
        for path, digest in expected.items()
    ]
    model = O1OptionNet()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checks = {
        "all_bound_files_exact": all(row["passes"] for row in rows),
        "schema_exact": schema_sha256() == O1_SCHEMA_SHA256,
        "parameter_count_exact": parameter_count == O1_PARAMETER_COUNT,
    }
    return {
        "rows": rows,
        "schema_sha256": schema_sha256(),
        "parameter_count": parameter_count,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _current_policy_manifest_matches(policy_lock: Mapping[str, Any]) -> bool:
    for path_text, expected in policy_lock["source_hashes"].items():
        path = Path(path_text)
        if not path.is_file() or sha256_path(path) != expected:
            return False
    for family in policy_lock["families"]:
        for manifest in family.get("checkpoint_manifests", []):
            for row in manifest["files"]:
                path = Path(row["path"])
                if (
                    not path.is_file()
                    or int(path.stat().st_size) != int(row["bytes"])
                    or sha256_path(path) != row["sha256"]
                ):
                    return False
        bundle = family.get("qd_policy_bundle")
        if bundle:
            for row in bundle["files"]:
                path = Path(bundle["path"]) / row["relative_path"]
                if (
                    not path.is_file()
                    or int(path.stat().st_size) != int(row["bytes"])
                    or sha256_path(path) != row["sha256"]
                ):
                    return False
    for row in policy_lock.get("qd_static_payloads", {}).values():
        path = Path(row["path"])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["bytes"])
            or sha256_path(path) != row["sha256"]
        ):
            return False
    return True


def family_evidence_audit() -> dict[str, Any]:
    lock = json.loads(PILOT_V2_LOCK_PATH.read_text())
    body = dict(lock)
    payload_sha = body.pop("preflight_payload_sha256", None)
    payload_valid = canonical_json_hash(body) == payload_sha
    action = lock["action_signature_audit"]
    policy_lock = lock["policy_lock"]
    expected_signatures = {
        historical_family: signature
        for _o2_family, historical_family, _spec, signature in FAMILIES
    }
    family_by_name = {
        row["family"]: row for row in policy_lock["families"]
    }
    exact_specs = all(
        historical in family_by_name
        and family_by_name[historical]["policy_spec"] == spec
        for _o2, historical, spec, _signature in FAMILIES
    )
    pair_rows = [
        row
        for row in action["pairwise"]
        if row["left"] in expected_signatures and row["right"] in expected_signatures
    ]
    checks = {
        "lock_file_exact": sha256_path(PILOT_V2_LOCK_PATH)
        == PILOT_V2_LOCK_SHA256,
        "lock_payload_valid": payload_valid,
        "pilot_ready": lock.get("decision")
        == "READY_G1R_PILOT_V2_QD5_PREFLIGHT",
        "action_audit_exact": action["audit_sha256"]
        == PILOT_V2_ACTION_AUDIT_SHA256,
        "panel_exact": action["panel_sha256"] == PILOT_V2_PANEL_SHA256,
        "signatures_exact": {
            name: action["signature_sha256"].get(name)
            for name in expected_signatures
        }
        == expected_signatures,
        "all_six_subset_pairs_pass": len(pair_rows) == 6
        and all(bool(row["passes"]) for row in pair_rows),
        "policy_lock_exact": policy_lock["policy_lock_sha256"]
        == PILOT_V2_POLICY_LOCK_SHA256,
        "policy_specs_exact": exact_specs,
        "current_policy_artifacts_exact": _current_policy_manifest_matches(
            policy_lock
        ),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "pilot_lock_path": str(PILOT_V2_LOCK_PATH),
        "pilot_lock_file_sha256": sha256_path(PILOT_V2_LOCK_PATH),
        "pilot_lock_payload_sha256": payload_sha,
        "panel_sha256": action["panel_sha256"],
        "action_audit_sha256": action["audit_sha256"],
        "policy_lock_sha256": policy_lock["policy_lock_sha256"],
        "family_order": [row[0] for row in FAMILIES],
        "historical_family_order": [row[1] for row in FAMILIES],
        "signatures": expected_signatures,
        "pairwise": pair_rows,
    }


def wilson_lower(k: int, n: int, z: float = 1.6448536269514722) -> float:
    if not 0 <= k <= n or n <= 0:
        raise ValueError("Invalid Wilson count")
    probability = k / n
    denominator = 1.0 + z * z / n
    return (
        probability
        + z * z / (2.0 * n)
        - z
        * math.sqrt(
            probability * (1.0 - probability) / n
            + z * z / (4.0 * n * n)
        )
    ) / denominator


def design_manifest() -> dict[str, Any]:
    pilot_structural_cells = []
    pilot_availability_cells = []
    for target in (768, 384, 192, 96, 48):
        for stage in STARTING_STAGES:
            transfer = target == 768
            pilot_structural_cells.append(
                {
                    "starting_stage": stage,
                    "target": target,
                    "quota": 7 if transfer else 4,
                    "minimum_families": 3,
                    "maximum_per_family": 3 if transfer else 2,
                    "root_disjoint": True,
                    "wilson_yield_claim": False,
                }
            )
            availability_minimum = 8 if transfer else 7
            final_demand = 20 if transfer else 18
            pilot_availability_cells.append(
                {
                    "starting_stage": stage,
                    "target": target,
                    "minimum_distinct_whole_roots": availability_minimum,
                    "minimum_families": 3,
                    "maximum_per_family": 3,
                    "root_may_support_other_cells": True,
                    "root_counted_at_most_once_in_this_cell": True,
                    "final_disjoint_demand": final_demand,
                    "final_rate_required": final_demand / 640,
                    "wilson_lower": wilson_lower(
                        availability_minimum, 128
                    ),
                }
            )

    train_cells = [
        {
            "starting_stage": stage,
            "target": target,
            "quota": 8,
            "minimum_families": 3,
            "maximum_per_family": 3,
        }
        for target in LOWER_TARGETS
        for stage in STARTING_STAGES
    ]
    development_cells = [
        {
            "starting_stage": stage,
            "target": target,
            "quota": 2,
        }
        for target in LOWER_TARGETS
        for stage in STARTING_STAGES
    ] + [
        {
            "starting_stage": stage,
            "target": 768,
            "quota": 4,
            "minimum_families": 3,
            "maximum_per_family": 2,
        }
        for stage in STARTING_STAGES
    ]
    test_cells = [
        {
            "starting_stage": stage,
            "target": target,
            "quota": 8,
            "minimum_families": 3,
            "maximum_per_family": 3,
        }
        for target in LOWER_TARGETS
        for stage in STARTING_STAGES
    ] + [
        {
            "starting_stage": stage,
            "target": 768,
            "quota": 16,
            "minimum_families": 3,
            "maximum_per_family": 6,
        }
        for stage in STARTING_STAGES
    ]
    prospective_roots = [
        {
            "family_index": family_index,
            "family": family[0],
            "family_game_index": game_index,
            "prospective_root_id": (
                f"o2-corpus:{family_index}:{game_index}"
            ),
        }
        for family_index, family in enumerate(FAMILIES)
        for game_index in range(CORPUS_ROOTS_PER_FAMILY)
    ]
    checks = {
        "pilot_structural_20_cells": len(pilot_structural_cells) == 20,
        "pilot_structural_92_unique_matching_slots": sum(
            row["quota"] for row in pilot_structural_cells
        )
        == 92,
        "pilot_structural_quotas_exact": Counter(
            (
                "transfer" if row["target"] == TRANSFER_TARGET else "lower",
                row["quota"],
            )
            for row in pilot_structural_cells
        )
        == {("lower", 4): 16, ("transfer", 7): 4},
        "pilot_availability_20_cells": len(pilot_availability_cells) == 20,
        "pilot_availability_minima_exact": Counter(
            (
                "transfer" if row["target"] == TRANSFER_TARGET else "lower",
                row["minimum_distinct_whole_roots"],
            )
            for row in pilot_availability_cells
        )
        == {("lower", 7): 16, ("transfer", 8): 4},
        "pilot_final_demands_exact": Counter(
            (
                "transfer" if row["target"] == TRANSFER_TARGET else "lower",
                row["final_disjoint_demand"],
            )
            for row in pilot_availability_cells
        )
        == {("lower", 18): 16, ("transfer", 20): 4},
        "pilot_availability_wilson_passes": all(
            row["wilson_lower"] > row["final_rate_required"]
            for row in pilot_availability_cells
        ),
        "pilot_fully_disjoint_wilson_slots_144": (
            16 * 7 + 4 * 8 == 144
        ),
        "pilot_fully_disjoint_wilson_interpretation_rejected": (
            16 * 7 + 4 * 8 > PILOT_ROOTS_PER_FAMILY * len(FAMILIES)
        ),
        "training_128_lower_only": sum(row["quota"] for row in train_cells)
        == TRAIN_ROOTS
        and {row["target"] for row in train_cells} == set(LOWER_TARGETS),
        "development_48": sum(row["quota"] for row in development_cells)
        == DEVELOPMENT_ROOTS,
        "development_16_transfer": sum(
            row["quota"] for row in development_cells if row["target"] == 768
        )
        == 16,
        "test_192": sum(row["quota"] for row in test_cells) == TEST_ROOTS,
        "test_64_transfer": sum(
            row["quota"] for row in test_cells if row["target"] == 768
        )
        == 64,
        "prospective_640_equal_families": len(prospective_roots) == 640
        and Counter(row["family"] for row in prospective_roots)
        == {family[0]: 160 for family in FAMILIES},
        "no_required_1536": all(
            row["target"] != 1536
            for row in train_cells + development_cells + test_cells
        ),
    }
    return {
        "version": "o2_online_option_design_manifest_v1_a4",
        "family_order": [row[0] for row in FAMILIES],
        "pilot_roots_per_family": PILOT_ROOTS_PER_FAMILY,
        "corpus_roots_per_family": CORPUS_ROOTS_PER_FAMILY,
        "pilot_contract": {
            "complete_unconditionally_retained_roots": 128,
            "fully_disjoint_wilson_slots": 144,
            "fully_disjoint_wilson_interpretation_allowed": False,
            "decision_requires_both_layers": True,
        },
        "pilot_structural_cells": pilot_structural_cells,
        "pilot_structural_cell_manifest_sha256": canonical_json_hash(
            pilot_structural_cells
        ),
        "pilot_availability_cells": pilot_availability_cells,
        "pilot_availability_cell_manifest_sha256": canonical_json_hash(
            pilot_availability_cells
        ),
        "train_cells": train_cells,
        "development_cells": development_cells,
        "untouched_test_cells": test_cells,
        "prospective_corpus_roots": prospective_roots,
        "prospective_root_manifest_sha256": canonical_json_hash(
            prospective_roots
        ),
        "state_selector": (
            'argmin SHA256("O2-state-v1"|partition|family|root|stage|'
            "target|frame|state_hash), then frame, then state_hash"
        ),
        "descriptive_1536_panel": {
            "maximum_roots": 16,
            "maximum_per_stage": 4,
            "minimum_roots": 0,
            "readiness_gate": False,
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def _stream_row(
    *,
    purpose: str,
    code: int,
    arm: str,
    paired: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    logical, deck, slot, policy = STREAM_BASES[purpose]
    arm_index = {"control": 0, "treatment": 1}.get(arm, 0)
    return {
        "purpose": purpose,
        "trajectory_code": code,
        "arm": arm,
        "paired": paired,
        "logical_seed": logical + code,
        "deck_stream_id": deck + code,
        "slot_stream_id": slot + code,
        "policy_stream_id": (
            policy + 2 * code + arm_index if paired else policy + code
        ),
        **dict(metadata),
    }


def stream_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for game_index in range(PILOT_ROOTS_PER_FAMILY):
            code = family_index * PILOT_ROOTS_PER_FAMILY + game_index
            rows.append(
                _stream_row(
                    purpose="pilot",
                    code=code,
                    arm="collector",
                    paired=False,
                    metadata={
                        "family": family[0],
                        "family_index": family_index,
                        "family_game_index": game_index,
                    },
                )
            )
        for game_index in range(CORPUS_ROOTS_PER_FAMILY):
            code = family_index * CORPUS_ROOTS_PER_FAMILY + game_index
            rows.append(
                _stream_row(
                    purpose="corpus",
                    code=code,
                    arm="collector",
                    paired=False,
                    metadata={
                        "family": family[0],
                        "family_index": family_index,
                        "family_game_index": game_index,
                    },
                )
            )

    for root_index in range(TRAIN_ROOTS):
        for round_index in range(4):
            for replicate in range(2):
                code = (
                    2_000_000
                    + root_index * 8
                    + round_index * 2
                    + replicate
                )
                rows.append(
                    _stream_row(
                        purpose="learning",
                        code=code,
                        arm="treatment",
                        paired=False,
                        metadata={
                            "partition": "train",
                            "root_index": root_index,
                            "round_index": round_index,
                            "replicate": replicate,
                        },
                    )
                )

    for partition, roots, offset in (
        ("development", DEVELOPMENT_ROOTS, 3_000_000),
        ("untouched_test", TEST_ROOTS, 4_000_000),
    ):
        for root_index in range(roots):
            for replicate in range(MECHANISM_REPLICATES):
                code = offset + root_index * 8 + replicate
                for arm in ("control", "treatment"):
                    rows.append(
                        _stream_row(
                            purpose="mechanism",
                            code=code,
                            arm=arm,
                            paired=True,
                            metadata={
                                "partition": partition,
                                "root_index": root_index,
                                "replicate": replicate,
                            },
                        )
                    )

    for purpose, roots, offset in (
        ("normal_development", NORMAL_DEVELOPMENT_ROOTS, 5_000_000),
        ("confirmation", CONFIRMATION_ROOTS, 6_000_000),
    ):
        for root_index in range(roots):
            code = offset + root_index
            for arm in ("control", "treatment"):
                rows.append(
                    _stream_row(
                        purpose=purpose,
                        code=code,
                        arm=arm,
                        paired=True,
                        metadata={"root_index": root_index},
                    )
                )
    return rows


def internal_stream_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["purpose"]) for row in rows)
    expected_counts = {
        "pilot": 128,
        "corpus": 640,
        "learning": 1_024,
        "mechanism": 768 + 3_072,
        "normal_development": 768,
        "confirmation": 5_120,
    }
    policy_ids = [int(row["policy_stream_id"]) for row in rows]
    pair_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_groups[(str(row["purpose"]), int(row["trajectory_code"]))].append(
            row
        )
    pair_checks = []
    for (_purpose, _code), group in pair_groups.items():
        paired = bool(group[0]["paired"])
        if paired:
            pair_checks.append(
                len(group) == 2
                and {row["arm"] for row in group} == {"control", "treatment"}
                and len({row["logical_seed"] for row in group}) == 1
                and len({row["deck_stream_id"] for row in group}) == 1
                and len({row["slot_stream_id"] for row in group}) == 1
                and len({row["policy_stream_id"] for row in group}) == 2
            )
        else:
            pair_checks.append(len(group) == 1)
    tape_keys = [
        (
            str(row["purpose"]),
            int(row["trajectory_code"]),
            int(row["logical_seed"]),
            int(row["deck_stream_id"]),
            int(row["slot_stream_id"]),
        )
        for row in rows
    ]
    unique_codes = {
        (str(row["purpose"]), int(row["trajectory_code"])) for row in rows
    }
    unique_tapes = set(tape_keys)
    checks = {
        "exact_purpose_counts": dict(sorted(counts.items()))
        == dict(sorted(expected_counts.items())),
        "policy_ids_globally_unique": len(policy_ids) == len(set(policy_ids)),
        "paired_crn_exact": all(pair_checks),
        "one_tape_per_trajectory_code": len(unique_tapes) == len(unique_codes),
        "all_requested_ids_positive": all(
            int(row[field]) > 0 for row in rows for field in STREAM_FIELDS
        ),
    }
    return {
        "row_count": len(rows),
        "purpose_counts": dict(sorted(counts.items())),
        "trajectory_tape_count": len(unique_tapes),
        "checks": checks,
        "passes": all(checks.values()),
    }


def historical_collision_audit(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prior, sources = history.historical_collision_union(exclude_dir=OUTPUT_DIR)
    collisions: dict[str, list[int]] = {}
    for field in STREAM_FIELDS:
        requested = {int(row[field]) for row in rows}
        historical = set(prior.get(field, set()))
        if field == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                historical.update(prior.get(alias, set()))
        collisions[field] = sorted(requested.intersection(historical))
    audit = {
        "version": "o2_historical_stream_collision_audit_v1_a4",
        "requested_row_count": len(rows),
        "requested_stream_set_sha256": canonical_json_hash(
            {
                field: sorted({int(row[field]) for row in rows})
                for field in STREAM_FIELDS
            }
        ),
        "collisions": collisions,
        "zero_historical_collisions": not any(collisions.values()),
        "historical_matched_source_count": sources["matched_source_count"],
        "historical_source_manifest_sha256": sources["matched_sources_sha256"],
        "historical_value_counts": sources["value_counts"],
    }
    return audit, {
        "version": "o2_collision_source_manifest_v1_a4",
        **sources,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _pair_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        row[field]
        for field in (
            "block",
            "index",
            "logical_seed",
            "deck_stream_id",
            "slot_stream_id",
        )
    )


def _p3072(row: Mapping[str, str]) -> bool:
    return int(row["max_tile_excl_starter"]) >= 3072


def historical_calibration() -> dict[str, Any]:
    groups: dict[str, list[tuple[dict[str, str], dict[str, str]]]] = {}
    source_rows = []
    for name, control_path, control_sha, treatment_path, treatment_sha in (
        HISTORICAL_PAIRS
    ):
        actual_control = sha256_path(control_path)
        actual_treatment = sha256_path(treatment_path)
        control = _read_rows(control_path)
        treatment = _read_rows(treatment_path)
        if len(control) != len(treatment):
            raise ValueError(f"Historical pair length mismatch: {name}")
        paired = []
        for control_row, treatment_row in zip(control, treatment):
            if _pair_key(control_row) != _pair_key(treatment_row):
                raise ValueError(f"Historical pair identity mismatch: {name}")
            paired.append((control_row, treatment_row))
        groups[name] = paired
        source_rows.append(
            {
                "name": name,
                "control_path": str(control_path),
                "control_expected_sha256": control_sha,
                "control_actual_sha256": actual_control,
                "treatment_path": str(treatment_path),
                "treatment_expected_sha256": treatment_sha,
                "treatment_actual_sha256": actual_treatment,
                "rows": len(paired),
                "hashes_pass": actual_control == control_sha
                and actual_treatment == treatment_sha,
            }
        )

    def summarize(
        pairs: Iterable[tuple[Mapping[str, str], Mapping[str, str]]],
    ) -> dict[str, Any]:
        values = list(pairs)
        control_success = sum(_p3072(control) for control, _treatment in values)
        treatment_success = sum(
            _p3072(treatment) for _control, treatment in values
        )
        both = sum(
            _p3072(control) and _p3072(treatment)
            for control, treatment in values
        )
        gains = sum(
            not _p3072(control) and _p3072(treatment)
            for control, treatment in values
        )
        losses = sum(
            _p3072(control) and not _p3072(treatment)
            for control, treatment in values
        )
        p0 = control_success / len(values)
        p1 = treatment_success / len(values)
        coupling = (both / len(values) - p0 * p1) / (p0 * (1.0 - p1))
        log_differences = [
            math.log1p(max(int(treatment["score_minus_starter"]), 0))
            - math.log1p(max(int(control["score_minus_starter"]), 0))
            for control, treatment in values
        ]
        block_counts = []
        for block_index in range(8):
            block = [
                (control, treatment)
                for control, treatment in values
                if int(control["logical_seed"]) % 8 == block_index
            ]
            block_counts.append(
                {
                    "stream_stratum": block_index,
                    "roots": len(block),
                    "control_p3072": sum(
                        _p3072(control) for control, _treatment in block
                    ),
                    "treatment_p3072": sum(
                        _p3072(treatment) for _control, treatment in block
                    ),
                }
            )
        return {
            "pairs": len(values),
            "control_p3072_count": control_success,
            "treatment_p3072_count": treatment_success,
            "both_p3072_count": both,
            "gains": gains,
            "losses": losses,
            "control_p3072_rate": p0,
            "treatment_p3072_rate": p1,
            "shared_uniform_coupling": coupling,
            "paired_log1p_mean": statistics.mean(log_differences),
            "paired_log1p_sd": statistics.stdev(log_differences),
            "stream_strata": block_counts,
        }

    development_pairs = [
        pair for name in ("D0", "D1", "D2") for pair in groups[name]
    ]
    d = summarize(development_pairs)
    c = summarize(groups["C"])
    checks = {
        "all_source_hashes_exact": all(row["hashes_pass"] for row in source_rows),
        "development_pairs_768": d["pairs"] == 768,
        "development_counts_29_40_2": (
            d["control_p3072_count"],
            d["treatment_p3072_count"],
            d["both_p3072_count"],
        )
        == (29, 40, 2),
        "development_coupling_exact": math.isclose(
            d["shared_uniform_coupling"],
            0.017809776430466086,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "development_log_sd_exact": math.isclose(
            d["paired_log1p_sd"],
            1.1167440698964322,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "confirmation_pairs_512": c["pairs"] == 512,
        "confirmation_counts_21_21_3": (
            c["control_p3072_count"],
            c["treatment_p3072_count"],
            c["both_p3072_count"],
        )
        == (21, 21, 3),
        "confirmation_gains_losses_18": (c["gains"], c["losses"]) == (18, 18),
        "confirmation_coupling_exact": math.isclose(
            c["shared_uniform_coupling"],
            0.10619726505673553,
            rel_tol=0.0,
            abs_tol=1e-14,
        ),
        "confirmation_log_mean_supplied": math.isclose(
            c["paired_log1p_mean"], 0.02383, rel_tol=0.0, abs_tol=5e-6
        ),
        "confirmation_log_sd_supplied": math.isclose(
            c["paired_log1p_sd"], 1.18043, rel_tol=0.0, abs_tol=5e-6
        ),
    }
    return {
        "version": "o2_historical_capability_calibration_v1_a4",
        "sources": source_rows,
        "source_manifest_sha256": canonical_json_hash(source_rows),
        "development_d0_d2": d,
        "spent_confirmation_sensitivity": c,
        "conservative_score_sd": max(
            d["paired_log1p_sd"], c["paired_log1p_sd"]
        ),
        "historical_rows_opened": 2 * (768 + 512),
        "use": "spent aggregate power calibration only",
        "new_o2_outcomes_opened": 0,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _mh_log_or(
    treatment_success: np.ndarray,
    control_success: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    a = treatment_success.astype(np.float64)
    b = totals - a
    c = control_success.astype(np.float64)
    d = totals - c
    n = a + b + c + d
    numerator = np.sum(a * d / n, axis=-1)
    denominator = np.sum(b * c / n, axis=-1)
    zero = (numerator <= 0.0) | (denominator <= 0.0)
    if np.any(zero):
        az = a[zero] + 0.5
        bz = b[zero] + 0.5
        cz = c[zero] + 0.5
        dz = d[zero] + 0.5
        nz = az + bz + cz + dz
        numerator[zero] = np.sum(az * dz / nz, axis=-1)
        denominator[zero] = np.sum(bz * cz / nz, axis=-1)
    return np.log(numerator / denominator)


def _bootstrap_cluster_bounds(
    control_by_root: Sequence[np.ndarray],
    treatment_by_root: Sequence[np.ndarray],
    *,
    rng: np.random.Generator,
    bootstraps: int,
) -> tuple[float, float]:
    control_counts = []
    treatment_counts = []
    totals = []
    for control, treatment in zip(control_by_root, treatment_by_root):
        roots, repeats = control.shape
        indices = rng.integers(0, roots, size=(bootstraps, roots))
        control_root_success = control.sum(axis=1)
        treatment_root_success = treatment.sum(axis=1)
        control_counts.append(control_root_success[indices].sum(axis=1))
        treatment_counts.append(treatment_root_success[indices].sum(axis=1))
        totals.append(roots * repeats)
    control_matrix = np.stack(control_counts, axis=1)
    treatment_matrix = np.stack(treatment_counts, axis=1)
    total_matrix = np.broadcast_to(
        np.asarray(totals, dtype=np.float64),
        control_matrix.shape,
    )
    values = _mh_log_or(treatment_matrix, control_matrix, total_matrix)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _mechanism_strata(estimand: str) -> list[tuple[int, int, int]]:
    lower = [
        (stage, target, 8)
        for target in LOWER_TARGETS
        for stage in STARTING_STAGES
    ]
    transfer = [(stage, 768, 16) for stage in STARTING_STAGES]
    if estimand == "lower":
        return lower
    if estimand == "transfer":
        return transfer
    if estimand == "pooled":
        return lower + transfer
    raise ValueError(f"Unknown mechanism estimand: {estimand}")


def simulate_mechanism_power(
    estimand: str,
    odds_ratio: float,
    *,
    designs: int = MECHANISM_POWER_DESIGNS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    seed = (
        MECHANISM_POWER_SEED
        + {"lower": 0, "transfer": 1_000_000, "pooled": 2_000_000}[estimand]
        + int(round(odds_ratio * 10_000))
    )
    rng = np.random.default_rng(seed)
    strata = _mechanism_strata(estimand)
    passes = 0
    points = []
    for _design in range(designs):
        controls = []
        treatments = []
        total_success_control = []
        total_success_treatment = []
        totals = []
        for stage, target, roots in strata:
            base = rng.beta(MECHANISM_ALPHA, MECHANISM_BETA, size=roots)
            p0 = np.clip(
                base * STAGE_FACTORS[stage] * TARGET_FACTORS[target],
                0.002,
                0.80,
            )
            p1 = odds_ratio * p0 / (1.0 - p0 + odds_ratio * p0)
            shared = (
                rng.random((roots, MECHANISM_REPLICATES))
                < MECHANISM_COUPLING
            )
            common = rng.random((roots, MECHANISM_REPLICATES))
            control_u = np.where(
                shared,
                common,
                rng.random((roots, MECHANISM_REPLICATES)),
            )
            treatment_u = np.where(
                shared,
                common,
                rng.random((roots, MECHANISM_REPLICATES)),
            )
            control = control_u < p0[:, None]
            treatment = treatment_u < p1[:, None]
            controls.append(control)
            treatments.append(treatment)
            total_success_control.append(int(control.sum()))
            total_success_treatment.append(int(treatment.sum()))
            totals.append(roots * MECHANISM_REPLICATES)
        point = float(
            _mh_log_or(
                np.asarray([total_success_treatment]),
                np.asarray([total_success_control]),
                np.asarray([totals], dtype=np.float64),
            )[0]
        )
        lower, _upper = _bootstrap_cluster_bounds(
            controls,
            treatments,
            rng=rng,
            bootstraps=bootstraps,
        )
        passed = lower > 0.0 and point >= math.log(POWER_POINT_FLOOR)
        passes += int(passed)
        points.append(point)
    power = passes / designs
    return {
        "estimand": estimand,
        "n_roots": sum(row[2] for row in strata),
        "strata": len(strata),
        "repeats_per_arm_root": MECHANISM_REPLICATES,
        "true_odds_ratio": odds_ratio,
        "designs": designs,
        "bootstrap_replicates": bootstraps,
        "seed": seed,
        "full_gate_power": power,
        "monte_carlo_standard_error": math.sqrt(
            max(power * (1.0 - power), 0.0) / designs
        ),
        "mean_log_common_or": float(np.mean(points)),
        "gate_point_floor": POWER_POINT_FLOOR,
        "gate_lower_ci_floor": 1.0,
    }


def _paired_binary_counts(
    p0: float,
    p1: float,
    coupling: float,
) -> np.ndarray:
    shared_00 = 1.0 - max(p0, p1)
    shared_01 = max(p1 - p0, 0.0)
    shared_10 = max(p0 - p1, 0.0)
    shared_11 = min(p0, p1)
    independent = np.asarray(
        [
            (1.0 - p0) * (1.0 - p1),
            (1.0 - p0) * p1,
            p0 * (1.0 - p1),
            p0 * p1,
        ],
        dtype=np.float64,
    )
    shared = np.asarray(
        [shared_00, shared_01, shared_10, shared_11],
        dtype=np.float64,
    )
    probabilities = coupling * shared + (1.0 - coupling) * independent
    return probabilities / probabilities.sum()


def _bootstrap_binary_bounds(
    cells: Sequence[np.ndarray],
    roots_per_stratum: int,
    *,
    rng: np.random.Generator,
    bootstraps: int,
) -> tuple[float, float]:
    treatment_success = []
    control_success = []
    for counts in cells:
        probabilities = counts / counts.sum()
        draws = rng.multinomial(
            roots_per_stratum,
            probabilities,
            size=bootstraps,
        )
        treatment_success.append(draws[:, 1] + draws[:, 3])
        control_success.append(draws[:, 2] + draws[:, 3])
    treatment = np.stack(treatment_success, axis=1)
    control = np.stack(control_success, axis=1)
    totals = np.full_like(treatment, roots_per_stratum, dtype=np.float64)
    values = _mh_log_or(treatment, control, totals)
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def simulate_capability_power(
    *,
    n_roots: int,
    odds_ratio: float,
    base_rates: Sequence[float],
    coupling: float,
    calibration_name: str,
    designs: int = CAPABILITY_POWER_DESIGNS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    if n_roots % 8:
        raise ValueError("Capability N must be divisible by eight")
    if len(base_rates) != 8:
        raise ValueError("Capability requires eight base-rate strata")
    seed = (
        CAPABILITY_POWER_SEED
        + n_roots * 100
        + int(round(odds_ratio * 10_000))
        + int(round(coupling * 1_000_000))
        + (0 if calibration_name == "D0_D2" else 500_000_000)
    )
    rng = np.random.default_rng(seed)
    roots_per = n_roots // 8
    passes = 0
    points = []
    for _design in range(designs):
        cells = []
        treatment_success = []
        control_success = []
        for p0 in base_rates:
            p1 = odds_ratio * p0 / (1.0 - p0 + odds_ratio * p0)
            counts = rng.multinomial(
                roots_per,
                _paired_binary_counts(p0, p1, coupling),
            )
            cells.append(counts)
            treatment_success.append(int(counts[1] + counts[3]))
            control_success.append(int(counts[2] + counts[3]))
        point = float(
            _mh_log_or(
                np.asarray([treatment_success]),
                np.asarray([control_success]),
                np.full((1, 8), roots_per, dtype=np.float64),
            )[0]
        )
        lower, _upper = _bootstrap_binary_bounds(
            cells,
            roots_per,
            rng=rng,
            bootstraps=bootstraps,
        )
        passed = lower > 0.0 and point >= math.log(POWER_POINT_FLOOR)
        passes += int(passed)
        points.append(point)
    power = passes / designs
    return {
        "calibration": calibration_name,
        "n_roots": n_roots,
        "roots_per_stream_stratum": roots_per,
        "base_rates": list(base_rates),
        "coupling": coupling,
        "true_odds_ratio": odds_ratio,
        "designs": designs,
        "bootstrap_replicates": bootstraps,
        "seed": seed,
        "full_gate_power": power,
        "monte_carlo_standard_error": math.sqrt(
            max(power * (1.0 - power), 0.0) / designs
        ),
        "mean_log_common_or": float(np.mean(points)),
        "gate_point_floor": POWER_POINT_FLOOR,
        "gate_lower_ci_floor": 1.0,
    }


def _base_rate_vector(summary: Mapping[str, Any]) -> list[float]:
    return [
        row["control_p3072"] / row["roots"]
        for row in summary["stream_strata"]
    ]


def power_report(calibration: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = {}
    for estimand in ("lower", "transfer", "pooled"):
        rows = [
            simulate_mechanism_power(estimand, odds_ratio)
            for odds_ratio in POWER_OR_GRID
        ]
        mde = next(
            (
                row["true_odds_ratio"]
                for row in rows
                if row["full_gate_power"] >= POWER_REQUIRED
            ),
            None,
        )
        mechanism[estimand] = {
            "rows": rows,
            "mde_80pct_grid": mde,
            "or_1_50_power": next(
                row["full_gate_power"]
                for row in rows
                if row["true_odds_ratio"] == 1.50
            ),
        }

    base_vectors = {
        "D0_D2": _base_rate_vector(calibration["development_d0_d2"]),
        "C": _base_rate_vector(calibration["spent_confirmation_sensitivity"]),
    }
    couplings = (0.0, 0.017809776430466086, 0.10619726505673553)
    capability = {}
    for n_roots in (NORMAL_DEVELOPMENT_ROOTS, CONFIRMATION_ROOTS):
        rows = []
        for name, base_rates in base_vectors.items():
            for coupling in couplings:
                for odds_ratio in POWER_OR_GRID:
                    rows.append(
                        simulate_capability_power(
                            n_roots=n_roots,
                            odds_ratio=odds_ratio,
                            base_rates=base_rates,
                            coupling=coupling,
                            calibration_name=name,
                        )
                    )
        worst_by_or = {}
        for odds_ratio in POWER_OR_GRID:
            candidates = [
                row
                for row in rows
                if row["true_odds_ratio"] == odds_ratio
            ]
            worst = min(candidates, key=lambda row: row["full_gate_power"])
            worst_by_or[f"{odds_ratio:.2f}"] = {
                "power": worst["full_gate_power"],
                "calibration": worst["calibration"],
                "coupling": worst["coupling"],
            }
        mde = next(
            (
                odds_ratio
                for odds_ratio in POWER_OR_GRID
                if worst_by_or[f"{odds_ratio:.2f}"]["power"] >= POWER_REQUIRED
            ),
            None,
        )
        capability[str(n_roots)] = {
            "rows": rows,
            "worst_case_by_or": worst_by_or,
            "worst_case_mde_80pct_grid": mde,
        }

    conservative_sd = float(calibration["conservative_score_sd"])
    score_mde = {
        str(n): math.exp(
            (1.959963984540054 + 0.8416212335729143)
            * conservative_sd
            / math.sqrt(n)
        )
        - 1.0
        for n in (NORMAL_DEVELOPMENT_ROOTS, CONFIRMATION_ROOTS)
    }
    worst_confirmation = capability[str(CONFIRMATION_ROOTS)][
        "worst_case_by_or"
    ]["1.50"]["power"]
    checks = {
        "lower_or150_power_ge_80pct": mechanism["lower"]["or_1_50_power"]
        >= POWER_REQUIRED,
        "pooled_or150_power_ge_80pct": mechanism["pooled"]["or_1_50_power"]
        >= POWER_REQUIRED,
        "confirmation_worst_or150_power_ge_80pct": worst_confirmation
        >= POWER_REQUIRED,
    }
    return {
        "version": "o2_power_mde_v1_a4",
        "power_contract": {
            "full_gate": "point_OR>=1.25 and root-bootstrap lower_OR>1.00",
            "required_power": POWER_REQUIRED,
            "mechanism_designs": MECHANISM_POWER_DESIGNS,
            "capability_designs": CAPABILITY_POWER_DESIGNS,
            "bootstrap_replicates": POWER_BOOTSTRAPS,
            "or_grid": list(POWER_OR_GRID),
        },
        "mechanism": mechanism,
        "capability_p3072": capability,
        "score_primary": {
            "paired_log1p_sd": conservative_sd,
            "normal_approximation_mde": score_mde,
            "not_a_tail_or_milestone_power_claim": True,
        },
        "worst_case_or_1_50_power_at_n2560": worst_confirmation,
        "checks": checks,
        "passes": all(checks.values()),
    }


def resource_projection() -> dict[str, Any]:
    acquisition_storage = math.ceil(
        1.25
        * (PILOT_ROOTS_PER_FAMILY * 4 + CORPUS_ROOTS_PER_FAMILY * 4)
        * (MAX_REPLAY_BYTES + MIB)
        + 512 * MIB
    )
    phases = {
        "pilot_plus_corpus": {
            "units": 768,
            "projected_seconds": 768 * ACQUISITION_GAME_SECONDS * 2.5,
            "hard_seconds": 6 * 3600,
            "projected_bytes": acquisition_storage,
            "hard_bytes": 3 * GIB,
        },
        "option_learning": {
            "units": 1_024,
            "projected_seconds": 1_024 * H40_PATH_SECONDS * 2.5,
            "hard_seconds": 8 * 3600,
            "projected_bytes": 512 * MIB,
            "hard_bytes": 3 * GIB,
        },
        "option_development": {
            "units": 768,
            "projected_seconds": 768 * H40_PATH_SECONDS * 2.5,
            "hard_seconds": 6 * 3600,
            "projected_bytes": 384 * MIB,
            "hard_bytes": 2 * GIB,
        },
        "mechanism_test": {
            "units": 3_072,
            "projected_seconds": 3_072 * H40_PATH_SECONDS * 2.5,
            "hard_seconds": 24 * 3600,
            "projected_bytes": 1 * GIB,
            "hard_bytes": 3 * GIB,
        },
        "normal_development": {
            "units": 384,
            "projected_seconds": 384 * ACQUISITION_GAME_SECONDS * 3.5,
            "hard_seconds": 8 * 3600,
            "projected_bytes": 512 * MIB,
            "hard_bytes": 3 * GIB,
        },
        "confirmation": {
            "units": 2_560,
            "projected_seconds": 2_560 * ACQUISITION_GAME_SECONDS * 3.5,
            "hard_seconds": 30 * 3600,
            "projected_bytes": 2 * GIB,
            "hard_bytes": 4 * GIB,
        },
    }
    for row in phases.values():
        row["runtime_pass"] = row["projected_seconds"] <= row["hard_seconds"]
        row["storage_pass"] = row["projected_bytes"] <= row["hard_bytes"]
    return {
        "complete_game_seconds": ACQUISITION_GAME_SECONDS,
        "h40_path_seconds": H40_PATH_SECONDS,
        "phases": phases,
        "checks": {
            "all_runtime_projections_within_caps": all(
                row["runtime_pass"] for row in phases.values()
            ),
            "all_storage_projections_within_caps": all(
                row["storage_pass"] for row in phases.values()
            ),
            "acquisition_storage_exact": acquisition_storage == 2_503_888_832,
        },
        "passes": all(
            row["runtime_pass"] and row["storage_pass"]
            for row in phases.values()
        )
        and acquisition_storage == 2_503_888_832,
    }


def operational_audit() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    free_gib = disk.free / GIB
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    service = history.service_health()
    heavy = _heavy_process_audit()
    checks = {
        "nice_at_least_10": nice >= 10,
        "no_competing_heavy_process": bool(heavy.get("passes")),
        "free_disk_hard_pass": free_gib >= MIN_FREE_GIB,
        "free_disk_target_pass": free_gib >= TARGET_FREE_GIB,
        "services_healthy": bool(service.get("passes")),
        "top_three_exact": tuple(service.get("dashboard_top_scores", [])[:3])
        == EXPECTED_TOP_THREE,
    }
    return {
        "nice": nice,
        "disk": {
            "free_bytes": disk.free,
            "free_gib": free_gib,
            "minimum_gib": MIN_FREE_GIB,
            "target_gib": TARGET_FREE_GIB,
        },
        "heavy_process_audit": heavy,
        "service_health": service,
        "checks": checks,
        "passes": all(checks.values()),
    }


def decide(
    *,
    integrity_passes: bool,
    power_passes: bool,
    resource_passes: bool,
    operational_passes: bool,
) -> str:
    if not integrity_passes:
        return "KILL_O2_PREFLIGHT_INTEGRITY"
    if not power_passes or not resource_passes or not operational_passes:
        return "HOLD_O2_COST_OR_POWER"
    return "READY_O2_YIELD_PILOT_PREFLIGHT"


def run_preflight() -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"O2 preflight output already exists: {OUTPUT_DIR}")
    started = time.time()
    documents = bound_document_audit()
    families = family_evidence_audit()
    design = design_manifest()
    rows = stream_rows()
    internal_streams = internal_stream_audit(rows)
    collision, collision_sources = historical_collision_audit(rows)
    calibration = historical_calibration()
    power = power_report(calibration)
    resources = resource_projection()
    operations = operational_audit()

    integrity_checks = {
        "documents": documents["passes"],
        "families": families["passes"],
        "design": design["passes"],
        "internal_streams": internal_streams["passes"],
        "zero_historical_stream_collisions": collision[
            "zero_historical_collisions"
        ],
        "calibration": calibration["passes"],
        "output_absent_before_seal": not OUTPUT_DIR.exists(),
    }
    integrity_passes = all(integrity_checks.values())
    decision = decide(
        integrity_passes=integrity_passes,
        power_passes=power["passes"],
        resource_passes=resources["passes"],
        operational_passes=operations["passes"],
    )

    stream_payload = {
        "version": "o2_stream_manifest_v1_a4",
        "stream_bases": {
            purpose: {
                field: value
                for field, value in zip(STREAM_FIELDS, bases)
            }
            for purpose, bases in STREAM_BASES.items()
        },
        "rows": rows,
        "row_manifest_sha256": canonical_json_hash(rows),
        "internal_audit": internal_streams,
        "collision_summary": collision,
        "streams_consumed": 0,
    }
    write_immutable_json(COLLISION_MANIFEST_PATH, collision_sources)
    write_immutable_json(STREAM_MANIFEST_PATH, stream_payload)
    write_immutable_json(DESIGN_MANIFEST_PATH, design)
    write_immutable_json(CALIBRATION_PATH, calibration)
    write_immutable_json(POWER_PATH, power)

    result = {
        "version": VERSION,
        "created_at": time.time(),
        "elapsed_seconds": time.time() - started,
        "decision": decision,
        "terminal_status": "HOLD_O2_AFTER_OUTCOME_FREE_PREFLIGHT",
        "promote": False,
        "bound_artifacts": {
            "charter": {
                "path": str(CHARTER_PATH),
                "sha256": CHARTER_SHA256,
            },
            "amendment_a1": {"path": str(A1_PATH), "sha256": A1_SHA256},
            "amendment_a2": {"path": str(A2_PATH), "sha256": A2_SHA256},
            "amendment_a3": {"path": str(A3_PATH), "sha256": A3_SHA256},
            "amendment_a4": {"path": str(A4_PATH), "sha256": A4_SHA256},
            "runner": {
                "path": str(RUNNER_PATH),
                "sha256": sha256_path(RUNNER_PATH),
            },
            "tests": {
                "path": str(TEST_PATH),
                "sha256": sha256_path(TEST_PATH),
            },
            "test_evidence": (
                artifact_identity(TEST_EVIDENCE_PATH)
                if TEST_EVIDENCE_PATH.is_file()
                else None
            ),
        },
        "documents": documents,
        "families": families,
        "design_artifact": artifact_identity(DESIGN_MANIFEST_PATH),
        "stream_artifact": artifact_identity(STREAM_MANIFEST_PATH),
        "collision_artifact": artifact_identity(COLLISION_MANIFEST_PATH),
        "calibration_artifact": artifact_identity(CALIBRATION_PATH),
        "power_artifact": artifact_identity(POWER_PATH),
        "power_summary": {
            "mechanism": {
                name: {
                    "or_1_50_power": value["or_1_50_power"],
                    "mde_80pct_grid": value["mde_80pct_grid"],
                }
                for name, value in power["mechanism"].items()
            },
            "worst_case_or_1_50_power_at_n2560": power[
                "worst_case_or_1_50_power_at_n2560"
            ],
            "score_mde": power["score_primary"][
                "normal_approximation_mde"
            ],
        },
        "resources": resources,
        "operations": operations,
        "integrity_checks": integrity_checks,
        "integrity_passes": integrity_passes,
        "zero_work": {
            "new_games": 0,
            "streams_consumed": 0,
            "new_rollouts": 0,
            "new_labels": 0,
            "model_fits": 0,
            "candidate_actions": 0,
            "policy_outcomes": 0,
            "score_outcomes_inspected": 0,
            "dashboard_changes": 0,
            "historical_spent_calibration_rows_read": calibration[
                "historical_rows_opened"
            ],
        },
        "pilot_execution_authorized": False,
    }
    write_immutable_json(RESULT_PATH, result)
    return json.loads(RESULT_PATH.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight",))
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(run_preflight(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

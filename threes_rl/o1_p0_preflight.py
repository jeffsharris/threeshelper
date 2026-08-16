"""Outcome-free support and power preflight for the O1 option policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
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
    geometry,
    root_goal,
    root_option_eligible,
    schema_manifest,
    schema_sha256,
)
from threes_rl.replay_provenance import (
    GENUINE_ROOT_ORIGINS,
    replay_provenance,
)
from threes_rl.restart_manifest import (
    canonical_ancestry_id,
    replay_behavior_family,
    state_signature,
)
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "o1_goal_conditioned_option_p0_v1_a6"
RESULT_VERSION = "o1_goal_conditioned_option_p0_result_v1_a6"
ROOT = Path("threes_rl/runs")
OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/o1_goal_conditioned_option_p0_v1"
)
CHARTER_PATH = Path(
    "threes_rl/O1_GOAL_CONDITIONED_GEOMETRY_OPTION_CHARTER.md"
)
A1_PATH = Path(
    "threes_rl/O1_GOAL_CONDITIONED_GEOMETRY_OPTION_CHARTER_AMENDMENT_A1.md"
)
A2_PATH = Path(
    "threes_rl/O1_GOAL_CONDITIONED_GEOMETRY_OPTION_CHARTER_AMENDMENT_A2.md"
)
A3_PATH = Path(
    "threes_rl/O1_GOAL_CONDITIONED_GEOMETRY_OPTION_CHARTER_AMENDMENT_A3.md"
)
A4_PATH = Path(
    "threes_rl/O1_GOAL_CONDITIONED_GEOMETRY_OPTION_CHARTER_AMENDMENT_A4.md"
)
A5_PATH = Path(
    "threes_rl/O1_GOAL_CONDITIONED_GEOMETRY_OPTION_CHARTER_AMENDMENT_A5.md"
)
A6_PATH = Path(
    "threes_rl/O1_GOAL_CONDITIONED_GEOMETRY_OPTION_CHARTER_AMENDMENT_A6.md"
)
GEOMETRY_PATH = Path("threes_rl/o1_geometry_option.py")
RUNNER_PATH = Path("threes_rl/o1_p0_preflight.py")
TEST_PATH = Path("tests/test_rl_o1_p0_preflight.py")
OLD_TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/o1_p0_test_evidence.json"
)
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/o1_p0_test_evidence_a5.json"
)
EVIDENCE_TRANSITION_PATH = Path(
    "threes_rl/runs/forensics/o1_p0_a6_evidence_transition.json"
)
SOURCE_INVENTORY_PATH = OUTPUT_DIR / "O1_P0_SOURCE_PATH_INVENTORY.json"
EXCLUSION_PATH = OUTPUT_DIR / "O1_P0_EXCLUSION_MANIFEST.json"
MARKER_PATH = OUTPUT_DIR / "O1_P0_CONTENT_OPENED.json"
ROOT_MANIFEST_PATH = OUTPUT_DIR / "O1_P0_ROOT_MANIFEST.json"
RESULT_PATH = OUTPUT_DIR / "O1_P0_RESULT.json"

CHARTER_SHA256 = (
    "d6ea7fb6f0ff547cbc84486d723c90fb4603900004dc181dff1e02e58622bdb4"
)
A1_SHA256 = (
    "712dada0815a696beb6040b15970515d454f9c7ddb1578e73fd98cc27a87955e"
)
A2_SHA256 = (
    "42bec962eecda69d83e9493d3c57645a869b6fc048cd26ad8d77d259c9cdef76"
)
A3_SHA256 = (
    "b5564af3af217c9e69fb88d40c1a3d8af140439819b7fa67bdfb687c80ad6d6a"
)
A4_SHA256 = (
    "01e8ec2270ea82610bc84b36f5ab8d8abf6dbbc47ad5f611b1e54c4e5b633cf6"
)
A5_SHA256 = (
    "e7e83607994f71fa5818e7c781511f22c344792f3a46eaec97a84a395829cf8d"
)
A6_SHA256 = (
    "5f43be10027d042ba693a065f14fabf77b6952fbf7e2ea47e575057f5f92adb6"
)
OLD_TEST_EVIDENCE_FILE_SHA256 = (
    "3354f76d42bc69cc425297b3e2aaa720847aec4d4d1fd746637c44a3ff633734"
)
OLD_TEST_EVIDENCE_PAYLOAD_SHA256 = (
    "1d08910bc8230a07d9ddccf53eb0f5631b955f2b597044e742caae8cd953135a"
)
PILOT_V1_PREFLIGHT = Path(
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v1/"
    "preflight_lock_pilot_v1.json"
)
PILOT_V1_PREFLIGHT_SHA256 = (
    "f78288b3f47bda6aa6d15c2157fd79f7b3d0685f0367d8b9964f5dc73981ea91"
)
PILOT_V1_ACTION_AUDIT_SHA256 = (
    "f78184001df46b9eab4e71a7e620fb9247c9a05b88613846fc22f1879512eab4"
)
QD_V2_ADMISSION = Path(
    "threes_rl/runs/forensics/"
    "g1r_qd_admission_v2_terminal_schema/admission_result.json"
)
QD_V2_ADMISSION_SHA256 = (
    "27bcb3328a02d6dc5094dcc5a8e52b8f27d2f3e4ea7b92f5c1a8153bc1326a8e"
)
ACTION_PANEL_SHA256 = (
    "b8862aa3c8eaf6278fc078fb3e03aa7222a01930673cfee497738c74e81eff9d"
)

STREAM_BASES = {
    "logical_seed": 77_000_000_000,
    "deck_stream_id": 78_000_000_000,
    "slot_stream_id": 79_000_000_000,
    "policy_stream_id": 80_000_000_000,
}
POWER_N_GRID = (144, 192, 264, 384, 516, 768, 1020, 1536)
POWER_OR_GRID = (1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00)
POWER_REQUIRED = 0.80
POWER_DESIGNS = 4096
POWER_REPEATS = 8
POWER_COUPLING = 0.50
POWER_ALPHA = 1.6
POWER_BETA = 18.4
Z_975 = 1.959963984540054
SE_INFLATION = 1.10
STAGES = (0, 1, 2, 3)
SCALE_BANDS = ("early", "mid", "late")
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
EXPECTED_TOP_THREE = (263670, 261369, 258561)
FAMILY_CAP = 0.40
MIN_TEST_ROOTS_PER_STAGE = 48
MIN_STRUCTURAL_TEST_N = 4 * MIN_TEST_ROOTS_PER_STAGE
GENUINE_FAMILIES = (
    "g1r_corner2",
    "g1r_expectimax2",
    "g1r_parent_mc1000",
    "g1r_qd_static_archive_oneply_v2_terminal_schema",
    "g1r_replaycal",
)
ROOT_TOKEN_PATTERN = re.compile(rb"fresh:[0-9]+:(?:[0-9]+|null|None)")
PRIOR_SUFFIXES = {".json", ".jsonl", ".csv"}


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
        raise FileExistsError(f"Refusing to overwrite immutable O1 artifact: {path}")
    value = payload_with_hash(payload)
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if not verify_payload_hash(json.loads(serialized)):
        raise ValueError("O1 payload failed pre-write JSON round trip")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    if not verify_payload_hash(json.loads(path.read_text())):
        raise ValueError("O1 payload failed post-write verification")


def artifact_identity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload.get("canonical_payload_sha256"),
        "payload_valid": verify_payload_hash(payload),
        "bytes": path.stat().st_size,
    }


def verify_evidence_transition() -> dict[str, Any]:
    old_payload = json.loads(OLD_TEST_EVIDENCE_PATH.read_text())
    transition = json.loads(EVIDENCE_TRANSITION_PATH.read_text())
    checks = {
        "old_evidence_file":
            sha256_path(OLD_TEST_EVIDENCE_PATH)
            == OLD_TEST_EVIDENCE_FILE_SHA256,
        "old_evidence_payload":
            verify_payload_hash(old_payload)
            and old_payload["canonical_payload_sha256"]
            == OLD_TEST_EVIDENCE_PAYLOAD_SHA256,
        "transition_payload": verify_payload_hash(transition),
        "transition_version":
            transition.get("version")
            == "o1_p0_a6_evidence_transition_v1",
        "a6_amendment":
            transition.get("a6_amendment_sha256") == A6_SHA256
            and sha256_path(A6_PATH) == A6_SHA256,
        "runner":
            transition.get("runner_sha256") == sha256_path(RUNNER_PATH),
        "tests":
            transition.get("tests_sha256") == sha256_path(TEST_PATH),
        "new_evidence":
            transition.get("new_test_evidence")
            == artifact_identity(TEST_EVIDENCE_PATH),
        "old_evidence_bound":
            transition.get("old_test_evidence")
            == artifact_identity(OLD_TEST_EVIDENCE_PATH),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "transition": artifact_identity(EVIDENCE_TRANSITION_PATH),
    }


def scale_band(target: int) -> str:
    if target in (48, 96, 192):
        return "early"
    if target in (384, 768):
        return "mid"
    if target == 1536:
        return "late"
    raise ValueError(f"Unsupported O1 target scale: {target}")


def trajectory_stream_ids(
    partition: str,
    root_index: int,
    *,
    round_index: int = 0,
    replicate: int = 0,
    incumbent_arm: bool = False,
) -> dict[str, int]:
    if partition == "train":
        if not 0 <= round_index < 4 or not 0 <= replicate < 2:
            raise ValueError("Bad train round/replicate")
        code = root_index * 8 + round_index * 2 + replicate
    elif partition == "development":
        if not 0 <= replicate < 8:
            raise ValueError("Bad development replicate")
        code = 1_000_000 + root_index * 8 + replicate
    elif partition == "untouched_test":
        if not 0 <= replicate < 8:
            raise ValueError("Bad test replicate")
        code = 2_000_000 + root_index * 8 + replicate
    else:
        raise ValueError(f"Unsupported partition: {partition}")
    return {
        "trajectory_code": code,
        "logical_seed": STREAM_BASES["logical_seed"] + code,
        "deck_stream_id": STREAM_BASES["deck_stream_id"] + code,
        "slot_stream_id": STREAM_BASES["slot_stream_id"] + code,
        "policy_stream_id":
            STREAM_BASES["policy_stream_id"] + 2 * code + int(incumbent_arm),
    }


def family_classifier(replay: dict[str, Any], path: Path) -> str:
    raw = replay_behavior_family(replay, path)
    policy = str(replay.get("policy", "")).lower()
    text = f"{raw} {policy} {path}".lower()
    if raw == "corner2_lineage" or policy.startswith("corner2"):
        return "g1r_corner2"
    if raw == "expectimax_baseline" or policy.startswith("expectimax2"):
        return "g1r_expectimax2"
    if "replaycal" in text or "replay_cal" in text:
        return "g1r_replaycal"
    if "g1r_qd_static_archive_oneply_v2_terminal_schema" in text:
        return "g1r_qd_static_archive_oneply_v2_terminal_schema"
    if (
        raw
        in {
            "phaseblend_incumbent_lineage",
            "phaseblend_cheap_lineage",
            "legacy_ntuple_lineage",
            "td_student_lineage",
            "ntuple",
        }
        or raw.startswith("train_td:")
        or any(
            token in text
            for token in ("student1", "parent_mc1000", "phaseblend")
        )
    ):
        return "g1r_parent_mc1000"
    return "ineligible_unverified_family"


def path_exclusion_reason(path: Path) -> str | None:
    text = str(path).lower()
    parts = {part.lower() for part in path.parts}
    if "forensics" in parts:
        return "prior_forensics"
    if "top_games" in parts or "replays" in parts:
        return "score_or_playlist_selected"
    if "continuations" in parts or "continuation" in text:
        return "continuation"
    if "human_diagnostics" in parts or "human" in text:
        return "human"
    if "diagnostic_games" in parts:
        return "diagnostic_selected"
    if any(part.startswith("rank_") or "score_" in part for part in parts):
        return "rank_or_score_path"
    return None


def source_path_inventory() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for path in sorted(ROOT.rglob("replay.json")):
        counts["discovered_replay_json"] += 1
        if path.is_symlink() or not path.is_file():
            counts["non_regular_or_symlink"] += 1
            rows.append(
                {
                    "path": str(path),
                    "eligible_for_content_scan": False,
                    "reason": "non_regular_or_symlink",
                }
            )
            continue
        reason = path_exclusion_reason(path)
        stat = path.stat()
        row = {
            "path": str(path),
            "bytes": int(stat.st_size),
            "sha256": sha256_path(path),
            "eligible_for_content_scan": reason is None,
            "reason": reason or "path_eligible",
        }
        rows.append(row)
        counts[row["reason"]] += 1
    return {
        "version": "o1_p0_source_path_inventory_v1",
        "scan_root": str(ROOT),
        "rows": rows,
        "rows_sha256": canonical_json_hash(rows),
        "counts": dict(sorted(counts.items())),
        "candidate_json_parsed": False,
    }


def _root_tokens_from_file(path: Path) -> set[str]:
    found: set[str] = set()
    overlap = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            chunk = overlap + chunk
            found.update(value.decode("ascii") for value in ROOT_TOKEN_PATTERN.findall(chunk))
            overlap = chunk[-128:]
    return found


def exclusion_manifest() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    roots: set[str] = set()
    scan_root = ROOT / "forensics"
    output_resolved = OUTPUT_DIR.resolve(strict=False)
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file() or path.suffix not in PRIOR_SUFFIXES:
            continue
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        matched = _root_tokens_from_file(path)
        if not matched:
            continue
        roots.update(matched)
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
                "root_token_count": len(matched),
            }
        )
    return {
        "version": "o1_p0_exclusion_manifest_v1",
        "scan_root": str(scan_root),
        "source_rows": rows,
        "source_rows_sha256": canonical_json_hash(rows),
        "excluded_roots": sorted(roots),
        "excluded_root_count": len(roots),
        "excluded_roots_sha256": canonical_json_hash(sorted(roots)),
        "protected_state_fields_opened": False,
        "root_tokens_only": True,
    }


def verify_family_evidence() -> dict[str, Any]:
    if sha256_path(PILOT_V1_PREFLIGHT) != PILOT_V1_PREFLIGHT_SHA256:
        raise ValueError("G1-R pilot-v1 signature evidence changed")
    if sha256_path(QD_V2_ADMISSION) != QD_V2_ADMISSION_SHA256:
        raise ValueError("QD-v2 admission evidence changed")
    pilot = json.loads(PILOT_V1_PREFLIGHT.read_text())
    qd = json.loads(QD_V2_ADMISSION.read_text())
    audit = pilot["action_distinctness_audit"]
    checks = {
        "pilot_action_audit_hash":
            audit["audit_sha256"] == PILOT_V1_ACTION_AUDIT_SHA256,
        "panel_hash":
            audit["panel_sha256"] == ACTION_PANEL_SHA256
            and qd["panel_sha256"] == ACTION_PANEL_SHA256,
        "four_base_components":
            audit["representative_families"]
            == [
                "g1r_corner2",
                "g1r_expectimax2",
                "g1r_parent_mc1000",
                "g1r_replaycal",
            ],
        "qd_ready": qd["decision"] == "READY_QD_FAMILY_ADMISSION",
        "qd_pairwise_all_pass":
            len(qd["pairwise"]) == 4
            and all(bool(row["passes"]) for row in qd["pairwise"]),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "pilot_v1_file_sha256": PILOT_V1_PREFLIGHT_SHA256,
        "pilot_action_audit_sha256": PILOT_V1_ACTION_AUDIT_SHA256,
        "qd_v2_file_sha256": QD_V2_ADMISSION_SHA256,
        "panel_sha256": ACTION_PANEL_SHA256,
        "genuine_families": list(GENUINE_FAMILIES),
    }


def _mh_log_or(
    treatment: np.ndarray,
    control: np.ndarray,
) -> np.ndarray:
    # Shapes: designs, 12 strata, roots-per-stratum, repeats.
    a = treatment.sum(axis=(2, 3)).astype(np.float64)
    b = treatment.shape[2] * treatment.shape[3] - a
    c = control.sum(axis=(2, 3)).astype(np.float64)
    d = control.shape[2] * control.shape[3] - c
    n = a + b + c + d
    numerator = np.sum(a * d / n, axis=1)
    denominator = np.sum(b * c / n, axis=1)
    zero = (numerator <= 0.0) | (denominator <= 0.0)
    if np.any(zero):
        az, bz, cz, dz = (
            values[zero] + 0.5 for values in (a, b, c, d)
        )
        nz = az + bz + cz + dz
        numerator[zero] = np.sum(az * dz / nz, axis=1)
        denominator[zero] = np.sum(bz * cz / nz, axis=1)
    return np.log(numerator / denominator)


def simulate_common_or_power(n_roots: int, odds_ratio: float) -> dict[str, Any]:
    if n_roots % 12:
        raise ValueError("O1 power N must be divisible by 12")
    roots_per = n_roots // 12
    seed = 2_026_072_601 + 1000 * n_roots + round(100 * odds_ratio)
    rng = np.random.default_rng(seed)
    stage_factors = np.asarray((0.50, 0.75, 1.00, 1.50), dtype=np.float64)
    scale_factors = np.asarray((1.25, 1.00, 0.75), dtype=np.float64)
    factors = np.asarray(
        [
            stage_factors[stage] * scale_factors[scale]
            for stage in range(4)
            for scale in range(3)
        ],
        dtype=np.float64,
    )
    log_ors: list[np.ndarray] = []
    remaining = POWER_DESIGNS
    while remaining:
        batch = min(128, remaining)
        root_base = rng.beta(
            POWER_ALPHA,
            POWER_BETA,
            size=(batch, 12, roots_per),
        )
        p0 = np.clip(root_base * factors[None, :, None], 0.002, 0.80)
        p1 = odds_ratio * p0 / (1.0 - p0 + odds_ratio * p0)
        shared_flag = rng.random((batch, 12, roots_per, POWER_REPEATS)) < POWER_COUPLING
        shared_u = rng.random((batch, 12, roots_per, POWER_REPEATS))
        control_u = np.where(
            shared_flag,
            shared_u,
            rng.random((batch, 12, roots_per, POWER_REPEATS)),
        )
        treatment_u = np.where(
            shared_flag,
            shared_u,
            rng.random((batch, 12, roots_per, POWER_REPEATS)),
        )
        control = control_u < p0[..., None]
        treatment = treatment_u < p1[..., None]
        log_ors.append(_mh_log_or(treatment, control))
        remaining -= batch
    values = np.concatenate(log_ors)
    standard_error = float(np.std(values, ddof=1))
    inflated = SE_INFLATION * standard_error
    lower = values - Z_975 * inflated
    power = float(np.mean(lower > 0.0))
    return {
        "n_roots": n_roots,
        "roots_per_stratum": roots_per,
        "odds_ratio": odds_ratio,
        "designs": POWER_DESIGNS,
        "seed": seed,
        "mean_log_common_or": float(np.mean(values)),
        "log_common_or_sd": standard_error,
        "inflated_standard_error": inflated,
        "power_lower_bound_above_zero": power,
    }


def power_table() -> dict[str, Any]:
    or150_rows = []
    for n_roots in POWER_N_GRID:
        row = simulate_common_or_power(n_roots, 1.50)
        row["power_pass"] = (
            row["power_lower_bound_above_zero"] >= POWER_REQUIRED
        )
        row["structural_minimum_pass"] = (
            n_roots >= MIN_STRUCTURAL_TEST_N
            and n_roots // 4 >= MIN_TEST_ROOTS_PER_STAGE
        )
        row["eligible_for_selection"] = (
            row["power_pass"] and row["structural_minimum_pass"]
        )
        or150_rows.append(row)
    selected = next(
        (
            int(row["n_roots"])
            for row in or150_rows
            if row["eligible_for_selection"]
        ),
        None,
    )
    mde_rows = (
        [
            simulate_common_or_power(selected, odds_ratio)
            for odds_ratio in POWER_OR_GRID
        ]
        if selected is not None
        else []
    )
    mde = next(
        (
            float(row["odds_ratio"])
            for row in mde_rows
            if row["power_lower_bound_above_zero"] >= POWER_REQUIRED
        ),
        None,
    )
    return {
        "version": "o1_p0_common_or_power_v1_a4",
        "primary_estimator": "mantel_haenszel_common_odds_ratio_12_strata",
        "cluster_unit": "whole_root",
        "target_or": 1.50,
        "required_power": POWER_REQUIRED,
        "minimum_test_roots_per_starting_stage":
            MIN_TEST_ROOTS_PER_STAGE,
        "minimum_structurally_eligible_n": MIN_STRUCTURAL_TEST_N,
        "or150_rows": or150_rows,
        "selected_smallest_passing_n": selected,
        "mde_at_selected_n": mde,
        "mde_rows": mde_rows,
    }


def _rotation(values: Sequence[str], key: str) -> list[str]:
    ordered = sorted(values)
    if not ordered:
        return []
    offset = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _allocate_cells(
    rows: list[dict[str, Any]],
    *,
    cells: Sequence[tuple[Any, ...]],
    quota_per_cell: int,
    total_target: int,
    key_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cap = int(math.floor(FAMILY_CAP * total_target))
    family_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    deficits: dict[str, int] = {}
    for cell in cells:
        cell_rows = [
            row
            for row in rows
            if tuple(row[field] for field in (
                ("stage", "scale_band")
                if len(cell) == 2
                else ("stage",)
            )) == cell
            and row not in selected
        ]
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cell_rows:
            by_family[str(row["behavior_family"])].append(row)
        for family, family_rows in by_family.items():
            family_rows.sort(
                key=lambda row: hashlib.sha256(
                    "\0".join(
                        (
                            key_prefix,
                            "|".join(str(part) for part in cell),
                            family,
                            str(row["root_cluster"]),
                        )
                    ).encode("utf-8")
                ).hexdigest()
            )
        family_order = _rotation(
            list(by_family),
            f"O1-family-rotation|{'|'.join(str(part) for part in cell)}",
        )
        filled = 0
        while filled < quota_per_cell:
            progressed = False
            for family in family_order:
                if filled >= quota_per_cell:
                    break
                if family_counts[family] >= cap or not by_family[family]:
                    continue
                selected.append(by_family[family].pop(0))
                family_counts[family] += 1
                filled += 1
                progressed = True
            if not progressed:
                break
        if filled < quota_per_cell:
            deficits["|".join(str(part) for part in cell)] = quota_per_cell - filled
    roots = [str(row["root_cluster"]) for row in selected]
    return selected, {
        "target": total_target,
        "selected": len(selected),
        "family_cap_count": cap,
        "family_counts": dict(sorted(family_counts.items())),
        "family_count": len(family_counts),
        "maximum_family_share":
            max(family_counts.values(), default=0) / len(selected)
            if selected else 0.0,
        "deficits": deficits,
        "root_sha256": canonical_json_hash(sorted(roots)),
        "passes":
            len(selected) == total_target
            and not deficits
            and len(family_counts) >= 4
            and max(family_counts.values(), default=0) <= cap,
    }


def allocate_partitions(
    records: list[dict[str, Any]],
    selected_test_n: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if selected_test_n is None:
        return [], {
            "passes": False,
            "reason": "no_power_passing_test_n",
        }
    test_cells = tuple(
        (stage, scale)
        for stage in STAGES
        for scale in SCALE_BANDS
    )
    test, test_report = _allocate_cells(
        records,
        cells=test_cells,
        quota_per_cell=selected_test_n // 12,
        total_target=selected_test_n,
        key_prefix="O1-test-v1",
    )
    used = {str(row["root_cluster"]) for row in test}
    remaining = [row for row in records if str(row["root_cluster"]) not in used]
    stage_cells = tuple((stage,) for stage in STAGES)
    development, dev_report = _allocate_cells(
        remaining,
        cells=stage_cells,
        quota_per_cell=20,
        total_target=80,
        key_prefix="O1-dev-v1",
    )
    used.update(str(row["root_cluster"]) for row in development)
    remaining = [row for row in remaining if str(row["root_cluster"]) not in used]
    train, train_report = _allocate_cells(
        remaining,
        cells=stage_cells,
        quota_per_cell=60,
        total_target=240,
        key_prefix="O1-train-v1",
    )
    assigned: list[dict[str, Any]] = []
    for partition, rows in (
        ("untouched_test", test),
        ("development", development),
        ("train", train),
    ):
        for row in rows:
            assigned.append({**row, "partition": partition})
    all_roots = [str(row["root_cluster"]) for row in assigned]
    test_stage_counts = Counter(int(row["stage"]) for row in test)
    test_structural_stage_minimum = all(
        test_stage_counts[stage] >= MIN_TEST_ROOTS_PER_STAGE
        for stage in STAGES
    )
    return assigned, {
        "test": test_report,
        "development": dev_report,
        "train": train_report,
        "test_stage_counts": {
            str(stage): test_stage_counts[stage] for stage in STAGES
        },
        "minimum_test_roots_per_stage": MIN_TEST_ROOTS_PER_STAGE,
        "test_structural_stage_minimum": test_structural_stage_minimum,
        "assigned_roots": len(set(all_roots)),
        "zero_cross_partition_root_overlap":
            len(all_roots) == len(set(all_roots)),
        "passes":
            test_report["passes"]
            and dev_report["passes"]
            and train_report["passes"]
            and test_structural_stage_minimum
            and len(all_roots) == len(set(all_roots)),
    }


def _compact_state(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "board": payload.get("board"),
        "preview": payload.get("preview"),
        "tile_cycle": payload.get("tile_cycle"),
        "move_count": payload.get("move_count"),
        "game_over": payload.get("game_over"),
        "legal_actions": payload.get("legal_actions"),
        "legal_mask": payload.get("legal_mask"),
        "max_tile": payload.get("max_tile"),
    }


def scan_candidate_content(
    source_inventory: dict[str, Any],
    exclusion: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_roots = set(exclusion["excluded_roots"])
    candidates_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    disqualified_roots: set[str] = set()
    counts: Counter[str] = Counter()
    hygiene_failures: list[dict[str, Any]] = []
    for source in source_inventory["rows"]:
        if not source.get("eligible_for_content_scan"):
            continue
        path = Path(source["path"])
        counts["candidate_files_opened"] += 1
        if sha256_path(path) != source["sha256"]:
            raise ValueError(f"Candidate source changed after marker: {path}")
        try:
            replay = json.loads(path.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            counts["invalid_json"] += 1
            counts["data_hygiene_failure_sources"] += 1
            hygiene_failures.append(
                {"path": str(path), "error": type(error).__name__}
            )
            continue
        provenance = replay_provenance(replay, path)
        if (
            provenance.get("replay_origin") not in GENUINE_ROOT_ORIGINS
            or provenance.get("root_origin") not in GENUINE_ROOT_ORIGINS
            or not provenance.get("replay_reset_invariant")
            or provenance.get("replay_origin") == "human"
        ):
            counts["non_fresh_machine"] += 1
            continue
        frames = replay.get("frames")
        if not isinstance(frames, list) or not frames:
            counts["missing_frames"] += 1
            continue
        final = frames[-1]
        final_state = final.get("state") if isinstance(final, dict) else None
        if not isinstance(final_state, dict) or not bool(final_state.get("game_over")):
            counts["incomplete"] += 1
            continue
        if provenance.get("root_seed") is None:
            counts["missing_root_seed"] += 1
            continue
        root = canonical_ancestry_id(replay, path)
        if root in excluded_roots:
            counts["historical_root_excluded"] += 1
            continue
        family = family_classifier(replay, path)
        if family not in GENUINE_FAMILIES:
            counts["unverified_family"] += 1
            continue
        starter_value = replay.get("starter_tile", 1536)
        starter_tile = None if starter_value is None else int(starter_value)
        validator = ThreesSim.from_stream_ids(
            deck_stream_id=2_026_072_611,
            slot_stream_id=2_026_072_612,
            starter_tile=starter_tile,
        )
        source_candidates: list[dict[str, Any]] = []
        source_hygiene_failed = False
        for fallback_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                source_hygiene_failed = True
                counts["malformed_frame"] += 1
                hygiene_failures.append(
                    {
                        "path": str(path),
                        "frame": fallback_index,
                        "root_cluster": root,
                        "error": "frame_not_object",
                    }
                )
                continue
            payload = frame.get("state")
            if not isinstance(payload, dict):
                source_hygiene_failed = True
                counts["malformed_frame_state"] += 1
                hygiene_failures.append(
                    {
                        "path": str(path),
                        "frame": fallback_index,
                        "root_cluster": root,
                        "error": "state_not_object",
                    }
                )
                continue
            if bool(payload.get("game_over")):
                continue
            try:
                state = state_from_replay_payload(payload)
                if not root_option_eligible(state, validator, starter_tile):
                    continue
                pair = geometry(state.board, starter_tile)
                if pair is None:
                    continue
                legal = validator.legal_actions(state)
                expected_names = [DIRECTION_NAMES[action] for action in legal]
                if payload.get("legal_actions") != expected_names:
                    raise ValueError("legal_action_mismatch")
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                source_hygiene_failed = True
                counts["state_restore_or_geometry_failure"] += 1
                hygiene_failures.append(
                    {
                        "path": str(path),
                        "frame": fallback_index,
                        "root_cluster": root,
                        "error": f"{type(error).__name__}:{error}",
                    }
                )
                continue
            frame_index = int(frame.get("index", fallback_index))
            state_hash = state_signature(payload, starter_tile)
            row = {
                "root_cluster": root,
                "behavior_family": family,
                "target_tile": pair.target_tile,
                "scale_band": scale_band(pair.target_tile),
                "stage": pair.stage,
                "stage_name": pair.stage_name,
                "goal": root_goal(pair),
                "source_replay": str(path),
                "source_replay_sha256": source["sha256"],
                "source_frame_index": frame_index,
                "state_sha1": state_hash,
                "starter_tile": starter_tile,
                "pair": [list(coord) for coord in pair.pair],
                "safe_merge_actions": list(pair.safe_merge_actions),
                "empty_count": int(np.count_nonzero(state.board == 0)),
                "legal_count": len(legal),
                "state": _compact_state(payload),
            }
            row["selection_sha256"] = hashlib.sha256(
                "\0".join(
                    (
                        "O1-P0-state-v1",
                        root,
                        str(pair.target_tile),
                        str(pair.stage),
                        str(frame_index),
                        state_hash,
                    )
                ).encode("utf-8")
            ).hexdigest()
            source_candidates.append(row)
            counts["eligible_frames"] += 1
        if source_hygiene_failed:
            counts["data_hygiene_failure_sources"] += 1
            counts["eligible_frames_lost_to_hygiene"] += len(source_candidates)
            disqualified_roots.add(root)
        else:
            candidates_by_root[root].extend(source_candidates)
    for root in disqualified_roots:
        candidates_by_root.pop(root, None)
    counts["data_hygiene_disqualified_roots"] = len(disqualified_roots)
    selected = [
        min(rows, key=lambda row: row["selection_sha256"])
        for rows in candidates_by_root.values()
    ]
    selected.sort(key=lambda row: str(row["root_cluster"]))
    counts["selected_unique_roots"] = len(selected)
    selected_integrity = validate_selected_records(selected)
    support = {
        "by_stage": dict(sorted(Counter(row["stage_name"] for row in selected).items())),
        "by_scale": dict(sorted(Counter(str(row["target_tile"]) for row in selected).items())),
        "by_scale_band": dict(sorted(Counter(row["scale_band"] for row in selected).items())),
        "by_family": dict(sorted(Counter(row["behavior_family"] for row in selected).items())),
        "stage_family": {
            stage: dict(
                sorted(
                    Counter(
                        row["behavior_family"]
                        for row in selected
                        if row["stage_name"] == stage
                    ).items()
                )
            )
            for stage in ("separated", "diagonal_touching", "adjacent", "merge_ready")
        },
    }
    return selected, {
        "counts": dict(sorted(counts.items())),
        "support": support,
        "data_hygiene_failures": hygiene_failures,
        "data_hygiene_failure_count": len(hygiene_failures),
        "selected_integrity": selected_integrity,
        "provenance_field_access": {
            "reset_or_root_score_fields_may_be_read":
                True,
            "purpose":
                "fresh-root/reset provenance validation only",
            "terminal_completion_flag_read":
                True,
            "final_or_future_score_fields_read":
                False,
            "future_milestone_or_terminal_max_tile_fields_read":
                False,
            "score_or_outcome_fields_used_for_selection":
                False,
        },
        "recorded_actions_accessed": False,
        "future_outcomes_accessed": False,
    }


def validate_selected_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    schema = schema_manifest()
    schema_checks = {
        "version_is_a3": schema.get("version", "").endswith("_a3"),
        "safe_full_state_empties": schema.get("minimum_safe_empties") == 2,
        "safe_prespawn_empties":
            schema.get("minimum_safe_prespawn_empties") == 3,
        "action_conditioned_forward":
            schema.get("action_conditioned_forward") is True,
        "pair_specific_tagged_merge":
            schema.get("pair_specific_tagged_merge") is True,
    }
    for row in records:
        try:
            path = Path(str(row["source_replay"]))
            if sha256_path(path) != row["source_replay_sha256"]:
                raise ValueError("selected_source_hash_mismatch")
            payload = dict(row["state"])
            starter_value = row.get("starter_tile")
            starter_tile = (
                None if starter_value is None else int(starter_value)
            )
            state = state_from_replay_payload(payload)
            validator = ThreesSim.from_stream_ids(
                deck_stream_id=2_026_072_611,
                slot_stream_id=2_026_072_612,
                starter_tile=starter_tile,
            )
            if not root_option_eligible(state, validator, starter_tile):
                raise ValueError("selected_state_no_longer_eligible")
            pair = geometry(state.board, starter_tile)
            if pair is None:
                raise ValueError("selected_geometry_missing")
            legal = validator.legal_actions(state)
            expected_names = [DIRECTION_NAMES[action] for action in legal]
            checks = {
                "state_hash":
                    state_signature(payload, starter_tile)
                    == row["state_sha1"],
                "legal_actions":
                    payload.get("legal_actions") == expected_names,
                "legal_count": len(legal) == int(row["legal_count"]),
                "target": pair.target_tile == int(row["target_tile"]),
                "stage": pair.stage == int(row["stage"]),
                "pair":
                    [list(coord) for coord in pair.pair] == row["pair"],
                "safe_merge_actions":
                    list(pair.safe_merge_actions)
                    == row["safe_merge_actions"],
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError(
                    "selected_state_invariant:" + ",".join(failed)
                )
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
            failures.append(
                {
                    "root_cluster": str(row.get("root_cluster", "")),
                    "source_replay": str(row.get("source_replay", "")),
                    "error": f"{type(error).__name__}:{error}",
                }
            )
    return {
        "checked_records": len(records),
        "schema_checks": schema_checks,
        "failure_count": len(failures),
        "failures": failures,
        "passes": all(schema_checks.values()) and not failures,
    }


def _current_nice() -> int:
    return os.getpriority(os.PRIO_PROCESS, 0)


def _requested_stream_rows(max_test_n: int = 1536) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_index in range(240):
        for round_index in range(4):
            for replicate in range(2):
                rows.append(
                    trajectory_stream_ids(
                        "train",
                        root_index,
                        round_index=round_index,
                        replicate=replicate,
                    )
                )
    for partition, count in (("development", 80), ("untouched_test", max_test_n)):
        for root_index in range(count):
            for replicate in range(8):
                for incumbent_arm in (False, True):
                    rows.append(
                        trajectory_stream_ids(
                            partition,
                            root_index,
                            replicate=replicate,
                            incumbent_arm=incumbent_arm,
                        )
                    )
    return rows


def _internal_stream_contract(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    shared_keys = ("logical_seed", "deck_stream_id", "slot_stream_id")
    violations: list[dict[str, Any]] = []
    expected_shared_duplicates: dict[str, int] = {}
    values_by_key: dict[str, set[int]] = {}
    for key in (*shared_keys, "policy_stream_id"):
        groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[int(row[key])].append(row)
        values_by_key[key] = set(groups)
        if key == "policy_stream_id":
            duplicates = [
                value for value, group in groups.items() if len(group) != 1
            ]
            if duplicates:
                violations.append(
                    {
                        "kind": "policy_stream_duplicate",
                        "key": key,
                        "values": duplicates[:20],
                    }
                )
            continue
        duplicate_count = 0
        for value, group in groups.items():
            trajectory_codes = {
                int(row["trajectory_code"]) for row in group
            }
            policy_streams = {
                int(row["policy_stream_id"]) for row in group
            }
            if len(group) == 2:
                duplicate_count += 1
            expected_arms = (
                1
                if len(trajectory_codes) == 1
                and next(iter(trajectory_codes)) < 1_000_000
                else 2
            )
            if (
                len(group) != expected_arms
                or len(trajectory_codes) != 1
                or len(policy_streams) != len(group)
            ):
                violations.append(
                    {
                        "kind": "invalid_shared_crn_multiplicity",
                        "key": key,
                        "value": value,
                        "row_count": len(group),
                        "expected_arms": expected_arms,
                        "trajectory_codes": sorted(trajectory_codes),
                        "policy_stream_ids": sorted(policy_streams),
                    }
                )
        expected_shared_duplicates[key] = duplicate_count
    keys = list(values_by_key)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            overlap = values_by_key[left].intersection(values_by_key[right])
            if overlap:
                violations.append(
                    {
                        "kind": "cross_namespace_collision",
                        "left": left,
                        "right": right,
                        "values": sorted(overlap)[:20],
                    }
                )
    return {
        "passes": not violations,
        "row_count": len(rows),
        "expected_shared_crn_duplicate_counts":
            expected_shared_duplicates,
        "policy_stream_ids_unique":
            len(values_by_key["policy_stream_id"]) == len(rows),
        "cross_namespace_disjoint":
            not any(
                row["kind"] == "cross_namespace_collision"
                for row in violations
            ),
        "violations": violations,
    }


def _stream_collision_audit() -> dict[str, Any]:
    rows = _requested_stream_rows()
    prior, sources = history.historical_collision_union(
        exclude_dir=OUTPUT_DIR
    )
    collisions: dict[str, list[int]] = {}
    for key in STREAM_BASES:
        requested = {int(row[key]) for row in rows}
        prior_values = set(prior.get(key, set()))
        if key == "logical_seed":
            for alias in (
                "seed",
                "root_seed",
                "source_seed",
                "fresh_root_seed",
            ):
                prior_values.update(prior.get(alias, set()))
        collisions[key] = sorted(requested.intersection(prior_values))
    internal = _internal_stream_contract(rows)
    audit = {
        "historical_union": sources,
        "collisions": collisions,
        "zero_collisions":
            internal["passes"] and not any(collisions.values()),
        "internal_stream_contract": internal,
    }
    audit["requested_row_count"] = len(rows)
    audit["requested_rows_sha256"] = canonical_json_hash(rows)
    return audit


def prepare() -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"O1 P0 output already exists: {OUTPUT_DIR}")
    for path, expected in (
        (CHARTER_PATH, CHARTER_SHA256),
        (A1_PATH, A1_SHA256),
        (A2_PATH, A2_SHA256),
        (A3_PATH, A3_SHA256),
        (A4_PATH, A4_SHA256),
        (A5_PATH, A5_SHA256),
        (A6_PATH, A6_SHA256),
    ):
        if sha256_path(path) != expected:
            raise ValueError(f"Frozen O1 charter hash mismatch: {path}")
    if not TEST_EVIDENCE_PATH.is_file():
        raise ValueError("O1 P0 test evidence is missing")
    if not EVIDENCE_TRANSITION_PATH.is_file():
        raise ValueError("O1 P0 A6 evidence transition is missing")

    evidence_transition = verify_evidence_transition()
    inventory = source_path_inventory()
    exclusion = exclusion_manifest()
    family_evidence = verify_family_evidence()
    streams = _stream_collision_audit()
    services = history.service_health()
    heavy = _heavy_process_audit()
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    nice = _current_nice()
    checks = {
        "family_evidence": family_evidence["passes"],
        "evidence_transition": evidence_transition["passes"],
        "stream_collisions_zero": streams["zero_collisions"],
        "heavy_process_clear": heavy["passes"],
        "nice_at_least_10": nice >= 10,
        "free_disk_above_hard_floor": free_gib >= MIN_FREE_GIB,
        "free_disk_above_target": free_gib >= TARGET_FREE_GIB,
        "services": services["passes"],
        "dashboard_top_three":
            tuple(services["dashboard_top_scores"][:3]) == EXPECTED_TOP_THREE,
        "candidate_json_not_parsed": not inventory["candidate_json_parsed"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"O1 P0 prepare checks failed: {checks}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_immutable_json(SOURCE_INVENTORY_PATH, inventory)
    write_immutable_json(EXCLUSION_PATH, exclusion)
    marker = {
        "version": "o1_p0_content_opened_marker_v1_a6",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "content_opened": True,
        "candidate_replay_json_parsed_before_marker": False,
        "source_inventory": artifact_identity(SOURCE_INVENTORY_PATH),
        "exclusion_manifest": artifact_identity(EXCLUSION_PATH),
        "charter": {"path": str(CHARTER_PATH), "sha256": CHARTER_SHA256},
        "amendments": {
            "a1": {"path": str(A1_PATH), "sha256": A1_SHA256},
            "a2": {"path": str(A2_PATH), "sha256": A2_SHA256},
            "a3": {"path": str(A3_PATH), "sha256": A3_SHA256},
            "a4": {"path": str(A4_PATH), "sha256": A4_SHA256},
            "a5": {"path": str(A5_PATH), "sha256": A5_SHA256},
            "a6": {"path": str(A6_PATH), "sha256": A6_SHA256},
        },
        "implementation": {
            "geometry": {"path": str(GEOMETRY_PATH), "sha256": sha256_path(GEOMETRY_PATH)},
            "runner": {"path": str(RUNNER_PATH), "sha256": sha256_path(RUNNER_PATH)},
            "tests": {"path": str(TEST_PATH), "sha256": sha256_path(TEST_PATH)},
            "test_evidence": artifact_identity(TEST_EVIDENCE_PATH),
            "evidence_transition":
                evidence_transition["transition"],
            "schema": schema_manifest(),
            "schema_sha256": schema_sha256(),
            "model_parameter_count": sum(
                parameter.numel() for parameter in O1OptionNet().parameters()
            ),
        },
        "family_evidence": family_evidence,
        "future_streams": {
            "bases": STREAM_BASES,
            "consumed": False,
            "collision_audit": streams,
        },
        "operations": {
            "nice": nice,
            "free_gib": free_gib,
            "services": services,
            "heavy_process": heavy,
            "checks": checks,
        },
        "forbidden_work": {
            "new_rollouts": 0,
            "new_labels": 0,
            "model_fits": 0,
            "training_outcomes": 0,
            "policy_outcomes": 0,
            "candidate_content_score_or_outcome_inspection_before_marker": 0,
            "candidate_actions": 0,
            "streams_consumed": 0,
            "dashboard_changes": 0,
        },
    }
    write_immutable_json(MARKER_PATH, marker)
    return artifact_identity(MARKER_PATH)


def _revalidate_marker() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    marker = json.loads(MARKER_PATH.read_text())
    if not verify_payload_hash(marker):
        raise ValueError("O1 content marker payload mismatch")
    if RESULT_PATH.exists() or ROOT_MANIFEST_PATH.exists():
        raise FileExistsError("O1 P0 scan already opened or completed")
    for key, path in (
        ("source_inventory", SOURCE_INVENTORY_PATH),
        ("exclusion_manifest", EXCLUSION_PATH),
    ):
        identity = artifact_identity(path)
        if identity != marker[key]:
            raise ValueError(f"O1 marker-bound artifact changed: {key}")
    for key, path in (
        ("geometry", GEOMETRY_PATH),
        ("runner", RUNNER_PATH),
        ("tests", TEST_PATH),
    ):
        if sha256_path(path) != marker["implementation"][key]["sha256"]:
            raise ValueError(f"O1 implementation changed after marker: {key}")
    for key, path in (
        ("test_evidence", TEST_EVIDENCE_PATH),
        ("evidence_transition", EVIDENCE_TRANSITION_PATH),
    ):
        if artifact_identity(path) != marker["implementation"][key]:
            raise ValueError(f"O1 evidence changed after marker: {key}")
    streams = _stream_collision_audit()
    if (
        not streams["zero_collisions"]
        or streams["requested_rows_sha256"]
        != marker["future_streams"]["collision_audit"]["requested_rows_sha256"]
    ):
        raise ValueError("O1 requested stream collision audit changed")
    services = history.service_health()
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    heavy = _heavy_process_audit()
    if (
        not services["passes"]
        or tuple(services["dashboard_top_scores"][:3]) != EXPECTED_TOP_THREE
        or free_gib < TARGET_FREE_GIB
        or not heavy["passes"]
        or _current_nice() < 10
    ):
        raise RuntimeError("O1 scan operational revalidation failed")
    return marker, json.loads(SOURCE_INVENTORY_PATH.read_text()), json.loads(EXCLUSION_PATH.read_text())


def scan() -> dict[str, Any]:
    marker, inventory, exclusion = _revalidate_marker()
    records, scan_report = scan_candidate_content(inventory, exclusion)
    power = power_table()
    assigned, allocation = allocate_partitions(
        records,
        power["selected_smallest_passing_n"],
    )
    assigned_roots = {str(row["root_cluster"]) for row in assigned}
    root_manifest = {
        "version": "o1_p0_root_manifest_v1_a6",
        "selection_rule":
            'argmin SHA256("O1-P0-state-v1"|root|target|stage|frame|state_hash)',
        "records": [
            {
                **row,
                "partition": next(
                    (
                        assigned_row["partition"]
                        for assigned_row in assigned
                        if assigned_row["root_cluster"] == row["root_cluster"]
                    ),
                    "inventory_only",
                ),
            }
            for row in records
        ],
        "selected_root_count": len(records),
        "assigned_root_count": len(assigned_roots),
        "records_sha256": canonical_json_hash(records),
        "scan_report": scan_report,
        "allocation": allocation,
        "one_state_per_root":
            len(records)
            == len({str(row["root_cluster"]) for row in records}),
        "selection_field_contract": {
            "reset_or_root_score_fields_may_be_read_for_provenance":
                True,
            "terminal_completion_flag_read_for_completeness":
                True,
            "final_or_future_score_fields_read":
                False,
            "future_milestone_or_terminal_max_tile_fields_read":
                False,
            "recorded_actions_read":
                False,
            "score_action_or_outcome_fields_used_for_selection":
                False,
        },
    }
    write_immutable_json(ROOT_MANIFEST_PATH, root_manifest)

    integrity = {
        "marker_valid": verify_payload_hash(marker),
        "root_manifest_valid":
            verify_payload_hash(json.loads(ROOT_MANIFEST_PATH.read_text())),
        "one_state_per_root": root_manifest["one_state_per_root"],
        "selected_state_failures_zero":
            scan_report["selected_integrity"]["passes"],
        "data_hygiene_failures_are_support_loss_only": True,
        "family_evidence": marker["family_evidence"]["passes"],
        "streams_unconsumed": not marker["future_streams"]["consumed"],
        "schema_exact": marker["implementation"]["schema_sha256"] == schema_sha256(),
    }
    if not all(integrity.values()):
        decision = "KILL_O1_REPRESENTATION_PREFLIGHT"
    elif not allocation.get("passes") or power["selected_smallest_passing_n"] is None:
        decision = "HOLD_O1_DATA_OR_POWER"
    else:
        decision = "READY_O1_E0_PILOT"

    services = history.service_health()
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    result = {
        "version": RESULT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "charter_sha256": CHARTER_SHA256,
        "amendment_sha256": {
            "a1": A1_SHA256,
            "a2": A2_SHA256,
            "a3": A3_SHA256,
            "a4": A4_SHA256,
            "a5": A5_SHA256,
            "a6": A6_SHA256,
        },
        "marker": artifact_identity(MARKER_PATH),
        "source_inventory": artifact_identity(SOURCE_INVENTORY_PATH),
        "exclusion_manifest": artifact_identity(EXCLUSION_PATH),
        "root_manifest": artifact_identity(ROOT_MANIFEST_PATH),
        "implementation": marker["implementation"],
        "power": power,
        "natural_support": scan_report["support"],
        "scan_counts": scan_report["counts"],
        "partition_allocation": allocation,
        "integrity": integrity,
        "operations": {
            "free_gib": free_gib,
            "services": services,
            "dashboard_top_three": services["dashboard_top_scores"][:3],
        },
        "training_or_policy_outcomes_opened": False,
        "e0_opened": False,
        "forbidden_work": {
            "new_rollouts": 0,
            "new_labels": 0,
            "model_fits": 0,
            "training_outcomes": 0,
            "policy_outcomes": 0,
            "final_or_future_score_outcome_inspection": 0,
            "reset_root_score_provenance_access":
                "permitted_and_used_only_for_fresh_root_validation",
            "recorded_actions_inspected": 0,
            "streams_consumed": 0,
            "dashboard_changes": 0,
        },
        "states": {
            "CONTINUE": (
                "O1_E0_PILOT"
                if decision == "READY_O1_E0_PILOT"
                else "none"
            ),
            "HOLD": [
                "human_training_ground",
                "normal_start_policy_evaluation",
                "promotion",
            ],
            "KILL": [
                "exact_depth3_program",
                *(
                    ["o1_representation"]
                    if decision == "KILL_O1_REPRESENTATION_PREFLIGHT"
                    else []
                ),
            ],
            "PROMOTE": False,
        },
    }
    write_immutable_json(RESULT_PATH, result)
    return artifact_identity(RESULT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "scan"))
    args = parser.parse_args()
    payload = prepare() if args.command == "prepare" else scan()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

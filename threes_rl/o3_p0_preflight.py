"""One-shot, outcome-free O3 feasibility and integrity preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from threes_rl import g1r_acquire as history
from threes_rl import g1r_acquire_v2_qd5 as qd5
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.o3_designated_pair_option import (
    O3DesignatedPairNet,
    schema_sha256,
)
from threes_rl.o3_power_contract import (
    ODDS_RATIO_GRID,
    POWER_REQUIRED,
    ROOT_CANDIDATES,
    simulate_mechanism_power,
)
from threes_rl.s3_power_preflight import sha256_path


VERSION = "o3_event_option_p0_v1"
ROOT = Path(".")
CHARTER_PATH = Path(
    "threes_rl/O3_EVENT_CONDITIONED_DESIGNATED_PAIR_CHARTER.md"
)
OPTION_PATH = Path("threes_rl/o3_designated_pair_option.py")
POWER_PATH = Path("threes_rl/o3_power_contract.py")
RUNNER_PATH = Path("threes_rl/o3_p0_preflight.py")
OPTION_TEST_PATH = Path("tests/test_rl_o3_designated_pair_option.py")
TEST_PATH = Path("tests/test_rl_o3_p0_preflight.py")
LEDGER_PATH = Path("threes_rl/CURRENT_DECISION_LEDGER.md")
LOG_PATH = Path("threes_rl/EXPERIMENT_LOG.md")
RETENTION_PATH = Path("threes_rl/ARTIFACT_RETENTION.md")
OUTPUT_DIR = Path("threes_rl/runs/forensics/o3_event_option_p0_v1")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/o3_event_option_p0_test_evidence.json"
)
MARKER_NAME = "O3_P0_OPENED.json"
RESULT_NAME = "O3_P0_RESULT.json"
STREAM_MANIFEST_NAME = "O3_P0_STREAM_MANIFEST.json"
PARTITION_MANIFEST_NAME = "O3_P0_PARTITION_PLAN.json"
COLLISION_MANIFEST_NAME = "O3_P0_COLLISION_SOURCES.json"
POWER_MANIFEST_NAME = "O3_P0_POWER_TABLE.json"
POLICY_MANIFEST_NAME = "O3_P0_POLICY_AUDIT.json"

FROZEN_HASHES = {
    str(CHARTER_PATH): (
        "26a117cd5b14a32e79e4a63c63b0fb707f34135193d587880c157ef8c11f4441"
    ),
    str(OPTION_PATH): (
        "659475fe596a9e96aa56e3fc4bbaf57bbfdfbefa5a569b1c1e24ce8f345064fd"
    ),
    str(POWER_PATH): (
        "9d4ea9eb28c0f929ea7c42d1e1fe1f5ec8e9ab36d170dff571b13cddb102dd38"
    ),
    str(OPTION_TEST_PATH): (
        "a72dfe0b9e6b76c125c00eaec29a7e541c7435dbb35928b49d01c5eb663bef19"
    ),
}
FROZEN_SCHEMA_SHA256 = (
    "a1c2efa6bd980d32138fb6026c1a5109685db8f1630e1b5fa732b2c2eb983602"
)
FROZEN_PARAMETER_COUNT = 102_557
G1R_LOCK_PATH = qd5.OUTPUT_DIR / "preflight_lock.json"
G1R_LOCK_FILE_SHA256 = (
    "0d50edaae52e9a6f6291c4b397fd03c9d7d8651b28bb9dbd05b53c8718ee22ad"
)
G1R_LOCK_PAYLOAD_SHA256 = (
    "1a0ca85b4115f220d0d7c857bde912be8570cf0b2e72d055e6cd88b285227e67"
)
G1R_ACTION_AUDIT_SHA256 = (
    "cc747bead64edfd3820f4547bc629e764339f3917e4e0a62ca71ba0979d0635d"
)
G1R_POLICY_LOCK_SHA256 = (
    "6b0384d9fedfc8f560853a050c28750194ec9c9d3d36cf2d9d7fd47a9a423ea0"
)
O2_SUPPORT_PATH = Path(
    "threes_rl/runs/forensics/o2_yield_pilot_scan_recovery_v1/"
    "O2_RECOVERED_SUPPORT.json"
)
O2_SUPPORT_FILE_SHA256 = (
    "a956d13d1366dc3ca343e84a49145367d7edab0d63c7b5b00a75aa090d64a1f9"
)

O3_FAMILY_ORDER = (
    "o3_corner2",
    "o3_expectimax2",
    "o3_parent_mc1000",
    "o3_replaycal",
    "o3_qd_v2",
)
G1R_FAMILY_ORDER = tuple(family for family, _spec in qd5.FAMILY_SLATE)
O3_TO_G1R = dict(zip(O3_FAMILY_ORDER, G1R_FAMILY_ORDER, strict=True))
EXPECTED_SIGNATURES = {
    o3_family: qd5.EXPECTED_SIGNATURES[g1r_family]
    for o3_family, g1r_family in O3_TO_G1R.items()
}
COLLECTOR_COUNT = 5
ROOTS_PER_FAMILY = 4_100
ACQUISITION_ROOTS = COLLECTOR_COUNT * ROOTS_PER_FAMILY
ROLE_RANGES = {
    "train": (0, 1_004),
    "development": (1_004, 1_339),
    "untouched_mechanism": (1_339, 4_100),
}
ROLE_COUNTS = {
    role: COLLECTOR_COUNT * (stop - start)
    for role, (start, stop) in ROLE_RANGES.items()
}
SELECTED_COUNTS = {
    "train": 96,
    "development": 32,
    "untouched_mechanism": 192,
}
TARGET_SELECTED_COUNTS = {
    "train": {48: 48, 96: 29, 192: 19},
    "development": {48: 16, 96: 10, 192: 6},
    "untouched_mechanism": {48: 96, 96: 58, 192: 38},
}
STREAM_BASES = {
    "acquisition": {
        "logical_seed": 105_000_000_000,
        "deck_stream_id": 106_000_000_000,
        "slot_stream_id": 107_000_000_000,
        "policy_stream_id": 108_000_000_000,
    },
    "learning": {
        "logical_seed": 109_000_000_000,
        "deck_stream_id": 110_000_000_000,
        "slot_stream_id": 111_000_000_000,
        "policy_stream_id": 112_000_000_000,
    },
    "option": {
        "logical_seed": 113_000_000_000,
        "deck_stream_id": 114_000_000_000,
        "slot_stream_id": 115_000_000_000,
        "policy_stream_id": 116_000_000_000,
    },
    "normal_development": {
        "logical_seed": 117_000_000_000,
        "deck_stream_id": 118_000_000_000,
        "slot_stream_id": 119_000_000_000,
        "policy_stream_id": 120_000_000_000,
    },
    "confirmation": {
        "logical_seed": 121_000_000_000,
        "deck_stream_id": 122_000_000_000,
        "slot_stream_id": 123_000_000_000,
        "policy_stream_id": 124_000_000_000,
    },
}
OPTION_PARTITION_OFFSETS = {
    "development": 0,
    "untouched_mechanism": 100_000,
}
NORMAL_DEVELOPMENT_ROOTS = 512
CONFIRMATION_ROOTS = 2_560
REPEATS = 8
MINIMUM_NICE = 10
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
ACQUISITION_ACTIVE_SECONDS = 144 * 3_600
ACQUISITION_BYTE_LIMIT = 28 * 1024**3
EXPECTED_POWER = {
    (192, 1.25): (0.4912109375, 0.4482421875),
    (192, 1.50): (0.9267578125, 0.9169921875),
    (264, 1.25): (0.64453125, 0.482421875),
    (264, 1.50): (0.9765625, 0.953125),
}
STREAM_FIELDS = (
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
)
O2_FORBIDDEN_CONTENT_DIR = Path(
    "threes_rl/runs/forensics/o2_yield_pilot_v1/source_replays"
)


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


def frozen_hash_audit() -> dict[str, Any]:
    rows = {
        path: {
            "expected_sha256": expected,
            "actual_sha256": sha256_path(Path(path)),
        }
        for path, expected in FROZEN_HASHES.items()
    }
    for row in rows.values():
        row["matches"] = row["actual_sha256"] == row["expected_sha256"]
    checks = {
        "all_frozen_files_exact": all(row["matches"] for row in rows.values()),
        "schema_exact": schema_sha256() == FROZEN_SCHEMA_SHA256,
        "parameter_count_exact": sum(
            parameter.numel()
            for parameter in O3DesignatedPairNet().parameters()
        )
        == FROZEN_PARAMETER_COUNT,
    }
    return {
        "files": rows,
        "schema_sha256": schema_sha256(),
        "parameter_count": sum(
            parameter.numel()
            for parameter in O3DesignatedPairNet().parameters()
        ),
        "checks": checks,
        "passes": all(checks.values()),
    }


def o2_aggregate_evidence(
    *,
    ledger_path: Path = LEDGER_PATH,
    log_path: Path = LOG_PATH,
    support_path: Path = O2_SUPPORT_PATH,
) -> dict[str, Any]:
    ledger = ledger_path.read_text()
    log = log_path.read_text()
    ledger_tokens = (
        "**Decision: `HOLD_O2_DATA_SUPPORT`.**",
        "`128` unique ancestries/replay",
        "`T192 5/5/0/9`",
        "`T96 9/9/0/9`",
        "`T48 9/9/2/9`",
        "`7/20`",
    )
    log_tokens = (
        "`7,192` frames",
        "`267` permitted",
        "`128` unique",
        "`32` roots per family",
        "`T192 5/5/0/9`",
        "`T96 9/9/0/9`",
        "`T48 9/9/2/9`",
        "`7/20`",
    )
    support_sha = sha256_path(support_path)
    facts = {
        "roots": 128,
        "roots_per_family": 32,
        "families": 4,
        "frames": 7_192,
        "candidates": 267,
        "availability_cells_passed": 7,
        "availability_cells_total": 20,
        "credited_stage_counts": {
            "T768": [0, 0, 0, 0],
            "T384": [4, 4, 0, 3],
            "T192": [5, 5, 0, 9],
            "T96": [9, 9, 0, 9],
            "T48": [9, 9, 2, 9],
        },
        "weakest_hard_start_support": {
            "target": 192,
            "successes": 5,
            "roots": 128,
            "wilson_lower": 0.01914143013104029,
        },
    }
    checks = {
        "ledger_aggregate_facts_exact": all(
            token in ledger for token in ledger_tokens
        ),
        "log_aggregate_facts_exact": all(token in log for token in log_tokens),
        "support_json_byte_hash_exact": support_sha
        == O2_SUPPORT_FILE_SHA256,
        "support_json_content_not_parsed": True,
        "o2_replay_content_not_read": True,
    }
    checks["only_aggregate_and_byte_hash_used"] = (
        checks["ledger_aggregate_facts_exact"]
        and checks["log_aggregate_facts_exact"]
        and checks["support_json_byte_hash_exact"]
        and checks["support_json_content_not_parsed"]
        and checks["o2_replay_content_not_read"]
    )
    return {
        "ledger": {
            "path": str(ledger_path),
            "sha256": sha256_path(ledger_path),
        },
        "experiment_log": {
            "path": str(log_path),
            "sha256": sha256_path(log_path),
        },
        "support_json": {
            "path": str(support_path),
            "file_sha256": support_sha,
            "handling": "byte_hash_only",
        },
        "facts": facts,
        "facts_sha256": canonical_json_hash(facts),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _planned_root_id(
    family: str,
    family_index: int,
    game_index: int,
    logical_seed: int,
) -> str:
    return hashlib.sha256(
        (
            "O3-event-acquisition-root-v1|"
            f"{family}|{family_index}|{game_index}|{logical_seed}"
        ).encode("ascii")
    ).hexdigest()


def acquisition_rows() -> list[dict[str, Any]]:
    bases = STREAM_BASES["acquisition"]
    rows = []
    for family_index, family in enumerate(O3_FAMILY_ORDER):
        for game_index in range(ROOTS_PER_FAMILY):
            code = family_index * ROOTS_PER_FAMILY + game_index
            role = next(
                role
                for role, (start, stop) in ROLE_RANGES.items()
                if start <= game_index < stop
            )
            logical_seed = bases["logical_seed"] + code
            rows.append(
                {
                    "purpose": "acquisition",
                    "family": family,
                    "family_index": family_index,
                    "game_index": game_index,
                    "role": role,
                    "planned_root_id": _planned_root_id(
                        family,
                        family_index,
                        game_index,
                        logical_seed,
                    ),
                    **{
                        field: base + code
                        for field, base in bases.items()
                    },
                }
            )
    return rows


def _single_arm_rows() -> list[dict[str, Any]]:
    bases = STREAM_BASES["learning"]
    rows = []
    for root_index in range(SELECTED_COUNTS["train"]):
        for round_index in range(4):
            for replicate in range(3):
                code = root_index * 12 + round_index * 3 + replicate
                rows.append(
                    {
                        "purpose": "learning",
                        "root_index": root_index,
                        "round_index": round_index,
                        "replicate": replicate,
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
    root_count: int,
    bases: Mapping[str, int],
    repeats: int,
    code_offset: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for root_index in range(root_count):
        for replicate in range(repeats):
            code = code_offset + root_index * repeats + replicate
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
    rows = acquisition_rows()
    rows.extend(_single_arm_rows())
    rows.extend(
        _paired_rows(
            purpose="option_development",
            root_count=SELECTED_COUNTS["development"],
            bases=STREAM_BASES["option"],
            repeats=REPEATS,
            code_offset=1_000_000
            + OPTION_PARTITION_OFFSETS["development"],
        )
    )
    rows.extend(
        _paired_rows(
            purpose="option_untouched_mechanism",
            root_count=SELECTED_COUNTS["untouched_mechanism"],
            bases=STREAM_BASES["option"],
            repeats=REPEATS,
            code_offset=1_000_000
            + OPTION_PARTITION_OFFSETS["untouched_mechanism"],
        )
    )
    rows.extend(
        _paired_rows(
            purpose="normal_development",
            root_count=NORMAL_DEVELOPMENT_ROOTS,
            bases=STREAM_BASES["normal_development"],
            repeats=1,
        )
    )
    rows.extend(
        _paired_rows(
            purpose="confirmation",
            root_count=CONFIRMATION_ROOTS,
            bases=STREAM_BASES["confirmation"],
            repeats=1,
        )
    )
    return rows


def _stream_sets(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {
        field: set() for field in STREAM_FIELDS
    }
    for row in rows:
        for field in STREAM_FIELDS[:3]:
            if field in row:
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


def partition_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acquisition = [
        row for row in rows if row["purpose"] == "acquisition"
    ]
    role_counts = Counter(str(row["role"]) for row in acquisition)
    per_role_family = {
        role: Counter(
            str(row["family"])
            for row in acquisition
            if row["role"] == role
        )
        for role in ROLE_RANGES
    }
    root_ids = [str(row["planned_root_id"]) for row in acquisition]
    checks = {
        "exact_20500_acquisition_roots": len(acquisition)
        == ACQUISITION_ROOTS,
        "exact_equal_family_counts": Counter(
            str(row["family"]) for row in acquisition
        )
        == {family: ROOTS_PER_FAMILY for family in O3_FAMILY_ORDER},
        "role_counts_exact": dict(role_counts) == ROLE_COUNTS,
        "every_root_one_role": len(root_ids) == len(set(root_ids)),
        "family_share_20pct_each_role": all(
            set(counts) == set(O3_FAMILY_ORDER)
            and len(set(counts.values())) == 1
            for counts in per_role_family.values()
        ),
        "selected_counts_exact": sum(SELECTED_COUNTS.values()) == 320,
        "target_selected_counts_exact": all(
            sum(counts.values()) == SELECTED_COUNTS[role]
            for role, counts in TARGET_SELECTED_COUNTS.items()
        ),
        "mechanism_n192": SELECTED_COUNTS["untouched_mechanism"] == 192,
    }
    compact = [
        {
            "family": row["family"],
            "family_index": row["family_index"],
            "game_index": row["game_index"],
            "role": row["role"],
            "planned_root_id": row["planned_root_id"],
            "logical_seed": row["logical_seed"],
        }
        for row in acquisition
    ]
    return {
        "root_universe_count": len(acquisition),
        "roots_per_family": ROOTS_PER_FAMILY,
        "family_order": list(O3_FAMILY_ORDER),
        "role_ranges": {
            role: {"start": start, "stop_exclusive": stop}
            for role, (start, stop) in ROLE_RANGES.items()
        },
        "role_counts": dict(role_counts),
        "per_role_family_counts": {
            role: dict(counts)
            for role, counts in per_role_family.items()
        },
        "selected_counts": SELECTED_COUNTS,
        "target_selected_counts": TARGET_SELECTED_COUNTS,
        "planned_root_manifest_sha256": canonical_json_hash(compact),
        "checks": checks,
        "passes": all(checks.values()),
    }


def stream_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    purpose_counts = Counter(str(row["purpose"]) for row in rows)
    values = _stream_sets(rows)
    acquisition = [
        row for row in rows if row["purpose"] == "acquisition"
    ]
    paired = [
        row
        for row in rows
        if "control_policy_stream_id" in row
    ]
    checks = {
        "purpose_counts_exact": purpose_counts
        == {
            "acquisition": 20_500,
            "learning": 1_152,
            "option_development": 256,
            "option_untouched_mechanism": 1_536,
            "normal_development": 512,
            "confirmation": 2_560,
        },
        "acquisition_streams_exact": all(
            int(row[field])
            == STREAM_BASES["acquisition"][field]
            + int(row["family_index"]) * ROOTS_PER_FAMILY
            + int(row["game_index"])
            for row in acquisition
            for field in STREAM_FIELDS
        ),
        "paired_exogenous_rows_unique": all(
            len(
                {
                    (
                        row["logical_seed"],
                        row["deck_stream_id"],
                        row["slot_stream_id"],
                    )
                    for row in paired
                }
            )
            == len(paired)
            for _once in (0,)
        ),
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
            len(items)
            == (
                sum(
                    2 if "control_policy_stream_id" in row else 1
                    for row in rows
                )
                if field == "policy_stream_id"
                else len(rows)
            )
            for field, items in values.items()
        ),
        "paired_crn_semantics_explicit": all(
            "logical_seed" in row
            and "deck_stream_id" in row
            and "slot_stream_id" in row
            and row["control_policy_stream_id"]
            != row["treatment_policy_stream_id"]
            for row in paired
        ),
        "streams_unconsumed": True,
    }
    return {
        "row_count": len(rows),
        "purpose_counts": dict(purpose_counts),
        "stream_value_counts": {
            field: len(items) for field, items in values.items()
        },
        "stream_bases": STREAM_BASES,
        "option_partition_offsets": OPTION_PARTITION_OFFSETS,
        "manifest_sha256": canonical_json_hash(rows),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def collision_audit(
    rows: list[dict[str, Any]],
    *,
    scan_root: Path = Path("threes_rl/runs"),
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    requested = _stream_sets(rows)
    found: dict[str, set[int]] = defaultdict(set)
    matched = []
    excluded = []
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        if _is_within(path, out_dir):
            excluded.append(
                {
                    "path": str(path),
                    "classification": "current_o3_p0_namespace",
                    "bytes": int(path.stat().st_size),
                }
            )
            continue
        if _is_within(path, O2_FORBIDDEN_CONTENT_DIR):
            excluded.append(
                {
                    "path": str(path),
                    "classification": "o2_replay_content_forbidden_unread",
                    "bytes": int(path.stat().st_size),
                }
            )
            continue
        if path.resolve() == O2_SUPPORT_PATH.resolve():
            excluded.append(
                {
                    "path": str(path),
                    "classification": "o2_support_byte_hash_only",
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_path(path),
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
    checks = {
        "zero_historical_collisions": not any(collisions.values()),
        "o2_support_excluded_as_byte_hash_only": any(
            row["classification"] == "o2_support_byte_hash_only"
            for row in excluded
        ),
        "o2_replays_excluded_unread": any(
            row["classification"] == "o2_replay_content_forbidden_unread"
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
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
    }


def power_contract() -> dict[str, Any]:
    rows = [
        simulate_mechanism_power(roots, odds_ratio)
        for roots in ROOT_CANDIDATES
        for odds_ratio in ODDS_RATIO_GRID
    ]
    lookup = {
        (int(row["roots"]), float(row["true_common_odds_ratio"])): row
        for row in rows
    }
    selected = next(
        (
            roots
            for roots in ROOT_CANDIDATES
            if lookup[(roots, 1.50)]["power_full_gate"] >= POWER_REQUIRED
        ),
        None,
    )
    mde = (
        next(
            (
                odds_ratio
                for odds_ratio in ODDS_RATIO_GRID
                if lookup[(selected, odds_ratio)]["power_full_gate"]
                >= POWER_REQUIRED
            ),
            None,
        )
        if selected is not None
        else None
    )
    checks = {
        "frozen_rows_reproduce": all(
            (
                lookup[key]["power_lower_ci_gt_1"],
                lookup[key]["power_full_gate"],
            )
            == expected
            for key, expected in EXPECTED_POWER.items()
        ),
        "selected_n192": selected == 192,
        "or150_full_gate_power_at_least_80pct": (
            selected is not None
            and lookup[(selected, 1.50)]["power_full_gate"]
            >= POWER_REQUIRED
        ),
        "grid_mde_or150": mde == 1.50,
        "pass_point_gate_or125": all(
            row["point_gate_odds_ratio"] == 1.25 for row in rows
        ),
    }
    return {
        "rows": rows,
        "rows_sha256": canonical_json_hash(rows),
        "selected_roots": selected,
        "grid_mde": mde,
        "checks": checks,
        "passes": all(checks.values()),
    }


def policy_audit() -> dict[str, Any]:
    if sha256_path(G1R_LOCK_PATH) != G1R_LOCK_FILE_SHA256:
        raise ValueError("Immutable G1-R QD5 lock changed")
    prior_lock = json.loads(G1R_LOCK_PATH.read_text())
    prior_payload = prior_lock.pop("preflight_payload_sha256")
    if prior_payload != G1R_LOCK_PAYLOAD_SHA256:
        raise ValueError("G1-R QD5 payload identity changed")
    if canonical_json_hash(prior_lock) != prior_payload:
        raise ValueError("G1-R QD5 payload hash mismatch")
    if (
        prior_lock["action_signature_audit"]["audit_sha256"]
        != G1R_ACTION_AUDIT_SHA256
    ):
        raise ValueError("G1-R action audit identity changed")
    if prior_lock["policy_lock"]["policy_lock_sha256"] != G1R_POLICY_LOCK_SHA256:
        raise ValueError("G1-R policy lock identity changed")

    current_lock, policies = qd5._policy_lock()
    panel, panel_source = qd5._load_panel()
    signatures = qd5.action_signature_audit(policies, panel)
    renamed = {
        o3_family: signatures["signature_sha256"][g1r_family]
        for o3_family, g1r_family in O3_TO_G1R.items()
    }
    pairwise = []
    for row in signatures["pairwise"]:
        pairwise.append(
            {
                **row,
                "left": O3_FAMILY_ORDER[
                    G1R_FAMILY_ORDER.index(str(row["left"]))
                ],
                "right": O3_FAMILY_ORDER[
                    G1R_FAMILY_ORDER.index(str(row["right"]))
                ],
            }
        )
    checks = {
        "prior_lock_file_exact": True,
        "prior_lock_payload_exact": True,
        "current_policy_lock_exact": current_lock["policy_lock_sha256"]
        == G1R_POLICY_LOCK_SHA256,
        "five_policies_loaded": len(policies) == COLLECTOR_COUNT,
        "o3_family_order_exact": tuple(renamed) == O3_FAMILY_ORDER,
        "signatures_exact": renamed == EXPECTED_SIGNATURES,
        "pairwise_distinctness_reproduced": signatures["passes"],
        "all_pairwise_gates_pass": all(row["passes"] for row in pairwise),
        "no_timing_performed": True,
    }
    return {
        "family_order": list(O3_FAMILY_ORDER),
        "o3_to_g1r_identity": O3_TO_G1R,
        "policy_lock": current_lock,
        "policy_lock_sha256": current_lock["policy_lock_sha256"],
        "panel_source": panel_source,
        "signature_sha256": renamed,
        "pairwise": pairwise,
        "tie_state_counts": {
            O3_FAMILY_ORDER[G1R_FAMILY_ORDER.index(family)]: count
            for family, count in signatures["tie_state_counts"].items()
        },
        "source_action_audit_sha256": signatures["audit_sha256"],
        "checks": checks,
        "passes": all(checks.values()),
    }


def _load_test_evidence() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not _verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise ValueError("O3 P0 test evidence payload mismatch")
    expected = {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "option_implementation_sha256": sha256_path(OPTION_PATH),
        "power_implementation_sha256": sha256_path(POWER_PATH),
        "p0_implementation_sha256": sha256_path(RUNNER_PATH),
        "option_tests_sha256": sha256_path(OPTION_TEST_PATH),
        "p0_tests_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("O3 P0 test evidence source identity mismatch")
    if not payload.get("passes"):
        raise ValueError("O3 P0 test evidence did not pass")
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
        "focused_tests_passed": int(focused_passed),
        "regression_tests_passed": int(regression_passed),
        "commands": commands,
        "passes": True,
        "games_generated": 0,
        "streams_consumed": 0,
        "labels_generated": 0,
        "models_fit": 0,
        "outcomes_inspected": False,
        "o2_row_level_content_read": False,
    }
    return _write_immutable_json(
        TEST_EVIDENCE_PATH,
        payload,
        self_hash_field="test_evidence_payload_sha256",
    )


def _bound_commands(out_dir: Path) -> dict[str, str]:
    base = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.o3_p0_preflight"
    )
    suffix = f" --out-dir {out_dir}'"
    return {
        "open": f"{base} open{suffix}",
        "run": f"{base} run{suffix}",
    }


def _marker_identity(out_dir: Path) -> dict[str, Any]:
    tests = _load_test_evidence()
    return {
        "version": VERSION,
        "bound_out_dir": str(out_dir.resolve()),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "option_implementation_sha256": sha256_path(OPTION_PATH),
        "power_implementation_sha256": sha256_path(POWER_PATH),
        "p0_implementation_sha256": sha256_path(RUNNER_PATH),
        "option_tests_sha256": sha256_path(OPTION_TEST_PATH),
        "p0_tests_sha256": sha256_path(TEST_PATH),
        "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "test_evidence_payload_sha256": tests[
            "test_evidence_payload_sha256"
        ],
        "schema_sha256": schema_sha256(),
        "g1r_lock_file_sha256": sha256_path(G1R_LOCK_PATH),
        "g1r_lock_payload_sha256": G1R_LOCK_PAYLOAD_SHA256,
        "o2_support_file_sha256_byte_only": sha256_path(O2_SUPPORT_PATH),
        "family_order": list(O3_FAMILY_ORDER),
        "expected_signatures": EXPECTED_SIGNATURES,
        "acquisition_roots": ACQUISITION_ROOTS,
        "roots_per_family": ROOTS_PER_FAMILY,
        "stream_bases": STREAM_BASES,
        "commands": _bound_commands(out_dir),
    }


def open_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    bound = out_dir.resolve()
    if bound != OUTPUT_DIR.resolve():
        raise ValueError("O3 P0 output directory is immutable")
    if out_dir.exists():
        raise FileExistsError(f"O3 P0 namespace already exists: {out_dir}")
    if history.current_nice() < MINIMUM_NICE:
        raise ValueError("O3 P0 requires nice >=10")
    hashes = frozen_hash_audit()
    tests = _load_test_evidence()
    heavy = _heavy_process_audit()
    services = history.service_health()
    free_gib = shutil.disk_usage(out_dir.parent).free / 1024**3
    checks = {
        "frozen_hashes_exact": hashes["passes"],
        "tests_exact": bool(tests["passes"]),
        "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
        "no_competing_heavy_process": heavy["passes"],
        "free_disk_above_120_gib": free_gib > TARGET_FREE_GIB,
        "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
        "services_dashboard_top_three": services["passes"],
        "zero_prior_o3_p0_namespace": True,
        "zero_games_streams_labels_models_outcomes": True,
    }
    if not all(checks.values()):
        raise ValueError(f"O3 P0 open checks failed: {checks}")
    marker = {
        **_marker_identity(out_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": "O3_P0_OPENED_ZERO_WORK",
        "preopen": {
            "frozen_hash_audit": hashes,
            "heavy_process_audit": heavy,
            "service_health": services,
            "free_gib": free_gib,
            "nice": history.current_nice(),
        },
        "checks": checks,
        "zero_work": {
            "fresh_games": 0,
            "fresh_streams_consumed": 0,
            "replay_content_opened": 0,
            "labels": 0,
            "fits": 0,
            "policy_outcomes": 0,
            "score_inspection": 0,
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
    if not marker_path.is_file():
        raise FileNotFoundError("O3 P0 marker is missing")
    marker = json.loads(marker_path.read_text())
    if not _verify_self_hash(marker, "opened_payload_sha256"):
        raise ValueError("O3 P0 marker payload mismatch")
    expected = _marker_identity(out_dir)
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"O3 P0 marker binding mismatch: {key}")
    return marker


def _write_manifest(
    out_dir: Path,
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    written = _write_immutable_json(
        out_dir / name,
        payload,
        self_hash_field="payload_sha256",
    )
    return {
        "path": str(out_dir / name),
        "file_sha256": sha256_path(out_dir / name),
        "payload_sha256": written["payload_sha256"],
    }


def _decision(
    *,
    representation_checks: Mapping[str, bool],
    readiness_checks: Mapping[str, bool],
) -> str:
    if not all(representation_checks.values()):
        return "KILL_O3_REPRESENTATION_PREFLIGHT"
    if not all(readiness_checks.values()):
        return "HOLD_O3_DATA_OR_POWER"
    return "READY_O3_EVENT_ACQUISITION"


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    marker = _load_marker(out_dir)
    result_path = out_dir / RESULT_NAME
    if result_path.exists():
        raise FileExistsError("O3 P0 terminal result already exists")
    allowed = {MARKER_NAME}
    unexpected = {
        path.name for path in out_dir.iterdir() if path.name not in allowed
    }
    if unexpected:
        raise ValueError(f"O3 P0 namespace contains unexpected work: {unexpected}")

    try:
        hashes = frozen_hash_audit()
        tests = _load_test_evidence()
        o2 = o2_aggregate_evidence()
        rows = future_stream_rows()
        partitions = partition_plan(rows)
        streams = stream_contract(rows)
        collision = collision_audit(rows, out_dir=out_dir)
        power = power_contract()
        policies = policy_audit()
        heavy = _heavy_process_audit()
        services = history.service_health()
        free_gib = shutil.disk_usage(out_dir).free / 1024**3

        stream_artifact = _write_manifest(
            out_dir,
            STREAM_MANIFEST_NAME,
            {
                "version": f"{VERSION}_streams",
                "rows": rows,
                "contract": streams,
                "streams_consumed": 0,
            },
        )
        partition_artifact = _write_manifest(
            out_dir,
            PARTITION_MANIFEST_NAME,
            {
                "version": f"{VERSION}_partitions",
                **partitions,
                "outcomes_opened": False,
            },
        )
        collision_artifact = _write_manifest(
            out_dir,
            COLLISION_MANIFEST_NAME,
            {
                "version": f"{VERSION}_collisions",
                **collision,
            },
        )
        power_artifact = _write_manifest(
            out_dir,
            POWER_MANIFEST_NAME,
            {
                "version": f"{VERSION}_power",
                **power,
                "outcomes_used": False,
            },
        )
        policy_artifact = _write_manifest(
            out_dir,
            POLICY_MANIFEST_NAME,
            {
                "version": f"{VERSION}_policies",
                **policies,
            },
        )

        representation_checks = {
            "frozen_hashes_schema_model_exact": hashes["passes"],
            "tests_passed": bool(tests["passes"]),
            "policy_action_exactness_and_distinctness": policies["passes"],
        }
        readiness_checks = {
            "o2_aggregate_only_contract": o2["passes"],
            "whole_root_partition_plan": partitions["passes"],
            "stream_contract": streams["passes"],
            "zero_historical_stream_collisions": collision["passes"],
            "power_n192_or150": power["passes"],
            "one_heavy_job": heavy["passes"],
            "nice_at_least_10": history.current_nice() >= MINIMUM_NICE,
            "free_disk_above_120_gib": free_gib > TARGET_FREE_GIB,
            "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
            "services_dashboard_top_three": services["passes"],
            "acquisition_runtime_cap_frozen": ACQUISITION_ACTIVE_SECONDS
            == 144 * 3_600,
            "acquisition_storage_cap_frozen": ACQUISITION_BYTE_LIMIT
            == 28 * 1024**3,
            "no_fresh_work": True,
        }
        decision = _decision(
            representation_checks=representation_checks,
            readiness_checks=readiness_checks,
        )
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "acquisition_authorized_by_result": decision
            == "READY_O3_EVENT_ACQUISITION",
            "marker": {
                "path": str(out_dir / MARKER_NAME),
                "file_sha256": sha256_path(out_dir / MARKER_NAME),
                "payload_sha256": marker["opened_payload_sha256"],
            },
            "frozen_hash_audit": hashes,
            "test_evidence": {
                "path": str(TEST_EVIDENCE_PATH),
                "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
                "payload_sha256": tests["test_evidence_payload_sha256"],
                "focused_tests_passed": tests["focused_tests_passed"],
                "regression_tests_passed": tests["regression_tests_passed"],
            },
            "o2_aggregate_evidence": o2,
            "artifacts": {
                "streams": stream_artifact,
                "partitions": partition_artifact,
                "collision": collision_artifact,
                "power": power_artifact,
                "policies": policy_artifact,
            },
            "summaries": {
                "root_universe_count": partitions["root_universe_count"],
                "role_counts": partitions["role_counts"],
                "family_order": list(O3_FAMILY_ORDER),
                "stream_row_count": streams["row_count"],
                "stream_purpose_counts": streams["purpose_counts"],
                "collision_source_count": collision["matched_source_count"],
                "collision_source_sha256": collision[
                    "matched_sources_sha256"
                ],
                "selected_mechanism_n": power["selected_roots"],
                "or150_full_gate_power": next(
                    row["power_full_gate"]
                    for row in power["rows"]
                    if row["roots"] == 192
                    and row["true_common_odds_ratio"] == 1.50
                ),
                "grid_mde": power["grid_mde"],
                "policy_signatures": policies["signature_sha256"],
            },
            "representation_checks": representation_checks,
            "readiness_checks": readiness_checks,
            "process": {
                "nice": history.current_nice(),
                "heavy_process_audit": heavy,
                "free_gib": free_gib,
                "service_health": services,
            },
            "zero_work": {
                "fresh_games": 0,
                "fresh_streams_consumed": 0,
                "o2_support_content_parsed": False,
                "o2_replay_content_read": False,
                "labels": 0,
                "fits": 0,
                "policy_outcomes": 0,
                "scores_inspected": 0,
                "incumbent_changed": False,
                "dashboard_changed": False,
            },
        }
    except Exception as error:
        result = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_O3_DATA_OR_POWER",
            "acquisition_authorized_by_result": False,
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
                "fits": 0,
                "policy_outcomes": 0,
                "scores_inspected": 0,
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

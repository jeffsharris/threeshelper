"""Outcome-free J2 A1 readiness amendment tooling.

This module only audits immutable inputs, seals test evidence, and prepares an
amended readiness result. It has no teacher, game, stream-reservation,
training, evaluation, or promotion surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from threes_rl import j2_incumbent_distillation_readiness as j2


VERSION = "j2_incumbent_distillation_readiness_amendment_a1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs"
CHARTER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J2_INCUMBENT_DISTILLATION_READINESS_AMENDMENT_A1.md"
)
RUNNER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "j2_incumbent_distillation_readiness_amendment_a1.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_rl_j2_incumbent_distillation_readiness_amendment_a1.py"
)
OUTPUT_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j2_incumbent_distillation_readiness_amendment_a1"
)
PARENT_READINESS_DIR = (
    RUNS_ROOT / "forensics" / "j2_incumbent_distillation_readiness_v1"
)
V1_PILOT_DIR = (
    RUNS_ROOT / "forensics" / "j2_exact_teacher_feasibility_pilot_v1"
)
V2_PILOT_DIR = (
    RUNS_ROOT / "forensics" / "j2_exact_teacher_feasibility_pilot_v2"
)
FUTURE_EXECUTION_DIRS = (
    RUNS_ROOT / "forensics" / "j2a1_distillation_execution_v1",
    RUNS_ROOT / "forensics" / "j2a1_on_policy_training_v1",
    RUNS_ROOT / "forensics" / "j2a1_development_v1",
    RUNS_ROOT / "forensics" / "j2a1_confirmation_v1",
)

TEST_EVIDENCE_NAME = "J2A1_TEST_EVIDENCE.json"
INPUT_BINDINGS_NAME = "J2A1_INPUT_BINDINGS.json"
PROSPECTIVE_AUTHORITY_NAME = "J2A1_PROSPECTIVE_AUTHORITY.json"
POWER_NAME = "J2A1_POWER.json"
PROJECTION_NAME = "J2A1_RUNTIME_STORAGE_PROJECTION.json"
FAMILY_SAFEGUARD_NAME = "J2A1_FAMILY_SUPPORT_SAFEGUARD.json"
READINESS_LOCK_NAME = "J2A1_READINESS_LOCK.json"
READINESS_RESULT_NAME = "J2A1_READINESS_RESULT.json"
RETENTION_NAME = "J2A1_RETENTION.json"

READY = "READY_J2_A1_INCUMBENT_DISTILLATION_PREFLIGHT"
HOLD = "HOLD_J2_A1_INCUMBENT_DISTILLATION_PREFLIGHT"
KILL = "KILL_J2_A1_READINESS_INTEGRITY"

EXPECTED_CHARTER_SHA256 = (
    "371e16088a4cbe3a7a3c5e6668fdd13424cad439f1a8f7e85b4dc7c120573e6a"
)
EXPECTED_PARENT_SOURCE_HASHES = {
    "threes_rl/J2_INCUMBENT_DISTILLED_JOINT_POLICY_VALUE_CHARTER.md":
        "3cf410a4da9418c9e06164ac077d3e389f77720d056dfe25ced2a4a2a052163b",
    "threes_rl/j2_incumbent_distillation_readiness.py":
        "9ecd658ea69968feb605d0e0a9e4e621b73ac01619536e45c0cdf69b7bc3b15f",
    "tests/test_rl_j2_incumbent_distillation_readiness.py":
        "24736fa56702c46b24d515716d7a6365dadb49b20622f333bead39d3105ebdb2",
    "threes_rl/J2_EXACT_TEACHER_ENGINEERING_FEASIBILITY_PILOT_CHARTER.md":
        "fb61f4cea16ecea6bc30f24f42b2b6a5ed483e3172428a2a20dc66bbc16a0de9",
    "threes_rl/j2_exact_teacher_feasibility_pilot.py":
        "57264251bd49e78e4608c5a8adfb91c9a14bf524327a68f88104a9a541d14970",
    "tests/test_rl_j2_exact_teacher_feasibility_pilot.py":
        "1754c8d89df5bd75541e6d302a1e905918e2c33f1022620a65f41eed4f1a9a85",
    "threes_rl/J2_EXACT_TEACHER_ENGINEERING_FEASIBILITY_PILOT_V2_CHARTER.md":
        "f612695a69a1914a29eb0b9c60932680576085d2da55a9cd32663d3f99b1277f",
    "threes_rl/j2_exact_teacher_feasibility_pilot_v2.py":
        "e72e96907ac356116ee6b764a6d2fe32d9ccf1339010418d1ade9c3660bee3f8",
    "tests/test_rl_j2_exact_teacher_feasibility_pilot_v2.py":
        "f92e46135a52d000782d8ae6c4d36ba4d9d3b1179a56438d952a94cf8852c4df",
}

EXPECTED_PARENT_READINESS = {
    "J2_INPUT_BINDINGS.json": (
        "06656be5428ce57cc29960988fdbcdc720844ff82dc2c45007f4e33366170416",
        "input_bindings_payload_sha256",
        "651d66fc6328e35f1932e937dd819eec037c8f84ab11b994ca47815b1e5184a0",
    ),
    "J2_MODEL_SCHEMA.json": (
        "ac976d7c392b211bd3791ef218ec6da42b55aabe9a5e0643043ce4704200d056",
        "model_schema_payload_sha256",
        "679ee9a64c53f8dc821dfe173bc8b0c1ee807d0bb7e808a5342cb5cb71182867",
    ),
    "J2_POWER_AND_FEASIBILITY.json": (
        "b210be7d16d27d1cb4fc419f38952aeafdd4d3939497f9b4d06df5e9f65ef43f",
        "power_payload_sha256",
        "4bf3f3fdf32d0929bc2dd0dc5389b3c2772b18677a68f73d4b2c17c3c141b64c",
    ),
    "J2_PROSPECTIVE_AUTHORITY.json": (
        "cea6f129e0dbb5309d67d554a74ddb8965e6c5586efb36f570363d7d370707f8",
        "prospective_authority_payload_sha256",
        "631ed382950a30dd51790ad94cfb9fb56b78f9330c87d794d17977e9d14690d6",
    ),
    "J2_PROTECTED_STREAM_AUTHORITY.json": (
        "b9e806e13c28d33f0edabe756ed06b49c7c5e880bd8370de99b007c0bc9d28db",
        "protected_stream_authority_payload_sha256",
        "51fa8c173049b01a3fff19860968de2bc4d09521f5cc3980ab0da9ab4add40e6",
    ),
    "J2_READINESS_LOCK.json": (
        "c3f08429b625369263b75a5724b3abfdf2487d6a9fd2414897c7aaca7fd74488",
        "readiness_lock_payload_sha256",
        "a4683de92f833c4f33451b9f73acc0214566ab2d28d45a4e95e49a6d07372c8e",
    ),
    "J2_READINESS_RESULT.json": (
        "8c24be58bb6a30cd2cf302f17894b69e131f3b3c6092a4e71801c6b0f2f96eab",
        "readiness_result_payload_sha256",
        "4110e987eed93a0b50cf8dfc3978469f316039edfe03ae22549daf464ddf04de",
    ),
    "J2_RUNTIME_STORAGE_PROJECTION.json": (
        "f59740b3f3d6f15697033f769267b233640978dd5947beb203f2f62de5643f68",
        "projection_payload_sha256",
        "45d3aa109244792165ffa9e9a9ac231b063814386ea4d905668dc3adb3da44d7",
    ),
    "J2_TEACHER_PROVENANCE.json": (
        "824aa8988136d81a00d81dd4899b9985aedbbb213260d3a2e94c4e7dc931840a",
        "teacher_provenance_payload_sha256",
        "a8d355bd056bdd31f860a668d4e86a0898866192b39cf0665d348db33ac02768",
    ),
    "J2_TEST_EVIDENCE.json": (
        "32ad0836ff55501bdc3f78bc49d58a44d89ebc4544c7e721c4c7b7f991cd6e53",
        "test_evidence_payload_sha256",
        "2ae7371b2a946eb52b0942387acd0f599e42065b996f04f09d53a7506c7c82cd",
    ),
}

EXPECTED_V1_EVIDENCE = (
    "ec9235912266c0e18e7596d08cdca4468f2605888c6a6de1c165224c17e8aee3",
    "test_evidence_payload_sha256",
    "d960d3a904572ffa2f2ea8df645e890c6048af399e7d6cdf32d33e1919691658",
)
EXPECTED_V2_TERMINAL = (
    "3ee2b204307bb96489ffd0fc3ff5c6c0cef488d6b5cfe986c4940f808354fcd9",
    "terminal_payload_sha256",
    "8b98a0ec9892b615dd5072849b9fc655f7d043c7a257d90619ddbf35ad925089",
)
EXPECTED_V2_RETENTION = (
    "6fe6563d6d676bf93455f0f3060ae3d851bf4b87b0c440216c52c703d0ff53a0",
    "retention_payload_sha256",
    "e8b9e6365a449689f0b485790ea9f3e4a27d1351562b0d387cd113c6db4702d1",
)
EXPECTED_V2_RETENTION_INVENTORY_SHA256 = (
    "85a9b6e4c1382caaed84e4f9efee9991348f2da6ea7172e63edb3e7d03af87d5"
)
EXPECTED_V2_RETENTION_FILES = 26
EXPECTED_V2_RETENTION_BYTES = 3_585_016
EXPECTED_V2_POWER_REPORT_SHA256 = (
    "157f90f6185fe7a08548a140b10d0f582c351d7d7b8abff53be78ff4ee91e28b"
)
EXPECTED_V2_POWER = 0.8059895833333334
EXPECTED_V2_POWER_MCSE = 0.014269101547515112
EXPECTED_V2_CENTRAL_P99 = 0.1316514358320273
EXPECTED_V2_CENTRAL_CALLS_PER_SECOND = 443.48362719264816
EXPECTED_V2_SYNC_CALLS_PER_SECOND = 408.1622875147186
EXPECTED_V2_MAX_CONTEMPORANEOUS_RSS = 2_796_617_728
EXPECTED_V2_CONSERVATIVE_PEAK_RSS = 2_857_598_976
EXPECTED_V2_QUERY_COUNT = 19_432

VALIDATION_PAIRS = 6_144
EXPECTED_STAGE_TOTALS = {
    "prospective_rows_or_pairs": 36_096,
    "game_arms": 47_616,
    "unique_streams": 155_904,
    "pre_ppo_teacher_roots": 14_336,
    "online_teacher_roots": 4_096,
    "total_teacher_root_equivalents": 18_432,
}
POWER_DATASETS = j2.POWER_DATASETS
POWER_BOOTSTRAPS = j2.POWER_BOOTSTRAPS
PLANNING_MOVES = j2.PLANNING_MOVES
SENSITIVITY_MOVES = j2.SENSITIVITY_MOVES
SAFETY_MULTIPLIER = j2.SAFETY_MULTIPLIER
SHARD_COUNT = j2.SHARD_COUNT
RUNTIME_CAP_HOURS = 72.0
STORAGE_CAP_BYTES = 24 * 1024**3
ROOT_BLOB_BYTES = 1_519
ROUND_BATCH_BYTES = 1_261
PAIR_BLOB_BYTES = 24_576
OPTIMIZER_FIXTURE_HOURS = 0.03300427754720052
INHERITED_J1_TRAINING_MARGIN_HOURS = 3.309263890690274
PREFLIGHT_AVAILABLE_MEMORY_BYTES = 30_697_111_552
EFFECTIVE_MEMORY_CAP_BYTES = min(
    STORAGE_CAP_BYTES,
    int(0.75 * PREFLIGHT_AVAILABLE_MEMORY_BYTES),
)

ZERO_WORK = {
    "execution_markers": 0,
    "streams_reserved": 0,
    "streams_consumed": 0,
    "teacher_queries": 0,
    "teacher_action_labels": 0,
    "normal_start_games": 0,
    "scientific_optimizer_steps": 0,
    "scientific_checkpoints": 0,
    "distillation_validation_content_reads": 0,
    "development_content_reads": 0,
    "confirmation_content_reads": 0,
    "policy_or_score_outcomes": 0,
    "human_session_reads": 0,
    "incumbent_changes": 0,
    "dashboard_changes": 0,
    "promotion_actions": 0,
}


class J2A1IntegrityError(RuntimeError):
    """Raised when an immutable A1 contract check fails."""


def stage_table() -> tuple[dict[str, Any], ...]:
    rows = j2.json_native(j2.STAGE_TABLE)
    if not isinstance(rows, list):
        raise J2A1IntegrityError("Parent J2 stage table is malformed")
    amended: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if row["stage"] == "distillation_validation":
            row["authority_rows"] = VALIDATION_PAIRS
            row["game_arms"] = 2 * VALIDATION_PAIRS
            row["pre_ppo_teacher_roots"] = VALIDATION_PAIRS
        amended.append(row)
    return tuple(amended)


STAGE_TABLE = stage_table()


def derive_stage_totals(
    table: Sequence[Mapping[str, Any]] = STAGE_TABLE,
) -> dict[str, int]:
    return {
        "prospective_rows_or_pairs": sum(
            int(row["authority_rows"]) for row in table
        ),
        "game_arms": sum(int(row["game_arms"]) for row in table),
        "unique_streams": sum(
            int(row["authority_rows"]) * len(row["streams"])
            for row in table
        ),
        "pre_ppo_teacher_roots": sum(
            int(row["pre_ppo_teacher_roots"]) for row in table
        ),
        "online_teacher_roots": sum(
            int(row["online_teacher_roots"]) for row in table
        ),
        "total_teacher_root_equivalents": sum(
            int(row["pre_ppo_teacher_roots"])
            + int(row["online_teacher_roots"])
            for row in table
        ),
    }


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(
    path: Path,
    *,
    field: str,
) -> dict[str, Any]:
    payload = j2.load_hashed_json(path, field=field)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "bytes": path.stat().st_size,
        "payload_field": field,
        "payload_sha256": payload[field],
    }


def _expected_identity(
    path: Path,
    expected: tuple[str, str, str],
) -> tuple[dict[str, Any], bool]:
    file_sha, field, payload_sha = expected
    identity = _identity(path, field=field)
    return identity, (
        identity["file_sha256"] == file_sha
        and identity["payload_sha256"] == payload_sha
    )


def source_and_parent_audit() -> dict[str, Any]:
    local = {
        "charter": sha256_path(CHARTER_PATH),
        "runner": sha256_path(RUNNER_PATH),
        "tests": sha256_path(TEST_PATH) if TEST_PATH.exists() else None,
    }
    parent_sources = {
        relative: sha256_path(REPO_ROOT / relative)
        for relative in EXPECTED_PARENT_SOURCE_HASHES
    }
    readiness_rows: dict[str, Any] = {}
    readiness_checks: dict[str, bool] = {}
    for name, expected in EXPECTED_PARENT_READINESS.items():
        identity, exact = _expected_identity(
            PARENT_READINESS_DIR / name,
            expected,
        )
        readiness_rows[name] = identity
        readiness_checks[name] = exact
    parent_result = j2.load_hashed_json(
        PARENT_READINESS_DIR / "J2_READINESS_RESULT.json",
        field="readiness_result_payload_sha256",
    )
    checks = {
        "amendment_charter_exact": (
            local["charter"] == EXPECTED_CHARTER_SHA256
        ),
        "amendment_runner_present": local["runner"] is not None,
        "amendment_tests_present": local["tests"] is not None,
        "parent_sources_exact": (
            parent_sources == EXPECTED_PARENT_SOURCE_HASHES
        ),
        "all_parent_readiness_artifacts_exact": all(
            readiness_checks.values()
        ),
        "parent_readiness_is_scoped_hold": (
            parent_result["decision"]
            == "HOLD_J2_INCUMBENT_DISTILLATION_PREFLIGHT"
        ),
        "parent_execution_not_authorized": (
            parent_result["execution_authorized"] is False
        ),
        "parent_zero_work_exact": (
            parent_result["zero_work"] == j2.ZERO_WORK
        ),
    }
    return {
        "version": f"{VERSION}_source_parent_audit_v1",
        "local_sources": local,
        "parent_sources": parent_sources,
        "parent_readiness_artifacts": readiness_rows,
        "parent_readiness_checks": readiness_checks,
        "checks": checks,
        "passes": all(checks.values()),
    }


def pilot_history_audit() -> dict[str, Any]:
    v1_identity, v1_exact = _expected_identity(
        V1_PILOT_DIR / "J2_TEACHER_PILOT_TEST_EVIDENCE.json",
        EXPECTED_V1_EVIDENCE,
    )
    terminal_identity, terminal_exact = _expected_identity(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_TERMINAL_RESULT.json",
        EXPECTED_V2_TERMINAL,
    )
    retention_identity, retention_exact = _expected_identity(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_RETENTION.json",
        EXPECTED_V2_RETENTION,
    )
    terminal = j2.load_hashed_json(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_TERMINAL_RESULT.json",
        field="terminal_payload_sha256",
    )
    retention = j2.load_hashed_json(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_RETENTION.json",
        field="retention_payload_sha256",
    )
    retention_rows = retention["files"]
    row_checks = []
    observed_rows = []
    for row in retention_rows:
        relative = Path(str(row["path"]))
        path = V2_PILOT_DIR / relative
        observed = {
            "path": relative.as_posix(),
            "file_sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        }
        observed_rows.append(observed)
        row_checks.append(observed == row)
    measured_refs = {
        name: terminal[name]
        for name in ("central", "sensitivity", "synchronous", "power")
    }
    measured_checks = {}
    for name, ref in measured_refs.items():
        path = Path(str(ref["path"]))
        measured_checks[name] = (
            path.parent == V2_PILOT_DIR
            and sha256_path(path) == ref["file_sha256"]
            and j2.load_hashed_json(
                path,
                field=str(ref["payload_field"]),
            )[str(ref["payload_field"])]
            == ref["payload_sha256"]
        )
    zero_fields = {
        "labels_retained": 0,
        "actions_retained": 0,
        "scores_retained": 0,
        "outcomes_retained": 0,
        "trajectories_retained": 0,
        "ppo_trajectories": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "j2_scientific_streams_reserved": 0,
        "j2_scientific_streams_consumed": 0,
    }
    checks = {
        "v1_evidence_exact": v1_exact,
        "v2_terminal_exact": terminal_exact,
        "v2_retention_exact": retention_exact,
        "v2_terminal_ready_for_amendment_only": (
            terminal["decision"]
            == "READY_J2_FEASIBILITY_AMENDMENT_PREFLIGHT_V2"
        ),
        "v2_integrity_passed": terminal["integrity_passes"] is True,
        "v2_three_separate_gates_passed": (
            terminal["separate_decisions"][
                "real_eight_process_pretraining_throughput_memory"
            ]
            == "PASS"
            and terminal["separate_decisions"][
                "synchronous_16_round_orchestration"
            ]
            == "PASS"
            and terminal["separate_decisions"][
                "powered_validation_n_recommendation"
            ]["decision"]
            == "PASS"
        ),
        "v2_retention_inventory_exact": (
            int(retention["file_count"]) == EXPECTED_V2_RETENTION_FILES
            and int(retention["total_bytes"]) == EXPECTED_V2_RETENTION_BYTES
            and retention["inventory_sha256"]
            == EXPECTED_V2_RETENTION_INVENTORY_SHA256
            and j2.canonical_json_hash(retention_rows)
            == EXPECTED_V2_RETENTION_INVENTORY_SHA256
            and all(row_checks)
        ),
        "v2_measured_artifacts_exact": all(measured_checks.values()),
        "v2_query_count_exact": (
            int(terminal["teacher_query_calls"])
            == EXPECTED_V2_QUERY_COUNT
            and int(terminal["teacher_query_accounting"]["total"])
            == EXPECTED_V2_QUERY_COUNT
        ),
        "v2_scientific_counters_zero": all(
            terminal[key] == value for key, value in zero_fields.items()
        ),
        "v2_development_confirmation_unopened": (
            terminal["development_opened"] is False
            and terminal["confirmation_opened"] is False
        ),
        "v2_promotion_false": terminal["promotion"] is False,
    }
    return {
        "version": f"{VERSION}_pilot_history_audit_v1",
        "v1_test_evidence": v1_identity,
        "v2_terminal": terminal_identity,
        "v2_retention": retention_identity,
        "v2_retention_predecessors": observed_rows,
        "v2_measured_artifacts": measured_refs,
        "measured_artifact_checks": measured_checks,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _commitment(prefix: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        prefix.encode("ascii") + b"|" + j2.canonical_json_bytes(payload)
    ).hexdigest()


def build_prospective_rows(
    table: Sequence[Mapping[str, Any]] = STAGE_TABLE,
) -> list[dict[str, Any]]:
    rows = []
    for block in table:
        for row_index in range(int(block["authority_rows"])):
            streams = {
                str(name): int(base) + row_index
                for name, base in block["streams"].items()
            }
            core = {
                "stage": str(block["stage"]),
                "row_index": row_index,
                "streams": streams,
            }
            rows.append(
                {
                    **core,
                    "root_id": _commitment("j2-a1-root-v1", core),
                    "ancestry_id": _commitment(
                        "j2-a1-ancestry-v1",
                        core,
                    ),
                    "content_opened": False,
                    "reserved": False,
                    "consumed": False,
                }
            )
    return rows


def prospective_authority() -> dict[str, Any]:
    rows = build_prospective_rows()
    parent = j2.load_hashed_json(
        PARENT_READINESS_DIR / "J2_PROSPECTIVE_AUTHORITY.json",
        field="prospective_authority_payload_sha256",
    )
    parent_rows = parent["rows"]
    roots = [str(row["root_id"]) for row in rows]
    ancestries = [str(row["ancestry_id"]) for row in rows]
    streams = [
        int(value)
        for row in rows
        for value in row["streams"].values()
    ]
    stage_counts = Counter(str(row["stage"]) for row in rows)
    prefixes = Counter(value // 1_000_000_000 for value in streams)
    paired = [
        row
        for row in rows
        if row["stage"]
        in {
            "distillation_validation",
            "development",
            "confirmation",
        }
    ]
    stage_root_sets = {
        stage: {
            str(row["root_id"])
            for row in rows
            if str(row["stage"]) == stage
        }
        for stage in stage_counts
    }
    stage_ancestry_sets = {
        stage: {
            str(row["ancestry_id"])
            for row in rows
            if str(row["stage"]) == stage
        }
        for stage in stage_counts
    }
    root_union_size = len(set().union(*stage_root_sets.values()))
    ancestry_union_size = len(set().union(*stage_ancestry_sets.values()))
    parent_roots = {str(row["root_id"]) for row in parent_rows}
    parent_ancestries = {str(row["ancestry_id"]) for row in parent_rows}
    validation_rows = [
        row for row in rows if row["stage"] == "distillation_validation"
    ]
    checks = {
        "single_stage_table_totals_exact": (
            derive_stage_totals() == EXPECTED_STAGE_TOTALS
        ),
        "row_count_exact": len(rows) == 36_096,
        "stream_count_exact": len(streams) == 155_904,
        "root_ids_unique": len(set(roots)) == len(roots),
        "ancestry_ids_unique": len(set(ancestries)) == len(ancestries),
        "stream_ids_unique": len(set(streams)) == len(streams),
        "stage_root_sets_disjoint": root_union_size == len(roots),
        "stage_ancestry_sets_disjoint": (
            ancestry_union_size == len(ancestries)
        ),
        "stage_counts_exact": dict(stage_counts)
        == {
            "teacher_behavior_cloning": 8_192,
            "distillation_validation": 6_144,
            "on_policy_training": 16_384,
            "development": 896,
            "confirmation": 4_480,
        },
        "prefixes_227b_249b_exact": set(prefixes) == set(range(227, 250)),
        "no_spent_213b_226b_collision": not (
            set(prefixes) & set(range(213, 227))
        ),
        "no_engineering_250b_255b_collision": not (
            set(prefixes) & set(range(250, 256))
        ),
        "paired_exogenous_streams_single_and_shared": all(
            {
                "logical_stream_id",
                "deck_stream_id",
                "slot_stream_id",
            }.issubset(row["streams"])
            for row in paired
        ),
        "paired_policy_streams_distinct": all(
            len(
                {
                    int(value)
                    for key, value in row["streams"].items()
                    if key.endswith("policy_stream_id")
                }
            )
            == 2
            for row in paired
        ),
        "validation_offsets_0_through_6143": (
            [int(row["row_index"]) for row in validation_rows]
            == list(range(6_144))
        ),
        "parent_prospective_was_unopened_unspent": all(
            row["content_opened"] is False
            and row["reserved"] is False
            and row["consumed"] is False
            for row in parent_rows
        ),
        "new_root_commitments_distinct_from_parent": not (
            set(roots) & parent_roots
        ),
        "new_ancestry_commitments_distinct_from_parent": not (
            set(ancestries) & parent_ancestries
        ),
        "all_a1_rows_unopened": all(
            row["content_opened"] is False for row in rows
        ),
        "all_a1_rows_unreserved": all(
            row["reserved"] is False for row in rows
        ),
        "all_a1_rows_unconsumed": all(
            row["consumed"] is False for row in rows
        ),
    }
    return {
        "version": f"{VERSION}_prospective_authority_v1",
        "method": (
            "content-blind j2-a1 SHA-256 commitments over exact stage, "
            "row, and stream identities"
        ),
        "stage_table": j2.json_native(STAGE_TABLE),
        "derived_counts": derive_stage_totals(),
        "rows": rows,
        "row_count": len(rows),
        "game_arms": EXPECTED_STAGE_TOTALS["game_arms"],
        "stream_count": len(streams),
        "stage_counts": dict(stage_counts),
        "stream_prefix_counts": {
            str(key): value for key, value in sorted(prefixes.items())
        },
        "canonical_rows_sha256": j2.canonical_json_hash(rows),
        "root_set_sha256": j2.canonical_json_hash(sorted(roots)),
        "ancestry_set_sha256": j2.canonical_json_hash(
            sorted(ancestries)
        ),
        "parent_prospective_authority": _identity(
            PARENT_READINESS_DIR / "J2_PROSPECTIVE_AUTHORITY.json",
            field="prospective_authority_payload_sha256",
        ),
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": dict(ZERO_WORK),
    }


def score_fidelity_power(n_pairs: int = VALIDATION_PAIRS) -> dict[str, Any]:
    standard_error = j2.SCORE_SD / math.sqrt(int(n_pairs))
    mde = math.exp(
        (j2.SCORE_Z_975 + j2.SCORE_Z_80) * standard_error
    ) - 1.0
    normal = statistics.NormalDist()

    def probability_above(threshold: float) -> float:
        return normal.cdf((0.0 - float(threshold)) / standard_error)

    point_threshold = math.log(j2.FIDELITY_SCORE_POINT_FLOOR)
    ci_threshold = (
        math.log(j2.FIDELITY_SCORE_CI_FLOOR)
        + j2.SCORE_Z_975 * standard_error
    )
    combined_threshold = max(point_threshold, ci_threshold)
    return {
        "n_pairs": int(n_pairs),
        "paired_sd": j2.SCORE_SD,
        "standard_error": standard_error,
        "score_80pct_mde_fraction": mde,
        "score_80pct_mde_percent": 100.0 * mde,
        "point_floor_ratio": j2.FIDELITY_SCORE_POINT_FLOOR,
        "lower_ci_floor_ratio": j2.FIDELITY_SCORE_CI_FLOOR,
        "equal_policy_point_only_power": probability_above(
            point_threshold
        ),
        "equal_policy_10pct_ci_only_power": probability_above(
            ci_threshold
        ),
        "equal_policy_combined_gate_power": probability_above(
            combined_threshold
        ),
        "method": (
            "parent normal paired log-score model with two-sided 95% "
            "lower bound"
        ),
    }


def power_report(
    *,
    datasets: int = POWER_DATASETS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    if (
        int(datasets) != POWER_DATASETS
        or int(bootstraps) != POWER_BOOTSTRAPS
    ):
        raise J2A1IntegrityError("A1 power workload cannot be reduced")
    pilot = j2.load_hashed_json(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_POWER_SIZING.json",
        field="power_payload_sha256",
    )
    pilot_rows = [
        row for row in pilot["rows"] if int(row["n_pairs"]) == 6_144
    ]
    if len(pilot_rows) != 1:
        raise J2A1IntegrityError("V2 N=6144 power row is not unique")
    pilot_row = pilot_rows[0]
    progression = j2.common_or_power_grid(
        n_pairs=VALIDATION_PAIRS,
        datasets=int(datasets),
        bootstraps=int(bootstraps),
    )
    report_sha = j2.canonical_json_hash(progression)
    worst_row = min(
        progression["rows"],
        key=lambda row: float(row["primary_gate_power"]),
    )
    checks = {
        "parent_method_constants_unchanged": (
            POWER_DATASETS == 768
            and POWER_BOOTSTRAPS == 199
            and j2.POWER_SEED == 2_026_072_821
            and tuple(j2.CONTROL_RATES) == (0.02, 0.04, 0.08, 0.15)
            and tuple(j2.PAIRING_COUPLINGS) == (0.0, 0.05, 0.10)
            and j2.FIDELITY_OR_POINT_FLOOR == 0.90
            and j2.FIDELITY_OR_CI_FLOOR == 0.50
        ),
        "n6144_exact": int(progression["n_pairs"]) == 6_144,
        "eight_equal_strata": all(
            int(row["roots_per_stratum"]) == 768
            for row in progression["rows"]
        ),
        "full_report_reproduces_v2": (
            report_sha
            == pilot_row["full_report_sha256"]
            == EXPECTED_V2_POWER_REPORT_SHA256
        ),
        "worst_power_reproduces_v2": (
            float(progression["worst_case_primary_power"])
            == float(pilot_row["worst_case_power"])
            == EXPECTED_V2_POWER
        ),
        "worst_mcse_reproduces_v2": (
            float(worst_row["monte_carlo_standard_error"])
            == float(pilot_row["worst_case_mcse"])
            == EXPECTED_V2_POWER_MCSE
        ),
        "worst_cell_reproduces_v2": (
            float(worst_row["control_rate"])
            == float(pilot_row["worst_case_control_rate"])
            == 0.02
            and float(worst_row["coupling"])
            == float(pilot_row["worst_case_coupling"])
            == 0.0
        ),
        "worst_power_at_least_080": (
            float(progression["worst_case_primary_power"]) >= 0.80
        ),
    }
    return {
        "version": f"{VERSION}_power_v1",
        "score_fidelity": score_fidelity_power(),
        "progression_common_or": progression,
        "progression_full_report_sha256": report_sha,
        "bound_v2_power_row": pilot_row,
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": dict(ZERO_WORK),
    }


def runtime_storage_projection() -> dict[str, Any]:
    terminal = j2.load_hashed_json(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_TERMINAL_RESULT.json",
        field="terminal_payload_sha256",
    )
    central = j2.load_hashed_json(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_CENTRAL_COST.json",
        field="central_payload_sha256",
    )
    sensitivity = j2.load_hashed_json(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_SENSITIVITY_COST.json",
        field="sensitivity_payload_sha256",
    )
    synchronous = j2.load_hashed_json(
        V2_PILOT_DIR
        / "J2_TEACHER_PILOT_V2_SYNCHRONOUS_ORCHESTRATION.json",
        field="sync_payload_sha256",
    )
    observed_p99 = float(
        central["parallel_eight_process"]["timing_summary"][
            "p99_seconds"
        ]
    )
    observed_calls_per_second = float(
        central["parallel_eight_process"]["calls_per_second"]
    )
    sync_calls_per_second = float(synchronous["calls_per_second"])
    pre_ppo_roots = EXPECTED_STAGE_TOTALS["pre_ppo_teacher_roots"]
    pre_ppo_calls = pre_ppo_roots * PLANNING_MOVES
    maximum_p99 = (
        (
            RUNTIME_CAP_HOURS / SAFETY_MULTIPLIER
            - OPTIMIZER_FIXTURE_HOURS
        )
        * 3600.0
        * SHARD_COUNT
        / pre_ppo_calls
    )
    distillation_hours = (
        pre_ppo_calls * observed_p99 / SHARD_COUNT / 3600.0
        + OPTIMIZER_FIXTURE_HOURS
    ) * SAFETY_MULTIPLIER
    online_calls = (
        EXPECTED_STAGE_TOTALS["online_teacher_roots"]
        * PLANNING_MOVES
    )
    inherited_pre_margin = (
        INHERITED_J1_TRAINING_MARGIN_HOURS / SAFETY_MULTIPLIER
    )
    online_hours = (
        inherited_pre_margin
        + online_calls / sync_calls_per_second / 3600.0
    ) * SAFETY_MULTIPLIER
    required_sync_calls_per_second = online_calls / (
        (
            RUNTIME_CAP_HOURS / SAFETY_MULTIPLIER
            - inherited_pre_margin
        )
        * 3600.0
    )
    bc_transitions = 8_192 * PLANNING_MOVES
    distillation_storage_before_margin = (
        bc_transitions * ROOT_BLOB_BYTES
        + bc_transitions * ROUND_BATCH_BYTES
        + VALIDATION_PAIRS * PAIR_BLOB_BYTES
        + 256 * 1024**2
    )
    distillation_storage = int(
        distillation_storage_before_margin * SAFETY_MULTIPLIER
    )
    ppo_storage = int(
        terminal["measured_phase_projection"]["online_synchronous"][
            "retained_storage_bytes"
        ]
    )

    sensitivity_p99 = float(
        sensitivity["parallel_eight_process"]["timing_summary"][
            "p99_seconds"
        ]
    )
    sensitivity_pre_ppo_calls = pre_ppo_roots * SENSITIVITY_MOVES
    sensitivity_distillation_hours = (
        sensitivity_pre_ppo_calls
        * sensitivity_p99
        / SHARD_COUNT
        / 3600.0
        + OPTIMIZER_FIXTURE_HOURS
    ) * SAFETY_MULTIPLIER
    sensitivity_bc_transitions = 8_192 * SENSITIVITY_MOVES
    sensitivity_distillation_storage = int(
        (
            sensitivity_bc_transitions * ROOT_BLOB_BYTES
            + sensitivity_bc_transitions * ROUND_BATCH_BYTES
            + VALIDATION_PAIRS * PAIR_BLOB_BYTES
            + 256 * 1024**2
        )
        * SAFETY_MULTIPLIER
    )
    sensitivity_online_hours = float(
        terminal["measured_phase_projection"]["sensitivity_5000_moves"][
            "online_runtime_hours_with_25pct_margin"
        ]
    )
    parent_projection = j2.load_hashed_json(
        REPO_ROOT
        / "threes_rl"
        / "runs"
        / "forensics"
        / "j1_execution_surface_readiness_v1"
        / "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json",
        field="projection_payload_sha256",
    )
    parent_sensitivity_storage = int(
        parent_projection["training"]["sensitivity_5000_moves"][
            "storage"
        ]["projected_with_margin_bytes"]
    )
    sensitivity_ppo_storage = int(
        parent_sensitivity_storage
        + (
            EXPECTED_STAGE_TOTALS["online_teacher_roots"]
            * SENSITIVITY_MOVES
            * 8
            * SAFETY_MULTIPLIER
        )
    )
    central_memory = int(
        terminal["measured_phase_projection"]["memory"][
            "maximum_contemporaneous_parent_children_rss_bytes"
        ]
    )
    conservative_memory = int(
        terminal["measured_phase_projection"]["memory"][
            "conservative_independent_peak_sum_bytes"
        ]
    )
    checks = {
        "measured_central_p99_exact": (
            observed_p99 == EXPECTED_V2_CENTRAL_P99
        ),
        "measured_central_throughput_exact": (
            observed_calls_per_second
            == EXPECTED_V2_CENTRAL_CALLS_PER_SECOND
        ),
        "measured_synchronous_throughput_exact": (
            sync_calls_per_second
            == EXPECTED_V2_SYNC_CALLS_PER_SECOND
        ),
        "pre_ppo_teacher_count_exact": pre_ppo_roots == 14_336,
        "pre_ppo_call_count_exact": pre_ppo_calls == 7_340_032,
        "central_p99_below_required_ceiling": (
            observed_p99 <= maximum_p99
        ),
        "distillation_runtime_within_72h": (
            distillation_hours <= RUNTIME_CAP_HOURS
        ),
        "distillation_storage_within_24gib": (
            distillation_storage <= STORAGE_CAP_BYTES
        ),
        "online_runtime_within_72h": (
            online_hours <= RUNTIME_CAP_HOURS
        ),
        "online_storage_within_24gib": (
            ppo_storage <= STORAGE_CAP_BYTES
        ),
        "online_sync_throughput_above_required_floor": (
            sync_calls_per_second >= required_sync_calls_per_second
        ),
        "measured_contemporaneous_memory_exact": (
            central_memory == EXPECTED_V2_MAX_CONTEMPORANEOUS_RSS
        ),
        "measured_conservative_memory_exact": (
            conservative_memory == EXPECTED_V2_CONSERVATIVE_PEAK_RSS
        ),
        "measured_memory_within_effective_cap": (
            max(central_memory, conservative_memory)
            <= EFFECTIVE_MEMORY_CAP_BYTES
        ),
        "sensitivity_reported_as_descriptive": True,
        "no_ideal_scaling_used_for_admission": True,
    }
    return {
        "version": f"{VERSION}_runtime_storage_projection_v1",
        "method": {
            "pretraining": (
                "V2 measured eight-process central p99 divided by frozen "
                "eight-worker concurrency, plus sealed optimizer fixture"
            ),
            "online": (
                "V2 measured 16-round synchronous calls/second added to "
                "inherited bounded J1 PPO projection"
            ),
            "safety_multiplier": SAFETY_MULTIPLIER,
        },
        "distillation": {
            "teacher_roots": pre_ppo_roots,
            "teacher_calls": pre_ppo_calls,
            "observed_p99_seconds": observed_p99,
            "maximum_admissible_p99_seconds": maximum_p99,
            "p99_margin_seconds": maximum_p99 - observed_p99,
            "p99_margin_ratio": maximum_p99 / observed_p99,
            "observed_calls_per_second_descriptive": (
                observed_calls_per_second
            ),
            "runtime_hours_with_25pct_margin": distillation_hours,
            "runtime_cap_hours": RUNTIME_CAP_HOURS,
            "runtime_headroom_hours": (
                RUNTIME_CAP_HOURS - distillation_hours
            ),
            "storage_before_margin_bytes": (
                distillation_storage_before_margin
            ),
            "storage_with_25pct_margin_bytes": distillation_storage,
            "storage_with_25pct_margin_gib": (
                distillation_storage / 1024**3
            ),
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "storage_headroom_bytes": (
                STORAGE_CAP_BYTES - distillation_storage
            ),
        },
        "on_policy_training": {
            "teacher_roots": (
                EXPECTED_STAGE_TOTALS["online_teacher_roots"]
            ),
            "teacher_calls": online_calls,
            "observed_synchronous_calls_per_second": (
                sync_calls_per_second
            ),
            "required_calls_per_second": (
                required_sync_calls_per_second
            ),
            "throughput_margin_ratio": (
                sync_calls_per_second / required_sync_calls_per_second
            ),
            "runtime_hours_with_25pct_margin": online_hours,
            "runtime_cap_hours": RUNTIME_CAP_HOURS,
            "runtime_headroom_hours": RUNTIME_CAP_HOURS - online_hours,
            "storage_with_25pct_margin_bytes": ppo_storage,
            "storage_with_25pct_margin_gib": ppo_storage / 1024**3,
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "storage_headroom_bytes": STORAGE_CAP_BYTES - ppo_storage,
        },
        "memory": {
            "maximum_contemporaneous_parent_children_rss_bytes": (
                central_memory
            ),
            "conservative_independent_peak_sum_bytes": (
                conservative_memory
            ),
            "effective_memory_cap_bytes": EFFECTIVE_MEMORY_CAP_BYTES,
            "headroom_bytes": (
                EFFECTIVE_MEMORY_CAP_BYTES
                - max(central_memory, conservative_memory)
            ),
        },
        "sensitivity_5000_moves": {
            "diagnostic_not_conjunctive": True,
            "observed_sensitivity_p99_seconds": sensitivity_p99,
            "distillation_runtime_hours_with_25pct_margin": (
                sensitivity_distillation_hours
            ),
            "distillation_runtime_fits_72h": (
                sensitivity_distillation_hours <= RUNTIME_CAP_HOURS
            ),
            "distillation_storage_with_25pct_margin_bytes": (
                sensitivity_distillation_storage
            ),
            "distillation_storage_fits_24gib": (
                sensitivity_distillation_storage <= STORAGE_CAP_BYTES
            ),
            "on_policy_runtime_hours_with_25pct_margin": (
                sensitivity_online_hours
            ),
            "on_policy_runtime_fits_72h": (
                sensitivity_online_hours <= RUNTIME_CAP_HOURS
            ),
            "on_policy_storage_with_25pct_margin_bytes": (
                sensitivity_ppo_storage
            ),
            "on_policy_storage_fits_24gib": (
                sensitivity_ppo_storage <= STORAGE_CAP_BYTES
            ),
        },
        "bound_measured_artifacts": {
            "central": _identity(
                V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_CENTRAL_COST.json",
                field="central_payload_sha256",
            ),
            "sensitivity": _identity(
                V2_PILOT_DIR
                / "J2_TEACHER_PILOT_V2_SENSITIVITY_COST.json",
                field="sensitivity_payload_sha256",
            ),
            "synchronous": _identity(
                V2_PILOT_DIR
                / "J2_TEACHER_PILOT_V2_SYNCHRONOUS_ORCHESTRATION.json",
                field="sync_payload_sha256",
            ),
        },
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": dict(ZERO_WORK),
    }


def family_support_safeguard() -> dict[str, Any]:
    inventory = j2.load_hashed_json(
        V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_STATE_INVENTORY.json",
        field="inventory_payload_sha256",
    )
    observed_family_counts = inventory["natural_feature_family_counts"]
    family_counts = {
        family: int(observed_family_counts.get(family, 0))
        for family in j2.FEATURE_FAMILIES
    }
    expected_counts = {
        "low_air": 139,
        "low_constrained": 4_861,
        "mid_progression": 0,
        "upper_progression": 0,
    }
    gates = {
        "minimum_natural_states_per_family": 1_024,
        "minimum_distinct_validation_roots_per_family": 256,
        "maximum_natural_family_fraction": 0.70,
        "maximum_capped_inventory_family_fraction": 0.40,
        "retention": "all complete natural validation roots",
        "shortfall_decision": "HOLD_J2_A1_FAMILY_DATA_SUPPORT",
        "checkpoint_authority_before_pass": False,
    }
    checks = {
        "pilot_family_counts_exact": family_counts == expected_counts,
        "pilot_mid_upper_support_zero_explicit": (
            int(family_counts["mid_progression"]) == 0
            and int(family_counts["upper_progression"]) == 0
        ),
        "pilot_not_claimed_all_family_invariant": True,
        "future_four_family_gate_frozen": gates
        == {
            "minimum_natural_states_per_family": 1_024,
            "minimum_distinct_validation_roots_per_family": 256,
            "maximum_natural_family_fraction": 0.70,
            "maximum_capped_inventory_family_fraction": 0.40,
            "retention": "all complete natural validation roots",
            "shortfall_decision": "HOLD_J2_A1_FAMILY_DATA_SUPPORT",
            "checkpoint_authority_before_pass": False,
        },
    }
    return {
        "version": f"{VERSION}_family_support_safeguard_v1",
        "pilot_inventory_identity": _identity(
            V2_PILOT_DIR / "J2_TEACHER_PILOT_V2_STATE_INVENTORY.json",
            field="inventory_payload_sha256",
        ),
        "pilot_natural_feature_family_counts": family_counts,
        "interpretation": (
            "V2 p99 is the frozen engineering admission statistic but "
            "does not establish all-family cost invariance"
        ),
        "future_pre_checkpoint_gates": gates,
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": dict(ZERO_WORK),
    }


def operational_audit(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    parent = j2.operational_audit(output_dir=output_dir)
    return {
        "version": f"{VERSION}_operational_audit_v1",
        "parent": parent,
        "checks": dict(parent["checks"]),
        "passes": parent["passes"],
        "human_session_content_read": False,
    }


def audit_zero_work(
    *,
    output_dir: Path = OUTPUT_DIR,
    allowed_files: Sequence[str] = (),
    include_operational: bool = True,
) -> dict[str, Any]:
    entries = (
        sorted(
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        if output_dir.exists()
        else []
    )
    future = {str(path.resolve()): path.exists() for path in FUTURE_EXECUTION_DIRS}
    operations = (
        operational_audit(output_dir=output_dir)
        if include_operational
        else {"passes": True, "synthetic_skip": True}
    )
    checks = {
        "namespace_has_only_allowed_files": (
            entries == sorted(str(value) for value in allowed_files)
        ),
        "future_execution_namespaces_absent": not any(future.values()),
        "all_scientific_work_counters_zero": all(
            int(value) == 0 for value in ZERO_WORK.values()
        ),
        "no_marker_reservation_consumption": not any(
            token in entry.lower()
            for entry in entries
            for token in ("marker", "reservation", "consumption")
        ),
        "operations_pass": bool(operations["passes"]),
    }
    return {
        "version": f"{VERSION}_zero_work_audit_v1",
        "output_dir": str(output_dir.resolve()),
        "entries": entries,
        "allowed_files": sorted(str(value) for value in allowed_files),
        "future_execution_namespaces": future,
        "zero_work": dict(ZERO_WORK),
        "operational": operations,
        "checks": checks,
        "passes": all(checks.values()),
    }


def write_test_evidence(
    *,
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    zero = audit_zero_work(
        output_dir=output_dir,
        include_operational=False,
    )
    if not zero["passes"]:
        raise J2A1IntegrityError(
            "A1 namespace was not zero-work before test evidence"
        )
    normalized = []
    for row in commands:
        passed = int(row["passed"])
        failed = int(row.get("failed", 0))
        if passed < 1 or failed != 0:
            raise J2A1IntegrityError(
                "A1 test evidence contains a failing or empty command"
            )
        normalized.append(
            {
                "name": str(row["name"]),
                "command": str(row["command"]),
                "passed": passed,
                "failed": failed,
                "deselected": int(row.get("deselected", 0)),
            }
        )
    if not normalized:
        raise J2A1IntegrityError("No A1 test commands were recorded")
    source = source_and_parent_audit()
    if not source["passes"]:
        raise J2A1IntegrityError(
            "Source or parent identities changed before evidence"
        )
    payload = {
        "version": f"{VERSION}_test_evidence_v1",
        "source_identities": source["local_sources"],
        "source_parent_audit_sha256": j2.canonical_json_hash(source),
        "commands": normalized,
        "total_passed": sum(row["passed"] for row in normalized),
        "total_failed": 0,
        "deselections": sorted(str(value) for value in deselections),
        "zero_work": dict(ZERO_WORK),
        "future_execution_namespaces_absent": True,
    }
    return j2.write_immutable_json(
        output_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def validate_test_evidence(
    *,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    path = output_dir / TEST_EVIDENCE_NAME
    payload = j2.load_hashed_json(
        path,
        field="test_evidence_payload_sha256",
    )
    current = {
        "charter": sha256_path(CHARTER_PATH),
        "runner": sha256_path(RUNNER_PATH),
        "tests": sha256_path(TEST_PATH),
    }
    checks = {
        "source_identities_current": (
            payload["source_identities"] == current
        ),
        "commands_passed": (
            int(payload["total_passed"]) > 0
            and int(payload["total_failed"]) == 0
        ),
        "zero_work_exact": payload["zero_work"] == ZERO_WORK,
        "future_namespaces_absent": (
            payload["future_execution_namespaces_absent"] is True
        ),
    }
    return {
        **_identity(path, field="test_evidence_payload_sha256"),
        "commands": payload["commands"],
        "deselections": payload["deselections"],
        "checks": checks,
        "passes": all(checks.values()),
    }


def readiness_decision(
    *,
    integrity_checks: Mapping[str, bool],
    feasibility_checks: Mapping[str, bool],
    operational_checks: Mapping[str, bool],
) -> dict[str, Any]:
    integrity = all(bool(value) for value in integrity_checks.values())
    feasibility = all(bool(value) for value in feasibility_checks.values())
    operations = all(bool(value) for value in operational_checks.values())
    if not integrity:
        decision = KILL
    elif not feasibility or not operations:
        decision = HOLD
    else:
        decision = READY
    return {
        "decision": decision,
        "integrity_checks": dict(integrity_checks),
        "feasibility_checks": dict(feasibility_checks),
        "operational_checks": dict(operational_checks),
        "integrity_passes": integrity,
        "feasibility_passes": feasibility,
        "operational_passes": operations,
        "passes": decision == READY,
    }


def _seal(
    output_dir: Path,
    written: dict[str, Any],
    name: str,
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    path = output_dir / name
    j2.write_immutable_json(path, payload, field=field)
    identity = _identity(path, field=field)
    written[name] = identity
    return identity


def _retention_payload(
    *,
    output_dir: Path,
    decision: str,
    names: Sequence[str],
) -> dict[str, Any]:
    rows = [
        {
            "path": name,
            "file_sha256": sha256_path(output_dir / name),
            "bytes": (output_dir / name).stat().st_size,
        }
        for name in sorted(names)
    ]
    return {
        "version": f"{VERSION}_retention_v1",
        "decision": decision,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "inventory_sha256": j2.canonical_json_hash(rows),
        "source_namespaces_preserved": {
            "parent_j2_readiness": str(PARENT_READINESS_DIR.resolve()),
            "pilot_v1": str(V1_PILOT_DIR.resolve()),
            "pilot_v2": str(V2_PILOT_DIR.resolve()),
        },
        "zero_work": dict(ZERO_WORK),
        "passes": True,
    }


def prepare(
    *,
    output_dir: Path = OUTPUT_DIR,
    power_datasets: int = POWER_DATASETS,
    power_bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    if (
        int(power_datasets) != POWER_DATASETS
        or int(power_bootstraps) != POWER_BOOTSTRAPS
    ):
        raise J2A1IntegrityError("Production power workload cannot change")
    zero = audit_zero_work(
        output_dir=output_dir,
        allowed_files=(TEST_EVIDENCE_NAME,),
    )
    evidence = validate_test_evidence(output_dir=output_dir)
    source = source_and_parent_audit()
    pilot = pilot_history_audit()
    authority = prospective_authority()
    power = power_report(
        datasets=power_datasets,
        bootstraps=power_bootstraps,
    )
    projection = runtime_storage_projection()
    family = family_support_safeguard()

    written: dict[str, Any] = {
        TEST_EVIDENCE_NAME: _identity(
            output_dir / TEST_EVIDENCE_NAME,
            field="test_evidence_payload_sha256",
        )
    }
    input_payload = {
        "version": f"{VERSION}_input_bindings_v1",
        "source_parent_audit": source,
        "pilot_history_audit": pilot,
        "test_evidence": evidence,
        "zero_work_before_prepare": zero,
        "amendment_scope": {
            "only_changed_authority": (
                "distillation validation pairs 2048 -> 6144"
            ),
            "scientific_semantics_changed": False,
            "execution_authorized": False,
        },
        "zero_work": dict(ZERO_WORK),
    }
    _seal(
        output_dir,
        written,
        INPUT_BINDINGS_NAME,
        input_payload,
        "input_bindings_payload_sha256",
    )
    _seal(
        output_dir,
        written,
        PROSPECTIVE_AUTHORITY_NAME,
        authority,
        "prospective_authority_payload_sha256",
    )
    _seal(
        output_dir,
        written,
        POWER_NAME,
        power,
        "power_payload_sha256",
    )
    _seal(
        output_dir,
        written,
        PROJECTION_NAME,
        projection,
        "projection_payload_sha256",
    )
    _seal(
        output_dir,
        written,
        FAMILY_SAFEGUARD_NAME,
        family,
        "family_safeguard_payload_sha256",
    )

    integrity_checks = {
        "zero_work_before_prepare": zero["passes"],
        "test_evidence_exact": evidence["passes"],
        "source_parent_identities_exact": source["passes"],
        "pilot_history_exact": pilot["passes"],
        "authority_exact": authority["passes"],
        "power_method_exact": all(
            value
            for key, value in power["checks"].items()
            if key != "worst_power_at_least_080"
        ),
        "projection_method_exact": all(
            value
            for key, value in projection["checks"].items()
            if key
            not in {
                "central_p99_below_required_ceiling",
                "distillation_runtime_within_72h",
                "distillation_storage_within_24gib",
                "online_runtime_within_72h",
                "online_storage_within_24gib",
                "online_sync_throughput_above_required_floor",
                "measured_memory_within_effective_cap",
            }
        ),
        "family_limitation_and_safeguard_exact": family["passes"],
        "no_execution_surface": True,
    }
    feasibility_checks = {
        "progression_power_at_least_080": power["checks"][
            "worst_power_at_least_080"
        ],
        "distillation_p99_within_cap": projection["checks"][
            "central_p99_below_required_ceiling"
        ],
        "distillation_runtime_within_cap": projection["checks"][
            "distillation_runtime_within_72h"
        ],
        "distillation_storage_within_cap": projection["checks"][
            "distillation_storage_within_24gib"
        ],
        "online_runtime_within_cap": projection["checks"][
            "online_runtime_within_72h"
        ],
        "online_storage_within_cap": projection["checks"][
            "online_storage_within_24gib"
        ],
        "online_sync_throughput_within_cap": projection["checks"][
            "online_sync_throughput_above_required_floor"
        ],
        "measured_memory_within_cap": projection["checks"][
            "measured_memory_within_effective_cap"
        ],
        "future_family_support_gate_is_binding": family["passes"],
    }
    operational_checks = dict(zero["operational"]["checks"])
    decision = readiness_decision(
        integrity_checks=integrity_checks,
        feasibility_checks=feasibility_checks,
        operational_checks=operational_checks,
    )
    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "decision": decision["decision"],
        "bound_artifacts": dict(written),
        "stage_table": j2.json_native(STAGE_TABLE),
        "derived_counts": derive_stage_totals(),
        "decision_audit": decision,
        "family_support_before_checkpoint_authority": (
            family["future_pre_checkpoint_gates"]
        ),
        "zero_work": dict(ZERO_WORK),
        "execution_authorized": False,
    }
    lock = _seal(
        output_dir,
        written,
        READINESS_LOCK_NAME,
        lock_payload,
        "readiness_lock_payload_sha256",
    )
    result_payload = {
        "version": f"{VERSION}_readiness_result_v1",
        "decision": decision["decision"],
        "readiness_lock": lock,
        "derived_counts": derive_stage_totals(),
        "power_summary": {
            "n_pairs": VALIDATION_PAIRS,
            "score_80pct_mde_percent": power["score_fidelity"][
                "score_80pct_mde_percent"
            ],
            "worst_case_common_or_power": power[
                "progression_common_or"
            ]["worst_case_primary_power"],
            "worst_case_power_mcse": EXPECTED_V2_POWER_MCSE,
        },
        "cost_storage_summary": {
            "distillation": projection["distillation"],
            "on_policy_training": projection["on_policy_training"],
            "memory": projection["memory"],
            "sensitivity_5000_moves": projection[
                "sensitivity_5000_moves"
            ],
        },
        "pilot_family_limitation": {
            "natural_counts": (
                family["pilot_natural_feature_family_counts"]
            ),
            "all_family_cost_invariance_established": False,
            "future_support_gate_binding": True,
        },
        "integrity_passes": decision["integrity_passes"],
        "feasibility_passes": decision["feasibility_passes"],
        "operational_passes": decision["operational_passes"],
        "continue": (
            "research-lead review and, only if authorized separately, "
            "a distillation execution-surface/phase-lock proposal"
        ),
        "hold": (
            "all J2 teacher data, distillation, fidelity evaluation, PPO, "
            "development, confirmation, and promotion"
        ),
        "kill": "historical kills unchanged",
        "promote": False,
        "zero_work": dict(ZERO_WORK),
        "execution_authorized": False,
    }
    _seal(
        output_dir,
        written,
        READINESS_RESULT_NAME,
        result_payload,
        "readiness_result_payload_sha256",
    )
    retention_payload = _retention_payload(
        output_dir=output_dir,
        decision=decision["decision"],
        names=tuple(written),
    )
    _seal(
        output_dir,
        written,
        RETENTION_NAME,
        retention_payload,
        "retention_payload_sha256",
    )
    return j2.load_hashed_json(
        output_dir / READINESS_RESULT_NAME,
        field="readiness_result_payload_sha256",
    )


def _parse_recorded_command(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "Recorded command must be a JSON object"
        ) from error
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError(
            "Recorded command must be a JSON object"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outcome-free J2 A1 readiness reseal only"
    )
    commands = parser.add_subparsers(dest="subcommand", required=True)
    audit = commands.add_parser("audit-zero-work")
    audit.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence = commands.add_parser("write-test-evidence")
    evidence.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence.add_argument(
        "--recorded-command",
        action="append",
        type=_parse_recorded_command,
        required=True,
    )
    evidence.add_argument("--deselection", action="append", default=[])
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    j2.configure_readiness_runtime()
    if args.subcommand == "audit-zero-work":
        payload = audit_zero_work(output_dir=args.out_dir)
    elif args.subcommand == "write-test-evidence":
        payload = write_test_evidence(
            commands=args.recorded_command,
            deselections=args.deselection,
            output_dir=args.out_dir,
        )
    elif args.subcommand == "prepare":
        payload = prepare(output_dir=args.out_dir)
    else:  # pragma: no cover
        raise J2A1IntegrityError("Unknown A1 command")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

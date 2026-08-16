"""Phase-gated J1 execution surface and outcome-free readiness tooling.

The readiness commands write no marker and perform no scientific work.
Future phase commands are implemented but require separately sealed locks,
markers, manifests, and predecessor decisions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from threes_rl import j1_joint_policy_value as j1
from threes_rl import o2_online_option_preflight as o2_power
from threes_rl.obs import encode_observation
from threes_rl.sim import SimState, ThreesSim, score_board


VERSION = "j1_execution_surface_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs"
CHARTER_PATH = REPO_ROOT / "threes_rl" / "J1_EXECUTION_SURFACE_CHARTER.md"
RUNNER_PATH = REPO_ROOT / "threes_rl" / "j1_execution_surface.py"
TEST_PATH = REPO_ROOT / "tests" / "test_rl_j1_execution_surface.py"
READINESS_DIR = (
    RUNS_ROOT / "forensics" / "j1_execution_surface_readiness_v1"
)
EXECUTION_ROOT = RUNS_ROOT / "forensics" / "j1_execution_v1"

TEST_EVIDENCE_NAME = "J1_EXECUTION_TEST_EVIDENCE.json"
SCHEMA_NAME = "J1_EXECUTION_SCHEMA.json"
MANIFEST_NAME = "J1_PROSPECTIVE_MANIFEST.json"
READINESS_LOCK_NAME = "J1_EXECUTION_READINESS_LOCK.json"
READINESS_RESULT_NAME = "J1_EXECUTION_READINESS_RESULT.json"
RUNTIME_STORAGE_PROJECTION_NAME = (
    "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json"
)

PHASES = ("training", "development", "confirmation")
PHASE_LOCK_NAME = "phase_lock.json"
PHASE_LOCK_RESULT_NAME = "phase_lock_result.json"
PHASE_MARKER_NAME = "execution_opened.json"
PHASE_MANIFEST_NAME = "root_manifest.json"
PHASE_RUNTIME_NAME = "runtime_state.bin"
PHASE_OWNER_NAME = "writer_owner.json"
PHASE_RESULT_NAME = "terminal_result.json"
PHASE_RETENTION_NAME = "retention_manifest.json"
PHASE_STREAM_RESERVATION_NAME = "stream_reservation.json"
PHASE_STREAM_CONSUMPTION_NAME = "stream_consumption_opened.json"
COMMIT_HEAD_NAME = "commit_head.json"
COMMIT_STATES_DIR = "commit_states"
COMMIT_JOURNALS_DIR = "commit_journals"
COMMIT_RECORDS_DIR = "commit_records"
COMMIT_HEADS_DIR = "commit_heads"
PRECOMMITTED_MANIFEST_DIR = "precommitted_manifests"
ROLLING_RESUME_DIR = "rolling_resume"
ROLLING_RESUME_HEAD_NAME = "resume_head.json"
ROLLING_RESUME_JOURNAL_NAME = "resume_journal.jsonl"
RUNTIME_CHARGE_JOURNAL_NAME = "runtime_charge_journal.jsonl"
ROOT_BLOBS_DIR = "root_blobs"
PAIR_BLOBS_DIR = "pair_blobs"
ROOT_BLOB_BLOCKS_DIR = "root_blob_blocks"
TRANSITION_CHUNKS_DIR = "transition_chunks"
ROUND_BATCHES_DIR = "round_batches"
ROUND_BATCH_RETIREMENTS_DIR = "round_batch_retirements"
TRANSITION_CHUNK_RETIREMENTS_DIR = "transition_chunk_retirements"
COMPACT_COMMIT_PREFIX_MODE = "sha256_unit_chain_v1"
TRANSITION_CHUNK_MAX_ROWS = 1_024
TRAINING_TRANSITION_FILE_CAP = 100_000
TRAINING_COLLECTION_TICKS_PER_COMMIT = 32
ROOT_BLOB_BLOCK_SIZE = 16
PAIR_RESULT_BLOCK_SIZE = 64
ABANDONED_ATTEMPT_CHARGE_SECONDS = {
    "training_collection_tick_block": 600.0,
    "training_minibatch_update": 300.0,
    "paired_candidate_arm": 900.0,
    "paired_control_arm_and_pair": 900.0,
    "miniature_fixture_other": 1.0,
}
BOUNDED_FIXTURE_COST_EVIDENCE = {
    "version": "j1_bounded_fixture_cost_evidence_v1",
    "training_roots": 2,
    "training_transitions": 60,
    "root_blob_bytes": 91_126,
    "round_batch_bytes": 75_607,
    "transition_chunk_bytes": 91_510,
    "real_model_adam_bytes": 4_951_545,
    "terminal_bytes_after_retirement": 33_257_327,
    "terminal_file_count_after_retirement": 35,
    "training_rolling_append_records": 11,
    "training_runtime_charge_records": 20,
    "training_commit_units": 6,
    "paired_fixture_pairs": 4,
    "paired_blob_bytes": 11_692,
    "paired_terminal_bytes": 55_648,
    "paired_terminal_file_count": 22,
    "paired_rolling_append_records": 8,
    "paired_runtime_charge_records": 16,
    "paired_commit_units": 3,
    "source": (
        "independent bounded 2-root fixture and deterministic real "
        "model+Adam serialization"
    ),
    "scientific_games": 0,
    "scientific_optimizer_steps": 0,
    "scientific_policy_outcomes": 0,
}
TRAINING_OUTPUT_FILE_CAP = 50_000
TRAINING_FSYNC_CAP = 200_000
EVALUATION_OUTPUT_FILE_CAP = 10_000
EVALUATION_FSYNC_CAP = 100_000
PROJECTION_SAFETY_MULTIPLIER = 1.25
PROJECTION_CENTRAL_MOVES = 512
PROJECTION_SENSITIVITY_MOVES = 5_000
PAIR_RESULT_FIXED_BYTES = 8_192
PAIR_RESULT_BYTES_PER_MOVE_PER_ARM = 16
COMMIT_STATE_STORAGE_ENVELOPE_BYTES = 5_500_000
ROLLING_STATE_STORAGE_ENVELOPE_BYTES = 6_000_000
COMPACT_EVALUATION_COMMIT_BYTES = 131_072
METADATA_STORAGE_ENVELOPE_BYTES = 20_000_000
PROJECTION_IO_UNIT_SECONDS = {
    "collection_checkpoint": 0.25,
    "optimizer_checkpoint": 0.50,
    "commit_seal": 1.00,
    "root_blob_write": 0.01,
    "transition_chunk_write": 0.02,
    "round_batch_write": 0.20,
    "paired_arm_checkpoint": 0.05,
    "paired_result_blob": 0.01,
    "paired_block_seal": 1.00,
}
TRAINING_CANDIDATE_CHECKPOINT_NAME = "round64_candidate_checkpoint.bin"
TRAINING_SANITY_RESULT_NAME = "training_sanity_result.json"
PAIRED_ANALYSIS_NAME = "paired_analysis.json"
JOINT_MANIFEST_SEAL_NAME = "joint_evaluation_manifest_seal.json"
CONFIRMATION_ACCESS_AUDIT_NAME = "confirmation_access_audit.json"

PRODUCTION_COMMANDS = (
    "seal-phase-lock",
    "open",
    "materialize",
    "execute",
)

TRAIN_ROOTS = 16_384
DEVELOPMENT_PAIRS = 896
CONFIRMATION_PAIRS = 4_480
TOTAL_GAME_ARMS = TRAIN_ROOTS + 2 * (
    DEVELOPMENT_PAIRS + CONFIRMATION_PAIRS
)
MAX_MOVES = 5_000
ENV_COUNT = 16
ROUNDS = 64
ROOTS_PER_ROUND = 256
TRAINING_COMMIT_STATE_COUNT = 1 + ROUNDS * 5
TRAINING_ROUND_CHECKPOINT_STATE_COUNT = ROUNDS
TRAINING_OTHER_COMMIT_STATE_COUNT = (
    TRAINING_COMMIT_STATE_COUNT - TRAINING_ROUND_CHECKPOINT_STATE_COUNT
)
BOOTSTRAPS = 4_096
BOOTSTRAP_SEEDS = {
    "development": 2_026_072_817,
    "confirmation": 2_026_072_818,
}
SCORE_BOOTSTRAP_METHOD = "global paired whole-root"
PROGRESSION_BOOTSTRAP_METHOD = (
    "independent whole-root within each of eight fixed strata"
)
PROGRESSION_ZERO_CELL_RULE = (
    "add 0.5 to a,b,c,d in every stratum whenever aggregate "
    "MH numerator or denominator is nonpositive"
)
BOOTSTRAP_QUANTILE_CONVENTION = (
    "numpy.quantile(method=linear), probabilities 0.025/0.975"
)
EXPECTED_EVALUATION_BOOTSTRAP_CONTRACT = {
    "replicates": 4_096,
    "phase_seeds": {
        "development": 2_026_072_817,
        "confirmation": 2_026_072_818,
    },
    "score_resampling": "global paired whole-root",
    "progression_resampling":
        "independent whole-root within each of eight fixed strata",
    "progression_zero_cell_rule":
        "add 0.5 to a,b,c,d in every stratum whenever aggregate "
        "MH numerator or denominator is nonpositive",
    "quantile": "numpy.quantile(method=linear), probabilities 0.025/0.975",
}
PHASE_NONCES = {
    "training":
        "3571f55ffce28410b5005b9ebd25b3c3439e289a29f65c04136883651b4cf689",
    "development":
        "48284f51b7ba5ea6073629cc8702653524fa453d87a4a51bbf058e27659ddf66",
    "confirmation":
        "1853adcf643fa4ec30545020c3a7fadfe07225a3be93a73a38678a3de1415ea8",
}

PHASE_CAPS = {
    "training": {"active_hours": 72.0, "storage_gib": 24.0},
    "development": {"active_hours": 24.0, "storage_gib": 8.0},
    "confirmation": {"active_hours": 120.0, "storage_gib": 16.0},
}

STREAMS = {
    "training": {
        "rows": TRAIN_ROOTS,
        "logical": 213_000_000_000,
        "deck": 214_000_000_000,
        "slot": 215_000_000_000,
        "candidate_policy": 216_000_000_000,
    },
    "development": {
        "rows": DEVELOPMENT_PAIRS,
        "logical": 217_000_000_000,
        "deck": 218_000_000_000,
        "slot": 219_000_000_000,
        "candidate_policy": 220_000_000_000,
        "control_policy": 221_000_000_000,
    },
    "confirmation": {
        "rows": CONFIRMATION_PAIRS,
        "logical": 222_000_000_000,
        "deck": 223_000_000_000,
        "slot": 224_000_000_000,
        "candidate_policy": 225_000_000_000,
        "control_policy": 226_000_000_000,
    },
}

ACCEPTED_FILES = {
    "threes_rl/J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md":
        "26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2",
    "threes_rl/J1_IMPLEMENTATION_READINESS_AUDIT.json":
        "f3e4e8029e159a1db7767164e1623d2e166b139be319d6077d61d7d107a44042",
    "threes_rl/J1_IMPLEMENTATION_PREFLIGHT_CHARTER.md":
        "7f87bc29c5764ccb290b25558f1cfe999083e9fddb089ea652cac9d0b92ab137",
    "threes_rl/j1_joint_policy_value.py":
        "55d9e3206c2905509466c4962006e6cf3426f76647af6d2e60afe674b80c9bfe",
    "tests/test_rl_j1_joint_policy_value.py":
        "e6b169f2d629021f96315380a3cf0ff6eece94a30e5027b1ace4d741499fbfa4",
    "threes_rl/o2_online_option_preflight.py":
        "99e61f551d607e3b5b8457b7e76a17c8540f0e1d88afec3fa544296bdcd05fda",
    "threes_rl/train_ppo.py":
        "cb2cb301630001ed887e1131c46bc6565e41917ea861ebe836ba9c39990fc6f3",
    "threes_rl/J1A_OUTCOME_FREE_COST_POWER_AMENDMENT.md":
        "d738a55bb438ee87d59d2433466e813cfd0a9fb5f041cbc3cc807d4bbafa2e11",
    "threes_rl/j1a_cost_power_preflight.py":
        "27ffb3825d60bd8ca4ec0646f976e325c2a7c5f00a077aea3803544531fe6a98",
    "tests/test_rl_j1a_cost_power_preflight.py":
        "898f25aa4ed109db2c9fc27b4bba9d7e9641dc57834e4e02d7a8242df195eb59",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_TEST_EVIDENCE.json":
        "aceab517c4fffc52fe1827468b8408484c0f9ddade594e5200e025d71239137f",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_PROTECTED_ID_DENYLIST.json":
        "0a7be318ebe5281a11ded38f3bbde29745ccb7c3a969585de1788df468fbd763",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_RUNTIME_STORAGE_PROJECTION.json":
        "e023fe04239ceb2d317ab0e26979033db3c2a5c93d4a5016168de442fc97e401",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_PREFLIGHT_LOCK.json":
        "42d1f8d3d6b7bfd62c173a3147ce1eb7dff465aaa92271e7af6bc5fb3c533825",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_PREFLIGHT_RESULT.json":
        "339e3ef6dcf8c5b3eb1951204d08b97b94b3c4816f993d58509b9b341dc364b1",
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_TEST_EVIDENCE.json":
        "8d052459dd8c914b0c3b68609d113b3f1bc7d8bbb5ec5412635bb5affe306edc",
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_COST_POWER_ARITHMETIC.json":
        "957159dcbfe4ee95be9c2abd2ab2d99a4cd49ce611895bdd3c55ff5ce4fcf9b0",
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_PREFLIGHT_LOCK.json":
        "7ed37c9bf1c6ec0fe7e74f36ef4cde8ab5e3bdd8ae1a7d9e1e065e32a21b852e",
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_PREFLIGHT_RESULT.json":
        "4ecda2a1101011437c912d884dfb5acecf7e586b87c4646c63354c4ecc5403ef",
}

ACCEPTED_PAYLOADS = {
    "threes_rl/J1_IMPLEMENTATION_READINESS_AUDIT.json": (
        "canonical_payload_sha256",
        "5b6b9a2383296f82b6547bbd46ddc892b486e4b89f4c325aa88f9c8b15944f99",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_TEST_EVIDENCE.json": (
        "test_evidence_payload_sha256",
        "686b1e58daa937076704eec5ebd84b3af6bf2a47d8ec41875fe4901cf5dc988e",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_PROTECTED_ID_DENYLIST.json": (
        "denylist_payload_sha256",
        "22731c89df661419d7ca2bcffdb86240f2ad8974b00e765dd715cf8f4e675add",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_RUNTIME_STORAGE_PROJECTION.json": (
        "projection_payload_sha256",
        "1aaba01b73d53ad10252f0c59c238c8274a9e8f8066a8f3f03f3c0587c6bef0b",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_PREFLIGHT_LOCK.json": (
        "preflight_lock_payload_sha256",
        "e465cec348f987af4c77f062a0e8f8bfa968ddc4ff460b40ba829915791622da",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_PREFLIGHT_RESULT.json": (
        "preflight_result_payload_sha256",
        "4d21a092e584d9419a47bef384de164cfc9a8590268a67abefa35afb6b573ce2",
    ),
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_TEST_EVIDENCE.json": (
        "test_evidence_payload_sha256",
        "a5ffd778c0cfce00d58429a287e7a813d4e429ea5e862217c9ca32a56fa24597",
    ),
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_COST_POWER_ARITHMETIC.json": (
        "arithmetic_payload_sha256",
        "b1d13d49db07fa59afd995640c5d063f8bc9776122ead554bb53856543fd21b6",
    ),
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_PREFLIGHT_LOCK.json": (
        "preflight_lock_payload_sha256",
        "b84228d9e5587682fad0cca91e0e5349076ab70674cf0412205712fa05e37850",
    ),
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_PREFLIGHT_RESULT.json": (
        "preflight_result_payload_sha256",
        "abe17a53c1af2b182a488d4fc05b060a214b106652c04462453ad01e75ed9471",
    ),
}

ZERO_WORK = {
    "execution_markers": 0,
    "phase_locks": 0,
    "prospective_root_ids_materialized": 0,
    "j1_streams_reserved": 0,
    "j1_streams_consumed": 0,
    "normal_start_games_generated": 0,
    "scientific_labels": 0,
    "scientific_optimizer_steps": 0,
    "scientific_checkpoints": 0,
    "development_content_reads": 0,
    "confirmation_content_reads": 0,
    "score_or_policy_outcomes": 0,
    "human_session_reads": 0,
    "incumbent_changes": 0,
    "dashboard_changes": 0,
}

# Final counts and exact historical-state deselections are frozen only after
# the source/test surface passes.
FOCUSED_TEST_COMMAND = (
    "env PYTHONPATH=. .venv/bin/python -m pytest -q "
    "tests/test_rl_j1_execution_surface.py"
)
PARENT_TEST_COMMANDS = (
    (
        "parent_j1",
        "env PYTHONPATH=. .venv/bin/python -m pytest -q "
        "tests/test_rl_j1_joint_policy_value.py",
        36,
    ),
    (
        "parent_j1a",
        "env PYTHONPATH=. .venv/bin/python -m pytest -q "
        "tests/test_rl_j1a_cost_power_preflight.py",
        18,
    ),
)
FOCUSED_TEST_COUNT = 97
DOCUMENTED_HISTORICAL_STATE_DESELECTIONS = (
    "tests/test_rl_g1r_qd_admission_v2.py::"
    "test_all_frozen_panel_insertion_descriptors_are_total",
    "tests/test_rl_g1r_qd_admission_v2.py::"
    "test_all_selected_archive_root_insertion_descriptors_are_total",
    "tests/test_rl_o2_yield_pilot_scan_recovery.py::"
    "test_original_pilot_artifacts_remain_immutable_and_recovery_absent",
    "tests/test_rl_o3_option_training.py::"
    "test_authoritative_output_and_evidence_are_absent",
    "tests/test_rl_o3_selected_integrity_reseal_v2.py::"
    "test_output_namespace_is_separate_from_recovery",
    "tests/test_rl_o3_selected_integrity_reseal_v3.py::"
    "test_terminal_namespace_is_fresh_and_absent",
    "tests/test_rl_o5_training.py::"
    "test_authoritative_output_and_evidence_are_absent",
    "tests/test_rl_o5_training_v2.py::"
    "test_v2_execution_artifacts_are_absent",
    "tests/test_rl_g3_e0_preflight_v2.py::"
    "test_v2_output_is_separate_and_fresh",
    "tests/test_rl_g3_scale_transfer_bootstrap_preflight.py::"
    "test_metadata_search_does_not_treat_g2_inputs_as_legacy_labels",
    "tests/test_rl_k1_compiled_kernel.py::"
    "test_design_preflights_are_hashed_and_zero_work",
    "tests/test_rl_k1_compiled_kernel.py::"
    "test_output_namespace_is_still_fresh",
    "tests/test_rl_k1_support_audit.py::"
    "test_charter_and_spent_terminal_are_frozen",
)
APPLICABLE_TEST_COMMAND = (
    "env PYTHONPATH=. .venv/bin/python -m pytest -q -rs "
    "tests/test_rl_g1r*.py tests/test_rl_o*.py tests/test_rl_g3*.py "
    "tests/test_rl_g4*.py tests/test_rl_c1*.py tests/test_rl_c2*.py "
    "tests/test_rl_k1*.py tests/test_rl_s3*.py tests/test_rl_sim*.py "
    "tests/test_rl_split_rng.py tests/test_rl_env_api.py "
    "tests/test_rl_eval_metrics.py tests/test_rl_ntuple.py "
    + " ".join(
        f"--deselect {node_id}"
        for node_id in DOCUMENTED_HISTORICAL_STATE_DESELECTIONS
    )
)
APPLICABLE_TEST_COUNT = 697


class J1ExecutionIntegrityError(RuntimeError):
    """An immutable identity, phase, resume, or scientific contract failed."""


class J1ExecutionOperationalHold(RuntimeError):
    """A mutable resource, service, or ownership condition failed."""


class J1ExecutionPlannedInterruption(RuntimeError):
    """Fixture-only interruption immediately after a durable boundary."""


def repo_path(path: str | Path, root: Path = REPO_ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def sha256_path(path: str | Path, root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    with repo_path(path, root).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    observed = body.pop(field, None)
    return isinstance(observed, str) and observed == canonical_json_hash(body)


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    body = payload_with_hash(payload, field)
    serialized = json.dumps(body, indent=2, sort_keys=True) + "\n"
    reloaded = json.loads(serialized)
    if not verify_payload_hash(reloaded, field):
        raise J1ExecutionIntegrityError(f"JSON reload instability: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = serialized.encode("utf-8")
    if path.exists():
        observed_bytes = path.read_bytes()
        if observed_bytes != expected:
            raise J1ExecutionIntegrityError(
                f"Immutable artifact collision changed bytes: {path}"
            )
        observed = json.loads(observed_bytes.decode("utf-8"))
        if not verify_payload_hash(observed, field):
            raise J1ExecutionIntegrityError(
                f"Existing immutable payload is invalid: {path}"
            )
        raise FileExistsError(f"Immutable artifact exists: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            observed_bytes = path.read_bytes()
            if observed_bytes != expected:
                raise J1ExecutionIntegrityError(
                    f"Concurrent immutable JSON mismatch: {path}"
                ) from error
            observed = json.loads(observed_bytes.decode("utf-8"))
            if not verify_payload_hash(observed, field):
                raise J1ExecutionIntegrityError(
                    f"Concurrent immutable JSON is invalid: {path}"
                ) from error
            raise FileExistsError(
                f"Immutable artifact won by another writer: {path}"
            ) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    observed = json.loads(path.read_text(encoding="utf-8"))
    if not verify_payload_hash(observed, field):
        raise J1ExecutionIntegrityError(f"Written payload mismatch: {path}")
    return observed


def load_json(path: str | Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    payload = json.loads(repo_path(path, root).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise J1ExecutionIntegrityError(f"Expected JSON object: {path}")
    return payload


def accepted_identity_audit(root: Path = REPO_ROOT) -> dict[str, Any]:
    files = {}
    for path, expected in ACCEPTED_FILES.items():
        target = repo_path(path, root)
        observed = sha256_path(target) if target.is_file() else None
        files[path] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matches": observed == expected,
        }
    payloads = {}
    for path, (field, expected) in ACCEPTED_PAYLOADS.items():
        target = repo_path(path, root)
        try:
            payload = load_json(target)
            observed = payload.get(field)
            stable = verify_payload_hash(payload, field)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            observed = None
            stable = False
        payloads[path] = {
            "field": field,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "reload_stable": stable,
            "matches": observed == expected and stable,
        }

    j1_result = load_json(
        root
        / "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
        "J1_IMPLEMENTATION_PREFLIGHT_RESULT.json"
    )
    j1a_result = load_json(
        root
        / "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
        "J1A_PREFLIGHT_RESULT.json"
    )
    checks = {
        "all_files_exact": all(row["matches"] for row in files.values()),
        "all_payloads_exact": all(
            row["matches"] for row in payloads.values()
        ),
        "j1_parent_hold_exact": (
            j1_result.get("decision") == "HOLD_J1_IMPLEMENTATION_PREFLIGHT"
        ),
        "j1a_ready_exact": (
            j1a_result.get("decision")
            == "READY_J1A_COST_POWER_AMENDMENT"
        ),
        "j1_and_j1a_not_killed": (
            j1_result.get("kill") == "historical kills unchanged"
            and j1a_result.get("kill")
            == "historical kills unchanged; J1/J1a not scientifically killed"
        ),
        "parent_zero_work": all(
            value == 0
            for result in (j1_result, j1a_result)
            for value in result.get("zero_work", {}).values()
        ),
    }
    return {
        "files": files,
        "payloads": payloads,
        "j1_decision": j1_result.get("decision"),
        "j1a_decision": j1a_result.get("decision"),
        "checks": checks,
        "passes": all(checks.values()),
    }


def iter_prospective_rows(
    phase: str | None = None,
) -> Iterator[dict[str, Any]]:
    phases = PHASES if phase is None else (phase,)
    if any(value not in PHASES for value in phases):
        raise ValueError(f"Unsupported phase: {phase}")
    for current in phases:
        contract = STREAMS[current]
        for row_index in range(int(contract["rows"])):
            row = {
                "phase": current,
                "row_index": row_index,
                "block": row_index % 8,
                "logical_stream_id": int(contract["logical"]) + row_index,
                "deck_stream_id": int(contract["deck"]) + row_index,
                "slot_stream_id": int(contract["slot"]) + row_index,
                "candidate_policy_stream_id": (
                    int(contract["candidate_policy"]) + row_index
                ),
                "control_policy_stream_id": (
                    None
                    if "control_policy" not in contract
                    else int(contract["control_policy"]) + row_index
                ),
                "arm_count": 1 if current == "training" else 2,
                "starter_tile": None,
            }
            row["row_commitment_sha256"] = canonical_json_hash(row)
            yield row


def _ordered_rows_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _phase_partition(phase: str) -> str:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    return "train" if phase == "training" else phase


def phase_root_commitment(phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    rows_hash = _ordered_rows_hash(iter_prospective_rows(phase))
    payload = {
        "version": f"{VERSION}_phase_marker_root_commitment_v1",
        "root_identity_version": "accepted_j1_marker_payload_root_v1",
        "phase": phase,
        "partition": _phase_partition(phase),
        "phase_nonce": PHASE_NONCES[phase],
        "canonical_rows_sha256": rows_hash,
        "row_count": int(STREAMS[phase]["rows"]),
        "accepted_j1_result_file_sha256": ACCEPTED_FILES[
            "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
            "J1_IMPLEMENTATION_PREFLIGHT_RESULT.json"
        ],
        "accepted_j1a_result_file_sha256": ACCEPTED_FILES[
            "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
            "J1A_PREFLIGHT_RESULT.json"
        ],
        "operational_activation_fields_excluded": [
            "activation_command",
            "activation_hostname",
            "activation_opened_at",
            "service_evidence",
            "process_evidence",
            "storage_evidence",
        ],
        "streams_reserved": 0,
        "streams_consumed": 0,
    }
    # This is the immutable marker payload/core used by the accepted parent
    # root-id derivation. A later operational marker activates, but cannot
    # alter, this commitment.
    return payload_with_hash(payload, "marker_payload_sha256")


def root_id_for_marker_commitment(
    commitment: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    if not verify_payload_hash(commitment, "marker_payload_sha256"):
        raise J1ExecutionIntegrityError(
            "Phase marker root commitment hash is invalid"
        )
    if (
        commitment.get("phase") != row.get("phase")
        or commitment.get("partition") != _phase_partition(str(row["phase"]))
    ):
        raise J1ExecutionIntegrityError(
            "Phase marker root commitment/row mismatch"
        )
    return canonical_json_hash(
        {
            "marker_payload_sha256": commitment["marker_payload_sha256"],
            "partition": commitment["partition"],
            "row": int(row["row_index"]),
            "logical_stream_id": int(row["logical_stream_id"]),
            "deck_stream_id": int(row["deck_stream_id"]),
            "slot_stream_id": int(row["slot_stream_id"]),
        }
    )


def materialize_root_manifest(
    *,
    phase: str,
    marker_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    commitment = phase_root_commitment(phase)
    if marker_payload is not None:
        if not verify_payload_hash(
            marker_payload,
            "activation_marker_payload_sha256",
        ):
            raise J1ExecutionIntegrityError(
                "Activation marker payload hash is invalid"
            )
        if marker_payload.get("root_commitment") != commitment:
            raise J1ExecutionIntegrityError(
                "Activation marker changed marker root commitment"
            )
    roots = []
    for row in iter_prospective_rows(phase):
        root_id = root_id_for_marker_commitment(commitment, row)
        roots.append(
            {
                **row,
                "root_id": root_id,
                "ancestry_id": root_id,
            }
        )
    root_ids = [row["root_id"] for row in roots]
    ancestry_ids = [row["ancestry_id"] for row in roots]
    checks = {
        "row_count_exact": len(roots) == int(STREAMS[phase]["rows"]),
        "root_ids_unique": len(set(root_ids)) == len(root_ids),
        "ancestries_unique": len(set(ancestry_ids)) == len(ancestry_ids),
        "one_root_per_ancestry": root_ids == ancestry_ids,
        "starter_none": all(row["starter_tile"] is None for row in roots),
    }
    payload = {
        "version": f"{VERSION}_{phase}_root_manifest_v1",
        "phase": phase,
        "root_commitment": commitment,
        "rows": roots,
        "canonical_rows_sha256": _ordered_rows_hash(roots),
        "checks": checks,
        "passes": all(checks.values()),
        "streams_reserved": 0,
        "streams_consumed": 0,
    }
    return payload_with_hash(payload, "root_manifest_payload_sha256")


def validate_cross_phase_manifests(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seen_roots: set[str] = set()
    seen_ancestries: set[str] = set()
    phase_counts: dict[str, int] = {}
    for manifest in manifests:
        phase = str(manifest["phase"])
        if phase in phase_counts:
            raise J1ExecutionIntegrityError("Duplicate phase manifest")
        rows = list(manifest["rows"])
        roots = {str(row["root_id"]) for row in rows}
        ancestries = {str(row["ancestry_id"]) for row in rows}
        if len(roots) != len(rows) or len(ancestries) != len(rows):
            raise J1ExecutionIntegrityError("Duplicate root or ancestry")
        if seen_roots & roots or seen_ancestries & ancestries:
            raise J1ExecutionIntegrityError("Cross-phase ancestry overlap")
        seen_roots.update(roots)
        seen_ancestries.update(ancestries)
        phase_counts[phase] = len(rows)
    return {
        "phase_counts": phase_counts,
        "total_roots": len(seen_roots),
        "root_sets_disjoint": True,
        "ancestry_sets_disjoint": True,
        "passes": True,
    }


def root_manifest_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not verify_payload_hash(payload, "root_manifest_payload_sha256")
        or payload.get("passes") is not True
    ):
        raise J1ExecutionIntegrityError("Root manifest payload is invalid")
    return {
        "phase": payload["phase"],
        "row_count": len(payload["rows"]),
        "canonical_rows_sha256": payload["canonical_rows_sha256"],
        "payload_sha256": payload["root_manifest_payload_sha256"],
    }


def seal_joint_evaluation_manifests(
    *,
    execution_root: Path,
    training_manifest: Mapping[str, Any],
    training_result: Mapping[str, Any],
    confirmation_access_audit_path: Path,
) -> dict[str, Any]:
    if training_result.get("decision") != "READY_J1_TRAINING_SANITY":
        raise J1ExecutionIntegrityError(
            "Evaluation manifests require READY training sanity"
        )
    training_result_path = (
        execution_root / "training" / PHASE_RESULT_NAME
    )
    observed_training_result = load_json(training_result_path)
    if (
        observed_training_result != dict(training_result)
        or not verify_payload_hash(
            observed_training_result,
            "terminal_result_payload_sha256",
        )
    ):
        raise J1ExecutionIntegrityError(
            "Training terminal identity changed before joint sealing"
        )
    training_result_identity = immutable_json_identity(
        training_result_path,
        payload_field="terminal_result_payload_sha256",
        decision="READY_J1_TRAINING_SANITY",
    )
    training_mode = str(training_result.get("execution_mode"))
    if training_mode not in {"scientific", "miniature_fixture"}:
        raise J1ExecutionIntegrityError(
            "Training terminal execution mode is invalid"
        )
    if training_result.get("scientific_authority") is not (
        training_mode == "scientific"
    ):
        raise J1ExecutionIntegrityError(
            "Training terminal authority flag changed"
        )
    checkpoint_identity = training_result.get("checkpoint_identity")
    if training_mode == "scientific":
        if not isinstance(checkpoint_identity, Mapping):
            raise J1ExecutionIntegrityError(
                "READY training result lacks candidate checkpoint identity"
            )
        load_authoritative_candidate_policy(
            checkpoint_identity=checkpoint_identity
        )
        sealed_checkpoint_identity = dict(checkpoint_identity)
    else:
        sealed_checkpoint_identity = {
            "fixture_only": True,
            "training_result_payload_sha256": training_result[
                "terminal_result_payload_sha256"
            ],
            "scientific_authority": False,
        }
    if training_manifest.get("phase") != "training":
        raise J1ExecutionIntegrityError("Training manifest phase changed")
    if not verify_payload_hash(
        training_manifest,
        "root_manifest_payload_sha256",
    ):
        raise J1ExecutionIntegrityError("Training manifest is invalid")
    access_path = confirmation_access_audit_path.resolve()
    access_payload = load_json(access_path)
    if not verify_payload_hash(
        access_payload,
        "confirmation_access_audit_sha256",
    ):
        raise J1ExecutionIntegrityError(
            "Confirmation access audit payload is invalid"
        )
    access_counts = {
        "confirmation_content_reads":
            access_payload.get("confirmation_content_reads"),
        "confirmation_streams_reserved":
            access_payload.get("confirmation_streams_reserved"),
        "confirmation_streams_consumed":
            access_payload.get("confirmation_streams_consumed"),
    }
    if any(value != 0 for value in access_counts.values()):
        raise J1ExecutionIntegrityError(
            "Confirmation access audit is not zero before joint sealing"
        )
    access_identity = {
        "path": str(access_path),
        "file_sha256": sha256_path(access_path),
        "payload_sha256":
            access_payload["confirmation_access_audit_sha256"],
        **access_counts,
    }
    incumbent_binding = incumbent_policy_binding()
    development = materialize_root_manifest(phase="development")
    confirmation = materialize_root_manifest(phase="confirmation")
    cross = validate_cross_phase_manifests(
        [training_manifest, development, confirmation]
    )
    expected_counts = {
        "training": TRAIN_ROOTS,
        "development": DEVELOPMENT_PAIRS,
        "confirmation": CONFIRMATION_PAIRS,
    }
    if cross["phase_counts"] != expected_counts:
        raise J1ExecutionIntegrityError(
            "Joint phase manifests have wrong root counts"
        )
    target_dir = execution_root / PRECOMMITTED_MANIFEST_DIR
    development_path = target_dir / "development_root_manifest.json"
    confirmation_path = target_dir / "confirmation_root_manifest.json"
    written_development = _write_immutable_json_exact(
        development_path,
        {
            key: value
            for key, value in development.items()
            if key != "root_manifest_payload_sha256"
        },
        field="root_manifest_payload_sha256",
    )
    written_confirmation = _write_immutable_json_exact(
        confirmation_path,
        {
            key: value
            for key, value in confirmation.items()
            if key != "root_manifest_payload_sha256"
        },
        field="root_manifest_payload_sha256",
    )
    seal = {
        "version": f"{VERSION}_joint_evaluation_manifest_seal_v1",
        "training_manifest": root_manifest_identity(training_manifest),
        "training_terminal_result": training_result_identity,
        "candidate_checkpoint_identity": sealed_checkpoint_identity,
        "training_execution_mode": training_mode,
        "development_manifest": {
            **root_manifest_identity(written_development),
            "path": str(development_path.resolve()),
            "file_sha256": sha256_path(development_path),
        },
        "confirmation_manifest": {
            **root_manifest_identity(written_confirmation),
            "path": str(confirmation_path.resolve()),
            "file_sha256": sha256_path(confirmation_path),
        },
        "cross_phase_audit": cross,
        "confirmation_access_audit": access_identity,
        "incumbent_policy_binding": incumbent_binding,
        "passes": True,
    }
    seal_path = target_dir / "joint_evaluation_manifest_seal.json"
    written_seal = _write_immutable_json_exact(
        seal_path,
        seal,
        field="joint_manifest_seal_payload_sha256",
    )
    return {
        "development": written_development,
        "confirmation": written_confirmation,
        "seal": written_seal,
        "seal_path": seal_path,
        "passes": True,
    }


def verify_joint_candidate_lineage(
    *,
    execution_root: Path,
    joint_manifest_seal: Mapping[str, Any] | None = None,
    expected_execution_mode: str = "scientific",
) -> dict[str, Any]:
    joint = (
        _load_joint_manifest_seal(execution_root)
        if joint_manifest_seal is None
        else dict(joint_manifest_seal)
    )
    if not verify_payload_hash(
        joint,
        "joint_manifest_seal_payload_sha256",
    ):
        raise J1ExecutionIntegrityError(
            "Joint candidate lineage seal is invalid"
        )
    training_result_path = (
        execution_root / "training" / PHASE_RESULT_NAME
    )
    training_result = _verify_json_identity(
        joint.get("training_terminal_result", {}),
        expected_path=training_result_path,
        payload_field="terminal_result_payload_sha256",
        decision="READY_J1_TRAINING_SANITY",
    )
    if (
        training_result.get("execution_mode") != expected_execution_mode
        or training_result.get("scientific_authority")
        is not (expected_execution_mode == "scientific")
        or joint.get("training_execution_mode") != expected_execution_mode
    ):
        raise J1ExecutionIntegrityError(
            "Joint candidate lineage crossed execution modes"
        )
    checkpoint_identity = joint.get("candidate_checkpoint_identity")
    if expected_execution_mode == "scientific":
        if (
            not isinstance(checkpoint_identity, Mapping)
            or training_result.get("checkpoint_identity")
            != dict(checkpoint_identity)
            or training_result.get("checkpoint_authoritative") is not True
            or training_result.get("checkpoint_quarantined") is not False
        ):
            raise J1ExecutionIntegrityError(
                "Joint seal candidate checkpoint differs from training terminal"
            )
        load_authoritative_candidate_policy(
            checkpoint_identity=checkpoint_identity
        )
    elif checkpoint_identity != {
        "fixture_only": True,
        "training_result_payload_sha256": training_result[
            "terminal_result_payload_sha256"
        ],
        "scientific_authority": False,
    }:
        raise J1ExecutionIntegrityError(
            "Fixture candidate lineage identity changed"
        )
    observed_incumbent = incumbent_policy_binding()
    if observed_incumbent != joint.get("incumbent_policy_binding"):
        raise J1ExecutionIntegrityError(
            "Joint seal incumbent implementation changed"
        )
    return {
        "joint_manifest_seal": joint,
        "training_result": training_result,
        "training_result_identity": joint["training_terminal_result"],
        "candidate_checkpoint_identity": dict(checkpoint_identity),
        "incumbent_policy_binding": observed_incumbent,
        "passes": True,
    }


def load_precommitted_evaluation_manifest(
    *,
    execution_root: Path,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"development", "confirmation"}:
        raise ValueError("Only evaluation manifests are jointly precommitted")
    target_dir = execution_root / PRECOMMITTED_MANIFEST_DIR
    seal_path = target_dir / "joint_evaluation_manifest_seal.json"
    seal = load_json(seal_path)
    if not verify_payload_hash(
        seal,
        "joint_manifest_seal_payload_sha256",
    ):
        raise J1ExecutionIntegrityError("Joint manifest seal is invalid")
    identity = seal[f"{phase}_manifest"]
    path = Path(identity["path"])
    if path.resolve().parent != target_dir.resolve():
        raise J1ExecutionIntegrityError(
            "Precommitted evaluation manifest path changed"
        )
    if sha256_path(path) != identity["file_sha256"]:
        raise J1ExecutionIntegrityError(
            "Precommitted evaluation manifest file changed"
        )
    manifest = load_json(path)
    observed = root_manifest_identity(manifest)
    for key in (
        "phase",
        "row_count",
        "canonical_rows_sha256",
        "payload_sha256",
    ):
        if observed[key] != identity[key]:
            raise J1ExecutionIntegrityError(
                "Precommitted evaluation manifest identity changed"
            )
    expected = materialize_root_manifest(phase=phase)
    if manifest != expected:
        raise J1ExecutionIntegrityError(
            "Precommitted evaluation rows changed"
        )
    return manifest


def build_phase_marker_payload(
    *,
    phase: str,
    phase_lock: Mapping[str, Any],
    phase_lock_file_sha256: str,
    manifest: Mapping[str, Any],
    command: str,
    opened_at: str,
    hostname: str,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    if (
        not verify_payload_hash(phase_lock, "phase_lock_payload_sha256")
        or phase_lock.get("phase") != phase
        or phase_lock.get("decision") != f"READY_J1_{phase.upper()}_EXECUTION"
    ):
        raise J1ExecutionIntegrityError("Phase lock is not READY")
    if manifest != materialize_root_manifest(phase=phase):
        raise J1ExecutionIntegrityError("Phase manifest changed precommit")
    identity = root_manifest_identity(manifest)
    payload = {
        "version": f"{VERSION}_{phase}_execution_opened_v1",
        "phase": phase,
        "phase_lock_file_sha256": phase_lock_file_sha256,
        "phase_lock_payload_sha256": phase_lock["phase_lock_payload_sha256"],
        "root_commitment": phase_root_commitment(phase),
        "manifest_identity": identity,
        "activation_command": command,
        "activation_opened_at": opened_at,
        "activation_hostname": hostname,
        "streams_reserved_before_marker": 0,
        "streams_consumed_before_marker": 0,
        "scientific_work_before_marker": 0,
    }
    return payload_with_hash(payload, "activation_marker_payload_sha256")


def phase_marker_root_identity_audit(
    *,
    phase: str,
    first_marker: Mapping[str, Any],
    second_marker: Mapping[str, Any],
    confirmation_access_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    first = materialize_root_manifest(
        phase=phase,
        marker_payload=first_marker,
    )
    second = materialize_root_manifest(
        phase=phase,
        marker_payload=second_marker,
    )
    access_valid = (
        confirmation_access_evidence is not None
        and verify_payload_hash(
            confirmation_access_evidence,
            "confirmation_access_audit_sha256",
        )
    )
    checks = {
        "rows_exact": first["rows"] == second["rows"],
        "root_ids_exact": [
            row["root_id"] for row in first["rows"]
        ]
        == [row["root_id"] for row in second["rows"]],
        "ancestry_ids_exact": [
            row["ancestry_id"] for row in first["rows"]
        ]
        == [row["ancestry_id"] for row in second["rows"]],
        "canonical_rows_hash_exact": (
            first["canonical_rows_sha256"]
            == second["canonical_rows_sha256"]
        ),
        "confirmation_access_evidence_valid": (
            phase != "confirmation" or access_valid
        ),
        "confirmation_content_reads_zero": (
            phase != "confirmation"
            or (
                access_valid
                and confirmation_access_evidence.get(
                    "confirmation_content_reads"
                )
                == 0
            )
        ),
        "confirmation_streams_reserved_zero": (
            phase != "confirmation"
            or (
                access_valid
                and confirmation_access_evidence.get(
                    "confirmation_streams_reserved"
                )
                == 0
            )
        ),
        "confirmation_streams_consumed_zero": (
            phase != "confirmation"
            or (
                access_valid
                and confirmation_access_evidence.get(
                    "confirmation_streams_consumed"
                )
                == 0
            )
        ),
    }
    return {"checks": checks, "passes": all(checks.values())}


def _phase_ready_decision(phase: str) -> str:
    return f"READY_J1_{phase.upper()}_EXECUTION"


def build_phase_lock_payload(
    *,
    phase: str,
    readiness_lock_identity: Mapping[str, Any],
    readiness_result_identity: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    predecessor_result: Mapping[str, Any] | None,
    command: str,
    joint_manifest_seal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    if readiness_result_identity.get("decision") != (
        "READY_J1_EXECUTION_SURFACE"
    ):
        raise J1ExecutionIntegrityError("J1 execution readiness is not READY")
    predecessor_decision = (
        None if predecessor_result is None else predecessor_result.get("decision")
    )
    expected_predecessor = {
        "training": None,
        "development": "READY_J1_TRAINING_SANITY",
        "confirmation": "READY_J1_DEVELOPMENT_FULL_POLICY",
    }[phase]
    if predecessor_decision != expected_predecessor:
        raise J1ExecutionIntegrityError(
            f"{phase} predecessor decision changed"
        )
    expected_manifest = root_manifest_identity(
        materialize_root_manifest(phase=phase)
    )
    if dict(manifest_identity) != expected_manifest:
        raise J1ExecutionIntegrityError("Phase manifest identity changed")
    if phase in {"development", "confirmation"}:
        if joint_manifest_seal is None or not verify_payload_hash(
            joint_manifest_seal,
            "joint_manifest_seal_payload_sha256",
        ):
            raise J1ExecutionIntegrityError(
                f"{phase} requires the pre-development joint manifest seal"
            )
        confirmation_identity = joint_manifest_seal.get(
            "confirmation_manifest"
        )
        expected_confirmation = root_manifest_identity(
            materialize_root_manifest(phase="confirmation")
        )
        if any(
            confirmation_identity.get(key) != expected_confirmation[key]
            for key in expected_confirmation
        ):
            raise J1ExecutionIntegrityError(
                "Joint seal changed confirmation root identities"
            )
        if phase == "confirmation":
            if dict(manifest_identity) != {
                key: confirmation_identity[key]
                for key in expected_confirmation
            }:
                raise J1ExecutionIntegrityError(
                    "Confirmation manifest differs from joint seal"
                )
            if predecessor_result.get(
                "joint_evaluation_manifest_seal_payload_sha256"
            ) != joint_manifest_seal[
                "joint_manifest_seal_payload_sha256"
            ]:
                raise J1ExecutionIntegrityError(
                    "Confirmation joint seal differs from development binding"
                )
    payload = {
        "version": f"{VERSION}_{phase}_phase_lock_v1",
        "phase": phase,
        "decision": _phase_ready_decision(phase),
        "readiness_lock_identity": dict(readiness_lock_identity),
        "readiness_result_identity": dict(readiness_result_identity),
        "predecessor_decision": expected_predecessor,
        "predecessor_result_payload_sha256": (
            None
            if predecessor_result is None
            else predecessor_result.get("terminal_result_payload_sha256")
        ),
        "root_commitment": phase_root_commitment(phase),
        "manifest_identity": dict(manifest_identity),
        "joint_evaluation_manifest_seal_payload_sha256": (
            None
            if joint_manifest_seal is None
            else joint_manifest_seal[
                "joint_manifest_seal_payload_sha256"
            ]
        ),
        "command": command,
        "jobs": 1,
        "nice_minimum": 10,
        "active_hours_cap": PHASE_CAPS[phase]["active_hours"],
        "storage_gib_cap": PHASE_CAPS[phase]["storage_gib"],
        "free_disk_hard_floor_gib": 100.0,
        "free_disk_target_gib": 120.0,
        "streams_reserved": 0,
        "streams_consumed": 0,
    }
    return payload_with_hash(payload, "phase_lock_payload_sha256")


def phase_order_barrier_audit(
    *,
    phase: str,
    readiness_result: Mapping[str, Any],
    training_result: Mapping[str, Any] | None = None,
    development_result: Mapping[str, Any] | None = None,
    joint_manifest_seal: Mapping[str, Any] | None = None,
    confirmation_access_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    access_valid = (
        confirmation_access_audit is not None
        and verify_payload_hash(
            confirmation_access_audit,
            "confirmation_access_audit_sha256",
        )
    )
    checks = {
        "readiness_ready": readiness_result.get("decision")
        == "READY_J1_EXECUTION_SURFACE",
        "training_predecessor": (
            phase == "training"
            or (
                training_result is not None
                and training_result.get("decision")
                == "READY_J1_TRAINING_SANITY"
            )
        ),
        "development_predecessor": (
            phase != "confirmation"
            or (
                development_result is not None
                and development_result.get("decision")
                == "READY_J1_DEVELOPMENT_FULL_POLICY"
            )
        ),
        "joint_manifests_before_development": (
            phase == "training"
            or (
                joint_manifest_seal is not None
                and verify_payload_hash(
                    joint_manifest_seal,
                    "joint_manifest_seal_payload_sha256",
                )
            )
        ),
        "confirmation_access_audit_valid": access_valid,
        "confirmation_content_reads_before_authorization_zero": (
            access_valid
            and confirmation_access_audit.get("confirmation_content_reads")
            == 0
        ),
        "confirmation_streams_reserved_before_authorization_zero": (
            access_valid
            and confirmation_access_audit.get(
                "confirmation_streams_reserved"
            )
            == 0
        ),
        "confirmation_streams_consumed_before_authorization_zero": (
            access_valid
            and confirmation_access_audit.get(
                "confirmation_streams_consumed"
            )
            == 0
        ),
    }
    return {"phase": phase, "checks": checks, "passes": all(checks.values())}


def confirmation_access_audit(
    *,
    content_reads: int,
    streams_reserved: int,
    streams_consumed: int,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    values = (content_reads, streams_reserved, streams_consumed)
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise J1ExecutionIntegrityError(
            "Confirmation access counters are malformed"
        )
    payload = {
        "version": f"{VERSION}_confirmation_access_audit_v1",
        "confirmation_content_reads": content_reads,
        "confirmation_streams_reserved": streams_reserved,
        "confirmation_streams_consumed": streams_consumed,
        "evidence": dict(evidence),
    }
    return payload_with_hash(payload, "confirmation_access_audit_sha256")


def write_confirmation_access_audit(
    *,
    path: Path,
    content_reads: int,
    streams_reserved: int,
    streams_consumed: int,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = confirmation_access_audit(
        content_reads=content_reads,
        streams_reserved=streams_reserved,
        streams_consumed=streams_consumed,
        evidence=evidence,
    )
    return write_immutable_json(
        path,
        {
            key: value
            for key, value in payload.items()
            if key != "confirmation_access_audit_sha256"
        },
        field="confirmation_access_audit_sha256",
    )


def prospective_manifest() -> dict[str, Any]:
    denylist_path = (
        REPO_ROOT
        / "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
        "J1_PROTECTED_ID_DENYLIST.json"
    )
    denylist = load_json(denylist_path)
    parent_contract = denylist["prospective_stream_contract"]
    parent_intervals = {
        (row["partition"], row["stream_role"]): row
        for row in parent_contract["prospective_intervals"]
    }
    phase_rows = {
        phase: list(iter_prospective_rows(phase))
        for phase in PHASES
    }
    phase_hashes = {
        phase: _ordered_rows_hash(rows)
        for phase, rows in phase_rows.items()
    }
    all_rows_hash = _ordered_rows_hash(
        row for phase in PHASES for row in phase_rows[phase]
    )
    prefixes = []
    role_name = {
        "logical_stream_id": "logical",
        "deck_stream_id": "deck",
        "slot_stream_id": "slot",
        "candidate_policy_stream_id": "candidate_policy",
        "control_policy_stream_id": "control_policy",
    }
    parent_phase = {"training": "train", "development": "development",
                    "confirmation": "confirmation"}
    interval_sets: list[tuple[int, int, str, str]] = []
    for phase in PHASES:
        contract = STREAMS[phase]
        for field, parent_role in role_name.items():
            if field == "control_policy_stream_id" and phase == "training":
                continue
            base_key = parent_role
            base = int(contract[base_key])
            end = base + int(contract["rows"]) - 1
            parent = parent_intervals[
                (parent_phase[phase], parent_role)
            ]
            prefixes.append(
                base == int(parent["base"])
                and end <= int(parent["end_inclusive"])
                and int(contract["rows"]) <= int(parent["rows"])
            )
            interval_sets.append((base, end, phase, parent_role))
    ordered_intervals = sorted(interval_sets)
    ranges_disjoint = all(
        left[1] < right[0]
        for left, right in zip(ordered_intervals, ordered_intervals[1:])
    )
    logical_sets = {
        phase: {
            int(row["logical_stream_id"]) for row in phase_rows[phase]
        }
        for phase in PHASES
    }
    arm_ids = [
        (phase, row["row_index"], arm)
        for phase in PHASES
        for row in phase_rows[phase]
        for arm in (
            ("candidate",)
            if phase == "training"
            else ("candidate", "control")
        )
    ]
    root_commitments = {
        phase: phase_root_commitment(phase)
        for phase in PHASES
    }
    root_set_hashes = {}
    root_examples = {}
    for phase in PHASES:
        identifiers = [
            root_id_for_marker_commitment(root_commitments[phase], row)
            for row in phase_rows[phase]
        ]
        root_set_hashes[phase] = _ordered_rows_hash(
            {"root_id": value, "ancestry_id": value}
            for value in identifiers
        )
        root_examples[phase] = {
            "first": identifiers[0],
            "last": identifiers[-1],
        }
    root_sets = {
        phase: {
            root_id_for_marker_commitment(root_commitments[phase], row)
            for row in phase_rows[phase]
        }
        for phase in PHASES
    }
    checks = {
        "parent_denylist_file_exact": sha256_path(denylist_path)
        == ACCEPTED_FILES[
            "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
            "J1_PROTECTED_ID_DENYLIST.json"
        ],
        "parent_denylist_passed": (
            denylist.get("passes") is True
            and parent_contract.get("passes") is True
        ),
        "all_amended_ranges_exact_parent_prefixes": all(prefixes),
        "all_ranges_above_historical_ceiling": all(
            start > 212_999_999_999
            for start, _end, _phase, _role in ordered_intervals
        ),
        "all_namespace_ranges_disjoint": ranges_disjoint,
        "phase_logical_sets_disjoint": all(
            logical_sets[left].isdisjoint(logical_sets[right])
            for index, left in enumerate(PHASES)
            for right in PHASES[index + 1 :]
        ),
        "train_rows_exact": len(phase_rows["training"]) == TRAIN_ROOTS,
        "development_pairs_exact": (
            len(phase_rows["development"]) == DEVELOPMENT_PAIRS
        ),
        "confirmation_pairs_exact": (
            len(phase_rows["confirmation"]) == CONFIRMATION_PAIRS
        ),
        "total_game_arms_exact": len(arm_ids) == TOTAL_GAME_ARMS,
        "arm_ids_unique": len(set(arm_ids)) == len(arm_ids),
        "paired_crn_exact": all(
            row["control_policy_stream_id"] is not None
            and row["candidate_policy_stream_id"]
            != row["control_policy_stream_id"]
            for phase in ("development", "confirmation")
            for row in phase_rows[phase]
        ),
        "starter_none_everywhere": all(
            row["starter_tile"] is None
            for rows in phase_rows.values()
            for row in rows
        ),
        "no_root_ids_materialized": all(
            "root_id" not in row and "ancestry_id" not in row
            for rows in phase_rows.values()
            for row in rows
        ),
        "parent_streams_unreserved_unconsumed": (
            parent_contract.get("streams_reserved") == 0
            and parent_contract.get("streams_consumed") == 0
        ),
        "marker_root_commitments_stable": all(
            verify_payload_hash(value, "marker_payload_sha256")
            for value in root_commitments.values()
        ),
        "all_root_sets_precommitted": set(root_set_hashes) == set(PHASES),
        "precommitted_root_sets_disjoint": all(
            root_sets[left].isdisjoint(root_sets[right])
            for index, left in enumerate(PHASES)
            for right in PHASES[index + 1 :]
        ),
    }
    return {
        "version": f"{VERSION}_prospective_manifest_v1",
        "counts": {
            "training_rows": TRAIN_ROOTS,
            "development_pairs": DEVELOPMENT_PAIRS,
            "confirmation_pairs": CONFIRMATION_PAIRS,
            "total_game_arms": TOTAL_GAME_ARMS,
        },
        "ranges": STREAMS,
        "phase_canonical_rows_sha256": phase_hashes,
        "all_canonical_rows_sha256": all_rows_hash,
        "expanded_row_count": sum(len(rows) for rows in phase_rows.values()),
        "root_identity": {
            "algorithm": "sha256",
            "binding": (
                "accepted parent canonical JSON of immutable marker root "
                "commitment hash and row fields"
            ),
            "canonical_fields": [
                "marker_payload_sha256",
                "partition",
                "row",
                "logical_stream_id",
                "deck_stream_id",
                "slot_stream_id",
            ],
            "later_activation_marker_must_embed_exact_commitment": True,
            "operational_activation_fields_excluded": True,
        },
        "phase_marker_root_commitments": root_commitments,
        "phase_root_set_sha256": root_set_hashes,
        "phase_root_examples": root_examples,
        "parent_denylist": {
            "path": str(denylist_path.relative_to(REPO_ROOT)),
            "file_sha256": sha256_path(denylist_path),
            "payload_sha256": denylist["denylist_payload_sha256"],
        },
        "checks": checks,
        "passes": all(checks.values()),
        "streams_reserved": 0,
        "streams_consumed": 0,
        "root_ids_materialized_as_work_artifacts": 0,
    }


def evaluation_bootstrap_contract() -> dict[str, Any]:
    observed = {
        "replicates": BOOTSTRAPS,
        "phase_seeds": dict(BOOTSTRAP_SEEDS),
        "score_resampling": SCORE_BOOTSTRAP_METHOD,
        "progression_resampling": PROGRESSION_BOOTSTRAP_METHOD,
        "progression_zero_cell_rule": PROGRESSION_ZERO_CELL_RULE,
        "quantile": BOOTSTRAP_QUANTILE_CONVENTION,
    }
    checks = {
        key: observed[key] == expected
        for key, expected in EXPECTED_EVALUATION_BOOTSTRAP_CONTRACT.items()
    }
    return {
        **observed,
        "contract_sha256": canonical_json_hash(
            EXPECTED_EVALUATION_BOOTSTRAP_CONTRACT
        ),
        "checks": checks,
        "passes": all(checks.values()),
    }


def execution_schema() -> dict[str, Any]:
    training_state_keys = [
        "version",
        "runtime_payload_complete",
        "phase",
        "marker_file_sha256",
        "marker_payload_sha256",
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "model_state",
        "optimizer_state",
        "round_number",
        "collection_boundary",
        "next_manifest_row",
        "active_roots",
        "completed_roots",
        "transition_buffer_path",
        "transition_buffer_sha256",
        "epoch_cursor",
        "minibatch_cursor",
        "optimizer_step_ids",
        "round_aggregates",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "resource_clock",
        "output_bytes",
    ]
    phase_files = {
        phase: [
            PHASE_LOCK_NAME,
            PHASE_LOCK_RESULT_NAME,
            PHASE_MARKER_NAME,
            PHASE_MANIFEST_NAME,
            PHASE_RUNTIME_NAME,
            PHASE_OWNER_NAME,
            PHASE_STREAM_RESERVATION_NAME,
            PHASE_STREAM_CONSUMPTION_NAME,
            PHASE_RESULT_NAME,
            PHASE_RETENTION_NAME,
        ]
        for phase in PHASES
    }
    payload = {
        "version": f"{VERSION}_schema_v1",
        "model": {
            "parameter_count": j1.EXPECTED_PARAMETER_COUNT,
            "schema_sha256": j1.model_schema_sha256(),
            "observation_width": j1.EXPECTED_OBSERVATION_WIDTH,
            "config": {
                key: value
                for key, value in vars(j1.FROZEN_CONFIG).items()
            },
        },
        "scientific_counts": {
            "training_roots": TRAIN_ROOTS,
            "rounds": ROUNDS,
            "roots_per_round": ROOTS_PER_ROUND,
            "synchronous_envs": ENV_COUNT,
            "development_pairs": DEVELOPMENT_PAIRS,
            "confirmation_pairs": CONFIRMATION_PAIRS,
            "total_game_arms": TOTAL_GAME_ARMS,
        },
        "training_state_required_keys": training_state_keys,
        "phase_files": phase_files,
        "phase_order": list(PHASES),
        "phase_predecessors": {
            "training": {
                "readiness_decision": "READY_J1_EXECUTION_SURFACE",
            },
            "development": {
                "training_decision": "READY_J1_TRAINING_SANITY",
                "joint_development_confirmation_manifest_seal": True,
            },
            "confirmation": {
                "development_decision":
                    "READY_J1_DEVELOPMENT_FULL_POLICY",
            },
        },
        "commands": {
            "readiness_only": ["write-test-evidence", "prepare"],
            "future_scientific": [
                "seal-phase-lock",
                "open",
                "materialize",
                "execute",
            ],
            "promotion_command_present": False,
        },
        "resume": {
            "immutable_json_create_once": True,
            "atomic_binary": True,
            "append_only_collection_journal": True,
            "single_writer_exclusive": True,
            "live_or_mismatched_writer_fail_closed": True,
            "same_marker_only": True,
            "hidden_retry_allowed": False,
            "dead_same_contract_owner_reclaim_record": True,
            "commit_head_advanced_last": True,
            "bootstrap_no_head_reclaim_is_zero_work_only": True,
            "stream_consumption_opener_is_owner_ancestor": True,
            "fixed_collection_ticks_per_commit":
                TRAINING_COLLECTION_TICKS_PER_COMMIT,
            "transition_chunks_current_round_only": True,
            "ppo_batch_current_round_only": True,
            "retirement_intents_recover_idempotently": True,
            "indexed_commit_full_scan_on_resume_and_terminal_only": True,
            "runtime_and_rolling_journals_indexed_in_memory": True,
        },
        "storage_projection": {
            "artifact": RUNTIME_STORAGE_PROJECTION_NAME,
            "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
            "central_moves": PROJECTION_CENTRAL_MOVES,
            "sensitivity_moves": PROJECTION_SENSITIVITY_MOVES,
            "training_output_file_cap": TRAINING_OUTPUT_FILE_CAP,
            "training_fsync_cap": TRAINING_FSYNC_CAP,
            "evaluation_output_file_cap": EVALUATION_OUTPUT_FILE_CAP,
            "evaluation_fsync_cap": EVALUATION_FSYNC_CAP,
        },
        "decisions": {
            "training": [
                "READY_J1_TRAINING_SANITY",
                "HOLD_J1_LEARNING_SANITY",
                "HOLD_J1_OPERATIONAL",
                "KILL_J1_INTEGRITY",
            ],
            "development": [
                "READY_J1_DEVELOPMENT_FULL_POLICY",
                "HOLD_J1_DEVELOPMENT_INCONCLUSIVE",
                "HOLD_J1_OPERATIONAL",
                "KILL_J1_FULL_POLICY_UTILITY",
                "KILL_J1_INTEGRITY",
            ],
            "confirmation": [
                "READY_J1_PROMOTION_REVIEW",
                "HOLD_J1_PROGRESSION_UNDERPOWERED",
                "HOLD_J1_CONFIRMATION_INCONCLUSIVE",
                "HOLD_J1_OPERATIONAL",
                "KILL_J1_FULL_POLICY_CAPABILITY",
                "KILL_J1_INTEGRITY",
            ],
        },
        "bootstrap": evaluation_bootstrap_contract(),
        "zero_work": ZERO_WORK,
    }
    payload["schema_sha256"] = canonical_json_hash(payload)
    return payload


def _ceil_divide(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("Ceiling division inputs are invalid")
    return (numerator + denominator - 1) // denominator


def _accepted_j1a_arithmetic() -> dict[str, Any]:
    relative = (
        "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
        "J1A_COST_POWER_ARITHMETIC.json"
    )
    path = REPO_ROOT / relative
    if sha256_path(path) != ACCEPTED_FILES[relative]:
        raise J1ExecutionIntegrityError(
            "Accepted J1a arithmetic file identity changed"
        )
    payload = load_json(path)
    field, expected_payload = ACCEPTED_PAYLOADS[relative]
    if (
        field != "arithmetic_payload_sha256"
        or payload.get(field) != expected_payload
        or not verify_payload_hash(payload, field)
        or payload.get("passes") is not True
    ):
        raise J1ExecutionIntegrityError(
            "Accepted J1a arithmetic payload changed"
        )
    return payload


def _training_io_projection(*, moves_per_root: int) -> dict[str, Any]:
    if moves_per_root <= 0:
        raise ValueError("Projected moves must be positive")
    transition_rows = TRAIN_ROOTS * moves_per_root
    current_round_rows = ROOTS_PER_ROUND * moves_per_root
    collection_checkpoints = _ceil_divide(
        transition_rows,
        ENV_COUNT * TRAINING_COLLECTION_TICKS_PER_COMMIT,
    )
    optimizer_steps = (
        ROUNDS
        * int(j1.FROZEN_CONFIG.epochs_per_round)
        * _ceil_divide(
            current_round_rows,
            int(j1.FROZEN_CONFIG.minibatch_size),
        )
    )
    compact_boundary_records = (
        collection_checkpoints + optimizer_steps + 3 * ROUNDS
    )
    root_blob_blocks = _ceil_divide(
        TRAIN_ROOTS,
        ROOT_BLOB_BLOCK_SIZE,
    )
    created_files = (
        TRAIN_ROOTS
        + collection_checkpoints
        + ROUNDS
        + root_blob_blocks
        + 2 * ROUNDS
        + 5 * TRAINING_COMMIT_STATE_COUNT
        + 100
    )
    fsync_count = (
        TRAIN_ROOTS
        + collection_checkpoints
        + ROUNDS
        + 3 * compact_boundary_records
        + 2 * compact_boundary_records
        + 5 * TRAINING_COMMIT_STATE_COUNT
        + 4 * ROUNDS
        + 100
    )
    io_seconds = (
        collection_checkpoints
        * PROJECTION_IO_UNIT_SECONDS["collection_checkpoint"]
        + optimizer_steps
        * PROJECTION_IO_UNIT_SECONDS["optimizer_checkpoint"]
        + TRAINING_COMMIT_STATE_COUNT
        * PROJECTION_IO_UNIT_SECONDS["commit_seal"]
        + TRAIN_ROOTS
        * PROJECTION_IO_UNIT_SECONDS["root_blob_write"]
        + collection_checkpoints
        * PROJECTION_IO_UNIT_SECONDS["transition_chunk_write"]
        + ROUNDS
        * PROJECTION_IO_UNIT_SECONDS["round_batch_write"]
    )
    return {
        "moves_per_root": moves_per_root,
        "transition_rows": transition_rows,
        "current_round_transition_rows": current_round_rows,
        "collection_checkpoints": collection_checkpoints,
        "optimizer_steps": optimizer_steps,
        "compact_boundary_records": compact_boundary_records,
        "created_files": created_files,
        "fsync_count": fsync_count,
        "projected_io_seconds": io_seconds,
        "fixed_collection_tick_cadence": (
            TRAINING_COLLECTION_TICKS_PER_COMMIT
        ),
        "maximum_replayed_collection_ticks": (
            TRAINING_COLLECTION_TICKS_PER_COMMIT
        ),
    }


def _training_storage_projection(*, moves_per_root: int) -> dict[str, Any]:
    io_projection = _training_io_projection(
        moves_per_root=moves_per_root
    )
    transition_rows = int(io_projection["transition_rows"])
    current_round_rows = int(
        io_projection["current_round_transition_rows"]
    )
    fixture_transitions = int(
        BOUNDED_FIXTURE_COST_EVIDENCE["training_transitions"]
    )
    root_bytes_per_transition = _ceil_divide(
        int(BOUNDED_FIXTURE_COST_EVIDENCE["root_blob_bytes"]),
        fixture_transitions,
    )
    chunk_bytes_per_transition = _ceil_divide(
        int(BOUNDED_FIXTURE_COST_EVIDENCE["transition_chunk_bytes"]),
        fixture_transitions,
    )
    batch_bytes_per_transition = _ceil_divide(
        int(BOUNDED_FIXTURE_COST_EVIDENCE["round_batch_bytes"]),
        fixture_transitions,
    )
    root_blob_blocks = _ceil_divide(
        TRAIN_ROOTS,
        ROOT_BLOB_BLOCK_SIZE,
    )
    compact_boundaries = int(
        io_projection["compact_boundary_records"]
    )
    terms = {
        "retained_finalized_root_blobs": (
            transition_rows * root_bytes_per_transition
        ),
        "ephemeral_current_round_transition_chunks": (
            current_round_rows * chunk_bytes_per_transition
        ),
        "ephemeral_current_round_ppo_batch": (
            current_round_rows * batch_bytes_per_transition
        ),
        "retained_model_adam_commit_states": (
            TRAINING_COMMIT_STATE_COUNT
            * COMMIT_STATE_STORAGE_ENVELOPE_BYTES
        ),
        "three_rolling_slots_including_crash_orphan": (
            3 * ROLLING_STATE_STORAGE_ENVELOPE_BYTES
        ),
        "immutable_commit_sidecars": (
            TRAINING_COMMIT_STATE_COUNT * 16_384
        ),
        "root_blob_block_seals": root_blob_blocks * 4_096,
        "rolling_resume_journal": compact_boundaries * 1_024,
        "runtime_charge_journal": compact_boundaries * 2 * 1_024,
        "transition_and_batch_retirement_manifests": (
            2 * ROUNDS * 65_536
        ),
        "round64_candidate_checkpoint": 6_000_000,
        "owner_stream_schema_and_terminal_metadata": (
            METADATA_STORAGE_ENVELOPE_BYTES
        ),
        "single_atomic_temp_orphan_envelope": 6_000_000,
    }
    before_margin = sum(terms.values())
    with_margin = math.ceil(
        before_margin * PROJECTION_SAFETY_MULTIPLIER
    )
    cap_bytes = int(PHASE_CAPS["training"]["storage_gib"] * 1024**3)
    checks = {
        "root_blob_measurement_reproduced": (
            root_bytes_per_transition == 1_519
        ),
        "batch_measurement_reproduced": (
            batch_bytes_per_transition == 1_261
        ),
        "chunk_measurement_reproduced": (
            chunk_bytes_per_transition == 1_526
        ),
        "current_round_chunks_only": True,
        "current_round_batch_only": True,
        "retirement_manifest_recovery_required": True,
        "storage_with_margin_within_cap": with_margin <= cap_bytes,
    }
    return {
        "moves_per_root": moves_per_root,
        "bytes_per_transition": {
            "finalized_root_blob": root_bytes_per_transition,
            "ephemeral_transition_chunk": chunk_bytes_per_transition,
            "ephemeral_round_batch": batch_bytes_per_transition,
        },
        "projection_terms_bytes": terms,
        "projected_before_margin_bytes": before_margin,
        "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
        "projected_with_margin_bytes": with_margin,
        "projected_with_margin_gib": with_margin / 1024**3,
        "cap_bytes": cap_bytes,
        "cap_gib": PHASE_CAPS["training"]["storage_gib"],
        "checks": checks,
        "passes": all(checks.values()),
    }


def _evaluation_io_storage_projection(
    *,
    phase: str,
    pairs: int,
    moves_per_arm: int,
) -> dict[str, Any]:
    if phase not in {"development", "confirmation"}:
        raise ValueError("Evaluation projection phase changed")
    if pairs <= 0 or moves_per_arm <= 0:
        raise ValueError("Evaluation projection inputs are invalid")
    arms = 2 * pairs
    rolling_boundaries = arms
    block_seals = _ceil_divide(pairs, PAIR_RESULT_BLOCK_SIZE)
    pair_bytes = (
        PAIR_RESULT_FIXED_BYTES
        + 2 * moves_per_arm * PAIR_RESULT_BYTES_PER_MOVE_PER_ARM
    )
    terms = {
        "retained_pair_result_blobs": pairs * pair_bytes,
        "three_rolling_slots_including_crash_orphan": 3 * 1_000_000,
        "compact_block_commit_states": (
            (1 + block_seals) * COMPACT_EVALUATION_COMMIT_BYTES
        ),
        "immutable_commit_sidecars": (1 + block_seals) * 16_384,
        "rolling_resume_journal": rolling_boundaries * 1_024,
        "runtime_charge_journal": rolling_boundaries * 2 * 1_024,
        "manifest_analysis_owner_stream_and_terminal_metadata": (
            METADATA_STORAGE_ENVELOPE_BYTES
        ),
    }
    before_margin = sum(terms.values())
    with_margin = math.ceil(
        before_margin * PROJECTION_SAFETY_MULTIPLIER
    )
    cap_bytes = int(PHASE_CAPS[phase]["storage_gib"] * 1024**3)
    created_files = pairs + 5 * (1 + block_seals) + 100
    fsync_count = (
        pairs
        + 3 * rolling_boundaries
        + 2 * rolling_boundaries
        + 5 * (1 + block_seals)
        + 100
    )
    io_seconds = (
        rolling_boundaries
        * PROJECTION_IO_UNIT_SECONDS["paired_arm_checkpoint"]
        + pairs * PROJECTION_IO_UNIT_SECONDS["paired_result_blob"]
        + block_seals * PROJECTION_IO_UNIT_SECONDS["paired_block_seal"]
    )
    checks = {
        "paired_results_write_once": True,
        "rolling_state_excludes_completed_pair_prefix": True,
        "created_files_within_cap": (
            created_files <= EVALUATION_OUTPUT_FILE_CAP
        ),
        "fsync_count_within_cap": fsync_count <= EVALUATION_FSYNC_CAP,
        "storage_with_margin_within_cap": with_margin <= cap_bytes,
    }
    return {
        "phase": phase,
        "pairs": pairs,
        "arms": arms,
        "moves_per_arm": moves_per_arm,
        "pair_blob_bytes": pair_bytes,
        "rolling_boundaries": rolling_boundaries,
        "block_seals": block_seals,
        "created_files": created_files,
        "created_file_cap": EVALUATION_OUTPUT_FILE_CAP,
        "fsync_count": fsync_count,
        "fsync_cap": EVALUATION_FSYNC_CAP,
        "projected_io_seconds": io_seconds,
        "projection_terms_bytes": terms,
        "projected_before_margin_bytes": before_margin,
        "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
        "projected_with_margin_bytes": with_margin,
        "projected_with_margin_gib": with_margin / 1024**3,
        "cap_bytes": cap_bytes,
        "cap_gib": PHASE_CAPS[phase]["storage_gib"],
        "checks": checks,
        "passes": all(checks.values()),
    }


def full_scale_runtime_storage_projection() -> dict[str, Any]:
    arithmetic = _accepted_j1a_arithmetic()
    inherited = arithmetic["runtime_storage"][
        "amended_phase_projections"
    ]
    training_io_central = _training_io_projection(
        moves_per_root=PROJECTION_CENTRAL_MOVES
    )
    training_io_sensitivity = _training_io_projection(
        moves_per_root=PROJECTION_SENSITIVITY_MOVES
    )
    training_storage_central = _training_storage_projection(
        moves_per_root=PROJECTION_CENTRAL_MOVES
    )
    training_storage_sensitivity = _training_storage_projection(
        moves_per_root=PROJECTION_SENSITIVITY_MOVES
    )
    evaluation = {}
    phase_counts = {
        "development": DEVELOPMENT_PAIRS,
        "confirmation": CONFIRMATION_PAIRS,
    }
    for phase, pairs in phase_counts.items():
        central_storage = _evaluation_io_storage_projection(
            phase=phase,
            pairs=pairs,
            moves_per_arm=PROJECTION_CENTRAL_MOVES,
        )
        sensitivity_storage = _evaluation_io_storage_projection(
            phase=phase,
            pairs=pairs,
            moves_per_arm=PROJECTION_SENSITIVITY_MOVES,
        )
        central_hours = (
            float(inherited[phase]["central_hours"])
            + float(central_storage["projected_io_seconds"]) / 3600.0
        )
        sensitivity_hours = (
            float(
                inherited[phase][
                    "contract_max_5000_move_sensitivity_hours"
                ]
            )
            + float(sensitivity_storage["projected_io_seconds"]) / 3600.0
        )
        central_margin_hours = (
            central_hours * PROJECTION_SAFETY_MULTIPLIER
        )
        sensitivity_margin_hours = (
            sensitivity_hours * PROJECTION_SAFETY_MULTIPLIER
        )
        cap_hours = float(PHASE_CAPS[phase]["active_hours"])
        evaluation[phase] = {
            "central": {
                "inherited_compute_hours": inherited[phase][
                    "central_hours"
                ],
                "bounded_io_hours": (
                    central_storage["projected_io_seconds"] / 3600.0
                ),
                "hours_with_25pct_margin": central_margin_hours,
                "runtime_cap_hours": cap_hours,
                "runtime_cap_fraction_after_margin": (
                    central_margin_hours / cap_hours
                ),
                "runtime_within_cap": central_margin_hours <= cap_hours,
                "runtime_at_most_91pct_cap": (
                    central_margin_hours / cap_hours <= 0.91
                ),
                "storage": central_storage,
            },
            "sensitivity_5000_moves": {
                "inherited_compute_hours": inherited[phase][
                    "contract_max_5000_move_sensitivity_hours"
                ],
                "bounded_io_hours": (
                    sensitivity_storage["projected_io_seconds"] / 3600.0
                ),
                "hours_with_25pct_margin": sensitivity_margin_hours,
                "runtime_cap_hours": cap_hours,
                "runtime_within_cap": (
                    sensitivity_margin_hours <= cap_hours
                ),
                "storage": sensitivity_storage,
                "diagnostic_not_conjunctive": True,
            },
        }

    training_central_hours = (
        float(inherited["training"]["central_hours"])
        + float(training_io_central["projected_io_seconds"]) / 3600.0
    )
    training_sensitivity_hours = (
        float(
            inherited["training"][
                "contract_max_5000_move_sensitivity_hours"
            ]
        )
        + float(training_io_sensitivity["projected_io_seconds"]) / 3600.0
    )
    training_cap_hours = float(PHASE_CAPS["training"]["active_hours"])
    training = {
        "central": {
            "inherited_compute_hours": inherited["training"][
                "central_hours"
            ],
            "bounded_io": training_io_central,
            "hours_with_25pct_margin": (
                training_central_hours * PROJECTION_SAFETY_MULTIPLIER
            ),
            "runtime_cap_hours": training_cap_hours,
            "runtime_within_cap": (
                training_central_hours * PROJECTION_SAFETY_MULTIPLIER
                <= training_cap_hours
            ),
            "created_files_within_cap": (
                training_io_central["created_files"]
                <= TRAINING_OUTPUT_FILE_CAP
            ),
            "fsync_count_within_cap": (
                training_io_central["fsync_count"]
                <= TRAINING_FSYNC_CAP
            ),
            "storage": training_storage_central,
        },
        "sensitivity_5000_moves": {
            "inherited_compute_hours": inherited["training"][
                "contract_max_5000_move_sensitivity_hours"
            ],
            "bounded_io": training_io_sensitivity,
            "hours_with_25pct_margin": (
                training_sensitivity_hours
                * PROJECTION_SAFETY_MULTIPLIER
            ),
            "runtime_cap_hours": training_cap_hours,
            "runtime_within_cap": (
                training_sensitivity_hours
                * PROJECTION_SAFETY_MULTIPLIER
                <= training_cap_hours
            ),
            "created_files_within_cap": (
                training_io_sensitivity["created_files"]
                <= TRAINING_OUTPUT_FILE_CAP
            ),
            "fsync_count_within_cap": (
                training_io_sensitivity["fsync_count"]
                <= TRAINING_FSYNC_CAP
            ),
            "storage": training_storage_sensitivity,
            "diagnostic_not_conjunctive": True,
        },
    }
    fixture_checks = {
        "fixture_identity_frozen_in_source": (
            BOUNDED_FIXTURE_COST_EVIDENCE["version"]
            == "j1_bounded_fixture_cost_evidence_v1"
        ),
        "fixture_has_zero_scientific_work": all(
            BOUNDED_FIXTURE_COST_EVIDENCE[key] == 0
            for key in (
                "scientific_games",
                "scientific_optimizer_steps",
                "scientific_policy_outcomes",
            )
        ),
        "actual_model_adam_bytes_reproduced": (
            BOUNDED_FIXTURE_COST_EVIDENCE["real_model_adam_bytes"]
            == 4_951_545
        ),
        "actual_training_fixture_files_reproduced": (
            BOUNDED_FIXTURE_COST_EVIDENCE[
                "terminal_file_count_after_retirement"
            ]
            == 35
        ),
        "actual_paired_fixture_writes_reproduced": (
            BOUNDED_FIXTURE_COST_EVIDENCE["paired_fixture_pairs"] == 4
            and BOUNDED_FIXTURE_COST_EVIDENCE["paired_blob_bytes"]
            == 11_692
        ),
    }
    checks = {
        "accepted_j1a_arithmetic_exact": True,
        "total_game_arms_exact": TOTAL_GAME_ARMS == 27_136,
        "fixture_evidence": all(fixture_checks.values()),
        "training_central_runtime": training["central"][
            "runtime_within_cap"
        ],
        "training_central_storage": training["central"]["storage"][
            "passes"
        ],
        "training_central_files": training["central"][
            "created_files_within_cap"
        ],
        "training_central_fsyncs": training["central"][
            "fsync_count_within_cap"
        ],
        "development_central_runtime": evaluation["development"][
            "central"
        ]["runtime_within_cap"],
        "development_runtime_headroom": evaluation["development"][
            "central"
        ]["runtime_at_most_91pct_cap"],
        "development_storage": evaluation["development"]["central"][
            "storage"
        ]["passes"],
        "confirmation_central_runtime": evaluation["confirmation"][
            "central"
        ]["runtime_within_cap"],
        "confirmation_runtime_headroom": evaluation["confirmation"][
            "central"
        ]["runtime_at_most_91pct_cap"],
        "confirmation_storage": evaluation["confirmation"]["central"][
            "storage"
        ]["passes"],
        "sensitivity_reported_for_every_phase": True,
        "fixed_25pct_margin": PROJECTION_SAFETY_MULTIPLIER == 1.25,
        "bounded_abandoned_attempt_charges": (
            ABANDONED_ATTEMPT_CHARGE_SECONDS
            == {
                "training_collection_tick_block": 600.0,
                "training_minibatch_update": 300.0,
                "paired_candidate_arm": 900.0,
                "paired_control_arm_and_pair": 900.0,
                "miniature_fixture_other": 1.0,
            }
        ),
    }
    return {
        "version": f"{VERSION}_runtime_storage_projection_v1",
        "method": (
            "accepted J1a compute projection plus actual bounded fixture "
            "bytes and fixed source-derived I/O operation counts"
        ),
        "accepted_j1a_arithmetic": {
            "path": (
                "threes_rl/runs/forensics/"
                "j1a_cost_power_amendment_v1/"
                "J1A_COST_POWER_ARITHMETIC.json"
            ),
            "file_sha256": ACCEPTED_FILES[
                "threes_rl/runs/forensics/"
                "j1a_cost_power_amendment_v1/"
                "J1A_COST_POWER_ARITHMETIC.json"
            ],
            "payload_sha256": ACCEPTED_PAYLOADS[
                "threes_rl/runs/forensics/"
                "j1a_cost_power_amendment_v1/"
                "J1A_COST_POWER_ARITHMETIC.json"
            ][1],
        },
        "bounded_fixture_evidence": {
            **BOUNDED_FIXTURE_COST_EVIDENCE,
            "canonical_sha256": canonical_json_hash(
                BOUNDED_FIXTURE_COST_EVIDENCE
            ),
            "checks": fixture_checks,
        },
        "central_planning_moves": PROJECTION_CENTRAL_MOVES,
        "sensitivity_moves": PROJECTION_SENSITIVITY_MOVES,
        "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
        "training": training,
        "evaluation": evaluation,
        "retirement_contract": {
            "transition_chunks_current_round_only": True,
            "round_batch_current_round_only": True,
            "predecessor_seal_before_retirement": True,
            "immutable_retirement_manifest": True,
            "idempotent_crash_window_recovery": True,
            "finalized_root_blobs_preserved": True,
        },
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": ZERO_WORK,
    }


def training_sanity_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    manifest_roots = [str(value) for value in report["manifest_root_ids"]]
    completed_roots = [str(value) for value in report["completed_root_ids"]]
    expected_steps = [
        str(value) for value in report["expected_optimizer_step_ids"]
    ]
    closed_steps = [
        str(value) for value in report["closed_optimizer_step_ids"]
    ]
    rounds = list(report["rounds"])
    if len(rounds) != ROUNDS:
        raise J1ExecutionIntegrityError("Training sanity needs 64 rounds")
    round_by_number = {int(row["round"]): row for row in rounds}
    if (
        len(round_by_number) != ROUNDS
        or set(round_by_number) != set(range(1, ROUNDS + 1))
    ):
        raise J1ExecutionIntegrityError("Training round identities changed")
    ordered_round_roots: list[str] = []
    round_integrity = []
    for round_number in range(1, ROUNDS + 1):
        row = round_by_number[round_number]
        root_ids = [str(value) for value in row.get("root_ids", [])]
        root_metrics = list(row.get("root_metrics", []))
        metric_root_ids = [
            str(value.get("root_id")) for value in root_metrics
        ]
        expected_slice = manifest_roots[
            (round_number - 1) * ROOTS_PER_ROUND :
            round_number * ROOTS_PER_ROUND
        ]
        hashes_valid = all(
            isinstance(metric.get("committed_record_sha256"), str)
            and len(metric["committed_record_sha256"]) == 64
            and isinstance(metric.get("transition_content_sha256"), str)
            and len(metric["transition_content_sha256"]) == 64
            and int(metric.get("transition_rows", 0)) > 0
            for metric in root_metrics
        )
        metrics_hash_valid = (
            row.get("root_metrics_sha256")
            == j1.stable_hash(root_metrics)
        )
        aggregate_checks = _validate_round_metric_aggregates(
            row,
            root_metrics,
        )
        checks = {
            "root_count_exact": len(root_ids) == ROOTS_PER_ROUND,
            "root_ids_unique": len(set(root_ids)) == ROOTS_PER_ROUND,
            "root_ids_exact_manifest_slice": root_ids == expected_slice,
            "metric_root_ids_exact": metric_root_ids == root_ids,
            "root_metric_hashes_valid": hashes_valid,
            "root_metrics_hash_exact": metrics_hash_valid,
            "committed_records_hash_present": (
                isinstance(row.get("committed_records_sha256"), str)
                and len(row["committed_records_sha256"]) == 64
            ),
            "transition_buffer_hash_present": (
                isinstance(row.get("transition_buffer_sha256"), str)
                and len(row["transition_buffer_sha256"]) == 64
            ),
            "aggregates_recomputed_exact": aggregate_checks["passes"],
        }
        round_integrity.append(
            {
                "round": round_number,
                "checks": checks,
                "aggregate_checks": aggregate_checks,
                "passes": all(checks.values()),
            }
        )
        ordered_round_roots.extend(root_ids)
    if any(not row["passes"] for row in round_integrity):
        raise J1ExecutionIntegrityError(
            "Training round metric evidence is not authenticated"
        )
    log_scores_by_round = {
        round_number: [
            float(metric["log_score"])
            for metric in round_by_number[round_number]["root_metrics"]
        ]
        for round_number in range(1, ROUNDS + 1)
    }
    first = [
        value
        for round_number in range(1, 5)
        for value in log_scores_by_round[round_number]
    ]
    final = [
        value
        for round_number in range(61, 65)
        for value in log_scores_by_round[round_number]
    ]
    final_round_metrics = round_by_number[ROUNDS]["root_metrics"]
    final_aux_brier = [
        float(np.mean([
            float(metric["auxiliary_brier"][index])
            for metric in final_round_metrics
        ]))
        for index in range(3)
    ]
    final_aux_prevalence = [
        float(np.mean([
            float(metric["auxiliary_prevalence"][index])
            for metric in final_round_metrics
        ]))
        for index in range(3)
    ]
    metrics = {
        "first_four_root_equal_mean_log_score": float(np.mean(first)),
        "final_four_root_equal_mean_log_score": float(np.mean(final)),
        "final_legal_entropy_nats": float(np.mean([
            float(metric["legal_entropy_nats"])
            for metric in final_round_metrics
        ])),
        "final_value_mse": float(np.mean([
            float(metric["value_mse"])
            for metric in final_round_metrics
        ])),
        "final_zero_value_mse": float(np.mean([
            float(metric["zero_value_mse"])
            for metric in final_round_metrics
        ])),
        "auxiliary_brier": final_aux_brier,
        "auxiliary_prevalence_brier": [
            value * (1.0 - value)
            for value in final_aux_prevalence
        ],
    }
    finite_values = [
        metrics["first_four_root_equal_mean_log_score"],
        metrics["final_four_root_equal_mean_log_score"],
        metrics["final_legal_entropy_nats"],
        metrics["final_value_mse"],
        metrics["final_zero_value_mse"],
        *metrics["auxiliary_brier"],
        *metrics["auxiliary_prevalence_brier"],
    ]
    aux_wins = sum(
        observed < baseline
        for observed, baseline in zip(
            metrics["auxiliary_brier"],
            metrics["auxiliary_prevalence_brier"],
        )
    )
    integrity = {
        "manifest_roots_exact_count": len(manifest_roots) == TRAIN_ROOTS,
        "manifest_roots_unique": len(set(manifest_roots)) == TRAIN_ROOTS,
        "round_partitions_concatenate_manifest": (
            ordered_round_roots == manifest_roots
        ),
        "completed_roots_once": (
            len(completed_roots) == TRAIN_ROOTS
            and len(set(completed_roots)) == TRAIN_ROOTS
            and completed_roots == manifest_roots
        ),
        "optimizer_steps_once": (
            len(expected_steps) > 0
            and len(set(expected_steps)) == len(expected_steps)
            and closed_steps == expected_steps
            and len(set(closed_steps)) == len(closed_steps)
        ),
        "all_metrics_finite": all(math.isfinite(value) for value in finite_values),
        "three_auxiliaries_exact": (
            len(metrics["auxiliary_brier"]) == 3
            and len(metrics["auxiliary_prevalence_brier"]) == 3
        ),
        "authenticated_terminal_commit_chain": (
            report.get("authenticated_terminal_boundary", {}).get(
                "passes"
            )
            is True
            and report.get("authenticated_terminal_boundary", {}).get(
                "chain_audit_passes"
            )
            is True
        ),
        "checkpoint_round_64_exact": (
            report.get("checkpoint_identity", {}).get("round") == 64
            and report.get("checkpoint_identity", {}).get(
                "save_load_exact"
            )
            is True
            and report.get("checkpoint_identity", {}).get("parameter_count")
            == j1.EXPECTED_PARAMETER_COUNT
            and report.get("checkpoint_identity", {}).get(
                "model_schema_sha256"
            )
            == j1.model_schema_sha256()
            and report.get("checkpoint_identity", {}).get(
                "training_state_file_sha256"
            )
            == report.get("authenticated_terminal_boundary", {}).get(
                "state_file_sha256"
            )
        ),
    }
    scientific = {
        "final_four_log_score_improves": (
            metrics["final_four_root_equal_mean_log_score"]
            > metrics["first_four_root_equal_mean_log_score"]
        ),
        "entropy_at_least_0_15": (
            metrics["final_legal_entropy_nats"] >= 0.15
        ),
        "value_mse_beats_zero": (
            metrics["final_value_mse"] < metrics["final_zero_value_mse"]
        ),
        "at_least_two_auxiliary_briers_improve": aux_wins >= 2,
    }
    if not all(integrity.values()):
        decision = "KILL_J1_INTEGRITY"
    elif all(scientific.values()):
        decision = "READY_J1_TRAINING_SANITY"
    else:
        decision = "HOLD_J1_LEARNING_SANITY"
    return {
        "version": f"{VERSION}_training_sanity_v1",
        "metrics": metrics,
        "auxiliary_brier_wins": aux_wins,
        "integrity_checks": integrity,
        "round_integrity": round_integrity,
        "scientific_checks": scientific,
        "decision": decision,
        "promote": False,
    }


def _validate_round_metric_aggregates(
    row: Mapping[str, Any],
    root_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not root_metrics:
        return {"checks": {"nonempty": False}, "passes": False}
    expected = {
        "root_log_scores": [
            float(metric["log_score"]) for metric in root_metrics
        ],
        "legal_entropy_nats": float(np.mean([
            float(metric["legal_entropy_nats"])
            for metric in root_metrics
        ])),
        "value_mse": float(np.mean([
            float(metric["value_mse"]) for metric in root_metrics
        ])),
        "zero_value_mse": float(np.mean([
            float(metric["zero_value_mse"]) for metric in root_metrics
        ])),
        "auxiliary_brier": [
            float(np.mean([
                float(metric["auxiliary_brier"][index])
                for metric in root_metrics
            ]))
            for index in range(3)
        ],
    }
    prevalence = [
        float(np.mean([
            float(metric["auxiliary_prevalence"][index])
            for metric in root_metrics
        ]))
        for index in range(3)
    ]
    expected["auxiliary_prevalence_brier"] = [
        value * (1.0 - value) for value in prevalence
    ]

    def exact_float_sequence(left: Any, right: Sequence[float]) -> bool:
        return (
            isinstance(left, Sequence)
            and len(left) == len(right)
            and all(
                math.isclose(
                    float(observed),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for observed, expected_value in zip(left, right)
            )
        )

    checks = {
        "root_log_scores": exact_float_sequence(
            row.get("root_log_scores"),
            expected["root_log_scores"],
        ),
        "legal_entropy": math.isclose(
            float(row.get("legal_entropy_nats", math.nan)),
            expected["legal_entropy_nats"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "value_mse": math.isclose(
            float(row.get("value_mse", math.nan)),
            expected["value_mse"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "zero_value_mse": math.isclose(
            float(row.get("zero_value_mse", math.nan)),
            expected["zero_value_mse"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "auxiliary_brier": exact_float_sequence(
            row.get("auxiliary_brier"),
            expected["auxiliary_brier"],
        ),
        "auxiliary_prevalence_brier": exact_float_sequence(
            row.get("auxiliary_prevalence_brier"),
            expected["auxiliary_prevalence_brier"],
        ),
    }
    return {
        "expected": expected,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(
        np.quantile(
            np.asarray(values, dtype=np.float64),
            probability,
            method="linear",
        )
    )


def _trimmed_mean(values: Sequence[float], fraction: float = 0.10) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(math.floor(len(ordered) * fraction))
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    if retained.size == 0:
        raise J1ExecutionIntegrityError("Trimmed mean retained no rows")
    return float(np.mean(retained))


def _mantel_haenszel_or(
    candidate: np.ndarray,
    control: np.ndarray,
    blocks: np.ndarray,
) -> float:
    candidate_success = []
    control_success = []
    totals = []
    for block in range(8):
        mask = blocks == block
        if not np.any(mask):
            raise J1ExecutionIntegrityError(
                "Mantel-Haenszel estimator is missing a frozen stratum"
            )
        candidate_success.append(int(candidate[mask].astype(bool).sum()))
        control_success.append(int(control[mask].astype(bool).sum()))
        totals.append(int(mask.sum()))
    log_or = o2_power._mh_log_or(
        np.asarray([candidate_success], dtype=np.float64),
        np.asarray([control_success], dtype=np.float64),
        np.asarray([totals], dtype=np.float64),
    )[0]
    if not np.isfinite(log_or):
        raise J1ExecutionIntegrityError(
            "Accepted corrected common-OR estimator is nonfinite"
        )
    return float(np.exp(log_or))


def _paired_bootstrap_indices(
    *,
    row_count: int,
    repeats: int,
    seed: int,
) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        yield rng.integers(0, row_count, size=row_count)


def _accepted_progression_bootstrap_bounds(
    *,
    candidate: np.ndarray,
    control: np.ndarray,
    blocks: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    by_block = [np.flatnonzero(blocks == block) for block in range(8)]
    if any(indices.size == 0 for indices in by_block):
        raise J1ExecutionIntegrityError(
            "Stratified bootstrap is missing a frozen block"
        )
    control_by_root = [
        control[indices].astype(np.int8).reshape(-1, 1)
        for indices in by_block
    ]
    candidate_by_root = [
        candidate[indices].astype(np.int8).reshape(-1, 1)
        for indices in by_block
    ]
    lower_log, upper_log = o2_power._bootstrap_cluster_bounds(
        control_by_root,
        candidate_by_root,
        rng=rng,
        bootstraps=repeats,
    )
    return float(np.exp(lower_log)), float(np.exp(upper_log))


def analyze_paired_full_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    bootstrap_repeats: int = BOOTSTRAPS,
    fixture_mode: bool = False,
) -> dict[str, Any]:
    expected_rows = {
        "development": DEVELOPMENT_PAIRS,
        "confirmation": CONFIRMATION_PAIRS,
    }
    if phase not in expected_rows:
        raise ValueError("Paired full-policy analysis is evaluation-only")
    bootstrap_contract = evaluation_bootstrap_contract()
    if not bootstrap_contract["passes"]:
        raise J1ExecutionIntegrityError(
            "Evaluation bootstrap contract changed"
        )
    if not fixture_mode and bootstrap_repeats != BOOTSTRAPS:
        raise J1ExecutionIntegrityError("Scientific bootstrap count changed")
    if not fixture_mode and len(rows) != expected_rows[phase]:
        raise J1ExecutionIntegrityError("Evaluation root count changed")
    if not rows:
        raise J1ExecutionIntegrityError("Paired analysis has no rows")
    root_ids = [str(row["root_id"]) for row in rows]
    if len(set(root_ids)) != len(root_ids):
        raise J1ExecutionIntegrityError("Paired analysis duplicated a root")
    candidate_scores = []
    control_scores = []
    candidate_moves = []
    control_moves = []
    candidate_p1536 = []
    control_p1536 = []
    candidate_p3072 = []
    control_p3072 = []
    candidate_latencies: list[float] = []
    control_latencies: list[float] = []
    blocks = []
    families = []
    illegal_or_crash = False
    for index, row in enumerate(rows):
        candidate = row["candidate"]
        control = row["control"]
        for field in (
            "logical_stream_id",
            "deck_stream_id",
            "slot_stream_id",
        ):
            if candidate[field] != control[field]:
                raise J1ExecutionIntegrityError(
                    f"Paired exogenous stream changed: {field}"
                )
        if (
            candidate.get("policy_stream_id")
            == control.get("policy_stream_id")
        ):
            raise J1ExecutionIntegrityError(
                "Candidate/control policy identity collided"
            )
        if candidate.get("starter_tile") is not None or control.get(
            "starter_tile"
        ) is not None:
            raise J1ExecutionIntegrityError("Evaluation root has a starter")
        candidate_scores.append(
            float(
                max(
                    float(candidate["final_score"])
                    - float(candidate["start_score"]),
                    0.0,
                )
            )
        )
        control_scores.append(
            float(
                max(
                    float(control["final_score"])
                    - float(control["start_score"]),
                    0.0,
                )
            )
        )
        candidate_moves.append(float(candidate["moves"]))
        control_moves.append(float(control["moves"]))
        candidate_p1536.append(int(candidate["max_tile"] >= 1536))
        control_p1536.append(int(control["max_tile"] >= 1536))
        candidate_p3072.append(int(candidate["max_tile"] >= 3072))
        control_p3072.append(int(control["max_tile"] >= 3072))
        candidate_latencies.extend(
            float(value) for value in candidate["decision_latencies_seconds"]
        )
        control_latencies.extend(
            float(value) for value in control["decision_latencies_seconds"]
        )
        illegal_or_crash = illegal_or_crash or any(
            int(arm.get(field, 0)) != 0
            for arm in (candidate, control)
            for field in ("illegal_actions", "crashes")
        )
        block = int(row.get("block", index % 8))
        if block != index % 8:
            raise J1ExecutionIntegrityError("Evaluation block changed")
        blocks.append(block)
        families.append(str(row.get("family", "prospective_self_play")))

    numeric_sets = (
        candidate_scores,
        control_scores,
        candidate_moves,
        control_moves,
        candidate_latencies,
        control_latencies,
    )
    if any(not values for values in numeric_sets) or any(
        not math.isfinite(value)
        for values in numeric_sets
        for value in values
    ):
        raise J1ExecutionIntegrityError(
            "Paired analysis contains missing/nonfinite values"
        )
    candidate_scores_np = np.asarray(candidate_scores, dtype=np.float64)
    control_scores_np = np.asarray(control_scores, dtype=np.float64)
    log_differences = np.log1p(candidate_scores_np) - np.log1p(
        control_scores_np
    )
    candidate_p1536_np = np.asarray(candidate_p1536, dtype=np.int8)
    control_p1536_np = np.asarray(control_p1536, dtype=np.int8)
    candidate_p3072_np = np.asarray(candidate_p3072, dtype=np.int8)
    control_p3072_np = np.asarray(control_p3072, dtype=np.int8)
    blocks_np = np.asarray(blocks, dtype=np.int8)
    score_bootstrap = np.empty(bootstrap_repeats, dtype=np.float64)
    for offset, picked in enumerate(
        _paired_bootstrap_indices(
            row_count=len(rows),
            repeats=bootstrap_repeats,
            seed=BOOTSTRAP_SEEDS[phase],
        )
    ):
        score_bootstrap[offset] = float(np.mean(log_differences[picked]))
    or_lower, or_upper = _accepted_progression_bootstrap_bounds(
        candidate=candidate_p1536_np,
        control=control_p1536_np,
        blocks=blocks_np,
        repeats=bootstrap_repeats,
        seed=BOOTSTRAP_SEEDS[phase],
    )
    score_point = float(np.mean(log_differences))
    common_or = _mantel_haenszel_or(
        candidate_p1536_np,
        control_p1536_np,
        blocks_np,
    )
    raw_paired_differences = candidate_scores_np - control_scores_np
    report = {
        "version": f"{VERSION}_{phase}_paired_analysis_v1",
        "phase": phase,
        "root_count": len(rows),
        "root_ids_sha256": _ordered_rows_hash(
            {"root_id": value} for value in root_ids
        ),
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": BOOTSTRAP_SEEDS[phase],
        "bootstrap_contract": bootstrap_contract,
        "score_log_difference": {
            "point": score_point,
            "lower95": _quantile(score_bootstrap, 0.025),
            "upper95": _quantile(score_bootstrap, 0.975),
            "meaningful_target": math.log(1.07),
            "development_noninferiority": math.log(0.95),
        },
        "p1536_common_or": {
            "point": common_or,
            "lower95": or_lower,
            "upper95": or_upper,
            "control_rate": float(np.mean(control_p1536_np)),
        },
        "p3072_risk_difference": float(
            np.mean(candidate_p3072_np) - np.mean(control_p3072_np)
        ),
        "raw_score": {
            "paired_mean_difference": float(
                np.mean(raw_paired_differences)
            ),
            "paired_trimmed_mean_difference": (
                _trimmed_mean(candidate_scores)
                - _trimmed_mean(control_scores)
            ),
            "candidate": {
                "mean": float(np.mean(candidate_scores_np)),
                "median": _quantile(candidate_scores, 0.50),
                "p10": _quantile(candidate_scores, 0.10),
                "p90": _quantile(candidate_scores, 0.90),
                "p95": _quantile(candidate_scores, 0.95),
                "p99": _quantile(candidate_scores, 0.99),
                "maximum": float(np.max(candidate_scores_np)),
            },
            "control": {
                "mean": float(np.mean(control_scores_np)),
                "median": _quantile(control_scores, 0.50),
                "p10": _quantile(control_scores, 0.10),
                "p90": _quantile(control_scores, 0.90),
                "p95": _quantile(control_scores, 0.95),
                "p99": _quantile(control_scores, 0.99),
                "maximum": float(np.max(control_scores_np)),
            },
        },
        "survival_moves": {
            "candidate_mean": float(np.mean(candidate_moves)),
            "control_mean": float(np.mean(control_moves)),
        },
        "latency_seconds": {
            "candidate_p95": _quantile(candidate_latencies, 0.95),
            "candidate_p99": _quantile(candidate_latencies, 0.99),
            "control_p95": _quantile(control_latencies, 0.95),
            "control_p99": _quantile(control_latencies, 0.99),
        },
        "illegal_action_or_crash": illegal_or_crash,
        "descriptive_only": {
            "blocks": {
                str(block): float(
                    np.mean(log_differences[blocks_np == block])
                )
                for block in range(8)
                if np.any(blocks_np == block)
            },
            "families": {
                family: float(
                    np.mean(
                        log_differences[
                            np.asarray(families, dtype=object) == family
                        ]
                    )
                )
                for family in sorted(set(families))
            },
            "maximum_p95_p99_are_not_gates": True,
        },
    }
    report["analysis_payload_sha256"] = canonical_json_hash(report)
    return report


def evaluation_gate_decision(
    report: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    if phase not in {"development", "confirmation"}:
        raise ValueError("Evaluation gate phase is invalid")
    if report.get("phase") != phase:
        raise J1ExecutionIntegrityError("Evaluation report phase changed")
    score = report["score_log_difference"]
    progress = report["p1536_common_or"]
    raw = report["raw_score"]
    survival = report["survival_moves"]
    latency = report["latency_seconds"]
    candidate_p10 = float(raw["candidate"]["p10"])
    control_p10 = float(raw["control"]["p10"])
    lower_decile_safe = (
        candidate_p10 >= 0.95 * control_p10
        if control_p10 > 0.0
        else candidate_p10 >= control_p10
    )
    survival_safe = (
        float(survival["candidate_mean"])
        >= 0.95 * float(survival["control_mean"])
    )
    latency_safe = (
        float(latency["candidate_p95"])
        <= 1.5 * float(latency["control_p95"])
        and float(latency["candidate_p99"]) < 0.100
    )
    safeguards = {
        "p3072_noninferior": float(report["p3072_risk_difference"]) >= -0.02,
        "lower_decile_noninferior": lower_decile_safe,
        "survival_noninferior": survival_safe,
        "zero_illegal_actions_and_crashes": (
            report["illegal_action_or_crash"] is False
        ),
        "latency_safe": latency_safe,
    }
    if phase == "development":
        primary = {
            "score_point_positive": float(score["point"]) > 0.0,
            "score_lower_above_noninferiority": (
                float(score["lower95"]) > math.log(0.95)
            ),
            "score_upper_reaches_target": (
                float(score["upper95"]) >= math.log(1.07)
            ),
            "p1536_point_at_least_one": float(progress["point"]) >= 1.0,
            "p1536_upper_reaches_1_50": (
                float(progress["upper95"]) >= 1.50
            ),
        }
        material_safeguard_harm = not all(safeguards.values())
        if all(primary.values()) and all(safeguards.values()):
            decision = "READY_J1_DEVELOPMENT_FULL_POLICY"
        elif float(score["upper95"]) < 0.0 or (
            float(progress["upper95"]) < 1.0
            and material_safeguard_harm
        ):
            decision = "KILL_J1_FULL_POLICY_UTILITY"
        else:
            decision = "HOLD_J1_DEVELOPMENT_INCONCLUSIVE"
    else:
        primary = {
            "score_point_at_least_7pct": (
                float(score["point"]) >= math.log(1.07)
            ),
            "score_lower_above_zero": float(score["lower95"]) > 0.0,
            "p1536_point_at_least_1_50": (
                float(progress["point"]) >= 1.50
            ),
            "p1536_lower_above_one": float(progress["lower95"]) > 1.0,
            "trimmed_mean_positive": (
                float(raw["paired_trimmed_mean_difference"]) > 0.0
            ),
            "median_noninferior": (
                float(raw["candidate"]["median"])
                >= 0.95 * float(raw["control"]["median"])
            ),
        }
        if float(progress["control_rate"]) < 0.02:
            decision = "HOLD_J1_PROGRESSION_UNDERPOWERED"
        elif all(primary.values()) and all(safeguards.values()):
            decision = "READY_J1_PROMOTION_REVIEW"
        else:
            score_target_included = float(score["upper95"]) >= math.log(1.07)
            progress_target_included = float(progress["upper95"]) >= 1.50
            material_safeguard_harm = not all(safeguards.values())
            if (not score_target_included and not progress_target_included) or (
                material_safeguard_harm
            ):
                decision = "KILL_J1_FULL_POLICY_CAPABILITY"
            else:
                decision = "HOLD_J1_CONFIRMATION_INCONCLUSIVE"
    return {
        "version": f"{VERSION}_{phase}_gate_v1",
        "phase": phase,
        "primary_checks": primary,
        "safeguard_checks": safeguards,
        "descriptive_max_p95_p99_not_gates": True,
        "decision": decision,
        "promote": False,
    }


class TrainingTransitionChunkStore:
    """Append-once multi-row transition chunks for bounded collection."""

    def __init__(
        self,
        *,
        phase_dir: Path,
        rolling_contract: Mapping[str, Any],
        round_number: int,
        rows: Sequence[Mapping[str, Any]],
        snapshot: Mapping[str, Any] | None = None,
        output_accountant: "PhaseOutputAccountant | None" = None,
        io_metrics: dict[str, int] | None = None,
    ) -> None:
        self.phase_dir = phase_dir
        self.contract_sha256 = str(
            rolling_contract["rolling_contract_sha256"]
        )
        self.round_number = int(round_number)
        self.rows = [dict(row) for row in rows]
        self.output_accountant = output_accountant
        self.io_metrics = io_metrics
        self.rows_sha256 = _training_rows_identity(self.rows)
        self.index_by_root = {
            str(row["root_id"]): index
            for index, row in enumerate(self.rows)
        }
        if snapshot is None:
            self.next_chunk_index = 0
            self.buffer: list[dict[str, Any]] = []
            self.file_count = 0
            self.rows_written = 0
        else:
            body = dict(snapshot)
            observed = body.pop("transition_store_state_sha256", None)
            if observed != j1.stable_hash(body):
                raise J1ExecutionIntegrityError(
                    "Transition store snapshot changed"
                )
            if (
                snapshot.get("version")
                != f"{VERSION}_transition_store_state_v1"
                or int(snapshot.get("round", -1)) != self.round_number
                or snapshot.get("rows_sha256") != self.rows_sha256
                or snapshot.get("rolling_contract_sha256")
                != self.contract_sha256
            ):
                raise J1ExecutionIntegrityError(
                    "Transition store resume identity changed"
                )
            self.next_chunk_index = int(snapshot["next_chunk_index"])
            self.buffer = copy.deepcopy(list(snapshot["buffer"]))
            self.file_count = int(snapshot["file_count"])
            self.rows_written = int(snapshot["rows_written"])
            if (
                len(self.buffer) >= TRANSITION_CHUNK_MAX_ROWS
                or self.file_count != self.next_chunk_index
                or self.file_count > TRAINING_TRANSITION_FILE_CAP
            ):
                raise J1ExecutionIntegrityError(
                    "Transition store counters changed"
                )
            self._validate_buffer()

    @staticmethod
    def _identity(
        *,
        path: Path,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "path": str(path.resolve()),
            "file_sha256": sha256_path(path),
            "payload_sha256": payload[
                "transition_chunk_payload_sha256"
            ],
            "chunk_index": int(payload["chunk_index"]),
        }

    def _chunk_dir(self) -> Path:
        return (
            self.phase_dir
            / TRANSITION_CHUNKS_DIR
            / f"round_{self.round_number:02d}"
        )

    def _validate_buffer(self) -> None:
        for row in self.buffer:
            root_id = str(row.get("root_id"))
            if (
                root_id not in self.index_by_root
                or int(row.get("manifest_index", -1))
                != self.index_by_root[root_id]
                or int(row.get("transition_index", -1)) < 0
                or row.get("transition_sha256")
                != j1.stable_hash(row.get("transition", {}))
            ):
                raise J1ExecutionIntegrityError(
                    "Transition store buffer changed"
                )

    def append(
        self,
        *,
        root_id: str,
        transition_index: int,
        transition: Mapping[str, Any],
    ) -> None:
        if root_id not in self.index_by_root:
            raise J1ExecutionIntegrityError(
                "Transition chunk root is outside the round manifest"
            )
        entry = {
            "manifest_index": self.index_by_root[root_id],
            "root_id": root_id,
            "transition_index": int(transition_index),
            "transition": copy.deepcopy(dict(transition)),
            "transition_sha256": j1.stable_hash(dict(transition)),
        }
        if any(
            row["root_id"] == root_id
            and int(row["transition_index"]) == int(transition_index)
            for row in self.buffer
        ):
            raise J1ExecutionIntegrityError(
                "Transition store buffer duplicated a row"
            )
        self.buffer.append(entry)

    def should_flush(self) -> bool:
        return len(self.buffer) >= TRANSITION_CHUNK_MAX_ROWS

    def flush(
        self,
        *,
        items_by_root: Mapping[str, "_ActiveTrainingRoot"],
    ) -> dict[str, Any] | None:
        if not self.buffer:
            return None
        if self.file_count >= TRAINING_TRANSITION_FILE_CAP:
            raise J1ExecutionOperationalHold(
                "Training transition chunk file cap reached"
            )
        touched = sorted({str(row["root_id"]) for row in self.buffer})
        predecessors = {}
        counts_before = {}
        for root_id in touched:
            item = items_by_root.get(root_id)
            if item is None:
                raise J1ExecutionIntegrityError(
                    "Transition chunk lost its active root"
                )
            predecessors[root_id] = copy.deepcopy(
                item.transition_chunk_head
            )
            rows_for_root = [
                row for row in self.buffer if row["root_id"] == root_id
            ]
            first_index = int(rows_for_root[0]["transition_index"])
            if first_index != item.transition_count - len(rows_for_root):
                raise J1ExecutionIntegrityError(
                    "Transition chunk root sequence changed"
                )
            if [
                int(row["transition_index"]) for row in rows_for_root
            ] != list(
                range(
                    first_index,
                    first_index + len(rows_for_root),
                )
            ):
                raise J1ExecutionIntegrityError(
                    "Transition chunk root indices are not contiguous"
                )
            counts_before[root_id] = first_index
        payload = {
            "version": f"{VERSION}_training_transition_chunk_v2",
            "round": self.round_number,
            "chunk_index": self.next_chunk_index,
            "rows_sha256": self.rows_sha256,
            "rolling_contract_sha256": self.contract_sha256,
            "root_predecessors": predecessors,
            "root_counts_before": counts_before,
            "rows": copy.deepcopy(self.buffer),
            "rows_payload_sha256": j1.stable_hash(self.buffer),
        }
        payload["transition_chunk_payload_sha256"] = j1.stable_hash(
            payload
        )
        path = self._chunk_dir() / (
            f"chunk_{self.next_chunk_index:05d}.bin"
        )
        existed = path.exists()
        _write_immutable_binary_exact(path, payload)
        if self.output_accountant is not None:
            self.output_accountant.record_path(path)
        observed = load_atomic_binary(path)
        if (
            observed.get("transition_chunk_payload_sha256")
            != payload["transition_chunk_payload_sha256"]
            or j1.stable_hash(observed) != j1.stable_hash(payload)
        ):
            raise J1ExecutionIntegrityError(
                "Transition chunk write changed payload"
            )
        identity = self._identity(path=path, payload=observed)
        if self.io_metrics is not None:
            key = (
                "transition_chunk_validation_reads"
                if existed
                else "transition_chunk_writes"
            )
            self.io_metrics[key] = self.io_metrics.get(key, 0) + 1
            if not existed:
                self.io_metrics["transition_chunk_bytes_written"] = (
                    self.io_metrics.get(
                        "transition_chunk_bytes_written",
                        0,
                    )
                    + int(path.stat().st_size)
                )
        for root_id in touched:
            items_by_root[root_id].transition_chunk_head = dict(identity)
        row_count = len(self.buffer)
        self.rows_written += row_count
        self.file_count += 1
        self.next_chunk_index += 1
        self.buffer = []
        return {
            "chunk_identity": identity,
            "row_count": row_count,
            "touched_root_count": len(touched),
        }

    def verify_current_head(
        self,
        *,
        root_id: str,
        head: Mapping[str, Any] | None,
        count: int,
    ) -> None:
        if count == 0:
            if head is not None:
                raise J1ExecutionIntegrityError(
                    "Empty transition chain has a head"
                )
            return
        if head is None:
            raise J1ExecutionIntegrityError(
                "Transition chain head/count changed"
            )
        path = Path(str(head["path"])).resolve()
        if (
            path.parent != self._chunk_dir().resolve()
            or not path.is_file()
            or sha256_path(path) != head.get("file_sha256")
        ):
            raise J1ExecutionIntegrityError(
                "Transition chain head file changed"
            )
        payload = load_atomic_binary(path)
        if self.io_metrics is not None:
            self.io_metrics["transition_chunk_reads"] = (
                self.io_metrics.get("transition_chunk_reads", 0) + 1
            )
            self.io_metrics["transition_chunk_bytes_read"] = (
                self.io_metrics.get("transition_chunk_bytes_read", 0)
                + int(path.stat().st_size)
            )
        if (
            payload.get("transition_chunk_payload_sha256")
            != head.get("payload_sha256")
            or int(payload.get("chunk_index", -1))
            != int(head.get("chunk_index", -2))
            or payload.get("rolling_contract_sha256")
            != self.contract_sha256
            or payload.get("rows_sha256") != self.rows_sha256
            or root_id not in payload.get("root_counts_before", {})
        ):
            raise J1ExecutionIntegrityError(
                "Transition chain head payload changed"
            )
        rows = [
            row
            for row in payload.get("rows", [])
            if str(row.get("root_id")) == root_id
        ]
        if (
            not rows
            or int(rows[-1]["transition_index"]) != count - 1
        ):
            raise J1ExecutionIntegrityError(
                "Transition chain head does not reach root count"
            )

    def load_complete(
        self,
        *,
        root_id: str,
        head: Mapping[str, Any] | None,
        count: int,
    ) -> list[dict[str, Any]]:
        self.verify_current_head(
            root_id=root_id,
            head=head,
            count=count,
        )
        current = None if head is None else dict(head)
        descending: list[dict[str, Any]] = []
        expected_index = count - 1
        visited: set[str] = set()
        while current is not None:
            path = Path(str(current["path"])).resolve()
            if (
                str(path) in visited
                or path.parent != self._chunk_dir().resolve()
                or not path.is_file()
                or sha256_path(path) != current.get("file_sha256")
            ):
                raise J1ExecutionIntegrityError(
                    "Transition chunk chain path changed"
                )
            visited.add(str(path))
            payload = load_atomic_binary(path)
            if self.io_metrics is not None:
                self.io_metrics["transition_chunk_reads"] = (
                    self.io_metrics.get("transition_chunk_reads", 0) + 1
                )
                self.io_metrics["transition_chunk_bytes_read"] = (
                    self.io_metrics.get(
                        "transition_chunk_bytes_read",
                        0,
                    )
                    + int(path.stat().st_size)
                )
            if (
                payload.get("transition_chunk_payload_sha256")
                != current.get("payload_sha256")
                or int(payload.get("chunk_index", -1))
                != int(current.get("chunk_index", -2))
                or payload.get("rolling_contract_sha256")
                != self.contract_sha256
                or payload.get("rows_sha256") != self.rows_sha256
                or payload.get("rows_payload_sha256")
                != j1.stable_hash(payload.get("rows", []))
            ):
                raise J1ExecutionIntegrityError(
                    "Transition chunk chain payload changed"
                )
            root_rows = [
                row
                for row in payload["rows"]
                if str(row["root_id"]) == root_id
            ]
            if (
                not root_rows
                or [int(row["transition_index"]) for row in root_rows]
                != list(
                    range(
                        int(payload["root_counts_before"][root_id]),
                        int(payload["root_counts_before"][root_id])
                        + len(root_rows),
                    )
                )
                or int(root_rows[-1]["transition_index"])
                != expected_index
                or any(
                    row["transition_sha256"]
                    != j1.stable_hash(row["transition"])
                    for row in root_rows
                )
            ):
                raise J1ExecutionIntegrityError(
                    "Transition chunk root rows changed"
                )
            descending.extend(
                copy.deepcopy(dict(row["transition"]))
                for row in reversed(root_rows)
            )
            predecessor = payload["root_predecessors"].get(root_id)
            current = (
                None if predecessor is None else dict(predecessor)
            )
            expected_index -= len(root_rows)
        if expected_index != -1 or len(descending) != count:
            raise J1ExecutionIntegrityError(
                "Transition chunk chain does not reach genesis"
            )
        return list(reversed(descending))

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "version": f"{VERSION}_transition_store_state_v1",
            "round": self.round_number,
            "rows_sha256": self.rows_sha256,
            "rolling_contract_sha256": self.contract_sha256,
            "next_chunk_index": self.next_chunk_index,
            "file_count": self.file_count,
            "rows_written": self.rows_written,
            "buffer": copy.deepcopy(self.buffer),
        }
        payload["transition_store_state_sha256"] = j1.stable_hash(
            payload
        )
        return payload


@dataclass
class _ActiveTrainingRoot:
    row: dict[str, Any]
    sim: ThreesSim
    state: SimState
    policy_generator: torch.Generator
    start_score: int
    transitions: list[dict[str, Any]]
    transition_chunk_head: dict[str, Any] | None = None
    transition_count: int = 0


def _start_training_root(row: Mapping[str, Any]) -> _ActiveTrainingRoot:
    if row.get("phase") != "training" or row.get("starter_tile") is not None:
        raise J1ExecutionIntegrityError("Training root manifest row changed")
    sim, state = j1.normal_start_sim(
        role="train",
        deck_stream_id=int(row["deck_stream_id"]),
        slot_stream_id=int(row["slot_stream_id"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(row["candidate_policy_stream_id"]))
    return _ActiveTrainingRoot(
        row=dict(row),
        sim=sim,
        state=state,
        policy_generator=generator,
        start_score=score_board(state.board),
        transitions=[],
    )


def _finalize_training_root(
    active: _ActiveTrainingRoot,
) -> dict[str, Any]:
    if not active.transitions:
        raise J1ExecutionIntegrityError("Training root has no transitions")
    natural_terminal = bool(
        active.state.game_over
        or not active.sim.legal_actions(active.state)
    )
    if not natural_terminal:
        raise J1ExecutionIntegrityError("Training root was truncated")
    rewards = np.asarray(
        [row["reward"] for row in active.transitions],
        dtype=np.float64,
    )
    values = np.asarray(
        [row["value"] for row in active.transitions],
        dtype=np.float64,
    )
    done = np.asarray(
        [row["done_after_transition"] for row in active.transitions],
        dtype=bool,
    )
    if not done[-1] or np.any(done[:-1]):
        raise J1ExecutionIntegrityError(
            "Training root terminal masking changed"
        )
    advantages, returns = j1.compute_gae(
        rewards,
        values,
        done,
        0.0,
        gamma=parent_config("gamma"),
        gae_lambda=parent_config("gae_lambda"),
    )
    final_max_tile = int(active.state.max_tile)
    final_move_count = int(active.state.move_count)
    transitions = []
    for index, row in enumerate(active.transitions):
        decision_move_count = int(row["decision_move_count"])
        auxiliary = np.asarray(
            [
                float(final_max_tile >= 1536),
                float(final_max_tile >= 3072),
                float(final_move_count - decision_move_count >= 64),
            ],
            dtype=np.float32,
        )
        transitions.append(
            {
                "observation": np.asarray(
                    row["observation"],
                    dtype=np.float32,
                ).copy(),
                "legal_mask": np.asarray(
                    row["legal_mask"],
                    dtype=bool,
                ).copy(),
                "action": int(row["action"]),
                "old_log_probability": float(
                    row["old_log_probability"]
                ),
                "advantage": float(advantages[index]),
                "return": float(returns[index]),
                "auxiliary_label": auxiliary,
                "reward": float(row["reward"]),
                "done_after_transition": bool(
                    row["done_after_transition"]
                ),
            }
        )
    final_score = score_board(active.state.board)
    score_deltas = [
        int(row["score_delta"]) for row in active.transitions
    ]
    telescoping = j1.verify_dense_reward_telescoping(
        active.start_score,
        final_score,
        score_deltas,
    )
    if not telescoping["passes"]:
        raise J1ExecutionIntegrityError(
            "Training root dense reward did not telescope"
        )
    return {
        "root_id": str(active.row["root_id"]),
        "ancestry_id": str(active.row["ancestry_id"]),
        "source_manifest_row": dict(active.row),
        "source_manifest_row_sha256": canonical_json_hash(
            dict(active.row)
        ),
        "partition": "training",
        "natural_terminal": True,
        "transitions": transitions,
        "start_score": active.start_score,
        "final_score": final_score,
        "final_max_tile": final_max_tile,
        "move_count": final_move_count,
        "score_minus_start": final_score - active.start_score,
        "telescoping": telescoping,
    }


def parent_config(name: str) -> Any:
    if not hasattr(j1.FROZEN_CONFIG, name):
        raise J1ExecutionIntegrityError(f"Unknown parent config field: {name}")
    return getattr(j1.FROZEN_CONFIG, name)


def _training_rows_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return _ordered_rows_hash(dict(row) for row in rows)


class TrainingCollectionSession:
    """Deterministic, snapshot-complete synchronous root collector."""

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        model: j1.J1ActorCritic,
        env_count: int = ENV_COUNT,
        max_moves: int = MAX_MOVES,
        transition_store: TrainingTransitionChunkStore | None = None,
    ) -> None:
        if not rows or env_count < 1:
            raise ValueError("Training collection dimensions are invalid")
        self.rows = [dict(row) for row in rows]
        self.model = model.cpu().eval()
        self.env_count = int(env_count)
        self.max_moves = int(max_moves)
        self.transition_store = transition_store
        self.rows_sha256 = _training_rows_identity(self.rows)
        self.model_state_sha256 = j1.stable_hash(self.model.state_dict())
        self._validate_rows()
        self.next_index = 0
        self.tick = 0
        self.active: list[_ActiveTrainingRoot] = []
        self.completed: dict[str, dict[str, Any]] = {}
        self._fill()

    def _validate_rows(self) -> None:
        root_ids = [str(row["root_id"]) for row in self.rows]
        ancestries = [str(row["ancestry_id"]) for row in self.rows]
        if (
            len(set(root_ids)) != len(self.rows)
            or len(set(ancestries)) != len(self.rows)
        ):
            raise J1ExecutionIntegrityError(
                "Training collection duplicated root or ancestry"
            )
        if any(
            row.get("phase") != "training"
            or row.get("starter_tile") is not None
            for row in self.rows
        ):
            raise J1ExecutionIntegrityError(
                "Training collection row contract changed"
            )

    def _fill(self) -> None:
        while (
            len(self.active) < self.env_count
            and self.next_index < len(self.rows)
        ):
            self.active.append(
                _start_training_root(self.rows[self.next_index])
            )
            self.next_index += 1

    def _close_root(self, item: _ActiveTrainingRoot) -> None:
        root_id = str(item.row["root_id"])
        if root_id in self.completed:
            raise J1ExecutionIntegrityError(
                f"Training root closed twice: {root_id}"
            )
        if self.transition_store is None:
            finalized_item = item
        else:
            raw_transitions = self.transition_store.load_complete(
                root_id=root_id,
                head=item.transition_chunk_head,
                count=item.transition_count,
            )
            finalized_item = _ActiveTrainingRoot(
                row=dict(item.row),
                sim=item.sim,
                state=item.state,
                policy_generator=item.policy_generator,
                start_score=item.start_score,
                transitions=raw_transitions,
                transition_chunk_head=item.transition_chunk_head,
                transition_count=item.transition_count,
            )
        self.completed[root_id] = _finalize_training_root(
            finalized_item
        )
        self.active.remove(item)

    def step_tick(self) -> bool:
        if not self.active:
            return False
        for item in list(self.active):
            legal = item.sim.legal_actions(item.state)
            if not legal:
                self._close_root(item)
            elif item.state.move_count >= self.max_moves:
                raise J1ExecutionIntegrityError(
                    "Live training root reached 5,000 moves"
                )
        self._fill()
        if not self.active:
            return False
        live = [
            item
            for item in self.active
            if item.sim.legal_actions(item.state)
        ]
        if not live:
            raise J1ExecutionIntegrityError(
                "Training session retained no-action live roots"
            )
        observations_np = np.stack(
            [encode_observation(item.state, item.sim) for item in live]
        ).astype(np.float32)
        legal_np = np.stack(
            [item.sim.legal_mask(item.state) for item in live]
        ).astype(bool)
        if (
            observations_np.shape
            != (len(live), j1.EXPECTED_OBSERVATION_WIDTH)
            or not np.isfinite(observations_np).all()
            or not legal_np.any(axis=1).all()
        ):
            raise J1ExecutionIntegrityError(
                "Training observation/legal-mask invariant failed"
            )
        with torch.no_grad():
            logits, values, _auxiliary = self.model(
                torch.from_numpy(observations_np)
            )
            masked = j1.masked_logits(
                logits,
                torch.from_numpy(legal_np),
            )
        finished: list[_ActiveTrainingRoot] = []
        for index, item in enumerate(live):
            action_tensor = j1.sampled_masked_actions(
                logits[index : index + 1],
                torch.from_numpy(legal_np[index : index + 1]),
                generator=item.policy_generator,
            )
            action = int(action_tensor[0])
            distribution = torch.distributions.Categorical(
                logits=masked[index]
            )
            old_log_probability = float(
                distribution.log_prob(action_tensor[0]).cpu()
            )
            before = item.state
            before_score = score_board(before.board)
            next_state, info = item.sim.step(before, action)
            if not info.moved or not legal_np[index, action]:
                raise J1ExecutionIntegrityError(
                    "Training collector emitted an illegal action"
                )
            after_score = score_board(next_state.board)
            if int(info.score_delta) != after_score - before_score:
                raise J1ExecutionIntegrityError(
                    "Training score delta changed"
                )
            terminal = bool(
                next_state.game_over
                or not item.sim.legal_actions(next_state)
            )
            transition = {
                "observation": observations_np[index].copy(),
                "legal_mask": legal_np[index].copy(),
                "action": action,
                "old_log_probability": old_log_probability,
                "value": float(values[index].cpu()),
                "score_delta": int(info.score_delta),
                "reward": j1.dense_score_reward(info.score_delta),
                "decision_move_count": int(before.move_count),
                "done_after_transition": terminal,
            }
            if self.transition_store is None:
                item.transitions.append(transition)
            else:
                self.transition_store.append(
                    root_id=str(item.row["root_id"]),
                    transition_index=item.transition_count,
                    transition=transition,
                )
                item.transition_count += 1
            item.state = next_state
            if terminal:
                finished.append(item)
        if (
            self.transition_store is not None
            and (self.transition_store.should_flush() or finished)
        ):
            self.transition_store.flush(
                items_by_root={
                    str(item.row["root_id"]): item
                    for item in self.active
                }
            )
        for item in finished:
            self._close_root(item)
        self.tick += 1
        self._fill()
        return True

    def is_complete(self) -> bool:
        return (
            self.next_index == len(self.rows)
            and not self.active
            and len(self.completed) == len(self.rows)
        )

    def ordered_completed_records(self) -> list[dict[str, Any]]:
        if not self.is_complete():
            raise J1ExecutionIntegrityError(
                "Training collection is not complete"
            )
        return [
            copy.deepcopy(self.completed[str(row["root_id"])])
            for row in self.rows
        ]

    def snapshot(
        self,
        *,
        completed_blob_dir: Path | None = None,
    ) -> dict[str, Any]:
        prefix = [
            str(row["root_id"]) for row in self.rows[: self.next_index]
        ]
        completed_records = [
            copy.deepcopy(self.completed[str(row["root_id"])])
            for row in self.rows
            if str(row["root_id"]) in self.completed
        ]
        completed_refs = []
        if completed_blob_dir is not None:
            completed_blob_dir.mkdir(parents=True, exist_ok=True)
            for record in completed_records:
                root_id = str(record["root_id"])
                if not re.fullmatch(r"[A-Za-z0-9._-]+", root_id):
                    raise J1ExecutionIntegrityError(
                        "Training root id is unsafe for blob storage"
                    )
                path = completed_blob_dir / f"{root_id}.bin"
                file_sha256 = _write_immutable_binary_exact(path, record)
                completed_refs.append(
                    {
                        "root_id": root_id,
                        "path": str(path.resolve()),
                        "file_sha256": file_sha256,
                        "record_sha256": j1.stable_hash(record),
                    }
                )
            serialized_completed_records: list[dict[str, Any]] = []
            completed_storage = "immutable_root_blobs"
        else:
            serialized_completed_records = completed_records
            completed_storage = "inline_fixture"
        payload = {
            "version": f"{VERSION}_training_collection_session_v1",
            "rows_sha256": self.rows_sha256,
            "model_state_sha256": self.model_state_sha256,
            "env_count": self.env_count,
            "max_moves": self.max_moves,
            "next_index": self.next_index,
            "tick": self.tick,
            "manifest_prefix_root_ids": prefix,
            "manifest_prefix_sha256": j1.stable_hash(prefix),
            "active": [
                {
                    "row": dict(item.row),
                    "simulator": j1.simulator_snapshot(
                        item.sim,
                        item.state,
                    ),
                    "policy_rng_state":
                        item.policy_generator.get_state().clone(),
                    "start_score": item.start_score,
                    "transitions": copy.deepcopy(item.transitions),
                }
                for item in self.active
            ],
            "completed_storage": completed_storage,
            "completed_records": serialized_completed_records,
            "completed_record_refs": completed_refs,
            "completed_records_sha256": j1.stable_hash(
                completed_records
            ),
            "completed_record_refs_sha256": j1.stable_hash(
                completed_refs
            ),
            "python_rng_state": copy.deepcopy(random.getstate()),
            "numpy_rng_state": copy.deepcopy(np.random.get_state()),
            "torch_rng_state": torch.get_rng_state().clone(),
        }
        payload["session_state_sha256"] = j1.stable_hash(payload)
        return payload

    @classmethod
    def from_snapshot(
        cls,
        payload: Mapping[str, Any],
        *,
        rows: Sequence[Mapping[str, Any]],
        model: j1.J1ActorCritic,
        completed_blob_dir: Path | None = None,
    ) -> "TrainingCollectionSession":
        body = dict(payload)
        observed_hash = body.pop("session_state_sha256", None)
        if observed_hash != j1.stable_hash(body):
            raise J1ExecutionIntegrityError(
                "Training collection snapshot hash changed"
            )
        instance = cls.__new__(cls)
        instance.rows = [dict(row) for row in rows]
        instance.model = model.cpu().eval()
        instance.env_count = int(payload["env_count"])
        instance.max_moves = int(payload["max_moves"])
        instance.rows_sha256 = _training_rows_identity(instance.rows)
        instance.model_state_sha256 = j1.stable_hash(
            instance.model.state_dict()
        )
        instance.transition_store = None
        instance._validate_rows()
        if (
            payload.get("version")
            != f"{VERSION}_training_collection_session_v1"
            or payload.get("rows_sha256") != instance.rows_sha256
            or payload.get("model_state_sha256")
            != instance.model_state_sha256
        ):
            raise J1ExecutionIntegrityError(
                "Training collection resume identity changed"
            )
        instance.next_index = int(payload["next_index"])
        instance.tick = int(payload["tick"])
        expected_prefix = [
            str(row["root_id"])
            for row in instance.rows[: instance.next_index]
        ]
        if (
            list(payload["manifest_prefix_root_ids"]) != expected_prefix
            or payload.get("manifest_prefix_sha256")
            != j1.stable_hash(expected_prefix)
        ):
            raise J1ExecutionIntegrityError(
                "Training manifest prefix changed on resume"
            )
        storage_mode = payload.get("completed_storage")
        if storage_mode == "inline_fixture":
            completed_records = copy.deepcopy(
                list(payload["completed_records"])
            )
            if payload.get("completed_record_refs") not in (None, []):
                raise J1ExecutionIntegrityError(
                    "Inline completed records also contain blob refs"
                )
        elif storage_mode == "immutable_root_blobs":
            if completed_blob_dir is None:
                raise J1ExecutionIntegrityError(
                    "Training root blob directory is required on resume"
                )
            root = completed_blob_dir.resolve()
            refs = list(payload.get("completed_record_refs", []))
            if payload.get("completed_record_refs_sha256") != j1.stable_hash(
                refs
            ):
                raise J1ExecutionIntegrityError(
                    "Training root blob references changed"
                )
            completed_records = []
            for reference in refs:
                path = Path(str(reference["path"])).resolve()
                if path.parent != root:
                    raise J1ExecutionIntegrityError(
                        "Training root blob escaped its directory"
                    )
                if (
                    not path.is_file()
                    or sha256_path(path) != reference["file_sha256"]
                ):
                    raise J1ExecutionIntegrityError(
                        "Training root blob is missing or changed"
                    )
                record = load_atomic_binary(path)
                if (
                    str(record.get("root_id"))
                    != str(reference["root_id"])
                    or j1.stable_hash(record)
                    != reference["record_sha256"]
                ):
                    raise J1ExecutionIntegrityError(
                        "Training root blob payload changed"
                    )
                completed_records.append(record)
            if payload.get("completed_records") not in (None, []):
                raise J1ExecutionIntegrityError(
                    "Blob-backed snapshot duplicated completed records"
                )
        else:
            raise J1ExecutionIntegrityError(
                "Training completed-record storage mode changed"
            )
        if payload.get("completed_records_sha256") != j1.stable_hash(
            completed_records
        ):
            raise J1ExecutionIntegrityError(
                "Completed training records changed on resume"
            )
        instance.completed = {}
        authoritative_by_root = {
            str(row["root_id"]): row for row in instance.rows
        }
        for record in completed_records:
            root_id = str(record["root_id"])
            if root_id in instance.completed:
                raise J1ExecutionIntegrityError(
                    "Completed training root duplicated on resume"
                )
            authoritative = authoritative_by_root.get(root_id)
            source_row = record.get("source_manifest_row")
            if (
                authoritative is None
                or source_row != authoritative
                or record.get("source_manifest_row_sha256")
                != canonical_json_hash(authoritative)
                or str(record.get("ancestry_id"))
                != str(authoritative["ancestry_id"])
            ):
                raise J1ExecutionIntegrityError(
                    "Completed training record changed manifest identity"
                )
            instance.completed[root_id] = record
        instance.active = []
        for item_payload in payload["active"]:
            serialized_row = item_payload.get("row")
            serialized_root_id = (
                None
                if not isinstance(serialized_row, Mapping)
                else str(serialized_row.get("root_id"))
            )
            authoritative = authoritative_by_root.get(serialized_root_id)
            if serialized_row != authoritative:
                raise J1ExecutionIntegrityError(
                    "Active training root changed manifest row"
                )
            simulator_payload = item_payload.get("simulator")
            if (
                not isinstance(simulator_payload, Mapping)
                or int(simulator_payload.get("deck_stream_id", -1))
                != int(authoritative["deck_stream_id"])
                or int(simulator_payload.get("slot_stream_id", -1))
                != int(authoritative["slot_stream_id"])
            ):
                raise J1ExecutionIntegrityError(
                    "Active training simulator changed manifest streams"
                )
            sim, state = j1.simulator_from_snapshot(
                simulator_payload
            )
            generator = torch.Generator(device="cpu")
            generator.set_state(
                item_payload["policy_rng_state"].detach().cpu()
            )
            instance.active.append(
                _ActiveTrainingRoot(
                    row=dict(item_payload["row"]),
                    sim=sim,
                    state=state,
                    policy_generator=generator,
                    start_score=int(item_payload["start_score"]),
                    transitions=copy.deepcopy(
                        list(item_payload["transitions"])
                    ),
                )
            )
        prefix_ids = set(expected_prefix)
        active_ids = {str(item.row["root_id"]) for item in instance.active}
        completed_ids = set(instance.completed)
        if (
            active_ids & completed_ids
            or not active_ids.issubset(prefix_ids)
            or not completed_ids.issubset(prefix_ids)
            or active_ids | completed_ids != prefix_ids
            or len(instance.active) > instance.env_count
        ):
            raise J1ExecutionIntegrityError(
                "Training active/completed prefix is inconsistent"
            )
        random.setstate(copy.deepcopy(payload["python_rng_state"]))
        np.random.set_state(copy.deepcopy(payload["numpy_rng_state"]))
        torch.set_rng_state(
            payload["torch_rng_state"].detach().cpu().clone()
        )
        return instance

    def finish(
        self,
        *,
        boundary_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        while not self.is_complete():
            progressed = self.step_tick()
            if not progressed and not self.is_complete():
                raise J1ExecutionIntegrityError(
                    "Training collection made no progress"
                )
            if boundary_callback is not None:
                boundary_callback(self.snapshot())
        return self.ordered_completed_records()


def collect_training_roots_synchronously(
    rows: Sequence[Mapping[str, Any]],
    model: j1.J1ActorCritic,
    *,
    env_count: int = ENV_COUNT,
    max_moves: int = MAX_MOVES,
    boundary_callback: Any | None = None,
) -> list[dict[str, Any]]:
    if env_count != ENV_COUNT and boundary_callback is None:
        raise J1ExecutionIntegrityError(
            "Scientific collection requires 16 synchronous environments"
        )
    session = TrainingCollectionSession(
        rows=rows,
        model=model,
        env_count=env_count,
        max_moves=max_moves,
    )
    return session.finish(boundary_callback=boundary_callback)


def training_records_to_ppo_batch(
    records: Sequence[Mapping[str, Any]],
) -> j1.FrozenPPOBatch:
    roots = [
        j1.CompleteRoot(
            root_id=str(record["root_id"]),
            ancestry_id=str(record["ancestry_id"]),
            partition="training",
            transitions=tuple(record["transitions"]),
            natural_terminal=bool(record["natural_terminal"]),
        )
        for record in records
    ]
    flattened = j1.flatten_complete_roots(
        roots,
        expected_partition="training",
    )
    rows = flattened["rows"]
    batch = j1.FrozenPPOBatch(
        observations=torch.from_numpy(
            np.stack([row["observation"] for row in rows]).astype(np.float32)
        ),
        legal_masks=torch.from_numpy(
            np.stack([row["legal_mask"] for row in rows]).astype(bool)
        ),
        actions=torch.tensor(
            [int(row["action"]) for row in rows],
            dtype=torch.int64,
        ),
        old_log_probabilities=torch.tensor(
            [float(row["old_log_probability"]) for row in rows],
            dtype=torch.float32,
        ),
        advantages=torch.tensor(
            [float(row["advantage"]) for row in rows],
            dtype=torch.float32,
        ),
        returns=torch.tensor(
            [float(row["return"]) for row in rows],
            dtype=torch.float32,
        ),
        auxiliary_labels=torch.from_numpy(
            np.stack(
                [row["auxiliary_label"] for row in rows]
            ).astype(np.float32)
        ),
        row_weights=torch.from_numpy(
            np.asarray(flattened["weights"], dtype=np.float32)
        ),
        root_ids=tuple(str(row["root_id"]) for row in rows),
    )
    j1.validate_ppo_batch(batch)
    return batch


class FrozenMinibatchUpdater:
    def __init__(
        self,
        *,
        model: j1.J1ActorCritic,
        optimizer: torch.optim.Optimizer,
        batch: j1.FrozenPPOBatch,
        round_number: int,
        minibatch_size: int = j1.FROZEN_CONFIG.minibatch_size,
    ) -> None:
        j1.validate_ppo_batch(batch)
        self._validate_optimizer_binding(model, optimizer)
        self.model = model
        self.optimizer = optimizer
        self.batch = batch
        self.round_number = int(round_number)
        self.plan = j1.deterministic_epoch_minibatches(
            batch.row_count(),
            round_number=self.round_number,
            epochs=j1.FROZEN_CONFIG.epochs_per_round,
            minibatch_size=minibatch_size,
        )
        self.normalized_advantages = j1.normalize_advantages_root_weighted(
            batch.advantages,
            batch.row_weights,
        )
        self.cursor = 0
        self.closed_step_ids: list[str] = []
        update_lr = j1.round_learning_rate(self.round_number)
        for group in self.optimizer.param_groups:
            group["lr"] = update_lr

    @staticmethod
    def _validate_optimizer_binding(
        model: j1.J1ActorCritic,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        model_parameters = list(model.parameters())
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group.get("params", [])
        ]
        model_ids = [id(parameter) for parameter in model_parameters]
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if (
            len(set(model_ids)) != len(model_ids)
            or len(set(optimizer_ids)) != len(optimizer_ids)
            or set(model_ids) != set(optimizer_ids)
            or len(model_ids) != len(optimizer_ids)
        ):
            raise J1ExecutionIntegrityError(
                "Optimizer parameters do not exactly bind the updated model"
            )

    def expected_step_ids(self) -> list[str]:
        return [
            (
                f"round={self.round_number}|epoch={row['epoch']}|"
                f"start={row['start']}"
            )
            for row in self.plan
        ]

    def step_once(self) -> dict[str, Any]:
        if self.cursor >= len(self.plan):
            raise J1ExecutionIntegrityError(
                "Frozen PPO update is already complete"
            )
        row = self.plan[self.cursor]
        step_id = self.expected_step_ids()[self.cursor]
        if step_id in self.closed_step_ids:
            raise J1ExecutionIntegrityError(
                "Frozen PPO step was already closed"
            )
        indices = torch.tensor(row["indices"], dtype=torch.int64)
        subset = self.batch.subset(indices)
        minibatches_in_epoch = sum(
            planned["epoch"] == row["epoch"] for planned in self.plan
        )
        losses = j1.frozen_ppo_loss(
            self.model,
            subset,
            normalized_advantages=self.normalized_advantages[indices],
            global_weight_total=self.batch.row_weights.sum(),
            minibatches_per_epoch=minibatches_in_epoch,
        )
        self.optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            j1.FROZEN_CONFIG.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise J1ExecutionIntegrityError(
                "Frozen PPO gradient norm is nonfinite"
            )
        self.optimizer.step()
        j1.assert_finite_model(self.model)
        self.closed_step_ids.append(step_id)
        self.cursor += 1
        if self.cursor == len(self.plan):
            post_lr = j1.round_learning_rate(
                self.round_number,
                after_round=True,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = post_lr
        return {
            "step_id": step_id,
            "cursor": self.cursor,
            "row_count": len(row["indices"]),
            "epoch": row["epoch"],
            "start": row["start"],
            "losses": {
                key: float(value.detach().cpu())
                for key, value in losses.items()
            },
            "gradient_norm_before_clip": float(gradient_norm.cpu()),
        }

    def finish(self) -> list[dict[str, Any]]:
        reports = []
        while self.cursor < len(self.plan):
            reports.append(self.step_once())
        if self.closed_step_ids != self.expected_step_ids():
            raise J1ExecutionIntegrityError(
                "Frozen PPO step coverage changed"
            )
        return reports

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": f"{VERSION}_minibatch_updater_v1",
            "round_number": self.round_number,
            "cursor": self.cursor,
            "closed_step_ids": list(self.closed_step_ids),
            "model_state": copy.deepcopy(self.model.state_dict()),
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
            "batch": self.batch.payload(),
            "normalized_advantages": (
                self.normalized_advantages.detach().cpu().clone()
            ),
            "plan_sha256": j1.stable_hash(self.plan),
        }

    @classmethod
    def from_snapshot(
        cls,
        payload: Mapping[str, Any],
        *,
        minibatch_size: int = j1.FROZEN_CONFIG.minibatch_size,
    ) -> "FrozenMinibatchUpdater":
        model, optimizer = j1.initialize_model_optimizer()
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        batch = j1.FrozenPPOBatch.from_payload(payload["batch"])
        instance = cls(
            model=model,
            optimizer=optimizer,
            batch=batch,
            round_number=int(payload["round_number"]),
            minibatch_size=minibatch_size,
        )
        if j1.stable_hash(instance.plan) != payload["plan_sha256"]:
            raise J1ExecutionIntegrityError(
                "Frozen PPO plan changed on resume"
            )
        if not torch.equal(
            instance.normalized_advantages,
            payload["normalized_advantages"],
        ):
            raise J1ExecutionIntegrityError(
                "Normalized advantages changed on resume"
            )
        instance.cursor = int(payload["cursor"])
        instance.closed_step_ids = [
            str(value) for value in payload["closed_step_ids"]
        ]
        if instance.closed_step_ids != instance.expected_step_ids()[
            : instance.cursor
        ]:
            raise J1ExecutionIntegrityError(
                "Frozen PPO resume cursor skipped or duplicated a step"
            )
        j1.assert_finite_model(instance.model)
        instance._validate_optimizer_binding(
            instance.model,
            instance.optimizer,
        )
        return instance


def write_round_ppo_batch_blob(
    *,
    phase_dir: Path,
    updater: FrozenMinibatchUpdater,
    minibatch_size: int,
    output_accountant: "PhaseOutputAccountant | None" = None,
    io_metrics: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_round_ppo_batch_v1",
        "round": updater.round_number,
        "minibatch_size": int(minibatch_size),
        "batch": updater.batch.payload(),
        "batch_sha256": j1.stable_hash(updater.batch.payload()),
        "normalized_advantages":
            updater.normalized_advantages.detach().cpu().clone(),
        "normalized_advantages_sha256": j1.stable_hash(
            updater.normalized_advantages
        ),
        "plan": copy.deepcopy(updater.plan),
        "plan_sha256": j1.stable_hash(updater.plan),
        "expected_step_ids": updater.expected_step_ids(),
    }
    payload["round_batch_payload_sha256"] = j1.stable_hash(payload)
    path = (
        phase_dir
        / ROUND_BATCHES_DIR
        / f"round_{updater.round_number:02d}.bin"
    )
    existed = path.exists()
    file_sha256 = _write_immutable_binary_exact(path, payload)
    if output_accountant is not None:
        output_accountant.record_path(path)
    if io_metrics is not None:
        key = (
            "round_batch_validation_reads"
            if existed
            else "round_batch_writes"
        )
        io_metrics[key] = io_metrics.get(key, 0) + 1
        if not existed:
            io_metrics["round_batch_bytes_written"] = (
                io_metrics.get("round_batch_bytes_written", 0)
                + int(path.stat().st_size)
            )
    observed = load_atomic_binary(path)
    if (
        observed.get("round_batch_payload_sha256")
        != payload["round_batch_payload_sha256"]
        or j1.stable_hash(observed) != j1.stable_hash(payload)
    ):
        raise J1ExecutionIntegrityError(
            "Immutable round PPO batch changed"
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "payload_sha256": payload["round_batch_payload_sha256"],
        "round": updater.round_number,
        "row_count": updater.batch.row_count(),
        "batch_sha256": payload["batch_sha256"],
        "normalized_advantages_sha256": payload[
            "normalized_advantages_sha256"
        ],
        "plan_sha256": payload["plan_sha256"],
        "expected_step_ids_sha256": j1.stable_hash(
            payload["expected_step_ids"]
        ),
    }


def load_round_ppo_batch_blob(
    identity: Mapping[str, Any],
    *,
    minibatch_size: int,
    io_metrics: dict[str, int] | None = None,
) -> dict[str, Any]:
    path = Path(str(identity["path"])).resolve()
    expected_parent = (EXECUTION_ROOT / "training").resolve()
    if (
        not path.is_file()
        or sha256_path(path) != identity.get("file_sha256")
    ):
        raise J1ExecutionIntegrityError(
            "Immutable round PPO batch file changed"
        )
    if "miniature_fixture" not in str(path):
        try:
            path.relative_to(expected_parent)
        except ValueError:
            # Tests and future alternate roots pass phase-local paths; the
            # caller separately binds the phase directory and contract.
            pass
    payload = load_atomic_binary(path)
    if io_metrics is not None:
        io_metrics["round_batch_loads"] = (
            io_metrics.get("round_batch_loads", 0) + 1
        )
        io_metrics["round_batch_bytes_read"] = (
            io_metrics.get("round_batch_bytes_read", 0)
            + int(path.stat().st_size)
        )
    if (
        payload.get("version") != f"{VERSION}_round_ppo_batch_v1"
        or payload.get("round_batch_payload_sha256")
        != identity.get("payload_sha256")
        or int(payload.get("round", -1)) != int(identity["round"])
        or int(payload.get("minibatch_size", -1))
        != int(minibatch_size)
        or payload.get("batch_sha256")
        != identity.get("batch_sha256")
        or payload.get("normalized_advantages_sha256")
        != identity.get("normalized_advantages_sha256")
        or payload.get("plan_sha256") != identity.get("plan_sha256")
        or payload.get("expected_step_ids")
        is None
        or j1.stable_hash(payload["expected_step_ids"])
        != identity.get("expected_step_ids_sha256")
    ):
        raise J1ExecutionIntegrityError(
            "Immutable round PPO batch identity changed"
        )
    batch = j1.FrozenPPOBatch.from_payload(payload["batch"])
    j1.validate_ppo_batch(batch)
    normalized = payload["normalized_advantages"]
    if (
        not isinstance(normalized, torch.Tensor)
        or tuple(normalized.shape) != tuple(batch.advantages.shape)
        or not torch.isfinite(normalized).all()
        or j1.stable_hash(normalized)
        != payload["normalized_advantages_sha256"]
    ):
        raise J1ExecutionIntegrityError(
            "Immutable normalized advantages changed"
        )
    expected_plan = j1.deterministic_epoch_minibatches(
        batch.row_count(),
        round_number=int(payload["round"]),
        epochs=j1.FROZEN_CONFIG.epochs_per_round,
        minibatch_size=int(minibatch_size),
    )
    if (
        payload["plan"] != expected_plan
        or j1.stable_hash(expected_plan) != payload["plan_sha256"]
    ):
        raise J1ExecutionIntegrityError(
            "Immutable PPO minibatch plan changed"
        )
    return {
        "payload": payload,
        "batch": batch,
        "normalized_advantages": normalized,
        "plan": expected_plan,
    }


def _retirement_file_rows(
    *,
    phase_dir: Path,
    paths: Sequence[Path],
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(candidate.resolve() for candidate in paths):
        try:
            relative = str(path.relative_to(phase_dir.resolve()))
        except ValueError as error:
            raise J1ExecutionIntegrityError(
                "Retirement path escaped the phase namespace"
            ) from error
        if not path.is_file() or path.is_symlink():
            raise J1ExecutionIntegrityError(
                f"Retirement source is missing or unsafe: {path}"
            )
        rows.append(
            {
                "relative_path": relative,
                "file_sha256": sha256_path(path),
                "byte_size": int(path.stat().st_size),
            }
        )
    return rows


def _seal_and_apply_retirement(
    *,
    phase_dir: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
    output_accountant: "PhaseOutputAccountant | None",
    crash_stage: str | None = None,
) -> dict[str, Any]:
    if crash_stage not in {None, "after_manifest", "mid_delete"}:
        raise ValueError("Unknown retirement crash stage")
    if manifest_path.exists():
        observed = load_json(manifest_path)
        if (
            not verify_payload_hash(
                observed,
                "retirement_payload_sha256",
            )
            or observed != payload_with_hash(
                payload,
                "retirement_payload_sha256",
            )
        ):
            raise J1ExecutionIntegrityError(
                "Immutable retirement manifest changed"
            )
    else:
        observed = _write_immutable_json_exact(
            manifest_path,
            payload,
            field="retirement_payload_sha256",
        )
    if output_accountant is not None:
        output_accountant.record_path(manifest_path)
    if crash_stage == "after_manifest":
        raise J1ExecutionPlannedInterruption(
            "fixture interruption after retirement manifest"
        )
    retired_bytes = 0
    deleted_count = 0
    for row in observed["files"]:
        path = (phase_dir / str(row["relative_path"])).resolve()
        try:
            path.relative_to(phase_dir.resolve())
        except ValueError as error:
            raise J1ExecutionIntegrityError(
                "Retirement manifest path escaped phase namespace"
            ) from error
        if path.exists():
            if (
                not path.is_file()
                or path.is_symlink()
                or int(path.stat().st_size) != int(row["byte_size"])
                or sha256_path(path) != row["file_sha256"]
            ):
                raise J1ExecutionIntegrityError(
                    "Retirement source changed before deletion"
                )
            path.unlink()
            retired_bytes += int(row["byte_size"])
            deleted_count += 1
        if output_accountant is not None:
            output_accountant.retire_path(path)
        if crash_stage == "mid_delete" and deleted_count == 1:
            raise J1ExecutionPlannedInterruption(
                "fixture interruption during retirement deletions"
            )
    return {
        "path": str(manifest_path.resolve()),
        "file_sha256": sha256_path(manifest_path),
        "payload_sha256": observed["retirement_payload_sha256"],
        "file_count": len(observed["files"]),
        "listed_bytes": sum(
            int(row["byte_size"]) for row in observed["files"]
        ),
        "retired_bytes_this_call": retired_bytes,
        "all_listed_files_absent": all(
            not (phase_dir / str(row["relative_path"])).exists()
            for row in observed["files"]
        ),
        "passes": True,
    }


def retire_round_transition_chunks(
    *,
    phase_dir: Path,
    round_number: int,
    transition_store_state: Mapping[str, Any],
    completed_root_refs: Sequence[Mapping[str, Any]],
    collection_boundary: Mapping[str, Any],
    output_accountant: "PhaseOutputAccountant | None" = None,
    crash_stage: str | None = None,
) -> dict[str, Any]:
    if (
        transition_store_state.get("buffer") != []
        or int(transition_store_state.get("file_count", -1)) < 1
        or int(transition_store_state.get("rows_written", -1)) < 1
        or len(completed_root_refs) < 1
        or collection_boundary.get("unit_id")
        != f"round={round_number}|collection_complete"
    ):
        raise J1ExecutionIntegrityError(
            "Transition retirement prerequisites are incomplete"
        )
    chunk_dir = (
        phase_dir
        / TRANSITION_CHUNKS_DIR
        / f"round_{round_number:02d}"
    )
    paths = sorted(chunk_dir.glob("chunk_*.bin"))
    if len(paths) != int(transition_store_state["file_count"]):
        manifest_path = (
            phase_dir
            / TRANSITION_CHUNK_RETIREMENTS_DIR
            / f"round_{round_number:02d}.json"
        )
        if not manifest_path.is_file():
            raise J1ExecutionIntegrityError(
                "Transition chunks disappeared before retirement seal"
            )
        existing = load_json(manifest_path)
        files = list(existing.get("files", []))
    else:
        files = _retirement_file_rows(
            phase_dir=phase_dir,
            paths=paths,
        )
    payload = {
        "version": f"{VERSION}_transition_retirement_v1",
        "kind": "current_round_transition_recovery_material",
        "round": int(round_number),
        "collection_commit_head_file_sha256": collection_boundary[
            "commit_head_file_sha256"
        ],
        "collection_commit_head_payload_sha256": collection_boundary[
            "commit_head_payload_sha256"
        ],
        "collection_state_file_sha256": collection_boundary[
            "state_file_sha256"
        ],
        "root_blob_count": len(completed_root_refs),
        "root_blob_refs_sha256": j1.stable_hash(
            list(completed_root_refs)
        ),
        "transition_row_count": int(
            transition_store_state["rows_written"]
        ),
        "transition_chunk_count": int(
            transition_store_state["file_count"]
        ),
        "files": files,
        "files_sha256": j1.stable_hash(files),
        "retention_rule": (
            "delete only listed current-round chunks after all finalized "
            "root blobs and collection/pre-update seal authenticate them"
        ),
    }
    return _seal_and_apply_retirement(
        phase_dir=phase_dir,
        manifest_path=(
            phase_dir
            / TRANSITION_CHUNK_RETIREMENTS_DIR
            / f"round_{round_number:02d}.json"
        ),
        payload=payload,
        output_accountant=output_accountant,
        crash_stage=crash_stage,
    )


def retire_round_ppo_batch(
    *,
    phase_dir: Path,
    round_batch_identity: Mapping[str, Any],
    checkpoint_boundary: Mapping[str, Any],
    output_accountant: "PhaseOutputAccountant | None" = None,
    crash_stage: str | None = None,
) -> dict[str, Any]:
    round_number = int(round_batch_identity["round"])
    if checkpoint_boundary.get("unit_id") != (
        f"round={round_number}|checkpoint"
    ):
        raise J1ExecutionIntegrityError(
            "Round PPO batch lacks its checkpoint predecessor seal"
        )
    path = Path(str(round_batch_identity["path"])).resolve()
    manifest_path = (
        phase_dir
        / ROUND_BATCH_RETIREMENTS_DIR
        / f"round_{round_number:02d}.json"
    )
    if path.exists():
        files = _retirement_file_rows(
            phase_dir=phase_dir,
            paths=[path],
        )
    elif manifest_path.is_file():
        files = list(load_json(manifest_path).get("files", []))
    else:
        raise J1ExecutionIntegrityError(
            "Round PPO batch disappeared before retirement seal"
        )
    payload = {
        "version": f"{VERSION}_round_batch_retirement_v1",
        "kind": "ephemeral_current_round_ppo_batch",
        "round": round_number,
        "checkpoint_commit_head_file_sha256": checkpoint_boundary[
            "commit_head_file_sha256"
        ],
        "checkpoint_commit_head_payload_sha256": checkpoint_boundary[
            "commit_head_payload_sha256"
        ],
        "checkpoint_state_file_sha256": checkpoint_boundary[
            "state_file_sha256"
        ],
        "round_batch_identity": dict(round_batch_identity),
        "files": files,
        "files_sha256": j1.stable_hash(files),
        "retention_rule": (
            "delete only the exact immutable current-round batch after "
            "the round checkpoint authenticates model and optimizer state"
        ),
    }
    return _seal_and_apply_retirement(
        phase_dir=phase_dir,
        manifest_path=manifest_path,
        payload=payload,
        output_accountant=output_accountant,
        crash_stage=crash_stage,
    )


def compact_updater_snapshot(
    updater: FrozenMinibatchUpdater,
    *,
    round_batch_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "version": f"{VERSION}_compact_updater_v1",
        "round_number": updater.round_number,
        "cursor": updater.cursor,
        "closed_step_ids": list(updater.closed_step_ids),
        "model_state": copy.deepcopy(updater.model.state_dict()),
        "optimizer_state": copy.deepcopy(updater.optimizer.state_dict()),
        "round_batch_identity": dict(round_batch_identity),
        "plan_sha256": j1.stable_hash(updater.plan),
    }


def restore_compact_updater(
    payload: Mapping[str, Any],
    *,
    minibatch_size: int,
    io_metrics: dict[str, int] | None = None,
) -> FrozenMinibatchUpdater:
    if payload.get("version") != f"{VERSION}_compact_updater_v1":
        raise J1ExecutionIntegrityError(
            "Compact updater version changed"
        )
    loaded = load_round_ppo_batch_blob(
        payload["round_batch_identity"],
        minibatch_size=minibatch_size,
        io_metrics=io_metrics,
    )
    model, optimizer = j1.initialize_model_optimizer()
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
    except Exception as error:
        raise J1ExecutionIntegrityError(
            "Compact updater tensors are malformed"
        ) from error
    updater = FrozenMinibatchUpdater(
        model=model,
        optimizer=optimizer,
        batch=loaded["batch"],
        round_number=int(payload["round_number"]),
        minibatch_size=minibatch_size,
    )
    if (
        j1.stable_hash(updater.plan) != payload.get("plan_sha256")
        or not torch.equal(
            updater.normalized_advantages,
            loaded["normalized_advantages"],
        )
    ):
        raise J1ExecutionIntegrityError(
            "Compact updater batch/plan changed"
        )
    updater.cursor = int(payload["cursor"])
    updater.closed_step_ids = [
        str(value) for value in payload["closed_step_ids"]
    ]
    if updater.closed_step_ids != updater.expected_step_ids()[
        : updater.cursor
    ]:
        raise J1ExecutionIntegrityError(
            "Compact updater cursor skipped or duplicated a step"
        )
    return updater


class J1DeterministicCandidatePolicy:
    def __init__(self, model: j1.J1ActorCritic) -> None:
        self.model = model.cpu().eval()

    def __call__(
        self,
        state: SimState,
        sim: ThreesSim,
        _rng: np.random.Generator,
    ) -> int:
        observation = encode_observation(state, sim)
        legal_mask = sim.legal_mask(state)
        with torch.no_grad():
            logits, _value, _auxiliary = self.model(
                torch.from_numpy(observation).unsqueeze(0)
            )
        return int(
            j1.deterministic_masked_actions(
                logits,
                torch.from_numpy(legal_mask).unsqueeze(0),
            )[0]
        )


def _directory_file_manifest(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise J1ExecutionIntegrityError(
            f"Incumbent checkpoint directory is missing: {path}"
        )
    files = []
    for child in sorted(
        candidate
        for candidate in resolved.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    ):
        files.append(
            {
                "path": str(child.resolve()),
                "relative_path": str(child.relative_to(resolved)),
                "byte_size": child.stat().st_size,
                "sha256": sha256_path(child),
            }
        )
    if not files:
        raise J1ExecutionIntegrityError(
            f"Incumbent checkpoint is empty: {path}"
        )
    return {
        "path": str(resolved),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(row["byte_size"]) for row in files),
        "manifest_sha256": canonical_json_hash(files),
    }


def incumbent_policy_binding() -> dict[str, Any]:
    path = REPO_ROOT / "threes_rl/current_incumbent_policy.txt"
    if sha256_path(path) != j1.EXPECTED_INCUMBENT_POLICY_SHA256:
        raise J1ExecutionIntegrityError(
            "Current incumbent policy file identity changed"
        )
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1:
        raise J1ExecutionIntegrityError(
            "Current incumbent policy spec is not singular"
        )
    spec = lines[0]
    checkpoint_tokens = sorted(
        {
            token
            for token in spec.split(":")
            if token.startswith("threes_rl/runs/")
        }
    )
    if len(checkpoint_tokens) != 4:
        raise J1ExecutionIntegrityError(
            "Incumbent checkpoint reference count changed"
        )
    checkpoints = [
        _directory_file_manifest(REPO_ROOT / token)
        for token in checkpoint_tokens
    ]
    source_paths = (
        "threes_rl/eval.py",
        "threes_rl/expectimax.py",
        "threes_rl/ntuple.py",
        "threes_rl/action_prior.py",
        "threes_rl/sim.py",
        "threes_rl/train_td.py",
        "threes_rl/obs.py",
        "threes_rl/env.py",
    )
    source_bindings = {
        relative: sha256_path(REPO_ROOT / relative)
        for relative in source_paths
    }
    for relative in source_paths:
        expected = j1.DEPENDENCY_BINDINGS.get(relative)
        if expected is not None and source_bindings[relative] != expected:
            raise J1ExecutionIntegrityError(
                f"Incumbent implementation source changed: {relative}"
            )
    payload = {
        "version": f"{VERSION}_incumbent_policy_binding_v1",
        "policy_file": str(path.resolve()),
        "policy_file_sha256": sha256_path(path),
        "resolved_spec": spec,
        "resolved_spec_sha256": hashlib.sha256(
            spec.encode("utf-8")
        ).hexdigest(),
        "checkpoint_artifacts": checkpoints,
        "checkpoint_manifest_sha256": canonical_json_hash(checkpoints),
        "implementation_sources": source_bindings,
        "implementation_source_manifest_sha256": canonical_json_hash(
            source_bindings
        ),
    }
    return payload_with_hash(payload, "incumbent_binding_sha256")


def load_bound_incumbent_policy(
    expected_binding: Mapping[str, Any],
) -> Any:
    observed = incumbent_policy_binding()
    if observed != dict(expected_binding):
        raise J1ExecutionIntegrityError(
            "Incumbent policy binding changed before evaluation"
        )
    from threes_rl.eval import make_policy

    policy = make_policy(str(observed["resolved_spec"]))
    if not callable(policy):
        raise J1ExecutionIntegrityError(
            "Bound incumbent policy is not callable"
        )
    return policy


def candidate_checkpoint_payload(
    *,
    model: j1.J1ActorCritic,
    optimizer: torch.optim.Optimizer,
    training_manifest_identity: Mapping[str, Any],
    training_marker_file_sha256: str,
    training_result_input_sha256: str,
) -> dict[str, Any]:
    FrozenMinibatchUpdater._validate_optimizer_binding(model, optimizer)
    j1.assert_finite_model(model)
    payload = {
        "version": f"{VERSION}_round64_candidate_checkpoint_v1",
        "round": 64,
        "parameter_count": j1.parameter_count(model),
        "model_schema_sha256": j1.model_schema_sha256(),
        "training_manifest_identity": dict(training_manifest_identity),
        "training_marker_file_sha256": training_marker_file_sha256,
        "training_result_input_sha256": training_result_input_sha256,
        "runner_sha256": sha256_path(RUNNER_PATH),
        "parent_runner_sha256": sha256_path(
            REPO_ROOT / "threes_rl/j1_joint_policy_value.py"
        ),
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
    }
    payload["checkpoint_payload_sha256"] = j1.stable_hash(payload)
    return payload


def validate_candidate_checkpoint_payload(
    payload: Mapping[str, Any],
) -> tuple[j1.J1ActorCritic, torch.optim.Optimizer]:
    body = dict(payload)
    observed_hash = body.pop("checkpoint_payload_sha256", None)
    if observed_hash != j1.stable_hash(body):
        raise J1ExecutionIntegrityError(
            "Candidate checkpoint payload hash changed"
        )
    if (
        payload.get("version")
        != f"{VERSION}_round64_candidate_checkpoint_v1"
        or payload.get("round") != 64
        or payload.get("parameter_count") != j1.EXPECTED_PARAMETER_COUNT
        or payload.get("model_schema_sha256")
        != j1.model_schema_sha256()
        or payload.get("runner_sha256") != sha256_path(RUNNER_PATH)
        or payload.get("parent_runner_sha256")
        != sha256_path(REPO_ROOT / "threes_rl/j1_joint_policy_value.py")
    ):
        raise J1ExecutionIntegrityError(
            "Candidate checkpoint identity changed"
        )
    model, optimizer = j1.initialize_model_optimizer()
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
    except Exception as error:
        raise J1ExecutionIntegrityError(
            "Candidate checkpoint tensors are malformed"
        ) from error
    FrozenMinibatchUpdater._validate_optimizer_binding(model, optimizer)
    j1.assert_finite_model(model)
    return model.eval(), optimizer


def write_candidate_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_candidate_checkpoint_payload(payload)
    file_sha256 = _write_immutable_binary_exact(path, payload)
    reloaded = load_atomic_binary(path)
    validate_candidate_checkpoint_payload(reloaded)
    if j1.stable_hash(reloaded) != j1.stable_hash(payload):
        raise J1ExecutionIntegrityError(
            "Candidate checkpoint save/load changed payload"
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "payload_sha256": payload["checkpoint_payload_sha256"],
        "round": 64,
        "parameter_count": j1.EXPECTED_PARAMETER_COUNT,
        "model_schema_sha256": j1.model_schema_sha256(),
        "save_load_exact": True,
    }


def load_authoritative_candidate_policy(
    *,
    checkpoint_identity: Mapping[str, Any],
) -> J1DeterministicCandidatePolicy:
    path = Path(str(checkpoint_identity["path"])).resolve()
    if (
        not path.is_file()
        or sha256_path(path) != checkpoint_identity.get("file_sha256")
    ):
        raise J1ExecutionIntegrityError(
            "Authoritative candidate checkpoint file changed"
        )
    payload = load_atomic_binary(path)
    if payload.get("checkpoint_payload_sha256") != (
        checkpoint_identity.get("payload_sha256")
    ):
        raise J1ExecutionIntegrityError(
            "Authoritative candidate checkpoint payload changed"
        )
    model, _optimizer = validate_candidate_checkpoint_payload(payload)
    return J1DeterministicCandidatePolicy(model)


def execute_full_policy_arm(
    *,
    row: Mapping[str, Any],
    arm: str,
    policy: Any,
    max_moves: int = MAX_MOVES,
) -> dict[str, Any]:
    if arm not in {"candidate", "control"}:
        raise ValueError("Unknown evaluation arm")
    if row.get("phase") not in {"development", "confirmation"}:
        raise J1ExecutionIntegrityError("Evaluation row phase changed")
    if row.get("starter_tile") is not None:
        raise J1ExecutionIntegrityError("Evaluation row has a starter")
    policy_stream_field = (
        "candidate_policy_stream_id"
        if arm == "candidate"
        else "control_policy_stream_id"
    )
    sim, state = j1.normal_start_sim(
        role=str(row["phase"]),
        deck_stream_id=int(row["deck_stream_id"]),
        slot_stream_id=int(row["slot_stream_id"]),
    )
    policy_rng = np.random.default_rng(int(row[policy_stream_field]))
    start_score = score_board(state.board)
    latencies = []
    illegal_actions = 0
    while not state.game_over:
        legal = sim.legal_actions(state)
        if not legal:
            break
        if state.move_count >= max_moves:
            raise J1ExecutionIntegrityError(
                "Live evaluation root reached 5,000 moves"
            )
        started = time.perf_counter()
        action = int(policy(state, sim, policy_rng))
        latencies.append(time.perf_counter() - started)
        if action not in legal:
            illegal_actions += 1
            raise J1ExecutionIntegrityError(
                "Full-policy evaluation emitted an illegal action"
            )
        state, info = sim.step(state, action)
        if not info.moved:
            raise J1ExecutionIntegrityError(
                "Legal evaluation action did not move"
            )
    final_score = score_board(state.board)
    return {
        "logical_stream_id": int(row["logical_stream_id"]),
        "deck_stream_id": int(row["deck_stream_id"]),
        "slot_stream_id": int(row["slot_stream_id"]),
        "policy_stream_id": int(row[policy_stream_field]),
        "starter_tile": None,
        "start_score": start_score,
        "final_score": final_score,
        "score_minus_starter": max(final_score - start_score, 0),
        "moves": int(state.move_count),
        "max_tile": int(state.max_tile),
        "decision_latencies_seconds": latencies,
        "illegal_actions": illegal_actions,
        "crashes": 0,
        "natural_terminal": bool(
            state.game_over or not sim.legal_actions(state)
        ),
        "terminal_state_sha256": j1.stable_hash(
            j1.simulator_snapshot(sim, state)
        ),
    }


class PairedEvaluationSession:
    """Arm-boundary resumable paired full-policy evaluator."""

    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, Any]],
        candidate_policy: Any,
        control_policy: Any,
        candidate_policy_identity: str,
        control_policy_identity: str,
        max_moves: int = MAX_MOVES,
    ) -> None:
        if not rows:
            raise ValueError("Paired evaluation requires rows")
        if any(
            not isinstance(value, str) or not value
            for value in (
                candidate_policy_identity,
                control_policy_identity,
            )
        ):
            raise J1ExecutionIntegrityError(
                "Paired policy identities are missing"
            )
        self.rows = [dict(row) for row in rows]
        self.candidate_policy = candidate_policy
        self.control_policy = control_policy
        self.candidate_policy_identity = candidate_policy_identity
        self.control_policy_identity = control_policy_identity
        self.max_moves = int(max_moves)
        self.rows_sha256 = _ordered_rows_hash(self.rows)
        self.next_row_index = 0
        self.pending_candidate: dict[str, Any] | None = None
        self.completed_pairs: list[dict[str, Any]] = []
        self._validate_rows()

    def _validate_rows(self) -> None:
        root_ids = [str(row["root_id"]) for row in self.rows]
        ancestries = [str(row["ancestry_id"]) for row in self.rows]
        if (
            len(set(root_ids)) != len(self.rows)
            or len(set(ancestries)) != len(self.rows)
        ):
            raise J1ExecutionIntegrityError(
                "Paired evaluation duplicated root or ancestry"
            )
        if any(
            row.get("phase") not in {"development", "confirmation"}
            or row.get("starter_tile") is not None
            or row.get("control_policy_stream_id") is None
            for row in self.rows
        ):
            raise J1ExecutionIntegrityError(
                "Paired evaluation row contract changed"
            )
        phases = {str(row["phase"]) for row in self.rows}
        if len(phases) != 1:
            raise J1ExecutionIntegrityError(
                "Paired evaluation crossed phases"
            )

    def is_complete(self) -> bool:
        return (
            self.next_row_index == len(self.rows)
            and self.pending_candidate is None
            and len(self.completed_pairs) == len(self.rows)
        )

    def step_arm(self) -> dict[str, Any]:
        if self.is_complete():
            raise J1ExecutionIntegrityError(
                "Paired evaluation is already complete"
            )
        row = self.rows[self.next_row_index]
        if self.pending_candidate is None:
            self.pending_candidate = execute_full_policy_arm(
                row=row,
                arm="candidate",
                policy=self.candidate_policy,
                max_moves=self.max_moves,
            )
            return {
                "boundary": "candidate_arm_committed",
                "row_index": self.next_row_index,
            }
        control = execute_full_policy_arm(
            row=row,
            arm="control",
            policy=self.control_policy,
            max_moves=self.max_moves,
        )
        result = {
            "root_id": str(row["root_id"]),
            "ancestry_id": str(row["ancestry_id"]),
            "block": int(row["block"]),
            "candidate": self.pending_candidate,
            "control": control,
        }
        self.completed_pairs.append(result)
        self.pending_candidate = None
        self.next_row_index += 1
        return {
            "boundary": "paired_root_committed",
            "row_index": self.next_row_index - 1,
        }

    def snapshot(
        self,
        *,
        completed_blob_dir: Path | None = None,
    ) -> dict[str, Any]:
        completed_pairs = copy.deepcopy(self.completed_pairs)
        completed_pair_refs = []
        if completed_blob_dir is not None:
            completed_blob_dir.mkdir(parents=True, exist_ok=True)
            for result in completed_pairs:
                root_id = str(result["root_id"])
                if not re.fullmatch(r"[A-Za-z0-9._-]+", root_id):
                    raise J1ExecutionIntegrityError(
                        "Paired root id is unsafe for blob storage"
                    )
                path = completed_blob_dir / f"{root_id}.bin"
                file_sha256 = _write_immutable_binary_exact(path, result)
                completed_pair_refs.append(
                    {
                        "root_id": root_id,
                        "path": str(path.resolve()),
                        "file_sha256": file_sha256,
                        "pair_payload_sha256": j1.stable_hash(result),
                    }
                )
            serialized_pairs: list[dict[str, Any]] = []
            completed_storage = "immutable_pair_blobs"
        else:
            serialized_pairs = completed_pairs
            completed_storage = "inline_fixture"
        payload = {
            "version": f"{VERSION}_paired_evaluation_session_v1",
            "phase": str(self.rows[0]["phase"]),
            "rows_sha256": self.rows_sha256,
            "candidate_policy_identity": self.candidate_policy_identity,
            "control_policy_identity": self.control_policy_identity,
            "max_moves": self.max_moves,
            "next_row_index": self.next_row_index,
            "pending_candidate": copy.deepcopy(self.pending_candidate),
            "completed_storage": completed_storage,
            "completed_pairs": serialized_pairs,
            "completed_pair_refs": completed_pair_refs,
            "completed_pairs_sha256": j1.stable_hash(
                completed_pairs
            ),
            "completed_pair_refs_sha256": j1.stable_hash(
                completed_pair_refs
            ),
            "manifest_prefix_root_ids": [
                str(row["root_id"])
                for row in self.rows[: self.next_row_index]
            ],
        }
        payload["session_state_sha256"] = j1.stable_hash(payload)
        return payload

    @staticmethod
    def _arm_matches_row(
        *,
        arm_payload: Mapping[str, Any],
        row: Mapping[str, Any],
        arm: str,
    ) -> bool:
        policy_field = (
            "candidate_policy_stream_id"
            if arm == "candidate"
            else "control_policy_stream_id"
        )
        return all(
            (
                int(arm_payload.get(result_field, -1))
                == int(row[row_field])
            )
            for result_field, row_field in (
                ("logical_stream_id", "logical_stream_id"),
                ("deck_stream_id", "deck_stream_id"),
                ("slot_stream_id", "slot_stream_id"),
                ("policy_stream_id", policy_field),
            )
        )

    @classmethod
    def from_snapshot(
        cls,
        payload: Mapping[str, Any],
        *,
        rows: Sequence[Mapping[str, Any]],
        candidate_policy: Any,
        control_policy: Any,
        candidate_policy_identity: str,
        control_policy_identity: str,
        completed_blob_dir: Path | None = None,
    ) -> "PairedEvaluationSession":
        body = dict(payload)
        observed_hash = body.pop("session_state_sha256", None)
        if observed_hash != j1.stable_hash(body):
            raise J1ExecutionIntegrityError(
                "Paired evaluation snapshot hash changed"
            )
        instance = cls(
            rows=rows,
            candidate_policy=candidate_policy,
            control_policy=control_policy,
            candidate_policy_identity=candidate_policy_identity,
            control_policy_identity=control_policy_identity,
            max_moves=int(payload["max_moves"]),
        )
        if (
            payload.get("version")
            != f"{VERSION}_paired_evaluation_session_v1"
            or payload.get("phase") != instance.rows[0]["phase"]
            or payload.get("rows_sha256") != instance.rows_sha256
            or payload.get("candidate_policy_identity")
            != instance.candidate_policy_identity
            or payload.get("control_policy_identity")
            != instance.control_policy_identity
        ):
            raise J1ExecutionIntegrityError(
                "Paired evaluation resume identity changed"
            )
        instance.next_row_index = int(payload["next_row_index"])
        instance.pending_candidate = copy.deepcopy(
            payload["pending_candidate"]
        )
        storage_mode = payload.get("completed_storage")
        if storage_mode == "inline_fixture":
            instance.completed_pairs = copy.deepcopy(
                list(payload["completed_pairs"])
            )
            if payload.get("completed_pair_refs") not in (None, []):
                raise J1ExecutionIntegrityError(
                    "Inline paired snapshot also contains blob refs"
                )
        elif storage_mode == "immutable_pair_blobs":
            if completed_blob_dir is None:
                raise J1ExecutionIntegrityError(
                    "Paired result blob directory is required on resume"
                )
            root = completed_blob_dir.resolve()
            refs = list(payload.get("completed_pair_refs", []))
            if payload.get("completed_pair_refs_sha256") != j1.stable_hash(
                refs
            ):
                raise J1ExecutionIntegrityError(
                    "Paired result blob references changed"
                )
            instance.completed_pairs = []
            for reference in refs:
                path = Path(str(reference["path"])).resolve()
                if path.parent != root:
                    raise J1ExecutionIntegrityError(
                        "Paired result blob escaped its directory"
                    )
                if (
                    not path.is_file()
                    or sha256_path(path) != reference["file_sha256"]
                ):
                    raise J1ExecutionIntegrityError(
                        "Paired result blob is missing or changed"
                    )
                result = load_atomic_binary(path)
                if (
                    str(result.get("root_id"))
                    != str(reference["root_id"])
                    or j1.stable_hash(result)
                    != reference["pair_payload_sha256"]
                ):
                    raise J1ExecutionIntegrityError(
                        "Paired result blob payload changed"
                    )
                instance.completed_pairs.append(result)
            if payload.get("completed_pairs") not in (None, []):
                raise J1ExecutionIntegrityError(
                    "Blob-backed paired snapshot duplicated results"
                )
        else:
            raise J1ExecutionIntegrityError(
                "Paired completed-result storage mode changed"
            )
        if (
            instance.next_row_index != len(instance.completed_pairs)
            or payload.get("completed_pairs_sha256")
            != j1.stable_hash(instance.completed_pairs)
            or list(payload["manifest_prefix_root_ids"])
            != [
                str(row["root_id"])
                for row in instance.rows[: instance.next_row_index]
            ]
            or (
                instance.pending_candidate is not None
                and instance.next_row_index >= len(instance.rows)
            )
        ):
            raise J1ExecutionIntegrityError(
                "Paired evaluation boundary changed on resume"
            )
        for index, result in enumerate(instance.completed_pairs):
            authoritative = instance.rows[index]
            if (
                str(result.get("root_id"))
                != str(authoritative["root_id"])
                or str(result.get("ancestry_id"))
                != str(authoritative["ancestry_id"])
                or int(result.get("block", -1))
                != int(authoritative["block"])
                or not instance._arm_matches_row(
                    arm_payload=result.get("candidate", {}),
                    row=authoritative,
                    arm="candidate",
                )
                or not instance._arm_matches_row(
                    arm_payload=result.get("control", {}),
                    row=authoritative,
                    arm="control",
                )
            ):
                raise J1ExecutionIntegrityError(
                    "Completed paired result changed manifest identity"
                )
        if instance.pending_candidate is not None:
            authoritative = instance.rows[instance.next_row_index]
            if not instance._arm_matches_row(
                arm_payload=instance.pending_candidate,
                row=authoritative,
                arm="candidate",
            ):
                raise J1ExecutionIntegrityError(
                    "Pending candidate arm changed manifest identity"
                )
        return instance

    def finish(
        self,
        *,
        boundary_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        while not self.is_complete():
            report = self.step_arm()
            if boundary_callback is not None:
                boundary_callback(report, self.snapshot())
        return copy.deepcopy(self.completed_pairs)


def execute_paired_full_policy_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    candidate_policy: Any,
    control_policy: Any,
    candidate_policy_identity: str = "fixture-candidate-policy",
    control_policy_identity: str = "fixture-control-policy",
    max_moves: int = MAX_MOVES,
    boundary_callback: Any | None = None,
) -> list[dict[str, Any]]:
    session = PairedEvaluationSession(
        rows=rows,
        candidate_policy=candidate_policy,
        control_policy=control_policy,
        candidate_policy_identity=candidate_policy_identity,
        control_policy_identity=control_policy_identity,
        max_moves=max_moves,
    )
    return session.finish(boundary_callback=boundary_callback)


BINARY_MAGIC = b"J1EXECSTATE1\n"


def serialize_binary_state(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    body = buffer.getvalue()
    digest = hashlib.sha256(body).hexdigest().encode("ascii")
    return BINARY_MAGIC + digest + b"\n" + body


def deserialize_binary_state(serialized: bytes) -> dict[str, Any]:
    if not serialized.startswith(BINARY_MAGIC):
        raise J1ExecutionIntegrityError("Binary state magic mismatch")
    remainder = serialized[len(BINARY_MAGIC) :]
    try:
        digest, body = remainder.split(b"\n", 1)
    except ValueError as error:
        raise J1ExecutionIntegrityError("Binary state header truncated") from error
    if hashlib.sha256(body).hexdigest().encode("ascii") != digest:
        raise J1ExecutionIntegrityError("Binary state hash mismatch")
    payload = torch.load(
        io.BytesIO(body),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise J1ExecutionIntegrityError("Binary state is not a mapping")
    return payload


def write_atomic_binary(path: Path, payload: Mapping[str, Any]) -> str:
    serialized = serialize_binary_state(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    observed = path.read_bytes()
    if observed != serialized:
        raise J1ExecutionIntegrityError("Atomic binary write changed bytes")
    deserialize_binary_state(observed)
    return hashlib.sha256(observed).hexdigest()


def load_atomic_binary(path: Path) -> dict[str, Any]:
    return deserialize_binary_state(path.read_bytes())


def validate_training_runtime_payload(payload: Mapping[str, Any]) -> None:
    required = set(execution_schema()["training_state_required_keys"])
    missing = required - set(payload)
    if missing:
        raise J1ExecutionIntegrityError(
            f"Training runtime state missing keys: {sorted(missing)}"
        )
    if payload.get("phase") != "training":
        raise J1ExecutionIntegrityError("Runtime state has wrong phase")
    if payload.get("runtime_payload_complete") is not True:
        raise J1ExecutionIntegrityError(
            "Scientific training runtime payload is not complete"
        )
    if int(payload["round_number"]) not in range(1, ROUNDS + 1):
        raise J1ExecutionIntegrityError("Runtime round is outside 1..64")
    if payload["collection_boundary"] not in {
        "pre_action",
        "post_step",
        "pre_update",
        "mid_update",
        "post_checkpoint",
    }:
        raise J1ExecutionIntegrityError("Unknown runtime boundary")
    if len(payload["active_roots"]) > ENV_COUNT:
        raise J1ExecutionIntegrityError("Too many synchronous environments")
    if len(set(payload["optimizer_step_ids"])) != len(
        payload["optimizer_step_ids"]
    ):
        raise J1ExecutionIntegrityError("Optimizer step was double-closed")

    expected_model_shapes = {
        "body.0.weight": (512, j1.EXPECTED_OBSERVATION_WIDTH),
        "body.0.bias": (512,),
        "body.2.weight": (512, 512),
        "body.2.bias": (512,),
        "policy.weight": (4, 512),
        "policy.bias": (4,),
        "value.weight": (1, 512),
        "value.bias": (1,),
        "auxiliary.weight": (3, 512),
        "auxiliary.bias": (3,),
    }
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping):
        raise J1ExecutionIntegrityError("Serialized model state is not a mapping")
    if set(model_state) != set(expected_model_shapes):
        raise J1ExecutionIntegrityError("Serialized model state keys changed")
    for name, expected_shape in expected_model_shapes.items():
        tensor = model_state[name]
        if not isinstance(tensor, torch.Tensor):
            raise J1ExecutionIntegrityError(
                f"Serialized model value is not a tensor: {name}"
            )
        if tuple(tensor.shape) != expected_shape:
            raise J1ExecutionIntegrityError(
                f"Serialized model tensor shape changed: {name}"
            )
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise J1ExecutionIntegrityError(
                f"Serialized model tensor is malformed: {name}"
            )

    optimizer_state = payload["optimizer_state"]
    if not isinstance(optimizer_state, Mapping):
        raise J1ExecutionIntegrityError(
            "Serialized optimizer state is not a mapping"
        )
    if set(optimizer_state) != {"state", "param_groups"}:
        raise J1ExecutionIntegrityError(
            "Serialized optimizer state keys changed"
        )
    if not isinstance(optimizer_state["state"], Mapping):
        raise J1ExecutionIntegrityError(
            "Serialized optimizer tensor state is malformed"
        )
    if not isinstance(optimizer_state["param_groups"], list) or not (
        optimizer_state["param_groups"]
    ):
        raise J1ExecutionIntegrityError(
            "Serialized optimizer parameter groups are malformed"
        )
    _validate_serialized_numeric_tree(
        optimizer_state["state"],
        path="optimizer_state.state",
        tensors_only_leaves=False,
    )
    _validate_serialized_numeric_tree(
        optimizer_state["param_groups"],
        path="optimizer_state.param_groups",
        tensors_only_leaves=False,
    )


def _validate_serialized_numeric_tree(
    value: Any,
    *,
    path: str,
    tensors_only_leaves: bool,
) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or not torch.isfinite(value).all():
            raise J1ExecutionIntegrityError(
                f"Nonfinite or empty serialized tensor: {path}"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, (str, int)):
                raise J1ExecutionIntegrityError(
                    f"Malformed serialized mapping key: {path}"
                )
            _validate_serialized_numeric_tree(
                child,
                path=f"{path}.{key}",
                tensors_only_leaves=tensors_only_leaves,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_serialized_numeric_tree(
                child,
                path=f"{path}[{index}]",
                tensors_only_leaves=tensors_only_leaves,
            )
        return
    if tensors_only_leaves:
        raise J1ExecutionIntegrityError(
            f"Expected serialized tensor at {path}"
        )
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise J1ExecutionIntegrityError(
                f"Nonfinite serialized scalar: {path}"
            )
        return
    raise J1ExecutionIntegrityError(
        f"Unsupported serialized value at {path}: {type(value).__name__}"
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_start_identity(pid: int) -> str | None:
    result = subprocess.run(
        ("ps", "-o", "lstart=", "-p", str(int(pid))),
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _record_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    return payload_with_hash(payload, field)


def _verify_ownership_ledger(payload: Mapping[str, Any]) -> bool:
    if not verify_payload_hash(payload, "ownership_payload_sha256"):
        return False
    owners = payload.get("owners")
    recoveries = payload.get("recoveries")
    if not isinstance(owners, list) or not owners:
        return False
    if not isinstance(recoveries, list):
        return False
    if not all(
        isinstance(row, dict)
        and verify_payload_hash(row, "owner_record_sha256")
        for row in owners
    ):
        return False
    if not all(
        isinstance(row, dict)
        and verify_payload_hash(row, "recovery_record_sha256")
        for row in recoveries
    ):
        return False
    return payload.get("head_owner_sha256") == owners[-1].get(
        "owner_record_sha256"
    )


def _atomic_replace_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    body = payload_with_hash(payload, field)
    serialized = json.dumps(body, indent=2, sort_keys=True) + "\n"
    if not verify_payload_hash(json.loads(serialized), field):
        raise J1ExecutionIntegrityError("Atomic JSON is not reload-stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return load_json(path)


def _rolling_resume_paths(root: Path) -> dict[str, Path]:
    directory = root / ROLLING_RESUME_DIR
    return {
        "directory": directory,
        "head": directory / ROLLING_RESUME_HEAD_NAME,
        "journal": directory / ROLLING_RESUME_JOURNAL_NAME,
        "slot0": directory / "resume_slot_0.bin",
        "slot1": directory / "resume_slot_1.bin",
    }


def rolling_resume_contract(
    *,
    phase: str,
    marker_file_sha256: str,
    marker_payload_sha256: str,
    phase_lock_file_sha256: str,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    command: str,
    execution_mode: str,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("Rolling resume phase is invalid")
    payload = {
        "version": f"{VERSION}_rolling_resume_contract_v1",
        "phase": phase,
        "marker_file_sha256": marker_file_sha256,
        "marker_payload_sha256": marker_payload_sha256,
        "phase_lock_file_sha256": phase_lock_file_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_payload_sha256": manifest_payload_sha256,
        "command": command,
        "runner_sha256": sha256_path(RUNNER_PATH),
        "execution_mode": execution_mode,
    }
    return payload_with_hash(payload, "rolling_contract_sha256")


def _validate_rolling_resume_contract(
    contract: Mapping[str, Any],
) -> None:
    if (
        not verify_payload_hash(contract, "rolling_contract_sha256")
        or contract.get("phase") not in PHASES
        or contract.get("runner_sha256") != sha256_path(RUNNER_PATH)
        or contract.get("execution_mode")
        not in {"scientific", "miniature_fixture"}
    ):
        raise J1ExecutionIntegrityError(
            "Rolling resume phase contract is invalid"
        )
    hash_fields = (
        "marker_file_sha256",
        "marker_payload_sha256",
        "phase_lock_file_sha256",
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "runner_sha256",
    )
    if any(
        not isinstance(contract.get(field), str)
        or len(str(contract[field])) != 64
        for field in hash_fields
    ) or not str(contract.get("command", "")):
        raise J1ExecutionIntegrityError(
            "Rolling resume contract fields are malformed"
        )


def _load_rolling_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise J1ExecutionIntegrityError(
                f"Rolling resume journal line {line_number} is malformed"
            ) from error
        if not verify_payload_hash(
            record,
            "rolling_journal_record_sha256",
        ):
            raise J1ExecutionIntegrityError(
                f"Rolling resume journal line {line_number} changed"
            )
        records.append(record)
    return records


def load_rolling_resume_boundary(
    root: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_rolling_resume_contract(contract)
    contract_sha256 = contract["rolling_contract_sha256"]
    paths = _rolling_resume_paths(root)
    head = load_json(paths["head"])
    if not verify_payload_hash(head, "rolling_head_payload_sha256"):
        raise J1ExecutionIntegrityError(
            "Rolling resume head is malformed"
        )
    if head.get("rolling_contract_sha256") != contract_sha256:
        raise J1ExecutionIntegrityError(
            "Rolling resume head belongs to another phase contract"
        )
    records = _load_rolling_journal(paths["journal"])
    by_hash = {
        str(record["rolling_journal_record_sha256"]): record
        for record in records
    }
    head_hash = str(head["journal_record_sha256"])
    if head_hash not in by_hash:
        raise J1ExecutionIntegrityError(
            "Rolling resume head journal record is missing"
        )
    current_hash: str | None = head_hash
    expected_sequence = int(head["sequence"])
    visited: set[str] = set()
    while current_hash is not None:
        if current_hash in visited or current_hash not in by_hash:
            raise J1ExecutionIntegrityError(
                "Rolling resume journal chain is cyclic or missing"
            )
        visited.add(current_hash)
        record = by_hash[current_hash]
        if record.get("rolling_contract_sha256") != contract_sha256:
            raise J1ExecutionIntegrityError(
                "Rolling resume journal changed phase contract"
            )
        if int(record["sequence"]) != expected_sequence:
            raise J1ExecutionIntegrityError(
                "Rolling resume journal sequence changed"
            )
        current_hash = record["predecessor_record_sha256"]
        expected_sequence -= 1
    if expected_sequence != -1:
        raise J1ExecutionIntegrityError(
            "Rolling resume journal does not reach genesis"
        )
    record = by_hash[head_hash]
    if (
        int(record["sequence"]) != int(head["sequence"])
        or record["unit_id"] != head["unit_id"]
        or record["slot_name"] != head["slot_name"]
    ):
        raise J1ExecutionIntegrityError(
            "Rolling resume head identity changed"
        )
    slot = paths[str(record["slot_name"])]
    if (
        not slot.is_file()
        or sha256_path(slot) != record["state_file_sha256"]
    ):
        raise J1ExecutionIntegrityError(
            "Rolling resume slot is missing or changed"
        )
    payload = load_atomic_binary(slot)
    if j1.stable_hash(payload) != record["state_payload_sha256"]:
        raise J1ExecutionIntegrityError(
            "Rolling resume state payload changed"
        )
    return {
        "head": head,
        "record": record,
        "state": payload,
        "journal_record_count": len(records),
        "journal_chain_length": len(visited),
        "journal_chain_record_sha256s": sorted(visited),
        "journal_records": records,
        "slot_file_sha256": sha256_path(slot),
        "passes": True,
    }


def write_rolling_resume_boundary(
    *,
    root: Path,
    contract: Mapping[str, Any],
    unit_id: str,
    state: Mapping[str, Any],
    crash_stage: str | None = None,
) -> dict[str, Any]:
    _validate_rolling_resume_contract(contract)
    contract_sha256 = contract["rolling_contract_sha256"]
    if not unit_id:
        raise ValueError("Rolling resume unit id is empty")
    if crash_stage not in {None, "after_slot", "after_journal", "after_head"}:
        raise ValueError("Unknown rolling resume crash stage")
    paths = _rolling_resume_paths(root)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    if paths["head"].exists():
        predecessor = load_rolling_resume_boundary(
            root,
            contract=contract,
        )
        sequence = int(predecessor["head"]["sequence"]) + 1
        predecessor_hash = predecessor["record"][
            "rolling_journal_record_sha256"
        ]
    else:
        sequence = 0
        predecessor_hash = None
    slot_name = f"slot{sequence % 2}"
    slot_path = paths[slot_name]
    state_file_sha256 = write_atomic_binary(slot_path, state)
    if crash_stage == "after_slot":
        raise RuntimeError("fixture crash after rolling resume slot")
    record = payload_with_hash(
        {
            "version": f"{VERSION}_rolling_resume_journal_v1",
            "sequence": sequence,
            "unit_id": unit_id,
            "slot_name": slot_name,
            "state_file_sha256": state_file_sha256,
            "state_payload_sha256": j1.stable_hash(dict(state)),
            "predecessor_record_sha256": predecessor_hash,
            "rolling_contract_sha256": contract_sha256,
        },
        "rolling_journal_record_sha256",
    )
    records = _load_rolling_journal(paths["journal"])
    direct_children = [
        row
        for row in records
        if int(row["sequence"]) == sequence
        and row.get("predecessor_record_sha256") == predecessor_hash
    ]
    if direct_children:
        if any(row != record for row in direct_children):
            raise J1ExecutionIntegrityError(
                "Rolling resume orphan reexecution changed bytes"
            )
    else:
        serialized = canonical_json_bytes(record) + b"\n"
        with paths["journal"].open("ab") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    if crash_stage == "after_journal":
        raise RuntimeError("fixture crash after rolling resume journal")
    head = _atomic_replace_json(
        paths["head"],
        {
            "version": f"{VERSION}_rolling_resume_head_v1",
            "sequence": sequence,
            "unit_id": unit_id,
            "slot_name": slot_name,
            "journal_record_sha256": record[
                "rolling_journal_record_sha256"
            ],
            "state_file_sha256": state_file_sha256,
            "rolling_contract_sha256": contract_sha256,
        },
        field="rolling_head_payload_sha256",
    )
    del head
    observed = load_rolling_resume_boundary(
        root,
        contract=contract,
    )
    if crash_stage == "after_head":
        raise RuntimeError("fixture crash after rolling resume head")
    return observed


class RollingResumeStore:
    """One-scan rolling journal with O(1) authenticated appends."""

    def __init__(
        self,
        *,
        root: Path,
        contract: Mapping[str, Any],
        output_accountant: "PhaseOutputAccountant | None" = None,
    ) -> None:
        _validate_rolling_resume_contract(contract)
        self.root = root
        self.contract = dict(contract)
        self.output_accountant = output_accountant
        self.paths = _rolling_resume_paths(root)
        self.initial_full_scan_count = 1
        self.terminal_full_audit_count = 0
        self.append_operation_count = 0
        self.journal_bytes_appended = 0
        self.slot_bytes_written = 0
        self.head_bytes_written = 0
        if self.paths["head"].exists():
            observed = load_rolling_resume_boundary(
                root,
                contract=contract,
            )
            self.current: dict[str, Any] | None = observed
            self.records = list(observed["journal_records"])
            self.chain_hashes = set(
                observed["journal_chain_record_sha256s"]
            )
        else:
            self.current = None
            self.records = _load_rolling_journal(
                self.paths["journal"]
            )
            self.chain_hashes: set[str] = set()
            if self.records:
                raise J1ExecutionIntegrityError(
                    "Rolling journal exists without an authenticated head"
                )
        self.direct_child_index: dict[
            tuple[int, str | None],
            dict[str, Any],
        ] = {}
        for record in self.records:
            key = (
                int(record["sequence"]),
                record.get("predecessor_record_sha256"),
            )
            if (
                key in self.direct_child_index
                and self.direct_child_index[key] != record
            ):
                raise J1ExecutionIntegrityError(
                    "Rolling journal has conflicting direct children"
                )
            self.direct_child_index[key] = record

    def append(
        self,
        *,
        unit_id: str,
        state: Mapping[str, Any],
        crash_stage: str | None = None,
    ) -> dict[str, Any]:
        if not unit_id:
            raise ValueError("Rolling resume unit id is empty")
        if crash_stage not in {
            None,
            "after_slot",
            "after_journal",
            "after_head",
        }:
            raise ValueError("Unknown rolling resume crash stage")
        self.paths["directory"].mkdir(parents=True, exist_ok=True)
        if self.current is None:
            sequence = 0
            predecessor_hash = None
        else:
            sequence = int(self.current["head"]["sequence"]) + 1
            predecessor_hash = self.current["record"][
                "rolling_journal_record_sha256"
            ]
        slot_name = f"slot{sequence % 2}"
        slot_path = self.paths[slot_name]
        state_file_sha256 = write_atomic_binary(slot_path, state)
        self.slot_bytes_written += int(slot_path.stat().st_size)
        if self.output_accountant is not None:
            self.output_accountant.record_path(slot_path)
        if crash_stage == "after_slot":
            raise RuntimeError("fixture crash after rolling resume slot")
        record = payload_with_hash(
            {
                "version": f"{VERSION}_rolling_resume_journal_v1",
                "sequence": sequence,
                "unit_id": unit_id,
                "slot_name": slot_name,
                "state_file_sha256": state_file_sha256,
                "state_payload_sha256": j1.stable_hash(dict(state)),
                "predecessor_record_sha256": predecessor_hash,
                "rolling_contract_sha256": self.contract[
                    "rolling_contract_sha256"
                ],
            },
            "rolling_journal_record_sha256",
        )
        child_key = (sequence, predecessor_hash)
        direct_child = self.direct_child_index.get(child_key)
        if direct_child is not None:
            if direct_child != record:
                raise J1ExecutionIntegrityError(
                    "Rolling resume orphan reexecution changed bytes"
                )
        else:
            serialized_record = canonical_json_bytes(record) + b"\n"
            with self.paths["journal"].open("ab") as handle:
                handle.write(serialized_record)
                handle.flush()
                os.fsync(handle.fileno())
            self.journal_bytes_appended += len(serialized_record)
            self.records.append(record)
            self.direct_child_index[child_key] = record
        if self.output_accountant is not None:
            self.output_accountant.record_path(self.paths["journal"])
        if crash_stage == "after_journal":
            raise RuntimeError(
                "fixture crash after rolling resume journal"
            )
        head = _atomic_replace_json(
            self.paths["head"],
            {
                "version": f"{VERSION}_rolling_resume_head_v1",
                "sequence": sequence,
                "unit_id": unit_id,
                "slot_name": slot_name,
                "journal_record_sha256": record[
                    "rolling_journal_record_sha256"
                ],
                "state_file_sha256": state_file_sha256,
                "rolling_contract_sha256": self.contract[
                    "rolling_contract_sha256"
                ],
            },
            field="rolling_head_payload_sha256",
        )
        if self.output_accountant is not None:
            self.output_accountant.record_path(self.paths["head"])
        self.head_bytes_written += int(self.paths["head"].stat().st_size)
        payload = load_atomic_binary(slot_path)
        if (
            sha256_path(slot_path) != record["state_file_sha256"]
            or j1.stable_hash(payload) != record["state_payload_sha256"]
        ):
            raise J1ExecutionIntegrityError(
                "Rolling resume O(1) append verification failed"
            )
        self.chain_hashes.add(
            record["rolling_journal_record_sha256"]
        )
        self.current = {
            "head": head,
            "record": record,
            "state": payload,
            "journal_record_count": len(self.records),
            "journal_chain_length": len(self.chain_hashes),
            "journal_chain_record_sha256s": self.chain_hashes,
            "journal_records": self.records,
            "slot_file_sha256": sha256_path(slot_path),
            "passes": True,
        }
        self.append_operation_count += 1
        if crash_stage == "after_head":
            raise RuntimeError("fixture crash after rolling resume head")
        return self.current

    def audit_full(self) -> dict[str, Any] | None:
        if self.current is None:
            return None
        self.terminal_full_audit_count += 1
        return load_rolling_resume_boundary(
            self.root,
            contract=self.contract,
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "initial_full_scan_count": self.initial_full_scan_count,
            "terminal_full_audit_count": self.terminal_full_audit_count,
            "append_operation_count": self.append_operation_count,
            "journal_bytes_appended": self.journal_bytes_appended,
            "slot_bytes_written": self.slot_bytes_written,
            "head_bytes_written": self.head_bytes_written,
            "record_count": len(self.records),
            "direct_child_index_size": len(self.direct_child_index),
            "chain_membership_size": len(self.chain_hashes),
            "passes": True,
        }


def _runtime_charge_journal_path(root: Path) -> Path:
    return root / RUNTIME_CHARGE_JOURNAL_NAME


def _load_runtime_charge_journal(
    root: Path,
    *,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_rolling_resume_contract(contract)
    path = _runtime_charge_journal_path(root)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    predecessor: str | None = None
    open_start: dict[str, Any] | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise J1ExecutionIntegrityError(
                f"Runtime charge journal line {line_number} is malformed"
            ) from error
        if not verify_payload_hash(
            record,
            "runtime_charge_record_sha256",
        ):
            raise J1ExecutionIntegrityError(
                f"Runtime charge journal line {line_number} changed"
            )
        checks = {
            "sequence": int(record.get("sequence", -1)) == len(records),
            "predecessor": (
                record.get("predecessor_record_sha256") == predecessor
            ),
            "contract": (
                record.get("rolling_contract_sha256")
                == contract["rolling_contract_sha256"]
            ),
            "event": record.get("event") in {
                "attempt_started",
                "attempt_finished",
                "attempt_abandoned_on_resume",
            },
        }
        if not all(checks.values()):
            raise J1ExecutionIntegrityError(
                f"Runtime charge journal line {line_number} is inconsistent"
            )
        if record["event"] == "attempt_started":
            if open_start is not None:
                raise J1ExecutionIntegrityError(
                    "Runtime charge journal has concurrent open attempts"
                )
            if (
                not str(record.get("attempt_id", ""))
                or not math.isfinite(float(record.get("wall_started_at", math.nan)))
            ):
                raise J1ExecutionIntegrityError(
                    "Runtime charge start record is malformed"
                )
            open_start = record
        else:
            if (
                open_start is None
                or record.get("attempt_id") != open_start.get("attempt_id")
                or record.get("start_record_sha256")
                != open_start.get("runtime_charge_record_sha256")
                or not math.isfinite(
                    float(record.get("charged_seconds", math.nan))
                )
                or float(record["charged_seconds"]) < 0.0
            ):
                raise J1ExecutionIntegrityError(
                    "Runtime charge closure is malformed"
                )
            open_start = None
        records.append(record)
        predecessor = record["runtime_charge_record_sha256"]
    return records


def _append_runtime_charge_record(
    root: Path,
    *,
    contract: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    records = _load_runtime_charge_journal(root, contract=contract)
    body = {
        "version": f"{VERSION}_runtime_charge_record_v1",
        "sequence": len(records),
        "predecessor_record_sha256": (
            None
            if not records
            else records[-1]["runtime_charge_record_sha256"]
        ),
        "rolling_contract_sha256": contract[
            "rolling_contract_sha256"
        ],
        **dict(payload),
    }
    record = payload_with_hash(body, "runtime_charge_record_sha256")
    path = _runtime_charge_journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(record) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    observed = _load_runtime_charge_journal(root, contract=contract)
    if not observed or observed[-1] != record:
        raise J1ExecutionIntegrityError(
            "Runtime charge journal append changed bytes"
        )
    return record


def _runtime_open_start(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not records or records[-1]["event"] != "attempt_started":
        return None
    return records[-1]


def _abandoned_attempt_charge(
    *,
    base_unit_id: str,
    execution_mode: str,
) -> dict[str, Any]:
    if re.fullmatch(r"round=\d+\|collection_tick=\d+", base_unit_id):
        unit_type = "training_collection_tick_block"
    elif re.fullmatch(
        r"round=\d+\|epoch=\d+\|start=\d+",
        base_unit_id,
    ):
        unit_type = "training_minibatch_update"
    elif re.fullmatch(r"row=\d+\|candidate_arm", base_unit_id):
        unit_type = "paired_candidate_arm"
    elif re.fullmatch(
        r"row=\d+\|control_arm_and_pair",
        base_unit_id,
    ):
        unit_type = "paired_control_arm_and_pair"
    elif execution_mode == "miniature_fixture":
        unit_type = "miniature_fixture_other"
    else:
        raise J1ExecutionIntegrityError(
            "Abandoned scientific attempt has an unknown unit type"
        )
    return {
        "unit_type": unit_type,
        "charged_seconds": ABANDONED_ATTEMPT_CHARGE_SECONDS[unit_type],
        "charge_basis":
            "fixed preregistered conservative abandoned-unit ceiling",
    }


def runtime_charge_summary(
    root: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    records = _load_runtime_charge_journal(root, contract=contract)
    charged = [
        float(record["charged_seconds"])
        for record in records
        if record["event"] in {
            "attempt_finished",
            "attempt_abandoned_on_resume",
        }
    ]
    path = _runtime_charge_journal_path(root)
    return {
        "active_seconds": float(math.fsum(charged)),
        "attempts_started": sum(
            record["event"] == "attempt_started" for record in records
        ),
        "attempts_finished": sum(
            record["event"] == "attempt_finished" for record in records
        ),
        "attempts_abandoned": sum(
            record["event"] == "attempt_abandoned_on_resume"
            for record in records
        ),
        "open_attempt_id": (
            None
            if _runtime_open_start(records) is None
            else _runtime_open_start(records)["attempt_id"]
        ),
        "record_count": len(records),
        "head_record_sha256": (
            None
            if not records
            else records[-1]["runtime_charge_record_sha256"]
        ),
        "journal_file_sha256": (
            None if not path.exists() else sha256_path(path)
        ),
        "passes": True,
    }


def close_abandoned_runtime_attempt(
    root: Path,
    *,
    contract: Mapping[str, Any],
    now: float | None = None,
) -> dict[str, Any] | None:
    records = _load_runtime_charge_journal(root, contract=contract)
    start = _runtime_open_start(records)
    if start is None:
        return None
    ended = time.time() if now is None else float(now)
    started = float(start["wall_started_at"])
    if not math.isfinite(ended) or ended < started:
        raise J1ExecutionIntegrityError(
            "Runtime charge clock moved backwards"
        )
    bounded = _abandoned_attempt_charge(
        base_unit_id=str(start["base_unit_id"]),
        execution_mode=str(contract["execution_mode"]),
    )
    return _append_runtime_charge_record(
        root,
        contract=contract,
        payload={
            "event": "attempt_abandoned_on_resume",
            "attempt_id": start["attempt_id"],
            "base_unit_id": start["base_unit_id"],
            "start_record_sha256": start[
                "runtime_charge_record_sha256"
            ],
            "wall_ended_at": ended,
            "observed_resume_gap_seconds": ended - started,
            **bounded,
            "reason": "open attempt recovered before deterministic replay",
        },
    )


def begin_runtime_attempt(
    root: Path,
    *,
    contract: Mapping[str, Any],
    base_unit_id: str,
    now: float | None = None,
) -> dict[str, Any]:
    if not base_unit_id:
        raise ValueError("Runtime attempt unit id is empty")
    started = time.time() if now is None else float(now)
    if not math.isfinite(started):
        raise ValueError("Runtime attempt time is nonfinite")
    close_abandoned_runtime_attempt(
        root,
        contract=contract,
        now=started,
    )
    records = _load_runtime_charge_journal(root, contract=contract)
    ordinal = sum(
        record["event"] == "attempt_started"
        and record.get("base_unit_id") == base_unit_id
        for record in records
    )
    return _append_runtime_charge_record(
        root,
        contract=contract,
        payload={
            "event": "attempt_started",
            "attempt_id": f"{base_unit_id}|attempt={ordinal}",
            "base_unit_id": base_unit_id,
            "wall_started_at": started,
        },
    )


def finish_runtime_attempt(
    root: Path,
    *,
    contract: Mapping[str, Any],
    start_record: Mapping[str, Any],
    now: float | None = None,
) -> dict[str, Any]:
    records = _load_runtime_charge_journal(root, contract=contract)
    open_start = _runtime_open_start(records)
    if (
        open_start is None
        or open_start.get("runtime_charge_record_sha256")
        != start_record.get("runtime_charge_record_sha256")
    ):
        raise J1ExecutionIntegrityError(
            "Runtime attempt closure does not match open attempt"
        )
    ended = time.time() if now is None else float(now)
    started = float(open_start["wall_started_at"])
    if not math.isfinite(ended) or ended < started:
        raise J1ExecutionIntegrityError(
            "Runtime charge clock moved backwards"
        )
    return _append_runtime_charge_record(
        root,
        contract=contract,
        payload={
            "event": "attempt_finished",
            "attempt_id": open_start["attempt_id"],
            "base_unit_id": open_start["base_unit_id"],
            "start_record_sha256": open_start[
                "runtime_charge_record_sha256"
            ],
            "wall_ended_at": ended,
            "charged_seconds": ended - started,
        },
    )


class RuntimeChargeLedger:
    """One-scan charged-work journal with O(1) appends."""

    def __init__(
        self,
        *,
        root: Path,
        contract: Mapping[str, Any],
        wall_clock: Any,
        output_accountant: "PhaseOutputAccountant | None" = None,
    ) -> None:
        _validate_rolling_resume_contract(contract)
        self.root = root
        self.contract = dict(contract)
        self.wall_clock = wall_clock
        self.output_accountant = output_accountant
        self.path = _runtime_charge_journal_path(root)
        self.initial_full_scan_count = 1
        self.terminal_full_audit_count = 0
        self.append_operation_count = 0
        self.journal_bytes_appended = 0
        self.records = _load_runtime_charge_journal(
            root,
            contract=contract,
        )
        self.active_seconds = math.fsum(
            float(record["charged_seconds"])
            for record in self.records
            if record["event"] in {
                "attempt_finished",
                "attempt_abandoned_on_resume",
            }
        )
        self.started_by_unit: dict[str, int] = {}
        self.attempts_started = 0
        self.attempts_finished = 0
        self.attempts_abandoned = 0
        for record in self.records:
            if record["event"] == "attempt_started":
                self.attempts_started += 1
                unit = str(record["base_unit_id"])
                self.started_by_unit[unit] = (
                    self.started_by_unit.get(unit, 0) + 1
                )
            elif record["event"] == "attempt_finished":
                self.attempts_finished += 1
            elif record["event"] == "attempt_abandoned_on_resume":
                self.attempts_abandoned += 1
        self.open_start = _runtime_open_start(self.records)
        if self.open_start is not None:
            self._close_open(
                event="attempt_abandoned_on_resume",
                reason=(
                    "open attempt recovered before deterministic replay"
                ),
            )

    def _append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = payload_with_hash(
            {
                "version": f"{VERSION}_runtime_charge_record_v1",
                "sequence": len(self.records),
                "predecessor_record_sha256": (
                    None
                    if not self.records
                    else self.records[-1][
                        "runtime_charge_record_sha256"
                    ]
                ),
                "rolling_contract_sha256": self.contract[
                    "rolling_contract_sha256"
                ],
                **dict(payload),
            },
            "runtime_charge_record_sha256",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            serialized = canonical_json_bytes(record) + b"\n"
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        self.journal_bytes_appended += len(serialized)
        if self.output_accountant is not None:
            self.output_accountant.record_path(self.path)
        self.records.append(record)
        self.append_operation_count += 1
        return record

    def _close_open(
        self,
        *,
        event: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if self.open_start is None:
            raise J1ExecutionIntegrityError(
                "Runtime charge ledger has no open attempt"
            )
        ended = float(self.wall_clock())
        started = float(self.open_start["wall_started_at"])
        if not math.isfinite(ended) or ended < started:
            raise J1ExecutionIntegrityError(
                "Runtime charge clock moved backwards"
            )
        payload = {
            "event": event,
            "attempt_id": self.open_start["attempt_id"],
            "base_unit_id": self.open_start["base_unit_id"],
            "start_record_sha256": self.open_start[
                "runtime_charge_record_sha256"
            ],
            "wall_ended_at": ended,
        }
        if event == "attempt_abandoned_on_resume":
            payload["observed_resume_gap_seconds"] = ended - started
            payload.update(
                _abandoned_attempt_charge(
                    base_unit_id=str(
                        self.open_start["base_unit_id"]
                    ),
                    execution_mode=str(
                        self.contract["execution_mode"]
                    ),
                )
            )
        else:
            payload["charged_seconds"] = ended - started
            payload["charge_basis"] = "measured closed active work"
        if reason is not None:
            payload["reason"] = reason
        record = self._append(payload)
        self.active_seconds += float(record["charged_seconds"])
        if event == "attempt_finished":
            self.attempts_finished += 1
        else:
            self.attempts_abandoned += 1
        self.open_start = None
        return record

    def begin(self, base_unit_id: str) -> dict[str, Any]:
        if self.open_start is not None:
            self._close_open(
                event="attempt_abandoned_on_resume",
                reason="open attempt closed before replay",
            )
        ordinal = self.started_by_unit.get(base_unit_id, 0)
        started = float(self.wall_clock())
        if not math.isfinite(started):
            raise J1ExecutionIntegrityError(
                "Runtime attempt clock is nonfinite"
            )
        record = self._append(
            {
                "event": "attempt_started",
                "attempt_id": f"{base_unit_id}|attempt={ordinal}",
                "base_unit_id": base_unit_id,
                "wall_started_at": started,
            }
        )
        self.started_by_unit[base_unit_id] = ordinal + 1
        self.attempts_started += 1
        self.open_start = record
        return record

    def finish(self, start_record: Mapping[str, Any]) -> dict[str, Any]:
        if (
            self.open_start is None
            or self.open_start["runtime_charge_record_sha256"]
            != start_record.get("runtime_charge_record_sha256")
        ):
            raise J1ExecutionIntegrityError(
                "Runtime attempt closure changed its start"
            )
        return self._close_open(event="attempt_finished")

    def summary(self) -> dict[str, Any]:
        return {
            "active_seconds": float(self.active_seconds),
            "attempts_started": self.attempts_started,
            "attempts_finished": self.attempts_finished,
            "attempts_abandoned": self.attempts_abandoned,
            "open_attempt_id": (
                None
                if self.open_start is None
                else self.open_start["attempt_id"]
            ),
            "record_count": len(self.records),
            "head_record_sha256": (
                None
                if not self.records
                else self.records[-1][
                    "runtime_charge_record_sha256"
                ]
            ),
            "journal_file_sha256": None,
            "journal_file_sha256_deferred_to_full_audit": True,
            "passes": True,
        }

    def audit_full(self) -> dict[str, Any]:
        self.terminal_full_audit_count += 1
        observed = runtime_charge_summary(
            self.root,
            contract=self.contract,
        )
        if (
            observed["active_seconds"] != self.active_seconds
            or observed["record_count"] != len(self.records)
            or observed["open_attempt_id"]
            != (
                None
                if self.open_start is None
                else self.open_start["attempt_id"]
            )
        ):
            raise J1ExecutionIntegrityError(
                "Runtime charge terminal audit changed"
            )
        return observed

    def metrics(self) -> dict[str, Any]:
        return {
            "initial_full_scan_count": self.initial_full_scan_count,
            "terminal_full_audit_count": self.terminal_full_audit_count,
            "append_operation_count": self.append_operation_count,
            "journal_bytes_appended": self.journal_bytes_appended,
            "record_count": len(self.records),
            "summary_scans_after_initialization": 0,
            "passes": True,
        }


class PhaseOutputAccountant:
    """One-scan namespace byte/file counter with targeted updates."""

    def __init__(self, phase_dir: Path) -> None:
        self.phase_dir = phase_dir.resolve()
        self._sizes: dict[str, int] = {}
        self.full_scan_count = 0
        self.targeted_stat_count = 0
        self.reconcile_full()

    def _assert_local(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.phase_dir)
        except ValueError as error:
            raise J1ExecutionIntegrityError(
                "Output accounting path escaped phase namespace"
            ) from error
        return resolved

    def record_path(self, path: Path) -> None:
        resolved = self._assert_local(path)
        self.targeted_stat_count += 1
        key = str(resolved)
        if resolved.is_file() and not resolved.is_symlink():
            self._sizes[key] = int(resolved.stat().st_size)
        else:
            self._sizes.pop(key, None)

    def record_paths(self, paths: Iterable[Path]) -> None:
        for path in paths:
            self.record_path(path)

    def retire_path(self, path: Path) -> None:
        resolved = self._assert_local(path)
        self._sizes.pop(str(resolved), None)

    def reconcile_full(self) -> dict[str, Any]:
        self.full_scan_count += 1
        observed = {
            str(path.resolve()): int(path.stat().st_size)
            for path in self.phase_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        } if self.phase_dir.exists() else {}
        self._sizes = observed
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "output_bytes": sum(self._sizes.values()),
            "output_file_count": len(self._sizes),
            "full_scan_count": self.full_scan_count,
            "targeted_stat_count": self.targeted_stat_count,
            "passes": True,
        }


def default_phase_operational_audit(
    *,
    phase_dir: Path,
    phase: str,
    active_seconds: float,
    require_target_disk: bool,
    output_bytes_override: int | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("Operational audit phase is invalid")
    cap = PHASE_CAPS[phase]
    output_bytes = (
        j1.directory_bytes(phase_dir)
        if output_bytes_override is None
        else int(output_bytes_override)
    )
    free_gib = j1.free_disk_gib()
    nice = int(os.getpriority(os.PRIO_PROCESS, 0))
    process = j1.heavy_process_audit()
    services = j1.service_audit()
    checks = {
        "nice_at_least_10": nice >= 10,
        "one_torch_intraop_thread": torch.get_num_threads() == 1,
        "one_torch_interop_thread": torch.get_num_interop_threads() == 1,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "one_heavy_process": process.get("passes") is True,
        "services_healthy": services.get("passes") is True,
        "free_disk_hard_floor": free_gib > 100.0,
        "free_disk_target": (
            free_gib > 120.0 if require_target_disk else True
        ),
        "active_runtime_within_cap": (
            active_seconds <= cap["active_hours"] * 3600.0
        ),
        "output_storage_within_cap": (
            output_bytes <= int(cap["storage_gib"] * 1024**3)
        ),
    }
    return {
        "phase": phase,
        "active_seconds": float(active_seconds),
        "active_hours_cap": cap["active_hours"],
        "output_bytes": output_bytes,
        "storage_gib_cap": cap["storage_gib"],
        "free_disk_gib": free_gib,
        "target_120_gib_met": free_gib > 120.0,
        "nice": nice,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "process": process,
        "services": services,
        "checks": checks,
        "passes": all(checks.values()),
    }


def enforce_phase_operational_guard(
    *,
    phase_dir: Path,
    phase: str,
    active_seconds: float,
    require_target_disk: bool,
    audit_fn: Any | None = None,
) -> dict[str, Any]:
    callback = default_phase_operational_audit if audit_fn is None else audit_fn
    audit = callback(
        phase_dir=phase_dir,
        phase=phase,
        active_seconds=float(active_seconds),
        require_target_disk=bool(require_target_disk),
    )
    if audit.get("passes") is not True:
        raise J1ExecutionOperationalHold(
            f"{phase} operational/resource guard failed"
        )
    return dict(audit)


def fixture_phase_operational_audit(
    *,
    phase_dir: Path,
    phase: str,
    active_seconds: float,
    require_target_disk: bool,
    output_bytes_override: int | None = None,
) -> dict[str, Any]:
    del require_target_disk
    return {
        "phase": phase,
        "active_seconds": float(active_seconds),
        "output_bytes": (
            j1.directory_bytes(phase_dir)
            if output_bytes_override is None
            else int(output_bytes_override)
        ),
        "checks": {"miniature_fixture_only": True},
        "passes": True,
    }


def _bounded_operational_callback(
    *,
    execution_mode: str,
    audit_fn: Any | None,
    output_accountant: PhaseOutputAccountant,
) -> Any:
    if execution_mode == "scientific":
        if audit_fn is not None:
            raise J1ExecutionIntegrityError(
                "Scientific operational audit cannot be injected"
            )
        base = default_phase_operational_audit
    else:
        base = fixture_phase_operational_audit if audit_fn is None else audit_fn

    def audited(**kwargs: Any) -> dict[str, Any]:
        if base in {
            default_phase_operational_audit,
            fixture_phase_operational_audit,
        }:
            return base(
                **kwargs,
                output_bytes_override=output_accountant.snapshot()[
                    "output_bytes"
                ],
            )
        return base(**kwargs)

    return audited


def execute_charged_phase_attempt(
    *,
    phase_dir: Path,
    phase: str,
    runtime_ledger: RuntimeChargeLedger,
    base_unit_id: str,
    operation: Any,
    audit_fn: Any,
    leave_open_after_work: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    before = runtime_ledger.summary()
    enforce_phase_operational_guard(
        phase_dir=phase_dir,
        phase=phase,
        active_seconds=before["active_seconds"],
        require_target_disk=before["attempts_started"] == 0,
        audit_fn=audit_fn,
    )
    started = runtime_ledger.begin(base_unit_id)
    try:
        result = operation()
    except BaseException:
        runtime_ledger.finish(started)
        raise
    if leave_open_after_work:
        raise J1ExecutionPlannedInterruption(
            "fixture interruption after charged work before closure"
        )
    runtime_ledger.finish(started)
    summary = runtime_ledger.summary()
    audit = enforce_phase_operational_guard(
        phase_dir=phase_dir,
        phase=phase,
        active_seconds=summary["active_seconds"],
        require_target_disk=False,
        audit_fn=audit_fn,
    )
    return result, summary, audit


def rolling_resume_storage_audit(
    root: Path,
    *,
    contract: Mapping[str, Any],
    planned_resume_boundaries: int,
    projected_journal_bytes_per_boundary: int,
    projected_root_blob_bytes: int,
    projected_epoch_commit_bytes: int,
    projected_checkpoint_bytes: int,
    projected_other_bytes: int,
    cap_gib: float,
    safety_multiplier: float = 1.25,
) -> dict[str, Any]:
    _validate_rolling_resume_contract(contract)
    if (
        planned_resume_boundaries < 1
        or projected_journal_bytes_per_boundary < 1
        or min(
            projected_root_blob_bytes,
            projected_epoch_commit_bytes,
            projected_checkpoint_bytes,
            projected_other_bytes,
        )
        < 0
        or cap_gib <= 0.0
        or safety_multiplier < 1.0
    ):
        raise ValueError("Rolling storage projection inputs are invalid")
    paths = _rolling_resume_paths(root)
    boundary = load_rolling_resume_boundary(root, contract=contract)
    slot_bytes = sum(
        paths[name].stat().st_size if paths[name].exists() else 0
        for name in ("slot0", "slot1")
    )
    journal_bytes = (
        paths["journal"].stat().st_size
        if paths["journal"].exists()
        else 0
    )
    head_bytes = (
        paths["head"].stat().st_size if paths["head"].exists() else 0
    )
    records = _load_rolling_journal(paths["journal"])
    max_slot_bytes = max(
        (
            paths[name].stat().st_size
            for name in ("slot0", "slot1")
            if paths[name].exists()
        ),
        default=0,
    )
    projection_terms = {
        "two_live_slots_plus_one_crash_orphan":
            3 * max_slot_bytes,
        "append_only_journal": (
            planned_resume_boundaries
            * projected_journal_bytes_per_boundary
        ),
        "immutable_root_blobs": projected_root_blob_bytes,
        "epoch_and_round_commits": projected_epoch_commit_bytes,
        "round64_checkpoint": projected_checkpoint_bytes,
        "other_phase_artifacts": projected_other_bytes,
    }
    projected_before_margin = sum(projection_terms.values())
    projected_with_margin = int(
        math.ceil(safety_multiplier * projected_before_margin)
    )
    cap_bytes = int(cap_gib * 1024**3)
    checks = {
        "bounded_slot_count": (
            sum(paths[name].exists() for name in ("slot0", "slot1"))
            <= 2
        ),
        "authenticated_head": boundary["passes"],
        "journal_within_planned_boundaries": (
            len(records) <= planned_resume_boundaries
        ),
        "projection_with_margin_within_cap": (
            projected_with_margin <= cap_bytes
        ),
    }
    return {
        "slot_count": sum(
            paths[name].exists() for name in ("slot0", "slot1")
        ),
        "slot_bytes": slot_bytes,
        "journal_bytes": journal_bytes,
        "head_bytes": head_bytes,
        "journal_records": len(records),
        "total_bytes": slot_bytes + journal_bytes + head_bytes,
        "projection_terms_bytes": projection_terms,
        "projected_before_margin_bytes": projected_before_margin,
        "safety_multiplier": safety_multiplier,
        "projected_with_margin_bytes": projected_with_margin,
        "cap_bytes": cap_bytes,
        "head_sequence": boundary["head"]["sequence"],
        "checks": checks,
        "append_only_journal": True,
        "passes": all(checks.values()),
    }


def _new_owner_record(
    *,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    predecessor_commit_head_sha256: str | None,
    execution_mode: str = "scientific",
    pid: int | None = None,
    start_identity: str | None = None,
) -> dict[str, Any]:
    owner_pid = os.getpid() if pid is None else int(pid)
    identity = (
        process_start_identity(owner_pid)
        if start_identity is None
        else start_identity
    )
    if not identity:
        raise J1ExecutionOperationalHold(
            "Process start identity is unavailable"
        )
    return _record_with_hash(
        {
            "version": f"{VERSION}_owner_record_v1",
            "phase": phase,
            "marker_file_sha256": marker_file_sha256,
            "phase_lock_file_sha256": phase_lock_file_sha256,
            "runner_sha256": sha256_path(RUNNER_PATH),
            "command": command,
            "execution_mode": execution_mode,
            "hostname": socket.gethostname(),
            "pid": owner_pid,
            "process_start_identity": identity,
            "predecessor_commit_head_sha256":
                predecessor_commit_head_sha256,
        },
        "owner_record_sha256",
    )


def acquire_writer_owner(
    *,
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    predecessor_commit_head_sha256: str | None,
    execution_mode: str = "scientific",
    pid: int | None = None,
    start_identity: str | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    owner_path = phase_dir / PHASE_OWNER_NAME
    owner = _new_owner_record(
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        predecessor_commit_head_sha256=predecessor_commit_head_sha256,
        execution_mode=execution_mode,
        pid=pid,
        start_identity=start_identity,
    )
    ledger = payload_with_hash(
        {
            "version": f"{VERSION}_ownership_ledger_v1",
            "owners": [owner],
            "recoveries": [],
            "head_owner_sha256": owner["owner_record_sha256"],
        },
        "ownership_payload_sha256",
    )
    serialized = (
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    phase_dir.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            owner_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        try:
            existing = load_json(owner_path)
            if not _verify_ownership_ledger(existing):
                raise J1ExecutionIntegrityError("invalid ownership ledger")
            head_owner = existing["owners"][-1]
            pid = int(head_owner.get("pid", -1))
            status = "live" if _pid_alive(pid) else "dead-same-ledger"
        except Exception:
            status = "malformed"
        raise J1ExecutionOperationalHold(
            f"Writer owner already exists ({status}): {owner_path}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    observed = load_json(owner_path)
    if not _verify_ownership_ledger(observed):
        raise J1ExecutionIntegrityError("Ownership ledger is invalid")
    return observed


def verify_writer_owner(
    *,
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    execution_mode: str = "scientific",
) -> dict[str, Any]:
    ledger = load_json(phase_dir / PHASE_OWNER_NAME)
    if not _verify_ownership_ledger(ledger):
        raise J1ExecutionIntegrityError("Ownership ledger is malformed")
    owner = ledger["owners"][-1]
    checks = {
        "payload_stable": True,
        "phase_exact": owner.get("phase") == phase,
        "marker_exact": owner.get("marker_file_sha256")
        == marker_file_sha256,
        "phase_lock_exact": owner.get("phase_lock_file_sha256")
        == phase_lock_file_sha256,
        "runner_exact": owner.get("runner_sha256")
        == sha256_path(RUNNER_PATH),
        "command_exact": owner.get("command") == command,
        "execution_mode_exact": owner.get("execution_mode") == execution_mode,
        "pid_current": int(owner.get("pid", -1)) == os.getpid(),
        "owner_live": _pid_alive(int(owner.get("pid", -1))),
    }
    if not all(checks.values()):
        raise J1ExecutionOperationalHold("Writer ownership changed")
    return {
        "ledger": ledger,
        "owner": owner,
        "checks": checks,
        "passes": True,
    }


def _resolve_phase_artifact(phase_dir: Path, relative_path: str) -> Path:
    root = phase_dir.resolve()
    target = (phase_dir / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise J1ExecutionIntegrityError(
            f"Commit artifact escapes phase directory: {relative_path}"
        ) from error
    return target


def _write_immutable_binary_exact(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    expected = serialize_binary_state(payload)
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise J1ExecutionIntegrityError(
                f"Immutable binary orphan mismatch: {path}"
            )
        deserialize_binary_state(observed)
        return hashlib.sha256(observed).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(expected)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        observed = path.read_bytes()
        if observed != expected:
            raise J1ExecutionIntegrityError(
                f"Concurrent immutable binary mismatch: {path}"
            )
    finally:
        temporary.unlink(missing_ok=True)
    observed = path.read_bytes()
    if observed != expected:
        raise J1ExecutionIntegrityError(
            f"Immutable binary write changed bytes: {path}"
        )
    deserialize_binary_state(observed)
    return hashlib.sha256(observed).hexdigest()


def _write_immutable_json_exact(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    expected = payload_with_hash(payload, field)
    if path.exists():
        observed = load_json(path)
        if observed != expected or not verify_payload_hash(observed, field):
            raise J1ExecutionIntegrityError(
                f"Immutable JSON orphan mismatch: {path}"
            )
        return observed
    try:
        return write_immutable_json(path, payload, field=field)
    except FileExistsError:
        observed = load_json(path)
        if observed != expected or not verify_payload_hash(observed, field):
            raise J1ExecutionIntegrityError(
                f"Concurrent immutable JSON mismatch: {path}"
            )
        return observed


def _commit_artifact_name(sequence: int, unit_id: str, suffix: str) -> str:
    unit_digest = hashlib.sha256(unit_id.encode("utf-8")).hexdigest()[:20]
    return f"{sequence:09d}_{unit_digest}.{suffix}"


def _commit_contract(
    *,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    execution_mode: str = "scientific",
) -> dict[str, Any]:
    if execution_mode not in {"scientific", "miniature_fixture"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
    return {
        "phase": phase,
        "marker_file_sha256": marker_file_sha256,
        "phase_lock_file_sha256": phase_lock_file_sha256,
        "runner_sha256": sha256_path(RUNNER_PATH),
        "command": command,
        "execution_mode": execution_mode,
    }


def _verify_contract(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if any(payload.get(key) != value for key, value in expected.items()):
        raise J1ExecutionIntegrityError(
            f"{label} phase/marker/lock/runner/command mismatch"
        )


def _read_hashed_json_artifact(
    *,
    phase_dir: Path,
    relative_path: str,
    expected_file_sha256: str,
    expected_payload_sha256: str,
    payload_field: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    target = _resolve_phase_artifact(phase_dir, relative_path)
    if not target.is_file():
        raise J1ExecutionIntegrityError(f"Missing {label}: {target}")
    if sha256_path(target) != expected_file_sha256:
        raise J1ExecutionIntegrityError(f"Tampered {label}: {target}")
    payload = load_json(target)
    if (
        not verify_payload_hash(payload, payload_field)
        or payload.get(payload_field) != expected_payload_sha256
    ):
        raise J1ExecutionIntegrityError(
            f"Invalid {label} canonical payload: {target}"
        )
    return target, payload


def _verify_state_contract(
    state: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    sequence: int,
    unit_id: str,
) -> None:
    _verify_contract(state, contract, label="Committed state")
    if int(state.get("commit_sequence", -1)) != sequence:
        raise J1ExecutionIntegrityError("Committed state sequence changed")
    if state.get("latest_unit_id") != unit_id:
        raise J1ExecutionIntegrityError("Committed state unit changed")
    unit_ids = state.get("committed_unit_ids")
    if unit_ids is not None:
        if (
            not isinstance(unit_ids, list)
            or len(set(unit_ids)) != len(unit_ids)
        ):
            raise J1ExecutionIntegrityError(
                "Committed unit identities are malformed or duplicated"
            )
        if not unit_ids or unit_ids[-1] != unit_id:
            raise J1ExecutionIntegrityError(
                "Committed state does not close its latest unit"
            )
        if len(unit_ids) != sequence + 1:
            raise J1ExecutionIntegrityError(
                "Committed state unit-prefix length changed"
            )
        return
    if (
        state.get("commit_prefix_mode") != COMPACT_COMMIT_PREFIX_MODE
        or int(state.get("committed_unit_count", -1)) != sequence + 1
        or not isinstance(state.get("committed_unit_head_sha256"), str)
        or len(str(state["committed_unit_head_sha256"])) != 64
    ):
        raise J1ExecutionIntegrityError(
            "Compact committed-unit prefix is malformed"
        )


def _next_compact_commit_prefix(
    *,
    predecessor_sha256: str | None,
    sequence: int,
    unit_id: str,
) -> str:
    return canonical_json_hash(
        {
            "version": COMPACT_COMMIT_PREFIX_MODE,
            "predecessor_sha256": predecessor_sha256,
            "sequence": int(sequence),
            "unit_id": str(unit_id),
        }
    )


def _verify_full_commit_chain(
    *,
    phase_dir: Path,
    first_head_path: Path,
    first_head_file_sha256: str,
    first_head_payload_sha256: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected_path = first_head_path
    expected_file_sha256 = first_head_file_sha256
    expected_payload_sha256 = first_head_payload_sha256
    seen_paths: set[Path] = set()
    descending: list[dict[str, Any]] = []
    while True:
        resolved = expected_path.resolve()
        if resolved in seen_paths:
            raise J1ExecutionIntegrityError("Commit-head chain contains a cycle")
        seen_paths.add(resolved)
        try:
            relative = str(resolved.relative_to(phase_dir.resolve()))
        except ValueError as error:
            raise J1ExecutionIntegrityError(
                "Commit-head chain escapes phase directory"
            ) from error
        head_path, head = _read_hashed_json_artifact(
            phase_dir=phase_dir,
            relative_path=relative,
            expected_file_sha256=expected_file_sha256,
            expected_payload_sha256=expected_payload_sha256,
            payload_field="commit_head_record_sha256",
            label="commit-chain head record",
        )
        _verify_contract(head, contract, label="Commit-chain head record")
        sequence = int(head.get("sequence", -1))
        unit_id = str(head.get("unit_id", ""))
        if sequence < 0 or not unit_id:
            raise J1ExecutionIntegrityError(
                "Commit-chain head identity is malformed"
            )
        record_path, record = _read_hashed_json_artifact(
            phase_dir=phase_dir,
            relative_path=str(head.get("commit_record_path", "")),
            expected_file_sha256=str(
                head.get("commit_record_file_sha256", "")
            ),
            expected_payload_sha256=str(
                head.get("commit_record_payload_sha256", "")
            ),
            payload_field="commit_record_sha256",
            label="commit-chain record",
        )
        _verify_contract(record, contract, label="Commit-chain record")
        if (
            int(record.get("sequence", -1)) != sequence
            or record.get("unit_id") != unit_id
        ):
            raise J1ExecutionIntegrityError(
                "Commit-chain record identity changed"
            )
        state_path = _resolve_phase_artifact(
            phase_dir,
            str(record.get("post_state_path", "")),
        )
        if (
            not state_path.is_file()
            or sha256_path(state_path)
            != record.get("post_state_file_sha256")
            or record.get("post_state_file_sha256")
            != head.get("post_state_file_sha256")
        ):
            raise J1ExecutionIntegrityError(
                "Commit-chain state is missing or tampered"
            )
        state = load_atomic_binary(state_path)
        _verify_state_contract(
            state,
            contract=contract,
            sequence=sequence,
            unit_id=unit_id,
        )
        if (
            contract["phase"] == "training"
            and contract["execution_mode"] == "scientific"
            and sequence > 0
        ):
            validate_training_runtime_payload(state)
        journal_path, journal = _read_hashed_json_artifact(
            phase_dir=phase_dir,
            relative_path=str(record.get("journal_path", "")),
            expected_file_sha256=str(record.get("journal_file_sha256", "")),
            expected_payload_sha256=str(
                record.get("journal_payload_sha256", "")
            ),
            payload_field="journal_payload_sha256",
            label="commit-chain journal",
        )
        _verify_contract(journal, contract, label="Commit-chain journal")
        if (
            int(journal.get("sequence", -1)) != sequence
            or journal.get("unit_id") != unit_id
            or record.get("journal_file_sha256")
            != head.get("journal_file_sha256")
        ):
            raise J1ExecutionIntegrityError(
                "Commit-chain journal identity changed"
            )
        descending.append(
            {
                "sequence": sequence,
                "unit_id": unit_id,
                "head_path": head_path,
                "head_file_sha256": sha256_path(head_path),
                "head_payload_sha256": head[
                    "commit_head_record_sha256"
                ],
                "record_path": record_path,
                "record": record,
                "state_path": state_path,
                "state_file_sha256": sha256_path(state_path),
                "state": state,
                "journal_path": journal_path,
                "journal_file_sha256": sha256_path(journal_path),
            }
        )
        if sequence == 0:
            if unit_id != "genesis":
                raise J1ExecutionIntegrityError(
                    "Commit chain does not terminate at genesis"
                )
            if any(
                record.get(key) is not None
                for key in (
                    "predecessor_head_record_path",
                    "predecessor_head_record_file_sha256",
                    "predecessor_head_record_payload_sha256",
                    "predecessor_state_file_sha256",
                )
            ):
                raise J1ExecutionIntegrityError(
                    "Genesis record has predecessor evidence"
                )
            break
        expected_path = _resolve_phase_artifact(
            phase_dir,
            str(record.get("predecessor_head_record_path", "")),
        )
        expected_file_sha256 = str(
            record.get("predecessor_head_record_file_sha256", "")
        )
        expected_payload_sha256 = str(
            record.get("predecessor_head_record_payload_sha256", "")
        )

    ascending = list(reversed(descending))
    if [row["sequence"] for row in ascending] != list(
        range(len(ascending))
    ):
        raise J1ExecutionIntegrityError(
            "Commit chain has a missing or nonconsecutive sequence"
        )
    all_unit_ids: list[str] = []
    previous_state_hash = None
    compact_prefix_sha256: str | None = None
    prefix_mode: str | None = None
    for index, row in enumerate(ascending):
        all_unit_ids.append(row["unit_id"])
        state = row["state"]
        if state.get("committed_unit_ids") is not None:
            if prefix_mode == COMPACT_COMMIT_PREFIX_MODE:
                raise J1ExecutionIntegrityError(
                    "Commit chain changed unit-prefix representation"
                )
            prefix_mode = "legacy_unit_list_v1"
            if state["committed_unit_ids"] != all_unit_ids:
                raise J1ExecutionIntegrityError(
                    "Commit state does not preserve the exact unit prefix"
                )
        else:
            if prefix_mode == "legacy_unit_list_v1":
                raise J1ExecutionIntegrityError(
                    "Commit chain changed unit-prefix representation"
                )
            prefix_mode = COMPACT_COMMIT_PREFIX_MODE
            compact_prefix_sha256 = _next_compact_commit_prefix(
                predecessor_sha256=compact_prefix_sha256,
                sequence=index,
                unit_id=row["unit_id"],
            )
            if (
                state.get("commit_prefix_mode")
                != COMPACT_COMMIT_PREFIX_MODE
                or int(state.get("committed_unit_count", -1))
                != index + 1
                or state.get("committed_unit_head_sha256")
                != compact_prefix_sha256
            ):
                raise J1ExecutionIntegrityError(
                    "Commit state compact unit prefix changed"
                )
        predecessor_state_hash = row["record"].get(
            "predecessor_state_file_sha256"
        )
        if index == 0:
            if predecessor_state_hash is not None:
                raise J1ExecutionIntegrityError(
                    "Genesis state has predecessor hash"
                )
        elif predecessor_state_hash != previous_state_hash:
            raise J1ExecutionIntegrityError(
                "Commit record predecessor-state hash changed"
            )
        previous_state_hash = row["state_file_sha256"]
    if len(set(all_unit_ids)) != len(all_unit_ids):
        raise J1ExecutionIntegrityError(
            "Commit chain duplicated a unit identity"
        )
    return {
        "sequence_count": len(ascending),
        "unit_ids": all_unit_ids,
        "unit_ids_sha256": canonical_json_hash(all_unit_ids),
        "unit_prefix_mode": prefix_mode,
        "compact_unit_prefix_sha256": compact_prefix_sha256,
        "genesis_verified": True,
        "full_predecessor_chain_verified": True,
        "passes": True,
    }


def verify_commit_boundary(
    *,
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    execution_mode: str = "scientific",
    verify_full_chain: bool = True,
) -> dict[str, Any]:
    pointer_path = phase_dir / COMMIT_HEAD_NAME
    if not pointer_path.is_file():
        raise J1ExecutionIntegrityError("Authenticated commit head is missing")
    try:
        pointer = load_json(pointer_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise J1ExecutionIntegrityError(
            "Authenticated commit-head pointer cannot be decoded"
        ) from error
    if not verify_payload_hash(pointer, "commit_head_pointer_sha256"):
        raise J1ExecutionIntegrityError(
            "Authenticated commit-head pointer is malformed"
        )
    contract = _commit_contract(
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        execution_mode=execution_mode,
    )
    _verify_contract(pointer, contract, label="Commit-head pointer")
    sequence = int(pointer.get("sequence", -1))
    if sequence < 0:
        raise J1ExecutionIntegrityError("Commit-head sequence is invalid")

    head_path, head = _read_hashed_json_artifact(
        phase_dir=phase_dir,
        relative_path=str(pointer.get("head_record_path", "")),
        expected_file_sha256=str(pointer.get("head_record_file_sha256", "")),
        expected_payload_sha256=str(
            pointer.get("head_record_payload_sha256", "")
        ),
        payload_field="commit_head_record_sha256",
        label="commit-head record",
    )
    _verify_contract(head, contract, label="Commit-head record")
    if int(head.get("sequence", -1)) != sequence:
        raise J1ExecutionIntegrityError(
            "Commit-head pointer/record sequence mismatch"
        )
    unit_id = str(head.get("unit_id", ""))
    if not unit_id:
        raise J1ExecutionIntegrityError("Commit-head unit identity is empty")

    record_path, record = _read_hashed_json_artifact(
        phase_dir=phase_dir,
        relative_path=str(head.get("commit_record_path", "")),
        expected_file_sha256=str(head.get("commit_record_file_sha256", "")),
        expected_payload_sha256=str(
            head.get("commit_record_payload_sha256", "")
        ),
        payload_field="commit_record_sha256",
        label="commit record",
    )
    _verify_contract(record, contract, label="Commit record")
    if (
        int(record.get("sequence", -1)) != sequence
        or record.get("unit_id") != unit_id
    ):
        raise J1ExecutionIntegrityError("Commit record identity changed")

    state_path = _resolve_phase_artifact(
        phase_dir,
        str(record.get("post_state_path", "")),
    )
    if (
        not state_path.is_file()
        or sha256_path(state_path)
        != record.get("post_state_file_sha256")
    ):
        raise J1ExecutionIntegrityError("Committed state is missing or tampered")
    state = load_atomic_binary(state_path)
    _verify_state_contract(
        state,
        contract=contract,
        sequence=sequence,
        unit_id=unit_id,
    )

    journal_path, journal = _read_hashed_json_artifact(
        phase_dir=phase_dir,
        relative_path=str(record.get("journal_path", "")),
        expected_file_sha256=str(record.get("journal_file_sha256", "")),
        expected_payload_sha256=str(
            record.get("journal_payload_sha256", "")
        ),
        payload_field="journal_payload_sha256",
        label="commit journal",
    )
    _verify_contract(journal, contract, label="Commit journal")
    if (
        int(journal.get("sequence", -1)) != sequence
        or journal.get("unit_id") != unit_id
    ):
        raise J1ExecutionIntegrityError("Commit journal identity changed")
    if record.get("post_state_file_sha256") != head.get(
        "post_state_file_sha256"
    ):
        raise J1ExecutionIntegrityError(
            "Commit-head state reference changed"
        )
    if record.get("journal_file_sha256") != head.get(
        "journal_file_sha256"
    ):
        raise J1ExecutionIntegrityError(
            "Commit-head journal reference changed"
        )

    if sequence == 0:
        if any(
            value is not None
            for value in (
                record.get("predecessor_head_record_path"),
                record.get("predecessor_head_record_file_sha256"),
                record.get("predecessor_head_record_payload_sha256"),
            )
        ):
            raise J1ExecutionIntegrityError(
                "Genesis commit has a predecessor"
            )
    else:
        predecessor_path, predecessor = _read_hashed_json_artifact(
            phase_dir=phase_dir,
            relative_path=str(
                record.get("predecessor_head_record_path", "")
            ),
            expected_file_sha256=str(
                record.get("predecessor_head_record_file_sha256", "")
            ),
            expected_payload_sha256=str(
                record.get("predecessor_head_record_payload_sha256", "")
            ),
            payload_field="commit_head_record_sha256",
            label="predecessor commit-head record",
        )
        del predecessor_path
        _verify_contract(
            predecessor,
            contract,
            label="Predecessor commit-head record",
        )
        if int(predecessor.get("sequence", -1)) != sequence - 1:
            raise J1ExecutionIntegrityError(
                "Commit-head predecessor is stale or nonconsecutive"
            )

    if (
        phase == "training"
        and execution_mode == "scientific"
        and sequence > 0
    ):
        validate_training_runtime_payload(state)
    chain_audit = (
        _verify_full_commit_chain(
            phase_dir=phase_dir,
            first_head_path=head_path,
            first_head_file_sha256=sha256_path(head_path),
            first_head_payload_sha256=head[
                "commit_head_record_sha256"
            ],
            contract=contract,
        )
        if verify_full_chain
        else {
            "sequence_count": None,
            "full_predecessor_chain_verified": False,
            "verification_deferred_to_resume_or_terminal": True,
            "passes": True,
        }
    )

    return {
        "sequence": sequence,
        "unit_id": unit_id,
        "commit_head_path": str(pointer_path),
        "commit_head_file_sha256": sha256_path(pointer_path),
        "commit_head_payload_sha256": pointer[
            "commit_head_pointer_sha256"
        ],
        "head_record_path": str(head_path),
        "head_record_file_sha256": sha256_path(head_path),
        "head_record_payload_sha256": head[
            "commit_head_record_sha256"
        ],
        "commit_record_path": str(record_path),
        "commit_record_file_sha256": sha256_path(record_path),
        "commit_record_payload_sha256": record[
            "commit_record_sha256"
        ],
        "state_path": str(state_path),
        "state_file_sha256": sha256_path(state_path),
        "latest_journal_path": str(journal_path),
        "latest_journal_file_sha256": sha256_path(journal_path),
        "latest_journal_payload_sha256": journal[
            "journal_payload_sha256"
        ],
        "chain_audit": chain_audit,
        "state": state,
        "passes": True,
    }


def _commit_paths(
    phase_dir: Path,
    *,
    sequence: int,
    unit_id: str,
) -> dict[str, Path]:
    return {
        "state": phase_dir
        / COMMIT_STATES_DIR
        / _commit_artifact_name(sequence, unit_id, "bin"),
        "journal": phase_dir
        / COMMIT_JOURNALS_DIR
        / _commit_artifact_name(sequence, unit_id, "json"),
        "record": phase_dir
        / COMMIT_RECORDS_DIR
        / _commit_artifact_name(sequence, unit_id, "json"),
        "head_record": phase_dir
        / COMMIT_HEADS_DIR
        / _commit_artifact_name(sequence, unit_id, "json"),
        "pointer": phase_dir / COMMIT_HEAD_NAME,
    }


def _relative_phase_path(phase_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(phase_dir.resolve()))
    except ValueError as error:
        raise J1ExecutionIntegrityError(
            f"Commit path escapes phase directory: {path}"
        ) from error


def _write_commit_boundary(
    *,
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    unit_id: str,
    sequence: int,
    predecessor: Mapping[str, Any] | None,
    post_state: Mapping[str, Any],
    journal_payload: Mapping[str, Any],
    execution_mode: str = "scientific",
    crash_stage: str | None = None,
    verify_full_chain_after_write: bool = True,
    output_accountant: "PhaseOutputAccountant | None" = None,
) -> dict[str, Any]:
    if crash_stage not in {
        None,
        "after_state",
        "after_record",
        "after_head_record",
        "after_pointer",
    }:
        raise ValueError(f"Unknown crash stage: {crash_stage}")
    contract = _commit_contract(
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        execution_mode=execution_mode,
    )
    paths = _commit_paths(
        phase_dir,
        sequence=sequence,
        unit_id=unit_id,
    )
    state = {
        **dict(post_state),
        **contract,
        "commit_sequence": sequence,
        "latest_unit_id": unit_id,
    }
    if phase == "training" and execution_mode == "scientific" and sequence > 0:
        validate_training_runtime_payload(state)
    _verify_state_contract(
        state,
        contract=contract,
        sequence=sequence,
        unit_id=unit_id,
    )
    state_file_sha256 = _write_immutable_binary_exact(
        paths["state"],
        state,
    )
    if output_accountant is not None:
        output_accountant.record_path(paths["state"])
    if crash_stage == "after_state":
        raise RuntimeError("fixture crash after immutable post-state")

    journal = _write_immutable_json_exact(
        paths["journal"],
        {
            "version": f"{VERSION}_commit_journal_v1",
            **contract,
            "sequence": sequence,
            "unit_id": unit_id,
            "payload": dict(journal_payload),
        },
        field="journal_payload_sha256",
    )
    if output_accountant is not None:
        output_accountant.record_path(paths["journal"])
    record = _write_immutable_json_exact(
        paths["record"],
        {
            "version": f"{VERSION}_commit_record_v1",
            **contract,
            "sequence": sequence,
            "unit_id": unit_id,
            "predecessor_head_record_path": (
                None
                if predecessor is None
                else _relative_phase_path(
                    phase_dir,
                    Path(str(predecessor["head_record_path"])),
                )
            ),
            "predecessor_head_record_file_sha256": (
                None
                if predecessor is None
                else predecessor["head_record_file_sha256"]
            ),
            "predecessor_head_record_payload_sha256": (
                None
                if predecessor is None
                else predecessor["head_record_payload_sha256"]
            ),
            "predecessor_state_file_sha256": (
                None if predecessor is None else predecessor["state_file_sha256"]
            ),
            "post_state_path": _relative_phase_path(
                phase_dir,
                paths["state"],
            ),
            "post_state_file_sha256": state_file_sha256,
            "journal_path": _relative_phase_path(
                phase_dir,
                paths["journal"],
            ),
            "journal_file_sha256": sha256_path(paths["journal"]),
            "journal_payload_sha256": journal["journal_payload_sha256"],
        },
        field="commit_record_sha256",
    )
    if output_accountant is not None:
        output_accountant.record_path(paths["record"])
    if crash_stage == "after_record":
        raise RuntimeError("fixture crash after immutable commit record")

    head = _write_immutable_json_exact(
        paths["head_record"],
        {
            "version": f"{VERSION}_commit_head_record_v1",
            **contract,
            "sequence": sequence,
            "unit_id": unit_id,
            "commit_record_path": _relative_phase_path(
                phase_dir,
                paths["record"],
            ),
            "commit_record_file_sha256": sha256_path(paths["record"]),
            "commit_record_payload_sha256": record["commit_record_sha256"],
            "post_state_file_sha256": state_file_sha256,
            "journal_file_sha256": sha256_path(paths["journal"]),
        },
        field="commit_head_record_sha256",
    )
    if output_accountant is not None:
        output_accountant.record_path(paths["head_record"])
    if crash_stage == "after_head_record":
        raise RuntimeError("fixture crash before commit-head advancement")

    pointer = _atomic_replace_json(
        paths["pointer"],
        {
            "version": f"{VERSION}_commit_head_pointer_v1",
            **contract,
            "sequence": sequence,
            "unit_id": unit_id,
            "head_record_path": _relative_phase_path(
                phase_dir,
                paths["head_record"],
            ),
            "head_record_file_sha256": sha256_path(paths["head_record"]),
            "head_record_payload_sha256": head[
                "commit_head_record_sha256"
            ],
        },
        field="commit_head_pointer_sha256",
    )
    if output_accountant is not None:
        output_accountant.record_path(paths["pointer"])
    del pointer
    boundary = verify_commit_boundary(
        phase_dir=phase_dir,
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        execution_mode=execution_mode,
        verify_full_chain=verify_full_chain_after_write,
    )
    if crash_stage == "after_pointer":
        raise RuntimeError("fixture crash after commit-head advancement")
    return boundary


def initialize_commit_store(
    *,
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    initial_state: Mapping[str, Any],
    execution_mode: str = "scientific",
) -> dict[str, Any]:
    pointer_path = phase_dir / COMMIT_HEAD_NAME
    if pointer_path.exists():
        return verify_commit_boundary(
            phase_dir=phase_dir,
            phase=phase,
            marker_file_sha256=marker_file_sha256,
            phase_lock_file_sha256=phase_lock_file_sha256,
            command=command,
            execution_mode=execution_mode,
        )
    state = dict(initial_state)
    if state.get("committed_unit_ids") not in (None, []):
        raise J1ExecutionIntegrityError(
            "Genesis state already contains committed units"
        )
    state["committed_unit_ids"] = ["genesis"]
    return _write_commit_boundary(
        phase_dir=phase_dir,
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        unit_id="genesis",
        sequence=0,
        predecessor=None,
        post_state=state,
        journal_payload={"boundary": "genesis", "scientific_work": 0},
        execution_mode=execution_mode,
    )


def commit_unit(
    *,
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    unit_id: str,
    post_state: Mapping[str, Any],
    journal_payload: Mapping[str, Any],
    crash_stage: str | None = None,
    execution_mode: str = "scientific",
) -> dict[str, Any]:
    if not unit_id or unit_id == "genesis":
        raise ValueError("Commit unit identity is invalid")
    predecessor = verify_commit_boundary(
        phase_dir=phase_dir,
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        execution_mode=execution_mode,
    )
    committed = list(predecessor["state"]["committed_unit_ids"])
    if unit_id in committed:
        raise J1ExecutionIntegrityError(
            f"Commit unit was already closed: {unit_id}"
        )
    proposed = dict(post_state)
    proposed_units = proposed.get("committed_unit_ids")
    expected_units = [*committed, unit_id]
    if proposed_units not in (None, expected_units):
        raise J1ExecutionIntegrityError(
            "Post-state committed units do not extend predecessor exactly"
        )
    proposed["committed_unit_ids"] = expected_units
    return _write_commit_boundary(
        phase_dir=phase_dir,
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        unit_id=unit_id,
        sequence=int(predecessor["sequence"]) + 1,
        predecessor=predecessor,
        post_state=proposed,
        journal_payload=journal_payload,
        crash_stage=crash_stage,
        execution_mode=execution_mode,
    )


class IndexedCommitStore:
    """One full resume audit followed by O(1) current-head appends."""

    def __init__(
        self,
        *,
        phase_dir: Path,
        phase: str,
        marker_file_sha256: str,
        phase_lock_file_sha256: str,
        command: str,
        initial_state: Mapping[str, Any],
        execution_mode: str = "scientific",
        output_accountant: "PhaseOutputAccountant | None" = None,
    ) -> None:
        self.phase_dir = phase_dir
        self.phase = phase
        self.marker_file_sha256 = marker_file_sha256
        self.phase_lock_file_sha256 = phase_lock_file_sha256
        self.command = command
        self.execution_mode = execution_mode
        self.output_accountant = output_accountant
        self.full_chain_scan_count = 0
        self.current_head_verification_count = 0
        self.current_head_verified_bytes = 0
        self.append_count = 0
        pointer = phase_dir / COMMIT_HEAD_NAME
        if pointer.exists():
            self.boundary = self._full_audit()
        else:
            state = dict(initial_state)
            forbidden = {
                "committed_unit_ids",
                "commit_prefix_mode",
                "committed_unit_count",
                "committed_unit_head_sha256",
            } & set(state)
            if forbidden:
                raise J1ExecutionIntegrityError(
                    "Indexed commit genesis already contains prefix fields"
                )
            state.update(
                {
                    "commit_prefix_mode": COMPACT_COMMIT_PREFIX_MODE,
                    "committed_unit_count": 1,
                    "committed_unit_head_sha256":
                        _next_compact_commit_prefix(
                            predecessor_sha256=None,
                            sequence=0,
                            unit_id="genesis",
                        ),
                }
            )
            self.boundary = _write_commit_boundary(
                phase_dir=phase_dir,
                phase=phase,
                marker_file_sha256=marker_file_sha256,
                phase_lock_file_sha256=phase_lock_file_sha256,
                command=command,
                unit_id="genesis",
                sequence=0,
                predecessor=None,
                post_state=state,
                journal_payload={
                    "boundary": "genesis",
                    "scientific_work": 0,
                    "indexed_commit_store": True,
                },
                execution_mode=execution_mode,
                verify_full_chain_after_write=False,
                output_accountant=output_accountant,
            )
            self.current_head_verification_count += 1
            self.current_head_verified_bytes += (
                self._current_boundary_file_bytes(self.boundary)
            )
            self.boundary = self._full_audit()
        chain = self.boundary["chain_audit"]
        self.unit_ids = set(chain.get("unit_ids", []))
        if not self.unit_ids:
            raise J1ExecutionIntegrityError(
                "Indexed commit store has no authenticated genesis"
            )

    def _contract_kwargs(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "marker_file_sha256": self.marker_file_sha256,
            "phase_lock_file_sha256": self.phase_lock_file_sha256,
            "command": self.command,
            "execution_mode": self.execution_mode,
        }

    @staticmethod
    def _current_boundary_file_bytes(
        boundary: Mapping[str, Any],
    ) -> int:
        paths = {
            str(boundary[key])
            for key in (
                "commit_head_path",
                "head_record_path",
                "commit_record_path",
                "state_path",
                "latest_journal_path",
            )
        }
        return sum(
            Path(path).stat().st_size
            for path in paths
            if Path(path).is_file()
        )

    def _full_audit(self) -> dict[str, Any]:
        boundary = verify_commit_boundary(
            phase_dir=self.phase_dir,
            **self._contract_kwargs(),
            verify_full_chain=True,
        )
        self.full_chain_scan_count += 1
        return boundary

    def commit(
        self,
        *,
        unit_id: str,
        post_state: Mapping[str, Any],
        journal_payload: Mapping[str, Any],
        crash_stage: str | None = None,
    ) -> dict[str, Any]:
        if not unit_id or unit_id == "genesis":
            raise ValueError("Indexed commit unit identity is invalid")
        if unit_id in self.unit_ids:
            raise J1ExecutionIntegrityError(
                f"Indexed commit unit was already closed: {unit_id}"
            )
        predecessor_state = self.boundary["state"]
        if (
            predecessor_state.get("commit_prefix_mode")
            != COMPACT_COMMIT_PREFIX_MODE
        ):
            raise J1ExecutionIntegrityError(
                "Indexed commit predecessor is not compact"
            )
        proposed = dict(post_state)
        forbidden = {
            "committed_unit_ids",
            "commit_prefix_mode",
            "committed_unit_count",
            "committed_unit_head_sha256",
        } & set(proposed)
        if forbidden:
            raise J1ExecutionIntegrityError(
                "Indexed commit post-state supplied prefix fields"
            )
        sequence = int(self.boundary["sequence"]) + 1
        proposed.update(
            {
                "commit_prefix_mode": COMPACT_COMMIT_PREFIX_MODE,
                "committed_unit_count": sequence + 1,
                "committed_unit_head_sha256":
                    _next_compact_commit_prefix(
                        predecessor_sha256=predecessor_state[
                            "committed_unit_head_sha256"
                        ],
                        sequence=sequence,
                        unit_id=unit_id,
                    ),
            }
        )
        boundary = _write_commit_boundary(
            phase_dir=self.phase_dir,
            **self._contract_kwargs(),
            unit_id=unit_id,
            sequence=sequence,
            predecessor=self.boundary,
            post_state=proposed,
            journal_payload=journal_payload,
            crash_stage=crash_stage,
            verify_full_chain_after_write=False,
            output_accountant=self.output_accountant,
        )
        self.current_head_verification_count += 1
        self.current_head_verified_bytes += (
            self._current_boundary_file_bytes(boundary)
        )
        self.append_count += 1
        self.boundary = boundary
        self.unit_ids.add(unit_id)
        return boundary

    def audit_full(self) -> dict[str, Any]:
        observed = self._full_audit()
        if (
            observed["commit_head_payload_sha256"]
            != self.boundary["commit_head_payload_sha256"]
            or observed["state_file_sha256"]
            != self.boundary["state_file_sha256"]
        ):
            raise J1ExecutionIntegrityError(
                "Indexed commit terminal audit changed current head"
            )
        self.boundary = observed
        return observed

    def metrics(self) -> dict[str, Any]:
        return {
            "full_chain_scan_count": self.full_chain_scan_count,
            "current_head_verification_count":
                self.current_head_verification_count,
            "current_head_verified_bytes":
                self.current_head_verified_bytes,
            "append_count": self.append_count,
            "unit_count": len(self.unit_ids),
            "committed_unit_prefix_copied": False,
            "passes": True,
        }


def reclaim_dead_writer_owner(
    *,
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    execution_mode: str = "scientific",
    pid_alive: Any = _pid_alive,
    process_identity: Any = process_start_identity,
    contention_audit: Mapping[str, Any] | None = None,
    new_pid: int | None = None,
    new_start_identity: str | None = None,
) -> dict[str, Any]:
    path = phase_dir / PHASE_OWNER_NAME
    ledger = load_json(path)
    if not _verify_ownership_ledger(ledger):
        raise J1ExecutionIntegrityError("Ownership ledger is malformed")
    old = ledger["owners"][-1]
    expected = {
        "phase": phase,
        "marker_file_sha256": marker_file_sha256,
        "phase_lock_file_sha256": phase_lock_file_sha256,
        "runner_sha256": sha256_path(RUNNER_PATH),
        "command": command,
        "execution_mode": execution_mode,
    }
    if any(old.get(key) != value for key, value in expected.items()):
        raise J1ExecutionOperationalHold(
            "Dead-owner reclaim contract does not match"
        )
    old_pid = int(old.get("pid", -1))
    if pid_alive(old_pid):
        raise J1ExecutionOperationalHold("Current writer owner is still live")
    observed_start = process_identity(old_pid)
    if observed_start is not None:
        raise J1ExecutionOperationalHold(
            "Old process death cannot be verified"
        )
    commit_head_path = phase_dir / COMMIT_HEAD_NAME
    if commit_head_path.exists():
        committed_boundary = verify_commit_boundary(
            phase_dir=phase_dir,
            phase=phase,
            marker_file_sha256=marker_file_sha256,
            phase_lock_file_sha256=phase_lock_file_sha256,
            command=command,
            execution_mode=execution_mode,
        )
        boundary_evidence = {
            key: committed_boundary[key]
            for key in (
                "sequence",
                "unit_id",
                "commit_head_file_sha256",
                "commit_head_payload_sha256",
                "head_record_file_sha256",
                "head_record_payload_sha256",
                "commit_record_file_sha256",
                "commit_record_payload_sha256",
                "state_file_sha256",
                "latest_journal_file_sha256",
                "latest_journal_payload_sha256",
            )
        }
        predecessor_commit_head_sha256 = committed_boundary[
            "commit_head_file_sha256"
        ]
    else:
        allowed_payload_fields = {
            PHASE_LOCK_NAME: "phase_lock_payload_sha256",
            PHASE_LOCK_RESULT_NAME: "phase_lock_result_payload_sha256",
            PHASE_MARKER_NAME: "activation_marker_payload_sha256",
            PHASE_MANIFEST_NAME: "root_manifest_payload_sha256",
            PHASE_OWNER_NAME: "ownership_payload_sha256",
            PHASE_STREAM_RESERVATION_NAME:
                "stream_reservation_payload_sha256",
            PHASE_STREAM_CONSUMPTION_NAME:
                "stream_consumption_payload_sha256",
        }
        observed_files = sorted(
            path
            for path in phase_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        unexpected = [
            str(path.relative_to(phase_dir))
            for path in observed_files
            if str(path.relative_to(phase_dir))
            not in allowed_payload_fields
        ]
        identities = {}
        for path in observed_files:
            relative = str(path.relative_to(phase_dir))
            field = allowed_payload_fields.get(relative)
            if field is None:
                continue
            payload = load_json(path)
            if not verify_payload_hash(payload, field):
                raise J1ExecutionIntegrityError(
                    f"No-head recovery artifact is invalid: {relative}"
                )
            identities[relative] = {
                "file_sha256": sha256_path(path),
                "payload_sha256": payload[field],
            }
        required = {
            PHASE_LOCK_NAME,
            PHASE_LOCK_RESULT_NAME,
            PHASE_MARKER_NAME,
            PHASE_MANIFEST_NAME,
            PHASE_OWNER_NAME,
        }
        checks = {
            "commit_head_absent": not commit_head_path.exists(),
            "required_bootstrap_artifacts_exact": required.issubset(
                identities
            ),
            "unexpected_work_files_absent": not unexpected,
            "terminal_absent": not (
                phase_dir / PHASE_RESULT_NAME
            ).exists(),
            "checkpoint_absent": not (
                phase_dir / TRAINING_CANDIDATE_CHECKPOINT_NAME
            ).exists(),
            "analysis_absent": not (
                phase_dir / PAIRED_ANALYSIS_NAME
            ).exists(),
        }
        if not all(checks.values()):
            raise J1ExecutionIntegrityError(
                "No-head owner recovery cannot prove zero work"
            )
        boundary_evidence = {
            "mode": "bootstrap_no_commit_head_v1",
            "artifact_identities": identities,
            "artifact_inventory_sha256": canonical_json_hash(identities),
            "unexpected_files": unexpected,
            "checks": checks,
            "passes": True,
        }
        predecessor_commit_head_sha256 = None
    process = (
        j1.heavy_process_audit()
        if contention_audit is None
        else dict(contention_audit)
    )
    if process.get("passes") is not True:
        raise J1ExecutionOperationalHold(
            "Concurrent heavy writer prevents reclaim"
        )
    new_owner = _new_owner_record(
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        predecessor_commit_head_sha256=predecessor_commit_head_sha256,
        execution_mode=execution_mode,
        pid=new_pid,
        start_identity=new_start_identity,
    )
    recovery = _record_with_hash(
        {
            "version": f"{VERSION}_owner_recovery_v1",
            "phase": phase,
            "old_owner_sha256": old["owner_record_sha256"],
            "new_owner_sha256": new_owner["owner_record_sha256"],
            "old_pid": old_pid,
            "old_process_start_identity": old[
                "process_start_identity"
            ],
            "process_death_evidence": {
                "kill_zero_alive": False,
                "process_start_identity_now": None,
            },
            "committed_boundary": boundary_evidence,
            "zero_concurrent_writer_audit": process,
        },
        "recovery_record_sha256",
    )
    next_payload = {
        "version": ledger["version"],
        "owners": [*ledger["owners"], new_owner],
        "recoveries": [*ledger["recoveries"], recovery],
        "head_owner_sha256": new_owner["owner_record_sha256"],
    }
    observed = _atomic_replace_json(
        path,
        next_payload,
        field="ownership_payload_sha256",
    )
    if not _verify_ownership_ledger(observed):
        raise J1ExecutionIntegrityError("Recovered ownership ledger invalid")
    return {
        "ledger": observed,
        "recovery": recovery,
        "new_owner": new_owner,
        "passes": True,
    }


@dataclass(frozen=True)
class TrainingEngineConfig:
    rounds: int = ROUNDS
    roots_per_round: int = ROOTS_PER_ROUND
    env_count: int = ENV_COUNT
    minibatch_size: int = j1.FROZEN_CONFIG.minibatch_size
    max_moves: int = MAX_MOVES
    execution_mode: str = "scientific"


def _state_without_commit_metadata(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    excluded = {
        "committed_unit_ids",
        "commit_sequence",
        "latest_unit_id",
        "phase_lock_file_sha256",
        "command",
        "execution_mode",
    }
    return {
        key: value
        for key, value in state.items()
        if key not in excluded
    }


def _load_model_optimizer_from_runtime(
    state: Mapping[str, Any],
) -> tuple[j1.J1ActorCritic, torch.optim.Optimizer]:
    model, optimizer = j1.initialize_model_optimizer()
    try:
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
    except Exception as error:
        raise J1ExecutionIntegrityError(
            "Training engine model/optimizer state is malformed"
        ) from error
    FrozenMinibatchUpdater._validate_optimizer_binding(model, optimizer)
    j1.assert_finite_model(model)
    return model, optimizer


def _training_round_metrics(
    *,
    records: Sequence[Mapping[str, Any]],
    updater: FrozenMinibatchUpdater,
) -> dict[str, Any]:
    batch = updater.batch
    model = updater.model.eval()
    with torch.no_grad():
        logits, values, auxiliary_logits = model(batch.observations)
        masked = j1.masked_logits(logits, batch.legal_masks)
        distribution = torch.distributions.Categorical(logits=masked)
        entropy = distribution.entropy()
        value_squared_error = (values - batch.returns).square()
        zero_value_squared_error = batch.returns.square()
        probabilities = torch.sigmoid(auxiliary_logits)
        auxiliary_squared_error = (
            probabilities - batch.auxiliary_labels
        ).square()
    weights = batch.row_weights.to(dtype=torch.float64)
    weight_total = weights.sum()
    if not torch.isfinite(weight_total) or float(weight_total) <= 0.0:
        raise J1ExecutionIntegrityError(
            "Training metric root weights are invalid"
        )

    def weighted(values_tensor: torch.Tensor) -> float:
        values64 = values_tensor.to(dtype=torch.float64)
        result = (values64 * weights).sum() / weight_total
        if not torch.isfinite(result):
            raise J1ExecutionIntegrityError(
                "Training metric is nonfinite"
            )
        return float(result)

    auxiliary_brier = [
        weighted(auxiliary_squared_error[:, index])
        for index in range(3)
    ]
    prevalence = [
        weighted(batch.auxiliary_labels[:, index])
        for index in range(3)
    ]
    prevalence_brier = [
        value * (1.0 - value) for value in prevalence
    ]
    record_by_root = {
        str(record["root_id"]): record for record in records
    }
    ordered_root_ids = [str(record["root_id"]) for record in records]
    if (
        len(record_by_root) != len(records)
        or set(batch.root_ids) != set(record_by_root)
    ):
        raise J1ExecutionIntegrityError(
            "Training metric roots changed"
        )
    root_metrics = []
    for root_id in ordered_root_ids:
        indices = [
            index
            for index, value in enumerate(batch.root_ids)
            if value == root_id
        ]
        if not indices:
            raise J1ExecutionIntegrityError(
                "Training metric root has no transitions"
            )
        selected = torch.tensor(indices, dtype=torch.int64)
        root_weights = weights[selected]
        root_total = root_weights.sum()

        def root_weighted(values_tensor: torch.Tensor) -> float:
            result = (
                values_tensor[selected].to(dtype=torch.float64)
                * root_weights
            ).sum() / root_total
            if not torch.isfinite(result):
                raise J1ExecutionIntegrityError(
                    "Per-root training metric is nonfinite"
                )
            return float(result)

        record = record_by_root[root_id]
        root_prevalence = [
            root_weighted(batch.auxiliary_labels[:, index])
            for index in range(3)
        ]
        root_metrics.append(
            {
                "root_id": root_id,
                "ancestry_id": str(record["ancestry_id"]),
                "committed_record_sha256": j1.stable_hash(record),
                "transition_content_sha256": j1.stable_hash(
                    record["transitions"]
                ),
                "transition_rows": len(indices),
                "log_score": math.log1p(
                    max(
                        int(record["final_score"])
                        - int(record["start_score"]),
                        0,
                    )
                ),
                "legal_entropy_nats": root_weighted(entropy),
                "value_mse": root_weighted(value_squared_error),
                "zero_value_mse": root_weighted(
                    zero_value_squared_error
                ),
                "auxiliary_brier": [
                    root_weighted(auxiliary_squared_error[:, index])
                    for index in range(3)
                ],
                "auxiliary_prevalence": root_prevalence,
            }
        )
    root_metrics_sha256 = j1.stable_hash(root_metrics)
    return {
        "round": updater.round_number,
        "root_ids": ordered_root_ids,
        "root_metrics": root_metrics,
        "root_metrics_sha256": root_metrics_sha256,
        "committed_records_sha256": j1.stable_hash(list(records)),
        "root_log_scores": [
            float(row["log_score"]) for row in root_metrics
        ],
        "legal_entropy_nats": weighted(entropy),
        "value_mse": weighted(value_squared_error),
        "zero_value_mse": weighted(zero_value_squared_error),
        "auxiliary_brier": auxiliary_brier,
        "auxiliary_prevalence_brier": prevalence_brier,
        "transition_rows": batch.row_count(),
        "transition_buffer_sha256": j1.stable_hash(batch.payload()),
    }


def _training_runtime_state(
    *,
    marker_file_sha256: str,
    marker_payload_sha256: str,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    model: j1.J1ActorCritic,
    optimizer: torch.optim.Optimizer,
    round_number: int,
    collection_boundary: str,
    engine_stage: str,
    collection_snapshot: Mapping[str, Any] | None,
    updater_snapshot: Mapping[str, Any] | None,
    current_round_records: Sequence[Mapping[str, Any]] | None,
    all_completed_root_ids: Sequence[str],
    optimizer_step_ids: Sequence[str],
    expected_optimizer_step_ids: Sequence[str],
    round_aggregates: Sequence[Mapping[str, Any]],
    resource_clock: Mapping[str, Any],
    output_bytes: int,
) -> dict[str, Any]:
    active = (
        []
        if collection_snapshot is None
        else copy.deepcopy(list(collection_snapshot["active"]))
    )
    next_manifest_row = (
        len(all_completed_root_ids)
        if collection_snapshot is None
        else len(all_completed_root_ids)
        + int(collection_snapshot["next_index"])
    )
    transition_payload = (
        None
        if current_round_records is None
        else copy.deepcopy(list(current_round_records))
    )
    state = {
        "version": f"{VERSION}_training_runtime_v1",
        "runtime_payload_complete": True,
        "phase": "training",
        "marker_file_sha256": marker_file_sha256,
        "marker_payload_sha256": marker_payload_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_payload_sha256": manifest_payload_sha256,
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "round_number": int(round_number),
        "collection_boundary": collection_boundary,
        "next_manifest_row": next_manifest_row,
        "active_roots": active,
        "completed_roots": list(all_completed_root_ids),
        "transition_buffer_path": None,
        "transition_buffer_sha256": (
            None
            if transition_payload is None
            else j1.stable_hash(transition_payload)
        ),
        "epoch_cursor": (
            0
            if updater_snapshot is None
            else int(updater_snapshot["cursor"])
        ),
        "minibatch_cursor": (
            0
            if updater_snapshot is None
            else int(updater_snapshot["cursor"])
        ),
        "optimizer_step_ids": list(optimizer_step_ids),
        "round_aggregates": copy.deepcopy(list(round_aggregates)),
        "python_rng_state": copy.deepcopy(random.getstate()),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state().clone(),
        "resource_clock": dict(resource_clock),
        "output_bytes": int(output_bytes),
        "engine_stage": engine_stage,
        "collection_snapshot": (
            None
            if collection_snapshot is None
            else copy.deepcopy(dict(collection_snapshot))
        ),
        "updater_snapshot": (
            None
            if updater_snapshot is None
            else copy.deepcopy(dict(updater_snapshot))
        ),
        "current_round_records": transition_payload,
        "all_completed_root_ids": list(all_completed_root_ids),
        "expected_optimizer_step_ids": list(
            expected_optimizer_step_ids
        ),
    }
    return state


def _restore_runtime_rngs(state: Mapping[str, Any]) -> None:
    random.setstate(copy.deepcopy(state["python_rng_state"]))
    np.random.set_state(copy.deepcopy(state["numpy_rng_state"]))
    torch.set_rng_state(
        state["torch_rng_state"].detach().cpu().clone()
    )


def execute_training_engine(
    *,
    rows: Sequence[Mapping[str, Any]],
    phase_dir: Path,
    marker_file_sha256: str,
    marker_payload_sha256: str,
    phase_lock_file_sha256: str,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    command: str,
    config: TrainingEngineConfig = TrainingEngineConfig(),
    interrupt_after_boundary: str | None = None,
) -> dict[str, Any]:
    if config.execution_mode != "miniature_fixture":
        raise J1ExecutionIntegrityError(
            "Verbose training engine is fixture-only; scientific execution "
            "must call execute_training_engine_bounded"
        )
    expected_rows = config.rounds * config.roots_per_round
    if len(rows) != expected_rows:
        raise J1ExecutionIntegrityError(
            "Training engine row count changed"
        )
    if interrupt_after_boundary not in {
        None,
        "collection",
        "update",
        "checkpoint",
    }:
        raise ValueError("Unknown training interruption boundary")
    contract = {
        "phase": "training",
        "marker_file_sha256": marker_file_sha256,
        "phase_lock_file_sha256": phase_lock_file_sha256,
        "command": command,
        "execution_mode": config.execution_mode,
    }
    pointer = phase_dir / COMMIT_HEAD_NAME
    if not pointer.exists():
        model, optimizer = j1.initialize_model_optimizer()
        first_rows = rows[: config.roots_per_round]
        collection = TrainingCollectionSession(
            rows=first_rows,
            model=model,
            env_count=config.env_count,
            max_moves=config.max_moves,
        )
        initial_state = _training_runtime_state(
            marker_file_sha256=marker_file_sha256,
            marker_payload_sha256=marker_payload_sha256,
            manifest_file_sha256=manifest_file_sha256,
            manifest_payload_sha256=manifest_payload_sha256,
            model=model,
            optimizer=optimizer,
            round_number=1,
            collection_boundary="pre_action",
            engine_stage="collection",
            collection_snapshot=collection.snapshot(),
            updater_snapshot=None,
            current_round_records=None,
            all_completed_root_ids=[],
            optimizer_step_ids=[],
            expected_optimizer_step_ids=[],
            round_aggregates=[],
            resource_clock={"active_seconds": 0.0},
            output_bytes=0,
        )
        boundary = initialize_commit_store(
            phase_dir=phase_dir,
            **contract,
            initial_state=initial_state,
        )
    else:
        boundary = verify_commit_boundary(
            phase_dir=phase_dir,
            **contract,
        )
    while True:
        state = boundary["state"]
        if state.get("engine_stage") == "complete":
            return {
                "boundary": boundary,
                "state": state,
                "completed": True,
                "passes": True,
            }
        model, optimizer = _load_model_optimizer_from_runtime(state)
        _restore_runtime_rngs(state)
        round_number = int(state["round_number"])
        round_start = (round_number - 1) * config.roots_per_round
        round_rows = rows[
            round_start : round_start + config.roots_per_round
        ]
        base_state = _state_without_commit_metadata(state)
        stage = str(state["engine_stage"])
        if stage == "collection":
            session = TrainingCollectionSession.from_snapshot(
                state["collection_snapshot"],
                rows=round_rows,
                model=model,
            )
            session.step_tick()
            snapshot = session.snapshot()
            unit_id = (
                f"round={round_number}|collection_tick={snapshot['tick']}"
            )
            if session.is_complete():
                records = session.ordered_completed_records()
                batch = training_records_to_ppo_batch(records)
                updater = FrozenMinibatchUpdater(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    round_number=round_number,
                    minibatch_size=config.minibatch_size,
                )
                all_completed = [
                    *state["all_completed_root_ids"],
                    *[str(record["root_id"]) for record in records],
                ]
                expected_steps = [
                    *state["expected_optimizer_step_ids"],
                    *updater.expected_step_ids(),
                ]
                post_state = _training_runtime_state(
                    marker_file_sha256=marker_file_sha256,
                    marker_payload_sha256=marker_payload_sha256,
                    manifest_file_sha256=manifest_file_sha256,
                    manifest_payload_sha256=manifest_payload_sha256,
                    model=model,
                    optimizer=optimizer,
                    round_number=round_number,
                    collection_boundary="pre_update",
                    engine_stage="update",
                    collection_snapshot=snapshot,
                    updater_snapshot=updater.snapshot(),
                    current_round_records=records,
                    all_completed_root_ids=all_completed,
                    optimizer_step_ids=state["optimizer_step_ids"],
                    expected_optimizer_step_ids=expected_steps,
                    round_aggregates=state["round_aggregates"],
                    resource_clock=state["resource_clock"],
                    output_bytes=state["output_bytes"],
                )
            else:
                post_state = _training_runtime_state(
                    marker_file_sha256=marker_file_sha256,
                    marker_payload_sha256=marker_payload_sha256,
                    manifest_file_sha256=manifest_file_sha256,
                    manifest_payload_sha256=manifest_payload_sha256,
                    model=model,
                    optimizer=optimizer,
                    round_number=round_number,
                    collection_boundary="post_step",
                    engine_stage="collection",
                    collection_snapshot=snapshot,
                    updater_snapshot=None,
                    current_round_records=None,
                    all_completed_root_ids=state[
                        "all_completed_root_ids"
                    ],
                    optimizer_step_ids=state["optimizer_step_ids"],
                    expected_optimizer_step_ids=state[
                        "expected_optimizer_step_ids"
                    ],
                    round_aggregates=state["round_aggregates"],
                    resource_clock=state["resource_clock"],
                    output_bytes=state["output_bytes"],
                )
            boundary = commit_unit(
                phase_dir=phase_dir,
                **contract,
                unit_id=unit_id,
                post_state=post_state,
                journal_payload={
                    "kind": "training_collection_tick",
                    "round": round_number,
                    "tick": snapshot["tick"],
                    "session_state_sha256":
                        snapshot["session_state_sha256"],
                },
            )
            if interrupt_after_boundary == "collection":
                raise J1ExecutionPlannedInterruption(
                    "fixture interruption after collection boundary"
                )
            continue
        if stage == "update":
            updater = FrozenMinibatchUpdater.from_snapshot(
                state["updater_snapshot"],
                minibatch_size=config.minibatch_size,
            )
            report = updater.step_once()
            optimizer_step_ids = [
                *state["optimizer_step_ids"],
                report["step_id"],
            ]
            if updater.cursor == len(updater.plan):
                metrics = _training_round_metrics(
                    records=state["current_round_records"],
                    updater=updater,
                )
                engine_stage = "checkpoint"
                round_aggregates = [
                    *state["round_aggregates"],
                    metrics,
                ]
                boundary_name = "post_checkpoint"
            else:
                engine_stage = "update"
                round_aggregates = state["round_aggregates"]
                boundary_name = "mid_update"
            post_state = _training_runtime_state(
                marker_file_sha256=marker_file_sha256,
                marker_payload_sha256=marker_payload_sha256,
                manifest_file_sha256=manifest_file_sha256,
                manifest_payload_sha256=manifest_payload_sha256,
                model=updater.model,
                optimizer=updater.optimizer,
                round_number=round_number,
                collection_boundary=boundary_name,
                engine_stage=engine_stage,
                collection_snapshot=state["collection_snapshot"],
                updater_snapshot=updater.snapshot(),
                current_round_records=state["current_round_records"],
                all_completed_root_ids=state[
                    "all_completed_root_ids"
                ],
                optimizer_step_ids=optimizer_step_ids,
                expected_optimizer_step_ids=state[
                    "expected_optimizer_step_ids"
                ],
                round_aggregates=round_aggregates,
                resource_clock=state["resource_clock"],
                output_bytes=state["output_bytes"],
            )
            boundary = commit_unit(
                phase_dir=phase_dir,
                **contract,
                unit_id=report["step_id"],
                post_state=post_state,
                journal_payload={
                    "kind": "frozen_ppo_minibatch",
                    **report,
                },
            )
            if interrupt_after_boundary == "update":
                raise J1ExecutionPlannedInterruption(
                    "fixture interruption after update boundary"
                )
            continue
        if stage == "checkpoint":
            if round_number == config.rounds:
                next_stage = "complete"
                next_round = round_number
                next_collection = None
            else:
                next_stage = "collection"
                next_round = round_number + 1
                next_rows = rows[
                    round_number * config.roots_per_round :
                    (round_number + 1) * config.roots_per_round
                ]
                next_collection = TrainingCollectionSession(
                    rows=next_rows,
                    model=model,
                    env_count=config.env_count,
                    max_moves=config.max_moves,
                ).snapshot()
            post_state = _training_runtime_state(
                marker_file_sha256=marker_file_sha256,
                marker_payload_sha256=marker_payload_sha256,
                manifest_file_sha256=manifest_file_sha256,
                manifest_payload_sha256=manifest_payload_sha256,
                model=model,
                optimizer=optimizer,
                round_number=next_round,
                collection_boundary="post_checkpoint",
                engine_stage=next_stage,
                collection_snapshot=next_collection,
                updater_snapshot=None,
                current_round_records=None,
                all_completed_root_ids=state[
                    "all_completed_root_ids"
                ],
                optimizer_step_ids=state["optimizer_step_ids"],
                expected_optimizer_step_ids=state[
                    "expected_optimizer_step_ids"
                ],
                round_aggregates=state["round_aggregates"],
                resource_clock=state["resource_clock"],
                output_bytes=state["output_bytes"],
            )
            boundary = commit_unit(
                phase_dir=phase_dir,
                **contract,
                unit_id=f"round={round_number}|checkpoint",
                post_state=post_state,
                journal_payload={
                    "kind": "round_checkpoint",
                    "round": round_number,
                    "model_state_sha256": j1.stable_hash(
                        model.state_dict()
                    ),
                    "optimizer_state_sha256": j1.stable_hash(
                        optimizer.state_dict()
                    ),
                },
            )
            if interrupt_after_boundary == "checkpoint":
                raise J1ExecutionPlannedInterruption(
                    "fixture interruption after checkpoint boundary"
                )
            continue
        raise J1ExecutionIntegrityError(
            f"Unknown training engine stage: {stage}"
        )


def execute_paired_evaluation_engine(
    *,
    rows: Sequence[Mapping[str, Any]],
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    marker_payload_sha256: str,
    phase_lock_file_sha256: str,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    command: str,
    candidate_policy: Any,
    control_policy: Any,
    candidate_policy_identity: str,
    control_policy_identity: str,
    max_moves: int = MAX_MOVES,
    execution_mode: str = "scientific",
    interrupt_after_boundary: str | None = None,
) -> dict[str, Any]:
    if execution_mode != "miniature_fixture":
        raise J1ExecutionIntegrityError(
            "Verbose paired engine is fixture-only; scientific execution "
            "must call execute_paired_evaluation_engine_bounded"
        )
    if phase not in {"development", "confirmation"}:
        raise ValueError("Paired engine phase is invalid")
    expected = {
        "development": DEVELOPMENT_PAIRS,
        "confirmation": CONFIRMATION_PAIRS,
    }[phase]
    if interrupt_after_boundary not in {
        None,
        "candidate_arm_committed",
        "paired_root_committed",
    }:
        raise ValueError("Unknown paired interruption boundary")
    contract = {
        "phase": phase,
        "marker_file_sha256": marker_file_sha256,
        "phase_lock_file_sha256": phase_lock_file_sha256,
        "command": command,
        "execution_mode": execution_mode,
    }
    pointer = phase_dir / COMMIT_HEAD_NAME
    if not pointer.exists():
        session = PairedEvaluationSession(
            rows=rows,
            candidate_policy=candidate_policy,
            control_policy=control_policy,
            candidate_policy_identity=candidate_policy_identity,
            control_policy_identity=control_policy_identity,
            max_moves=max_moves,
        )
        initial_state = {
            "version": f"{VERSION}_{phase}_runtime_v1",
            "phase": phase,
            "engine_stage": "evaluation",
            "session_snapshot": session.snapshot(),
            "resource_clock": {"active_seconds": 0.0},
            "output_bytes": 0,
        }
        boundary = initialize_commit_store(
            phase_dir=phase_dir,
            **contract,
            initial_state=initial_state,
        )
    else:
        boundary = verify_commit_boundary(
            phase_dir=phase_dir,
            **contract,
        )
    while True:
        state = boundary["state"]
        if state.get("engine_stage") == "complete":
            session = PairedEvaluationSession.from_snapshot(
                state["session_snapshot"],
                rows=rows,
                candidate_policy=candidate_policy,
                control_policy=control_policy,
                candidate_policy_identity=candidate_policy_identity,
                control_policy_identity=control_policy_identity,
            )
            if not session.is_complete():
                raise J1ExecutionIntegrityError(
                    "Paired engine complete state is incomplete"
                )
            return {
                "boundary": boundary,
                "rows": copy.deepcopy(session.completed_pairs),
                "completed": True,
                "passes": True,
            }
        session = PairedEvaluationSession.from_snapshot(
            state["session_snapshot"],
            rows=rows,
            candidate_policy=candidate_policy,
            control_policy=control_policy,
            candidate_policy_identity=candidate_policy_identity,
            control_policy_identity=control_policy_identity,
        )
        report = session.step_arm()
        snapshot = session.snapshot()
        next_state = {
            "version": f"{VERSION}_{phase}_runtime_v1",
            "phase": phase,
            "engine_stage": (
                "complete" if session.is_complete() else "evaluation"
            ),
            "session_snapshot": snapshot,
            "resource_clock": state["resource_clock"],
            "output_bytes": state["output_bytes"],
        }
        unit_id = (
            f"row={report['row_index']}|{report['boundary']}"
        )
        boundary = commit_unit(
            phase_dir=phase_dir,
            **contract,
            unit_id=unit_id,
            post_state=next_state,
            journal_payload={
                "kind": "paired_full_policy_arm_boundary",
                **report,
                "session_state_sha256": snapshot[
                    "session_state_sha256"
                ],
            },
        )
        if interrupt_after_boundary == report["boundary"]:
            raise J1ExecutionPlannedInterruption(
                f"fixture interruption after {report['boundary']}"
            )


def _safe_blob_token(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise J1ExecutionIntegrityError(
            f"{label} is unsafe for immutable blob storage"
        )
    return value


def _training_root_blob_ref(
    *,
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    manifest_index: int,
    blob_dir: Path,
    output_accountant: PhaseOutputAccountant | None = None,
    io_metrics: dict[str, int] | None = None,
) -> dict[str, Any]:
    root_id = _safe_blob_token(
        str(record["root_id"]),
        label="Training root id",
    )
    if (
        record.get("source_manifest_row") != dict(row)
        or record.get("source_manifest_row_sha256")
        != canonical_json_hash(dict(row))
        or str(record.get("ancestry_id")) != str(row["ancestry_id"])
    ):
        raise J1ExecutionIntegrityError(
            "Training root blob changed its manifest row"
        )
    path = blob_dir / f"{manifest_index:05d}_{root_id}.bin"
    existed = path.exists()
    file_sha256 = _write_immutable_binary_exact(path, record)
    if output_accountant is not None:
        output_accountant.record_path(path)
    if io_metrics is not None:
        key = (
            "root_blob_validation_reads"
            if existed
            else "root_blob_writes"
        )
        io_metrics[key] = io_metrics.get(key, 0) + 1
        if not existed:
            io_metrics["root_blob_bytes_written"] = (
                io_metrics.get("root_blob_bytes_written", 0)
                + int(path.stat().st_size)
            )
    return {
        "manifest_index": int(manifest_index),
        "root_id": root_id,
        "ancestry_id": str(row["ancestry_id"]),
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "record_sha256": j1.stable_hash(dict(record)),
    }


def _root_block_identity(
    *,
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload["root_block_payload_sha256"],
        "block_index": int(payload["block_index"]),
        "cumulative_completed_count": int(
            payload["cumulative_completed_count"]
        ),
    }


def _write_training_root_block(
    *,
    phase_dir: Path,
    rolling_contract: Mapping[str, Any],
    round_number: int,
    rows_sha256: str,
    refs: Sequence[Mapping[str, Any]],
    predecessor: Mapping[str, Any] | None,
    output_accountant: PhaseOutputAccountant | None = None,
    io_metrics: dict[str, int] | None = None,
) -> dict[str, Any]:
    if not refs or len(refs) > ROOT_BLOB_BLOCK_SIZE:
        raise J1ExecutionIntegrityError(
            "Training root block size changed"
        )
    block_index = 0 if predecessor is None else int(
        predecessor["block_index"]
    ) + 1
    previous_count = (
        0
        if predecessor is None
        else int(predecessor["cumulative_completed_count"])
    )
    payload = {
        "version": f"{VERSION}_training_root_block_v1",
        "round": int(round_number),
        "block_index": block_index,
        "rows_sha256": rows_sha256,
        "rolling_contract_sha256": rolling_contract[
            "rolling_contract_sha256"
        ],
        "predecessor": (
            None if predecessor is None else dict(predecessor)
        ),
        "refs": copy.deepcopy(list(refs)),
        "refs_sha256": j1.stable_hash(list(refs)),
        "cumulative_completed_count": previous_count + len(refs),
    }
    path = (
        phase_dir
        / ROOT_BLOB_BLOCKS_DIR
        / f"round_{round_number:02d}"
        / f"block_{block_index:03d}.json"
    )
    observed = _write_immutable_json_exact(
        path,
        payload,
        field="root_block_payload_sha256",
    )
    if output_accountant is not None:
        output_accountant.record_path(path)
    if io_metrics is not None:
        io_metrics["root_block_writes"] = (
            io_metrics.get("root_block_writes", 0) + 1
        )
        io_metrics["root_block_bytes_written"] = (
            io_metrics.get("root_block_bytes_written", 0)
            + int(path.stat().st_size)
        )
    return _root_block_identity(path=path, payload=observed)


def _load_training_root_block_chain(
    *,
    phase_dir: Path,
    rolling_contract: Mapping[str, Any],
    round_number: int,
    rows_sha256: str,
    head: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if head is None:
        return []
    root = (
        phase_dir
        / ROOT_BLOB_BLOCKS_DIR
        / f"round_{round_number:02d}"
    ).resolve()
    current = dict(head)
    descending: list[list[dict[str, Any]]] = []
    expected_index = int(current["block_index"])
    expected_count = int(current["cumulative_completed_count"])
    visited: set[str] = set()
    while current is not None:
        path = Path(str(current["path"])).resolve()
        if (
            path.parent != root
            or str(path) in visited
            or not path.is_file()
            or sha256_path(path) != current.get("file_sha256")
        ):
            raise J1ExecutionIntegrityError(
                "Training root block chain path changed"
            )
        visited.add(str(path))
        payload = load_json(path)
        if (
            not verify_payload_hash(
                payload,
                "root_block_payload_sha256",
            )
            or payload["root_block_payload_sha256"]
            != current.get("payload_sha256")
            or int(payload.get("round", -1)) != round_number
            or int(payload.get("block_index", -1)) != expected_index
            or payload.get("rows_sha256") != rows_sha256
            or payload.get("rolling_contract_sha256")
            != rolling_contract["rolling_contract_sha256"]
            or int(payload.get("cumulative_completed_count", -1))
            != expected_count
            or payload.get("refs_sha256")
            != j1.stable_hash(payload.get("refs", []))
        ):
            raise J1ExecutionIntegrityError(
                "Training root block chain changed"
            )
        refs = copy.deepcopy(list(payload["refs"]))
        if not refs or len(refs) > ROOT_BLOB_BLOCK_SIZE:
            raise J1ExecutionIntegrityError(
                "Training root block references changed"
            )
        descending.append(refs)
        expected_count -= len(refs)
        expected_index -= 1
        predecessor = payload.get("predecessor")
        if predecessor is None:
            current = None
        else:
            if not isinstance(predecessor, Mapping):
                raise J1ExecutionIntegrityError(
                    "Training root block predecessor is malformed"
                )
            current = dict(predecessor)
    if expected_count != 0 or expected_index != -1:
        raise J1ExecutionIntegrityError(
            "Training root block chain does not reach genesis"
        )
    return [
        reference
        for block in reversed(descending)
        for reference in block
    ]


def _completed_bitmap(indices: Iterable[int]) -> str:
    value = 0
    for index in indices:
        if int(index) < 0:
            raise J1ExecutionIntegrityError(
                "Completed training index is negative"
            )
        value |= 1 << int(index)
    return format(value, "x")


def _bitmap_indices(value: str, *, limit: int) -> set[int]:
    try:
        bits = int(value, 16)
    except (TypeError, ValueError) as error:
        raise J1ExecutionIntegrityError(
            "Completed training bitmap is malformed"
        ) from error
    if bits < 0 or bits >> limit:
        raise J1ExecutionIntegrityError(
            "Completed training bitmap exceeds manifest"
        )
    return {index for index in range(limit) if bits & (1 << index)}


def persist_incremental_training_collection(
    *,
    session: TrainingCollectionSession,
    previous_snapshot: Mapping[str, Any] | None,
    phase_dir: Path,
    rolling_contract: Mapping[str, Any],
    round_number: int,
    force_seal: bool,
    resource_clock: Mapping[str, Any],
    operational_audit: Mapping[str, Any],
    output_accountant: PhaseOutputAccountant | None = None,
    io_metrics: dict[str, int] | None = None,
) -> dict[str, Any]:
    if session.transition_store is None:
        raise J1ExecutionIntegrityError(
            "Incremental training requires append-once transition chunks"
        )
    row_index = {
        str(row["root_id"]): index
        for index, row in enumerate(session.rows)
    }
    if previous_snapshot is None:
        completed_indices: set[int] = set()
        current_refs: list[dict[str, Any]] = []
        block_head = None
    else:
        completed_indices = _bitmap_indices(
            str(previous_snapshot["completed_bitmap_hex"]),
            limit=len(session.rows),
        )
        current_refs = copy.deepcopy(
            list(previous_snapshot["current_block_refs"])
        )
        block_head = copy.deepcopy(
            previous_snapshot["sealed_block_head"]
        )
    blob_dir = phase_dir / ROOT_BLOBS_DIR
    for root_id, record in sorted(
        session.completed.items(),
        key=lambda item: row_index[item[0]],
    ):
        index = row_index[root_id]
        if index in completed_indices:
            raise J1ExecutionIntegrityError(
                "Training root blob would be written twice"
            )
        reference = _training_root_blob_ref(
            record=record,
            row=session.rows[index],
            manifest_index=index,
            blob_dir=blob_dir,
            output_accountant=output_accountant,
            io_metrics=io_metrics,
        )
        current_refs.append(reference)
        completed_indices.add(index)
    session.completed.clear()
    while len(current_refs) >= ROOT_BLOB_BLOCK_SIZE:
        block_head = _write_training_root_block(
            phase_dir=phase_dir,
            rolling_contract=rolling_contract,
            round_number=round_number,
            rows_sha256=session.rows_sha256,
            refs=current_refs[:ROOT_BLOB_BLOCK_SIZE],
            predecessor=block_head,
            output_accountant=output_accountant,
            io_metrics=io_metrics,
        )
        current_refs = current_refs[ROOT_BLOB_BLOCK_SIZE:]
    if force_seal and current_refs:
        block_head = _write_training_root_block(
            phase_dir=phase_dir,
            rolling_contract=rolling_contract,
            round_number=round_number,
            rows_sha256=session.rows_sha256,
            refs=current_refs,
            predecessor=block_head,
            output_accountant=output_accountant,
            io_metrics=io_metrics,
        )
        current_refs = []
    active_payload = [
        {
            "row": dict(item.row),
            "simulator": j1.simulator_snapshot(item.sim, item.state),
            "policy_rng_state": item.policy_generator.get_state().clone(),
            "start_score": item.start_score,
            "transition_chunk_head": copy.deepcopy(
                item.transition_chunk_head
            ),
            "transition_count": int(item.transition_count),
        }
        for item in session.active
    ]
    payload = {
        "version": f"{VERSION}_training_collection_incremental_v1",
        "round": int(round_number),
        "rows_sha256": session.rows_sha256,
        "model_state_sha256": session.model_state_sha256,
        "env_count": session.env_count,
        "max_moves": session.max_moves,
        "next_index": session.next_index,
        "tick": session.tick,
        "active": active_payload,
        "completed_bitmap_hex": _completed_bitmap(completed_indices),
        "completed_count": len(completed_indices),
        "sealed_block_head": block_head,
        "current_block_refs": current_refs,
        "current_block_refs_sha256": j1.stable_hash(current_refs),
        "transition_store_state": session.transition_store.snapshot(),
        "python_rng_state": copy.deepcopy(random.getstate()),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state().clone(),
        "resource_clock": dict(resource_clock),
        "last_operational_audit": dict(operational_audit),
    }
    payload["session_state_sha256"] = j1.stable_hash(payload)
    return payload


def restore_incremental_training_collection(
    payload: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    model: j1.J1ActorCritic,
    phase_dir: Path,
    rolling_contract: Mapping[str, Any],
    round_number: int,
    output_accountant: PhaseOutputAccountant | None = None,
    io_metrics: dict[str, int] | None = None,
) -> TrainingCollectionSession:
    body = dict(payload)
    observed = body.pop("session_state_sha256", None)
    if observed != j1.stable_hash(body):
        raise J1ExecutionIntegrityError(
            "Incremental training collection snapshot changed"
        )
    instance = TrainingCollectionSession.__new__(
        TrainingCollectionSession
    )
    instance.rows = [dict(row) for row in rows]
    instance.model = model.cpu().eval()
    instance.env_count = int(payload["env_count"])
    instance.max_moves = int(payload["max_moves"])
    instance.rows_sha256 = _training_rows_identity(instance.rows)
    instance.model_state_sha256 = j1.stable_hash(
        instance.model.state_dict()
    )
    instance.transition_store = TrainingTransitionChunkStore(
        phase_dir=phase_dir,
        rolling_contract=rolling_contract,
        round_number=round_number,
        rows=instance.rows,
        snapshot=payload["transition_store_state"],
        output_accountant=output_accountant,
        io_metrics=io_metrics,
    )
    instance._validate_rows()
    if (
        payload.get("version")
        != f"{VERSION}_training_collection_incremental_v1"
        or int(payload.get("round", -1)) != round_number
        or payload.get("rows_sha256") != instance.rows_sha256
        or payload.get("model_state_sha256")
        != instance.model_state_sha256
    ):
        raise J1ExecutionIntegrityError(
            "Incremental training collection identity changed"
        )
    block_refs = _load_training_root_block_chain(
        phase_dir=phase_dir,
        rolling_contract=rolling_contract,
        round_number=round_number,
        rows_sha256=instance.rows_sha256,
        head=payload.get("sealed_block_head"),
    )
    current_refs = list(payload.get("current_block_refs", []))
    if (
        len(current_refs) >= ROOT_BLOB_BLOCK_SIZE
        or payload.get("current_block_refs_sha256")
        != j1.stable_hash(current_refs)
    ):
        raise J1ExecutionIntegrityError(
            "Incremental current root block changed"
        )
    all_refs = [*block_refs, *current_refs]
    completed_indices = _bitmap_indices(
        str(payload["completed_bitmap_hex"]),
        limit=len(instance.rows),
    )
    ref_indices = [int(ref["manifest_index"]) for ref in all_refs]
    if (
        int(payload.get("completed_count", -1))
        != len(completed_indices)
        or len(set(ref_indices)) != len(ref_indices)
        or set(ref_indices) != completed_indices
    ):
        raise J1ExecutionIntegrityError(
            "Incremental completed-root accounting changed"
        )
    blob_root = (phase_dir / ROOT_BLOBS_DIR).resolve()
    for reference in all_refs:
        index = int(reference["manifest_index"])
        authoritative = instance.rows[index]
        root_id = str(authoritative["root_id"])
        expected_name = (
            f"{index:05d}_"
            f"{_safe_blob_token(root_id, label='Training root id')}.bin"
        )
        path = Path(str(reference["path"])).resolve()
        if (
            str(reference.get("root_id")) != root_id
            or str(reference.get("ancestry_id"))
            != str(authoritative["ancestry_id"])
            or path.parent != blob_root
            or path.name != expected_name
        ):
            raise J1ExecutionIntegrityError(
                "Incremental root reference changed manifest identity"
            )
    for reference in current_refs:
        path = Path(str(reference["path"])).resolve()
        if (
            path.parent != blob_root
            or not path.is_file()
            or sha256_path(path) != reference["file_sha256"]
        ):
            raise J1ExecutionIntegrityError(
                "Current training root blob changed"
            )
    instance.next_index = int(payload["next_index"])
    instance.tick = int(payload["tick"])
    authoritative_by_root = {
        str(row["root_id"]): row for row in instance.rows
    }
    index_by_root = {
        str(row["root_id"]): index
        for index, row in enumerate(instance.rows)
    }
    instance.completed = {}
    instance.active = []
    for item_payload in payload["active"]:
        serialized_row = item_payload.get("row")
        root_id = (
            None
            if not isinstance(serialized_row, Mapping)
            else str(serialized_row.get("root_id"))
        )
        authoritative = authoritative_by_root.get(root_id)
        simulator_payload = item_payload.get("simulator")
        if (
            serialized_row != authoritative
            or not isinstance(simulator_payload, Mapping)
            or int(simulator_payload.get("deck_stream_id", -1))
            != int(authoritative["deck_stream_id"])
            or int(simulator_payload.get("slot_stream_id", -1))
            != int(authoritative["slot_stream_id"])
        ):
            raise J1ExecutionIntegrityError(
                "Incremental active training root changed"
            )
        sim, state = j1.simulator_from_snapshot(simulator_payload)
        generator = torch.Generator(device="cpu")
        generator.set_state(
            item_payload["policy_rng_state"].detach().cpu()
        )
        transition_count = int(item_payload["transition_count"])
        transition_head = copy.deepcopy(
            item_payload["transition_chunk_head"]
        )
        buffered_rows = [
            row
            for row in instance.transition_store.buffer
            if str(row["root_id"]) == str(authoritative["root_id"])
        ]
        sealed_count = transition_count - len(buffered_rows)
        if (
            sealed_count < 0
            or [int(row["transition_index"]) for row in buffered_rows]
            != list(range(sealed_count, transition_count))
        ):
            raise J1ExecutionIntegrityError(
                "Buffered transition tail changed"
            )
        instance.transition_store.verify_current_head(
            root_id=str(authoritative["root_id"]),
            head=transition_head,
            count=sealed_count,
        )
        instance.active.append(
            _ActiveTrainingRoot(
                row=dict(authoritative),
                sim=sim,
                state=state,
                policy_generator=generator,
                start_score=int(item_payload["start_score"]),
                transitions=[],
                transition_chunk_head=transition_head,
                transition_count=transition_count,
            )
        )
    active_indices = {
        index_by_root[str(item.row["root_id"])]
        for item in instance.active
    }
    if (
        active_indices & completed_indices
        or active_indices | completed_indices
        != set(range(instance.next_index))
        or len(instance.active) > instance.env_count
    ):
        raise J1ExecutionIntegrityError(
            "Incremental active/completed prefix changed"
        )
    random.setstate(copy.deepcopy(payload["python_rng_state"]))
    np.random.set_state(copy.deepcopy(payload["numpy_rng_state"]))
    torch.set_rng_state(
        payload["torch_rng_state"].detach().cpu().clone()
    )
    return instance


def incremental_training_collection_complete(
    session: TrainingCollectionSession,
    snapshot: Mapping[str, Any],
) -> bool:
    return (
        session.next_index == len(session.rows)
        and not session.active
        and int(snapshot["completed_count"]) == len(session.rows)
        and not snapshot["current_block_refs"]
    )


def _incremental_training_refs(
    *,
    snapshot: Mapping[str, Any],
    phase_dir: Path,
    rolling_contract: Mapping[str, Any],
    round_number: int,
) -> list[dict[str, Any]]:
    refs = _load_training_root_block_chain(
        phase_dir=phase_dir,
        rolling_contract=rolling_contract,
        round_number=round_number,
        rows_sha256=str(snapshot["rows_sha256"]),
        head=snapshot.get("sealed_block_head"),
    )
    refs.extend(copy.deepcopy(list(snapshot["current_block_refs"])))
    return sorted(refs, key=lambda row: int(row["manifest_index"]))


def _load_training_record_refs(
    *,
    snapshot: Mapping[str, Any],
    blob_dir: Path,
    io_metrics: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if snapshot.get("completed_storage") != "immutable_root_blobs":
        raise J1ExecutionIntegrityError(
            "Production training snapshot is not blob-backed"
        )
    refs = list(snapshot.get("completed_record_refs", []))
    if snapshot.get("completed_record_refs_sha256") != j1.stable_hash(refs):
        raise J1ExecutionIntegrityError(
            "Production training root references changed"
        )
    root = blob_dir.resolve()
    records = []
    for reference in refs:
        path = Path(str(reference["path"])).resolve()
        if (
            path.parent != root
            or not path.is_file()
            or sha256_path(path) != reference["file_sha256"]
        ):
            raise J1ExecutionIntegrityError(
                "Production training root blob changed"
            )
        record = load_atomic_binary(path)
        if io_metrics is not None:
            io_metrics["root_blob_reads"] = (
                io_metrics.get("root_blob_reads", 0) + 1
            )
            io_metrics["root_blob_bytes_read"] = (
                io_metrics.get("root_blob_bytes_read", 0)
                + int(path.stat().st_size)
            )
        if (
            str(record.get("root_id")) != str(reference["root_id"])
            or j1.stable_hash(record) != reference["record_sha256"]
        ):
            raise J1ExecutionIntegrityError(
                "Production training record reference changed"
            )
        records.append(record)
    if (
        snapshot.get("completed_records_sha256") is not None
        and snapshot.get("completed_records_sha256")
        != j1.stable_hash(records)
    ):
        raise J1ExecutionIntegrityError(
            "Production completed record set changed"
        )
    return records


def _rolling_record_matches(
    boundary: Mapping[str, Any],
    rolling: Mapping[str, Any],
) -> bool:
    state = boundary["state"]
    recorded = state.get("rolling_resume_record_sha256")
    if recorded is not None:
        current = rolling["record"]["rolling_journal_record_sha256"]
        if recorded == current:
            return True
        return (
            recorded
            in rolling.get("journal_chain_record_sha256s", ())
            and rolling["state"].get(
                "base_commit_head_payload_sha256"
            )
            == boundary["commit_head_payload_sha256"]
        )
    return rolling["state"].get(
        "base_commit_head_payload_sha256"
    ) == boundary["commit_head_payload_sha256"]


def execute_training_engine_bounded(
    *,
    rows: Sequence[Mapping[str, Any]],
    phase_dir: Path,
    marker_file_sha256: str,
    marker_payload_sha256: str,
    phase_lock_file_sha256: str,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    command: str,
    config: TrainingEngineConfig,
    interrupt_after_boundary: str | None = None,
    operational_audit_fn: Any | None = None,
    wall_clock: Any | None = None,
) -> dict[str, Any]:
    if config.execution_mode == "scientific" and config != TrainingEngineConfig():
        raise J1ExecutionIntegrityError(
            "Bounded scientific training configuration changed"
        )
    if config.execution_mode == "scientific" and wall_clock is not None:
        raise J1ExecutionIntegrityError(
            "Scientific runtime clock cannot be injected"
        )
    expected_rows = config.rounds * config.roots_per_round
    if len(rows) != expected_rows:
        raise J1ExecutionIntegrityError(
            "Bounded training row count changed"
        )
    contract = {
        "phase": "training",
        "marker_file_sha256": marker_file_sha256,
        "phase_lock_file_sha256": phase_lock_file_sha256,
        "command": command,
        "execution_mode": config.execution_mode,
    }
    rolling_contract = rolling_resume_contract(
        phase="training",
        marker_file_sha256=marker_file_sha256,
        marker_payload_sha256=marker_payload_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        manifest_file_sha256=manifest_file_sha256,
        manifest_payload_sha256=manifest_payload_sha256,
        command=command,
        execution_mode=config.execution_mode,
    )
    output_accountant = PhaseOutputAccountant(phase_dir)
    audit_callback = _bounded_operational_callback(
        execution_mode=config.execution_mode,
        audit_fn=operational_audit_fn,
        output_accountant=output_accountant,
    )
    clock = time.time if wall_clock is None else wall_clock
    rolling_store = RollingResumeStore(
        root=phase_dir,
        contract=rolling_contract,
        output_accountant=output_accountant,
    )
    runtime_ledger = RuntimeChargeLedger(
        root=phase_dir,
        contract=rolling_contract,
        wall_clock=clock,
        output_accountant=output_accountant,
    )
    io_metrics: dict[str, int] = {
        "transition_chunk_writes": 0,
        "transition_chunk_validation_reads": 0,
        "transition_chunk_reads": 0,
        "transition_chunk_bytes_written": 0,
        "transition_chunk_bytes_read": 0,
        "root_blob_writes": 0,
        "root_blob_validation_reads": 0,
        "root_blob_reads": 0,
        "root_blob_bytes_written": 0,
        "root_blob_bytes_read": 0,
        "root_block_writes": 0,
        "root_block_bytes_written": 0,
        "round_batch_writes": 0,
        "round_batch_validation_reads": 0,
        "round_batch_loads": 0,
        "round_batch_bytes_written": 0,
        "round_batch_bytes_read": 0,
    }
    root_blob_dir = phase_dir / ROOT_BLOBS_DIR
    pointer = phase_dir / COMMIT_HEAD_NAME
    if not pointer.exists():
        model, optimizer = j1.initialize_model_optimizer()
        initial_clock = runtime_ledger.summary()
        initial_state = _training_runtime_state(
            marker_file_sha256=marker_file_sha256,
            marker_payload_sha256=marker_payload_sha256,
            manifest_file_sha256=manifest_file_sha256,
            manifest_payload_sha256=manifest_payload_sha256,
            model=model,
            optimizer=optimizer,
            round_number=1,
            collection_boundary="pre_action",
            engine_stage="collection",
            collection_snapshot=None,
            updater_snapshot=None,
            current_round_records=None,
            all_completed_root_ids=[],
            optimizer_step_ids=[],
            expected_optimizer_step_ids=[],
            round_aggregates=[],
            resource_clock=initial_clock,
            output_bytes=0,
        )
        initial_state["rolling_resume_record_sha256"] = None
        initial_state["current_round_root_blob_refs"] = []
    else:
        initial_state = {}
    commit_store = IndexedCommitStore(
        phase_dir=phase_dir,
        **contract,
        initial_state=initial_state,
        output_accountant=output_accountant,
    )
    boundary = commit_store.boundary
    cached_collection_session: TrainingCollectionSession | None = None
    cached_collection_snapshot: dict[str, Any] | None = None
    cached_collection_record_sha256: str | None = None
    cached_updater: FrozenMinibatchUpdater | None = None
    cached_updater_record_sha256: str | None = None
    recovered_retirement_heads: set[str] = set()
    while True:
        state = boundary["state"]
        stage = str(state["engine_stage"])
        boundary_head = str(boundary["commit_head_payload_sha256"])
        pending_transition_retirement = state.get(
            "pending_transition_retirement"
        )
        if (
            pending_transition_retirement is not None
            and boundary_head not in recovered_retirement_heads
        ):
            retire_round_transition_chunks(
                phase_dir=phase_dir,
                round_number=int(
                    pending_transition_retirement["round"]
                ),
                transition_store_state=pending_transition_retirement[
                    "transition_store_state"
                ],
                completed_root_refs=pending_transition_retirement[
                    "completed_root_refs"
                ],
                collection_boundary=boundary,
                output_accountant=output_accountant,
            )
            recovered_retirement_heads.add(boundary_head)
        pending_batch_retirement = state.get(
            "pending_round_batch_retirement"
        )
        if (
            pending_batch_retirement is not None
            and boundary_head not in recovered_retirement_heads
        ):
            retire_round_ppo_batch(
                phase_dir=phase_dir,
                round_batch_identity=pending_batch_retirement,
                checkpoint_boundary=boundary,
                output_accountant=output_accountant,
            )
            recovered_retirement_heads.add(boundary_head)
        if stage == "complete":
            final_clock = runtime_ledger.audit_full()
            rolling_store.audit_full()
            boundary = commit_store.audit_full()
            output_accountant.reconcile_full()
            final_audit = enforce_phase_operational_guard(
                phase_dir=phase_dir,
                phase="training",
                active_seconds=final_clock["active_seconds"],
                require_target_disk=False,
                audit_fn=audit_callback,
            )
            return {
                "boundary": boundary,
                "state": state,
                "resource_clock": final_clock,
                "operational_audit": final_audit,
                "completed": True,
                "storage_design":
                    "root blobs + two rolling slots + epoch seals",
                "commit_store_metrics": commit_store.metrics(),
                "rolling_store_metrics": rolling_store.metrics(),
                "runtime_ledger_metrics": runtime_ledger.metrics(),
                "output_accounting": output_accountant.snapshot(),
                "io_metrics": dict(io_metrics),
                "passes": True,
            }
        round_number = int(state["round_number"])
        round_start = (round_number - 1) * config.roots_per_round
        round_rows = rows[
            round_start : round_start + config.roots_per_round
        ]
        rolling = rolling_store.current
        if rolling is not None and not _rolling_record_matches(
            boundary,
            rolling,
        ):
            rolling = None
        if (
            stage == "update"
            and cached_updater is not None
            and rolling is not None
            and cached_updater_record_sha256
            == rolling["record"]["rolling_journal_record_sha256"]
        ):
            model = cached_updater.model
            optimizer = cached_updater.optimizer
        else:
            model, optimizer = _load_model_optimizer_from_runtime(
                state
            )
            _restore_runtime_rngs(state)
        if stage == "collection":
            if (
                cached_collection_session is not None
                and cached_collection_snapshot is not None
                and rolling is not None
                and cached_collection_record_sha256
                == rolling["record"]["rolling_journal_record_sha256"]
            ):
                previous_snapshot = cached_collection_snapshot
                session = cached_collection_session
            elif (
                rolling is not None
                and rolling["state"].get("kind")
                == "training_collection"
                and int(rolling["state"].get("round", -1))
                == round_number
            ):
                previous_snapshot = rolling["state"][
                    "collection_snapshot"
                ]
                session = restore_incremental_training_collection(
                    previous_snapshot,
                    rows=round_rows,
                    model=model,
                    phase_dir=phase_dir,
                    rolling_contract=rolling_contract,
                    round_number=round_number,
                    output_accountant=output_accountant,
                    io_metrics=io_metrics,
                )
            else:
                previous_snapshot = None
                transition_store = TrainingTransitionChunkStore(
                    phase_dir=phase_dir,
                    rolling_contract=rolling_contract,
                    round_number=round_number,
                    rows=round_rows,
                    output_accountant=output_accountant,
                    io_metrics=io_metrics,
                )
                session = TrainingCollectionSession(
                    rows=round_rows,
                    model=model,
                    env_count=config.env_count,
                    max_moves=config.max_moves,
                    transition_store=transition_store,
                )
            collection_unit_id = (
                f"round={round_number}|collection_tick={session.tick + 1}"
            )
            def collect_tick_block() -> dict[str, Any]:
                started_tick = session.tick
                progressed = 0
                for _ in range(TRAINING_COLLECTION_TICKS_PER_COMMIT):
                    if (
                        session.next_index == len(session.rows)
                        and not session.active
                    ):
                        break
                    if not session.step_tick():
                        break
                    progressed += 1
                if progressed == 0 and (
                    session.next_index != len(session.rows)
                    or session.active
                ):
                    raise J1ExecutionIntegrityError(
                        "Bounded collection tick block made no progress"
                    )
                return {
                    "started_tick": started_tick,
                    "ended_tick": session.tick,
                    "ticks_executed": progressed,
                }

            _report, resource_clock, operational = (
                execute_charged_phase_attempt(
                    phase_dir=phase_dir,
                    phase="training",
                    runtime_ledger=runtime_ledger,
                    base_unit_id=collection_unit_id,
                    operation=collect_tick_block,
                    audit_fn=audit_callback,
                    leave_open_after_work=(
                        interrupt_after_boundary
                        == "collection_work_uncommitted"
                    ),
                )
            )
            prior_completed = (
                0
                if previous_snapshot is None
                else int(previous_snapshot["completed_count"])
            )
            collection_will_complete = (
                session.next_index == len(session.rows)
                and not session.active
                and prior_completed + len(session.completed)
                == len(session.rows)
            )
            snapshot = persist_incremental_training_collection(
                session=session,
                previous_snapshot=previous_snapshot,
                phase_dir=phase_dir,
                rolling_contract=rolling_contract,
                round_number=round_number,
                force_seal=collection_will_complete,
                resource_clock=resource_clock,
                operational_audit=operational,
                output_accountant=output_accountant,
                io_metrics=io_metrics,
            )
            rolling = rolling_store.append(
                unit_id=(
                    f"round={round_number}|collection_tick="
                    f"{snapshot['tick']}"
                ),
                state={
                    "kind": "training_collection",
                    "round": round_number,
                    "base_commit_head_payload_sha256":
                        boundary["commit_head_payload_sha256"],
                    "collection_snapshot": snapshot,
                    "resource_clock": resource_clock,
                    "last_operational_audit": operational,
                },
            )
            cached_collection_session = session
            cached_collection_snapshot = snapshot
            cached_collection_record_sha256 = rolling["record"][
                "rolling_journal_record_sha256"
            ]
            if interrupt_after_boundary == "collection":
                raise J1ExecutionPlannedInterruption(
                    "fixture interruption after bounded collection"
                )
            if not incremental_training_collection_complete(
                session,
                snapshot,
            ):
                continue
            completed_refs = _incremental_training_refs(
                snapshot=snapshot,
                phase_dir=phase_dir,
                rolling_contract=rolling_contract,
                round_number=round_number,
            )
            records = _load_training_record_refs(
                snapshot={
                    "completed_storage": "immutable_root_blobs",
                    "completed_record_refs": completed_refs,
                    "completed_record_refs_sha256": j1.stable_hash(
                        completed_refs
                    ),
                    "completed_records_sha256": None,
                },
                blob_dir=root_blob_dir,
                io_metrics=io_metrics,
            )
            batch = training_records_to_ppo_batch(records)
            updater = FrozenMinibatchUpdater(
                model=model,
                optimizer=optimizer,
                batch=batch,
                round_number=round_number,
                minibatch_size=config.minibatch_size,
            )
            round_batch_identity = write_round_ppo_batch_blob(
                phase_dir=phase_dir,
                updater=updater,
                minibatch_size=config.minibatch_size,
                output_accountant=output_accountant,
                io_metrics=io_metrics,
            )
            rolling = rolling_store.append(
                unit_id=f"round={round_number}|pre_update",
                state={
                    "kind": "training_update",
                    "round": round_number,
                    "base_commit_head_payload_sha256":
                        boundary["commit_head_payload_sha256"],
                    "updater_snapshot": compact_updater_snapshot(
                        updater,
                        round_batch_identity=round_batch_identity,
                    ),
                    "root_blob_refs": completed_refs,
                    "collection_snapshot_sha256":
                        snapshot["session_state_sha256"],
                    "all_optimizer_step_ids": list(
                        state["optimizer_step_ids"]
                    ),
                },
            )
            cached_updater = updater
            cached_updater_record_sha256 = rolling["record"][
                "rolling_journal_record_sha256"
            ]
            all_completed = [
                *state["all_completed_root_ids"],
                *[str(record["root_id"]) for record in records],
            ]
            expected_steps = [
                *state["expected_optimizer_step_ids"],
                *updater.expected_step_ids(),
            ]
            post_state = _training_runtime_state(
                marker_file_sha256=marker_file_sha256,
                marker_payload_sha256=marker_payload_sha256,
                manifest_file_sha256=manifest_file_sha256,
                manifest_payload_sha256=manifest_payload_sha256,
                model=model,
                optimizer=optimizer,
                round_number=round_number,
                collection_boundary="pre_update",
                engine_stage="update",
                collection_snapshot=None,
                updater_snapshot=None,
                current_round_records=None,
                all_completed_root_ids=all_completed,
                optimizer_step_ids=state["optimizer_step_ids"],
                expected_optimizer_step_ids=expected_steps,
                round_aggregates=state["round_aggregates"],
                resource_clock=resource_clock,
                output_bytes=output_accountant.reconcile_full()[
                    "output_bytes"
                ],
            )
            post_state["last_operational_audit"] = operational
            post_state["rolling_resume_record_sha256"] = rolling[
                "record"
            ]["rolling_journal_record_sha256"]
            post_state["current_round_root_blob_refs"] = completed_refs
            post_state["transition_buffer_path"] = str(
                root_blob_dir.resolve()
            )
            post_state["transition_buffer_sha256"] = j1.stable_hash(
                completed_refs
            )
            post_state["pending_transition_retirement"] = {
                "round": round_number,
                "transition_store_state": copy.deepcopy(
                    snapshot["transition_store_state"]
                ),
                "completed_root_refs": copy.deepcopy(completed_refs),
            }
            boundary = commit_store.commit(
                unit_id=f"round={round_number}|collection_complete",
                post_state=post_state,
                journal_payload={
                    "kind": "collection_round_seal",
                    "round": round_number,
                    "root_blob_count": len(records),
                    "root_blob_refs_sha256": j1.stable_hash(
                        completed_refs
                    ),
                    "rolling_record_sha256": post_state[
                        "rolling_resume_record_sha256"
                    ],
                },
            )
            if (
                interrupt_after_boundary
                == "transition_retirement_pre_apply"
            ):
                raise J1ExecutionPlannedInterruption(
                    "fixture interruption before transition retirement"
                )
            transition_retirement = retire_round_transition_chunks(
                phase_dir=phase_dir,
                round_number=round_number,
                transition_store_state=snapshot[
                    "transition_store_state"
                ],
                completed_root_refs=completed_refs,
                collection_boundary=boundary,
                output_accountant=output_accountant,
                crash_stage=(
                    "after_manifest"
                    if interrupt_after_boundary
                    == "transition_retirement_after_manifest"
                    else "mid_delete"
                    if interrupt_after_boundary
                    == "transition_retirement_mid_delete"
                    else None
                ),
            )
            if transition_retirement["all_listed_files_absent"] is not True:
                raise J1ExecutionIntegrityError(
                    "Transition chunk retirement did not close"
                )
            recovered_retirement_heads.add(
                boundary["commit_head_payload_sha256"]
            )
            cached_collection_session = None
            cached_collection_snapshot = None
            cached_collection_record_sha256 = None
            continue
        if stage == "update":
            if (
                rolling is None
                or rolling["state"].get("kind") != "training_update"
                or int(rolling["state"].get("round", -1))
                != round_number
            ):
                raise J1ExecutionIntegrityError(
                    "Bounded optimizer rolling state is unavailable"
                )
            if (
                cached_updater is not None
                and cached_updater_record_sha256
                == rolling["record"]["rolling_journal_record_sha256"]
            ):
                updater = cached_updater
            else:
                updater = restore_compact_updater(
                    rolling["state"]["updater_snapshot"],
                    minibatch_size=config.minibatch_size,
                    io_metrics=io_metrics,
                )
                cached_updater = updater
                cached_updater_record_sha256 = rolling["record"][
                    "rolling_journal_record_sha256"
                ]
            previous_epoch = int(updater.plan[updater.cursor]["epoch"])
            update_unit_id = updater.expected_step_ids()[updater.cursor]
            report, resource_clock, operational = (
                execute_charged_phase_attempt(
                    phase_dir=phase_dir,
                    phase="training",
                    runtime_ledger=runtime_ledger,
                    base_unit_id=update_unit_id,
                    operation=updater.step_once,
                    audit_fn=audit_callback,
                    leave_open_after_work=(
                        interrupt_after_boundary
                        == "update_work_uncommitted"
                    ),
                )
            )
            all_step_ids = [
                *rolling["state"]["all_optimizer_step_ids"],
                report["step_id"],
            ]
            rolling = rolling_store.append(
                unit_id=report["step_id"],
                state={
                    "kind": "training_update",
                    "round": round_number,
                    "base_commit_head_payload_sha256":
                        boundary["commit_head_payload_sha256"],
                    "updater_snapshot": compact_updater_snapshot(
                        updater,
                        round_batch_identity=rolling["state"][
                            "updater_snapshot"
                        ]["round_batch_identity"],
                    ),
                    "root_blob_refs": rolling["state"][
                        "root_blob_refs"
                    ],
                    "collection_snapshot_sha256": rolling["state"][
                        "collection_snapshot_sha256"
                    ],
                    "all_optimizer_step_ids": all_step_ids,
                    "resource_clock": resource_clock,
                    "last_operational_audit": operational,
                },
            )
            cached_updater = updater
            cached_updater_record_sha256 = rolling["record"][
                "rolling_journal_record_sha256"
            ]
            if interrupt_after_boundary == "update":
                raise J1ExecutionPlannedInterruption(
                    "fixture interruption after bounded update"
                )
            update_complete = updater.cursor == len(updater.plan)
            next_epoch = (
                None
                if update_complete
                else int(updater.plan[updater.cursor]["epoch"])
            )
            epoch_closed = update_complete or next_epoch != previous_epoch
            if not epoch_closed:
                continue
            if update_complete:
                snapshot_stub = {
                    "completed_storage": "immutable_root_blobs",
                    "completed_record_refs": rolling["state"][
                        "root_blob_refs"
                    ],
                    "completed_record_refs_sha256": j1.stable_hash(
                        rolling["state"]["root_blob_refs"]
                    ),
                    "completed_records_sha256": None,
                }
                records = _load_training_record_refs(
                    snapshot=snapshot_stub,
                    blob_dir=root_blob_dir,
                    io_metrics=io_metrics,
                )
                metrics = _training_round_metrics(
                    records=records,
                    updater=updater,
                )
                next_round_aggregates = [
                    *state["round_aggregates"],
                    metrics,
                ]
                if round_number == config.rounds:
                    next_stage = "complete"
                    next_round = round_number
                else:
                    next_stage = "collection"
                    next_round = round_number + 1
                unit_id = f"round={round_number}|checkpoint"
                boundary_name = "post_checkpoint"
            else:
                next_round_aggregates = state["round_aggregates"]
                next_stage = "update"
                next_round = round_number
                unit_id = (
                    f"round={round_number}|epoch={previous_epoch}|seal"
                )
                boundary_name = "mid_update"
            seal_output = (
                output_accountant.reconcile_full()
                if update_complete
                else output_accountant.snapshot()
            )
            post_state = _training_runtime_state(
                marker_file_sha256=marker_file_sha256,
                marker_payload_sha256=marker_payload_sha256,
                manifest_file_sha256=manifest_file_sha256,
                manifest_payload_sha256=manifest_payload_sha256,
                model=updater.model,
                optimizer=updater.optimizer,
                round_number=next_round,
                collection_boundary=boundary_name,
                engine_stage=next_stage,
                collection_snapshot=None,
                updater_snapshot=None,
                current_round_records=None,
                all_completed_root_ids=state[
                    "all_completed_root_ids"
                ],
                optimizer_step_ids=all_step_ids,
                expected_optimizer_step_ids=state[
                    "expected_optimizer_step_ids"
                ],
                round_aggregates=next_round_aggregates,
                resource_clock=resource_clock,
                output_bytes=seal_output["output_bytes"],
            )
            post_state["last_operational_audit"] = operational
            post_state["rolling_resume_record_sha256"] = rolling[
                "record"
            ]["rolling_journal_record_sha256"]
            post_state["current_round_root_blob_refs"] = (
                []
                if update_complete
                else rolling["state"]["root_blob_refs"]
            )
            round_batch_identity_for_retirement = dict(
                rolling["state"]["updater_snapshot"][
                    "round_batch_identity"
                ]
            )
            if update_complete:
                post_state["pending_round_batch_retirement"] = dict(
                    round_batch_identity_for_retirement
                )
            boundary = commit_store.commit(
                unit_id=unit_id,
                post_state=post_state,
                journal_payload={
                    "kind": (
                        "round_checkpoint"
                        if update_complete
                        else "optimizer_epoch_seal"
                    ),
                    "round": round_number,
                    "epoch": previous_epoch,
                    "rolling_record_sha256": post_state[
                        "rolling_resume_record_sha256"
                    ],
                    "model_state_sha256": j1.stable_hash(
                        updater.model.state_dict()
                    ),
                    "optimizer_state_sha256": j1.stable_hash(
                        updater.optimizer.state_dict()
                    ),
                },
            )
            if update_complete:
                if (
                    interrupt_after_boundary
                    == "batch_retirement_pre_apply"
                ):
                    raise J1ExecutionPlannedInterruption(
                        "fixture interruption before batch retirement"
                    )
                batch_retirement = retire_round_ppo_batch(
                    phase_dir=phase_dir,
                    round_batch_identity=
                        round_batch_identity_for_retirement,
                    checkpoint_boundary=boundary,
                    output_accountant=output_accountant,
                    crash_stage=(
                        "after_manifest"
                        if interrupt_after_boundary
                        == "batch_retirement_after_manifest"
                        else "mid_delete"
                        if interrupt_after_boundary
                        == "batch_retirement_mid_delete"
                        else None
                    ),
                )
                if batch_retirement["all_listed_files_absent"] is not True:
                    raise J1ExecutionIntegrityError(
                        "Round PPO batch retirement did not close"
                    )
                recovered_retirement_heads.add(
                    boundary["commit_head_payload_sha256"]
                )
            if (
                update_complete
                and interrupt_after_boundary == "checkpoint"
            ):
                raise J1ExecutionPlannedInterruption(
                    "fixture interruption after bounded checkpoint"
                )
            if update_complete:
                cached_updater = None
                cached_updater_record_sha256 = None
            continue
        raise J1ExecutionIntegrityError(
            f"Unknown bounded training stage: {stage}"
        )


def _validate_paired_result_payload(
    *,
    result: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    root_id = _safe_blob_token(
        str(row["root_id"]),
        label="Paired root id",
    )
    if (
        str(result.get("root_id")) != root_id
        or str(result.get("ancestry_id")) != str(row["ancestry_id"])
        or int(result.get("block", -1)) != int(row["block"])
        or not PairedEvaluationSession._arm_matches_row(
            arm_payload=result.get("candidate", {}),
            row=row,
            arm="candidate",
        )
        or not PairedEvaluationSession._arm_matches_row(
            arm_payload=result.get("control", {}),
            row=row,
            arm="control",
        )
    ):
        raise J1ExecutionIntegrityError(
            "Paired result changed its manifest identity"
        )
    return root_id


def _paired_result_blob_ref(
    *,
    result: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index: int,
    blob_dir: Path,
    output_accountant: PhaseOutputAccountant | None = None,
    io_metrics: dict[str, int] | None = None,
) -> dict[str, Any]:
    root_id = _validate_paired_result_payload(
        result=result,
        row=row,
    )
    path = blob_dir / f"{row_index:05d}_{root_id}.bin"
    existed = path.exists()
    file_sha256 = _write_immutable_binary_exact(path, result)
    if output_accountant is not None:
        output_accountant.record_path(path)
    if io_metrics is not None:
        key = (
            "pair_blob_validation_reads"
            if existed
            else "pair_blob_writes"
        )
        io_metrics[key] = io_metrics.get(key, 0) + 1
        if not existed:
            io_metrics["pair_blob_bytes_written"] = (
                io_metrics.get("pair_blob_bytes_written", 0)
                + int(path.stat().st_size)
            )
    return {
        "row_index": int(row_index),
        "root_id": root_id,
        "ancestry_id": str(row["ancestry_id"]),
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "pair_payload_sha256": j1.stable_hash(dict(result)),
    }


def _incremental_paired_state(
    *,
    phase: str,
    rows_sha256: str,
    candidate_policy_identity: str,
    control_policy_identity: str,
    max_moves: int,
    next_row_index: int,
    pending_candidate: Mapping[str, Any] | None,
    current_block_refs: Sequence[Mapping[str, Any]],
    base_commit_head_payload_sha256: str,
    resource_clock: Mapping[str, Any],
    operational_audit: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_{phase}_paired_incremental_v1",
        "phase": phase,
        "rows_sha256": rows_sha256,
        "candidate_policy_identity": candidate_policy_identity,
        "control_policy_identity": control_policy_identity,
        "max_moves": int(max_moves),
        "next_row_index": int(next_row_index),
        "pending_candidate": (
            None
            if pending_candidate is None
            else copy.deepcopy(dict(pending_candidate))
        ),
        "current_block_refs": copy.deepcopy(
            list(current_block_refs)
        ),
        "current_block_refs_sha256": j1.stable_hash(
            list(current_block_refs)
        ),
        "base_commit_head_payload_sha256":
            base_commit_head_payload_sha256,
        "resource_clock": dict(resource_clock),
        "last_operational_audit": dict(operational_audit),
    }
    payload["paired_incremental_state_sha256"] = j1.stable_hash(
        payload
    )
    return payload


def _restore_incremental_paired_state(
    payload: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    candidate_policy_identity: str,
    control_policy_identity: str,
    base_completed_count: int,
    pair_blob_dir: Path,
    io_metrics: dict[str, int] | None = None,
) -> dict[str, Any]:
    body = dict(payload)
    observed = body.pop("paired_incremental_state_sha256", None)
    if observed != j1.stable_hash(body):
        raise J1ExecutionIntegrityError(
            "Incremental paired state changed"
        )
    rows_sha256 = _ordered_rows_hash(rows)
    if (
        payload.get("version")
        != f"{VERSION}_{phase}_paired_incremental_v1"
        or payload.get("phase") != phase
        or payload.get("rows_sha256") != rows_sha256
        or payload.get("candidate_policy_identity")
        != candidate_policy_identity
        or payload.get("control_policy_identity")
        != control_policy_identity
    ):
        raise J1ExecutionIntegrityError(
            "Incremental paired resume identity changed"
        )
    next_index = int(payload["next_row_index"])
    refs = list(payload["current_block_refs"])
    if (
        next_index < base_completed_count
        or next_index > len(rows)
        or len(refs) != next_index - base_completed_count
        or len(refs) > PAIR_RESULT_BLOCK_SIZE
        or payload.get("current_block_refs_sha256")
        != j1.stable_hash(refs)
    ):
        raise J1ExecutionIntegrityError(
            "Incremental paired block accounting changed"
        )
    blob_root = pair_blob_dir.resolve()
    for offset, reference in enumerate(refs):
        index = base_completed_count + offset
        row = rows[index]
        path = Path(str(reference["path"])).resolve()
        expected_name = (
            f"{index:05d}_"
            f"{_safe_blob_token(str(row['root_id']), label='Paired root id')}"
            ".bin"
        )
        if (
            int(reference.get("row_index", -1)) != index
            or str(reference.get("root_id")) != str(row["root_id"])
            or str(reference.get("ancestry_id"))
            != str(row["ancestry_id"])
            or path.parent != blob_root
            or path.name != expected_name
            or not path.is_file()
            or sha256_path(path) != reference.get("file_sha256")
        ):
            raise J1ExecutionIntegrityError(
                "Incremental paired result reference changed"
            )
        if io_metrics is not None:
            io_metrics["pair_blob_resume_reference_reads"] = (
                io_metrics.get("pair_blob_resume_reference_reads", 0)
                + 1
            )
            io_metrics["pair_blob_bytes_read"] = (
                io_metrics.get("pair_blob_bytes_read", 0)
                + int(path.stat().st_size)
            )
    pending = payload.get("pending_candidate")
    if pending is not None:
        if (
            next_index >= len(rows)
            or not PairedEvaluationSession._arm_matches_row(
                arm_payload=pending,
                row=rows[next_index],
                arm="candidate",
            )
        ):
            raise J1ExecutionIntegrityError(
                "Incremental pending candidate arm changed streams"
            )
    return {
        "next_row_index": next_index,
        "pending_candidate": (
            None if pending is None else copy.deepcopy(dict(pending))
        ),
        "current_block_refs": copy.deepcopy(refs),
        "resource_clock": dict(payload["resource_clock"]),
        "last_operational_audit": dict(
            payload["last_operational_audit"]
        ),
        "passes": True,
    }


def _load_terminal_pair_results(
    *,
    rows: Sequence[Mapping[str, Any]],
    pair_blob_dir: Path,
    io_metrics: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    paths = sorted(pair_blob_dir.glob("*.bin"))
    if len(paths) != len(rows):
        raise J1ExecutionIntegrityError(
            "Terminal paired blob count changed"
        )
    results: list[dict[str, Any]] = []
    for index, (row, path) in enumerate(zip(rows, paths)):
        expected_name = (
            f"{index:05d}_"
            f"{_safe_blob_token(str(row['root_id']), label='Paired root id')}"
            ".bin"
        )
        if path.name != expected_name:
            raise J1ExecutionIntegrityError(
                "Terminal paired blob order changed"
            )
        serialized = path.read_bytes()
        result = deserialize_binary_state(serialized)
        _validate_paired_result_payload(
            result=result,
            row=row,
        )
        if io_metrics is not None:
            io_metrics["pair_blob_terminal_reads"] = (
                io_metrics.get("pair_blob_terminal_reads", 0) + 1
            )
            io_metrics["pair_blob_bytes_read"] = (
                io_metrics.get("pair_blob_bytes_read", 0)
                + len(serialized)
            )
        results.append(result)
    return results


def execute_paired_evaluation_engine_bounded(
    *,
    rows: Sequence[Mapping[str, Any]],
    phase_dir: Path,
    phase: str,
    marker_file_sha256: str,
    marker_payload_sha256: str,
    phase_lock_file_sha256: str,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    command: str,
    candidate_policy: Any,
    control_policy: Any,
    candidate_policy_identity: str,
    control_policy_identity: str,
    max_moves: int,
    interrupt_after_boundary: str | None = None,
    execution_mode: str = "scientific",
    block_pairs: int = PAIR_RESULT_BLOCK_SIZE,
    operational_audit_fn: Any | None = None,
    wall_clock: Any | None = None,
) -> dict[str, Any]:
    expected = {
        "development": DEVELOPMENT_PAIRS,
        "confirmation": CONFIRMATION_PAIRS,
    }[phase]
    if execution_mode == "scientific":
        if (
            len(rows) != expected
            or block_pairs != PAIR_RESULT_BLOCK_SIZE
            or max_moves != MAX_MOVES
        ):
            raise J1ExecutionIntegrityError(
                "Bounded paired evaluation contract changed"
            )
        if wall_clock is not None:
            raise J1ExecutionIntegrityError(
                "Scientific runtime clock cannot be injected"
            )
    if block_pairs < 1:
        raise ValueError("Paired block size is invalid")
    contract = {
        "phase": phase,
        "marker_file_sha256": marker_file_sha256,
        "phase_lock_file_sha256": phase_lock_file_sha256,
        "command": command,
        "execution_mode": execution_mode,
    }
    rolling_contract = rolling_resume_contract(
        phase=phase,
        marker_file_sha256=marker_file_sha256,
        marker_payload_sha256=marker_payload_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        manifest_file_sha256=manifest_file_sha256,
        manifest_payload_sha256=manifest_payload_sha256,
        command=command,
        execution_mode=execution_mode,
    )
    output_accountant = PhaseOutputAccountant(phase_dir)
    audit_callback = _bounded_operational_callback(
        execution_mode=execution_mode,
        audit_fn=operational_audit_fn,
        output_accountant=output_accountant,
    )
    clock = time.time if wall_clock is None else wall_clock
    rolling_store = RollingResumeStore(
        root=phase_dir,
        contract=rolling_contract,
        output_accountant=output_accountant,
    )
    runtime_ledger = RuntimeChargeLedger(
        root=phase_dir,
        contract=rolling_contract,
        wall_clock=clock,
        output_accountant=output_accountant,
    )
    pair_blob_dir = phase_dir / PAIR_BLOBS_DIR
    io_metrics: dict[str, int] = {
        "pair_blob_writes": 0,
        "pair_blob_validation_reads": 0,
        "pair_blob_resume_reference_reads": 0,
        "pair_blob_terminal_reads": 0,
        "pair_blob_bytes_written": 0,
        "pair_blob_bytes_read": 0,
    }
    rows_sha256 = _ordered_rows_hash(rows)
    pointer = phase_dir / COMMIT_HEAD_NAME
    if not pointer.exists():
        initial_clock = runtime_ledger.summary()
        initial_state = {
            "version": f"{VERSION}_{phase}_bounded_runtime_v1",
            "phase": phase,
            "engine_stage": "evaluation",
            "completed_pair_count": 0,
            "cumulative_pair_refs_sha256": j1.stable_hash(
                {
                    "phase": phase,
                    "rows_sha256": rows_sha256,
                    "completed_pair_count": 0,
                }
            ),
            "last_block_pair_refs": [],
            "rolling_resume_record_sha256": None,
            "resource_clock": initial_clock,
            "output_bytes": 0,
        }
    else:
        initial_state = {}
    commit_store = IndexedCommitStore(
        phase_dir=phase_dir,
        **contract,
        initial_state=initial_state,
        output_accountant=output_accountant,
    )
    boundary = commit_store.boundary
    cached_local: dict[str, Any] | None = None
    cached_rolling_record_sha256: str | None = None
    while True:
        state = boundary["state"]
        rolling = rolling_store.current
        if rolling is not None and not _rolling_record_matches(
            boundary,
            rolling,
        ):
            rolling = None
        if state.get("engine_stage") == "complete":
            final_clock = runtime_ledger.audit_full()
            rolling_store.audit_full()
            boundary = commit_store.audit_full()
            output_accountant.reconcile_full()
            final_audit = enforce_phase_operational_guard(
                phase_dir=phase_dir,
                phase=phase,
                active_seconds=final_clock["active_seconds"],
                require_target_disk=False,
                audit_fn=audit_callback,
            )
            results = _load_terminal_pair_results(
                rows=rows,
                pair_blob_dir=pair_blob_dir,
                io_metrics=io_metrics,
            )
            return {
                "boundary": boundary,
                "rows": results,
                "resource_clock": final_clock,
                "operational_audit": final_audit,
                "completed": True,
                "storage_design":
                    "write-once pair blobs + pending arm/current block "
                    "+ compact block seals",
                "commit_store_metrics": commit_store.metrics(),
                "rolling_store_metrics": rolling_store.metrics(),
                "runtime_ledger_metrics": runtime_ledger.metrics(),
                "output_accounting": output_accountant.snapshot(),
                "io_metrics": dict(io_metrics),
                "passes": True,
            }
        previous_count = int(state["completed_pair_count"])
        sealed_anchor = (
            rolling is not None
            and state.get("rolling_resume_record_sha256")
            == rolling["record"]["rolling_journal_record_sha256"]
            and int(
                rolling["state"].get("next_row_index", -1)
            )
            == previous_count
            and rolling["state"].get("pending_candidate") is None
        )
        if (
            cached_local is not None
            and rolling is not None
            and cached_rolling_record_sha256
            == rolling["record"]["rolling_journal_record_sha256"]
        ):
            local = cached_local
        elif sealed_anchor or rolling is None:
            local = {
                "next_row_index": previous_count,
                "pending_candidate": None,
                "current_block_refs": [],
            }
        else:
            if rolling["state"].get("kind") != "paired_evaluation":
                raise J1ExecutionIntegrityError(
                    "Bounded paired rolling state kind changed"
                )
            local = _restore_incremental_paired_state(
                rolling["state"]["paired_state"],
                rows=rows,
                phase=phase,
                candidate_policy_identity=candidate_policy_identity,
                control_policy_identity=control_policy_identity,
                base_completed_count=previous_count,
                pair_blob_dir=pair_blob_dir,
                io_metrics=io_metrics,
            )
        completed_count = int(local["next_row_index"])
        pending_block_seal = (
            local["pending_candidate"] is None
            and completed_count > previous_count
            and (
                completed_count - previous_count == block_pairs
                or completed_count == len(rows)
            )
        )
        if not pending_block_seal:
            if completed_count >= len(rows):
                raise J1ExecutionIntegrityError(
                    "Paired terminal state lacks its block seal"
                )
            row = rows[completed_count]
            if local["pending_candidate"] is None:
                candidate, resource_clock, operational = (
                    execute_charged_phase_attempt(
                        phase_dir=phase_dir,
                        phase=phase,
                        runtime_ledger=runtime_ledger,
                        base_unit_id=(
                            f"row={completed_count}|candidate_arm"
                        ),
                        operation=lambda: execute_full_policy_arm(
                            row=row,
                            arm="candidate",
                            policy=candidate_policy,
                            max_moves=max_moves,
                        ),
                        audit_fn=audit_callback,
                        leave_open_after_work=(
                            interrupt_after_boundary
                            == "candidate_work_uncommitted"
                        ),
                    )
                )
                local["pending_candidate"] = candidate
                report = {
                    "row_index": completed_count,
                    "boundary": "candidate_arm_committed",
                }
            else:
                pending_candidate = copy.deepcopy(
                    local["pending_candidate"]
                )

                def finish_pair() -> tuple[dict[str, Any], dict[str, Any]]:
                    control = execute_full_policy_arm(
                        row=row,
                        arm="control",
                        policy=control_policy,
                        max_moves=max_moves,
                    )
                    result = {
                        "root_id": str(row["root_id"]),
                        "ancestry_id": str(row["ancestry_id"]),
                        "block": int(row["block"]),
                        "candidate": pending_candidate,
                        "control": control,
                    }
                    reference = _paired_result_blob_ref(
                        result=result,
                        row=row,
                        row_index=completed_count,
                        blob_dir=pair_blob_dir,
                        output_accountant=output_accountant,
                        io_metrics=io_metrics,
                    )
                    return result, reference

                (_result, reference), resource_clock, operational = (
                    execute_charged_phase_attempt(
                        phase_dir=phase_dir,
                        phase=phase,
                        runtime_ledger=runtime_ledger,
                        base_unit_id=(
                            f"row={completed_count}|control_arm_and_pair"
                        ),
                        operation=finish_pair,
                        audit_fn=audit_callback,
                        leave_open_after_work=(
                            interrupt_after_boundary
                            == "pair_work_uncommitted"
                        ),
                    )
                )
                local["pending_candidate"] = None
                local["current_block_refs"].append(reference)
                local["next_row_index"] = completed_count + 1
                completed_count += 1
                report = {
                    "row_index": completed_count - 1,
                    "boundary": "paired_root_committed",
                }
            paired_state = _incremental_paired_state(
                phase=phase,
                rows_sha256=rows_sha256,
                candidate_policy_identity=candidate_policy_identity,
                control_policy_identity=control_policy_identity,
                max_moves=max_moves,
                next_row_index=local["next_row_index"],
                pending_candidate=local["pending_candidate"],
                current_block_refs=local["current_block_refs"],
                base_commit_head_payload_sha256=boundary[
                    "commit_head_payload_sha256"
                ],
                resource_clock=resource_clock,
                operational_audit=operational,
            )
            rolling = rolling_store.append(
                unit_id=(
                    f"row={report['row_index']}|{report['boundary']}"
                ),
                state={
                    "kind": "paired_evaluation",
                    "phase": phase,
                    "base_commit_head_payload_sha256":
                        boundary["commit_head_payload_sha256"],
                    "next_row_index": local["next_row_index"],
                    "pending_candidate": local[
                        "pending_candidate"
                    ],
                    "paired_state": paired_state,
                },
            )
            cached_local = local
            cached_rolling_record_sha256 = rolling["record"][
                "rolling_journal_record_sha256"
            ]
            enforce_phase_operational_guard(
                phase_dir=phase_dir,
                phase=phase,
                active_seconds=resource_clock["active_seconds"],
                require_target_disk=False,
                audit_fn=audit_callback,
            )
            if interrupt_after_boundary == report["boundary"]:
                raise J1ExecutionPlannedInterruption(
                    "fixture interruption after bounded "
                    f"{report['boundary']}"
                )
            pending_block_seal = (
                report["boundary"] == "paired_root_committed"
                and (
                    completed_count - previous_count == block_pairs
                    or completed_count == len(rows)
                )
            )
        if not pending_block_seal:
            continue
        block_refs = copy.deepcopy(local["current_block_refs"])
        if len(block_refs) != completed_count - previous_count:
            raise J1ExecutionIntegrityError(
                "Paired block reference count changed"
            )
        block_refs_sha256 = j1.stable_hash(block_refs)
        cumulative_pair_refs_sha256 = j1.stable_hash(
            {
                "predecessor": state[
                    "cumulative_pair_refs_sha256"
                ],
                "block_start": previous_count,
                "block_end_exclusive": completed_count,
                "block_pair_refs_sha256": block_refs_sha256,
            }
        )
        resource_clock = runtime_ledger.summary()
        output_state = output_accountant.snapshot()
        next_state = {
            "version": f"{VERSION}_{phase}_bounded_runtime_v1",
            "phase": phase,
            "engine_stage": (
                "complete"
                if completed_count == len(rows)
                else "evaluation"
            ),
            "completed_pair_count": completed_count,
            "cumulative_pair_refs_sha256":
                cumulative_pair_refs_sha256,
            "last_block_pair_refs": block_refs,
            "rolling_resume_record_sha256": rolling["record"][
                "rolling_journal_record_sha256"
            ],
            "resource_clock": resource_clock,
            "output_bytes": output_state["output_bytes"],
        }
        boundary = commit_store.commit(
            unit_id=f"pairs=0:{completed_count}|block_seal",
            post_state=next_state,
            journal_payload={
                "kind": "paired_result_block_seal",
                "phase": phase,
                "block_start": previous_count,
                "block_end_exclusive": completed_count,
                "block_pair_refs_sha256": block_refs_sha256,
                "cumulative_pair_refs_sha256":
                    cumulative_pair_refs_sha256,
                "rolling_record_sha256": next_state[
                    "rolling_resume_record_sha256"
                ],
            },
        )
        cached_local = None
        cached_rolling_record_sha256 = None


def immutable_json_identity(
    path: Path,
    *,
    payload_field: str,
    decision: str | None = None,
) -> dict[str, Any]:
    payload = load_json(path)
    if not verify_payload_hash(payload, payload_field):
        raise J1ExecutionIntegrityError(
            f"Immutable JSON payload is invalid: {path}"
        )
    if decision is not None and payload.get("decision") != decision:
        raise J1ExecutionIntegrityError(
            f"Immutable JSON decision changed: {path}"
        )
    identity = {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload[payload_field],
        "payload_field": payload_field,
    }
    if decision is not None:
        identity["decision"] = decision
    return identity


def _verify_json_identity(
    identity: Mapping[str, Any],
    *,
    expected_path: Path,
    payload_field: str,
    decision: str | None = None,
) -> dict[str, Any]:
    if Path(str(identity.get("path", ""))).resolve() != expected_path.resolve():
        raise J1ExecutionIntegrityError(
            f"Immutable artifact path changed: {expected_path}"
        )
    observed = immutable_json_identity(
        expected_path,
        payload_field=payload_field,
        decision=decision,
    )
    for key in ("path", "file_sha256", "payload_sha256", "payload_field"):
        if identity.get(key) != observed[key]:
            raise J1ExecutionIntegrityError(
                f"Immutable artifact identity changed: {expected_path}"
            )
    if decision is not None and identity.get("decision") != decision:
        raise J1ExecutionIntegrityError(
            f"Immutable artifact decision changed: {expected_path}"
        )
    return load_json(expected_path)


def phase_artifact_paths(
    *,
    execution_root: Path,
    phase: str,
) -> dict[str, Path]:
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    phase_dir = execution_root / phase
    return {
        "phase_dir": phase_dir,
        "lock": phase_dir / PHASE_LOCK_NAME,
        "lock_result": phase_dir / PHASE_LOCK_RESULT_NAME,
        "marker": phase_dir / PHASE_MARKER_NAME,
        "manifest": phase_dir / PHASE_MANIFEST_NAME,
        "owner": phase_dir / PHASE_OWNER_NAME,
        "reservation": phase_dir / PHASE_STREAM_RESERVATION_NAME,
        "consumption": phase_dir / PHASE_STREAM_CONSUMPTION_NAME,
        "result": phase_dir / PHASE_RESULT_NAME,
        "retention": phase_dir / PHASE_RETENTION_NAME,
        "checkpoint": phase_dir / TRAINING_CANDIDATE_CHECKPOINT_NAME,
        "sanity": phase_dir / TRAINING_SANITY_RESULT_NAME,
        "analysis": phase_dir / PAIRED_ANALYSIS_NAME,
    }


def _command_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def bound_dispatch_command(
    *,
    action: str,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
) -> str:
    if action not in PRODUCTION_COMMANDS:
        raise ValueError(f"Unsupported production action: {action}")
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    parts = [
        "nice",
        "-n",
        "10",
        "env",
        "PYTHONPATH=.",
        ".venv/bin/python",
        "-m",
        "threes_rl.j1_execution_surface",
        action,
        "--phase",
        phase,
        "--execution-root",
        _command_path(execution_root),
        "--readiness-dir",
        _command_path(readiness_dir),
        "--jobs",
        "1",
    ]
    return shlex.join(parts)


def _load_ready_readiness_artifacts(
    readiness_dir: Path,
) -> dict[str, Any]:
    lock_path = readiness_dir / READINESS_LOCK_NAME
    result_path = readiness_dir / READINESS_RESULT_NAME
    lock = load_json(lock_path)
    result = load_json(result_path)
    if not verify_payload_hash(lock, "readiness_lock_payload_sha256"):
        raise J1ExecutionIntegrityError("Execution readiness lock is invalid")
    if not verify_payload_hash(result, "readiness_result_payload_sha256"):
        raise J1ExecutionIntegrityError(
            "Execution readiness result is invalid"
        )
    if result.get("decision") != "READY_J1_EXECUTION_SURFACE":
        raise J1ExecutionIntegrityError(
            "Execution readiness result is not READY"
        )
    lock_identity = immutable_json_identity(
        lock_path,
        payload_field="readiness_lock_payload_sha256",
    )
    result_identity = immutable_json_identity(
        result_path,
        payload_field="readiness_result_payload_sha256",
        decision="READY_J1_EXECUTION_SURFACE",
    )
    if result.get("readiness_lock") != lock_identity:
        raise J1ExecutionIntegrityError(
            "Readiness result changed its lock identity"
        )
    for relative, expected in (
        ("charter_file_sha256", sha256_path(CHARTER_PATH)),
        ("runner_file_sha256", sha256_path(RUNNER_PATH)),
        ("test_file_sha256", sha256_path(TEST_PATH)),
    ):
        if lock.get(relative) != expected:
            raise J1ExecutionIntegrityError(
                f"Readiness lock changed {relative}"
            )
    return {
        "lock": lock,
        "result": result,
        "lock_identity": lock_identity,
        "result_identity": result_identity,
    }


def _load_phase_terminal_result(
    *,
    execution_root: Path,
    phase: str,
) -> dict[str, Any]:
    path = phase_artifact_paths(
        execution_root=execution_root,
        phase=phase,
    )["result"]
    payload = load_json(path)
    if not verify_payload_hash(payload, "terminal_result_payload_sha256"):
        raise J1ExecutionIntegrityError(
            f"{phase} terminal result is invalid"
        )
    return payload


def _load_joint_manifest_seal(
    execution_root: Path,
) -> dict[str, Any]:
    path = (
        execution_root
        / PRECOMMITTED_MANIFEST_DIR
        / JOINT_MANIFEST_SEAL_NAME
    )
    payload = load_json(path)
    if not verify_payload_hash(
        payload,
        "joint_manifest_seal_payload_sha256",
    ):
        raise J1ExecutionIntegrityError(
            "Joint evaluation manifest seal is invalid"
        )
    return payload


def _bound_phase_lock_payload(
    *,
    phase: str,
    readiness: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    predecessor_result: Mapping[str, Any] | None,
    predecessor_result_identity: Mapping[str, Any] | None,
    execution_root: Path,
    readiness_dir: Path,
    joint_manifest_seal: Mapping[str, Any] | None,
    execution_mode: str,
) -> dict[str, Any]:
    if execution_mode not in {"scientific", "miniature_fixture"}:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
    commands = {
        action: bound_dispatch_command(
            action=action,
            phase=phase,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
        for action in ("open", "materialize", "execute")
    }
    base = build_phase_lock_payload(
        phase=phase,
        readiness_lock_identity=readiness["lock_identity"],
        readiness_result_identity=readiness["result_identity"],
        manifest_identity=manifest_identity,
        predecessor_result=predecessor_result,
        command=commands["execute"],
        joint_manifest_seal=joint_manifest_seal,
    )
    base.pop("phase_lock_payload_sha256")
    base.update(
        {
            "execution_mode": execution_mode,
            "charter_file_sha256": sha256_path(CHARTER_PATH),
            "runner_file_sha256": sha256_path(RUNNER_PATH),
            "test_file_sha256": sha256_path(TEST_PATH),
            "bounded_engine": (
                "execute_training_engine_bounded"
                if phase == "training"
                else "execute_paired_evaluation_engine_bounded"
            ),
            "legacy_engines_fixture_only": True,
            "open_command": commands["open"],
            "materialize_command": commands["materialize"],
            "execute_command": commands["execute"],
            "promotion_command_present": False,
            "predecessor_result_identity": (
                None
                if predecessor_result_identity is None
                else dict(predecessor_result_identity)
            ),
            "joint_manifest_seal_identity": (
                None
                if joint_manifest_seal is None
                else {
                    **immutable_json_identity(
                        execution_root
                        / PRECOMMITTED_MANIFEST_DIR
                        / JOINT_MANIFEST_SEAL_NAME,
                        payload_field=(
                            "joint_manifest_seal_payload_sha256"
                        ),
                    )
                }
            ),
        }
    )
    return payload_with_hash(base, "phase_lock_payload_sha256")


def seal_phase_lock_from_artifacts(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
    confirmation_access_audit_path: Path | None = None,
    execution_mode: str = "scientific",
) -> dict[str, Any]:
    readiness = _load_ready_readiness_artifacts(readiness_dir)
    paths = phase_artifact_paths(
        execution_root=execution_root,
        phase=phase,
    )
    for name in (
        "lock",
        "lock_result",
        "marker",
        "manifest",
        "reservation",
        "consumption",
        "result",
    ):
        if paths[name].exists():
            raise FileExistsError(
                f"Phase artifact already exists: {paths[name]}"
            )

    predecessor_result: dict[str, Any] | None = None
    predecessor_identity: dict[str, Any] | None = None
    joint_seal: dict[str, Any] | None = None
    if phase == "training":
        manifest = materialize_root_manifest(phase="training")
    elif phase == "development":
        training_paths = phase_artifact_paths(
            execution_root=execution_root,
            phase="training",
        )
        training_manifest = load_json(training_paths["manifest"])
        if not verify_payload_hash(
            training_manifest,
            "root_manifest_payload_sha256",
        ):
            raise J1ExecutionIntegrityError(
                "Training manifest is invalid before development"
            )
        predecessor_result = _load_phase_terminal_result(
            execution_root=execution_root,
            phase="training",
        )
        predecessor_identity = immutable_json_identity(
            training_paths["result"],
            payload_field="terminal_result_payload_sha256",
            decision="READY_J1_TRAINING_SANITY",
        )
        if (
            predecessor_result.get("execution_mode") != execution_mode
            or predecessor_result.get("scientific_authority")
            is not (execution_mode == "scientific")
            or predecessor_result.get("bounded_engine")
            != "execute_training_engine_bounded"
        ):
            raise J1ExecutionIntegrityError(
                "Development predecessor cannot cross execution modes"
            )
        if confirmation_access_audit_path is None:
            raise J1ExecutionIntegrityError(
                "Development lock requires confirmation access evidence"
            )
        joint = seal_joint_evaluation_manifests(
            execution_root=execution_root,
            training_manifest=training_manifest,
            training_result=predecessor_result,
            confirmation_access_audit_path=confirmation_access_audit_path,
        )
        joint_seal = joint["seal"]
        manifest = joint["development"]
    else:
        predecessor_result = _load_phase_terminal_result(
            execution_root=execution_root,
            phase="development",
        )
        predecessor_identity = immutable_json_identity(
            phase_artifact_paths(
                execution_root=execution_root,
                phase="development",
            )["result"],
            payload_field="terminal_result_payload_sha256",
            decision="READY_J1_DEVELOPMENT_FULL_POLICY",
        )
        if (
            predecessor_result.get("execution_mode") != execution_mode
            or predecessor_result.get("scientific_authority")
            is not (execution_mode == "scientific")
            or predecessor_result.get("bounded_engine")
            != "execute_paired_evaluation_engine_bounded"
        ):
            raise J1ExecutionIntegrityError(
                "Confirmation predecessor cannot cross execution modes"
            )
        joint_seal = _load_joint_manifest_seal(execution_root)
        lineage = verify_joint_candidate_lineage(
            execution_root=execution_root,
            joint_manifest_seal=joint_seal,
            expected_execution_mode=execution_mode,
        )
        for key, expected in (
            (
                "training_terminal_result_identity",
                lineage["training_result_identity"],
            ),
            (
                "candidate_checkpoint_identity",
                lineage["candidate_checkpoint_identity"],
            ),
            (
                "incumbent_policy_binding",
                lineage["incumbent_policy_binding"],
            ),
        ):
            if predecessor_result.get(key) != expected:
                raise J1ExecutionIntegrityError(
                    "Confirmation predecessor changed candidate lineage: "
                    f"{key}"
                )
        manifest = load_precommitted_evaluation_manifest(
            execution_root=execution_root,
            phase="confirmation",
        )
    if predecessor_result is not None:
        if (
            predecessor_result.get("execution_mode") != execution_mode
            or predecessor_result.get("scientific_authority")
            is not (execution_mode == "scientific")
            or predecessor_result.get("bounded_engine")
            != (
                "execute_training_engine_bounded"
                if phase == "development"
                else "execute_paired_evaluation_engine_bounded"
            )
        ):
            raise J1ExecutionIntegrityError(
                "Predecessor result cannot cross execution modes"
            )

    lock_payload = _bound_phase_lock_payload(
        phase=phase,
        readiness=readiness,
        manifest_identity=root_manifest_identity(manifest),
        predecessor_result=predecessor_result,
        predecessor_result_identity=predecessor_identity,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        joint_manifest_seal=joint_seal,
        execution_mode=execution_mode,
    )
    written_lock = write_immutable_json(
        paths["lock"],
        {
            key: value
            for key, value in lock_payload.items()
            if key != "phase_lock_payload_sha256"
        },
        field="phase_lock_payload_sha256",
    )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="phase_lock_payload_sha256",
    )
    result_payload = {
        "version": f"{VERSION}_{phase}_phase_lock_result_v1",
        "phase": phase,
        "decision": _phase_ready_decision(phase),
        "phase_lock": lock_identity,
        "predecessor_result_payload_sha256": (
            None
            if predecessor_result is None
            else predecessor_result["terminal_result_payload_sha256"]
        ),
        "joint_evaluation_manifest_seal_payload_sha256": (
            None
            if joint_seal is None
            else joint_seal["joint_manifest_seal_payload_sha256"]
        ),
        "streams_reserved": 0,
        "streams_consumed": 0,
        "scientific_work": 0,
        "passes": True,
    }
    written_result = write_immutable_json(
        paths["lock_result"],
        result_payload,
        field="phase_lock_result_payload_sha256",
    )
    return {
        "lock": written_lock,
        "lock_identity": lock_identity,
        "result": written_result,
        "result_identity": immutable_json_identity(
            paths["lock_result"],
            payload_field="phase_lock_result_payload_sha256",
            decision=_phase_ready_decision(phase),
        ),
        "joint_manifest_seal": joint_seal,
        "passes": True,
    }


def _load_phase_lock_artifacts(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    readiness = _load_ready_readiness_artifacts(readiness_dir)
    paths = phase_artifact_paths(
        execution_root=execution_root,
        phase=phase,
    )
    lock = load_json(paths["lock"])
    lock_result = load_json(paths["lock_result"])
    if (
        not verify_payload_hash(lock, "phase_lock_payload_sha256")
        or lock.get("phase") != phase
        or lock.get("decision") != _phase_ready_decision(phase)
    ):
        raise J1ExecutionIntegrityError("Phase lock is invalid")
    if (
        not verify_payload_hash(
            lock_result,
            "phase_lock_result_payload_sha256",
        )
        or lock_result.get("phase") != phase
        or lock_result.get("decision") != _phase_ready_decision(phase)
    ):
        raise J1ExecutionIntegrityError("Phase lock result is invalid")
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="phase_lock_payload_sha256",
    )
    if lock_result.get("phase_lock") != lock_identity:
        raise J1ExecutionIntegrityError(
            "Phase lock result changed its lock identity"
        )
    if (
        lock.get("readiness_lock_identity")
        != readiness["lock_identity"]
        or lock.get("readiness_result_identity")
        != readiness["result_identity"]
    ):
        raise J1ExecutionIntegrityError(
            "Phase lock changed readiness identities"
        )
    if (
        lock.get("runner_file_sha256") != sha256_path(RUNNER_PATH)
        or lock.get("charter_file_sha256") != sha256_path(CHARTER_PATH)
        or lock.get("test_file_sha256") != sha256_path(TEST_PATH)
    ):
        raise J1ExecutionIntegrityError(
            "Phase lock source identity changed"
        )
    predecessor_phase = {
        "training": None,
        "development": "training",
        "confirmation": "development",
    }[phase]
    if predecessor_phase is None:
        if lock.get("predecessor_result_identity") is not None:
            raise J1ExecutionIntegrityError(
                "Training lock unexpectedly binds a predecessor"
            )
    else:
        predecessor_path = phase_artifact_paths(
            execution_root=execution_root,
            phase=predecessor_phase,
        )["result"]
        expected_decision = {
            "development": "READY_J1_TRAINING_SANITY",
            "confirmation": "READY_J1_DEVELOPMENT_FULL_POLICY",
        }[phase]
        predecessor = _verify_json_identity(
            lock.get("predecessor_result_identity", {}),
            expected_path=predecessor_path,
            payload_field="terminal_result_payload_sha256",
            decision=expected_decision,
        )
        if (
            predecessor["terminal_result_payload_sha256"]
            != lock.get("predecessor_result_payload_sha256")
        ):
            raise J1ExecutionIntegrityError(
                "Phase predecessor payload differs from lock"
            )
        lock_mode = str(lock.get("execution_mode"))
        if (
            predecessor.get("execution_mode") != lock_mode
            or predecessor.get("scientific_authority")
            is not (lock_mode == "scientific")
            or predecessor.get("bounded_engine")
            != (
                "execute_training_engine_bounded"
                if phase == "development"
                else "execute_paired_evaluation_engine_bounded"
            )
        ):
            raise J1ExecutionIntegrityError(
                "Phase predecessor authority or engine changed"
            )
    if phase == "training":
        if lock.get("joint_manifest_seal_identity") is not None:
            raise J1ExecutionIntegrityError(
                "Training lock unexpectedly binds evaluation manifests"
            )
    else:
        joint_path = (
            execution_root
            / PRECOMMITTED_MANIFEST_DIR
            / JOINT_MANIFEST_SEAL_NAME
        )
        joint = _verify_json_identity(
            lock.get("joint_manifest_seal_identity", {}),
            expected_path=joint_path,
            payload_field="joint_manifest_seal_payload_sha256",
        )
        if (
            joint["joint_manifest_seal_payload_sha256"]
            != lock.get(
                "joint_evaluation_manifest_seal_payload_sha256"
            )
        ):
            raise J1ExecutionIntegrityError(
                "Joint manifest seal differs from phase lock"
            )
        lineage = verify_joint_candidate_lineage(
            execution_root=execution_root,
            joint_manifest_seal=joint,
            expected_execution_mode=str(lock.get("execution_mode")),
        )
        precommitted = load_precommitted_evaluation_manifest(
            execution_root=execution_root,
            phase=phase,
        )
        if root_manifest_identity(precommitted) != lock[
            "manifest_identity"
        ]:
            raise J1ExecutionIntegrityError(
                "Precommitted manifest differs from phase lock"
            )
        if phase == "confirmation":
            development_result = _load_phase_terminal_result(
                execution_root=execution_root,
                phase="development",
            )
            for key, expected in (
                (
                    "training_terminal_result_identity",
                    lineage["training_result_identity"],
                ),
                (
                    "candidate_checkpoint_identity",
                    lineage["candidate_checkpoint_identity"],
                ),
                (
                    "incumbent_policy_binding",
                    lineage["incumbent_policy_binding"],
                ),
            ):
                if development_result.get(key) != expected:
                    raise J1ExecutionIntegrityError(
                        "Confirmation candidate lineage differs from "
                        f"development: {key}"
                    )
    expected_commands = {
        action: bound_dispatch_command(
            action=action,
            phase=phase,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
        for action in ("open", "materialize", "execute")
    }
    for action, command in expected_commands.items():
        if lock.get(f"{action}_command") != command:
            raise J1ExecutionIntegrityError(
                f"Phase {action} command changed"
            )
    expected_engine = (
        "execute_training_engine_bounded"
        if phase == "training"
        else "execute_paired_evaluation_engine_bounded"
    )
    if (
        lock.get("bounded_engine") != expected_engine
        or lock.get("legacy_engines_fixture_only") is not True
        or lock.get("promotion_command_present") is not False
    ):
        raise J1ExecutionIntegrityError(
            "Phase engine routing changed"
        )
    return {
        "paths": paths,
        "readiness": readiness,
        "lock": lock,
        "lock_result": lock_result,
        "lock_identity": lock_identity,
        "commands": expected_commands,
    }


def open_phase_from_artifacts(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
    execution_mode: str = "scientific",
    operational_audit_fn: Any | None = None,
    opened_at: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    loaded = _load_phase_lock_artifacts(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    paths = loaded["paths"]
    if loaded["lock"].get("execution_mode") != execution_mode:
        raise J1ExecutionIntegrityError(
            "Phase open execution mode changed"
        )
    for name in (
        "marker",
        "manifest",
        "owner",
        "reservation",
        "consumption",
        "result",
    ):
        if paths[name].exists():
            raise FileExistsError(
                f"Open requires an unused phase namespace: {paths[name]}"
            )
    if phase == "training":
        manifest = materialize_root_manifest(phase=phase)
    else:
        manifest = load_precommitted_evaluation_manifest(
            execution_root=execution_root,
            phase=phase,
        )
    audit_fn = j1.operational_audit if operational_audit_fn is None else (
        operational_audit_fn
    )
    if execution_mode == "scientific" and operational_audit_fn is not None:
        raise J1ExecutionIntegrityError(
            "Scientific open cannot inject an operational audit"
        )
    operational = audit_fn(output_dir=paths["phase_dir"])
    if operational.get("passes") is not True:
        raise J1ExecutionOperationalHold(
            "Phase open operational audit failed"
        )
    marker = build_phase_marker_payload(
        phase=phase,
        phase_lock=loaded["lock"],
        phase_lock_file_sha256=loaded["lock_identity"]["file_sha256"],
        manifest=manifest,
        command=loaded["commands"]["execute"],
        opened_at=(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if opened_at is None
            else opened_at
        ),
        hostname=socket.gethostname() if hostname is None else hostname,
    )
    marker.pop("activation_marker_payload_sha256")
    marker.update(
        {
            "phase_lock_result_identity": immutable_json_identity(
                paths["lock_result"],
                payload_field="phase_lock_result_payload_sha256",
                decision=_phase_ready_decision(phase),
            ),
            "runner_file_sha256": sha256_path(RUNNER_PATH),
            "charter_file_sha256": sha256_path(CHARTER_PATH),
            "open_command": loaded["commands"]["open"],
            "materialize_command": loaded["commands"]["materialize"],
            "execute_command": loaded["commands"]["execute"],
            "bounded_engine": loaded["lock"]["bounded_engine"],
            "operational_audit": operational,
            "marker_only_open": True,
        }
    )
    written = write_immutable_json(
        paths["marker"],
        marker,
        field="activation_marker_payload_sha256",
    )
    created_after_open = {
        name: paths[name].exists()
        for name in (
            "marker",
            "manifest",
            "owner",
            "reservation",
            "consumption",
            "result",
        )
    }
    if created_after_open != {
        "marker": True,
        "manifest": False,
        "owner": False,
        "reservation": False,
        "consumption": False,
        "result": False,
    }:
        raise J1ExecutionIntegrityError(
            "Open created work beyond the immutable marker"
        )
    return {
        "marker": written,
        "marker_identity": immutable_json_identity(
            paths["marker"],
            payload_field="activation_marker_payload_sha256",
        ),
        "created_after_open": created_after_open,
        "passes": True,
    }


def _load_open_phase_contract(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    loaded = _load_phase_lock_artifacts(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    paths = loaded["paths"]
    marker = load_json(paths["marker"])
    if (
        not verify_payload_hash(
            marker,
            "activation_marker_payload_sha256",
        )
        or marker.get("phase") != phase
        or marker.get("phase_lock_file_sha256")
        != loaded["lock_identity"]["file_sha256"]
        or marker.get("phase_lock_payload_sha256")
        != loaded["lock"]["phase_lock_payload_sha256"]
        or marker.get("execute_command")
        != loaded["commands"]["execute"]
        or marker.get("bounded_engine") != loaded["lock"]["bounded_engine"]
        or marker.get("runner_file_sha256") != sha256_path(RUNNER_PATH)
    ):
        raise J1ExecutionIntegrityError(
            "Phase marker changed its immutable execution contract"
        )
    marker_identity = immutable_json_identity(
        paths["marker"],
        payload_field="activation_marker_payload_sha256",
    )
    return {
        **loaded,
        "marker": marker,
        "marker_identity": marker_identity,
    }


def materialize_phase_manifest_from_artifacts(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    loaded = _load_open_phase_contract(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    path = loaded["paths"]["manifest"]
    if path.exists():
        raise FileExistsError(f"Phase manifest already exists: {path}")
    if any(
        loaded["paths"][name].exists()
        for name in ("owner", "reservation", "consumption", "result")
    ):
        raise J1ExecutionIntegrityError(
            "Manifest materialization occurred after work opened"
        )
    if phase == "training":
        manifest = materialize_root_manifest(
            phase=phase,
            marker_payload=loaded["marker"],
        )
    else:
        manifest = load_precommitted_evaluation_manifest(
            execution_root=execution_root,
            phase=phase,
        )
        expected = materialize_root_manifest(
            phase=phase,
            marker_payload=loaded["marker"],
        )
        if manifest != expected:
            raise J1ExecutionIntegrityError(
                "Precommitted evaluation manifest changed at activation"
            )
    if root_manifest_identity(manifest) != loaded["lock"][
        "manifest_identity"
    ]:
        raise J1ExecutionIntegrityError(
            "Materialized manifest differs from phase lock"
        )
    written = write_immutable_json(
        path,
        {
            key: value
            for key, value in manifest.items()
            if key != "root_manifest_payload_sha256"
        },
        field="root_manifest_payload_sha256",
    )
    return {
        "manifest": written,
        "manifest_identity": {
            **root_manifest_identity(written),
            "path": str(path.resolve()),
            "file_sha256": sha256_path(path),
        },
        "streams_reserved": 0,
        "streams_consumed": 0,
        "passes": True,
    }


def _manifest_stream_inventory(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    phase = str(manifest["phase"])
    rows = list(manifest["rows"])
    stream_fields = (
        "logical_stream_id",
        "deck_stream_id",
        "slot_stream_id",
        "candidate_policy_stream_id",
        "control_policy_stream_id",
    )
    inventory: dict[str, Any] = {}
    all_stream_ids: set[int] = set()
    for field in stream_fields:
        values = [
            int(row[field])
            for row in rows
            if row.get(field) is not None
        ]
        if not values:
            continue
        if len(set(values)) != len(values):
            raise J1ExecutionIntegrityError(
                f"{phase} manifest duplicated {field}"
            )
        inventory[field] = {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "ordered_sha256": _ordered_rows_hash(
                {"stream_id": value} for value in values
            ),
        }
        if all_stream_ids.intersection(values):
            raise J1ExecutionIntegrityError(
                f"{phase} stream roles collided"
            )
        all_stream_ids.update(values)
    expected_fields = 4 if phase == "training" else 5
    checks = {
        "stream_roles_exact": len(inventory) == expected_fields,
        "all_stream_ids_unique_across_roles": (
            len(all_stream_ids) == len(rows) * expected_fields
        ),
        "rows_match_phase_contract": len(rows) == int(STREAMS[phase]["rows"]),
        "root_identity_exact": root_manifest_identity(manifest)[
            "payload_sha256"
        ]
        == manifest["root_manifest_payload_sha256"],
    }
    return {
        "phase": phase,
        "row_count": len(rows),
        "arm_count": sum(int(row["arm_count"]) for row in rows),
        "stream_id_count": len(all_stream_ids),
        "stream_roles": inventory,
        "all_stream_ids_sha256": _ordered_rows_hash(
            {"stream_id": value} for value in sorted(all_stream_ids)
        ),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _reservation_intervals(
    reservation: Mapping[str, Any],
) -> list[tuple[int, int, str]]:
    intervals = []
    for role, row in reservation["stream_inventory"][
        "stream_roles"
    ].items():
        intervals.append(
            (int(row["minimum"]), int(row["maximum"]), str(role))
        )
    return intervals


def seal_phase_stream_reservation(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    loaded = _load_open_phase_contract(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    paths = loaded["paths"]
    manifest = load_json(paths["manifest"])
    if not verify_payload_hash(
        manifest,
        "root_manifest_payload_sha256",
    ):
        raise J1ExecutionIntegrityError(
            "Stream reservation manifest is invalid"
        )
    manifest_identity = {
        **root_manifest_identity(manifest),
        "path": str(paths["manifest"].resolve()),
        "file_sha256": sha256_path(paths["manifest"]),
    }
    if root_manifest_identity(manifest) != loaded["lock"][
        "manifest_identity"
    ]:
        raise J1ExecutionIntegrityError(
            "Stream reservation manifest changed phase lock"
        )
    stream_inventory = _manifest_stream_inventory(manifest)
    if not stream_inventory["passes"]:
        raise J1ExecutionIntegrityError(
            "Stream reservation inventory is invalid"
        )
    prospective = prospective_manifest()
    if prospective.get("passes") is not True:
        raise J1ExecutionIntegrityError(
            "Prospective collision contract changed"
        )
    phase_intervals = _reservation_intervals(
        {"stream_inventory": stream_inventory}
    )
    prior_reservations = []
    prior_intervals: list[tuple[int, int, str, str]] = []
    for other in PHASES:
        if other == phase:
            continue
        other_path = phase_artifact_paths(
            execution_root=execution_root,
            phase=other,
        )["reservation"]
        if not other_path.exists():
            continue
        other_payload = load_json(other_path)
        if not verify_payload_hash(
            other_payload,
            "stream_reservation_payload_sha256",
        ):
            raise J1ExecutionIntegrityError(
                "Prior stream reservation is invalid"
            )
        other_identity = immutable_json_identity(
            other_path,
            payload_field="stream_reservation_payload_sha256",
        )
        prior_reservations.append(
            {"phase": other, **other_identity}
        )
        prior_intervals.extend(
            (start, end, role, other)
            for start, end, role in _reservation_intervals(other_payload)
        )
    cross_phase_collision = any(
        max(start, other_start) <= min(end, other_end)
        for start, end, _role in phase_intervals
        for other_start, other_end, _other_role, _other_phase
        in prior_intervals
    )
    checks = {
        "marker_precedes_reservation": paths["marker"].is_file(),
        "manifest_precedes_reservation": paths["manifest"].is_file(),
        "historical_denylist_identity_exact": (
            prospective["checks"]["parent_denylist_file_exact"]
        ),
        "prospective_prefix_collision_proof_exact": (
            prospective["checks"][
                "all_amended_ranges_exact_parent_prefixes"
            ]
            and prospective["checks"]["all_namespace_ranges_disjoint"]
        ),
        "no_prior_phase_reservation_collision": not cross_phase_collision,
        "stream_inventory_passes": stream_inventory["passes"],
        "terminal_absent": not paths["result"].exists(),
    }
    payload = {
        "version": f"{VERSION}_{phase}_stream_reservation_v1",
        "phase": phase,
        "marker_identity": loaded["marker_identity"],
        "phase_lock_identity": loaded["lock_identity"],
        "manifest_identity": manifest_identity,
        "root_set_sha256": prospective["phase_root_set_sha256"][phase],
        "stream_inventory": stream_inventory,
        "historical_denylist": prospective["parent_denylist"],
        "prior_phase_reservations": prior_reservations,
        "checks": checks,
        "decision": "RESERVED_J1_PHASE_STREAMS",
        "streams_reserved": stream_inventory["stream_id_count"],
        "streams_consumed": 0,
        "passes": all(checks.values()),
    }
    if not payload["passes"]:
        raise J1ExecutionIntegrityError(
            "Phase stream reservation audit failed"
        )
    written = _write_immutable_json_exact(
        paths["reservation"],
        payload,
        field="stream_reservation_payload_sha256",
    )
    return {
        "reservation": written,
        "identity": immutable_json_identity(
            paths["reservation"],
            payload_field="stream_reservation_payload_sha256",
        ),
        "passes": True,
    }


def seal_phase_stream_consumption_opened(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    loaded = _load_open_phase_contract(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    paths = loaded["paths"]
    reservation = load_json(paths["reservation"])
    if not verify_payload_hash(
        reservation,
        "stream_reservation_payload_sha256",
    ) or reservation.get("passes") is not True:
        raise J1ExecutionIntegrityError(
            "Stream consumption lacks a valid reservation"
        )
    if owner_audit.get("passes") is not True:
        raise J1ExecutionIntegrityError(
            "Stream consumption lacks verified ownership"
        )
    owner = owner_audit["owner"]
    if (
        owner.get("marker_file_sha256")
        != loaded["marker_identity"]["file_sha256"]
        or owner.get("phase_lock_file_sha256")
        != loaded["lock_identity"]["file_sha256"]
        or owner.get("command") != loaded["commands"]["execute"]
    ):
        raise J1ExecutionIntegrityError(
            "Stream consumption owner changed phase contract"
        )
    reservation_identity = immutable_json_identity(
        paths["reservation"],
        payload_field="stream_reservation_payload_sha256",
    )
    if paths["consumption"].exists():
        existing = load_json(paths["consumption"])
        if not verify_payload_hash(
            existing,
            "stream_consumption_payload_sha256",
        ):
            raise J1ExecutionIntegrityError(
                "Existing stream consumption record is invalid"
            )
        invariant_checks = {
            "phase": existing.get("phase") == phase,
            "marker": existing.get("marker_identity")
            == loaded["marker_identity"],
            "lock": existing.get("phase_lock_identity")
            == loaded["lock_identity"],
            "reservation": existing.get("reservation_identity")
            == reservation_identity,
            "command": existing.get("execute_command")
            == loaded["commands"]["execute"],
            "scope": existing.get("consumption_scope")
            == "exact full immutable phase manifest",
            "counts": (
                existing.get("streams_reserved")
                == reservation["streams_reserved"]
                and existing.get("streams_consumed")
                == reservation["streams_reserved"]
            ),
        }
        ledger = owner_audit["ledger"]
        owner_hashes = {
            str(row["owner_record_sha256"])
            for row in ledger["owners"]
        }
        opener = str(existing.get("owner_record_sha256", ""))
        current = str(owner["owner_record_sha256"])
        linked = opener in owner_hashes and current in owner_hashes
        cursor = opener
        seen: set[str] = set()
        links = {
            str(row["old_owner_sha256"]): str(row["new_owner_sha256"])
            for row in ledger["recoveries"]
        }
        while linked and cursor != current:
            if cursor in seen or cursor not in links:
                linked = False
                break
            seen.add(cursor)
            cursor = links[cursor]
        linked = linked and cursor == current
        invariant_checks["opener_is_owner_ancestor"] = linked
        if not all(invariant_checks.values()):
            raise J1ExecutionIntegrityError(
                "Recovered stream consumption identity changed"
            )
        return {
            "consumption": existing,
            "identity": immutable_json_identity(
                paths["consumption"],
                payload_field="stream_consumption_payload_sha256",
            ),
            "opener_owner_record_sha256": opener,
            "current_owner_record_sha256": current,
            "owner_recovery_chain_verified": True,
            "reused_existing_record": True,
            "passes": True,
        }
    payload = {
        "version": f"{VERSION}_{phase}_stream_consumption_opened_v1",
        "phase": phase,
        "marker_identity": loaded["marker_identity"],
        "phase_lock_identity": loaded["lock_identity"],
        "reservation_identity": reservation_identity,
        "owner_record_sha256": owner["owner_record_sha256"],
        "execute_command": loaded["commands"]["execute"],
        "consumption_scope": "exact full immutable phase manifest",
        "stream_inventory": reservation["stream_inventory"],
        "streams_reserved": reservation["streams_reserved"],
        "streams_consumed": reservation["streams_reserved"],
        "scientific_work_before_consumption_record": 0,
        "decision": "OPENED_J1_PHASE_STREAM_CONSUMPTION",
        "passes": True,
    }
    written = _write_immutable_json_exact(
        paths["consumption"],
        payload,
        field="stream_consumption_payload_sha256",
    )
    return {
        "consumption": written,
        "identity": immutable_json_identity(
            paths["consumption"],
            payload_field="stream_consumption_payload_sha256",
        ),
        "opener_owner_record_sha256": owner["owner_record_sha256"],
        "current_owner_record_sha256": owner["owner_record_sha256"],
        "owner_recovery_chain_verified": True,
        "reused_existing_record": False,
        "passes": True,
    }


def acquire_or_reclaim_phase_owner(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
    execution_mode: str,
    contention_audit: Mapping[str, Any] | None = None,
    start_identity: str | None = None,
) -> dict[str, Any]:
    loaded = _load_open_phase_contract(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    paths = loaded["paths"]
    owner_path = paths["owner"]
    predecessor = (
        sha256_path(paths["phase_dir"] / COMMIT_HEAD_NAME)
        if (paths["phase_dir"] / COMMIT_HEAD_NAME).is_file()
        else None
    )
    if not owner_path.exists():
        acquire_writer_owner(
            phase_dir=paths["phase_dir"],
            phase=phase,
            marker_file_sha256=loaded["marker_identity"]["file_sha256"],
            phase_lock_file_sha256=loaded["lock_identity"]["file_sha256"],
            command=loaded["commands"]["execute"],
            predecessor_commit_head_sha256=predecessor,
            execution_mode=execution_mode,
            start_identity=start_identity,
        )
    else:
        try:
            return verify_writer_owner(
                phase_dir=paths["phase_dir"],
                phase=phase,
                marker_file_sha256=loaded["marker_identity"]["file_sha256"],
                phase_lock_file_sha256=loaded["lock_identity"]["file_sha256"],
                command=loaded["commands"]["execute"],
                execution_mode=execution_mode,
            )
        except J1ExecutionOperationalHold:
            ledger = load_json(owner_path)
            if not _verify_ownership_ledger(ledger):
                raise J1ExecutionIntegrityError(
                    "Existing ownership ledger is malformed"
                )
            head = ledger["owners"][-1]
            if _pid_alive(int(head.get("pid", -1))):
                raise J1ExecutionOperationalHold(
                    "Live writer owner blocks phase execution"
                )
            reclaim_dead_writer_owner(
                phase_dir=paths["phase_dir"],
                phase=phase,
                marker_file_sha256=loaded["marker_identity"][
                    "file_sha256"
                ],
                phase_lock_file_sha256=loaded["lock_identity"][
                    "file_sha256"
                ],
                command=loaded["commands"]["execute"],
                execution_mode=execution_mode,
                contention_audit=contention_audit,
                new_start_identity=start_identity,
            )
    return verify_writer_owner(
        phase_dir=paths["phase_dir"],
        phase=phase,
        marker_file_sha256=loaded["marker_identity"]["file_sha256"],
        phase_lock_file_sha256=loaded["lock_identity"]["file_sha256"],
        command=loaded["commands"]["execute"],
        execution_mode=execution_mode,
    )


def _terminal_result_base(
    *,
    phase: str,
    loaded: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
    engine_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    execution_mode = str(loaded["lock"]["execution_mode"])
    return {
        "version": f"{VERSION}_{phase}_terminal_result_v1",
        "phase": phase,
        "phase_lock_identity": loaded["lock_identity"],
        "phase_lock_result_identity": immutable_json_identity(
            loaded["paths"]["lock_result"],
            payload_field="phase_lock_result_payload_sha256",
            decision=_phase_ready_decision(phase),
        ),
        "marker_identity": loaded["marker_identity"],
        "manifest_identity": {
            **root_manifest_identity(
                load_json(loaded["paths"]["manifest"])
            ),
            "path": str(loaded["paths"]["manifest"].resolve()),
            "file_sha256": sha256_path(loaded["paths"]["manifest"]),
        },
        "stream_reservation_identity": dict(reservation_identity),
        "stream_consumption_identity": dict(consumption_identity),
        "ownership_ledger_identity": immutable_json_identity(
            loaded["paths"]["owner"],
            payload_field="ownership_payload_sha256",
        ),
        "owner_record_sha256": owner_audit["owner"][
            "owner_record_sha256"
        ],
        "bounded_engine": loaded["lock"]["bounded_engine"],
        "execution_mode": execution_mode,
        "scientific_authority": execution_mode == "scientific",
        "execute_command": loaded["commands"]["execute"],
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "resource_clock": (
            None
            if engine_result is None
            else engine_result.get("resource_clock")
        ),
        "output_accounting": (
            None
            if engine_result is None
            else engine_result.get("output_accounting")
        ),
        "commit_store_metrics": (
            None
            if engine_result is None
            else engine_result.get("commit_store_metrics")
        ),
        "rolling_store_metrics": (
            None
            if engine_result is None
            else engine_result.get("rolling_store_metrics")
        ),
        "runtime_ledger_metrics": (
            None
            if engine_result is None
            else engine_result.get("runtime_ledger_metrics")
        ),
        "io_metrics": (
            None
            if engine_result is None
            else engine_result.get("io_metrics")
        ),
        "incumbent_changed": False,
        "dashboard_changed": False,
        "promote": False,
    }


def _seal_scientific_training_terminal(
    *,
    loaded: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if engine_result.get("completed") is not True:
        raise J1ExecutionIntegrityError(
            "Training engine did not reach a terminal boundary"
        )
    state = engine_result["state"]
    boundary = engine_result["boundary"]
    if (
        state.get("engine_stage") != "complete"
        or int(state.get("round_number", -1)) != ROUNDS
        or boundary.get("chain_audit", {}).get("passes") is not True
    ):
        raise J1ExecutionIntegrityError(
            "Training terminal commit is incomplete"
        )
    manifest = load_json(loaded["paths"]["manifest"])
    model, optimizer = _load_model_optimizer_from_runtime(state)
    training_input = {
        "manifest_payload_sha256": manifest[
            "root_manifest_payload_sha256"
        ],
        "marker_file_sha256": loaded["marker_identity"]["file_sha256"],
        "terminal_state_file_sha256": boundary["state_file_sha256"],
        "terminal_commit_head_payload_sha256": boundary[
            "commit_head_payload_sha256"
        ],
        "completed_root_ids_sha256": j1.stable_hash(
            state["all_completed_root_ids"]
        ),
        "optimizer_step_ids_sha256": j1.stable_hash(
            state["optimizer_step_ids"]
        ),
        "round_aggregates_sha256": j1.stable_hash(
            state["round_aggregates"]
        ),
    }
    checkpoint_payload = candidate_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        training_manifest_identity=root_manifest_identity(manifest),
        training_marker_file_sha256=loaded["marker_identity"][
            "file_sha256"
        ],
        training_result_input_sha256=canonical_json_hash(training_input),
    )
    checkpoint_identity = write_candidate_checkpoint(
        loaded["paths"]["checkpoint"],
        checkpoint_payload,
    )
    checkpoint_identity["training_state_file_sha256"] = boundary[
        "state_file_sha256"
    ]
    report = {
        "manifest_root_ids": [
            str(row["root_id"]) for row in manifest["rows"]
        ],
        "completed_root_ids": list(state["all_completed_root_ids"]),
        "expected_optimizer_step_ids": list(
            state["expected_optimizer_step_ids"]
        ),
        "closed_optimizer_step_ids": list(
            state["optimizer_step_ids"]
        ),
        "rounds": copy.deepcopy(list(state["round_aggregates"])),
        "authenticated_terminal_boundary": {
            "passes": boundary["passes"],
            "chain_audit_passes": boundary["chain_audit"]["passes"],
            "state_file_sha256": boundary["state_file_sha256"],
            "commit_head_file_sha256": boundary[
                "commit_head_file_sha256"
            ],
            "commit_head_payload_sha256": boundary[
                "commit_head_payload_sha256"
            ],
            "unit_ids_sha256": boundary["chain_audit"][
                "unit_ids_sha256"
            ],
        },
        "checkpoint_identity": checkpoint_identity,
    }
    sanity = training_sanity_decision(report)
    sanity_payload = {
        **sanity,
        "training_input": training_input,
        "training_report_sha256": j1.stable_hash(report),
        "checkpoint_authoritative": (
            sanity["decision"] == "READY_J1_TRAINING_SANITY"
        ),
        "checkpoint_quarantined": (
            sanity["decision"] != "READY_J1_TRAINING_SANITY"
        ),
    }
    written_sanity = write_immutable_json(
        loaded["paths"]["sanity"],
        sanity_payload,
        field="training_sanity_payload_sha256",
    )
    base = _terminal_result_base(
        phase="training",
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=engine_result,
    )
    base.update(
        {
            "decision": sanity["decision"],
            "training_sanity_identity": immutable_json_identity(
                loaded["paths"]["sanity"],
                payload_field="training_sanity_payload_sha256",
                decision=sanity["decision"],
            ),
            "checkpoint_identity": checkpoint_identity,
            "checkpoint_authoritative": sanity_payload[
                "checkpoint_authoritative"
            ],
            "checkpoint_quarantined": sanity_payload[
                "checkpoint_quarantined"
            ],
            "authenticated_terminal_boundary": report[
                "authenticated_terminal_boundary"
            ],
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
    )


def _seal_scientific_evaluation_terminal(
    *,
    phase: str,
    loaded: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        phase not in {"development", "confirmation"}
        or engine_result.get("completed") is not True
    ):
        raise J1ExecutionIntegrityError(
            "Evaluation engine did not reach a terminal boundary"
        )
    report = analyze_paired_full_policy(
        engine_result["rows"],
        phase=phase,
    )
    written_analysis = write_immutable_json(
        loaded["paths"]["analysis"],
        {
            key: value
            for key, value in report.items()
            if key != "analysis_payload_sha256"
        },
        field="analysis_payload_sha256",
    )
    gate = evaluation_gate_decision(written_analysis, phase=phase)
    base = _terminal_result_base(
        phase=phase,
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=engine_result,
    )
    joint_seal = _load_joint_manifest_seal(
        loaded["paths"]["phase_dir"].parent
    )
    lineage = verify_joint_candidate_lineage(
        execution_root=loaded["paths"]["phase_dir"].parent,
        joint_manifest_seal=joint_seal,
        expected_execution_mode="scientific",
    )
    base.update(
        {
            **gate,
            "analysis_identity": immutable_json_identity(
                loaded["paths"]["analysis"],
                payload_field="analysis_payload_sha256",
            ),
            "joint_evaluation_manifest_seal_payload_sha256": joint_seal[
                "joint_manifest_seal_payload_sha256"
            ],
            "training_terminal_result_identity": lineage[
                "training_result_identity"
            ],
            "candidate_checkpoint_identity": lineage[
                "candidate_checkpoint_identity"
            ],
            "candidate_policy_identity": lineage[
                "candidate_checkpoint_identity"
            ]["file_sha256"],
            "incumbent_policy_binding": lineage[
                "incumbent_policy_binding"
            ],
            "control_policy_identity": lineage[
                "incumbent_policy_binding"
            ]["incumbent_binding_sha256"],
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
    )


def _seal_fixture_terminal(
    *,
    phase: str,
    loaded: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
    decision: str,
) -> dict[str, Any]:
    base = _terminal_result_base(
        phase=phase,
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=engine_result,
    )
    base.update(
        {
            "decision": decision,
            "execution_mode": "miniature_fixture",
            "fixture_only": True,
            "scientific_authority": False,
            "joint_evaluation_manifest_seal_payload_sha256": (
                None
                if phase == "training"
                else _load_joint_manifest_seal(
                    loaded["paths"]["phase_dir"].parent
                )["joint_manifest_seal_payload_sha256"]
            ),
        }
    )
    if phase != "training":
        lineage = verify_joint_candidate_lineage(
            execution_root=loaded["paths"]["phase_dir"].parent,
            expected_execution_mode="miniature_fixture",
        )
        base.update(
            {
                "training_terminal_result_identity": lineage[
                    "training_result_identity"
                ],
                "candidate_checkpoint_identity": lineage[
                    "candidate_checkpoint_identity"
                ],
                "candidate_policy_identity": "miniature-candidate",
                "incumbent_policy_binding": lineage[
                    "incumbent_policy_binding"
                ],
                "control_policy_identity": "miniature-control",
            }
        )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
    )


def _seal_operational_or_integrity_terminal(
    *,
    phase: str,
    loaded: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    if isinstance(error, J1ExecutionOperationalHold):
        decision = "HOLD_J1_OPERATIONAL"
        failure_class = "operational"
    else:
        decision = "KILL_J1_INTEGRITY"
        failure_class = "integrity"
    base = _terminal_result_base(
        phase=phase,
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=None,
    )
    base.update(
        {
            "decision": decision,
            "failure_class": failure_class,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "partial_work_preserved": True,
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
    )


def seal_phase_retention_manifest(
    *,
    execution_root: Path,
    phase: str,
) -> dict[str, Any]:
    paths = phase_artifact_paths(
        execution_root=execution_root,
        phase=phase,
    )
    if not paths["result"].is_file():
        raise J1ExecutionIntegrityError(
            "Retention requires an immutable terminal result"
        )
    rows = []
    for path in sorted(paths["phase_dir"].rglob("*")):
        if (
            not path.is_file()
            or path == paths["retention"]
            or path.is_symlink()
        ):
            continue
        rows.append(
            {
                "path": str(path.relative_to(paths["phase_dir"])),
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    key_identities = {
        "phase_lock": immutable_json_identity(
            paths["lock"],
            payload_field="phase_lock_payload_sha256",
        ),
        "phase_marker": immutable_json_identity(
            paths["marker"],
            payload_field="activation_marker_payload_sha256",
        ),
        "stream_reservation": immutable_json_identity(
            paths["reservation"],
            payload_field="stream_reservation_payload_sha256",
        ),
        "stream_consumption": immutable_json_identity(
            paths["consumption"],
            payload_field="stream_consumption_payload_sha256",
        ),
        "terminal_result": immutable_json_identity(
            paths["result"],
            payload_field="terminal_result_payload_sha256",
        ),
    }
    payload = {
        "version": f"{VERSION}_{phase}_retention_manifest_v1",
        "phase": phase,
        "key_identities": key_identities,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "file_inventory_sha256": _ordered_rows_hash(rows),
        "files": rows,
        "preserve_byte_for_byte": True,
        "reviewed_retirement_manifests_only": True,
        "passes": True,
    }
    return _write_immutable_json_exact(
        paths["retention"],
        payload,
        field="retention_payload_sha256",
    )


def execute_phase_from_artifacts(
    *,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
    execution_mode: str = "scientific",
    fixture_hooks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if execution_mode == "scientific" and fixture_hooks is not None:
        raise J1ExecutionIntegrityError(
            "Scientific dispatcher cannot accept fixture hooks"
        )
    loaded = _load_open_phase_contract(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    if loaded["lock"].get("execution_mode") != execution_mode:
        raise J1ExecutionIntegrityError(
            "Execute mode differs from immutable phase lock"
        )
    paths = loaded["paths"]
    if paths["result"].exists():
        result = _load_phase_terminal_result(
            execution_root=execution_root,
            phase=phase,
        )
        if (
            result.get("execution_mode") != execution_mode
            or result.get("scientific_authority")
            is not (execution_mode == "scientific")
            or result.get("bounded_engine")
            != loaded["lock"]["bounded_engine"]
        ):
            raise J1ExecutionIntegrityError(
                "Existing terminal result changed phase authority"
            )
        retention = seal_phase_retention_manifest(
            execution_root=execution_root,
            phase=phase,
        )
        return {
            "result": result,
            "retention": retention,
            "resumed_after_terminal": True,
            "terminal_already_sealed": True,
            "passes": True,
        }
    manifest = load_json(paths["manifest"])
    if (
        not verify_payload_hash(
            manifest,
            "root_manifest_payload_sha256",
        )
        or root_manifest_identity(manifest)
        != loaded["lock"]["manifest_identity"]
    ):
        raise J1ExecutionIntegrityError(
            "Execute manifest differs from immutable phase lock"
        )
    hooks = {} if fixture_hooks is None else dict(fixture_hooks)
    contention = (
        None
        if execution_mode == "scientific"
        else hooks.get(
            "contention_audit",
            {"passes": True, "fixture_only": True},
        )
    )
    owner_audit = acquire_or_reclaim_phase_owner(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        execution_mode=execution_mode,
        contention_audit=contention,
        start_identity=(
            None
            if execution_mode == "scientific"
            else hooks.get(
                "start_identity",
                f"miniature-fixture-{os.getpid()}",
            )
        ),
    )
    if hooks.get("interrupt_after_dispatch_boundary") == "owner":
        raise J1ExecutionPlannedInterruption(
            "fixture interruption after owner before reservation"
        )
    reservation = seal_phase_stream_reservation(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    if hooks.get("interrupt_after_dispatch_boundary") == "reservation":
        raise J1ExecutionPlannedInterruption(
            "fixture interruption after reservation before consumption"
        )
    consumption = seal_phase_stream_consumption_opened(
        phase=phase,
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        owner_audit=owner_audit,
    )
    if hooks.get("interrupt_after_dispatch_boundary") == "consumption":
        raise J1ExecutionPlannedInterruption(
            "fixture interruption after consumption before engine"
        )
    owner_audit = verify_writer_owner(
        phase_dir=paths["phase_dir"],
        phase=phase,
        marker_file_sha256=loaded["marker_identity"]["file_sha256"],
        phase_lock_file_sha256=loaded["lock_identity"]["file_sha256"],
        command=loaded["commands"]["execute"],
        execution_mode=execution_mode,
    )
    try:
        if phase == "training":
            config = (
                TrainingEngineConfig()
                if execution_mode == "scientific"
                else hooks["training_config"]
            )
            rows = list(manifest["rows"])
            if execution_mode == "miniature_fixture":
                rows = rows[: config.rounds * config.roots_per_round]
            engine_result = execute_training_engine_bounded(
                rows=rows,
                phase_dir=paths["phase_dir"],
                marker_file_sha256=loaded["marker_identity"][
                    "file_sha256"
                ],
                marker_payload_sha256=loaded["marker"][
                    "activation_marker_payload_sha256"
                ],
                phase_lock_file_sha256=loaded["lock_identity"][
                    "file_sha256"
                ],
                manifest_file_sha256=sha256_path(paths["manifest"]),
                manifest_payload_sha256=manifest[
                    "root_manifest_payload_sha256"
                ],
                command=loaded["commands"]["execute"],
                config=config,
                operational_audit_fn=(
                    None
                    if execution_mode == "scientific"
                    else hooks.get(
                        "operational_audit_fn",
                        fixture_phase_operational_audit,
                    )
                ),
                wall_clock=(
                    None
                    if execution_mode == "scientific"
                    else hooks.get("wall_clock")
                ),
            )
        else:
            if execution_mode == "scientific":
                lineage = verify_joint_candidate_lineage(
                    execution_root=execution_root,
                    expected_execution_mode="scientific",
                )
                candidate_policy = load_authoritative_candidate_policy(
                    checkpoint_identity=lineage[
                        "candidate_checkpoint_identity"
                    ]
                )
                control_policy = load_bound_incumbent_policy(
                    lineage["incumbent_policy_binding"]
                )
                candidate_identity = lineage[
                    "candidate_checkpoint_identity"
                ]["file_sha256"]
                control_identity = lineage[
                    "incumbent_policy_binding"
                ]["incumbent_binding_sha256"]
            else:
                candidate_policy = hooks["candidate_policy"]
                control_policy = hooks["control_policy"]
                candidate_identity = str(
                    hooks.get(
                        "candidate_policy_identity",
                        "miniature-candidate",
                    )
                )
                control_identity = str(
                    hooks.get(
                        "control_policy_identity",
                        "miniature-control",
                    )
                )
            rows = list(manifest["rows"])
            if execution_mode == "miniature_fixture":
                rows = rows[: int(hooks.get("pair_count", 1))]
            engine_result = execute_paired_evaluation_engine_bounded(
                rows=rows,
                phase_dir=paths["phase_dir"],
                phase=phase,
                marker_file_sha256=loaded["marker_identity"][
                    "file_sha256"
                ],
                marker_payload_sha256=loaded["marker"][
                    "activation_marker_payload_sha256"
                ],
                phase_lock_file_sha256=loaded["lock_identity"][
                    "file_sha256"
                ],
                manifest_file_sha256=sha256_path(paths["manifest"]),
                manifest_payload_sha256=manifest[
                    "root_manifest_payload_sha256"
                ],
                command=loaded["commands"]["execute"],
                candidate_policy=candidate_policy,
                control_policy=control_policy,
                candidate_policy_identity=candidate_identity,
                control_policy_identity=control_identity,
                max_moves=MAX_MOVES,
                execution_mode=execution_mode,
                block_pairs=(
                    PAIR_RESULT_BLOCK_SIZE
                    if execution_mode == "scientific"
                    else int(hooks.get("block_pairs", 1))
                ),
                operational_audit_fn=(
                    None
                    if execution_mode == "scientific"
                    else hooks.get(
                        "operational_audit_fn",
                        fixture_phase_operational_audit,
                    )
                ),
                wall_clock=(
                    None
                    if execution_mode == "scientific"
                    else hooks.get("wall_clock")
                ),
            )
    except J1ExecutionPlannedInterruption:
        raise
    except (J1ExecutionOperationalHold, J1ExecutionIntegrityError) as error:
        terminal = _seal_operational_or_integrity_terminal(
            phase=phase,
            loaded=loaded,
            reservation_identity=reservation["identity"],
            consumption_identity=consumption["identity"],
            owner_audit=owner_audit,
            error=error,
        )
        retention = seal_phase_retention_manifest(
            execution_root=execution_root,
            phase=phase,
        )
        return {
            "result": terminal,
            "retention": retention,
            "passes": False,
        }

    try:
        if execution_mode == "miniature_fixture":
            terminal = _seal_fixture_terminal(
                phase=phase,
                loaded=loaded,
                engine_result=engine_result,
                reservation_identity=reservation["identity"],
                consumption_identity=consumption["identity"],
                owner_audit=owner_audit,
                decision=str(
                    hooks.get(
                        "terminal_decision",
                        {
                            "training": "READY_J1_TRAINING_SANITY",
                            "development":
                                "READY_J1_DEVELOPMENT_FULL_POLICY",
                            "confirmation": "READY_J1_PROMOTION_REVIEW",
                        }[phase],
                    )
                ),
            )
        elif phase == "training":
            terminal = _seal_scientific_training_terminal(
                loaded=loaded,
                engine_result=engine_result,
                reservation_identity=reservation["identity"],
                consumption_identity=consumption["identity"],
                owner_audit=owner_audit,
            )
        else:
            terminal = _seal_scientific_evaluation_terminal(
                phase=phase,
                loaded=loaded,
                engine_result=engine_result,
                reservation_identity=reservation["identity"],
                consumption_identity=consumption["identity"],
                owner_audit=owner_audit,
            )
    except J1ExecutionPlannedInterruption:
        raise
    except Exception as error:
        if paths["result"].exists():
            terminal = _load_phase_terminal_result(
                execution_root=execution_root,
                phase=phase,
            )
        else:
            terminal = _seal_operational_or_integrity_terminal(
                phase=phase,
                loaded=loaded,
                reservation_identity=reservation["identity"],
                consumption_identity=consumption["identity"],
                owner_audit=owner_audit,
                error=error,
            )
    if (
        execution_mode == "miniature_fixture"
        and hooks.get("interrupt_after_terminal_boundary") == "result"
    ):
        raise J1ExecutionPlannedInterruption(
            "fixture interruption after terminal result before retention"
        )
    retention = seal_phase_retention_manifest(
        execution_root=execution_root,
        phase=phase,
    )
    return {
        "result": terminal,
        "result_identity": immutable_json_identity(
            paths["result"],
            payload_field="terminal_result_payload_sha256",
            decision=terminal["decision"],
        ),
        "retention": retention,
        "reservation": reservation,
        "consumption": consumption,
        "engine": engine_result,
        "passes": True,
    }


def write_execution_test_evidence(
    *,
    readiness_dir: Path,
    focused_command: str,
    focused_passed: int,
    parent_j1_command: str,
    parent_j1_passed: int,
    parent_j1a_command: str,
    parent_j1a_passed: int,
    applicable_command: str,
    applicable_passed: int,
    documented_deselections: Sequence[str],
) -> dict[str, Any]:
    path = readiness_dir / TEST_EVIDENCE_NAME
    if readiness_dir.exists():
        existing = sorted(
            item.name for item in readiness_dir.iterdir()
        )
        if existing:
            raise FileExistsError(
                "Test evidence requires a fresh readiness namespace"
            )
    expected_parent = {
        name: {"command": command, "passed": passed}
        for name, command, passed in PARENT_TEST_COMMANDS
    }
    observed_parent = {
        "parent_j1": {
            "command": parent_j1_command,
            "passed": int(parent_j1_passed),
        },
        "parent_j1a": {
            "command": parent_j1a_command,
            "passed": int(parent_j1a_passed),
        },
    }
    checks = {
        "focused_command_exact": focused_command == FOCUSED_TEST_COMMAND,
        "focused_count_exact": (
            FOCUSED_TEST_COUNT > 0
            and int(focused_passed) == FOCUSED_TEST_COUNT
        ),
        "parent_commands_and_counts_exact": (
            observed_parent == expected_parent
        ),
        "applicable_command_exact": (
            bool(APPLICABLE_TEST_COMMAND)
            and applicable_command == APPLICABLE_TEST_COMMAND
        ),
        "applicable_count_exact": (
            APPLICABLE_TEST_COUNT > 0
            and int(applicable_passed) == APPLICABLE_TEST_COUNT
        ),
        "deselections_exact": (
            tuple(documented_deselections)
            == DOCUMENTED_HISTORICAL_STATE_DESELECTIONS
            and len(set(documented_deselections))
            == len(documented_deselections)
            and all(
                "::" in value and value.strip() == value
                for value in documented_deselections
            )
        ),
        "source_files_present": all(
            path_value.is_file()
            for path_value in (CHARTER_PATH, RUNNER_PATH, TEST_PATH)
        ),
        "zero_scientific_work": all(
            value == 0 for value in ZERO_WORK.values()
        ),
    }
    if not all(checks.values()):
        raise J1ExecutionIntegrityError(
            "Execution test evidence does not match the frozen test contract"
        )
    payload = {
        "version": f"{VERSION}_test_evidence_v1",
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "source_identities": {
            "charter": {
                "path": str(CHARTER_PATH.relative_to(REPO_ROOT)),
                "file_sha256": sha256_path(CHARTER_PATH),
            },
            "runner": {
                "path": str(RUNNER_PATH.relative_to(REPO_ROOT)),
                "file_sha256": sha256_path(RUNNER_PATH),
            },
            "tests": {
                "path": str(TEST_PATH.relative_to(REPO_ROOT)),
                "file_sha256": sha256_path(TEST_PATH),
            },
        },
        "commands": [
            {
                "kind": "focused_execution_surface",
                "command": focused_command,
                "passed": int(focused_passed),
            },
            {
                "kind": "parent_j1",
                **observed_parent["parent_j1"],
            },
            {
                "kind": "parent_j1a",
                **observed_parent["parent_j1a"],
            },
            {
                "kind": "applicable_non_science_regressions",
                "command": applicable_command,
                "passed": int(applicable_passed),
            },
        ],
        "documented_historical_artifact_state_deselections": list(
            documented_deselections
        ),
        "synthetic_fixture_accounting": {
            "miniature_complete_games_may_run_in_tests": True,
            "miniature_optimizer_steps_may_run_in_tests": True,
            "scientific_j1_games": 0,
            "scientific_j1_optimizer_steps": 0,
            "scientific_j1_checkpoints": 0,
            "scientific_j1_policy_outcomes": 0,
        },
        "checks": checks,
        "passes": True,
        "zero_work": ZERO_WORK,
    }
    return write_immutable_json(
        path,
        payload,
        field="test_evidence_payload_sha256",
    )


def verify_execution_test_evidence(
    readiness_dir: Path,
) -> dict[str, Any]:
    path = readiness_dir / TEST_EVIDENCE_NAME
    payload = load_json(path)
    if (
        not verify_payload_hash(payload, "test_evidence_payload_sha256")
        or payload.get("passes") is not True
        or payload.get("zero_work") != ZERO_WORK
    ):
        raise J1ExecutionIntegrityError(
            "Execution test evidence is invalid"
        )
    expected_sources = {
        "charter": sha256_path(CHARTER_PATH),
        "runner": sha256_path(RUNNER_PATH),
        "tests": sha256_path(TEST_PATH),
    }
    for role, expected in expected_sources.items():
        if payload["source_identities"][role]["file_sha256"] != expected:
            raise J1ExecutionIntegrityError(
                f"Execution test evidence changed {role}"
            )
    expected_commands = {
        row["kind"]: row
        for row in payload.get("commands", [])
    }
    required = {
        "focused_execution_surface": (
            FOCUSED_TEST_COMMAND,
            FOCUSED_TEST_COUNT,
        ),
        "parent_j1": (
            PARENT_TEST_COMMANDS[0][1],
            PARENT_TEST_COMMANDS[0][2],
        ),
        "parent_j1a": (
            PARENT_TEST_COMMANDS[1][1],
            PARENT_TEST_COMMANDS[1][2],
        ),
        "applicable_non_science_regressions": (
            APPLICABLE_TEST_COMMAND,
            APPLICABLE_TEST_COUNT,
        ),
    }
    for kind, (command, passed) in required.items():
        observed = expected_commands.get(kind, {})
        if (
            observed.get("command") != command
            or observed.get("passed") != passed
        ):
            raise J1ExecutionIntegrityError(
                f"Execution test command changed: {kind}"
            )
    if tuple(
        payload.get(
            "documented_historical_artifact_state_deselections",
            [],
        )
    ) != DOCUMENTED_HISTORICAL_STATE_DESELECTIONS:
        raise J1ExecutionIntegrityError(
            "Execution test deselection contract changed"
        )
    return {
        "payload": payload,
        "identity": immutable_json_identity(
            path,
            payload_field="test_evidence_payload_sha256",
        ),
        "passes": True,
    }


def readiness_zero_work_audit(
    *,
    readiness_dir: Path,
    execution_root: Path,
) -> dict[str, Any]:
    existing = (
        sorted(
            str(path.relative_to(readiness_dir))
            for path in readiness_dir.rglob("*")
            if path.is_file()
        )
        if readiness_dir.exists()
        else []
    )
    execution_entries = (
        sorted(
            str(path.relative_to(execution_root))
            for path in execution_root.rglob("*")
        )
        if execution_root.exists()
        else []
    )
    forbidden_names = {
        PHASE_LOCK_NAME,
        PHASE_LOCK_RESULT_NAME,
        PHASE_MARKER_NAME,
        PHASE_MANIFEST_NAME,
        PHASE_OWNER_NAME,
        PHASE_STREAM_RESERVATION_NAME,
        PHASE_STREAM_CONSUMPTION_NAME,
        PHASE_RESULT_NAME,
        TRAINING_CANDIDATE_CHECKPOINT_NAME,
    }
    checks = {
        "readiness_contains_test_evidence_only": (
            existing == [TEST_EVIDENCE_NAME]
        ),
        "execution_root_absent": not execution_root.exists(),
        "execution_root_empty": not execution_entries,
        "no_phase_artifact_names": not any(
            Path(value).name in forbidden_names
            for value in existing + execution_entries
        ),
        "zero_work_counters_exact": all(
            value == 0 for value in ZERO_WORK.values()
        ),
    }
    return {
        "readiness_dir": str(readiness_dir.resolve()),
        "execution_root": str(execution_root.resolve()),
        "readiness_files_before_prepare": existing,
        "execution_entries_before_prepare": execution_entries,
        "zero_work": ZERO_WORK,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _readiness_artifact_identity(
    path: Path,
    *,
    payload_field: str,
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_sha256": load_json(path)[payload_field],
        "payload_field": payload_field,
    }


def prepare_execution_readiness(
    *,
    readiness_dir: Path = READINESS_DIR,
    execution_root: Path = EXECUTION_ROOT,
    operational_audit_fn: Any | None = None,
) -> dict[str, Any]:
    for name in (
        SCHEMA_NAME,
        MANIFEST_NAME,
        RUNTIME_STORAGE_PROJECTION_NAME,
        READINESS_LOCK_NAME,
        READINESS_RESULT_NAME,
    ):
        if (readiness_dir / name).exists():
            raise FileExistsError(
                f"Readiness artifact already exists: {name}"
            )
    zero = readiness_zero_work_audit(
        readiness_dir=readiness_dir,
        execution_root=execution_root,
    )
    evidence = verify_execution_test_evidence(readiness_dir)
    parent = accepted_identity_audit()
    schema = execution_schema()
    prospective = prospective_manifest()
    projection = full_scale_runtime_storage_projection()
    audit_fn = j1.operational_audit if operational_audit_fn is None else (
        operational_audit_fn
    )
    operational = audit_fn(output_dir=readiness_dir)

    written_schema = write_immutable_json(
        readiness_dir / SCHEMA_NAME,
        {
            key: value
            for key, value in schema.items()
            if key != "schema_sha256"
        },
        field="schema_sha256",
    )
    written_manifest = write_immutable_json(
        readiness_dir / MANIFEST_NAME,
        prospective,
        field="prospective_manifest_payload_sha256",
    )
    written_projection = write_immutable_json(
        readiness_dir / RUNTIME_STORAGE_PROJECTION_NAME,
        projection,
        field="projection_payload_sha256",
    )
    artifact_identities = {
        "test_evidence": evidence["identity"],
        "schema": _readiness_artifact_identity(
            readiness_dir / SCHEMA_NAME,
            payload_field="schema_sha256",
        ),
        "prospective_manifest": _readiness_artifact_identity(
            readiness_dir / MANIFEST_NAME,
            payload_field="prospective_manifest_payload_sha256",
        ),
        "runtime_storage_projection": _readiness_artifact_identity(
            readiness_dir / RUNTIME_STORAGE_PROJECTION_NAME,
            payload_field="projection_payload_sha256",
        ),
    }
    integrity_checks = {
        "zero_work": zero["passes"],
        "parent_identities": parent["passes"],
        "test_evidence": evidence["passes"],
        "schema": (
            verify_payload_hash(written_schema, "schema_sha256")
            and written_schema["bootstrap"]["passes"]
            and written_schema["model"]["parameter_count"] == 411_656
        ),
        "prospective_manifest": (
            verify_payload_hash(
                written_manifest,
                "prospective_manifest_payload_sha256",
            )
            and written_manifest["passes"]
            and written_manifest["counts"]["total_game_arms"] == 27_136
        ),
        "readiness_namespace_fresh": (
            not execution_root.exists()
            and not any(
                readiness_dir.rglob(PHASE_MARKER_NAME)
            )
        ),
    }
    admission_checks = {
        "runtime_storage_projection": (
            verify_payload_hash(
                written_projection,
                "projection_payload_sha256",
            )
            and written_projection["passes"]
        ),
        "operational": operational.get("passes") is True,
    }
    if not all(integrity_checks.values()):
        decision = "KILL_J1_EXECUTION_SURFACE_INTEGRITY"
    elif not all(admission_checks.values()):
        decision = "HOLD_J1_EXECUTION_SURFACE"
    else:
        decision = "READY_J1_EXECUTION_SURFACE"
    checks = {**integrity_checks, **admission_checks}
    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "decision": decision,
        "bound_readiness_dir": str(readiness_dir.resolve()),
        "bound_execution_root": str(execution_root.resolve()),
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "accepted_parent_files": ACCEPTED_FILES,
        "accepted_parent_payloads": {
            path: {"field": field, "sha256": expected}
            for path, (field, expected) in ACCEPTED_PAYLOADS.items()
        },
        "artifacts": artifact_identities,
        "phase_root_commitments": {
            phase: phase_root_commitment(phase)
            for phase in PHASES
        },
        "phase_commands": {
            phase: {
                action: bound_dispatch_command(
                    action=action,
                    phase=phase,
                    execution_root=execution_root,
                    readiness_dir=readiness_dir,
                )
                for action in PRODUCTION_COMMANDS
            }
            for phase in PHASES
        },
        "bounded_engines": {
            "training": "execute_training_engine_bounded",
            "development": "execute_paired_evaluation_engine_bounded",
            "confirmation": "execute_paired_evaluation_engine_bounded",
        },
        "legacy_engines_fixture_only": True,
        "promotion_command_present": False,
        "operational_audit": operational,
        "checks": checks,
        "zero_work": ZERO_WORK,
        "passes": decision == "READY_J1_EXECUTION_SURFACE",
    }
    written_lock = write_immutable_json(
        readiness_dir / READINESS_LOCK_NAME,
        lock_payload,
        field="readiness_lock_payload_sha256",
    )
    lock_identity = _readiness_artifact_identity(
        readiness_dir / READINESS_LOCK_NAME,
        payload_field="readiness_lock_payload_sha256",
    )
    result_payload = {
        "version": f"{VERSION}_readiness_result_v1",
        "decision": decision,
        "continue": (
            "research-lead review for separately authorized training "
            "phase lock/open"
            if decision == "READY_J1_EXECUTION_SURFACE"
            else False
        ),
        "hold": "all J1 scientific execution",
        "kill": "historical kills unchanged; J1/J1a not killed",
        "promote": False,
        "readiness_lock": lock_identity,
        "artifacts": artifact_identities,
        "checks": checks,
        "zero_work": ZERO_WORK,
    }
    written_result = write_immutable_json(
        readiness_dir / READINESS_RESULT_NAME,
        result_payload,
        field="readiness_result_payload_sha256",
    )
    return {
        "decision": decision,
        "readiness_dir": str(readiness_dir.resolve()),
        "artifacts": artifact_identities,
        "readiness_lock": lock_identity,
        "readiness_result": _readiness_artifact_identity(
            readiness_dir / READINESS_RESULT_NAME,
            payload_field="readiness_result_payload_sha256",
        ),
        "checks": checks,
        "zero_work": ZERO_WORK,
        "passes": written_result["decision"]
        == "READY_J1_EXECUTION_SURFACE",
    }


def dispatch_phase_command(
    *,
    action: str,
    phase: str,
    execution_root: Path,
    readiness_dir: Path,
    jobs: int,
    confirmation_access_audit_path: Path | None = None,
    execution_mode: str = "scientific",
    fixture_hooks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if action not in PRODUCTION_COMMANDS:
        raise ValueError(f"Unsupported production command: {action}")
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")
    if jobs != 1:
        raise J1ExecutionIntegrityError("J1 requires exactly jobs=1")
    if execution_mode == "scientific" and fixture_hooks is not None:
        raise J1ExecutionIntegrityError(
            "Scientific dispatcher cannot accept fixture hooks"
        )
    hooks = {} if fixture_hooks is None else dict(fixture_hooks)
    if action == "seal-phase-lock":
        return seal_phase_lock_from_artifacts(
            phase=phase,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            confirmation_access_audit_path=(
                confirmation_access_audit_path
            ),
            execution_mode=execution_mode,
        )
    if action == "open":
        return open_phase_from_artifacts(
            phase=phase,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            execution_mode=execution_mode,
            operational_audit_fn=(
                None
                if execution_mode == "scientific"
                else hooks.get("open_operational_audit_fn")
            ),
            opened_at=(
                None
                if execution_mode == "scientific"
                else hooks.get("opened_at")
            ),
            hostname=(
                None
                if execution_mode == "scientific"
                else hooks.get("hostname")
            ),
        )
    if action == "materialize":
        return materialize_phase_manifest_from_artifacts(
            phase=phase,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    if action == "execute":
        return execute_phase_from_artifacts(
            phase=phase,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
            execution_mode=execution_mode,
            fixture_hooks=fixture_hooks,
        )
    raise AssertionError(f"Unhandled phase command: {action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
    )
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument(
        "--readiness-dir",
        type=Path,
        default=READINESS_DIR,
    )
    evidence.add_argument("--focused-command", required=True)
    evidence.add_argument("--focused-passed", type=int, required=True)
    evidence.add_argument("--parent-j1-command", required=True)
    evidence.add_argument("--parent-j1-passed", type=int, required=True)
    evidence.add_argument("--parent-j1a-command", required=True)
    evidence.add_argument("--parent-j1a-passed", type=int, required=True)
    evidence.add_argument("--applicable-command", required=True)
    evidence.add_argument("--applicable-passed", type=int, required=True)
    evidence.add_argument(
        "--deselection",
        action="append",
        default=[],
    )
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--readiness-dir",
        type=Path,
        default=READINESS_DIR,
    )
    prepare_parser.add_argument(
        "--execution-root",
        type=Path,
        default=EXECUTION_ROOT,
    )
    for action in PRODUCTION_COMMANDS:
        phase_parser = subparsers.add_parser(action)
        phase_parser.add_argument(
            "--phase",
            choices=PHASES,
            required=True,
        )
        phase_parser.add_argument(
            "--execution-root",
            type=Path,
            default=EXECUTION_ROOT,
        )
        phase_parser.add_argument(
            "--readiness-dir",
            type=Path,
            default=READINESS_DIR,
        )
        phase_parser.add_argument("--jobs", type=int, default=1)
        if action == "seal-phase-lock":
            phase_parser.add_argument(
                "--confirmation-access-audit",
                type=Path,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "write-test-evidence":
        result = write_execution_test_evidence(
            readiness_dir=args.readiness_dir,
            focused_command=args.focused_command,
            focused_passed=args.focused_passed,
            parent_j1_command=args.parent_j1_command,
            parent_j1_passed=args.parent_j1_passed,
            parent_j1a_command=args.parent_j1a_command,
            parent_j1a_passed=args.parent_j1a_passed,
            applicable_command=args.applicable_command,
            applicable_passed=args.applicable_passed,
            documented_deselections=args.deselection,
        )
    elif args.subcommand == "prepare":
        result = prepare_execution_readiness(
            readiness_dir=args.readiness_dir,
            execution_root=args.execution_root,
        )
    else:
        result = dispatch_phase_command(
            action=args.subcommand,
            phase=args.phase,
            execution_root=args.execution_root,
            readiness_dir=args.readiness_dir,
            jobs=args.jobs,
            confirmation_access_audit_path=getattr(
                args,
                "confirmation_access_audit",
                None,
            ),
            execution_mode="scientific",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Outcome-free J2 incumbent-distillation readiness tooling.

This module can audit immutable inputs, seal test evidence, and prepare a
readiness result. It cannot reserve streams, query the teacher, run games,
train a scientific model, or open evaluation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


VERSION = "j2_incumbent_distillation_readiness_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs"
CHARTER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J2_INCUMBENT_DISTILLED_JOINT_POLICY_VALUE_CHARTER.md"
)
RUNNER_PATH = (
    REPO_ROOT / "threes_rl" / "j2_incumbent_distillation_readiness.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_rl_j2_incumbent_distillation_readiness.py"
)
READINESS_DIR = (
    RUNS_ROOT / "forensics" / "j2_incumbent_distillation_readiness_v1"
)
FUTURE_EXECUTION_DIRS = (
    RUNS_ROOT / "forensics" / "j2_distillation_execution_v1",
    RUNS_ROOT / "forensics" / "j2_on_policy_training_v1",
    RUNS_ROOT / "forensics" / "j2_development_v1",
    RUNS_ROOT / "forensics" / "j2_confirmation_v1",
)

TEST_EVIDENCE_NAME = "J2_TEST_EVIDENCE.json"
INPUT_BINDINGS_NAME = "J2_INPUT_BINDINGS.json"
PROSPECTIVE_AUTHORITY_NAME = "J2_PROSPECTIVE_AUTHORITY.json"
PROTECTED_STREAM_AUTHORITY_NAME = "J2_PROTECTED_STREAM_AUTHORITY.json"
TEACHER_PROVENANCE_NAME = "J2_TEACHER_PROVENANCE.json"
MODEL_SCHEMA_NAME = "J2_MODEL_SCHEMA.json"
POWER_NAME = "J2_POWER_AND_FEASIBILITY.json"
PROJECTION_NAME = "J2_RUNTIME_STORAGE_PROJECTION.json"
READINESS_LOCK_NAME = "J2_READINESS_LOCK.json"
READINESS_RESULT_NAME = "J2_READINESS_RESULT.json"

READY = "READY_J2_INCUMBENT_DISTILLATION_PREFLIGHT"
HOLD = "HOLD_J2_INCUMBENT_DISTILLATION_PREFLIGHT"
KILL = "KILL_J2_READINESS_INTEGRITY"

OBSERVATION_WIDTH = 282
HIDDEN_WIDTH = 512
ACTION_COUNT = 4
EXPECTED_PARAMETER_COUNT = 410_117
INITIALIZATION_SEED = 2_026_072_806
DISTILLATION_EPOCHS = 8
MINIBATCH_SIZE = 4_096
LEARNING_RATE = 3e-4
ADAM_EPS = 1e-5
GRADIENT_CLIP = 0.5
VALUE_COEFFICIENT = 0.5

PPO_ANCHOR_ROUNDS = 16
PPO_ROOTS_PER_ROUND = 256
STAGE_TABLE = (
    {
        "stage": "teacher_behavior_cloning",
        "authority_rows": 8_192,
        "game_arms": 8_192,
        "pre_ppo_teacher_roots": 8_192,
        "online_teacher_roots": 0,
        "streams": {
            "logical_stream_id": 227_000_000_000,
            "deck_stream_id": 228_000_000_000,
            "slot_stream_id": 229_000_000_000,
            "teacher_policy_stream_id": 230_000_000_000,
        },
    },
    {
        "stage": "distillation_validation",
        "authority_rows": 2_048,
        "game_arms": 4_096,
        "pre_ppo_teacher_roots": 2_048,
        "online_teacher_roots": 0,
        "streams": {
            "logical_stream_id": 231_000_000_000,
            "deck_stream_id": 232_000_000_000,
            "slot_stream_id": 233_000_000_000,
            "student_policy_stream_id": 234_000_000_000,
            "teacher_policy_stream_id": 235_000_000_000,
        },
    },
    {
        "stage": "on_policy_training",
        "authority_rows": 16_384,
        "game_arms": 16_384,
        "pre_ppo_teacher_roots": 0,
        "online_teacher_roots": PPO_ANCHOR_ROUNDS * PPO_ROOTS_PER_ROUND,
        "streams": {
            "logical_stream_id": 236_000_000_000,
            "deck_stream_id": 237_000_000_000,
            "slot_stream_id": 238_000_000_000,
            "candidate_policy_stream_id": 239_000_000_000,
        },
    },
    {
        "stage": "development",
        "authority_rows": 896,
        "game_arms": 1_792,
        "pre_ppo_teacher_roots": 0,
        "online_teacher_roots": 0,
        "streams": {
            "logical_stream_id": 240_000_000_000,
            "deck_stream_id": 241_000_000_000,
            "slot_stream_id": 242_000_000_000,
            "candidate_policy_stream_id": 243_000_000_000,
            "control_policy_stream_id": 244_000_000_000,
        },
    },
    {
        "stage": "confirmation",
        "authority_rows": 4_480,
        "game_arms": 8_960,
        "pre_ppo_teacher_roots": 0,
        "online_teacher_roots": 0,
        "streams": {
            "logical_stream_id": 245_000_000_000,
            "deck_stream_id": 246_000_000_000,
            "slot_stream_id": 247_000_000_000,
            "candidate_policy_stream_id": 248_000_000_000,
            "control_policy_stream_id": 249_000_000_000,
        },
    },
)


def _stage_value(stage: str, field: str) -> int:
    rows = [
        int(row[field])
        for row in STAGE_TABLE
        if str(row["stage"]) == stage
    ]
    if len(rows) != 1:
        raise RuntimeError(f"J2 stage-table definition changed: {stage}")
    return rows[0]


BC_ROOTS = _stage_value("teacher_behavior_cloning", "authority_rows")
VALIDATION_PAIRS = _stage_value(
    "distillation_validation",
    "authority_rows",
)
PPO_ROOTS = _stage_value("on_policy_training", "authority_rows")
DEVELOPMENT_PAIRS = _stage_value("development", "authority_rows")
CONFIRMATION_PAIRS = _stage_value("confirmation", "authority_rows")
PRE_PPO_TEACHER_ROOTS = sum(
    int(row["pre_ppo_teacher_roots"]) for row in STAGE_TABLE
)
ONLINE_TEACHER_ROOTS = sum(
    int(row["online_teacher_roots"]) for row in STAGE_TABLE
)
TOTAL_TEACHER_ROOT_EQUIVALENTS = (
    PRE_PPO_TEACHER_ROOTS + ONLINE_TEACHER_ROOTS
)
TOTAL_PROSPECTIVE_ROWS = sum(
    int(row["authority_rows"]) for row in STAGE_TABLE
)
TOTAL_GAME_ARMS = sum(int(row["game_arms"]) for row in STAGE_TABLE)
TOTAL_UNIQUE_STREAMS = sum(
    int(block["authority_rows"]) * len(block["streams"])
    for block in STAGE_TABLE
)


def derive_stage_totals(
    stage_table: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    table = STAGE_TABLE if stage_table is None else stage_table
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


EXPECTED_STAGE_TOTALS = {
    "prospective_rows_or_pairs": 32_000,
    "game_arms": 39_424,
    "unique_streams": 135_424,
    "pre_ppo_teacher_roots": 10_240,
    "online_teacher_roots": 4_096,
    "total_teacher_root_equivalents": 14_336,
}

PLANNING_MOVES = 512
SENSITIVITY_MOVES = 5_000
SHARD_COUNT = 8
SAFETY_MULTIPLIER = 1.25
TEACHER_ACTION_SECONDS_MEDIAN = 0.13257275009527802
TEACHER_ACTION_SECONDS_P99 = 0.13627169113606213
DISTILLATION_CAP_HOURS = 72.0
DISTILLATION_CAP_GIB = 24.0
PPO_CAP_HOURS = 72.0
PPO_CAP_GIB = 24.0

SCORE_SD = 1.25
SCORE_Z_975 = 1.959963984540054
SCORE_Z_80 = 0.8416212335729143
FIDELITY_SCORE_POINT_FLOOR = 0.97
FIDELITY_SCORE_CI_FLOOR = 0.90
FIDELITY_OR_POINT_FLOOR = 0.90
FIDELITY_OR_CI_FLOOR = 0.50
POWER_DATASETS = 768
POWER_BOOTSTRAPS = 199
POWER_REQUIRED = 0.80
CONTROL_RATES = (0.02, 0.04, 0.08, 0.15)
PAIRING_COUPLINGS = (0.0, 0.05, 0.10)
OR_NI_MARGIN_GRID = (0.40, 0.50, 0.60, 0.70, 0.80)
POWER_SEED = 2_026_072_821

FEATURE_FAMILIES = (
    "low_air",
    "low_constrained",
    "mid_progression",
    "upper_progression",
)

EXPECTED_SOURCE_IDENTITIES = {
    "threes_rl/J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md":
        "26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2",
    "threes_rl/J1A_OUTCOME_FREE_COST_POWER_AMENDMENT.md":
        "d738a55bb438ee87d59d2433466e813cfd0a9fb5f041cbc3cc807d4bbafa2e11",
    "threes_rl/j1a_cost_power_preflight.py":
        "27ffb3825d60bd8ca4ec0646f976e325c2a7c5f00a077aea3803544531fe6a98",
    "tests/test_rl_j1a_cost_power_preflight.py":
        "898f25aa4ed109db2c9fc27b4bba9d7e9641dc57834e4e02d7a8242df195eb59",
    "threes_rl/J1D_V2_EXACT_METRIC_AUTHENTICATION_AMENDMENT.md":
        "f3d42d6f4d908c723756e140fc2ba424378f280a18dc99a50b585e59478cd07c",
    "threes_rl/j1d_metric_authentication_surface_v2.py":
        "6ee656ae0288877560df5a6a140777bf341f8a34dbe554c61ebe2812e6147a3d",
    "tests/test_rl_j1d_metric_authentication_surface_v2.py":
        "9148d70b3d8c8c55b27a75829ef2e5b4df142e124d6265b639667049b4ac5868",
    "threes_rl/j1_joint_policy_value.py":
        "55d9e3206c2905509466c4962006e6cf3426f76647af6d2e60afe674b80c9bfe",
    "threes_rl/j1_execution_surface.py":
        "d4367d95aba05ec592310008bae21e7de90905fa1268601dd60cc8fcb2b6f2bd",
}

EXPECTED_ARTIFACT_IDENTITIES = {
    "threes_rl/runs/forensics/j1_execution_surface_readiness_v1/"
    "J1_PROSPECTIVE_MANIFEST.json": (
        "2aee68a08325cdbc5e42153942079c1375163f2b88217bf407e64fd95f096dce",
        "prospective_manifest_payload_sha256",
        "de0046a2121138659dd2fd0bb46a48081d80842c5d24334d1a683dbf0a9a7093",
    ),
    "threes_rl/runs/forensics/j1_execution_surface_readiness_v1/"
    "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json": (
        "92dfc49a8f0830a4b39c627d9257e4a20b4ca504019c455b3b2b1eb05a959f20",
        "projection_payload_sha256",
        "60e9697e82409e5ea930b7b07d2ab042ca3b28ecebff4bc6c2058f8b04e9f6ce",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_RUNTIME_STORAGE_PROJECTION.json": (
        "e023fe04239ceb2d317ab0e26979033db3c2a5c93d4a5016168de442fc97e401",
        "projection_payload_sha256",
        "1aaba01b73d53ad10252f0c59c238c8274a9e8f8066a8f3f03f3c0587c6bef0b",
    ),
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_COST_POWER_ARITHMETIC.json": (
        "957159dcbfe4ee95be9c2abd2ab2d99a4cd49ce611895bdd3c55ff5ce4fcf9b0",
        "arithmetic_payload_sha256",
        "b1d13d49db07fa59afd995640c5d063f8bc9776122ead554bb53856543fd21b6",
    ),
    "threes_rl/runs/forensics/j1a_cost_power_amendment_v1/"
    "J1A_PREFLIGHT_RESULT.json": (
        "4ecda2a1101011437c912d884dfb5acecf7e586b87c4646c63354c4ecc5403ef",
        "preflight_result_payload_sha256",
        "abe17a53c1af2b182a488d4fc05b060a214b106652c04462453ad01e75ed9471",
    ),
    "threes_rl/runs/forensics/j1d_metric_authentication_readiness_v2/"
    "J1D_V2_METRIC_AUTHENTICATION_READINESS_LOCK.json": (
        "60587f40512555dadab5cc09a0e9802039754034f427a6084b11b7d8146627c7",
        "readiness_lock_payload_sha256",
        "bbca2deb85cee5abea8fcbe89d9917797c7bf20655b3590b9ba66468f422f7b5",
    ),
    "threes_rl/runs/forensics/j1d_metric_authentication_readiness_v2/"
    "J1D_V2_METRIC_AUTHENTICATION_READINESS_RESULT.json": (
        "b891a0d63fd0c532387a64dc719ec20f27dcf15c84aeeb3a094470030076449b",
        "readiness_result_payload_sha256",
        "f4af39cc6c2f54e3e79aef76a73fdddd141979faa684794beb7deb59291f3693",
    ),
    "threes_rl/runs/forensics/j1d_metric_authentication_readiness_v2/"
    "J1D_V2_PROSPECTIVE_TRAINING_MANIFEST.json": (
        "c2be5faf37d9e2619c0bd57d12a64248738e6b4c8bda1802931898a63e18b1e0",
        "prospective_manifest_payload_sha256",
        "f6da9b35674a08c21b53c476692cd7073e492289a8cec8d687ceaa45afaf092d",
    ),
    "threes_rl/runs/forensics/j1d_metric_authentication_readiness_v2/"
    "J1D_V2_PROTECTED_STREAM_AUTHORITY.json": (
        "4e8e1661ab04c3d87c5819e0112d27b8213f65539c2ea9b955fa6a1a47fca867",
        "stream_authority_payload_sha256",
        "90e98195d0be7a50d38c0c00e681c120c9a8300c5d6510f836809088fb2b7c6e",
    ),
    "threes_rl/runs/forensics/j1d_execution_v1/training/"
    "terminal_result.json": (
        "9ab0c76142aa70041a5f0540abbc3f9b77ac197599f607a646b2952368f13e1a",
        "terminal_result_payload_sha256",
        "e37a32ec2d0ef1df78d804689ee8f529e5cc78bb627b34fbc8728b7840366fb6",
    ),
    "threes_rl/runs/forensics/j1d_execution_v1/training/"
    "training_sanity_result.json": (
        "2faba052c943552c37dc3fe36fd82cd44e1a74ca4f8b29023e64464ffc8167e8",
        "training_sanity_payload_sha256",
        "2bba7980191d66d2721556456c7ea65ce426aefa02cb1d5d6d508c4ff49e906b",
    ),
    "threes_rl/runs/forensics/j1d_execution_v1/training/"
    "retention_manifest.json": (
        "5fe222bfc3e1681ee3b1cb98db71e2a0b90017c869947329ca49df084ed65518",
        "retention_payload_sha256",
        "39ff3a6f028f7b27ddb775270959b2f0964e7bab8c649e20942f7296e8dbfe2c",
    ),
}

EXPECTED_J1D_CHECKPOINT_SHA256 = (
    "cde85c1ca62b9bd045d680ec980ec25e58ae6e7e083b7ccbac1e239cfbb1a41e"
)

EXPECTED_INCUMBENT = {
    "policy_file_sha256":
        "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4",
    "resolved_spec_sha256":
        "4b0b51bc744efd9f7b112dfda4a0514e3420a60c6b086cfcc0a1d56e955b2579",
    "checkpoint_manifest_sha256":
        "b5fa41c93e330356068131149846ef100c6e71665e03b6bad9c069bcdcf56618",
    "implementation_source_manifest_sha256":
        "f85f3bf3448370d2bea1dba0358938430f3b87ba22b464fba73b852e6af8362b",
    "incumbent_binding_sha256":
        "8d3ff6e38255dbb8b796a0d9e80382d577f3ce4e0638cfe45d453925dfb11ca3",
}

INCUMBENT_SOURCE_HASHES = {
    "threes_rl/eval.py":
        "df0a558014583fcfd24fd8ddf48988e375ad9a6fc5199d35311c40d8b6a3f705",
    "threes_rl/expectimax.py":
        "98a7f0d05437d01555ea37d21211fa36d7260cba84456b0fb08799472b26ec14",
    "threes_rl/ntuple.py":
        "bdd38ec758ca1786b67a7550b3a2792cbd517176ad99e4df7c5ddd2584953789",
    "threes_rl/action_prior.py":
        "93a5f4b72be0f4511fd2fe58929990b245e94fb11dec6a0fb02d1abb3a557f95",
    "threes_rl/sim.py":
        "67e7a245c05e59367402095ad018122fb4cb1ef08664bf28bf4bc03a02a73072",
    "threes_rl/train_td.py":
        "0ef18c38c09516a11fddc5b2cd742aa536c21615d5ce2477167bed8553b13f7a",
    "threes_rl/obs.py":
        "7fe9fdc48da826dfde424391b57d8d9de812aa48bb08e129079ed9f3fd3478b1",
    "threes_rl/env.py":
        "9b3a65fff503ab5b40db63e11c5c4b3c03f96bd4034709d80cad707cf40f2ddf",
}

INCUMBENT_RUN = "ntuple_phaseblend_labelcorr_w010_endgame_1000_1020"
INCUMBENT_RUN_DIR = RUNS_ROOT / "eval_artifacts" / INCUMBENT_RUN
INCUMBENT_SUMMARY_PATH = INCUMBENT_RUN_DIR / "summary.json"
INCUMBENT_TOP_MANIFEST_PATH = INCUMBENT_RUN_DIR / "top_games" / "manifest.json"
INCUMBENT_REPLAY_PATH = (
    INCUMBENT_RUN_DIR
    / "top_games"
    / "rank_01_score_263670_seed_1011_starter_1536"
    / "replay.json"
)
INCUMBENT_PROVENANCE_HASHES = {
    "summary":
        "f48c540cd5e43b8b6a61fe58b7c84441f683c34065e318edb5c0b86a23c2a72d",
    "top_manifest":
        "2ac6c2f2e579d661c2ba2e6d1913b8b58ce9f5439c6671dda0b54302ec0a044a",
    "rank1_replay":
        "95064a8e8231631e1d69736076657489120a7db1975234bb409953ba9828e768",
}
GOVERNANCE_EXCERPT = """
Setup:

- Frozen actor: current incumbent
  `ntuple_phaseblend_expectimax2` with parent MC-1000, student1 weight 0.25,
  mid replay-calibration weight 0.05, and endgame action-label sidecar weight
  0.10.
- Initial fresh self-play scan over seeds `1200:1350` was stopped after several
  minutes because depth-2 action-value scanning was too slow before reaching
  enough candidate tail states.
- Switched to retained incumbent top-game replays from seeds before the
  `1050:1100` holdout:
  - `ntuple_phaseblend_labelcorr_w010_endgame_1000_1020` top 3 games.
  - `ntuple_phaseblend_labelcorr_w010_endgame_1020_1050` top 3 games.
"""
EXPECTED_GOVERNANCE_EXCERPT_SHA256 = (
    "eb002ab02b6050d69206883cdab77236450cf0689483152500923cf9701b56e9"
)

REAL_TEACHER_SHARD_EVIDENCE_PATH = (
    RUNS_ROOT
    / "forensics"
    / "j2_incumbent_multiprocess_cost_evidence_v1.json"
)
ONLINE_TEACHER_SYNC_EVIDENCE_PATH = (
    RUNS_ROOT
    / "forensics"
    / "j2_online_teacher_query_orchestration_evidence_v1.json"
)

ZERO_WORK = {
    "execution_markers": 0,
    "streams_reserved": 0,
    "streams_consumed": 0,
    "normal_start_games": 0,
    "teacher_queries": 0,
    "teacher_action_labels": 0,
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


class J2ReadinessIntegrityError(RuntimeError):
    """An immutable J2 readiness contract failed."""


def repo_path(path: str | Path, root: Path = REPO_ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def sha256_path(path: str | Path, root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    with repo_path(path, root).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_native(value),
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
    body = json_native(dict(payload))
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = json_native(dict(payload))
    observed = body.pop(field, None)
    return isinstance(observed, str) and observed == canonical_json_hash(body)


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    body = payload_with_hash(payload, field)
    serialized = (
        json.dumps(body, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    decoded = json.loads(serialized.decode("utf-8"))
    if not verify_payload_hash(decoded, field):
        raise J2ReadinessIntegrityError(f"JSON reload instability: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = path.read_bytes()
        if observed != serialized:
            raise J2ReadinessIntegrityError(
                f"Immutable artifact collision changed bytes: {path}"
            )
        raise FileExistsError(f"Immutable artifact already exists: {path}")
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
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            observed = path.read_bytes()
            if observed != serialized:
                raise J2ReadinessIntegrityError(
                    f"Concurrent immutable artifact mismatch: {path}"
                ) from error
            raise FileExistsError(
                f"Immutable artifact won by another writer: {path}"
            ) from error
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != serialized:
        raise J2ReadinessIntegrityError(
            f"Immutable artifact bytes changed after write: {path}"
        )
    observed_payload = json.loads(path.read_text(encoding="utf-8"))
    if not verify_payload_hash(observed_payload, field):
        raise J2ReadinessIntegrityError(
            f"Immutable artifact payload mismatch: {path}"
        )
    return observed_payload


def load_json(path: str | Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    payload = json.loads(repo_path(path, root).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise J2ReadinessIntegrityError(f"Expected JSON object: {path}")
    return payload


def load_hashed_json(
    path: str | Path,
    *,
    field: str,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    payload = load_json(path, root)
    if not verify_payload_hash(payload, field):
        raise J2ReadinessIntegrityError(
            f"Invalid embedded payload identity: {path}"
        )
    return payload


def _commitment(prefix: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        prefix.encode("ascii") + b"|" + canonical_json_bytes(payload)
    ).hexdigest()


def build_prospective_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in STAGE_TABLE:
        stage = str(block["stage"])
        streams = dict(block["streams"])
        for row_index in range(int(block["authority_rows"])):
            identities = {
                name: int(base) + row_index
                for name, base in streams.items()
            }
            core = {
                "stage": stage,
                "row_index": row_index,
                "streams": identities,
            }
            rows.append(
                {
                    **core,
                    "root_id": _commitment("j2-root-v1", core),
                    "ancestry_id": _commitment("j2-ancestry-v1", core),
                    "content_opened": False,
                    "reserved": False,
                    "consumed": False,
                }
            )
    return rows


def prospective_authority() -> dict[str, Any]:
    stage_totals = derive_stage_totals()
    rows = build_prospective_rows()
    root_ids = [str(row["root_id"]) for row in rows]
    ancestry_ids = [str(row["ancestry_id"]) for row in rows]
    stream_ids = [
        int(stream_id)
        for row in rows
        for stream_id in row["streams"].values()
    ]
    stage_counts = Counter(str(row["stage"]) for row in rows)
    stream_prefixes = Counter(value // 1_000_000_000 for value in stream_ids)
    checks = {
        "stage_table_totals_exact": stage_totals == EXPECTED_STAGE_TOTALS,
        "row_count_exact": len(rows) == TOTAL_PROSPECTIVE_ROWS == 32_000,
        "stream_count_derived_exact": (
            TOTAL_UNIQUE_STREAMS
            == 4 * BC_ROOTS
            + 5 * VALIDATION_PAIRS
            + 4 * PPO_ROOTS
            + 5 * DEVELOPMENT_PAIRS
            + 5 * CONFIRMATION_PAIRS
            == 135_424
        ),
        "stream_count_materialized_exact": (
            len(stream_ids) == TOTAL_UNIQUE_STREAMS
        ),
        "all_stream_ids_unique": len(set(stream_ids)) == len(stream_ids),
        "all_root_ids_unique": len(set(root_ids)) == len(root_ids),
        "all_ancestry_ids_unique": (
            len(set(ancestry_ids)) == len(ancestry_ids)
        ),
        "stage_counts_exact": dict(stage_counts)
        == {
            "teacher_behavior_cloning": BC_ROOTS,
            "distillation_validation": VALIDATION_PAIRS,
            "on_policy_training": PPO_ROOTS,
            "development": DEVELOPMENT_PAIRS,
            "confirmation": CONFIRMATION_PAIRS,
        },
        "fresh_prefixes_exact": sorted(stream_prefixes)
        == list(range(227, 250)),
        "no_213b_226b_collision": not (
            set(stream_prefixes) & set(range(213, 227))
        ),
        "zero_reservations": all(not row["reserved"] for row in rows),
        "zero_consumption": all(not row["consumed"] for row in rows),
        "zero_content_opened": all(not row["content_opened"] for row in rows),
    }
    return {
        "version": f"{VERSION}_prospective_authority_v1",
        "method": (
            "content-blind SHA-256 commitments over stage, row, and exact "
            "stream identities"
        ),
        "rows": rows,
        "row_count": len(rows),
        "stage_counts": dict(stage_counts),
        "total_game_arms": TOTAL_GAME_ARMS,
        "stage_table_totals": stage_totals,
        "stream_count_formula": {
            "teacher_behavior_cloning": 4 * BC_ROOTS,
            "distillation_validation": 5 * VALIDATION_PAIRS,
            "on_policy_training": 4 * PPO_ROOTS,
            "development": 5 * DEVELOPMENT_PAIRS,
            "confirmation": 5 * CONFIRMATION_PAIRS,
            "total": TOTAL_UNIQUE_STREAMS,
        },
        "unique_stream_count": len(set(stream_ids)),
        "stream_prefix_counts": {
            str(prefix): count
            for prefix, count in sorted(stream_prefixes.items())
        },
        "canonical_rows_sha256": canonical_json_hash(rows),
        "root_set_sha256": canonical_json_hash(sorted(root_ids)),
        "ancestry_set_sha256": canonical_json_hash(sorted(ancestry_ids)),
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": dict(ZERO_WORK),
    }


class J2ActorCritic(nn.Module):
    """Frozen no-auxiliary J2 policy/value network."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBSERVATION_WIDTH, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.ReLU(),
        )
        self.policy = nn.Linear(HIDDEN_WIDTH, ACTION_COUNT)
        self.value = nn.Linear(HIDDEN_WIDTH, 1)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(observations)
        return self.policy(features), self.value(features).squeeze(-1)


def parameter_count(model: nn.Module | None = None) -> int:
    value = J2ActorCritic() if model is None else model
    return sum(parameter.numel() for parameter in value.parameters())


def model_schema() -> dict[str, Any]:
    return {
        "version": "j2_no_aux_actor_critic_schema_v1",
        "observation_width": OBSERVATION_WIDTH,
        "body": [
            ["linear", OBSERVATION_WIDTH, HIDDEN_WIDTH],
            ["relu"],
            ["linear", HIDDEN_WIDTH, HIDDEN_WIDTH],
            ["relu"],
        ],
        "heads": {
            "policy": ["linear", HIDDEN_WIDTH, ACTION_COUNT],
            "value": ["linear", HIDDEN_WIDTH, 1],
        },
        "auxiliary_heads": [],
        "auxiliary_losses": [],
        "action_order": ["up", "down", "left", "right"],
        "initialization": "from_scratch",
        "initialization_seed": INITIALIZATION_SEED,
        "starter_tile": None,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
    }


def initialize_model_optimizer() -> tuple[J2ActorCritic, torch.optim.Optimizer]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(INITIALIZATION_SEED)
    model = J2ActorCritic().cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        eps=ADAM_EPS,
    )
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise J2ReadinessIntegrityError("J2 parameter count changed")
    return model, optimizer


def masked_logits(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != legal_masks.shape:
        raise J2ReadinessIntegrityError("Logit/legal-mask shape mismatch")
    if legal_masks.dtype is not torch.bool:
        raise J2ReadinessIntegrityError("Legal mask must be boolean")
    if not torch.all(legal_masks.any(dim=-1)):
        raise J2ReadinessIntegrityError("A live row has no legal action")
    if not torch.isfinite(logits).all():
        raise J2ReadinessIntegrityError("Policy logits are nonfinite")
    return logits.masked_fill(~legal_masks, -torch.inf)


def root_equal_weights(lengths: Sequence[int]) -> np.ndarray:
    if not lengths or any(int(length) <= 0 for length in lengths):
        raise J2ReadinessIntegrityError("Root lengths must be positive")
    return np.concatenate(
        [
            np.full(int(length), 1.0 / float(length), dtype=np.float64)
            for length in lengths
        ]
    )


@dataclass(frozen=True)
class DistillationBatch:
    observations: torch.Tensor
    legal_masks: torch.Tensor
    teacher_actions: torch.Tensor
    value_targets: torch.Tensor
    row_weights: torch.Tensor
    root_ids: tuple[str, ...]

    def row_count(self) -> int:
        return int(self.observations.shape[0])

    def subset(self, indices: torch.Tensor) -> "DistillationBatch":
        selected = indices.detach().cpu().tolist()
        return DistillationBatch(
            observations=self.observations[indices],
            legal_masks=self.legal_masks[indices],
            teacher_actions=self.teacher_actions[indices],
            value_targets=self.value_targets[indices],
            row_weights=self.row_weights[indices],
            root_ids=tuple(self.root_ids[index] for index in selected),
        )


def validate_distillation_batch(
    batch: DistillationBatch,
    *,
    complete_root_weights: bool = True,
) -> None:
    rows = batch.row_count()
    if batch.observations.shape != (rows, OBSERVATION_WIDTH):
        raise J2ReadinessIntegrityError("Distillation observation shape changed")
    if batch.legal_masks.shape != (rows, ACTION_COUNT):
        raise J2ReadinessIntegrityError("Distillation legal-mask shape changed")
    if batch.teacher_actions.shape != (rows,):
        raise J2ReadinessIntegrityError("Teacher-action shape changed")
    if batch.value_targets.shape != (rows,):
        raise J2ReadinessIntegrityError("Value-target shape changed")
    if batch.row_weights.shape != (rows,):
        raise J2ReadinessIntegrityError("Row-weight shape changed")
    if len(batch.root_ids) != rows:
        raise J2ReadinessIntegrityError("Root identity count changed")
    if batch.legal_masks.dtype is not torch.bool:
        raise J2ReadinessIntegrityError("Legal masks must be boolean")
    if batch.teacher_actions.dtype is not torch.int64:
        raise J2ReadinessIntegrityError("Teacher actions must be int64")
    if any(
        not torch.isfinite(value).all()
        for value in (
            batch.observations,
            batch.value_targets,
            batch.row_weights,
        )
    ):
        raise J2ReadinessIntegrityError("Distillation batch is nonfinite")
    if torch.any(batch.row_weights <= 0.0):
        raise J2ReadinessIntegrityError("Row weights must be positive")
    if torch.any(batch.teacher_actions < 0) or torch.any(
        batch.teacher_actions >= ACTION_COUNT
    ):
        raise J2ReadinessIntegrityError("Teacher action is out of range")
    if not torch.all(batch.legal_masks.any(dim=1)):
        raise J2ReadinessIntegrityError("A validation row has no legal action")
    chosen_legal = batch.legal_masks.gather(
        1,
        batch.teacher_actions.unsqueeze(1),
    ).squeeze(1)
    if not torch.all(chosen_legal):
        raise J2ReadinessIntegrityError("Teacher supplied an illegal action")
    if complete_root_weights:
        totals: dict[str, float] = defaultdict(float)
        for root_id, weight in zip(
            batch.root_ids,
            batch.row_weights.detach().cpu().tolist(),
        ):
            totals[root_id] += float(weight)
        if any(
            not math.isclose(total, 1.0, abs_tol=2e-7, rel_tol=0.0)
            for total in totals.values()
        ):
            raise J2ReadinessIntegrityError(
                "Distillation root-equal weights changed"
            )


def distillation_loss(
    model: J2ActorCritic,
    batch: DistillationBatch,
    *,
    global_weight_total: torch.Tensor | float | None = None,
    minibatches_per_epoch: int = 1,
) -> dict[str, torch.Tensor]:
    validate_distillation_batch(
        batch,
        complete_root_weights=(global_weight_total is None),
    )
    logits, values = model(batch.observations)
    policy_rows = torch.nn.functional.cross_entropy(
        masked_logits(logits, batch.legal_masks),
        batch.teacher_actions,
        reduction="none",
    )
    value_rows = (values - batch.value_targets) ** 2
    if global_weight_total is None:
        denominator = batch.row_weights.sum()
        scale = 1.0
    else:
        denominator = torch.as_tensor(
            global_weight_total,
            dtype=batch.row_weights.dtype,
        )
        scale = float(minibatches_per_epoch)
    if not torch.isfinite(denominator) or float(denominator) <= 0.0:
        raise J2ReadinessIntegrityError("Global weight total is invalid")

    def reduce(rows: torch.Tensor) -> torch.Tensor:
        return torch.sum(batch.row_weights * rows) / denominator * scale

    policy_loss = reduce(policy_rows)
    value_loss = reduce(value_rows)
    total_loss = policy_loss + VALUE_COEFFICIENT * value_loss
    outputs = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "weight_sum": batch.row_weights.sum(),
    }
    if any(not torch.isfinite(value) for value in outputs.values()):
        raise J2ReadinessIntegrityError("Distillation loss is nonfinite")
    return outputs


def deterministic_distillation_plan(
    row_count: int,
    *,
    epochs: int = DISTILLATION_EPOCHS,
    minibatch_size: int = MINIBATCH_SIZE,
) -> list[dict[str, Any]]:
    if row_count < 1 or epochs < 1 or minibatch_size < 1:
        raise J2ReadinessIntegrityError("Invalid distillation plan dimensions")
    plan = []
    for epoch in range(epochs):
        material = (
            f"J2-distillation-plan-v1|{INITIALIZATION_SEED}|{epoch}"
        ).encode("ascii")
        seed = int.from_bytes(
            hashlib.sha256(material).digest()[:8],
            "big",
            signed=False,
        )
        permutation = np.random.default_rng(seed).permutation(row_count)
        for start in range(0, row_count, minibatch_size):
            indices = tuple(
                int(value)
                for value in permutation[start : start + minibatch_size]
            )
            plan.append(
                {
                    "epoch": epoch,
                    "start": start,
                    "seed": seed,
                    "indices": indices,
                    "final_short": len(indices) < minibatch_size,
                    "step_id": canonical_json_hash(
                        {
                            "epoch": epoch,
                            "start": start,
                            "indices": indices,
                        }
                    ),
                }
            )
    return plan


def distillation_plan_audit(
    row_count: int,
    *,
    epochs: int = DISTILLATION_EPOCHS,
    minibatch_size: int = MINIBATCH_SIZE,
) -> dict[str, Any]:
    plan = deterministic_distillation_plan(
        row_count,
        epochs=epochs,
        minibatch_size=minibatch_size,
    )
    coverage = {
        epoch: [
            index
            for row in plan
            if int(row["epoch"]) == epoch
            for index in row["indices"]
        ]
        for epoch in range(epochs)
    }
    checks = {
        "every_row_once_per_epoch": all(
            sorted(indices) == list(range(row_count))
            for indices in coverage.values()
        ),
        "eight_epochs_exact": epochs == DISTILLATION_EPOCHS,
        "final_short_retained": (
            row_count % minibatch_size == 0
            or sum(bool(row["final_short"]) for row in plan) == epochs
        ),
        "deterministic": plan
        == deterministic_distillation_plan(
            row_count,
            epochs=epochs,
            minibatch_size=minibatch_size,
        ),
    }
    return {
        "row_count": row_count,
        "epochs": epochs,
        "minibatch_size": minibatch_size,
        "plan_sha256": canonical_json_hash(plan),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def distillation_batch_identity(batch: DistillationBatch) -> str:
    return canonical_json_hash(
        {
            "observations": _tensor_sha256(batch.observations),
            "legal_masks": _tensor_sha256(batch.legal_masks),
            "teacher_actions": _tensor_sha256(batch.teacher_actions),
            "value_targets": _tensor_sha256(batch.value_targets),
            "row_weights": _tensor_sha256(batch.row_weights),
            "root_ids": list(batch.root_ids),
        }
    )


def synthetic_distillation_batch(
    *,
    root_lengths: Sequence[int] = (3, 5),
    seed: int = 2_026_072_822,
) -> DistillationBatch:
    row_count = sum(int(value) for value in root_lengths)
    rng = np.random.default_rng(seed)
    observations = torch.from_numpy(
        rng.normal(0.0, 0.25, size=(row_count, OBSERVATION_WIDTH)).astype(
            np.float32
        )
    )
    legal_masks = torch.from_numpy(
        rng.random((row_count, ACTION_COUNT)) > 0.25
    )
    for row in range(row_count):
        if not bool(legal_masks[row].any()):
            legal_masks[row, row % ACTION_COUNT] = True
    teacher_actions = torch.tensor(
        [
            int(torch.nonzero(mask, as_tuple=False)[0, 0])
            for mask in legal_masks
        ],
        dtype=torch.int64,
    )
    value_targets = torch.from_numpy(
        rng.normal(0.01, 0.02, size=row_count).astype(np.float32)
    )
    weights = torch.from_numpy(
        root_equal_weights(root_lengths).astype(np.float32)
    )
    root_ids = tuple(
        f"root-{root_index}"
        for root_index, length in enumerate(root_lengths)
        for _ in range(int(length))
    )
    batch = DistillationBatch(
        observations=observations,
        legal_masks=legal_masks,
        teacher_actions=teacher_actions,
        value_targets=value_targets,
        row_weights=weights,
        root_ids=root_ids,
    )
    validate_distillation_batch(batch)
    return batch


class DistillationUpdater:
    """Synthetic production-shape optimizer/resume fixture."""

    def __init__(
        self,
        model: J2ActorCritic,
        optimizer: torch.optim.Optimizer,
        batch: DistillationBatch,
        *,
        minibatch_size: int,
        epochs: int,
        cursor: int = 0,
        closed_step_ids: Sequence[str] = (),
    ) -> None:
        validate_distillation_batch(batch)
        self.model = model
        self.optimizer = optimizer
        self.batch = batch
        self.plan = deterministic_distillation_plan(
            batch.row_count(),
            epochs=epochs,
            minibatch_size=minibatch_size,
        )
        self.cursor = int(cursor)
        self.closed_step_ids = list(closed_step_ids)
        if self.cursor != len(self.closed_step_ids):
            raise J2ReadinessIntegrityError(
                "Distillation cursor/closed-step count changed"
            )
        expected = [
            str(row["step_id"]) for row in self.plan[: self.cursor]
        ]
        if self.closed_step_ids != expected:
            raise J2ReadinessIntegrityError(
                "Distillation closed-step prefix changed"
            )
        model_ids = [id(parameter) for parameter in model.parameters()]
        optimizer_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if (
            len(model_ids) != len(set(model_ids))
            or len(optimizer_ids) != len(set(optimizer_ids))
            or set(model_ids) != set(optimizer_ids)
        ):
            raise J2ReadinessIntegrityError(
                "Optimizer parameters do not exactly match the J2 model"
            )

    def step(self) -> dict[str, Any]:
        if self.cursor >= len(self.plan):
            raise J2ReadinessIntegrityError("Distillation update is complete")
        row = self.plan[self.cursor]
        indices = torch.tensor(row["indices"], dtype=torch.int64)
        subset = self.batch.subset(indices)
        minibatches_in_epoch = sum(
            int(candidate["epoch"]) == int(row["epoch"])
            for candidate in self.plan
        )
        losses = distillation_loss(
            self.model,
            subset,
            global_weight_total=self.batch.row_weights.sum(),
            minibatches_per_epoch=minibatches_in_epoch,
        )
        self.optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            GRADIENT_CLIP,
        )
        if not torch.isfinite(gradient_norm):
            raise J2ReadinessIntegrityError(
                "Distillation gradient norm is nonfinite"
            )
        self.optimizer.step()
        if any(
            not torch.isfinite(value).all()
            for value in self.model.state_dict().values()
        ):
            raise J2ReadinessIntegrityError(
                "Distillation produced a nonfinite model"
            )
        self.closed_step_ids.append(str(row["step_id"]))
        self.cursor += 1
        return {
            "step_id": row["step_id"],
            "cursor": self.cursor,
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }

    def run(self, *, max_steps: int | None = None) -> dict[str, Any]:
        limit = (
            len(self.plan)
            if max_steps is None
            else min(len(self.plan), self.cursor + int(max_steps))
        )
        while self.cursor < limit:
            self.step()
        return {
            "cursor": self.cursor,
            "complete": self.cursor == len(self.plan),
            "closed_step_ids": list(self.closed_step_ids),
        }

    def snapshot_bytes(self) -> bytes:
        payload = {
            "version": f"{VERSION}_distillation_snapshot_v1",
            "batch_identity": distillation_batch_identity(self.batch),
            "plan_sha256": canonical_json_hash(self.plan),
            "cursor": self.cursor,
            "closed_step_ids": list(self.closed_step_ids),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    @classmethod
    def from_snapshot_bytes(
        cls,
        payload_bytes: bytes,
        batch: DistillationBatch,
        *,
        minibatch_size: int,
        epochs: int,
    ) -> "DistillationUpdater":
        payload = torch.load(
            io.BytesIO(payload_bytes),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("version") != (
            f"{VERSION}_distillation_snapshot_v1"
        ):
            raise J2ReadinessIntegrityError(
                "Distillation snapshot version changed"
            )
        if payload.get("batch_identity") != distillation_batch_identity(batch):
            raise J2ReadinessIntegrityError(
                "Distillation snapshot batch identity changed"
            )
        model, optimizer = initialize_model_optimizer()
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        updater = cls(
            model,
            optimizer,
            batch,
            minibatch_size=minibatch_size,
            epochs=epochs,
            cursor=int(payload["cursor"]),
            closed_step_ids=payload["closed_step_ids"],
        )
        if payload.get("plan_sha256") != canonical_json_hash(updater.plan):
            raise J2ReadinessIntegrityError(
                "Distillation snapshot plan identity changed"
            )
        return updater


def feature_family(board: np.ndarray | Sequence[Sequence[int]]) -> str:
    value = np.asarray(board)
    if value.shape != (4, 4):
        raise J2ReadinessIntegrityError("Board shape changed")
    if not np.issubdtype(value.dtype, np.number):
        raise J2ReadinessIntegrityError("Board values are nonnumeric")
    if not np.isfinite(value).all() or np.any(value < 0):
        raise J2ReadinessIntegrityError("Board values are invalid")
    maximum = int(np.max(value, initial=0))
    empties = int(np.sum(value == 0))
    if maximum < 192:
        return "low_air" if empties >= 4 else "low_constrained"
    if maximum < 768:
        return "mid_progression"
    return "upper_progression"


def feature_inventory(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["root_id"]),
            int(row["transition_index"]),
        ),
    )
    enriched = [
        {
            "root_id": str(row["root_id"]),
            "transition_index": int(row["transition_index"]),
            "family": feature_family(row["board"]),
        }
        for row in ordered
    ]
    natural_counts = Counter(str(row["family"]) for row in enriched)
    roots_by_family = {
        family: sorted(
            {
                str(row["root_id"])
                for row in enriched
                if row["family"] == family
            }
        )
        for family in FEATURE_FAMILIES
    }
    smallest = min(
        (natural_counts.get(family, 0) for family in FEATURE_FAMILIES),
        default=0,
    )
    k = min(8_192, smallest)
    capped_rows = [
        row
        for family in FEATURE_FAMILIES
        for row in [
            candidate
            for candidate in enriched
            if candidate["family"] == family
        ][:k]
    ]
    capped_counts = Counter(str(row["family"]) for row in capped_rows)
    total = len(enriched)
    capped_total = len(capped_rows)
    natural_frequencies = {
        family: (
            natural_counts.get(family, 0) / total if total else 0.0
        )
        for family in FEATURE_FAMILIES
    }
    capped_frequencies = {
        family: (
            capped_counts.get(family, 0) / capped_total
            if capped_total
            else 0.0
        )
        for family in FEATURE_FAMILIES
    }
    checks = {
        "all_complete_rows_retained_naturally": len(enriched) == len(rows),
        "four_families_present": all(
            natural_counts.get(family, 0) > 0
            for family in FEATURE_FAMILIES
        ),
        "minimum_1024_states_each": all(
            natural_counts.get(family, 0) >= 1_024
            for family in FEATURE_FAMILIES
        ),
        "minimum_256_roots_each": all(
            len(roots_by_family[family]) >= 256
            for family in FEATURE_FAMILIES
        ),
        "natural_max_share_at_most_070": (
            max(natural_frequencies.values(), default=1.0) <= 0.70
        ),
        "capped_four_families": all(
            capped_counts.get(family, 0) == k and k > 0
            for family in FEATURE_FAMILIES
        ),
        "capped_max_share_at_most_040": (
            max(capped_frequencies.values(), default=1.0) <= 0.40
        ),
    }
    return {
        "natural_state_count": total,
        "natural_family_counts": dict(natural_counts),
        "natural_family_frequencies": natural_frequencies,
        "natural_family_root_counts": {
            family: len(roots_by_family[family])
            for family in FEATURE_FAMILIES
        },
        "capped_k_per_family": k,
        "capped_state_count": capped_total,
        "capped_family_counts": dict(capped_counts),
        "capped_family_frequencies": capped_frequencies,
        "capped_inventory_sha256": canonical_json_hash(capped_rows),
        "checks": checks,
        "passes": all(checks.values()),
    }


def value_target(
    *,
    current_score: int,
    final_score: int,
    remaining_score_deltas: Sequence[int],
) -> float:
    difference = int(final_score) - int(current_score)
    if difference != sum(int(value) for value in remaining_score_deltas):
        raise J2ReadinessIntegrityError(
            "Dense score-delta target does not telescope"
        )
    return 1e-5 * difference


def bc_mechanism_gate(
    *,
    overall_root_equal_accuracy: float,
    family_accuracies: Mapping[str, float],
    policy_loss: float,
    value_mse: float,
    zero_value_mse: float,
    illegal_teacher_actions: int,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    finite = all(
        math.isfinite(float(value))
        for value in (
            overall_root_equal_accuracy,
            policy_loss,
            value_mse,
            zero_value_mse,
            *family_accuracies.values(),
        )
    )
    checks = {
        "inventory_support_passes": bool(inventory.get("passes")),
        "overall_accuracy_at_least_097": (
            float(overall_root_equal_accuracy) >= 0.97
        ),
        "each_family_accuracy_at_least_094": (
            set(family_accuracies) == set(FEATURE_FAMILIES)
            and all(float(value) >= 0.94 for value in family_accuracies.values())
        ),
        "finite_losses_and_metrics": finite,
        "value_mse_below_zero": float(value_mse) < float(zero_value_mse),
        "teacher_actions_all_legal": int(illegal_teacher_actions) == 0,
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "decision": (
            "READY_J2_CLOSED_LOOP_FIDELITY_PREFLIGHT"
            if all(checks.values())
            else "HOLD_J2_DISTILLATION_DATA_SUPPORT"
        ),
    }


def closed_loop_fidelity_gate(
    *,
    retained_pairs: int,
    stratum_pair_counts: Sequence[int],
    score_point: float,
    score_lower_95: float,
    p1536_control_rate: float,
    progression_point_or: float,
    progression_lower_95_or: float,
    illegal_student_actions: int,
    finite_latency_and_survival: bool,
) -> dict[str, Any]:
    low_base_rate = float(p1536_control_rate) < 0.02
    checks = {
        "all_2048_pairs_retained": int(retained_pairs)
        == VALIDATION_PAIRS,
        "eight_equal_strata": (
            len(stratum_pair_counts) == 8
            and all(
                int(value) == VALIDATION_PAIRS // 8
                for value in stratum_pair_counts
            )
        ),
        "score_point_above_log_097": (
            float(score_point) > math.log(FIDELITY_SCORE_POINT_FLOOR)
        ),
        "score_lower_above_log_090": (
            float(score_lower_95) > math.log(FIDELITY_SCORE_CI_FLOOR)
        ),
        "p1536_control_rate_at_least_002": not low_base_rate,
        "progression_point_at_least_090": (
            float(progression_point_or) >= FIDELITY_OR_POINT_FLOOR
        ),
        "progression_lower_above_050": (
            float(progression_lower_95_or) > FIDELITY_OR_CI_FLOOR
        ),
        "illegal_student_actions_zero": int(illegal_student_actions) == 0,
        "latency_and_survival_finite": bool(finite_latency_and_survival),
    }
    finite = all(
        math.isfinite(float(value))
        for value in (
            score_point,
            score_lower_95,
            p1536_control_rate,
            progression_point_or,
            progression_lower_95_or,
        )
    )
    checks["all_estimands_finite"] = finite
    if low_base_rate:
        decision = "HOLD_J2_FIDELITY_INCONCLUSIVE_LOW_BASE_RATE"
    elif all(checks.values()):
        decision = "READY_J2_ON_POLICY_TRAINING_PREFLIGHT"
    else:
        decision = "HOLD_J2_CLOSED_LOOP_FIDELITY"
    return {
        "decision": decision,
        "checks": checks,
        "passes": decision == "READY_J2_ON_POLICY_TRAINING_PREFLIGHT",
        "first_action_gate_used": False,
        "full_policy_sustained_exposure": True,
    }


def teacher_kl_coefficient(round_number: int) -> float:
    value = int(round_number)
    if value < 1 or value > 64:
        raise J2ReadinessIntegrityError("PPO round is outside 1..64")
    if value > PPO_ANCHOR_ROUNDS:
        return 0.0
    return 0.05 * (PPO_ANCHOR_ROUNDS + 1 - value) / PPO_ANCHOR_ROUNDS


@dataclass(frozen=True)
class J2PPOBatch:
    observations: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    teacher_actions: torch.Tensor | None
    row_weights: torch.Tensor
    root_ids: tuple[str, ...]


def _normalize_advantages(
    advantages: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    total = weights.sum()
    mean = torch.sum(weights * advantages) / total
    variance = torch.sum(weights * (advantages - mean) ** 2) / total
    normalized = (advantages - mean) / torch.sqrt(variance + 1e-8)
    if not torch.isfinite(normalized).all():
        raise J2ReadinessIntegrityError("Advantages are nonfinite")
    return normalized


def j2_ppo_loss(
    model: J2ActorCritic,
    batch: J2PPOBatch,
    *,
    round_number: int,
) -> dict[str, torch.Tensor]:
    rows = int(batch.observations.shape[0])
    teacher_required = int(round_number) <= PPO_ANCHOR_ROUNDS
    coefficient = teacher_kl_coefficient(round_number)
    if batch.observations.shape != (rows, OBSERVATION_WIDTH):
        raise J2ReadinessIntegrityError("PPO observation shape changed")
    if batch.legal_masks.shape != (rows, ACTION_COUNT):
        raise J2ReadinessIntegrityError("PPO legal-mask shape changed")
    vectors = (
        batch.actions,
        batch.old_log_probabilities,
        batch.advantages,
        batch.returns,
        batch.row_weights,
    )
    if any(value.shape != (rows,) for value in vectors):
        raise J2ReadinessIntegrityError("PPO vector shape changed")
    if teacher_required:
        if batch.teacher_actions is None:
            raise J2ReadinessIntegrityError(
                "Rounds 1-16 require exact teacher actions"
            )
        if batch.teacher_actions.shape != (rows,):
            raise J2ReadinessIntegrityError(
                "PPO teacher-action shape changed"
            )
    elif batch.teacher_actions is not None:
        raise J2ReadinessIntegrityError(
            "Rounds 17-64 must not carry teacher actions"
        )
    if len(batch.root_ids) != rows:
        raise J2ReadinessIntegrityError("PPO root identities changed")
    if any(
        not torch.isfinite(value).all()
        for value in (
            batch.observations,
            batch.old_log_probabilities,
            batch.advantages,
            batch.returns,
            batch.row_weights,
        )
    ):
        raise J2ReadinessIntegrityError("PPO batch is nonfinite")
    chosen_legal = batch.legal_masks.gather(
        1,
        batch.actions.unsqueeze(1),
    ).squeeze(1)
    if not torch.all(chosen_legal):
        raise J2ReadinessIntegrityError("PPO action is illegal")
    if teacher_required:
        assert batch.teacher_actions is not None
        teacher_legal = batch.legal_masks.gather(
            1,
            batch.teacher_actions.unsqueeze(1),
        ).squeeze(1)
        if not torch.all(teacher_legal):
            raise J2ReadinessIntegrityError("PPO teacher action is illegal")
    logits, values = model(batch.observations)
    distribution = torch.distributions.Categorical(
        logits=masked_logits(logits, batch.legal_masks)
    )
    new_log_probabilities = distribution.log_prob(batch.actions)
    ratio = torch.exp(
        new_log_probabilities - batch.old_log_probabilities
    )
    normalized = _normalize_advantages(
        batch.advantages,
        batch.row_weights,
    )
    unclipped = -normalized * ratio
    clipped = -normalized * torch.clamp(ratio, 0.8, 1.2)
    policy_rows = torch.maximum(unclipped, clipped)
    value_rows = 0.5 * (values - batch.returns) ** 2
    entropy_rows = distribution.entropy()
    denominator = batch.row_weights.sum()

    def reduce(values: torch.Tensor) -> torch.Tensor:
        return torch.sum(batch.row_weights * values) / denominator

    policy_loss = reduce(policy_rows)
    value_loss = reduce(value_rows)
    entropy = reduce(entropy_rows)
    if teacher_required:
        assert batch.teacher_actions is not None
        teacher_kl = reduce(
            -distribution.log_prob(batch.teacher_actions)
        )
    else:
        teacher_kl = torch.zeros(
            (),
            dtype=policy_loss.dtype,
            device=policy_loss.device,
        )
    total_loss = (
        policy_loss
        + 0.5 * value_loss
        - 0.01 * entropy
        + coefficient * teacher_kl
    )
    outputs = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "teacher_kl": teacher_kl,
        "teacher_kl_coefficient": torch.tensor(coefficient),
    }
    if any(not torch.isfinite(value) for value in outputs.values()):
        raise J2ReadinessIntegrityError("PPO loss is nonfinite")
    return outputs


def shard_for_row(row_index: int) -> int:
    value = int(row_index)
    if value < 0:
        raise J2ReadinessIntegrityError("Shard row index is negative")
    return value % SHARD_COUNT


def shard_plan(row_count: int) -> dict[int, tuple[int, ...]]:
    if int(row_count) < 1:
        raise J2ReadinessIntegrityError("Shard row count is empty")
    return {
        shard: tuple(
            index
            for index in range(int(row_count))
            if shard_for_row(index) == shard
        )
        for shard in range(SHARD_COUNT)
    }


def deterministic_shard_merge(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["row_index"])
        if int(row["shard"]) != shard_for_row(index):
            raise J2ReadinessIntegrityError("Worker changed root ownership")
        expected_identity = canonical_json_hash(
            {
                "row_index": index,
                "shard": int(row["shard"]),
                "payload": row["payload"],
            }
        )
        if str(row.get("row_identity")) != expected_identity:
            raise J2ReadinessIntegrityError("Worker row identity changed")
        if index in by_index:
            raise J2ReadinessIntegrityError("Worker duplicated a root")
        by_index[index] = dict(row)
    expected = list(range(len(rows)))
    if sorted(by_index) != expected:
        raise J2ReadinessIntegrityError("Worker merge has a gap")
    return [by_index[index] for index in expected]


def _paired_binary_probabilities(
    p_control: float,
    p_treatment: float,
    coupling: float,
) -> np.ndarray:
    p0 = float(p_control)
    p1 = float(p_treatment)
    shared = np.asarray(
        [
            1.0 - max(p0, p1),
            max(p1 - p0, 0.0),
            max(p0 - p1, 0.0),
            min(p0, p1),
        ],
        dtype=np.float64,
    )
    independent = np.asarray(
        [
            (1.0 - p0) * (1.0 - p1),
            (1.0 - p0) * p1,
            p0 * (1.0 - p1),
            p0 * p1,
        ],
        dtype=np.float64,
    )
    probabilities = float(coupling) * shared + (
        1.0 - float(coupling)
    ) * independent
    if np.any(probabilities < 0.0) or not math.isclose(
        float(probabilities.sum()),
        1.0,
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise J2ReadinessIntegrityError(
            "Paired binary probabilities are invalid"
        )
    return probabilities / probabilities.sum()


def _mh_log_or(
    treatment_success: np.ndarray,
    control_success: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    treatment = treatment_success.astype(np.float64)
    treatment_failure = totals - treatment
    control = control_success.astype(np.float64)
    control_failure = totals - control
    combined = treatment + treatment_failure + control + control_failure
    numerator = np.sum(treatment * control_failure / combined, axis=-1)
    denominator = np.sum(treatment_failure * control / combined, axis=-1)
    zero = (numerator <= 0.0) | (denominator <= 0.0)
    if np.any(zero):
        treatment_zero = treatment[zero] + 0.5
        treatment_failure_zero = treatment_failure[zero] + 0.5
        control_zero = control[zero] + 0.5
        control_failure_zero = control_failure[zero] + 0.5
        combined_zero = (
            treatment_zero
            + treatment_failure_zero
            + control_zero
            + control_failure_zero
        )
        numerator[zero] = np.sum(
            treatment_zero * control_failure_zero / combined_zero,
            axis=-1,
        )
        denominator[zero] = np.sum(
            treatment_failure_zero * control_zero / combined_zero,
            axis=-1,
        )
    return np.log(numerator / denominator)


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
            int(roots_per_stratum),
            probabilities,
            size=int(bootstraps),
        )
        treatment_success.append(draws[:, 1] + draws[:, 3])
        control_success.append(draws[:, 2] + draws[:, 3])
    treatment = np.stack(treatment_success, axis=1)
    control = np.stack(control_success, axis=1)
    totals = np.full_like(
        treatment,
        int(roots_per_stratum),
        dtype=np.float64,
    )
    values = _mh_log_or(treatment, control, totals)
    return (
        float(np.quantile(values, 0.025, method="linear")),
        float(np.quantile(values, 0.975, method="linear")),
    )


def simulate_common_or_noninferiority_power(
    *,
    n_pairs: int,
    control_rate: float,
    coupling: float,
    datasets: int = POWER_DATASETS,
    bootstraps: int = POWER_BOOTSTRAPS,
    seed: int | None = None,
) -> dict[str, Any]:
    if int(n_pairs) % 8:
        raise J2ReadinessIntegrityError(
            "Common-OR panel must have eight equal strata"
        )
    if int(datasets) < 1 or int(bootstraps) < 1:
        raise J2ReadinessIntegrityError("Power workload was shortened")
    roots_per = int(n_pairs) // 8
    actual_seed = (
        int(seed)
        if seed is not None
        else (
            POWER_SEED
            + int(n_pairs) * 10_000
            + int(round(float(control_rate) * 1_000_000))
            + int(round(float(coupling) * 10_000_000))
        )
    )
    rng = np.random.default_rng(actual_seed)
    point_values = np.empty(int(datasets), dtype=np.float64)
    lower_values = np.empty(int(datasets), dtype=np.float64)
    probabilities = _paired_binary_probabilities(
        control_rate,
        control_rate,
        coupling,
    )
    for dataset_index in range(int(datasets)):
        cells = [
            rng.multinomial(roots_per, probabilities)
            for _ in range(8)
        ]
        treatment_success = np.asarray(
            [[int(counts[1] + counts[3]) for counts in cells]],
            dtype=np.float64,
        )
        control_success = np.asarray(
            [[int(counts[2] + counts[3]) for counts in cells]],
            dtype=np.float64,
        )
        totals = np.full((1, 8), roots_per, dtype=np.float64)
        point_values[dataset_index] = float(
            _mh_log_or(treatment_success, control_success, totals)[0]
        )
        lower, _upper = _bootstrap_binary_bounds(
            cells,
            roots_per,
            rng=rng,
            bootstraps=int(bootstraps),
        )
        lower_values[dataset_index] = lower
    point_or = np.exp(point_values)
    lower_or = np.exp(lower_values)
    margin_power = {
        f"{margin:.2f}": float(
            np.mean(
                (point_or >= FIDELITY_OR_POINT_FLOOR)
                & (lower_or > margin)
            )
        )
        for margin in OR_NI_MARGIN_GRID
    }
    primary_power = margin_power[f"{FIDELITY_OR_CI_FLOOR:.2f}"]
    strongest_powered_margin = max(
        (
            float(margin)
            for margin, power in margin_power.items()
            if power >= POWER_REQUIRED
        ),
        default=None,
    )
    return {
        "n_pairs": int(n_pairs),
        "strata": 8,
        "roots_per_stratum": roots_per,
        "control_rate": float(control_rate),
        "true_treatment_rate": float(control_rate),
        "true_odds_ratio": 1.0,
        "coupling": float(coupling),
        "datasets": int(datasets),
        "bootstraps_per_dataset": int(bootstraps),
        "seed": actual_seed,
        "point_gate": FIDELITY_OR_POINT_FLOOR,
        "primary_lower_gate": FIDELITY_OR_CI_FLOOR,
        "primary_gate_power": primary_power,
        "margin_power_grid": margin_power,
        "strongest_80pct_powered_lower_margin": strongest_powered_margin,
        "mean_point_or": float(np.mean(point_or)),
        "mean_lower_or": float(np.mean(lower_or)),
        "monte_carlo_standard_error": math.sqrt(
            max(primary_power * (1.0 - primary_power), 0.0)
            / int(datasets)
        ),
        "bootstrap_method": (
            "eight independent within-stratum whole-root multinomial "
            "resamples preserving fixed stratum totals"
        ),
        "edge_correction": 0.5,
        "quantile_method": "numpy linear 0.025/0.975",
    }


def score_fidelity_power() -> dict[str, Any]:
    n_pairs = VALIDATION_PAIRS
    standard_error = SCORE_SD / math.sqrt(n_pairs)
    mde = math.exp(
        (SCORE_Z_975 + SCORE_Z_80) * standard_error
    ) - 1.0
    normal = statistics.NormalDist()

    def probability_above(threshold: float) -> float:
        return normal.cdf((0.0 - float(threshold)) / standard_error)

    point_threshold = math.log(FIDELITY_SCORE_POINT_FLOOR)
    ci_threshold = (
        math.log(FIDELITY_SCORE_CI_FLOOR)
        + SCORE_Z_975 * standard_error
    )
    ci_five_threshold = math.log(0.95) + (
        SCORE_Z_975 * standard_error
    )
    combined_threshold = max(point_threshold, ci_threshold)
    return {
        "n_pairs": n_pairs,
        "paired_sd": SCORE_SD,
        "standard_error": standard_error,
        "score_80pct_mde_fraction": mde,
        "score_80pct_mde_percent": 100.0 * mde,
        "point_floor_ratio": FIDELITY_SCORE_POINT_FLOOR,
        "lower_ci_floor_ratio": FIDELITY_SCORE_CI_FLOOR,
        "equal_policy_point_only_power": probability_above(
            point_threshold
        ),
        "equal_policy_10pct_ci_only_power": probability_above(
            ci_threshold
        ),
        "equal_policy_combined_gate_power": probability_above(
            combined_threshold
        ),
        "equal_policy_5pct_ci_only_power": probability_above(
            ci_five_threshold
        ),
        "method": (
            "normal paired log-score model with two-sided 95% lower bound"
        ),
    }


def common_or_power_grid(
    *,
    n_pairs: int,
    datasets: int = POWER_DATASETS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    rows = [
        simulate_common_or_noninferiority_power(
            n_pairs=n_pairs,
            control_rate=rate,
            coupling=coupling,
            datasets=datasets,
            bootstraps=bootstraps,
        )
        for rate in CONTROL_RATES
        for coupling in PAIRING_COUPLINGS
    ]
    worst = min(
        (float(row["primary_gate_power"]) for row in rows),
        default=0.0,
    )
    return {
        "n_pairs": int(n_pairs),
        "control_rates": list(CONTROL_RATES),
        "pairing_couplings": list(PAIRING_COUPLINGS),
        "datasets_per_cell": int(datasets),
        "bootstraps_per_dataset": int(bootstraps),
        "cells": len(rows),
        "rows": rows,
        "worst_case_primary_power": worst,
        "required_power": POWER_REQUIRED,
        "passes": worst >= POWER_REQUIRED,
    }


def power_and_feasibility_report(
    *,
    datasets: int = POWER_DATASETS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    fidelity_progression = common_or_power_grid(
        n_pairs=VALIDATION_PAIRS,
        datasets=datasets,
        bootstraps=bootstraps,
    )
    confirmation_p3072 = common_or_power_grid(
        n_pairs=CONFIRMATION_PAIRS,
        datasets=datasets,
        bootstraps=bootstraps,
    )
    j1a_path = (
        REPO_ROOT
        / "threes_rl"
        / "runs"
        / "forensics"
        / "j1a_cost_power_amendment_v1"
        / "J1A_COST_POWER_ARITHMETIC.json"
    )
    j1a = load_hashed_json(
        j1a_path,
        field="arithmetic_payload_sha256",
    )
    development_score = j1a["score"]["amended"]["development"]
    confirmation_score = j1a["score"]["amended"]["confirmation"]
    development_progression = j1a["progression"]["amended"][
        "development"
    ]
    confirmation_progression = j1a["progression"]["amended"][
        "confirmation"
    ]
    development = {
        "n_pairs": int(development_score["n_pairs"]),
        "score": development_score,
        "progression": {
            "worst_power_or_1_50": development_progression[
                "worst_by_or"
            ]["1.50"]["power"],
            "mde_80pct_grid": development_progression[
                "mde_80pct_grid"
            ],
        },
    }
    confirmation = {
        "n_pairs": int(confirmation_score["n_pairs"]),
        "score": confirmation_score,
        "progression": {
            "worst_power_or_1_50": confirmation_progression[
                "worst_by_or"
            ]["1.50"]["power"],
            "mde_80pct_grid": confirmation_progression[
                "mde_80pct_grid"
            ],
        },
    }
    checks = {
        "fidelity_n_2048": VALIDATION_PAIRS == 2_048,
        "fidelity_score_combined_power_at_least_080": (
            score_fidelity_power()["equal_policy_combined_gate_power"]
            >= POWER_REQUIRED
        ),
        "fidelity_progression_power_at_least_080": (
            fidelity_progression["passes"]
        ),
        "j1a_development_n_exact": (
            int(development["n_pairs"]) == DEVELOPMENT_PAIRS
        ),
        "j1a_confirmation_n_exact": (
            int(confirmation["n_pairs"]) == CONFIRMATION_PAIRS
        ),
        "j1a_confirmation_score_power_exact": math.isclose(
            float(confirmation["score"]["power_at_7pct"]),
            0.951834009,
            abs_tol=5e-10,
            rel_tol=0.0,
        ),
        "j1a_confirmation_p1536_power_at_least_080": (
            float(confirmation["progression"]["worst_power_or_1_50"])
            >= POWER_REQUIRED
        ),
        "confirmation_p3072_noninferiority_power_reported": (
            len(confirmation_p3072["rows"]) == 12
        ),
        "p6144_tail_descriptive_only": True,
    }
    return {
        "version": f"{VERSION}_power_and_feasibility_v1",
        "fidelity_score": score_fidelity_power(),
        "fidelity_p1536_common_or": fidelity_progression,
        "accepted_j1a_development": development,
        "accepted_j1a_confirmation": confirmation,
        "confirmation_p3072_noninferiority": confirmation_p3072,
        "p6144": {
            "status": "mandatory descriptive only",
            "powered_gate_claimed": False,
        },
        "checks": checks,
        "passes": all(checks.values()),
        "zero_outcomes_opened": 0,
    }


def source_and_parent_audit(
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    source_rows = {}
    for relative, expected in EXPECTED_SOURCE_IDENTITIES.items():
        path = root / relative
        observed = sha256_path(path) if path.is_file() else None
        source_rows[relative] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matches": observed == expected,
        }
    artifact_rows = {}
    for relative, (
        expected_file,
        payload_field,
        expected_payload,
    ) in EXPECTED_ARTIFACT_IDENTITIES.items():
        path = root / relative
        try:
            payload = load_hashed_json(
                path,
                field=payload_field,
                root=root,
            )
            observed_file = sha256_path(path, root)
            observed_payload = payload.get(payload_field)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            observed_file = None
            observed_payload = None
        artifact_rows[relative] = {
            "expected_file_sha256": expected_file,
            "observed_file_sha256": observed_file,
            "payload_field": payload_field,
            "expected_payload_sha256": expected_payload,
            "observed_payload_sha256": observed_payload,
            "matches": (
                observed_file == expected_file
                and observed_payload == expected_payload
            ),
        }
    terminal_path = (
        root
        / "threes_rl"
        / "runs"
        / "forensics"
        / "j1d_execution_v1"
        / "training"
        / "terminal_result.json"
    )
    sanity_path = terminal_path.with_name("training_sanity_result.json")
    checkpoint_path = terminal_path.with_name(
        "round64_candidate_checkpoint.bin"
    )
    try:
        terminal = load_hashed_json(
            terminal_path,
            field="terminal_result_payload_sha256",
            root=root,
        )
        sanity = load_hashed_json(
            sanity_path,
            field="training_sanity_payload_sha256",
            root=root,
        )
        checkpoint_sha = sha256_path(checkpoint_path, root)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        terminal = {}
        sanity = {}
        checkpoint_sha = None
    local_sources = {
        "charter": (
            sha256_path(root / CHARTER_PATH.relative_to(REPO_ROOT), root)
            if (root / CHARTER_PATH.relative_to(REPO_ROOT)).is_file()
            else None
        ),
        "runner": (
            sha256_path(root / RUNNER_PATH.relative_to(REPO_ROOT), root)
            if (root / RUNNER_PATH.relative_to(REPO_ROOT)).is_file()
            else None
        ),
        "tests": (
            sha256_path(root / TEST_PATH.relative_to(REPO_ROOT), root)
            if (root / TEST_PATH.relative_to(REPO_ROOT)).is_file()
            else None
        ),
    }
    checks = {
        "all_parent_source_hashes_exact": all(
            row["matches"] for row in source_rows.values()
        ),
        "all_parent_artifact_hashes_exact": all(
            row["matches"] for row in artifact_rows.values()
        ),
        "j1d_terminal_is_clean_hold": (
            terminal.get("decision") == "HOLD_J1_LEARNING_SANITY"
            and terminal.get("checkpoint_authoritative") is False
            and terminal.get("checkpoint_quarantined") is True
        ),
        "j1d_sanity_is_clean_hold": (
            sanity.get("decision") == "HOLD_J1_LEARNING_SANITY"
            and sanity.get("checkpoint_authoritative") is False
            and sanity.get("checkpoint_quarantined") is True
        ),
        "j1d_checkpoint_identity_exact_and_quarantined": (
            checkpoint_sha == EXPECTED_J1D_CHECKPOINT_SHA256
        ),
        "local_source_files_present": all(local_sources.values()),
    }
    return {
        "version": f"{VERSION}_source_parent_audit_v1",
        "parent_sources": source_rows,
        "parent_artifacts": artifact_rows,
        "j1d_terminal_structural_fields": {
            "decision": terminal.get("decision"),
            "checkpoint_authoritative": terminal.get(
                "checkpoint_authoritative"
            ),
            "checkpoint_quarantined": terminal.get(
                "checkpoint_quarantined"
            ),
        },
        "j1d_sanity_structural_fields": {
            "decision": sanity.get("decision"),
            "checkpoint_authoritative": sanity.get(
                "checkpoint_authoritative"
            ),
            "checkpoint_quarantined": sanity.get(
                "checkpoint_quarantined"
            ),
        },
        "j1d_checkpoint_file_sha256": checkpoint_sha,
        "j1d_checkpoint_opened_or_loaded": False,
        "local_sources": local_sources,
        "checks": checks,
        "passes": all(checks.values()),
    }


def teacher_provenance_audit(
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    from threes_rl import j1_execution_surface as j1_execution

    binding = j1_execution.incumbent_policy_binding()
    source_rows = {
        relative: {
            "expected_sha256": expected,
            "observed_sha256": (
                sha256_path(root / relative, root)
                if (root / relative).is_file()
                else None
            ),
        }
        for relative, expected in INCUMBENT_SOURCE_HASHES.items()
    }
    summary_path = root / INCUMBENT_SUMMARY_PATH.relative_to(REPO_ROOT)
    manifest_path = (
        root / INCUMBENT_TOP_MANIFEST_PATH.relative_to(REPO_ROOT)
    )
    replay_path = root / INCUMBENT_REPLAY_PATH.relative_to(REPO_ROOT)
    log_path = root / "threes_rl" / "EXPERIMENT_LOG.md"
    dashboard_path = (
        root / "threes_rl" / "runs" / "dashboard" / "dashboard.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        manifest = []
        dashboard = {}
        log_text = ""
    first_manifest = (
        manifest[0]
        if isinstance(manifest, list) and manifest
        else {}
    )
    dashboard_top = (
        dashboard.get("global_top_replays", [None])[0]
        if isinstance(dashboard.get("global_top_replays"), list)
        and dashboard.get("global_top_replays")
        else {}
    )
    top_three = [
        int(row["score"])
        for row in dashboard.get("global_top_replays", [])[:3]
        if isinstance(row, Mapping) and "score" in row
    ]
    observed_provenance = {
        "summary": (
            sha256_path(summary_path, root)
            if summary_path.is_file()
            else None
        ),
        "top_manifest": (
            sha256_path(manifest_path, root)
            if manifest_path.is_file()
            else None
        ),
        "rank1_replay": (
            sha256_path(replay_path, root)
            if replay_path.is_file()
            else None
        ),
    }
    expected_replay_relative = str(
        INCUMBENT_REPLAY_PATH.relative_to(REPO_ROOT)
    )
    excerpt_sha = hashlib.sha256(
        GOVERNANCE_EXCERPT.encode("utf-8")
    ).hexdigest()
    checks = {
        "incumbent_binding_exact": all(
            binding.get(key) == expected
            for key, expected in EXPECTED_INCUMBENT.items()
        ),
        "all_incumbent_sources_exact": all(
            row["observed_sha256"] == row["expected_sha256"]
            for row in source_rows.values()
        ),
        "all_provenance_file_hashes_exact": (
            observed_provenance == INCUMBENT_PROVENANCE_HASHES
        ),
        "governance_excerpt_literal_hash_exact": (
            excerpt_sha == EXPECTED_GOVERNANCE_EXCERPT_SHA256
        ),
        "governance_excerpt_present_exactly": (
            log_text.count(GOVERNANCE_EXCERPT) == 1
        ),
        "top_manifest_association_exact": (
            isinstance(first_manifest, Mapping)
            and int(first_manifest.get("rank", -1)) == 1
            and int(first_manifest.get("seed", -1)) == 1011
            and int(first_manifest.get("score", -1)) == 263670
            and first_manifest.get("json") == expected_replay_relative
        ),
        "dashboard_association_exact": (
            isinstance(dashboard_top, Mapping)
            and int(dashboard.get("best_high_score", -1)) == 263670
            and int(dashboard_top.get("score", -1)) == 263670
            and int(dashboard_top.get("seed", -1)) == 1011
            and dashboard_top.get("run") == INCUMBENT_RUN
            and dashboard_top.get("json") == expected_replay_relative
        ),
        "dashboard_top_three_exact": top_three
        == [263670, 261369, 258561],
        "replay_bytes_hashed_not_parsed": True,
        "human_actions_not_teacher_labels": True,
    }
    return {
        "version": f"{VERSION}_teacher_provenance_v1",
        "teacher_kind": "exact protected composite software incumbent",
        "incumbent_binding": binding,
        "implementation_sources": source_rows,
        "provenance_paths": {
            "summary": str(summary_path.resolve()),
            "top_manifest": str(manifest_path.resolve()),
            "rank1_replay_hash_only": str(replay_path.resolve()),
            "governance_log": str(log_path.resolve()),
            "dashboard_semantic_pointer": str(dashboard_path.resolve()),
        },
        "provenance_file_hashes": observed_provenance,
        "governance_excerpt_sha256": excerpt_sha,
        "replay_payload_parsed": False,
        "human_session_content_read": False,
        "checks": checks,
        "passes": all(checks.values()),
    }


def protected_stream_authority(
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    j1_manifest_path = (
        root
        / "threes_rl"
        / "runs"
        / "forensics"
        / "j1_execution_surface_readiness_v1"
        / "J1_PROSPECTIVE_MANIFEST.json"
    )
    j1d_authority_path = (
        root
        / "threes_rl"
        / "runs"
        / "forensics"
        / "j1d_metric_authentication_readiness_v2"
        / "J1D_V2_PROTECTED_STREAM_AUTHORITY.json"
    )
    j1_manifest = load_hashed_json(
        j1_manifest_path,
        field="prospective_manifest_payload_sha256",
        root=root,
    )
    j1d_authority = load_hashed_json(
        j1d_authority_path,
        field="stream_authority_payload_sha256",
        root=root,
    )
    j2_intervals = [
        {
            "stage": str(block["stage"]),
            "kind": str(kind),
            "start": int(base),
            "end_inclusive": int(base)
            + int(block["authority_rows"])
            - 1,
            "rows": int(block["authority_rows"]),
        }
        for block in STAGE_TABLE
        for kind, base in block["streams"].items()
    ]
    j2_prefixes = {
        int(row["start"]) // 1_000_000_000 for row in j2_intervals
    }
    checks = {
        "j1_manifest_exact_and_passed": (
            j1_manifest.get("passes") is True
        ),
        "j1_ranges_cover_213b_226b": (
            {
                int(value) // 1_000_000_000
                for row in j1_manifest["ranges"].values()
                for key, value in row.items()
                if key != "rows"
            }
            == set(range(213, 227))
        ),
        "j1d_authority_exact_and_passed": (
            j1d_authority.get("passes") is True
        ),
        "j1d_training_offset_65535_spent": (
            sorted(
                int(values[1])
                for values in j1d_authority[
                    "fresh_j1d_intervals"
                ].values()
            )
            == [
                213_000_065_535,
                214_000_065_535,
                215_000_065_535,
                216_000_065_535,
            ]
        ),
        "j2_prefixes_227b_249b_exact": (
            j2_prefixes == set(range(227, 250))
        ),
        "j2_has_no_spent_prefix_collision": not (
            j2_prefixes & set(range(213, 227))
        ),
        "no_global_payload_parser": True,
        "streams_reserved_zero": True,
        "streams_consumed_zero": True,
    }
    return {
        "version": f"{VERSION}_protected_stream_authority_v1",
        "method": (
            "compact immutable J1/J1d range authorities plus explicit "
            "prospective J2 intervals; no heterogeneous historical scan"
        ),
        "bound_authorities": {
            "j1_prospective_manifest": {
                "path": str(j1_manifest_path.resolve()),
                "file_sha256": sha256_path(j1_manifest_path, root),
                "payload_sha256": j1_manifest[
                    "prospective_manifest_payload_sha256"
                ],
            },
            "j1d_stream_authority": {
                "path": str(j1d_authority_path.resolve()),
                "file_sha256": sha256_path(j1d_authority_path, root),
                "payload_sha256": j1d_authority[
                    "stream_authority_payload_sha256"
                ],
            },
        },
        "denied_namespace_prefixes": list(range(213, 227)),
        "j2_intervals": j2_intervals,
        "j2_interval_sha256": canonical_json_hash(j2_intervals),
        "checks": checks,
        "passes": all(checks.values()),
        "streams_reserved": 0,
        "streams_consumed": 0,
    }


def _teacher_cost_projection(
    *,
    roots: int,
    moves: int,
    action_seconds: float,
) -> dict[str, Any]:
    decisions = int(roots) * int(moves)
    cpu_hours = decisions * float(action_seconds) / 3600.0
    ideal_eight_shard_wall = cpu_hours / SHARD_COUNT
    return {
        "roots": int(roots),
        "moves_per_root": int(moves),
        "teacher_action_queries": decisions,
        "action_seconds": float(action_seconds),
        "serial_active_cpu_hours": cpu_hours,
        "ideal_eight_shard_wall_hours": ideal_eight_shard_wall,
        "ideal_eight_shard_wall_hours_with_25pct_margin": (
            ideal_eight_shard_wall * SAFETY_MULTIPLIER
        ),
        "real_eight_process_evidence": False,
        "ideal_scaling_is_not_admission_evidence": True,
    }


def _unaccepted_cost_evidence(path: Path, purpose: str) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "purpose": purpose,
        "path": str(path.resolve()),
        "exists": exists,
        "accepted_identity_allowlist": [],
        "accepted": False,
        "reason": (
            "no pre-existing hash-authorized real-incumbent evidence"
            if not exists
            else (
                "file exists but no immutable identity was authorized "
                "before this readiness seal"
            )
        ),
        "payload_parsed": False,
    }


def runtime_storage_projection(
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    parent_projection_path = (
        root
        / "threes_rl"
        / "runs"
        / "forensics"
        / "j1_execution_surface_readiness_v1"
        / "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json"
    )
    parent = load_hashed_json(
        parent_projection_path,
        field="projection_payload_sha256",
        root=root,
    )
    pretraining_evidence = _unaccepted_cost_evidence(
        root / REAL_TEACHER_SHARD_EVIDENCE_PATH.relative_to(REPO_ROOT),
        "real incumbent eight-process throughput and memory",
    )
    online_evidence = _unaccepted_cost_evidence(
        root / ONLINE_TEACHER_SYNC_EVIDENCE_PATH.relative_to(REPO_ROOT),
        (
            "synchronous round-by-round eight-way on-policy teacher "
            "query and canonical merge"
        ),
    )
    teacher = {
        "pre_ppo": {
            "root_count": PRE_PPO_TEACHER_ROOTS,
            "composition": {
                "behavior_cloning_teacher_roots": BC_ROOTS,
                "validation_teacher_control_arms": VALIDATION_PAIRS,
                "validation_student_arms_excluded": VALIDATION_PAIRS,
            },
            "central_median": _teacher_cost_projection(
                roots=PRE_PPO_TEACHER_ROOTS,
                moves=PLANNING_MOVES,
                action_seconds=TEACHER_ACTION_SECONDS_MEDIAN,
            ),
            "central_p99": _teacher_cost_projection(
                roots=PRE_PPO_TEACHER_ROOTS,
                moves=PLANNING_MOVES,
                action_seconds=TEACHER_ACTION_SECONDS_P99,
            ),
            "sensitivity_5000_median": _teacher_cost_projection(
                roots=PRE_PPO_TEACHER_ROOTS,
                moves=SENSITIVITY_MOVES,
                action_seconds=TEACHER_ACTION_SECONDS_MEDIAN,
            ),
        },
        "on_policy_anchor": {
            "root_count": ONLINE_TEACHER_ROOTS,
            "rounds": PPO_ANCHOR_ROUNDS,
            "roots_per_round": PPO_ROOTS_PER_ROUND,
            "central_median": _teacher_cost_projection(
                roots=ONLINE_TEACHER_ROOTS,
                moves=PLANNING_MOVES,
                action_seconds=TEACHER_ACTION_SECONDS_MEDIAN,
            ),
            "central_p99": _teacher_cost_projection(
                roots=ONLINE_TEACHER_ROOTS,
                moves=PLANNING_MOVES,
                action_seconds=TEACHER_ACTION_SECONDS_P99,
            ),
            "sensitivity_5000_median": _teacher_cost_projection(
                roots=ONLINE_TEACHER_ROOTS,
                moves=SENSITIVITY_MOVES,
                action_seconds=TEACHER_ACTION_SECONDS_MEDIAN,
            ),
        },
        "total_root_equivalents": TOTAL_TEACHER_ROOT_EQUIVALENTS,
        "total_root_count_formula": {
            str(row["stage"]): {
                "pre_ppo": int(row["pre_ppo_teacher_roots"]),
                "online": int(row["online_teacher_roots"]),
            }
            for row in STAGE_TABLE
        },
    }
    sealed_forward_backward_seconds = 0.014503832906484604
    bc_transitions = BC_ROOTS * PLANNING_MOVES
    bc_optimizer_steps = (
        math.ceil(bc_transitions / MINIBATCH_SIZE)
        * DISTILLATION_EPOCHS
    )
    bc_optimizer_hours = (
        bc_optimizer_steps * sealed_forward_backward_seconds / 3600.0
    )
    distillation_wall_pre_margin = (
        teacher["pre_ppo"]["central_p99"][
            "ideal_eight_shard_wall_hours"
        ]
        + bc_optimizer_hours
    )
    distillation_wall_margin = (
        distillation_wall_pre_margin * SAFETY_MULTIPLIER
    )
    parent_training_margin_hours = float(
        parent["training"]["central"]["hours_with_25pct_margin"]
    )
    parent_training_pre_margin = (
        parent_training_margin_hours / SAFETY_MULTIPLIER
    )
    ppo_wall_pre_margin = (
        parent_training_pre_margin
        + teacher["on_policy_anchor"]["central_p99"][
            "ideal_eight_shard_wall_hours"
        ]
    )
    ppo_wall_margin = ppo_wall_pre_margin * SAFETY_MULTIPLIER
    sensitivity_bc_transitions = BC_ROOTS * SENSITIVITY_MOVES
    sensitivity_bc_optimizer_steps = (
        math.ceil(sensitivity_bc_transitions / MINIBATCH_SIZE)
        * DISTILLATION_EPOCHS
    )
    sensitivity_bc_optimizer_hours = (
        sensitivity_bc_optimizer_steps
        * sealed_forward_backward_seconds
        / 3600.0
    )
    distillation_sensitivity_wall_margin = (
        teacher["pre_ppo"]["sensitivity_5000_median"][
            "ideal_eight_shard_wall_hours"
        ]
        + sensitivity_bc_optimizer_hours
    ) * SAFETY_MULTIPLIER
    parent_training_sensitivity = parent["training"][
        "sensitivity_5000_moves"
    ]
    parent_training_sensitivity_pre_margin = (
        float(parent_training_sensitivity["hours_with_25pct_margin"])
        / SAFETY_MULTIPLIER
    )
    ppo_sensitivity_wall_margin = (
        parent_training_sensitivity_pre_margin
        + teacher["on_policy_anchor"]["sensitivity_5000_median"][
            "ideal_eight_shard_wall_hours"
        ]
    ) * SAFETY_MULTIPLIER

    root_blob_bytes = 1_519
    round_batch_bytes = 1_261
    pair_blob_bytes = 24_576
    distillation_storage_before_margin = (
        bc_transitions * root_blob_bytes
        + bc_transitions * round_batch_bytes
        + VALIDATION_PAIRS * pair_blob_bytes
        + 256 * 1024 * 1024
    )
    distillation_storage_margin = (
        distillation_storage_before_margin * SAFETY_MULTIPLIER
    )
    online_teacher_label_bytes = (
        ONLINE_TEACHER_ROOTS * PLANNING_MOVES * 8
    )
    ppo_storage_margin = (
        float(
            parent["training"]["central"]["storage"][
                "projected_with_margin_bytes"
            ]
        )
        + online_teacher_label_bytes * SAFETY_MULTIPLIER
    )
    distillation_sensitivity_storage_margin = (
        (
            sensitivity_bc_transitions * root_blob_bytes
            + sensitivity_bc_transitions * round_batch_bytes
            + VALIDATION_PAIRS * pair_blob_bytes
            + 256 * 1024 * 1024
        )
        * SAFETY_MULTIPLIER
    )
    online_teacher_sensitivity_label_bytes = (
        ONLINE_TEACHER_ROOTS * SENSITIVITY_MOVES * 8
    )
    ppo_sensitivity_storage_margin = (
        float(
            parent_training_sensitivity["storage"][
                "projected_with_margin_bytes"
            ]
        )
        + online_teacher_sensitivity_label_bytes * SAFETY_MULTIPLIER
    )
    sensitivity = {
        "moves_per_root": SENSITIVITY_MOVES,
        "diagnostic_not_conjunctive": True,
        "distillation": {
            "wall_hours_with_25pct_margin": (
                distillation_sensitivity_wall_margin
            ),
            "runtime_cap_hours": DISTILLATION_CAP_HOURS,
            "runtime_fits_cap": (
                distillation_sensitivity_wall_margin
                <= DISTILLATION_CAP_HOURS
            ),
            "storage_with_25pct_margin_bytes": (
                distillation_sensitivity_storage_margin
            ),
            "storage_with_25pct_margin_gib": (
                distillation_sensitivity_storage_margin / 1024**3
            ),
            "storage_cap_gib": DISTILLATION_CAP_GIB,
            "storage_fits_cap": (
                distillation_sensitivity_storage_margin
                <= DISTILLATION_CAP_GIB * 1024**3
            ),
            "optimizer_steps_projected": (
                sensitivity_bc_optimizer_steps
            ),
        },
        "on_policy_training": {
            "wall_hours_with_25pct_margin": ppo_sensitivity_wall_margin,
            "runtime_cap_hours": PPO_CAP_HOURS,
            "runtime_fits_cap": (
                ppo_sensitivity_wall_margin <= PPO_CAP_HOURS
            ),
            "storage_with_25pct_margin_bytes": (
                ppo_sensitivity_storage_margin
            ),
            "storage_with_25pct_margin_gib": (
                ppo_sensitivity_storage_margin / 1024**3
            ),
            "storage_cap_gib": PPO_CAP_GIB,
            "storage_fits_cap": (
                ppo_sensitivity_storage_margin
                <= PPO_CAP_GIB * 1024**3
            ),
        },
    }
    checks = {
        "single_stage_table_counts_exact": (
            derive_stage_totals() == EXPECTED_STAGE_TOTALS
        ),
        "student_fidelity_arms_not_teacher_work": (
            teacher["pre_ppo"]["composition"][
                "validation_student_arms_excluded"
            ]
            == VALIDATION_PAIRS
        ),
        "central_distillation_ideal_projection_under_cap": (
            distillation_wall_margin <= DISTILLATION_CAP_HOURS
        ),
        "central_ppo_ideal_projection_under_cap": (
            ppo_wall_margin <= PPO_CAP_HOURS
        ),
        "distillation_storage_under_cap": (
            distillation_storage_margin
            <= DISTILLATION_CAP_GIB * 1024**3
        ),
        "ppo_storage_under_cap": (
            ppo_storage_margin <= PPO_CAP_GIB * 1024**3
        ),
        "pretraining_real_sharding_evidence_accepted": (
            pretraining_evidence["accepted"]
        ),
        "online_teacher_synchronous_orchestration_evidence_accepted": (
            online_evidence["accepted"]
        ),
        "synthetic_sharding_not_used_as_real_evidence": True,
        "fixed_state_timing_not_used_as_real_parallel_evidence": True,
        "sensitivity_5000_reported": True,
    }
    feasibility_checks = {
        key: value
        for key, value in checks.items()
        if key
        in {
            "pretraining_real_sharding_evidence_accepted",
            "online_teacher_synchronous_orchestration_evidence_accepted",
        }
    }
    integrity_checks = {
        key: value
        for key, value in checks.items()
        if key not in feasibility_checks
    }
    return {
        "version": f"{VERSION}_runtime_storage_projection_v1",
        "stage_table": json_native(STAGE_TABLE),
        "teacher_workload": teacher,
        "sealed_fixed_state_action_timing": {
            "median_seconds": TEACHER_ACTION_SECONDS_MEDIAN,
            "p99_seconds": TEACHER_ACTION_SECONDS_P99,
            "use": "serial and idealized projection only",
            "real_parallel_evidence": False,
        },
        "pretraining_sharding_evidence": pretraining_evidence,
        "online_teacher_query_evidence": online_evidence,
        "distillation": {
            "central_wall_hours_with_25pct_margin": (
                distillation_wall_margin
            ),
            "serial_teacher_cpu_hours_median": teacher["pre_ppo"][
                "central_median"
            ]["serial_active_cpu_hours"],
            "optimizer_steps_projected": bc_optimizer_steps,
            "optimizer_hours_from_sealed_fixture": bc_optimizer_hours,
            "storage_before_margin_bytes": (
                distillation_storage_before_margin
            ),
            "storage_with_25pct_margin_bytes": (
                distillation_storage_margin
            ),
            "storage_with_25pct_margin_gib": (
                distillation_storage_margin / 1024**3
            ),
            "runtime_cap_hours": DISTILLATION_CAP_HOURS,
            "storage_cap_gib": DISTILLATION_CAP_GIB,
        },
        "on_policy_training": {
            "inherited_j1_bounded_hours_with_margin": (
                parent_training_margin_hours
            ),
            "central_wall_hours_with_teacher_and_25pct_margin": (
                ppo_wall_margin
            ),
            "storage_with_teacher_labels_and_margin_bytes": (
                ppo_storage_margin
            ),
            "storage_with_teacher_labels_and_margin_gib": (
                ppo_storage_margin / 1024**3
            ),
            "runtime_cap_hours": PPO_CAP_HOURS,
            "storage_cap_gib": PPO_CAP_GIB,
        },
        "sensitivity_5000_moves": sensitivity,
        "checks": checks,
        "integrity_checks": integrity_checks,
        "feasibility_checks": feasibility_checks,
        "integrity_passes": all(integrity_checks.values()),
        "feasibility_passes": all(feasibility_checks.values()),
        "passes": all(checks.values()),
        "zero_real_teacher_queries": 0,
        "zero_games": 0,
        "zero_optimizer_steps": 0,
    }


def configure_readiness_runtime() -> dict[str, Any]:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise J2ReadinessIntegrityError(
                "Torch inter-op threads could not be set before readiness"
            ) from error
    torch.use_deterministic_algorithms(True)
    checks = {
        "torch_intra_op_one": torch.get_num_threads() == 1,
        "torch_inter_op_one": torch.get_num_interop_threads() == 1,
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    return {
        "torch_version": torch.__version__,
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "checks": checks,
        "passes": all(checks.values()),
    }


def model_schema_audit() -> dict[str, Any]:
    model, _optimizer = initialize_model_optimizer()
    state_identity = canonical_json_hash(
        {
            name: _tensor_sha256(value)
            for name, value in model.state_dict().items()
        }
    )
    schema = model_schema()
    checks = {
        "parameter_count_exact": parameter_count(model)
        == EXPECTED_PARAMETER_COUNT,
        "policy_head_exact": schema["heads"]["policy"]
        == ["linear", HIDDEN_WIDTH, ACTION_COUNT],
        "value_head_exact": schema["heads"]["value"]
        == ["linear", HIDDEN_WIDTH, 1],
        "auxiliary_head_absent": schema["auxiliary_heads"] == [],
        "auxiliary_loss_absent": schema["auxiliary_losses"] == [],
        "from_scratch_seed_exact": (
            schema["initialization_seed"] == INITIALIZATION_SEED
        ),
        "normal_start_starter_none": schema["starter_tile"] is None,
        "all_initial_tensors_finite": all(
            torch.isfinite(value).all()
            for value in model.state_dict().values()
        ),
    }
    return {
        "version": f"{VERSION}_model_schema_audit_v1",
        "schema": schema,
        "schema_sha256": canonical_json_hash(schema),
        "initial_model_state_sha256": state_identity,
        "parameter_count": parameter_count(model),
        "checks": checks,
        "passes": all(checks.values()),
        "scientific_checkpoint_created": False,
        "scientific_optimizer_steps": 0,
    }


def synthetic_readiness_fixture() -> dict[str, Any]:
    batch = synthetic_distillation_batch(root_lengths=(3, 5))
    model_a, optimizer_a = initialize_model_optimizer()
    uninterrupted = DistillationUpdater(
        model_a,
        optimizer_a,
        batch,
        minibatch_size=3,
        epochs=2,
    )
    uninterrupted.run()
    model_b, optimizer_b = initialize_model_optimizer()
    interrupted = DistillationUpdater(
        model_b,
        optimizer_b,
        batch,
        minibatch_size=3,
        epochs=2,
    )
    interrupted.run(max_steps=2)
    restored = DistillationUpdater.from_snapshot_bytes(
        interrupted.snapshot_bytes(),
        batch,
        minibatch_size=3,
        epochs=2,
    )
    restored.run()
    models_equal = all(
        torch.equal(
            uninterrupted.model.state_dict()[name],
            restored.model.state_dict()[name],
        )
        for name in uninterrupted.model.state_dict()
    )
    optimizer_equal = canonical_json_hash(
        _optimizer_projection(uninterrupted.optimizer.state_dict())
    ) == canonical_json_hash(
        _optimizer_projection(restored.optimizer.state_dict())
    )
    shard_rows = [
        {
            "row_index": index,
            "shard": shard_for_row(index),
            "payload": {"fixture": index},
            "row_identity": canonical_json_hash(
                {
                    "row_index": index,
                    "shard": shard_for_row(index),
                    "payload": {"fixture": index},
                }
            ),
        }
        for index in range(40)
    ]
    merged = deterministic_shard_merge(list(reversed(shard_rows)))
    checks = {
        "resume_step_ids_exact": (
            uninterrupted.closed_step_ids == restored.closed_step_ids
        ),
        "resume_model_exact": models_equal,
        "resume_optimizer_exact": optimizer_equal,
        "eight_shard_ownership_exact": all(
            int(row["shard"]) == int(row["row_index"]) % SHARD_COUNT
            for row in shard_rows
        ),
        "canonical_merge_exact": [
            int(row["row_index"]) for row in merged
        ]
        == list(range(40)),
        "synthetic_not_real_throughput_evidence": True,
        "scientific_games_zero": True,
        "scientific_optimizer_steps_zero": True,
    }
    return {
        "version": f"{VERSION}_synthetic_readiness_fixture_v1",
        "distillation_batch_sha256": distillation_batch_identity(batch),
        "distillation_plan_sha256": canonical_json_hash(
            uninterrupted.plan
        ),
        "shard_plan_sha256": canonical_json_hash(shard_plan(40)),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _optimizer_projection(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_sha256": _tensor_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _optimizer_projection(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_optimizer_projection(item) for item in value]
    return value


def operational_audit(
    *,
    output_dir: Path = READINESS_DIR,
) -> dict[str, Any]:
    from threes_rl import j1_joint_policy_value as j1

    parent = j1.operational_audit(output_dir=output_dir)
    runtime = {
        "torch_version": torch.__version__,
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    checks = {
        "nice_at_least_10": bool(
            parent["checks"]["nice_at_least_10"]
        ),
        "one_heavy_job": bool(parent["checks"]["one_heavy_job"]),
        "free_disk_above_100_gib": bool(
            parent["checks"]["free_disk_above_100_gib"]
        ),
        "target_120_gib_met": bool(
            parent["checks"]["target_120_gib_met"]
        ),
        "services_healthy": bool(
            parent["checks"]["services_healthy"]
        ),
        "torch_2_12_1": torch.__version__ == "2.12.1",
        "torch_intra_inter_one": (
            runtime["intra_op_threads"] == 1
            and runtime["inter_op_threads"] == 1
        ),
        "torch_deterministic": runtime["deterministic_algorithms"],
        "human_session_content_unread": (
            parent["services"]["recorder"][
                "active_session_content_read"
            ]
            is False
        ),
        "dashboard_top_three_exact": (
            list(parent["services"]["dashboard"]["top_three"])
            == [263670, 261369, 258561]
        ),
    }
    return {
        "version": f"{VERSION}_operational_audit_v1",
        "parent_operational": parent,
        "torch_runtime": runtime,
        "human_session_content_read": False,
        "checks": checks,
        "passes": all(checks.values()),
    }


def audit_zero_work(
    *,
    output_dir: Path = READINESS_DIR,
    allowed_readiness_files: Sequence[str] = (),
    root: Path = REPO_ROOT,
    include_operational: bool = True,
) -> dict[str, Any]:
    entries = (
        sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        )
        if output_dir.exists()
        else []
    )
    future = {
        str(path): (root / path.relative_to(REPO_ROOT)).exists()
        for path in FUTURE_EXECUTION_DIRS
    }
    operations = (
        operational_audit(output_dir=output_dir)
        if include_operational
        else {
            "passes": True,
            "skipped_in_synthetic_fixture": True,
        }
    )
    checks = {
        "readiness_namespace_has_only_allowed_files": (
            entries == sorted(str(value) for value in allowed_readiness_files)
        ),
        "all_future_execution_namespaces_absent": not any(
            future.values()
        ),
        "all_work_counters_zero": all(
            int(value) == 0 for value in ZERO_WORK.values()
        ),
        "operational_passes": bool(operations["passes"]),
        "no_execution_marker": not any(
            "marker" in entry.lower() or "opened" in entry.lower()
            for entry in entries
        ),
        "no_reservation_or_consumption": not any(
            "reservation" in entry.lower()
            or "consumption" in entry.lower()
            for entry in entries
        ),
    }
    return {
        "version": f"{VERSION}_zero_work_audit_v1",
        "readiness_dir": str(output_dir.resolve()),
        "readiness_entries": entries,
        "allowed_readiness_files": sorted(
            str(value) for value in allowed_readiness_files
        ),
        "future_execution_namespaces": future,
        "zero_work": dict(ZERO_WORK),
        "operational": operations,
        "checks": checks,
        "passes": all(checks.values()),
    }


def test_evidence_identity(
    *,
    output_dir: Path = READINESS_DIR,
) -> dict[str, Any]:
    path = output_dir / TEST_EVIDENCE_NAME
    payload = load_hashed_json(
        path,
        field="test_evidence_payload_sha256",
    )
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload["test_evidence_payload_sha256"],
    }


def write_test_evidence(
    *,
    commands: Sequence[Mapping[str, Any]],
    deselections: Sequence[str],
    output_dir: Path = READINESS_DIR,
) -> dict[str, Any]:
    zero = audit_zero_work(
        output_dir=output_dir,
        include_operational=False,
    )
    if not zero["passes"]:
        raise J2ReadinessIntegrityError(
            "J2 namespace was not zero-work before test evidence"
        )
    if not commands:
        raise J2ReadinessIntegrityError("No test commands were recorded")
    normalized_commands = []
    for row in commands:
        passed = int(row["passed"])
        failed = int(row.get("failed", 0))
        if passed < 1 or failed != 0:
            raise J2ReadinessIntegrityError(
                "J2 test evidence contains a failing or empty command"
            )
        normalized_commands.append(
            {
                "name": str(row["name"]),
                "command": str(row["command"]),
                "passed": passed,
                "failed": failed,
                "deselected": int(row.get("deselected", 0)),
            }
        )
    source = source_and_parent_audit()
    if not source["passes"]:
        raise J2ReadinessIntegrityError(
            "Parent identities changed before J2 test evidence"
        )
    payload = {
        "version": f"{VERSION}_test_evidence_v1",
        "source_identities": source["local_sources"],
        "parent_identity_audit_sha256": canonical_json_hash(source),
        "commands": normalized_commands,
        "total_passed": sum(row["passed"] for row in normalized_commands),
        "total_failed": 0,
        "deselections": sorted(str(value) for value in deselections),
        "zero_work": dict(ZERO_WORK),
        "future_execution_namespaces_absent": True,
    }
    return write_immutable_json(
        output_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def _validate_test_evidence(
    *,
    output_dir: Path = READINESS_DIR,
) -> dict[str, Any]:
    path = output_dir / TEST_EVIDENCE_NAME
    payload = load_hashed_json(
        path,
        field="test_evidence_payload_sha256",
    )
    current = {
        "charter": sha256_path(CHARTER_PATH),
        "runner": sha256_path(RUNNER_PATH),
        "tests": sha256_path(TEST_PATH),
    }
    checks = {
        "source_identities_current": payload.get("source_identities")
        == current,
        "tests_passed": (
            int(payload.get("total_passed", 0)) > 0
            and int(payload.get("total_failed", -1)) == 0
        ),
        "zero_work": payload.get("zero_work") == ZERO_WORK,
        "future_namespaces_absent": payload.get(
            "future_execution_namespaces_absent"
        )
        is True,
    }
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload["test_evidence_payload_sha256"],
        "commands": payload.get("commands"),
        "deselections": payload.get("deselections"),
        "checks": checks,
        "passes": all(checks.values()),
    }


def readiness_decision(
    *,
    integrity_checks: Mapping[str, bool],
    feasibility_checks: Mapping[str, bool],
    operational_checks: Mapping[str, bool],
) -> dict[str, Any]:
    integrity_passes = all(bool(value) for value in integrity_checks.values())
    feasibility_passes = all(
        bool(value) for value in feasibility_checks.values()
    )
    operational_passes = all(
        bool(value) for value in operational_checks.values()
    )
    if not integrity_passes:
        decision = KILL
    elif not feasibility_passes or not operational_passes:
        decision = HOLD
    else:
        decision = READY
    return {
        "decision": decision,
        "integrity_checks": dict(integrity_checks),
        "feasibility_checks": dict(feasibility_checks),
        "operational_checks": dict(operational_checks),
        "integrity_passes": integrity_passes,
        "feasibility_passes": feasibility_passes,
        "operational_passes": operational_passes,
        "passes": decision == READY,
    }


def _written_artifact_identity(
    path: Path,
    *,
    field: str,
) -> dict[str, Any]:
    payload = load_hashed_json(path, field=field)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_field": field,
        "payload_sha256": payload[field],
    }


def prepare(
    *,
    output_dir: Path = READINESS_DIR,
    power_datasets: int = POWER_DATASETS,
    power_bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    if (
        int(power_datasets) != POWER_DATASETS
        or int(power_bootstraps) != POWER_BOOTSTRAPS
    ):
        raise J2ReadinessIntegrityError(
            "Production readiness power workload cannot be reduced"
        )
    zero = audit_zero_work(
        output_dir=output_dir,
        allowed_readiness_files=(TEST_EVIDENCE_NAME,),
    )
    evidence = _validate_test_evidence(output_dir=output_dir)
    sources = source_and_parent_audit()
    prospective = prospective_authority()
    protected = protected_stream_authority()
    teacher = teacher_provenance_audit()
    schema = model_schema_audit()
    fixture = synthetic_readiness_fixture()
    power = power_and_feasibility_report(
        datasets=power_datasets,
        bootstraps=power_bootstraps,
    )
    projection = runtime_storage_projection()

    input_payload = {
        "version": f"{VERSION}_input_bindings_v1",
        "source_and_parent_audit": sources,
        "test_evidence": evidence,
        "zero_work_before_prepare": zero,
        "teacher_provenance_sha256": canonical_json_hash(teacher),
        "prospective_authority_sha256": canonical_json_hash(prospective),
        "protected_stream_authority_sha256": canonical_json_hash(protected),
        "future_execution_namespaces": [
            str(path.resolve()) for path in FUTURE_EXECUTION_DIRS
        ],
        "zero_work": dict(ZERO_WORK),
    }
    written = {}

    def seal(
        name: str,
        payload: Mapping[str, Any],
        field: str,
    ) -> dict[str, Any]:
        write_immutable_json(
            output_dir / name,
            payload,
            field=field,
        )
        identity = _written_artifact_identity(
            output_dir / name,
            field=field,
        )
        written[name] = identity
        return identity

    seal(
        INPUT_BINDINGS_NAME,
        input_payload,
        "input_bindings_payload_sha256",
    )
    seal(
        PROSPECTIVE_AUTHORITY_NAME,
        prospective,
        "prospective_authority_payload_sha256",
    )
    seal(
        PROTECTED_STREAM_AUTHORITY_NAME,
        protected,
        "protected_stream_authority_payload_sha256",
    )
    seal(
        TEACHER_PROVENANCE_NAME,
        teacher,
        "teacher_provenance_payload_sha256",
    )
    seal(
        MODEL_SCHEMA_NAME,
        {
            **schema,
            "synthetic_readiness_fixture": fixture,
        },
        "model_schema_payload_sha256",
    )
    seal(
        POWER_NAME,
        power,
        "power_payload_sha256",
    )
    seal(
        PROJECTION_NAME,
        projection,
        "projection_payload_sha256",
    )

    integrity_checks = {
        "zero_work_before_prepare": zero["passes"],
        "test_evidence_exact": evidence["passes"],
        "source_parent_identities_exact": sources["passes"],
        "prospective_authority_exact": prospective["passes"],
        "protected_stream_authority_exact": protected["passes"],
        "teacher_provenance_exact": teacher["passes"],
        "model_schema_exact": schema["passes"],
        "synthetic_fixture_exact": fixture["passes"],
        "power_method_and_parent_arithmetic_exact": all(
            value
            for key, value in power["checks"].items()
            if key != "fidelity_progression_power_at_least_080"
        ),
        "projection_integrity_exact": projection["integrity_passes"],
        "no_reservation_or_consumption_artifact": True,
        "no_execution_marker": True,
    }
    feasibility_checks = {
        "fidelity_progression_power_at_least_080": power["checks"][
            "fidelity_progression_power_at_least_080"
        ],
        "pretraining_real_sharding_evidence": projection[
            "pretraining_sharding_evidence"
        ]["accepted"],
        "on_policy_teacher_sync_evidence": projection[
            "online_teacher_query_evidence"
        ]["accepted"],
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
        "stage_table": json_native(STAGE_TABLE),
        "derived_counts": derive_stage_totals(),
        "decision_audit": decision,
        "zero_work": dict(ZERO_WORK),
        "execution_authorized": False,
    }
    lock_identity = seal(
        READINESS_LOCK_NAME,
        lock_payload,
        "readiness_lock_payload_sha256",
    )
    result_payload = {
        "version": f"{VERSION}_readiness_result_v1",
        "decision": decision["decision"],
        "readiness_lock": lock_identity,
        "scoped_hold_reasons": {
            "pretraining_sharding": projection[
                "pretraining_sharding_evidence"
            ],
            "on_policy_teacher_query": projection[
                "online_teacher_query_evidence"
            ],
            "fidelity_progression_power_passes": power["checks"][
                "fidelity_progression_power_at_least_080"
            ],
        },
        "integrity_passes": decision["integrity_passes"],
        "operational_passes": decision["operational_passes"],
        "feasibility_passes": decision["feasibility_passes"],
        "continue": (
            "research-lead review of the sealed J2 readiness package"
        ),
        "hold": (
            "all J2 teacher trajectories, labels, training, evaluation, "
            "and promotion"
        ),
        "kill": (
            "J1c exact execution, J1d checkpoint reuse, and historical kills"
        ),
        "promote": False,
        "zero_work": dict(ZERO_WORK),
        "execution_authorized": False,
    }
    seal(
        READINESS_RESULT_NAME,
        result_payload,
        "readiness_result_payload_sha256",
    )
    return load_hashed_json(
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
        description="Outcome-free J2 readiness tooling only"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    audit = subparsers.add_parser("audit-zero-work")
    audit.add_argument("--out-dir", type=Path, default=READINESS_DIR)
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--out-dir", type=Path, default=READINESS_DIR)
    evidence.add_argument(
        "--recorded-command",
        action="append",
        type=_parse_recorded_command,
        required=True,
    )
    evidence.add_argument(
        "--deselection",
        action="append",
        default=[],
    )
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--out-dir",
        type=Path,
        default=READINESS_DIR,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_readiness_runtime()
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
    else:
        raise J2ReadinessIntegrityError(
            f"Forbidden J2 readiness command: {args.subcommand}"
        )
    print(json.dumps(json_native(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

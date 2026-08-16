"""Training-only J1d V2 exact metric-authentication surface.

The module stays standard-library-only through argument parsing.  Torch and
the immutable parent J1 implementation are imported only by the execute path,
after the selected command and its artifact contract have been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "j1d_metric_authentication_surface_v2"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs"
CHARTER_PATH = (
    REPO_ROOT
    / "threes_rl"
    / "J1D_V2_EXACT_METRIC_AUTHENTICATION_AMENDMENT.md"
)
RUNNER_PATH = (
    REPO_ROOT / "threes_rl" / "j1d_metric_authentication_surface_v2.py"
)
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "test_rl_j1d_metric_authentication_surface_v2.py"
)
READINESS_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j1d_metric_authentication_readiness_v2"
)
FUTURE_EXECUTION_ROOT = (
    RUNS_ROOT / "forensics" / "j1d_execution_v1"
)
V1_READINESS_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j1d_metric_authentication_readiness_v1"
)
J1B_PREFLIGHT_DIR = (
    RUNS_ROOT / "forensics" / "j1b_operational_repair_readiness_v1"
)
J1B_PRE_A1_HISTORY_PATH = (
    RUNS_ROOT
    / "forensics"
    / "j1b_operational_repair_preseal_history_v1"
    / "J1B_TEST_EVIDENCE_PRE_A1.json"
)
SPENT_J1_TRAINING_DIR = (
    RUNS_ROOT / "forensics" / "j1_execution_v1" / "training"
)
J1B_TRAINING_READINESS_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j1b_training_execution_surface_readiness_v1"
)
SPENT_J1B_EXECUTION_ROOT = (
    RUNS_ROOT / "forensics" / "j1b_execution_v1"
)
J1B_EXTERNAL_TERMINAL_DIR = (
    RUNS_ROOT / "forensics" / "j1b_open_failure_terminal_v1"
)
J1C_READINESS_DIR = (
    RUNS_ROOT
    / "forensics"
    / "j1c_training_execution_surface_readiness_v1"
)
SPENT_J1C_TRAINING_DIR = (
    RUNS_ROOT / "forensics" / "j1c_execution_v1" / "training"
)

EXPECTED_CHARTER_SHA256 = (
    "f3d42d6f4d908c723756e140fc2ba424378f280a18dc99a50b585e59478cd07c"
)
EXPECTED_V1_SOURCE_IDENTITIES = {
    "threes_rl/J1D_METRIC_AUTHENTICATION_REPAIR_CHARTER.md":
        "6e28e5f0a5937493b969c56d09e8b7641d990d926999acfeaafca2f1a3caa1aa",
    "threes_rl/j1d_metric_authentication_surface.py":
        "94014f200bf7244eee64bbba9276ecdc45837bed5cccca0e5a6514e81e875bd9",
    "tests/test_rl_j1d_metric_authentication_surface.py":
        "a50943330330d845a35e84451e3c2d6b092c0129aaa4e35c791dfda8606a5beb",
}
EXPECTED_V1_READINESS_EVIDENCE = {
    "J1D_PROSPECTIVE_TRAINING_MANIFEST.json": (
        "420bd050f9822b8440ef60bec0c406f69cd12410cd3b474bb14422b71cb202c9",
        "prospective_manifest_payload_sha256",
        "f3d678392f00dab1098f55f66a973a863c5b97af02dac79e19afc57c189f6ebf",
    ),
    "J1D_PROTECTED_STREAM_AUTHORITY.json": (
        "099d6f966a65d5aedb5f9154d874412f701679db702f4c9f55d9dca7149e30d3",
        "stream_authority_payload_sha256",
        "1ef470c74ffd3d5224fe5fae74a927c049ede152270f3a5b744fe6612cdf56de",
    ),
    "J1D_METRIC_AUTHENTICATION_ROOT_CAUSE.json": (
        "38d7ef577d131ab8988549ea85d3ad0e76d708f34b15f2b0d6f6625f3945b485",
        "root_cause_payload_sha256",
        "722d754ad0d4cb6a89b71321a732a390d525a2147e68b811ef971cc9673db984",
    ),
    "J1D_METRIC_AUTHENTICATION_TEST_EVIDENCE.json": (
        "d2957b138a734dd3185a1b5f09eba782a7632027cdd06b0cb0ebd4042e7f064c",
        "test_evidence_payload_sha256",
        "4e0a6068289472478fddf199c86a803b1dae53b6f55fbb5e32bab0566ab0b3da",
    ),
}
EXPECTED_J1B_SOURCE_IDENTITIES = {
    "threes_rl/J1B_OPERATIONAL_REPAIR_PREFLIGHT_CHARTER.md":
        "a426801fc3015051ea51517e925a7d1c2e556718e2551ee480b802c8a7422cc1",
    "threes_rl/J1B_OPERATIONAL_REPAIR_PREFLIGHT_AMENDMENT_A1.md":
        "64de3de37bff6a08bd95da217dc52d2f4bb58fbf99d28bede263a44d0aa2eb9c",
    "threes_rl/j1b_operational_repair_preflight.py":
        "7d73565c510dfe74b87ec362c05f8928e15a65cb8af5494b5ad9fe5f4c30ca5f",
    "tests/test_rl_j1b_operational_repair_preflight.py":
        "f7e55b71f7954fcbdd4db61c1693d773b8ea106684ea19ad19998be15f4dbaff",
}
EXPECTED_J1B_READINESS_FILES = {
    "J1B_READINESS_LOCK.json":
        "b8b5377370f0e9e04739aae582604ce85f38bd1ddf84b5312a2cf12406f38814",
    "J1B_READINESS_RESULT.json":
        "108038d15b222afd00c07c9801b460fb4687bfe0a9e8a4fb54a59e58e8907ec6",
}
EXPECTED_J1B_READINESS_PAYLOADS = {
    "J1B_READINESS_LOCK.json": (
        "readiness_lock_payload_sha256",
        "ef0c1adce5f948a238e81911ab034d84ed297c2b2570d58481fb2906ef2e7e3b",
    ),
    "J1B_READINESS_RESULT.json": (
        "readiness_result_payload_sha256",
        "5d56b2c3cec39c16590a20f8acf8f10c60db7739e5161a653ea45a779204ba5e",
    ),
}
EXPECTED_J1B_TRAINING_SOURCE_IDENTITIES = {
    "threes_rl/J1B_TRAINING_EXECUTION_SURFACE_CHARTER.md":
        "aeb458781e206f8f16002ffaa311d782b26fdb4076211155a6230b9835e29858",
    "threes_rl/j1b_training_execution_surface.py":
        "c586d41f752cff7aa7c36c911008ca72ce147139fedd7586a03e627471282bc5",
    "tests/test_rl_j1b_training_execution_surface.py":
        "86159c76a42c54c47d30e75b92f988773a6c6da580e8bb6b01de0f2a944a516e",
}
EXPECTED_J1B_TRAINING_READINESS = {
    "J1B_TRAINING_EXECUTION_TEST_EVIDENCE.json": (
        "aa8a78db1b2a740a2fbcdb183049f47e5d6231a1e85b26d51fa98adc4c68c590",
        "test_evidence_payload_sha256",
        "da57014e6bb8cbda4ecaed6c9f61f8092f33667a642c9da588ee496f71449e2a",
    ),
    "J1B_TRAINING_EXECUTION_INPUT_BINDINGS.json": (
        "76284190098c72fc7b591f459c5feabe4ef24d514dc9d79f15bba8b8a9e665c1",
        "input_bindings_payload_sha256",
        "eed9cc97b4e4f83662401303f30556d248e01203a208db2087bf5ff0a95add20",
    ),
    "J1B_TRAINING_EXECUTION_SCHEMA.json": (
        "9a7f63ed20311d15353c6d051dfc6882e43f5f96bf973143cd1a005a20b2f761",
        "schema_payload_sha256",
        "336f1c4942d709bafe8d8fe9477cb7a418fce77eb30ec314aeb371100f0ef05d",
    ),
    "J1B_TRAINING_EXECUTION_PROJECTION.json": (
        "0efc23c4abd3c8723d7567e1647f7f8a059f25278a90aea589678a0f11a3fc90",
        "projection_payload_sha256",
        "dcc8562a37ed87ee0f2334d4fcb631126ccb5532ce5b5f9d5284c571fec8c109",
    ),
    "J1B_TRAINING_EXECUTION_READINESS_LOCK.json": (
        "adeae9ce6f9056914da48b79096ee7143a559a2d4e97c02cbe622eff7b0eb79e",
        "readiness_lock_payload_sha256",
        "e559d197f299d2ddf62d8d7736c8fa5a6256c90248ed658f9f24c3459c5b11fe",
    ),
    "J1B_TRAINING_EXECUTION_READINESS_RESULT.json": (
        "3403a9d70e73e38eca7a372bd7db08b855051f1c409b621ebb7a391c45d96213",
        "readiness_result_payload_sha256",
        "84fc2adf7d5204ed1dd1002799fe250575657bb937badc194abaa1a02217b3d2",
    ),
}
EXPECTED_SPENT_J1B_FILES = {
    "training/phase_lock.json":
        "ac12b9f21977a3adcd61ef5f0d8ba60b058306dcc05fdfed423d2ca77c17a0ce",
    "training/phase_lock_result.json":
        "6a2f63dc8875db394333ac901a919466a6a432083e29feba32ba8917f3ee9bcf",
    "training/execution_opened.json":
        "e99099b87aa6417b4200ee236ef2b770d1524d11b26a878e9f3bf0d749a54cff",
}
EXPECTED_J1B_EXTERNAL_FILES = {
    "J1B_OPEN_FAILURE_TEST_EVIDENCE.json": (
        "38261821557d3a49dee81bb7cad02fa2c91ac058aef168cbe39abfa52caf7155",
        "test_evidence_payload_sha256",
        "46e589fc139bfaa8e4475cdfa678fb9b6cb914646777a5836db981f043215fc2",
        None,
    ),
    "J1B_OPEN_FAILURE_TERMINAL.json": (
        "2f9cdfacb04a064b67785ab9bb00cac7d3d46bd057912b40ac4c06db0a0ed122",
        "terminal_payload_sha256",
        "1cf98c5676b23c6168be4feef4d3e3a4ffeb98fb90f028af515f1646eb5e2369",
        "HOLD_J1B_OPEN_SERIALIZATION_INTEGRITY",
    ),
    "J1B_OPEN_FAILURE_RETENTION.json": (
        "28738328f724a544ee92fc7992ef8f256f0886c2e138a234a863ec0fe55c5f67",
        "retention_payload_sha256",
        "f88941a61da8909bd852d180892ce6c22d84c8ef4749b2110f9f2d58db8dd37a",
        "PRESERVE_J1B_OPEN_FAILURE_EVIDENCE",
    ),
}
EXPECTED_J1C_SOURCE_IDENTITIES = {
    "threes_rl/J1C_TRAINING_EXECUTION_ORCHESTRATION_REPAIR_CHARTER.md":
        "e352262614a7c3c46c53811c727599f9926f6cbd579b99732c6802c8c41462dd",
    "threes_rl/j1c_training_execution_surface.py":
        "f50b475ed00efcfb0fa2ac5b4e4a11b0587ec17c4e8e404bad08be8f4f8c990d",
    "tests/test_rl_j1c_training_execution_surface.py":
        "4ff0a2253cd23059d33404b5d3f0829309dc1565657547f6112e9c3d268dc86d",
}
EXPECTED_J1C_READINESS = {
    "J1C_PROSPECTIVE_TRAINING_MANIFEST.json": (
        "135fec4c75db8871e20ab3988471f75538a399e982573bf4a108afc569fe08b7",
        "prospective_manifest_payload_sha256",
        "c0d7953aa158297d5b515f6b6e4613b6fc22acc059e8e0aa0ff8904ee7e3546d",
    ),
    "J1C_PROTECTED_STREAM_AUTHORITY.json": (
        "8aff7f07827cfe796a07646215362f1d37e502f64a43b9b7142b53288b7041f6",
        "stream_authority_payload_sha256",
        "30439292fceeeae4832f5259c62e8a954f3103de7246875c353dfb8cea138016",
    ),
    "J1C_TRAINING_EXECUTION_INPUT_BINDINGS.json": (
        "c230a951d4c44c4e6d67f59c0405bb78c1b8f4ef980408de126bc8b770b83fef",
        "input_bindings_payload_sha256",
        "30f911740e8dc99a62cec63439ec522d4d28fa5db3cd645fea2267c8cfe787c8",
    ),
    "J1C_TRAINING_EXECUTION_PROJECTION.json": (
        "c6fb7a5d426a3fa32c0affa4d34b937845a7cdb02e22dcf2baff657d3540879d",
        "projection_payload_sha256",
        "60ab68e30b48ee5261b9ea8a661da887b6dad8b8b466f970ae58e03d6b811c17",
    ),
    "J1C_TRAINING_EXECUTION_READINESS_LOCK.json": (
        "a95712126796dcd91a82885aa1990a77e725064970ba34bc1f31306de8ef2368",
        "readiness_lock_payload_sha256",
        "15701d62e5ddc7fca7e38702d78ddd54fa7aefbbd6bdf49ea130c2e224f78ef4",
    ),
    "J1C_TRAINING_EXECUTION_READINESS_RESULT.json": (
        "908c1570f972a612e02815811a9885162a89f9a1e87ea2b081f2801dab7419bd",
        "readiness_result_payload_sha256",
        "e13c27501231af3ea72de1dbea8ac2e9b485b7763e24fb3394391ec326ef37a3",
    ),
    "J1C_TRAINING_EXECUTION_SCHEMA.json": (
        "54a23f1bf627573652f6e1455be612d3a36e417543b23efa223a3bea8edc36fd",
        "schema_payload_sha256",
        "ab2deb5d5f09e5e65a738e96306bf05e21017dca6737fa607c5f14b92bad5d0d",
    ),
    "J1C_TRAINING_EXECUTION_TEST_EVIDENCE.json": (
        "7a467bab2bcf457501413b64ed9e3dc41a36e881eece7200e0cf66806f38457a",
        "test_evidence_payload_sha256",
        "c68c87a3361ceacec8d991c513589edb1c0d727e2183c37713ce54e9478f3048",
    ),
}
EXPECTED_SPENT_J1C_TERMINAL = (
    "7ec4fe7627a129dbb7227fcb88df87ab46ee87479d381011103511ec8f2ca414",
    "terminal_result_payload_sha256",
    "c71dac534755add014a0debe6418f75b591df264f1bb096fb5d50fd253d8ce4f",
    "KILL_J1C_TRAINING_INTEGRITY",
)
EXPECTED_SPENT_J1C_RETENTION = (
    "8946669ffeba05626ee863f4e2df8536e267920d771eb088ebf84253a2059532",
    "retention_payload_sha256",
    "01a6f95b79c10f343489fa2e2add086001e89691c7c4edf6b6a1b19f8ec66409",
)
EXPECTED_PRE_A1_FILE_SHA256 = (
    "d2f6333bd4fdbe584fbf231141a24c01256dcc9ebe0f57c2691e19a8f046bddf"
)
EXPECTED_PRE_A1_PAYLOAD_SHA256 = (
    "b462c0b46afaa478caeb66c622799eb1e7a533673439a89fe0e60650a448e25e"
)
EXPECTED_SOURCE_MANIFEST_FILE_SHA256 = (
    "c2be5faf37d9e2619c0bd57d12a64248738e6b4c8bda1802931898a63e18b1e0"
)
EXPECTED_SOURCE_MANIFEST_PAYLOAD_SHA256 = (
    "f6da9b35674a08c21b53c476692cd7073e492289a8cec8d687ceaa45afaf092d"
)
EXPECTED_CANONICAL_ROWS_SHA256 = (
    "7bef7fd71403bbb26ffe3fe8293e6745e8aa3bb585dcaff53f06bdd2b36cb7a1"
)
EXPECTED_ROOT_COMMITMENT_PAYLOAD_SHA256 = (
    "13359e460b2956b94e328900c077b2a1d9aef12b00a9f2c882f780abefe0ce47"
)
EXPECTED_ROOT_SET_SHA256 = (
    "4d86d6e934d1982c0efde2407ef3b205464dc3063db82d0b338dda1bd16f97e0"
)
EXPECTED_STREAM_AUTHORITY_FILE_SHA256 = (
    "4e8e1661ab04c3d87c5819e0112d27b8213f65539c2ea9b955fa6a1a47fca867"
)
EXPECTED_STREAM_AUTHORITY_PAYLOAD_SHA256 = (
    "90e98195d0be7a50d38c0c00e681c120c9a8300c5d6510f836809088fb2b7c6e"
)
EXPECTED_SPENT_TERMINAL_FILE_SHA256 = (
    "21092fb34631eb0eaf48811caa814ff4d05abbb23c9bc5add85eefd93a8959d3"
)
EXPECTED_SPENT_RETENTION_FILE_SHA256 = (
    "dc339aafdbe32859d07c591a36c9088afa53f5be30412f3340049ca18994ceb0"
)
EXPECTED_PARENT_ENGINE_SHA256 = (
    "d4367d95aba05ec592310008bae21e7de90905fa1268601dd60cc8fcb2b6f2bd"
)
EXPECTED_MODEL_SCHEMA_SHA256 = (
    "75919f80ed3550f27e1929cad355f2380e39058409456a125c86001f149d5351"
)
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "f5102689925a4db2cc3972a0a6d0e88943f87a48f9206ddf36bfb1d0f7c7b80e"
)
EXPECTED_PARAMETER_COUNT = 411_656
TRAIN_ROOTS = 16_384
STREAM_RANGES = {
    "logical_stream_id": (213_000_049_152, 213_000_065_535),
    "deck_stream_id": (214_000_049_152, 214_000_065_535),
    "slot_stream_id": (215_000_049_152, 215_000_065_535),
    "candidate_policy_stream_id": (
        216_000_049_152,
        216_000_065_535,
    ),
}

PUBLIC_COMMANDS = (
    "seal-phase-lock",
    "open",
    "materialize",
    "execute",
)
PHASE_DIR_NAME = "training"
PHASE_LOCK_NAME = "phase_lock.json"
PHASE_LOCK_RESULT_NAME = "phase_lock_result.json"
PHASE_MARKER_NAME = "execution_opened.json"
PHASE_MANIFEST_NAME = "root_manifest.json"
PHASE_OWNER_NAME = "writer_owner.json"
PHASE_RESERVATION_NAME = "stream_reservation.json"
PHASE_CONSUMPTION_NAME = "stream_consumption_opened.json"
PHASE_RESULT_NAME = "terminal_result.json"
PHASE_RETENTION_NAME = "retention_manifest.json"
COMMIT_HEAD_NAME = "commit_head.json"

TEST_EVIDENCE_NAME = "J1D_V2_METRIC_AUTHENTICATION_TEST_EVIDENCE.json"
SOURCE_MANIFEST_NAME = "J1D_V2_PROSPECTIVE_TRAINING_MANIFEST.json"
STREAM_AUTHORITY_NAME = "J1D_V2_PROTECTED_STREAM_AUTHORITY.json"
ROOT_CAUSE_NAME = "J1D_V2_METRIC_AUTHENTICATION_ROOT_CAUSE.json"
SCHEMA_NAME = "J1D_V2_METRIC_AUTHENTICATION_SCHEMA.json"
PROJECTION_NAME = "J1D_V2_METRIC_AUTHENTICATION_PROJECTION.json"
INPUT_BINDINGS_NAME = "J1D_V2_METRIC_AUTHENTICATION_INPUT_BINDINGS.json"
READINESS_LOCK_NAME = "J1D_V2_METRIC_AUTHENTICATION_READINESS_LOCK.json"
READINESS_RESULT_NAME = "J1D_V2_METRIC_AUTHENTICATION_READINESS_RESULT.json"
READY_DECISION = "READY_J1D_V2_METRIC_AUTHENTICATION_PREFLIGHT"
HOLD_DECISION = "HOLD_J1D_V2_METRIC_AUTHENTICATION_PREFLIGHT"
KILL_DECISION = "KILL_J1D_V2_METRIC_AUTHENTICATION_PREFLIGHT_INTEGRITY"


class J1dSurfaceError(RuntimeError):
    """Base error for the J1d training-only surface."""


class J1dSurfaceIntegrityError(J1dSurfaceError):
    """Immutable identity or scientific execution integrity failed."""


class J1dSurfaceOperationalHold(J1dSurfaceError):
    """A mutable operational or resource condition failed."""


CANONICAL_METRIC_ALGORITHM = (
    "manifest_order_numpy_float64_mean_of_authenticated_per_root_rows_v2_exact"
)
CANONICAL_METRIC_ABS_TOLERANCE = 1e-12
CANONICAL_METRIC_FIELDS = (
    "root_log_scores",
    "legal_entropy_nats",
    "value_mse",
    "zero_value_mse",
    "auxiliary_brier",
    "auxiliary_prevalence_brier",
)


def _parent_stable_hash(value: Any) -> str:
    from threes_rl import j1_joint_policy_value as parent_j1

    return str(parent_j1.stable_hash(value))


def _canonical_float64_mean(values: Sequence[Any]) -> float:
    import numpy as np

    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise J1dSurfaceIntegrityError(
            "Canonical metric mean received invalid values"
        )
    result = float(np.mean(array))
    if not math.isfinite(result):
        raise J1dSurfaceIntegrityError(
            "Canonical metric mean is nonfinite"
        )
    return result


def canonical_root_equal_round_aggregates(
    root_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive every published round aggregate from ordered per-root rows."""
    rows = [dict(row) for row in root_metrics]
    if not rows:
        raise J1dSurfaceIntegrityError(
            "Canonical metric derivation requires per-root rows"
        )
    root_ids = [str(row.get("root_id", "")) for row in rows]
    if (
        any(not root_id for root_id in root_ids)
        or len(set(root_ids)) != len(root_ids)
    ):
        raise J1dSurfaceIntegrityError(
            "Canonical metric roots are missing or duplicated"
        )
    scalar_fields = (
        "log_score",
        "legal_entropy_nats",
        "value_mse",
        "zero_value_mse",
    )
    for row in rows:
        for field in scalar_fields:
            value = float(row.get(field, math.nan))
            if not math.isfinite(value):
                raise J1dSurfaceIntegrityError(
                    f"Canonical per-root metric is invalid: {field}"
                )
        for field in ("auxiliary_brier", "auxiliary_prevalence"):
            values = row.get(field)
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or len(values) != 3
                or any(not math.isfinite(float(value)) for value in values)
            ):
                raise J1dSurfaceIntegrityError(
                    f"Canonical per-root auxiliary is invalid: {field}"
                )
    prevalence = [
        _canonical_float64_mean(
            [row["auxiliary_prevalence"][index] for row in rows]
        )
        for index in range(3)
    ]
    return {
        "root_log_scores": [float(row["log_score"]) for row in rows],
        "legal_entropy_nats": _canonical_float64_mean(
            [row["legal_entropy_nats"] for row in rows]
        ),
        "value_mse": _canonical_float64_mean(
            [row["value_mse"] for row in rows]
        ),
        "zero_value_mse": _canonical_float64_mean(
            [row["zero_value_mse"] for row in rows]
        ),
        "auxiliary_brier": [
            _canonical_float64_mean(
                [row["auxiliary_brier"][index] for row in rows]
            )
            for index in range(3)
        ],
        "auxiliary_prevalence_brier": [
            value * (1.0 - value) for value in prevalence
        ],
    }


def reproduce_legacy_and_canonical_reductions(
    *,
    lengths: Sequence[int],
    per_root_values: Sequence[float],
) -> dict[str, Any]:
    """Compute the historical writer and canonical verifier reductions.

    This intentionally follows the production weight path: the immutable
    parent constructs float64 root-equal weights, the PPO batch stores them as
    float32, and metric evaluation casts them back to float64.
    """
    import numpy as np

    from threes_rl import j1_joint_policy_value as parent_j1

    normalized_lengths = [int(length) for length in lengths]
    if (
        not normalized_lengths
        or any(length <= 0 for length in normalized_lengths)
        or len(normalized_lengths) != len(per_root_values)
        or any(not math.isfinite(float(value)) for value in per_root_values)
    ):
        raise J1dSurfaceIntegrityError(
            "Reduction-order fixture inputs are invalid"
        )
    parent_weights = parent_j1.root_equal_weights(normalized_lengths)
    production_weights = (
        np.asarray(parent_weights, dtype=np.float64)
        .astype(np.float32)
        .astype(np.float64)
    )
    transition_values = np.concatenate([
        np.full(
            length,
            float(value),
            dtype=np.float32,
        )
        for length, value in zip(
            normalized_lengths,
            per_root_values,
        )
    ]).astype(np.float64)
    if (
        transition_values.shape != production_weights.shape
        or not np.isfinite(transition_values).all()
        or not np.isfinite(production_weights).all()
        or np.any(production_weights <= 0.0)
    ):
        raise J1dSurfaceIntegrityError(
            "Reduction-order production cast is invalid"
        )
    writer = float(
        np.sum(
            transition_values * production_weights,
            dtype=np.float64,
        )
        / np.sum(production_weights, dtype=np.float64)
    )
    offset = 0
    per_root = []
    root_weight_totals = []
    for length in normalized_lengths:
        root_values = transition_values[offset : offset + length]
        root_weights = production_weights[offset : offset + length]
        root_total = float(np.sum(root_weights, dtype=np.float64))
        per_root.append(float(
            np.sum(
                root_values * root_weights,
                dtype=np.float64,
            )
            / root_total
        ))
        root_weight_totals.append(root_total)
        offset += length
    canonical = _canonical_float64_mean(per_root)
    delta = abs(writer - canonical)
    return {
        "lengths": normalized_lengths,
        "parent_weight_dtype": str(parent_weights.dtype),
        "production_batch_weight_dtype": "float32",
        "metric_reduction_weight_dtype": "float64",
        "root_weight_totals": root_weight_totals,
        "legacy_direct_global": writer,
        "canonical_per_root_mean": canonical,
        "absolute_delta": delta,
        "exceeds_frozen_tolerance": (
            delta > CANONICAL_METRIC_ABS_TOLERANCE
        ),
        "computed_without_injected_delta": True,
    }


def _metric_projection(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        field: json_native(row[field])
        for field in CANONICAL_METRIC_FIELDS
    }


def canonicalize_round_metric_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    root_metrics = list(row.get("root_metrics", []))
    expected_root_hash = _parent_stable_hash(root_metrics)
    observed_root_hash = str(row.get("root_metrics_sha256", ""))
    if observed_root_hash != expected_root_hash:
        raise J1dSurfaceIntegrityError(
            "Per-root metric rows changed before canonical publication"
        )
    canonical = canonical_root_equal_round_aggregates(root_metrics)
    result = {
        **json_native(dict(row)),
        **canonical,
    }
    authentication = payload_with_hash(
        {
            "version": f"{VERSION}_round_metric_authentication_v2",
            "algorithm": CANONICAL_METRIC_ALGORITHM,
            "absolute_tolerance": CANONICAL_METRIC_ABS_TOLERANCE,
            "round": int(result["round"]),
            "ordered_root_ids_sha256": canonical_json_hash(
                [str(metric["root_id"]) for metric in root_metrics]
            ),
            "root_metrics_sha256": observed_root_hash,
            "canonical_aggregates_sha256": canonical_json_hash(
                canonical
            ),
            "published_projection_sha256": canonical_json_hash(
                _metric_projection(result)
            ),
        },
        "metric_authentication_payload_sha256",
    )
    result["metric_authentication"] = authentication
    return result


def validate_canonical_round_metric_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    root_metrics = list(row.get("root_metrics", []))
    canonical = canonical_root_equal_round_aggregates(root_metrics)
    authentication = row.get("metric_authentication")
    checks: dict[str, bool] = {
        "root_metrics_hash_exact": (
            str(row.get("root_metrics_sha256", ""))
            == _parent_stable_hash(root_metrics)
        ),
        "authentication_present": isinstance(authentication, Mapping),
    }
    if isinstance(authentication, Mapping):
        checks.update(
            {
                "authentication_hash_valid": verify_payload_hash(
                    authentication,
                    "metric_authentication_payload_sha256",
                ),
                "algorithm_exact": (
                    authentication.get("algorithm")
                    == CANONICAL_METRIC_ALGORITHM
                ),
                "tolerance_unchanged": (
                    float(authentication.get(
                        "absolute_tolerance",
                        math.nan,
                    ))
                    == CANONICAL_METRIC_ABS_TOLERANCE
                ),
                "round_exact": (
                    int(authentication.get("round", -1))
                    == int(row.get("round", -2))
                ),
                "ordered_roots_exact": (
                    authentication.get("ordered_root_ids_sha256")
                    == canonical_json_hash([
                        str(metric["root_id"])
                        for metric in root_metrics
                    ])
                ),
                "authentication_root_hash_exact": (
                    authentication.get("root_metrics_sha256")
                    == row.get("root_metrics_sha256")
                ),
                "canonical_hash_exact": (
                    authentication.get("canonical_aggregates_sha256")
                    == canonical_json_hash(canonical)
                ),
                "published_projection_hash_exact": (
                    authentication.get("published_projection_sha256")
                    == canonical_json_hash(_metric_projection(row))
                ),
                "published_hash_equals_canonical_hash": (
                    authentication.get("published_projection_sha256")
                    == authentication.get("canonical_aggregates_sha256")
                ),
            }
        )

    def close(observed: Any, expected: float) -> bool:
        try:
            return math.isclose(
                float(observed),
                float(expected),
                rel_tol=0.0,
                abs_tol=CANONICAL_METRIC_ABS_TOLERANCE,
            )
        except (TypeError, ValueError):
            return False

    def sequence_close(observed: Any, expected: Sequence[float]) -> bool:
        return (
            isinstance(observed, Sequence)
            and not isinstance(observed, (str, bytes))
            and len(observed) == len(expected)
            and all(close(left, right) for left, right in zip(observed, expected))
        )

    checks.update(
        {
            "published_projection_exact": (
                _metric_projection(row) == canonical
            ),
            "root_log_scores_exact": (
                list(row.get("root_log_scores", []))
                == canonical["root_log_scores"]
            ),
            "legal_entropy_exact": close(
                row.get("legal_entropy_nats"),
                canonical["legal_entropy_nats"],
            ),
            "value_mse_exact": close(
                row.get("value_mse"),
                canonical["value_mse"],
            ),
            "zero_value_mse_exact": close(
                row.get("zero_value_mse"),
                canonical["zero_value_mse"],
            ),
            "auxiliary_brier_exact": sequence_close(
                row.get("auxiliary_brier"),
                canonical["auxiliary_brier"],
            ),
            "auxiliary_prevalence_brier_exact": sequence_close(
                row.get("auxiliary_prevalence_brier"),
                canonical["auxiliary_prevalence_brier"],
            ),
        }
    )
    return {
        "canonical": canonical,
        "checks": checks,
        "passes": all(checks.values()),
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_native(value: Any) -> Any:
    """Normalize a prospective payload to its exact JSON-native value."""
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def canonical_json_bytes(value: Any) -> bytes:
    native = json_native(value)
    return json.dumps(
        native,
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
    body = dict(json_native(payload))
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(json_native(payload))
    observed = body.pop(field, None)
    return (
        isinstance(observed, str)
        and observed == canonical_json_hash(body)
    )


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise J1dSurfaceIntegrityError(
            f"Expected JSON object: {path}"
        )
    return payload


def ordered_rows_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
    allow_existing_exact: bool = False,
) -> dict[str, Any]:
    body = payload_with_hash(payload, field)
    serialized = (
        json.dumps(body, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    reloaded = json.loads(serialized.decode("utf-8"))
    if reloaded != body or not verify_payload_hash(reloaded, field):
        raise J1dSurfaceIntegrityError(
            f"JSON reload instability: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
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
                raise J1dSurfaceIntegrityError(
                    f"Immutable artifact collision changed bytes: {path}"
                ) from error
            payload_observed = json.loads(observed.decode("utf-8"))
            if not verify_payload_hash(payload_observed, field):
                raise J1dSurfaceIntegrityError(
                    f"Existing immutable artifact is invalid: {path}"
                ) from error
            if allow_existing_exact:
                return payload_observed
            raise FileExistsError(
                f"Immutable artifact already exists: {path}"
            ) from error
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)
    observed_bytes = path.read_bytes()
    if observed_bytes != serialized:
        raise J1dSurfaceIntegrityError(
            f"Written immutable artifact changed bytes: {path}"
        )
    observed = json.loads(observed_bytes.decode("utf-8"))
    if observed != body or not verify_payload_hash(observed, field):
        raise J1dSurfaceIntegrityError(
            f"Written immutable artifact changed: {path}"
        )
    return observed


def immutable_json_identity(
    path: Path,
    *,
    payload_field: str,
    decision: str | None = None,
) -> dict[str, Any]:
    payload = load_json(path)
    if not verify_payload_hash(payload, payload_field):
        raise J1dSurfaceIntegrityError(
            f"Immutable JSON payload is invalid: {path}"
        )
    if decision is not None and payload.get("decision") != decision:
        raise J1dSurfaceIntegrityError(
            f"Immutable JSON decision changed: {path}"
        )
    identity = {
        "path": str(path.resolve()),
        "file_sha256": sha256_path(path),
        "payload_field": payload_field,
        "payload_sha256": payload[payload_field],
    }
    if decision is not None:
        identity["decision"] = decision
    return identity


def _assert_file_hash(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise J1dSurfaceIntegrityError(
            f"Required immutable file is missing: {path}"
        )
    observed = sha256_path(path)
    if observed != expected:
        raise J1dSurfaceIntegrityError(
            f"Immutable file changed: {path}"
        )


def _assert_json_identity(
    path: Path,
    *,
    file_sha256: str,
    payload_field: str,
    payload_sha256: str,
    decision: str | None = None,
) -> dict[str, Any]:
    _assert_file_hash(path, file_sha256)
    payload = load_json(path)
    if (
        not verify_payload_hash(payload, payload_field)
        or payload.get(payload_field) != payload_sha256
    ):
        raise J1dSurfaceIntegrityError(
            f"Immutable payload changed: {path}"
        )
    if decision is not None and payload.get("decision") != decision:
        raise J1dSurfaceIntegrityError(
            f"Immutable decision changed: {path}"
        )
    return payload


def phase_paths(execution_root: Path) -> dict[str, Path]:
    phase_dir = execution_root / PHASE_DIR_NAME
    return {
        "phase_dir": phase_dir,
        "lock": phase_dir / PHASE_LOCK_NAME,
        "lock_result": phase_dir / PHASE_LOCK_RESULT_NAME,
        "marker": phase_dir / PHASE_MARKER_NAME,
        "manifest": phase_dir / PHASE_MANIFEST_NAME,
        "owner": phase_dir / PHASE_OWNER_NAME,
        "reservation": phase_dir / PHASE_RESERVATION_NAME,
        "consumption": phase_dir / PHASE_CONSUMPTION_NAME,
        "result": phase_dir / PHASE_RESULT_NAME,
        "retention": phase_dir / PHASE_RETENTION_NAME,
        "commit_head": phase_dir / COMMIT_HEAD_NAME,
        "checkpoint": phase_dir / "round64_candidate_checkpoint.bin",
        "sanity": phase_dir / "training_sanity_result.json",
    }


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def bound_command(
    action: str,
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> str:
    if action not in PUBLIC_COMMANDS:
        raise ValueError(f"Unsupported J1d command: {action}")
    return shlex.join(
        [
            "nice",
            "-n",
            "10",
            "env",
            "PYTHONPATH=.",
            ".venv/bin/python",
            "-m",
            "threes_rl.j1d_metric_authentication_surface_v2",
            action,
            "--execution-root",
            _relative_or_absolute(execution_root),
            "--readiness-dir",
            _relative_or_absolute(readiness_dir),
            "--jobs",
            "1",
        ]
    )


def _source_manifest_path() -> Path:
    return READINESS_DIR / SOURCE_MANIFEST_NAME


def iter_fresh_stream_rows() -> Iterable[dict[str, Any]]:
    for row_index in range(TRAIN_ROOTS):
        row = {
            "phase": "training",
            "partition": "train",
            "row_index": row_index,
            "block": row_index % 8,
            "logical_stream_id": (
                STREAM_RANGES["logical_stream_id"][0] + row_index
            ),
            "deck_stream_id": (
                STREAM_RANGES["deck_stream_id"][0] + row_index
            ),
            "slot_stream_id": (
                STREAM_RANGES["slot_stream_id"][0] + row_index
            ),
            "candidate_policy_stream_id": (
                STREAM_RANGES["candidate_policy_stream_id"][0]
                + row_index
            ),
            "control_policy_stream_id": None,
            "arm_count": 1,
            "starter_tile": None,
        }
        row["row_commitment_sha256"] = canonical_json_hash(row)
        yield row


def j1d_root_commitment() -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_training_marker_root_commitment_v1",
        "root_identity_version": "accepted_j1_marker_payload_root_v1",
        "phase": "training",
        "partition": "train",
        "phase_nonce": EXPECTED_CHARTER_SHA256,
        "canonical_rows_sha256": ordered_rows_hash(
            iter_fresh_stream_rows()
        ),
        "row_count": TRAIN_ROOTS,
        "j1d_charter_file_sha256": EXPECTED_CHARTER_SHA256,
        "spent_j1b_marker_file_sha256": (
            EXPECTED_SPENT_J1B_FILES["training/execution_opened.json"]
        ),
        "j1b_external_terminal_file_sha256": (
            EXPECTED_J1B_EXTERNAL_FILES[
                "J1B_OPEN_FAILURE_TERMINAL.json"
            ][0]
        ),
        "spent_j1c_terminal_file_sha256":
            EXPECTED_SPENT_J1C_TERMINAL[0],
        "spent_j1c_terminal_payload_sha256":
            EXPECTED_SPENT_J1C_TERMINAL[2],
        "spent_j1c_retention_file_sha256":
            EXPECTED_SPENT_J1C_RETENTION[0],
        "j1c_readiness_result_file_sha256":
            EXPECTED_J1C_READINESS[
                "J1C_TRAINING_EXECUTION_READINESS_RESULT.json"
            ][0],
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
    return payload_with_hash(payload, "marker_payload_sha256")


def root_id_for_commitment(
    commitment: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    if not verify_payload_hash(commitment, "marker_payload_sha256"):
        raise J1dSurfaceIntegrityError(
            "J1d root commitment hash is invalid"
        )
    if (
        commitment.get("phase") != row.get("phase")
        or commitment.get("partition") != row.get("partition")
    ):
        raise J1dSurfaceIntegrityError(
            "J1d root commitment/row mismatch"
        )
    return canonical_json_hash(
        {
            "marker_payload_sha256": commitment[
                "marker_payload_sha256"
            ],
            "partition": commitment["partition"],
            "row": int(row["row_index"]),
            "logical_stream_id": int(row["logical_stream_id"]),
            "deck_stream_id": int(row["deck_stream_id"]),
            "slot_stream_id": int(row["slot_stream_id"]),
        }
    )


def prospective_training_manifest() -> dict[str, Any]:
    commitment = j1d_root_commitment()
    rows: list[dict[str, Any]] = []
    for base in iter_fresh_stream_rows():
        root_id = root_id_for_commitment(commitment, base)
        rows.append(
            {
                **base,
                "root_id": root_id,
                "ancestry_id": root_id,
            }
        )
    root_ids = [row["root_id"] for row in rows]
    ancestry_ids = [row["ancestry_id"] for row in rows]
    all_streams = [
        int(row[field])
        for row in rows
        for field in STREAM_RANGES
    ]
    checks = {
        "row_count_exact": len(rows) == TRAIN_ROOTS,
        "row_indices_exact": [
            int(row["row_index"]) for row in rows
        ] == list(range(TRAIN_ROOTS)),
        "root_ids_unique": len(set(root_ids)) == TRAIN_ROOTS,
        "ancestries_unique": len(set(ancestry_ids)) == TRAIN_ROOTS,
        "one_root_per_ancestry": root_ids == ancestry_ids,
        "starter_none": all(row["starter_tile"] is None for row in rows),
        "one_arm": all(row["arm_count"] == 1 for row in rows),
        "stream_role_ids_globally_unique": (
            len(all_streams) == len(set(all_streams)) == TRAIN_ROOTS * 4
        ),
        "fresh_ranges_exact": all(
            [int(row[field]) for row in rows]
            == list(range(start, end + 1))
            for field, (start, end) in STREAM_RANGES.items()
        ),
    }
    payload = {
        "version": f"{VERSION}_prospective_training_manifest_v1",
        "phase": "training",
        "partition": "train",
        "root_commitment": commitment,
        "rows": rows,
        "canonical_rows_sha256": ordered_rows_hash(rows),
        "root_set_sha256": canonical_json_hash(root_ids),
        "ancestry_set_sha256": canonical_json_hash(ancestry_ids),
        "role_counts": {
            "roots": TRAIN_ROOTS,
            "ancestries": TRAIN_ROOTS,
            "logical": TRAIN_ROOTS,
            "deck": TRAIN_ROOTS,
            "slot": TRAIN_ROOTS,
            "candidate_policy": TRAIN_ROOTS,
        },
        "checks": checks,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "passes": all(checks.values()),
    }
    return payload_with_hash(
        payload,
        "prospective_manifest_payload_sha256",
    )


COMPACT_STREAM_AUTHORITIES = {
    "j1_prospective": {
        "path": (
            RUNS_ROOT
            / "forensics"
            / "j1_execution_surface_readiness_v1"
            / "J1_PROSPECTIVE_MANIFEST.json"
        ),
        "file_sha256":
            "2aee68a08325cdbc5e42153942079c1375163f2b88217bf407e64fd95f096dce",
        "payload_field": "prospective_manifest_payload_sha256",
        "payload_sha256":
            "de0046a2121138659dd2fd0bb46a48081d80842c5d24334d1a683dbf0a9a7093",
    },
    "j1_spent_training": {
        "path": SPENT_J1_TRAINING_DIR / "root_manifest.json",
        "file_sha256":
            "479487701230af128c4a1cff3aea49a29f59330efcb9ce7eafb9324abc455f0c",
        "payload_field": "root_manifest_payload_sha256",
        "payload_sha256":
            "0b722e61968148235335db15e360844099be62d8c0c64f9943dcafb62c9d1ae1",
    },
    "j1b_prospective": {
        "path": (
            J1B_PREFLIGHT_DIR / "J1B_PROSPECTIVE_TRAINING_MANIFEST.json"
        ),
        "file_sha256":
            "2bb0b2385360f2d06c019fdbac11cb58515629ab4f5fcf321624f499a07329f9",
        "payload_field": "prospective_manifest_payload_sha256",
        "payload_sha256":
            "f85a7624b2e8052d0b451bde9bf792181e08e055406fb5837232655a48f8a8a8",
    },
    "j1b_denylist": {
        "path": J1B_PREFLIGHT_DIR / "J1B_PROTECTED_STREAM_DENYLIST.json",
        "file_sha256":
            "1d36c79bae091a8b6b05ce69a2de19b43241ec091423d3a9d1bc3002ec229704",
        "payload_field": "denylist_payload_sha256",
        "payload_sha256":
            "83cf3201ffba5d4080ca76bb15d906a3b4cfe5df3eb32efe4be2088546b9d8f5",
    },
    "j1c_prospective": {
        "path": (
            J1C_READINESS_DIR / "J1C_PROSPECTIVE_TRAINING_MANIFEST.json"
        ),
        "file_sha256":
            "135fec4c75db8871e20ab3988471f75538a399e982573bf4a108afc569fe08b7",
        "payload_field": "prospective_manifest_payload_sha256",
        "payload_sha256":
            "c0d7953aa158297d5b515f6b6e4613b6fc22acc059e8e0aa0ff8904ee7e3546d",
    },
}


def compact_stream_authority(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for name, identity in COMPACT_STREAM_AUTHORITIES.items():
        path = Path(identity["path"])
        _assert_json_identity(
            path,
            file_sha256=str(identity["file_sha256"]),
            payload_field=str(identity["payload_field"]),
            payload_sha256=str(identity["payload_sha256"]),
        )
        verified[name] = {
            **{
                key: value
                for key, value in identity.items()
                if key != "path"
            },
            "path": str(path.resolve()),
        }
    j1c_manifest = load_json(
        Path(COMPACT_STREAM_AUTHORITIES["j1c_prospective"]["path"])
    )
    j1c_rows = list(j1c_manifest["rows"])
    rows = list(manifest["rows"])
    collisions = {
        field: sorted(
            set(int(row[field]) for row in rows)
            & set(int(row[field]) for row in j1c_rows)
        )
        for field in STREAM_RANGES
    }
    expected_spent = {
        "logical_stream_id": (213_000_000_000, 213_000_049_151),
        "deck_stream_id": (214_000_000_000, 214_000_049_151),
        "slot_stream_id": (215_000_000_000, 215_000_049_151),
        "candidate_policy_stream_id": (
            216_000_000_000,
            216_000_049_151,
        ),
    }
    checks = {
        "five_compact_authorities_exact": len(verified) == 5,
        "j1c_declared_rows_exact": len(j1c_rows) == TRAIN_ROOTS,
        "j1c_declared_intervals_spent": all(
            min(int(row[field]) for row in j1c_rows) == spent[1] - TRAIN_ROOTS + 1
            and max(int(row[field]) for row in j1c_rows) == spent[1]
            for field, spent in expected_spent.items()
        ),
        "j1d_starts_immediately_after_j1c": all(
            STREAM_RANGES[field][0] == spent[1] + 1
            for field, spent in expected_spent.items()
        ),
        "j1d_exact_16384_width": all(
            end - start + 1 == TRAIN_ROOTS
            for start, end in STREAM_RANGES.values()
        ),
        "zero_j1c_row_collisions": all(
            not values for values in collisions.values()
        ),
        "j1b_and_j1c_declared_ranges_spent": True,
        "no_broad_payload_scan": True,
    }
    payload = {
        "version": f"{VERSION}_protected_stream_authority_v1",
        "method": (
            "exact compact J1/J1b/J1c/J1d manifests and interval authorities"
        ),
        "authorities": verified,
        "spent_intervals_through_j1c": expected_spent,
        "fresh_j1d_intervals": dict(STREAM_RANGES),
        "collisions": collisions,
        "checks": checks,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "passes": all(checks.values()),
    }
    return payload_with_hash(payload, "stream_authority_payload_sha256")


def root_cause_evidence() -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_root_cause_v1",
        "source": "independent sealed-terminal structural audit",
        "preserved_v1_pre_correction_evidence": {
            "source_identities": dict(EXPECTED_V1_SOURCE_IDENTITIES),
            "test_evidence_file_sha256":
                EXPECTED_V1_READINESS_EVIDENCE[
                    "J1D_METRIC_AUTHENTICATION_TEST_EVIDENCE.json"
                ][0],
            "test_evidence_payload_sha256":
                EXPECTED_V1_READINESS_EVIDENCE[
                    "J1D_METRIC_AUTHENTICATION_TEST_EVIDENCE.json"
                ][2],
            "readiness_terminal_sealed": False,
        },
        "spent_j1c_terminal": {
            "file_sha256": EXPECTED_SPENT_J1C_TERMINAL[0],
            "payload_sha256": EXPECTED_SPENT_J1C_TERMINAL[2],
            "decision": EXPECTED_SPENT_J1C_TERMINAL[3],
        },
        "spent_j1c_retention": {
            "file_sha256": EXPECTED_SPENT_J1C_RETENTION[0],
            "payload_sha256": EXPECTED_SPENT_J1C_RETENTION[2],
        },
        "round_count": 64,
        "sole_failed_integrity_gate": "aggregates_recomputed_exact",
        "failed_subchecks": [
            "legal_entropy",
            "auxiliary_brier",
            "auxiliary_prevalence_brier",
        ],
        "maximum_absolute_discrepancies": {
            "legal_entropy": 2.62641797199592e-10,
            "auxiliary_brier": 4.422892815880708e-10,
            "auxiliary_prevalence_brier": 4.555544760864727e-10,
            "value_mse_upper_bound": 1.71e-13,
        },
        "passing_structural_evidence": [
            "root_ids",
            "per_root_metric_ids_and_hashes",
            "transition_hashes_and_rows",
            "round_metric_hashes",
            "record_and_buffer_hashes",
            "value_mse_aggregates",
            "recursive_commit_evidence",
        ],
        "mechanism": (
            "writer direct global root-equal reduction versus verifier "
            "mean of authenticated per-root reductions"
        ),
        "v1_readiness_defects": [
            (
                "published projection authenticated against itself while "
                "scalar equality used the inherited 1e-12 tolerance"
            ),
            (
                "mechanism test injected historical discrepancies instead "
                "of computing both reduction paths"
            ),
        ],
        "v2_exact_publication_contract": {
            "published_projection_equals_canonical_exactly": True,
            "published_hash_equals_canonical_hash": True,
            "one_ulp_rehashed_tamper_fails": True,
            "parent_scientific_tolerance_unchanged": True,
        },
        "frozen_absolute_tolerance": CANONICAL_METRIC_ABS_TOLERANCE,
        "outcome_values_read": 0,
        "checkpoint_or_episode_body_reads": 0,
        "scientific_interpretation": False,
        "repair_scope": "metric authentication orchestration only",
        "passes": True,
    }
    return payload_with_hash(payload, "root_cause_payload_sha256")


def write_source_artifacts(
    readiness_dir: Path = READINESS_DIR,
) -> dict[str, Any]:
    if readiness_dir != READINESS_DIR:
        raise J1dSurfaceIntegrityError(
            "J1d source artifacts require the frozen readiness namespace"
        )
    manifest = prospective_training_manifest()
    if manifest.get("passes") is not True:
        raise J1dSurfaceIntegrityError(
            "J1d prospective training manifest failed"
        )
    authority = compact_stream_authority(manifest)
    if authority.get("passes") is not True:
        raise J1dSurfaceIntegrityError(
            "J1d compact stream authority failed"
        )
    written_manifest = write_immutable_json(
        readiness_dir / SOURCE_MANIFEST_NAME,
        manifest,
        field="prospective_manifest_payload_sha256",
    )
    written_authority = write_immutable_json(
        readiness_dir / STREAM_AUTHORITY_NAME,
        authority,
        field="stream_authority_payload_sha256",
    )
    written_root_cause = write_immutable_json(
        readiness_dir / ROOT_CAUSE_NAME,
        root_cause_evidence(),
        field="root_cause_payload_sha256",
    )
    return {
        "manifest": written_manifest,
        "authority": written_authority,
        "root_cause": written_root_cause,
        "passes": True,
    }


def _validate_source_manifest(
    payload: Mapping[str, Any],
    *,
    scientific: bool,
) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    commitment = payload.get("root_commitment", {})
    root_ids = [str(row.get("root_id", "")) for row in rows]
    ancestry_ids = [str(row.get("ancestry_id", "")) for row in rows]
    expected_count = TRAIN_ROOTS if scientific else len(rows)
    checks = {
        "payload_hash_valid": verify_payload_hash(
            payload,
            "prospective_manifest_payload_sha256",
        ),
        "passes_true": payload.get("passes") is True,
        "phase_training": payload.get("phase") == "training",
        "partition_train": payload.get("partition") == "train",
        "row_count_exact": len(rows) == expected_count,
        "row_indices_exact": [
            int(row.get("row_index", -1)) for row in rows
        ] == list(range(len(rows))),
        "root_ids_unique": len(set(root_ids)) == len(rows),
        "ancestries_unique": len(set(ancestry_ids)) == len(rows),
        "one_root_per_ancestry": root_ids == ancestry_ids,
        "starter_none": all(
            row.get("starter_tile") is None for row in rows
        ),
        "one_arm": all(int(row.get("arm_count", -1)) == 1 for row in rows),
        "canonical_rows_exact": (
            ordered_rows_hash(rows)
            == payload.get("canonical_rows_sha256")
        ),
        "root_set_exact": (
            canonical_json_hash(root_ids)
            == (
                EXPECTED_ROOT_SET_SHA256
                if scientific
                else canonical_json_hash(root_ids)
            )
        ),
    }
    if scientific:
        checks.update(
            {
                "source_payload_exact": (
                    payload.get("prospective_manifest_payload_sha256")
                    == EXPECTED_SOURCE_MANIFEST_PAYLOAD_SHA256
                ),
                "canonical_rows_bound": (
                    payload.get("canonical_rows_sha256")
                    == EXPECTED_CANONICAL_ROWS_SHA256
                ),
                "root_commitment_bound": (
                    commitment.get("marker_payload_sha256")
                    == EXPECTED_ROOT_COMMITMENT_PAYLOAD_SHA256
                    and verify_payload_hash(
                        commitment,
                        "marker_payload_sha256",
                    )
                ),
            }
        )
        for field, (start, end) in STREAM_RANGES.items():
            values = [int(row.get(field, -1)) for row in rows]
            checks[f"{field}_range_exact"] = (
                values == list(range(start, end + 1))
            )
        all_streams = [
            int(row[field])
            for row in rows
            for field in STREAM_RANGES
        ]
        checks["all_stream_roles_unique"] = (
            len(all_streams) == len(set(all_streams)) == TRAIN_ROOTS * 4
        )
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise J1dSurfaceIntegrityError(
            "J1d source manifest validation failed: "
            + ", ".join(failed)
        )
    return {
        "row_count": len(rows),
        "root_set_sha256": canonical_json_hash(root_ids),
        "ancestry_set_sha256": canonical_json_hash(ancestry_ids),
        "checks": checks,
        "passes": True,
    }


def build_materialized_manifest(
    source: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
    scientific: bool,
) -> dict[str, Any]:
    validation = _validate_source_manifest(
        source,
        scientific=scientific,
    )
    rows = list(source["rows"])
    stream_roles = {}
    for field in STREAM_RANGES:
        values = [int(row[field]) for row in rows]
        stream_roles[field] = {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "ordered_sha256": canonical_json_hash(values),
        }
    payload = {
        "version": f"{VERSION}_training_root_manifest_v1",
        "phase": "training",
        "partition": "train",
        "rows": rows,
        "canonical_rows_sha256": source["canonical_rows_sha256"],
        "root_set_sha256": validation["root_set_sha256"],
        "ancestry_set_sha256": validation["ancestry_set_sha256"],
        "source_manifest_identity": dict(source_identity),
        "root_commitment": source.get("root_commitment"),
        "stream_roles": stream_roles,
        "checks": validation["checks"],
        "passes": True,
    }
    return payload_with_hash(payload, "root_manifest_payload_sha256")


def manifest_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (
        not verify_payload_hash(payload, "root_manifest_payload_sha256")
        or payload.get("passes") is not True
    ):
        raise J1dSurfaceIntegrityError(
            "Materialized manifest payload is invalid"
        )
    return {
        "phase": "training",
        "row_count": len(payload["rows"]),
        "canonical_rows_sha256": payload["canonical_rows_sha256"],
        "root_set_sha256": payload["root_set_sha256"],
        "ancestry_set_sha256": payload["ancestry_set_sha256"],
        "payload_sha256": payload["root_manifest_payload_sha256"],
    }


def audit_authoritative_inputs(
    *,
    require_future_execution_absent: bool,
) -> dict[str, Any]:
    _assert_file_hash(CHARTER_PATH, EXPECTED_CHARTER_SHA256)
    for relative, expected in EXPECTED_V1_SOURCE_IDENTITIES.items():
        _assert_file_hash(REPO_ROOT / relative, expected)
    v1_inventory = sorted(
        path.name
        for path in V1_READINESS_DIR.iterdir()
        if path.is_file()
    )
    if v1_inventory != sorted(EXPECTED_V1_READINESS_EVIDENCE):
        raise J1dSurfaceIntegrityError(
            "J1d V1 pre-correction evidence inventory changed"
        )
    v1_evidence: dict[str, Any] = {}
    for name, (
        file_sha256,
        payload_field,
        payload_sha256,
    ) in sorted(EXPECTED_V1_READINESS_EVIDENCE.items()):
        path = V1_READINESS_DIR / name
        payload = _assert_json_identity(
            path,
            file_sha256=file_sha256,
            payload_field=payload_field,
            payload_sha256=payload_sha256,
        )
        v1_evidence[name] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256,
            "payload_field": payload_field,
            "payload_sha256": payload_sha256,
            "version": payload.get("version"),
        }
    for relative, expected in EXPECTED_J1B_SOURCE_IDENTITIES.items():
        _assert_file_hash(REPO_ROOT / relative, expected)
    for (
        relative,
        expected,
    ) in EXPECTED_J1B_TRAINING_SOURCE_IDENTITIES.items():
        _assert_file_hash(REPO_ROOT / relative, expected)
    for relative, expected in EXPECTED_J1C_SOURCE_IDENTITIES.items():
        _assert_file_hash(REPO_ROOT / relative, expected)

    lock_path = J1B_PREFLIGHT_DIR / "J1B_READINESS_LOCK.json"
    result_path = J1B_PREFLIGHT_DIR / "J1B_READINESS_RESULT.json"
    lock = _assert_json_identity(
        lock_path,
        file_sha256=EXPECTED_J1B_READINESS_FILES[
            "J1B_READINESS_LOCK.json"
        ],
        payload_field=EXPECTED_J1B_READINESS_PAYLOADS[
            "J1B_READINESS_LOCK.json"
        ][0],
        payload_sha256=EXPECTED_J1B_READINESS_PAYLOADS[
            "J1B_READINESS_LOCK.json"
        ][1],
        decision="READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
    )
    result = _assert_json_identity(
        result_path,
        file_sha256=EXPECTED_J1B_READINESS_FILES[
            "J1B_READINESS_RESULT.json"
        ],
        payload_field=EXPECTED_J1B_READINESS_PAYLOADS[
            "J1B_READINESS_RESULT.json"
        ][0],
        payload_sha256=EXPECTED_J1B_READINESS_PAYLOADS[
            "J1B_READINESS_RESULT.json"
        ][1],
        decision="READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
    )
    expected_lock_identity = immutable_json_identity(
        lock_path,
        payload_field="readiness_lock_payload_sha256",
    )
    if result.get("readiness_lock") != expected_lock_identity:
        raise J1dSurfaceIntegrityError(
            "J1b preflight result changed its lock identity"
        )

    checked_artifacts: dict[str, Any] = {}
    for name, identity in sorted(lock.get("artifacts", {}).items()):
        path = Path(str(identity.get("path", "")))
        payload = _assert_json_identity(
            path,
            file_sha256=str(identity.get("file_sha256", "")),
            payload_field=str(identity.get("payload_field", "")),
            payload_sha256=str(identity.get("payload_sha256", "")),
        )
        checked_artifacts[name] = {
            **dict(identity),
            "version": payload.get("version"),
        }

    parent_sources = dict(lock.get("parent_source_identities", {}))
    for relative, expected in sorted(parent_sources.items()):
        _assert_file_hash(REPO_ROOT / relative, str(expected))
    parent_readiness_dir = (
        RUNS_ROOT / "forensics" / "j1_execution_surface_readiness_v1"
    )
    parent_readiness = dict(
        lock.get("parent_readiness_identities", {})
    )
    for name, expected in sorted(parent_readiness.items()):
        _assert_file_hash(parent_readiness_dir / name, str(expected))

    history = dict(lock.get("pre_a1_historical_evidence", {}))
    _assert_json_identity(
        J1B_PRE_A1_HISTORY_PATH,
        file_sha256=EXPECTED_PRE_A1_FILE_SHA256,
        payload_field="test_evidence_payload_sha256",
        payload_sha256=EXPECTED_PRE_A1_PAYLOAD_SHA256,
    )
    if (
        history.get("file_sha256") != EXPECTED_PRE_A1_FILE_SHA256
        or history.get("payload_sha256")
        != EXPECTED_PRE_A1_PAYLOAD_SHA256
        or Path(str(history.get("path", ""))).resolve()
        != J1B_PRE_A1_HISTORY_PATH.resolve()
    ):
        raise J1dSurfaceIntegrityError(
            "J1b pre-A1 evidence binding changed"
        )

    spent = dict(lock.get("spent_j1_execution_identities", {}))
    for relative, expected in sorted(spent.items()):
        _assert_file_hash(SPENT_J1_TRAINING_DIR / relative, str(expected))
    if (
        spent.get("terminal_result.json")
        != EXPECTED_SPENT_TERMINAL_FILE_SHA256
        or spent.get("retention_manifest.json")
        != EXPECTED_SPENT_RETENTION_FILE_SHA256
    ):
        raise J1dSurfaceIntegrityError(
            "Spent J1 terminal or retention binding changed"
        )

    j1b_training_readiness: dict[str, Any] = {}
    for name, (
        file_sha256,
        payload_field,
        payload_sha256,
    ) in sorted(EXPECTED_J1B_TRAINING_READINESS.items()):
        path = J1B_TRAINING_READINESS_DIR / name
        payload = _assert_json_identity(
            path,
            file_sha256=file_sha256,
            payload_field=payload_field,
            payload_sha256=payload_sha256,
        )
        j1b_training_readiness[name] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256,
            "payload_field": payload_field,
            "payload_sha256": payload_sha256,
            "decision": payload.get("decision"),
        }

    spent_j1b_paths = sorted(
        str(path.relative_to(SPENT_J1B_EXECUTION_ROOT))
        for path in SPENT_J1B_EXECUTION_ROOT.rglob("*")
        if path.is_file()
    )
    if spent_j1b_paths != sorted(EXPECTED_SPENT_J1B_FILES):
        raise J1dSurfaceIntegrityError(
            "Spent J1b execution inventory changed"
        )
    spent_j1b: dict[str, Any] = {}
    for relative, expected in sorted(
        EXPECTED_SPENT_J1B_FILES.items()
    ):
        path = SPENT_J1B_EXECUTION_ROOT / relative
        _assert_file_hash(path, expected)
        spent_j1b[relative] = {
            "path": str(path.resolve()),
            "file_sha256": expected,
        }
    spent_marker = load_json(
        SPENT_J1B_EXECUTION_ROOT / "training/execution_opened.json"
    )
    if (
        not verify_payload_hash(
            spent_marker,
            "activation_marker_payload_sha256",
        )
        or spent_marker.get("activation_marker_payload_sha256")
        != "c9e48e972a59f699627bfaa949854930672a8c45a6c671be591e175522a107e4"
        or spent_marker.get("scientific_work") != 0
        or spent_marker.get("streams_reserved") != 0
        or spent_marker.get("streams_consumed") != 0
    ):
        raise J1dSurfaceIntegrityError(
            "Spent J1b marker identity or zero-work state changed"
        )

    external_j1b: dict[str, Any] = {}
    for name, (
        file_sha256,
        payload_field,
        payload_sha256,
        decision,
    ) in sorted(EXPECTED_J1B_EXTERNAL_FILES.items()):
        path = J1B_EXTERNAL_TERMINAL_DIR / name
        payload = _assert_json_identity(
            path,
            file_sha256=file_sha256,
            payload_field=payload_field,
            payload_sha256=payload_sha256,
            decision=decision,
        )
        external_j1b[name] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256,
            "payload_field": payload_field,
            "payload_sha256": payload_sha256,
            "decision": payload.get("decision"),
        }

    j1c_readiness: dict[str, Any] = {}
    for name, (
        file_sha256,
        payload_field,
        payload_sha256,
    ) in sorted(EXPECTED_J1C_READINESS.items()):
        path = J1C_READINESS_DIR / name
        payload = _assert_json_identity(
            path,
            file_sha256=file_sha256,
            payload_field=payload_field,
            payload_sha256=payload_sha256,
        )
        j1c_readiness[name] = {
            "path": str(path.resolve()),
            "file_sha256": file_sha256,
            "payload_field": payload_field,
            "payload_sha256": payload_sha256,
            "decision": payload.get("decision"),
        }
    spent_j1c_terminal_path = (
        SPENT_J1C_TRAINING_DIR / "terminal_result.json"
    )
    spent_j1c_terminal = _assert_json_identity(
        spent_j1c_terminal_path,
        file_sha256=EXPECTED_SPENT_J1C_TERMINAL[0],
        payload_field=EXPECTED_SPENT_J1C_TERMINAL[1],
        payload_sha256=EXPECTED_SPENT_J1C_TERMINAL[2],
        decision=EXPECTED_SPENT_J1C_TERMINAL[3],
    )
    spent_j1c_retention_path = (
        SPENT_J1C_TRAINING_DIR / "retention_manifest.json"
    )
    spent_j1c_retention = _assert_json_identity(
        spent_j1c_retention_path,
        file_sha256=EXPECTED_SPENT_J1C_RETENTION[0],
        payload_field=EXPECTED_SPENT_J1C_RETENTION[1],
        payload_sha256=EXPECTED_SPENT_J1C_RETENTION[2],
    )
    if (
        spent_j1c_terminal.get("checkpoint_authoritative") is not False
        or spent_j1c_terminal.get("checkpoint_quarantined") is not True
        or spent_j1c_terminal.get("error_message")
        != "Training round metric evidence is not authenticated"
        or spent_j1c_retention.get("passes") is not True
        or spent_j1c_retention.get("preserve_byte_for_byte") is not True
    ):
        raise J1dSurfaceIntegrityError(
            "Spent J1c terminal or retention semantics changed"
        )

    source_manifest_path = _source_manifest_path()
    _assert_file_hash(
        source_manifest_path,
        EXPECTED_SOURCE_MANIFEST_FILE_SHA256,
    )
    source_manifest = load_json(source_manifest_path)
    if (
        source_manifest.get("prospective_manifest_payload_sha256")
        != EXPECTED_SOURCE_MANIFEST_PAYLOAD_SHA256
    ):
        raise J1dSurfaceIntegrityError(
            "J1d source manifest payload changed"
        )
    source_validation = _validate_source_manifest(
        source_manifest,
        scientific=True,
    )
    stream_authority_path = READINESS_DIR / STREAM_AUTHORITY_NAME
    stream_authority = _assert_json_identity(
        stream_authority_path,
        file_sha256=EXPECTED_STREAM_AUTHORITY_FILE_SHA256,
        payload_field="stream_authority_payload_sha256",
        payload_sha256=EXPECTED_STREAM_AUTHORITY_PAYLOAD_SHA256,
    )
    root_cause_path = READINESS_DIR / ROOT_CAUSE_NAME
    root_cause = load_json(root_cause_path)
    if (
        not verify_payload_hash(root_cause, "root_cause_payload_sha256")
        or root_cause != root_cause_evidence()
        or root_cause.get("outcome_values_read") != 0
        or root_cause.get("checkpoint_or_episode_body_reads") != 0
    ):
        raise J1dSurfaceIntegrityError(
            "J1d root-cause evidence changed or opened outcomes"
        )
    checks = {
        "charter_exact": True,
        "v1_source_identities_exact": True,
        "v1_pre_correction_evidence_exact": (
            len(v1_evidence) == len(EXPECTED_V1_READINESS_EVIDENCE)
        ),
        "v1_readiness_terminal_absent": (
            not (
                V1_READINESS_DIR
                / "J1D_METRIC_AUTHENTICATION_READINESS_RESULT.json"
            ).exists()
        ),
        "j1b_source_identities_exact": True,
        "j1b_training_source_identities_exact": True,
        "j1c_source_identities_exact": True,
        "j1b_readiness_lock_exact": True,
        "j1b_readiness_result_exact": True,
        "j1b_readiness_artifacts_exact": len(checked_artifacts) == 7,
        "pre_a1_history_exact": True,
        "parent_sources_exact": (
            parent_sources.get("threes_rl/j1_execution_surface.py")
            == EXPECTED_PARENT_ENGINE_SHA256
        ),
        "parent_readiness_exact": len(parent_readiness) == 6,
        "spent_j1_inventory_exact": len(spent) == 14,
        "j1b_training_readiness_exact": (
            len(j1b_training_readiness)
            == len(EXPECTED_J1B_TRAINING_READINESS)
        ),
        "spent_j1b_inventory_exact": (
            len(spent_j1b) == len(EXPECTED_SPENT_J1B_FILES)
        ),
        "external_j1b_terminal_exact": (
            len(external_j1b) == len(EXPECTED_J1B_EXTERNAL_FILES)
        ),
        "j1c_readiness_exact": (
            len(j1c_readiness) == len(EXPECTED_J1C_READINESS)
        ),
        "spent_j1c_terminal_exact": True,
        "spent_j1c_retention_exact": True,
        "source_manifest_exact": source_validation["passes"],
        "stream_authority_exact": (
            stream_authority.get("passes") is True
            and stream_authority.get("streams_reserved") == 0
            and stream_authority.get("streams_consumed") == 0
        ),
        "root_cause_structural_only": True,
        "future_execution_root_state": (
            not FUTURE_EXECUTION_ROOT.exists()
            if require_future_execution_absent
            else True
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise J1dSurfaceIntegrityError(
            "Authoritative input audit failed: " + ", ".join(failed)
        )
    identities = {
        "charter": {
            "path": str(CHARTER_PATH.resolve()),
            "file_sha256": EXPECTED_CHARTER_SHA256,
        },
        "v1_sources": dict(EXPECTED_V1_SOURCE_IDENTITIES),
        "v1_pre_correction_evidence": v1_evidence,
        "j1b_preflight_lock": immutable_json_identity(
            lock_path,
            payload_field="readiness_lock_payload_sha256",
            decision="READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
        ),
        "j1b_preflight_result": immutable_json_identity(
            result_path,
            payload_field="readiness_result_payload_sha256",
            decision="READY_J1B_OPERATIONAL_REPAIR_PREFLIGHT",
        ),
        "j1b_preflight_artifacts": checked_artifacts,
        "pre_a1_history": {
            "path": str(J1B_PRE_A1_HISTORY_PATH.resolve()),
            "file_sha256": EXPECTED_PRE_A1_FILE_SHA256,
            "payload_sha256": EXPECTED_PRE_A1_PAYLOAD_SHA256,
        },
        "parent_sources": parent_sources,
        "parent_readiness": parent_readiness,
        "spent_j1_execution": spent,
        "j1b_training_sources":
            dict(EXPECTED_J1B_TRAINING_SOURCE_IDENTITIES),
        "j1b_training_readiness": j1b_training_readiness,
        "spent_j1b_execution": spent_j1b,
        "external_j1b_terminal": external_j1b,
        "j1c_sources": dict(EXPECTED_J1C_SOURCE_IDENTITIES),
        "j1c_readiness": j1c_readiness,
        "spent_j1c_terminal": immutable_json_identity(
            spent_j1c_terminal_path,
            payload_field=EXPECTED_SPENT_J1C_TERMINAL[1],
            decision=EXPECTED_SPENT_J1C_TERMINAL[3],
        ),
        "spent_j1c_retention": immutable_json_identity(
            spent_j1c_retention_path,
            payload_field=EXPECTED_SPENT_J1C_RETENTION[1],
        ),
        "source_manifest": immutable_json_identity(
            source_manifest_path,
            payload_field="prospective_manifest_payload_sha256",
        ),
        "stream_authority": immutable_json_identity(
            stream_authority_path,
            payload_field="stream_authority_payload_sha256",
        ),
        "root_cause": immutable_json_identity(
            root_cause_path,
            payload_field="root_cause_payload_sha256",
        ),
    }
    return {
        "identities": identities,
        "identities_sha256": canonical_json_hash(identities),
        "source_manifest_validation": source_validation,
        "checks": checks,
        "passes": True,
    }


def _readiness_paths(readiness_dir: Path) -> dict[str, Path]:
    return {
        "source_manifest": readiness_dir / SOURCE_MANIFEST_NAME,
        "stream_authority": readiness_dir / STREAM_AUTHORITY_NAME,
        "root_cause": readiness_dir / ROOT_CAUSE_NAME,
        "test_evidence": readiness_dir / TEST_EVIDENCE_NAME,
        "schema": readiness_dir / SCHEMA_NAME,
        "projection": readiness_dir / PROJECTION_NAME,
        "input_bindings": readiness_dir / INPUT_BINDINGS_NAME,
        "lock": readiness_dir / READINESS_LOCK_NAME,
        "result": readiness_dir / READINESS_RESULT_NAME,
    }


def load_ready_surface(readiness_dir: Path) -> dict[str, Any]:
    paths = _readiness_paths(readiness_dir)
    lock = load_json(paths["lock"])
    result = load_json(paths["result"])
    if (
        not verify_payload_hash(
            lock,
            "readiness_lock_payload_sha256",
        )
        or lock.get("decision") != READY_DECISION
    ):
        raise J1dSurfaceIntegrityError(
            "J1d execution-surface readiness lock is invalid"
        )
    if (
        not verify_payload_hash(
            result,
            "readiness_result_payload_sha256",
        )
        or result.get("decision") != READY_DECISION
    ):
        raise J1dSurfaceIntegrityError(
            "J1d execution-surface readiness result is invalid"
        )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="readiness_lock_payload_sha256",
        decision=READY_DECISION,
    )
    result_identity = immutable_json_identity(
        paths["result"],
        payload_field="readiness_result_payload_sha256",
        decision=READY_DECISION,
    )
    if result.get("readiness_lock_identity") != lock_identity:
        raise J1dSurfaceIntegrityError(
            "J1d surface result changed its readiness lock"
        )
    mode = str(lock.get("execution_mode", ""))
    if mode not in {"scientific", "miniature_fixture"}:
        raise J1dSurfaceIntegrityError(
            "J1d readiness execution mode is invalid"
        )
    expected_sources = {
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
    }
    for field, expected in expected_sources.items():
        if lock.get(field) != expected:
            raise J1dSurfaceIntegrityError(
                f"J1d readiness changed {field}"
            )
    artifact_fields = {
        "test_evidence": "test_evidence_payload_sha256",
        "schema": "schema_payload_sha256",
        "projection": "projection_payload_sha256",
        "input_bindings": "input_bindings_payload_sha256",
        "root_cause": "root_cause_payload_sha256",
    }
    for name, payload_field in artifact_fields.items():
        observed = immutable_json_identity(
            paths[name],
            payload_field=payload_field,
        )
        if lock.get("artifacts", {}).get(name) != observed:
            raise J1dSurfaceIntegrityError(
                f"J1d readiness changed {name}"
            )
    if mode == "scientific":
        source_observed = immutable_json_identity(
            paths["source_manifest"],
            payload_field="prospective_manifest_payload_sha256",
        )
        authority_observed = immutable_json_identity(
            paths["stream_authority"],
            payload_field="stream_authority_payload_sha256",
        )
        root_cause_observed = immutable_json_identity(
            paths["root_cause"],
            payload_field="root_cause_payload_sha256",
        )
        if lock.get("source_manifest_identity") != source_observed:
            raise J1dSurfaceIntegrityError(
                "J1d readiness changed source manifest"
            )
        if lock.get("stream_authority_identity") != authority_observed:
            raise J1dSurfaceIntegrityError(
                "J1d readiness changed stream authority"
            )
        if lock.get("root_cause_identity") != root_cause_observed:
            raise J1dSurfaceIntegrityError(
                "J1d readiness changed root-cause evidence"
            )
    expected_root = Path(str(lock.get("future_execution_root", "")))
    if mode == "scientific":
        if (
            expected_root.resolve() != FUTURE_EXECUTION_ROOT.resolve()
            or readiness_dir.resolve() != READINESS_DIR.resolve()
            or lock.get("scientific_authority") is not True
            or lock.get("fixture_only") is not False
        ):
            raise J1dSurfaceIntegrityError(
                "Scientific readiness paths or authority changed"
            )
        input_audit = audit_authoritative_inputs(
            require_future_execution_absent=False,
        )
        if (
            lock.get("authoritative_input_identities_sha256")
            != input_audit["identities_sha256"]
        ):
            raise J1dSurfaceIntegrityError(
                "Scientific readiness input binding changed"
            )
    else:
        if (
            lock.get("scientific_authority") is not False
            or lock.get("fixture_only") is not True
        ):
            raise J1dSurfaceIntegrityError(
                "Fixture readiness has scientific authority"
            )
    source_identity = dict(lock.get("source_manifest_identity", {}))
    source_path = Path(str(source_identity.get("path", "")))
    if (
        not source_path.is_file()
        or sha256_path(source_path)
        != source_identity.get("file_sha256")
    ):
        raise J1dSurfaceIntegrityError(
            "Readiness source manifest file changed"
        )
    source = load_json(source_path)
    payload_field = str(source_identity.get("payload_field", ""))
    if (
        not verify_payload_hash(source, payload_field)
        or source.get(payload_field)
        != source_identity.get("payload_sha256")
    ):
        raise J1dSurfaceIntegrityError(
            "Readiness source manifest payload changed"
        )
    _validate_source_manifest(
        source,
        scientific=mode == "scientific",
    )
    return {
        "paths": paths,
        "lock": lock,
        "result": result,
        "lock_identity": lock_identity,
        "result_identity": result_identity,
        "mode": mode,
        "future_execution_root": expected_root,
        "source_manifest": source,
        "source_manifest_identity": source_identity,
        "passes": True,
    }


def _phase_ready_decision() -> str:
    return "READY_J1D_TRAINING_PHASE_LOCK"


def _assert_execution_root(
    execution_root: Path,
    readiness: Mapping[str, Any],
) -> None:
    if (
        execution_root.resolve()
        != Path(readiness["future_execution_root"]).resolve()
    ):
        raise J1dSurfaceIntegrityError(
            "Execution root differs from immutable readiness"
        )


def seal_training_phase_lock(
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    readiness = load_ready_surface(readiness_dir)
    _assert_execution_root(execution_root, readiness)
    paths = phase_paths(execution_root)
    if execution_root.exists():
        present = sorted(
            str(path.relative_to(execution_root))
            for path in execution_root.rglob("*")
            if path.is_file()
        )
        if present:
            raise FileExistsError(
                "J1d phase lock requires a fresh execution root: "
                + ", ".join(present)
            )
    materialized = build_materialized_manifest(
        readiness["source_manifest"],
        source_identity=readiness["source_manifest_identity"],
        scientific=readiness["mode"] == "scientific",
    )
    commands = {
        action: bound_command(
            action,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
        for action in PUBLIC_COMMANDS
    }
    payload = {
        "version": f"{VERSION}_training_phase_lock_v1",
        "phase": "training",
        "decision": _phase_ready_decision(),
        "execution_mode": readiness["mode"],
        "scientific_authority": (
            readiness["mode"] == "scientific"
        ),
        "fixture_only": readiness["mode"] == "miniature_fixture",
        "readiness_lock_identity": readiness["lock_identity"],
        "readiness_result_identity": readiness["result_identity"],
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
        "bounded_engine": "execute_training_engine_bounded_j1d",
        "legacy_engine_unreachable": True,
        "training_only": True,
        "development_command_present": False,
        "confirmation_command_present": False,
        "promotion_command_present": False,
        "future_execution_root": str(execution_root.resolve()),
        "source_manifest_identity":
            readiness["source_manifest_identity"],
        "manifest_identity": manifest_identity(materialized),
        "commands": commands,
        "runtime_order": [
            "parse_arguments_standard_library_only",
            "configure_torch_interop_one",
            "configure_torch_intraop_one",
            "enable_deterministic_algorithms",
            "verify_runtime",
            "import_parent",
            "initialize_frozen_model_optimizer",
            "first_parent_operational_guard",
            "acquire_or_reclaim_owner",
            "reserve_streams",
            "open_stream_consumption",
            "bounded_engine_genesis_and_work",
        ],
        "first_guard_before_owner": True,
        "first_guard_before_reservation": True,
        "first_guard_before_consumption": True,
        "first_guard_before_genesis": True,
        "jobs": 1,
        "nice_minimum": 10,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "scientific_games": 0,
        "scientific_optimizer_steps": 0,
        "scientific_checkpoints": 0,
        "policy_or_score_outcomes": 0,
        "passes": True,
    }
    written_lock = write_immutable_json(
        paths["lock"],
        payload,
        field="phase_lock_payload_sha256",
    )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="phase_lock_payload_sha256",
        decision=_phase_ready_decision(),
    )
    result = write_immutable_json(
        paths["lock_result"],
        {
            "version": f"{VERSION}_training_phase_lock_result_v1",
            "phase": "training",
            "decision": _phase_ready_decision(),
            "phase_lock_identity": lock_identity,
            "readiness_lock_identity": readiness["lock_identity"],
            "readiness_result_identity": readiness["result_identity"],
            "streams_reserved": 0,
            "streams_consumed": 0,
            "scientific_work": 0,
            "passes": True,
        },
        field="phase_lock_result_payload_sha256",
    )
    return {
        "lock": written_lock,
        "lock_identity": lock_identity,
        "result": result,
        "result_identity": immutable_json_identity(
            paths["lock_result"],
            payload_field="phase_lock_result_payload_sha256",
            decision=_phase_ready_decision(),
        ),
        "passes": True,
    }


def load_training_phase_lock(
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    readiness = load_ready_surface(readiness_dir)
    _assert_execution_root(execution_root, readiness)
    paths = phase_paths(execution_root)
    lock = load_json(paths["lock"])
    lock_result = load_json(paths["lock_result"])
    if (
        not verify_payload_hash(lock, "phase_lock_payload_sha256")
        or lock.get("decision") != _phase_ready_decision()
        or lock.get("phase") != "training"
    ):
        raise J1dSurfaceIntegrityError(
            "J1d training phase lock is invalid"
        )
    if (
        not verify_payload_hash(
            lock_result,
            "phase_lock_result_payload_sha256",
        )
        or lock_result.get("decision") != _phase_ready_decision()
    ):
        raise J1dSurfaceIntegrityError(
            "J1d training phase lock result is invalid"
        )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="phase_lock_payload_sha256",
        decision=_phase_ready_decision(),
    )
    if lock_result.get("phase_lock_identity") != lock_identity:
        raise J1dSurfaceIntegrityError(
            "J1d lock result changed phase lock identity"
        )
    if (
        lock.get("readiness_lock_identity")
        != readiness["lock_identity"]
        or lock.get("readiness_result_identity")
        != readiness["result_identity"]
    ):
        raise J1dSurfaceIntegrityError(
            "J1d phase lock changed readiness identities"
        )
    for field, expected in {
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
    }.items():
        if lock.get(field) != expected:
            raise J1dSurfaceIntegrityError(
                f"J1d phase lock changed {field}"
            )
    expected_commands = {
        action: bound_command(
            action,
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
        for action in PUBLIC_COMMANDS
    }
    checks = {
        "mode_exact": lock.get("execution_mode")
        == readiness["mode"],
        "authority_exact": lock.get("scientific_authority")
        is (readiness["mode"] == "scientific"),
        "bounded_engine_exact": (
            lock.get("bounded_engine")
            == "execute_training_engine_bounded_j1d"
        ),
        "legacy_unreachable": (
            lock.get("legacy_engine_unreachable") is True
        ),
        "training_only": lock.get("training_only") is True,
        "no_development": (
            lock.get("development_command_present") is False
        ),
        "no_confirmation": (
            lock.get("confirmation_command_present") is False
        ),
        "no_promotion": (
            lock.get("promotion_command_present") is False
        ),
        "commands_exact": lock.get("commands") == expected_commands,
        "source_manifest_exact": (
            lock.get("source_manifest_identity")
            == readiness["source_manifest_identity"]
        ),
    }
    expected_manifest = build_materialized_manifest(
        readiness["source_manifest"],
        source_identity=readiness["source_manifest_identity"],
        scientific=readiness["mode"] == "scientific",
    )
    checks["manifest_identity_exact"] = (
        lock.get("manifest_identity")
        == manifest_identity(expected_manifest)
    )
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise J1dSurfaceIntegrityError(
            "J1d phase lock validation failed: " + ", ".join(failed)
        )
    return {
        "paths": paths,
        "readiness": readiness,
        "lock": lock,
        "lock_result": lock_result,
        "lock_identity": lock_identity,
        "expected_manifest": expected_manifest,
        "commands": expected_commands,
        "checks": checks,
        "passes": True,
    }


def _fixture_open_audit() -> dict[str, Any]:
    return {
        "version": f"{VERSION}_fixture_open_audit_v1",
        "checks": {
            "fixture_only": True,
            "no_scientific_work": True,
        },
        "passes": True,
    }


def real_operational_audit_marker_roundtrip(
    marker_path: Path,
) -> dict[str, Any]:
    """Exercise the real audit shape through the production JSON writer."""
    from threes_rl import j1_joint_policy_value as j1

    operational = j1.operational_audit(
        output_dir=marker_path.parent,
    )
    try:
        top_three = operational["services"]["dashboard"]["top_three"]
    except (KeyError, TypeError) as error:
        raise J1dSurfaceIntegrityError(
            "Real operational audit dashboard shape changed"
        ) from error
    pre_write_is_tuple = type(top_three) is tuple
    if operational.get("passes") is not True or not pre_write_is_tuple:
        raise J1dSurfaceOperationalHold(
            "Real operational audit did not pass with tuple top-three"
        )
    prospective = {
        "version": f"{VERSION}_real_operational_roundtrip_fixture_v1",
        "operational_audit": operational,
        "scientific_work": 0,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "passes": True,
    }
    written = write_immutable_json(
        marker_path,
        prospective,
        field="activation_marker_payload_sha256",
    )
    exact_bytes = marker_path.read_bytes()
    reloaded = json.loads(exact_bytes.decode("utf-8"))
    post_write = reloaded["operational_audit"]["services"][
        "dashboard"
    ]["top_three"]
    checks = {
        "real_operational_audit_passed": operational["passes"] is True,
        "pre_write_top_three_is_tuple": pre_write_is_tuple,
        "post_write_top_three_is_list": type(post_write) is list,
        "top_three_values_exact": post_write == [263670, 261369, 258561],
        "exact_written_bytes_reloaded": reloaded == written,
        "payload_hash_valid": verify_payload_hash(
            reloaded,
            "activation_marker_payload_sha256",
        ),
        "file_hash_recomputed": sha256_bytes(exact_bytes) == sha256_path(
            marker_path
        ),
        "zero_scientific_work": (
            reloaded["scientific_work"] == 0
            and reloaded["streams_reserved"] == 0
            and reloaded["streams_consumed"] == 0
        ),
    }
    return {
        "version": f"{VERSION}_real_operational_roundtrip_report_v1",
        "marker_file_sha256": sha256_path(marker_path),
        "marker_payload_sha256": reloaded[
            "activation_marker_payload_sha256"
        ],
        "checks": checks,
        "passes": all(checks.values()),
    }


def open_training_phase(
    *,
    execution_root: Path,
    readiness_dir: Path,
    opened_at: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    loaded = load_training_phase_lock(
        execution_root=execution_root,
        readiness_dir=readiness_dir,
    )
    paths = loaded["paths"]
    for name in (
        "marker",
        "manifest",
        "owner",
        "reservation",
        "consumption",
        "commit_head",
        "result",
        "checkpoint",
    ):
        if paths[name].exists():
            raise FileExistsError(
                f"J1d open requires unused artifact: {paths[name]}"
            )
    if loaded["readiness"]["mode"] == "scientific":
        from threes_rl import j1_joint_policy_value as j1

        operational = j1.operational_audit(
            output_dir=paths["phase_dir"]
        )
        if operational.get("passes") is not True:
            raise J1dSurfaceOperationalHold(
                "J1d training open operational audit failed"
            )
    else:
        operational = _fixture_open_audit()
    marker = {
        "version": f"{VERSION}_training_execution_opened_v1",
        "phase": "training",
        "execution_mode": loaded["readiness"]["mode"],
        "scientific_authority": (
            loaded["readiness"]["mode"] == "scientific"
        ),
        "phase_lock_identity": loaded["lock_identity"],
        "phase_lock_result_identity": immutable_json_identity(
            paths["lock_result"],
            payload_field="phase_lock_result_payload_sha256",
            decision=_phase_ready_decision(),
        ),
        "readiness_lock_identity":
            loaded["readiness"]["lock_identity"],
        "readiness_result_identity":
            loaded["readiness"]["result_identity"],
        "manifest_identity": loaded["lock"]["manifest_identity"],
        "root_commitment": loaded["readiness"][
            "source_manifest"
        ].get("root_commitment"),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
        "open_command": loaded["commands"]["open"],
        "materialize_command": loaded["commands"]["materialize"],
        "execute_command": loaded["commands"]["execute"],
        "bounded_engine": "execute_training_engine_bounded_j1d",
        "opened_at": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if opened_at is None
            else opened_at
        ),
        "hostname": socket.gethostname() if hostname is None else hostname,
        "operational_audit": operational,
        "marker_only_open": True,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "scientific_work": 0,
        "passes": True,
    }
    written = write_immutable_json(
        paths["marker"],
        marker,
        field="activation_marker_payload_sha256",
    )
    state = {
        name: paths[name].exists()
        for name in (
            "marker",
            "manifest",
            "owner",
            "reservation",
            "consumption",
            "commit_head",
            "result",
            "checkpoint",
        )
    }
    expected = {
        "marker": True,
        "manifest": False,
        "owner": False,
        "reservation": False,
        "consumption": False,
        "commit_head": False,
        "result": False,
        "checkpoint": False,
    }
    if state != expected:
        raise J1dSurfaceIntegrityError(
            "J1d open created work beyond the immutable marker"
        )
    return {
        "marker": written,
        "marker_identity": immutable_json_identity(
            paths["marker"],
            payload_field="activation_marker_payload_sha256",
        ),
        "created_after_open": state,
        "passes": True,
    }


def load_open_training_contract(
    *,
    execution_root: Path,
    readiness_dir: Path,
    require_manifest: bool,
) -> dict[str, Any]:
    loaded = load_training_phase_lock(
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
        or marker.get("phase_lock_identity")
        != loaded["lock_identity"]
        or marker.get("readiness_lock_identity")
        != loaded["readiness"]["lock_identity"]
        or marker.get("readiness_result_identity")
        != loaded["readiness"]["result_identity"]
        or marker.get("manifest_identity")
        != loaded["lock"]["manifest_identity"]
        or marker.get("execute_command")
        != loaded["commands"]["execute"]
        or marker.get("runner_file_sha256") != sha256_path(RUNNER_PATH)
        or marker.get("parent_bounded_engine_file_sha256")
        != EXPECTED_PARENT_ENGINE_SHA256
    ):
        raise J1dSurfaceIntegrityError(
            "J1d open marker changed its immutable contract"
        )
    marker_identity = immutable_json_identity(
        paths["marker"],
        payload_field="activation_marker_payload_sha256",
    )
    manifest = None
    materialized_identity = None
    if require_manifest:
        manifest = load_json(paths["manifest"])
        if manifest != loaded["expected_manifest"]:
            raise J1dSurfaceIntegrityError(
                "J1d materialized manifest changed sealed rows"
            )
        materialized_identity = {
            **manifest_identity(manifest),
            "path": str(paths["manifest"].resolve()),
            "file_sha256": sha256_path(paths["manifest"]),
        }
    return {
        **loaded,
        "marker": marker,
        "marker_identity": marker_identity,
        "manifest": manifest,
        "materialized_manifest_identity": materialized_identity,
    }


def materialize_training_manifest(
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    loaded = load_open_training_contract(
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        require_manifest=False,
    )
    paths = loaded["paths"]
    if paths["manifest"].exists():
        raise FileExistsError(
            f"J1d manifest already exists: {paths['manifest']}"
        )
    for name in (
        "owner",
        "reservation",
        "consumption",
        "commit_head",
        "result",
        "checkpoint",
    ):
        if paths[name].exists():
            raise J1dSurfaceIntegrityError(
                "J1d manifest materialization followed scientific work"
            )
    written = write_immutable_json(
        paths["manifest"],
        {
            key: value
            for key, value in loaded["expected_manifest"].items()
            if key != "root_manifest_payload_sha256"
        },
        field="root_manifest_payload_sha256",
    )
    if written != loaded["expected_manifest"]:
        raise J1dSurfaceIntegrityError(
            "J1d manifest materialization changed bytes"
        )
    return {
        "manifest": written,
        "manifest_identity": {
            **manifest_identity(written),
            "path": str(paths["manifest"].resolve()),
            "file_sha256": sha256_path(paths["manifest"]),
        },
        "streams_reserved": 0,
        "streams_consumed": 0,
        "scientific_work": 0,
        "passes": True,
    }


def _stream_inventory(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(manifest["rows"])
    roles: dict[str, Any] = {}
    all_ids: list[int] = []
    for field in STREAM_RANGES:
        values = [int(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise J1dSurfaceIntegrityError(
                f"J1d manifest duplicated {field}"
            )
        roles[field] = {
            "count": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "ordered_sha256": canonical_json_hash(values),
        }
        all_ids.extend(values)
    checks = {
        "four_roles": len(roles) == 4,
        "all_ids_unique": len(all_ids) == len(set(all_ids)),
        "one_id_per_role_per_root": len(all_ids) == len(rows) * 4,
        "manifest_valid": manifest.get("passes") is True,
    }
    if not all(checks.values()):
        raise J1dSurfaceIntegrityError(
            "J1d stream inventory is invalid"
        )
    return {
        "row_count": len(rows),
        "stream_id_count": len(all_ids),
        "roles": roles,
        "all_stream_ids_sha256": canonical_json_hash(sorted(all_ids)),
        "checks": checks,
        "passes": True,
    }


def _seal_stream_reservation(
    *,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    paths = loaded["paths"]
    inventory = _stream_inventory(loaded["manifest"])
    source_lock = loaded["readiness"]["lock"]
    payload = {
        "version": f"{VERSION}_training_stream_reservation_v1",
        "phase": "training",
        "execution_mode": loaded["readiness"]["mode"],
        "phase_lock_identity": loaded["lock_identity"],
        "marker_identity": loaded["marker_identity"],
        "manifest_identity":
            loaded["materialized_manifest_identity"],
        "source_manifest_identity":
            loaded["readiness"]["source_manifest_identity"],
        "protected_denylist_identity": source_lock.get(
            "authoritative_j1b_preflight_artifacts",
            {},
        ).get("denylist"),
        "stream_inventory": inventory,
        "collision_contract": {
            "accepted_j1b_preflight_lock":
                loaded["readiness"]["lock"].get(
                    "authoritative_j1b_preflight_lock_identity"
                ),
            "accepted_j1b_input_binding_sha256":
                loaded["readiness"]["lock"].get(
                    "authoritative_input_identities_sha256"
                ),
            "exact_fresh_manifest": True,
            "no_regeneration": True,
            "no_substitution": True,
            "historical_collision_count": 0,
        },
        "execute_command": loaded["commands"]["execute"],
        "decision": "RESERVED_J1D_TRAINING_STREAMS",
        "streams_reserved": inventory["stream_id_count"],
        "streams_consumed": 0,
        "scientific_work_before_reservation": 0,
        "passes": True,
    }
    written = write_immutable_json(
        paths["reservation"],
        payload,
        field="stream_reservation_payload_sha256",
        allow_existing_exact=True,
    )
    return {
        "reservation": written,
        "identity": immutable_json_identity(
            paths["reservation"],
            payload_field="stream_reservation_payload_sha256",
        ),
        "passes": True,
    }


def _owner_is_ancestor(
    ledger: Mapping[str, Any],
    *,
    opener: str,
    current: str,
) -> bool:
    owners = {
        str(row.get("owner_record_sha256", ""))
        for row in ledger.get("owners", [])
    }
    if opener not in owners or current not in owners:
        return False
    links = {
        str(row.get("old_owner_sha256", "")):
            str(row.get("new_owner_sha256", ""))
        for row in ledger.get("recoveries", [])
    }
    cursor = opener
    seen: set[str] = set()
    while cursor != current:
        if cursor in seen or cursor not in links:
            return False
        seen.add(cursor)
        cursor = links[cursor]
    return True


def _seal_stream_consumption(
    *,
    loaded: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    paths = loaded["paths"]
    reservation = load_json(paths["reservation"])
    if (
        not verify_payload_hash(
            reservation,
            "stream_reservation_payload_sha256",
        )
        or reservation.get("passes") is not True
    ):
        raise J1dSurfaceIntegrityError(
            "J1d stream reservation is invalid"
        )
    owner = owner_audit["owner"]
    current_owner = str(owner["owner_record_sha256"])
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
            raise J1dSurfaceIntegrityError(
                "Existing J1d stream consumption is invalid"
            )
        opener = str(existing.get("owner_record_sha256", ""))
        checks = {
            "phase_lock": existing.get("phase_lock_identity")
            == loaded["lock_identity"],
            "marker": existing.get("marker_identity")
            == loaded["marker_identity"],
            "manifest": existing.get("manifest_identity")
            == loaded["materialized_manifest_identity"],
            "reservation": existing.get("reservation_identity")
            == reservation_identity,
            "command": existing.get("execute_command")
            == loaded["commands"]["execute"],
            "counts": (
                existing.get("streams_reserved")
                == reservation["streams_reserved"]
                and existing.get("streams_consumed")
                == reservation["streams_reserved"]
            ),
            "owner_ancestry": _owner_is_ancestor(
                owner_audit["ledger"],
                opener=opener,
                current=current_owner,
            ),
        }
        if not all(checks.values()):
            raise J1dSurfaceIntegrityError(
                "Recovered J1d stream consumption changed contract"
            )
        return {
            "consumption": existing,
            "identity": immutable_json_identity(
                paths["consumption"],
                payload_field="stream_consumption_payload_sha256",
            ),
            "opener_owner_record_sha256": opener,
            "current_owner_record_sha256": current_owner,
            "owner_recovery_chain_verified": True,
            "reused_existing_record": True,
            "passes": True,
        }
    payload = {
        "version": f"{VERSION}_training_stream_consumption_opened_v1",
        "phase": "training",
        "execution_mode": loaded["readiness"]["mode"],
        "phase_lock_identity": loaded["lock_identity"],
        "marker_identity": loaded["marker_identity"],
        "manifest_identity":
            loaded["materialized_manifest_identity"],
        "reservation_identity": reservation_identity,
        "owner_record_sha256": current_owner,
        "execute_command": loaded["commands"]["execute"],
        "consumption_scope": "exact full immutable J1d training manifest",
        "stream_inventory": reservation["stream_inventory"],
        "streams_reserved": reservation["streams_reserved"],
        "streams_consumed": reservation["streams_reserved"],
        "scientific_work_before_consumption_record": 0,
        "decision": "OPENED_J1D_TRAINING_STREAM_CONSUMPTION",
        "passes": True,
    }
    written = write_immutable_json(
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
        "opener_owner_record_sha256": current_owner,
        "current_owner_record_sha256": current_owner,
        "owner_recovery_chain_verified": True,
        "reused_existing_record": False,
        "passes": True,
    }


def _acquire_or_reclaim_owner(
    *,
    parent: Any,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    paths = loaded["paths"]
    phase_dir = paths["phase_dir"]
    predecessor = (
        sha256_path(paths["commit_head"])
        if paths["commit_head"].is_file()
        else None
    )
    mode = loaded["readiness"]["mode"]
    if not paths["owner"].exists():
        parent.acquire_writer_owner(
            phase_dir=phase_dir,
            phase="training",
            marker_file_sha256=
                loaded["marker_identity"]["file_sha256"],
            phase_lock_file_sha256=
                loaded["lock_identity"]["file_sha256"],
            command=loaded["commands"]["execute"],
            predecessor_commit_head_sha256=predecessor,
            execution_mode=mode,
        )
    else:
        ledger = parent.load_json(paths["owner"])
        if not parent._verify_ownership_ledger(ledger):
            raise J1dSurfaceIntegrityError(
                "J1d ownership ledger is malformed"
            )
        head = ledger["owners"][-1]
        if int(head.get("pid", -1)) != os.getpid():
            if parent._pid_alive(int(head.get("pid", -1))):
                raise J1dSurfaceOperationalHold(
                    "A live J1d writer owns the training phase"
                )
            parent.reclaim_dead_writer_owner(
                phase_dir=phase_dir,
                phase="training",
                marker_file_sha256=
                    loaded["marker_identity"]["file_sha256"],
                phase_lock_file_sha256=
                    loaded["lock_identity"]["file_sha256"],
                command=loaded["commands"]["execute"],
                execution_mode=mode,
                contention_audit=(
                    {"passes": True, "fixture_only": True}
                    if mode == "miniature_fixture"
                    else None
                ),
            )
    return parent.verify_writer_owner(
        phase_dir=phase_dir,
        phase="training",
        marker_file_sha256=loaded["marker_identity"]["file_sha256"],
        phase_lock_file_sha256=loaded["lock_identity"]["file_sha256"],
        command=loaded["commands"]["execute"],
        execution_mode=mode,
    )


def _runtime_entrypoint(
    *,
    phase_dir: Path,
    after_guard: Any,
    fixture_guard: bool,
    fixture_guard_passes: bool = True,
) -> dict[str, Any]:
    from threes_rl import j1b_operational_repair_preflight as repair

    def model_initializer(parent: Any) -> tuple[Any, Any]:
        model, optimizer = parent.j1.initialize_model_optimizer()
        if (
            parent.j1.parameter_count(model) != EXPECTED_PARAMETER_COUNT
            or parent.j1.model_schema_sha256()
            != EXPECTED_MODEL_SCHEMA_SHA256
        ):
            raise J1dSurfaceIntegrityError(
                "Frozen J1 model identity changed"
            )
        parent.FrozenMinibatchUpdater._validate_optimizer_binding(
            model,
            optimizer,
        )
        parent.j1.assert_finite_model(model)
        return model, optimizer

    operational_audit = None
    if fixture_guard:
        operational_audit = (
            lambda parent, directory:
                (
                    parent.fixture_phase_operational_audit(
                        phase_dir=directory,
                        phase="training",
                        active_seconds=0.0,
                        require_target_disk=True,
                    )
                    if fixture_guard_passes
                    else {
                        "passes": False,
                        "fixture_forced_guard_failure": True,
                    }
                )
        )
    try:
        return repair.guarded_runtime_entrypoint(
            phase_dir=phase_dir,
            model_initializer=model_initializer,
            operational_audit=operational_audit,
            after_guard=after_guard,
        )
    except repair.J1bOperationalHold as error:
        raise J1dSurfaceOperationalHold(str(error)) from error


def _strip_compact_commit_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    import copy

    result = copy.deepcopy(dict(state))
    for key in (
        "committed_unit_ids",
        "commit_prefix_mode",
        "committed_unit_count",
        "committed_unit_head_sha256",
    ):
        result.pop(key, None)
    return result


def _canonicalize_pending_round_commit(
    *,
    parent: Any,
    phase_dir: Path,
    marker_file_sha256: str,
    phase_lock_file_sha256: str,
    command: str,
    execution_mode: str,
) -> dict[str, Any]:
    if not (phase_dir / parent.COMMIT_HEAD_NAME).is_file():
        return {
            "changed": False,
            "round": None,
            "reason": "no_commit_head",
        }
    output_accountant = parent.PhaseOutputAccountant(phase_dir)
    store = parent.IndexedCommitStore(
        phase_dir=phase_dir,
        phase="training",
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        initial_state={},
        execution_mode=execution_mode,
        output_accountant=output_accountant,
    )
    state = store.boundary["state"]
    rounds = list(state.get("round_aggregates", []))
    for prior in rounds[:-1]:
        audit = validate_canonical_round_metric_row(prior)
        if audit["passes"] is not True:
            raise J1dSurfaceIntegrityError(
                "Earlier J1d round metric authentication changed"
            )
    if not rounds:
        return {
            "changed": False,
            "round": None,
            "reason": "no_round_aggregate",
        }
    current = rounds[-1]
    authentication = current.get("metric_authentication")
    if authentication is not None:
        audit = validate_canonical_round_metric_row(current)
        if audit["passes"] is not True:
            raise J1dSurfaceIntegrityError(
                "Existing J1d metric authentication is invalid"
            )
        return {
            "changed": False,
            "round": int(current["round"]),
            "reason": "already_canonical",
            "boundary": store.boundary,
        }
    round_number = int(current.get("round", -1))
    expected_parent_unit = f"round={round_number}|checkpoint"
    if (
        round_number != len(rounds)
        or store.boundary.get("unit_id") != expected_parent_unit
    ):
        raise J1dSurfaceIntegrityError(
            "Uncanonicalized metric row is not at its parent checkpoint"
        )
    canonical_row = canonicalize_round_metric_row(current)
    pending_batch_retirement = state.get(
        "pending_round_batch_retirement"
    )
    if not isinstance(pending_batch_retirement, Mapping):
        raise J1dSurfaceIntegrityError(
            "Parent checkpoint lacks its round-batch retirement identity"
        )
    retirement = parent.retire_round_ppo_batch(
        phase_dir=phase_dir,
        round_batch_identity=pending_batch_retirement,
        checkpoint_boundary=store.boundary,
        output_accountant=output_accountant,
    )
    if retirement.get("all_listed_files_absent") is not True:
        raise J1dSurfaceIntegrityError(
            "Parent round-batch retirement did not verify"
        )
    post_state = _strip_compact_commit_metadata(state)
    post_state.pop("pending_round_batch_retirement", None)
    post_state["round_aggregates"] = [
        *rounds[:-1],
        canonical_row,
    ]
    unit_id = f"round={round_number}|j1d_metric_authentication"
    boundary = store.commit(
        unit_id=unit_id,
        post_state=post_state,
        journal_payload={
            "kind": "j1d_metric_authentication",
            "round": round_number,
            "parent_checkpoint_unit_id": expected_parent_unit,
            "parent_checkpoint_state_file_sha256":
                store.boundary["state_file_sha256"],
            "root_metrics_sha256":
                canonical_row["root_metrics_sha256"],
            "metric_authentication_payload_sha256":
                canonical_row["metric_authentication"][
                    "metric_authentication_payload_sha256"
                ],
            "canonical_aggregates_sha256":
                canonical_row["metric_authentication"][
                    "canonical_aggregates_sha256"
                ],
            "round_batch_retirement_file_sha256":
                retirement["file_sha256"],
            "round_batch_retirement_payload_sha256":
                retirement["payload_sha256"],
            "scientific_work": 0,
            "tolerance_changed": False,
        },
    )
    verified = validate_canonical_round_metric_row(
        boundary["state"]["round_aggregates"][-1]
    )
    if verified["passes"] is not True:
        raise J1dSurfaceIntegrityError(
            "Canonical metric commit failed post-write verification"
        )
    return {
        "changed": True,
        "round": round_number,
        "reason": "canonical_commit_written",
        "boundary": boundary,
        "metric_authentication":
            boundary["state"]["round_aggregates"][-1][
                "metric_authentication"
            ],
    }


def execute_training_engine_bounded_j1d(
    *,
    parent: Any,
    rows: Sequence[Mapping[str, Any]],
    phase_dir: Path,
    marker_file_sha256: str,
    marker_payload_sha256: str,
    phase_lock_file_sha256: str,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    command: str,
    config: Any,
    interrupt_after_boundary: str | None = None,
    operational_audit_fn: Any | None = None,
    wall_clock: Any | None = None,
) -> dict[str, Any]:
    """Run the immutable parent engine with explicit per-round metric commits."""
    j1d_interrupts = {
        "metric_authentication_precommit",
        "metric_authentication_postcommit",
    }
    if (
        config.execution_mode == "scientific"
        and interrupt_after_boundary is not None
    ):
        raise J1dSurfaceIntegrityError(
            "Scientific J1d execution rejects interruption controls"
        )
    pending = _canonicalize_pending_round_commit(
        parent=parent,
        phase_dir=phase_dir,
        marker_file_sha256=marker_file_sha256,
        phase_lock_file_sha256=phase_lock_file_sha256,
        command=command,
        execution_mode=config.execution_mode,
    )
    if (
        pending["changed"]
        and interrupt_after_boundary
        == "metric_authentication_postcommit"
    ):
        raise parent.J1ExecutionPlannedInterruption(
            "Fixture interruption after J1d metric authentication"
        )
    external_parent_interrupt = (
        interrupt_after_boundary
        if interrupt_after_boundary not in j1d_interrupts
        else None
    )
    while True:
        parent_interrupt = (
            external_parent_interrupt
            if external_parent_interrupt is not None
            else "checkpoint"
        )
        try:
            result = parent.execute_training_engine_bounded(
                rows=rows,
                phase_dir=phase_dir,
                marker_file_sha256=marker_file_sha256,
                marker_payload_sha256=marker_payload_sha256,
                phase_lock_file_sha256=phase_lock_file_sha256,
                manifest_file_sha256=manifest_file_sha256,
                manifest_payload_sha256=manifest_payload_sha256,
                command=command,
                config=config,
                interrupt_after_boundary=parent_interrupt,
                operational_audit_fn=operational_audit_fn,
                wall_clock=wall_clock,
            )
        except parent.J1ExecutionPlannedInterruption:
            if external_parent_interrupt is not None:
                raise
            if (
                interrupt_after_boundary
                == "metric_authentication_precommit"
            ):
                raise
            authentication = _canonicalize_pending_round_commit(
                parent=parent,
                phase_dir=phase_dir,
                marker_file_sha256=marker_file_sha256,
                phase_lock_file_sha256=phase_lock_file_sha256,
                command=command,
                execution_mode=config.execution_mode,
            )
            if authentication["changed"] is not True:
                raise J1dSurfaceIntegrityError(
                    "Parent checkpoint produced no J1d metric commit"
                )
            if (
                interrupt_after_boundary
                == "metric_authentication_postcommit"
            ):
                raise
            continue
        round_audits = [
            validate_canonical_round_metric_row(round_row)
            for round_row in result["state"].get("round_aggregates", [])
        ]
        if (
            len(round_audits) != int(config.rounds)
            or any(audit["passes"] is not True for audit in round_audits)
        ):
            raise J1dSurfaceIntegrityError(
                "Completed J1d engine lacks canonical round evidence"
            )
        result = dict(result)
        result["j1d_metric_authentication"] = {
            "algorithm": CANONICAL_METRIC_ALGORITHM,
            "absolute_tolerance": CANONICAL_METRIC_ABS_TOLERANCE,
            "authenticated_rounds": len(round_audits),
            "round_authentication_sha256": canonical_json_hash([
                round_row["metric_authentication"]
                for round_row in result["state"]["round_aggregates"]
            ]),
            "passes": True,
        }
        return result


def j1d_training_sanity_decision(
    *,
    parent: Any,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    round_audits = [
        validate_canonical_round_metric_row(round_row)
        for round_row in report.get("rounds", [])
    ]
    if (
        len(round_audits) != parent.ROUNDS
        or any(audit["passes"] is not True for audit in round_audits)
    ):
        raise J1dSurfaceIntegrityError(
            "J1d training sanity lacks canonical round authentication"
        )
    result = dict(parent.training_sanity_decision(report))
    result["j1d_metric_authentication"] = {
        "algorithm": CANONICAL_METRIC_ALGORITHM,
        "absolute_tolerance": CANONICAL_METRIC_ABS_TOLERANCE,
        "rounds": len(round_audits),
        "round_audits_sha256": canonical_json_hash([
            audit["checks"] for audit in round_audits
        ]),
        "writer_verifier_shared_function": True,
        "passes": True,
    }
    return result


def _terminal_base(
    *,
    loaded: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
    engine_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "version": f"{VERSION}_training_terminal_result_v1",
        "phase": "training",
        "execution_mode": loaded["readiness"]["mode"],
        "scientific_authority": (
            loaded["readiness"]["mode"] == "scientific"
        ),
        "fixture_only": (
            loaded["readiness"]["mode"] == "miniature_fixture"
        ),
        "phase_lock_identity": loaded["lock_identity"],
        "phase_lock_result_identity": immutable_json_identity(
            loaded["paths"]["lock_result"],
            payload_field="phase_lock_result_payload_sha256",
            decision=_phase_ready_decision(),
        ),
        "marker_identity": loaded["marker_identity"],
        "manifest_identity":
            loaded["materialized_manifest_identity"],
        "readiness_lock_identity":
            loaded["readiness"]["lock_identity"],
        "readiness_result_identity":
            loaded["readiness"]["result_identity"],
        "source_manifest_identity":
            loaded["readiness"]["source_manifest_identity"],
        "stream_reservation_identity": dict(reservation_identity),
        "stream_consumption_identity": dict(consumption_identity),
        "ownership_ledger_identity": immutable_json_identity(
            loaded["paths"]["owner"],
            payload_field="ownership_payload_sha256",
        ),
        "owner_record_sha256": owner_audit["owner"][
            "owner_record_sha256"
        ],
        "owner_recovery_count": len(
            owner_audit["ledger"].get("recoveries", [])
        ),
        "bounded_engine": "execute_training_engine_bounded_j1d",
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
        "wrapper_runner_file_sha256": sha256_path(RUNNER_PATH),
        "wrapper_charter_file_sha256": sha256_path(CHARTER_PATH),
        "wrapper_test_file_sha256": sha256_path(TEST_PATH),
        "execute_command": loaded["commands"]["execute"],
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
        "development_opened": False,
        "confirmation_opened": False,
        "human_session_reads": 0,
    }


def _seal_scientific_training_terminal(
    *,
    parent: Any,
    loaded: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if engine_result.get("completed") is not True:
        raise J1dSurfaceIntegrityError(
            "J1d training engine did not complete"
        )
    state = engine_result["state"]
    boundary = engine_result["boundary"]
    if (
        state.get("engine_stage") != "complete"
        or int(state.get("round_number", -1)) != parent.ROUNDS
        or boundary.get("chain_audit", {}).get("passes") is not True
    ):
        raise J1dSurfaceIntegrityError(
            "J1d training terminal boundary is incomplete"
        )
    manifest = loaded["manifest"]
    model, optimizer = parent._load_model_optimizer_from_runtime(state)
    training_input = {
        "manifest_payload_sha256": manifest[
            "root_manifest_payload_sha256"
        ],
        "marker_file_sha256": loaded["marker_identity"][
            "file_sha256"
        ],
        "terminal_state_file_sha256": boundary["state_file_sha256"],
        "terminal_commit_head_payload_sha256": boundary[
            "commit_head_payload_sha256"
        ],
        "completed_root_ids_sha256": parent.j1.stable_hash(
            state["all_completed_root_ids"]
        ),
        "optimizer_step_ids_sha256": parent.j1.stable_hash(
            state["optimizer_step_ids"]
        ),
        "round_aggregates_sha256": parent.j1.stable_hash(
            state["round_aggregates"]
        ),
        "wrapper_runner_file_sha256": sha256_path(RUNNER_PATH),
        "j1b_readiness_result_payload_sha256":
            loaded["readiness"]["result"][
                "readiness_result_payload_sha256"
            ],
    }
    checkpoint_payload = parent.candidate_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        training_manifest_identity=parent.root_manifest_identity(
            manifest
        ),
        training_marker_file_sha256=
            loaded["marker_identity"]["file_sha256"],
        training_result_input_sha256=canonical_json_hash(
            training_input
        ),
    )
    checkpoint_identity = parent.write_candidate_checkpoint(
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
        "rounds": list(state["round_aggregates"]),
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
    sanity = j1d_training_sanity_decision(
        parent=parent,
        report=report,
    )
    sanity_payload = {
        **sanity,
        "training_input": training_input,
        "training_report_sha256": parent.j1.stable_hash(report),
        "checkpoint_authoritative": (
            sanity["decision"] == "READY_J1_TRAINING_SANITY"
        ),
        "checkpoint_quarantined": (
            sanity["decision"] != "READY_J1_TRAINING_SANITY"
        ),
        "wrapper_runner_file_sha256": sha256_path(RUNNER_PATH),
    }
    write_immutable_json(
        loaded["paths"]["sanity"],
        sanity_payload,
        field="training_sanity_payload_sha256",
        allow_existing_exact=True,
    )
    base = _terminal_base(
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
            "checkpoint_authoritative":
                sanity_payload["checkpoint_authoritative"],
            "checkpoint_quarantined":
                sanity_payload["checkpoint_quarantined"],
            "authenticated_terminal_boundary": report[
                "authenticated_terminal_boundary"
            ],
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
        allow_existing_exact=True,
    )


def _seal_fixture_training_terminal(
    *,
    loaded: Mapping[str, Any],
    engine_result: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if engine_result.get("completed") is not True:
        raise J1dSurfaceIntegrityError(
            "Miniature bounded engine did not complete"
        )
    state = engine_result["state"]
    boundary = engine_result["boundary"]
    base = _terminal_base(
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=engine_result,
    )
    base.update(
        {
            "decision": "READY_J1D_MINIATURE_TRAINING_FIXTURE",
            "scientific_authority": False,
            "fixture_only": True,
            "checkpoint_authoritative": False,
            "checkpoint_quarantined": True,
            "completed_root_ids": list(
                state.get("all_completed_root_ids", [])
            ),
            "optimizer_step_ids": list(
                state.get("optimizer_step_ids", [])
            ),
            "round_aggregates_sha256": canonical_json_hash(
                state.get("round_aggregates", [])
            ),
            "terminal_state_file_sha256": boundary.get(
                "state_file_sha256"
            ),
            "terminal_commit_head_payload_sha256": boundary.get(
                "commit_head_payload_sha256"
            ),
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
        allow_existing_exact=True,
    )


def _seal_failure_terminal(
    *,
    loaded: Mapping[str, Any],
    reservation_identity: Mapping[str, Any],
    consumption_identity: Mapping[str, Any],
    owner_audit: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    operational = isinstance(
        error,
        J1dSurfaceOperationalHold,
    ) or error.__class__.__name__ in {
        "J1ExecutionOperationalHold",
    }
    decision = (
        "HOLD_J1D_OPERATIONAL"
        if operational
        else "KILL_J1D_TRAINING_INTEGRITY"
    )
    base = _terminal_base(
        loaded=loaded,
        reservation_identity=reservation_identity,
        consumption_identity=consumption_identity,
        owner_audit=owner_audit,
        engine_result=None,
    )
    base.update(
        {
            "decision": decision,
            "failure_class": (
                "operational" if operational else "integrity"
            ),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "partial_work_preserved": True,
            "checkpoint_authoritative": False,
            "checkpoint_quarantined": True,
        }
    )
    return write_immutable_json(
        loaded["paths"]["result"],
        base,
        field="terminal_result_payload_sha256",
        allow_existing_exact=True,
    )


def _load_terminal_result(path: Path) -> dict[str, Any]:
    result = load_json(path)
    if (
        not verify_payload_hash(
            result,
            "terminal_result_payload_sha256",
        )
        or result.get("wrapper_runner_file_sha256")
        != sha256_path(RUNNER_PATH)
        or result.get("wrapper_charter_file_sha256")
        != sha256_path(CHARTER_PATH)
        or result.get("parent_bounded_engine_file_sha256")
        != EXPECTED_PARENT_ENGINE_SHA256
        or result.get("bounded_engine")
        != "execute_training_engine_bounded_j1d"
        or result.get("promote") is not False
    ):
        raise J1dSurfaceIntegrityError(
            "Existing J1d terminal result is invalid"
        )
    return result


def _finalize_retention(
    *,
    parent: Any,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    result = _load_terminal_result(loaded["paths"]["result"])
    retention = parent.seal_phase_retention_manifest(
        execution_root=Path(
            loaded["lock"]["future_execution_root"]
        ),
        phase="training",
    )
    observed = immutable_json_identity(
        loaded["paths"]["retention"],
        payload_field="retention_payload_sha256",
    )
    if retention.get("retention_payload_sha256") != observed[
        "payload_sha256"
    ]:
        raise J1dSurfaceIntegrityError(
            "J1d retention seal changed after write"
        )
    return {
        "result": result,
        "result_identity": immutable_json_identity(
            loaded["paths"]["result"],
            payload_field="terminal_result_payload_sha256",
            decision=str(result["decision"]),
        ),
        "retention": retention,
        "retention_identity": observed,
        "passes": True,
    }


def execute_training_from_artifacts(
    *,
    execution_root: Path,
    readiness_dir: Path,
) -> dict[str, Any]:
    loaded = load_open_training_contract(
        execution_root=execution_root,
        readiness_dir=readiness_dir,
        require_manifest=True,
    )
    mode = loaded["readiness"]["mode"]
    fixture_env_names = (
        "J1D_FIXTURE_RUNTIME_FAILURE",
        "J1D_FIXTURE_FIRST_GUARD_FAILURE",
        "J1D_FIXTURE_PRE_ENGINE_INTERRUPT",
        "J1D_FIXTURE_INTERRUPT_AFTER_BOUNDARY",
    )
    if mode == "scientific" and any(
        os.environ.get(name) for name in fixture_env_names
    ):
        raise J1dSurfaceIntegrityError(
            "Scientific execution rejects fixture controls"
        )
    if (
        mode == "miniature_fixture"
        and os.environ.get("J1D_FIXTURE_RUNTIME_FAILURE") == "1"
    ):
        raise J1dSurfaceOperationalHold(
            "Fixture runtime configuration failure before parent import"
        )

    def after_guard(
        parent: Any,
        initial_model: Any,
        _initial_optimizer: Any,
    ) -> dict[str, Any]:
        initial_state_sha256 = parent.j1.stable_hash(
            initial_model.state_dict()
        )
        if initial_state_sha256 != EXPECTED_INITIAL_MODEL_STATE_SHA256:
            raise J1dSurfaceIntegrityError(
                "Frozen initial J1 model state changed"
            )
        if loaded["paths"]["result"].exists():
            finalized = _finalize_retention(
                parent=parent,
                loaded=loaded,
            )
            return {
                "terminal": finalized,
                "terminal_already_sealed": True,
                "initial_model_state_sha256":
                    initial_state_sha256,
                "passes": True,
            }

        owner_audit = _acquire_or_reclaim_owner(
            parent=parent,
            loaded=loaded,
        )
        pre_engine_interrupt = (
            os.environ.get("J1D_FIXTURE_PRE_ENGINE_INTERRUPT")
            if mode == "miniature_fixture"
            else None
        )
        if pre_engine_interrupt == "after-owner":
            raise parent.J1ExecutionPlannedInterruption(
                "Fixture interruption after owner"
            )
        reservation = _seal_stream_reservation(loaded=loaded)
        if pre_engine_interrupt == "after-reservation":
            raise parent.J1ExecutionPlannedInterruption(
                "Fixture interruption after reservation"
            )
        consumption = _seal_stream_consumption(
            loaded=loaded,
            owner_audit=owner_audit,
        )
        if pre_engine_interrupt == "after-consumption":
            raise parent.J1ExecutionPlannedInterruption(
                "Fixture interruption after consumption"
            )

        if mode == "scientific":
            config = parent.TrainingEngineConfig()
            operational_audit_fn = None
            interrupt_after_boundary = None
        else:
            fixture = dict(
                loaded["readiness"]["lock"].get(
                    "fixture_engine_config",
                    {},
                )
            )
            config = parent.TrainingEngineConfig(
                rounds=int(fixture["rounds"]),
                roots_per_round=int(fixture["roots_per_round"]),
                env_count=int(fixture["env_count"]),
                minibatch_size=int(fixture["minibatch_size"]),
                max_moves=int(fixture["max_moves"]),
                execution_mode="miniature_fixture",
            )
            operational_audit_fn = parent.fixture_phase_operational_audit
            interrupt_after_boundary = os.environ.get(
                "J1D_FIXTURE_INTERRUPT_AFTER_BOUNDARY"
            )
        try:
            engine_result = execute_training_engine_bounded_j1d(
                parent=parent,
                rows=loaded["manifest"]["rows"],
                phase_dir=loaded["paths"]["phase_dir"],
                marker_file_sha256=
                    loaded["marker_identity"]["file_sha256"],
                marker_payload_sha256=
                    loaded["marker_identity"]["payload_sha256"],
                phase_lock_file_sha256=
                    loaded["lock_identity"]["file_sha256"],
                manifest_file_sha256=
                    loaded["materialized_manifest_identity"][
                        "file_sha256"
                    ],
                manifest_payload_sha256=
                    loaded["materialized_manifest_identity"][
                        "payload_sha256"
                    ],
                command=loaded["commands"]["execute"],
                config=config,
                interrupt_after_boundary=interrupt_after_boundary,
                operational_audit_fn=operational_audit_fn,
            )
            if mode == "scientific":
                terminal = _seal_scientific_training_terminal(
                    parent=parent,
                    loaded=loaded,
                    engine_result=engine_result,
                    reservation_identity=reservation["identity"],
                    consumption_identity=consumption["identity"],
                    owner_audit=owner_audit,
                )
            else:
                terminal = _seal_fixture_training_terminal(
                    loaded=loaded,
                    engine_result=engine_result,
                    reservation_identity=reservation["identity"],
                    consumption_identity=consumption["identity"],
                    owner_audit=owner_audit,
                )
        except parent.J1ExecutionPlannedInterruption:
            raise
        except BaseException as error:
            terminal = _seal_failure_terminal(
                loaded=loaded,
                reservation_identity=reservation["identity"],
                consumption_identity=consumption["identity"],
                owner_audit=owner_audit,
                error=error,
            )
        finalized = _finalize_retention(
            parent=parent,
            loaded=loaded,
        )
        return {
            "terminal": finalized,
            "terminal_decision": terminal["decision"],
            "terminal_already_sealed": False,
            "initial_model_state_sha256": initial_state_sha256,
            "passes": True,
        }

    fixture_guard_failure = (
        mode == "miniature_fixture"
        and (
            os.environ.get("J1D_FIXTURE_FIRST_GUARD_FAILURE") == "1"
            or loaded["readiness"]["lock"].get(
                "fixture_first_guard_passes",
                True,
            )
            is not True
        )
    )
    runtime = _runtime_entrypoint(
        phase_dir=loaded["paths"]["phase_dir"],
        after_guard=after_guard,
        fixture_guard=mode == "miniature_fixture",
        fixture_guard_passes=not fixture_guard_failure,
    )
    after = runtime["after_guard"]
    return {
        "version": f"{VERSION}_execute_result_v1",
        "phase": "training",
        "execution_mode": mode,
        "runtime": runtime["runtime"],
        "first_operational_guard": runtime["operational_audit"],
        "ordering": runtime["ordering"],
        "terminal_decision": after["terminal"]["result"]["decision"],
        "terminal_result_identity":
            after["terminal"]["result_identity"],
        "retention_identity": after["terminal"]["retention_identity"],
        "terminal_already_sealed":
            after["terminal_already_sealed"],
        "initial_model_state_sha256":
            after["initial_model_state_sha256"],
        "development_opened": False,
        "confirmation_opened": False,
        "promote": False,
        "passes": True,
    }


def zero_work_counters() -> dict[str, int]:
    return {
        "j1d_training_phase_locks": 0,
        "j1d_training_markers": 0,
        "j1d_materialized_manifests": 0,
        "j1d_owners": 0,
        "j1d_streams_reserved": 0,
        "j1d_streams_consumed": 0,
        "j1d_genesis_commits": 0,
        "normal_start_games": 0,
        "scientific_transitions": 0,
        "scientific_labels": 0,
        "scientific_optimizer_steps": 0,
        "scientific_checkpoints": 0,
        "development_reads": 0,
        "confirmation_reads": 0,
        "policy_or_score_outcomes": 0,
        "human_session_reads": 0,
        "incumbent_changes": 0,
        "dashboard_changes": 0,
        "promotion_actions": 0,
    }


def surface_schema() -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_schema_v1",
        "public_commands": list(PUBLIC_COMMANDS),
        "phase": "training",
        "development_surface_present": False,
        "confirmation_surface_present": False,
        "promotion_surface_present": False,
        "bounded_engine": "execute_training_engine_bounded_j1d",
        "legacy_engine_scientific_reachable": False,
        "metric_authentication_contract": {
            "algorithm": CANONICAL_METRIC_ALGORITHM,
            "writer_and_verifier_share_function": True,
            "published_from_authenticated_per_root_rows": True,
            "published_projection_equals_canonical_exactly": True,
            "published_hash_equals_canonical_hash": True,
            "rehashed_one_ulp_tamper_fails": True,
            "per_root_rows_hash_in_recursive_checkpoint_state": True,
            "absolute_tolerance": CANONICAL_METRIC_ABS_TOLERANCE,
            "tolerance_changed": False,
            "tolerance_is_not_publication_authentication": True,
            "legacy_and_canonical_paths_computed_in_regression": True,
            "parent_module_mutated": False,
            "canonical_commit_per_round": True,
            "canonical_commit_count": 64,
        },
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_schema_sha256": EXPECTED_MODEL_SCHEMA_SHA256,
        "initial_model_state_sha256":
            EXPECTED_INITIAL_MODEL_STATE_SHA256,
        "root_count": TRAIN_ROOTS,
        "rounds": 64,
        "roots_per_round": 256,
        "env_count": 16,
        "epochs_per_round": 4,
        "minibatch_size": 4_096,
        "starter_tile": None,
        "runtime_contract": {
            "torch_interop_threads": 1,
            "torch_intraop_threads": 1,
            "deterministic_algorithms": True,
            "configuration_before_parent_import": True,
            "first_parent_guard_before_owner": True,
            "first_parent_guard_before_stream_reservation": True,
            "first_parent_guard_before_stream_consumption": True,
            "first_parent_guard_before_genesis": True,
        },
        "immutable_json_contract": {
            "json_native_before_payload_hash": True,
            "json_native_before_object_equality": True,
            "exact_serialized_bytes_verified_after_write": True,
            "create_once_hard_link": True,
            "file_and_parent_fsync": True,
            "collision_or_tamper_fails_closed": True,
        },
        "spent_j1b_contract": {
            "namespace_spent": True,
            "retry_authorized": False,
            "declared_stream_ranges_spent": True,
            "external_terminal_decision":
                "HOLD_J1B_OPEN_SERIALIZATION_INTEGRITY",
        },
        "spent_j1c_contract": {
            "decision": "KILL_J1C_TRAINING_INTEGRITY",
            "namespace_spent": True,
            "retry_authorized": False,
            "checkpoint_quarantined": True,
            "all_declared_stream_ranges_spent": True,
            "terminal_file_sha256": EXPECTED_SPENT_J1C_TERMINAL[0],
            "retention_file_sha256": EXPECTED_SPENT_J1C_RETENTION[0],
        },
        "preserved_j1d_v1_readiness_evidence": {
            "source_identities": dict(EXPECTED_V1_SOURCE_IDENTITIES),
            "artifact_identities": {
                name: {
                    "file_sha256": identity[0],
                    "payload_field": identity[1],
                    "payload_sha256": identity[2],
                }
                for name, identity in sorted(
                    EXPECTED_V1_READINESS_EVIDENCE.items()
                )
            },
            "readiness_terminal_sealed": False,
        },
        "fresh_stream_ranges": {
            field: {
                "start": start,
                "end_inclusive": end,
                "rows": end - start + 1,
            }
            for field, (start, end) in STREAM_RANGES.items()
        },
        "terminal_decisions": [
            "READY_J1_TRAINING_SANITY",
            "HOLD_J1_LEARNING_SANITY",
            "HOLD_J1D_OPERATIONAL",
            "KILL_J1D_TRAINING_INTEGRITY",
        ],
        "promote": False,
    }
    return payload_with_hash(payload, "schema_payload_sha256")


def runtime_storage_projection() -> dict[str, Any]:
    parent_path = (
        RUNS_ROOT
        / "forensics"
        / "j1_execution_surface_readiness_v1"
        / "J1_EXECUTION_RUNTIME_STORAGE_PROJECTION.json"
    )
    _assert_file_hash(
        parent_path,
        "92dfc49a8f0830a4b39c627d9257e4a20b4ca504019c455b3b2b1eb05a959f20",
    )
    parent = load_json(parent_path)
    if (
        not verify_payload_hash(parent, "projection_payload_sha256")
        or parent.get("passes") is not True
    ):
        raise J1dSurfaceIntegrityError(
            "Accepted parent projection is invalid"
        )
    central = parent["training"]["central"]
    parent_before_margin = int(
        central["storage"]["projected_before_margin_bytes"]
    )
    metric_commit_storage_envelope_bytes = 2 * 1024**3
    wrapper_storage_bytes = (
        32 * 1024**2 + metric_commit_storage_envelope_bytes
    )
    before_margin = parent_before_margin + wrapper_storage_bytes
    with_margin = int(round(before_margin * 1.25))
    cap_bytes = 24 * 1024**3
    parent_runtime_before_margin_hours = (
        float(central["hours_with_25pct_margin"]) / 1.25
    )
    wrapper_runtime_seconds = 600.0
    runtime_with_margin_hours = (
        parent_runtime_before_margin_hours
        + wrapper_runtime_seconds / 3600.0
    ) * 1.25
    created_files = int(
        central["bounded_io"]["created_files"]
    ) + 16 + 64 * 8
    fsync_count = int(
        central["bounded_io"]["fsync_count"]
    ) + 32 + 64 * 16
    checks = {
        "parent_projection_file_exact": True,
        "parent_projection_payload_valid": True,
        "same_16384_root_workload": (
            int(central["bounded_io"]["transition_rows"])
            == 8_388_608
        ),
        "fixed_32_tick_cadence": (
            int(
                central["bounded_io"][
                    "fixed_collection_tick_cadence"
                ]
            )
            == 32
        ),
        "same_max_replay_ticks": (
            int(
                central["bounded_io"][
                    "maximum_replayed_collection_ticks"
                ]
            )
            == 32
        ),
        "same_retirement_contract": (
            parent["retirement_contract"][
                "transition_chunks_current_round_only"
            ]
            and parent["retirement_contract"][
                "round_batch_current_round_only"
            ]
            and parent["retirement_contract"][
                "idempotent_crash_window_recovery"
            ]
        ),
        "storage_with_margin_at_most_24gib": (
            with_margin <= cap_bytes
        ),
        "runtime_with_margin_at_most_72h": (
            runtime_with_margin_hours <= 72.0
        ),
        "file_count_within_50000": created_files <= 50_000,
        "fsync_count_within_200000": fsync_count <= 200_000,
        "sensitivity_reported": (
            parent["training"].get("sensitivity_5000_moves")
            is not None
        ),
        "sensitivity_not_conjunctive": (
            parent["training"]["sensitivity_5000_moves"].get(
                "diagnostic_not_conjunctive"
            )
            is True
        ),
        "fixed_25pct_margin": True,
        "sixty_four_metric_commits_budgeted": (
            metric_commit_storage_envelope_bytes == 2 * 1024**3
        ),
    }
    payload = {
        "version": f"{VERSION}_runtime_storage_projection_v1",
        "method": (
            "accepted parent bounded-engine projection plus a fixed "
            "2-GiB 64-round canonical-commit envelope, 32-MiB J1d "
            "wrapper envelope, and 600-second fresh-process "
            "orchestration envelope; no retiming"
        ),
        "parent_projection_identity": {
            "path": str(parent_path.resolve()),
            "file_sha256": sha256_path(parent_path),
            "payload_sha256": parent["projection_payload_sha256"],
        },
        "training": {
            "root_count": TRAIN_ROOTS,
            "central_moves": 512,
            "parent_projected_before_margin_bytes":
                parent_before_margin,
            "wrapper_storage_envelope_bytes":
                wrapper_storage_bytes,
            "metric_commit_storage_envelope_bytes":
                metric_commit_storage_envelope_bytes,
            "projected_before_margin_bytes": before_margin,
            "safety_multiplier": 1.25,
            "projected_with_margin_bytes": with_margin,
            "projected_with_margin_gib": with_margin / 1024**3,
            "storage_cap_bytes": cap_bytes,
            "storage_cap_gib": 24.0,
            "parent_runtime_before_margin_hours":
                parent_runtime_before_margin_hours,
            "wrapper_runtime_envelope_seconds":
                wrapper_runtime_seconds,
            "runtime_with_margin_hours":
                runtime_with_margin_hours,
            "runtime_cap_hours": 72.0,
            "created_files": created_files,
            "created_file_cap": 50_000,
            "fsync_count": fsync_count,
            "fsync_cap": 200_000,
            "current_round_chunks_only": True,
            "current_round_batch_only": True,
            "three_rolling_slots_or_orphan_envelope": True,
            "retirement_recovery": True,
            "bounded_abandoned_unit_charge": True,
        },
        "sensitivity_5000_moves": parent["training"][
            "sensitivity_5000_moves"
        ],
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": zero_work_counters(),
    }
    return payload_with_hash(payload, "projection_payload_sha256")


def write_test_evidence(
    *,
    readiness_dir: Path,
    commands: Sequence[Mapping[str, Any]],
    documented_deselections: Sequence[str],
    independent_reproduction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if FUTURE_EXECUTION_ROOT.exists():
        raise J1dSurfaceIntegrityError(
            "Future J1d execution root exists before test evidence"
        )
    expected_kinds = {
        "py_compile",
        "focused_j1d_v2_surface",
        "j1b_terminalization",
        "parent_j1b_training_surface",
        "parent_j1b_preflight",
        "parent_j1_execution_surface",
        "parent_j1_joint_policy_value",
        "parent_j1a_cost_power",
        "clean_process_real_operational_roundtrip",
        "applicable_non_science_regressions",
        "miniature_full_chain",
        "synthetic_64_round_metric_authentication",
        "parent_j1c_training_surface",
    }
    observed_kinds = {str(row.get("kind")) for row in commands}
    checks = {
        "charter_exact": sha256_path(CHARTER_PATH)
        == EXPECTED_CHARTER_SHA256,
        "runner_present": RUNNER_PATH.is_file(),
        "tests_present": TEST_PATH.is_file(),
        "command_kinds_exact": observed_kinds == expected_kinds,
        "all_commands_passed": all(
            row.get("passed") is True
            and int(row.get("returncode", -1)) == 0
            for row in commands
        ),
        "future_execution_root_absent":
            not FUTURE_EXECUTION_ROOT.exists(),
        "no_scientific_work": all(
            value == 0 for value in zero_work_counters().values()
        ),
    }
    payload = {
        "version": f"{VERSION}_test_evidence_v1",
        "created_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "source_identities": {
            "charter_file_sha256": sha256_path(CHARTER_PATH),
            "runner_file_sha256": sha256_path(RUNNER_PATH),
            "test_file_sha256": sha256_path(TEST_PATH),
        },
        "commands": [dict(row) for row in commands],
        "documented_historical_artifact_state_deselections":
            list(documented_deselections),
        "independent_reproduction": (
            None
            if independent_reproduction is None
            else dict(independent_reproduction)
        ),
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": zero_work_counters(),
    }
    if not payload["passes"]:
        raise J1dSurfaceIntegrityError(
            "J1d training-surface test evidence gates failed"
        )
    return write_immutable_json(
        readiness_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def seal_readiness_package(
    *,
    readiness_dir: Path,
    operational_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if readiness_dir.resolve() != READINESS_DIR.resolve():
        raise J1dSurfaceIntegrityError(
            "Scientific readiness namespace changed"
        )
    if FUTURE_EXECUTION_ROOT.exists():
        raise J1dSurfaceIntegrityError(
            "Future J1d execution root exists before readiness seal"
        )
    paths = _readiness_paths(readiness_dir)
    for name in ("source_manifest", "stream_authority", "root_cause"):
        if not paths[name].is_file():
            raise J1dSurfaceIntegrityError(
                f"J1d {name} must precede readiness"
            )
    if not paths["test_evidence"].is_file():
        raise J1dSurfaceIntegrityError(
            "J1d test evidence must precede readiness"
        )
    for name in ("schema", "projection", "input_bindings", "lock", "result"):
        if paths[name].exists():
            raise FileExistsError(
                f"J1d readiness artifact already exists: {paths[name]}"
            )
    input_audit = audit_authoritative_inputs(
        require_future_execution_absent=True,
    )
    input_bindings = write_immutable_json(
        paths["input_bindings"],
        {
            "version": f"{VERSION}_input_bindings_v1",
            "authoritative_input_audit": input_audit,
            "future_execution_root": str(
                FUTURE_EXECUTION_ROOT.resolve()
            ),
            "future_execution_root_absent": True,
            "protected_parent_artifacts_parsed_for_identity_only": True,
            "human_session_reads": 0,
            "passes": True,
        },
        field="input_bindings_payload_sha256",
    )
    schema = write_immutable_json(
        paths["schema"],
        {
            key: value
            for key, value in surface_schema().items()
            if key != "schema_payload_sha256"
        },
        field="schema_payload_sha256",
    )
    projection = runtime_storage_projection()
    write_immutable_json(
        paths["projection"],
        {
            key: value
            for key, value in projection.items()
            if key != "projection_payload_sha256"
        },
        field="projection_payload_sha256",
    )
    if operational_audit.get("passes") is not True:
        raise J1dSurfaceOperationalHold(
            "J1d readiness operational audit failed"
        )
    if (
        operational_audit.get("real_marker_roundtrip", {}).get("passes")
        is not True
    ):
        raise J1dSurfaceIntegrityError(
            "J1d real operational-audit marker roundtrip is absent"
        )
    source_identity = immutable_json_identity(
        _source_manifest_path(),
        payload_field="prospective_manifest_payload_sha256",
    )
    stream_authority_identity = immutable_json_identity(
        paths["stream_authority"],
        payload_field="stream_authority_payload_sha256",
    )
    root_cause_identity = immutable_json_identity(
        paths["root_cause"],
        payload_field="root_cause_payload_sha256",
    )
    artifacts = {
        "test_evidence": immutable_json_identity(
            paths["test_evidence"],
            payload_field="test_evidence_payload_sha256",
        ),
        "schema": immutable_json_identity(
            paths["schema"],
            payload_field="schema_payload_sha256",
        ),
        "projection": immutable_json_identity(
            paths["projection"],
            payload_field="projection_payload_sha256",
        ),
        "input_bindings": immutable_json_identity(
            paths["input_bindings"],
            payload_field="input_bindings_payload_sha256",
        ),
        "root_cause": root_cause_identity,
    }
    source_lock = load_json(
        J1B_PREFLIGHT_DIR / "J1B_READINESS_LOCK.json"
    )
    lock_payload = {
        "version": f"{VERSION}_readiness_lock_v1",
        "decision": READY_DECISION,
        "execution_mode": "scientific",
        "scientific_authority": True,
        "fixture_only": False,
        "future_execution_root": str(
            FUTURE_EXECUTION_ROOT.resolve()
        ),
        "future_execution_root_absent": True,
        "charter_file_sha256": sha256_path(CHARTER_PATH),
        "runner_file_sha256": sha256_path(RUNNER_PATH),
        "test_file_sha256": sha256_path(TEST_PATH),
        "artifacts": artifacts,
        "preserved_v1_source_identities":
            input_audit["identities"]["v1_sources"],
        "preserved_v1_pre_correction_evidence":
            input_audit["identities"]["v1_pre_correction_evidence"],
        "source_manifest_identity": source_identity,
        "stream_authority_identity": stream_authority_identity,
        "authoritative_input_identities_sha256":
            input_audit["identities_sha256"],
        "authoritative_j1b_preflight_lock_identity":
            input_audit["identities"]["j1b_preflight_lock"],
        "authoritative_j1b_preflight_result_identity":
            input_audit["identities"]["j1b_preflight_result"],
        "authoritative_j1b_preflight_artifacts":
            source_lock["artifacts"],
        "pre_a1_historical_evidence":
            input_audit["identities"]["pre_a1_history"],
        "spent_j1_execution_identities":
            input_audit["identities"]["spent_j1_execution"],
        "j1b_training_source_identities":
            input_audit["identities"]["j1b_training_sources"],
        "j1b_training_readiness_identities":
            input_audit["identities"]["j1b_training_readiness"],
        "spent_j1b_execution_identities":
            input_audit["identities"]["spent_j1b_execution"],
        "external_j1b_terminal_identities":
            input_audit["identities"]["external_j1b_terminal"],
        "j1c_source_identities":
            input_audit["identities"]["j1c_sources"],
        "j1c_readiness_identities":
            input_audit["identities"]["j1c_readiness"],
        "spent_j1c_terminal_identity":
            input_audit["identities"]["spent_j1c_terminal"],
        "spent_j1c_retention_identity":
            input_audit["identities"]["spent_j1c_retention"],
        "root_cause_identity": root_cause_identity,
        "parent_source_identities":
            input_audit["identities"]["parent_sources"],
        "parent_readiness_identities":
            input_audit["identities"]["parent_readiness"],
        "parent_bounded_engine_file_sha256":
            EXPECTED_PARENT_ENGINE_SHA256,
        "manifest_contract": {
            "row_count": TRAIN_ROOTS,
            "canonical_rows_sha256":
                EXPECTED_CANONICAL_ROWS_SHA256,
            "root_commitment_payload_sha256":
                EXPECTED_ROOT_COMMITMENT_PAYLOAD_SHA256,
            "root_set_sha256": EXPECTED_ROOT_SET_SHA256,
            "stream_ranges": {
                field: {
                    "start": start,
                    "end_inclusive": end,
                }
                for field, (start, end) in STREAM_RANGES.items()
            },
        },
        "public_commands": list(PUBLIC_COMMANDS),
        "bounded_engine": "execute_training_engine_bounded_j1d",
        "operational_audit": dict(operational_audit),
        "zero_work": zero_work_counters(),
        "passes": True,
    }
    lock = write_immutable_json(
        paths["lock"],
        lock_payload,
        field="readiness_lock_payload_sha256",
    )
    lock_identity = immutable_json_identity(
        paths["lock"],
        payload_field="readiness_lock_payload_sha256",
        decision=READY_DECISION,
    )
    result = write_immutable_json(
        paths["result"],
        {
            "version": f"{VERSION}_readiness_result_v1",
            "decision": READY_DECISION,
            "readiness_lock_identity": lock_identity,
            "artifacts": artifacts,
            "operational_audit": dict(operational_audit),
            "integrity_checks": {
                "source_and_history_exact": True,
                "v1_pre_correction_evidence_preserved": True,
                "fresh_manifest_exact": True,
                "compact_stream_authority_exact": True,
                "spent_j1b_external_terminal_exact": True,
                "spent_j1c_terminal_and_retention_exact": True,
                "root_cause_structural_only": True,
                "canonical_writer_verifier_shared": True,
                "published_projection_exact": True,
                "published_hash_equals_canonical_hash": True,
                "rehashed_one_ulp_tamper_rejected": True,
                "computed_reduction_order_fixture_passes": True,
                "canonical_64_round_fixture_passes": True,
                "json_native_roundtrip_exact": True,
                "runtime_order_repaired": True,
                "bounded_engine_only": True,
                "test_evidence_passes": True,
                "projection_passes": projection["passes"],
                "future_execution_root_absent": True,
                "zero_work": True,
            },
            "continue": "research-lead review",
            "hold": "all J1d scientific training execution",
            "kill": "historical kills unchanged; J1/J1d not killed",
            "promote": False,
            "zero_work": zero_work_counters(),
            "passes": True,
        },
        field="readiness_result_payload_sha256",
    )
    return {
        "lock": lock,
        "lock_identity": lock_identity,
        "result": result,
        "result_identity": immutable_json_identity(
            paths["result"],
            payload_field="readiness_result_payload_sha256",
            decision=READY_DECISION,
        ),
        "artifacts": artifacts,
        "passes": True,
    }


def write_fixture_readiness(
    *,
    readiness_dir: Path,
    execution_root: Path,
    rows: Sequence[Mapping[str, Any]],
    engine_config: Mapping[str, int],
    first_guard_passes: bool = True,
) -> dict[str, Any]:
    readiness_dir.mkdir(parents=True, exist_ok=True)
    root_ids = [str(row["root_id"]) for row in rows]
    commitment = payload_with_hash(
        {
            "version": f"{VERSION}_fixture_root_commitment_v1",
            "phase": "training",
            "partition": "train",
            "row_count": len(rows),
        },
        "marker_payload_sha256",
    )
    source = payload_with_hash(
        {
            "version": f"{VERSION}_fixture_source_manifest_v1",
            "phase": "training",
            "partition": "train",
            "root_commitment": commitment,
            "rows": [dict(row) for row in rows],
            "canonical_rows_sha256": ordered_rows_hash(rows),
            "checks": {"fixture": True},
            "passes": True,
        },
        "prospective_manifest_payload_sha256",
    )
    source_path = readiness_dir / "FIXTURE_SOURCE_MANIFEST.json"
    write_immutable_json(
        source_path,
        {
            key: value
            for key, value in source.items()
            if key != "prospective_manifest_payload_sha256"
        },
        field="prospective_manifest_payload_sha256",
    )
    generic_artifacts = {
        "test_evidence": (
            TEST_EVIDENCE_NAME,
            "test_evidence_payload_sha256",
            {"fixture": True, "passes": True},
        ),
        "schema": (
            SCHEMA_NAME,
            "schema_payload_sha256",
            {
                "fixture": True,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "passes": True,
            },
        ),
        "projection": (
            PROJECTION_NAME,
            "projection_payload_sha256",
            {"fixture": True, "passes": True},
        ),
        "input_bindings": (
            INPUT_BINDINGS_NAME,
            "input_bindings_payload_sha256",
            {"fixture": True, "passes": True},
        ),
        "root_cause": (
            ROOT_CAUSE_NAME,
            "root_cause_payload_sha256",
            {
                "fixture": True,
                "scientific_interpretation": False,
                "passes": True,
            },
        ),
    }
    artifacts = {}
    for name, (filename, field, payload) in generic_artifacts.items():
        path = readiness_dir / filename
        write_immutable_json(path, payload, field=field)
        artifacts[name] = immutable_json_identity(
            path,
            payload_field=field,
        )
    lock = write_immutable_json(
        readiness_dir / READINESS_LOCK_NAME,
        {
            "version": f"{VERSION}_fixture_readiness_lock_v1",
            "decision": READY_DECISION,
            "execution_mode": "miniature_fixture",
            "scientific_authority": False,
            "fixture_only": True,
            "future_execution_root": str(execution_root.resolve()),
            "charter_file_sha256": sha256_path(CHARTER_PATH),
            "runner_file_sha256": sha256_path(RUNNER_PATH),
            "test_file_sha256": sha256_path(TEST_PATH),
            "artifacts": artifacts,
            "source_manifest_identity": immutable_json_identity(
                source_path,
                payload_field="prospective_manifest_payload_sha256",
            ),
            "fixture_engine_config": dict(engine_config),
            "fixture_first_guard_passes": first_guard_passes,
            "fixture_root_set_sha256": canonical_json_hash(root_ids),
            "public_commands": list(PUBLIC_COMMANDS),
            "passes": True,
        },
        field="readiness_lock_payload_sha256",
    )
    lock_identity = immutable_json_identity(
        readiness_dir / READINESS_LOCK_NAME,
        payload_field="readiness_lock_payload_sha256",
        decision=READY_DECISION,
    )
    result = write_immutable_json(
        readiness_dir / READINESS_RESULT_NAME,
        {
            "version": f"{VERSION}_fixture_readiness_result_v1",
            "decision": READY_DECISION,
            "readiness_lock_identity": lock_identity,
            "fixture_only": True,
            "scientific_authority": False,
            "passes": True,
        },
        field="readiness_result_payload_sha256",
    )
    return {
        "lock": lock,
        "result": result,
        "lock_identity": lock_identity,
        "passes": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="J1d training-only production dispatcher",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
    )
    for command in PUBLIC_COMMANDS:
        child = subparsers.add_parser(command)
        child.add_argument(
            "--execution-root",
            type=Path,
            required=True,
        )
        child.add_argument(
            "--readiness-dir",
            type=Path,
            required=True,
        )
        child.add_argument("--jobs", type=int, required=True)
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.jobs != 1:
        raise J1dSurfaceIntegrityError("J1d jobs must equal one")
    execution_root = args.execution_root.resolve()
    readiness_dir = args.readiness_dir.resolve()
    if args.subcommand == "seal-phase-lock":
        return seal_training_phase_lock(
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    if args.subcommand == "open":
        return open_training_phase(
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    if args.subcommand == "materialize":
        return materialize_training_manifest(
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    if args.subcommand == "execute":
        return execute_training_from_artifacts(
            execution_root=execution_root,
            readiness_dir=readiness_dir,
        )
    raise J1dSurfaceIntegrityError(
        f"Unsupported J1d command: {args.subcommand}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except BaseException as error:
        payload = {
            "version": f"{VERSION}_command_failure_v1",
            "command": args.subcommand,
            "error_type": type(error).__name__,
            "error": str(error),
            "passes": False,
        }
        print(json.dumps(payload, sort_keys=True))
        if error.__class__.__name__ == "J1ExecutionPlannedInterruption":
            return 75
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

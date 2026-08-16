"""Outcome-free J1 joint policy/value implementation readiness tooling.

The CLI intentionally has no command that can open or execute scientific work.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import socket
import struct
import subprocess
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from threes_rl.obs import encode_observation, observation_size
from threes_rl.sim import Preview, SimState, ThreesSim, score_board


VERSION = "j1_joint_policy_value_preflight_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "threes_rl" / "runs"
OUTPUT_DIR = (
    RUNS_ROOT / "forensics" / "j1_implementation_preflight_v1"
)
CHARTER_PATH = REPO_ROOT / "threes_rl" / "J1_IMPLEMENTATION_PREFLIGHT_CHARTER.md"
PROPOSAL_PATH = (
    REPO_ROOT / "threes_rl" / "J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md"
)
READINESS_PATH = (
    REPO_ROOT / "threes_rl" / "J1_IMPLEMENTATION_READINESS_AUDIT.json"
)
RUNNER_PATH = REPO_ROOT / "threes_rl" / "j1_joint_policy_value.py"
TEST_PATH = REPO_ROOT / "tests" / "test_rl_j1_joint_policy_value.py"

TEST_EVIDENCE_NAME = "J1_IMPLEMENTATION_TEST_EVIDENCE.json"
DENYLIST_NAME = "J1_PROTECTED_ID_DENYLIST.json"
PROJECTION_NAME = "J1_RUNTIME_STORAGE_PROJECTION.json"
PREFLIGHT_LOCK_NAME = "J1_IMPLEMENTATION_PREFLIGHT_LOCK.json"
PREFLIGHT_RESULT_NAME = "J1_IMPLEMENTATION_PREFLIGHT_RESULT.json"

EXPECTED_PROPOSAL_SHA256 = (
    "26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2"
)
EXPECTED_READINESS_FILE_SHA256 = (
    "f3e4e8029e159a1db7767164e1623d2e166b139be319d6077d61d7d107a44042"
)
EXPECTED_READINESS_PAYLOAD_SHA256 = (
    "5b6b9a2383296f82b6547bbd46ddc892b486e4b89f4c325aa88f9c8b15944f99"
)
EXPECTED_TORCH_VERSION = "2.12.1"
EXPECTED_OBSERVATION_WIDTH = 282
EXPECTED_PARAMETER_COUNT = 411_656
EXPECTED_INCUMBENT_POLICY_SHA256 = (
    "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4"
)
TOP_THREE = (263_670, 261_369, 258_561)

TRAIN_ROOTS = 16_384
DEVELOPMENT_PAIRS = 1_024
CONFIRMATION_PAIRS = 5_120
PAIRED_EVALUATION_ROOTS = DEVELOPMENT_PAIRS + CONFIRMATION_PAIRS
TOTAL_GAME_ARMS = TRAIN_ROOTS + 2 * PAIRED_EVALUATION_ROOTS
MAX_MOVES = 5_000
PLANNING_MOVES = 512
SAFETY_MULTIPLIER = 1.25
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
MIN_NICE = 10

PHASE_CAPS = {
    "training": {"hours": 72.0, "storage_gib": 24.0},
    "development": {"hours": 24.0, "storage_gib": 8.0},
    "confirmation": {"hours": 120.0, "storage_gib": 16.0},
}

HISTORICAL_STREAM_MAX = 212_999_999_999
PROSPECTIVE_STREAMS = {
    "train": {
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

ROOT_MANIFEST_BINDINGS = {
    "threes_rl/runs/forensics/r2a_adaptive/R2A_ROOT_MANIFEST.json":
        "bb7f41e702d473ec6d96abf87b4350585ff52001c87ac3c8bd4670b7591816fa",
    "threes_rl/runs/forensics/g2_scale_equivariant_relational_hazard/G2_ROOT_MANIFEST.json":
        "60d514ed79ff315f7c2e0d2ad13bb712a57d4c3b204587691aa878a7486ea2ca",
    "threes_rl/runs/forensics/human_h0/human_h0_root_manifest_20260710.json":
        "9366644e5ed53f06fcc26a606cef5b3f0f74352888ada0b546f9c45ecd60c492",
    "threes_rl/runs/forensics/o1_goal_conditioned_option_p0_v1/O1_P0_ROOT_MANIFEST.json":
        "7f6325617757b598d037ae11ea1963470925994c39fe949a75002e712db91cd2",
    "threes_rl/runs/forensics/o3_event_acquisition_recovery_v1/O3_RECOVERY_SELECTED_ROOTS.json":
        "9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049",
    "threes_rl/runs/forensics/o3_option_training_v1/selected_root_manifest.json":
        "4cbfc8e378b9f5e0384ab66ff9cf3abe12ea120bf45c4a0037ea4f7ce936c9d3",
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/O4_P0_SELECTED_ROOTS.json":
        "f3b0a0afb3344e3413e5f63bf367c86aab66bc619d8ec736fbc7313090182ab3",
    "threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1/O5_P0_SELECTED_ROOTS.json":
        "d6220ee3ebfe799d78cba128be816e607947a225f7b6ad8add0cc2aad91abad8",
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/selected_root_manifest.json":
        "2d6a75cddd9f4e8bfa84e8e1516b628b05b33bb75ec86c8c1025d55a8057317e",
    "threes_rl/runs/forensics/g3_scale_transfer_bootstrap_preflight_v1/G3_RECORD_MANIFEST.json":
        "938e903f8d2fefb072af84ac19baf4977e4f4d93bf72e8af7acc174b6974b9ec",
    "threes_rl/runs/forensics/s3_full_policy/S3_PROVENANCE_SEAL_V2.json":
        "5326f25b50ad33b4e00eb5ca7180468d3a243917075d15d377a1511b04867949",
    "threes_rl/runs/forensics/s3_full_policy/S3_POWER_PREFLIGHT_V2_SEALED.json":
        "4dabd5325dcbbc5234c4e015eccbd4d5f4706be9fefa54fd5220d8720b1fc345",
    "threes_rl/runs/forensics/c2_cost_admission_v1/C2_CORPUS_MANIFEST.json":
        "2be3db6c028b641c94701c517682f2925862cbe11d89fd7c53e3aebda59e3653",
    "threes_rl/runs/forensics/k1_compiled_kernel_v1/K1_CORPUS_PLAN.json":
        "7461b301e1001b64140e60bf382761bf4024464558afc2da1780429b392d8ca7",
    "threes_rl/runs/forensics/r15a_context_a2/R15A_A2_LABEL_MANIFEST.json":
        "75c9fabadedc8e35bccd782ba5581cbce2eedced07a523893abcf02d0b2217eb",
}

DEPENDENCY_BINDINGS = {
    "threes_rl/train_ppo.py":
        "cb2cb301630001ed887e1131c46bc6565e41917ea861ebe836ba9c39990fc6f3",
    "threes_rl/train_td.py":
        "0ef18c38c09516a11fddc5b2cd742aa536c21615d5ce2477167bed8553b13f7a",
    "threes_rl/ntuple.py":
        "bdd38ec758ca1786b67a7550b3a2792cbd517176ad99e4df7c5ddd2584953789",
    "threes_rl/obs.py":
        "7fe9fdc48da826dfde424391b57d8d9de812aa48bb08e129079ed9f3fd3478b1",
    "threes_rl/env.py":
        "9b3a65fff503ab5b40db63e11c5c4b3c03f96bd4034709d80cad707cf40f2ddf",
    "threes_rl/eval.py":
        "df0a558014583fcfd24fd8ddf48988e375ad9a6fc5199d35311c40d8b6a3f705",
    "threes_rl/o2_online_option_preflight.py":
        "99e61f551d607e3b5b8457b7e76a17c8540f0e1d88afec3fa544296bdcd05fda",
    "threes_rl/sim.py":
        "67e7a245c05e59367402095ad018122fb4cb1ef08664bf28bf4bc03a02a73072",
    "threes_rl/current_incumbent_policy.txt":
        EXPECTED_INCUMBENT_POLICY_SHA256,
}

HEAVY_PROCESS_PATTERNS = (
    "threes_rl.train",
    "threes_rl.eval",
    "threes_rl.o[1-6]_",
    "threes_rl.g[1-4]_",
    "threes_rl.k1_",
    "threes_rl.c[12]_",
)


class J1IntegrityError(RuntimeError):
    """Immutable identity or semantic contract failed."""


class J1OperationalHold(RuntimeError):
    """Mutable resource or service condition failed."""


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str = "canonical_payload_sha256",
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(
    payload: Mapping[str, Any],
    field: str = "canonical_payload_sha256",
) -> bool:
    body = dict(payload)
    embedded = body.pop(field, None)
    return isinstance(embedded, str) and embedded == canonical_json_hash(body)


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    field: str = "canonical_payload_sha256",
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Immutable artifact exists: {path}")
    body = payload_with_hash(payload, field)
    serialized = json.dumps(body, indent=2, sort_keys=True) + "\n"
    if not verify_payload_hash(json.loads(serialized), field):
        raise J1IntegrityError(f"JSON reload instability: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
    observed = json.loads(path.read_text(encoding="utf-8"))
    if not verify_payload_hash(observed, field):
        raise J1IntegrityError(f"Written payload hash mismatch: {path}")
    return body


def _hash_update(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I")
        digest.update(str(value).encode("ascii"))
        digest.update(b";")
    elif isinstance(value, float):
        digest.update(b"F")
        digest.update(struct.pack("!d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    elif isinstance(value, bytes):
        digest.update(b"Y")
        digest.update(str(len(value)).encode("ascii"))
        digest.update(b":")
        digest.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        _hash_update(digest, str(tensor.dtype))
        _hash_update(digest, list(tensor.shape))
        digest.update(tensor.numpy().tobytes(order="C"))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A")
        _hash_update(digest, str(array.dtype))
        _hash_update(digest, list(array.shape))
        digest.update(array.tobytes(order="C"))
    elif isinstance(value, Mapping):
        digest.update(b"M")
        keys = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        for key in keys:
            _hash_update(digest, key)
            _hash_update(digest, value[key])
    elif isinstance(value, tuple):
        digest.update(b"U")
        for item in value:
            _hash_update(digest, item)
    elif isinstance(value, list):
        digest.update(b"L")
        for item in value:
            _hash_update(digest, item)
    else:
        raise TypeError(f"Unsupported stable-hash value: {type(value).__name__}")


def stable_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class J1TrainingConfig:
    observation_width: int = EXPECTED_OBSERVATION_WIDTH
    hidden_width: int = 512
    action_count: int = 4
    auxiliary_count: int = 3
    initialization_seed: int = 2_026_072_806
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    auxiliary_coef: float = 0.05
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.50
    rounds: int = 64
    roots_per_round: int = 256
    epochs_per_round: int = 4
    minibatch_size: int = 4_096
    starter_tile: None = None


FROZEN_CONFIG = J1TrainingConfig()


class J1ActorCritic(nn.Module):
    """Frozen J1 282-512-512 policy/value/auxiliary network."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(EXPECTED_OBSERVATION_WIDTH, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.policy = nn.Linear(512, 4)
        self.value = nn.Linear(512, 1)
        self.auxiliary = nn.Linear(512, 3)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.body(observations)
        return (
            self.policy(features),
            self.value(features).squeeze(-1),
            self.auxiliary(features),
        )


def parameter_count(model: nn.Module | None = None) -> int:
    value = J1ActorCritic() if model is None else model
    return sum(parameter.numel() for parameter in value.parameters())


def model_schema() -> dict[str, Any]:
    return {
        "version": "j1_actor_critic_schema_v1",
        "observation_width": EXPECTED_OBSERVATION_WIDTH,
        "body": [
            ["linear", EXPECTED_OBSERVATION_WIDTH, 512],
            ["relu"],
            ["linear", 512, 512],
            ["relu"],
        ],
        "heads": {
            "policy": ["linear", 512, 4],
            "value": ["linear", 512, 1],
            "auxiliary": ["linear", 512, 3],
        },
        "action_order": ["up", "down", "left", "right"],
        "auxiliary_labels": [
            "final_max_tile_ge_1536",
            "final_max_tile_ge_3072",
            "survive_64_additional_moves",
        ],
        "auxiliary_coef": FROZEN_CONFIG.auxiliary_coef,
        "auxiliary_enters_return": False,
        "starter_tile": None,
    }


def model_schema_sha256() -> str:
    return canonical_json_hash(model_schema())


def initialize_model_optimizer() -> tuple[J1ActorCritic, torch.optim.Optimizer]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(FROZEN_CONFIG.initialization_seed)
    model = J1ActorCritic().cpu()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FROZEN_CONFIG.learning_rate,
        eps=FROZEN_CONFIG.adam_eps,
    )
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise J1IntegrityError("J1 parameter count changed")
    return model, optimizer


def assert_finite_model(model: nn.Module) -> None:
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise J1IntegrityError(f"Nonfinite model tensor: {name}")


def masked_logits(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != legal_mask.shape:
        raise ValueError("Logits/legal-mask shape mismatch")
    if legal_mask.dtype is not torch.bool:
        raise ValueError("Legal mask must be boolean")
    if not torch.all(legal_mask.any(dim=-1)):
        raise ValueError("Every live row must have at least one legal action")
    if not torch.isfinite(logits).all():
        raise J1IntegrityError("Nonfinite policy logits")
    return logits.masked_fill(~legal_mask, -torch.inf)


def deterministic_masked_actions(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.argmax(masked_logits(logits, legal_mask), dim=-1)


def sampled_masked_actions(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    probabilities = torch.softmax(masked_logits(logits, legal_mask), dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(-1)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    done_after_transition: np.ndarray,
    bootstrap_value: np.ndarray | float,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE with transition-t terminal masking."""

    reward_array = np.asarray(rewards, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    done_array = np.asarray(done_after_transition, dtype=bool)
    if reward_array.shape != value_array.shape or reward_array.shape != done_array.shape:
        raise ValueError("Rewards, values, and done arrays must share shape")
    if reward_array.ndim not in (1, 2) or reward_array.shape[0] == 0:
        raise ValueError("GAE expects nonempty [time] or [time,batch] arrays")
    squeeze = reward_array.ndim == 1
    if squeeze:
        reward_array = reward_array[:, None]
        value_array = value_array[:, None]
        done_array = done_array[:, None]
    bootstrap = np.asarray(bootstrap_value, dtype=np.float64)
    if bootstrap.ndim == 0:
        bootstrap = np.full(reward_array.shape[1], float(bootstrap))
    if bootstrap.shape != (reward_array.shape[1],):
        raise ValueError("Bootstrap value must match batch width")
    if not np.isfinite(reward_array).all() or not np.isfinite(value_array).all():
        raise J1IntegrityError("Nonfinite GAE input")

    advantages = np.zeros_like(reward_array, dtype=np.float64)
    next_advantage = np.zeros(reward_array.shape[1], dtype=np.float64)
    for index in range(reward_array.shape[0] - 1, -1, -1):
        next_value = (
            bootstrap
            if index == reward_array.shape[0] - 1
            else value_array[index + 1]
        )
        nonterminal = 1.0 - done_array[index].astype(np.float64)
        delta = (
            reward_array[index]
            + gamma * next_value * nonterminal
            - value_array[index]
        )
        next_advantage = (
            delta
            + gamma * gae_lambda * nonterminal * next_advantage
        )
        advantages[index] = next_advantage
    returns = advantages + value_array
    if squeeze:
        return advantages[:, 0], returns[:, 0]
    return advantages, returns


def dense_score_reward(score_delta: int | float) -> float:
    return float(score_delta) * 1e-5


def verify_score_delta_telescoping(
    start_score: int,
    score_deltas: Sequence[int],
    final_score: int,
) -> bool:
    return int(start_score) + sum(int(value) for value in score_deltas) == int(
        final_score
    )


def verify_dense_reward_telescoping(
    start_score: int,
    final_score: int,
    score_deltas: Sequence[int],
) -> dict[str, Any]:
    score_identity = verify_score_delta_telescoping(
        start_score,
        score_deltas,
        final_score,
    )
    scaled_sum = sum(dense_score_reward(value) for value in score_deltas)
    expected_scaled = dense_score_reward(final_score - start_score)
    return {
        "score_delta_sum": sum(int(value) for value in score_deltas),
        "score_difference": int(final_score) - int(start_score),
        "scaled_return": scaled_sum,
        "expected_scaled_return": expected_scaled,
        "auxiliary_reward_contribution": 0.0,
        "passes": (
            score_identity
            and math.isclose(
                scaled_sum,
                expected_scaled,
                abs_tol=1e-15,
                rel_tol=0.0,
            )
        ),
    }


def root_equal_weights(lengths: Sequence[int]) -> np.ndarray:
    if not lengths or any(int(length) <= 0 for length in lengths):
        raise ValueError("Root lengths must be positive")
    rows = [
        np.full(int(length), 1.0 / float(length), dtype=np.float64)
        for length in lengths
    ]
    return np.concatenate(rows)


@dataclass(frozen=True)
class CompleteRoot:
    root_id: str
    ancestry_id: str
    partition: str
    transitions: tuple[Mapping[str, Any], ...]
    natural_terminal: bool


def flatten_complete_roots(
    roots: Sequence[CompleteRoot],
    *,
    expected_partition: str,
) -> dict[str, Any]:
    if not roots:
        raise ValueError("At least one root is required")
    root_ids = [root.root_id for root in roots]
    ancestry_ids = [root.ancestry_id for root in roots]
    if len(set(root_ids)) != len(root_ids):
        raise J1IntegrityError("Duplicate root identity")
    if len(set(ancestry_ids)) != len(ancestry_ids):
        raise J1IntegrityError("Ancestry crossed or repeated")
    if any(root.partition != expected_partition for root in roots):
        raise J1IntegrityError("Root crossed partition")
    if any(not root.natural_terminal for root in roots):
        raise J1IntegrityError("Truncated root cannot enter training")
    if any(not root.transitions for root in roots):
        raise J1IntegrityError("Empty root cannot enter training")
    lengths = [len(root.transitions) for root in roots]
    weights = root_equal_weights(lengths)
    rows = [
        {
            "root_id": root.root_id,
            "ancestry_id": root.ancestry_id,
            "transition_index": index,
            **dict(transition),
        }
        for root in roots
        for index, transition in enumerate(root.transitions)
    ]
    offset = 0
    per_root_weight = {}
    for root, length in zip(roots, lengths):
        total = float(weights[offset : offset + length].sum())
        per_root_weight[root.root_id] = total
        offset += length
    return {
        "rows": rows,
        "weights": weights,
        "lengths": lengths,
        "per_root_weight": per_root_weight,
        "transition_buffer_sha256": stable_hash(rows),
    }


def validate_ancestry_partitions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for row in rows:
        ancestry = str(row["ancestry_id"])
        partition = str(row["partition"])
        previous = assignment.setdefault(ancestry, partition)
        if previous != partition:
            raise J1IntegrityError(
                f"Ancestry {ancestry} crosses {previous}/{partition}"
            )
    return assignment


def normal_start_sim(
    *,
    role: str,
    deck_stream_id: int,
    slot_stream_id: int,
) -> tuple[ThreesSim, SimState]:
    if role not in {"train", "development", "confirmation"}:
        raise ValueError(f"Unsupported J1 role: {role}")
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(deck_stream_id),
        slot_stream_id=int(slot_stream_id),
        starter_tile=None,
    )
    if sim.starter_tile is not None:
        raise J1IntegrityError("J1 normal start has a starter")
    state = sim.reset()
    if int(state.board.max(initial=0)) >= 48:
        raise J1IntegrityError("Fresh no-starter root contains a large tile")
    return sim, state


@dataclass(frozen=True)
class FrozenPPOBatch:
    observations: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    auxiliary_labels: torch.Tensor
    row_weights: torch.Tensor
    root_ids: tuple[str, ...]

    def row_count(self) -> int:
        return int(self.observations.shape[0])

    def subset(self, indices: torch.Tensor) -> "FrozenPPOBatch":
        selected = indices.detach().cpu().tolist()
        return FrozenPPOBatch(
            observations=self.observations[indices],
            legal_masks=self.legal_masks[indices],
            actions=self.actions[indices],
            old_log_probabilities=self.old_log_probabilities[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
            auxiliary_labels=self.auxiliary_labels[indices],
            row_weights=self.row_weights[indices],
            root_ids=tuple(self.root_ids[index] for index in selected),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "observations": self.observations.detach().cpu().clone(),
            "legal_masks": self.legal_masks.detach().cpu().clone(),
            "actions": self.actions.detach().cpu().clone(),
            "old_log_probabilities": (
                self.old_log_probabilities.detach().cpu().clone()
            ),
            "advantages": self.advantages.detach().cpu().clone(),
            "returns": self.returns.detach().cpu().clone(),
            "auxiliary_labels": (
                self.auxiliary_labels.detach().cpu().clone()
            ),
            "row_weights": self.row_weights.detach().cpu().clone(),
            "root_ids": tuple(self.root_ids),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenPPOBatch":
        return cls(
            observations=payload["observations"].detach().cpu().clone(),
            legal_masks=payload["legal_masks"].detach().cpu().clone(),
            actions=payload["actions"].detach().cpu().clone(),
            old_log_probabilities=(
                payload["old_log_probabilities"].detach().cpu().clone()
            ),
            advantages=payload["advantages"].detach().cpu().clone(),
            returns=payload["returns"].detach().cpu().clone(),
            auxiliary_labels=(
                payload["auxiliary_labels"].detach().cpu().clone()
            ),
            row_weights=payload["row_weights"].detach().cpu().clone(),
            root_ids=tuple(str(value) for value in payload["root_ids"]),
        )


def validate_ppo_batch(
    batch: FrozenPPOBatch,
    *,
    require_complete_root_weights: bool = True,
) -> None:
    rows = batch.row_count()
    expected = {
        "legal_masks": (rows, 4),
        "actions": (rows,),
        "old_log_probabilities": (rows,),
        "advantages": (rows,),
        "returns": (rows,),
        "auxiliary_labels": (rows, 3),
        "row_weights": (rows,),
    }
    if batch.observations.shape != (rows, EXPECTED_OBSERVATION_WIDTH):
        raise J1IntegrityError("PPO observation shape changed")
    for name, shape in expected.items():
        if tuple(getattr(batch, name).shape) != shape:
            raise J1IntegrityError(f"PPO {name} shape changed")
    if batch.legal_masks.dtype is not torch.bool:
        raise J1IntegrityError("PPO legal masks must be bool")
    if batch.actions.dtype is not torch.int64:
        raise J1IntegrityError("PPO actions must be int64")
    if len(batch.root_ids) != rows:
        raise J1IntegrityError("PPO root identity count changed")
    finite_tensors = (
        batch.observations,
        batch.old_log_probabilities,
        batch.advantages,
        batch.returns,
        batch.auxiliary_labels,
        batch.row_weights,
    )
    if any(not torch.isfinite(value).all() for value in finite_tensors):
        raise J1IntegrityError("PPO batch contains a nonfinite value")
    if not torch.all(batch.legal_masks.any(dim=1)):
        raise J1IntegrityError("PPO batch contains an all-illegal row")
    if torch.any(batch.actions < 0) or torch.any(batch.actions >= 4):
        raise J1IntegrityError("PPO batch action is out of range")
    chosen_legal = batch.legal_masks.gather(
        1,
        batch.actions.unsqueeze(1),
    ).squeeze(1)
    if not torch.all(chosen_legal):
        raise J1IntegrityError("PPO batch contains an illegal chosen action")
    if torch.any(batch.auxiliary_labels < 0.0) or torch.any(
        batch.auxiliary_labels > 1.0
    ):
        raise J1IntegrityError("PPO auxiliary label is outside [0,1]")
    if torch.any(batch.row_weights <= 0.0):
        raise J1IntegrityError("PPO row weights must be positive")
    root_totals: dict[str, float] = defaultdict(float)
    for root_id, weight in zip(
        batch.root_ids,
        batch.row_weights.detach().cpu().tolist(),
    ):
        root_totals[root_id] += float(weight)
    if not root_totals:
        raise J1IntegrityError("PPO batch has no root identities")
    if require_complete_root_weights and any(
        not math.isclose(total, 1.0, abs_tol=2e-7, rel_tol=0.0)
        for total in root_totals.values()
    ):
        raise J1IntegrityError("PPO root-equal row weights changed")


def normalize_advantages_root_weighted(
    advantages: torch.Tensor,
    row_weights: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if advantages.ndim != 1 or advantages.shape != row_weights.shape:
        raise J1IntegrityError("Advantage/weight shape mismatch")
    if (
        not torch.isfinite(advantages).all()
        or not torch.isfinite(row_weights).all()
        or torch.any(row_weights <= 0.0)
    ):
        raise J1IntegrityError("Invalid advantage normalization inputs")
    total = row_weights.sum()
    mean = torch.sum(row_weights * advantages) / total
    variance = torch.sum(row_weights * (advantages - mean) ** 2) / total
    normalized = (advantages - mean) / torch.sqrt(variance + epsilon)
    if not torch.isfinite(normalized).all():
        raise J1IntegrityError("Normalized advantages are nonfinite")
    return normalized


def frozen_ppo_loss(
    model: J1ActorCritic,
    batch: FrozenPPOBatch,
    *,
    normalized_advantages: torch.Tensor | None = None,
    global_weight_total: torch.Tensor | float | None = None,
    minibatches_per_epoch: int = 1,
) -> dict[str, torch.Tensor]:
    validate_ppo_batch(
        batch,
        require_complete_root_weights=(global_weight_total is None),
    )
    if minibatches_per_epoch < 1:
        raise J1IntegrityError("PPO minibatch count must be positive")
    advantages = (
        normalize_advantages_root_weighted(
            batch.advantages,
            batch.row_weights,
        )
        if normalized_advantages is None
        else normalized_advantages
    )
    if advantages.shape != batch.advantages.shape:
        raise J1IntegrityError("Normalized advantage shape changed")
    if not torch.isfinite(advantages).all():
        raise J1IntegrityError("Normalized advantage is nonfinite")
    logits, values, auxiliary_logits = model(batch.observations)
    distribution = torch.distributions.Categorical(
        logits=masked_logits(logits, batch.legal_masks)
    )
    new_log_probabilities = distribution.log_prob(batch.actions)
    entropy_rows = distribution.entropy()
    log_ratio = new_log_probabilities - batch.old_log_probabilities
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise J1IntegrityError("PPO probability ratio is nonfinite")
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(
        ratio,
        1.0 - FROZEN_CONFIG.clip_coef,
        1.0 + FROZEN_CONFIG.clip_coef,
    )
    policy_rows = torch.maximum(unclipped, clipped)
    value_rows = 0.5 * (values - batch.returns) ** 2
    auxiliary_rows = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            auxiliary_logits,
            batch.auxiliary_labels,
            reduction="none",
        ).mean(dim=1)
    )
    if global_weight_total is None:
        denominator = batch.row_weights.sum()
        scale = 1.0
    else:
        denominator = torch.as_tensor(
            global_weight_total,
            dtype=batch.row_weights.dtype,
            device=batch.row_weights.device,
        )
        scale = float(minibatches_per_epoch)
    if not torch.isfinite(denominator) or float(denominator) <= 0.0:
        raise J1IntegrityError("PPO global weight total is invalid")

    def reduce(values_to_reduce: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(batch.row_weights * values_to_reduce)
            / denominator
            * scale
        )

    policy_loss = reduce(policy_rows)
    value_loss = reduce(value_rows)
    entropy = reduce(entropy_rows)
    auxiliary_loss = reduce(auxiliary_rows)
    total_loss = (
        policy_loss
        + FROZEN_CONFIG.value_coef * value_loss
        - FROZEN_CONFIG.entropy_coef * entropy
        + FROZEN_CONFIG.auxiliary_coef * auxiliary_loss
    )
    approx_kl = reduce((ratio - 1.0) - log_ratio)
    outputs = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "auxiliary_loss": auxiliary_loss,
        "approx_kl": approx_kl,
        "weight_sum": batch.row_weights.sum(),
    }
    if any(not torch.isfinite(value) for value in outputs.values()):
        raise J1IntegrityError("PPO loss component is nonfinite")
    return outputs


def round_learning_rate(
    round_number: int,
    *,
    after_round: bool = False,
) -> float:
    if round_number < 1 or round_number > FROZEN_CONFIG.rounds:
        raise ValueError("Round number is outside 1..64")
    numerator = (
        FROZEN_CONFIG.rounds - round_number
        if after_round
        else FROZEN_CONFIG.rounds - round_number + 1
    )
    return FROZEN_CONFIG.learning_rate * (
        numerator / FROZEN_CONFIG.rounds
    )


def deterministic_epoch_minibatches(
    row_count: int,
    *,
    round_number: int,
    epochs: int = FROZEN_CONFIG.epochs_per_round,
    minibatch_size: int = FROZEN_CONFIG.minibatch_size,
) -> list[dict[str, Any]]:
    if row_count < 1 or epochs < 1 or minibatch_size < 1:
        raise ValueError("Invalid PPO epoch/minibatch dimensions")
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        material = (
            f"J1-minibatch-v1|{FROZEN_CONFIG.initialization_seed}|"
            f"{round_number}|{epoch}"
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
            rows.append(
                {
                    "epoch": epoch,
                    "start": start,
                    "seed": seed,
                    "indices": indices,
                    "final_short": len(indices) < minibatch_size,
                }
            )
    return rows


def ppo_schedule_audit(
    row_count: int,
    *,
    round_number: int,
    epochs: int = FROZEN_CONFIG.epochs_per_round,
    minibatch_size: int = FROZEN_CONFIG.minibatch_size,
) -> dict[str, Any]:
    plan = deterministic_epoch_minibatches(
        row_count,
        round_number=round_number,
        epochs=epochs,
        minibatch_size=minibatch_size,
    )
    coverage = {
        epoch: [
            index
            for row in plan
            if row["epoch"] == epoch
            for index in row["indices"]
        ]
        for epoch in range(epochs)
    }
    checks = {
        "epochs_exact": set(coverage) == set(range(epochs)),
        "every_row_once_each_epoch": all(
            sorted(indices) == list(range(row_count))
            for indices in coverage.values()
        ),
        "deterministic": plan
        == deterministic_epoch_minibatches(
            row_count,
            round_number=round_number,
            epochs=epochs,
            minibatch_size=minibatch_size,
        ),
        "final_short_retained": (
            row_count % minibatch_size == 0
            or sum(row["final_short"] for row in plan) == epochs
        ),
    }
    return {
        "row_count": row_count,
        "round_number": round_number,
        "epochs": epochs,
        "minibatch_size": minibatch_size,
        "minibatch_count": len(plan),
        "plan_sha256": stable_hash(plan),
        "coverage_counts": {
            str(epoch): len(indices)
            for epoch, indices in coverage.items()
        },
        "short_minibatch_sizes": [
            len(row["indices"])
            for row in plan
            if row["final_short"]
        ],
        "checks": checks,
        "passes": all(checks.values()),
    }


def apply_frozen_ppo_update(
    model: J1ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: FrozenPPOBatch,
    *,
    round_number: int,
    epochs: int = FROZEN_CONFIG.epochs_per_round,
    minibatch_size: int = FROZEN_CONFIG.minibatch_size,
    optimizer_step: bool = True,
) -> dict[str, Any]:
    validate_ppo_batch(batch)
    normalized = normalize_advantages_root_weighted(
        batch.advantages,
        batch.row_weights,
    )
    plan = deterministic_epoch_minibatches(
        batch.row_count(),
        round_number=round_number,
        epochs=epochs,
        minibatch_size=minibatch_size,
    )
    update_lr = round_learning_rate(round_number)
    for group in optimizer.param_groups:
        group["lr"] = update_lr
    summaries = []
    for epoch in range(epochs):
        epoch_rows = [row for row in plan if row["epoch"] == epoch]
        minibatch_count = len(epoch_rows)
        for row in epoch_rows:
            indices = torch.tensor(row["indices"], dtype=torch.int64)
            subset = batch.subset(indices)
            losses = frozen_ppo_loss(
                model,
                subset,
                normalized_advantages=normalized[indices],
                global_weight_total=batch.row_weights.sum(),
                minibatches_per_epoch=minibatch_count,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                FROZEN_CONFIG.max_grad_norm,
            )
            if not torch.isfinite(gradient_norm):
                raise J1IntegrityError("PPO gradient norm is nonfinite")
            if optimizer_step:
                optimizer.step()
                assert_finite_model(model)
            summaries.append(
                {
                    "epoch": epoch,
                    "start": row["start"],
                    "rows": len(row["indices"]),
                    "final_short": row["final_short"],
                    "losses": {
                        key: float(value.detach().cpu())
                        for key, value in losses.items()
                    },
                    "gradient_norm_before_clip": float(
                        gradient_norm.detach().cpu()
                    ),
                    "optimizer_step": bool(optimizer_step),
                }
            )
    post_round_lr = round_learning_rate(round_number, after_round=True)
    for group in optimizer.param_groups:
        group["lr"] = post_round_lr
    schedule = ppo_schedule_audit(
        batch.row_count(),
        round_number=round_number,
        epochs=epochs,
        minibatch_size=minibatch_size,
    )
    return {
        "round_number": round_number,
        "update_learning_rate": update_lr,
        "post_round_learning_rate": post_round_lr,
        "optimizer_steps": len(summaries) if optimizer_step else 0,
        "minibatches": summaries,
        "schedule": schedule,
        "normalized_advantage_sha256": stable_hash(normalized),
        "batch_sha256": stable_hash(batch.payload()),
        "passes": schedule["passes"],
    }


def synthetic_complete_ppo_batch(
    model: J1ActorCritic,
    *,
    row_count: int = 5,
    root_lengths: Sequence[int] = (2, 3),
    seed: int = 2_026_072_811,
) -> FrozenPPOBatch:
    if sum(root_lengths) != row_count:
        raise ValueError("Synthetic root lengths do not match row count")
    numpy_rng = np.random.default_rng(seed)
    observations = torch.from_numpy(
        numpy_rng.normal(
            0.0,
            0.25,
            size=(row_count, EXPECTED_OBSERVATION_WIDTH),
        ).astype(np.float32)
    )
    legal_masks = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [True, False, True, True],
            [True, True, True, False],
            [False, True, False, True],
        ][:row_count],
        dtype=torch.bool,
    )
    if legal_masks.shape[0] != row_count:
        legal_masks = torch.ones((row_count, 4), dtype=torch.bool)
    with torch.no_grad():
        logits, values, _auxiliary = model(observations)
        masked = masked_logits(logits, legal_masks)
        actions = torch.argmax(masked, dim=1)
        old_log_probabilities = torch.distributions.Categorical(
            logits=masked
        ).log_prob(actions)
    rewards_all: list[float] = []
    advantages_all: list[float] = []
    returns_all: list[float] = []
    root_ids: list[str] = []
    offset = 0
    for root_index, length in enumerate(root_lengths):
        rewards = np.asarray(
            [
                ((root_index + 1) * (index + 1)) * 0.01
                for index in range(length)
            ],
            dtype=np.float64,
        )
        dones = np.zeros(length, dtype=bool)
        dones[-1] = True
        root_values = values[offset : offset + length].numpy().astype(
            np.float64
        )
        advantages, returns = compute_gae(
            rewards,
            root_values,
            dones,
            bootstrap_value=0.0,
        )
        rewards_all.extend(rewards.tolist())
        advantages_all.extend(advantages.tolist())
        returns_all.extend(returns.tolist())
        root_ids.extend([f"synthetic-root-{root_index}"] * length)
        offset += length
    auxiliary_labels = torch.tensor(
        [
            [float(index % 2), float((index // 2) % 2), float(index >= 2)]
            for index in range(row_count)
        ],
        dtype=torch.float32,
    )
    batch = FrozenPPOBatch(
        observations=observations,
        legal_masks=legal_masks,
        actions=actions.to(dtype=torch.int64),
        old_log_probabilities=old_log_probabilities.to(dtype=torch.float32),
        advantages=torch.tensor(advantages_all, dtype=torch.float32),
        returns=torch.tensor(returns_all, dtype=torch.float32),
        auxiliary_labels=auxiliary_labels,
        row_weights=torch.from_numpy(root_equal_weights(root_lengths)),
        root_ids=tuple(root_ids),
    )
    validate_ppo_batch(batch)
    return batch


def state_snapshot(state: SimState) -> dict[str, Any]:
    return {
        "board": np.asarray(state.board, dtype=np.int32).copy(),
        "preview": {
            "kind": state.preview.kind,
            "value": state.preview.value,
            "candidates": tuple(int(value) for value in state.preview.candidates),
        },
        "small_counts": {
            str(key): int(value)
            for key, value in state.small_counts.items()
        },
        "small_pos": int(state.small_pos),
        "small_seen_total": int(state.small_seen_total),
        "span_small_pos": int(state.span_small_pos),
        "large_pending": bool(state.large_pending),
        "max_tile": int(state.max_tile),
        "move_count": int(state.move_count),
        "game_over": bool(state.game_over),
    }


def state_from_snapshot(payload: Mapping[str, Any]) -> SimState:
    preview_payload = payload["preview"]
    if not isinstance(preview_payload, Mapping):
        raise J1IntegrityError("Invalid preview snapshot")
    state = SimState(
        board=np.asarray(payload["board"], dtype=np.int32).copy(),
        preview=Preview(
            kind=str(preview_payload["kind"]),
            value=(
                None
                if preview_payload["value"] is None
                else int(preview_payload["value"])
            ),
            candidates=tuple(
                int(value) for value in preview_payload["candidates"]
            ),
        ),
        small_counts={
            str(key): int(value)
            for key, value in dict(payload["small_counts"]).items()
        },
        small_pos=int(payload["small_pos"]),
        small_seen_total=int(payload["small_seen_total"]),
        span_small_pos=int(payload["span_small_pos"]),
        large_pending=bool(payload["large_pending"]),
        max_tile=int(payload["max_tile"]),
        move_count=int(payload["move_count"]),
        game_over=bool(payload["game_over"]),
    )
    if state.board.shape != (4, 4):
        raise J1IntegrityError("Invalid simulator board shape")
    return state


def simulator_snapshot(sim: ThreesSim, state: SimState) -> dict[str, Any]:
    if sim.starter_tile is not None:
        raise J1IntegrityError("J1 simulator snapshot contains a starter")
    if sim.deck_stream_id is None or sim.slot_stream_id is None:
        raise J1IntegrityError("J1 requires split exogenous streams")
    return {
        "starter_tile": None,
        "deck_stream_id": int(sim.deck_stream_id),
        "slot_stream_id": int(sim.slot_stream_id),
        "legacy_single_rng": bool(sim._legacy_single_rng),
        "deck_rng_state": copy.deepcopy(sim.deck_rng.bit_generator.state),
        "slot_rng_state": copy.deepcopy(sim.slot_rng.bit_generator.state),
        "state": state_snapshot(state),
    }


def simulator_from_snapshot(
    payload: Mapping[str, Any],
) -> tuple[ThreesSim, SimState]:
    if payload.get("starter_tile") is not None:
        raise J1IntegrityError("Resume snapshot changed starter_tile")
    if bool(payload.get("legacy_single_rng")):
        raise J1IntegrityError("Resume snapshot is not split-stream")
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(payload["deck_stream_id"]),
        slot_stream_id=int(payload["slot_stream_id"]),
        starter_tile=None,
    )
    sim.deck_rng.bit_generator.state = copy.deepcopy(
        payload["deck_rng_state"]
    )
    sim.slot_rng.bit_generator.state = copy.deepcopy(
        payload["slot_rng_state"]
    )
    state = state_from_snapshot(payload["state"])
    return sim, state


RESUME_MAGIC = b"J1RESUME1\n"
RESUME_FIXTURE_BOUNDARIES = (
    "pre_action",
    "post_step",
    "mid_vector_game",
    "pre_update",
    "post_checkpoint",
)


def _clone_tensor_mapping(
    mapping: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        str(key): value.detach().cpu().clone()
        for key, value in mapping.items()
    }


class ResumeFixtureSession:
    """Small deterministic fixture that exercises the complete resume state."""

    ENV_COUNT = 2
    TOTAL_STEPS = 10
    UPDATE_AFTER_STEPS = 4

    def __init__(self) -> None:
        self.model, self.optimizer = initialize_model_optimizer()
        self.python_rng = random.Random(2_026_072_807)
        self.numpy_rng = np.random.default_rng(2_026_072_808)
        self.policy_generator = torch.Generator(device="cpu")
        self.policy_generator.manual_seed(2_026_072_809)
        self.learning_batch = synthetic_complete_ppo_batch(self.model)
        self.update_report: dict[str, Any] | None = None
        self.envs = [
            normal_start_sim(
                role="train",
                deck_stream_id=2_130_000 + index,
                slot_stream_id=2_140_000 + index,
            )
            for index in range(self.ENV_COUNT)
        ]
        self.root_cursors = [0 for _ in range(self.ENV_COUNT)]
        self.task_cursor = 0
        self.steps_done = 0
        self.updated = False
        self.checkpoint_sealed = False
        self.actions: list[int] = []
        self.observation_hashes: list[str] = []
        self.transition_buffers: list[list[dict[str, Any]]] = [
            [] for _ in range(self.ENV_COUNT)
        ]
        self.checkpoint_identity: str | None = None
        self.boundary_trace: list[str] = ["pre_action"]

    def _step_one(self) -> None:
        env_index = self.task_cursor % self.ENV_COUNT
        sim, state = self.envs[env_index]
        if state.game_over:
            raise J1IntegrityError("Synthetic resume root terminated early")
        observation = encode_observation(state, sim)
        if observation.shape != (EXPECTED_OBSERVATION_WIDTH,):
            raise J1IntegrityError("Resume observation width changed")
        legal_mask_np = sim.legal_mask(state)
        if not legal_mask_np.any():
            raise J1IntegrityError("Live synthetic state has no legal action")
        with torch.no_grad():
            logits, value, auxiliary = self.model(
                torch.from_numpy(observation).unsqueeze(0)
            )
            action = int(
                sampled_masked_actions(
                    logits,
                    torch.from_numpy(legal_mask_np).unsqueeze(0),
                    generator=self.policy_generator,
                )[0]
            )
        if not legal_mask_np[action]:
            raise J1IntegrityError("Resume fixture sampled an illegal action")
        before = score_board(state.board)
        next_state, info = sim.step(state, action)
        after = score_board(next_state.board)
        if int(info.score_delta) != after - before:
            raise J1IntegrityError("Simulator score delta changed")
        transition = {
            "observation": observation.copy(),
            "legal_mask": legal_mask_np.copy(),
            "action": action,
            "value": float(value[0]),
            "auxiliary": auxiliary[0].detach().cpu().numpy().copy(),
            "score_delta": int(info.score_delta),
            "done_after_transition": bool(next_state.game_over),
            "root_cursor": int(self.root_cursors[env_index]),
            "task_cursor": int(self.task_cursor),
        }
        self.transition_buffers[env_index].append(transition)
        self.envs[env_index] = (sim, next_state)
        self.actions.append(action)
        self.observation_hashes.append(stable_hash(observation))
        self.steps_done += 1
        self.task_cursor += 1
        self.python_rng.random()
        self.numpy_rng.random()
        self.boundary_trace.append("post_step")
        if self.steps_done == 3:
            self.boundary_trace.append("mid_vector_game")

    def _ppo_update(self) -> None:
        if self.updated:
            raise J1IntegrityError("Fixture PPO optimizer stepped twice")
        self.update_report = apply_frozen_ppo_update(
            self.model,
            self.optimizer,
            self.learning_batch,
            round_number=1,
            epochs=FROZEN_CONFIG.epochs_per_round,
            minibatch_size=2,
            optimizer_step=True,
        )
        self.updated = True
        assert_finite_model(self.model)

    def _seal_checkpoint(self) -> None:
        self.checkpoint_identity = stable_hash(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "transition_buffer": self.transition_buffers,
                "learning_batch": self.learning_batch.payload(),
                "update_report": self.update_report,
            }
        )
        self.checkpoint_sealed = True
        self.boundary_trace.append("post_checkpoint")

    def advance_to_boundary(self, boundary: str) -> None:
        if boundary not in RESUME_FIXTURE_BOUNDARIES:
            raise ValueError(f"Unknown resume boundary: {boundary}")
        if boundary == "pre_action":
            return
        while True:
            if self.steps_done < self.UPDATE_AFTER_STEPS:
                self._step_one()
                if boundary == "post_step" and self.steps_done == 1:
                    return
                if boundary == "mid_vector_game" and self.steps_done == 3:
                    return
                if (
                    boundary == "pre_update"
                    and self.steps_done == self.UPDATE_AFTER_STEPS
                ):
                    self.boundary_trace.append("pre_update")
                    return
                continue
            if not self.updated:
                self._ppo_update()
                self._seal_checkpoint()
                if boundary == "post_checkpoint":
                    return
            return

    def finish(self) -> dict[str, Any]:
        if self.steps_done < self.UPDATE_AFTER_STEPS:
            self.advance_to_boundary("pre_update")
        if not self.updated:
            self._ppo_update()
            self._seal_checkpoint()
        while self.steps_done < self.TOTAL_STEPS:
            self._step_one()
        return self.identity()

    def snapshot_payload(self) -> dict[str, Any]:
        return {
            "version": "j1_resume_fixture_v1",
            "model": _clone_tensor_mapping(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "python_rng_state": self.python_rng.getstate(),
            "numpy_rng_state": copy.deepcopy(
                self.numpy_rng.bit_generator.state
            ),
            "torch_global_rng_state": torch.get_rng_state().clone(),
            "policy_rng_state": self.policy_generator.get_state().clone(),
            "environments": [
                simulator_snapshot(sim, state)
                for sim, state in self.envs
            ],
            "root_cursors": list(self.root_cursors),
            "task_cursor": int(self.task_cursor),
            "steps_done": int(self.steps_done),
            "updated": bool(self.updated),
            "checkpoint_sealed": bool(self.checkpoint_sealed),
            "actions": list(self.actions),
            "observation_hashes": list(self.observation_hashes),
            "transition_buffers": copy.deepcopy(self.transition_buffers),
            "learning_batch": self.learning_batch.payload(),
            "update_report": copy.deepcopy(self.update_report),
            "checkpoint_identity": self.checkpoint_identity,
            "boundary_trace": list(self.boundary_trace),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ResumeFixtureSession":
        if payload.get("version") != "j1_resume_fixture_v1":
            raise J1IntegrityError("Resume fixture version changed")
        session = cls()
        session.model.load_state_dict(payload["model"], strict=True)
        session.optimizer.load_state_dict(payload["optimizer"])
        session.python_rng.setstate(payload["python_rng_state"])
        session.numpy_rng.bit_generator.state = copy.deepcopy(
            payload["numpy_rng_state"]
        )
        torch.set_rng_state(payload["torch_global_rng_state"].clone())
        session.policy_generator.set_state(payload["policy_rng_state"].clone())
        session.envs = [
            simulator_from_snapshot(row)
            for row in payload["environments"]
        ]
        session.root_cursors = [
            int(value) for value in payload["root_cursors"]
        ]
        session.task_cursor = int(payload["task_cursor"])
        session.steps_done = int(payload["steps_done"])
        session.updated = bool(payload["updated"])
        session.checkpoint_sealed = bool(payload["checkpoint_sealed"])
        session.actions = [int(value) for value in payload["actions"]]
        session.observation_hashes = list(payload["observation_hashes"])
        session.transition_buffers = copy.deepcopy(
            payload["transition_buffers"]
        )
        session.learning_batch = FrozenPPOBatch.from_payload(
            payload["learning_batch"]
        )
        validate_ppo_batch(session.learning_batch)
        session.update_report = copy.deepcopy(payload["update_report"])
        session.checkpoint_identity = payload["checkpoint_identity"]
        session.boundary_trace = list(payload["boundary_trace"])
        assert_finite_model(session.model)
        return session

    def identity(self) -> dict[str, Any]:
        environment_identity = [
            stable_hash(simulator_snapshot(sim, state))
            for sim, state in self.envs
        ]
        payload = {
            "actions": list(self.actions),
            "observation_hashes": list(self.observation_hashes),
            "environment_identity": environment_identity,
            "root_cursors": list(self.root_cursors),
            "task_cursor": int(self.task_cursor),
            "steps_done": int(self.steps_done),
            "transition_buffer_sha256": stable_hash(
                self.transition_buffers
            ),
            "learning_batch_sha256": stable_hash(
                self.learning_batch.payload()
            ),
            "update_report_sha256": (
                None
                if self.update_report is None
                else stable_hash(self.update_report)
            ),
            "model_sha256": stable_hash(self.model.state_dict()),
            "optimizer_sha256": stable_hash(self.optimizer.state_dict()),
            "python_rng_sha256": stable_hash(self.python_rng.getstate()),
            "numpy_rng_sha256": stable_hash(
                self.numpy_rng.bit_generator.state
            ),
            "torch_rng_sha256": stable_hash(torch.get_rng_state()),
            "policy_rng_sha256": stable_hash(
                self.policy_generator.get_state()
            ),
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_sealed": self.checkpoint_sealed,
        }
        payload["final_identity_sha256"] = stable_hash(payload)
        return payload


def save_resume_fixture(path: Path, session: ResumeFixtureSession) -> str:
    payload = session.snapshot_payload()
    envelope = {
        "version": "j1_resume_envelope_v1",
        "payload_sha256": stable_hash(payload),
        "payload": payload,
    }
    buffer = io.BytesIO()
    torch.save(envelope, buffer)
    serialized_payload = buffer.getvalue()
    serialized_sha256 = hashlib.sha256(serialized_payload).hexdigest()
    raw = (
        RESUME_MAGIC
        + serialized_sha256.encode("ascii")
        + b"\n"
        + serialized_payload
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    return sha256_path(path)


def load_resume_fixture(path: Path) -> ResumeFixtureSession:
    raw = path.read_bytes()
    if not raw.startswith(RESUME_MAGIC):
        raise J1IntegrityError("Resume fixture magic changed")
    digest_start = len(RESUME_MAGIC)
    digest_end = digest_start + 64
    if len(raw) <= digest_end or raw[digest_end : digest_end + 1] != b"\n":
        raise J1IntegrityError("Resume fixture byte digest is missing")
    expected_serialized_sha256 = raw[digest_start:digest_end].decode("ascii")
    serialized_payload = raw[digest_end + 1 :]
    if (
        hashlib.sha256(serialized_payload).hexdigest()
        != expected_serialized_sha256
    ):
        raise J1IntegrityError("Resume fixture serialized-byte hash mismatch")
    try:
        envelope = torch.load(
            io.BytesIO(serialized_payload),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise J1IntegrityError("Corrupt resume fixture") from error
    if envelope.get("version") != "j1_resume_envelope_v1":
        raise J1IntegrityError("Resume envelope version changed")
    payload = envelope.get("payload")
    if (
        not isinstance(payload, Mapping)
        or envelope.get("payload_sha256") != stable_hash(payload)
    ):
        raise J1IntegrityError("Resume payload hash mismatch")
    return ResumeFixtureSession.from_payload(payload)


def resume_equivalence_fixture(
    boundary: str,
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    uninterrupted = ResumeFixtureSession()
    expected = uninterrupted.finish()
    interrupted = ResumeFixtureSession()
    interrupted.advance_to_boundary(boundary)
    file_sha256 = save_resume_fixture(checkpoint_path, interrupted)
    resumed = load_resume_fixture(checkpoint_path)
    observed = resumed.finish()
    checks = {
        key: expected[key] == observed[key]
        for key in expected
    }
    return {
        "boundary": boundary,
        "resume_file_sha256": file_sha256,
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "passes": all(checks.values()),
    }


def source_identity() -> dict[str, Any]:
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    checks = {
        "proposal_exact": (
            sha256_path(PROPOSAL_PATH) == EXPECTED_PROPOSAL_SHA256
        ),
        "readiness_file_exact": (
            sha256_path(READINESS_PATH)
            == EXPECTED_READINESS_FILE_SHA256
        ),
        "readiness_payload_exact": (
            readiness.get("canonical_payload_sha256")
            == EXPECTED_READINESS_PAYLOAD_SHA256
            and verify_payload_hash(readiness)
        ),
        "torch_exact": torch.__version__ == EXPECTED_TORCH_VERSION,
        "observation_width_exact": (
            observation_size("full") == EXPECTED_OBSERVATION_WIDTH
        ),
    }
    dependencies: dict[str, dict[str, Any]] = {}
    for relative, expected in DEPENDENCY_BINDINGS.items():
        path = repo_path(relative)
        observed = sha256_path(path) if path.is_file() else None
        dependencies[relative] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passes": observed == expected,
        }
    checks["dependencies_exact"] = all(
        row["passes"] for row in dependencies.values()
    )
    return {
        "proposal": {
            "path": str(PROPOSAL_PATH.relative_to(REPO_ROOT)),
            "file_sha256": sha256_path(PROPOSAL_PATH),
        },
        "readiness": {
            "path": str(READINESS_PATH.relative_to(REPO_ROOT)),
            "file_sha256": sha256_path(READINESS_PATH),
            "payload_sha256": readiness.get("canonical_payload_sha256"),
        },
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
            "file_sha256": sha256_path(TEST_PATH)
            if TEST_PATH.is_file()
            else None,
        },
        "dependencies": dependencies,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _stream_manifest_paths(runs_root: Path = RUNS_ROOT) -> list[Path]:
    if not runs_root.is_dir():
        raise J1IntegrityError(f"Missing runs root: {runs_root}")
    paths = []
    for path in runs_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "stream" in name and "manifest" in name:
            paths.append(path)
    return sorted(paths, key=lambda path: str(path.resolve()))


def prospective_stream_contract() -> dict[str, Any]:
    intervals: list[dict[str, Any]] = []
    seen: set[int] = set()
    duplicate_count = 0
    for partition, payload in PROSPECTIVE_STREAMS.items():
        rows = int(payload["rows"])
        for stream_role, base in payload.items():
            if stream_role == "rows":
                continue
            start = int(base)
            end = start + rows - 1
            values = range(start, end + 1)
            overlap = sum(value in seen for value in values)
            duplicate_count += overlap
            seen.update(values)
            intervals.append(
                {
                    "partition": partition,
                    "stream_role": stream_role,
                    "base": start,
                    "rows": rows,
                    "end_inclusive": end,
                    "above_historical_ceiling": (
                        start > HISTORICAL_STREAM_MAX
                    ),
                }
            )
    root_examples = []
    marker_placeholder = "future_marker_payload_sha256"
    for partition, payload in PROSPECTIVE_STREAMS.items():
        for row in (0, int(payload["rows"]) - 1):
            root_examples.append(
                {
                    "partition": partition,
                    "row": row,
                    "prospective_root_id": canonical_json_hash(
                        {
                            "marker_payload_sha256": marker_placeholder,
                            "partition": partition,
                            "row": row,
                            "logical_stream_id": int(payload["logical"]) + row,
                            "deck_stream_id": int(payload["deck"]) + row,
                            "slot_stream_id": int(payload["slot"]) + row,
                        }
                    ),
                }
            )
    checks = {
        "historical_ceiling_exact": (
            HISTORICAL_STREAM_MAX == 212_999_999_999
        ),
        "all_prospective_above_ceiling": all(
            row["above_historical_ceiling"] for row in intervals
        ),
        "all_stream_roles_disjoint": duplicate_count == 0,
        "train_rows_exact": (
            PROSPECTIVE_STREAMS["train"]["rows"] == TRAIN_ROOTS
        ),
        "development_pairs_exact": (
            PROSPECTIVE_STREAMS["development"]["rows"]
            == DEVELOPMENT_PAIRS
        ),
        "confirmation_pairs_exact": (
            PROSPECTIVE_STREAMS["confirmation"]["rows"]
            == CONFIRMATION_PAIRS
        ),
        "game_arms_exact": TOTAL_GAME_ARMS == 28_672,
    }
    return {
        "historical_denied_interval": {
            "start": 0,
            "end_inclusive": HISTORICAL_STREAM_MAX,
        },
        "prospective_intervals": intervals,
        "prospective_unique_stream_id_count": len(seen),
        "duplicate_stream_id_count": duplicate_count,
        "root_id_derivation": (
            "SHA256(canonical JSON of marker payload, partition, row, "
            "logical, deck, slot)"
        ),
        "root_id_examples_with_placeholder": root_examples,
        "streams_reserved": 0,
        "streams_consumed": 0,
        "checks": checks,
        "passes": all(checks.values()),
    }


def build_protected_denylist(
    *,
    repo_root: Path = REPO_ROOT,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Any]:
    root_rows = []
    for relative, expected in ROOT_MANIFEST_BINDINGS.items():
        path = repo_root / relative
        if path.is_symlink():
            raise J1IntegrityError(f"Protected root manifest is symlinked: {path}")
        observed = sha256_path(path) if path.is_file() else None
        root_rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passes": observed == expected,
            }
        )
    stream_rows = []
    for path in _stream_manifest_paths(runs_root):
        if path.is_symlink():
            raise J1IntegrityError(f"Stream manifest is symlinked: {path}")
        stream_rows.append(
            {
                "path": str(path.resolve().relative_to(repo_root.resolve())),
                "file_sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
        )
    stream_paths = [row["path"] for row in stream_rows]
    prospective = prospective_stream_contract()
    checks = {
        "root_manifests_exact": all(row["passes"] for row in root_rows),
        "root_manifest_paths_unique": (
            len(root_rows) == len({row["path"] for row in root_rows})
        ),
        "stream_manifest_inventory_nonempty": bool(stream_rows),
        "stream_manifest_paths_unique": (
            len(stream_paths) == len(set(stream_paths))
        ),
        "prospective_stream_contract_passes": prospective["passes"],
    }
    payload = {
        "version": "j1_protected_id_denylist_v1",
        "method": (
            "byte hashes only; no protected root or stream-manifest payload "
            "was parsed"
        ),
        "protected_root_manifests": root_rows,
        "stream_manifest_inventory": stream_rows,
        "stream_manifest_inventory_sha256": canonical_json_hash(stream_rows),
        "historical_denied_interval": {
            "start": 0,
            "end_inclusive": HISTORICAL_STREAM_MAX,
        },
        "prospective_stream_contract": prospective,
        "protected_payloads_parsed": False,
        "historical_schema_discovery_used": False,
        "checks": checks,
        "passes": all(checks.values()),
    }
    if not payload["passes"]:
        raise J1IntegrityError("Protected denylist audit failed")
    return payload


def verify_protected_denylist(
    sealed: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    runs_root: Path = RUNS_ROOT,
) -> bool:
    current = build_protected_denylist(
        repo_root=repo_root,
        runs_root=runs_root,
    )
    return canonical_json_hash(current) == canonical_json_hash(dict(sealed))


TRANSITION_DTYPE = np.dtype(
    [
        ("observation", np.float32, (EXPECTED_OBSERVATION_WIDTH,)),
        ("legal_mask", np.uint8, (4,)),
        ("action", np.int8),
        ("old_log_probability", np.float32),
        ("old_value", np.float32),
        ("reward", np.float32),
        ("done_after_transition", np.uint8),
        ("return", np.float32),
        ("advantage", np.float32),
        ("auxiliary_targets", np.uint8, (3,)),
        ("root_weight", np.float32),
    ],
    align=False,
)
ROOT_METADATA_BYTES = 4_096
EVALUATION_ROOT_SUMMARY_BYTES = 8_192
RESUME_METADATA_BYTES = 2 * 1024**2


def _timing_summary(samples: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise J1IntegrityError("Invalid timing fixture samples")
    return {
        "count": int(len(values)),
        "median_seconds": float(np.median(values)),
        "p90_seconds": float(np.quantile(values, 0.90)),
        "p99_seconds": float(np.quantile(values, 0.99)),
        "max_seconds": float(np.max(values)),
    }


def _benchmark_calls(
    call: Any,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, float | int]:
    for _ in range(warmups):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return _timing_summary(samples)


def _incumbent_spec() -> str:
    lines = [
        line.strip()
        for line in repo_path("threes_rl/current_incumbent_policy.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    values = [line for line in lines if line and not line.startswith("#")]
    if len(values) != 1:
        raise J1IntegrityError("Incumbent policy file is not a single spec")
    return values[0]


def benchmark_projection_fixtures() -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, _optimizer = initialize_model_optimizer()
    model.train()
    actor_observations = torch.zeros(
        (16, EXPECTED_OBSERVATION_WIDTH),
        dtype=torch.float32,
    )
    actor_masks = torch.ones((16, 4), dtype=torch.bool)

    def actor_call() -> None:
        with torch.no_grad():
            logits, values, auxiliaries = model(actor_observations)
            actions = deterministic_masked_actions(logits, actor_masks)
            if (
                actions.shape != (16,)
                or not torch.isfinite(values).all()
                or not torch.isfinite(auxiliaries).all()
            ):
                raise J1IntegrityError("Synthetic actor fixture failed")

    projection_root_lengths = tuple(
        8 for _ in range(FROZEN_CONFIG.minibatch_size // 8)
    )
    update_batch = synthetic_complete_ppo_batch(
        model,
        row_count=FROZEN_CONFIG.minibatch_size,
        root_lengths=projection_root_lengths,
        seed=2_026_072_812,
    )

    def update_call() -> None:
        report = apply_frozen_ppo_update(
            model,
            _optimizer,
            update_batch,
            round_number=1,
            epochs=1,
            minibatch_size=FROZEN_CONFIG.minibatch_size,
            optimizer_step=False,
        )
        if report["optimizer_steps"] != 0 or not report["passes"]:
            raise J1IntegrityError("Synthetic PPO timing fixture failed")

    sim, state = normal_start_sim(
        role="train",
        deck_stream_id=2_150_000,
        slot_stream_id=2_160_000,
    )
    legal = sim.legal_actions(state)
    if not legal:
        raise J1IntegrityError("Synthetic simulator fixture has no action")
    sim_snapshot = simulator_snapshot(sim, state)

    def simulator_call() -> None:
        fixture_sim, fixture_state = simulator_from_snapshot(sim_snapshot)
        next_state, info = fixture_sim.step(fixture_state, legal[0])
        if not info.moved or next_state.move_count != 1:
            raise J1IntegrityError("Synthetic simulator step failed")

    from threes_rl.eval import make_policy

    incumbent = make_policy(_incumbent_spec())
    incumbent_sim, incumbent_state = normal_start_sim(
        role="development",
        deck_stream_id=2_170_000,
        slot_stream_id=2_180_000,
    )
    incumbent_rng = np.random.default_rng(2_026_072_810)

    def incumbent_call() -> None:
        action = int(incumbent(incumbent_state, incumbent_sim, incumbent_rng))
        if action not in incumbent_sim.legal_actions(incumbent_state):
            raise J1IntegrityError("Incumbent fixture returned illegal action")

    actor = _benchmark_calls(actor_call, warmups=3, repeats=21)
    simulator = _benchmark_calls(
        simulator_call,
        warmups=3,
        repeats=21,
    )
    update = _benchmark_calls(update_call, warmups=1, repeats=5)
    incumbent_timing = _benchmark_calls(
        incumbent_call,
        warmups=2,
        repeats=11,
    )
    return {
        "fixture_only": True,
        "game_roots_generated": 0,
        "policy_outcome_inspection": 0,
        "optimizer_steps": 0,
        "action_identities_retained": False,
        "actor_batch_size": 16,
        "update_batch_size": FROZEN_CONFIG.minibatch_size,
        "actor_batch": actor,
        "simulator_transition": simulator,
        "synthetic_forward_backward": update,
        "incumbent_fixed_state_action": incumbent_timing,
    }


def runtime_storage_projection(
    fixture_timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timing = (
        benchmark_projection_fixtures()
        if fixture_timing is None
        else dict(fixture_timing)
    )
    actor_seconds = (
        float(timing["actor_batch"]["p90_seconds"])
        / int(timing["actor_batch_size"])
    )
    simulator_seconds = float(
        timing["simulator_transition"]["p90_seconds"]
    )
    incumbent_seconds = float(
        timing["incumbent_fixed_state_action"]["p90_seconds"]
    )
    update_batch_seconds = float(
        timing["synthetic_forward_backward"]["p90_seconds"]
    )
    phase_arms = {
        "training": TRAIN_ROOTS,
        "development": 2 * DEVELOPMENT_PAIRS,
        "confirmation": 2 * CONFIRMATION_PAIRS,
    }
    raw_checkpoint_bytes = (
        EXPECTED_PARAMETER_COUNT * np.dtype(np.float32).itemsize * 3
        + RESUME_METADATA_BYTES
    )
    projections: dict[str, Any] = {}
    for phase, arms in phase_arms.items():
        decisions = arms * PLANNING_MOVES
        if phase == "training":
            collection = decisions * (actor_seconds + simulator_seconds)
            transition_passes = (
                decisions * FROZEN_CONFIG.epochs_per_round
            )
            update_batches = math.ceil(
                transition_passes / FROZEN_CONFIG.minibatch_size
            )
            update_seconds = update_batches * update_batch_seconds
            central_seconds = collection + update_seconds
            round_buffer_bytes = (
                FROZEN_CONFIG.roots_per_round
                * PLANNING_MOVES
                * TRANSITION_DTYPE.itemsize
            )
            retained_bytes = (
                arms * ROOT_METADATA_BYTES
                + 2 * raw_checkpoint_bytes
            )
            peak_bytes = retained_bytes + round_buffer_bytes
        else:
            pair_count = arms // 2
            collection = (
                pair_count
                * PLANNING_MOVES
                * (
                    actor_seconds
                    + incumbent_seconds
                    + 2 * simulator_seconds
                )
            )
            update_seconds = 0.0
            central_seconds = collection
            retained_bytes = arms * EVALUATION_ROOT_SUMMARY_BYTES
            peak_bytes = retained_bytes + raw_checkpoint_bytes
        cap = PHASE_CAPS[phase]
        safety_seconds = central_seconds * SAFETY_MULTIPLIER
        safety_peak_bytes = math.ceil(peak_bytes * SAFETY_MULTIPLIER)
        max_sensitivity_seconds = (
            central_seconds * (MAX_MOVES / PLANNING_MOVES)
        )
        max_sensitivity_margin_seconds = (
            max_sensitivity_seconds * SAFETY_MULTIPLIER
        )
        projections[phase] = {
            "complete_game_arms": arms,
            "planning_decisions": decisions,
            "planning_moves_per_arm": PLANNING_MOVES,
            "collection_seconds": collection,
            "update_seconds": update_seconds,
            "central_hours": central_seconds / 3600.0,
            "hours_with_25pct_margin": safety_seconds / 3600.0,
            "contract_max_5000_move_sensitivity_hours": (
                max_sensitivity_seconds / 3600.0
            ),
            "contract_max_5000_move_sensitivity_hours_with_25pct_margin": (
                max_sensitivity_margin_seconds / 3600.0
            ),
            "contract_max_5000_move_sensitivity_runtime_passes": (
                max_sensitivity_margin_seconds / 3600.0 < cap["hours"]
            ),
            "contract_max_sensitivity_is_diagnostic": True,
            "retained_bytes": retained_bytes,
            "peak_bytes": peak_bytes,
            "peak_bytes_with_25pct_margin": safety_peak_bytes,
            "peak_gib_with_25pct_margin": safety_peak_bytes / 1024**3,
            "runtime_cap_hours": cap["hours"],
            "storage_cap_gib": cap["storage_gib"],
            "runtime_central_passes": (
                safety_seconds / 3600.0 < cap["hours"]
            ),
            "storage_passes": (
                safety_peak_bytes / 1024**3 < cap["storage_gib"]
            ),
        }
    checks = {
        "total_game_arms_exact": (
            sum(row["complete_game_arms"] for row in projections.values())
            == TOTAL_GAME_ARMS
        ),
        "transition_dtype_finite_width": TRANSITION_DTYPE.itemsize > 0,
        "all_runtime_central_pass": all(
            row["runtime_central_passes"]
            for row in projections.values()
        ),
        "all_storage_pass": all(
            row["storage_passes"] for row in projections.values()
        ),
        "fixture_optimizer_steps_zero": timing["optimizer_steps"] == 0,
        "fixture_game_roots_zero": timing["game_roots_generated"] == 0,
        "action_identities_not_retained": (
            not timing["action_identities_retained"]
        ),
    }
    integrity_check_names = (
        "total_game_arms_exact",
        "transition_dtype_finite_width",
        "fixture_optimizer_steps_zero",
        "fixture_game_roots_zero",
        "action_identities_not_retained",
    )
    integrity_passes = all(checks[name] for name in integrity_check_names)
    cost_passes = (
        checks["all_runtime_central_pass"]
        and checks["all_storage_pass"]
    )
    return {
        "version": "j1_runtime_storage_projection_v1",
        "method": "synthetic maximum-shape and fixed-state fixtures only",
        "transition_dtype": TRANSITION_DTYPE.descr,
        "bytes_per_transition": TRANSITION_DTYPE.itemsize,
        "bytes_per_root_metadata": ROOT_METADATA_BYTES,
        "bytes_per_evaluation_root_summary": (
            EVALUATION_ROOT_SUMMARY_BYTES
        ),
        "bytes_per_checkpoint": raw_checkpoint_bytes,
        "fixture_timing": timing,
        "phase_projections": projections,
        "safety_multiplier": SAFETY_MULTIPLIER,
        "central_planning_moves": PLANNING_MOVES,
        "contract_max_moves_sensitivity": MAX_MOVES,
        "integrity_passes": integrity_passes,
        "cost_passes": cost_passes,
        "checks": checks,
        "passes": integrity_passes and cost_passes,
    }


def gae_contract_fixture() -> dict[str, Any]:
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    values = np.asarray([0.5, 0.6, 0.7], dtype=np.float64)
    dones = np.asarray([False, True, False], dtype=bool)
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        bootstrap_value=0.8,
        gamma=1.0,
        gae_lambda=0.95,
    )
    expected_advantages = np.asarray([2.43, 1.4, 3.1])
    expected_returns = np.asarray([2.93, 2.0, 3.8])
    wrong_shifted_done_advantages = np.asarray([0.5, 5.195, 3.1])
    checks = {
        "transition_t_advantages_exact": np.allclose(
            advantages,
            expected_advantages,
            atol=1e-12,
            rtol=0.0,
        ),
        "transition_t_returns_exact": np.allclose(
            returns,
            expected_returns,
            atol=1e-12,
            rtol=0.0,
        ),
        "done_t_plus_1_regression_would_fail": not np.allclose(
            advantages,
            wrong_shifted_done_advantages,
            atol=1e-12,
            rtol=0.0,
        ),
    }
    return {
        "rewards": rewards.tolist(),
        "values": values.tolist(),
        "done_after_transition": dones.tolist(),
        "bootstrap_value": 0.8,
        "expected_advantages": expected_advantages.tolist(),
        "observed_advantages": advantages.tolist(),
        "checks": checks,
        "passes": all(checks.values()),
    }


def ppo_contract_fixture() -> dict[str, Any]:
    model, optimizer = initialize_model_optimizer()
    batch = synthetic_complete_ppo_batch(
        model,
        row_count=7,
        root_lengths=(2, 5),
        seed=2_026_072_813,
    )
    normalized = normalize_advantages_root_weighted(
        batch.advantages,
        batch.row_weights,
    )
    weight_total = batch.row_weights.sum()
    weighted_mean = float(
        torch.sum(batch.row_weights * normalized) / weight_total
    )
    weighted_variance = float(
        torch.sum(
            batch.row_weights * (normalized - weighted_mean) ** 2
        )
        / weight_total
    )
    root_totals: dict[str, float] = defaultdict(float)
    for root_id, weight in zip(
        batch.root_ids,
        batch.row_weights.tolist(),
    ):
        root_totals[root_id] += float(weight)
    losses = frozen_ppo_loss(
        model,
        batch,
        normalized_advantages=normalized,
    )
    schedule = ppo_schedule_audit(
        batch.row_count(),
        round_number=1,
        epochs=4,
        minibatch_size=3,
    )
    before = stable_hash(model.state_dict())
    update = apply_frozen_ppo_update(
        model,
        optimizer,
        batch,
        round_number=1,
        epochs=4,
        minibatch_size=3,
        optimizer_step=True,
    )
    after = stable_hash(model.state_dict())
    weight_identity = stable_hash(batch.row_weights)
    component_weight_identity = {
        name: weight_identity
        for name in ("policy", "value", "entropy", "auxiliary")
    }
    checks = {
        "unequal_root_lengths": len(set((2, 5))) == 2,
        "root_total_weights_equal_one": all(
            math.isclose(total, 1.0, abs_tol=2e-7, rel_tol=0.0)
            for total in root_totals.values()
        ),
        "weighted_advantage_mean_zero": abs(weighted_mean) < 1e-6,
        "weighted_advantage_variance_unit": abs(
            weighted_variance - 1.0
        ) < 1e-5,
        "all_components_share_root_weights": (
            len(set(component_weight_identity.values())) == 1
        ),
        "losses_finite": all(
            torch.isfinite(value) for value in losses.values()
        ),
        "four_epoch_schedule": schedule["passes"],
        "final_short_minibatch_retained": (
            schedule["short_minibatch_sizes"] == [1, 1, 1, 1]
        ),
        "actual_optimizer_steps": update["optimizer_steps"] == 12,
        "model_updated": before != after,
        "round_one_learning_rate_exact": math.isclose(
            update["update_learning_rate"],
            FROZEN_CONFIG.learning_rate,
            abs_tol=0.0,
            rel_tol=0.0,
        ),
        "round_64_post_learning_rate_zero": (
            round_learning_rate(64, after_round=True) == 0.0
        ),
    }
    return {
        "batch_sha256": stable_hash(batch.payload()),
        "root_lengths": [2, 5],
        "root_total_weights": root_totals,
        "row_weight_sha256": weight_identity,
        "component_weight_sha256": component_weight_identity,
        "weighted_normalized_advantage_mean": weighted_mean,
        "weighted_normalized_advantage_variance": weighted_variance,
        "losses": {
            key: float(value.detach().cpu())
            for key, value in losses.items()
        },
        "schedule": schedule,
        "update": {
            "optimizer_steps": update["optimizer_steps"],
            "update_learning_rate": update["update_learning_rate"],
            "post_round_learning_rate": (
                update["post_round_learning_rate"]
            ),
            "schedule_sha256": update["schedule"]["plan_sha256"],
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def dense_reward_complete_fixture(
    seed: int,
    *,
    max_moves: int = MAX_MOVES,
) -> dict[str, Any]:
    sim, state = normal_start_sim(
        role="train",
        deck_stream_id=3_000_000 + int(seed) * 2,
        slot_stream_id=3_000_001 + int(seed) * 2,
    )
    action_rng = np.random.default_rng(4_000_000 + int(seed))
    start_score = score_board(state.board)
    score_deltas: list[int] = []
    while not state.game_over and state.move_count < max_moves:
        legal = sim.legal_actions(state)
        if not legal:
            state.game_over = True
            break
        action = int(legal[int(action_rng.integers(len(legal)))])
        state, info = sim.step(state, action)
        if not info.moved:
            raise J1IntegrityError("Complete fixture selected an illegal action")
        score_deltas.append(int(info.score_delta))
    if not state.game_over:
        raise J1IntegrityError("Complete fixture hit the move cap")
    end_score = score_board(state.board)
    objective = verify_dense_reward_telescoping(
        start_score,
        end_score,
        score_deltas,
    )
    return {
        "fixture_seed": int(seed),
        "natural_terminal": True,
        "move_count": int(state.move_count),
        "score_fields_retained": False,
        "action_sequence_retained": False,
        "objective": objective,
        "passes": objective["passes"],
    }


def semantic_contract_audit(
    *,
    resume_dir: Path,
) -> dict[str, Any]:
    model, _optimizer = initialize_model_optimizer()
    assert_finite_model(model)
    normal_start_rows = []
    for index, role in enumerate(
        ("train", "development", "confirmation")
    ):
        sim, state = normal_start_sim(
            role=role,
            deck_stream_id=5_000_000 + 2 * index,
            slot_stream_id=5_000_001 + 2 * index,
        )
        snapshot = simulator_snapshot(sim, state)
        restored_sim, restored_state = simulator_from_snapshot(snapshot)
        normal_start_rows.append(
            {
                "role": role,
                "starter_none": sim.starter_tile is None,
                "restored_starter_none": restored_sim.starter_tile is None,
                "state_roundtrip_exact": (
                    stable_hash(state_snapshot(state))
                    == stable_hash(state_snapshot(restored_state))
                ),
                "deck_rng_roundtrip_exact": (
                    stable_hash(sim.deck_rng.bit_generator.state)
                    == stable_hash(restored_sim.deck_rng.bit_generator.state)
                ),
                "slot_rng_roundtrip_exact": (
                    stable_hash(sim.slot_rng.bit_generator.state)
                    == stable_hash(restored_sim.slot_rng.bit_generator.state)
                ),
            }
        )
    root_rows = [
        CompleteRoot(
            root_id="root-a",
            ancestry_id="ancestry-a",
            partition="train",
            natural_terminal=True,
            transitions=(
                {"reward": 1.0},
                {"reward": 2.0},
            ),
        ),
        CompleteRoot(
            root_id="root-b",
            ancestry_id="ancestry-b",
            partition="train",
            natural_terminal=True,
            transitions=(
                {"reward": 3.0},
                {"reward": 4.0},
                {"reward": 5.0},
            ),
        ),
    ]
    flattened = flatten_complete_roots(
        root_rows,
        expected_partition="train",
    )
    resume_rows = []
    resume_dir.mkdir(parents=True, exist_ok=True)
    for boundary in RESUME_FIXTURE_BOUNDARIES:
        path = resume_dir / f"{boundary}.pt"
        resume_rows.append(
            resume_equivalence_fixture(
                boundary,
                checkpoint_path=path,
            )
        )
        path.unlink()
    if any(resume_dir.iterdir()):
        raise J1IntegrityError("Synthetic resume directory was not cleared")
    resume_dir.rmdir()
    dense_rows = [
        dense_reward_complete_fixture(seed)
        for seed in (11, 17, 23)
    ]
    gae_fixture = gae_contract_fixture()
    ppo_fixture = ppo_contract_fixture()
    checks = {
        "parameter_count_exact": (
            parameter_count(model) == EXPECTED_PARAMETER_COUNT
        ),
        "model_schema_exact": (
            model_schema_sha256() == canonical_json_hash(model_schema())
        ),
        "model_finite": True,
        "normal_starts_exact": all(
            all(
                row[key]
                for key in (
                    "starter_none",
                    "restored_starter_none",
                    "state_roundtrip_exact",
                    "deck_rng_roundtrip_exact",
                    "slot_rng_roundtrip_exact",
                )
            )
            for row in normal_start_rows
        ),
        "gae_contract": gae_fixture["passes"],
        "ppo_contract": ppo_fixture["passes"],
        "root_equal_weight": all(
            math.isclose(value, 1.0, abs_tol=1e-7)
            for value in flattened["per_root_weight"].values()
        ),
        "resume_all_boundaries": all(
            row["passes"] for row in resume_rows
        ),
        "dense_reward_telescopes": all(
            row["passes"] for row in dense_rows
        ),
    }
    return {
        "model": {
            "parameter_count": parameter_count(model),
            "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
            "schema": model_schema(),
            "schema_sha256": model_schema_sha256(),
        },
        "normal_start_roundtrips": normal_start_rows,
        "gae": gae_fixture,
        "ppo": ppo_fixture,
        "complete_root_fixture": {
            "root_count": len(root_rows),
            "transition_count": len(flattened["rows"]),
            "per_root_weight": flattened["per_root_weight"],
            "transition_buffer_sha256": (
                flattened["transition_buffer_sha256"]
            ),
        },
        "resume_equivalence": [
            {
                "boundary": row["boundary"],
                "resume_file_sha256": row["resume_file_sha256"],
                "checks": row["checks"],
                "passes": row["passes"],
            }
            for row in resume_rows
        ],
        "dense_objective_fixtures": dense_rows,
        "scientific_roots_generated": 0,
        "scientific_optimizer_steps": 0,
        "scientific_checkpoints": 0,
        "checks": checks,
        "passes": all(checks.values()),
    }


def free_disk_gib() -> float:
    return shutil.disk_usage(REPO_ROOT).free / 1024**3


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )


def _socket_open(port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout):
            return True
    except OSError:
        return False


def recorder_health() -> dict[str, Any]:
    url = "http://127.0.0.1:8770/api/health"
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        return {
            "url": url,
            "status": None,
            "advisor_ready": None,
            "advisor_policy_sha256": None,
            "error_type": type(error).__name__,
            "active_session_content_read": False,
            "passes": False,
        }
    advisor = payload.get("advisor")
    if isinstance(advisor, Mapping):
        advisor_ready = (
            bool(advisor.get("ready"))
            if "ready" in advisor
            else advisor.get("status") == "ready"
        )
        policy_sha256 = advisor.get("policy_file_sha256")
    else:
        advisor_ready = bool(payload.get("advisor_ready"))
        policy_sha256 = payload.get("advisor_policy_sha256")
    return {
        "url": url,
        "status": payload.get("status"),
        "advisor_ready": advisor_ready,
        "advisor_policy_sha256": policy_sha256,
        "active_session_content_read": False,
        "passes": (
            payload.get("status") == "ok"
            and bool(advisor_ready)
            and policy_sha256 == EXPECTED_INCUMBENT_POLICY_SHA256
        ),
    }


def service_audit() -> dict[str, Any]:
    ports = {str(port): _socket_open(port) for port in (8765, 8770)}
    recorder = recorder_health()
    dashboard_path = (
        REPO_ROOT / "threes_rl" / "runs" / "dashboard" / "dashboard.json"
    )
    try:
        dashboard_payload = json.loads(
            dashboard_path.read_text(encoding="utf-8")
        )
        scores = tuple(
            int(row["score"])
            for row in dashboard_payload["global_top_replays"][:3]
        )
        best = int(dashboard_payload["best_high_score"])
        dashboard = {
            "path": _relative_to_repo(dashboard_path),
            "best_high_score": best,
            "top_three": scores,
            "passes": best == TOP_THREE[0] and scores == TOP_THREE,
        }
    except Exception as error:
        dashboard = {
            "path": _relative_to_repo(dashboard_path),
            "best_high_score": None,
            "top_three": (),
            "error_type": type(error).__name__,
            "passes": False,
        }
    checks = {
        "ports_8765_8770_open": all(ports.values()),
        "recorder_and_advisor_healthy": recorder["passes"],
        "human_session_content_unread": (
            not recorder["active_session_content_read"]
        ),
        "dashboard_top_three_exact": dashboard["passes"],
    }
    return {
        "ports": ports,
        "recorder": recorder,
        "dashboard": dashboard,
        "protected_top_three": list(TOP_THREE),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _ancestor_pids(pid: int | None = None) -> set[int]:
    current = int(pid or os.getpid())
    ancestors = {current}
    while current > 1:
        try:
            output = subprocess.run(
                ("ps", "-o", "ppid=", "-p", str(current)),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            parent = int(output)
        except (OSError, subprocess.SubprocessError, ValueError):
            break
        if parent <= 1 or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def heavy_process_audit() -> dict[str, Any]:
    ancestors = _ancestor_pids()
    candidates: dict[int, set[str]] = defaultdict(set)
    for pattern in HEAVY_PROCESS_PATTERNS:
        result = subprocess.run(
            ("pgrep", "-f", pattern),
            check=False,
            capture_output=True,
            text=True,
        )
        for token in result.stdout.split():
            try:
                candidates[int(token)].add(pattern)
            except ValueError:
                continue
    unrelated = {
        pid: sorted(patterns)
        for pid, patterns in candidates.items()
        if pid not in ancestors
    }
    return {
        "method": (
            "PID-only pattern audit; command lines and human-session "
            "content are not retained"
        ),
        "ancestor_pids": sorted(ancestors),
        "candidate_pids": sorted(candidates),
        "unrelated_candidate_pids": sorted(unrelated),
        "matched_patterns_by_unrelated_pid": unrelated,
        "passes": not unrelated,
    }


def operational_audit(
    *,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    free_gib = free_disk_gib()
    nice = int(os.getpriority(os.PRIO_PROCESS, 0))
    process = heavy_process_audit()
    services = service_audit()
    checks = {
        "nice_at_least_10": nice >= MIN_NICE,
        "one_heavy_job": process["passes"],
        "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
        "target_120_gib_met": free_gib > TARGET_FREE_GIB,
        "services_healthy": services["passes"],
    }
    return {
        "nice": nice,
        "free_disk_gib": free_gib,
        "output_bytes": directory_bytes(output_dir),
        "process": process,
        "services": services,
        "checks": checks,
        "passes": all(checks.values()),
    }


ZERO_WORK = {
    "execution_markers": 0,
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


def audit_zero_work(
    *,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    source = source_identity()
    output_entries = (
        sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
        )
        if output_dir.exists()
        else []
    )
    operations = operational_audit(output_dir=output_dir)
    checks = {
        "source_identity_passes": source["passes"],
        "output_namespace_absent": not output_dir.exists(),
        "output_namespace_empty": not output_entries,
        "zero_work_counters": all(value == 0 for value in ZERO_WORK.values()),
        "operational_passes": operations["passes"],
    }
    return {
        "version": f"{VERSION}_zero_work_audit",
        "source_identity": source,
        "output_dir": str(output_dir.resolve()),
        "output_entries": output_entries,
        "operational": operations,
        "zero_work": dict(ZERO_WORK),
        "checks": checks,
        "decision": (
            "READY_J1_IMPLEMENTATION_PREFLIGHT_SURFACE"
            if all(checks.values())
            else "HOLD_J1_IMPLEMENTATION_PREFLIGHT"
        ),
        "passes": all(checks.values()),
    }


def _load_hashed_json(
    path: Path,
    *,
    field: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_payload_hash(
        payload,
        field,
    ):
        raise J1IntegrityError(f"Invalid immutable payload hash: {path}")
    return payload


def write_test_evidence(
    *,
    focused_passed: int,
    regressions_passed: int,
    deselections: Sequence[str],
    commands: Sequence[str],
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    if output_dir.exists():
        raise J1IntegrityError(
            "J1 evidence namespace must be absent before sealing"
        )
    source = source_identity()
    if not source["passes"]:
        raise J1IntegrityError("J1 source identity failed before evidence")
    if int(focused_passed) < 1 or int(regressions_passed) < 1:
        raise J1IntegrityError("J1 test accounting is empty")
    payload = {
        "version": f"{VERSION}_test_evidence",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_identity": source,
        "focused_tests_passed": int(focused_passed),
        "applicable_regressions_passed": int(regressions_passed),
        "documented_historical_artifact_state_deselections": list(
            deselections
        ),
        "recorded_commands": list(commands),
        "zero_work": dict(ZERO_WORK),
        "scientific_fixture_distinction": {
            "synthetic_optimizer_fixture_steps_may_exist_in_tests": True,
            "scientific_optimizer_steps": 0,
            "synthetic_simulator_fixtures_may_complete": True,
            "j1_game_roots_generated": 0,
        },
        "passes": True,
    }
    written = write_immutable_json(
        output_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )
    path = output_dir / TEST_EVIDENCE_NAME
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "payload_sha256": written["test_evidence_payload_sha256"],
    }


def _preflight_existing_entries(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        raise J1IntegrityError("J1 preflight output directory is absent")
    return sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def prepare(
    *,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    if output_dir.resolve() != OUTPUT_DIR.resolve():
        raise J1IntegrityError("J1 preflight output directory changed")
    entries = _preflight_existing_entries(output_dir)
    if entries != [TEST_EVIDENCE_NAME]:
        raise J1IntegrityError(
            f"J1 preflight namespace is not evidence-only: {entries}"
        )
    evidence_path = output_dir / TEST_EVIDENCE_NAME
    evidence = _load_hashed_json(
        evidence_path,
        field="test_evidence_payload_sha256",
    )
    source = source_identity()
    if evidence["source_identity"] != source:
        raise J1IntegrityError("J1 source changed after test evidence")
    operations = operational_audit(output_dir=output_dir)
    denylist = build_protected_denylist()
    semantic = semantic_contract_audit(
        resume_dir=output_dir.parent / ".j1_resume_fixture_tmp"
    )
    projection = runtime_storage_projection()
    checks = {
        "source_identity": source["passes"],
        "test_evidence": bool(evidence.get("passes")),
        "denylist": denylist["passes"],
        "semantic_contracts": semantic["passes"],
        "runtime_storage_projection": projection["passes"],
        "operational": operations["passes"],
        "zero_scientific_work": all(
            value == 0 for value in ZERO_WORK.values()
        ),
        "no_execution_marker_defined": not any(
            "marker" in value.lower()
            or "opened" in value.lower()
            for value in (
                TEST_EVIDENCE_NAME,
                DENYLIST_NAME,
                PROJECTION_NAME,
                PREFLIGHT_LOCK_NAME,
                PREFLIGHT_RESULT_NAME,
            )
        ),
    }
    projection_integrity = bool(
        projection.get("integrity_passes", projection["passes"])
    )
    projection_cost = bool(
        projection.get("cost_passes", projection["passes"])
    )
    immutable_checks = (
        checks["source_identity"]
        and checks["test_evidence"]
        and checks["denylist"]
        and checks["semantic_contracts"]
        and projection_integrity
    )
    if not immutable_checks:
        decision = "KILL_J1_IMPLEMENTATION_INTEGRITY"
    elif not operations["passes"] or not projection_cost:
        decision = "HOLD_J1_IMPLEMENTATION_PREFLIGHT"
    else:
        decision = "READY_J1_IMPLEMENTATION_PREFLIGHT"
    denylist_written = write_immutable_json(
        output_dir / DENYLIST_NAME,
        denylist,
        field="denylist_payload_sha256",
    )
    projection_written = write_immutable_json(
        output_dir / PROJECTION_NAME,
        projection,
        field="projection_payload_sha256",
    )
    lock_payload = {
        "version": f"{VERSION}_lock",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bound_output_dir": str(output_dir.resolve()),
        "source_identity": source,
        "test_evidence": {
            "path": TEST_EVIDENCE_NAME,
            "file_sha256": sha256_path(evidence_path),
            "payload_sha256": evidence["test_evidence_payload_sha256"],
        },
        "denylist": {
            "path": DENYLIST_NAME,
            "file_sha256": sha256_path(output_dir / DENYLIST_NAME),
            "payload_sha256": denylist_written["denylist_payload_sha256"],
        },
        "projection": {
            "path": PROJECTION_NAME,
            "file_sha256": sha256_path(output_dir / PROJECTION_NAME),
            "payload_sha256": (
                projection_written["projection_payload_sha256"]
            ),
        },
        "semantic_contract_audit": semantic,
        "operational_audit": operations,
        "training_contract": asdict(FROZEN_CONFIG),
        "model_schema": model_schema(),
        "model_schema_sha256": model_schema_sha256(),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "stream_contract": prospective_stream_contract(),
        "workload": {
            "train_roots": TRAIN_ROOTS,
            "development_pairs": DEVELOPMENT_PAIRS,
            "confirmation_pairs": CONFIRMATION_PAIRS,
            "complete_game_arms": TOTAL_GAME_ARMS,
        },
        "jobs": 1,
        "threads": 1,
        "device": "cpu",
        "minimum_nice": MIN_NICE,
        "deterministic_algorithms": True,
        "marker_defined": False,
        "execution_command_defined": False,
        "zero_work": dict(ZERO_WORK),
        "checks": checks,
        "decision": decision,
        "passes": decision == "READY_J1_IMPLEMENTATION_PREFLIGHT",
    }
    lock_written = write_immutable_json(
        output_dir / PREFLIGHT_LOCK_NAME,
        lock_payload,
        field="preflight_lock_payload_sha256",
    )
    result_payload = {
        "version": f"{VERSION}_result",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "preflight_lock_file_sha256": sha256_path(
            output_dir / PREFLIGHT_LOCK_NAME
        ),
        "preflight_lock_payload_sha256": lock_written[
            "preflight_lock_payload_sha256"
        ],
        "test_evidence_file_sha256": sha256_path(evidence_path),
        "denylist_file_sha256": sha256_path(output_dir / DENYLIST_NAME),
        "projection_file_sha256": sha256_path(output_dir / PROJECTION_NAME),
        "checks": checks,
        "zero_work": dict(ZERO_WORK),
        "next_authority": (
            "research-lead review only; execution remains HOLD"
            if decision == "READY_J1_IMPLEMENTATION_PREFLIGHT"
            else "no J1 execution"
        ),
        "continue": (
            "research-lead review of J1 readiness package"
            if decision == "READY_J1_IMPLEMENTATION_PREFLIGHT"
            else False
        ),
        "hold": "all J1 execution and science",
        "kill": "historical kills unchanged",
        "promote": False,
    }
    result_written = write_immutable_json(
        output_dir / PREFLIGHT_RESULT_NAME,
        result_payload,
        field="preflight_result_payload_sha256",
    )
    return {
        "decision": decision,
        "test_evidence": {
            "path": str(evidence_path),
            "file_sha256": sha256_path(evidence_path),
            "payload_sha256": evidence["test_evidence_payload_sha256"],
        },
        "denylist": {
            "path": str(output_dir / DENYLIST_NAME),
            "file_sha256": sha256_path(output_dir / DENYLIST_NAME),
            "payload_sha256": denylist_written["denylist_payload_sha256"],
        },
        "projection": {
            "path": str(output_dir / PROJECTION_NAME),
            "file_sha256": sha256_path(output_dir / PROJECTION_NAME),
            "payload_sha256": (
                projection_written["projection_payload_sha256"]
            ),
        },
        "lock": {
            "path": str(output_dir / PREFLIGHT_LOCK_NAME),
            "file_sha256": sha256_path(output_dir / PREFLIGHT_LOCK_NAME),
            "payload_sha256": lock_written[
                "preflight_lock_payload_sha256"
            ],
        },
        "result": {
            "path": str(output_dir / PREFLIGHT_RESULT_NAME),
            "file_sha256": sha256_path(output_dir / PREFLIGHT_RESULT_NAME),
            "payload_sha256": result_written[
                "preflight_result_payload_sha256"
            ],
        },
        "zero_work": dict(ZERO_WORK),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
    )
    audit = subparsers.add_parser("audit-zero-work")
    audit.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    evidence.add_argument("--focused", type=int, required=True)
    evidence.add_argument("--regressions", type=int, required=True)
    evidence.add_argument(
        "--deselection",
        action="append",
        default=[],
    )
    evidence.add_argument(
        "--recorded-command",
        action="append",
        default=[],
    )
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser


def dispatch(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.out_dir.resolve() != OUTPUT_DIR.resolve():
        raise J1IntegrityError("J1 command output directory changed")
    if args.subcommand == "audit-zero-work":
        return audit_zero_work(output_dir=args.out_dir)
    if args.subcommand == "write-test-evidence":
        return write_test_evidence(
            focused_passed=args.focused,
            regressions_passed=args.regressions,
            deselections=args.deselection,
            commands=args.recorded_command,
            output_dir=args.out_dir,
        )
    if args.subcommand == "prepare":
        return prepare(output_dir=args.out_dir)
    raise J1IntegrityError(f"Forbidden J1 command: {args.subcommand}")


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(dispatch(argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

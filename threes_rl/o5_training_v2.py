"""Marker-bound adaptive O5 domain-safe option label and fit execution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from threes_rl import g1r_acquire as history
from threes_rl import o5_four_family_p0 as p0
from threes_rl.o1_geometry_option import pair_safe_merge_actions
from threes_rl.o4_domain_safe_pair_option import (
    CHECKPOINTS,
    EVENT_WIDTH,
    GEOMETRY_WIDTH,
    LINEAGE_A,
    LINEAGE_B,
    OPTION_HORIZON,
    OUTPUT_WIDTH,
    TRAIN_TARGETS,
    DesignatedPair,
    O4DesignatedPairNet,
    advance_lineage_base,
    apply_spawn_to_lineage,
    blocker_geometry,
    build_decision_targets,
    choose_option_action,
    exhaustive_blocker_domain_proof,
    initial_lineage,
    lineage_integrity,
    option_features,
    parameter_count,
    root_option_eligible,
    schema_sha256,
    select_designated_pair,
    successor_geometry,
    transition_status,
)
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import SimState, ThreesSim


VERSION = "o5_domain_safe_training_v2"
STARTER_TILE = 1536

ROOT = Path("threes_rl/runs")
CHARTER_PATH = Path("threes_rl/O5_TRAINING_EXECUTION_CHARTER_V2.md")
RUNNER_PATH = Path("threes_rl/o5_training_v2.py")
TEST_PATH = Path("tests/test_rl_o5_training_v2.py")
TEST_EVIDENCE_PATH = (
    ROOT / "forensics/o5_domain_safe_training_v2_test_evidence.json"
)
OUTPUT_DIR = ROOT / "forensics/o5_domain_safe_training_v2"
CONFIG_PATH = OUTPUT_DIR / "training_config.json"
ROOT_MANIFEST_PATH = OUTPUT_DIR / "selected_root_manifest.json"
SOURCE_AUDIT_PATH = OUTPUT_DIR / "source_audit.json"
TASK_MANIFEST_PATH = OUTPUT_DIR / "learning_task_manifest.json"
COLLISION_PATH = OUTPUT_DIR / "learning_collision_audit.json"
MODEL_IDENTITY_PATH = OUTPUT_DIR / "model_identity.json"
PREFLIGHT_LOCK_PATH = OUTPUT_DIR / "preflight_lock.json"
PREFLIGHT_RESULT_PATH = OUTPUT_DIR / "preflight_result.json"
MARKER_PATH = OUTPUT_DIR / "O5_TRAINING_V2_OPENED.json"
ATTEMPT_PATH = OUTPUT_DIR / "attempts.jsonl"
FIT_ATTEMPT_PATH = OUTPUT_DIR / "fit_attempts.jsonl"
RUNTIME_PATH = OUTPUT_DIR / "runtime_state.json"
EPISODE_DIR = OUTPUT_DIR / "episodes"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SUPPORT_REPORT_PATH = OUTPUT_DIR / "label_support_report.json"
CHECKPOINT_AUTHORITY_PATH = OUTPUT_DIR / "checkpoint_authority.json"
CHECKPOINT_QUARANTINE_PATH = OUTPUT_DIR / "checkpoint_quarantine.json"
RESULT_PATH = OUTPUT_DIR / "training_result.json"

P0_DIR = p0.OUTPUT_DIR
P0_FILES = {
    "marker": P0_DIR / "O5_P0_OPENED.json",
    "result": P0_DIR / "O5_P0_RESULT.json",
    "selection": P0_DIR / "O5_P0_SELECTED_ROOTS.json",
    "streams": P0_DIR / "O5_P0_STREAM_MANIFEST.json",
    "collision": P0_DIR / "O5_P0_COLLISION_AUDIT.json",
    "domain": P0_DIR / "O5_P0_DOMAIN_PROOF.json",
    "policies": P0_DIR / "O5_P0_POLICY_AUDIT.json",
    "power": P0_DIR / "O5_P0_POWER_TABLE.json",
    "source_pool": P0_DIR / "O5_P0_SOURCE_POOL.json",
    "source_replays": P0_DIR / "O5_P0_SOURCE_REPLAY_MANIFEST.json",
    "test_evidence": p0.TEST_EVIDENCE_PATH,
}
EXPECTED_P0_FILE_SHA256 = {
    "marker": "902df97928d2b393c8819887717c213b831f8321ac0270a0761633737b668c13",
    "result": "b2ca5368dd6f29debfd0fb0e4c86005c9bae7b92d736ebc5750c5ec71f97a96f",
    "selection": "d6220ee3ebfe799d78cba128be816e607947a225f7b6ad8add0cc2aad91abad8",
    "streams": "bf114875f9ff24f4456fdf85aa8fcba86f4c9d7eadc3df43f6e931a50eb35186",
    "collision": "21eb13c8a7c1b3ee7f5f110540a06e45dccc58bf1a79f300b490b806d6d1420e",
    "domain": "0ed83b017c6600c118589da8c955c65b1387a295aabf38e1d6da5fff17a79b3d",
    "policies": "283acb7c2d44dd0c4eea776db8a87dfe810297c0be5cf888e18bed2956a5ff8d",
    "power": "ff831d0f388073e16d35c7b80b56fd778d6d3586ec921a5a76f0e2719e51b399",
    "source_pool": "c1de22ef0f018bfe0531d24bdc96d8c67b4128c7fa222a7153a3c92b009a063b",
    "source_replays": "e2fdc621ef1eff216375ccb8b21a2d1ca0ec63c2f4d4745462fc80946cc55f30",
    "test_evidence": "7a3e0edf7a3b1aabfe775e1a14a101b90fb18aa4153fed4518eca88224e39447",
}
EXPECTED_P0_PAYLOAD_SHA256 = {
    "marker": "12cc4b06cb2e9fae81bd569d6681ad15a5ed512152679d201b793d51693512dc",
    "result": "1707c2982e62a29787b69dae9f6e31c9a042162f0573203ab6f2f38d9d3b7fe1",
    "selection": "d4843884f64bf4c167b7d4ba5c5a40bc0aed7988206f6749ec9e6cef3005bf49",
    "streams": "dc9701367a07fcdd491bd693a1aae8f740b7a7e14e7f417cac179f69377e041e",
    "collision": "823f49fad9792e878cb3c8ada9f5eee7e61820246ab2b2f779167ef82afae5e2",
    "domain": "9baf6782d404ab4da6ed53a4a62f3ec61afcb3b155af1efd384ecac9ce3c8b0d",
    "policies": "3b824308ef5a2c7dc73d57770dd80973caffb926ef2e217fdc6c045d69e8d0d5",
    "power": "a04eaafea464cbd0eef458c94d2fcc738eec83e28bc9cb565e6f1e0853244a1d",
    "source_pool": "efbb5cf8c0845119080633045187e62cb7f63f50d47eb6181fd9eed3d2847a7f",
    "source_replays": "eed658004d23b441d9f2fe176d2db1d4d8e9c228bbc722e57bf45bd626db3533",
    "test_evidence": "22fe9b0ca21c9ba10f2060115d20bdd405ce637a11fa7e7d6a9570e8abb2162c",
}

P0_SELECTED_MANIFEST_SHA256 = (
    "05850e87eaa03010e06c27b548d04d22bf22768dfa94d62d0a2a1cba96d20612"
)
P0_FUTURE_STREAM_MANIFEST_SHA256 = (
    "a536eb66a4afc73822bcf0448afe7f8229531c40866555191d9d41bb39048302"
)
EXPECTED_SCHEMA_SHA256 = (
    "60a83881d8e8275a4aa2d03df06815d65e5b247b16f36118009f42f2ce3098ba"
)
EXPECTED_PARAMETER_COUNT = 102_557
EXPECTED_TORCH_VERSION = "2.12.1"

V1_DRAFT_FILES = {
    "charter": Path("threes_rl/O5_TRAINING_EXECUTION_CHARTER.md"),
    "runner": Path("threes_rl/o5_training.py"),
    "tests": Path("tests/test_rl_o5_training.py"),
}
EXPECTED_V1_DRAFT_SHA256 = {
    "charter": "9348f4b24930df6ec2e463bb6f74272c0bd18cadc8b7a7d9a544ad2b04f4952f",
    "runner": "e9baa328f2091e8f51ed287f774b862314af08e828eb2fe032dda4409e06504d",
    "tests": "cfe8a2b4e090d2b3e73b550adeeedb2a0058eab4f40d8e7eb4af28c99d815a9e",
}

DEPENDENCY_PATHS = (
    Path("threes_rl/o4_domain_safe_pair_option.py"),
    Path("threes_rl/o4_power_contract.py"),
    Path("threes_rl/o5_four_family_p0.py"),
    Path("threes_rl/sim.py"),
    Path("threes_rl/g1r_acquire.py"),
    Path("threes_rl/g1r_acquire_v2_qd5.py"),
    Path("threes_rl/g1r_qd_admission_v2.py"),
    Path("threes_rl/replay_provenance.py"),
    Path("threes_rl/current_incumbent_policy.txt"),
    Path("threes_rl/eval.py"),
    Path("threes_rl/ntuple.py"),
    Path("threes_rl/expectimax.py"),
)
EXPECTED_DEPENDENCY_SHA256 = {
    "threes_rl/o4_domain_safe_pair_option.py":
        "95a4da48fb7550e87b09e1f1594cdbdc062a52c7df544b7445b5e58878c87f41",
    "threes_rl/o4_power_contract.py":
        "16e2c26c9e1f2b176937f1a0546604b878d45875b4c29dbc83a441588f7fc5cd",
    "threes_rl/o5_four_family_p0.py":
        "f0ffcc17578581b6e4783e63beef28e59ffab16676ddbd84126127fab47bcff6",
    "threes_rl/sim.py":
        "67e7a245c05e59367402095ad018122fb4cb1ef08664bf28bf4bc03a02a73072",
    "threes_rl/g1r_acquire.py":
        "73ba88103024e6cf62ba4418d88a9bbe71cf42aafc1b911ef39818647f655d6a",
    "threes_rl/g1r_acquire_v2_qd5.py":
        "f195026041e25aeb22ffc72cc57c49d1da96a1af3dfa9fa9180e31345a13d776",
    "threes_rl/g1r_qd_admission_v2.py":
        "191c612d183832bc79ec376322a5c15eae92512360231b6596b307240532c51b",
    "threes_rl/replay_provenance.py":
        "2867cdd23973a4c5464905bf05373a6a0ae3e4439bfd9ac9de1e30892848e992",
    "threes_rl/current_incumbent_policy.txt":
        "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4",
    "threes_rl/eval.py":
        "df0a558014583fcfd24fd8ddf48988e375ad9a6fc5199d35311c40d8b6a3f705",
    "threes_rl/ntuple.py":
        "bdd38ec758ca1786b67a7550b3a2792cbd517176ad99e4df7c5ddd2584953789",
    "threes_rl/expectimax.py":
        "98a7f0d05437d01555ea37d21211fa36d7260cba84456b0fb08799472b26ec14",
}

TRAIN_ROOTS = 192
DEVELOPMENT_ROOTS = 64
UNTOUCHED_ROOTS = 192
ROUNDS = 4
TRAJECTORIES_BY_ROUND = (2, 2, 1, 1)
TRAJECTORIES_PER_ROOT = 6
EPISODES = 1_152
TRAIN_SEED = 2026072804
EPSILON_BY_ROUND = (1.0, 0.15, 0.10, 0.05)
EPOCHS_PER_ROUND = 5
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
GEOMETRY_LOSS_COEFFICIENT = 0.10 / 3.0
MIN_NICE = 10
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
MAX_RUNTIME_SECONDS = 18.0 * 3600.0
MAX_OUTPUT_BYTES = 4 * 1024**3
STREAM_FIELDS = (
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
)
FAMILY_ORDER = (
    "o5_corner2",
    "o5_expectimax2",
    "o5_parent_mc1000",
    "o5_replaycal",
)


class O5TrainingOperationalHold(RuntimeError):
    """A transient resource/service stop that cannot kill O5 science."""


class O5TrainingIntegrityError(RuntimeError):
    """An immutable identity, semantic, or numerical integrity failure."""


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    embedded = body.pop(field, None)
    return isinstance(embedded, str) and embedded == canonical_json_hash(body)


def _write_immutable(
    path: Path,
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Immutable artifact exists: {path}")
    body = _payload_with_hash(payload, field)
    serialized = json.dumps(body, indent=2, sort_keys=True) + "\n"
    if not _verify_self_hash(json.loads(serialized), field):
        raise O5TrainingIntegrityError(f"JSON reload instability: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized)
    os.replace(temporary, path)
    if not _verify_self_hash(json.loads(path.read_text()), field):
        raise O5TrainingIntegrityError(f"Written self hash mismatch: {path}")
    return body


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file()
    )


def _p0_self_hash_field(name: str) -> str:
    return (
        "opened_payload_sha256"
        if name == "marker"
        else "result_payload_sha256"
        if name == "result"
        else "test_evidence_payload_sha256"
        if name == "test_evidence"
        else "payload_sha256"
    )


def p0_input_audit() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name, path in P0_FILES.items():
        observed = sha256_path(path)
        if observed != EXPECTED_P0_FILE_SHA256[name]:
            raise O5TrainingIntegrityError(f"O5 P0 file changed: {name}")
        payload = json.loads(path.read_text())
        field = _p0_self_hash_field(name)
        expected_payload = EXPECTED_P0_PAYLOAD_SHA256[name]
        if not _verify_self_hash(payload, field):
            raise O5TrainingIntegrityError(f"O5 P0 self hash failed: {name}")
        if payload.get(field) != expected_payload:
            raise O5TrainingIntegrityError(
                f"O5 P0 payload changed: {name}"
            )
        files[name] = {
            "path": str(path),
            "file_sha256": observed,
            "payload_sha256": expected_payload,
            "self_hash_field": field,
        }
    result = json.loads(P0_FILES["result"].read_text())
    selection = json.loads(P0_FILES["selection"].read_text())
    streams = json.loads(P0_FILES["streams"].read_text())
    domain = json.loads(P0_FILES["domain"].read_text())
    checks = {
        "p0_ready": (
            result.get("decision")
            == "READY_O5_FOUR_FAMILY_DOMAIN_SAFE_PREFLIGHT"
        ),
        "selection_passes": selection.get("passes") is True,
        "selected_manifest_exact": (
            selection.get("selected_manifest_sha256")
            == P0_SELECTED_MANIFEST_SHA256
        ),
        "stream_manifest_exact": (
            streams.get("contract", {}).get("manifest_sha256")
            == P0_FUTURE_STREAM_MANIFEST_SHA256
        ),
        "streams_unconsumed": streams.get("streams_consumed") == 0,
        "domain_passes": domain.get("passes") is True,
        "schema_exact": (
            domain.get("schema_sha256") == EXPECTED_SCHEMA_SHA256
        ),
        "parameter_count_exact": (
            domain.get("parameter_count") == EXPECTED_PARAMETER_COUNT
        ),
    }
    if not all(checks.values()):
        raise O5TrainingIntegrityError(f"O5 P0 contract failed: {checks}")
    return {"files": files, "checks": checks, "passes": True}


def v1_draft_audit() -> dict[str, Any]:
    rows = {}
    for name, path in V1_DRAFT_FILES.items():
        observed = sha256_path(path)
        if observed != EXPECTED_V1_DRAFT_SHA256[name]:
            raise O5TrainingIntegrityError(f"O5 V1 draft changed: {name}")
        rows[name] = {"path": str(path), "sha256": observed}
    return {"files": rows, "passes": True, "authoritative": False}


def policy_checkpoint_audit() -> dict[str, Any]:
    marker = json.loads(P0_FILES["marker"].read_text())
    immutable = marker.get("immutable_file_hashes", {})
    immutable_rows: list[dict[str, Any]] = []
    for raw_path, expected in sorted(immutable.items()):
        path = Path(raw_path)
        observed = sha256_path(path)
        if observed != expected:
            raise O5TrainingIntegrityError(
                f"P0 immutable input changed: {path}"
            )
        immutable_rows.append(
            {"path": str(path), "sha256": observed}
        )

    policy_path = Path(
        "threes_rl/runs/forensics/o3_event_option_p0_v1/"
        "O3_P0_POLICY_AUDIT.json"
    )
    prior = json.loads(policy_path.read_text())
    if not _verify_self_hash(prior, "payload_sha256"):
        raise O5TrainingIntegrityError("O5 policy-lock self hash failed")
    if (
        sha256_path(policy_path)
        != immutable[str(policy_path)]
        or not _verify_self_hash(
            prior["policy_lock"],
            "policy_lock_sha256",
        )
        or prior["policy_lock"]["policy_lock_sha256"]
        != prior["policy_lock_sha256"]
    ):
        raise O5TrainingIntegrityError("O5 policy-lock identity changed")
    policy_lock = prior["policy_lock"]
    if (
        policy_lock.get("incumbent_policy_file_sha256")
        != EXPECTED_DEPENDENCY_SHA256[
            "threes_rl/current_incumbent_policy.txt"
        ]
        or sha256_path(
            Path(str(policy_lock["incumbent_policy_file"]))
        )
        != policy_lock["incumbent_policy_file_sha256"]
    ):
        raise O5TrainingIntegrityError("O5 incumbent identity changed")

    included_source_families = {
        "g1r_corner2",
        "g1r_expectimax2",
        "g1r_parent_mc1000",
        "g1r_replaycal",
    }
    checkpoint_rows: list[dict[str, Any]] = []
    included_families: list[str] = []
    for family in policy_lock["families"]:
        name = str(family["family"])
        if name not in included_source_families:
            continue
        included_families.append(name)
        for manifest in family.get("checkpoint_manifests", ()):
            for item in manifest["files"]:
                path = Path(str(item["path"]))
                observed = sha256_path(path)
                if (
                    observed != item["sha256"]
                    or int(path.stat().st_size) != int(item["bytes"])
                ):
                    raise O5TrainingIntegrityError(
                        f"O5 checkpoint artifact changed: {path}"
                    )
                checkpoint_rows.append(
                    {
                        "family": name,
                        "path": str(path),
                        "sha256": observed,
                        "bytes": int(item["bytes"]),
                    }
                )
    if set(included_families) != included_source_families:
        raise O5TrainingIntegrityError("O5 policy family set changed")
    for raw_path, expected in policy_lock.get(
        "source_hashes",
        {},
    ).items():
        if sha256_path(Path(raw_path)) != expected:
            raise O5TrainingIntegrityError(
                f"O5 policy source changed: {raw_path}"
            )
    return {
        "policy_audit_path": str(policy_path),
        "policy_audit_file_sha256": sha256_path(policy_path),
        "policy_audit_payload_sha256": prior["payload_sha256"],
        "policy_lock_sha256": prior["policy_lock_sha256"],
        "included_families": included_families,
        "checkpoint_file_count": len(checkpoint_rows),
        "checkpoint_total_bytes": sum(
            row["bytes"] for row in checkpoint_rows
        ),
        "checkpoint_manifest_sha256": canonical_json_hash(
            checkpoint_rows
        ),
        "immutable_input_count": len(immutable_rows),
        "immutable_inputs_sha256": canonical_json_hash(immutable_rows),
        "passes": True,
    }


def dependency_audit() -> dict[str, Any]:
    marker = json.loads(P0_FILES["marker"].read_text())
    p0_dependencies = marker.get("dependency_hashes", {})
    observed = {str(path): sha256_path(path) for path in DEPENDENCY_PATHS}
    for path, expected in EXPECTED_DEPENDENCY_SHA256.items():
        if observed.get(path) != expected:
            raise O5TrainingIntegrityError(
                f"Frozen O5 dependency changed: {path}"
            )
    for path, expected in p0_dependencies.items():
        if observed.get(path) != expected:
            raise O5TrainingIntegrityError(
                f"P0 dependency changed: {path}"
            )
    checks = {
        "p0_dependency_subset_exact": all(
            observed.get(path) == expected
            for path, expected in p0_dependencies.items()
        ),
        "o4_operator_exact": (
            observed["threes_rl/o4_domain_safe_pair_option.py"]
            == EXPECTED_DEPENDENCY_SHA256[
                "threes_rl/o4_domain_safe_pair_option.py"
            ]
        ),
        "all_dependencies_present": len(observed) == len(DEPENDENCY_PATHS),
    }
    policy = policy_checkpoint_audit()
    checks["policy_and_checkpoint_files_exact"] = policy["passes"]
    return {
        "files": observed,
        "p0_dependency_hashes": p0_dependencies,
        "policy_and_checkpoints": policy,
        "checks": checks,
        "passes": all(checks.values()),
    }


def training_config() -> dict[str, Any]:
    return {
        "version": VERSION,
        "schema_sha256": schema_sha256(),
        "parameter_count": parameter_count(),
        "roots": {
            "train": TRAIN_ROOTS,
            "development_hash_only": DEVELOPMENT_ROOTS,
            "untouched_hash_only": UNTOUCHED_ROOTS,
        },
        "rounds": ROUNDS,
        "trajectories_by_round": list(TRAJECTORIES_BY_ROUND),
        "trajectories_per_root": TRAJECTORIES_PER_ROOT,
        "episodes": EPISODES,
        "epsilon_by_round": list(EPSILON_BY_ROUND),
        "adaptive_sequence": (
            "collect_round_then_five_cumulative_epochs_with_continued_"
            "model_and_optimizer"
        ),
        "epochs_per_round": EPOCHS_PER_ROUND,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "geometry_loss_coefficient_per_checkpoint":
            GEOMETRY_LOSS_COEFFICIENT,
        "training_seed": TRAIN_SEED,
        "epoch_permutation_seed": (
            "2026072804 + 100*round_number_1_based + epoch_index_0_based"
        ),
        "device": "cpu",
        "torch_version": torch.__version__,
        "threads": 1,
        "checkpoint_authority_before_support": False,
        "score_target_used": False,
        "human_or_behavior_action_label_used": False,
        "counterfactual_action_labels_used": False,
        "development_replay_content_opened": False,
        "untouched_replay_content_opened": False,
        "runtime_limit_seconds": MAX_RUNTIME_SECONDS,
        "output_limit_bytes": MAX_OUTPUT_BYTES,
    }


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def initialize_training() -> tuple[
    O4DesignatedPairNet,
    torch.optim.Optimizer,
]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(TRAIN_SEED)
    model = O4DesignatedPairNet().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    return model, optimizer


def model_identity() -> dict[str, Any]:
    config = training_config()
    if config["schema_sha256"] != EXPECTED_SCHEMA_SHA256:
        raise O5TrainingIntegrityError("O5 schema hash changed")
    if config["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise O5TrainingIntegrityError("O5 parameter count changed")
    if config["torch_version"] != EXPECTED_TORCH_VERSION:
        raise O5TrainingIntegrityError(
            f"Frozen PyTorch changed: {config['torch_version']}"
        )
    model, optimizer = initialize_training()
    return {
        "schema_sha256": config["schema_sha256"],
        "parameter_count": config["parameter_count"],
        "torch_version": config["torch_version"],
        "initial_model_state_sha256": model_state_sha256(model),
        "initial_optimizer_state": optimizer.state_dict(),
        "domain_proof": exhaustive_blocker_domain_proof(),
        "passes": True,
    }


def load_selected_rows() -> list[dict[str, Any]]:
    payload = json.loads(P0_FILES["selection"].read_text())
    if not _verify_self_hash(payload, "payload_sha256"):
        raise O5TrainingIntegrityError("O5 selected-root self hash failed")
    if (
        payload.get("payload_sha256")
        != EXPECTED_P0_PAYLOAD_SHA256["selection"]
        or payload.get("selected_manifest_sha256")
        != P0_SELECTED_MANIFEST_SHA256
        or payload.get("passes") is not True
        or payload.get("deficits") != []
    ):
        raise O5TrainingIntegrityError("O5 selected-root identity changed")
    raw = payload.get("selected")
    if not isinstance(raw, list) or len(raw) != 448:
        raise O5TrainingIntegrityError("O5 selected-root count changed")
    rows = [dict(row) for row in raw]
    role_counts = Counter(str(row.get("role")) for row in rows)
    if role_counts != Counter(
        {
            "train": TRAIN_ROOTS,
            "development": DEVELOPMENT_ROOTS,
            "untouched_mechanism": UNTOUCHED_ROOTS,
        }
    ):
        raise O5TrainingIntegrityError(
            f"O5 selected role counts changed: {role_counts}"
        )
    roots = [str(row["root_cluster"]) for row in rows]
    if len(roots) != len(set(roots)):
        raise O5TrainingIntegrityError("O5 selected roots overlap")
    return rows


def _find_selected_frame(
    replay: Mapping[str, Any],
    frame_index: int,
) -> Mapping[str, Any]:
    frames = replay.get("frames")
    if not isinstance(frames, list):
        raise O5TrainingIntegrityError("Selected replay lacks frames")
    matches = [
        frame
        for fallback, frame in enumerate(frames)
        if isinstance(frame, Mapping)
        and int(frame.get("index", fallback)) == int(frame_index)
    ]
    if len(matches) != 1:
        raise O5TrainingIntegrityError(
            f"Selected frame {frame_index} is not unique"
        )
    return matches[0]


def restore_train_root(
    row: Mapping[str, Any],
) -> tuple[SimState, DesignatedPair]:
    if row.get("role") != "train":
        raise O5TrainingIntegrityError(
            "Training restore received a non-train root"
        )
    path = Path(str(row["source_replay"]))
    if sha256_path(path) != row["source_replay_sha256"]:
        raise O5TrainingIntegrityError(f"Train source changed: {path}")
    replay = json.loads(path.read_text())
    frame = _find_selected_frame(replay, int(row["frame_index"]))
    state_payload = frame.get("state")
    if not isinstance(state_payload, Mapping):
        raise O5TrainingIntegrityError("Selected state is missing")
    state, identity = p0.whitelisted_state_payload(state_payload)
    if canonical_json_hash(identity) != row["o5_whitelisted_state_sha256"]:
        raise O5TrainingIntegrityError("Whitelisted state hash changed")
    simulator = ThreesSim.from_stream_ids(
        deck_stream_id=0,
        slot_stream_id=1,
        starter_tile=STARTER_TILE,
    )
    target = int(row["target"])
    pair = select_designated_pair(
        state.board,
        STARTER_TILE,
        requested_target=target,
        allowed_targets=TRAIN_TARGETS,
    )
    if pair is None or pair.safe_merge_actions:
        raise O5TrainingIntegrityError("Selected O5 pair is not a hard start")
    if [list(value) for value in pair.coordinates] != row["pair"]:
        raise O5TrainingIntegrityError("Selected O5 pair changed")
    if not root_option_eligible(
        state,
        simulator,
        STARTER_TILE,
        allowed_targets=(target,),
    ):
        raise O5TrainingIntegrityError("Selected O5 root is ineligible")
    legal = tuple(int(action) for action in simulator.legal_actions(state))
    if len(legal) != int(row["legal_count"]):
        raise O5TrainingIntegrityError("Selected O5 legal count changed")
    lineage = initial_lineage(pair)
    for action in legal:
        tokens, globals_array = option_features(
            state,
            simulator,
            starter_tile=STARTER_TILE,
            pair=pair,
            lineage=lineage,
            action=action,
        )
        if not (
            np.isfinite(tokens).all()
            and np.isfinite(globals_array).all()
            and np.all((0.0 <= tokens) & (tokens <= 1.0))
            and np.all((0.0 <= globals_array) & (globals_array <= 1.0))
        ):
            raise O5TrainingIntegrityError(
                "Selected O5 feature domain failed"
            )
    return state, pair


def source_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train_rows = [row for row in rows if row.get("role") == "train"]
    sealed_rows = [row for row in rows if row.get("role") != "train"]
    train_audit: list[dict[str, Any]] = []
    for root_index, row in enumerate(train_rows):
        state, pair = restore_train_root(row)
        train_audit.append(
            {
                "root_index": root_index,
                "root_cluster": str(row["root_cluster"]),
                "family": str(row["family"]),
                "target": int(row["target"]),
                "source_replay": str(row["source_replay"]),
                "source_replay_sha256": str(
                    row["source_replay_sha256"]
                ),
                "frame_index": int(row["frame_index"]),
                "whitelisted_state_sha256": str(
                    row["o5_whitelisted_state_sha256"]
                ),
                "pair": [list(value) for value in pair.coordinates],
                "board_shape": list(state.board.shape),
                "restored": True,
            }
        )
    sealed_audit: list[dict[str, Any]] = []
    for row in sealed_rows:
        # No geometry/action/target/frame key is accessed on holdout rows.
        path = Path(str(row["source_replay"]))
        observed = sha256_path(path)
        if observed != row["source_replay_sha256"]:
            raise O5TrainingIntegrityError(
                f"Hash-only holdout source changed: {path}"
            )
        sealed_audit.append(
            {
                "role": str(row["role"]),
                "root_cluster": str(row["root_cluster"]),
                "source_replay": str(path),
                "source_replay_sha256": observed,
                "content_opened": False,
            }
        )
    role_counts = Counter(str(row["role"]) for row in sealed_audit)
    checks = {
        "train_roots_restored_exact_192": len(train_audit) == TRAIN_ROOTS,
        "development_hash_only_exact_64": (
            role_counts["development"] == DEVELOPMENT_ROOTS
        ),
        "untouched_hash_only_exact_192": (
            role_counts["untouched_mechanism"] == UNTOUCHED_ROOTS
        ),
        "development_content_unopened": True,
        "untouched_content_unopened": True,
        "forbidden_fields_unread": True,
        "o3_training_bodies_unread": True,
    }
    return {
        "train_rows": train_audit,
        "sealed_rows": sealed_audit,
        "checks": checks,
        "passes": all(checks.values()),
    }


def learning_rows() -> list[dict[str, Any]]:
    payload = json.loads(P0_FILES["streams"].read_text())
    if (
        not _verify_self_hash(payload, "payload_sha256")
        or payload.get("payload_sha256")
        != EXPECTED_P0_PAYLOAD_SHA256["streams"]
    ):
        raise O5TrainingIntegrityError("O5 stream artifact changed")
    rows = [
        dict(row)
        for row in payload.get("rows", ())
        if row.get("purpose") == "learning"
    ]
    rows.sort(
        key=lambda row: (
            int(row["round_index"]),
            int(row["root_index"]),
            int(row["trajectory_index"]),
        )
    )
    expected = []
    offsets = (0, 2, 4, 5)
    for round_index, trajectories in enumerate(
        TRAJECTORIES_BY_ROUND,
        start=1,
    ):
        for root_index in range(TRAIN_ROOTS):
            for within_round in range(trajectories):
                expected.append(
                    (
                        round_index,
                        root_index,
                        offsets[round_index - 1] + within_round,
                    )
                )
    observed = [
        (
            int(row["round_index"]),
            int(row["root_index"]),
            int(row["trajectory_index"]),
        )
        for row in rows
    ]
    if observed != expected or len(rows) != EPISODES:
        raise O5TrainingIntegrityError("O5 learning schedule changed")
    for row in rows:
        row["replicate_in_round"] = (
            int(row["trajectory_index"])
            - offsets[int(row["round_index"]) - 1]
        )
    for field in STREAM_FIELDS:
        values = [int(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise O5TrainingIntegrityError(
                f"O5 learning stream repeats: {field}"
            )
    if canonical_json_hash(
        [
            {
                key: value
                for key, value in row.items()
                if key != "replicate_in_round"
            }
            for row in sorted(
                rows,
                key=lambda item: (
                    int(item["root_index"]),
                    int(item["trajectory_index"]),
                ),
            )
        ]
    ) != canonical_json_hash(
        [
            row
            for row in payload["rows"]
            if row.get("purpose") == "learning"
        ]
    ):
        raise O5TrainingIntegrityError("O5 learning rows are not exact")
    return rows


def _task_id(task: Mapping[str, Any]) -> str:
    return (
        f"r{int(task['round_index'])}_"
        f"root{int(task['root_index']):03d}_"
        f"traj{int(task['trajectory_index'])}"
    )


def validate_attempt_ledger(
    tasks: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    task_by_id = {_task_id(task): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise O5TrainingIntegrityError("O5 task IDs repeat")
    states = {
        task_id: {
            "opened": False,
            "completed": False,
            "resume_count": 0,
        }
        for task_id in task_by_id
    }
    source = _read_jsonl(ATTEMPT_PATH) if rows is None else rows
    for row_index, raw in enumerate(source):
        row = dict(raw)
        task_id = str(row.get("task_id", ""))
        if task_id not in task_by_id:
            raise O5TrainingIntegrityError(
                f"Unknown task in attempt row {row_index}"
            )
        task = task_by_id[task_id]
        state = states[task_id]
        status = row.get("status")
        if status == "opened":
            if state["opened"] or state["completed"]:
                raise O5TrainingIntegrityError(
                    f"Duplicate task open: {task_id}"
                )
            identity = {
                "root_index": int(task["root_index"]),
                "round_index": int(task["round_index"]),
                "trajectory_index": int(task["trajectory_index"]),
            }
            streams = {
                field: int(task[field]) for field in STREAM_FIELDS
            }
            if any(row.get(key) != value for key, value in identity.items()):
                raise O5TrainingIntegrityError(
                    f"Attempt identity changed: {task_id}"
                )
            if row.get("stream_ids") != streams:
                raise O5TrainingIntegrityError(
                    f"Attempt streams changed: {task_id}"
                )
            state["opened"] = True
        elif status == "resumed_same_stream":
            if not state["opened"] or state["completed"]:
                raise O5TrainingIntegrityError(
                    f"Invalid task resume: {task_id}"
                )
            state["resume_count"] += 1
        elif status == "completed":
            if not state["opened"] or state["completed"]:
                raise O5TrainingIntegrityError(
                    f"Duplicate/unopened task close: {task_id}"
                )
            if not isinstance(row.get("artifact_sha256"), str):
                raise O5TrainingIntegrityError(
                    f"Task close lacks artifact SHA: {task_id}"
                )
            state["completed"] = True
        else:
            raise O5TrainingIntegrityError(
                f"Unknown attempt status at row {row_index}: {status}"
            )
    return states


def _stream_sets(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, set[int]]:
    return {
        field: {int(row[field]) for row in rows}
        for field in STREAM_FIELDS
    }


def collision_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    scan_root: Path = ROOT,
) -> dict[str, Any]:
    requested = _stream_sets(rows)
    if any(len(values) != EPISODES for values in requested.values()):
        raise O5TrainingIntegrityError("Requested learning streams repeat")
    found: dict[str, set[int]] = defaultdict(set)
    matched: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    excluded_dirs = (
        P0_DIR.resolve(),
        OUTPUT_DIR.resolve(),
        p0.O3_OPTION_TRAINING_DIR.resolve(),
        (p0.O3_ACQUISITION_DIR / "source_replays").resolve(),
        (p0.O3_RECOVERY_DIR / "source_replays").resolve(),
    )
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file() or path.suffix not in {
            ".json",
            ".jsonl",
            ".csv",
        }:
            continue
        resolved = path.resolve()
        classification = next(
            (
                "internal_or_protected_unread"
                for directory in excluded_dirs
                if resolved == directory or directory in resolved.parents
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
                "counts": {
                    field: len(items)
                    for field, items in sorted(values.items())
                },
            }
        )
    collisions: dict[str, list[int]] = {}
    for field, values in requested.items():
        prior = set(found.get(field, set()))
        if field == "logical_seed":
            for alias in (
                "seed",
                "root_seed",
                "source_seed",
                "fresh_root_seed",
            ):
                prior.update(found.get(alias, set()))
        collisions[field] = sorted(values & prior)
    p0_streams = json.loads(P0_FILES["streams"].read_text())
    exact_learning = [
        dict(row)
        for row in p0_streams["rows"]
        if row.get("purpose") == "learning"
    ]
    checks = {
        "requested_exact_1152": len(rows) == EPISODES,
        "requested_is_exact_p0_learning_set": (
            {
                tuple(int(row[field]) for field in STREAM_FIELDS)
                for row in rows
            }
            == {
                tuple(int(row[field]) for field in STREAM_FIELDS)
                for row in exact_learning
            }
        ),
        "p0_collision_artifact_exact": (
            sha256_path(P0_FILES["collision"])
            == EXPECTED_P0_FILE_SHA256["collision"]
        ),
        "zero_external_collisions": not any(collisions.values()),
        "protected_bodies_unread": True,
    }
    return {
        "requested_rows": len(rows),
        "requested_sha256": canonical_json_hash(list(rows)),
        "matched_sources": matched,
        "matched_source_count": len(matched),
        "matched_sources_sha256": canonical_json_hash(matched),
        "excluded_sources": excluded,
        "excluded_source_count": len(excluded),
        "collisions": collisions,
        "checks": checks,
        "passes": all(checks.values()),
    }


def operational_audit() -> dict[str, Any]:
    process = p0._heavy_process_audit()
    services = history.service_health()
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    output_bytes = _directory_bytes(OUTPUT_DIR)
    checks = {
        "nice_at_least_10": history.current_nice() >= MIN_NICE,
        "no_competing_heavy_process": process["passes"],
        "free_disk_above_100_gib": free_gib > MIN_FREE_GIB,
        "output_below_4_gib": output_bytes < MAX_OUTPUT_BYTES,
        "services_healthy": services["passes"],
    }
    return {
        "nice": history.current_nice(),
        "free_gib": free_gib,
        "target_free_gib_met": free_gib >= TARGET_FREE_GIB,
        "output_bytes": output_bytes,
        "process": process,
        "services": services,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _require_operational(stage: str) -> dict[str, Any]:
    try:
        audit = operational_audit()
    except Exception as error:
        raise O5TrainingOperationalHold(
            f"{stage} operational audit failed: {error}"
        ) from error
    if not audit["passes"]:
        failed = sorted(
            name for name, passed in audit["checks"].items() if not passed
        )
        raise O5TrainingOperationalHold(
            f"{stage} operational guard failed: {failed}"
        )
    return audit


def _train_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "root_cluster",
        "family",
        "source_family",
        "target",
        "source_replay",
        "source_replay_sha256",
        "frame_index",
        "state_sha1",
        "o5_whitelisted_state_sha256",
        "pair",
        "legal_count",
        "deck_stream_id",
        "slot_stream_id",
        "selection_sha256",
    )
    return {"role": "train", **{key: row[key] for key in keys}}


def _sealed_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "role",
        "root_cluster",
        "source_replay",
        "source_replay_sha256",
        "selection_sha256",
    )
    return {key: row[key] for key in keys}


def build_root_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train = [
        _train_manifest_row(row)
        for row in rows
        if row.get("role") == "train"
    ]
    sealed = [
        _sealed_manifest_row(row)
        for row in rows
        if row.get("role") != "train"
    ]
    if len(train) != TRAIN_ROOTS or len(sealed) != (
        DEVELOPMENT_ROOTS + UNTOUCHED_ROOTS
    ):
        raise O5TrainingIntegrityError("O5 root manifest count changed")
    return {
        "version": f"{VERSION}_roots",
        "p0_selected_manifest_sha256": P0_SELECTED_MANIFEST_SHA256,
        "train_rows": train,
        "sealed_holdout_rows": sealed,
        "train_manifest_sha256": canonical_json_hash(train),
        "sealed_holdout_manifest_sha256": canonical_json_hash(sealed),
        "role_counts": {
            "train": TRAIN_ROOTS,
            "development": DEVELOPMENT_ROOTS,
            "untouched_mechanism": UNTOUCHED_ROOTS,
        },
        "development_content_opened": False,
        "untouched_content_opened": False,
    }


def _binding_manifest() -> dict[str, Any]:
    return {
        "p0": p0_input_audit(),
        "v1_drafts": v1_draft_audit(),
        "dependencies": dependency_audit(),
        "charter": {
            "path": str(CHARTER_PATH),
            "sha256": sha256_path(CHARTER_PATH),
        },
        "runner": {
            "path": str(RUNNER_PATH),
            "sha256": sha256_path(RUNNER_PATH),
        },
        "tests": {
            "path": str(TEST_PATH),
            "sha256": sha256_path(TEST_PATH),
        },
    }


def _test_evidence_identity() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not _verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise O5TrainingIntegrityError("V2 test evidence self hash failed")
    expected = {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise O5TrainingIntegrityError("V2 test evidence source changed")
    if payload.get("passes") is not True:
        raise O5TrainingIntegrityError("V2 test evidence is not passing")
    if any(
        int(payload.get(key, -1)) != 0
        for key in (
            "streams_consumed",
            "labels",
            "optimizer_steps",
            "checkpoints",
            "policy_outcomes",
        )
    ):
        raise O5TrainingIntegrityError("V2 test evidence is not zero-work")
    return {
        "path": str(TEST_EVIDENCE_PATH),
        "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "payload_sha256": payload["test_evidence_payload_sha256"],
        "focused_tests_passed": int(payload["focused_tests_passed"]),
        "regression_tests_passed": int(payload["regression_tests_passed"]),
        "passes": True,
    }


def _commands() -> dict[str, str]:
    prefix = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.o5_training_v2"
    )
    return {
        command: f"{prefix} {command} --out-dir {OUTPUT_DIR}'"
        for command in ("prepare", "open", "execute")
    }


def _write_preflight_artifact(
    path: Path,
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    return _write_immutable(path, payload, field)


def prepare(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise O5TrainingIntegrityError("V2 output directory is frozen")
    if out_dir.exists():
        raise FileExistsError(f"V2 namespace exists: {out_dir}")
    try:
        bindings = _binding_manifest()
        evidence = _test_evidence_identity()
        config = training_config()
        identity = model_identity()
        selected_rows = load_selected_rows()
        roots = build_root_manifest(selected_rows)
        sources = source_audit(selected_rows)
        tasks = learning_rows()
        collision = collision_audit(tasks)
        operations = operational_audit()
        checks = {
            "bindings_exact": bool(bindings),
            "tests_pass": evidence["passes"],
            "config_exact": (
                config["schema_sha256"] == EXPECTED_SCHEMA_SHA256
                and config["parameter_count"] == EXPECTED_PARAMETER_COUNT
            ),
            "model_identity_exact": identity["passes"],
            "root_counts_exact": (
                len(roots["train_rows"]) == TRAIN_ROOTS
                and len(roots["sealed_holdout_rows"])
                == DEVELOPMENT_ROOTS + UNTOUCHED_ROOTS
            ),
            "train_restore_exact": sources["passes"],
            "holdouts_hash_only": (
                not sources["development_content_opened"]
                if "development_content_opened" in sources
                else sources["checks"]["development_content_unopened"]
            ),
            "tasks_exact_1152": len(tasks) == EPISODES,
            "streams_collision_free": collision["passes"],
            "operations_pass": operations["passes"],
            "zero_training_work": True,
        }
        decision = (
            "READY_O5_TRAINING_V2_EXECUTION"
            if all(checks.values())
            else "HOLD_O5_TRAINING_V2_PREFLIGHT"
        )
        out_dir.mkdir(parents=True, exist_ok=False)
        config_artifact = _write_preflight_artifact(
            CONFIG_PATH,
            config,
            "config_payload_sha256",
        )
        root_artifact = _write_preflight_artifact(
            ROOT_MANIFEST_PATH,
            roots,
            "root_payload_sha256",
        )
        source_artifact = _write_preflight_artifact(
            SOURCE_AUDIT_PATH,
            sources,
            "source_audit_payload_sha256",
        )
        task_artifact = _write_preflight_artifact(
            TASK_MANIFEST_PATH,
            {
                "version": f"{VERSION}_tasks",
                "rows": tasks,
                "task_manifest_sha256": canonical_json_hash(tasks),
            },
            "task_payload_sha256",
        )
        collision_artifact = _write_preflight_artifact(
            COLLISION_PATH,
            collision,
            "collision_payload_sha256",
        )
        identity_artifact = _write_preflight_artifact(
            MODEL_IDENTITY_PATH,
            identity,
            "model_identity_payload_sha256",
        )
        lock_payload = {
            "version": f"{VERSION}_preflight_lock",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "bound_out_dir": str(out_dir.resolve()),
            "bindings": bindings,
            "test_evidence": evidence,
            "config": {
                "path": str(CONFIG_PATH),
                "file_sha256": sha256_path(CONFIG_PATH),
                "payload_sha256": config_artifact[
                    "config_payload_sha256"
                ],
            },
            "roots": {
                "path": str(ROOT_MANIFEST_PATH),
                "file_sha256": sha256_path(ROOT_MANIFEST_PATH),
                "payload_sha256": root_artifact["root_payload_sha256"],
            },
            "source_audit": {
                "path": str(SOURCE_AUDIT_PATH),
                "file_sha256": sha256_path(SOURCE_AUDIT_PATH),
                "payload_sha256": source_artifact[
                    "source_audit_payload_sha256"
                ],
            },
            "tasks": {
                "path": str(TASK_MANIFEST_PATH),
                "file_sha256": sha256_path(TASK_MANIFEST_PATH),
                "payload_sha256": task_artifact["task_payload_sha256"],
            },
            "collision": {
                "path": str(COLLISION_PATH),
                "file_sha256": sha256_path(COLLISION_PATH),
                "payload_sha256": collision_artifact[
                    "collision_payload_sha256"
                ],
            },
            "model_identity": {
                "path": str(MODEL_IDENTITY_PATH),
                "file_sha256": sha256_path(MODEL_IDENTITY_PATH),
                "payload_sha256": identity_artifact[
                    "model_identity_payload_sha256"
                ],
            },
            "commands": _commands(),
            "operations": operations,
            "counts": {
                "train_roots": TRAIN_ROOTS,
                "development_roots_hash_only": DEVELOPMENT_ROOTS,
                "untouched_roots_hash_only": UNTOUCHED_ROOTS,
                "learning_tasks": EPISODES,
            },
            "checks": checks,
            "zero_work": {
                "streams": 0,
                "episodes": 0,
                "labels": 0,
                "optimizer_steps": 0,
                "checkpoints": 0,
                "development_content_opened": False,
                "untouched_content_opened": False,
                "policy_outcomes": 0,
            },
        }
        lock = _write_preflight_artifact(
            PREFLIGHT_LOCK_PATH,
            lock_payload,
            "preflight_lock_payload_sha256",
        )
        result = _write_preflight_artifact(
            PREFLIGHT_RESULT_PATH,
            {
                "version": f"{VERSION}_preflight_result",
                "decision": decision,
                "continue": decision == "READY_O5_TRAINING_V2_EXECUTION",
                "hold": decision != "READY_O5_TRAINING_V2_EXECUTION",
                "kill": False,
                "promote": False,
                "preflight_lock_file_sha256": sha256_path(
                    PREFLIGHT_LOCK_PATH
                ),
                "preflight_lock_payload_sha256": lock[
                    "preflight_lock_payload_sha256"
                ],
                "checks": checks,
                "zero_work": lock_payload["zero_work"],
            },
            "preflight_result_payload_sha256",
        )
        return {
            "decision": decision,
            "preflight_lock_file_sha256": sha256_path(
                PREFLIGHT_LOCK_PATH
            ),
            "preflight_lock_payload_sha256": lock[
                "preflight_lock_payload_sha256"
            ],
            "preflight_result_file_sha256": sha256_path(
                PREFLIGHT_RESULT_PATH
            ),
            "preflight_result_payload_sha256": result[
                "preflight_result_payload_sha256"
            ],
        }
    except O5TrainingOperationalHold as error:
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
        result = _write_immutable(
            PREFLIGHT_RESULT_PATH,
            {
                "version": f"{VERSION}_preflight_result",
                "decision": "HOLD_O5_TRAINING_V2_PREFLIGHT",
                "continue": False,
                "hold": True,
                "kill": False,
                "promote": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "zero_work": True,
            },
            "preflight_result_payload_sha256",
        )
        return {
            "decision": result["decision"],
            "error": str(error),
        }
    except Exception as error:
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
        result = _write_immutable(
            PREFLIGHT_RESULT_PATH,
            {
                "version": f"{VERSION}_preflight_result",
                "decision": "KILL_O5_TRAINING_INTEGRITY",
                "continue": False,
                "hold": False,
                "kill": True,
                "promote": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "zero_work": True,
            },
            "preflight_result_payload_sha256",
        )
        return {
            "decision": result["decision"],
            "error": str(error),
        }


def _load_preflight_lock() -> dict[str, Any]:
    payload = json.loads(PREFLIGHT_LOCK_PATH.read_text())
    if not _verify_self_hash(payload, "preflight_lock_payload_sha256"):
        raise O5TrainingIntegrityError("V2 preflight lock self hash failed")
    if payload.get("decision") != "READY_O5_TRAINING_V2_EXECUTION":
        raise O5TrainingIntegrityError("V2 preflight is not READY")
    if payload.get("bound_out_dir") != str(OUTPUT_DIR.resolve()):
        raise O5TrainingIntegrityError("V2 preflight out-dir changed")
    current = _binding_manifest()
    if payload.get("bindings") != current:
        raise O5TrainingIntegrityError("V2 preflight bindings changed")
    for name in (
        "test_evidence",
        "config",
        "roots",
        "source_audit",
        "tasks",
        "collision",
        "model_identity",
    ):
        item = payload[name]
        path = Path(str(item["path"]))
        expected = item.get("file_sha256")
        if expected is None:
            expected = sha256_path(path)
        if sha256_path(path) != expected:
            raise O5TrainingIntegrityError(
                f"V2 preflight artifact changed: {name}"
            )
    return payload


def open_execution() -> dict[str, Any]:
    if MARKER_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("V2 marker or terminal result exists")
    lock = _load_preflight_lock()
    forbidden = (
        ATTEMPT_PATH,
        FIT_ATTEMPT_PATH,
        RUNTIME_PATH,
        EPISODE_DIR,
        CHECKPOINT_DIR,
        SUPPORT_REPORT_PATH,
        CHECKPOINT_AUTHORITY_PATH,
        CHECKPOINT_QUARANTINE_PATH,
    )
    if any(path.exists() for path in forbidden):
        raise O5TrainingIntegrityError("V2 work exists before marker")
    tasks = learning_rows()
    collision = collision_audit(tasks)
    operations = _require_operational("V2 open")
    if not collision["passes"]:
        raise O5TrainingIntegrityError("V2 open collision recheck failed")
    marker = _write_immutable(
        MARKER_PATH,
        {
            "version": f"{VERSION}_opened",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "preflight_lock_file_sha256": sha256_path(
                PREFLIGHT_LOCK_PATH
            ),
            "preflight_lock_payload_sha256": lock[
                "preflight_lock_payload_sha256"
            ],
            "execute_command": lock["commands"]["execute"],
            "collision_recheck": collision,
            "operations": operations,
            "zero_work_before_marker": {
                "streams": 0,
                "episodes": 0,
                "labels": 0,
                "optimizer_steps": 0,
                "checkpoints": 0,
                "development_content_opened": False,
                "untouched_content_opened": False,
                "policy_outcomes": 0,
            },
        },
        "marker_payload_sha256",
    )
    return {
        "decision": "READY_O5_TRAINING_V2_OPENED",
        "marker_file_sha256": sha256_path(MARKER_PATH),
        "marker_payload_sha256": marker["marker_payload_sha256"],
    }


def _load_marker() -> dict[str, Any]:
    payload = json.loads(MARKER_PATH.read_text())
    if not _verify_self_hash(payload, "marker_payload_sha256"):
        raise O5TrainingIntegrityError("V2 marker self hash failed")
    lock = _load_preflight_lock()
    checks = {
        "lock_file": payload.get("preflight_lock_file_sha256")
        == sha256_path(PREFLIGHT_LOCK_PATH),
        "lock_payload": payload.get("preflight_lock_payload_sha256")
        == lock["preflight_lock_payload_sha256"],
        "execute_command": payload.get("execute_command")
        == lock["commands"]["execute"],
    }
    if not all(checks.values()):
        raise O5TrainingIntegrityError(f"V2 marker changed: {checks}")
    return payload


def _pair_from_lineage(
    state: SimState,
    lineage: np.ndarray,
    target: int,
) -> DesignatedPair:
    if lineage_integrity(lineage) != "live":
        raise O5TrainingIntegrityError("Live O5 pair lineage is invalid")
    a = np.argwhere((lineage & LINEAGE_A) != 0)
    b = np.argwhere((lineage & LINEAGE_B) != 0)
    if a.shape != (1, 2) or b.shape != (1, 2):
        raise O5TrainingIntegrityError(
            "O5 pair requires one A and one B descendant"
        )
    coordinates = tuple(
        sorted(
            (
                (int(a[0, 0]), int(a[0, 1])),
                (int(b[0, 0]), int(b[0, 1])),
            )
        )
    )
    (r0, c0), (r1, c1) = coordinates
    blockers = blocker_geometry(state.board, coordinates)
    same_row = r0 == r1
    same_column = c0 == c1
    safe_actions = pair_safe_merge_actions(
        state.board,
        coordinates,
        int(target),
        STARTER_TILE,
    )
    return DesignatedPair(
        target=int(target),
        coordinates=coordinates,
        manhattan=abs(r0 - r1) + abs(c0 - c1),
        chebyshev=max(abs(r0 - r1), abs(c0 - c1)),
        blocker_occupied=blockers.occupied,
        blocker_capacity=blockers.capacity,
        blocker_density=blockers.density,
        same_row=same_row,
        same_column=same_column,
        clear_line=bool(
            (same_row or same_column) and blockers.occupied == 0
        ),
        safe_merge_actions=tuple(int(action) for action in safe_actions),
    )


def _model_outputs(
    model: O4DesignatedPairNet,
    state: SimState,
    sim: ThreesSim,
    lineage: np.ndarray,
    pair: DesignatedPair,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, tuple[np.ndarray, np.ndarray]],
]:
    legal = tuple(int(action) for action in sim.legal_actions(state))
    features = {
        action: option_features(
            state,
            sim,
            starter_tile=STARTER_TILE,
            pair=pair,
            lineage=lineage,
            action=action,
        )
        for action in legal
    }
    tokens = torch.from_numpy(
        np.stack([features[action][0] for action in legal])
    )
    globals_array = torch.from_numpy(
        np.stack([features[action][1] for action in legal])
    )
    model.eval()
    with torch.no_grad():
        outputs = model(tokens, globals_array).cpu().numpy()
    if outputs.shape != (len(legal), OUTPUT_WIDTH):
        raise O5TrainingIntegrityError("O5 model output shape changed")
    if not np.isfinite(outputs).all():
        raise O5TrainingIntegrityError("O5 model output is nonfinite")
    return (
        {action: outputs[index] for index, action in enumerate(legal)},
        features,
    )


def generate_episode(
    *,
    root_row: Mapping[str, Any],
    task: Mapping[str, Any],
    model: O4DesignatedPairNet | None,
    collection_model_round: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    state, initial_pair = restore_train_root(root_row)
    state = copy.deepcopy(state)
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=int(task["deck_stream_id"]),
        slot_stream_id=int(task["slot_stream_id"]),
        starter_tile=STARTER_TILE,
    )
    policy_rng = np.random.default_rng(int(task["policy_stream_id"]))
    lineage = initial_lineage(initial_pair)
    round_number = int(task["round_index"])
    if not 1 <= round_number <= ROUNDS:
        raise O5TrainingIntegrityError("O5 task round is invalid")
    expected_model_round = 0 if round_number == 1 else round_number - 1
    if int(collection_model_round) != expected_model_round:
        raise O5TrainingIntegrityError(
            "O5 collection model is not the immediate predecessor"
        )
    if round_number == 1 and model is not None:
        raise O5TrainingIntegrityError("O5 R1 must be uniform-only")
    if round_number > 1 and model is None:
        raise O5TrainingIntegrityError(
            "O5 R2-R4 require the prior-round model"
        )
    epsilon = EPSILON_BY_ROUND[round_number - 1]

    decisions: list[dict[str, Any]] = []
    live_geometry_by_move: dict[int, np.ndarray] = {
        0: successor_geometry(
            state,
            sim,
            lineage=lineage,
            target=int(root_row["target"]),
        )
    }
    terminal_status = "censor"
    terminal_move = OPTION_HORIZON
    for option_move in range(OPTION_HORIZON):
        legal = tuple(int(action) for action in sim.legal_actions(state))
        if not legal:
            terminal_status = "failure"
            terminal_move = option_move
            break
        pair = _pair_from_lineage(
            state,
            lineage,
            int(root_row["target"]),
        )
        if round_number == 1:
            features_by_action = {
                action: option_features(
                    state,
                    sim,
                    starter_tile=STARTER_TILE,
                    pair=pair,
                    lineage=lineage,
                    action=action,
                )
                for action in legal
            }
            action = int(policy_rng.choice(np.asarray(legal)))
        else:
            assert model is not None
            outputs, features_by_action = _model_outputs(
                model,
                state,
                sim,
                lineage,
                pair,
            )
            if float(policy_rng.random()) < epsilon:
                action = int(policy_rng.choice(np.asarray(legal)))
            else:
                action = choose_option_action(
                    outputs,
                    remaining_horizon=OPTION_HORIZON - option_move,
                    safe_merge_actions=pair.safe_merge_actions,
                )
        tokens, globals_array = features_by_action[action]
        decisions.append(
            {
                "decision_move": option_move,
                "action": action,
                "tokens": tokens,
                "globals": globals_array,
            }
        )

        base = advance_lineage_base(state.board, lineage, action)
        next_state, info = sim.step(state, action)
        if not info.moved:
            raise O5TrainingIntegrityError(
                "Chosen legal O5 action did not move"
            )
        shifted = next_state.board.copy()
        if info.inserted_pos is not None:
            shifted[info.inserted_pos] = 0
        if not np.array_equal(base.board, shifted):
            raise O5TrainingIntegrityError(
                "O5 tagged afterstate diverged from simulator"
            )
        if tuple(base.eligible_slots) != tuple(info.eligible_positions):
            raise O5TrainingIntegrityError(
                "O5 tagged insertion slots diverged"
            )
        next_lineage = (
            base.lineage
            if info.inserted_pos is None
            else apply_spawn_to_lineage(base.lineage, info.inserted_pos)
        )
        status = transition_status(
            next_state,
            sim,
            starter_tile=STARTER_TILE,
            lineage=next_lineage,
            base_event=base.event,
        )
        completed_move = option_move + 1
        state = next_state
        lineage = next_lineage
        if status == "live":
            live_geometry_by_move[completed_move] = successor_geometry(
                state,
                sim,
                lineage=lineage,
                target=int(root_row["target"]),
            )
        else:
            terminal_status = status
            terminal_move = completed_move
            break
    else:
        terminal_status = "censor"
        terminal_move = OPTION_HORIZON

    if (
        terminal_status == "censor"
        and terminal_move == OPTION_HORIZON
        and OPTION_HORIZON not in live_geometry_by_move
    ):
        live_geometry_by_move[OPTION_HORIZON] = successor_geometry(
            state,
            sim,
            lineage=lineage,
            target=int(root_row["target"]),
        )

    event_targets: list[np.ndarray] = []
    event_masks: list[bool] = []
    geometry: list[np.ndarray] = []
    geometry_masks: list[np.ndarray] = []
    for decision in decisions:
        targets = build_decision_targets(
            decision_move=int(decision["decision_move"]),
            terminal_move=int(terminal_move),
            terminal_status=terminal_status,
            live_geometry_by_move=live_geometry_by_move,
        )
        one_hot = np.zeros(EVENT_WIDTH, dtype=np.float32)
        if targets.event_mask:
            if targets.event_class is None:
                raise O5TrainingIntegrityError(
                    "Masked O5 event lacks a class"
                )
            one_hot[int(targets.event_class)] = 1.0
        event_targets.append(one_hot)
        event_masks.append(bool(targets.event_mask))
        geometry.append(targets.geometry)
        geometry_masks.append(targets.geometry_mask)

    if not decisions:
        raise O5TrainingIntegrityError("O5 episode has no decision row")
    arrays = {
        "tokens": np.stack(
            [row["tokens"] for row in decisions]
        ).astype(np.float32),
        "globals": np.stack(
            [row["globals"] for row in decisions]
        ).astype(np.float32),
        "actions": np.asarray(
            [row["action"] for row in decisions],
            dtype=np.int8,
        ),
        "decision_moves": np.asarray(
            [row["decision_move"] for row in decisions],
            dtype=np.int8,
        ),
        "event_target": np.stack(event_targets).astype(np.float32),
        "event_mask": np.asarray(event_masks, dtype=np.bool_),
        "geometry": np.stack(geometry).astype(np.float32),
        "geometry_mask": np.stack(geometry_masks).astype(np.bool_),
    }
    domain_arrays = (
        arrays["tokens"],
        arrays["globals"],
        arrays["event_target"],
        arrays["geometry"],
    )
    if any(
        not np.isfinite(value).all()
        or np.any(value < 0.0)
        or np.any(value > 1.0)
        for value in domain_arrays
    ):
        raise O5TrainingIntegrityError(
            "O5 episode input/target escaped [0,1]"
        )
    metadata = {
        "version": f"{VERSION}_episode",
        "task_id": _task_id(task),
        "root_index": int(task["root_index"]),
        "round_index": round_number,
        "trajectory_index": int(task["trajectory_index"]),
        "root_cluster": str(root_row["root_cluster"]),
        "family": str(root_row["family"]),
        "target": int(root_row["target"]),
        "terminal_status": terminal_status,
        "terminal_move": int(terminal_move),
        "decision_rows": len(decisions),
        "collection_model_round": int(collection_model_round),
        "epsilon": float(epsilon),
        "stream_ids": {
            field: int(task[field]) for field in STREAM_FIELDS
        },
        "score_or_behavior_action_label_used": False,
    }
    return arrays, metadata


def _episode_path(task: Mapping[str, Any]) -> Path:
    return EPISODE_DIR / f"{_task_id(task)}.npz"


def _metadata_bytes(metadata: Mapping[str, Any]) -> tuple[bytes, str]:
    body = _payload_with_hash(metadata, "metadata_payload_sha256")
    serialized = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return serialized, body["metadata_payload_sha256"]


def _arrays_equal(
    observed: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
) -> bool:
    return (
        set(observed) == set(expected)
        and all(
            observed[name].dtype == expected[name].dtype
            and observed[name].shape == expected[name].shape
            and np.array_equal(observed[name], expected[name])
            for name in expected
        )
    )


def _write_episode_atomic(
    task: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> tuple[str, str]:
    path = _episode_path(task)
    if path.exists():
        observed, observed_metadata = _load_episode(task)
        if not _arrays_equal(observed, arrays):
            raise O5TrainingIntegrityError(
                "Existing O5 episode differs on regeneration"
            )
        for key, value in metadata.items():
            if observed_metadata.get(key) != value:
                raise O5TrainingIntegrityError(
                    "Existing O5 metadata differs on regeneration"
                )
        return (
            sha256_path(path),
            str(observed_metadata["metadata_payload_sha256"]),
        )
    metadata_json, metadata_sha = _metadata_bytes(metadata)
    payload = {
        **dict(arrays),
        "__metadata_json__": np.frombuffer(
            metadata_json,
            dtype=np.uint8,
        ).copy(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    observed, observed_metadata = _load_episode(task)
    if not _arrays_equal(observed, arrays):
        raise O5TrainingIntegrityError("Committed O5 episode changed")
    if observed_metadata["metadata_payload_sha256"] != metadata_sha:
        raise O5TrainingIntegrityError("Committed O5 metadata changed")
    return sha256_path(path), metadata_sha


def _load_episode(
    task: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = _episode_path(task)
    with np.load(path, allow_pickle=False) as loaded:
        names = set(loaded.files)
        if "__metadata_json__" not in names:
            raise O5TrainingIntegrityError("O5 episode lacks metadata")
        metadata_raw = loaded["__metadata_json__"].tobytes()
        arrays = {
            name: loaded[name]
            for name in loaded.files
            if name != "__metadata_json__"
        }
    metadata = json.loads(metadata_raw.decode("utf-8"))
    if not _verify_self_hash(metadata, "metadata_payload_sha256"):
        raise O5TrainingIntegrityError("O5 episode metadata hash failed")
    if metadata.get("task_id") != _task_id(task):
        raise O5TrainingIntegrityError("O5 episode task identity changed")
    expected_streams = {
        field: int(task[field]) for field in STREAM_FIELDS
    }
    if metadata.get("stream_ids") != expected_streams:
        raise O5TrainingIntegrityError("O5 episode streams changed")
    required = {
        "tokens",
        "globals",
        "actions",
        "decision_moves",
        "event_target",
        "event_mask",
        "geometry",
        "geometry_mask",
    }
    if set(arrays) != required:
        raise O5TrainingIntegrityError("O5 episode array schema changed")
    row_count = arrays["tokens"].shape[0]
    shapes = {
        "tokens": (row_count, 16, 37),
        "globals": (row_count, 35),
        "actions": (row_count,),
        "decision_moves": (row_count,),
        "event_target": (row_count, EVENT_WIDTH),
        "event_mask": (row_count,),
        "geometry": (
            row_count,
            len(CHECKPOINTS),
            GEOMETRY_WIDTH,
        ),
        "geometry_mask": (row_count, len(CHECKPOINTS)),
    }
    if any(arrays[name].shape != shape for name, shape in shapes.items()):
        raise O5TrainingIntegrityError("O5 episode shape changed")
    for name in ("tokens", "globals", "event_target", "geometry"):
        values = arrays[name]
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise O5TrainingIntegrityError(
                f"O5 episode domain failed: {name}"
            )
    return arrays, metadata


def _hash_nested(value: Any, digest: hashlib._Hash) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(b"tensor")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode("utf-8"))
            _hash_nested(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _hash_nested(item, digest)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(
            json.dumps(value, sort_keys=True).encode("utf-8")
        )
    else:
        raise O5TrainingIntegrityError(
            f"Unsupported state-hash value: {type(value).__name__}"
        )


def optimizer_state_sha256(
    optimizer: torch.optim.Optimizer,
) -> str:
    digest = hashlib.sha256()
    _hash_nested(optimizer.state_dict(), digest)
    return digest.hexdigest()


def optimizer_step_count(optimizer: torch.optim.Optimizer) -> int:
    steps = []
    for state in optimizer.state.values():
        raw = state.get("step", 0)
        steps.append(int(raw.item()) if isinstance(raw, torch.Tensor) else int(raw))
    return max(steps, default=0)


def _checkpoint_path(round_number: int) -> Path:
    if not 1 <= int(round_number) <= ROUNDS:
        raise O5TrainingIntegrityError("Checkpoint round must be 1..4")
    return CHECKPOINT_DIR / f"round_{int(round_number)}_provisional.pt"


def _save_checkpoint(
    path: Path,
    *,
    model: O4DesignatedPairNet,
    optimizer: torch.optim.Optimizer,
    round_number: int,
    config_sha256: str,
    predecessor_file_sha256: str | None,
    pre_fit_model_sha256: str,
    pre_fit_optimizer_sha256: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"O5 checkpoint exists: {path}")
    payload = {
        "version": f"{VERSION}_checkpoint",
        "round_number": int(round_number),
        "authoritative": False,
        "candidate": False,
        "collection_only_until_support_gate": True,
        "schema_sha256": schema_sha256(),
        "config_sha256": config_sha256,
        "predecessor_file_sha256": predecessor_file_sha256,
        "pre_fit_model_sha256": pre_fit_model_sha256,
        "pre_fit_optimizer_sha256": pre_fit_optimizer_sha256,
        "post_fit_model_sha256": model_state_sha256(model),
        "post_fit_optimizer_sha256": optimizer_state_sha256(optimizer),
        "optimizer_step_count": optimizer_step_count(optimizer),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_version": torch.__version__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    loaded_model, loaded_optimizer, metadata = _load_checkpoint(
        path,
        config_sha256=config_sha256,
        expected_round=round_number,
        expected_predecessor_sha256=predecessor_file_sha256,
    )
    if model_state_sha256(loaded_model) != payload["post_fit_model_sha256"]:
        raise O5TrainingIntegrityError("Checkpoint model reload changed")
    if (
        optimizer_state_sha256(loaded_optimizer)
        != payload["post_fit_optimizer_sha256"]
    ):
        raise O5TrainingIntegrityError(
            "Checkpoint optimizer reload changed"
        )
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "round_number": int(round_number),
        "post_fit_model_sha256": metadata["post_fit_model_sha256"],
        "post_fit_optimizer_sha256": metadata[
            "post_fit_optimizer_sha256"
        ],
        "optimizer_step_count": int(metadata["optimizer_step_count"]),
        "authoritative": False,
    }


def _load_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    expected_round: int,
    expected_predecessor_sha256: str | None,
) -> tuple[
    O4DesignatedPairNet,
    torch.optim.Optimizer,
    dict[str, Any],
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "version": payload.get("version") == f"{VERSION}_checkpoint",
        "round": int(payload.get("round_number", -1)) == int(expected_round),
        "not_authoritative": payload.get("authoritative") is False,
        "not_candidate": payload.get("candidate") is False,
        "schema": payload.get("schema_sha256") == schema_sha256(),
        "config": payload.get("config_sha256") == config_sha256,
        "predecessor": payload.get("predecessor_file_sha256")
        == expected_predecessor_sha256,
        "torch": payload.get("torch_version") == EXPECTED_TORCH_VERSION,
    }
    if not all(checks.values()):
        raise O5TrainingIntegrityError(
            f"O5 checkpoint identity failed: {checks}"
        )
    model, optimizer = initialize_training()
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if model_state_sha256(model) != payload["post_fit_model_sha256"]:
        raise O5TrainingIntegrityError("O5 checkpoint model hash failed")
    if (
        optimizer_state_sha256(optimizer)
        != payload["post_fit_optimizer_sha256"]
    ):
        raise O5TrainingIntegrityError(
            "O5 checkpoint optimizer hash failed"
        )
    if optimizer_step_count(optimizer) != int(
        payload["optimizer_step_count"]
    ):
        raise O5TrainingIntegrityError(
            "O5 checkpoint optimizer step count failed"
        )
    return model, optimizer, dict(payload)


def _buffer_arrays(
    tasks: Sequence[Mapping[str, Any]],
    roots: Sequence[Mapping[str, Any]],
    maximum_round: int,
) -> dict[str, np.ndarray]:
    family_root_counts = Counter(str(row["family"]) for row in roots)
    if tuple(sorted(family_root_counts)) != tuple(sorted(FAMILY_ORDER)):
        raise O5TrainingIntegrityError("O5 training family set changed")
    represented_families = len(family_root_counts)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    included = [
        task
        for task in tasks
        if int(task["round_index"]) <= int(maximum_round)
    ]
    for task in included:
        arrays, metadata = _load_episode(task)
        row_count = arrays["tokens"].shape[0]
        event_valid = int(np.count_nonzero(arrays["event_mask"]))
        event_weight = np.zeros(row_count, dtype=np.float32)
        if event_valid:
            value = 1.0 / (
                represented_families
                * family_root_counts[str(metadata["family"])]
                * TRAJECTORIES_PER_ROOT
                * event_valid
            )
            event_weight[arrays["event_mask"]] = value
        geometry_weight = np.zeros(
            (row_count, len(CHECKPOINTS)),
            dtype=np.float32,
        )
        for checkpoint_index in range(len(CHECKPOINTS)):
            valid = int(
                np.count_nonzero(
                    arrays["geometry_mask"][:, checkpoint_index]
                )
            )
            if valid:
                value = 1.0 / (
                    represented_families
                    * family_root_counts[str(metadata["family"])]
                    * TRAJECTORIES_PER_ROOT
                    * valid
                )
                geometry_weight[
                    arrays["geometry_mask"][:, checkpoint_index],
                    checkpoint_index,
                ] = value
        for name, value in arrays.items():
            chunks[name].append(value)
        chunks["event_weight"].append(event_weight)
        chunks["geometry_weight"].append(geometry_weight)
    if not chunks:
        raise O5TrainingIntegrityError("O5 cumulative buffer is empty")
    result = {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }
    for name in ("tokens", "globals", "event_target", "geometry"):
        values = result[name]
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise O5TrainingIntegrityError(
                f"O5 buffer domain failed: {name}"
            )
    return result


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    denominator = weights.sum()
    if float(denominator) <= 0.0:
        return values.sum() * 0.0
    return (values * weights).sum() / denominator


def fit_cumulative_round(
    *,
    model: O4DesignatedPairNet,
    optimizer: torch.optim.Optimizer,
    arrays: Mapping[str, np.ndarray],
    round_number: int,
    epoch_complete: Callable[[int, float], None] | None = None,
) -> None:
    row_count = int(arrays["tokens"].shape[0])
    continuous = torch.tensor([0, 1, 2, 5, 6], dtype=torch.long)
    binary = torch.tensor([3, 4, 7], dtype=torch.long)
    for epoch_index in range(EPOCHS_PER_ROUND):
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(
            TRAIN_SEED + 100 * int(round_number) + epoch_index
        )
        permutation = torch.randperm(row_count, generator=generator)
        model.train()
        for start in range(0, row_count, BATCH_SIZE):
            index = permutation[start : start + BATCH_SIZE].numpy()
            tokens = torch.from_numpy(arrays["tokens"][index])
            globals_array = torch.from_numpy(arrays["globals"][index])
            output = model(tokens, globals_array)
            total = output.sum() * 0.0

            event_mask = torch.from_numpy(arrays["event_mask"][index])
            if bool(event_mask.any()):
                target = torch.from_numpy(
                    arrays["event_target"][index]
                )
                event_losses = -(
                    target * F.log_softmax(output[:, :EVENT_WIDTH], dim=1)
                ).sum(dim=1)
                event_weights = torch.from_numpy(
                    arrays["event_weight"][index]
                )
                total = total + _weighted_mean(
                    event_losses[event_mask],
                    event_weights[event_mask],
                )

            geometry_output = output[:, EVENT_WIDTH:].reshape(
                -1,
                len(CHECKPOINTS),
                GEOMETRY_WIDTH,
            )
            geometry_target = torch.from_numpy(arrays["geometry"][index])
            geometry_mask = torch.from_numpy(
                arrays["geometry_mask"][index]
            )
            geometry_weights = torch.from_numpy(
                arrays["geometry_weight"][index]
            )
            for checkpoint_index in range(len(CHECKPOINTS)):
                mask = geometry_mask[:, checkpoint_index]
                if not bool(mask.any()):
                    continue
                prediction = torch.sigmoid(
                    geometry_output[mask, checkpoint_index]
                )
                target = geometry_target[mask, checkpoint_index]
                smooth = F.smooth_l1_loss(
                    prediction[:, continuous],
                    target[:, continuous],
                    reduction="none",
                ).mean(dim=1)
                bce = F.binary_cross_entropy(
                    prediction[:, binary],
                    target[:, binary],
                    reduction="none",
                ).mean(dim=1)
                per_row = (smooth * 5.0 + bce * 3.0) / 8.0
                weights = geometry_weights[mask, checkpoint_index]
                total = total + (
                    GEOMETRY_LOSS_COEFFICIENT
                    * _weighted_mean(per_row, weights)
                )
            if not torch.isfinite(total):
                raise O5TrainingIntegrityError(
                    "O5 training loss is nonfinite"
                )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )
            optimizer.step()
        elapsed = time.perf_counter() - started
        if epoch_complete is not None:
            epoch_complete(epoch_index, elapsed)
    if any(
        not torch.isfinite(parameter).all()
        for parameter in model.parameters()
    ):
        raise O5TrainingIntegrityError(
            "O5 checkpoint has nonfinite parameters"
        )


def _runtime_state() -> dict[str, Any]:
    if not RUNTIME_PATH.exists():
        return {
            "version": f"{VERSION}_runtime",
            "active_seconds": 0.0,
            "completed_tasks": 0,
            "completed_rounds": 0,
            "fit_epochs_charged": 0,
        }
    return json.loads(RUNTIME_PATH.read_text())


def _write_runtime(payload: Mapping[str, Any]) -> None:
    temporary = RUNTIME_PATH.with_name(
        f".{RUNTIME_PATH.name}.tmp.{os.getpid()}"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, RUNTIME_PATH)


def _support_report(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata_rows = [_load_episode(task)[1] for task in tasks]
    if len(metadata_rows) != EPISODES:
        raise O5TrainingIntegrityError(
            "Support opened before all O5 episodes"
        )
    statuses = Counter(str(row["terminal_status"]) for row in metadata_rows)
    successes = [
        row for row in metadata_rows if row["terminal_status"] == "success"
    ]
    success_by_target = Counter(int(row["target"]) for row in successes)
    success_by_family = Counter(str(row["family"]) for row in successes)
    time_bins = Counter(
        "1_10"
        if int(row["terminal_move"]) <= 10
        else "11_20"
        if int(row["terminal_move"]) <= 20
        else "21_40"
        for row in successes
    )
    true_h40_censors = sum(
        row["terminal_status"] == "censor"
        and int(row["terminal_move"]) == OPTION_HORIZON
        for row in metadata_rows
    )
    finite = True
    for task in tasks:
        arrays, _ = _load_episode(task)
        finite = finite and all(
            not np.issubdtype(value.dtype, np.floating)
            or bool(np.isfinite(value).all())
            for value in arrays.values()
        )
    checks = {
        "successes_at_least_40": len(successes) >= 40,
        "six_successes_each_target": all(
            success_by_target[target] >= 6 for target in TRAIN_TARGETS
        ),
        "three_successes_each_family": all(
            success_by_family[family] >= 3 for family in FAMILY_ORDER
        ),
        "failures_at_least_40": statuses["failure"] >= 40,
        "true_h40_censors_at_least_40": true_h40_censors >= 40,
        "finite_arrays": finite,
        "two_nonempty_success_bins": sum(
            value > 0 for value in time_bins.values()
        )
        >= 2,
    }
    return {
        "version": f"{VERSION}_label_support",
        "episodes": len(metadata_rows),
        "status_counts": dict(statuses),
        "true_h40_censors": true_h40_censors,
        "success_by_target": {
            str(target): success_by_target[target]
            for target in TRAIN_TARGETS
        },
        "success_by_family": {
            family: success_by_family[family] for family in FAMILY_ORDER
        },
        "success_time_bins": {
            name: time_bins[name]
            for name in ("1_10", "11_20", "21_40")
        },
        "checks": checks,
        "passes": all(checks.values()),
        "opened_after_all_episodes_and_fits": True,
        "development_content_opened": False,
        "untouched_content_opened": False,
        "score_or_human_action_labels_used": False,
    }


def seal_checkpoint_disposition(
    *,
    support_passes: bool,
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(checkpoints) != ROUNDS:
        raise O5TrainingIntegrityError(
            "Checkpoint disposition requires four rounds"
        )
    expected_rounds = list(range(1, ROUNDS + 1))
    if [int(row["round_number"]) for row in checkpoints] != expected_rounds:
        raise O5TrainingIntegrityError(
            "Checkpoint disposition round order changed"
        )
    if support_passes:
        return _write_immutable(
            CHECKPOINT_AUTHORITY_PATH,
            {
                "version": f"{VERSION}_checkpoint_authority",
                "support_passes": True,
                "candidate_round": 4,
                "candidate": dict(checkpoints[-1]),
                "provisional_non_candidates": [
                    dict(row) for row in checkpoints[:-1]
                ],
                "development_or_mechanism_opened": False,
            },
            "authority_payload_sha256",
        )
    return _write_immutable(
        CHECKPOINT_QUARANTINE_PATH,
        {
            "version": f"{VERSION}_checkpoint_quarantine",
            "support_passes": False,
            "candidate_round": None,
            "checkpoints": [
                {
                    **dict(row),
                    "authoritative": False,
                    "usable_downstream": False,
                    "quarantined": True,
                }
                for row in checkpoints
            ],
            "development_or_mechanism_opened": False,
        },
        "quarantine_payload_sha256",
    )


def _validate_fit_ledger(
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    states = {
        round_number: {
            "opened": False,
            "completed": False,
            "resume_count": 0,
        }
        for round_number in range(1, ROUNDS + 1)
    }
    source = _read_jsonl(FIT_ATTEMPT_PATH) if rows is None else rows
    for index, row in enumerate(source):
        round_number = int(row.get("round_number", -1))
        if round_number not in states:
            raise O5TrainingIntegrityError(
                f"Unknown fit round at row {index}"
            )
        state = states[round_number]
        status = row.get("status")
        if status == "opened":
            if state["opened"] or state["completed"]:
                raise O5TrainingIntegrityError("Duplicate fit open")
            state["opened"] = True
        elif status == "resumed_from_predecessor":
            if not state["opened"] or state["completed"]:
                raise O5TrainingIntegrityError("Invalid fit resume")
            state["resume_count"] += 1
        elif status == "completed":
            if not state["opened"] or state["completed"]:
                raise O5TrainingIntegrityError("Duplicate/unopened fit close")
            if not isinstance(row.get("checkpoint_sha256"), str):
                raise O5TrainingIntegrityError(
                    "Fit close lacks checkpoint SHA"
                )
            state["completed"] = True
        else:
            raise O5TrainingIntegrityError(
                f"Unknown fit status at row {index}: {status}"
            )
    return states


def _load_execution_manifests() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    lock = _load_preflight_lock()
    roots_payload = json.loads(ROOT_MANIFEST_PATH.read_text())
    tasks_payload = json.loads(TASK_MANIFEST_PATH.read_text())
    roots = [dict(row) for row in roots_payload["train_rows"]]
    tasks = [dict(row) for row in tasks_payload["rows"]]
    if len(roots) != TRAIN_ROOTS or len(tasks) != EPISODES:
        raise O5TrainingIntegrityError(
            "O5 execution manifest count changed"
        )
    if canonical_json_hash(tasks) != tasks_payload["task_manifest_sha256"]:
        raise O5TrainingIntegrityError("O5 task manifest hash changed")
    return roots, tasks, str(lock["config"]["payload_sha256"])


def load_round_state(
    round_number: int,
    *,
    config_sha256: str,
) -> tuple[
    O4DesignatedPairNet,
    torch.optim.Optimizer,
    O4DesignatedPairNet | None,
    int,
    dict[str, Any] | None,
]:
    if not 1 <= int(round_number) <= ROUNDS:
        raise O5TrainingIntegrityError("O5 round must be 1..4")
    if int(round_number) == 1:
        model, optimizer = initialize_training()
        return model, optimizer, None, 0, None
    predecessor_round = int(round_number) - 1
    predecessor_path = _checkpoint_path(predecessor_round)
    model, optimizer, payload = _load_checkpoint(
        predecessor_path,
        config_sha256=config_sha256,
        expected_round=predecessor_round,
        expected_predecessor_sha256=(
            None
            if predecessor_round == 1
            else sha256_path(_checkpoint_path(predecessor_round - 1))
        ),
    )
    if (
        model_state_sha256(model) != payload["post_fit_model_sha256"]
        or optimizer_state_sha256(optimizer)
        != payload["post_fit_optimizer_sha256"]
    ):
        raise O5TrainingIntegrityError(
            "O5 predecessor load changed state"
        )
    return model, optimizer, model, predecessor_round, payload


def _reconcile_runtime(
    runtime: dict[str, Any],
    *,
    completed_tasks: int,
    completed_rounds: int,
) -> None:
    if int(runtime.get("completed_tasks", 0)) > int(completed_tasks):
        raise O5TrainingIntegrityError(
            "Runtime task count exceeds attempt closes"
        )
    if int(runtime.get("completed_rounds", 0)) > int(completed_rounds):
        raise O5TrainingIntegrityError(
            "Runtime round count exceeds checkpoints"
        )
    runtime["completed_tasks"] = int(completed_tasks)
    runtime["completed_rounds"] = int(completed_rounds)
    _write_runtime(runtime)


def _collect_round(
    *,
    round_number: int,
    model: O4DesignatedPairNet | None,
    collection_model_round: int,
    roots: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    runtime: dict[str, Any],
) -> int:
    round_tasks = [
        task
        for task in tasks
        if int(task["round_index"]) == int(round_number)
    ]
    expected = TRAIN_ROOTS * TRAJECTORIES_BY_ROUND[round_number - 1]
    if len(round_tasks) != expected:
        raise O5TrainingIntegrityError(
            f"O5 round {round_number} task count changed"
        )
    attempts = _read_jsonl(ATTEMPT_PATH)
    states = validate_attempt_ledger(tasks, attempts)
    completed_count = sum(
        state["completed"] for state in states.values()
    )
    _reconcile_runtime(
        runtime,
        completed_tasks=completed_count,
        completed_rounds=int(runtime.get("completed_rounds", 0)),
    )
    for task in round_tasks:
        task_id = _task_id(task)
        artifact_path = _episode_path(task)
        state = states[task_id]
        if state["completed"]:
            if not artifact_path.exists():
                raise O5TrainingIntegrityError(
                    "Closed O5 task lacks its episode"
                )
            _, metadata = _load_episode(task)
            if int(metadata["collection_model_round"]) != (
                0 if round_number == 1 else round_number - 1
            ):
                raise O5TrainingIntegrityError(
                    "Closed O5 task used the wrong collection model"
                )
            continue
        if artifact_path.exists() and not state["opened"]:
            raise O5TrainingIntegrityError(
                "O5 episode exists before attempt open"
            )
        if artifact_path.exists():
            _, metadata = _load_episode(task)
            _append_jsonl(
                ATTEMPT_PATH,
                {
                    "task_id": task_id,
                    "status": "completed",
                    "artifact_sha256": sha256_path(artifact_path),
                    "metadata_payload_sha256": metadata[
                        "metadata_payload_sha256"
                    ],
                },
            )
            state["completed"] = True
            completed_count += 1
            runtime["completed_tasks"] = completed_count
            _write_runtime(runtime)
            continue
        if not state["opened"]:
            _append_jsonl(
                ATTEMPT_PATH,
                {
                    "task_id": task_id,
                    "status": "opened",
                    "root_index": int(task["root_index"]),
                    "round_index": int(task["round_index"]),
                    "trajectory_index": int(task["trajectory_index"]),
                    "stream_ids": {
                        field: int(task[field]) for field in STREAM_FIELDS
                    },
                },
            )
            state["opened"] = True
        else:
            _append_jsonl(
                ATTEMPT_PATH,
                {
                    "task_id": task_id,
                    "status": "resumed_same_stream",
                },
            )
            state["resume_count"] += 1
        started = time.perf_counter()
        arrays, metadata = generate_episode(
            root_row=roots[int(task["root_index"])],
            task=task,
            model=model,
            collection_model_round=collection_model_round,
        )
        runtime["active_seconds"] = (
            float(runtime.get("active_seconds", 0.0))
            + time.perf_counter()
            - started
        )
        # Runtime is committed before the label artifact.
        _write_runtime(runtime)
        artifact_sha, metadata_sha = _write_episode_atomic(
            task,
            arrays,
            metadata,
        )
        _append_jsonl(
            ATTEMPT_PATH,
            {
                "task_id": task_id,
                "status": "completed",
                "artifact_sha256": artifact_sha,
                "metadata_payload_sha256": metadata_sha,
            },
        )
        state["completed"] = True
        completed_count += 1
        runtime["completed_tasks"] = completed_count
        _write_runtime(runtime)
        if float(runtime["active_seconds"]) > MAX_RUNTIME_SECONDS:
            raise O5TrainingOperationalHold(
                "O5 active runtime exceeded 18 hours"
            )
        if completed_count % 24 == 0:
            _require_operational("O5 episode boundary")
    return completed_count


def _round_checkpoint_identity(
    round_number: int,
    *,
    config_sha256: str,
) -> dict[str, Any]:
    path = _checkpoint_path(round_number)
    predecessor = (
        None
        if round_number == 1
        else sha256_path(_checkpoint_path(round_number - 1))
    )
    _, _, payload = _load_checkpoint(
        path,
        config_sha256=config_sha256,
        expected_round=round_number,
        expected_predecessor_sha256=predecessor,
    )
    return {
        "path": str(path),
        "file_sha256": sha256_path(path),
        "round_number": round_number,
        "post_fit_model_sha256": payload["post_fit_model_sha256"],
        "post_fit_optimizer_sha256": payload[
            "post_fit_optimizer_sha256"
        ],
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "authoritative": False,
    }


def _fit_round_and_checkpoint(
    *,
    round_number: int,
    model: O4DesignatedPairNet,
    optimizer: torch.optim.Optimizer,
    roots: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    config_sha256: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    path = _checkpoint_path(round_number)
    predecessor_path = (
        None if round_number == 1 else _checkpoint_path(round_number - 1)
    )
    predecessor_sha = (
        None if predecessor_path is None else sha256_path(predecessor_path)
    )
    fit_states = _validate_fit_ledger()
    fit_state = fit_states[round_number]
    if path.exists():
        identity = _round_checkpoint_identity(
            round_number,
            config_sha256=config_sha256,
        )
        if not fit_state["completed"]:
            if not fit_state["opened"]:
                raise O5TrainingIntegrityError(
                    "O5 checkpoint exists before fit open"
                )
            _append_jsonl(
                FIT_ATTEMPT_PATH,
                {
                    "round_number": round_number,
                    "status": "completed",
                    "checkpoint_sha256": identity["file_sha256"],
                },
            )
        runtime["completed_rounds"] = max(
            int(runtime.get("completed_rounds", 0)),
            round_number,
        )
        _write_runtime(runtime)
        return identity
    if not fit_state["opened"]:
        _append_jsonl(
            FIT_ATTEMPT_PATH,
            {
                "round_number": round_number,
                "status": "opened",
                "predecessor_file_sha256": predecessor_sha,
                "pre_fit_model_sha256": model_state_sha256(model),
                "pre_fit_optimizer_sha256":
                    optimizer_state_sha256(optimizer),
            },
        )
    else:
        _append_jsonl(
            FIT_ATTEMPT_PATH,
            {
                "round_number": round_number,
                "status": "resumed_from_predecessor",
                "predecessor_file_sha256": predecessor_sha,
            },
        )
    pre_model_sha = model_state_sha256(model)
    pre_optimizer_sha = optimizer_state_sha256(optimizer)
    if round_number > 1:
        _, _, predecessor_payload = _load_checkpoint(
            predecessor_path,
            config_sha256=config_sha256,
            expected_round=round_number - 1,
            expected_predecessor_sha256=(
                None
                if round_number == 2
                else sha256_path(_checkpoint_path(round_number - 2))
            ),
        )
        if (
            pre_model_sha != predecessor_payload["post_fit_model_sha256"]
            or pre_optimizer_sha
            != predecessor_payload["post_fit_optimizer_sha256"]
        ):
            raise O5TrainingIntegrityError(
                "O5 model/optimizer continuity failed"
            )
    arrays = _buffer_arrays(tasks, roots, round_number)

    def charge_epoch(_epoch: int, elapsed: float) -> None:
        runtime["active_seconds"] = (
            float(runtime.get("active_seconds", 0.0)) + float(elapsed)
        )
        runtime["fit_epochs_charged"] = (
            int(runtime.get("fit_epochs_charged", 0)) + 1
        )
        _write_runtime(runtime)
        if float(runtime["active_seconds"]) > MAX_RUNTIME_SECONDS:
            raise O5TrainingOperationalHold(
                "O5 active runtime exceeded during fit"
            )

    fit_cumulative_round(
        model=model,
        optimizer=optimizer,
        arrays=arrays,
        round_number=round_number,
        epoch_complete=charge_epoch,
    )
    identity = _save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        round_number=round_number,
        config_sha256=config_sha256,
        predecessor_file_sha256=predecessor_sha,
        pre_fit_model_sha256=pre_model_sha,
        pre_fit_optimizer_sha256=pre_optimizer_sha,
    )
    _append_jsonl(
        FIT_ATTEMPT_PATH,
        {
            "round_number": round_number,
            "status": "completed",
            "checkpoint_sha256": identity["file_sha256"],
        },
    )
    runtime["completed_rounds"] = round_number
    _write_runtime(runtime)
    return identity


def _load_or_write_support(
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    computed = _support_report(tasks)
    if SUPPORT_REPORT_PATH.exists():
        observed = json.loads(SUPPORT_REPORT_PATH.read_text())
        if not _verify_self_hash(observed, "support_payload_sha256"):
            raise O5TrainingIntegrityError("O5 support self hash failed")
        body = dict(observed)
        body.pop("support_payload_sha256")
        if body != computed:
            raise O5TrainingIntegrityError("O5 support report changed")
        return computed, observed
    artifact = _write_immutable(
        SUPPORT_REPORT_PATH,
        computed,
        "support_payload_sha256",
    )
    return computed, artifact


def _terminal_checkpoint_rows(
    *,
    config_sha256: str,
) -> list[dict[str, Any]]:
    return [
        _round_checkpoint_identity(
            round_number,
            config_sha256=config_sha256,
        )
        for round_number in range(1, ROUNDS + 1)
    ]


def _seal_terminal(
    *,
    marker: Mapping[str, Any],
    decision: str,
    support: Mapping[str, Any] | None,
    support_artifact: Mapping[str, Any] | None,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    disposition: Mapping[str, Any] | None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    hold = decision == "HOLD_O5_TRAINING_DATA_SUPPORT"
    kill = decision == "KILL_O5_TRAINING_INTEGRITY"
    ready = decision == "READY_O5_TRAINED_CHECKPOINT"
    payload: dict[str, Any] = {
        "version": f"{VERSION}_result",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "continue": ready,
        "hold": hold,
        "kill": kill,
        "promote": False,
        "marker_file_sha256": sha256_path(MARKER_PATH),
        "marker_payload_sha256": marker["marker_payload_sha256"],
        "episodes_completed": sum(
            state["completed"]
            for state in validate_attempt_ledger(learning_rows()).values()
        ),
        "checkpoint_rows": [dict(row) for row in checkpoint_rows],
        "runtime": _runtime_state(),
        "zero_downstream_work": {
            "development_content_opened": False,
            "untouched_content_opened": False,
            "mechanism_outcomes": 0,
            "normal_start_games": 0,
            "incumbent_changes": 0,
            "dashboard_changes": 0,
            "promotion": False,
        },
    }
    if support is not None and support_artifact is not None:
        payload["support"] = {
            "path": str(SUPPORT_REPORT_PATH),
            "file_sha256": sha256_path(SUPPORT_REPORT_PATH),
            "payload_sha256": support_artifact[
                "support_payload_sha256"
            ],
            "passes": bool(support["passes"]),
        }
    if disposition is not None:
        path = (
            CHECKPOINT_AUTHORITY_PATH
            if ready
            else CHECKPOINT_QUARANTINE_PATH
        )
        field = (
            "authority_payload_sha256"
            if ready
            else "quarantine_payload_sha256"
        )
        payload["checkpoint_disposition"] = {
            "path": str(path),
            "file_sha256": sha256_path(path),
            "payload_sha256": disposition[field],
        }
    if error is not None:
        payload.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "partial_work_preserved": True,
            }
        )
    result = _write_immutable(
        RESULT_PATH,
        payload,
        "result_payload_sha256",
    )
    return {
        "decision": decision,
        "result_file_sha256": sha256_path(RESULT_PATH),
        "result_payload_sha256": result["result_payload_sha256"],
    }


def execute() -> dict[str, Any]:
    marker = _load_marker()
    if RESULT_PATH.exists():
        raise FileExistsError("O5 V2 terminal result exists")
    checkpoint_rows: list[dict[str, Any]] = []
    try:
        _require_operational("O5 V2 execution start")
        roots, tasks, config_sha256 = _load_execution_manifests()
        runtime = _runtime_state()
        attempt_states = validate_attempt_ledger(tasks)
        completed_tasks = sum(
            state["completed"] for state in attempt_states.values()
        )
        completed_rounds = sum(
            _checkpoint_path(round_number).exists()
            for round_number in range(1, ROUNDS + 1)
        )
        _reconcile_runtime(
            runtime,
            completed_tasks=completed_tasks,
            completed_rounds=completed_rounds,
        )

        for round_number in range(1, ROUNDS + 1):
            predecessor_sha = (
                None
                if round_number == 1
                else sha256_path(_checkpoint_path(round_number - 1))
            )
            (
                model,
                optimizer,
                collection_model,
                collection_model_round,
                _prior_payload,
            ) = load_round_state(
                round_number,
                config_sha256=config_sha256,
            )

            round_tasks = [
                task
                for task in tasks
                if int(task["round_index"]) == round_number
            ]
            current_checkpoint = _checkpoint_path(round_number)
            if current_checkpoint.exists():
                if any(
                    not validate_attempt_ledger(tasks)[
                        _task_id(task)
                    ]["completed"]
                    for task in round_tasks
                ):
                    raise O5TrainingIntegrityError(
                        "O5 checkpoint precedes round task completion"
                    )
                checkpoint_rows.append(
                    _round_checkpoint_identity(
                        round_number,
                        config_sha256=config_sha256,
                    )
                )
                continue

            _collect_round(
                round_number=round_number,
                model=collection_model,
                collection_model_round=collection_model_round,
                roots=roots,
                tasks=tasks,
                runtime=runtime,
            )
            if round_number > 1:
                if predecessor_sha != sha256_path(
                    _checkpoint_path(round_number - 1)
                ):
                    raise O5TrainingIntegrityError(
                        "O5 predecessor changed during collection"
                    )
            checkpoint = _fit_round_and_checkpoint(
                round_number=round_number,
                model=model,
                optimizer=optimizer,
                roots=roots,
                tasks=tasks,
                config_sha256=config_sha256,
                runtime=runtime,
            )
            checkpoint_rows.append(checkpoint)
            _require_operational(f"O5 round {round_number} boundary")
            print(
                f"phase=round_{round_number}_complete "
                f"episodes={runtime['completed_tasks']}/{EPISODES}",
                flush=True,
            )

        states = validate_attempt_ledger(tasks)
        if sum(state["completed"] for state in states.values()) != EPISODES:
            raise O5TrainingIntegrityError(
                "O5 support boundary lacks all episodes"
            )
        if len(checkpoint_rows) != ROUNDS:
            checkpoint_rows = _terminal_checkpoint_rows(
                config_sha256=config_sha256
            )
        support, support_artifact = _load_or_write_support(tasks)
        disposition = seal_checkpoint_disposition(
            support_passes=bool(support["passes"]),
            checkpoints=checkpoint_rows,
        )
        _require_operational("O5 V2 terminal boundary")
        decision = (
            "READY_O5_TRAINED_CHECKPOINT"
            if support["passes"]
            else "HOLD_O5_TRAINING_DATA_SUPPORT"
        )
        return _seal_terminal(
            marker=marker,
            decision=decision,
            support=support,
            support_artifact=support_artifact,
            checkpoint_rows=checkpoint_rows,
            disposition=disposition,
        )
    except O5TrainingOperationalHold as error:
        return _seal_terminal(
            marker=marker,
            decision="HOLD_O5_TRAINING_DATA_SUPPORT",
            support=None,
            support_artifact=None,
            checkpoint_rows=checkpoint_rows,
            disposition=None,
            error=error,
        )
    except Exception as error:
        return _seal_terminal(
            marker=marker,
            decision="KILL_O5_TRAINING_INTEGRITY",
            support=None,
            support_artifact=None,
            checkpoint_rows=checkpoint_rows,
            disposition=None,
            error=error,
        )


def write_test_evidence(
    *,
    focused_tests_passed: int,
    regression_tests_passed: int,
    recorded_commands: Sequence[str],
) -> dict[str, Any]:
    payload = {
        "version": f"{VERSION}_test_evidence",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
        "focused_tests_passed": int(focused_tests_passed),
        "regression_tests_passed": int(regression_tests_passed),
        "recorded_commands": list(recorded_commands),
        "passes": (
            int(focused_tests_passed) > 0
            and int(regression_tests_passed) > 0
        ),
        "streams_consumed": 0,
        "labels": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "policy_outcomes": 0,
    }
    artifact = _write_immutable(
        TEST_EVIDENCE_PATH,
        payload,
        "test_evidence_payload_sha256",
    )
    return {
        "path": str(TEST_EVIDENCE_PATH),
        "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "payload_sha256": artifact["test_evidence_payload_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
    )
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--focused", type=int, required=True)
    evidence.add_argument("--regressions", type=int, required=True)
    evidence.add_argument(
        "--recorded-command",
        dest="recorded_commands",
        action="append",
        default=[],
    )
    for command in ("prepare", "open", "execute"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--out-dir",
            type=Path,
            default=OUTPUT_DIR,
        )
    return parser


def dispatch(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.subcommand == "write-test-evidence":
        return write_test_evidence(
            focused_tests_passed=args.focused,
            regression_tests_passed=args.regressions,
            recorded_commands=args.recorded_commands,
        )
    if args.out_dir.resolve() != OUTPUT_DIR.resolve():
        raise O5TrainingIntegrityError("V2 command out-dir changed")
    if args.subcommand == "prepare":
        return prepare(args.out_dir)
    if args.subcommand == "open":
        return open_execution()
    if args.subcommand == "execute":
        return execute()
    raise O5TrainingIntegrityError(
        f"Unknown V2 command: {args.subcommand}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(dispatch(argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

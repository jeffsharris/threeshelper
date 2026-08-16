"""Marker-bound O3 closed-loop option training and label-support gate."""

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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from threes_rl import g1r_acquire as history
from threes_rl import o3_event_acquire as acquisition
from threes_rl import o3_event_acquire_recovery as recovery
from threes_rl import o3_p0_preflight as p0
from threes_rl import o3_selected_integrity_reseal_v3 as integrity_v3
from threes_rl.o1_geometry_option import (
    air_safe,
    anchor_safe,
    pair_safe_merge_actions,
)
from threes_rl.o3_designated_pair_option import (
    CHECKPOINTS,
    EVENT_WIDTH,
    GEOMETRY_WIDTH,
    LINEAGE_A,
    LINEAGE_B,
    OPTION_HORIZON,
    OUTPUT_WIDTH,
    TRAIN_TARGETS,
    DesignatedPair,
    O3DesignatedPairNet,
    advance_lineage_base,
    apply_spawn_to_lineage,
    build_decision_targets,
    canonical_json_hash,
    choose_option_action,
    initial_lineage,
    lineage_integrity,
    option_features,
    pair_blocker_count,
    root_option_eligible,
    schema_sha256,
    select_designated_pair,
    transition_status,
)
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "o3_option_training_v1"
ROOT = Path("threes_rl/runs")
CHARTER_PATH = Path("threes_rl/O3_OPTION_TRAINING_EXECUTION_CHARTER.md")
RUNNER_PATH = Path("threes_rl/o3_option_training.py")
TEST_PATH = Path("tests/test_rl_o3_option_training.py")
TEST_EVIDENCE_PATH = (
    ROOT / "forensics/o3_option_training_test_evidence.json"
)
OUTPUT_DIR = ROOT / "forensics/o3_option_training_v1"
CONFIG_PATH = OUTPUT_DIR / "training_config.json"
ROOT_MANIFEST_PATH = OUTPUT_DIR / "selected_root_manifest.json"
SOURCE_AUDIT_PATH = OUTPUT_DIR / "source_audit.json"
TASK_MANIFEST_PATH = OUTPUT_DIR / "learning_task_manifest.json"
COLLISION_PATH = OUTPUT_DIR / "learning_collision_audit.json"
PREFLIGHT_LOCK_PATH = OUTPUT_DIR / "preflight_lock.json"
PREFLIGHT_RESULT_PATH = OUTPUT_DIR / "preflight_result.json"
MARKER_PATH = OUTPUT_DIR / "O3_OPTION_TRAINING_OPENED.json"
ATTEMPT_PATH = OUTPUT_DIR / "attempts.jsonl"
RUNTIME_PATH = OUTPUT_DIR / "runtime_state.json"
EPISODE_DIR = OUTPUT_DIR / "episodes"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
SUPPORT_REPORT_PATH = OUTPUT_DIR / "label_support_report.json"
RESULT_PATH = OUTPUT_DIR / "training_result.json"

COURSE_CHARTER_PATH = Path(
    "threes_rl/O3_EVENT_CONDITIONED_DESIGNATED_PAIR_CHARTER.md"
)
V3_ENVELOPE_PATH = integrity_v3.ENVELOPE_PATH
SELECTED_PATH = recovery.SELECTED_PATH
P0_STREAM_PATH = p0.OUTPUT_DIR / "O3_P0_STREAM_MANIFEST.json"
P0_RESULT_PATH = p0.OUTPUT_DIR / "O3_P0_RESULT.json"

EXPECTED_BINDINGS = {
    "course_charter": (
        COURSE_CHARTER_PATH,
        "26a117cd5b14a32e79e4a63c63b0fb707f34135193d587880c157ef8c11f4441",
    ),
    "v3_envelope": (
        V3_ENVELOPE_PATH,
        "5bb80bc02597ea934c02f8ebd07eaf0158623232f88ea0408532cdc0039e6696",
    ),
    "selected_roots": (
        SELECTED_PATH,
        "9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049",
    ),
    "p0_stream_manifest": (
        P0_STREAM_PATH,
        "94e7b0dfe83e568b4e9686dd3ee44cc70739c0312349fe36a05bb6df80c77225",
    ),
    "p0_result": (
        P0_RESULT_PATH,
        "9ced80be3e2a784372f50fd2a99b0b41bdcc98920820796daa03a8db1640ced5",
    ),
    "o3_schema": (
        Path("threes_rl/o3_designated_pair_option.py"),
        "659475fe596a9e96aa56e3fc4bbaf57bbfdfbefa5a569b1c1e24ce8f345064fd",
    ),
    "simulator": (
        Path("threes_rl/sim.py"),
        "67e7a245c05e59367402095ad018122fb4cb1ef08664bf28bf4bc03a02a73072",
    ),
    "replay_restore": (
        Path("threes_rl/train_td.py"),
        "0ef18c38c09516a11fddc5b2cd742aa536c21615d5ce2477167bed8553b13f7a",
    ),
    "acquisition_runner": (
        Path("threes_rl/o3_event_acquire.py"),
        "842fee2b41526d6c37770b7deee09500354e9140731753da905c1900e974bd5b",
    ),
    "recovery_runner": (
        Path("threes_rl/o3_event_acquire_recovery.py"),
        "b67be9537e7728855d63006f12503038d3414fc600a9adc08809869ab8e64525",
    ),
}
EXPECTED_V3_PAYLOAD = (
    "622ebf6361527be7283fd51c7a7acff99aa8125b06c76dbc4ee8a801faf3904d"
)
EXPECTED_P0_STREAM_PAYLOAD = (
    "27e3200e88d31d4f38a921965b631f264aa43f0ef02cb380f41b0c04d8455d1b"
)
EXPECTED_SELECTED_POST_JSON = (
    "d9600cf420d947826c812b88225633b78a889f94f94ce39270dd71bc11b12f0e"
)
STARTER_TILE = 1536
TRAIN_ROOTS = 96
DEVELOPMENT_ROOTS = 32
UNTOUCHED_ROOTS = 192
ROUNDS = 4
TRAJECTORIES_PER_ROUND = 3
TRAJECTORIES_PER_ROOT = 12
EPISODES = 1152
TRAIN_SEED = 2026072703
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


class O3TrainingOperationalHold(RuntimeError):
    """A transient resource or service guard that must not kill O3."""


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
        raise ValueError(f"JSON reload instability: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized)
    os.replace(temporary, path)
    if not _verify_self_hash(json.loads(path.read_text()), field):
        raise ValueError(f"Written self-hash mismatch: {path}")
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


def _task_id(task: Mapping[str, Any]) -> str:
    return _episode_stem(task)


def _validate_attempt_ledger(
    tasks: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    task_by_id = {_task_id(task): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Learning task IDs are not unique")
    states = {
        task_id: {
            "opened": False,
            "completed": False,
            "resume_count": 0,
        }
        for task_id in task_by_id
    }
    for row_index, raw_row in enumerate(
        _read_jsonl(ATTEMPT_PATH) if rows is None else rows
    ):
        row = dict(raw_row)
        task_id = str(row.get("task_id", ""))
        if task_id not in task_by_id:
            raise ValueError(f"Unknown attempt task at row {row_index}")
        state = states[task_id]
        status = row.get("status")
        if status == "opened":
            if state["opened"] or state["completed"]:
                raise ValueError(f"Duplicate attempt open: {task_id}")
            task = task_by_id[task_id]
            expected_identity = {
                "root_index": int(task["root_index"]),
                "round_index": int(task["round_index"]),
                "replicate": int(task["replicate"]),
            }
            if any(row.get(key) != value for key, value in expected_identity.items()):
                raise ValueError(f"Attempt identity mismatch: {task_id}")
            expected_streams = {
                field: int(task[field]) for field in STREAM_FIELDS
            }
            if row.get("stream_ids") != expected_streams:
                raise ValueError(f"Attempt stream mismatch: {task_id}")
            state["opened"] = True
        elif status == "resumed_same_stream":
            if not state["opened"] or state["completed"]:
                raise ValueError(f"Invalid attempt resume: {task_id}")
            state["resume_count"] += 1
        elif status == "completed":
            if not state["opened"] or state["completed"]:
                raise ValueError(f"Duplicate or unopened close: {task_id}")
            if not isinstance(row.get("array_sha256"), str) or not isinstance(
                row.get("metadata_payload_sha256"),
                str,
            ):
                raise ValueError(f"Attempt close lacks hashes: {task_id}")
            state["completed"] = True
        else:
            raise ValueError(f"Unknown attempt status at row {row_index}: {status}")
    return states


def _binding_manifest() -> dict[str, Any]:
    rows = {}
    for name, (path, expected) in EXPECTED_BINDINGS.items():
        observed = sha256_path(path)
        if observed != expected:
            raise ValueError(f"Binding changed: {name}")
        rows[name] = {"path": str(path), "sha256": observed}
    v3_payload = json.loads(V3_ENVELOPE_PATH.read_text())
    if (
        not integrity_v3.verify_self_hash(
            v3_payload,
            "v3_reseal_payload_sha256",
        )
        or v3_payload.get("v3_reseal_payload_sha256") != EXPECTED_V3_PAYLOAD
        or v3_payload.get("decision") != integrity_v3.READY
    ):
        raise ValueError("V3 integrity envelope is not authoritative READY")
    return rows


def training_config() -> dict[str, Any]:
    return {
        "version": VERSION,
        "schema_sha256": schema_sha256(),
        "parameter_count": sum(
            parameter.numel()
            for parameter in O3DesignatedPairNet().parameters()
        ),
        "train_roots": TRAIN_ROOTS,
        "rounds": ROUNDS,
        "trajectories_per_round": TRAJECTORIES_PER_ROUND,
        "trajectories_per_root": TRAJECTORIES_PER_ROOT,
        "episodes": EPISODES,
        "option_horizon": OPTION_HORIZON,
        "epsilon_by_round": list(EPSILON_BY_ROUND),
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
            "2026072703 + 100*round_number_1_based + epoch_index_0_based"
        ),
        "device": "cpu",
        "torch_version": torch.__version__,
        "threads": 1,
        "score_target_used": False,
        "human_or_behavior_action_label_used": False,
        "counterfactual_action_labels_used": False,
        "development_replays_opened": False,
        "untouched_replays_opened": False,
        "runtime_limit_seconds": MAX_RUNTIME_SECONDS,
        "output_limit_bytes": MAX_OUTPUT_BYTES,
    }


def _load_selected_rows() -> list[dict[str, Any]]:
    if sha256_path(SELECTED_PATH) != EXPECTED_BINDINGS["selected_roots"][1]:
        raise ValueError("Selected-root file changed")
    payload = json.loads(SELECTED_PATH.read_text())
    if payload.get("passes") is not True or payload.get("deficits") != []:
        raise ValueError("Selected-root scientific gates do not pass")
    body = dict(payload)
    body.pop("selected_payload_sha256", None)
    if canonical_json_hash(body) != EXPECTED_SELECTED_POST_JSON:
        raise ValueError("Selected-root post-JSON payload changed")
    rows = payload.get("selected")
    if not isinstance(rows, list) or len(rows) != 320:
        raise ValueError("Selected-root row count mismatch")
    counts = Counter(str(row["role"]) for row in rows)
    if counts != Counter(
        {
            "train": TRAIN_ROOTS,
            "development": DEVELOPMENT_ROOTS,
            "untouched_mechanism": UNTOUCHED_ROOTS,
        }
    ):
        raise ValueError(f"Selected role counts mismatch: {counts}")
    roots = [str(row["root_cluster"]) for row in rows]
    if len(roots) != len(set(roots)):
        raise ValueError("Selected roots are not ancestry-disjoint")
    return [dict(row) for row in rows]


def _find_selected_frame(
    replay: Mapping[str, Any],
    frame_index: int,
) -> Mapping[str, Any]:
    frames = replay.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Selected replay has no frame list")
    matches = [
        frame
        for fallback, frame in enumerate(frames)
        if isinstance(frame, dict)
        and int(frame.get("index", fallback)) == int(frame_index)
    ]
    if len(matches) != 1:
        raise ValueError("Selected frame identity is not unique")
    return matches[0]


def restore_train_root(row: Mapping[str, Any]) -> tuple[SimState, DesignatedPair]:
    if row.get("role") != "train":
        raise ValueError("Training restore received a non-train row")
    path = Path(str(row["source_replay"]))
    if sha256_path(path) != row["source_replay_sha256"]:
        raise ValueError(f"Selected source changed: {path}")
    replay = json.loads(path.read_text())
    frame = _find_selected_frame(replay, int(row["frame_index"]))
    state_payload = frame.get("state")
    if not isinstance(state_payload, dict):
        raise ValueError("Selected frame state is malformed")
    state = state_from_replay_payload(state_payload)
    if acquisition.state_signature(state_payload, STARTER_TILE) != row[
        "state_sha1"
    ]:
        raise ValueError("Selected state hash mismatch")
    validator = ThreesSim.from_stream_ids(
        deck_stream_id=0,
        slot_stream_id=1,
        starter_tile=STARTER_TILE,
    )
    pair = select_designated_pair(
        state.board,
        STARTER_TILE,
        requested_target=int(row["target"]),
        allowed_targets=TRAIN_TARGETS,
    )
    if pair is None:
        raise ValueError("Selected canonical pair is missing")
    if [list(value) for value in pair.coordinates] != row["pair"]:
        raise ValueError("Selected canonical pair changed")
    if not root_option_eligible(
        state,
        validator,
        STARTER_TILE,
        allowed_targets=(int(row["target"]),),
    ):
        raise ValueError("Selected root is not a hard start")
    expected_legal = [
        DIRECTION_NAMES[action]
        for action in validator.legal_actions(state)
    ]
    if state_payload.get("legal_actions") != expected_legal:
        raise ValueError("Selected legal actions changed")
    lineage = initial_lineage(pair)
    for action in validator.legal_actions(state):
        tokens, globals_array = option_features(
            state,
            validator,
            starter_tile=STARTER_TILE,
            pair=pair,
            lineage=lineage,
            action=action,
        )
        if not np.isfinite(tokens).all() or not np.isfinite(
            globals_array
        ).all():
            raise ValueError("Selected root features are nonfinite")
    return state, pair


def _source_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in rows if row["role"] == "train"]
    sealed_rows = [row for row in rows if row["role"] != "train"]
    train_audit = []
    for index, row in enumerate(train_rows):
        state, pair = restore_train_root(row)
        train_audit.append(
            {
                "root_index": index,
                "root_cluster": row["root_cluster"],
                "family": row["family"],
                "target": int(row["target"]),
                "source_replay": row["source_replay"],
                "source_replay_sha256": row["source_replay_sha256"],
                "frame_index": int(row["frame_index"]),
                "state_sha1": row["state_sha1"],
                "pair": [list(value) for value in pair.coordinates],
                "legal_count": len(
                    ThreesSim.from_stream_ids(
                        deck_stream_id=0,
                        slot_stream_id=1,
                        starter_tile=STARTER_TILE,
                    ).legal_actions(state)
                ),
                "restored": True,
            }
        )
    sealed_audit = []
    for row in sealed_rows:
        path = Path(str(row["source_replay"]))
        observed = sha256_path(path)
        if observed != row["source_replay_sha256"]:
            raise ValueError(f"Sealed source changed: {path}")
        sealed_audit.append(
            {
                "role": row["role"],
                "root_cluster": row["root_cluster"],
                "source_replay": str(path),
                "source_replay_sha256": observed,
                "content_opened": False,
            }
        )
    return {
        "train_roots_restored": len(train_audit),
        "train_rows": train_audit,
        "sealed_source_files_hashed": len(sealed_audit),
        "sealed_rows": sealed_audit,
        "development_content_opened": False,
        "untouched_content_opened": False,
        "passes": (
            len(train_audit) == TRAIN_ROOTS
            and len(sealed_audit) == DEVELOPMENT_ROOTS + UNTOUCHED_ROOTS
        ),
    }


def _learning_rows() -> list[dict[str, Any]]:
    if sha256_path(P0_STREAM_PATH) != EXPECTED_BINDINGS[
        "p0_stream_manifest"
    ][1]:
        raise ValueError("P0 stream manifest changed")
    payload = json.loads(P0_STREAM_PATH.read_text())
    if (
        not acquisition._verify_self_hash(payload, "payload_sha256")
        or payload.get("payload_sha256") != EXPECTED_P0_STREAM_PAYLOAD
    ):
        raise ValueError("P0 stream payload mismatch")
    rows = [
        dict(row)
        for row in payload["rows"]
        if row.get("purpose") == "learning"
    ]
    rows.sort(
        key=lambda row: (
            int(row["root_index"]),
            int(row["round_index"]),
            int(row["replicate"]),
        )
    )
    expected = [
        (root, round_index, replicate)
        for root in range(TRAIN_ROOTS)
        for round_index in range(ROUNDS)
        for replicate in range(TRAJECTORIES_PER_ROUND)
    ]
    observed = [
        (
            int(row["root_index"]),
            int(row["round_index"]),
            int(row["replicate"]),
        )
        for row in rows
    ]
    if observed != expected or len(rows) != EPISODES:
        raise ValueError("Learning task identity mismatch")
    flat = [int(row[field]) for row in rows for field in STREAM_FIELDS]
    if len(flat) != len(set(flat)):
        raise ValueError("Learning streams are not globally unique")
    return rows


def _collision_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    found: dict[str, set[int]] = defaultdict(set)
    sources = []
    excluded = []
    excluded_dirs = {
        p0.OUTPUT_DIR.resolve(),
        OUTPUT_DIR.resolve(),
    }
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv"}:
            continue
        resolved = path.resolve()
        if any(
            resolved == directory or directory in resolved.parents
            for directory in excluded_dirs
        ):
            excluded.append(str(path))
            continue
        values = history._scan_history_file(path)
        if not values:
            continue
        for key, items in values.items():
            found[key].update(items)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "counts": {
                    key: len(items)
                    for key, items in sorted(values.items())
                },
            }
        )
    collisions = {}
    for field in STREAM_FIELDS:
        requested = {int(row[field]) for row in rows}
        prior = set(found.get(field, set()))
        if field == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                prior.update(found.get(alias, set()))
        collisions[field] = sorted(requested & prior)
    return {
        "requested_rows": len(rows),
        "requested_sha256": canonical_json_hash(list(rows)),
        "scanned_source_count": len(sources),
        "scanned_sources_sha256": canonical_json_hash(sources),
        "excluded_p0_and_current_output_count": len(excluded),
        "collisions": collisions,
        "zero_collisions": not any(collisions.values()),
        "p0_reservation_bound": sha256_path(P0_STREAM_PATH)
        == EXPECTED_BINDINGS["p0_stream_manifest"][1],
        "passes": not any(collisions.values()),
    }


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file()
    )


def operational_audit() -> dict[str, Any]:
    process = recovery.process_audit()
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
        raise O3TrainingOperationalHold(
            f"{stage} operational audit failed: {error}"
        ) from error
    if not audit["passes"]:
        failed = sorted(
            name for name, passed in audit["checks"].items() if not passed
        )
        raise O3TrainingOperationalHold(
            f"{stage} operational guard failed: {failed}"
        )
    return audit


def _execution_error_decision(error: BaseException) -> str:
    if isinstance(error, O3TrainingOperationalHold):
        return "HOLD_O3_TRAINING_OPERATIONAL"
    return "KILL_O3_TRAINING_INTEGRITY"


def _config_identity() -> dict[str, Any]:
    config = training_config()
    if config["schema_sha256"] != (
        "a1c2efa6bd980d32138fb6026c1a5109685db8f1630e1b5fa732b2c2eb983602"
    ):
        raise ValueError("O3 schema hash mismatch")
    if config["parameter_count"] != 102557:
        raise ValueError("O3 parameter count mismatch")
    if config["torch_version"] != "2.12.1":
        raise ValueError(f"Frozen PyTorch mismatch: {config['torch_version']}")
    return config


def _test_evidence_identity() -> dict[str, Any]:
    payload = json.loads(TEST_EVIDENCE_PATH.read_text())
    if not _verify_self_hash(payload, "test_evidence_payload_sha256"):
        raise ValueError("Training test evidence self hash mismatch")
    expected = {
        "charter_sha256": sha256_path(CHARTER_PATH),
        "runner_sha256": sha256_path(RUNNER_PATH),
        "tests_sha256": sha256_path(TEST_PATH),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Training test evidence source mismatch")
    if not payload.get("passes"):
        raise ValueError("Training tests did not pass")
    return {
        "path": str(TEST_EVIDENCE_PATH),
        "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "payload_sha256": payload["test_evidence_payload_sha256"],
        "focused_tests_passed": payload["focused_tests_passed"],
        "regression_tests_passed": payload["regression_tests_passed"],
    }


def _commands() -> dict[str, str]:
    prefix = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.o3_option_training"
    )
    return {
        command: (
            f"{prefix} {command} --out-dir {OUTPUT_DIR}'"
        )
        for command in ("prepare", "open", "execute")
    }


def prepare(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir != OUTPUT_DIR:
        raise ValueError("Training output directory is frozen")
    if out_dir.exists():
        raise FileExistsError(f"Training namespace exists: {out_dir}")
    try:
        bindings = _binding_manifest()
        config = _config_identity()
        evidence = _test_evidence_identity()
        rows = _load_selected_rows()
        source_audit = _source_audit(rows)
        tasks = _learning_rows()
        collision = _collision_audit(tasks)
        operations = operational_audit()
        checks = {
            "bindings_exact": bool(bindings),
            "config_exact": config["parameter_count"] == 102557,
            "tests_pass": bool(evidence),
            "root_counts_exact": len(rows) == 320,
            "train_restoration_pass": source_audit["passes"],
            "learning_tasks_exact": len(tasks) == EPISODES,
            "learning_streams_collision_free": collision["passes"],
            "operations_pass": operations["passes"],
            "zero_training_work": True,
        }
        decision = (
            "READY_O3_OPTION_TRAINING_EXECUTION"
            if all(checks.values())
            else "HOLD_O3_TRAINING_PREFLIGHT"
        )
        out_dir.mkdir(parents=True, exist_ok=False)
        config_artifact = _write_immutable(
            CONFIG_PATH,
            config,
            "config_payload_sha256",
        )
        root_artifact = _write_immutable(
            ROOT_MANIFEST_PATH,
            {
                "version": f"{VERSION}_roots",
                "rows": rows,
                "role_counts": dict(
                    Counter(str(row["role"]) for row in rows)
                ),
                "root_manifest_sha256": canonical_json_hash(rows),
            },
            "root_artifact_payload_sha256",
        )
        source_artifact = _write_immutable(
            SOURCE_AUDIT_PATH,
            source_audit,
            "source_audit_payload_sha256",
        )
        task_artifact = _write_immutable(
            TASK_MANIFEST_PATH,
            {
                "version": f"{VERSION}_tasks",
                "rows": tasks,
                "task_manifest_sha256": canonical_json_hash(tasks),
            },
            "task_artifact_payload_sha256",
        )
        collision_artifact = _write_immutable(
            COLLISION_PATH,
            collision,
            "collision_payload_sha256",
        )
        lock_payload = {
            "version": f"{VERSION}_preflight_lock",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": decision,
            "bindings": bindings,
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
                "payload_sha256": root_artifact[
                    "root_artifact_payload_sha256"
                ],
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
                "payload_sha256": task_artifact[
                    "task_artifact_payload_sha256"
                ],
            },
            "collision": {
                "path": str(COLLISION_PATH),
                "file_sha256": sha256_path(COLLISION_PATH),
                "payload_sha256": collision_artifact[
                    "collision_payload_sha256"
                ],
            },
            "commands": _commands(),
            "operations": operations,
            "counts": {
                "train_roots": TRAIN_ROOTS,
                "development_roots_sealed": DEVELOPMENT_ROOTS,
                "untouched_roots_sealed": UNTOUCHED_ROOTS,
                "learning_tasks": EPISODES,
            },
            "zero_work": {
                "labels": 0,
                "episodes": 0,
                "streams": 0,
                "models": 0,
                "development_content_opened": False,
                "untouched_content_opened": False,
                "policy_outcomes": 0,
            },
            "checks": checks,
        }
        lock = _write_immutable(
            PREFLIGHT_LOCK_PATH,
            lock_payload,
            "preflight_lock_payload_sha256",
        )
        result = _write_immutable(
            PREFLIGHT_RESULT_PATH,
            {
                "version": f"{VERSION}_preflight_result",
                "decision": decision,
                "continue": decision
                == "READY_O3_OPTION_TRAINING_EXECUTION",
                "hold": decision != "READY_O3_OPTION_TRAINING_EXECUTION",
                "kill": False,
                "promote": False,
                "preflight_lock_file_sha256":
                    sha256_path(PREFLIGHT_LOCK_PATH),
                "preflight_lock_payload_sha256":
                    lock["preflight_lock_payload_sha256"],
                "checks": checks,
                "operations": operations,
                "zero_work": lock_payload["zero_work"],
            },
            "preflight_result_payload_sha256",
        )
        return {
            "decision": decision,
            "preflight_lock_file_sha256": sha256_path(PREFLIGHT_LOCK_PATH),
            "preflight_lock_payload_sha256":
                lock["preflight_lock_payload_sha256"],
            "preflight_result_file_sha256": sha256_path(
                PREFLIGHT_RESULT_PATH
            ),
            "preflight_result_payload_sha256":
                result["preflight_result_payload_sha256"],
        }
    except Exception as error:
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
        result = _write_immutable(
            PREFLIGHT_RESULT_PATH,
            {
                "version": f"{VERSION}_preflight_result",
                "decision": "KILL_O3_TRAINING_INTEGRITY",
                "continue": False,
                "hold": False,
                "kill": True,
                "promote": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "zero_work": {
                    "labels": 0,
                    "episodes": 0,
                    "streams": 0,
                    "models": 0,
                    "policy_outcomes": 0,
                },
            },
            "preflight_result_payload_sha256",
        )
        return {
            "decision": result["decision"],
            "preflight_result_file_sha256": sha256_path(
                PREFLIGHT_RESULT_PATH
            ),
            "preflight_result_payload_sha256":
                result["preflight_result_payload_sha256"],
            "error": str(error),
        }


def _load_preflight_lock() -> dict[str, Any]:
    payload = json.loads(PREFLIGHT_LOCK_PATH.read_text())
    if not _verify_self_hash(payload, "preflight_lock_payload_sha256"):
        raise ValueError("Training preflight lock self hash mismatch")
    if payload.get("decision") != "READY_O3_OPTION_TRAINING_EXECUTION":
        raise ValueError("Training preflight is not READY")
    for name, item in payload.get("bindings", {}).items():
        if sha256_path(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Training dependency binding changed: {name}")
    for name in ("charter", "runner", "tests", "test_evidence", "config",
                 "roots", "source_audit", "tasks", "collision"):
        item = payload[name]
        expected = item.get("file_sha256", item.get("sha256"))
        if not isinstance(expected, str):
            raise ValueError(f"Training preflight binding lacks SHA: {name}")
        if sha256_path(Path(item["path"])) != expected:
            raise ValueError(f"Training preflight binding changed: {name}")
    return payload


def open_execution() -> dict[str, Any]:
    if MARKER_PATH.exists() or RESULT_PATH.exists():
        raise FileExistsError("Training marker or terminal result exists")
    lock = _load_preflight_lock()
    if ATTEMPT_PATH.exists() or RUNTIME_PATH.exists() \
            or EPISODE_DIR.exists() or CHECKPOINT_DIR.exists():
        raise ValueError("Training work exists before marker")
    collision = _collision_audit(_learning_rows())
    operations = operational_audit()
    if not collision["passes"] or not operations["passes"]:
        raise ValueError("Training open operational revalidation failed")
    payload = {
        "version": f"{VERSION}_opened",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preflight_lock_file_sha256": sha256_path(PREFLIGHT_LOCK_PATH),
        "preflight_lock_payload_sha256":
            lock["preflight_lock_payload_sha256"],
        "execute_command": lock["commands"]["execute"],
        "collision_recheck": collision,
        "operations": operations,
        "zero_work_before_marker": {
            "labels": 0,
            "episodes": 0,
            "streams": 0,
            "models": 0,
            "development_content_opened": False,
            "untouched_content_opened": False,
            "policy_outcomes": 0,
        },
    }
    marker = _write_immutable(
        MARKER_PATH,
        payload,
        "marker_payload_sha256",
    )
    return {
        "decision": "READY_O3_OPTION_TRAINING_OPENED",
        "marker_file_sha256": sha256_path(MARKER_PATH),
        "marker_payload_sha256": marker["marker_payload_sha256"],
    }


def _load_marker() -> dict[str, Any]:
    payload = json.loads(MARKER_PATH.read_text())
    if not _verify_self_hash(payload, "marker_payload_sha256"):
        raise ValueError("Training marker self hash mismatch")
    lock = _load_preflight_lock()
    if payload.get("preflight_lock_file_sha256") != sha256_path(
        PREFLIGHT_LOCK_PATH
    ):
        raise ValueError("Training marker lock file mismatch")
    if payload.get("preflight_lock_payload_sha256") != lock[
        "preflight_lock_payload_sha256"
    ]:
        raise ValueError("Training marker lock payload mismatch")
    if payload.get("execute_command") != lock["commands"]["execute"]:
        raise ValueError("Training marker command mismatch")
    return payload


def _pair_from_lineage(
    state: SimState,
    lineage: np.ndarray,
    target: int,
) -> DesignatedPair:
    if lineage_integrity(lineage) != "live":
        raise ValueError("Live pair requested from invalid lineage")
    a = np.argwhere((lineage & LINEAGE_A) != 0)
    b = np.argwhere((lineage & LINEAGE_B) != 0)
    if a.shape != (1, 2) or b.shape != (1, 2):
        raise ValueError("Expected exactly one A and one B descendant")
    coordinates = tuple(
        sorted(
            (
                (int(a[0, 0]), int(a[0, 1])),
                (int(b[0, 0]), int(b[0, 1])),
            )
        )
    )
    (r0, c0), (r1, c1) = coordinates
    manhattan = abs(r0 - r1) + abs(c0 - c1)
    chebyshev = max(abs(r0 - r1), abs(c0 - c1))
    blockers = pair_blocker_count(state.board, coordinates)
    same_row = r0 == r1
    same_column = c0 == c1
    return DesignatedPair(
        target=int(target),
        coordinates=coordinates,
        manhattan=int(manhattan),
        chebyshev=int(chebyshev),
        blocker_count=int(blockers),
        same_row=same_row,
        same_column=same_column,
        clear_line=bool((same_row or same_column) and blockers == 0),
        safe_merge_actions=tuple(
            pair_safe_merge_actions(
                state.board,
                coordinates,
                int(target),
                STARTER_TILE,
            )
        ),
    )


def _normalized_geometry(
    state: SimState,
    sim: ThreesSim,
    lineage: np.ndarray,
    target: int,
) -> np.ndarray:
    pair = _pair_from_lineage(state, lineage, target)
    values = np.asarray(
        (
            pair.manhattan / 6.0,
            pair.chebyshev / 3.0,
            pair.blocker_count / 6.0,
            float(pair.same_row),
            float(pair.same_column),
            np.count_nonzero(state.board == 0) / 16.0,
            len(sim.legal_actions(state)) / 4.0,
            float(lineage_integrity(lineage) == "live"),
        ),
        dtype=np.float32,
    )
    if values.shape != (GEOMETRY_WIDTH,) or not np.isfinite(values).all():
        raise ValueError("O3 normalized geometry is invalid")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("O3 normalized geometry is outside [0,1]")
    return values


def _model_outputs(
    model: O3DesignatedPairNet,
    state: SimState,
    sim: ThreesSim,
    lineage: np.ndarray,
    pair: DesignatedPair,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
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
    return (
        {action: outputs[index] for index, action in enumerate(legal)},
        features,
    )


def generate_episode(
    *,
    root_row: Mapping[str, Any],
    task: Mapping[str, Any],
    model: O3DesignatedPairNet | None,
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
    round_index = int(task["round_index"])
    epsilon = EPSILON_BY_ROUND[round_index]

    decisions: list[dict[str, Any]] = []
    live_geometry_by_move: dict[int, np.ndarray] = {
        0: _normalized_geometry(
            state,
            sim,
            lineage,
            int(root_row["target"]),
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
        outputs: dict[int, np.ndarray] | None = None
        features_by_action: dict[
            int,
            tuple[np.ndarray, np.ndarray],
        ]
        if round_index == 0:
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
            if model is None:
                raise ValueError("Learned O3 round requires a frozen checkpoint")
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
            raise RuntimeError("Chosen legal O3 action did not move")
        shifted = next_state.board.copy()
        if info.inserted_pos is not None:
            shifted[info.inserted_pos] = 0
        if not np.array_equal(base.board, shifted):
            raise RuntimeError("Tagged O3 afterstate diverged from simulator")
        if tuple(base.eligible_slots) != tuple(info.eligible_positions):
            raise RuntimeError("Tagged O3 insertion slots diverged")
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
            live_geometry_by_move[completed_move] = _normalized_geometry(
                state,
                sim,
                lineage,
                int(root_row["target"]),
            )
        else:
            terminal_status = status
            terminal_move = completed_move
            break
    else:
        terminal_status = "censor"
        terminal_move = OPTION_HORIZON

    if terminal_status == "censor" and terminal_move == OPTION_HORIZON:
        if OPTION_HORIZON not in live_geometry_by_move:
            live_geometry_by_move[OPTION_HORIZON] = _normalized_geometry(
                state,
                sim,
                lineage,
                int(root_row["target"]),
            )

    event_class = []
    event_mask = []
    geometry = []
    geometry_mask = []
    for decision in decisions:
        targets = build_decision_targets(
            decision_move=int(decision["decision_move"]),
            terminal_move=int(terminal_move),
            terminal_status=terminal_status,
            live_geometry_by_move=live_geometry_by_move,
        )
        event_class.append(
            -1 if targets.event_class is None else targets.event_class
        )
        event_mask.append(targets.event_mask)
        geometry.append(targets.geometry)
        geometry_mask.append(targets.geometry_mask)

    row_count = len(decisions)
    if row_count == 0:
        raise ValueError("O3 episode produced no decision row")
    arrays = {
        "tokens": np.stack([row["tokens"] for row in decisions]).astype(
            np.float32
        ),
        "globals": np.stack([row["globals"] for row in decisions]).astype(
            np.float32
        ),
        "actions": np.asarray(
            [row["action"] for row in decisions],
            dtype=np.int8,
        ),
        "decision_moves": np.asarray(
            [row["decision_move"] for row in decisions],
            dtype=np.int8,
        ),
        "event_class": np.asarray(event_class, dtype=np.int8),
        "event_mask": np.asarray(event_mask, dtype=np.bool_),
        "geometry": np.stack(geometry).astype(np.float32),
        "geometry_mask": np.stack(geometry_mask).astype(np.bool_),
    }
    if any(
        np.issubdtype(value.dtype, np.floating)
        and not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise ValueError("O3 episode contains nonfinite arrays")
    metadata = {
        "version": f"{VERSION}_episode",
        "root_index": int(task["root_index"]),
        "round_index": round_index,
        "replicate": int(task["replicate"]),
        "root_cluster": root_row["root_cluster"],
        "family": root_row["family"],
        "target": int(root_row["target"]),
        "terminal_status": terminal_status,
        "terminal_move": int(terminal_move),
        "decision_rows": row_count,
        "stream_ids": {
            field: int(task[field]) for field in STREAM_FIELDS
        },
    }
    return arrays, metadata


def _episode_stem(task: Mapping[str, Any]) -> str:
    return (
        f"r{int(task['round_index'])}_root{int(task['root_index']):03d}_"
        f"rep{int(task['replicate'])}"
    )


def _episode_paths(task: Mapping[str, Any]) -> tuple[Path, Path]:
    stem = _episode_stem(task)
    return EPISODE_DIR / f"{stem}.npz", EPISODE_DIR / f"{stem}.json"


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


def _load_array_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _write_episode(
    task: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> tuple[str, str]:
    array_path, metadata_path = _episode_paths(task)
    if metadata_path.exists() and not array_path.exists():
        raise ValueError("Committed episode metadata lacks its array artifact")
    if metadata_path.exists():
        observed_arrays, observed_metadata = _load_episode(task)
        if not _arrays_equal(observed_arrays, arrays):
            raise ValueError("Committed episode arrays differ on regeneration")
        if any(
            observed_metadata.get(key) != value
            for key, value in metadata.items()
        ):
            raise ValueError("Committed episode metadata differs on regeneration")
        return (
            sha256_path(array_path),
            str(observed_metadata["episode_payload_sha256"]),
        )
    if array_path.exists():
        observed_arrays = _load_array_artifact(array_path)
        if not _arrays_equal(observed_arrays, arrays):
            raise ValueError("Orphan episode array differs on regeneration")
    else:
        array_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_name(
            f".{array_path.name}.tmp.{os.getpid()}"
        )
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, array_path)
    metadata_payload = _write_immutable(
        metadata_path,
        {
            **dict(metadata),
            "array_file_sha256": sha256_path(array_path),
        },
        "episode_payload_sha256",
    )
    return sha256_path(array_path), metadata_payload["episode_payload_sha256"]


def _load_episode(
    task: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    array_path, metadata_path = _episode_paths(task)
    metadata = json.loads(metadata_path.read_text())
    if not _verify_self_hash(metadata, "episode_payload_sha256"):
        raise ValueError("Episode metadata self hash mismatch")
    expected = (
        int(task["root_index"]),
        int(task["round_index"]),
        int(task["replicate"]),
    )
    observed = (
        int(metadata["root_index"]),
        int(metadata["round_index"]),
        int(metadata["replicate"]),
    )
    if observed != expected:
        raise ValueError("Episode task identity mismatch")
    if sha256_path(array_path) != metadata["array_file_sha256"]:
        raise ValueError("Episode array hash mismatch")
    arrays = _load_array_artifact(array_path)
    return arrays, metadata


def _initialize_training() -> tuple[O3DesignatedPairNet, torch.optim.Optimizer]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(TRAIN_SEED)
    model = O3DesignatedPairNet().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    return model, optimizer


def _checkpoint_path(round_number: int) -> Path:
    return CHECKPOINT_DIR / (
        "initial.pt" if round_number == 0 else f"round_{round_number}.pt"
    )


def _save_checkpoint(
    path: Path,
    *,
    model: O3DesignatedPairNet,
    optimizer: torch.optim.Optimizer,
    round_number: int,
    config_sha256: str,
) -> str:
    if path.exists():
        raise FileExistsError(f"Checkpoint exists: {path}")
    payload = {
        "version": f"{VERSION}_checkpoint",
        "round_number": int(round_number),
        "schema_sha256": schema_sha256(),
        "config_sha256": config_sha256,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_version": torch.__version__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if (
        reloaded["round_number"] != round_number
        or reloaded["schema_sha256"] != schema_sha256()
        or reloaded["config_sha256"] != config_sha256
    ):
        raise ValueError("Checkpoint reload identity mismatch")
    for name, value in model.state_dict().items():
        if not torch.equal(value, reloaded["model_state"][name]):
            raise ValueError("Checkpoint reload model mismatch")
    return sha256_path(path)


def _load_checkpoint(
    path: Path,
    *,
    config_sha256: str,
) -> tuple[O3DesignatedPairNet, torch.optim.Optimizer, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_sha256") != schema_sha256()
        or payload.get("config_sha256") != config_sha256
    ):
        raise ValueError("Checkpoint schema/config mismatch")
    model, optimizer = _initialize_training()
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return model, optimizer, int(payload["round_number"])


def _buffer_arrays(
    tasks: Sequence[Mapping[str, Any]],
    roots: Sequence[Mapping[str, Any]],
    maximum_round_index: int,
) -> dict[str, np.ndarray]:
    family_root_counts = Counter(str(row["family"]) for row in roots)
    represented_families = len(family_root_counts)
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for task in tasks:
        if int(task["round_index"]) > maximum_round_index:
            continue
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
    return {name: np.concatenate(values, axis=0) for name, values in chunks.items()}


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum()
    if float(denominator) <= 0.0:
        return values.sum() * 0.0
    return (values * weights).sum() / denominator


def fit_round(
    *,
    model: O3DesignatedPairNet,
    optimizer: torch.optim.Optimizer,
    arrays: Mapping[str, np.ndarray],
    round_number: int,
) -> None:
    row_count = arrays["tokens"].shape[0]
    continuous = torch.tensor([0, 1, 2, 5, 6], dtype=torch.long)
    binary = torch.tensor([3, 4, 7], dtype=torch.long)
    for epoch_index in range(EPOCHS_PER_ROUND):
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
                classes = torch.from_numpy(
                    arrays["event_class"][index].astype(np.int64)
                )
                event_losses = F.cross_entropy(
                    output[:, :EVENT_WIDTH],
                    classes.clamp_min(0),
                    reduction="none",
                )
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
            geometry_mask = torch.from_numpy(arrays["geometry_mask"][index])
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
                total = total + GEOMETRY_LOSS_COEFFICIENT * _weighted_mean(
                    per_row,
                    weights,
                )
            if not torch.isfinite(total):
                raise ValueError("O3 training loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("O3 checkpoint contains nonfinite parameters")


def _runtime_state() -> dict[str, Any]:
    if not RUNTIME_PATH.exists():
        return {
            "version": f"{VERSION}_runtime",
            "active_seconds": 0.0,
            "completed_tasks": 0,
        }
    return json.loads(RUNTIME_PATH.read_text())


def _write_runtime(payload: Mapping[str, Any]) -> None:
    temporary = RUNTIME_PATH.with_name(f".{RUNTIME_PATH.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, RUNTIME_PATH)


def _reconcile_runtime_completions(
    runtime: dict[str, Any],
    completed_count: int,
) -> bool:
    recorded = int(runtime.get("completed_tasks", 0))
    if recorded > completed_count:
        raise ValueError("Runtime completion count exceeds attempt closes")
    if recorded == completed_count:
        return False
    runtime["completed_tasks"] = int(completed_count)
    return True


def _support_report(
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metadata_rows = [_load_episode(task)[1] for task in tasks]
    statuses = Counter(row["terminal_status"] for row in metadata_rows)
    success_rows = [
        row for row in metadata_rows if row["terminal_status"] == "success"
    ]
    success_by_target = Counter(int(row["target"]) for row in success_rows)
    success_by_family = Counter(str(row["family"]) for row in success_rows)
    time_bins = Counter(
        "1_10" if int(row["terminal_move"]) <= 10
        else "11_20" if int(row["terminal_move"]) <= 20
        else "21_40"
        for row in success_rows
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
        "successes_at_least_40": len(success_rows) >= 40,
        "six_successes_each_target": all(
            success_by_target[target] >= 6 for target in TRAIN_TARGETS
        ),
        "four_families_three_successes": sum(
            count >= 3 for count in success_by_family.values()
        ) >= 4,
        "failures_at_least_40": statuses["failure"] >= 40,
        "censors_at_least_40": statuses["censor"] >= 40,
        "finite_arrays": finite,
        "two_nonempty_success_bins": sum(
            count > 0 for count in time_bins.values()
        ) >= 2,
    }
    return {
        "version": f"{VERSION}_label_support",
        "episodes": len(metadata_rows),
        "status_counts": dict(statuses),
        "success_by_target": {
            str(target): success_by_target[target]
            for target in TRAIN_TARGETS
        },
        "success_by_family": dict(sorted(success_by_family.items())),
        "success_time_bins": dict(time_bins),
        "checks": checks,
        "passes": all(checks.values()),
        "score_or_human_action_labels_used": False,
    }


def execute() -> dict[str, Any]:
    marker = _load_marker()
    if RESULT_PATH.exists():
        raise FileExistsError("Training terminal result exists")
    try:
        operations = _require_operational("execution start")
        lock = _load_preflight_lock()
        roots_payload = json.loads(ROOT_MANIFEST_PATH.read_text())
        tasks_payload = json.loads(TASK_MANIFEST_PATH.read_text())
        roots = [
            row for row in roots_payload["rows"] if row["role"] == "train"
        ]
        tasks = list(tasks_payload["rows"])
        config_sha = lock["config"]["payload_sha256"]
        if len(roots) != TRAIN_ROOTS or len(tasks) != EPISODES:
            raise ValueError("Training execution manifest count mismatch")

        initial_path = _checkpoint_path(0)
        if not initial_path.exists():
            model, optimizer = _initialize_training()
            _save_checkpoint(
                initial_path,
                model=model,
                optimizer=optimizer,
                round_number=0,
                config_sha256=config_sha,
            )

        runtime = _runtime_state()
        attempts = _read_jsonl(ATTEMPT_PATH)
        attempt_states = _validate_attempt_ledger(tasks, attempts)
        completed = {
            task_id
            for task_id, state in attempt_states.items()
            if state["completed"]
        }
        if _reconcile_runtime_completions(runtime, len(completed)):
            _write_runtime(runtime)
        for round_index in range(ROUNDS):
            prior_path = _checkpoint_path(round_index)
            model, _, loaded_round = _load_checkpoint(
                prior_path,
                config_sha256=config_sha,
            )
            if loaded_round != round_index:
                raise ValueError("Prior checkpoint round mismatch")
            round_tasks = [
                task for task in tasks
                if int(task["round_index"]) == round_index
            ]
            checkpoint = _checkpoint_path(round_index + 1)
            if checkpoint.exists() and any(
                not attempt_states[_task_id(task)]["completed"]
                for task in round_tasks
            ):
                raise ValueError(
                    "Round checkpoint precedes complete episode artifacts"
                )
            for task in round_tasks:
                task_id = _task_id(task)
                array_path, metadata_path = _episode_paths(task)
                state = attempt_states[task_id]
                if state["completed"]:
                    if not array_path.exists() or not metadata_path.exists():
                        raise ValueError(
                            "Closed attempt lacks committed episode artifacts"
                        )
                    _load_episode(task)
                    continue
                if metadata_path.exists() and not array_path.exists():
                    raise ValueError(
                        "Episode metadata exists without its array artifact"
                    )
                if (array_path.exists() or metadata_path.exists()) and not state[
                    "opened"
                ]:
                    raise ValueError("Episode artifact exists before attempt open")
                if array_path.exists() and metadata_path.exists():
                    _, committed_metadata = _load_episode(task)
                    _append_jsonl(
                        ATTEMPT_PATH,
                        {
                            "task_id": task_id,
                            "status": "completed",
                            "array_sha256": sha256_path(array_path),
                            "metadata_payload_sha256": committed_metadata[
                                "episode_payload_sha256"
                            ],
                        },
                    )
                    state["completed"] = True
                    completed.add(task_id)
                    runtime["completed_tasks"] = len(completed)
                    _write_runtime(runtime)
                    continue
                if not state["opened"]:
                    _append_jsonl(
                        ATTEMPT_PATH,
                        {
                            "task_id": task_id,
                            "status": "opened",
                            "root_index": int(task["root_index"]),
                            "round_index": round_index,
                            "replicate": int(task["replicate"]),
                            "stream_ids": {
                                field: int(task[field])
                                for field in STREAM_FIELDS
                            },
                        },
                    )
                    state["opened"] = True
                else:
                    _append_jsonl(
                        ATTEMPT_PATH,
                        {"task_id": task_id, "status": "resumed_same_stream"},
                    )
                    state["resume_count"] += 1
                started = time.perf_counter()
                arrays, metadata = generate_episode(
                    root_row=roots[int(task["root_index"])],
                    task=task,
                    model=None if round_index == 0 else model,
                )
                elapsed = time.perf_counter() - started
                runtime["active_seconds"] = (
                    float(runtime.get("active_seconds", 0.0)) + elapsed
                )
                _write_runtime(runtime)
                array_sha, metadata_sha = _write_episode(
                    task,
                    arrays,
                    metadata,
                )
                _append_jsonl(
                    ATTEMPT_PATH,
                    {
                        "task_id": task_id,
                        "status": "completed",
                        "array_sha256": array_sha,
                        "metadata_payload_sha256": metadata_sha,
                    },
                )
                state["completed"] = True
                completed.add(task_id)
                runtime["completed_tasks"] = len(completed)
                _write_runtime(runtime)
                if runtime["active_seconds"] > MAX_RUNTIME_SECONDS:
                    raise O3TrainingOperationalHold(
                        "Training active-runtime limit exceeded"
                    )
                if len(completed) % 24 == 0:
                    _require_operational("episode boundary")
            if not checkpoint.exists():
                model, optimizer, loaded_round = _load_checkpoint(
                    prior_path,
                    config_sha256=config_sha,
                )
                if loaded_round != round_index:
                    raise ValueError("Fit checkpoint predecessor mismatch")
                arrays = _buffer_arrays(tasks, roots, round_index)
                started = time.perf_counter()
                fit_round(
                    model=model,
                    optimizer=optimizer,
                    arrays=arrays,
                    round_number=round_index + 1,
                )
                runtime["active_seconds"] += time.perf_counter() - started
                _write_runtime(runtime)
                if runtime["active_seconds"] > MAX_RUNTIME_SECONDS:
                    raise O3TrainingOperationalHold(
                        "Training active-runtime limit exceeded during fit"
                    )
                _save_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    round_number=round_index + 1,
                    config_sha256=config_sha,
                )
            print(
                f"phase=round_{round_index + 1}_complete "
                f"episodes={len(completed)}/{EPISODES}",
                flush=True,
            )

        if len(completed) != EPISODES:
            raise ValueError("Training episode completion count mismatch")
        support = _support_report(tasks)
        support_artifact = _write_immutable(
            SUPPORT_REPORT_PATH,
            support,
            "support_payload_sha256",
        )
        final_checkpoint = _checkpoint_path(ROUNDS)
        operations = _require_operational("training terminal")
        decision = (
            "READY_O3_OPTION_DEVELOPMENT"
            if support["passes"]
            else "HOLD_O3_LABEL_SUPPORT"
        )
        result = _write_immutable(
            RESULT_PATH,
            {
                "version": f"{VERSION}_result",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "decision": decision,
                "continue": decision == "READY_O3_OPTION_DEVELOPMENT",
                "hold": decision == "HOLD_O3_LABEL_SUPPORT",
                "kill": False,
                "promote": False,
                "marker_file_sha256": sha256_path(MARKER_PATH),
                "marker_payload_sha256": marker["marker_payload_sha256"],
                "episodes": EPISODES,
                "attempt_rows": len(_read_jsonl(ATTEMPT_PATH)),
                "final_checkpoint": {
                    "path": str(final_checkpoint),
                    "sha256": sha256_path(final_checkpoint),
                },
                "support_report": {
                    "path": str(SUPPORT_REPORT_PATH),
                    "file_sha256": sha256_path(SUPPORT_REPORT_PATH),
                    "payload_sha256": support_artifact[
                        "support_payload_sha256"
                    ],
                },
                "runtime": _runtime_state(),
                "operations": operations,
                "zero_downstream_work": {
                    "development_replay_content_opened": False,
                    "untouched_replay_content_opened": False,
                    "mechanism_outcomes": 0,
                    "normal_start_policy_games": 0,
                    "incumbent_changes": 0,
                    "dashboard_changes": 0,
                },
            },
            "result_payload_sha256",
        )
        return {
            "decision": decision,
            "result_file_sha256": sha256_path(RESULT_PATH),
            "result_payload_sha256": result["result_payload_sha256"],
        }
    except Exception as error:
        decision = _execution_error_decision(error)
        result = _write_immutable(
            RESULT_PATH,
            {
                "version": f"{VERSION}_result",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "decision": decision,
                "continue": False,
                "hold": decision == "HOLD_O3_TRAINING_OPERATIONAL",
                "kill": decision == "KILL_O3_TRAINING_INTEGRITY",
                "promote": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "marker_file_sha256": sha256_path(MARKER_PATH),
                "marker_payload_sha256": marker["marker_payload_sha256"],
                "partial_work_preserved": True,
                "development_replay_content_opened": False,
                "untouched_replay_content_opened": False,
            },
            "result_payload_sha256",
        )
        return {
            "decision": result["decision"],
            "result_file_sha256": sha256_path(RESULT_PATH),
            "result_payload_sha256": result["result_payload_sha256"],
            "error": str(error),
        }


def write_test_evidence(
    *,
    focused: int,
    regressions: int,
    recorded_commands: Sequence[str],
) -> dict[str, Any]:
    if TEST_EVIDENCE_PATH.exists():
        raise FileExistsError("Training test evidence exists")
    payload = _write_immutable(
        TEST_EVIDENCE_PATH,
        {
            "version": f"{VERSION}_test_evidence",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "charter_sha256": sha256_path(CHARTER_PATH),
            "runner_sha256": sha256_path(RUNNER_PATH),
            "tests_sha256": sha256_path(TEST_PATH),
            "focused_tests_passed": int(focused),
            "regression_tests_passed": int(regressions),
            "recorded_commands": list(recorded_commands),
            "passes": focused > 0 and regressions > 0,
            "zero_work": {
                "labels": 0,
                "episodes": 0,
                "streams": 0,
                "models": 0,
                "policy_outcomes": 0,
            },
        },
        "test_evidence_payload_sha256",
    )
    return {
        "file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "payload_sha256": payload["test_evidence_payload_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--focused", type=int, required=True)
    evidence.add_argument("--regressions", type=int, required=True)
    evidence.add_argument(
        "--recorded-command",
        dest="recorded_commands",
        action="append",
        required=True,
    )
    for command in ("prepare", "open", "execute"):
        child = subparsers.add_parser(command)
        child.add_argument("--out-dir", type=Path, required=True)
    return parser


def dispatch(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.subcommand == "write-test-evidence":
        return write_test_evidence(
            focused=args.focused,
            regressions=args.regressions,
            recorded_commands=args.recorded_commands,
        )
    if args.out_dir != OUTPUT_DIR:
        raise ValueError("Training output directory does not match contract")
    if args.subcommand == "prepare":
        return prepare(args.out_dir)
    if args.subcommand == "open":
        return open_execution()
    if args.subcommand == "execute":
        return execute()
    raise AssertionError(args.subcommand)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(dispatch(argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

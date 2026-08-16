"""Fresh normal-start source acquisition for G2 upward-transfer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl import g1r_acquire as base
from threes_rl.eval import (
    EvalJob,
    EvalStreamIds,
    iter_eval_job_outputs,
    make_policy,
    max_tile_excluding_initial_starter,
)
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.record_replay import state_payload
from threes_rl.replay_provenance import ORIGIN_FRESH, direct_root_fields
from threes_rl.restart_manifest import state_signature
from threes_rl.run_artifacts import write_json
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "g2_fresh_transfer_acquisition_v1"
CHARTER_PATH = Path("threes_rl/G2_FRESH_TRANSFER_ACQUISITION_CHARTER.md")
IMPLEMENTATION_PATH = Path("threes_rl/g2_fresh_transfer_acquire.py")
TEST_PATH = Path("tests/test_rl_g2_fresh_transfer_acquire.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1_test_evidence.json"
)
OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1"
)
INCUMBENT_PATH = Path("threes_rl/current_incumbent_policy.txt")
PILOT_V1_LOCK = Path(
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v1/"
    "preflight_lock_pilot_v1.json"
)
G2_PROPOSAL = Path("threes_rl/G2_SCALE_EQUIVARIANT_RELATIONAL_HAZARD_PROPOSAL.md")
G2_FEATURE_IMPL = Path("threes_rl/g2_scale_relational_hazard.py")
G2_PREFLIGHT_IMPL = Path("threes_rl/g2_scale_relational_hazard_preflight.py")
G2_PREFLIGHT = Path(
    "threes_rl/runs/forensics/g2_scale_equivariant_relational_hazard/"
    "G2_PREFLIGHT.json"
)
G2_ROOT_MANIFEST = G2_PREFLIGHT.parent / "G2_ROOT_MANIFEST.json"
G2_TEST_EVIDENCE = Path(
    "threes_rl/runs/forensics/g2_scale_equivariant_relational_hazard_test_evidence.json"
)
QD_STORAGE_AUDIT = Path(
    "threes_rl/runs/forensics/g1r_qd_admission_v2_terminal_schema/"
    "QD_V2_STORAGE_ADMISSION_AUDIT.json"
)
QD_STORAGE_INVENTORY = QD_STORAGE_AUDIT.with_name(
    "QD_V2_STORAGE_REPLAY_INVENTORY.json"
)
QD5_SEAL = Path(
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v2_qd5/PILOT_V2_SEAL.json"
)

CHARTER_SHA256 = "bebfbbce45c25750c5a9247288f4c02fa1f1deb7d8fe0f842d3b54d522a2526d"
PILOT_V1_LOCK_SHA256 = (
    "f78288b3f47bda6aa6d15c2157fd79f7b3d0685f0367d8b9964f5dc73981ea91"
)
G2_LOCKS = {
    G2_PROPOSAL: "43b413c1a8145a25750009cc3048bbda6127a44cfccbf72c7d1710e1e6027099",
    G2_FEATURE_IMPL: "9ffaa45dd36b633cdae10110fdaefc8cd27053ab3f0216ddb3f1886ea625af8a",
    G2_PREFLIGHT_IMPL: "b5feebe5965258016480aca95f9a690392f0c3bdd7d0a3b73d5efddf35f02559",
    G2_PREFLIGHT: "2e1084f2a0673935866839e89765d3d1a31a2c2348e99c01edc9abc2405f05cc",
    G2_ROOT_MANIFEST: "60d514ed79ff315f7c2e0d2ad13bb712a57d4c3b204587691aa878a7486ea2ca",
    G2_TEST_EVIDENCE: "6319a2946ae2c40ec2eea55da8ddc16f6755fe604dcd3a8514699aa69ce76b25",
}
G2_SCHEMA_SHA256 = (
    "6af0cd515e5886b5fd8bc4d9f52cc9202bd3ed1f149d0ae146829681aea8340e"
)
EXPECTED_SIGNATURES = {
    "g2_transfer_corner2": (
        "4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043"
    ),
    "g2_transfer_expectimax2": (
        "2ad642cdca7739cc73af4f570de5054c422815f9a7d8f93a2619921b46b74b38"
    ),
    "g2_transfer_phaseblend_incumbent": (
        "868a6337d932cc034a633272d10fea3fc733a3f542b49a13eda1c075371d1ccb"
    ),
}
EXPECTED_PAIRWISE = {
    ("g2_transfer_corner2", "g2_transfer_expectimax2"): (
        0.59375,
        {"pre1536": 0.78125, "pre3072": 0.40625},
    ),
    ("g2_transfer_corner2", "g2_transfer_phaseblend_incumbent"): (
        0.53125,
        {"pre1536": 0.59375, "pre3072": 0.46875},
    ),
    ("g2_transfer_expectimax2", "g2_transfer_phaseblend_incumbent"): (
        0.375,
        {"pre1536": 0.5, "pre3072": 0.25},
    ),
}
STREAM_BASES = {
    "logical_seed": 53_000_000_000,
    "deck_stream_id": 54_000_000_000,
    "slot_stream_id": 55_000_000_000,
    "policy_stream_id": 56_000_000_000,
}
QUOTA_PER_FAMILY = 32
GAME_CAP_PER_FAMILY = 640
FROZEN_JOBS = 1
MAX_CHUNK_SIZE = 6
MAX_MOVES = 5000
STARTER_TILE = 1536
MINIMUM_NICE = 10
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
WALL_SECONDS_LIMIT = 12 * 3600
BYTE_LIMIT = 4 * 1024**3

SOURCE_PATHS = (
    Path("threes_rl/eval.py"),
    Path("threes_rl/expectimax.py"),
    Path("threes_rl/ntuple.py"),
    Path("threes_rl/action_prior.py"),
    Path("threes_rl/sim.py"),
    Path("threes_rl/record_replay.py"),
    Path("threes_rl/replay_provenance.py"),
    Path("threes_rl/restart_manifest.py"),
    Path("threes_rl/train_td.py"),
    base.__file__ and Path(base.__file__),
)


def canonical_json_hash(value: Any) -> str:
    return base.canonical_json_hash(value)


def incumbent_spec() -> str:
    lines = [
        line.strip()
        for line in INCUMBENT_PATH.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1:
        raise ValueError("Expected exactly one incumbent policy line")
    return lines[0]


def policy_slate() -> tuple[tuple[str, str], ...]:
    return (
        ("g2_transfer_corner2", "corner2"),
        ("g2_transfer_expectimax2", "expectimax2"),
        ("g2_transfer_phaseblend_incumbent", incumbent_spec()),
    )


def stream_ids(family_index: int, game_index: int) -> dict[str, int]:
    if not 0 <= family_index < 3:
        raise ValueError("family_index outside frozen slate")
    if not 0 <= game_index < GAME_CAP_PER_FAMILY:
        raise ValueError("game_index outside frozen family cap")
    offset = family_index * 1_000_000 + game_index
    return {name: base_value + offset for name, base_value in STREAM_BASES.items()}


def requested_stream_manifest() -> list[dict[str, Any]]:
    rows = []
    for family_index, (family, spec) in enumerate(policy_slate()):
        for game_index in range(GAME_CAP_PER_FAMILY):
            rows.append(
                {
                    "family_index": family_index,
                    "family": family,
                    "policy_spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
                    "game_index": game_index,
                    **stream_ids(family_index, game_index),
                }
            )
    return rows


def round_robin_rows(
    manifest: Iterable[dict[str, Any]],
    completed: set[tuple[str, int]],
    quotas: dict[str, int],
) -> list[list[dict[str, Any]]]:
    by_key = {
        (str(row["family"]), int(row["game_index"])): row for row in manifest
    }
    families = [family for family, _spec in policy_slate()]
    chunks: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    for game_index in range(GAME_CAP_PER_FAMILY):
        for family in families:
            key = (family, game_index)
            if quotas.get(family, 0) >= QUOTA_PER_FAMILY or key in completed:
                continue
            pending.append(by_key[key])
            if len(pending) == MAX_CHUNK_SIZE:
                chunks.append(pending)
                pending = []
    if pending:
        chunks.append(pending)
    return chunks


def stream_collision_audit(
    rows: list[dict[str, Any]], *, exclude_dir: Path
) -> dict[str, Any]:
    prior, sources = base.historical_collision_union(exclude_dir=exclude_dir)
    collisions: dict[str, list[int]] = {}
    for key in STREAM_BASES:
        prior_values = set(prior.get(key, set()))
        if key == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                prior_values.update(prior.get(alias, set()))
        values = {int(row[key]) for row in rows}
        collisions[key] = sorted(values.intersection(prior_values))
    flat = [int(row[key]) for row in rows for key in STREAM_BASES]
    checks = {
        "internal_unique": len(flat) == len(set(flat)),
        "historical_zero": not any(collisions.values()),
    }
    return {
        "historical_union": sources,
        "collisions": collisions,
        "checks": checks,
        "zero_collisions": all(checks.values()),
    }


def _directory_manifest(path: Path) -> dict[str, Any]:
    return base._directory_artifact_manifest(path)


def load_and_lock_policies() -> tuple[dict[str, Any], dict[str, Any]]:
    policies: dict[str, Any] = {}
    families = []
    for family, spec in policy_slate():
        policy = make_policy(spec)
        policies[family] = policy
        checkpoints = [
            _directory_manifest(path) for path in base._checkpoint_dirs(spec)
        ]
        families.append(
            {
                "family": family,
                "policy_spec": spec,
                "policy_spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
                "loaded_type": type(policy).__name__,
                "checkpoint_manifests": checkpoints,
            }
        )
    source_hashes = {
        str(path): sha256_path(path)
        for path in SOURCE_PATHS
        if path is not None and path.is_file()
    }
    lock = {
        "family_order": [family for family, _spec in policy_slate()],
        "families": families,
        "incumbent_file_sha256": sha256_path(INCUMBENT_PATH),
        "policy_source_hashes": source_hashes,
    }
    lock["policy_lock_sha256"] = canonical_json_hash(lock)
    return lock, policies


def _load_signature_panel() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_path(PILOT_V1_LOCK) != PILOT_V1_LOCK_SHA256:
        raise ValueError("Immutable pilot-v1 signature lock changed")
    lock = json.loads(PILOT_V1_LOCK.read_text())
    panel = lock["action_distinctness_panel"]
    embedded = panel.pop("panel_sha256")
    computed = canonical_json_hash(panel)
    panel["panel_sha256"] = embedded
    if computed != embedded:
        raise ValueError("Signature panel payload hash mismatch")
    return panel, {
        "path": str(PILOT_V1_LOCK),
        "file_sha256": PILOT_V1_LOCK_SHA256,
        "panel_sha256": panel["panel_sha256"],
    }


def action_signature_audit(
    policies: dict[str, Any], panel: dict[str, Any]
) -> dict[str, Any]:
    signatures: dict[str, list[int]] = {}
    tie_counts: dict[str, int] = {}
    for family, _spec in policy_slate():
        first = [
            base.deterministic_policy_action(policies[family], row["state"])
            for row in panel["records"]
        ]
        second = [
            base.deterministic_policy_action(policies[family], row["state"])
            for row in panel["records"]
        ]
        if first != second:
            raise ValueError(f"Nondeterministic action signature: {family}")
        signatures[family] = [int(row["action"]) for row in first]
        tie_counts[family] = sum(int(row["exact_tie_count"] > 1) for row in first)
    hashes = {
        family: canonical_json_hash(actions)
        for family, actions in signatures.items()
    }
    families = [family for family, _spec in policy_slate()]
    pairwise = []
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            strata = {}
            for stratum in ("pre1536", "pre3072"):
                indices = [
                    index
                    for index, row in enumerate(panel["records"])
                    if row["stratum"] == stratum
                ]
                strata[stratum] = sum(
                    signatures[left][index] != signatures[right][index]
                    for index in indices
                ) / len(indices)
            overall = sum(
                a != b
                for a, b in zip(signatures[left], signatures[right], strict=True)
            ) / len(panel["records"])
            expected = EXPECTED_PAIRWISE[(left, right)]
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "overall_disagreement": overall,
                    "stratum_disagreement": strata,
                    "exact_match": (overall, strata) == expected,
                }
            )
    checks = {
        "family_order_exact": families == list(EXPECTED_SIGNATURES),
        "signature_hashes_exact": hashes == EXPECTED_SIGNATURES,
        "pairwise_exact": all(row["exact_match"] for row in pairwise),
        "three_transitively_distinct_components": all(
            row["overall_disagreement"] >= 0.02
            and all(value > 0 for value in row["stratum_disagreement"].values())
            for row in pairwise
        ),
        "repeated_actions_exact": True,
        "no_timing_performed": True,
    }
    result = {
        "family_order": families,
        "signature_sha256": hashes,
        "tie_state_counts": tie_counts,
        "pairwise": pairwise,
        "checks": checks,
        "passes": all(checks.values()),
    }
    result["audit_sha256"] = canonical_json_hash(result)
    return result


def _fresh_reset_audit(replay: dict[str, Any], expected_seed: int) -> dict[str, Any]:
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        return {"passes": False, "reason": "missing_frames"}
    first = frames[0].get("state") if isinstance(frames[0], dict) else None
    if not isinstance(first, dict):
        return {"passes": False, "reason": "missing_first_state"}
    board = np.asarray(first.get("board"), dtype=np.int32)
    cycle = first.get("tile_cycle")
    working = board.copy() if board.shape == (4, 4) else np.empty((0, 0))
    starter_ok = board.shape == (4, 4) and int(board[0, 0]) == STARTER_TILE
    if starter_ok:
        working[0, 0] = 0
    smalls = [int(value) for value in working.reshape(-1) if int(value) > 0]
    checks = {
        "seed_exact": int(replay.get("seed", -1)) == expected_seed,
        "starter_exact": int(replay.get("starter_tile", -1)) == STARTER_TILE,
        "game_completed": bool(replay.get("game_over")),
        "first_move_zero": int(first.get("move_count", -1)) == 0,
        "starter_top_left": starter_ok,
        "eight_initial_smalls": len(smalls) == 8
        and all(value in (1, 2, 3) for value in smalls),
        "initial_preview_small": isinstance(first.get("preview"), dict)
        and first["preview"].get("kind") in ("blue", "red", "gray"),
        "initial_cycle_exact": isinstance(cycle, dict)
        and int(cycle.get("small_pos", -1)) == 8
        and int(cycle.get("small_seen_total", -1)) == 0
        and int(cycle.get("span_small_pos", -1)) == 0
        and not bool(cycle.get("large_pending")),
        "explicit_fresh_origin": replay.get("replay_origin") == ORIGIN_FRESH,
        "direct_root_exact": replay.get("root_origin") == ORIGIN_FRESH
        and int(replay.get("root_seed", -1)) == expected_seed
        and int(replay.get("root_frame_index", -1)) == 0
        and int(replay.get("root_move_count", -1)) == 0,
        "not_descended": replay.get("source_replay") is None
        and replay.get("start_case_id") is None
        and replay.get("human_import") is None
        and not bool(replay.get("synthetic")),
    }
    return {"checks": checks, "passes": all(checks.values())}


def extract_first_transfer_state(
    replay: dict[str, Any], *, family: str, expected_seed: int
) -> dict[str, Any] | None:
    provenance = _fresh_reset_audit(replay, expected_seed)
    if not provenance["passes"]:
        raise ValueError(f"Replay is not a completed direct fresh root: {provenance}")
    validator = ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_601,
        slot_stream_id=2_026_072_602,
        starter_tile=STARTER_TILE,
    )
    indexed = sorted(
        enumerate(replay["frames"]),
        key=lambda pair: (
            int(pair[1].get("index", pair[0])),
            pair[0],
        ),
    )
    for physical_index, frame in indexed:
        payload = frame.get("state") if isinstance(frame, dict) else None
        if not isinstance(payload, dict) or bool(payload.get("game_over")):
            continue
        board = np.asarray(payload.get("board"), dtype=np.int32)
        if board.shape != (4, 4):
            raise ValueError("Malformed replay board")
        if (
            max_tile_excluding_initial_starter(board, STARTER_TILE)
            != STARTER_TILE
        ):
            continue
        state = state_from_replay_payload(payload)
        legal = validator.legal_actions(state)
        names = [DIRECTION_NAMES[action] for action in legal]
        if payload.get("legal_actions") != names:
            raise ValueError("Stored legal actions disagree with simulator")
        if not legal:
            continue
        base._roundtrip_state(payload)
        state_sha1 = state_signature(payload, STARTER_TILE)
        return {
            "root_cluster": f"{family}:fresh:{expected_seed}:{STARTER_TILE}",
            "root_seed": expected_seed,
            "behavior_family": family,
            "scale": "pre3072_transfer",
            "target": 3072,
            "source_frame_index": int(frame.get("index", physical_index)),
            "source_physical_index": physical_index,
            "state_sha1": state_sha1,
            "state": payload,
        }
    return None


def split_reset_roundtrip_fixture() -> dict[str, Any]:
    ids = {"deck_stream_id": 57_999_999_901, "slot_stream_id": 57_999_999_902}
    left = ThreesSim.from_stream_ids(starter_tile=STARTER_TILE, **ids)
    right = ThreesSim.from_stream_ids(starter_tile=STARTER_TILE, **ids)
    left_state = left.reset()
    right_state = right.reset()
    left_payload = state_payload(left_state, left)
    right_payload = state_payload(right_state, right)
    base._roundtrip_state(left_payload)
    checks = {
        "split_reset_exact": left_payload == right_payload,
        "state_roundtrip_exact": True,
    }
    return {"stream_ids": ids, "checks": checks, "passes": all(checks.values())}


def _immutable_g2_audit() -> dict[str, Any]:
    files = {
        str(path): {
            "expected_sha256": expected,
            "actual_sha256": sha256_path(path),
        }
        for path, expected in G2_LOCKS.items()
    }
    files[str(CHARTER_PATH)] = {
        "expected_sha256": CHARTER_SHA256,
        "actual_sha256": sha256_path(CHARTER_PATH),
    }
    preflight = json.loads(G2_PREFLIGHT.read_text())
    checks = {
        "all_file_hashes_exact": all(
            row["expected_sha256"] == row["actual_sha256"]
            for row in files.values()
        ),
        "g2_decision_exact": preflight["decision"] == "HOLD_G2_DATA_OR_POWER",
        "g2_schema_exact": preflight["locks"]["feature_schema_sha256"]
        == G2_SCHEMA_SHA256,
        "representation_passed": preflight["representation_audit"]["passes"],
        "only_transfer_data_failed": not preflight["readiness_checks"][
            "transfer_min_roots"
        ]
        and not preflight["readiness_checks"]["transfer_min_families"],
    }
    return {"files": files, "checks": checks, "passes": all(checks.values())}


def _load_test_evidence() -> dict[str, Any]:
    evidence = json.loads(TEST_EVIDENCE_PATH.read_text())
    checks = {
        "implementation_hash": evidence["implementation_sha256"]
        == sha256_path(IMPLEMENTATION_PATH),
        "test_hash": evidence["test_sha256"] == sha256_path(TEST_PATH),
        "charter_hash": evidence["charter_sha256"] == CHARTER_SHA256,
        "tests_passed": bool(evidence["passes"]),
    }
    return {**evidence, "checks": checks, "passes": all(checks.values())}


def storage_runtime_projection() -> dict[str, Any]:
    inventory = json.loads(QD_STORAGE_INVENTORY.read_text())
    qd5 = json.loads(QD5_SEAL.read_text())
    maximum_replay_bytes = max(
        int(row["bytes"]) for row in inventory["rows"]
    )
    compact_row_bytes = 4096
    projected_bytes = math.ceil(
        1.25
        * (
            QUOTA_PER_FAMILY * 3 * (maximum_replay_bytes + 1024**2)
            + GAME_CAP_PER_FAMILY * 3 * compact_row_bytes
            + 16 * 1024**2
        )
    )
    seconds_per_game = float(qd5["runtime"]["active_runtime_seconds"]) / int(
        qd5["completeness"]["games"]
    )
    projected_seconds = seconds_per_game * GAME_CAP_PER_FAMILY * 3
    checks = {
        "projected_bytes_below_4_gib": projected_bytes < BYTE_LIMIT,
        "projected_seconds_below_12_hours": projected_seconds
        < WALL_SECONDS_LIMIT,
    }
    return {
        "maximum_existing_replay_bytes": maximum_replay_bytes,
        "retained_replay_count": QUOTA_PER_FAMILY * 3,
        "compact_completion_rows": GAME_CAP_PER_FAMILY * 3,
        "projected_bytes": projected_bytes,
        "projected_gib": projected_bytes / 1024**3,
        "sealed_seconds_per_game": seconds_per_game,
        "projected_active_seconds": projected_seconds,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _write_new_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _prepare_preflight_in_staging(
    staging_dir: Path, final_dir: Path
) -> dict[str, Any]:
    if base.current_nice() < MINIMUM_NICE:
        raise ValueError("G2 transfer preflight requires nice >=10")
    immutable = _immutable_g2_audit()
    tests = _load_test_evidence()
    policy_lock, policies = load_and_lock_policies()
    panel, panel_source = _load_signature_panel()
    signatures = action_signature_audit(policies, panel)
    streams = requested_stream_manifest()
    collision = stream_collision_audit(streams, exclude_dir=staging_dir)
    fixture = split_reset_roundtrip_fixture()
    projection = storage_runtime_projection()
    heavy = _heavy_process_audit()
    services = base.service_health()
    free_gib = shutil.disk_usage(staging_dir).free / 1024**3
    checks = {
        "fresh_output": not final_dir.exists(),
        "immutable_g2_exact": immutable["passes"],
        "tests_bound_and_passed": tests["passes"],
        "three_exact_loadable_policies": len(policies) == 3,
        "signatures_and_components_exact": signatures["passes"],
        "stream_manifest_1920_rows": len(streams) == 1920,
        "stream_collisions_zero": collision["zero_collisions"],
        "split_reset_roundtrip": fixture["passes"],
        "quota_and_cap_exact": QUOTA_PER_FAMILY == 32
        and GAME_CAP_PER_FAMILY == 640,
        "round_robin_chunk_bound": all(
            len(chunk) <= MAX_CHUNK_SIZE
            for chunk in round_robin_rows(streams, set(), {})
        ),
        "one_worker": FROZEN_JOBS == 1,
        "nice_at_least_10": base.current_nice() >= MINIMUM_NICE,
        "no_heavy_contention": heavy["passes"],
        "storage_and_runtime_feasible": projection["passes"],
        "free_disk_above_120_gib": free_gib > TARGET_FREE_GIB,
        "services_dashboard_top_three": services["passes"],
        "active_human_session_content_not_read": True,
        "streams_not_consumed": True,
    }
    if not all(checks.values()):
        cost_only = not projection["passes"] or free_gib <= TARGET_FREE_GIB
        decision = (
            "HOLD_G2_ACQUISITION_COST_OR_YIELD"
            if cost_only
            else "KILL_G2_ACQUISITION_PREFLIGHT"
        )
    else:
        decision = "READY_G2_FRESH_TRANSFER_ACQUISITION"
    lock = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "execution_authorized": False,
        "bound_out_dir": str(final_dir),
        "identity": {
            "charter_sha256": CHARTER_SHA256,
            "implementation_sha256": sha256_path(IMPLEMENTATION_PATH),
            "test_sha256": sha256_path(TEST_PATH),
            "test_evidence_sha256": sha256_path(TEST_EVIDENCE_PATH),
        },
        "immutable_g2_audit": immutable,
        "test_evidence": tests,
        "policy_lock": policy_lock,
        "panel_source": panel_source,
        "action_signature_audit": signatures,
        "quota_per_family": QUOTA_PER_FAMILY,
        "game_cap_per_family": GAME_CAP_PER_FAMILY,
        "family_order": [family for family, _spec in policy_slate()],
        "stream_bases": STREAM_BASES,
        "stream_rows": streams,
        "stream_manifest_sha256": canonical_json_hash(streams),
        "historical_stream_collision_audit": collision,
        "split_reset_roundtrip_fixture": fixture,
        "storage_runtime_projection": projection,
        "frozen_jobs": FROZEN_JOBS,
        "maximum_chunk_size": MAX_CHUNK_SIZE,
        "required_minimum_nice": MINIMUM_NICE,
        "preflight_nice": base.current_nice(),
        "active_wall_seconds_limit": WALL_SECONDS_LIMIT,
        "byte_limit": BYTE_LIMIT,
        "minimum_free_gib": MIN_FREE_GIB,
        "target_free_gib": TARGET_FREE_GIB,
        "heavy_process_audit": heavy,
        "free_gib": free_gib,
        "service_health": services,
        "checks": checks,
        "zero_work": {
            "games": 0,
            "streams_consumed": 0,
            "labels": 0,
            "rollouts": 0,
            "h10_h20_h40_outcomes": 0,
            "models": 0,
            "score_or_policy_outcomes_inspected": False,
            "continuations": 0,
            "dashboard_changes": 0,
            "incumbent_changed": False,
        },
    }
    lock["preflight_payload_sha256"] = canonical_json_hash(lock)
    _write_new_json_atomic(staging_dir / "preflight_lock.json", lock)
    return lock


def prepare_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    final_dir = out_dir.resolve()
    if final_dir != OUTPUT_DIR.resolve():
        raise ValueError(f"Output must be {OUTPUT_DIR.resolve()}")
    if final_dir.exists():
        raise FileExistsError(f"Output already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.with_name(f"{final_dir.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Staging already exists: {staging}")
    staging.mkdir()
    try:
        lock = _prepare_preflight_in_staging(staging, final_dir)
        os.replace(staging, final_dir)
        return lock
    except Exception as error:
        failure = {
            "version": VERSION,
            "decision": "KILL_G2_ACQUISITION_PREFLIGHT",
            "error_type": type(error).__name__,
            "error": str(error),
            "bound_out_dir": str(final_dir),
            "zero_games": 0,
            "zero_streams": 0,
            "zero_labels": 0,
            "zero_outcomes": 0,
        }
        _write_new_json_atomic(staging / "PREFLIGHT_FAILURE.json", failure)
        raise


def _validate_preflight(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text())
    embedded = lock.pop("preflight_payload_sha256")
    if canonical_json_hash(lock) != embedded:
        raise ValueError("Preflight payload mismatch")
    lock["preflight_payload_sha256"] = embedded
    if lock["version"] != VERSION:
        raise ValueError("Preflight version mismatch")
    if lock["decision"] != "READY_G2_FRESH_TRANSFER_ACQUISITION":
        raise ValueError("Preflight is not READY")
    if str(path.parent.resolve()) != lock["bound_out_dir"]:
        raise ValueError("Preflight directory binding mismatch")
    return lock


def _directory_bytes(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _completion_row(
    *,
    output: Any,
    stream_row: dict[str, Any],
    family: str,
    replay_dir: Path,
    retained_count: int,
) -> dict[str, Any]:
    replay = output.replay
    if replay is None:
        raise RuntimeError("Replay capture missing")
    seed = int(stream_row["logical_seed"])
    replay.update(
        direct_root_fields(
            origin=ORIGIN_FRESH,
            seed=seed,
            policy=family,
            first_score=None,
        )
    )
    replay["behavior_family"] = family
    replay["dashboard_eligible"] = False
    candidate = extract_first_transfer_state(
        replay, family=family, expected_seed=seed
    )
    retain = candidate is not None and retained_count < QUOTA_PER_FAMILY
    replay_path = None
    state_path = None
    compact_candidate = None
    if retain:
        replay_path = replay_dir / f"{family}_{int(stream_row['game_index']):04d}.json"
        replay_path.write_text(
            json.dumps(replay, sort_keys=True, separators=(",", ":")) + "\n"
        )
        candidate["source_replay"] = str(replay_path)
        candidate["source_replay_sha256"] = sha256_path(replay_path)
        state_path = replay_path.with_suffix(".state.json")
        state_path.write_text(
            json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n"
        )
        compact_candidate = {
            key: value
            for key, value in candidate.items()
            if key != "state"
        }
        compact_candidate["source_state"] = str(state_path)
        compact_candidate["source_state_sha256"] = sha256_path(state_path)
    return {
        "family": family,
        "family_index": int(stream_row["family_index"]),
        "game_index": int(stream_row["game_index"]),
        "logical_seed": seed,
        "deck_stream_id": int(stream_row["deck_stream_id"]),
        "slot_stream_id": int(stream_row["slot_stream_id"]),
        "policy_stream_id": int(stream_row["policy_stream_id"]),
        "completed": bool(replay.get("game_over")),
        "qualified": candidate is not None,
        "retained": retain,
        "source_replay": str(replay_path) if replay_path else None,
        "source_state": str(state_path) if state_path else None,
        "candidate": compact_candidate,
        "dashboard_eligible": False,
    }


def run_acquisition(
    *, out_dir: Path, preflight_lock: Path, jobs: int
) -> dict[str, Any]:
    """Execute only after a separate authorization; preflight never calls this."""
    lock = _validate_preflight(preflight_lock)
    if out_dir.resolve() != Path(lock["bound_out_dir"]):
        raise ValueError("Output binding mismatch")
    if jobs != FROZEN_JOBS:
        raise ValueError("jobs differs from frozen value")
    if base.current_nice() < MINIMUM_NICE:
        raise ValueError("nice priority below frozen minimum")
    policy_lock, _policies = load_and_lock_policies()
    if policy_lock["policy_lock_sha256"] != lock["policy_lock"]["policy_lock_sha256"]:
        raise ValueError("Policy artifacts changed after preflight")
    collision = stream_collision_audit(lock["stream_rows"], exclude_dir=out_dir)
    if not collision["zero_collisions"]:
        raise ValueError("Historical stream collision appeared")
    if _heavy_process_audit()["passes"] is False:
        raise ValueError("Competing heavy process")
    completed_path = out_dir / "completion_rows.jsonl"
    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if completed_path.exists():
        for line in completed_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[(row["family"], int(row["game_index"]))] = row
    quotas = {
        family: sum(
            int(row["retained"])
            for row in existing.values()
            if row["family"] == family
        )
        for family, _spec in policy_slate()
    }
    runtime_path = out_dir / "runtime_state.json"
    runtime = (
        json.loads(runtime_path.read_text())
        if runtime_path.exists()
        else {"active_seconds": 0.0, "chunks": 0}
    )
    replay_dir = out_dir / "qualifying_sources"
    replay_dir.mkdir(exist_ok=True)
    specs = dict(policy_slate())
    for chunk in round_robin_rows(lock["stream_rows"], set(existing), quotas):
        if all(value >= QUOTA_PER_FAMILY for value in quotas.values()):
            break
        if runtime["active_seconds"] >= WALL_SECONDS_LIMIT:
            break
        if _directory_bytes(out_dir) >= BYTE_LIMIT:
            break
        if shutil.disk_usage(out_dir).free / 1024**3 < MIN_FREE_GIB:
            break
        if not base.service_health()["passes"] or not _heavy_process_audit()["passes"]:
            break
        started = time.perf_counter()
        for row in chunk:
            family = str(row["family"])
            if quotas[family] >= QUOTA_PER_FAMILY:
                continue
            output = next(
                iter_eval_job_outputs(
                    policy=make_policy(specs[family]),
                    policy_name=family,
                    eval_jobs=[
                        EvalJob(
                            index=0,
                            seed=int(row["logical_seed"]),
                            starter_tile=STARTER_TILE,
                            stream_ids=EvalStreamIds(
                                deck_stream_id=int(row["deck_stream_id"]),
                                slot_stream_id=int(row["slot_stream_id"]),
                                policy_stream_id=int(row["policy_stream_id"]),
                            ),
                        )
                    ],
                    max_moves=MAX_MOVES,
                    capture_replay=True,
                    jobs=1,
                )
            )
            compact = _completion_row(
                output=output,
                stream_row=row,
                family=family,
                replay_dir=replay_dir,
                retained_count=quotas[family],
            )
            with completed_path.open("a") as handle:
                handle.write(json.dumps(compact, sort_keys=True) + "\n")
            existing[(family, int(row["game_index"]))] = compact
            quotas[family] += int(compact["retained"])
        runtime["active_seconds"] += time.perf_counter() - started
        runtime["chunks"] += 1
        write_json(runtime_path, runtime)
    decision = (
        "READY_G2_TRANSFER_ROOTS"
        if all(value == QUOTA_PER_FAMILY for value in quotas.values())
        else "HOLD_G2_ACQUISITION_COST_OR_YIELD"
    )
    summary = {
        "version": VERSION,
        "decision": decision,
        "quotas": quotas,
        "completed_games": len(existing),
        "runtime": runtime,
        "labels": 0,
        "models": 0,
        "outcomes_inspected": False,
        "dashboard_changed": False,
    }
    summary["summary_payload_sha256"] = canonical_json_hash(summary)
    write_json(out_dir / "acquisition_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-preflight")
    prepare.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    run = commands.add_parser("run-acquisition")
    run.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    run.add_argument("--preflight-lock", type=Path, required=True)
    run.add_argument("--jobs", type=int, default=FROZEN_JOBS)
    args = parser.parse_args()
    if args.command == "prepare-preflight":
        payload = prepare_preflight(args.out_dir)
    else:
        payload = run_acquisition(
            out_dir=args.out_dir,
            preflight_lock=args.preflight_lock,
            jobs=args.jobs,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

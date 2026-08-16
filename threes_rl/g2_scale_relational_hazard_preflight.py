"""Outcome-free corpus, feature-schema, and power preflight for G2."""

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
from typing import Any

import numpy as np

from threes_rl.eval import max_tile_excluding_initial_starter
from threes_rl.g1r_acquire import service_health
from threes_rl.g2_scale_relational_hazard import (
    FEATURE_NAMES,
    FEATURE_WIDTH,
    HORIZONS,
    TARGETS,
    VERSION as FEATURE_VERSION,
    feature_vector,
    schema_manifest,
    schema_sha256,
)
from threes_rl.r15a_context_inventory import (
    coalesced_behavior_family,
    deterministic_key,
)
from threes_rl.replay_provenance import GENUINE_ROOT_ORIGINS, replay_provenance
from threes_rl.restart_manifest import canonical_ancestry_id, state_signature
from threes_rl.sim import (
    DIRECTION_NAMES,
    LEFT,
    Preview,
    SimState,
    ThreesSim,
)
from threes_rl.train_td import state_from_replay_payload


VERSION = "g2_scale_equivariant_relational_hazard_preflight_v1"
PROPOSAL_PATH = Path(
    "threes_rl/G2_SCALE_EQUIVARIANT_RELATIONAL_HAZARD_PROPOSAL.md"
)
PROPOSAL_SHA256 = "43b413c1a8145a25750009cc3048bbda6127a44cfccbf72c7d1710e1e6027099"
FEATURE_SOURCE_PATH = Path("threes_rl/g2_scale_relational_hazard.py")
PREFLIGHT_SOURCE_PATH = Path(
    "threes_rl/g2_scale_relational_hazard_preflight.py"
)
TEST_SOURCE_PATH = Path("tests/test_rl_g2_scale_relational_hazard.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/"
    "g2_scale_equivariant_relational_hazard_test_evidence.json"
)
A2_INVENTORY_PATH = Path(
    "threes_rl/runs/forensics/r15a_context_a1/"
    "r15a_natural_state_inventory_a1_20260711.json"
)
S3_PROVENANCE_PATH = Path(
    "threes_rl/runs/forensics/s3_full_policy/S3_PROVENANCE_SEAL_V2.json"
)
QD5_SUMMARY_PATH = Path(
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v2_qd5/pilot_summary.json"
)
DEFAULT_OUT_DIR = Path(
    "threes_rl/runs/forensics/g2_scale_equivariant_relational_hazard"
)

SCALES = (
    ("pre768", 384, 768),
    ("pre1536", 768, 1536),
    ("pre3072_transfer", 1536, 3072),
)
SCALE_BY_BUILT_MAX = {built: (name, target) for name, built, target in SCALES}
TRAIN_SCALES = ("pre768", "pre1536")
TRANSFER_SCALE = "pre3072_transfer"

TRAIN_MIN_ROOTS = 240
TRAIN_SCALE_MIN = 100
TRAIN_MIN_FAMILIES = 5
DEV_MIN_ROOTS = 60
DEV_SCALE_MIN = 24
DEV_MIN_FAMILIES = 3
TRANSFER_MIN_ROOTS = 96
TRANSFER_MIN_FAMILIES = 3
TRANSFER_MAX_FAMILY_SHARE = 0.50
FAMILY_WEIGHT_CAP = 0.40
FREE_DISK_MIN_BYTES = 100 * 1024**3

POWER_BASE_RATE = 0.04
POWER_ROOT_RHO = 0.15
POWER_REPEATS = 8
POWER_ACTIVITY = 0.30
POWER_TARGET_OR = 1.75
POWER_REQUIRED = 0.80
POWER_DESIGNS = (96, 128, 192, 256, 384, 512)
POWER_OR_GRID = (1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00)
POWER_DRAWS = 10_000
Z_975 = 1.959963984540054


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def _compact_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Retain exact simulator state without score or recorded-action fields."""
    return {
        "board": payload.get("board"),
        "preview": payload.get("preview"),
        "tile_cycle": payload.get("tile_cycle"),
        "move_count": payload.get("move_count"),
        "game_over": payload.get("game_over"),
        "legal_actions": payload.get("legal_actions"),
        "legal_mask": payload.get("legal_mask"),
    }


def _source_manifest(source_paths: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for text in sorted(set(source_paths)):
        path = Path(text)
        if not path.is_file():
            rows.append({"path": text, "exists": False})
            continue
        stat = path.stat()
        rows.append(
            {
                "path": text,
                "exists": True,
                "bytes": int(stat.st_size),
                "sha256": sha256_path(path),
            }
        )
    return {
        "rows": rows,
        "source_count": len(rows),
        "existing_count": sum(bool(row["exists"]) for row in rows),
        "missing_count": sum(not bool(row["exists"]) for row in rows),
        "manifest_sha256": canonical_sha256(rows),
    }


def _load_source_headers(
    source_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    headers: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in source_manifest["rows"]:
        counts["source_rows"] += 1
        if not row["exists"]:
            counts["missing_source"] += 1
            continue
        path = Path(row["path"])
        try:
            replay = _json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            counts["invalid_json"] += 1
            continue
        provenance = replay_provenance(replay, path)
        if (
            provenance.get("replay_origin") not in GENUINE_ROOT_ORIGINS
            or provenance.get("root_origin") not in GENUINE_ROOT_ORIGINS
            or not provenance.get("replay_reset_invariant")
        ):
            counts["non_natural"] += 1
            continue
        family = coalesced_behavior_family(replay, path)
        if family == "human_observed":
            counts["human_excluded"] += 1
            continue
        frames = replay.get("frames")
        if not isinstance(frames, list) or not frames:
            counts["missing_frames"] += 1
            continue
        final_frame = frames[-1]
        final_state = (
            final_frame.get("state") if isinstance(final_frame, dict) else None
        )
        if not isinstance(final_state, dict) or not bool(final_state.get("game_over")):
            counts["incomplete_replay"] += 1
            continue
        root = canonical_ancestry_id(replay, path)
        headers.append(
            {
                "root_cluster": root,
                "behavior_family": family,
                "path": str(path),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "starter_tile": replay.get("starter_tile", 1536),
                "replay": replay,
            }
        )
        counts["valid_completed_natural_machine_replay"] += 1
    return headers, counts


def _representative_headers(
    headers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for header in headers:
        by_root[str(header["root_cluster"])].append(header)
    selected = [
        min(
            rows,
            key=lambda row: deterministic_key(
                "G2-source-representative-v1",
                row["root_cluster"],
                row["behavior_family"],
                row["sha256"],
                row["path"],
            ),
        )
        for rows in by_root.values()
    ]
    return sorted(selected, key=lambda row: str(row["root_cluster"])), (
        len(headers) - len(selected)
    )


def _candidate_records(
    representatives: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    validator = ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_501,
        slot_stream_id=2_026_072_502,
        starter_tile=1536,
    )
    for header in representatives:
        replay = header["replay"]
        starter_value = header["starter_tile"]
        starter_tile = None if starter_value is None else int(starter_value)
        by_scale: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fallback_index, frame in enumerate(replay["frames"]):
            payload = frame.get("state") if isinstance(frame, dict) else None
            if not isinstance(payload, dict) or bool(payload.get("game_over")):
                continue
            try:
                board = np.asarray(payload.get("board"), dtype=np.int32)
            except (TypeError, ValueError):
                counts["invalid_board"] += 1
                continue
            if board.shape != (4, 4):
                counts["invalid_board"] += 1
                continue
            built_max = int(
                max_tile_excluding_initial_starter(board, starter_tile)
            )
            scale = SCALE_BY_BUILT_MAX.get(built_max)
            if scale is None:
                continue
            scale_name, target = scale
            try:
                state = state_from_replay_payload(payload)
                legal = validator.legal_actions(state)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                counts["state_restore_failure"] += 1
                failures.append(
                    {
                        "root_cluster": header["root_cluster"],
                        "path": header["path"],
                        "frame": fallback_index,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            expected_names = [DIRECTION_NAMES[action] for action in legal]
            if payload.get("legal_actions") != expected_names:
                counts["legal_action_mismatch"] += 1
                failures.append(
                    {
                        "root_cluster": header["root_cluster"],
                        "path": header["path"],
                        "frame": fallback_index,
                        "error": "legal_action_mismatch",
                    }
                )
                continue
            if not legal:
                continue
            frame_index = int(frame.get("index", fallback_index))
            state_hash = state_signature(payload, starter_tile)
            by_scale[scale_name].append(
                {
                    "record_id": deterministic_key(
                        "G2-record-v1",
                        header["root_cluster"],
                        scale_name,
                        frame_index,
                        state_hash,
                    )[:24],
                    "root_cluster": header["root_cluster"],
                    "behavior_family": header["behavior_family"],
                    "scale": scale_name,
                    "target": target,
                    "built_max": built_max,
                    "source_replay": header["path"],
                    "source_replay_sha256": header["sha256"],
                    "source_frame_index": frame_index,
                    "state_sha1": state_hash,
                    "starter_tile": starter_tile,
                    "state": _compact_state(payload),
                }
            )
            counts[f"eligible_frames:{scale_name}"] += 1
        for scale_name, rows in by_scale.items():
            records.append(
                min(
                    rows,
                    key=lambda row: deterministic_key(
                        "G2-state-choice-v1",
                        row["root_cluster"],
                        scale_name,
                        row["source_frame_index"],
                        row["state_sha1"],
                    ),
                )
            )
        if by_scale:
            counts["roots_with_any_scale"] += 1
    return (
        sorted(records, key=lambda row: (row["root_cluster"], row["scale"])),
        counts,
        failures,
    )


def _historical_sets(a2: dict[str, Any]) -> dict[str, Any]:
    s3 = _json(S3_PROVENANCE_PATH)
    qd5 = _json(QD5_SUMMARY_PATH)
    sets = {
        "S3_historical_exclusion_union": set(s3["excluded_roots"]["roots"]),
        "S3_sealed_surviving_roots": set(s3["surviving_inventory"]["roots"]),
        "A2_selected_or_labeled_roots": {
            str(row["root_cluster"]) for row in a2["selected_records"]
        },
        "QD5_sealed_pilot_roots": {
            str(row["root_cluster"]) for row in qd5["root_capped_candidates"]
        },
    }
    return {
        "sets": sets,
        "counts": {name: len(roots) for name, roots in sets.items()},
        "hashes": {
            name: hashlib.sha256(
                "\n".join(sorted(roots)).encode("utf-8")
            ).hexdigest()
            for name, roots in sets.items()
        },
    }


def _partition_records(
    records: list[dict[str, Any]],
    historical: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sets = historical["sets"]
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_root[str(record["root_cluster"])].append(record)

    transfer_roots: set[str] = set()
    diagnostic_roots: set[str] = set(sets["S3_sealed_surviving_roots"])
    prior_transfer_forbidden = (
        set(sets["S3_historical_exclusion_union"])
        | set(sets["S3_sealed_surviving_roots"])
        | set(sets["A2_selected_or_labeled_roots"])
        | set(sets["QD5_sealed_pilot_roots"])
    )
    for root, rows in by_root.items():
        has_transfer = any(row["scale"] == TRANSFER_SCALE for row in rows)
        if has_transfer and root not in prior_transfer_forbidden:
            transfer_roots.add(root)

    earlier_roots_by_family: dict[str, list[str]] = defaultdict(list)
    for root, rows in by_root.items():
        if root in diagnostic_roots or root in transfer_roots:
            continue
        earlier = [row for row in rows if row["scale"] in TRAIN_SCALES]
        if not earlier:
            continue
        family = str(earlier[0]["behavior_family"])
        earlier_roots_by_family[family].append(root)

    dev_roots: set[str] = set()
    for family, roots in earlier_roots_by_family.items():
        ordered = sorted(
            set(roots),
            key=lambda root: deterministic_key("G2-dev-v1", family, root),
        )
        dev_roots.update(ordered[: int(math.floor(0.20 * len(ordered)))])

    partitioned: list[dict[str, Any]] = []
    withheld_earlier_from_transfer = 0
    for record in records:
        row = dict(record)
        root = str(row["root_cluster"])
        if root in diagnostic_roots:
            row["partition"] = "diagnostic_only"
        elif root in transfer_roots:
            if row["scale"] == TRANSFER_SCALE:
                row["partition"] = "untouched_transfer"
            else:
                row["partition"] = "withheld_earlier_from_transfer"
                withheld_earlier_from_transfer += 1
        elif row["scale"] == TRANSFER_SCALE:
            row["partition"] = "diagnostic_prior_overlap_transfer"
        elif root in dev_roots:
            row["partition"] = "development"
        else:
            row["partition"] = "train"
        partitioned.append(row)

    root_partitions: dict[str, set[str]] = defaultdict(set)
    for row in partitioned:
        if row["partition"] in {"train", "development", "untouched_transfer"}:
            root_partitions[str(row["root_cluster"])].add(str(row["partition"]))
    cross_partition = {
        root: sorted(parts)
        for root, parts in root_partitions.items()
        if len(parts) > 1
    }
    return partitioned, {
        "transfer_forbidden_union_count": len(prior_transfer_forbidden),
        "transfer_forbidden_union_sha256": hashlib.sha256(
            "\n".join(sorted(prior_transfer_forbidden)).encode("utf-8")
        ).hexdigest(),
        "clean_transfer_roots": len(transfer_roots),
        "diagnostic_sealed_roots": len(diagnostic_roots.intersection(by_root)),
        "withheld_earlier_records_from_transfer_roots": withheld_earlier_from_transfer,
        "cross_partition_roots": cross_partition,
    }


def _root_summary(records: list[dict[str, Any]], partition: str) -> dict[str, Any]:
    rows = [row for row in records if row["partition"] == partition]
    roots = {str(row["root_cluster"]) for row in rows}
    families = Counter(
        {
            family: len(
                {
                    str(row["root_cluster"])
                    for row in rows
                    if row["behavior_family"] == family
                }
            )
            for family in {str(row["behavior_family"]) for row in rows}
        }
    )
    scale_roots = {
        scale: len(
            {
                str(row["root_cluster"])
                for row in rows
                if row["scale"] == scale
            }
        )
        for scale, _built, _target in SCALES
    }
    raw_max_share = (
        max(families.values(), default=0) / len(roots) if roots else 0.0
    )
    if families:
        family_total_weight = 1.0 / len(families)
        root_weights = [
            family_total_weight / count
            for count in families.values()
            for _ in range(count)
        ]
        effective_roots = 1.0 / sum(weight * weight for weight in root_weights)
        effective_family_shares = {
            family: family_total_weight for family in sorted(families)
        }
    else:
        effective_roots = 0.0
        effective_family_shares = {}
    return {
        "records": len(rows),
        "unique_roots": len(roots),
        "roots_by_scale": scale_roots,
        "roots_by_family": dict(sorted(families.items())),
        "family_count": len(families),
        "raw_max_family_share": raw_max_share,
        "effective_family_shares": effective_family_shares,
        "effective_ancestry_count": effective_roots,
    }


def _joint_scale_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_root: dict[str, set[str]] = defaultdict(set)
    for row in records:
        by_root[str(row["root_cluster"])].add(str(row["scale"]))
    combinations_count = Counter(
        "+".join(sorted(scales)) for scales in by_root.values()
    )
    pairwise = {}
    scale_names = [name for name, _built, _target in SCALES]
    for left_index, left in enumerate(scale_names):
        left_roots = {root for root, scales in by_root.items() if left in scales}
        for right in scale_names[left_index + 1 :]:
            right_roots = {root for root, scales in by_root.items() if right in scales}
            pairwise[f"{left}&{right}"] = {
                "joint": len(left_roots & right_roots),
                "left_total": len(left_roots),
                "right_total": len(right_roots),
                "p_right_given_left": (
                    len(left_roots & right_roots) / len(left_roots)
                    if left_roots
                    else 0.0
                ),
                "p_left_given_right": (
                    len(left_roots & right_roots) / len(right_roots)
                    if right_roots
                    else 0.0
                ),
            }
    return {
        "root_scale_combinations": dict(sorted(combinations_count.items())),
        "pairwise": pairwise,
    }


def _state_for_record(record: dict[str, Any]) -> SimState:
    return state_from_replay_payload(record["state"])


def _feature_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    minima = np.full(FEATURE_WIDTH, np.inf, dtype=np.float64)
    maxima = np.full(FEATURE_WIDTH, -np.inf, dtype=np.float64)
    nonfinite = 0
    rows = 0
    by_partition_scale: Counter[str] = Counter()
    for record in records:
        state = _state_for_record(record)
        starter_tile = record["starter_tile"]
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=2_026_072_503,
            slot_stream_id=2_026_072_504,
            starter_tile=starter_tile,
        )
        for action in sim.legal_actions(state):
            for horizon in HORIZONS:
                vector = feature_vector(
                    state,
                    sim,
                    action,
                    target=int(record["target"]),
                    horizon=horizon,
                    starter_tile=starter_tile,
                )
                rows += 1
                nonfinite += int(np.count_nonzero(~np.isfinite(vector)))
                minima = np.minimum(minima, vector)
                maxima = np.maximum(maxima, vector)
                digest.update(
                    _canonical_json(
                        {
                            "record_id": record["record_id"],
                            "partition": record["partition"],
                            "scale": record["scale"],
                            "action": action,
                            "horizon": horizon,
                            "features": vector.tolist(),
                        }
                    )
                )
                by_partition_scale[
                    f"{record['partition']}:{record['scale']}"
                ] += 1
    return {
        "feature_rows": rows,
        "nonfinite_values": nonfinite,
        "out_of_bounds_values": int(
            np.count_nonzero(minima < -1e-12) + np.count_nonzero(maxima > 1 + 1e-12)
        ),
        "minimum_by_column": {
            name: float(minima[index]) if rows else None
            for index, name in enumerate(FEATURE_NAMES)
        },
        "maximum_by_column": {
            name: float(maxima[index]) if rows else None
            for index, name in enumerate(FEATURE_NAMES)
        },
        "rows_by_partition_scale": dict(sorted(by_partition_scale.items())),
        "feature_rows_sha256": digest.hexdigest(),
    }


def _copy_state(state: SimState, **updates: Any) -> SimState:
    values = {
        "board": state.board.copy(),
        "preview": state.preview,
        "small_counts": state.small_counts.copy(),
        "small_pos": state.small_pos,
        "small_seen_total": state.small_seen_total,
        "span_small_pos": state.span_small_pos,
        "large_pending": state.large_pending,
        "max_tile": state.max_tile,
        "move_count": state.move_count,
        "game_over": state.game_over,
    }
    values.update(updates)
    return SimState(**values)


def representation_self_audit() -> dict[str, Any]:
    board = np.asarray(
        [
            [1536, 384, 192, 24],
            [0, 384, 96, 12],
            [3, 48, 0, 6],
            [1, 0, 2, 0],
        ],
        dtype=np.int32,
    )
    state = SimState(
        board=board,
        preview=Preview("bonus", None, (24, 48, 96)),
        small_counts={"red": 2, "blue": 3, "gray": 4},
        small_pos=3,
        small_seen_total=55,
        span_small_pos=7,
        large_pending=True,
        max_tile=1536,
        move_count=120,
        game_over=False,
    )
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_505,
        slot_stream_id=2_026_072_506,
        starter_tile=1536,
    )
    before_board = state.board.copy()
    before_counts = state.small_counts.copy()
    deck_before = json.dumps(sim.deck_rng.bit_generator.state, sort_keys=True)
    slot_before = json.dumps(sim.slot_rng.bit_generator.state, sort_keys=True)
    base = feature_vector(
        state,
        sim,
        LEFT,
        target=768,
        horizon=40,
        starter_tile=1536,
    )

    transposed_state = _copy_state(state, board=state.board.T.copy())
    transposed = feature_vector(
        transposed_state,
        sim,
        0,
        target=768,
        horizon=40,
        starter_tile=1536,
    )

    scaled_board = state.board.copy()
    for row in range(4):
        for column in range(4):
            if (row, column) == (0, 0):
                continue
            if int(scaled_board[row, column]) >= 3:
                scaled_board[row, column] *= 2
    scaled_state = _copy_state(
        state,
        board=scaled_board,
        preview=Preview("bonus", None, (48, 96, 192)),
    )
    scaled = feature_vector(
        scaled_state,
        sim,
        LEFT,
        target=1536,
        horizon=40,
        starter_tile=1536,
    )
    schema = schema_manifest()
    checks = {
        "proposal_hash_exact": sha256_path(PROPOSAL_PATH) == PROPOSAL_SHA256,
        "schema_width_64": schema["width"] == FEATURE_WIDTH == 64,
        "schema_names_unique": len(set(FEATURE_NAMES)) == 64,
        "feature_width_64": base.shape == (64,),
        "features_finite": bool(np.all(np.isfinite(base))),
        "features_bounded": bool(np.all((base >= 0.0) & (base <= 1.0))),
        "orientation_equivariant": bool(np.array_equal(base, transposed)),
        "scale_equivariant": bool(np.array_equal(base, scaled)),
        "input_board_unmutated": bool(np.array_equal(state.board, before_board)),
        "input_cycle_unmutated": state.small_counts == before_counts,
        "deck_rng_unmutated": (
            json.dumps(sim.deck_rng.bit_generator.state, sort_keys=True)
            == deck_before
        ),
        "slot_rng_unmutated": (
            json.dumps(sim.slot_rng.bit_generator.state, sort_keys=True)
            == slot_before
        ),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "schema_sha256": schema_sha256(),
        "fixture_feature_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
    }


def _beta_parameters(mean: float, rho: float) -> tuple[float, float]:
    concentration = 1.0 / rho - 1.0
    return mean * concentration, (1.0 - mean) * concentration


def _odds_shift(probability: np.ndarray, odds_ratio: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    odds = clipped / (1.0 - clipped)
    shifted = odds_ratio * odds
    return shifted / (1.0 + shifted)


def _active_or_for_policy_or(policy_or: float, activity: float) -> float:
    alpha, beta = _beta_parameters(POWER_BASE_RATE, POWER_ROOT_RHO)
    calibration_rng = np.random.default_rng(2_026_072_507)
    probabilities = calibration_rng.beta(alpha, beta, size=200_000)

    def realized(active_or: float) -> float:
        active_mean = float(np.mean(_odds_shift(probabilities, active_or)))
        treatment = (1.0 - activity) * POWER_BASE_RATE + activity * active_mean
        control_odds = POWER_BASE_RATE / (1.0 - POWER_BASE_RATE)
        treatment_odds = treatment / (1.0 - treatment)
        return treatment_odds / control_odds

    low = 1.0
    high = 2.0
    while realized(high) < policy_or:
        high *= 2.0
        if high > 1_000_000:
            raise ValueError("Cannot solve G2 active-root odds ratio")
    for _ in range(80):
        middle = math.sqrt(low * high)
        if realized(middle) >= policy_or:
            high = middle
        else:
            low = middle
    return high


def simulate_power(
    roots: int,
    policy_or: float,
    *,
    draws: int = POWER_DRAWS,
) -> dict[str, Any]:
    active = int(math.floor(roots * POWER_ACTIVITY))
    realized_activity = active / roots
    active_or = _active_or_for_policy_or(policy_or, realized_activity)
    alpha, beta = _beta_parameters(POWER_BASE_RATE, POWER_ROOT_RHO)
    rng = np.random.default_rng(
        2_026_072_508 + roots * 100 + int(round(policy_or * 100))
    )
    passes = 0
    useful = 0
    estimates: list[float] = []
    chunk_size = 100
    for start in range(0, draws, chunk_size):
        chunk = min(chunk_size, draws - start)
        probability = rng.beta(alpha, beta, size=(chunk, roots))
        control = rng.binomial(POWER_REPEATS, probability) / POWER_REPEATS
        treatment = control.copy()
        shifted = _odds_shift(probability[:, :active], active_or)
        treatment[:, :active] = (
            rng.binomial(POWER_REPEATS, shifted) / POWER_REPEATS
        )
        differences = treatment - control
        mean_difference = np.mean(differences, axis=1)
        standard_error = np.std(differences, axis=1, ddof=1) / math.sqrt(roots)
        lower = mean_difference - Z_975 * standard_error
        control_mean = np.clip(
            np.mean(control, axis=1),
            0.5 / (roots * POWER_REPEATS + 1),
            1.0 - 0.5 / (roots * POWER_REPEATS + 1),
        )
        treatment_mean = np.clip(
            np.mean(treatment, axis=1),
            0.5 / (roots * POWER_REPEATS + 1),
            1.0 - 0.5 / (roots * POWER_REPEATS + 1),
        )
        estimated_or = (
            treatment_mean / (1.0 - treatment_mean)
        ) / (control_mean / (1.0 - control_mean))
        significant = lower > 0.0
        passes += int(np.count_nonzero(significant))
        useful += int(np.count_nonzero(significant & (estimated_or >= 1.25)))
        estimates.extend(estimated_or.tolist())
    return {
        "roots": roots,
        "draws": draws,
        "base_rate": POWER_BASE_RATE,
        "root_rho": POWER_ROOT_RHO,
        "repeats": POWER_REPEATS,
        "assumed_activity": POWER_ACTIVITY,
        "realized_activity": realized_activity,
        "active_roots": active,
        "target_policy_odds_ratio": policy_or,
        "implied_active_root_odds_ratio": active_or,
        "power_ci_above_zero": passes / draws,
        "power_pass_point_or_1_25_and_ci": useful / draws,
        "median_estimated_policy_odds_ratio": float(np.median(estimates)),
    }


def prospective_power_audit() -> dict[str, Any]:
    rows = [
        simulate_power(roots, odds_ratio)
        for roots in POWER_DESIGNS
        for odds_ratio in POWER_OR_GRID
    ]
    by_design: dict[int, dict[str, Any]] = {}
    for roots in POWER_DESIGNS:
        design_rows = [row for row in rows if row["roots"] == roots]
        target = next(
            row
            for row in design_rows
            if row["target_policy_odds_ratio"] == POWER_TARGET_OR
        )
        passing_grid = [
            row["target_policy_odds_ratio"]
            for row in design_rows
            if row["power_pass_point_or_1_25_and_ci"] >= POWER_REQUIRED
        ]
        by_design[roots] = {
            "target_or_row": target,
            "mde_grid_or": min(passing_grid) if passing_grid else None,
        }
    viable = [
        roots
        for roots, row in by_design.items()
        if row["target_or_row"]["power_pass_point_or_1_25_and_ci"]
        >= POWER_REQUIRED
        and row["mde_grid_or"] is not None
        and row["mde_grid_or"] <= 2.0
    ]
    return {
        "assumptions": {
            "base_rate": POWER_BASE_RATE,
            "root_rho": POWER_ROOT_RHO,
            "repeats": POWER_REPEATS,
            "activity": POWER_ACTIVITY,
            "target_policy_odds_ratio": POWER_TARGET_OR,
            "required_power": POWER_REQUIRED,
            "draws": POWER_DRAWS,
            "designs": list(POWER_DESIGNS),
            "mde_grid": list(POWER_OR_GRID),
        },
        "rows": rows,
        "design_summary": {str(key): value for key, value in by_design.items()},
        "minimum_viable_roots": min(viable) if viable else None,
    }


def run_preflight(out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"G2 preflight output already exists: {out_dir}")
    proposal_hash = sha256_path(PROPOSAL_PATH)
    if proposal_hash != PROPOSAL_SHA256:
        raise ValueError(
            f"G2 proposal hash mismatch: {proposal_hash} != {PROPOSAL_SHA256}"
        )
    for required_path in (
        FEATURE_SOURCE_PATH,
        PREFLIGHT_SOURCE_PATH,
        TEST_SOURCE_PATH,
        TEST_EVIDENCE_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing G2 lock input: {required_path}")
    started = time.time()
    a2 = _json(A2_INVENTORY_PATH)
    source_manifest = _source_manifest(list(a2["source_paths"]))
    headers, source_counts = _load_source_headers(source_manifest)
    representatives, alias_count = _representative_headers(headers)
    candidate_records, scan_counts, restore_failures = _candidate_records(
        representatives
    )
    historical = _historical_sets(a2)
    partitioned, partition_audit = _partition_records(
        candidate_records,
        historical,
    )
    feature_audit = _feature_coverage(partitioned)
    representation = representation_self_audit()
    power = prospective_power_audit()
    summaries = {
        partition: _root_summary(partitioned, partition)
        for partition in (
            "train",
            "development",
            "untouched_transfer",
            "diagnostic_only",
            "diagnostic_prior_overlap_transfer",
            "withheld_earlier_from_transfer",
        )
    }
    all_roots = {str(row["root_cluster"]) for row in partitioned}
    overlap_counts = {
        name: len(all_roots.intersection(roots))
        for name, roots in historical["sets"].items()
    }

    manifest_rows = []
    for row in partitioned:
        manifest_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "record_id",
                    "root_cluster",
                    "behavior_family",
                    "scale",
                    "target",
                    "built_max",
                    "source_replay",
                    "source_replay_sha256",
                    "source_frame_index",
                    "state_sha1",
                    "starter_tile",
                    "state",
                    "partition",
                }
            }
        )
    root_manifest_payload = {
        "version": "g2_scale_equivariant_root_manifest_v1",
        "proposal_sha256": proposal_hash,
        "selection_uses_future_outcome": False,
        "score_or_recorded_action_inspected": False,
        "source_manifest": source_manifest,
        "records": manifest_rows,
        "records_sha256": canonical_sha256(manifest_rows),
    }
    root_manifest_payload["canonical_payload_sha256"] = canonical_sha256(
        root_manifest_payload
    )

    train = summaries["train"]
    development = summaries["development"]
    transfer = summaries["untouched_transfer"]
    minimum_viable_roots = power["minimum_viable_roots"]
    data_checks = {
        "source_manifest_missing_zero": source_manifest["missing_count"] == 0,
        "provenance_restore_failures_zero": not restore_failures,
        "cross_partition_roots_zero": not partition_audit["cross_partition_roots"],
        "train_min_roots": train["unique_roots"] >= TRAIN_MIN_ROOTS,
        "train_scale_minimums": all(
            train["roots_by_scale"][scale] >= TRAIN_SCALE_MIN
            for scale in TRAIN_SCALES
        ),
        "train_min_families": train["family_count"] >= TRAIN_MIN_FAMILIES,
        "train_effective_family_cap": all(
            share <= FAMILY_WEIGHT_CAP + 1e-12
            for share in train["effective_family_shares"].values()
        ),
        "development_min_roots": development["unique_roots"] >= DEV_MIN_ROOTS,
        "development_scale_minimums": all(
            development["roots_by_scale"][scale] >= DEV_SCALE_MIN
            for scale in TRAIN_SCALES
        ),
        "development_min_families": development["family_count"]
        >= DEV_MIN_FAMILIES,
        "transfer_min_roots": transfer["unique_roots"] >= TRANSFER_MIN_ROOTS,
        "transfer_min_families": transfer["family_count"]
        >= TRANSFER_MIN_FAMILIES,
        "transfer_family_cap": transfer["raw_max_family_share"]
        <= TRANSFER_MAX_FAMILY_SHARE,
        "transfer_meets_power_design": minimum_viable_roots is not None
        and transfer["unique_roots"] >= minimum_viable_roots,
        "feature_nonfinite_zero": feature_audit["nonfinite_values"] == 0,
        "feature_bounds_pass": feature_audit["out_of_bounds_values"] == 0,
    }
    disk = shutil.disk_usage("threes_rl/runs")
    data_checks["free_disk_above_100_gib"] = disk.free >= FREE_DISK_MIN_BYTES

    representation_checks = dict(representation["checks"])
    representation_checks["schema_feature_version_exact"] = (
        schema_manifest()["version"] == FEATURE_VERSION
    )
    representation_pass = all(representation_checks.values())
    if not representation_pass:
        decision = "KILL_G2_REPRESENTATION_PREFLIGHT"
    elif all(data_checks.values()):
        decision = "READY_G2_RELATIONAL_HAZARD_LABEL_PREFLIGHT"
    else:
        decision = "HOLD_G2_DATA_OR_POWER"

    services = service_health()
    preflight_payload = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "proposal": {
            "path": str(PROPOSAL_PATH),
            "sha256": proposal_hash,
        },
        "locks": {
            "a2_inventory_path": str(A2_INVENTORY_PATH),
            "a2_inventory_sha256": sha256_path(A2_INVENTORY_PATH),
            "s3_provenance_path": str(S3_PROVENANCE_PATH),
            "s3_provenance_sha256": sha256_path(S3_PROVENANCE_PATH),
            "qd5_summary_path": str(QD5_SUMMARY_PATH),
            "qd5_summary_sha256": sha256_path(QD5_SUMMARY_PATH),
            "feature_schema_sha256": schema_sha256(),
            "feature_schema": schema_manifest(),
            "feature_source_path": str(FEATURE_SOURCE_PATH),
            "feature_source_sha256": sha256_path(FEATURE_SOURCE_PATH),
            "preflight_source_path": str(PREFLIGHT_SOURCE_PATH),
            "preflight_source_sha256": sha256_path(PREFLIGHT_SOURCE_PATH),
            "test_source_path": str(TEST_SOURCE_PATH),
            "test_source_sha256": sha256_path(TEST_SOURCE_PATH),
            "test_evidence_path": str(TEST_EVIDENCE_PATH),
            "test_evidence_sha256": sha256_path(TEST_EVIDENCE_PATH),
        },
        "source_inventory": {
            "counts": dict(sorted(source_counts.items())),
            "source_manifest_count": source_manifest["source_count"],
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "valid_headers": len(headers),
            "representative_roots": len(representatives),
            "root_aliases_removed": alias_count,
            "scan_counts": dict(sorted(scan_counts.items())),
            "candidate_records": len(candidate_records),
            "candidate_roots": len(
                {str(row["root_cluster"]) for row in candidate_records}
            ),
            "restore_failures": restore_failures,
        },
        "availability": {
            "all_natural_candidates": _root_summary(
                [
                    {**row, "partition": "all"}
                    for row in candidate_records
                ],
                "all",
            ),
            "joint_and_conditional_scale_overlap": _joint_scale_summary(
                candidate_records
            ),
            "partitions": summaries,
        },
        "historical_overlap": {
            "catalog_counts": historical["counts"],
            "catalog_hashes": historical["hashes"],
            "candidate_overlap_counts": overlap_counts,
            "partition_audit": partition_audit,
        },
        "feature_audit": feature_audit,
        "representation_audit": {
            **representation,
            "checks": representation_checks,
            "passes": representation_pass,
        },
        "prospective_power": power,
        "readiness_checks": data_checks,
        "disk": {
            "free_bytes": int(disk.free),
            "free_gib": disk.free / 1024**3,
            "minimum_bytes": FREE_DISK_MIN_BYTES,
        },
        "services": services,
        "runtime_seconds": time.time() - started,
        "zero_forbidden_work": {
            "new_games": 0,
            "streams_consumed": 0,
            "labels": 0,
            "rollouts": 0,
            "h10_h20_h40_outcomes": 0,
            "models_fit": 0,
            "candidate_actions": 0,
            "score_inspection": False,
            "continuations": 0,
            "dashboard_changed": False,
            "incumbent_changed": False,
        },
        "dashboard_eligible": False,
        "labels_authorized_by_ready": decision
        == "READY_G2_RELATIONAL_HAZARD_LABEL_PREFLIGHT",
        "fitting_authorized": False,
        "promotion_authorized": False,
    }
    preflight_payload["root_manifest_canonical_payload_sha256"] = (
        root_manifest_payload["canonical_payload_sha256"]
    )
    preflight_payload["canonical_payload_sha256"] = canonical_sha256(
        preflight_payload
    )

    staging = out_dir.with_name(out_dir.name + f".staging_{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"G2 staging output already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        root_manifest_path = staging / "G2_ROOT_MANIFEST.json"
        preflight_path = staging / "G2_PREFLIGHT.json"
        _write_immutable_json(root_manifest_path, root_manifest_payload)
        preflight_payload["root_manifest_file_sha256"] = sha256_path(
            root_manifest_path
        )
        preflight_payload["canonical_payload_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in preflight_payload.items()
                if key != "canonical_payload_sha256"
            }
        )
        _write_immutable_json(preflight_path, preflight_payload)
        staging.rename(out_dir)
    except Exception:
        raise
    return preflight_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    payload = run_preflight(args.out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

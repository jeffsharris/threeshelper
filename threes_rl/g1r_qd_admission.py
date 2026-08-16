"""Outcome-free admission for the frozen G1-R static-archive QD family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl.eval import make_policy
from threes_rl.g1r_acquire import (
    AUTHORITATIVE_G1_V5_PATH,
    DASHBOARD_PATH,
    DIAGNOSTIC_INVENTORY_PATH,
    INCUMBENT_PATH,
    MIN_FREE_GIB,
    TARGET_FREE_GIB,
    _directory_artifact_manifest,
    _policy_action_values,
    _roundtrip_state,
    canonical_json_hash,
    current_nice,
    historical_collision_union,
    policy_slate,
    service_health,
)
from threes_rl.ntuple import NtupleValue
from threes_rl.record_replay import state_payload
from threes_rl.replay_provenance import ORIGIN_FRESH, replay_provenance
from threes_rl.restart_manifest import state_signature
from threes_rl.run_artifacts import write_json
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import (
    DIRECTION_NAMES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    SimState,
    ThreesSim,
    simulate_base_move,
)
from threes_rl.train_td import state_from_replay_payload


VERSION = "g1r_qd_admission_v1"
PROPOSAL_PATH = Path("threes_rl/G1R_QUALITY_DIVERSITY_FAMILY_PROPOSAL.md")
PROPOSAL_SHA256 = (
    "e9a72c659ae43302a3f646614c1a5e1c09daf8c696f6d5d1d7d5d150a50bf880"
)
CHARTER_PATH = Path("threes_rl/G1R_QD_ADMISSION_EXECUTION_CHARTER.md")
IMPLEMENTATION_PATH = Path("threes_rl/g1r_qd_admission.py")
TEST_PATH = Path("tests/test_rl_g1r_qd_admission.py")
PILOT_V1_PREFLIGHT_PATH = Path(
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v1/"
    "preflight_lock_pilot_v1.json"
)
PILOT_V1_PREFLIGHT_SHA256 = (
    "f78288b3f47bda6aa6d15c2157fd79f7b3d0685f0367d8b9964f5dc73981ea91"
)
PANEL_SHA256 = (
    "b8862aa3c8eaf6278fc078fb3e03aa7222a01930673cfee497738c74e81eff9d"
)
OUTPUT_DIR = Path("threes_rl/runs/forensics/g1r_qd_admission_v1")
PARENT_CHECKPOINT = Path(
    "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_"
    "20260706/latest"
)
MINIMUM_NICE = 10
MAX_RESERVED_GAMES = 12_000
PAIRWISE_FLOOR = 0.02
STRATA = ("pre1536", "pre3072")
REFERENCE_FAMILIES = (
    "g1r_corner2",
    "g1r_expectimax2",
    "g1r_parent_mc1000",
    "g1r_replaycal",
)
REFERENCE_ACTION_SIGNATURE_SHA256 = {
    "g1r_corner2": (
        "4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043"
    ),
    "g1r_expectimax2": (
        "2ad642cdca7739cc73af4f570de5054c422815f9a7d8f93a2619921b46b74b38"
    ),
    "g1r_parent_mc1000": (
        "e43dc11f3220557d7f9aef228db96dc6f06f49b26300d5a4128ea00bf8ba2064"
    ),
    "g1r_replaycal": (
        "e07c566b55d86a889ab7ca54d01c00c9b6cdf808fdb1627f70596bd829fdeab3"
    ),
}
STREAM_BASES = {
    "logical_seed": 45_000_000_000,
    "deck_stream_id": 46_000_000_000,
    "slot_stream_id": 47_000_000_000,
    "policy_stream_id": 48_000_000_000,
}
SOURCE_PATHS = (
    Path("threes_rl/sim.py"),
    Path("threes_rl/eval.py"),
    Path("threes_rl/expectimax.py"),
    Path("threes_rl/ntuple.py"),
    Path("threes_rl/record_replay.py"),
)
CATEGORICAL_INDICES = (0, 1, 2, 10, 11, 12, 13)
ORDINAL_DENOMINATORS = {
    3: 6.0,
    4: 4.0,
    5: 4.0,
    6: 4.0,
    7: 3.0,
    8: 3.0,
    9: 3.0,
}
DESCRIPTOR_WIDTH = 14
DESCRIPTOR_SCHEMA = {
    "version": "g1r_qd_descriptor_v1",
    "ordered_columns": [
        "built_max_band",
        "built_max_cell",
        "second_largest_cell",
        "built_second_manhattan",
        "support_component_count_cap4",
        "target_support_edge_count_cap4",
        "empty_count_bin_cap4",
        "legal_action_count",
        "top_row_monotonic_violations",
        "left_column_monotonic_violations",
        "starter_cell",
        "anchor_integrity",
        "preview_category",
        "large_pending",
    ],
    "starter_removal": (
        "remove value1536 at cell0 when present, else row-major-smallest "
        "value1536, else none"
    ),
    "tile_order": "value descending then row-major index ascending",
    "missing_cell": 16,
    "missing_distance": 6,
    "connectivity": "4-neighbor, independently per support level",
    "support_levels": "built_max/2,/4,/8 when integral and >=3",
    "monotonic_formula": "adjacent left_or_top rank < right_or_bottom rank",
    "preview_order": ["blue", "red", "gray", "bonus"],
    "categorical_indices": list(CATEGORICAL_INDICES),
    "ordinal_denominators": {
        str(index): denominator
        for index, denominator in ORDINAL_DENOMINATORS.items()
    },
    "distance": "unweighted categorical Hamming plus normalized ordinal L1",
}
DESCRIPTOR_SCHEMA_SHA256 = canonical_json_hash(DESCRIPTOR_SCHEMA)
ABSOLUTE_LATENCY_LIMITS_NS = {
    "median": 75_000_000.0,
    "p90": 150_000_000.0,
    "p99": 250_000_000.0,
    "max": 500_000_000.0,
}
RELATIVE_LATENCY_LIMITS = {"median": 1.0, "p90": 1.5}


def _write_new_json_atomic(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Atomic-write temporary already exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _tile_rank(value: int) -> int:
    value = int(value)
    if value == 0:
        return 0
    if value in (1, 2):
        return 1
    if value < 3 or value % 3 != 0:
        raise ValueError(f"Invalid Threes tile value: {value}")
    quotient = value // 3
    if quotient & (quotient - 1):
        raise ValueError(f"Invalid Threes doubling tile: {value}")
    return 2 + int(math.log2(quotient))


def _working_board_and_starter(board: np.ndarray) -> tuple[np.ndarray, int]:
    working = np.asarray(board, dtype=np.int32).copy()
    matches = np.argwhere(working == 1536)
    starter_index = 16
    if len(matches):
        selected = 0
        for index, (row, col) in enumerate(matches):
            if int(row) == 0 and int(col) == 0:
                selected = index
                break
        row, col = matches[selected]
        starter_index = int(row) * 4 + int(col)
        working[int(row), int(col)] = 0
    return working, starter_index


def _built_band(value: int) -> int:
    if value < 384:
        return 0
    if value < 768:
        return 1
    if value < 1536:
        return 2
    if value < 3072:
        return 3
    return 4


def _support_levels(built_max: int) -> tuple[int, ...]:
    levels = []
    for divisor in (2, 4, 8):
        if built_max >= 3 * divisor and built_max % divisor == 0:
            value = built_max // divisor
            if value >= 3:
                levels.append(value)
    return tuple(levels)


def _component_count(board: np.ndarray, value: int) -> int:
    positions = {
        (int(row), int(col))
        for row, col in np.argwhere(board == int(value))
    }
    count = 0
    while positions:
        count += 1
        stack = [positions.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in positions:
                    positions.remove(neighbor)
                    stack.append(neighbor)
    return count


def _target_support_edges(
    board: np.ndarray,
    built_max: int,
    support_levels: tuple[int, ...],
) -> int:
    if built_max <= 0 or not support_levels:
        return 0
    supports = set(support_levels)
    count = 0
    for row in range(4):
        for col in range(4):
            value = int(board[row, col])
            for other_row, other_col in ((row + 1, col), (row, col + 1)):
                if other_row >= 4 or other_col >= 4:
                    continue
                other = int(board[other_row, other_col])
                if (value == built_max and other in supports) or (
                    other == built_max and value in supports
                ):
                    count += 1
    return min(4, count)


def _copy_state_with_board(state: SimState, board: np.ndarray) -> SimState:
    return SimState(
        board=np.asarray(board, dtype=np.int32).copy(),
        preview=state.preview,
        small_counts=dict(state.small_counts),
        small_pos=int(state.small_pos),
        small_seen_total=int(state.small_seen_total),
        span_small_pos=int(state.span_small_pos),
        large_pending=bool(state.large_pending),
        max_tile=max(int(state.max_tile), int(np.max(board, initial=0))),
        move_count=int(state.move_count),
        game_over=False,
    )


def board_descriptor(state: SimState, sim: ThreesSim) -> tuple[int, ...]:
    original = np.asarray(state.board, dtype=np.int32)
    working, starter_cell = _working_board_and_starter(original)
    positive = [
        (int(working[row, col]), row * 4 + col)
        for row in range(4)
        for col in range(4)
        if int(working[row, col]) > 0
    ]
    positive.sort(key=lambda item: (-item[0], item[1]))
    built_max = positive[0][0] if positive else 0
    built_cell = positive[0][1] if positive else 16
    second_cell = positive[1][1] if len(positive) >= 2 else 16
    if built_cell == 16 or second_cell == 16:
        distance = 6
    else:
        distance = abs(built_cell // 4 - second_cell // 4) + abs(
            built_cell % 4 - second_cell % 4
        )
    support_levels = _support_levels(built_max)
    components = min(
        4,
        sum(_component_count(working, value) for value in support_levels),
    )
    edges = _target_support_edges(working, built_max, support_levels)
    ranks = np.vectorize(_tile_rank, otypes=[np.int32])(original)
    top_violations = sum(
        int(ranks[0, col] < ranks[0, col + 1]) for col in range(3)
    )
    left_violations = sum(
        int(ranks[row, 0] < ranks[row + 1, 0]) for row in range(3)
    )
    anchor_integrity = int(
        starter_cell == 0 and top_violations == 0 and left_violations == 0
    )
    preview_order = {"blue": 0, "red": 1, "gray": 2, "bonus": 3}
    if state.preview.kind not in preview_order:
        raise ValueError(f"Unsupported preview category: {state.preview.kind}")
    legal_count = len(sim.legal_actions(state))
    if not 1 <= legal_count <= 4:
        raise ValueError(f"Descriptor requires a live state, legal={legal_count}")
    descriptor = (
        _built_band(built_max),
        built_cell,
        second_cell,
        distance,
        components,
        edges,
        min(4, int(np.count_nonzero(original == 0))),
        legal_count,
        top_violations,
        left_violations,
        starter_cell,
        anchor_integrity,
        preview_order[state.preview.kind],
        int(bool(state.large_pending)),
    )
    if len(descriptor) != DESCRIPTOR_WIDTH:
        raise AssertionError("Descriptor width changed")
    return tuple(int(value) for value in descriptor)


def mixed_descriptor_distance(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> float:
    if len(left) != DESCRIPTOR_WIDTH or len(right) != DESCRIPTOR_WIDTH:
        raise ValueError("Descriptor width mismatch")
    total = sum(float(left[index] != right[index]) for index in CATEGORICAL_INDICES)
    total += sum(
        abs(float(left[index]) - float(right[index])) / denominator
        for index, denominator in ORDINAL_DENOMINATORS.items()
    )
    return float(total)


class StaticArchive:
    def __init__(self, counts: dict[tuple[int, ...], int]) -> None:
        if not counts:
            raise ValueError("Static archive must not be empty")
        normalized: dict[tuple[int, ...], int] = {}
        for key, count in counts.items():
            cell = tuple(int(value) for value in key)
            if len(cell) != DESCRIPTOR_WIDTH:
                raise ValueError("Archive descriptor width mismatch")
            if int(count) <= 0:
                raise ValueError("Archive counts must be positive")
            normalized[cell] = int(count)
        self.counts = dict(sorted(normalized.items()))

    def nearest(self, cell: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        return min(
            (
                mixed_descriptor_distance(cell, archived),
                archived,
            )
            for archived in self.counts
        )

    def novelty(self, cell: tuple[int, ...]) -> float:
        distance, _nearest = self.nearest(cell)
        return 1.0 / (1.0 + self.counts.get(cell, 0)) + distance / 14.0

    def payload(self) -> dict[str, Any]:
        cells = [
            {"cell": list(cell), "count": count}
            for cell, count in self.counts.items()
        ]
        return {
            "version": "g1r_qd_static_archive_v1",
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "descriptor_schema_sha256": DESCRIPTOR_SCHEMA_SHA256,
            "cells": cells,
            "cell_table_sha256": canonical_json_hash(cells),
            "root_count": sum(self.counts.values()),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "StaticArchive":
        if payload.get("descriptor_schema_sha256") != DESCRIPTOR_SCHEMA_SHA256:
            raise ValueError("Archive descriptor schema mismatch")
        cells = payload.get("cells")
        if not isinstance(cells, list):
            raise ValueError("Archive cells missing")
        if canonical_json_hash(cells) != payload.get("cell_table_sha256"):
            raise ValueError("Archive cell table hash mismatch")
        archive = cls(
            {
                tuple(int(value) for value in row["cell"]): int(row["count"])
                for row in cells
            }
        )
        if archive.payload()["root_count"] != int(payload.get("root_count", -1)):
            raise ValueError("Archive root count mismatch")
        return archive


def _ordinal_ranks(values: dict[int, float]) -> dict[int, int]:
    ordered = sorted(set(values.values()), reverse=True)
    return {
        action: ordered.index(value)
        for action, value in values.items()
    }


class StaticArchiveQDPolicy:
    def __init__(
        self,
        *,
        archive: StaticArchive,
        parent_checkpoint: Path,
        parent_model: Any | None = None,
        archive_path: Path | None = None,
    ) -> None:
        self.archive = archive
        self.parent_checkpoint = Path(parent_checkpoint)
        self.parent_model = (
            parent_model
            if parent_model is not None
            else NtupleValue.load(self.parent_checkpoint, mmap_mode="r")
        )
        self.archive_path = archive_path
        self.name = "g1r_qd_static_archive_oneply_v1"

    def action_values(
        self,
        state: SimState,
        sim: ThreesSim,
    ) -> dict[int, dict[str, float]]:
        values: dict[int, dict[str, float]] = {}
        for action in sim.legal_actions(state):
            shifted, eligible_positions = simulate_base_move(state.board, action)
            if not eligible_positions:
                continue
            insert_options = sim._insert_value_options(state.preview)
            quality = 0.0
            novelty = 0.0
            probability_total = 0.0
            for position in eligible_positions:
                for inserted_value, value_probability in insert_options:
                    probability = float(value_probability) / len(eligible_positions)
                    board = shifted.copy()
                    board[position] = int(inserted_value)
                    outcome_state = _copy_state_with_board(state, board)
                    descriptor = board_descriptor(outcome_state, sim)
                    quality += probability * float(self.parent_model.value(board))
                    novelty += probability * self.archive.novelty(descriptor)
                    probability_total += probability
            if not np.isclose(probability_total, 1.0, atol=1e-12):
                raise ValueError(
                    f"Spawn probabilities sum to {probability_total} for {action}"
                )
            values[int(action)] = {
                "quality": float(quality),
                "novelty": float(novelty),
                "spawn_probability": float(probability_total),
            }
        return values

    def decision(
        self,
        state: SimState,
        sim: ThreesSim,
    ) -> dict[str, Any]:
        values = self.action_values(state, sim)
        if not values:
            raise ValueError("QD policy received a state without legal actions")
        quality_ranks = _ordinal_ranks(
            {action: row["quality"] for action, row in values.items()}
        )
        novelty_ranks = _ordinal_ranks(
            {action: row["novelty"] for action, row in values.items()}
        )
        objective = {
            action: (
                quality_ranks[action] + novelty_ranks[action],
                -values[action]["quality"],
                -values[action]["novelty"],
            )
            for action in values
        }
        best_without_action = min(objective.values())
        tied = sorted(
            action for action, value in objective.items() if value == best_without_action
        )
        selected = tied[0]
        return {
            "action": int(selected),
            "action_name": DIRECTION_NAMES[selected],
            "tie_count_before_action_priority": len(tied),
            "values": {
                str(action): {
                    **row,
                    "quality_rank": quality_ranks[action],
                    "novelty_rank": novelty_ranks[action],
                    "rank_sum": quality_ranks[action] + novelty_ranks[action],
                }
                for action, row in sorted(values.items())
            },
        }

    def __call__(self, state: SimState, sim: ThreesSim, rng: Any) -> int:
        del rng
        return int(self.decision(state, sim)["action"])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        archive_payload = self.archive.payload()
        write_json(path / "archive.json", archive_payload)
        archive_sha = sha256_path(path / "archive.json")
        parent_manifest = _directory_artifact_manifest(self.parent_checkpoint)
        write_json(
            path / "policy.json",
            {
                "version": VERSION,
                "name": self.name,
                "archive_file": "archive.json",
                "archive_file_sha256": archive_sha,
                "descriptor_schema_sha256": DESCRIPTOR_SCHEMA_SHA256,
                "parent_checkpoint": str(self.parent_checkpoint),
                "parent_checkpoint_manifest_sha256": parent_manifest[
                    "manifest_sha256"
                ],
                "objective": "ordinal_quality_rank_plus_novelty_rank_v1",
            },
        )

    @classmethod
    def load(cls, path: Path) -> "StaticArchiveQDPolicy":
        meta = json.loads((path / "policy.json").read_text())
        if meta.get("version") != VERSION:
            raise ValueError("QD policy version mismatch")
        if meta.get("descriptor_schema_sha256") != DESCRIPTOR_SCHEMA_SHA256:
            raise ValueError("QD policy descriptor schema mismatch")
        archive_path = path / str(meta["archive_file"])
        if sha256_path(archive_path) != meta.get("archive_file_sha256"):
            raise ValueError("QD policy archive hash mismatch")
        archive = StaticArchive.from_payload(json.loads(archive_path.read_text()))
        parent = Path(str(meta["parent_checkpoint"]))
        manifest = _directory_artifact_manifest(parent)
        if manifest["manifest_sha256"] != meta.get(
            "parent_checkpoint_manifest_sha256"
        ):
            raise ValueError("QD policy parent checkpoint hash mismatch")
        return cls(
            archive=archive,
            parent_checkpoint=parent,
            archive_path=archive_path,
        )


def _selected_archive_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = json.loads(DIAGNOSTIC_INVENTORY_PATH.read_text())
    by_root: dict[str, dict[str, Any]] = {}
    scan_counts = Counter()
    for record in inventory["selected_records"]:
        if record.get("root_origin") != "fresh":
            scan_counts["nonfresh"] += 1
            continue
        if int(record.get("starter_tile", 1536)) != 1536:
            scan_counts["wrong_starter"] += 1
            continue
        payload = record["state"]
        computed_state_sha = state_signature(payload, 1536)
        root = str(record["root_cluster"])
        candidate = dict(record)
        candidate["_computed_state_sha1"] = computed_state_sha
        key = canonical_json_hash(
            [
                "G1R-QD-archive-state-v1",
                root,
                str(record["record_id"]),
                computed_state_sha,
            ]
        )
        candidate["_selection_key"] = key
        prior = by_root.get(root)
        if prior is None or key < prior["_selection_key"]:
            by_root[root] = candidate
        scan_counts["eligible_records"] += 1
    selected = sorted(by_root.values(), key=lambda row: str(row["root_cluster"]))
    return selected, {
        "inventory_records": len(inventory["selected_records"]),
        "eligible_records": scan_counts["eligible_records"],
        "selected_roots": len(selected),
        "nonfresh": scan_counts["nonfresh"],
        "wrong_starter": scan_counts["wrong_starter"],
    }


def _canonical_fresh_root(root_cluster: object) -> tuple[int, int]:
    parts = str(root_cluster).split(":")
    if len(parts) != 3 or parts[0] != ORIGIN_FRESH:
        raise ValueError(f"Archive root is not canonical fresh ancestry: {root_cluster}")
    try:
        seed = int(parts[1])
        starter_tile = int(parts[2])
    except ValueError as error:
        raise ValueError(
            f"Archive root has invalid seed/starter: {root_cluster}"
        ) from error
    return seed, starter_tile


def _validate_archive_root_provenance(
    record: dict[str, Any],
    replay: dict[str, Any],
    replay_path: Path,
) -> dict[str, Any]:
    canonical_seed, canonical_starter = _canonical_fresh_root(
        record["root_cluster"]
    )
    provenance = replay_provenance(replay, replay_path)
    checks = {
        "record_root_origin": record.get("root_origin") == ORIGIN_FRESH,
        "record_root_seed": int(record.get("root_seed", -1)) == canonical_seed,
        "record_starter": int(record.get("starter_tile", -1))
        == canonical_starter,
        "canonical_starter": canonical_starter == 1536,
        "replay_seed": int(replay.get("seed", -1)) == canonical_seed,
        "replay_starter": int(replay.get("starter_tile", -1))
        == canonical_starter,
        "replay_origin": provenance["replay_origin"] == ORIGIN_FRESH,
        "replay_reset_invariant": bool(provenance["replay_reset_invariant"]),
        "root_origin": provenance["root_origin"] == ORIGIN_FRESH,
        "root_seed": provenance["root_seed"] == canonical_seed,
        "root_frame_index": provenance["root_frame_index"] == 0,
        "root_move_count": provenance["root_move_count"] == 0,
    }
    explicit_expectations = {
        "replay_origin": ORIGIN_FRESH,
        "root_origin": ORIGIN_FRESH,
        "root_seed": canonical_seed,
        "root_frame_index": 0,
        "root_move_count": 0,
    }
    for field, expected in explicit_expectations.items():
        if replay.get(field) is not None:
            checks[f"explicit_{field}"] = replay[field] == expected
    if replay.get("root_replay") is not None:
        checks["explicit_root_replay"] = (
            Path(str(replay["root_replay"])).resolve() == replay_path.resolve()
        )
    failures = sorted(name for name, passes in checks.items() if not passes)
    if failures:
        raise ValueError(
            "Archive canonical ancestry mismatch "
            f"{record['root_cluster']} at {replay_path}: {failures}"
        )
    return {
        "canonical_seed": canonical_seed,
        "canonical_starter_tile": canonical_starter,
        "replay_origin": provenance["replay_origin"],
        "root_origin": provenance["root_origin"],
        "root_seed": provenance["root_seed"],
        "root_frame_index": provenance["root_frame_index"],
        "root_move_count": provenance["root_move_count"],
        "replay_reset_invariant": provenance["replay_reset_invariant"],
        "replay_reset_reason": provenance["replay_reset_reason"],
        "checks": checks,
        "passes": True,
    }


def construct_archive(out_dir: Path) -> dict[str, Any]:
    selected, scan = _selected_archive_records()
    by_replay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        by_replay[str(record["source_replay"])].append(record)
    source_rows = []
    counts: Counter[tuple[int, ...]] = Counter()
    for source_path in sorted(by_replay):
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing archive source replay: {path}")
        actual_sha = sha256_path(path)
        expected_hashes = {
            str(record["source_replay_sha256"])
            for record in by_replay[source_path]
        }
        if expected_hashes != {actual_sha}:
            raise ValueError(f"Archive source replay hash mismatch: {path}")
        replay = json.loads(path.read_text())
        provenance_by_root = {
            str(record["root_cluster"]): _validate_archive_root_provenance(
                record,
                replay,
                path,
            )
            for record in by_replay[source_path]
        }
        frame_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for frame in replay["frames"]:
            frame_map[int(frame["index"])].append(frame)
        for record in by_replay[source_path]:
            frame_index = int(record["source_frame_index"])
            frames = frame_map.get(frame_index, [])
            if len(frames) != 1:
                raise ValueError(f"Archive frame lookup is not unique: {path}:{frame_index}")
            payload = record["state"]
            if frames[0]["state"] != payload:
                raise ValueError(f"Archive frame payload mismatch: {path}:{frame_index}")
            _roundtrip_state(payload)
            state = state_from_replay_payload(payload)
            sim = ThreesSim.from_stream_ids(
                deck_stream_id=1,
                slot_stream_id=2,
                starter_tile=1536,
            )
            if state_payload(state, sim) != payload:
                raise ValueError(f"Archive simulator payload mismatch: {path}:{frame_index}")
            descriptor = board_descriptor(state, sim)
            counts[descriptor] += 1
            source_rows.append(
                {
                    "root_cluster": str(record["root_cluster"]),
                    "record_id": str(record["record_id"]),
                    "selection_key": str(record["_selection_key"]),
                    "source_replay": source_path,
                    "source_replay_sha256": actual_sha,
                    "source_frame_index": frame_index,
                    "state_sha1": str(record["_computed_state_sha1"]),
                    "descriptor_cell": list(descriptor),
                    "canonical_provenance": provenance_by_root[
                        str(record["root_cluster"])
                    ],
                }
            )
    if len({row["root_cluster"] for row in source_rows}) != len(source_rows):
        raise ValueError("Archive selection is not one state per root")
    archive = StaticArchive(dict(counts))
    source_manifest = {
        "version": "g1r_qd_archive_sources_v1",
        "source_inventory": str(DIAGNOSTIC_INVENTORY_PATH),
        "source_inventory_sha256": sha256_path(DIAGNOSTIC_INVENTORY_PATH),
        "selection_rule": (
            'argmin SHA256(canonical_json(["G1R-QD-archive-state-v1", '
            "root_cluster, record_id, state_sha1]))"
        ),
        "scan": scan,
        "records": source_rows,
    }
    source_manifest["selected_source_manifest_sha256"] = canonical_json_hash(
        source_manifest
    )
    archive_payload = archive.payload()
    archive_payload["selected_source_manifest_sha256"] = source_manifest[
        "selected_source_manifest_sha256"
    ]
    write_json(out_dir / "archive_sources.json", source_manifest)
    write_json(out_dir / "archive.json", archive_payload)
    return {
        "source_manifest": source_manifest,
        "archive": archive_payload,
        "archive_file_sha256": sha256_path(out_dir / "archive.json"),
        "archive_sources_file_sha256": sha256_path(
            out_dir / "archive_sources.json"
        ),
    }


def _reserved_stream_audit(out_dir: Path) -> dict[str, Any]:
    prior, sources = historical_collision_union(exclude_dir=out_dir)
    collisions = {}
    for key, base in STREAM_BASES.items():
        values = set(prior.get(key, set()))
        if key == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                values.update(prior.get(alias, set()))
        collisions[key] = sorted(
            value
            for value in values
            if base <= value < base + MAX_RESERVED_GAMES
        )
    return {
        "ranges": {
            key: {
                "start": base,
                "stop_exclusive": base + MAX_RESERVED_GAMES,
            }
            for key, base in STREAM_BASES.items()
        },
        "streams_consumed": False,
        "historical_union": sources,
        "collisions": collisions,
        "zero_collisions": not any(collisions.values()),
    }


def _heavy_process_audit() -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "ax", "-o", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    current_pid = os.getpid()
    processes: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        processes[pid] = {
            "pid": pid,
            "ppid": ppid,
            "command": parts[2] if len(parts) == 3 else "",
        }

    ancestor_pids = {current_pid}
    cursor = current_pid
    while cursor in processes:
        parent = int(processes[cursor]["ppid"])
        if parent <= 0 or parent in ancestor_pids:
            break
        ancestor_pids.add(parent)
        cursor = parent

    heavy = []
    for pid, process in sorted(processes.items()):
        if pid in ancestor_pids:
            continue
        command = str(process["command"])
        if "python" not in command or "threes_rl" not in command:
            continue
        if any(
            allowed in command
            for allowed in (
                "threes_rl.dashboard",
                "threes_rl.human_play_server",
            )
        ):
            continue
        if any(
            token in command
            for token in (
                "train",
                "eval",
                "acquire",
                "admission",
                "continuation",
                "label",
            )
        ):
            heavy.append(
                {
                    "pid": pid,
                    "ppid": int(process["ppid"]),
                    "command": command,
                }
            )
    return {
        "current_pid": current_pid,
        "excluded_ancestor_pids": sorted(ancestor_pids),
        "other_heavy_processes": heavy,
        "passes": not heavy,
    }


def _panel_payload() -> dict[str, Any]:
    if sha256_path(PILOT_V1_PREFLIGHT_PATH) != PILOT_V1_PREFLIGHT_SHA256:
        raise ValueError("Original G1-R preflight changed")
    preflight = json.loads(PILOT_V1_PREFLIGHT_PATH.read_text())
    panel = preflight["action_distinctness_panel"]
    stored_hash = panel["panel_sha256"]
    unhashed = dict(panel)
    unhashed.pop("panel_sha256")
    if stored_hash != PANEL_SHA256 or canonical_json_hash(unhashed) != PANEL_SHA256:
        raise ValueError("Immutable action panel hash mismatch")
    if Counter(row["stratum"] for row in panel["records"]) != {
        "pre1536": 32,
        "pre3072": 32,
    }:
        raise ValueError("Immutable action panel lost stratum balance")
    stored_signatures = preflight["action_distinctness_audit"][
        "action_signature_sha256"
    ]
    mismatches = {
        family: {
            "expected": expected,
            "stored": stored_signatures.get(family),
        }
        for family, expected in REFERENCE_ACTION_SIGNATURE_SHA256.items()
        if stored_signatures.get(family) != expected
    }
    if mismatches:
        raise ValueError(
            f"Immutable reference signature locks changed: {mismatches}"
        )
    return panel


def _verify_reference_action_signatures(
    signatures: dict[str, list[int]],
) -> dict[str, str]:
    actual = {
        family: canonical_json_hash(signatures[family])
        for family in REFERENCE_FAMILIES
    }
    mismatches = {
        family: {
            "expected": REFERENCE_ACTION_SIGNATURE_SHA256[family],
            "actual": actual[family],
        }
        for family in REFERENCE_FAMILIES
        if actual[family] != REFERENCE_ACTION_SIGNATURE_SHA256[family]
    }
    if mismatches:
        raise ValueError(
            f"Recomputed immutable reference signatures changed: {mismatches}"
        )
    return actual


def _policy_source_hashes() -> dict[str, str]:
    return {str(path): sha256_path(path) for path in SOURCE_PATHS}


def _thermal_power_snapshot() -> dict[str, Any]:
    snapshots = {}
    for name, command in (
        ("pmset_therm", ["pmset", "-g", "therm"]),
        ("pmset_batt", ["pmset", "-g", "batt"]),
    ):
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            snapshots[name] = {
                "available": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            snapshots[name] = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
    return snapshots


def _prepare_execution_lock_in_staging(
    staging_dir: Path,
    final_dir: Path,
) -> dict[str, Any]:
    free_gib = shutil.disk_usage(staging_dir).free / (1024**3)
    services = service_health()
    heavy = _heavy_process_audit()
    panel = _panel_payload()
    archive_artifacts = construct_archive(staging_dir)
    parent_manifest = _directory_artifact_manifest(PARENT_CHECKPOINT)
    policy_dir = staging_dir / "policy"
    policy = StaticArchiveQDPolicy(
        archive=StaticArchive.from_payload(archive_artifacts["archive"]),
        parent_checkpoint=PARENT_CHECKPOINT,
        archive_path=final_dir / "archive.json",
    )
    policy.save(policy_dir)
    reloaded = StaticArchiveQDPolicy.load(policy_dir)
    if reloaded.archive.payload() != policy.archive.payload():
        raise ValueError("QD policy save/reload archive mismatch")
    stream_audit = _reserved_stream_audit(staging_dir)
    checks = {
        "proposal_hash": sha256_path(PROPOSAL_PATH) == PROPOSAL_SHA256,
        "original_preflight_unchanged": sha256_path(PILOT_V1_PREFLIGHT_PATH)
        == PILOT_V1_PREFLIGHT_SHA256,
        "panel_hash": panel["panel_sha256"] == PANEL_SHA256,
        "archive_root_capped": archive_artifacts["archive"]["root_count"]
        == archive_artifacts["source_manifest"]["scan"]["selected_roots"],
        "archive_sources_validated": len(
            archive_artifacts["source_manifest"]["records"]
        )
        == archive_artifacts["archive"]["root_count"],
        "archive_canonical_provenance": all(
            row["canonical_provenance"]["passes"]
            for row in archive_artifacts["source_manifest"]["records"]
        ),
        "policy_save_reload": True,
        "reserved_streams_collision_free_and_unused": stream_audit[
            "zero_collisions"
        ],
        "nice_at_least_10": current_nice() >= MINIMUM_NICE,
        "one_heavy_process": heavy["passes"],
        "free_disk_above_100_gib": free_gib >= MIN_FREE_GIB,
        "free_disk_target_120_gib": free_gib >= TARGET_FREE_GIB,
        "services_dashboard_top3": services["passes"],
    }
    if not all(checks.values()):
        raise ValueError(f"QD admission preparation failed: {checks}")
    final_policy_dir = final_dir / "policy"
    lock = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bound_out_dir": str(final_dir),
        "proposal": str(PROPOSAL_PATH),
        "proposal_sha256": PROPOSAL_SHA256,
        "charter": str(CHARTER_PATH),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "implementation_sha256": sha256_path(IMPLEMENTATION_PATH),
        "focused_test_sha256": sha256_path(TEST_PATH),
        "authoritative_g1_v5_sha256": sha256_path(AUTHORITATIVE_G1_V5_PATH),
        "pilot_v1_preflight_sha256": PILOT_V1_PREFLIGHT_SHA256,
        "panel_sha256": PANEL_SHA256,
        "panel_record_count": len(panel["records"]),
        "reference_action_signature_sha256": REFERENCE_ACTION_SIGNATURE_SHA256,
        "a2_inventory": str(DIAGNOSTIC_INVENTORY_PATH),
        "a2_inventory_sha256": sha256_path(DIAGNOSTIC_INVENTORY_PATH),
        "selected_archive_source_manifest_sha256": archive_artifacts[
            "source_manifest"
        ]["selected_source_manifest_sha256"],
        "archive_sources_file_sha256": archive_artifacts[
            "archive_sources_file_sha256"
        ],
        "archive_file_sha256": archive_artifacts["archive_file_sha256"],
        "archive_cell_table_sha256": archive_artifacts["archive"][
            "cell_table_sha256"
        ],
        "archive_root_count": archive_artifacts["archive"]["root_count"],
        "archive_cell_count": len(archive_artifacts["archive"]["cells"]),
        "descriptor_schema_sha256": DESCRIPTOR_SCHEMA_SHA256,
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "parent_checkpoint_artifacts": parent_manifest,
        "policy_bundle": str(final_policy_dir),
        "policy_json_sha256": sha256_path(policy_dir / "policy.json"),
        "policy_archive_sha256": sha256_path(policy_dir / "archive.json"),
        "policy_source_hashes": _policy_source_hashes(),
        "incumbent_policy_file_sha256": sha256_path(INCUMBENT_PATH),
        "reserved_stream_audit": stream_audit,
        "streams_consumed": False,
        "process_nice": current_nice(),
        "required_minimum_nice": MINIMUM_NICE,
        "timing_processes": 1,
        "heavy_process_audit": heavy,
        "service_health": services,
        "free_gib": free_gib,
        "free_disk_floor_gib": MIN_FREE_GIB,
        "free_disk_target_gib": TARGET_FREE_GIB,
        "thermal_power_preparation": _thermal_power_snapshot(),
        "checks": checks,
        "admission_actions_measured": False,
        "games_generated": 0,
        "labels_generated": 0,
        "models_fit": 0,
        "score_outcomes_inspected": False,
        "dashboard_eligible": False,
    }
    lock["lock_payload_sha256"] = canonical_json_hash(lock)
    _write_new_json_atomic(staging_dir / "execution_lock.json", lock)
    return lock


def prepare_execution_lock(out_dir: Path) -> dict[str, Any]:
    expected_out = OUTPUT_DIR.resolve()
    final_dir = out_dir.resolve()
    if final_dir != expected_out:
        raise ValueError(f"QD admission output must be {expected_out}")
    if sha256_path(PROPOSAL_PATH) != PROPOSAL_SHA256:
        raise ValueError("Authoritative QD proposal hash mismatch")
    if current_nice() < MINIMUM_NICE:
        raise ValueError("QD preparation requires nice priority >=10")
    if final_dir.exists():
        raise FileExistsError(f"QD admission output is immutable/existing: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_dir.with_name(f"{final_dir.name}.staging.{os.getpid()}")
    if staging_dir.exists():
        raise FileExistsError(f"QD staging output already exists: {staging_dir}")
    staging_dir.mkdir()
    stage = "staging_created"
    try:
        stage = "archive_policy_lock"
        lock = _prepare_execution_lock_in_staging(staging_dir, final_dir)
        stage = "atomic_promotion"
        os.replace(staging_dir, final_dir)
        return lock
    except Exception as error:
        failure = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_QD_PREPARATION_ERROR",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "bound_final_out_dir": str(final_dir),
            "staging_out_dir": str(staging_dir),
            "admission_opened": False,
            "games_generated": 0,
            "labels_generated": 0,
            "models_fit": 0,
            "score_outcomes_inspected": False,
            "dashboard_changed": False,
            "pilot_authorized": False,
        }
        if staging_dir.exists():
            _write_new_json_atomic(
                staging_dir / "PREPARATION_FAILED.json",
                failure,
            )
        raise


def _load_and_validate_lock(out_dir: Path) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("QD admission output directory mismatch")
    lock_path = out_dir / "execution_lock.json"
    lock = json.loads(lock_path.read_text())
    payload_hash = lock.pop("lock_payload_sha256")
    if canonical_json_hash(lock) != payload_hash:
        raise ValueError("QD execution lock payload mismatch")
    lock["lock_payload_sha256"] = payload_hash
    if lock["bound_out_dir"] != str(out_dir.resolve()):
        raise ValueError("QD execution lock is bound to another directory")
    if (
        lock.get("reference_action_signature_sha256")
        != REFERENCE_ACTION_SIGNATURE_SHA256
    ):
        raise ValueError("QD execution reference signature locks changed")
    identities = {
        "proposal_sha256": sha256_path(PROPOSAL_PATH),
        "charter_sha256": sha256_path(CHARTER_PATH),
        "implementation_sha256": sha256_path(IMPLEMENTATION_PATH),
        "focused_test_sha256": sha256_path(TEST_PATH),
        "pilot_v1_preflight_sha256": sha256_path(PILOT_V1_PREFLIGHT_PATH),
        "a2_inventory_sha256": sha256_path(DIAGNOSTIC_INVENTORY_PATH),
        "archive_sources_file_sha256": sha256_path(
            out_dir / "archive_sources.json"
        ),
        "archive_file_sha256": sha256_path(out_dir / "archive.json"),
        "policy_json_sha256": sha256_path(out_dir / "policy/policy.json"),
        "policy_archive_sha256": sha256_path(out_dir / "policy/archive.json"),
        "incumbent_policy_file_sha256": sha256_path(INCUMBENT_PATH),
    }
    mismatches = {
        key: {"expected": lock.get(key), "actual": value}
        for key, value in identities.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"QD execution identity mismatch: {mismatches}")
    if _policy_source_hashes() != lock["policy_source_hashes"]:
        raise ValueError("QD policy/simulator source hashes changed")
    parent_manifest = _directory_artifact_manifest(PARENT_CHECKPOINT)
    if parent_manifest["manifest_sha256"] != lock["parent_checkpoint_artifacts"][
        "manifest_sha256"
    ]:
        raise ValueError("QD parent checkpoint changed")
    if current_nice() < int(lock["required_minimum_nice"]):
        raise ValueError("QD admission requires nice priority >=10")
    if not _heavy_process_audit()["passes"]:
        raise ValueError("Another heavy Threes process is active")
    services = service_health()
    if not services["passes"]:
        raise ValueError("QD admission service/dashboard health failed")
    free_gib = shutil.disk_usage(out_dir).free / (1024**3)
    if free_gib < MIN_FREE_GIB:
        raise ValueError("QD admission free disk below 100 GiB")
    stream_audit = _reserved_stream_audit(out_dir)
    if not stream_audit["zero_collisions"]:
        raise ValueError("QD reserved stream collision appeared")
    return lock


def _verify_retained_archive_sources(out_dir: Path) -> dict[str, Any]:
    manifest = json.loads((out_dir / "archive_sources.json").read_text())
    rows_by_replay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["records"]:
        rows_by_replay[str(row["source_replay"])].append(row)
    verified_states = 0
    replay_hashes: dict[str, str] = {}
    for replay_text in sorted(rows_by_replay):
        replay_path = Path(replay_text)
        if not replay_path.is_file():
            raise FileNotFoundError(f"Retained archive replay missing: {replay_path}")
        actual_sha = sha256_path(replay_path)
        expected = {
            str(row["source_replay_sha256"])
            for row in rows_by_replay[replay_text]
        }
        if expected != {actual_sha}:
            raise ValueError(f"Retained archive replay hash changed: {replay_path}")
        replay = json.loads(replay_path.read_text())
        frame_map = {
            int(frame["index"]): frame
            for frame in replay["frames"]
        }
        if len(frame_map) != len(replay["frames"]):
            raise ValueError(f"Retained replay has duplicate frame indices: {replay_path}")
        for row in rows_by_replay[replay_text]:
            frame_index = int(row["source_frame_index"])
            if frame_index not in frame_map:
                raise ValueError(
                    f"Retained archive frame missing: {replay_path}:{frame_index}"
                )
            payload = frame_map[frame_index]["state"]
            if state_signature(payload, 1536) != row["state_sha1"]:
                raise ValueError(
                    f"Retained archive state changed: {replay_path}:{frame_index}"
                )
            _roundtrip_state(payload)
            canonical_seed, canonical_starter = _canonical_fresh_root(
                row["root_cluster"]
            )
            provenance = _validate_archive_root_provenance(
                {
                    "root_cluster": row["root_cluster"],
                    "root_origin": ORIGIN_FRESH,
                    "root_seed": canonical_seed,
                    "starter_tile": canonical_starter,
                },
                replay,
                replay_path,
            )
            if not provenance["passes"]:
                raise ValueError(
                    f"Retained archive provenance failed: {replay_path}"
                )
            verified_states += 1
        replay_hashes[replay_text] = actual_sha
    return {
        "source_manifest_sha256": sha256_path(out_dir / "archive_sources.json"),
        "replay_count": len(rows_by_replay),
        "state_count": verified_states,
        "replay_hashes_sha256": canonical_json_hash(replay_hashes),
        "passes": verified_states == len(manifest["records"]),
    }


def _state_fingerprint(state: SimState) -> str:
    payload = {
        "board": np.asarray(state.board, dtype=int).tolist(),
        "preview": {
            "kind": state.preview.kind,
            "value": state.preview.value,
            "candidates": list(state.preview.candidates),
        },
        "small_counts": dict(state.small_counts),
        "small_pos": state.small_pos,
        "small_seen_total": state.small_seen_total,
        "span_small_pos": state.span_small_pos,
        "large_pending": state.large_pending,
        "max_tile": state.max_tile,
        "move_count": state.move_count,
        "game_over": state.game_over,
    }
    return canonical_json_hash(payload)


def _baseline_action(policy: Any, state: SimState, sim: ThreesSim) -> tuple[int, int]:
    values = _policy_action_values(policy, state, sim)
    best = max(value for _action, value in values)
    tied = sorted(action for action, value in values if value == best)
    return int(tied[0]), len(tied)


def _clear_transient_caches(policy: Any) -> None:
    for name in (
        "_cache",
        "_action_cache",
        "_eval_cache",
        "_afterstate_cache",
        "_post_spawn_cache",
        "_score_cache",
        "_legal_cache",
        "_base_move_cache",
    ):
        cache = getattr(policy, name, None)
        if cache is not None:
            cache.clear()


def latency_distribution(values_ns: Iterable[int]) -> dict[str, float]:
    values = np.asarray(list(values_ns), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Latency distribution is empty")
    return {
        "count": int(values.size),
        "median": float(np.quantile(values, 0.50, method="linear")),
        "p90": float(np.quantile(values, 0.90, method="linear")),
        "p99": float(np.quantile(values, 0.99, method="linear")),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def latency_gate(
    candidate_ns: Iterable[int],
    incumbent_ns: Iterable[int],
) -> dict[str, Any]:
    candidate = latency_distribution(candidate_ns)
    incumbent = latency_distribution(incumbent_ns)
    ratios = {
        "median": candidate["median"] / incumbent["median"],
        "p90": candidate["p90"] / incumbent["p90"],
    }
    checks = {
        f"absolute_{name}": candidate[name] <= limit
        for name, limit in ABSOLUTE_LATENCY_LIMITS_NS.items()
    }
    checks.update(
        {
            f"relative_{name}": ratios[name] <= limit
            for name, limit in RELATIVE_LATENCY_LIMITS.items()
        }
    )
    return {
        "candidate_ns": candidate,
        "incumbent_ns": incumbent,
        "ratios": ratios,
        "absolute_limits_ns": ABSOLUTE_LATENCY_LIMITS_NS,
        "relative_limits": RELATIVE_LATENCY_LIMITS,
        "checks": checks,
        "passes": all(checks.values()),
    }


def run_admission(out_dir: Path) -> dict[str, Any]:
    result_path = out_dir / "admission_result.json"
    opened_path = out_dir / "ADMISSION_OPENED.json"
    hold_path = out_dir / "HOLD_QD_ADMISSION_ERROR.json"
    if opened_path.exists() or result_path.exists() or hold_path.exists():
        raise FileExistsError("QD admission is one-shot and already opened")
    lock = _load_and_validate_lock(out_dir)
    panel = _panel_payload()
    qd = StaticArchiveQDPolicy.load(out_dir / "policy")
    specs = dict(policy_slate())
    references = {
        family: make_policy(specs[family])
        for family in REFERENCE_FAMILIES
    }
    incumbent = make_policy(
        next(
            line.strip()
            for line in INCUMBENT_PATH.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    )
    states = []
    for index, record in enumerate(panel["records"]):
        state = state_from_replay_payload(record["state"])
        sim = ThreesSim.from_stream_ids(
            deck_stream_id=90_000 + index,
            slot_stream_id=100_000 + index,
            starter_tile=1536,
        )
        if state_payload(state, sim) != record["state"]:
            raise ValueError(f"Panel state round-trip failed at {index}")
        states.append((record, state, sim))
    retained_source_audit = _verify_retained_archive_sources(out_dir)
    if not retained_source_audit["passes"]:
        raise ValueError("Retained archive source audit failed before admission")
    opened = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "admission_opened": True,
        "execution_lock_payload_sha256": lock["lock_payload_sha256"],
        "panel_sha256": PANEL_SHA256,
        "implementation_sha256": lock["implementation_sha256"],
        "focused_test_sha256": lock["focused_test_sha256"],
        "charter_sha256": lock["charter_sha256"],
        "retained_source_audit": retained_source_audit,
        "streams_consumed": False,
        "games_generated": 0,
        "labels_generated": 0,
        "models_fit": 0,
        "score_outcomes_inspected": False,
        "dashboard_changed": False,
        "pilot_authorized": False,
    }
    opened["opened_payload_sha256"] = canonical_json_hash(opened)
    _write_new_json_atomic(opened_path, opened)
    progress = {"stage": "reference_action_signatures"}
    try:
        return _execute_opened_admission(
            out_dir=out_dir,
            lock=lock,
            panel=panel,
            qd=qd,
            references=references,
            incumbent=incumbent,
            states=states,
            opened=opened,
            progress=progress,
        )
    except Exception as error:
        hold = {
            "version": VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "decision": "HOLD_QD_ADMISSION_ERROR",
            "stage": progress["stage"],
            "error_type": type(error).__name__,
            "error": str(error),
            "admission_opened": True,
            "admission_opened_file_sha256": sha256_path(opened_path),
            "execution_lock_payload_sha256": lock["lock_payload_sha256"],
            "panel_sha256": PANEL_SHA256,
            "implementation_sha256": lock["implementation_sha256"],
            "focused_test_sha256": lock["focused_test_sha256"],
            "streams_consumed": False,
            "games_generated": 0,
            "labels_generated": 0,
            "models_fit": 0,
            "continuation_outcomes_generated": 0,
            "score_outcomes_inspected": False,
            "dashboard_changed": False,
            "dashboard_eligible": False,
            "promotion": False,
            "pilot_authorized": False,
        }
        hold["result_payload_sha256"] = canonical_json_hash(hold)
        _write_new_json_atomic(hold_path, hold)
        return hold


def _execute_opened_admission(
    *,
    out_dir: Path,
    lock: dict[str, Any],
    panel: dict[str, Any],
    qd: StaticArchiveQDPolicy,
    references: dict[str, Any],
    incumbent: Any,
    states: list[tuple[dict[str, Any], SimState, ThreesSim]],
    opened: dict[str, Any],
    progress: dict[str, str],
) -> dict[str, Any]:
    result_path = out_dir / "admission_result.json"
    signatures: dict[str, list[int]] = {
        "g1r_qd_static_archive_oneply_v1": []
    }
    tie_counts: dict[str, int] = {
        "g1r_qd_static_archive_oneply_v1": 0
    }
    exactness_checks = []
    for family in REFERENCE_FAMILIES:
        signatures[family] = []
        tie_counts[family] = 0
    for index, (_record, state, sim) in enumerate(states):
        before = _state_fingerprint(state)
        qd_decision = qd.decision(state, sim)
        signatures["g1r_qd_static_archive_oneply_v1"].append(
            int(qd_decision["action"])
        )
        tie_counts["g1r_qd_static_archive_oneply_v1"] += int(
            qd_decision["tie_count_before_action_priority"] > 1
        )
        for family, policy in references.items():
            action, tie_count = _baseline_action(policy, state, sim)
            signatures[family].append(action)
            tie_counts[family] += int(tie_count > 1)
        after = _state_fingerprint(state)
        exactness_checks.append(
            {
                "panel_index": index,
                "state_unmutated": before == after,
                "qd_spawn_probabilities_one": all(
                    np.isclose(row["spawn_probability"], 1.0, atol=1e-12)
                    for row in qd_decision["values"].values()
                ),
            }
        )

    reference_signature_hashes = _verify_reference_action_signatures(signatures)
    progress["stage"] = "pairwise_distinctness"
    pairwise = []
    candidate_signature = signatures["g1r_qd_static_archive_oneply_v1"]
    for family in REFERENCE_FAMILIES:
        reference = signatures[family]
        stratum_rates = {}
        for stratum in STRATA:
            indices = [
                index
                for index, (record, _state, _sim) in enumerate(states)
                if record["stratum"] == stratum
            ]
            stratum_rates[stratum] = sum(
                candidate_signature[index] != reference[index]
                for index in indices
            ) / len(indices)
        overall = sum(
            left != right
            for left, right in zip(candidate_signature, reference, strict=True)
        ) / len(candidate_signature)
        passes = overall >= PAIRWISE_FLOOR and all(
            stratum_rates[stratum] > 0.0 for stratum in STRATA
        )
        pairwise.append(
            {
                "candidate": "g1r_qd_static_archive_oneply_v1",
                "reference": family,
                "overall_disagreement": overall,
                "stratum_disagreement": stratum_rates,
                "passes": passes,
            }
        )

    progress["stage"] = "timing"
    thermal_before = _thermal_power_snapshot()
    candidate_timings = []
    incumbent_timings = []
    timing_rows = []
    warmup_actions = []
    for index, (_record, state, sim) in enumerate(states):
        _clear_transient_caches(qd)
        qd_action = int(qd.decision(state, sim)["action"])
        _clear_transient_caches(incumbent)
        incumbent_action, _tie = _baseline_action(incumbent, state, sim)
        warmup_actions.append(
            {
                "panel_index": index,
                "candidate_action": qd_action,
                "incumbent_action": incumbent_action,
            }
        )
    deterministic_actions = []
    for pass_index in range(5):
        for state_index, (_record, state, sim) in enumerate(states):
            call_order = (
                ("candidate", "incumbent")
                if (pass_index + state_index) % 2 == 0
                else ("incumbent", "candidate")
            )
            row = {
                "pass_index": pass_index,
                "panel_index": state_index,
                "call_order": list(call_order),
            }
            for arm in call_order:
                if arm == "candidate":
                    _clear_transient_caches(qd)
                    started = time.perf_counter_ns()
                    action = int(qd.decision(state, sim)["action"])
                    elapsed = time.perf_counter_ns() - started
                    candidate_timings.append(elapsed)
                    row["candidate_action"] = action
                    row["candidate_ns"] = elapsed
                    deterministic_actions.append(
                        action == candidate_signature[state_index]
                    )
                else:
                    _clear_transient_caches(incumbent)
                    started = time.perf_counter_ns()
                    action, _tie = _baseline_action(incumbent, state, sim)
                    elapsed = time.perf_counter_ns() - started
                    incumbent_timings.append(elapsed)
                    row["incumbent_action"] = action
                    row["incumbent_ns"] = elapsed
            timing_rows.append(row)
    thermal_after = _thermal_power_snapshot()
    timing = latency_gate(candidate_timings, incumbent_timings)
    action_pass = all(row["passes"] for row in pairwise)
    exactness_pass = all(
        row["state_unmutated"] and row["qd_spawn_probabilities_one"]
        for row in exactness_checks
    ) and all(deterministic_actions)
    if not action_pass:
        decision = "KILL_QD_ALIAS"
    elif not timing["passes"]:
        decision = "KILL_QD_COST"
    elif exactness_pass:
        decision = "READY_QD_FAMILY_ADMISSION"
    else:
        raise ValueError("QD exactness failed after admission measurement")
    progress["stage"] = "postflight_integrity"
    post_services = service_health()
    post_free_gib = shutil.disk_usage(out_dir).free / (1024**3)
    if not post_services["passes"] or post_free_gib < MIN_FREE_GIB:
        raise ValueError("QD post-admission operational integrity failed")
    result = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "execution_lock_payload_sha256": lock["lock_payload_sha256"],
        "proposal_sha256": PROPOSAL_SHA256,
        "panel_sha256": PANEL_SHA256,
        "archive_file_sha256": lock["archive_file_sha256"],
        "archive_cell_table_sha256": lock["archive_cell_table_sha256"],
        "parent_checkpoint_manifest_sha256": lock[
            "parent_checkpoint_artifacts"
        ]["manifest_sha256"],
        "admission_opened_file_sha256": sha256_path(
            out_dir / "ADMISSION_OPENED.json"
        ),
        "admission_opened_payload_sha256": opened["opened_payload_sha256"],
        "reference_action_signature_sha256": reference_signature_hashes,
        "action_signature_sha256": {
            family: canonical_json_hash(actions)
            for family, actions in signatures.items()
        },
        "action_signatures": signatures,
        "tie_state_counts": tie_counts,
        "pairwise": pairwise,
        "pairwise_floor": PAIRWISE_FLOOR,
        "exactness_checks": exactness_checks,
        "deterministic_timing_actions": all(deterministic_actions),
        "warmup_actions": warmup_actions,
        "timing_schedule": {
            "warmups_per_state_per_arm": 1,
            "timed_passes": 5,
            "states": 64,
            "processes": 1,
            "nice": current_nice(),
            "cache_clear_before_call": True,
            "alternating_order": "(pass_index+state_index)%2",
        },
        "timing": timing,
        "timing_rows": timing_rows,
        "thermal_power_before": thermal_before,
        "thermal_power_after": thermal_after,
        "post_service_health": post_services,
        "post_free_gib": post_free_gib,
        "streams_consumed": False,
        "games_generated": 0,
        "labels_generated": 0,
        "models_fit": 0,
        "continuation_outcomes_generated": 0,
        "score_outcomes_inspected": False,
        "dashboard_changed": False,
        "dashboard_eligible": False,
        "pilot_authorized": False,
    }
    result["result_payload_sha256"] = canonical_json_hash(result)
    progress["stage"] = "result_sealing"
    _write_new_json_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare-lock", "run-admission"),
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    if args.command == "prepare-lock":
        payload = prepare_execution_lock(args.out_dir)
        compact = {
            "decision": "READY_QD_ADMISSION_LOCK",
            "lock_payload_sha256": payload["lock_payload_sha256"],
            "archive_root_count": payload["archive_root_count"],
            "archive_cell_count": payload["archive_cell_count"],
            "free_gib": payload["free_gib"],
            "actions_measured": False,
        }
    else:
        payload = run_admission(args.out_dir)
        compact = {
            "decision": payload["decision"],
            "result_payload_sha256": payload["result_payload_sha256"],
            "pairwise": payload.get("pairwise"),
            "timing": payload.get("timing"),
            "games_generated": 0,
        }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

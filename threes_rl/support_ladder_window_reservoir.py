"""Extract raw support-ladder milestone windows after first built 3072."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable

import numpy as np

from threes_rl.compare_replays import state_from_payload
from threes_rl.geometry_forensics import geometry_features
from threes_rl.replay_provenance import ORIGIN_FRESH, replay_provenance
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.sim import ThreesSim
from threes_rl.swing_label import state_features

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
MILESTONE_TARGET_TILES = {
    "raw_duplicate_768": 768,
    "raw_adjacent_768": 768,
    "raw_three_768_no_1536": 768,
    "raw_four_768_no_1536": 768,
    "raw_four_adjacent_768_no_1536": 768,
    "raw_one_1536": 1536,
    "raw_adjacent_768_with_1536": 1536,
    "raw_duplicate_1536": 1536,
    "raw_near_adjacent_1536": 1536,
    "raw_adjacent_1536": 1536,
    "second_3072": 6144,
}


@dataclass(frozen=True)
class MilestoneSpec:
    name: str
    prerequisite: str
    predicate: Callable[[dict[str, Any]], bool]


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _glob_paths(patterns: list[str] | None) -> list[Path]:
    if not patterns:
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(match) for match in glob.glob(pattern, recursive=True))
    return sorted(path for path in paths if path.is_file())


def _int_or_none(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _starter_from_replay(replay: dict[str, Any], default: int | None) -> int | None:
    value = replay.get("starter_tile", default)
    return None if value is None else int(value)


def _move_action(frame: object) -> str | None:
    if not isinstance(frame, dict):
        return None
    move = frame.get("move")
    if not isinstance(move, dict):
        return None
    action = move.get("action")
    return str(action) if action is not None else None


def _positions(board: np.ndarray, value: int) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in pos) for pos in np.argwhere(np.asarray(board, dtype=np.int32) == int(value))]


def _has_adjacent_pair(board: np.ndarray, value: int) -> bool:
    positions = _positions(board, value)
    for idx, left in enumerate(positions):
        for right in positions[idx + 1 :]:
            if abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1:
                return True
    return False


def _has_diagonal_touch_pair(board: np.ndarray, value: int) -> bool:
    positions = _positions(board, value)
    for idx, left in enumerate(positions):
        for right in positions[idx + 1 :]:
            if abs(left[0] - right[0]) == 1 and abs(left[1] - right[1]) == 1:
                return True
    return False


def _min_pair_distances(board: np.ndarray, value: int) -> tuple[int | None, int | None]:
    positions = _positions(board, value)
    if len(positions) < 2:
        return None, None
    min_manhattan: int | None = None
    min_chebyshev: int | None = None
    for idx, left in enumerate(positions):
        for right in positions[idx + 1 :]:
            manhattan = abs(left[0] - right[0]) + abs(left[1] - right[1])
            chebyshev = max(abs(left[0] - right[0]), abs(left[1] - right[1]))
            min_manhattan = manhattan if min_manhattan is None else min(min_manhattan, manhattan)
            min_chebyshev = chebyshev if min_chebyshev is None else min(min_chebyshev, chebyshev)
    return min_manhattan, min_chebyshev


def _tile_counts(board: np.ndarray) -> Counter[int]:
    return Counter(int(value) for value in np.asarray(board, dtype=np.int32).reshape(-1) if int(value) > 0)


def _highest_duplicate_tile(board: np.ndarray) -> int:
    counts = _tile_counts(board)
    return max((value for value, count in counts.items() if count >= 2), default=0)


def _highest_adjacent_pair_tile(board: np.ndarray) -> int:
    values = sorted({int(value) for value in np.asarray(board, dtype=np.int32).reshape(-1) if int(value) > 0}, reverse=True)
    for value in values:
        if _has_adjacent_pair(board, value):
            return int(value)
    return 0


def raw_ladder_features(board: np.ndarray, starter_tile: int | None) -> dict[str, Any]:
    arr = np.asarray(board, dtype=np.int32)
    geo = geometry_features(arr, starter_tile)
    raw_min_pair_distance_1536, raw_min_pair_chebyshev_1536 = _min_pair_distances(arr, 1536)
    raw_has_diagonal_touch_1536 = _has_diagonal_touch_pair(arr, 1536)
    raw_has_adjacent_1536 = _has_adjacent_pair(arr, 1536)
    return {
        "raw_count_768": int(np.count_nonzero(arr == 768)),
        "raw_count_1536": int(np.count_nonzero(arr == 1536)),
        "raw_count_3072": int(np.count_nonzero(arr == 3072)),
        "raw_count_6144": int(np.count_nonzero(arr == 6144)),
        "raw_highest_duplicate_tile": _highest_duplicate_tile(arr),
        "raw_highest_adjacent_pair_tile": _highest_adjacent_pair_tile(arr),
        "raw_has_adjacent_768": _has_adjacent_pair(arr, 768),
        "raw_has_adjacent_1536": raw_has_adjacent_1536,
        "raw_has_diagonal_touch_1536": raw_has_diagonal_touch_1536,
        "raw_has_near_adjacent_1536": bool(raw_has_adjacent_1536 or raw_has_diagonal_touch_1536),
        "raw_min_pair_distance_1536": raw_min_pair_distance_1536,
        "raw_min_pair_chebyshev_1536": raw_min_pair_chebyshev_1536,
        "masked_max_tile_excl_starter": int(geo["max_tile_excl_starter"]),
        "masked_count_1536": int(geo["count_1536"]),
        "masked_count_3072": int(geo["count_3072"]),
        "masked_highest_duplicate_tile": int(geo["highest_duplicate_tile"]),
        "masked_highest_adjacent_pair_tile": int(geo["highest_adjacent_pair_tile"]),
    }


def milestone_specs() -> dict[str, MilestoneSpec]:
    return {
        "first_3072": MilestoneSpec(
            "first_3072",
            "start",
            lambda features: int(features["masked_max_tile_excl_starter"]) >= 3072 or int(features["raw_count_3072"]) >= 1,
        ),
        "raw_duplicate_768": MilestoneSpec(
            "raw_duplicate_768",
            "first_3072",
            lambda features: int(features["raw_highest_duplicate_tile"]) >= 768,
        ),
        "raw_adjacent_768": MilestoneSpec(
            "raw_adjacent_768",
            "raw_duplicate_768",
            lambda features: int(features["raw_highest_adjacent_pair_tile"]) >= 768,
        ),
        "raw_three_768_no_1536": MilestoneSpec(
            "raw_three_768_no_1536",
            "raw_duplicate_768",
            lambda features: int(features["raw_count_768"]) >= 3 and int(features["raw_count_1536"]) == 0,
        ),
        "raw_four_768_no_1536": MilestoneSpec(
            "raw_four_768_no_1536",
            "raw_three_768_no_1536",
            lambda features: int(features["raw_count_768"]) >= 4 and int(features["raw_count_1536"]) == 0,
        ),
        "raw_four_adjacent_768_no_1536": MilestoneSpec(
            "raw_four_adjacent_768_no_1536",
            "raw_four_768_no_1536",
            lambda features: int(features["raw_count_768"]) >= 4
            and int(features["raw_count_1536"]) == 0
            and bool(features["raw_has_adjacent_768"]),
        ),
        "raw_one_1536": MilestoneSpec(
            "raw_one_1536",
            "raw_duplicate_768",
            lambda features: int(features["raw_count_1536"]) >= 1,
        ),
        "raw_adjacent_768_with_1536": MilestoneSpec(
            "raw_adjacent_768_with_1536",
            "raw_one_1536",
            lambda features: bool(features["raw_has_adjacent_768"]) and int(features["raw_count_1536"]) >= 1,
        ),
        "raw_duplicate_1536": MilestoneSpec(
            "raw_duplicate_1536",
            "raw_duplicate_768",
            lambda features: int(features["raw_count_1536"]) >= 2,
        ),
        "raw_near_adjacent_1536": MilestoneSpec(
            "raw_near_adjacent_1536",
            "raw_duplicate_1536",
            lambda features: bool(features["raw_has_near_adjacent_1536"]),
        ),
        "raw_adjacent_1536": MilestoneSpec(
            "raw_adjacent_1536",
            "raw_duplicate_1536",
            lambda features: bool(features["raw_has_adjacent_1536"]),
        ),
        "second_3072": MilestoneSpec(
            "second_3072",
            "raw_adjacent_1536",
            lambda features: int(features["raw_count_3072"]) >= 2 or int(features["masked_max_tile_excl_starter"]) >= 6144,
        ),
    }


def parse_targets(text: str | None) -> list[str]:
    raw = (
        text
        or "raw_duplicate_768,raw_adjacent_768,raw_three_768_no_1536,raw_four_768_no_1536,raw_four_adjacent_768_no_1536,raw_one_1536,raw_adjacent_768_with_1536,raw_duplicate_1536,raw_adjacent_1536,second_3072"
    )
    specs = milestone_specs()
    targets: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in specs or name == "first_3072":
            raise ValueError(f"Unsupported target milestone: {name}")
        if name not in seen:
            targets.append(name)
            seen.add(name)
    if not targets:
        raise ValueError("at least one target milestone is required")
    return targets


def _record_id(
    *,
    source_replay: Path,
    seed: int | None,
    frame_index: int,
    target_milestone: str,
    outcome: str,
) -> str:
    raw = json.dumps(
        {
            "source_replay": str(source_replay),
            "seed": seed,
            "frame_index": int(frame_index),
            "target_milestone": target_milestone,
            "outcome": outcome,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return safe_name(
        f"{source_replay.stem}_seed{seed}_{target_milestone}_{outcome}_frame{int(frame_index)}_{digest}",
        max_length=128,
    )


def _frame_rows(
    replay_path: Path,
    replay: dict[str, Any],
    *,
    default_starter_tile: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frames = replay.get("frames", [])
    if not isinstance(frames, list):
        return [], {"bad_replay": 1}
    starter_tile = _starter_from_replay(replay, default_starter_tile)
    seed = _int_or_none(replay.get("seed"))
    sim = ThreesSim(np.random.default_rng(seed or 0), starter_tile=starter_tile)
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for frame_pos, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        payload = frame.get("state")
        if not isinstance(payload, dict):
            rejected["missing_state"] += 1
            continue
        try:
            state = state_from_payload(payload)
        except (TypeError, ValueError):
            rejected["bad_state"] += 1
            continue
        board = np.asarray(state.board, dtype=np.int32)
        features = state_features(state, sim, starter_tile)
        raw = raw_ladder_features(board, starter_tile)
        rows.append(
            {
                "frame_position": int(frame_pos),
                "frame_index": int(frame.get("index", frame_pos)),
                "state": state,
                "state_payload": payload,
                "features": features,
                "raw": raw,
                "starter_tile": starter_tile,
                "source_seed": seed,
                "source_next_action": _move_action(frames[frame_pos + 1]) if frame_pos + 1 < len(frames) else None,
                "game_over": bool(state.game_over),
            }
        )
    return rows, dict(rejected)


def first_milestone_positions(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    specs = milestone_specs()
    positions: dict[str, int | None] = {"start": 0 if rows else None}
    for name, spec in specs.items():
        start = 0
        prereq = positions.get(spec.prerequisite)
        if prereq is None:
            positions[name] = None
            continue
        start = int(prereq)
        positions[name] = next(
            (
                idx
                for idx in range(start, len(rows))
                if spec.predicate(rows[idx]["raw"])
            ),
            None,
        )
    return positions


def _record_from_row(
    row: dict[str, Any],
    *,
    replay_path: Path,
    replay: dict[str, Any],
    target_milestone: str,
    prerequisite_milestone: str,
    outcome: str,
    window_start_position: int,
    window_end_position: int,
    prerequisite_position: int | None,
    milestone_position: int | None,
    terminal_position: int | None,
) -> dict[str, Any]:
    state = row["state"]
    features = dict(row["features"])
    raw = dict(row["raw"])
    frame_position = int(row["frame_position"])
    frame_index = int(row["frame_index"])
    seed = _int_or_none(replay.get("seed"))
    provenance = replay_provenance(replay, replay_path)
    original_source_replay = replay.get("source_replay")
    original_source_seed = _int_or_none(replay.get("source_seed"))
    original_source_frame_index = _int_or_none(replay.get("source_frame_index"))
    source_group = json.dumps(
        {
            "source_replay": str(original_source_replay or replay_path),
            "source_seed": original_source_seed if original_source_seed is not None else seed,
            "source_frame_index": original_source_frame_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    starter_tile = row.get("starter_tile", _starter_from_replay(replay, 1536))
    if starter_tile is not None:
        starter_tile = int(starter_tile)
    legal_count = int(len(ThreesSim(np.random.default_rng(seed or 0), starter_tile=starter_tile).legal_actions(state)))
    moves_to_milestone = None if milestone_position is None else int(milestone_position - frame_position)
    moves_to_terminal = None if terminal_position is None else int(terminal_position - frame_position)
    merged_features = {
        **features,
        **raw,
        "legal_count": legal_count,
        "target_milestone": target_milestone,
        "prerequisite_milestone": prerequisite_milestone,
    }
    return {
        "id": _record_id(
            source_replay=replay_path,
            seed=seed,
            frame_index=frame_index,
            target_milestone=target_milestone,
            outcome=outcome,
        ),
        "kind": "support_ladder_window_state",
        "target_milestone": target_milestone,
        "prerequisite_milestone": prerequisite_milestone,
        "target_tile": MILESTONE_TARGET_TILES.get(target_milestone),
        "outcome": outcome,
        "moves_to_milestone": moves_to_milestone,
        "moves_to_promotion": moves_to_milestone,
        "moves_to_terminal": moves_to_terminal,
        "window_start_position": int(window_start_position),
        "window_end_position": int(window_end_position),
        "prerequisite_frame_position": None if prerequisite_position is None else int(prerequisite_position),
        "milestone_frame_position": None if milestone_position is None else int(milestone_position),
        "terminal_frame_position": None if terminal_position is None else int(terminal_position),
        "source_replay": str(replay_path),
        "source_origin": provenance["replay_origin"],
        "source_group": source_group,
        "original_source_replay": str(original_source_replay) if original_source_replay is not None else None,
        "original_source_seed": original_source_seed,
        "original_source_frame_index": original_source_frame_index,
        "continuation_replay": str(replay_path),
        "continuation_seed": seed,
        "continuation_frame_index": frame_index,
        "source_policy": replay.get("policy"),
        "source_policy_family": provenance["source_policy_family"],
        "source_seed": seed,
        "seed": seed,
        "root_origin": provenance["root_origin"],
        "root_replay": provenance["root_replay"],
        "root_seed": provenance["root_seed"],
        "root_frame_index": provenance["root_frame_index"],
        "root_move_count": provenance["root_move_count"],
        "root_score": provenance["root_score"],
        "root_policy": provenance["root_policy"],
        "root_policy_family": provenance["root_policy_family"],
        "root_is_genuine": provenance["root_is_genuine"],
        "ancestry_key": provenance["ancestry_key"],
        "starter_tile": starter_tile,
        "source_frame_index": frame_index,
        "frame_position": frame_position,
        "source_next_action": row.get("source_next_action"),
        "move_count": int(state.move_count),
        "score": int(row["state_payload"].get("score", 0)),
        "score_minus_starter": int(features["score_minus_starter"]),
        "max_tile": int(features["max_tile"]),
        "max_tile_excl_starter": int(features["max_tile_excl_starter"]),
        "phase": str(features["phase"]),
        "corner_risk": str(features["corner_risk"]),
        "stratum": str(features["stratum"]),
        "empty_count": int(features["empty_count"]),
        "legal_count": legal_count,
        "preview": str(features["preview"]),
        "large_pending": bool(features["large_pending"]),
        "raw_count_768": int(raw["raw_count_768"]),
        "raw_count_1536": int(raw["raw_count_1536"]),
        "raw_count_3072": int(raw["raw_count_3072"]),
        "raw_count_6144": int(raw["raw_count_6144"]),
        "raw_highest_duplicate_tile": int(raw["raw_highest_duplicate_tile"]),
        "raw_highest_adjacent_pair_tile": int(raw["raw_highest_adjacent_pair_tile"]),
        "raw_has_adjacent_768": bool(raw["raw_has_adjacent_768"]),
        "raw_has_adjacent_1536": bool(raw["raw_has_adjacent_1536"]),
        "features": merged_features,
        "state": row["state_payload"],
    }


def _candidate_rows(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows[max(0, int(start)) : max(0, int(end)) + 1]
        if row.get("source_next_action") is not None and not bool(row.get("game_over"))
    ]


def _records_for_target(
    rows: list[dict[str, Any]],
    *,
    replay_path: Path,
    replay: dict[str, Any],
    target_milestone: str,
    window_size: int,
    include_failures: bool,
    positions: dict[str, int | None],
) -> list[dict[str, Any]]:
    spec = milestone_specs()[target_milestone]
    prerequisite = positions.get(spec.prerequisite)
    if prerequisite is None:
        return []
    target = positions.get(target_milestone)
    if target is not None:
        start = max(int(prerequisite), int(target) - int(window_size))
        end = int(target) - 1
        if end < start:
            return []
        return [
            _record_from_row(
                row,
                replay_path=replay_path,
                replay=replay,
                target_milestone=target_milestone,
                prerequisite_milestone=spec.prerequisite,
                outcome="success",
                window_start_position=start,
                window_end_position=end,
                prerequisite_position=prerequisite,
                milestone_position=target,
                terminal_position=None,
            )
            for row in _candidate_rows(rows, start, end)
        ]
    if not include_failures:
        return []
    candidates = _candidate_rows(rows, int(prerequisite), len(rows) - 1)
    if not candidates:
        return []
    window_rows = candidates[-int(window_size) :]
    start = int(window_rows[0]["frame_position"])
    end = int(window_rows[-1]["frame_position"])
    terminal_position = int(rows[-1]["frame_position"]) if rows else None
    return [
        _record_from_row(
            row,
            replay_path=replay_path,
            replay=replay,
            target_milestone=target_milestone,
            prerequisite_milestone=spec.prerequisite,
            outcome="failure",
            window_start_position=start,
            window_end_position=end,
            prerequisite_position=prerequisite,
            milestone_position=None,
            terminal_position=terminal_position,
        )
        for row in window_rows
    ]


def collect_support_ladder_records(
    replay_paths: Iterable[Path],
    *,
    targets: Iterable[str],
    window_size: int = 40,
    include_failures: bool = True,
    default_starter_tile: int | None = 1536,
    max_records: int = 0,
    root_origins: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    paths = [Path(path) for path in replay_paths]
    target_list = list(targets)
    replays_scanned = 0
    replays_considered = 0
    origin_counts: Counter[str] = Counter()
    root_origin_counts: Counter[str] = Counter()
    milestone_presence: Counter[str] = Counter()
    for replay_path in paths:
        try:
            replay = json.loads(replay_path.read_text())
        except (OSError, json.JSONDecodeError):
            rejected["bad_replay"] += 1
            continue
        if not isinstance(replay, dict):
            rejected["bad_replay"] += 1
            continue
        replays_considered += 1
        provenance = replay_provenance(replay, replay_path)
        origin_counts[str(provenance["replay_origin"])] += 1
        root_origin_counts[str(provenance["root_origin"])] += 1
        if root_origins is not None and str(provenance["root_origin"]) not in root_origins:
            rejected[f"root_origin:{provenance['root_origin']}"] += 1
            continue
        replays_scanned += 1
        rows, row_rejected = _frame_rows(replay_path, replay, default_starter_tile=default_starter_tile)
        rejected.update(row_rejected)
        if not rows:
            rejected["no_frames"] += 1
            continue
        positions = first_milestone_positions(rows)
        for name, position in positions.items():
            if name != "start" and position is not None:
                milestone_presence[name] += 1
        for target in target_list:
            records.extend(
                _records_for_target(
                    rows,
                    replay_path=replay_path,
                    replay=replay,
                    target_milestone=target,
                    window_size=window_size,
                    include_failures=include_failures,
                    positions=positions,
                )
            )
    if max_records > 0:
        records = records[: int(max_records)]
    summary = summarize_records(
        records,
        source_replay_paths=[str(path) for path in paths],
        replays_scanned=replays_scanned,
        replays_considered=replays_considered,
        targets=target_list,
        window_size=window_size,
        include_failures=include_failures,
        max_records=max_records,
        milestone_presence=dict(milestone_presence),
        rejected=dict(rejected),
        root_origins=None if root_origins is None else sorted(root_origins),
        replay_origin_counts=dict(origin_counts),
        root_origin_counts=dict(root_origin_counts),
    )
    return records, summary


def _stats(values: list[int]) -> dict[str, Any]:
    return {
        "mean": float(mean(values)) if values else None,
        "median": float(median(values)) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize_records(
    records: list[dict[str, Any]],
    *,
    source_replay_paths: list[str],
    replays_scanned: int,
    replays_considered: int,
    targets: list[str],
    window_size: int,
    include_failures: bool,
    max_records: int,
    milestone_presence: dict[str, int],
    rejected: dict[str, int],
    root_origins: list[str] | None,
    replay_origin_counts: dict[str, int],
    root_origin_counts: dict[str, int],
) -> dict[str, Any]:
    scores = [int(record["score_minus_starter"]) for record in records]
    moves = [int(record["moves_to_milestone"]) for record in records if record.get("moves_to_milestone") is not None]
    by_target_outcome = Counter(f"{record.get('target_milestone')}:{record.get('outcome')}" for record in records)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_replays": len(source_replay_paths),
        "source_replays_with_records": len({str(record.get("source_replay")) for record in records}),
        "unique_root_replays_with_records": len(
            {
                str(record.get("root_replay"))
                for record in records
                if record.get("root_replay") is not None
            }
        ),
        "unique_ancestry_keys_with_records": len(
            {
                str(record.get("ancestry_key"))
                for record in records
                if record.get("ancestry_key") is not None
            }
        ),
        "replays_scanned": int(replays_scanned),
        "replays_considered": int(replays_considered),
        "root_origin_filter": root_origins,
        "targets": list(targets),
        "window_size": int(window_size),
        "include_failures": bool(include_failures),
        "max_records": int(max_records),
        "records": len(records),
        "by_target_outcome": dict(by_target_outcome),
        "by_target": dict(Counter(str(record.get("target_milestone")) for record in records)),
        "by_outcome": dict(Counter(str(record.get("outcome")) for record in records)),
        "by_source_origin": dict(Counter(str(record.get("source_origin", "unknown")) for record in records)),
        "by_root_origin": dict(Counter(str(record.get("root_origin", "unknown")) for record in records)),
        "by_root_policy_family": dict(Counter(str(record.get("root_policy_family", "unknown")) for record in records)),
        "by_source_policy_family": dict(Counter(str(record.get("source_policy_family", "unknown")) for record in records)),
        "by_stratum": dict(Counter(str(record.get("stratum", "unknown")) for record in records)),
        "milestone_presence_replays": milestone_presence,
        "replay_origin_counts_considered": replay_origin_counts,
        "root_origin_counts_considered": root_origin_counts,
        "score_minus_starter": {
            "mean": float(mean(scores)) if scores else 0.0,
            "median": float(median(scores)) if scores else 0.0,
            "max": int(max(scores)) if scores else 0,
        },
        "moves_to_milestone": _stats(moves),
        "max_raw_highest_duplicate_tile": max((int(record.get("raw_highest_duplicate_tile") or 0) for record in records), default=0),
        "max_raw_highest_adjacent_pair_tile": max((int(record.get("raw_highest_adjacent_pair_tile") or 0) for record in records), default=0),
        "source_replay_paths": source_replay_paths,
        "rejected": rejected,
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])

    def cell(value: object) -> str:
        return escape(str(value))

    rows = []
    for record in records[:400] if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{cell(record.get('target_milestone'))}</td>"
            f"<td>{cell(record.get('outcome'))}</td>"
            f"<td>{cell(record.get('moves_to_milestone'))}</td>"
            f"<td>{cell(record.get('move_count'))}</td>"
            f"<td>{cell(record.get('stratum'))}</td>"
            f"<td>{cell(record.get('raw_count_768'))}</td>"
            f"<td>{cell(record.get('raw_count_1536'))}</td>"
            f"<td>{cell(record.get('raw_highest_duplicate_tile'))}</td>"
            f"<td>{cell(record.get('raw_highest_adjacent_pair_tile'))}</td>"
            f"<td>{cell(record.get('source_next_action'))}</td>"
            f"<td>{cell(record.get('source_replay'))}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Support Ladder Windows</title>
  <style>
    :root {{ color-scheme: dark; --bg:#101318; --panel:#171d24; --line:#34404d; --text:#edf2f7; --muted:#aab6c2; --gold:#f2c14e; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width:min(1240px, calc(100vw - 32px)); margin:0 auto; padding:24px 0 40px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .muted {{ color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:18px 0; }}
    .card {{ border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .value {{ margin-top:4px; color:var(--gold); font-size:22px; font-weight:800; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
    th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }}
    th:nth-child(1), td:nth-child(1), th:nth-child(5), td:nth-child(5), th:last-child, td:last-child {{ text-align:left; }}
    td:last-child {{ max-width:360px; overflow-wrap:anywhere; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Support Ladder Windows</h1>
    <p class="muted">Raw milestone windows after first built 3072.</p>
    <section class="cards">
      <div class="card"><div class="label">Records</div><div class="value">{cell(summary.get('records', 0))}</div></div>
      <div class="card"><div class="label">Source Replays</div><div class="value">{cell(summary.get('source_replays_with_records', 0))}</div></div>
      <div class="card"><div class="label">Targets</div><div class="value">{cell(summary.get('targets', []))}</div></div>
      <div class="card"><div class="label">Window</div><div class="value">{cell(summary.get('window_size', 0))}</div></div>
    </section>
    <table><thead><tr><th>Target</th><th>Outcome</th><th>To Milestone</th><th>Move</th><th>Stratum</th><th>768 Count</th><th>1536 Count</th><th>Dup</th><th>Adj</th><th>Next Action</th><th>Replay</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    <h2>Summary JSON</h2>
    <pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    replay_paths = _flatten_paths(args.replay_json) + _glob_paths(args.replay_glob)
    exclude_parts = [str(part) for part in getattr(args, "exclude_path_substring", []) if str(part)]
    if exclude_parts:
        replay_paths = [
            path
            for path in replay_paths
            if not any(part in str(path) for part in exclude_parts)
        ]
    starter_text = args.starter.strip().lower()
    default_starter = None if starter_text == "none" else int(starter_text)
    root_origin_text = str(getattr(args, "root_origin", "") or "").strip()
    root_origins = None
    if bool(getattr(args, "fresh_root_only", False)):
        root_origins = {ORIGIN_FRESH}
    elif root_origin_text:
        root_origins = {part.strip() for part in root_origin_text.split(",") if part.strip()}
    records, summary = collect_support_ladder_records(
        replay_paths,
        targets=parse_targets(args.targets),
        window_size=args.window_size,
        include_failures=not args.no_failures,
        default_starter_tile=default_starter,
        max_records=args.max_records,
        root_origins=root_origins,
    )
    payload = {
        "version": 1,
        "kind": "support_ladder_window_reservoir",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "support_ladder_windows.json")
    payload["records_json"] = str(args.out_dir / "records.json")
    payload["html"] = str(args.out_dir / "support_ladder_windows.html")
    write_json(args.out_dir / "support_ladder_windows.json", payload)
    write_json(args.out_dir / "records.json", records)
    write_html(args.out_dir / "support_ladder_windows.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-json", type=Path, nargs="+", action="append", default=[])
    parser.add_argument("--replay-glob", action="append", default=[])
    parser.add_argument(
        "--exclude-path-substring",
        action="append",
        default=[],
        help="Skip replay paths containing this substring. May be repeated.",
    )
    parser.add_argument(
        "--targets",
        default="raw_duplicate_768,raw_adjacent_768,raw_three_768_no_1536,raw_four_768_no_1536,raw_duplicate_1536,raw_adjacent_1536,second_3072",
    )
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--no-failures", action="store_true")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--fresh-root-only",
        action="store_true",
        help="Only scan replays whose root provenance is a genuine simulator fresh start.",
    )
    parser.add_argument(
        "--root-origin",
        help="Comma-separated root origins to scan, e.g. fresh,human. Ignored when --fresh-root-only is set.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/support_ladder_windows/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"records={payload['records_json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()

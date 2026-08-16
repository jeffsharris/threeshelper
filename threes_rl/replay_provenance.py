"""Replay origin and root-provenance helpers.

The project creates several replay-like artifacts: normal games, human imports,
continuations, and TD episodes that start from replay reservoirs.  Folder names
are not enough to distinguish those cases because a single TD run can emit both
fresh and replay-start episodes.  These helpers keep the classification explicit
for new artifacts and provide conservative reset-invariant checks for old ones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

ORIGIN_FRESH = "fresh"
ORIGIN_HUMAN = "human"
ORIGIN_REPLAY_START = "replay_start"
ORIGIN_CONTINUATION = "continuation"
ORIGIN_SYNTHETIC = "synthetic"
ORIGIN_UNKNOWN = "unknown"

GENUINE_ROOT_ORIGINS = {ORIGIN_FRESH, ORIGIN_HUMAN}
SMALL_TILE_VALUES = {1, 2, 3}


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def first_state_payload(replay: dict[str, Any]) -> dict[str, Any] | None:
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    first = frames[0]
    if not isinstance(first, dict):
        return None
    state = first.get("state")
    return state if isinstance(state, dict) else None


def initial_reset_diagnostics(replay: dict[str, Any]) -> dict[str, Any]:
    """Return conservative simulator-reset invariant diagnostics."""

    state = first_state_payload(replay)
    if state is None:
        return {"is_reset_start": False, "reason": "missing_first_state"}

    failures: list[str] = []
    move_count = _int_or_none(state.get("move_count"))
    if move_count != 0:
        failures.append(f"move_count={move_count}")

    board_obj = state.get("board")
    try:
        board = np.asarray(board_obj, dtype=np.int32)
    except (TypeError, ValueError):
        return {"is_reset_start": False, "reason": "bad_board"}
    if board.shape != (4, 4):
        return {"is_reset_start": False, "reason": f"bad_board_shape={board.shape}"}

    starter_value = replay.get("starter_tile", 1536)
    starter_tile = None if starter_value is None else _int_or_none(starter_value)
    working = board.copy()
    if starter_tile is not None:
        if int(working[0, 0]) != int(starter_tile):
            failures.append(f"top_left={int(working[0, 0])}")
        working[0, 0] = 0

    nonzero = [int(value) for value in working.reshape(-1) if int(value) > 0]
    if len(nonzero) != 8:
        failures.append(f"small_tile_count={len(nonzero)}")
    if any(value not in SMALL_TILE_VALUES for value in nonzero):
        failures.append("non_small_initial_tiles")

    preview = state.get("preview")
    preview_kind = preview.get("kind") if isinstance(preview, dict) else None
    if preview_kind not in ("blue", "red", "gray"):
        failures.append(f"preview={preview_kind}")

    cycle = state.get("tile_cycle")
    if isinstance(cycle, dict):
        if _int_or_none(cycle.get("small_pos")) != 8:
            failures.append(f"small_pos={cycle.get('small_pos')}")
        if _int_or_none(cycle.get("small_seen_total")) != 0:
            failures.append(f"small_seen_total={cycle.get('small_seen_total')}")
        if _int_or_none(cycle.get("span_small_pos")) != 0:
            failures.append(f"span_small_pos={cycle.get('span_small_pos')}")
        if bool(cycle.get("large_pending")):
            failures.append("large_pending=True")
    else:
        failures.append("missing_tile_cycle")

    score = _int_or_none(state.get("score"))
    return {
        "is_reset_start": not failures,
        "reason": "ok" if not failures else ",".join(failures),
        "move_count": move_count,
        "score": score,
        "starter_tile": starter_tile,
        "initial_small_tile_count": len(nonzero),
    }


def classify_replay_origin(replay: dict[str, Any]) -> str:
    existing = replay.get("replay_origin")
    if isinstance(existing, str) and existing:
        return existing
    policy = str(replay.get("policy", ""))
    if isinstance(replay.get("human_import"), dict) or policy == "human_observed":
        return ORIGIN_HUMAN
    if replay.get("synthetic") or str(replay.get("kind", "")).startswith("synthetic"):
        return ORIGIN_SYNTHETIC
    if replay.get("start_case_id") is not None or (
        replay.get("source_replay") is not None
        and (replay.get("final_score_delta") is not None or replay.get("final_moves_delta") is not None)
    ):
        return ORIGIN_CONTINUATION

    reset = initial_reset_diagnostics(replay)
    if bool(reset.get("is_reset_start")):
        return ORIGIN_FRESH

    state = first_state_payload(replay)
    move_count = _int_or_none(state.get("move_count")) if state is not None else None
    training_config = replay.get("training_config")
    if isinstance(training_config, dict) and move_count is not None and move_count > 0:
        return ORIGIN_REPLAY_START
    if move_count is not None and move_count > 0:
        return ORIGIN_REPLAY_START
    return ORIGIN_UNKNOWN


def policy_family(policy: object) -> str:
    if policy is None:
        return ORIGIN_UNKNOWN
    text = str(policy)
    if not text:
        return ORIGIN_UNKNOWN
    if text.startswith("train_td:"):
        run = text.split(":", 1)[1]
        return "train_td:" + run.split("/", 1)[0]
    return text.split(":", 1)[0].split("|", 1)[0]


def direct_root_fields(
    *,
    origin: str,
    seed: int | None,
    policy: str | None,
    replay_path: Path | str | None = None,
    first_score: int | None = None,
) -> dict[str, Any]:
    root_replay = None if replay_path is None else str(replay_path)
    return {
        "replay_origin": origin,
        "root_origin": origin,
        "root_replay": root_replay,
        "root_seed": seed,
        "root_frame_index": 0,
        "root_move_count": 0,
        "root_score": first_score,
        "root_policy": policy,
        "root_policy_family": policy_family(policy),
    }


def replay_provenance(replay: dict[str, Any], replay_path: Path | str | None = None) -> dict[str, Any]:
    origin = classify_replay_origin(replay)
    reset = initial_reset_diagnostics(replay)
    seed = _int_or_none(replay.get("seed"))
    policy = _str_or_none(replay.get("policy"))

    root_origin = _str_or_none(replay.get("root_origin"))
    root_replay = _str_or_none(replay.get("root_replay"))
    root_seed = _int_or_none(replay.get("root_seed"))
    root_frame_index = _int_or_none(replay.get("root_frame_index"))
    root_move_count = _int_or_none(replay.get("root_move_count"))
    root_score = _int_or_none(replay.get("root_score"))
    root_policy = _str_or_none(replay.get("root_policy"))

    if root_origin is None and origin in GENUINE_ROOT_ORIGINS and bool(reset.get("is_reset_start")):
        root_origin = origin
        root_replay = str(replay_path) if replay_path is not None else None
        root_seed = seed
        root_frame_index = 0
        root_move_count = 0
        root_score = _int_or_none(reset.get("score"))
        root_policy = policy

    root_origin = root_origin or ORIGIN_UNKNOWN
    source_replay = _str_or_none(replay.get("source_replay"))
    source_seed = _int_or_none(replay.get("source_seed"))
    source_frame_index = _int_or_none(replay.get("source_frame_index"))
    source_policy = _str_or_none(replay.get("source_policy")) or policy
    source_origin = _str_or_none(replay.get("source_origin")) or ORIGIN_UNKNOWN

    if root_replay is None and root_origin in GENUINE_ROOT_ORIGINS and replay_path is not None and origin in GENUINE_ROOT_ORIGINS:
        root_replay = str(replay_path)
    ancestry_key = (
        f"root:{root_origin}:{root_replay}:{root_seed}:{root_frame_index}"
        if root_replay is not None
        else f"source:{source_origin}:{source_replay}:{source_seed}:{source_frame_index}"
        if source_replay is not None
        else f"replay:{origin}:{replay_path}:{seed}"
    )
    return {
        "replay_origin": origin,
        "replay_reset_invariant": bool(reset.get("is_reset_start")),
        "replay_reset_reason": str(reset.get("reason")),
        "source_origin": source_origin,
        "source_replay": source_replay,
        "source_seed": source_seed,
        "source_frame_index": source_frame_index,
        "source_policy": source_policy,
        "source_policy_family": policy_family(source_policy),
        "root_origin": root_origin,
        "root_replay": root_replay,
        "root_seed": root_seed,
        "root_frame_index": root_frame_index,
        "root_move_count": root_move_count,
        "root_score": root_score,
        "root_policy": root_policy,
        "root_policy_family": policy_family(root_policy),
        "root_is_genuine": root_origin in GENUINE_ROOT_ORIGINS,
        "ancestry_key": ancestry_key,
    }


def provenance_fields_from_record(record: dict[str, Any], fallback_replay: Path | str | None = None) -> dict[str, Any]:
    """Normalize provenance fields stored on a state-record artifact."""

    root_origin = _str_or_none(record.get("root_origin")) or ORIGIN_UNKNOWN
    source_policy = _str_or_none(record.get("source_policy"))
    root_policy = _str_or_none(record.get("root_policy"))
    source_replay = _str_or_none(record.get("source_replay"))
    if source_replay is None and fallback_replay is not None:
        source_replay = str(fallback_replay)
    return {
        "source_origin": _str_or_none(record.get("source_origin")) or _str_or_none(record.get("replay_origin")) or ORIGIN_UNKNOWN,
        "source_replay": source_replay,
        "source_seed": _int_or_none(record.get("source_seed", record.get("seed"))),
        "source_frame_index": _int_or_none(record.get("source_frame_index", record.get("frame_index"))),
        "source_policy": source_policy,
        "source_policy_family": policy_family(source_policy),
        "root_origin": root_origin,
        "root_replay": _str_or_none(record.get("root_replay")),
        "root_seed": _int_or_none(record.get("root_seed")),
        "root_frame_index": _int_or_none(record.get("root_frame_index")),
        "root_move_count": _int_or_none(record.get("root_move_count")),
        "root_score": _int_or_none(record.get("root_score")),
        "root_policy": root_policy,
        "root_policy_family": policy_family(root_policy),
        "root_is_genuine": root_origin in GENUINE_ROOT_ORIGINS,
        "ancestry_key": _str_or_none(record.get("ancestry_key"))
        or f"root:{root_origin}:{record.get('root_replay')}:{record.get('root_seed')}:{record.get('root_frame_index')}",
    }

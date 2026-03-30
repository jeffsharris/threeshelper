import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import mirroring_control as mc
import window_stream as ws


KNOWN_PREVIEW_LABELS = ("red", "blue", "gray", "large_candidates")


@dataclass
class TransitionValidation:
    valid: bool
    reason: str
    eligible_positions: List[Tuple[int, int]]
    expected_values: List[int]
    inserted_value: Optional[int] = None
    inserted_pos: Optional[Tuple[int, int]] = None
    best_mismatch: Optional[Dict[str, object]] = None


@dataclass
class TransitionStep:
    direction: str
    preview_label: str
    inserted_value: int
    inserted_pos: Tuple[int, int]
    eligible_positions: List[Tuple[int, int]]
    expected_values: List[int]
    after_board: List[List[str]]


@dataclass
class TransitionPath:
    steps: List[TransitionStep]
    preview_check: Dict[str, object]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def can_merge(a: int, b: int) -> bool:
    if a <= 0 or b <= 0:
        return False
    if {a, b} == {1, 2}:
        return True
    return a >= 3 and a == b


def merge_value(a: int, b: int) -> int:
    if not can_merge(a, b):
        raise ValueError(f"Cannot merge {a} and {b}")
    if {a, b} == {1, 2}:
        return 3
    return a * 2


def advance_line_toward_start(line: Sequence[int]) -> Tuple[List[int], bool]:
    cells = list(line)
    moved_into = [False] * len(cells)
    merged_into = [False] * len(cells)
    original = list(cells)
    for idx in range(1, len(cells)):
        val = cells[idx]
        if val == 0:
            continue
        if cells[idx - 1] == 0:
            cells[idx - 1] = val
            cells[idx] = 0
            moved_into[idx - 1] = True
            merged_into[idx - 1] = False
        elif can_merge(cells[idx - 1], val) and not moved_into[idx - 1] and not merged_into[idx - 1]:
            cells[idx - 1] = merge_value(cells[idx - 1], val)
            cells[idx] = 0
            moved_into[idx - 1] = False
            merged_into[idx - 1] = True
    return cells, cells != original


def transpose(board: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*board)]


def simulate_base_move(board: Sequence[Sequence[int]], direction: str) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    if direction not in mc.DIRECTIONS:
        raise ValueError(f"Unsupported direction: {direction}")

    board_list = [list(row) for row in board]
    if direction in ("up", "down"):
        working = transpose(board_list)
    else:
        working = [list(row) for row in board_list]

    moved_rows: List[bool] = []
    result_rows: List[List[int]] = []
    for row in working:
        if direction in ("right", "down"):
            shifted, changed = advance_line_toward_start(list(reversed(row)))
            shifted = list(reversed(shifted))
        else:
            shifted, changed = advance_line_toward_start(row)
        result_rows.append(shifted)
        moved_rows.append(changed)

    if direction in ("up", "down"):
        board_after = transpose(result_rows)
    else:
        board_after = result_rows

    eligible_positions: List[Tuple[int, int]] = []
    for idx, changed in enumerate(moved_rows):
        if not changed:
            continue
        if direction == "left" and board_after[idx][3] == 0:
            eligible_positions.append((idx, 3))
        elif direction == "right" and board_after[idx][0] == 0:
            eligible_positions.append((idx, 0))
        elif direction == "up" and board_after[3][idx] == 0:
            eligible_positions.append((3, idx))
        elif direction == "down" and board_after[0][idx] == 0:
            eligible_positions.append((0, idx))
    return board_after, eligible_positions


def board_to_values(board: Sequence[Sequence[str]]) -> Optional[List[List[int]]]:
    values: List[List[int]] = []
    for row in board:
        value_row: List[int] = []
        for token in row:
            if token == ws.TOKEN_OTHER or token == "?":
                return None
            value_row.append(ws.token_to_value(token))
        values.append(value_row)
    return values


def values_to_board(values: Sequence[Sequence[int]]) -> List[List[str]]:
    board: List[List[str]] = []
    for row in values:
        token_row: List[str] = []
        for value in row:
            if value <= 0:
                token_row.append(ws.TOKEN_EMPTY)
            elif value == 1:
                token_row.append(ws.SMALL_COLOR_MAP["red"])
            elif value == 2:
                token_row.append(ws.SMALL_COLOR_MAP["blue"])
            else:
                token_row.append(str(value))
        board.append(token_row)
    return board


def bonus_values_for_max(max_tile: int) -> List[int]:
    cycle = ws.TileCycle()
    cycle.set_max_tile(max_tile)
    return cycle.bonus_values()


def expected_insert_values(preview_label: str, max_tile_before: int) -> List[int]:
    if preview_label == "red":
        return [1]
    if preview_label == "blue":
        return [2]
    if preview_label == "gray":
        return [3]
    if preview_label == "large_candidates":
        return bonus_values_for_max(max_tile_before)
    return []


def diff_positions(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    diffs: List[Tuple[int, int]] = []
    for r in range(4):
        for c in range(4):
            if a[r][c] != b[r][c]:
                diffs.append((r, c))
    return diffs


def validate_transition(
    before_board: Sequence[Sequence[str]],
    direction: str,
    expected_preview: str,
    after_board: Sequence[Sequence[str]],
) -> TransitionValidation:
    before_values = board_to_values(before_board)
    after_values = board_to_values(after_board)
    if before_values is None or after_values is None:
        return TransitionValidation(
            valid=False,
            reason="board contains unknown cells",
            eligible_positions=[],
            expected_values=[],
        )

    max_tile_before = ws.board_max_tile(before_board)
    expected_values = expected_insert_values(expected_preview, max_tile_before)
    if not expected_values:
        return TransitionValidation(
            valid=False,
            reason=f"cannot infer inserted tile from preview {expected_preview!r}",
            eligible_positions=[],
            expected_values=[],
        )

    afterstate, eligible_positions = simulate_base_move(before_values, direction)
    if not eligible_positions:
        return TransitionValidation(
            valid=False,
            reason="move produced no legal insertion positions",
            eligible_positions=[],
            expected_values=expected_values,
        )

    best_mismatch: Optional[Dict[str, object]] = None
    for pos in eligible_positions:
        for value in expected_values:
            candidate = [row[:] for row in afterstate]
            candidate[pos[0]][pos[1]] = value
            if candidate == after_values:
                return TransitionValidation(
                    valid=True,
                    reason="legal move",
                    eligible_positions=eligible_positions,
                    expected_values=expected_values,
                    inserted_value=value,
                    inserted_pos=pos,
                )
            mismatch = diff_positions(candidate, after_values)
            info = {
                "inserted_value": value,
                "inserted_pos": list(pos),
                "mismatch_count": len(mismatch),
                "mismatch_positions": [list(p) for p in mismatch],
            }
            if best_mismatch is None or info["mismatch_count"] < best_mismatch["mismatch_count"]:
                best_mismatch = info

    return TransitionValidation(
        valid=False,
        reason="board does not match any legal result",
        eligible_positions=eligible_positions,
        expected_values=expected_values,
        best_mismatch=best_mismatch,
    )


def preview_check_from_snapshot(
    snapshot: Optional[Tuple[Dict[str, int], int, int, int, bool, int]],
    after_board: Optional[Sequence[Sequence[str]]],
    after_preview: Optional[str],
) -> Dict[str, object]:
    if snapshot is None or after_board is None or after_preview is None:
        return {"valid": True, "reason": "tracking disabled"}
    cycle = ws.TileCycle()
    cycle.restore(snapshot)
    cycle.set_max_tile(ws.board_max_tile(after_board))
    probs = cycle.probabilities()
    ok, reason = ws.preview_possible(cycle, after_preview)
    out = {
        "valid": ok,
        "reason": reason,
        "probabilities": probs,
        "bonus_values": cycle.bonus_values(),
        "max_tile": cycle.max_tile,
    }
    if ok:
        cycle.update(after_preview)
        out["next_snapshot"] = cycle.snapshot()
    return out


def possible_preview_labels(
    snapshot: Optional[Tuple[Dict[str, int], int, int, int, bool, int]],
    board: Sequence[Sequence[str]],
) -> List[str]:
    if snapshot is None:
        return list(KNOWN_PREVIEW_LABELS)
    cycle = ws.TileCycle()
    cycle.restore(snapshot)
    cycle.set_max_tile(ws.board_max_tile(board))
    probs = cycle.probabilities()
    labels: List[str] = []
    for label in KNOWN_PREVIEW_LABELS:
        if probs.get(label, 0.0) > 0:
            labels.append(label)
    return labels


def advance_snapshot_with_preview(
    snapshot: Optional[Tuple[Dict[str, int], int, int, int, bool, int]],
    board: Sequence[Sequence[str]],
    preview_label: str,
) -> Optional[Tuple[Dict[str, int], int, int, int, bool, int]]:
    if snapshot is None:
        return None
    check = preview_check_from_snapshot(snapshot, board, preview_label)
    if not check.get("valid", False):
        return None
    next_snapshot = check.get("next_snapshot")
    if isinstance(next_snapshot, tuple) or next_snapshot is None:
        return next_snapshot
    return tuple(next_snapshot)


def generate_transition_steps(
    before_board: Sequence[Sequence[str]],
    preview_label: str,
) -> List[TransitionStep]:
    before_values = board_to_values(before_board)
    if before_values is None:
        return []
    max_tile_before = ws.board_max_tile(before_board)
    expected_values = expected_insert_values(preview_label, max_tile_before)
    if not expected_values:
        return []

    steps: List[TransitionStep] = []
    for direction in mc.DIRECTIONS:
        afterstate, eligible_positions = simulate_base_move(before_values, direction)
        if not eligible_positions:
            continue
        for pos in eligible_positions:
            for value in expected_values:
                candidate = [row[:] for row in afterstate]
                candidate[pos[0]][pos[1]] = value
                steps.append(
                    TransitionStep(
                        direction=direction,
                        preview_label=preview_label,
                        inserted_value=value,
                        inserted_pos=pos,
                        eligible_positions=list(eligible_positions),
                        expected_values=list(expected_values),
                        after_board=values_to_board(candidate),
                    )
                )
    return steps


def serialize_transition_step(step: TransitionStep) -> Dict[str, object]:
    return {
        "direction": step.direction,
        "preview_label": step.preview_label,
        "inserted_value": step.inserted_value,
        "inserted_pos": list(step.inserted_pos),
        "eligible_positions": [list(pos) for pos in step.eligible_positions],
        "expected_values": step.expected_values,
        "after_board": step.after_board,
    }


def find_transition_paths(
    before_board: Sequence[Sequence[str]],
    before_preview: str,
    before_snapshot: Optional[Tuple[Dict[str, int], int, int, int, bool, int]],
    after_board: Sequence[Sequence[str]],
    after_preview: str,
    max_depth: int = 2,
    max_paths: int = 8,
) -> List[TransitionPath]:
    matches: List[TransitionPath] = []
    target_board = [list(row) for row in after_board]
    seen: set[Tuple[Tuple[Tuple[str, ...], ...], str, Optional[Tuple[Tuple[Tuple[str, int], ...], int, int, int, bool, int]], int]] = set()

    def freeze_board(board: Sequence[Sequence[str]]) -> Tuple[Tuple[str, ...], ...]:
        return tuple(tuple(row) for row in board)

    def freeze_snapshot(
        snapshot: Optional[Tuple[Dict[str, int], int, int, int, bool, int]],
    ) -> Optional[Tuple[Tuple[Tuple[str, int], ...], int, int, int, bool, int]]:
        if snapshot is None:
            return None
        counts, deck_index, dealt, max_tile, bonus_active, bonus_index = snapshot
        return (
            tuple(sorted((str(k), int(v)) for k, v in counts.items())),
            int(deck_index),
            int(dealt),
            int(max_tile),
            bool(bonus_active),
            int(bonus_index),
        )

    def dfs(
        current_board: Sequence[Sequence[str]],
        current_preview: str,
        current_snapshot: Optional[Tuple[Dict[str, int], int, int, int, bool, int]],
        depth: int,
        steps: List[TransitionStep],
    ) -> None:
        if len(matches) >= max_paths:
            return
        state_key = (freeze_board(current_board), current_preview, freeze_snapshot(current_snapshot), depth)
        if state_key in seen:
            return
        seen.add(state_key)

        for step in generate_transition_steps(current_board, current_preview):
            if step.after_board == target_board:
                preview_check = preview_check_from_snapshot(current_snapshot, step.after_board, after_preview)
                if preview_check.get("valid", False):
                    matches.append(TransitionPath(steps=steps + [step], preview_check=preview_check))
                    if len(matches) >= max_paths:
                        return
            if depth <= 1:
                continue
            for next_preview in possible_preview_labels(current_snapshot, step.after_board):
                next_snapshot = advance_snapshot_with_preview(current_snapshot, step.after_board, next_preview)
                if current_snapshot is not None and next_snapshot is None:
                    continue
                dfs(step.after_board, next_preview, next_snapshot, depth - 1, steps + [step])
                if len(matches) >= max_paths:
                    return

    dfs(before_board, before_preview, before_snapshot, max_depth, [])
    return matches


def ordered_directions(order_mode: str, rng, rotation: int) -> Tuple[List[str], int]:
    directions = list(mc.DIRECTIONS)
    if order_mode == "random":
        rng.shuffle(directions)
        return directions, rotation
    if order_mode == "cycle":
        directions = directions[rotation:] + directions[:rotation]
        return directions, (rotation + 1) % len(directions)
    return directions, rotation


def valid_directions_for_transition(
    before_board: Sequence[Sequence[str]],
    before_preview: str,
    after_board: Sequence[Sequence[str]],
) -> List[Tuple[str, TransitionValidation]]:
    matches: List[Tuple[str, TransitionValidation]] = []
    for direction in mc.DIRECTIONS:
        transition = validate_transition(before_board, direction, before_preview, after_board)
        if transition.valid:
            matches.append((direction, transition))
    return matches


class HarnessRecorder:
    def __init__(self, base_dir: Path, window_info: Dict[str, object]) -> None:
        self.dataset = ws.DatasetRecorder(base_dir, window_info=window_info)
        self.events_path = self.dataset.session_dir / "events.jsonl"
        self.failures_path = self.dataset.session_dir / "failures.jsonl"
        self.status_path = self.dataset.session_dir / "live_status.json"
        self.scene_dir = self.dataset.session_dir / "scenes"
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.scene_idx = 0

    @property
    def session_dir(self) -> Path:
        return self.dataset.session_dir

    def record_game_state(self, frame: mc.FrameState, window_id: int, ts_event: float) -> int:
        return self.dataset.record_capture(
            frame.arr,
            frame.board,
            frame.preview_label,
            frame.preview_debug,
            window_id,
            ts_event,
        )

    def record_scene(self, screen_state: mc.ScreenState, label: str, extra: Optional[Dict[str, object]] = None) -> str:
        self.scene_idx += 1
        stem = f"{self.scene_idx:04d}_{label}"
        img_path = self.scene_dir / f"{stem}.png"
        meta_path = self.scene_dir / f"{stem}.json"
        Image.fromarray(screen_state.arr).save(img_path)
        meta = {
            "scene": screen_state.scene,
            "scene_score": screen_state.scene_score,
            "scene_scores": screen_state.scene_scores,
        }
        if extra:
            meta.update(extra)
        meta_path.write_text(json.dumps(_json_safe(meta), indent=2))
        return img_path.name

    def append_event(self, event: Dict[str, object]) -> None:
        with self.events_path.open("a") as f:
            f.write(json.dumps(_json_safe(event)) + "\n")

    def append_failure(self, failure: Dict[str, object]) -> None:
        with self.failures_path.open("a") as f:
            f.write(json.dumps(_json_safe(failure)) + "\n")

    def write_status(self, status: Dict[str, object]) -> None:
        self.status_path.write_text(json.dumps(_json_safe(status), indent=2))
